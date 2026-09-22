//! Qwen3.5's linear-attention layers, as typed tile programs.
//!
//! Three quarters of the model's 64 layers are "gated delta" layers rather
//! than attention: instead of a growing KV cache they carry a fixed state
//! `S[head, v_dim, k_dim]` — 3 MB a layer, whatever the context length —
//! and update it once per token by the delta rule
//!
//! ```text
//! S = S * g                      decay, one scalar per value head
//! S = S + k (v - S k)ᵀ β         write the residual of what S already predicts
//! y = S q                        read
//! ```
//!
//! The whole layer is memory-bound on that state, and every row of it is
//! touched exactly once per token, so the kernel here does the decay, both
//! products and the write-back in a single pass: a row of `S` is loaded,
//! used four times and stored, never revisited.
//!
//! What surrounds it (the depthwise convolution over the last four
//! positions, the gates, the q/k norms) is ordinary elementwise work.

use tile_ir::{DType, Space};

use crate::ir::{Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, TileTy, View};

fn reg(shape: &[usize]) -> TileTy {
    TileTy::new(DType::F32, shape, Space::Reg)
}

/// Shape of one linear-attention layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaNet {
    /// Value heads (Qwen3.5-27B: 48).
    pub v_heads: usize,
    /// Key heads; each serves `v_heads / k_heads` value heads (16).
    pub k_heads: usize,
    /// Columns of the state, and the width of `q` and `k` (128).
    pub k_dim: usize,
    /// Rows of the state, and the width of `v` and the output (128).
    pub v_dim: usize,
    /// State rows per grid instance.
    pub rows: usize,
}

impl DeltaNet {
    /// One decode step: `S` is read and written once, `y` is `S q`.
    ///
    /// Parameters: `state [v_heads * v_dim, k_dim]`, `q` and `k`
    /// `[v_heads, k_dim]`, `v`, `g`, `beta` and `y`, each flat
    /// `[v_heads * v_dim]`.
    ///
    /// `q` and `k` arrive with one row per *value* head, the key head's row
    /// repeated `v_heads / k_heads` times, and the two gates with one value
    /// per state row. Both are the producer's job. Index expressions are
    /// affine, so a kernel cannot divide `pid` by the group size to find its
    /// key head, and repeating 6 KB of keys costs nothing next to the 3 MB
    /// of state this pass moves.
    ///
    /// Instance `(chunk, head)` covers `rows` state rows: the head's `q` and
    /// `k` are one row each, broadcast along the tile's columns, while `v`,
    /// `g` and `beta` vary down its rows.
    pub fn build_step(&self) -> Result<Program, String> {
        use Arg::{Borrow, Move};
        let c = *self;
        if c.v_heads == 0 || c.k_heads == 0 || !c.v_heads.is_multiple_of(c.k_heads) {
            return Err("value heads must be a whole multiple of key heads".into());
        }
        if c.rows == 0 || !c.v_dim.is_multiple_of(c.rows) {
            return Err(format!(
                "{} state rows do not split into {}",
                c.v_dim, c.rows
            ));
        }
        let (hv, dk, dv, rows) = (c.v_heads, c.k_dim, c.v_dim, c.rows);
        let mut b = Builder::new(&format!(
            "delta_step_h{hv}k{}_d{dk}x{dv}_r{rows}",
            c.k_heads
        ));
        let ps = b.param("state", DType::F32, &[hv * dv, dk], true);
        let pq = b.param("q", DType::F32, &[hv, dk], false);
        let pk = b.param("k", DType::F32, &[hv, dk], false);
        // Flat `[v_heads * v_dim]`: a tile of state rows takes a
        // contiguous slice of each, whatever head it falls in.
        let pv = b.param("v", DType::F32, &[hv * dv], false);
        let pg = b.param("g", DType::F32, &[hv * dv], false);
        let pb = b.param("beta", DType::F32, &[hv * dv], false);
        let py = b.param("y", DType::F32, &[hv * dv], true);
        let chunk = b.grid(dv / rows);
        let head = b.grid2(hv);

        let tile = reg(&[rows, dk]);
        let vecr = reg(&[rows]);
        let vecc = reg(&[1, dk]);
        // Row `head * dv + chunk * rows` of the state, and of the gates.
        let first = IdxExpr::scaled(head, dv, 0).plus(chunk, rows);
        let state_at = View {
            param: ps,
            offset: vec![first.clone(), IdxExpr::lit(0)],
            shape: vec![rows, dk],
        };
        let kv_row = |param| View {
            param,
            offset: vec![IdxExpr::scaled(head, 1, 0), IdxExpr::lit(0)],
            shape: vec![1, dk],
        };
        let gate = |param| View {
            param,
            offset: vec![first.clone()],
            shape: vec![rows],
        };

        let s = b.op("s", Op::Load(state_at.clone(), tile.clone()));
        let g = b.op("g", Op::Load(gate(pg), vecr.clone()));
        let s = b.op("s", Op::Binary(BinOp::Mul, Move(s), Move(g)));

        // `S k`: the key broadcast along the rows, summed over the state's
        // columns.
        let k = b.op("k", Op::Load(kv_row(pk), vecc.clone()));
        let sk = b.op("sk", Op::Binary(BinOp::Mul, Borrow(s), Borrow(k)));
        let sk = b.op("sk", Op::RowReduce(Reduce::Sum, Move(sk)));

        // `δ = (v - S k) β`, then the rank-one update `S += k δᵀ`.
        let v = b.op("v", Op::Load(gate(pv), vecr.clone()));
        let d = b.op("d", Op::Binary(BinOp::Sub, Move(v), Move(sk)));
        let beta = b.op("beta", Op::Load(gate(pb), vecr));
        let d = b.op("d", Op::Binary(BinOp::Mul, Move(d), Move(beta)));
        let ones = b.op("ones", Op::Fill(tile.clone(), 1.0));
        let kb = b.op("kb", Op::Binary(BinOp::Mul, Move(ones), Move(k)));
        let upd = b.op("upd", Op::Binary(BinOp::Mul, Move(kb), Move(d)));
        let s = b.op("s", Op::Binary(BinOp::Add, Move(s), Move(upd)));

        // `y = S q` with the updated state, which is what is stored.
        let q = b.op("q", Op::Load(kv_row(pq), vecc));
        let y = b.op("y", Op::Binary(BinOp::Mul, Borrow(s), Move(q)));
        let y = b.op("y", Op::RowReduce(Reduce::Sum, Move(y)));
        b.effect(Op::Store(
            Move(y),
            View {
                param: py,
                offset: vec![first],
                shape: vec![rows],
            },
        ));
        b.effect(Op::Store(Move(s), state_at));
        Ok(b.finish())
    }
}

/// The delta rule in f64, for the tests: `state` is `[v_heads * v_dim, k_dim]`
/// and is updated in place; returns `y [v_heads, v_dim]`. `q` and `k` have
/// one row per value head, as [`DeltaNet::build_step`] takes them.
pub fn reference(
    c: &DeltaNet,
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let (hv, dk, dv) = (c.v_heads, c.k_dim, c.v_dim);
    let mut y = vec![0.0f32; hv * dv];
    for h in 0..hv {
        // `q` and `k` carry one row per value head (see `build_step`).
        let kh = h;
        for r in 0..dv {
            let row = &mut state[(h * dv + r) * dk..(h * dv + r + 1) * dk];
            let (gr, br) = (g[h * dv + r] as f64, beta[h * dv + r] as f64);
            let mut sk = 0.0f64;
            for (j, s) in row.iter_mut().enumerate() {
                *s = (*s as f64 * gr) as f32;
                sk += *s as f64 * k[kh * dk + j] as f64;
            }
            let d = (v[h * dv + r] as f64 - sk) * br;
            let mut dot = 0.0f64;
            for (j, s) in row.iter_mut().enumerate() {
                *s = (*s as f64 + k[kh * dk + j] as f64 * d) as f32;
                dot += *s as f64 * q[kh * dk + j] as f64;
            }
            y[h * dv + r] = dot as f32;
        }
    }
    y
}
