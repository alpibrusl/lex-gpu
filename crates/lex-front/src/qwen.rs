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

use lex_ir::{DType, Space};

use crate::ir::{Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, TileTy, Ty, UnOp, View};

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
    /// Where `v` starts in its buffer, and how wide that buffer is. The
    /// values come out of the convolution's output row, after the queries
    /// and the keys, so the step reads them where they lie.
    pub v_base: usize,
    pub v_width: usize,
}

impl DeltaNet {
    /// A layer whose `v` has a buffer to itself.
    pub fn packed(
        v_heads: usize,
        k_heads: usize,
        k_dim: usize,
        v_dim: usize,
        rows: usize,
    ) -> DeltaNet {
        DeltaNet {
            v_heads,
            k_heads,
            k_dim,
            v_dim,
            rows,
            v_base: 0,
            v_width: v_heads * v_dim,
        }
    }
}

impl DeltaNet {
    /// One decode step: `S` is read and written once, `y` is `S q`.
    pub fn build_step(&self) -> Result<Program, String> {
        self.build_steps(1)
    }

    /// `tokens` steps of the delta rule in one pass over the state.
    ///
    /// The recurrence is sequential — token `t + 1` sees the state token
    /// `t` left — so a batch cannot be parallel over tokens. What it can do
    /// is pay for the state once: a tile of rows is loaded, carried through
    /// the whole batch in registers, and stored at the end. At 3 MB a layer
    /// that is the difference between verifying a speculative batch and
    /// running it token by token.
    ///
    /// Parameters: `state [v_heads * v_dim, k_dim]`, `q` and `k`
    /// `[tokens * v_heads, k_dim]` (one row per value head, the key head's
    /// row repeated), `v [tokens * v_width]` (the convolution's output
    /// rows, values at `v_base`), `g`, `beta` and `y`, each
    /// `[tokens * v_heads * v_dim]`.
    pub fn build_steps(&self, tokens: usize) -> Result<Program, String> {
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
        if c.v_base + c.v_heads * c.v_dim > c.v_width {
            return Err(format!(
                "v at {} of a {}-wide buffer does not hold {} x {}",
                c.v_base, c.v_width, c.v_heads, c.v_dim
            ));
        }
        if tokens == 0 {
            return Err("a batch is at least one token".into());
        }
        let (hv, dk, dv, rows) = (c.v_heads, c.k_dim, c.v_dim, c.rows);
        let gates = hv * dv;
        let mut b = Builder::new(&format!(
            "delta_step{tokens}_h{hv}k{}_d{dk}x{dv}_r{rows}",
            c.k_heads
        ));
        let ps = b.param("state", DType::F32, &[hv * dv, dk], true);
        let pq = b.param("q", DType::F32, &[tokens * hv, dk], false);
        let pk = b.param("k", DType::F32, &[tokens * hv, dk], false);
        let pv = b.param("v", DType::F32, &[tokens * c.v_width], false);
        let pg = b.param("g", DType::F32, &[tokens * gates], false);
        let pb = b.param("beta", DType::F32, &[tokens * gates], false);
        let py = b.param("y", DType::F32, &[tokens * gates], true);
        let chunk = b.grid(dv / rows);
        let head = b.grid2(hv);

        let tile = reg(&[rows, dk]);
        let vecr = reg(&[rows]);
        let vecc = reg(&[1, dk]);
        // This instance's rows, within one token's worth of gates.
        let first = IdxExpr::scaled(head, dv, 0).plus(chunk, rows);
        let state_at = View {
            param: ps,
            offset: vec![first.clone(), IdxExpr::lit(0)],
            shape: vec![rows, dk],
        };
        let s0 = b.op("s", Op::Load(state_at.clone(), tile.clone()));
        let out = b.for_range(
            0,
            tokens,
            vec![s0],
            vec![Ty::Tile(tile.clone())],
            |b, tok, p| {
                // Token `tok`'s slice of every per-token parameter.
                let kv_row = |param| View {
                    param,
                    offset: vec![IdxExpr::scaled(tok, hv, 0).plus(head, 1), IdxExpr::lit(0)],
                    shape: vec![1, dk],
                };
                let gate = |param| View {
                    param,
                    offset: vec![first.clone().plus(tok, gates)],
                    shape: vec![rows],
                };
                let s = p[0];
                let g = b.op("g", Op::Load(gate(pg), vecr.clone()));
                let s = b.op("s", Op::Binary(BinOp::Mul, Move(s), Move(g)));

                let k = b.op("k", Op::Load(kv_row(pk), vecc.clone()));
                let sk = b.op("sk", Op::Binary(BinOp::Mul, Borrow(s), Borrow(k)));
                let sk = b.op("sk", Op::RowReduce(Reduce::Sum, Move(sk)));

                let v = b.op(
                    "v",
                    Op::Load(
                        View {
                            param: pv,
                            offset: vec![first.clone().shift(c.v_base).plus(tok, c.v_width)],
                            shape: vec![rows],
                        },
                        vecr.clone(),
                    ),
                );
                let d = b.op("d", Op::Binary(BinOp::Sub, Move(v), Move(sk)));
                let beta = b.op("beta", Op::Load(gate(pb), vecr.clone()));
                let d = b.op("d", Op::Binary(BinOp::Mul, Move(d), Move(beta)));
                let ones = b.op("ones", Op::Fill(tile.clone(), 1.0));
                let kb = b.op("kb", Op::Binary(BinOp::Mul, Move(ones), Move(k)));
                let upd = b.op("upd", Op::Binary(BinOp::Mul, Move(kb), Move(d)));
                let s = b.op("s", Op::Binary(BinOp::Add, Move(s), Move(upd)));

                let q = b.op("q", Op::Load(kv_row(pq), vecc.clone()));
                let y = b.op("y", Op::Binary(BinOp::Mul, Borrow(s), Move(q)));
                let y = b.op("y", Op::RowReduce(Reduce::Sum, Move(y)));
                b.effect(Op::Store(Move(y), gate(py)));
                vec![s]
            },
        );
        b.effect(Op::Store(Move(out[0]), state_at));
        Ok(b.finish())
    }
}

/// The two gates of a linear-attention layer, expanded to one value per
/// state row so [`DeltaNet::build_step`] can scale a tile of rows by them.
///
/// `g = exp(-A softplus(a + dt_bias))` decays the state, `beta =
/// sigmoid(b)` weights the write. `A` is `exp(A_log)`, folded by the
/// loader: it is a weight, so the exponential is free.
///
/// Parameters: `a`, `b`, `A` and `dt_bias`, each `[v_heads]`, then
/// `g` and `beta`, each `[v_heads, v_dim]`. One instance per head.
pub fn build_gates(v_heads: usize, v_dim: usize) -> Program {
    build_gates_rows(1, v_heads, v_dim)
}

/// [`build_gates`] for a batch: `a` and `b` are `[tokens * v_heads]`, the
/// outputs `[tokens * v_heads, v_dim]`.
pub fn build_gates_rows(tokens: usize, v_heads: usize, v_dim: usize) -> Program {
    use Arg::Move;
    let mut b = Builder::new(&format!("delta_gates{tokens}_h{v_heads}x{v_dim}"));
    let pa = b.param("a", DType::F32, &[tokens * v_heads], false);
    let pb = b.param("b", DType::F32, &[tokens * v_heads], false);
    let p_amp = b.param("A", DType::F32, &[v_heads], false);
    let pdt = b.param("dt_bias", DType::F32, &[v_heads], false);
    let pg = b.param("g", DType::F32, &[tokens * v_heads, v_dim], true);
    let pbeta = b.param("beta", DType::F32, &[tokens * v_heads, v_dim], true);
    let head = b.grid(v_heads);

    let one = reg(&[1]);
    let row = TileTy::new(DType::F32, &[1, v_dim], Space::Reg);
    b.for_range(0, tokens, vec![], vec![], |b, tok, _| {
        // This token's `a` and `b`; `A` and the bias are weights.
        let per_token = |param| View {
            param,
            offset: vec![IdxExpr::scaled(tok, v_heads, 0).plus(head, 1)],
            shape: vec![1],
        };
        let weight = |param| View {
            param,
            offset: vec![IdxExpr::scaled(head, 1, 0)],
            shape: vec![1],
        };
        let a = b.op("a", Op::Load(per_token(pa), one.clone()));
        let dt = b.op("dt", Op::Load(weight(pdt), one.clone()));
        let u = b.op("u", Op::Binary(BinOp::Add, Move(a), Move(dt)));
        let sp = b.op("sp", Op::Unary(UnOp::Softplus, Move(u)));
        let a_coef = b.op("A", Op::Load(weight(p_amp), one.clone()));
        let e = b.op("e", Op::Binary(BinOp::Mul, Move(sp), Move(a_coef)));
        let e = b.op("e", Op::Scale(Move(e), -1.0));
        let g = b.op("g", Op::Exp(Move(e)));
        let bb = b.op("b", Op::Load(per_token(pb), one.clone()));
        let beta = b.op("beta", Op::Unary(UnOp::Sigmoid, Move(bb)));

        // One value per state row: ones scaled by the head's gate.
        let spread = |b: &mut Builder, v, param| {
            let ones = b.op("ones", Op::Fill(row.clone(), 1.0));
            let out = b.op("out", Op::Binary(BinOp::Mul, Move(ones), Move(v)));
            b.effect(Op::Store(
                Move(out),
                View {
                    param,
                    offset: vec![
                        IdxExpr::scaled(tok, v_heads, 0).plus(head, 1),
                        IdxExpr::lit(0),
                    ],
                    shape: vec![1, v_dim],
                },
            ));
        };
        spread(b, g, pg);
        spread(b, beta, pbeta);
        vec![]
    });
    b.finish()
}

/// The depthwise convolution that precedes the delta rule: four taps over
/// the last four positions of every channel, then SiLU.
///
/// Decoding one token, the three previous positions are the layer's
/// `conv` state; this kernel consumes them with the new row, writes the
/// output, and leaves the state holding the last three rows again.
///
/// Parameters: `state [kernel - 1, channels]` (read and written), `x
/// [1, channels]`, `w [kernel, channels]` — the file's `[channels, 1,
/// kernel]` transposed by the loader, so a tap is one contiguous row —
/// and `y [1, channels]`.
pub fn build_conv_silu(channels: usize, kernel: usize, chunk: usize) -> Result<Program, String> {
    build_conv_silu_rows(1, channels, kernel, chunk)
}

/// [`build_conv_silu`] over a batch: `x` and `y` are `[tokens, channels]`
/// and the window slides through them, so token `t` sees the `kernel - 1`
/// positions before it wherever they came from -- the carried state for
/// the first of the batch, earlier tokens of the batch after that. The
/// state is left holding the batch's own last `kernel - 1` rows.
///
/// The token loop is unrolled in the builder rather than a `for_range`,
/// because which source a tap reads (the state or the batch) is decided
/// per token and an index expression cannot branch.
pub fn build_conv_silu_rows(
    tokens: usize,
    channels: usize,
    kernel: usize,
    chunk: usize,
) -> Result<Program, String> {
    use Arg::{Borrow, Move};
    if kernel < 2 || chunk == 0 || !channels.is_multiple_of(chunk) {
        return Err(format!(
            "conv of {channels} channels in chunks of {chunk}, kernel {kernel}"
        ));
    }
    let mut b = Builder::new(&format!("conv_silu_c{channels}k{kernel}_x{chunk}"));
    let ps = b.param("state", DType::F32, &[kernel - 1, channels], true);
    let px = b.param("x", DType::F32, &[tokens, channels], false);
    let pw = b.param("w", DType::F32, &[kernel, channels], false);
    let py = b.param("y", DType::F32, &[tokens, channels], true);
    let col = b.grid(channels / chunk);

    let tile = TileTy::new(DType::F32, &[1, chunk], Space::Reg);
    let slice = |param, r: usize| View {
        param,
        offset: vec![IdxExpr::lit(r), IdxExpr::scaled(col, chunk, 0)],
        shape: vec![1, chunk],
    };
    // Position `p` of the window for token `t` is `t + p - (kernel - 1)`:
    // negative indices are the carried state, the rest are batch rows.
    let source = |p: isize| -> (usize, usize) {
        if p < 0 {
            (ps, (p + kernel as isize - 1) as usize)
        } else {
            (px, p as usize)
        }
    };
    for t in 0..tokens {
        let mut acc = None;
        for i in 0..kernel {
            let (param, r) = source(t as isize + i as isize - (kernel as isize - 1));
            let v = b.op("v", Op::Load(slice(param, r), tile.clone()));
            let wt = b.op("w", Op::Load(slice(pw, i), tile.clone()));
            let m = b.op("t", Op::Binary(BinOp::Mul, Move(v), Move(wt)));
            acc = Some(match acc {
                None => m,
                Some(a) => b.op("acc", Op::Binary(BinOp::Add, Move(a), Move(m))),
            });
        }
        let acc = acc.expect("kernel >= 2");
        let sg = b.op("sg", Op::Unary(UnOp::Sigmoid, Borrow(acc)));
        let y = b.op("y", Op::Binary(BinOp::Mul, Move(acc), Move(sg)));
        b.effect(Op::Store(Move(y), slice(py, t)));
    }

    // The window the batch leaves: its own last `kernel - 1` rows, which
    // for a short batch still includes some of the old state.
    for i in 0..kernel - 1 {
        let (param, r) = source(tokens as isize + i as isize - (kernel as isize - 1));
        let v = b.op("v", Op::Load(slice(param, r), tile.clone()));
        b.effect(Op::Store(Move(v), slice(ps, i)));
    }
    Ok(b.finish())
}

/// Which rows a view of the attention prologue's tensors spans.
#[derive(Clone, Copy)]
enum Rows {
    /// One row, shared by every token and head (the norm weight).
    Shared,
    /// One row per token (the RoPE tables).
    PerToken,
    /// One row per token, heads side by side within it (the projection).
    Packed,
    /// One row per (token, head) (the outputs).
    PerHead,
}

/// Qwen3.5's attention prologue: normalise each head, rotate the first
/// `rot` of its dimensions, and (for queries) take the sigmoid of the gate
/// the projection carries alongside them.
///
/// The rotation pairs `i` with `i + rot/2`, not adjacent elements as the
/// GGUF Llama weights do, and leaves dimensions past `rot` alone — a
/// quarter of a 256-wide head is rotated. `cos` and `sin` are `rot/2`
/// wide, the host's tables for this position.
///
/// Parameters: `x [heads, 2 * head_dim]` for queries (the gate is the
/// second half of each head) or `[heads, head_dim]` for keys, the norm
/// weight `[1, head_dim]` (with Qwen's `1 +` already folded in by the
/// loader), `cos` and `sin` `[1, rot/2]`, `y [heads, head_dim]` in `out`,
/// and for queries `gate [heads, head_dim]`.
pub fn build_qk_rope(
    heads: usize,
    head_dim: usize,
    rot: usize,
    gate: bool,
    out: DType,
    eps: f32,
) -> Result<Program, String> {
    build_qk_rope_rows(1, heads, head_dim, rot, gate, out, eps)
}

/// [`build_qk_rope`] over a batch. Each token sits at its own position, so
/// `cos` and `sin` are `[tokens, rot / 2]`; `x` is `[tokens, heads * …]`
/// and the outputs are `[tokens * heads, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn build_qk_rope_rows(
    tokens: usize,
    heads: usize,
    head_dim: usize,
    rot: usize,
    gate: bool,
    out: DType,
    eps: f32,
) -> Result<Program, String> {
    use Arg::{Borrow, Move};
    if rot == 0 || !rot.is_multiple_of(2) || rot > head_dim {
        return Err(format!(
            "{rot} rotated dimensions of a {head_dim}-wide head"
        ));
    }
    let half = rot / 2;
    let width = if gate { 2 * head_dim } else { head_dim };
    let mut b = Builder::new(&format!(
        "qk_rope{tokens}_h{heads}d{head_dim}r{rot}{}",
        if gate { "_gated" } else { "" }
    ));
    let px = b.param("x", DType::F32, &[tokens, heads * width], false);
    let pw = b.param("nw", DType::F32, &[1, head_dim], false);
    let pc = b.param("cos", DType::F32, &[tokens, half], false);
    let psin = b.param("sin", DType::F32, &[tokens, half], false);
    let py = b.param("y", out, &[tokens * heads, head_dim], true);
    let pg = gate.then(|| b.param("gate", DType::F32, &[tokens * heads, head_dim], true));
    let head = b.grid(heads);
    let token = b.grid2(tokens);

    let span = |param, col: usize, n: usize, rows: Rows| View {
        param,
        offset: match rows {
            Rows::Shared => vec![IdxExpr::lit(0), IdxExpr::lit(col)],
            Rows::PerToken => vec![IdxExpr::scaled(token, 1, 0), IdxExpr::lit(col)],
            Rows::Packed => vec![
                IdxExpr::scaled(token, 1, 0),
                IdxExpr::scaled(head, width, col),
            ],
            Rows::PerHead => vec![
                IdxExpr::scaled(token, heads, 0).plus(head, 1),
                IdxExpr::lit(col),
            ],
        },
        shape: vec![1, n],
    };
    let tile = |n| TileTy::new(DType::F32, &[1, n], Space::Reg);

    // The head's own scale: one pass over its `head_dim` values.
    let x = b.op(
        "x",
        Op::Load(span(px, 0, head_dim, Rows::Packed), tile(head_dim)),
    );
    let sq = b.op("sq", Op::Binary(BinOp::Mul, Borrow(x), Borrow(x)));
    b.drop(x);
    let ss = b.op("ss", Op::RowReduce(Reduce::Sum, Move(sq)));
    let ms = b.op("ms", Op::Scale(Move(ss), 1.0 / head_dim as f32));
    let e = b.op(
        "eps",
        Op::Fill(TileTy::new(DType::F32, &[1], Space::Reg), eps),
    );
    let t = b.op("t", Op::Binary(BinOp::Add, Move(ms), Move(e)));
    let r = b.op("r", Op::Unary(UnOp::Rsqrt, Move(t)));

    // Column ranges of the same row, each normalised by that scale. A tile
    // cannot be sliced, so the halves are read from the parameter again.
    let normed = |b: &mut Builder, col: usize, n: usize| {
        let v = b.op("v", Op::Load(span(px, col, n, Rows::Packed), tile(n)));
        let w = b.op("w", Op::Load(span(pw, col, n, Rows::Shared), tile(n)));
        let v = b.op("v", Op::Binary(BinOp::Mul, Move(v), Move(w)));
        b.op("v", Op::Binary(BinOp::Mul, Move(v), Borrow(r)))
    };
    let lo = normed(&mut b, 0, half);
    let hi = normed(&mut b, half, half);
    let tail = (rot < head_dim).then(|| normed(&mut b, rot, head_dim - rot));

    // `x cos - y sin`, `y cos + x sin` over the rotated quarter.
    let c = b.op(
        "cos",
        Op::Load(span(pc, 0, half, Rows::PerToken), tile(half)),
    );
    let sn = b.op(
        "sin",
        Op::Load(span(psin, 0, half, Rows::PerToken), tile(half)),
    );
    let ac = b.op("ac", Op::Binary(BinOp::Mul, Borrow(lo), Borrow(c)));
    let bs = b.op("bs", Op::Binary(BinOp::Mul, Borrow(hi), Borrow(sn)));
    let o1 = b.op("o1", Op::Binary(BinOp::Sub, Move(ac), Move(bs)));
    let bc = b.op("bc", Op::Binary(BinOp::Mul, Move(hi), Move(c)));
    let as_ = b.op("as", Op::Binary(BinOp::Mul, Move(lo), Move(sn)));
    let o2 = b.op("o2", Op::Binary(BinOp::Add, Move(bc), Move(as_)));
    b.drop(r);

    let put = |b: &mut Builder, v, col: usize, n: usize| {
        let v = if out == DType::F32 {
            v
        } else {
            b.op("c", Op::Convert(Move(v), out))
        };
        b.effect(Op::Store(Move(v), span(py, col, n, Rows::PerHead)));
    };
    put(&mut b, o1, 0, half);
    put(&mut b, o2, half, half);
    if let Some(tail) = tail {
        put(&mut b, tail, rot, head_dim - rot);
    }
    if let Some(pg) = pg {
        let g = b.op(
            "g",
            Op::Load(span(px, head_dim, head_dim, Rows::Packed), tile(head_dim)),
        );
        let g = b.op("g", Op::Unary(UnOp::Sigmoid, Move(g)));
        b.effect(Op::Store(Move(g), span(pg, 0, head_dim, Rows::PerHead)));
    }
    Ok(b.finish())
}

/// A linear-attention layer's queries or keys, out of the convolution's
/// output row: normalised (no weight), scaled, and repeated once per value
/// head.
///
/// `q` is scaled by `1 / k_dim` and `k` by `1 / sqrt(k_dim)`, which is
/// where the attention scale lives in this architecture. The repetition is
/// what lets [`DeltaNet::build_step`] find its key with an affine index.
///
/// Parameters: `x [1, width]` — the convolution's output, with the heads
/// starting at `first` — and `y [k_heads * per, k_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn build_delta_qk(
    k_heads: usize,
    per: usize,
    k_dim: usize,
    width: usize,
    first: usize,
    scale: f32,
    eps: f32,
) -> Result<Program, String> {
    build_delta_qk_rows(1, k_heads, per, k_dim, width, first, scale, eps)
}

/// [`build_delta_qk`] over a batch: `x` is `[tokens, width]` and `y`
/// `[tokens * k_heads * per, k_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn build_delta_qk_rows(
    tokens: usize,
    k_heads: usize,
    per: usize,
    k_dim: usize,
    width: usize,
    first: usize,
    scale: f32,
    eps: f32,
) -> Result<Program, String> {
    use Arg::{Borrow, Move};
    if per == 0 || first + k_heads * k_dim > width {
        return Err(format!(
            "{k_heads} heads of {k_dim} at {first} do not fit a {width}-wide row"
        ));
    }
    let mut b = Builder::new(&format!(
        "delta_qk{tokens}_k{k_heads}x{per}_d{k_dim}_at{first}"
    ));
    let px = b.param("x", DType::F32, &[tokens, width], false);
    let py = b.param("y", DType::F32, &[tokens * k_heads * per, k_dim], true);
    // Instance `(copy, head)`: the head's row read once, written `per`
    // times. Both indices stay affine this way.
    let copy = b.grid(per);
    let head = b.grid2(k_heads);

    let tile = TileTy::new(DType::F32, &[1, k_dim], Space::Reg);
    b.for_range(0, tokens, vec![], vec![], |b, tok, _| {
        let x = b.op(
            "x",
            Op::Load(
                View {
                    param: px,
                    offset: vec![
                        IdxExpr::scaled(tok, 1, 0),
                        IdxExpr::scaled(head, k_dim, 0).shift(first),
                    ],
                    shape: vec![1, k_dim],
                },
                tile.clone(),
            ),
        );
        let sq = b.op("sq", Op::Binary(BinOp::Mul, Borrow(x), Borrow(x)));
        let ss = b.op("ss", Op::RowReduce(Reduce::Sum, Move(sq)));
        let ms = b.op("ms", Op::Scale(Move(ss), 1.0 / k_dim as f32));
        let e = b.op(
            "eps",
            Op::Fill(TileTy::new(DType::F32, &[1], Space::Reg), eps),
        );
        let t = b.op("t", Op::Binary(BinOp::Add, Move(ms), Move(e)));
        let r = b.op("r", Op::Unary(UnOp::Rsqrt, Move(t)));
        let y = b.op("y", Op::Binary(BinOp::Mul, Move(x), Move(r)));
        let y = b.op("y", Op::Scale(Move(y), scale));
        b.effect(Op::Store(
            Move(y),
            View {
                param: py,
                offset: vec![
                    IdxExpr::scaled(tok, k_heads * per, 0)
                        .plus(head, per)
                        .plus(copy, 1),
                    IdxExpr::lit(0),
                ],
                shape: vec![1, k_dim],
            },
        ));
        vec![]
    });
    Ok(b.finish())
}

/// `y = W x` for a small dense matrix: Qwen3.5's `in_proj_a` and
/// `in_proj_b` are 48 rows of bf16, too small to be worth quantising and
/// the only unquantised matrices in a layer.
pub fn build_matvec_dense(
    n_in: usize,
    n_out: usize,
    bo: usize,
    w: DType,
) -> Result<Program, String> {
    build_matvec_dense_rows(1, n_in, n_out, bo, w)
}

/// [`build_matvec_dense`] for `tokens` rows of activations at once.
pub fn build_matvec_dense_rows(
    tokens: usize,
    n_in: usize,
    n_out: usize,
    bo: usize,
    w: DType,
) -> Result<Program, String> {
    use Arg::Move;
    if bo == 0 || !n_out.is_multiple_of(bo) {
        return Err(format!("{n_out} rows do not split into {bo}"));
    }
    let mut b = Builder::new(&format!(
        "matvec_dense{tokens}_{n_out}x{n_in}_{}",
        w.suffix()
    ));
    let px = b.param("x", DType::F32, &[tokens, n_in], false);
    let pw = b.param("w", w, &[n_out, n_in], false);
    let py = b.param("y", DType::F32, &[tokens, n_out], true);
    let row = b.grid(n_out / bo);
    let x = b.op(
        "x",
        Op::Load(
            View {
                param: px,
                offset: vec![IdxExpr::lit(0), IdxExpr::lit(0)],
                shape: vec![tokens, n_in],
            },
            TileTy::new(DType::F32, &[tokens, n_in], Space::Reg),
        ),
    );
    let wt = b.op(
        "w",
        Op::Load(
            View {
                param: pw,
                offset: vec![IdxExpr::scaled(row, bo, 0), IdxExpr::lit(0)],
                shape: vec![bo, n_in],
            },
            TileTy::new(w, &[bo, n_in], Space::Reg),
        ),
    );
    let y = b.op("y", Op::MatMulNT(Move(x), Move(wt), DType::F32));
    b.effect(Op::Store(
        Move(y),
        View {
            param: py,
            offset: vec![IdxExpr::lit(0), IdxExpr::scaled(row, bo, 0)],
            shape: vec![tokens, bo],
        },
    ));
    Ok(b.finish())
}

/// The gated norm that ends a linear-attention layer: normalise each value
/// head's output with a weight, then multiply by `silu(z)`.
///
/// This norm is *not* one of the ones Qwen3.5 stores as a delta from 1:
/// its weight is used as it comes.
///
/// Parameters: `y [v_heads, v_dim]`, the norm weight `[1, v_dim]`,
/// `z [v_heads, v_dim]`, and the output `[v_heads, v_dim]`.
pub fn build_gated_norm(v_heads: usize, v_dim: usize, eps: f32) -> Program {
    build_gated_norm_rows(1, v_heads, v_dim, eps)
}

/// [`build_gated_norm`] over a batch: every tensor but the weight gains a
/// token dimension, `[tokens * v_heads, v_dim]`.
pub fn build_gated_norm_rows(tokens: usize, v_heads: usize, v_dim: usize, eps: f32) -> Program {
    use Arg::{Borrow, Move};
    let mut b = Builder::new(&format!("delta_gated_norm{tokens}_h{v_heads}x{v_dim}"));
    let py = b.param("y", DType::F32, &[tokens * v_heads, v_dim], false);
    let pw = b.param("w", DType::F32, &[1, v_dim], false);
    let pz = b.param("z", DType::F32, &[tokens * v_heads, v_dim], false);
    let po = b.param("o", DType::F32, &[tokens * v_heads, v_dim], true);
    let head = b.grid(v_heads);
    let token = b.grid2(tokens);
    let tile = TileTy::new(DType::F32, &[1, v_dim], Space::Reg);
    let rows = |param| View {
        param,
        offset: vec![
            IdxExpr::scaled(token, v_heads, 0).plus(head, 1),
            IdxExpr::lit(0),
        ],
        shape: vec![1, v_dim],
    };
    let y = b.op("y", Op::Load(rows(py), tile.clone()));
    let sq = b.op("sq", Op::Binary(BinOp::Mul, Borrow(y), Borrow(y)));
    let ss = b.op("ss", Op::RowReduce(Reduce::Sum, Move(sq)));
    let ms = b.op("ms", Op::Scale(Move(ss), 1.0 / v_dim as f32));
    let e = b.op(
        "eps",
        Op::Fill(TileTy::new(DType::F32, &[1], Space::Reg), eps),
    );
    let t = b.op("t", Op::Binary(BinOp::Add, Move(ms), Move(e)));
    let r = b.op("r", Op::Unary(UnOp::Rsqrt, Move(t)));
    let y = b.op("y", Op::Binary(BinOp::Mul, Move(y), Move(r)));
    // The norm weight is one row, shared by every head.
    let w = b.op(
        "w",
        Op::Load(
            View {
                param: pw,
                offset: vec![IdxExpr::lit(0), IdxExpr::lit(0)],
                shape: vec![1, v_dim],
            },
            tile.clone(),
        ),
    );
    let y = b.op("y", Op::Binary(BinOp::Mul, Move(y), Move(w)));
    let z = b.op("z", Op::Load(rows(pz), tile));
    let sz = b.op("sz", Op::Unary(UnOp::Sigmoid, Borrow(z)));
    let sz = b.op("sz", Op::Binary(BinOp::Mul, Move(z), Move(sz)));
    let o = b.op("o", Op::Binary(BinOp::Mul, Move(y), Move(sz)));
    b.effect(Op::Store(Move(o), rows(po)));
    b.finish()
}

/// Elementwise `y = a * b` over `n` values, `chunk` per instance: the
/// attention output meeting its gate before the output projection.
pub fn build_mul(n: usize, chunk: usize) -> Result<Program, String> {
    use Arg::Move;
    if chunk == 0 || !n.is_multiple_of(chunk) {
        return Err(format!("{n} values do not split into chunks of {chunk}"));
    }
    let mut b = Builder::new(&format!("mul_{n}x{chunk}"));
    let pa = b.param("a", DType::F32, &[1, n], false);
    let pb = b.param("b", DType::F32, &[1, n], false);
    let py = b.param("y", DType::F32, &[1, n], true);
    let i = b.grid(n / chunk);
    let span = |param| View {
        param,
        offset: vec![IdxExpr::lit(0), IdxExpr::scaled(i, chunk, 0)],
        shape: vec![1, chunk],
    };
    let t = TileTy::new(DType::F32, &[1, chunk], Space::Reg);
    let a = b.op("a", Op::Load(span(pa), t.clone()));
    let bb = b.op("b", Op::Load(span(pb), t));
    let y = b.op("y", Op::Binary(BinOp::Mul, Move(a), Move(bb)));
    b.effect(Op::Store(Move(y), span(py)));
    Ok(b.finish())
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
