//! Flash-attention decode: the P1 linearity stress test.
//!
//! One KV head is shared by `q_rows` query rows (GQA/MQA decode, one token per
//! query head), split into query blocks of `bq`. The loop order is the one that
//! makes linear types uncomfortable:
//!
//! ```text
//! for each KV block (outer, pipelined `stages` deep):
//!     K, V = wait(oldest in-flight copy)
//!     issue the copy for block i + stages - 1 into the free buffer
//!     for each query block (inner):          # K and V reused, borrowed
//!         online-softmax update of (m, l, acc)[qb]
//!     K, V become the free buffers
//! ```
//!
//! Each K/V tile is loaded once and read by every query block. That needs:
//! borrows that outlive one op (the whole inner loop), per-query-block state
//! that survives across outer iterations (a linear array walked by
//! `map_each`), and buffer rotation expressed as a loop carry
//! `(futures in flight, free buffer)` whose type does not change.
//!
//! The schedule — `bq`, `bk`, `stages`, where K/V live — is a parameter of the
//! builder, not part of the algorithm. The same function builds the Metal and
//! the Hopper variants; the checker decides whether each fits its target.

use tile_ir::{DType, Space};

use crate::ir::{Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, TileTy, Ty, Var, View};

/// Problem shape plus schedule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashDecode {
    /// Query rows sharing one KV head.
    pub q_rows: usize,
    /// Head dimension.
    pub d: usize,
    /// KV sequence length.
    pub seq: usize,
    /// Query rows per query block.
    pub bq: usize,
    /// KV rows per KV block.
    pub bk: usize,
    /// K/V buffers in the pipeline. 1 is no overlap; 2 is double-buffered.
    pub stages: usize,
    /// Storage dtype of Q, K and V in global memory and in their tiles.
    pub dtype: DType,
    /// Where K/V tiles are staged. Async copies need `Threadgroup`.
    pub kv_space: Space,
}

impl FlashDecode {
    pub fn n_qb(&self) -> usize {
        self.q_rows / self.bq
    }

    pub fn n_kb(&self) -> usize {
        self.seq / self.bk
    }

    pub fn kv_tile(&self) -> TileTy {
        TileTy::new(self.dtype, &[self.bk, self.d], self.kv_space)
    }

    /// Build the kernel. Shape errors that are about the schedule rather than
    /// the types (divisibility, pipeline deeper than the sequence) are
    /// reported here; everything else is the checker's job.
    pub fn build(&self) -> Result<Program, String> {
        let c = *self;
        if c.bq == 0 || c.bk == 0 || c.stages == 0 {
            return Err("bq, bk and stages must be nonzero".into());
        }
        if !c.q_rows.is_multiple_of(c.bq) || !c.seq.is_multiple_of(c.bk) {
            return Err(format!(
                "q_rows {} / bq {} and seq {} / bk {} must divide",
                c.q_rows, c.bq, c.seq, c.bk
            ));
        }
        if c.kv_space != Space::Threadgroup && c.stages != 1 {
            return Err("only threadgroup K/V can be pipelined: async copies land there".into());
        }
        if c.n_kb() < c.stages - 1 {
            return Err(format!(
                "{} KV blocks cannot fill a {}-stage pipeline",
                c.n_kb(),
                c.stages
            ));
        }

        let mut b = Builder::new(&format!(
            "flash_decode_{}_bq{}_bk{}_s{}",
            c.dtype.suffix(),
            c.bq,
            c.bk,
            c.stages
        ));
        let pq = b.param("q", c.dtype, &[c.q_rows, c.d], false);
        let pk = b.param("k", c.dtype, &[c.seq, c.d], false);
        let pv = b.param("v", c.dtype, &[c.seq, c.d], false);
        let po = b.param("o", DType::F32, &[c.q_rows, c.d], true);

        let reg = |dt, shape: &[usize]| TileTy::new(dt, shape, Space::Reg);
        let m_ty = reg(DType::F32, &[c.bq]);
        let acc_ty = reg(DType::F32, &[c.bq, c.d]);
        let kv = c.kv_tile();

        // Query blocks, loaded once and only ever borrowed.
        let q_blocks: Vec<Var> = (0..c.n_qb())
            .map(|qb| {
                let view = rows(pq, IdxExpr::lit(qb * c.bq), c.bq, c.d);
                b.op("q", Op::Load(view, reg(c.dtype, &[c.bq, c.d])))
            })
            .collect();
        let qs = b.op("qs", Op::MakeArray(q_blocks));

        // Online-softmax state, one entry per query block.
        let state = |b: &mut Builder, name: &str, ty: &TileTy, x: f32| {
            let v: Vec<Var> = (0..c.n_qb())
                .map(|_| b.op(name, Op::Fill(ty.clone(), x)))
                .collect();
            b.op(name, Op::MakeArray(v))
        };
        let ms = state(&mut b, "m", &m_ty, f32::NEG_INFINITY);
        let ls = state(&mut b, "l", &m_ty, 0.0);
        let accs = state(&mut b, "acc", &acc_ty, 0.0);

        if c.kv_space != Space::Threadgroup {
            // Synchronous loads straight into registers: nothing to rotate.
            let m_arr = Ty::Array(m_ty.clone(), c.n_qb());
            let carry = vec![m_arr.clone(), m_arr, Ty::Array(acc_ty.clone(), c.n_qb())];
            let out = b.for_range(0, c.n_kb(), vec![ms, ls, accs], carry, |b, i, p| {
                let at = IdxExpr::scaled(i, c.bk, 0);
                let k = b.op("k", Op::Load(rows(pk, at.clone(), c.bk, c.d), kv.clone()));
                let v = b.op("v", Op::Load(rows(pv, at, c.bk, c.d), kv.clone()));
                let out = attend(b, &c, qs, k, v, [p[0], p[1], p[2]]);
                b.drop(k);
                b.drop(v);
                out.to_vec()
            });
            finish(&mut b, &c, po, qs, [out[0], out[1], out[2]]);
            return Ok(b.finish());
        }

        // Prologue: `stages` buffers per operand; the first stages-1 are
        // issued for blocks 0..stages-1, the last is free.
        let prologue = |b: &mut Builder, p: usize, name: &str| {
            let mut inflight = vec![];
            for blk in 0..c.stages - 1 {
                let buf = b.op(name, Op::Alloc(kv.clone()));
                let view = rows(p, IdxExpr::lit(blk * c.bk), c.bk, c.d);
                inflight.push(b.op(name, Op::CopyAsync(view, buf)));
            }
            let free = b.op(name, Op::Alloc(kv.clone()));
            (inflight, free)
        };
        let (kf, kfree) = prologue(&mut b, pk, "kbuf");
        let (vf, vfree) = prologue(&mut b, pv, "vbuf");

        let mut init = vec![ms, ls, accs];
        init.extend(&kf);
        init.push(kfree);
        init.extend(&vf);
        init.push(vfree);

        let mut carry = vec![
            Ty::Array(m_ty.clone(), c.n_qb()),
            Ty::Array(m_ty.clone(), c.n_qb()),
            Ty::Array(acc_ty.clone(), c.n_qb()),
        ];
        let pipe: Vec<Ty> = (0..c.stages - 1)
            .map(|_| Ty::Future(kv.clone()))
            .chain([Ty::Tile(kv.clone())])
            .collect();
        carry.extend(pipe.iter().cloned());
        carry.extend(pipe.iter().cloned());

        let s = c.stages;
        let steady = c.n_kb() - (s - 1);
        let out = b.for_range(0, steady, init, carry, |b, i, p| {
            let (ms, ls, accs) = (p[0], p[1], p[2]);
            let (kq, kfree) = (&p[3..3 + s - 1], p[3 + s - 1]);
            let (vq, vfree) = (&p[3 + s..3 + 2 * s - 1], p[3 + 2 * s - 1]);

            // Issue block i + stages - 1 into the free buffer, then wait on
            // the oldest copy. With one stage this is copy-then-wait.
            let ahead = IdxExpr::scaled(i, c.bk, (s - 1) * c.bk);
            let knew = b.op(
                "kf",
                Op::CopyAsync(rows(pk, ahead.clone(), c.bk, c.d), kfree),
            );
            let vnew = b.op("vf", Op::CopyAsync(rows(pv, ahead, c.bk, c.d), vfree));
            let mut kq: Vec<Var> = kq.iter().copied().chain([knew]).collect();
            let mut vq: Vec<Var> = vq.iter().copied().chain([vnew]).collect();
            let kcur = b.op("k", Op::Wait(kq.remove(0)));
            let vcur = b.op("v", Op::Wait(vq.remove(0)));

            let [ms, ls, accs] = attend(b, &c, qs, kcur, vcur, [ms, ls, accs]);

            // The tiles just consumed become the free buffers.
            let mut y = vec![ms, ls, accs];
            y.extend(kq);
            y.push(kcur);
            y.extend(vq);
            y.push(vcur);
            y
        });

        // Epilogue: drain the stages-1 copies still in flight.
        let (mut ms, mut ls, mut accs) = (out[0], out[1], out[2]);
        let (kq, kfree) = (&out[3..3 + s - 1], out[3 + s - 1]);
        let (vq, vfree) = (&out[3 + s..3 + 2 * s - 1], out[3 + 2 * s - 1]);
        for (&kf, &vf) in kq.iter().zip(vq) {
            let kcur = b.op("k", Op::Wait(kf));
            let vcur = b.op("v", Op::Wait(vf));
            [ms, ls, accs] = attend(&mut b, &c, qs, kcur, vcur, [ms, ls, accs]);
            b.drop(kcur);
            b.drop(vcur);
        }
        b.drop(kfree);
        b.drop(vfree);
        finish(&mut b, &c, po, qs, [ms, ls, accs]);
        Ok(b.finish())
    }
}

/// `o[qb] = acc / l`, then release the query blocks.
fn finish(b: &mut Builder, c: &FlashDecode, po: usize, qs: Var, [ms, ls, accs]: [Var; 3]) {
    let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
    let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
    b.map_each(
        vec![ms, ls, accs],
        vec![
            Ty::Tile(m_ty.clone()),
            Ty::Tile(m_ty.clone()),
            Ty::Tile(acc_ty.clone()),
        ],
        0,
        |b, qb, p| {
            let o = b.op(
                "o",
                Op::Binary(BinOp::Div, Arg::Move(p[2]), Arg::Move(p[1])),
            );
            b.drop(p[0]);
            let view = rows(po, IdxExpr::scaled(qb, c.bq, 0), c.bq, c.d);
            b.effect(Op::Store(Arg::Move(o), view));
            vec![]
        },
    );
    b.drop(qs);
}

/// `rows` rows starting at `start`, all `d` columns.
fn rows(param: usize, start: IdxExpr, rows: usize, d: usize) -> View {
    View {
        param,
        offset: vec![start, IdxExpr::lit(0)],
        shape: vec![rows, d],
    }
}

/// One KV block against every query block: the online-softmax update.
/// `k` and `v` are borrowed by every iteration of the inner loop.
fn attend(b: &mut Builder, c: &FlashDecode, qs: Var, k: Var, v: Var, state: [Var; 3]) -> [Var; 3] {
    use Arg::{Borrow, Move};
    let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
    let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
    let elem = vec![Ty::Tile(m_ty.clone()), Ty::Tile(m_ty), Ty::Tile(acc_ty)];
    let scale = 1.0 / (c.d as f32).sqrt();
    let out = b.map_each(state.to_vec(), elem, 3, |b, qb, p| {
        let (m, l, acc) = (p[0], p[1], p[2]);
        let s = b.op(
            "s",
            Op::MatMulNT(Arg::BorrowElem(qs, qb), Borrow(k), DType::F32),
        );
        let s = b.op("s", Op::Scale(Move(s), scale));
        let mb = b.op("mb", Op::RowReduce(Reduce::Max, Borrow(s)));
        let m_new = b.op("m", Op::Binary(BinOp::Max, Borrow(m), Move(mb)));
        let d = b.op("dm", Op::Binary(BinOp::Sub, Move(m), Borrow(m_new)));
        let alpha = b.op("alpha", Op::Exp(Move(d)));
        let s = b.op("s", Op::Binary(BinOp::Sub, Move(s), Borrow(m_new)));
        let p = b.op("p", Op::Exp(Move(s)));
        let ps = b.op("ps", Op::RowReduce(Reduce::Sum, Borrow(p)));
        let l = b.op("l", Op::Binary(BinOp::Mul, Move(l), Borrow(alpha)));
        let l = b.op("l", Op::Binary(BinOp::Add, Move(l), Move(ps)));
        let pv = b.op("pv", Op::MatMul(Move(p), Borrow(v), DType::F32));
        let acc = b.op("acc", Op::Binary(BinOp::Mul, Move(acc), Move(alpha)));
        let acc = b.op("acc", Op::Binary(BinOp::Add, Move(acc), Move(pv)));
        vec![m_new, l, acc]
    });
    [out[0], out[1], out[2]]
}

/// Plain softmax attention in f64: `softmax(q kᵀ / √d) v`, row by row.
/// Inputs are the values the kernel sees (already rounded to storage dtype).
pub fn reference(q: &[f32], k: &[f32], v: &[f32], q_rows: usize, seq: usize, d: usize) -> Vec<f32> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut out = vec![0.0f32; q_rows * d];
    let mut w = vec![0.0f64; seq];
    for r in 0..q_rows {
        let qr = &q[r * d..(r + 1) * d];
        for (j, wj) in w.iter_mut().enumerate() {
            let kj = &k[j * d..(j + 1) * d];
            *wj = qr
                .iter()
                .zip(kj)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum::<f64>()
                * scale;
        }
        let mx = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut z = 0.0;
        for wj in w.iter_mut() {
            *wj = (*wj - mx).exp();
            z += *wj;
        }
        for c in 0..d {
            let acc: f64 = (0..seq).map(|j| w[j] * v[j * d + c] as f64).sum();
            out[r * d + c] = (acc / z) as f32;
        }
    }
    out
}
