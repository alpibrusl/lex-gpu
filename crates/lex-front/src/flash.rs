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
//!
//! With `consumers > 0` the kernel is warp-specialised the way FlashAttention-3
//! is on Hopper: one producer warp fills a K/V pipe, and `consumers` consumer
//! warpgroups each own a slice of the query blocks and read every slot:
//!
//! ```text
//! role producer (1 warp):         for i: slot = acquire; commit [K_i, V_i] -> slot
//! role consumer c (4 warps each): for i: kv = receive; attend(&kv.0, &kv.1); release kv
//! ```

use lex_ir::{DType, Space};

use crate::ir::{
    Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, RoleDef, TileTy, Ty, Var, View,
};

/// Problem shape plus schedule.
/// Splits the split-KV combine merges per step of its loop; a capacity
/// holds a whole number of these chunks of splits.
pub const COMBINE_CHUNK: usize = 8;

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
    /// 0: one role does everything. N > 0: a producer warp and N consumer
    /// warpgroups connected by a `stages`-deep pipe.
    pub consumers: usize,
    /// Independent KV heads (or sequences), one grid instance each. Every
    /// tensor gets a leading `heads` factor on its row dimension.
    pub heads: usize,
    /// Rows per head in the K/V tensors: the cache capacity when attention
    /// reads the first `seq` rows of a longer cache. 0 means `seq`.
    pub kv_cap: usize,
}

impl FlashDecode {
    pub fn n_qb(&self) -> usize {
        self.q_rows / self.bq
    }

    pub fn n_kb(&self) -> usize {
        self.seq / self.bk
    }

    pub fn kv_rows(&self) -> usize {
        if self.kv_cap == 0 {
            self.seq
        } else {
            self.kv_cap
        }
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
        if c.heads == 0 {
            return Err("heads must be nonzero".into());
        }
        let pq = b.param("q", c.dtype, &[c.heads * c.q_rows, c.d], false);
        if c.kv_rows() < c.seq {
            return Err(format!("kv_cap {} is shorter than seq {}", c.kv_cap, c.seq));
        }
        let pk = b.param("k", c.dtype, &[c.heads * c.kv_rows(), c.d], false);
        let pv = b.param("v", c.dtype, &[c.heads * c.kv_rows(), c.d], false);
        let po = b.param("o", DType::F32, &[c.heads * c.q_rows, c.d], true);
        let pid = b.grid(c.heads);

        let reg = |dt, shape: &[usize]| TileTy::new(dt, shape, Space::Reg);
        let m_ty = reg(DType::F32, &[c.bq]);
        let acc_ty = reg(DType::F32, &[c.bq, c.d]);
        let kv = c.kv_tile();

        if c.consumers > 0 {
            c.specialized(&mut b, pid, [pq, pk, pv, po])?;
            return Ok(b.finish());
        }

        let (qs, [ms, ls, accs]) = setup(&mut b, &c, pid, pq, 0, c.n_qb(), (c.bq, c.q_rows));

        if c.kv_space != Space::Threadgroup {
            // Synchronous loads straight into registers: nothing to rotate.
            let m_arr = Ty::Array(m_ty.clone(), c.n_qb());
            let carry = vec![m_arr.clone(), m_arr, Ty::Array(acc_ty.clone(), c.n_qb())];
            let out = b.for_range(0, c.n_kb(), vec![ms, ls, accs], carry, |b, i, p| {
                let at = IdxExpr::scaled(i, c.bk, 0);
                let k = b.op(
                    "k",
                    Op::Load(
                        rows(pk, (at.clone()).plus(pid, c.kv_rows()), c.bk, c.d),
                        kv.clone(),
                    ),
                );
                let v = b.op(
                    "v",
                    Op::Load(rows(pv, (at).plus(pid, c.kv_rows()), c.bk, c.d), kv.clone()),
                );
                let out = attend(
                    b,
                    &c,
                    qs,
                    Arg::Borrow(k),
                    Arg::Borrow(v),
                    [p[0], p[1], p[2]],
                    None,
                );
                b.drop(k);
                b.drop(v);
                out.to_vec()
            });
            finish(
                &mut b,
                &c,
                pid,
                po,
                0,
                qs,
                [out[0], out[1], out[2]],
                (c.bq, c.q_rows),
            );
            return Ok(b.finish());
        }

        // Prologue: `stages` buffers per operand; the first stages-1 are
        // issued for blocks 0..stages-1, the last is free.
        let prologue = |b: &mut Builder, p: usize, name: &str| {
            let mut inflight = vec![];
            for blk in 0..c.stages - 1 {
                let buf = b.op(name, Op::Alloc(kv.clone()));
                let view = rows(
                    p,
                    (IdxExpr::lit(blk * c.bk)).plus(pid, c.kv_rows()),
                    c.bk,
                    c.d,
                );
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
                Op::CopyAsync(
                    rows(pk, (ahead.clone()).plus(pid, c.kv_rows()), c.bk, c.d),
                    kfree,
                ),
            );
            let vnew = b.op(
                "vf",
                Op::CopyAsync(rows(pv, (ahead).plus(pid, c.kv_rows()), c.bk, c.d), vfree),
            );
            let mut kq: Vec<Var> = kq.iter().copied().chain([knew]).collect();
            let mut vq: Vec<Var> = vq.iter().copied().chain([vnew]).collect();
            let kcur = b.op("k", Op::Wait(kq.remove(0)));
            let vcur = b.op("v", Op::Wait(vq.remove(0)));

            let [ms, ls, accs] = attend(
                b,
                &c,
                qs,
                Arg::Borrow(kcur),
                Arg::Borrow(vcur),
                [ms, ls, accs],
                None,
            );

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
            [ms, ls, accs] = attend(
                &mut b,
                &c,
                qs,
                Arg::Borrow(kcur),
                Arg::Borrow(vcur),
                [ms, ls, accs],
                None,
            );
            b.drop(kcur);
            b.drop(vcur);
        }
        b.drop(kfree);
        b.drop(vfree);
        finish(&mut b, &c, pid, po, 0, qs, [ms, ls, accs], (c.bq, c.q_rows));
        Ok(b.finish())
    }

    /// Decode attention over a runtime length: one program for every
    /// position of a generation. `kv_rows()` is the cache capacity (and
    /// the static bound); two runtime scalars carry the live length and the
    /// number of `bk` blocks covering it (`ceil(len / bk)`). K/V blocks are
    /// loaded synchronously into threadgroup memory; the last block's
    /// columns past `len` are masked before the softmax.
    pub fn build_dynamic(&self) -> Result<Program, String> {
        let c = *self;
        if c.bq == 0 || c.bk == 0 || !c.q_rows.is_multiple_of(c.bq) {
            return Err("bq must divide q_rows and bk must be nonzero".into());
        }
        if c.consumers != 0 || c.heads == 0 {
            return Err("dynamic attention is single-role and needs heads".into());
        }
        let cap = c.kv_rows();
        if !cap.is_multiple_of(c.bk) {
            return Err(format!(
                "cache capacity {cap} must be whole blocks of {}",
                c.bk
            ));
        }
        let mut b = Builder::new(&format!(
            "flash_decode_dyn_{}_bq{}_bk{}_cap{cap}",
            c.dtype.suffix(),
            c.bq,
            c.bk
        ));
        let pq = b.param("q", c.dtype, &[c.heads * c.q_rows, c.d], false);
        let pk = b.param("k", c.dtype, &[c.heads * cap, c.d], false);
        let pv = b.param("v", c.dtype, &[c.heads * cap, c.d], false);
        let po = b.param("o", DType::F32, &[c.heads * c.q_rows, c.d], true);
        let pid = b.grid(c.heads);
        let len = b.dyn_index("len", cap);
        let nkb = b.dyn_index("nkb", cap / c.bk);

        let (qs, [ms, ls, accs]) = setup(&mut b, &c, pid, pq, 0, c.n_qb(), (c.bq, c.q_rows));
        let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
        let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
        let carry = vec![
            Ty::Array(m_ty.clone(), c.n_qb()),
            Ty::Array(m_ty, c.n_qb()),
            Ty::Array(acc_ty, c.n_qb()),
        ];
        let kv = TileTy::new(c.dtype, &[c.bk, c.d], c.kv_space);
        let out = b.for_range_dyn(0, cap / c.bk, nkb, vec![ms, ls, accs], carry, |b, i, p| {
            let at = IdxExpr::scaled(i, c.bk, 0).plus(pid, cap);
            let k = b.op("k", Op::Load(rows(pk, at.clone(), c.bk, c.d), kv.clone()));
            let v = b.op("v", Op::Load(rows(pv, at, c.bk, c.d), kv.clone()));
            let out = attend(
                b,
                &c,
                qs,
                Arg::Borrow(k),
                Arg::Borrow(v),
                [p[0], p[1], p[2]],
                Some((IdxExpr::scaled(i, c.bk, 0), &|_| IdxExpr::scaled(len, 1, 0))),
            );
            b.drop(k);
            b.drop(v);
            out.to_vec()
        });
        finish(
            &mut b,
            &c,
            pid,
            po,
            0,
            qs,
            [out[0], out[1], out[2]],
            (c.bq, c.q_rows),
        );
        Ok(b.finish())
    }

    /// Causal attention for `tokens` new queries at positions
    /// `pos0 .. pos0 + tokens` (prefill, or verifying drafted tokens), over a
    /// cache that already holds their keys and values.
    ///
    /// Queries and outputs are in the natural `[token, kv head, group, d]`
    /// order the projections produce, so nothing is permuted: with `bq` equal
    /// to the GQA group (`q_rows`), query block `t` of KV head `h` is one
    /// token's heads, at row `t * heads * group + h * group`. Each block is
    /// masked at its own position (`limit = pos0 + t + 1`). Runtime scalars:
    /// `pos0`, and `nkb`, the blocks covering `pos0 + tokens`.
    pub fn build_causal(&self, tokens: usize) -> Result<Program, String> {
        let c = *self;
        let cap = c.kv_rows();
        if c.bq != c.q_rows || c.bq == 0 || c.bk == 0 || c.heads == 0 || c.consumers != 0 {
            return Err("causal attention takes one token's group per query block".into());
        }
        if !cap.is_multiple_of(c.bk) || tokens == 0 || tokens > cap {
            return Err(format!(
                "{tokens} tokens over a cache of {cap} in blocks of {}",
                c.bk
            ));
        }
        let group = c.q_rows;
        let hg = c.heads * group;
        let mut b = Builder::new(&format!(
            "flash_causal_{}_t{tokens}_g{group}_bk{}_cap{cap}",
            c.dtype.suffix(),
            c.bk
        ));
        let pq = b.param("q", c.dtype, &[tokens * hg, c.d], false);
        let pk = b.param("k", c.dtype, &[c.heads * cap, c.d], false);
        let pv = b.param("v", c.dtype, &[c.heads * cap, c.d], false);
        let po = b.param("o", DType::F32, &[tokens * hg, c.d], true);
        let pid = b.grid(c.heads);
        let pos0 = b.dyn_index("pos0", cap - tokens);
        let nkb = b.dyn_index("nkb", cap / c.bk);

        let strides = (hg, group);
        let (qs, [ms, ls, accs]) = setup(&mut b, &c, pid, pq, 0, tokens, strides);
        let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
        let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
        let carry = vec![
            Ty::Array(m_ty.clone(), tokens),
            Ty::Array(m_ty, tokens),
            Ty::Array(acc_ty, tokens),
        ];
        let kv = TileTy::new(c.dtype, &[c.bk, c.d], c.kv_space);
        let limit = move |qb: Var| IdxExpr::lit(1).plus(pos0, 1).plus(qb, 1);
        let out = b.for_range_dyn(0, cap / c.bk, nkb, vec![ms, ls, accs], carry, |b, i, p| {
            let at = IdxExpr::scaled(i, c.bk, 0).plus(pid, cap);
            let k = b.op("k", Op::Load(rows(pk, at.clone(), c.bk, c.d), kv.clone()));
            let v = b.op("v", Op::Load(rows(pv, at, c.bk, c.d), kv.clone()));
            let out = attend(
                b,
                &c,
                qs,
                Arg::Borrow(k),
                Arg::Borrow(v),
                [p[0], p[1], p[2]],
                Some((IdxExpr::scaled(i, c.bk, 0), &limit)),
            );
            b.drop(k);
            b.drop(v);
            out.to_vec()
        });
        finish(
            &mut b,
            &c,
            pid,
            po,
            0,
            qs,
            [out[0], out[1], out[2]],
            strides,
        );
        Ok(b.finish())
    }

    /// Split-KV decode attention, first half: instance `(head, split)`
    /// attends over the `bps` blocks of the cache that split covers and
    /// writes its partial softmax state — running max `m`, sum `l` and
    /// unnormalised `acc` — for [`FlashDecode::build_combine`] to merge.
    ///
    /// The splits' grid dimension is the capacity's; a runtime launches only
    /// `ceil(nkb / bps)` of them. Positions past the live length (scalar
    /// `len`) are masked, so the last split's tail costs nothing but loads.
    /// One query block per head: `bq == q_rows` (the GQA group).
    ///
    /// Partials are laid out by query row, split-minor: `part_m[row, split]`
    /// and `part_acc[row, split * d ..]`, so the combine reads one row's
    /// splits contiguously.
    pub fn build_split(&self, bps: usize) -> Result<Program, String> {
        let c = *self;
        let cap = c.kv_rows();
        if c.bq != c.q_rows || c.consumers != 0 || c.heads == 0 || bps == 0 {
            return Err("split attention takes one query block per head".into());
        }
        if !cap.is_multiple_of(c.bk * bps * COMBINE_CHUNK) {
            return Err(format!(
                "capacity {cap} is not whole chunks of {COMBINE_CHUNK} splits of {bps} x {}",
                c.bk
            ));
        }
        let splits = cap / (c.bk * bps);
        let (group, hg) = (c.q_rows, c.heads * c.q_rows);
        let mut b = Builder::new(&format!(
            "flash_split_{}_g{group}_bk{}_bps{bps}_cap{cap}",
            c.dtype.suffix(),
            c.bk
        ));
        let pq = b.param("q", c.dtype, &[hg, c.d], false);
        let pk = b.param("k", c.dtype, &[c.heads * cap, c.d], false);
        let pv = b.param("v", c.dtype, &[c.heads * cap, c.d], false);
        let pm = b.param("part_m", DType::F32, &[hg, splits], true);
        let pl = b.param("part_l", DType::F32, &[hg, splits], true);
        let pa = b.param("part_acc", DType::F32, &[hg, splits * c.d], true);
        let pid = b.grid(c.heads);
        let split = b.grid2(splits);
        let len = b.dyn_index("len", cap);

        let (qs, [ms, ls, accs]) = setup_from(&mut b, &c, pid, pq, 0, 1, (c.bq, group), -1e30);
        let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
        let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
        let carry = vec![
            Ty::Array(m_ty.clone(), 1),
            Ty::Array(m_ty.clone(), 1),
            Ty::Array(acc_ty.clone(), 1),
        ];
        let kv = TileTy::new(c.dtype, &[c.bk, c.d], c.kv_space);
        let out = b.for_range(0, bps, vec![ms, ls, accs], carry, |b, i, p| {
            // Block `split * bps + i` of this head.
            let first = IdxExpr::scaled(i, c.bk, 0).plus(split, bps * c.bk);
            let at = first.clone().plus(pid, cap);
            let k = b.op("k", Op::Load(rows(pk, at.clone(), c.bk, c.d), kv.clone()));
            let v = b.op("v", Op::Load(rows(pv, at, c.bk, c.d), kv.clone()));
            let out = attend(
                b,
                &c,
                qs,
                Arg::Borrow(k),
                Arg::Borrow(v),
                [p[0], p[1], p[2]],
                Some((first, &|_| IdxExpr::scaled(len, 1, 0))),
            );
            b.drop(k);
            b.drop(v);
            out.to_vec()
        });
        // This head's rows, this split's column.
        let row = IdxExpr::scaled(pid, group, 0);
        b.map_each(
            vec![out[0], out[1], out[2]],
            vec![Ty::Tile(m_ty.clone()), Ty::Tile(m_ty), Ty::Tile(acc_ty)],
            0,
            |b, _, p| {
                let column = |param| View {
                    param,
                    offset: vec![row.clone(), IdxExpr::scaled(split, 1, 0)],
                    shape: vec![group, 1],
                };
                b.effect(Op::Store(Arg::Move(p[0]), column(pm)));
                b.effect(Op::Store(Arg::Move(p[1]), column(pl)));
                b.effect(Op::Store(
                    Arg::Move(p[2]),
                    View {
                        param: pa,
                        offset: vec![row.clone(), IdxExpr::scaled(split, c.d, 0)],
                        shape: vec![group, c.d],
                    },
                ));
                vec![]
            },
        );
        b.drop(qs);
        Ok(b.finish())
    }

    /// Split-KV attention for `tokens` queries at once, with causal
    /// masking: instance `(head, split)` attends every query to the `bps`
    /// blocks that split covers, and writes a partial state per query for
    /// [`FlashDecode::build_combine_rows`] to merge.
    ///
    /// This is [`FlashDecode::build_split`] carrying a token dimension and
    /// [`FlashDecode::build_causal`]'s mask. It exists because a batched
    /// verify scans the whole cache in one threadgroup per head, which is
    /// what a decode step did before split-KV: at 1440 positions a
    /// two-token verify costs 67.9 ms where two separate decode steps cost
    /// 74, so batching has stopped buying anything. Speculation needs the
    /// verify to be cheaper than the steps it replaces.
    ///
    /// The mask is the causal one, `1 + pos0 + qb`, which also bounds the
    /// live length: nothing past the last query's position is attended, so
    /// a split beyond it costs its loads and no arithmetic.
    pub fn build_causal_split(&self, tokens: usize, bps: usize) -> Result<Program, String> {
        let c = *self;
        let cap = c.kv_rows();
        if c.bq != c.q_rows || c.consumers != 0 || c.heads == 0 || bps == 0 {
            return Err("split attention takes one query block per head".into());
        }
        if !cap.is_multiple_of(c.bk * bps * COMBINE_CHUNK) {
            return Err(format!(
                "capacity {cap} is not whole chunks of {COMBINE_CHUNK} splits of {bps} x {}",
                c.bk
            ));
        }
        if tokens == 0 || tokens > cap {
            return Err(format!("{tokens} tokens over a cache of {cap}"));
        }
        let splits = cap / (c.bk * bps);
        let (group, hg) = (c.q_rows, c.heads * c.q_rows);
        let rows_total = tokens * hg;
        let mut b = Builder::new(&format!(
            "flash_causal_split{tokens}_{}_g{group}_bk{}_bps{bps}_cap{cap}",
            c.dtype.suffix(),
            c.bk
        ));
        let pq = b.param("q", c.dtype, &[rows_total, c.d], false);
        let pk = b.param("k", c.dtype, &[c.heads * cap, c.d], false);
        let pv = b.param("v", c.dtype, &[c.heads * cap, c.d], false);
        let pm = b.param("part_m", DType::F32, &[rows_total, splits], true);
        let pl = b.param("part_l", DType::F32, &[rows_total, splits], true);
        let pa = b.param("part_acc", DType::F32, &[rows_total, splits * c.d], true);
        let pid = b.grid(c.heads);
        let split = b.grid2(splits);
        let pos0 = b.dyn_index("pos0", cap - tokens);

        let strides = (hg, group);
        // A large finite negative, not -inf: a split that sees no position
        // must produce a zero-weight partial rather than a NaN.
        let (qs, [ms, ls, accs]) = setup_from(&mut b, &c, pid, pq, 0, tokens, strides, -1e30);
        let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
        let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
        let carry = vec![
            Ty::Array(m_ty.clone(), tokens),
            Ty::Array(m_ty.clone(), tokens),
            Ty::Array(acc_ty.clone(), tokens),
        ];
        let kv = TileTy::new(c.dtype, &[c.bk, c.d], c.kv_space);
        let limit = move |qb: Var| IdxExpr::lit(1).plus(pos0, 1).plus(qb, 1);
        let out = b.for_range(0, bps, vec![ms, ls, accs], carry, |b, i, p| {
            let first = IdxExpr::scaled(i, c.bk, 0).plus(split, bps * c.bk);
            let at = first.clone().plus(pid, cap);
            let k = b.op("k", Op::Load(rows(pk, at.clone(), c.bk, c.d), kv.clone()));
            let v = b.op("v", Op::Load(rows(pv, at, c.bk, c.d), kv.clone()));
            let out = attend(
                b,
                &c,
                qs,
                Arg::Borrow(k),
                Arg::Borrow(v),
                [p[0], p[1], p[2]],
                Some((first, &limit)),
            );
            b.drop(k);
            b.drop(v);
            out.to_vec()
        });
        // Row `token * hg + head * group`, this split's column. `map_each`
        // hands the token index in, which is what makes this taller than
        // the decode store rather than different from it.
        b.map_each(
            vec![out[0], out[1], out[2]],
            vec![Ty::Tile(m_ty.clone()), Ty::Tile(m_ty), Ty::Tile(acc_ty)],
            0,
            |b, qb, p| {
                let row = IdxExpr::scaled(qb, hg, 0).plus(pid, group);
                let column = |param| View {
                    param,
                    offset: vec![row.clone(), IdxExpr::scaled(split, 1, 0)],
                    shape: vec![group, 1],
                };
                b.effect(Op::Store(Arg::Move(p[0]), column(pm)));
                b.effect(Op::Store(Arg::Move(p[1]), column(pl)));
                b.effect(Op::Store(
                    Arg::Move(p[2]),
                    View {
                        param: pa,
                        offset: vec![row.clone(), IdxExpr::scaled(split, c.d, 0)],
                        shape: vec![group, c.d],
                    },
                ));
                vec![]
            },
        );
        b.drop(qs);
        Ok(b.finish())
    }

    /// Split-KV decode attention, second half: one instance per query row
    /// merges that row's first `nsplit` partial states (a runtime scalar;
    /// `nchunk = ceil(nsplit / COMBINE_CHUNK)`) and writes `o = acc / l`.
    ///
    /// Every split at once, as llama.cpp's reduce does: the global max `M`
    /// over the row's split maxima, weights `w = exp(m - M)`, `l = Σ w·l_s`,
    /// and `acc = w · acc_s` as a `[1, splits] x [splits, d]` matmul, taken
    /// `COMBINE_CHUNK` splits at a time so only live splits are read.
    pub fn build_combine(&self, bps: usize) -> Result<Program, String> {
        self.build_combine_rows(1, bps)
    }

    /// [`FlashDecode::build_combine`] for `tokens` query rows at once, to
    /// merge what [`FlashDecode::build_causal_split`] wrote. One instance
    /// per (token, head, query) row; the rows are independent, so this is
    /// the decode combine with a taller grid.
    pub fn build_combine_rows(&self, tokens: usize, bps: usize) -> Result<Program, String> {
        use Arg::{Borrow, Move};
        let c = *self;
        let cap = c.kv_rows();
        let splits = cap / (c.bk * bps);
        if !splits.is_multiple_of(COMBINE_CHUNK) {
            return Err(format!(
                "{splits} splits are not whole chunks of {COMBINE_CHUNK}"
            ));
        }
        let ch = COMBINE_CHUNK;
        let hg = tokens * c.heads * c.q_rows;
        let mut b = Builder::new(&format!("flash_combine_splits{splits}_d{}", c.d));
        let pm = b.param("part_m", DType::F32, &[hg, splits], false);
        let pl = b.param("part_l", DType::F32, &[hg, splits], false);
        let pa = b.param("part_acc", DType::F32, &[hg * splits, c.d], false);
        let po = b.param("o", DType::F32, &[hg, c.d], true);
        let pid = b.grid(hg);
        let nsplit = b.dyn_index("nsplit", splits);
        let nchunk = b.dyn_index("nchunk", splits / ch);
        let live = IdxExpr::scaled(nsplit, 1, 0);
        let row_of = |param, first: IdxExpr, n| View {
            param,
            offset: vec![IdxExpr::scaled(pid, 1, 0), first],
            shape: vec![1, n],
        };
        let reg = |shape: &[usize]| TileTy::new(DType::F32, shape, Space::Reg);

        let m = b.op(
            "m",
            Op::Load(row_of(pm, IdxExpr::lit(0), splits), reg(&[1, splits])),
        );
        let m = b.op("m", Op::MaskCols(Move(m), IdxExpr::lit(0), live.clone()));
        let mx = b.op("mx", Op::RowReduce(Reduce::Max, Borrow(m)));
        let d = b.op("dm", Op::Binary(BinOp::Sub, Move(m), Borrow(mx)));
        let w = b.op("w", Op::Exp(Move(d)));
        let ls = b.op(
            "ls",
            Op::Load(row_of(pl, IdxExpr::lit(0), splits), reg(&[1, splits])),
        );
        let wl = b.op("wl", Op::Binary(BinOp::Mul, Move(w), Move(ls)));
        let l = b.op("l", Op::RowReduce(Reduce::Sum, Move(wl)));

        let acc0 = b.op("acc", Op::Fill(reg(&[1, c.d]), 0.0));
        let chunk = TileTy::new(DType::F32, &[ch, c.d], Space::Threadgroup);
        let out = b.for_range_dyn(
            0,
            splits / ch,
            nchunk,
            vec![acc0],
            vec![Ty::Tile(reg(&[1, c.d]))],
            |b, k, p| {
                let first = IdxExpr::scaled(k, ch, 0);
                let mc = b.op("mc", Op::Load(row_of(pm, first.clone(), ch), reg(&[1, ch])));
                let mc = b.op("mc", Op::MaskCols(Move(mc), first.clone(), live.clone()));
                let d = b.op("dc", Op::Binary(BinOp::Sub, Move(mc), Borrow(mx)));
                let wc = b.op("wc", Op::Exp(Move(d)));
                let at = IdxExpr::scaled(pid, splits, 0).plus(k, ch);
                let ac = b.op("ac", Op::Load(rows(pa, at, ch, c.d), chunk.clone()));
                let pv = b.op("pv", Op::MatMul(Move(wc), Borrow(ac), DType::F32));
                b.drop(ac);
                let acc = b.op("acc", Op::Binary(BinOp::Add, Move(p[0]), Move(pv)));
                vec![acc]
            },
        );
        b.drop(mx);
        let o = b.op("o", Op::Binary(BinOp::Div, Move(out[0]), Move(l)));
        b.effect(Op::Store(
            Move(o),
            rows(po, IdxExpr::scaled(pid, 1, 0), 1, c.d),
        ));
        Ok(b.finish())
    }

    /// Producer warp + `consumers` consumer warpgroups over one K/V pipe.
    fn specialized(
        &self,
        b: &mut Builder,
        pid: Var,
        [pq, pk, pv, po]: [usize; 4],
    ) -> Result<(), String> {
        let c = *self;
        if c.kv_space != Space::Threadgroup {
            return Err("a pipe's slots live in threadgroup memory".into());
        }
        if !c.n_qb().is_multiple_of(c.consumers) {
            return Err(format!(
                "{} query blocks do not split over {} consumers",
                c.n_qb(),
                c.consumers
            ));
        }
        let per = c.n_qb() / c.consumers;
        let kv = c.kv_tile();
        let pipe = b.pipe(vec![kv.clone(), kv], c.stages, c.consumers);

        let mut roles = vec![RoleDef {
            name: "producer",
            warps: 1,
            inputs: vec![(pipe.ty.id, Ty::Producer(pipe.ty.clone()))],
            n_results: 0,
            body: Box::new(move |b: &mut Builder, p: &[Var]| {
                let h = p[0];
                b.for_range(0, c.n_kb(), vec![], vec![], |b, i, _| {
                    let at = IdxExpr::scaled(i, c.bk, 0);
                    let slot = b.op("slot", Op::Acquire(h));
                    let views = vec![
                        rows(pk, (at.clone()).plus(pid, c.kv_rows()), c.bk, c.d),
                        rows(pv, (at).plus(pid, c.kv_rows()), c.bk, c.d),
                    ];
                    b.effect(Op::Commit(h, views, slot));
                    vec![]
                });
                vec![]
            }),
        }];
        for (k, &cons) in pipe.consumers.iter().enumerate() {
            let ty = Ty::Consumer(pipe.ty.clone(), k);
            roles.push(RoleDef {
                name: "consumer",
                warps: 4,
                inputs: vec![(cons, ty)],
                n_results: 0,
                body: Box::new(move |b: &mut Builder, p: &[Var]| {
                    let h = p[0];
                    let q0 = k * per * c.bq;
                    let (qs, state) = setup(b, &c, pid, pq, q0, per, (c.bq, c.q_rows));
                    let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
                    let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
                    let carry = vec![
                        Ty::Array(m_ty.clone(), per),
                        Ty::Array(m_ty, per),
                        Ty::Array(acc_ty, per),
                    ];
                    let out = b.for_range(0, c.n_kb(), state.to_vec(), carry, |b, _, p| {
                        let kv = b.op("kv", Op::Receive(h));
                        let out = attend(
                            b,
                            &c,
                            qs,
                            Arg::BorrowPart(kv, 0),
                            Arg::BorrowPart(kv, 1),
                            [p[0], p[1], p[2]],
                            None,
                        );
                        b.effect(Op::Release(h, kv));
                        out.to_vec()
                    });
                    finish(
                        b,
                        &c,
                        pid,
                        po,
                        q0,
                        qs,
                        [out[0], out[1], out[2]],
                        (c.bq, c.q_rows),
                    );
                    vec![]
                }),
            });
        }
        b.specialize(vec![pipe], roles);
        Ok(())
    }
}

/// Load `n` query blocks starting at row `q0` (only ever borrowed after
/// this), and the online-softmax state for each.
/// Query block `qb` of instance `pid` starts at row
/// `q0 + qb * strides.0 + pid * strides.1` (decode: `(bq, q_rows)`).
fn setup(
    b: &mut Builder,
    c: &FlashDecode,
    pid: Var,
    pq: usize,
    q0: usize,
    n: usize,
    strides: (usize, usize),
) -> (Var, [Var; 3]) {
    setup_from(b, c, pid, pq, q0, n, strides, f32::NEG_INFINITY)
}

/// [`setup`] with the running max starting at `m0`. Split-KV attention
/// starts at a large finite negative so a split that sees no position (all
/// masked) produces a zero-weight partial instead of NaN.
#[allow(clippy::too_many_arguments)]
fn setup_from(
    b: &mut Builder,
    c: &FlashDecode,
    pid: Var,
    pq: usize,
    q0: usize,
    n: usize,
    strides: (usize, usize),
    m0: f32,
) -> (Var, [Var; 3]) {
    let reg = |dt, shape: &[usize]| TileTy::new(dt, shape, Space::Reg);
    let q_blocks: Vec<Var> = (0..n)
        .map(|qb| {
            let view = rows(
                pq,
                (IdxExpr::lit(q0 + qb * strides.0)).plus(pid, strides.1),
                c.bq,
                c.d,
            );
            b.op("q", Op::Load(view, reg(c.dtype, &[c.bq, c.d])))
        })
        .collect();
    let qs = b.op("qs", Op::MakeArray(q_blocks));
    let mut state = |name: &str, ty: TileTy, x: f32| {
        let v: Vec<Var> = (0..n)
            .map(|_| b.op(name, Op::Fill(ty.clone(), x)))
            .collect();
        b.op(name, Op::MakeArray(v))
    };
    let ms = state("m", reg(DType::F32, &[c.bq]), m0);
    let ls = state("l", reg(DType::F32, &[c.bq]), 0.0);
    let accs = state("acc", reg(DType::F32, &[c.bq, c.d]), 0.0);
    (qs, [ms, ls, accs])
}

/// `o[qb] = acc / l`, then release the query blocks.
#[allow(clippy::too_many_arguments)]
fn finish(
    b: &mut Builder,
    c: &FlashDecode,
    pid: Var,
    po: usize,
    q0: usize,
    qs: Var,
    [ms, ls, accs]: [Var; 3],
    strides: (usize, usize),
) {
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
            let view = rows(
                po,
                (IdxExpr::scaled(qb, strides.0, q0)).plus(pid, strides.1),
                c.bq,
                c.d,
            );
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
/// With `mask = Some((block, len))`, the columns of block `block` at or past
/// the runtime length `len` are masked out before the softmax.
fn attend(
    b: &mut Builder,
    c: &FlashDecode,
    qs: Var,
    k: Arg,
    v: Arg,
    state: [Var; 3],
    mask: Option<(IdxExpr, &dyn Fn(Var) -> IdxExpr)>,
) -> [Var; 3] {
    use Arg::{Borrow, Move};
    let m_ty = TileTy::new(DType::F32, &[c.bq], Space::Reg);
    let acc_ty = TileTy::new(DType::F32, &[c.bq, c.d], Space::Reg);
    let elem = vec![Ty::Tile(m_ty.clone()), Ty::Tile(m_ty), Ty::Tile(acc_ty)];
    let scale = 1.0 / (c.d as f32).sqrt();
    let out = b.map_each(state.to_vec(), elem, 3, |b, qb, p| {
        let (m, l, acc) = (p[0], p[1], p[2]);
        let s = b.op("s", Op::MatMulNT(Arg::BorrowElem(qs, qb), k, DType::F32));
        let mut s = b.op("s", Op::Scale(Move(s), scale));
        if let Some((first, limit)) = &mask {
            s = b.op("s", Op::MaskCols(Move(s), first.clone(), limit(qb)));
        }
        let mb = b.op("mb", Op::RowReduce(Reduce::Max, Borrow(s)));
        let m_new = b.op("m", Op::Binary(BinOp::Max, Borrow(m), Move(mb)));
        let d = b.op("dm", Op::Binary(BinOp::Sub, Move(m), Borrow(m_new)));
        let alpha = b.op("alpha", Op::Exp(Move(d)));
        let s = b.op("s", Op::Binary(BinOp::Sub, Move(s), Borrow(m_new)));
        let p = b.op("p", Op::Exp(Move(s)));
        let ps = b.op("ps", Op::RowReduce(Reduce::Sum, Borrow(p)));
        let l = b.op("l", Op::Binary(BinOp::Mul, Move(l), Borrow(alpha)));
        let l = b.op("l", Op::Binary(BinOp::Add, Move(l), Move(ps)));
        let pv = b.op("pv", Op::MatMul(Move(p), v, DType::F32));
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
