//! The layer kernels of a Llama decode step, as typed tile programs.
//!
//! Each is an ordinary `tile-front` program: it type-checks, runs in the
//! interpreter, and lowers to MSL like flash decode does. Activations are f32
//! rows `[1, n]`. Weights (Q8_0, Q4_K, Q6_K) are repacked by the loader into
//! an `I8` value tensor plus an f32 scale (and min) per group, and meet their
//! scales through `dequant` — the only way the checker lets anyone compute
//! with `I8`.
//!
//! Attention is [`crate::flash::FlashDecode`]; what is here is everything
//! around it.

use tile_ir::{DType, Space};

use crate::ir::{Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, TileTy, Ty, UnOp, Var, View};

fn reg(dt: DType, shape: &[usize]) -> TileTy {
    TileTy::new(dt, shape, Space::Reg)
}

fn at(param: usize, offset: [IdxExpr; 2], shape: [usize; 2]) -> View {
    View {
        param,
        offset: offset.to_vec(),
        shape: shape.to_vec(),
    }
}

fn row(param: usize, n: usize) -> View {
    at(param, [IdxExpr::lit(0), IdxExpr::lit(0)], [1, n])
}

/// `y = x * rsqrt(mean(x^2) + eps) * w` over one row of `n`.
pub fn rmsnorm(n: usize, eps: f32) -> Program {
    use Arg::{Borrow, Move};
    let mut b = Builder::new(&format!("rmsnorm_{n}"));
    let px = b.param("x", DType::F32, &[1, n], false);
    let pw = b.param("w", DType::F32, &[1, n], false);
    let py = b.param("y", DType::F32, &[1, n], true);
    let x = b.op("x", Op::Load(row(px, n), reg(DType::F32, &[1, n])));
    let sq = b.op("sq", Op::Binary(BinOp::Mul, Borrow(x), Borrow(x)));
    let ss = b.op("ss", Op::RowReduce(Reduce::Sum, Move(sq)));
    let ms = b.op("ms", Op::Scale(Move(ss), 1.0 / n as f32));
    let e = b.op("eps", Op::Fill(reg(DType::F32, &[1]), eps));
    let t = b.op("t", Op::Binary(BinOp::Add, Move(ms), Move(e)));
    let r = b.op("r", Op::Unary(UnOp::Rsqrt, Move(t)));
    let xn = b.op("xn", Op::Binary(BinOp::Mul, Move(x), Move(r)));
    let w = b.op("w", Op::Load(row(pw, n), reg(DType::F32, &[1, n])));
    let y = b.op("y", Op::Binary(BinOp::Mul, Move(xn), Move(w)));
    b.effect(Op::Store(Move(y), row(py, n)));
    b.finish()
}

/// How a quantised matrix is laid out for the kernels: `I8` values, and per
/// `group` values along a row either an f16 scale, or a two-level scale as
/// the K-quants store it — a small integer `sc` per group times an f16 `d`
/// per super-block of `sub` groups (and likewise a min, for affine formats).
/// Scales stay in the file's own form; the kernel forms `d * sc` inline,
/// through the same checked `dequant` that meets values with their scales.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QLayout {
    pub group: usize,
    /// `None`: one f16 scale per group (Q8_0). `Some(sub)`: two-level, `sub`
    /// groups per super-block (Q4_K, Q6_K).
    pub super_groups: Option<usize>,
    /// Affine: a two-level min per group as well (`q * scale - min`).
    pub min: bool,
    /// Unsigned 4-bit values, two per byte, paired across a chunk of the
    /// matvec's width `kc`: byte `k` of each chunk holds columns `k` (low
    /// nibble) and `k + kc/2` (high nibble). The loader packs for the width
    /// the schedule will use ([`Split::pack`]); the layout is the schedule's
    /// choice, made once at load time.
    pub packed4: bool,
}

impl QLayout {
    /// Q8_0: 32 values, one f16 scale.
    pub const Q8_0: QLayout = QLayout {
        group: 32,
        super_groups: None,
        min: false,
        packed4: false,
    };
    /// Q6_K: 16 values per `i8` sub-block scale, 16 sub-blocks per f16 `d`;
    /// values centred at 0.
    pub const Q6_K: QLayout = QLayout {
        group: 16,
        super_groups: Some(16),
        min: false,
        packed4: false,
    };
    /// Q4_K: 32 values per 6-bit scale and min, 8 per f16 `d` and `dmin`:
    /// `d * sc * q - dmin * m`, with the 4-bit values packed two per byte.
    pub const Q4_K: QLayout = QLayout {
        group: 32,
        super_groups: Some(8),
        min: true,
        packed4: true,
    };

    fn tag(self) -> String {
        format!(
            "g{}{}{}{}",
            self.group,
            self.super_groups.map_or(String::new(), |s| format!("s{s}")),
            if self.min { "m" } else { "" },
            if self.packed4 { "p4" } else { "" }
        )
    }

    /// Weight parameters after the values, in order, with their dtypes and
    /// widths as a divisor of the row length.
    fn scale_params(self) -> Vec<(&'static str, DType, usize)> {
        let g = self.group;
        match self.super_groups {
            None => vec![("ws", DType::F16, g)],
            Some(sub) => {
                let mut v = vec![("wsc", DType::I8, g), ("wd", DType::F16, g * sub)];
                if self.min {
                    v.extend([("wmn", DType::I8, g), ("wdmin", DType::F16, g * sub)]);
                }
                v
            }
        }
    }
}

/// `y = W x (+ r)` for a quantised matrix `W: [n_out, n_in]`.
///
/// One grid instance per `bo` output rows; the input is walked in chunks of
/// `kc` with an f32 accumulator carried through the loop. Parameters are
/// `x, values, <scales per QLayout::scale_params>, [r], y`.
pub fn matvec_q(
    n_in: usize,
    n_out: usize,
    bo: usize,
    kc: usize,
    layout: QLayout,
    residual: bool,
) -> Result<Program, String> {
    use Arg::Move;
    let g = layout.group;
    let sup = g * layout.super_groups.unwrap_or(1);
    let half = if layout.packed4 { kc / 2 } else { kc };
    if !n_out.is_multiple_of(bo) || !n_in.is_multiple_of(kc) || !half.is_multiple_of(sup) {
        return Err(format!(
            "matvec {n_out}x{n_in}: bo {bo} must divide the rows, kc {kc} the columns, and \
             each {}chunk must be whole super-blocks of {sup}",
            if layout.packed4 { "half-" } else { "" }
        ));
    }
    let name = format!(
        "matvec_{}_{n_out}x{n_in}{}",
        layout.tag(),
        if residual { "_res" } else { "" }
    );
    let mut b = Builder::new(&name);
    let px = b.param("x", DType::F32, &[1, n_in], false);
    let qcols = if layout.packed4 { n_in / 2 } else { n_in };
    let pq = b.param("wq", DType::I8, &[n_out, qcols], false);
    let pscales: Vec<(usize, DType, usize)> = layout
        .scale_params()
        .into_iter()
        .map(|(name, dt, per)| (b.param(name, dt, &[n_out, n_in / per], false), dt, per))
        .collect();
    let pr = residual.then(|| b.param("r", DType::F32, &[1, n_out], false));
    let py = b.param("y", DType::F32, &[1, n_out], true);
    let pid = b.grid(n_out / bo);

    let acc_ty = reg(DType::F32, &[1, bo]);
    let acc = b.op("acc", Op::Fill(acc_ty.clone(), 0.0));
    let out = b.for_range(
        0,
        n_in / kc,
        vec![acc],
        vec![Ty::Tile(acc_ty)],
        |b, c, p| {
            let rows = IdxExpr::scaled(pid, bo, 0);
            let x_at = |b: &mut Builder, col0: usize, cols: usize| {
                b.op(
                    "x",
                    Op::Load(
                        at(
                            px,
                            [IdxExpr::lit(0), IdxExpr::scaled(c, kc, col0)],
                            [1, cols],
                        ),
                        reg(DType::F32, &[1, cols]),
                    ),
                )
            };
            // One weight-parameter tile for columns `col0..col0+cols` of
            // this chunk, at `per` values per element.
            let load =
                |b: &mut Builder, (p, dt, per): (usize, DType, usize), col0: usize, cols: usize| {
                    b.op(
                        "w",
                        Op::Load(
                            at(
                                p,
                                [rows.clone(), IdxExpr::scaled(c, kc / per, col0 / per)],
                                [bo, cols / per],
                            ),
                            reg(dt, &[bo, cols / per]),
                        ),
                    )
                };
            // Per-group scales (and mins) for a column range: f16 directly,
            // or `sc * d` through `dequant` for the two-level K-quants.
            let scales = |b: &mut Builder, col0: usize, cols: usize| -> (Var, Option<Arg>) {
                match layout.super_groups {
                    None => (load(b, pscales[0], col0, cols), None),
                    Some(sub) => {
                        let sc = load(b, pscales[0], col0, cols);
                        let d = load(b, pscales[1], col0, cols);
                        let s = b.op("s", Op::Dequant(Move(sc), Move(d), None, sub));
                        let m = layout.min.then(|| {
                            let mn = load(b, pscales[2], col0, cols);
                            let dmin = load(b, pscales[3], col0, cols);
                            Move(b.op("m", Op::Dequant(Move(mn), Move(dmin), None, sub)))
                        });
                        (s, m)
                    }
                }
            };
            let part = if layout.packed4 {
                // Byte k of a chunk holds columns k (low nibble) and
                // k + kc/2 (high): each half dequantises in the thread that
                // owns the byte, and the dot product splits in two.
                let h = kc / 2;
                let q = b.op(
                    "q",
                    Op::Load(
                        at(pq, [rows.clone(), IdxExpr::scaled(c, h, 0)], [bo, h]),
                        reg(DType::I8, &[bo, h]),
                    ),
                );
                let mut halves = vec![];
                for (high, col0) in [(false, 0), (true, h)] {
                    let x = x_at(b, col0, h);
                    let (s, m) = scales(b, col0, h);
                    let qa = if high { Move(q) } else { Arg::Borrow(q) };
                    let w = b.op("w", Op::Dequant4(qa, Move(s), m, g, high));
                    halves.push(b.op("part", Op::MatMulNT(Move(x), Move(w), DType::F32)));
                }
                b.op(
                    "part",
                    Op::Binary(BinOp::Add, Move(halves[0]), Move(halves[1])),
                )
            } else {
                let x = x_at(b, 0, kc);
                let q = b.op(
                    "q",
                    Op::Load(
                        at(pq, [rows.clone(), IdxExpr::scaled(c, kc, 0)], [bo, kc]),
                        reg(DType::I8, &[bo, kc]),
                    ),
                );
                let (s, m) = scales(b, 0, kc);
                let w = b.op("w", Op::Dequant(Move(q), Move(s), m, g));
                b.op("part", Op::MatMulNT(Move(x), Move(w), DType::F32))
            };
            vec![b.op("acc", Op::Binary(BinOp::Add, Move(p[0]), Move(part)))]
        },
    );
    let mut y = out[0];
    let dst = at(py, [IdxExpr::lit(0), IdxExpr::scaled(pid, bo, 0)], [1, bo]);
    if let Some(pr) = pr {
        let r = b.op(
            "r",
            Op::Load(
                at(pr, [IdxExpr::lit(0), IdxExpr::scaled(pid, bo, 0)], [1, bo]),
                reg(DType::F32, &[1, bo]),
            ),
        );
        y = b.op("y", Op::Binary(BinOp::Add, Move(y), Move(r)));
    }
    b.effect(Op::Store(Move(y), dst));
    Ok(b.finish())
}

/// RoPE on `heads` rows of `hd`, rotating adjacent pairs (llama.cpp's
/// layout for GGUF Llama weights), narrowing to `out` for the attention
/// kernel. `cos` is `[c0, c0, c1, c1, …]` and `sin` is sign-folded
/// `[-s0, s0, -s1, s1, …]`, so the rotation is `x*cos + swap_pairs(x)*sin`.
pub fn rope(heads: usize, hd: usize, out: DType) -> Program {
    use Arg::{Borrow, Move};
    let mut b = Builder::new(&format!("rope_{heads}x{hd}_{}", out.suffix()));
    let px = b.param("x", DType::F32, &[heads, hd], false);
    let pc = b.param("cos", DType::F32, &[1, hd], false);
    let ps = b.param("sin", DType::F32, &[1, hd], false);
    let py = b.param("y", out, &[heads, hd], true);
    let all = [IdxExpr::lit(0), IdxExpr::lit(0)];
    let x = b.op(
        "x",
        Op::Load(
            at(px, all.clone(), [heads, hd]),
            reg(DType::F32, &[heads, hd]),
        ),
    );
    let c = b.op("cos", Op::Load(row(pc, hd), reg(DType::F32, &[1, hd])));
    let s = b.op("sin", Op::Load(row(ps, hd), reg(DType::F32, &[1, hd])));
    let sw = b.op("sw", Op::SwapPairs(Borrow(x)));
    let a = b.op("a", Op::Binary(BinOp::Mul, Move(x), Move(c)));
    let z = b.op("z", Op::Binary(BinOp::Mul, Move(sw), Move(s)));
    let y = b.op("y", Op::Binary(BinOp::Add, Move(a), Move(z)));
    let y = if out == DType::F32 {
        y
    } else {
        b.op("y", Op::Convert(Move(y), out))
    };
    b.effect(Op::Store(Move(y), at(py, all, [heads, hd])));
    b.finish()
}

/// Append one position to a head-major KV cache: row `h` of `x`
/// (`[heads, hd]`, any float dtype) goes to cache row `h * cap + pos`, in
/// f16. `pos` is a runtime scalar bounded by the capacity, so the checker
/// proves the write lands inside the cache for every position.
pub fn kv_append(heads: usize, hd: usize, cap: usize, x_dtype: DType) -> Program {
    use Arg::Move;
    let mut b = Builder::new(&format!(
        "kv_append_{heads}x{hd}_cap{cap}_{}",
        x_dtype.suffix()
    ));
    let px = b.param("x", x_dtype, &[heads, hd], false);
    let pc = b.param("cache", DType::F16, &[heads * cap, hd], true);
    let pos = b.dyn_index("pos", cap - 1);
    for h in 0..heads {
        let r = b.op(
            "row",
            Op::Load(
                at(px, [IdxExpr::lit(h), IdxExpr::lit(0)], [1, hd]),
                reg(x_dtype, &[1, hd]),
            ),
        );
        let r = if x_dtype == DType::F16 {
            r
        } else {
            b.op("row16", Op::Convert(Move(r), DType::F16))
        };
        let dst = at(
            pc,
            [IdxExpr::lit(h * cap).plus(pos, 1), IdxExpr::lit(0)],
            [1, hd],
        );
        b.effect(Op::Store(Move(r), dst));
    }
    b.finish()
}

/// `y = silu(g) * u = g * sigmoid(g) * u`, in chunks of `chunk` per instance.
pub fn silu_mul(n: usize, chunk: usize) -> Result<Program, String> {
    use Arg::{Borrow, Move};
    if !n.is_multiple_of(chunk) {
        return Err(format!("silu_mul: chunk {chunk} must divide {n}"));
    }
    let mut b = Builder::new(&format!("silu_mul_{n}"));
    let pg = b.param("g", DType::F32, &[1, n], false);
    let pu = b.param("u", DType::F32, &[1, n], false);
    let py = b.param("y", DType::F32, &[1, n], true);
    let pid = b.grid(n / chunk);
    let v = |p| {
        at(
            p,
            [IdxExpr::lit(0), IdxExpr::scaled(pid, chunk, 0)],
            [1, chunk],
        )
    };
    let g = b.op("g", Op::Load(v(pg), reg(DType::F32, &[1, chunk])));
    let u = b.op("u", Op::Load(v(pu), reg(DType::F32, &[1, chunk])));
    let sg = b.op("sg", Op::Unary(UnOp::Sigmoid, Borrow(g)));
    let a = b.op("a", Op::Binary(BinOp::Mul, Move(g), Move(sg)));
    let y = b.op("y", Op::Binary(BinOp::Mul, Move(a), Move(u)));
    b.effect(Op::Store(Move(y), v(py)));
    Ok(b.finish())
}

/// A quantised matrix repacked for [`matvec_q`]: one `i8` per value (two per
/// byte when packed), and scales in the file's own form. Dequantising gives
/// bit-for-bit the floats ggml's reference dequantisation does: every
/// product that forms a scale or min is exact in f32, and it is evaluated
/// in the same order.
#[derive(Clone, Debug, PartialEq)]
pub struct Split {
    pub layout: QLayout,
    /// Row length in values. Needed to find a packed value's byte.
    pub cols: usize,
    /// For packed layouts, the chunk width values are paired across; the
    /// matvec that reads them must use it as its `kc`. 0 when unpacked.
    pub pack: usize,
    pub q: Vec<i8>,
    /// Two-level layouts: the small integer scale per group. Empty otherwise.
    pub sc: Vec<i8>,
    /// Q8_0: the f16 scale per group. Two-level: `d` per super-block.
    pub d: Vec<half::f16>,
    /// Affine layouts: min per group, and `dmin` per super-block.
    pub mn: Option<(Vec<i8>, Vec<half::f16>)>,
}

impl Split {
    /// The weight parameters after the values, in [`QLayout::scale_params`]
    /// order, as raw bytes for upload.
    pub fn scale_bytes(&self) -> Vec<Vec<u8>> {
        let i8s = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
        let f16s = |v: &[half::f16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        match self.layout.super_groups {
            None => vec![f16s(&self.d)],
            Some(_) => {
                let mut out = vec![i8s(&self.sc), f16s(&self.d)];
                if let Some((mn, dmin)) = &self.mn {
                    out.extend([i8s(mn), f16s(dmin)]);
                }
                out
            }
        }
    }

    /// Scale of group `g`: `d * sc` for two-level layouts.
    pub fn scale(&self, g: usize) -> f32 {
        match self.layout.super_groups {
            None => self.d[g].to_f32(),
            Some(sub) => self.sc[g] as f32 * self.d[g / sub].to_f32(),
        }
    }

    /// Min of group `g` (0 for non-affine layouts).
    pub fn min(&self, g: usize) -> f32 {
        match (&self.mn, self.layout.super_groups) {
            (Some((mn, dmin)), Some(sub)) => mn[g] as f32 * dmin[g / sub].to_f32(),
            _ => 0.0,
        }
    }

    /// Dequantise values `lo..hi` (row-major flat indices).
    pub fn dequant(&self, lo: usize, hi: usize) -> Vec<f32> {
        let g = self.layout.group;
        (lo..hi)
            .map(|i| self.value(i) * self.scale(i / g) - self.min(i / g))
            .collect()
    }

    /// Quantised value `i`, unpacked if the layout is packed.
    pub fn value(&self, i: usize) -> f32 {
        if self.layout.packed4 {
            let (row, col) = (i / self.cols, i % self.cols);
            let (chunk, w, h) = (col / self.pack, col % self.pack, self.pack / 2);
            let b = self.q[row * self.cols / 2 + chunk * h + w % h] as u8;
            (if w < h { b & 0xF } else { b >> 4 }) as f32
        } else {
            self.q[i] as f32
        }
    }

    /// Values as the kernels' `I8` tensor holds them (packed bytes when
    /// packed), for building interpreter tensors.
    pub fn q_f32(&self) -> Vec<f32> {
        self.q.iter().map(|&b| b as f32).collect()
    }

    /// The matvec's weight parameters (values, then scales in
    /// [`QLayout::scale_params`] order) as interpreter tensors, for `rows`
    /// rows.
    pub fn weight_tensors(&self, rows: usize) -> Vec<crate::Tensor> {
        use crate::Tensor;
        let qf = self.q_f32();
        let n_in = if self.layout.packed4 {
            2 * qf.len() / rows
        } else {
            qf.len() / rows
        };
        let mut out = vec![Tensor::new(DType::I8, &[rows, qf.len() / rows], &qf)];
        let f16s = |v: &[half::f16]| v.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
        let i8s = |v: &[i8]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
        let params = self.layout.scale_params();
        let data: Vec<Vec<f32>> = match self.layout.super_groups {
            None => vec![f16s(&self.d)],
            Some(_) => {
                let mut v = vec![i8s(&self.sc), f16s(&self.d)];
                if let Some((mn, dmin)) = &self.mn {
                    v.extend([i8s(mn), f16s(dmin)]);
                }
                v
            }
        };
        for ((_, dt, per), d) in params.into_iter().zip(data) {
            out.push(Tensor::new(dt, &[rows, n_in / per], &d));
        }
        out
    }

    /// Bytes the kernels read for this matrix.
    pub fn device_bytes(&self) -> usize {
        self.q.len()
            + self.sc.len()
            + 2 * self.d.len()
            + self.mn.as_ref().map_or(0, |(m, d)| m.len() + 2 * d.len())
    }
}

fn f16_raw(b: &[u8], i: usize) -> half::f16 {
    half::f16::from_le_bytes([b[i], b[i + 1]])
}

/// Q8_0 blocks (34 bytes: f16 scale, 32 × i8).
pub fn split_q8_0(blocks: &[u8]) -> Split {
    let (blocks, rest) = blocks.as_chunks::<34>();
    assert!(rest.is_empty(), "not whole Q8_0 blocks");
    let mut q = Vec::with_capacity(blocks.len() * 32);
    let mut d = Vec::with_capacity(blocks.len());
    for blk in blocks {
        d.push(f16_raw(blk, 0));
        q.extend(blk[2..].iter().map(|&b| b as i8));
    }
    Split {
        layout: QLayout::Q8_0,
        cols: 0,
        pack: 0,
        q,
        sc: vec![],
        d,
        mn: None,
    }
}

/// Q6_K super-blocks (210 bytes, 256 values): `ql[128]` low nibbles,
/// `qh[64]` high 2-bit pairs, 16 × i8 sub-block scales, f16 `d`.
/// Value `v` of a half is `d * sc[v / 16] * (q - 32)`.
pub fn split_q6_k(blocks: &[u8]) -> Split {
    let (blocks, rest) = blocks.as_chunks::<210>();
    assert!(rest.is_empty(), "not whole Q6_K blocks");
    let mut q = Vec::with_capacity(blocks.len() * 256);
    let mut sc = Vec::with_capacity(blocks.len() * 16);
    let mut d = Vec::with_capacity(blocks.len());
    for blk in blocks {
        d.push(f16_raw(blk, 208));
        for half in 0..2 {
            let ql = &blk[64 * half..64 * half + 64];
            let qh = &blk[128 + 32 * half..128 + 32 * half + 32];
            let mut vals = [0i8; 128];
            for l in 0..32 {
                let h = qh[l];
                vals[l] = ((ql[l] & 0xF) | ((h & 3) << 4)) as i8 - 32;
                vals[l + 32] = ((ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4)) as i8 - 32;
                vals[l + 64] = ((ql[l] >> 4) | (((h >> 4) & 3) << 4)) as i8 - 32;
                vals[l + 96] = ((ql[l + 32] >> 4) | (((h >> 6) & 3) << 4)) as i8 - 32;
            }
            q.extend_from_slice(&vals);
            sc.extend(
                blk[192 + 8 * half..192 + 8 * half + 8]
                    .iter()
                    .map(|&b| b as i8),
            );
        }
    }
    Split {
        layout: QLayout::Q6_K,
        cols: 0,
        pack: 0,
        q,
        sc,
        d,
        mn: None,
    }
}

/// Q4_K's 12 bytes of packed 6-bit (scale, min) pairs, sub-block `j`.
fn scale_min_k4(j: usize, sc: &[u8]) -> (u8, u8) {
    if j < 4 {
        (sc[j] & 63, sc[j + 4] & 63)
    } else {
        (
            (sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4),
            (sc[j + 4] >> 4) | ((sc[j] >> 6) << 4),
        )
    }
}

/// Q4_K super-blocks (144 bytes, 256 values): f16 `d`, f16 `dmin`, 12
/// bytes of 6-bit scales and mins, 128 bytes of nibbles. Sub-block `j`
/// (32 values) is `d * sc_j * q - dmin * m_j`; even sub-blocks take the low
/// nibbles of a 32-byte run, odd ones the high nibbles.
///
/// `cols` is the matrix's row length: values are repacked two per byte in
/// the pairing across chunks of `pack` that the matvec schedule expects (the
/// runtime uses whole rows: `pack == cols`).
pub fn split_q4_k(blocks: &[u8], cols: usize, pack: usize) -> Split {
    assert!(
        pack > 0 && pack.is_multiple_of(2) && cols.is_multiple_of(pack),
        "Q4_K rows of {cols} cannot be packed in chunks of {pack}"
    );
    let (blocks, rest) = blocks.as_chunks::<144>();
    assert!(rest.is_empty(), "not whole Q4_K blocks");
    let mut v = Vec::with_capacity(blocks.len() * 256);
    let mut sc = Vec::with_capacity(blocks.len() * 8);
    let mut mn = Vec::with_capacity(blocks.len() * 8);
    let mut d = Vec::with_capacity(blocks.len());
    let mut dmin = Vec::with_capacity(blocks.len());
    for blk in blocks {
        d.push(f16_raw(blk, 0));
        dmin.push(f16_raw(blk, 2));
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for j in 0..8 {
            let (scj, mj) = scale_min_k4(j, scales);
            sc.push(scj as i8);
            mn.push(mj as i8);
            let run = &qs[32 * (j / 2)..32 * (j / 2) + 32];
            v.extend(
                run.iter()
                    .map(|&b| if j % 2 == 0 { b & 0xF } else { b >> 4 }),
            );
        }
    }
    // Pair column k of each chunk with column k + chunk/2 in one byte.
    let h = pack / 2;
    let q = v
        .chunks(pack)
        .flat_map(|ch| (0..h).map(move |k| (ch[k] | (ch[k + h] << 4)) as i8))
        .collect();
    Split {
        layout: QLayout::Q4_K,
        cols,
        pack,
        q,
        sc,
        d,
        mn: Some((mn, dmin)),
    }
}

/// RoPE tables for one position in the layout [`rope`] takes.
/// `inv_freq[i] = base^(-2i/hd) / factor[i]`, as llama.cpp computes it.
pub fn rope_tables(
    pos: usize,
    hd: usize,
    base: f32,
    factors: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let mut cos = vec![0.0; hd];
    let mut sin = vec![0.0; hd];
    for i in 0..hd / 2 {
        let mut inv = (base as f64).powf(-((2 * i) as f64) / hd as f64);
        if let Some(f) = factors {
            inv /= f[i] as f64;
        }
        let ang = pos as f32 * inv as f32;
        let (s, c) = ang.sin_cos();
        cos[2 * i] = c;
        cos[2 * i + 1] = c;
        sin[2 * i] = -s;
        sin[2 * i + 1] = s;
    }
    (cos, sin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Tensor, check, run};
    use tile_ir::Target;
    use tile_ir::reference::{fill_pattern_f32, max_rel_err};

    fn pattern(n: usize, seed: u32) -> Vec<f32> {
        let mut x = vec![0.0; n];
        fill_pattern_f32(&mut x, seed);
        x
    }

    fn ok(p: &Program) {
        check(p, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{}: {e:#?}", p.name));
    }

    #[test]
    fn rmsnorm_matches_the_formula() {
        let n = 64;
        let p = rmsnorm(n, 1e-5);
        ok(&p);
        let (x, w) = (pattern(n, 1), pattern(n, 2));
        let mut t = vec![
            Tensor::new(DType::F32, &[1, n], &x),
            Tensor::new(DType::F32, &[1, n], &w),
            Tensor::zeros(DType::F32, &[1, n]),
        ];
        run(&p, &mut t).unwrap();
        let r = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / n as f32 + 1e-5).sqrt();
        let want: Vec<f32> = x.iter().zip(&w).map(|(a, b)| a * r * b).collect();
        assert!(max_rel_err(&t[2].data, &want) < 1e-6);
    }

    /// A random matrix in `layout`: values, and scales in the file's form.
    pub(crate) fn random_split(layout: QLayout, rows: usize, cols: usize, pack: usize) -> Split {
        let groups = rows * cols / layout.group;
        let supers = rows * cols / (layout.group * layout.super_groups.unwrap_or(1));
        let f16s = |n, seed| {
            pattern(n, seed)
                .iter()
                .map(|v| half::f16::from_f32(0.001 + v.abs() * 0.01))
                .collect()
        };
        let small = |n, seed| {
            pattern(n, seed)
                .iter()
                .map(|v| ((v + 1.0) * 31.5) as i8)
                .collect()
        };
        Split {
            layout,
            cols,
            pack,
            q: if layout.packed4 {
                pattern(rows * cols / 2, 4)
                    .iter()
                    .map(|v| ((v + 1.0) * 127.5) as u8 as i8)
                    .collect()
            } else {
                pattern(rows * cols, 4)
                    .iter()
                    .map(|v| (v * 31.0).round() as i8)
                    .collect()
            },
            sc: if layout.super_groups.is_some() {
                small(groups, 5)
            } else {
                vec![]
            },
            d: f16s(
                if layout.super_groups.is_some() {
                    supers
                } else {
                    groups
                },
                6,
            ),
            mn: layout.min.then(|| (small(groups, 7), f16s(supers, 8))),
        }
    }

    #[test]
    fn matvec_matches_dequantised_weights_in_every_layout() {
        // Two chunks of 512: a packed half-chunk must hold whole 256-value
        // super-blocks.
        let (n_in, n_out, kc) = (1024, 16, 512);
        for layout in [QLayout::Q8_0, QLayout::Q6_K, QLayout::Q4_K] {
            for residual in [false, true] {
                let p = matvec_q(n_in, n_out, 4, kc, layout, residual).unwrap();
                ok(&p);
                let sp = random_split(layout, n_out, n_in, if layout.packed4 { kc } else { 0 });
                let x = pattern(n_in, 3);
                let r = pattern(n_out, 9);
                let mut t = vec![Tensor::new(DType::F32, &[1, n_in], &x)];
                t.extend(sp.weight_tensors(n_out));
                if residual {
                    t.push(Tensor::new(DType::F32, &[1, n_out], &r));
                }
                t.push(Tensor::zeros(DType::F32, &[1, n_out]));
                run(&p, &mut t).unwrap();
                let w = sp.dequant(0, n_in * n_out);
                let want: Vec<f32> = (0..n_out)
                    .map(|o| {
                        let dot: f32 = (0..n_in).map(|i| w[o * n_in + i] * x[i]).sum();
                        dot + if residual { r[o] } else { 0.0 }
                    })
                    .collect();
                let got = &t.last().unwrap().data;
                // 1024 terms summed in two chunks vs. one pass: last-bit
                // differences, not a wrong answer.
                let err = max_rel_err(got, &want);
                assert!(
                    err < 1e-4,
                    "{layout:?} residual {residual}: max rel err {err:e}, got {:?}, want {:?}",
                    &got[..3],
                    &want[..3]
                );
            }
        }
    }

    #[test]
    fn i8_cannot_be_computed_with_directly() {
        let mut b = Builder::new("t");
        let pq = b.param("q", DType::I8, &[1, 32], false);
        let q = b.op("q", Op::Load(row(pq, 32), reg(DType::I8, &[1, 32])));
        let y = b.op("y", Op::Exp(Arg::Move(q)));
        b.drop(y);
        let errs = check(&b.finish(), &Target::apple_m_series()).unwrap_err();
        assert!(errs[0].msg.contains("dequant"), "{errs:?}");
    }

    #[test]
    fn rope_rotates_adjacent_pairs() {
        let (h, hd) = (2, 8);
        let p = rope(h, hd, DType::F32);
        ok(&p);
        let x = pattern(h * hd, 7);
        let (c, s) = rope_tables(5, hd, 500000.0, None);
        let mut t = vec![
            Tensor::new(DType::F32, &[h, hd], &x),
            Tensor::new(DType::F32, &[1, hd], &c),
            Tensor::new(DType::F32, &[1, hd], &s),
            Tensor::zeros(DType::F32, &[h, hd]),
        ];
        run(&p, &mut t).unwrap();
        let mut want = vec![0.0; h * hd];
        for r in 0..h {
            for i in 0..hd / 2 {
                let (x0, x1) = (x[r * hd + 2 * i], x[r * hd + 2 * i + 1]);
                let (cs, sn) = (c[2 * i], s[2 * i + 1]);
                want[r * hd + 2 * i] = x0 * cs - x1 * sn;
                want[r * hd + 2 * i + 1] = x0 * sn + x1 * cs;
            }
        }
        assert!(max_rel_err(&t[3].data, &want) < 1e-6);
    }

    #[test]
    fn silu_mul_matches_the_formula() {
        let n = 64;
        let p = silu_mul(n, 16).unwrap();
        ok(&p);
        let (g, u) = (pattern(n, 8), pattern(n, 9));
        let mut t = vec![
            Tensor::new(DType::F32, &[1, n], &g),
            Tensor::new(DType::F32, &[1, n], &u),
            Tensor::zeros(DType::F32, &[1, n]),
        ];
        run(&p, &mut t).unwrap();
        let want: Vec<f32> = g
            .iter()
            .zip(&u)
            .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        assert!(max_rel_err(&t[2].data, &want) < 1e-6);
    }

    #[test]
    fn q8_0_blocks_split_losslessly() {
        let mut blk = vec![0u8; 68];
        blk[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        blk[2] = 0xFF; // -1
        blk[34..36].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        blk[36] = 3;
        let sp = split_q8_0(&blk);
        assert_eq!((sp.q.len(), sp.d.len()), (64, 2));
        assert_eq!((sp.q[0], sp.q[32]), (-1, 3));
        assert_eq!((sp.scale(0), sp.scale(1)), (0.5, 2.0));
        assert_eq!(sp.dequant(0, 1), vec![-0.5]);
    }

    /// Random bytes with sane f16 fields at `f16s`.
    fn blocks(n: usize, size: usize, f16s: &[usize], seed: u32) -> Vec<u8> {
        let mut x = pattern(n * size, seed);
        let mut b: Vec<u8> = x.iter_mut().map(|v| ((*v + 1.0) * 127.5) as u8).collect();
        for i in 0..n {
            for &o in f16s {
                let v = half::f16::from_f32(0.01 + 0.001 * ((i + o) % 7) as f32);
                b[i * size + o..i * size + o + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        b
    }

    /// ggml's `dequantize_row_q4_K`, transcribed loop for loop.
    fn ggml_q4_k(blk: &[u8]) -> Vec<f32> {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let (sc, mut q) = (&blk[4..16], &blk[16..]);
        let mut y = vec![];
        let mut is = 0;
        for _ in 0..4 {
            let (s1, m1) = scale_min_k4(is, sc);
            let (d1, m1) = (d * s1 as f32, dmin * m1 as f32);
            let (s2, m2) = scale_min_k4(is + 1, sc);
            let (d2, m2) = (d * s2 as f32, dmin * m2 as f32);
            y.extend(q[..32].iter().map(|&b| d1 * (b & 0xF) as f32 - m1));
            y.extend(q[..32].iter().map(|&b| d2 * (b >> 4) as f32 - m2));
            q = &q[32..];
            is += 2;
        }
        y
    }

    /// ggml's `dequantize_row_q6_K`, transcribed loop for loop.
    fn ggml_q6_k(blk: &[u8]) -> Vec<f32> {
        let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
        let mut y = vec![0.0; 256];
        for n in 0..2 {
            let (ql, qh) = (&blk[64 * n..], &blk[128 + 32 * n..]);
            let sc = &blk[192 + 8 * n..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                let s = |k: usize| d * sc[is + k] as i8 as f32;
                y[128 * n + l] = s(0) * q1 as f32;
                y[128 * n + l + 32] = s(2) * q2 as f32;
                y[128 * n + l + 64] = s(4) * q3 as f32;
                y[128 * n + l + 96] = s(6) * q4 as f32;
            }
        }
        y
    }

    #[test]
    fn q4_k_split_dequantises_exactly_like_ggml() {
        let raw = blocks(3, 144, &[0, 2], 21);
        let sp = split_q4_k(&raw, 256, 256);
        let want: Vec<f32> = raw.chunks(144).flat_map(ggml_q4_k).collect();
        assert_eq!(sp.dequant(0, 3 * 256), want, "must be bit-identical");
    }

    #[test]
    fn q6_k_split_dequantises_exactly_like_ggml() {
        let raw = blocks(3, 210, &[208], 22);
        let sp = split_q6_k(&raw);
        let want: Vec<f32> = raw.chunks(210).flat_map(ggml_q6_k).collect();
        assert_eq!(sp.dequant(0, 3 * 256), want, "must be bit-identical");
    }
}
