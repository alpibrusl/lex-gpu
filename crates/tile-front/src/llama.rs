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

use crate::ir::{
    Arg, BinOp, Builder, IdxExpr, Nibbles, Op, Program, Reduce, TileTy, Ty, UnOp, Var, View,
};

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
    /// Unsigned 4-bit values, two per byte: byte `k` of a row holds columns
    /// `2k` (low nibble) and `2k + 1` (high nibble).
    pub packed4: bool,
    /// 6-bit values in two bit planes (see [`Op::Dequant6`]): 0.75 bytes per
    /// value, as the file stores them.
    pub six: bool,
    /// NVFP4 (see [`Op::DequantFp4`]): 4-bit E2M1 values two per byte, an
    /// FP8 E4M3 scale per group, and one f32 scale for the whole tensor,
    /// held per row. 4.5 bits per weight, what Qwen3.5's MLX build uses.
    pub fp4: bool,
}

impl QLayout {
    /// Q8_0: 32 values, one f16 scale.
    pub const Q8_0: QLayout = QLayout {
        group: 32,
        super_groups: None,
        min: false,
        packed4: false,
        six: false,
        fp4: false,
    };
    /// Q6_K: 16 values per `i8` sub-block scale, 16 sub-blocks per f16 `d`;
    /// 6-bit values in a 4-bit and a 2-bit plane, centred at 0.
    pub const Q6_K: QLayout = QLayout {
        group: 16,
        super_groups: Some(16),
        min: false,
        packed4: false,
        six: true,
        fp4: false,
    };
    /// Q4_K: 32 values per 6-bit scale and min, 8 per f16 `d` and `dmin`:
    /// `d * sc * q - dmin * m`, with the 4-bit values packed two per byte.
    pub const Q4_K: QLayout = QLayout {
        group: 32,
        super_groups: Some(8),
        min: true,
        packed4: true,
        six: false,
        fp4: false,
    };

    /// NVFP4: 16 values, an FP8 E4M3 scale each, one f32 per tensor.
    pub const NVFP4: QLayout = QLayout {
        group: 16,
        super_groups: None,
        min: false,
        packed4: true,
        six: false,
        fp4: true,
    };

    fn tag(self) -> String {
        format!(
            "g{}{}{}{}{}",
            self.group,
            self.super_groups.map_or(String::new(), |s| format!("s{s}")),
            if self.min { "m" } else { "" },
            if self.packed4 { "p4" } else { "" },
            if self.six { "q6" } else { "" }
        )
        .replace("g16p4", "nvfp4")
    }

    /// Weight parameters after the values, in order, with their dtypes and
    /// widths as a divisor of the row length.
    fn scale_params(self) -> Vec<(&'static str, DType, usize)> {
        let g = self.group;
        if self.fp4 {
            // The FP8 scales; the tensor's own f32 scale is a separate
            // per-row parameter (see `QParams::declare`).
            return vec![("ws", DType::I8, g)];
        }
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
    matmul_q(1, n_in, n_out, bo, kc, layout, residual)
}

/// A quantised matrix's parameters: values, the 6-bit layouts' high plane,
/// and the scale parameters in [`QLayout::scale_params`] order.
struct QParams {
    q: usize,
    qh: Option<usize>,
    scales: Vec<(usize, DType, usize)>,
    /// NVFP4's per-tensor f32 scale, held once per row: `[n_out]`.
    gs: Option<usize>,
}

impl QParams {
    /// Declare `[n_out, n_in]` in `layout`, parameter names prefixed.
    fn declare(b: &mut Builder, pre: &str, n_in: usize, n_out: usize, layout: QLayout) -> QParams {
        let qcols = if layout.packed4 || layout.six {
            n_in / 2
        } else {
            n_in
        };
        let q = b.param(&format!("{pre}q"), DType::I8, &[n_out, qcols], false);
        let qh = layout
            .six
            .then(|| b.param(&format!("{pre}qh"), DType::I8, &[n_out, n_in / 4], false));
        let scales = layout
            .scale_params()
            .into_iter()
            .map(|(name, dt, per)| {
                // `scale_params` names start with `w`: keep the rest.
                let name = format!("{pre}{}", name.strip_prefix('w').unwrap_or(name));
                (b.param(&name, dt, &[n_out, n_in / per], false), dt, per)
            })
            .collect();
        let gs = layout
            .fp4
            .then(|| b.param(&format!("{pre}gs"), DType::F32, &[n_out], false));
        QParams { q, qh, scales, gs }
    }

    /// The dequantised `[bo, kc]` tile of rows `rows..rows+bo`, chunk `c`.
    fn tile(
        &self,
        b: &mut Builder,
        layout: QLayout,
        rows: &IdxExpr,
        c: Var,
        (bo, kc): (usize, usize),
    ) -> Var {
        use Arg::Move;
        let g = layout.group;
        let load = |b: &mut Builder, (p, dt, per): (usize, DType, usize)| {
            b.op(
                "w",
                Op::Load(
                    at(
                        p,
                        [rows.clone(), IdxExpr::scaled(c, kc / per, 0)],
                        [bo, kc / per],
                    ),
                    reg(dt, &[bo, kc / per]),
                ),
            )
        };
        if let Some(gs) = self.gs {
            // NVFP4: E2M1 codes two per byte, an E4M3 scale per 16, and
            // this row's copy of the tensor's f32 scale.
            let q = load(b, (self.q, DType::I8, 2));
            let sc = load(b, (self.scales[0].0, DType::I8, g));
            let row = b.op(
                "gs",
                Op::Load(
                    View {
                        param: gs,
                        offset: vec![rows.clone()],
                        shape: vec![bo],
                    },
                    reg(DType::F32, &[bo]),
                ),
            );
            return b.op("w", Op::DequantFp4(Move(q), Move(sc), Move(row), g));
        }
        // Per-group scales (and mins): f16 directly, or `sc * d` through
        // `dequant` for the two-level K-quants.
        let (s, m) = match layout.super_groups {
            None => (load(b, self.scales[0]), None),
            Some(sub) => {
                let sc = load(b, self.scales[0]);
                let d = load(b, self.scales[1]);
                let s = b.op("s", Op::Dequant(Move(sc), Move(d), None, sub));
                let m = layout.min.then(|| {
                    let mn = load(b, self.scales[2]);
                    let dmin = load(b, self.scales[3]);
                    Move(b.op("m", Op::Dequant(Move(mn), Move(dmin), None, sub)))
                });
                (s, m)
            }
        };
        if let Some(qh) = self.qh {
            // Two bit planes: 4 low bits two per byte, 2 high bits four per
            // byte.
            let lo = load(b, (self.q, DType::I8, 2));
            let hi = load(b, (qh, DType::I8, 4));
            b.op("w", Op::Dequant6(Move(lo), Move(hi), Move(s), g))
        } else if layout.packed4 {
            // Two values per byte: one load feeds both.
            let q = load(b, (self.q, DType::I8, 2));
            b.op("w", Op::Dequant4(Move(q), Move(s), m, g, Nibbles::Pairs))
        } else {
            let q = load(b, (self.q, DType::I8, 1));
            b.op("w", Op::Dequant(Move(q), Move(s), m, g))
        }
    }
}

fn check_shape(
    n_in: usize,
    n_out: usize,
    bo: usize,
    kc: usize,
    layout: QLayout,
) -> Result<(), String> {
    let sup = layout.group * layout.super_groups.unwrap_or(1);
    if !n_out.is_multiple_of(bo) || !n_in.is_multiple_of(kc) || !kc.is_multiple_of(sup) {
        return Err(format!(
            "matvec {n_out}x{n_in}: bo {bo} must divide the rows, kc {kc} the columns, and \
             each chunk must be whole super-blocks of {sup}"
        ));
    }
    Ok(())
}

fn matmul_tag(t: usize) -> String {
    if t == 1 {
        "matvec".to_string()
    } else {
        format!("matmul{t}")
    }
}

/// [`matvec_q`] for `t` rows of activations at once (`x: [t, n_in]`,
/// `y: [t, n_out]`): prefill's shape, where each weight is read once for
/// every token of the batch.
pub fn matmul_q(
    t: usize,
    n_in: usize,
    n_out: usize,
    bo: usize,
    kc: usize,
    layout: QLayout,
    residual: bool,
) -> Result<Program, String> {
    use Arg::Move;
    check_shape(n_in, n_out, bo, kc, layout)?;
    let name = format!(
        "{}_{}_{n_out}x{n_in}{}",
        matmul_tag(t),
        layout.tag(),
        if residual { "_res" } else { "" }
    );
    let mut b = Builder::new(&name);
    let px = b.param("x", DType::F32, &[t, n_in], false);
    let w = QParams::declare(&mut b, "w", n_in, n_out, layout);
    let pr = residual.then(|| b.param("r", DType::F32, &[t, n_out], false));
    let py = b.param("y", DType::F32, &[t, n_out], true);
    let pid = b.grid(n_out / bo);

    let acc_ty = reg(DType::F32, &[t, bo]);
    let acc = b.op("acc", Op::Fill(acc_ty.clone(), 0.0));
    let out = b.for_range(
        0,
        n_in / kc,
        vec![acc],
        vec![Ty::Tile(acc_ty)],
        |b, c, p| {
            let rows = IdxExpr::scaled(pid, bo, 0);
            let x = x_chunk(b, px, c, (t, kc));
            let wt = w.tile(b, layout, &rows, c, (bo, kc));
            let part = b.op("part", Op::MatMulNT(Move(x), Move(wt), DType::F32));
            vec![b.op("acc", Op::Binary(BinOp::Add, Move(p[0]), Move(part)))]
        },
    );
    let mut y = out[0];
    let dst = at(py, [IdxExpr::lit(0), IdxExpr::scaled(pid, bo, 0)], [t, bo]);
    if let Some(pr) = pr {
        let r = b.op(
            "r",
            Op::Load(
                at(pr, [IdxExpr::lit(0), IdxExpr::scaled(pid, bo, 0)], [t, bo]),
                reg(DType::F32, &[t, bo]),
            ),
        );
        y = b.op("y", Op::Binary(BinOp::Add, Move(y), Move(r)));
    }
    b.effect(Op::Store(Move(y), dst));
    Ok(b.finish())
}

/// RMSNorm folded into a matvec: `W (g ⊙ x) · rsqrt(mean(x²) + eps)` is
/// `W · rmsnorm(x, g)` with the scalar applied to the `bo` outputs instead
/// of the `n_in` inputs. Every threadgroup forms the scalar itself from x
/// (16 KB, cache-resident), so no separate norm dispatch, and no barrier
/// the whole GPU waits on, sits between the residual and the matvec.
/// Returns the `g ⊙ x` tile to multiply and the `[1]` scale. Decode only
/// (one row): `g` is `[1, n]` and multiplies `x` elementwise.
fn normed_x(b: &mut Builder, px: usize, pg: usize, c: Var, n: usize, eps: f32) -> (Var, Var) {
    use Arg::{Borrow, Move};
    let x = x_chunk(b, px, c, (1, n));
    let sq = b.op("sq", Op::Binary(BinOp::Mul, Borrow(x), Borrow(x)));
    let ss = b.op("ss", Op::RowReduce(Reduce::Sum, Move(sq)));
    let ms = b.op("ms", Op::Scale(Move(ss), 1.0 / n as f32));
    let e = b.op("eps", Op::Fill(reg(DType::F32, &[1]), eps));
    let t = b.op("t", Op::Binary(BinOp::Add, Move(ms), Move(e)));
    let r = b.op("r", Op::Unary(UnOp::Rsqrt, Move(t)));
    let g = b.op("g", Op::Load(row(pg, n), reg(DType::F32, &[1, n])));
    let xg = b.op("xg", Op::Binary(BinOp::Mul, Move(x), Move(g)));
    (xg, r)
}

/// [`matvec_q`] of `rmsnorm(x, g)` with the norm folded in (see
/// [`normed_x`]). Parameters: `x, g, <weights>, y`.
pub fn matvec_q_rms(
    n_in: usize,
    n_out: usize,
    bo: usize,
    layout: QLayout,
    eps: f32,
) -> Result<Program, String> {
    use Arg::Move;
    check_shape(n_in, n_out, bo, n_in, layout)?;
    let mut b = Builder::new(&format!("matvec_rms_{}_{n_out}x{n_in}", layout.tag()));
    let px = b.param("x", DType::F32, &[1, n_in], false);
    let pg = b.param("gn", DType::F32, &[1, n_in], false);
    let w = QParams::declare(&mut b, "w", n_in, n_out, layout);
    let py = b.param("y", DType::F32, &[1, n_out], true);
    let pid = b.grid(n_out / bo);
    let rows = IdxExpr::scaled(pid, bo, 0);
    b.for_range(0, 1, vec![], vec![], |b, c, _| {
        let (xg, r) = normed_x(b, px, pg, c, n_in, eps);
        let wt = w.tile(b, layout, &rows, c, (bo, n_in));
        let y = b.op("y", Op::MatMulNT(Move(xg), Move(wt), DType::F32));
        let y = b.op("y", Op::Binary(BinOp::Mul, Move(y), Move(r)));
        b.effect(Op::Store(
            Move(y),
            at(py, [IdxExpr::lit(0), rows.clone()], [1, bo]),
        ));
        vec![]
    });
    Ok(b.finish())
}

/// The activations `x[.., chunk c]`, as a lazy `[t, kc]` tile.
fn x_chunk(b: &mut Builder, px: usize, c: Var, (t, kc): (usize, usize)) -> Var {
    // A batch re-reads these activations for every block of weight rows a
    // threadgroup owns, and replacing the loads with a constant shows they
    // cost about half the kernel's throughput at four tokens. Staging them
    // in threadgroup memory is not the fix: it needs a chunked reduction,
    // and 5120 -> 17408 at four tokens measured 92 GB/s with chunks of
    // 1024 against 141 reading the whole row straight from device memory.
    b.op(
        "x",
        Op::Load(
            at(px, [IdxExpr::lit(0), IdxExpr::scaled(c, kc, 0)], [t, kc]),
            reg(DType::F32, &[t, kc]),
        ),
    )
}

/// The feed-forward's gated input in one kernel:
/// `a = silu(W_gate x) * (W_up x)` for `t` rows of `x`, both matrices
/// `[n_out, n_in]` in `layout`. Replaces two matmuls and `silu_mul`: one
/// dispatch instead of three, and `g`, `u` never reach memory.
/// Parameters: `x, <gate weights>, <up weights>, a`.
///
/// With `norm = Some(eps)` (decode only, `t == 1`), `x` is the residual
/// and the FFN's RMSNorm is folded in (see [`normed_x`]); parameters are
/// then `x, g, <gate>, <up>, a`.
pub fn matmul_q_glu(
    t: usize,
    n_in: usize,
    n_out: usize,
    bo: usize,
    layout: QLayout,
    norm: Option<f32>,
) -> Result<Program, String> {
    use Arg::{Borrow, Move};
    let kc = n_in;
    check_shape(n_in, n_out, bo, kc, layout)?;
    if norm.is_some() && t != 1 {
        return Err("a folded norm is for one row".into());
    }
    let mut b = Builder::new(&format!(
        "{}_glu{}_{}_{n_out}x{n_in}",
        matmul_tag(t),
        if norm.is_some() { "_rms" } else { "" },
        layout.tag()
    ));
    let px = b.param("x", DType::F32, &[t, n_in], false);
    let pg = norm.map(|_| b.param("gn", DType::F32, &[1, n_in], false));
    let wg = QParams::declare(&mut b, "g", n_in, n_out, layout);
    let wu = QParams::declare(&mut b, "u", n_in, n_out, layout);
    let pa = b.param("a", DType::F32, &[t, n_out], true);
    let pid = b.grid(n_out / bo);
    let rows = IdxExpr::scaled(pid, bo, 0);
    // One chunk, the whole row (`kc == n_in`).
    b.for_range(0, 1, vec![], vec![], |b, c, _| {
        let (x, r) = match (norm, pg) {
            (Some(eps), Some(pg)) => {
                let (xg, r) = normed_x(b, px, pg, c, n_in, eps);
                (xg, Some(r))
            }
            _ => (x_chunk(b, px, c, (t, kc)), None),
        };
        let gt = wg.tile(b, layout, &rows, c, (bo, kc));
        let mut g = b.op("g", Op::MatMulNT(Borrow(x), Move(gt), DType::F32));
        let ut = wu.tile(b, layout, &rows, c, (bo, kc));
        let mut u = b.op("u", Op::MatMulNT(Move(x), Move(ut), DType::F32));
        if let Some(r) = r {
            g = b.op("g", Op::Binary(BinOp::Mul, Move(g), Borrow(r)));
            u = b.op("u", Op::Binary(BinOp::Mul, Move(u), Move(r)));
        }
        let sg = b.op("sg", Op::Unary(UnOp::Sigmoid, Borrow(g)));
        let sl = b.op("silu", Op::Binary(BinOp::Mul, Move(g), Move(sg)));
        let a = b.op("a", Op::Binary(BinOp::Mul, Move(sl), Move(u)));
        b.effect(Op::Store(
            Move(a),
            at(pa, [IdxExpr::lit(0), rows.clone()], [t, bo]),
        ));
        vec![]
    });
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

/// RMSNorm over `rows` rows of `n` (`x, w: [1, n], y`), one grid instance
/// per row. With `pick = Some(r)`, only row `r` is normalised, into a
/// `[1, n]` output: prefill's last row, for the output head.
pub fn rmsnorm_rows(rows: usize, n: usize, eps: f32, pick: Option<usize>) -> Program {
    use Arg::{Borrow, Move};
    let name = match pick {
        Some(r) => format!("rmsnorm_row{r}_of{rows}_{n}"),
        None => format!("rmsnorm_{rows}x{n}"),
    };
    let mut b = Builder::new(&name);
    let px = b.param("x", DType::F32, &[rows, n], false);
    let pw = b.param("w", DType::F32, &[1, n], false);
    let out_rows = if pick.is_some() { 1 } else { rows };
    let py = b.param("y", DType::F32, &[out_rows, n], true);
    let (src, dst) = match pick {
        Some(r) => (IdxExpr::lit(r), IdxExpr::lit(0)),
        None => {
            let pid = b.grid(rows);
            (IdxExpr::scaled(pid, 1, 0), IdxExpr::scaled(pid, 1, 0))
        }
    };
    let x = b.op(
        "x",
        Op::Load(
            at(px, [src, IdxExpr::lit(0)], [1, n]),
            reg(DType::F32, &[1, n]),
        ),
    );
    let sq = b.op("sq", Op::Binary(BinOp::Mul, Borrow(x), Borrow(x)));
    let ss = b.op("ss", Op::RowReduce(Reduce::Sum, Move(sq)));
    let ms = b.op("ms", Op::Scale(Move(ss), 1.0 / n as f32));
    let e = b.op("eps", Op::Fill(reg(DType::F32, &[1]), eps));
    let t = b.op("t", Op::Binary(BinOp::Add, Move(ms), Move(e)));
    let r = b.op("r", Op::Unary(UnOp::Rsqrt, Move(t)));
    let xn = b.op("xn", Op::Binary(BinOp::Mul, Move(x), Move(r)));
    let w = b.op("w", Op::Load(row(pw, n), reg(DType::F32, &[1, n])));
    let y = b.op("y", Op::Binary(BinOp::Mul, Move(xn), Move(w)));
    b.effect(Op::Store(Move(y), at(py, [dst, IdxExpr::lit(0)], [1, n])));
    b.finish()
}

/// RoPE for `tokens` tokens of `heads` heads each (`x: [tokens*heads, hd]`),
/// with per-token tables `cos, sin: [tokens, hd]`. One grid instance per
/// token; output in the same order, narrowed to `out`.
pub fn rope_rows(tokens: usize, heads: usize, hd: usize, out: DType) -> Program {
    use Arg::{Borrow, Move};
    let mut b = Builder::new(&format!("rope_{tokens}x{heads}x{hd}_{}", out.suffix()));
    let px = b.param("x", DType::F32, &[tokens * heads, hd], false);
    let pc = b.param("cos", DType::F32, &[tokens, hd], false);
    let ps = b.param("sin", DType::F32, &[tokens, hd], false);
    let py = b.param("y", out, &[tokens * heads, hd], true);
    let pid = b.grid(tokens);
    let block = [IdxExpr::scaled(pid, heads, 0), IdxExpr::lit(0)];
    let table = [IdxExpr::scaled(pid, 1, 0), IdxExpr::lit(0)];
    let x = b.op(
        "x",
        Op::Load(
            at(px, block.clone(), [heads, hd]),
            reg(DType::F32, &[heads, hd]),
        ),
    );
    let c = b.op(
        "cos",
        Op::Load(at(pc, table.clone(), [1, hd]), reg(DType::F32, &[1, hd])),
    );
    let s = b.op(
        "sin",
        Op::Load(at(ps, table, [1, hd]), reg(DType::F32, &[1, hd])),
    );
    let sw = b.op("sw", Op::SwapPairs(Borrow(x)));
    let a = b.op("a", Op::Binary(BinOp::Mul, Move(x), Move(c)));
    let z = b.op("z", Op::Binary(BinOp::Mul, Move(sw), Move(s)));
    let y = b.op("y", Op::Binary(BinOp::Add, Move(a), Move(z)));
    let y = if out == DType::F32 {
        y
    } else {
        b.op("y", Op::Convert(Move(y), out))
    };
    b.effect(Op::Store(Move(y), at(py, block, [heads, hd])));
    b.finish()
}

/// Append `tokens` positions to a head-major KV cache: row `t * heads + h`
/// of `x` goes to cache row `h * cap + pos0 + t`, in f16. `pos0` is a
/// runtime scalar bounded so the last write stays inside the cache.
pub fn kv_append_rows(
    tokens: usize,
    heads: usize,
    hd: usize,
    cap: usize,
    x_dtype: DType,
) -> Program {
    use Arg::Move;
    let mut b = Builder::new(&format!(
        "kv_append_{tokens}x{heads}x{hd}_cap{cap}_{}",
        x_dtype.suffix()
    ));
    let px = b.param("x", x_dtype, &[tokens * heads, hd], false);
    let pc = b.param("cache", DType::F16, &[heads * cap, hd], true);
    let pos0 = b.dyn_index("pos0", cap - tokens);
    for t in 0..tokens {
        for h in 0..heads {
            let r = b.op(
                "row",
                Op::Load(
                    at(px, [IdxExpr::lit(t * heads + h), IdxExpr::lit(0)], [1, hd]),
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
                [IdxExpr::lit(h * cap + t).plus(pos0, 1), IdxExpr::lit(0)],
                [1, hd],
            );
            b.effect(Op::Store(Move(r), dst));
        }
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
    pub q: Vec<i8>,
    /// Six-bit layouts: the 2-bit plane, four values per byte. Empty
    /// otherwise (`q` then holds the 4-bit plane).
    pub qh: Vec<i8>,
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
        if self.layout.six {
            let lo = self.q[i / 2] as u8;
            let l = if i.is_multiple_of(2) {
                lo & 0xF
            } else {
                lo >> 4
            };
            let h = (self.qh[i / 4] as u8 >> (2 * (i % 4))) & 3;
            ((l | (h << 4)) as i32 - 32) as f32
        } else if self.layout.packed4 {
            let b = self.q[i / 2] as u8;
            (if i.is_multiple_of(2) { b & 0xF } else { b >> 4 }) as f32
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
        let n_in = if self.layout.packed4 || self.layout.six {
            2 * qf.len() / rows
        } else {
            qf.len() / rows
        };
        let mut out = vec![Tensor::new(DType::I8, &[rows, qf.len() / rows], &qf)];
        if self.layout.six {
            let hf: Vec<f32> = self.qh.iter().map(|&b| b as f32).collect();
            out.push(Tensor::new(DType::I8, &[rows, hf.len() / rows], &hf));
        }
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
            + self.qh.len()
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
        q,
        qh: vec![],
        sc: vec![],
        d,
        mn: None,
    }
}

/// Q6_K super-blocks (210 bytes, 256 values): `ql[128]` low nibbles,
/// `qh[64]` high 2-bit pairs, 16 × i8 sub-block scales, f16 `d`.
/// Value `v` of a half is `d * sc[v / 16] * (q - 32)`. Repacked into two
/// bit planes in column order, 0.75 bytes per value like the file.
pub fn split_q6_k(blocks: &[u8]) -> Split {
    let (blocks, rest) = blocks.as_chunks::<210>();
    assert!(rest.is_empty(), "not whole Q6_K blocks");
    let mut v = Vec::with_capacity(blocks.len() * 256);
    let mut sc = Vec::with_capacity(blocks.len() * 16);
    let mut d = Vec::with_capacity(blocks.len());
    for blk in blocks {
        d.push(f16_raw(blk, 208));
        for half in 0..2 {
            let ql = &blk[64 * half..64 * half + 64];
            let qh = &blk[128 + 32 * half..128 + 32 * half + 32];
            // Unsigned 6-bit values, in column order.
            let mut vals = [0u8; 128];
            for l in 0..32 {
                let h = qh[l];
                vals[l] = (ql[l] & 0xF) | ((h & 3) << 4);
                vals[l + 32] = (ql[l + 32] & 0xF) | (((h >> 2) & 3) << 4);
                vals[l + 64] = (ql[l] >> 4) | (((h >> 4) & 3) << 4);
                vals[l + 96] = (ql[l + 32] >> 4) | (((h >> 6) & 3) << 4);
            }
            v.extend_from_slice(&vals);
            sc.extend(
                blk[192 + 8 * half..192 + 8 * half + 8]
                    .iter()
                    .map(|&b| b as i8),
            );
        }
    }
    // Two planes: the low 4 bits two per byte, the high 2 bits four per byte.
    let q = v
        .chunks(2)
        .map(|p| ((p[0] & 0xF) | ((p[1] & 0xF) << 4)) as i8)
        .collect();
    let qh = v
        .chunks(4)
        .map(|p| ((p[0] >> 4) | ((p[1] >> 4) << 2) | ((p[2] >> 4) << 4) | ((p[3] >> 4) << 6)) as i8)
        .collect();
    Split {
        layout: QLayout::Q6_K,
        cols: 0,
        q,
        qh,
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
/// Values are repacked two per byte in plain order (`2k` low, `2k + 1`
/// high); `cols` is the matrix's row length.
pub fn split_q4_k(blocks: &[u8], cols: usize) -> Split {
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
    let q = v.chunks(2).map(|p| (p[0] | (p[1] << 4)) as i8).collect();
    Split {
        layout: QLayout::Q4_K,
        cols,
        q,
        qh: vec![],
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
    pub(crate) fn random_split(layout: QLayout, rows: usize, cols: usize) -> Split {
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
            qh: if layout.six {
                pattern(rows * cols / 4, 10)
                    .iter()
                    .map(|v| ((v + 1.0) * 127.5) as u8 as i8)
                    .collect()
            } else {
                vec![]
            },
            q: if layout.packed4 || layout.six {
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
        // Two chunks of whole 256-value super-blocks.
        let (n_in, n_out, kc) = (1024, 16, 512);
        for layout in [QLayout::Q8_0, QLayout::Q6_K, QLayout::Q4_K] {
            for residual in [false, true] {
                let p = matvec_q(n_in, n_out, 4, kc, layout, residual).unwrap();
                ok(&p);
                let sp = random_split(layout, n_out, n_in);
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
    fn nvfp4_codes_and_fp8_scales_decode_to_their_spec() {
        use crate::interp::{e2m1, e4m3};
        // E2M1: sign, 2-bit exponent, 1-bit mantissa.
        let mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (c, &want) in mags.iter().enumerate() {
            assert_eq!(e2m1(c as u8), want, "code {c}");
            assert_eq!(e2m1(c as u8 | 8), -want, "code {c} negated");
        }
        // E4M3 (OCP e4m3fn): 1.0, 1.5, the smallest subnormal, the largest
        // finite value, and the one NaN encoding.
        assert_eq!(e4m3(0x38), 1.0);
        assert_eq!(e4m3(0x3C), 1.5);
        assert_eq!(e4m3(0x01), 2.0f32.powi(-9));
        assert_eq!(e4m3(0x7E), 448.0);
        assert_eq!(e4m3(0xB8), -1.0);
        assert!(e4m3(0x7F).is_nan());
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
        let sp = split_q4_k(&raw, 256);
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
