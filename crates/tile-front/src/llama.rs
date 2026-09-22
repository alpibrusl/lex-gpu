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

use crate::ir::{Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, TileTy, Ty, UnOp, View};

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

/// How a quantised matrix is laid out for the kernels: `I8` values, one f32
/// scale (and, for affine formats, one f32 min) per `group` values along a
/// row. Every GGUF block format the loader supports repacks into this.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct QLayout {
    pub group: usize,
    pub min: bool,
}

impl QLayout {
    /// Q8_0: 32 values, one scale.
    pub const Q8_0: QLayout = QLayout {
        group: 32,
        min: false,
    };
    /// Q6_K: 16 values per sub-block scale (`d * sc`), values centred at 0.
    pub const Q6_K: QLayout = QLayout {
        group: 16,
        min: false,
    };
    /// Q4_K: 32 values per sub-block, `d * sc * q - dmin * m`.
    pub const Q4_K: QLayout = QLayout {
        group: 32,
        min: true,
    };

    fn tag(self) -> String {
        format!("g{}{}", self.group, if self.min { "m" } else { "" })
    }
}

/// `y = W x (+ r)` for a quantised matrix `W: [n_out, n_in]`.
///
/// One grid instance per `bo` output rows; the input is walked in chunks of
/// `kc` with an f32 accumulator carried through the loop. Parameters are
/// `x, values, scales, [mins], [r], y`.
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
    if !n_out.is_multiple_of(bo) || !n_in.is_multiple_of(kc) || !kc.is_multiple_of(g) {
        return Err(format!(
            "matvec {n_out}x{n_in}: bo {bo} must divide the rows, kc {kc} the columns, \
             and kc must be whole groups of {g}"
        ));
    }
    let name = format!(
        "matvec_{}_{n_out}x{n_in}{}",
        layout.tag(),
        if residual { "_res" } else { "" }
    );
    let mut b = Builder::new(&name);
    let px = b.param("x", DType::F32, &[1, n_in], false);
    let pq = b.param("wq", DType::I8, &[n_out, n_in], false);
    let ps = b.param("ws", DType::F32, &[n_out, n_in / g], false);
    let pm = layout
        .min
        .then(|| b.param("wm", DType::F32, &[n_out, n_in / g], false));
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
            let x = b.op(
                "x",
                Op::Load(
                    at(px, [IdxExpr::lit(0), IdxExpr::scaled(c, kc, 0)], [1, kc]),
                    reg(DType::F32, &[1, kc]),
                ),
            );
            let q = b.op(
                "q",
                Op::Load(
                    at(pq, [rows.clone(), IdxExpr::scaled(c, kc, 0)], [bo, kc]),
                    reg(DType::I8, &[bo, kc]),
                ),
            );
            let groups = |b: &mut Builder, p, name| {
                b.op(
                    name,
                    Op::Load(
                        at(
                            p,
                            [rows.clone(), IdxExpr::scaled(c, kc / g, 0)],
                            [bo, kc / g],
                        ),
                        reg(DType::F32, &[bo, kc / g]),
                    ),
                )
            };
            let s = groups(b, ps, "s");
            let m = pm.map(|pm| Move(groups(b, pm, "m")));
            let w = b.op("w", Op::Dequant(Move(q), Move(s), m, g));
            let part = b.op("part", Op::MatMulNT(Move(x), Move(w), DType::F32));
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

/// A quantised matrix repacked for [`matvec_q`]: one `i8` per value, and one
/// f32 scale (and min) per group. Every product that forms a scale or min is
/// exact in f32, so the kernels dequantise to the same floats llama.cpp's
/// reference dequantisation does.
#[derive(Clone, Debug, PartialEq)]
pub struct Split {
    pub layout: QLayout,
    pub q: Vec<i8>,
    pub s: Vec<f32>,
    pub m: Option<Vec<f32>>,
}

impl Split {
    /// Dequantise values `lo..hi` (row-major flat indices).
    pub fn dequant(&self, lo: usize, hi: usize) -> Vec<f32> {
        let g = self.layout.group;
        (lo..hi)
            .map(|i| {
                let min = self.m.as_ref().map_or(0.0, |m| m[i / g]);
                self.q[i] as f32 * self.s[i / g] - min
            })
            .collect()
    }
}

fn f16_at(b: &[u8], i: usize) -> f32 {
    half::f16::from_le_bytes([b[i], b[i + 1]]).to_f32()
}

/// Q8_0 blocks (34 bytes: f16 scale, 32 × i8).
pub fn split_q8_0(blocks: &[u8]) -> Split {
    let (blocks, rest) = blocks.as_chunks::<34>();
    assert!(rest.is_empty(), "not whole Q8_0 blocks");
    let mut q = Vec::with_capacity(blocks.len() * 32);
    let mut s = Vec::with_capacity(blocks.len());
    for blk in blocks {
        s.push(f16_at(blk, 0));
        q.extend(blk[2..].iter().map(|&b| b as i8));
    }
    Split {
        layout: QLayout::Q8_0,
        q,
        s,
        m: None,
    }
}

/// Q6_K super-blocks (210 bytes, 256 values): `ql[128]` low nibbles,
/// `qh[64]` high 2-bit pairs, 16 × i8 sub-block scales, f16 `d`.
/// Value `v` of a half is `d * sc[v / 16] * (q - 32)`.
pub fn split_q6_k(blocks: &[u8]) -> Split {
    let (blocks, rest) = blocks.as_chunks::<210>();
    assert!(rest.is_empty(), "not whole Q6_K blocks");
    let mut q = Vec::with_capacity(blocks.len() * 256);
    let mut s = Vec::with_capacity(blocks.len() * 16);
    for blk in blocks {
        let d = f16_at(blk, 208);
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
            for g in 0..8 {
                s.push(d * blk[192 + 8 * half + g] as i8 as f32);
            }
        }
    }
    Split {
        layout: QLayout::Q6_K,
        q,
        s,
        m: None,
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
pub fn split_q4_k(blocks: &[u8]) -> Split {
    let (blocks, rest) = blocks.as_chunks::<144>();
    assert!(rest.is_empty(), "not whole Q4_K blocks");
    let mut q = Vec::with_capacity(blocks.len() * 256);
    let mut s = Vec::with_capacity(blocks.len() * 8);
    let mut m = Vec::with_capacity(blocks.len() * 8);
    for blk in blocks {
        let (d, dmin) = (f16_at(blk, 0), f16_at(blk, 2));
        let sc = &blk[4..16];
        let qs = &blk[16..144];
        for j in 0..8 {
            let (scj, mj) = scale_min_k4(j, sc);
            s.push(d * scj as f32);
            m.push(dmin * mj as f32);
            let run = &qs[32 * (j / 2)..32 * (j / 2) + 32];
            q.extend(
                run.iter()
                    .map(|&b| if j % 2 == 0 { b & 0xF } else { b >> 4 } as i8),
            );
        }
    }
    Split {
        layout: QLayout::Q4_K,
        q,
        s,
        m: Some(m),
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

    #[test]
    fn matvec_matches_dequantised_weights_in_every_layout() {
        let (n_in, n_out) = (128, 16);
        for layout in [QLayout::Q8_0, QLayout::Q6_K, QLayout::Q4_K] {
            for residual in [false, true] {
                let p = matvec_q(n_in, n_out, 4, 64, layout, residual).unwrap();
                ok(&p);
                let groups = n_out * n_in / layout.group;
                let sp = Split {
                    layout,
                    q: pattern(n_in * n_out, 4)
                        .iter()
                        .map(|v| (v * 31.0).round() as i8)
                        .collect(),
                    s: pattern(groups, 5).iter().map(|v| v.abs() * 0.01).collect(),
                    m: layout
                        .min
                        .then(|| pattern(groups, 7).iter().map(|v| v * 0.05).collect()),
                };
                let x = pattern(n_in, 3);
                let r = pattern(n_out, 6);
                let qf: Vec<f32> = sp.q.iter().map(|&v| v as f32).collect();
                let mut t = vec![
                    Tensor::new(DType::F32, &[1, n_in], &x),
                    Tensor::new(DType::I8, &[n_out, n_in], &qf),
                    Tensor::new(DType::F32, &[n_out, n_in / layout.group], &sp.s),
                ];
                if let Some(m) = &sp.m {
                    t.push(Tensor::new(DType::F32, &[n_out, n_in / layout.group], m));
                }
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
                assert!(
                    max_rel_err(got, &want) < 1e-5,
                    "{layout:?} residual {residual}"
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
        assert_eq!((sp.q.len(), sp.s.len()), (64, 2));
        assert_eq!((sp.q[0], sp.q[32]), (-1, 3));
        assert_eq!((sp.s[0], sp.s[1]), (0.5, 2.0));
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
        let sp = split_q4_k(&raw);
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
