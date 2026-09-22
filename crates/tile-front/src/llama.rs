//! The layer kernels of a Llama decode step, as typed tile programs.
//!
//! Each is an ordinary `tile-front` program: it type-checks, runs in the
//! interpreter, and lowers to MSL like flash decode does. Activations are f32
//! rows `[1, n]`; weights are Q8_0, repacked by the loader into an `I8` value
//! tensor plus an f16 scale per 32 values, and meet their scales through
//! `dequant` — the only way the checker lets anyone compute with `I8`.
//!
//! Attention is [`crate::flash::FlashDecode`]; what is here is everything
//! around it.

use tile_ir::{DType, Space};

use crate::ir::{Arg, BinOp, Builder, IdxExpr, Op, Program, Reduce, TileTy, Ty, UnOp, View};

/// Q8_0's block: 32 values share one scale.
pub const Q8_GROUP: usize = 32;

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

/// `y = W x (+ r)` for a Q8_0 matrix `W: [n_out, n_in]`.
///
/// One grid instance per `bo` output rows; the input is walked in chunks of
/// `kc` with an f32 accumulator carried through the loop.
pub fn matvec_q8(
    n_in: usize,
    n_out: usize,
    bo: usize,
    kc: usize,
    residual: bool,
) -> Result<Program, String> {
    use Arg::Move;
    if !n_out.is_multiple_of(bo) || !n_in.is_multiple_of(kc) || !kc.is_multiple_of(Q8_GROUP) {
        return Err(format!(
            "matvec {n_out}x{n_in}: bo {bo} must divide the rows, kc {kc} the columns, \
             and kc must be whole Q8_0 blocks"
        ));
    }
    let name = format!(
        "matvec_q8_{n_out}x{n_in}{}",
        if residual { "_res" } else { "" }
    );
    let mut b = Builder::new(&name);
    let px = b.param("x", DType::F32, &[1, n_in], false);
    let pq = b.param("wq", DType::I8, &[n_out, n_in], false);
    let ps = b.param("ws", DType::F16, &[n_out, n_in / Q8_GROUP], false);
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
                    at(
                        pq,
                        [IdxExpr::scaled(pid, bo, 0), IdxExpr::scaled(c, kc, 0)],
                        [bo, kc],
                    ),
                    reg(DType::I8, &[bo, kc]),
                ),
            );
            let g = kc / Q8_GROUP;
            let s = b.op(
                "s",
                Op::Load(
                    at(
                        ps,
                        [IdxExpr::scaled(pid, bo, 0), IdxExpr::scaled(c, g, 0)],
                        [bo, g],
                    ),
                    reg(DType::F16, &[bo, g]),
                ),
            );
            let s = b.op("s32", Op::Convert(Move(s), DType::F32));
            let w = b.op("w", Op::Dequant(Move(q), Move(s), Q8_GROUP));
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

/// Split Q8_0 blocks (34 bytes: f16 scale, 32 × i8) into the value and scale
/// tensors the kernels take. Lossless: both halves are stored as they are.
pub fn split_q8_0(blocks: &[u8]) -> (Vec<i8>, Vec<half::f16>) {
    assert_eq!(blocks.len() % 34, 0, "not whole Q8_0 blocks");
    let n = blocks.len() / 34;
    let mut q = Vec::with_capacity(n * 32);
    let mut s = Vec::with_capacity(n);
    for blk in blocks.chunks_exact(34) {
        s.push(half::f16::from_le_bytes([blk[0], blk[1]]));
        q.extend(blk[2..].iter().map(|&b| b as i8));
    }
    (q, s)
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
    fn matvec_q8_matches_dequantised_weights() {
        let (n_in, n_out) = (128, 16);
        for residual in [false, true] {
            let p = matvec_q8(n_in, n_out, 4, 64, residual).unwrap();
            ok(&p);
            let x = pattern(n_in, 3);
            let q: Vec<f32> = pattern(n_in * n_out, 4)
                .iter()
                .map(|v| (v * 127.0).round())
                .collect();
            let s: Vec<f32> = pattern(n_out * n_in / 32, 5)
                .iter()
                .map(|v| v.abs() * 0.01)
                .collect();
            let r = pattern(n_out, 6);
            let mut t = vec![
                Tensor::new(DType::F32, &[1, n_in], &x),
                Tensor::new(DType::I8, &[n_out, n_in], &q),
                Tensor::new(DType::F16, &[n_out, n_in / 32], &s),
            ];
            if residual {
                t.push(Tensor::new(DType::F32, &[1, n_out], &r));
            }
            t.push(Tensor::zeros(DType::F32, &[1, n_out]));
            let sf = t[2].data.clone();
            run(&p, &mut t).unwrap();
            let want: Vec<f32> = (0..n_out)
                .map(|o| {
                    let dot: f32 = (0..n_in)
                        .map(|i| q[o * n_in + i] * sf[o * n_in / 32 + i / 32] * x[i])
                        .sum();
                    dot + if residual { r[o] } else { 0.0 }
                })
                .collect();
            let got = &t.last().unwrap().data;
            assert!(max_rel_err(got, &want) < 1e-5, "residual {residual}");
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
        let (q, s) = split_q8_0(&blk);
        assert_eq!((q.len(), s.len()), (64, 2));
        assert_eq!((q[0], q[32]), (-1, 3));
        assert_eq!((s[0].to_f32(), s[1].to_f32()), (0.5, 2.0));
    }
}
