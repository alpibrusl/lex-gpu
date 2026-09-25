//! A Walsh-Hadamard transform, built from butterfly stages.
//!
//! Low-bit weight formats increasingly store a *rotated* basis. Bonsai 2
//! folds a blockwise Hadamard into its ternary weights offline and the
//! runtime has to apply the matching rotation to activations, so a
//! compiler that cannot express one cannot run those weights at all.
//!
//! `Op::Butterfly` is one stage; a width-`2^k` transform is `k` of them.
//! The transform is therefore a sequence in the builder and not an op, so
//! nothing in the IR has to know how wide the caller wanted it.
//!
//! Checked against a Hadamard computed the slow, obvious way -- the
//! recursive definition applied as a dense matrix -- rather than against
//! another butterfly. Two butterflies agreeing proves they were written
//! by the same person.

use lex_front::ir::{Arg, Builder, IdxExpr, Op, Program, TileTy, View};
use lex_ir::{DType, Space, Target};
use lex_front::{check, interp::run, interp::Tensor};


/// `H_n x` by the definition: `H[i][j] = (-1)^popcount(i & j)`.
fn hadamard_reference(x: &[f32], n: usize) -> Vec<f32> {
    (0..x.len())
        .map(|i| {
            let (block, row) = (i / n, i % n);
            (0..n)
                .map(|col| {
                    let sign = if (row & col).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                    sign * x[block * n + col]
                })
                .sum()
        })
        .collect()
}

/// `y = H_width x`, as butterfly stages over the last dimension.
fn program(rows: usize, cols: usize, width: usize) -> Program {
    let mut b = Builder::new(&format!("hadamard_{rows}x{cols}_w{width}"));
    let px = b.param("x", DType::F32, &[rows, cols], false);
    let py = b.param("y", DType::F32, &[rows, cols], true);
    let whole = |p| View {
        param: p,
        offset: vec![IdxExpr::lit(0), IdxExpr::lit(0)],
        shape: vec![rows, cols],
    };
    let ty = TileTy::new(DType::F32, &[rows, cols], Space::Reg);
    let mut v = b.op("x", Op::Load(whole(px), ty));
    let mut stride = 1;
    while stride < width {
        v = b.op("h", Op::Butterfly(Arg::Move(v), stride));
        stride *= 2;
    }
    b.effect(Op::Store(Arg::Move(v), whole(py)));
    b.finish()
}

fn pattern(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((h >> 9) as f32 / 4.194e6) - 1.0
        })
        .collect()
}

#[test]
fn butterfly_stages_make_a_hadamard() {
    // A width that is not the whole row, because that is the case Bonsai
    // needs: 1024-wide blocks inside a 5120-wide matrix.
    for (rows, cols, width) in [(1, 8, 8), (1, 16, 4), (3, 16, 16), (2, 40, 8)] {
        let prog = program(rows, cols, width);
        check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));
        let x = pattern(rows * cols, 7);
        let mut t = vec![
            Tensor::new(DType::F32, &[rows, cols], &x),
            Tensor::zeros(DType::F32, &[rows, cols]),
        ];
        run(&prog, &mut t).expect("interpret");

        // The reference transforms each `width` block independently.
        let want: Vec<f32> = x
            .chunks(width)
            .flat_map(|c| hadamard_reference(c, width))
            .collect();
        let worst = t[1]
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "{rows}x{cols} width {width}: off by {worst:e}\n  got  {:?}\n  want {:?}",
            &t[1].data[..width.min(8)],
            &want[..width.min(8)]
        );
    }
}

#[test]
fn a_hadamard_is_its_own_inverse_up_to_scale() {
    // H H x = width * x. An independent property: it would catch a
    // transform that is self-consistent but not orthogonal.
    let (rows, cols, width) = (2, 32, 32);
    let prog = program(rows, cols, width);
    let x = pattern(rows * cols, 11);
    let mut once = vec![
        Tensor::new(DType::F32, &[rows, cols], &x),
        Tensor::zeros(DType::F32, &[rows, cols]),
    ];
    run(&prog, &mut once).expect("first");
    let mid = once[1].data.clone();
    let mut twice = vec![
        Tensor::new(DType::F32, &[rows, cols], &mid),
        Tensor::zeros(DType::F32, &[rows, cols]),
    ];
    run(&prog, &mut twice).expect("second");
    let worst = twice[1]
        .data
        .iter()
        .zip(&x)
        .map(|(a, b)| (a - b * width as f32).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-3, "H H x != {width} x, off by {worst:e}");
}

#[test]
fn a_bad_stride_is_rejected() {
    for (cols, stride, why) in [(8, 3, "not a power of two"), (8, 8, "multiple of 16")] {
        let mut b = Builder::new("bad");
        let px = b.param("x", DType::F32, &[1, cols], false);
        let py = b.param("y", DType::F32, &[1, cols], true);
        let v = |p| View {
            param: p,
            offset: vec![IdxExpr::lit(0), IdxExpr::lit(0)],
            shape: vec![1, cols],
        };
        let x = b.op("x", Op::Load(v(px), TileTy::new(DType::F32, &[1, cols], Space::Reg)));
        let h = b.op("h", Op::Butterfly(Arg::Move(x), stride));
        b.effect(Op::Store(Arg::Move(h), v(py)));
        let errs = check(&b.finish(), &Target::apple_m_series());
        let msg = format!("{:?}", errs.unwrap_err());
        assert!(msg.contains(why), "stride {stride} on {cols}: wanted {why:?}, got {msg}");
    }
}

/// The whole activation transform Bonsai 2 asks for: sign, then a
/// normalized blockwise Hadamard. Checked against the definition, not
/// against another butterfly.
#[test]
fn the_rotation_matches_a_direct_normalized_hadamard() {
    use lex_front::llama::hadamard_rotate;

    for (rows, cols, block) in [(1usize, 1024usize, 1024usize), (3, 2048, 1024), (2, 32, 8)] {
        let prog = hadamard_rotate(rows, cols, block, false).expect("build");
        check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));
        let x = pattern(rows * cols, 3);
        // An explicit +-1 per column, as the file stores it.
        let sign: Vec<f32> = (0..cols)
            .map(|c| if (c * 2_654_435_761usize).is_multiple_of(3) { -1.0 } else { 1.0 })
            .collect();
        let mut t = vec![
            Tensor::new(DType::F32, &[rows, cols], &x),
            Tensor::new(DType::F32, &[1, cols], &sign),
            Tensor::zeros(DType::F32, &[rows, cols]),
        ];
        run(&prog, &mut t).expect("interpret");

        let norm = 1.0 / (block as f32).sqrt();
        let want: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let (base, row) = (i - i % block, i % block);
                norm * (0..block)
                    .map(|col| {
                        let sgn = if (row & col).count_ones() % 2 == 0 { 1.0 } else { -1.0 };
                        sgn * sign[(base + col) % cols] * x[base + col]
                    })
                    .sum::<f32>()
            })
            .collect();
        let worst = t[2]
            .data
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let scale = want.iter().map(|v| v.abs()).fold(1e-6f32, f32::max);
        assert!(
            worst / scale < 1e-5,
            "{rows}x{cols} block {block}: off by {worst:e} on {scale:e}"
        );
    }
}

/// Forward then inverse is the identity. An independent property: it holds
/// only if the transform is orthogonal *and* the sign lands on the right
/// side of it both ways.
#[test]
fn the_inverse_rotation_undoes_the_forward_one() {
    use lex_front::llama::hadamard_rotate;

    let (rows, cols, block) = (2usize, 2048usize, 1024usize);
    let x = pattern(rows * cols, 17);
    let sign: Vec<f32> = (0..cols)
        .map(|c| if c % 5 < 2 { -1.0 } else { 1.0 })
        .collect();
    let mut fwd = vec![
        Tensor::new(DType::F32, &[rows, cols], &x),
        Tensor::new(DType::F32, &[1, cols], &sign),
        Tensor::zeros(DType::F32, &[rows, cols]),
    ];
    run(&hadamard_rotate(rows, cols, block, false).unwrap(), &mut fwd).expect("forward");
    let mid = fwd[2].data.clone();
    assert!(
        mid.iter().zip(&x).any(|(a, b)| (a - b).abs() > 1e-3),
        "the forward rotation did nothing, so the round trip proves nothing"
    );
    let mut back = vec![
        Tensor::new(DType::F32, &[rows, cols], &mid),
        Tensor::new(DType::F32, &[1, cols], &sign),
        Tensor::zeros(DType::F32, &[rows, cols]),
    ];
    run(&hadamard_rotate(rows, cols, block, true).unwrap(), &mut back).expect("inverse");
    let worst = back[2]
        .data
        .iter()
        .zip(&x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-4, "round trip off by {worst:e}");
}
