//! The hand-scheduled NVFP4 GEMM against the interpreter running the
//! program it stands in for (`matmul_q_x`, same parameters, same order).
//!
//! The GEMM rounds each dequantised weight to half on its way into the
//! matrix units, where the batched matvec keeps f32, so the two agree to
//! half's precision and not to the last bit.

#![cfg(target_os = "macos")]

use half::f16;
use lex_front::llama::{QLayout, matmul_q_x};
use lex_front::{Tensor, run};
use lex_ir::DType;
use lex_ir::reference::fill_pattern_f32;
use lex_metal::{Buffer, Gpu};
use lex_msl::gemm::{Backend, Gemm, gemm_nvfp4};

fn upload(gpu: &Gpu, t: &Tensor) -> Buffer {
    match t.dtype {
        DType::F32 => gpu.upload(&t.data),
        DType::F16 => gpu.upload(&t.data.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>()),
        DType::I8 => gpu.upload(&t.data.iter().map(|&x| x as i8).collect::<Vec<_>>()),
    }
}

/// The GEMM for `g` against the interpreter, as a fraction of the output's
/// largest magnitude.
fn gemm_error(gpu: &Gpu, g: Gemm) -> f32 {
    let (m, n, k) = (g.m, g.n, g.k);
    let xt = if g.x_half { DType::F16 } else { DType::F32 };
    // `bo` only shapes the reference program's schedule; 4 divides every n
    // used here.
    let prog = matmul_q_x(m, k, n, 4, k, QLayout::NVFP4, g.residual, xt).expect("program");
    let mut x = vec![0.0; m * k];
    fill_pattern_f32(&mut x, 3);
    // Real codes and scales, not zeros: all-zero NVFP4 decodes to nothing
    // and agrees with anything.
    let codes: Vec<f32> = (0..n * k / 2)
        .map(|i| ((i.wrapping_mul(97).wrapping_add(3) & 0xFF) as i32 - 128) as f32)
        .collect();
    let scales: Vec<f32> = (0..n * k / 16)
        .map(|i| (0x34i32 + (i % 7) as i32 - 128) as f32)
        .collect();
    let rows: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.125).collect();
    let mut tensors = vec![
        Tensor::new(xt, &[m, k], &x),
        Tensor::new(DType::I8, &[n, k / 2], &codes),
        Tensor::new(DType::I8, &[n, k / 16], &scales),
        Tensor::new(DType::F32, &[n], &rows),
    ];
    if g.residual {
        let mut r = vec![0.0; m * n];
        fill_pattern_f32(&mut r, 11);
        tensors.push(Tensor::new(DType::F32, &[m, n], &r));
    }
    tensors.push(Tensor::zeros(DType::F32, &[m, n]));
    let out = tensors.len() - 1;

    let pipe = gpu
        .build_lowered(&gemm_nvfp4(&g, Backend::Metal).expect("gemm"))
        .expect("compile");
    let bufs: Vec<Buffer> = tensors.iter().map(|t| upload(gpu, t)).collect();
    let refs: Vec<&Buffer> = bufs.iter().collect();
    gpu.run(&pipe, &refs);
    let mut got = vec![0.0f32; m * n];
    gpu.download(&bufs[out], &mut got);

    run(&prog, &mut tensors).expect("interpret");
    let want = &tensors[out].data;
    let scale = want.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
    // Two kernels that both write nothing agree perfectly.
    assert!(scale > 1.0, "{g:?}: the reference output is all but zero ({scale})");
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
        / scale
}

#[test]
fn the_gemm_matches_the_matmul_it_replaces() {
    let Some(gpu) = Gpu::open().ok() else {
        eprintln!("no Metal device");
        return;
    };
    // Ragged on purpose: 40 tokens and 100 rows fill no tile exactly, so the
    // guards on both are exercised, and 512 is 16 K-tiles.
    for g in [
        Gemm { m: 40, n: 100, k: 512, residual: false, x_half: true },
        Gemm { m: 40, n: 100, k: 512, residual: true, x_half: true },
        Gemm { m: 64, n: 128, k: 512, residual: false, x_half: false },
        Gemm { m: 17, n: 48, k: 256, residual: true, x_half: false },
    ] {
        // With half activations the match is exact, not close: an f16 input
        // times a dequantised NVFP4 weight has few enough significant bits
        // that every product and partial sum is exact in f32, in any order.
        let err = gemm_error(&gpu, g);
        eprintln!("{g:?}: {err:e} of scale");
        assert!(err < 2e-3, "{g:?}: GEMM vs interpreter {err:e} of scale");
    }
}
