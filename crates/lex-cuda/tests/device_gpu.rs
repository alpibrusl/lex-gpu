//! The emitted CUDA, run on a real NVIDIA GPU, against the interpreter.
//!
//! This is the layer nothing else can stand in for. Golden files prove the
//! text has not changed; `scripts/cuda_check.sh` proves it compiles and
//! assembles. Neither says the kernel computes the right thing, and a
//! backend can be wrong in ways that compile cleanly — CUDA binds buffers
//! positionally, so a mis-ordered launch reads the wrong memory and
//! returns plausible numbers.
//!
//! Linux with a driver only. It skips itself elsewhere, as the Metal tests
//! do without a Mac, which is what lets `cargo test --workspace` stay green
//! on the machine this was written on.
//!
//! Run it with `scripts/gcp/nvidia_test.sh`, which `remote.sh` already
//! wires up.

#![cfg(target_os = "linux")]

use half::f16;
use lex_cuda::device::Gpu;
use lex_front::llama::{QLayout, matvec_q, rmsnorm_rows, silu_mul};
use lex_front::{Program, Tensor, check, run};
use lex_ir::{DType, Target};
use lex_msl::dialect::Cuda;
use lex_msl::program::lower_with;

/// Deterministic, and spread far enough that a wrong buffer or a dropped
/// term shows up rather than averaging out.
fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x >> 9) as f32 / 4_194_304.0) - 1.0
        })
        .collect()
}

/// Lower `prog` for CUDA, run it, and compare output `out` with the
/// interpreter.
fn same(gpu: &Gpu, prog: &Program, mut tensors: Vec<Tensor>, out: usize, tol: f32) {
    let target = Target::nvidia_ada();
    check(prog, &target).unwrap_or_else(|e| panic!("{}: {e:#?}", prog.name));
    let lowered = lower_with(prog, &target, 256, &Cuda).expect("lower");
    let pipe = gpu
        .build_lowered(&lowered)
        .unwrap_or_else(|e| panic!("{}: {e}", prog.name));

    let bufs: Vec<_> = tensors
        .iter()
        .map(|t| match t.dtype {
            DType::F32 => gpu.upload(&t.data),
            DType::F16 => gpu.upload(&t.data.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>()),
            DType::I8 => gpu.upload(&t.data.iter().map(|&x| x as i8).collect::<Vec<_>>()),
        })
        .collect();
    let refs: Vec<_> = bufs.iter().collect();
    gpu.run(&pipe, &refs)
        .unwrap_or_else(|e| panic!("{}: {e}", prog.name));

    assert_eq!(tensors[out].dtype, DType::F32, "compare an f32 output");
    let mut got = vec![0.0f32; tensors[out].data.len()];
    gpu.download(&bufs[out], &mut got);

    run(prog, &mut tensors).expect("interpret");
    let want = tensors[out].data.clone();

    // Relative to the output's scale: a reduction summed in a different
    // order legitimately differs in the last bits, and an output that
    // cancels near zero makes a per-element relative error meaningless.
    let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        err / scale <= tol,
        "{}: CUDA vs interpreter {:e} of scale {scale:e} (tolerance {tol:e})",
        prog.name,
        err / scale
    );
    eprintln!("{}: {:e} of scale", prog.name, err / scale);
}

fn gpu() -> Option<Gpu> {
    match Gpu::open() {
        Ok(g) => {
            eprintln!("{} ({})", g.name(), g.arch());
            Some(g)
        }
        Err(e) => {
            eprintln!("SKIPPED: no CUDA device ({e})");
            None
        }
    }
}

#[test]
fn rmsnorm_matches_the_interpreter() {
    let Some(g) = gpu() else { return };
    let (rows, n) = (4, 4096);
    let prog = rmsnorm_rows(rows, n, 1e-5, None, DType::F32);
    let tensors = vec![
        Tensor::new(DType::F32, &[rows, n], &fill(rows * n, 1)),
        Tensor::new(DType::F32, &[1, n], &fill(n, 2)),
        Tensor::zeros(DType::F32, &[rows, n]),
    ];
    same(&g, &prog, tensors, 2, 1e-5);
}

#[test]
fn silu_mul_matches_the_interpreter() {
    let Some(g) = gpu() else { return };
    let n = 4096;
    let prog = silu_mul(n, 256, DType::F32).expect("program");
    let tensors = vec![
        Tensor::new(DType::F32, &[1, n], &fill(n, 3)),
        Tensor::new(DType::F32, &[1, n], &fill(n, 4)),
        Tensor::zeros(DType::F32, &[1, n]),
    ];
    same(&g, &prog, tensors, 2, 1e-5);
}

/// The one that matters: NVFP4 dequantisation fused into a matvec, which is
/// 94% of a decode step.
///
/// The decode is a bit trick that relies on IEEE half's denormal boundary
/// coinciding with E2M1's. That is a property of the format, not of Metal,
/// and this is where that stops being an argument.
#[test]
fn nvfp4_matvec_matches_the_interpreter() {
    let Some(g) = gpu() else { return };
    let (n_in, n_out) = (512, 64);
    let prog = matvec_q(n_in, n_out, 8, n_in, QLayout::NVFP4, false).expect("program");
    // Real bytes, not zeros: every NVFP4 code being 0 would hit one cache
    // line and decode to nothing, which is how this repository once
    // measured 435 GB/s that did not exist.
    let codes: Vec<f32> = (0..n_out * n_in / 2)
        .map(|i| ((i.wrapping_mul(31).wrapping_add(7) & 0xFF) as i32 - 128) as f32)
        .collect();
    let scales: Vec<f32> = (0..n_out * n_in / 16)
        .map(|i| (0x38i32 + (i % 5) as i32 - 128) as f32)
        .collect();
    let tensors = vec![
        Tensor::new(DType::F32, &[1, n_in], &fill(n_in, 5)),
        Tensor::new(DType::I8, &[n_out, n_in / 2], &codes),
        Tensor::new(DType::I8, &[n_out, n_in / 16], &scales),
        Tensor::new(DType::F32, &[n_out, 1], &vec![1.0; n_out]),
        Tensor::zeros(DType::F32, &[1, n_out]),
    ];
    same(&g, &prog, tensors, 4, 1e-4);
}
