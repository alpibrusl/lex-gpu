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
use lex_front::llama::{QLayout, matmul_q, matvec_q, rmsnorm_rows, silu_mul};
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
        Tensor::new(DType::F32, &[n_out], &vec![1.0; n_out]),
        Tensor::zeros(DType::F32, &[1, n_out]),
    ];
    same(&g, &prog, tensors, 4, 1e-4);
}

/// The same at the shape and schedule the L4 runs: a 5120-wide row, 16
/// lanes to a row (`bo` 16, two rows a warp). That is the wide-load path --
/// one `uint2` of weights and four `float4` of input per run -- and a load
/// that lands off its alignment faults here rather than in a model.
#[test]
fn nvfp4_matvec_loaded_wide_matches_the_interpreter() {
    let Some(g) = gpu() else { return };
    let (n_in, n_out) = (5120, 32);
    let prog = matvec_q(n_in, n_out, 16, n_in, QLayout::NVFP4, false).expect("program");
    let codes: Vec<f32> = (0..n_out * n_in / 2)
        .map(|i| ((i.wrapping_mul(131).wrapping_add(17) & 0xFF) as i32 - 128) as f32)
        .collect();
    let scales: Vec<f32> = (0..n_out * n_in / 16)
        .map(|i| (0x30i32 + (i % 11) as i32 - 128) as f32)
        .collect();
    let rows: Vec<f32> = (0..n_out).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
    let tensors = vec![
        Tensor::new(DType::F32, &[1, n_in], &fill(n_in, 9)),
        Tensor::new(DType::I8, &[n_out, n_in / 2], &codes),
        Tensor::new(DType::I8, &[n_out, n_in / 16], &scales),
        Tensor::new(DType::F32, &[n_out], &rows),
        Tensor::zeros(DType::F32, &[1, n_out]),
    ];
    same(&g, &prog, tensors, 4, 1e-4);
}

/// The batched form -- a verify's three tokens, a prefill chunk's eight --
/// loaded wide: each row's run as one `uint2`, each token's quarter-run as
/// one `float4`. Eight is where the kernel holds the most registers.
#[test]
fn nvfp4_matmul_loaded_wide_matches_the_interpreter() {
    let Some(g) = gpu() else { return };
    for m in [3, 8] {
        let (n_in, n_out) = (5120, 64);
        let prog = matmul_q(m, n_in, n_out, 32, n_in, QLayout::NVFP4, false).expect("program");
        let codes: Vec<f32> = (0..n_out * n_in / 2)
            .map(|i| ((i.wrapping_mul(97).wrapping_add(3) & 0xFF) as i32 - 128) as f32)
            .collect();
        let scales: Vec<f32> = (0..n_out * n_in / 16)
            .map(|i| (0x34i32 + (i % 7) as i32 - 128) as f32)
            .collect();
        let tensors = vec![
            Tensor::new(DType::F32, &[m, n_in], &fill(m * n_in, 13)),
            Tensor::new(DType::I8, &[n_out, n_in / 2], &codes),
            Tensor::new(DType::I8, &[n_out, n_in / 16], &scales),
            Tensor::new(DType::F32, &[n_out], &vec![0.75; n_out]),
            Tensor::zeros(DType::F32, &[m, n_out]),
        ];
        same(&g, &prog, tensors, 4, 1e-4);
    }
}

/// The hand-scheduled prefill GEMM (`lex_msl::gemm`) against the
/// interpreter running the `matmul_q_x` it stands in for. Ragged tokens and
/// rows, with and without the residual, half and f32 activations. The GEMM
/// rounds weights to half for the tensor cores, so the bar is half's.
#[test]
fn the_gemm_matches_the_matmul_it_replaces() {
    use lex_front::llama::matmul_q_x;
    use lex_msl::gemm::{Backend, Gemm, gemm_nvfp4};
    let Some(g) = gpu() else { return };
    for c in [
        Gemm {
            m: 40,
            n: 100,
            k: 512,
            residual: false,
            x_half: true,
        },
        Gemm {
            m: 40,
            n: 100,
            k: 512,
            residual: true,
            x_half: true,
        },
        Gemm {
            m: 64,
            n: 128,
            k: 512,
            residual: false,
            x_half: false,
        },
        Gemm {
            m: 17,
            n: 48,
            k: 256,
            residual: true,
            x_half: false,
        },
        Gemm {
            m: 96,
            n: 192,
            k: 1024,
            residual: false,
            x_half: true,
        },
    ] {
        let (m, n, k) = (c.m, c.n, c.k);
        let xt = if c.x_half { DType::F16 } else { DType::F32 };
        let prog = matmul_q_x(m, k, n, 4, k, QLayout::NVFP4, c.residual, xt).expect("program");
        let codes: Vec<f32> = (0..n * k / 2)
            .map(|i| ((i.wrapping_mul(97).wrapping_add(3) & 0xFF) as i32 - 128) as f32)
            .collect();
        let scales: Vec<f32> = (0..n * k / 16)
            .map(|i| (0x34i32 + (i % 7) as i32 - 128) as f32)
            .collect();
        let rows: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.125).collect();
        let mut tensors = vec![
            Tensor::new(xt, &[m, k], &fill(m * k, 3)),
            Tensor::new(DType::I8, &[n, k / 2], &codes),
            Tensor::new(DType::I8, &[n, k / 16], &scales),
            Tensor::new(DType::F32, &[n], &rows),
        ];
        if c.residual {
            tensors.push(Tensor::new(DType::F32, &[m, n], &fill(m * n, 11)));
        }
        tensors.push(Tensor::zeros(DType::F32, &[m, n]));
        let out = tensors.len() - 1;

        let pipe = g
            .build_lowered(&gemm_nvfp4(&c, Backend::Cuda).expect("gemm"))
            .unwrap_or_else(|e| panic!("{c:?}: {e}"));
        let bufs: Vec<_> = tensors
            .iter()
            .map(|t| match t.dtype {
                DType::F32 => g.upload(&t.data),
                DType::F16 => {
                    g.upload(&t.data.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>())
                }
                DType::I8 => g.upload(&t.data.iter().map(|&x| x as i8).collect::<Vec<_>>()),
            })
            .collect();
        let refs: Vec<_> = bufs.iter().collect();
        g.run(&pipe, &refs).unwrap_or_else(|e| panic!("{c:?}: {e}"));
        let mut got = vec![0.0f32; m * n];
        g.download(&bufs[out], &mut got);

        run(&prog, &mut tensors).expect("interpret");
        let want = &tensors[out].data;
        let scale = want.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
        assert!(scale > 1e-3, "{c:?}: the reference output is all but zero");
        let err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!("{c:?}: {err:e} of scale");
        assert!(err < 2e-3, "{c:?}: GEMM vs interpreter {err:e} of scale");
    }
}
