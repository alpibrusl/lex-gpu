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

/// The int8 matmul (`lex_msl::int8`): the input quantised to int16 by
/// `quant16`, E2M1 through `__byte_perm` tables, `__dp2a`. Against the
/// interpreter running `matmul_q_x`: activations carry 16 bits, so the bar
/// is the GEMM's, not the percent 8-bit activations needed.
/// And a token's result must not depend on its batch: bit-identical alone
/// and among others, which a speculative verify checked against plain
/// decode relies on.
#[test]
fn the_int8_matmul_matches_the_matmul_and_ignores_its_batch() {
    use lex_front::llama::matmul_q_x;
    use lex_msl::int8::{matmul_int8, quant16};
    let Some(g) = gpu() else { return };
    let (n, k): (usize, usize) = (96, 1024);
    let codes: Vec<f32> = (0..n * k / 2)
        .map(|i| ((i.wrapping_mul(97).wrapping_add(3) & 0xFF) as i32 - 128) as f32)
        .collect();
    let scales: Vec<f32> = (0..n * k / 16)
        .map(|i| (0x34i32 + (i % 7) as i32 - 128) as f32)
        .collect();
    let rows: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.125).collect();
    let as_i8 = |v: &[f32]| v.iter().map(|&x| x as i8).collect::<Vec<_>>();
    let (qb, sb, gb) = (
        g.upload(&as_i8(&codes)),
        g.upload(&as_i8(&scales)),
        g.upload(&rows),
    );
    let mut alone: Option<Vec<f32>> = None;
    for (m, residual, x_half) in [
        (1, false, false),
        (3, false, true),
        (3, true, false),
        (8, true, true),
    ] {
        let xt = if x_half { DType::F16 } else { DType::F32 };
        let x = fill(m * k, 3);
        let r = fill(m * n, 11);
        let xb = if x_half {
            g.upload(&x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>())
        } else {
            g.upload(&x)
        };
        let xq = g.zeroed::<i16>(m * k);
        let xs = g.zeroed::<f32>(m * k / 16);
        let rb = g.upload(&r);
        let yb = g.zeroed::<f32>(m * n);
        let quant = g
            .build_lowered(&quant16(m, k, x_half).expect("quant"))
            .expect("build");
        let mm = g
            .build_lowered(&matmul_int8(m, n, k, residual).expect("mm"))
            .expect("build");
        g.run(&quant, &[&xb, &xq, &xs]).expect("quant");
        let mut bufs = vec![&xq, &xs, &qb, &sb, &gb];
        if residual {
            bufs.push(&rb);
        }
        bufs.push(&yb);
        g.run(&mm, &bufs).expect("mm");
        let mut got = vec![0.0f32; m * n];
        g.download(&yb, &mut got);

        let prog = matmul_q_x(m, k, n, 4, k, QLayout::NVFP4, residual, xt).expect("program");
        let mut t = vec![
            Tensor::new(xt, &[m, k], &x),
            Tensor::new(DType::I8, &[n, k / 2], &codes),
            Tensor::new(DType::I8, &[n, k / 16], &scales),
            Tensor::new(DType::F32, &[n], &rows),
        ];
        if residual {
            t.push(Tensor::new(DType::F32, &[m, n], &r));
        }
        t.push(Tensor::zeros(DType::F32, &[m, n]));
        let out = t.len() - 1;
        run(&prog, &mut t).expect("interpret");
        let want = &t[out].data;
        let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        assert!(scale > 1e-3, "m={m}: the reference is all but zero");
        let err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!("int8 m={m} residual={residual} x_half={x_half}: {err:e} of scale");
        assert!(err < 2e-3, "m={m}: int8 vs interpreter {err:e} of scale");

        // The first token of the f32, no-residual batches: alone (m=1) and
        // first of three must agree to the bit -- same input, same order.
        if !residual && !x_half {
            match &alone {
                None => alone = Some(got[..n].to_vec()),
                Some(a) => assert_eq!(
                    a,
                    &got[..n].to_vec(),
                    "a token's output depends on its batch"
                ),
            }
        }
    }
    // The (3, false, true) case is half input, so run the batch-independence
    // check explicitly on f32 at m=3 too.
    let x = fill(3 * k, 3);
    let xb = g.upload(&x);
    let xq = g.zeroed::<i16>(3 * k);
    let xs = g.zeroed::<f32>(3 * k / 16);
    let yb = g.zeroed::<f32>(3 * n);
    let quant = g
        .build_lowered(&quant16(3, k, false).expect("quant"))
        .expect("build");
    let mm = g
        .build_lowered(&matmul_int8(3, n, k, false).expect("mm"))
        .expect("build");
    g.run(&quant, &[&xb, &xq, &xs]).expect("quant");
    g.run(&mm, &[&xq, &xs, &qb, &sb, &gb, &yb]).expect("mm");
    let mut three = vec![0.0f32; 3 * n];
    g.download(&yb, &mut three);
    assert_eq!(
        alone.expect("the m=1 case ran"),
        three[..n].to_vec(),
        "a token's output depends on its batch"
    );
}

/// The chunked gated delta rule (`lex_msl::delta`) against the interpreter
/// running the step program it stands in for, token by token: outputs and
/// the final state, to f32 rounding (the two sum in different orders).
#[test]
fn chunked_delta_rule_matches_the_step_program() {
    use lex_front::qwen::DeltaNet;
    use lex_msl::delta::{DeltaChunk, delta_chunked};
    use lex_msl::gemm::Backend;
    let Some(g) = gpu() else { return };
    // (tokens, v_base, v_width, gate by head): mild decay; several chunks;
    // v behind other columns; a head whose gate underflows to nothing.
    type Gate = fn(usize) -> f32;
    // 37 and 9 are not whole chunks of 16: the last is padding.
    let cases: [(usize, usize, usize, Gate); 6] = [
        (37, 0, 384, |h| 0.9 + 0.09 * ((h % 5) as f32 / 4.0)),
        (9, 100, 600, |_| 0.97),
        (16, 0, 384, |h| 0.9 + 0.09 * ((h % 5) as f32 / 4.0)),
        (64, 0, 384, |h| 0.5 + 0.49 * ((h * 7 % 9) as f32 / 8.0)),
        (48, 100, 600, |_| 0.97),
        (32, 0, 384, |h| if h % 3 == 1 { 0.0 } else { 0.95 }),
    ];
    let (hv, dk, dv) = (3usize, 128usize, 128usize);
    for (t, v_base, v_width, gate) in cases {
        let d = DeltaChunk {
            tokens: t,
            v_heads: hv,
            k_dim: dk,
            v_dim: dv,
            v_base,
            v_width,
        };
        let prog = DeltaNet {
            v_heads: hv,
            k_heads: hv,
            k_dim: dk,
            v_dim: dv,
            rows: 8,
            v_base,
            v_width,
        }
        .build_steps(t)
        .expect("step program");
        let unit = |seed: u32| {
            let mut x = fill(t * hv * dk, seed);
            for row in x.chunks_mut(dk) {
                let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
                row.iter_mut().for_each(|v| *v /= n);
            }
            x
        };
        let spread =
            |f: &dyn Fn(usize) -> f32| (0..t * hv * dv).map(|i| f(i / dv)).collect::<Vec<f32>>();
        let beta = |h: usize| 0.2 + 0.6 * ((h * 37 % 11) as f32 / 10.0);
        let state0: Vec<f32> = fill(hv * dv * dk, 5).iter().map(|v| v * 0.3).collect();
        let mut tensors = vec![
            Tensor::new(DType::F32, &[hv * dv, dk], &state0),
            Tensor::new(DType::F32, &[t * hv, dk], &unit(7)),
            Tensor::new(DType::F32, &[t * hv, dk], &unit(9)),
            Tensor::new(DType::F32, &[t * v_width], &fill(t * v_width, 11)),
            Tensor::new(DType::F32, &[t * hv * dv], &spread(&gate)),
            Tensor::new(DType::F32, &[t * hv * dv], &spread(&beta)),
            Tensor::zeros(DType::F32, &[t * hv * dv]),
        ];
        let pipe = g
            .build_lowered(&delta_chunked(&d, Backend::Cuda).expect("chunked"))
            .unwrap_or_else(|e| panic!("{d:?}: {e}"));
        let bufs: Vec<_> = tensors.iter().map(|x| g.upload(&x.data)).collect();
        let refs: Vec<_> = bufs.iter().collect();
        g.run(&pipe, &refs).unwrap_or_else(|e| panic!("{d:?}: {e}"));
        let mut y = vec![0.0f32; t * hv * dv];
        g.download(&bufs[6], &mut y);
        let mut s = vec![0.0f32; hv * dv * dk];
        g.download(&bufs[0], &mut s);
        run(&prog, &mut tensors).expect("interpret");
        let err = |got: &[f32], want: &[f32]| {
            let scale = want.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
            assert!(
                scale > 0.1,
                "{d:?}: the reference is all but zero ({scale})"
            );
            got.iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / scale
        };
        let (ey, es) = (err(&y, &tensors[6].data), err(&s, &tensors[0].data));
        eprintln!("{d:?}: y {ey:e}, state {es:e} of scale");
        assert!(ey < 1e-4 && es < 1e-4, "{d:?}: y {ey:e}, state {es:e}");
    }
}

/// The `.lx` GEMM on the matrix units, run: `gemm_mma.lx` parsed, checked,
/// lowered to `wmma` with the Ada schedule's warp grid, padding and 16-byte
/// copies, and compared with a host product of the f16-rounded inputs. Shapes
/// that are one block and several, and an inner dimension of one step and
/// several, so every carried accumulator and every offset is used.
#[test]
fn the_lx_gemm_runs_on_the_matrix_units() {
    use lex_msl::program::{Sched, lower_sched};
    let Some(g) = gpu() else { return };
    let unit = lex_front::syntax::parse(include_str!("../../lex-front/lx/gemm_mma.lx"))
        .unwrap_or_else(|e| panic!("{e}"));
    let target = Target::nvidia_ada();
    let s = unit.schedule_for(target.name).expect("an Ada schedule");
    for (m, n, k) in [(128usize, 128usize, 64usize), (256, 384, 192)] {
        let (prog, threads) = unit
            .compile(
                target.name,
                &[("m", m as f64), ("n", n as f64), ("k", k as f64)],
            )
            .unwrap_or_else(|e| panic!("{e}"));
        check(&prog, &target).unwrap_or_else(|e| panic!("{e:#?}"));
        let l = lower_sched(
            &prog,
            &target,
            &Cuda,
            &Sched {
                threads,
                warps: s.warps,
                pad: s.pad,
            },
        )
        .unwrap_or_else(|e| panic!("{e}"));
        let pipe = g.build_lowered(&l).expect("compile");
        let (a, w) = (fill(m * k, 21), fill(n * k, 22));
        let h = |x: &[f32]| x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>();
        let (ab, wb) = (g.upload(&h(&a)), g.upload(&h(&w)));
        let yb = g.upload(&vec![9.0f32; m * n]);
        g.run(&pipe, &[&ab, &wb, &yb]).expect("run");
        let mut y = vec![0.0f32; m * n];
        g.download(&yb, &mut y);
        let (ar, wr) = (
            h(&a).iter().map(|v| v.to_f32() as f64).collect::<Vec<_>>(),
            h(&w).iter().map(|v| v.to_f32() as f64).collect::<Vec<_>>(),
        );
        let mut worst = 0.0f64;
        let mut peak = 1e-9f64;
        for i in 0..m {
            for j in 0..n {
                let want: f64 = (0..k).map(|p| ar[i * k + p] * wr[j * k + p]).sum();
                peak = peak.max(want.abs());
                worst = worst.max((y[i * n + j] as f64 - want).abs());
            }
        }
        eprintln!("lx gemm {m}x{n}x{k}: worst {:.2e} of scale", worst / peak);
        assert!(worst / peak < 1e-4, "off by {:.2e} of scale", worst / peak);
    }
}

/// The `.lx` NVFP4 GEMM against `lex_msl::gemm`'s, on the same weights: the
/// language's kernel has to be the hand-written one's answer, to f32
/// accumulation order, before its speed means anything
/// (`examples/lx_gemm_bench` times it).
#[test]
fn the_lx_fp4_gemm_agrees_with_the_hand_written_one() {
    use lex_msl::gemm::{Backend, Gemm, gemm_nvfp4};
    use lex_msl::program::{Sched, lower_sched};
    let Some(g) = gpu() else { return };
    let unit = lex_front::syntax::parse(include_str!("../../lex-front/lx/gemm_fp4.lx"))
        .unwrap_or_else(|e| panic!("{e}"));
    let target = Target::nvidia_ada();
    let s = unit.schedule_for(target.name).expect("an Ada schedule");
    for (m, n, k) in [(128usize, 128usize, 64usize), (256, 512, 1024)] {
        let hand = g
            .build_lowered(
                &gemm_nvfp4(
                    &Gemm {
                        m,
                        n,
                        k,
                        residual: false,
                        x_half: true,
                    },
                    Backend::Cuda,
                )
                .expect("hand-written"),
            )
            .expect("compile hand-written");
        let (prog, threads) = unit
            .compile(
                target.name,
                &[("m", m as f64), ("n", n as f64), ("k", k as f64)],
            )
            .unwrap_or_else(|e| panic!("{e}"));
        let lx = g
            .build_lowered(
                &lower_sched(
                    &prog,
                    &target,
                    &Cuda,
                    &Sched {
                        threads,
                        warps: s.warps,
                        pad: s.pad,
                    },
                )
                .unwrap_or_else(|e| panic!("{e}")),
            )
            .expect("compile lx");
        let x = g.upload(
            &fill(m * k, 41)
                .iter()
                .map(|&v| f16::from_f32(v))
                .collect::<Vec<_>>(),
        );
        let q = g.upload(
            &(0..n * k / 2)
                .map(|i| (i.wrapping_mul(97) + 5) as u8)
                .collect::<Vec<_>>(),
        );
        let sc = g.upload(
            &(0..n * k / 16)
                .map(|i| 0x30u8 + (i % 7) as u8)
                .collect::<Vec<_>>(),
        );
        let gs = g.upload(
            &(0..n)
                .map(|i| 0.5 + (i % 5) as f32 * 0.25)
                .collect::<Vec<f32>>(),
        );
        let (yh, yl) = (
            g.upload(&vec![9.0f32; m * n]),
            g.upload(&vec![9.0f32; m * n]),
        );
        g.run(&hand, &[&x, &q, &sc, &gs, &yh]).expect("run hand");
        g.run(&lx, &[&x, &q, &sc, &gs, &yl]).expect("run lx");
        let (mut a, mut b) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        g.download(&yh, &mut a);
        g.download(&yl, &mut b);
        let peak = a.iter().fold(1e-9f32, |p, v| p.max(v.abs()));
        let worst = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "lx fp4 gemm {m}x{n}x{k}: worst {:.2e} of scale against the hand-written",
            worst / peak
        );
        assert!(worst / peak < 1e-4, "off by {:.2e} of scale", worst / peak);
    }
}
