//! The prefill GEMM alone at Qwen3.8's four shapes, on whichever device
//! this machine has, each schedule checked against the default one.
//!
//!     cargo run --release -p lex-rt --example gemm_bench -- --tokens 512
//!     LEX_GEMM_CU_BM=128 LEX_GEMM_CU_VEC=1 cargo run --release -p lex-rt --example gemm_bench
//!     cargo run --release -p lex-rt --example gemm_bench -- --emit out/   # CUDA sources only
//!
//! `LEX_GEMM_CU_*` (see `lex_msl::gemm`) pick the CUDA schedule. The
//! default schedule is built too and run on the same inputs, and the
//! variant must agree with it to f16 rounding: a faster kernel that is
//! wrong would otherwise look like a win. Each shape runs over several
//! copies of its weights so they stream from memory as in the model.
//! `--emit <dir>` writes the CUDA source of every shape and needs no GPU:
//! `scripts/cuda_check.sh <dir>/*.cu` then compiles them with nvcc and
//! NVRTC before a cloud run.

fn main() -> Result<(), String> {
    use lex_msl::gemm::{Backend, Gemm};

    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let tokens: usize = arg("--tokens").and_then(|v| v.parse().ok()).unwrap_or(512);
    // (label, rows, inputs, residual, f16 activations): one layer's matmuls.
    let shapes = [
        ("gate/up", 17408usize, 5120usize, false, true),
        ("down", 5120, 17408, true, true),
        ("qkv", 10240, 5120, false, true),
        ("out_proj", 5120, 6144, true, true),
    ];
    let gemm = |n, k, residual, x_half| Gemm {
        m: tokens,
        n,
        k,
        residual,
        x_half,
    };

    if let Some(dir) = arg("--emit") {
        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        // `--all`: every schedule a tuner may pick, at every chunk size the
        // runtime compiles, so each can go through nvcc and NVRTC here.
        let all = args.iter().any(|a| a == "--all");
        let sizes: Vec<usize> = if all {
            vec![512, 256, 128, 64, 32, 16]
        } else {
            vec![tokens]
        };
        let mut written = 0;
        for t in sizes {
            for (label, n, k, res, xh) in shapes {
                let g = Gemm {
                    m: t,
                    n,
                    k,
                    residual: res,
                    x_half: xh,
                };
                let scheds: Vec<Option<lex_msl::gemm::CudaSchedule>> = if all {
                    lex_msl::gemm::cuda_candidates(&g)
                        .into_iter()
                        .map(Some)
                        .collect()
                } else {
                    vec![None]
                };
                for sc in scheds {
                    let l = lex_msl::gemm::gemm_nvfp4_with(&g, Backend::Cuda, sc)?;
                    let tag = sc.map(|s| s.name()).unwrap_or_default();
                    let path = format!("{dir}/{}_{t}_{tag}.cu", label.replace('/', "_"));
                    std::fs::write(&path, &l.source).map_err(|e| e.to_string())?;
                    written += 1;
                    if !all {
                        println!(
                            "{path}: {} threads, {} bytes shared",
                            l.threads, l.threadgroup_bytes
                        );
                    }
                }
            }
        }
        println!("{written} kernels in {dir}");
        return Ok(());
    }
    run(tokens, &shapes, gemm)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn run(
    tokens: usize,
    shapes: &[(&str, usize, usize, bool, bool)],
    gemm: impl Fn(usize, usize, bool, bool) -> lex_msl::gemm::Gemm,
) -> Result<(), String> {
    use half::f16;
    use lex_msl::gemm::gemm_nvfp4;
    use lex_rt::dev::{Buffer, Gpu, Step};

    let gpu = Gpu::open()?;
    let backend = lex_rt::dev::gemm_backend();
    let (copies, reps) = (4usize, 5usize);
    println!("{}, {tokens} tokens", gpu.info().name);
    println!(
        "{:<10} {:>6} {:>6} {:>9} {:>8} {:>12}",
        "shape", "n", "k", "ms", "TFLOPS", "vs default"
    );
    for &(label, n, k, residual, x_half) in shapes {
        let g = gemm(n, k, residual, x_half);
        let pipe = gpu.build_lowered(&gemm_nvfp4(&g, backend)?)?;
        // The default schedule, built with the knobs cleared.
        let saved: Vec<(String, String)> = std::env::vars()
            .filter(|(k, _)| k.starts_with("LEX_GEMM_"))
            .collect();
        for (k, _) in &saved {
            // SAFETY: single-threaded here; nothing else reads the environment.
            unsafe { std::env::remove_var(k) };
        }
        let base = gpu.build_lowered(&gemm_nvfp4(&g, backend)?)?;
        for (k, v) in &saved {
            // SAFETY: as above.
            unsafe { std::env::set_var(k, v) };
        }

        let m = tokens;
        let x: Buffer = gpu.upload(
            &(0..m * k)
                .map(|i| f16::from_f32(((i * 7) % 13) as f32 * 0.01 - 0.06))
                .collect::<Vec<_>>(),
        );
        let sets: Vec<[Buffer; 3]> = (0..copies)
            .map(|c| {
                [
                    gpu.upload(
                        &(0..n * k / 2)
                            .map(|i| (i.wrapping_mul(97) + c) as u8)
                            .collect::<Vec<_>>(),
                    ),
                    gpu.upload(
                        &(0..n * k / 16)
                            .map(|i| 0x30u8 + (i % 7) as u8)
                            .collect::<Vec<_>>(),
                    ),
                    gpu.upload(&vec![1.0f32; n]),
                ]
            })
            .collect();
        let r = gpu.upload(
            &(0..m * n)
                .map(|i| ((i * 3) % 11) as f32 * 0.1)
                .collect::<Vec<_>>(),
        );
        let (y, y0) = (gpu.zeroed::<f32>(m * n), gpu.zeroed::<f32>(m * n));
        fn bound<'a>(
            x: &'a Buffer,
            [q, s, gs]: &'a [Buffer; 3],
            r: Option<&'a Buffer>,
            out: &'a Buffer,
        ) -> Vec<&'a Buffer> {
            let mut b = vec![x, q, s, gs];
            b.extend(r);
            b.push(out);
            b
        }
        let res = residual.then_some(&r);
        let bind = |i: usize, out| bound(&x, &sets[i % copies], res, out);

        // Agreement first, on copy 0.
        let (b1, b0) = (bind(0, &y), bind(0, &y0));
        gpu.run_launches(&[(&pipe, b1.as_slice(), None)]);
        gpu.run_launches(&[(&base, b0.as_slice(), None)]);
        let (mut got, mut want) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        gpu.download(&y, &mut got);
        gpu.download(&y0, &mut want);
        let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;

        let binds: Vec<Vec<&Buffer>> = (0..copies * 4).map(|i| bind(i, &y)).collect();
        let steps: Vec<Step<'_>> = binds.iter().map(|b| (&pipe, b.as_slice(), None)).collect();
        gpu.run_launches(&steps);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let (_, gpu_s) = gpu.run_launches(&steps);
            best = best.min(gpu_s / steps.len() as f64);
        }
        println!(
            "{label:<10} {n:>6} {k:>6} {:>9.3} {:>8.2} {:>12}",
            best * 1e3,
            2.0 * (m * n * k) as f64 / best / 1e12,
            format!("{err:.1e}")
        );
        if err > 1e-2 {
            return Err(format!(
                "{label}: the schedule disagrees with the default ({err:e} of scale)"
            ));
        }
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn run(
    _: usize,
    _: &[(&str, usize, usize, bool, bool)],
    _: impl Fn(usize, usize, bool, bool) -> lex_msl::gemm::Gemm,
) -> Result<(), String> {
    Err("gemm_bench needs a Metal or CUDA device".into())
}
