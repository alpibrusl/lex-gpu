//! The `.lx` NVFP4 GEMM against the hand-scheduled one, at Qwen3.8's four
//! prefill shapes, on whichever device this machine has.
//!
//!     cargo run --release -p lex-rt --example lx_gemm_bench -- --tokens 512
//!
//! This is the test of the claim that `lex-gpu` is a language: the same
//! numbers, from `crates/lex-front/lx/gemm_fp4.lx` and a schedule, at the
//! speed of `lex_msl::gemm`. Both kernels get the same inputs (no residual:
//! the `.lx` has no epilogue yet), the outputs must agree, and each is timed
//! over several copies of its weights so they stream from memory as in the
//! model. The last column is the language's time over the hand-written
//! kernel's: 1.00 is parity, above it is the cost of the language.

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn main() -> Result<(), String> {
    use half::f16;
    use lex_msl::dialect::{Cuda, Dialect, Msl};
    use lex_msl::gemm::{Backend, Gemm, gemm_nvfp4};
    use lex_msl::program::{Sched, lower_sched};
    use lex_rt::dev::{Buffer, Gpu, Step};

    let args: Vec<String> = std::env::args().collect();
    let tokens: usize = args
        .iter()
        .position(|a| a == "--tokens")
        .and_then(|i| args.get(i + 1)?.parse().ok())
        .unwrap_or(512);
    let shapes = [
        ("gate/up", 17408usize, 5120usize),
        ("down", 5120, 17408),
        ("qkv", 10240, 5120),
        ("out_proj", 5120, 6144),
    ];

    let backend = lex_rt::dev::gemm_backend();
    let (target, dialect): (lex_ir::Target, &dyn Dialect) = match backend {
        Backend::Metal => (lex_ir::Target::apple_m_series(), &Msl),
        Backend::Cuda => (lex_ir::Target::nvidia_ada(), &Cuda),
    };
    let unit = lex_front::syntax::parse(include_str!("../../lex-front/lx/gemm_fp4.lx"))?;
    let sched = unit.schedule_for(target.name)?;

    let gpu = Gpu::open()?;
    let (copies, reps) = (4usize, 5usize);
    println!("{}, {tokens} tokens, schedule {:?}", gpu.info().name, sched);
    println!(
        "{:<10} {:>6} {:>6} {:>10} {:>8} {:>10} {:>8} {:>8} {:>8}",
        "shape", "n", "k", "hand ms", "TFLOPS", "lx ms", "TFLOPS", "error", "lx/hand"
    );
    for (label, n, k) in shapes {
        let m = tokens;
        let hand = gpu.build_lowered(&gemm_nvfp4(
            &Gemm {
                m,
                n,
                k,
                residual: false,
                x_half: true,
            },
            backend,
        )?)?;
        let (prog, threads) = unit.compile(
            target.name,
            &[("m", m as f64), ("n", n as f64), ("k", k as f64)],
        )?;
        lex_front::check(&prog, &target).map_err(|e| format!("{label}: {e:#?}"))?;
        let lx = gpu.build_lowered(&lower_sched(
            &prog,
            &target,
            dialect,
            &Sched {
                threads,
                warps: sched.warps,
                pad: sched.pad,
            },
        )?)?;

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
        let (yh, yl) = (gpu.zeroed::<f32>(m * n), gpu.zeroed::<f32>(m * n));
        fn bound<'a>(
            x: &'a Buffer,
            [q, s, gs]: &'a [Buffer; 3],
            out: &'a Buffer,
        ) -> Vec<&'a Buffer> {
            vec![x, q, s, gs, out]
        }

        let (bh, bl) = (bound(&x, &sets[0], &yh), bound(&x, &sets[0], &yl));
        gpu.run_launches(&[(&hand, bh.as_slice(), None)]);
        gpu.run_launches(&[(&lx, bl.as_slice(), None)]);
        let (mut want, mut got) = (vec![0.0f32; m * n], vec![0.0f32; m * n]);
        gpu.download(&yh, &mut want);
        gpu.download(&yl, &mut got);
        let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;

        let time = |pipe, out: &Buffer| -> f64 {
            let binds: Vec<Vec<&Buffer>> = (0..copies * 4)
                .map(|i| bound(&x, &sets[i % copies], out))
                .collect();
            let steps: Vec<Step<'_>> = binds.iter().map(|b| (pipe, b.as_slice(), None)).collect();
            gpu.run_launches(&steps);
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let (_, s) = gpu.run_launches(&steps);
                best = best.min(s / steps.len() as f64);
            }
            best
        };
        let (th, tl) = (time(&hand, &yh), time(&lx, &yl));
        let tflops = |t: f64| 2.0 * (m * n * k) as f64 / t / 1e12;
        println!(
            "{label:<10} {n:>6} {k:>6} {:>10.3} {:>8.2} {:>10.3} {:>8.2} {:>8} {:>8.2}",
            th * 1e3,
            tflops(th),
            tl * 1e3,
            tflops(tl),
            format!("{err:.1e}"),
            tl / th
        );
        if err > 1e-2 {
            return Err(format!(
                "{label}: the `.lx` kernel disagrees with the hand-written one ({err:e} of scale)"
            ));
        }
    }
    Ok(())
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    eprintln!("lx_gemm_bench needs a Metal or CUDA device");
}
