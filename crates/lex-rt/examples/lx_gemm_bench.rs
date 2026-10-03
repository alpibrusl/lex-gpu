//! The `.lx` NVFP4 GEMM against the hand-scheduled one, at Qwen3.8's four
//! prefill shapes, on whichever device this machine has.
//!
//!     cargo run --release -p lex-rt --example lx_gemm_bench -- --tokens 512
//!     ... -- --sched bm=64,bn=128,bk=32,warps=2x4,pad=8,threads=256   # try a schedule
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
    let mut sched = unit.schedule_for(target.name)?.clone();
    // `--sched key=value,...` overrides the file's schedule, to try one
    // without editing it.
    if let Some(spec) = args
        .iter()
        .position(|a| a == "--sched")
        .and_then(|i| args.get(i + 1))
    {
        for kv in spec.split(',') {
            let (k, v) = kv
                .split_once('=')
                .ok_or(format!("`{kv}` is not key=value"))?;
            let num = |v: &str| {
                v.parse::<usize>()
                    .map_err(|_| format!("`{v}` is not a number"))
            };
            match k {
                "threads" => sched.threads = num(v)?,
                "pad" => sched.pad = Some(num(v)?),
                "warps" => {
                    let (r, c) = v.split_once('x').ok_or("warps=RxC")?;
                    sched.warps = Some((num(r)?, num(c)?));
                }
                tile => match sched.extents.iter_mut().find(|(n, _)| n == tile) {
                    Some(e) => e.1 = num(v)?,
                    None => return Err(format!("`{tile}` is not a schedule key")),
                },
            }
        }
    }

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
        let mut consts = vec![("m", m as f64), ("n", n as f64), ("k", k as f64)];
        consts.extend(sched.extents.iter().map(|(t, v)| (t.as_str(), *v as f64)));
        let prog = unit.algo.build_with(&consts, None)?;
        let threads = sched.threads;
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
        // An error of zero between two all-zero outputs would be no
        // evidence, so say how big the answer is.
        if scale < 1e-3 {
            return Err(format!("{label}: the outputs are all zero"));
        }
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;

        // Alternate the two kernels and keep each one's best, so whatever
        // else the GPU is doing lands on both and the ratio survives it.
        fn steps_for<'a>(
            x: &'a Buffer,
            sets: &'a [[Buffer; 3]],
            out: &'a Buffer,
        ) -> Vec<Vec<&'a Buffer>> {
            (0..sets.len() * 4)
                .map(|i| bound(x, &sets[i % sets.len()], out))
                .collect()
        }
        let (binds_h, binds_l) = (steps_for(&x, &sets, &yh), steps_for(&x, &sets, &yl));
        let steps_h: Vec<Step<'_>> = binds_h
            .iter()
            .map(|b| (&hand, b.as_slice(), None))
            .collect();
        let steps_l: Vec<Step<'_>> = binds_l.iter().map(|b| (&lx, b.as_slice(), None)).collect();
        gpu.run_launches(&steps_h);
        gpu.run_launches(&steps_l);
        let (mut th, mut tl) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..reps * 3 {
            th = th.min(gpu.run_launches(&steps_h).1 / steps_h.len() as f64);
            tl = tl.min(gpu.run_launches(&steps_l).1 / steps_l.len() as f64);
        }
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
