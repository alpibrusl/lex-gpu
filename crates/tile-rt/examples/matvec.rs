//! Bandwidth of each decode matvec the 8B runs, in isolation.
//!
//! Every weight set is allocated several times over (more than the GPU's
//! system cache holds) and the dispatches rotate through them, so each one
//! reads its weights from DRAM as in a real step. GB/s counts weight bytes
//! only; the input vector and output are noise at these sizes.
//!
//! cargo run --release -p tile-rt --example matvec
//! BO=16 THREADS=128 cargo run --release -p tile-rt --example matvec

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use tile_front::llama::{QLayout, matmul_q, matvec_q};
    use tile_ir::DType;
    use tile_metal::{Buffer, Gpu, Step};
    use tile_msl::program::lower;

    let env = |k: &str, d: usize| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let (bo, threads) = (env("BO", 8), env("THREADS", 256));
    let gpu = Gpu::open()?;
    println!("{}: BO={bo} THREADS={threads}", gpu.info().name);
    println!(
        "  {:<22} {:>6} {:>6} {:>9} {:>9} {:>8}",
        "matrix", "n_in", "n_out", "MB", "us", "GB/s"
    );
    let shapes = [
        // Qwen3.5-27B (MLX, nvfp4), the decode matvecs by size.
        ("qwen gate/up (nvfp4)", 5120, 17408, QLayout::NVFP4),
        ("qwen down (nvfp4)", 17408, 5120, QLayout::NVFP4),
        ("qwen qkv-in (nvfp4)", 5120, 12288, QLayout::NVFP4),
        ("qwen lm head (nvfp4)", 5120, 248320, QLayout::NVFP4),
        // NVFP4 with a 32-value scale group (not the real layout): does
        // the scale's frequency explain the gap to Q4_K?
        (
            "qwen gate/up (fp4 g32)",
            5120,
            17408,
            QLayout {
                group: 32,
                ..QLayout::NVFP4
            },
        ),
        // The same shapes in Q4_K, to separate format from shape.
        ("qwen gate/up (Q4_K)", 5120, 17408, QLayout::Q4_K),
        ("qwen down (Q4_K)", 17408, 5120, QLayout::Q4_K),
        // Llama-3 8B (Q4_K_M).
        ("q/o (Q4_K)", 4096, 4096, QLayout::Q4_K),
        ("k (Q4_K)", 4096, 1024, QLayout::Q4_K),
        ("v (Q6_K)", 4096, 1024, QLayout::Q6_K),
        ("gate/up (Q4_K)", 4096, 14336, QLayout::Q4_K),
        ("down (Q4_K)", 14336, 4096, QLayout::Q4_K),
        ("down (Q6_K)", 14336, 4096, QLayout::Q6_K),
        ("lm head (Q6_K)", 4096, 128256, QLayout::Q6_K),
    ];
    for (label, n_in, n_out, layout) in shapes {
        let prog = matvec_q(n_in, n_out, bo, n_in, layout, false)?;
        tile_front::check(&prog, gpu.target()).map_err(|e| format!("{e:?}"))?;
        let pipe = gpu.build_lowered(&lower(&prog, gpu.target(), threads)?)?;
        let bytes = |p: &tile_front::ir::Param| {
            let n: usize = p.shape.iter().product();
            n * match p.dtype {
                DType::F32 => 4,
                DType::F16 => 2,
                DType::I8 => 1,
            }
        };
        // Parameter 0 is x, the last is y; the rest are weight data.
        let np = prog.params.len();
        let wbytes: usize = prog.params[1..np - 1].iter().map(bytes).sum();
        let copies = (1usize << 30).div_ceil(wbytes).clamp(2, 64);
        let x = gpu.zeroed::<u8>(bytes(&prog.params[0]));
        let y = gpu.zeroed::<u8>(bytes(&prog.params[np - 1]));
        // Real bytes, not `zeroed`: a buffer that has never been written
        // may read from a shared zero page rather than from DRAM, which
        // flatters the measurement.
        let filler: Vec<u8> = (0..1 << 16).map(|i| (i * 37 + 11) as u8).collect();
        let sets: Vec<Vec<Buffer>> = (0..copies)
            .map(|_| {
                prog.params[1..np - 1]
                    .iter()
                    .map(|p| {
                        let n = bytes(p);
                        let mut v = Vec::with_capacity(n);
                        while v.len() < n {
                            let take = filler.len().min(n - v.len());
                            v.extend_from_slice(&filler[..take]);
                        }
                        gpu.upload(&v)
                    })
                    .collect()
            })
            .collect();
        let binds: Vec<Vec<&Buffer>> = sets
            .iter()
            .map(|w| {
                let mut b = vec![&x];
                b.extend(w.iter());
                b.push(&y);
                b
            })
            .collect();
        let reps = (4 * copies).max(32);
        let steps: Vec<Step<'_>> = (0..reps)
            .map(|i| (&pipe, binds[i % copies].as_slice(), None))
            .collect();
        gpu.run_launches(&steps); // warm up
        let best = (0..3)
            .map(|_| gpu.run_launches(&steps).1)
            .fold(f64::INFINITY, f64::min)
            / reps as f64;
        println!(
            "  {label:<22} {n_in:>6} {n_out:>6} {:>9.1} {:>9.1} {:>8.0}",
            wbytes as f64 / 1e6,
            best * 1e6,
            wbytes as f64 / best / 1e9
        );
    }

    // The same weights against a batch of tokens: what a speculative
    // verify pays. `tok/s*` is what a whole 14.5 GB pass would reach at
    // this rate if every matvec behaved the same.
    println!();
    println!(
        "  {:<16} {:>6} {:>4} {:>6} {:>9} {:>10} {:>9}",
        "gate/up, batched", "tokens", "bo", "kc", "us", "GB/s", "ms/token"
    );
    let (n_in, n_out) = (5120usize, 17408usize);
    // One set of weights for every configuration: allocating a gigabyte
    // per configuration made the numbers wander by 3x between runs.
    let probe = matmul_q(2, n_in, n_out, 16, n_in, QLayout::NVFP4, false)?;
    let np = probe.params.len();
    let wbytes_of = |p: &tile_front::ir::Param| {
        let n: usize = p.shape.iter().product();
        n * match p.dtype {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I8 => 1,
        }
    };
    let wsizes: Vec<usize> = probe.params[1..np - 1].iter().map(wbytes_of).collect();
    let wbytes: usize = wsizes.iter().sum();
    let copies = (1usize << 30).div_ceil(wbytes).clamp(2, 32);
    let filler: Vec<u8> = (0..1 << 16).map(|i| (i * 37 + 11) as u8).collect();
    let sets: Vec<Vec<Buffer>> = (0..copies)
        .map(|_| {
            wsizes
                .iter()
                .map(|&n| {
                    let mut v = Vec::with_capacity(n);
                    while v.len() < n {
                        let take = filler.len().min(n - v.len());
                        v.extend_from_slice(&filler[..take]);
                    }
                    gpu.upload(&v)
                })
                .collect()
        })
        .collect();

    for tokens in [1usize, 2, 4] {
        let x = gpu.zeroed::<f32>(tokens * n_in);
        let y = gpu.zeroed::<f32>(tokens * n_out);
        for (bo, kc) in [
            (8, n_in),
            (16, n_in),
            (32, n_in),
            (16, 1024),
            (32, 1024),
            (16, 512),
        ] {
            let Ok(prog) = matmul_q(tokens, n_in, n_out, bo, kc, QLayout::NVFP4, false) else {
                continue;
            };
            if tile_front::check(&prog, gpu.target()).is_err() {
                continue;
            }
            let Ok(pipe) = gpu.build_lowered(&lower(&prog, gpu.target(), threads)?) else {
                continue;
            };
            let binds: Vec<Vec<&Buffer>> = sets
                .iter()
                .map(|w| {
                    let mut b = vec![&x];
                    b.extend(w.iter());
                    b.push(&y);
                    b
                })
                .collect();
            let reps = (4 * copies).max(32);
            let steps: Vec<Step<'_>> = (0..reps)
                .map(|i| (&pipe, binds[i % copies].as_slice(), None))
                .collect();
            gpu.run_launches(&steps);
            // Median of five: one slow run is the machine, not the kernel.
            let mut runs: Vec<f64> = (0..5)
                .map(|_| gpu.run_launches(&steps).1 / reps as f64)
                .collect();
            runs.sort_by(f64::total_cmp);
            let per = runs[2];
            println!(
                "  {:<16} {tokens:>6} {bo:>4} {kc:>6} {:>9.1} {:>10.0} {:>9.2}",
                "",
                per * 1e6,
                wbytes as f64 / per / 1e9,
                14.5e9 / (wbytes as f64 / per) * 1e3 / tokens as f64
            );
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("matvec needs a Metal device");
    std::process::exit(1);
}
