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
    use tile_front::llama::{QLayout, matvec_q};
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
        let sets: Vec<Vec<Buffer>> = (0..copies)
            .map(|_| {
                prog.params[1..np - 1]
                    .iter()
                    .map(|p| gpu.zeroed::<u8>(bytes(p)))
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
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("matvec needs a Metal device");
    std::process::exit(1);
}
