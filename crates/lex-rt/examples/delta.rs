//! Throughput of Qwen3.5's gated-delta state update.
//!
//! Each of the model's 48 linear-attention layers reads and writes a
//! 3 MB state once per token, so a step moves ~288 MB through this kernel
//! whatever the context length. GB/s counts the state twice (read and
//! written); everything else is noise.
//!
//! cargo run --release -p lex-rt --example delta

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use lex_front::qwen::DeltaNet;
    use lex_metal::{Buffer, Gpu, Step};
    use lex_msl::program::lower;

    let gpu = Gpu::open()?;
    let (hv, dk, dv) = (48usize, 128usize, 128usize);
    let layers = 48;
    println!(
        "{}: Qwen3.5 gated delta, {hv} heads of {dv}x{dk}",
        gpu.info().name
    );
    println!(
        "  {:>5} {:>8} {:>10} {:>9} {:>9}",
        "rows", "threads", "us/layer", "GB/s", "ms/token"
    );
    for rows in [8usize] {
        for threads in [128usize, 256] {
            let c = DeltaNet::packed(hv, 16, dk, dv, rows);
            let prog = c.build_step()?;
            lex_front::check(&prog, gpu.target()).map_err(|e| format!("{e:?}"))?;
            let pipe = gpu.build_lowered(&lower(&prog, gpu.target(), threads)?)?;
            // One state per layer, so the kernel reads cold memory as a
            // real step does rather than the same 3 MB over and over.
            let states: Vec<Buffer> = (0..layers)
                .map(|_| gpu.zeroed::<f32>(hv * dv * dk))
                .collect();
            let q = gpu.zeroed::<f32>(hv * dk);
            let k = gpu.zeroed::<f32>(hv * dk);
            let v = gpu.zeroed::<f32>(hv * dv);
            let g = gpu.zeroed::<f32>(hv * dv);
            let b = gpu.zeroed::<f32>(hv * dv);
            let y = gpu.zeroed::<f32>(hv * dv);
            let binds: Vec<Vec<&Buffer>> = states
                .iter()
                .map(|s| vec![s, &q, &k, &v, &g, &b, &y])
                .collect();
            let steps: Vec<Step<'_>> = binds.iter().map(|b| (&pipe, b.as_slice(), None)).collect();
            gpu.run_launches(&steps);
            let best = (0..3)
                .map(|_| gpu.run_launches(&steps).1)
                .fold(f64::INFINITY, f64::min);
            let per = best / layers as f64;
            let bytes = 2.0 * (hv * dv * dk * 4) as f64;
            println!(
                "  {rows:>5} {threads:>8} {:>10.1} {:>9.0} {:>9.2}",
                per * 1e6,
                bytes / per / 1e9,
                best * 1e3
            );

            // The same work as a batch: the state is read and written once
            // for the whole batch, which is what a speculative verify pays.
            for tokens in [2usize, 4] {
                let batch = c.build_steps(tokens)?;
                let pipe = gpu.build_lowered(&lower(&batch, gpu.target(), threads)?)?;
                let q = gpu.zeroed::<f32>(tokens * hv * dk);
                let k = gpu.zeroed::<f32>(tokens * hv * dk);
                let v = gpu.zeroed::<f32>(tokens * hv * dv);
                let g = gpu.zeroed::<f32>(tokens * hv * dv);
                let b = gpu.zeroed::<f32>(tokens * hv * dv);
                let y = gpu.zeroed::<f32>(tokens * hv * dv);
                let binds: Vec<Vec<&Buffer>> = states
                    .iter()
                    .map(|s| vec![s, &q, &k, &v, &g, &b, &y])
                    .collect();
                let steps: Vec<Step<'_>> =
                    binds.iter().map(|b| (&pipe, b.as_slice(), None)).collect();
                gpu.run_launches(&steps);
                let best = (0..3)
                    .map(|_| gpu.run_launches(&steps).1)
                    .fold(f64::INFINITY, f64::min);
                println!(
                    "  {:>5} {threads:>8} {:>10.1} {:>9.0} {:>9.2}   ({tokens} tokens: {:.2} ms/token)",
                    rows,
                    best / layers as f64 * 1e6,
                    bytes / (best / layers as f64) / 1e9,
                    best * 1e3,
                    best * 1e3 / tokens as f64
                );
            }
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("delta needs a Metal device");
    std::process::exit(1);
}
