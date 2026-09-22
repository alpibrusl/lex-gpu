//! Throughput of the batched quantised matmul prefill needs, as lowered
//! today, against the compute Ollama's prefill implies.
//!
//! cargo run --release -p tile-rt --example gemm

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use half::f16;
    use tile_front::llama::{QLayout, matmul_q};
    use tile_metal::Gpu;
    use tile_msl::program::lower;

    let gpu = Gpu::open()?;
    let (n_in, n_out) = (4096usize, 14336usize); // Llama-3-8B gate/up
    println!("{} — Q4_K {n_out}x{n_in}", gpu.info().name);
    println!("{:>5} {:>10} {:>12} {:>10}", "T", "ms", "GFLOP/s", "tok/s*");
    for t in [1usize, 8, 32, 128] {
        let bo = 8;
        let prog = matmul_q(t, n_in, n_out, bo, n_in, QLayout::Q4_K, false)?;
        tile_front::check(&prog, gpu.target()).map_err(|e| format!("{e:?}"))?;
        let pipe = gpu.build_lowered(&lower(&prog, gpu.target(), 256)?)?;
        let x = gpu.upload(&vec![0.01f32; t * n_in]);
        let q = gpu.zeroed::<u8>(n_out * n_in / 2);
        let sc = gpu.zeroed::<u8>(n_out * n_in / 32);
        let d = gpu.upload(&vec![f16::from_f32(0.001); n_out * n_in / 256]);
        let mn = gpu.zeroed::<u8>(n_out * n_in / 32);
        let dmin = gpu.upload(&vec![f16::from_f32(0.001); n_out * n_in / 256]);
        let y = gpu.zeroed::<f32>(t * n_out);
        let bufs = [&x, &q, &sc, &d, &mn, &dmin, &y];
        let secs = gpu.time(&pipe, &bufs, 5, 3);
        let flops = 2.0 * (t * n_in * n_out) as f64;
        // Tokens/s if every matmul of the 8B ran at this rate (16 GFLOP/token).
        println!(
            "{t:>5} {:>10.3} {:>12.0} {:>10.0}",
            secs * 1e3,
            flops / secs / 1e9,
            flops / secs / 16e9
        );
    }
    println!("* if every matmul in the 8B ran at this rate; Ollama prefills ~1170 tok/s");
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
