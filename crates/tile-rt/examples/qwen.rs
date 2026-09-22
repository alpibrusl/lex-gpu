//! Decode with Qwen3.5 on tile kernels.
//!
//! cargo run --release -p tile-rt --example qwen -- --steps 8

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use std::time::Instant;
    use tile_rt::qwen_run::Runner;

    let mut model = "qwen3.8:27b-mlx".to_string();
    let mut steps = 8usize;
    let mut ids = vec![760u32, 6511, 314, 9338, 369]; // "The capital of France is"
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--steps" => steps = val()?.parse().map_err(|_| "bad --steps")?,
            "--ids" => {
                ids = val()?
                    .split(',')
                    .map(|s| s.trim().parse().map_err(|_| "bad --ids".to_string()))
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    let t = Instant::now();
    let mut rt = Runner::load(&model, ids.len() + steps + 8)?;
    println!(
        "{model}: loaded and compiled in {:.1} s on {}",
        t.elapsed().as_secs_f64(),
        rt.device()
    );

    let mut logits = vec![];
    for &id in &ids {
        logits = rt.step(id)?;
    }
    let t = Instant::now();
    for i in 0..steps {
        let (best, lp) = top(&logits);
        println!("  {i:2}  token {best:6}  logprob {lp:+.4}");
        logits = rt.step(best)?;
    }
    let per = t.elapsed().as_secs_f64() / steps as f64;
    println!("{:.1} ms/token ({:.1} tok/s)", 1e3 * per, 1.0 / per);

    // Where the time goes, per call site, from the command buffer's own
    // timestamps. Each dispatch waits, so the step itself is slower here.
    rt.sync = true;
    rt.clear_profile();
    let n = 4;
    for _ in 0..n {
        logits = rt.step(top(&logits).0)?;
    }
    println!(
        "\n  {:<20} {:>7} {:>10} {:>9}",
        "call site", "calls", "ms/token", "of sum"
    );
    let prof = rt.profile();
    let sum: f64 = prof.iter().map(|p| p.2).sum();
    for (label, calls, secs) in prof {
        println!(
            "  {label:<20} {:>7} {:>10.2} {:>8.1}%",
            calls / n,
            1e3 * secs / n as f64,
            100.0 * secs / sum
        );
    }
    Ok(())
}

/// The likeliest token and its log-probability.
#[cfg(target_os = "macos")]
fn top(logits: &[f32]) -> (u32, f32) {
    let best = (0..logits.len())
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
        .unwrap_or(0);
    let max = logits[best];
    let sum: f32 = logits.iter().map(|x| (x - max).exp()).sum();
    (best as u32, -sum.ln())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen needs a Metal device");
    std::process::exit(1);
}
