//! Where a decode step's time goes, per call site.
//!
//! Decodes `--tokens` positions twice. First the normal way, one command
//! buffer per token: that is the speed. Then one dispatch at a time, each
//! timed: that is where the time goes, at the price of a CPU round trip per
//! dispatch that the first pass does not pay.
//!
//! `--context N` first fills the cache with N positions (untimed, batched),
//! so the steps measure decode at that context length.
//!
//! cargo run --release -p lex-rt --example profile -- --model llama3.1:8b --tokens 32
//! cargo run --release -p lex-rt --example profile -- --model llama3.1:8b --context 512
//!
//! `--skip rmsnorm,rope` leaves those call sites out of every step (wrong
//! logits, real timing): the drop in ms/token is what they cost on the
//! overlapped critical path, which the per-dispatch pass overstates.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use std::time::Instant;

    use lex_rt::gguf::ollama_model;
    use lex_rt::llama::{Runner, Weights};

    let mut model = "llama3.2:1b".to_string();
    let mut tokens = 32usize;
    let mut context = 0usize;
    let mut skip = vec![];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--tokens" => tokens = val()?.parse().map_err(|_| "bad --tokens")?,
            "--context" => context = val()?.parse().map_err(|_| "bad --context")?,
            "--skip" => skip = val()?.split(',').map(String::from).collect(),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let t = Instant::now();
    let w = Weights::load(&ollama_model(&model)?, context + tokens)?;
    let mut rt = Runner::new(&w)?;
    rt.skip = skip;
    println!(
        "{model}: loaded and compiled in {:.1} s on {}",
        t.elapsed().as_secs_f64(),
        rt.device()
    );

    for (pass, sync) in [
        ("one command buffer per token", false),
        (
            "per dispatch, GPU time from command-buffer timestamps",
            true,
        ),
    ] {
        rt.sync = sync;
        rt.reset();
        let fill: Vec<u32> = (0..context).map(|i| (500 + i % 20000) as u32).collect();
        rt.prefill(&fill, 16)?;
        rt.clear_profile();
        let d0 = rt.dispatches;
        let t = Instant::now();
        for i in 0..tokens {
            rt.step((1000 + i) as u32)?;
        }
        let total = t.elapsed().as_secs_f64();
        let per = total / tokens as f64;
        println!(
            "\n{pass}, context {context}..{}: {:.1} ms/token ({:.1} tok/s), {} dispatches/token, \
             {:.2} GB of weights/token = {:.0} GB/s effective",
            context + tokens,
            1e3 * per,
            1.0 / per,
            (rt.dispatches - d0) / tokens,
            w.bytes_per_token() as f64 / 1e9,
            w.bytes_per_token() as f64 / 1e9 / per
        );
        println!(
            "  {:<22} {:>7} {:>10} {:>9} {:>7}",
            "call site", "calls", "ms/token", "us/call", "of sum"
        );
        let prof = rt.profile();
        let sum: f64 = prof.iter().map(|p| p.2).sum();
        for (label, n, s) in prof {
            println!(
                "  {label:<22} {n:>7} {:>10.2} {:>9.1} {:>6.1}%",
                1e3 * s / tokens as f64,
                1e6 * s / n as f64,
                100.0 * s / sum
            );
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("profile needs a Metal device");
    std::process::exit(1);
}
