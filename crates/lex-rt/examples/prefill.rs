//! What it costs Qwen3.5 to read a prompt.
//!
//!     cargo run --release -p lex-rt --example prefill -- --tokens 512
//!
//! Decode has been measured to death in this repo; prefill has not been
//! measured at all for this model, which matters the moment a prompt is
//! longer than a sentence. `Runner::forward` takes at most `MAX_BATCH`
//! tokens, so a prompt is read in chunks of that, and this reports the rate
//! at each chunk size up to the cap.
//!
//! The comparison to want is Ollama on the same machine
//! (`scripts/ollama_bench.py`), and the shape to watch for is whether the
//! rate keeps climbing with the chunk: if it flattens early, the batched
//! path is bound by something other than weight traffic, exactly as the
//! decode matvec was.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use std::time::Instant;

    use lex_rt::qwen_run::{MAX_BATCH, Runner};

    let mut model = "qwen3.8:27b-mlx".to_string();
    let mut tokens = 512usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--tokens" => tokens = val()?.parse().map_err(|_| "bad --tokens")?,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    let mut rt = Runner::load(&model, tokens + 16)?;
    println!("{model} on {}, {tokens} tokens", rt.device());
    println!("  chunk   ms total   tok/s   ms/chunk");

    // Arbitrary but in-vocabulary; prefill cost does not depend on which
    // tokens, only how many, and a fixed set keeps runs comparable.
    let prompt: Vec<u32> = (0..tokens)
        .map(|i| 1000 + (i as u32 * 7919) % 200000)
        .collect();

    let mut chunk = 1;
    while chunk <= MAX_BATCH {
        // Larger chunks amortise the weights over more tokens, until the
        // accumulators per (row, token) stop fitting in registers.
        // Compile this chunk size's kernels *before* timing. `forward`
        // builds them on first use and caches them, and a 16-token kernel
        // is a lot of unrolled source -- left inside the loop it is the
        // measurement, not the thing measured.
        rt.reset();
        rt.forward(&prompt[..chunk], false)?;

        rt.reset();
        let t = Instant::now();
        for part in prompt.chunks(chunk) {
            rt.forward(part, false)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "  {chunk:>5}   {ms:>8.0}   {:>5.0}   {:>8.1}",
            tokens as f64 / (ms / 1e3),
            ms / prompt.chunks(chunk).count() as f64
        );
        // With LEX_SYNC, say *which* kernel the time went to. A chunk size
        // that costs more than the one below it is either a kernel that
        // does not scale or one that is wrong, and the per-call-site
        // breakdown is the only thing that tells them apart.
        if std::env::var_os("LEX_SYNC").is_some() {
            let mut prof = rt.profile();
            prof.sort_by(|a, b| b.2.total_cmp(&a.2));
            for (site, calls, ms) in prof.iter().take(6) {
                println!("          {site:<22} {calls:>5} calls {ms:>8.2} ms");
            }
            rt.clear_profile();
        }
        chunk *= 2;
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this example needs Metal");
}
