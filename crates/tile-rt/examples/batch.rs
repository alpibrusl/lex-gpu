//! Batched forward-pass throughput: prefill at each chunk size, and what
//! verifying `k + 1` tokens costs relative to one decode step (the number
//! speculative decoding's payoff depends on).
//!
//! cargo run --release -p tile-rt --example batch -- --model llama3.1:8b

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use std::time::Instant;

    use tile_rt::gguf::ollama_model;
    use tile_rt::llama::{Runner, Weights};

    let mut model = "llama3.1:8b".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().ok_or("--model needs a value")?,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    const PROMPT: usize = 512;
    let w = Weights::load(&ollama_model(&model)?, PROMPT + 64)?;
    let mut rt = Runner::new(&w)?;
    // Arbitrary ordinary-vocabulary tokens: prefill cost does not depend on
    // which tokens they are.
    let prompt: Vec<u32> = (0..PROMPT as u32)
        .map(|i| 1000 + (i * 7919) % 50000)
        .collect();

    println!("{model} on {}", rt.device());
    println!("\nprefill, {PROMPT}-token prompt:");
    for chunk in [1usize, 2, 4, 8, 16] {
        rt.reset();
        rt.prefill(&prompt[..32], chunk)?; // compile this chunk size's kernels
        rt.reset();
        let t = Instant::now();
        rt.prefill(&prompt, chunk)?;
        let s = t.elapsed().as_secs_f64();
        println!("  chunk {chunk:>2}: {:7.0} tok/s", PROMPT as f64 / s);
    }

    println!("\nverify cost, relative to one decode step (after a {PROMPT}-token prompt):");
    let reps = 8;
    rt.reset();
    rt.prefill(&prompt, 16)?;
    let base = rt.pos();
    let t = Instant::now();
    for _ in 0..reps {
        rt.step(2000)?;
        rt.truncate(base);
    }
    let step = t.elapsed().as_secs_f64() / reps as f64;
    println!("  1 token (decode step): {:.2} ms", step * 1e3);
    for t_tok in 2..=8usize {
        let toks: Vec<u32> = (0..t_tok as u32).map(|i| 3000 + i).collect();
        rt.forward(&toks, true)?; // compile
        rt.truncate(base);
        let t = Instant::now();
        for _ in 0..reps {
            rt.forward(&toks, true)?;
            rt.truncate(base);
        }
        let s = t.elapsed().as_secs_f64() / reps as f64;
        println!(
            "  {t_tok} tokens: {:.2} ms = {:.2}x a step",
            s * 1e3,
            s / step
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
