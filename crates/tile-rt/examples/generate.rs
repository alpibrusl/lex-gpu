//! Greedy decoding with tile kernels on the GPU.
//!
//! Takes token ids (tokenisation lives in Python for now — see
//! `scripts/tile_vs_ollama.py`, which drives this and compares with Ollama):
//!
//! ```text
//! cargo run --release -p tile-rt --example generate -- \
//!     --model llama3.2:1b --ids 128000,791,6864,315,9822,374 --steps 16 --top 5
//! ```
//!
//! Output, one line per generated token, then timing:
//!
//! ```text
//! step <id> <id>:<logprob> ...      top-k, best first
//! time prefill_s <s> decode_tok_s <tok/s> dispatches <n>
//! ```

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use std::time::Instant;

    use tile_rt::gguf::ollama_model;
    use tile_rt::llama::{Runner, Weights, log_softmax};

    let mut model = "llama3.2:1b".to_string();
    let mut ids: Vec<u32> = vec![];
    let (mut steps, mut top) = (16usize, 5usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--ids" => {
                ids = val()?
                    .split(',')
                    .map(|t| t.trim().parse().map_err(|_| format!("bad token id `{t}`")))
                    .collect::<Result<_, _>>()?
            }
            "--steps" => steps = val()?.parse().map_err(|_| "bad --steps")?,
            "--top" => top = val()?.parse().map_err(|_| "bad --top")?,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    if ids.is_empty() {
        return Err("--ids is required (comma-separated token ids)".into());
    }
    let w = Weights::load(&ollama_model(&model)?, ids.len() + steps)?;
    let mut rt = Runner::new(&w)?;

    let t0 = Instant::now();
    let mut logits = vec![];
    for &t in &ids {
        logits = rt.step(t)?;
    }
    let prefill = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for i in 0..steps {
        let lp = log_softmax(&logits);
        let mut order: Vec<usize> = (0..lp.len()).collect();
        order.select_nth_unstable_by(top, |&a, &b| lp[b].total_cmp(&lp[a]));
        let mut best: Vec<usize> = order[..top].to_vec();
        best.sort_by(|&a, &b| lp[b].total_cmp(&lp[a]));
        let pairs: Vec<String> = best.iter().map(|&j| format!("{j}:{:.6}", lp[j])).collect();
        println!("step {} {}", best[0], pairs.join(" "));
        if i + 1 < steps {
            logits = rt.step(best[0] as u32)?;
        }
    }
    let decode = t1.elapsed().as_secs_f64();
    println!(
        "time prefill_s {prefill:.3} decode_tok_s {:.2} dispatches {}",
        (steps.saturating_sub(1)) as f64 / decode,
        rt.dispatches
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("generate needs a Metal device");
    std::process::exit(1);
}
