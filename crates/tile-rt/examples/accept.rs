//! How often would llama3.2:1b guess llama3.1:8b's next greedy token?
//!
//! The acceptance rate that decides what speculative decoding (1B drafts,
//! 8B verifies) could buy. Both models share Llama 3's tokenizer. The 8B
//! greedy-decodes each prompt; the 1B is then fed the same tokens and its
//! greedy guess at every position is compared with the 8B's actual token.
//!
//! cargo run --release -p tile-rt --example accept

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use tile_rt::gguf::ollama_model;
    use tile_rt::llama::{Runner, Weights};

    const STEPS: usize = 64;
    // Prompts from the checked-in goldens (already tokenised, BOS first).
    let golden = include_str!("../tests/data/llama31_8b_golden.txt");
    let prompts: Vec<Vec<u32>> = golden
        .lines()
        .filter_map(|l| l.strip_prefix("case "))
        .map(|l| l.split_whitespace().map(|t| t.parse().unwrap()).collect())
        .collect();
    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b])).unwrap() as u32;

    let big = Weights::load(&ollama_model("llama3.1:8b")?, 128)?;
    let small = Weights::load(&ollama_model("llama3.2:1b")?, 128)?;
    let mut target = Runner::new(&big)?;
    let mut draft = Runner::new(&small)?;

    let (mut hits, mut total) = (0usize, 0usize);
    let mut runs = vec![]; // lengths of consecutive-hit runs, for k-token drafts
    for p in &prompts {
        target.reset();
        let mut logits = vec![];
        for &t in p {
            logits = target.step(t)?;
        }
        let mut seq = vec![];
        for _ in 0..STEPS {
            let t = argmax(&logits);
            seq.push(t);
            logits = target.step(t)?;
        }
        draft.reset();
        let mut dl = vec![];
        for &t in p {
            dl = draft.step(t)?;
        }
        let mut run = 0;
        for &t in &seq {
            let hit = argmax(&dl) == t;
            hits += hit as usize;
            total += 1;
            if hit {
                run += 1;
            } else {
                runs.push(run);
                run = 0;
            }
            dl = draft.step(t)?;
        }
        runs.push(run);
    }
    let a = hits as f64 / total as f64;
    println!(
        "acceptance: 1B guesses the 8B's next token {hits}/{total} = {:.0}%",
        100.0 * a
    );
    // Expected tokens per 8B pass with k drafted tokens, and the speedup it
    // buys if verifying k+1 tokens costs one decode step (bandwidth-bound)
    // and each draft step costs the 1B's decode time.
    let (t8, t1) = (1.0 / 60.0, 1.0 / 214.0); // measured tile decode, s/token
    println!("{:>3} {:>16} {:>12}", "k", "tokens/8B pass", "speedup");
    for k in 1..=6 {
        let e = (1.0 - a.powi(k + 1)) / (1.0 - a);
        let per_round = k as f64 * t1 + t8;
        println!("{k:>3} {e:>16.2} {:>11.2}x", e * t8 / per_round);
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {}
