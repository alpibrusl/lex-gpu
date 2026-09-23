//! The draft head on the GPU: what it accepts, and what it costs.
//!
//!     cargo run --release -p lex-rt --example mtp -- --steps 64 --depth 2
//!
//! At every step the model produces a token and the head drafts the next
//! `depth`; this reports how many of those drafts the model then agreed
//! with. `scripts/mtp_accept.py` measures the same thing in f32 over a
//! trace, so the two should land on the same number — that is the check
//! that these kernels implement the head the reference describes.
//!
//! The timing is what decides whether depth is worth having: a draft is
//! the head plus a pass over `lm_head`, and it buys a token only if the
//! model would have agreed anyway.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use lex_rt::qwen_run::Runner;
    use std::time::Instant;

    let mut model = "qwen3.8:27b-mlx".to_string();
    let (mut steps, mut depth) = (64usize, 1usize);
    let mut ids = vec![760u32, 6511, 314, 9338, 369]; // "The capital of France is"
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--steps" => steps = val()?.parse().map_err(|_| "bad --steps")?,
            "--depth" => depth = val()?.parse().map_err(|_| "bad --depth")?,
            "--ids" => {
                ids = val()?
                    .split(',')
                    .map(|s| s.trim().parse().map_err(|_| "bad --ids".to_string()))
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    // Decode speed against context, the way scripts/ollama_bench.py
    // measures Ollama: fill N positions, then time the tokens after them.
    // Qwen should barely move -- 48 of its 64 layers carry a fixed-size
    // state rather than a cache -- and that is a claim worth checking
    // rather than repeating.
    let context: usize = std::env::var("LEX_CONTEXT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if context > ids.len() {
        ids.extend((ids.len()..context).map(|i| 1000 + (i as u32 * 7919) % 200000));
    }
    let mut rt = Runner::load(&model, ids.len() + steps + depth + 8)?;
    if !rt.has_mtp() {
        return Err(format!("{model} carries no mtp head"));
    }
    let argmax = |v: &[f32]| {
        (0..v.len())
            .max_by(|&a, &b| v[a].total_cmp(&v[b]))
            .expect("logits") as u32
    };

    // LEX_SKIP=attention leaves that call site out, so the difference
    // says what it costs at this context. Decode on this model should
    // barely move with context -- 48 of 64 layers carry a fixed-size
    // state -- so whatever does move is in the other 16.
    if let Ok(s) = std::env::var("LEX_SKIP") {
        rt.skip = s.split(',').map(String::from).collect();
    }

    let mut logits = vec![];
    for &t in &ids {
        logits = rt.step(t)?;
    }

    // Record first, score afterwards: the draft made at step `s` is only
    // judged once the model has produced the tokens it guessed at.
    let mut actual: Vec<u32> = vec![];
    let mut drafts: Vec<Vec<u32>> = vec![];
    let (mut draft_ms, mut step_ms) = (0.0f64, 0.0f64);

    for _ in 0..steps {
        let next = argmax(&logits);
        actual.push(next);

        let t = Instant::now();
        drafts.push(rt.draft(depth, next)?);
        draft_ms += t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        logits = rt.step(next)?;
        step_ms += t.elapsed().as_secs_f64() * 1e3;
    }

    // A draft counts only while every draft before it was right, because a
    // verify takes the longest matching prefix, not a scattering of hits.
    let mut hit = vec![0usize; depth];
    let mut seen = vec![0usize; depth];
    for (s, d) in drafts.iter().enumerate() {
        for (i, &guess) in d.iter().enumerate() {
            match actual.get(s + 1 + i) {
                None => break,
                Some(&want) => {
                    seen[i] += 1;
                    if guess != want {
                        break;
                    }
                    hit[i] += 1;
                }
            }
        }
    }

    // What it is all for: the same tokens, in less time.
    rt.reset();
    let mut logits = vec![];
    for &t in &ids {
        logits = rt.step(t)?;
    }
    let mut got = 0usize;
    let mut next = argmax(&logits);
    let t = Instant::now();
    while got < steps {
        let (committed, after) = rt.speculate(next, depth)?;
        got += committed.len();
        next = after;
    }
    let spec_s = t.elapsed().as_secs_f64();

    let n = steps as f64;
    println!("{model} on {}", rt.device());
    // The expected length of an accepted run, which is what a pass buys.
    let mut run = 1.0f64;
    for i in 0..depth {
        let rate = hit[i] as f64 / seen[i].max(1) as f64;
        println!(
            "  offset {}: {}/{} = {:.1}%",
            i + 1,
            hit[i],
            seen[i].max(1),
            100.0 * rate
        );
        run += hit[i] as f64 / seen[0].max(1) as f64;
    }
    println!("  tokens a verify would accept: {run:.2} of {}", depth + 1);
    println!("  a step  : {:.2} ms", step_ms / n);
    println!(
        "  a draft : {:.2} ms ({:.1}% of a step) x depth {depth}",
        draft_ms / n / depth as f64,
        100.0 * (draft_ms / depth as f64) / step_ms
    );
    println!("  plain   : {:.1} tok/s", 1000.0 / (step_ms / n));
    println!(
        "  spec    : {:.1} tok/s  ({:.2}x) at depth {depth}",
        got as f64 / spec_s,
        (got as f64 / spec_s) / (1000.0 / (step_ms / n))
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this example needs Metal");
}
