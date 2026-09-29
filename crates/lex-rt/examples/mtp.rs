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

#[cfg(any(target_os = "macos", target_os = "linux"))]
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

    // `prefill` feeds the prompt batched *and* runs the draft head over
    // it, so the head's cache ends up holding the same history the model's
    // does. LEX_NOWARM feeds it a token at a time instead, which is what
    // left the head drafting from position 0 at a context of 1440.
    fn fill(rt: &mut Runner, ids: &[u32]) -> Result<Vec<f32>, String> {
        if std::env::var_os("LEX_NOWARM").is_some() {
            let mut logits = vec![];
            for &t in ids {
                logits = rt.step(t)?;
            }
            return Ok(logits);
        }
        rt.prefill(ids)
    }

    let mut logits = fill(&mut rt, &ids)?;

    // The draft head's attention cache only advances when it drafts, so
    // after a prompt fed with `step` it is empty while the model is deep
    // into a sequence: it guesses from a state the text never passed
    // through. LEX_WARM drafts (and throws away) this many tokens first,
    // to find out how much history the head actually needs -- if a short
    // window recovers acceptance, warming is cheap; if only the whole
    // prompt does, the head has to be run over the prompt properly.
    let warm: usize = std::env::var("LEX_WARM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    for _ in 0..warm {
        let next = argmax(&logits);
        rt.draft(1, next)?;
        logits = rt.step(next)?;
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

    // One line per round: whether the first draft was right.
    //
    // `actual` is the model's own greedy continuation and does not depend
    // on the drafts at all, so two runs over the same ids predict exactly
    // the same positions. That makes the comparison paired, and a paired
    // test settles in 191 rounds what an unpaired one leaves at 1.5 sigma.
    if std::env::var_os("LEX_PAIRS").is_some() {
        for (s, d) in drafts.iter().enumerate() {
            if let (Some(&g), Some(&want)) = (d.first(), actual.get(s + 1)) {
                println!("PAIR {s} {} {g} {want}", u8::from(g == want));
            }
        }
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
    //
    // One untimed speculation first. A batch size's kernels compile the
    // first time it is used, and on CUDA that is an NVRTC compile of every
    // kernel -- seconds, inside a timed loop of a few seconds -- which is
    // how speculation came to measure 0.69x there while the verify's own
    // kernels cost less than one step.
    rt.reset();
    let logits = fill(&mut rt, &ids)?;
    rt.speculate(argmax(&logits), depth)?;
    rt.reset();
    let logits = fill(&mut rt, &ids)?;
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

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    eprintln!("this example needs a Metal or CUDA device");
}
