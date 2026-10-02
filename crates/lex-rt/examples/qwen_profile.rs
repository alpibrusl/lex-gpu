//! Where a Qwen3.8 decode step's time goes, per call site -- and what a
//! speculative verify costs against it. Metal or CUDA.
//!
//!     cargo run --release -p lex-rt --example qwen_profile -- --tokens 32
//!     cargo run --release -p lex-rt --example qwen_profile -- --context 1024 --verify 3
//!     cargo run --release -p lex-rt --example qwen_profile -- --prefill-only --context 16000
//!
//! Three passes over the same positions:
//!
//! 1. **Normal decode**, one submission per token: the speed.
//! 2. **Timed decode**, every dispatch timed and booked to its call site.
//!    On CUDA the launches queue as in pass 1, with an event between each,
//!    so the times sum to about the real step. On Metal each dispatch is
//!    its own command buffer, which serialises what the concurrent encoder
//!    overlaps, so there the times are upper bounds and sum to more.
//! 3. **Verify**: one batched forward of `--verify` tokens against a single
//!    step, both normal, as a ratio. Speculation pays only while that
//!    ratio stays well under the tokens it commits per cycle. Then the
//!    batch timed per call site, so the part that grew is named.
//! 5. **Prefill**: `--prefill` tokens (512 by default) in chunks of the
//!    largest batch, warmed first, as tokens per second. This is the batch
//!    kernel at its widest, where register pressure bites first.
//! 4. **Decode through the batch path**: the same positions as pass 1, one
//!    token at a time through `forward` instead of `step`. On an L4 a
//!    verify of three cost 0.97 of a step, so the batched kernels may beat
//!    the decode ones even at one token; this says whether, and where.

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn main() -> Result<(), String> {
    use std::time::Instant;

    use lex_rt::qwen_run::Runner;

    let mut prefill_only = false;
    let (mut model, mut tokens, mut context, mut verify, mut pre) = (
        "qwen3.8:27b-mlx".to_string(),
        32usize,
        0usize,
        3usize,
        512usize,
    );
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--tokens" => tokens = val()?.parse().map_err(|_| "bad --tokens")?,
            "--context" => context = val()?.parse().map_err(|_| "bad --context")?,
            "--verify" => verify = val()?.parse().map_err(|_| "bad --verify")?,
            "--prefill" => pre = val()?.parse().map_err(|_| "bad --prefill")?,
            "--prefill-only" => prefill_only = true,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    let t = Instant::now();
    // Room for the context, three passes of steps, and the verify rounds.
    let mut rt = Runner::load(
        &model,
        (context + 3 * tokens + 16 * verify + 64).max(context + pre + 16),
    )?;
    println!(
        "{model}: loaded in {:.1} s on {}",
        t.elapsed().as_secs_f64(),
        rt.device()
    );
    // Ordinary token ids, cycling, so no step is a special case.
    let fill: Vec<u32> = (0..context).map(|i| (500 + i % 20000) as u32).collect();
    let start = |rt: &mut Runner| -> Result<(), String> {
        rt.reset();
        if !fill.is_empty() {
            rt.prefill(&fill)?;
        }
        Ok(())
    };
    'decode: {
        if prefill_only {
            break 'decode;
        }
        // One untimed step first: the first dispatch of a pipeline pays for
        // things the rest do not.
        start(&mut rt)?;
        rt.step(1000)?;

        // 1. The speed.
        rt.sync = false;
        start(&mut rt)?;
        let t = Instant::now();
        for i in 0..tokens {
            rt.step((1000 + i) as u32)?;
        }
        let step_s = t.elapsed().as_secs_f64() / tokens as f64;
        println!(
            "\ndecode at context {context}: {:.2} ms/token ({:.1} tok/s)",
            1e3 * step_s,
            1.0 / step_s
        );

        // 2. Where it goes.
        rt.sync = true;
        start(&mut rt)?;
        rt.clear_profile();
        for i in 0..tokens {
            rt.step((1000 + i) as u32)?;
        }
        table(&rt.profile(), tokens, step_s);

        // 3. The verify.
        rt.sync = false;
        start(&mut rt)?;
        let rounds = 8;
        let batch: Vec<u32> = (0..verify).map(|i| (2000 + i) as u32).collect();
        // A batch size's kernels compile the first time it is used -- on CUDA
        // an NVRTC compile of every kernel, seconds of it -- so the first
        // forward is not a forward. Timing it made a verify of three look like
        // 5.27 steps on an L4 whose kernels summed to less than one.
        rt.forward(&batch, true)?;
        let (mut one, mut many) = (0.0, 0.0);
        for _ in 0..rounds {
            let t = Instant::now();
            rt.step(1500)?;
            one += t.elapsed().as_secs_f64();
            let t = Instant::now();
            rt.forward(&batch, true)?;
            many += t.elapsed().as_secs_f64();
        }
        println!(
            "\nverify of {verify}: {:.2} ms against a step's {:.2} ms = {:.2} steps",
            1e3 * many / rounds as f64,
            1e3 * one / rounds as f64,
            many / one
        );
        rt.sync = true;
        rt.clear_profile();
        for _ in 0..rounds {
            rt.forward(&batch, true)?;
        }
        table(&rt.profile(), rounds, many / rounds as f64);

        // 4. Decode, one token at a time, through the batch path.
        rt.sync = false;
        start(&mut rt)?;
        rt.forward(&[1000], false)?;
        start(&mut rt)?;
        let t = Instant::now();
        for i in 0..tokens {
            rt.forward(&[(1000 + i) as u32], false)?;
        }
        let one_s = t.elapsed().as_secs_f64() / tokens as f64;
        println!(
            "\ndecode via the batch path: {:.2} ms/token ({:.1} tok/s) against step's {:.2} = {:.2}x",
            1e3 * one_s,
            1.0 / one_s,
            1e3 * step_s,
            step_s / one_s
        );
        rt.sync = true;
        start(&mut rt)?;
        rt.clear_profile();
        for i in 0..tokens {
            rt.forward(&[(1000 + i) as u32], false)?;
        }
        table(&rt.profile(), tokens, one_s);
    }

    // 5. Prefill, warmed: the first chunk of each size compiles.
    // After `--context` tokens, so the attention reads that much cache: a
    // lex-code turn is a few hundred to a few thousand new tokens at 10-30K.
    rt.sync = false;
    rt.reset();
    rt.compile_batches()?;
    let ids: Vec<u32> = (0..pre).map(|i| (700 + (i * 37) % 20000) as u32).collect();
    rt.prefill(&ids[..16.min(pre)])?;
    start(&mut rt)?;
    let t = Instant::now();
    rt.prefill(&ids)?;
    let s = t.elapsed().as_secs_f64();
    println!(
        "\nprefill of {pre} after {context}: {:.0} ms = {:.1} tok/s",
        1e3 * s,
        pre as f64 / s
    );
    // And where it goes: the chunk size changes every kernel's shape, not
    // only the matmuls', so a size that is slower has to be taken apart.
    rt.sync = true;
    start(&mut rt)?;
    rt.clear_profile();
    rt.prefill(&ids)?;
    table(&rt.profile(), 1, s);
    Ok(())
}

/// Per call site, per pass: calls, ms, share of the timed sum. The sum is
/// printed next to the untimed pass, because how far apart they are says
/// how much to trust the split.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn table(prof: &[(&'static str, usize, f64)], passes: usize, real_s: f64) {
    let sum: f64 = prof.iter().map(|p| p.2).sum();
    println!(
        "  timed sum {:.2} ms per pass against {:.2} ms untimed",
        1e3 * sum / passes as f64,
        1e3 * real_s
    );
    println!(
        "  {:<24} {:>7} {:>9} {:>9} {:>7}",
        "call site", "calls", "ms/pass", "us/call", "of sum"
    );
    for (label, n, s) in prof {
        println!(
            "  {label:<24} {:>7} {:>9.3} {:>9.1} {:>6.1}%",
            n / passes,
            1e3 * s / passes as f64,
            1e6 * s / *n as f64,
            100.0 * s / sum
        );
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    eprintln!("qwen_profile needs a Metal or CUDA device");
}
