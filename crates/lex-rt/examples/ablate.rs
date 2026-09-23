//! What each call site costs Qwen's prefill *on the critical path*.
//!
//!     cargo run --release -p lex-rt --example ablate -- --tokens 256
//!
//! Run the pass, then run it again with one call site left out, and the
//! difference is what that call site costs. Nothing is serialised and
//! nothing is instrumented, so the schedule being measured is the schedule
//! that normally runs.
//!
//! This exists because the obvious tool lies. Under `LEX_SYNC` the
//! dispatches are issued one at a time, which inflates a 128-token prefill
//! at chunk 8 from 1465 ms to 5068 ms and makes the per-kernel times sum to
//! **210%** of the real elapsed time — they overlap that much, which is
//! what the concurrent encoder is for. Those numbers are still good for
//! comparing one kernel against itself across configurations. They are not
//! shares of anything, and reading them as shares is how this repository
//! spent an afternoon blaming register pressure for an activation-traffic
//! problem.
//!
//! The output is wrong while a kernel is missing. The dispatch count, the
//! shapes and the scheduling are not, and those are what is being timed.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use std::time::Instant;

    use lex_rt::qwen_run::Runner;

    let mut model = "qwen3.8:27b-mlx".to_string();
    let (mut tokens, mut chunk) = (256usize, 8usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--tokens" => tokens = val()?.parse().map_err(|_| "bad --tokens")?,
            "--chunk" => chunk = val()?.parse().map_err(|_| "bad --chunk")?,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    // Every call site the batched path dispatches, by the label prefix the
    // plan gives it.
    const SITES: &[&str] = &[
        "matvec gate/up",
        "matvec down",
        "matvec qkv",
        "matvec out_proj",
        "matvec z",
        "matvec a/b",
        "matvec q",
        "matvec k/v",
        "matvec o_proj",
        "matvec lm head",
        "rmsnorm",
        "silu_mul",
        "delta step",
        "delta q/k",
        "attention",
        "conv",
        "gates",
        "gated norm",
        "rope",
        "kv append",
        "gate mul",
    ];

    let mut rt = Runner::load(&model, tokens + 16)?;
    let prompt: Vec<u32> = (0..tokens)
        .map(|i| 1000 + (i as u32 * 7919) % 200000)
        .collect();

    // Median of three: one slow pass is the machine, not the schedule.
    let run = |rt: &mut Runner| -> Result<f64, String> {
        let mut runs = vec![];
        for _ in 0..3 {
            rt.reset();
            let t = Instant::now();
            for part in prompt.chunks(chunk) {
                rt.forward(part, false)?;
            }
            runs.push(t.elapsed().as_secs_f64() * 1e3);
        }
        runs.sort_by(f64::total_cmp);
        Ok(runs[1])
    };

    // Warm: every pipeline compiled before anything is timed.
    rt.reset();
    rt.forward(&prompt[..chunk], false)?;
    let full = run(&mut rt)?;
    println!(
        "{model} on {}, {tokens} tokens in chunks of {chunk}",
        rt.device()
    );
    println!(
        "  whole pass {full:.0} ms = {:.0} tok/s\n",
        tokens as f64 / (full / 1e3)
    );
    println!(
        "  {:<18} {:>9} {:>9} {:>7}",
        "without", "ms", "saved", "share"
    );

    let mut rows: Vec<(f64, &str)> = vec![];
    for site in SITES {
        rt.skip = vec![(*site).to_string()];
        let without = run(&mut rt)?;
        rows.push((full - without, site));
    }
    rt.skip.clear();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));

    let accounted: f64 = rows.iter().map(|(d, _)| d.max(0.0)).sum();
    for (saved, site) in &rows {
        println!(
            "  {site:<18} {:>9.0} {saved:>9.0} {:>6.1}%",
            full - saved,
            100.0 * saved / full
        );
    }
    println!(
        "\n  accounted {accounted:.0} ms of {full:.0} ({:.0}%)",
        100.0 * accounted / full
    );
    println!(
        "  a total over 100% means the removed kernels were overlapping each\n  \
         other; under 100% means the rest is dispatch, host writes and\n  \
         downloads that no kernel owns."
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this example needs Metal");
}
