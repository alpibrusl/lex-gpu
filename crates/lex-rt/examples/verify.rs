//! What a speculative verify costs, against the decode step it replaces.
//!
//!     cargo run --release -p lex-rt --example verify -- --context 1440
//!
//! Speculation pays only when verifying `t` tokens costs less than `t`
//! decode steps. That ratio is reported here directly, at each batch size
//! and on both attention paths, because inferring it from tokens per second
//! means dividing by an acceptance rate that moves at the same time.
//!
//! `LEX_MIN_SPLITS` picks the path: the default threshold uses the split
//! kernel once the cache is long enough, and a large value forces the
//! serial causal kernel the split one replaced.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use std::time::Instant;

    use lex_rt::qwen_run::{MAX_BATCH, Runner};

    let mut model = "qwen3.8:27b-mlx".to_string();
    let (mut context, mut reps) = (1440usize, 5usize);
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--context" => context = val()?.parse().map_err(|_| "bad --context")?,
            "--reps" => reps = val()?.parse().map_err(|_| "bad --reps")?,
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    const SIZES: [usize; 4] = [1, 2, 3, 4];
    let top = MAX_BATCH.min(*SIZES.iter().max().expect("sizes"));
    let mut rt = Runner::load(&model, context + top * reps + 64)?;
    println!("{model} on {}, context {context}", rt.device());

    let filler: Vec<u32> = (0..context)
        .map(|i| 1000 + (i as u32 * 7919) % 200000)
        .collect();
    let batch: Vec<u32> = (0..top).map(|i| 4000 + i as u32 * 131).collect();

    // Median of `reps`: one slow pass is the machine, not the schedule.
    // Every measurement rewinds to the same position, so each is taken at
    // the context asked for rather than at a drifting one.
    let timed = |rt: &mut Runner, t: usize| -> Result<f64, String> {
        let mut runs = vec![];
        for _ in 0..reps {
            rt.restore();
            let s = Instant::now();
            rt.forward(&batch[..t], true)?;
            runs.push(s.elapsed().as_secs_f64() * 1e3);
        }
        runs.sort_by(f64::total_cmp);
        Ok(runs[reps / 2])
    };

    // Fill once, then checkpoint: every measurement below rolls back to
    // exactly this state, recurrent memory included.
    rt.reset();
    for &t in &filler {
        rt.step(t)?;
    }
    rt.save();
    for t in SIZES {
        rt.restore();
        rt.forward(&batch[..t], true)?;
    }
    let step = {
        rt.restore();
        let mut runs = vec![];
        for _ in 0..reps {
            rt.restore();
            let s = Instant::now();
            rt.step(batch[0])?;
            runs.push(s.elapsed().as_secs_f64() * 1e3);
        }
        runs.sort_by(f64::total_cmp);
        runs[reps / 2]
    };
    println!("  one decode step {step:.1} ms\n");

    println!(
        "  {:>3} {:>10} {:>10} {:>10} {:>10}",
        "t", "split", "serial", "split/step", "serial/step"
    );
    for t in SIZES {
        let split = timed(&mut rt, t)?;
        // Same Runner, same cache, only the threshold changes -- it is
        // read per step, so this takes effect on the next pass.
        unsafe { std::env::set_var("LEX_MIN_SPLITS", "999999") };
        let serial = timed(&mut rt, t)?;
        unsafe { std::env::remove_var("LEX_MIN_SPLITS") };
        println!(
            "  {t:>3} {split:>8.1}ms {serial:>8.1}ms {:>10.2} {:>10.2}",
            split / step,
            serial / step
        );
    }
    println!(
        "\n  The last two columns are the verify in units of decode steps,\n  \
         which is also the break-even acceptance: a verify of `t` that\n  \
         costs 1.8 steps has to accept 1.8 tokens to be worth running."
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this example needs Metal");
}
