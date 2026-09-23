//! Qwen3.8 on the GPU against the f32 reference that agrees with Ollama.
//!
//! The fixture is `qwen35_27b_golden.txt`, written by
//! `scripts/qwen_ref.py --write-golden`: for each prompt, the token the
//! reference picks at every step and the log-probabilities of its top few.
//! Needs macOS and the model pulled; skips itself otherwise.

#![cfg(target_os = "macos")]

use lex_rt::qwen_run::Runner;

/// Held for the length of any test that loads the model.
///
/// The weights are 14.5 GB and `cargo test` runs a binary's tests in
/// parallel, so four of these at once asks for 58 GB on a machine with a
/// 55 GB working set. What that looks like is the whole binary dying with
/// SIGKILL and no failing assertion -- a test suite that reports a failure
/// it cannot explain. One model at a time instead.
static MODEL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The lock, ignoring poisoning: a panicking test has already reported.
fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    MODEL.lock().unwrap_or_else(|e| e.into_inner())
}

/// Log-probabilities differ from the reference by f32 rounding and by the
/// attention kernel's f16 cache; the reference itself sits within 0.10 of
/// Ollama's own numbers.
const TOL: f64 = 0.02;

struct Step {
    next: u32,
    top: Vec<(u32, f64)>,
}

struct Case {
    prompt: Vec<u32>,
    steps: Vec<Step>,
}

fn parse(golden: &str) -> (String, Vec<Case>) {
    let mut model = String::new();
    let mut cases: Vec<Case> = vec![];
    for line in golden.lines() {
        let mut w = line.split_whitespace();
        match w.next() {
            Some("model") => model = w.next().unwrap_or_default().to_string(),
            Some("case") => cases.push(Case {
                prompt: w.filter_map(|t| t.parse().ok()).collect(),
                steps: vec![],
            }),
            Some("step") => {
                let next = w.next().and_then(|t| t.parse().ok()).expect("step token");
                let top = w
                    .filter_map(|t| {
                        let (id, lp) = t.split_once(':')?;
                        Some((id.parse().ok()?, lp.parse().ok()?))
                    })
                    .collect();
                cases
                    .last_mut()
                    .expect("a case first")
                    .steps
                    .push(Step { next, top });
            }
            _ => {}
        }
    }
    (model, cases)
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
    let ln = sum.ln();
    v.iter().map(|&x| x as f64 - m - ln).collect()
}

#[test]
fn qwen35_matches_the_reference_that_matches_ollama() {
    let _lock = one_at_a_time();
    let golden = include_str!("data/qwen35_27b_golden.txt");
    let (model, cases) = parse(golden);
    let longest = cases
        .iter()
        .map(|c| c.prompt.len() + c.steps.len())
        .max()
        .unwrap_or(0);
    let mut rt = match Runner::load(&model, longest + 8) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };

    let mut worst = 0.0f64;
    let mut steps = 0;
    for case in &cases {
        rt.reset();
        let mut logits = vec![];
        for &t in &case.prompt {
            logits = rt.step(t).expect("step");
        }
        for (i, st) in case.steps.iter().enumerate() {
            let lp = log_softmax(&logits);
            let argmax = (0..lp.len())
                .max_by(|&a, &b| lp[a].total_cmp(&lp[b]))
                .expect("logits") as u32;
            assert_eq!(
                argmax, st.next,
                "prompt {:?} step {i}: tile picks {argmax}, the reference {}",
                case.prompt, st.next
            );
            for &(id, want) in &st.top {
                let d = (lp[id as usize] - want).abs();
                worst = worst.max(d);
                assert!(
                    d <= TOL,
                    "prompt {:?} step {i}: token {id} at {} vs {want}",
                    case.prompt,
                    lp[id as usize]
                );
            }
            steps += 1;
            logits = rt.step(st.next).expect("step");
        }
    }
    eprintln!(
        "{model}: {steps} steps over {} prompts on {}: worst |dlogprob| {worst:.5} \
         (tolerance {TOL})",
        cases.len(),
        rt.device()
    );
}

/// A prompt fed as one batch must land where the same prompt lands token
/// by token. This is the gate speculative decoding sits behind: a verify
/// is only sound if a batch and a sequence agree.
#[test]
fn a_batch_lands_where_the_same_tokens_land_one_by_one() {
    let _lock = one_at_a_time();
    let golden = include_str!("data/qwen35_27b_golden.txt");
    let (model, cases) = parse(golden);
    let prompt = &cases[0].prompt;
    let mut rt = match Runner::load(&model, prompt.len() + 16) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };

    // One at a time.
    rt.reset();
    let mut one = vec![];
    for &t in prompt {
        one.push(rt.step(t).expect("step"));
    }

    // The same tokens as a single batch, every position's logits.
    rt.reset();
    let many = rt.forward(prompt, true).expect("batch");
    assert_eq!(many.len(), prompt.len());

    unsafe { std::env::remove_var("LEX_MIN_SPLITS") };

    let mut worst = 0.0f32;
    for (i, (b, s)) in many.iter().zip(&one).enumerate() {
        let top = |v: &[f32]| {
            (0..v.len())
                .max_by(|&a, &c| v[a].total_cmp(&v[c]))
                .expect("logits")
        };
        assert_eq!(
            top(b),
            top(s),
            "position {i}: the batch and the sequence pick different tokens"
        );
        let scale = s.iter().map(|x| x.abs()).fold(1e-6, f32::max);
        let d = b
            .iter()
            .zip(s)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
            / scale;
        worst = worst.max(d);
    }
    eprintln!("batch vs sequence: worst logit difference {worst:e} of scale");
    assert!(worst < 1e-3, "batch differs from the sequence by {worst:e}");
}

/// Speculation must be invisible in the output: the same tokens as greedy
/// decoding, in the same order, whatever the draft head guesses.
///
/// That is the whole contract. A draft is only ever a guess about what the
/// model was going to say, and a verify that accepts a token the model
/// would not have produced is a wrong answer delivered faster.
#[test]
fn speculation_lands_exactly_where_greedy_lands() {
    let _lock = one_at_a_time();
    let golden = include_str!("data/qwen35_27b_golden.txt");
    let (model, cases) = parse(golden);
    let prompt = &cases[0].prompt;
    const STEPS: usize = 12;
    let mut rt = match Runner::load(&model, prompt.len() + STEPS + 8) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };
    if !rt.has_mtp() {
        eprintln!("SKIPPED: {model} carries no draft head");
        return;
    }
    let top = |v: &[f32]| {
        (0..v.len())
            .max_by(|&a, &b| v[a].total_cmp(&v[b]))
            .expect("logits") as u32
    };

    // Greedy, one token at a time.
    rt.reset();
    let mut logits = vec![];
    for &t in prompt {
        logits = rt.step(t).expect("step");
    }
    let mut plain = vec![];
    for _ in 0..STEPS {
        let t = top(&logits);
        plain.push(t);
        logits = rt.step(t).expect("step");
    }

    // The same, but drafting two tokens ahead every round.
    rt.reset();
    let mut logits = vec![];
    for &t in prompt {
        logits = rt.step(t).expect("step");
    }
    let mut spec = vec![];
    let mut next = top(&logits);
    while spec.len() < STEPS {
        let (committed, after) = rt.speculate(next, 2).expect("speculate");
        spec.extend(committed);
        next = after;
    }
    spec.truncate(STEPS);

    assert_eq!(
        spec, plain,
        "speculation changed the output: {spec:?} against {plain:?}"
    );
    eprintln!("{STEPS} tokens identical with a draft depth of 2");
}

/// The same tokens, from the same state, with a long enough context that
/// decode attention takes the split-KV path.
///
/// The golden tests above use prompts of about a dozen positions, where
/// `nsplit` is 1 and the serial kernel runs. So the split pair — two
/// kernels and two scalars — was never exercised by any test, and a wrong
/// scalar in the combine (the head count where the chunk count belongs)
/// reduced a prefix of the splits and dropped the rest of the cache, while
/// every test stayed green and the speed went *up*.
///
/// Feeding the same tokens from position 0 and from deep in a sequence
/// must give the same logits for the layers that carry no cache, and must
/// give *correct* ones for the sixteen that do. The check here is against
/// the serial kernel: same prompt, same continuation, split path against
/// non-split.
#[test]
fn split_kv_decode_agrees_with_the_serial_kernel() {
    let _lock = one_at_a_time();
    let golden = include_str!("data/qwen35_27b_golden.txt");
    let (model, cases) = parse(golden);
    // Long enough that the combine needs more chunks than the model has
    // KV heads. At 600 positions a wrong second scalar happens to be
    // *larger* than the right one, so it over-reduces into zeroed partials
    // and nothing shows; past ~1024 it is smaller and drops cache. The
    // first version of this test used 600 and passed with the bug in.
    const FILL: usize = 1200;
    const STEPS: usize = 6;
    let mut rt = match Runner::load(&model, FILL + STEPS + 16) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };
    let top = |v: &[f32]| {
        (0..v.len())
            .max_by(|&a, &b| v[a].total_cmp(&v[b]))
            .expect("logits") as u32
    };
    let filler: Vec<u32> = (0..FILL)
        .map(|i| 1000 + (i as u32 * 7919) % 200000)
        .collect();

    // With the split path.
    rt.reset();
    let mut logits = vec![];
    for &t in filler.iter().chain(&cases[0].prompt) {
        logits = rt.step(t).expect("step");
    }
    // Logits, not argmax. Dropping the last splits of a 1200-position
    // cache moves the distribution and often leaves the top token alone:
    // an earlier version of this test compared tokens, and passed with the
    // bug deliberately put back in.
    let mut split: Vec<Vec<f32>> = vec![];
    for _ in 0..STEPS {
        let t = top(&logits);
        split.push(logits.clone());
        logits = rt.step(t).expect("step");
    }

    // The same, forced down the serial kernel.
    rt.skip.clear();
    // One Runner at a time: this model is 14.5 GB of weights and the suite
    // runs its model tests in parallel. Holding two was enough to have the
    // whole binary killed with SIGKILL, which reads as a test failure with
    // no failing assertion.
    drop(rt);

    // The threshold is read per step, at plan time -- so it has to stay
    // set for the whole of this run, not just while the Runner is built.
    // Removing it after construction left both runs on the split path,
    // comparing it against itself, and the test passed with the bug in.
    unsafe { std::env::set_var("LEX_MIN_SPLITS", "999999") };
    let mut rt2 = match Runner::load(&model, FILL + STEPS + 16) {
        Ok(rt) => rt,
        Err(e) => {
            unsafe { std::env::remove_var("LEX_MIN_SPLITS") };
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };
    rt2.reset();
    let mut logits = vec![];
    for &t in filler.iter().chain(&cases[0].prompt) {
        logits = rt2.step(t).expect("step");
    }
    let mut serial: Vec<Vec<f32>> = vec![];
    for _ in 0..STEPS {
        let t = top(&logits);
        serial.push(logits.clone());
        logits = rt2.step(t).expect("step");
    }

    let mut worst = 0.0f32;
    for (a, b) in split.iter().zip(&serial) {
        let scale = b.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
            / scale;
        worst = worst.max(d);
    }
    assert!(
        worst < 1e-3,
        "split-KV decode differs from the serial kernel by {worst:e} of scale \
         at {FILL} positions"
    );
    eprintln!("split vs serial at {FILL} positions: worst {worst:e} of scale");
}

/// The batched path's split attention against the serial causal kernel.
///
/// This is the verify a speculative step runs, and it is the reason
/// speculation lost as context grew: the serial kernel scans the cache
/// with one threadgroup per KV head, so two tokens at 1440 positions cost
/// 1.84 passes. The split kernel cuts the cache across threadgroups with
/// each query still masked to its own position, which is a different
/// kernel from the decode split -- so it needs its own proof.
#[test]
fn split_kv_batch_agrees_with_the_serial_kernel() {
    let _lock = one_at_a_time();
    let (model, _) = parse(include_str!("data/qwen35_27b_golden.txt"));
    const FILL: usize = 1200;
    /// More than one, so masking per query row is actually exercised.
    const BATCH: usize = 4;

    // Fill by decoding, then run one batch at that context and keep every
    // token's logits -- the masking is what differs between query rows.
    let run = |serial: bool| -> Option<Vec<Vec<f32>>> {
        if serial {
            unsafe { std::env::set_var("LEX_MIN_SPLITS", "999999") };
        }
        let out = (|| {
            let mut rt = match Runner::load(&model, FILL + BATCH + 16) {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("SKIPPED: {model} ({e})");
                    return None;
                }
            };
            rt.reset();
            let filler: Vec<u32> = (0..FILL)
                .map(|i| 1000 + (i as u32 * 7919) % 200000)
                .collect();
            for &t in &filler {
                rt.step(t).expect("step");
            }
            let batch: Vec<u32> = (0..BATCH).map(|i| 4000 + i as u32 * 131).collect();
            Some(rt.forward(&batch, true).expect("forward"))
        })();
        unsafe { std::env::remove_var("LEX_MIN_SPLITS") };
        out
    };

    // One Runner at a time: 14.5 GB of weights, and two of them is a
    // SIGKILL that reads as a failure with no failing assertion.
    let Some(split) = run(false) else { return };
    let Some(serial) = run(true) else { return };

    let mut worst = 0.0f32;
    for (row, (a, b)) in split.iter().zip(&serial).enumerate() {
        let scale = b.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
        let d = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
            / scale;
        assert!(
            d < 1e-3,
            "batch row {row} of {BATCH} differs from the serial kernel by \
             {d:e} of scale at {FILL} positions"
        );
        worst = worst.max(d);
    }
    eprintln!("batch split vs serial at {FILL} positions: worst {worst:e} of scale");
}
