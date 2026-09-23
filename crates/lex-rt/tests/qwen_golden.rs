//! Qwen3.8 on the GPU against the f32 reference that agrees with Ollama.
//!
//! The fixture is `qwen35_27b_golden.txt`, written by
//! `scripts/qwen_ref.py --write-golden`: for each prompt, the token the
//! reference picks at every step and the log-probabilities of its top few.
//! Needs macOS and the model pulled; skips itself otherwise.

#![cfg(target_os = "macos")]

use lex_rt::qwen_run::Runner;

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
