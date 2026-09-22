//! P2's end-to-end tests: Llama models, as Ollama serves them, run by tile
//! kernels on the GPU. One test per golden file:
//!
//! - `llama3.2:1b`: Q8_0 weights;
//! - `llama3.1:8b`: Q4_K_M (Q4_K + Q6_K) — P2's exit test, Llama-3-8B int4.
//!
//! The oracle chain:
//! - `scripts/llama_ref.py` checks a PyTorch reference against Ollama's own
//!   log-probabilities (same GGUF file, same prompts), and writes what the
//!   reference computed to `tests/data/<model>_golden.txt`;
//! - this test runs the same prompts through tile and checks, at every step,
//!   that tile's greedy token matches and its top-5 log-probabilities agree
//!   with the reference's.
//!
//! Each needs its model in the local Ollama store (`ollama pull <tag>`) and
//! a Metal device. Without the model it skips, loudly.
#![cfg(target_os = "macos")]

use tile_rt::gguf::ollama_model;
use tile_rt::llama::{Runner, Weights, log_softmax};

/// tile keeps its KV cache in f16 (as llama.cpp does); the reference keeps
/// K and V in f32. That, not the kernels, is most of the budget.
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
            Some("model") => model = w.next().unwrap().to_string(),
            Some("case") => cases.push(Case {
                prompt: w.map(|t| t.parse().unwrap()).collect(),
                steps: vec![],
            }),
            Some("step") => {
                let next = w.next().unwrap().parse().unwrap();
                let top = w
                    .map(|p| {
                        let (i, lp) = p.split_once(':').unwrap();
                        (i.parse().unwrap(), lp.parse().unwrap())
                    })
                    .collect();
                cases.last_mut().unwrap().steps.push(Step { next, top });
            }
            _ => {}
        }
    }
    (model, cases)
}

#[test]
fn llama32_1b_matches_the_reference_that_matches_ollama() {
    check_golden(include_str!("data/llama32_1b_golden.txt"));
}

/// P2's exit test: Llama-3-8B in int4, as Ollama serves it (Q4_K_M).
#[test]
fn llama31_8b_q4_k_m_matches_the_reference_that_matches_ollama() {
    check_golden(include_str!("data/llama31_8b_golden.txt"));
}

/// How the tokens reach the model: the three paths must agree.
#[derive(Clone, Copy, Debug)]
enum Feed {
    /// One token at a time (decode).
    Steps,
    /// The prompt in batches of `n` (prefill), then one token at a time.
    Prefill(usize),
    /// Prompt prefilled in 4s, then the continuation in batches of `n` with
    /// every position's logits (speculative verification).
    Verify(usize),
}

fn check_logits(what: &str, lp: &[f64], st: &Step, worst: &mut f64) {
    let argmax = (0..lp.len())
        .max_by(|&a, &b| lp[a].total_cmp(&lp[b]))
        .unwrap() as u32;
    assert_eq!(
        argmax, st.top[0].0,
        "{what}: tile picks {argmax}, reference {}",
        st.top[0].0
    );
    for &(id, want) in &st.top {
        let d = (lp[id as usize] - want).abs();
        *worst = worst.max(d);
        assert!(
            d <= TOL,
            "{what}: token {id} logprob {} vs reference {want}",
            lp[id as usize]
        );
    }
}

/// Run every case of a golden file through tile, fed each way, and compare.
fn check_golden(golden: &str) {
    for feed in [
        Feed::Steps,
        Feed::Prefill(4),
        Feed::Prefill(16),
        Feed::Verify(5),
    ] {
        check_golden_fed(golden, feed);
    }
}

fn check_golden_fed(golden: &str, feed: Feed) {
    let (model, cases) = parse(golden);
    let path = match ollama_model(&model) {
        Ok(p) if p.exists() => p,
        other => {
            eprintln!(
                "SKIPPED: {model} is not in the local Ollama store ({other:?}); `ollama pull {model}`"
            );
            return;
        }
    };
    let longest = cases
        .iter()
        .map(|c| c.prompt.len() + c.steps.len())
        .max()
        .unwrap_or(0);
    let w = Weights::load(&path, longest + 16).expect("load");
    let mut rt = Runner::new(&w).expect("runner");

    let mut worst = 0.0f64;
    let mut steps = 0;
    for case in &cases {
        rt.reset();
        let what = |i: usize| format!("{feed:?} prompt {:?} step {i}", case.prompt);
        let mut logits = match feed {
            Feed::Steps => {
                let mut l = vec![];
                for &t in &case.prompt {
                    l = rt.step(t).expect("step");
                }
                l
            }
            Feed::Prefill(n) => rt.prefill(&case.prompt, n).expect("prefill"),
            Feed::Verify(_) => rt.prefill(&case.prompt, 4).expect("prefill"),
        };
        match feed {
            Feed::Verify(n) => {
                // Step 0 comes from the prompt; then each batch of `n`
                // continuation tokens yields the next `n` predictions at once.
                check_logits(&what(0), &log_softmax(&logits), &case.steps[0], &mut worst);
                steps += 1;
                let nexts: Vec<u32> = case.steps.iter().map(|s| s.next).collect();
                let mut i = 1;
                for batch in nexts[..nexts.len() - 1].chunks(n) {
                    let rows = rt.forward(batch, true).expect("verify");
                    for row in rows {
                        check_logits(&what(i), &log_softmax(&row), &case.steps[i], &mut worst);
                        i += 1;
                        steps += 1;
                    }
                }
            }
            _ => {
                for (i, st) in case.steps.iter().enumerate() {
                    check_logits(&what(i), &log_softmax(&logits), st, &mut worst);
                    steps += 1;
                    logits = rt.step(st.next).expect("step");
                }
            }
        }
    }
    eprintln!(
        "{model} [{feed:?}]: {steps} steps over {} prompts on {}: worst |dlogprob| vs reference \
         {worst:.5} (tolerance {TOL})",
        cases.len(),
        rt.device()
    );
}
