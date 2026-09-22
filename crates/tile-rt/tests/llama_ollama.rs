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

/// Run every case of a golden file through tile and compare.
fn check_golden(golden: &str) {
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
    let w = Weights::load(&path, longest + 1).expect("load");
    let mut rt = Runner::new(&w).expect("runner");

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
                .unwrap() as u32;
            assert_eq!(
                argmax, st.top[0].0,
                "prompt {:?} step {i}: tile picks {argmax}, reference {}",
                case.prompt, st.top[0].0
            );
            for &(id, want) in &st.top {
                let d = (lp[id as usize] - want).abs();
                worst = worst.max(d);
                assert!(
                    d <= TOL,
                    "prompt {:?} step {i}: token {id} logprob {} vs reference {want}",
                    case.prompt,
                    lp[id as usize]
                );
            }
            steps += 1;
            logits = rt.step(st.next).expect("step");
        }
    }
    eprintln!(
        "{model}: {steps} steps over {} prompts on {}: worst |dlogprob| vs reference {worst:.5} \
         (tolerance {TOL})",
        cases.len(),
        rt.device()
    );
}
