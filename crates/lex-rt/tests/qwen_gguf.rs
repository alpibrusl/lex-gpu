//! A Qwen3.5-family model from a GGUF, against Ollama running the same file.
//!
//! The runtime was written for Qwen3.8's MLX checkpoint. MiMo-v2.6 is the
//! same architecture shipped as GGUF, and three of its conventions differ
//! from that checkpoint in ways that do not error when got wrong: norm
//! weights already carry their 1, `ssm_a` is `-exp(A_log)`, and the value
//! heads are tiled rather than grouped. The last of those left the model
//! fluent and wrong, so this checks tokens *and* log-probabilities, and
//! checks the one fact the wrong version lost.
//!
//! The fixture was recorded from Ollama with `logprobs`, its tokens
//! resolved to ids through the GGUF's own vocabulary. Needs the model
//! pulled; skips itself otherwise, like the other model-backed suites.

#![cfg(any(target_os = "macos", target_os = "linux"))]

use lex_rt::json::Json;
use lex_rt::qwen_run::Runner;

/// Ollama and these kernels round differently; the largest gap over the
/// fixture was 0.012.
const TOL: f64 = 0.05;

fn ints(j: &Json, k: &str) -> Vec<u32> {
    j.get(k)
        .and_then(Json::arr)
        .expect(k)
        .iter()
        .map(|v| v.usize().expect("id") as u32)
        .collect()
}

#[test]
fn mimo_matches_ollama_token_for_token() {
    let j = Json::parse(include_str!("data/mimo_golden.json")).expect("fixture");
    let model = j
        .get("model")
        .and_then(Json::str)
        .expect("model")
        .to_string();
    let prompt = ints(&j, "prompt");
    let want = ints(&j, "ids");
    let want_lp: Vec<f64> = j
        .get("logprobs")
        .and_then(Json::arr)
        .expect("logprobs")
        .iter()
        .map(|v| v.num().expect("lp"))
        .collect();

    let mut rt = match Runner::load(&model, prompt.len() + want.len() + 8) {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };
    let mut logits = vec![];
    for &id in &prompt {
        logits = rt.step(id).expect("prompt");
    }
    let log_softmax = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::MIN, f32::max);
        let z: f64 = v.iter().map(|x| ((x - m) as f64).exp()).sum();
        v.iter()
            .map(|x| (x - m) as f64 - z.ln())
            .collect::<Vec<f64>>()
    };
    let mut worst: f64 = 0.0;
    for (i, (&w, &wlp)) in want.iter().zip(&want_lp).enumerate() {
        let lp = log_softmax(&logits);
        let got = (0..lp.len())
            .max_by(|&a, &b| lp[a].total_cmp(&lp[b]))
            .unwrap() as u32;
        assert_eq!(got, w, "step {i}: lex chose {got}, Ollama chose {w}");
        let gap = (lp[got as usize] - wlp).abs();
        worst = worst.max(gap);
        assert!(
            gap < TOL,
            "step {i}: log-prob {:.4} against Ollama's {wlp:.4}",
            lp[got as usize]
        );
        logits = rt.step(got).expect("decode");
    }
    // The fact the grouped value-head order lost, checked by name so that a
    // regression there reads as what it is.
    assert_eq!(
        want[0], 11751,
        "the fixture's first token should be ' Paris'"
    );
    eprintln!(
        "{model}: {}/{} tokens identical to Ollama, worst |dlogprob| {worst:.4}",
        want.len(),
        want.len()
    );
}
