//! The sampler, which decides what the model actually says.
//!
//! Greedy decoding is what this served for months, and it is what made a
//! thinking model run past a client's timeout. These check the two things
//! that matter: that temperature 0 is still exactly argmax, so every
//! token-for-token comparison against a reference still holds, and that
//! sampling stays inside the nucleus it was told to.

use lex_rt::sample::Sampler;

/// Logits with a clear ordering and a long tail of noise.
fn logits(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32).wrapping_mul(2_654_435_761);
            -20.0 + (h >> 18) as f32 / 1000.0
        })
        .collect()
}

#[test]
fn temperature_zero_is_argmax() {
    let mut v = logits(5000);
    v[1234] = 99.0;
    let mut s = Sampler::new(0.0, 0.95, 20, 1);
    for _ in 0..8 {
        assert_eq!(s.pick(&v), 1234);
    }
}

#[test]
fn top_k_of_one_is_argmax_whatever_the_temperature() {
    let mut v = logits(5000);
    v[77] = 50.0;
    let mut s = Sampler::new(2.0, 0.95, 1, 1);
    for _ in 0..8 {
        assert_eq!(s.pick(&v), 77);
    }
}

#[test]
fn sampling_stays_inside_the_top_k() {
    let v = logits(20_000);
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_unstable_by(|&a, &b| v[b].total_cmp(&v[a]));
    let allowed: std::collections::HashSet<u32> =
        order[..20].iter().map(|&i| i as u32).collect();
    let mut s = Sampler::seeded(7);
    for _ in 0..500 {
        let t = s.pick(&v);
        assert!(allowed.contains(&t), "sampled {t}, outside the top 20");
    }
}

/// It has to actually vary, or it is greedy decoding with extra steps --
/// which is the bug this was written to fix.
#[test]
fn sampling_does_not_always_pick_the_same_token() {
    let v = logits(20_000);
    let mut s = Sampler::seeded(11);
    let picks: std::collections::HashSet<u32> = (0..200).map(|_| s.pick(&v)).collect();
    assert!(
        picks.len() > 1,
        "every draw was the same token; this is argmax wearing a hat"
    );
}

#[test]
fn a_seed_repeats_exactly() {
    let v = logits(10_000);
    let draw = |seed| {
        let mut s = Sampler::seeded(seed);
        (0..50).map(|_| s.pick(&v)).collect::<Vec<_>>()
    };
    assert_eq!(draw(42), draw(42));
    assert_ne!(draw(42), draw(43), "the seed is not reaching the stream");
}

/// A tight nucleus must collapse onto the most likely token; a loose one
/// must not. Without this, top_p could be ignored and nothing would say so.
#[test]
fn top_p_narrows_what_can_be_drawn() {
    let mut v = vec![-30.0f32; 1000];
    // One dominant token, then a few plausible ones.
    v[3] = 10.0;
    v[4] = 3.0;
    v[5] = 2.9;
    v[6] = 2.8;
    let tight: std::collections::HashSet<u32> = {
        let mut s = Sampler::new(1.0, 0.05, 20, 5);
        (0..200).map(|_| s.pick(&v)).collect()
    };
    assert_eq!(
        tight,
        std::collections::HashSet::from([3]),
        "a 0.05 nucleus should hold only the dominant token"
    );
    let loose: std::collections::HashSet<u32> = {
        let mut s = Sampler::new(2.0, 1.0, 20, 5);
        (0..400).map(|_| s.pick(&v)).collect()
    };
    assert!(loose.len() > 1, "a full nucleus drew only {loose:?}");
}

/// A distribution with a clear favourite and a few real alternatives,
/// which is what a language model's next token usually looks like.
fn plausible() -> Vec<f32> {
    let mut v = vec![-40.0f32; 2000];
    v[10] = 4.0; // the favourite -- what a greedy draft proposes
    v[11] = 3.2;
    v[12] = 2.9;
    v[13] = 2.1;
    v[14] = 1.0;
    v
}

/// What comes out of one speculative round against `draft`: the draft if
/// accepted, the correction if not.
fn emitted(s: &mut Sampler, logits: &[f32], draft: u32) -> u32 {
    match s.verify_draft(logits, draft) {
        Ok(()) => draft,
        Err(t) => t,
    }
}

fn frequencies(draws: impl Iterator<Item = u32>, n: usize) -> std::collections::HashMap<u32, f64> {
    let mut f = std::collections::HashMap::new();
    for t in draws {
        *f.entry(t).or_insert(0.0) += 1.0 / n as f64;
    }
    f
}

/// Speculation must change how fast tokens arrive and nothing about which
/// ones. Whatever the draft proposed -- the favourite, an also-ran, or a
/// token outside the nucleus altogether -- the emitted token has to be
/// distributed exactly as plain sampling would have drawn it.
#[test]
fn speculative_sampling_reproduces_the_distribution_whatever_is_drafted() {
    const N: usize = 200_000;
    let logits = plausible();
    let target = Sampler::new(1.0, 1.0, 20, 0).nucleus(&logits);
    for draft in [10u32, 12, 14, 1999] {
        let mut s = Sampler::new(1.0, 1.0, 20, 99 + draft as u64);
        let got = frequencies((0..N).map(|_| emitted(&mut s, &logits, draft)), N);
        for (t, p) in target.0.iter().zip(&target.1) {
            let q = got.get(t).copied().unwrap_or(0.0);
            // Four standard errors of a proportion at this N.
            let tol = 4.0 * ((*p as f64) * (1.0 - *p as f64) / N as f64).sqrt() + 1e-4;
            assert!(
                (q - *p as f64).abs() < tol,
                "draft {draft}: token {t} came out {q:.4}, plain sampling gives {p:.4}"
            );
        }
        let outside: f64 = got
            .iter()
            .filter(|(t, _)| !target.0.contains(t))
            .map(|(_, f)| f)
            .sum();
        assert!(outside == 0.0, "draft {draft}: {outside} of draws fell outside the nucleus");
    }
}

/// The test above has to be able to fail. Checking a sampled draft the
/// greedy way -- accept only the argmax, correct to the argmax -- is the
/// obvious wrong rule, and it collapses the distribution onto one token.
/// If that passed the check, the check would prove nothing.
#[test]
fn the_greedy_rule_under_sampling_is_caught() {
    const N: usize = 50_000;
    let logits = plausible();
    let target = Sampler::new(1.0, 1.0, 20, 0).nucleus(&logits);
    let p_second = target.1[1] as f64;
    // Greedy verification of a greedy draft always emits the argmax.
    let wrong = frequencies(std::iter::repeat_n(10u32, N), N);
    let q_second = wrong.get(&target.0[1]).copied().unwrap_or(0.0);
    assert!(
        (q_second - p_second).abs() > 0.05,
        "the greedy rule gave the runner-up {q_second:.3} against {p_second:.3}; \
         the distribution test could not tell it apart"
    );
}

/// At temperature 0 the rule is the greedy check it replaces, so the
/// token-for-token comparisons against a reference still hold.
#[test]
fn at_temperature_zero_it_is_exactly_the_greedy_check() {
    let logits = plausible();
    let mut s = Sampler::new(0.0, 1.0, 20, 3);
    for _ in 0..100 {
        assert_eq!(s.verify_draft(&logits, 10), Ok(()), "the argmax draft must be accepted");
        assert_eq!(s.verify_draft(&logits, 12), Err(10), "anything else corrects to the argmax");
    }
}
