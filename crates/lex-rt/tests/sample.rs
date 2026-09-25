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
