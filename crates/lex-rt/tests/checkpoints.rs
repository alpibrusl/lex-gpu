//! Which resumable point a full pool gives up.
//!
//! No GPU and no model: the positions are the whole of the policy, and the
//! policy is what decided whether a prefix cache hit or missed over a real
//! lex-code run.

use lex_rt::qwen_run::evict_index;

/// Keeping the newest N is the obvious rule. It is why ten of forty-two
/// requests in a four-task run re-read their whole prompt: turn boundaries
/// bunch up where turns are short, so the pool ends up entirely inside the
/// last few thousand tokens and a prompt diverging earlier finds nothing.
#[test]
fn the_crowded_end_is_thinned_before_the_sparse_start() {
    // One early point, then five bunched at the end.
    let at = [500, 9000, 9100, 9200, 9300, 9400];
    let i = evict_index(&at).expect("something to drop");
    assert!(
        (2..=4).contains(&i),
        "dropped index {i} (position {}), which is not in the crowd",
        at[i]
    );
}

#[test]
fn neither_end_is_ever_dropped() {
    // The newest is what the next turn most likely wants; the oldest is
    // the only fallback for a prompt that diverges early.
    for at in [
        vec![0, 100, 200, 300],
        vec![10, 5000, 5001, 5002, 9000],
        vec![1, 2, 3],
    ] {
        let i = evict_index(&at).expect("something to drop");
        assert!(i > 0 && i + 1 < at.len(), "dropped an end: {i} of {at:?}");
    }
}

#[test]
fn a_pool_too_small_to_thin_says_so() {
    assert_eq!(evict_index(&[]), None);
    assert_eq!(evict_index(&[5]), None);
    assert_eq!(evict_index(&[5, 9]), None, "both are ends");
}

/// Repeated eviction has to leave a pool that still spans the
/// conversation, because that span is what a prefix cache resumes from.
#[test]
fn thinning_keeps_the_pool_spread_out() {
    const KEEP: usize = 6;
    // Boundaries as an agent produces them: a long opening turn, then many
    // short tool exchanges.
    let mut at: Vec<usize> = vec![0, 4000];
    at.extend((1..40).map(|i| 4000 + i * 120));
    let mut pool: Vec<usize> = vec![];
    for p in at {
        pool.push(p);
        while pool.len() > KEEP {
            let i = evict_index(&pool).expect("full pool thins");
            pool.remove(i);
        }
    }
    assert_eq!(pool.len(), KEEP);
    assert_eq!(pool[0], 0, "the earliest fallback was dropped");
    // The gaps must not all be at one end: with the newest-N rule every
    // survivor sat within 600 tokens of the last.
    let span = pool[KEEP - 1] - pool[0];
    assert!(
        span > 8000,
        "pool spans only {span} tokens: {pool:?} — that is the bunching bug"
    );
}
