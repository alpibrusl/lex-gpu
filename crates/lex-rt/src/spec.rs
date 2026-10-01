//! How deep to draft: chosen each cycle from what drafting has been
//! earning, rather than fixed.
//!
//! A cycle drafts `d` tokens and verifies `d + 1`; it commits one more than
//! the drafts accepted before the first rejection. Deeper drafting commits
//! more when the head is right and costs a draft and a verify row either
//! way, so the best depth moves with the text: a list, code or a repeated
//! phrase is predictable for a stretch and pays for depth 3, prose pays for
//! 2, and a hard passage for 1. Through the server on prose a fixed depth
//! of 3 was slower than 2 (43.9 against 46.3 tok/s); Ollama's runner, which
//! adapts, spent that same text at 2.5-3 on average.
//!
//! The controller keeps, as running averages, the chance that the draft at
//! each position is accepted given the one before it was, and the wall
//! time of a cycle at each depth. Each cycle it picks the depth with the
//! most expected tokens a millisecond,
//!
//! ```text
//!   E(d) = 1 + a1 + a1 a2 + ... + a1 ... ad,      pick argmax E(d) / cost(d)
//! ```
//!
//! and every so often tries a depth next to the chosen one, so a cost or an
//! acceptance it has stopped observing does not go stale. Depths it has
//! never timed are tried first.

/// Draft depth chosen from observed acceptance and cost.
#[derive(Clone, Debug)]
pub struct DepthController {
    max: usize,
    /// `accept[i]`: the chance the draft at position `i + 1` is accepted,
    /// given every one before it was.
    accept: Vec<f64>,
    /// `cost[d - 1]`: milliseconds a cycle at depth `d`, once timed.
    cost: Vec<Option<f64>>,
    cycles: usize,
    /// How far each new observation moves an average.
    alpha: f64,
    /// Cycles between probes of a neighbouring depth.
    probe_every: usize,
}

impl DepthController {
    /// Depths `1..=max`; acceptance starts at `prior` at every position.
    pub fn new(max: usize, prior: f64) -> DepthController {
        assert!(max >= 1, "a controller needs at least depth 1");
        DepthController {
            max,
            accept: vec![prior.clamp(0.0, 1.0); max],
            cost: vec![None; max],
            cycles: 0,
            alpha: 0.15,
            probe_every: 24,
        }
    }

    /// The expected tokens a cycle at depth `d` commits.
    pub fn expected(&self, d: usize) -> f64 {
        let mut e = 1.0;
        let mut p = 1.0;
        for a in &self.accept[..d] {
            p *= a;
            e += p;
        }
        e
    }

    /// The depth with the most expected tokens a millisecond, among those
    /// timed.
    pub fn best(&self) -> usize {
        (1..=self.max)
            .filter_map(|d| self.cost[d - 1].map(|c| (d, self.expected(d) / c)))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(1, |(d, _)| d)
    }

    /// The depth for the next cycle.
    pub fn pick(&mut self) -> usize {
        self.cycles += 1;
        if let Some(d) = (1..=self.max).find(|&d| self.cost[d - 1].is_none()) {
            return d;
        }
        let best = self.best();
        if self.cycles.is_multiple_of(self.probe_every) {
            // Alternate above and below, within range.
            let up = (self.cycles / self.probe_every).is_multiple_of(2);
            let d = if up { best + 1 } else { best.saturating_sub(1) };
            if (1..=self.max).contains(&d) {
                return d;
            }
        }
        best
    }

    /// A cycle at depth `depth` accepted `kept` drafts and took `ms`.
    pub fn record(&mut self, depth: usize, kept: usize, ms: f64) {
        if depth == 0 || depth > self.max {
            return;
        }
        let kept = kept.min(depth);
        // Positions 1..=kept were accepted; kept + 1, if drafted, was not.
        // Positions past that were never reached and say nothing.
        for i in 0..(kept + 1).min(depth) {
            let seen = if i < kept { 1.0 } else { 0.0 };
            self.accept[i] += self.alpha * (seen - self.accept[i]);
        }
        let c = &mut self.cost[depth - 1];
        *c = Some(match *c {
            None => ms,
            Some(old) => old + self.alpha * (ms - old),
        });
    }

    /// The acceptance estimates, position 1 first.
    pub fn acceptance(&self) -> &[f64] {
        &self.accept
    }
}

/// Drafts from the context: the tokens that followed the most recent
/// earlier occurrence of the context's last `n` tokens, for the longest
/// `n` from `n_max` down to `n_min` that occurs; at most `k` of them.
///
/// A coding agent's reply copies spans of its prompt -- the file it is
/// editing, the call it is repeating -- and there the model's own next
/// tokens are already written down. Replayed over real requests
/// (`scripts/lookup_replay.py`): on edits that return a whole file, 77-87%
/// of the reply came from lookups at 3.3-3.7 tokens a round, 54 -> 74
/// tok/s; on prose a match of four is rare enough to cost nothing, while
/// matching on two fired on phrases like "of the" and cost 13%.
///
/// `context` ends with the token about to be fed; the drafts are what
/// would follow it.
pub fn lookup(context: &[u32], n_max: usize, n_min: usize, k: usize) -> Vec<u32> {
    let len = context.len();
    for n in (n_min.max(1)..=n_max).rev() {
        if len <= n {
            continue;
        }
        let pat = &context[len - n..];
        // Most recent first, and never the suffix matching itself.
        if let Some(s) = (0..len - n).rev().find(|&s| &context[s..s + n] == pat) {
            let from = s + n;
            return context[from..(from + k).min(len)].to_vec();
        }
    }
    vec![]
}

#[cfg(test)]
mod lookup_tests {
    use super::lookup;

    #[test]
    fn it_proposes_what_followed_the_last_occurrence() {
        // ... 1 2 3 4 [5 6 7] ... 1 2 3 4 -> 5 6 7
        let c = [9, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4];
        assert_eq!(lookup(&c, 6, 4, 3), vec![5, 6, 7]);
        // The most recent occurrence wins over an older one.
        let c = [1, 2, 3, 4, 10, 0, 1, 2, 3, 4, 20, 21, 1, 2, 3, 4];
        assert_eq!(lookup(&c, 4, 4, 2), vec![20, 21]);
    }

    #[test]
    fn it_takes_the_longest_match_and_stops_at_the_end() {
        // Suffix [7, 1, 2, 3, 4] occurs once, at the start; a shorter
        // [1, 2, 3, 4] occurs more recently with a different follower.
        let c = [7, 1, 2, 3, 4, 50, 1, 2, 3, 4, 60, 7, 1, 2, 3, 4];
        assert_eq!(lookup(&c, 5, 4, 1), vec![50]);
        // Fewer than k tokens left after the match: what there is.
        let c = [1, 2, 3, 4, 1, 2, 3, 4];
        assert_eq!(lookup(&c, 4, 4, 8), vec![1, 2, 3, 4]);
    }

    #[test]
    fn a_short_or_absent_match_proposes_nothing() {
        let c = [1, 2, 3, 9, 8, 1, 2, 3];
        assert!(
            lookup(&c, 6, 4, 3).is_empty(),
            "only a match of three exists"
        );
        assert_eq!(lookup(&c, 6, 3, 3), vec![9, 8, 1]);
        assert!(lookup(&[1, 2, 3], 6, 4, 3).is_empty());
        assert!(lookup(&[], 6, 4, 3).is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::DepthController;

    /// A cycle's wall time: a verify that grows with its rows, and a draft
    /// a token -- the shape measured on the M4 Max, roughly.
    fn cost(d: usize) -> f64 {
        37.0 + 2.2 * d as f64 + 2.6 * d as f64
    }

    /// Drive the controller against a head accepting each position with
    /// `p`, deterministically: position i is accepted on a fraction p of
    /// cycles, spread evenly.
    fn settle(p: f64, cycles: usize) -> (DepthController, Vec<usize>) {
        let mut c = DepthController::new(3, 0.6);
        let mut picks = vec![];
        let mut acc = [0.0f64; 3];
        for _ in 0..cycles {
            let d = c.pick();
            picks.push(d);
            let mut kept = 0;
            for a in acc.iter_mut().take(d) {
                *a += p;
                if *a >= 1.0 {
                    *a -= 1.0;
                    kept += 1;
                } else {
                    break;
                }
            }
            c.record(d, kept, cost(d));
        }
        (c, picks)
    }

    #[test]
    fn it_tries_every_depth_before_trusting_any() {
        let mut c = DepthController::new(3, 0.6);
        let first: Vec<usize> = (0..3)
            .map(|_| {
                let d = c.pick();
                c.record(d, 1, cost(d));
                d
            })
            .collect();
        assert_eq!(first, vec![1, 2, 3]);
    }

    #[test]
    fn a_predictable_stretch_drafts_deep_and_a_hard_one_shallow() {
        let (hi, picks_hi) = settle(0.95, 200);
        let (lo, picks_lo) = settle(0.2, 200);
        let tail = |p: &[usize]| p[150..].iter().sum::<usize>() as f64 / 50.0;
        assert_eq!(hi.best(), 3, "acceptance {:?}", hi.acceptance());
        assert_eq!(lo.best(), 1, "acceptance {:?}", lo.acceptance());
        assert!(tail(&picks_hi) > 2.5 && tail(&picks_lo) < 1.5);
    }

    #[test]
    fn it_keeps_probing_a_neighbour() {
        let (_, picks) = settle(0.95, 200);
        assert!(
            picks[100..].contains(&2),
            "never re-timed depth 2 once 3 won"
        );
    }

    #[test]
    fn a_rejection_is_only_counted_where_it_happened() {
        let mut c = DepthController::new(3, 0.5);
        // Depth 3, first draft rejected: position 1 learns, 2 and 3 do not.
        c.record(3, 0, 40.0);
        let a = c.acceptance().to_vec();
        assert!(a[0] < 0.5 && a[1] == 0.5 && a[2] == 0.5, "{a:?}");
        // Depth 2, both accepted: positions 1 and 2 go up, 3 unseen.
        c.record(2, 2, 40.0);
        let b = c.acceptance();
        assert!(b[0] > a[0] && b[1] > 0.5 && b[2] == 0.5, "{b:?}");
    }
}
