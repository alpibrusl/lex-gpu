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
