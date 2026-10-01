//! Picking the next token.
//!
//! This model's own `generation_config.json` says `do_sample: true`, with
//! temperature 1.0, top-k 20 and top-p 0.95, and Ollama runs it that way.
//! Decoding it greedily instead is not a neutral simplification: Qwen's
//! guidance for its thinking models warns that greedy decoding leads to
//! repetition and endless generation, and that is what it did here -- one
//! lex-code task failed twice in a row on a request that ran past the
//! client's ten-minute timeout while three harder ones passed.
//!
//! Temperature 0 still means argmax, which is what the golden tests and
//! every token-for-token comparison against a reference need.

/// splitmix64, to turn a seed into a state.
///
/// `seed | 1` was the first attempt and it gives 42 and 43 the same
/// stream, which a test caught: forcing the low bit is not seeding, it is
/// rounding. This decorrelates neighbours and cannot return zero, which
/// xorshift would never leave.
fn mix(seed: u64) -> u64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (z ^ (z >> 31)) | 1
}

/// The `k` largest logits' indices, largest first, equal logits in index
/// order.
///
/// Speculation takes a nucleus of every row it drafts or verifies, five
/// a cycle at depth 2, so this is on the decode path. Selecting over an
/// index array of the whole vocabulary took 0.40 ms a row on the M4 Max;
/// one pass keeping the best `k` so far takes 0.10: past the first few
/// hundred entries almost nothing beats the k-th, and the pass is a
/// compare. A large `k` would make each insert cost, so it selects.
fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let desc = |a: &u32, b: &u32| {
        logits[*b as usize]
            .total_cmp(&logits[*a as usize])
            .then(a.cmp(b))
    };
    if k > 64 {
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.select_nth_unstable_by(k - 1, desc);
        idx.truncate(k);
        idx.sort_unstable_by(desc);
        return idx;
    }
    let mut top: Vec<u32> = Vec::with_capacity(k + 1);
    for (i, x) in logits.iter().enumerate() {
        if top.len() == k && x.total_cmp(&logits[top[k - 1] as usize]).is_le() {
            continue;
        }
        // After every entry at least as large: equal logits stay in index
        // order.
        let at = top.partition_point(|&j| logits[j as usize].total_cmp(x).is_ge());
        top.insert(at, i as u32);
        top.truncate(k);
    }
    top
}

/// A distribution over a few tokens, most likely first, as
/// [`Sampler::nucleus`] gives it: the tokens and their probabilities.
pub type Nucleus = (Vec<u32>, Vec<f32>);

/// How to turn logits into a token.
#[derive(Clone, Copy, Debug)]
pub struct Sampler {
    /// 0 is greedy.
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    state: u64,
}

impl Default for Sampler {
    /// The checkpoint's own generation config.
    fn default() -> Self {
        Sampler {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
            state: 0x2545_F491_4F6C_DD1D,
        }
    }
}

impl Sampler {
    /// Everything spelled out. `temperature` 0 is greedy.
    pub fn new(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Self {
        Sampler {
            temperature,
            top_p,
            top_k: top_k.max(1),
            state: mix(seed),
        }
    }

    /// Seeded, so a run can be repeated exactly.
    pub fn seeded(seed: u64) -> Self {
        Sampler {
            state: mix(seed),
            ..Sampler::default()
        }
    }

    fn next_unit(&mut self) -> f32 {
        // xorshift64*: a whole RNG crate for one stream of doubles is not
        // a dependency this repository needs.
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let v = x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        v as f32 / (1u64 << 53) as f32
    }

    /// The tokens this sampler can draw, most likely first, with their
    /// probabilities -- temperature, then top-k, then top-p, renormalised.
    /// Greedy is the one-token nucleus.
    ///
    /// Everything below is defined on this, which is what keeps `pick`,
    /// `prob` and `verify_draft` describing the same distribution. A
    /// speculative rule that drew from one distribution and checked
    /// against another would be wrong in a way no output would show.
    pub fn nucleus(&self, logits: &[f32]) -> Nucleus {
        if self.temperature <= 0.0 || self.top_k == 1 {
            let best = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .expect("logits") as u32;
            return (vec![best], vec![1.0]);
        }
        // Top-k first: the vocabulary is 248320 wide and all but a handful
        // of it is noise, so everything below is done on k entries.
        let k = self.top_k.min(logits.len());
        let mut idx = top_k(logits, k);

        let top = logits[idx[0] as usize];
        let mut p: Vec<f32> = idx
            .iter()
            .map(|&i| ((logits[i as usize] - top) / self.temperature).exp())
            .collect();
        let sum: f32 = p.iter().sum();
        for x in &mut p {
            *x /= sum;
        }
        // Top-p over the same, already sorted, list. At least one always
        // survives -- a nucleus can be empty only by rounding.
        let mut cut = p.len();
        let mut run = 0.0;
        for (j, &x) in p.iter().enumerate() {
            run += x;
            if run >= self.top_p {
                cut = j + 1;
                break;
            }
        }
        idx.truncate(cut);
        p.truncate(cut);
        let kept: f32 = p.iter().sum();
        for x in &mut p {
            *x /= kept;
        }
        (idx, p)
    }

    /// One draw from a nucleus, as [`Self::nucleus`] returns it.
    pub fn draw(&mut self, idx: &[u32], p: &[f32]) -> u32 {
        let r = self.next_unit();
        let mut acc = 0.0;
        for (j, &x) in p.iter().enumerate() {
            acc += x;
            if acc > r {
                return idx[j];
            }
        }
        // Rounding can leave the sum a hair under 1.
        idx[idx.len() - 1]
    }

    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        let (idx, p) = self.nucleus(logits);
        self.draw(&idx, &p)
    }

    /// The probability `pick` would give `token`: zero outside the nucleus.
    pub fn prob(&self, logits: &[f32], token: u32) -> f32 {
        let (idx, p) = self.nucleus(logits);
        idx.iter().position(|&t| t == token).map_or(0.0, |i| p[i])
    }

    /// Speculative sampling's accept/reject for one drafted token.
    ///
    /// The draft head drafts greedily, so its proposal is a point mass on
    /// `draft` and the general rule -- accept with min(1, p/q), else draw
    /// from (p - q)+ -- collapses to: accept with probability p(draft),
    /// else draw from p with `draft` removed. What comes out is then
    /// distributed exactly as `pick` would have been:
    ///
    ///   P(y) = p(x)[y = x] + (1 - p(x)) p(y) / (1 - p(x)) [y != x] = p(y)
    ///
    /// so speculation changes how fast the tokens arrive and nothing about
    /// which ones. At temperature 0, p(draft) is exactly 1 or 0 and this is
    /// the greedy check it replaces.
    ///
    /// `Ok(())` accepts the draft; `Err(t)` rejects it, `t` being the token
    /// to say instead.
    pub fn verify_draft(&mut self, logits: &[f32], draft: u32) -> Result<(), u32> {
        let (idx, p) = self.nucleus(logits);
        let px = idx.iter().position(|&t| t == draft).map_or(0.0, |i| p[i]);
        if self.next_unit() < px {
            return Ok(());
        }
        let (ri, mut rp): (Vec<u32>, Vec<f32>) = idx
            .iter()
            .zip(&p)
            .filter(|(t, _)| **t != draft)
            .map(|(t, x)| (*t, *x))
            .unzip();
        let left: f32 = rp.iter().sum();
        if ri.is_empty() || left <= 0.0 {
            // Only reachable if the nucleus was the draft alone, which
            // p(draft) = 1 has already accepted; kept for rounding.
            return Err(idx[0]);
        }
        for x in &mut rp {
            *x /= left;
        }
        Err(self.draw(&ri, &rp))
    }

    /// Speculative sampling's accept/reject for a draft *sampled* from a
    /// proposal `q` (`q_idx`, `q_p`, as [`Self::nucleus`] gives them):
    /// accept with min(1, p(x)/q(x)), else draw from (p - q)+ normalised.
    /// Out comes `p` exactly, whatever `q` was (Leviathan et al.):
    ///
    ///   P(y) = q(y) min(1, p(y)/q(y)) + (1 - sum_x min(p, q)) (p(y) - q(y))+ / Z
    ///        = min(p(y), q(y)) + (p(y) - q(y))+ = p(y)
    ///
    /// since Z = 1 - sum_x min(p(x), q(x)). Drafting the head's argmax and
    /// accepting with p(x) ([`Self::verify_draft`]) is the special case of
    /// a one-point `q`, and accepts sum min(p, q) less often whenever the
    /// two spread their mass over the same few tokens -- at temperature 1
    /// over top-20, most steps. At temperature 0 both are one point and
    /// this is the greedy check.
    pub fn verify_proposal(
        &mut self,
        logits: &[f32],
        draft: u32,
        q_idx: &[u32],
        q_p: &[f32],
    ) -> Result<(), u32> {
        let (idx, p) = self.nucleus(logits);
        let at =
            |ix: &[u32], v: &[f32], t: u32| ix.iter().position(|&x| x == t).map_or(0.0, |i| v[i]);
        let (px, qx) = (at(&idx, &p, draft), at(q_idx, q_p, draft));
        if qx > 0.0 && self.next_unit() < (px / qx).min(1.0) {
            return Ok(());
        }
        // (p - q)+ over p's support: where p is zero it is zero.
        let (ri, mut rp): (Vec<u32>, Vec<f32>) = idx
            .iter()
            .zip(&p)
            .map(|(&t, &x)| (t, (x - at(q_idx, q_p, t)).max(0.0)))
            .filter(|(_, x)| *x > 0.0)
            .unzip();
        let left: f32 = rp.iter().sum();
        if ri.is_empty() || left <= 0.0 {
            // p <= q everywhere means p == q, which accepts with
            // certainty; reachable only by rounding.
            return Err(self.draw(&idx, &p));
        }
        for x in &mut rp {
            *x /= left;
        }
        Err(self.draw(&ri, &rp))
    }
}

#[cfg(test)]
mod tests {
    use super::Sampler;

    /// Logits whose nucleus under the default sampler is a handful of
    /// tokens with spread-out mass.
    fn logits(bias: &[f32]) -> Vec<f32> {
        let mut v = vec![-30.0f32; 64];
        for (i, b) in bias.iter().enumerate() {
            v[i] = *b;
        }
        v
    }

    /// Draw a draft from q, verify against p, and the tokens that come out
    /// are distributed as p -- measured, over many draws -- while accepting
    /// sum min(p, q) of the time, more than a point proposal's p(argmax q).
    #[test]
    fn a_sampled_proposal_yields_the_target_and_accepts_more() {
        let target = logits(&[1.0, 0.8, 0.5, 0.1]);
        let draft = logits(&[0.8, 1.0, 0.4, 0.2]);
        let mut s = Sampler::seeded(7);
        let (pi, pp) = s.nucleus(&target);
        let (qi, qp) = s.nucleus(&draft);
        let n = 200_000;
        let mut count = vec![0usize; 64];
        let mut accepted = 0;
        for _ in 0..n {
            let x = s.draw(&qi, &qp);
            let y = match s.verify_proposal(&target, x, &qi, &qp) {
                Ok(()) => {
                    accepted += 1;
                    x
                }
                Err(t) => t,
            };
            count[y as usize] += 1;
        }
        for (t, p) in pi.iter().zip(&pp) {
            let got = count[*t as usize] as f32 / n as f32;
            assert!((got - p).abs() < 0.006, "token {t}: {got} against p = {p}");
        }
        let overlap: f32 = pi
            .iter()
            .zip(&pp)
            .map(|(t, p)| p.min(qi.iter().position(|q| q == t).map_or(0.0, |i| qp[i])))
            .sum();
        let rate = accepted as f32 / n as f32;
        assert!(
            (rate - overlap).abs() < 0.006,
            "accepted {rate}, sum min(p, q) = {overlap}"
        );
        let point = pp[pi.iter().position(|&t| t == qi[0]).expect("in nucleus")];
        assert!(
            rate > point + 0.05,
            "sampled {rate} against a point proposal's {point}"
        );
    }

    /// At temperature 0 both nuclei are one token and the rule is the
    /// greedy check.
    #[test]
    fn at_temperature_zero_it_is_the_greedy_check() {
        let mut s = Sampler::new(0.0, 1.0, 1, 3);
        let target = logits(&[0.2, 1.0]);
        let (qi, qp) = s.nucleus(&logits(&[1.0, 0.2]));
        assert_eq!(s.verify_proposal(&target, qi[0], &qi, &qp), Err(1));
        let (qi, qp) = s.nucleus(&logits(&[0.1, 0.9]));
        assert_eq!(s.verify_proposal(&target, qi[0], &qi, &qp), Ok(()));
    }

    #[test]
    fn top_k_is_a_full_sorts_prefix() {
        // Coarse values, so there are many ties to order.
        let mut z = 7u64;
        let v: Vec<f32> = (0..5000)
            .map(|_| {
                z = super::mix(z);
                (z % 97) as f32 * 0.25 - 12.0
            })
            .collect();
        let mut all: Vec<u32> = (0..v.len() as u32).collect();
        all.sort_by(|&a, &b| v[b as usize].total_cmp(&v[a as usize]).then(a.cmp(&b)));
        for k in [1, 2, 20, 64, 65, 300, 5000] {
            assert_eq!(super::top_k(&v, k), all[..k], "k = {k}");
        }
    }
}
