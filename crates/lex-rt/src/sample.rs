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
    pub fn nucleus(&self, logits: &[f32]) -> (Vec<u32>, Vec<f32>) {
        if self.temperature <= 0.0 || self.top_k == 1 {
            let best = (0..logits.len())
                .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                .expect("logits") as u32;
            return (vec![best], vec![1.0]);
        }
        // Top-k first: the vocabulary is 248320 wide and all but a handful
        // of it is noise, so everything below is done on k entries.
        let k = self.top_k.min(logits.len());
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.select_nth_unstable_by(k - 1, |&a, &b| {
            logits[b as usize].total_cmp(&logits[a as usize])
        });
        idx.truncate(k);
        idx.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));

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

    /// One draw from a nucleus.
    fn draw(&mut self, idx: &[u32], p: &[f32]) -> u32 {
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
}
