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

    pub fn pick(&mut self, logits: &[f32]) -> u32 {
        let best = |v: &[f32]| {
            (0..v.len())
                .max_by(|&a, &b| v[a].total_cmp(&v[b]))
                .expect("logits") as u32
        };
        if self.temperature <= 0.0 || self.top_k == 1 {
            return best(logits);
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
        let total: f32 = p[..cut].iter().sum();
        let r = self.next_unit() * total;
        let mut acc = 0.0;
        for j in 0..cut {
            acc += p[j];
            if acc >= r {
                return idx[j];
            }
        }
        idx[cut - 1]
    }
}
