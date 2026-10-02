//! Choosing a kernel's schedule on the device it will run on.
//!
//! A schedule -- how a matvec spreads its rows over simdgroups, how a GEMM
//! tiles -- has no single best value. Measured: the decode matvec wanted
//! one row a simdgroup on an M4 Max and two on an L4, and on the L4 two
//! shapes of the same kernel wanted different ones (`docs/roadmap-weeks.md`).
//! A constant per target is right for the shape it was measured on and
//! wrong for some other. So a kernel declares its candidates, and the first
//! load on a machine times them on the model's own weights and keeps the
//! fastest -- once: the choice is cached per device, and later loads read it.
//!
//! A candidate is kept only if its output matches the default schedule's;
//! a faster kernel that is wrong is not a candidate. The search has a time
//! budget; a shape it does not reach keeps its default and is not cached,
//! so a later load finishes the job.
//!
//! `LEX_TUNE=0` turns it off (every kernel at its default), `LEX_TUNE=retune`
//! ignores the cache, `LEX_TUNE_SECONDS` sets the budget (default 60).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Use,
    Retune,
}

/// Choices made and cached for one device.
pub struct Tuner {
    mode: Mode,
    path: Option<PathBuf>,
    cache: BTreeMap<String, String>,
    deadline: Instant,
    dirty: bool,
    /// What this load measured, for the log: (key, chosen, gain over the
    /// default as a fraction).
    pub measured: Vec<(String, String, f64)>,
    /// Shapes the budget did not reach.
    pub skipped: usize,
}

impl Tuner {
    /// The tuner for `device`, its cache under `~/.cache/lex-gpu/tune/`.
    pub fn open(device: &str) -> Tuner {
        let mode = match std::env::var("LEX_TUNE").as_deref() {
            Ok("0") | Ok("off") => Mode::Off,
            Ok("retune") => Mode::Retune,
            _ => Mode::Use,
        };
        let budget = std::env::var("LEX_TUNE_SECONDS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(60);
        let name: String = device
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        let path = std::env::var_os("HOME").map(|h| {
            PathBuf::from(h)
                .join(".cache/lex-gpu/tune")
                .join(format!("{name}.json"))
        });
        let cache = match (&path, mode) {
            (Some(p), Mode::Use) => std::fs::read_to_string(p)
                .ok()
                .and_then(|s| crate::json::Json::parse(&s).ok())
                .and_then(|j| match j {
                    crate::json::Json::Obj(m) => Some(
                        m.iter()
                            .filter_map(|(k, v)| v.str().map(|s| (k.clone(), s.to_string())))
                            .collect(),
                    ),
                    _ => None,
                })
                .unwrap_or_default(),
            _ => BTreeMap::new(),
        };
        Tuner {
            mode,
            path,
            cache,
            deadline: Instant::now() + Duration::from_secs(budget),
            dirty: false,
            measured: vec![],
            skipped: 0,
        }
    }

    /// A tuner that always answers the default: for tests and `LEX_TUNE=0`.
    pub fn off() -> Tuner {
        Tuner {
            mode: Mode::Off,
            path: None,
            cache: BTreeMap::new(),
            deadline: Instant::now(),
            dirty: false,
            measured: vec![],
            skipped: 0,
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The candidate to use for `key`. `name` writes a candidate as the
    /// cache stores it. `measure` times one: `Ok(Some(seconds))`, or
    /// `Ok(None)` when its output disagrees with the default's and it is
    /// therefore not a candidate at all.
    pub fn choose<C: Clone>(
        &mut self,
        key: &str,
        candidates: &[C],
        default: C,
        name: impl Fn(&C) -> String,
        mut measure: impl FnMut(&C) -> Result<Option<f64>, String>,
    ) -> Result<C, String> {
        if self.mode == Mode::Off {
            return Ok(default);
        }
        if let Some(c) = self.cache.get(key) {
            if let Some(hit) = candidates.iter().find(|x| name(x) == *c) {
                return Ok(hit.clone());
            }
        }
        if Instant::now() > self.deadline {
            self.skipped += 1;
            return Ok(default);
        }
        let base = measure(&default)?.ok_or("the default schedule disagreed with itself")?;
        let (mut best, mut best_t) = (default.clone(), base);
        for c in candidates {
            if name(c) == name(&default) {
                continue;
            }
            if let Some(t) = measure(c)? {
                if t < best_t {
                    best = c.clone();
                    best_t = t;
                }
            }
        }
        self.cache.insert(key.to_string(), name(&best));
        self.dirty = true;
        self.measured
            .push((key.to_string(), name(&best), base / best_t - 1.0));
        // At once: a pipeline built after load (a batch size first used by
        // a request) is otherwise measured again on the next load.
        self.save();
        Ok(best)
    }

    /// Write the cache back, if anything was measured.
    pub fn save(&mut self) {
        let (Some(path), true) = (&self.path, self.dirty) else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let body: Vec<String> = self
            .cache
            .iter()
            .map(|(k, v)| format!("  {}: {}", quote(k), quote(v)))
            .collect();
        let _ = std::fs::write(path, format!("{{\n{}\n}}\n", body.join(",\n")));
        self.dirty = false;
    }
}

/// A short, stable digest of a kernel's source: part of every cache key,
/// so a changed kernel is measured afresh rather than given an old choice.
pub fn digest(source: &str) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in source.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:08x}", h >> 32)
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

#[cfg(test)]
mod tests {
    use super::{Mode, Tuner};

    fn fresh(mode: Mode) -> Tuner {
        let mut t = Tuner::off();
        t.mode = mode;
        t.deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        t
    }

    #[test]
    fn it_keeps_the_fastest_candidate_that_agrees() {
        let mut t = fresh(Mode::Retune);
        // 3 is fastest but disagrees with the default; 2 is the fastest
        // that agrees.
        let times = [5.0, 4.0, 3.0, 1.0];
        let got = t
            .choose(
                "k",
                &[0usize, 1, 2, 3],
                0,
                |c| c.to_string(),
                |c| Ok(if *c == 3 { None } else { Some(times[*c]) }),
            )
            .unwrap();
        assert_eq!(got, 2);
        assert_eq!(t.measured[0].1, "2");
        assert!((t.measured[0].2 - (5.0 / 3.0 - 1.0)).abs() < 1e-9);
    }

    #[test]
    fn a_cached_choice_is_not_measured_again() {
        let mut t = fresh(Mode::Use);
        t.cache.insert("k".into(), "1".into());
        let got = t
            .choose(
                "k",
                &[0usize, 1, 2],
                0,
                |c| c.to_string(),
                |_| panic!("measured a cached shape"),
            )
            .unwrap();
        assert_eq!(got, 1);
    }

    #[test]
    fn off_and_past_the_budget_give_the_default() {
        let mut t = Tuner::off();
        let got = t
            .choose(
                "k",
                &[0usize, 1],
                0,
                |c| c.to_string(),
                |_| panic!("measured while off"),
            )
            .unwrap();
        assert_eq!(got, 0);
        let mut t = fresh(Mode::Retune);
        t.deadline = std::time::Instant::now() - std::time::Duration::from_secs(1);
        let got = t
            .choose(
                "k",
                &[0usize, 1],
                0,
                |c| c.to_string(),
                |_| panic!("measured past the budget"),
            )
            .unwrap();
        assert_eq!((got, t.skipped), (0, 1));
    }
}
