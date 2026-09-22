//! Reading a Qwen3.5 (MLX) model out of Ollama's store.
//!
//! Ollama keeps one blob per tensor, each headed by a safetensors header:
//! an 8-byte length, that much JSON, then the bytes. A quantised weight is
//! three entries in one blob — `<name>` (NVFP4 codes as `U32`),
//! `<name>.scale` (an FP8 E4M3 byte per 16 values) and
//! `<name>.global_scale` (one f32 for the tensor) — which is exactly what
//! [`tile_front::llama::QLayout::NVFP4`] takes, so the codes and scales go
//! to the GPU as they lie on disk.
//!
//! What this module does change, it changes once, at load:
//! - **Norm weights gain 1.** Qwen3.5's RMSNorm is `x * (1 + w)` and the
//!   checkpoint stores the delta ([`SHIFTED`]). The gated norm inside a
//!   linear-attention layer is not one of them.
//! - **`A_log` becomes `A = exp(A_log)`**, the form the decay gate needs.
//! - **The convolution weight is transposed** from the file's
//!   `[channels, 1, kernel]` to `[kernel, channels]`, so one tap is one
//!   contiguous row.
//! - **The global scale is spread over the rows** of its matrix, because a
//!   matvec reads one per row rather than one per tensor.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::json::Json;

/// Norm weights the checkpoint stores as a delta from 1.
pub const SHIFTED: [&str; 5] = [
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.language_model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
];

/// How a tensor is stored in its blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    /// NVFP4 codes, eight to a word.
    U32,
    /// FP8 E4M3 scale bytes.
    U8,
    BF16,
    F16,
    F32,
}

impl Dtype {
    fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "U32" | "I32" => Dtype::U32,
            "U8" | "I8" => Dtype::U8,
            "BF16" => Dtype::BF16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            _ => return None,
        })
    }
}

/// One entry of a blob's safetensors header.
#[derive(Clone, Debug)]
pub struct Entry {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range within the blob's data section.
    pub range: (usize, usize),
}

/// A model in the local Ollama store: tensor names to blob paths.
pub struct Store {
    pub model: String,
    blobs: BTreeMap<String, PathBuf>,
}

fn store_root() -> PathBuf {
    std::env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".ollama/models")
        })
}

impl Store {
    /// Open `name:tag` (default tag `latest`).
    pub fn open(model: &str) -> Result<Store, String> {
        let (name, tag) = model.split_once(':').unwrap_or((model, "latest"));
        let root = store_root();
        let path = root
            .join("manifests/registry.ollama.ai/library")
            .join(name)
            .join(tag);
        let text = fs::read_to_string(&path).map_err(|e| {
            format!(
                "{model} is not in the Ollama store ({}): {e}",
                path.display()
            )
        })?;
        let j = Json::parse(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let layers = j
            .get("layers")
            .and_then(Json::arr)
            .ok_or("manifest has no layers")?;
        let mut blobs = BTreeMap::new();
        for l in layers {
            let (Some(name), Some(digest)) = (
                l.get("name").and_then(Json::str),
                l.get("digest").and_then(Json::str),
            ) else {
                continue; // The licence and parameter layers carry no name.
            };
            let file = digest.replace(':', "-");
            blobs.insert(name.to_string(), root.join("blobs").join(file));
        }
        if blobs.is_empty() {
            return Err(format!("{model} has no named tensors"));
        }
        Ok(Store {
            model: model.to_string(),
            blobs,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.blobs.keys().map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.blobs.contains_key(name)
    }

    /// A blob's bytes and its header entries.
    fn blob(&self, name: &str) -> Result<(Vec<u8>, BTreeMap<String, Entry>), String> {
        let path = self
            .blobs
            .get(name)
            .ok_or_else(|| format!("no tensor `{name}` in {}", self.model))?;
        let mut raw = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if raw.len() < 8 {
            return Err(format!("{}: too short for a header", path.display()));
        }
        let len = u64::from_le_bytes(raw[..8].try_into().expect("8 bytes")) as usize;
        let head = raw
            .get(8..8 + len)
            .ok_or_else(|| format!("{}: header runs past the blob", path.display()))?;
        let j = Json::parse(&String::from_utf8_lossy(head))?;
        let mut entries = BTreeMap::new();
        for key in j.keys() {
            if key == "__metadata__" {
                continue;
            }
            let e = j.get(key).expect("key from this object");
            let (Some(dt), Some(shape), Some(off)) = (
                e.get("dtype").and_then(Json::str).and_then(Dtype::parse),
                e.get("shape").and_then(Json::arr),
                e.get("data_offsets").and_then(Json::arr),
            ) else {
                return Err(format!("{key}: unreadable header entry"));
            };
            let shape: Vec<usize> = shape.iter().filter_map(Json::usize).collect();
            let off: Vec<usize> = off.iter().filter_map(Json::usize).collect();
            if off.len() != 2 {
                return Err(format!("{key}: bad data_offsets"));
            }
            entries.insert(
                key.to_string(),
                Entry {
                    dtype: dt,
                    shape,
                    range: (off[0], off[1]),
                },
            );
        }
        Ok((raw.split_off(8 + len), entries))
    }

    /// A tensor's own bytes, with its entry.
    pub fn raw(&self, name: &str) -> Result<(Vec<u8>, Entry), String> {
        let (data, entries) = self.blob(name)?;
        let e = entries
            .get(name)
            .ok_or_else(|| format!("{name}: not in its own blob"))?
            .clone();
        let bytes = data
            .get(e.range.0..e.range.1)
            .ok_or_else(|| format!("{name}: data runs past the blob"))?
            .to_vec();
        Ok((bytes, e))
    }

    /// An NVFP4 weight as the matvec takes it: codes, FP8 scale bytes, the
    /// tensor's scale spread over its rows, and `[rows, cols]`.
    pub fn nvfp4(&self, name: &str) -> Result<Nvfp4, String> {
        let (data, entries) = self.blob(name)?;
        let slice = |key: &str| -> Result<(&[u8], Entry), String> {
            let e = entries
                .get(key)
                .ok_or_else(|| format!("{key}: missing (is this weight NVFP4?)"))?;
            let b = data
                .get(e.range.0..e.range.1)
                .ok_or_else(|| format!("{key}: data runs past the blob"))?;
            Ok((b, e.clone()))
        };
        let (codes, ce) = slice(name)?;
        let (scales, _) = slice(&format!("{name}.scale"))?;
        let (gs, _) = slice(&format!("{name}.global_scale"))?;
        if ce.shape.len() != 2 {
            return Err(format!("{name}: expected a matrix, got {:?}", ce.shape));
        }
        let (rows, words) = (ce.shape[0], ce.shape[1]);
        let gs = f32::from_le_bytes(
            gs.get(..4)
                .ok_or("global scale is not four bytes")?
                .try_into()
                .expect("4 bytes"),
        );
        Ok(Nvfp4 {
            rows,
            cols: words * 8,
            codes: codes.to_vec(),
            scales: scales.to_vec(),
            // One per row: a matvec reads its row's scale with its row.
            row_scale: vec![gs; rows],
        })
    }

    /// A tensor as f32, with the load-time folds applied: `1 +` for the
    /// norm weights Qwen3.5 stores as deltas, `exp` for `A_log`, and the
    /// convolution weight transposed to `[kernel, channels]`.
    pub fn floats(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let (bytes, e) = self.raw(name)?;
        let n: usize = e.shape.iter().product();
        let mut v: Vec<f32> = match e.dtype {
            Dtype::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect(),
            Dtype::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes(*c).to_f32())
                .collect(),
            Dtype::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
                .collect(),
            other => return Err(format!("{name}: {other:?} is not a float tensor")),
        };
        if v.len() != n {
            return Err(format!(
                "{name}: {} values for shape {:?}",
                v.len(),
                e.shape
            ));
        }
        if SHIFTED.iter().any(|s| name.ends_with(s)) {
            for x in &mut v {
                *x += 1.0;
            }
        }
        if name.ends_with("A_log") {
            for x in &mut v {
                *x = x.exp();
            }
        }
        if name.ends_with("conv1d.weight") && e.shape.len() == 3 {
            // `[channels, 1, kernel]` -> `[kernel, channels]`.
            let (ch, kern) = (e.shape[0], e.shape[2]);
            let mut t = vec![0.0; ch * kern];
            for c in 0..ch {
                for k in 0..kern {
                    t[k * ch + c] = v[c * kern + k];
                }
            }
            return Ok((t, vec![kern, ch]));
        }
        Ok((v, e.shape))
    }

    /// The `config.json` blob's text model section.
    pub fn config(&self) -> Result<Json, String> {
        let path = self
            .blobs
            .get("config.json")
            .ok_or("no config.json in the manifest")?;
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let j = Json::parse(&text)?;
        j.get("text_config")
            .cloned()
            .ok_or_else(|| "config.json has no text_config".to_string())
    }

    /// The tokenizer blob's path, for a caller that wants to parse it.
    pub fn tokenizer(&self) -> Option<&Path> {
        self.blobs.get("tokenizer.json").map(PathBuf::as_path)
    }
}

/// An NVFP4 matrix, laid out as the matvec's parameters.
pub struct Nvfp4 {
    pub rows: usize,
    pub cols: usize,
    /// Two 4-bit codes per byte.
    pub codes: Vec<u8>,
    /// One FP8 E4M3 byte per 16 codes.
    pub scales: Vec<u8>,
    /// The tensor's f32 scale, once per row.
    pub row_scale: Vec<f32>,
}
