//! A GGUF reader: metadata, the tensor directory, and raw tensor bytes.
//!
//! Only what a Llama checkpoint needs. Values are parsed (arrays of strings
//! included, because the tokenizer lives in them and must be skipped
//! correctly), tensors are handed out as byte slices of the file — the
//! caller decides how to repack them for the kernels.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(*i),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
}

/// ggml tensor types this reader knows the block layout of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Other(u32),
}

impl GgmlType {
    fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            8 => GgmlType::Q8_0,
            x => GgmlType::Other(x),
        }
    }

    /// (values per block, bytes per block)
    fn block(self) -> Option<(usize, usize)> {
        match self {
            GgmlType::F32 => Some((1, 4)),
            GgmlType::F16 => Some((1, 2)),
            GgmlType::Q8_0 => Some((32, 34)),
            GgmlType::Other(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    /// ggml order: `dims[0]` is the contiguous dimension.
    pub dims: Vec<usize>,
    pub ty: GgmlType,
    offset: usize,
}

impl TensorInfo {
    pub fn elems(&self) -> usize {
        self.dims.iter().product()
    }
}

pub struct Gguf {
    pub path: PathBuf,
    pub meta: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    bytes: Vec<u8>,
    data: usize,
}

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self
            .b
            .get(self.pos..self.pos + n)
            .ok_or("truncated GGUF file")?;
        self.pos += n;
        Ok(s)
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }

    fn value(&mut self, ty: u32) -> Result<Value, String> {
        let v = |b: &[u8]| {
            let mut a = [0u8; 8];
            a[..b.len()].copy_from_slice(b);
            a
        };
        Ok(match ty {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(i32::from_le_bytes(self.take(4)?.try_into().unwrap()) as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut xs = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    xs.push(self.value(et)?);
                }
                Value::Array(xs)
            }
            10 => Value::Int(u64::from_le_bytes(v(self.take(8)?)) as i64),
            11 => Value::Int(i64::from_le_bytes(v(self.take(8)?))),
            12 => Value::Float(f64::from_le_bytes(v(self.take(8)?))),
            t => return Err(format!("unknown GGUF value type {t}")),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut c = Cursor { b: &bytes, pos: 0 };
        if c.take(4)? != b"GGUF" {
            return Err(format!("{} is not a GGUF file", path.display()));
        }
        let version = c.u32()?;
        if version < 2 {
            return Err(format!("GGUF version {version} is too old"));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let k = c.string()?;
            let t = c.u32()?;
            meta.insert(k, c.value(t)?);
        }
        let mut tensors = HashMap::new();
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            let dims = (0..nd)
                .map(|_| c.u64().map(|d| d as usize))
                .collect::<Result<_, _>>()?;
            let ty = GgmlType::from_id(c.u32()?);
            let offset = c.u64()? as usize;
            tensors.insert(name, TensorInfo { dims, ty, offset });
        }
        let align = meta
            .get("general.alignment")
            .and_then(Value::as_int)
            .unwrap_or(32) as usize;
        let data = c.pos.next_multiple_of(align);
        Ok(Gguf {
            path: path.to_path_buf(),
            meta,
            tensors,
            bytes,
            data,
        })
    }

    pub fn int(&self, key: &str) -> Result<i64, String> {
        self.meta
            .get(key)
            .and_then(Value::as_int)
            .ok_or_else(|| format!("GGUF metadata `{key}` missing or not an integer"))
    }

    pub fn float(&self, key: &str) -> Result<f64, String> {
        self.meta
            .get(key)
            .and_then(Value::as_float)
            .ok_or_else(|| format!("GGUF metadata `{key}` missing or not a number"))
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("tensor `{name}` not in {}", self.path.display()))
    }

    /// The tensor's bytes, exactly as stored.
    pub fn raw(&self, name: &str) -> Result<(&TensorInfo, &[u8]), String> {
        let t = self.info(name)?;
        let (per, size) =
            t.ty.block()
                .ok_or_else(|| format!("tensor `{name}` has unsupported type {:?}", t.ty))?;
        let n = t.elems() / per * size;
        let start = self.data + t.offset;
        let b = self
            .bytes
            .get(start..start + n)
            .ok_or_else(|| format!("tensor `{name}` runs past the end of the file"))?;
        Ok((t, b))
    }

    /// An F32 tensor's values.
    pub fn f32s(&self, name: &str) -> Result<Vec<f32>, String> {
        let (t, b) = self.raw(name)?;
        if t.ty != GgmlType::F32 {
            return Err(format!("tensor `{name}` is {:?}, expected F32", t.ty));
        }
        let (words, _) = b.as_chunks::<4>();
        Ok(words.iter().map(|w| f32::from_le_bytes(*w)).collect())
    }
}

/// The GGUF blob behind an Ollama model tag such as `llama3.2:1b`, found
/// through Ollama's manifest (`$OLLAMA_MODELS`, else `~/.ollama/models`).
pub fn ollama_model(tag: &str) -> Result<PathBuf, String> {
    let root = std::env::var_os("OLLAMA_MODELS")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".ollama/models")))
        .ok_or("cannot locate the Ollama model store")?;
    let (repo, t) = tag.split_once(':').unwrap_or((tag, "latest"));
    let manifest = root
        .join("manifests/registry.ollama.ai/library")
        .join(repo)
        .join(t);
    let text = std::fs::read_to_string(&manifest).map_err(|e| {
        format!(
            "no Ollama manifest for {tag} at {}: {e}",
            manifest.display()
        )
    })?;
    // The manifest is small JSON; find the model layer's digest without a
    // JSON dependency.
    let key = "application/vnd.ollama.image.model";
    let at = text
        .find(key)
        .ok_or_else(|| format!("{tag} has no GGUF model layer"))?;
    let rest = &text[at..];
    let d = rest.find("sha256:").ok_or("manifest layer has no digest")?;
    let digest: String = rest[d..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == ':')
        .collect();
    Ok(root.join("blobs").join(digest.replace(':', "-")))
}
