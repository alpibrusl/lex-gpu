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

/// ggml tensor types this reader knows the block layout of. Named as ggml
/// names them, so they can be grepped for in its source.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_K,
    Q6_K,
    /// Prism's ternary types, which are not in ggml: 128 values per block
    /// against one f16 scale, codes in 2-bit slots or densely packed.
    /// See `docs/ternary.md`.
    PQ2_0,
    PTQ1_0,
    Other(u32),
}

impl GgmlType {
    fn from_id(id: u32) -> GgmlType {
        match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            30 => GgmlType::BF16,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4_K,
            14 => GgmlType::Q6_K,
            142 => GgmlType::PQ2_0,
            143 => GgmlType::PTQ1_0,
            x => GgmlType::Other(x),
        }
    }

    /// (values per block, bytes per block)
    fn block(self) -> Option<(usize, usize)> {
        match self {
            GgmlType::F32 => Some((1, 4)),
            GgmlType::F16 => Some((1, 2)),
            GgmlType::BF16 => Some((1, 2)),
            GgmlType::Q8_0 => Some((32, 34)),
            GgmlType::Q4_K => Some((256, 144)),
            GgmlType::Q6_K => Some((256, 210)),
            GgmlType::PQ2_0 => Some((128, 34)),
            GgmlType::PTQ1_0 => Some((128, 28)),
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

/// What running off the end of the bytes reads as; `open_header` grows
/// its prefix on exactly this and on nothing else.
const TRUNCATED: &str = "truncated GGUF file";

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], String> {
        let s = self
            .b
            .get(self.pos..self.pos + n)
            .ok_or(TRUNCATED)?;
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
        Gguf::parse(path, bytes)
    }

    /// The metadata and the tensor directory, without the tensors.
    ///
    /// A tokenizer needs only the header -- tens of megabytes of vocabulary
    /// and merges -- and `open` reads the whole file, 5.6 GB for MiMo, on
    /// top of the runtime reading it again. This reads a prefix and grows
    /// it until the header fits. `raw` then fails for every tensor, which
    /// is the honest answer for a file that was not read.
    pub fn open_header(path: &Path) -> Result<Gguf, String> {
        use std::io::{Read, Seek, SeekFrom};
        let io = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut f = std::fs::File::open(path).map_err(io)?;
        let len = f.metadata().map_err(io)?.len() as usize;
        let mut want = 32usize << 20;
        loop {
            let n = want.min(len);
            let mut bytes = vec![0u8; n];
            f.seek(SeekFrom::Start(0)).map_err(io)?;
            f.read_exact(&mut bytes).map_err(io)?;
            match Gguf::parse(path, bytes) {
                Err(e) if e == TRUNCATED && n < len => want *= 2,
                other => return other,
            }
        }
    }

    fn parse(path: &Path, bytes: Vec<u8>) -> Result<Gguf, String> {
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

/// What `ollama_model` says about a tag it found that is not a GGUF.
const NO_GGUF_LAYER: &str = "has no GGUF model layer";

/// `ollama_model`, telling "not a GGUF" apart from "not found": `Ok(None)`
/// is a manifest with no GGUF layer -- an MLX checkpoint, say -- which the
/// caller should read some other way, not an error to report.
pub fn gguf_path(tag: &str) -> Result<Option<PathBuf>, String> {
    match ollama_model(tag) {
        Ok(p) => Ok(Some(p)),
        Err(e) if e.ends_with(NO_GGUF_LAYER) => Ok(None),
        Err(e) => Err(e),
    }
}

/// The GGUF blob behind an Ollama model tag such as `llama3.2:1b`, found
/// through Ollama's manifest (`$OLLAMA_MODELS`, else `~/.ollama/models`).
pub fn ollama_model(tag: &str) -> Result<PathBuf, String> {
    // Where a store might be, in the order worth trying. On Linux the
    // installer creates an `ollama` system user and the service keeps its
    // blobs under that user's home, so `ollama pull` as yourself leaves
    // nothing in *your* `~/.ollama` -- the models are there, just not
    // where a Mac-shaped guess looks. That cost a cloud run.
    let mut roots: Vec<PathBuf> = vec![];
    if let Some(r) = std::env::var_os("OLLAMA_MODELS") {
        roots.push(PathBuf::from(r));
    }
    if let Some(h) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(h).join(".ollama/models"));
    }
    roots.push(PathBuf::from("/usr/share/ollama/.ollama/models"));
    if roots.is_empty() {
        return Err("cannot locate the Ollama model store".into());
    }

    let (repo, t) = tag.split_once(':').unwrap_or((tag, "latest"));
    // `llama3.2` is shorthand for `library/llama3.2`; a tag that names its
    // own namespace, like `maternion/mimo-v2.6`, lives beside `library`
    // rather than inside it. Joining `library/` onto every tag sent those
    // looking for `library/maternion/mimo-v2.6`, which does not exist.
    let rel = if repo.contains('/') {
        format!("manifests/registry.ollama.ai/{repo}")
    } else {
        format!("manifests/registry.ollama.ai/library/{repo}")
    };
    let path_in = |r: &PathBuf| r.join(&rel).join(t);
    // "Not there" and "there but unreadable" are different problems with
    // the same symptom, and telling them apart matters more than it
    // sounds: on a Linux box the store belongs to the `ollama` service
    // user, the installer adds you to its group, and your *running shell*
    // does not get that membership because its groups were fixed at
    // login. The path then exists, `sudo` can read it, and you cannot --
    // which reads exactly like a layout change if the error only says
    // "no manifest". It cost several rented GPUs to see.
    let mut tried: Vec<String> = vec![];
    let mut found = None;
    for r in &roots {
        let m = path_in(r);
        match std::fs::read_to_string(&m) {
            Ok(s) => {
                found = Some((r.clone(), s));
                break;
            }
            Err(e) => tried.push(format!("{} ({})", m.display(), e.kind())),
        }
    }
    // The store is the root the manifest was found under. It used to be
    // worked out by counting five components up from the manifest, which
    // holds only for `library/` tags and would have silently counted wrong
    // for any other depth.
    let (root, text) =
        found.ok_or_else(|| format!("no readable Ollama manifest for {tag}; tried {}", tried.join(", ")))?;
    // The manifest is small JSON; find the model layer's digest without a
    // JSON dependency.
    let key = "application/vnd.ollama.image.model";
    let at = text
        .find(key)
        .ok_or_else(|| format!("{tag} {NO_GGUF_LAYER}"))?;
    let rest = &text[at..];
    let d = rest.find("sha256:").ok_or("manifest layer has no digest")?;
    let digest: String = rest[d..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == ':')
        .collect();
    Ok(root.join("blobs").join(digest.replace(':', "-")))
}
