//! Where a Qwen3.5-family model's weights come from.
//!
//! The runtime was written against one checkpoint format: Ollama's MLX
//! store, one blob per tensor, every matrix NVFP4. The same architecture
//! also ships as GGUF -- MiMo-v2.6, Bonsai 2, the non-MLX Qwen3.8 -- with
//! different names, different quantisations per tensor, and different
//! conventions for what has already been folded in.
//!
//! This is the one place those differences live. The runtime asks for a
//! tensor by its MLX name and gets back the same values either way: the
//! GGUF side translates the name and undoes whatever its converter did
//! differently, so a convention mistake shows up here, next to the
//! evidence for it, rather than as fluent nonsense forty layers later.

use lex_front::ir::Trits;
use lex_front::llama::{QLayout, Split, split_q4_k, split_q6_k, split_q8_0, split_ternary};

use crate::gguf::{GgmlType, Gguf, gguf_path};
use crate::qwen::Store;

/// A quantised matrix, host side, laid out as the matvec's parameters:
/// the values, the six-bit layouts' high plane, then the scale parameters
/// in `QLayout::scale_params` order -- with NVFP4's per-row tensor scale
/// last, which is where the kernel reads it.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub layout: QLayout,
    pub q: Vec<u8>,
    /// Empty unless the layout is six-bit.
    pub qh: Vec<u8>,
    pub scales: Vec<Vec<u8>>,
}

pub enum Source {
    Mlx(Store),
    Gguf(Box<Gguf>),
}

impl Source {
    /// An Ollama tag, in whichever format its manifest says it is.
    pub fn open(model: &str) -> Result<Source, String> {
        match gguf_path(model) {
            Ok(Some(path)) => Ok(Source::Gguf(Box::new(Gguf::open(&path)?))),
            Ok(None) => Ok(Source::Mlx(Store::open(model)?)),
            Err(e) => Store::open(model)
                .map(Source::Mlx)
                .map_err(|m| format!("{e}; and as MLX: {m}")),
        }
    }

    pub fn has(&self, name: &str) -> bool {
        match self {
            Source::Mlx(s) => s.has(name),
            Source::Gguf(g) => gguf_name(name).is_some_and(|n| g.tensors.contains_key(&n)),
        }
    }

    /// A tensor as f32, with the load-time folds already applied.
    pub fn floats(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        match self {
            Source::Mlx(s) => s.floats(name),
            Source::Gguf(g) => gguf_floats(g, name),
        }
    }

    pub fn matrix(&self, name: &str) -> Result<Matrix, String> {
        match self {
            Source::Mlx(s) => {
                let w = s.nvfp4(name)?;
                Ok(Matrix {
                    rows: w.rows,
                    cols: w.cols,
                    layout: QLayout::NVFP4,
                    q: w.codes,
                    qh: vec![],
                    scales: vec![w.scales, f32_bytes(&w.row_scale)],
                })
            }
            Source::Gguf(g) => gguf_matrix(g, name),
        }
    }

    /// The MLX store, for what only it has: `config.json` and the
    /// tokenizer blob.
    pub fn mlx(&self) -> Option<&Store> {
        match self {
            Source::Mlx(s) => Some(s),
            Source::Gguf(_) => None,
        }
    }

    pub fn gguf(&self) -> Option<&Gguf> {
        match self {
            Source::Gguf(g) => Some(g),
            Source::Mlx(_) => None,
        }
    }
}

/// The GGUF name for an MLX tensor name, or `None` for one the file does
/// not carry -- the draft head, which no GGUF of this family ships.
pub fn gguf_name(mlx: &str) -> Option<String> {
    if let Some(rest) = mlx.strip_prefix("model.language_model.layers.") {
        let (i, tail) = rest.split_once('.')?;
        let g = match tail {
            "input_layernorm.weight" => "attn_norm.weight",
            "post_attention_layernorm.weight" => "post_attention_norm.weight",
            "mlp.gate_proj.weight" => "ffn_gate.weight",
            "mlp.up_proj.weight" => "ffn_up.weight",
            "mlp.down_proj.weight" => "ffn_down.weight",
            "linear_attn.in_proj_qkv.weight" => "attn_qkv.weight",
            "linear_attn.in_proj_z.weight" => "attn_gate.weight",
            "linear_attn.in_proj_a.weight" => "ssm_alpha.weight",
            "linear_attn.in_proj_b.weight" => "ssm_beta.weight",
            "linear_attn.A_log" => "ssm_a",
            "linear_attn.dt_bias" => "ssm_dt.bias",
            "linear_attn.conv1d.weight" => "ssm_conv1d.weight",
            "linear_attn.norm.weight" => "ssm_norm.weight",
            "linear_attn.out_proj.weight" => "ssm_out.weight",
            "self_attn.q_proj.weight" => "attn_q.weight",
            "self_attn.k_proj.weight" => "attn_k.weight",
            "self_attn.v_proj.weight" => "attn_v.weight",
            "self_attn.o_proj.weight" => "attn_output.weight",
            "self_attn.q_norm.weight" => "attn_q_norm.weight",
            "self_attn.k_norm.weight" => "attn_k_norm.weight",
            _ => return None,
        };
        return Some(format!("blk.{i}.{g}"));
    }
    let g = match mlx {
        "model.language_model.embed_tokens.weight" => "token_embd.weight",
        "model.language_model.norm.weight" => "output_norm.weight",
        "lm_head.weight" => "output.weight",
        _ => return None,
    };
    Some(g.to_string())
}

fn located(g: &Gguf, mlx: &str) -> Result<String, String> {
    let n = gguf_name(mlx).ok_or_else(|| format!("{mlx}: no GGUF counterpart"))?;
    if !g.tensors.contains_key(&n) {
        return Err(format!("{mlx}: expected `{n}` in {}", g.path.display()));
    }
    Ok(n)
}

/// A GGUF tensor repacked the way the kernels read it.
fn split(g: &Gguf, name: &str) -> Result<(Split, usize, usize), String> {
    let (t, b) = g.raw(name)?;
    if t.dims.len() != 2 {
        return Err(format!("{name}: {}-d, expected a matrix", t.dims.len()));
    }
    // ggml order: dims[0] is the contiguous one, so it is the row length.
    let (cols, rows) = (t.dims[0], t.dims[1]);
    let s = match t.ty {
        GgmlType::Q4_K => split_q4_k(b, cols),
        GgmlType::Q6_K => split_q6_k(b),
        GgmlType::Q8_0 => split_q8_0(b),
        GgmlType::PTQ1_0 => split_ternary(b, Trits::Dense),
        GgmlType::PQ2_0 => split_ternary(b, Trits::Slots2),
        other => return Err(format!("{name}: {other:?} is not a quantised matrix this reads")),
    };
    Ok((s, rows, cols))
}

fn gguf_matrix(g: &Gguf, mlx: &str) -> Result<Matrix, String> {
    let n = located(g, mlx)?;
    let (s, rows, cols) = split(g, &n)?;
    let bytes = |v: &[i8]| v.iter().map(|&x| x as u8).collect::<Vec<u8>>();
    Ok(Matrix {
        rows,
        cols,
        layout: s.layout,
        q: bytes(&s.q),
        qh: bytes(&s.qh),
        scales: s.scale_bytes(),
    })
}

/// Values of any tensor, dequantised if it is stored quantised: MiMo keeps
/// `ssm_alpha`, `ssm_beta` and the embedding in Q4_K where the MLX
/// checkpoint keeps them in bf16, and the runtime reads all three as f32.
fn values(g: &Gguf, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let (t, b) = g.raw(name)?;
    let shape: Vec<usize> = t.dims.iter().rev().copied().collect();
    let v = match t.ty {
        GgmlType::F32 => b
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        GgmlType::F16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| half::f16::from_le_bytes(*c).to_f32())
            .collect(),
        GgmlType::BF16 => b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect(),
        _ => {
            let (s, rows, cols) = split(g, name)?;
            s.dequant(0, rows * cols)
        }
    };
    Ok((v, shape))
}

fn gguf_floats(g: &Gguf, mlx: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
    let n = located(g, mlx)?;
    let (mut v, shape) = values(g, &n)?;

    // Norm weights: nothing to do, and that is the point. The MLX
    // checkpoint stores them as a delta from 1 and `Store::floats` adds
    // the 1; llama.cpp's converter has already added it. Measured, not
    // assumed: Qwen3.8's input norm averages -0.03 in MLX, MiMo's +1.03 in
    // GGUF. Adding 1 here would double it.

    if mlx.ends_with("A_log") {
        // The runtime wants A = exp(A_log). llama.cpp stores -exp(A_log)
        // as `ssm_a`, so A is its negation. Checked against the working
        // checkpoint: Qwen3.8's exp(A_log) spans 0.0038 to 0.3376, MiMo's
        // -ssm_a 0.0089 to 0.1429 -- the same quantity. Get the sign wrong
        // and the decay is a growth, and the state runs away.
        for x in &mut v {
            *x = -*x;
        }
    }

    if mlx.ends_with("conv1d.weight") && shape.len() == 2 {
        // ggml `(kernel, channels)` is `[channels][kernel]` in memory --
        // the MLX layout without its singleton -- and the runtime wants
        // `[kernel, channels]`, one tap per contiguous row.
        let (ch, kern) = (shape[0], shape[1]);
        let mut t = vec![0.0; ch * kern];
        for c in 0..ch {
            for k in 0..kern {
                t[k * ch + c] = v[c * kern + k];
            }
        }
        return Ok((t, vec![kern, ch]));
    }
    Ok((v, shape))
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
