//! A Llama decode loop over tile kernels.
//!
//! Every layer op is a typed `tile-front` program, checked against the target
//! and lowered to MSL by `tile-msl` — the same path flash decode takes:
//!
//! ```text
//!   x = embed(token)                                   host glue
//!   per layer:
//!     h = rmsnorm(x)                                   tile
//!     q, k, v = W_q h, W_k h, W_v h                    tile (Q8_0 matvec)
//!     q, k = rope(q), rope(k)                          tile
//!     cache[pos] = k, v                                host glue
//!     o = attention(q, cache[..=pos])                  tile (flash decode)
//!     x = x + W_o o                                    tile
//!     x = x + W_down (silu(W_gate h') * W_up h')       tile
//!   logits = W_out rmsnorm(x)                          tile
//! ```
//!
//! "Host glue" is two row copies per step, done by the CPU on unified memory
//! between dispatches. They move a few KB and are listed in `docs/P2.md` as
//! the next things to become kernels.
//!
//! P2's contract is correctness: every dispatch waits for the previous one.

use std::path::Path;

use tile_front::llama::{Split, split_q4_k, split_q6_k, split_q8_0};

use crate::gguf::{GgmlType, Gguf};

/// Hyperparameters, from GGUF metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub n_layer: usize,
    pub dim: usize,
    pub n_head: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_base: f32,
    pub rope_factors: Option<Vec<f32>>,
    /// KV cache capacity, in positions.
    pub max_seq: usize,
}

/// A quantised matrix `[rows, cols]` (Q8_0, Q4_K or Q6_K in the file),
/// repacked into values plus per-group scales and mins.
pub struct QMat {
    pub rows: usize,
    pub cols: usize,
    pub w: Split,
}

impl QMat {
    fn load(g: &Gguf, name: &str) -> Result<QMat, String> {
        let (t, b) = g.raw(name)?;
        if t.dims.len() != 2 {
            return Err(format!(
                "`{name}` is {:?}-d, expected a matrix",
                t.dims.len()
            ));
        }
        let w = match t.ty {
            GgmlType::Q8_0 => split_q8_0(b),
            GgmlType::Q4_K => split_q4_k(b, t.dims[0]),
            GgmlType::Q6_K => split_q6_k(b),
            other => {
                return Err(format!(
                    "`{name}` is {other:?}; supported: Q8_0, Q4_K, Q6_K"
                ));
            }
        };
        Ok(QMat {
            rows: t.dims[1],
            cols: t.dims[0],
            w,
        })
    }

    /// Row `r`, dequantised (the embedding lookup).
    pub fn row(&self, r: usize) -> Vec<f32> {
        self.w.dequant(r * self.cols, (r + 1) * self.cols)
    }
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub wq: QMat,
    pub wk: QMat,
    pub wv: QMat,
    pub wo: QMat,
    pub ffn_norm: Vec<f32>,
    pub gate: QMat,
    pub up: QMat,
    pub down: QMat,
}

pub struct Weights {
    pub cfg: Config,
    pub emb: QMat,
    /// `None` when the output projection is tied to the embedding.
    pub out: Option<QMat>,
    pub out_norm: Vec<f32>,
    pub layers: Vec<Layer>,
}

impl Weights {
    pub fn load(path: &Path, max_seq: usize) -> Result<Weights, String> {
        let g = Gguf::open(path)?;
        let arch = match g.meta.get("general.architecture") {
            Some(crate::gguf::Value::Str(s)) => s.clone(),
            _ => return Err("GGUF has no general.architecture".into()),
        };
        if arch != "llama" {
            return Err(format!(
                "architecture `{arch}` is not supported (llama only)"
            ));
        }
        let int = |k: &str| g.int(&format!("llama.{k}")).map(|v| v as usize);
        let n_head = int("attention.head_count")?;
        let dim = int("embedding_length")?;
        let emb = QMat::load(&g, "token_embd.weight")?;
        let cfg = Config {
            n_layer: int("block_count")?,
            dim,
            n_head,
            n_kv: int("attention.head_count_kv")?,
            head_dim: dim / n_head,
            ffn: int("feed_forward_length")?,
            vocab: emb.rows,
            eps: g.float("llama.attention.layer_norm_rms_epsilon")? as f32,
            rope_base: g.float("llama.rope.freq_base")? as f32,
            rope_factors: g
                .tensors
                .contains_key("rope_freqs.weight")
                .then(|| g.f32s("rope_freqs.weight"))
                .transpose()?,
            max_seq,
        };
        let mut layers = vec![];
        for i in 0..cfg.n_layer {
            let n = |k: &str| format!("blk.{i}.{k}.weight");
            layers.push(Layer {
                attn_norm: g.f32s(&n("attn_norm"))?,
                wq: QMat::load(&g, &n("attn_q"))?,
                wk: QMat::load(&g, &n("attn_k"))?,
                wv: QMat::load(&g, &n("attn_v"))?,
                wo: QMat::load(&g, &n("attn_output"))?,
                ffn_norm: g.f32s(&n("ffn_norm"))?,
                gate: QMat::load(&g, &n("ffn_gate"))?,
                up: QMat::load(&g, &n("ffn_up"))?,
                down: QMat::load(&g, &n("ffn_down"))?,
            });
        }
        let out = g
            .tensors
            .contains_key("output.weight")
            .then(|| QMat::load(&g, "output.weight"))
            .transpose()?;
        Ok(Weights {
            out_norm: g.f32s("output_norm.weight")?,
            cfg,
            emb,
            out,
            layers,
        })
    }
}

impl QMat {
    /// Bytes the kernels read for this matrix: values, scales, mins.
    pub fn device_bytes(&self) -> usize {
        self.w.device_bytes()
    }
}

impl Weights {
    /// Weight bytes one decode step reads (every matrix once; the embedding
    /// table only for the output head when it is tied).
    pub fn bytes_per_token(&self) -> usize {
        let mats: usize = self
            .layers
            .iter()
            .map(|l| {
                [&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down]
                    .iter()
                    .map(|m| m.device_bytes())
                    .sum::<usize>()
            })
            .sum();
        mats + self.out.as_ref().unwrap_or(&self.emb).device_bytes()
    }
}

/// Log-softmax in f64, for comparing against reference log-probabilities.
pub fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = logits.iter().map(|&x| (x as f64 - m).exp()).sum();
    let lz = z.ln() + m;
    logits.iter().map(|&x| x as f64 - lz).collect()
}

#[cfg(target_os = "macos")]
pub use gpu::{Attention, Runner};

#[cfg(target_os = "macos")]
mod gpu {
    use std::cell::RefCell;
    use std::collections::hash_map::Entry;
    use std::collections::{BTreeMap, HashMap};
    use std::time::Instant;

    use half::f16;
    use tile_front::flash::{COMBINE_CHUNK, FlashDecode};
    use tile_front::llama::{
        QLayout, kv_append, kv_append_rows, matmul_q, matmul_q_glu, matvec_q, matvec_q_rms,
        rmsnorm, rmsnorm_rows, rope, rope_rows, rope_tables, silu_mul,
    };
    use tile_front::{Program, check};
    use tile_ir::{DType, Space, Target};
    use tile_metal::{Buffer, Gpu, Pipeline, Step};
    use tile_msl::program::lower;

    use super::{Config, QMat, Weights};

    const THREADS: usize = 256;

    /// How a decode step attends over the cache.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Attention {
        /// Blocks of the cache per split of split-KV attention (see
        /// [`FlashDecode::build_split`]).
        pub bps: usize,
        /// Fewest splits worth splitting for: below it, the serial kernel
        /// (one threadgroup per KV head) is cheaper than split + combine.
        pub min_splits: usize,
        /// KV positions per attention block, for every attention kernel;
        /// the cache capacity is rounded up to whole splits of them.
        pub bk: usize,
        /// Read K and V blocks straight from device memory (lazy register
        /// tiles) instead of staging them through threadgroup memory. On by
        /// default: it halves the split kernel's time (8B at 1,440
        /// positions: 57 -> 29 µs per layer) and saves 4 barriers per block.
        pub kv_direct: bool,
    }

    impl Attention {
        fn kv_space(&self) -> Space {
            if self.kv_direct {
                Space::Reg
            } else {
                Space::Threadgroup
            }
        }
    }

    impl Default for Attention {
        /// Two 16-position blocks per split, split from two splits up
        /// (measured on the 8B: one split costs 1.17 ms/token split +
        /// combined, 0.78 serial), K/V read directly. `TILE_BPS`,
        /// `TILE_MIN_SPLITS`, `TILE_ATTN_BK` and `TILE_KV_DIRECT=0` override
        /// it for tuning.
        fn default() -> Attention {
            let env = |k: &str, d: usize| {
                std::env::var(k)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(d)
            };
            Attention {
                bps: env("TILE_BPS", 2),
                min_splits: env("TILE_MIN_SPLITS", 2),
                bk: env("TILE_ATTN_BK", 16),
                kv_direct: env("TILE_KV_DIRECT", 1) != 0,
            }
        }
    }

    /// One dispatch of a step: a profiling label, the pipeline, its buffers,
    /// and a launch smaller than the pipeline's planned grid (x, y), if any.
    type Dispatch<'a> = (
        &'static str,
        &'a Pipeline,
        Vec<&'a Buffer>,
        Option<[usize; 2]>,
    );
    const BO: usize = 8;

    fn compile(gpu: &Gpu, prog: &Program, threads: usize) -> Result<Pipeline, String> {
        let target: &Target = gpu.target();
        check(prog, target).map_err(|errs| {
            let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
            format!("`{}` does not check:\n{}", prog.name, msgs.join("\n"))
        })?;
        gpu.build_lowered(&lower(prog, target, threads)?)
    }

    /// Which matvec pipeline a matrix needs: (cols, rows, layout, residual).
    type MvKey = (usize, usize, QLayout, bool);
    /// Which fused gate/up pipeline a layer needs: (cols, rows, layout).
    type GluKey = (usize, usize, QLayout);

    /// Fused gate/up/SiLU·mul pipelines for `t` rows, one per shape and
    /// layout among layers whose gate and up share a layout.
    fn glu_pipelines(
        gpu: &Gpu,
        w: &Weights,
        t: usize,
        bo: usize,
        norm: Option<f32>,
    ) -> Result<HashMap<GluKey, Pipeline>, String> {
        let mut glu = HashMap::new();
        for l in &w.layers {
            let k = (l.gate.cols, l.gate.rows, l.gate.w.layout);
            if l.up.w.layout == k.2 && !glu.contains_key(&k) {
                let p = matmul_q_glu(t, k.0, k.1, bo, k.2, norm)?;
                glu.insert(k, compile(gpu, &p, THREADS)?);
            }
        }
        Ok(glu)
    }

    /// One matvec-with-folded-norm pipeline per (shape, layout) a decode
    /// step needs: q, k and v read the residual through the attention norm,
    /// the output head through the final norm.
    fn rms_mv_pipelines(
        gpu: &Gpu,
        w: &Weights,
        bo: usize,
        threads: usize,
    ) -> Result<HashMap<GluKey, Pipeline>, String> {
        let mut m: HashMap<GluKey, Pipeline> = HashMap::new();
        let mut want: Vec<&QMat> = vec![w.out.as_ref().unwrap_or(&w.emb)];
        for l in &w.layers {
            want.extend([&l.wq, &l.wk, &l.wv]);
        }
        for q in want {
            let k = (q.cols, q.rows, q.w.layout);
            if let Entry::Vacant(slot) = m.entry(k) {
                let p = matvec_q_rms(k.0, k.1, bo, k.2, w.cfg.eps)?;
                slot.insert(compile(gpu, &p, threads)?);
            }
        }
        Ok(m)
    }

    /// `y = W rmsnorm(x, g)` for a matrix with a folded-norm pipeline.
    fn rms_dispatch<'a>(
        m: &'a HashMap<GluKey, Pipeline>,
        w: &'a QBuf,
        x: &'a Buffer,
        g: &'a Buffer,
        y: &'a Buffer,
    ) -> Option<(&'a Pipeline, Vec<&'a Buffer>)> {
        let p = m.get(&(w.cols, w.rows, w.layout))?;
        let mut bufs = vec![x, g, &w.q];
        bufs.extend(w.qh.as_ref());
        bufs.extend(w.scales.iter());
        bufs.push(y);
        Some((p, bufs))
    }

    /// The fused gate/up dispatch for a layer, if it has one:
    /// `x, <gate>, <up>, a`.
    fn glu_dispatch<'a>(
        glu: &'a HashMap<GluKey, Pipeline>,
        gate: &'a QBuf,
        up: &'a QBuf,
        x: &'a Buffer,
        norm: Option<&'a Buffer>,
        a: &'a Buffer,
    ) -> Option<(&'a Pipeline, Vec<&'a Buffer>)> {
        if gate.layout != up.layout {
            return None;
        }
        let p = glu.get(&(gate.cols, gate.rows, gate.layout))?;
        let mut bufs = vec![x];
        bufs.extend(norm);
        for w in [gate, up] {
            bufs.push(&w.q);
            bufs.extend(w.qh.as_ref());
            bufs.extend(w.scales.iter());
        }
        bufs.push(a);
        Some((p, bufs))
    }

    struct QBuf {
        rows: usize,
        cols: usize,
        layout: QLayout,
        q: Buffer,
        /// Six-bit layouts' 2-bit plane.
        qh: Option<Buffer>,
        /// Scale parameters in `QLayout::scale_params` order, in the file's
        /// own encoding.
        scales: Vec<Buffer>,
    }

    fn upload_q(gpu: &Gpu, w: &QMat) -> QBuf {
        QBuf {
            rows: w.rows,
            cols: w.cols,
            layout: w.w.layout,
            q: gpu.upload(&w.w.q),
            qh: (!w.w.qh.is_empty()).then(|| gpu.upload(&w.w.qh)),
            scales: w.w.scale_bytes().iter().map(|b| gpu.upload(b)).collect(),
        }
    }

    struct LayerBufs {
        attn_norm: Buffer,
        wq: QBuf,
        wk: QBuf,
        wv: QBuf,
        wo: QBuf,
        ffn_norm: Buffer,
        gate: QBuf,
        up: QBuf,
        down: QBuf,
        kcache: Buffer,
        vcache: Buffer,
    }

    struct Kernels {
        rms: Pipeline,
        /// One matvec pipeline per (shape, layout, residual) the model uses:
        /// Q4_K_M files mix Q4_K and Q6_K within a layer.
        mv: HashMap<MvKey, Pipeline>,
        /// Gate + up + SiLU·mul in one kernel, per (n_in, n_out, layout),
        /// for layers whose gate and up share a layout. Decode's also folds
        /// in the FFN's RMSNorm.
        glu: HashMap<GluKey, Pipeline>,
        /// Matvecs with the preceding RMSNorm folded in (decode).
        rms_mv: HashMap<GluKey, Pipeline>,
        /// Which call sites fold their norm: (q/k/v, gate/up, lm head).
        fold: (bool, bool, bool),
        rope_q: Pipeline,
        rope_k: Pipeline,
        silu: Pipeline,
        /// Flash decode over a runtime length: one pipeline for every step.
        /// Serial over the cache, for contexts under `attention.min_splits`.
        attn: Pipeline,
        /// Split-KV decode attention: (head, split) partials, then a merge.
        attn_split: Pipeline,
        attn_combine: Pipeline,
        attention: Attention,
        kv_k: Pipeline,
        kv_v: Pipeline,
    }

    /// Activation buffers, reused every step.
    struct Acts {
        x: Buffer,
        x2: Buffer,
        h: Buffer,
        q32: Buffer,
        k32: Buffer,
        v32: Buffer,
        q16: Buffer,
        k16: Buffer,
        o: Buffer,
        g: Buffer,
        u: Buffer,
        a: Buffer,
        cos: Buffer,
        sin: Buffer,
        logits: Buffer,
        /// Split-KV partial states (max, sum, accumulator), shared by layers.
        part_m: Buffer,
        part_l: Buffer,
        part_acc: Buffer,
    }

    /// A matmul dispatch for weight `w`: the pipeline for its shape and
    /// layout, and its buffers in parameter order.
    fn mv_dispatch<'a>(
        mv: &'a HashMap<MvKey, Pipeline>,
        w: &'a QBuf,
        x: &'a Buffer,
        r: Option<&'a Buffer>,
        y: &'a Buffer,
    ) -> (&'a Pipeline, Vec<&'a Buffer>) {
        let p = &mv[&(w.cols, w.rows, w.layout, r.is_some())];
        let mut bufs = vec![x, &w.q];
        bufs.extend(w.qh.as_ref());
        bufs.extend(w.scales.iter());
        bufs.extend(r);
        bufs.push(y);
        (p, bufs)
    }

    /// Largest batch of tokens one [`Runner::forward`] takes.
    pub const MAX_BATCH: usize = 16;

    /// Kernels for a batch of `t` tokens, compiled on first use.
    struct Batch {
        rms: Pipeline,
        rms_last: Pipeline,
        mv: HashMap<MvKey, Pipeline>,
        glu: HashMap<GluKey, Pipeline>,
        rope_q: Pipeline,
        rope_k: Pipeline,
        kv_k: Pipeline,
        kv_v: Pipeline,
        attn: Pipeline,
        silu: Pipeline,
    }

    /// Activations for up to `MAX_BATCH` tokens.
    struct BatchActs {
        x: Buffer,
        x2: Buffer,
        h: Buffer,
        q32: Buffer,
        k32: Buffer,
        v32: Buffer,
        q16: Buffer,
        k16: Buffer,
        o: Buffer,
        g: Buffer,
        u: Buffer,
        a: Buffer,
        cos: Buffer,
        sin: Buffer,
        logits: Buffer,
        scalars_attn: Buffer,
    }

    /// Runs decode steps for one sequence on the Metal device.
    pub struct Runner<'w> {
        gpu: Gpu,
        w: &'w Weights,
        k: Kernels,
        layers: Vec<LayerBufs>,
        emb: QBuf,
        out: Option<QBuf>,
        out_norm: Buffer,
        acts: Acts,
        /// KV cache capacity, in positions (a whole number of blocks).
        cap: usize,
        scalars_pos: Buffer,
        scalars_attn: Buffer,
        /// Split attention's live length; its combine's split and chunk
        /// counts.
        scalars_len: Buffer,
        scalars_nsplit: Buffer,
        batches: HashMap<usize, Batch>,
        bacts: BatchActs,
        pos: usize,
        /// Dispatch one kernel at a time and record each call site's GPU
        /// time, instead of one command buffer per token. For profiling; the
        /// step itself is much slower.
        pub sync: bool,
        /// Dispatches issued so far.
        pub dispatches: usize,
        /// Call-site labels to leave out of decode steps (ablation: how much
        /// a kind of kernel costs on the real, overlapped critical path).
        /// The logits are then wrong; for profiling only.
        pub skip: Vec<String>,
        /// Wall time per call site: (calls, seconds). Every dispatch waits
        /// for completion, so this is GPU time plus submit/wait overhead.
        prof: RefCell<BTreeMap<&'static str, (usize, f64)>>,
    }

    impl<'w> Runner<'w> {
        pub fn new(w: &'w Weights) -> Result<Runner<'w>, String> {
            Runner::with_attention(w, Attention::default())
        }

        pub fn with_attention(w: &'w Weights, attention: Attention) -> Result<Runner<'w>, String> {
            let bps = attention.bps;
            let gpu = Gpu::open()?;
            let c: &Config = &w.cfg;
            let kvd = c.n_kv * c.head_dim;
            let key = |m: &QMat, res: bool| (m.cols, m.rows, m.w.layout, res);
            let mut keys = vec![key(w.out.as_ref().unwrap_or(&w.emb), false)];
            for l in &w.layers {
                keys.extend([
                    key(&l.wq, false),
                    key(&l.wk, false),
                    key(&l.wv, false),
                    key(&l.wo, true),
                    key(&l.gate, false),
                    key(&l.up, false),
                    key(&l.down, true),
                ]);
            }
            let tune = |k: &str, d: usize| {
                std::env::var(k)
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(d)
            };
            let (mv_bo, mv_threads) = (tune("TILE_MV_BO", BO), tune("TILE_MV_THREADS", THREADS));
            // Folding a norm into a matvec saves a dispatch the whole GPU
            // waits on, but costs every threadgroup a read of x and a
            // reduction. Measured on the M4 Max at 512 positions: the 8B
            // loses (80 tok/s unfolded, 66 folded -- 1,792 threadgroups
            // re-reading x cost more than the dispatches save) and the 1B
            // is a wash (264 / 266). Off by default; the kernels stay,
            // with these switches, because the trade-off moves with the
            // GPU and the model shape.
            let fold = (
                tune("TILE_FOLD_QKV", 0) != 0,
                tune("TILE_FOLD_GLU", 0) != 0,
                tune("TILE_FOLD_HEAD", 0) != 0,
            );
            let mut mv = HashMap::new();
            for kk in keys {
                if let Entry::Vacant(slot) = mv.entry(kk) {
                    let (n_in, n_out, layout, res) = kk;
                    // One chunk per row: lazy operands need no registers,
                    // so each output reduces across its whole row once.
                    let p = matvec_q(n_in, n_out, mv_bo, n_in, layout, res)?;
                    slot.insert(compile(&gpu, &p, mv_threads)?);
                }
            }
            let bk = attention.bk;
            let cap = c.max_seq.next_multiple_of(bk * bps * COMBINE_CHUNK);
            let splits = cap / (bk * bps);
            let group = c.n_head / c.n_kv;
            let attn = FlashDecode {
                q_rows: group,
                d: c.head_dim,
                seq: cap,
                bq: group,
                bk,
                stages: 1,
                dtype: DType::F16,
                kv_space: attention.kv_space(),
                consumers: 0,
                heads: c.n_kv,
                kv_cap: cap,
            };
            let k = Kernels {
                attn: compile(&gpu, &attn.build_dynamic()?, 128)?,
                attn_split: compile(&gpu, &attn.build_split(bps)?, 128)?,
                attn_combine: compile(&gpu, &attn.build_combine(bps)?, 128)?,
                attention,
                kv_k: compile(&gpu, &kv_append(c.n_kv, c.head_dim, cap, DType::F16), 64)?,
                kv_v: compile(&gpu, &kv_append(c.n_kv, c.head_dim, cap, DType::F32), 64)?,
                rms: compile(&gpu, &rmsnorm(c.dim, c.eps), THREADS)?,
                mv,
                rope_q: compile(&gpu, &rope(c.n_head, c.head_dim, DType::F16), THREADS)?,
                rope_k: compile(&gpu, &rope(c.n_kv, c.head_dim, DType::F16), THREADS)?,
                silu: compile(&gpu, &silu_mul(c.ffn, THREADS)?, THREADS)?,
                glu: glu_pipelines(&gpu, w, 1, BO, fold.1.then_some(c.eps))?,
                rms_mv: rms_mv_pipelines(&gpu, w, mv_bo, mv_threads)?,
                fold,
            };
            let layers = w
                .layers
                .iter()
                .map(|l| LayerBufs {
                    attn_norm: gpu.upload(&l.attn_norm),
                    wq: upload_q(&gpu, &l.wq),
                    wk: upload_q(&gpu, &l.wk),
                    wv: upload_q(&gpu, &l.wv),
                    wo: upload_q(&gpu, &l.wo),
                    ffn_norm: gpu.upload(&l.ffn_norm),
                    gate: upload_q(&gpu, &l.gate),
                    up: upload_q(&gpu, &l.up),
                    down: upload_q(&gpu, &l.down),
                    kcache: gpu.zeroed::<f16>(c.n_kv * cap * c.head_dim),
                    vcache: gpu.zeroed::<f16>(c.n_kv * cap * c.head_dim),
                })
                .collect();
            let f = |n: usize| gpu.zeroed::<f32>(n);
            let acts = Acts {
                x: f(c.dim),
                x2: f(c.dim),
                h: f(c.dim),
                q32: f(c.dim),
                k32: f(kvd),
                v32: f(kvd),
                q16: gpu.zeroed::<f16>(c.dim),
                k16: gpu.zeroed::<f16>(kvd),
                o: f(c.dim),
                g: f(c.ffn),
                u: f(c.ffn),
                a: f(c.ffn),
                cos: f(c.head_dim),
                sin: f(c.head_dim),
                logits: f(c.vocab),
                part_m: f(splits * c.n_head),
                part_l: f(splits * c.n_head),
                part_acc: f(splits * c.n_head * c.head_dim),
            };
            let gpu_scalars_pos = gpu.zeroed::<u32>(1);
            let gpu_scalars_attn = gpu.zeroed::<u32>(2);
            let gpu_scalars_len = gpu.zeroed::<u32>(1);
            let gpu_scalars_nsplit = gpu.zeroed::<u32>(2);
            let bt = MAX_BATCH;
            let bacts = BatchActs {
                x: f(bt * c.dim),
                x2: f(bt * c.dim),
                h: f(bt * c.dim),
                q32: f(bt * c.dim),
                k32: f(bt * kvd),
                v32: f(bt * kvd),
                q16: gpu.zeroed::<f16>(bt * c.dim),
                k16: gpu.zeroed::<f16>(bt * kvd),
                o: f(bt * c.dim),
                g: f(bt * c.ffn),
                u: f(bt * c.ffn),
                a: f(bt * c.ffn),
                cos: f(bt * c.head_dim),
                sin: f(bt * c.head_dim),
                logits: f(bt * c.vocab),
                scalars_attn: gpu.zeroed::<u32>(2),
            };
            Ok(Runner {
                emb: upload_q(&gpu, &w.emb),
                out: w.out.as_ref().map(|o| upload_q(&gpu, o)),
                out_norm: gpu.upload(&w.out_norm),
                gpu,
                w,
                k,
                layers,
                acts,
                cap,
                scalars_pos: gpu_scalars_pos,
                scalars_attn: gpu_scalars_attn,
                scalars_len: gpu_scalars_len,
                scalars_nsplit: gpu_scalars_nsplit,
                batches: HashMap::new(),
                bacts,
                pos: 0,
                sync: std::env::var_os("TILE_SYNC").is_some(),
                dispatches: 0,
                skip: vec![],
                prof: RefCell::new(BTreeMap::new()),
            })
        }

        pub fn device(&self) -> String {
            self.gpu.info().name
        }

        pub fn pos(&self) -> usize {
            self.pos
        }

        /// Forget the sequence; the cache is overwritten from position 0.
        pub fn reset(&mut self) {
            self.pos = 0;
        }

        /// Roll the sequence back to `pos` positions (speculative decoding
        /// rejecting drafted tokens). Cache entries past it are simply
        /// overwritten later; attention never reads past the live length.
        pub fn truncate(&mut self, pos: usize) {
            assert!(pos <= self.pos, "truncate forward");
            self.pos = pos;
        }

        fn batch(&mut self, t: usize) -> Result<(), String> {
            if self.batches.contains_key(&t) {
                return Ok(());
            }
            let c = self.w.cfg.clone();
            let gpu = &self.gpu;
            // Rows per threadgroup: 32 for the smallest batches, 16 beyond
            // (measured: register pressure at larger t).
            let bo = if t <= 4 { 32 } else { 16 };
            let mut mv = HashMap::new();
            let key = |m: &QMat, res: bool| (m.cols, m.rows, m.w.layout, res);
            let mut keys = vec![key(self.w.out.as_ref().unwrap_or(&self.w.emb), false)];
            for l in &self.w.layers {
                keys.extend([
                    key(&l.wq, false),
                    key(&l.wk, false),
                    key(&l.wv, false),
                    key(&l.wo, true),
                    key(&l.gate, false),
                    key(&l.up, false),
                    key(&l.down, true),
                ]);
            }
            for kk in keys {
                if let Entry::Vacant(slot) = mv.entry(kk) {
                    let (n_in, n_out, layout, res) = kk;
                    let p = matmul_q(t, n_in, n_out, bo, n_in, layout, res)?;
                    slot.insert(compile(gpu, &p, THREADS)?);
                }
            }
            let group = c.n_head / c.n_kv;
            let attn = FlashDecode {
                q_rows: group,
                d: c.head_dim,
                seq: self.cap,
                bq: group,
                bk: self.k.attention.bk,
                stages: 1,
                dtype: DType::F16,
                kv_space: self.k.attention.kv_space(),
                consumers: 0,
                heads: c.n_kv,
                kv_cap: self.cap,
            };
            let b = Batch {
                rms: compile(gpu, &rmsnorm_rows(t, c.dim, c.eps, None), THREADS)?,
                rms_last: compile(gpu, &rmsnorm_rows(t, c.dim, c.eps, Some(t - 1)), THREADS)?,
                mv,
                rope_q: compile(
                    gpu,
                    &rope_rows(t, c.n_head, c.head_dim, DType::F16),
                    THREADS,
                )?,
                rope_k: compile(gpu, &rope_rows(t, c.n_kv, c.head_dim, DType::F16), THREADS)?,
                kv_k: compile(
                    gpu,
                    &kv_append_rows(t, c.n_kv, c.head_dim, self.cap, DType::F16),
                    64,
                )?,
                kv_v: compile(
                    gpu,
                    &kv_append_rows(t, c.n_kv, c.head_dim, self.cap, DType::F32),
                    64,
                )?,
                attn: compile(gpu, &attn.build_causal(t)?, 128)?,
                silu: compile(gpu, &silu_mul(t * c.ffn, THREADS)?, THREADS)?,
                glu: glu_pipelines(gpu, self.w, t, bo, None)?,
            };
            self.batches.insert(t, b);
            Ok(())
        }

        /// Feed `tokens` (at most [`MAX_BATCH`]) at the next positions in one
        /// pass. Returns the logits after every token (`all`: speculative
        /// verify) or only after the last (prefill).
        pub fn forward(&mut self, tokens: &[u32], all: bool) -> Result<Vec<Vec<f32>>, String> {
            let t = tokens.len();
            if t == 0 || t > MAX_BATCH {
                return Err(format!("a batch is 1..={MAX_BATCH} tokens, not {t}"));
            }
            if self.pos + t > self.cap {
                return Err(format!("KV cache is full ({} positions)", self.cap));
            }
            self.batch(t)?;
            let c = self.w.cfg.clone();
            let pos0 = self.pos;
            let (dim, hd) = (c.dim, c.head_dim);
            for (i, &tok) in tokens.iter().enumerate() {
                self.gpu
                    .write(&self.bacts.x, i * dim, &self.w.emb.row(tok as usize));
                let (cs, sn) = rope_tables(pos0 + i, hd, c.rope_base, c.rope_factors.as_deref());
                self.gpu.write(&self.bacts.cos, i * hd, &cs);
                self.gpu.write(&self.bacts.sin, i * hd, &sn);
            }
            self.gpu.write(&self.scalars_pos, 0, &[pos0 as u32]);
            self.gpu.write(
                &self.bacts.scalars_attn,
                0,
                &[pos0 as u32, (pos0 + t).div_ceil(self.k.attention.bk) as u32],
            );
            let n = {
                let bk = &self.batches[&t];
                let a = &self.bacts;
                let mv = |w, x, r, y| mv_dispatch(&bk.mv, w, x, r, y);
                let mut d: Vec<(&Pipeline, Vec<&Buffer>)> = vec![];
                for l in &self.layers {
                    d.push((&bk.rms, vec![&a.x, &l.attn_norm, &a.h]));
                    d.push(mv(&l.wq, &a.h, None, &a.q32));
                    d.push(mv(&l.wk, &a.h, None, &a.k32));
                    d.push(mv(&l.wv, &a.h, None, &a.v32));
                    d.push((&bk.rope_q, vec![&a.q32, &a.cos, &a.sin, &a.q16]));
                    d.push((&bk.rope_k, vec![&a.k32, &a.cos, &a.sin, &a.k16]));
                    d.push((&bk.kv_k, vec![&a.k16, &l.kcache, &self.scalars_pos]));
                    d.push((&bk.kv_v, vec![&a.v32, &l.vcache, &self.scalars_pos]));
                    d.push((
                        &bk.attn,
                        vec![&a.q16, &l.kcache, &l.vcache, &a.o, &a.scalars_attn],
                    ));
                    d.push(mv(&l.wo, &a.o, Some(&a.x), &a.x2));
                    d.push((&bk.rms, vec![&a.x2, &l.ffn_norm, &a.h]));
                    match glu_dispatch(&bk.glu, &l.gate, &l.up, &a.h, None, &a.a) {
                        Some(g) => d.push(g),
                        None => {
                            d.push(mv(&l.gate, &a.h, None, &a.g));
                            d.push(mv(&l.up, &a.h, None, &a.u));
                            d.push((&bk.silu, vec![&a.g, &a.u, &a.a]));
                        }
                    }
                    d.push(mv(&l.down, &a.a, Some(&a.x2), &a.x));
                }
                let out = self.out.as_ref().unwrap_or(&self.emb);
                if all {
                    d.push((&bk.rms, vec![&a.x, &self.out_norm, &a.h]));
                    d.push(mv(out, &a.h, None, &a.logits));
                } else {
                    // Only the last token's logits: its row, then the
                    // single-token output head.
                    d.push((&bk.rms_last, vec![&a.x, &self.out_norm, &self.acts.h]));
                    let p = &self.k.mv[&(out.cols, out.rows, out.layout, false)];
                    let mut bufs = vec![&self.acts.h, &out.q];
                    bufs.extend(out.qh.as_ref());
                    bufs.extend(out.scales.iter());
                    bufs.push(&self.acts.logits);
                    d.push((p, bufs));
                }
                let steps: Vec<(&Pipeline, &[&Buffer])> =
                    d.iter().map(|(p, b)| (*p, b.as_slice())).collect();
                let t0 = Instant::now();
                self.gpu.run_all(&steps);
                let mut prof = self.prof.borrow_mut();
                let e = prof.entry("batch (one command buffer)").or_insert((0, 0.0));
                e.0 += 1;
                e.1 += t0.elapsed().as_secs_f64();
                d.len()
            };
            self.dispatches += n;
            self.pos += t;
            let v = c.vocab;
            if all {
                let mut flat = vec![0.0f32; t * v];
                self.gpu.download(&self.bacts.logits, &mut flat);
                Ok(flat.chunks(v).map(|r| r.to_vec()).collect())
            } else {
                let mut last = vec![0.0f32; v];
                self.gpu.download(&self.acts.logits, &mut last);
                Ok(vec![last])
            }
        }

        /// Feed a whole prompt in batches of `chunk` tokens; returns the logits
        /// after its last token.
        pub fn prefill(&mut self, tokens: &[u32], chunk: usize) -> Result<Vec<f32>, String> {
            let mut last = vec![];
            for part in tokens.chunks(chunk.clamp(1, MAX_BATCH)) {
                last = self.forward(part, false)?.pop().expect("one row");
            }
            Ok(last)
        }

        /// Feed one token at the next position; returns the logits.
        pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
            let c = self.w.cfg.clone();
            if self.pos >= self.cap {
                return Err(format!("KV cache is full ({} positions)", self.cap));
            }
            let pos = self.pos;

            // Everything the CPU contributes happens here, before the token's
            // command buffer: the embedding row, this position's RoPE tables,
            // and the runtime scalars (position, live length, block count).
            let x = self.w.emb.row(token as usize);
            self.gpu.write(&self.acts.x, 0, &x);
            let (cos, sin) = rope_tables(pos, c.head_dim, c.rope_base, c.rope_factors.as_deref());
            self.gpu.write(&self.acts.cos, 0, &cos);
            self.gpu.write(&self.acts.sin, 0, &sin);
            let len = pos + 1;
            self.gpu.write(&self.scalars_pos, 0, &[pos as u32]);
            self.gpu.write(
                &self.scalars_attn,
                0,
                &[len as u32, len.div_ceil(self.k.attention.bk) as u32],
            );
            self.gpu.write(&self.scalars_len, 0, &[len as u32]);
            let nsplit = self.nsplit(len);
            self.gpu.write(
                &self.scalars_nsplit,
                0,
                &[nsplit as u32, nsplit.div_ceil(COMBINE_CHUNK) as u32],
            );

            let mut plan = self.plan();
            if !self.skip.is_empty() {
                plan.retain(|d| !self.skip.iter().any(|s| d.0.starts_with(s.as_str())));
            }
            let n = plan.len();
            if self.sync {
                // GPU time per dispatch, from the command buffer's own
                // timestamps: the CPU round trip is not the kernel's cost.
                for (label, p, bufs, groups) in &plan {
                    let (_, t) = self.gpu.run_launches(&[(p, bufs, *groups)]);
                    let mut prof = self.prof.borrow_mut();
                    let e = prof.entry(label).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += t;
                }
            } else {
                let steps: Vec<Step<'_>> = plan
                    .iter()
                    .map(|(_, p, b, g)| (*p, b.as_slice(), *g))
                    .collect();
                let t = Instant::now();
                let (encode, gpu) = self.gpu.run_launches(&steps);
                let wall = t.elapsed().as_secs_f64();
                let mut prof = self.prof.borrow_mut();
                for (label, secs) in [
                    ("token: wall, encode to completion", wall),
                    ("token: CPU encoding", encode),
                    ("token: GPU execution", gpu),
                ] {
                    let e = prof.entry(label).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += secs;
                }
            }
            drop(plan);
            self.dispatches += n;
            self.pos += 1;
            let mut logits = vec![0.0f32; c.vocab];
            self.gpu.download(&self.acts.logits, &mut logits);
            Ok(logits)
        }

        /// Per-call-site timings so far, slowest first: (label, calls, seconds).
        pub fn profile(&self) -> Vec<(&'static str, usize, f64)> {
            let mut v: Vec<_> = self
                .prof
                .borrow()
                .iter()
                .map(|(k, (n, t))| (*k, *n, *t))
                .collect();
            v.sort_by(|a, b| b.2.total_cmp(&a.2));
            v
        }

        pub fn clear_profile(&self) {
            self.prof.borrow_mut().clear();
        }

        /// Matvec pipelines compiled for this model.
        pub fn matvec_variants(&self) -> usize {
            self.k.mv.len()
        }

        /// `y = W x (+ r)` with the pipeline for `W`'s shape and layout.
        fn mv<'a>(
            &'a self,
            label: &'static str,
            w: &'a QBuf,
            x: &'a Buffer,
            r: Option<&'a Buffer>,
            y: &'a Buffer,
        ) -> Dispatch<'a> {
            let p = &self.k.mv[&(w.cols, w.rows, w.layout, r.is_some())];
            let mut bufs = vec![x, &w.q];
            bufs.extend(w.qh.as_ref());
            bufs.extend(w.scales.iter());
            bufs.extend(r);
            bufs.push(y);
            (label, p, bufs, None)
        }

        /// Splits of the cache a step at live length `len` launches.
        fn nsplit(&self, len: usize) -> usize {
            len.div_ceil(self.k.attention.bk * self.k.attention.bps)
        }

        /// Every dispatch of one decode step, in order.
        fn plan(&self) -> Vec<Dispatch<'_>> {
            let (a, k) = (&self.acts, &self.k);
            let mut d: Vec<Dispatch<'_>> = vec![];
            for l in &self.layers {
                // q, k and v read the residual with the attention RMSNorm
                // folded in: no separate norm dispatch, and no barrier the
                // whole GPU waits on before the three of them.
                let normed = k
                    .fold
                    .0
                    .then(|| {
                        [
                            ("matvec q", &l.wq, &a.q32),
                            ("matvec k/v", &l.wk, &a.k32),
                            ("matvec k/v", &l.wv, &a.v32),
                        ]
                        .into_iter()
                        .map(|(label, w, y)| {
                            rms_dispatch(&k.rms_mv, w, &a.x, &l.attn_norm, y)
                                .map(|(p, bufs)| (label, p, bufs, None))
                        })
                        .collect::<Option<Vec<_>>>()
                    })
                    .flatten();
                match normed {
                    Some(steps) => d.extend(steps),
                    None => {
                        d.push(("rmsnorm", &k.rms, vec![&a.x, &l.attn_norm, &a.h], None));
                        d.push(self.mv("matvec q", &l.wq, &a.h, None, &a.q32));
                        d.push(self.mv("matvec k/v", &l.wk, &a.h, None, &a.k32));
                        d.push(self.mv("matvec k/v", &l.wv, &a.h, None, &a.v32));
                    }
                }
                d.push((
                    "rope",
                    &k.rope_q,
                    vec![&a.q32, &a.cos, &a.sin, &a.q16],
                    None,
                ));
                d.push((
                    "rope",
                    &k.rope_k,
                    vec![&a.k32, &a.cos, &a.sin, &a.k16],
                    None,
                ));
                d.push((
                    "kv append",
                    &k.kv_k,
                    vec![&a.k16, &l.kcache, &self.scalars_pos],
                    None,
                ));
                d.push((
                    "kv append",
                    &k.kv_v,
                    vec![&a.v32, &l.vcache, &self.scalars_pos],
                    None,
                ));
                let nsplit = self.nsplit(self.pos + 1);
                if nsplit >= k.attention.min_splits {
                    let n_kv = self.w.cfg.n_kv;
                    d.push((
                        "attention split",
                        &k.attn_split,
                        vec![
                            &a.q16,
                            &l.kcache,
                            &l.vcache,
                            &a.part_m,
                            &a.part_l,
                            &a.part_acc,
                            &self.scalars_len,
                        ],
                        Some([n_kv, nsplit]),
                    ));
                    d.push((
                        "attention combine",
                        &k.attn_combine,
                        vec![
                            &a.part_m,
                            &a.part_l,
                            &a.part_acc,
                            &a.o,
                            &self.scalars_nsplit,
                        ],
                        None,
                    ));
                } else {
                    d.push((
                        "attention",
                        &k.attn,
                        vec![&a.q16, &l.kcache, &l.vcache, &a.o, &self.scalars_attn],
                        None,
                    ));
                }
                d.push(self.mv("matvec o", &l.wo, &a.o, Some(&a.x), &a.x2));
                // With the FFN norm folded in, gate/up read the residual
                // and the norm weight; otherwise a norm dispatch first.
                let (gx, gn) = if k.fold.1 {
                    (&a.x2, Some(&l.ffn_norm))
                } else {
                    (&a.h, None)
                };
                let fused = glu_dispatch(&k.glu, &l.gate, &l.up, gx, gn, &a.a);
                if !k.fold.1 {
                    d.push(("rmsnorm", &k.rms, vec![&a.x2, &l.ffn_norm, &a.h], None));
                }
                match fused {
                    Some((p, bufs)) => {
                        let label = if k.fold.1 {
                            "matvec gate/up + silu + rms"
                        } else {
                            "matvec gate/up + silu"
                        };
                        d.push((label, p, bufs, None));
                    }
                    None => {
                        d.push(self.mv("matvec gate/up", &l.gate, &a.h, None, &a.g));
                        d.push(self.mv("matvec gate/up", &l.up, &a.h, None, &a.u));
                        d.push(("silu_mul", &k.silu, vec![&a.g, &a.u, &a.a], None));
                    }
                }
                d.push(self.mv("matvec down", &l.down, &a.a, Some(&a.x2), &a.x));
            }
            let out = self.out.as_ref().unwrap_or(&self.emb);
            match k
                .fold
                .2
                .then(|| rms_dispatch(&k.rms_mv, out, &a.x, &self.out_norm, &a.logits))
                .flatten()
            {
                Some((p, bufs)) => d.push(("matvec lm head", p, bufs, None)),
                None => {
                    d.push(("rmsnorm", &k.rms, vec![&a.x, &self.out_norm, &a.h], None));
                    d.push(self.mv("matvec lm head", out, &a.h, None, &a.logits));
                }
            }
            d
        }
    }
}
