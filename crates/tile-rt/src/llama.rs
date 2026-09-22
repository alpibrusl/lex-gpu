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
pub use gpu::Runner;

#[cfg(target_os = "macos")]
mod gpu {
    use std::cell::RefCell;
    use std::collections::hash_map::Entry;
    use std::collections::{BTreeMap, HashMap};
    use std::time::Instant;

    use half::f16;
    use tile_front::flash::FlashDecode;
    use tile_front::llama::{QLayout, kv_append, matvec_q, rmsnorm, rope, rope_tables, silu_mul};
    use tile_front::{Program, check};
    use tile_ir::{DType, Space, Target};
    use tile_metal::{Buffer, Gpu, Pipeline};
    use tile_msl::program::lower;

    use super::{Config, QMat, Weights};

    const THREADS: usize = 256;
    /// KV positions per attention block; the cache capacity is rounded up to
    /// a whole number of them.
    const ATTN_BK: usize = 16;

    /// One dispatch of a step: a profiling label, the pipeline, its buffers.
    type Dispatch<'a> = (&'static str, &'a Pipeline, Vec<&'a Buffer>);
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
        rope_q: Pipeline,
        rope_k: Pipeline,
        silu: Pipeline,
        /// Flash decode over a runtime length: one pipeline for every step.
        attn: Pipeline,
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
        pos: usize,
        /// Dispatch one kernel at a time and record each call site's GPU
        /// time, instead of one command buffer per token. For profiling; the
        /// step itself is much slower.
        pub sync: bool,
        /// Dispatches issued so far.
        pub dispatches: usize,
        /// Wall time per call site: (calls, seconds). Every dispatch waits
        /// for completion, so this is GPU time plus submit/wait overhead.
        prof: RefCell<BTreeMap<&'static str, (usize, f64)>>,
    }

    impl<'w> Runner<'w> {
        pub fn new(w: &'w Weights) -> Result<Runner<'w>, String> {
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
            let mut mv = HashMap::new();
            for kk in keys {
                if let Entry::Vacant(slot) = mv.entry(kk) {
                    let (n_in, n_out, layout, res) = kk;
                    // One chunk per row: lazy operands need no registers,
                    // so each output reduces across its whole row once.
                    let p = matvec_q(n_in, n_out, BO, n_in, layout, res)?;
                    slot.insert(compile(&gpu, &p, THREADS)?);
                }
            }
            let cap = c.max_seq.next_multiple_of(ATTN_BK);
            let group = c.n_head / c.n_kv;
            let attn = FlashDecode {
                q_rows: group,
                d: c.head_dim,
                seq: cap,
                bq: group,
                bk: ATTN_BK,
                stages: 1,
                dtype: DType::F16,
                kv_space: Space::Threadgroup,
                consumers: 0,
                heads: c.n_kv,
                kv_cap: cap,
            };
            let k = Kernels {
                attn: compile(&gpu, &attn.build_dynamic()?, 128)?,
                kv_k: compile(&gpu, &kv_append(c.n_kv, c.head_dim, cap, DType::F16), 64)?,
                kv_v: compile(&gpu, &kv_append(c.n_kv, c.head_dim, cap, DType::F32), 64)?,
                rms: compile(&gpu, &rmsnorm(c.dim, c.eps), THREADS)?,
                mv,
                rope_q: compile(&gpu, &rope(c.n_head, c.head_dim, DType::F16), THREADS)?,
                rope_k: compile(&gpu, &rope(c.n_kv, c.head_dim, DType::F16), THREADS)?,
                silu: compile(&gpu, &silu_mul(c.ffn, THREADS)?, THREADS)?,
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
            };
            let gpu_scalars_pos = gpu.zeroed::<u32>(1);
            let gpu_scalars_attn = gpu.zeroed::<u32>(2);
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
                pos: 0,
                sync: std::env::var_os("TILE_SYNC").is_some(),
                dispatches: 0,
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
                &[len as u32, len.div_ceil(ATTN_BK) as u32],
            );

            let plan = self.plan();
            let n = plan.len();
            if self.sync {
                // GPU time per dispatch, from the command buffer's own
                // timestamps: the CPU round trip is not the kernel's cost.
                for (label, p, bufs) in &plan {
                    let t = self.gpu.run_gpu_timed(p, bufs);
                    let mut prof = self.prof.borrow_mut();
                    let e = prof.entry(label).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += t;
                }
            } else {
                let steps: Vec<(&Pipeline, &[&Buffer])> =
                    plan.iter().map(|(_, p, b)| (*p, b.as_slice())).collect();
                let t = Instant::now();
                let (encode, gpu) = self.gpu.run_all_timed(&steps);
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
            (label, p, bufs)
        }

        /// Every dispatch of one decode step, in order.
        fn plan(&self) -> Vec<Dispatch<'_>> {
            let (a, k) = (&self.acts, &self.k);
            let mut d: Vec<Dispatch<'_>> = vec![];
            for l in &self.layers {
                d.push(("rmsnorm", &k.rms, vec![&a.x, &l.attn_norm, &a.h]));
                d.push(self.mv("matvec q", &l.wq, &a.h, None, &a.q32));
                d.push(self.mv("matvec k/v", &l.wk, &a.h, None, &a.k32));
                d.push(self.mv("matvec k/v", &l.wv, &a.h, None, &a.v32));
                d.push(("rope", &k.rope_q, vec![&a.q32, &a.cos, &a.sin, &a.q16]));
                d.push(("rope", &k.rope_k, vec![&a.k32, &a.cos, &a.sin, &a.k16]));
                d.push((
                    "kv append",
                    &k.kv_k,
                    vec![&a.k16, &l.kcache, &self.scalars_pos],
                ));
                d.push((
                    "kv append",
                    &k.kv_v,
                    vec![&a.v32, &l.vcache, &self.scalars_pos],
                ));
                d.push((
                    "attention",
                    &k.attn,
                    vec![&a.q16, &l.kcache, &l.vcache, &a.o, &self.scalars_attn],
                ));
                d.push(self.mv("matvec o", &l.wo, &a.o, Some(&a.x), &a.x2));
                d.push(("rmsnorm", &k.rms, vec![&a.x2, &l.ffn_norm, &a.h]));
                d.push(self.mv("matvec gate/up", &l.gate, &a.h, None, &a.g));
                d.push(self.mv("matvec gate/up", &l.up, &a.h, None, &a.u));
                d.push(("silu_mul", &k.silu, vec![&a.g, &a.u, &a.a]));
                d.push(self.mv("matvec down", &l.down, &a.a, Some(&a.x2), &a.x));
            }
            d.push(("rmsnorm", &k.rms, vec![&a.x, &self.out_norm, &a.h]));
            let out = self.out.as_ref().unwrap_or(&self.emb);
            d.push(self.mv("matvec lm head", out, &a.h, None, &a.logits));
            d
        }
    }
}
