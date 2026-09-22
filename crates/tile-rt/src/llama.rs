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

use half::f16;
use tile_front::llama::split_q8_0;

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

/// A Q8_0 matrix `[rows, cols]`, split into values and per-32 scales.
pub struct Q8 {
    pub rows: usize,
    pub cols: usize,
    pub q: Vec<i8>,
    pub s: Vec<f16>,
}

impl Q8 {
    fn load(g: &Gguf, name: &str) -> Result<Q8, String> {
        let (t, b) = g.raw(name)?;
        if t.ty != GgmlType::Q8_0 || t.dims.len() != 2 {
            return Err(format!(
                "`{name}` is {:?} {:?}; only 2-d Q8_0 matrices are supported",
                t.ty, t.dims
            ));
        }
        let (q, s) = split_q8_0(b);
        Ok(Q8 {
            rows: t.dims[1],
            cols: t.dims[0],
            q,
            s,
        })
    }

    /// Row `r`, dequantised (the embedding lookup).
    pub fn row(&self, r: usize) -> Vec<f32> {
        let q = &self.q[r * self.cols..(r + 1) * self.cols];
        let s = &self.s[r * self.cols / 32..(r + 1) * self.cols / 32];
        q.iter()
            .enumerate()
            .map(|(i, &v)| v as f32 * s[i / 32].to_f32())
            .collect()
    }
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub wq: Q8,
    pub wk: Q8,
    pub wv: Q8,
    pub wo: Q8,
    pub ffn_norm: Vec<f32>,
    pub gate: Q8,
    pub up: Q8,
    pub down: Q8,
}

pub struct Weights {
    pub cfg: Config,
    pub emb: Q8,
    /// `None` when the output projection is tied to the embedding.
    pub out: Option<Q8>,
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
        let emb = Q8::load(&g, "token_embd.weight")?;
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
                wq: Q8::load(&g, &n("attn_q"))?,
                wk: Q8::load(&g, &n("attn_k"))?,
                wv: Q8::load(&g, &n("attn_v"))?,
                wo: Q8::load(&g, &n("attn_output"))?,
                ffn_norm: g.f32s(&n("ffn_norm"))?,
                gate: Q8::load(&g, &n("ffn_gate"))?,
                up: Q8::load(&g, &n("ffn_up"))?,
                down: Q8::load(&g, &n("ffn_down"))?,
            });
        }
        let out = g
            .tensors
            .contains_key("output.weight")
            .then(|| Q8::load(&g, "output.weight"))
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
    use std::collections::HashMap;

    use half::f16;
    use tile_front::flash::FlashDecode;
    use tile_front::llama::{matvec_q8, rmsnorm, rope, rope_tables, silu_mul};
    use tile_front::{Program, check};
    use tile_ir::{DType, Space, Target};
    use tile_metal::{Buffer, Gpu, Pipeline};
    use tile_msl::program::lower;

    use super::{Config, Q8, Weights};

    const THREADS: usize = 256;
    const BO: usize = 8;
    const KC: usize = 256;

    fn compile(gpu: &Gpu, prog: &Program, threads: usize) -> Result<Pipeline, String> {
        let target: &Target = gpu.target();
        check(prog, target).map_err(|errs| {
            let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
            format!("`{}` does not check:\n{}", prog.name, msgs.join("\n"))
        })?;
        gpu.build_lowered(&lower(prog, target, threads)?)
    }

    struct QBuf {
        q: Buffer,
        s: Buffer,
    }

    fn upload_q8(gpu: &Gpu, w: &Q8) -> QBuf {
        QBuf {
            q: gpu.upload(&w.q),
            s: gpu.upload(&w.s),
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
        q: Pipeline,
        kv: Pipeline,
        o_res: Pipeline,
        gate_up: Pipeline,
        down_res: Pipeline,
        lm: Pipeline,
        rope_q: Pipeline,
        rope_k: Pipeline,
        silu: Pipeline,
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
        attention: HashMap<usize, Pipeline>,
        pos: usize,
        /// Dispatches issued so far.
        pub dispatches: usize,
    }

    impl<'w> Runner<'w> {
        pub fn new(w: &'w Weights) -> Result<Runner<'w>, String> {
            let gpu = Gpu::open()?;
            let c: &Config = &w.cfg;
            let kvd = c.n_kv * c.head_dim;
            let k = Kernels {
                rms: compile(&gpu, &rmsnorm(c.dim, c.eps), THREADS)?,
                q: compile(&gpu, &matvec_q8(c.dim, c.dim, BO, KC, false)?, THREADS)?,
                kv: compile(&gpu, &matvec_q8(c.dim, kvd, BO, KC, false)?, THREADS)?,
                o_res: compile(&gpu, &matvec_q8(c.dim, c.dim, BO, KC, true)?, THREADS)?,
                gate_up: compile(&gpu, &matvec_q8(c.dim, c.ffn, BO, KC, false)?, THREADS)?,
                down_res: compile(&gpu, &matvec_q8(c.ffn, c.dim, BO, KC, true)?, THREADS)?,
                lm: compile(&gpu, &matvec_q8(c.dim, c.vocab, BO, KC, false)?, THREADS)?,
                rope_q: compile(&gpu, &rope(c.n_head, c.head_dim, DType::F16), THREADS)?,
                rope_k: compile(&gpu, &rope(c.n_kv, c.head_dim, DType::F16), THREADS)?,
                silu: compile(&gpu, &silu_mul(c.ffn, THREADS)?, THREADS)?,
            };
            let layers = w
                .layers
                .iter()
                .map(|l| LayerBufs {
                    attn_norm: gpu.upload(&l.attn_norm),
                    wq: upload_q8(&gpu, &l.wq),
                    wk: upload_q8(&gpu, &l.wk),
                    wv: upload_q8(&gpu, &l.wv),
                    wo: upload_q8(&gpu, &l.wo),
                    ffn_norm: gpu.upload(&l.ffn_norm),
                    gate: upload_q8(&gpu, &l.gate),
                    up: upload_q8(&gpu, &l.up),
                    down: upload_q8(&gpu, &l.down),
                    kcache: gpu.zeroed::<f16>(c.n_kv * c.max_seq * c.head_dim),
                    vcache: gpu.zeroed::<f16>(c.n_kv * c.max_seq * c.head_dim),
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
            Ok(Runner {
                emb: upload_q8(&gpu, &w.emb),
                out: w.out.as_ref().map(|o| upload_q8(&gpu, o)),
                out_norm: gpu.upload(&w.out_norm),
                gpu,
                w,
                k,
                layers,
                acts,
                attention: HashMap::new(),
                pos: 0,
                dispatches: 0,
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

        /// Attention over positions `0..len`, compiled once per length. The
        /// program is flash decode with the cache's capacity as its per-head
        /// stride; `bk` is the largest block size that divides `len`.
        fn attention(&mut self, len: usize) -> Result<(), String> {
            if self.attention.contains_key(&len) {
                return Ok(());
            }
            let c = &self.w.cfg;
            let bk = [16, 8, 4, 2, 1]
                .into_iter()
                .find(|b| len.is_multiple_of(*b))
                .unwrap();
            let group = c.n_head / c.n_kv;
            let cfg = FlashDecode {
                q_rows: group,
                d: c.head_dim,
                seq: len,
                bq: group,
                bk,
                stages: 1,
                dtype: DType::F16,
                kv_space: Space::Threadgroup,
                consumers: 0,
                heads: c.n_kv,
                kv_cap: c.max_seq,
            };
            let p = compile(&self.gpu, &cfg.build()?, 128)?;
            self.attention.insert(len, p);
            Ok(())
        }

        /// Feed one token at the next position; returns the logits.
        pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
            let c = self.w.cfg.clone();
            if self.pos >= c.max_seq {
                return Err(format!("KV cache is full ({} positions)", c.max_seq));
            }
            let pos = self.pos;
            let hd = c.head_dim;
            self.attention(pos + 1)?;

            // Host glue: the embedding row, and this position's RoPE tables.
            let x = self.w.emb.row(token as usize);
            self.gpu.write(&self.acts.x, 0, &x);
            let (cos, sin) = rope_tables(pos, hd, c.rope_base, c.rope_factors.as_deref());
            self.gpu.write(&self.acts.cos, 0, &cos);
            self.gpu.write(&self.acts.sin, 0, &sin);

            self.dispatches += self.layers_and_head(pos, &c);
            self.pos += 1;
            let mut logits = vec![0.0f32; c.vocab];
            self.gpu.download(&self.acts.logits, &mut logits);
            Ok(logits)
        }

        /// One step through every layer and the output head. Returns the
        /// number of dispatches.
        fn layers_and_head(&self, pos: usize, c: &Config) -> usize {
            let hd = c.head_dim;
            let kvd = c.n_kv * hd;
            let (gpu, a, k) = (&self.gpu, &self.acts, &self.k);
            let attn = &self.attention[&(pos + 1)];
            for l in &self.layers {
                gpu.run(&k.rms, &[&a.x, &l.attn_norm, &a.h]);
                gpu.run(&k.q, &[&a.h, &l.wq.q, &l.wq.s, &a.q32]);
                gpu.run(&k.kv, &[&a.h, &l.wk.q, &l.wk.s, &a.k32]);
                gpu.run(&k.kv, &[&a.h, &l.wv.q, &l.wv.s, &a.v32]);
                gpu.run(&k.rope_q, &[&a.q32, &a.cos, &a.sin, &a.q16]);
                gpu.run(&k.rope_k, &[&a.k32, &a.cos, &a.sin, &a.k16]);

                // Host glue: append this position's K and V to the cache.
                let mut k16 = vec![f16::ZERO; kvd];
                gpu.download(&a.k16, &mut k16);
                let mut v32 = vec![0.0f32; kvd];
                gpu.download(&a.v32, &mut v32);
                let v16: Vec<f16> = v32.iter().map(|&v| f16::from_f32(v)).collect();
                for h in 0..c.n_kv {
                    let at = (h * c.max_seq + pos) * hd;
                    gpu.write(&l.kcache, at, &k16[h * hd..(h + 1) * hd]);
                    gpu.write(&l.vcache, at, &v16[h * hd..(h + 1) * hd]);
                }

                gpu.run(attn, &[&a.q16, &l.kcache, &l.vcache, &a.o]);
                gpu.run(&k.o_res, &[&a.o, &l.wo.q, &l.wo.s, &a.x, &a.x2]);
                gpu.run(&k.rms, &[&a.x2, &l.ffn_norm, &a.h]);
                gpu.run(&k.gate_up, &[&a.h, &l.gate.q, &l.gate.s, &a.g]);
                gpu.run(&k.gate_up, &[&a.h, &l.up.q, &l.up.s, &a.u]);
                gpu.run(&k.silu, &[&a.g, &a.u, &a.a]);
                gpu.run(&k.down_res, &[&a.a, &l.down.q, &l.down.s, &a.x2, &a.x]);
            }
            gpu.run(&k.rms, &[&a.x, &self.out_norm, &a.h]);
            let out = self.out.as_ref().unwrap_or(&self.emb);
            gpu.run(&k.lm, &[&a.h, &out.q, &out.s, &a.logits]);
            self.layers.len() * 13 + 2
        }
    }
}
