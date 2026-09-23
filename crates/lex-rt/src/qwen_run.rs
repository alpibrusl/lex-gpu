//! A Qwen3.5 decode step on the GPU, over tile kernels.
//!
//! Sixty-four layers, of which every fourth is grouped attention over a
//! KV cache and the rest are gated-delta layers whose whole memory is a
//! fixed `[v_heads, v_dim, k_dim]` state and a four-position convolution
//! window. The weights are NVFP4 and reach the GPU as they lie on disk;
//! [`crate::qwen::Store`] does the few folds the kernels expect.

use crate::json::Json;
use crate::qwen::Store;

/// The text model's shape, from `config.json`.
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    /// Every `interval`-th layer is attention; the rest are linear.
    pub interval: usize,
    pub ffn: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Rotated dimensions of a head (a quarter of it).
    pub rot: usize,
    pub theta: f32,
    pub eps: f32,
    pub v_heads: usize,
    pub k_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub conv_kernel: usize,
    pub vocab: usize,
    pub max_seq: usize,
}

impl Config {
    pub fn read(store: &Store, max_seq: usize) -> Result<Config, String> {
        let c = store.config()?;
        let usize_of = |k: &str| -> Result<usize, String> {
            c.get(k)
                .and_then(Json::usize)
                .ok_or_else(|| format!("config has no {k}"))
        };
        let rope = c
            .get("rope_parameters")
            .ok_or("config has no rope_parameters")?;
        let f32_of =
            |j: &Json, k: &str, d: f32| j.get(k).and_then(Json::num).map_or(d, |v| v as f32);
        let head_dim = usize_of("head_dim")?;
        let rot = (head_dim as f32 * f32_of(rope, "partial_rotary_factor", 0.25)) as usize;
        Ok(Config {
            hidden: usize_of("hidden_size")?,
            layers: usize_of("num_hidden_layers")?,
            interval: usize_of("full_attention_interval")?,
            ffn: usize_of("intermediate_size")?,
            heads: usize_of("num_attention_heads")?,
            kv_heads: usize_of("num_key_value_heads")?,
            head_dim,
            rot,
            theta: f32_of(rope, "rope_theta", 1e7),
            eps: f32_of(&c, "rms_norm_eps", 1e-6),
            v_heads: usize_of("linear_num_value_heads")?,
            k_heads: usize_of("linear_num_key_heads")?,
            k_dim: usize_of("linear_key_head_dim")?,
            v_dim: usize_of("linear_value_head_dim")?,
            conv_kernel: usize_of("linear_conv_kernel_dim")?,
            vocab: usize_of("vocab_size")?,
            max_seq,
        })
    }

    /// Attention layers are the last of each group of `interval`.
    pub fn is_linear(&self, layer: usize) -> bool {
        !(layer + 1).is_multiple_of(self.interval)
    }

    /// Value heads served by one key head.
    pub fn per_key(&self) -> usize {
        self.v_heads / self.k_heads
    }

    /// Channels the convolution covers: queries, keys, then values.
    pub fn conv_channels(&self) -> usize {
        2 * self.k_heads * self.k_dim + self.v_heads * self.v_dim
    }

    /// Where the values start in that row.
    pub fn v_base(&self) -> usize {
        2 * self.k_heads * self.k_dim
    }
}

/// `cos` and `sin` for one position, over the rotated half of a head.
pub fn rope_tables(pos: usize, rot: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rot / 2;
    let (mut cos, mut sin) = (vec![0.0; half], vec![0.0; half]);
    for i in 0..half {
        let f = (pos as f32) * theta.powf(-(i as f32) / half as f32);
        cos[i] = f.cos();
        sin[i] = f.sin();
    }
    (cos, sin)
}

#[cfg(target_os = "macos")]
pub use gpu::Runner;

#[cfg(target_os = "macos")]
mod gpu {
    use std::collections::HashMap;

    use half::f16;
    use lex_front::flash::FlashDecode;
    use lex_front::llama::{
        QLayout, kv_append, kv_append_rows, matmul_q, matvec_q, rmsnorm, rmsnorm_rows,
    };
    use lex_front::qwen::{
        DeltaNet, build_conv_silu_rows, build_delta_qk_rows, build_gated_norm_rows,
        build_gates_rows, build_matvec_dense_rows, build_mul, build_qk_rope_rows,
    };
    use lex_front::{Program, check};
    use lex_ir::{DType, Space, Target};
    use lex_metal::{Buffer, Gpu, Pipeline, Step};
    use lex_msl::program::lower;

    use super::{Config, rope_tables};
    use crate::qwen::Store;

    const THREADS: usize = 256;
    const BO: usize = 8;
    const ATTN_BK: usize = 16;
    /// State rows per instance of the delta step.
    const DELTA_ROWS: usize = 8;

    fn compile(gpu: &Gpu, prog: &Program, threads: usize) -> Result<Pipeline, String> {
        let target: &Target = gpu.target();
        check(prog, target).map_err(|errs| {
            let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
            format!("`{}` does not check:\n{}", prog.name, msgs.join("\n"))
        })?;
        gpu.build_lowered(&lower(prog, target, threads)?)
    }

    /// An NVFP4 matrix on the GPU, as the matvec binds it.
    struct QBuf {
        rows: usize,
        cols: usize,
        codes: Buffer,
        scales: Buffer,
        row_scale: Buffer,
    }

    impl QBuf {
        fn load(gpu: &Gpu, store: &Store, name: &str) -> Result<QBuf, String> {
            let w = store.nvfp4(name)?;
            Ok(QBuf {
                rows: w.rows,
                cols: w.cols,
                codes: gpu.upload(&w.codes),
                scales: gpu.upload(&w.scales),
                row_scale: gpu.upload(&w.row_scale),
            })
        }

        fn bind<'a>(
            &'a self,
            x: &'a Buffer,
            r: Option<&'a Buffer>,
            y: &'a Buffer,
        ) -> Vec<&'a Buffer> {
            let mut v = vec![x, &self.codes, &self.scales, &self.row_scale];
            v.extend(r);
            v.push(y);
            v
        }
    }

    fn floats(gpu: &Gpu, store: &Store, name: &str) -> Result<Buffer, String> {
        let (v, _) = store.floats(name)?;
        Ok(gpu.upload(&v))
    }

    /// One gated-delta layer's weights and its carried state.
    struct Linear {
        norm: Buffer,
        qkv: QBuf,
        z: QBuf,
        a: Buffer,
        b: Buffer,
        amp: Buffer,
        dt_bias: Buffer,
        conv_w: Buffer,
        gnorm: Buffer,
        out: QBuf,
        conv_state: Buffer,
        state: Buffer,
    }

    /// One attention layer's weights and its cache.
    struct Attn {
        norm: Buffer,
        q: QBuf,
        k: QBuf,
        v: QBuf,
        o: QBuf,
        q_norm: Buffer,
        k_norm: Buffer,
        kcache: Buffer,
        vcache: Buffer,
    }

    enum Mixer {
        Linear(Box<Linear>),
        Attn(Box<Attn>),
    }

    /// The feed-forward every layer ends with.
    struct Ffn {
        norm: Buffer,
        gate: QBuf,
        up: QBuf,
        down: QBuf,
    }

    struct Layer {
        mixer: Mixer,
        ffn: Ffn,
    }

    /// Every pipeline a step dispatches.
    struct Kernels {
        rms: Pipeline,
        mv: HashMap<(usize, usize, bool), Pipeline>,
        dense: Pipeline,
        conv: Pipeline,
        qk_q: Pipeline,
        qk_k: Pipeline,
        gates: Pipeline,
        delta: Pipeline,
        gated_norm: Pipeline,
        rope_q: Pipeline,
        rope_k: Pipeline,
        kv_k: Pipeline,
        kv_v: Pipeline,
        attn: Pipeline,
        mul: Pipeline,
        silu: Pipeline,
    }

    /// Largest batch a single [`Runner::forward`] takes.
    pub const MAX_BATCH: usize = 8;

    /// Every pipeline a batch of `t` tokens dispatches, compiled on first
    /// use of that size.
    struct Batch {
        rms: Pipeline,
        mv: HashMap<(usize, usize, bool), Pipeline>,
        dense: Pipeline,
        conv: Pipeline,
        qk_q: Pipeline,
        qk_k: Pipeline,
        gates: Pipeline,
        delta: Pipeline,
        gated_norm: Pipeline,
        rope_q: Pipeline,
        rope_k: Pipeline,
        kv_k: Pipeline,
        kv_v: Pipeline,
        attn: Pipeline,
        mul: Pipeline,
        silu: Pipeline,
    }

    /// Activation buffers, reused every step.
    struct Acts {
        x: Buffer,
        x2: Buffer,
        h: Buffer,
        qkv: Buffer,
        z: Buffer,
        a: Buffer,
        b: Buffer,
        conv: Buffer,
        qe: Buffer,
        ke: Buffer,
        g: Buffer,
        beta: Buffer,
        y: Buffer,
        mixed: Buffer,
        q32: Buffer,
        k32: Buffer,
        v32: Buffer,
        q16: Buffer,
        k16: Buffer,
        gate: Buffer,
        attn: Buffer,
        gated: Buffer,
        ffn_g: Buffer,
        ffn_u: Buffer,
        ffn_a: Buffer,
        cos: Buffer,
        sin: Buffer,
        logits: Buffer,
        scalars_pos: Buffer,
        scalars_attn: Buffer,
    }

    /// A Qwen3.5 model on the GPU.
    pub struct Runner {
        gpu: Gpu,
        pub cfg: Config,
        k: Kernels,
        layers: Vec<Layer>,
        embed: Vec<f32>,
        out_norm: Buffer,
        lm_head: QBuf,
        acts: Acts,
        /// Activations for a batch, and the kernels for each size seen.
        bacts: Acts,
        batches: HashMap<usize, Batch>,
        cap: usize,
        pos: usize,
        /// Time each dispatch on its own, from the command buffer's
        /// timestamps, instead of one buffer per token. The step is much
        /// slower this way; it says where the time goes.
        pub sync: bool,
        prof: std::cell::RefCell<std::collections::BTreeMap<&'static str, (usize, f64)>>,
    }

    impl Runner {
        pub fn load(model: &str, max_seq: usize) -> Result<Runner, String> {
            let store = Store::open(model)?;
            let cfg = Config::read(&store, max_seq)?;
            let gpu = Gpu::open()?;
            let cap = max_seq.next_multiple_of(ATTN_BK);
            let (hv, dv, dk) = (cfg.v_heads, cfg.v_dim, cfg.k_dim);
            let ch = cfg.conv_channels();

            // One matvec pipeline per (shape, residual) the model uses.
            let mut mv = HashMap::new();
            let mut want: Vec<(usize, usize, bool)> = vec![(cfg.hidden, cfg.vocab, false)];
            for l in 0..cfg.layers {
                want.push((cfg.hidden, cfg.ffn, false));
                want.push((cfg.ffn, cfg.hidden, true));
                if cfg.is_linear(l) {
                    want.push((cfg.hidden, ch, false));
                    want.push((cfg.hidden, hv * dv, false));
                    want.push((hv * dv, cfg.hidden, true));
                } else {
                    want.push((cfg.hidden, 2 * cfg.heads * cfg.head_dim, false));
                    want.push((cfg.hidden, cfg.kv_heads * cfg.head_dim, false));
                    want.push((cfg.heads * cfg.head_dim, cfg.hidden, true));
                }
            }
            for key in want {
                if let std::collections::hash_map::Entry::Vacant(slot) = mv.entry(key) {
                    let (n_in, n_out, res) = key;
                    let p = matvec_q(n_in, n_out, BO, n_in, QLayout::NVFP4, res)?;
                    slot.insert(compile(&gpu, &p, THREADS)?);
                }
            }

            let delta = DeltaNet {
                v_heads: hv,
                k_heads: cfg.k_heads,
                k_dim: dk,
                v_dim: dv,
                rows: DELTA_ROWS,
                v_base: cfg.v_base(),
                v_width: ch,
            };
            let attn = FlashDecode {
                q_rows: cfg.heads / cfg.kv_heads,
                d: cfg.head_dim,
                seq: cap,
                bq: cfg.heads / cfg.kv_heads,
                bk: ATTN_BK,
                stages: 1,
                dtype: DType::F16,
                kv_space: Space::Reg,
                consumers: 0,
                heads: cfg.kv_heads,
                kv_cap: cap,
            };
            let k = Kernels {
                rms: compile(&gpu, &rmsnorm(cfg.hidden, cfg.eps), THREADS)?,
                mv,
                dense: compile(
                    &gpu,
                    &build_matvec_dense_rows(1, cfg.hidden, hv, 1, DType::F32)?,
                    THREADS,
                )?,
                conv: compile(
                    &gpu,
                    &build_conv_silu_rows(1, ch, cfg.conv_kernel, 256)?,
                    THREADS,
                )?,
                qk_q: compile(
                    &gpu,
                    &build_delta_qk_rows(
                        1,
                        cfg.k_heads,
                        cfg.per_key(),
                        dk,
                        ch,
                        0,
                        1.0 / dk as f32,
                        1e-6,
                    )?,
                    128,
                )?,
                qk_k: compile(
                    &gpu,
                    &build_delta_qk_rows(
                        1,
                        cfg.k_heads,
                        cfg.per_key(),
                        dk,
                        ch,
                        cfg.k_heads * dk,
                        (dk as f32).powf(-0.5),
                        1e-6,
                    )?,
                    128,
                )?,
                gates: compile(&gpu, &build_gates_rows(1, hv, dv), 64)?,
                delta: compile(&gpu, &delta.build_step()?, 128)?,
                gated_norm: compile(&gpu, &build_gated_norm_rows(1, hv, dv, cfg.eps), 128)?,
                rope_q: compile(
                    &gpu,
                    &build_qk_rope_rows(
                        1,
                        cfg.heads,
                        cfg.head_dim,
                        cfg.rot,
                        true,
                        DType::F16,
                        cfg.eps,
                    )?,
                    THREADS,
                )?,
                rope_k: compile(
                    &gpu,
                    &build_qk_rope_rows(
                        1,
                        cfg.kv_heads,
                        cfg.head_dim,
                        cfg.rot,
                        false,
                        DType::F16,
                        cfg.eps,
                    )?,
                    THREADS,
                )?,
                kv_k: compile(
                    &gpu,
                    &kv_append(cfg.kv_heads, cfg.head_dim, cap, DType::F16),
                    64,
                )?,
                kv_v: compile(
                    &gpu,
                    &kv_append(cfg.kv_heads, cfg.head_dim, cap, DType::F32),
                    64,
                )?,
                attn: compile(&gpu, &attn.build_dynamic()?, 128)?,
                mul: compile(&gpu, &build_mul(cfg.heads * cfg.head_dim, 256)?, THREADS)?,
                silu: compile(
                    &gpu,
                    &lex_front::llama::silu_mul(cfg.ffn, THREADS)?,
                    THREADS,
                )?,
            };

            let mut layers = Vec::with_capacity(cfg.layers);
            for i in 0..cfg.layers {
                let p = format!("model.language_model.layers.{i}.");
                let ffn = Ffn {
                    norm: floats(&gpu, &store, &format!("{p}post_attention_layernorm.weight"))?,
                    gate: QBuf::load(&gpu, &store, &format!("{p}mlp.gate_proj.weight"))?,
                    up: QBuf::load(&gpu, &store, &format!("{p}mlp.up_proj.weight"))?,
                    down: QBuf::load(&gpu, &store, &format!("{p}mlp.down_proj.weight"))?,
                };
                let norm = floats(&gpu, &store, &format!("{p}input_layernorm.weight"))?;
                let mixer = if cfg.is_linear(i) {
                    let l = format!("{p}linear_attn.");
                    Mixer::Linear(Box::new(Linear {
                        norm,
                        qkv: QBuf::load(&gpu, &store, &format!("{l}in_proj_qkv.weight"))?,
                        z: QBuf::load(&gpu, &store, &format!("{l}in_proj_z.weight"))?,
                        a: floats(&gpu, &store, &format!("{l}in_proj_a.weight"))?,
                        b: floats(&gpu, &store, &format!("{l}in_proj_b.weight"))?,
                        amp: floats(&gpu, &store, &format!("{l}A_log"))?,
                        dt_bias: floats(&gpu, &store, &format!("{l}dt_bias"))?,
                        conv_w: floats(&gpu, &store, &format!("{l}conv1d.weight"))?,
                        gnorm: floats(&gpu, &store, &format!("{l}norm.weight"))?,
                        out: QBuf::load(&gpu, &store, &format!("{l}out_proj.weight"))?,
                        conv_state: gpu.zeroed::<f32>((cfg.conv_kernel - 1) * ch),
                        state: gpu.zeroed::<f32>(hv * dv * dk),
                    }))
                } else {
                    let a = format!("{p}self_attn.");
                    Mixer::Attn(Box::new(Attn {
                        norm,
                        q: QBuf::load(&gpu, &store, &format!("{a}q_proj.weight"))?,
                        k: QBuf::load(&gpu, &store, &format!("{a}k_proj.weight"))?,
                        v: QBuf::load(&gpu, &store, &format!("{a}v_proj.weight"))?,
                        o: QBuf::load(&gpu, &store, &format!("{a}o_proj.weight"))?,
                        q_norm: floats(&gpu, &store, &format!("{a}q_norm.weight"))?,
                        k_norm: floats(&gpu, &store, &format!("{a}k_norm.weight"))?,
                        kcache: gpu.zeroed::<f16>(cfg.kv_heads * cap * cfg.head_dim),
                        vcache: gpu.zeroed::<f16>(cfg.kv_heads * cap * cfg.head_dim),
                    }))
                };
                layers.push(Layer { mixer, ffn });
            }

            let (embed, _) = store.floats("model.language_model.embed_tokens.weight")?;
            let acts_for = |t: usize| {
                let f = |n: usize| gpu.zeroed::<f32>(t * n);
                Acts {
                    x: f(cfg.hidden),
                    x2: f(cfg.hidden),
                    h: f(cfg.hidden),
                    qkv: f(ch),
                    z: f(hv * dv),
                    a: f(hv),
                    b: f(hv),
                    conv: f(ch),
                    qe: f(hv * dk),
                    ke: f(hv * dk),
                    g: f(hv * dv),
                    beta: f(hv * dv),
                    y: f(hv * dv),
                    mixed: f(hv * dv),
                    q32: f(2 * cfg.heads * cfg.head_dim),
                    k32: f(cfg.kv_heads * cfg.head_dim),
                    v32: f(cfg.kv_heads * cfg.head_dim),
                    q16: gpu.zeroed::<f16>(t * cfg.heads * cfg.head_dim),
                    k16: gpu.zeroed::<f16>(t * cfg.kv_heads * cfg.head_dim),
                    gate: f(cfg.heads * cfg.head_dim),
                    attn: f(cfg.heads * cfg.head_dim),
                    gated: f(cfg.heads * cfg.head_dim),
                    ffn_g: f(cfg.ffn),
                    ffn_u: f(cfg.ffn),
                    ffn_a: f(cfg.ffn),
                    cos: f(cfg.rot / 2),
                    sin: f(cfg.rot / 2),
                    logits: f(cfg.vocab),
                    scalars_pos: gpu.zeroed::<u32>(1),
                    scalars_attn: gpu.zeroed::<u32>(2),
                }
            };
            let acts = acts_for(1);
            let bacts = acts_for(MAX_BATCH);
            let out_norm = floats(&gpu, &store, "model.language_model.norm.weight")?;
            let lm_head = QBuf::load(&gpu, &store, "lm_head.weight")?;
            Ok(Runner {
                gpu,
                cfg,
                k,
                layers,
                embed,
                out_norm,
                lm_head,
                acts,
                bacts,
                batches: HashMap::new(),
                cap,
                pos: 0,
                sync: std::env::var_os("LEX_SYNC").is_some(),
                prof: std::cell::RefCell::new(std::collections::BTreeMap::new()),
            })
        }

        pub fn device(&self) -> String {
            self.gpu.info().name
        }

        pub fn pos(&self) -> usize {
            self.pos
        }

        /// The residual stream after the last layer and *before* the final
        /// norm, as it stands after the most recent step.
        ///
        /// This is what the checkpoint's multi-token-prediction head takes:
        /// it carries its own `pre_fc_norm_hidden`, so it wants the
        /// unnormalised state. Reading it out is how the draft head is
        /// measured without a second forward pass.
        pub fn hidden(&self) -> Vec<f32> {
            let mut h = vec![0.0f32; self.cfg.hidden];
            self.gpu.download(&self.acts.x, &mut h);
            h
        }

        /// Forget the sequence. A linear layer's memory is its state and
        /// its convolution window, so both are cleared; an attention
        /// layer's cache is simply overwritten as positions are refilled.
        pub fn reset(&mut self) {
            self.pos = 0;
            let (hv, dv, dk) = (self.cfg.v_heads, self.cfg.v_dim, self.cfg.k_dim);
            let zeros = vec![0.0f32; hv * dv * dk];
            let win = vec![0.0f32; (self.cfg.conv_kernel - 1) * self.cfg.conv_channels()];
            for l in &self.layers {
                if let Mixer::Linear(lin) = &l.mixer {
                    self.gpu.write(&lin.state, 0, &zeros);
                    self.gpu.write(&lin.conv_state, 0, &win);
                }
            }
        }

        /// Feed one token at the next position; returns its logits.
        pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
            let c = self.cfg.clone();
            if self.pos >= self.cap {
                return Err(format!("the cache is full ({} positions)", self.cap));
            }
            let a = &self.acts;
            let row = token as usize * c.hidden;
            self.gpu.write(&a.x, 0, &self.embed[row..row + c.hidden]);
            let (cos, sin) = rope_tables(self.pos, c.rot, c.theta);
            self.gpu.write(&a.cos, 0, &cos);
            self.gpu.write(&a.sin, 0, &sin);
            let len = self.pos + 1;
            self.gpu.write(&a.scalars_pos, 0, &[self.pos as u32]);
            self.gpu.write(
                &a.scalars_attn,
                0,
                &[len as u32, len.div_ceil(ATTN_BK) as u32],
            );

            let plan = self.plan();
            if self.sync {
                for (label, p, b) in &plan {
                    let (_, t) = self.gpu.run_launches(&[(p, b.as_slice(), None)]);
                    let mut prof = self.prof.borrow_mut();
                    let e = prof.entry(label).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 += t;
                }
            } else {
                let steps: Vec<Step<'_>> = plan
                    .iter()
                    .map(|(_, p, b)| (*p, b.as_slice(), None))
                    .collect();
                self.gpu.run_launches(&steps);
            }
            drop(plan);
            self.pos += 1;
            let mut logits = vec![0.0f32; c.vocab];
            self.gpu.download(&a.logits, &mut logits);
            Ok(logits)
        }

        /// Feed `tokens` at the next positions in one pass over the
        /// weights. Returns the logits after every token (`all`, which a
        /// speculative verify needs) or only after the last.
        ///
        /// The weights are read once for the whole batch, which is the
        /// point: they are 95% of a step. The state a gated delta layer
        /// carries is read once too, and its recurrence runs inside the
        /// kernel, so the batch lands exactly where the same tokens would
        /// have one at a time.
        pub fn forward(&mut self, tokens: &[u32], all: bool) -> Result<Vec<Vec<f32>>, String> {
            let t = tokens.len();
            if t == 0 || t > MAX_BATCH {
                return Err(format!("a batch is 1..={MAX_BATCH} tokens, not {t}"));
            }
            if self.pos + t > self.cap {
                return Err(format!("the cache is full ({} positions)", self.cap));
            }
            self.batch(t)?;
            let c = self.cfg.clone();
            let pos0 = self.pos;
            for (i, &tok) in tokens.iter().enumerate() {
                let row = tok as usize * c.hidden;
                self.gpu.write(
                    &self.bacts.x,
                    i * c.hidden,
                    &self.embed[row..row + c.hidden],
                );
                let (cos, sin) = rope_tables(pos0 + i, c.rot, c.theta);
                self.gpu.write(&self.bacts.cos, i * c.rot / 2, &cos);
                self.gpu.write(&self.bacts.sin, i * c.rot / 2, &sin);
            }
            self.gpu.write(&self.bacts.scalars_pos, 0, &[pos0 as u32]);
            self.gpu.write(
                &self.bacts.scalars_attn,
                0,
                &[pos0 as u32, (pos0 + t).div_ceil(ATTN_BK) as u32],
            );

            let plan = self.batch_plan(t);
            let steps: Vec<Step<'_>> = plan
                .iter()
                .map(|(_, p, b)| (*p, b.as_slice(), None))
                .collect();
            self.gpu.run_launches(&steps);
            drop(plan);
            self.pos += t;

            let v = c.vocab;
            if all {
                let mut flat = vec![0.0f32; t * v];
                self.gpu.download(&self.bacts.logits, &mut flat);
                Ok(flat.chunks(v).map(<[f32]>::to_vec).collect())
            } else {
                let mut flat = vec![0.0f32; t * v];
                self.gpu.download(&self.bacts.logits, &mut flat);
                Ok(vec![flat[(t - 1) * v..].to_vec()])
            }
        }

        /// Compile the pipelines for a batch of `t`, once.
        fn batch(&mut self, t: usize) -> Result<(), String> {
            if self.batches.contains_key(&t) {
                return Ok(());
            }
            let c = self.cfg.clone();
            let gpu = &self.gpu;
            let (hv, dv, dk) = (c.v_heads, c.v_dim, c.k_dim);
            let ch = c.conv_channels();
            // Rows per threadgroup, measured on the 5120 -> 17408 shape
            // (`examples/matvec`): 16 is best from two tokens up, and a
            // single-token batch is better served by the decode path.
            let bo = if t == 1 { 32 } else { 16 };
            let mut mv = HashMap::new();
            let mut want: Vec<(usize, usize, bool)> = vec![(c.hidden, c.vocab, false)];
            for l in 0..c.layers {
                want.push((c.hidden, c.ffn, false));
                want.push((c.ffn, c.hidden, true));
                if c.is_linear(l) {
                    want.push((c.hidden, ch, false));
                    want.push((c.hidden, hv * dv, false));
                    want.push((hv * dv, c.hidden, true));
                } else {
                    want.push((c.hidden, 2 * c.heads * c.head_dim, false));
                    want.push((c.hidden, c.kv_heads * c.head_dim, false));
                    want.push((c.heads * c.head_dim, c.hidden, true));
                }
            }
            for key in want {
                if let std::collections::hash_map::Entry::Vacant(slot) = mv.entry(key) {
                    let (n_in, n_out, res) = key;
                    let p = matmul_q(t, n_in, n_out, bo, n_in, QLayout::NVFP4, res)?;
                    slot.insert(compile(gpu, &p, THREADS)?);
                }
            }
            let delta = DeltaNet {
                v_heads: hv,
                k_heads: c.k_heads,
                k_dim: dk,
                v_dim: dv,
                rows: DELTA_ROWS,
                v_base: c.v_base(),
                v_width: ch,
            };
            let attn = FlashDecode {
                q_rows: c.heads / c.kv_heads,
                d: c.head_dim,
                seq: self.cap,
                bq: c.heads / c.kv_heads,
                bk: ATTN_BK,
                stages: 1,
                dtype: DType::F16,
                kv_space: Space::Reg,
                consumers: 0,
                heads: c.kv_heads,
                kv_cap: self.cap,
            };
            let b = Batch {
                rms: compile(gpu, &rmsnorm_rows(t, c.hidden, c.eps, None), THREADS)?,
                mv,
                dense: compile(
                    gpu,
                    &build_matvec_dense_rows(t, c.hidden, hv, 1, DType::F32)?,
                    THREADS,
                )?,
                conv: compile(
                    gpu,
                    &build_conv_silu_rows(t, ch, c.conv_kernel, 256)?,
                    THREADS,
                )?,
                qk_q: compile(
                    gpu,
                    &build_delta_qk_rows(
                        t,
                        c.k_heads,
                        c.per_key(),
                        dk,
                        ch,
                        0,
                        1.0 / dk as f32,
                        1e-6,
                    )?,
                    128,
                )?,
                qk_k: compile(
                    gpu,
                    &build_delta_qk_rows(
                        t,
                        c.k_heads,
                        c.per_key(),
                        dk,
                        ch,
                        c.k_heads * dk,
                        (dk as f32).powf(-0.5),
                        1e-6,
                    )?,
                    128,
                )?,
                gates: compile(gpu, &build_gates_rows(t, hv, dv), 64)?,
                delta: compile(gpu, &delta.build_steps(t)?, 128)?,
                gated_norm: compile(gpu, &build_gated_norm_rows(t, hv, dv, c.eps), 128)?,
                rope_q: compile(
                    gpu,
                    &build_qk_rope_rows(t, c.heads, c.head_dim, c.rot, true, DType::F16, c.eps)?,
                    THREADS,
                )?,
                rope_k: compile(
                    gpu,
                    &build_qk_rope_rows(
                        t,
                        c.kv_heads,
                        c.head_dim,
                        c.rot,
                        false,
                        DType::F16,
                        c.eps,
                    )?,
                    THREADS,
                )?,
                kv_k: compile(
                    gpu,
                    &kv_append_rows(t, c.kv_heads, c.head_dim, self.cap, DType::F16),
                    64,
                )?,
                kv_v: compile(
                    gpu,
                    &kv_append_rows(t, c.kv_heads, c.head_dim, self.cap, DType::F32),
                    64,
                )?,
                attn: compile(gpu, &attn.build_causal(t)?, 128)?,
                mul: compile(gpu, &build_mul(t * c.heads * c.head_dim, 256)?, THREADS)?,
                silu: compile(
                    gpu,
                    &lex_front::llama::silu_mul(t * c.ffn, THREADS)?,
                    THREADS,
                )?,
            };
            self.batches.insert(t, b);
            Ok(())
        }

        fn mv(&self, w: &QBuf, res: bool) -> &Pipeline {
            &self.k.mv[&(w.cols, w.rows, res)]
        }

        /// Per-call-site GPU time so far, slowest first.
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

        /// Every dispatch of a batch of `t`, in order.
        fn batch_plan(&self, t: usize) -> Vec<(&'static str, &Pipeline, Vec<&Buffer>)> {
            let a = &self.bacts;
            let k = &self.batches[&t];
            let mut d: Vec<(&'static str, &Pipeline, Vec<&Buffer>)> = vec![];
            for layer in &self.layers {
                d.push((
                    "rmsnorm",
                    &k.rms,
                    vec![&a.x, layer_norm(&layer.mixer), &a.h],
                ));
                match &layer.mixer {
                    Mixer::Linear(l) => {
                        d.push((
                            "matvec qkv",
                            &k.mv[&(l.qkv.cols, l.qkv.rows, false)],
                            l.qkv.bind(&a.h, None, &a.qkv),
                        ));
                        d.push((
                            "matvec z",
                            &k.mv[&(l.z.cols, l.z.rows, false)],
                            l.z.bind(&a.h, None, &a.z),
                        ));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.a, &a.a]));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.b, &a.b]));
                        d.push((
                            "conv",
                            &k.conv,
                            vec![&l.conv_state, &a.qkv, &l.conv_w, &a.conv],
                        ));
                        d.push(("delta q/k", &k.qk_q, vec![&a.conv, &a.qe]));
                        d.push(("delta q/k", &k.qk_k, vec![&a.conv, &a.ke]));
                        d.push((
                            "gates",
                            &k.gates,
                            vec![&a.a, &a.b, &l.amp, &l.dt_bias, &a.g, &a.beta],
                        ));
                        d.push((
                            "delta step",
                            &k.delta,
                            vec![&l.state, &a.qe, &a.ke, &a.conv, &a.g, &a.beta, &a.y],
                        ));
                        d.push((
                            "gated norm",
                            &k.gated_norm,
                            vec![&a.y, &l.gnorm, &a.z, &a.mixed],
                        ));
                        d.push((
                            "matvec out_proj",
                            &k.mv[&(l.out.cols, l.out.rows, true)],
                            l.out.bind(&a.mixed, Some(&a.x), &a.x2),
                        ));
                    }
                    Mixer::Attn(at) => {
                        d.push((
                            "matvec q",
                            &k.mv[&(at.q.cols, at.q.rows, false)],
                            at.q.bind(&a.h, None, &a.q32),
                        ));
                        d.push((
                            "matvec k/v",
                            &k.mv[&(at.k.cols, at.k.rows, false)],
                            at.k.bind(&a.h, None, &a.k32),
                        ));
                        d.push((
                            "matvec k/v",
                            &k.mv[&(at.v.cols, at.v.rows, false)],
                            at.v.bind(&a.h, None, &a.v32),
                        ));
                        d.push((
                            "rope",
                            &k.rope_q,
                            vec![&a.q32, &at.q_norm, &a.cos, &a.sin, &a.q16, &a.gate],
                        ));
                        d.push((
                            "rope",
                            &k.rope_k,
                            vec![&a.k32, &at.k_norm, &a.cos, &a.sin, &a.k16],
                        ));
                        d.push((
                            "kv append",
                            &k.kv_k,
                            vec![&a.k16, &at.kcache, &a.scalars_pos],
                        ));
                        d.push((
                            "kv append",
                            &k.kv_v,
                            vec![&a.v32, &at.vcache, &a.scalars_pos],
                        ));
                        d.push((
                            "attention",
                            &k.attn,
                            vec![&a.q16, &at.kcache, &at.vcache, &a.attn, &a.scalars_attn],
                        ));
                        d.push(("gate mul", &k.mul, vec![&a.attn, &a.gate, &a.gated]));
                        d.push((
                            "matvec o_proj",
                            &k.mv[&(at.o.cols, at.o.rows, true)],
                            at.o.bind(&a.gated, Some(&a.x), &a.x2),
                        ));
                    }
                }
                let f = &layer.ffn;
                d.push(("rmsnorm", &k.rms, vec![&a.x2, &f.norm, &a.h]));
                d.push((
                    "matvec gate/up",
                    &k.mv[&(f.gate.cols, f.gate.rows, false)],
                    f.gate.bind(&a.h, None, &a.ffn_g),
                ));
                d.push((
                    "matvec gate/up",
                    &k.mv[&(f.up.cols, f.up.rows, false)],
                    f.up.bind(&a.h, None, &a.ffn_u),
                ));
                d.push(("silu_mul", &k.silu, vec![&a.ffn_g, &a.ffn_u, &a.ffn_a]));
                d.push((
                    "matvec down",
                    &k.mv[&(f.down.cols, f.down.rows, true)],
                    f.down.bind(&a.ffn_a, Some(&a.x2), &a.x),
                ));
            }
            // Every token's logits: a verify needs them all, and the head
            // is 0.7 GB against the 14.5 the batch has already moved.
            let out = &self.lm_head;
            d.push(("rmsnorm", &k.rms, vec![&a.x, &self.out_norm, &a.h]));
            d.push((
                "matvec lm head",
                &k.mv[&(out.cols, out.rows, false)],
                out.bind(&a.h, None, &a.logits),
            ));
            d
        }

        /// Every dispatch of one step, in order.
        fn plan(&self) -> Vec<(&'static str, &Pipeline, Vec<&Buffer>)> {
            let (a, k) = (&self.acts, &self.k);
            let mut d: Vec<(&'static str, &Pipeline, Vec<&Buffer>)> = vec![];
            for layer in &self.layers {
                d.push((
                    "rmsnorm",
                    &k.rms,
                    vec![&a.x, layer_norm(&layer.mixer), &a.h],
                ));
                match &layer.mixer {
                    Mixer::Linear(l) => {
                        d.push((
                            "matvec qkv",
                            self.mv(&l.qkv, false),
                            l.qkv.bind(&a.h, None, &a.qkv),
                        ));
                        d.push(("matvec z", self.mv(&l.z, false), l.z.bind(&a.h, None, &a.z)));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.a, &a.a]));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.b, &a.b]));
                        d.push((
                            "conv",
                            &k.conv,
                            vec![&l.conv_state, &a.qkv, &l.conv_w, &a.conv],
                        ));
                        d.push(("delta q/k", &k.qk_q, vec![&a.conv, &a.qe]));
                        d.push(("delta q/k", &k.qk_k, vec![&a.conv, &a.ke]));
                        d.push((
                            "gates",
                            &k.gates,
                            vec![&a.a, &a.b, &l.amp, &l.dt_bias, &a.g, &a.beta],
                        ));
                        d.push((
                            "delta step",
                            &k.delta,
                            vec![&l.state, &a.qe, &a.ke, &a.conv, &a.g, &a.beta, &a.y],
                        ));
                        d.push((
                            "gated norm",
                            &k.gated_norm,
                            vec![&a.y, &l.gnorm, &a.z, &a.mixed],
                        ));
                        d.push((
                            "matvec out_proj",
                            self.mv(&l.out, true),
                            l.out.bind(&a.mixed, Some(&a.x), &a.x2),
                        ));
                    }
                    Mixer::Attn(at) => {
                        d.push((
                            "matvec q",
                            self.mv(&at.q, false),
                            at.q.bind(&a.h, None, &a.q32),
                        ));
                        d.push((
                            "matvec k/v",
                            self.mv(&at.k, false),
                            at.k.bind(&a.h, None, &a.k32),
                        ));
                        d.push((
                            "matvec k/v",
                            self.mv(&at.v, false),
                            at.v.bind(&a.h, None, &a.v32),
                        ));
                        d.push((
                            "rope",
                            &k.rope_q,
                            vec![&a.q32, &at.q_norm, &a.cos, &a.sin, &a.q16, &a.gate],
                        ));
                        d.push((
                            "rope",
                            &k.rope_k,
                            vec![&a.k32, &at.k_norm, &a.cos, &a.sin, &a.k16],
                        ));
                        d.push((
                            "kv append",
                            &k.kv_k,
                            vec![&a.k16, &at.kcache, &a.scalars_pos],
                        ));
                        d.push((
                            "kv append",
                            &k.kv_v,
                            vec![&a.v32, &at.vcache, &a.scalars_pos],
                        ));
                        d.push((
                            "attention",
                            &k.attn,
                            vec![&a.q16, &at.kcache, &at.vcache, &a.attn, &a.scalars_attn],
                        ));
                        d.push(("gate mul", &k.mul, vec![&a.attn, &a.gate, &a.gated]));
                        d.push((
                            "matvec o_proj",
                            self.mv(&at.o, true),
                            at.o.bind(&a.gated, Some(&a.x), &a.x2),
                        ));
                    }
                }
                let f = &layer.ffn;
                d.push(("rmsnorm", &k.rms, vec![&a.x2, &f.norm, &a.h]));
                d.push((
                    "matvec gate/up",
                    self.mv(&f.gate, false),
                    f.gate.bind(&a.h, None, &a.ffn_g),
                ));
                d.push((
                    "matvec gate/up",
                    self.mv(&f.up, false),
                    f.up.bind(&a.h, None, &a.ffn_u),
                ));
                d.push(("silu_mul", &k.silu, vec![&a.ffn_g, &a.ffn_u, &a.ffn_a]));
                d.push((
                    "matvec down",
                    self.mv(&f.down, true),
                    f.down.bind(&a.ffn_a, Some(&a.x2), &a.x),
                ));
            }
            d.push(("rmsnorm", &k.rms, vec![&a.x, &self.out_norm, &a.h]));
            d.push((
                "matvec lm head",
                self.mv(&self.lm_head, false),
                self.lm_head.bind(&a.h, None, &a.logits),
            ));
            d
        }
    }

    fn layer_norm(m: &Mixer) -> &Buffer {
        match m {
            Mixer::Linear(l) => &l.norm,
            Mixer::Attn(a) => &a.norm,
        }
    }
}
