//! A Qwen3.8 decode step on the GPU, over tile kernels.
//!
//! Sixty-four layers, of which every fourth is grouped attention over a
//! KV cache and the rest are gated-delta layers whose whole memory is a
//! fixed `[v_heads, v_dim, k_dim]` state and a four-position convolution
//! window. The weights are NVFP4 and reach the GPU as they lie on disk;
//! [`crate::qwen::Store`] does the few folds the kernels expect.

use crate::gguf::{Gguf, Value};
use crate::json::Json;
use crate::qwen_source::Source;

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
    /// Value heads in ggml's tiled order rather than Hugging Face's
    /// grouped one; see `build_delta_qk_rows_in`.
    pub v_tiled: bool,
}

impl Config {
    pub fn read(src: &Source, max_seq: usize) -> Result<Config, String> {
        if let Some(g) = src.gguf() {
            return Config::from_gguf(g, max_seq);
        }
        let store = src.mlx().ok_or("no config.json: not an MLX checkpoint")?;
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
            v_tiled: false,
        })
    }

    /// The same, from a GGUF's metadata, which carries every field under
    /// the architecture's own prefix. The SSM names are llama.cpp's:
    /// `time_step_rank` is the number of value heads, `group_count` the
    /// number of key heads, `state_size` a key head's width, and
    /// `inner_size` all the value heads together.
    fn from_gguf(g: &Gguf, max_seq: usize) -> Result<Config, String> {
        let arch = match g.meta.get("general.architecture") {
            Some(Value::Str(s)) => s.clone(),
            _ => return Err("GGUF has no general.architecture".into()),
        };
        if arch != "qwen35" {
            return Err(format!("{arch} is not the Qwen3.5 architecture this runs"));
        }
        let u = |k: &str| g.int(&format!("{arch}.{k}")).map(|v| v as usize);
        let f = |k: &str| g.float(&format!("{arch}.{k}")).map(|v| v as f32);
        let v_heads = u("ssm.time_step_rank")?;
        let cfg = Config {
            hidden: u("embedding_length")?,
            layers: u("block_count")?,
            interval: u("full_attention_interval")?,
            ffn: u("feed_forward_length")?,
            heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rot: u("rope.dimension_count")?,
            theta: f("rope.freq_base")?,
            eps: f("attention.layer_norm_rms_epsilon")?,
            v_heads,
            k_heads: u("ssm.group_count")?,
            k_dim: u("ssm.state_size")?,
            v_dim: u("ssm.inner_size")? / v_heads,
            conv_kernel: u("ssm.conv_kernel")?,
            vocab: g.info("output.weight")?.dims[1],
            max_seq,
            // Value-head order. Hugging Face groups value heads by key
            // head; ggml's broadcast tiles them, and llama.cpp's converter
            // reorders the heads to suit it. MiMo-v2.6 is tiled: read that
            // way it matches Ollama token for token, read grouped it puts
            // ' Paris' outside its top five for "The capital of France is".
            // Prism's converter keeps the grouped order and says so --
            // Bonsai 2 carries `prism.hadamard.gdn_v_grouped = true` -- so a
            // file that says grouped is read grouped, and one that says
            // nothing is read as llama.cpp writes it.
            v_tiled: !matches!(
                g.meta.get("prism.hadamard.gdn_v_grouped"),
                Some(Value::Bool(true))
            ),
        };
        // Which layers are attention is worked out from the interval. The
        // file also lists it outright, and two sources for one fact had
        // better agree: a layer run as the wrong kind reads the wrong
        // weights and says nothing about it.
        if let Some(Value::Array(flags)) = g.meta.get(&format!("{arch}.attention.recurrent_layers"))
        {
            for (i, flag) in flags.iter().enumerate() {
                if let Value::Bool(recurrent) = flag
                    && *recurrent != cfg.is_linear(i)
                {
                    return Err(format!(
                        "layer {i}: the file says recurrent={recurrent}, the interval {} says {}",
                        cfg.interval,
                        cfg.is_linear(i)
                    ));
                }
            }
        }
        Ok(cfg)
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

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub use gpu::{Checkpoint, MAX_BATCH, Runner, evict_index};

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod gpu {
    use crate::sample::Sampler;
    use std::collections::HashMap;

    use crate::dev::{Buffer, Gpu, Pipeline, Step};
    use half::f16;
    use lex_front::flash::{COMBINE_CHUNK, FlashDecode};
    use lex_front::llama::{
        QLayout, kv_append, kv_append_rows, matmul_q_x, matvec_q, rmsnorm_rows,
    };
    use lex_front::qwen::{
        DeltaNet, build_conv_silu_rows, build_conv_silu_rows_snap, build_delta_qk_rows_in,
        build_gated_norm_rows, build_gates_rows, build_matvec_dense_rows, build_mul,
        build_qk_rope_rows,
    };
    use lex_front::{Program, check};
    use lex_ir::{DType, Kernel, Space, Target, plan};
    use lex_msl::gemm::{Gemm, gemm_nvfp4};
    use lex_msl::program::lower_with;

    use super::{Config, rope_tables};
    use crate::qwen_source::Source;

    const THREADS: usize = 256;
    /// Output rows a decode matvec gives one threadgroup.
    ///
    /// Derived from the target rather than fixed, because the right
    /// answer is a property of the machine: Apple wants one row per
    /// simdgroup and Ada wants two, and the constant that used to be here
    /// was Apple's. On an L4 that difference is 107 GB/s against 189.
    fn bo(target: &Target) -> usize {
        target.matvec_rows_per_simd * (THREADS / target.simd_width)
    }
    const ATTN_BK: usize = 16;
    /// Cache blocks per split, and the fewest splits worth splitting for.
    ///
    /// Without this the decode attention scans the cache in one
    /// threadgroup per KV head, and a step at 1440 positions costs 52.16 ms
    /// against 35.55 with the attention left out -- 16.6 ms in one kernel,
    /// against 0.58 ms at zero context. Everything else on this model is
    /// flat with context, because 48 of its 64 layers carry a fixed-size
    /// state. The Llama path has had this since P3; the Qwen path never
    /// got it.
    /// One dispatch: a label for ablation, the pipeline, its buffers, and
    /// a launch shape when the kernel wants one other than its own.
    type Dispatch<'a> = (
        &'static str,
        &'a Pipeline,
        Vec<&'a Buffer>,
        Option<[usize; 2]>,
    );

    /// Batch sizes a speculative verify keeps per-token snapshots for.
    ///
    /// The snapshots are what make a rejection a copy instead of a replay,
    /// and they cost `SPEC_MAX` copies of the recurrent state -- 3.1 MB a
    /// layer, 48 layers, so 604 MB at four. Prefill runs at MAX_BATCH and
    /// never rolls back, so it does not pay this.
    const SPEC_MAX: usize = 4;

    /// The decode RMSNorm, parsed from `lex-front/lx/rmsnorm.lx`.
    ///
    /// Embedded at build time rather than read from disk: a runtime that
    /// needed a source file beside it to start would be a worse runtime,
    /// and the file is 600 bytes.
    ///
    /// The Rust `rmsnorm` it replaces is still there and still tested
    /// against this one for byte-identical MSL
    /// (`lex-front/tests/syntax.rs`). Two ways to build the same kernel
    /// is one more than necessary, and the second one goes when the
    /// surface can express the rest of them.
    fn surface_rmsnorm(n: usize, eps: f32) -> Result<Program, String> {
        const SRC: &str = include_str!("../../lex-front/lx/rmsnorm.lx");
        lex_front::syntax::parse(SRC)?
            .algo
            .build(&[("n", n as f64), ("eps", eps as f64)])
    }

    const ATTN_BPS: usize = 2;
    /// Splits below which the serial kernel is used instead.
    ///
    /// Read fresh every time on purpose: the golden test builds one Runner,
    /// decodes with splits, then sets `LEX_MIN_SPLITS` and decodes again to
    /// compare against the serial kernel. Caching this in a `OnceLock` would
    /// pin the first value and leave that test comparing the split path
    /// against itself -- passing while checking nothing. It costs one lookup
    /// per attention layer per step, about 0.04% of a step, which is below
    /// the noise in every measurement here.
    fn attn_min_splits() -> usize {
        std::env::var("LEX_MIN_SPLITS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2)
    }
    /// State rows per instance of the delta step.
    const DELTA_ROWS: usize = 8;
    /// The dtype the *batched* path keeps normalised activations in.
    ///
    /// A batched matmul re-reads every token's activations once per
    /// threadgroup, and at `bo = 32` on the feed-forward shape that is 544
    /// threadgroups: at eight tokens, 89 MB of activations against 50 MB of
    /// weights. Halving them is worth 2.1x at eight tokens and nothing at
    /// one, which is why the single-token path stays f32.
    ///
    /// The reduction inside the norm is still f32; only what it stores, and
    /// what the matmuls then re-read, is narrowed.
    const X_DTYPE: DType = DType::F16;

    fn compile(gpu: &Gpu, prog: &Program, threads: usize) -> Result<Pipeline, String> {
        let target: &Target = gpu.target();
        check(prog, target).map_err(|errs| {
            let msgs: Vec<String> = errs.iter().map(|e| e.to_string()).collect();
            format!("`{}` does not check:\n{}", prog.name, msgs.join("\n"))
        })?;
        crate::dev::dump_cuda(prog, threads);
        gpu.build_lowered(&lower_with(prog, target, threads, crate::dev::dialect())?)
    }

    /// An NVFP4 matrix on the GPU, as the matvec binds it.
    /// A pipeline key: shape, whether the matvec accumulates into the
    /// residual, and the layout its weights are stored in.
    type MvKey = (usize, usize, bool, QLayout);

    /// A quantised matrix on the device, in any layout `matvec_q` reads:
    /// the values, the six-bit layouts' high plane, then the scale
    /// parameters in `QLayout::scale_params` order with NVFP4's per-row
    /// scale last. NVFP4 binds exactly as it did when this held three
    /// named buffers -- values, scales, row scale.
    struct QBuf {
        rows: usize,
        cols: usize,
        layout: QLayout,
        q: Buffer,
        qh: Option<Buffer>,
        scales: Vec<Buffer>,
    }

    impl QBuf {
        fn load(gpu: &Gpu, src: &Source, name: &str) -> Result<QBuf, String> {
            let m = src.matrix(name)?;
            Ok(QBuf {
                rows: m.rows,
                cols: m.cols,
                layout: m.layout,
                q: gpu.upload(&m.q),
                qh: (!m.qh.is_empty()).then(|| gpu.upload(&m.qh)),
                scales: m.scales.iter().map(|b| gpu.upload(b)).collect(),
            })
        }

        fn key(&self, res: bool) -> MvKey {
            (self.cols, self.rows, res, self.layout)
        }

        fn bind<'a>(
            &'a self,
            x: &'a Buffer,
            r: Option<&'a Buffer>,
            y: &'a Buffer,
        ) -> Vec<&'a Buffer> {
            let mut v = vec![x, &self.q];
            v.extend(self.qh.as_ref());
            v.extend(self.scales.iter());
            v.extend(r);
            v.push(y);
            v
        }
    }

    /// Every matvec the model runs, keyed by what it needs compiled.
    ///
    /// Taken from the loaded weights, not from the config's shapes: a GGUF
    /// mixes quantisations within one kind of tensor -- MiMo's `ffn_down`
    /// is Q6_K in some layers and Q4_K in others -- and a pipeline compiled
    /// for the wrong one reads the right bytes as the wrong numbers.
    fn mv_keys(layers: &[Layer], lm_head: &QBuf, mtp: Option<&Mtp>) -> Vec<MvKey> {
        fn layer(l: &Layer, v: &mut Vec<MvKey>) {
            v.push(l.ffn.gate.key(false));
            v.push(l.ffn.up.key(false));
            v.push(l.ffn.down.key(true));
            match &l.mixer {
                Mixer::Linear(x) => {
                    v.push(x.qkv.key(false));
                    v.push(x.z.key(false));
                    v.push(x.out.key(true));
                }
                Mixer::Attn(a) => {
                    v.push(a.q.key(false));
                    v.push(a.k.key(false));
                    v.push(a.v.key(false));
                    v.push(a.o.key(true));
                }
            }
        }
        let mut v = vec![lm_head.key(false)];
        for l in layers {
            layer(l, &mut v);
        }
        if let Some(m) = mtp {
            v.push(m.fc.key(false));
            layer(&m.layer, &mut v);
        }
        v
    }

    /// RMSNorm on the host, for the draft head's two pre-norms: 5120
    /// values each, against the 239 MB the head then reads.
    fn rms_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
        let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (mean + eps).sqrt();
        for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
            *o = v * inv * g;
        }
    }

    fn floats(gpu: &Gpu, store: &Source, name: &str) -> Result<Buffer, String> {
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

    /// A copy of every gated-delta layer's memory, so a rejected draft can
    /// be undone.
    ///
    /// Attention layers need nothing kept: their cache is overwritten as
    /// positions are refilled, so rolling one back is just moving `pos`. A
    /// gated-delta layer has no such luxury -- its whole memory is a state
    /// updated in place, and a verify that feeds four tokens updates it
    /// four times. 48 layers x 3.1 MB, copied on the GPU at roughly 0.6 ms
    /// a round, under 2% of a pass.
    /// Which checkpoint to drop when a pool is full, by position.
    ///
    /// Dropping the oldest is the obvious rule and the wrong one. Turn
    /// boundaries bunch up wherever turns are short, so a pool of the six
    /// newest can sit entirely inside the last few thousand tokens -- and
    /// then a prompt that diverges earlier than all of them resumes from
    /// nothing. Measured over a four-task lex-code run: ten of forty-two
    /// requests re-read their whole prompt with a usable prefix sitting
    /// right there, 105326 tokens thrown away.
    ///
    /// So drop the most redundant one instead: the interior point whose
    /// removal widens the smallest gap. That thins the crowded end first
    /// and keeps the pool spread across the conversation. The newest is
    /// never dropped -- it is the one the next turn most likely wants --
    /// and neither is the oldest, which is the only fallback for a prompt
    /// that diverges early.
    pub fn evict_index(positions: &[usize]) -> Option<usize> {
        (1..positions.len().checked_sub(1)?).min_by_key(|&i| positions[i + 1] - positions[i - 1])
    }

    /// A resumable point in a conversation: the position, and the whole
    /// recurrent state there. See [`Runner::checkpoint`].
    pub struct Checkpoint {
        pos: usize,
        mtp_pos: usize,
        state: Vec<Vec<f32>>,
        conv: Vec<Vec<f32>>,
    }

    impl Checkpoint {
        pub fn pos(&self) -> usize {
            self.pos
        }

        /// Host bytes held, so a pool of these can bound itself.
        pub fn bytes(&self) -> usize {
            let f = size_of::<f32>();
            f * (self.state.iter().map(Vec::len).sum::<usize>()
                + self.conv.iter().map(Vec::len).sum::<usize>())
        }
    }

    struct Snapshot {
        state: Vec<Buffer>,
        conv: Vec<Buffer>,
        copy_state: Pipeline,
        copy_conv: Pipeline,
        pos: usize,
        /// Per-token state and window, written by a snapshotting verify:
        /// `SPEC_MAX` blocks each, of which a batch of `t` fills the first
        /// `t`. Rolling back to the last token the model agreed with is
        /// then one copy out of these, not a replay of the whole pass.
        rows_state: Vec<Buffer>,
        rows_conv: Vec<Buffer>,
        /// `copy_block` per rollback target. The block is baked into the
        /// pipeline, so picking the wrong one is a visible wrong dispatch
        /// rather than a wrong number in a buffer.
        roll_state: Vec<Pipeline>,
        roll_conv: Vec<Pipeline>,
    }

    /// The checkpoint's multi-token-prediction head.
    ///
    /// One ordinary attention layer over a cache of its own, fed by a
    /// projection that fuses the model's hidden state at `t` with the
    /// embedding of the token it just produced. Its output goes through
    /// the *shared* `lm_head`, so a draft costs the head plus a second
    /// pass over `lm_head`: 239 + 715 MB, 6.6% of a model pass.
    ///
    /// Both of its conventions were settled by measurement rather than by
    /// documentation, because `mlx_lm` drops every `mtp.` weight before it
    /// normalises anything (see [`crate::qwen::SHIFTED`]): every norm here
    /// is a delta from 1, and the **embedding leads** the concatenation.
    /// Reversed, the head drafts correctly 0.4% of the time instead of 97%.
    struct Mtp {
        pre_h: Vec<f32>,
        pre_e: Vec<f32>,
        fc: QBuf,
        layer: Layer,
        norm: Buffer,
    }

    /// The int8 matvec path (`lex_msl::int8`): its kernels by batch size and
    /// shape, and the quantised input they share.
    ///
    /// On an L4 the float matvec is clock-bound under the 72 W cap -- its
    /// cost is instructions per weight byte -- and this spends about a fifth
    /// as many. It changes the arithmetic (8-bit activations, a scale per 16
    /// values), so it is CUDA-only, NVFP4-only, opt-in with `LEX_INT8=1`
    /// (8-bit activations fail the golden tolerance; see where it is set), and
    /// covers decode and verify batches (up to `SPEC_MAX` tokens): prefill
    /// chunks go through the GEMM, and every shape here is an NVRTC compile
    /// at load.
    struct Int8 {
        /// `(tokens, n_out, n_in, residual)`.
        mm: HashMap<(usize, usize, usize, bool), Pipeline>,
        /// `(tokens, n_in, x is f16)`.
        quant: HashMap<(usize, usize, bool), Pipeline>,
        xq: Buffer,
        xs: Buffer,
    }

    /// Whether the batched matmul for this shape reads f16 activations.
    ///
    /// `res` marks the matmuls that accumulate into the residual -- down and
    /// o_proj -- which read `ffn_a` and `gated`, not `h`. Every batched
    /// matmul reads a narrowed buffer except o_proj, which reads `gated`,
    /// and the draft head's `fc`, which reads a fused input built on the
    /// host: the decode path's `fc` reads it in f32, the two write the same
    /// cache, and handed f16 it reads the f32 bytes as half pairs.
    fn batch_x_half(c: &Config, n_in: usize, res: bool) -> bool {
        !(n_in == 2 * c.hidden || (res && n_in == c.heads * c.head_dim)) && X_DTYPE == DType::F16
    }

    /// Every pipeline a step dispatches.
    struct Kernels {
        rms: Pipeline,
        mv: HashMap<MvKey, Pipeline>,
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
        /// Split-KV: many threadgroups over the cache, then a reduce.
        attn_split: Pipeline,
        attn_combine: Pipeline,
        mul: Pipeline,
        silu: Pipeline,
    }

    /// Largest batch the batched matvec serves: a verify, or a prefill
    /// whose weights are not all NVFP4. Its accumulators live in registers,
    /// one per (row, token), and past about 16 tokens they spill.
    pub const MAX_BATCH: usize = 8;

    /// Largest batch a single [`Runner::forward`] takes. Past `MAX_BATCH`
    /// the matmuls go through the matrix-unit GEMM (`lex_msl::gemm`), which
    /// keeps its accumulators in fragments and so does not spill -- and
    /// every token more in a chunk is one fewer read of the 14.5 GB of
    /// weights over a prompt.
    pub const PREFILL_MAX: usize = 128;

    /// The GEMM chunk sizes. A prompt is cut into these, largest first, and
    /// its last few tokens into one batched-matvec chunk of at most
    /// `MAX_BATCH`: a fixed set of sizes, so the kernels for each are
    /// compiled once at load -- on CUDA each size is seconds of NVRTC, and a
    /// set that grew with every new prompt length would pay them per prompt.
    const GEMM_SIZES: [usize; 4] = [128, 64, 32, 16];

    /// Every pipeline a batch of `t` tokens dispatches, compiled on first
    /// use of that size.
    struct Batch {
        rms: Pipeline,
        mv: HashMap<MvKey, Pipeline>,
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
        attn_split: Pipeline,
        attn_combine: Pipeline,
        mul: Pipeline,
        silu: Pipeline,
        /// The same delta and conv kernels, writing per-token snapshots.
        /// Only for batches a speculative verify can use.
        delta_snap: Option<Pipeline>,
        conv_snap: Option<Pipeline>,
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
        scalars_len: Buffer,
        scalars_nsplit: Buffer,
        /// Per-split partial softmax: max, denominator, accumulator.
        part_m: Buffer,
        part_l: Buffer,
        part_acc: Buffer,
    }

    /// A Qwen3.8 model on the GPU.
    pub struct Runner {
        gpu: Gpu,
        pub cfg: Config,
        k: Kernels,
        layers: Vec<Layer>,
        embed: Vec<f32>,
        out_norm: Buffer,
        lm_head: QBuf,
        /// The draft head, when the checkpoint ships one.
        mtp: Option<Mtp>,
        /// `fc`'s input: the two normalised halves, embedding first.
        mtp_in: Buffer,
        /// The same, for `MAX_BATCH` rows at once.
        mtp_in_b: Buffer,
        /// The draft head's own cache length.
        mtp_pos: usize,
        snap: Option<Snapshot>,
        /// The hidden state a verify left, for the next draft.
        spec_h: Option<Vec<f32>>,
        /// Call sites to leave out, by label prefix.
        ///
        /// For ablation: run without a kernel and the difference is what it
        /// costs *on the critical path*, in a normally-scheduled pass. That
        /// is not what `LEX_SYNC` measures -- serialised, the per-kernel
        /// times sum to 210% of the real elapsed time, because the whole
        /// point of the concurrent encoder is that they overlap. The
        /// answers are wrong in different directions and only this one is
        /// a share of anything.
        ///
        /// The results are nonsense once a kernel is missing. The shapes,
        /// the dispatch count and the scheduling are not.
        pub skip: Vec<String>,
        acts: Acts,
        /// Activations for a batch, and the kernels for each size seen.
        bacts: Acts,
        batches: HashMap<usize, Batch>,
        cap: usize,
        pos: usize,
        /// Time each dispatch (`LEX_SYNC`): on Metal each in its own
        /// command buffer, from its timestamps, which serialises what the
        /// concurrent encoder would overlap; on CUDA from events between
        /// launches queued as normal. Either way it says where the time
        /// goes, and on Metal the step is much slower for it.
        pub sync: bool,
        /// Every matrix is NVFP4, which is what the prefill GEMM reads.
        /// Otherwise prefill stays at `MAX_BATCH`.
        gemm_ok: bool,
        int8: Option<Int8>,
        prof: std::cell::RefCell<std::collections::BTreeMap<&'static str, (usize, f64)>>,
    }

    impl Runner {
        pub fn load(model: &str, max_seq: usize) -> Result<Runner, String> {
            let store = Source::open(model)?;
            let cfg = Config::read(&store, max_seq)?;
            let gpu = Gpu::open()?;
            // Whole splits, and whole chunks of splits for the combine
            // kernel: the split pair indexes them directly and refuses a
            // capacity that does not divide.
            let cap = max_seq.next_multiple_of(ATTN_BK * ATTN_BPS * COMBINE_CHUNK);
            let (hv, dv, dk) = (cfg.v_heads, cfg.v_dim, cfg.k_dim);
            let ch = cfg.conv_channels();

            // Matvec pipelines are compiled once the weights are loaded,
            // from the layouts those weights turn out to have.
            let mv = HashMap::new();

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
            let mut k = Kernels {
                // From `.lx`, not from Rust. The decode path's RMSNorm is
                // the first kernel in the model to come out of the surface
                // language, and `qwen_golden` is what says it is the same
                // kernel -- the model's tokens are checked against an f32
                // reference, so a parser that built something subtly
                // different would show up as wrong output rather than as
                // a passing parse.
                rms: compile(&gpu, &surface_rmsnorm(cfg.hidden, cfg.eps)?, THREADS)?,
                mv,
                dense: compile(
                    &gpu,
                    &build_matvec_dense_rows(1, cfg.hidden, hv, 1, DType::F32, DType::F32)?,
                    THREADS,
                )?,
                conv: compile(
                    &gpu,
                    &build_conv_silu_rows(1, ch, cfg.conv_kernel, 256)?,
                    THREADS,
                )?,
                qk_q: compile(
                    &gpu,
                    &build_delta_qk_rows_in(
                        1,
                        cfg.k_heads,
                        cfg.per_key(),
                        dk,
                        ch,
                        0,
                        1.0 / dk as f32,
                        1e-6,
                        cfg.v_tiled,
                    )?,
                    128,
                )?,
                qk_k: compile(
                    &gpu,
                    &build_delta_qk_rows_in(
                        1,
                        cfg.k_heads,
                        cfg.per_key(),
                        dk,
                        ch,
                        cfg.k_heads * dk,
                        (dk as f32).powf(-0.5),
                        1e-6,
                        cfg.v_tiled,
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
                attn_split: compile(&gpu, &attn.build_split(ATTN_BPS)?, 128)?,
                attn_combine: compile(&gpu, &attn.build_combine(ATTN_BPS)?, 128)?,
                mul: compile(&gpu, &build_mul(cfg.heads * cfg.head_dim, 256)?, THREADS)?,
                silu: compile(
                    &gpu,
                    &lex_front::llama::silu_mul(cfg.ffn, THREADS, DType::F32)?,
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

            // The draft head, when the checkpoint ships one. Its attention
            // keeps a cache the same length as the model's, because it sees
            // the same sequence.
            let mtp = if store.has("mtp.fc.weight") {
                let a = "mtp.layers.0.self_attn.";
                Some(Mtp {
                    pre_h: store.floats("mtp.pre_fc_norm_hidden.weight")?.0,
                    pre_e: store.floats("mtp.pre_fc_norm_embedding.weight")?.0,
                    fc: QBuf::load(&gpu, &store, "mtp.fc.weight")?,
                    layer: Layer {
                        mixer: Mixer::Attn(Box::new(Attn {
                            norm: floats(&gpu, &store, "mtp.layers.0.input_layernorm.weight")?,
                            q: QBuf::load(&gpu, &store, &format!("{a}q_proj.weight"))?,
                            k: QBuf::load(&gpu, &store, &format!("{a}k_proj.weight"))?,
                            v: QBuf::load(&gpu, &store, &format!("{a}v_proj.weight"))?,
                            o: QBuf::load(&gpu, &store, &format!("{a}o_proj.weight"))?,
                            q_norm: floats(&gpu, &store, &format!("{a}q_norm.weight"))?,
                            k_norm: floats(&gpu, &store, &format!("{a}k_norm.weight"))?,
                            kcache: gpu.zeroed::<f16>(cfg.kv_heads * cap * cfg.head_dim),
                            vcache: gpu.zeroed::<f16>(cfg.kv_heads * cap * cfg.head_dim),
                        })),
                        ffn: Ffn {
                            norm: floats(
                                &gpu,
                                &store,
                                "mtp.layers.0.post_attention_layernorm.weight",
                            )?,
                            gate: QBuf::load(&gpu, &store, "mtp.layers.0.mlp.gate_proj.weight")?,
                            up: QBuf::load(&gpu, &store, "mtp.layers.0.mlp.up_proj.weight")?,
                            down: QBuf::load(&gpu, &store, "mtp.layers.0.mlp.down_proj.weight")?,
                        },
                    },
                    norm: floats(&gpu, &store, "mtp.norm.weight")?,
                })
            } else {
                None
            };

            let (embed, _) = store.floats("model.language_model.embed_tokens.weight")?;
            // Whole splits of the cache, as the split kernel indexes them.
            let splits = cap.div_ceil(ATTN_BK * ATTN_BPS);
            let acts_for = |t: usize, narrow: bool| {
                let f = |n: usize| gpu.zeroed::<f32>(t * n);
                Acts {
                    x: f(cfg.hidden),
                    x2: f(cfg.hidden),
                    // The batched path reads this back once per threadgroup.
                    h: if narrow {
                        gpu.zeroed::<f16>(t * cfg.hidden)
                    } else {
                        f(cfg.hidden)
                    },
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
                    ffn_a: if narrow {
                        gpu.zeroed::<f16>(t * cfg.ffn)
                    } else {
                        f(cfg.ffn)
                    },
                    cos: f(cfg.rot / 2),
                    sin: f(cfg.rot / 2),
                    logits: f(cfg.vocab),
                    scalars_pos: gpu.zeroed::<u32>(1),
                    scalars_attn: gpu.zeroed::<u32>(2),
                    scalars_len: gpu.zeroed::<u32>(1),
                    scalars_nsplit: gpu.zeroed::<u32>(2),
                    part_m: f(splits * cfg.heads),
                    part_l: f(splits * cfg.heads),
                    part_acc: f(splits * cfg.heads * cfg.head_dim),
                }
            };
            let acts = acts_for(1, false);
            let bacts = acts_for(PREFILL_MAX, X_DTYPE != DType::F32);
            let out_norm = floats(&gpu, &store, "model.language_model.norm.weight")?;
            let lm_head = QBuf::load(&gpu, &store, "lm_head.weight")?;
            let gemm_ok = mv_keys(&layers, &lm_head, mtp.as_ref())
                .iter()
                .all(|&(_, _, _, layout)| layout == QLayout::NVFP4);
            let keys = mv_keys(&layers, &lm_head, mtp.as_ref());
            let int8_on = crate::dev::gemm_backend() == lex_msl::gemm::Backend::Cuda
                && gemm_ok
                // Opt-in: 66% faster decode on an L4 (112 -> 67 ms a token),
                // but 8-bit activations moved a golden log-prob by 0.027
                // against a 0.02 tolerance (2026-09-29). Off until the
                // arithmetic passes.
                && std::env::var("LEX_INT8").is_ok_and(|v| v == "1")
                && keys.iter().all(|&(n_in, ..)| lex_msl::int8::fits(n_in));
            let int8 = if int8_on {
                let widest = keys.iter().map(|&(n_in, ..)| n_in).max().unwrap_or(0);
                let mut i8k = Int8 {
                    mm: HashMap::new(),
                    quant: HashMap::new(),
                    xq: gpu.zeroed::<i16>(SPEC_MAX * widest),
                    xs: gpu.zeroed::<f32>(SPEC_MAX * widest / 16),
                };
                // Decode's: one token, f32 activations.
                for &(n_in, n_out, res, _) in &keys {
                    let mm = lex_msl::int8::matmul_int8(1, n_out, n_in, res)?;
                    i8k.mm
                        .entry((1, n_out, n_in, res))
                        .or_insert(gpu.build_lowered(&mm)?);
                    if let std::collections::hash_map::Entry::Vacant(v) =
                        i8k.quant.entry((1, n_in, false))
                    {
                        v.insert(gpu.build_lowered(&lex_msl::int8::quant16(1, n_in, false)?)?);
                    }
                }
                Some(i8k)
            } else {
                None
            };
            for key in mv_keys(&layers, &lm_head, mtp.as_ref()) {
                if let std::collections::hash_map::Entry::Vacant(slot) = k.mv.entry(key) {
                    let (n_in, n_out, res, layout) = key;
                    let p = matvec_q(n_in, n_out, bo(gpu.target()), n_in, layout, res)?;
                    slot.insert(compile(&gpu, &p, THREADS)?);
                }
            }
            let mtp_in = gpu.zeroed::<f32>(2 * cfg.hidden);
            let mtp_in_b = gpu.zeroed::<f32>(PREFILL_MAX * 2 * cfg.hidden);

            // Only a checkpoint with a draft head can reject anything.
            let snap = if mtp.is_some() {
                let (sn, cn) = (hv * dv * dk, (cfg.conv_kernel - 1) * ch);
                let build = |n: usize| -> Result<Pipeline, String> {
                    let kern = Kernel::copy(DType::F32, n);
                    let pl = plan(&kern, gpu.target()).map_err(|e| e.to_string())?;
                    gpu.build(&kern, &pl)
                };
                let delta = layers
                    .iter()
                    .filter(|l| matches!(l.mixer, Mixer::Linear(_)))
                    .count();
                let block = |rows: usize, cols: usize| -> Result<Vec<Pipeline>, String> {
                    (0..SPEC_MAX)
                        .map(|w| {
                            let p = lex_front::qwen::copy_block(SPEC_MAX, rows, cols, w)?;
                            compile(&gpu, &p, THREADS)
                        })
                        .collect()
                };
                Some(Snapshot {
                    state: (0..delta).map(|_| gpu.zeroed::<f32>(sn)).collect(),
                    conv: (0..delta).map(|_| gpu.zeroed::<f32>(cn)).collect(),
                    copy_state: build(sn)?,
                    copy_conv: build(cn)?,
                    pos: 0,
                    rows_state: (0..delta)
                        .map(|_| gpu.zeroed::<f32>(SPEC_MAX * sn))
                        .collect(),
                    rows_conv: (0..delta)
                        .map(|_| gpu.zeroed::<f32>(SPEC_MAX * cn))
                        .collect(),
                    roll_state: block(hv * dv, dk)?,
                    roll_conv: block(cfg.conv_kernel - 1, ch)?,
                })
            } else {
                None
            };
            Ok(Runner {
                gpu,
                cfg,
                k,
                layers,
                embed,
                out_norm,
                lm_head,
                mtp,
                mtp_in,
                mtp_in_b,
                mtp_pos: 0,
                snap,
                spec_h: None,
                skip: vec![],
                acts,
                bacts,
                batches: HashMap::new(),
                cap,
                pos: 0,
                sync: std::env::var_os("LEX_SYNC").is_some(),
                gemm_ok,
                int8,
                prof: std::cell::RefCell::new(std::collections::BTreeMap::new()),
            })
        }

        pub fn device(&self) -> String {
            self.gpu.info().name
        }

        pub fn pos(&self) -> usize {
            self.pos
        }

        /// Splits the cache is cut into at this length.
        fn nsplit(&self, len: usize) -> usize {
            len.div_ceil(ATTN_BK * ATTN_BPS)
        }

        /// The residual stream after the last layer and *before* the final
        /// norm, as it stands after the most recent step.
        ///
        /// This is what the checkpoint's multi-token-prediction head takes:
        /// it carries its own `pre_fc_norm_hidden`, so it wants the
        /// unnormalised state. Reading it out is how the draft head is
        /// measured without a second forward pass.
        /// Copy every gated-delta layer's memory aside, with the position
        /// it belongs to.
        ///
        /// Public because a rollback is also the only honest way to time
        /// the same work twice: 48 layers carry a recurrent state that a
        /// forward pass mutates, so a second pass at "the same context" is
        /// not at the same context unless the state goes back too.
        pub fn save(&mut self) {
            let Some(s) = &self.snap else { return };
            let mut steps: Vec<(&Pipeline, Vec<&Buffer>)> = vec![];
            let mut i = 0;
            for l in &self.layers {
                if let Mixer::Linear(d) = &l.mixer {
                    steps.push((&s.copy_state, vec![&d.state, &s.state[i]]));
                    steps.push((&s.copy_conv, vec![&d.conv_state, &s.conv[i]]));
                    i += 1;
                }
            }
            let launches: Vec<Step<'_>> = steps
                .iter()
                .map(|(p, b)| (*p, b.as_slice(), None))
                .collect();
            self.gpu.run_launches(&launches);
            drop(launches);
            drop(steps);
            let pos = self.pos;
            if let Some(s) = &mut self.snap {
                s.pos = pos;
            }
        }

        /// Put it back, and with it the position. An attention layer needs
        /// nothing: its cache is overwritten as positions are refilled.
        pub fn restore(&mut self) {
            let Some(s) = &self.snap else { return };
            let mut steps: Vec<(&Pipeline, Vec<&Buffer>)> = vec![];
            let mut i = 0;
            for l in &self.layers {
                if let Mixer::Linear(d) = &l.mixer {
                    steps.push((&s.copy_state, vec![&s.state[i], &d.state]));
                    steps.push((&s.copy_conv, vec![&s.conv[i], &d.conv_state]));
                    i += 1;
                }
            }
            let launches: Vec<Step<'_>> = steps
                .iter()
                .map(|(p, b)| (*p, b.as_slice(), None))
                .collect();
            self.gpu.run_launches(&launches);
            drop(launches);
            drop(steps);
            self.pos = s.pos;
        }

        /// Put every gated-delta layer back to where it stood after token
        /// `kept` of the batch just run, and the position with it.
        ///
        /// This replaces `restore()` plus a replay of the accepted prefix.
        /// The replay was a whole extra pass over the weights -- 38.6 ms
        /// against a 42.8 ms verify -- on every round the model disagreed
        /// with a draft. This is two copies a layer.
        ///
        /// Returns false when the batch was not run with snapshots, and
        /// the caller has to fall back to restoring and replaying.
        fn roll_back_to(&mut self, kept: usize, t: usize) -> bool {
            let Some(s) = &self.snap else { return false };
            if kept + 1 >= t || t > SPEC_MAX || kept >= s.roll_state.len() {
                return false;
            }
            let mut steps: Vec<(&Pipeline, Vec<&Buffer>)> = vec![];
            let mut i = 0;
            for l in &self.layers {
                if let Mixer::Linear(d) = &l.mixer {
                    steps.push((&s.roll_state[kept], vec![&s.rows_state[i], &d.state]));
                    steps.push((&s.roll_conv[kept], vec![&s.rows_conv[i], &d.conv_state]));
                    i += 1;
                }
            }
            let launches: Vec<Step<'_>> = steps
                .iter()
                .map(|(p, b)| (*p, b.as_slice(), None))
                .collect();
            self.gpu.run_launches(&launches);
            drop(launches);
            drop(steps);
            // The batch advanced the position by `t`; only `kept + 1` of
            // those tokens were really said. An attention layer needs
            // nothing -- its cache is overwritten as positions are
            // refilled -- which is the same reason `restore` gives.
            self.pos = self.pos - t + kept + 1;
            true
        }

        /// One speculative round, greedy.
        ///
        /// `last` is the token to be fed next; it is not yet in the state.
        /// The head drafts `depth` tokens after it, and all `depth + 1` are
        /// fed in a single pass over the weights. A draft is kept while
        /// every draft before it was right, because the pass only tells us
        /// what the model would have said *given* that prefix.
        ///
        /// Returns the tokens now committed, and the token to feed next --
        /// which this pass produced for free, so a round always yields at
        /// least one token more than it drafted correctly.
        ///
        /// On a full acceptance the state is already right and nothing is
        /// undone. Otherwise the gated-delta layers are restored and the
        /// accepted prefix replayed, which costs a second pass; at the
        /// acceptance this head reaches that is rare enough to be worth it.
        pub fn speculate(&mut self, last: u32, depth: usize) -> Result<(Vec<u32>, u32), String> {
            self.speculate_with(last, depth, &mut Sampler::new(0.0, 1.0, 1, 0))
        }

        /// [`Self::speculate`] for a sampled distribution.
        ///
        /// The drafts are checked with [`Sampler::verify_draft`] rather
        /// than against the argmax, so what comes out is distributed
        /// exactly as sampling one token at a time would have been:
        /// speculation changes how fast the tokens arrive, never which
        /// ones. Before this, sampling and speculation could not both be
        /// on, and turning sampling on cost 1.64x of decode on an M4 --
        /// which, since the two engines draw the same power, was 1.64x of
        /// the joules as well.
        ///
        /// At temperature 0 this is the greedy check, token for token.
        pub fn speculate_with(
            &mut self,
            last: u32,
            depth: usize,
            sampler: &mut Sampler,
        ) -> Result<(Vec<u32>, u32), String> {
            if self.mtp.is_none() || depth == 0 {
                let logits = self.step(last)?;
                return Ok((vec![last], sampler.pick(&logits)));
            }
            let trace = std::env::var_os("LEX_SPEC_TRACE").is_some();
            let mark = std::time::Instant::now();
            let drafts = self.draft(depth, last)?;
            let t_draft = mark.elapsed().as_secs_f64() * 1e3;
            if drafts.is_empty() {
                let logits = self.step(last)?;
                return Ok((vec![last], sampler.pick(&logits)));
            }

            // `save` exists for the replay path: restore the pre-batch
            // state, feed the accepted prefix again. With per-token
            // snapshots nothing reads it, so on that path it is 0.8 ms of
            // copying a state that is never put back.
            let mark = std::time::Instant::now();
            let batch = 1 + drafts.len();
            let snapped = batch <= SPEC_MAX
                && self
                    .batches
                    .get(&batch)
                    .is_some_and(|b| b.delta_snap.is_some() && b.conv_snap.is_some());
            if !snapped {
                self.save();
            }
            let t_save = mark.elapsed().as_secs_f64() * 1e3;
            let mut fed = vec![last];
            fed.extend(&drafts);
            let mark = std::time::Instant::now();
            let logits = self.forward_with(&fed, true, true)?;
            let t_verify = mark.elapsed().as_secs_f64() * 1e3;

            // How many drafts survive, longest prefix only. The first
            // rejection also decides what is said in its place, drawn
            // from the target with the rejected draft taken out; if every
            // draft survives, the row after the last one is a free token.
            let mut kept = 0;
            let mut instead = None;
            while kept < drafts.len() {
                match sampler.verify_draft(&logits[kept], drafts[kept]) {
                    Ok(()) => kept += 1,
                    Err(t) => {
                        instead = Some(t);
                        break;
                    }
                }
            }
            let next = instead.unwrap_or_else(|| sampler.pick(&logits[kept]));
            let committed = fed[..=kept].to_vec();
            // Row `kept` of the verify is the state after exactly the tokens
            // being committed, so it is the right input for the next draft
            // whether or not the rest of the pass is undone. Without this the
            // next draft reads whatever the last single step left behind, and
            // guesses from a state the model was never in.
            let carry = self.batch_hidden(kept);

            if kept < drafts.len() {
                // The pass ran further than the model agreed with, so the
                // delta states carry tokens that were never really said.
                // With snapshots that is a copy; without, the only way
                // back is the pre-batch state and a replay of the prefix.
                if !self.roll_back_to(kept, fed.len()) {
                    self.restore();
                    if kept > 0 {
                        self.forward(&committed, false)?;
                    } else {
                        self.step(last)?;
                    }
                }
            }
            self.spec_h = Some(carry);
            if trace {
                eprintln!(
                    "draft {t_draft:.1}  save {t_save:.1}  verify {t_verify:.1}  \
                     undo {:.1}  kept {kept}",
                    mark.elapsed().as_secs_f64() * 1e3 - t_verify
                );
            }
            Ok((committed, next))
        }

        /// Does this checkpoint carry a draft head?
        pub fn has_mtp(&self) -> bool {
            self.mtp.is_some()
        }

        /// Draft the next `depth` tokens with the checkpoint's own
        /// multi-token-prediction head, continuing from the last step.
        ///
        /// `last` is the token that step produced. Each draft fuses a
        /// hidden state with the embedding of the token before it, so the
        /// first uses the model's own hidden state and the rest use the
        /// head's, which is what lets one head draft more than one token.
        ///
        /// A draft is the head plus a pass over `lm_head`, about 6.6% of a
        /// model pass, so depth is not free: it buys tokens only while
        /// acceptance holds up.
        ///
        /// The head's cache is its own and advances with `mtp_pos`;
        /// [`Self::reset`] clears it. Nothing here touches the model's
        /// state, so a rejected draft costs only the time.
        pub fn draft(&mut self, depth: usize, last: u32) -> Result<Vec<u32>, String> {
            if self.mtp.is_none() || depth == 0 {
                return Ok(vec![]);
            }
            let c = self.cfg.clone();
            let mut h = self.spec_h.take().unwrap_or_else(|| self.hidden());
            let mut tok = last;
            let mut out = Vec::with_capacity(depth);
            for _ in 0..depth {
                if self.mtp_pos >= self.cap {
                    break;
                }
                // The two halves, normalised on the host: 10240 values
                // against the 239 MB the head is about to read.
                let m = self.mtp.as_ref().expect("checked");
                let row = tok as usize * c.hidden;
                let mut fused = vec![0.0f32; 2 * c.hidden];
                rms_into(
                    &self.embed[row..row + c.hidden],
                    &m.pre_e,
                    c.eps,
                    &mut fused[..c.hidden],
                );
                rms_into(&h, &m.pre_h, c.eps, &mut fused[c.hidden..]);
                self.gpu.write(&self.mtp_in, 0, &fused);

                let (cos, sin) = rope_tables(self.mtp_pos, c.rot, c.theta);
                self.gpu.write(&self.acts.cos, 0, &cos);
                self.gpu.write(&self.acts.sin, 0, &sin);
                self.gpu
                    .write(&self.acts.scalars_pos, 0, &[self.mtp_pos as u32]);
                let len = self.mtp_pos + 1;
                self.gpu.write(
                    &self.acts.scalars_attn,
                    0,
                    &[len as u32, len.div_ceil(ATTN_BK) as u32],
                );

                let plan: Vec<Dispatch<'_>> = self
                    .mtp_plan()
                    .into_iter()
                    .map(|(p, b)| ("draft", p, b, None))
                    .collect();
                let plan = self.int8_plan(plan, 1, false);
                let steps: Vec<Step<'_>> = plan
                    .iter()
                    .map(|(_, p, b, g)| (*p, b.as_slice(), *g))
                    .collect();
                self.gpu.run_launches(&steps);
                drop(plan);
                self.mtp_pos += 1;

                let mut logits = vec![0.0f32; c.vocab];
                self.gpu.download(&self.acts.logits, &mut logits);
                tok = (0..logits.len())
                    .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                    .expect("logits") as u32;
                out.push(tok);
                // The head's own hidden state carries the next draft.
                h = self.hidden();
            }
            Ok(out)
        }

        /// The draft head's dispatches: `fc`, then one attention layer and
        /// its feed-forward, then the shared norm and `lm_head`. The same
        /// shape as any attention layer in the model, which is why it needs
        /// no kernels of its own.
        /// Every row of the last batched pass, before the head overwrites
        /// them. One download, because `batch_hidden` of the last row
        /// moves every earlier row anyway.
        fn batch_hidden_all(&self, t: usize) -> Vec<f32> {
            let mut all = vec![0.0f32; t * self.cfg.hidden];
            self.gpu.download(&self.bacts.x, &mut all);
            all
        }

        /// Feed a prompt, keeping the draft head's cache in step with the
        /// model's.
        ///
        /// The head drafts token `t+2` from the model's hidden state at
        /// `t` and the embedding of token `t+1`. Those hidden states exist
        /// only while the prompt is being fed, which makes this the only
        /// place the head can be warmed: afterwards they are gone, and
        /// `mtp_pos` sits at 0 while the model is at 1440. The head then
        /// drafts from a position the text never passed through, and
        /// acceptance falls from 0.89 to 0.54 -- which costs more than the
        /// verify does, because every rejection replays a whole pass.
        ///
        /// Returns the logits after the last token, as `step` would.
        pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
            if tokens.is_empty() {
                return Err("nothing to prefill".into());
            }
            let n = tokens.len();
            let c = self.cfg.clone();
            let mut logits = vec![];
            let mut done = 0;
            while done < n {
                let t = self.chunk(n - done);
                // Only the last row's logits: the rest are never read, and
                // at 64 tokens they are 64 MB of download a chunk.
                let out = self.forward(&tokens[done..done + t], false)?;
                logits = out.last().cloned().expect("a batch is never empty");
                let hs = self.batch_hidden_all(t);
                // The last prompt position pairs with a token the prompt
                // does not have -- the one the model is about to generate.
                // That row is the first real draft's, so it is left for it.
                let warm = t.min(n - 1 - done);
                if self.mtp.is_some() && warm > 0 {
                    self.mtp_warm(&hs, &tokens[done + 1..done + 1 + warm])?;
                }
                // `forward` leaves the state in `bacts`, and `draft` reads
                // the single-row `acts`. Handing the last row over is what
                // keeps the first draft from guessing off a stale step.
                if done + t == n {
                    let h = c.hidden;
                    self.spec_h = Some(hs[(t - 1) * h..t * h].to_vec());
                }
                done += t;
            }
            Ok(logits)
        }

        /// Advance the head's cache over `next.len()` positions at once.
        ///
        /// `hs` is the model's hidden state per row and `next[j]` is the
        /// token after row `j` -- the pair the head predicts from. The
        /// two halves are normalised on the host, as the single-row draft
        /// does: 10240 values per row against the 239 MB the head reads.
        fn mtp_warm(&mut self, hs: &[f32], next: &[u32]) -> Result<(), String> {
            // In the chunk sizes prefill uses, which are compiled at load.
            // A prefill chunk warms one row fewer than it read -- the last
            // prompt position pairs with a token the prompt does not have --
            // so warming in one piece asked for a batch of 63 or 127, a size
            // nothing had compiled: on CUDA a whole batch's worth of NVRTC,
            // 43 s inside a 512-token prefill whose kernels took 2.7.
            let h = self.cfg.hidden;
            let mut done = 0;
            while done < next.len() {
                let t = self.chunk(next.len() - done);
                self.mtp_warm_rows(&hs[done * h..(done + t) * h], &next[done..done + t])?;
                done += t;
            }
            Ok(())
        }

        fn mtp_warm_rows(&mut self, hs: &[f32], next: &[u32]) -> Result<(), String> {
            let t = next.len();
            self.batch(t)?;
            let c = self.cfg.clone();
            let m = self.mtp.as_ref().expect("checked by the caller");
            let (pre_e, pre_h) = (m.pre_e.clone(), m.pre_h.clone());
            let mut fused = vec![0.0f32; t * 2 * c.hidden];
            for (j, &tok) in next.iter().enumerate() {
                let row = tok as usize * c.hidden;
                let at = j * 2 * c.hidden;
                rms_into(
                    &self.embed[row..row + c.hidden],
                    &pre_e,
                    c.eps,
                    &mut fused[at..at + c.hidden],
                );
                rms_into(
                    &hs[j * c.hidden..(j + 1) * c.hidden],
                    &pre_h,
                    c.eps,
                    &mut fused[at + c.hidden..at + 2 * c.hidden],
                );
            }
            self.gpu.write(&self.mtp_in_b, 0, &fused);

            let pos0 = self.mtp_pos;
            if pos0 + t > self.cap {
                return Err(format!("the head's cache is full ({} positions)", self.cap));
            }
            for j in 0..t {
                let (cos, sin) = rope_tables(pos0 + j, c.rot, c.theta);
                self.gpu.write(&self.bacts.cos, j * c.rot / 2, &cos);
                self.gpu.write(&self.bacts.sin, j * c.rot / 2, &sin);
            }
            self.gpu.write(&self.bacts.scalars_pos, 0, &[pos0 as u32]);
            self.gpu.write(
                &self.bacts.scalars_attn,
                0,
                &[pos0 as u32, (pos0 + t).div_ceil(ATTN_BK) as u32],
            );
            let nsplit = self.nsplit(pos0 + t);
            self.gpu.write(
                &self.bacts.scalars_nsplit,
                0,
                &[nsplit as u32, nsplit.div_ceil(COMBINE_CHUNK) as u32],
            );

            let plan = self.int8_plan(self.mtp_batch_plan(t), t, true);
            let steps: Vec<Step<'_>> = plan
                .iter()
                .map(|(_, p, b, g)| (*p, b.as_slice(), *g))
                .collect();
            self.gpu.run_launches(&steps);
            drop(steps);
            drop(plan);
            self.mtp_pos += t;
            Ok(())
        }

        /// The draft head over `t` rows at once, stopping before the
        /// final norm and `lm_head`.
        ///
        /// Warming the head only needs its KV cache filled; the drafts it
        /// would have made for tokens the model has already read are of no
        /// use to anyone. Leaving `lm_head` out drops the most expensive
        /// dispatch in the plan and a `t x vocab` download with it.
        fn mtp_batch_plan(&self, t: usize) -> Vec<Dispatch<'_>> {
            let (m, a, k) = (
                self.mtp.as_ref().expect("checked by the caller"),
                &self.bacts,
                &self.batches[&t],
            );
            let at = match &m.layer.mixer {
                Mixer::Attn(at) => at,
                Mixer::Linear(_) => unreachable!("the draft head is an attention layer"),
            };
            let f = &m.layer.ffn;
            let mut d: Vec<Dispatch<'_>> = vec![
                (
                    "mtp fc",
                    &k.mv[&m.fc.key(false)],
                    m.fc.bind(&self.mtp_in_b, None, &a.x),
                    None,
                ),
                ("mtp rmsnorm", &k.rms, vec![&a.x, &at.norm, &a.h], None),
                (
                    "mtp matvec q",
                    &k.mv[&at.q.key(false)],
                    at.q.bind(&a.h, None, &a.q32),
                    None,
                ),
                (
                    "mtp matvec k/v",
                    &k.mv[&at.k.key(false)],
                    at.k.bind(&a.h, None, &a.k32),
                    None,
                ),
                (
                    "mtp matvec k/v",
                    &k.mv[&at.v.key(false)],
                    at.v.bind(&a.h, None, &a.v32),
                    None,
                ),
                (
                    "mtp rope",
                    &k.rope_q,
                    vec![&a.q32, &at.q_norm, &a.cos, &a.sin, &a.q16, &a.gate],
                    None,
                ),
                (
                    "mtp rope",
                    &k.rope_k,
                    vec![&a.k32, &at.k_norm, &a.cos, &a.sin, &a.k16],
                    None,
                ),
                (
                    "mtp kv append",
                    &k.kv_k,
                    vec![&a.k16, &at.kcache, &a.scalars_pos],
                    None,
                ),
                (
                    "mtp kv append",
                    &k.kv_v,
                    vec![&a.v32, &at.vcache, &a.scalars_pos],
                    None,
                ),
            ];
            // The head's cache is its own and shorter than the model's, so
            // it crosses the split threshold later -- but it crosses it.
            let nsplit = self.nsplit(self.mtp_pos + t);
            if nsplit >= attn_min_splits() {
                d.push((
                    "mtp attention",
                    &k.attn_split,
                    vec![
                        &a.q16,
                        &at.kcache,
                        &at.vcache,
                        &a.part_m,
                        &a.part_l,
                        &a.part_acc,
                        &a.scalars_pos,
                    ],
                    Some([self.cfg.kv_heads, nsplit]),
                ));
                d.push((
                    "mtp attention",
                    &k.attn_combine,
                    vec![
                        &a.part_m,
                        &a.part_l,
                        &a.part_acc,
                        &a.attn,
                        &a.scalars_nsplit,
                    ],
                    None,
                ));
            } else {
                d.push((
                    "mtp attention",
                    &k.attn,
                    vec![&a.q16, &at.kcache, &at.vcache, &a.attn, &a.scalars_attn],
                    None,
                ));
            }
            d.extend([
                (
                    "mtp gate mul",
                    &k.mul,
                    vec![&a.attn, &a.gate, &a.gated],
                    None,
                ),
                (
                    "mtp matvec o_proj",
                    &k.mv[&at.o.key(true)],
                    at.o.bind(&a.gated, Some(&a.x), &a.x2),
                    None,
                ),
                ("mtp rmsnorm", &k.rms, vec![&a.x2, &f.norm, &a.h], None),
                (
                    "mtp matvec gate/up",
                    &k.mv[&f.gate.key(false)],
                    f.gate.bind(&a.h, None, &a.ffn_g),
                    None,
                ),
                (
                    "mtp matvec gate/up",
                    &k.mv[&f.up.key(false)],
                    f.up.bind(&a.h, None, &a.ffn_u),
                    None,
                ),
                (
                    "mtp silu_mul",
                    &k.silu,
                    vec![&a.ffn_g, &a.ffn_u, &a.ffn_a],
                    None,
                ),
                (
                    "mtp matvec down",
                    &k.mv[&f.down.key(true)],
                    f.down.bind(&a.ffn_a, Some(&a.x2), &a.x),
                    None,
                ),
            ]);
            d
        }

        fn mtp_plan(&self) -> Vec<(&Pipeline, Vec<&Buffer>)> {
            let (m, a, k) = (
                self.mtp.as_ref().expect("checked by the caller"),
                &self.acts,
                &self.k,
            );
            let at = match &m.layer.mixer {
                Mixer::Attn(at) => at,
                Mixer::Linear(_) => unreachable!("the draft head is an attention layer"),
            };
            let f = &m.layer.ffn;
            vec![
                (self.mv(&m.fc, false), m.fc.bind(&self.mtp_in, None, &a.x)),
                (&k.rms, vec![&a.x, &at.norm, &a.h]),
                (self.mv(&at.q, false), at.q.bind(&a.h, None, &a.q32)),
                (self.mv(&at.k, false), at.k.bind(&a.h, None, &a.k32)),
                (self.mv(&at.v, false), at.v.bind(&a.h, None, &a.v32)),
                (
                    &k.rope_q,
                    vec![&a.q32, &at.q_norm, &a.cos, &a.sin, &a.q16, &a.gate],
                ),
                (&k.rope_k, vec![&a.k32, &at.k_norm, &a.cos, &a.sin, &a.k16]),
                (&k.kv_k, vec![&a.k16, &at.kcache, &a.scalars_pos]),
                (&k.kv_v, vec![&a.v32, &at.vcache, &a.scalars_pos]),
                (
                    &k.attn,
                    vec![&a.q16, &at.kcache, &at.vcache, &a.attn, &a.scalars_attn],
                ),
                (&k.mul, vec![&a.attn, &a.gate, &a.gated]),
                (self.mv(&at.o, true), at.o.bind(&a.gated, Some(&a.x), &a.x2)),
                (&k.rms, vec![&a.x2, &f.norm, &a.h]),
                (self.mv(&f.gate, false), f.gate.bind(&a.h, None, &a.ffn_g)),
                (self.mv(&f.up, false), f.up.bind(&a.h, None, &a.ffn_u)),
                (&k.silu, vec![&a.ffn_g, &a.ffn_u, &a.ffn_a]),
                (
                    self.mv(&f.down, true),
                    f.down.bind(&a.ffn_a, Some(&a.x2), &a.x),
                ),
                (&k.rms, vec![&a.x, &m.norm, &a.h]),
                (
                    self.mv(&self.lm_head, false),
                    self.lm_head.bind(&a.h, None, &a.logits),
                ),
            ]
        }

        /// The hidden state a batched pass left for its `i`-th token.
        fn batch_hidden(&self, i: usize) -> Vec<f32> {
            let n = self.cfg.hidden;
            let mut all = vec![0.0f32; (i + 1) * n];
            self.gpu.download(&self.bacts.x, &mut all);
            all[i * n..(i + 1) * n].to_vec()
        }

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
            self.mtp_pos = 0;
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

        /// The recurrent state of every linear layer, so a later prompt
        /// sharing this prefix can resume from here instead of re-reading
        /// it.
        ///
        /// The 16 attention layers need nothing kept: their KV cache is
        /// indexed by position, so a shared prefix is still sitting there.
        /// A gated-delta layer is the opposite -- it compresses its whole
        /// prefix into one evolving state, with nothing to index into, so
        /// reuse means having kept a copy. 48 of this model's 64 layers
        /// are that kind, which is why a prefix cache here is not the
        /// usual one.
        ///
        /// 151 MB and about 0.4 ms for this model, against the minutes a
        /// re-prefill costs.
        pub fn checkpoint(&self) -> Checkpoint {
            let n = self.cfg.v_heads * self.cfg.v_dim * self.cfg.k_dim;
            let w = (self.cfg.conv_kernel - 1) * self.cfg.conv_channels();
            let (mut state, mut conv) = (vec![], vec![]);
            for l in &self.layers {
                if let Mixer::Linear(lin) = &l.mixer {
                    let mut x = vec![0.0f32; n];
                    self.gpu.download(&lin.state, &mut x);
                    state.push(x);
                    let mut c = vec![0.0f32; w];
                    self.gpu.download(&lin.conv_state, &mut c);
                    conv.push(c);
                }
            }
            Checkpoint {
                pos: self.pos,
                mtp_pos: self.mtp_pos,
                state,
                conv,
            }
        }

        /// Resume from a checkpoint. The caller must have established that
        /// the tokens up to `c.pos` are the same ones that produced it --
        /// the attention layers read their KV cache straight through, so a
        /// mismatched prefix is not an error, it is wrong numbers.
        ///
        /// Named apart from `restore`, which puts back a speculation
        /// snapshot and is a different thing at a different scale.
        pub fn resume(&mut self, c: &Checkpoint) {
            self.pos = c.pos;
            self.mtp_pos = c.mtp_pos;
            self.spec_h = None;
            let mut i = 0;
            for l in &self.layers {
                if let Mixer::Linear(lin) = &l.mixer {
                    self.gpu.write(&lin.state, 0, &c.state[i]);
                    self.gpu.write(&lin.conv_state, 0, &c.conv[i]);
                    i += 1;
                }
            }
        }

        /// Feed one token at the next position; returns its logits.
        pub fn step(&mut self, token: u32) -> Result<Vec<f32>, String> {
            self.spec_h = None;
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
            self.gpu.write(&a.scalars_len, 0, &[len as u32]);
            // The combine takes the live split count and how many chunks
            // of COMBINE_CHUNK they make -- not the head count. Writing the
            // wrong second scalar makes it reduce a prefix of the splits
            // and quietly drop the rest of the cache.
            let nsplit = self.nsplit(len);
            self.gpu.write(
                &a.scalars_nsplit,
                0,
                &[nsplit as u32, nsplit.div_ceil(COMBINE_CHUNK) as u32],
            );

            let mut plan = self.plan();
            if !self.skip.is_empty() {
                plan.retain(|d| !self.skip.iter().any(|s| d.0 == s));
            }
            let plan = self.int8_plan(plan, 1, false);
            self.dispatch(&plan);
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
            self.forward_with(tokens, all, false)
        }

        /// [`Self::forward`] recording where each gated-delta layer stood
        /// after every token, so a rejected speculative batch can be undone
        /// with a copy instead of a replay. Prefill does not ask for this:
        /// it never rolls back, and the snapshots are 3.1 MB a layer.
        fn forward_with(
            &mut self,
            tokens: &[u32],
            all: bool,
            snap: bool,
        ) -> Result<Vec<Vec<f32>>, String> {
            let t = tokens.len();
            let most = if self.gemm_ok { PREFILL_MAX } else { MAX_BATCH };
            if t == 0 || t > most {
                return Err(format!("a batch is 1..={most} tokens, not {t}"));
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
            // The split path's two scalars: live splits, and the chunks of
            // COMBINE_CHUNK they make. The last token of the batch is the
            // one that reaches furthest, so it sets the count.
            let nsplit = self.nsplit(pos0 + t);
            self.gpu.write(
                &self.bacts.scalars_nsplit,
                0,
                &[nsplit as u32, nsplit.div_ceil(COMBINE_CHUNK) as u32],
            );

            let mut plan = self.batch_plan_with(t, snap);
            if !self.skip.is_empty() {
                // Exact labels, not prefixes. `"matvec qkv"` starts with
                // `"matvec q"`, so a prefix match silently ablates two call
                // sites and attributes both to one -- which is how the
                // query projection came to look like it cost 9% of prefill
                // when its own shape runs at 213 GB/s, the same as the
                // feed-forward's.
                plan.retain(|d| !self.skip.iter().any(|s| d.0 == s));
            }
            let plan = self.int8_plan(plan, t, true);
            // Same as `step`: with LEX_SYNC, dispatch one at a time and
            // record where the time went. Without it prefill can only be
            // measured in total, which is enough to see a chunk size cost
            // more than the one below it and not enough to say why.
            self.dispatch(&plan);
            drop(plan);
            self.pos += t;

            let v = c.vocab;
            if all {
                let mut flat = vec![0.0f32; t * v];
                self.gpu.download(&self.bacts.logits, &mut flat);
                Ok(flat.chunks(v).map(<[f32]>::to_vec).collect())
            } else {
                // Only the last row is wanted, so only the last row moves.
                let mut last = vec![0.0f32; v];
                self.gpu
                    .download_at(&self.bacts.logits, (t - 1) * v, &mut last);
                Ok(vec![last])
            }
        }

        /// Compile the pipelines for every batch size a prefill chunk or a
        /// verify can have, now rather than on first use.
        ///
        /// On CUDA each size is an NVRTC compile of every kernel, seconds
        /// of it, and a size is otherwise compiled inside whichever request
        /// first needs it -- so a server's first speculative reply pays for
        /// it, and so did a benchmark that timed a first verify and
        /// concluded speculation was a loss. On Metal it is quick.
        pub fn compile_batches(&mut self) -> Result<(), String> {
            for t in 1..=MAX_BATCH {
                self.batch(t)?;
            }
            if self.gemm_ok {
                for t in GEMM_SIZES {
                    if t <= self.prefill_limit() {
                        self.batch(t)?;
                    }
                }
            }
            Ok(())
        }

        /// The batch sizes compiled so far, ascending.
        pub fn compiled_batch_sizes(&self) -> Vec<usize> {
            let mut v: Vec<usize> = self.batches.keys().copied().collect();
            v.sort_unstable();
            v
        }

        /// The largest prefill chunk: `LEX_PREFILL_CHUNK` if set (rounded
        /// down to a GEMM size, so a sweep never compiles a size the load
        /// did not), else 128 where the GEMM applies and `MAX_BATCH` where
        /// it does not. On an M4 Max, 512 tokens prefill at 87.4 tok/s in
        /// chunks of 8, 96.2 in 64s and 103 in 128s.
        fn prefill_limit(&self) -> usize {
            if !self.gemm_ok {
                return MAX_BATCH;
            }
            let want = std::env::var("LEX_PREFILL_CHUNK")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(128);
            GEMM_SIZES
                .into_iter()
                .find(|&g| g <= want)
                .unwrap_or(MAX_BATCH)
        }

        /// How many of `left` prompt tokens the next prefill chunk takes: the
        /// largest GEMM size that fits under the limit, else the rest up to
        /// `MAX_BATCH` for the batched matvec.
        fn chunk(&self, left: usize) -> usize {
            let limit = self.prefill_limit();
            GEMM_SIZES
                .into_iter()
                .find(|&g| self.gemm_ok && g <= limit && g <= left)
                .unwrap_or(left.min(MAX_BATCH))
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
            // Rows of the output one threadgroup owns. It sets how many
            // threadgroups there are, and every one of them re-reads every
            // token's activations -- so the wider the input and the more
            // tokens, the more `bo` is worth. LEX_BATCH_BO to sweep it.
            let bo = match std::env::var("LEX_BATCH_BO")
                .ok()
                .and_then(|v| v.parse().ok())
            {
                Some(v) if t > 1 => v,
                // 32 beats 16 at every batch size measured, and 64 is
                // worse than either: past 32 the accumulators per (row,
                // token) stop fitting.
                _ => 32,
            };
            let mut mv = HashMap::new();
            let want = mv_keys(&self.layers, &self.lm_head, self.mtp.as_ref());
            for key in want {
                if let std::collections::hash_map::Entry::Vacant(slot) = mv.entry(key) {
                    let (n_in, n_out, res, layout) = key;
                    // `res` marks the matmuls that accumulate into the
                    // residual -- down and o_proj -- and those read ffn_a
                    // and gated, not `h`, so they keep f32 inputs.
                    // Every batched matmul now reads a narrowed buffer
                    // except o_proj, which reads `gated`.
                    // The draft head's `fc` is the exception: it reads a
                    // fused input built on the host rather than a narrowed
                    // `h`, and the decode path's `fc` reads it in f32. The
                    // two write the same cache -- the batched one warming
                    // it over a prompt, the decode one drafting from it --
                    // so they have to agree. Handed f16 it reads the f32
                    // bytes as half pairs, and the head answers with one
                    // constant token whatever it is asked.
                    let xt = if batch_x_half(&c, n_in, res) {
                        DType::F16
                    } else {
                        DType::F32
                    };
                    if t > MAX_BATCH {
                        // Past the batched matvec's reach: the GEMM, bound
                        // exactly as `matmul_q_x` is.
                        if layout != QLayout::NVFP4 {
                            return Err(format!(
                                "a batch of {t} needs the GEMM, which reads NVFP4, not {layout:?}"
                            ));
                        }
                        let g = Gemm {
                            m: t,
                            n: n_out,
                            k: n_in,
                            residual: res,
                            x_half: xt == DType::F16,
                        };
                        let l = gemm_nvfp4(&g, crate::dev::gemm_backend())?;
                        slot.insert(gpu.build_lowered(&l)?);
                        continue;
                    }
                    let p = matmul_q_x(t, n_in, n_out, bo, n_in, layout, res, xt)?;
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
            let want_snap = t <= SPEC_MAX && self.snap.is_some();
            if t <= SPEC_MAX
                && let Some(i8k) = self.int8.as_mut()
            {
                for &(n_in, n_out, res, _) in
                    &mv_keys(&self.layers, &self.lm_head, self.mtp.as_ref())
                {
                    let xh = batch_x_half(&c, n_in, res);
                    if let std::collections::hash_map::Entry::Vacant(v) =
                        i8k.mm.entry((t, n_out, n_in, res))
                    {
                        v.insert(
                            gpu.build_lowered(&lex_msl::int8::matmul_int8(t, n_out, n_in, res)?)?,
                        );
                    }
                    if let std::collections::hash_map::Entry::Vacant(v) =
                        i8k.quant.entry((t, n_in, xh))
                    {
                        v.insert(gpu.build_lowered(&lex_msl::int8::quant16(t, n_in, xh)?)?);
                    }
                }
            }
            let b = Batch {
                rms: compile(
                    gpu,
                    &rmsnorm_rows(t, c.hidden, c.eps, None, X_DTYPE),
                    THREADS,
                )?,
                mv,
                dense: compile(
                    gpu,
                    &build_matvec_dense_rows(t, c.hidden, hv, 1, DType::F32, X_DTYPE)?,
                    THREADS,
                )?,
                conv: compile(
                    gpu,
                    &build_conv_silu_rows(t, ch, c.conv_kernel, 256)?,
                    THREADS,
                )?,
                qk_q: compile(
                    gpu,
                    &build_delta_qk_rows_in(
                        t,
                        c.k_heads,
                        c.per_key(),
                        dk,
                        ch,
                        0,
                        1.0 / dk as f32,
                        1e-6,
                        c.v_tiled,
                    )?,
                    128,
                )?,
                qk_k: compile(
                    gpu,
                    &build_delta_qk_rows_in(
                        t,
                        c.k_heads,
                        c.per_key(),
                        dk,
                        ch,
                        c.k_heads * dk,
                        (dk as f32).powf(-0.5),
                        1e-6,
                        c.v_tiled,
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
                attn_split: compile(gpu, &attn.build_causal_split(t, ATTN_BPS)?, 128)?,
                attn_combine: compile(gpu, &attn.build_combine_rows(t, ATTN_BPS)?, 128)?,
                mul: compile(gpu, &build_mul(t * c.heads * c.head_dim, 256)?, THREADS)?,
                silu: compile(
                    gpu,
                    &lex_front::llama::silu_mul(t * c.ffn, THREADS, X_DTYPE)?,
                    THREADS,
                )?,
                // Only a batch a verify can roll back from, and only when
                // there is a draft head to reject anything in the first
                // place. Prefill runs at MAX_BATCH and skips both.
                delta_snap: (want_snap)
                    .then(|| compile(gpu, &delta.build_steps_snap(t)?, 128))
                    .transpose()?,
                conv_snap: (want_snap)
                    .then(|| {
                        compile(
                            gpu,
                            &build_conv_silu_rows_snap(t, ch, c.conv_kernel, 256)?,
                            THREADS,
                        )
                    })
                    .transpose()?,
            };
            self.batches.insert(t, b);
            Ok(())
        }

        fn mv(&self, w: &QBuf, res: bool) -> &Pipeline {
            &self.k.mv[&w.key(res)]
        }

        /// A plan with its NVFP4 matvecs through the int8 path, where that is
        /// on and has kernels for `t` tokens; otherwise the plan as it was.
        ///
        /// Each matvec becomes a quantisation of its input and an int8
        /// matmul reading the quantised copy -- except that consecutive
        /// matvecs reading the same buffer (the q/k/v, z and a/b projections
        /// of one `h`) share one quantisation. `batch` says whose pipelines
        /// the plan used: the decode step's, or the batch of `t`'s.
        fn int8_plan<'a>(
            &'a self,
            plan: Vec<Dispatch<'a>>,
            t: usize,
            batch: bool,
        ) -> Vec<Dispatch<'a>> {
            let Some(i8k) = &self.int8 else { return plan };
            let mv = if batch {
                match self.batches.get(&t) {
                    Some(b) => &b.mv,
                    None => return plan,
                }
            } else {
                &self.k.mv
            };
            let c = &self.cfg;
            let mut out: Vec<Dispatch<'a>> = Vec::with_capacity(plan.len() + 32);
            // The input last quantised, and where its matmul went in `out`.
            let mut last: Option<(*const Buffer, usize)> = None;
            for (label, p, bufs, g) in plan {
                let key = mv
                    .iter()
                    .find(|(_, q)| std::ptr::eq(*q, p))
                    .map(|(k, _)| *k);
                let Some((n_in, n_out, res, _)) = key else {
                    out.push((label, p, bufs, g));
                    continue;
                };
                let xh = batch && batch_x_half(c, n_in, res);
                let (Some(mm), Some(quant)) = (
                    i8k.mm.get(&(t, n_out, n_in, res)),
                    i8k.quant.get(&(t, n_in, xh)),
                ) else {
                    out.push((label, p, bufs, g));
                    continue;
                };
                let x: &'a Buffer = bufs[0];
                let fresh = last.is_some_and(|(px, at)| std::ptr::eq(px, x) && at + 1 == out.len());
                if !fresh {
                    out.push(("quant", quant, vec![x, &i8k.xq, &i8k.xs], None));
                }
                // [x, q, s, gs, (r), y] becomes [xq, xs, q, s, gs, (r), y].
                let mut b: Vec<&'a Buffer> = vec![&i8k.xq, &i8k.xs];
                b.extend_from_slice(&bufs[1..]);
                out.push((label, mm, b, None));
                last = Some((x as *const Buffer, out.len() - 1));
            }
            out
        }

        /// Run a plan in one go -- or, with `sync`, timed per dispatch
        /// (`Gpu::run_each_timed`) and booked to each call site.
        fn dispatch(&self, plan: &[Dispatch<'_>]) {
            let steps: Vec<Step<'_>> = plan
                .iter()
                .map(|(_, p, b, g)| (*p, b.as_slice(), *g))
                .collect();
            if !self.sync {
                self.gpu.run_launches(&steps);
                return;
            }
            let times = self.gpu.run_each_timed(&steps);
            let mut prof = self.prof.borrow_mut();
            for ((label, ..), t) in plan.iter().zip(times) {
                let e = prof.entry(*label).or_insert((0, 0.0));
                e.0 += 1;
                e.1 += t;
            }
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
        /// [`Self::batch_plan`], optionally writing per-token snapshots of
        /// each gated-delta layer's memory.
        ///
        /// `snap` is what a speculative verify asks for: the layers whose
        /// state a rejected batch would otherwise have to replay record
        /// where they were after every token, so the undo becomes a copy.
        /// It falls back to the ordinary kernels when the batch is larger
        /// than `SPEC_MAX` or the checkpoint has no draft head, and the
        /// caller then pays the replay as before.
        fn batch_plan_with(&self, t: usize, snap: bool) -> Vec<Dispatch<'_>> {
            let a = &self.bacts;
            let k = &self.batches[&t];
            // The gated-delta layers in order, for indexing the snapshots.
            let mut li = 0usize;
            let snap = self
                .snap
                .as_ref()
                .filter(|_| snap && k.delta_snap.is_some() && k.conv_snap.is_some());
            let mut d: Vec<Dispatch<'_>> = vec![];
            for layer in &self.layers {
                d.push((
                    "rmsnorm",
                    &k.rms,
                    vec![&a.x, layer_norm(&layer.mixer), &a.h],
                    None,
                ));
                match &layer.mixer {
                    Mixer::Linear(l) => {
                        d.push((
                            "matvec qkv",
                            &k.mv[&l.qkv.key(false)],
                            l.qkv.bind(&a.h, None, &a.qkv),
                            None,
                        ));
                        d.push((
                            "matvec z",
                            &k.mv[&l.z.key(false)],
                            l.z.bind(&a.h, None, &a.z),
                            None,
                        ));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.a, &a.a], None));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.b, &a.b], None));
                        d.push(match snap {
                            Some(s) => (
                                "conv",
                                k.conv_snap.as_ref().expect("checked"),
                                vec![&l.conv_state, &a.qkv, &l.conv_w, &a.conv, &s.rows_conv[li]],
                                None,
                            ),
                            None => (
                                "conv",
                                &k.conv,
                                vec![&l.conv_state, &a.qkv, &l.conv_w, &a.conv],
                                None,
                            ),
                        });
                        d.push(("delta q/k", &k.qk_q, vec![&a.conv, &a.qe], None));
                        d.push(("delta q/k", &k.qk_k, vec![&a.conv, &a.ke], None));
                        d.push((
                            "gates",
                            &k.gates,
                            vec![&a.a, &a.b, &l.amp, &l.dt_bias, &a.g, &a.beta],
                            None,
                        ));
                        d.push(match snap {
                            Some(s) => (
                                "delta step",
                                k.delta_snap.as_ref().expect("checked"),
                                vec![
                                    &l.state,
                                    &a.qe,
                                    &a.ke,
                                    &a.conv,
                                    &a.g,
                                    &a.beta,
                                    &a.y,
                                    &s.rows_state[li],
                                ],
                                None,
                            ),
                            None => (
                                "delta step",
                                &k.delta,
                                vec![&l.state, &a.qe, &a.ke, &a.conv, &a.g, &a.beta, &a.y],
                                None,
                            ),
                        });
                        li += 1;
                        d.push((
                            "gated norm",
                            &k.gated_norm,
                            vec![&a.y, &l.gnorm, &a.z, &a.mixed],
                            None,
                        ));
                        d.push((
                            "matvec out_proj",
                            &k.mv[&l.out.key(true)],
                            l.out.bind(&a.mixed, Some(&a.x), &a.x2),
                            None,
                        ));
                    }
                    Mixer::Attn(at) => {
                        d.push((
                            "matvec q",
                            &k.mv[&at.q.key(false)],
                            at.q.bind(&a.h, None, &a.q32),
                            None,
                        ));
                        d.push((
                            "matvec k/v",
                            &k.mv[&at.k.key(false)],
                            at.k.bind(&a.h, None, &a.k32),
                            None,
                        ));
                        d.push((
                            "matvec k/v",
                            &k.mv[&at.v.key(false)],
                            at.v.bind(&a.h, None, &a.v32),
                            None,
                        ));
                        d.push((
                            "rope",
                            &k.rope_q,
                            vec![&a.q32, &at.q_norm, &a.cos, &a.sin, &a.q16, &a.gate],
                            None,
                        ));
                        d.push((
                            "rope",
                            &k.rope_k,
                            vec![&a.k32, &at.k_norm, &a.cos, &a.sin, &a.k16],
                            None,
                        ));
                        d.push((
                            "kv append",
                            &k.kv_k,
                            vec![&a.k16, &at.kcache, &a.scalars_pos],
                            None,
                        ));
                        d.push((
                            "kv append",
                            &k.kv_v,
                            vec![&a.v32, &at.vcache, &a.scalars_pos],
                            None,
                        ));
                        // The serial causal kernel scans the cache with one
                        // threadgroup per KV head, so a verify of two tokens
                        // went from 40.6 ms at no context to 67.9 ms at 1440
                        // -- 1.84 passes, which is why speculation lost as
                        // context grew. Past the threshold the cache is cut
                        // across threadgroups instead, each query still
                        // masked to its own position, and a second kernel
                        // merges the partials per query row.
                        let nsplit = self.nsplit(self.pos + t);
                        if nsplit >= attn_min_splits() {
                            d.push((
                                "attention",
                                &k.attn_split,
                                vec![
                                    &a.q16,
                                    &at.kcache,
                                    &at.vcache,
                                    &a.part_m,
                                    &a.part_l,
                                    &a.part_acc,
                                    &a.scalars_pos,
                                ],
                                Some([self.cfg.kv_heads, nsplit]),
                            ));
                            d.push((
                                "attention",
                                &k.attn_combine,
                                vec![
                                    &a.part_m,
                                    &a.part_l,
                                    &a.part_acc,
                                    &a.attn,
                                    &a.scalars_nsplit,
                                ],
                                None,
                            ));
                        } else {
                            d.push((
                                "attention",
                                &k.attn,
                                vec![&a.q16, &at.kcache, &at.vcache, &a.attn, &a.scalars_attn],
                                None,
                            ));
                        }
                        d.push(("gate mul", &k.mul, vec![&a.attn, &a.gate, &a.gated], None));
                        d.push((
                            "matvec o_proj",
                            &k.mv[&at.o.key(true)],
                            at.o.bind(&a.gated, Some(&a.x), &a.x2),
                            None,
                        ));
                    }
                }
                let f = &layer.ffn;
                d.push(("rmsnorm", &k.rms, vec![&a.x2, &f.norm, &a.h], None));
                d.push((
                    "matvec gate/up",
                    &k.mv[&f.gate.key(false)],
                    f.gate.bind(&a.h, None, &a.ffn_g),
                    None,
                ));
                d.push((
                    "matvec gate/up",
                    &k.mv[&f.up.key(false)],
                    f.up.bind(&a.h, None, &a.ffn_u),
                    None,
                ));
                d.push((
                    "silu_mul",
                    &k.silu,
                    vec![&a.ffn_g, &a.ffn_u, &a.ffn_a],
                    None,
                ));
                d.push((
                    "matvec down",
                    &k.mv[&f.down.key(true)],
                    f.down.bind(&a.ffn_a, Some(&a.x2), &a.x),
                    None,
                ));
            }
            // Every token's logits: a verify needs them all, and the head
            // is 0.7 GB against the 14.5 the batch has already moved.
            let out = &self.lm_head;
            d.push(("rmsnorm", &k.rms, vec![&a.x, &self.out_norm, &a.h], None));
            d.push((
                "matvec lm head",
                &k.mv[&out.key(false)],
                out.bind(&a.h, None, &a.logits),
                None,
            ));
            d
        }

        /// Every dispatch of one step, in order.
        fn plan(&self) -> Vec<Dispatch<'_>> {
            let (a, k) = (&self.acts, &self.k);
            let mut d: Vec<Dispatch<'_>> = vec![];
            for layer in &self.layers {
                d.push((
                    "rmsnorm",
                    &k.rms,
                    vec![&a.x, layer_norm(&layer.mixer), &a.h],
                    None,
                ));
                match &layer.mixer {
                    Mixer::Linear(l) => {
                        d.push((
                            "matvec qkv",
                            self.mv(&l.qkv, false),
                            l.qkv.bind(&a.h, None, &a.qkv),
                            None,
                        ));
                        d.push((
                            "matvec z",
                            self.mv(&l.z, false),
                            l.z.bind(&a.h, None, &a.z),
                            None,
                        ));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.a, &a.a], None));
                        d.push(("matvec a/b", &k.dense, vec![&a.h, &l.b, &a.b], None));
                        d.push((
                            "conv",
                            &k.conv,
                            vec![&l.conv_state, &a.qkv, &l.conv_w, &a.conv],
                            None,
                        ));
                        d.push(("delta q/k", &k.qk_q, vec![&a.conv, &a.qe], None));
                        d.push(("delta q/k", &k.qk_k, vec![&a.conv, &a.ke], None));
                        d.push((
                            "gates",
                            &k.gates,
                            vec![&a.a, &a.b, &l.amp, &l.dt_bias, &a.g, &a.beta],
                            None,
                        ));
                        d.push((
                            "delta step",
                            &k.delta,
                            vec![&l.state, &a.qe, &a.ke, &a.conv, &a.g, &a.beta, &a.y],
                            None,
                        ));
                        d.push((
                            "gated norm",
                            &k.gated_norm,
                            vec![&a.y, &l.gnorm, &a.z, &a.mixed],
                            None,
                        ));
                        d.push((
                            "matvec out_proj",
                            self.mv(&l.out, true),
                            l.out.bind(&a.mixed, Some(&a.x), &a.x2),
                            None,
                        ));
                    }
                    Mixer::Attn(at) => {
                        d.push((
                            "matvec q",
                            self.mv(&at.q, false),
                            at.q.bind(&a.h, None, &a.q32),
                            None,
                        ));
                        d.push((
                            "matvec k/v",
                            self.mv(&at.k, false),
                            at.k.bind(&a.h, None, &a.k32),
                            None,
                        ));
                        d.push((
                            "matvec k/v",
                            self.mv(&at.v, false),
                            at.v.bind(&a.h, None, &a.v32),
                            None,
                        ));
                        d.push((
                            "rope",
                            &k.rope_q,
                            vec![&a.q32, &at.q_norm, &a.cos, &a.sin, &a.q16, &a.gate],
                            None,
                        ));
                        d.push((
                            "rope",
                            &k.rope_k,
                            vec![&a.k32, &at.k_norm, &a.cos, &a.sin, &a.k16],
                            None,
                        ));
                        d.push((
                            "kv append",
                            &k.kv_k,
                            vec![&a.k16, &at.kcache, &a.scalars_pos],
                            None,
                        ));
                        d.push((
                            "kv append",
                            &k.kv_v,
                            vec![&a.v32, &at.vcache, &a.scalars_pos],
                            None,
                        ));
                        // One threadgroup per KV head scanning the whole
                        // cache is fine while the cache is short, and most
                        // of a step once it is not: 16.6 ms of a 52.16 ms
                        // step at 1440 positions against 0.58 ms at zero.
                        // Past ATTN_MIN_SPLITS the cache is cut into splits
                        // attended in parallel, their partial softmaxes
                        // merged by a second kernel.
                        let nsplit = self.nsplit(self.pos + 1);
                        if nsplit >= attn_min_splits() {
                            d.push((
                                "attention",
                                &k.attn_split,
                                vec![
                                    &a.q16,
                                    &at.kcache,
                                    &at.vcache,
                                    &a.part_m,
                                    &a.part_l,
                                    &a.part_acc,
                                    &a.scalars_len,
                                ],
                                Some([self.cfg.kv_heads, nsplit]),
                            ));
                            d.push((
                                "attention",
                                &k.attn_combine,
                                vec![
                                    &a.part_m,
                                    &a.part_l,
                                    &a.part_acc,
                                    &a.attn,
                                    &a.scalars_nsplit,
                                ],
                                None,
                            ));
                        } else {
                            d.push((
                                "attention",
                                &k.attn,
                                vec![&a.q16, &at.kcache, &at.vcache, &a.attn, &a.scalars_attn],
                                None,
                            ));
                        }
                        d.push(("gate mul", &k.mul, vec![&a.attn, &a.gate, &a.gated], None));
                        d.push((
                            "matvec o_proj",
                            self.mv(&at.o, true),
                            at.o.bind(&a.gated, Some(&a.x), &a.x2),
                            None,
                        ));
                    }
                }
                let f = &layer.ffn;
                d.push(("rmsnorm", &k.rms, vec![&a.x2, &f.norm, &a.h], None));
                d.push((
                    "matvec gate/up",
                    self.mv(&f.gate, false),
                    f.gate.bind(&a.h, None, &a.ffn_g),
                    None,
                ));
                d.push((
                    "matvec gate/up",
                    self.mv(&f.up, false),
                    f.up.bind(&a.h, None, &a.ffn_u),
                    None,
                ));
                d.push((
                    "silu_mul",
                    &k.silu,
                    vec![&a.ffn_g, &a.ffn_u, &a.ffn_a],
                    None,
                ));
                d.push((
                    "matvec down",
                    self.mv(&f.down, true),
                    f.down.bind(&a.ffn_a, Some(&a.x2), &a.x),
                    None,
                ));
            }
            d.push(("rmsnorm", &k.rms, vec![&a.x, &self.out_norm, &a.h], None));
            d.push((
                "matvec lm head",
                self.mv(&self.lm_head, false),
                self.lm_head.bind(&a.h, None, &a.logits),
                None,
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
