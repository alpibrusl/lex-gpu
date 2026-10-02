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
pub use gpu::{Checkpoint, MAX_BATCH, MAX_DEPTH, Runner, evict_index};

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod gpu {
    use crate::sample::{Nucleus, Sampler};
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
    use lex_msl::delta::DeltaChunk;
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
    /// The deepest a speculative round can draft: a verify of `SPEC_MAX`.
    pub const MAX_DEPTH: usize = SPEC_MAX - 1;

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
    /// Tokens per instance of the batched causal attention: one, or the
    /// largest block up to `LEX_ATTN_TQ` that divides the batch. At a
    /// 128-token prefill chunk on an M4 Max, a layer's attention cost 1.09
    /// ms in blocks of one token, 1.27 in two, 1.46 in four, 4.24 in
    /// sixteen: the finest cut fills the GPU and each block still stops
    /// masking where its token does.
    fn attn_tq(t: usize) -> usize {
        let want = std::env::var("LEX_ATTN_TQ")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1usize);
        (1..=want.min(t))
            .rev()
            .find(|q| t.is_multiple_of(*q))
            .unwrap_or(1)
    }
    /// Whether a batch of `t` attends through the split-KV kernels.
    ///
    /// Splitting the cache is what keeps a verify's few tokens from being
    /// one threadgroup per KV head scanning a long cache. A prefill chunk
    /// is already spread across the GPU by its tokens ([`attn_tq`]), and
    /// splitting it as well made every split carry all of the chunk's query
    /// rows: 512-token prefill spent 329 ms in attention split against 70
    /// in token blocks, and 2048 tokens 3.1 s against 0.92.
    fn split_attn(t: usize, nsplit: usize) -> bool {
        t <= MAX_BATCH && nsplit >= attn_min_splits()
    }
    /// State rows per instance of the delta step.
    const DELTA_ROWS: usize = 8;
    /// The same for a prefill chunk, whose instances each walk 128 tokens
    /// in sequence. Per call at a 128-token chunk on an M4 Max: 4 rows
    /// 2.15 ms, 8 rows 1.70, 16 rows 1.53, 32 rows 2.17.
    const DELTA_ROWS_PREFILL: usize = 16;
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
        all_mats(layers, lm_head, mtp)
            .into_iter()
            .map(|(m, res)| m.key(res))
            .collect()
    }

    /// Every quantised matrix the model multiplies by, with whether its
    /// product goes into the residual: what `mv_keys` keys, and what the
    /// tuner times a shape on.
    fn all_mats<'a>(
        layers: &'a [Layer],
        lm_head: &'a QBuf,
        mtp: Option<&'a Mtp>,
    ) -> Vec<(&'a QBuf, bool)> {
        fn layer<'a>(l: &'a Layer, v: &mut Vec<(&'a QBuf, bool)>) {
            v.push((&l.ffn.gate, false));
            v.push((&l.ffn.up, false));
            v.push((&l.ffn.down, true));
            match &l.mixer {
                Mixer::Linear(x) => {
                    v.push((&x.qkv, false));
                    v.push((&x.z, false));
                    v.push((&x.out, true));
                }
                Mixer::Attn(a) => {
                    v.push((&a.q, false));
                    v.push((&a.k, false));
                    v.push((&a.v, false));
                    v.push((&a.o, true));
                }
            }
        }
        let mut v = vec![(lm_head, false)];
        for l in layers {
            layer(l, &mut v);
        }
        if let Some(m) = mtp {
            v.push((&m.fc, false));
            layer(&m.layer, &mut v);
        }
        v
    }

    /// The few-token NVFP4 matvec for `few`, its layout chosen on this
    /// device (`crate::tune`): every layout `lex_msl::few::layouts` offers,
    /// run on up to four of the model's own matrices of this shape -- so the
    /// weights stream from memory as in a step, not from cache -- and kept
    /// only if its output is the default layout's.
    fn tuned_few(
        gpu: &Gpu,
        tuner: &mut crate::tune::Tuner,
        few: lex_msl::few::Few,
        mats: &[&QBuf],
    ) -> Result<Pipeline, String> {
        use lex_msl::few::{Layout, layouts, matvec_few_nvfp4_with};
        let def = Layout::default();
        let base = matvec_few_nvfp4_with(&few, def)?;
        if tuner.mode() == crate::tune::Mode::Off || mats.is_empty() {
            return gpu.build_lowered(&base);
        }
        let key = format!(
            "few/{}/{}x{}x{}{}{}",
            crate::tune::digest(&base.source),
            few.tokens,
            few.n,
            few.k,
            if few.residual { "r" } else { "" },
            if few.x_half { "h" } else { "" }
        );
        let (t, n, k) = (few.tokens, few.n, few.k);
        let xs: Vec<f32> = (0..t * k)
            .map(|i| ((i * 7) % 13) as f32 * 0.01 - 0.06)
            .collect();
        let x = if few.x_half {
            gpu.upload(&xs.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>())
        } else {
            gpu.upload(&xs)
        };
        let r = gpu.upload(&(0..t * n).map(|i| (i % 5) as f32 * 0.1).collect::<Vec<_>>());
        let y = gpu.zeroed::<f32>(t * n);
        let res = few.residual.then_some(&r);
        let want = {
            let p = gpu.build_lowered(&base)?;
            gpu.run_launches(&[(&p, mats[0].bind(&x, res, &y).as_slice(), None)]);
            let mut v = vec![0.0f32; t * n];
            gpu.download(&y, &mut v);
            v
        };
        let scale = want.iter().fold(1e-6f32, |a, v| a.max(v.abs()));
        let name = |l: &Layout| format!("r{}s{}", l.rows, l.simdgroups);
        let chosen = tuner.choose(&key, &layouts(), def, name, |l| {
            let p = match matvec_few_nvfp4_with(&few, *l) {
                Ok(lw) => gpu.build_lowered(&lw)?,
                Err(_) => return Ok(None),
            };
            gpu.run_launches(&[(&p, mats[0].bind(&x, res, &y).as_slice(), None)]);
            let mut got = vec![0.0f32; t * n];
            gpu.download(&y, &mut got);
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / scale;
            if err > 1e-5 {
                return Ok(None);
            }
            let binds: Vec<Vec<&Buffer>> = (0..8)
                .map(|i| mats[i % mats.len()].bind(&x, res, &y))
                .collect();
            let steps: Vec<Step<'_>> = binds.iter().map(|b| (&p, b.as_slice(), None)).collect();
            gpu.run_launches(&steps);
            let mut best = f64::INFINITY;
            for _ in 0..3 {
                best = best.min(gpu.run_launches(&steps).1 / steps.len() as f64);
            }
            Ok(Some(best))
        })?;
        gpu.build_lowered(&matvec_few_nvfp4_with(&few, chosen)?)
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
        /// `norm` on the host, for the head's own output seeding its next
        /// draft (see [`Runner::head_input`]).
        norm_host: Vec<f32>,
    }

    /// The int8 matvec path (`lex_msl::int8`): its kernels by batch size and
    /// shape, and the quantised input they share.
    ///
    /// On an L4 the float matvec is clock-bound under the 72 W cap -- its
    /// cost is instructions per weight byte -- and this spends about a fifth
    /// as many. It changes the arithmetic (16-bit activations, a scale per
    /// 16 values), so it is CUDA-only and NVFP4-only; on by default,
    /// `LEX_INT8=0` turns it off. It covers decode and verify batches (up to
    /// `SPEC_MAX` tokens): prefill chunks go through the GEMM, and every
    /// shape here is an NVRTC compile at load.
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
    /// Every batched matmul reads a narrowed buffer -- `h`, `ffn_a`, and
    /// `mixed` and `gated` for out_proj and o_proj -- except the draft
    /// head's `fc`, which reads a fused input built on the host: the decode
    /// path's `fc` reads it in f32, the two write the same cache, and handed
    /// f16 it reads the f32 bytes as half pairs.
    fn batch_x_half(c: &Config, n_in: usize, _res: bool) -> bool {
        n_in != 2 * c.hidden && X_DTYPE == DType::F16
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
    /// weights over a prompt, one fewer set of GEMM tails and one fewer
    /// warm of the draft head. 512-token prefill on an M4 Max in chunks of
    /// 128: 204 tok/s; of 256: 218; of 512: 221.
    pub const PREFILL_MAX: usize = 512;

    /// The GEMM chunk sizes. A prompt is cut into these, largest first, and
    /// its last few tokens into one batched-matvec chunk of at most
    /// `MAX_BATCH`: a fixed set of sizes, so the kernels for each are
    /// compiled once at load -- on CUDA each size is seconds of NVRTC, and a
    /// set that grew with every new prompt length would pay them per prompt.
    const GEMM_SIZES: [usize; 6] = [512, 256, 128, 64, 32, 16];

    /// What one prefill pass costs, in rows of compute: the rows it runs,
    /// plus a fixed cost for reading the weights, launching ~700 kernels and
    /// the host's per-pass work. `LEX_PAD_OVERHEAD` sets the fixed part, for
    /// calibration (`examples/qwen_profile --prefill N` over the sizes).
    fn pass_cost(rows: usize) -> usize {
        static OVERHEAD: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        let fixed = *OVERHEAD.get_or_init(|| {
            std::env::var("LEX_PAD_OVERHEAD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32)
        });
        rows + fixed
    }

    /// The pieces of a prefill of `n` tokens at cache position `pos`, as
    /// `(real rows, size)`: exact pieces -- one of `sizes`, or up to
    /// `MAX_BATCH` rows through the batched matvec -- and at most one
    /// padded last piece, whichever way is cheapest by `cost`. A padded
    /// piece must fit under `cap`. `sizes` is descending.
    fn plan_pieces(
        n: usize,
        pos: usize,
        cap: usize,
        sizes: &[usize],
        cost: impl Fn(usize) -> usize,
    ) -> Vec<(usize, usize)> {
        let exact: Vec<usize> = sizes.iter().copied().chain(1..=MAX_BATCH).collect();
        // f[i]: the cheapest way to cover i tokens in exact pieces.
        let mut f = vec![usize::MAX; n + 1];
        let mut from = vec![0usize; n + 1];
        f[0] = 0;
        for i in 1..=n {
            for &j in &exact {
                if j <= i && f[i - j] != usize::MAX {
                    let c = f[i - j] + cost(j);
                    if c < f[i] {
                        f[i] = c;
                        from[i] = j;
                    }
                }
            }
        }
        // The best split into exact pieces and one last piece, which may be
        // padded up to the smallest size that holds it.
        let (mut best, mut best_i, mut best_size) = (usize::MAX, 0, 0);
        for (i, &fi) in f.iter().enumerate() {
            if fi == usize::MAX {
                continue;
            }
            let r = n - i;
            let (c, size) = if r == 0 {
                (fi, 0)
            } else if r <= MAX_BATCH {
                (fi + cost(r), r)
            } else {
                match sizes.iter().rev().find(|&&g| g >= r) {
                    Some(&g) if pos + i + g <= cap => (fi + cost(g), g),
                    _ => continue,
                }
            };
            if c < best || (c == best && size < best_size) {
                (best, best_i, best_size) = (c, i, size);
            }
        }
        let mut pieces = vec![];
        let mut i = best_i;
        while i > 0 {
            pieces.push((from[i], from[i]));
            i -= from[i];
        }
        pieces.sort_unstable_by(|a, b| b.cmp(a));
        if best_i < n {
            pieces.push((n - best_i, best_size));
        }
        debug_assert_eq!(pieces.iter().map(|p| p.0).sum::<usize>(), n);
        pieces
    }

    #[cfg(test)]
    mod plan_tests {
        use super::{GEMM_SIZES, MAX_BATCH, plan_pieces};

        fn plan(n: usize, pos: usize, cap: usize) -> Vec<(usize, usize)> {
            plan_pieces(n, pos, cap, &GEMM_SIZES, |rows| rows + 32)
        }

        #[test]
        fn a_prompt_runs_in_few_passes_and_only_the_last_is_padded() {
            for n in 1..=1500 {
                let p = plan(n, 0, 1 << 20);
                assert_eq!(p.iter().map(|x| x.0).sum::<usize>(), n, "{n}: {p:?}");
                for (i, &(real, size)) in p.iter().enumerate() {
                    assert!(real <= size && real > 0, "{n}: {p:?}");
                    if real < size {
                        assert_eq!(i, p.len() - 1, "{n}: only the last may be padded: {p:?}");
                        assert!(real > MAX_BATCH, "{n}: padded {real} rows: {p:?}");
                        assert!(GEMM_SIZES.contains(&size), "{n}: {p:?}");
                    } else {
                        assert!(
                            GEMM_SIZES.contains(&size) || size <= MAX_BATCH,
                            "{n}: {p:?}"
                        );
                    }
                }
            }
        }

        #[test]
        fn the_shapes_that_cost_the_server_most() {
            // Exactly a size: one pass. A whole chunk and a short tail: the
            // tail padded to the size above it, not cut into five.
            assert_eq!(plan(512, 0, 1 << 20), vec![(512, 512)]);
            assert_eq!(plan(500, 0, 1 << 20), vec![(500, 512)]);
            // 431 was 256 + 128 + 32 + 15 (five passes through the server).
            let p = plan(431, 0, 1 << 20);
            assert!(p.len() <= 3, "{p:?}");
            assert_eq!(p.last().map(|x| x.1), p.last().map(|x| x.1.max(1)));
            // Long prompts: full chunks, then the tail.
            let p = plan(2048 + 300, 0, 1 << 20);
            assert_eq!(&p[..4], &[(512, 512); 4], "{p:?}");
            assert!(p.len() <= 6, "{p:?}");
        }

        #[test]
        fn nothing_pads_past_the_cache() {
            // 500 tokens at position 100 in a cache of 600: a padded 512
            // would write to 612.
            let p = plan(500, 100, 600);
            assert!(p.iter().all(|&(r, s)| r == s), "{p:?}");
            assert_eq!(p.iter().map(|x| x.0).sum::<usize>(), 500);
        }

        #[test]
        fn a_higher_pass_cost_pads_more_and_a_lower_one_cuts_more() {
            let cheap = plan_pieces(431, 0, 1 << 20, &GEMM_SIZES, |r| r + 1);
            let dear = plan_pieces(431, 0, 1 << 20, &GEMM_SIZES, |r| r + 4096);
            assert!(dear.len() <= cheap.len(), "{cheap:?} {dear:?}");
            let waste = |p: &[(usize, usize)]| p.iter().map(|x| x.1 - x.0).sum::<usize>();
            assert!(waste(&dear) >= waste(&cheap), "{cheap:?} {dear:?}");
        }
    }

    /// Every pipeline a batch of `t` tokens dispatches, compiled on first
    /// use of that size.
    /// Which rows of a batch go through `lm_head`.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Head {
        /// Every row: a verify judges each token.
        All,
        /// The last row only, through the decode step's head: what the
        /// next token is sampled from. The batched head over a prefill
        /// chunk computed 128 rows of logits to keep one, 35 ms a chunk
        /// on an M4 Max.
        Last,
        /// None: a prefill chunk before the last, or a replay whose logits
        /// the speculation already has.
        Skip,
    }

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
        /// Split-KV attention, only for a batch that can split (`split_attn`):
        /// unrolled over every token, a prefill chunk's took most of the
        /// minute-plus a new cache size spent compiling, and never ran.
        attn_split: Option<Pipeline>,
        attn_combine: Option<Pipeline>,
        mul: Pipeline,
        silu: Pipeline,
        /// The batch's last row of the residual into the decode step's
        /// `x`, for a head over that row alone ([`Head::Last`]).
        last_row: Pipeline,
        /// The same delta and conv kernels, writing per-token snapshots.
        /// Only for batches a speculative verify can use.
        delta_snap: Option<Pipeline>,
        conv_snap: Option<Pipeline>,
        /// The a/b projections of a CUDA prefill pass, writing no-op gate
        /// inputs for rows past the real count, and the convolution window
        /// rewritten from the last real rows (`lex_msl::pad`): what lets a
        /// pass run on more rows than the prompt has.
        ab: Option<Pipeline>,
        conv_restore: Option<Pipeline>,
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
        /// The rows of a batched pass that are real, when it runs on more
        /// (`forward_pass`): what `lex_msl::pad`'s kernels read.
        scalars_real: Buffer,
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
        /// `out_norm` on the host, for the model's hidden state entering
        /// the draft head (see [`Runner::head_input`]).
        out_norm_host: Vec<f32>,
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
        /// Schedules chosen on this device (`crate::tune`), for the
        /// pipelines `batch` builds after load.
        tune: std::cell::RefCell<crate::tune::Tuner>,
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
                gated_norm: compile(
                    &gpu,
                    &build_gated_norm_rows(1, hv, dv, cfg.eps, DType::F32),
                    128,
                )?,
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
                mul: compile(
                    &gpu,
                    &build_mul(cfg.heads * cfg.head_dim, 256, DType::F32)?,
                    THREADS,
                )?,
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
                    norm_host: store.floats("mtp.norm.weight")?.0,
                })
            } else {
                None
            };

            let (embed, _) = store.floats("model.language_model.embed_tokens.weight")?;
            // Whole splits of the cache, as the split kernel indexes them.
            let splits = cap.div_ceil(ATTN_BK * ATTN_BPS);
            let acts_for = |t: usize, narrow: bool| {
                let f = |n: usize| gpu.zeroed::<f32>(t * n);
                let z = |n: usize| gpu.zeroed::<f32>(t.min(MAX_BATCH) * n);
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
                    // Read back by out_proj and o_proj, every token's row
                    // held in cache at once: f32 at four tokens was 96 KB
                    // of it, and those two matvecs cost 3.5x a step's
                    // (`lex_msl::few`).
                    mixed: if narrow {
                        gpu.zeroed::<f16>(t * hv * dv)
                    } else {
                        f(hv * dv)
                    },
                    q32: f(2 * cfg.heads * cfg.head_dim),
                    k32: f(cfg.kv_heads * cfg.head_dim),
                    v32: f(cfg.kv_heads * cfg.head_dim),
                    q16: gpu.zeroed::<f16>(t * cfg.heads * cfg.head_dim),
                    k16: gpu.zeroed::<f16>(t * cfg.kv_heads * cfg.head_dim),
                    gate: f(cfg.heads * cfg.head_dim),
                    attn: f(cfg.heads * cfg.head_dim),
                    gated: if narrow {
                        gpu.zeroed::<f16>(t * cfg.heads * cfg.head_dim)
                    } else {
                        f(cfg.heads * cfg.head_dim)
                    },
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
                    scalars_real: gpu.zeroed::<u32>(1),
                    scalars_nsplit: gpu.zeroed::<u32>(2),
                    // Split-KV attention's partials. Only a batch of up to
                    // MAX_BATCH ever splits (`split_attn`), so only that
                    // many rows: sized by the batch, at 16k of context and
                    // a 512-token chunk they were 6.4 GB never touched.
                    part_m: z(splits * cfg.heads),
                    part_l: z(splits * cfg.heads),
                    part_acc: z(splits * cfg.heads * cfg.head_dim),
                }
            };
            let acts = acts_for(1, false);
            let bacts = acts_for(PREFILL_MAX, X_DTYPE != DType::F32);
            let out_norm = floats(&gpu, &store, "model.language_model.norm.weight")?;
            let out_norm_host = store.floats("model.language_model.norm.weight")?.0;
            let lm_head = QBuf::load(&gpu, &store, "lm_head.weight")?;
            let gemm_ok = mv_keys(&layers, &lm_head, mtp.as_ref())
                .iter()
                .all(|&(_, _, _, layout)| layout == QLayout::NVFP4);
            let keys = mv_keys(&layers, &lm_head, mtp.as_ref());
            let int8_on = crate::dev::gemm_backend() == lex_msl::gemm::Backend::Cuda
                && gemm_ok
                // Decode on an L4 112 -> 67 ms a token. 8-bit activations
                // moved a golden log-prob by 0.027 against a 0.02 tolerance;
                // 16-bit ones move it 0.00077, and the whole golden suite has
                // passed with them on an L4 twice (2026-09-29). `LEX_INT8=0`
                // keeps the float matvec, to measure one against the other.
                && std::env::var("LEX_INT8").map_or(true, |v| v != "0")
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
            let mut tuner = if crate::dev::gemm_backend() == lex_msl::gemm::Backend::Metal {
                crate::tune::Tuner::open(&gpu.info().name, false)
            } else {
                crate::tune::Tuner::off()
            };
            let mats = all_mats(&layers, &lm_head, mtp.as_ref());
            let mats_of = |key: MvKey| -> Vec<&QBuf> {
                mats.iter()
                    .filter(|(m, r)| m.key(*r) == key)
                    .map(|(m, _)| *m)
                    .take(4)
                    .collect()
            };
            for key in mv_keys(&layers, &lm_head, mtp.as_ref()) {
                if let std::collections::hash_map::Entry::Vacant(slot) = k.mv.entry(key) {
                    let (n_in, n_out, res, layout) = key;
                    // NVFP4 on Metal: the hand-scheduled matvec
                    // (`lex_msl::few` at one token), 6% faster than the
                    // emitted one at the same bytes. `LEX_FEW=0` keeps the
                    // emitted kernel.
                    let few = lex_msl::few::Few {
                        tokens: 1,
                        n: n_out,
                        k: n_in,
                        residual: res,
                        x_half: false,
                    };
                    if crate::dev::gemm_backend() == lex_msl::gemm::Backend::Metal
                        && layout == QLayout::NVFP4
                        && lex_msl::few::fits(&few)
                        && std::env::var("LEX_FEW").map_or(true, |v| v != "0")
                    {
                        slot.insert(tuned_few(&gpu, &mut tuner, few, &mats_of(key))?);
                        continue;
                    }
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
                out_norm_host,
                lm_head,
                mtp,
                mtp_in,
                mtp_in_b,
                mtp_pos: 0,
                snap,
                tune: std::cell::RefCell::new(tuner),
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
            let wall = std::time::Instant::now();
            let (enc, gpu_s) = self.gpu.run_launches(&launches);
            if std::env::var_os("LEX_SPEC_TRACE").is_some() {
                eprintln!(
                    "rollback: {} dispatches, encode {:.2} ms, gpu {:.2} ms, wall {:.2} ms",
                    launches.len(),
                    1e3 * enc,
                    1e3 * gpu_s,
                    1e3 * wall.elapsed().as_secs_f64()
                );
            }
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
            let mark = std::time::Instant::now();
            let (drafts, dists) = self.draft_with(depth, last, Some(&mut *sampler))?;
            let t_draft = mark.elapsed().as_secs_f64() * 1e3;
            if drafts.is_empty() {
                let logits = self.step(last)?;
                return Ok((vec![last], sampler.pick(&logits)));
            }
            self.verify_drafts(last, &drafts, &dists, sampler, None, t_draft)
        }

        /// A speculative round on drafts proposed from outside the model --
        /// the tokens that followed an earlier occurrence of the context's
        /// last few (`crate::spec::lookup`).
        ///
        /// A proposal is a distribution with all its weight on one token,
        /// and [`Sampler::verify_proposal`] takes any distribution, so what
        /// comes out is distributed exactly as without it, sampled or not.
        /// At most [`MAX_DEPTH`] drafts are verified; more are dropped.
        pub fn speculate_proposed(
            &mut self,
            last: u32,
            drafts: &[u32],
            sampler: &mut Sampler,
        ) -> Result<(Vec<u32>, u32), String> {
            let drafts = &drafts[..drafts.len().min(MAX_DEPTH)];
            if drafts.is_empty() {
                let logits = self.step(last)?;
                return Ok((vec![last], sampler.pick(&logits)));
            }
            let dists: Vec<Nucleus> = drafts.iter().map(|&t| (vec![t], vec![1.0])).collect();
            // The hidden state before `last`: the head never drafted from it
            // this round, so its cache has no pair for `last` yet.
            let before = self.spec_h.take().unwrap_or_else(|| self.hidden());
            self.verify_drafts(last, drafts, &dists, sampler, Some(before), 0.0)
        }

        /// Feed `last` and the drafts in one pass, keep the drafts up to the
        /// first the model disagrees with, and undo the rest.
        ///
        /// `unheaded` is the hidden state before `last` when the drafts did
        /// not come from the head, so the head's cache is missing the pair
        /// its own first draft would have written; it is written here with
        /// the accepted rows'.
        fn verify_drafts(
            &mut self,
            last: u32,
            drafts: &[u32],
            dists: &[Nucleus],
            sampler: &mut Sampler,
            unheaded: Option<Vec<f32>>,
            t_draft: f64,
        ) -> Result<(Vec<u32>, u32), String> {
            let trace = std::env::var_os("LEX_SPEC_TRACE").is_some();
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
            fed.extend(drafts);
            let mark = std::time::Instant::now();
            let logits = self.forward_with(&fed, Head::All, true)?;
            let t_verify = mark.elapsed().as_secs_f64() * 1e3;

            // How many drafts survive, longest prefix only. The first
            // rejection also decides what is said in its place, drawn
            // from the target with the rejected draft taken out; if every
            // draft survives, the row after the last one is a free token.
            let mut kept = 0;
            let mut instead = None;
            while kept < drafts.len() {
                let (qi, qp) = &dists[kept];
                match sampler.verify_proposal(&logits[kept], drafts[kept], qi, qp) {
                    Ok(()) => kept += 1,
                    Err(t) => {
                        instead = Some(t);
                        break;
                    }
                }
            }
            let next = instead.unwrap_or_else(|| sampler.pick(&logits[kept]));
            let t_judge = mark.elapsed().as_secs_f64() * 1e3 - t_verify;
            let committed = fed[..=kept].to_vec();
            // Row `kept` of the verify is the state after exactly the tokens
            // being committed, so it is the right input for the next draft
            // whether or not the rest of the pass is undone. Without this the
            // next draft reads whatever the last single step left behind, and
            // guesses from a state the model was never in.
            // Every row's hidden state: row `kept` starts the next draft, and
            // rows `0..kept` rewrite the head's cache below.
            let h = self.cfg.hidden;
            let rows = self.batch_hidden_all(fed.len());
            let carry = rows[kept * h..(kept + 1) * h].to_vec();
            let t_carry = mark.elapsed().as_secs_f64() * 1e3 - t_verify - t_judge;

            if kept < drafts.len() {
                // The pass ran further than the model agreed with, so the
                // delta states carry tokens that were never really said.
                // With snapshots that is a copy; without, the only way
                // back is the pre-batch state and a replay of the prefix.
                if !self.roll_back_to(kept, fed.len()) {
                    self.restore();
                    if kept > 0 {
                        self.forward_with(&committed, Head::Skip, false)?;
                    } else {
                        self.step(last)?;
                    }
                }
            }
            // Keep the head's cache in step with the model's, as prefill
            // does. Drafting advanced it one position per *draft*, accepted
            // or not, and wrote every draft after the first from the head's
            // own guessed hidden state; left alone it drifted from one
            // position behind the model to twenty over 300 tokens, the head
            // drafting from a history the text never had. Cut it back to
            // the first draft's pair -- built from the model's own hidden
            // state, so right -- and rewrite the accepted drafts' pairs from
            // the model's hidden states: row i of the verify with the token
            // after it.
            if self.mtp.is_some() {
                match unheaded {
                    None => {
                        // The round's first token sat at `pos - kept - 1`;
                        // its pair is at the position before that, which
                        // the first draft wrote, so the head resumes here.
                        self.mtp_pos = self.pos - kept - 1;
                        if kept > 0 {
                            self.mtp_warm(&rows[..kept * h], &fed[1..=kept])?;
                        }
                    }
                    Some(before) => {
                        // No draft wrote that pair: one position further
                        // back, with the state before `last` paired with it.
                        self.mtp_pos = self.pos - kept - 2;
                        let mut hs = before;
                        hs.extend_from_slice(&rows[..kept * h]);
                        self.mtp_warm(&hs, &fed[..=kept])?;
                    }
                }
            }
            self.spec_h = Some(carry);
            if trace {
                // `undo` is everything after the verify, split: judging the
                // drafts on the host, fetching the hidden state the next
                // draft starts from, and rolling rejected rows back.
                let after = mark.elapsed().as_secs_f64() * 1e3 - t_verify;
                // The head's cache should end one short of the model's: the
                // pair for the next token is written by the next draft.
                eprintln!("head gap {}", self.pos as i64 - self.mtp_pos as i64);
                eprintln!(
                    "draft {t_draft:.1}  save {t_save:.1}  verify {t_verify:.1}  \
                     undo {after:.1}  kept {kept}  (judge {t_judge:.2} carry {t_carry:.2} rollback {:.2})",
                    after - t_judge - t_carry
                );
            }
            Ok((committed, next))
        }

        /// Does this checkpoint carry a draft head?
        pub fn has_mtp(&self) -> bool {
            self.mtp.is_some()
        }

        /// The draft head's cached keys for positions `0..n`, head by head,
        /// for the tests: a cache entry left unwritten changes the head's
        /// drafts too little to see from outside (it leans on its input far
        /// more than on its cache), so the tests compare the cache itself.
        #[doc(hidden)]
        pub fn head_keys(&self, n: usize) -> Vec<f32> {
            let Some(m) = &self.mtp else { return vec![] };
            let Mixer::Attn(at) = &m.layer.mixer else {
                return vec![];
            };
            let (kh, d, cap) = (self.cfg.kv_heads, self.cfg.head_dim, self.cap);
            let mut all = vec![f16::ZERO; kh * cap * d];
            self.gpu.download(&at.kcache, &mut all);
            let n = n.min(cap);
            (0..kh)
                .flat_map(|h| {
                    all[h * cap * d..(h * cap + n) * d]
                        .iter()
                        .map(|x| x.to_f32())
                })
                .collect()
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
            Ok(self.draft_with(depth, last, None)?.0)
        }

        /// [`Self::draft`], each draft *drawn* from the head's distribution
        /// under `sampler` when there is one, and that distribution (as
        /// [`Sampler::nucleus`] gives it) returned alongside, for
        /// [`Sampler::verify_proposal`]. Without a sampler, or with
        /// `LEX_GREEDY_DRAFT=1`, the head's argmax and a one-point
        /// distribution.
        pub fn draft_with(
            &mut self,
            depth: usize,
            last: u32,
            mut sampler: Option<&mut Sampler>,
        ) -> Result<(Vec<u32>, Vec<Nucleus>), String> {
            if std::env::var_os("LEX_GREEDY_DRAFT").is_some() {
                sampler = None;
            }
            let mut dists = Vec::with_capacity(depth);
            if self.mtp.is_none() || depth == 0 {
                return Ok((vec![], dists));
            }
            let c = self.cfg.clone();
            let mut h = self.spec_h.take().unwrap_or_else(|| self.hidden());
            let mut tok = last;
            let mut out = Vec::with_capacity(depth);
            for i in 0..depth {
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
                let hin = self.head_input(&h, i > 0);
                let m = self.mtp.as_ref().expect("checked");
                rms_into(&hin, &m.pre_h, c.eps, &mut fused[c.hidden..]);
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
                tok = match sampler.as_deref_mut() {
                    Some(s) => {
                        let (idx, p) = s.nucleus(&logits);
                        let t = s.draw(&idx, &p);
                        dists.push((idx, p));
                        t
                    }
                    None => {
                        let t = (0..logits.len())
                            .max_by(|&a, &b| logits[a].total_cmp(&logits[b]))
                            .expect("logits") as u32;
                        dists.push((vec![t], vec![1.0]));
                        t
                    }
                };
                out.push(tok);
                // The head's own hidden state carries the next draft.
                h = self.hidden();
            }
            Ok((out, dists))
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
            // The pieces and the size each runs at: the same when a prompt
            // is cut, a larger size for the last when it is padded.
            let trace = std::env::var_os("LEX_PREFILL_TRACE").is_some();
            for (t, size) in self.prefill_plan(n) {
                let t0 = std::time::Instant::now();
                // Logits for the prompt's last token only: every other row's
                // are never read, and no chunk before the last has it.
                let head = if done + t == n {
                    Head::Last
                } else {
                    Head::Skip
                };
                let out = self.forward_pass(&tokens[done..done + t], size, head, false)?;
                if let Some(last) = out.into_iter().last() {
                    logits = last;
                }
                let hs = self.batch_hidden_all(t);
                let t_pass = t0.elapsed().as_secs_f64() * 1e3;
                // The last prompt position pairs with a token the prompt
                // does not have -- the one the model is about to generate.
                // That row is the first real draft's, so it is left for it.
                let warm = t.min(n - 1 - done);
                let t1 = std::time::Instant::now();
                if self.mtp.is_some() && warm > 0 {
                    self.mtp_warm(&hs, &tokens[done + 1..done + 1 + warm])?;
                }
                if trace {
                    eprintln!(
                        "  prefill piece {t} rows as {size}: pass + hidden {t_pass:.0} ms, \
                         draft-head warm {:.0} ms",
                        t1.elapsed().as_secs_f64() * 1e3
                    );
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

        /// Whether a prompt may be padded up to a compiled size: on CUDA,
        /// where `lex_msl::pad`'s kernels exist. `LEX_PAD_PREFILL=0` cuts
        /// prompts into exact pieces instead, to measure one against the
        /// other.
        fn pad_ok(&self) -> bool {
            self.gemm_ok
                && crate::dev::gemm_backend() == lex_msl::gemm::Backend::Cuda
                && std::env::var_os("LEX_NO_PAD_KERNELS").is_none()
                && std::env::var("LEX_PAD_PREFILL").map_or(true, |v| v != "0")
        }

        /// How a prompt of `n` tokens is run, as `(real rows, size)` pieces
        /// in order; they sum to `n`.
        ///
        /// Each pass reads every weight once, so a prompt in fewer passes is
        /// faster even when the last runs some rows it does not need: the
        /// plan covers it with exact pieces (the compiled GEMM sizes, or up
        /// to `MAX_BATCH` rows through the batched matvec) and at most one
        /// padded last piece, whichever is cheapest by [`pass_cost`]. A
        /// padded piece must fit the cache, as its extra rows write
        /// positions past the prompt.
        pub fn prefill_plan(&self, n: usize) -> Vec<(usize, usize)> {
            if !self.pad_ok() {
                let (mut v, mut left) = (vec![], n);
                while left > 0 {
                    let t = self.chunk(left);
                    v.push((t, t));
                    left -= t;
                }
                return v;
            }
            let limit = self.prefill_limit();
            let sizes: Vec<usize> = GEMM_SIZES.into_iter().filter(|&g| g <= limit).collect();
            plan_pieces(n, self.pos, self.cap, &sizes, pass_cost)
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
                    &self.head_input(&hs[j * c.hidden..(j + 1) * c.hidden], false),
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
            if split_attn(t, nsplit) {
                d.push((
                    "mtp attention",
                    k.attn_split
                        .as_ref()
                        .expect("split_attn: a batch that splits has them"),
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
                    k.attn_combine
                        .as_ref()
                        .expect("split_attn: a batch that splits has them"),
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

        pub fn hidden(&self) -> Vec<f32> {
            let mut h = vec![0.0f32; self.cfg.hidden];
            self.gpu.download(&self.acts.x, &mut h);
            h
        }

        /// The hidden state the draft head is seeded with: `x`, a residual
        /// stream, through the norm that ends its stack -- the model's
        /// final norm for the model's own state, the head's `mtp.norm` for
        /// the head's output feeding its next draft. That is what the
        /// model hands its head (the state its `lm_head` reads), and what
        /// Ollama's runner does; the head's `pre_fc_norm_hidden` then
        /// normalises again, but RMS normalisation takes out the scale and
        /// not the final norm's per-channel weights, so seeding the raw
        /// residual gave the head a differently weighted input than it was
        /// trained on. `LEX_MTP_PRENORM=1` keeps the raw residual, to
        /// measure the difference.
        fn head_input(&self, x: &[f32], from_head: bool) -> Vec<f32> {
            if std::env::var_os("LEX_MTP_PRENORM").is_some() {
                return x.to_vec();
            }
            let w = if from_head {
                &self.mtp.as_ref().expect("a head").norm_host
            } else {
                &self.out_norm_host
            };
            let mut out = vec![0.0f32; x.len()];
            rms_into(x, w, self.cfg.eps, &mut out);
            out
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
            let head = if all { Head::All } else { Head::Last };
            self.forward_with(tokens, head, false)
        }

        /// [`Self::forward`] recording where each gated-delta layer stood
        /// after every token, so a rejected speculative batch can be undone
        /// with a copy instead of a replay. Prefill does not ask for this:
        /// it never rolls back, and the snapshots are 3.1 MB a layer.
        fn forward_with(
            &mut self,
            tokens: &[u32],
            head: Head,
            snap: bool,
        ) -> Result<Vec<Vec<f32>>, String> {
            self.forward_pass(tokens, tokens.len(), head, snap)
        }

        /// One pass over `size` rows of which the first `tokens.len()` are
        /// the prompt's: a prompt padded up to a size the load compiled and
        /// run in one pass over the weights, where cutting it into the sizes
        /// that fit re-reads them once a piece (a 431-token prompt as 256 +
        /// 128 + 32 + 15 prefilled at 260 tok/s through the server, against
        /// 490 for one chunk of 512).
        ///
        /// What makes extra rows harmless is in `lex_msl::pad`: a padded
        /// row repeats the last token, writes a KV position the next real
        /// token overwrites, is seen by no real row (attention is causal),
        /// and leaves the gated-delta state alone -- its gate inputs are a
        /// no-op, and the convolution's window is rewritten from the last
        /// real rows. The position advances by the real rows only, and a
        /// [`Head::Last`] head reads the last real row.
        fn forward_pass(
            &mut self,
            tokens: &[u32],
            size: usize,
            head: Head,
            snap: bool,
        ) -> Result<Vec<Vec<f32>>, String> {
            let (real, t) = (tokens.len(), size);
            let padded = t > real;
            let most = if self.gemm_ok { PREFILL_MAX } else { MAX_BATCH };
            if real == 0 || t > most || t < real {
                return Err(format!("a batch is 1..={most} tokens, not {real} in {t}"));
            }
            if self.pos + t > self.cap {
                return Err(format!("the cache is full ({} positions)", self.cap));
            }
            self.batch(t)?;
            if padded {
                // Only a pass the kernels were built for: a prefill chunk's
                // size, not a verify's, with one head row and no snapshots.
                if snap
                    || head == Head::All
                    || real <= MAX_BATCH
                    || self.batches[&t].conv_restore.is_none()
                {
                    return Err(format!("{real} rows cannot be padded to {t}"));
                }
            }
            let c = self.cfg.clone();
            let pos0 = self.pos;
            for i in 0..t {
                let tok = tokens[i.min(real - 1)];
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
            self.gpu.write(&self.bacts.scalars_real, 0, &[real as u32]);
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

            // A padded pass takes its head row from the host afterwards: the
            // plan's own copies row `t - 1`.
            let plan_head = if padded { Head::Skip } else { head };
            let mut plan = self.batch_plan_with(t, snap, plan_head, padded);
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
            self.pos += real;

            let v = c.vocab;
            if padded && head == Head::Last {
                // The last real row's residual, through the decode step's
                // own norm and head, as `Head::Last` does for an unpadded one.
                let h = c.hidden;
                let mut row = vec![0.0f32; h];
                self.gpu
                    .download_at(&self.bacts.x, (real - 1) * h, &mut row);
                self.gpu.write(&self.acts.x, 0, &row);
                let one = &self.acts;
                let out = &self.lm_head;
                let plan: Vec<Dispatch<'_>> = vec![
                    (
                        "rmsnorm",
                        &self.k.rms,
                        vec![&one.x, &self.out_norm, &one.h],
                        None,
                    ),
                    (
                        "matvec lm head",
                        self.mv(out, false),
                        out.bind(&one.h, None, &one.logits),
                        None,
                    ),
                ];
                self.dispatch(&plan);
                drop(plan);
                let mut last = vec![0.0f32; v];
                self.gpu.download(&self.acts.logits, &mut last);
                return Ok(vec![last]);
            }
            match head {
                Head::All => {
                    let mut flat = vec![0.0f32; t * v];
                    self.gpu.download(&self.bacts.logits, &mut flat);
                    Ok(flat.chunks(v).map(<[f32]>::to_vec).collect())
                }
                Head::Last => {
                    let mut last = vec![0.0f32; v];
                    self.gpu.download(&self.acts.logits, &mut last);
                    Ok(vec![last])
                }
                Head::Skip => Ok(vec![]),
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
            self.tune_report();
            Ok(())
        }

        /// Log what the tuner measured this load, and cache it.
        pub fn tune_report(&self) {
            let mut t = self.tune.borrow_mut();
            if !t.measured.is_empty() {
                let changed: Vec<String> = t
                    .measured
                    .iter()
                    .filter(|(_, c, _)| c != "r2s2")
                    .map(|(k, c, g)| {
                        format!(
                            "{} -> {c} ({:+.1}%)",
                            k.rsplit('/').next().unwrap_or(k),
                            100.0 * g
                        )
                    })
                    .collect();
                eprintln!(
                    "tuned {} shapes on this device, {} changed from the default{}{}",
                    t.measured.len(),
                    changed.len(),
                    if changed.is_empty() { "" } else { ": " },
                    changed.join(", ")
                );
            }
            if t.skipped > 0 {
                eprintln!(
                    "{} shapes past the tuning budget kept their default",
                    t.skipped
                );
            }
            t.save();
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
                .unwrap_or(PREFILL_MAX);
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
                    // Every batched matmul reads a narrowed buffer (see
                    // `batch_x_half`). The draft head's `fc` is the
                    // exception: it reads a
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
                    // A verify's 2-4 tokens on Metal: each weight decoded
                    // once for all of them, in the one-token kernel's
                    // structure (`lex_msl::few`) -- a verify's matvecs cost
                    // what a step's do instead of 1.18x. `LEX_FEW=0` keeps
                    // the batched program, to measure one against the other.
                    let few = lex_msl::few::Few {
                        tokens: t,
                        n: n_out,
                        k: n_in,
                        residual: res,
                        x_half: xt == DType::F16,
                    };
                    if crate::dev::gemm_backend() == lex_msl::gemm::Backend::Metal
                        && layout == QLayout::NVFP4
                        && lex_msl::few::fits(&few)
                        && std::env::var("LEX_FEW").map_or(true, |v| v != "0")
                    {
                        let mats: Vec<&QBuf> =
                            all_mats(&self.layers, &self.lm_head, self.mtp.as_ref())
                                .into_iter()
                                .filter(|(m, r)| m.key(*r) == key)
                                .map(|(m, _)| m)
                                .take(4)
                                .collect();
                        slot.insert(tuned_few(gpu, &mut self.tune.borrow_mut(), few, &mats)?);
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
                rows: if t > MAX_BATCH {
                    DELTA_ROWS_PREFILL
                } else {
                    DELTA_ROWS
                },
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
            // CUDA only, and only for the sizes a prompt is padded up to.
            let (ab, conv_restore) = if crate::dev::gemm_backend() == lex_msl::gemm::Backend::Cuda
                && t > MAX_BATCH
                && lex_msl::pad::ab_fits(t, c.hidden, hv)
                && std::env::var_os("LEX_NO_PAD_KERNELS").is_none()
            {
                (
                    Some(gpu.build_lowered(&lex_msl::pad::ab_rows(
                        t,
                        c.hidden,
                        hv,
                        X_DTYPE == DType::F16,
                    )?)?),
                    Some(gpu.build_lowered(&lex_msl::pad::conv_restore(
                        t,
                        ch,
                        c.conv_kernel - 1,
                    )?)?),
                )
            } else {
                (None, None)
            };
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
                    // A prefill chunk walks every token in each instance:
                    // 128 channels an instance gives it 80 of them rather
                    // than 40 (452 -> 334 us a call at 512 tokens).
                    &build_conv_silu_rows(
                        t,
                        ch,
                        c.conv_kernel,
                        if t > MAX_BATCH { 128 } else { 256 },
                    )?,
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
                delta: {
                    // A prefill chunk solves the recurrence a chunk of
                    // tokens at a time (`lex_msl::delta`), rather than
                    // token by token. `LEX_DELTA_STEPS=1` keeps the step
                    // kernel, to measure one against the other.
                    let dc = DeltaChunk {
                        tokens: t,
                        v_heads: hv,
                        k_dim: dk,
                        v_dim: dv,
                        v_base: c.v_base(),
                        v_width: ch,
                    };
                    if t > MAX_BATCH
                        && lex_msl::delta::fits(&dc)
                        && std::env::var_os("LEX_DELTA_STEPS").is_none()
                    {
                        let l = lex_msl::delta::delta_chunked(&dc, crate::dev::gemm_backend())?;
                        gpu.build_lowered(&l)?
                    } else {
                        compile(gpu, &delta.build_steps(t)?, 128)?
                    }
                },
                gated_norm: compile(gpu, &build_gated_norm_rows(t, hv, dv, c.eps, X_DTYPE), 128)?,
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
                attn: {
                    // A prefill chunk's attention on the matrix units on
                    // Metal (`lex_msl::attn`); verify batches and CUDA keep
                    // the program. `LEX_ATTN_PROGRAM=1` keeps it everywhere.
                    let mma = lex_msl::attn::Causal {
                        tokens: t,
                        kv_heads: c.kv_heads,
                        group: c.heads / c.kv_heads,
                        head_dim: c.head_dim,
                        cap: self.cap,
                    };
                    if t > MAX_BATCH
                        && crate::dev::gemm_backend() == lex_msl::gemm::Backend::Metal
                        && lex_msl::attn::fits(&mma)
                        && std::env::var_os("LEX_ATTN_PROGRAM").is_none()
                    {
                        gpu.build_lowered(&lex_msl::attn::causal_mma(&mma)?)?
                    } else {
                        compile(gpu, &attn.build_causal_blocks(t, attn_tq(t))?, 128)?
                    }
                },
                attn_split: (t <= MAX_BATCH)
                    .then(|| {
                        // A verify's tokens on the matrix units, every token
                        // in one pass over each block (`lex_msl::attn`): the
                        // program's per-token scalar accumulators made a
                        // verify of four cost 1.60 steps at 8000 positions.
                        // `LEX_ATTN_SPLIT_PROGRAM=1` keeps the program.
                        let mma = lex_msl::attn::Causal {
                            tokens: t,
                            kv_heads: c.kv_heads,
                            group: c.heads / c.kv_heads,
                            head_dim: c.head_dim,
                            cap: self.cap,
                        };
                        let span = ATTN_BK * ATTN_BPS;
                        // From two tokens: one fills a quarter of the
                        // tile, and at 8000 positions was 44.8 ms against
                        // the program's 42.3.
                        if t >= 2
                            && crate::dev::gemm_backend() == lex_msl::gemm::Backend::Metal
                            && lex_msl::attn::fits_split(&mma, span)
                            && std::env::var_os("LEX_ATTN_SPLIT_PROGRAM").is_none()
                        {
                            gpu.build_lowered(&lex_msl::attn::causal_mma_split(&mma, span)?)
                        } else {
                            compile(gpu, &attn.build_causal_split(t, ATTN_BPS)?, 128)
                        }
                    })
                    .transpose()?,
                attn_combine: (t <= MAX_BATCH)
                    .then(|| compile(gpu, &attn.build_combine_rows(t, ATTN_BPS)?, 128))
                    .transpose()?,
                mul: compile(
                    gpu,
                    &build_mul(t * c.heads * c.head_dim, 256, X_DTYPE)?,
                    THREADS,
                )?,
                silu: compile(
                    gpu,
                    // Four elements a thread where the size allows: one a
                    // thread read 170 GB/s at a 512-token chunk on an M4
                    // Max (530 us a call), four 560 (162 us).
                    &lex_front::llama::silu_mul(
                        t * c.ffn,
                        if (t * c.ffn).is_multiple_of(4 * THREADS) {
                            4 * THREADS
                        } else {
                            THREADS
                        },
                        X_DTYPE,
                    )?,
                    THREADS,
                )?,
                last_row: compile(
                    gpu,
                    &lex_front::qwen::copy_block(t, 1, c.hidden, t - 1)?,
                    THREADS,
                )?,
                // Only a batch a verify can roll back from, and only when
                // there is a draft head to reject anything in the first
                // place. Prefill runs at MAX_BATCH and skips both.
                delta_snap: (want_snap)
                    .then(|| compile(gpu, &delta.build_steps_snap(t)?, 128))
                    .transpose()?,
                ab,
                conv_restore,
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
        /// `padded`: the pass runs on more rows than the prompt has, so the
        /// convolution's window is rewritten from the last real rows.
        fn batch_plan_with(
            &self,
            t: usize,
            snap: bool,
            head: Head,
            padded: bool,
        ) -> Vec<Dispatch<'_>> {
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
                        match &k.ab {
                            Some(ab) => d.push((
                                "matvec a/b",
                                ab,
                                vec![&a.h, &l.a, &l.b, &a.a, &a.b, &a.scalars_real],
                                None,
                            )),
                            None => {
                                d.push(("matvec a/b", &k.dense, vec![&a.h, &l.a, &a.a], None));
                                d.push(("matvec a/b", &k.dense, vec![&a.h, &l.b, &a.b], None));
                            }
                        }
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
                        if padded {
                            let r = k.conv_restore.as_ref().expect("padded pass: checked");
                            d.push((
                                "conv",
                                r,
                                vec![&l.conv_state, &a.qkv, &a.scalars_real],
                                None,
                            ));
                        }
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
                        if split_attn(t, nsplit) {
                            d.push((
                                "attention",
                                k.attn_split
                                    .as_ref()
                                    .expect("split_attn: a batch that splits has them"),
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
                                k.attn_combine
                                    .as_ref()
                                    .expect("split_attn: a batch that splits has them"),
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
            let out = &self.lm_head;
            match head {
                // Every token's logits: a verify needs them all, and the
                // head is 0.7 GB against the 14.5 the batch has moved.
                Head::All => {
                    d.push(("rmsnorm", &k.rms, vec![&a.x, &self.out_norm, &a.h], None));
                    d.push((
                        "matvec lm head",
                        &k.mv[&out.key(false)],
                        out.bind(&a.h, None, &a.logits),
                        None,
                    ));
                }
                // The residual, not the normed `h`: the batch stores `h` as
                // f16 and the decode step reads it as f32. Normed here by
                // the decode step's own kernel, the row goes through the
                // same arithmetic a step would give it.
                Head::Last => {
                    let one = &self.acts;
                    d.push(("lm head row", &k.last_row, vec![&a.x, &one.x], None));
                    d.push((
                        "rmsnorm",
                        &self.k.rms,
                        vec![&one.x, &self.out_norm, &one.h],
                        None,
                    ));
                    d.push((
                        "matvec lm head",
                        self.mv(out, false),
                        out.bind(&one.h, None, &one.logits),
                        None,
                    ));
                }
                Head::Skip => {}
            }
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
