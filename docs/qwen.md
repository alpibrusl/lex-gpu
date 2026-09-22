# Qwen3.5-27B on tile kernels

The model in daily use here is `qwen3.8:27b-mlx`: Qwen3.5, 27.8B parameters
dense, NVFP4 weights, served by Ollama's MLX engine. It now runs end to end
on kernels this compiler generates.

## Status

| | tile | Ollama (MLX) |
| --- | --- | --- |
| decode | **21.8 tok/s** | 58 (hard text) – 76 (predictable) |
| answers | ` Paris` `.` `\n` `The` … | the same tokens |

`tests/qwen_golden.rs` checks 24 steps over three prompts against
`scripts/qwen_ref.py`'s f32 reference: worst |dlogprob| **0.00071**,
tolerance 0.02. The reference in turn agrees with Ollama's own
log-probabilities to 0.10, which is quantised arithmetic, not a bug.

## The shape of the model

Sixty-four layers. Every fourth is grouped attention over a KV cache:
24 query heads of 256, 4 key/value heads, RoPE over the first quarter of
each head, q/k norms, and a sigmoid gate the query projection carries
beside the queries. The other 48 are **gated delta** layers, whose entire
memory is a `[48, 128, 128]` f32 state and a four-position convolution
window — 3 MB a layer, whatever the context length. That is why Ollama's
decode speed barely moves with context on this model, and why ours does
not either.

Every layer ends in the same gated feed-forward (5120 → 17408 → 5120).

One thing that is easy to get wrong and silent when you do: Qwen3.5's
RMSNorm is `x * (1 + w)` and the checkpoint stores `w` as a delta from 1,
for every norm except the gated one inside a linear-attention layer.
`mlx_lm` patches the weights on load, so its module code multiplies by `w`
directly; following that code gives a model that predicts noise while
every weight statistic looks perfectly ordinary.

## Where the time goes

45.9 ms a token, per call site (`examples/qwen`, `TILE_SYNC`):

| call site | ms/token | share |
| --- | --- | --- |
| matvec gate/up | 17.7 | 38% |
| matvec down | 9.4 | 20% |
| matvec qkv (linear layers) | 4.1 | 9% |
| matvec out_proj | 2.8 | 6% |
| matvec z | 2.6 | 6% |
| matvec lm head | 1.9 | 4% |
| delta step | 0.9 | 2% |
| attention | 0.5 | 1% |
| everything else | < 1.5 | 3% |

Matvecs are 95% of a step. A forward pass reads 14.5 GB of weights, so at
the 367 GB/s the NVFP4 matvec reaches, ~40 ms is the floor — which is what
we measure.

## NVFP4, and a measurement that lied

The weights are 4-bit E2M1 codes with an FP8 E4M3 scale per 16 and one f32
scale per tensor. The dequantisation matches MLX's own bit for bit
(`mx.quantize`/`mx.dequantize` round trip, max difference 0.0).

The first bandwidth figures for it — 435 GB/s — were measured against
buffers made with `zeroed()`. Every code is then 0, every table lookup hits
one cache line, and the decode looks free. With real weights the same
kernel ran at 236 GB/s, where Q4_K reaches 484 at the same shape.
`examples/matvec.rs` now fills weights with real bytes.

Swapping the decode for a plain integer convert gave 459 GB/s, which said
the loop was short of ALU rather than bandwidth. Three decodes were
measured: a 256-entry pair table (246), a 16-entry table gathered with
`simd_shuffle` (236), and scalar arithmetic (203). What worked was
decoding both codes of a byte at once on vector ALU: `(c & 7) << 22` lands
E2M1's exponent and mantissa exactly where an IEEE float wants them, so
the bias is a single add, and `2t - 1` corrects the two subnormals. That
is **367 GB/s**, and it took a step from 54.9 to 45.9 ms.

Decoding four values at a time (two bytes into a `float4`) measured 350 —
slower, not faster.

## What a batch costs, and why speculation does not pay yet

`Runner::forward` runs a batch in one pass over the weights, and a test
requires a prompt fed as a batch to land where the same prompt lands token
by token (it does, to 1.9e-5 of scale). The question is what the batch
costs. On 5120 → 17408, reading the weights once per batch
(`examples/matvec`, median of five):

| tokens | GB/s | ms/token for a whole pass |
| --- | --- | --- |
| 1 (the decode matvec) | 363 | 40 |
| 2 | 205 | 35 |
| 4 | 183 | 20 |
| 8 | 89 | 20 |

Amortisation stops at four tokens. Three explanations were measured and
only the last survives:

- **Activation traffic.** Replacing the activation loads with a constant
  doubles throughput, which looked conclusive — but staging them in
  threadgroup memory made it worse (92 GB/s against 141), and raising the
  rows a threadgroup owns, which cuts the number of threadgroups and so
  the traffic, did nothing: every `(bo, threads)` pair that keeps
  registers sane plateaus at 180–183 GB/s.
- **Register pressure.** `bo = 256` collapses to 12 GB/s, and `bo = 64` at
  256 threads is already half speed, because each lane holds one
  accumulator per (row, token). It bounds how large `bo` can go but does
  not explain the plateau.
- **Arithmetic.** At four tokens the loop does one decode and four
  multiply-adds per weight value, and 183 is almost exactly half of the
  matvec's 363. The kernel is short of ALU, as the NVFP4 decode is.

That sets what speculation is worth today. A two-token verify costs 1.77
passes, so drafting one token ahead yields about 1.13 tokens per pass —
13%, before counting a draft that is refused. Speculation is worth
building when a verify of `k` costs near one pass, not `k` of them.

## Matching Ollama, and beating it

Ollama's decode speed on this model tracks how predictable the text is —
76 tok/s counting, 58 on random words. With 14.5 GB a pass and ~460 GB/s
of achievable bandwidth, one token per pass cannot exceed ~32 tok/s. It is
getting **about two tokens per weight pass** from the model's own
multi-token-prediction head, which ships in the checkpoint (`mtp.*`,
239 MB — 1.6% of a pass, so drafting is nearly free).

Ollama's numbers imply its own pass runs at about 480 GB/s and yields two
tokens. Ours runs at 368 and yields one. So matching is two pieces:

1. **The last of the decode gap**, 368 → ~480 GB/s: a pass drops from 40
   to 30 ms, which is ~33 tok/s on its own.
2. **Speculation**, which needs a verify of two tokens to cost about one
   pass rather than the 1.77 it costs now.

Both are the same problem seen twice: these kernels are short of
arithmetic, not bandwidth. The untried lever for both is half precision
for the multiply-add — Apple's f16 ALU runs at twice the f32 rate, the
NVFP4 values are already decoded in half, and MLX and llama.cpp both feed
their matmuls activations narrower than f32. Everything the batched path
needs is built and tested; it is the arithmetic per weight value that has
to come down.

Beating it is the same lever, used harder: acceptance decides how many
tokens a pass yields. A linear draft of one or two tokens is what MLX
appears to do; verifying a small *tree* of candidates in the same pass
accepts more of them. Colibri, which runs much larger MoE models off NVMe,
reports 2.2–2.8 tokens per forward from GLM's MTP head "when it pays" —
evidence that past two is reachable, and that a policy for switching
speculation off when acceptance drops is worth having.

Kernel efficiency alone tops out near 32 tok/s. Everything above that is
tokens per pass.
