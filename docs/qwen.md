# Qwen3.5-27B on tile kernels

The model in daily use here is `qwen3.8:27b-mlx`: Qwen3.5, 27.8B parameters
dense, NVFP4 weights, served by Ollama's MLX engine. It now runs end to end
on kernels this compiler generates.

## Status

| | tile | Ollama (MLX) |
| --- | --- | --- |
| decode | **26.5 tok/s** | 58 (hard text) – 76 (predictable) |
| a 2-token verify | **1.05 passes** | ~1.20 (implied) |
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

38 ms a token, per call site (`examples/qwen`, `TILE_SYNC`):

| call site | ms/token | share |
| --- | --- | --- |
| matvec gate/up | 13.9 | 37% |
| matvec down | 7.5 | 20% |
| matvec qkv (linear layers) | 3.2 | 9% |
| matvec out_proj | 2.4 | 6% |
| matvec z | 2.1 | 6% |
| matvec lm head | 1.5 | 4% |
| rmsnorm | 1.2 | 3% |
| delta step | 0.9 | 2% |
| attention | 0.5 | 1% |
| everything else | < 1.5 | 4% |

Matvecs are 94% of a step. A forward pass reads 14.5 GB of weights, so at
the 455 GB/s the NVFP4 matvec reaches, ~32 ms is the floor; the gap to 38
is the shapes that do not reach it (`k` at 4096 → 1024 runs at 251) and
the 3% that is not a matvec.

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
the loop was short of ALU rather than bandwidth. Four decodes were
measured: a 256-entry pair table (246), a 16-entry table gathered with
`simd_shuffle` (236), scalar arithmetic (203), and two codes of a byte at
once on vector ALU with `(c & 7) << 22` for the IEEE bias and `2t - 1` to
correct the subnormals (367).

None of them needed to exist. Drop a code's three magnitude bits at the
**bottom** of *half*'s exponent field and half reads back the E2M1 value
exactly, times 2⁻¹⁴ — the two subnormals included, because half's denormal
boundary lands where E2M1's does once the exponent sits at the bottom of
the field. There is no bias to add and no subnormal to fix. Two codes 16
bits apart share a shift, so a byte decodes in two shifts, two masks and
an or, and the 2¹⁴ rides back on the group's scale, which is read once per
16 values instead of once per value:

```metal
const uint w = b | (b << 12u);
float2(as_type<half2>(((w & 0x00070007u) << 9u) | ((w & 0x00080008u) << 12u)))
```

That is **455 GB/s** against Q4_K's 477 at the same shape, and it took a
step from 45.9 to 38 ms. The trick is MLX's, found by reading
`mlx/backend/metal/kernels/fp4.h`; a host-side check confirms all 256 byte
values, signed zero included, before trusting it on the GPU.

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
| 1 (the decode matvec) | 454 | 32 |
| 2 | 396 | 18 |
| 4 | 329 | 11 |
| 8 (f16 activations) | 210 | 8.6 |

For a long time this table read 363 / 205 / 183 / 89 and amortisation
stopped at four tokens. Three explanations were measured, and the one
believed — *the loop is short of ALU, and 183 is almost exactly half of
363* — was wrong. The arithmetic was a coincidence. **The batched kernel
was still decoding NVFP4 with the 16-entry `simd_shuffle` table**, the
decode the single-row matvec had abandoned two rounds of work earlier; a
`simd_shuffle` with 32 divergent indices costs about 32 cycles against 2
for a uniform one, and nothing about `(bo, threads)` could reach it. It
went unnoticed because the two paths are separate arms of the emitter and
only the one that was being benchmarked got fixed.

The lesson is narrower than "measure": every one of those measurements was
real. Sweeping the parameters a kernel *exposes* cannot find a constant
factor sitting in the code it *emits*, and a plateau flat across every
parameter is evidence for exactly that.

What remains true from that work: `bo = 256` collapses to 12 GB/s because
each lane holds an accumulator per (row, token), which bounds `bo`; and
half activations pay only at eight tokens, where they are worth 2.4×.

A whole forward pass, end to end (`examples/qwen`):

| tokens | ms/pass | ms/token | vs one step |
| --- | --- | --- | --- |
| 1 | 42.4 | 42.4 | 1.00x |
| 2 | 44.6 | 22.3 | **1.05x** |
| 3 | 53.2 | 17.7 | 1.26x |
| 4 | 67.2 | 16.8 | 1.59x |

A two-token verify costs 1.05 passes, where it cost 1.77. That is the
condition this document set for speculation being worth building, and it
is met.

## Matching Ollama, and beating it

Ollama's decode speed on this model tracks how predictable the text is —
76 tok/s counting, 58 on random words. With 14.5 GB a pass and a roof of
about 507 GB/s (a read-only stream; copy benchmarks land near 460 because
they pay write-allocate), one token per pass cannot exceed ~35 tok/s. So
Ollama is getting **about two tokens per weight pass** from the model's
own multi-token-prediction head.

Ollama's numbers imply its own pass runs at about 480 GB/s and yields two
tokens. Ours now runs at 455 on the shapes that matter and still yields
one. Both halves of the matching problem were the same bug — the decode —
and the first half is essentially closed: 21.8 → 26.5 tok/s, with the
remainder of the gap in the small shapes (`k` at 4096 → 1024 reaches 251
GB/s) rather than in the format.

What is left is tokens per pass. The head ships in the checkpoint (`mtp.*`,
239 MB — 1.6% of a pass, so drafting is nearly free), and `mlx-lm` measures
**88.3% acceptance** for it on this exact model, greedy. At a 2-token
verify costing 1.05 passes that is 1.88 tokens per 1.05 passes — **1.79 a
pass, or about 47 tok/s** — which would put us inside Ollama's range
rather than at a third of it. Two things have to be true and only one is
proven: the verify cost is measured, the acceptance is someone else's
number on someone else's implementation.

Beating it is the same lever used harder, and the research says less about
trees than expected: vLLM closed tree verification as not planned, and
EAGLE's own ablation buys +0.6–0.8 accepted tokens for only +0.3–0.5×.
Depth is the cheaper axis — llama.cpp's adaptive 3..12 draft on this size
of model reports 30 → 56 tok/s on reasoning text and 78 on prose — but
depth needs the verify to stay cheap past two tokens, and ours costs 1.59
passes at four. The next kernel is a narrow `simdgroup_matrix` tile:
`simdgroup_matrix<half, 8, 8>` has a minimum M of 8, so every draft from
2 to 8 tokens is one tile and costs the same.

Kernel efficiency alone tops out near 38 tok/s. Everything above that is
tokens per pass.
