# The next few weeks

Written at the end of day one, against measurements rather than intentions.
Every number here was taken on an M4 Max unless it says otherwise, and the
things that are guesses are marked as guesses.

## Two models, and why

**Work on: `qwen3.8:27b-mlx`.** It is what gets used here daily, which means
a regression is noticed rather than reported. It is also the harder of the
two: a hybrid where 48 of 64 layers carry a fixed-size recurrent state
instead of a KV cache, NVFP4 weights, and a multi-token-prediction head. If
the compiler can express this it can express most things.

**Hold out: `llama3.1:8b`.** Already supported, already fast (77–80 tok/s
against Ollama's 83–87), and deliberately *not* the thing being optimised.
It is the control. When a change to the shared lowering makes Qwen faster
and Llama slower, that is the change telling on itself — and the Llama
tests run in 12 seconds.

The rule: **tune on Qwen, never tune on Llama, and require Llama not to
regress.** A held-out model is worth more than a held-out benchmark,
because the failure it catches is "we specialised the compiler into a
model" — which is the failure that would make this project pointless.

## Where things actually stand

| | lex | Ollama |
| --- | --- | --- |
| decode, no context | 27.4 (39.5 speculating) | 42.7 |
| decode, 1440 context | 27.0 (15.3 speculating) | 53.1 |
| prefill, 512 | 75 | 247 |
| CUDA | three kernels, verified on an L4 | — |

Decode is flat with context now, which it was not this morning — M1. What
is left is that speculation still *loses* as context grows, and that
prefill is 3.3x off. Both are measured below rather than guessed at.

## M1 — decode stops degrading with context (days)

Ablated: at 1440 positions a step costs 52.16 ms, and **35.55 ms without
the attention call site**. At zero context, 36.13 and 35.55. So attention
goes from 0.58 ms to 16.6 ms — 29x — and is the entire degradation.
Everything else is flat, as it should be when 48 of 64 layers carry a
fixed-size state.

The cause is not subtle: `qwen_run` compiles `attn.build_dynamic()` and
nothing else. The Llama path has `build_split` and `build_combine` and a
`min_splits` threshold — split-KV, the work that made Llama's decode flat
across 0–1440 positions. It was never ported.

**Done when:** a step at 1440 costs what a step at 0 costs, within noise,
and `llama3.1:8b` has not moved.

## M2 — speculation stops losing at context (1–2 weeks)

Two faults were hiding behind each other, and M1 separated them. One is
fixed. The other turned out not to be the fault it looked like.

**The verify did not scale — fixed.** The batch plan compiled
`build_causal` and nothing else: a serial scan of the cache with every
query masked to its own position. `build_causal_split` cuts the cache
across threadgroups and `build_combine_rows` merges the partials per
query row. At 1440 positions:

| t | split | serial | split/step | serial/step |
| --- | --- | --- | --- | --- |
| 2 | 42.6 ms | 68.0 ms | 1.14 | 1.82 |
| 4 | 55.4 ms | 98.0 ms | 1.48 | 2.62 |

The verify of two is now flat against the 40.5 ms it costs at no
context. `examples/verify` measures this against the decode step it
replaces, rather than inferring it from tokens per second — which
divides by an acceptance rate that moves at the same time.

**The draft head not seeing the context was real after all.** The head's
cache only advanced when it drafted, so after a prompt fed with `step` it
sat at position 0 while the model was at 1440. `Runner::prefill` fixes it:
the prompt is fed batched and the head runs over it, which is the only
place it can be done -- the model's hidden state at each position exists
only while the prompt is being fed.

This document previously said the warm bought nothing. That was measured
on 1440 *random* filler tokens, and it is the one text where it cannot
possibly help: a cache full of noise carries nothing for the head to use.
Measured paired, 192 rounds, same context, same positions:

| context | warm | cold | warm-only | cold-only | McNemar p |
| --- | --- | --- | --- | --- | --- |
| coherent 1440 | 71.7% | 64.4% | 16 | 2 | **0.0013** |
| random 1440 | 79.1% | 79.1% | 1 | 1 | 1.0 |

On prose the warm is worth 7.3 points and takes speculation from 0.97x to
1.05x. On noise it is worth nothing, and two discordant pairs out of 191
say so about as flatly as a measurement can.

The pairing is what made this decidable. `actual` is the model's own
greedy continuation and does not depend on the drafts, so two runs over
the same ids predict exactly the same positions -- the script checks that
and refuses to report if it is ever false. Unpaired, the same 191 rounds
sit at 1.5 sigma and say nothing; paired, they are p = 0.0013.

Two older numbers here are withdrawn. Acceptance at 1440 random was
reported as 68-70%; at 192 rounds instead of 47 it is 79.1%. Random
filler does not depress acceptance -- if anything the model's
continuation of noise is *more* predictable than prose. Any acceptance
figure in this repository taken over ~48 rounds should be assumed noisy
to several points.

`scripts/ollama_context.py` is where coherent ids come from: Ollama's
`/api/generate` returns `context`, the token ids of the prompt and its
own reply, so no tokenizer is needed in the repo. It reports the share of
immediate repeats so a degenerate passage can be thrown away (this one:
1440 ids, 628 distinct, 0.0%).

**What is actually left is the undo.** With the verify fixed the trace
at 1440 reads:

    draft 3.5  save 0.8  verify 42.8  undo 18.0 (avg)  -> 1.68 tokens

Every rejection restores the gated-delta state and replays the accepted
prefix — a whole extra pass, 38.6 ms, on 32% of rounds. That is 30% of
the round and the entire remaining gap: without it the same acceptance
gives 35.7 tok/s (1.34x) instead of 28.3 (1.06x).

The fix is not more acceptance, it is a cheaper rollback. The batched
delta kernel already walks the `t` tokens in order; if it wrote its state
after each one, rolling back to row `kept` would be the 0.8 ms copy
`save` already costs instead of a replay.

**Done when:** speculation is a win at 0, 512 and 1440, measured on
coherent context rather than on random tokens.

## M3 — prefill attention — done, by M2's kernel

Attention was 6.7% of prefill at 128 tokens, 11.6% at 256 and 20.0% at
512: the quadratic term, and the only share that grew. Wiring the split
kernel into the batch plan for M2 fixed prefill at the same time, because
prefill *is* the batched path.

| tokens | prefill | attention share |
| --- | --- | --- |
| 128 | 90 tok/s | 3.3% |
| 256 | 90 tok/s | 3.4% |
| 512 | 90 tok/s | 4.0% |

Prefill was 75 tok/s and falling; it is 90 and flat. No separate kernel
was written for it. `bq = 6, bk = 16` is still inherited from decode and
is now not worth touching.

**Explicitly not:** a `simdgroup_matrix` GEMM. `examples/gemm_probe` is a
hand-written one against the batched matvec it would replace, and the
matvec wins — 8.29 ms of a notional pass per token against 9.24. It is
committed so the next attempt has to beat that number in an afternoon
rather than in the IR over weeks.

## M3b — the batched matvec reaches neither roof (1–2 weeks)

90 tok/s against Ollama's ~250. Attention is no longer where it is: the
feed-forward matvecs are 54% of the pass (`gate/up` 33%, `down` 21%) and
everything else is single digits.

`examples/matvec` already measures the shape that matters — Qwen's
`5120 -> 17408` gate/up in NVFP4, batched — and it separates the two
candidates. Best configuration at each batch (f16 activations, `bo=32`):

| tokens | kernel | GB/s | ms/token |
| --- | --- | --- | --- |
| 1 | 122 µs | 410 | 35.3 |
| 2 | 117 µs | 428 | 17.0 |
| 4 | 149 µs | 336 | 10.8 |
| 8 | 232 µs | 217 | 8.4 |

**It is not weight re-reads.** Eight tokens instead of one is 8x the work
for one set of weights and buys 4.2x. Raising `MAX_BATCH` climbs a curve
that is already flattening, so the 2.7x is not sitting there.

**Nor is it honestly "compute-bound".** At eight tokens the kernel does
1.43 GFLOP in 232 µs — **6.2 TFLOP/s against the 15.1 measured for MPS
f16**, while moving 217 GB/s against a 507 GB/s read roof. That is 41% of
one roof and 43% of the other. Arithmetic intensity is 28.4 FLOP/byte and
the machine's ridge is 29.8, so this shape sits *exactly* at the ridge
point and reaches neither side of it. There is ~2.4x in the kernel.

Two things the same table says, worth keeping in view:

- `bo` past 32 collapses at eight tokens — 217 GB/s at 32, 90 at 64, 44 at
  128. That is the register-spill cliff already recorded: tiles must be
  indexed by literals or Metal spills the array to stack.
- f16 activations are worth 2.1x at eight tokens (217 against 104) and
  nothing at two, because 544 threadgroups each re-read every token's
  activations.

So the work is the kernel, and the thing to beat is 6.2 TFLOP/s at the
ridge. The research already gathered points at 2-D register blocking
(activations held in registers across two or more output rows, with
compile-time-constant indices). A `simdgroup_matrix` GEMM was measured and
lost — `examples/gemm_probe`, 8.29 ms against 9.24 — so that is not the
first thing to try.

## M4 — a model runs on CUDA (2–3 weeks)

`lex-cuda` emits, compiles and runs three kernels, verified against the
interpreter on an L4. It does not run a model: no buffer offsets, no
multi-dispatch plan, no `Runner`.

This is the milestone that matters most for the project's actual claim, and
the least for anyone's tokens per second. One typed program, two backends,
one model — until that exists, "no per-target kernel rewrites" is an
argument with one backend behind it.

**Done when:** `llama3.2:1b` produces Ollama's tokens on an L4, from the
same `lex-front` programs the Mac runs.

## M5 — scaling beyond one machine (unscheduled)

Everything measured so far is one model, one Mac, one context regime. The
honest gaps:

- **Bigger than memory.** 27.8B at 4.5 bits is 14.5 GB and fits. A 70B does
  not fit two of these. There is no offload, no paging, no story.
- **MoE.** Qwen3.8 is dense. Routing changes the shape of every matmul and
  the research says quantised MTP collapses on MoE where it is fine on
  dense.
- **Multi-device.** Nothing.
- **Long context.** Measured to 1440. The gated-delta layers should not
  care; after M1 the attention layers should not either. That is a
  prediction and it is cheap to check to 8k.

The first three are design work, not optimisation, and `docs/design.md`
claims answers the code does not have.

## How each of these gets measured

- `examples/ablate` — remove one call site, measure the difference, on the
  normally-scheduled pass. Use this for attribution.
- `LEX_SYNC` — **not** for attribution. It serialises the dispatches,
  inflates a 128-token prefill from 1465 ms to 5068 ms, and makes the
  per-kernel times sum to 210% of the real elapsed time because the kernels
  overlap. It is sound for comparing one kernel against itself across
  configurations, which is what found `matvec down` blowing up at sixteen
  tokens.
- `scripts/ollama_bench.py` — the baseline, re-measured rather than
  remembered. Random words, so the prompt cache cannot skip prefill.
- `scripts/gcp/nvidia_test.sh` — NVIDIA, on a Spot L4, about $0.30 a run,
  and it deletes the machine three different ways.
- The container in `scripts/cuda_check.sh` — everything CUDA except whether
  the numbers are right: builds, links, clippy, NVRTC, ptxas. Five layers
  on the laptop, one in the cloud.
