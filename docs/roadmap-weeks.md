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

| | lex | Ollama | |
| --- | --- | --- | --- |
| decode, no context | 27.5 (37.4 speculating) | 42.7 | |
| decode, 1440 context | **19.1** (16.7 speculating) | **53.1** | we get worse, they get better |
| prefill, 512 | 75 | 247 | |
| CUDA | three kernels, verified on an L4 | — | no model yet |

The two decode numbers are the story. Ollama is flat with context and we
are not, and our speculation turns into a *loss* at real context lengths.
Both have causes, both are measured, and both are already-solved problems
in this repository.

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

## M2 — the draft head sees the context (days)

`step()` never advances `mtp_pos`, so after a 512-token prompt the model is
at position 512 and the draft head's attention cache is empty. It drafts
from a state the sequence never passed through. Acceptance falls from 90.5%
to 74.6%, and with the step cost rising too, speculation goes from 1.36x to
0.88x — worse than not speculating.

Needs the head run over the prompt as the model is, which means a batched
forward for the head, not only the single-token one it has.

**Done when:** acceptance at 1440 is within a few points of acceptance at
0, and speculation is a win at every context measured.

## M3 — prefill attention (1–2 weeks)

After M1, re-ablate. Today attention is 6.7% of prefill at 128 tokens,
11.6% at 256 and **20.0% at 512** — the quadratic term, and the only share
that grows. The kernel was written for one query against a long cache;
prefill is every query in a chunk against a cache that grows throughout.
`bq = 6, bk = 16` was inherited from decode unchanged.

The feed-forward is 50% of prefill and within ~10% of the 507 GB/s
bandwidth roof, so there is no 3.5x hiding there. Attention is where the
shape is wrong.

**Explicitly not:** a `simdgroup_matrix` GEMM. `examples/gemm_probe` is a
hand-written one against the batched matvec it would replace, and the
matvec wins — 8.29 ms of a notional pass per token against 9.24. It is
committed so the next attempt has to beat that number in an afternoon
rather than in the IR over weeks.

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
