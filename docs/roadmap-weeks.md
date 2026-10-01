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
| Qwen decode, ctx 0 (speculating) | 42.9 | 42.7 | parity (superseded below) |
| Qwen decode, prose, 2026-09-28, greedy | 37.9 | 58.5 | 65% |
| Qwen decode, prose, 2026-09-28, sampled | 38.3 | 57.1 | 67% |
| MiMo-v2.6 9B decode, 2026-09-28, greedy | 74.6 | 66.9 | 113% |
| Qwen decode, 512, real prose | 34.8 | 45.0 | 77% |
| Qwen decode, 1440, real prose | 34.4 | 39.7 | 87% |
| Qwen prefill, 512 | 90 | ~250 | 36% |
| llama3.1:8b decode, Metal | 77-80 | 83-87 | ~92% |
| llama3.2:1b decode, L4 | 124 | 163 | 76% |
| CUDA | a model, token-identical to Metal | — | |

**The Ollama decode figures here are not the ones this document used to
quote**, which were 56.8 and 53.1. Those came from `ollama_bench.py`,
which prompts with random words -- correct for timing prefill, because it
defeats the prompt cache, and wrong for timing decode on a model that
speculates. What the model writes after noise is more predictable than
prose, its draft head accepts more of it, and the rate that comes back is
of an easier text than anyone runs. Measured both ways on the same
machine:

| context | Ollama, random words | Ollama, real prose |
| --- | --- | --- |
| 512 | 53.5 | 45.0 |
| 1440 | 57.8 | 39.7 |

So the gap at length is 13%, not 35%. Use `--prompt-file` for decode
comparisons and random words for prefill. The same caution applies to
every acceptance figure here: they move several points with content, and
`scripts/ollama_context.py` exists to supply prose rather than noise.

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

**The undo is fixed too.** A rejected round used to restore the
pre-batch state and replay the accepted prefix -- a whole extra pass over
the weights, 38.6 ms against a 42.8 ms verify, on 28% of rounds.

The batched delta and conv kernels now write where they stood after
*every* token (`DeltaNet::build_steps_snap`, `build_conv_silu_rows_snap`),
so the undo is `copy_block` out of those snapshots: two copies a layer.

| | before | after |
| --- | --- | --- |
| verify | 42.8 ms | 44.3 ms |
| undo, on a rejected round | 38.6 ms | 2.8 ms |
| round | 57.9 ms | 49.6 ms |

The verify pays 1.5 ms for the snapshot writes on every round and the
undo saves 36 ms on a third of them. Snapshots are kept for batches up to
`SPEC_MAX = 4` and only when the checkpoint has a draft head; prefill runs
at `MAX_BATCH` and never rolls back, so it does not pay this. The memory
is 604 MB, 4% on top of a 14.5 GB model.

**M2 is met.** Speculation is a win at all three contexts, on coherent
text, and flat from 512 to 1440:

| context | acceptance | plain | speculating | Ollama |
| --- | --- | --- | --- | --- |
| 0, real prompt | 93.7% | 27.1 | **42.9** (1.58x) | 42.7 |
| 512, coherent | 66.3% | 26.9 | **34.8** (1.29x) | 56.8 |
| 1440, coherent | 70.5% | 26.4 | **34.0** (1.28x) | 53.1 |

At no context this is parity with Ollama. At length it is 61-64% of it,
and the gap is now acceptance rather than anything in the schedule: the
verify is flat, the undo is nearly free, and 70% acceptance on prose at
1440 against 94% on a short prompt is what is left to explain.

A caution on the testing. `speculation_lands_exactly_where_greedy_lands`
runs on a short easy prompt where every draft is accepted, so it passes
with the rollback deleted outright -- it was never exercising the undo.
`speculation_survives_rejected_drafts` uses a prompt the head misses on,
and fails if no draft is rejected, because a run with no rejection proves
nothing and must not look like a pass.

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

## M3b — the batched matvec has less room than it looked (revised)

An earlier version of this section said the kernel sat at the roofline
ridge reaching 41% of one roof and 43% of the other, with ~2.4x in it.
**That is withdrawn.** It counted weight bytes only. At eight tokens the
kernel also issues 356 MB of activation loads against 50 MB of weights,
so it moves about 406 MB in 232 µs -- 1,750 GB/s, three and a half times
the DRAM roof, i.e. served from cache. The kernel is much closer to its
limit than a weights-only roofline suggests.

Two hypotheses were tested against that and both failed.

**Cross-threadgroup redundancy is not the cost.** Every simdgroup in a
threadgroup reads the same activation vector, so halving the threadgroup
count should halve those loads. Holding rows per simdgroup at 4 and
raising both `threads` and `bo` does exactly that, and it gets *slower*:

| threads | bo | rows/simdgroup | GB/s |
| --- | --- | --- | --- |
| 256 | 32 | 4 | **214** |
| 512 | 64 | 4 | 188 |
| 1024 | 128 | 4 | 159 |

Those loads are cache-served and cheap; the larger threadgroups cost more
in occupancy than the saved traffic is worth.

**What binds is rows per simdgroup**, at fixed threads: r=4 gives 214
GB/s, r=8 gives 90, r=16 gives 44. `r` wants to be larger -- it amortises
each activation load over more output rows -- and cannot be, because the
accumulators are `r x tokens` and 8x8 spills. At r=4 that is 32
accumulators plus 16 for the staged activations, which is about right for
the register file. The `bo` cliff is real and not an emitter artifact:
the emitted MSL indexes every register array by a literal, which was
checked.

Threadgroup staging of the activations was tried before this and lost (92
GB/s against 141, because it needs a chunked reduction). Those numbers
predate a kernel that is now 2.4x faster, so the result is stale rather
than wrong, but it is not the obvious win either.

So there is no easy multiple in *this* kernel at eight tokens. But eight
tokens is the wrong question for prefill.

## M3c — prefill wants a GEMM, and the probe that said otherwise was broken

Prefill reads the weights once per chunk. At `MAX_BATCH = 8` a 512-token
prompt reads 14.5 GB **sixty-four times**, about 928 GB on top of the
2.08 s of arithmetic the pass actually needs. Ollama prefills in large
batches and pays only the arithmetic: 28.5 TFLOP at the measured 13.7
TFLOP/s is 246 tok/s, which is what it gets. We pay both and get 90.

Raising the chunk with the current kernel is not possible -- `r x tokens`
accumulators spill and the matvec collapses to 258 ms/token past 16. The
kernel for large batches is a tiled GEMM on the matrix units, and
`examples/gemm_probe` was supposed to have ruled that out.

**It had not.** The probe fixed `BM = 32` and gave its grid no token
dimension, so it could not compute more than 32 tokens; asking for 64 gave
a 32-token result with 64 in the denominator. Three token counts all took
about 1,040 µs, which is the tell. It now asserts both kernels wrote every
row, because two kernels that stop early agree perfectly.

| tokens | batched matvec | simdgroup_matrix |
| --- | --- | --- |
| 8 | **8.49** | 28.32 |
| 16 | 20.00 | 15.01 |
| 32 | 258.32 | 9.98 |
| 64 | 258.57 | 7.99 |
| 128 | 258.76 | **6.95** |

The crossover is 16 tokens, and at 128 the GEMM beats the matvec's
best-ever point by 18%. That is a real result and a thin reason to spend
weeks on matrix fragments in the IR.

What makes it worth pursuing is the gap above it. 6.95 ms/token is 144
tok/s of a notional pass against the 246 the arithmetic allows: this
kernel reaches 59% of the compute roof, and it stages through threadgroup
memory with no double buffering and no vectorised loads.

That afternoon has been spent on the configuration knobs, and they are
exhausted. At 128 tokens, against 6.95 at `BN=32, BK=32`: `BK=64` gives
7.46, `BN=64` gives 7.06, `BN=128` gives 10.61. `BK=64` was meant to fix
the weight staging, where half the threadgroup stands idle; the extra
threadgroup memory costs more in occupancy than the idle threads do.
`BN` halves the activation re-reads and changes nothing -- **the same
answer the batched matvec gave to the same question.**

That is worth stating plainly, because it went the same way five times
today: in both kernels, reducing the number of threadgroups to cut
redundant activation traffic does nothing or hurts. Those loads are
cache-served. The thing that moves either kernel is register pressure and
occupancy, and the roofline arithmetic that kept predicting otherwise was
counting DRAM traffic that never happens.

So the GEMM sits at 3.09 ms against a 1.67 ms compute floor -- 54% of
peak -- and the remaining work is the shape of the loop: double buffering
so the dequantise of the next tile overlaps the matrix ops on this one,
and bulk loads. That is kernel work, not a sweep, and it should be done
in `gemm_probe` before any of it reaches the IR. **The bar is 6.95, and
five configuration hypotheses have already died against it.**

## M4 — a model runs on CUDA — done, and so does the hard one

`qwen3.8:27b-mlx` -- 48 of 64 layers carrying a recurrent state, NVFP4
weights, a multi-token-prediction head -- runs on an NVIDIA L4 and passes
the whole golden suite there, against the same f32 reference the Metal
tests are held to:

```text
24 steps over 3 prompts on NVIDIA L4: worst |dlogprob| 0.00064  (tol 0.02)
batch vs sequence                      2.95e-4 of scale
prefill vs stepping                    3.27e-4
split vs serial at 1200 positions      8.0e-5
batch split vs serial                  0e0
MTP acceptance                         87.1%
```

The port was three lines -- the cfg gate, the device import, and the
dialect -- and then the module type-checked against CUDA unchanged. All
109 kernels it builds lowered for CUDA and compiled under NVRTC first
try. The part that looked hard, a 4-bit format with a per-16 scale on a
card with no hardware for it, was never in question: the NVFP4 matvec has
been passing against the interpreter on a real L4 since the backend
existed.

Ollama cannot pull this model on Linux -- "this model requires MLX
support, but the MLX runtime is not available" -- but that is the
client's check, not the registry's, and `lex-rt` reads the store rather
than asking Ollama to run anything. `scripts/ollama_fetch.py` takes the
blobs over plain HTTP. There is consequently **no like-for-like Ollama
baseline for Qwen on NVIDIA**: the non-MLX `qwen3.8:27b` is a 16.8 GB
GGUF against our 14.5 GB NVFP4, a different quantisation of the same
weights, so any comparison would be of two different models.

**Speed: 9.1 tok/s, up from 6.4, against 26.3 on an M4 Max.**

| | tok/s | GB/s | of its read roof |
| --- | --- | --- | --- |
| M4 Max (~507 GB/s) | 26.3 | 381 | 75% |
| L4 (~300 GB/s) | 9.1 | 132 | 44% |

The 42% came from one number in the target table. The decode matvec
gives a threadgroup `bo` output rows and splits them across its
simdgroups; `bo = 8` at 256 threads is one row per simdgroup, which is
Apple's optimum and which CUDA inherited. Sweeping both knobs on an L4
shows the ratio is what predicts the bandwidth:

| rows per warp | configurations | GB/s |
| --- | --- | --- |
| 1 | 256/8, 128/4 | 107, 109 |
| **2** | **256/16, 128/8** | **189, 180** |
| 4 | 128/16, 256/32 | 170, 166 |
| 8 | 128/32, 256/64, 128/64 | 103, 105, 104 |

`Target::matvec_rows_per_simd` now carries it: 1 for Apple, 2 for
NVIDIA. Apple's `1 * (256/32) = 8` is the constant that was already
there, so Metal is unchanged by construction.

**Two things went the other way, and both are honest costs.**

`llama3.2:1b` on the same card went 124.4 to 116.9 tok/s, −6%. Sweeping
both models' shapes over the same ten configurations says why, and says
the target table is the wrong home for this:

| rows per warp | qwen `5120 -> 17408` | 1b `2048 -> 8192` |
| --- | --- | --- |
| 0.5 | 105 | **243** |
| 1 | 107, 112 | **241, 242** |
| 2 | **181, 182** | 225, 226 |
| 4 | 169, 166 | 181 |
| 8 | 103, 105 | 104, 100 |

Rows per warp is the right metric for both — it predicts the number
where neither knob does alone — but they peak in different places, and
no rule fits both. Threadgroup count does not: Qwen at 2,176 is bad
(107) where the 1B at 2,048 is its best (243). Bytes per threadgroup
points the opposite way for the two. Values per warp is not constant.
Two shapes do not determine a rule, and one fitted to two points would
be a guess wearing a formula.

**So `bo` should be searched, not tabulated.** `docs/design.md` already
says this — "in P0 the planner computes it from two constants; in P3 it
searches for it" — and this is the first measurement that makes the
case concrete rather than architectural. Timing two or three candidates
per distinct matvec shape at load costs a second and would take Qwen's
+42% *and* keep the 1B's 124, on any card, without a constant to be
wrong about.

Until then the target value stays at 2, which is right for the model in
daily use and wrong by 6% for the 1B.

~~Speculation on CUDA is now a loss: 6.2 tok/s against 9.1 plain.~~
**Retracted 2026-09-28: that was the measurement, not the machine.** A
batch size's kernels compile the first time it is used, and on CUDA that
is an NVRTC compile of every kernel -- seconds, inside a timed loop of a
few seconds. `examples/mtp` timed the first speculation, so it timed the
compile. Warmed, on the same L4:

| | tok/s |
| --- | --- |
| plain decode | 9.6 |
| speculating, depth 1 (87.1% accepted) | **21.6 (2.25x)** |

`examples/qwen_profile` (events between launches, which sum to the
untimed step within 1%) says why it can beat 2x at depth 1: a verify of
*three* tokens costs 0.97 of a plain step (101.4 against 105.0 ms at
context 0; 108.7 against 112.2 at 1024). The batched path's matvec is
faster than the decode path's even though it does three times the
arithmetic.

**Where the decode step's time actually goes** on the L4, from the same
profile: matvecs are 92% of it, and everything else -- norms, the
recurrence, attention, rope, the KV append -- is about 5 ms of 106. So
streams and events, the other candidate, could buy at most about 5%. The
matvecs run at 140-156 GB/s inside the model, half the card's ~300. The
189 GB/s above came from `examples/matvec`, which repeats one 50 MB
matrix on a card with 48 MB of L2, and so flattered it.

So the next CUDA work is the decode matvec: first, whether the batched
kernel at one token already beats it, since at three tokens it does.

**It does, a little, and not on Metal** (`qwen_profile`'s fourth pass,
decoding the same positions through `forward(&[t])`, warmed):

| | step | batch path | |
| --- | --- | --- | --- |
| L4, context 0 | 103.9 ms | 95.5 ms | 1.09x |
| L4, context 1024 | 107.9 ms | 102.0 ms | 1.06x |
| M4 Max, context 0 | 39.7 ms | 44.3 ms | 0.90x |

The difference is the matvec (gate/up 304 against 324 us a call), and
both sit near 150-165 GB/s of ~300 -- so this is not the fix, and `step`
stays as it is: on CUDA the server decodes Qwen through speculation, which
already runs the batch path, and rerouting `step` would move the draft
head's hidden state between buffers for a few percent on the path that
runs least. The fix is a CUDA matvec that reads at 80% of the roof, which
is where the remaining 2x is. (Speculation reproduced on a second L4, in
another zone: 21.8 against 9.7, 2.24x.)

## M5a — prefill on the matrix units, and why the L4's decode is slow (2026-09-28)

**Prefill.** The batched matvec keeps an accumulator per (row, token) in
registers and spills past ~16 tokens, so prefill ran in chunks of 8 and a
512-token prompt read the 14.5 GB of weights 64 times. `lex_msl::gemm` is
a tiled GEMM on the matrix units -- `wmma` on CUDA, `simdgroup_matrix` on
Metal -- dequantising NVFP4 to half as it stages. It is hand-scheduled,
not lowered from a program (the IR has `MatMulNT`, not fragments). Qwen's
runner cuts prompts into 128/64/32/16-token GEMM chunks and a tail of at
most 8 through the matvec, all compiled at load.

| 512-token prefill | chunk 8 | chunk 64 | chunk 128 |
| --- | --- | --- | --- |
| M4 Max | 87.4 | 114 | **123** |
| NVIDIA L4 | 39.6 | 172.8 | **199.9** |

On Metal the tile is 32 tokens by 64 rows in one shared buffer: taller
token tiles (which dequantise each weight fewer times) were *slower*,
wider weight tiles faster (32x32 100, 64x32 82, 128x32 67, 64x64 101,
32x64 123, 32x128 120 tok/s). The first L4 numbers said 64-token chunks
ran at 11.2 tok/s; the kernels summed to 2.7 s of 45.7. The draft head's
warm-up asked for batches of 63 and 127 rows, sizes nothing had compiled,
and NVRTC compiled a whole batch set inside the request. It now walks the
precompiled sizes, and a golden fails if a prefill compiles anything.

**The L4's decode matvec is clock-bound, not bandwidth-bound.**
`examples/mv_variants` times the emitted kernel alone at the gate/up shape:
237 us, 212 GB/s, against 258 GB/s for a plain read of the same bytes --
82%. Inside the model the same kernel takes 351 us. Not the working set:
cycling 3.2 GB instead of 200 MB costs 2% (208 GB/s). The clocks: through
the model runs the card sits at its 72 W cap (reason 0x4 in 347 of 1491
samples, 75 W peak) and the SM clock drops to a median 1395 MHz from 2040,
while the memory clock stays at 6251. 2040/1395 = 1.46; 351/237 = 1.48.

So on this card the matvec's cost is instructions per weight byte, paid
in power: widening its loads changed nothing (the load count was not the
limit), and restructuring it gains 11% at full clock (one warp a row, 236
GB/s) and loses it over a large working set (185 GB/s at 64 matrices).
The lever is fewer instructions per byte: decode E2M1 through a table to
int8 and multiply with `dp4a` against int8-quantised activations, as
llama.cpp does for its four-bit formats -- which changes the arithmetic,
so it is a golden-suite question as much as a speed one.

## M5b — where Metal's Qwen decode goes (2026-09-29, GPU idle)

Plain decode is at the memory roof: 37.0 ms a token, gate/up reading
~460 GB/s of a 463 GB/s copy, so a one-token step has nowhere to go.
Ollama's 57 tok/s is speculation. Ours, traced over 143 greedy cycles:

| per cycle | ms |
| --- | --- |
| draft (2 tokens) | 5.3 |
| verify (3 tokens) | 48.7 |
| undo | 2.6 |
| total, 2.09 tokens committed | 55.7 -> 37.5 tok/s |

At this acceptance, 57 tok/s is a 36 ms cycle -- about one plain step.
The verify is the lever: 1.28 steps for three tokens. `LEX_BATCH_BO` is
not (16, 32, 64 all 1.28). Summing each NVFP4 run unscaled and scaling
once, as the single-row kernel already did, took it to 1.21 (47.3 -> 45.0
ms). Through the servers (`scripts/serve_bench.py`): greedy 39.4 against
Ollama's 57.1, sampled 42.4 against 53.4. What remains is the verify's
other 0.2 steps (the recurrence and attention over three tokens cost 2.7x
and 2.5x a step's), undo, and acceptance -- a third draft is accepted
often enough to try again now that the verify is cheaper.

## M5c — Metal prefill, 120 -> 203 tok/s (2026-09-29)

512 tokens on an M4 Max, 128-token chunks. Each step was found by the
per-call-site profile (`examples/qwen_profile`) and checked by the golden
suite before the next; Ollama is 250-260 on the same machine
(`scripts/ollama_bench.py`, and one 520-token prompt traced).

| change | where it went | tok/s |
| --- | --- | --- |
| start | | 120.7 |
| vocabulary projection for the prompt's last token only | 138 -> 1.5 ms | 126 |
| GEMM: next K step's loads in flight, vector loads, 64-token tiles | gate/up 2.65 -> 1.97 ms a call | 164 |
| KV append on a grid (was one instance, every row in turn) | 578 -> 6 us a call | 166 |
| causal attention over tokens, not split-KV, for prefill | 329 -> 70 ms | 187 |
| 16 delta rows per instance for prefill | 1.70 -> 1.53 ms a call | 189 |
| gated delta rule a 16-token chunk at a time (`lex_msl::delta`) | 295 -> 92 ms | 202 |
| q/k norm, gates, dense a/b spread over tokens | 26 + 20 + 39 -> 3 + 3 + 33 ms | 203 |

What is left, measured rather than guessed:

- **The GEMM is 8-10% behind MLX.** Compared the same way -- independent
  outputs, so the concurrent encoder may overlap dispatches, which is how
  MLX's own benchmark runs (`examples/gemm_metal --overlap`) -- ours does
  12.7 TFLOPS on the model's shapes against `mx.quantized_matmul`'s 13.9.
  With loads and staging removed the multiply loop alone reaches 14.3;
  the global loads are hidden (removing them changes nothing), and the
  rest is staging the tiles in front of a barrier. Not the fix, each
  measured: row padding, the untransposed load, float fragments, half
  arithmetic in the dequantise, more simdgroups, double-buffered tiles
  (slower), taller or wider tiles (a simdgroup holding more than 32x32
  spills, ten times slower).
- **The GPU is busy 97% of the prefill** (4 chunks of ~610 ms GPU time in
  2.52 s, encoding 0.5 ms a chunk): what is left is GPU work, not the CPU.
- **Warming the draft head costs ~19 ms a chunk**: a download, a host-side
  norm, an upload and the head's layer, serially. On the GPU and
  overlapped with the next chunk it would be most of that back.
- **CUDA gained too.** The attention, KV-append, last-row head and
  small-kernel changes are shared: 512-token prefill on an L4 went 199 ->
  234 tok/s, and all 11 Qwen goldens pass there with the int16 matvec
  (`results/gcp/20260929-202622-l4`). The chunked delta kernel has a CUDA
  form too, its products in f32 on the ordinary cores (the matrix units
  take f32 only as TF32): 245 tok/s against 227 with the step kernel in the
  same run, goldens 11/11, and decode 66 ms a token with the int16 matvec
  now on by default (`results/gcp/20260930-081300-l4`).
- Ollama's kernels, from the trace: MLX's `nvfp4_qmm_t` in bfloat16, a
  fused chunked gated-delta kernel with chunks of 16, attention as GEMMs
  plus a softmax, the convolution fused with its SiLU. The same shape of
  solution; the remaining difference is in execution.

The simdgroup_matrix 8x8 lane layout, measured by loading a known matrix:
lane L holds row `(L%8)/2 + 4*(L/16)`, columns `2*(L%2) + 4*((L/8)%2)` and
the next. Reading `thread_elements()` works; writing its elements one at a
time made a kernel twelve times slower on this SDK.

## M5d — Metal decode: 0.93x, and a benchmark that compared two prompts (2026-09-30)

Qwen3.8 through both servers on prose, same prompt token for token:
**greedy 52.9 against Ollama's 56.7 (0.93x), sampled 47.2 against 54.0
(0.87x).** What moved it, and what was wrong with the measurement:

| change | effect |
| --- | --- |
| `lex_msl::few`: a verify's 2-4 tokens read each weight once, in the one-token kernel's structure | verify of 3: 1.20 -> 1.08 steps |
| the draft head seeded with the *normalised* hidden state, as the model hands it (Ollama does the same) | tokens a verify accepts 2.63 -> 2.79 of 3 |
| the same kernel at one token for decode | a step 36.25 -> 35.10 ms (Ollama 35) |
| `--depth auto` (`lex_rt::spec`) | no change on this Mac; picks per machine |

**The benchmark was comparing different prompts.** lex renders the
checkpoint's chat template, which opens with a "Reasoning effort is set to
xhigh" system turn by default; the Ollama measured against (0.34.4) renders
none -- 40 prompt tokens against 82. Greedy decoding then wrote different
text: under xhigh, terse notes ("Need produce final essay. Need likely no
need web."); without, an outline ("The user wants a detailed ... Let me
structure this"). A draft head predicts the second far better: Ollama's
first draft was accepted 0.90-0.94 of the time, lex's 0.75 -- which looked
like a defect in lex's head until the prompts were printed. Every
through-the-server Qwen comparison before this (the 0.65-0.80x figures) had
the same mismatch. `scripts/serve_bench.py` now asks lex for
`reasoning_effort: medium`, which renders exactly Ollama's prompt, and
prints both prompt token counts. `examples/mtp --prompt` measures
acceptance on a chat-rendered prompt.

## M5e — the best engine on each machine as the bar; Mac prefill 229 (2026-09-30)

**Who to beat.** The same measurement (`scripts/engine_bench.py`: decode
by the difference of a 1- and a 257-token request, prefill on fresh
random prompts, prompt tokens printed) against every engine that runs
Qwen3.8-27B on the machine:

| M4 Max | decode greedy | decode sampled | prefill (~430 tok) |
| --- | --- | --- | --- |
| Ollama 0.34.4 (MLX, draft head) | **57.5** | **56.4** | **236** |
| mlx-lm 0.31.3, mlx-community 4-bit | 29.5 | 29.5 | 228 |
| lex | 52.3 | 50.6 | 186 -> 198* |

\* through the server, before and after the chunking, attention and
elementwise changes below (431-token prompts cut 256 + 128 + 32 + 15);
the profiler's 512-token prefill, one chunk, went 203 -> 229. mlx-lm drafts only with a
separate draft model, so it decodes plainly; Ollama is the bar on the Mac
for both. On NVIDIA the bar is vLLM with NVIDIA's own NVFP4 checkpoint
(Ollama cannot run the MLX build there): `scripts/gcp/bench_engines.sh`.

**Prefill, 512 tokens, 203 -> 229 tok/s; 2048 tokens 177 -> 221:**

| change | effect |
| --- | --- |
| chunks of up to 512 tokens (were 128); split-KV scratch sized by MAX_BATCH, not the chunk | 204 -> 221; 1.6-6.4 GB of scratch never touched, gone |
| causal attention on the matrix units (`lex_msl::attn`) | attention 97 -> 25 ms at 512, 949 -> 354 at 2048 |
| silu_mul four elements a thread; the convolution in 128-channel pieces | 530 -> 162 us, 452 -> 334 us a call |

Tried and dropped, measured: cutting prompts at any size on Metal. A
never-built size took one to two minutes to compile -- the convolution
and split-KV attention unroll over every token -- and the 0.1 s the
server reported was Metal's shader cache. (The delta kernel keeps its new
support for a ragged last chunk.) Taller GEMM tiles at 512 tokens: no
gain; the GEMM is ~13 TFLOPS in the model, ~90% of its own multiply-only
loop and ~93% of MLX's.

## M5f — a verify's inputs, not its weights: Mac decode 0.97x (2026-09-30)

Traced through the server (`LEX_SPEC_TRACE=1`, 211 cycles at depth 2):
a cycle was 48.2 ms for 2.52 tokens -- draft 5.0, verify 41.7, undo
1.5 -- and the verify of three cost 1.15 steps, of four 1.43, so the
controller almost never drafted three. Per call site (`qwen_profile
--verify 4`), gate/up and down barely grew with the tokens, but out_proj
and o_proj went 44 -> 82 -> 156 us at one, three and four.

Those were the two batched matvecs still fed f32 (`mixed` from the gated
norm, `gated` from the gate multiply). `lex_msl::few` timed alone
at the model's shapes gives the rule: the time follows the bytes of
input every token's row holds in cache together, not the token count --
at the gate/up shape 231 us with 40 KB of it, 384 with 60, 823 with 80,
whatever the dtype -- and four f32 rows of 6144 are 96 KB. Reading the
inputs once for both of a simdgroup's rows instead of once a row did
not help (slower at three tokens); narrowing them did:

| M4 Max, context 300 | before | after |
| --- | --- | --- |
| verify of 3 | 40.8 ms (1.15 steps) | 38.5 ms (1.09) |
| verify of 4 | 50.6 ms (1.43) | 43.3 ms (1.23) |
| through the server, greedy / sampled | 52.3 / 51.2 | **55.8 / 52.9** |
| Ollama, same measurement | 57.5 / 56.4 | |

Depth 3 now pays often enough that `--depth auto` picks it on a third of
the cycles. The sampler's top-k is also one pass now (0.40 -> 0.10 ms a
row, five rows a cycle). What is left is the draft (2.5 ms a token), the
delta step and attention at three tokens (2.7x and 2.1x a step's), and
the second draft's acceptance (0.73 after an accepted first, against
Ollama's 0.90-0.94 first).

**A draft's `lm_head` over fewer rows** -- 1.4 of its 2.5 ms is the pass
over all 248320 -- measured before building (`LEX_PAIRS=1 examples/mtp`,
399 rounds of prose): the first N ids by BPE rank are a poor frequency
list. Outside the first 32768 fell 8.8% of the model's own tokens, which
takes 4.5 points off first-draft acceptance; 65536 cost 1.5 points for a
saving of ~1 ms a draft, about +1% net. Worth doing only with a ranking
taken from the model's own output (the FR-Spec construction), a 32k-row
head reordered to it.

## M5i — the CUDA prefill GEMM, 2x on the L4 (2026-10-01)

vLLM prefills this model 5.2x faster than lex on an L4 (M5g). The
GEMM alone, at the model's four shapes and 512 tokens
(`examples/gemm_bench`, each schedule checked against the interpreter
first), TFLOPS for gate/up, down, qkv, out_proj:

| CUDA schedule | TFLOPS |
| --- | --- |
| 64x64 tiles, 32 inputs a step, 4 warps, scalar tile loads (the first) | 19 22 25 28 |
| the same, 16-byte tile loads | 23 27 35 36 |
| 128x64, 16-byte loads | 32 32 47 42 |
| **128x128, 64 inputs a step, 8 warps, 16-byte loads** | **41 36 55 47** |

Taller tiles decode each FP4 weight for more tokens -- on Ada the decode,
not the multiply, is what a tile costs -- and a deeper step halves the
barriers. The output no longer stages a tile of f32 in shared memory
(which past 128x128 would not fit in the 48 KB static limit): whole
fragments go straight to `y`. Each element still sums over K in the same
order, so the result is bit-identical to the old kernel's.

**512-token prefill on the L4: 245 -> 489 tok/s** (vLLM 828). What is
left, per prefill: the GEMMs ~640 ms of 1046, the delta recurrence 120,
attention 84 (CUDA still runs the scalar program; the matrix-unit kernel
is Metal's), and the a/b projections 69 -- a 48-row bf16 matvec at 714
us a call for a quarter of a GFLOP, which should take ~20.

## M5h — drafting from the context, and a verify's attention on the matrix units (2026-10-01)

Decode through the server on a small corpus of requests a coding agent
makes (`LEX_REQUEST_LOG`, 1000-token replies, greedy, M4 Max, GPU idle):

| decode tok/s | before | split verify attention | + context lookup |
| --- | --- | --- | --- |
| whole-file edit, 3.2k-token prompt | 49.1 | 54.4 | **60.8** (+24%) |
| whole-file edit, 1.3k prompt | 58.1 | 61.2 | **66.7** (+15%) |
| change one function, 3.2k prompt | 46.9 | 49.6 | **53.4** (+14%) |
| prose essay | 46.9 | 47.5 | 47.6 |

**Context lookup (#27).** A round whose last four or more tokens occur
earlier in the prompt or reply drafts the tokens that followed them
(`lex_rt::spec::lookup`); otherwise the head drafts. Lookup rounds are
accepted 3.83 tokens of 4 on average; they cover ~64% of a whole-file
edit's reply, the rest (the reasoning before the code) is prose and goes
to the head. Verified like any other draft (`Runner::speculate_proposed`,
a point-mass proposal), so sampled output keeps its distribution. The
head never drafts in a lookup round, so its cache entry for that round's
first token is written with the accepted rows afterwards -- a hole there
changed the head's next draft by only 1e-5, so the golden test compares
the head's key cache with a clean prefill's (1.2e-3 of scale, 0.50 with
the fix deleted). `scripts/lookup_replay.py` replays logged requests to
project a scheme before building it; its first version priced verifies at
300 positions and promised 74 where 63 came, now it interpolates the
measured costs by context and matches (66.6 projected, 66.7 measured).

**The verify's attention (#29)** was the cost that grew with context: at
8000 positions a verify of four tokens cost 1.60 steps (1.25 at 300) --
the split-KV program carries a scalar accumulator per token and walks
them one after another. `lex_msl::attn::causal_mma_split` puts every
token of the verify in one tile of the matrix units, a threadgroup per
(KV head, 32 positions), and writes the program's partials so the same
combine merges them:

| verify of 4, M4 Max | before | after |
| --- | --- | --- |
| 300 positions | 1.25 steps | 1.19 |
| 2300 | 1.31 | 1.22 |
| 8000 | 1.60 | 1.24 |

Also: the batched delta kernel no longer writes the last token's snapshot
(never rolled back to), and #26 (a frequency-ranked draft vocabulary) was
measured and dropped -- on code the lookup now drafts most of the reply
without the head, on prose the list missed 16% of tokens.

## M5g — vLLM on the L4: lex decodes faster, vLLM prefills 5x faster (2026-10-01)

The same `scripts/engine_bench.py` through both servers on one L4, the
82-token prompt both render by default:

| NVIDIA L4 | decode greedy | decode sampled | prefill (473 tok) |
| --- | --- | --- | --- |
| vLLM 0.30.1 nightly, RedHatAI INT4, plain, CUDA graphs | 16.1 | 16.0 | **828** |
| lex, MLX NVFP4, speculating | **22.8** | **21.2** | 158 |

What vLLM could run is itself the finding. NVIDIA's NVFP4 checkpoint does
not fit: Ada has no FP4, so vLLM repacks it for Marlin, and that held
21.5 of 22 GB before any cache. Red Hat's INT4 fits only without the draft
head, which asked for 2.4 GB more beside 20 GB of weights. So on a 24 GB
card vLLM decodes plainly, and lex -- 17 GB with its head -- speculates.
Its plain step is a little faster than ours (16.1 against ~15).

Prefill is where lex is behind, and by a lot: 473 tokens in 0.57 s is
~45 TFLOPS of the L4's ~121, where lex's GEMM (`lex_msl::gemm` on wmma,
the Metal tile shape) does ~13 even at a full 512-token chunk (245 tok/s).
That, and the chunk cutting (158 through the server against 245), is the
NVIDIA work.

Getting vLLM to run took four image rebuilds, each fixing what the last
L4 run died of, all now checked on the CPU builder: the nightly wheel
needs torch for CUDA 13 (`libcudart.so.13`); Triton builds a C helper at
start against `Python.h`; FlashInfer compiles attention at the first
request unless its prebuilt kernels (`flashinfer-jit-cache`) are
installed. A big-card run (NVFP4 native on Blackwell) waits on quota: the
project has none above 24 GB.

## M4 — first proof: a Llama on CUDA

`llama3.2:1b` runs end to end on an L4, from the same `lex-front`
programs the Mac runs, and produces the same tokens:

    Metal: 12366 13 578 469 3168 301 22703 374 7559 304 12366 13 578 9928 49606 16730
    L4   : 12366 13 578 469 3168 301 22703 374 7559 304 12366 13 578 9928 49606 16730

Identical, with logprobs agreeing to about 5e-4 and the same 4,410
dispatches. Metal is pinned to Ollama's tokens by the Llama golden tests,
so the chain holds at both ends. Speed is 124 tok/s on the L4 against
Ollama's 163 on the same machine (76%), and 219 on an M4 Max.

This is the milestone that matters most for the project's claim and least
for anyone's tokens per second. Until it existed, "no per-target kernel
rewrites" was an argument with one backend behind it.

### What it cost, and why

Seven cloud runs, because the checks on the laptop were weaker than they
looked. In order:

1. `ollama_model` looked only in `$HOME/.ollama`. On Linux the store
   belongs to the `ollama` service user.
2. The CUDA step was gated on `ollama pull`, which failed when the server
   fell over, with the model already on disk.
3. Two runs guessing at store layout, when the layout was right all
   along.
4. The real cause of (3): the store belongs to another user and the
   installer's `usermod` does nothing for a shell whose groups were fixed
   at login. The file was there, `sudo` read it, we could not, and the
   error said only "no manifest". It now reports each candidate's io
   error kind, so "permission denied" and "entity not found" are told
   apart.
5. `INFINITY` undefined. The emitter writes it for an online softmax's
   running max; MSL has it, and CUDA has it in `math.h`, which NVRTC does
   not give you.

(5) is the one worth keeping. `scripts/cuda_check.sh` had reported all 28
kernels compiling, and it was right -- it used `nvcc`, which has the full
toolchain's headers, while the runtime uses NVRTC in-process. **The gate
was checking a different compiler from the one that runs.** It now runs
both, via `scripts/nvrtc_check.c` in the same container with no GPU.
Deleting the fix and re-running gives the number worth remembering: nvcc
0 errors, NVRTC 44.

The general lesson is the one this repository keeps relearning: a check
that cannot fail, or that checks something adjacent to what ships, reads
exactly like a check that passes.

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
