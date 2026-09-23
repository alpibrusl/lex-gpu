# lex

A compiler for LLM inference kernels: one typed program for a forward pass,
lowered to roofline-class kernels with no per-target kernel rewrites.

A **tile** is what the type system tracks — a block of values with an owner, a
layout and a lifetime — and the checker enforces that every one is consumed
exactly once. Tiles are what lex is made of.

**Two things the design promises and this does not do yet, stated plainly
because the rest of this file is measurements and they deserve the same
honesty:**

- **There is no surface syntax.** No parser, no file extension, nothing you
  write a program *in*. Programs are Rust that builds the typed IR — see
  [`llama.rs`](crates/lex-front/src/llama.rs) for what a kernel looks like
  today. [`docs/design.md`](docs/design.md) has the intended syntax, with the
  algorithm/schedule split; it is a design, not an implementation.
- **Only Metal exists.** The claim that matters here is portability, and
  nothing demonstrates it until a second backend does. Until then the type
  system's case — that linear tiles and effects catch across targets what
  each target's own tooling catches only on that target — is an argument,
  not a result.

What *is* real: the linear type system, the checker, the reference
interpreter, the MSL backend, and a 27.8B model that runs end to end on the
kernels it generates and answers with the same tokens as Ollama.

The full design is in [`docs/design.md`](docs/design.md). The plan for getting
there is in [`docs/roadmap.md`](docs/roadmap.md).

## Status

**Where it stands:** a Llama-3.1-8B served by Ollama, in 4-bit, runs entirely
on kernels this compiler generated, and produces the same tokens as Ollama.
Decode runs at 89–95% of Ollama's speed on the 8B and 95–99% on the 1B,
from an empty context to 1,440 positions: split-KV attention made decode
flat with context, as Ollama's is. That's 12× and 20× faster than where
correctness left it. Prefill has a correct batched path, but it's 5–8×
short of Ollama.

**Qwen3.5-27B** (`qwen3.8:27b-mlx`, NVFP4, the model in daily use here)
also runs on these kernels now, at 26.5 tok/s, or **42.7 speculating with
the checkpoint's own draft head** — against Ollama's 58–76, with the same
answers either way. [`docs/qwen.md`](docs/qwen.md) has the shape of the
model, where the time goes, and what matching Ollama needs.

| Phase | State | Details |
| --- | --- | --- |
| **P0** Spine | closed | RMSNorm at 98.1% of the copy ceiling (463.6 GB/s) on an M4 Max, matching the reference. [`docs/P0.md`](docs/P0.md) |
| **P1** Types | closed (in the interpreter) | Linear tiles, effect-typed copies and barrier-synchronised pipes check a flash-attention decode loop: plain, double-buffered, and warp-specialised for Hopper. All variants match PyTorch. [`docs/P1.md`](docs/P1.md) |
| **P2** Metal, correct | **exit test met** | Llama-3.1-8B in int4 (Q4_K_M, from Ollama) runs on lex kernels on the GPU, and its greedy tokens are identical to Ollama's. [`docs/P2.md`](docs/P2.md) |
| **P3** Metal, fast | in progress | Decode at 89–99% of Ollama at every context measured (0–1,440 positions), reading as many bytes per token as llama.cpp. A batched forward pass (prefill, speculative verify) is correct but slow. Next: `simdgroup_matrix` and split-KV for prefill. [`docs/P3.md`](docs/P3.md) |

Measured on an M4 Max, each model greedy-decoded on 4 prompts × 24 tokens
next to Ollama itself (`scripts/lex_vs_ollama.py`):

| Model | Weights | Tokens identical to Ollama | Log-prob gap vs Ollama | vs f32 reference | lex decode, 0–1,440 context | Ollama |
| --- | --- | --- | --- | --- | --- | --- |
| `llama3.2:1b` | Q8_0 | 96 / 96 | ≤ 0.009 | ≤ 0.007 | ~244–265 tok/s | ~256–275 tok/s |
| `llama3.1:8b` | Q4_K_M | 96 / 96 | ≤ 0.08 | ≤ 0.007 | ~77–80 tok/s | ~83–87 tok/s |

How to read the table:
- **Correctness:** lex agrees with an f32 PyTorch reference to within 0.007 on
  both models. Ollama differs from both by more, up to 0.09 on the 8B,
  because llama.cpp's quantised kernels round differently. So the remaining
  gap is on Ollama's side, not lex's.
- **Speed:** lex reaches 95–99% of Ollama on the 1B and 89–95% on the 8B,
  reading 4.71 GB of weights per token against llama.cpp's 4.62. Its big
  matvecs read at 437–523 GB/s, against a 463 GB/s copy benchmark. Like
  Ollama's, the speed barely moves with context: split-KV attention spreads
  the cache over many threadgroups and merges their partial softmaxes
  (before it, the 8B fell to 20 tok/s at 1,440 positions).
  [`docs/P3.md`](docs/P3.md) has the table. At P2's end it was 13 and
  6.8 tok/s: every op was a separate dispatch that waited for the last, and
  the kernels were the simplest correct ones.
  `cargo run --release -p lex-rt --example profile -- --model llama3.1:8b --context 512`
  shows where the time goes at a given context, per kernel from GPU
  timestamps.
  [`docs/P3.md`](docs/P3.md) covers what's left: prefill and fused small
  ops.

What runs where:
- **Copy and RMSNorm** are hand-planned P0 kernels. They run on the GPU at
  parity with PyTorch (example 3).
- **Llama inference** is typed `lex-front` programs: RMSNorm, quantised
  matvec (Q8_0 / Q4_K / Q6_K), RoPE, flash-decode attention (serial and
  split-KV), SiLU·mul. They are checked, run in the interpreter, and are
  lowered to MSL to run on the GPU.
- **Host glue:** the embedding-row lookup and the RoPE tables are written by
  the CPU, on unified memory, before each token's command buffer. Everything
  inside the token, KV-cache append included, runs on the GPU.

## Examples

Rust ≥ 1.88 for everything. The Python scripts need `torch`, `numpy` and
`tokenizers` (`pip install torch numpy tokenizers`); nothing in `cargo test`
does.

### 1. A real model, against Ollama (Mac only)

```sh
ollama pull llama3.1:8b                            # or llama3.2:1b (1.3 GB)
python3 scripts/lex_vs_ollama.py --model llama3.1:8b
```

It tokenises each prompt the way Ollama does and greedy-decodes it with lex
on the GPU. It asks Ollama for the same continuation, then compares every
token and every top-5 log-probability. On `llama3.1:8b`:

```text
'The capital of France is'
  lex    : ' a city of grandeur and beauty, with a rich history and culture that is reflected in its stunning architecture, world-class'
  24/24 tokens identical to Ollama, worst |dlogprob| 0.0202 (tolerance 0.1)  PASS
  lex decode 33.9 tok/s on the GPU

'def fibonacci(n):'
  lex    : ' \n    if n <= 0: \n        return "Input should be a positive integer" \n    elif n =='
  24/24 tokens identical to Ollama, worst |dlogprob| 0.0828 (tolerance 0.1)  PASS
  lex decode 64.3 tok/s on the GPU
  ...
```

In this run the first prompt measured slower than the other three, which
settled at ~63–64 tok/s against Ollama's ~88. That outlier isn't explained
yet (it didn't show up in the previous run), so it's shown as measured.

The weights are the GGUF blob in Ollama's own store, found through its
manifest; there is no conversion step. `--prompt "..."` and `--steps N` take
your own prompts.

There are two more pieces, and neither needs Ollama running at test time:
- `scripts/llama_ref.py --model <tag>` checks a float32 PyTorch reference
  against Ollama. It writes that reference's outputs as a golden file.
- `cargo test --release -p lex-rt --test llama_ollama` checks lex on the GPU
  against the golden for each model that is pulled. It takes about 12 s for
  the 8B.
  [`docs/P2.md`](docs/P2.md) has the full chain.

### 2. Bandwidth on the GPU (Mac only)

```sh
cargo run --release -p lex-bench
```

```text
device : Apple M4 Max (unified memory, 32768 B threadgroup, 55.7 GB recommended working set)
target : apple-m-series (simd 32, 1024 threads/tg, 32768 B threadgroup)

kernel          ideal bytes         time       GB/s    % of copy
copy_f32           536.9 MB     1.207 ms      444.9       100.0%
rmsnorm_f32        536.9 MB     1.253 ms      428.5        96.3%

exit test : rmsnorm at 96.3% of the copy ceiling (need >= 90.0%)  PASS
            max rel err 1.42e-6 (tolerance 1e-5)  PASS
```

`--dtype f16`, `--rows`, `--cols`, `--mib`, `--iters` and `--repeats` all
move; `--help` lists them. `--emit` prints the generated MSL instead, and works
on any host.

### 3. Benchmark against PyTorch (Mac only)

```sh
python3 scripts/bench_vs_pytorch.py              # f32
python3 scripts/bench_vs_pytorch.py --dtype f16  # takes the lex-bench flags too
```

It runs `lex-bench` and the PyTorch MPS equivalents on the same shapes. Both
are timed the same way (`iters` back-to-back calls, synchronise, fastest of
`repeats` batches), and both use the same ideal-bytes accounting. It also
checks that the two PyTorch paths agree. On an M4 Max:

```text
kernel                    lex GB/s  torch GB/s  lex / torch
copy_f32                      433.8       437.2         0.99x
rmsnorm_f32                   422.7       433.3         0.98x
rmsnorm_f32 (eager)           422.7        67.3         6.28x

copy_f16                      453.9       445.2         1.02x
rmsnorm_f16                   444.5       452.4         0.98x
rmsnorm_f16 (eager)           444.5        27.4        16.24x
flash_decode_f16               23.4       379.5         0.06x
flash_decode_f16 (gqa)         23.4        12.5         1.88x
```

How to read the rows:
- **`rmsnorm`** is PyTorch's fused `F.rms_norm`. Parity is the expected result
  for a bandwidth-bound op, and it is the comparison that matters.
- **`(eager)`** is the textbook expression most model code runs: upcast,
  square, mean, rsqrt and multiply, one kernel each.
- **`flash_decode_f16`** uses Llama-3-8B's decode shape (batch 4, seq 4096; set
  with `--batch` / `--seq`) and compares against SDPA on the same grouped
  layout. That is the fair comparison, and lex loses it by about 16× for now:
  one threadgroup per KV head, serial K/V streaming, conservative barriers.
  [`docs/P2.md`](docs/P2.md) says what P3 changes.
- **`flash_decode_f16 (gqa)`** is SDPA with `enable_gqa=True`. It is slower on
  MPS, so it would flatter lex. It is shown so nobody quotes it by mistake.

Expect a few percent of run-to-run noise either way.

### 4. Flash-attention decode vs PyTorch (any host)

```sh
cargo run -p lex-front --example flash_decode
```

It builds one algorithm under three schedules and checks each against three
target tables. It runs every schedule in the interpreter and compares against
PyTorch's `scaled_dot_product_attention`:

```text
metal: bq 8, bk 32, 2 stages
  apple-m-series   ok       32768 B threadgroup, 18 borrows, 0 dups
                           warning: apple-m-series has no async copy engine; ...
  nvidia-hopper    ok       32768 B threadgroup, 18 borrows, 0 dups
  amd-cdna3        ok       32768 B threadgroup, 18 borrows, 0 dups
  interpreter vs PyTorch SDPA: max rel err 1.02e-5

hopper: bq 16, bk 128, 3 stages
  apple-m-series   REJECT  Budget: peak threadgroup footprint is 196608 B; apple-m-series allows 32768 B
  nvidia-hopper    ok      196608 B threadgroup, 27 borrows, 0 dups
  amd-cdna3        REJECT  Budget: peak threadgroup footprint is 196608 B; amd-cdna3 allows 65536 B
  interpreter vs PyTorch SDPA: max rel err 1.84e-5

ws: producer + 2 consumer warpgroups, bk 128, 3 stages
  apple-m-series   REJECT  Target: apple-m-series has no split barriers: ...
  nvidia-hopper    ok      196608 B threadgroup, 24 borrows, 0 dups
                           pipe: 3 slots, full barrier 1 arrival + 65536 B tx, empty barrier 2 arrivals, 288 threads
  amd-cdna3        REJECT  Target: amd-cdna3 has no split barriers: ...
  interpreter vs PyTorch SDPA: max rel err 1.84e-5
```

To read a program:

```sh
cargo run -p lex-front --example flash_decode -- --ir ws   # or metal, hopper
```

The PyTorch output is checked in, so this and `cargo test` need no Python. To
regenerate it, run `scripts/flash_decode_golden.py`. It rebuilds the inputs from
the same xorshift pattern the Rust side uses, and writes only SDPA's output:

```sh
python3 scripts/flash_decode_golden.py
```

### 5. Flash decode on the GPU (Mac only)

```sh
cargo run --release -p lex-bench -- --flash    # --batch, --seq, --emit
```

```text
flash decode: 32 threadgroups x 128 threads, 18432 B threadgroup, 67.2 MB per step

device : Apple M4 Max
shape  : batch 4 x 8 kv heads x 4 q heads, head dim 128, seq 4096, f16
kernel : 32 threadgroups x 128 threads, 18432 B threadgroup (16384 tiles + 2048 scratch), 34 barrier sites

kernel              ideal bytes         time       GB/s    % of copy
copy_f16               268.4 MB     0.574 ms      467.6       100.0%
flash_decode_f16        67.2 MB     2.864 ms       23.5         5.0%

correct : max rel err 8.21e-6 vs f64 reference (tolerance 1e-4)  PASS
```

The kernel is the same `lex-front` flash-decode program as example 4, with a
grid of one instance per (sequence, KV head), lowered by
`lex_msl::program::lower`. `--emit` prints the MSL, and
`cargo test -p lex-metal --test flash_gpu` checks it against PyTorch on the
GPU.

### 6. Tests (any host, including Linux CI)

```sh
cargo test --workspace
```

This covers the IR, planner, emitter goldens (the lowered flash kernel
included) and reference, plus these suites:
- lex-front `flash`: schedules × targets, against PyTorch;
- lex-front `roles`: warp specialisation, run under many thread
  interleavings;
- lex-front `linearity`: every checker rule rejecting the bug it exists for,
  one diagnostic each;
- lex-metal `flash_gpu`: the lowered kernel on the GPU, against PyTorch.
  This one needs macOS; CI's paravirtualised device is enough.
- lex-metal `llama_kernels_gpu`: every Llama kernel on the GPU against the
  interpreter, at real sizes and in every weight layout.
- lex-rt `llama_ollama`: Llama 3.2 1B and Llama 3.1 8B on the GPU against
  the Ollama-checked reference, fed four ways (token by token, prefill in 4s
  and 16s, batched verify). It needs macOS and the models pulled; for any
  model missing, it prints `SKIPPED`. Run it with `--release`: in a debug
  build it takes many minutes.

### 7. Ollama's baseline, here or on an NVIDIA GPU in the cloud

```sh
python3 scripts/ollama_bench.py --model llama3.1:8b          # decode and prefill at 0/512/1440 context
GCP_PROJECT=<project> scripts/gcp/nvidia_test.sh              # same, plus the test suite, on an L4 in europe-west4
```

On this M4 Max, Ollama decodes the 8B at 86/84/82 tok/s (0/512/1,440
positions) and prefills at ~900 tok/s. The 1B decodes at 263–266 tok/s and
prefills at 5,300–6,200 tok/s. The cloud script creates the VM, runs the
workspace tests and the same benchmark on the GPU, copies the results home,
and deletes the VM. [`docs/cloud.md`](docs/cloud.md) covers GPUs, EU zones,
quota and the cost guard rails. It becomes the CUDA backend's test bed
once that backend exists.

## Layering

```text
  Kernel            what to compute. No target anywhere in it.
    + Target        the hardware table. Data, not code.
    = Plan          launch geometry, memory budget, derived constants.
      -> backend    emits source from (Kernel, Plan).
```

`Plan` is a separate value on purpose. In P0 the planner computes it from two
constants; in P3 it searches for it. Nothing above or below has to change shape
for that to happen — which is the algorithm/schedule split from the design doc,
in its smallest possible form.

| Crate | Responsibility | Builds off a Mac |
| --- | --- | --- |
| `lex-ir` | Tile IR, target table (Apple, Hopper, CDNA3), planner, CPU reference | yes |
| `lex-front` | Typed tile programs: linearity, effect and pipe-protocol checker; concurrent reference interpreter | yes |
| `lex-msl` | MSL emission for P0 kernels; lowering of `lex-front` programs; golden files | yes |
| `lex-metal` | Compile, allocate, dispatch, time | **no** |
| `lex-bench` | Harness: emit, verify, measure (`--flash` for decode attention) | yes (device path gated) |
| `lex-rt` | Runtime: GGUF reader, Q8_0/Q4_K/Q6_K repacking, Llama decode loop over lex kernels | yes (decode loop gated) |

That boundary is load-bearing. Everything except device dispatch is ordinary
Rust with tests, so the compiler can be developed anywhere and only the numbers
need the Mac.

## Golden files

`crates/lex-msl/tests/golden/*.metal` are the emitter's committed output. They
are the only readable artifact the backend produces, and on a host with no Metal
compiler they are the strongest available signal. After an intentional change:

```sh
LEX_BLESS=1 cargo test -p lex-msl
```

Read the diff before committing it.
`crates/lex-front/tests/data/*.f32` is PyTorch output (see example 3).

## Not built yet

- **Layouts in the type:** no swizzle or MMA-fragment layouts are checked yet.
- **A surface syntax:** no lexer, no parser, no file extension. Programs are
  built through the Rust IR API, and schedules are builder parameters. The
  syntax in `docs/design.md` is deliberately unbuilt until a second backend
  says what it has to express — designing it against one target would mean
  designing it twice.
- **Fast lowering:** simdgroup matrices, split-K decode and minimal barriers
  are P3.
- **Around the model:** the embedding lookup and KV append as kernels (host
  glue today), paged KV, sampling beyond greedy, and batched prefill.
- **Other parts of the design:** quantised formats, a graph compiler, MLIR.

[`docs/P1.md`](docs/P1.md) lists exactly what the type system does and does not
cover yet.

## Licence

Copyright © 2026 Alfonso Sastre

Licensed under the EUPL, Version 1.2 — see [`LICENSE`](LICENSE).

The EUPL is copyleft: a derivative work must be released under the EUPL or one
of the compatible licences in its Appendix (GPL, AGPL, LGPL, MPL-2.0, EPL,
OSL, CeCILL, LiLiQ). Apache-2.0 and MIT are *not* on that list, so code from
Apache- or MIT-licensed projects cannot be copied into this one, and this code
cannot be vendored into an Apache-2.0 project. Everything here is written from
the published behaviour of other kernels, not from their source.
