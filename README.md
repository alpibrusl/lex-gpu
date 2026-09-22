# tile

A GPU-native language for LLM inference: one source program for a forward pass
that compiles to roofline-class kernels on Metal, NVIDIA and AMD, with no
per-target kernel rewrites.

The full design is in [`docs/design.md`](docs/design.md). The plan for getting
there is in [`docs/roadmap.md`](docs/roadmap.md).

## Status

**Where it stands:** a Llama-3.1-8B served by Ollama, in 4-bit, runs entirely
on kernels this compiler generated, and produces the same tokens as Ollama.
At short contexts, decode runs at about 70% of Ollama's speed on the 8B and
80% on the 1B, 9× and 16× faster than where correctness left it. **At
realistic contexts it's about 45%**: tile's attention slows as the context
grows and Ollama's doesn't. Split-KV attention is the next job. Prefill has
a correct batched path, but it's 8–10× short of Ollama.

| Phase | State | Details |
| --- | --- | --- |
| **P0** Spine | closed | RMSNorm at 98.1% of the copy ceiling (463.6 GB/s) on an M4 Max, matching the reference. [`docs/P0.md`](docs/P0.md) |
| **P1** Types | closed (in the interpreter) | Linear tiles, effect-typed copies and barrier-synchronised pipes check a flash-attention decode loop: plain, double-buffered, and warp-specialised for Hopper. All variants match PyTorch. [`docs/P1.md`](docs/P1.md) |
| **P2** Metal, correct | **exit test met** | Llama-3.1-8B in int4 (Q4_K_M, from Ollama) runs on tile kernels on the GPU, and its greedy tokens are identical to Ollama's. [`docs/P2.md`](docs/P2.md) |
| **P3** Metal, fast | in progress | Short-context decode at ≥ 70% of Ollama, reading as many bytes per token as llama.cpp. At 512 tokens of context it falls to ~45%. A batched forward pass (prefill, speculative verify) is correct but slow. Next: split-KV attention, then `simdgroup_matrix` for prefill. [`docs/P3.md`](docs/P3.md) |

Measured on an M4 Max, each model greedy-decoded on 4 prompts × 24 tokens
next to Ollama itself (`scripts/tile_vs_ollama.py`):

| Model | Weights | Tokens identical to Ollama | Log-prob gap vs Ollama | vs f32 reference | tile (short context) | Ollama |
| --- | --- | --- | --- | --- | --- | --- |
| `llama3.2:1b` | Q8_0 | 96 / 96 | ≤ 0.009 | ≤ 0.007 | ~214–218 tok/s | ~268 tok/s |
| `llama3.1:8b` | Q4_K_M | 96 / 96 | ≤ 0.08 | ≤ 0.007 | ~60–64 tok/s | ~88 tok/s |

How to read the table:
- **Correctness:** tile agrees with an f32 PyTorch reference to within 0.007 on
  both models. Ollama differs from both by more, up to 0.09 on the 8B,
  because llama.cpp's quantised kernels round differently. So the remaining
  gap is on Ollama's side, not tile's.
- **Speed:** at short contexts, tile reaches about 80% of Ollama on the 1B
  and 68–72% on the 8B, reading 4.71 GB of weights per token against
  llama.cpp's 4.62. At 512 tokens of context it's about 45% on both
  (8B ~37 vs ~85 tok/s), because tile's attention slows with context and
  Ollama's doesn't. [`docs/P3.md`](docs/P3.md) has the table. At P2's
  end it was 13 and 6.8 tok/s: every op was a separate dispatch that waited
  for the last, and the kernels were the simplest correct ones.
  `cargo run --release -p tile-rt --example profile -- --model llama3.1:8b`
  shows where the time goes now, per kernel from GPU timestamps.
  [`docs/P3.md`](docs/P3.md) covers what's left: prefill and fused small
  ops.

What runs where:
- **Copy and RMSNorm** are hand-planned P0 kernels. They run on the GPU at
  parity with PyTorch (example 3).
- **Llama inference** is typed `tile-front` programs: RMSNorm, quantised
  matvec (Q8_0 / Q4_K / Q6_K), RoPE, flash-decode attention, SiLU·mul. They
  are checked, run in the interpreter, and are lowered to MSL to run on the
  GPU.
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
python3 scripts/tile_vs_ollama.py --model llama3.1:8b
```

It tokenises each prompt the way Ollama does and greedy-decodes it with tile
on the GPU. It asks Ollama for the same continuation, then compares every
token and every top-5 log-probability. On `llama3.1:8b`:

```text
'The capital of France is'
  tile   : ' a city of grandeur and beauty, with a rich history and culture that is reflected in its stunning architecture, world-class'
  24/24 tokens identical to Ollama, worst |dlogprob| 0.0202 (tolerance 0.1)  PASS
  tile decode 33.9 tok/s on the GPU

'def fibonacci(n):'
  tile   : ' \n    if n <= 0: \n        return "Input should be a positive integer" \n    elif n =='
  24/24 tokens identical to Ollama, worst |dlogprob| 0.0828 (tolerance 0.1)  PASS
  tile decode 64.3 tok/s on the GPU
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
- `cargo test --release -p tile-rt --test llama_ollama` checks tile on the GPU
  against the golden for each model that is pulled. It takes about 12 s for
  the 8B.
  [`docs/P2.md`](docs/P2.md) has the full chain.

### 2. Bandwidth on the GPU (Mac only)

```sh
cargo run --release -p tile-bench
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
python3 scripts/bench_vs_pytorch.py --dtype f16  # takes the tile-bench flags too
```

It runs `tile-bench` and the PyTorch MPS equivalents on the same shapes. Both
are timed the same way (`iters` back-to-back calls, synchronise, fastest of
`repeats` batches), and both use the same ideal-bytes accounting. It also
checks that the two PyTorch paths agree. On an M4 Max:

```text
kernel                    tile GB/s  torch GB/s  tile / torch
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
  layout. That is the fair comparison, and tile loses it by about 16× for now:
  one threadgroup per KV head, serial K/V streaming, conservative barriers.
  [`docs/P2.md`](docs/P2.md) says what P3 changes.
- **`flash_decode_f16 (gqa)`** is SDPA with `enable_gqa=True`. It is slower on
  MPS, so it would flatter tile. It is shown so nobody quotes it by mistake.

Expect a few percent of run-to-run noise either way.

### 4. Flash-attention decode vs PyTorch (any host)

```sh
cargo run -p tile-front --example flash_decode
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
cargo run -p tile-front --example flash_decode -- --ir ws   # or metal, hopper
```

The PyTorch output is checked in, so this and `cargo test` need no Python. To
regenerate it, run `scripts/flash_decode_golden.py`. It rebuilds the inputs from
the same xorshift pattern the Rust side uses, and writes only SDPA's output:

```sh
python3 scripts/flash_decode_golden.py
```

### 5. Flash decode on the GPU (Mac only)

```sh
cargo run --release -p tile-bench -- --flash    # --batch, --seq, --emit
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

The kernel is the same `tile-front` flash-decode program as example 4, with a
grid of one instance per (sequence, KV head), lowered by
`tile_msl::program::lower`. `--emit` prints the MSL, and
`cargo test -p tile-metal --test flash_gpu` checks it against PyTorch on the
GPU.

### 6. Tests (any host, including Linux CI)

```sh
cargo test --workspace
```

This covers the IR, planner, emitter goldens (the lowered flash kernel
included) and reference, plus these suites:
- tile-front `flash`: schedules × targets, against PyTorch;
- tile-front `roles`: warp specialisation, run under many thread
  interleavings;
- tile-front `linearity`: every checker rule rejecting the bug it exists for,
  one diagnostic each;
- tile-metal `flash_gpu`: the lowered kernel on the GPU, against PyTorch.
  This one needs macOS; CI's paravirtualised device is enough.
- tile-metal `llama_kernels_gpu`: every Llama kernel on the GPU against the
  interpreter, at real sizes and in every weight layout.
- tile-rt `llama_ollama`: Llama 3.2 1B and Llama 3.1 8B on the GPU against
  the Ollama-checked reference, fed four ways (token by token, prefill in 4s
  and 16s, batched verify). It needs macOS and the models pulled; for any
  model missing, it prints `SKIPPED`. Run it with `--release`: in a debug
  build it takes many minutes.

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
| `tile-ir` | Tile IR, target table (Apple, Hopper, CDNA3), planner, CPU reference | yes |
| `tile-front` | Typed tile programs: linearity, effect and pipe-protocol checker; concurrent reference interpreter | yes |
| `tile-msl` | MSL emission for P0 kernels; lowering of `tile-front` programs; golden files | yes |
| `tile-metal` | Compile, allocate, dispatch, time | **no** |
| `tile-bench` | Harness: emit, verify, measure (`--flash` for decode attention) | yes (device path gated) |
| `tile-rt` | Runtime: GGUF reader, Q8_0/Q4_K/Q6_K repacking, Llama decode loop over tile kernels | yes (decode loop gated) |

That boundary is load-bearing. Everything except device dispatch is ordinary
Rust with tests, so the compiler can be developed anywhere and only the numbers
need the Mac.

## Golden files

`crates/tile-msl/tests/golden/*.metal` are the emitter's committed output. They
are the only readable artifact the backend produces, and on a host with no Metal
compiler they are the strongest available signal. After an intentional change:

```sh
TILE_BLESS=1 cargo test -p tile-msl
```

Read the diff before committing it.
`crates/tile-front/tests/data/*.f32` is PyTorch output (see example 3).

## Not built yet

- **Layouts in the type:** no swizzle or MMA-fragment layouts are checked yet.
- **Schedule language:** schedules are Rust builder parameters for now.
- **Fast lowering:** simdgroup matrices, split-K decode and minimal barriers
  are P3.
- **Around the model:** the embedding lookup and KV append as kernels (host
  glue today), paged KV, sampling beyond greedy, and batched prefill.
- **Other parts of the design:** quantised formats, a graph compiler, MLIR.

[`docs/P1.md`](docs/P1.md) lists exactly what the type system does and does not
cover yet.
