# tile

A GPU-native language for LLM inference: one source program for a forward pass
that compiles to roofline-class kernels on Metal, NVIDIA and AMD, with no
per-target kernel rewrites.

The full design is in [`docs/design.md`](docs/design.md). The plan for getting
there is in [`docs/roadmap.md`](docs/roadmap.md).

## Status

| Phase | State | Details |
| --- | --- | --- |
| **P0** Spine | closed | RMSNorm at 98.1% of the copy ceiling (463.6 GB/s) on an M4 Max, matching the reference. [`docs/P0.md`](docs/P0.md) |
| **P1** Types | exit test met in the interpreter | Linear tiles, effect-typed copies and barrier-synchronised pipes check a flash-attention decode loop: plain, double-buffered, and warp-specialised for Hopper. All variants match PyTorch. [`docs/P1.md`](docs/P1.md) |
| P2 Metal, correct | next | Full op set, int4, paged KV, Llama-3-8B tokens. |

Two things are true at once, and it is worth being precise about which is
which. The **Metal kernels** (copy, RMSNorm) run on the GPU and are benchmarked
below. The **typed kernels** (flash-attention decode) are checked and run in
the reference interpreter; nothing lowers them to a GPU yet — that is P2.

## Examples

Rust ≥ 1.88 for everything. The PyTorch scripts need `torch` and `numpy`
(`pip install torch numpy`); nothing in `cargo test` does.

### 1. Bandwidth on the GPU (Mac only)

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

### 2. Benchmark against PyTorch (Mac only)

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

copy_f16                      444.9       439.4         1.01x
rmsnorm_f16                   446.2       438.5         1.02x
rmsnorm_f16 (eager)           446.2        27.3        16.34x
```

`rmsnorm` is PyTorch's fused `F.rms_norm`: parity is the expected result for a
bandwidth-bound op, and it is the one that matters. `(eager)` is the textbook
expression most model code runs (upcast, square, mean, rsqrt, multiply, one
kernel each). Expect a few percent of run-to-run noise either way.

### 3. Flash-attention decode vs PyTorch (any host)

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

### 4. Tests (any host, including Linux CI)

```sh
cargo test --workspace
```

This covers the IR, planner, emitter goldens and reference, plus the tile-front
suites:
- `flash`: schedules × targets, against PyTorch;
- `roles`: warp specialisation, run under many thread interleavings;
- `linearity`: every checker rule rejecting the bug it exists for, one
  diagnostic each.

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
| `tile-msl` | MSL text emission + golden files | yes |
| `tile-metal` | Compile, allocate, dispatch, time | **no** |
| `tile-bench` | P0 harness: emit, verify, measure | yes (device path gated) |

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
- **Lowering typed kernels to a GPU:** the P2 work.
- **Other parts of the design:** quantised formats, a graph compiler, MLIR.

[`docs/P1.md`](docs/P1.md) lists exactly what the type system does and does not
cover yet.
