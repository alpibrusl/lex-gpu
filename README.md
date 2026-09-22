# tile

A GPU-native language for LLM inference: one source program for a forward pass
that compiles to roofline-class kernels on Metal, NVIDIA and AMD, with no
per-target kernel rewrites.

The full design is in [`docs/design.md`](docs/design.md). The plan for getting
there is in [`docs/roadmap.md`](docs/roadmap.md).

**Status: P0.** Two kernels, one target, no language yet. What exists is the
spine the rest is built on — IR, target table, planner, MSL emitter, reference
interpreter, device glue, and a harness that prints a bandwidth number.

## What P0 answers

One question: *does a kernel this compiler generated move memory as fast as the
machine can move memory at all?*

`copy` establishes the ceiling. `rmsnorm` is scored against it. The exit test is
RMSNorm at **≥ 90% of the copy ceiling**, with output matching the CPU reference
inside a declared tolerance. Until that number exists, every performance claim
further up the roadmap is a guess.

## Running it

On the Mac:

```sh
cargo run --release -p tile-bench
```

```text
device : Apple M… (unified memory, 32768 B threadgroup, … GB recommended working set)
target : apple-m-series (simd 32, 1024 threads/tg, 32768 B threadgroup)

kernel          ideal bytes         time       GB/s    % of copy
copy_f32          536.9 MB      … ms          …          100.0%
rmsnorm_f32       537.0 MB      … ms          …            … %

exit test : rmsnorm at …% of the copy ceiling (need >= 90.0%)  …
            max rel err …e-07 (tolerance 1e-05)  …
```

`--dtype f16`, `--rows`, `--cols`, `--mib`, `--iters` and `--repeats` all move;
`--help` lists them.

Anywhere else — including a Linux CI box with no Metal toolchain in sight:

```sh
cargo test --workspace          # IR, planner, emitter goldens, reference
cargo run -p tile-bench -- --emit   # print the generated MSL
```

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
| `tile-ir` | Tile IR, target table, planner, CPU reference interpreter | yes |
| `tile-msl` | MSL text emission + golden files | yes |
| `tile-metal` | Compile, allocate, dispatch, time | **no** |
| `tile-bench` | P0 harness: emit, verify, measure | yes (device path gated) |

That boundary is load-bearing. Everything except device dispatch is ordinary
Rust with tests, so the compiler can be developed anywhere and only the numbers
need the Mac Studio.

## Golden files

`crates/tile-msl/tests/golden/*.metal` are the emitter's committed output. They
are the only readable artifact the backend produces, and on a host with no Metal
compiler they are the strongest available signal. After an intentional change:

```sh
TILE_BLESS=1 cargo test -p tile-msl
```

Read the diff before committing it.

## Deliberately absent

No layout algebra, no linear/affine tile types, no effect-typed barriers, no
schedule language, no quantised formats, no graph compiler, no MLIR. Those are
P1 and later, in that order. P0 exists so that when the type system lands it
lands on a toolchain that already produces a number.
