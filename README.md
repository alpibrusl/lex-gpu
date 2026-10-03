# lex

A compiler for LLM inference kernels: one typed program for a forward pass,
lowered to roofline-class kernels for Apple GPUs (Metal) and NVIDIA (CUDA)
with no per-target kernel rewrites. A **tile** — a block of values with an
owner, a layout and a lifetime — is what the type system tracks, and the
checker holds every one to being consumed exactly once.

It runs real models end to end on the kernels it generates, behind an
OpenAI-compatible server: Llama 3.x, MiMo-v2.6 and Qwen3.8-27B (hybrid
gated-delta and attention layers, NVFP4 weights, a draft head for
speculative decoding).

## Status (2026-09-30)

Decode, tok/s, against Ollama on the same machine:

| Model | Hardware | lex | Ollama |
| --- | --- | --- | --- |
| `maternion/mimo-v2.6:9b` | M4 Max, greedy / sampled | **74.6 / 73.4** | 66.9 / 62.0 |
| `qwen3.8:27b-mlx` | M4 Max, speculating, greedy / sampled | 55.8 / 52.9 | 57.5 / 56.4 |
| `llama3.1:8b` | M4 Max | 79.2 | 86.0 |
| `llama3.2:1b` | M4 Max | 233.7 | 261.9 |
| `llama3.2:1b` | NVIDIA L4 | 124.4 | 162.9 |
| `qwen3.8:27b-mlx` | NVIDIA L4, plain / speculating | 15.1 / 24.4 | — (MLX build) |

Prefill, `qwen3.8:27b-mlx`, 512 tokens: **229 tok/s** on the M4 Max
(221 at 2048) against Ollama's 250–260, and **489** on the L4 (vLLM 828).

On the L4 the bar is vLLM (Ollama cannot run the MLX build there). Same
prompt, through both servers: decode **lex 22.8** against vLLM's 16.1
(Red Hat's INT4 checkpoint, plain -- its draft head does not fit a 24 GB
card beside the weights, and NVIDIA's NVFP4 checkpoint does not fit at
all); prefill **vLLM 828** against lex's 158. The CUDA GEMM is the gap.

Correctness: Llama gives Ollama's tokens exactly (96/96 on both sizes),
MiMo 16/16; Qwen3.8 passes its whole golden suite on Metal and on the L4,
worst log-prob difference 0.00077 against an f32 reference (tolerance
0.02).

Qwen decode on the Mac is at 0.97x greedy and 0.94x sampled
(`scripts/engine_bench.py`, same prompt token for token): a plain step
matches Ollama's (35 ms), and a verify of three tokens costs 1.09 steps.
mlx-lm decodes the same model at 29.5. How each number was measured, and
what is being tried next, is in
[`docs/roadmap-weeks.md`](docs/roadmap-weeks.md).

The language is partly there: a kernel can be written in `.lx` (see
[The language](#the-language)), and the NVFP4 prefill GEMM written that way
runs the model on an M4 Max at the hand-written kernel's speed. Attention, the
gated-delta rule and the matvecs are still built through the Rust IR API or
hand-scheduled; see [`docs/guide.md`](docs/guide.md#not-built-yet).

## Quick start

Rust ≥ 1.88. Models come from Ollama's own store (`ollama pull ...`).

```bash
cargo run --release -p lex-rt --example serve -- --model qwen3.8:27b-mlx
```

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{"model":"lex","messages":[{"role":"user","content":"hello"}]}'
```

Streaming, tool calls and the checkpoint's own chat template work; the
official OpenAI client and [lex-code](https://github.com/alpibrusl/lex-code)
drive it unmodified.

```bash
cargo test --workspace                                   # any host; GPU suites need --release
python3 scripts/lex_vs_ollama.py --model llama3.1:8b     # tokens and log-probs against Ollama
GCP_PROJECT=<project> scripts/gcp/nvidia_test.sh         # the suite on an NVIDIA L4, VM deleted after
```

More examples, benchmarks against PyTorch, and how to read the numbers:
[`docs/guide.md`](docs/guide.md).

## The language

A kernel is an `algo` -- what it computes, over tiles, with no machine in it --
and a `schedule` per target that says how big the pieces are and how the warps
divide them. This is `crates/lex-front/lx/gemm_fp4.lx`, the matmul the model's
prefill runs, with NVFP4 weights:

```text
algo gemm_fp4(m, n, k)
  in  x:  f16[m, k]
  in  wq: i8[n, k / 2]          // two 4-bit codes a byte
  in  ws: i8[n, k / 16]         // an E4M3 scale per 16 values
  in  wg: f32[n]                // a per-row scale
  out y:  f32[m, n]
  tile bm, bn, bk
{
  grid i over m / bm, j over n / bn
  let acc = zeros f32[bm, bn] @frag                 // lives in matrix-unit fragments
  let acc = for p in 0 .. k / bk with acc {         // the loop carries the tile
    let xt = load x[i * bm, p * bk ; bm, bk] @shared
    let q  = load wq[j * bn, p * (bk / 2) ; bn, bk / 2]
    let s  = load ws[j * bn, p * (bk / 16) ; bn, bk / 16]
    let g  = load wg[j * bn ; bn]
    let wt = stage f16 (dequant_fp4 q s g 16)       // decoded into shared memory
    yield mma acc xt wt
  }
  store acc -> y[i * bm, j * bn]
}

schedule gemm_fp4 for nvidia-ada     { threads 256 warps 2 4 bm 128 bn 128 bk 64 }
schedule gemm_fp4 for apple-m-series { threads 256 warps 2 4 bm 64 bn 128 bk 32 }
```

```bash
# check it, lower it for CUDA and Metal, and run it on the CPU interpreter
cargo run -p lex-msl --example emit_lx -- crates/lex-front/lx/gemm_fp4.lx out/ m=128 n=128 k=64 --run
scripts/cuda_check.sh out/*.cu                                       # nvcc + NVRTC, no GPU needed
cargo run --release -p lex-rt --example lx_gemm_bench -- --tokens 512   # against the hand-written kernel (Metal or CUDA)
LEX_GEMM_LX=1 cargo run --release -p lex-rt --example serve -- --model qwen3.8:27b-mlx   # the server, prefill from .lx
```

More kernels and how to write your own: [`docs/guide.md`](docs/guide.md#8-write-a-kernel-in-lx-any-host).

## Crates

| Crate | What it is |
| --- | --- |
| `lex-ir` | Tile IR, target table (Apple, Hopper, Ada, CDNA3), planner, CPU reference |
| `lex-front` | Typed tile programs and the `.lx` surface; the linearity, effect and pipe checker; reference interpreter |
| `lex-msl` | Lowering from `.lx` programs and schedules to MSL and CUDA C, matrix units included; still hand-scheduled: causal attention, chunked gated-delta, int16 matvec, and the GEMM the model ships |
| `lex-metal` | Metal: compile, allocate, dispatch, time |
| `lex-cuda` | CUDA: NVRTC and the driver API through `dlopen`, no link-time dependency on a driver |
| `lex-rt` | Runtime: GGUF and safetensors, quantised formats, Llama and Qwen loops, speculation, tokenizer, server |
| `lex-bench` | Emit, verify, measure |

## Docs

[`design.md`](docs/design.md) — the language and the type system.
[`roadmap.md`](docs/roadmap.md) — the plan.
[`roadmap-weeks.md`](docs/roadmap-weeks.md) — the measurement log.
[`guide.md`](docs/guide.md) — examples and details.
[`cloud.md`](docs/cloud.md) — running on NVIDIA in the cloud.

## Licence

Copyright © 2026 Alfonso Sastre. Licensed under the EUPL, Version 1.2 —
see [`LICENSE`](LICENSE). The EUPL is copyleft, and Apache-2.0 and MIT are
not among its compatible licences: code from Apache- or MIT-licensed
projects cannot be copied into this one, and this code cannot be vendored
into an Apache-2.0 project. Everything here is written from the published
behaviour of other kernels, not from their source.
