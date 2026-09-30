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
| `qwen3.8:27b-mlx` | M4 Max, speculating, greedy / sampled | 52.8 / 51.1 | 56.8 / 53.2 |
| `llama3.1:8b` | M4 Max | 79.2 | 86.0 |
| `llama3.2:1b` | M4 Max | 233.7 | 261.9 |
| `llama3.2:1b` | NVIDIA L4 | 124.4 | 162.9 |
| `qwen3.8:27b-mlx` | NVIDIA L4, plain / speculating | 15.1 / 24.4 | — (MLX build) |

Prefill, `qwen3.8:27b-mlx`, 512 tokens: **226 tok/s** on the M4 Max
(221 at 2048) against Ollama's 250–260, and **245** on the L4.

Correctness: Llama gives Ollama's tokens exactly (96/96 on both sizes),
MiMo 16/16; Qwen3.8 passes its whole golden suite on Metal and on the L4,
worst log-prob difference 0.00077 against an f32 reference (tolerance
0.02).

Qwen decode on the Mac is at 0.93x greedy and 0.96x sampled: a plain step
matches Ollama's (35 ms), and what remains is in the speculation cycle. How each number was
measured, and what is being tried next, is in
[`docs/roadmap-weeks.md`](docs/roadmap-weeks.md).

Not done yet: the `.lx` surface syntax covers two kernels (everything the
runtime runs is built through the Rust IR API, typed and checked), and the
matrix-unit kernels are hand-scheduled rather than lowered. See
[`docs/guide.md`](docs/guide.md#not-built-yet).

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

## Crates

| Crate | What it is |
| --- | --- |
| `lex-ir` | Tile IR, target table (Apple, Hopper, Ada, CDNA3), planner, CPU reference |
| `lex-front` | Typed tile programs and the `.lx` surface; the linearity, effect and pipe checker; reference interpreter |
| `lex-msl` | Lowering to MSL and CUDA C; hand-scheduled GEMM, chunked gated-delta and int16 matvec kernels |
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
