# Testing on NVIDIA in Google Cloud (EU)

`scripts/gcp/nvidia_test.sh` rents one NVIDIA GPU in an EU region, runs the
tests and the Ollama baseline on it, copies the results to
`results/gcp/<time>-<gpu>/`, and deletes the VM.

```sh
gcloud auth login                                   # once; the script cannot prompt
GCP_PROJECT=<project> scripts/gcp/nvidia_test.sh    # L4, europe-west4 first
GCP_PROJECT=<project> GPU=a100 SPOT=1 scripts/gcp/nvidia_test.sh
```

## What it runs

`scripts/gcp/remote.sh`, on the VM, against the committed `HEAD`:

1. `nvidia-smi`, the CPU, and the CUDA and Rust versions (`machine.txt`).
2. `cargo test --release --workspace`: the IR, checker, interpreter and
   emitters. This is the same suite CI runs on Linux.
3. `cargo test -p lex-cuda`, once the CUDA backend exists. Until then the
   step is skipped, and the log says so.
4. Ollama's decode and prefill speeds on that GPU, for
   `llama3.2:1b` and `llama3.1:8b`, at 0, 512 and 1,440 positions
   (`scripts/ollama_bench.py`, the same script used on the Mac). That's the
   bar the CUDA backend has to meet.

## The first run: what an L4 actually is

2026-09-23, `g2-standard-8` Spot in europe-west4-b, driver 580.173.02,
CUDA 13.0. The 91 target-independent tests — IR, checker, interpreter,
MSL emitter goldens — pass on x86 Linux exactly as they do on the Mac.
The Mac runs 108; the other 17 are Metal-gated.

Ollama on that L4, which is the bar a CUDA backend has to meet:

| model | decode tok/s | prefill tok/s (0 / 512 / 1440) |
| --- | --- | --- |
| `llama3.2:1b` | 158–164 | 317 / 16,068 / 19,167 |
| `llama3.1:8b` | 48–50 | 98 / 3,106 / 2,935 |

**Read that next to the M4 Max, because they are opposite machines.** The
8B decodes at 48–50 here against 83–87 on the Mac, and prefills at 2,935
against ~900. An L4 has roughly half the memory bandwidth and several
times the arithmetic throughput.

Every choice in `lex-msl` was made against a bandwidth-bound machine: the
NVFP4 bit-layout decode, split-KV attention, the activation-traffic work
in the batched matmul. On an L4 the binding constraint is the other one,
so a backend that inherits Metal's schedule will be wrong here in a
specific and predictable direction. That is exactly the claim the
algorithm/schedule split makes — same algorithm, different schedule — and
it is now testable rather than asserted.

It also moves the target. On this hardware the interesting number is not
decode but **prefill**, where Metal is 5–8x short and an L4 has compute
to spare.

## GPUs and regions

| `GPU=` | Machine | GPU | Zones tried (in order) |
| --- | --- | --- | --- |
| `l4` (default) | `g2-standard-8` | 1× L4, 24 GB | europe-west4-a/b/c, europe-west1-b/c, europe-west3-a/b, europe-west2-a/b |
| `a100` | `a2-highgpu-1g` | 1× A100, 40 GB | europe-west4-a/b |
| `h100` | `a3-highgpu-1g` | 1× H100, 80 GB | europe-west4-b/c, europe-west1-b |

GPUs are often out of stock in one zone and free in the next, so the script
tries each zone until one accepts. `ZONES="..."` overrides the list. A new
project usually has a GPU quota of 0. Request `GPUS_ALL_REGIONS` and the
per-region quota for the GPU (for example `NVIDIA_L4_GPUS` in
europe-west4) in the console first.

## Cost guard rails

- The VM is deleted when the script exits, whether it succeeds, fails or you
  press Ctrl-C.
- The VM is also created with `--max-run-duration` (`MAX_RUN`, default 2h)
  and `--instance-termination-action DELETE`, so GCE deletes it even if the
  script dies.
- `SPOT=1` gives a Spot VM, roughly 60–70% cheaper, but it can be preempted.
- `KEEP=1` keeps the VM for debugging, and prints the command to delete it.
- The VM is labelled `purpose=lex-gpu-test`, so strays are easy to find:
  `gcloud compute instances list --filter=labels.purpose=lex-gpu-test`.
