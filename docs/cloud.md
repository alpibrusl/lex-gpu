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
