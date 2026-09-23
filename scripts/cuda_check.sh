#!/usr/bin/env bash
# Compile emitted CUDA on a machine with no NVIDIA GPU.
#
#   scripts/cuda_check.sh out/*.cu              # compile and report registers
#   ARCH=sm_90 scripts/cuda_check.sh out/foo.cu # another target
#
# `nvcc` needs a GPU to *run* a kernel, not to compile one, and NVIDIA ships
# CUDA for arm64 — so both it and `ptxas` run natively on an Apple Silicon
# Mac in a container. That puts three of the four test layers on the laptop
# and leaves the cloud only what a real device can answer:
#
#   golden files   the emitted text, diffed        no toolchain
#   interpreter    that the program is correct     no toolchain
#   this script    that it compiles and assembles  docker
#   an L4          races, numerics, speed          the cloud
#
# `ptxas -v` reports register count and spill bytes directly. On Metal that
# had to be inferred from throughput cliffs, and this repo misdiagnosed it
# twice doing so; here the assembler simply says.
set -euo pipefail

ARCH="${ARCH:-sm_89}"          # Ada, which is what an L4 is
IMAGE="${IMAGE:-nvidia/cuda:12.6.3-devel-ubuntu24.04}"

[ $# -gt 0 ] || { echo "usage: $0 <file.cu> [...]" >&2; exit 2; }

# Docker on macOS shares /Users but not /private/tmp, and a bind mount of an
# unshared path is silently empty rather than an error — so stage into a
# directory under $HOME and copy the sources in.
WORK="$(mktemp -d "$HOME/.cache/lex-cuda-check.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
cp "$@" "$WORK/"

docker run --rm --platform linux/arm64 -v "$WORK":/w -w /w "$IMAGE" bash -c '
  set -e
  fail=0
  for f in *.cu; do
    echo "=== $f ==="
    if nvcc -arch='"$ARCH"' -ptx "$f" -o "${f%.cu}.ptx"; then
      ptxas -arch='"$ARCH"' -v "${f%.cu}.ptx" -o /dev/null 2>&1 \
        | grep -E "Used|spill" | sed "s/^/  /"
    else
      fail=1
    fi
  done
  exit $fail
' 2>&1 | grep -vE "NVIDIA Driver was not detected|Container Toolkit|docs.nvidia.com|^=*$|CUDA Version|Container image|NVIDIA Deep Learning|developer.nvidia.com|A copy of this license|^== CUDA ==|By pulling|^$"
