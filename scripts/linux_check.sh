#!/usr/bin/env bash
# Type-check and lint the Linux-only code on the laptop.
#
#   scripts/linux_check.sh                 # check + clippy + the tests that
#   scripts/linux_check.sh --test          # need no GPU
#
# `lex-cuda`'s device module is `#[cfg(target_os = "linux")]`, so on macOS
# it is not compiled at all: a type error in it survives a clean
# `cargo clippy` here and is found by the cloud, twenty minutes and a VM
# later. `cargo check` needs the target's std but no linker and no GPU, so
# a container settles it in seconds.
#
# Homebrew's rust has no rustup and so no cross target, which is why this
# goes through Docker rather than `--target aarch64-unknown-linux-gnu`.
#
# The container's target directory is cached under ~/.cache and kept apart
# from the host's: sharing one makes the two toolchains rebuild over each
# other every time.
set -euo pipefail

IMAGE="${IMAGE:-rust:1-bookworm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="$HOME/.cache/lex-linux-check"
mkdir -p "$CACHE/target" "$CACHE/registry"

WHAT="check"
[ "${1:-}" = "--test" ] && WHAT="test"

# `--platform linux/arm64` so this is native on Apple Silicon rather than
# emulated, which is the difference between seconds and minutes.
docker run --rm --platform linux/arm64 \
  -v "$ROOT":/src:ro \
  -v "$CACHE/target":/target \
  -v "$CACHE/registry":/usr/local/cargo/registry \
  -w /src \
  -e CARGO_TARGET_DIR=/target \
  "$IMAGE" bash -c '
    # `|| true` after a grep would swallow the compiler\'"'"'s exit code and
    # leave this reporting success on a type error, so each step keeps its
    # own status and the filtering happens on the way out.
    set -eo pipefail
    quiet() { grep -vE "^\s+(Compiling|Checking|Downloaded|Updating|Adding|Downloading)" || [ $? = 1 ]; }
    rustup component add clippy >/dev/null 2>&1

    echo "=== cargo check (all targets)"
    cargo check --workspace --all-targets 2>&1 | quiet
    echo "=== clippy"
    cargo clippy --workspace --all-targets -- -D warnings 2>&1 | quiet
    if [ "'"$WHAT"'" = "test" ]; then
      echo "=== tests that need no GPU"
      cargo test --workspace 2>&1 | grep -E "^test result|^error|FAILED" || [ $? = 1 ]
    fi
    echo "=== ok"
  '
