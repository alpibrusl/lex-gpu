#!/usr/bin/env bash
# Runs ON the NVIDIA VM (started by nvidia_test.sh), from the home directory
# with the source unpacked in ./lex-gpu. Writes everything to ~/results.
#
# 1. The machine: nvidia-smi, CPU, driver and CUDA versions.
# 2. The workspace tests: IR, checker, interpreter, emitters. The same suite
#    CI runs on Linux; lex-metal compiles to nothing off macOS.
# 3. The CUDA backend's tests, once a `lex-cuda` crate exists: skipped
#    (and said so) until then.
# 4. The baseline: Ollama's decode and prefill speed on this GPU for $MODELS,
#    at the same contexts as on the Mac (scripts/ollama_bench.py).
set -uo pipefail
MODELS="${MODELS:-llama3.2:1b llama3.1:8b}"
R="$HOME/results"
mkdir -p "$R"
cd "$HOME/lex-gpu"
fail=0
step() { echo; echo "=== $*"; }

step "waiting for the NVIDIA driver"
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi | tee "$R/nvidia-smi.txt" || { echo "no NVIDIA driver after 15 min"; exit 1; }
{ lscpu | head -20; nvcc --version 2>/dev/null || ls /usr/local | grep -i cuda; } > "$R/machine.txt"

step "Rust toolchain"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
fi
. "$HOME/.cargo/env"
rustc --version | tee -a "$R/machine.txt"

step "workspace tests (frontend, interpreter, emitters)"
cargo test --release --workspace 2>&1 | tee "$R/cargo-test.log" | grep -E "test result|FAILED|panicked"
[ "${PIPESTATUS[0]}" = 0 ] || fail=1

step "CUDA backend"
if [ -d crates/lex-cuda ]; then
  cargo test --release -p lex-cuda 2>&1 | tee "$R/cuda-test.log" | grep -E "test result|FAILED|panicked"
  [ "${PIPESTATUS[0]}" = 0 ] || fail=1
else
  echo "no crates/lex-cuda yet: skipped" | tee "$R/cuda-test.log"
fi

step "Ollama baseline"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh >/dev/null
fi
(ollama serve >"$R/ollama-serve.log" 2>&1 &)
for i in $(seq 1 30); do curl -s localhost:11434 >/dev/null && break; sleep 2; done
ollama --version | tee -a "$R/machine.txt"
for m in $MODELS; do
  ollama pull "$m" >/dev/null && \
    python3 scripts/ollama_bench.py --model "$m" --json "$R/ollama-${m//[:\/]/_}.json" \
      | tee -a "$R/ollama-bench.txt" || fail=1
done

step "done (failures: $fail)"
exit $fail
