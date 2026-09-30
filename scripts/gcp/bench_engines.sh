#!/usr/bin/env bash
# lex against vLLM on this GPU, Qwen3.8-27B, same prompts -- one engine at a
# time, because a 24 GB card holds one copy of the model.
#
#   GCP_PROJECT=<project> JOB='bash scripts/gcp/bench_engines.sh' scripts/gcp/nvidia_test.sh
#
# Needs the pre-built image (build_image.sh): the release build, qwen3.8 in
# Ollama's store for lex, and vLLM (~/vllm) with NVIDIA's NVFP4
# checkpoint. Results go to ~/results/engines.jsonl, one line an engine.
#
# vLLM's first try is what serves a user: the draft head speculating two
# tokens (lex's default depth) on CUDA graphs. The checkpoint is 21.9 GB
# against the card's 23, so if that does not start it tries eager mode,
# then plain decode, and says which one ran.
set -uo pipefail
cd "$HOME/lex-gpu"
R="$HOME/results"
mkdir -p "$R"

wait_up() { # url, seconds, the server's pid: gives up when it dies, not
  # after the whole wait -- a vLLM that runs out of memory at start would
  # otherwise hold the GPU for its full timeout.
  for _ in $(seq 1 "$2"); do
    curl -sf "$1/v1/models" >/dev/null && return 0
    kill -0 "$3" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

echo "=== lex"
cargo build --release -q -p lex-rt --example serve
LEX_NO_PREFIX_CACHE=1 target/release/examples/serve --model qwen3.8:27b-mlx --port 8094 \
  >"$R/lex-serve.log" 2>&1 &
LEX=$!
if wait_up http://localhost:8094 900 "$LEX"; then
  python3 scripts/engine_bench.py --url http://localhost:8094 --model lex --name lex \
    --json "$R/engines.jsonl" 2>&1 | tee -a "$R/engines.log"
  python3 scripts/engine_bench.py --url http://localhost:8094 --model lex --name lex-medium \
    --extra '{"reasoning_effort":"medium"}' --json "$R/engines.jsonl" 2>&1 | tee -a "$R/engines.log"
else
  echo "lex did not come up; see lex-serve.log" | tee -a "$R/engines.log"
fi
kill "$LEX" 2>/dev/null; wait "$LEX" 2>/dev/null
nvidia-smi --query-gpu=memory.used --format=csv,noheader

echo "=== $(cat "$HOME/vllm-version.txt" 2>/dev/null)"
SPEC='{"method":"mtp","num_speculative_tokens":2}'
try_vllm() { # name, then extra vLLM arguments
  local name=$1; shift
  "$HOME/vllm/bin/vllm" serve "$HOME/hf/Qwen3.8-27B-NVFP4" --port 8000 \
    --served-model-name qwen3.8 \
    --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.95 \
    --kv-cache-dtype fp8_e4m3 --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' "$@" >"$R/vllm-$name.log" 2>&1 &
  local pid=$!
  if wait_up http://localhost:8000 1500 "$pid"; then
    echo "vLLM up as $name" | tee -a "$R/engines.log"
    python3 scripts/engine_bench.py --url http://localhost:8000 --model qwen3.8 --name "vllm-$name" \
      --json "$R/engines.jsonl" 2>&1 | tee -a "$R/engines.log"
    kill "$pid"; wait "$pid" 2>/dev/null
    return 0
  fi
  echo "vLLM as $name did not come up" | tee -a "$R/engines.log"
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
  return 1
}
try_vllm mtp2 --speculative-config "$SPEC" \
  || try_vllm mtp2-eager --speculative-config "$SPEC" --enforce-eager \
  || try_vllm plain-eager --enforce-eager
# The plain one as well when speculation ran: it separates the kernels
# from the speculation.
grep -q "vllm-mtp2" "$R/engines.jsonl" 2>/dev/null && try_vllm plain
echo "=== results"
cat "$R/engines.jsonl" 2>/dev/null
