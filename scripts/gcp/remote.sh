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
# The pre-built image carries the driver already; with no module for the
# kernel it booted there is nothing to wait for, only GPU time to lose.
if [ -d "$HOME/lex-target" ] && ! find "/lib/modules/$(uname -r)" -name 'nvidia.ko*' | grep -q .; then
  echo "no NVIDIA module for kernel $(uname -r) in the pre-built image; rebuild it (build_image.sh)"
  exit 1
fi
for i in $(seq 1 90); do nvidia-smi >/dev/null 2>&1 && break; sleep 10; done
nvidia-smi | tee "$R/nvidia-smi.txt" || { echo "no NVIDIA driver after 15 min"; exit 1; }
{ lscpu | head -20; nvcc --version 2>/dev/null || ls /usr/local | grep -i cuda; } > "$R/machine.txt"

step "Rust toolchain"
# The Deep Learning VM image ships CUDA and Python but no C compiler, and
# rustc needs one to link. Without this every build dies at `linker `cc`
# not found`, long after the image has convinced you it is a build machine.
if ! command -v cc >/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential
fi
cc --version | head -1 | tee -a "$R/machine.txt"
# A non-interactive shell does not read the profile that puts cargo on the
# PATH, so without this the pre-built image's toolchain looked absent and
# the installer ran again on every run.
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
fi
. "$HOME/.cargo/env"
rustc --version | tee -a "$R/machine.txt"
# A pre-built image (build_image.sh) carries a release build here; building
# into it recompiles only what changed since, not the whole workspace.
# Cargo decides by modification time, and `git archive` stamps every file
# with the commit's time -- newer than the image's build, so all of it
# would look changed. Files whose contents match the ones the image was
# built from are put back in the past; the rest keep their new time.
if [ -d "$HOME/lex-target" ]; then
  export CARGO_TARGET_DIR="$HOME/lex-target"
  if [ -f "$HOME/lex-src.sha256" ]; then
    sha256sum -c "$HOME/lex-src.sha256" 2>/dev/null | sed -n 's/: OK$//p' |
      xargs -r -d '\n' touch -d 2000-01-01
  fi
  echo "CARGO_TARGET_DIR=$CARGO_TARGET_DIR (pre-built)" | tee -a "$R/machine.txt"
fi

# JOB: one command, and nothing else (nvidia_test.sh copied it to
# ~/job.sh). The driver and the toolchain above are all it needs; the
# pre-built image already holds the release build and qwen3.8.
if [ -n "${JOB:-}" ]; then
  step "job"
  export OLLAMA_MODELS="$HOME/.ollama/models"
  # The service would keep whatever it last served on the GPU.
  sudo systemctl stop ollama 2>/dev/null || true
  cat "$HOME/job.sh"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$R/machine.txt"
  bash -e -o pipefail "$HOME/job.sh" 2>&1 | tee "$R/job.log"
  status=${PIPESTATUS[0]}
  echo "=== job exit $status"
  exit "$status"
fi

# SPEED=1 is for a question about speed: it skips everything that checks
# correctness or measures Ollama, which is most of an hour, and keeps the
# Qwen fetch and the timing. It is not a test run and says nothing about
# whether the numbers it times are right -- run without it for that.
if [ -z "${SPEED:-}" ]; then
step "workspace tests (frontend, interpreter, emitters)"
# The Qwen goldens take 17-25 minutes on an L4, and with QWEN=1 the step
# below runs them anyway. They used to skip here for want of the model;
# the pre-built image carries it, and one run paid for them twice.
skip=()
if [ -n "${QWEN:-}" ]; then
  for t in $(cargo test --release -p lex-rt --test qwen_golden -- --list 2>/dev/null | sed -n 's/: test$//p'); do
    skip+=(--skip "$t")
  done
  echo "the Qwen goldens (${#skip[@]} args) run in the QWEN step, not here"
fi
cargo test --release --workspace -- --exact "${skip[@]}" 2>&1 | tee "$R/cargo-test.log" | grep -E "test result|FAILED|panicked"
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
# The installer enables a systemd service, so this usually loses the port
# to it and exits -- which is fine, the service answers. It stays for the
# case where the binary was already present and no service was set up.
(ollama serve >"$R/ollama-serve.log" 2>&1 &)
for i in $(seq 1 30); do curl -s localhost:11434 >/dev/null && break; sleep 2; done
ollama --version | tee -a "$R/machine.txt"
for m in $MODELS; do
  ollama pull "$m" >/dev/null && \
    python3 scripts/ollama_bench.py --model "$m" --json "$R/ollama-${m//[:\/]/_}.json" \
      | tee -a "$R/ollama-bench.txt" || fail=1
done

# The milestone: not a kernel against the interpreter, but a whole model
# through the same `lex-front` programs the Mac runs, on an NVIDIA device.
# Everything up to here says the emitted CUDA compiles; only a real device
# says the numbers are right.
# Where this Ollama version keeps its blobs is not worth another guess:
# it is not $HOME/.ollama on Linux, and on 0.34.4 it is not the ollama
# user's home either. Two runs went on guesses. Ask the filesystem.
step "locate the Ollama model store"
STORE=$(sudo find / -xdev -type d -path "*/manifests/registry.ollama.ai" 2>/dev/null | head -1)
if [ -n "$STORE" ]; then
  export OLLAMA_MODELS="$(dirname "$(dirname "$STORE")")"
  echo "OLLAMA_MODELS=$OLLAMA_MODELS" | tee -a "$R/machine.txt"
  # Recorded so the next change to this can be made from fact.
  sudo ls -la "$OLLAMA_MODELS" 2>&1 | tee -a "$R/machine.txt"
  # The store belongs to the `ollama` service user and its home is not
  # world-readable. The installer adds us to the `ollama` group, which
  # does nothing for a shell whose group membership was fixed at login --
  # so the files are there, `sudo` reads them, and we do not. Open the
  # whole chain, not just the store: it was `/usr/share/ollama` itself
  # that blocked, two runs after the store was already being found.
  sudo chmod a+rX /usr/share/ollama /usr/share/ollama/.ollama 2>/dev/null || true
  sudo chmod -R a+rX "$OLLAMA_MODELS" 2>/dev/null || true
  # Prove it from this shell, since that is the process that has to read.
  test -r "$OLLAMA_MODELS/manifests/registry.ollama.ai/library/llama3.2/1b" \
    && echo "manifest readable as $(whoami)" | tee -a "$R/machine.txt" \
    || echo "manifest STILL unreadable as $(whoami)" | tee -a "$R/machine.txt"
else
  echo "no Ollama model store anywhere on this disk" | tee -a "$R/machine.txt"
fi

step "a model on CUDA"
# The baseline step already pulled this, and the runtime reads the model
# store directly -- it needs no server. So the pull here is a fallback for
# a run that skipped the baseline, and its failure must not block the
# milestone: one run died on exactly that, `ollama pull` refusing because
# the server had fallen over, with the model sitting in the store.
ollama pull llama3.2:1b >/dev/null 2>&1 || echo "pull failed; using whatever is in the store"
# Ask Ollama which blob the tag resolves to rather than reconstructing
# the manifest path. The store layout changed between versions -- 0.34.4
# grew a `metadata/` directory and the old
# manifests/registry.ollama.ai/library/<repo>/<tag> guess stopped
# resolving -- and three runs were lost to guessing at it. `--modelfile`
# prints `FROM <blob>`, which is the file we actually want.
GGUF=$(ollama show llama3.2:1b --modelfile 2>/dev/null | awk '/^FROM /{print $2; exit}')
echo "gguf: ${GGUF:-<not resolved, falling back to the manifest>}" | tee -a "$R/machine.txt"
# The layout, recorded either way, so the fallback can be fixed from fact.
sudo find "${OLLAMA_MODELS:-/usr/share/ollama/.ollama/models}/manifests" -maxdepth 4 \
  2>/dev/null | head -20 | tee -a "$R/machine.txt"

# "The capital of France is" -- greedy, so the continuation is fixed and
# `scripts/lex_vs_ollama.py` has the reference this is checked against.
if [ -n "$GGUF" ] && [ -r "$GGUF" ]; then
  MODEL_ARG=(--gguf "$GGUF")
else
  MODEL_ARG=(--model llama3.2:1b)
fi
cargo run --release -p lex-rt --example generate -- \
  "${MODEL_ARG[@]}" --ids 128000,791,6864,315,9822,374 --steps 16 --top 5 \
  2>&1 | tee "$R/cuda-generate.txt" | tail -20
[ "${PIPESTATUS[0]}" = 0 ] || fail=1

fi

# The hard model on the other backend: 48 of 64 layers carry a recurrent
# state instead of a KV cache, the weights are NVFP4, and there is a
# multi-token-prediction head. Ollama cannot run it on this machine --
# it is an MLX build and that engine is macOS-only -- so there is no
# baseline here and the check is the golden file, an f32 reference the
# Metal tests are held to as well. Opt-in: 14.5 GB to pull.
if [ -n "${QWEN:-}" ]; then
  step "the hard model on CUDA"
  # Not `ollama pull`: the client refuses this one here -- "this model
  # requires MLX support, but the MLX runtime is not available" -- and
  # that is the client's check, not the registry's. The weights are
  # ordinary blobs behind ordinary HTTP and `lex-rt` reads the store
  # directly, so Ollama's opinion about what this machine can execute is
  # not one we need. Into our own home, which also skips the permission
  # dance the service user's store needs.
  # Point the resolver at the store we are about to fill, rather than
  # trusting them to agree: the earlier step exported OLLAMA_MODELS to
  # the service user's store, which takes priority over $HOME, so the
  # first attempt fetched 18 GB into one place and looked in another.
  export OLLAMA_MODELS="$HOME/.ollama/models"
  python3 scripts/ollama_fetch.py qwen3.8:27b-mlx \
    --root "$OLLAMA_MODELS" 2>&1 | tail -12 | tee -a "$R/qwen-cuda.log"
  [ "${PIPESTATUS[0]}" = 0 ] || fail=1
  # Ollama holds the baseline models on the GPU -- 6.6 GB of a 23 GB
  # card after llama3.1:8b -- and this one needs about 15.5.
  sudo systemctl stop ollama 2>/dev/null || true
  nvidia-smi --query-gpu=memory.total,memory.used --format=csv | tee -a "$R/machine.txt"

  # The matvec structures, timed against the emitted one on this card
  # (no model needed; synthetic weights, four matrices to defeat the L2).
  # A burst at full clock, a model-sized working set, and a sustained run
  # long enough to reach the 72 W cap -- where the model decodes.
  for args in "--mats 4" "--mats 64" "--mats 4 --reps 400"; do
    tag=$(echo "$args" | tr -d ' -')
    cargo run --release -p lex-cuda --example mv_variants -- $args 2>&1 \
      | tee "$R/mv-variants-$tag.log" | grep -E "matrices|variant|GB/s|read|emitted|unroll|warp|quant|int8"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
    grep -q "int8_row" "$R/mv-variants-$tag.log" || { echo "mv_variants ($args) printed no table"; fail=1; }
  done

  if [ -n "${SPEED:-}" ]; then
    # Speed runs still check what they time: the kernels against the
    # interpreter (the GEMM among them), and prefill against stepping --
    # a minute or two, not the golden suite's fifteen.
    cargo test --release -p lex-cuda 2>&1 | tee "$R/cuda-test.log" | grep -E "test result|FAILED|panicked"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
    cargo test --release -p lex-rt --test qwen_golden prefill -- --nocapture 2>&1 \
      | tee "$R/qwen-prefill-golden.log" | grep -E "prefill vs|test result|panicked|SKIPPED"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
    grep -q "test result: ok. [1-9]" "$R/qwen-prefill-golden.log" \
      || { echo "no prefill golden ran"; fail=1; }
    # A golden that skips reports "ok" and checked nothing: when the model
    # did not arrive, all four did, and the line above passed.
    if grep -q SKIPPED "$R/qwen-prefill-golden.log"; then
      echo "a prefill golden skipped -- the model is not there"
      fail=1
    fi
  fi

  if [ -z "${SPEED:-}" ]; then
    QOUT=$(cargo test --release -p lex-rt --test qwen_golden -- --nocapture 2>&1)
    echo "$QOUT" >> "$R/qwen-cuda.log"
    echo "$QOUT" | grep -E "test result|worst|SKIPPED|panicked|differs" || true
    # A suite that runs nothing is not a pass. The first version of this
    # step reported success off `test result: ok. 0 passed`, because the
    # test file was still gated to macOS and compiled to nothing.
    if ! echo "$QOUT" | grep -qE "test result: ok\. [1-9]"; then
      echo "no qwen test actually ran -- gated out, or the model is missing"
      fail=1
    fi
    # And a test that skips is not a test that ran. These print SKIPPED and
    # return Ok when the model is absent, which is right for a laptop with
    # no checkpoint and wrong here, where fetching it is the point.
    if echo "$QOUT" | grep -q SKIPPED; then
      echo "a qwen test skipped -- the model was asked for and is not there"
      fail=1
    fi
  fi

  cargo run --release -p lex-rt --example mtp -- --steps 32 --depth 1 2>&1 \
    | tee -a "$R/qwen-cuda.log" | grep -E "tok/s|offset 1"
  [ "${PIPESTATUS[0]}" = 0 ] || fail=1

  # Where the decode step's time goes, per call site, from events between
  # launches queued as normal -- and what a verify costs in steps, which
  # is why speculation loses here (2.7 steps for two tokens, against 1.15
  # on Metal). At two contexts, because attention's share grows with it.
  # Clocks, power and temperature through the profiles: the L4 is capped
  # at 72 W, and the decode matvec runs half as long again inside the model
  # as it does alone in mv_variants.
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active \
    --format=csv -lms 500 > "$R/clocks.csv" 2>&1 &
  SMI=$!

  # And again with LEX_NARROW=1, the loads as they were before the
  # emitter widened them: the same binary both ways, so the difference is
  # the loads and nothing else about the build or the machine.
  for narrow in "" 1; do
    for ctx in 0 1024; do
      log="$R/qwen-profile-ctx$ctx${narrow:+-narrow}.log"
      echo "--- context $ctx${narrow:+, LEX_NARROW=1}"
      LEX_NARROW=$narrow cargo run --release -p lex-rt --example qwen_profile -- \
        --tokens 32 --context $ctx --verify 3 2>&1 | tee "$log" \
        | grep -E "ms/token|verify of|timed sum|batch path|prefill of"
      [ "${PIPESTATUS[0]}" = 0 ] || fail=1
      grep -q "prefill of" "$log" \
        || { echo "qwen_profile at context $ctx printed no prefill line"; fail=1; }
    done
  done

  kill $SMI 2>/dev/null || true

  # Prefill by chunk size: 8 is the batched matvec alone, 64 and 128 go
  # through the tensor-core GEMM. The per-kernel table is in each log.
  for chunk in 8 64 128; do
    log="$R/qwen-prefill-chunk$chunk.log"
    LEX_PREFILL_CHUNK=$chunk cargo run --release -p lex-rt --example qwen_profile -- \
      --tokens 4 --verify 3 --prefill 512 2>&1 | tee "$log" | grep -E "prefill of"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
    grep -q "prefill of" "$log" \
      || { echo "no prefill line at chunk $chunk"; fail=1; }
  done

  # Every schedule constant in this repository was chosen on an M4 Max.
  # `bo` and `threads` set rows per simdgroup and therefore register
  # pressure, and the right value is a property of the register file, not
  # of the kernel -- so there is no reason the Metal answer should be the
  # Ada one, and nobody has ever looked.
fi

# `bo` is the one that matters and the first sweep missed it: the
# decode table takes BO from the environment, so three runs varying
# only THREADS all measured the Metal-tuned default of 8 and reported
# the same ~100 GB/s. The batched table varies `bo` itself and reaches
# twice that, which is what says the decode kernel is the problem.
if [ -z "${SPEED:-}" ]; then
step "the schedule, swept on this card"
for th in 128 256; do
  for bo in 4 8 16 32 64; do
    echo "--- THREADS=$th BO=$bo"
    # No `head`: the whole table goes to the log. Truncating it is how
    # the rows that mattered were lost last time.
    THREADS=$th BO=$bo cargo run --release -p lex-rt --example matvec 2>&1 \
      | tee -a "$R/matvec-cuda.log" | grep -E "qwen (gate|down)|1b (gate|down)"
    [ "${PIPESTATUS[0]}" = 0 ] || fail=1
  done
done
# A step that measured nothing is not a step that passed. This sweep sat
# inside the QWEN block once, a run without it skipped the whole thing,
# and the harness reported `failures: 0` -- which cost a VM to discover.
rows=$(grep -c "qwen gate/up (nvfp4)" "$R/matvec-cuda.log" 2>/dev/null || true)
if [ "${rows:-0}" -lt 10 ]; then
  echo "the sweep produced ${rows:-0} rows, expected 10 -- it did not run"
  fail=1
fi

fi

step "done (failures: $fail)"
exit $fail
