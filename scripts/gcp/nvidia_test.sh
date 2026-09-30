#!/usr/bin/env bash
# Run the test suite and the Ollama baseline on an NVIDIA GPU in a Google
# Cloud EU region, bring the results home, and delete the VM.
#
#   gcloud auth login                       # once; the script cannot prompt
#   GCP_PROJECT=my-project scripts/gcp/nvidia_test.sh
#   GCP_PROJECT=my-project GPU=a100 scripts/gcp/nvidia_test.sh
#
# What runs on the VM is scripts/gcp/remote.sh, against the committed HEAD
# (`git archive`): uncommitted changes are not tested.
#
# Knobs (environment):
#   GCP_PROJECT  required. Billing goes here.
#   GPU          l4 (default, g2-standard-8, 24 GB), a100 (a2-highgpu-1g,
#                40 GB) or h100 (a3-highgpu-1g, 80 GB). These machine types
#                come with their GPU attached; no --accelerator flag.
#   ZONES        space-separated zones to try in order. EU first, then US:
#                GPUs are often out of stock in one zone and free in the
#                next, and L4 capacity in europe-west was unavailable for
#                an entire afternoon.  On 2026-09-28 Spot L4s were out in every
#                EU and US region at once and Tokyo had them, so Asia
#                follows. Set it explicitly to stay in one
#                region -- these runs carry no user data, only public
#                model weights and this repository, which is why leaving
#                the EU is allowed here and would not be for everything.
#   SPOT=0       On-demand instead of the default Spot. Spot is 60-70%
#                cheaper and can be preempted mid-run; use SPOT=0 only for
#                a run long enough that losing it matters.
#   MODELS       Ollama models for the baseline (default: llama3.2:1b llama3.1:8b).
#   IMAGE_FAMILY boot from this family in GCP_PROJECT (default lex-gpu-l4,
#                made by build_image.sh), else the stock Deep Learning image.
#   LEX_INT8=0   run with the float decode matvec instead of the default
#                int16 one (lex_msl::int8), to measure one against the other.
#   SPEED=1      only Qwen's timing (qwen_profile, mtp): no test suites, no
#                Ollama baseline, no sweep. Implies QWEN=1. Minutes instead
#                of most of an hour, for a question about speed.
#   QWEN=1       also run qwen3.8:27b-mlx on the GPU -- the hybrid
#                gated-delta / NVFP4 / draft-head model. 14.5 GB to pull,
#                and Ollama cannot run it here to compare against (MLX is
#                macOS-only), so the check is our own golden file.
#   JOB          run this one shell command on the GPU and nothing else --
#                no test suites, no Ollama baseline, no Llama models:
#                  JOB='cargo test --release -p lex-rt --test qwen_golden'
#                  JOB='cargo run --release -p lex-rt --example qwen_profile -- --tokens 8'
#                It runs in the source tree, from the pre-built image (Rust,
#                a release build, qwen3.8 on disk), with Ollama's service
#                stopped so its models do not hold GPU memory. Its output
#                comes home as job.log; the exit status is the job's.
#   MAX_RUN      hard cap on the VM's life (default 2h). GCE deletes the VM
#                when it expires, even if this script is killed.
#   KEEP=1       leave the VM running afterwards (debugging); you delete it.
#
# Exit status: the remote run's, or 75 when Google preempted the Spot VM
# (retry: `for i in 1 2 3; do ...nvidia_test.sh; [ $? = 75 ] || break; done`).
#
# Cost guard rails: the VM is deleted on exit (success, failure or Ctrl-C),
# and independently by GCE after MAX_RUN via --max-run-duration.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT to the Google Cloud project to bill}"
GPU="${GPU:-l4}"
# Spot by default: 60-70% cheaper, and this is a test harness whose runs
# are all repeatable. A preemption costs the run, not the results, and
# the alternative was quietly paying on-demand for every one of them --
# which is what happened for a day because the default was 0 and the
# flag was never passed.
SPOT="${SPOT:-1}"
MAX_RUN="${MAX_RUN:-2h}"
KEEP="${KEEP:-0}"
MODELS="${MODELS:-llama3.2:1b llama3.1:8b}"
[ -n "${SPEED:-}" ] && QWEN=1

case "$GPU" in
  l4)   MACHINE=g2-standard-8; DEFAULT_ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west3-b europe-west2-a europe-west2-b us-central1-a us-central1-b us-central1-c us-east1-c us-east1-d us-east4-a us-east4-c us-west1-a us-west1-b us-west4-a asia-northeast1-a asia-northeast1-c asia-east1-a asia-east1-b asia-east1-c asia-northeast3-a asia-northeast3-b asia-southeast1-a asia-southeast1-b asia-southeast1-c asia-south1-a asia-south1-b asia-south1-c" ;;
  a100) MACHINE=a2-highgpu-1g; DEFAULT_ZONES="europe-west4-a europe-west4-b us-central1-a us-central1-b us-central1-c us-east1-b" ;;
  h100) MACHINE=a3-highgpu-1g; DEFAULT_ZONES="europe-west4-b europe-west4-c europe-west1-b us-central1-a us-east4-a us-east5-a" ;;
  *) echo "GPU must be l4, a100 or h100" >&2; exit 2 ;;
esac
ZONES="${ZONES:-$DEFAULT_ZONES}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
NAME="lex-gpu-${GPU}-${STAMP}"
OUT="$ROOT/results/gcp/${STAMP}-${GPU}"
mkdir -p "$OUT"
gc() { gcloud --project "$GCP_PROJECT" --quiet "$@"; }

# Newest Deep Learning VM image with CUDA 12 and the NVIDIA driver installer.
FAMILY="$(gcloud compute images list --project deeplearning-platform-release \
  --filter='family~^common-cu12.*ubuntu' --format='value(family)' | sort | tail -1)"
[ -n "$FAMILY" ] || { echo "no common-cu12 ubuntu image family found" >&2; exit 1; }
# The image scripts/gcp/build_image.sh makes on a CPU VM -- Rust, a release
# build, Ollama, the models already on it -- when there is one: installing
# all that here took the first 15-20 minutes of every run at GPU prices.
IMAGE_FAMILY="${IMAGE_FAMILY:-lex-gpu-l4}"
if gc compute images describe-from-family "$IMAGE_FAMILY" >/dev/null 2>&1; then
  IMG=(--image-project "$GCP_PROJECT" --image-family "$IMAGE_FAMILY")
  echo "image: $GCP_PROJECT/$(gc compute images describe-from-family "$IMAGE_FAMILY" --format='value(name)') (pre-built)"
else
  IMG=(--image-project deeplearning-platform-release --image-family "$FAMILY")
  echo "image family: deeplearning-platform-release/$FAMILY (no pre-built $IMAGE_FAMILY; run build_image.sh)"
fi

ZONE=""
# Where the VM named $NAME is, if it exists anywhere in the project. Asked
# by name rather than remembered, because a request is made per *region*
# and Google picks the zone: an interrupt between the request succeeding
# and this script learning the zone would otherwise leave a GPU running
# that nothing knows to delete -- the expensive mistake here.
where() {
  gc compute instances list --filter="name=$NAME" --format="value(zone.basename())" 2>/dev/null \
    | head -1 || true
}
cleanup() {
  local z
  z="$(where)"
  [ -n "$z" ] || return 0
  if [ "$KEEP" = 1 ]; then
    echo "KEEP=1: $NAME is still running in $z. Delete it with:"
    echo "  gcloud --project $GCP_PROJECT compute instances delete $NAME --zone $z"
    return
  fi
  gc compute instances delete "$NAME" --zone "$z" >/dev/null 2>&1 && echo "deleted $NAME in $z"
}
# EXIT alone does not fire when the shell is killed by a signal.
trap cleanup EXIT INT TERM HUP

spot_flags=()
[ "$SPOT" = 1 ] && spot_flags=(--provisioning-model=SPOT)
# One request per region, not per zone: Google places the VM in whichever
# zone of the region has capacity, so a region is one question instead of
# three or four asked in turn -- and never more than one VM, which asking
# every zone at once would risk. Nothing publishes free GPU capacity; asking
# for a machine is the only way to find out. Regions in the order their
# zones appear in $ZONES, so EU still comes first.
REGIONS=$(for z in $ZONES; do echo "${z%-*}"; done | awk '!seen[$0]++')
for r in $REGIONS; do
  # Only this region's zones from $ZONES: an explicit ZONES list is a
  # restriction, and the region's other zones are not in it.
  echo "trying $MACHINE in $r"
  # An `if`, not `[ ] && printf`: under `set -e` the loop's status is its
  # last test's, a zone of another region fails it, and the assignment
  # then ends the script -- which is how the first run of this died
  # silently after "trying ... in europe-west4".
  allow=$(for z in $ZONES; do if [ "${z%-*}" = "$r" ]; then printf '%s=allow,' "$z"; fi; done)
  if gc compute instances bulk create --region "$r" --count 1 \
      --predefined-names "$NAME" \
      --location-policy "${allow%,}" \
      --machine-type "$MACHINE" \
      --maintenance-policy TERMINATE ${spot_flags[@]+"${spot_flags[@]}"} \
      --max-run-duration "$MAX_RUN" --instance-termination-action DELETE \
      "${IMG[@]}" \
      --boot-disk-size 150GB --boot-disk-type pd-ssd \
      --metadata install-nvidia-driver=True \
      --labels purpose=lex-gpu-test 2>"$OUT/create-$r.log" >/dev/null; then
    ZONE="$(where)"
    [ -n "$ZONE" ] && break
    echo "  $r: created, but $NAME is not listed anywhere" >&2
    continue
  fi
  # One line per region: the error code, not the last two lines of a YAML
  # dump that cut the reason in half.
  why=$(grep -oE "code: [A-Z_]+|currently unavailable|[Qq]uota [^.]*" "$OUT/create-$r.log" | head -1 || true)
  echo "  $r: ${why:-failed, see $OUT/create-$r.log}" >&2
done
[ -n "$ZONE" ] || { echo "no region had capacity (or quota) for $MACHINE; see $OUT/create-*.log" >&2; exit 1; }
echo "$NAME up in $ZONE"

# SSH comes up before the driver finishes installing; remote.sh waits for it.
for i in $(seq 1 60); do
  gc compute ssh "$NAME" --zone "$ZONE" --command true >/dev/null 2>&1 && break
  sleep 10
done

git -C "$ROOT" archive --format=tar.gz -o "$OUT/src.tar.gz" HEAD
gc compute scp --zone "$ZONE" "$OUT/src.tar.gz" "$NAME:~/src.tar.gz"
# A job travels as a file, not inside the ssh command line below, so its
# own quotes need no escaping.
if [ -n "${JOB:-}" ]; then
  printf '%s\n' "$JOB" > "$OUT/job.sh"
  gc compute scp --zone "$ZONE" "$OUT/job.sh" "$NAME:~/job.sh"
fi
# A failing run must still bring its logs home: no errexit from here on.
set +e
gc compute ssh "$NAME" --zone "$ZONE" --command \
  "mkdir -p lex-gpu && tar -xzf src.tar.gz -C lex-gpu && MODELS='$MODELS' QWEN='${QWEN:-}' SPEED='${SPEED:-}' LEX_INT8='${LEX_INT8:-}' JOB='${JOB:+1}' bash lex-gpu/scripts/gcp/remote.sh" \
  2>&1 | tee "$OUT/remote.log"
status=${PIPESTATUS[0]}
gc compute scp --zone "$ZONE" --recurse "$NAME:~/results/*" "$OUT/" || true
echo "results in $OUT (remote exit $status)"
# A Spot VM Google took back is not a failed run: it is no run, and says
# nothing about the code. Exit 75 (EX_TEMPFAIL) so a caller can try again
# -- three runs on 2026-09-29 were preempted seven minutes in, and each sat
# out its SSH timeout before exiting like any other failure.
if [ "$status" != 0 ] && [ -n "$(gc compute operations list \
    --filter="operationType=compute.instances.preempted AND targetLink~/$NAME\$" \
    --format='value(name)' 2>/dev/null)" ]; then
  echo "$NAME was preempted: no result, exit 75 to retry"
  exit 75
fi
exit "$status"
