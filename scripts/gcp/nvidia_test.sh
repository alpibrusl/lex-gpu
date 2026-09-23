#!/usr/bin/env bash
# Run the test suite and the Ollama baseline on an NVIDIA GPU in a Google
# Cloud EU region, bring the results home, and delete the VM.
#
#   gcloud auth login                       # once; the script cannot prompt
#   GCP_PROJECT=my-project scripts/gcp/nvidia_test.sh
#   GCP_PROJECT=my-project GPU=a100 SPOT=1 scripts/gcp/nvidia_test.sh
#
# What runs on the VM is scripts/gcp/remote.sh, against the committed HEAD
# (`git archive`): uncommitted changes are not tested.
#
# Knobs (environment):
#   GCP_PROJECT  required. Billing goes here.
#   GPU          l4 (default, g2-standard-8, 24 GB), a100 (a2-highgpu-1g,
#                40 GB) or h100 (a3-highgpu-1g, 80 GB). These machine types
#                come with their GPU attached; no --accelerator flag.
#   ZONES        space-separated zones to try in order; EU only by default.
#                GPUs are often out of stock in one zone and free in the next.
#   SPOT=1       Spot VM: ~60-70% cheaper, can be preempted mid-run.
#   MODELS       Ollama models for the baseline (default: llama3.2:1b llama3.1:8b).
#   MAX_RUN      hard cap on the VM's life (default 2h). GCE deletes the VM
#                when it expires, even if this script is killed.
#   KEEP=1       leave the VM running afterwards (debugging); you delete it.
#
# Cost guard rails: the VM is deleted on exit (success, failure or Ctrl-C),
# and independently by GCE after MAX_RUN via --max-run-duration.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT to the Google Cloud project to bill}"
GPU="${GPU:-l4}"
SPOT="${SPOT:-0}"
MAX_RUN="${MAX_RUN:-2h}"
KEEP="${KEEP:-0}"
MODELS="${MODELS:-llama3.2:1b llama3.1:8b}"

case "$GPU" in
  l4)   MACHINE=g2-standard-8; DEFAULT_ZONES="europe-west4-a europe-west4-b europe-west4-c europe-west1-b europe-west1-c europe-west3-a europe-west3-b europe-west2-a europe-west2-b" ;;
  a100) MACHINE=a2-highgpu-1g; DEFAULT_ZONES="europe-west4-a europe-west4-b" ;;
  h100) MACHINE=a3-highgpu-1g; DEFAULT_ZONES="europe-west4-b europe-west4-c europe-west1-b" ;;
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
echo "image family: deeplearning-platform-release/$FAMILY"

ZONE=""
# Every zone we have asked for a VM in, recorded *before* the request, because
# an interrupt during create leaves a machine that $ZONE does not know about
# yet. A GPU left running is the expensive mistake here, so the sweep is over
# everything we might have started, not just the one we settled on.
ATTEMPTED=()
cleanup() {
  if [ "$KEEP" = 1 ] && [ -n "$ZONE" ]; then
    echo "KEEP=1: $NAME is still running in $ZONE. Delete it with:"
    echo "  gcloud --project $GCP_PROJECT compute instances delete $NAME --zone $ZONE"
    return
  fi
  local z
  for z in ${ZONE:+$ZONE} ${ATTEMPTED[@]+"${ATTEMPTED[@]}"}; do
    if gc compute instances delete "$NAME" --zone "$z" >/dev/null 2>&1; then
      echo "deleted $NAME in $z"
    fi
  done
}
# EXIT alone does not fire when the shell is killed by a signal.
trap cleanup EXIT INT TERM HUP

spot_flags=()
[ "$SPOT" = 1 ] && spot_flags=(--provisioning-model=SPOT)
for z in $ZONES; do
  echo "trying $MACHINE in $z"
  ATTEMPTED+=("$z")
  if gc compute instances create "$NAME" --zone "$z" \
      --machine-type "$MACHINE" \
      --maintenance-policy TERMINATE ${spot_flags[@]+"${spot_flags[@]}"} \
      --max-run-duration "$MAX_RUN" --instance-termination-action DELETE \
      --image-project deeplearning-platform-release --image-family "$FAMILY" \
      --boot-disk-size 150GB --boot-disk-type pd-ssd \
      --metadata install-nvidia-driver=True \
      --labels purpose=lex-gpu-test 2>"$OUT/create-$z.log"; then
    ZONE="$z"
    break
  fi
  tail -2 "$OUT/create-$z.log" >&2
done
[ -n "$ZONE" ] || { echo "no zone had capacity (or quota) for $MACHINE; see $OUT/create-*.log" >&2; exit 1; }
echo "$NAME up in $ZONE"

# SSH comes up before the driver finishes installing; remote.sh waits for it.
for i in $(seq 1 60); do
  gc compute ssh "$NAME" --zone "$ZONE" --command true >/dev/null 2>&1 && break
  sleep 10
done

git -C "$ROOT" archive --format=tar.gz -o "$OUT/src.tar.gz" HEAD
gc compute scp --zone "$ZONE" "$OUT/src.tar.gz" "$NAME:~/src.tar.gz"
# A failing run must still bring its logs home: no errexit from here on.
set +e
gc compute ssh "$NAME" --zone "$ZONE" --command \
  "mkdir -p lex-gpu && tar -xzf src.tar.gz -C lex-gpu && MODELS='$MODELS' bash lex-gpu/scripts/gcp/remote.sh" \
  2>&1 | tee "$OUT/remote.log"
status=${PIPESTATUS[0]}
gc compute scp --zone "$ZONE" --recurse "$NAME:~/results/*" "$OUT/" || true
echo "results in $OUT (remote exit $status)"
exit "$status"
