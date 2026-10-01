#!/usr/bin/env bash
# Build the disk image GPU runs boot from -- on a cheap machine with no GPU.
#
#   GCP_PROJECT=my-project scripts/gcp/build_image.sh
#
# A GPU run from the stock Deep Learning image spent its first 15-20 minutes
# at GPU prices installing Rust, building the workspace, installing Ollama
# and downloading 20 GB of models. All of that is done here instead, on a
# Spot e2 VM at a fraction of the price, and saved as an image in the family
# `lex-gpu-l4`; nvidia_test.sh boots from the newest one when it exists.
# Re-run this to update it -- a new toolchain, new dependencies, new models --
# and the older images past the newest KEEP_IMAGES are deleted.
#
# Knobs (environment):
#   GCP_PROJECT   required.
#   MACHINE       builder machine type (default e2-standard-16: the release
#                 build is most of the time, and it is CPU-bound).
#   ZONES         zones to try, in order.
#   IMAGE_FAMILY  default lex-gpu-l4.
#   KEEP_IMAGES   how many images of the family to keep (default 1: each
#                 is billed monthly, and a rebuild is cheap).
#   MODELS        Ollama baseline models (default: llama3.2:1b llama3.1:8b).
#   FROM=image    update the current image instead of starting from the
#                 stock one: only what changed is done.
#
# The builder is deleted on exit, success or not, and by GCE after 2 hours.
set -euo pipefail
: "${GCP_PROJECT:?set GCP_PROJECT to the Google Cloud project to bill}"
MACHINE="${MACHINE:-e2-standard-16}"
ZONES="${ZONES:-europe-west4-a europe-west4-b europe-west1-b europe-west1-c us-central1-a us-central1-b us-east1-b}"
IMAGE_FAMILY="${IMAGE_FAMILY:-lex-gpu-l4}"
KEEP_IMAGES="${KEEP_IMAGES:-1}"
MODELS="${MODELS:-llama3.2:1b llama3.1:8b}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
NAME="lex-image-builder-${STAMP}"
IMAGE="${IMAGE_FAMILY}-${STAMP}"
OUT="$ROOT/results/gcp/${STAMP}-image"
mkdir -p "$OUT"
gc() { gcloud --project "$GCP_PROJECT" --quiet "$@"; }

# FROM=image: start from the current image of the family instead of the
# stock one -- an update then only does what changed (prepare_image.sh is
# idempotent: a toolchain, build, model or checkpoint already there is
# kept), minutes instead of most of an hour.
if [ "${FROM:-}" = "image" ]; then
  SRC=(--image-project "$GCP_PROJECT" --image-family "$IMAGE_FAMILY")
  echo "base: $GCP_PROJECT/$(gc compute images describe-from-family "$IMAGE_FAMILY" --format='value(name)')"
else
  BASE="$(gcloud compute images list --project deeplearning-platform-release \
    --filter='family~^common-cu12.*ubuntu' --format='value(family)' | sort | tail -1)"
  [ -n "$BASE" ] || { echo "no common-cu12 ubuntu image family found" >&2; exit 1; }
  SRC=(--image-project deeplearning-platform-release --image-family "$BASE")
  echo "base: deeplearning-platform-release/$BASE"
fi

ZONE=""
cleanup() {
  local z
  z="$(gc compute instances list --filter="name=$NAME" --format="value(zone.basename())" 2>/dev/null | head -1 || true)"
  [ -n "$z" ] && gc compute instances delete "$NAME" --zone "$z" >/dev/null 2>&1 && echo "deleted $NAME in $z"
  return 0
}
trap cleanup EXIT INT TERM HUP

for z in $ZONES; do
  echo "trying $MACHINE in $z"
  if gc compute instances create "$NAME" --zone "$z" \
      --machine-type "$MACHINE" \
      --provisioning-model=SPOT --instance-termination-action DELETE --max-run-duration 2h \
      "${SRC[@]}" \
      --boot-disk-size 150GB --boot-disk-type pd-balanced \
      --labels purpose=lex-image-build >/dev/null 2>"$OUT/create-$z.log"; then
    ZONE="$z"
    break
  fi
  why=$(grep -oE "code: [A-Z_]+|[A-Z_]+EXHAUSTED[A-Z_]*|[Qq]uota [^.]*" "$OUT/create-$z.log" | head -1 || true)
  echo "  $z: ${why:-failed, see $OUT/create-$z.log}"
done
[ -n "$ZONE" ] || { echo "no zone had a $MACHINE" >&2; exit 1; }
echo "$NAME up in $ZONE"

for i in $(seq 1 60); do
  gc compute ssh "$NAME" --zone "$ZONE" --command true >/dev/null 2>&1 && break
  sleep 10
done

git -C "$ROOT" archive --format=tar.gz -o "$OUT/src.tar.gz" HEAD
gc compute scp --zone "$ZONE" "$OUT/src.tar.gz" "$NAME:~/src.tar.gz"
gc compute ssh "$NAME" --zone "$ZONE" --command \
  "mkdir -p lex-gpu && tar -xzf src.tar.gz -C lex-gpu && MODELS='$MODELS' bash lex-gpu/scripts/gcp/prepare_image.sh" \
  2>&1 | tee "$OUT/prepare.log"
grep -q "image contents ready" "$OUT/prepare.log" || { echo "preparation did not finish; no image made" >&2; exit 1; }

# An image of a running disk can be inconsistent; stop the builder first.
gc compute instances stop "$NAME" --zone "$ZONE" >/dev/null
gc compute images create "$IMAGE" --source-disk "$NAME" --source-disk-zone "$ZONE" \
  --family "$IMAGE_FAMILY" --labels purpose=lex-gpu-test >/dev/null
echo "image $IMAGE (family $IMAGE_FAMILY)"

# Keep the newest few; images cost storage every month they exist.
gc compute images list --filter="family=$IMAGE_FAMILY" \
  --sort-by=~creationTimestamp --format="value(name)" | tail -n +"$((KEEP_IMAGES + 1))" |
  while read -r old; do
    gc compute images delete "$old" >/dev/null && echo "deleted old image $old"
  done
