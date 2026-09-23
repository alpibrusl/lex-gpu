#!/usr/bin/env bash
# Find, and optionally delete, anything this project's GPU tests left running.
#
#   scripts/gcp/sweep.sh                  # list only
#   DELETE=1 scripts/gcp/sweep.sh         # delete what it lists
#
#   GCP_PROJECT  required (or `gcloud config set project`)
#
# `nvidia_test.sh` deletes its VM on exit and asks GCE to delete it again
# after MAX_RUN, so this should find nothing. It exists because "should"
# is not a billing guarantee: a machine killed with SIGKILL, a laptop that
# sleeps mid-run, or a crashed gcloud all skip the first guard, and the
# second is up to two hours of a GPU you are not using.
#
# Every VM the harness creates is labelled `purpose=lex-gpu-test`, so this
# never touches anything else in the project.
set -euo pipefail

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || {
  echo "set GCP_PROJECT (or gcloud config set project)" >&2; exit 2; }

mapfile -t FOUND < <(gcloud --project "$PROJECT" compute instances list \
  --filter="labels.purpose=lex-gpu-test" \
  --format="value(name,zone,status)" 2>/dev/null)

if [ "${#FOUND[@]}" -eq 0 ]; then
  echo "$PROJECT: nothing running with purpose=lex-gpu-test"
  exit 0
fi

printf '%s\n' "$PROJECT: ${#FOUND[@]} left over"
printf '  %s\n' "${FOUND[@]}"

if [ "${DELETE:-0}" != 1 ]; then
  echo
  echo "these are still billing. delete them with:"
  echo "  DELETE=1 GCP_PROJECT=$PROJECT $0"
  exit 1
fi

for row in "${FOUND[@]}"; do
  name="$(echo "$row" | awk '{print $1}')"
  zone="$(echo "$row" | awk '{print $2}')"
  echo "deleting $name in $zone"
  gcloud --project "$PROJECT" --quiet compute instances delete "$name" --zone "$zone"
done
