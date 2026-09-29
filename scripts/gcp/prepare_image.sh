#!/usr/bin/env bash
# Runs on the image builder (a CPU VM, no GPU): install and pre-build
# everything a GPU run would otherwise spend its first twenty minutes on.
# Called by build_image.sh with the source unpacked in ~/lex-gpu.
#
# What ends up on the disk, and where the GPU run finds it:
#   ~/.cargo, ~/.rustup        the toolchain and the crate registry
#   ~/lex-target               a release build of the workspace, tests too
#                              (remote.sh points CARGO_TARGET_DIR here, so a
#                              run recompiles only what changed since)
#   /usr/local/bin/ollama      Ollama, and its service's store with the
#                              baseline Llama models
#   ~/.ollama/models           qwen3.8:27b-mlx, fetched from the registry
#   ~/lex-src.sha256           the source the build was made from
#
# The source itself is removed at the end: every run brings its own.
set -euo pipefail
cd "$HOME/lex-gpu"
step() { echo; echo "=== $*"; }

step "keep the kernel the driver was built for"
# The base image's NVIDIA driver is a precompiled module for the kernel it
# ships, and its packages are held. Left running, unattended-upgrades
# installed a newer kernel on the first builder; the GPU VM booted into it
# and had no driver (7.0.0-1013, modules only for 1011). Stopped here, the
# kernel packages held, and checked again at the end.
sudo systemctl disable --now unattended-upgrades apt-daily.timer apt-daily-upgrade.timer \
  >/dev/null 2>&1 || true
while sudo fuser /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock >/dev/null 2>&1; do sleep 5; done
sudo apt-mark hold linux-image-gcp linux-gcp linux-headers-gcp >/dev/null 2>&1 || true
uname -r

step "system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential python3 >/dev/null

step "Rust toolchain"
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
fi
. "$HOME/.cargo/env"
rustup update stable >/dev/null 2>&1 || true
rustc --version

step "release build of the workspace, tests included"
export CARGO_TARGET_DIR="$HOME/lex-target"
cargo build --release --workspace --all-targets 2>&1 | tail -2
cargo test --release --workspace --no-run 2>&1 | tail -2

step "Ollama and the baseline models"
if ! command -v ollama >/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh >/dev/null
fi
sudo systemctl enable --now ollama >/dev/null 2>&1 || true
for i in $(seq 1 30); do curl -s localhost:11434 >/dev/null && break; sleep 2; done
for m in ${MODELS:-llama3.2:1b llama3.1:8b}; do
  ollama pull "$m" >/dev/null && echo "pulled $m"
done

step "qwen3.8:27b-mlx into ~/.ollama/models"
python3 scripts/ollama_fetch.py qwen3.8:27b-mlx --root "$HOME/.ollama/models" 2>&1 | tail -3

step "leave the disk clean"
# What the build was made from, so a run can tell which of its files are
# unchanged (remote.sh) and let Cargo keep what it built from them.
find . -type f -print0 | xargs -0 sha256sum > "$HOME/lex-src.sha256"
wc -l < "$HOME/lex-src.sha256"
cd "$HOME"
rm -rf "$HOME/lex-gpu" "$HOME/src.tar.gz" "$HOME/results"
# Any kernel that slipped in without the driver goes, and the one a GPU VM
# will boot -- the newest left -- must have it, or there is no image.
for k in $(ls /lib/modules); do
  [ -e "/boot/vmlinuz-$k" ] || continue
  if ! find "/lib/modules/$k" -name 'nvidia.ko*' | grep -q .; then
    echo "removing kernel $k: no NVIDIA module for it"
    sudo DEBIAN_FRONTEND=noninteractive apt-get purge -y -qq "linux-image-$k" "linux-modules-$k" >/dev/null
  fi
done
sudo update-grub >/dev/null 2>&1 || true
boot=$(ls /boot/vmlinuz-* | sed 's|^/boot/vmlinuz-||' | sort -V | tail -1)
find "/lib/modules/$boot" -name 'nvidia.ko*' | grep -q . ||
  { echo "kernel $boot has no NVIDIA module; not an image a GPU can use"; exit 1; }
echo "boot kernel $boot has the NVIDIA module"
sudo apt-get clean
sync
df -h / | tail -1
echo "image contents ready"
