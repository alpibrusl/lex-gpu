"""Fetch an Ollama model's blobs straight from the registry.

    python3 scripts/ollama_fetch.py qwen3.8:27b-mlx --root ~/.ollama/models

`ollama pull` refuses some models on some machines -- an MLX build on
Linux gives "this model requires MLX support, but the MLX runtime is not
available" -- and that check is the client's, not the registry's. The
weights are ordinary blobs behind ordinary HTTP, and `lex-rt` reads the
store directly and never asks Ollama to run anything. So the client's
opinion about what this machine can execute is not one we need.

Which matters for testing the CUDA backend against the one model worth
testing it on: a hybrid gated-delta, NVFP4, draft-head checkpoint that
Ollama itself will only run on a Mac.

Writes the layout `ollama_model()` expects:

    <root>/manifests/registry.ollama.ai/library/<repo>/<tag>
    <root>/blobs/sha256-<digest>
"""

import argparse
import hashlib
import json
import pathlib
import sys
import urllib.request

REGISTRY = "https://registry.ollama.ai/v2/library"


def get(url, out=None):
    req = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        if out is None:
            return r.read()
        # Streamed: these run to gigabytes and need not be held in memory.
        # Hashed on the way past, because a truncated blob is a corrupt
        # model that will be blamed on the kernels.
        h, n = hashlib.sha256(), 0
        with open(out, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                h.update(chunk)
                n += len(chunk)
        return h.hexdigest(), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="e.g. qwen3.8:27b-mlx")
    ap.add_argument("--root", required=True, help="the model store to fill")
    a = ap.parse_args()

    repo, tag = a.tag.split(":", 1) if ":" in a.tag else (a.tag, "latest")
    root = pathlib.Path(a.root).expanduser()
    blobs = root / "blobs"
    mdir = root / "manifests" / "registry.ollama.ai" / "library" / repo
    blobs.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    raw = get(f"{REGISTRY}/{repo}/manifests/{tag}")
    manifest = json.loads(raw)
    layers = list(manifest.get("layers", []))
    if cfg := manifest.get("config"):
        layers.append(cfg)

    total = 0
    for layer in layers:
        digest = layer["digest"]
        want = digest.split(":", 1)[1]
        dst = blobs / f"sha256-{want}"
        if dst.exists() and dst.stat().st_size == layer.get("size", -1):
            print(f"  have {digest[:19]} ({layer['size'] / 1e9:.2f} GB)")
            continue
        print(f"  get  {digest[:19]} ({layer.get('size', 0) / 1e9:.2f} GB) ...", flush=True)
        got, n = get(f"{REGISTRY}/{repo}/blobs/{digest}", dst)
        if got != want:
            dst.unlink(missing_ok=True)
            raise SystemExit(f"{digest}: got sha256:{got} over {n} bytes")
        total += n

    # The manifest last: its presence is what `ollama_model` tests, so
    # writing it only after every blob is verified means a half-finished
    # fetch looks missing rather than broken.
    (mdir / tag).write_bytes(raw)
    print(f"{a.tag}: {len(layers)} blobs, {total / 1e9:.2f} GB fetched, into {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
