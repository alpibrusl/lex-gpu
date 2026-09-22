"""Read Qwen3.5 (MLX, nvfp4) tensors out of the local Ollama store.

    python3 scripts/qwen_nvfp4.py --model qwen3.8:27b-mlx --tensor model.language_model.layers.0.mlp.gate_proj.weight

Ollama keeps each tensor as its own blob, headed by a safetensors header.
An nvfp4 tensor is three entries: `weight` (U32, eight 4-bit E2M1 codes per
word), `weight.scale` (U8, one FP8 E4M3 scale per 16 values) and
`weight.global_scale` (F32, one per tensor), so a value is

    e2m1(code) * e4m3(scale) * global_scale

This is the reference the GPU kernels are checked against, and the reader
the Qwen runner will use. Standard library plus numpy.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

STORE = pathlib.Path.home() / ".ollama/models"

# E2M1: sign, 2-bit exponent, 1-bit mantissa.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] * 2, dtype=np.float32)
E2M1[8:] *= -1.0


def e4m3_table():
    """Every FP8 E4M3 byte as a float (OCP e4m3fn: no inf, max 448)."""
    b = np.arange(256, dtype=np.uint32)
    sign = np.where(b >> 7 == 1, -1.0, 1.0)
    e = (b >> 3) & 0xF
    m = (b & 7).astype(np.float32)
    v = np.where(e == 0, m * 2.0**-9, (1.0 + m / 8.0) * 2.0 ** (e.astype(np.float32) - 7.0))
    v = np.where((e == 15) & (m == 7), np.nan, v)
    return (sign * v).astype(np.float32)


E4M3 = e4m3_table()


def manifest(model):
    name, _, tag = model.partition(":")
    p = STORE / "manifests/registry.ollama.ai/library" / name / (tag or "latest")
    if not p.exists():
        sys.exit(f"{model} is not in the local Ollama store ({p})")
    return json.loads(p.read_text())


def blob(digest):
    return STORE / "blobs" / f"sha256-{digest[7:]}"


def read_tensor(model, name):
    """(values, header) for one tensor: nvfp4 dequantised, or plain."""
    m = manifest(model)
    by = {l["name"]: l for l in m["layers"] if "name" in l}
    if name not in by:
        sys.exit(f"no tensor {name!r}; try --list")
    with open(blob(by[name]["digest"]), "rb") as f:
        raw = f.read()
    hlen = struct.unpack("<Q", raw[:8])[0]
    head = json.loads(raw[8 : 8 + hlen])
    data = raw[8 + hlen :]

    def part(key):
        e = head[key]
        lo, hi = e["data_offsets"]
        return e, data[lo:hi]

    e, buf = part(name)
    if e["dtype"] == "U32":  # nvfp4
        rows, words = e["shape"]
        codes = np.frombuffer(buf, dtype=np.uint32).reshape(rows, words)
        nib = np.empty((rows, words * 8), dtype=np.uint8)
        for k in range(8):  # little-endian: value k is bits 4k..4k+3
            nib[:, k::8] = (codes >> (4 * k)) & 0xF
        se, sbuf = part(name + ".scale")
        scale = E4M3[np.frombuffer(sbuf, dtype=np.uint8).reshape(se["shape"])]
        _, gbuf = part(name + ".global_scale")
        gs = float(np.frombuffer(gbuf, dtype=np.float32)[0])
        w = E2M1[nib].reshape(rows, -1)
        w *= np.repeat(scale, 16, axis=1)
        return w * gs, head
    if e["dtype"] == "BF16":
        u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(e["shape"]), head
    if e["dtype"] in ("F32", "F16"):
        dt = np.float32 if e["dtype"] == "F32" else np.float16
        return np.frombuffer(buf, dtype=dt).reshape(e["shape"]).astype(np.float32), head
    sys.exit(f"unhandled dtype {e['dtype']} for {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen3.8:27b-mlx")
    ap.add_argument("--tensor", default="model.language_model.layers.0.mlp.gate_proj.weight")
    ap.add_argument("--list", action="store_true", help="list tensor names and sizes")
    a = ap.parse_args()
    if a.list:
        for l in manifest(a.model)["layers"]:
            if "name" in l:
                print(f"{l['size']:>13,}  {l['name']}")
        return
    w, head = read_tensor(a.model, a.tensor)
    meta = head.get("__metadata__", {})
    print(f"{a.tensor}\n  {meta}  shape {w.shape}")
    print(f"  mean {w.mean():+.5f}  std {w.std():.5f}  min {w.min():+.4f}  max {w.max():+.4f}")
    print(f"  zeros {100 * (w == 0).mean():.1f}%  |w|>0 median {np.median(np.abs(w[w != 0])):.5f}")


if __name__ == "__main__":
    main()
