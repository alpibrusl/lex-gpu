"""Minimal GGUF reader: metadata, tensor directory, and dequantisation of the
types small Llama checkpoints use (F32, F16, Q8_0, Q4_0, Q4_K, Q6_K).

It exists so the reference model and the tests depend on nothing but numpy
and the file. Layout follows ggml's `gguf.c` and `ggml-quants.c`.
"""

import mmap
import pathlib
import struct
import json

import numpy as np

GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    8: ("Q8_0", 32, 34),
    12: ("Q4_K", 256, 144),
    14: ("Q6_K", 256, 210),
    30: ("BF16", 1, 2),
}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class GGUF:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        f = open(self.path, "rb")
        self.buf = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        assert self._read("4s") == b"GGUF", "not a GGUF file"
        self.version = self._read("<I")
        n_tensors, n_kv = self._read("<Q"), self._read("<Q")
        self.meta = {}
        for _ in range(n_kv):
            k = self._str()
            self.meta[k] = self._value(self._read("<I"))
        self.tensors = {}
        for _ in range(n_tensors):
            name = self._str()
            nd = self._read("<I")
            dims = [self._read("<Q") for _ in range(nd)]
            ty = self._read("<I")
            off = self._read("<Q")
            self.tensors[name] = (dims, ty, off)
        align = self.meta.get("general.alignment", 32)
        self.data = (self.pos + align - 1) // align * align

    def _read(self, fmt):
        (v,) = struct.unpack_from(fmt, self.buf, self.pos)
        self.pos += struct.calcsize(fmt)
        return v

    def _str(self):
        n = self._read("<Q")
        s = bytes(self.buf[self.pos : self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def _value(self, t):
        if t in _SCALAR:
            return self._read(_SCALAR[t])
        if t == 8:
            return self._str()
        if t == 9:
            et, n = self._read("<I"), self._read("<Q")
            return [self._value(et) for _ in range(n)]
        raise ValueError(f"unknown gguf value type {t}")

    def type_name(self, name):
        return GGML_TYPES.get(self.tensors[name][1], (f"type{self.tensors[name][1]}",))[0]

    def tensor(self, name):
        """Dequantised to float32, shaped [rows, cols] (ggml's ne reversed)."""
        dims, ty, off = self.tensors[name]
        tname, blk, bsz = GGML_TYPES[ty]
        n = int(np.prod(dims))
        nbytes = n // blk * bsz
        raw = np.frombuffer(self.buf, np.uint8, nbytes, self.data + off)
        x = DEQUANT[tname](raw, n)
        return x.reshape(list(reversed(dims)))


def _f32(raw, n):
    return raw.view("<f4").astype(np.float32)


def _f16(raw, n):
    return raw.view("<f2").astype(np.float32)


def _bf16(raw, n):
    return (raw.view("<u2").astype(np.uint32) << 16).view(np.float32)


def _q8_0(raw, n):
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    q = b[:, 2:].view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)


def _q4_0(raw, n):
    b = raw.reshape(-1, 18)
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    qs = b[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    return (d * np.concatenate([lo, hi], axis=1).astype(np.float32)).reshape(-1)


def _k_scale_min(scales):
    """Unpack Q4_K's 12 bytes of 6-bit scales and mins (8 each)."""
    s = scales.astype(np.uint8)
    sc = np.empty((s.shape[0], 8), np.uint8)
    mn = np.empty((s.shape[0], 8), np.uint8)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def _q4_k(raw, n):
    b = raw.reshape(-1, 144)
    d = b[:, 0:2].copy().view("<f2").astype(np.float32)
    dmin = b[:, 2:4].copy().view("<f2").astype(np.float32)
    sc, mn = _k_scale_min(b[:, 4:16])
    qs = b[:, 16:144].reshape(-1, 4, 32)  # 4 chunks of 64 values
    out = np.empty((b.shape[0], 8, 32), np.float32)
    out[:, 0::2] = qs & 0x0F
    out[:, 1::2] = qs >> 4
    out = d[:, :, None] * sc[:, :, None] * out - dmin[:, :, None] * mn[:, :, None]
    return out.reshape(-1)


def _q6_k(raw, n):
    b = raw.reshape(-1, 210)
    ql = b[:, 0:128].reshape(-1, 2, 64)
    qh = b[:, 128:192].reshape(-1, 2, 32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32).reshape(-1, 2, 8)
    d = b[:, 208:210].copy().view("<f2").astype(np.float32)
    out = np.empty((b.shape[0], 2, 128), np.float32)
    for half in range(2):
        l, h = ql[:, half].astype(np.int32), qh[:, half].astype(np.int32)
        q1 = (l[:, 0:32] & 0xF) | (((h >> 0) & 3) << 4)
        q2 = (l[:, 32:64] & 0xF) | (((h >> 2) & 3) << 4)
        q3 = (l[:, 0:32] >> 4) | (((h >> 4) & 3) << 4)
        q4 = (l[:, 32:64] >> 4) | (((h >> 6) & 3) << 4)
        q = np.concatenate([q1, q2, q3, q4], axis=1) - 32  # 128 values
        s = np.repeat(sc[:, half], 16, axis=1)  # one scale per 16
        out[:, half] = d * s * q
    return out.reshape(-1)


DEQUANT = {"F32": _f32, "F16": _f16, "BF16": _bf16, "Q8_0": _q8_0, "Q4_0": _q4_0, "Q4_K": _q4_k, "Q6_K": _q6_k}


def ollama_model(name, root=pathlib.Path.home() / ".ollama/models"):
    """Path of the GGUF blob behind an Ollama model tag, e.g. 'llama3.2:1b'."""
    repo, tag = (name.split(":") + ["latest"])[:2]
    m = json.loads((root / "manifests/registry.ollama.ai/library" / repo / tag).read_text())
    for layer in m["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            return root / "blobs" / layer["digest"].replace(":", "-")
    raise FileNotFoundError(f"{name} has no GGUF model layer")


if __name__ == "__main__":
    import collections
    import sys

    g = GGUF(ollama_model(sys.argv[1] if len(sys.argv) > 1 else "llama3.2:1b"))
    print(g.path, "gguf v", g.version)
    for k, v in g.meta.items():
        if isinstance(v, list):
            v = f"[{len(v)} items] {v[:3]}..."
        print(f"  {k}: {v}")
    print(collections.Counter(g.type_name(n) for n in g.tensors))
    for n in list(g.tensors)[:12]:
        print(f"  {n}: {g.tensors[n][0]} {g.type_name(n)}")
