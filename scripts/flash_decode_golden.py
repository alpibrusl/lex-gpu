"""Golden output for the P1 flash-attention decode test, computed by PyTorch.

Inputs are not stored: they are regenerated bit-for-bit from the same xorshift32
pattern as `lex_ir::reference::fill_pattern_f32`, rounded to f16 storage. Only
PyTorch's output is checked in, so the Rust test compares against an
independent implementation without CI needing torch.

    python3 scripts/flash_decode_golden.py

Shape: 32 query heads sharing one KV head (MQA decode), head dim 128,
512 KV positions. Seeds: q=1, k=2, v=3.
"""

import pathlib

import numpy as np
import torch
import torch.nn.functional as F

Q_ROWS, D, SEQ = 32, 128, 512
OUT = pathlib.Path(__file__).resolve().parent.parent / "crates/lex-front/tests/data/flash_decode_f16_q32_d128_s512.f32"


def fill_pattern(n: int, seed: int) -> np.ndarray:
    """Mirror of lex_ir::reference::fill_pattern_f32 (xorshift32)."""
    s = (seed | 1) & 0xFFFFFFFF
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s ^= (s << 13) & 0xFFFFFFFF
        s ^= s >> 17
        s ^= (s << 5) & 0xFFFFFFFF
        out[i] = np.float32((s >> 8) / float(1 << 23) - 1.0)
    return out


def f16_input(shape, seed):
    x = fill_pattern(int(np.prod(shape)), seed).astype(np.float16)
    return torch.from_numpy(x.astype(np.float64).reshape(shape))


def main():
    q = f16_input((Q_ROWS, D), 1)
    k = f16_input((SEQ, D), 2)
    v = f16_input((SEQ, D), 3)
    # [batch, heads, rows, d]; float64 so the golden is not itself the thing
    # with rounding error in it.
    o = F.scaled_dot_product_attention(q[None, None], k[None, None], v[None, None])[0, 0]
    OUT.write_bytes(o.to(torch.float32).numpy().astype("<f4").tobytes())
    print(f"wrote {OUT} ({OUT.stat().st_size} B), torch {torch.__version__}")


if __name__ == "__main__":
    main()
