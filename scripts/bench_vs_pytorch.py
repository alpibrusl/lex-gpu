"""Benchmark tile's Metal kernels against PyTorch on the same GPU.

Runs `tile-bench` (release build) and the PyTorch MPS equivalents on the same
shapes, times both the same way, and prints one table:

    python3 scripts/bench_vs_pytorch.py
    python3 scripts/bench_vs_pytorch.py --dtype f16 --rows 8192
    python3 scripts/bench_vs_pytorch.py --batch 8 --seq 8192   # flash shape

Two groups of kernels:

- copy / rmsnorm: P0's hand-planned kernels.
- flash_decode_f16: a typed tile-front kernel, checked, lowered to MSL. It
  uses Llama-3-8B's decode shape (8 KV heads x 4 query heads, head dim 128,
  f16). The main row compares it with SDPA called on the same grouped layout,
  where each KV head's 4 query heads are 4 query rows. The `(gqa)` row calls
  `enable_gqa=True` on the 32-head layout instead.
  P2 lowers it for correctness, not speed. Expect PyTorch to win by a wide
  margin until P3.

Method, matching `tile_metal::Gpu::time`: warm up, then `iters` back-to-back
calls per timed batch, synchronise, keep the fastest of `repeats` batches.
GB/s uses the same ideal-bytes accounting as tile (each input read once, each
output written once), so the numbers are directly comparable.

PyTorch is measured two ways for RMSNorm: `F.rms_norm` (one fused op where the
backend has one) and the textbook eager expression (several kernels, each
round-tripping memory) -- the second is what most model code actually runs.
It upcasts to f32 and back, as Hugging Face's LlamaRMSNorm does, which is why
its f16 number is the worse of the two.
"""

import argparse
import pathlib
import subprocess
import sys
import time

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
DTYPES = {"f32": (torch.float32, 4), "f16": (torch.float16, 2)}


def timed(fn, iters, repeats):
    """Fastest per-call wall time over `repeats` batches of `iters` calls."""
    fn()
    fn()
    torch.mps.synchronize()
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.mps.synchronize()
        best = min(best, (time.perf_counter() - start) / iters)
    return best


def torch_numbers(a):
    dt, size = DTYPES[a.dtype]
    dev = torch.device("mps")
    n = a.mib * 1024 * 1024 // size
    g = torch.Generator(device="cpu").manual_seed(0)
    cx = torch.rand(n, generator=g).mul_(2).sub_(1).to(dev, dt)
    cy = torch.empty_like(cx)
    x = torch.rand(a.rows, a.cols, generator=g).mul_(2).sub_(1).to(dev, dt)
    w = torch.rand(a.cols, generator=g).to(dev, dt)

    def eager():
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + a.eps)).to(dt) * w

    copy_bytes = 2 * n * size
    norm_bytes = (2 * a.rows * a.cols + a.cols) * size
    out = {
        f"copy_{a.dtype}": copy_bytes / timed(lambda: cy.copy_(cx), a.iters, a.repeats) / 1e9,
        f"rmsnorm_{a.dtype}": norm_bytes
        / timed(lambda: F.rms_norm(x, (a.cols,), w, a.eps), a.iters, a.repeats)
        / 1e9,
        f"rmsnorm_{a.dtype} (eager)": norm_bytes / timed(eager, a.iters, a.repeats) / 1e9,
    }
    # Same op, same inputs: the two PyTorch paths must agree before either
    # number means anything.
    ref = eager().float()
    got = F.rms_norm(x, (a.cols,), w, a.eps).float()
    err = ((got - ref).abs() / ref.abs().clamp_min(1e-3)).max().item()
    return out, err


def torch_flash(a):
    """SDPA on the flash-decode shape, in tile's memory layout."""
    kvh, group, d = 8, 4, 128
    dev = torch.device("mps")
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.rand(a.batch, kvh * group, 1, d, generator=g).sub_(0.5).to(dev, torch.float16)
    k = torch.rand(a.batch, kvh, a.seq, d, generator=g).sub_(0.5).to(dev, torch.float16)
    v = torch.rand(a.batch, kvh, a.seq, d, generator=g).sub_(0.5).to(dev, torch.float16)
    h = a.batch * kvh
    # Same accounting as tile: q and k/v in f16, o in f32.
    nbytes = h * group * d * 2 + 2 * h * a.seq * d * 2 + h * group * d * 4
    # The same query heads as 4 query rows per KV head: no GQA expansion.
    # This is the layout tile's kernel uses.
    qg = q.view(a.batch, kvh, group, d)
    gqa = timed(lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True), a.iters, a.repeats)
    grouped = timed(lambda: F.scaled_dot_product_attention(qg, k, v), a.iters, a.repeats)
    # Both layouts must compute the same attention before either timing counts.
    a1 = F.scaled_dot_product_attention(q, k, v, enable_gqa=True).view(a.batch, kvh, group, d)
    a2 = F.scaled_dot_product_attention(qg, k, v)
    assert torch.allclose(a1.float(), a2.float(), atol=2e-3), "SDPA layouts disagree"
    return {
        "flash_decode_f16": nbytes / grouped / 1e9,
        "flash_decode_f16 (gqa)": nbytes / gqa / 1e9,
    }


def tile_numbers(a, flash=False):
    cmd = [
        "cargo", "run", "-q", "--release", "-p", "tile-bench", "--",
        "--dtype", a.dtype, "--mib", str(a.mib), "--rows", str(a.rows),
        "--cols", str(a.cols), "--iters", str(a.iters), "--repeats", str(a.repeats),
    ]
    if flash:
        cmd += ["--flash", "--batch", str(a.batch), "--seq", str(a.seq)]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        tok = line.split()
        if tok and tok[0].startswith(("copy_", "rmsnorm_", "flash_")):
            out[tok[0]] = float(tok[-2])
    if not out:
        sys.exit(f"tile-bench produced no numbers:\n{p.stdout}{p.stderr}")
    return out, p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dtype", choices=DTYPES, default="f32")
    ap.add_argument("--mib", type=int, default=256)
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--cols", type=int, default=4096)
    ap.add_argument("--eps", type=float, default=1e-5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4, help="flash: sequences")
    ap.add_argument("--seq", type=int, default=4096, help="flash: cached positions")
    a = ap.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("PyTorch has no MPS device here; this comparison needs Apple silicon.")

    tile, raw = tile_numbers(a)
    tile.update(tile_numbers(a, flash=True)[0])
    ours, err = torch_numbers(a)
    ours.update(torch_flash(a))
    print(raw.split("\n\n")[0])  # tile-bench's device/target header
    print(
        f"\npytorch {torch.__version__}, mps; rows={a.rows} cols={a.cols} copy={a.mib} MiB; "
        f"flash batch={a.batch} seq={a.seq}\n"
    )
    print(f"{'kernel':<24} {'tile GB/s':>10} {'torch GB/s':>11} {'tile / torch':>13}")
    for name, gbs in ours.items():
        base = name.split()[0]
        t = tile.get(base)
        ratio = f"{t / gbs:12.2f}x" if t else f"{'-':>13}"
        tcol = f"{t:10.1f}" if t else f"{'-':>10}"
        print(f"{name:<24} {tcol} {gbs:11.1f} {ratio}")
    print(f"\nF.rms_norm vs eager expression: max rel err {err:.1e}")


if __name__ == "__main__":
    main()
