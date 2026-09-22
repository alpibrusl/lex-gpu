"""Benchmark tile's Metal kernels against PyTorch on the same GPU.

Runs `tile-bench` (release build) and the PyTorch MPS equivalents on the same
shapes, times both the same way, and prints one table:

    python3 scripts/bench_vs_pytorch.py
    python3 scripts/bench_vs_pytorch.py --dtype f16 --rows 8192

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


def tile_numbers(a):
    cmd = [
        "cargo", "run", "-q", "--release", "-p", "tile-bench", "--",
        "--dtype", a.dtype, "--mib", str(a.mib), "--rows", str(a.rows),
        "--cols", str(a.cols), "--iters", str(a.iters), "--repeats", str(a.repeats),
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        tok = line.split()
        if tok and (tok[0].startswith("copy_") or tok[0].startswith("rmsnorm_")):
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
    a = ap.parse_args()

    if not torch.backends.mps.is_available():
        sys.exit("PyTorch has no MPS device here; this comparison needs Apple silicon.")

    tile, raw = tile_numbers(a)
    ours, err = torch_numbers(a)
    print(raw.split("\n\n")[0])  # tile-bench's device/target header
    print(f"\npytorch {torch.__version__}, mps; rows={a.rows} cols={a.cols} copy={a.mib} MiB\n")
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
