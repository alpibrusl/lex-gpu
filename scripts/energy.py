#!/usr/bin/env python3
"""Joules per token, for this engine and for Ollama on the same machine.

Tokens per second says how fast; this says what it cost. On a decode that
is memory-bound the two are not the same question -- an engine reading
fewer bytes per token can be both slower and cheaper -- so it is worth
measuring rather than inferring.

    # NVIDIA, no privileges needed
    python3 scripts/energy.py --engine lex    --tokens 256
    python3 scripts/energy.py --engine ollama --model llama3.2:1b --tokens 256

    # Apple: powermetrics is root-only, so the whole thing runs under sudo
    sudo python3 scripts/energy.py --engine lex --tokens 256

What it reports, and why both numbers:

  total    the board (or SoC) draw while generating, divided by tokens.
           What the wall socket sees.
  marginal the same with the idle baseline subtracted. What the work
           itself cost. The gap between them is the machine sitting there
           being on, which is large on a laptop and not the engine's doing.

Read the two backends against another engine on the same box, never
against each other: `nvidia-smi` reports whole-board power and
`powermetrics` reports SoC rails, which are different boundaries.
"""

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


class Sampler:
    """Power in watts, sampled on a thread until stopped."""

    def __init__(self, interval_ms):
        self.interval_ms = interval_ms
        self.points = []  # (monotonic seconds, watts)
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def _read(self):
        raise NotImplementedError

    def start(self):
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        # A sampler that has not produced a point yet would leave a hole at
        # the start of the window.
        deadline = time.monotonic() + 15
        while not self.points and time.monotonic() < deadline:
            time.sleep(0.05)
        if not self.points:
            raise SystemExit("the power sampler produced nothing; see --help")

    def stop(self):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=5)

    def joules(self, t0, t1):
        """Integrate watts over [t0, t1] by the trapezoid rule.

        The window edges are interpolated from the samples bracketing
        them, not dropped. Integrating only between the first and last
        sample *inside* the window and then dividing by the whole window
        reports about 15% less power than the sampler was emitting -- a
        stub that produced a known constant is what caught that.
        """
        pts = self.points
        if len(pts) < 2:
            return None, 0

        def at(t):
            if t <= pts[0][0]:
                return pts[0][1]
            if t >= pts[-1][0]:
                return pts[-1][1]
            for a, b in zip(pts, pts[1:]):
                if a[0] <= t <= b[0]:
                    span = b[0] - a[0]
                    f = (t - a[0]) / span if span > 0 else 0.0
                    return a[1] + f * (b[1] - a[1])
            return pts[-1][1]

        inner = [p for p in pts if t0 < p[0] < t1]
        seq = [(t0, at(t0))] + inner + [(t1, at(t1))]
        total = sum(
            (b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(seq, seq[1:])
        )
        return total, len(inner)

    def mean_watts(self, t0, t1):
        j, n = self.joules(t0, t1)
        return (j / (t1 - t0) if j is not None and t1 > t0 else None), n


class Nvidia(Sampler):
    """`nvidia-smi` board power. No privileges required."""

    def _read(self):
        cmd = [
            "nvidia-smi",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            f"-lms{self.interval_ms}",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            try:
                self.points.append((time.monotonic(), float(line.strip())))
            except ValueError:
                pass


class Powermetrics(Sampler):
    """Apple SoC rails. Root only -- run the whole script under sudo.

    The GPU rail is the one that matters for decode, but the combined
    figure is reported too: the CPU is not idle while the GPU works, and
    an engine that spends more host time is not free.
    """

    GPU = re.compile(r"^GPU Power:\s+([\d.]+)\s*mW", re.M)
    COMBINED = re.compile(r"^Combined Power \(.*?\):\s+([\d.]+)\s*mW", re.M)

    def __init__(self, interval_ms):
        super().__init__(interval_ms)
        self.combined = []

    def _read(self):
        cmd = [
            "powermetrics",
            "--samplers",
            "gpu_power,cpu_power",
            "-i",
            str(self.interval_ms),
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        block = []
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            block.append(line)
            if "Combined Power" in line or line.startswith("*** Sampled"):
                text = "".join(block)
                now = time.monotonic()
                g = self.GPU.search(text)
                c = self.COMBINED.search(text)
                if g:
                    self.points.append((now, float(g.group(1)) / 1000.0))
                if c:
                    self.combined.append((now, float(c.group(1)) / 1000.0))
                block = []


def make_sampler(interval_ms):
    if platform.system() == "Darwin":
        if shutil.which("powermetrics") is None:
            sys.exit("no powermetrics on this machine")
        return Powermetrics(interval_ms)
    if shutil.which("nvidia-smi") is None:
        sys.exit("no nvidia-smi on this machine, and not macOS")
    return Nvidia(interval_ms)


def post(url, body, timeout=1800):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# A prompt that keeps going, so the run is decode and not an early stop.
PROMPT = (
    "Write a long, detailed description of how a bicycle works, part by part. "
    "Do not stop early; keep going until you are told to stop."
)


def run_lex(host, model, tokens, temperature):
    """Returns (started, finished, tokens generated)."""
    body = {
        "model": model or "lex",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": tokens,
        "temperature": temperature,
    }
    t0 = time.monotonic()
    out = post(f"{host}/v1/chat/completions", body)
    t1 = time.monotonic()
    return t0, t1, out["usage"]["completion_tokens"]


def run_ollama(host, model, tokens, temperature):
    body = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_predict": tokens, "temperature": temperature},
    }
    t0 = time.monotonic()
    out = post(f"{host}/api/generate", body)
    t1 = time.monotonic()
    # Ollama counts them itself, which is better than trusting the clock.
    return t0, t1, out.get("eval_count", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["lex", "ollama"], default="lex")
    ap.add_argument("--host", default=None, help="default: 8080 for lex, 11434 for ollama")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--idle", type=float, default=8.0, help="seconds of baseline")
    ap.add_argument("--interval-ms", type=int, default=200)
    ap.add_argument("--warmup", action="store_true", help="one discarded run first")
    args = ap.parse_args()

    host = args.host or (
        "http://127.0.0.1:8080" if args.engine == "lex" else "http://127.0.0.1:11434"
    )
    runner = run_lex if args.engine == "lex" else run_ollama
    if args.engine == "ollama" and not args.model:
        sys.exit("--model is required for ollama")

    sampler = make_sampler(args.interval_ms)
    sampler.start()

    if args.warmup:
        # A cold engine pays for page faults and clock ramp that have
        # nothing to do with the steady state.
        print("warming up ...", flush=True)
        try:
            runner(host, args.model, min(32, args.tokens), args.temperature)
        except (urllib.error.URLError, OSError) as e:
            sampler.stop()
            sys.exit(f"cannot reach the {args.engine} server at {host}: {e}")

    print(f"idle baseline for {args.idle:.0f}s ...", flush=True)
    i0 = time.monotonic()
    time.sleep(args.idle)
    i1 = time.monotonic()
    idle_w, idle_n = sampler.mean_watts(i0, i1)

    print(f"generating {args.tokens} tokens on {args.engine} ...", flush=True)
    try:
        t0, t1, n = runner(host, args.model, args.tokens, args.temperature)
    except (urllib.error.URLError, OSError) as e:
        sampler.stop()
        sys.exit(f"cannot reach the {args.engine} server at {host}: {e}")
    sampler.stop()

    total_j, samples = sampler.joules(t0, t1)
    if total_j is None or not n:
        sys.exit(f"nothing to report: {samples} samples over {n} tokens")
    dur = t1 - t0
    mean_w = total_j / dur
    marginal_j = total_j - (idle_w or 0.0) * dur

    print()
    print(f"engine        {args.engine}" + (f" ({args.model})" if args.model else ""))
    print(f"tokens        {n} in {dur:.1f}s = {n / dur:.1f} tok/s")
    print(f"samples       {samples} over the window, {idle_n} for idle")
    print(f"idle          {idle_w:.1f} W" if idle_w is not None else "idle          -")
    print(f"while running {mean_w:.1f} W")
    print()
    print(f"total    {total_j / n * 1000:8.1f} mJ/token   ({total_j:.0f} J)")
    print(f"marginal {marginal_j / n * 1000:8.1f} mJ/token   ({marginal_j:.0f} J above idle)")
    if isinstance(sampler, Powermetrics) and sampler.combined:
        c = [p for p in sampler.combined if t0 <= p[0] <= t1]
        if len(c) > 1:
            cj = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(c, c[1:]))
            print(f"         {cj / n * 1000:8.1f} mJ/token   CPU+GPU+ANE combined")
    print()
    print("Compare against another engine on this same machine, not across")
    print("machines: the two samplers measure different boundaries.")


if __name__ == "__main__":
    main()
