"""Run the same prompts through tile (on the GPU) and through Ollama, and
compare them token by token.

    python3 scripts/tile_vs_ollama.py
    python3 scripts/tile_vs_ollama.py --prompt "Why is the sky blue?" --steps 32

For every prompt: tokenise it the way Ollama does (Llama 3 BPE plus BOS,
checked in `llama_ref.py`), greedy-decode with tile's `generate` example,
ask Ollama for the same greedy continuation with top-k log-probabilities, and
compare:

- the generated tokens must be identical;
- for every token in Ollama's top-k at every step, tile's log-probability
  must agree within `--tol`.

Ollama runs llama.cpp, which quantises activations to 8 bits inside its
Q8_0 matmuls; tile does not. That, not a bug, is why the numbers differ in
the third decimal.
"""

import argparse
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf import GGUF, ollama_model  # noqa: E402
from llama_ref import ollama, tokenizer  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent


def tile(model, ids, steps, top):
    cmd = [
        "cargo", "run", "-q", "--release", "-p", "tile-rt", "--example", "generate", "--",
        "--model", model, "--ids", ",".join(map(str, ids)), "--steps", str(steps), "--top", str(top),
    ]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"tile generate failed:\n{p.stderr}")
    out, timing = [], {}
    for line in p.stdout.splitlines():
        w = line.split()
        if w[0] == "step":
            out.append((int(w[1]), {int(i): float(lp) for i, lp in (x.split(":") for x in w[2:])}))
        elif w[0] == "time":
            timing = dict(zip(w[1::2], map(float, w[2::2])))
    return out, timing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.05)
    a = ap.parse_args()
    prompts = a.prompt or [
        "The capital of France is",
        "def fibonacci(n):",
        "Once upon a time, in a small village,",
        "The three laws of thermodynamics are",
    ]

    g = GGUF(ollama_model(a.model))
    tk = tokenizer(g)
    bos = g.meta["tokenizer.ggml.bos_token_id"]
    by_bytes = {bytes(tk.decode([i]).encode("utf-8", "surrogatepass")): i for i in range(len(tk.get_vocab()))}

    all_ok = True
    for prompt in prompts:
        ids = [bos] + tk.encode(prompt, add_special_tokens=False).ids
        mine, timing = tile(a.model, ids, a.steps, a.top)
        ref = ollama(a.model, prompt, a.steps, a.top)
        worst, same, n = 0.0, 0, min(len(mine), len(ref))
        diverged = None
        for i in range(n):
            want = by_bytes.get(bytes(ref[i]["bytes"]))
            if mine[i][0] != want:
                diverged = i
                break
            same += 1
            for alt in ref[i]["top_logprobs"]:
                j = by_bytes.get(bytes(alt["bytes"]))
                if j in mine[i][1]:
                    worst = max(worst, abs(mine[i][1][j] - alt["logprob"]))
        text = tk.decode([t for t, _ in mine[:same]])
        ok = diverged is None and worst <= a.tol
        all_ok &= ok
        print(f"\n{prompt!r}")
        print(f"  tile   : {text!r}")
        print(
            f"  {same}/{n} tokens identical to Ollama, worst |dlogprob| {worst:.4f} (tolerance {a.tol})  "
            f"{'PASS' if ok else 'FAIL'}"
        )
        if diverged is not None:
            print(f"  diverged at step {diverged}: tile {tk.decode([mine[diverged][0]])!r}, "
                  f"ollama {ref[diverged]['token']!r}")
        print(f"  tile decode {timing.get('decode_tok_s', 0):.1f} tok/s on the GPU (P2: correct, not fast)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
