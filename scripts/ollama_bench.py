"""Ollama's own speed: decode at several context lengths, and prefill.

    python3 scripts/ollama_bench.py --model llama3.1:8b
    python3 scripts/ollama_bench.py --model llama3.2:1b --contexts 0,512,1440 --json out.json

The baseline every tile number is compared against, measured the same way
on any machine Ollama runs on (the Mac here, an NVIDIA VM in the cloud:
`scripts/gcp/`). Standard library only.

For each context N, a fresh prompt of about N tokens of random words is sent
with greedy decoding for `--steps` tokens. Ollama reports its own timings:
`prompt_eval_*` (prefill of the N tokens) and `eval_*` (decode). The words
are random on every request, so Ollama's prompt cache cannot skip prefill;
repetitive prompts once inflated prefill figures here by 1.3x (8B) to 8x.
"""

import argparse
import json
import random
import sys
import urllib.request

WORDS = (
    "river stone garden paper window silver engine market winter forest "
    "letter orange bridge candle mirror planet harbor violin castle meadow "
    "thunder pencil island rocket basket coffee dragon feather lantern marble "
    "needle ocean pepper quartz saddle tunnel velvet walnut yellow zebra"
).split()


def generate(host, model, prompt, steps, ctx):
    req = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "num_predict": steps, "num_ctx": ctx, "seed": 0},
    }
    r = urllib.request.Request(
        f"{host}/api/generate", json.dumps(req).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=1800) as f:
        return json.load(f)


def rate(count, ns):
    return count / (ns / 1e9) if ns else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--contexts", default="0,512,1440", help="approximate prompt lengths in tokens")
    ap.add_argument("--steps", type=int, default=64, help="decoded tokens per request")
    ap.add_argument("--repeat", type=int, default=3, help="requests per context; the median is reported")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--json", help="also write the results here")
    a = ap.parse_args()

    rng = random.Random()
    # Load the model once so the first measurement is not a cold start.
    generate(a.host, a.model, "Hello", 4, 4096)
    rows = []
    print(f"{a.model}: decode {a.steps} tokens after N tokens of prompt (median of {a.repeat})")
    print(f"  {'N (asked)':>9} {'N (real)':>9} {'decode tok/s':>13} {'prefill tok/s':>14}")
    for n in (int(x) for x in a.contexts.split(",")):
        runs = []
        for _ in range(a.repeat):
            # About 1.05 tokens per word (Llama 3); Ollama reports the real count.
            prompt = " ".join(rng.choice(WORDS) for _ in range(max(1, int(n / 1.05))))
            r = generate(a.host, a.model, prompt, a.steps, max(2048, n + a.steps + 64))
            runs.append(
                (
                    r.get("prompt_eval_count", 0),
                    rate(r.get("eval_count", 0), r.get("eval_duration", 0)),
                    rate(r.get("prompt_eval_count", 0), r.get("prompt_eval_duration", 0)),
                )
            )
        runs.sort(key=lambda x: x[1])
        real, dec, pre = runs[len(runs) // 2]
        rows.append({"context": n, "prompt_tokens": real, "decode_tok_s": dec, "prefill_tok_s": pre})
        print(f"  {n:>9} {real:>9} {dec:>13.1f} {pre:>14.0f}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model": a.model, "steps": a.steps, "rows": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
