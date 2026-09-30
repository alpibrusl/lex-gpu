#!/usr/bin/env python3
"""Decode and prefill speed of one OpenAI-compatible server, measured the
same way whatever is behind it -- lex, vLLM, Ollama's /v1 -- so two runs can
be set side by side.

    python3 scripts/engine_bench.py --url http://localhost:8094 --model lex --name lex
    python3 scripts/engine_bench.py --url http://localhost:8000 --model Qwen3.8-27B-NVFP4 --name vllm

- **Decode:** the wall-clock difference between a 1-token and an (N+1)-token
  request for the same prose prompt, greedy and at the server's own default
  sampling. Both prefill the same prompt, so the difference is decode alone
  -- which is why prefix caching must be off (lex: LEX_NO_PREFIX_CACHE=1;
  vLLM: --no-enable-prefix-caching), or the second request skips a prefill
  the first paid for.
- **Prefill:** a fresh ~512-token prompt of random words per request, one
  generated token, prompt tokens over wall time. Random so no cache can hold
  it; the one decode step it includes is a few tens of ms against seconds.
- **Prompt tokens** are printed for the prose prompt: engines that render a
  chat template differently are timing different texts, and a draft head
  accepts some texts far more than others (docs/roadmap-weeks.md M5d).

One JSON line per run goes to --json, for a table built from several runs.
"""
import argparse, json, random, statistics, time, urllib.request

PROSE = ("Write a detailed, multi-paragraph essay on the history of the printing press, "
         "from Gutenberg to the digital age, covering its social and economic effects.")
WORDS = ("apple river stone light music paper garden window silver cloud forest ocean "
         "mountain story letter bridge candle harbor meadow lantern").split()


def post(url, body):
    r = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    j = json.load(urllib.request.urlopen(r, timeout=3600))
    return j, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True, help="the model name the server expects")
    ap.add_argument("--name", required=True, help="what to call this engine in the output")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prefill-words", type=int, default=400)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request")
    ap.add_argument("--json", help="append a result line here")
    a = ap.parse_args()
    url = a.url.rstrip("/") + "/v1/chat/completions"
    extra = json.loads(a.extra)
    msgs = [{"role": "user", "content": PROSE}]

    def chat(messages, n, **kw):
        return post(url, {"model": a.model, "messages": messages, "max_tokens": n, **extra, **kw})

    j, _ = chat(msgs, 8)  # warm: the first request after loading pays for paging
    prompt_tokens = j["usage"]["prompt_tokens"]
    print(f"{a.name}: prompt tokens {prompt_tokens}", flush=True)

    out = {"name": a.name, "prompt_tokens": prompt_tokens}
    for mode, samp in [("greedy", {"temperature": 0}), ("sampled", {})]:
        rates = []
        for rep in range(a.reps):
            _, t1 = chat(msgs, 1, seed=rep, **samp)
            j, t2 = chat(msgs, a.tokens + 1, seed=rep, **samp)
            n = j["usage"]["completion_tokens"]
            rates.append((n - 1) / (t2 - t1))
        out[f"decode_{mode}"] = statistics.median(rates)
        print(f"{a.name}: decode {mode:7} {statistics.median(rates):6.1f} tok/s "
              f"({', '.join(f'{r:.1f}' for r in rates)})", flush=True)

    rates, counts = [], []
    for rep in range(a.reps):
        rnd = random.Random(1000 + rep)
        text = " ".join(rnd.choice(WORDS) for _ in range(a.prefill_words))
        j, t = chat([{"role": "user", "content": text}], 1, temperature=0)
        counts.append(j["usage"]["prompt_tokens"])
        rates.append(counts[-1] / t)
    out["prefill"] = statistics.median(rates)
    out["prefill_tokens"] = statistics.median(counts)
    print(f"{a.name}: prefill {statistics.median(rates):6.1f} tok/s over "
          f"{int(statistics.median(counts))} tokens ({', '.join(f'{r:.1f}' for r in rates)})",
          flush=True)
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
