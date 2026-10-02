#!/usr/bin/env python3
"""Decode and prefill speed of any OpenAI-compatible server, timed from the
stream itself -- so engines whose prefix cache cannot be turned off
(Magnitude, Ollama) are measured exactly as those whose can (lex).

    python3 scripts/stream_bench.py --url http://127.0.0.1:8094/v1 --model lex --name lex
    python3 scripts/stream_bench.py --url http://127.0.0.1:10100/inference/v1 \\
        --model qwen3.8-27b:gguf:q4 --name magnitude

- **Decode:** tokens after the first, over the time from the first streamed
  token to the last. The prompt's prefill is before the first token, so it
  is outside the window whatever a cache did with it. The count comes from
  the stream's own `usage` (`stream_options.include_usage`): with
  speculation one chunk carries several tokens, so counting chunks would
  undercount.
- **Prefill:** prompt tokens over the time to the first streamed token, on
  a fresh prompt of random words each time so no cache can hold it. It
  includes one decode step, tens of ms against seconds.
- **Prompt tokens** are printed: engines that render a chat template
  differently are timing different texts (docs/roadmap-weeks.md M5d).

`--corpus FILE` adds the edit-corpus prompts (JSON list of {name, prompt}),
each timed for decode the same way.
"""
import argparse, json, random, statistics, time, urllib.request

PROSE = ("Write a detailed, multi-paragraph essay on the history of the printing press, "
         "from Gutenberg to the digital age, covering its social and economic effects.")
WORDS = ("apple river stone light music paper garden window silver cloud forest ocean "
         "mountain story letter bridge candle harbor meadow lantern").split()


def stream(url, body, timeout=3600):
    """(prompt tokens, completion tokens, t_request, t_first, t_last)."""
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    first = last = None
    usage = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            j = json.loads(data)
            if j.get("usage"):
                usage = j["usage"]
            for c in j.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                    now = time.time()
                    first = first or now
                    last = now
                    chunks += 1
    if usage is None:
        raise SystemExit("the server sent no usage in the stream; cannot count tokens")
    return usage["prompt_tokens"], usage["completion_tokens"], t0, first, last, chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="the OpenAI base URL, ending in /v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--prefill-words", type=int, default=400)
    ap.add_argument("--extra", default="{}", help="JSON merged into every request")
    ap.add_argument("--corpus", help="JSON list of {name, prompt} timed for decode")
    ap.add_argument("--json", help="append a result line here")
    a = ap.parse_args()
    url = a.url.rstrip("/") + "/chat/completions"
    extra = json.loads(a.extra)

    def chat(text, n, **kw):
        return stream(url, {"model": a.model, "messages": [{"role": "user", "content": text}],
                            "max_tokens": n, **extra, **kw})

    chat("hi", 4)  # load and warm
    out = {"name": a.name}

    def decode(label, text, n, **kw):
        rates, prompts, lens = [], [], []
        for rep in range(a.reps):
            p, c, t0, t1, t2, chunks = chat(text, n, seed=rep, **kw)
            prompts.append(p)
            lens.append(c)
            if c > 1 and t2 > t1:
                rates.append((c - 1) / (t2 - t1))
        r = statistics.median(rates) if rates else 0.0
        print(f"{a.name}: decode {label:16} {r:6.1f} tok/s  ({', '.join(f'{x:.1f}' for x in rates)}; "
              f"prompt {prompts[0]} tokens, {statistics.median(lens):.0f} generated)", flush=True)
        return r, prompts[0]

    out["decode_greedy"], out["prompt_tokens"] = decode("prose greedy", PROSE, a.tokens, temperature=0)
    out["decode_sampled"], _ = decode("prose sampled", PROSE, a.tokens)

    rates, counts = [], []
    for rep in range(a.reps):
        rnd = random.Random(1000 + rep)
        text = " ".join(rnd.choice(WORDS) for _ in range(a.prefill_words))
        p, c, t0, t1, t2, _ = chat(text, 1, temperature=0)
        counts.append(p)
        rates.append(p / (t1 - t0))
    out["prefill"] = statistics.median(rates)
    print(f"{a.name}: prefill {statistics.median(rates):6.1f} tok/s over {int(statistics.median(counts))} "
          f"tokens ({', '.join(f'{x:.1f}' for x in rates)})", flush=True)

    if a.corpus:
        for task in json.load(open(a.corpus)):
            out[f"corpus_{task['name']}"], _ = decode(task["name"], task["prompt"], 1000, temperature=0)

    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
