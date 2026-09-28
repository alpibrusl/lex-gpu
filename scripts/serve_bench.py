#!/usr/bin/env python3
"""Decode speed through the servers, lex against Ollama, the way a client sees it.

    LEX_NO_PREFIX_CACHE=1 target/release/examples/serve --model M --port 8094 &
    python3 scripts/serve_bench.py M 8094

Both are asked the same prose prompt through their chat endpoints, greedy
and at the model's own sampling defaults, for N decode tokens.

- **lex:** the difference between a 1-token and an (N+1)-token request.
  Both prefill the same prompt, so the difference is decode alone -- which
  is why the server must run with `LEX_NO_PREFIX_CACHE`, or the second
  request skips a prefill the first one paid for. One untimed request goes
  first: the first pair after loading pays for paging the weights in, and
  it lands on whichever request comes first.
- **Ollama:** its own `eval_count / eval_duration`, which is decode alone.

Prose, not random words: a model that speculates accepts more of noise
than of text, so random words flatter the decode of both engines.
"""
import json, statistics, sys, time, urllib.request

PROMPT = ("Write a detailed, multi-paragraph essay on the history of the printing press, "
          "from Gutenberg to the digital age, covering its social and economic effects.")


def post(url, body):
    r = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.time()
    j = json.load(urllib.request.urlopen(r, timeout=1800))
    return j, time.time() - t


def main():
    model, port = sys.argv[1], int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    lex = f"http://localhost:{port}/v1/chat/completions"
    msgs = [{"role": "user", "content": PROMPT}]
    post(lex, {"model": "lex", "messages": msgs, "max_tokens": 8})

    for mode, samp in [("greedy", {"temperature": 0}), ("sampled", {})]:
        ours = []
        for rep in range(reps):
            _, t1 = post(lex, dict(samp, model="lex", messages=msgs, max_tokens=1, seed=rep))
            j, t2 = post(lex, dict(samp, model="lex", messages=msgs, max_tokens=n + 1, seed=rep))
            ours.append((j["usage"]["completion_tokens"] - 1) / (t2 - t1))
        theirs = []
        for rep in range(reps):
            opts = {"num_predict": n, "seed": rep, **samp}
            j, _ = post("http://localhost:11434/api/chat",
                        {"model": model, "messages": msgs, "stream": False, "options": opts})
            theirs.append(j["eval_count"] / (j["eval_duration"] / 1e9))
        a, b = statistics.median(ours), statistics.median(theirs)
        show = lambda v: ", ".join(f"{x:.1f}" for x in v)
        print(f"{model:24} {mode:8} lex {a:6.1f} ({show(ours)})  "
              f"ollama {b:6.1f} ({show(theirs)})  {a / b:.2f}x", flush=True)


if __name__ == "__main__":
    main()
