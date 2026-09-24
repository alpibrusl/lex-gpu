"""Get a long stretch of *coherent* token ids, for measuring acceptance.

    python3 scripts/ollama_context.py --tokens 1440 > /tmp/ctx.txt
    LEX_CONTEXT=0 cargo run --release -p lex-rt --example mtp -- \
        --steps 48 --depth 1 --ids "$(cat /tmp/ctx.txt)"

Every acceptance number in this repository so far was measured on 1440
*random* token ids, because that is what `examples/mtp` fills with and
what `ollama_bench.py` prompts with. Random filler is right for timing --
it defeats the prompt cache, and prefill cost does not depend on which
tokens they are -- and wrong for acceptance, which is entirely a question
of how predictable the text is. Measured: 89% acceptance after a real
five-token prompt, 83% after a random five-token one, 70% after 1440
random ones.

So the question the roadmap cannot answer is what acceptance does at
length on text that reads like text. This gets that without a tokenizer
in the repo: Ollama's `/api/generate` returns `context`, the token ids of
the prompt and its own reply, and the model generating the passage is the
same one that will be asked to predict it.

That last part is worth being explicit about, because it cuts both ways.
Self-generated text is in-distribution and greedy decoding can drift into
repetition, which would flatter acceptance. `--temperature` defaults to
0.8 rather than 0 for exactly that reason, and the script reports the
fraction of tokens that are repeats of the previous one so the result can
be thrown away if it looks degenerate.
"""

import argparse
import json
import urllib.request

PROMPT = (
    "Write a long, detailed essay about the history of maritime navigation, "
    "from Polynesian wayfinding to satellite positioning. Cover instruments, "
    "the longitude problem, notable voyages and the people involved. Write "
    "continuous prose in many paragraphs. Do not use lists or headings."
)


def generate(host, model, prompt, steps, temperature):
    req = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": steps, "temperature": temperature},
    }
    r = urllib.request.Request(
        f"{host}/api/generate",
        json.dumps(req).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=1800) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8:27b-mlx")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--tokens", type=int, default=1440, help="ids to emit")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompt", default=PROMPT)
    a = ap.parse_args()

    # Ask for more than wanted: `context` includes the prompt, and the
    # model may stop early.
    r = generate(a.host, a.model, a.prompt, a.tokens, a.temperature)
    ids = r.get("context") or []
    if len(ids) < a.tokens:
        raise SystemExit(
            f"only {len(ids)} ids came back for {a.tokens} asked; "
            "raise --tokens or check the model is loaded"
        )
    ids = ids[: a.tokens]

    repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
    print(",".join(str(i) for i in ids))
    print(
        f"# {len(ids)} ids, {len(set(ids))} distinct, "
        f"{100.0 * repeats / max(1, len(ids) - 1):.1f}% immediate repeats",
        file=__import__("sys").stderr,
    )


if __name__ == "__main__":
    main()
