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
import pathlib
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


def tokenise(host, model, text):
    """The ids Ollama gives this exact text, so both engines see one thing.

    There is no tokenizer in this repository and no tokenise endpoint, but
    a `raw` generate returns `context` -- the ids of the prompt plus what
    it produced -- so one token of output and a trim gives the prompt's.
    `raw` skips the chat template, which is the point: a templated prompt
    would put markup in front of the text and the two engines would be
    reading different things.
    """
    req = {
        "model": model,
        "prompt": text,
        "raw": True,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 1},
    }
    r = urllib.request.Request(
        f"{host}/api/generate",
        json.dumps(req).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(r, timeout=1800) as f:
        return json.load(f).get("context") or []


def main():
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8:27b-mlx")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--tokens", type=int, default=1440, help="ids to emit")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--text-out", help="write the passage here, for a raw prompt")
    a = ap.parse_args()

    # Generate a passage, then ask for the ids of that passage alone. The
    # `context` of the generating call would also carry the instruction
    # that produced it, which is not what the other engine would be given.
    r = generate(a.host, a.model, a.prompt, a.tokens + 200, a.temperature)
    text = r.get("response") or ""
    if not text:
        raise SystemExit("the model returned nothing")

    ids = tokenise(a.host, a.model, text)[:-1]
    if len(ids) < a.tokens:
        raise SystemExit(
            f"only {len(ids)} ids for the passage, {a.tokens} asked; raise --tokens"
        )
    ids = ids[: a.tokens]

    # Trim the text to the same ids, so the two engines read one passage.
    # Bisect on characters: tokenising is the only length oracle there is.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenise(a.host, a.model, text[:mid])) - 1 < a.tokens:
            lo = mid + 1
        else:
            hi = mid
    cut = text[:lo]
    if a.text_out:
        pathlib.Path(a.text_out).write_text(cut)

    print(",".join(str(i) for i in ids))
    repeats = sum(1 for i in range(1, len(ids)) if ids[i] == ids[i - 1])
    print(
        f"# {len(ids)} ids, {len(set(ids))} distinct, "
        f"{100.0 * repeats / max(1, len(ids) - 1):.1f}% immediate repeats, "
        f"{len(cut)} chars of text",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
