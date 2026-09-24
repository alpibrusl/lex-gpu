"""Expected token ids for a spread of texts, from the reference tokenizer.

    python3 scripts/tokenizer_fixture.py > crates/lex-rt/tests/data/qwen_tokenizer.txt

`tokenizers` is the implementation `tokenizer.json` is written for, so it
is the oracle for ours. It is a build-time dependency of this fixture and
not of the crate: the file it writes is what the test reads, so the Rust
side has no Python and no network in its path.

The cases are chosen to break things rather than to pass: the byte-level
alphabet has a hole for every character outside Latin-1, digits split one
at a time in this vocabulary, the split regex has a case-insensitive
contraction rule and a lookahead for trailing whitespace, and `ignore_merges`
means a pre-token present in the vocabulary must not be merged into.
"""

import json
import pathlib
import sys

from tokenizers import Tokenizer

CASES = [
    "",
    " ",
    "\n",
    "   \n\n  ",
    "The capital of France is",
    "hello world",
    "Hello, World!",
    "don't  DON'T  Don'T",          # the case-insensitive contraction rule
    "12345 007 3.14159",            # digits, which this vocab splits singly
    "trailing spaces   ",           # the (?!\\S) branch
    "  leading spaces",
    "tabs\tand\r\nnewlines\n\n",
    "naïve café résumé",            # Latin-1 beyond ASCII
    "日本語のテキストです",              # no Latin-1 at all
    "Здравствуй, мир",
    "🙂🚀 emoji 👨‍👩‍👧‍👦 zwj",         # astral plane and a ZWJ sequence
    "é vs é",            # combining mark against precomposed
    "def f(x: int) -> int:\n    return x * 2\n",
    "a" * 200,                      # one long merge chain
    "<|im_start|>user\nhi<|im_end|>",  # added tokens
]


def main():
    root = pathlib.Path.home() / ".ollama/models"
    man = root / "manifests/registry.ollama.ai/library/qwen3.8/27b-mlx"
    layers = json.loads(man.read_text())["layers"]
    blob = None
    for l in layers:
        if l["mediaType"].endswith(".json") and l["size"] > 10e6:
            blob = root / "blobs" / ("sha256-" + l["digest"].split(":")[1])
    if blob is None:
        raise SystemExit("no tokenizer.json in the qwen3.8:27b-mlx manifest")

    tok = Tokenizer.from_file(str(blob))
    print("# text and ids from `tokenizers`, via scripts/tokenizer_fixture.py")
    print(f"# tokenizer.json sha256-{blob.name.split('-')[1]}")
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False).ids
        # The text escaped onto one line; the ids after a tab.
        # `ensure_ascii=False`: the escapes stay for the characters that
        # need them (quote, backslash, the control ones) and every other
        # character is itself. With escaping on, an emoji becomes a
        # surrogate pair, which is not a Rust `char` and cannot be read
        # back one escape at a time.
        print(json.dumps(text, ensure_ascii=False) + "\t" + ",".join(str(i) for i in ids))
    print(f"# {len(CASES)} cases", file=sys.stderr)


if __name__ == "__main__":
    main()
