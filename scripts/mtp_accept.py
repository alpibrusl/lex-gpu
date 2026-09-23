"""Measure the acceptance rate of Qwen3.8's own multi-token-prediction head.

    cargo run --release -p lex-rt --example mtp_trace -- --steps 256
    python3 scripts/mtp_accept.py --trace mtp_trace.bin

Speculative decoding is worth exactly `accepted tokens / passes`, and the
verify cost is already measured (1.04 passes for two tokens). Acceptance is
the other half, and every projection in `docs/qwen.md` currently borrows it
from someone else's implementation. This measures it here, in f32, before
any of it is committed to kernels.

The head drafts token `t+2` from the model's hidden state at `t` and the
embedding of token `t+1`:

    z = fc(concat(norm_hidden(h_t), norm_embedding(embed(x_{t+1}))))
    z = z + attention(input_layernorm(z))
    z = z + mlp(post_attention_layernorm(z))
    draft = argmax(lm_head(norm(z)))

`mtp_trace.bin` supplies `h_t` and `x_{t+1}` from a real decode run, so this
needs one small layer rather than a second copy of the model.

Two things about the checkpoint are inferred rather than documented, and
`--variants` tries them all and reports every combination:

- **Which norms are stored as a delta from 1.** `mlx_lm`'s sanitize() shifts
  a fixed list of suffixes, and it drops the mtp weights before it runs, so
  the head is not covered. Shifting all of them is the reading that makes
  every gain plausible: unshifted, `pre_fc_norm_embedding` runs -0.75..-0.19,
  entirely negative, which no RMSNorm gain should be.
- **The order of the concatenation** into `fc`.

A wrong choice does not look marginal. It reads as acceptance near chance,
the same way the `x * (1 + w)` convention showed up as a model that
predicted noise while every weight statistic looked ordinary.
"""

import argparse
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from qwen_nvfp4 import read_tensor  # noqa: E402
from qwen_ref import attention, config, mlp, rms_norm  # noqa: E402

MODEL = "qwen3.8:27b-mlx"
MAGIC = 0x4D545030

# Shifted by mlx_lm's sanitize() for the main model; the same suffixes appear
# inside the head.
SHIFTED_SUFFIX = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)
# Not covered by any of mlx_lm's suffixes, so their convention is a guess.
UNCOVERED = ("mtp.norm.weight", "mtp.pre_fc_norm_hidden.weight",
             "mtp.pre_fc_norm_embedding.weight")


def read_trace(path):
    raw = pathlib.Path(path).read_bytes()
    steps, hidden, magic = struct.unpack_from("<III", raw, 0)
    if magic != MAGIC:
        sys.exit(f"{path}: not an mtp trace (magic {magic:#x})")
    rec = 4 + hidden * 4
    if len(raw) != 12 + steps * rec:
        sys.exit(f"{path}: {len(raw)} bytes for {steps} x {hidden}")
    toks = np.empty(steps, dtype=np.uint32)
    hs = np.empty((steps, hidden), dtype=np.float32)
    for i in range(steps):
        off = 12 + i * rec
        toks[i] = struct.unpack_from("<I", raw, off)[0]
        hs[i] = np.frombuffer(raw, np.float32, hidden, off + 4)
    return toks, hs


class Head:
    """The mtp weights, loaded once, with a chosen norm convention."""

    def __init__(self, shift_uncovered, verbose=False):
        self.cache = {}
        self.shift_uncovered = shift_uncovered
        self.verbose = verbose

    def __call__(self, name):
        if name not in self.cache:
            t, _ = read_tensor(MODEL, name)
            v = np.ascontiguousarray(np.asarray(t, dtype=np.float32))
            if v.ndim == 1:
                shift = name.endswith(SHIFTED_SUFFIX) or (
                    self.shift_uncovered and name in UNCOVERED
                )
                if shift:
                    v = v + 1.0
            self.cache[name] = v
            if self.verbose:
                print(f"    loaded {name} {v.shape}", file=sys.stderr)
        return self.cache[name]


def run(head, cfg, embed, lm_head, toks, hs, hidden_first):
    eps = cfg["rms_norm_eps"]
    cache = {"k": None, "v": None}
    drafts = np.zeros(len(toks), dtype=np.uint32)
    for i in range(len(toks)):
        hn = rms_norm(hs[i], head("mtp.pre_fc_norm_hidden.weight"), eps)
        en = rms_norm(embed[toks[i]], head("mtp.pre_fc_norm_embedding.weight"), eps)
        pair = (hn, en) if hidden_first else (en, hn)
        z = (np.concatenate(pair)[None, :] @ head("mtp.fc.weight").T).astype(np.float32)
        z = z + attention(
            head,
            "mtp.layers.0.self_attn.",
            rms_norm(z, head("mtp.layers.0.input_layernorm.weight"), eps),
            cfg,
            cache,
            i,
        )
        z = z + mlp(
            head,
            "mtp.layers.0.mlp.",
            rms_norm(z, head("mtp.layers.0.post_attention_layernorm.weight"), eps),
        )
        out = rms_norm(z, head("mtp.norm.weight"), eps)
        drafts[i] = int(np.argmax(out[0] @ lm_head.T))
    # Step i drafts the token that step i+1 actually chose.
    return drafts[:-1], toks[1:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", default="mtp_trace.bin")
    p.add_argument("--variants", action="store_true",
                   help="try both norm conventions and both concat orders")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()

    toks, hs = read_trace(a.trace)
    print(f"{a.trace}: {len(toks)} steps x {hs.shape[1]}")
    cfg = config(MODEL)
    print("loading embeddings and lm_head ...", file=sys.stderr)
    embed, _ = read_tensor(MODEL, "model.language_model.embed_tokens.weight")
    embed = np.asarray(embed, dtype=np.float32)
    lm_head, _ = read_tensor(MODEL, "lm_head.weight")
    lm_head = np.asarray(lm_head, dtype=np.float32)

    # The measured answer: every norm shifted, and the embedding ahead of
    # the hidden state in the concatenation. The other three combinations
    # accept under 1%, so this is not a preference, it is the checkpoint.
    combos = (
        [(s, h) for s in (True, False) for h in (True, False)]
        if a.variants
        else [(True, False)]
    )
    print(f"\n{'shift uncovered':>16} {'concat':>14} {'accepted':>10} {'rate':>8}")
    best = None
    for shift, hidden_first in combos:
        head = Head(shift, a.verbose)
        drafts, actual = run(head, cfg, embed, lm_head, toks, hs, hidden_first)
        ok = int((drafts == actual).sum())
        rate = ok / len(actual)
        order = "hidden,embed" if hidden_first else "embed,hidden"
        print(f"{str(shift):>16} {order:>14} {ok:>6}/{len(actual):<3} {rate:>7.1%}")
        if best is None or rate > best[0]:
            best = (rate, shift, hidden_first)

    rate, shift, hidden_first = best
    print(f"\nbest: {rate:.1%} (shift uncovered norms: {shift}, "
          f"{'hidden' if hidden_first else 'embed'} first)")
    if rate < 0.5:
        print("\nUnder 50% is not a draft head, it is a bug: every variant "
              "here is wrong, or the head takes a different hidden state.")


if __name__ == "__main__":
    main()
