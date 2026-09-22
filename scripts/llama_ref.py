"""Reference Llama forward pass, checked against a model served by Ollama.

This is the oracle chain for P2's exit test:

    Ollama (llama.cpp, black box)  <-- this script checks -->  PyTorch reference
    PyTorch reference              <-- Rust tests check   -->  tile on the GPU

The reference reads the *same GGUF file* Ollama serves, dequantises it with
`gguf.py`, and runs a plain float32 Llama forward pass in PyTorch. It asks
Ollama for the same greedy continuation with log-probabilities, and compares
step by step:

- the greedy token at every step must be identical;
- the log-probability of every token in Ollama's top-k must agree within a
  tolerance. Exact equality is not expected: llama.cpp quantises activations
  to 8 bits inside its Q8_0 matmuls, and the reference does not.

    python3 scripts/llama_ref.py                          # compare, default prompts
    python3 scripts/llama_ref.py --prompt "Once upon a time" --steps 16
    python3 scripts/llama_ref.py --write-golden           # fixture for the Rust tests
"""

import argparse
import json
import math
import pathlib
import sys
import urllib.request

import numpy as np
import torch
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf import GGUF, ollama_model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "crates/tile-rt/tests/data/llama32_1b_golden.txt"

# Llama 3's pre-tokenizer split, as llama.cpp's "llama-bpe" uses it.
LLAMA3_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def tokenizer(g):
    tokens = g.meta["tokenizer.ggml.tokens"]
    merges = [tuple(m.split(" ", 1)) for m in g.meta["tokenizer.ggml.merges"]]
    tk = Tokenizer(models.BPE(vocab={t: i for i, t in enumerate(tokens)}, merges=merges, ignore_merges=True))
    tk.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(LLAMA3_SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tk.decoder = decoders.ByteLevel()
    return tk


class Llama:
    """Float32 Llama 3 forward pass over dequantised GGUF weights."""

    def __init__(self, g, device="cpu"):
        m = g.meta
        self.n_layer = m["llama.block_count"]
        self.n_head = m["llama.attention.head_count"]
        self.n_kv = m["llama.attention.head_count_kv"]
        self.eps = m["llama.attention.layer_norm_rms_epsilon"]
        self.dim = m["llama.embedding_length"]
        self.hd = self.dim // self.n_head
        t = lambda n: torch.from_numpy(np.ascontiguousarray(g.tensor(n))).to(device)
        self.emb = t("token_embd.weight")  # [vocab, dim]
        self.out = t("output.weight") if "output.weight" in g.tensors else self.emb
        self.norm = t("output_norm.weight")
        self.layers = []
        for i in range(self.n_layer):
            p = f"blk.{i}."
            self.layers.append(
                {k: t(p + k + ".weight") for k in
                 ["attn_norm", "attn_q", "attn_k", "attn_v", "attn_output",
                  "ffn_norm", "ffn_gate", "ffn_up", "ffn_down"]}
            )
        # Llama 3 RoPE: llama.cpp divides each frequency by a stored factor.
        base = m["llama.rope.freq_base"]
        n_rot = m["llama.rope.dimension_count"]
        inv = base ** (-torch.arange(0, n_rot, 2, dtype=torch.float64) / n_rot)
        if "rope_freqs.weight" in g.tensors:
            inv = inv / torch.from_numpy(g.tensor("rope_freqs.weight")).double()
        self.inv_freq = inv.float().to(device)
        self.device = device

    def rms(self, x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos):
        # GGUF's Q/K are permuted for llama.cpp's "normal" RoPE: rotate
        # adjacent pairs (2i, 2i+1), not HF's split halves.
        ang = pos[:, None].float() * self.inv_freq[None, :]  # [T, hd/2]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x0 * cos - x1 * sin
        out[..., 1::2] = x0 * sin + x1 * cos
        return out

    @torch.no_grad()
    def logits(self, ids):
        T = len(ids)
        pos = torch.arange(T, device=self.device)
        x = self.emb[torch.tensor(ids, device=self.device)]
        mask = torch.full((T, T), float("-inf"), device=self.device).triu(1)
        for L in self.layers:
            h = self.rms(x, L["attn_norm"])
            q = (h @ L["attn_q"].T).view(T, self.n_head, self.hd)
            k = (h @ L["attn_k"].T).view(T, self.n_kv, self.hd)
            v = (h @ L["attn_v"].T).view(T, self.n_kv, self.hd)
            q, k = self.rope(q, pos), self.rope(k, pos)
            rep = self.n_head // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            att = torch.einsum("thd,shd->hts", q, k) / math.sqrt(self.hd) + mask
            o = torch.einsum("hts,shd->thd", att.softmax(-1), v).reshape(T, self.dim)
            x = x + o @ L["attn_output"].T
            h = self.rms(x, L["ffn_norm"])
            x = x + (torch.nn.functional.silu(h @ L["ffn_gate"].T) * (h @ L["ffn_up"].T)) @ L["ffn_down"].T
        return self.rms(x, self.norm) @ self.out.T  # [T, vocab]


def ollama(model, prompt, steps, top):
    req = {
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": False,
        "logprobs": True,
        "top_logprobs": top,
        "options": {"temperature": 0, "num_predict": steps, "seed": 0},
    }
    r = urllib.request.Request(
        "http://localhost:11434/api/generate", json.dumps(req).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(r, timeout=600) as f:
        return json.load(f)["logprobs"]


def compare(model, lm, tk, prompt, steps, top, bos, tol):
    """Greedy-decode `steps` tokens and compare with Ollama. Returns (ok, record)."""
    ref = ollama(model, prompt, steps, top)
    ids = ([bos] if bos is not None else []) + tk.encode(prompt, add_special_tokens=False).ids
    prompt_ids = list(ids)
    by_bytes = {}
    for tok, i in tk.get_vocab().items():
        by_bytes[bytes(tk.decode([i]).encode("utf-8", "surrogatepass"))] = i
    worst, ok, steps_out = 0.0, True, []
    print(f"\nprompt {prompt!r}  ({len(prompt_ids)} tokens, bos={bos is not None})")
    for step, o in enumerate(ref):
        lp = torch.log_softmax(lm.logits(ids)[-1].double(), -1)
        mine = int(lp.argmax())
        want = by_bytes.get(bytes(o["bytes"]))
        diffs = []
        for alt in o["top_logprobs"]:
            j = by_bytes.get(bytes(alt["bytes"]))
            if j is None:
                continue
            diffs.append(abs(float(lp[j]) - alt["logprob"]))
        d = max(diffs) if diffs else float("inf")
        worst = max(worst, d)
        same = mine == want
        ok &= same and d <= tol
        print(
            f"  {step:2d}  ollama {o['token']!r:14} tile-ref {tk.decode([mine])!r:14} "
            f"{'same' if same else 'DIFF'}   max |dlogprob| over top-{top} {d:.4f}"
        )
        topk = torch.topk(lp, top)
        nxt = want if want is not None else mine
        steps_out.append(
            {
                "next": nxt,
                "top_ids": topk.indices.tolist(),
                "top_logprobs": [float(x) for x in topk.values],
            }
        )
        ids.append(nxt)
    print(f"  worst |dlogprob| {worst:.4f} (tolerance {tol})  {'PASS' if ok else 'FAIL'}")
    return ok, {"prompt": prompt, "prompt_ids": prompt_ids, "steps": steps_out}


def write_golden(model, blob, cases):
    """A line format the Rust test reads without a JSON dependency.

        model <tag> <blob>
        case <prompt ids...>
        step <next id> <id>:<logprob> ... (reference top-k, best first)
    """
    lines = [f"model {model} {blob}"]
    for c in cases:
        lines.append("case " + " ".join(map(str, c["prompt_ids"])))
        for st in c["steps"]:
            pairs = " ".join(f"{i}:{lp:.6f}" for i, lp in zip(st["top_ids"], st["top_logprobs"]))
            lines.append(f"step {st['next']} {pairs}")
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {GOLDEN}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama3.2:1b")
    ap.add_argument("--prompt", action="append")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--no-bos", action="store_true", help="do not prepend <|begin_of_text|>")
    ap.add_argument("--write-golden", action="store_true")
    a = ap.parse_args()
    prompts = a.prompt or ["The capital of France is", "def fibonacci(n):", "Once upon a time, in a small village,"]

    g = GGUF(ollama_model(a.model))
    tk = tokenizer(g)
    bos = None if a.no_bos else g.meta["tokenizer.ggml.bos_token_id"]
    print(f"model {a.model}: {g.path.name[:19]}..., {len(g.tensors)} tensors")
    lm = Llama(g)
    results, all_ok = [], True
    for p in prompts:
        ok, rec = compare(a.model, lm, tk, p, a.steps, a.top, bos, a.tol)
        all_ok &= ok
        results.append(rec)
    if a.write_golden:
        write_golden(a.model, g.path.name, results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
