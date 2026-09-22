"""An f32 reference for Qwen3.5 (qwen3.8:27b-mlx), straight from Ollama's store.

    python3 scripts/qwen_ref.py --steps 8
    python3 scripts/qwen_ref.py --prompt "The capital of France is" --compare

What `llama_ref.py` is for the Llama models: an independent implementation
that owns no kernels, so a disagreement with tile points at tile, and a
disagreement with Ollama points at quantised rounding. It follows the
published architecture (`mlx_lm/models/qwen3_5.py`, `qwen3_next.py`,
`gated_delta.py`), in f32 numpy.

The model is 27.8B parameters and does not fit in memory dequantised, so
each layer's weights are read and dequantised as it runs and dropped after
(~15 s per forward pass, most of it dequantisation).

Sixty-four layers: every fourth (`full_attention_interval`) is grouped
attention with a 256-wide head, partial RoPE over the first quarter of it,
q/k norms and a sigmoid output gate. The other 48 are gated delta layers:
a depthwise conv over the last four positions, then a recurrent state
`S[head, v_dim, k_dim]` updated by the delta rule

    S = S * g;  S += k (v - S k)^T beta;  y = S q

so their "cache" is a fixed 3 MB per layer whatever the context length.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from qwen_nvfp4 import manifest, read_tensor  # noqa: E402

MODEL = "qwen3.8:27b-mlx"


def config(model):
    """The text model's configuration, from the config.json blob."""
    m = manifest(model)
    store = pathlib.Path.home() / ".ollama/models/blobs"
    for l in m["layers"]:
        if l.get("name") == "config.json":
            c = json.loads((store / f"sha256-{l['digest'][7:]}").read_text())
            return c["text_config"]
    sys.exit("no config.json in the manifest")


def tokenizer(model):
    from tokenizers import Tokenizer

    m = manifest(model)
    store = pathlib.Path.home() / ".ollama/models/blobs"
    for l in m["layers"]:
        if l.get("name") == "tokenizer.json":
            return Tokenizer.from_file(str(store / f"sha256-{l['digest'][7:]}"))
    sys.exit("no tokenizer.json in the manifest")


def rms_norm(x, w, eps):
    v = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
    y = x / np.sqrt(v + eps)
    return y if w is None else y * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


class Weights:
    """Tensors by name, dequantised on demand and not held."""

    #: Qwen3.5 stores these norm weights as deltas from 1: the norm is
    #: `x * (1 + w)` (`Qwen3_5RMSNorm` in transformers, `sanitize` in
    #: mlx_lm). The gated norm inside a linear-attention layer is not one
    #: of them and multiplies by `w` directly. Missing this is silent: the
    #: weights look perfectly ordinary and the model predicts noise.
    SHIFTED = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.language_model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )

    def __init__(self, model):
        self.model = model
        self.names = {l["name"] for l in manifest(model)["layers"] if "name" in l}

    def __call__(self, name):
        w = read_tensor(self.model, name)[0]
        return w + 1.0 if name.endswith(Weights.SHIFTED) else w

    def has(self, name):
        return name in self.names


def rope(x, pos, dims, theta):
    """RoPE over the first `dims` of the last axis, halves paired
    (`traditional=False`): `x[i]` with `x[i + dims/2]`."""
    half = dims // 2
    freqs = theta ** (-np.arange(half, dtype=np.float32) / half)
    ang = np.asarray(pos, dtype=np.float32)[:, None] * freqs[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    out = x.copy()
    a, b = x[..., :half], x[..., half:dims]
    out[..., :half] = a * cos - b * sin
    out[..., half:dims] = b * cos + a * sin
    return out


def attention(w, pre, x, cfg, cache, pos0):
    """Full attention: q carries its own gate, RoPE is partial, and the
    output is gated by `sigmoid(gate)` before `o_proj`."""
    t, d = x.shape
    heads, kv, hd = (
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
        cfg["head_dim"],
    )
    eps = cfg["rms_norm_eps"]
    rot = int(hd * cfg["rope_parameters"]["partial_rotary_factor"])
    theta = float(cfg["rope_parameters"]["rope_theta"])

    qg = (x @ w(pre + "q_proj.weight").T).reshape(t, heads, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:].reshape(t, heads * hd)
    k = (x @ w(pre + "k_proj.weight").T).reshape(t, kv, hd)
    v = (x @ w(pre + "v_proj.weight").T).reshape(t, kv, hd)
    q = rms_norm(q, w(pre + "q_norm.weight"), eps)
    k = rms_norm(k, w(pre + "k_norm.weight"), eps)
    pos = np.arange(pos0, pos0 + t)
    q, k = rope(q, pos, rot, theta), rope(k, pos, rot, theta)

    if cache["k"] is None:
        cache["k"], cache["v"] = k, v
    else:
        cache["k"] = np.concatenate([cache["k"], k], 0)
        cache["v"] = np.concatenate([cache["v"], v], 0)
    kk, vv = cache["k"], cache["v"]
    n = kk.shape[0]

    out = np.empty((t, heads, hd), dtype=np.float32)
    per = heads // kv
    for h in range(heads):
        s = (q[:, h, :] @ kk[:, h // per, :].T) * (hd**-0.5)
        # Causal: query i is at position pos0 + i, keys run 0..n-1.
        j = np.arange(n)[None, :]
        s = np.where(j <= (pos0 + np.arange(t))[:, None], s, -np.inf)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        out[:, h, :] = s @ vv[:, h // per, :]
    o = out.reshape(t, heads * hd) * (1.0 / (1.0 + np.exp(-gate)))
    return o @ w(pre + "o_proj.weight").T


def gated_delta(w, pre, x, cfg, cache):
    """A linear-attention layer: depthwise conv over the last four
    positions, then the delta rule over a per-head state."""
    t = x.shape[0]
    hk, hv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
    dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
    key_dim, val_dim = hk * dk, hv * dv
    kern = cfg["linear_conv_kernel_dim"]

    qkv = x @ w(pre + "in_proj_qkv.weight").T  # [t, 2*key_dim + val_dim]
    z = (x @ w(pre + "in_proj_z.weight").T).reshape(t, hv, dv)
    b = x @ w(pre + "in_proj_b.weight").T  # [t, hv]
    a = x @ w(pre + "in_proj_a.weight").T

    # Depthwise conv over [conv_state, qkv]; the state is the last
    # `kern - 1` inputs.
    cw = w(pre + "conv1d.weight").reshape(qkv.shape[1], kern)
    if cache["conv"] is None:
        cache["conv"] = np.zeros((kern - 1, qkv.shape[1]), dtype=np.float32)
    seq = np.concatenate([cache["conv"], qkv], 0)
    cache["conv"] = seq[-(kern - 1) :].copy()
    conv = np.zeros((t, qkv.shape[1]), dtype=np.float32)
    for i in range(kern):
        conv += seq[i : i + t] * cw[:, i]
    conv = silu(conv)

    q = conv[:, :key_dim].reshape(t, hk, dk)
    k = conv[:, key_dim : 2 * key_dim].reshape(t, hk, dk)
    v = conv[:, 2 * key_dim :].reshape(t, hv, dv)
    inv = dk**-0.5
    q = rms_norm(q, None, 1e-6) * inv * inv
    k = rms_norm(k, None, 1e-6) * inv

    beta = 1.0 / (1.0 + np.exp(-b))
    g = np.exp(-np.exp(w(pre + "A_log").astype(np.float32)) * softplus(a + w(pre + "dt_bias")))

    if cache["state"] is None:
        cache["state"] = np.zeros((hv, dv, dk), dtype=np.float32)
    state = cache["state"]
    y = np.empty((t, hv, dv), dtype=np.float32)
    rep = hv // hk
    for i in range(t):
        ki = np.repeat(k[i], rep, axis=0)  # [hv, dk]
        qi = np.repeat(q[i], rep, axis=0)
        state *= g[i][:, None, None]
        kv_mem = (state * ki[:, None, :]).sum(-1)  # [hv, dv]
        delta = (v[i] - kv_mem) * beta[i][:, None]
        state += ki[:, None, :] * delta[..., None]
        y[i] = (state * qi[:, None, :]).sum(-1)
    cache["state"] = state

    # Gated RMSNorm per value head, then out_proj.
    y = rms_norm(y, w(pre + "norm.weight"), cfg["rms_norm_eps"]) * silu(z)
    return y.reshape(t, val_dim) @ w(pre + "out_proj.weight").T


def mlp(w, pre, x):
    g = silu(x @ w(pre + "gate_proj.weight").T)
    u = x @ w(pre + "up_proj.weight").T
    return (g * u) @ w(pre + "down_proj.weight").T


class Model:
    def __init__(self, model=MODEL, verbose=False):
        self.cfg = config(model)
        self.w = Weights(model)
        self.verbose = verbose
        n = self.cfg["num_hidden_layers"]
        self.caches = [{"k": None, "v": None, "conv": None, "state": None} for _ in range(n)]
        self.pos = 0

    def is_linear(self, i):
        return (i + 1) % self.cfg["full_attention_interval"] != 0

    def forward(self, ids):
        """Logits for the last of `ids`, advancing the caches."""
        w, cfg = self.w, self.cfg
        emb = w("model.language_model.embed_tokens.weight")
        x = emb[np.asarray(ids)].astype(np.float32)
        del emb
        eps = cfg["rms_norm_eps"]
        t0 = time.time()
        for i in range(cfg["num_hidden_layers"]):
            pre = f"model.language_model.layers.{i}."
            h = rms_norm(x, w(pre + "input_layernorm.weight"), eps)
            if self.is_linear(i):
                r = gated_delta(w, pre + "linear_attn.", h, cfg, self.caches[i])
            else:
                r = attention(w, pre + "self_attn.", h, cfg, self.caches[i], self.pos)
            x = x + r
            h = rms_norm(x, w(pre + "post_attention_layernorm.weight"), eps)
            x = x + mlp(w, pre + "mlp.", h)
            if self.verbose and i % 16 == 0:
                print(f"    layer {i:2d}  {time.time() - t0:5.1f}s", file=sys.stderr)
        self.pos += len(ids)
        x = rms_norm(x[-1:], w("model.language_model.norm.weight"), eps)
        return (x @ w("lm_head.weight").T)[0]


def log_softmax(v):
    v = v.astype(np.float64)
    m = v.max()
    return v - m - np.log(np.exp(v - m).sum())


PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a small village,",
]


def write_golden(a, tk):
    """The fixture the Rust test reads, in `llama_ref.py`'s line format:

        model <tag>
        case <prompt ids...>
        step <next id> <id>:<logprob> ...   (top-k, best first)
    """
    out = [f"model {a.model}"]
    for prompt in PROMPTS:
        ids = tk.encode(prompt, add_special_tokens=False).ids
        m = Model(a.model)
        logits = m.forward(ids)
        out.append("case " + " ".join(map(str, ids)))
        print(f"{prompt!r}: {len(ids)} tokens", file=sys.stderr)
        for i in range(a.steps):
            lp = log_softmax(logits)
            top = np.argsort(lp)[::-1][: a.top]
            nxt = int(top[0])
            out.append(
                f"step {nxt} " + " ".join(f"{int(j)}:{lp[j]:.6f}" for j in top)
            )
            print(f"  {i:2d} {tk.decode([nxt])!r}", file=sys.stderr)
            logits = m.forward([nxt])
    path = pathlib.Path(a.write_golden)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--compare", action="store_true", help="check against Ollama's own log-probs")
    ap.add_argument(
        "--write-golden",
        nargs="?",
        const="crates/tile-rt/tests/data/qwen35_27b_golden.txt",
        help="write the fixture the Rust tests read (several prompts)",
    )
    ap.add_argument("--tol", type=float, default=0.15)
    a = ap.parse_args()

    tk = tokenizer(a.model)
    if a.write_golden:
        return write_golden(a, tk)
    ids = tk.encode(a.prompt, add_special_tokens=False).ids
    print(f"{a.model}: {a.prompt!r} -> {len(ids)} tokens {ids}")
    m = Model(a.model, verbose=True)
    ref = None
    if a.compare:
        from llama_ref import ollama

        ref = ollama(a.model, a.prompt, a.steps, a.top)

    t0 = time.time()
    logits = m.forward(ids)
    worst, ok = 0.0, True
    for i in range(a.steps):
        lp = log_softmax(logits)
        nxt = int(np.argmax(lp))
        line = f"  {i:2d}  {tk.decode([nxt])!r:20s} logprob {lp[nxt]:+.4f}"
        if ref is not None and i < len(ref):
            want = {bytes(alt["bytes"]): alt["logprob"] for alt in ref[i]["top_logprobs"]}
            by = {bytes(tk.decode([j]).encode("utf-8", "surrogatepass")): j for j in range(len(lp))}
            d = max((abs(lp[by[b]] - v) for b, v in want.items() if b in by), default=0.0)
            worst = max(worst, d)
            same = bytes(ref[i]["bytes"]) == tk.decode([nxt]).encode("utf-8", "surrogatepass")
            ok &= same and d <= a.tol
            line += f"   ollama {ref[i]['token']!r:20s} {'same' if same else 'DIFFERENT'}  max |dlogprob| {d:.4f}"
        print(line + f"   ({time.time() - t0:.0f}s)")
        logits = m.forward([nxt])
    if ref is not None:
        print(f"worst |dlogprob| {worst:.4f} (tolerance {a.tol})  {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
