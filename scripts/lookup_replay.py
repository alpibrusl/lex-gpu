#!/usr/bin/env python3
"""What drafting from the context would have got on real requests -- offline,
from the server's request log, before any of it is built (lex-gpu#27).

    LEX_REQUEST_LOG=requests.jsonl target/release/examples/serve ...
    python3 scripts/lookup_replay.py requests.jsonl

A speculative cycle feeds the token just chosen plus `k` drafts, and commits
the drafts up to the first wrong one plus one token from the verify. Here
the drafts come from the context: the last `n` tokens of everything so far
(prompt and reply) are looked up earlier in it, most recent occurrence
first, and the `k` tokens that followed are proposed. The reply the model
actually wrote says how many of them it would have accepted -- exactly, for
greedy decoding, since a verify accepts a draft iff it is the token the
model picks.

Positions with no match fall back to the draft head, which is not
replayed here: it is charged its rate as measured through the server
(`--head-tokens` per `--head-ms`) a token at a time. Costs of a verify of t rows are the measured ones on an M4 Max
(`examples/verify`) at the round's context length -- the first version took
them at 300 positions and over-promised on 2000-token prompts, where a
verify costs more.
"""
import argparse, json, statistics

# ms for a verify of t rows at a context of n positions, M4 Max, after the
# split matrix-unit verify attention (examples/verify); interpolated
# linearly in n. Past four rows a verify leaves the few-token matvec and
# the matrix-unit attention, which is the jump from 4 to 5.
VERIFY_MS = {  # context: [t = 1 .. 8]
    300: [36.2, 37.1, 37.7, 42.7, 58.1, 66.1, 74.1, 83.1],
    2300: [37.2, 39.2, 40.6, 45.3, 65.0, 74.3, 83.7, 95.2],
    8000: [44.8, 46.6, 48.2, 51.9],  # 5-8 not measured here: 2300's slope
}


def verify_ms(t, n):
    pts = sorted(VERIFY_MS)
    n = min(max(n, pts[0]), pts[-1])
    lo = max(p for p in pts if p <= n)
    hi = min(p for p in pts if p >= n)
    f = 0.0 if hi == lo else (n - lo) / (hi - lo)

    def row(p):
        v = VERIFY_MS[p]
        if t <= len(v):
            return v[t - 1]
        return v[-1] + VERIFY_MS[2300][t - 1] - VERIFY_MS[2300][len(v) - 1]

    return row(lo) + f * (row(hi) - row(lo))


def lookup(hist, n_max, n_min, k):
    """Up to k tokens that followed the most recent earlier occurrence of
    the longest suffix of hist (n_max down to n_min tokens); and that n."""
    L = len(hist)
    for n in range(n_max, n_min - 1, -1):
        if L <= n:
            continue
        pat = hist[L - n:]
        # Most recent occurrence ending before the suffix itself.
        for s in range(L - n - 1, -1, -1):
            if hist[s:s + n] == pat:
                return hist[s + n:s + n + k], n
    return [], 0


def replay(prompt, reply, n_max, n_min, k, head_tokens, head_ms, min_accept_n):
    """Simulate the cycles over one reply. Returns (ms, tokens, stats)."""
    ms, i, stats = 0.0, 0, {"lookup_cycles": 0, "lookup_tokens": 0, "head_cycles": 0,
                            "by_n": {}}
    hist = list(prompt) + list(reply[:1])
    # reply[0] comes from the prefill's logits; every cycle after it starts
    # with reply[i] chosen and not yet fed.
    while i < len(reply) - 1:
        drafts, n = lookup(hist, n_max, n_min, k)
        if drafts and n >= min_accept_n:
            want = reply[i + 1:i + 1 + len(drafts)]
            acc = 0
            while acc < len(drafts) and acc < len(want) and drafts[acc] == want[acc]:
                acc += 1
            adv = min(acc + 1, len(reply) - 1 - i)
            ms += verify_ms(len(drafts) + 1, len(hist))
            stats["lookup_cycles"] += 1
            stats["lookup_tokens"] += adv
            b = stats["by_n"].setdefault(n, [0, 0, 0])
            b[0] += 1
            b[1] += acc
            b[2] += len(drafts)
        else:
            # The head's measured rate, charged a token at a time, so the
            # lookup is tried again at the very next position.
            adv = 1
            ms += head_ms / head_tokens
            stats["head_cycles"] += 1
        hist.extend(reply[i + 1:i + 1 + adv])
        i += adv
    return ms, len(reply) - 1, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--head-tokens", type=float, default=2.49,
                    help="tokens a head round commits, as traced through the server")
    ap.add_argument("--head-ms", type=float, default=50.7, help="ms a head round takes")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.log) if l.strip()]
    print(f"{len(rows)} requests; head alone: {1000 * a.head_tokens / a.head_ms:.1f} tok/s")
    for k in (3, 5, 7):
        for n_min in (2, 3, 4):
            line = []
            for r in rows:
                ms, n, st = replay(r["prompt"], r["completion"], 6, n_min, k,
                                   a.head_tokens, a.head_ms, n_min)
                lc = st["lookup_cycles"]
                share = st["lookup_tokens"] / max(n, 1)
                per = st["lookup_tokens"] / lc if lc else 0
                line.append(f"{1000 * n / ms:6.1f} tok/s ({share:4.0%} of tokens by lookup, "
                            f"{per:.2f}/cycle)")
            tag = f"k={k} n>={n_min}"
            print(tag)
            for r, l in zip(rows, line):
                print(f"   {len(r['prompt']):5} + {len(r['completion']):5} tokens: {l}")


if __name__ == "__main__":
    main()
