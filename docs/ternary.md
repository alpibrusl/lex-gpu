# Prism ternary GGUF formats (PQ2_0, PTQ1_0)

Bonsai 2 27B ships as ternary-quantised GGUF in two storage types. Neither has a
public specification: the reference layouts live only in PrismML's llama.cpp
fork, which is MIT and therefore cannot be copied into this repository in either
direction. Everything below was derived from the published weight files
themselves — legitimate observation of data, not translation of source.

The weights are Apache-2.0 (Prism ML, from Qwen3.8-27B). We read them at
runtime; we do not redistribute them.

## How the layout was derived

`PQ2_0` is fully described in public prose: 128 weights, one trit per 2-bit
slot, one FP16 scale, 2.13 bpw. The file confirms it — the maximum payload byte
is `0b10101010`, so every 2-bit slot is in `{0,1,2}` and never `3`.

`PTQ1_0` had no such description. The crack is that both files quantise the
*same* model: for `output.weight` the FP16 scales are bit-identical for all
200000 blocks tested. That makes `PQ2_0` a ground-truth oracle, so the `PTQ1_0`
packing could be *measured* rather than guessed:

- mutual information between each payload byte and each of the 128 known trits
  is 1.585 bits (the full trit entropy) for exactly five trits and 0.008 bits
  for every other one, which yields the lane map with no ambiguity;
- the byte-to-digit code then falls out of the tabulated transfer function.

Two plausible guesses were falsified on the way, both worth recording because
they are what the prose suggests: the trits are *not* packed five-per-byte as a
plain base-3 integer (payload bytes reach 255, a base-3 quintet caps at 242),
and they are *not* in sequential or `stride 26` order.

## Common structure

Both types quantise 128 weights per block against one shared FP16 scale, and
both store trit codes `{0,1,2}` meaning `{-1, 0, +1}` — offset binary. The trit
histogram over 25.6M slots is 33.59 / 32.74 / 33.67 %, near-uniform and
symmetric about the zero code. That near-maximal entropy is what makes the dense
packing worth its decode cost: there is almost nothing for a cheaper code to
exploit.

A dequantised weight is `scale * (code - 1)`.

## PQ2_0 — 34 bytes per block, 2.125 bpw

```
offset  0 ..  2   FP16 scale
offset  2 .. 34   128 trit codes, 4 per byte, little end first
```

Trit `i` is at byte `2 + i/4`, bit position `2 * (i % 4)`.

## PTQ1_0 — 28 bytes per block, 1.75 bpw

```
offset  0 .. 26   128 trit codes, densely packed
offset 26 .. 28   FP16 scale
```

Note the scale is at the *end* here and at the start in PQ2_0.

The 26 payload bytes are three lane groups. Each byte carries several trits
drawn at the group's stride, so a SIMD lane reads one byte and produces a column
of trits:

| bytes   | trits    | stride | trits per byte |
|---------|----------|--------|----------------|
| 0 .. 16 | 0 .. 80  | 16     | 5              |
| 16 .. 24| 80 .. 120| 8      | 5              |
| 24 .. 26| 120 ..128| 2      | 4              |

That is 16*5 + 8*5 + 2*4 = 128 trits in 26 bytes.

Within a byte the trits are a base-3 code *scaled across the full byte range*
rather than the plain base-3 integer: `(0,0,0,0,0)` is 0, `(1,1,1,1,1)` is 128,
`(2,2,2,2,2)` is 255. Equivalently the stored byte is `ceil(256 * v / 243)` for
base-3 value `v`, with the first trit as the most significant digit. This is why
the payload bytes look like uniform random data and why every range-based test
of a base-3 hypothesis fails.

It decodes by the streaming carry trick — no division, no lookup table:

```
b = byte
repeat depth times:          // 5, or 4 for the tail group
    b    *= 3
    trit  = b >> 8
    b    &= 0xff
```

The first trit extracted is the one at the group's base index, and each
successive trit is one stride further along.

Verified by decoding 600000 blocks of `output.weight` and comparing all
76800000 trits against PQ2_0: zero mismatches.

## Activation rotation

The file documents its own transform in metadata, so this part needed no
reverse engineering:

```
prism.hadamard.transform   = normalized-sylvester-walsh-hadamard
prism.hadamard.block_size  = 1024
prism.hadamard.axis        = input-last-dimension
prism.hadamard.sign_mode   = explicit
prism.hadamard.sign_widths = [5120, 6144, 17408]
prism.hadamard.version     = 1
```

Sylvester-Walsh-Hadamard is `(-1)^popcount(i & j)`, which is what `Op::Butterfly`
computes one stage at a time: ten stages for block 1024, then a `1/sqrt(1024)`
scale for the `normalized` part. `sign_values` holds one explicit ±1 per input
element, 28672 of them, covering the three distinct input widths.
`token_embd.weight` takes the inverse transform, listed separately under
`prism.hadamard.inverse_weight_names`.
