//! Causal attention for a prefill chunk on the matrix units, for Metal.
//!
//! The typed program (`lex_front::flash::FlashDecode::build_causal_blocks`)
//! gives each (KV head, token) an instance that walks the cache a block at
//! a time with scalar arithmetic: 6 ms a layer for a 512-token chunk on an
//! M4 Max, 0.5 TFLOPS, and quadratic in the prompt -- a coding agent's
//! 10k-token prompt is where it would dominate.
//!
//! Here a threadgroup takes one KV head and a block of `BT` tokens -- their
//! `BT * group` query rows, a whole number of 8-row fragments -- and walks
//! the cache in blocks of 32 keys:
//!
//! - `S = Q Kᵀ` on simdgroup matrices, one 8-key column of it a simdgroup,
//!   `Q` staged once in threadgroup memory and `K` read as fragments
//!   straight from the cache, whose rows are contiguous;
//! - the online softmax, a thread a row: the running max and sum, and the
//!   factor the rows already accumulated are scaled by;
//! - `O = diag(alpha) O + P V`, each simdgroup owning a quarter of the
//!   head dimension, `V` read from the cache like `K`.
//!
//! Key blocks wholly past a tile's last token are never visited; the
//! program walks every block up to the chunk's end and masks them.
//!
//! **Hand-scheduled, like `crate::gemm`**: the IR has no matrix fragments.
//! It takes the program's parameters in the program's order -- `q`, the two
//! caches, `o`, then the runtime scalars `[pos0, nkb]` -- so a runtime binds
//! either one the same way. `P` is rounded to half for the `P V` product,
//! which the program keeps in f32: the difference is half's rounding of
//! values in [0, 1].

use crate::dialect::{Dialect, Msl};
use crate::program::Lowered;

/// Keys a block, and threads a threadgroup (four simdgroups).
const KB: usize = 32;
const THREADS: usize = 128;

/// One layer's attention shape for a chunk.
#[derive(Clone, Copy, Debug)]
pub struct Causal {
    pub tokens: usize,
    /// KV heads, and query heads each one serves.
    pub kv_heads: usize,
    pub group: usize,
    pub head_dim: usize,
    /// Cache rows a KV head holds.
    pub cap: usize,
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Tokens a threadgroup: the fewest whose query rows fill whole fragments.
fn block_tokens(group: usize) -> usize {
    8 / gcd(group, 8)
}

pub fn fits(c: &Causal) -> bool {
    let rows = block_tokens(c.group) * c.group;
    c.tokens > 0
        && c.group > 0
        && rows <= 32
        && c.head_dim.is_multiple_of(4 * 8)
        && c.cap.is_multiple_of(KB)
}

pub fn causal_mma(c: &Causal) -> Result<Lowered, String> {
    if !fits(c) {
        return Err(format!(
            "attn: group {} (rows a block must fit 32), head dim {} (whole fragments \
             for four simdgroups), cache {} (whole blocks of {KB})",
            c.group, c.head_dim, c.cap
        ));
    }
    let bt = block_tokens(c.group);
    let r = bt * c.group;
    let nrf = r / 8;
    let ds = c.head_dim / 4;
    let qld = c.head_dim + 8;
    let entry = format!(
        "attn_mma_t{}_h{}x{}_d{}_cap{}",
        c.tokens, c.kv_heads, c.group, c.head_dim, c.cap
    );
    let source = SOURCE
        .replace("$ENTRY", &entry)
        .replace("$INCLUDES", &Msl.includes())
        .replace("$TOKENS", &c.tokens.to_string())
        .replace("$GROUP", &c.group.to_string())
        .replace("$HG", &(c.kv_heads * c.group).to_string())
        .replace("$HD", &c.head_dim.to_string())
        .replace("$CAP", &c.cap.to_string())
        .replace("$BT", &bt.to_string())
        .replace("$NRF", &nrf.to_string())
        .replace("$DSF", &(ds / 8).to_string())
        .replace("$DS", &ds.to_string())
        .replace("$QLD", &qld.to_string())
        .replace("$SCALE", &format!("{:?}", 1.0 / (c.head_dim as f32).sqrt()))
        .replace("$KB", &KB.to_string())
        .replace("$R", &r.to_string())
        .replace("$T", &THREADS.to_string());
    Ok(Lowered {
        entry,
        source,
        grid: c.tokens.div_ceil(bt),
        grid2: c.kv_heads,
        threads: THREADS,
        threadgroup_bytes: 2 * r * qld + 4 * r * KB + 2 * r * KB + 4 * 3 * r + 4 * nrf * 64,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 5,
        // q, k cache, v cache, o, scalars
        writes: vec![false, false, false, true, false],
    })
}

/// Whether `c` fits the split kernel: every token in one tile of rows.
pub fn fits_split(c: &Causal, span: usize) -> bool {
    fits(c)
        && c.tokens <= block_tokens(c.group)
        && span > 0
        && span.is_multiple_of(KB)
        && c.cap.is_multiple_of(span)
}

/// [`causal_mma`] cut across the cache, for a speculative verify: a
/// threadgroup takes one KV head and one `span` of positions, for every
/// token of the batch at once, and writes its partial softmax state --
/// running max, sum and unnormalised accumulator -- in the layout
/// `lex_front::flash::FlashDecode::build_causal_split` writes, so the same
/// combine merges it. The program it replaces carries a separate
/// accumulator per token through scalar arithmetic, and its cost grew with
/// the tokens: at 8000 positions a verify of four cost 1.60 decode steps
/// against 1.25 at 300, nearly all of the difference attention.
///
/// Parameters as that program's: `q`, the two caches, `part_m`, `part_l`,
/// `part_acc`, then the scalars `[pos0]`. Grid: KV heads by splits, of
/// which a runtime launches only the live ones.
pub fn causal_mma_split(c: &Causal, span: usize) -> Result<Lowered, String> {
    if !fits_split(c, span) {
        return Err(format!(
            "attn split: {} tokens (at most {} for a group of {}), span {span} (whole blocks \
             of {KB} dividing the cache of {})",
            c.tokens,
            block_tokens(c.group),
            c.group,
            c.cap
        ));
    }
    let bt = block_tokens(c.group);
    let r = bt * c.group;
    let nrf = r / 8;
    let ds = c.head_dim / 4;
    let qld = c.head_dim + 8;
    let splits = c.cap / span;
    let entry = format!(
        "attn_mma_split_t{}_h{}x{}_d{}_cap{}_s{span}",
        c.tokens, c.kv_heads, c.group, c.head_dim, c.cap
    );
    let source = SPLIT_SOURCE
        .replace("$ENTRY", &entry)
        .replace("$INCLUDES", &Msl.includes())
        .replace("$TOKENS", &c.tokens.to_string())
        .replace("$GROUP", &c.group.to_string())
        .replace("$HG", &(c.kv_heads * c.group).to_string())
        .replace("$HD", &c.head_dim.to_string())
        .replace("$CAP", &c.cap.to_string())
        .replace("$SPLITS", &splits.to_string())
        .replace("$SPAN", &span.to_string())
        .replace("$NRF", &nrf.to_string())
        .replace("$DSF", &(ds / 8).to_string())
        .replace("$DS", &ds.to_string())
        .replace("$QLD", &qld.to_string())
        .replace("$SCALE", &format!("{:?}", 1.0 / (c.head_dim as f32).sqrt()))
        .replace("$KB", &KB.to_string())
        .replace("$R", &r.to_string())
        .replace("$T", &THREADS.to_string());
    Ok(Lowered {
        entry,
        source,
        grid: c.kv_heads,
        grid2: splits,
        threads: THREADS,
        threadgroup_bytes: 2 * r * qld + 4 * r * KB + 2 * r * KB + 4 * 3 * r + 4 * nrf * 64,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 5,
        // q, k cache, v cache, part_m, part_l, part_acc, scalars
        writes: vec![false, false, false, true, true, true, false],
    })
}

const SPLIT_SOURCE: &str = r#"// Generated by lex-msl::attn. Hand-scheduled, not lowered from a program.
// kernel : $ENTRY
// launch : kv heads x live splits threadgroups x $T threads
$INCLUDES#include <metal_simdgroup_matrix>

kernel void $ENTRY(
    device const half *q [[buffer(0)]],
    device const half *kc [[buffer(1)]],
    device const half *vc [[buffer(2)]],
    device float *pm [[buffer(3)]],
    device float *pl [[buffer(4)]],
    device float *pa [[buffer(5)]],
    device const uint *sc [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint T = $TOKENSu, G = $GROUPu, HG = $HGu, D = $HDu, CAP = $CAPu, S = $SPLITSu;
    const uint h = tg.x, split = tg.y, pos0 = sc[0];
    threadgroup half Qs[$R * $QLD];
    threadgroup float Ss[$R * $KB];
    threadgroup half Ps[$R * $KB];
    threadgroup float Ms[$R], Ls[$R], Al[$R];
    threadgroup float Dg[$NRF * 64];

    // Row r is token r / G, query head h * G + r % G: every token at once.
    for (uint e = tid; e < $Ru * D / 4u; e += $Tu) {
        const uint r = e / (D / 4u), c = (e % (D / 4u)) * 4u;
        const uint tok = r / G;
        half4 v = half4(0.0h);
        if (tok < T) v = *(device const half4 *)(q + ((size_t)tok * HG + h * G + r % G) * D + c);
        *(threadgroup half4 *)(Qs + r * $QLDu + c) = v;
    }
    if (tid < $Ru) {
        Ms[tid] = -INFINITY;
        Ls[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // This split's blocks, up to the last token's position.
    const uint first = split * $SPANu, qlast = pos0 + T - 1u;
    const uint end = min(first + $SPANu, qlast + 1u);
    const uint nblk = end > first ? (end - first + $KBu - 1u) / $KBu : 0u;
    const uint dsl = sgid * $DSu;
    const device half *kb = kc + (size_t)h * CAP * D;
    const device half *vb = vc + (size_t)h * CAP * D;
    simdgroup_matrix<float, 8, 8> O[$NRF][$DSF];
    for (uint i = 0; i < $NRFu; ++i)
        for (uint j = 0; j < $DSFu; ++j) O[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint b = 0; b < nblk; ++b) {
        const uint key0 = first + b * $KBu;
        {
            simdgroup_matrix<float, 8, 8> s[$NRF];
            for (uint i = 0; i < $NRFu; ++i) s[i] = simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kk = 0; kk < D; kk += 8u) {
                simdgroup_matrix<half, 8, 8> kt;
                simdgroup_load(kt, kb + (size_t)(key0 + sgid * 8u) * D + kk, D, ulong2(0, 0), true);
                for (uint i = 0; i < $NRFu; ++i) {
                    simdgroup_matrix<half, 8, 8> qa;
                    simdgroup_load(qa, Qs + (i * 8u) * $QLDu + kk, $QLDu);
                    simdgroup_multiply_accumulate(s[i], qa, kt, s[i]);
                }
            }
            for (uint i = 0; i < $NRFu; ++i)
                simdgroup_store(s[i], Ss + (i * 8u) * $KBu + sgid * 8u, $KBu);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < $Ru) {
            const uint r = tid, tok = r / G, qpos = pos0 + tok;
            const float mold = Ms[r];
            float v[$KB];
            float mx = mold;
            for (uint j = 0; j < $KBu; ++j) {
                const bool live = tok < T && key0 + j <= qpos;
                v[j] = live ? Ss[r * $KBu + j] * $SCALEf : -INFINITY;
                mx = max(mx, v[j]);
            }
            float alpha = 1.0f, sum = 0.0f;
            if (mx == -INFINITY) {
                for (uint j = 0; j < $KBu; ++j) Ps[r * $KBu + j] = 0.0h;
            } else {
                alpha = mold == -INFINITY ? 0.0f : exp(mold - mx);
                for (uint j = 0; j < $KBu; ++j) {
                    const float p = exp(v[j] - mx);
                    Ps[r * $KBu + j] = half(p);
                    sum += p;
                }
                Ms[r] = mx;
            }
            Ls[r] = Ls[r] * alpha + sum;
            Al[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < $NRFu * 64u; e += $Tu) {
            const uint i = e / 64u, rr = (e % 64u) / 8u, cc = e % 8u;
            Dg[e] = rr == cc ? Al[i * 8u + rr] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < $NRFu; ++i) {
            simdgroup_matrix<float, 8, 8> dg;
            simdgroup_load(dg, Dg + i * 64u, 8);
            for (uint j = 0; j < $DSFu; ++j) simdgroup_multiply(O[i][j], dg, O[i][j]);
        }
        for (uint kf = 0; kf < $KBu / 8u; ++kf) {
            simdgroup_matrix<half, 8, 8> p[$NRF];
            for (uint i = 0; i < $NRFu; ++i) simdgroup_load(p[i], Ps + (i * 8u) * $KBu + kf * 8u, $KBu);
            for (uint j = 0; j < $DSFu; ++j) {
                simdgroup_matrix<half, 8, 8> vv;
                simdgroup_load(vv, vb + (size_t)(key0 + kf * 8u) * D + dsl + j * 8u, D);
                for (uint i = 0; i < $NRFu; ++i) simdgroup_multiply_accumulate(O[i][j], p[i], vv, O[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The partial state, unnormalised, row `tok * HG + h * G + r % G`,
    // column `split`. A split that saw nothing writes a large finite
    // negative max and zero sum, as the program does, so it weighs nothing
    // in the combine rather than making a NaN.
    if (tid < $Ru) {
        const uint r = tid, tok = r / G;
        if (tok < T) {
            const size_t row = (size_t)tok * HG + h * G + r % G;
            pm[row * S + split] = max(Ms[r], -1e30f);
            pl[row * S + split] = Ls[r];
        }
    }
    const uint fr = (lane % 8u) / 2u + 4u * (lane / 16u);
    const uint fc = 2u * (lane % 2u) + 4u * ((lane / 8u) % 2u);
    for (uint i = 0; i < $NRFu; ++i) {
        const uint r = i * 8u + fr, tok = r / G;
        for (uint j = 0; j < $DSFu; ++j) {
            const auto e = O[i][j].thread_elements();
            if (tok < T) {
                const size_t row = (size_t)tok * HG + h * G + r % G;
                const size_t at = (row * S + split) * D + dsl + j * 8u + fc;
                pa[at] = e[0];
                pa[at + 1u] = e[1];
            }
        }
    }
}
"#;

const SOURCE: &str = r#"// Generated by lex-msl::attn. Hand-scheduled, not lowered from a program.
// kernel : $ENTRY
// launch : ($TOKENS / $BT) x kv heads threadgroups x $T threads
$INCLUDES#include <metal_simdgroup_matrix>

kernel void $ENTRY(
    device const half *q [[buffer(0)]],
    device const half *kc [[buffer(1)]],
    device const half *vc [[buffer(2)]],
    device float *o [[buffer(3)]],
    device const uint *sc [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint T = $TOKENSu, G = $GROUPu, HG = $HGu, D = $HDu, CAP = $CAPu;
    const uint h = tg.y, t0 = tg.x * $BTu, pos0 = sc[0];
    threadgroup half Qs[$R * $QLD];     // the tile's query rows
    threadgroup float Ss[$R * $KB];     // S for one key block
    threadgroup half Ps[$R * $KB];      // exp(S - max), for P V
    threadgroup float Ms[$R], Ls[$R], Al[$R];
    threadgroup float Dg[$NRF * 64];    // diagonal 8x8s, to scale rows

    // Row r is token t0 + r / G, query head h * G + r % G.
    for (uint e = tid; e < $Ru * D / 4u; e += $Tu) {
        const uint r = e / (D / 4u), c = (e % (D / 4u)) * 4u;
        const uint tok = t0 + r / G;
        half4 v = half4(0.0h);
        if (tok < T) v = *(device const half4 *)(q + ((size_t)tok * HG + h * G + r % G) * D + c);
        *(threadgroup half4 *)(Qs + r * $QLDu + c) = v;
    }
    if (tid < $Ru) {
        Ms[tid] = -INFINITY;
        Ls[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Blocks up to the tile's last token; the rest are all masked.
    const uint last = min(t0 + $BTu, T) - 1u;
    const uint nblk = (pos0 + last) / $KBu + 1u;
    const uint dsl = sgid * $DSu;
    const device half *kb = kc + (size_t)h * CAP * D;
    const device half *vb = vc + (size_t)h * CAP * D;
    simdgroup_matrix<float, 8, 8> O[$NRF][$DSF];
    for (uint i = 0; i < $NRFu; ++i)
        for (uint j = 0; j < $DSFu; ++j) O[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint b = 0; b < nblk; ++b) {
        const uint key0 = b * $KBu;
        // S's 8-key column `sgid`: Q against those keys, over the head.
        {
            simdgroup_matrix<float, 8, 8> s[$NRF];
            for (uint i = 0; i < $NRFu; ++i) s[i] = simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kk = 0; kk < D; kk += 8u) {
                simdgroup_matrix<half, 8, 8> kt;
                simdgroup_load(kt, kb + (size_t)(key0 + sgid * 8u) * D + kk, D, ulong2(0, 0), true);
                for (uint i = 0; i < $NRFu; ++i) {
                    simdgroup_matrix<half, 8, 8> qa;
                    simdgroup_load(qa, Qs + (i * 8u) * $QLDu + kk, $QLDu);
                    simdgroup_multiply_accumulate(s[i], qa, kt, s[i]);
                }
            }
            for (uint i = 0; i < $NRFu; ++i)
                simdgroup_store(s[i], Ss + (i * 8u) * $KBu + sgid * 8u, $KBu);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The online softmax, a thread a row.
        if (tid < $Ru) {
            const uint r = tid, tok = t0 + r / G, qpos = pos0 + tok;
            const float mold = Ms[r];
            float v[$KB];
            float mx = mold;
            for (uint j = 0; j < $KBu; ++j) {
                const bool live = tok < T && key0 + j <= qpos;
                v[j] = live ? Ss[r * $KBu + j] * $SCALEf : -INFINITY;
                mx = max(mx, v[j]);
            }
            float alpha = 1.0f, sum = 0.0f;
            if (mx == -INFINITY) {
                for (uint j = 0; j < $KBu; ++j) Ps[r * $KBu + j] = 0.0h;
            } else {
                alpha = mold == -INFINITY ? 0.0f : exp(mold - mx);
                for (uint j = 0; j < $KBu; ++j) {
                    const float p = exp(v[j] - mx);
                    Ps[r * $KBu + j] = half(p);
                    sum += p;
                }
                Ms[r] = mx;
            }
            Ls[r] = Ls[r] * alpha + sum;
            Al[r] = alpha;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < $NRFu * 64u; e += $Tu) {
            const uint i = e / 64u, rr = (e % 64u) / 8u, cc = e % 8u;
            Dg[e] = rr == cc ? Al[i * 8u + rr] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(alpha) O + P V, over this simdgroup's quarter of the head.
        for (uint i = 0; i < $NRFu; ++i) {
            simdgroup_matrix<float, 8, 8> dg;
            simdgroup_load(dg, Dg + i * 64u, 8);
            for (uint j = 0; j < $DSFu; ++j) simdgroup_multiply(O[i][j], dg, O[i][j]);
        }
        for (uint kf = 0; kf < $KBu / 8u; ++kf) {
            simdgroup_matrix<half, 8, 8> p[$NRF];
            for (uint i = 0; i < $NRFu; ++i) simdgroup_load(p[i], Ps + (i * 8u) * $KBu + kf * 8u, $KBu);
            for (uint j = 0; j < $DSFu; ++j) {
                simdgroup_matrix<half, 8, 8> vv;
                simdgroup_load(vv, vb + (size_t)(key0 + kf * 8u) * D + dsl + j * 8u, D);
                for (uint i = 0; i < $NRFu; ++i) simdgroup_multiply_accumulate(O[i][j], p[i], vv, O[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // O / l, and each lane's two elements of every fragment out (row fr,
    // columns fc and fc + 1 -- the layout measured on an M4 Max).
    for (uint e = tid; e < $NRFu * 64u; e += $Tu) {
        const uint i = e / 64u, rr = (e % 64u) / 8u, cc = e % 8u;
        const float l = Ls[i * 8u + rr];
        Dg[e] = rr == cc ? (l > 0.0f ? 1.0f / l : 0.0f) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint fr = (lane % 8u) / 2u + 4u * (lane / 16u);
    const uint fc = 2u * (lane % 2u) + 4u * ((lane / 8u) % 2u);
    for (uint i = 0; i < $NRFu; ++i) {
        simdgroup_matrix<float, 8, 8> dg;
        simdgroup_load(dg, Dg + i * 64u, 8);
        const uint r = i * 8u + fr, tok = t0 + r / G;
        for (uint j = 0; j < $DSFu; ++j) {
            simdgroup_multiply(O[i][j], dg, O[i][j]);
            const auto e = O[i][j].thread_elements();
            if (tok < T) {
                const size_t at = ((size_t)tok * HG + h * G + r % G) * D + dsl + j * 8u + fc;
                o[at] = e[0];
                o[at + 1u] = e[1];
            }
        }
    }
}
"#;
// ---------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------

/// Keys a block on CUDA: four warps, each the 16 keys of one fragment.
const CKB: usize = 64;
/// Query rows a block holds at most: with `CKB`, what 48 KB of static shared
/// memory takes (Q 25 KB at a 256-wide head, S 12 KB, P 6 KB).
const CMAX_ROWS: usize = 48;

/// Tokens a block: the fewest whose query rows fill whole 16-row fragments.
fn cuda_block_tokens(group: usize) -> usize {
    16 / gcd(group, 16)
}

/// Whether [`causal_wmma`] takes this shape.
pub fn fits_cuda(c: &Causal) -> bool {
    c.group > 0 && c.tokens > 0 && {
        let rows = cuda_block_tokens(c.group) * c.group;
        rows <= CMAX_ROWS
            && c.head_dim.is_multiple_of(4 * 16)
            && c.head_dim <= 256
            && c.cap.is_multiple_of(CKB)
    }
}

/// [`causal_mma`] on `wmma` 16x16x16: the same walk -- a block takes one KV
/// head and the query rows of a few tokens, and visits the cache 64 keys at
/// a time -- with `S = Q Kᵀ` one 16-key column a warp, the online softmax a
/// thread a row, and `O = diag(alpha) O + P V` a warp's quarter of the head
/// held in accumulator fragments across the whole walk.
///
/// The fragment layout is what lets `O` stay in registers: an f32 `wmma`
/// accumulator hands thread `t` of the warp, for element `e`, the row
/// `t / 4 + 8 * ((e / 2) % 2)` and the column `2 * (t % 4) + e % 2 + 8 *
/// (e / 4)` (PTX ISA, "matrix fragments for wmma"). The `alpha` rescale and
/// the final `1 / l` use it, so a wrong layout would be silent; the device
/// test compares a whole kernel against a host softmax.
///
/// Same parameters, grid and scalars as [`causal_mma`], and K and V are f16.
pub fn causal_wmma(c: &Causal) -> Result<Lowered, String> {
    if !fits_cuda(c) {
        return Err(format!(
            "attn: group {} (rows a block must fit {CMAX_ROWS}), head dim {} (a multiple \
             of 64, at most 256), cache {} (whole blocks of {CKB})",
            c.group, c.head_dim, c.cap
        ));
    }
    let bt = cuda_block_tokens(c.group);
    let r = bt * c.group;
    let nrf = r / 16;
    let ds = c.head_dim / 4;
    let qld = c.head_dim + 8;
    let entry = format!(
        "attn_wmma_t{}_h{}x{}_d{}_cap{}",
        c.tokens, c.kv_heads, c.group, c.head_dim, c.cap
    );
    let source = CUDA_SOURCE
        .replace("$ENTRY", &entry)
        .replace("$INCLUDES", &crate::dialect::Cuda.includes())
        .replace("$TOKENS", &c.tokens.to_string())
        .replace("$GROUP", &c.group.to_string())
        .replace("$HG", &(c.kv_heads * c.group).to_string())
        .replace("$HD", &c.head_dim.to_string())
        .replace("$BT", &bt.to_string())
        .replace("$NRF", &nrf.to_string())
        .replace("$DSF", &(ds / 16).to_string())
        .replace("$DS", &ds.to_string())
        .replace("$QLD", &qld.to_string())
        .replace("$CAP", &c.cap.to_string())
        .replace("$SCALE", &format!("{:?}", 1.0 / (c.head_dim as f32).sqrt()))
        .replace("$KB", &CKB.to_string())
        .replace("$R", &r.to_string());
    Ok(Lowered {
        entry,
        source,
        grid: c.tokens.div_ceil(bt),
        grid2: c.kv_heads,
        threads: 128,
        threadgroup_bytes: 2 * r * qld + 4 * r * CKB + 2 * r * CKB + 4 * 3 * r,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 4,
        // q, k cache, v cache, o, scalars
        writes: vec![false, false, false, true, false],
    })
}

const CUDA_SOURCE: &str = r#"// Generated by lex-msl::attn. Hand-scheduled, not lowered from a program.
// kernel : $ENTRY
// launch : ($TOKENS / $BT) x kv heads blocks x 128 threads
$INCLUDES#include <mma.h>

extern "C" __global__ void __launch_bounds__(128) $ENTRY(
    const __half* __restrict__ q,
    const __half* __restrict__ kc,
    const __half* __restrict__ vc,
    float* __restrict__ o,
    const unsigned int* __restrict__ sc)
{
    using namespace nvcuda;
    const uint T = $TOKENSu, G = $GROUPu, HG = $HGu, D = $HDu;
    const uint h = blockIdx.y, t0 = blockIdx.x * $BTu, pos0 = sc[0];
    const uint tid = threadIdx.x, warp = tid / 32u, lane = tid % 32u;
    __shared__ __align__(32) __half Qs[$R][$QLD];   // the block's query rows
    __shared__ __align__(32) float Ss[$R][$KB];     // S for one key block
    __shared__ __align__(32) __half Ps[$R][$KB];    // exp(S - max), for P V
    __shared__ float Ms[$R], Ls[$R], Al[$R];

    // Row r is token t0 + r / G, query head h * G + r % G.
    for (uint e = tid; e < $Ru * D / 8u; e += 128u) {
        const uint r = e / (D / 8u), c = (e % (D / 8u)) * 8u;
        const uint tok = t0 + r / G;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (tok < T) v = *reinterpret_cast<const uint4*>(q + ((size_t)tok * HG + h * G + r % G) * D + c);
        *reinterpret_cast<uint4*>(&Qs[r][c]) = v;
    }
    if (tid < $Ru) {
        Ms[tid] = -INFINITY;
        Ls[tid] = 0.0f;
    }
    __syncthreads();

    // Blocks up to the block's last token; the rest are all masked.
    const uint last = min(t0 + $BTu, T) - 1u;
    const uint nblk = (pos0 + last) / $KBu + 1u;
    const uint dsl = warp * $DSu;
    const __half* kh = kc + (size_t)h * $CAPu * D;
    const __half* vh = vc + (size_t)h * $CAPu * D;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> O[$NRF][$DSF];
    for (int i = 0; i < $NRF; ++i)
        for (int j = 0; j < $DSF; ++j) wmma::fill_fragment(O[i][j], 0.0f);

    for (uint b = 0; b < nblk; ++b) {
        const uint key0 = b * $KBu;
        // S's 16-key column `warp`: Q against those keys, over the head.
        {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[$NRF];
            for (int i = 0; i < $NRF; ++i) wmma::fill_fragment(s[i], 0.0f);
            for (uint kk = 0; kk < D; kk += 16u) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kt;
                wmma::load_matrix_sync(kt, kh + (size_t)(key0 + warp * 16u) * D + kk, D);
                for (int i = 0; i < $NRF; ++i) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qa;
                    wmma::load_matrix_sync(qa, &Qs[i * 16][kk], $QLD);
                    wmma::mma_sync(s[i], qa, kt, s[i]);
                }
            }
            for (int i = 0; i < $NRF; ++i)
                wmma::store_matrix_sync(&Ss[i * 16][warp * 16u], s[i], $KB, wmma::mem_row_major);
        }
        __syncthreads();

        // The online softmax, a thread a row.
        if (tid < $Ru) {
            const uint r = tid, tok = t0 + r / G, qpos = pos0 + tok;
            const float mold = Ms[r];
            float v[$KB];
            float mx = mold;
            for (uint j = 0; j < $KBu; ++j) {
                const bool live = tok < T && key0 + j <= qpos;
                v[j] = live ? Ss[r][j] * $SCALEf : -INFINITY;
                mx = fmaxf(mx, v[j]);
            }
            float alpha = 1.0f, sum = 0.0f;
            if (mx == -INFINITY) {
                for (uint j = 0; j < $KBu; ++j) Ps[r][j] = __float2half(0.0f);
            } else {
                alpha = mold == -INFINITY ? 0.0f : __expf(mold - mx);
                for (uint j = 0; j < $KBu; ++j) {
                    const float p = __expf(v[j] - mx);
                    Ps[r][j] = __float2half(p);
                    sum += p;
                }
                Ms[r] = mx;
            }
            Ls[r] = Ls[r] * alpha + sum;
            Al[r] = alpha;
        }
        __syncthreads();

        // O = diag(alpha) O + P V, over this warp's quarter of the head.
        // Accumulator element e of lane t is row t / 4 + 8 * ((e / 2) % 2).
        for (int i = 0; i < $NRF; ++i) {
            const float a0 = Al[i * 16 + (lane >> 2)], a1 = Al[i * 16 + (lane >> 2) + 8];
            for (int j = 0; j < $DSF; ++j)
                for (int e = 0; e < 8; ++e) O[i][j].x[e] *= ((e >> 1) & 1) ? a1 : a0;
        }
        for (uint kf = 0; kf < $KBu / 16u; ++kf) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p[$NRF];
            for (int i = 0; i < $NRF; ++i) wmma::load_matrix_sync(p[i], &Ps[i * 16][kf * 16u], $KB);
            for (int j = 0; j < $DSF; ++j) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vv;
                wmma::load_matrix_sync(vv, vh + (size_t)(key0 + kf * 16u) * D + dsl + j * 16u, D);
                for (int i = 0; i < $NRF; ++i) wmma::mma_sync(O[i][j], p[i], vv, O[i][j]);
            }
        }
        __syncthreads();
    }

    // O / l straight out: lane t holds, for element e, row t / 4 + 8 *
    // ((e / 2) % 2) of the fragment and column 2 * (t % 4) + e % 2 + 8 * (e / 4).
    for (int i = 0; i < $NRF; ++i) {
        for (int hf = 0; hf < 2; ++hf) {
            const uint r = i * 16 + (lane >> 2) + 8 * hf, tok = t0 + r / G;
            const float l = Ls[r];
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            if (tok >= T) continue;
            float* row = o + ((size_t)tok * HG + h * G + r % G) * D + dsl;
            for (int j = 0; j < $DSF; ++j)
                for (int half = 0; half < 2; ++half) {
                    const int e = 4 * half + 2 * hf;
                    const uint col = j * 16u + 2u * (lane & 3u) + 8u * half;
                    *reinterpret_cast<float2*>(row + col) = make_float2(O[i][j].x[e] * inv, O[i][j].x[e + 1] * inv);
                }
        }
    }
}
"#;
