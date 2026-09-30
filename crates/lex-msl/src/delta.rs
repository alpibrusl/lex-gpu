//! The gated delta rule over a prefill chunk, in chunks of tokens, for Metal.
//!
//! The step kernel (`lex_front::qwen::DeltaNet::build_steps`) carries a
//! tile of the state through the batch one token at a time, and each token
//! costs two reductions across the threadgroup: 1.5 ms a layer for 128
//! tokens on an M4 Max, where the arithmetic is a few microseconds. That
//! latency is the whole cost, and it is the recurrence's.
//!
//! Within a chunk of `C` tokens the recurrence can be solved in closed
//! form. With `S` the state (rows are value dimensions, columns key
//! dimensions), per token `j`
//!
//! ```text
//! S_j = g_j S_{j-1} + u_j k_jᵀ,   u_j = β_j (v_j − g_j S_{j-1} k_j),   o_j = S_j q_j
//! ```
//!
//! and with `Γ_j` the chunk's running sum of `log g`, unrolling from the
//! state `S_0` the chunk starts from gives
//!
//! ```text
//! u_j = β_j (v_j − e^{Γ_j} S_0 k_j) − Σ_{l<j} β_j e^{Γ_j−Γ_l} (k_j·k_l) u_l
//! o_i = e^{Γ_i} S_0 q_i + Σ_{l≤i} e^{Γ_i−Γ_l} (q_i·k_l) u_l
//! S_C = e^{Γ_C} S_0 + Σ_l e^{Γ_C−Γ_l} u_l k_lᵀ
//! ```
//!
//! The first line is a triangular system for the chunk's `u`, solved by
//! substitution; everything else is small matrix products, which go on the
//! matrix units. Only the substitution is sequential, and it is `C` steps
//! of a few multiply-adds rather than `C` barriered reductions.
//!
//! Rows of the state never mix -- `g` and `β` are one value per head, and
//! `k`, `q` are shared -- so, like the step kernel, a threadgroup owns a
//! block of `R` rows of one head and the grid is (row blocks, heads).
//!
//! **Hand-scheduled, like `crate::gemm`**: the IR has no triangular solve
//! and no matrix fragments. It takes exactly the parameters the step
//! kernel does, in the same order, so a runtime binds either the same way.
//! f32 throughout, the matrix products included: the recurrence carries
//! its error from token to token and layer to layer.
//!
//! One source per backend, same plan. Metal does the products on
//! simdgroup_matrix in f32. CUDA does them on the ordinary cores: its
//! matrix units take f32 only as TF32, a 10-bit mantissa, and at 16 tokens
//! a chunk the products are ~1,300 multiply-adds a thread -- small beside
//! the latency the chunking removes.

use crate::dialect::{Cuda, Dialect, Msl};
use crate::gemm::Backend;
use crate::program::Lowered;

/// A layer's shape, and the chunking.
#[derive(Clone, Copy, Debug)]
pub struct DeltaChunk {
    pub tokens: usize,
    pub v_heads: usize,
    /// Width of `q` and `k`, and the state's columns.
    pub k_dim: usize,
    /// The state's rows per head, and the width of `v` and `y` per head.
    pub v_dim: usize,
    /// Where `v` starts in its buffer, and how wide a token's row of it is.
    pub v_base: usize,
    pub v_width: usize,
}

/// State rows per threadgroup, and tokens per chunk. Threadgroup memory
/// holds the state block, a chunk's `k` and `q`, and six `C x R` or
/// `C x C` tiles: 31.6 KB of the 32 a threadgroup may declare.
const R: usize = 16;
const C: usize = 16;
const THREADS: usize = 128;

pub fn fits(d: &DeltaChunk) -> bool {
    d.tokens > 0
        && d.tokens.is_multiple_of(C)
        && d.k_dim.is_multiple_of(8)
        && d.v_dim.is_multiple_of(R)
        && d.v_base + d.v_heads * d.v_dim <= d.v_width
}

pub fn delta_chunked(d: &DeltaChunk, backend: Backend) -> Result<Lowered, String> {
    if !fits(d) {
        return Err(format!(
            "delta_chunked: {} tokens in chunks of {C}, {} key dims in 8s, {} rows in blocks of {R}",
            d.tokens, d.k_dim, d.v_dim
        ));
    }
    let entry = format!(
        "delta_chunked{}_h{}_d{}x{}_c{C}_r{R}",
        d.tokens, d.v_heads, d.k_dim, d.v_dim
    );
    let ld = d.k_dim + 4;
    let words = R * ld + 2 * C * ld + 6 * C * C.max(R) + 2 * C;
    let (template, includes) = match backend {
        Backend::Metal => (SOURCE, Msl.includes()),
        Backend::Cuda => (SOURCE_CUDA, Cuda.includes()),
    };
    let source = template
        .replace("$ENTRY", &entry)
        .replace("$INCLUDES", &includes)
        .replace("$TOKENS", &d.tokens.to_string())
        .replace("$HV", &d.v_heads.to_string())
        .replace("$DK", &d.k_dim.to_string())
        .replace("$DV", &d.v_dim.to_string())
        .replace("$VBASE", &d.v_base.to_string())
        .replace("$VW", &d.v_width.to_string())
        .replace("$LD", &ld.to_string())
        .replace("$WORDS", &words.to_string())
        .replace("$R", &R.to_string())
        .replace("$C", &C.to_string())
        .replace("$T", &THREADS.to_string());
    Ok(Lowered {
        entry,
        source,
        grid: d.v_dim / R,
        grid2: d.v_heads,
        threads: THREADS,
        threadgroup_bytes: 4 * words,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 6,
        // state, q, k, v, g, beta, y
        writes: vec![true, false, false, false, false, false, true],
    })
}

/// `$`-names are substituted; no other text is.
const SOURCE: &str = r#"// Generated by lex-msl::delta. Hand-scheduled, not lowered from a program.
// kernel : $ENTRY
// launch : ($DV / $R) x $HV threadgroups x $T threads, chunks of $C tokens
$INCLUDES#include <metal_simdgroup_matrix>

kernel void $ENTRY(
    device float *state [[buffer(0)]],
    device const float *q [[buffer(1)]],
    device const float *k [[buffer(2)]],
    device const float *v [[buffer(3)]],
    device const float *g [[buffer(4)]],
    device const float *beta [[buffer(5)]],
    device float *y [[buffer(6)]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint DK = $DKu, DV = $DVu, HV = $HVu, GATES = $HVu * $DVu;
    const uint row0 = tgpos.x * $Ru, head = tgpos.y;
    threadgroup float buf[$WORDS];
    threadgroup float *Ss = buf;                  // [R][LD]  the state block
    threadgroup float *Ks = Ss + $Ru * $LDu;      // [C][LD]  the chunk's k
    threadgroup float *Qs = Ks + $Cu * $LDu;      // [C][LD]  and q
    threadgroup float *KK = Qs + $Cu * $LDu;      // [C][C]   k_j . k_l
    threadgroup float *QK = KK + $Cu * $Cu;       // [C][C]   q_i . k_l
    threadgroup float *KS = QK + $Cu * $Cu;       // [C][R]   S_0 k_j, per row
    threadgroup float *QS = KS + $Cu * $Ru;       // [C][R]   S_0 q_i
    threadgroup float *U = QS + $Cu * $Ru;        // [C][R]   u_j
    threadgroup float *Vb = U + $Cu * $Ru;        // [C][R]   v_j, this block's rows
    threadgroup float *G = Vb + $Cu * $Ru;        // [C]      running sum of log g
    threadgroup float *B = G + $Cu;               // [C]      beta

    // The state block in, once.
    for (uint e = tid; e < $Ru * DK / 4u; e += $Tu) {
        const uint r = e / (DK / 4u), c = (e % (DK / 4u)) * 4u;
        *(threadgroup float4 *)(Ss + r * $LDu + c) =
            *(device const float4 *)(state + (size_t)(head * DV + row0 + r) * DK + c);
    }

    for (uint t0 = 0; t0 < $TOKENSu; t0 += $Cu) {
        // 1. The chunk: k and q rows, this block's v, the gates.
        for (uint e = tid; e < $Cu * DK / 4u; e += $Tu) {
            const uint j = e / (DK / 4u), c = (e % (DK / 4u)) * 4u;
            const size_t at = (size_t)((t0 + j) * HV + head) * DK + c;
            *(threadgroup float4 *)(Ks + j * $LDu + c) = *(device const float4 *)(k + at);
            *(threadgroup float4 *)(Qs + j * $LDu + c) = *(device const float4 *)(q + at);
        }
        for (uint e = tid; e < $Cu * $Ru; e += $Tu) {
            const uint j = e / $Ru, r = e % $Ru;
            Vb[e] = v[(size_t)(t0 + j) * $VWu + $VBASEu + head * DV + row0 + r];
        }
        if (sgid == 0u) {
            // g and beta are one value per head, spread over its rows.
            float lg = 0.0f;
            if (lane < $Cu) {
                const size_t at = (size_t)(t0 + lane) * GATES + head * DV + row0;
                // A gate that underflowed to zero would make the running
                // sum -inf and its differences NaN; e^-80 is zero anyway.
                lg = max(log(g[at]), -80.0f);
                B[lane] = beta[at];
            }
            const float run = simd_prefix_inclusive_sum(lg);
            if (lane < $Cu) G[lane] = run;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Four products over the key dimension, one per simdgroup:
        //    K Kᵀ, Q Kᵀ (C x C) and K Sᵀ, Q Sᵀ (C x R).
        {
            const threadgroup float *A = (sgid == 1u || sgid == 3u) ? Qs : Ks;
            const threadgroup float *Bt = sgid < 2u ? Ks : Ss;
            threadgroup float *O = sgid == 0u ? KK : sgid == 1u ? QK : sgid == 2u ? KS : QS;
            const uint n = sgid < 2u ? $Cu : $Ru;
            simdgroup_matrix<float, 8, 8> acc[$C / 8][2];
            for (uint i = 0; i < $Cu / 8u; ++i)
                for (uint jj = 0; jj < 2u; ++jj) acc[i][jj] = simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint kk = 0; kk < DK; kk += 8u) {
                simdgroup_matrix<float, 8, 8> a[$C / 8], b[2];
                for (uint i = 0; i < $Cu / 8u; ++i)
                    simdgroup_load(a[i], A + i * 8u * $LDu + kk, $LDu);
                for (uint jj = 0; jj < n / 8u; ++jj)
                    simdgroup_load(b[jj], Bt + jj * 8u * $LDu + kk, $LDu, ulong2(0, 0), true);
                for (uint i = 0; i < $Cu / 8u; ++i)
                    for (uint jj = 0; jj < n / 8u; ++jj)
                        simdgroup_multiply_accumulate(acc[i][jj], a[i], b[jj], acc[i][jj]);
            }
            for (uint i = 0; i < $Cu / 8u; ++i)
                for (uint jj = 0; jj < n / 8u; ++jj)
                    simdgroup_store(acc[i][jj], O + i * 8u * n + jj * 8u, n);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 3. u by substitution. Eight lanes per state row: lane `s` of a
        //    row's eight keeps u_l for l = s, s + 8, ... and adds those
        //    terms; the eight partial sums meet by shuffle.
        {
            const uint r = tid / 8u, s = tid % 8u;
            float own[$C / 8];
            for (uint j = 0; j < $Cu; ++j) {
                const float gj = G[j], bj = B[j];
                float part = 0.0f;
                for (uint m = 0; m * 8u + s < j; ++m) {
                    const uint l = m * 8u + s;
                    part += exp(gj - G[l]) * KK[j * $Cu + l] * own[m];
                }
                part += simd_shuffle_xor(part, 1u);
                part += simd_shuffle_xor(part, 2u);
                part += simd_shuffle_xor(part, 4u);
                const float u = bj * (Vb[j * $Ru + r] - exp(gj) * KS[j * $Ru + r] - part);
                if (s == j % 8u) {
                    own[j / 8u] = u;
                    U[j * $Ru + r] = u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. The chunk's outputs.
        for (uint e = tid; e < $Cu * $Ru; e += $Tu) {
            const uint i = e / $Ru, r = e % $Ru;
            const float gi = G[i];
            float o = exp(gi) * QS[i * $Ru + r];
            for (uint l = 0; l <= i; ++l) o += exp(gi - G[l]) * QK[i * $Cu + l] * U[l * $Ru + r];
            y[(size_t)(t0 + i) * GATES + head * DV + row0 + r] = o;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. The state at the chunk's end: decay it, weight each u by what
        //    is left of the decay after it, and add Uᵀ K.
        {
            const float glast = G[$Cu - 1u];
            const float d = exp(glast);
            for (uint e = tid; e < $Ru * DK; e += $Tu) {
                const uint r = e / DK, c = e % DK;
                Ss[r * $LDu + c] *= d;
            }
            for (uint e = tid; e < $Cu * $Ru; e += $Tu) U[e] *= exp(glast - G[e / $Ru]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // R x DK outputs as 8x8 fragments, split over the simdgroups.
        for (uint f = sgid; f < ($Ru / 8u) * (DK / 8u); f += $T / 32u) {
            const uint fr = (f / (DK / 8u)) * 8u, fc = (f % (DK / 8u)) * 8u;
            simdgroup_matrix<float, 8, 8> acc, a, b;
            simdgroup_load(acc, Ss + fr * $LDu + fc, $LDu);
            for (uint l = 0; l < $Cu; l += 8u) {
                // Uᵀ: U is [C][R], the fragment wants rows of R.
                simdgroup_load(a, U + l * $Ru + fr, $Ru, ulong2(0, 0), true);
                simdgroup_load(b, Ks + l * $LDu + fc, $LDu);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, Ss + fr * $LDu + fc, $LDu);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint e = tid; e < $Ru * DK / 4u; e += $Tu) {
        const uint r = e / (DK / 4u), c = (e % (DK / 4u)) * 4u;
        *(device float4 *)(state + (size_t)(head * DV + row0 + r) * DK + c) =
            *(threadgroup float4 *)(Ss + r * $LDu + c);
    }
}
"#;

/// The same kernel in CUDA C, the products on the ordinary cores.
const SOURCE_CUDA: &str = r#"// Generated by lex-msl::delta. Hand-scheduled, not lowered from a program.
// kernel : $ENTRY
// launch : ($DV / $R) x $HV blocks x $T threads, chunks of $C tokens
$INCLUDES
extern "C" __global__ void $ENTRY(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    float* __restrict__ y)
{
    const uint DK = $DKu, DV = $DVu, HV = $HVu, GATES = $HVu * $DVu;
    const uint tid = threadIdx.x, warp = tid / 32u, lane = tid % 32u;
    const uint row0 = blockIdx.x * $Ru, head = blockIdx.y;
    __shared__ __align__(16) float buf[$WORDS];
    float *Ss = buf;                  // [R][LD]  the state block
    float *Ks = Ss + $Ru * $LDu;      // [C][LD]  the chunk's k
    float *Qs = Ks + $Cu * $LDu;      // [C][LD]  and q
    float *KK = Qs + $Cu * $LDu;      // [C][C]   k_j . k_l
    float *QK = KK + $Cu * $Cu;       // [C][C]   q_i . k_l
    float *KS = QK + $Cu * $Cu;       // [C][R]   S_0 k_j, per row
    float *QS = KS + $Cu * $Ru;       // [C][R]   S_0 q_i
    float *U = QS + $Cu * $Ru;        // [C][R]   u_j
    float *Vb = U + $Cu * $Ru;        // [C][R]   v_j, this block's rows
    float *G = Vb + $Cu * $Ru;        // [C]      running sum of log g
    float *B = G + $Cu;               // [C]      beta

    for (uint e = tid; e < $Ru * DK / 4u; e += $Tu) {
        const uint r = e / (DK / 4u), c = (e % (DK / 4u)) * 4u;
        *reinterpret_cast<float4*>(Ss + r * $LDu + c) =
            *reinterpret_cast<const float4*>(state + (size_t)(head * DV + row0 + r) * DK + c);
    }

    for (uint t0 = 0; t0 < $TOKENSu; t0 += $Cu) {
        // 1. The chunk: k and q rows, this block's v, the gates.
        for (uint e = tid; e < $Cu * DK / 4u; e += $Tu) {
            const uint j = e / (DK / 4u), c = (e % (DK / 4u)) * 4u;
            const size_t at = (size_t)((t0 + j) * HV + head) * DK + c;
            *reinterpret_cast<float4*>(Ks + j * $LDu + c) = *reinterpret_cast<const float4*>(k + at);
            *reinterpret_cast<float4*>(Qs + j * $LDu + c) = *reinterpret_cast<const float4*>(q + at);
        }
        for (uint e = tid; e < $Cu * $Ru; e += $Tu) {
            const uint j = e / $Ru, r = e % $Ru;
            Vb[e] = v[(size_t)(t0 + j) * $VWu + $VBASEu + head * DV + row0 + r];
        }
        if (warp == 0u) {
            // g and beta are one value per head, spread over its rows.
            float lg = 0.0f;
            if (lane < $Cu) {
                const size_t at = (size_t)(t0 + lane) * GATES + head * DV + row0;
                // A gate that underflowed to zero would make the running
                // sum -inf and its differences NaN; e^-80 is zero anyway.
                lg = fmaxf(logf(g[at]), -80.0f);
                B[lane] = beta[at];
            }
            for (uint d = 1u; d < 32u; d <<= 1u) {
                const float o = __shfl_up_sync(0xffffffffu, lg, d);
                if (lane >= d) lg += o;
            }
            if (lane < $Cu) G[lane] = lg;
        }
        __syncthreads();

        // 2. Four products over the key dimension, one per warp: K Kᵀ,
        //    Q Kᵀ (C x C) and K Sᵀ, Q Sᵀ (C x R), eight outputs a lane.
        {
            const float *A = (warp == 1u || warp == 3u) ? Qs : Ks;
            const float *Bt = warp < 2u ? Ks : Ss;
            float *O = warp == 0u ? KK : warp == 1u ? QK : warp == 2u ? KS : QS;
            const uint n = warp < 2u ? $Cu : $Ru;
            for (uint e = lane; e < $Cu * n; e += 32u) {
                const uint i = e / n, j = e % n;
                const float4 *a = reinterpret_cast<const float4*>(A + i * $LDu);
                const float4 *b = reinterpret_cast<const float4*>(Bt + j * $LDu);
                float acc = 0.0f;
                for (uint c = 0; c < DK / 4u; ++c) {
                    const float4 x = a[c], w = b[c];
                    acc += x.x * w.x + x.y * w.y + x.z * w.z + x.w * w.w;
                }
                O[i * n + j] = acc;
            }
        }
        __syncthreads();

        // 3. u by substitution: eight lanes a state row, partial sums
        //    meeting by shuffle.
        {
            const uint r = tid / 8u, s = tid % 8u;
            float own[$C / 8];
            for (uint j = 0; j < $Cu; ++j) {
                const float gj = G[j], bj = B[j];
                float part = 0.0f;
                for (uint m = 0; m * 8u + s < j; ++m) {
                    const uint l = m * 8u + s;
                    part += expf(gj - G[l]) * KK[j * $Cu + l] * own[m];
                }
                part += __shfl_xor_sync(0xffffffffu, part, 1u);
                part += __shfl_xor_sync(0xffffffffu, part, 2u);
                part += __shfl_xor_sync(0xffffffffu, part, 4u);
                const float u = bj * (Vb[j * $Ru + r] - expf(gj) * KS[j * $Ru + r] - part);
                if (s == j % 8u) {
                    own[j / 8u] = u;
                    U[j * $Ru + r] = u;
                }
            }
        }
        __syncthreads();

        // 4. The chunk's outputs.
        for (uint e = tid; e < $Cu * $Ru; e += $Tu) {
            const uint i = e / $Ru, r = e % $Ru;
            const float gi = G[i];
            float o = expf(gi) * QS[i * $Ru + r];
            for (uint l = 0; l <= i; ++l) o += expf(gi - G[l]) * QK[i * $Cu + l] * U[l * $Ru + r];
            y[(size_t)(t0 + i) * GATES + head * DV + row0 + r] = o;
        }
        __syncthreads();

        // 5. The state at the chunk's end: each u weighted by what is left
        //    of the decay after it, then S = e^G_C S + Uᵀ K.
        {
            const float glast = G[$Cu - 1u];
            for (uint e = tid; e < $Cu * $Ru; e += $Tu) U[e] *= expf(glast - G[e / $Ru]);
        }
        __syncthreads();
        {
            const float d = expf(G[$Cu - 1u]);
            for (uint e = tid; e < $Ru * DK; e += $Tu) {
                const uint r = e / DK, c = e % DK;
                float s = d * Ss[r * $LDu + c];
                for (uint l = 0; l < $Cu; ++l) s += U[l * $Ru + r] * Ks[l * $LDu + c];
                Ss[r * $LDu + c] = s;
            }
        }
        __syncthreads();
    }

    for (uint e = tid; e < $Ru * DK / 4u; e += $Tu) {
        const uint r = e / (DK / 4u), c = (e % (DK / 4u)) * 4u;
        *reinterpret_cast<float4*>(state + (size_t)(head * DV + row0 + r) * DK + c) =
            *reinterpret_cast<const float4*>(Ss + r * $LDu + c);
    }
}
"#;
