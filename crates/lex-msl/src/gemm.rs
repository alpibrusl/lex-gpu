//! A tiled GEMM on the matrix units, for NVFP4 weights: `y = x Wᵀ (+ r)`.
//!
//! Prefill reads every weight once per chunk of tokens. The batched matvec
//! that serves a verify's few tokens keeps an accumulator per (row, token)
//! in registers, so its chunk cannot grow past about 16 before it spills
//! and collapses (258 ms/token at 32, `docs/roadmap-weeks.md` M3c) -- and at
//! 8 a 512-token prompt reads the 14.5 GB of weights sixty-four times. A
//! GEMM keeps the accumulators in the matrix units' fragments instead, so
//! the chunk can be 64 or 128 tokens and the weights are read that many
//! fewer times.
//!
//! **This kernel is hand-scheduled, not lowered from a `lex-front`
//! program**, like the P0 copy and RMSNorm. The IR has the operation
//! (`MatMulNT` against lazily dequantised weights) but no matrix fragments
//! yet, and `docs/roadmap-weeks.md` already says the GEMM should be made
//! fast as a kernel before it is made expressible. It takes exactly the
//! parameters `lex_front::llama::matmul_q_x` does, in the same order, so a
//! runtime binds either one the same way and can choose per chunk size.
//!
//! One source per backend, same tiling idea: stage a tile of activations
//! and a tile of weights -- dequantised to half on the way into shared
//! memory, which is what makes NVFP4 and the matrix units compatible at
//! all -- multiply-accumulate in fragments, and write back through shared
//! memory so partial tiles can be guarded.
//! - CUDA: `wmma` 16x16x16, 128x128 tiles at a full chunk on eight warps
//!   of 64x32, 64 inputs a step, tiles loaded 16 bytes at a time.
//! - Metal: `simdgroup_matrix` 8x8, 64x128 tiles at a full chunk on eight
//!   simdgroups of 32x32, the next step's loads in registers while this
//!   step multiplies, and each lane's accumulators written straight out.

use crate::dialect::{Cuda, Dialect, Msl};
use crate::program::Lowered;

/// Which matrix units to write for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Cuda,
}

/// A GEMM's shape: `m` tokens, `n` output rows, `k` inputs.
#[derive(Clone, Copy, Debug)]
pub struct Gemm {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    /// Add `r` (the residual) to the product, as the matmuls writing back
    /// into the stream do.
    pub residual: bool,
    /// The activations are half rather than f32.
    pub x_half: bool,
}

/// The K tile, and so the granularity `k` has to meet.
const BK: usize = 32;

/// Whether `g` can go through [`gemm_nvfp4`] at all. Output rows and tokens
/// may be ragged (they are guarded); the reduction may not.
pub fn fits(g: &Gemm) -> bool {
    g.m > 0 && g.n > 0 && g.k.is_multiple_of(BK)
}

pub fn gemm_nvfp4(g: &Gemm, backend: Backend) -> Result<Lowered, String> {
    gemm_nvfp4_with(g, backend, None)
}

/// [`gemm_nvfp4`] with a CUDA schedule chosen by the caller (`None`: the
/// default for the shape, [`cuda_default`]). Metal ignores it.
pub fn gemm_nvfp4_with(
    g: &Gemm,
    backend: Backend,
    schedule: Option<CudaSchedule>,
) -> Result<Lowered, String> {
    if !fits(g) {
        return Err(format!(
            "gemm {}x{}x{}: the reduction must be a multiple of {BK}",
            g.m, g.n, g.k
        ));
    }
    let entry = format!(
        "gemm_nvfp4_{}x{}x{}{}{}",
        g.m,
        g.n,
        g.k,
        if g.residual { "_res" } else { "" },
        if g.x_half { "_xh" } else { "" }
    );
    let mut writes = vec![false; 4];
    if g.residual {
        writes.push(false);
    }
    writes.push(true);
    Ok(match backend {
        Backend::Cuda => {
            let sched = schedule.unwrap_or_else(|| cuda_default(g));
            if !cuda_valid(g, &sched) {
                return Err(format!(
                    "gemm {}x{}x{}: schedule {sched:?} does not fit",
                    g.m, g.n, g.k
                ));
            }
            // The schedule is in the entry name, so two schedules of one
            // shape are two functions, however a loader keys them.
            let entry = format!("{entry}_c{}", sched.name());
            cuda(g, entry, writes, sched)
        }
        Backend::Metal => metal(g, entry, writes),
    })
}

/// How a CUDA block is cut: `bm` tokens by `bn` rows, `bk` inputs a step,
/// `wgm` x `wgn` warps each owning a block of 16x16 fragments, and whether
/// the tile loads and stores are 16 bytes wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CudaSchedule {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wgm: usize,
    pub wgn: usize,
    pub vec: bool,
}

impl CudaSchedule {
    /// A short stable name, for cache keys and entry names.
    pub fn name(&self) -> String {
        format!(
            "{}x{}k{}w{}x{}{}",
            self.bm,
            self.bn,
            self.bk,
            self.wgm,
            self.wgn,
            if self.vec { "v" } else { "s" }
        )
    }
}

/// Whether `s` is a schedule the kernel can run for `g`: warps tile the
/// block in whole fragments, `bk` divides the reduction, the block fits in
/// the 48 KB of static shared memory, and it is not mostly padding.
pub fn cuda_valid(g: &Gemm, s: &CudaSchedule) -> bool {
    let warps = s.wgm * s.wgn;
    let smem = 2 * s.bm * (s.bk + 8) + 2 * s.bn * (s.bk + 8) + 4 * warps * 256;
    s.wgm > 0
        && s.wgn > 0
        && s.bm.is_multiple_of(16 * s.wgm)
        && s.bn.is_multiple_of(16 * s.wgn)
        && s.bk.is_multiple_of(16)
        && g.k.is_multiple_of(s.bk)
        && 32 * warps <= 1024
        && smem <= 48 * 1024
        && (s.bm < 2 * g.m || s.bm == 32)
}

/// The schedules worth timing for `g`. All run the same sums in the same
/// order, so they agree with the default's output.
pub fn cuda_candidates(g: &Gemm) -> Vec<CudaSchedule> {
    [
        (32, 64, 32, 2, 2),
        (64, 64, 32, 2, 2),
        (64, 64, 64, 2, 2),
        (64, 128, 32, 2, 4),
        (64, 128, 64, 2, 4),
        (128, 64, 32, 2, 2),
        (128, 64, 64, 4, 2),
        (128, 128, 32, 2, 4),
        (128, 128, 64, 2, 4),
    ]
    .into_iter()
    .map(|(bm, bn, bk, wgm, wgn)| CudaSchedule {
        bm,
        bn,
        bk,
        wgm,
        wgn,
        vec: true,
    })
    .filter(|s| cuda_valid(g, s))
    .collect()
}

/// The default schedule for `g`, with `LEX_GEMM_CU_{BM,BN,BK,WM,WN,VEC}`
/// overriding it (`examples/gemm_bench` sweeps with them).
pub fn cuda_default(g: &Gemm) -> CudaSchedule {
    let pick = |var: &str, d: usize| {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    // Measured on an L4 at 512 tokens (`examples/gemm_bench`), TFLOPS for
    // gate/up, down, qkv, out_proj:
    //
    //   64x64, BK 32, 4 warps, scalar loads (the first)   19 22 25 28
    //   the same, 16-byte loads                           23 27 35 36
    //   128x64, 16-byte loads                             32 32 47 42
    //   128x128, BK 64, 8 warps, 16-byte loads            41 36 55 47
    //
    // Taller tiles decode each weight for more tokens -- the FP4 decode,
    // not the multiply, is what a tile costs on Ada -- and a deeper K step
    // halves the barriers. A chunk of 64 tokens or fewer keeps the smaller
    // tile rather than spend most of one on padding.
    let full = g.m > 64;
    CudaSchedule {
        bm: pick(
            "LEX_GEMM_CU_BM",
            if g.m <= 32 {
                32
            } else if full {
                128
            } else {
                64
            },
        ),
        bn: pick("LEX_GEMM_CU_BN", if full { 128 } else { 64 }),
        bk: pick(
            "LEX_GEMM_CU_BK",
            if full && g.k.is_multiple_of(64) {
                64
            } else {
                BK
            },
        ),
        wgm: pick("LEX_GEMM_CU_WM", 2),
        wgn: pick("LEX_GEMM_CU_WN", if full { 4 } else { 2 }),
        vec: pick("LEX_GEMM_CU_VEC", 1) == 1,
    }
}

/// Whether any `LEX_GEMM_CU_*` knob is set: an explicit schedule, which a
/// tuner leaves alone.
pub fn cuda_overridden() -> bool {
    std::env::vars().any(|(k, _)| k.starts_with("LEX_GEMM_CU_"))
}

fn cuda(g: &Gemm, entry: String, writes: Vec<bool>, sched: CudaSchedule) -> Lowered {
    let CudaSchedule {
        bm,
        bn,
        bk,
        wgm,
        wgn,
        vec,
    } = sched;
    let threads = 32 * wgm * wgn;
    // Each warp owns a (bm/wgm) x (bn/wgn) block of 16x16 fragments.
    let (wtm, wtn) = (bm / wgm, bn / wgn);
    let (fm, fnn) = (wtm / 16, wtn / 16);
    // Rows padded by 8 halves (16 bytes): off the bank a column read would
    // otherwise hit every time, and still a multiple of the 16 bytes
    // `load_matrix_sync` requires of its stride.
    let bkp = bk + 8;
    let xt = if g.x_half { "__half" } else { "float" };
    let to_half = if g.x_half { "{v}" } else { "__float2half({v})" };
    let xload = to_half.replace("{v}", "x[(m0 + rr) * K + k0 + c]");
    // The two tile loads, scalar or 16 bytes at a time.
    let aload = if vec {
        let eight = if g.x_half {
            "*reinterpret_cast<const uint4*>(&x[(m0 + rr) * K + k0 + c])".to_string()
        } else {
            "pack8(*reinterpret_cast<const float4*>(&x[(m0 + rr) * K + k0 + c]), \
             *reinterpret_cast<const float4*>(&x[(m0 + rr) * K + k0 + c + 4u]))"
                .to_string()
        };
        format!(
            "for (uint e = tid; e < {bm}u * {bk}u / 8u; e += {threads}u) {{\n\
             \x20           const uint rr = e / ({bk}u / 8u), c = (e % ({bk}u / 8u)) * 8u;\n\
             \x20           *reinterpret_cast<uint4*>(&As[rr][c]) = (m0 + rr < M) ? {eight} : make_uint4(0u, 0u, 0u, 0u);\n\
             \x20       }}"
        )
    } else {
        format!(
            "for (uint e = tid; e < {bm}u * {bk}u; e += {threads}u) {{\n\
             \x20           const uint rr = e / {bk}u, c = e % {bk}u;\n\
             \x20           As[rr][c] = (m0 + rr < M) ? {xload} : __float2half(0.0f);\n\
             \x20       }}"
        )
    };
    let bstore = if vec {
        "__half2 h[8];
                for (uint b = 0; b < 8u; ++b) {
                    const float2 v = fp4_pair(((b < 4u ? w.x : w.y) >> (8u * (b & 3u))) & 0xFFu);
                    h[b] = __floats2half2_rn(v.x * sc, v.y * sc);
                }
                *reinterpret_cast<uint4*>(&Bs[rr][c0]) = *reinterpret_cast<uint4*>(&h[0]);
                *reinterpret_cast<uint4*>(&Bs[rr][c0 + 8u]) = *reinterpret_cast<uint4*>(&h[4]);"
    } else {
        "for (uint b = 0; b < 8u; ++b) {
                    const float2 v = fp4_pair(((b < 4u ? w.x : w.y) >> (8u * (b & 3u))) & 0xFFu);
                    Bs[rr][c0 + 2u * b] = __float2half(v.x * sc);
                    Bs[rr][c0 + 2u * b + 1u] = __float2half(v.y * sc);
                }"
    };
    let pack8 = if vec && !g.x_half {
        "__device__ __forceinline__ uint4 pack8(float4 a, float4 b) {
    __half2 h[4] = {__floats2half2_rn(a.x, a.y), __floats2half2_rn(a.z, a.w),
                    __floats2half2_rn(b.x, b.y), __floats2half2_rn(b.z, b.w)};
    return *reinterpret_cast<uint4*>(&h[0]);
}
"
    } else {
        ""
    };
    let rparam = if g.residual {
        "    const float* __restrict__ r,\n"
    } else {
        ""
    };
    let radd_t = if g.residual {
        " + r[(row + rr) * N + col + c]"
    } else {
        ""
    };
    let rfrag = if g.residual {
        "wmma::fragment<wmma::accumulator, 16, 16, 16, float> rf;
                wmma::load_matrix_sync(rf, &r[row * N + col], N, wmma::mem_row_major);
                for (int t = 0; t < rf.num_elements; ++t) acc[i][j].x[t] += rf.x[t];"
    } else {
        ""
    };
    let warps = wgm * wgn;
    let (m, n, k) = (g.m, g.n, g.k);
    let source = format!(
        r#"// Generated by lex-msl::gemm. Hand-scheduled, not lowered from a program.
// kernel : {entry}
// launch : {gx} x {gy} blocks x {threads} threads
{includes}#include <mma.h>

{fp4}
{pack8}extern "C" __global__ void {entry}(
    const {xt}* __restrict__ x,
    const char* __restrict__ q,
    const char* __restrict__ s,
    const float* __restrict__ gs,
{rparam}    float* __restrict__ y
)
{{
    using namespace nvcuda;
    const uint M = {m}u, N = {n}u, K = {k}u;
    const uint tid = threadIdx.x, warp = tid / 32u;
    const uint n0 = blockIdx.x * {bn}u, m0 = blockIdx.y * {bm}u;
    const uint wm = warp / {wgn}u, wn = warp % {wgn}u;
    __shared__ __align__(32) __half As[{bm}][{bkp}];
    __shared__ __align__(32) __half Bs[{bn}][{bkp}];
    // One 16x16 tile a warp, for a fragment that hangs past the last
    // token or row; whole ones go straight to `y`.
    __shared__ __align__(32) float Ct[{warps}][16 * 16];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[{fm}][{fnn}];
    for (int i = 0; i < {fm}; ++i)
        for (int j = 0; j < {fnn}; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    for (uint k0 = 0; k0 < K; k0 += {bk}u) {{
        // Activations: a {bm} x {bk} tile, zero past the last token.
        {aload}
        // Weights, dequantised: one thread per group of 16, so the FP8
        // scale is decoded once per group, and its 8 bytes are one load.
        // `fp4_pair` leaves values 2^14 small; the 2^14 rides on the scale.
        for (uint gi = tid; gi < {bn}u * {bk}u / 16u; gi += {threads}u) {{
            const uint rr = gi / ({bk}u / 16u), c0 = (gi % ({bk}u / 16u)) * 16u;
            const uint j = n0 + rr;
            if (j < N) {{
                const float sc = fp8_e4m3((uint)(uchar)s[j * (K / 16u) + (k0 + c0) / 16u])
                    * gs[j] * 16384.0f;
                const uint2 w = *reinterpret_cast<const uint2*>(&q[j * (K / 2u) + (k0 + c0) / 2u]);
                {bstore}
            }} else {{
                for (uint c = 0; c < 16u; ++c) Bs[rr][c0 + c] = __float2half(0.0f);
            }}
        }}
        __syncthreads();
        for (uint kk = 0; kk < {bk}u; kk += 16u) {{
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[{fm}];
            // Bs is [row][k]: as a K x N matrix that is column-major.
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[{fnn}];
            for (int i = 0; i < {fm}; ++i)
                wmma::load_matrix_sync(a[i], &As[wm * {wtm}u + i * 16u][kk], {bkp});
            for (int j = 0; j < {fnn}; ++j)
                wmma::load_matrix_sync(b[j], &Bs[wn * {wtn}u + j * 16u][kk], {bkp});
            for (int i = 0; i < {fm}; ++i)
                for (int j = 0; j < {fnn}; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }}
        __syncthreads();
    }}

    // Each warp writes its own fragments: whole ones straight to `y`, the
    // residual added fragment-wise (an accumulator loaded from `r` holds
    // its elements in the same places), ragged ones through the warp's
    // tile with each element guarded.
    const uint lane = tid % 32u;
    for (int i = 0; i < {fm}; ++i)
        for (int j = 0; j < {fnn}; ++j) {{
            const uint row = m0 + wm * {wtm}u + i * 16u, col = n0 + wn * {wtn}u + j * 16u;
            if (row + 16u <= M && col + 16u <= N) {{
                {rfrag}
                wmma::store_matrix_sync(&y[row * N + col], acc[i][j], N, wmma::mem_row_major);
            }} else {{
                wmma::store_matrix_sync(Ct[warp], acc[i][j], 16, wmma::mem_row_major);
                __syncwarp();
                for (uint e = lane; e < 256u; e += 32u) {{
                    const uint rr = e / 16u, c = e % 16u;
                    if (row + rr < M && col + c < N)
                        y[(row + rr) * N + col + c] = Ct[warp][e]{radd_t};
                }}
                __syncwarp();
            }}
        }}
}}
"#,
        includes = Cuda.includes(),
        fp4 = Cuda.fp4_preamble(),
        gx = n.div_ceil(bn),
        gy = m.div_ceil(bm),
        bk = bk,
        wgn = wgn,
    );
    Lowered {
        entry,
        source,
        grid: n.div_ceil(bn),
        grid2: m.div_ceil(bm),
        threads,
        threadgroup_bytes: 2 * bm * bkp + 2 * bn * bkp + 4 * warps * 256,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 3,
        writes,
    }
}

fn metal(g: &Gemm, entry: String, writes: Vec<bool>) -> Lowered {
    // The tile, measured rather than reasoned (`examples/gemm_metal`, ms a
    // call on an M4 Max, weights streaming from memory; tokens x rows x K,
    // then the simdgroups as tokens x rows):
    //
    //   tokens  tile            gate/up   down   qkv   out_proj
    //   128     32x64x32  2x2     2.18    2.33   1.28    0.85
    //   128     64x64x64  2x2     1.95    2.05   1.16    0.77
    //   128     64x128x64 2x4     1.90    2.05   1.13    0.74
    //    64     64x64x64  2x2     1.03    1.35   0.61    0.52
    //    64     32x64x64  2x2     1.12    1.22   0.67    0.44
    //    32     32x64x64  2x2     0.61    0.88   0.36    0.32
    //
    // Before the next step's loads were issued ahead of the multiplies the
    // answer was the other way round: taller token tiles were *slower*
    // (64x64 101 tok/s of prefill against 32x64's 123), each step waiting on
    // its own loads with nothing to hide them behind. A simdgroup holding
    // more than 32x32 of the output spills (a 128x64 tile on four
    // simdgroups ran ten times slower), so a larger tile takes more
    // simdgroups rather than larger ones. Below 128 tokens a 64-token tile
    // is half padding. `LEX_GEMM_BM`, `LEX_GEMM_BN`, `LEX_GEMM_BK`,
    // `LEX_GEMM_WM` and `LEX_GEMM_WN` measure others.
    let pick = |var: &str, default: usize| {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|t| [32, 64, 128].contains(t))
            .unwrap_or(default)
    };
    let full = g.m > 64;
    let bm = pick("LEX_GEMM_BM", if full { 64 } else { 32 });
    let bn = pick("LEX_GEMM_BN", if full { 128 } else { 64 });
    let sg = |var: &str, default: usize| {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|t| [1, 2, 4].contains(t))
            .unwrap_or(default)
    };
    let (sgm, sgn) = (
        sg("LEX_GEMM_WM", 2),
        sg("LEX_GEMM_WN", if full { 4 } else { 2 }),
    );
    let threads = 32 * sgm * sgn;
    let bk = match pick("LEX_GEMM_BK", 64) {
        64 if g.k.is_multiple_of(64) => 64,
        _ => BK,
    };
    // Simdgroups as sgm x sgn, each (bm/sgm) tokens by (bn/sgn) rows of 8x8.
    let (fm, nf) = ((bm / sgm) / 8, (bn / sgn) / 8);
    let bkp = bk + 4;
    // The two half tiles, in floats. The output is not staged through here:
    // each lane writes its accumulators straight out.
    let words = (bm * bkp + bn * bkp).div_ceil(2);
    let (xt, xv) = if g.x_half {
        ("half", "half4")
    } else {
        ("float", "float4")
    };
    // What each thread stages a K step: `na` four-value runs of activations
    // and `nb` sixteen-value weight groups, rounded up and guarded.
    let (nav, ng) = (bm * bk / 4, bn * bk / 16);
    let (na, nb) = (nav.div_ceil(threads), ng.div_ceil(threads));
    let rparam = if g.residual {
        "    device const float *r [[buffer(4)]],\n"
    } else {
        ""
    };
    let ybuf = if g.residual { 5 } else { 4 };
    let radd = if g.residual {
        " + r[row * N + col]"
    } else {
        ""
    };
    // The next K step's activations and weights, from device memory into
    // registers: issued before the current step's multiplies, so their
    // latency hides behind them instead of in front of a barrier. The
    // weights come as the group's eight bytes in one load and its scale;
    // out-of-range rows fetch nothing and stage as zeros.
    let fetch = |p: &str| {
        format!(
            r#"        for (uint i = 0; i < {na}u; ++i) {{
            const uint e = tid + i * {threads}u;
            const uint rr = e / {bk4}u, c = (e % {bk4}u) * 4u;
            xa[i] = {xv}(0);
            if (e < {nav}u && m0 + rr < M) xa[i] = *(device const {xv} *)(x + (m0 + rr) * K + {p} + c);
        }}
        for (uint i = 0; i < {nb}u; ++i) {{
            const uint gi = tid + i * {threads}u;
            const uint rr = gi / {bk16}u, c0 = (gi % {bk16}u) * 16u;
            const uint j = j0 + rr;
            wq[i] = uint2(0);
            ws[i] = 0;
            if (gi < {ng}u && j < N) {{
                wq[i] = *(device const uint2 *)(q + j * (K / 2u) + ({p} + c0) / 2u);
                ws[i] = (uchar)s[j * (K / 16u) + ({p} + c0) / 16u];
            }}
        }}
"#,
            bk4 = bk / 4,
            bk16 = bk / 16,
        )
    };
    let (m, n, k) = (g.m, g.n, g.k);
    let source = format!(
        r#"// Generated by lex-msl::gemm. Hand-scheduled, not lowered from a program.
// kernel : {entry}
// launch : {gx} x {gy} threadgroups x {threads} threads, tile {bm}x{bn}
{includes}#include <metal_simdgroup_matrix>

{fp4}
kernel void {entry}(
    device const {xt} *x [[buffer(0)]],
    device const char *q [[buffer(1)]],
    device const char *s [[buffer(2)]],
    device const float *gs [[buffer(3)]],
{rparam}    device float *y [[buffer({ybuf})]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint M = {m}u, N = {n}u, K = {k}u;
    threadgroup float buf[{words}];
    threadgroup half *As = (threadgroup half *)buf;       // [{bm}][{bkp}]
    threadgroup half *Bs = As + {bm}u * {bkp}u;            // [{bn}][{bkp}]
    const uint j0 = tgpos.x * {bn}u, m0 = tgpos.y * {bm}u;
    const uint wm = sgid / {sgn}u, wn = sgid % {sgn}u;

    simdgroup_matrix<float, 8, 8> acc[{fm}][{nf}];
    for (uint a = 0; a < {fm}u; ++a)
        for (uint b = 0; b < {nf}u; ++b) acc[a][b] = simdgroup_matrix<float, 8, 8>(0.0f);

    // A thread's weight groups keep their row from one K step to the next,
    // so the row scale is read once. The 16384 undoes `fp4_pair`'s bias.
    float gsr[{nb}];
    for (uint i = 0; i < {nb}u; ++i) {{
        const uint gi = tid + i * {threads}u, j = j0 + gi / {bk16}u;
        gsr[i] = (gi < {ng}u && j < N) ? gs[j] * 16384.0f : 0.0f;
    }}
    {xv} xa[{na}];
    uint2 wq[{nb}];
    uchar ws[{nb}];
{fetch0}
    for (uint p0 = 0; p0 < K; p0 += {bk}u) {{
        for (uint i = 0; i < {na}u; ++i) {{
            const uint e = tid + i * {threads}u;
            const uint rr = e / {bk4}u, c = (e % {bk4}u) * 4u;
            if (e < {nav}u) *(threadgroup half4 *)(As + rr * {bkp}u + c) = half4(xa[i]);
        }}
        // A group's sixteen values as four half4, two bytes each.
        for (uint i = 0; i < {nb}u; ++i) {{
            const uint gi = tid + i * {threads}u;
            if (gi >= {ng}u) continue;
            const uint rr = gi / {bk16}u, c0 = (gi % {bk16}u) * 16u;
            const float sc = fp8_e4m3((uint)ws[i]) * gsr[i];
            threadgroup half4 *row = (threadgroup half4 *)(Bs + rr * {bkp}u + c0);
            for (uint b = 0; b < 4u; ++b) {{
                const uint w = (b < 2u ? wq[i].x : wq[i].y) >> ((b & 1u) * 16u);
                const float2 lo = fp4_pair(w & 0xFFu), hi = fp4_pair((w >> 8u) & 0xFFu);
                row[b] = half4(half(lo.x * sc), half(lo.y * sc), half(hi.x * sc), half(hi.y * sc));
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (p0 + {bk}u < K) {{
{fetch1}        }}
        for (uint kk = 0; kk < {bk}u; kk += 8u) {{
            simdgroup_matrix<half, 8, 8> a[{fm}], b[{nf}];
            for (uint i = 0; i < {fm}u; ++i)
                simdgroup_load(a[i], As + (wm * {fm8}u + i * 8u) * {bkp}u + kk, {bkp});
            // Bs is [row][k]; the fragment wants it transposed.
            for (uint i = 0; i < {nf}u; ++i)
                simdgroup_load(b[i], Bs + (wn * {nf8}u + i * 8u) * {bkp}u + kk, {bkp},
                               ulong2(0, 0), true);
            for (uint i = 0; i < {fm}u; ++i)
                for (uint jj = 0; jj < {nf}u; ++jj)
                    simdgroup_multiply_accumulate(acc[i][jj], a[i], b[jj], acc[i][jj]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Each lane holds two elements of every 8x8 accumulator: row fr,
    // columns fc and fc + 1 (the layout measured on an M4 Max; a
    // simdgroup_store to a staging tile gave the same numbers and cost
    // the threadgroup memory that kept a second threadgroup off the core).
    const uint fr = (lane % 8u) / 2u + 4u * (lane / 16u);
    const uint fc = 2u * (lane % 2u) + 4u * ((lane / 8u) % 2u);
    for (uint i = 0; i < {fm}u; ++i)
        for (uint jj = 0; jj < {nf}u; ++jj) {{
            const auto e = acc[i][jj].thread_elements();
            const uint row = m0 + wm * {fm8}u + i * 8u + fr;
            const uint col = j0 + wn * {nf8}u + jj * 8u + fc;
            if (row < M && col < N) y[row * N + col] = e[0]{radd};
            if (row < M && col + 1u < N) y[row * N + col + 1u] = e[1]{radd1};
        }}
}}
"#,
        includes = Msl.includes(),
        fp4 = Msl.fp4_preamble(),
        gx = n.div_ceil(bn),
        gy = m.div_ceil(bm),
        nf8 = nf * 8,
        fm8 = fm * 8,
        bk4 = bk / 4,
        bk16 = bk / 16,
        radd1 = if g.residual {
            " + r[row * N + col + 1u]"
        } else {
            ""
        },
        fetch0 = fetch("0u"),
        fetch1 = fetch(&format!("(p0 + {bk}u)")),
    );
    Lowered {
        entry,
        source,
        grid: n.div_ceil(bn),
        grid2: m.div_ceil(bm),
        threads,
        threadgroup_bytes: 4 * words,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 3,
        writes,
    }
}
