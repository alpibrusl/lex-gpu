//! Is `simdgroup_matrix` worth building into the compiler?
//!
//!     cargo run --release -p lex-rt --example gemm_probe
//!
//! MLX has two quantised paths: `qmv*` for a matrix-vector, and `qmm_t`, a
//! tiled GEMM on `BlockMMA` — 32x32x32 tiles, 2x2 simdgroups, weights
//! dequantised into threadgroup memory by `QuantizedBlockLoader` and then
//! read as MMA fragments. This compiler has only the first shape, and that
//! is the whole of the prefill gap: 75 tok/s against Ollama's ~250.
//!
//! Before teaching the IR about matrix fragments — which is weeks, and
//! which `docs/design.md` has wanted since the beginning — this is a
//! hand-written kernel of the same shape, benchmarked against the batched
//! matvec it would replace. If it does not win here it will not win from
//! inside the compiler either.
//!
//! ## What it found
//!
//! **Revised 2026-09-24, and the earlier conclusion was wrong.** It read:
//! the best MMA configuration (32 tokens) costs 9.24 ms of a notional pass
//! per token against the batched matvec's 8.29 at eight, so the matvec
//! wins and matrix fragments are not worth building.
//!
//! That was measured with `BM = 32` and a grid with no token dimension.
//! The kernel could not compute more than 32 tokens, so 32 was not "the
//! best configuration", it was the only one -- and asking for 64 gave a
//! 32-token result with 64 in the denominator. The tell was the clock: 32,
//! 64 and 128 tokens all took about 1,040 µs. The probe now asserts that
//! both kernels wrote every row, because two kernels that stop early agree
//! with each other perfectly.
//!
//! With a token dimension on the grid:
//!
//! ```text
//!   tokens   batched matvec   simdgroup_matrix
//!        8             8.49              28.32
//!       16            20.00              15.01
//!       32           258.32               9.98
//!       64           258.57               7.99
//!      128           258.76               6.95
//! ```
//!
//! So MMA does win, and the crossover is at 16 tokens. At 128 it is 6.95
//! against the matvec's best-ever 8.49 -- **18%**, not the multiple the
//! shape of the curve suggested. The matvec past 16 tokens is not a
//! contest: `r x tokens` accumulators spill and it collapses to 258.
//!
//! 18% is a real result and a thin reason to spend weeks on matrix
//! fragments in the IR. What makes it interesting is the gap above it:
//! 6.95 ms/token is 144 tok/s of a notional pass, and prefill at this
//! shape is compute-bound at about 246 -- 28.5 TFLOP for 512 tokens
//! against a measured 13.7 TFLOP/s. This kernel reaches 59% of that, and
//! it stages through threadgroup memory with no double buffering and no
//! vectorised loads.
//!
//! ## The configuration knobs are exhausted
//!
//! All measured at 128 tokens, against 6.95 at `BN=32, BK=32`:
//!
//! ```text
//!   BK=64                     7.46    (all 128 threads staging weights,
//!                                      against BK=32 leaving half idle)
//!   BN=64  (272 threadgroups) 7.06
//!   BN=128 (136 threadgroups) 10.61
//! ```
//!
//! Neither lever moves it. `BK=64` was meant to fix the weight staging,
//! where `BN*BK/16 = 64` groups over 128 threads leaves half the
//! threadgroup idle through the expensive part of the loop; the extra
//! threadgroup memory costs more in occupancy than the idle threads cost.
//! `BN` controls how many threadgroups re-read the activations -- 713 MB
//! of them at `BN=32` -- and halving that changes nothing, which is the
//! same answer the batched matvec gave to the same question. Those loads
//! are cache-served and cheap in both kernels.
//!
//! So: 3.09 ms at 128 tokens against a 1.67 ms compute floor, 54% of peak,
//! and it is not bandwidth, not threadgroup count, and not the idle
//! threads. What is left is the shape of the loop -- double buffering, so
//! the dequantise of the next tile overlaps the matrix ops on this one,
//! and bulk loads -- and that is real kernel work rather than a sweep.
//!
//! Anyone picking this up: the bar is 6.95, and five configuration
//! hypotheses have already died against it.
//!
//! The thing that made this look impossible earlier was NVFP4\'s per-16
//! scale: it has nowhere to go inside an MMA accumulator. Staging solves
//! it. The scale is applied while dequantising into threadgroup memory, so
//! the matrix units only ever see plain halves. That is also why an earlier
//! measurement here -- threadgroup staging at 92 GB/s against 141 -- was
//! true and irrelevant: it was measured at one token, where a staged tile
//! is reused once and staging is pure overhead.
//!
#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use lex_front::llama::{QLayout, matmul_q_x};
    use lex_ir::DType;
    use lex_metal::{Buffer, Gpu, Step};
    use lex_msl::program::lower;

    let gpu = Gpu::open()?;
    // Qwen's feed-forward, which is 60% of a prefill chunk.
    let (k, n) = (5120usize, 17408usize);
    println!("{}: {k} -> {n}, NVFP4", gpu.info().name);

    // Real bytes. Every NVFP4 code being zero decodes to one cache line and
    // has flattered this repository's numbers before.
    let codes: Vec<u8> = (0..n * k / 2)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8)
        .collect();
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|i| 0x38u8.wrapping_add(i as u8 % 7))
        .collect();
    let gs: Vec<f32> = vec![1.0 / 16384.0; n];
    let weight_bytes = codes.len() + scales.len() + gs.len() * 4;
    // Enough copies to exceed any cache, rotated across the reps. One set
    // read eight times in a row is a cache benchmark, which is how this
    // probe first reported 175 TB/s.
    let copies = (1usize << 30).div_ceil(weight_bytes).clamp(2, 16);
    let sets: Vec<(Buffer, Buffer, Buffer)> = (0..copies)
        .map(|_| (gpu.upload(&codes), gpu.upload(&scales), gpu.upload(&gs)))
        .collect();

    println!("\n  tokens   path                us     GB/s   ms/token(pass)");
    // BN sets how many threadgroups there are, and every one of them
    // re-reads every token's activations: at BN=32 that is 544 of them,
    // moving 178 MB of activations against 50 MB of weights.
    let bn: usize = std::env::var("LEX_BN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    println!("  BN={bn}, {} threadgroups", n / bn);
    // Past 32 because that is where the question actually is: prefill
    // re-reads the weights once per chunk, so a 512-token prompt at
    // MAX_BATCH=8 reads 14.5 GB sixty-four times. Whether a matrix-unit
    // kernel wins at eight tokens is beside the point if it is the only
    // thing that can run at sixty-four.
    let toks: Vec<usize> = std::env::var("LEX_TOKENS")
        .ok()
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 8, 16, 32, 64, 128]);
    for m in toks {
        // Varied, not a constant. With every activation 0.5 and every E2M1
        // value a multiple of 0.25, all partial sums are exact in f32 and
        // the two kernels agree bit-for-bit whatever order they sum in --
        // which looks like a passing comparison and tests nothing.
        let x16: Vec<u16> = (0..m * k)
            .map(|i| {
                let f = ((i.wrapping_mul(2_654_435_761) >> 11) as f32 / 1.048e6) - 1.0;
                half::f16::from_f32(f).to_bits()
            })
            .collect();
        let x = gpu.upload(&x16);
        let y = gpu.zeroed::<f32>(m * n);

        // What we have today, at its best setting.
        let prog = matmul_q_x(m, k, n, 32, k, QLayout::NVFP4, false, DType::F16)?;
        let ours = lower(&prog, gpu.target(), 256)?;
        let p_ours = gpu.build_lowered(&ours)?;

        // What MLX has: a 32x32x32 tile on simdgroup_matrix, weights
        // dequantised into threadgroup memory on the way in.
        let mma = mma_kernel(m, k, n, bn);
        let p_mma = match gpu.build_lowered(&mma) {
            Ok(p) => p,
            Err(e) => return Err(format!("the probe kernel does not compile:\n{e}")),
        };

        // Correctness first. A fast kernel that computes the wrong thing is
        // the easiest way to believe a bad idea, and this probe exists to
        // decide whether to spend weeks on the idea.
        {
            let (q0, s0, g0) = &sets[0];
            let check: Vec<&Buffer> = vec![&x, q0, s0, g0, &y];
            gpu.run_launches(&[(&p_ours, check.as_slice(), None)]);
            let mut want = vec![0.0f32; m * n];
            gpu.download(&y, &mut want);

            let y2 = gpu.zeroed::<f32>(m * n);
            let check2: Vec<&Buffer> = vec![&x, q0, s0, g0, &y2];
            gpu.run_launches(&[(&p_mma, check2.as_slice(), None)]);
            let mut got = vec![0.0f32; m * n];
            gpu.download(&y2, &mut got);

            // Which rows each kernel actually wrote. A kernel whose tile
            // does not cover `m` leaves the rest of `y` as it found it --
            // zero -- and two kernels that both stop early agree
            // perfectly, which reads as a pass. The MMA tile is BM=32 and
            // its grid had no token dimension, so every number this probe
            // reported past 32 tokens was a 32-token result divided by m.
            let rows = |v: &[f32]| (0..m).filter(|r| v[r * n..(r + 1) * n].iter().any(|x| *x != 0.0)).count();
            let (rw, rg) = (rows(&want), rows(&got));
            assert!(
                rw == m && rg == m,
                "m={m}: matvec wrote {rw} of {m} rows, mma wrote {rg} -- \
                 a kernel that stops early agrees with another that does"
            );
            let scale = want.iter().fold(1e-6f32, |a, b| a.max(b.abs()));
            // An all-zero output agrees with anything. This repository has
            // measured 435 GB/s off zeroed buffers once already.
            assert!(
                scale > 1e-3,
                "m={m}: the reference output is ~zero ({scale:e}); nothing is being computed"
            );
            let err = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / scale;
            // f16 accumulation in the staged tile against f32 in the
            // matvec: a few 1e-3 of scale is the format, not a bug.
            assert!(
                err < 5e-3,
                "m={m}: the MMA kernel disagrees with the matvec by {err:e} of scale"
            );
            println!(
                "  {m:>6}   agreement {err:.2e} of scale, |max| {scale:.3}, \
                 got[0] {:.4} want[0] {:.4}",
                got[0], want[0]
            );
        }

        let binds: Vec<Vec<&Buffer>> = sets.iter().map(|(q, s, g)| vec![&x, q, s, g, &y]).collect();
        for (name, pipe) in [("batched matvec", &p_ours), ("simdgroup_matrix", &p_mma)] {
            let reps = (4 * copies).max(32);
            let steps: Vec<Step<'_>> = (0..reps)
                .map(|i| (pipe, binds[i % copies].as_slice(), None))
                .collect();
            gpu.run_launches(&steps);
            // `run_launches` returns seconds; median of five.
            let mut runs: Vec<f64> = (0..5)
                .map(|_| gpu.run_launches(&steps).1 / reps as f64)
                .collect();
            runs.sort_by(f64::total_cmp);
            let per = runs[2];
            println!(
                "  {m:>6}   {name:<16} {:>7.1}  {:>7.0}   {:>8.2}",
                per * 1e6,
                weight_bytes as f64 / per / 1e9,
                // A whole 14.5 GB pass at this rate, per token.
                14.5e9 / (weight_bytes as f64 / per) * 1e3 / m as f64
            );
        }
    }
    Ok(())
}

/// A 32x32x32 quantised GEMM on `simdgroup_matrix`, by hand.
///
/// `y[m, j] = sum_p x[m, p] * w[j, p]`, with `w` NVFP4. One threadgroup owns
/// a 32x32 tile of the output and four simdgroups split it 2x2, each holding
/// a 16x16 accumulator as four 8x8 fragments.
#[cfg(target_os = "macos")]
fn mma_kernel(m: usize, k: usize, n: usize, bn: usize) -> lex_msl::program::Lowered {
    // The K tile. At BK=32 with BN=32 the weight staging has BN*BK/16 = 64
    // groups for 128 threads, so half the threadgroup stands idle through
    // the expensive part of the loop; BK=64 gives it all of them and
    // halves the barrier count with it.
    let bk: usize = std::env::var("LEX_BK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32);
    // Weights are 8 codes to a u32 word; scales are one byte per 16 values.
    let kw = k / 8;
    let ks = k / 16;
    let nf = (bn / 2) / 8;
    let bnp = bn + 4;
    let source = format!(
        r#"#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Both E2M1 codes of a byte, straight into half's exponent field: the value
// times 2^-14, subnormals included. The 2^14 rides on the group scale.
inline half2 fp4_pair(uint b) {{
    const uint w = b | (b << 12u);
    return as_type<half2>(((w & 0x00070007u) << 9u) | ((w & 0x00080008u) << 12u));
}}
inline float fp8_e4m3(uint b) {{
    const uint e = (b >> 3u) & 0xFu, mm = b & 7u;
    const uint bits = select(((e + 120u) << 23u) | (mm << 20u),
                             as_type<uint>(float(mm) * 0.001953125f), e == 0u);
    return as_type<float>(bits | ((b & 0x80u) << 24u));
}}

kernel void gemm_nvfp4(
    device const half *x [[buffer(0)]],
    device const uchar *q [[buffer(1)]],
    device const uchar *s [[buffer(2)]],
    device const float *gs [[buffer(3)]],
    device float *y [[buffer(4)]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint M = {m}u, K = {k}u, N = {n}u;
    const uint BM = 32u, BN = {bn}u, BK = {bk}u;
    // Padded to keep a column read off one bank.
    threadgroup half As[32][{bkp}];
    threadgroup half Bs[{bn}][{bkp}];

    const uint j0 = tgpos.x * BN;              // first output column block
    // Token tiles down the grid's second dimension. Without this the tile
    // is the whole kernel: BM=32 rows and nothing past them, so a batch of
    // 64 quietly produced 32 rows and a timing to match.
    const uint m0 = tgpos.y * BM;              // first token of this tile
    const uint wm = sgid / 2u, wn = sgid % 2u; // 2x2 simdgroups

    // 2x2 simdgroups over a BM x BN tile: each owns 16 rows and BN/2
    // columns, held as 2 x NF fragments of 8x8.
    const uint NF = {nf}u;
    simdgroup_matrix<float, 8, 8> acc[2][{nf}];
    for (uint a = 0; a < 2u; ++a)
        for (uint b = 0; b < NF; ++b)
            acc[a][b] = simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint p0 = 0; p0 < K; p0 += BK) {{
        // Stage activations: BM x BK, zero-padded past M.
        for (uint e = tid; e < BM * BK; e += 128u) {{
            const uint r = e / BK, c = e % BK;
            As[r][c] = (m0 + r < M) ? x[(m0 + r) * K + p0 + c] : half(0.0h);
        }}
        // Stage weights, dequantised. This is where the per-16 scale is
        // applied -- the matrix units only ever see plain halves, which is
        // what makes NVFP4 and MMA compatible at all.
        // One thread per group of 16, so the FP8 scale is decoded once per
        // group rather than once per value. Sixteen values is eight bytes:
        // a single aligned load, and eight `fp4_pair` on vector ALU.
        for (uint g = tid; g < BN * BK / 16u; g += 128u) {{
            const uint r = g / (BK / 16u), gc = g % (BK / 16u);
            const uint j = j0 + r;
            const uint c0 = gc * 16u;
            const half sc =
                half(fp8_e4m3((uint)s[j * {ks}u + (p0 + c0) / 16u]) * gs[j] * 16384.0f);
            const uint base = (j * {kw}u * 4u) + (p0 + c0) / 2u;
            for (uint b = 0; b < 8u; ++b) {{
                const half2 v = fp4_pair((uint)q[base + b]);
                Bs[r][c0 + b * 2u] = v.x * sc;
                Bs[r][c0 + b * 2u + 1u] = v.y * sc;
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 16x16 per simdgroup, as four 8x8 fragments.
        for (uint kk = 0; kk < BK; kk += 8u) {{
            simdgroup_matrix<half, 8, 8> a[2], b[{nf}];
            for (uint i = 0; i < 2u; ++i)
                simdgroup_load(a[i], &As[wm * 16u + i * 8u][kk], {bkp});
            for (uint i = 0; i < NF; ++i)
                // B is [row][k]; the fragment wants it transposed.
                simdgroup_load(b[i], &Bs[wn * (NF * 8u) + i * 8u][kk], {bkp}, ulong2(0, 0), true);
            for (uint i = 0; i < 2u; ++i)
                for (uint jj = 0; jj < NF; ++jj)
                    simdgroup_multiply_accumulate(acc[i][jj], a[i], b[jj], acc[i][jj]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    // Write back through threadgroup memory: simdgroup_store wants a
    // contiguous destination and the output rows are strided by N.
    threadgroup float Cs[32][{bnp}];
    for (uint i = 0; i < 2u; ++i)
        for (uint jj = 0; jj < NF; ++jj)
            simdgroup_store(
                acc[i][jj], &Cs[wm * 16u + i * 8u][wn * (NF * 8u) + jj * 8u], {bnp});
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BM * BN; e += 128u) {{
        const uint r = e / BN, c = e % BN;
        if (m0 + r < M) y[(m0 + r) * N + j0 + c] = Cs[r][c];
    }}
}}
"#,
        m = m,
        k = k,
        n = n,
        kw = kw,
        ks = ks,
        bn = bn,
        nf = nf,
        bnp = bnp,
        bk = bk,
        bkp = bk + 4,
    );
    lex_msl::program::Lowered {
        entry: "gemm_nvfp4".into(),
        source,
        grid: n / bn,
        grid2: m.div_ceil(32),
        threads: 128,
        threadgroup_bytes: 32 * (bk + 4) * 2 + bn * (bk + 4) * 2 + 32 * (bn + 4) * 4,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 3,
        writes: vec![false, false, false, false, true],
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this example needs Metal");
}
