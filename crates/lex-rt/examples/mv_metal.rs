//! Which NVFP4 matvec structure reads fastest on this Mac -- hand-written
//! variants against the one the emitter writes, at Qwen3.8's gate/up shape.
//!
//!     cargo run --release -p lex-rt --example mv_metal
//!
//! MLX's NVFP4 matvec reads the same weights at 521 GB/s here (96 us a call)
//! where ours reads ~460 (109 us). The emitted kernel gives each simdgroup one
//! output row, so every simdgroup loads the input for itself; `rows_R_sgS`
//! gives each simdgroup R rows over which one load of the input is reused,
//! S simdgroups a threadgroup. Four matrices are cycled so the system cache
//! cannot hold the 50 MB being read, and each variant's output is checked
//! against the emitted kernel's.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use lex_front::llama::{QLayout, matvec_q};
    use lex_metal::{Gpu, Step};
    use lex_msl::dialect::{Dialect, Msl};
    use lex_msl::program::{Lowered, lower_with};

    const K: usize = 5120;
    const N: usize = 17408;
    const MATS: usize = 4;
    let reps: usize = std::env::args()
        .skip_while(|a| a != "--reps")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(40);

    let gpu = Gpu::open()?;
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let x: Vec<f32> = (0..4 * K)
        .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let xb = gpu.upload(&x);
    let xh: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
    let xhb = gpu.upload(&xh);
    let mut mats = vec![];
    for _ in 0..MATS {
        let q: Vec<u8> = (0..N * K / 2).map(|_| next() as u8).collect();
        let s: Vec<u8> = (0..N * K / 16)
            .map(|_| (0x28 + (next() % 24) as u8) | (((next() & 1) as u8) << 7))
            .collect();
        let gs: Vec<f32> = (0..N).map(|_| 1e-3 * (1.0 + (next() % 8) as f32)).collect();
        mats.push((gpu.upload(&q), gpu.upload(&s), gpu.upload(&gs)));
    }
    let y = gpu.zeroed::<f32>(4 * N);
    // `--overlap`: each matrix writes its own output, so consecutive calls are
    // independent and the concurrent encoder may overlap them -- as MLX's
    // benchmark did, and as independent matvecs in a step (gate and up, q k
    // v) can. Without it every call writes one buffer and runs alone.
    let overlap = std::env::args().any(|a| a == "--overlap");
    let ys: Vec<lex_metal::Buffer> = (0..MATS).map(|_| gpu.zeroed::<f32>(4 * N)).collect();
    let bytes = (N * K / 2 + N * K / 16 + 4 * N + 4 * K) as f64;

    let emitted = lower_with(
        &matvec_q(K, N, 8, K, QLayout::NVFP4, false)?,
        gpu.target(),
        256,
        &Msl,
    )?;
    let pre = format!("{}{}", Msl.includes(), Msl.fp4_preamble());
    // R rows a simdgroup, S simdgroups a threadgroup. A lane takes 16 values
    // (8 bytes, one group, one scale) a step, loads their 16 inputs once, and
    // runs them against each of its R rows, summing a run unscaled and
    // scaling once, as the emitted kernel does per row.
    let rows = |r: usize, sg: usize, xh: bool, lut: bool, t: usize, ep: bool| -> Lowered {
        let entry = format!(
            "rows_{r}_sg{sg}{}{}{}{}",
            if xh { "_xh" } else { "" },
            if lut { "_lut" } else { "" },
            if t > 1 {
                format!("_t{t}")
            } else {
                String::new()
            },
            if ep { "_scratch" } else { "" }
        );
        let (xt, x4) = if xh {
            ("half", "half4")
        } else {
            ("float", "float4")
        };
        // The decode of one byte to two values: the preamble's bit trick
        // into half's exponent field (values 2^-14 small, the 2^14 on the
        // scale), or the 16-entry table in constant memory (true values).
        let (dec, unit) = if lut {
            (
                "float2(FP4_V[({b}) & 0xFu], FP4_V[(({b}) >> 4u) & 0xFu])",
                "1.0f",
            )
        } else {
            ("fp4_pair({b})", "16384.0f")
        };
        let d = |b: &str| dec.replace("{b}", b);
        let source = format!(
            r#"{pre}
kernel void {entry}(
    device const {xt} *x [[buffer(0)]],
    device const uchar *q [[buffer(1)]],
    device const uchar *s [[buffer(2)]],
    device const float *gs [[buffer(3)]],
    device float *y [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint K = {K}u;
    // Tokens along x, the dimension the GPU walks fastest: a row block's
    // threadgroups for every token run back to back, so the weights the
    // first reads are still in cache for the rest. Along y, a token's whole
    // pass came before the next and every token re-read 50 MB from memory.
    const uint row0 = (tg.y * {sg}u + sgid) * {r}u;
    x += tg.x * K;
    y += tg.x * {N}u;
    float acc[{r}];
    for (uint i = 0; i < {r}u; ++i) acc[i] = 0.0f;
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {{
        const device {x4} *xp = (const device {x4} *)(x + p0);
        const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
        #pragma unroll
        for (uint i = 0; i < {r}u; ++i) {{
            const uint row = row0 + i;
            const uint2 w = *(const device uint2 *)(q + row * (K / 2u) + p0 / 2u);
            float run = 0.0f; float2 v;
            v = {d0}; run += x0.x * v.x + x0.y * v.y;
            v = {d1}; run += x0.z * v.x + x0.w * v.y;
            v = {d2}; run += x1.x * v.x + x1.y * v.y;
            v = {d3}; run += x1.z * v.x + x1.w * v.y;
            v = {d4}; run += x2.x * v.x + x2.y * v.y;
            v = {d5}; run += x2.z * v.x + x2.w * v.y;
            v = {d6}; run += x3.x * v.x + x3.y * v.y;
            v = {d7}; run += x3.z * v.x + x3.w * v.y;
            acc[i] += run * fp8_e4m3(s[row * (K / 16u) + p0 / 16u]);
        }}
    }}
    {epilogue}
}}
"#,
            epilogue = if ep {
                // The emitted kernel's ending: a shuffle-down tree, a trip
                // through threadgroup memory, a barrier, one thread a row
                // storing.
                format!(
                    "threadgroup float red[{rt}];\n    \
                     for (uint i = 0; i < {r}u; ++i) {{ float v = acc[i]; \
                     for (uint d = 16u; d > 0; d /= 2) v += simd_shuffle_down(v, d); \
                     if (lane == 0u) red[sgid * {r}u + i] = v; }}\n    \
                     threadgroup_barrier(mem_flags::mem_threadgroup);\n    \
                     const uint t = sgid * 32u + lane;\n    \
                     if (t < {rt}u) {{ const uint row = tg.y * {rt}u + t; \
                     y[row] = red[t] * gs[row] * {unit}; }}",
                    rt = r * sg
                )
            } else {
                format!(
                    "for (uint i = 0; i < {r}u; ++i) {{ const float v = simd_sum(acc[i]); \
                     if (lane == 0u) y[row0 + i] = v * gs[row0 + i] * {unit}; }}"
                )
            },
            d0 = d("w.x & 0xFFu"),
            d1 = d("(w.x >> 8u) & 0xFFu"),
            d2 = d("(w.x >> 16u) & 0xFFu"),
            d3 = d("(w.x >> 24u)"),
            d4 = d("w.y & 0xFFu"),
            d5 = d("(w.y >> 8u) & 0xFFu"),
            d6 = d("(w.y >> 16u) & 0xFFu"),
            d7 = d("(w.y >> 24u)"),
        );
        Lowered {
            entry,
            source,
            grid: t,
            grid2: N / (r * sg),
            threads: 32 * sg,
            threadgroup_bytes: 0,
            arena_bytes: 0,
            scratch_bytes: 0,
            barriers: 0,
            writes: vec![false, false, false, false, true],
        }
    };
    // All `t` tokens in one pass: each lane decodes its 16 weights once and
    // runs them against every token's 16 inputs, one accumulator per
    // (token, row). The weights are read once, as the batched kernel does,
    // in the one-token structure that reads at the memory roof.
    let multi = |r: usize, sg: usize, t: usize| -> Lowered {
        let entry = format!("multi_{r}_sg{sg}_t{t}");
        let source = format!(
            r#"{pre}
kernel void {entry}(
    device const half *x [[buffer(0)]],
    device const uchar *q [[buffer(1)]],
    device const uchar *s [[buffer(2)]],
    device const float *gs [[buffer(3)]],
    device float *y [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const uint K = {K}u;
    const uint row0 = (tg.x * {sg}u + sgid) * {r}u;
    float acc[{t}][{r}];
    for (uint j = 0; j < {t}u; ++j)
        for (uint i = 0; i < {r}u; ++i) acc[j][i] = 0.0f;
    for (uint p0 = lane * 16u; p0 < K; p0 += 512u) {{
        #pragma unroll
        for (uint i = 0; i < {r}u; ++i) {{
            const uint row = row0 + i;
            const uint2 w = *(const device uint2 *)(q + row * (K / 2u) + p0 / 2u);
            const float sc = fp8_e4m3(s[row * (K / 16u) + p0 / 16u]);
            const float2 v0 = fp4_pair(w.x & 0xFFu), v1 = fp4_pair((w.x >> 8u) & 0xFFu),
                         v2 = fp4_pair((w.x >> 16u) & 0xFFu), v3 = fp4_pair(w.x >> 24u),
                         v4 = fp4_pair(w.y & 0xFFu), v5 = fp4_pair((w.y >> 8u) & 0xFFu),
                         v6 = fp4_pair((w.y >> 16u) & 0xFFu), v7 = fp4_pair(w.y >> 24u);
            #pragma unroll
            for (uint j = 0; j < {t}u; ++j) {{
                const device half4 *xp = (const device half4 *)(x + j * K + p0);
                const float4 x0 = float4(xp[0]), x1 = float4(xp[1]), x2 = float4(xp[2]), x3 = float4(xp[3]);
                const float run = x0.x * v0.x + x0.y * v0.y + x0.z * v1.x + x0.w * v1.y
                                + x1.x * v2.x + x1.y * v2.y + x1.z * v3.x + x1.w * v3.y
                                + x2.x * v4.x + x2.y * v4.y + x2.z * v5.x + x2.w * v5.y
                                + x3.x * v6.x + x3.y * v6.y + x3.z * v7.x + x3.w * v7.y;
                acc[j][i] += run * sc;
            }}
        }}
    }}
    for (uint j = 0; j < {t}u; ++j)
        for (uint i = 0; i < {r}u; ++i) {{
            const float v = simd_sum(acc[j][i]);
            if (lane == 0u) y[j * {N}u + row0 + i] = v * gs[row0 + i] * 16384.0f;
        }}
}}
"#
        );
        Lowered {
            entry,
            source,
            grid: N / (r * sg),
            grid2: 1,
            threads: 32 * sg,
            threadgroup_bytes: 0,
            arena_bytes: 0,
            scratch_bytes: 0,
            barriers: 0,
            writes: vec![false, false, false, false, true],
        }
    };
    let mut variants: Vec<(String, Lowered, bool)> = vec![("emitted".into(), emitted, false)];
    for (r, sg, xh, lut) in [
        (1, 8, false, false),
        (2, 2, false, false),
        (2, 2, false, true),
        (1, 8, true, false),
        (2, 2, true, false),
        (2, 4, true, false),
        (4, 2, true, false),
        (2, 2, true, true),
        (4, 2, true, true),
    ] {
        let l = rows(r, sg, xh, lut, 1, false);
        variants.push((l.entry.clone(), l, xh));
    }
    let l = rows(1, 8, false, false, 1, true);
    variants.push((l.entry.clone(), l, false));
    // Tokens 2..4 through the best one-token structure, one grid row a token,
    // against the batched kernel the verify uses now. x is f16, as a batch's is.
    let tok: usize = std::env::args()
        .skip_while(|a| a != "--tokens")
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    if tok > 1 {
        variants.clear();
        let batched = lower_with(
            &lex_front::llama::matmul_q_x(
                tok,
                K,
                N,
                32,
                K,
                QLayout::NVFP4,
                false,
                lex_ir::DType::F16,
            )?,
            gpu.target(),
            256,
            &Msl,
        )?;
        variants.push((format!("batched_t{tok}"), batched, true));
        for (r, sg) in [(1, 8), (1, 4), (2, 2), (2, 4), (4, 2)] {
            let l = multi(r, sg, tok);
            variants.push((l.entry.clone(), l, true));
        }
    }

    let mut reference: Option<Vec<f32>> = None;
    println!(
        "{}  ({reps} rounds over {MATS} matrices{})",
        gpu.info().name,
        if overlap { ", independent outputs" } else { "" }
    );
    println!("{:<14} {:>9} {:>8}  check", "variant", "us/call", "GB/s");
    for (name, lowered, half_x) in &variants {
        let xb = if *half_x { &xhb } else { &xb };
        let pipe = gpu
            .build_lowered(lowered)
            .map_err(|e| format!("{name}: {e}"))?;
        let (q0, s0, g0) = &mats[0];
        gpu.run(&pipe, &[xb, q0, s0, g0, &y]);
        let mut got = vec![0.0f32; 4 * N];
        gpu.download(&y, &mut got);
        let check = match &reference {
            None => {
                reference = Some(got);
                "reference".to_string()
            }
            Some(want) => {
                let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
                let err = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                format!("{:.1e} of scale", err / scale)
            }
        };
        let bufs: Vec<[&lex_metal::Buffer; 5]> = mats
            .iter()
            .enumerate()
            .map(|(i, (q, s, g))| [xb, q, s, g, if overlap { &ys[i] } else { &y }])
            .collect();
        let steps: Vec<Step<'_>> = (0..reps * MATS)
            .map(|i| (&pipe, &bufs[i % MATS][..], None))
            .collect();
        // One command buffer for all of them, as a decode step issues its
        // work: a command buffer per call leaves the GPU idle between calls
        // and its clock low, and measured this kernel at twice what it costs
        // in the model. Warmed once, then the best of three.
        gpu.run_launches(&steps);
        let t = (0..3)
            .map(|_| gpu.run_launches(&steps).1 / steps.len() as f64)
            .fold(f64::MAX, f64::min);
        println!(
            "{name:<14} {:>9.1} {:>8.1}  {check}",
            1e6 * t,
            bytes / t / 1e9
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("mv_metal needs a Metal device");
}
