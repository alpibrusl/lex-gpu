//! Which NVFP4 matvec reads fastest on this card -- hand-written variants
//! against the one the emitter writes, at Qwen3.8's gate/up shape.
//!
//!     cargo run --release -p lex-cuda --example mv_variants
//!
//! The emitted decode matvec reads ~143 GB/s on an L4 of ~300, and making
//! its loads wide changed nothing (351.9 us a call against 350.5), so the
//! load instruction count was not what limits it. Rather than one cloud
//! round trip per guess, this times several structures in one run:
//!
//! - `read`      a plain read of the weight bytes: what the card delivers.
//! - `emitted`   the emitter's kernel (16 lanes a row, 16 values a run).
//! - `unroll4`   the same with its main loop unrolled by four.
//! - `warp_row`  one warp a row, 16 bytes (32 values) a lane a step.
//! - `warp_row_smem` the same with the input staged in shared memory.
//! - `warp_4rows` one warp for four rows, each input loaded once for all.
//!
//! Four different matrices are cycled so the 48 MB L2 cannot hold the
//! 50 MB one being read -- repeating one matrix flattered an earlier sweep.
//! Every variant's output is checked against the emitted kernel's.

#[cfg(target_os = "linux")]
fn main() -> Result<(), String> {
    use lex_cuda::device::{Gpu, Step};
    use lex_front::llama::{QLayout, matvec_q};
    use lex_ir::Target;
    use lex_msl::dialect::{Cuda, Dialect};
    use lex_msl::program::{Lowered, lower_with};

    const K: usize = 5120;
    const N: usize = 17408;
    const MATS: usize = 4;
    const REPS: usize = 12;
    // What one call moves: codes, scales, row scales, the input.
    let bytes = (N * K / 2 + N * K / 16 + 4 * N + 4 * K) as f64;

    let target = Target::nvidia_ada();
    let emitted = lower_with(
        &matvec_q(K, N, 16, K, QLayout::NVFP4, false)?,
        &target,
        256,
        &Cuda,
    )?;
    let unroll4 = Lowered {
        entry: emitted.entry.clone(),
        source: emitted.source.replacen(
            "for (uint p0 = lane * 16u;",
            "#pragma unroll 4\n                for (uint p0 = lane * 16u;",
            1,
        ),
        ..emitted.clone()
    };
    if unroll4.source == emitted.source {
        return Err("the unroll variant did not change the emitted source".into());
    }

    let pre = format!("{}{}", Cuda.includes(), Cuda.fp4_preamble());
    // Sixteen values from two words and four float4s of input.
    let helpers = r#"
__device__ __forceinline__ float dot16(uint a, uint b, float4 x0, float4 x1, float4 x2, float4 x3) {
    float r = 0.0f; float2 w;
    w = fp4_pair(a & 0xFFu);         r += x0.x * w.x + x0.y * w.y;
    w = fp4_pair((a >> 8u) & 0xFFu); r += x0.z * w.x + x0.w * w.y;
    w = fp4_pair((a >> 16u) & 0xFFu);r += x1.x * w.x + x1.y * w.y;
    w = fp4_pair(a >> 24u);          r += x1.z * w.x + x1.w * w.y;
    w = fp4_pair(b & 0xFFu);         r += x2.x * w.x + x2.y * w.y;
    w = fp4_pair((b >> 8u) & 0xFFu); r += x2.z * w.x + x2.w * w.y;
    w = fp4_pair((b >> 16u) & 0xFFu);r += x3.x * w.x + x3.y * w.y;
    w = fp4_pair(b >> 24u);          r += x3.z * w.x + x3.w * w.y;
    return r;
}
__device__ __forceinline__ float warp_sum(float v) {
    for (uint d = 16u; d > 0u; d >>= 1u) v += __shfl_down_sync(0xffffffffu, v, d);
    return v;
}
"#;
    let sig = "(const float* __restrict__ x, const char* __restrict__ q, \
               const char* __restrict__ s, const float* __restrict__ gs, float* __restrict__ y)";
    let warp_row = |name: &str, smem: bool| -> String {
        let (stage, xsrc) = if smem {
            (
                format!(
                    "__shared__ float4 xs[{k4}];\n    \
                     for (uint i = threadIdx.x; i < {k4}u; i += 256u) \
                     xs[i] = reinterpret_cast<const float4*>(x)[i];\n    __syncthreads();",
                    k4 = K / 4
                ),
                "xs",
            )
        } else {
            (String::new(), "reinterpret_cast<const float4*>(x)")
        };
        format!(
            r#"{pre}{helpers}
extern "C" __global__ void {name}{sig} {{
    {stage}
    const uint warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint row = blockIdx.x * 8u + warp;
    const uint4* qr = reinterpret_cast<const uint4*>(q + (size_t)row * {kb}u);
    const uchar* sr = reinterpret_cast<const uchar*>(s) + (size_t)row * {ks}u;
    const float4* xv = {xsrc};
    float acc = 0.0f;
    #pragma unroll
    for (uint it = 0; it < {iters}u; ++it) {{
        const uint c = it * 32u + lane;
        const uint4 w = qr[c];
        const uint p = c * 32u;
        const float s0 = fp8_e4m3(sr[p / 16u]), s1 = fp8_e4m3(sr[p / 16u + 1u]);
        const float4* xp = xv + p / 4u;
        acc += s0 * dot16(w.x, w.y, xp[0], xp[1], xp[2], xp[3])
             + s1 * dot16(w.z, w.w, xp[4], xp[5], xp[6], xp[7]);
    }}
    acc = warp_sum(acc);
    if (lane == 0u) y[row] = acc * gs[row] * 16384.0f;
}}
"#,
            kb = K / 2,
            ks = K / 16,
            iters = K / 1024,
        )
    };
    let warp_4rows = format!(
        r#"{pre}{helpers}
extern "C" __global__ void warp_4rows{sig} {{
    const uint warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint row0 = blockIdx.x * 32u + warp * 4u;
    const float4* xv = reinterpret_cast<const float4*>(x);
    float acc[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    #pragma unroll
    for (uint it = 0; it < {iters}u; ++it) {{
        const uint c = it * 32u + lane;
        const uint p = c * 32u;
        const float4* xp = xv + p / 4u;
        const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3],
                     x4 = xp[4], x5 = xp[5], x6 = xp[6], x7 = xp[7];
        #pragma unroll
        for (uint r = 0; r < 4u; ++r) {{
            const size_t row = row0 + r;
            const uint4 w = reinterpret_cast<const uint4*>(q + row * {kb}u)[c];
            const uchar* sr = reinterpret_cast<const uchar*>(s) + row * {ks}u;
            const float s0 = fp8_e4m3(sr[p / 16u]), s1 = fp8_e4m3(sr[p / 16u + 1u]);
            acc[r] += s0 * dot16(w.x, w.y, x0, x1, x2, x3) + s1 * dot16(w.z, w.w, x4, x5, x6, x7);
        }}
    }}
    #pragma unroll
    for (uint r = 0; r < 4u; ++r) {{
        const float v = warp_sum(acc[r]);
        if (lane == 0u) y[row0 + r] = v * gs[row0 + r] * 16384.0f;
    }}
}}
"#,
        kb = K / 2,
        ks = K / 16,
        iters = K / 1024,
    );
    let read = format!(
        r#"{pre}
extern "C" __global__ void read{sig} {{
    const uint4* v = reinterpret_cast<const uint4*>(q);
    uint acc = 0u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < {n16}ull;
         i += (size_t)gridDim.x * blockDim.x) {{
        const uint4 w = v[i];
        acc ^= w.x ^ w.y ^ w.z ^ w.w;
    }}
    if (acc == 0x9E3779B9u) y[0] = 1.0f;
}}
"#,
        n16 = N * K / 2 / 16,
    );
    let hand = |entry: &str, source: String, grid: usize| Lowered {
        entry: entry.into(),
        source,
        grid,
        grid2: 1,
        threads: 256,
        threadgroup_bytes: 0,
        arena_bytes: 0,
        scratch_bytes: 0,
        barriers: 0,
        writes: vec![false, false, false, false, true],
    };
    let variants: Vec<(&str, Lowered, f64)> = vec![
        ("read", hand("read", read, 58 * 16), (N * K / 2) as f64),
        ("emitted", emitted, bytes),
        ("unroll4", unroll4, bytes),
        ("warp_row", hand("warp_row", warp_row("warp_row", false), N / 8), bytes),
        ("warp_row_smem", hand("warp_row_smem", warp_row("warp_row_smem", true), N / 8), bytes),
        ("warp_4rows", hand("warp_4rows", warp_4rows, N / 32), bytes),
    ];

    // `--emit DIR`: write the sources for scripts/cuda_check.sh and stop,
    // so they are compiled on the laptop before a GPU is rented for them.
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--emit") {
        let dir = args.get(i + 1).ok_or("--emit needs a directory")?;
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        for (name, l, _) in &variants {
            std::fs::write(format!("{dir}/mv_{name}.cu"), &l.source).map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    let gpu = Gpu::open()?;
    println!("{}", gpu.name());

    // Deterministic, valid bytes: every E2M1 code is a number, and E4M3
    // scales kept clear of NaN (0x7F / 0xFF).
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let x: Vec<f32> = (0..K).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let xb = gpu.upload(&x);
    let mut mats = vec![];
    for _ in 0..MATS {
        let q: Vec<u8> = (0..N * K / 2).map(|_| next() as u8).collect();
        let s: Vec<u8> = (0..N * K / 16)
            .map(|_| (0x28 + (next() % 24) as u8) | (((next() & 1) as u8) << 7))
            .collect();
        let gs: Vec<f32> = (0..N).map(|_| 1e-3 * (1.0 + (next() % 8) as f32)).collect();
        mats.push((gpu.upload(&q), gpu.upload(&s), gpu.upload(&gs)));
    }
    let y = gpu.zeroed::<f32>(N);

    let mut reference: Option<Vec<f32>> = None;
    println!("{:<14} {:>9} {:>8}  check", "variant", "us/call", "GB/s");
    for (name, lowered, moved) in &variants {
        let pipe = gpu.build_lowered(lowered).map_err(|e| format!("{name}: {e}"))?;
        // Once on matrix 0 for the answer, then timed over all four.
        let (q0, s0, g0) = &mats[0];
        gpu.run(&pipe, &[&xb, q0, s0, g0, &y])?;
        let mut got = vec![0.0f32; N];
        gpu.download(&y, &mut got);
        let check = match (&reference, *name) {
            (_, "read") => "-".to_string(),
            (None, _) => {
                reference = Some(got);
                "reference".to_string()
            }
            (Some(want), _) => {
                let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
                let err = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                format!("{:.1e} of scale", err / scale)
            }
        };
        let bufs: Vec<[&lex_cuda::device::Buffer; 5]> =
            mats.iter().map(|(q, s, g)| [&xb, q, s, g, &y]).collect();
        let steps: Vec<Step<'_>> = (0..REPS * MATS)
            .map(|i| (&pipe, &bufs[i % MATS][..], None))
            .collect();
        let times = gpu.run_each_timed(&steps);
        // The first round warms the pipeline and the TLB; not counted.
        let t: f64 = times[MATS..].iter().sum::<f64>() / (times.len() - MATS) as f64;
        println!("{name:<14} {:>9.1} {:>8.1}  {check}", 1e6 * t, moved / t / 1e9);
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("mv_variants needs an NVIDIA GPU (Linux)");
}
