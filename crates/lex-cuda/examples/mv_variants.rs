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
//! - `quant16`   the input to int8, one scale per 16 values -- what the
//!   int8 path adds, once per matvec input.
//! - `int8_row`  one warp a row in integer arithmetic: E2M1 codes to int8
//!   through `__byte_perm` tables, four products a `__dp4a`, against the
//!   quantised input. About 20 instructions per 16 weights where the float
//!   path spends about 100, which is what matters on a card whose clock the
//!   power cap sets.
//!
//! `--reps N` repeats each variant N times over the matrices (default 12).
//! A short burst runs at full clock; a long one reaches the L4's 72 W cap,
//! which is where the model runs and where the float path slows by 1.5x.
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
    // Four E2M1 codes (the low 16 bits) to four int8 values, twice the
    // true value so they are integers: magnitudes {0,1,2,3,4,6,8,12}. One
    // `__byte_perm` looks the magnitudes up in a positive table, one in a
    // negative one -- a selector nibble's low three bits index eight bytes
    // -- and a third picks, per byte, whichever the code's sign bit says.
    let int8 = r#"
__device__ __forceinline__ int e2m1x4(uint codes) {
    const uint idx = codes & 0x7777u;
    const uint pos = __byte_perm(0x03020100u, 0x0C080604u, idx);
    const uint neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, idx);
    return (int)__byte_perm(pos, neg, 0x3210u | ((codes & 0x8888u) >> 1u));
}
"#;
    // The input to int8 with one scale per 16 values: one thread a group.
    let quant16 = format!(
        r#"{pre}
extern "C" __global__ void quant16(const float* __restrict__ x, char* __restrict__ xq,
                                   float* __restrict__ xs) {{
    const uint g = blockIdx.x * 256u + threadIdx.x;
    if (g >= {groups}u) return;
    const float4* xp = reinterpret_cast<const float4*>(x) + g * 4u;
    const float4 v[4] = {{xp[0], xp[1], xp[2], xp[3]}};
    float m = 0.0f;
    for (int i = 0; i < 4; ++i)
        m = fmaxf(m, fmaxf(fmaxf(fabsf(v[i].x), fabsf(v[i].y)), fmaxf(fabsf(v[i].z), fabsf(v[i].w))));
    const float inv = m > 0.0f ? 127.0f / m : 0.0f;
    uint w[4];
    for (int i = 0; i < 4; ++i)
        w[i] = ((uint)__float2int_rn(v[i].x * inv) & 0xFFu)
             | (((uint)__float2int_rn(v[i].y * inv) & 0xFFu) << 8u)
             | (((uint)__float2int_rn(v[i].z * inv) & 0xFFu) << 16u)
             | (((uint)__float2int_rn(v[i].w * inv) & 0xFFu) << 24u);
    reinterpret_cast<uint4*>(xq)[g] = make_uint4(w[0], w[1], w[2], w[3]);
    xs[g] = m / 127.0f;
}}
"#,
        groups = K / 16
    );
    let int8_row = format!(
        r#"{pre}{helpers}{int8}
extern "C" __global__ void int8_row(const char* __restrict__ xq, const float* __restrict__ xs,
                                    const char* __restrict__ q, const char* __restrict__ s,
                                    const float* __restrict__ gs, float* __restrict__ y) {{
    const uint warp = threadIdx.x >> 5u, lane = threadIdx.x & 31u;
    const uint row = blockIdx.x * 8u + warp;
    const uint4* qr = reinterpret_cast<const uint4*>(q + (size_t)row * {kb}u);
    const uchar* sr = reinterpret_cast<const uchar*>(s) + (size_t)row * {ks}u;
    const uint4* xv = reinterpret_cast<const uint4*>(xq);
    float acc = 0.0f;
    #pragma unroll
    for (uint it = 0; it < {iters}u; ++it) {{
        const uint c = it * 32u + lane;          // 16 bytes of the row: 32 values
        const uint4 w = qr[c];
        const uint g = c * 2u;                   // their two groups of 16
        const uint4 a = xv[g], b = xv[g + 1u];   // 16 int8 inputs each
        int d0 = 0, d1 = 0;
        d0 = __dp4a(e2m1x4(w.x), (int)a.x, d0);
        d0 = __dp4a(e2m1x4(w.x >> 16u), (int)a.y, d0);
        d0 = __dp4a(e2m1x4(w.y), (int)a.z, d0);
        d0 = __dp4a(e2m1x4(w.y >> 16u), (int)a.w, d0);
        d1 = __dp4a(e2m1x4(w.z), (int)b.x, d1);
        d1 = __dp4a(e2m1x4(w.z >> 16u), (int)b.y, d1);
        d1 = __dp4a(e2m1x4(w.w), (int)b.z, d1);
        d1 = __dp4a(e2m1x4(w.w >> 16u), (int)b.w, d1);
        acc += (float)d0 * (fp8_e4m3(sr[g]) * xs[g]) + (float)d1 * (fp8_e4m3(sr[g + 1u]) * xs[g + 1u]);
    }}
    acc = warp_sum(acc);
    // Halved: the table holds twice each E2M1 value.
    if (lane == 0u) y[row] = acc * gs[row] * 0.5f;
}}
"#,
        kb = K / 2,
        ks = K / 16,
        iters = K / 1024,
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
    // What each variant binds: the f32 input; the quantiser's in and out;
    // or the quantised input and its scales in place of `x`.
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Float,
        Quant,
        Int8,
    }
    // The int8 path moves the quantised input (1 B a value, and a scale per
    // 16) where the float path moves 4 B a value.
    let bytes8 = bytes - (4 * K) as f64 + (K + 4 * K / 16) as f64;
    let variants: Vec<(&str, Lowered, f64, Kind)> = vec![
        (
            "read",
            hand("read", read, 58 * 16),
            (N * K / 2) as f64,
            Kind::Float,
        ),
        ("emitted", emitted, bytes, Kind::Float),
        ("unroll4", unroll4, bytes, Kind::Float),
        (
            "warp_row",
            hand("warp_row", warp_row("warp_row", false), N / 8),
            bytes,
            Kind::Float,
        ),
        (
            "warp_row_smem",
            hand("warp_row_smem", warp_row("warp_row_smem", true), N / 8),
            bytes,
            Kind::Float,
        ),
        (
            "warp_4rows",
            hand("warp_4rows", warp_4rows, N / 32),
            bytes,
            Kind::Float,
        ),
        (
            "quant16",
            hand("quant16", quant16, (K / 16).div_ceil(256)),
            (4 * K + K + 4 * K / 16) as f64,
            Kind::Quant,
        ),
        (
            "int8_row",
            hand("int8_row", int8_row, N / 8),
            bytes8,
            Kind::Int8,
        ),
    ];

    // `--emit DIR`: write the sources for scripts/cuda_check.sh and stop,
    // so they are compiled on the laptop before a GPU is rented for them.
    let args: Vec<String> = std::env::args().collect();
    // `--mats N`: how many different matrices to cycle. Four (200 MB) keep
    // the 48 MB L2 honest; many more (64 is 3.2 GB) ask whether a working
    // set the size of a model's -- 14.5 GB -- is what slows the kernel
    // there, since in the model it takes half as long again as here.
    let mats_n: usize = args
        .iter()
        .position(|a| a == "--mats")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let reps: usize = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);
    if let Some(i) = args.iter().position(|a| a == "--emit") {
        let dir = args.get(i + 1).ok_or("--emit needs a directory")?;
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        for (name, l, _, _) in &variants {
            std::fs::write(format!("{dir}/mv_{name}.cu"), &l.source).map_err(|e| e.to_string())?;
        }
        return Ok(());
    }
    let gpu = Gpu::open()?;
    println!("{}, {mats_n} matrices", gpu.name());

    // Deterministic, valid bytes: every E2M1 code is a number, and E4M3
    // scales kept clear of NaN (0x7F / 0xFF).
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    let x: Vec<f32> = (0..K)
        .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
        .collect();
    let xb = gpu.upload(&x);
    let mut mats = vec![];
    for _ in 0..mats_n {
        let q: Vec<u8> = (0..N * K / 2).map(|_| next() as u8).collect();
        let s: Vec<u8> = (0..N * K / 16)
            .map(|_| (0x28 + (next() % 24) as u8) | (((next() & 1) as u8) << 7))
            .collect();
        let gs: Vec<f32> = (0..N).map(|_| 1e-3 * (1.0 + (next() % 8) as f32)).collect();
        mats.push((gpu.upload(&q), gpu.upload(&s), gpu.upload(&gs)));
    }
    let y = gpu.zeroed::<f32>(N);
    let xq = gpu.zeroed::<i8>(K);
    let xs = gpu.zeroed::<f32>(K / 16);

    let mut reference: Option<Vec<f32>> = None;
    println!("{:<14} {:>9} {:>8}  check", "variant", "us/call", "GB/s");
    for (name, lowered, moved, kind) in &variants {
        let pipe = gpu
            .build_lowered(lowered)
            .map_err(|e| format!("{name}: {e}"))?;
        // Once on matrix 0 for the answer, then timed over all of them.
        let (q0, s0, g0) = &mats[0];
        match kind {
            Kind::Float => gpu.run(&pipe, &[&xb, q0, s0, g0, &y])?,
            // Leaves the quantised input in place for `int8_row`, which
            // comes after it in the list.
            Kind::Quant => gpu.run(&pipe, &[&xb, &xq, &xs])?,
            Kind::Int8 => gpu.run(&pipe, &[&xq, &xs, q0, s0, g0, &y])?,
        }
        let mut got = vec![0.0f32; N];
        gpu.download(&y, &mut got);
        let check = match (&reference, *name) {
            (_, "read") | (_, "quant16") => "-".to_string(),
            (None, _) => {
                reference = Some(got);
                "reference".to_string()
            }
            (Some(want), _) => {
                let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
                let err = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                format!("{:.1e} of scale", err / scale)
            }
        };
        let bufs: Vec<Vec<&lex_cuda::device::Buffer>> = mats
            .iter()
            .map(|(q, s, g)| match kind {
                Kind::Float => vec![&xb, q, s, g, &y],
                Kind::Quant => vec![&xb, &xq, &xs],
                Kind::Int8 => vec![&xq, &xs, q, s, g, &y],
            })
            .collect();
        let steps: Vec<Step<'_>> = (0..reps * mats_n)
            .map(|i| (&pipe, &bufs[i % mats_n][..], None))
            .collect();
        let times = gpu.run_each_timed(&steps);
        // The first round warms the pipeline and the TLB; not counted.
        let t: f64 = times[mats_n..].iter().sum::<f64>() / (times.len() - mats_n) as f64;
        println!(
            "{name:<14} {:>9.1} {:>8.1}  {check}",
            1e6 * t,
            moved / t / 1e9
        );
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("mv_variants needs an NVIDIA GPU (Linux)");
}
