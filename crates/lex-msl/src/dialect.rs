//! The spellings a GPU language uses for the things every backend needs.
//!
//! The lowering in [`crate::program`] is 1,849 lines, of which about 51
//! mention anything Metal-specific: barriers, lane shuffles, address-space
//! qualifiers, the entry-point signature. Everything else is walking the
//! typed IR and deciding what to emit, which is the same decision on any
//! target.
//!
//! So a second backend copies 1,798 lines to vary 51 — and this repository
//! has already paid for what that costs. The batched matmul kept decoding
//! NVFP4 with a `simd_shuffle` table that the single-row matvec had
//! abandoned two rounds of work earlier, because they were separate arms of
//! the same emitter and only the one being benchmarked got fixed. That cost
//! a 2.7x regression that three rounds of measurement misattributed to
//! arithmetic. Duplicating the whole lowering would make that failure mode
//! structural, across targets, forever.
//!
//! Hence a seam instead. `Dialect` is the part that differs; the lowering is
//! the part that does not. The golden files are the proof that moving code
//! behind this trait changed nothing: they are byte-identical across the
//! refactor, or the refactor is wrong.

use lex_ir::DType;

/// One buffer the kernel takes, already named and typed by the lowering.
pub struct Param<'a> {
    pub ty: &'a str,
    pub name: &'a str,
    pub writable: bool,
}

/// How a target spells the primitives the lowering emits.
///
/// Every method returns text rather than writing, so a dialect stays a pure
/// description with no opinion about how the emitter buffers its output.
pub trait Dialect {
    /// Wait for every thread in the group, and make threadgroup writes
    /// visible to them.
    fn barrier(&self) -> String;

    /// Read lane `src`'s copy of `value`. `src` need not be uniform: the
    /// NVFP4 decode uses a divergent index to gather from a lane-held table.
    fn shuffle(&self, value: &str, src: &str) -> String;

    /// Read the value held `delta` lanes further up, for a reduction that
    /// halves `delta` each step.
    ///
    /// Metal's `simd_shuffle_down` leaves lanes near the top reading past
    /// the end, which is fine when only lane 0 is used. A dialect whose
    /// primitive differs in that respect has to say so here rather than let
    /// the lowering assume Metal's.
    fn shuffle_down(&self, value: &str, delta: &str) -> String;

    /// Scalar type name.
    fn scalar(&self, d: DType) -> &'static str;

    /// A pointer into threadgroup / shared memory.
    ///
    /// Metal puts the address space in the *type*, so a pointer to shared
    /// memory is a different type from a pointer to device memory and the
    /// compiler enforces it. CUDA puts it on the *declaration* and the
    /// pointer is plain, so the same mistake compiles. A dialect cannot fix
    /// that, but the lowering should not have to know which world it is in.
    fn shared_ptr(&self, ty: &str) -> String;

    /// Declare an array in threadgroup / shared memory.
    fn shared_array(&self, ty: &str, name: &str, len: usize) -> String;

    /// `exp`, `log` and `rsqrt`, at the accuracy the reference interpreter
    /// is written against.
    ///
    /// Metal has an explicit `precise::` namespace because its default
    /// versions are allowed to be sloppier; CUDA's unsuffixed `expf` and
    /// `rsqrtf` already meet the accuracy the interpreter assumes, and its
    /// fast versions are the ones that need asking for (`__expf`). So the
    /// *default* is the dangerous one in Metal and the safe one in CUDA,
    /// which is a good reason for neither spelling to appear in the
    /// lowering.
    fn exp(&self, x: &str) -> String;
    fn log(&self, x: &str) -> String;
    fn rsqrt(&self, x: &str) -> String;
    /// Absolute value of a float.
    fn fabs(&self, x: &str) -> String;
    /// Larger of two floats.
    fn fmax(&self, a: &str, b: &str) -> String;

    /// Convert `expr` to `d`.
    ///
    /// Metal spells every narrowing as a functional cast, so `half(x)` is
    /// the whole story. CUDA's `__half` has no such constructor from float
    /// in device code, and wants `__float2half`. Same operation, different
    /// grammar -- which is exactly the kind of thing that compiles as a
    /// silent no-op if a backend is copied and not read.
    fn convert(&self, d: DType, expr: &str) -> String;

    /// Whatever has to precede the kernel: includes, and any typedef the
    /// emitted body assumes.
    fn includes(&self) -> String;

    /// The NVFP4 decode helpers, when a kernel dequantises four-bit
    /// weights.
    ///
    /// The bit trick inside `fp4_pair` is the same on both targets and has
    /// to be: dropping an E2M1 code's magnitude bits at the bottom of
    /// half's exponent field gives back the value exactly, times 2^-14,
    /// subnormals included, because the two formats' denormal boundaries
    /// coincide there. That is a property of IEEE half, not of Metal. Only
    /// the spelling of the reinterpretation differs.
    fn fp4_preamble(&self) -> String;

    /// The entry point, from its name through the opening brace and the
    /// declarations that tell the body where this thread is.
    ///
    /// This is the one place the two languages disagree about *shape* and
    /// not merely spelling, so it is a whole-prologue method rather than a
    /// set of substitutions. Metal declares the thread and threadgroup
    /// indices as parameters carrying attributes, and binds buffers by an
    /// explicit index. CUDA takes buffers positionally -- there is no index
    /// to get wrong, and equally none to get right -- and reads the indices
    /// from builtins inside the body. The body that follows is identical
    /// either way, which is the point.
    fn entry(&self, name: &str, params: &[Param<'_>], scalars: bool) -> String;
}

/// Metal Shading Language.
#[derive(Clone, Copy, Debug, Default)]
pub struct Msl;

impl Dialect for Msl {
    fn barrier(&self) -> String {
        "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string()
    }

    fn shuffle(&self, value: &str, src: &str) -> String {
        format!("simd_shuffle({value}, {src})")
    }

    fn shuffle_down(&self, value: &str, delta: &str) -> String {
        format!("simd_shuffle_down({value}, {delta})")
    }

    fn scalar(&self, d: DType) -> &'static str {
        d.msl_scalar()
    }

    fn shared_ptr(&self, ty: &str) -> String {
        format!("threadgroup {ty}*")
    }

    fn shared_array(&self, ty: &str, name: &str, len: usize) -> String {
        format!("threadgroup {ty} {name}[{len}];")
    }

    fn exp(&self, x: &str) -> String {
        format!("precise::exp({x})")
    }

    fn log(&self, x: &str) -> String {
        format!("precise::log({x})")
    }

    fn rsqrt(&self, x: &str) -> String {
        format!("precise::rsqrt({x})")
    }

    fn fabs(&self, x: &str) -> String {
        format!("fabs({x})")
    }

    fn fmax(&self, a: &str, b: &str) -> String {
        format!("max({a}, {b})")
    }

    fn convert(&self, d: DType, expr: &str) -> String {
        format!("{}({expr})", d.msl_scalar())
    }

    fn includes(&self) -> String {
        "\n#include <metal_stdlib>\nusing namespace metal;\n\n".to_string()
    }

    fn fp4_preamble(&self) -> String {
        FP4_TABLES_MSL.to_string()
    }

    fn entry(&self, name: &str, params: &[Param<'_>], scalars: bool) -> String {
        let mut s = format!("kernel void {name}(\n");
        for (i, p) in params.iter().enumerate() {
            let cv = if p.writable { "" } else { "const " };
            s.push_str(&format!(
                "    device {cv}{}* {} [[buffer({i})]],\n",
                p.ty, p.name
            ));
        }
        if scalars {
            s.push_str(&format!(
                "    constant uint* scalars [[buffer({})]],\n",
                params.len()
            ));
        }
        s.push_str("    uint tid [[thread_index_in_threadgroup]],\n");
        s.push_str("    uint3 tgpos [[threadgroup_position_in_grid]])\n{\n");
        s.push_str("    const uint gid = tgpos.x, gid2 = tgpos.y;\n");
        s
    }
}

/// CUDA C.
///
/// The mask is `0xffffffff` throughout because the lowering only shuffles
/// where every lane of the warp reaches the call — the reductions run
/// outside divergent control flow, and the NVFP4 gather is uniform in
/// *reaching* the shuffle even when its index is not. A dialect for a
/// lowering that shuffled under divergence would need the mask threaded
/// through, and `__shfl_sync` would deadlock rather than quietly misbehave.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cuda;

impl Dialect for Cuda {
    fn barrier(&self) -> String {
        "__syncthreads();".to_string()
    }

    fn shuffle(&self, value: &str, src: &str) -> String {
        format!("__shfl_sync(0xffffffffu, {value}, {src})")
    }

    fn shuffle_down(&self, value: &str, delta: &str) -> String {
        format!("__shfl_down_sync(0xffffffffu, {value}, {delta})")
    }

    fn scalar(&self, d: DType) -> &'static str {
        match d {
            DType::F16 => "__half",
            DType::F32 => "float",
            DType::I8 => "char",
        }
    }

    fn shared_ptr(&self, ty: &str) -> String {
        format!("{ty}*")
    }

    fn shared_array(&self, ty: &str, name: &str, len: usize) -> String {
        format!("__shared__ {ty} {name}[{len}];")
    }

    fn exp(&self, x: &str) -> String {
        format!("expf({x})")
    }

    fn log(&self, x: &str) -> String {
        format!("logf({x})")
    }

    fn rsqrt(&self, x: &str) -> String {
        format!("rsqrtf({x})")
    }

    fn fabs(&self, x: &str) -> String {
        format!("fabsf({x})")
    }

    fn fmax(&self, a: &str, b: &str) -> String {
        format!("fmaxf({a}, {b})")
    }

    fn convert(&self, d: DType, expr: &str) -> String {
        match d {
            DType::F16 => format!("__float2half({expr})"),
            DType::F32 => format!("float({expr})"),
            DType::I8 => format!("char({expr})"),
        }
    }

    /// `uint` and `uchar` are Metal spellings the lowering uses throughout
    /// its index arithmetic. Typedefs here are worth far more than editing
    /// several hundred body sites, and they keep the emitted CUDA readable
    /// next to the emitted MSL when the two are diffed.
    fn includes(&self) -> String {
        "\n#include <cuda_fp16.h>\n\ntypedef unsigned int uint;\ntypedef unsigned char uchar;\n\n"
            .to_string()
    }

    fn fp4_preamble(&self) -> String {
        FP4_TABLES_CUDA.to_string()
    }

    fn entry(&self, name: &str, params: &[Param<'_>], scalars: bool) -> String {
        let mut s = format!("extern \"C\" __global__ void {name}(\n");
        for p in params {
            let cv = if p.writable { "" } else { "const " };
            s.push_str(&format!("    {cv}{}* __restrict__ {},\n", p.ty, p.name));
        }
        if scalars {
            s.push_str("    const uint* __restrict__ scalars,\n");
        }
        // Trailing comma: Metal ends its list with the index parameters,
        // CUDA has none to end with.
        if s.ends_with(",\n") {
            s.truncate(s.len() - 2);
            s.push('\n');
        }
        s.push_str(")\n{\n");
        s.push_str("    const uint tid = threadIdx.x;\n");
        s.push_str("    const uint gid = blockIdx.x, gid2 = blockIdx.y;\n");
        s
    }
}

const FP4_TABLES_MSL: &str = concat!(
    "constant float FP4_V[16] = {\n",
    "    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n",
    "    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};\n",
    "inline float2 fp4_pair(uint b) {\n",
    "    const uint w = b | (b << 12u);\n",
    "    return float2(as_type<half2>(((w & 0x00070007u) << 9u)\n",
    "                               | ((w & 0x00080008u) << 12u)));\n",
    "}\n",
    "inline float fp8_e4m3(uint b) {\n",
    "    const uint e = (b >> 3u) & 0xFu, m = b & 7u;\n",
    "    const uint bits = select(((e + 120u) << 23u) | (m << 20u),\n",
    "                             as_type<uint>(float(m) * 0.001953125f), e == 0u);\n",
    "    return as_type<float>(bits | ((b & 0x80u) << 24u));\n",
    "}\n\n"
);

/// The same decode in CUDA.
///
/// `as_type` has no CUDA spelling, and a pointer cast between types of the
/// same size is undefined behaviour that nvcc is entitled to miscompile.
/// `memcpy` of four bytes is the portable reinterpretation and lowers to
/// nothing; this repository has a measurement habit and `ptxas -v` reports
/// no extra registers for it.
///
/// MSL's `select(a, b, c)` is `c ? b : a` -- the condition is *last* and
/// the arms read backwards from a C ternary. Transcribing it in order is a
/// mistake that produces plausible-looking weights, so the ternary here is
/// written out rather than mirrored.
const FP4_TABLES_CUDA: &str = concat!(
    "__constant__ float FP4_V[16] = {\n",
    "    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,\n",
    "    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};\n",
    "__device__ __forceinline__ float2 fp4_pair(uint b) {\n",
    "    const uint w = b | (b << 12u);\n",
    "    const uint bits = ((w & 0x00070007u) << 9u)\n",
    "                    | ((w & 0x00080008u) << 12u);\n",
    "    __half2 h;\n",
    "    memcpy(&h, &bits, sizeof(h));\n",
    "    return __half22float2(h);\n",
    "}\n",
    "__device__ __forceinline__ float fp8_e4m3(uint b) {\n",
    "    const uint e = (b >> 3u) & 0xFu, m = b & 7u;\n",
    "    const float sub = float(m) * 0.001953125f;\n",
    "    uint subbits;\n",
    "    memcpy(&subbits, &sub, sizeof(subbits));\n",
    "    const uint bits = (e == 0u) ? subbits : (((e + 120u) << 23u) | (m << 20u));\n",
    "    const uint out = bits | ((b & 0x80u) << 24u);\n",
    "    float r;\n",
    "    memcpy(&r, &out, sizeof(r));\n",
    "    return r;\n",
    "}\n\n"
);

#[cfg(test)]
mod tests {
    use super::*;

    /// The two dialects must differ in every method. A `Dialect` whose
    /// implementation was copied and not edited would pass every other test
    /// in this repository.
    #[test]
    fn the_dialects_actually_differ() {
        let (m, c) = (Msl, Cuda);
        assert_ne!(m.barrier(), c.barrier());
        assert_ne!(m.shuffle("v", "i"), c.shuffle("v", "i"));
        assert_ne!(m.shuffle_down("v", "d"), c.shuffle_down("v", "d"));
        assert_ne!(m.scalar(DType::F16), c.scalar(DType::F16));
        assert_ne!(m.shared_ptr("float"), c.shared_ptr("float"));
        assert_ne!(
            m.shared_array("float", "s", 8),
            c.shared_array("float", "s", 8)
        );
    }

    #[test]
    fn operands_reach_the_emitted_text() {
        assert!(Cuda.shuffle_down("s[0][1]", "d").contains("s[0][1]"));
        assert!(Msl.shuffle("fp4_lane", "code").contains("fp4_lane"));
    }
}
