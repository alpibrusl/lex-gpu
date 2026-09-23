//! Compile the emitted CUDA with NVRTC, on a machine with no GPU.
//!
//! NVRTC is a compiler: it needs the toolkit, not a device. That makes this
//! the layer between `scripts/cuda_check.sh` — which runs `nvcc`, a
//! different compiler with different defaults — and the device tests, which
//! need silicon.
//!
//! It exists because the gap between those two cost a rented L4. `nvcc`
//! compiled the emitted source happily; NVRTC rejected it, because NVRTC
//! compiles from a string in memory and has **no include search path at
//! all**, so `#include <cuda_fp16.h>` failed with "no directories in
//! search list". Nothing local could see that, because the only NVRTC call
//! sat behind `Gpu::open()` and therefore behind having a GPU.
//!
//! Splitting `compile_ptx` out from `build_lowered` is what makes this
//! testable at all, and it runs in the same container `cuda_check.sh` uses.

#![cfg(target_os = "linux")]

use lex_cuda::device::{compile_ptx, nvrtc};
use lex_front::llama::{QLayout, matvec_q, rmsnorm_rows, silu_mul};
use lex_ir::{DType, Target};
use lex_msl::dialect::Cuda;
use lex_msl::program::lower_with;

/// Every emitted kernel must compile for a real architecture.
///
/// `sm_89` is an L4. The point is not that architecture in particular, but
/// that NVRTC is asked for a concrete one rather than whatever it defaults
/// to — a default target can accept code a real one rejects.
#[test]
fn every_lowered_program_compiles_with_nvrtc() {
    let Ok(rtc) = nvrtc() else {
        eprintln!("SKIPPED: no libnvrtc (install the CUDA toolkit)");
        return;
    };
    let target = Target::nvidia_ada();
    let progs = [
        rmsnorm_rows(4, 4096, 1e-5, None, DType::F32),
        rmsnorm_rows(4, 4096, 1e-5, None, DType::F16),
        silu_mul(4096, 256, DType::F32).expect("silu f32"),
        silu_mul(4096, 256, DType::F16).expect("silu f16"),
        // The one that matters: 94% of a decode step.
        matvec_q(512, 64, 8, 512, QLayout::NVFP4, false).expect("nvfp4 matvec"),
        matvec_q(4096, 4096, 8, 4096, QLayout::Q4_K, false).expect("q4k matvec"),
    ];
    for p in &progs {
        let l = lower_with(p, &target, 256, &Cuda).expect("lower");
        let ptx = compile_ptx(&rtc, &l.source, &l.entry, "compute_89")
            .unwrap_or_else(|e| panic!("{}", e));
        assert!(!ptx.is_empty(), "{}: empty PTX", l.entry);
        // NVRTC names the entry point in the PTX it emits; if it does not,
        // the module would load and `cuModuleGetFunction` would fail later
        // with nothing to point at.
        let text = String::from_utf8_lossy(&ptx);
        assert!(
            text.contains(&l.entry),
            "{}: entry point missing from PTX",
            l.entry
        );
        eprintln!("{}: {} bytes of PTX", l.entry, ptx.len());
    }
}
