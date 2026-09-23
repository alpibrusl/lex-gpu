//! Lower typed lex-front programs to CUDA and write them out, so
//! `scripts/cuda_check.sh` can compile them without a GPU.
//!
//!     cargo run -p lex-msl --example emit_cuda -- out/
//!     scripts/cuda_check.sh out/*.cu
//!
//! These are the same programs the Metal backend lowers, through the same
//! `lower_with`; only the dialect differs.

use lex_front::llama::{QLayout, matvec_q, rmsnorm_rows, silu_mul};
use lex_ir::{DType, Target};
use lex_msl::dialect::Cuda;
use lex_msl::program::lower_with;

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    std::fs::create_dir_all(&dir).map_err(|e| format!("{dir}: {e}"))?;
    let target = Target::nvidia_ada();
    let progs = [
        rmsnorm_rows(4, 4096, 1e-5, None, DType::F32),
        rmsnorm_rows(4, 4096, 1e-5, None, DType::F16),
        silu_mul(4096, 256, DType::F32)?,
        silu_mul(4096, 256, DType::F16)?,
        // The one that matters: NVFP4 dequantisation fused into a matvec.
        // 94% of a decode step is this shape.
        matvec_q(5120, 17408, 8, 5120, QLayout::NVFP4, false)?,
    ];
    for p in &progs {
        let l = lower_with(p, &target, 256, &Cuda)?;
        let path = format!("{dir}/{}.cu", l.entry);
        std::fs::write(&path, &l.source).map_err(|e| format!("{path}: {e}"))?;
        println!("{path}");
    }
    Ok(())
}
