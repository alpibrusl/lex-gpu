//! Write the CUDA form of the prefill GEMM at Qwen3.8's shapes, so
//! `scripts/cuda_check.sh` can compile it -- nvcc, ptxas and NVRTC -- on a
//! machine with no NVIDIA GPU.
//!
//!     cargo run -p lex-msl --example emit_gemm -- out/
//!     scripts/cuda_check.sh out/gemm_*.cu

use lex_msl::gemm::{Backend, Gemm, gemm_nvfp4};

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).ok_or("usage: emit_gemm <dir>")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    // gate/up, down (residual), the a/b projection's ragged 48 rows, and a
    // chunk of 32 tokens, which takes the half-height tile.
    for g in [
        Gemm {
            m: 64,
            n: 17408,
            k: 5120,
            residual: false,
            x_half: true,
        },
        Gemm {
            m: 64,
            n: 5120,
            k: 17408,
            residual: true,
            x_half: false,
        },
        Gemm {
            m: 64,
            n: 48,
            k: 5120,
            residual: false,
            x_half: true,
        },
        Gemm {
            m: 32,
            n: 17408,
            k: 5120,
            residual: false,
            x_half: true,
        },
    ] {
        let l = gemm_nvfp4(&g, Backend::Cuda)?;
        std::fs::write(format!("{dir}/{}.cu", l.entry), &l.source).map_err(|e| e.to_string())?;
        println!(
            "{}: {} x {} blocks, {} B shared",
            l.entry, l.grid, l.grid2, l.threadgroup_bytes
        );
    }
    Ok(())
}
