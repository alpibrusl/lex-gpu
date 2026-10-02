//! Write the CUDA causal attention kernel's source for `scripts/cuda_check.sh`
//! (nvcc and NVRTC, no GPU):
//!
//!     cargo run -p lex-msl --example emit_attn -- out/ && scripts/cuda_check.sh out/*.cu

use lex_msl::attn::{Causal, causal_wmma};

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).ok_or("usage: emit_attn <dir>")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut n = 0;
    // Qwen3.8-27B's shape at the chunk sizes, and the smaller ones a test uses.
    for (tokens, kv_heads, group, head_dim, cap) in [
        (512usize, 4usize, 6usize, 256usize, 65536usize),
        (64, 4, 6, 256, 65536),
        (37, 2, 6, 64, 256),
        (48, 2, 4, 128, 256),
    ] {
        let l = causal_wmma(&Causal {
            tokens,
            kv_heads,
            group,
            head_dim,
            cap,
        })?;
        std::fs::write(format!("{dir}/{}.cu", l.entry), &l.source).map_err(|e| e.to_string())?;
        n += 1;
    }
    println!("{n} kernels in {dir}");
    Ok(())
}
