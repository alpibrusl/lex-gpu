//! Write the padded-prefill kernels' CUDA source for `scripts/cuda_check.sh`
//! (nvcc and NVRTC, no GPU):
//!
//!     cargo run -p lex-msl --example emit_pad -- out/ && scripts/cuda_check.sh out/*.cu

use lex_msl::pad::{ab_rows, conv_restore};

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).ok_or("usage: emit_pad <dir>")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let mut n = 0;
    for tokens in [16usize, 32, 64, 128, 256, 512] {
        for x_half in [true, false] {
            let l = ab_rows(tokens, 5120, 48, x_half)?;
            std::fs::write(format!("{dir}/{}.cu", l.entry), &l.source)
                .map_err(|e| e.to_string())?;
            n += 1;
        }
        let l = conv_restore(tokens, 10240, 3)?;
        std::fs::write(format!("{dir}/{}.cu", l.entry), &l.source).map_err(|e| e.to_string())?;
        n += 1;
    }
    println!("{n} kernels in {dir}");
    Ok(())
}
