//! Write the emitted CUDA for every P0 kernel, so `scripts/cuda_check.sh`
//! can compile it on a machine with no GPU.
//!
//!     cargo run -p lex-cuda --example emit -- out/
//!     scripts/cuda_check.sh out/*.cu

use lex_cuda::emit;
use lex_ir::{DType, Kernel, Target, plan};

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    std::fs::create_dir_all(&dir).map_err(|e| format!("{dir}: {e}"))?;
    let target = Target::nvidia_ada();
    for k in [
        Kernel::copy(DType::F32, 4096),
        Kernel::copy(DType::F16, 4096),
        Kernel::rmsnorm(DType::F32, 8, 4096, 1e-5),
        Kernel::rmsnorm(DType::F16, 8, 4096, 1e-5),
    ] {
        let p = plan(&k, &target).map_err(|e| e.to_string())?;
        let path = format!("{dir}/{}.cu", k.name);
        std::fs::write(&path, emit(&k, &p, &target)).map_err(|e| format!("{path}: {e}"))?;
        println!("{path}");
    }
    Ok(())
}
