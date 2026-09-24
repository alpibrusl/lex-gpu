//! The GPU this build talks to.
//!
//! The runtime is written against one device API and compiled against
//! whichever backend the host has: Metal on macOS, CUDA on Linux. The two
//! offer the same surface deliberately -- see `lex_cuda::device`, which
//! documents where their *semantics* differ even though their signatures
//! do not.
//!
//! A trait would let one binary hold both, which is worth nothing here: no
//! machine has a Metal device and an NVIDIA one, so the choice is made at
//! compile time and costs no dispatch. What this does buy is the claim the
//! project is actually making -- that the same `lex-front` programs run on
//! both -- being checked by the compiler rather than asserted in a README.

#[cfg(target_os = "macos")]
pub use lex_metal::{Buffer, DeviceInfo, Gpu, Pipeline, Step};

#[cfg(target_os = "linux")]
pub use lex_cuda::device::{Buffer, DeviceInfo, Gpu, Pipeline, Step};

/// The dialect the emitter lowers through for this build.
///
/// Every program in the runtime goes through here. It is easy to miss
/// because nothing type-checks it: `lower` defaults to MSL, and a runtime
/// that kept the default would compile perfectly well against the CUDA
/// device and then hand Metal source to NVRTC at load time. The failure
/// arrives on a rented GPU rather than on the laptop, which is the worst
/// place for it, so the choice lives beside the device it belongs to.
pub fn dialect() -> &'static dyn lex_msl::dialect::Dialect {
    #[cfg(target_os = "macos")]
    {
        &lex_msl::dialect::Msl
    }
    #[cfg(target_os = "linux")]
    {
        &lex_msl::dialect::Cuda
    }
}

/// Write the CUDA form of every program the runtime compiles, when
/// `LEX_DUMP_CUDA` names a directory.
///
/// `examples/emit_cuda` does the same job from a list of kernels written
/// out by hand, and says in its own docstring that a kernel added to the
/// runtime and not to the list is simply not covered. This cannot drift,
/// because the runtime building the model is what feeds it:
///
/// ```text
/// LEX_DUMP_CUDA=out cargo run --release -p lex-rt --example qwen -- --steps 2
/// scripts/cuda_check.sh out/*.cu
/// ```
///
/// Run on a Mac it emits CUDA for a Metal run, which is the point: the
/// question is whether the *other* backend would accept these programs,
/// and that is answerable without one. A program that will not lower at
/// all leaves a `.err` file rather than passing quietly.
pub fn dump_cuda(prog: &lex_front::Program, threads: usize) {
    let Some(dir) = std::env::var_os("LEX_DUMP_CUDA") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    // Ada rather than the host's target: a threadgroup budget of 32 KiB
    // would reject schedules the machine being checked for allows.
    let target = lex_ir::Target::nvidia_ada();
    match lex_msl::program::lower_with(prog, &target, threads, &lex_msl::dialect::Cuda) {
        Ok(l) => {
            let _ = std::fs::write(dir.join(format!("{}.cu", l.entry)), &l.source);
        }
        Err(e) => {
            eprintln!("LEX_DUMP_CUDA: `{}` does not lower for CUDA: {e}", prog.name);
            let _ = std::fs::write(dir.join(format!("{}.err", prog.name)), e);
        }
    }
}
