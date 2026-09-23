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
