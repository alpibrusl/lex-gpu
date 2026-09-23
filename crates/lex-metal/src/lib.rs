//! Device glue for the Metal backend.
//!
//! This is the only crate in the workspace that cannot be built or tested off a
//! Mac. Everything above it -- the IR, the planner, the MSL emitter, the
//! reference interpreter -- is ordinary Rust with golden-file tests, and runs
//! in CI on any host. Keeping that boundary sharp is what makes it possible to
//! develop the compiler somewhere other than the machine that runs the kernels.

#[cfg(target_os = "macos")]
mod device;

#[cfg(target_os = "macos")]
pub use device::{DeviceInfo, Gpu, Pipeline, Step};

/// Re-exported so callers do not need a direct `metal` dependency. A later
/// runtime hands out typed tile handles instead of raw buffers.
#[cfg(target_os = "macos")]
pub use metal::Buffer;

/// True when this build can actually reach a GPU.
pub const fn available() -> bool {
    cfg!(target_os = "macos")
}
