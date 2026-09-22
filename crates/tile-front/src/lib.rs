//! The typed half of the tile language: programs over linear tiles, the
//! checker that enforces linearity, effects, bounds and the target budget,
//! and a reference interpreter.
//!
//! P1's question is whether linear tiles can express the kernels inference
//! actually needs without fighting the author. [`flash`] is the stress test:
//! flash-attention decode with K/V reused across query blocks and copies
//! pipelined `stages` deep.
//!
//! ```text
//!   Builder   (embedded DSL)  -> Program
//!   check     (Program, Target) -> Report | [Diag]
//!   interp    (Program, tensors) -> tensors
//! ```

pub mod check;
pub mod flash;
pub mod interp;
pub mod ir;
pub mod llama;
pub mod print;

pub use check::{Diag, Kind, Report, check};
pub use interp::{Tensor, run};
pub use ir::{Builder, Program};
