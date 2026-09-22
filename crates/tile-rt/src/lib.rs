//! Inference runtime: load a checkpoint, run decode steps over tile kernels.
//!
//! P2 supports GGUF Llama checkpoints with Q8_0 weights — what Ollama serves
//! as `llama3.2:1b`. The loader and weight repacking run anywhere; the
//! decode loop needs a Metal device.

pub mod gguf;
pub mod llama;
