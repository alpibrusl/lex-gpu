//! Inference runtime: load a checkpoint, run decode steps over tile kernels.
//!
//! P2 supports GGUF Llama checkpoints with Q8_0, Q4_K and Q6_K weights —
//! what Ollama serves as `llama3.2:1b` (Q8_0) and `llama3.1:8b` (Q4_K_M). The loader and weight repacking run anywhere; the
//! decode loop needs a Metal device.

pub mod gguf;
pub mod json;
pub mod llama;
pub mod qwen;
pub mod qwen_run;
