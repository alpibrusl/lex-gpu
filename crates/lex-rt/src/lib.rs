//! Inference runtime: load a checkpoint, run decode steps over tile kernels.
//!
//! P2 supports GGUF Llama checkpoints with Q8_0, Q4_K and Q6_K weights —
//! what Ollama serves as `llama3.2:1b` (Q8_0) and `llama3.1:8b` (Q4_K_M). The loader and weight repacking run anywhere; the
//! decode loop needs a Metal device.

#[cfg(any(target_os = "macos", target_os = "linux"))]
pub mod chat;
pub mod dev;
pub mod gguf;
pub mod json;
pub mod llama;
// Generated (`scripts/unicode_nfc_table.py`): kept as the generator
// writes it, one entry a line, not spread over thousands by rustfmt.
#[rustfmt::skip]
pub mod nfc_table;
pub mod qwen;
pub mod qwen_run;
pub mod qwen_source;
pub mod sample;
pub mod tokenizer;
