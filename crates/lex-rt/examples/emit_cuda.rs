//! Emit CUDA for every kernel the Llama decode and prefill paths build, so
//! `scripts/cuda_check.sh` can compile it on a machine with no GPU.
//!
//!     cargo run --release -p lex-rt --example emit_cuda -- out/
//!     scripts/cuda_check.sh out/*.cu
//!
//! `cargo check` proves the runtime *types* against the CUDA device. It
//! cannot prove the emitted text is CUDA at all: `lower` defaults to MSL,
//! and a runtime that kept the default compiles perfectly well and then
//! hands Metal source to NVRTC at load time. That failure would otherwise
//! arrive on a rented GPU, twenty minutes and a VM after the mistake.
//!
//! The shapes come from the checkpoint rather than from guesses -- the
//! loader runs anywhere, GPU or not -- so these are the programs that
//! actually run, at the sizes they actually run at.
//!
//! **This list is maintained by hand.** It mirrors the `compile` calls in
//! `llama.rs`; a kernel added there and not here is simply not covered,
//! and the first thing to check when the cloud fails on something this
//! passed is whether the two have drifted.

use std::path::PathBuf;

use lex_front::Program;
use lex_front::flash::FlashDecode;
use lex_front::llama::{
    QLayout, kv_append, kv_append_rows, matmul_q, matmul_q_glu, matvec_q, rmsnorm, rmsnorm_rows,
    rope, rope_rows, silu_mul,
};
use lex_ir::{DType, Space, Target};
use lex_msl::dialect::Cuda;
use lex_msl::program::lower_with;
use lex_rt::gguf::ollama_model;
use lex_rt::llama::Weights;

const THREADS: usize = 256;
const BO: usize = 8;

fn main() -> Result<(), String> {
    let mut dir = PathBuf::from("out");
    let mut model = "llama3.2:1b".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next().ok_or("--model needs a value")?,
            other => dir = PathBuf::from(other),
        }
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;

    let w = Weights::load(&ollama_model(&model)?, 2048)?;
    let c = &w.cfg;
    let target = Target::nvidia_ada();
    println!(
        "{model}: dim {} heads {}/{} ffn {} vocab {}",
        c.dim, c.n_head, c.n_kv, c.ffn, c.vocab
    );

    let mut progs: Vec<Program> = vec![];
    let cap = 2048;

    // Decode: one token.
    progs.push(rmsnorm(c.dim, c.eps));
    progs.push(rope(c.n_head, c.head_dim, DType::F16));
    progs.push(rope(c.n_kv, c.head_dim, DType::F16));
    progs.push(silu_mul(c.ffn, THREADS, DType::F32)?);
    progs.push(kv_append(c.n_kv, c.head_dim, cap, DType::F16));
    progs.push(kv_append(c.n_kv, c.head_dim, cap, DType::F32));

    // Prefill: `t` tokens at once. One representative batch is enough --
    // the shapes differ only in the leading factor.
    let t = 8;
    progs.push(rmsnorm_rows(t, c.dim, c.eps, None, DType::F32));
    progs.push(rope_rows(t, c.n_kv, c.head_dim, DType::F16));
    progs.push(silu_mul(t * c.ffn, THREADS, DType::F32)?);
    progs.push(kv_append_rows(t, c.n_kv, c.head_dim, cap, DType::F16));
    progs.push(kv_append_rows(t, c.n_kv, c.head_dim, cap, DType::F32));

    // Every distinct matrix shape, in the layout this checkpoint stores.
    let layout = QLayout::Q8_0;
    let shapes = [
        (c.dim, c.n_head * c.head_dim, false),
        (c.dim, c.n_kv * c.head_dim, false),
        (c.n_head * c.head_dim, c.dim, true),
        (c.dim, c.ffn, false),
        (c.ffn, c.dim, true),
        (c.dim, c.vocab, false),
    ];
    for (n_in, n_out, res) in shapes {
        progs.push(matvec_q(n_in, n_out, BO, n_in, layout, res)?);
        progs.push(matmul_q(t, n_in, n_out, BO, n_in, layout, res)?);
    }
    progs.push(matmul_q_glu(t, c.dim, c.ffn, BO, layout, None)?);

    // Attention, all four builders.
    let attn = FlashDecode {
        q_rows: c.n_head / c.n_kv,
        d: c.head_dim,
        seq: cap,
        bq: c.n_head / c.n_kv,
        bk: 16,
        stages: 1,
        dtype: DType::F16,
        kv_space: Space::Reg,
        consumers: 0,
        heads: c.n_kv,
        kv_cap: cap,
    };
    progs.push(attn.build_dynamic()?);
    progs.push(attn.build_split(2)?);
    progs.push(attn.build_combine(2)?);
    progs.push(attn.build_causal(t)?);

    let mut n = 0;
    for p in &progs {
        // 128 for attention, as the runtime builds it; the rest take the
        // default. Lowering is what can fail here, and it fails per kernel.
        let threads = if p.name.starts_with("flash") { 128 } else { THREADS };
        let lowered = lower_with(p, &target, threads, &Cuda)
            .map_err(|e| format!("`{}` does not lower for CUDA: {e}", p.name))?;
        let path = dir.join(format!("{}.cu", lowered.entry));
        std::fs::write(&path, &lowered.source).map_err(|e| format!("{}: {e}", path.display()))?;
        n += 1;
    }
    println!("{n} kernels written to {}", dir.display());
    Ok(())
}
