//! The Metal prefill GEMM alone, on the model's own shapes at a full chunk.
//!
//!     cargo run --release -p lex-rt --example gemm_metal
//!     LEX_GEMM_BN=32 cargo run --release -p lex-rt --example gemm_metal -- --tokens 32
//!
//! Prefill is 80% matmuls, and this is where a change to
//! `lex_msl::gemm`'s Metal kernel is measured before the whole model is.
//! Each shape runs over several copies of its weights in one command
//! buffer, so the weights stream from memory as they do in the model
//! rather than sitting in the system cache between repeats; every dispatch
//! writes the same output, which keeps them in order. Correctness is
//! `crates/lex-metal/tests/gemm_gpu.rs`, against the interpreter.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use half::f16;
    use lex_metal::{Buffer, Gpu};
    use lex_msl::gemm::{Backend, Gemm, gemm_nvfp4};

    let gpu = Gpu::open()?;
    let args: Vec<String> = std::env::args().collect();
    let tokens: usize = match args.iter().position(|a| a == "--tokens") {
        Some(i) => args
            .get(i + 1)
            .and_then(|v| v.parse().ok())
            .ok_or("--tokens takes a count")?,
        None => 128,
    };
    let copies = 4;
    let reps = 5;
    // (label, tokens, rows, inputs, residual, f16 activations): the
    // shapes of one Qwen3.8 layer's matmuls, at a chunk of `--tokens`.
    let shapes = [
        ("gate/up", tokens, 17408, 5120, false, true),
        ("down", tokens, 5120, 17408, true, true),
        ("qkv", tokens, 10240, 5120, false, true),
        ("out_proj", tokens, 5120, 6144, true, false),
    ];
    println!(
        "{:<10} {:>6} {:>6} {:>6} {:>9} {:>8}",
        "shape", "m", "n", "k", "ms", "TFLOPS"
    );
    for (label, m, n, k, residual, x_half) in shapes {
        let g = Gemm {
            m,
            n,
            k,
            residual,
            x_half,
        };
        let pipe = gpu.build_lowered(&gemm_nvfp4(&g, Backend::Metal)?)?;
        let x: Buffer = if x_half {
            gpu.upload(
                &(0..m * k)
                    .map(|i| f16::from_f32(((i * 7) % 13) as f32 * 0.01 - 0.06))
                    .collect::<Vec<_>>(),
            )
        } else {
            gpu.upload(
                &(0..m * k)
                    .map(|i| ((i * 7) % 13) as f32 * 0.01 - 0.06)
                    .collect::<Vec<_>>(),
            )
        };
        let sets: Vec<[Buffer; 3]> = (0..copies)
            .map(|c| {
                [
                    gpu.upload(
                        &(0..n * k / 2)
                            .map(|i| (i.wrapping_mul(97) + c) as u8)
                            .collect::<Vec<_>>(),
                    ),
                    gpu.upload(
                        &(0..n * k / 16)
                            .map(|i| 0x30u8 + (i % 7) as u8)
                            .collect::<Vec<_>>(),
                    ),
                    gpu.upload(&vec![1.0f32; n]),
                ]
            })
            .collect();
        let r = gpu.zeroed::<f32>(m * n);
        let y = gpu.zeroed::<f32>(m * n);
        let bufs: Vec<Vec<&Buffer>> = sets
            .iter()
            .map(|[q, s, gs]| {
                let mut b = vec![&x, q, s, gs];
                if residual {
                    b.push(&r);
                }
                b.push(&y);
                b
            })
            .collect();
        let steps: Vec<(&lex_metal::Pipeline, &[&Buffer])> = (0..copies * 4)
            .map(|i| (&pipe, bufs[i % copies].as_slice()))
            .collect();
        gpu.run_all(&steps);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let (_, gpu_s) = gpu.run_all_timed(&steps);
            best = best.min(gpu_s / steps.len() as f64);
        }
        println!(
            "{label:<10} {m:>6} {n:>6} {k:>6} {:>9.3} {:>8.2}",
            best * 1e3,
            2.0 * (m * n * k) as f64 / best / 1e12
        );
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("gemm_metal needs a Metal device");
}
