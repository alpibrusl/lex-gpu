//! The chunked gated delta rule (`lex_msl::delta`) against the interpreter
//! running the step program it stands in for, token by token.
//!
//! The two sum in different orders -- the chunked form solves each chunk
//! in closed form and multiplies on the matrix units -- so they agree to
//! f32 rounding, not to the bit.

#![cfg(target_os = "macos")]

use lex_front::qwen::DeltaNet;
use lex_front::{Tensor, run};
use lex_ir::DType;
use lex_ir::reference::fill_pattern_f32;
use lex_metal::{Buffer, Gpu};
use lex_msl::delta::{DeltaChunk, delta_chunked};
use lex_msl::gemm::Backend;

/// Largest difference over largest magnitude, for `y` and the final state.
fn chunked_error(gpu: &Gpu, d: DeltaChunk, gate: impl Fn(usize) -> f32) -> (f32, f32) {
    let (t, hv, dk, dv) = (d.tokens, d.v_heads, d.k_dim, d.v_dim);
    let prog = DeltaNet {
        v_heads: hv,
        k_heads: hv,
        k_dim: dk,
        v_dim: dv,
        rows: 8,
        v_base: d.v_base,
        v_width: d.v_width,
    }
    .build_steps(t)
    .expect("step program");
    let pattern = |n: usize, seed: u32, scale: f32| {
        let mut x = vec![0.0; n];
        fill_pattern_f32(&mut x, seed);
        x.iter().map(|v| v * scale).collect::<Vec<f32>>()
    };
    // Unit-length q and k rows, as the model's normalisation leaves them.
    let unit = |seed: u32| {
        let mut x = pattern(t * hv * dk, seed, 1.0);
        for row in x.chunks_mut(dk) {
            let n = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
            row.iter_mut().for_each(|v| *v /= n);
        }
        x
    };
    // One gate and one beta per head and token, spread over the head's rows.
    let spread =
        |f: &dyn Fn(usize) -> f32| (0..t * hv * dv).map(|i| f(i / dv)).collect::<Vec<f32>>();
    let beta = |h: usize| 0.2 + 0.6 * ((h * 37 % 11) as f32 / 10.0);
    let mut tensors = vec![
        Tensor::new(DType::F32, &[hv * dv, dk], &pattern(hv * dv * dk, 5, 0.3)),
        Tensor::new(DType::F32, &[t * hv, dk], &unit(7)),
        Tensor::new(DType::F32, &[t * hv, dk], &unit(9)),
        Tensor::new(
            DType::F32,
            &[t * d.v_width],
            &pattern(t * d.v_width, 11, 1.0),
        ),
        Tensor::new(DType::F32, &[t * hv * dv], &spread(&gate)),
        Tensor::new(DType::F32, &[t * hv * dv], &spread(&beta)),
        Tensor::zeros(DType::F32, &[t * hv * dv]),
    ];
    let pipe = gpu
        .build_lowered(&delta_chunked(&d, Backend::Metal).expect("chunked"))
        .expect("compile");
    let bufs: Vec<Buffer> = tensors.iter().map(|x| gpu.upload(&x.data)).collect();
    let refs: Vec<&Buffer> = bufs.iter().collect();
    gpu.run(&pipe, &refs);
    let mut y = vec![0.0f32; t * hv * dv];
    gpu.download(&bufs[6], &mut y);
    let mut s = vec![0.0f32; hv * dv * dk];
    gpu.download(&bufs[0], &mut s);

    run(&prog, &mut tensors).expect("interpret");
    let err = |got: &[f32], want: &[f32]| {
        let scale = want.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
        assert!(scale > 0.1, "the reference is all but zero ({scale})");
        got.iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale
    };
    (err(&y, &tensors[6].data), err(&s, &tensors[0].data))
}

/// A head's gate, by head.
type Gate = Box<dyn Fn(usize) -> f32>;

#[test]
fn chunked_delta_rule_matches_the_step_program() {
    let Some(gpu) = Gpu::open().ok() else {
        eprintln!("no Metal device");
        return;
    };
    let shape = |tokens, v_base, v_width| DeltaChunk {
        tokens,
        v_heads: 3,
        k_dim: 128,
        v_dim: 128,
        v_base,
        v_width,
    };
    // Mild decay; one chunk and several; `v` behind other columns as the
    // model's convolution output lays it; and a head whose gate underflows
    // to nothing, which a log of the gate must survive.
    let cases: Vec<(DeltaChunk, Gate)> = vec![
        (
            shape(16, 0, 384),
            Box::new(|i| 0.9 + 0.09 * ((i % 5) as f32 / 4.0)),
        ),
        (
            shape(64, 0, 384),
            Box::new(|i| 0.5 + 0.49 * ((i * 7 % 9) as f32 / 8.0)),
        ),
        (shape(48, 100, 600), Box::new(|_| 0.97)),
        (
            shape(32, 0, 384),
            Box::new(|i| if i % 3 == 1 { 0.0 } else { 0.95 }),
        ),
    ];
    for (d, gate) in cases {
        let (ey, es) = chunked_error(&gpu, d, gate);
        eprintln!("{d:?}: y {ey:e}, state {es:e} of scale");
        assert!(ey < 1e-4 && es < 1e-4, "{d:?}: y {ey:e}, state {es:e}");
    }
}
