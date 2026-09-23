//! The tensors a caller hands a program must match what it declared.
//!
//! This is a shape check, not a numerical one, and it needs no GPU — the
//! interpreter rejects a wrong shape or dtype on any machine. It exists
//! because a device test built its tensors by hand, got one of them wrong
//! (`gs` is `[n_out]`, not `[n_out, 1]`), and the mismatch was discovered
//! by a rented L4 that never reached the kernel it was rented to test.
//!
//! Anything a device test feeds a program should be checked here first, so
//! the cloud only ever answers questions about silicon.

use lex_front::llama::{QLayout, matvec_q, rmsnorm_rows, silu_mul};
use lex_front::{Tensor, run};
use lex_ir::DType;

fn fill(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed);
            ((x >> 9) as f32 / 4_194_304.0) - 1.0
        })
        .collect()
}

/// Exactly the tensors `lex-cuda`'s device tests build, run through the
/// interpreter. If this passes, those tests reach the GPU.
#[test]
fn the_device_tests_hand_over_tensors_the_programs_accept() {
    let (rows, n) = (4, 4096);
    let mut cases: Vec<(&str, _, Vec<Tensor>)> = vec![
        (
            "rmsnorm",
            rmsnorm_rows(rows, n, 1e-5, None, DType::F32),
            vec![
                Tensor::new(DType::F32, &[rows, n], &fill(rows * n, 1)),
                Tensor::new(DType::F32, &[1, n], &fill(n, 2)),
                Tensor::zeros(DType::F32, &[rows, n]),
            ],
        ),
        (
            "silu_mul",
            silu_mul(n, 256, DType::F32).expect("silu"),
            vec![
                Tensor::new(DType::F32, &[1, n], &fill(n, 3)),
                Tensor::new(DType::F32, &[1, n], &fill(n, 4)),
                Tensor::zeros(DType::F32, &[1, n]),
            ],
        ),
    ];

    let (n_in, n_out) = (512, 64);
    let codes: Vec<f32> = (0..n_out * n_in / 2)
        .map(|i: usize| ((i.wrapping_mul(31).wrapping_add(7) & 0xFF) as i32 - 128) as f32)
        .collect();
    let scales: Vec<f32> = (0..n_out * n_in / 16)
        .map(|i: usize| (0x38i32 + (i % 5) as i32 - 128) as f32)
        .collect();
    cases.push((
        "nvfp4 matvec",
        matvec_q(n_in, n_out, 8, n_in, QLayout::NVFP4, false).expect("matvec"),
        vec![
            Tensor::new(DType::F32, &[1, n_in], &fill(n_in, 5)),
            Tensor::new(DType::I8, &[n_out, n_in / 2], &codes),
            Tensor::new(DType::I8, &[n_out, n_in / 16], &scales),
            // `[n_out]`, not `[n_out, 1]`: the tensor's own f32 scale is one
            // value per row, and the interpreter is strict about rank.
            Tensor::new(DType::F32, &[n_out], &vec![1.0; n_out]),
            Tensor::zeros(DType::F32, &[1, n_out]),
        ],
    ));

    for (name, prog, mut tensors) in cases {
        run(&prog, &mut tensors).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}
