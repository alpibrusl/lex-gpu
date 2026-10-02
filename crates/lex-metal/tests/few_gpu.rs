//! The few-token NVFP4 matvec (`lex_msl::few`) against the interpreter
//! running the program it stands in for (`matmul_q_x`, same parameters,
//! same order). The two sum in different orders, so they agree to f32
//! rounding, not to the bit.

#![cfg(target_os = "macos")]

use half::f16;
use lex_front::llama::{QLayout, matmul_q_x};
use lex_front::{Tensor, run};
use lex_ir::DType;
use lex_ir::reference::fill_pattern_f32;
use lex_metal::{Buffer, Gpu};
use lex_msl::few::{Few, layouts, matvec_few_nvfp4_with};

fn upload(gpu: &Gpu, t: &Tensor) -> Buffer {
    match t.dtype {
        DType::F32 => gpu.upload(&t.data),
        DType::F16 => gpu.upload(&t.data.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>()),
        DType::I8 => gpu.upload(&t.data.iter().map(|&x| x as i8).collect::<Vec<_>>()),
    }
}

#[test]
fn the_few_token_matvec_matches_the_matmul_it_replaces() {
    let Some(gpu) = Gpu::open().ok() else {
        eprintln!("no Metal device");
        return;
    };
    // 102 rows leave the last threadgroup (four rows) half full; 1040
    // inputs are not whole steps of 512, so lanes run out unevenly.
    for (tokens, n, k, residual, x_half) in [
        (1, 128, 512, true, false),
        (1, 102, 1040, false, false),
        (2, 128, 512, false, true),
        (3, 102, 1040, true, true),
        (4, 128, 512, true, false),
        (3, 64, 256, false, false),
    ] {
        let f = Few {
            tokens,
            n,
            k,
            residual,
            x_half,
        };
        let xt = if x_half { DType::F16 } else { DType::F32 };
        let prog = matmul_q_x(tokens, k, n, 2, k, QLayout::NVFP4, residual, xt).expect("program");
        let mut x = vec![0.0; tokens * k];
        fill_pattern_f32(&mut x, 3);
        // Real codes and scales: all-zero NVFP4 decodes to nothing and
        // agrees with anything.
        let codes: Vec<f32> = (0..n * k / 2)
            .map(|i| ((i.wrapping_mul(97).wrapping_add(3) & 0xFF) as i32 - 128) as f32)
            .collect();
        let scales: Vec<f32> = (0..n * k / 16)
            .map(|i| (0x34i32 + (i % 7) as i32 - 128) as f32)
            .collect();
        let rows: Vec<f32> = (0..n).map(|i| 0.5 + (i % 5) as f32 * 0.125).collect();
        let mut tensors = vec![
            Tensor::new(xt, &[tokens, k], &x),
            Tensor::new(DType::I8, &[n, k / 2], &codes),
            Tensor::new(DType::I8, &[n, k / 16], &scales),
            Tensor::new(DType::F32, &[n], &rows),
        ];
        if residual {
            let mut r = vec![0.0; tokens * n];
            fill_pattern_f32(&mut r, 11);
            tensors.push(Tensor::new(DType::F32, &[tokens, n], &r));
        }
        tensors.push(Tensor::zeros(DType::F32, &[tokens, n]));
        let out = tensors.len() - 1;

        // Every layout a tuner may pick, each against the interpreter.
        let bufs: Vec<Buffer> = tensors.iter().map(|t| upload(&gpu, t)).collect();
        let refs: Vec<&Buffer> = bufs.iter().collect();
        let mut outs = vec![];
        for layout in layouts() {
            let pipe = gpu
                .build_lowered(&matvec_few_nvfp4_with(&f, layout).expect("few"))
                .expect("compile");
            gpu.run(&pipe, &refs);
            let mut got = vec![0.0f32; tokens * n];
            gpu.download(&bufs[out], &mut got);
            outs.push((layout, got));
        }

        run(&prog, &mut tensors).expect("interpret");
        let want = &tensors[out].data;
        let scale = want.iter().fold(1e-6f32, |a, x| a.max(x.abs()));
        assert!(
            scale > 1.0,
            "{f:?}: the reference is all but zero ({scale})"
        );
        for (layout, got) in &outs {
            let err = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
                / scale;
            assert!(
                err < 1e-5,
                "{f:?} {layout:?}: few-token matvec vs interpreter {err:e} of scale"
            );
        }
        eprintln!("{f:?}: {} layouts agree with the interpreter", outs.len());
    }
}
