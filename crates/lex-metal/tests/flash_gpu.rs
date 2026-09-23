//! P2, first slice: a typed lex-front kernel runs on the GPU.
//!
//! Flash-attention decode is checked, lowered to MSL, compiled, dispatched on
//! the Metal device, and compared against PyTorch's SDPA output and against
//! the interpreter. On CI's paravirtualised device this is a correctness test;
//! the numbers only mean something on real silicon.
#![cfg(target_os = "macos")]

use half::f16;
use lex_front::flash::FlashDecode;
use lex_front::{Tensor, check, run};
use lex_ir::reference::{fill_pattern_f32, max_rel_err};
use lex_ir::{DType, Space, Target};
use lex_metal::Gpu;
use lex_msl::program::lower;

const Q_ROWS: usize = 32;
const D: usize = 128;
const SEQ: usize = 512;
const THREADS: usize = 256;
const PYTORCH: &[u8] =
    include_bytes!("../../lex-front/tests/data/flash_decode_f16_q32_d128_s512.f32");

fn schedule(bq: usize, bk: usize, stages: usize, heads: usize) -> FlashDecode {
    FlashDecode {
        q_rows: Q_ROWS,
        d: D,
        seq: SEQ,
        bq,
        bk,
        stages,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 0,
        heads,
        kv_cap: 0,
    }
}

fn pattern(n: usize, seed: u32) -> Vec<f32> {
    let mut x = vec![0.0; n];
    fill_pattern_f32(&mut x, seed);
    x
}

/// Run `cfg` on the GPU; inputs are the golden's pattern, one seed per head.
fn gpu_run(gpu: &Gpu, cfg: &FlashDecode) -> (Vec<f32>, Vec<Tensor>) {
    let prog = cfg.build().unwrap();
    check(&prog, &Target::apple_m_series()).expect("check");
    let lowered = lower(&prog, gpu.target(), THREADS).expect("lower");
    let pipe = gpu.build_lowered(&lowered).expect("compile");

    let h = cfg.heads;
    let mut q = vec![];
    let mut k = vec![];
    let mut v = vec![];
    for head in 0..h as u32 {
        q.extend(pattern(Q_ROWS * D, 1 + 3 * head));
        k.extend(pattern(SEQ * D, 2 + 3 * head));
        v.extend(pattern(SEQ * D, 3 + 3 * head));
    }
    let to16 = |x: &[f32]| x.iter().map(|&a| f16::from_f32(a)).collect::<Vec<_>>();
    let bufs = [
        gpu.upload(&to16(&q)),
        gpu.upload(&to16(&k)),
        gpu.upload(&to16(&v)),
        gpu.zeroed::<f32>(h * Q_ROWS * D),
    ];
    gpu.run(&pipe, &[&bufs[0], &bufs[1], &bufs[2], &bufs[3]]);
    let mut out = vec![0.0f32; h * Q_ROWS * D];
    gpu.download(&bufs[3], &mut out);

    let tensors = vec![
        Tensor::new(DType::F16, &[h * Q_ROWS, D], &q),
        Tensor::new(DType::F16, &[h * SEQ, D], &k),
        Tensor::new(DType::F16, &[h * SEQ, D], &v),
        Tensor::zeros(DType::F32, &[h * Q_ROWS, D]),
    ];
    (out, tensors)
}

fn golden() -> Vec<f32> {
    let (words, _) = PYTORCH.as_chunks::<4>();
    words.iter().map(|b| f32::from_le_bytes(*b)).collect()
}

#[test]
fn flash_decode_on_the_gpu_matches_pytorch() {
    let gpu = Gpu::open().expect("metal device");
    // Double-buffered, and unpipelined with a wider block. Both leave room
    // for the lowering's staging scratch inside 32 KiB.
    for cfg in [
        schedule(8, 16, 2, 1),
        schedule(8, 32, 1, 1),
        schedule(16, 16, 3, 1),
    ] {
        let (out, _) = gpu_run(&gpu, &cfg);
        let err = max_rel_err(&out, &golden());
        assert!(err < 1e-4, "{cfg:?}: max rel err vs PyTorch {err:e}");
    }
}

#[test]
fn every_grid_instance_computes_its_own_head() {
    let gpu = Gpu::open().expect("metal device");
    let cfg = schedule(8, 16, 2, 5);
    let (out, mut tensors) = gpu_run(&gpu, &cfg);
    run(&cfg.build().unwrap(), &mut tensors).expect("interpret");
    let err = max_rel_err(&out, &tensors[3].data);
    // `exp` differs in the last bits between Metal and Rust; nothing else does.
    assert!(err < 1e-4, "max rel err vs interpreter {err:e}");
    // Head 0 used the golden's seeds.
    let err0 = max_rel_err(&out[..Q_ROWS * D], &golden());
    assert!(err0 < 1e-4, "head 0 vs PyTorch {err0:e}");
}

/// The Metal schedule that fills 32 KiB exactly type-checks, but the lowering
/// needs staging scratch on top. The backend says so instead of failing at
/// pipeline creation.
#[test]
fn lowering_accounts_for_its_own_scratch() {
    let prog = schedule(8, 32, 2, 1).build().unwrap();
    check(&prog, &Target::apple_m_series()).expect("the program itself fits");
    let err = lower(&prog, &Target::apple_m_series(), THREADS).unwrap_err();
    assert!(err.contains("staging scratch"), "{err}");
}
