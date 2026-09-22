//! P1 exit test, interpreter half: flash-attention decode with K/V reused
//! across query blocks and pipelined copies type-checks, and matches PyTorch.

use tile_front::flash::{self, FlashDecode};
use tile_front::{Kind, Tensor, check, run};
use tile_ir::reference::{fill_pattern_f32, max_rel_err};
use tile_ir::{DType, Space, Target};

const Q_ROWS: usize = 32;
const D: usize = 128;
const SEQ: usize = 512;

/// Written by `scripts/flash_decode_golden.py` (PyTorch SDPA, float64).
const GOLDEN: &[u8] = include_bytes!("data/flash_decode_f16_q32_d128_s512.f32");

/// Interpreter vs. PyTorch. The interpreter accumulates in f32 in a different
/// order (online softmax over blocks); PyTorch ran in f64.
const TOL: f32 = 1e-4;

fn golden() -> Vec<f32> {
    GOLDEN
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn input(rows: usize, seed: u32) -> Tensor {
    let mut x = vec![0.0; rows * D];
    fill_pattern_f32(&mut x, seed);
    Tensor::new(DType::F16, &[rows, D], &x)
}

fn schedule(bq: usize, bk: usize, stages: usize) -> FlashDecode {
    FlashDecode {
        q_rows: Q_ROWS,
        d: D,
        seq: SEQ,
        bq,
        bk,
        stages,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
    }
}

fn run_flash(cfg: &FlashDecode) -> Vec<f32> {
    let prog = cfg.build().expect("build");
    let mut t = vec![
        input(Q_ROWS, 1),
        input(SEQ, 2),
        input(SEQ, 3),
        Tensor::zeros(DType::F32, &[Q_ROWS, D]),
    ];
    run(&prog, &mut t).expect("interpret");
    t.pop().unwrap().data
}

/// Metal: 32 KiB threadgroup, no async copy. Double-buffered f16 K/V at
/// bk = 32, d = 128 is exactly 2 stages x 2 operands x 8 KiB = 32 KiB.
fn metal() -> FlashDecode {
    schedule(8, 32, 2)
}

/// Hopper: the design doc's schedule shape — kv 128, 3 stages — in 192 KiB.
fn hopper() -> FlashDecode {
    schedule(16, 128, 3)
}

#[test]
fn metal_schedule_type_checks_and_fits() {
    let r = check(&metal().build().unwrap(), &Target::apple_m_series()).expect("check");
    assert_eq!(r.peak_threadgroup_bytes, 32 * 1024);
    assert_eq!(r.dups, 0, "K reuse must not need dup");
    // Metal has no copy engine: the pipeline type-checks but is flagged.
    assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
}

#[test]
fn hopper_schedule_type_checks_and_fits() {
    let r = check(&hopper().build().unwrap(), &Target::nvidia_hopper()).expect("check");
    assert_eq!(r.peak_threadgroup_bytes, 3 * 2 * 128 * D * 2);
    assert_eq!(r.dups, 0);
    assert!(r.warnings.is_empty(), "{:?}", r.warnings);
}

#[test]
fn hopper_schedule_is_rejected_on_metal() {
    let errs = check(&hopper().build().unwrap(), &Target::apple_m_series()).unwrap_err();
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(errs[0].kind, Kind::Budget);
}

#[test]
fn metal_schedule_matches_pytorch() {
    let err = max_rel_err(&run_flash(&metal()), &golden());
    assert!(err < TOL, "max rel err {err:e}");
}

#[test]
fn hopper_schedule_matches_pytorch() {
    let err = max_rel_err(&run_flash(&hopper()), &golden());
    assert!(err < TOL, "max rel err {err:e}");
}

/// The schedule changes placement, never the answer: unpipelined, deep
/// pipelines, one query block, KV in registers.
#[test]
fn every_schedule_computes_the_same_attention() {
    let want = golden();
    // The whole sequence in one block is 256 KiB of K/V: over budget in
    // threadgroup memory, so it stages in (unbudgeted) registers instead.
    let mut whole = schedule(8, 512, 1);
    whole.kv_space = Space::Reg;
    let cfgs = [schedule(32, 64, 1), schedule(4, 16, 4), whole];
    for cfg in cfgs {
        let prog = cfg.build().unwrap();
        check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{cfg:?}: {e:?}"));
        let err = max_rel_err(&run_flash(&cfg), &want);
        assert!(err < TOL, "{cfg:?}: max rel err {err:e}");
    }
}

/// The Rust f64 reference agrees with PyTorch, which pins the input
/// generator in the Python script to `fill_pattern_f32`.
#[test]
fn f64_reference_agrees_with_pytorch() {
    let (q, k, v) = (input(Q_ROWS, 1), input(SEQ, 2), input(SEQ, 3));
    let got = flash::reference(&q.data, &k.data, &v.data, Q_ROWS, SEQ, D);
    let err = max_rel_err(&got, &golden());
    assert!(err < 1e-6, "max rel err {err:e}");
}
