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
    let (words, rest) = GOLDEN.as_chunks::<4>();
    assert!(rest.is_empty());
    words.iter().map(|b| f32::from_le_bytes(*b)).collect()
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
        consumers: 0,
        heads: 1,
        kv_cap: 0,
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

/// One program for every length: the live length and block count are
/// runtime scalars, the last block is masked. Checked at lengths that end
/// mid-block, on a block boundary, at 1, and at capacity.
#[test]
fn dynamic_length_attention_matches_the_reference_at_every_length() {
    use tile_front::run_dyn;
    let cap = 128;
    let cfg = FlashDecode {
        seq: cap,
        kv_cap: cap,
        bk: 16,
        stages: 1,
        ..schedule(8, 16, 1)
    };
    let prog = cfg.build_dynamic().unwrap();
    check(&prog, &Target::apple_m_series()).expect("check");
    for len in [1usize, 5, 16, 17, 100, cap] {
        let mut t = vec![
            input(Q_ROWS, 1),
            input(cap, 2),
            input(cap, 3),
            Tensor::zeros(DType::F32, &[Q_ROWS, D]),
        ];
        let nkb = len.div_ceil(16) as u32;
        run_dyn(&prog, &mut t, &[len as u32, nkb]).expect("interpret");
        let want = flash::reference(
            &t[0].data,
            &t[1].data[..len * D],
            &t[2].data[..len * D],
            Q_ROWS,
            len,
            D,
        );
        let err = max_rel_err(&t[3].data, &want);
        assert!(err < TOL, "len {len}: max rel err {err:e}");
    }
}

/// Prefill / speculative verify: `tokens` new queries at `pos0..`, each
/// seeing only the positions up to its own. Queries and outputs in the
/// natural [token, kv head, group, d] order.
#[test]
fn causal_attention_sees_exactly_its_own_prefix() {
    use tile_front::run_dyn;
    let (heads, group, cap, bk) = (2usize, 4usize, 64usize, 16usize);
    let hg = heads * group;
    let cfg = FlashDecode {
        q_rows: group,
        d: D,
        seq: cap,
        bq: group,
        bk,
        stages: 1,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 0,
        heads,
        kv_cap: cap,
    };
    for (tokens, pos0) in [(1usize, 0usize), (5, 20), (4, 12), (8, 56)] {
        let prog = cfg.build_causal(tokens).unwrap();
        check(&prog, &Target::apple_m_series()).expect("check");
        let mk = |rows: usize, seed: u32| {
            let mut x = vec![0.0; rows * D];
            fill_pattern_f32(&mut x, seed);
            Tensor::new(DType::F16, &[rows, D], &x)
        };
        let mut t = vec![
            mk(tokens * hg, 1),
            mk(heads * cap, 2),
            mk(heads * cap, 3),
            Tensor::zeros(DType::F32, &[tokens * hg, D]),
        ];
        let nkb = (pos0 + tokens).div_ceil(bk) as u32;
        run_dyn(&prog, &mut t, &[pos0 as u32, nkb]).expect("interpret");
        for tok in 0..tokens {
            let len = pos0 + tok + 1;
            for h in 0..heads {
                let r0 = tok * hg + h * group;
                let q = &t[0].data[r0 * D..(r0 + group) * D];
                let k = &t[1].data[h * cap * D..(h * cap + len) * D];
                let v = &t[2].data[h * cap * D..(h * cap + len) * D];
                let want = flash::reference(q, k, v, group, len, D);
                let got = &t[3].data[r0 * D..(r0 + group) * D];
                let err = max_rel_err(got, &want);
                assert!(
                    err < TOL,
                    "tokens {tokens} pos0 {pos0} token {tok} head {h}: {err:e}"
                );
            }
        }
    }
}
