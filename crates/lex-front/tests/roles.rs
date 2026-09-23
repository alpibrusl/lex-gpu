//! P1 exit test, second half: the Hopper schedule in its real shape — a
//! producer warp and consumer warpgroups over a barrier-synchronised pipe —
//! type-checks, derives its barrier arrival counts, and matches PyTorch.

use std::time::Duration;

use lex_front::flash::FlashDecode;
use lex_front::interp::{RunOptions, run_with};
use lex_front::ir::{Arg, Builder, IdxExpr, Op, RoleDef, TileTy, Ty, Var, View};
use lex_front::{Kind, Program, Tensor, check, run};
use lex_ir::reference::{fill_pattern_f32, max_rel_err};
use lex_ir::{DType, Space, Target};

const Q_ROWS: usize = 32;
const D: usize = 128;
const SEQ: usize = 512;
const GOLDEN: &[u8] = include_bytes!("data/flash_decode_f16_q32_d128_s512.f32");

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

/// FA3's decode shape on H100: bk 128, 3 stages, two consumer warpgroups.
fn hopper_ws() -> FlashDecode {
    FlashDecode {
        q_rows: Q_ROWS,
        d: D,
        seq: SEQ,
        bq: 16,
        bk: 128,
        stages: 3,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 2,
        heads: 1,
        kv_cap: 0,
    }
}

fn run_flash(prog: &Program) -> Vec<f32> {
    let mut t = vec![
        input(Q_ROWS, 1),
        input(SEQ, 2),
        input(SEQ, 3),
        Tensor::zeros(DType::F32, &[Q_ROWS, D]),
    ];
    run(prog, &mut t).expect("interpret");
    t.pop().unwrap().data
}

#[test]
fn warp_specialised_schedule_checks_on_hopper() {
    let r = check(&hopper_ws().build().unwrap(), &Target::nvidia_hopper()).expect("check");
    assert_eq!(r.peak_threadgroup_bytes, 3 * 2 * 128 * D * 2);
    assert_eq!(r.dups, 0);
    assert_eq!(r.pipes.len(), 1);
    let p = &r.pipes[0];
    // The mbarrier inits a Hopper backend would emit, read off the types.
    assert_eq!(p.full_arrivals, 1);
    assert_eq!(p.full_tx_bytes, 2 * 128 * D * 2);
    assert_eq!(p.empty_arrivals, 2);
    assert_eq!(p.transfers, SEQ / 128);
    assert_eq!(p.threads, 32 + 2 * 4 * 32);
}

#[test]
fn warp_specialised_schedule_has_no_lowering_on_metal_or_cdna3() {
    for t in [Target::apple_m_series(), Target::amd_cdna3()] {
        let errs = check(&hopper_ws().build().unwrap(), &t).unwrap_err();
        assert_eq!(errs.len(), 1, "{}: {errs:#?}", t.name);
        assert_eq!(errs[0].kind, Kind::Target, "{}", t.name);
    }
}

/// Roles really run concurrently here, so every run is a different
/// interleaving. All of them must match PyTorch, and each other bit for bit:
/// each consumer's arithmetic is sequential, only its timing varies.
#[test]
fn warp_specialised_schedule_matches_pytorch_under_any_interleaving() {
    let prog = hopper_ws().build().unwrap();
    let first = run_flash(&prog);
    let err = max_rel_err(&first, &golden());
    assert!(err < 1e-4, "max rel err {err:e}");
    for _ in 0..20 {
        assert_eq!(run_flash(&prog), first);
    }
}

#[test]
fn deeper_and_wider_pipes_compute_the_same_attention() {
    let want = golden();
    for (bq, bk, stages, consumers) in [(8, 64, 2, 4), (4, 32, 1, 4), (32, 16, 6, 1)] {
        let cfg = FlashDecode {
            bq,
            bk,
            stages,
            consumers,
            ..hopper_ws()
        };
        let prog = cfg.build().unwrap();
        check(&prog, &Target::nvidia_hopper()).unwrap_or_else(|e| panic!("{cfg:?}: {e:#?}"));
        let err = max_rel_err(&run_flash(&prog), &want);
        assert!(err < 1e-4, "{cfg:?}: max rel err {err:e}");
    }
}

// ---- protocol violations -------------------------------------------------

const N: usize = 4;

fn tg() -> TileTy {
    TileTy::new(DType::F16, &[N, 8], Space::Threadgroup)
}

fn rows(param: usize, i: Var) -> View {
    View {
        param,
        offset: vec![IdxExpr::scaled(i, N, 0), IdxExpr::lit(0)],
        shape: vec![N, 8],
    }
}

/// One pipe, a producer committing `sent` slots, and one consumer whose
/// loop body is `consume` over `got` iterations.
fn pipeline(
    stages: usize,
    sent: usize,
    got: usize,
    consume: impl Fn(&mut Builder, Var, Var) + 'static,
) -> Program {
    let mut b = Builder::new("t");
    let k = b.param("k", DType::F16, &[64, 8], false);
    let pipe = b.pipe(vec![tg()], stages, 1);
    let (prod, cons) = (pipe.ty.id, pipe.consumers[0]);
    let roles = vec![
        RoleDef {
            name: "producer",
            warps: 1,
            inputs: vec![(prod, Ty::Producer(pipe.ty.clone()))],
            n_results: 0,
            body: Box::new(move |b: &mut Builder, p: &[Var]| {
                b.for_range(0, sent, vec![], vec![], |b, i, _| {
                    let s = b.op("slot", Op::Acquire(p[0]));
                    b.effect(Op::Commit(p[0], vec![rows(k, i)], s));
                    vec![]
                });
                vec![]
            }),
        },
        RoleDef {
            name: "consumer",
            warps: 4,
            inputs: vec![(cons, Ty::Consumer(pipe.ty.clone(), 0))],
            n_results: 0,
            body: Box::new(move |b: &mut Builder, p: &[Var]| {
                b.for_range(0, got, vec![], vec![], |b, i, _| {
                    consume(b, p[0], i);
                    vec![]
                });
                vec![]
            }),
        },
    ];
    b.specialize(vec![pipe], roles);
    b.finish()
}

fn well_behaved(b: &mut Builder, h: Var, _: Var) {
    let s = b.op("kv", Op::Receive(h));
    let x = b.op("x", Op::Exp(Arg::BorrowPart(s, 0)));
    b.drop(x);
    b.effect(Op::Release(h, s));
}

fn one(p: &Program) -> Kind {
    let errs = check(p, &Target::nvidia_hopper()).expect_err("must be rejected");
    assert_eq!(errs.len(), 1, "one mistake, one diagnostic: {errs:#?}");
    errs[0].kind
}

#[test]
fn a_well_behaved_pipeline_passes_and_runs() {
    let p = pipeline(2, 16, 16, well_behaved);
    check(&p, &Target::nvidia_hopper()).expect("well formed");
    let mut t = vec![Tensor::zeros(DType::F16, &[64, 8])];
    run(&p, &mut t).expect("runs");
}

/// The consumer stops one slot short: the producer would block forever on a
/// slot nobody frees. Counted statically from the loop trip counts.
#[test]
fn consumer_receives_fewer_than_the_producer_sends() {
    assert_eq!(one(&pipeline(2, 16, 15, well_behaved)), Kind::Protocol);
}

/// The same bug, unchecked, in the interpreter: reported as a deadlock
/// instead of hanging.
#[test]
fn the_interpreter_reports_the_deadlock_the_checker_prevents() {
    let p = pipeline(2, 16, 12, well_behaved);
    let mut t = vec![Tensor::zeros(DType::F16, &[64, 8])];
    let opts = RunOptions {
        deadlock_after: Duration::from_millis(200),
    };
    let err = run_with(&p, &mut t, opts).unwrap_err();
    assert!(err.contains("deadlock"), "{err}");
}

#[test]
fn share_never_released() {
    let p = pipeline(2, 16, 16, |b, h, _| {
        b.op("kv", Op::Receive(h));
    });
    assert_eq!(one(&p), Kind::Leak);
}

#[test]
fn share_dropped_instead_of_released() {
    let p = pipeline(2, 16, 16, |b, h, _| {
        let s = b.op("kv", Op::Receive(h));
        b.drop(s);
    });
    assert_eq!(one(&p), Kind::Protocol);
}

/// Reading K after handing the slot back: the producer may already be
/// overwriting it. Releasing moves the share, so this is a use-after-move.
#[test]
fn read_after_release() {
    let p = pipeline(2, 16, 16, |b, h, _| {
        let s = b.op("kv", Op::Receive(h));
        b.effect(Op::Release(h, s));
        let x = b.op("x", Op::Exp(Arg::BorrowPart(s, 0)));
        b.drop(x);
    });
    assert_eq!(one(&p), Kind::UseAfterMove);
}

/// Holding every slot of the ring and asking for another: deadlock.
#[test]
fn receive_while_holding_the_whole_ring() {
    let p = pipeline(1, 16, 8, |b, h, _| {
        let a = b.op("a", Op::Receive(h));
        let c = b.op("c", Op::Receive(h));
        b.effect(Op::Release(h, a));
        b.effect(Op::Release(h, c));
    });
    let errs = check(&p, &Target::nvidia_hopper()).unwrap_err();
    assert_eq!(errs[0].kind, Kind::Protocol, "{errs:#?}");
}

/// ...while two in flight on a two-stage ring is fine.
#[test]
fn holding_fewer_than_stages_is_fine() {
    let p = pipeline(3, 16, 8, |b, h, _| {
        let a = b.op("a", Op::Receive(h));
        let c = b.op("c", Op::Receive(h));
        b.effect(Op::Release(h, a));
        b.effect(Op::Release(h, c));
    });
    check(&p, &Target::nvidia_hopper()).expect("fits the ring");
    let mut t = vec![Tensor::zeros(DType::F16, &[64, 8])];
    run(&p, &mut t).expect("runs");
}

/// The producer may not look into a slot: it is not a tile until committed.
#[test]
fn producer_reads_its_slot() {
    let mut b = Builder::new("t");
    let pipe = b.pipe(vec![tg()], 2, 1);
    let (prod, cons) = (pipe.ty.id, pipe.consumers[0]);
    let roles = vec![
        RoleDef {
            name: "producer",
            warps: 1,
            inputs: vec![(prod, Ty::Producer(pipe.ty.clone()))],
            n_results: 0,
            body: Box::new(|b: &mut Builder, p: &[Var]| {
                let s = b.op("slot", Op::Acquire(p[0]));
                b.op("x", Op::Exp(Arg::Borrow(s)));
                vec![]
            }),
        },
        RoleDef {
            name: "consumer",
            warps: 1,
            inputs: vec![(cons, Ty::Consumer(pipe.ty.clone(), 0))],
            n_results: 0,
            body: Box::new(|_: &mut Builder, _: &[Var]| vec![]),
        },
    ];
    b.specialize(vec![pipe], roles);
    let errs = check(&b.finish(), &Target::nvidia_hopper()).unwrap_err();
    assert_eq!(errs[0].kind, Kind::Type, "{errs:#?}");
}

/// A consumer handle nobody owns means an empty barrier nobody arrives on.
#[test]
fn consumer_handle_given_to_no_role() {
    let mut b = Builder::new("t");
    let pipe = b.pipe(vec![tg()], 2, 2);
    let prod = pipe.ty.id;
    let cons0 = pipe.consumers[0];
    let ty = pipe.ty.clone();
    let roles = vec![
        RoleDef {
            name: "producer",
            warps: 1,
            inputs: vec![(prod, Ty::Producer(ty.clone()))],
            n_results: 0,
            body: Box::new(|_: &mut Builder, _: &[Var]| vec![]),
        },
        RoleDef {
            name: "consumer",
            warps: 1,
            inputs: vec![(cons0, Ty::Consumer(ty, 0))],
            n_results: 0,
            body: Box::new(|_: &mut Builder, _: &[Var]| vec![]),
        },
    ];
    b.specialize(vec![pipe], roles);
    assert_eq!(one(&b.finish()), Kind::Leak);
}

/// Two roles cannot share one consumer end: the arrival count would be off.
#[test]
fn consumer_handle_given_to_two_roles() {
    let mut b = Builder::new("t");
    let pipe = b.pipe(vec![tg()], 2, 1);
    let prod = pipe.ty.id;
    let cons = pipe.consumers[0];
    let ty = pipe.ty.clone();
    let consumer = |ty: lex_front::ir::PipeTy| RoleDef {
        name: "consumer",
        warps: 1,
        inputs: vec![(cons, Ty::Consumer(ty, 0))],
        n_results: 0,
        body: Box::new(|_: &mut Builder, _: &[Var]| vec![]),
    };
    let roles = vec![
        RoleDef {
            name: "producer",
            warps: 1,
            inputs: vec![(prod, Ty::Producer(ty.clone()))],
            n_results: 0,
            body: Box::new(|_: &mut Builder, _: &[Var]| vec![]),
        },
        consumer(ty.clone()),
        consumer(ty),
    ];
    b.specialize(vec![pipe], roles);
    assert_eq!(one(&b.finish()), Kind::UseAfterMove);
}

#[test]
fn too_many_warps() {
    let cfg = FlashDecode {
        bq: 1,
        consumers: 32,
        ..hopper_ws()
    };
    let errs = check(&cfg.build().unwrap(), &Target::nvidia_hopper()).unwrap_err();
    assert_eq!(errs.len(), 1, "{errs:#?}");
    assert_eq!(errs[0].kind, Kind::Budget);
}
