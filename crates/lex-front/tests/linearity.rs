//! Every rule the checker enforces, shown rejecting the bug it exists for.
//! Each case is one mistake and must produce exactly one diagnostic.

use lex_front::ir::{Arg, BinOp, Builder, IdxExpr, Op, TileTy, Ty, View};
use lex_front::{Kind, Program, check};
use lex_ir::{DType, Space, Target};

const N: usize = 4;
const D: usize = 8;
const SEQ: usize = 32;

fn tg() -> TileTy {
    TileTy::new(DType::F16, &[N, D], Space::Threadgroup)
}

fn reg(dt: DType) -> TileTy {
    TileTy::new(dt, &[N, D], Space::Reg)
}

fn rows(param: usize, start: IdxExpr) -> View {
    View {
        param,
        offset: vec![start, IdxExpr::lit(0)],
        shape: vec![N, D],
    }
}

/// `k: f16[SEQ, D]` in, `o: f32[N, D]` out.
fn kernel() -> (Builder, usize, usize) {
    let mut b = Builder::new("t");
    let k = b.param("k", DType::F16, &[SEQ, D], false);
    let o = b.param("o", DType::F32, &[N, D], true);
    (b, k, o)
}

fn one(p: &Program) -> Kind {
    let errs = check(p, &Target::apple_m_series()).expect_err("must be rejected");
    assert_eq!(errs.len(), 1, "one mistake, one diagnostic: {errs:#?}");
    errs[0].kind
}

#[test]
fn a_well_formed_kernel_passes() {
    let (mut b, k, o) = kernel();
    let buf = b.op("buf", Op::Alloc(tg()));
    let f = b.op("f", Op::CopyAsync(rows(k, IdxExpr::lit(0)), buf));
    let t = b.op("t", Op::Wait(f));
    let x = b.op("x", Op::Convert(Arg::Borrow(t), DType::F32));
    b.effect(Op::Store(Arg::Move(x), rows(o, IdxExpr::lit(0))));
    b.drop(t);
    check(&b.finish(), &Target::apple_m_series()).expect("well formed");
}

#[test]
fn leaked_buffer() {
    let (mut b, _, _) = kernel();
    b.op("buf", Op::Alloc(tg()));
    assert_eq!(one(&b.finish()), Kind::Leak);
}

#[test]
fn double_wait() {
    let (mut b, k, _) = kernel();
    let buf = b.op("buf", Op::Alloc(tg()));
    let f = b.op("f", Op::CopyAsync(rows(k, IdxExpr::lit(0)), buf));
    let t = b.op("t", Op::Wait(f));
    b.drop(t);
    b.op("t2", Op::Wait(f));
    assert_eq!(one(&b.finish()), Kind::UseAfterMove);
}

/// Reading a tile whose copy has not landed: the future is not a tile.
#[test]
fn read_before_wait() {
    let (mut b, k, _) = kernel();
    let buf = b.op("buf", Op::Alloc(tg()));
    let f = b.op("f", Op::CopyAsync(rows(k, IdxExpr::lit(0)), buf));
    b.op("x", Op::Exp(Arg::Borrow(f)));
    assert_eq!(one(&b.finish()), Kind::Type);
}

/// The classic double-buffering race: prefetching into the buffer that is
/// still being computed on. Issuing the copy moves the buffer, so the read
/// that follows is a use-after-move.
#[test]
fn prefetch_into_the_buffer_being_read() {
    let (mut b, k, _) = kernel();
    let buf = b.op("buf", Op::Alloc(tg()));
    let f = b.op("f", Op::CopyAsync(rows(k, IdxExpr::lit(0)), buf));
    let cur = b.op("cur", Op::Wait(f));
    let next = b.op("next", Op::CopyAsync(rows(k, IdxExpr::lit(N)), cur));
    b.op("x", Op::Exp(Arg::Borrow(cur)));
    let _ = next;
    let errs = check(&b.finish(), &Target::apple_m_series()).unwrap_err();
    assert_eq!(errs[0].kind, Kind::UseAfterMove, "{errs:#?}");
}

/// A loop body consuming a value from outside would consume it every
/// iteration. K reuse must be a borrow.
#[test]
fn loop_consumes_outer_value() {
    let (mut b, k, _) = kernel();
    let kt = b.op("k", Op::Load(rows(k, IdxExpr::lit(0)), reg(DType::F16)));
    b.for_range(0, 4, vec![], vec![], |b, _, _| {
        b.drop(kt);
        vec![]
    });
    assert_eq!(one(&b.finish()), Kind::ConsumeOuter);
}

#[test]
fn loop_borrowing_outer_value_is_fine() {
    let (mut b, k, _) = kernel();
    let kt = b.op("k", Op::Load(rows(k, IdxExpr::lit(0)), reg(DType::F16)));
    b.for_range(0, 4, vec![], vec![], |b, _, _| {
        let x = b.op("x", Op::Exp(Arg::Borrow(kt)));
        b.drop(x);
        vec![]
    });
    b.drop(kt);
    check(&b.finish(), &Target::apple_m_series()).expect("borrow in loop");
}

/// A double buffer only checks if the rotated carry keeps its type.
#[test]
fn loop_carry_changes_type() {
    let (mut b, _, _) = kernel();
    let acc = b.op("acc", Op::Fill(reg(DType::F32), 0.0));
    let out = b.for_range(
        0,
        4,
        vec![acc],
        vec![Ty::Tile(reg(DType::F32))],
        |b, _, p| vec![b.op("n", Op::Convert(Arg::Move(p[0]), DType::F16))],
    );
    b.drop(out[0]);
    assert_eq!(one(&b.finish()), Kind::Carry);
}

#[test]
fn loop_parameter_neither_consumed_nor_yielded() {
    let (mut b, _, _) = kernel();
    let acc = b.op("acc", Op::Fill(reg(DType::F32), 0.0));
    let out = b.for_range(
        0,
        4,
        vec![acc],
        vec![Ty::Tile(reg(DType::F32))],
        |b, _, _| vec![b.op("fresh", Op::Fill(reg(DType::F32), 1.0))],
    );
    b.drop(out[0]);
    assert_eq!(one(&b.finish()), Kind::Leak);
}

#[test]
fn moved_and_borrowed_by_one_op() {
    let (mut b, _, _) = kernel();
    let a = b.op("a", Op::Fill(reg(DType::F32), 1.0));
    let c = b.op("c", Op::Binary(BinOp::Add, Arg::Move(a), Arg::Borrow(a)));
    b.drop(c);
    assert_eq!(one(&b.finish()), Kind::MovedWhileBorrowed);
}

#[test]
fn implicit_narrowing_on_store() {
    let mut b = Builder::new("t");
    let o = b.param("o", DType::F16, &[N, D], true);
    let x = b.op("x", Op::Fill(reg(DType::F32), 1.0));
    b.effect(Op::Store(Arg::Move(x), rows(o, IdxExpr::lit(0))));
    assert_eq!(one(&b.finish()), Kind::Narrowing);
}

#[test]
fn implicit_narrowing_in_the_accumulator() {
    let (mut b, _, _) = kernel();
    let a = b.op("a", Op::Fill(reg(DType::F32), 1.0));
    let c = b.op(
        "c",
        Op::MatMulNT(Arg::Borrow(a), Arg::Borrow(a), DType::F16),
    );
    b.drop(a);
    let _ = c;
    let errs = check(&b.finish(), &Target::apple_m_series()).unwrap_err();
    assert_eq!(errs[0].kind, Kind::Narrowing, "{errs:#?}");
}

/// The prefetch-past-the-end bug: issuing block i+1 on the last iteration.
/// Caught statically from the loop range, not at run time.
#[test]
fn prefetch_runs_off_the_end() {
    let (mut b, k, _) = kernel();
    b.for_range(0, SEQ / N, vec![], vec![], |b, i, _| {
        let t = b.op(
            "t",
            Op::Load(rows(k, IdxExpr::scaled(i, N, N)), reg(DType::F16)),
        );
        b.drop(t);
        vec![]
    });
    assert_eq!(one(&b.finish()), Kind::Bounds);
}

#[test]
fn async_copy_into_registers() {
    let (mut b, k, _) = kernel();
    let buf = b.op("buf", Op::Alloc(reg(DType::F16)));
    let f = b.op("f", Op::CopyAsync(rows(k, IdxExpr::lit(0)), buf));
    let t = b.op("t", Op::Wait(f));
    b.drop(t);
    assert_eq!(one(&b.finish()), Kind::Type);
}

/// An in-flight copy pins its buffer: two futures and a live tile are three
/// buffers' worth of threadgroup memory, even though none is readable.
#[test]
fn futures_count_against_the_budget() {
    let (mut b, k, _) = kernel();
    let tiny = Target {
        max_threadgroup_bytes: 2 * tg().bytes(),
        ..Target::apple_m_series()
    };
    let mut live = vec![];
    for blk in 0..3 {
        let buf = b.op("buf", Op::Alloc(tg()));
        live.push(b.op("f", Op::CopyAsync(rows(k, IdxExpr::lit(blk * N)), buf)));
    }
    for f in live {
        let t = b.op("t", Op::Wait(f));
        b.drop(t);
    }
    let errs = check(&b.finish(), &tiny).unwrap_err();
    assert_eq!(errs.len(), 1);
    assert_eq!(errs[0].kind, Kind::Budget);
}
