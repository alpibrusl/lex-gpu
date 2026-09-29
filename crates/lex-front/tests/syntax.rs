//! The surface against the Rust it replaces.
//!
//! A parser is easy to believe and hard to trust: it will happily build
//! *a* program. The claim worth testing is that it builds *the* program,
//! so these compare against `lex_front::llama::rmsnorm` rather than
//! against a recorded parse, and compare the lowered MSL rather than a
//! debug print -- two programs can differ in ways a `Debug` impl hides
//! and an emitter does not.

use lex_front::syntax;

const RMSNORM: &str = include_str!("../lx/rmsnorm.lx");
const N: usize = 4096;
const EPS: f32 = 1e-5;

#[test]
fn the_surface_builds_the_same_program_as_the_rust() {
    let unit = syntax::parse(RMSNORM).unwrap_or_else(|e| panic!("{e}"));
    let got = unit
        .algo
        .build(&[("n", N as f64), ("eps", EPS as f64)])
        .unwrap_or_else(|e| panic!("{e}"));
    let want = lex_front::llama::rmsnorm(N, EPS);

    assert_eq!(got.name, want.name, "the name is part of the program");

    let target = lex_ir::Target::apple_m_series();
    lex_front::check(&got, &target).unwrap_or_else(|e| panic!("the parsed program: {e:#?}"));

    let a = lex_msl::program::lower(&got, &target, 256).expect("lower parsed");
    let b = lex_msl::program::lower(&want, &target, 256).expect("lower rust");
    if a.source != b.source {
        let (x, y): (Vec<_>, Vec<_>) = (a.source.lines().collect(), b.source.lines().collect());
        let at = x
            .iter()
            .zip(&y)
            .position(|(p, q)| p != q)
            .unwrap_or(x.len().min(y.len()));
        panic!(
            "the emitted MSL differs at line {}:\n  parsed: {:?}\n  rust  : {:?}",
            at + 1,
            x.get(at),
            y.get(at)
        );
    }
}

#[test]
fn a_moved_tile_cannot_be_used_again() {
    // The surface's whole reason for a sigil: `x` is spent by the first
    // multiply, so the second use has nothing to take. The checker is
    // what must say so -- if this ever parses *and* checks, the sigil is
    // decorative.
    let src = RMSNORM.replace("let sq = &x * &x", "let sq = x * x");
    let unit = syntax::parse(&src).expect("it should still parse");
    let prog = unit
        .algo
        .build(&[("n", N as f64), ("eps", EPS as f64)])
        .expect("and build");
    let errs = lex_front::check(&prog, &lex_ir::Target::apple_m_series());
    assert!(
        errs.is_err(),
        "using a moved tile twice checked clean -- the linear discipline is not being enforced"
    );
}

#[test]
fn errors_say_where() {
    for (src, want) in [
        ("algo", "expected a name"),
        ("algo f in x: f32[1, 4] { let a = ", "expected"),
        ("algo f in x: q8[1, 4] { }", "not a dtype"),
        ("algo f in x: f32[1, 4] { let a = nope }", "not bound"),
    ] {
        let got = syntax::parse(src)
            .and_then(|u| u.algo.build(&[]).map(|_| ()))
            .unwrap_err();
        assert!(
            got.contains(want),
            "for {src:?}\n  wanted an error mentioning {want:?}\n  got {got:?}"
        );
    }
}

/// One algorithm, two schedules, two backends.
///
/// This is the claim the whole design rests on, and the reason the
/// surface waited for a second target: the algorithm names no machine,
/// each schedule names exactly one, and what comes out has to be what
/// the hand-written Rust produces on both. Not similar to it -- the same
/// text, because a backend that is merely close is a backend with a bug
/// nobody has found yet.
#[test]
fn one_algorithm_two_schedules_two_backends() {
    use lex_msl::dialect::{Cuda, Msl};
    use lex_msl::program::lower_with;

    let unit = syntax::parse(RMSNORM).unwrap_or_else(|e| panic!("{e}"));
    let prog = unit
        .algo
        .build(&[("n", N as f64), ("eps", EPS as f64)])
        .unwrap_or_else(|e| panic!("{e}"));
    let want = lex_front::llama::rmsnorm(N, EPS);

    for (target, dialect) in [
        (
            lex_ir::Target::apple_m_series(),
            &Msl as &dyn lex_msl::dialect::Dialect,
        ),
        (lex_ir::Target::nvidia_ada(), &Cuda),
    ] {
        let sched = unit
            .schedule_for(target.name)
            .unwrap_or_else(|e| panic!("{e}"));
        lex_front::check(&prog, &target).unwrap_or_else(|e| panic!("{}: {e:#?}", target.name));

        let a = lower_with(&prog, &target, sched.threads, dialect)
            .unwrap_or_else(|e| panic!("{}: {e}", target.name));
        let b = lower_with(&want, &target, sched.threads, dialect)
            .unwrap_or_else(|e| panic!("{}: {e}", target.name));
        assert_eq!(
            a.source, b.source,
            "{}: the parsed program and the Rust one emit different source",
            target.name
        );
        // Each backend must have emitted its own language. Comparing
        // the two outputs for inequality does not test this: the targets
        // differ in name and budget, so the text differs even with one
        // dialect wired to both -- which is exactly what happened when I
        // checked, and why this asserts on the dialect's own keywords
        // instead.
        let (want, avoid) = if target.name.starts_with("apple") {
            (
                ["kernel void", "threadgroup"],
                ["__global__", "__syncthreads"],
            )
        } else {
            (
                ["__global__", "__syncthreads"],
                ["kernel void", "threadgroup "],
            )
        };
        for w in want {
            assert!(
                a.source.contains(w),
                "{}: emitted source has no `{w}` in it",
                target.name
            );
        }
        for v in avoid {
            assert!(
                !a.source.contains(v),
                "{}: emitted source contains `{v}`, which belongs to the other backend",
                target.name
            );
        }
    }
}

#[test]
fn a_missing_schedule_is_an_error() {
    let unit = syntax::parse(RMSNORM).expect("parses");
    let err = unit.schedule_for("amd-cdna3").unwrap_err();
    assert!(
        err.contains("no schedule for") && err.contains("apple-m-series"),
        "the error should say what is missing and what is there: {err}"
    );
}

const SILU: &str = include_str!("../lx/silu_mul.lx");

/// A second kernel, which is what says the first was not a coincidence.
///
/// It also exercises the part rmsnorm does not: a grid. The algorithm is
/// written over the whole tensor and the schedule cuts it into pieces of
/// 256, so the partition -- a fact about a machine -- never appears in
/// the algorithm. The Rust equivalent takes `chunk` as an argument and
/// builds the grid itself; both must emit the same text.
#[test]
fn a_grid_comes_from_the_schedule() {
    let unit = syntax::parse(SILU).unwrap_or_else(|e| panic!("{e}"));
    let target = lex_ir::Target::apple_m_series();
    let (got, threads) = unit
        .compile(target.name, &[])
        .unwrap_or_else(|e| panic!("{e}"));
    let want = lex_front::llama::silu_mul(16384, 256, lex_ir::DType::F32).expect("rust silu_mul");

    assert_eq!(got.name, want.name);
    lex_front::check(&got, &target).unwrap_or_else(|e| panic!("{e:#?}"));

    let a = lex_msl::program::lower(&got, &target, threads).expect("lower parsed");
    let b = lex_msl::program::lower(&want, &target, threads).expect("lower rust");
    assert_eq!(a.grid, b.grid, "the grid the schedule asked for");
    assert!(
        a.grid > 1,
        "a chunked algo should have more than one instance"
    );
    if a.source != b.source {
        let (x, y): (Vec<_>, Vec<_>) = (a.source.lines().collect(), b.source.lines().collect());
        let at = x.iter().zip(&y).position(|(p, q)| p != q).unwrap_or(0);
        panic!(
            "MSL differs at line {}:\n  parsed: {:?}\n  rust  : {:?}",
            at + 1,
            x.get(at),
            y.get(at)
        );
    }
}

#[test]
fn a_chunk_that_does_not_divide_is_rejected() {
    let src = SILU.replace("chunk 256", "chunk 300");
    let unit = syntax::parse(&src).expect("parses");
    let err = unit.compile("apple-m-series", &[]).unwrap_err();
    assert!(err.contains("does not divide"), "got {err}");
}
