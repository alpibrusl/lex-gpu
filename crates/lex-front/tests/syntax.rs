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

const GEMM: &str = include_str!("../lx/gemm.lx");

/// The same program, written with the builder: what `gemm.lx` has to be.
fn gemm_by_hand(
    m: usize,
    n: usize,
    k: usize,
    bm: usize,
    bn: usize,
    bk: usize,
) -> lex_front::Program {
    use lex_front::ir::{Arg::Move, BinOp, Builder, IdxExpr, Op, Ty, View};
    use lex_ir::{DType::*, Space::Reg};
    let tile = |dt, s: &[usize]| lex_front::ir::TileTy::new(dt, s, Reg);
    let mut b = Builder::new(&format!("gemm_{m}_{n}_{k}_{bm}_{bn}_{bk}"));
    let a = b.param("a", F16, &[m, k], false);
    let w = b.param("w", F16, &[n, k], false);
    let y = b.param("y", F32, &[m, n], true);
    let pi = b.grid(m / bm);
    let pj = b.grid2(n / bn);
    let zero = b.op("acc", Op::Fill(tile(F32, &[bm, bn]), 0.0));
    let out = b.for_range(
        0,
        k / bk,
        vec![zero],
        vec![Ty::Tile(tile(F32, &[bm, bn]))],
        |b, p, carry| {
            let view = |param, row: usize, rows| View {
                param,
                offset: vec![
                    IdxExpr::scaled(row_var(row, pi, pj), rows, 0),
                    IdxExpr::scaled(p, bk, 0),
                ],
                shape: vec![rows, bk],
            };
            let at = b.op("a", Op::Load(view(a, 0, bm), tile(F16, &[bm, bk])));
            let wt = b.op("w", Op::Load(view(w, 1, bn), tile(F16, &[bn, bk])));
            let mm = b.op("y", Op::MatMulNT(Move(at), Move(wt), F32));
            vec![b.op("y", Op::Binary(BinOp::Add, Move(carry[0]), Move(mm)))]
        },
    );
    b.effect(Op::Store(
        Move(out[0]),
        View {
            param: y,
            offset: vec![IdxExpr::scaled(pi, bm, 0), IdxExpr::scaled(pj, bn, 0)],
            shape: vec![bm, bn],
        },
    ));
    b.finish()
}

/// Which grid index a window's rows follow: `a` by `i`, `w` by `j`.
fn row_var(which: usize, pi: lex_front::ir::Var, pj: lex_front::ir::Var) -> lex_front::ir::Var {
    if which == 0 { pi } else { pj }
}

#[test]
fn loops_and_windows_build_the_program_the_builder_builds() {
    let unit = syntax::parse(GEMM).unwrap_or_else(|e| panic!("{e}"));
    let (m, n, k) = (64usize, 48usize, 96usize);
    let (prog, threads) = unit
        .compile(
            "apple-m-series",
            &[("m", m as f64), ("n", n as f64), ("k", k as f64)],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(threads, 128);
    let want = gemm_by_hand(m, n, k, 16, 16, 32);
    assert_eq!(prog, want, "the surface and the builder disagree");
    lex_front::check(&prog, &lex_ir::Target::apple_m_series())
        .unwrap_or_else(|e| panic!("the parsed gemm: {e:#?}"));
}

#[test]
fn the_parsed_gemm_computes_a_matrix_product() {
    use lex_front::interp::{Tensor, run};
    let unit = syntax::parse(GEMM).unwrap_or_else(|e| panic!("{e}"));
    let (m, n, k) = (32usize, 48usize, 64usize);
    let (prog, _) = unit
        .compile(
            "nvidia-ada",
            &[("m", m as f64), ("n", n as f64), ("k", k as f64)],
        )
        .unwrap_or_else(|e| panic!("{e}"));
    let fill = |len: usize, seed: u32| -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                ((x >> 9) as f32 / (1u32 << 23) as f32) - 0.5
            })
            .collect()
    };
    let (av, wv) = (fill(m * k, 1), fill(n * k, 2));
    let mut t = vec![
        Tensor::new(lex_ir::DType::F16, &[m, k], &av),
        Tensor::new(lex_ir::DType::F16, &[n, k], &wv),
        Tensor::new(lex_ir::DType::F32, &[m, n], &vec![0.0; m * n]),
    ];
    run(&prog, &mut t).unwrap_or_else(|e| panic!("{e:?}"));
    // The reference sees what the kernel sees: the inputs rounded to f16.
    let (ar, wr) = (&t[0].data, &t[1].data);
    for i in 0..m {
        for j in 0..n {
            let want: f64 = (0..k)
                .map(|p| ar[i * k + p] as f64 * wr[j * k + p] as f64)
                .sum();
            let got = t[2].data[i * n + j] as f64;
            assert!((got - want).abs() < 1e-4, "y[{i},{j}] = {got}, want {want}");
        }
    }
}

#[test]
fn a_schedule_that_leaves_a_tile_open_or_invents_one_is_refused() {
    // `tile bm, bn, bk` are the algorithm's holes; a schedule that does not
    // fill them has chosen nothing, and one that sets another name has
    // misspelt one.
    let missing = GEMM.replace(" bk 32 }", " }");
    let err = syntax::parse(&missing).err().expect("a missing tile");
    assert!(err.contains("sets no `bk`"), "{err}");
    let invented = GEMM.replace("bk 32 }", "bk 32 bq 8 }");
    let err = syntax::parse(&invented).err().expect("an unknown key");
    assert!(err.contains("not a schedule key"), "{err}");
}

#[test]
fn a_loop_that_uses_its_accumulator_twice_does_not_check() {
    // `acc + acc * ...` spends the carried tile twice.
    let src = GEMM.replace("yield acc + matmul_nt at wt", "yield acc + acc");
    let unit = syntax::parse(&src).expect("it parses");
    let (prog, _) = unit
        .compile("apple-m-series", &[("m", 16.0), ("n", 16.0), ("k", 32.0)])
        .expect("it builds");
    assert!(
        lex_front::check(&prog, &lex_ir::Target::apple_m_series()).is_err(),
        "a carried tile was used twice and the checker said nothing"
    );
}

#[test]
fn the_parsed_gemm_lowers_for_both_backends() {
    use lex_msl::dialect::{Cuda, Msl};
    let unit = syntax::parse(GEMM).unwrap_or_else(|e| panic!("{e}"));
    for (target, dialect) in [
        (
            lex_ir::Target::apple_m_series(),
            &Msl as &dyn lex_msl::dialect::Dialect,
        ),
        (lex_ir::Target::nvidia_ada(), &Cuda),
    ] {
        let (prog, threads) = unit
            .compile(target.name, &[("m", 64.0), ("n", 64.0), ("k", 128.0)])
            .unwrap_or_else(|e| panic!("{e}"));
        lex_front::check(&prog, &target).unwrap_or_else(|e| panic!("{}: {e:#?}", target.name));
        let l = lex_msl::program::lower_with(&prog, &target, threads, dialect)
            .unwrap_or_else(|e| panic!("{}: {e}", target.name));
        assert_eq!(
            (l.grid, l.grid2),
            (4, 4),
            "{}: one instance a 16x16 tile",
            target.name
        );
    }
}
