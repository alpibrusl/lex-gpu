//! The surface against the Rust it replaces.
//!
//! A parser is easy to believe and hard to trust: it will happily build
//! *a* program. The claim worth testing is that it builds *the* program,
//! so these compare against `lex_front::llama::rmsnorm` rather than
//! against a recorded parse, and compare the lowered MSL rather than a
//! debug print -- two programs can differ in ways a `Debug` impl hides
//! and an emitter does not.

use lex_front::syntax;

const RMSNORM: &str = include_str!("lx/rmsnorm.lx");
const N: usize = 4096;
const EPS: f32 = 1e-5;

#[test]
fn the_surface_builds_the_same_program_as_the_rust() {
    let algo = syntax::parse(RMSNORM).unwrap_or_else(|e| panic!("{e}"));
    let got = algo
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
        let at = x.iter().zip(&y).position(|(p, q)| p != q).unwrap_or(x.len().min(y.len()));
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
    let algo = syntax::parse(&src).expect("it should still parse");
    let prog = algo.build(&[("n", N as f64), ("eps", EPS as f64)]).expect("and build");
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
            .and_then(|a| a.build(&[]).map(|_| ()))
            .unwrap_err();
        assert!(
            got.contains(want),
            "for {src:?}\n  wanted an error mentioning {want:?}\n  got {got:?}"
        );
    }
}
