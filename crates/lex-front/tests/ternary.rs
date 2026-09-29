//! Prism's two ternary GGUF packings, against real Bonsai 2 27B weights.
//!
//! Neither `PTQ1_0` nor `PQ2_0` has a public specification -- the reference
//! layouts live only in PrismML's MIT llama.cpp fork, which this repository
//! cannot copy from. Both were derived from the published weight files
//! instead; `docs/ternary.md` records how.
//!
//! That makes the test below the load-bearing one. `PQ2_0` is described in
//! public prose and the file confirms it (no 2-bit slot ever holds 3), while
//! `PTQ1_0` was reverse engineered. Both files quantise the *same* model --
//! their FP16 scales are bit-identical block for block -- so the simple
//! packing is ground truth for the derived one. The fixture is 64 real
//! blocks of `output.weight` taken from each file.

use lex_front::ir::{Arg, Builder, IdxExpr, Op, Program, TileTy, Trits, View};
use lex_front::{check, interp::Tensor, interp::run};
use lex_ir::{DType, Space, Target};

const BLOCKS: usize = 64;
const GROUP: usize = 128;
const FIXTURE: &[u8] = include_bytes!("data/bonsai_blocks.bin");

/// `PTQ1_0` block `i`: 26 code bytes then an f16 scale.
fn ptq1(i: usize) -> &'static [u8] {
    &FIXTURE[i * 28..(i + 1) * 28]
}

/// `PQ2_0` block `i`: an f16 scale then 32 code bytes.
fn pq2(i: usize) -> &'static [u8] {
    let base = 28 * BLOCKS;
    &FIXTURE[base + i * 34..base + (i + 1) * 34]
}

fn f16(b: [u8; 2]) -> f32 {
    let h = u16::from_le_bytes(b) as u32;
    let (s, e, m) = (h >> 15, (h >> 10) & 0x1F, h & 0x3FF);
    let bits = match e {
        0 if m == 0 => s << 31,
        0 => return (if s == 1 { -1.0 } else { 1.0 }) * (m as f32) * 2f32.powi(-24),
        0x1F => (s << 31) | 0x7F80_0000 | (m << 13),
        _ => (s << 31) | ((e + 112) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

#[test]
fn the_derived_packing_agrees_with_the_documented_one() {
    let mut checked = 0;
    for i in 0..BLOCKS {
        let (dense, slots) = (ptq1(i), pq2(i));
        assert_eq!(
            [dense[26], dense[27]],
            [slots[0], slots[1]],
            "block {i}: the two files disagree on the scale, so they are not \
             the same quantisation and one cannot check the other"
        );
        for col in 0..GROUP {
            let a = Trits::Dense.code(&dense[..26], col);
            let b = Trits::Slots2.code(&slots[2..], col);
            assert_eq!(a, b, "block {i} column {col}");
            assert!(a < 3, "block {i} column {col}: {a} is not a trit");
            checked += 1;
        }
    }
    assert_eq!(checked, BLOCKS * GROUP);
}

/// The fixture has to actually exercise what the packing gets wrong, or the
/// test above passes for the wrong reason.
#[test]
fn the_fixture_would_catch_the_hypotheses_that_were_wrong() {
    // A byte over 242 rules out a plain five-trit base-3 quintet, which is
    // what the public prose about this format suggests.
    let over = (0..BLOCKS)
        .flat_map(|i| ptq1(i)[..26].iter())
        .filter(|&&b| b > 242)
        .count();
    assert!(
        over > 0,
        "no payload byte above 242; plain base-3 not excluded"
    );

    // All three codes present, so a decode that collapsed two would fail.
    let mut seen = [0usize; 3];
    for i in 0..BLOCKS {
        for col in 0..GROUP {
            seen[Trits::Dense.code(&ptq1(i)[..26], col) as usize] += 1;
        }
    }
    assert!(
        seen.iter().all(|&n| n > 0),
        "codes not all present: {seen:?}"
    );

    // Reading the lanes sequentially -- five consecutive trits per byte --
    // is the other natural guess. It must disagree, or the lane map is not
    // being tested.
    let wrong = (0..BLOCKS)
        .flat_map(|i| {
            (0..GROUP).map(move |col| {
                let (byte, step) = (col / 5, col % 5);
                let mut v = ptq1(i)[..26][byte.min(25)] as u32;
                let mut code = 0;
                for _ in 0..=step {
                    v *= 3;
                    code = (v >> 8) as u8;
                    v &= 0xff;
                }
                code != Trits::Dense.code(&ptq1(i)[..26], col)
            })
        })
        .filter(|&d| d)
        .count();
    assert!(
        wrong > BLOCKS * GROUP / 4,
        "a sequential lane map disagrees on only {wrong} of {} values",
        BLOCKS * GROUP
    );
}

fn program(trits: Trits, cols: usize) -> Program {
    let groups = cols / GROUP;
    let nb = groups * trits.bytes();
    let mut b = Builder::new("ternary");
    let pq = b.param("q", DType::I8, &[1, nb], false);
    let ps = b.param("s", DType::F32, &[1, groups], false);
    let py = b.param("y", DType::F32, &[1, cols], true);
    let whole = |p, n| View {
        param: p,
        offset: vec![IdxExpr::lit(0), IdxExpr::lit(0)],
        shape: vec![1, n],
    };
    let q = b.op(
        "q",
        Op::Load(whole(pq, nb), TileTy::new(DType::I8, &[1, nb], Space::Reg)),
    );
    let s = b.op(
        "s",
        Op::Load(
            whole(ps, groups),
            TileTy::new(DType::F32, &[1, groups], Space::Reg),
        ),
    );
    let w = b.op(
        "w",
        Op::DequantTernary(Arg::Move(q), Arg::Move(s), GROUP, trits),
    );
    b.effect(Op::Store(Arg::Move(w), whole(py, cols)));
    b.finish()
}

/// Inputs and the expected weights for `blocks` blocks of the fixture.
fn case(trits: Trits, blocks: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (mut q, mut s, mut want) = (vec![], vec![], vec![]);
    for i in 0..blocks {
        let (codes, scale) = match trits {
            Trits::Dense => (&ptq1(i)[..26], f16([ptq1(i)[26], ptq1(i)[27]])),
            Trits::Slots2 => (&pq2(i)[2..], f16([pq2(i)[0], pq2(i)[1]])),
        };
        q.extend(codes.iter().map(|&b| b as i8 as f32));
        s.push(scale);
        want.extend((0..GROUP).map(|c| (trits.code(codes, c) as f32 - 1.0) * scale));
    }
    (q, s, want)
}

#[test]
fn the_dequant_op_reads_both_packings() {
    for trits in [Trits::Dense, Trits::Slots2] {
        let blocks = BLOCKS;
        let cols = blocks * GROUP;
        let prog = program(trits, cols);
        check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{trits:?}: {e:#?}"));
        let (q, s, want) = case(trits, blocks);
        let mut t = vec![
            Tensor::new(DType::I8, &[1, q.len()], &q),
            Tensor::new(DType::F32, &[1, s.len()], &s),
            Tensor::zeros(DType::F32, &[1, cols]),
        ];
        run(&prog, &mut t).unwrap_or_else(|e| panic!("{trits:?}: {e}"));
        assert_eq!(t[2].data, want, "{trits:?}");
        // Non-trivial output: a dequant that returned zeros would pass an
        // all-zero comparison.
        assert!(want.iter().any(|&x| x != 0.0), "{trits:?}: all zero");
    }
}
