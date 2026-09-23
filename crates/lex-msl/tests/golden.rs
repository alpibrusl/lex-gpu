//! Golden-file tests for the MSL emitter.
//!
//! These exist because the emitted source is the backend's real output and the
//! only thing a reviewer can actually read. A diff here is either an
//! improvement someone should look at, or a regression -- and on a host with
//! no Metal toolchain (CI, or a Linux box) it is the strongest signal
//! available.
//!
//! Regenerate after an intentional change with:
//!
//! ```text
//! LEX_BLESS=1 cargo test -p lex-msl
//! ```

use std::path::PathBuf;

use lex_ir::{DType, Kernel, Target, plan};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn check(case: &str, kernel: &Kernel) {
    let target = Target::apple_m_series();
    let plan = plan(kernel, &target).unwrap_or_else(|e| panic!("cannot plan {case}: {e}"));
    compare(case, &lex_msl::emit(kernel, &plan, &target));
}

/// Flash-attention decode, a typed lex-front program, lowered to MSL. The
/// lowering is the part of the backend a Linux box cannot otherwise exercise.
#[test]
fn flash_decode_lowered() {
    use lex_front::flash::FlashDecode;
    use lex_ir::Space;
    let cfg = FlashDecode {
        q_rows: 8,
        d: 64,
        seq: 64,
        bq: 4,
        bk: 16,
        stages: 2,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 0,
        heads: 2,
        kv_cap: 0,
    };
    let prog = cfg.build().unwrap();
    let target = Target::apple_m_series();
    lex_front::check(&prog, &target).expect("check");
    let low = lex_msl::program::lower(&prog, &target, 128).expect("lower");
    compare("flash_decode_f16_lowered", &low.source);
}

fn compare(case: &str, got: &str) {
    compare_ext(case, got, "metal")
}

fn compare_ext(case: &str, got: &str, ext: &str) {
    let got = got.to_string();
    let path = golden_dir().join(format!("{case}.{ext}"));

    if std::env::var_os("LEX_BLESS").is_some() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&path, &got).unwrap();
        return;
    }

    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden file {}: {e}\nrun with LEX_BLESS=1 to create it",
            path.display()
        )
    });

    if got != want {
        // Line-oriented, because a 60-line shader diffed as one string is
        // unreadable in test output.
        let mut report = format!(
            "emitted source for `{case}` does not match {}\n",
            path.display()
        );
        for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
            if g != w {
                report.push_str(&format!(
                    "  line {}:\n    want: {w}\n    got : {g}\n",
                    i + 1
                ));
            }
        }
        if got.lines().count() != want.lines().count() {
            report.push_str(&format!(
                "  line count: want {}, got {}\n",
                want.lines().count(),
                got.lines().count()
            ));
        }
        report.push_str("\nrun with LEX_BLESS=1 to accept the new output\n");
        panic!("{report}");
    }
}

#[test]
fn copy_f32() {
    check("copy_f32", &Kernel::copy(DType::F32, 4096));
}

#[test]
fn copy_f16() {
    check("copy_f16", &Kernel::copy(DType::F16, 4096));
}

/// A tail that does not divide evenly into threadgroups: the emitted bounds
/// check is load-bearing here.
#[test]
fn copy_f32_ragged_tail() {
    check("copy_f32_ragged", &Kernel::copy(DType::F32, 4100));
}

/// Llama-3-8B's hidden size, the shape that actually matters.
#[test]
fn rmsnorm_f32_4096() {
    check(
        "rmsnorm_f32_4096",
        &Kernel::rmsnorm(DType::F32, 8, 4096, 1e-5),
    );
}

#[test]
fn rmsnorm_f16_4096() {
    check(
        "rmsnorm_f16_4096",
        &Kernel::rmsnorm(DType::F16, 8, 4096, 1e-5),
    );
}

/// A row narrow enough to fit in a single simdgroup, which collapses the
/// second reduction stage to a one-element array. Easy to break, silently.
#[test]
fn rmsnorm_f32_narrow() {
    check(
        "rmsnorm_f32_narrow",
        &Kernel::rmsnorm(DType::F32, 8, 128, 1e-5),
    );
}

/// The same typed program, lowered through the same `lower_with`, to CUDA.
///
/// This is the claim the whole design rests on, reduced to a file someone
/// can read: one program, two targets, no per-target kernel rewrite. The
/// Metal goldens beside it come from the identical lowering.
///
/// `scripts/cuda_check.sh crates/lex-msl/tests/golden/*.cu` compiles these
/// with nvcc and assembles them for sm_89 — no GPU, and not in CI, because
/// CI has no container runtime. The golden is what CI can check.
#[test]
fn rmsnorm_rows_lowered_to_cuda() {
    use lex_msl::dialect::Cuda;
    let prog = lex_front::llama::rmsnorm_rows(4, 4096, 1e-5, None, DType::F32);
    let target = Target::nvidia_ada();
    let l = lex_msl::program::lower_with(&prog, &target, 256, &Cuda).expect("lower");
    compare_ext("rmsnorm_4x4096_f32_cuda", &l.source, "cu");
}

/// Narrowing to f16 is where the two languages stop agreeing on grammar:
/// Metal casts with `half(x)`, CUDA calls `__float2half(x)`. A dialect that
/// was copied and not read would emit a cast CUDA silently accepts as
/// something else.
#[test]
fn a_narrowing_store_lowered_to_cuda() {
    use lex_msl::dialect::Cuda;
    let prog = lex_front::llama::rmsnorm_rows(4, 4096, 1e-5, None, DType::F16);
    let target = Target::nvidia_ada();
    let l = lex_msl::program::lower_with(&prog, &target, 256, &Cuda).expect("lower");
    assert!(
        l.source.contains("__float2half"),
        "no CUDA narrowing in:\n{}",
        l.source
    );
    compare_ext("rmsnorm_4x4096_f16_cuda", &l.source, "cu");
}

/// The kernel that is 94% of a decode step, lowered to CUDA: an NVFP4
/// dequantisation fused into a matvec.
///
/// The decode itself is target-independent and has to be. Dropping an E2M1
/// code's magnitude bits at the bottom of half's exponent field returns the
/// value exactly, times 2^-14, subnormals included — a property of IEEE
/// half, not of Metal. Only the spelling of the reinterpretation differs:
/// `as_type<half2>` against a four-byte `memcpy`, which is the portable
/// one, since a pointer cast between same-sized types is undefined
/// behaviour that nvcc may miscompile.
#[test]
fn nvfp4_matvec_lowered_to_cuda() {
    use lex_front::llama::{QLayout, matvec_q};
    use lex_msl::dialect::Cuda;
    let prog = matvec_q(5120, 17408, 8, 5120, QLayout::NVFP4, false).expect("program");
    let target = Target::nvidia_ada();
    let l = lex_msl::program::lower_with(&prog, &target, 256, &Cuda).expect("lower");
    // The fast arithmetic decode, not the lane-table gather.
    assert!(l.source.contains("fp4_pair"), "no NVFP4 decode emitted");
    assert!(
        l.source.contains("__half22float2"),
        "the decode did not reach CUDA spelling:\n{}",
        l.source
    );
    compare_ext("matvec_nvfp4_17408x5120_cuda", &l.source, "cu");
}

/// Nothing the lowering emits may be declared and unused, or used and
/// undeclared.
///
/// This exists because a change that gated `fp4_lane` on whether the body
/// actually used it removed the *declaration* as well, and every test in
/// this repository still passed: the fused path never uses it, and the
/// path that does is reached only with `LEX_NO_LAZY=1`, which nothing
/// exercises. The bug was visible solely as an nvcc warning about an
/// unused variable in a CUDA golden.
///
/// So the invariant is written down rather than left to a warning on one
/// of the two backends.
#[test]
fn nothing_is_declared_unused_or_used_undeclared() {
    use lex_front::llama::{QLayout, matmul_q, matvec_q};
    use lex_msl::dialect::Cuda;
    use lex_msl::program::lower_with;

    let progs = [
        matvec_q(5120, 17408, 8, 5120, QLayout::NVFP4, false).expect("matvec"),
        matvec_q(5120, 17408, 8, 5120, QLayout::NVFP4, true).expect("matvec res"),
        matmul_q(4, 5120, 17408, 32, 5120, QLayout::NVFP4, false).expect("matmul"),
        matvec_q(4096, 4096, 8, 4096, QLayout::Q4_K, false).expect("q4k"),
    ];
    let targets = [Target::apple_m_series(), Target::nvidia_ada()];
    for p in &progs {
        for t in &targets {
            for (dialect, name) in [
                (
                    &lex_msl::dialect::Msl as &dyn lex_msl::dialect::Dialect,
                    "msl",
                ),
                (&Cuda as &dyn lex_msl::dialect::Dialect, "cuda"),
            ] {
                let l = lower_with(p, t, 256, dialect).expect("lower");
                for var in ["fp4_lane", "gid2", "scratch", "arena"] {
                    let declared = l.source.contains(&format!("{var} ="))
                        || l.source.contains(&format!("{var}["));
                    let mentions = l.source.matches(var).count();
                    assert!(
                        mentions == 0 || declared,
                        "`{var}` used but never declared in {} for {name}/{}",
                        p.name,
                        t.name
                    );
                }
            }
        }
    }
}
