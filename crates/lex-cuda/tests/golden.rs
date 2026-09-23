//! Golden-file tests for the CUDA emitter.
//!
//! The emitted source is the backend's real output and the only thing a
//! reviewer can read. A diff here is either an improvement someone should
//! look at or a regression — and on a Mac, with no NVIDIA GPU and no CUDA
//! toolkit, it is the strongest signal available without Docker.
//!
//! The layer above this is `scripts/cuda_check.sh`, which compiles the same
//! output with `nvcc` and assembles it with `ptxas` in a container. Neither
//! needs a GPU. Between them, the only thing left for a real device is
//! whether the kernel *runs*.
//!
//! Regenerate after an intentional change with:
//!
//! ```text
//! LEX_BLESS=1 cargo test -p lex-cuda
//! ```

use std::path::PathBuf;

use lex_ir::{DType, Kernel, Target, plan};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn check(case: &str, kernel: &Kernel) {
    let target = Target::nvidia_ada();
    let plan = plan(kernel, &target).unwrap_or_else(|e| panic!("cannot plan {case}: {e}"));
    compare(case, &lex_cuda::emit(kernel, &plan, &target));
}

#[test]
fn copy_f32() {
    check("copy_f32", &Kernel::copy(DType::F32, 4096));
}

#[test]
fn copy_f16() {
    check("copy_f16", &Kernel::copy(DType::F16, 4096));
}

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

/// Hopper differs from Ada in shared-memory budget and in nothing else that
/// reaches the source. Keeping a golden for it is how a change that silently
/// specialises the emitter to one NVIDIA generation gets noticed.
#[test]
fn rmsnorm_f32_hopper() {
    let target = Target::nvidia_hopper();
    let kernel = Kernel::rmsnorm(DType::F32, 8, 4096, 1e-5);
    let plan = plan(&kernel, &target).expect("plan");
    compare(
        "rmsnorm_f32_hopper",
        &lex_cuda::emit(&kernel, &plan, &target),
    );
}

fn compare(case: &str, got: &str) {
    let got = got.to_string();
    let path = golden_dir().join(format!("{case}.cu"));

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
        // Line-oriented, because a 60-line kernel diffed as one string is
        // unreadable in test output.
        let mut report = format!(
            "emitted CUDA for `{case}` does not match {}\n",
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
