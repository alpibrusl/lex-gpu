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
//! TILE_BLESS=1 cargo test -p tile-msl
//! ```

use std::path::PathBuf;

use tile_ir::{DType, Kernel, Target, plan};

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn check(case: &str, kernel: &Kernel) {
    let target = Target::apple_m_series();
    let plan = plan(kernel, &target).unwrap_or_else(|e| panic!("cannot plan {case}: {e}"));
    compare(case, &tile_msl::emit(kernel, &plan, &target));
}

/// Flash-attention decode, a typed tile-front program, lowered to MSL. The
/// lowering is the part of the backend a Linux box cannot otherwise exercise.
#[test]
fn flash_decode_lowered() {
    use tile_front::flash::FlashDecode;
    use tile_ir::Space;
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
    };
    let prog = cfg.build().unwrap();
    let target = Target::apple_m_series();
    tile_front::check(&prog, &target).expect("check");
    let low = tile_msl::program::lower(&prog, &target, 128).expect("lower");
    compare("flash_decode_f16_lowered", &low.source);
}

fn compare(case: &str, got: &str) {
    let got = got.to_string();
    let path = golden_dir().join(format!("{case}.metal"));

    if std::env::var_os("TILE_BLESS").is_some() {
        std::fs::create_dir_all(golden_dir()).unwrap();
        std::fs::write(&path, &got).unwrap();
        return;
    }

    let want = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "missing golden file {}: {e}\nrun with TILE_BLESS=1 to create it",
            path.display()
        )
    });

    if got != want {
        // Line-oriented, because a 60-line shader diffed as one string is
        // unreadable in test output.
        let mut report = format!(
            "emitted MSL for `{case}` does not match {}\n",
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
        report.push_str("\nrun with TILE_BLESS=1 to accept the new output\n");
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
