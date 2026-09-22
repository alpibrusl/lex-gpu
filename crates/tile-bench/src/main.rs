//! P0 harness.
//!
//! It answers one question: does a kernel this compiler generated move memory
//! as fast as the machine can move memory at all? `copy` establishes the
//! ceiling, `rmsnorm` is scored against it, and the reference interpreter says
//! whether the answer was also correct.
//!
//! Off macOS it still emits and verifies -- only the dispatch needs a device.

use std::process::ExitCode;

#[cfg(target_os = "macos")]
use tile_ir::DType;
use tile_ir::{Kernel, Target, plan, reference};

mod args;
use args::Args;

#[cfg(target_os = "macos")]
/// Rows used for correctness checks. Small, because the CPU reference is
/// single-threaded and a wrong kernel is wrong on 64 rows too.
const VERIFY_ROWS: usize = 64;

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(Some(a)) => a,
        Ok(None) => return ExitCode::SUCCESS, // --help
        Err(e) => {
            eprintln!("error: {e}\n\n{}", Args::usage());
            return ExitCode::FAILURE;
        }
    };

    let target = Target::apple_m_series();
    let copy = Kernel::copy(args.dtype, args.copy_elems());
    let norm = Kernel::rmsnorm(args.dtype, args.rows, args.cols, args.eps);

    if args.emit {
        for k in [&copy, &norm] {
            match plan(k, &target) {
                Ok(p) => println!("{}", tile_msl::emit(k, &p, &target)),
                Err(e) => {
                    eprintln!("cannot plan {}: {e}", k.name);
                    return ExitCode::FAILURE;
                }
            }
        }
        return ExitCode::SUCCESS;
    }

    // Planning is target-dependent but device-independent: it fails or succeeds
    // identically on a Linux CI box and on the Mac Studio.
    for k in [&copy, &norm] {
        if let Err(e) = plan(k, &target) {
            eprintln!("cannot plan {}: {e}", k.name);
            return ExitCode::FAILURE;
        }
    }

    if !reference_self_test() {
        return ExitCode::FAILURE;
    }

    run_device(&args, &target)
}

/// Sanity-check the reference against a case with a known closed form, so that
/// a GPU/reference mismatch is never blamed on the wrong side.
fn reference_self_test() -> bool {
    let (rows, cols) = (2, 128);
    let x = vec![2.0f32; rows * cols];
    let w = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows * cols];
    reference::rmsnorm_f32(&x, &w, &mut y, rows, cols, 0.0);
    // rms of a constant row is that constant, so the output is all ones.
    let ok = y.iter().all(|v| (v - 1.0).abs() < 1e-6);
    if !ok {
        eprintln!("reference self-test failed: rmsnorm of a constant row is not 1");
    }
    ok
}

#[cfg(not(target_os = "macos"))]
fn run_device(_args: &Args, _target: &Target) -> ExitCode {
    println!("planning and reference checks passed.");
    println!();
    println!("This host has no Metal device, so nothing was dispatched.");
    println!("Run `cargo run --release -p tile-bench` on the Mac Studio for the numbers,");
    println!("or `--emit` here to print the generated MSL.");
    ExitCode::SUCCESS
}

#[cfg(target_os = "macos")]
fn run_device(args: &Args, target: &Target) -> ExitCode {
    use tile_metal::Gpu;

    let gpu = match Gpu::open() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let info = gpu.info();
    println!(
        "device : {} ({}, {} B threadgroup, {:.1} GB recommended working set)",
        info.name,
        if info.unified_memory {
            "unified memory"
        } else {
            "discrete"
        },
        info.max_threadgroup_bytes,
        info.recommended_working_set_bytes as f64 / 1e9,
    );
    println!(
        "target : {} (simd {}, {} threads/tg, {} B threadgroup)",
        target.name,
        target.simd_width,
        target.max_threads_per_threadgroup,
        target.max_threadgroup_bytes,
    );
    if info.max_threadgroup_bytes < target.max_threadgroup_bytes {
        eprintln!(
            "warning: the target table claims {} B of threadgroup memory but this device reports {} B",
            target.max_threadgroup_bytes, info.max_threadgroup_bytes
        );
    }
    println!();

    let correct = match args.dtype {
        DType::F32 => verify_f32(&gpu, args, target),
        DType::F16 => verify_f16(&gpu, args, target),
    };
    let correct = match correct {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let perf = match measure(&gpu, args, target) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    println!(
        "{:<14} {:>12} {:>12} {:>10} {:>12}",
        "kernel", "ideal bytes", "time", "GB/s", "% of copy"
    );
    for m in &perf {
        println!(
            "{:<14} {:>12} {:>12} {:>10.1} {:>11.1}%",
            m.name,
            human_bytes(m.bytes),
            human_time(m.seconds),
            m.gb_per_s(),
            100.0 * m.gb_per_s() / perf[0].gb_per_s(),
        );
    }
    println!();

    let ratio = perf[1].gb_per_s() / perf[0].gb_per_s();
    let fast_enough = ratio >= 0.90;
    println!(
        "exit test : rmsnorm at {:.1}% of the copy ceiling (need >= 90.0%)  {}",
        100.0 * ratio,
        pass(fast_enough),
    );
    println!(
        "            max rel err {:.2e} (tolerance {:.0e})  {}",
        correct.err,
        correct.tol,
        pass(correct.ok()),
    );

    // A virtualised GPU reaches a small fraction of the bandwidth of the
    // silicon it runs on, and it throttles both kernels equally -- so the ratio
    // passes while meaning nothing. Say so, or a green CI run on a hosted macOS
    // runner reads as P0 being finished.
    if !is_real_gpu(&info.name) {
        println!();
        println!(
            "NOTE: `{}` is a virtualised GPU. The correctness result above is real;",
            info.name
        );
        println!(
            "      the {:.1} GB/s ceiling is not, and neither is the ratio measured against it.",
            perf[0].gb_per_s()
        );
        println!("      P0's exit test needs a physical M-series device.");
    }

    if fast_enough && correct.ok() {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Whether the Metal device is physical silicon rather than a VM's
/// paravirtualised adapter (what hosted CI runners expose).
#[cfg(target_os = "macos")]
fn is_real_gpu(device_name: &str) -> bool {
    !device_name.contains("Paravirtual") && !device_name.contains("Virtual")
}

#[cfg(target_os = "macos")]
fn pass(ok: bool) -> &'static str {
    if ok { "PASS" } else { "FAIL" }
}

#[cfg(target_os = "macos")]
struct Accuracy {
    err: f32,
    tol: f32,
}

#[cfg(target_os = "macos")]
impl Accuracy {
    fn ok(&self) -> bool {
        self.err <= self.tol
    }
}

#[cfg(target_os = "macos")]
struct Measurement {
    name: String,
    bytes: usize,
    seconds: f64,
}

#[cfg(target_os = "macos")]
impl Measurement {
    fn gb_per_s(&self) -> f64 {
        self.bytes as f64 / self.seconds / 1e9
    }
}

#[cfg(target_os = "macos")]
fn verify_f32(gpu: &tile_metal::Gpu, args: &Args, target: &Target) -> Result<Accuracy, String> {
    let (rows, cols) = (VERIFY_ROWS, args.cols);
    let mut x = vec![0.0f32; rows * cols];
    let mut w = vec![0.0f32; cols];
    reference::fill_pattern_f32(&mut x, 0x5eed);
    reference::fill_pattern_f32(&mut w, 0xc0ffee);

    let mut want = vec![0.0f32; rows * cols];
    reference::rmsnorm_f32(&x, &w, &mut want, rows, cols, args.eps);

    let k = Kernel::rmsnorm(DType::F32, rows, cols, args.eps);
    let p = plan(&k, target).map_err(|e| e.to_string())?;
    let pipeline = gpu.build(&k, &p)?;
    check_execution_width(&pipeline, target);

    let bx = gpu.upload(&x);
    let bw = gpu.upload(&w);
    let by = gpu.zeroed::<f32>(rows * cols);
    gpu.run(&pipeline, &[&bx, &bw, &by]);

    let mut got = vec![0.0f32; rows * cols];
    gpu.download(&by, &mut got);
    Ok(Accuracy {
        err: reference::max_rel_err(&got, &want),
        tol: reference::TOL_F32,
    })
}

#[cfg(target_os = "macos")]
fn verify_f16(gpu: &tile_metal::Gpu, args: &Args, target: &Target) -> Result<Accuracy, String> {
    use half::f16;

    let (rows, cols) = (VERIFY_ROWS, args.cols);
    let mut x = vec![f16::ZERO; rows * cols];
    let mut w = vec![f16::ZERO; cols];
    reference::fill_pattern_f16(&mut x, 0x5eed);
    reference::fill_pattern_f16(&mut w, 0xc0ffee);

    let mut want = vec![f16::ZERO; rows * cols];
    reference::rmsnorm_f16(&x, &w, &mut want, rows, cols, args.eps);

    let k = Kernel::rmsnorm(DType::F16, rows, cols, args.eps);
    let p = plan(&k, target).map_err(|e| e.to_string())?;
    let pipeline = gpu.build(&k, &p)?;
    check_execution_width(&pipeline, target);

    let bx = gpu.upload(&x);
    let bw = gpu.upload(&w);
    let by = gpu.zeroed::<f16>(rows * cols);
    gpu.run(&pipeline, &[&bx, &bw, &by]);

    let mut got = vec![f16::ZERO; rows * cols];
    gpu.download(&by, &mut got);

    let got: Vec<f32> = got.iter().map(|v| v.to_f32()).collect();
    let want: Vec<f32> = want.iter().map(|v| v.to_f32()).collect();
    Ok(Accuracy {
        err: reference::max_rel_err(&got, &want),
        tol: reference::TOL_F16,
    })
}

/// The emitted reduction assumes the target table's simd width. If the device
/// disagrees, the kernel is silently wrong, so say so loudly.
#[cfg(target_os = "macos")]
fn check_execution_width(pipeline: &tile_metal::Pipeline, target: &Target) {
    let actual = pipeline.thread_execution_width();
    if actual != target.simd_width {
        eprintln!(
            "warning: `{}` runs {} threads in lockstep but the target table says {}; \
             the two-stage reduction assumes the table is right",
            pipeline.name, actual, target.simd_width
        );
    }
}

#[cfg(target_os = "macos")]
fn measure(
    gpu: &tile_metal::Gpu,
    args: &Args,
    target: &Target,
) -> Result<Vec<Measurement>, String> {
    let copy = Kernel::copy(args.dtype, args.copy_elems());
    let norm = Kernel::rmsnorm(args.dtype, args.rows, args.cols, args.eps);
    let copy_plan = plan(&copy, target).map_err(|e| e.to_string())?;
    let norm_plan = plan(&norm, target).map_err(|e| e.to_string())?;
    let copy_pipe = gpu.build(&copy, &copy_plan)?;
    let norm_pipe = gpu.build(&norm, &norm_plan)?;

    // Inputs carry a real bit pattern rather than zeros. Nothing on this
    // hardware should make all-zero traffic faster, but "should" is doing a lot
    // of work in a benchmark whose only job is to be believed.
    let n = args.copy_elems();
    let cx = filled(gpu, args.dtype, n, 0x1234);
    let nx = filled(gpu, args.dtype, args.rows * args.cols, 0x5eed);
    let nw = filled(gpu, args.dtype, args.cols, 0xc0ffee);
    let (cy, ny) = match args.dtype {
        DType::F32 => (
            gpu.zeroed::<f32>(n),
            gpu.zeroed::<f32>(args.rows * args.cols),
        ),
        DType::F16 => (
            gpu.zeroed::<half::f16>(n),
            gpu.zeroed::<half::f16>(args.rows * args.cols),
        ),
    };

    let copy_s = gpu.time(&copy_pipe, &[&cx, &cy], args.iters, args.repeats);
    let norm_s = gpu.time(&norm_pipe, &[&nx, &nw, &ny], args.iters, args.repeats);

    Ok(vec![
        Measurement {
            name: copy.name.clone(),
            bytes: copy.ideal_bytes(),
            seconds: copy_s,
        },
        Measurement {
            name: norm.name.clone(),
            bytes: norm.ideal_bytes(),
            seconds: norm_s,
        },
    ])
}

#[cfg(target_os = "macos")]
fn human_bytes(b: usize) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else {
        format!("{:.1} MB", b / 1e6)
    }
}

#[cfg(target_os = "macos")]
fn human_time(s: f64) -> String {
    if s >= 1e-3 {
        format!("{:.3} ms", s * 1e3)
    } else {
        format!("{:.1} us", s * 1e6)
    }
}

/// A device buffer of `len` elements of `dtype`, filled with the reference
/// pattern. The host staging vector is dropped as soon as the upload copies it.
#[cfg(target_os = "macos")]
fn filled(gpu: &tile_metal::Gpu, dtype: DType, len: usize, seed: u32) -> tile_metal::Buffer {
    match dtype {
        DType::F32 => {
            let mut host = vec![0.0f32; len];
            reference::fill_pattern_f32(&mut host, seed);
            gpu.upload(&host)
        }
        DType::F16 => {
            let mut host = vec![half::f16::ZERO; len];
            reference::fill_pattern_f16(&mut host, seed);
            gpu.upload(&host)
        }
    }
}
