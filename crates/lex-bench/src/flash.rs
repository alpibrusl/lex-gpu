//! `--flash`: flash-attention decode, a typed lex-front kernel lowered to MSL.
//!
//! Llama-3-8B's decode shape: 8 KV heads, 4 query heads each (GQA), head dim
//! 128, f16 storage; `--batch` sequences of `--seq` cached positions. Decode
//! attention is bandwidth-bound — every K/V byte is read once per step — so it
//! is scored the way P0 scores RMSNorm: GB/s against the copy ceiling.
//!
//! P2 is *correct at any speed*. One threadgroup per (sequence, KV head) and a
//! conservative barrier schedule leave most of the GPU idle; the number is a
//! baseline for P3 (split-K across threadgroups, simdgroup matrices, fewer
//! barriers), not a claim.

use lex_front::flash::FlashDecode;
use lex_ir::{DType, Space, Target};
use lex_msl::program::{Lowered, lower};

use crate::args::Args;

pub const KV_HEADS: usize = 8;
pub const GROUP: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const THREADS: usize = 128;

pub fn config(args: &Args) -> FlashDecode {
    FlashDecode {
        q_rows: GROUP,
        d: HEAD_DIM,
        seq: args.seq,
        bq: GROUP,
        bk: 16,
        stages: 2,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 0,
        heads: args.batch * KV_HEADS,
        kv_cap: 0,
    }
}

/// Build, check and lower. Device-independent.
pub fn lowered(args: &Args, target: &Target) -> Result<(FlashDecode, Lowered), String> {
    let cfg = config(args);
    let prog = cfg.build()?;
    lex_front::check(&prog, target).map_err(|errs| {
        errs.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("\n")
    })?;
    let low = lower(&prog, target, THREADS)?;
    Ok((cfg, low))
}

/// Bytes a decode step must move: Q and O once, all of K and V once.
pub fn ideal_bytes(cfg: &FlashDecode) -> usize {
    let h = cfg.heads;
    h * cfg.q_rows * cfg.d * 2 + 2 * h * cfg.seq * cfg.d * 2 + h * cfg.q_rows * cfg.d * 4
}

#[cfg(target_os = "macos")]
pub fn run(args: &Args, target: &Target) -> std::process::ExitCode {
    use std::process::ExitCode;

    use half::f16;
    use lex_ir::reference::{fill_pattern_f32, max_rel_err};

    let (cfg, low) = match lowered(args, target) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let gpu = match lex_metal::Gpu::open() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    let pipe = match gpu.build_lowered(&low) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!();
    println!("device : {}", gpu.info().name);
    println!(
        "shape  : batch {} x {KV_HEADS} kv heads x {GROUP} q heads, head dim {HEAD_DIM}, seq {}, f16",
        args.batch, args.seq
    );
    println!(
        "kernel : {} threadgroups x {} threads, {} B threadgroup ({} tiles + {} scratch), {} barrier sites",
        low.grid,
        low.threads,
        low.threadgroup_bytes,
        low.arena_bytes,
        low.scratch_bytes,
        low.barriers
    );
    println!();

    let h = cfg.heads;
    let pattern = |n: usize, seed: u32| {
        let mut x = vec![0.0f32; n];
        fill_pattern_f32(&mut x, seed);
        x.iter().map(|&a| f16::from_f32(a)).collect::<Vec<_>>()
    };
    let q = pattern(h * cfg.q_rows * cfg.d, 1);
    let k = pattern(h * cfg.seq * cfg.d, 2);
    let v = pattern(h * cfg.seq * cfg.d, 3);
    let bq = gpu.upload(&q);
    let bk = gpu.upload(&k);
    let bv = gpu.upload(&v);
    let bo = gpu.zeroed::<f32>(h * cfg.q_rows * cfg.d);
    let bufs = [&bq, &bk, &bv, &bo];

    // Correctness first: the first and last heads against the f64 reference.
    gpu.run(&pipe, &bufs);
    let mut out = vec![0.0f32; h * cfg.q_rows * cfg.d];
    gpu.download(&bo, &mut out);
    let f = |x: &[f16]| x.iter().map(|a| a.to_f32()).collect::<Vec<_>>();
    let (qf, kf, vf) = (f(&q), f(&k), f(&v));
    let mut err = 0.0f32;
    for head in [0, h - 1] {
        let (qn, kn) = (cfg.q_rows * cfg.d, cfg.seq * cfg.d);
        let want = lex_front::flash::reference(
            &qf[head * qn..(head + 1) * qn],
            &kf[head * kn..(head + 1) * kn],
            &vf[head * kn..(head + 1) * kn],
            cfg.q_rows,
            cfg.seq,
            cfg.d,
        );
        err = err.max(max_rel_err(&out[head * qn..(head + 1) * qn], &want));
    }

    let flash_s = gpu.time(&pipe, &bufs, args.iters, args.repeats);

    // The ceiling, measured the P0 way, in the same process.
    let copy = lex_ir::Kernel::copy(DType::F16, args.copy_elems());
    let copy_s = match lex_ir::plan(&copy, target)
        .map_err(|e| e.to_string())
        .and_then(|p| {
            let pipe = gpu.build(&copy, &p)?;
            let n = args.copy_elems();
            let (x, y) = (gpu.zeroed::<f16>(n), gpu.zeroed::<f16>(n));
            Ok(gpu.time(&pipe, &[&x, &y], args.iters, args.repeats))
        }) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::FAILURE;
        }
    };

    let copy_gbs = copy.ideal_bytes() as f64 / copy_s / 1e9;
    let flash_bytes = ideal_bytes(&cfg);
    let flash_gbs = flash_bytes as f64 / flash_s / 1e9;
    println!(
        "{:<18} {:>12} {:>12} {:>10} {:>12}",
        "kernel", "ideal bytes", "time", "GB/s", "% of copy"
    );
    println!(
        "{:<18} {:>9.1} MB {:>9.3} ms {:>10.1} {:>11.1}%",
        "copy_f16",
        copy.ideal_bytes() as f64 / 1e6,
        copy_s * 1e3,
        copy_gbs,
        100.0
    );
    println!(
        "{:<18} {:>9.1} MB {:>9.3} ms {:>10.1} {:>11.1}%",
        "flash_decode_f16",
        flash_bytes as f64 / 1e6,
        flash_s * 1e3,
        flash_gbs,
        100.0 * flash_gbs / copy_gbs
    );
    println!();
    let ok = err <= 1e-4;
    println!(
        "correct : max rel err {err:.2e} vs f64 reference (tolerance 1e-4)  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    println!("P2 is correct-at-any-speed; the % of copy is the baseline P3 has to raise.");
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
