//! Kernels the runtime gets from `.lx` source instead of from a hand-written
//! generator.
//!
//! The first is the prefill GEMM: `gemm_fp4.lx` and `gemm_fp4_res.lx`, which
//! say what the matmul means and where its tiles live, with a schedule per
//! machine. [`gemm`] binds a chunk size to one of them and lowers it, in the
//! parameter order `lex_msl::gemm::gemm_nvfp4` uses, so the runtime binds
//! either the same way.
//!
//! It is opt-in (`LEX_GEMM_LX=1`) until the model's own tests have run on it
//! on every device: the hand-written kernel is what ships. A shape the
//! schedule cannot tile, or that this does not handle (f32 activations), is
//! `None` and the caller keeps the hand-written one.

use std::sync::OnceLock;

use lex_front::syntax::{Unit, parse};
use lex_ir::Target;
use lex_msl::dialect::{Cuda, Dialect, Msl};
use lex_msl::gemm::Backend;
use lex_msl::program::{Lowered, Sched, lower_sched};

fn unit(residual: bool) -> &'static Unit {
    static PLAIN: OnceLock<Unit> = OnceLock::new();
    static RES: OnceLock<Unit> = OnceLock::new();
    let (cell, src) = if residual {
        (&RES, include_str!("../../lex-front/lx/gemm_fp4_res.lx"))
    } else {
        (&PLAIN, include_str!("../../lex-front/lx/gemm_fp4.lx"))
    };
    cell.get_or_init(|| parse(src).expect("the shipped .lx parses"))
}

/// Whether the runtime should take its GEMM from `.lx`.
pub fn enabled() -> bool {
    std::env::var("LEX_GEMM_LX").is_ok_and(|v| v == "1")
}

/// The prefill GEMM for `t` tokens, `n_out` rows and `n_in` inputs, from the
/// `.lx` file and the target's schedule, with the token tile cut down to the
/// chunk where the chunk is smaller than the schedule's. `None` where it does
/// not tile.
pub fn gemm(
    t: usize,
    n_out: usize,
    n_in: usize,
    residual: bool,
    x_half: bool,
    backend: Backend,
) -> Result<Option<Lowered>, String> {
    if !x_half {
        return Ok(None);
    }
    let (target, dialect): (Target, &dyn Dialect) = match backend {
        Backend::Metal => (Target::apple_m_series(), &Msl),
        Backend::Cuda => (Target::nvidia_ada(), &Cuda),
    };
    let unit = unit(residual);
    let s = unit.schedule_for(target.name)?;
    let ext = |name: &str| {
        s.extents
            .iter()
            .find(|(n, _)| n == name)
            .map(|e| e.1)
            .ok_or(format!("the schedule sets no `{name}`"))
    };
    let (bm, bn, bk) = (ext("bm")?, ext("bn")?, ext("bk")?);
    let (wm, wn) = s.warps.ok_or("the schedule sets no warps")?;
    let atom = dialect.matrix().ok_or("no matrix unit")?.atom();
    // A chunk smaller than the token tile takes a tile of its own size, and
    // fewer warps along the tokens if the rows would not fill an atom.
    let bm = bm.min(t);
    let wm = if bm.is_multiple_of(wm * atom) { wm } else { 1 };
    if !t.is_multiple_of(bm) || !bm.is_multiple_of(wm * atom) {
        return Ok(None);
    }
    if !n_out.is_multiple_of(bn) || !n_in.is_multiple_of(bk) {
        return Ok(None);
    }
    let consts = [
        ("m", t as f64),
        ("n", n_out as f64),
        ("k", n_in as f64),
        ("bm", bm as f64),
        ("bn", bn as f64),
        ("bk", bk as f64),
    ];
    let prog = unit.algo.build_with(&consts, None)?;
    lex_front::check(&prog, &target).map_err(|e| format!("gemm_fp4 for {t}: {e:#?}"))?;
    let lowered = lower_sched(
        &prog,
        &target,
        dialect,
        &Sched {
            threads: target.simd_width * wm * wn,
            warps: Some((wm, wn)),
            pad: s.pad,
        },
    )?;
    if std::env::var_os("LEX_GEMM_LX_TRACE").is_some() {
        eprintln!(
            "gemm from .lx: {t} tokens x {n_out} rows x {n_in}, residual {residual}: {}",
            lowered.entry
        );
    }
    Ok(Some(lowered))
}
