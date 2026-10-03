//! Lower a `.lx` file for every target it has a schedule for and write the
//! source out, for `scripts/cuda_check.sh` (CUDA) or `xcrun metal` (Metal),
//! neither of which needs a GPU:
//!
//!     cargo run -p lex-msl --example emit_lx -- crates/lex-front/lx/gemm_mma.lx out/ m=256 n=256 k=128
//!     scripts/cuda_check.sh out/*.cu
//!     for f in out/*.metal; do xcrun -sdk macosx metal -c $f -o /dev/null; done

use lex_msl::dialect::{Cuda, Dialect, Msl};
use lex_msl::program::{Sched, lower_sched};

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let file = args
        .next()
        .ok_or("usage: emit_lx <file.lx> <dir> [name=value ...]")?;
    let dir = args
        .next()
        .ok_or("usage: emit_lx <file.lx> <dir> [name=value ...]")?;
    let mut consts: Vec<(String, f64)> = vec![];
    for a in args {
        let (k, v) = a
            .split_once('=')
            .ok_or(format!("`{a}` is not name=value"))?;
        consts.push((
            k.to_string(),
            v.parse().map_err(|_| format!("`{v}` is not a number"))?,
        ));
    }
    let src = std::fs::read_to_string(&file).map_err(|e| format!("{file}: {e}"))?;
    let unit = lex_front::syntax::parse(&src)?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let args: Vec<(&str, f64)> = consts.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    for (target, dialect, ext) in [
        (lex_ir::Target::nvidia_ada(), &Cuda as &dyn Dialect, "cu"),
        (lex_ir::Target::apple_m_series(), &Msl, "metal"),
    ] {
        let Ok(s) = unit.schedule_for(target.name) else {
            continue;
        };
        let warps = s.warps;
        let (prog, threads) = unit.compile(target.name, &args)?;
        lex_front::check(&prog, &target).map_err(|e| format!("{}: {e:#?}", target.name))?;
        let l = lower_sched(&prog, &target, dialect, &Sched { threads, warps })?;
        let path = format!("{dir}/{}.{ext}", l.entry);
        std::fs::write(&path, &l.source).map_err(|e| e.to_string())?;
        println!(
            "{path}: {} threads, grid {}x{}, {} B of tiles",
            l.threads, l.grid, l.grid2, l.arena_bytes
        );
    }
    Ok(())
}
