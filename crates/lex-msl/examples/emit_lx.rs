//! Check a `.lx` file, lower it for every target it has a schedule for and
//! write the source out, for `scripts/cuda_check.sh` (CUDA) or `xcrun metal`
//! (Metal), neither of which needs a GPU:
//!
//!     cargo run -p lex-msl --example emit_lx -- crates/lex-front/lx/gemm_mma.lx out/ m=256 n=256 k=128
//!     scripts/cuda_check.sh out/*.cu
//!     for f in out/*.metal; do xcrun -sdk macosx metal -c $f -o /dev/null; done
//!
//! `--run` also interprets the program on the CPU over made-up inputs and
//! prints a summary of what it writes, which is how to see that a new
//! `.lx` computes something before any device is involved.

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
    let mut run = false;
    for a in args {
        if a == "--run" {
            run = true;
            continue;
        }
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
        if run {
            interpret(&prog, target.name)?;
        }
        let l = lower_sched(
            &prog,
            &target,
            dialect,
            &Sched {
                threads,
                warps,
                pad: s.pad,
            },
        )?;
        let path = format!("{dir}/{}.{ext}", l.entry);
        std::fs::write(&path, &l.source).map_err(|e| e.to_string())?;
        println!(
            "{path}: {} threads, grid {}x{}, {} B of tiles",
            l.threads, l.grid, l.grid2, l.arena_bytes
        );
    }
    Ok(())
}

/// Run `prog` in the reference interpreter over made-up inputs -- floats in
/// [-0.5, 0.5), and bytes that are valid quantised codes and scales -- and
/// print what the writable parameters hold.
fn interpret(prog: &lex_front::Program, target: &str) -> Result<(), String> {
    use lex_front::interp::{Tensor, run};
    use lex_ir::DType;
    let mut seed = 12345u32;
    let mut next = move || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 9) as f32 / (1u32 << 23) as f32 - 0.5
    };
    let mut tensors: Vec<Tensor> = prog
        .params
        .iter()
        .map(|p| {
            let n: usize = p.shape.iter().product();
            let data: Vec<f32> = match p.dtype {
                DType::I8 => (0..n).map(|i| (0x30 + i % 8) as f32).collect(),
                _ if p.writable => vec![0.0; n],
                _ => (0..n).map(|_| next()).collect(),
            };
            Tensor::new(p.dtype, &p.shape, &data)
        })
        .collect();
    run(prog, &mut tensors).map_err(|e| format!("{target}: interpreter: {e:?}"))?;
    for (p, t) in prog.params.iter().zip(&tensors) {
        if p.writable {
            let peak = t.data.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let mean = t.data.iter().sum::<f32>() / t.data.len() as f32;
            println!(
                "{target}: {} {:?} = mean {mean:.4}, peak {peak:.4}, first {:?}",
                p.name,
                p.shape,
                &t.data[..4.min(t.data.len())]
            );
        }
    }
    Ok(())
}
