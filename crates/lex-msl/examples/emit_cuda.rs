//! Lower typed lex-front programs to CUDA and write them out, so
//! `scripts/cuda_check.sh` can compile them without a GPU.
//!
//!     cargo run -p lex-msl --example emit_cuda -- out/
//!     scripts/cuda_check.sh out/*.cu
//!
//! These are the same programs the Metal backend lowers, through the same
//! `lower_with`; only the dialect differs.

use lex_front::ir::{Arg, Builder, IdxExpr, Op, Program, TileTy, View};
use lex_front::llama::{QLayout, matvec_q, rmsnorm_rows, silu_mul};
use lex_ir::{DType, Space, Target};
use lex_msl::dialect::Cuda;
use lex_msl::program::lower_with;

fn main() -> Result<(), String> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    std::fs::create_dir_all(&dir).map_err(|e| format!("{dir}: {e}"))?;
    let target = Target::nvidia_ada();
    let progs = [
        rmsnorm_rows(4, 4096, 1e-5, None, DType::F32),
        rmsnorm_rows(4, 4096, 1e-5, None, DType::F16),
        silu_mul(4096, 256, DType::F32)?,
        silu_mul(4096, 256, DType::F16)?,
        // The one that matters: NVFP4 dequantisation fused into a matvec.
        // 94% of a decode step is this shape.
        matvec_q(5120, 17408, 8, 5120, QLayout::NVFP4, false)?,
        // A blockwise Hadamard, which low-bit formats that store a
        // rotated basis need applied to activations. Ten butterfly
        // stages, each reading the partner another thread owns -- the
        // shape most likely to lower differently between the two
        // backends, and therefore the one most worth compiling here.
        hadamard(5120, 1024),
    ];
    for p in &progs {
        let l = lower_with(p, &target, 256, &Cuda)?;
        let path = format!("{dir}/{}.cu", l.entry);
        std::fs::write(&path, &l.source).map_err(|e| format!("{path}: {e}"))?;
        println!("{path}");
    }
    Ok(())
}

/// `y = H_width x` over a row of `cols`, as butterfly stages.
fn hadamard(cols: usize, width: usize) -> Program {
    let mut b = Builder::new(&format!("hadamard_{cols}_w{width}"));
    let px = b.param("x", DType::F32, &[1, cols], false);
    let py = b.param("y", DType::F32, &[1, cols], true);
    let v = |p| View {
        param: p,
        offset: vec![IdxExpr::lit(0), IdxExpr::lit(0)],
        shape: vec![1, cols],
    };
    let ty = TileTy::new(DType::F32, &[1, cols], Space::Reg);
    let mut x = b.op("x", Op::Load(v(px), ty));
    let mut stride = 1;
    while stride < width {
        x = b.op("h", Op::Butterfly(Arg::Move(x), stride));
        stride *= 2;
    }
    b.effect(Op::Store(Arg::Move(x), v(py)));
    b.finish()
}
