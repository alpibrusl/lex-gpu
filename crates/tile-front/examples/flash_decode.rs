//! Print the flash-decode program and check each schedule on each target.
//!
//! cargo run -p tile-front --example flash_decode [-- --ir]

use tile_front::flash::FlashDecode;
use tile_front::{Tensor, check, print, run};
use tile_ir::reference::{fill_pattern_f32, max_rel_err};
use tile_ir::{DType, Space, Target};

fn main() {
    let shape = |bq, bk, stages| FlashDecode {
        q_rows: 32,
        d: 128,
        seq: 512,
        bq,
        bk,
        stages,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
    };
    let schedules = [
        ("metal  bq8  bk32  x2", shape(8, 32, 2)),
        ("hopper bq16 bk128 x3", shape(16, 128, 3)),
    ];
    let targets = [
        Target::apple_m_series(),
        Target::nvidia_hopper(),
        Target::amd_cdna3(),
    ];

    if std::env::args().any(|a| a == "--ir") {
        print!("{}", print::program(&schedules[0].1.build().unwrap()));
        return;
    }

    for (name, cfg) in &schedules {
        let prog = cfg.build().unwrap();
        println!("schedule {name}");
        for t in &targets {
            match check(&prog, t) {
                Ok(r) => {
                    println!(
                        "  {:<16} ok    {:>6} B threadgroup, {} moves, {} borrows, {} dups",
                        t.name, r.peak_threadgroup_bytes, r.moves, r.borrows, r.dups
                    );
                    for w in r.warnings {
                        println!("  {:<16}       warning: {w}", "");
                    }
                }
                Err(errs) => {
                    for e in errs {
                        println!("  {:<16} REJECT {e}", t.name);
                    }
                }
            }
        }
        let mut ts = vec![
            input(32, 1),
            input(512, 2),
            input(512, 3),
            Tensor::zeros(DType::F32, &[32, 128]),
        ];
        run(&prog, &mut ts).unwrap();
        let want =
            tile_front::flash::reference(&ts[0].data, &ts[1].data, &ts[2].data, 32, 512, 128);
        println!(
            "  interpreter vs f64 reference: max rel err {:.2e}\n",
            max_rel_err(&ts[3].data, &want)
        );
    }
}

fn input(rows: usize, seed: u32) -> Tensor {
    let mut x = vec![0.0; rows * 128];
    fill_pattern_f32(&mut x, seed);
    Tensor::new(DType::F16, &[rows, 128], &x)
}
