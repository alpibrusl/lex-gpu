//! Flash-attention decode, three schedules, three targets.
//!
//! Prints what the checker says about each schedule on each target (budget,
//! warnings, barrier arrival counts for warp-specialised pipes), runs each in
//! the interpreter, and compares the result against PyTorch's
//! `scaled_dot_product_attention` (the golden written by
//! `scripts/flash_decode_golden.py`).
//!
//! cargo run -p tile-front --example flash_decode
//! cargo run -p tile-front --example flash_decode -- --ir [metal|hopper|ws]

use tile_front::flash::FlashDecode;
use tile_front::{Tensor, check, print, run};
use tile_ir::reference::{fill_pattern_f32, max_rel_err};
use tile_ir::{DType, Space, Target};

const Q_ROWS: usize = 32;
const D: usize = 128;
const SEQ: usize = 512;
const PYTORCH: &[u8] = include_bytes!("../tests/data/flash_decode_f16_q32_d128_s512.f32");

fn main() {
    let shape = |bq, bk, stages, consumers| FlashDecode {
        q_rows: Q_ROWS,
        d: D,
        seq: SEQ,
        bq,
        bk,
        stages,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers,
        heads: 1,
    };
    let schedules = [
        ("metal", "bq 8, bk 32, 2 stages", shape(8, 32, 2, 0)),
        ("hopper", "bq 16, bk 128, 3 stages", shape(16, 128, 3, 0)),
        (
            "ws",
            "producer + 2 consumer warpgroups, bk 128, 3 stages",
            shape(16, 128, 3, 2),
        ),
    ];
    let targets = [
        Target::apple_m_series(),
        Target::nvidia_hopper(),
        Target::amd_cdna3(),
    ];

    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--ir") {
        let which = args.get(i + 1).map_or("metal", String::as_str);
        let Some((_, _, cfg)) = schedules.iter().find(|s| s.0 == which) else {
            eprintln!("--ir takes one of: metal, hopper, ws");
            std::process::exit(2);
        };
        print!("{}", print::program(&cfg.build().unwrap()));
        return;
    }

    let (words, _) = PYTORCH.as_chunks::<4>();
    let pytorch: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();

    for (key, name, cfg) in &schedules {
        let prog = cfg.build().unwrap();
        println!("{key}: {name}");
        for t in &targets {
            match check(&prog, t) {
                Ok(r) => {
                    println!(
                        "  {:<16} ok      {:>6} B threadgroup, {} borrows, {} dups",
                        t.name, r.peak_threadgroup_bytes, r.borrows, r.dups
                    );
                    for p in &r.pipes {
                        println!(
                            "  {:<16}         pipe: {} slots, full barrier {} arrival + {} B tx, \
                             empty barrier {} arrivals, {} threads",
                            "",
                            p.stages,
                            p.full_arrivals,
                            p.full_tx_bytes,
                            p.empty_arrivals,
                            p.threads
                        );
                    }
                    for w in r.warnings {
                        println!("  {:<16}         warning: {w}", "");
                    }
                }
                Err(errs) => {
                    for e in errs {
                        println!("  {:<16} REJECT  {e}", t.name);
                    }
                }
            }
        }
        let mut ts = vec![
            input(Q_ROWS, 1),
            input(SEQ, 2),
            input(SEQ, 3),
            Tensor::zeros(DType::F32, &[Q_ROWS, D]),
        ];
        run(&prog, &mut ts).unwrap();
        println!(
            "  interpreter vs PyTorch SDPA: max rel err {:.2e}\n",
            max_rel_err(&ts[3].data, &pytorch)
        );
    }
}

fn input(rows: usize, seed: u32) -> Tensor {
    let mut x = vec![0.0; rows * D];
    fill_pattern_f32(&mut x, seed);
    Tensor::new(DType::F16, &[rows, D], &x)
}
