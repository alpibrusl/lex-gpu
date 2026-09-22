//! Every Llama layer kernel on the GPU against the interpreter, kernel by
//! kernel and layout by layout. The model-level test says *whether* the
//! GPU is right; this one says *which* kernel is not.
#![cfg(target_os = "macos")]

use half::f16;
use tile_front::llama::{
    QLayout, Split, kv_append, matmul_q, matmul_q_glu, matvec_q, matvec_q_rms, rmsnorm, rope,
    rope_tables, silu_mul,
};
use tile_front::{Program, Tensor, check, run_dyn};
use tile_ir::reference::fill_pattern_f32;
use tile_ir::{DType, Target};
use tile_metal::{Buffer, Gpu};
use tile_msl::program::lower;

fn pattern(n: usize, seed: u32) -> Vec<f32> {
    let mut x = vec![0.0; n];
    fill_pattern_f32(&mut x, seed);
    x
}

fn upload(gpu: &Gpu, t: &Tensor) -> Buffer {
    match t.dtype {
        DType::F32 => gpu.upload(&t.data),
        DType::F16 => gpu.upload(&t.data.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>()),
        DType::I8 => gpu.upload(&t.data.iter().map(|&x| x as i8).collect::<Vec<_>>()),
    }
}

fn download(gpu: &Gpu, b: &Buffer, t: &Tensor) -> Vec<f32> {
    let n = t.data.len();
    match t.dtype {
        DType::F32 => {
            let mut v = vec![0.0f32; n];
            gpu.download(b, &mut v);
            v
        }
        DType::F16 => {
            let mut v = vec![f16::ZERO; n];
            gpu.download(b, &mut v);
            v.iter().map(|x| x.to_f32()).collect()
        }
        DType::I8 => unreachable!(),
    }
}

/// Run `prog` on the GPU and in the interpreter; compare output `out`.
fn same(
    gpu: &Gpu,
    prog: &Program,
    tensors: Vec<Tensor>,
    scalars: &[u32],
    out: usize,
    threads: usize,
) {
    check(prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{}: {e:#?}", prog.name));
    let pipe = gpu
        .build_lowered(&lower(prog, gpu.target(), threads).expect("lower"))
        .expect("compile");
    let bufs: Vec<Buffer> = tensors.iter().map(|t| upload(gpu, t)).collect();
    // Metal will not allocate an empty buffer; only kernels with runtime
    // scalars get one.
    let sc = (!scalars.is_empty()).then(|| gpu.upload(scalars));
    let mut refs: Vec<&Buffer> = bufs.iter().collect();
    refs.extend(sc.as_ref());
    gpu.run(&pipe, &refs);
    let got = download(gpu, &bufs[out], &tensors[out]);
    let mut want = tensors;
    run_dyn(prog, &mut want, scalars).expect("interpret");
    // Error relative to the output's scale: a reduction summed in a
    // different order legitimately differs in the last bits, and an output
    // that cancels to near zero would make a per-element relative error
    // meaningless.
    let abs = got
        .iter()
        .zip(&want[out].data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let scale = want[out]
        .data
        .iter()
        .map(|x| x.abs())
        .fold(1e-6f32, f32::max);
    // An f16 output carries about 5e-4 of relative precision, so one
    // rounding step apart is all a correct kernel can promise there.
    let tol = match want[out].dtype {
        DType::F16 => 1e-3,
        _ => 1e-5,
    };
    assert!(
        abs / scale < tol,
        "{}: GPU vs interpreter max abs err {abs:e} on outputs up to {scale:e} (tolerance {tol:e})",
        prog.name
    );
}

#[test]
fn matvec_every_layout_matches_the_interpreter() {
    let gpu = Gpu::open().expect("metal device");
    // Llama-3-8B's hidden size, one chunk per row as the runtime runs it.
    let (n_in, n_out) = (4096, 64);
    for layout in [QLayout::Q8_0, QLayout::Q6_K, QLayout::Q4_K] {
        for residual in [false, true] {
            let groups = n_out * n_in / layout.group;
            let supers = n_out * n_in / (layout.group * layout.super_groups.unwrap_or(1));
            let f16s = |n, seed| {
                pattern(n, seed)
                    .iter()
                    .map(|v| f16::from_f32(0.001 + v.abs() * 0.01))
                    .collect::<Vec<_>>()
            };
            let small = |n, seed| {
                pattern(n, seed)
                    .iter()
                    .map(|v| ((v + 1.0) * 31.5) as i8)
                    .collect::<Vec<_>>()
            };
            let two_level = layout.super_groups.is_some();
            let split = |o: u32| Split {
                layout,
                cols: n_in,
                qh: if layout.six {
                    pattern(n_in * n_out / 4, 10 + o)
                        .iter()
                        .map(|v| ((v + 1.0) * 127.5) as u8 as i8)
                        .collect()
                } else {
                    vec![]
                },
                q: if layout.packed4 || layout.six {
                    pattern(n_in * n_out / 2, 4 + o)
                        .iter()
                        .map(|v| ((v + 1.0) * 127.5) as u8 as i8)
                        .collect()
                } else {
                    pattern(n_in * n_out, 4 + o)
                        .iter()
                        .map(|v| (v * 31.0).round() as i8)
                        .collect()
                },
                sc: if two_level {
                    small(groups, 5 + o)
                } else {
                    vec![]
                },
                d: f16s(if two_level { supers } else { groups }, 6 + o),
                mn: layout
                    .min
                    .then(|| (small(groups, 7 + o), f16s(supers, 8 + o))),
            };
            let sp = split(0);
            let mut t = vec![Tensor::new(DType::F32, &[1, n_in], &pattern(n_in, 3))];
            t.extend(sp.weight_tensors(n_out));
            if residual {
                t.push(Tensor::new(DType::F32, &[1, n_out], &pattern(n_out, 6)));
            }
            t.push(Tensor::zeros(DType::F32, &[1, n_out]));
            let out = t.len() - 1;
            let p = matvec_q(n_in, n_out, 8, n_in, layout, residual).unwrap();
            same(&gpu, &p, t, &[], out, 256);

            // The same weights against a small batch of token rows: each
            // weight read once for all of them (speculative verify, prefill).
            for rows in [3usize, 8, 16] {
                let mut t = vec![Tensor::new(
                    DType::F32,
                    &[rows, n_in],
                    &pattern(rows * n_in, 11),
                )];
                t.extend(sp.weight_tensors(n_out));
                if residual {
                    t.push(Tensor::new(
                        DType::F32,
                        &[rows, n_out],
                        &pattern(rows * n_out, 12),
                    ));
                }
                t.push(Tensor::zeros(DType::F32, &[rows, n_out]));
                let out = t.len() - 1;
                let p = matmul_q(rows, n_in, n_out, 8, n_in, layout, residual).unwrap();
                same(&gpu, &p, t, &[], out, 256);
            }

            // Gate and up fused with SiLU·mul, for one row and a batch.
            if !residual {
                let up = split(100);
                for rows in [1usize, 4] {
                    let mut t = vec![Tensor::new(
                        DType::F32,
                        &[rows, n_in],
                        &pattern(rows * n_in, 13),
                    )];
                    t.extend(sp.weight_tensors(n_out));
                    t.extend(up.weight_tensors(n_out));
                    t.push(Tensor::zeros(DType::F32, &[rows, n_out]));
                    let out = t.len() - 1;
                    let p = matmul_q_glu(rows, n_in, n_out, 8, layout, None).unwrap();
                    same(&gpu, &p, t, &[], out, 256);
                }

                // The same, with the FFN's RMSNorm folded into the matvec,
                // and a plain matvec with the attention norm folded in.
                let g = Tensor::new(DType::F32, &[1, n_in], &pattern(n_in, 14));
                for (p, extra) in [
                    (
                        matmul_q_glu(1, n_in, n_out, 8, layout, Some(1e-5)).unwrap(),
                        true,
                    ),
                    (matvec_q_rms(n_in, n_out, 8, layout, 1e-5).unwrap(), false),
                ] {
                    let mut t = vec![
                        Tensor::new(DType::F32, &[1, n_in], &pattern(n_in, 13)),
                        g.clone(),
                    ];
                    t.extend(sp.weight_tensors(n_out));
                    if extra {
                        t.extend(up.weight_tensors(n_out));
                    }
                    t.push(Tensor::zeros(DType::F32, &[1, n_out]));
                    let out = t.len() - 1;
                    same(&gpu, &p, t, &[], out, 256);
                }
            }
        }
    }
}

#[test]
fn rmsnorm_rope_silu_and_kv_append_match_the_interpreter() {
    let gpu = Gpu::open().expect("metal device");
    let n = 4096;
    same(
        &gpu,
        &rmsnorm(n, 1e-5),
        vec![
            Tensor::new(DType::F32, &[1, n], &pattern(n, 1)),
            Tensor::new(DType::F32, &[1, n], &pattern(n, 2)),
            Tensor::zeros(DType::F32, &[1, n]),
        ],
        &[],
        2,
        256,
    );
    let (h, hd) = (8, 64);
    let (c, s) = rope_tables(7, hd, 500000.0, None);
    same(
        &gpu,
        &rope(h, hd, DType::F16),
        vec![
            Tensor::new(DType::F32, &[h, hd], &pattern(h * hd, 3)),
            Tensor::new(DType::F32, &[1, hd], &c),
            Tensor::new(DType::F32, &[1, hd], &s),
            Tensor::zeros(DType::F16, &[h, hd]),
        ],
        &[],
        3,
        256,
    );
    same(
        &gpu,
        &silu_mul(n, 256).unwrap(),
        vec![
            Tensor::new(DType::F32, &[1, n], &pattern(n, 4)),
            Tensor::new(DType::F32, &[1, n], &pattern(n, 5)),
            Tensor::zeros(DType::F32, &[1, n]),
        ],
        &[],
        2,
        256,
    );
    let cap = 16;
    same(
        &gpu,
        &kv_append(h, hd, cap, DType::F32),
        vec![
            Tensor::new(DType::F32, &[h, hd], &pattern(h * hd, 6)),
            Tensor::zeros(DType::F16, &[h * cap, hd]),
        ],
        &[9],
        1,
        64,
    );
}

#[test]
fn dynamic_attention_and_f16_kv_append_match_the_interpreter() {
    use tile_front::flash::FlashDecode;
    use tile_ir::Space;
    let gpu = Gpu::open().expect("metal device");
    let (heads, group, hd, cap) = (8, 4, 128, 64);
    let cfg = FlashDecode {
        q_rows: group,
        d: hd,
        seq: cap,
        bq: group,
        bk: 16,
        stages: 1,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 0,
        heads,
        kv_cap: cap,
    };
    let prog = cfg.build_dynamic().unwrap();
    for len in [1usize, 7, 16, 37, 64] {
        same(
            &gpu,
            &prog,
            vec![
                Tensor::new(
                    DType::F16,
                    &[heads * group, hd],
                    &pattern(heads * group * hd, 1),
                ),
                Tensor::new(
                    DType::F16,
                    &[heads * cap, hd],
                    &pattern(heads * cap * hd, 2),
                ),
                Tensor::new(
                    DType::F16,
                    &[heads * cap, hd],
                    &pattern(heads * cap * hd, 3),
                ),
                Tensor::zeros(DType::F32, &[heads * group, hd]),
            ],
            &[len as u32, len.div_ceil(16) as u32],
            3,
            128,
        );
    }
    same(
        &gpu,
        &kv_append(heads, hd, cap, DType::F16),
        vec![
            Tensor::new(DType::F16, &[heads, hd], &pattern(heads * hd, 6)),
            Tensor::zeros(DType::F16, &[heads * cap, hd]),
        ],
        &[37],
        1,
        64,
    );
}

#[test]
fn split_kv_attention_matches_the_interpreter() {
    use tile_front::flash::FlashDecode;
    use tile_ir::Space;
    let gpu = Gpu::open().expect("metal device");
    let (heads, group, hd, cap, bk) = (8, 4, 128, 256, 16);
    let hg = heads * group;
    let cfg = FlashDecode {
        q_rows: group,
        d: hd,
        seq: cap,
        bq: group,
        bk,
        stages: 1,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 0,
        heads,
        kv_cap: cap,
    };
    for bps in [1usize, 2] {
        let splits = cap / (bk * bps);
        let split = cfg.build_split(bps).unwrap();
        let combine = cfg.build_combine(bps).unwrap();
        for len in [1usize, 17, 40, 129, 256] {
            let tensors = || {
                vec![
                    Tensor::new(DType::F16, &[hg, hd], &pattern(hg * hd, 1)),
                    Tensor::new(
                        DType::F16,
                        &[heads * cap, hd],
                        &pattern(heads * cap * hd, 2),
                    ),
                    Tensor::new(
                        DType::F16,
                        &[heads * cap, hd],
                        &pattern(heads * cap * hd, 3),
                    ),
                    Tensor::zeros(DType::F32, &[hg, splits]),
                    Tensor::zeros(DType::F32, &[hg, splits]),
                    Tensor::zeros(DType::F32, &[hg, splits * hd]),
                ]
            };
            for out in [3, 4, 5] {
                same(&gpu, &split, tensors(), &[len as u32], out, 128);
            }
        }
        // Partial states with a spread of maxima, positive sums.
        let m: Vec<f32> = pattern(splits * hg, 4).iter().map(|x| 4.0 * x).collect();
        let l: Vec<f32> = pattern(splits * hg, 5).iter().map(|x| 1.5 + x).collect();
        for nsplit in [1usize, 3, 9, splits].into_iter().filter(|&n| n <= splits) {
            same(
                &gpu,
                &combine,
                vec![
                    Tensor::new(DType::F32, &[hg, splits], &m),
                    Tensor::new(DType::F32, &[hg, splits], &l),
                    Tensor::new(
                        DType::F32,
                        &[splits * hg, hd],
                        &pattern(splits * hg * hd, 6),
                    ),
                    Tensor::zeros(DType::F32, &[hg, hd]),
                ],
                &[nsplit as u32, nsplit.div_ceil(8) as u32],
                3,
                128,
            );
        }
    }
}

/// Qwen3.5's attention prologue at the model's shape: 24 query heads of
/// 256, a rotated quarter, and f16 out for the attention kernel.
#[test]
fn qwen_qk_rope_matches_the_interpreter() {
    use tile_front::qwen::{build_mul, build_qk_rope};
    let gpu = Gpu::open().expect("metal device");
    let (hd, rot) = (256usize, 64usize);
    for (heads, gate) in [(24usize, true), (4, false)] {
        let width = if gate { 2 * hd } else { hd };
        let prog = build_qk_rope(heads, hd, rot, gate, DType::F16, 1e-6).unwrap();
        let mut t = vec![
            Tensor::new(DType::F32, &[heads, width], &pattern(heads * width, 50)),
            Tensor::new(
                DType::F32,
                &[1, hd],
                &pattern(hd, 51).iter().map(|v| 1.0 + v).collect::<Vec<_>>(),
            ),
            Tensor::new(DType::F32, &[1, rot / 2], &pattern(rot / 2, 52)),
            Tensor::new(DType::F32, &[1, rot / 2], &pattern(rot / 2, 53)),
            Tensor::zeros(DType::F16, &[heads, hd]),
        ];
        if gate {
            t.push(Tensor::zeros(DType::F32, &[heads, hd]));
        }
        let outs: &[usize] = if gate { &[4, 5] } else { &[4] };
        for &out in outs {
            same(&gpu, &prog, t.clone(), &[], out, 256);
        }
    }
    let n = 24 * hd;
    same(
        &gpu,
        &build_mul(n, 256).unwrap(),
        vec![
            Tensor::new(DType::F32, &[1, n], &pattern(n, 54)),
            Tensor::new(DType::F32, &[1, n], &pattern(n, 55)),
            Tensor::zeros(DType::F32, &[1, n]),
        ],
        &[],
        2,
        256,
    );
}

/// Qwen3.5's gates and its depthwise convolution, at the model's shapes.
#[test]
fn delta_gates_and_conv_match_the_interpreter() {
    use tile_front::qwen::{build_conv_silu, build_gates};
    let gpu = Gpu::open().expect("metal device");
    let (hv, dv) = (48usize, 128usize);
    // `dt_bias` reaches 19 in the real weights: softplus must stay linear
    // there rather than overflowing through `exp`.
    let dt: Vec<f32> = (0..hv)
        .map(|i| -6.0 + 25.0 * (i as f32) / hv as f32)
        .collect();
    let gates = vec![
        Tensor::new(DType::F32, &[hv], &pattern(hv, 40)),
        Tensor::new(DType::F32, &[hv], &pattern(hv, 41)),
        Tensor::new(
            DType::F32,
            &[hv],
            &pattern(hv, 42)
                .iter()
                .map(|x| 0.5 + x.abs())
                .collect::<Vec<_>>(),
        ),
        Tensor::new(DType::F32, &[hv], &dt),
        Tensor::zeros(DType::F32, &[hv, dv]),
        Tensor::zeros(DType::F32, &[hv, dv]),
    ];
    for out in [4, 5] {
        same(&gpu, &build_gates(hv, dv), gates.clone(), &[], out, 64);
    }

    let (ch, kern) = (10240usize, 4usize);
    let conv = vec![
        Tensor::new(DType::F32, &[kern - 1, ch], &pattern((kern - 1) * ch, 43)),
        Tensor::new(DType::F32, &[1, ch], &pattern(ch, 44)),
        Tensor::new(DType::F32, &[kern, ch], &pattern(kern * ch, 45)),
        Tensor::zeros(DType::F32, &[1, ch]),
    ];
    // The output, and the window the step leaves behind.
    for out in [3, 0] {
        same(
            &gpu,
            &build_conv_silu(ch, kern, 256).unwrap(),
            conv.clone(),
            &[],
            out,
            256,
        );
    }
}

/// Qwen3.5's gated-delta state update, at the model's own shape.
#[test]
fn delta_state_matches_the_interpreter() {
    use tile_front::qwen::DeltaNet;
    let gpu = Gpu::open().expect("metal device");
    let c = DeltaNet::packed(48, 16, 128, 128, 8);
    let prog = c.build_step().unwrap();
    let (hv, dk, dv) = (c.v_heads, c.k_dim, c.v_dim);
    let gates = |seed, lo: f32, hi: f32| -> Vec<f32> {
        pattern(hv * dv, seed)
            .iter()
            .map(|x| lo + (hi - lo) * x.abs().min(1.0))
            .collect()
    };
    let t = vec![
        Tensor::new(DType::F32, &[hv * dv, dk], &pattern(hv * dv * dk, 30)),
        Tensor::new(DType::F32, &[hv, dk], &pattern(hv * dk, 31)),
        Tensor::new(DType::F32, &[hv, dk], &pattern(hv * dk, 32)),
        Tensor::new(DType::F32, &[hv * dv], &pattern(hv * dv, 33)),
        Tensor::new(DType::F32, &[hv * dv], &gates(34, 0.5, 1.0)),
        Tensor::new(DType::F32, &[hv * dv], &gates(35, 0.1, 0.9)),
        Tensor::zeros(DType::F32, &[hv * dv]),
    ];
    // The output, and the state the step leaves behind.
    for out in [6, 0] {
        same(&gpu, &prog, t.clone(), &[], out, 128);
    }
}

/// NVFP4 (Qwen3.5's MLX weights): E2M1 codes two per byte, an FP8 E4M3
/// scale per 16, one f32 scale per tensor held per row.
#[test]
fn nvfp4_matvec_matches_the_interpreter() {
    let gpu = Gpu::open().expect("metal device");
    let (n_in, n_out) = (512, 64);
    // Scale bytes with a sane exponent (no NaN, no subnormals).
    let scales: Vec<f32> = pattern(n_out * n_in / 16, 21)
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let e = 4 + (i % 8) as u8;
            let m = ((v.abs() * 8.0) as u8) & 7;
            ((e << 3) | m) as i8 as f32
        })
        .collect();
    let codes: Vec<f32> = pattern(n_out * n_in / 2, 22)
        .iter()
        .map(|v| ((v + 1.0) * 127.5) as u8 as i8 as f32)
        .collect();
    for rows in [1usize, 4] {
        let t = vec![
            Tensor::new(DType::F32, &[rows, n_in], &pattern(rows * n_in, 23)),
            Tensor::new(DType::I8, &[n_out, n_in / 2], &codes),
            Tensor::new(DType::I8, &[n_out, n_in / 16], &scales),
            Tensor::new(DType::F32, &[n_out], &vec![0.0125f32; n_out]),
            Tensor::zeros(DType::F32, &[rows, n_out]),
        ];
        let p = matmul_q(rows, n_in, n_out, 8, n_in, QLayout::NVFP4, false).unwrap();
        same(&gpu, &p, t, &[], 4, 256);
    }
}

/// The batched forward pass's kernels (prefill, speculative verify).
#[test]
fn batched_kernels_match_the_interpreter() {
    use tile_front::flash::FlashDecode;
    use tile_front::llama::{kv_append_rows, rmsnorm_rows, rope_rows};
    use tile_ir::Space;
    let gpu = Gpu::open().expect("metal device");
    let (t, n) = (5usize, 4096usize);
    for pick in [None, Some(t - 1)] {
        let out_rows = if pick.is_some() { 1 } else { t };
        same(
            &gpu,
            &rmsnorm_rows(t, n, 1e-5, pick),
            vec![
                Tensor::new(DType::F32, &[t, n], &pattern(t * n, 1)),
                Tensor::new(DType::F32, &[1, n], &pattern(n, 2)),
                Tensor::zeros(DType::F32, &[out_rows, n]),
            ],
            &[],
            2,
            256,
        );
    }
    let (heads, hd) = (8usize, 128usize);
    let mut cos = vec![];
    let mut sin = vec![];
    for pos in 0..t {
        let (c, s) = rope_tables(20 + pos, hd, 500000.0, None);
        cos.extend(c);
        sin.extend(s);
    }
    same(
        &gpu,
        &rope_rows(t, heads, hd, DType::F16),
        vec![
            Tensor::new(DType::F32, &[t * heads, hd], &pattern(t * heads * hd, 3)),
            Tensor::new(DType::F32, &[t, hd], &cos),
            Tensor::new(DType::F32, &[t, hd], &sin),
            Tensor::zeros(DType::F16, &[t * heads, hd]),
        ],
        &[],
        3,
        256,
    );
    let cap = 64;
    for dt in [DType::F16, DType::F32] {
        same(
            &gpu,
            &kv_append_rows(t, heads, hd, cap, dt),
            vec![
                Tensor::new(dt, &[t * heads, hd], &pattern(t * heads * hd, 4)),
                Tensor::zeros(DType::F16, &[heads * cap, hd]),
            ],
            &[30],
            1,
            64,
        );
    }
    let group = 4;
    let cfg = FlashDecode {
        q_rows: group,
        d: hd,
        seq: cap,
        bq: group,
        bk: 16,
        stages: 1,
        dtype: DType::F16,
        kv_space: Space::Threadgroup,
        consumers: 0,
        heads,
        kv_cap: cap,
    };
    let hg = heads * group;
    let prog = cfg.build_causal(t).unwrap();
    let pos0 = 30usize;
    same(
        &gpu,
        &prog,
        vec![
            Tensor::new(DType::F16, &[t * hg, hd], &pattern(t * hg * hd, 5)),
            Tensor::new(
                DType::F16,
                &[heads * cap, hd],
                &pattern(heads * cap * hd, 6),
            ),
            Tensor::new(
                DType::F16,
                &[heads * cap, hd],
                &pattern(heads * cap * hd, 7),
            ),
            Tensor::zeros(DType::F32, &[t * hg, hd]),
        ],
        &[pos0 as u32, (pos0 + t).div_ceil(16) as u32],
        3,
        128,
    );
}
