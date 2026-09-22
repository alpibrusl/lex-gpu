//! Every Llama layer kernel on the GPU against the interpreter, kernel by
//! kernel and layout by layout. The model-level test says *whether* the
//! GPU is right; this one says *which* kernel is not.
#![cfg(target_os = "macos")]

use half::f16;
use tile_front::llama::{
    QLayout, Split, kv_append, matvec_q, rmsnorm, rope, rope_tables, silu_mul,
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
    assert!(
        abs / scale < 1e-5,
        "{}: GPU vs interpreter max abs err {abs:e} on outputs up to {scale:e}",
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
            let sp = Split {
                layout,
                cols: n_in,
                pack: if layout.packed4 { n_in } else { 0 },
                q: if layout.packed4 {
                    pattern(n_in * n_out / 2, 4)
                        .iter()
                        .map(|v| ((v + 1.0) * 127.5) as u8 as i8)
                        .collect()
                } else {
                    pattern(n_in * n_out, 4)
                        .iter()
                        .map(|v| (v * 31.0).round() as i8)
                        .collect()
                },
                sc: if two_level { small(groups, 5) } else { vec![] },
                d: f16s(if two_level { supers } else { groups }, 6),
                mn: layout.min.then(|| (small(groups, 7), f16s(supers, 8))),
            };
            let mut t = vec![Tensor::new(DType::F32, &[1, n_in], &pattern(n_in, 3))];
            t.extend(sp.weight_tensors(n_out));
            if residual {
                t.push(Tensor::new(DType::F32, &[1, n_out], &pattern(n_out, 6)));
            }
            t.push(Tensor::zeros(DType::F32, &[1, n_out]));
            let out = t.len() - 1;
            let p = matvec_q(n_in, n_out, 8, n_in, layout, residual).unwrap();
            same(&gpu, &p, t, &[], out, 256);
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
