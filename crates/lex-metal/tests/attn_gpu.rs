//! Causal prefill attention on the matrix units (`lex_msl::attn`) against
//! the interpreter running the program it stands in for
//! (`FlashDecode::build_causal_blocks`, same parameters and scalars). `P`
//! is rounded to half for `P V` where the program keeps f32, so they agree
//! to half's precision on values in [0, 1], not to the bit.

#![cfg(target_os = "macos")]

use half::f16;
use lex_front::flash::FlashDecode;
use lex_front::{Tensor, run_dyn};
use lex_ir::reference::fill_pattern_f32;
use lex_ir::{DType, Space};
use lex_metal::{Buffer, Gpu};
use lex_msl::attn::{Causal, causal_mma};

#[test]
fn causal_attention_on_the_matrix_units_matches_the_program() {
    let Some(gpu) = Gpu::open().ok() else {
        eprintln!("no Metal device");
        return;
    };
    let (heads, group, d, cap) = (2usize, 6usize, 64usize, 128usize);
    let hg = heads * group;
    // (tokens, pos0): from the start; after a cached prefix; a chunk that
    // does not fill its last tile of four; one that ends at the cache's end.
    for (tokens, pos0) in [(16usize, 0usize), (32, 40), (13, 70), (24, 104)] {
        let cfg = FlashDecode {
            q_rows: group,
            d,
            seq: cap,
            bq: group,
            bk: 16,
            stages: 1,
            dtype: DType::F16,
            kv_space: Space::Reg,
            consumers: 0,
            heads,
            kv_cap: cap,
        };
        let prog = cfg.build_causal_blocks(tokens, 1).expect("program");
        let half_of = |n: usize, seed: u32, scale: f32| {
            let mut x = vec![0.0; n];
            fill_pattern_f32(&mut x, seed);
            // Rounded to f16 up front, so both sides read the same values.
            x.iter()
                .map(|v| f16::from_f32(v * scale).to_f32())
                .collect::<Vec<f32>>()
        };
        let q = half_of(tokens * hg * d, 3, 0.5);
        let k = half_of(heads * cap * d, 5, 0.5);
        let v = half_of(heads * cap * d, 7, 1.0);
        let mut tensors = vec![
            Tensor::new(DType::F16, &[tokens * hg, d], &q),
            Tensor::new(DType::F16, &[heads * cap, d], &k),
            Tensor::new(DType::F16, &[heads * cap, d], &v),
            Tensor::zeros(DType::F32, &[tokens * hg, d]),
        ];
        let nkb = (pos0 + tokens).div_ceil(16) as u32;
        let up16 =
            |t: &Tensor| gpu.upload(&t.data.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>());
        let bufs: Vec<Buffer> = vec![
            up16(&tensors[0]),
            up16(&tensors[1]),
            up16(&tensors[2]),
            gpu.zeroed::<f32>(tokens * hg * d),
            gpu.upload(&[pos0 as u32, nkb]),
        ];
        let c = Causal {
            tokens,
            kv_heads: heads,
            group,
            head_dim: d,
            cap,
        };
        let pipe = gpu
            .build_lowered(&causal_mma(&c).expect("kernel"))
            .expect("compile");
        let refs: Vec<&Buffer> = bufs.iter().collect();
        gpu.run(&pipe, &refs);
        let mut got = vec![0.0f32; tokens * hg * d];
        gpu.download(&bufs[3], &mut got);

        run_dyn(&prog, &mut tensors, &[pos0 as u32, nkb]).expect("interpret");
        let want = &tensors[3].data;
        let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
        assert!(scale > 0.1, "the reference is all but zero ({scale})");
        let err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!("tokens {tokens}, pos0 {pos0}: {err:e} of scale");
        assert!(err < 3e-3, "tokens {tokens} pos0 {pos0}: {err:e} of scale");
    }
}

/// The split verify kernel (`causal_mma_split`) against the program it
/// stands in for (`build_causal_split`), each merged by the same combine
/// (`build_combine_rows`, interpreted): partials in the same layout, so the
/// final rows must agree. Covers a single token, a batch that does not fill
/// its tile, and positions where the last split is partly live.
#[test]
fn split_verify_attention_on_the_matrix_units_matches_the_program() {
    use lex_front::flash::COMBINE_CHUNK;
    use lex_msl::attn::causal_mma_split;
    let Some(gpu) = Gpu::open().ok() else {
        eprintln!("no Metal device");
        return;
    };
    let (heads, group, d, cap, bk, bps) = (2usize, 6usize, 64usize, 256usize, 16usize, 2usize);
    let (hg, span) = (heads * group, bk * bps);
    let splits = cap / span;
    for (tokens, pos0) in [(1usize, 0usize), (2, 37), (3, 100), (4, 220)] {
        let cfg = FlashDecode {
            q_rows: group,
            d,
            seq: cap,
            bq: group,
            bk,
            stages: 1,
            dtype: DType::F16,
            kv_space: Space::Reg,
            consumers: 0,
            heads,
            kv_cap: cap,
        };
        let half_of = |n: usize, seed: u32, scale: f32| {
            let mut x = vec![0.0; n];
            fill_pattern_f32(&mut x, seed);
            x.iter()
                .map(|v| f16::from_f32(v * scale).to_f32())
                .collect::<Vec<f32>>()
        };
        let rows = tokens * hg;
        let q = half_of(rows * d, 13, 0.5);
        let k = half_of(heads * cap * d, 15, 0.5);
        let v = half_of(heads * cap * d, 17, 1.0);
        let nsplit = (pos0 + tokens).div_ceil(span);
        let combine = cfg.build_combine_rows(tokens, bps).expect("combine");
        let merge = |m: &[f32], l: &[f32], acc: &[f32]| {
            let mut t = vec![
                Tensor::new(DType::F32, &[rows, splits], m),
                Tensor::new(DType::F32, &[rows, splits], l),
                Tensor::new(DType::F32, &[rows * splits, d], acc),
                Tensor::zeros(DType::F32, &[rows, d]),
            ];
            run_dyn(
                &combine,
                &mut t,
                &[nsplit as u32, nsplit.div_ceil(COMBINE_CHUNK) as u32],
            )
            .expect("combine");
            t[3].data.clone()
        };

        // The program's partials, interpreted.
        let prog = cfg.build_causal_split(tokens, bps).expect("program");
        let mut t = vec![
            Tensor::new(DType::F16, &[rows, d], &q),
            Tensor::new(DType::F16, &[heads * cap, d], &k),
            Tensor::new(DType::F16, &[heads * cap, d], &v),
            Tensor::zeros(DType::F32, &[rows, splits]),
            Tensor::zeros(DType::F32, &[rows, splits]),
            Tensor::zeros(DType::F32, &[rows, splits * d]),
        ];
        run_dyn(&prog, &mut t, &[pos0 as u32]).expect("interpret");
        let want = merge(&t[3].data, &t[4].data, &t[5].data);

        // The kernel's, on the GPU, through the same combine.
        let c = Causal {
            tokens,
            kv_heads: heads,
            group,
            head_dim: d,
            cap,
        };
        let pipe = gpu
            .build_lowered(&causal_mma_split(&c, span).expect("kernel"))
            .expect("compile");
        let up16 = |x: &[f32]| gpu.upload(&x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>());
        let bufs: Vec<Buffer> = vec![
            up16(&q),
            up16(&k),
            up16(&v),
            gpu.zeroed::<f32>(rows * splits),
            gpu.zeroed::<f32>(rows * splits),
            gpu.zeroed::<f32>(rows * splits * d),
            gpu.upload(&[pos0 as u32]),
        ];
        let refs: Vec<&Buffer> = bufs.iter().collect();
        gpu.run(&pipe, &refs);
        let get = |b: &Buffer, n: usize| {
            let mut x = vec![0.0f32; n];
            gpu.download(b, &mut x);
            x
        };
        let got = merge(
            &get(&bufs[3], rows * splits),
            &get(&bufs[4], rows * splits),
            &get(&bufs[5], rows * splits * d),
        );

        let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
        assert!(scale > 0.1, "the reference is all but zero ({scale})");
        let err = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!("split verify, tokens {tokens}, pos0 {pos0}: {err:e} of scale");
        assert!(err < 3e-3, "tokens {tokens} pos0 {pos0}: {err:e} of scale");
    }
}
