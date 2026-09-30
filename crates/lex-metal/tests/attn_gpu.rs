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
