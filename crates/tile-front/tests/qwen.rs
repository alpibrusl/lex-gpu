//! Qwen3.5's gated-delta state update, against the rule it implements.

use tile_front::qwen::{
    DeltaNet, build_conv_silu, build_delta_qk, build_gated_norm, build_gates, build_matvec_dense,
    build_qk_rope, reference,
};
use tile_front::{Tensor, check, run};
use tile_ir::reference::fill_pattern_f32;
use tile_ir::{DType, Target};

fn pattern(n: usize, seed: u32) -> Vec<f32> {
    let mut v = vec![0.0; n];
    fill_pattern_f32(&mut v, seed);
    v
}

/// Several steps in a row: the state carries between them, so an error in
/// the decay or the rank-one update compounds instead of cancelling.
#[test]
fn delta_state_matches_the_rule_over_successive_steps() {
    // Qwen3.5-27B's shape, shrunk: the same 3:1 value/key head ratio.
    let c = DeltaNet::packed(6, 2, 32, 16, 4);
    let prog = c.build_step().unwrap();
    check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));

    let (hv, dk, dv) = (c.v_heads, c.k_dim, c.v_dim);
    let mut state = vec![0.0f32; hv * dv * dk];
    let mut want = state.clone();
    for step in 0..4u32 {
        let q = pattern(hv * dk, 10 + step);
        let k = pattern(hv * dk, 20 + step);
        let v = pattern(hv * dv, 30 + step);
        // Gates in their real ranges: a decay in (0, 1], beta in (0, 1).
        let g: Vec<f32> = pattern(hv * dv, 40 + step)
            .iter()
            .map(|x| 0.5 + 0.5 * x.abs().min(1.0))
            .collect();
        let beta: Vec<f32> = pattern(hv * dv, 50 + step)
            .iter()
            .map(|x| 0.5 * (1.0 + x.abs().min(1.0)))
            .collect();

        let mut t = vec![
            Tensor::new(DType::F32, &[hv * dv, dk], &state),
            Tensor::new(DType::F32, &[hv, dk], &q),
            Tensor::new(DType::F32, &[hv, dk], &k),
            Tensor::new(DType::F32, &[hv * dv], &v),
            Tensor::new(DType::F32, &[hv * dv], &g),
            Tensor::new(DType::F32, &[hv * dv], &beta),
            Tensor::zeros(DType::F32, &[hv * dv]),
        ];
        run(&prog, &mut t).expect("interpret");

        let y = reference(&c, &mut want, &q, &k, &v, &g, &beta);
        // Against each tensor's own scale: the kernel accumulates in f32
        // and the reference in f64, and a state element that cancels to
        // near zero would make a per-element relative error meaningless.
        let err = |a: &[f32], b: &[f32]| {
            let d = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            d / b.iter().map(|x| x.abs()).fold(1e-6f32, f32::max)
        };
        assert!(
            err(&t[6].data, &y) < 1e-5,
            "step {step}: output err {:e} of scale",
            err(&t[6].data, &y)
        );
        assert!(
            err(&t[0].data, &want) < 1e-5,
            "step {step}: state err {:e} of scale",
            err(&t[0].data, &want)
        );
        state = t[0].data.clone();
    }
}

/// A decayed state forgets: with `g` small and `beta` one, the output after
/// a step depends on the new key and value, not on what was there before.
#[test]
fn a_small_decay_forgets_the_old_state() {
    let c = DeltaNet::packed(2, 1, 16, 8, 8);
    let prog = c.build_step().unwrap();
    let (hv, dk, dv) = (c.v_heads, c.k_dim, c.v_dim);
    let run_once = |state: &[f32], g: f32| {
        let mut t = vec![
            Tensor::new(DType::F32, &[hv * dv, dk], state),
            Tensor::new(DType::F32, &[hv, dk], &pattern(hv * dk, 1)),
            Tensor::new(DType::F32, &[hv, dk], &pattern(hv * dk, 2)),
            Tensor::new(DType::F32, &[hv * dv], &pattern(hv * dv, 3)),
            Tensor::new(DType::F32, &[hv * dv], &vec![g; hv * dv]),
            Tensor::new(DType::F32, &[hv * dv], &vec![1.0; hv * dv]),
            Tensor::zeros(DType::F32, &[hv * dv]),
        ];
        run(&prog, &mut t).expect("interpret");
        t[6].data.clone()
    };
    let old = pattern(hv * dv * dk, 7);
    let fresh = vec![0.0f32; hv * dv * dk];
    let forgotten = run_once(&old, 0.0);
    let from_zero = run_once(&fresh, 0.0);
    let remembered = run_once(&old, 1.0);
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    assert!(
        diff(&forgotten, &from_zero) < 1e-5,
        "a zero decay should leave no trace of the old state"
    );
    assert!(
        diff(&remembered, &from_zero) > 1e-3,
        "an undecayed state should still be visible in the output"
    );
}

/// The gates, against their formulas, with a `dt_bias` in the range the
/// model actually uses (up to 19, where a naive `log(1 + exp(x))` would
/// lose the linear part or overflow).
#[test]
fn gates_match_their_formulas_across_the_models_range() {
    let (hv, dv) = (6usize, 8usize);
    let prog = build_gates(hv, dv);
    check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));
    let a: Vec<f32> = pattern(hv, 1).iter().map(|x| 3.0 * x).collect();
    let b: Vec<f32> = pattern(hv, 2).iter().map(|x| 4.0 * x).collect();
    let amp: Vec<f32> = pattern(hv, 3).iter().map(|x| 0.5 + x.abs()).collect();
    let dt: Vec<f32> = (0..hv)
        .map(|i| [-6.0, -1.0, 0.0, 1.0, 9.0, 19.0][i % 6])
        .collect();
    let mut t = vec![
        Tensor::new(DType::F32, &[hv], &a),
        Tensor::new(DType::F32, &[hv], &b),
        Tensor::new(DType::F32, &[hv], &amp),
        Tensor::new(DType::F32, &[hv], &dt),
        Tensor::zeros(DType::F32, &[hv, dv]),
        Tensor::zeros(DType::F32, &[hv, dv]),
    ];
    run(&prog, &mut t).expect("interpret");
    for h in 0..hv {
        let u = (a[h] + dt[h]) as f64;
        let sp = u.max(0.0) + (-u.abs()).exp().ln_1p();
        let g = (-(amp[h] as f64) * sp).exp() as f32;
        let beta = (1.0 / (1.0 + (-b[h] as f64).exp())) as f32;
        for r in 0..dv {
            let (gg, bb) = (t[4].data[h * dv + r], t[5].data[h * dv + r]);
            assert!(
                (gg - g).abs() <= 1e-6 * g.abs().max(1e-3),
                "head {h} row {r}: decay {gg} vs {g} (u = {u})"
            );
            assert!(
                (bb - beta).abs() < 1e-6,
                "head {h} row {r}: beta {bb} vs {beta}"
            );
        }
    }
}

/// The depthwise convolution, and the window it leaves behind: feeding a
/// sequence one row at a time must match convolving the whole sequence.
#[test]
fn conv_silu_matches_a_sliding_window() {
    let (channels, kernel, chunk) = (16usize, 4usize, 8usize);
    let prog = build_conv_silu(channels, kernel, chunk).unwrap();
    check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));
    let w = pattern(kernel * channels, 4);
    let steps = 5;
    let rows: Vec<Vec<f32>> = (0..steps)
        .map(|i| pattern(channels, 60 + i as u32))
        .collect();
    let mut state = vec![0.0f32; (kernel - 1) * channels];
    for (i, x) in rows.iter().enumerate() {
        let mut t = vec![
            Tensor::new(DType::F32, &[kernel - 1, channels], &state),
            Tensor::new(DType::F32, &[1, channels], x),
            Tensor::new(DType::F32, &[kernel, channels], &w),
            Tensor::zeros(DType::F32, &[1, channels]),
        ];
        run(&prog, &mut t).expect("interpret");
        // The same window, convolved directly: zeros before the sequence.
        for c in 0..channels {
            let mut acc = 0.0f64;
            for tap in 0..kernel {
                let pos = i as isize + tap as isize - (kernel - 1) as isize;
                let v = if pos < 0 {
                    0.0
                } else {
                    rows[pos as usize][c] as f64
                };
                acc += v * w[tap * channels + c] as f64;
            }
            let want = (acc / (1.0 + (-acc).exp())) as f32;
            let got = t[3].data[c];
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1e-2),
                "step {i} channel {c}: {got} vs {want}"
            );
        }
        state = t[0].data.clone();
    }
}

/// The attention prologue: a per-head norm, a rotation of the first
/// quarter of the head pairing `i` with `i + rot/2`, the rest untouched,
/// and the query's gate through a sigmoid.
#[test]
fn qk_rope_normalises_rotates_and_gates() {
    let (heads, hd, rot, eps) = (4usize, 16usize, 8usize, 1e-6f32);
    let half = rot / 2;
    for gate in [false, true] {
        let width = if gate { 2 * hd } else { hd };
        let prog = build_qk_rope(heads, hd, rot, gate, DType::F32, eps).unwrap();
        check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));
        let x = pattern(heads * width, 70);
        let nw: Vec<f32> = pattern(hd, 71).iter().map(|v| 1.0 + v).collect();
        let cos = pattern(half, 72);
        let sin = pattern(half, 73);
        let mut t = vec![
            Tensor::new(DType::F32, &[heads, width], &x),
            Tensor::new(DType::F32, &[1, hd], &nw),
            Tensor::new(DType::F32, &[1, half], &cos),
            Tensor::new(DType::F32, &[1, half], &sin),
            Tensor::zeros(DType::F32, &[heads, hd]),
        ];
        if gate {
            t.push(Tensor::zeros(DType::F32, &[heads, hd]));
        }
        run(&prog, &mut t).expect("interpret");

        for h in 0..heads {
            let row = &x[h * width..h * width + hd];
            let ms = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / hd as f64;
            let r = 1.0 / (ms + eps as f64).sqrt();
            let n: Vec<f64> = row
                .iter()
                .zip(&nw)
                .map(|(v, w)| *v as f64 * *w as f64 * r)
                .collect();
            for i in 0..hd {
                let want = if i < half {
                    n[i] * cos[i] as f64 - n[i + half] * sin[i] as f64
                } else if i < rot {
                    n[i] * cos[i - half] as f64 + n[i - half] * sin[i - half] as f64
                } else {
                    n[i]
                };
                let got = t[4].data[h * hd + i] as f64;
                assert!(
                    (got - want).abs() <= 1e-5 * want.abs().max(1e-3),
                    "gate {gate} head {h} dim {i}: {got} vs {want}"
                );
            }
            if gate {
                for i in 0..hd {
                    let g = x[h * width + hd + i] as f64;
                    let want = 1.0 / (1.0 + (-g).exp());
                    let got = t[5].data[h * hd + i] as f64;
                    assert!((got - want).abs() < 1e-6, "head {h} gate {i}");
                }
            }
        }
    }
}

/// The linear layer's queries and keys: normalised without a weight,
/// scaled, and repeated once per value head so the delta step can find
/// its key with an affine index.
#[test]
fn delta_qk_normalises_scales_and_repeats_per_value_head() {
    let (hk, per, dk) = (2usize, 3usize, 16usize);
    let width = 2 * hk * dk + 8; // queries, keys, then values
    let first = hk * dk; // the keys
    let scale = (dk as f32).powf(-0.5);
    let prog = build_delta_qk(hk, per, dk, width, first, scale, 1e-6).unwrap();
    check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));
    let x = pattern(width, 80);
    let mut t = vec![
        Tensor::new(DType::F32, &[1, width], &x),
        Tensor::zeros(DType::F32, &[hk * per, dk]),
    ];
    run(&prog, &mut t).expect("interpret");
    for h in 0..hk {
        let row = &x[first + h * dk..first + (h + 1) * dk];
        let ms = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / dk as f64;
        let r = 1.0 / (ms + 1e-6).sqrt();
        for copy in 0..per {
            let out = &t[1].data[(h * per + copy) * dk..(h * per + copy + 1) * dk];
            for (i, (&x, &got)) in row.iter().zip(out).enumerate() {
                let want = x as f64 * r * scale as f64;
                assert!(
                    (got as f64 - want).abs() <= 1e-5 * want.abs().max(1e-3),
                    "head {h} copy {copy} dim {i}: {got} vs {want}"
                );
            }
        }
    }
}

/// The gated norm that ends a linear-attention layer, and the small dense
/// matvec its gates come from.
#[test]
fn gated_norm_and_dense_matvec_match_their_formulas() {
    let (hv, dv, eps) = (3usize, 8usize, 1e-6f32);
    let prog = build_gated_norm(hv, dv, eps);
    check(&prog, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));
    let y = pattern(hv * dv, 90);
    let w = pattern(dv, 91);
    let z = pattern(hv * dv, 92);
    let mut t = vec![
        Tensor::new(DType::F32, &[hv, dv], &y),
        Tensor::new(DType::F32, &[1, dv], &w),
        Tensor::new(DType::F32, &[hv, dv], &z),
        Tensor::zeros(DType::F32, &[hv, dv]),
    ];
    run(&prog, &mut t).expect("interpret");
    for h in 0..hv {
        let row = &y[h * dv..(h + 1) * dv];
        let ms = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / dv as f64;
        let r = 1.0 / (ms + eps as f64).sqrt();
        for i in 0..dv {
            let zi = z[h * dv + i] as f64;
            let want = row[i] as f64 * r * w[i] as f64 * (zi / (1.0 + (-zi).exp()));
            let got = t[3].data[h * dv + i] as f64;
            assert!(
                (got - want).abs() <= 1e-5 * want.abs().max(1e-3),
                "head {h} dim {i}: {got} vs {want}"
            );
        }
    }

    let (n_in, n_out) = (32usize, 6usize);
    let prog = build_matvec_dense(n_in, n_out, 2, DType::F32).unwrap();
    let x = pattern(n_in, 93);
    let wd = pattern(n_in * n_out, 94);
    let mut t = vec![
        Tensor::new(DType::F32, &[1, n_in], &x),
        Tensor::new(DType::F32, &[n_out, n_in], &wd),
        Tensor::zeros(DType::F32, &[1, n_out]),
    ];
    run(&prog, &mut t).expect("interpret");
    for r in 0..n_out {
        let want: f64 = (0..n_in)
            .map(|i| x[i] as f64 * wd[r * n_in + i] as f64)
            .sum();
        let got = t[2].data[r] as f64;
        assert!(
            (got - want).abs() <= 1e-5 * want.abs().max(1e-3),
            "row {r}: {got} vs {want}"
        );
    }
}

/// A batch of tokens through the delta rule must land exactly where the
/// same tokens land one at a time: the recurrence is sequential, and the
/// batch only saves reading the state once.
#[test]
fn a_batch_of_delta_steps_equals_the_same_steps_one_by_one() {
    let tokens = 3usize;
    let c = DeltaNet::packed(6, 2, 32, 16, 4);
    let (hv, dk, dv) = (c.v_heads, c.k_dim, c.v_dim);
    let gates = hv * dv;
    let one = c.build_step().unwrap();
    let many = c.build_steps(tokens).unwrap();
    check(&many, &Target::apple_m_series()).unwrap_or_else(|e| panic!("{e:#?}"));

    let q = pattern(tokens * hv * dk, 100);
    let k = pattern(tokens * hv * dk, 101);
    let v = pattern(tokens * gates, 102);
    let g: Vec<f32> = pattern(tokens * gates, 103)
        .iter()
        .map(|x| 0.5 + 0.5 * x.abs().min(1.0))
        .collect();
    let beta: Vec<f32> = pattern(tokens * gates, 104)
        .iter()
        .map(|x| 0.5 * (1.0 + x.abs().min(1.0)))
        .collect();
    let start = pattern(hv * dv * dk, 105);

    // The batch.
    let mut batch = vec![
        Tensor::new(DType::F32, &[hv * dv, dk], &start),
        Tensor::new(DType::F32, &[tokens * hv, dk], &q),
        Tensor::new(DType::F32, &[tokens * hv, dk], &k),
        Tensor::new(DType::F32, &[tokens * gates], &v),
        Tensor::new(DType::F32, &[tokens * gates], &g),
        Tensor::new(DType::F32, &[tokens * gates], &beta),
        Tensor::zeros(DType::F32, &[tokens * gates]),
    ];
    run(&many, &mut batch).expect("batch");

    // The same tokens, one at a time.
    let mut state = start.clone();
    for t in 0..tokens {
        let slice = |v: &[f32], n: usize| v[t * n..(t + 1) * n].to_vec();
        let mut step = vec![
            Tensor::new(DType::F32, &[hv * dv, dk], &state),
            Tensor::new(DType::F32, &[hv, dk], &slice(&q, hv * dk)),
            Tensor::new(DType::F32, &[hv, dk], &slice(&k, hv * dk)),
            Tensor::new(DType::F32, &[gates], &slice(&v, gates)),
            Tensor::new(DType::F32, &[gates], &slice(&g, gates)),
            Tensor::new(DType::F32, &[gates], &slice(&beta, gates)),
            Tensor::zeros(DType::F32, &[gates]),
        ];
        run(&one, &mut step).expect("step");
        state = step[0].data.clone();
        let want = &step[6].data;
        let got = &batch[6].data[t * gates..(t + 1) * gates];
        let d = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(d == 0.0, "token {t}: batch output differs by {d:e}");
    }
    let d = batch[0]
        .data
        .iter()
        .zip(&state)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(d == 0.0, "the state after the batch differs by {d:e}");
}
