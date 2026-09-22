//! Qwen3.5's gated-delta state update, against the rule it implements.

use tile_front::qwen::{DeltaNet, reference};
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
    let c = DeltaNet {
        v_heads: 6,
        k_heads: 2,
        k_dim: 32,
        v_dim: 16,
        rows: 4,
    };
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
    let c = DeltaNet {
        v_heads: 2,
        k_heads: 1,
        k_dim: 16,
        v_dim: 8,
        rows: 8,
    };
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
