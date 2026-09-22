//! Reading Qwen3.5 out of the local Ollama store.
//!
//! Skips itself when the model is not pulled, as the Llama tests do.

use tile_rt::qwen::{Dtype, Store};

const MODEL: &str = "qwen3.8:27b-mlx";

fn store() -> Option<Store> {
    match Store::open(MODEL) {
        Ok(s) => Some(s),
        Err(e) => {
            eprintln!("SKIPPED: {e}");
            None
        }
    }
}

#[test]
fn reads_the_model_the_way_the_kernels_want_it() {
    let Some(s) = store() else { return };
    let cfg = s.config().expect("config");
    let get = |k: &str| cfg.get(k).and_then(tile_rt::json::Json::usize).expect(k);
    assert_eq!(get("hidden_size"), 5120);
    assert_eq!(get("num_hidden_layers"), 64);
    assert_eq!(get("full_attention_interval"), 4);
    assert_eq!(get("linear_num_value_heads"), 48);
    assert_eq!(get("head_dim"), 256);

    // An NVFP4 matrix arrives as the matvec's parameters: 4 bits of codes
    // and half a byte of scale per 16 values, with the tensor's scale
    // repeated down the rows.
    let w = s
        .nvfp4("model.language_model.layers.0.mlp.gate_proj.weight")
        .expect("gate_proj");
    assert_eq!((w.rows, w.cols), (17408, 5120));
    assert_eq!(w.codes.len(), w.rows * w.cols / 2);
    assert_eq!(w.scales.len(), w.rows * w.cols / 16);
    assert_eq!(w.row_scale.len(), w.rows);
    let gs = w.row_scale[0];
    assert!(gs > 0.0 && w.row_scale.iter().all(|&x| x == gs));

    // The embedding is plain bf16 and semantic: two spellings of the same
    // word sit close together, unrelated tokens do not.
    let (emb, shape) = s
        .floats("model.language_model.embed_tokens.weight")
        .expect("embeddings");
    assert_eq!(shape, vec![248320, 5120]);
    let row = |t: usize| &emb[t * 5120..(t + 1) * 5120];
    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-9)
    };
    // 760 is "The" and 561 " The"; 47358 is "France" and 9338 " France".
    for (a, b) in [(760, 561), (47358, 9338)] {
        let c = cos(row(a), row(b));
        assert!(c > 0.4, "{a} and {b} are the same word: cos {c}");
    }
    let unrelated = cos(row(760), row(9338));
    assert!(unrelated.abs() < 0.3, "unrelated tokens: cos {unrelated}");
}

#[test]
fn folds_what_the_kernels_expect_at_load() {
    let Some(s) = store() else { return };
    // Norm weights are stored as a delta from 1 and come back shifted.
    let raw_mean = |name: &str| {
        let (v, _) = s.floats(name).expect(name);
        v.iter().sum::<f32>() / v.len() as f32
    };
    let shifted = raw_mean("model.language_model.layers.0.input_layernorm.weight");
    assert!(
        (0.5..1.5).contains(&shifted),
        "a shifted norm weight should sit around 1, got {shifted}"
    );
    // The gated norm inside a linear-attention layer is not shifted, and
    // the checkpoint's own value is already near 1.
    let plain = raw_mean("model.language_model.layers.0.linear_attn.norm.weight");
    assert!((0.5..1.5).contains(&plain), "got {plain}");

    // `A_log` becomes `A = exp(A_log)`: the file's values are negative, so
    // the folded ones are positive and small.
    let (a, _) = s
        .floats("model.language_model.layers.0.linear_attn.A_log")
        .expect("A_log");
    assert_eq!(a.len(), 48);
    assert!(
        a.iter().all(|&x| x > 0.0 && x < 16.0),
        "A should be exp of a negative log: {:?}",
        &a[..4]
    );

    // The convolution weight is transposed to `[kernel, channels]`.
    let (c, shape) = s
        .floats("model.language_model.layers.0.linear_attn.conv1d.weight")
        .expect("conv1d");
    assert_eq!(shape, vec![4, 10240]);
    assert_eq!(c.len(), 4 * 10240);
}

#[test]
fn says_what_is_wrong_rather_than_panicking() {
    let Some(s) = store() else { return };
    assert!(s.raw("no.such.tensor").is_err());
    // A bf16 tensor has no NVFP4 companions.
    let e = match s.nvfp4("model.language_model.embed_tokens.weight") {
        Err(e) => e,
        Ok(_) => panic!("a bf16 tensor has no NVFP4 companions"),
    };
    assert!(e.contains("NVFP4"), "{e}");
    assert!(!s.has("nope"));
    assert_eq!(
        s.raw("model.language_model.layers.0.linear_attn.A_log")
            .expect("A_log")
            .1
            .dtype,
        Dtype::BF16
    );
}
