//! What a Bonsai 2 GGUF actually contains, read with our own loader.
//!
//! Prism's ternary types have no public specification; `docs/ternary.md`
//! records how the layouts were derived from the weight files. This walks
//! the real 27B checkpoint with that reading and reports whether every
//! tensor lands where the derivation says it should -- the first check of
//! the format against the whole file rather than a fixture.
//!
//!     cargo run --release -p lex-rt --example bonsai -- <path to gguf>

use std::collections::BTreeMap;
use std::path::PathBuf;

use lex_front::ir::Trits;
use lex_front::llama::split_ternary;
use lex_rt::gguf::{GgmlType, Gguf, Value};

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).map(PathBuf::from).ok_or(
        "usage: bonsai <path to a Ternary-Bonsai GGUF>".to_string(),
    )?;
    let g = Gguf::open(&path)?;

    let s = |k: &str| match g.meta.get(k) {
        Some(Value::Str(v)) => v.clone(),
        Some(v) => format!("{v:?}"),
        None => "-".into(),
    };
    println!("architecture     {}", s("general.architecture"));
    for k in [
        "prism.hadamard.transform",
        "prism.hadamard.block_size",
        "prism.hadamard.axis",
        "prism.hadamard.sign_mode",
    ] {
        println!("{k:<24} {}", s(k));
    }

    // Every tensor, by type, and whether our block arithmetic explains its
    // size exactly.
    let mut by_type: BTreeMap<String, (usize, usize)> = BTreeMap::new();
    let mut unreadable = vec![];
    for (name, t) in &g.tensors {
        let e = by_type.entry(format!("{:?}", t.ty)).or_default();
        e.0 += 1;
        match g.raw(name) {
            Ok((_, b)) => e.1 += b.len(),
            Err(err) => unreadable.push(err),
        }
    }
    println!("\ntensors by type (count, bytes):");
    for (ty, (n, bytes)) in &by_type {
        println!("  {ty:<10} {n:>4}  {:>14.3} GB", *bytes as f64 / 1e9);
    }
    if unreadable.is_empty() {
        println!("  every tensor's bytes are accounted for by its block size");
    } else {
        println!("  {} unreadable:", unreadable.len());
        for e in unreadable.iter().take(5) {
            println!("    {e}");
        }
        return Err("the loader does not explain this file".into());
    }

    // The BF16 tensors: check they really are bf16 and not something else
    // of the same width, by looking at what the bytes decode to.
    if let Some((name, _)) = g.tensors.iter().find(|(_, t)| t.ty == GgmlType::BF16) {
        let (t, b) = g.raw(name)?;
        let v: Vec<f32> = b
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from_bits((u16::from_le_bytes(*c) as u32) << 16))
            .collect();
        let finite = v.iter().filter(|x| x.is_finite()).count();
        let rms = (v.iter().filter(|x| x.is_finite()).map(|x| x * x).sum::<f32>()
            / finite.max(1) as f32)
            .sqrt();
        println!(
            "\n{name} {:?} as bf16: {finite}/{} finite, rms {rms:.5}, \
             max |x| {:.5}",
            t.dims,
            v.len(),
            v.iter().filter(|x| x.is_finite()).fold(0.0f32, |a, x| a.max(x.abs()))
        );
    }

    // Dequantise a row of a real ternary matrix and check it is what a
    // ternary format should give: three values, evenly spread.
    for name in ["blk.0.ffn_gate.weight", "blk.0.attn_qkv.weight", "output.weight"] {
        let (t, b) = g.raw(name)?;
        let trits = match t.ty {
            GgmlType::PTQ1_0 => Trits::Dense,
            GgmlType::PQ2_0 => Trits::Slots2,
            other => {
                println!("\n{name}: {other:?}, not ternary");
                continue;
            }
        };
        let cols = t.dims[0];
        let sp = split_ternary(&b[..cols / 128 * (trits.bytes() + 2)], trits);
        let row = sp.dequant(0, cols);
        let mut counts = [0usize; 3];
        for (i, v) in row.iter().enumerate() {
            let g = i / 128;
            let scale = sp.scale(g);
            counts[((v / scale).round() as i32 + 1).clamp(0, 2) as usize] += 1;
        }
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / cols as f32).sqrt();
        println!(
            "\n{name:<24} {:?} {trits:?}\n  row 0 of {cols}: -1/0/+1 = {:?}, rms {rms:.5}, \
             scales {:.5}..{:.5}",
            t.dims,
            counts,
            (0..cols / 128).map(|g| sp.scale(g)).fold(f32::MAX, f32::min),
            (0..cols / 128).map(|g| sp.scale(g)).fold(0.0f32, f32::max),
        );
    }
    Ok(())
}
