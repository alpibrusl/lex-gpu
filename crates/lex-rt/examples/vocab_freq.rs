//! How often each token of the vocabulary occurs in a corpus: the ranking a
//! draft head scored over only the common tokens needs (lex-gpu#26).
//!
//!     find ~/src -name '*.rs' | cargo run --release -p lex-rt --example vocab_freq -- \
//!         --out counts.bin
//!
//! Reads file paths from standard input, tokenizes each with the model's own
//! tokenizer, and writes one little-endian u64 count per token id. Files over
//! `--max-bytes` are skipped (a vendored bundle says little about what gets
//! written), as are files that are not UTF-8.

use std::io::{BufRead, Write};

fn main() -> Result<(), String> {
    let mut model = "qwen3.8:27b-mlx".to_string();
    let mut out = "counts.bin".to_string();
    let mut max_bytes = 200_000usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--out" => out = val()?,
            "--max-bytes" => max_bytes = val()?.parse().map_err(|_| "bad --max-bytes")?,
            _ => return Err(format!("unknown argument {a}")),
        }
    }
    let tok = lex_rt::tokenizer::Tokenizer::for_model(&model)?;
    let mut counts: Vec<u64> = vec![];
    let (mut files, mut tokens) = (0usize, 0u64);
    for path in std::io::stdin().lock().lines() {
        let path = path.map_err(|e| e.to_string())?;
        let Ok(meta) = std::fs::metadata(&path) else { continue };
        if meta.len() as usize > max_bytes {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        for id in tok.encode(&text) {
            let i = id as usize;
            if i >= counts.len() {
                counts.resize(i + 1, 0);
            }
            counts[i] += 1;
            tokens += 1;
        }
        files += 1;
    }
    let mut f = std::fs::File::create(&out).map_err(|e| format!("{out}: {e}"))?;
    for c in &counts {
        f.write_all(&c.to_le_bytes()).map_err(|e| e.to_string())?;
    }
    let seen = counts.iter().filter(|&&c| c > 0).count();
    eprintln!("{files} files, {tokens} tokens, {seen} distinct ids -> {out}");
    Ok(())
}
