//! Our byte-level BPE against the implementation `tokenizer.json` is
//! written for.
//!
//! The tokenizer was written from the format and not ported from a
//! reference -- this repository is EUPL-1.2 and the references are
//! Apache-2.0 -- so nothing about it is obviously right. The fixture is
//! what makes that claim checkable: `scripts/tokenizer_fixture.py` runs
//! the reference over a spread of awkward text and writes down the ids,
//! and these compare against them exactly.
//!
//! Exactly, not approximately: a tokenizer that is nearly right produces
//! a model that is subtly wrong in a way no logit comparison would catch,
//! because the logits would be correct for the tokens it actually sent.

use lex_rt::tokenizer::Tokenizer;

const FIXTURE: &str = include_str!("data/qwen_tokenizer.txt");
const MODEL: &str = "qwen3.8:27b-mlx";
/// The same tokenizer, carried as GGUF metadata instead of a
/// `tokenizer.json`: MiMo-v2.6's arrays are Qwen3.8's JSON exactly, token
/// for token and merge for merge. So the reference's ids for Qwen3.8 are
/// the reference's ids for MiMo, and the GGUF reader is held to them.
const GGUF_MODEL: &str = "maternion/mimo-v2.6:9b";

/// `"..."` as the fixture writes it, back to a string.
fn unquote(s: &str) -> String {
    let b: Vec<char> = s.trim().chars().collect();
    assert!(
        b.len() >= 2 && b[0] == '"' && b[b.len() - 1] == '"',
        "not quoted: {s}"
    );
    let mut out = String::new();
    let mut i = 1;
    while i < b.len() - 1 {
        if b[i] == '\\' {
            i += 1;
            match b[i] {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let hex: String = b[i + 1..i + 5].iter().collect();
                    let c = u32::from_str_radix(&hex, 16).expect("\\u escape");
                    out.push(char::from_u32(c).expect("a char"));
                    i += 4;
                }
                other => out.push(other),
            }
        } else {
            out.push(b[i]);
        }
        i += 1;
    }
    out
}

fn cases() -> Vec<(String, Vec<u32>)> {
    FIXTURE
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let (text, ids) = l.split_once('\t').expect("text<TAB>ids");
            let ids = ids
                .split(',')
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().parse().expect("an id"))
                .collect();
            (unquote(text), ids)
        })
        .collect()
}

fn encodes_like_the_reference(model: &str) {
    let tok = match Tokenizer::for_model(model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };
    let cases = cases();
    assert!(cases.len() >= 20, "the fixture is too small to mean much");
    let mut bad = vec![];
    for (text, want) in &cases {
        let got = tok.encode(text);
        if &got != want {
            bad.push(format!("  {text:?}\n    want {want:?}\n    got  {got:?}"));
        }
    }
    assert!(
        bad.is_empty(),
        "{model}: {} of {} cases differ from the reference:\n{}",
        bad.len(),
        cases.len(),
        bad.join("\n")
    );
    eprintln!("{model}: {} cases match the reference exactly", cases.len());
}

#[test]
fn encodes_what_the_reference_encodes() {
    encodes_like_the_reference(MODEL);
}

/// Read from the GGUF header, never touching the 5.6 GB of tensors.
#[test]
fn a_gguf_tokenizer_encodes_what_the_reference_encodes() {
    encodes_like_the_reference(GGUF_MODEL);
}

fn decodes_the_references_ids(model: &str) {
    let tok = match Tokenizer::for_model(model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIPPED: {model} ({e})");
            return;
        }
    };
    // Round-tripping the reference's *own* ids, so this tests the decoder
    // rather than agreeing with whatever the encoder just did.
    //
    // Against `nfc(text)` and not `text`: the tokenizer normalises, and
    // NFC is lossy. `e` followed by a combining acute encodes to the
    // token for the precomposed character, and no decoder can put the
    // two code points back. The reference does not round-trip that case
    // either, and a test demanding it would be demanding a bug.
    for (text, ids) in cases() {
        assert_eq!(
            tok.decode(&ids),
            lex_rt::tokenizer::nfc(&text),
            "{model}: decoding {ids:?}"
        );
    }
}

#[test]
fn decoding_undoes_encoding() {
    decodes_the_references_ids(MODEL);
}

#[test]
fn a_gguf_tokenizer_decodes_the_references_ids() {
    decodes_the_references_ids(GGUF_MODEL);
}

/// The control markers the chat template is built from, by the ids the
/// model was trained on. They are matched whole before any splitting, and
/// a marker that fell through to BPE would come apart into punctuation.
#[test]
fn a_gguf_tokenizer_keeps_the_chat_markers_whole() {
    let tok = match Tokenizer::for_model(GGUF_MODEL) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("SKIPPED: {GGUF_MODEL} ({e})");
            return;
        }
    };
    for (marker, id) in [
        ("<|endoftext|>", 248044),
        ("<|im_start|>", 248045),
        ("<|im_end|>", 248046),
    ] {
        assert_eq!(tok.encode(marker), vec![id], "{marker}");
        assert_eq!(tok.decode(&[id]), marker, "{marker}");
    }
}
