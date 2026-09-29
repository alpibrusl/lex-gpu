//! Byte-level BPE, as `tokenizer.json` describes it.
//!
//! Until now the runtime took token ids and gave token ids, and every
//! example in this repository has a comma-separated list of integers in
//! it. That is enough to check a kernel against a reference and not
//! enough for anything to use, so this is the piece between the two.
//!
//! Written from the format rather than ported from an implementation:
//! this repository is EUPL-1.2 and the reference tokenizers are Apache-2.0,
//! which cannot be copied in either direction. `tokenizer.json` is data,
//! and BPE is an algorithm; `scripts/tokenizer_fixture.py` pins the result
//! against the reference so "written from the format" has to mean the same
//! answers.
//!
//! What the format asks for, and what is implemented:
//!
//! - **Byte level.** Each of the 256 bytes becomes one character, so that
//!   arbitrary bytes survive a vocabulary of text. The 188 printable
//!   Latin-1 characters stand for themselves and the other 68 are lifted
//!   to U+0100 upward, which is why a token string is not the text.
//! - **A split, then merges within each piece.** The pattern is the
//!   GPT-4-shaped one: contractions case-insensitively, then letters,
//!   then a single digit at a time, then punctuation runs, then
//!   whitespace with a trailing-space special case. Merges never cross a
//!   piece boundary, which is what keeps " the" and "the" distinct.
//! - **`ignore_merges`.** When a whole piece is already in the vocabulary
//!   it is taken whole, without consulting merges at all.

use std::collections::HashMap;

use crate::json::Json;
use crate::nfc_table::{CCC, COMP};

/// A byte-level BPE tokenizer built from a `tokenizer.json`.
pub struct Tokenizer {
    /// Token text (in the byte-level alphabet, as UTF-8) to id.
    vocab: HashMap<String, u32>,
    /// Id to token text, for decoding.
    text: Vec<String>,
    /// Merge rank of an adjacent pair: lower merges first.
    ranks: HashMap<(String, String), usize>,
    /// Literal strings that are matched before anything else, longest
    /// first. `<|im_start|>` must not be split into punctuation runs.
    added: Vec<(String, u32)>,
    /// Take a piece whole when the vocabulary has it, skipping merges.
    ignore_merges: bool,
}

/// The byte-level alphabet: byte -> char, as the format defines it.
///
/// The printable Latin-1 characters stand for themselves; everything else
/// is lifted into the private-use-adjacent range at U+0100 so that a
/// token string is always printable and never whitespace that the split
/// would have divided.
fn byte_to_char() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut spare = 0u32;
    for (b, slot) in map.iter_mut().enumerate() {
        let b = b as u32;
        let printable =
            (0x21..=0x7E).contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b);
        *slot = if printable {
            char::from_u32(b).expect("latin-1 is a char")
        } else {
            let c = char::from_u32(0x100 + spare).expect("below the surrogates");
            spare += 1;
            c
        };
    }
    map
}

impl Tokenizer {
    /// Parse a `tokenizer.json`.
    pub fn from_json(src: &str) -> Result<Tokenizer, String> {
        let j = Json::parse(src)?;
        let model = j.get("model").ok_or("tokenizer.json has no `model`")?;
        match model.get("type").and_then(Json::str) {
            Some("BPE") => {}
            other => return Err(format!("unsupported tokenizer model {other:?}")),
        }

        let vocab_json = model.get("vocab").ok_or("the model has no `vocab`")?;
        let mut vocab = HashMap::new();
        let mut text: Vec<String> = vec![];
        for key in vocab_json.keys() {
            let id = vocab_json
                .get(key)
                .and_then(Json::usize)
                .ok_or_else(|| format!("vocab entry `{key}` is not an id"))?
                as u32;
            if text.len() <= id as usize {
                text.resize(id as usize + 1, String::new());
            }
            text[id as usize] = key.to_string();
            vocab.insert(key.to_string(), id);
        }

        // Merges are `"a b"` or `["a", "b"]` depending on the version that
        // wrote the file; a space is a legal token character, so a split
        // on the first space is wrong for the pair `" " + " "`. Rank is
        // the position, which is the whole of their meaning.
        let mut ranks = HashMap::new();
        if let Some(list) = model.get("merges").and_then(Json::arr) {
            for (rank, m) in list.iter().enumerate() {
                let pair = match m.arr() {
                    Some([a, b]) => match (a.str(), b.str()) {
                        (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                        _ => return Err(format!("merge {rank} is not a pair of strings")),
                    },
                    _ => {
                        let s = m
                            .str()
                            .ok_or_else(|| format!("merge {rank} is not a string"))?;
                        split_joined(s, &vocab).map_err(|e| format!("merge {rank} {e}"))?
                    }
                };
                ranks.entry(pair).or_insert(rank);
            }
        }

        let mut added: Vec<(String, u32)> = vec![];
        if let Some(list) = j.get("added_tokens").and_then(Json::arr) {
            for a in list {
                if let (Some(c), Some(id)) = (
                    a.get("content").and_then(Json::str),
                    a.get("id").and_then(Json::usize),
                ) {
                    added.push((c.to_string(), id as u32));
                    if text.len() <= id {
                        text.resize(id + 1, String::new());
                    }
                    // The content is literal text rather than the
                    // byte-level alphabet, and every character of a
                    // `<|...|>` marker maps to itself there, so decoding
                    // it byte-wise gives it back. Without this the id has
                    // no text at all and decodes to nothing.
                    text[id] = c.to_string();
                }
            }
        }
        // Longest first: `<|im_start|>` before any prefix of it.
        added.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));

        Ok(Tokenizer {
            ignore_merges: matches!(model.get("ignore_merges"), Some(Json::Bool(true))),
            vocab,
            text,
            ranks,
            added,
        })
    }

    /// The same tokenizer from a GGUF's metadata.
    ///
    /// A GGUF carries a byte-level BPE as parallel arrays rather than a
    /// `tokenizer.json`: the token texts by id, the merges in rank order,
    /// and a type per token. Checked against the real thing rather than
    /// assumed: MiMo-v2.6's arrays are Qwen3.8's `tokenizer.json` exactly --
    /// every token text, all 247587 merges in order, and its 33 control
    /// tokens are the JSON's 33 added tokens, id for id.
    ///
    /// What the arrays do not say is how to split text before merging.
    /// The name they give for it, `tokenizer.ggml.pre`, is a label rather
    /// than a pattern, and a pre-tokenizer that disagreed would tokenise
    /// wrongly without an error -- so this refuses any label it has not
    /// been checked against, instead of splitting every model the Qwen way.
    pub fn from_gguf(g: &crate::gguf::Gguf) -> Result<Tokenizer, String> {
        use crate::gguf::Value;
        let text_of = |k: &str| match g.meta.get(k) {
            Some(Value::Str(s)) => Some(s.as_str()),
            _ => None,
        };
        match text_of("tokenizer.ggml.model") {
            Some("gpt2") => {}
            other => return Err(format!("GGUF tokenizer {other:?} is not byte-level BPE")),
        }
        match text_of("tokenizer.ggml.pre") {
            Some("qwen35") => {}
            other => {
                return Err(format!(
                    "GGUF pre-tokenizer {other:?}: only qwen35 has been checked against this splitter"
                ));
            }
        }
        let strings = |k: &str| -> Result<Vec<String>, String> {
            match g.meta.get(k) {
                Some(Value::Array(v)) => v
                    .iter()
                    .map(|x| match x {
                        Value::Str(s) => Ok(s.clone()),
                        _ => Err(format!("{k} holds a non-string")),
                    })
                    .collect(),
                _ => Err(format!("GGUF has no {k}")),
            }
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let kinds: Vec<i64> = match g.meta.get("tokenizer.ggml.token_type") {
            Some(Value::Array(v)) => v.iter().map(|x| x.as_int().unwrap_or(1)).collect(),
            _ => vec![1; tokens.len()],
        };
        // llama.cpp's token types: 3 is control and 4 user-defined -- the
        // `<|...|>` markers, matched whole before anything else -- and 5 is
        // unused, the padding that rounds the embedding up to 248320. The
        // padding has text but no place in the vocabulary.
        const CONTROL: i64 = 3;
        const USER: i64 = 4;
        const UNUSED: i64 = 5;
        let mut vocab = HashMap::new();
        let mut added = vec![];
        for (id, (t, &k)) in tokens.iter().zip(&kinds).enumerate() {
            match k {
                UNUSED => {}
                CONTROL | USER => added.push((t.clone(), id as u32)),
                _ => {
                    vocab.insert(t.clone(), id as u32);
                }
            }
        }
        let mut ranks = HashMap::new();
        for (rank, m) in strings("tokenizer.ggml.merges")?.iter().enumerate() {
            let pair = split_joined(m, &vocab).map_err(|e| format!("merge {rank} {e}"))?;
            ranks.entry(pair).or_insert(rank);
        }
        added.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
        Ok(Tokenizer {
            // Not in the GGUF. Qwen's tokenizer.json sets it false, and
            // qwen35 is the only pre-tokenizer accepted above.
            ignore_merges: false,
            vocab,
            text: tokens,
            ranks,
            added,
        })
    }

    /// How many ids the vocabulary holds.
    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The id of an exact token string, for specials like `<|im_end|>`.
    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.added
            .iter()
            .find(|(t, _)| t == token)
            .map(|(_, id)| *id)
            .or_else(|| self.vocab.get(token).copied())
    }

    /// Text to ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let normalised = nfc(text);
        let mut out = vec![];
        let mut rest = normalised.as_str();
        // Added tokens are matched literally and never merged through.
        while !rest.is_empty() {
            let hit = self
                .added
                .iter()
                .filter_map(|(t, id)| rest.find(t.as_str()).map(|at| (at, t.len(), *id)))
                .min_by_key(|(at, len, _)| (*at, usize::MAX - len));
            match hit {
                Some((at, len, id)) => {
                    self.encode_ordinary(&rest[..at], &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_ordinary(rest, &mut out);
                    break;
                }
            }
        }
        out
    }

    fn encode_ordinary(&self, text: &str, out: &mut Vec<u32>) {
        let map = byte_to_char();
        for piece in split(text) {
            // Byte level: the piece's bytes become the piece's characters.
            let mut s = String::with_capacity(piece.len());
            for b in piece.bytes() {
                s.push(map[b as usize]);
            }
            if self.ignore_merges
                && let Some(&id) = self.vocab.get(&s)
            {
                out.push(id);
                continue;
            }
            self.merge(&s, out);
        }
    }

    /// BPE within one piece: repeatedly join the adjacent pair with the
    /// lowest rank, then look up what is left.
    fn merge(&self, piece: &str, out: &mut Vec<u32>) {
        let mut parts: Vec<String> = piece.chars().map(String::from).collect();
        if parts.is_empty() {
            return;
        }
        loop {
            let best = (0..parts.len().saturating_sub(1))
                .filter_map(|i| {
                    let key = (parts[i].clone(), parts[i + 1].clone());
                    self.ranks.get(&key).map(|r| (*r, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let joined = format!("{}{}", parts[i], parts[i + 1]);
            parts.splice(i..i + 2, [joined]);
        }
        for p in parts {
            match self.vocab.get(&p) {
                Some(&id) => out.push(id),
                // A piece with no id at all would be a hole in the
                // vocabulary; byte-level BPE is built so this cannot
                // happen, and saying so is better than a silent drop.
                None => eprintln!("tokenizer: no id for `{p}`"),
            }
        }
    }

    /// Ids back to text.
    ///
    /// Bytes are reassembled before UTF-8 is decoded, because a token can
    /// end in the middle of a character: a three-byte character split
    /// across two tokens is ordinary, and decoding them separately would
    /// produce two replacement characters instead of one letter.
    pub fn decode(&self, ids: &[u32]) -> String {
        let map = byte_to_char();
        let mut back = HashMap::new();
        for (b, c) in map.iter().enumerate() {
            back.insert(*c, b as u8);
        }
        let mut bytes = vec![];
        for &id in ids {
            let Some(t) = self.text.get(id as usize) else {
                continue;
            };
            for c in t.chars() {
                match back.get(&c) {
                    Some(&b) => bytes.push(b),
                    // An added token's content is literal text, not the
                    // byte-level alphabet.
                    None => {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Canonical combining class: 0 for everything that is not a mark.
fn ccc(c: char) -> u8 {
    CCC.binary_search_by_key(&(c as u32), |(k, _)| *k)
        .map_or(0, |i| CCC[i].1)
}

/// The composite of an adjacent pair, if they compose.
fn compose(a: char, b: char) -> Option<char> {
    // Hangul is arithmetic rather than tabular: the syllables are laid
    // out so that composing is a multiply and an add, which is why they
    // are not in the table and would add 11,172 entries if they were.
    let (s_base, l_base, v_base, t_base) = (0xAC00u32, 0x1100u32, 0x1161u32, 0x11A7u32);
    let (v_count, t_count) = (21u32, 28u32);
    let (a32, b32) = (a as u32, b as u32);
    if (l_base..l_base + 19).contains(&a32) && (v_base..v_base + v_count).contains(&b32) {
        let i = (a32 - l_base) * v_count + (b32 - v_base);
        return char::from_u32(s_base + i * t_count);
    }
    if (s_base..s_base + 11172).contains(&a32)
        && (a32 - s_base) % t_count == 0
        && (t_base + 1..t_base + t_count).contains(&b32)
    {
        return char::from_u32(a32 + (b32 - t_base));
    }
    COMP.binary_search_by_key(&(a32, b32), |(x, y, _)| (*x, *y))
        .ok()
        .and_then(|i| char::from_u32(COMP[i].2))
}

/// NFC, which `tokenizer.json` asks for in its `normalizer`.
///
/// Canonical ordering of marks, then composition. Decomposition is *not*
/// done: text that is already composed is returned unchanged, and text
/// that is decomposed is composed, which covers what arrives in practice.
/// What it does not cover is a singleton decomposition -- a character
/// whose NFC form is a *different* single character, such as U+212B
/// ANGSTROM SIGN becoming U+00C5 -- which would need the decomposition
/// table as well. Those stay as they are, and tokenize as themselves.
///
/// Skipping this entirely is not an option: the reference normalises, so
/// `e` followed by a combining acute is one token there and would be two
/// here. That is a model reading different text from the one it was
/// given, and no logit comparison would show it -- the logits would be
/// right for the tokens actually sent.
/// A merge stored as its two halves joined by a space.
///
/// A space is itself a legal token character, so splitting on the first
/// one is wrong for the pair `" " + " "`. The left side is the shortest
/// prefix ending in a space whose two sides are both tokens. Shared by
/// both readers, so that a JSON and a GGUF of the same tokenizer cannot
/// rank a merge differently.
fn split_joined(s: &str, vocab: &HashMap<String, u32>) -> Result<(String, String), String> {
    let split = (1..s.len())
        .filter(|i| s.is_char_boundary(*i) && s.as_bytes()[*i - 1] == b' ')
        .find(|i| vocab.contains_key(&s[..i - 1]) && vocab.contains_key(&s[*i..]))
        .ok_or_else(|| format!("`{s}` splits into no two tokens"))?;
    Ok((s[..split - 1].to_string(), s[split..].to_string()))
}

pub fn nfc(text: &str) -> String {
    let mut out: Vec<char> = Vec::with_capacity(text.len());
    for c in text.chars() {
        // Canonical ordering: a mark sorts back past any mark of a
        // higher class, and never past a starter.
        let k = ccc(c);
        let mut at = out.len();
        if k != 0 {
            while at > 0 {
                let prev = ccc(out[at - 1]);
                if prev == 0 || prev <= k {
                    break;
                }
                at -= 1;
            }
        }
        out.insert(at, c);
    }

    // Compose each starter with the marks that follow it, skipping any
    // that a closer mark of the same class blocks.
    let mut res: Vec<char> = Vec::with_capacity(out.len());
    let mut starter: Option<usize> = None;
    let mut last = 0u8;
    for c in out {
        let k = ccc(c);
        if let Some(s) = starter {
            let blocked = k != 0 && last >= k;
            if !blocked && let Some(j) = compose(res[s], c) {
                res[s] = j;
                continue;
            }
        }
        if k == 0 {
            starter = Some(res.len());
            last = 0;
        } else {
            last = k;
        }
        res.push(c);
    }
    res.into_iter().collect()
}

/// The pre-tokenizer's split, by hand.
///
/// The pattern in `tokenizer.json` is
///
/// ```text
/// (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}
///   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
/// ```
///
/// which is not a regular expression this crate can run: it has a
/// lookahead, and there is no regex engine here to give it to. It is
/// short enough to read directly, and reading it directly also avoids
/// paying for a dependency in the one place where a dependency would be
/// easy to justify and hard to remove.
fn split(text: &str) -> Vec<&str> {
    let b: Vec<char> = text.chars().collect();
    // Character offsets to byte offsets, so slices are cheap and exact.
    let mut at: Vec<usize> = Vec::with_capacity(b.len() + 1);
    let mut n = 0;
    for c in &b {
        at.push(n);
        n += c.len_utf8();
    }
    at.push(n);

    let letter = |c: char| c.is_alphabetic() || is_mark(c);
    let number = |c: char| c.is_numeric();
    let newline = |c: char| c == '\r' || c == '\n';

    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        let start = i;

        // 1. A contraction, case-insensitively.
        if b[i] == '\'' && i + 1 < b.len() {
            const TAILS: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
            let mut hit = 0;
            for t in TAILS {
                let k = t.chars().count();
                if i + k < b.len() + 1
                    && b[i + 1..(i + 1 + k).min(b.len())]
                        .iter()
                        .map(|c| c.to_ascii_lowercase())
                        .eq(t.chars())
                    && k > hit
                {
                    hit = k;
                }
            }
            if hit > 0 {
                i += 1 + hit;
                out.push(&text[at[start]..at[i]]);
                continue;
            }
        }

        // 2. An optional non-letter, non-digit, non-newline lead-in, then
        //    a run of letters. The lead-in is what puts the space on the
        //    front of " the".
        let lead = usize::from(!letter(b[i]) && !number(b[i]) && !newline(b[i]));
        if i + lead < b.len() && letter(b[i + lead]) {
            i += lead;
            while i < b.len() && letter(b[i]) {
                i += 1;
            }
            out.push(&text[at[start]..at[i]]);
            continue;
        }

        // 3. One digit. This vocabulary does not group them.
        if number(b[i]) {
            i += 1;
            out.push(&text[at[start]..at[i]]);
            continue;
        }

        // 4. An optional space, then a run of punctuation, then any
        //    newlines that follow it.
        let lead = usize::from(b[i] == ' ');
        if i + lead < b.len() && {
            let c = b[i + lead];
            !c.is_whitespace() && !letter(c) && !number(c)
        } {
            i += lead;
            while i < b.len() && !b[i].is_whitespace() && !letter(b[i]) && !number(b[i]) {
                i += 1;
            }
            while i < b.len() && newline(b[i]) {
                i += 1;
            }
            out.push(&text[at[start]..at[i]]);
            continue;
        }

        // 5. Whitespace running into newlines: the newlines go with it.
        if b[i].is_whitespace() {
            let mut j = i;
            while j < b.len() && b[j].is_whitespace() && !newline(b[j]) {
                j += 1;
            }
            if j < b.len() && newline(b[j]) {
                while j < b.len() && newline(b[j]) {
                    j += 1;
                }
                out.push(&text[at[i]..at[j]]);
                i = j;
                continue;
            }

            // 6. Whitespace with something after it keeps its last space
            //    for that something -- this is the `(?!\S)` branch, and
            //    it is why "a  b" is "a", " ", " b" and not "a", "  ", "b".
            let mut j = i;
            while j < b.len() && b[j].is_whitespace() {
                j += 1;
            }
            let end = if j < b.len() && j > i { j - 1 } else { j };
            if end > i {
                out.push(&text[at[i]..at[end]]);
                i = end;
                continue;
            }

            // 7. Whatever whitespace is left.
            i = j;
            out.push(&text[at[start]..at[i]]);
            continue;
        }

        // Nothing matched: take one character rather than loop forever.
        i += 1;
        out.push(&text[at[start]..at[i]]);
    }
    out
}

/// `\p{M}`: a combining mark.
///
/// `char` has no predicate for this and the full table is large, so this
/// covers the blocks that appear in text rather than every one that
/// exists. A mark outside them is treated as punctuation, which splits a
/// piece that should have stayed whole -- wrong, and wrong in a way the
/// fixture would show, so it is written down rather than assumed away.
fn is_mark(c: char) -> bool {
    matches!(c as u32,
        0x0300..=0x036F   // combining diacriticals
        | 0x0483..=0x0489 // cyrillic
        | 0x0591..=0x05BD | 0x05BF | 0x05C1..=0x05C2 | 0x05C4..=0x05C5 | 0x05C7
        | 0x0610..=0x061A | 0x064B..=0x065F | 0x0670 // arabic
        | 0x06D6..=0x06DC | 0x06DF..=0x06E4 | 0x06E7..=0x06E8 | 0x06EA..=0x06ED
        | 0x0900..=0x0903 | 0x093A..=0x094F | 0x0951..=0x0957 // devanagari
        | 0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E // thai
        | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF // more diacriticals
        | 0x20D0..=0x20F0 // combining for symbols
        | 0x3099..=0x309A // japanese voicing
        | 0xFE00..=0xFE0F // variation selectors
        | 0xFE20..=0xFE2F
        | 0xE0100..=0xE01EF)
}

impl Tokenizer {
    /// The tokenizer belonging to a model in the Ollama store.
    ///
    /// From the checkpoint rather than from a copy kept here: a tokenizer
    /// that drifted from the weights it is paired with would be worse
    /// than not having one, and harder to notice.
    pub fn for_model(model: &str) -> Result<Tokenizer, String> {
        // A GGUF carries its tokenizer in the header, so only the header
        // is read: the tensors are the runtime's business, and reading
        // them here too would load the model twice.
        if let Some(path) = crate::gguf::gguf_path(model)? {
            return Tokenizer::from_gguf(&crate::gguf::Gguf::open_header(&path)?);
        }
        let store = crate::qwen::Store::open(model)?;
        let raw = store.file("tokenizer.json")?;
        let text = String::from_utf8(raw).map_err(|e| format!("tokenizer.json: {e}"))?;
        Tokenizer::from_json(&text)
    }
}
