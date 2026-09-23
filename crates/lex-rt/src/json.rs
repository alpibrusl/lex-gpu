//! Just enough JSON to read an Ollama manifest and a safetensors header.
//!
//! Both are small, machine-written documents whose shape is known, and the
//! runtime otherwise has no dependencies. Numbers are `f64` (a safetensors
//! offset fits exactly up to 2^53, far past any tensor we load), strings
//! are unescaped, and anything malformed is an error rather than a panic.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(BTreeMap<String, Json>),
}

impl Json {
    pub fn parse(s: &str) -> Result<Json, String> {
        let b = s.as_bytes();
        let mut p = Parser { b, i: 0 };
        p.ws();
        let v = p.value()?;
        p.ws();
        if p.i != b.len() {
            return Err(format!("trailing input at byte {}", p.i));
        }
        Ok(v)
    }

    /// `obj[key]`, for an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(m) => m.get(key),
            _ => None,
        }
    }

    pub fn arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }

    pub fn str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn num(&self) -> Option<f64> {
        match self {
            Json::Num(n) => Some(*n),
            _ => None,
        }
    }

    /// A number as a `usize`, when it is one exactly.
    pub fn usize(&self) -> Option<usize> {
        let n = self.num()?;
        (n >= 0.0 && n.fract() == 0.0 && n <= 9.007_199_254_740_992e15).then_some(n as usize)
    }

    /// The keys of an object, in order.
    pub fn keys(&self) -> Vec<&str> {
        match self {
            Json::Obj(m) => m.keys().map(String::as_str).collect(),
            _ => vec![],
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Result<(), String> {
        if self.b.get(self.i) == Some(&c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("expected `{}` at byte {}", c as char, self.i))
        }
    }

    fn lit(&mut self, s: &str, v: Json) -> Result<Json, String> {
        if self.b[self.i..].starts_with(s.as_bytes()) {
            self.i += s.len();
            Ok(v)
        } else {
            Err(format!("bad literal at byte {}", self.i))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.b.get(self.i) {
            None => Err("unexpected end of input".into()),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.lit("true", Json::Bool(true)),
            Some(b'f') => self.lit("false", Json::Bool(false)),
            Some(b'n') => self.lit("null", Json::Null),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut m = BTreeMap::new();
        self.ws();
        if self.b.get(self.i) == Some(&b'}') {
            self.i += 1;
            return Ok(Json::Obj(m));
        }
        loop {
            self.ws();
            let k = self.string()?;
            self.ws();
            self.eat(b':')?;
            self.ws();
            m.insert(k, self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b'}') => {
                    self.i += 1;
                    return Ok(Json::Obj(m));
                }
                _ => return Err(format!("expected `,` or `}}` at byte {}", self.i)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut v = vec![];
        self.ws();
        if self.b.get(self.i) == Some(&b']') {
            self.i += 1;
            return Ok(Json::Arr(v));
        }
        loop {
            self.ws();
            v.push(self.value()?);
            self.ws();
            match self.b.get(self.i) {
                Some(b',') => self.i += 1,
                Some(b']') => {
                    self.i += 1;
                    return Ok(Json::Arr(v));
                }
                _ => return Err(format!("expected `,` or `]` at byte {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut s = String::new();
        loop {
            let c = *self.b.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => return Ok(s),
                b'\\' => {
                    let e = *self.b.get(self.i).ok_or("unterminated escape")?;
                    self.i += 1;
                    match e {
                        b'"' => s.push('"'),
                        b'\\' => s.push('\\'),
                        b'/' => s.push('/'),
                        b'b' => s.push('\u{8}'),
                        b'f' => s.push('\u{c}'),
                        b'n' => s.push('\n'),
                        b'r' => s.push('\r'),
                        b't' => s.push('\t'),
                        b'u' => {
                            let h = self
                                .b
                                .get(self.i..self.i + 4)
                                .ok_or("short \\u escape")?
                                .to_vec();
                            self.i += 4;
                            let n = u32::from_str_radix(
                                std::str::from_utf8(&h).map_err(|e| e.to_string())?,
                                16,
                            )
                            .map_err(|e| e.to_string())?;
                            s.push(char::from_u32(n).unwrap_or('\u{fffd}'));
                        }
                        _ => return Err(format!("bad escape `\\{}`", e as char)),
                    }
                }
                _ => {
                    // Copy the UTF-8 sequence this byte starts.
                    let extra = match c {
                        0x00..=0x7f => 0,
                        0xc0..=0xdf => 1,
                        0xe0..=0xef => 2,
                        _ => 3,
                    };
                    let bytes = self
                        .b
                        .get(self.i - 1..self.i + extra)
                        .ok_or("truncated UTF-8")?;
                    s.push_str(std::str::from_utf8(bytes).map_err(|e| e.to_string())?);
                    self.i += extra;
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        while self
            .b
            .get(self.i)
            .is_some_and(|c| matches!(c, b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'))
        {
            self.i += 1;
        }
        std::str::from_utf8(&self.b[start..self.i])
            .map_err(|e| e.to_string())?
            .parse()
            .map(Json::Num)
            .map_err(|_| format!("bad number at byte {start}"))
    }
}

#[cfg(test)]
mod tests {
    use super::Json;

    #[test]
    fn reads_the_shapes_a_manifest_and_a_safetensors_header_use() {
        let src = r#"{"layers":[{"name":"a.weight","digest":"sha256:ab","size":12},
                     {"mediaType":"x","digest":"sha256:cd","size":3.0}],"ok":true,"none":null}"#;
        let j = Json::parse(src).expect("parse");
        let layers = j.get("layers").and_then(Json::arr).expect("layers");
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].get("name").and_then(Json::str), Some("a.weight"));
        assert_eq!(layers[0].get("size").and_then(Json::usize), Some(12));
        assert_eq!(layers[1].get("name"), None);
        assert_eq!(j.get("ok"), Some(&Json::Bool(true)));
        assert_eq!(j.get("none"), Some(&Json::Null));

        let head = r#"{"__metadata__":{"quant_type":"nvfp4"},
            "w":{"dtype":"U32","shape":[17408,640],"data_offsets":[0,44564480]}}"#;
        let h = Json::parse(head).expect("parse");
        let w = h.get("w").expect("w");
        assert_eq!(w.get("dtype").and_then(Json::str), Some("U32"));
        let shape: Vec<usize> = w
            .get("shape")
            .and_then(Json::arr)
            .expect("shape")
            .iter()
            .filter_map(Json::usize)
            .collect();
        assert_eq!(shape, vec![17408, 640]);
        assert_eq!(h.keys(), vec!["__metadata__", "w"]);
    }

    #[test]
    fn rejects_malformed_input_instead_of_panicking() {
        for bad in [
            "{",
            "{\"a\" 1}",
            "[1,]",
            "\"unterminated",
            "{\"a\":}",
            "tru",
            "",
        ] {
            assert!(Json::parse(bad).is_err(), "{bad:?} should not parse");
        }
    }

    #[test]
    fn keeps_escapes_and_unicode() {
        let j = Json::parse(r#"{"k":"a\"b\\c\né ñ"}"#).expect("parse");
        assert_eq!(j.get("k").and_then(Json::str), Some("a\"b\\c\né ñ"));
    }
}
