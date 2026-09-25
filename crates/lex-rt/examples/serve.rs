//! An OpenAI-compatible endpoint, so an agent can use this.
//!
//!     cargo run --release -p lex-rt --example serve -- --model qwen3.8:27b-mlx
//!     curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
//!       -d '{"model":"lex","messages":[{"role":"user","content":"hello"}]}'
//!
//! Everything else here takes token ids and returns logits, which is what
//! a benchmark needs and not what a caller does. This is the thin part
//! that makes the rest reachable: tokenize, run, detokenize, speak the
//! protocol agents already speak.
//!
//! **One request at a time, deliberately.** There is one GPU and a 14.5 GB
//! model on it; concurrency here would mean either interleaving two
//! sequences through one KV cache or queueing, and queueing is what this
//! does. A second caller waits.
//!
//! No HTTP crate: this repository writes its own JSON reader, and HTTP/1.1
//! with a content-length body and an SSE stream is a couple of hundred
//! lines against the same standard library. The parsing is deliberately
//! strict and small rather than general -- it serves two routes.

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn main() -> Result<(), String> {
    serve::main()
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod serve {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::time::{SystemTime, UNIX_EPOCH};

    use lex_rt::json::Json;
    use lex_rt::qwen_run::Runner;
    use lex_rt::tokenizer::Tokenizer;

    /// What the model says to end a turn. `generation_config.json` lists
    /// both of these, and a run that ignores them does not stop.
    const STOP: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

    pub fn main() -> Result<(), String> {
        let mut model = "qwen3.8:27b-mlx".to_string();
        let (mut port, mut max_seq, mut depth) = (8080u16, 4096usize, 1usize);
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let mut val = || args.next().ok_or(format!("{a} needs a value"));
            match a.as_str() {
                "--model" => model = val()?,
                "--port" => port = val()?.parse().map_err(|_| "bad --port")?,
                "--max-seq" => max_seq = val()?.parse().map_err(|_| "bad --max-seq")?,
                "--depth" => depth = val()?.parse().map_err(|_| "bad --depth")?,
                other => return Err(format!("unknown argument `{other}`")),
            }
        }

        let tok = Tokenizer::for_model(&model)?;
        let mut rt = Runner::load(&model, max_seq)?;
        let stop: Vec<u32> = STOP.iter().filter_map(|s| tok.id_of(s)).collect();
        // Speculation is only a win where there is a draft head to do it.
        let depth = if rt.has_mtp() { depth } else { 0 };
        eprintln!(
            "{model} on {} — {} tokens of context, depth {depth}\n\
             listening on http://127.0.0.1:{port}",
            rt.device(),
            max_seq
        );

        let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { continue };
            if let Err(e) = handle(&mut conn, &mut rt, &tok, &stop, depth, &model, max_seq) {
                // A client that hangs up mid-stream is ordinary, not an
                // error worth stopping the server over.
                eprintln!("request: {e}");
            }
        }
        Ok(())
    }

    fn handle(
        conn: &mut TcpStream,
        rt: &mut Runner,
        tok: &Tokenizer,
        stop: &[u32],
        depth: usize,
        model: &str,
        max_seq: usize,
    ) -> Result<(), String> {
        let mut r = BufReader::new(conn.try_clone().map_err(|e| e.to_string())?);
        let mut line = String::new();
        r.read_line(&mut line).map_err(|e| e.to_string())?;
        let mut parts = line.split_whitespace();
        let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));

        let mut len = 0usize;
        loop {
            let mut h = String::new();
            if r.read_line(&mut h).map_err(|e| e.to_string())? == 0 || h.trim().is_empty() {
                break;
            }
            if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                len = v.trim().parse().unwrap_or(0);
            }
        }
        let mut body = vec![0u8; len];
        r.read_exact(&mut body).map_err(|e| e.to_string())?;
        let body = String::from_utf8_lossy(&body).into_owned();

        match (method, path) {
            ("GET", "/v1/models") => send(
                conn,
                200,
                "application/json",
                &format!(
                    r#"{{"object":"list","data":[{{"id":{},"object":"model","owned_by":"lex"}}]}}"#,
                    quote(model)
                ),
            ),
            ("POST", "/v1/chat/completions") => {
                chat(conn, rt, tok, stop, depth, model, max_seq, &body)
            }
            ("GET", "/health") => send(conn, 200, "text/plain", "ok\n"),
            _ => send(
                conn,
                404,
                "application/json",
                r#"{"error":{"message":"try POST /v1/chat/completions"}}"#,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn chat(
        conn: &mut TcpStream,
        rt: &mut Runner,
        tok: &Tokenizer,
        stop: &[u32],
        depth: usize,
        model: &str,
        max_seq: usize,
        body: &str,
    ) -> Result<(), String> {
        let j = match Json::parse(body) {
            Ok(j) => j,
            Err(e) => {
                return send(
                    conn,
                    400,
                    "application/json",
                    &format!(r#"{{"error":{{"message":{}}}}}"#, quote(&e)),
                );
            }
        };
        let stream = matches!(j.get("stream"), Some(Json::Bool(true)));
        let asked = j.get("max_tokens").and_then(Json::usize).unwrap_or(512);

        let prompt = match chat_ml(&j) {
            Ok(p) => p,
            Err(e) => {
                return send(
                    conn,
                    400,
                    "application/json",
                    &format!(r#"{{"error":{{"message":{}}}}}"#, quote(&e)),
                );
            }
        };
        let ids = tok.encode(&prompt);
        // `max_tokens` is a ceiling on the reply, not a reservation of
        // context. Clients routinely ask for the whole window and mean
        // "as much as fits" -- lex-llm's OpenAI adapter sends 8192 -- so
        // refusing that is refusing every such client. Only a prompt with
        // no room left after it is an error.
        if ids.len() >= max_seq {
            return send(
                conn,
                400,
                "application/json",
                &format!(
                    r#"{{"error":{{"message":"{} prompt tokens leaves no room in {max_seq}"}}}}"#,
                    ids.len()
                ),
            );
        }
        let want = asked.min(max_seq - ids.len());

        // Each request is its own conversation: the cache holds one, and
        // a caller that resends its history expects to be answered on it
        // rather than on what the last caller left behind.
        rt.reset();
        let logits = rt.prefill(&ids)?;
        let id = format!("chatcmpl-{}", now());

        if stream {
            let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                        cache-control: no-cache\r\nconnection: close\r\n\r\n";
            conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
            let first = format!(
                r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"role":"assistant"}},"finish_reason":null}}]}}"#,
                quote(&id),
                now(),
                quote(model)
            );
            sse(conn, &first)?;
            let (_, reason, _) = generate(rt, tok, stop, depth, want, logits, &mut |piece| {
                let chunk = format!(
                    r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{"content":{}}},"finish_reason":null}}]}}"#,
                    quote(&id),
                    now(),
                    quote(model),
                    quote(piece)
                );
                sse(conn, &chunk)
            })?;
            let last = format!(
                r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{{}},"finish_reason":{}}}]}}"#,
                quote(&id),
                now(),
                quote(model),
                quote(reason)
            );
            sse(conn, &last)?;
            conn.write_all(b"data: [DONE]\n\n").map_err(|e| e.to_string())?;
            return conn.flush().map_err(|e| e.to_string());
        }

        let (text, reason, n) = generate(rt, tok, stop, depth, want, logits, &mut |_| Ok(()))?;
        let payload = format!(
            r#"{{"id":{},"object":"chat.completion","created":{},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":{}}}],"usage":{{"prompt_tokens":{},"completion_tokens":{n},"total_tokens":{}}}}}"#,
            quote(&id),
            now(),
            quote(model),
            quote(&text),
            quote(reason),
            ids.len(),
            ids.len() + n
        );
        send(conn, 200, "application/json", &payload)
    }

    /// Decode greedily, handing each new piece of text to `emit`.
    ///
    /// Text comes out per token, and a token can end in the middle of a
    /// character -- a three-byte character split across two tokens is
    /// ordinary. So the pieces are cut from the decode of everything so
    /// far rather than from each token alone, which is the difference
    /// between an emoji and two replacement characters.
    fn generate(
        rt: &mut Runner,
        tok: &Tokenizer,
        stop: &[u32],
        depth: usize,
        want: usize,
        logits: Vec<f32>,
        emit: &mut dyn FnMut(&str) -> Result<(), String>,
    ) -> Result<(String, &'static str, usize), String> {
        let argmax = |v: &[f32]| {
            (0..v.len())
                .max_by(|&a, &b| v[a].total_cmp(&v[b]))
                .expect("logits") as u32
        };
        let mut out: Vec<u32> = vec![];
        let mut said = String::new();
        let mut next = argmax(&logits);
        let mut reason = "length";

        while out.len() < want {
            if stop.contains(&next) {
                reason = "stop";
                break;
            }
            let committed = if depth > 0 {
                let (c, after) = rt.speculate(next, depth)?;
                next = after;
                c
            } else {
                let l = rt.step(next)?;
                let c = vec![next];
                next = argmax(&l);
                c
            };
            for t in committed {
                if stop.contains(&t) {
                    reason = "stop";
                    break;
                }
                out.push(t);
            }
            let full = tok.decode(&out);
            if let Some(piece) = full.strip_prefix(said.as_str())
                && !piece.is_empty()
            {
                emit(piece)?;
            }
            said = full;
            if reason == "stop" {
                break;
            }
        }
        let n = out.len();
        Ok((said, reason, n))
    }

    /// The messages as ChatML, which is what this checkpoint was trained
    /// on and what its `tokenizer_config.json` template writes.
    ///
    /// The template in the checkpoint is Jinja, and evaluating Jinja is a
    /// long way outside what this is for. The shape it produces for a
    /// plain conversation is three lines of string building, so that is
    /// what happens here -- and a checkpoint whose template is *not* this
    /// shape would need reading rather than assuming.
    fn chat_ml(j: &Json) -> Result<String, String> {
        let msgs = j
            .get("messages")
            .and_then(Json::arr)
            .ok_or("no `messages` array")?;
        if msgs.is_empty() {
            return Err("`messages` is empty".into());
        }
        let mut s = String::new();
        for m in msgs {
            let role = m.get("role").and_then(Json::str).unwrap_or("user");
            let content = m.get("content").and_then(Json::str).unwrap_or("");
            s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
        }
        s.push_str("<|im_start|>assistant\n");
        Ok(s)
    }

    fn sse(conn: &mut TcpStream, data: &str) -> Result<(), String> {
        conn.write_all(format!("data: {data}\n\n").as_bytes())
            .and_then(|()| conn.flush())
            .map_err(|e| e.to_string())
    }

    fn send(conn: &mut TcpStream, code: u16, ty: &str, body: &str) -> Result<(), String> {
        let head = format!(
            "HTTP/1.1 {code} {}\r\ncontent-type: {ty}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            if code == 200 { "OK" } else { "Error" },
            body.len()
        );
        conn.write_all(head.as_bytes())
            .and_then(|()| conn.write_all(body.as_bytes()))
            .and_then(|()| conn.flush())
            .map_err(|e| e.to_string())
    }

    /// A JSON string, escaped. The model emits newlines, quotes and
    /// control characters, and every one of them would end the response
    /// early if written straight through.
    fn quote(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        out.push('"');
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    eprintln!("this example needs a Metal or CUDA device");
}
