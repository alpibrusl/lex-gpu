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

    use lex_rt::chat::{self, Piece, Stream, Template};
use lex_rt::json::Json;
use lex_rt::sample::Sampler;
    use lex_rt::qwen_run::{Checkpoint, Runner, evict_index};
    use lex_rt::tokenizer::Tokenizer;

    /// What the model says to end a turn. `generation_config.json` lists
    /// both of these, and a run that ignores them does not stop.
    const STOP: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

    pub fn main() -> Result<(), String> {
        let mut model = "qwen3.8:27b-mlx".to_string();
        // 4096 was a decode benchmark's window; an agent's transcript passes
        // it inside a few tool calls, and one of lex-code's tool results
        // measured 10497 tokens on its own -- at 8192 that is elided on
        // every turn. 16384 holds a big tool result and some history.
        //
        // Not larger, because prefill runs at 66-82 tok/s here and there is
        // no prefix cache: every turn re-reads the whole transcript, so a
        // 32k window costs about eight minutes a turn. Raise it with
        // --max-seq when context matters more than latency.
        // Depth 2, measured under sampling at temperature 1.0 on an M4:
        // 25.8 tok/s without speculation, 31.9 at depth 1, 34.9 at 2, 33.8
        // at 3. A sampled target accepts deep drafts less often than a
        // greedy one, so past 2 the drafting costs more than it saves. One
        // run each; the step from 1 to 2 is the robust part.
        let (mut port, mut max_seq, mut depth) = (8080u16, 16384usize, 2usize);
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
        // Before the weights: a model whose prompt format is unknown should
        // be refused in a second, not after a minute of loading.
        let template = Template::for_model(&model)?;
        let defaults = Defaults::for_model(&model)?;
        let mut rt = Runner::load(&model, max_seq)?;
        let stop: Vec<u32> = STOP.iter().filter_map(|s| tok.id_of(s)).collect();
        // Speculation is only a win where there is a draft head to do it.
        let depth = if rt.has_mtp() { depth } else { 0 };
        eprintln!(
            "{model} on {} — {} tokens of context, depth {depth}, {template:?} template\n\
             listening on http://127.0.0.1:{port}",
            rt.device(),
            max_seq
        );

        let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
        let mut cache = Cache::default();
        let ctx = Ctx {
            tok: &tok,
            stop: &stop,
            depth,
            model: &model,
            max_seq,
            template,
            defaults,
        };
        for conn in listener.incoming() {
            let Ok(mut conn) = conn else { continue };
            if let Err(e) = handle(&mut conn, &mut rt, &ctx, &mut cache) {
                // A client that hangs up mid-stream is ordinary, not an
                // error worth stopping the server over.
                eprintln!("request: {e}");
            }
        }
        Ok(())
    }

    /// What the runner already holds: the tokens it was last driven with,
    /// and resumable points inside them.
    ///
    /// An agent resends its whole history every turn, so most of each
    /// prompt has already been read once -- measured at 83-100% shared
    /// with the turn before, 92% overall. Re-reading it is the difference
    /// between a task taking minutes and taking an hour.
    ///
    /// Checkpoints sit at turn boundaries rather than at the end of the
    /// last prompt, because the end is exactly what changes: the harness
    /// re-renders the assistant turn it just received, so the shared
    /// prefix stops short of it. A single end-of-prompt checkpoint would
    /// miss every time.
    #[derive(Default)]
    struct Cache {
        tokens: Vec<u32>,
        points: Vec<Checkpoint>,
    }

    /// Roughly a gigabyte of them, at 151 MB each for this model.
    const KEEP: usize = 6;

    impl Cache {
        /// How many leading tokens of `ids` the runner has already read.
        fn shared(&self, ids: &[u32]) -> usize {
            ids.iter()
                .zip(&self.tokens)
                .take_while(|(a, b)| a == b)
                .count()
        }

        fn push(&mut self, c: Checkpoint) {
            self.points.push(c);
            while self.points.len() > KEEP {
                let at: Vec<usize> = self.points.iter().map(Checkpoint::pos).collect();
                match evict_index(&at) {
                    Some(i) => self.points.remove(i),
                    None => self.points.remove(0),
                };
            }
        }

        fn clear(&mut self) {
            self.points.clear();
            self.tokens.clear();
        }
    }

    /// What every request is served with.
    struct Ctx<'a> {
        tok: &'a Tokenizer,
        stop: &'a [u32],
        depth: usize,
        model: &'a str,
        max_seq: usize,
        template: Template,
        defaults: Defaults,
    }

    /// Sampling when the request does not say: the checkpoint's own
    /// recommendation. Greedy decoding is not a neutral default for a
    /// thinking model -- it repeats until the client gives up.
    #[derive(Clone, Copy)]
    struct Defaults {
        temperature: f32,
        top_p: f32,
        top_k: usize,
    }

    impl Defaults {
        /// Qwen3.8's `generation_config.json` says 1.0, 0.95, 20. A GGUF
        /// carries its model's own in `general.sampling.*` -- MiMo's say
        /// 0.6, 0.95, 20 -- and each key it has overrides.
        fn for_model(model: &str) -> Result<Defaults, String> {
            let mut d = Defaults {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
            };
            if let Some(path) = lex_rt::gguf::gguf_path(model)? {
                let g = lex_rt::gguf::Gguf::open_header(&path)?;
                let f = |k: &str| g.meta.get(k).and_then(|v| v.as_float());
                if let Some(t) = f("general.sampling.temp") {
                    d.temperature = t as f32;
                }
                if let Some(p) = f("general.sampling.top_p") {
                    d.top_p = p as f32;
                }
                if let Some(k) = g.meta.get("general.sampling.top_k").and_then(|v| v.as_int()) {
                    d.top_k = k.max(0) as usize;
                }
            }
            Ok(d)
        }
    }

    fn handle(
        conn: &mut TcpStream,
        rt: &mut Runner,
        ctx: &Ctx,
        cache: &mut Cache,
    ) -> Result<(), String> {
        let model = ctx.model;
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
            ("POST", "/v1/chat/completions") => chat(conn, rt, ctx, &body, cache),
            ("GET", "/health") => send(conn, 200, "text/plain", "ok\n"),
            _ => send(
                conn,
                404,
                "application/json",
                r#"{"error":{"message":"try POST /v1/chat/completions"}}"#,
            ),
        }
    }

    fn chat(
        conn: &mut TcpStream,
        rt: &mut Runner,
        ctx: &Ctx,
        body: &str,
        cache: &mut Cache,
    ) -> Result<(), String> {
        let Ctx {
            tok,
            stop,
            depth,
            model,
            max_seq,
            template,
            defaults,
        } = *ctx;
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
        let num = |k: &str| j.get(k).and_then(Json::num);
        let temperature = num("temperature").map_or(defaults.temperature, |t| t as f32);
        let top_p = num("top_p").map_or(defaults.top_p, |p| p as f32);
        let top_k = j.get("top_k").and_then(Json::usize).unwrap_or(defaults.top_k);
        let seed = j
            .get("seed")
            .and_then(Json::usize)
            .map_or_else(now, |s| s as u64);
        let mut sampler = Sampler::new(temperature, top_p, top_k, seed);

        // Keep room for a reply: a prompt that fills the window exactly can
        // generate nothing, which is a refusal by another name.
        let room = (max_seq / 4).min(1024);
        let (prompt, dropped) =
            match template.render_within(&j, max_seq - room, &mut |t| tok.encode(t).len()) {
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
        if dropped > 0 {
            eprintln!("dropped {dropped} of the oldest messages to fit {max_seq} tokens");
        }
        let ids = tok.encode(&prompt);
        // `max_tokens` is a ceiling on the reply, not a reservation of
        // context. Clients routinely ask for the whole window and mean
        // "as much as fits" -- lex-llm's OpenAI adapter sends 8192 -- so
        // refusing that is refusing every such client.
        let want = asked.min(max_seq - ids.len());

        // Resume as far into this prompt as the runner has already read.
        // A caller that resends its history is answered on the history it
        // sent -- the tokens are compared, so a different conversation
        // shares nothing and starts over.
        // The ablation: same binary, cache off, so the difference it makes
        // is measured rather than argued.
        if std::env::var_os("LEX_NO_PREFIX_CACHE").is_some() {
            cache.clear();
        }
        let shared = cache.shared(&ids);
        let at = cache.points.iter().rposition(|c| c.pos() <= shared);
        let start = match at {
            // Never resume at the very end: the prompt's last token has to
            // go through the model for its logits.
            Some(i) if cache.points[i].pos() > 0 && cache.points[i].pos() < ids.len() => {
                rt.resume(&cache.points[i]);
                cache.points.truncate(i + 1);
                cache.points[i].pos()
            }
            _ => {
                rt.reset();
                cache.clear();
                0
            }
        };
        eprintln!(
            "prefill {} tokens, {shared} already read, resuming at {start} ({:.0}% skipped)",
            ids.len(),
            100.0 * start as f64 / ids.len() as f64
        );
        let logits = prefill_cached(rt, tok, &ids, start, cache)?;
        cache.tokens.clone_from(&ids);
        let id = format!("chatcmpl-{}", now());

        // The declared tools, so a call's arguments can come back typed:
        // the model writes `17` and only the schema knows it meant a number.
        let types: Vec<Json> = j.get("tools").and_then(Json::arr).unwrap_or(&[]).to_vec();

        if stream {
            let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                        cache-control: no-cache\r\nconnection: close\r\n\r\n";
            conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
            let chunk = |delta: &str, finish: String| {
                format!(
                    r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[{{"index":0,"delta":{delta},"finish_reason":{finish}}}]}}"#,
                    quote(&id),
                    now(),
                    quote(model)
                )
            };
            sse(conn, &chunk(r#"{"role":"assistant"}"#, "null".into()))?;
            let mut split = Stream::new(template);
            let (text, reason, _) = generate(rt, tok, stop, depth, want, logits, &mut sampler, &mut |all| {
                for p in split.push(all) {
                    // Reasoning goes in its own field. A client that shows
                    // `content` should not be shown the model's notes, and
                    // an adapter looking for a tool call must not have to
                    // dig it out of them.
                    let d = match &p {
                        Piece::Reasoning(t) => format!(r#"{{"reasoning_content":{}}}"#, quote(t)),
                        Piece::Content(t) => format!(r#"{{"content":{}}}"#, quote(t)),
                    };
                    sse(conn, &chunk(&d, "null".into()))?;
                }
                Ok(())
            })?;
            let (last, reply) = split.finish(&text, &types);
            for p in last {
                let d = match &p {
                    Piece::Reasoning(t) => format!(r#"{{"reasoning_content":{}}}"#, quote(t)),
                    Piece::Content(t) => format!(r#"{{"content":{}}}"#, quote(t)),
                };
                sse(conn, &chunk(&d, "null".into()))?;
            }
            // A call is only useful whole, so it goes in one chunk rather
            // than as argument fragments.
            for (n, c) in reply.calls.iter().enumerate() {
                let d = format!(
                    r#"{{"tool_calls":[{{"index":{n},"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}]}}"#,
                    quote(&format!("call_{n}_{}", c.name)),
                    quote(&c.name),
                    quote(&c.arguments)
                );
                sse(conn, &chunk(&d, "null".into()))?;
            }
            let reason = finish_reason(reason, &reply);
            sse(conn, &chunk("{}", quote(reason)))?;
            conn.write_all(b"data: [DONE]\n\n").map_err(|e| e.to_string())?;
            return conn.flush().map_err(|e| e.to_string());
        }

        let (text, reason, n) = generate(rt, tok, stop, depth, want, logits, &mut sampler, &mut |_| Ok(()))?;
        let reply = template.parse_reply(&text, &types);
        let calls: Vec<String> = reply
            .calls
            .iter()
            .enumerate()
            .map(|(i, c)| {
                format!(
                    r#"{{"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}"#,
                    quote(&format!("call_{i}_{}", c.name)),
                    quote(&c.name),
                    quote(&c.arguments)
                )
            })
            .collect();
        let tool_calls = if calls.is_empty() {
            String::new()
        } else {
            format!(r#","tool_calls":[{}]"#, calls.join(","))
        };
        let payload = format!(
            r#"{{"id":{},"object":"chat.completion","created":{},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{},"reasoning_content":{}{tool_calls}}},"finish_reason":{}}}],"usage":{{"prompt_tokens":{},"completion_tokens":{n},"total_tokens":{}}}}}"#,
            quote(&id),
            now(),
            quote(model),
            quote(&reply.content),
            quote(&reply.reasoning),
            quote(finish_reason(reason, &reply)),
            ids.len(),
            ids.len() + n
        );
        send(conn, 200, "application/json", &payload)
    }

    /// Prefill `ids[start..]`, stopping at turn boundaries to keep a
    /// resumable point at each of the last few.
    ///
    /// Boundaries, not fixed strides, because that is where the next
    /// prompt will diverge: the harness re-renders the assistant turn it
    /// just received, so the shared prefix ends where that turn began.
    fn prefill_cached(
        rt: &mut Runner,
        tok: &Tokenizer,
        ids: &[u32],
        start: usize,
        cache: &mut Cache,
    ) -> Result<Vec<f32>, String> {
        let turn = tok.id_of("<|im_start|>");
        let bounds: Vec<usize> = match turn {
            Some(t) => (start + 1..ids.len()).filter(|&i| ids[i] == t).collect(),
            None => vec![],
        };
        let mut at = start;
        let mut logits = vec![];
        for &b in &bounds {
            if b <= at {
                continue;
            }
            logits = rt.prefill(&ids[at..b])?;
            at = b;
            cache.push(rt.checkpoint());
        }
        if at < ids.len() {
            logits = rt.prefill(&ids[at..])?;
        }
        Ok(logits)
    }

    #[allow(clippy::too_many_arguments)]
    /// Decode, handing everything said so far to `emit` as it grows.
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
        sampler: &mut Sampler,
        emit: &mut dyn FnMut(&str) -> Result<(), String>,  // everything said so far
    ) -> Result<(String, &'static str, usize), String> {
        let mut out: Vec<u32> = vec![];
        let mut said = String::new();
        let mut next = sampler.pick(&logits);
        let mut reason = "length";

        while out.len() < want {
            if stop.contains(&next) {
                reason = "stop";
                break;
            }
            let committed = if depth > 0 {
                let (c, after) = rt.speculate_with(next, depth, sampler)?;
                next = after;
                c
            } else {
                let l = rt.step(next)?;
                let c = vec![next];
                next = sampler.pick(&l);
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
            if full != said {
                emit(&full)?;
            }
            said = full;
            if reason == "stop" {
                break;
            }
        }
        let n = out.len();
        Ok((said, reason, n))
    }

    /// `tool_calls` whenever the model asked for one, whatever the
    /// decode loop stopped on: the agent loop dispatches on this field and
    /// nothing else, so a call reported as "stop" is a call never run.
    fn finish_reason(reason: &'static str, reply: &chat::Reply) -> &'static str {
        if reply.calls.is_empty() { reason } else { "tool_calls" }
    }

    fn sse(conn: &mut TcpStream, data: &str) -> Result<(), String> {
        conn.write_all(format!("data: {data}\n\n").as_bytes())
            .and_then(|()| conn.flush())
            .map_err(|e| e.to_string())
    }

    fn send(conn: &mut TcpStream, code: u16, ty: &str, body: &str) -> Result<(), String> {
        // A refusal the caller only sees as "HTTP 400" is a refusal nobody
        // can act on: the client library reports the status and drops the
        // body, so the reason has to reach this side's log too.
        if code != 200 {
            eprintln!("{code}: {body}");
        }
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
