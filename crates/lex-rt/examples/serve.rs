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
    use lex_rt::json::{Json, Map};
    use lex_rt::qwen_run::{Checkpoint, MAX_DEPTH, Runner, evict_index};
    use lex_rt::sample::Sampler;
    use lex_rt::spec::DepthController;
    use lex_rt::tokenizer::Tokenizer;
    use std::cell::RefCell;
    use std::time::Instant;

    /// What the model says to end a turn. `generation_config.json` lists
    /// both of these, and a run that ignores them does not stop.
    const STOP: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];

    pub fn main() -> Result<(), String> {
        let mut model = "qwen3.8:27b-mlx".to_string();
        // 4096 was a decode benchmark's window; an agent's transcript passes
        // it inside a few tool calls, and one of lex-code's tool results
        // measured 10497 tokens on its own. 16384 then held a big tool result
        // and some history, and was kept small because prefill ran at 66-82
        // tok/s with no prefix cache. Neither holds now: prefill is ~200
        // tok/s and the prefix cache re-reads only what is new (95% skipped
        // through a lex-code session, #32). At 16384 that session dropped
        // 93 messages on 32 turns and the agent looped rediscovering what it
        // had dropped, so 65536: about 4 GB of cache over the weights, and a
        // position costs nothing until it is used.
        // Depth 2, measured under sampling at temperature 1.0 on an M4:
        // 25.8 tok/s without speculation, 31.9 at depth 1, 34.9 at 2, 33.8
        // at 3. A sampled target accepts deep drafts less often than a
        // greedy one, so past 2 the drafting costs more than it saves. One
        // run each; the step from 1 to 2 is the robust part.
        // `auto` chooses each round from what drafting has been earning
        // (`lex_rt::spec`); a number fixes it, for measuring one depth.
        let (mut port, mut max_seq, mut depth) = (8080u16, 65536usize, "auto".to_string());
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let mut val = || args.next().ok_or(format!("{a} needs a value"));
            match a.as_str() {
                "--model" => model = val()?,
                "--port" => port = val()?.parse().map_err(|_| "bad --port")?,
                "--max-seq" => max_seq = val()?.parse().map_err(|_| "bad --max-seq")?,
                "--depth" => depth = val()?,
                other => return Err(format!("unknown argument `{other}`")),
            }
        }

        let tok = Tokenizer::for_model(&model)?;
        // Before the weights: a model whose prompt format is unknown should
        // be refused in a second, not after a minute of loading.
        let template = Template::for_model(&model)?;
        let defaults = Defaults::for_model(&model)?;
        let mut rt = Runner::load(&model, max_seq)?;
        // Every batch size now, not inside the first request that needs
        // one: on CUDA each is seconds of NVRTC.
        let t = std::time::Instant::now();
        rt.compile_batches()?;
        eprintln!(
            "batch kernels compiled in {:.1} s",
            t.elapsed().as_secs_f64()
        );
        let stop: Vec<u32> = STOP.iter().filter_map(|s| tok.id_of(s)).collect();
        // Speculation is only a win where there is a draft head to do it.
        let depth = if !rt.has_mtp() {
            Depth::Fixed(0)
        } else if depth == "auto" {
            Depth::Auto(RefCell::new(DepthController::new(MAX_DEPTH, 0.6)))
        } else {
            let d: usize = depth
                .parse()
                .map_err(|_| "--depth takes a number or `auto`")?;
            Depth::Fixed(d.min(MAX_DEPTH))
        };
        eprintln!(
            "{model} on {} — {} tokens of context, depth {}, {template:?} template\n\
             listening on http://127.0.0.1:{port}",
            rt.device(),
            max_seq,
            match &depth {
                Depth::Fixed(d) => d.to_string(),
                Depth::Auto(_) => format!("auto (1-{MAX_DEPTH})"),
            }
        );

        let listener = TcpListener::bind(("127.0.0.1", port)).map_err(|e| e.to_string())?;
        let mut cache = Cache::default();
        let ctx = Ctx {
            tok: &tok,
            stop: &stop,
            depth: &depth,
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

    /// How deep to draft: fixed, or chosen each round (`lex_rt::spec`).
    /// The controller lives for the server, so what it learned about the
    /// costs carries from one request to the next.
    enum Depth {
        Fixed(usize),
        Auto(RefCell<DepthController>),
    }

    impl Depth {
        fn pick(&self) -> usize {
            match self {
                Depth::Fixed(d) => *d,
                Depth::Auto(c) => c.borrow_mut().pick(),
            }
        }

        fn record(&self, depth: usize, kept: usize, ms: f64) {
            if let Depth::Auto(c) = self {
                c.borrow_mut().record(depth, kept, ms);
            }
        }
    }

    /// What every request is served with.
    struct Ctx<'a> {
        tok: &'a Tokenizer,
        stop: &'a [u32],
        depth: &'a Depth,
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
                if let Some(k) = g
                    .meta
                    .get("general.sampling.top_k")
                    .and_then(|v| v.as_int())
                {
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
            ("POST", "/v1/chat/completions") => match Json::parse(&body) {
                Ok(j) => chat(conn, rt, ctx, &j, cache, Api::OpenAi),
                Err(e) => bad_request(conn, &e),
            },
            // Ollama's own API, so a client written for Ollama -- lex-code's
            // `--ollama` path -- runs on lex by pointing OLLAMA_HOST here.
            ("POST", "/api/chat") => match Json::parse(&body) {
                Ok(o) => chat(conn, rt, ctx, &from_ollama(&o), cache, Api::Ollama),
                Err(e) => bad_request(conn, &e),
            },
            ("GET", "/api/tags") => send(
                conn,
                200,
                "application/json",
                &format!(r#"{{"models":[{{"name":{0},"model":{0}}}]}}"#, quote(model)),
            ),
            ("GET", "/health") => send(conn, 200, "text/plain", "ok\n"),
            _ => send(
                conn,
                404,
                "application/json",
                r#"{"error":{"message":"try POST /v1/chat/completions"}}"#,
            ),
        }
    }

    /// Which wire a request came in on, and so which shape the reply takes.
    #[derive(Clone, Copy, PartialEq)]
    enum Api {
        OpenAi,
        Ollama,
    }

    fn bad_request(conn: &mut TcpStream, e: &str) -> Result<(), String> {
        send(
            conn,
            400,
            "application/json",
            &format!(r#"{{"error":{{"message":{}}}}}"#, quote(e)),
        )
    }

    /// An Ollama `/api/chat` request as the OpenAI-shaped one the template
    /// reads. Ollama's defaults where it has them: a stream unless asked
    /// otherwise, and a reply bounded only by the context (`num_predict`
    /// absent or negative). `think: false` turns thinking off as the
    /// template's `enable_thinking` does; a level is a reasoning effort
    /// (`high` is the template's `xhigh`). An assistant message's
    /// `thinking` is its `reasoning_content`; tool-call arguments arrive as
    /// objects, which the template takes as they are.
    fn from_ollama(o: &Json) -> Json {
        let mut m = Map::new();
        let msgs: Vec<Json> = o
            .get("messages")
            .and_then(Json::arr)
            .unwrap_or(&[])
            .iter()
            .map(|msg| match (msg, msg.get("thinking")) {
                (Json::Obj(mm), Some(t)) => {
                    let mut mm = mm.clone();
                    mm.insert("reasoning_content".into(), t.clone());
                    Json::Obj(mm)
                }
                _ => msg.clone(),
            })
            .collect();
        m.insert("messages".into(), Json::Arr(msgs));
        if let Some(t) = o.get("tools") {
            m.insert("tools".into(), t.clone());
        }
        m.insert(
            "stream".into(),
            Json::Bool(!matches!(o.get("stream"), Some(Json::Bool(false)))),
        );
        let opt = |k: &str| o.get("options").and_then(|x| x.get(k)).cloned();
        let predict = opt("num_predict").and_then(|n| n.num()).unwrap_or(-1.0);
        let max = if predict > 0.0 { predict } else { 1e9 };
        m.insert("max_tokens".into(), Json::Num(max));
        for k in ["temperature", "top_p", "top_k", "seed"] {
            if let Some(v) = opt(k) {
                m.insert(k.into(), v);
            }
        }
        match o.get("think") {
            Some(Json::Bool(false)) => {
                let mut kw = Map::new();
                kw.insert("enable_thinking".into(), Json::Bool(false));
                m.insert("chat_template_kwargs".into(), Json::Obj(kw));
            }
            Some(Json::Str(level)) => {
                let effort = if level == "high" { "xhigh" } else { level };
                m.insert("reasoning_effort".into(), Json::Str(effort.into()));
            }
            _ => {}
        }
        Json::Obj(m)
    }

    /// A tool call's arguments as Ollama sends them: the object itself, not
    /// a string holding it.
    fn args_object(arguments: &str) -> String {
        match Json::parse(arguments) {
            Ok(Json::Obj(_)) => arguments.to_string(),
            _ => "{}".to_string(),
        }
    }

    fn chat(
        conn: &mut TcpStream,
        rt: &mut Runner,
        ctx: &Ctx,
        j: &Json,
        cache: &mut Cache,
        api: Api,
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
        let stream = matches!(j.get("stream"), Some(Json::Bool(true)));
        let asked = j.get("max_tokens").and_then(Json::usize).unwrap_or(512);
        let num = |k: &str| j.get(k).and_then(Json::num);
        let temperature = num("temperature").map_or(defaults.temperature, |t| t as f32);
        let top_p = num("top_p").map_or(defaults.top_p, |p| p as f32);
        let top_k = j
            .get("top_k")
            .and_then(Json::usize)
            .unwrap_or(defaults.top_k);
        let seed = j
            .get("seed")
            .and_then(Json::usize)
            .map_or_else(now, |s| s as u64);
        let mut sampler = Sampler::new(temperature, top_p, top_k, seed);

        // Keep room for a reply: a prompt that fills the window exactly can
        // generate nothing, which is a refusal by another name.
        let room = (max_seq / 4).min(1024);
        let (prompt, dropped) =
            match template.render_within(j, max_seq - room, &mut |t| tok.encode(t).len()) {
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
        // Said to the client as well as here: a client whose history was cut
        // has no other way to know (#32), and an agent that does not know
        // goes looking for what it already found.
        let dropped_header = if dropped > 0 {
            format!("x-lex-dropped-messages: {dropped}\r\n")
        } else {
            String::new()
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

        if stream && api == Api::Ollama {
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/x-ndjson\r\n\
                 cache-control: no-cache\r\n{dropped_header}connection: close\r\n\r\n"
            );
            conn.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
            // Reasoning in `thinking`, as Ollama streams a thinking model.
            let piece = |p: &Piece| {
                let message = match p {
                    Piece::Reasoning(t) => {
                        format!(r#""content":"","thinking":{}"#, quote(t))
                    }
                    Piece::Content(t) => format!(r#""content":{}"#, quote(t)),
                };
                format!(
                    r#"{{"model":{},"message":{{"role":"assistant",{message}}},"done":false}}"#,
                    quote(model)
                )
            };
            let mut split = Stream::for_request(template, j);
            let (text, reason, n) = generate(
                rt,
                tok,
                &ids,
                stop,
                depth,
                want,
                logits,
                &mut sampler,
                &mut |all| {
                    for p in split.push(all) {
                        ndjson(conn, &piece(&p))?;
                    }
                    Ok(())
                },
            )?;
            let (last, reply) = split.finish(&text, &types);
            for p in last {
                ndjson(conn, &piece(&p))?;
            }
            let calls: Vec<String> = reply
                .calls
                .iter()
                .map(|c| {
                    format!(
                        r#"{{"function":{{"name":{},"arguments":{}}}}}"#,
                        quote(&c.name),
                        args_object(&c.arguments)
                    )
                })
                .collect();
            ndjson(
                conn,
                &format!(
                    r#"{{"model":{},"message":{{"role":"assistant","content":""{}}},"done":true,"done_reason":{},"prompt_eval_count":{},"eval_count":{n}}}"#,
                    quote(model),
                    if calls.is_empty() {
                        String::new()
                    } else {
                        format!(r#","tool_calls":[{}]"#, calls.join(","))
                    },
                    quote(if reason == "length" { "length" } else { "stop" }),
                    ids.len()
                ),
            )?;
            return conn.flush().map_err(|e| e.to_string());
        }

        if stream {
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n\
                 cache-control: no-cache\r\n{dropped_header}connection: close\r\n\r\n"
            );
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
            let mut split = Stream::for_request(template, j);
            let (text, reason, n) = generate(
                rt,
                tok,
                &ids,
                stop,
                depth,
                want,
                logits,
                &mut sampler,
                &mut |all| {
                    for p in split.push(all) {
                        // Reasoning goes in its own field. A client that shows
                        // `content` should not be shown the model's notes, and
                        // an adapter looking for a tool call must not have to
                        // dig it out of them.
                        let d = match &p {
                            Piece::Reasoning(t) => {
                                format!(r#"{{"reasoning_content":{}}}"#, quote(t))
                            }
                            Piece::Content(t) => format!(r#"{{"content":{}}}"#, quote(t)),
                        };
                        sse(conn, &chunk(&d, "null".into()))?;
                    }
                    Ok(())
                },
            )?;
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
            // OpenAI's `stream_options.include_usage`: one last chunk with
            // no choices and the token counts, so a client timing a stream
            // knows how many tokens it received -- with speculation a chunk
            // carries several, and counting chunks undercounts.
            let usage = j.get("stream_options").and_then(|o| o.get("include_usage"))
                == Some(&Json::Bool(true));
            if usage {
                sse(
                    conn,
                    &format!(
                        r#"{{"id":{},"object":"chat.completion.chunk","created":{},"model":{},"choices":[],"usage":{{"prompt_tokens":{},"completion_tokens":{n},"total_tokens":{}}}}}"#,
                        quote(&id),
                        now(),
                        quote(model),
                        ids.len(),
                        ids.len() + n
                    ),
                )?;
            }
            conn.write_all(b"data: [DONE]\n\n")
                .map_err(|e| e.to_string())?;
            return conn.flush().map_err(|e| e.to_string());
        }

        let (text, reason, n) = generate(
            rt,
            tok,
            &ids,
            stop,
            depth,
            want,
            logits,
            &mut sampler,
            &mut |_| Ok(()),
        )?;
        let reply = template.parse_reply_after(&text, &types, lex_rt::chat::thinks(j));
        if api == Api::Ollama {
            let calls: Vec<String> = reply
                .calls
                .iter()
                .map(|c| {
                    format!(
                        r#"{{"function":{{"name":{},"arguments":{}}}}}"#,
                        quote(&c.name),
                        args_object(&c.arguments)
                    )
                })
                .collect();
            let payload = format!(
                r#"{{"model":{},"message":{{"role":"assistant","content":{},"thinking":{}{}}},"done":true,"done_reason":{},"prompt_eval_count":{},"eval_count":{n}}}"#,
                quote(model),
                quote(&reply.content),
                quote(&reply.reasoning),
                if calls.is_empty() {
                    String::new()
                } else {
                    format!(r#","tool_calls":[{}]"#, calls.join(","))
                },
                quote(if reason == "length" { "length" } else { "stop" }),
                ids.len()
            );
            return send_with(conn, 200, "application/json", &dropped_header, &payload);
        }
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
        send_with(conn, 200, "application/json", &dropped_header, &payload)
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
        let trace = std::env::var_os("LEX_PREFILL_TRACE").is_some();
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        let mut at = start;
        let mut logits = vec![];
        for &b in &bounds {
            if b <= at {
                continue;
            }
            let t = std::time::Instant::now();
            logits = rt.prefill(&ids[at..b])?;
            let took = ms(t);
            at = b;
            let t = std::time::Instant::now();
            cache.push(rt.checkpoint());
            if trace {
                eprintln!(
                    "segment {} tokens: {took:.0} ms, checkpoint {:.0} ms",
                    b - (at - (b - at)).min(b),
                    ms(t)
                );
            }
        }
        if at < ids.len() {
            let t = std::time::Instant::now();
            logits = rt.prefill(&ids[at..])?;
            if trace {
                eprintln!("last segment {} tokens: {:.0} ms", ids.len() - at, ms(t));
            }
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
        prompt: &[u32],
        stop: &[u32],
        depth: &Depth,
        want: usize,
        logits: Vec<f32>,
        sampler: &mut Sampler,
        emit: &mut dyn FnMut(&str) -> Result<(), String>, // everything said so far
    ) -> Result<(String, &'static str, usize), String> {
        let mut out: Vec<u32> = vec![];
        let mut said = String::new();
        let mut next = sampler.pick(&logits);
        let mut reason = "length";
        // Everything so far, for drafting from the context (lex-gpu#27):
        // the prompt, the reply, and while a round is chosen, `next`.
        let lookup_on = std::env::var("LEX_LOOKUP").map_or(true, |v| v != "0");
        let mut context = prompt.to_vec();

        while out.len() < want {
            if stop.contains(&next) {
                reason = "stop";
                break;
            }
            // A match of four or more tokens earlier in the context drafts
            // this round; otherwise the head does. Four, not two: on prose
            // two-token matches ("of the") are common and mostly wrong.
            let proposed = if lookup_on && !matches!(depth, Depth::Fixed(0)) {
                context.push(next);
                let p = lex_rt::spec::lookup(&context, 8, 4, MAX_DEPTH);
                context.pop();
                p
            } else {
                vec![]
            };
            let committed = if !proposed.is_empty() {
                let (c, after) = rt.speculate_proposed(next, &proposed, sampler)?;
                next = after;
                c
            } else if let d @ 1.. = depth.pick() {
                let t0 = Instant::now();
                let (c, after) = rt.speculate_with(next, d, sampler)?;
                // Wall time, the cost the client sees: drafting, the
                // verify, any undo and the sampler all count.
                depth.record(
                    d,
                    c.len().saturating_sub(1),
                    t0.elapsed().as_secs_f64() * 1e3,
                );
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
                context.push(t);
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
        log_request(prompt, &out);
        let n = out.len();
        Ok((said, reason, n))
    }

    /// `LEX_REQUEST_LOG=<file>` appends one JSON line a request -- the
    /// prompt's token ids and the reply's -- so real sessions can be
    /// replayed offline (`scripts/lookup_replay.py`) to measure what a
    /// drafting scheme would have got from them. Off by default: it is the
    /// whole conversation, in tokens.
    fn log_request(prompt: &[u32], reply: &[u32]) {
        let Some(path) = std::env::var_os("LEX_REQUEST_LOG") else {
            return;
        };
        let ids = |v: &[u32]| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
        let line = format!(
            "{{\"prompt\":[{}],\"completion\":[{}]}}\n",
            ids(prompt),
            ids(reply)
        );
        let wrote = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| f.write_all(line.as_bytes()));
        if let Err(e) = wrote {
            eprintln!("request log {}: {e}", path.to_string_lossy());
        }
    }

    /// `tool_calls` whenever the model asked for one, whatever the
    /// decode loop stopped on: the agent loop dispatches on this field and
    /// nothing else, so a call reported as "stop" is a call never run.
    fn finish_reason(reason: &'static str, reply: &chat::Reply) -> &'static str {
        if reply.calls.is_empty() {
            reason
        } else {
            "tool_calls"
        }
    }

    fn ndjson(conn: &mut TcpStream, line: &str) -> Result<(), String> {
        conn.write_all(line.as_bytes())
            .and_then(|()| conn.write_all(b"\n"))
            .and_then(|()| conn.flush())
            .map_err(|e| e.to_string())
    }

    fn sse(conn: &mut TcpStream, data: &str) -> Result<(), String> {
        conn.write_all(format!("data: {data}\n\n").as_bytes())
            .and_then(|()| conn.flush())
            .map_err(|e| e.to_string())
    }

    fn send(conn: &mut TcpStream, code: u16, ty: &str, body: &str) -> Result<(), String> {
        send_with(conn, code, ty, "", body)
    }

    /// [`send`] with extra header lines, each ending in `\r\n`.
    fn send_with(
        conn: &mut TcpStream,
        code: u16,
        ty: &str,
        extra: &str,
        body: &str,
    ) -> Result<(), String> {
        // A refusal the caller only sees as "HTTP 400" is a refusal nobody
        // can act on: the client library reports the status and drops the
        // body, so the reason has to reach this side's log too.
        if code != 200 {
            eprintln!("{code}: {body}");
        }
        let head = format!(
            "HTTP/1.1 {code} {}\r\ncontent-type: {ty}\r\ncontent-length: {}\r\n{extra}connection: close\r\n\r\n",
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
