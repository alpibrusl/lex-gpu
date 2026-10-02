//! The chat template, and the reply that comes back through it.
//!
//! An OpenAI request is messages and a tool list; a model reads one flat
//! string. Qwen3.8 defines that translation in a Jinja template inside its
//! own checkpoint, and this is that template written out in Rust, because
//! this repository does not carry a Jinja engine.
//!
//! Getting it wrong does not fail loudly. A tool block in the wrong shape
//! produces a model that answers from its own head and never calls
//! anything -- it will even say so -- so the agreement is pinned by golden
//! files rendered from the checkpoint's own template by
//! `scripts/chat_fixtures.py`, compared byte for byte in `tests/chat.rs`.
//!
//! Two things about this template are worth knowing before reading:
//!
//! - **Tool calls are not JSON.** Qwen3.8 wants
//!   `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`,
//!   not the `{"name":..,"arguments":..}` that earlier Qwens used.
//! - **The prompt opens `<think>` for the model**, so a reply *starts*
//!   inside its reasoning and [`parse_reply`] treats it that way.
//!
//! Other models of the same architecture bring their own template, and
//! "same architecture" says nothing about the prompt: MiMo-v2.6 runs on
//! Qwen3.5's layers and writes its tool calls as JSON. [`Template`] names
//! the templates written out here; a model whose template is none of them
//! is refused rather than handed a prompt it was never trained on.

use crate::gguf::{Gguf, Value, gguf_path};
use crate::json::{Json, Map};

/// A chat template this module can render.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Template {
    /// Qwen3.8's, from the MLX checkpoint's `tokenizer_config.json`.
    #[default]
    Qwen38,
    /// MiMo-v2.6's, embedded in its GGUF: the llama.cpp adapter of
    /// XiaomiMiMo's own, with JSON tool arguments.
    Mimo,
}

/// What MiMo's template does that the Rust rendering depends on. All of
/// it, not a name: a template that changed any of these renders
/// differently, and should be refused until it is checked.
const MIMO_MARKS: [&str; 4] = [
    "You are provided with the following tools:\\n\\n<tools>",
    "'<tool_call><function=' ~ tool_call.name ~ '>'",
    "tojson(ensure_ascii=False)",
    "'<|im_start|>assistant\\n<think>' ~ reasoning ~ '</think>' ~ content",
];

impl Template {
    /// The template an Ollama tag's model uses. An MLX store is Qwen3.8's
    /// layout and carries its template; a GGUF says which in its header.
    pub fn for_model(model: &str) -> Result<Template, String> {
        let Some(path) = gguf_path(model)? else {
            return Ok(Template::Qwen38);
        };
        let g = Gguf::open_header(&path)?;
        let Some(Value::Str(jinja)) = g.meta.get("tokenizer.chat_template") else {
            return Err(format!("{model}: the GGUF carries no chat template"));
        };
        Template::recognise(jinja).ok_or_else(|| {
            format!(
                "{model}: its chat template is not one this server renders \
                 (Qwen3.8, MiMo-v2.6); refusing rather than guessing the prompt format"
            )
        })
    }

    /// Which template a Jinja source is, if it is one written out here.
    pub fn recognise(jinja: &str) -> Option<Template> {
        MIMO_MARKS
            .iter()
            .all(|m| jinja.contains(m))
            .then_some(Template::Mimo)
    }

    /// `messages` and `tools` as one prompt string.
    pub fn render(self, req: &Json) -> Result<String, String> {
        match self {
            Template::Qwen38 => render_qwen(req),
            Template::Mimo => render_mimo(req),
        }
    }

    /// Split a reply into reasoning, visible text and tool calls. See
    /// [`parse_reply`].
    pub fn parse_reply(self, text: &str, types: &[Json]) -> Reply {
        match self {
            Template::Qwen38 => split_reply(text, &|b| parse_call(b, types)),
            Template::Mimo => split_reply(text, &|b| parse_json_call(b, types)),
        }
    }

    /// [`Template::parse_reply`] for a reply the prompt may already have
    /// closed the reasoning block for. With thinking off the prompt ends
    /// past `</think>`, so the reply holds no close tag, and read as one
    /// that is still thinking it is all reasoning -- its tool calls never
    /// looked for.
    pub fn parse_reply_after(self, text: &str, types: &[Json], thinking: bool) -> Reply {
        if thinking {
            self.parse_reply(text, types)
        } else {
            self.parse_reply(&format!("</think>{text}"), types)
        }
    }
}

/// Whether the request leaves thinking on: only an explicit
/// `chat_template_kwargs.enable_thinking: false` turns it off, as in both
/// templates (`enable_thinking is defined and enable_thinking is false`).
pub fn thinks(req: &Json) -> bool {
    req.get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        != Some(&Json::Bool(false))
}

const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, \
                     validate key assumptions, consider plausible alternatives, and prioritize \
                     correctness, consistency, and clarity in the final answer.";
const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving \
                   directly to the conclusion without unnecessary elaboration.";

const TOOLS_HEAD: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_TAIL: &str = "\n\nIf you choose to call a function ONLY reply in the following format \
with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n\
value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n\
that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must \
be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the \
function call, but NOT after\n- If there is no function call available, answer the question like \
normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// One call the model asked for, in the shape an OpenAI reply carries.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub name: String,
    /// A JSON object, as a string -- that is how the wire holds it.
    pub arguments: String,
}

/// A reply, split the way an OpenAI response wants it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reply {
    /// The `<think>` block. Belongs in `reasoning_content`, not `content`:
    /// clients that show `content` should not show the model's notes, and
    /// a tool call hiding behind them would never be found.
    pub reasoning: String,
    pub content: String,
    pub calls: Vec<Call>,
}

/// Qwen3.8's prompt for `messages` and `tools`.
pub fn render(req: &Json) -> Result<String, String> {
    Template::Qwen38.render(req)
}

fn render_qwen(req: &Json) -> Result<String, String> {
    let msgs = req
        .get("messages")
        .and_then(Json::arr)
        .ok_or("no `messages` array")?;
    if msgs.is_empty() {
        return Err("`messages` is empty".into());
    }
    let tools = req.get("tools").and_then(Json::arr).unwrap_or(&[]);
    // With thinking off the template sets no reasoning instruction and
    // does not look at `reasoning_effort` at all.
    let instr = match req.get("reasoning_effort").and_then(Json::str) {
        _ if !thinks(req) => "",
        None | Some("xhigh") => XHIGH,
        Some("low") => LOW,
        // The template accepts `medium` and then says nothing at all about
        // reasoning, which is how medium *is* less reasoning.
        Some("medium") => "",
        Some(other) => {
            return Err(format!(
                "unexpected reasoning effort {other}; supported are xhigh (default), medium, low"
            ));
        }
    };

    // A system message is only legal first, and it never prints in the loop
    // below: with tools it is appended to the tool block, without them it
    // shares a block with the reasoning instructions.
    let lead = if msgs[0].get("role").and_then(Json::str) == Some("system") {
        text_of(&msgs[0])
    } else {
        String::new()
    };
    for (i, m) in msgs.iter().enumerate() {
        if i > 0 && m.get("role").and_then(Json::str) == Some("system") {
            return Err("system message must be at the beginning".into());
        }
    }

    let mut s = String::new();
    if tools.is_empty() {
        if !lead.is_empty() {
            s.push_str("<|im_start|>system\n");
            if !instr.is_empty() {
                s.push_str(instr);
                s.push_str("\n\n");
            }
            s.push_str(&lead);
            s.push_str("<|im_end|>\n");
        } else if !instr.is_empty() {
            s.push_str(&format!("<|im_start|>system\n{instr}<|im_end|>\n"));
        }
    } else {
        s.push_str("<|im_start|>system\n");
        if !instr.is_empty() {
            s.push_str(instr);
            s.push_str("\n\n");
        }
        s.push_str(TOOLS_HEAD);
        for t in tools {
            s.push('\n');
            s.push_str(&tojson(t));
        }
        s.push_str("\n</tools>");
        s.push_str(TOOLS_TAIL);
        if !lead.is_empty() {
            s.push_str("\n\n");
            s.push_str(&lead);
        }
        s.push_str("<|im_end|>\n");
    }

    for (i, m) in msgs.iter().enumerate() {
        let role = m.get("role").and_then(Json::str).unwrap_or("user");
        let content = text_of(m);
        match role {
            "system" => {}
            "user" => s.push_str(&format!("<|im_start|>user\n{content}<|im_end|>\n")),
            "assistant" => {
                let reasoning = m
                    .get("reasoning_content")
                    .and_then(Json::str)
                    .unwrap_or("")
                    .trim();
                s.push_str(&format!(
                    "<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}"
                ));
                for (n, c) in m
                    .get("tool_calls")
                    .and_then(Json::arr)
                    .unwrap_or(&[])
                    .iter()
                    .enumerate()
                {
                    let f = c.get("function").unwrap_or(c);
                    let name = f.get("name").and_then(Json::str).unwrap_or("");
                    // Only a first call that follows prose needs the blank
                    // line; the template is specific about it.
                    s.push_str(match (n, content.is_empty()) {
                        (0, true) => "",
                        (0, false) => "\n\n",
                        _ => "\n",
                    });
                    s.push_str(&format!("<tool_call>\n<function={name}>\n"));
                    for (k, v) in arguments_of(f) {
                        let v = match &v {
                            Json::Str(x) => x.clone(),
                            other => tojson(other),
                        };
                        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
                    }
                    s.push_str("</function>\n</tool_call>");
                }
                s.push_str("<|im_end|>\n");
            }
            "tool" => {
                // Consecutive results share one user turn, which is how the
                // template keeps a parallel-call round from looking like
                // several turns of conversation.
                let prev = i
                    .checked_sub(1)
                    .map(|p| msgs[p].get("role").and_then(Json::str));
                if prev.is_some() && prev != Some(Some("tool")) {
                    s.push_str("<|im_start|>user");
                }
                s.push_str(&format!("\n<tool_response>\n{content}\n</tool_response>"));
                let next = msgs.get(i + 1).map(|n| n.get("role").and_then(Json::str));
                if next.is_none() || next != Some(Some("tool")) {
                    s.push_str("<|im_end|>\n");
                }
            }
            other => return Err(format!("unexpected message role `{other}`")),
        }
    }
    // The template opens the reasoning block itself rather than leaving the
    // model to remember the tag -- or, with thinking off, opens and closes
    // it empty.
    s.push_str(if thinks(req) {
        "<|im_start|>assistant\n<think>\n"
    } else {
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    });
    Ok(s)
}

/// `arguments`, which the wire sends as a JSON string and the template
/// wants as a mapping.
fn arguments_of(f: &Json) -> Vec<(String, Json)> {
    let parsed;
    let obj = match f.get("arguments") {
        Some(Json::Str(s)) if !s.is_empty() => {
            parsed = Json::parse(s).unwrap_or(Json::Null);
            &parsed
        }
        Some(j) => j,
        None => return vec![],
    };
    match obj {
        Json::Obj(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => vec![],
    }
}

fn text_of(m: &Json) -> String {
    m.get("content")
        .and_then(Json::str)
        .unwrap_or("")
        .trim()
        .to_string()
}

/// MiMo-v2.6's prompt, after the template in its GGUF. It differs from
/// Qwen3.8's in ways that each change the tokens:
///
/// - nothing follows `<|im_end|>` -- no newline between turns;
/// - the tool block is one short sentence and the schemas, with no
///   instructions on how to call, and it is its own system turn;
/// - a system message prints where it stands, like any other;
/// - a `tool` result is its own `tool` turn, not wrapped in a user turn;
/// - content is not trimmed;
/// - arguments that arrive as a string -- which is how the wire sends them
///   -- print as that string, exactly as the model wrote them;
/// - there is no reasoning-effort setting, only thinking on or off.
fn render_mimo(req: &Json) -> Result<String, String> {
    let msgs = req
        .get("messages")
        .and_then(Json::arr)
        .ok_or("no `messages` array")?;
    if msgs.is_empty() {
        return Err("`messages` is empty".into());
    }
    let tools = req.get("tools").and_then(Json::arr).unwrap_or(&[]);

    let mut s = String::new();
    if !tools.is_empty() {
        s.push_str("<|im_start|>system\n");
        mimo_tools(tools, &mut s);
        s.push_str("<|im_end|>");
    }
    for m in msgs {
        let role = m
            .get("role")
            .and_then(Json::str)
            .ok_or("a message has no `role`")?;
        let content = mimo_content(m.get("content"))?;
        if role == "assistant" {
            let reasoning = m.get("reasoning_content").and_then(Json::str).unwrap_or("");
            s.push_str("<|im_start|>assistant\n<think>");
            s.push_str(reasoning);
            s.push_str("</think>");
            s.push_str(&content);
            for c in m.get("tool_calls").and_then(Json::arr).unwrap_or(&[]) {
                let f = c.get("function").or_else(|| c.get("custom")).unwrap_or(c);
                s.push_str("<tool_call><function=");
                s.push_str(f.get("name").and_then(Json::str).unwrap_or(""));
                s.push('>');
                match (f.get("input"), f.get("arguments")) {
                    (Some(Json::Str(input)), _) => s.push_str(input),
                    (_, Some(Json::Str(args))) => s.push_str(args),
                    (_, Some(args)) => s.push_str(&tojson(args)),
                    (_, None) => {}
                }
                s.push_str("</function></tool_call>");
            }
            s.push_str("<|im_end|>");
        } else {
            s.push_str("<|im_start|>");
            s.push_str(role);
            s.push('\n');
            s.push_str(&content);
            let own = m.get("tools").and_then(Json::arr).unwrap_or(&[]);
            if !own.is_empty() {
                if !content.is_empty() {
                    s.push_str("\n\n");
                }
                mimo_tools(own, &mut s);
            }
            s.push_str("<|im_end|>");
        }
    }
    s.push_str("<|im_start|>assistant\n");
    s.push_str(if thinks(req) {
        "<think>\n"
    } else {
        "<think></think>"
    });
    Ok(s)
}

fn mimo_tools(tools: &[Json], s: &mut String) {
    s.push_str("You are provided with the following tools:\n\n<tools>");
    for t in tools {
        s.push('\n');
        s.push_str(&tojson(t));
    }
    s.push_str("\n</tools>");
}

/// A message's content: a string, or a list of parts of which only text
/// is something this server can read. The template would print a
/// placeholder for an image and the model would then look for the image
/// it stands for, so a picture is refused here rather than silently lost.
fn mimo_content(c: Option<&Json>) -> Result<String, String> {
    let parts = match c {
        Some(Json::Str(s)) => return Ok(s.clone()),
        Some(Json::Arr(parts)) => parts,
        Some(Json::Obj(_)) => return Err("message content is an object, not text".into()),
        // Absent, null, a number: the template prints nothing for these.
        _ => return Ok(String::new()),
    };
    let mut out = String::new();
    for p in parts {
        match p {
            Json::Str(s) => out.push_str(s),
            Json::Obj(_) => {
                let ty = p.get("type").and_then(Json::str).unwrap_or("");
                let media = [
                    "image",
                    "image_url",
                    "audio",
                    "audio_url",
                    "input_audio",
                    "video",
                    "video_url",
                ];
                if media.contains(&ty) || media.iter().any(|k| p.get(k).is_some()) {
                    return Err(format!(
                        "a `{ty}` content part: this server reads text only"
                    ));
                }
                if let Some(t) = p.get("text") {
                    match t {
                        Json::Str(t) => out.push_str(t),
                        other => out.push_str(&tojson(other)),
                    }
                }
            }
            other => {
                return Err(format!(
                    "a content part that is not text: {}",
                    tojson(other)
                ));
            }
        }
    }
    Ok(out)
}

/// `tojson` as Hugging Face's template environment defines it, which is
/// the one both models' prompts went through in training: `json.dumps`
/// with `ensure_ascii=False` -- keys in the order the client wrote them,
/// spaced separators, nothing escaped past what JSON itself requires.
///
/// Plain Jinja's filter is a different function: it sorts keys and escapes
/// `< > & '` and everything past ASCII, to be safe inside a web page.
/// Rendering Qwen3.8's tools that way cost 55 extra tokens on one tool
/// whose description said "don't" and "<pattern>" (390 against 335), all
/// of it escapes the model never saw in training.
pub fn tojson(j: &Json) -> String {
    let mut s = String::new();
    write_json(j, &mut s);
    s
}

fn write_json(j: &Json, out: &mut String) {
    match j {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => out.push_str(&number(*n)),
        Json::Str(s) => write_str(s, out),
        Json::Arr(a) => {
            out.push('[');
            for (i, v) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json(v, out);
            }
            out.push(']');
        }
        Json::Obj(m) => {
            out.push('{');
            for (i, (k, v)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_str(k, out);
                out.push_str(": ");
                write_json(v, out);
            }
            out.push('}');
        }
    }
}

/// Whole floats print as integers, because the reader turns every number
/// into `f64` and a schema that said `1` must not come back as `1.0`.
fn number(n: f64) -> String {
    if n.is_finite() && n.fract() == 0.0 && n.abs() < 1e16 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

fn write_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Split a reply into reasoning, visible text and tool calls.
///
/// The prompt already opened `<think>`, so the reply begins inside it and
/// the first `</think>` ends it. `types` is the tool list from the request:
/// the wire form of a call has typed arguments and the model writes plain
/// text, so the declared schema is the only thing that can say whether
/// `1` meant a number or a string.
pub fn parse_reply(text: &str, types: &[Json]) -> Reply {
    Template::Qwen38.parse_reply(text, types)
}

/// The split both templates share -- reasoning, then prose with
/// `<tool_call>` blocks in it -- with what is inside a block left to
/// `call`.
fn split_reply(text: &str, call: &dyn Fn(&str) -> Option<Call>) -> Reply {
    let (reasoning, rest) = match text.split_once("</think>") {
        Some((a, b)) => (a, b),
        // No close tag: the whole thing is still reasoning, which is what a
        // reply cut off by a token budget looks like.
        None => (text, ""),
    };
    let mut r = Reply {
        reasoning: reasoning.trim().to_string(),
        ..Reply::default()
    };
    let mut prose = String::new();
    let mut tail = rest;
    while let Some(open) = tail.find("<tool_call>") {
        prose.push_str(&tail[..open]);
        let after = &tail[open + "<tool_call>".len()..];
        let (body, next) = match after.split_once("</tool_call>") {
            Some((b, n)) => (b, n),
            // Truncated mid-call: take what there is rather than drop it.
            None => (after, ""),
        };
        if let Some(c) = call(body) {
            r.calls.push(c);
        }
        tail = next;
    }
    prose.push_str(tail);
    r.content = prose.trim().to_string();
    r
}

fn parse_call(body: &str, types: &[Json]) -> Option<Call> {
    let open = body.find("<function=")?;
    let after = &body[open + "<function=".len()..];
    let close = after.find('>')?;
    let name = after[..close].trim().to_string();
    let schema = properties_of(types, &name);
    let mut args = Map::new();
    let mut tail = &after[close + 1..];
    while let Some(p) = tail.find("<parameter=") {
        let a = &tail[p + "<parameter=".len()..];
        let Some(gt) = a.find('>') else { break };
        let key = a[..gt].trim().to_string();
        let rest = &a[gt + 1..];
        let (raw, next) = match rest.split_once("</parameter>") {
            Some((v, n)) => (v, n),
            None => (rest, ""),
        };
        args.insert(key.clone(), coerce(raw.trim(), schema.as_ref(), &key));
        tail = next;
    }
    Some(Call {
        name,
        arguments: tojson(&Json::Obj(args)),
    })
}

/// MiMo's call: `<function=name>{"k": v}</function>`, the arguments
/// already JSON and already typed by the model.
///
/// The arguments go back to the client as the model wrote them, not
/// re-serialised: the client returns them in the next request's history,
/// the template prints a string argument verbatim, and so the model reads
/// back exactly what it wrote. A body that is not a JSON object still
/// comes back as a call -- dropping it would end an agent's turn as if the
/// model had asked for nothing -- and the client's error on it is feedback
/// the model can act on. A body in Qwen's `<parameter=` form is read as
/// that, since the two families are close enough to trade habits.
fn parse_json_call(body: &str, types: &[Json]) -> Option<Call> {
    let open = body.find("<function=")?;
    let after = &body[open + "<function=".len()..];
    let close = after.find('>')?;
    let name = after[..close].trim().to_string();
    let inner = &after[close + 1..];
    let inner = inner.split_once("</function>").map_or(inner, |(a, _)| a);
    if inner.contains("<parameter=") {
        return parse_call(body, types);
    }
    let raw = inner.trim();
    let arguments = if raw.is_empty() {
        "{}".to_string()
    } else {
        raw.to_string()
    };
    Some(Call { name, arguments })
}

fn properties_of(types: &[Json], name: &str) -> Option<Json> {
    types.iter().find_map(|t| {
        let f = t.get("function").unwrap_or(t);
        (f.get("name").and_then(Json::str) == Some(name))
            .then(|| f.get("parameters")?.get("properties").cloned())
            .flatten()
    })
}

/// A parameter's text as the declared type, or as a string when nothing
/// says otherwise. Guessing from the text instead would turn a zip code
/// into a number.
fn coerce(raw: &str, props: Option<&Json>, key: &str) -> Json {
    let ty = props
        .and_then(|p| p.get(key))
        .and_then(|p| p.get("type"))
        .and_then(Json::str);
    match ty {
        Some("number") | Some("integer") => raw
            .parse::<f64>()
            .map_or_else(|_| Json::Str(raw.into()), Json::Num),
        Some("boolean") => match raw {
            "true" => Json::Bool(true),
            "false" => Json::Bool(false),
            _ => Json::Str(raw.into()),
        },
        Some("object") | Some("array") => Json::parse(raw).unwrap_or(Json::Str(raw.into())),
        _ => Json::Str(raw.into()),
    }
}

/// The tags [`parse_reply`] splits on.
const TAGS: [&str; 3] = ["</think>", "<tool_call>", "</tool_call>"];

/// What a streamed reply emits as it grows.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
}

/// Splits a reply into pieces while it is still being generated.
///
/// Both halves only ever grow -- prose before a tool call stays prose, and
/// the markup around the call never reaches the client -- so each step can
/// emit the difference and never has to retract.
#[derive(Default)]
pub struct Stream {
    template: Template,
    reasoning: usize,
    content: usize,
    /// The prompt closed the reasoning block (thinking off): the reply is
    /// content from its first token.
    closed: bool,
}

impl Stream {
    pub fn new(template: Template) -> Stream {
        Stream {
            template,
            ..Stream::default()
        }
    }

    /// A stream for a request, thinking or not ([`thinks`]).
    pub fn for_request(template: Template, req: &Json) -> Stream {
        Stream {
            template,
            closed: !thinks(req),
            ..Stream::default()
        }
    }

    /// New pieces, given everything generated so far.
    pub fn push(&mut self, all: &str) -> Vec<Piece> {
        self.advance(&all[..settled(all)])
    }

    /// The last pieces, plus the finished reply.
    pub fn finish(&mut self, all: &str, types: &[Json]) -> (Vec<Piece>, Reply) {
        let pieces = self.advance(all);
        (
            pieces,
            self.template.parse_reply_after(all, types, !self.closed),
        )
    }

    fn advance(&mut self, text: &str) -> Vec<Piece> {
        let r = self.template.parse_reply_after(text, &[], !self.closed);
        let mut out = vec![];
        if r.reasoning.len() > self.reasoning {
            out.push(Piece::Reasoning(r.reasoning[self.reasoning..].to_string()));
            self.reasoning = r.reasoning.len();
        }
        if r.content.len() > self.content {
            out.push(Piece::Content(r.content[self.content..].to_string()));
            self.content = r.content.len();
        }
        out
    }
}

/// How much of `all` can be split without the answer changing later.
///
/// Withholding a fixed number of characters is not enough: a cut that lands
/// *inside* `</think>` hides the tag, and the text before it gets streamed
/// as reasoning that a later cut would have called content. So withhold
/// exactly the trailing characters that could still grow into a tag, and
/// nothing else.
fn settled(all: &str) -> usize {
    let longest = TAGS.iter().map(|t| t.len()).max().unwrap_or(0);
    for k in (1..longest.min(all.len()) + 1).rev() {
        let at = all.len() - k;
        if !all.is_char_boundary(at) {
            continue;
        }
        let tail = &all[at..];
        if TAGS.iter().any(|t| t.len() > k && t.starts_with(tail)) {
            return at;
        }
    }
    all.len()
}

/// The messages that are never dropped: a leading system message, and the
/// first user message after it.
///
/// The system message carries the tools and the agent's goal. The first
/// user message carries the *task*, and pinning it is not a nicety --
/// without it a long lex-code run ends with the agent asking the user what
/// they would like built, having read the codebase and forgotten why.
fn pinned(msgs: &[Json]) -> Vec<usize> {
    let mut out = vec![];
    let mut i = 0;
    if msgs.first().and_then(|m| m.get("role")).and_then(Json::str) == Some("system") {
        out.push(0);
        i = 1;
    }
    if msgs.get(i).and_then(|m| m.get("role")).and_then(Json::str) == Some("user") {
        out.push(i);
    }
    out
}

/// Where each droppable turn starts, oldest first.
///
/// A turn is an assistant message with the tool results that answer it, or
/// a lone user message. Grouping matters: the template opens a turn for a
/// `tool` message only when the message before it is not one, so dropping
/// half a tool exchange leaves a `<tool_response>` with nothing to open it.
fn turns(msgs: &[Json]) -> Vec<usize> {
    let after = pinned(msgs).last().map_or(0, |&i| i + 1);
    (after..msgs.len())
        .filter(|&i| msgs[i].get("role").and_then(Json::str) != Some("tool"))
        .collect()
}

/// Render `req`, dropping the oldest turns until the prompt fits `budget`
/// tokens. Returns the prompt and how many messages were dropped.
///
/// Refusing an over-long conversation is what the OpenAI spec says and it
/// is useless here: an agent loop's transcript only grows, so the run dies
/// partway through with the work half done. Ollama drops history and keeps
/// going, which is why it finishes tasks this did not. So this drops too --
/// whole turns, oldest first, never the system message and never the last
/// turn, which is the one actually being answered -- and in steps rather
/// than a turn at a time, so that consecutive turns of one conversation
/// keep beginning the same way and the prefix cache keeps working.
///
/// This is Qwen3.8's; [`Template::render_within`] is any template's.
pub fn render_within(
    req: &Json,
    budget: usize,
    count: &mut dyn FnMut(&str) -> usize,
) -> Result<(String, usize), String> {
    Template::Qwen38.render_within(req, budget, count)
}

impl Template {
    /// Render `req`, trimmed to `budget` tokens. See [`render_within`].
    pub fn render_within(
        self,
        req: &Json,
        budget: usize,
        count: &mut dyn FnMut(&str) -> usize,
    ) -> Result<(String, usize), String> {
        trim(self, req, budget, count)
    }
}

fn trim(
    t: Template,
    req: &Json,
    budget: usize,
    count: &mut dyn FnMut(&str) -> usize,
) -> Result<(String, usize), String> {
    let full = t.render(req)?;
    let total = count(&full);
    if total <= budget {
        return Ok((full, 0));
    }
    let msgs = req
        .get("messages")
        .and_then(Json::arr)
        .ok_or("no messages")?;
    let starts = turns(msgs);
    let head: Vec<Json> = pinned(msgs).iter().map(|&i| msgs[i].clone()).collect();
    let keep_from = |d: usize| -> Result<String, String> {
        let mut kept = head.clone();
        kept.extend(msgs[starts[d]..].iter().cloned());
        render_msgs(t, req, &kept)
    };

    // Where to cut. Not at the fewest turns that fit: an agent's history
    // grows every turn, so that cut moves every turn, the kept history
    // starts somewhere new each time, and the prefix cache -- which only
    // helps a prompt that begins the way the last one did -- misses on
    // every turn of a long task, re-reading the whole window at prefill
    // speed.
    //
    // So the cut sits on a grid: at the first turn where what it removes
    // reaches a multiple of `step`, taking the smallest multiple that
    // fits. What a cut at a given turn removes does not change as turns
    // are appended after it -- turns render independently between
    // `<|im_start|>` markers -- so the cut only moves when the overflow
    // crosses the next multiple of `step`: once per `step` tokens of
    // growth, where cutting at the fewest turns that fit moves on every
    // turn. It depends on nothing but the request, so it is the same cut
    // whichever request asks.
    //
    // The price is up to `step` tokens of history dropped that would have
    // fit, a quarter of the budget at worst and an eighth on average,
    // against re-reading the window on every turn.
    let step = (budget / 4).max(1);
    let need = (total - budget).div_ceil(step) * step;
    let last = starts.len().saturating_sub(1);
    // The first turn whose cut removes at least `need`; what a cut removes
    // only grows with the turn, so it can be searched for.
    let (mut lo, mut hi) = (0usize, last);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if total.saturating_sub(count(&keep_from(mid)?)) >= need {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    let dropped = starts[lo] - head.len();
    let out = keep_from(lo)?;
    if count(&out) <= budget {
        return Ok((out, dropped));
    }

    // Dropping turns is not always enough: one tool result can be larger
    // than the whole window -- a coding agent reads files -- and the turn
    // that is too big is the one being answered, which is never dropped.
    // So cut the middle out of the largest message instead, keeping the
    // head (what the tool was asked) and the tail (usually the answer).
    let mut kept = head.clone();
    kept.extend(msgs[starts[lo]..].iter().cloned());
    loop {
        let rendered = render_msgs(t, req, &kept)?;
        if count(&rendered) <= budget {
            return Ok((rendered, dropped));
        }
        let biggest = kept
            .iter()
            .enumerate()
            .skip(head.len())
            .max_by_key(|(_, m)| m.get("content").and_then(Json::str).unwrap_or("").len())
            .map(|(i, _)| i);
        match biggest.filter(|&i| {
            kept[i]
                .get("content")
                .and_then(Json::str)
                .unwrap_or("")
                .len()
                > 256
        }) {
            Some(i) => kept[i] = elide(&kept[i]),
            None => {
                return Err(format!(
                    "the last turn is {} tokens with nothing left to elide, over the {budget} \
                     the window leaves",
                    count(&rendered)
                ));
            }
        }
    }
}

/// `req` with its messages replaced.
fn render_msgs(t: Template, req: &Json, msgs: &[Json]) -> Result<String, String> {
    let Json::Obj(mut o) = req.clone() else {
        return Err("request is not an object".into());
    };
    o.insert("messages".into(), Json::Arr(msgs.to_vec()));
    t.render(&Json::Obj(o))
}

/// Halve a message's content, cutting from the middle and saying so. Two
/// thirds of the budget goes to the head, because a tool result's first
/// lines say what it is.
fn elide(m: &Json) -> Json {
    let text = m.get("content").and_then(Json::str).unwrap_or("");
    let target = text.len() / 2;
    let (mut h, mut t) = (target * 2 / 3, target / 3);
    while h > 0 && !text.is_char_boundary(h) {
        h -= 1;
    }
    while t < text.len() && !text.is_char_boundary(text.len() - t) {
        t += 1;
    }
    let cut = text.len() - h - t;
    let shorter = format!(
        "{}\n... {cut} characters elided ...\n{}",
        &text[..h],
        &text[text.len() - t..]
    );
    let Json::Obj(mut o) = m.clone() else {
        return m.clone();
    };
    o.insert("content".into(), Json::Str(shorter));
    Json::Obj(o)
}
