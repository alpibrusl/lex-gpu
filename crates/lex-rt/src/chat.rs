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

use std::collections::BTreeMap;

use crate::json::Json;

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

/// `messages` and `tools` as one prompt string.
pub fn render(req: &Json) -> Result<String, String> {
    let msgs = req
        .get("messages")
        .and_then(Json::arr)
        .ok_or("no `messages` array")?;
    if msgs.is_empty() {
        return Err("`messages` is empty".into());
    }
    let tools = req.get("tools").and_then(Json::arr).unwrap_or(&[]);
    let instr = match req.get("reasoning_effort").and_then(Json::str) {
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
                let prev = i.checked_sub(1).map(|p| msgs[p].get("role").and_then(Json::str));
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
    // model to remember the tag.
    s.push_str("<|im_start|>assistant\n<think>\n");
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

/// Jinja's `tojson`, which is not `serde_json::to_string`: keys sort, the
/// separators carry a space, non-ASCII escapes, and `< > & '` escape on
/// top of that so the result is safe to drop into a page. All four are
/// reachable from an ordinary tool description, so all four are copied.
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
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            '\'' => out.push_str("\\u0027"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if (c as u32) < 0x7f => out.push(c),
            c => {
                // Astral characters escape as the surrogate pair, as
                // `json.dumps` writes them.
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
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
        if let Some(c) = parse_call(body, types) {
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
    let mut args: BTreeMap<String, Json> = BTreeMap::new();
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
    reasoning: usize,
    content: usize,
}

impl Stream {
    /// New pieces, given everything generated so far.
    pub fn push(&mut self, all: &str) -> Vec<Piece> {
        self.advance(&all[..settled(all)])
    }

    /// The last pieces, plus the finished reply.
    pub fn finish(&mut self, all: &str, types: &[Json]) -> (Vec<Piece>, Reply) {
        let pieces = self.advance(all);
        (pieces, parse_reply(all, types))
    }

    fn advance(&mut self, text: &str) -> Vec<Piece> {
        let r = parse_reply(text, &[]);
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
