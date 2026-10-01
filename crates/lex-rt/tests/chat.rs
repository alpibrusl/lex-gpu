//! The Rust chat template against the model's own Jinja one.
//!
//! `serve.rs` builds prompts in Rust because this repository carries no
//! Jinja engine, which is only safe if the two agree. Every `.txt` here was
//! rendered by the checkpoint's own `chat_template`
//! (`scripts/chat_fixtures.py`) and is compared byte for byte -- a tool
//! block in the wrong shape does not fail loudly, it just produces a model
//! that never calls anything.

use std::fs;
use std::path::PathBuf;

use lex_rt::chat::{Call, Template, parse_reply, render, render_within, tojson};
use lex_rt::json::{Json, Map};

fn data(sub: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(sub)
}

/// Every fixture in `sub`, rendered by `t` and compared byte for byte.
fn matches_its_fixtures(t: Template, sub: &str, needed: &[&str]) {
    let dir = data(sub);
    let mut cases: Vec<String> = fs::read_dir(&dir)
        .expect("fixture directory")
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension()? == "json").then(|| p.file_stem()?.to_str().map(str::to_string))?
        })
        .collect();
    cases.sort();
    // A fixture directory that quietly emptied would otherwise pass.
    assert!(
        cases.len() >= 10,
        "only {} fixtures in {sub}; run scripts/chat_fixtures.py",
        cases.len()
    );

    for name in &cases {
        let req = fs::read_to_string(dir.join(format!("{name}.json"))).expect("request");
        let want = fs::read_to_string(dir.join(format!("{name}.txt"))).expect("expected");
        let got = t
            .render(&Json::parse(&req).expect("request json"))
            .unwrap_or_else(|e| panic!("{sub}/{name}: {e}"));
        if got != want {
            let at = got
                .char_indices()
                .zip(want.chars())
                .find(|((_, a), b)| a != b)
                .map_or(got.len().min(want.len()), |((i, _), _)| i);
            let near = |s: &str| {
                let mut a = at.saturating_sub(40);
                while !s.is_char_boundary(a) {
                    a -= 1;
                }
                let mut b = (at + 60).min(s.len());
                while !s.is_char_boundary(b) {
                    b += 1;
                }
                s[a..b].to_string()
            };
            panic!(
                "{sub}/{name}: prompts differ at byte {at}\n  ours: {:?}\n  jinja: {:?}",
                near(&got),
                near(&want),
            );
        }
    }
    // The cases with no second opinion in the wild, so make sure they are
    // actually among what just passed.
    for n in needed {
        assert!(cases.iter().any(|c| c == n), "{sub}: missing case {n}");
    }
}

#[test]
fn the_rust_template_matches_the_models_own() {
    matches_its_fixtures(
        Template::Qwen38,
        "chatml",
        &[
            "tools",
            "tool_result",
            "two_tool_results",
            "odd_chars",
            "no_think",
            "no_think_tools",
        ],
    );
}

/// MiMo's fixtures come from transformers' Jinja environment, which is
/// what its template was written for (`tojson(ensure_ascii=False)`,
/// `{% generation %}`).
#[test]
fn the_mimo_template_matches_its_own() {
    matches_its_fixtures(
        Template::Mimo,
        "chatml_mimo",
        &[
            "tools",
            "tool_result",
            "odd_chars",
            "object_args",
            "no_think",
            "text_parts",
        ],
    );
}

/// `tojson` is Hugging Face's -- `json.dumps(ensure_ascii=False)` -- and
/// not plain Jinja's: the client's key order, and every character as
/// itself. Plain Jinja would sort these keys and write `\u0027z\u0027`.
#[test]
fn tojson_is_the_one_the_models_were_trained_through() {
    let mut m = Map::new();
    m.insert("b".to_string(), Json::Num(1.0));
    m.insert("a".to_string(), Json::Str("x < y & 'z' — naïve\n".into()));
    m.insert(
        "c".to_string(),
        Json::Arr(vec![Json::Bool(true), Json::Null]),
    );
    assert_eq!(
        tojson(&Json::Obj(m)),
        "{\"b\": 1, \"a\": \"x < y & 'z' — naïve\\n\", \"c\": [true, null]}"
    );
}

fn tool_types() -> Vec<Json> {
    let src = r#"[{"type":"function","function":{"name":"add","parameters":{"type":"object",
        "properties":{"a":{"type":"integer"},"b":{"type":"integer"},"note":{"type":"string"}}}}}]"#;
    Json::parse(src).unwrap().arr().unwrap().to_vec()
}

#[test]
fn a_reply_splits_into_reasoning_text_and_calls() {
    // The prompt opened `<think>`, so the reply starts inside it.
    let r = parse_reply("weighing it up\n</think>\n\nHere you go.", &[]);
    assert_eq!(r.reasoning, "weighing it up");
    assert_eq!(r.content, "Here you go.");
    assert!(r.calls.is_empty());

    let r = parse_reply(
        "thinking\n</think>\n\nLet me add them.\n\n<tool_call>\n<function=add>\n\
         <parameter=a>\n17\n</parameter>\n<parameter=b>\n25\n</parameter>\n\
         </function>\n</tool_call>",
        &tool_types(),
    );
    assert_eq!(r.reasoning, "thinking");
    assert_eq!(r.content, "Let me add them.");
    assert_eq!(
        r.calls,
        vec![Call {
            name: "add".into(),
            // Declared integers, so not "17": lex-schema would reject that.
            arguments: "{\"a\": 17, \"b\": 25}".into(),
        }]
    );
}

#[test]
fn two_calls_in_one_turn_stay_separate() {
    let r = parse_reply(
        "\n</think>\n\n<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n\
         </function>\n</tool_call>\n<tool_call>\n<function=add>\n<parameter=a>\n2\n</parameter>\n\
         </function>\n</tool_call>",
        &tool_types(),
    );
    assert_eq!(r.calls.len(), 2);
    assert_eq!(r.calls[0].arguments, "{\"a\": 1}");
    assert_eq!(r.calls[1].arguments, "{\"a\": 2}");
    assert_eq!(r.content, "", "the call markup must not leak into content");
}

/// Without the declared schema there is nothing to say `17` was a number,
/// and guessing would turn a zip code into one.
#[test]
fn an_undeclared_parameter_stays_a_string() {
    let r = parse_reply(
        "\n</think>\n<tool_call>\n<function=add>\n<parameter=note>\n01234\n</parameter>\n\
         </function>\n</tool_call>",
        &tool_types(),
    );
    assert_eq!(r.calls[0].arguments, "{\"note\": \"01234\"}");
}

/// A reply cut off by the token budget still has to come back as something.
#[test]
fn a_truncated_reply_is_all_reasoning() {
    let r = parse_reply("still thinking about", &[]);
    assert_eq!(r.reasoning, "still thinking about");
    assert_eq!(r.content, "");
    assert!(r.calls.is_empty());
}

#[test]
fn a_bad_reasoning_effort_is_refused() {
    let req = Json::parse(
        r#"{"messages":[{"role":"user","content":"hi"}],
        "reasoning_effort":"high"}"#,
    )
    .unwrap();
    assert!(
        render(&req).is_err(),
        "`high` is not one the template takes"
    );
}

#[test]
fn a_system_message_must_come_first() {
    let req = Json::parse(
        r#"{"messages":[{"role":"user","content":"hi"},{"role":"system","content":"no"}]}"#,
    )
    .unwrap();
    assert!(render(&req).is_err());
}

/// Streaming must never hand the client markup it would have to unsay: a
/// half-arrived `</tool_call>` looks exactly like prose until its last
/// character.
#[test]
fn a_streamed_reply_never_leaks_call_markup() {
    use lex_rt::chat::{Piece, Stream};

    let full = "mulling\n</think>\n\nOn it.\n\n<tool_call>\n<function=add>\n\
                <parameter=a>\n7\n</parameter>\n</function>\n</tool_call>";
    let mut s = Stream::default();
    let mut pieces = vec![];
    // One character at a time: the worst case for a partial tag.
    for i in 1..full.len() {
        if full.is_char_boundary(i) {
            pieces.extend(s.push(&full[..i]));
        }
    }
    let (last, reply) = s.finish(full, &tool_types());
    pieces.extend(last);

    let reasoning: String = pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Reasoning(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    let content: String = pieces
        .iter()
        .filter_map(|p| match p {
            Piece::Content(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "mulling");
    assert_eq!(content, "On it.");
    for bad in ["<tool_call", "</tool", "<function", "<parameter", "</think"] {
        assert!(
            !content.contains(bad),
            "{bad:?} leaked into streamed content"
        );
    }
    assert_eq!(reply.calls.len(), 1);
    assert_eq!(reply.calls[0].arguments, "{\"a\": 7}");
    // What was streamed must be what the finished reply says it was.
    assert_eq!(content, reply.content);
    assert_eq!(reasoning, reply.reasoning);
}

fn long_conversation(turns: usize, chars: usize) -> Json {
    let mut m = vec![
        r#"{"role":"system","content":"SYSTEM MARKER"}"#.to_string(),
        r#"{"role":"user","content":"FIRST USER"}"#.to_string(),
    ];
    for i in 0..turns {
        m.push(format!(
            r#"{{"role":"assistant","content":"","tool_calls":[{{"id":"c{i}","type":"function",
               "function":{{"name":"grep","arguments":"{{\"pattern\":\"p{i}\"}}"}}}}]}}"#
        ));
        m.push(format!(
            r#"{{"role":"tool","content":"TURNMARK{i} {}"}}"#,
            "out ".repeat(chars / 4)
        ));
    }
    m.push(r#"{"role":"user","content":"LAST USER"}"#.to_string());
    Json::parse(&format!(r#"{{"messages":[{}]}}"#, m.join(","))).expect("json")
}

/// Every `<tool_response>` has to sit inside a turn someone opened. Drop
/// an assistant message but keep the tool results that answered it and the
/// prompt is malformed in a way nothing downstream reports.
fn tool_responses_are_inside_a_user_turn(prompt: &str) -> bool {
    prompt
        .split("<|im_start|>")
        .skip(1)
        .all(|block| !block.contains("<tool_response>") || block.starts_with("user\n"))
}

#[test]
fn a_conversation_that_fits_is_left_alone() {
    let req = long_conversation(2, 40);
    let mut count = |s: &str| s.len() / 4;
    let (prompt, dropped) = render_within(&req, 1_000_000, &mut count).expect("render");
    assert_eq!(dropped, 0);
    assert_eq!(prompt, render(&req).unwrap());
}

#[test]
fn an_overlong_conversation_drops_whole_turns_from_the_front() {
    let req = long_conversation(40, 400);
    // A crude token count: the point is the trimming, not the tokenizer.
    let mut count = |s: &str| s.len() / 4;
    let budget = 4000;
    let (prompt, dropped) = render_within(&req, budget, &mut count).expect("render");

    assert!(
        dropped > 0,
        "nothing was dropped, so nothing is being tested"
    );
    assert!(count(&prompt) <= budget, "still over budget after trimming");
    // The two that must survive: the tools and goal live in the system
    // block, and the last turn is the one being answered.
    assert!(
        prompt.contains("SYSTEM MARKER"),
        "system message was dropped"
    );
    assert!(
        prompt.contains("LAST USER"),
        "the turn being answered was dropped"
    );
    // The task itself is pinned. Dropping it is what made a lex-code run
    // end with the agent asking the user what they would like built.
    assert!(prompt.contains("FIRST USER"), "the task was dropped");
    assert!(
        !prompt.contains("TURNMARK0 "),
        "the oldest middle turn survived"
    );
    assert!(
        tool_responses_are_inside_a_user_turn(&prompt),
        "a tool response was orphaned by the trim:\n{}",
        prompt
    );
    // It must drop no more than it has to.
    let kept_tools = prompt.matches("<tool_response>").count();
    assert!(kept_tools > 0, "trimmed all the way past every tool result");
}

/// A coding agent reads files, so one tool result can be bigger than the
/// whole window. Dropping turns cannot help -- the turn that is too big is
/// the one being answered -- and refusing ends the run. This is the case
/// that actually killed a lex-code eval: "the last turn alone is 10497
/// tokens, over the 7168 the window leaves".
#[test]
fn one_oversized_tool_result_is_elided_not_refused() {
    let huge = "x".repeat(60_000);
    let req = Json::parse(&format!(
        r#"{{"messages":[{{"role":"system","content":"SYSTEM MARKER"}},
           {{"role":"user","content":"read the file"}},
           {{"role":"assistant","content":"","tool_calls":[{{"id":"c","type":"function",
             "function":{{"name":"read","arguments":"{{}}"}}}}]}},
           {{"role":"tool","content":"HEAD-OF-OUTPUT{huge}TAIL-OF-OUTPUT"}}]}}"#
    ))
    .expect("json");
    let mut count = |s: &str| s.len() / 4;
    let budget = 2000;
    let (prompt, _) = render_within(&req, budget, &mut count).expect("must not refuse");
    assert!(count(&prompt) <= budget, "still over budget");
    assert!(prompt.contains("SYSTEM MARKER"), "system message lost");
    // Both ends survive: the head says what the tool was, the tail is
    // usually where the answer is.
    assert!(
        prompt.contains("HEAD-OF-OUTPUT"),
        "head of the output was lost"
    );
    assert!(
        prompt.contains("TAIL-OF-OUTPUT"),
        "tail of the output was lost"
    );
    assert!(
        prompt.contains("characters elided"),
        "elision was not declared"
    );
    assert!(
        tool_responses_are_inside_a_user_turn(&prompt),
        "eliding broke the turn structure"
    );
}

#[test]
fn a_window_too_small_for_anything_still_says_so() {
    let req = long_conversation(1, 40);
    let mut count = |s: &str| s.len() / 4;
    assert!(
        render_within(&req, 5, &mut count).is_err(),
        "a window that cannot hold even an elided turn has to say so"
    );
}

/// MiMo writes its arguments as JSON, and they go back to the client as
/// written: the model already typed them.
#[test]
fn a_mimo_call_comes_back_as_the_model_wrote_it() {
    let r = Template::Mimo.parse_reply(
        "Adding.\n</think>Sure.<tool_call><function=add>{\"a\": 17, \"note\": \"01234\"}\
         </function></tool_call>",
        &tool_types(),
    );
    assert_eq!(r.reasoning, "Adding.");
    assert_eq!(r.content, "Sure.");
    assert_eq!(
        r.calls,
        vec![Call {
            name: "add".into(),
            arguments: "{\"a\": 17, \"note\": \"01234\"}".into(),
        }]
    );

    // No arguments at all is still a call, with an empty object.
    let r = Template::Mimo.parse_reply(
        "</think><tool_call><function=now></function></tool_call>",
        &[],
    );
    assert_eq!(r.calls[0].arguments, "{}");

    // Qwen's parameter form, which a relative may slip into, reads as that.
    let r = Template::Mimo.parse_reply(
        "</think><tool_call>\n<function=add>\n<parameter=a>\n17\n</parameter>\n</function>\n</tool_call>",
        &tool_types(),
    );
    assert_eq!(r.calls[0].arguments, "{\"a\": 17}");
}

/// What the model wrote, returned by the client as history, must print
/// back as exactly what the model wrote. Anything else and every later
/// turn reads a call it never made.
#[test]
fn a_mimo_call_round_trips_through_the_history_unchanged() {
    let said = "<tool_call><function=add>{\"a\":17,\"b\": 25}</function></tool_call>";
    let reply = Template::Mimo.parse_reply(&format!("x</think>{said}"), &tool_types());
    let c = &reply.calls[0];
    let req = Json::parse(&format!(
        r#"{{"messages":[{{"role":"user","content":"go"}},
            {{"role":"assistant","content":"","tool_calls":[{{"id":"c","type":"function",
              "function":{{"name":{},"arguments":{}}}}}]}},
            {{"role":"tool","content":"42"}}]}}"#,
        tojson(&Json::Str(c.name.clone())),
        tojson(&Json::Str(c.arguments.clone())),
    ))
    .expect("json");
    let prompt = Template::Mimo.render(&req).expect("render");
    assert!(
        prompt.contains(said),
        "the call changed on the way back:\n{prompt}"
    );
}

#[test]
fn a_mimo_stream_splits_like_the_finished_reply() {
    use lex_rt::chat::{Piece, Stream};

    let full = "mulling\n</think>On it.<tool_call><function=add>{\"a\": 7}</function></tool_call>";
    let mut s = Stream::new(Template::Mimo);
    let mut content = String::new();
    for i in 1..full.len() {
        if full.is_char_boundary(i) {
            for p in s.push(&full[..i]) {
                if let Piece::Content(t) = p {
                    content.push_str(&t);
                }
            }
        }
    }
    let (last, reply) = s.finish(full, &tool_types());
    for p in last {
        if let Piece::Content(t) = p {
            content.push_str(&t);
        }
    }
    assert_eq!(content, "On it.");
    assert_eq!(reply.calls[0].arguments, "{\"a\": 7}");
}

/// The template would print a placeholder and the model would then go
/// looking for a picture that is not there.
#[test]
fn a_picture_is_refused_not_dropped() {
    let req = Json::parse(
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"what is this"},
            {"type":"image_url","image_url":{"url":"data:,"}}]}]}"#,
    )
    .unwrap();
    let e = Template::Mimo
        .render(&req)
        .expect_err("an image must not vanish");
    assert!(e.contains("text only"), "{e}");
}

/// The template is picked from the model's own file, and a GGUF whose
/// template is not one written out here is refused, not given Qwen's.
#[test]
fn each_model_gets_its_own_template() {
    assert_eq!(Template::recognise(""), None);
    assert_eq!(
        Template::recognise("{{ '<|im_start|>' + message.role }}<tool_call>"),
        None
    );
    if lex_rt::gguf::gguf_path("maternion/mimo-v2.6:9b")
        .ok()
        .flatten()
        .is_some()
    {
        assert_eq!(
            Template::for_model("maternion/mimo-v2.6:9b"),
            Ok(Template::Mimo)
        );
    } else {
        eprintln!("skipping the MiMo half: maternion/mimo-v2.6:9b is not pulled");
    }
    if lex_rt::qwen::Store::open("qwen3.8:27b-mlx").is_ok() {
        assert_eq!(Template::for_model("qwen3.8:27b-mlx"), Ok(Template::Qwen38));
    }
}

/// An agent's history grows by a turn per request. Once it is over budget,
/// trimming must not move the cut every turn: the prefix cache only helps a
/// prompt that begins the way the last one did, and a cut that slides one
/// turn per request makes every prompt begin somewhere new -- the whole
/// window re-read, every turn, for the rest of a long task.
#[test]
fn trimming_keeps_the_start_of_the_prompt_still_while_history_grows() {
    for t in [Template::Qwen38, Template::Mimo] {
        let budget = 8000;
        // Exact and additive, as a tokenizer is across `<|im_start|>`:
        // special tokens split the text, so turns count independently.
        let mut count = |s: &str| s.len();
        let step = budget / 4;
        let (mut prev, mut prev_removed) = (String::new(), 0usize);
        let (mut jumps, mut trimmed, mut over) = (0, 0, 0usize);
        for rounds in 1..80 {
            let mut m = vec![
                r#"{"role":"system","content":"SYSTEM"}"#.to_string(),
                r#"{"role":"user","content":"TASK"}"#.to_string(),
            ];
            for i in 0..rounds {
                m.push(format!(
                    r#"{{"role":"assistant","content":"","tool_calls":[{{"id":"c{i}","type":"function",
                       "function":{{"name":"grep","arguments":"{{\"p\":\"{i}\"}}"}}}}]}}"#
                ));
                // Results of uneven size, as a real task's are.
                m.push(format!(
                    r#"{{"role":"tool","content":"R{i} {}"}}"#,
                    "x".repeat(150 + (i * 97) % 400)
                ));
            }
            let req = Json::parse(&format!(r#"{{"messages":[{}]}}"#, m.join(","))).expect("json");
            let (prompt, _) = t.render_within(&req, budget, &mut count).expect("render");
            let removed = count(&t.render(&req).unwrap()) - count(&prompt);
            assert!(
                count(&prompt) <= budget,
                "{t:?}: over budget at {rounds} rounds"
            );
            assert!(prompt.contains("TASK") && prompt.contains(&format!("R{} ", rounds - 1)));
            if removed > 0 {
                trimmed += 1;
            }
            if removed == prev_removed {
                // Nothing more dropped, so the new prompt must begin with
                // all of the old one but its generation prompt -- which the
                // new one replaces with the turn the model gave.
                let shared = prev
                    .bytes()
                    .zip(prompt.bytes())
                    .take_while(|(a, b)| a == b)
                    .count();
                assert!(
                    shared + 40 >= prev.len(),
                    "{t:?}: the start moved at {rounds} rounds with nothing more dropped"
                );
            } else {
                assert!(
                    removed > prev_removed,
                    "{t:?}: the cut moved backwards at {rounds}"
                );
                jumps += 1;
            }
            over = count(&t.render(&req).unwrap()).saturating_sub(budget);
            prev = prompt;
            prev_removed = removed;
        }
        // The cut moves when the overflow crosses a multiple of the step,
        // so at most once per step of growth.
        assert!(
            jumps <= over.div_ceil(step),
            "{t:?}: {jumps} moves for {over} tokens of overflow at a step of {step}"
        );
        assert!(
            trimmed > 30,
            "{t:?}: only {trimmed} trimmed turns, so nothing is being tested"
        );
        // Cutting at the fewest turns that fit would move on every one.
        assert!(
            jumps < trimmed / 2,
            "{t:?}: {jumps} jumps over {trimmed} trimmed turns"
        );
        eprintln!("{t:?}: the start moved {jumps} times over {trimmed} trimmed turns");
    }
}

/// `enable_thinking: false` (Ollama's `think: false`, which lex-code sends
/// as `OLLAMA_THINK=false`) closes the reasoning block in the prompt, as
/// the checkpoint's template does:
/// `{%- if enable_thinking is defined and enable_thinking is false %}
/// {{- '<think>\n\n</think>\n\n' }}`. The reply then holds no `</think>`,
/// and must not be read as all reasoning -- that would hide its tool calls.
#[test]
fn thinking_off_closes_the_block_and_the_reply_is_content() {
    use lex_rt::chat::{Piece, Stream, thinks};
    let on = Json::parse(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    let off = Json::parse(
        r#"{"messages":[{"role":"user","content":"hi"}],
            "chat_template_kwargs":{"enable_thinking":false}}"#,
    )
    .unwrap();
    assert!(thinks(&on) && !thinks(&off));
    assert!(
        render(&on)
            .unwrap()
            .ends_with("<|im_start|>assistant\n<think>\n")
    );
    assert!(
        render(&off)
            .unwrap()
            .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    );

    let reply = "Let me add them.\n\n<tool_call>\n<function=add>\n<parameter=a>\n17\n\
                 </parameter>\n<parameter=b>\n25\n</parameter>\n</function>\n</tool_call>";
    let r = Template::Qwen38.parse_reply_after(reply, &tool_types(), false);
    assert_eq!(r.reasoning, "");
    assert_eq!(r.content, "Let me add them.");
    assert_eq!(r.calls.len(), 1, "the call was read as reasoning");
    // Read as if thinking were on, the same reply is all reasoning.
    assert!(parse_reply(reply, &tool_types()).calls.is_empty());

    // Streamed, the first piece is content, not reasoning.
    let mut s = Stream::for_request(Template::Qwen38, &off);
    assert_eq!(
        s.push("Let me add"),
        vec![Piece::Content("Let me add".into())]
    );
}
