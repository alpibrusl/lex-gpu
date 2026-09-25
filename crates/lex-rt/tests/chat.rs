//! The Rust chat template against the model's own Jinja one.
//!
//! `serve.rs` builds prompts in Rust because this repository carries no
//! Jinja engine, which is only safe if the two agree. Every `.txt` here was
//! rendered by the checkpoint's own `chat_template`
//! (`scripts/chat_fixtures.py`) and is compared byte for byte -- a tool
//! block in the wrong shape does not fail loudly, it just produces a model
//! that never calls anything.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use lex_rt::chat::{Call, parse_reply, render, render_within, tojson};
use lex_rt::json::Json;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/chatml")
}

#[test]
fn the_rust_template_matches_the_models_own() {
    let mut cases: Vec<String> = fs::read_dir(dir())
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
        "only {} fixtures; run scripts/chat_fixtures.py",
        cases.len()
    );

    for name in &cases {
        let req = fs::read_to_string(dir().join(format!("{name}.json"))).expect("request");
        let want = fs::read_to_string(dir().join(format!("{name}.txt"))).expect("expected");
        let got = render(&Json::parse(&req).expect("request json"))
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        if got != want {
            let at = got
                .char_indices()
                .zip(want.chars())
                .find(|((_, a), b)| a != b)
                .map_or(got.len().min(want.len()), |((i, _), _)| i);
            panic!(
                "{name}: prompts differ at byte {at}\n  ours: {:?}\n  jinja: {:?}",
                &got[at.saturating_sub(40)..(at + 60).min(got.len())],
                &want[at.saturating_sub(40)..(at + 60).min(want.len())],
            );
        }
    }
    // The tool cases are the ones with no second opinion in the wild, so
    // make sure they are actually among what just passed.
    for needed in ["tools", "tool_result", "two_tool_results", "odd_chars"] {
        assert!(cases.iter().any(|c| c == needed), "missing case {needed}");
    }
}

/// `tojson` is Jinja's, not `json.dumps`: sorted keys, spaced separators,
/// and `< > & '` escaped on top of the non-ASCII escaping.
#[test]
fn tojson_escapes_what_jinja_escapes() {
    let mut m = BTreeMap::new();
    m.insert("b".to_string(), Json::Num(1.0));
    m.insert("a".to_string(), Json::Str("x < y & 'z' — naïve".into()));
    m.insert("c".to_string(), Json::Arr(vec![Json::Bool(true), Json::Null]));
    assert_eq!(
        tojson(&Json::Obj(m)),
        "{\"a\": \"x \\u003c y \\u0026 \\u0027z\\u0027 \\u2014 na\\u00efve\", \"b\": 1, \
         \"c\": [true, null]}"
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
    let req = Json::parse(r#"{"messages":[{"role":"user","content":"hi"}],
        "reasoning_effort":"high"}"#)
    .unwrap();
    assert!(render(&req).is_err(), "`high` is not one the template takes");
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
        assert!(!content.contains(bad), "{bad:?} leaked into streamed content");
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

    assert!(dropped > 0, "nothing was dropped, so nothing is being tested");
    assert!(count(&prompt) <= budget, "still over budget after trimming");
    // The two that must survive: the tools and goal live in the system
    // block, and the last turn is the one being answered.
    assert!(prompt.contains("SYSTEM MARKER"), "system message was dropped");
    assert!(prompt.contains("LAST USER"), "the turn being answered was dropped");
    // The task itself is pinned. Dropping it is what made a lex-code run
    // end with the agent asking the user what they would like built.
    assert!(prompt.contains("FIRST USER"), "the task was dropped");
    assert!(!prompt.contains("TURNMARK0 "), "the oldest middle turn survived");
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
    assert!(prompt.contains("HEAD-OF-OUTPUT"), "head of the output was lost");
    assert!(prompt.contains("TAIL-OF-OUTPUT"), "tail of the output was lost");
    assert!(prompt.contains("characters elided"), "elision was not declared");
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
