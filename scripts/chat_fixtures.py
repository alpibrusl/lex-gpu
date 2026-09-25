#!/usr/bin/env python3
"""Expected ChatML prompts, rendered by the model's own chat template.

`serve.rs` builds the prompt in Rust because this repository does not carry
a Jinja engine. That is only safe if the two agree, so this renders the
template out of the checkpoint and writes each case as a fixture the Rust
test compares against byte for byte.

A wrong tool template does not fail loudly -- the model simply never emits
a tool call -- which is why this is a golden test and not an eyeball.

    python3 scripts/chat_fixtures.py [--out crates/lex-rt/tests/data/chatml]
"""
import argparse, glob, json, os, sys

TPL_KEY = "chat_template"


def find_template():
    """The tokenizer_config.json blob in Ollama's store, by content."""
    roots = [
        os.path.expanduser("~/.ollama/models/blobs"),
        "/usr/share/ollama/.ollama/models/blobs",
    ]
    for r in roots:
        for p in sorted(glob.glob(os.path.join(r, "sha256-*"))):
            if not (0 < os.path.getsize(p) < 1 << 20):
                continue
            try:
                with open(p) as f:
                    j = json.load(f)
            except Exception:
                continue
            if isinstance(j, dict) and TPL_KEY in j and "<|im_start|>" in j[TPL_KEY]:
                return p, j[TPL_KEY]
    sys.exit("no chat_template blob found; is qwen3.8:27b-mlx pulled?")


# The OpenAI wire form sends tool-call arguments as a JSON *string*; the
# template iterates them as a mapping. Converting here is not a convenience
# -- it is the same conversion serve.rs has to make, so the fixture pins it.
def for_template(messages):
    out = []
    for m in messages:
        m = dict(m)
        for c in m.get("tool_calls") or []:
            a = c["function"].get("arguments")
            if isinstance(a, str):
                c["function"] = dict(c["function"], arguments=json.loads(a or "{}"))
        out.append(m)
    return out


ADD = {"type": "function", "function": {
    "name": "add", "description": "Add two numbers.",
    "parameters": {"type": "object",
                   "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                   "required": ["a", "b"]}}}
NOW = {"type": "function", "function": {
    "name": "now", "description": "Current time.",
    "parameters": {"type": "object", "properties": {}}}}
# jinja's tojson is not plain json.dumps: it sorts keys, escapes non-ASCII,
# and then escapes < > & ' as \uXXXX. A description is free text, so all of
# that is reachable from a real tool list.
ODD = {"type": "function", "function": {
    "name": "cmp", "description": "True when a < b & b > c, the 'usual' way — na\u00efvely.",
    "parameters": {"type": "object",
                   "properties": {"a": {"type": "string", "description": "<left>"}},
                   "required": ["a"]}}}

U = lambda s: {"role": "user", "content": s}
S = lambda s: {"role": "system", "content": s}
CALL = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "call_1", "type": "function",
     "function": {"name": "add", "arguments": '{"a":"17","b":"25"}'}}]}

CASES = {
    "plain":            {"messages": [U("hi")]},
    "system":           {"messages": [S("Be brief."), U("hi")]},
    "tools":            {"messages": [U("What is 17 + 25?")], "tools": [ADD]},
    "system_tools":     {"messages": [S("Be brief."), U("hi")], "tools": [ADD]},
    "two_tools":        {"messages": [U("hi")], "tools": [ADD, NOW]},
    "tool_result":      {"messages": [U("What is 17 + 25?"), CALL,
                                      {"role": "tool", "content": '{"result":"42"}'}],
                         "tools": [ADD]},
    "two_tool_results": {"messages": [U("go"), CALL,
                                      {"role": "tool", "content": "a"},
                                      {"role": "tool", "content": "b"}],
                         "tools": [ADD]},
    "odd_chars":        {"messages": [U("hi")], "tools": [ODD]},
    "effort_low":       {"messages": [U("hi")], "reasoning_effort": "low"},
    "effort_medium":    {"messages": [U("hi")], "reasoning_effort": "medium"},
    "prose_then_call":  {"messages": [U("go"),
                                      {"role": "assistant", "content": "Let me add them.",
                                       "tool_calls": CALL["tool_calls"]},
                                      {"role": "tool", "content": "42"}],
                         "tools": [ADD]},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crates/lex-rt/tests/data/chatml")
    args = ap.parse_args()
    from jinja2 import BaseLoader, Environment

    path, tpl = find_template()
    env = Environment(loader=BaseLoader())
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    t = env.from_string(tpl)

    os.makedirs(args.out, exist_ok=True)
    print(f"template from {path}")
    for name, req in sorted(CASES.items()):
        kw = {k: v for k, v in req.items() if k != "messages"}
        want = t.render(messages=for_template(req["messages"]),
                        add_generation_prompt=True, **kw)
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(req, f, indent=2, sort_keys=True)
            f.write("\n")
        with open(os.path.join(args.out, name + ".txt"), "w") as f:
            f.write(want)
        print(f"  {name:18} {len(want):6} bytes")


if __name__ == "__main__":
    main()
