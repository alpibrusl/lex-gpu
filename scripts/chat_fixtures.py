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
import argparse, copy, glob, json, os, sys

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
    # A deep copy: the cases share CALL, and converting it in place would
    # hand every later case -- MiMo's among them -- a mapping where the
    # wire has a string.
    out = []
    for m in copy.deepcopy(messages):
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
# Characters plain Jinja's tojson would escape (< > & ' and non-ASCII) and
# transformers' does not. A description is free text, so all of them are
# reachable from a real tool list.
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


def gguf_path(tag):
    """The model blob an Ollama tag's manifest points at."""
    repo, _, ver = tag.partition(":")
    ns = repo if "/" in repo else "library/" + repo
    for root in [os.path.expanduser("~/.ollama/models"), "/usr/share/ollama/.ollama/models"]:
        m = os.path.join(root, "manifests/registry.ollama.ai", ns, ver or "latest")
        if os.path.exists(m):
            for l in json.load(open(m))["layers"]:
                if l["mediaType"] == "application/vnd.ollama.image.model":
                    return os.path.join(root, "blobs", l["digest"].replace(":", "-"))
    sys.exit(f"{tag} is not pulled")


def gguf_meta(path, key):
    """One metadata value from a GGUF header, read without the weights."""
    import struct
    fixed = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
             7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    with open(path, "rb") as f:
        rd = lambda fmt: struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
        text = lambda: f.read(rd("<Q")).decode("utf-8")
        if f.read(4) != b"GGUF":
            sys.exit(f"{path}: not a GGUF")
        rd("<I"); rd("<Q"); n = rd("<Q")

        def value(t):
            if t == 8:
                return text()
            if t == 9:
                et, count = rd("<I"), rd("<Q")
                if et == 8:
                    return [text() for _ in range(count)]
                f.seek(struct.calcsize(fixed[et]) * count, 1)
                return None
            return rd(fixed[t])

        for _ in range(n):
            k = text()
            v = value(rd("<I"))
            if k == key:
                return v
    sys.exit(f"{path}: no {key}")


# MiMo's template is the llama.cpp adapter shipped in its GGUF, written for
# Hugging Face's Jinja environment: it calls `tojson(ensure_ascii=False)`,
# which plain Jinja's filter does not accept, and wraps assistant turns in
# `{% generation %}`, which plain Jinja does not parse. So it renders
# through transformers' own compiler -- the environment the model's
# training data went through -- and not the plain one above.
#
# Tool-call arguments stay the JSON *strings* the wire carries: this
# template prints a string argument as-is, so what the model reads back is
# exactly what it wrote. Only `object_args` sends a mapping.
MIMO = "maternion/mimo-v2.6:9b"
ODD_MIMO = {"type": "function", "function": {
    "name": "cmp", "description": "True when a < b & b > c, the 'usual' way — naïvely \U0001F600.",
    "parameters": {"type": "object",
                   "properties": {"a": {"type": "string", "description": "<left>"}},
                   "required": ["a"]}}}
MIMO_CASES = {
    "plain":             {"messages": [U("hi")]},
    "system":            {"messages": [S("Be brief."), U("hi")]},
    "tools":             {"messages": [U("What is 17 + 25?")], "tools": [ADD]},
    "system_tools":      {"messages": [S("Be brief."), U("hi")], "tools": [ADD]},
    "two_tools":         {"messages": [U("hi")], "tools": [ADD, NOW]},
    "tool_result":       {"messages": [U("What is 17 + 25?"), CALL,
                                       {"role": "tool", "content": '{"result":"42"}'}],
                          "tools": [ADD]},
    "two_tool_results":  {"messages": [U("go"), CALL,
                                       {"role": "tool", "content": "a"},
                                       {"role": "tool", "content": "b"}],
                          "tools": [ADD]},
    "odd_chars":         {"messages": [U("naïve 😀 <b>")], "tools": [ODD_MIMO]},
    "prose_then_call":   {"messages": [U("go"),
                                       {"role": "assistant", "content": "Let me add them.",
                                        "tool_calls": CALL["tool_calls"]},
                                       {"role": "tool", "content": "42"}],
                          "tools": [ADD]},
    "null_content_call": {"messages": [U("go"), dict(CALL, content=None),
                                       {"role": "tool", "content": "42"}],
                          "tools": [ADD]},
    "object_args":       {"messages": [U("go"),
                                       {"role": "assistant", "content": "", "tool_calls": [
                                           {"id": "c", "type": "function", "function": {
                                               "name": "add", "arguments": {"b": "25", "a": "17"}}}]},
                                       {"role": "tool", "content": "42"}],
                          "tools": [ADD]},
    "history_reasoning": {"messages": [U("hi"),
                                       {"role": "assistant", "content": "Hello!",
                                        "reasoning_content": "\nA greeting.\n"},
                                       U("again")]},
    "no_think":          {"messages": [U("hi")],
                          "chat_template_kwargs": {"enable_thinking": False}},
    "text_parts":        {"messages": [{"role": "user", "content": [
                                           {"type": "text", "text": "one "},
                                           {"type": "text", "text": "two"}]}]},
    "untrimmed":         {"messages": [S("  padded \n"), U("\n hi  ")]},
    "late_system":       {"messages": [U("hi"), S("Now be brief."), U("again")]},
}


def mimo(out):
    from transformers.utils.chat_template_utils import _compile_jinja_template

    path = gguf_path(MIMO)
    t = _compile_jinja_template(gguf_meta(path, "tokenizer.chat_template"))
    os.makedirs(out, exist_ok=True)
    print(f"MiMo template from {path}")
    for name, req in sorted(MIMO_CASES.items()):
        kw = dict(req.get("chat_template_kwargs", {}))
        if "tools" in req:
            kw["tools"] = req["tools"]
        want = t.render(messages=req["messages"], add_generation_prompt=True, **kw)
        # Key order is part of the prompt here, so the request is written
        # in the order it was built -- and ASCII-escaped, so the reader's
        # surrogate pairs are exercised by the emoji.
        with open(os.path.join(out, name + ".json"), "w") as f:
            json.dump(req, f, indent=2)
            f.write("\n")
        with open(os.path.join(out, name + ".txt"), "w") as f:
            f.write(want)
        print(f"  {name:18} {len(want):6} bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crates/lex-rt/tests/data/chatml")
    ap.add_argument("--mimo-out", default="crates/lex-rt/tests/data/chatml_mimo")
    args = ap.parse_args()
    from transformers.utils.chat_template_utils import _compile_jinja_template

    # Through transformers' environment, not plain jinja2: the two define
    # `tojson` differently (plain sorts keys and escapes ' < > & and
    # non-ASCII), and transformers' is the one the training data saw.
    path, tpl = find_template()
    t = _compile_jinja_template(tpl)

    os.makedirs(args.out, exist_ok=True)
    print(f"template from {path}")
    for name, req in sorted(CASES.items()):
        kw = {k: v for k, v in req.items() if k != "messages"}
        want = t.render(messages=for_template(req["messages"]),
                        add_generation_prompt=True, **kw)
        # Key order is part of the prompt, so the request keeps its own.
        with open(os.path.join(args.out, name + ".json"), "w") as f:
            json.dump(req, f, indent=2)
            f.write("\n")
        with open(os.path.join(args.out, name + ".txt"), "w") as f:
            f.write(want)
        print(f"  {name:18} {len(want):6} bytes")

    mimo(args.mimo_out)


if __name__ == "__main__":
    main()
