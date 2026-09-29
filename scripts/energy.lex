# Joules per token, for this engine and for Ollama on the same machine.
#
# Tokens per second says how fast; this says what it cost. On a decode that
# is memory-bound those are different questions -- an engine reading fewer
# bytes per token can be slower and still cheaper -- so it is worth
# measuring rather than inferring.
#
#   lex run --allow-effects io,net,proc,time --allow-proc nvidia-smi \
#     scripts/energy.lex main '"nvidia"' '"ollama"' '"llama3.2:1b"' '256' '8'
#
# Measured on an NVIDIA L4: Ollama serving llama3.2:1b at 160 tok/s draws
# 323-363 mJ/token, 133-151 of that above idle.
#
# `--allow-proc nvidia-smi` is the point of writing this in Lex rather than
# Python: the binary it may spawn is named at the command line, the effect
# row says it reaches a subprocess, a socket and the clock, and
# `lex authority derive` proves it reaches nothing else. A measurement tool
# is exactly the kind of program that should have to show that.
#
# There is no background thread here. `conc.spawn` is a message handler,
# not a poller, so instead the sampler timestamps its own output and its
# pipe buffers while the request is in flight; the samples are drained
# afterwards and placed on the timeline by their own clocks, which is more
# accurate than timing them as they are read.

import "std.process" as process
import "std.http"    as http
import "std.json"    as json
import "std.bytes"   as bytes
import "std.list"    as list
import "std.map"     as map
import "std.str"     as str
import "std.int"     as int
import "std.float"   as float
import "std.io"      as io
import "std.time"    as time

# A power reading: seconds since midnight, and watts.
type Sample = { t :: Float, w :: Float }

# "18:04:05.123" -> seconds since midnight. Differences are all this is
# used for, so the date is dropped; a run across midnight would be wrong
# and is not a thing that happens to a two-minute benchmark.
fn clock_seconds(hms :: Str) -> Option[Float] {
  let parts := str.split(hms, ":")
  if list.len(parts) != 3 {
    None
  } else {
    match (str.to_float(nth(parts, 0)), str.to_float(nth(parts, 1)), str.to_float(nth(parts, 2))) {
      (Some(h), Some(m), Some(s)) => Some(h * 3600.0 + m * 60.0 + s),
      _ => None,
    }
  }
}

fn nth(xs :: List[Str], i :: Int) -> Str {
  match list.head(drop(xs, i)) { Some(v) => v, None => "" }
}

fn drop(xs :: List[Str], i :: Int) -> List[Str] {
  if i <= 0 { xs } else { drop(list.tail(xs), i - 1) }
}

# nvidia-smi --format=csv,noheader,nounits --query-gpu=timestamp,power.draw
# emits "2026/09/25 18:04:05.123, 40.20".
fn parse_sample(line :: Str) -> Option[Sample] {
  let cols := str.split(str.trim(line), ",")
  if list.len(cols) < 2 {
    None
  } else {
    # "2026/09/25 18:04:05.123" -- the date half is dropped, so splitting
    # on the space is all the parsing the timestamp needs.
    let stamp := str.split(str.trim(nth(cols, 0)), " ")
    match str.to_float(str.trim(nth(cols, 1))) {
      None => None,
      Some(w) => if list.len(stamp) < 2 {
        None
      } else {
        match clock_seconds(nth(stamp, 1)) {
          Some(t) => Some({ t: t, w: w }),
          None => None,
        }
      },
    }
  }
}

# Read until a sample lands at or past `until`, or the budget runs out.
# The budget is the guard against a sampler that died: read_stdout_line
# blocks, and a benchmark that hangs forever is worse than one that fails.
fn drain(h :: ProcessHandle, until :: Float, budget :: Int, acc :: List[Sample]) -> [proc] List[Sample] {
  if budget <= 0 {
    list.reverse(acc)
  } else {
    match process.read_stdout_line(h) {
      None => list.reverse(acc),
      Some(line) => match parse_sample(line) {
        None => drain(h, until, budget - 1, acc),
        Some(s) => if s.t >= until {
          list.reverse(list.cons(s, acc))
        } else {
          drain(h, until, budget - 1, list.cons(s, acc))
        },
      },
    }
  }
}

# powermetrics has no per-sample wall clock, but it prints how long each
# sample took -- "(200.53ms elapsed)" -- so the timeline is the running sum
# of those. Nominal cadence would drift against a window measured in real
# seconds; this does not.
fn parse_elapsed(line :: Str) -> Option[Float] {
  if not str.contains(line, "ms elapsed") {
    None
  } else {
    # "*** Sampled system activity (...) (200.53ms elapsed) ***"
    let tail := nth(str.split(line, "("), list.len(str.split(line, "(")) - 1)
    match str.to_float(nth(str.split(tail, "ms"), 0)) {
      Some(ms) => Some(ms / 1000.0),
      None => None,
    }
  }
}

fn parse_gpu_mw(line :: Str) -> Option[Float] {
  let t := str.trim(line)
  if not str.starts_with(t, "GPU Power:") {
    None
  } else {
    match str.strip_prefix(t, "GPU Power:") {
      None => None,
      Some(rest) => match str.to_float(nth(str.split(str.trim(rest), " "), 0)) {
        Some(mw) => Some(mw / 1000.0),
        None => None,
      },
    }
  }
}

fn drain_apple(h :: ProcessHandle, until :: Float, budget :: Int, clock :: Float, acc :: List[Sample]) -> [proc] List[Sample] {
  if budget <= 0 {
    list.reverse(acc)
  } else {
    match process.read_stdout_line(h) {
      None => list.reverse(acc),
      Some(line) => match parse_elapsed(line) {
        Some(dt) => drain_apple(h, until, budget - 1, clock + dt, acc),
        None => match parse_gpu_mw(line) {
          None => drain_apple(h, until, budget - 1, clock, acc),
          Some(w) => if clock >= until {
            list.reverse(list.cons({ t: clock, w: w }, acc))
          } else {
            drain_apple(h, until, budget - 1, clock, list.cons({ t: clock, w: w }, acc))
          },
        },
      },
    }
  }
}

# Watts at `t`, interpolated between the samples bracketing it and clamped
# at the ends. Integrating only between the samples strictly inside a
# window and then dividing by the whole window under-reports by about 15%,
# which is a mistake the Python version made first.
fn watts_at(ss :: List[Sample], t :: Float) -> Float {
  match (list.head(ss), list.head(list.reverse(ss))) {
    (Some(a), Some(z)) => if t <= a.t { a.w } else { if t >= z.t { z.w } else { between(ss, t) } },
    _ => 0.0,
  }
}

fn between(ss :: List[Sample], t :: Float) -> Float {
  match list.head(ss) {
    None => 0.0,
    Some(a) => {
      let rest := list.tail(ss)
      match list.head(rest) {
        None => a.w,
        Some(b) => if a.t <= t and t <= b.t {
          let span := b.t - a.t
          if span <= 0.0 { a.w } else { a.w + (t - a.t) / span * (b.w - a.w) }
        } else {
          between(rest, t)
        },
      }
    },
  }
}

# Trapezoid over [t0, t1], edges included.
fn joules(ss :: List[Sample], t0 :: Float, t1 :: Float) -> Float {
  let inner := list.filter(ss, fn (s :: Sample) -> Bool { s.t > t0 and s.t < t1 })
  let seq := list.concat(list.cons({ t: t0, w: watts_at(ss, t0) }, inner), [{ t: t1, w: watts_at(ss, t1) }])
  trapezoid(seq, 0.0)
}

fn trapezoid(ss :: List[Sample], acc :: Float) -> Float {
  match list.head(ss) {
    None => acc,
    Some(a) => {
      let rest := list.tail(ss)
      match list.head(rest) {
        None => acc,
        Some(b) => trapezoid(rest, acc + (b.t - a.t) * (a.w + b.w) / 2.0),
      }
    },
  }
}

# A prompt that keeps going, so the run measures decode and not an early
# stop.
fn prompt() -> Str {
  "Write a long, detailed description of how a bicycle works, part by part. Do not stop early."
}

fn post_json(url :: Str, body :: Str) -> [net] Result[Str, Str] {
  let hdrs := map.set(map.new(), "content-type", "application/json")
  let req := { method: "POST", url: url, headers: hdrs, body: Some(bytes.from_str(body)), timeout_ms: Some(1800000) }
  match http.send(req) {
    # HttpError is its own type; the caller only needs to know it failed.
    Err(_) => Err("request failed or timed out"),
    Ok(r) => if r.status >= 400 {
      Err(str.concat("HTTP ", int.to_str(r.status)))
    } else {
      # Bytes may not be valid UTF-8, so this is a Result too.
      match bytes.to_str(r.body) {
        Err(e) => Err(str.concat("response was not text: ", e)),
        Ok(t) => Ok(t),
      }
    },
  }
}

# std.json.parse decodes straight into a record and ignores the fields
# that are not asked for, so the shape of the reply is declared rather
# than walked.
type Usage      = { completion_tokens :: Int }
type ChatReply  = { usage :: Usage }
# Ollama counts them itself, which beats trusting the clock.
type OllamaReply = { eval_count :: Int }

fn decode_chat(text :: Str) -> Result[ChatReply, Str] {
  json.parse(text)
}

fn decode_ollama(text :: Str) -> Result[OllamaReply, Str] {
  json.parse(text)
}

# engine "cmd": `model` is a command line, and `tokens` is how many tokens
# it was told to produce. For engines that are a binary rather than a
# server -- lex-gpu's own `generate` takes token ids and a step count --
# there is nothing to POST to, and wrapping one in a server to measure it
# would measure the server too.
fn run_command(line :: Str, tokens :: Int) -> [proc] Result[Int, Str] {
  let words := list.filter(str.split(line, " "), fn (w :: Str) -> Bool { not str.is_empty(w) })
  match list.head(words) {
    None => Err("empty command"),
    Some(bin) => {
      match process.run(bin, list.tail(words)) {
        Err(e) => Err(str.join(["cannot run ", bin, ": ", e], "")),
        Ok(out) => if out.exit_code != 0 {
          Err(str.join([bin, " exited ", int.to_str(out.exit_code), ": ", str.slice(out.stderr, 0, 300)], ""))
        } else {
          Ok(tokens)
        },
      }
    },
  }
}

fn generate(engine :: Str, host :: Str, model :: Str, tokens :: Int) -> [net] Result[Int, Str] {
  let body := if engine == "ollama" {
    str.join(["{\"model\":\"", model, "\",\"prompt\":\"", prompt(),
              "\",\"stream\":false,\"options\":{\"num_predict\":", int.to_str(tokens), "}}"], "")
  } else {
    str.join(["{\"model\":\"", model, "\",\"messages\":[{\"role\":\"user\",\"content\":\"", prompt(),
              "\"}],\"max_tokens\":", int.to_str(tokens), "}"], "")
  }
  let url := if engine == "ollama" { str.concat(host, "/api/generate") } else { str.concat(host, "/v1/chat/completions") }
  match post_json(url, body) {
    Err(e) => Err(e),
    Ok(text) => if engine == "ollama" {
      match decode_ollama(text) { Err(e) => Err(e), Ok(o) => Ok(o.eval_count) }
    } else {
      match decode_chat(text) { Err(e) => Err(e), Ok(o) => Ok(o.usage.completion_tokens) }
    },
  }
}

# `from` is where the apple clock has already reached. It has to be carried
# across calls: powermetrics' timeline is a running sum of its own elapsed
# figures, so restarting it at zero for each window puts the second window
# behind the first and the integral lands on nothing.
fn drive(engine :: Str, host :: Str, model :: Str, tokens :: Int) -> [net, proc] Result[Int, Str] {
  if engine == "cmd" { run_command(model, tokens) } else { generate(engine, host, model, tokens) }
}

fn sample_to(h :: ProcessHandle, apple :: Bool, until :: Float, budget :: Int, from :: Float) -> [proc] List[Sample] {
  if apple { drain_apple(h, until, budget, from, []) } else { drain(h, until, budget, []) }
}

fn last_t(ss :: List[Sample], fallback :: Float) -> Float {
  match list.head(list.reverse(ss)) { Some(s) => s.t, None => fallback }
}

# Warm first, then baseline, then measure.
#
# The order is the measurement. A cold first request carries the model load:
# on an L4, llama3.2:1b read 6984 mJ/token cold against 327 warm, a factor
# of twenty-one, and nothing in the output says which you got. Taking the
# baseline *after* the warm-up matters too -- a GPU straight off a run has
# not clocked down, and idle read 17 W cold against 30-34 W warm. Measuring
# a warm run against a cold baseline is what makes `marginal` swing by a
# third while `total` holds to 3%.
# `baseline_mw` is the background to subtract, in milliwatts; 0 measures
# it here instead.
#
# Measuring it per run is what the first comparison did, and it is wrong
# for comparing two engines: each one subtracted its own background, which
# differed by eighty times on a Mac (5.54 W against 0.07 W), so the
# marginal ratio said 1.69x where the totals said 1.92x. With one common
# background the ratio comes back to 1.92x whichever value is used --
# subtracting b*t from both changes the joules and never the ratio. Pass a
# common one when comparing; measure one when you want this machine's own
# absolute cost.
#
# `engine` "idle" measures the background and stops: run it with nothing
# serving for the machine's floor, and again with a server up but untouched
# for what merely holding a model costs.
fn main(backend :: Str, engine :: Str, model :: Str, tokens :: Int, idle_s :: Int, baseline_mw :: Int) -> [proc, net, io, time] Str {
  let host := if engine == "ollama" { "http://127.0.0.1:11434" } else { "http://127.0.0.1:8080" }
  let apple := backend == "apple"
  let opts := { cwd: None, env: map.new(), stdin: None }
  let bin := if apple { "powermetrics" } else { "nvidia-smi" }
  let args := if apple {
    ["--samplers", "gpu_power", "-i", "200"]
  } else {
    ["--query-gpu=timestamp,power.draw", "--format=csv,noheader,nounits", "--loop-ms=200"]
  }
  # Warm up before the sampler exists. Spawning it first and warming up
  # second leaves the warm-up's own samples sitting in the pipe, and the
  # idle window reads them: with a three-second warm-up the baseline came
  # back at 193 W against a 40 W idle rail, and `marginal` went negative.
  let measuring := engine != "idle"
  let _w := if measuring { io.print("warming up ...") } else { io.print("background only") }
  let _r := if measuring { drive(engine, host, model, 16) } else { Ok(0) }
  match process.spawn(bin, args, opts) {
    Err(e) => str.join(["cannot start ", bin, ": ", e], ""),
    Ok(h) => {
      # Anchor the timeline on the sampler's own clock, not ours.
      let first := sample_to(h, apple, 0.0, 200, 0.0)
      let t_start := match list.head(first) { Some(s) => s.t, None => 0.0 }
      let _n := io.print(str.join(["idle baseline for ", int.to_str(idle_s), "s ..."], ""))
      let idle := sample_to(h, apple, t_start + int.to_float(idle_s), 4000, t_start)
      let idle_end := last_t(idle, t_start)
      let measured_w := if idle_end > t_start { joules(idle, t_start, idle_end) / (idle_end - t_start) } else { 0.0 }
      # A baseline given on the command line wins, so two engines can be
      # compared against the same background.
      let idle_w := if baseline_mw > 0 { int.to_float(baseline_mw) / 1000.0 } else { measured_w }
      if not measuring {
        let _k2 := process.kill(h, "TERM")
        str.join(["\nbackground    ", float.to_str(measured_w), " W over ", int.to_str(idle_s), "s",
                  "\n\nRun this with nothing serving for the machine's floor, and again",
                  "\nwith a server up but untouched for what holding a model costs."], "")
      } else {

      let _g := io.print(str.join(["generating ", int.to_str(tokens), " tokens on ", engine, " ..."], ""))
      let t0 := idle_end
      let m0 := time.mono_ns()
      match drive(engine, host, model, tokens) {
        Err(e) => {
          let _k := process.kill(h, "TERM")
          str.concat("generation failed: ", e)
        },
        Ok(n) => {
          # How long the request took, by our own monotonic clock. The
          # sampler's pipe has been filling the whole time, so the samples
          # are drained up to that same length on its clock -- reading a
          # fixed few instead reports the first half-second of a
          # three-second request, which is what the stub caught.
          let m1 := time.mono_ns()
          let dur := int.to_float(m1 - m0) / 1000000000.0
          let t1 := t0 + dur
          let after := sample_to(h, apple, t1, 40000, idle_end)
          let _k := process.kill(h, "TERM")
          let window := list.concat(idle, after)
          let total := joules(window, t0, t1)
          let marginal := total - idle_w * dur
          let per := fn (j :: Float) -> Str { str.concat(float.to_str(j / int.to_float(n) * 1000.0), " mJ/token") }
          str.join([
            "\nengine        ", engine, " (", model, ")",
            "\ntokens        ", int.to_str(n), " in ", float.to_str(dur), "s",
            "\nidle          ", float.to_str(idle_w), " W",
            "\nwhile running ", float.to_str(if dur > 0.0 { total / dur } else { 0.0 }), " W",
            "\n\ntotal    ", per(total),
            "\nmarginal ", per(marginal), "  (above idle)",
            "\n\nCompare against another engine on this same machine, not across",
            "\nmachines: nvidia-smi reports board power and powermetrics SoC rails.",
          ], "")
        },
      }
      }
    },
  }
}
