//! Dump what the multi-token-prediction head needs, so its acceptance can
//! be measured before a single kernel is written for it.
//!
//! The head drafts token `t+2` from the hidden state at `t` and the
//! embedding of `t+1`. So for every step we record the model's own hidden
//! state and the token it actually chose; `scripts/mtp_accept.py` then runs
//! the head over that trace in f32 and counts how often its draft matches
//! what the model went on to pick. That is the acceptance rate, and it is
//! the number the whole speculation plan rests on.
//!
//! Doing it this way costs one ordinary decode run rather than a second
//! implementation of the model.
//!
//!     cargo run --release -p lex-rt --example mtp_trace -- --steps 256
//!
//! Writes `mtp_trace.bin`: a little-endian header of three u32 (steps,
//! hidden, magic 0x4d545030) then, per step, one u32 token and `hidden`
//! f32. The token recorded with a hidden state is the one the model chose
//! *from* that state, so step `i`'s draft target is step `i+1`'s token.

#[cfg(target_os = "macos")]
fn main() -> Result<(), String> {
    use lex_rt::qwen_run::Runner;
    use std::io::Write;

    let mut model = "qwen3.8:27b-mlx".to_string();
    let mut steps = 256usize;
    let mut out = "mtp_trace.bin".to_string();
    // "The capital of France is" -- replaced by --ids for other text.
    let mut ids = vec![760u32, 6511, 314, 9338, 369];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().ok_or(format!("{a} needs a value"));
        match a.as_str() {
            "--model" => model = val()?,
            "--steps" => steps = val()?.parse().map_err(|_| "bad --steps")?,
            "--out" => out = val()?,
            "--ids" => {
                ids = val()?
                    .split(',')
                    .map(|s| s.trim().parse().map_err(|_| "bad --ids".to_string()))
                    .collect::<Result<_, _>>()?
            }
            other => return Err(format!("unknown argument `{other}`")),
        }
    }

    let mut rt = Runner::load(&model, ids.len() + steps + 8)?;
    println!("{model}: {} on {}", rt.pos(), rt.device());

    let argmax = |v: &[f32]| {
        (0..v.len())
            .max_by(|&a, &b| v[a].total_cmp(&v[b]))
            .expect("logits") as u32
    };

    // Feed the prompt, keeping only the last step's state.
    let mut logits = vec![];
    for &t in &ids {
        logits = rt.step(t)?;
    }

    let mut trace: Vec<(u32, Vec<f32>)> = vec![];
    for _ in 0..steps {
        // The state that produced this token, and the token itself.
        let next = argmax(&logits);
        trace.push((next, rt.hidden()));
        logits = rt.step(next)?;
    }

    let hidden = trace[0].1.len();
    let mut f = std::fs::File::create(&out).map_err(|e| format!("{out}: {e}"))?;
    let head = [trace.len() as u32, hidden as u32, 0x4d54_5030];
    let mut bytes: Vec<u8> = head.iter().flat_map(|w| w.to_le_bytes()).collect();
    for (tok, h) in &trace {
        bytes.extend(tok.to_le_bytes());
        bytes.extend(h.iter().flat_map(|x| x.to_le_bytes()));
    }
    f.write_all(&bytes).map_err(|e| format!("{out}: {e}"))?;
    println!(
        "{} steps x {hidden} hidden -> {out} ({:.1} MB)",
        trace.len(),
        bytes.len() as f64 / 1e6
    );
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("this example needs Metal");
}
