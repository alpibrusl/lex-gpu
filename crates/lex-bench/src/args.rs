//! Hand-rolled argument parsing. A dependency-free binary is one less thing
//! between a fresh clone and a number.

use lex_ir::DType;

pub struct Args {
    pub dtype: DType,
    /// Size of each side of the copy benchmark, in MiB. Large enough to defeat
    /// caches, or the ceiling it measures is not the memory ceiling.
    pub copy_mib: usize,
    pub rows: usize,
    pub cols: usize,
    pub eps: f32,
    pub iters: usize,
    pub repeats: usize,
    pub emit: bool,
    /// Benchmark flash-attention decode instead of copy/rmsnorm.
    pub flash: bool,
    pub batch: usize,
    pub seq: usize,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            dtype: DType::F32,
            copy_mib: 256,
            // 4096 is the Llama-3-8B hidden size; the row count is chosen to
            // make each tensor a few hundred MiB.
            rows: 16384,
            cols: 4096,
            eps: 1e-5,
            iters: 50,
            repeats: 5,
            emit: false,
            flash: false,
            batch: 4,
            seq: 4096,
        }
    }
}

impl Args {
    pub fn copy_elems(&self) -> usize {
        self.copy_mib * 1024 * 1024 / self.dtype.size_bytes()
    }

    pub fn usage() -> String {
        let d = Args::default();
        format!(
            "lex-bench -- P0 bandwidth harness

usage: lex-bench [options]

  --dtype <f32|f16>   element type                     [{}]
  --mib <n>           copy buffer size per side, MiB   [{}]
  --rows <n>          rmsnorm rows                     [{}]
  --cols <n>          rmsnorm cols (multiple of 4)     [{}]
  --eps <f>           rmsnorm epsilon                  [{:e}]
  --iters <n>         dispatches per timed batch       [{}]
  --repeats <n>       timed batches, best is kept      [{}]
  --emit              print the generated MSL and exit
  --flash             flash-attention decode instead (Llama-3-8B shape)
  --batch <n>         flash: sequences                  [{}]
  --seq <n>           flash: cached positions per seq   [{}]
  -h, --help          this text
",
            d.dtype.suffix(),
            d.copy_mib,
            d.rows,
            d.cols,
            d.eps,
            d.iters,
            d.repeats,
            d.batch,
            d.seq,
        )
    }

    /// Returns `Ok(None)` when the caller asked for help and should exit.
    pub fn parse<I: Iterator<Item = String>>(mut it: I) -> Result<Option<Args>, String> {
        let mut a = Args::default();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print!("{}", Args::usage());
                    return Ok(None);
                }
                "--emit" => a.emit = true,
                "--flash" => a.flash = true,
                "--batch" => a.batch = num(&mut it, "--batch")?,
                "--seq" => a.seq = num(&mut it, "--seq")?,
                "--dtype" => {
                    a.dtype = match value(&mut it, "--dtype")?.as_str() {
                        "f32" => DType::F32,
                        "f16" => DType::F16,
                        other => {
                            return Err(format!("unknown dtype `{other}`, expected f32 or f16"));
                        }
                    }
                }
                "--mib" => a.copy_mib = num(&mut it, "--mib")?,
                "--rows" => a.rows = num(&mut it, "--rows")?,
                "--cols" => a.cols = num(&mut it, "--cols")?,
                "--iters" => a.iters = num(&mut it, "--iters")?,
                "--repeats" => a.repeats = num(&mut it, "--repeats")?,
                "--eps" => {
                    let raw = value(&mut it, "--eps")?;
                    a.eps = raw
                        .parse()
                        .map_err(|_| format!("--eps wants a float, got `{raw}`"))?;
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }

        // Catch the mistakes that would otherwise show up as a division by zero
        // or an empty benchmark rather than as a message.
        if a.copy_mib == 0 {
            return Err("--mib must be at least 1".into());
        }
        if a.rows == 0 || a.cols == 0 {
            return Err("--rows and --cols must be at least 1".into());
        }
        if a.batch == 0 || a.seq == 0 || !a.seq.is_multiple_of(16) {
            return Err("--batch must be at least 1 and --seq a positive multiple of 16".into());
        }
        if a.iters == 0 || a.repeats == 0 {
            return Err("--iters and --repeats must be at least 1".into());
        }
        Ok(Some(a))
    }
}

fn value<I: Iterator<Item = String>>(it: &mut I, flag: &str) -> Result<String, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn num<I: Iterator<Item = String>>(it: &mut I, flag: &str) -> Result<usize, String> {
    let raw = value(it, flag)?;
    raw.parse()
        .map_err(|_| format!("{flag} wants a whole number, got `{raw}`"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Args>, String> {
        Args::parse(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_describe_a_llama_sized_rmsnorm() {
        let a = parse(&[]).unwrap().unwrap();
        assert_eq!(a.cols, 4096);
        assert_eq!(a.dtype, DType::F32);
        assert_eq!(a.copy_elems(), 256 * 1024 * 1024 / 4);
    }

    #[test]
    fn dtype_changes_the_copy_element_count_not_the_byte_count() {
        let f32s = parse(&["--dtype", "f32"]).unwrap().unwrap();
        let f16s = parse(&["--dtype", "f16"]).unwrap().unwrap();
        assert_eq!(f16s.copy_elems(), 2 * f32s.copy_elems());
        assert_eq!(
            f16s.copy_elems() * 2,
            f32s.copy_elems() * 4,
            "both dtypes must move the same number of bytes, or the ceiling is not comparable"
        );
    }

    #[test]
    fn bad_values_are_rejected_with_a_message() {
        assert!(parse(&["--dtype", "bf16"]).is_err());
        assert!(parse(&["--rows"]).is_err());
        assert!(parse(&["--iters", "0"]).is_err());
        assert!(parse(&["--mib", "0"]).is_err());
        assert!(parse(&["--nope"]).is_err());
    }
}
