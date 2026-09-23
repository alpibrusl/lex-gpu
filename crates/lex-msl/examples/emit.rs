//! Print the MSL a Llama layer kernel lowers to.
//!
//! cargo run -p lex-msl --example emit -- matvec q4k 4096 14336
//! cargo run -p lex-msl --example emit -- matvec q6k|q8 <n_in> <n_out>
//! cargo run -p lex-msl --example emit -- rmsnorm 4096
//! cargo run -p lex-msl --example emit -- split <cap> <bps> [direct]   (8B attention shapes)

use lex_front::llama::{QLayout, matvec_q, rmsnorm};
use lex_ir::Target;
use lex_msl::program::lower;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| -> Result<usize, String> {
        a.get(i)
            .ok_or("missing size")?
            .parse()
            .map_err(|_| "bad size".to_string())
    };
    let mut threads = 256;
    let prog = match a.first().map(String::as_str) {
        Some("matvec") => {
            let layout = match a.get(1).map(String::as_str) {
                Some("q4k") => QLayout::Q4_K,
                Some("q6k") => QLayout::Q6_K,
                Some("q8") => QLayout::Q8_0,
                Some("nvfp4") => QLayout::NVFP4,
                _ => return Err("layout: q4k, q6k, q8 or nvfp4".into()),
            };
            let (n_in, n_out) = (num(2)?, num(3)?);
            matvec_q(n_in, n_out, 8, n_in, layout, false)?
        }
        Some("rmsnorm") => rmsnorm(num(1)?, 1e-5),
        Some(k @ ("split" | "combine")) => {
            let (cap, bps) = (num(1)?, num(2)?);
            let f = lex_front::flash::FlashDecode {
                q_rows: 4,
                d: 128,
                seq: cap,
                bq: 4,
                bk: 16,
                stages: 1,
                dtype: lex_ir::DType::F16,
                kv_space: if a.get(3).is_some_and(|x| x == "direct") {
                    lex_ir::Space::Reg
                } else {
                    lex_ir::Space::Threadgroup
                },
                consumers: 0,
                heads: 8,
                kv_cap: cap,
            };
            threads = 128;
            if k == "split" {
                f.build_split(bps)?
            } else {
                f.build_combine(bps)?
            }
        }
        _ => {
            return Err("usage: emit matvec <q4k|q6k|q8> <n_in> <n_out> | rmsnorm <n> | split|combine <cap> <bps> [direct]".into());
        }
    };
    lex_front::check(&prog, &Target::apple_m_series()).map_err(|e| format!("{e:?}"))?;
    print!(
        "{}",
        lower(&prog, &Target::apple_m_series(), threads)?.source
    );
    Ok(())
}
