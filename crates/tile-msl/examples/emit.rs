//! Print the MSL a Llama layer kernel lowers to.
//!
//! cargo run -p tile-msl --example emit -- matvec q4k 4096 14336
//! cargo run -p tile-msl --example emit -- matvec q6k|q8 <n_in> <n_out>
//! cargo run -p tile-msl --example emit -- rmsnorm 4096

use tile_front::llama::{QLayout, matvec_q, rmsnorm};
use tile_ir::Target;
use tile_msl::program::lower;

fn main() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let num = |i: usize| -> Result<usize, String> {
        a.get(i)
            .ok_or("missing size")?
            .parse()
            .map_err(|_| "bad size".to_string())
    };
    let prog = match a.first().map(String::as_str) {
        Some("matvec") => {
            let layout = match a.get(1).map(String::as_str) {
                Some("q4k") => QLayout::Q4_K,
                Some("q6k") => QLayout::Q6_K,
                Some("q8") => QLayout::Q8_0,
                _ => return Err("layout: q4k, q6k or q8".into()),
            };
            let (n_in, n_out) = (num(2)?, num(3)?);
            matvec_q(n_in, n_out, 8, n_in, layout, false)?
        }
        Some("rmsnorm") => rmsnorm(num(1)?, 1e-5),
        _ => {
            return Err("usage: emit matvec <q4k|q6k|q8> <n_in> <n_out> | emit rmsnorm <n>".into());
        }
    };
    tile_front::check(&prog, &Target::apple_m_series()).map_err(|e| format!("{e:?}"))?;
    print!("{}", lower(&prog, &Target::apple_m_series(), 256)?.source);
    Ok(())
}
