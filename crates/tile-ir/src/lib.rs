//! Tile IR: the target-independent description of a kernel, plus the target
//! table it is planned against.
//!
//! P0 scope is deliberately tiny: two ops, one dtype family, no layout algebra
//! and no schedule language. What it does establish is the layering the rest of
//! the compiler depends on:
//!
//! ```text
//!   Kernel  (what to compute — no target in it)
//!      + Target  (the hardware table — data, not code)
//!      = Plan    (launch geometry, memory budget, derived constants)
//!      -> backend emits source from (Kernel, Plan)
//! ```
//!
//! Keeping `Plan` a separate value is the seed of the algorithm/schedule split:
//! in P3 the planner stops hardcoding tile sizes and starts searching them, and
//! nothing above or below it has to change shape.

pub mod reference;

use std::fmt;

/// Errors raised while planning or emitting a kernel. All of them are things a
/// real compiler should catch before any code runs on a device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileError {
    /// A shape the current lowering cannot express (e.g. not vector-aligned).
    Shape(String),
    /// The plan exceeds a limit in the target table.
    Budget(String),
}

impl fmt::Display for TileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TileError::Shape(m) => write!(f, "shape error: {m}"),
            TileError::Budget(m) => write!(f, "budget error: {m}"),
        }
    }
}

impl std::error::Error for TileError {}

pub type Result<T> = std::result::Result<T, TileError>;

/// Element type of a tile.
///
/// P0 carries only the two float types needed to measure bandwidth. The quant
/// formats from the design doc (`int4`, `mxfp4`, `nvfp4`) arrive in P2 together
/// with the `Quant` wrapper that holds their group size and scale layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F16,
    F32,
}

impl DType {
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F16 => 2,
            DType::F32 => 4,
        }
    }

    /// Scalar spelling in Metal Shading Language.
    pub const fn msl_scalar(self) -> &'static str {
        match self {
            DType::F16 => "half",
            DType::F32 => "float",
        }
    }

    /// 4-wide vector spelling in Metal Shading Language.
    pub const fn msl_vec4(self) -> &'static str {
        match self {
            DType::F16 => "half4",
            DType::F32 => "float4",
        }
    }

    /// Short name used in kernel symbols and golden-file names.
    pub const fn suffix(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F32 => "f32",
        }
    }
}

/// Where a tile lives. P0 only needs two of these, but the enum is the thing
/// that later makes placement a type-level decision rather than a `T*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    /// Device memory. On Apple silicon this is the same physical memory the CPU
    /// sees; the cost table differs, the type does not.
    Global,
    /// Metal `threadgroup` / CUDA `__shared__` / AMD LDS.
    Threadgroup,
}

/// How a kernel parameter is used. Enough, for now, to emit `const` correctly
/// and to let the bench allocate the right buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    Read,
    Write,
}

/// One buffer argument of a kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferParam {
    pub name: &'static str,
    pub dtype: DType,
    pub space: Space,
    pub access: Access,
    /// Element count, in scalars (not vectors).
    pub elems: usize,
}

/// The hardware table. This is *data*: adding a target means adding a row, not
/// a code path. P0 has one row; the NVIDIA and AMD rows land with P1 so that
/// schedules for them can be type-checked long before anything lowers to them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub name: &'static str,
    /// Lanes that execute in lockstep: 32 on Apple and NVIDIA, 64 on CDNA.
    pub simd_width: usize,
    pub max_threads_per_threadgroup: usize,
    /// Threadgroup / shared / LDS bytes available to one threadgroup.
    pub max_threadgroup_bytes: usize,
    /// True when device and host share physical memory (no weight staging).
    pub unified_memory: bool,
}

impl Target {
    /// Apple M-series GPUs (M1 through M4, all tiers).
    ///
    /// 32 KiB is the threadgroup limit the design doc records, and it is the
    /// binding constraint that forces smaller tiles than Hopper's 228 KB.
    pub const fn apple_m_series() -> Target {
        Target {
            name: "apple-m-series",
            simd_width: 32,
            max_threads_per_threadgroup: 1024,
            max_threadgroup_bytes: 32 * 1024,
            unified_memory: true,
        }
    }
}

/// The operations P0 can express.
#[derive(Clone, Debug, PartialEq)]
pub enum Op {
    /// `y[i] = x[i]`. Not useful on its own — it exists to measure the
    /// bandwidth ceiling that every other kernel is scored against.
    Copy { dtype: DType, n: usize },
    /// `y[r,c] = x[r,c] * w[c] * rsqrt(mean(x[r,:]^2) + eps)`.
    RmsNorm {
        dtype: DType,
        rows: usize,
        cols: usize,
        eps: f32,
    },
}

/// A named kernel: one op, for now.
#[derive(Clone, Debug, PartialEq)]
pub struct Kernel {
    pub name: String,
    pub op: Op,
}

impl Kernel {
    pub fn copy(dtype: DType, n: usize) -> Kernel {
        Kernel {
            name: format!("copy_{}", dtype.suffix()),
            op: Op::Copy { dtype, n },
        }
    }

    pub fn rmsnorm(dtype: DType, rows: usize, cols: usize, eps: f32) -> Kernel {
        Kernel {
            name: format!("rmsnorm_{}", dtype.suffix()),
            op: Op::RmsNorm {
                dtype,
                rows,
                cols,
                eps,
            },
        }
    }

    pub fn dtype(&self) -> DType {
        match self.op {
            Op::Copy { dtype, .. } | Op::RmsNorm { dtype, .. } => dtype,
        }
    }

    /// Buffer arguments, in binding order.
    pub fn params(&self) -> Vec<BufferParam> {
        match self.op {
            Op::Copy { dtype, n } => vec![
                BufferParam {
                    name: "x",
                    dtype,
                    space: Space::Global,
                    access: Access::Read,
                    elems: n,
                },
                BufferParam {
                    name: "y",
                    dtype,
                    space: Space::Global,
                    access: Access::Write,
                    elems: n,
                },
            ],
            Op::RmsNorm {
                dtype, rows, cols, ..
            } => vec![
                BufferParam {
                    name: "x",
                    dtype,
                    space: Space::Global,
                    access: Access::Read,
                    elems: rows * cols,
                },
                BufferParam {
                    name: "w",
                    dtype,
                    space: Space::Global,
                    access: Access::Read,
                    elems: cols,
                },
                BufferParam {
                    name: "y",
                    dtype,
                    space: Space::Global,
                    access: Access::Write,
                    elems: rows * cols,
                },
            ],
        }
    }

    /// Bytes that *must* cross the memory interface for a correct
    /// implementation: each input read once, each output written once.
    ///
    /// This is the denominator of every bandwidth number the bench prints. It
    /// is an ideal, not a measurement — a kernel that re-reads its input pays
    /// real traffic this does not count, which is exactly the gap a later
    /// scheduling pass has to close.
    pub fn ideal_bytes(&self) -> usize {
        self.params()
            .iter()
            .map(|p| p.elems * p.dtype.size_bytes())
            .sum()
    }
}

/// Launch geometry, in Metal's terms (CUDA: blocks and threads-per-block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub threadgroups: [usize; 3],
    pub threads_per_threadgroup: [usize; 3],
}

/// Everything the backend needs that depends on the target rather than on the
/// algorithm. In P0 the planner computes this; in P3 it searches for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub launch: Launch,
    /// Bytes of `Space::Threadgroup` the kernel allocates. Checked against the
    /// target table here, not discovered at pipeline-creation time.
    pub threadgroup_bytes: usize,
    /// Elements per vector access. Fixed at 4 for P0 (16 B for f32, 8 B for
    /// f16); a tunable in P3.
    pub vec_width: usize,
    pub threads_per_tg: usize,
    pub simdgroups_per_tg: usize,
    /// Vector lanes each threadgroup strides over. For `Copy` this is the whole
    /// grid; for `RmsNorm` it is one row.
    pub vec_lanes: usize,
}

/// Largest power of two `<= n`, for `n >= 1`.
fn prev_pow2(n: usize) -> usize {
    debug_assert!(n >= 1);
    1usize << (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// Bind a kernel to a target: choose launch geometry and check it fits.
///
/// Every failure here is a compile-time error rather than a silent slow path,
/// which is the whole point of having a target table in the first place.
pub fn plan(kernel: &Kernel, target: &Target) -> Result<Plan> {
    const VEC: usize = 4;
    // One vec4 per thread for `Copy`, and a 256-thread threadgroup, is not a
    // tuned choice — it is the simplest thing that issues wide enough loads to
    // be worth measuring. P3 replaces both with searched parameters.
    const PREFERRED_TG: usize = 256;

    if target.simd_width == 0 || !target.simd_width.is_power_of_two() {
        return Err(TileError::Budget(format!(
            "target {} has a nonsensical simd width {}",
            target.name, target.simd_width
        )));
    }

    match kernel.op {
        Op::Copy { dtype, n } => {
            if n == 0 || n % VEC != 0 {
                return Err(TileError::Shape(format!(
                    "copy length {n} must be a nonzero multiple of {VEC} ({} elements per vector access)",
                    VEC
                )));
            }
            let _ = dtype;
            let lanes = n / VEC;
            let tg = PREFERRED_TG.min(target.max_threads_per_threadgroup);
            let groups = lanes.div_ceil(tg);
            Ok(Plan {
                launch: Launch {
                    threadgroups: [groups, 1, 1],
                    threads_per_threadgroup: [tg, 1, 1],
                },
                threadgroup_bytes: 0,
                vec_width: VEC,
                threads_per_tg: tg,
                simdgroups_per_tg: tg.div_ceil(target.simd_width),
                vec_lanes: lanes,
            })
        }
        Op::RmsNorm {
            rows, cols, eps, ..
        } => {
            if rows == 0 {
                return Err(TileError::Shape("rmsnorm needs at least one row".into()));
            }
            if cols == 0 || cols % VEC != 0 {
                return Err(TileError::Shape(format!(
                    "rmsnorm cols {cols} must be a nonzero multiple of {VEC}"
                )));
            }
            if !(eps.is_finite() && eps >= 0.0) {
                return Err(TileError::Shape(format!(
                    "rmsnorm eps {eps} must be finite and non-negative"
                )));
            }

            let lanes = cols / VEC;
            // One threadgroup per row. Never more threads than there is work
            // for, never fewer than one simdgroup, never more than the target
            // (or our preference) allows.
            let tg = prev_pow2(lanes).clamp(
                target.simd_width,
                PREFERRED_TG.min(target.max_threads_per_threadgroup),
            );

            if !tg.is_multiple_of(target.simd_width) {
                return Err(TileError::Budget(format!(
                    "threadgroup size {tg} is not a multiple of {}'s simd width {}",
                    target.name, target.simd_width
                )));
            }

            // One f32 partial sum per simdgroup, reduced in a second stage.
            let simdgroups = tg / target.simd_width;
            let tg_bytes = simdgroups * size_of::<f32>();
            if tg_bytes > target.max_threadgroup_bytes {
                return Err(TileError::Budget(format!(
                    "rmsnorm needs {tg_bytes} B of threadgroup memory, {} allows {}",
                    target.name, target.max_threadgroup_bytes
                )));
            }
            // The second reduction stage has simdgroup 0 read one partial per
            // lane, so there must not be more simdgroups than lanes.
            if simdgroups > target.simd_width {
                return Err(TileError::Budget(format!(
                    "{simdgroups} simdgroups exceeds the {}-lane single-stage reduction on {}",
                    target.simd_width, target.name
                )));
            }

            Ok(Plan {
                launch: Launch {
                    threadgroups: [rows, 1, 1],
                    threads_per_threadgroup: [tg, 1, 1],
                },
                threadgroup_bytes: tg_bytes,
                vec_width: VEC,
                threads_per_tg: tg,
                simdgroups_per_tg: simdgroups,
                vec_lanes: lanes,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_plan_covers_every_lane() {
        let t = Target::apple_m_series();
        let k = Kernel::copy(DType::F32, 1024);
        let p = plan(&k, &t).unwrap();
        assert_eq!(p.vec_lanes, 256);
        let threads = p.launch.threadgroups[0] * p.launch.threads_per_threadgroup[0];
        assert!(threads >= p.vec_lanes);
        assert_eq!(p.threadgroup_bytes, 0);
    }

    #[test]
    fn copy_rejects_unaligned_length() {
        let t = Target::apple_m_series();
        assert!(matches!(
            plan(&Kernel::copy(DType::F32, 1023), &t),
            Err(TileError::Shape(_))
        ));
    }

    #[test]
    fn rmsnorm_plan_for_llama3_8b_hidden_size() {
        let t = Target::apple_m_series();
        let k = Kernel::rmsnorm(DType::F32, 4096, 4096, 1e-5);
        let p = plan(&k, &t).unwrap();
        assert_eq!(p.launch.threadgroups, [4096, 1, 1]);
        assert_eq!(p.threads_per_tg, 256);
        assert_eq!(p.simdgroups_per_tg, 8);
        assert_eq!(p.threadgroup_bytes, 32);
        assert_eq!(p.vec_lanes, 1024);
    }

    #[test]
    fn rmsnorm_narrow_rows_do_not_over_allocate_threads() {
        let t = Target::apple_m_series();
        // 128 cols is 32 vec lanes: exactly one simdgroup, not 256 threads.
        let p = plan(&Kernel::rmsnorm(DType::F32, 8, 128, 1e-5), &t).unwrap();
        assert_eq!(p.threads_per_tg, 32);
        assert_eq!(p.simdgroups_per_tg, 1);
    }

    #[test]
    fn rmsnorm_rejects_unaligned_cols() {
        let t = Target::apple_m_series();
        assert!(matches!(
            plan(&Kernel::rmsnorm(DType::F32, 4, 4094, 1e-5), &t),
            Err(TileError::Shape(_))
        ));
    }

    #[test]
    fn threadgroup_budget_is_checked_against_the_target_table() {
        // A target with no threadgroup memory cannot host the two-stage
        // reduction; the planner must say so rather than emit and hope.
        let starved = Target {
            name: "starved",
            max_threadgroup_bytes: 0,
            ..Target::apple_m_series()
        };
        assert!(matches!(
            plan(&Kernel::rmsnorm(DType::F32, 4, 4096, 1e-5), &starved),
            Err(TileError::Budget(_))
        ));
    }

    #[test]
    fn ideal_bytes_counts_each_operand_once() {
        let k = Kernel::rmsnorm(DType::F32, 1000, 4096, 1e-5);
        assert_eq!(k.ideal_bytes(), (2 * 1000 * 4096 + 4096) * 4);
    }
}
