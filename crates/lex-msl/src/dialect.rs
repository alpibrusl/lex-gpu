//! The spellings a GPU language uses for the things every backend needs.
//!
//! The lowering in [`crate::program`] is 1,849 lines, of which about 51
//! mention anything Metal-specific: barriers, lane shuffles, address-space
//! qualifiers, the entry-point signature. Everything else is walking the
//! typed IR and deciding what to emit, which is the same decision on any
//! target.
//!
//! So a second backend copies 1,798 lines to vary 51 — and this repository
//! has already paid for what that costs. The batched matmul kept decoding
//! NVFP4 with a `simd_shuffle` table that the single-row matvec had
//! abandoned two rounds of work earlier, because they were separate arms of
//! the same emitter and only the one being benchmarked got fixed. That cost
//! a 2.7x regression that three rounds of measurement misattributed to
//! arithmetic. Duplicating the whole lowering would make that failure mode
//! structural, across targets, forever.
//!
//! Hence a seam instead. `Dialect` is the part that differs; the lowering is
//! the part that does not. The golden files are the proof that moving code
//! behind this trait changed nothing: they are byte-identical across the
//! refactor, or the refactor is wrong.

/// How a target spells the primitives the lowering emits.
///
/// Every method returns text rather than writing, so a dialect stays a pure
/// description with no opinion about how the emitter buffers its output.
pub trait Dialect {
    /// Wait for every thread in the group, and make threadgroup writes
    /// visible to them.
    fn barrier(&self) -> String;

    /// Read lane `src`'s copy of `value`. `src` need not be uniform: the
    /// NVFP4 decode uses a divergent index to gather from a lane-held table.
    fn shuffle(&self, value: &str, src: &str) -> String;

    /// Read the value held `delta` lanes further up, for a reduction that
    /// halves `delta` each step.
    ///
    /// Metal's `simd_shuffle_down` leaves lanes near the top reading past
    /// the end, which is fine when only lane 0 is used. A dialect whose
    /// primitive differs in that respect has to say so here rather than let
    /// the lowering assume Metal's.
    fn shuffle_down(&self, value: &str, delta: &str) -> String;
}

/// Metal Shading Language.
#[derive(Clone, Copy, Debug, Default)]
pub struct Msl;

impl Dialect for Msl {
    fn barrier(&self) -> String {
        "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string()
    }

    fn shuffle(&self, value: &str, src: &str) -> String {
        format!("simd_shuffle({value}, {src})")
    }

    fn shuffle_down(&self, value: &str, delta: &str) -> String {
        format!("simd_shuffle_down({value}, {delta})")
    }
}

/// CUDA C.
///
/// The mask is `0xffffffff` throughout because the lowering only shuffles
/// where every lane of the warp reaches the call — the reductions run
/// outside divergent control flow, and the NVFP4 gather is uniform in
/// *reaching* the shuffle even when its index is not. A dialect for a
/// lowering that shuffled under divergence would need the mask threaded
/// through, and `__shfl_sync` would deadlock rather than quietly misbehave.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cuda;

impl Dialect for Cuda {
    fn barrier(&self) -> String {
        "__syncthreads();".to_string()
    }

    fn shuffle(&self, value: &str, src: &str) -> String {
        format!("__shfl_sync(0xffffffffu, {value}, {src})")
    }

    fn shuffle_down(&self, value: &str, delta: &str) -> String {
        format!("__shfl_down_sync(0xffffffffu, {value}, {delta})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two dialects must differ in every method. A `Dialect` whose
    /// implementation was copied and not edited would pass every other test
    /// in this repository.
    #[test]
    fn the_dialects_actually_differ() {
        let (m, c) = (Msl, Cuda);
        assert_ne!(m.barrier(), c.barrier());
        assert_ne!(m.shuffle("v", "i"), c.shuffle("v", "i"));
        assert_ne!(m.shuffle_down("v", "d"), c.shuffle_down("v", "d"));
    }

    #[test]
    fn operands_reach_the_emitted_text() {
        assert!(Cuda.shuffle_down("s[0][1]", "d").contains("s[0][1]"));
        assert!(Msl.shuffle("fp4_lane", "code").contains("fp4_lane"));
    }
}
