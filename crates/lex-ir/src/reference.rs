//! CPU reference implementations.
//!
//! Every backend is scored against these. Integer paths will be required to
//! match bit-for-bit; float paths match within a declared tolerance, because a
//! GPU tree reduction and a CPU sequential one legitimately disagree in the
//! last bits. Stating the tolerance is the point — an unstated one is how
//! numerical bugs hide for months.

use half::f16;

/// Relative tolerance a backend must meet against the reference, per dtype.
///
/// f32 is loose enough to absorb a different reduction order over 4096 terms.
/// f16 is dominated by the storage format itself: 2^-11 is one ulp of the
/// mantissa, and the accumulate-in-f32 / store-in-f16 path costs a few of them.
pub const TOL_F32: f32 = 1e-5;
pub const TOL_F16: f32 = 2e-3;

pub fn copy_f32(x: &[f32], y: &mut [f32]) {
    y.copy_from_slice(x);
}

pub fn copy_f16(x: &[f16], y: &mut [f16]) {
    y.copy_from_slice(x);
}

/// `y[r,c] = x[r,c] * w[c] * rsqrt(mean(x[r,:]^2) + eps)`
pub fn rmsnorm_f32(x: &[f32], w: &[f32], y: &mut [f32], rows: usize, cols: usize, eps: f32) {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(y.len(), rows * cols);
    assert_eq!(w.len(), cols);
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for &v in row {
            acc += v * v;
        }
        let scale = (acc / cols as f32 + eps).sqrt().recip();
        let out = &mut y[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out[c] = row[c] * w[c] * scale;
        }
    }
}

/// Same, for f16 storage. Accumulation and the scale are computed in f32,
/// which is what the emitted kernel does; only the store narrows.
pub fn rmsnorm_f16(x: &[f16], w: &[f16], y: &mut [f16], rows: usize, cols: usize, eps: f32) {
    assert_eq!(x.len(), rows * cols);
    assert_eq!(y.len(), rows * cols);
    assert_eq!(w.len(), cols);
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mut acc = 0.0f32;
        for &v in row {
            let v = v.to_f32();
            acc += v * v;
        }
        let scale = (acc / cols as f32 + eps).sqrt().recip();
        let out = &mut y[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out[c] = f16::from_f32(row[c].to_f32() * w[c].to_f32() * scale);
        }
    }
}

/// Worst relative error between two slices, with an absolute floor so that
/// values near zero do not report infinite relative error.
pub fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let mut worst = 0.0f32;
    for (&g, &w) in got.iter().zip(want) {
        if !g.is_finite() || !w.is_finite() {
            return f32::INFINITY;
        }
        let err = (g - w).abs() / w.abs().max(1e-3);
        if err > worst {
            worst = err;
        }
    }
    worst
}

/// A cheap deterministic filler. Not random — reproducible failures matter more
/// than statistical realism at this stage, and the values are centred so that
/// the rms scale lands near 1.
pub fn fill_pattern_f32(buf: &mut [f32], seed: u32) {
    let mut s = seed | 1;
    for v in buf.iter_mut() {
        // xorshift32, mapped to roughly [-1, 1)
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        *v = (s >> 8) as f32 / (1u32 << 23) as f32 - 1.0;
    }
}

pub fn fill_pattern_f16(buf: &mut [f16], seed: u32) {
    let mut tmp = vec![0.0f32; buf.len()];
    fill_pattern_f32(&mut tmp, seed);
    for (d, s) in buf.iter_mut().zip(&tmp) {
        *d = f16::from_f32(*s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_of_all_ones_is_all_ones() {
        let (rows, cols) = (3, 8);
        let x = vec![1.0f32; rows * cols];
        let w = vec![1.0f32; cols];
        let mut y = vec![0.0f32; rows * cols];
        rmsnorm_f32(&x, &w, &mut y, rows, cols, 0.0);
        for v in y {
            assert!((v - 1.0).abs() < 1e-6, "got {v}");
        }
    }

    #[test]
    fn rmsnorm_is_scale_invariant_up_to_eps() {
        let (rows, cols) = (2, 16);
        let mut x = vec![0.0f32; rows * cols];
        fill_pattern_f32(&mut x, 7);
        let w = vec![1.0f32; cols];

        let mut y1 = vec![0.0f32; rows * cols];
        rmsnorm_f32(&x, &w, &mut y1, rows, cols, 0.0);

        let x10: Vec<f32> = x.iter().map(|v| v * 10.0).collect();
        let mut y2 = vec![0.0f32; rows * cols];
        rmsnorm_f32(&x10, &w, &mut y2, rows, cols, 0.0);

        assert!(max_rel_err(&y2, &y1) < 1e-5);
    }

    #[test]
    fn f16_path_tracks_the_f32_path() {
        let (rows, cols) = (2, 64);
        let mut x32 = vec![0.0f32; rows * cols];
        fill_pattern_f32(&mut x32, 11);
        let w32 = vec![1.0f32; cols];
        let mut y32 = vec![0.0f32; rows * cols];
        rmsnorm_f32(&x32, &w32, &mut y32, rows, cols, 1e-5);

        let x16: Vec<f16> = x32.iter().map(|v| f16::from_f32(*v)).collect();
        let w16: Vec<f16> = w32.iter().map(|v| f16::from_f32(*v)).collect();
        let mut y16 = vec![f16::ZERO; rows * cols];
        rmsnorm_f16(&x16, &w16, &mut y16, rows, cols, 1e-5);

        let got: Vec<f32> = y16.iter().map(|v| v.to_f32()).collect();
        let err = max_rel_err(&got, &y32);
        assert!(err < TOL_F16, "f16 reference drifted from f32 by {err}");
    }

    #[test]
    fn pattern_is_deterministic_and_bounded() {
        let mut a = vec![0.0f32; 256];
        let mut b = vec![0.0f32; 256];
        fill_pattern_f32(&mut a, 3);
        fill_pattern_f32(&mut b, 3);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| (-1.0..1.0).contains(v)));
    }
}
