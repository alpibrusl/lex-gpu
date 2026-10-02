//! The CUDA GEMM's candidate schedules: every one is valid for the shape it
//! is offered for, the default is among the valid ones, no two share a name
//! (the name is in the entry point), and small chunks are not offered tiles
//! that are mostly padding.

use lex_msl::gemm::{Backend, Gemm, cuda_candidates, cuda_default, cuda_valid, gemm_nvfp4_with};

const SHAPES: [(usize, usize, bool); 4] = [
    (17408, 5120, false),
    (5120, 17408, true),
    (10240, 5120, false),
    (5120, 6144, true),
];
const SIZES: [usize; 6] = [512, 256, 128, 64, 32, 16];

#[test]
fn candidates_are_valid_distinct_and_include_the_default() {
    for m in SIZES {
        for (n, k, residual) in SHAPES {
            let g = Gemm {
                m,
                n,
                k,
                residual,
                x_half: true,
            };
            let def = cuda_default(&g);
            assert!(cuda_valid(&g, &def), "default invalid for {g:?}: {def:?}");
            let cands = cuda_candidates(&g);
            let mut names: Vec<String> = cands.iter().map(|s| s.name()).collect();
            names.sort();
            names.dedup();
            assert_eq!(names.len(), cands.len(), "duplicate names for {g:?}");
            for s in &cands {
                assert!(cuda_valid(&g, s), "{s:?} offered for {g:?}");
                // It lowers, and its entry point names it.
                let l = gemm_nvfp4_with(&g, Backend::Cuda, Some(*s)).expect("lowers");
                assert!(l.entry.ends_with(&format!("_c{}", s.name())), "{}", l.entry);
                assert!(l.threadgroup_bytes <= 48 * 1024);
            }
            // Under 128 tokens a 128-row tile is mostly padding; at 128 and
            // up the default itself is a candidate.
            if m < 64 {
                assert!(cands.iter().all(|s| s.bm <= 32 || s.bm < 2 * m), "{m}");
            }
            if m >= 128 {
                assert!(cands.contains(&def), "default not among candidates at {m}");
            }
            assert_eq!(
                cands.len() >= 2,
                m >= 64,
                "{m} tokens: {} candidates",
                cands.len()
            );
        }
    }
}
