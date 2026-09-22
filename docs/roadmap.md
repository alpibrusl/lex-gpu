# Roadmap

The design document ([`design.md`](design.md)) proposes M0–M7 over ~55 weeks.
This is the same destination with a different order, adjusted for two facts:
the hardware on the desk is a Mac Studio, and the work is solo.

## The constraint, and what it does and does not decide

Metal is the daily driver. That is settled by the hardware, and it is a
genuinely good development target: compile times in seconds, no cloud loop,
Xcode GPU capture available locally, and unified memory means no host/device
transfer bugs while the type system is still moving.

The risk it creates is different from the risk of *running* on Metal: it is
that the abstraction ends up Metal-shaped. Metal has no async copy engine, no
cluster level, 32 KB threadgroups and forgiving bandwidth-bound physics. An
abstraction validated only against it will not survive contact with TMA,
mbarrier and warp specialisation.

The mitigation is cheap and is a P1 deliverable: **write the NVIDIA target row
and the Hopper schedules for the P1 kernels, and make the frontend type-check
them, with nothing lowering them.** If the hierarchy algebra cannot express a
warp-specialised producer/consumer pipeline over TMA with 3-stage mbarriers,
that is discoverable in a text file in week 7, not on rented hardware in month
8.

The second mitigation is a rented H100 for about six hours, twice (P4, and
again during P5). That is roughly $50, versus four months of rework.

## Phases

| Phase | Weeks | Deliverable | Exit test |
| --- | --- | --- | --- |
| **P0** Spine | 2 | Workspace, tile IR, target table, planner, MSL emitter, device glue, harness | RMSNorm at ≥ 90% of the copy ceiling on the Studio, matching the reference |
| **P1** Types | 5 | Tile/Layout/Space, linearity, effects, CPU reference interpreter, NVIDIA target row | Flash-attention **decode** inner loop — K reuse across query blocks, double-buffered — type-checks and matches PyTorch in the interpreter. Its Hopper schedule type-checks too. |
| **P2** Metal, correct | 7 | Full op set in MSL, int4 dequant, paged KV, safetensors, sampling. No fusion, no tuning. | Llama-3-8B int4 emits correct tokens on the Studio, at any speed |
| **P3** Metal, fast | 8 | Schedule language does something: simdgroup atoms, threadgroup pipelining, graph fusion | ≥ 70% of llama.cpp Metal **decode** *and* ≥ 70% of its **prefill** |
| **P4** NVIDIA spike | 2 | One kernel, tile → PTX → H100, TMA + wgmma. Slow is fine. | It runs. Re-plan from what it teaches. |
| **P5+** NVIDIA | — | Re-planned against P4's findings | — |

## Differences from M0–M7, and why

**P0 replaces M0.** "Type system on paper, peer review, no code" is the wrong
first move solo. The plumbing — compile pipeline, buffer binding, readback, a
test harness anyone trusts — is what actually kills these projects. Get one
boring kernel end to end first, then design types against a toolchain that
already works.

**P1's exit test is the linearity question, moved forward ~25 weeks.** The
design doc rates "linearity makes common kernels awkward" as a medium risk to
be validated on five hand-typed kernels at M0. Hand-typing always looks fine.
Attention reuses K across query blocks, GEMM reuses A fragments across the N
loop, double-buffering keeps tiles live across iterations, and a persistent
megakernel deliberately reuses one threadgroup arena across phases with
different types. Affine tiles plus `dup` plus borrows is a borrow checker, in
loop-heavy code, which is where borrow checkers hurt. An interpreter running a
real flash-attention decode loop is the only thing that answers it — and the
answer is wanted while it still costs a type-system revision rather than three
backend rewrites.

**P3 adds a prefill number.** Batch-1 decode on unified memory is pure
bandwidth: ~4 GB of weights and ~16 GFLOP per token for an 8B model. A decent
weight layout reaches 70% of llama.cpp without exercising the MMA atom
abstraction at all. Prefill is MMA-bound, and it is the only thing on Metal
that tests `Frag[atom]` — the single abstraction the three-target story rests
on.

**`Cluster` is dropped from the v1 level set.** One target has it, it is the
least-proven lowering, and it costs abstraction budget. It goes back in when
there is real Hopper hardware to test it against.

## Open questions, tracked

- Ragged batch in the type or only at the graph boundary. Decide in P2, when
  paged KV forces the issue.
- Persistent megakernel vs. kernel sequence on Metal. Measure in P3; Metal's
  command-buffer overhead is low enough that fusion may not pay.
- Whether the schedule language is user-facing. Current lean: yes. The tuning
  cache will never cover the shapes that matter for next year's model, and the
  author needs the language anyway.
- The megakernel's dynamic threadgroup arena is in tension with a type-level
  static budget sum. Needs a region or scope concept, not just a sum. P3.
