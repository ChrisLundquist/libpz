# pz2-G32 stage 2: Metal GPU decode — gate verdicts and the splice wall

**Date:** 2026-06-10. Continuation of `pz2-g32-stage1-findings.md` (stage 1: the
32-lane GDeflate-style literal relayout costs +0.0066pp — format is free).
Stage 2 asks whether a Metal compute decoder of that wire can beat the CPU.
This was the #1-ranked (and only unanimously endorsed) candidate from
`gpu-path-research.md`, and the last surviving GPU-decode candidate for libpz
on Apple silicon.

**Bottom line: gate 1 negative, gate 2 PASS (64.75 GB/s), gate 3 KILL
(end-to-end GPU = 0.21x CPU all-cores at the real operating point; parity at
best — 1.02x — under 16x synthetic saturation). The cooperative splice is
the wall, as the research report's adversarial verifiers predicted, though
the mechanism is occupancy/latency at dup=1 and a shared-bandwidth parity
ceiling at saturation — not a fixed block-count cap. This closes the
GPU-decode question on M5-class hardware with data.**

## Setup

Probe: `examples/pz2_g32_metal.rs` + `kernels/pz2_g32_lit.metal` +
`kernels/pz2_g32_splice.metal`. Full Silesia corpus (12 files, 211.9 MB),
encoded with shipped `-p pz2` settings (2 MiB blocks, window = block,
auto-greedy): 106 blocks (104 LIT_HUFF), 551 literal tiles, 32.6 MB Huffman
literals, 12.7 M sequences. Persistent shared buffers (unified memory, no
copies); GPU timing via GPUStartTime/GPUEndTime over 9 reps of chained
command buffers (holds DVFS clocks steady; spreads ≤0.2% on all GPU
configs). Device init 13.2 ms + kernel compile 0.8 ms reported separately.
Both round-trips byte-exact: literal phase (551 tiles) and end-to-end
(106 blocks, 211.9 MB).

## Gate 1 — NEON decoder of the 32-lane wire (the honest CPU denominator)

Round-based portable + NEON (aarch64) decoders for the G32 literal layout
(`examples/pz2_g32_cpu_simd.rs`). NEON doubles the stage-1 scalar decode of
this wire (sao 553 → 1209 MB/s ST), **but the 32-lane wire is NOT a CPU win
in its own right**: 0.55–0.61x the shipped 8-lane literal phase
single-thread. The shipped 8-lane layout is already the right shape for a
CPU; 32 lanes only pay off when 32 hardware lanes execute them in lockstep.

All-cores CPU numbers for this wire varied with measurement conditions:
5.0 GB/s from the gate-1 probe (`pz2_g32_cpu_simd`, per-block pooled decode
with per-call allocation, loaded machine), 9–10.5 GB/s same-tiles in the
gate-2 session, 16.5 GB/s in the canonical run (spread 60.9%). CPU-side
numbers on this shared machine are indicative; the GPU comparisons below use
the same-session CPU measurements.

## Gate 2 — Metal literal-phase kernel: PASS

`kernels/pz2_g32_lit.metal`: one simdgroup per tile, 32 symbols/round via
`simd_ballot` + popcount-prefix word fetch, 4 KB decode table in threadgroup
memory (8 simdgroups per threadgroup share one table).

| Metric | Value |
|---|---|
| Literal kernel, steady-state | **0.50 ms for 32.6 MB = 64.75 GB/s** (spread 0.1%) |
| vs NEON all-cores, same wire | 3.93x |
| vs 5 GB/s kill line | PASS (13x over) |

The lane-pinned format thesis is fully vindicated at the kernel level: with
the wire designed for it, the M5 GPU decodes Huffman literals at ~65 GB/s —
the dietgpu/GDeflate topology lesson reproduced on Apple silicon in WGSL-free
MSL. The stage-1 prediction that the entropy phase was never the problem
holds.

## Gate 3 — cooperative splice: KILL

`kernels/pz2_g32_splice.metal`: one 32-thread threadgroup per block; phase A
decodes the 3 sequence streams (3 serial Huffman chains per block), phase B
executes the LZ splice (literal copies + match copies in sequence order).

| Metric | Value |
|---|---|
| Splice kernel | **53.43 ms = 3.97 GB/s** (spread 0.1%) |
| — phase A alone (sequence entropy) | 20.24 ms = 38% of splice (5 reps) |
| End-to-end (blit + literal + splice) | **54.01 ms = 3.92 GB/s** |
| CPU all-cores full pz2 decode (same session, spread 17.6%) | 11.59 ms = 18.28 GB/s |
| **GPU end-to-end vs CPU all-cores, single corpus** | **0.21x — KILL** |

(Phase A's "318 concurrent chains" — 3 per block — is an upper bound;
CODES_CONST lanes carry no chain.)

### Occupancy diagnosis: latency-bound at dup=1, parity ceiling at saturation

The 0.21x is NOT a fixed-throughput wall. At 106 threadgroups the splice
kernel runs at ~5% occupancy, latency-bound on each block's serial chain —
phase A and phase B are mostly *idle*, bounded by the longest block, so
subtraction arithmetic between them is invalid (review-session subset runs
showed the splice doing 8x work in +7% time going dup 1→8). The probe's
`--dup N` flag (N independent copies of every block descriptor, distinct
outputs) measures the saturated ceiling on the full corpus:

| Config (full corpus) | Splice | End-to-end | vs CPU all-cores |
|---|---|---|---|
| dup=1 (106 TGs — the real operating point) | 3.97 GB/s | 3.92 GB/s | **0.21x** |
| dup=8 (848 TGs) | 15.95 GB/s (spread 17.8%, flagged) | — | — |
| dup=16 (1,696 TGs, 3.4 GB in flight) | 20.81 GB/s (spread 15.7%, flagged) | **19.65 GB/s** (spread 10.3%) | **1.02x** |

The corrected mechanism: concurrent block sets fill the machine with zero
format change, but the saturated ceiling lands at **parity with the CPU
(1.02x ± ~10%), never above it** — both engines converge on the same
shared-memory-system ceiling (~20 GB/s on this data). The GPU therefore
offers no wall-clock win at any occupancy:

1. **At the real operating point (one stream in flight), 0.21x.** A decode
   call has 106 blocks, not 1,696; nothing in the CLI or library path keeps
   16 corpora in flight.
2. **At full saturation, parity at best** — which still has to pay the
   ~13 ms device init, a Metal backend, the spike→production hardening gap,
   and the (tiny, +0.0066pp) format tax, against a CPU path that needs none
   of it.
3. The honest residual: parity-at-saturation means **core-offload value
   exists in principle** — a persistent process decoding many streams
   concurrently could route them through the GPU at no wall-clock loss while
   freeing 18 CPU cores. That is a product hypothesis (no such embedding
   exists today), not a perf win; it was the research report's pre-stated
   condition for reopening, and it now has a measured ceiling to plan
   against.

Timing-fairness note: the probe's untimed host setup (decode-table
expansion, descriptor build, G32 word prep) would be paid by a real GPU
decoder, while the CPU baseline pays its table builds in-region — the
remaining asymmetries favor the GPU, so 0.21x/1.02x are upper bounds.

### Generalization

This was the #1-ranked and only unanimously-endorsed candidate from
`gpu-path-research.md`; it failed end-to-end against the honest CPU
denominator at every occupancy, consistent with the report's prediction that
GDeflate-class decode would "bracket rather than clear the CPU wall."
Candidates #3–#6 (dietpz rANS lane, num-G, gpu-ibwt, pz2-GA) were each
already Pareto-rejected on their own grounds in that report; nothing
measured here weakens those rejections. The GPU-decode question for libpz
on unified-memory Apple silicon is closed — with data, not extrapolation.

What would reopen it (record for the future, not a recommendation): a
persistent-process embedding decoding many streams concurrently (the
measured parity ceiling makes this a core-offload play, not a speedup); a
*discrete*-GPU machine (own bandwidth pool, CPU comparison pays PCIe); or a
workload where output stays GPU-resident (decode-into-texture/buffer) so the
CPU path pays an upload the GPU path skips. All three change the
denominator, not the kernel.

## Artifacts

- `kernels/pz2_g32_lit.metal`, `kernels/pz2_g32_splice.metal` — the kernels
  (MSL; `SKIP_PHASE_B` compile flag isolates phase A for attribution)
- `examples/pz2_g32_metal.rs` — host probe (`--reps/--inner/--tile-lits/--dup`),
  builds GPU data from real pz2 blocks, NEON + CPU-all-cores baselines,
  byte-exact round-trip verification, DVFS-stable GPU timing
- `examples/pz2_g32_cpu_simd.rs` — gate-1 NEON/portable decoders + probe
- `transcode_g32`/`decode_g32` in `src/pz2.rs` (stage 1, spike-only — no
  wire or default changes shipped)
- Full-corpus logs: dup=1 9 reps/config (phase A 5 reps), GPU spreads ≤0.2%;
  dup=8/16 saturation runs carry 10–18% spreads (flagged inline above); raw
  per-rep times in the probe output

## Verdict ledger

| Gate | Question | Verdict |
|---|---|---|
| 1 | Is the 32-lane wire a CPU win via NEON? | **No** (0.55–0.61x shipped 8-lane ST) |
| 2 | Can Metal decode the literal phase > 5 GB/s? | **PASS — 64.75 GB/s** |
| 3 | Does end-to-end GPU block decode beat CPU all-cores? | **KILL — 0.21x at dup=1; 1.02x (parity) at 16x saturation** |
