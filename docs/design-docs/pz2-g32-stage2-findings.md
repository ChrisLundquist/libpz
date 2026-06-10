# pz2-G32 stage 2: Metal GPU decode — gate verdicts and the splice wall

**Date:** 2026-06-10. Continuation of `pz2-g32-stage1-findings.md` (stage 1: the
32-lane GDeflate-style literal relayout costs +0.0066pp — format is free).
Stage 2 asks whether a Metal compute decoder of that wire can beat the CPU.
This was the #1-ranked (and only unanimously endorsed) candidate from
`gpu-path-research.md`, and the last surviving GPU-decode candidate for libpz
on Apple silicon.

**Bottom line: gate 1 negative, gate 2 PASS (64.75 GB/s), gate 3 KILL
(end-to-end GPU = 0.21x CPU all-cores). The cooperative splice is the wall,
exactly as the research report's adversarial verifiers predicted. This closes
the GPU-decode question on M5-class hardware with data.**

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

All-cores NEON on the same tiles measured 5.0 GB/s (initial run, loaded
machine) to 16.5 GB/s (canonical run, spread 60.9% — the CPU-side numbers
were taken on a shared machine and are indicative; the GPU comparisons below
use the same-session CPU measurements).

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
| — phase A alone (sequence entropy) | 20.24 ms = 38% of splice |
| End-to-end (blit + literal + splice) | **54.01 ms = 3.92 GB/s** |
| CPU all-cores full pz2 decode (same session) | 11.59 ms = 18.28 GB/s |
| **GPU end-to-end vs CPU all-cores** | **0.21x — KILL** |

### Why the splice is structural, not an optimization target

The arithmetic that closes the question:

1. **Parallelism collapses at the splice.** The literal phase exposes
   551 tiles × 32 lanes ≈ 17,600 independent decode positions; the splice
   exposes 106 blocks × 1 serial copy chain. Phase A is 318 concurrent
   Huffman chains (3 per block); phase B is 106 sequential splice walks.
   The GPU's only lever — occupancy — is capped by the format's block count.
2. **Even a free phase A doesn't save it.** Subtracting all 20.2 ms of
   sequence-entropy cost leaves 33.2 ms of copy loop — still 2.9x slower
   than the *entire* CPU decode. The in-block LZ copy chain (each copy may
   read bytes written by the previous one) is serially dependent by the
   nature of LZ; 32 threads per block mostly wait on it.
3. **Raising splice parallelism requires changing the format**, either
   smaller blocks (ratio loss — stage 1's whole point was keeping the 2 MiB
   window) or intra-block splice checkpoints — which is exactly the pz2-GA
   candidate (#6) that the research already Pareto-rejected: pz2's own
   iteration history measured the separated-phases design at 1.7–1.9x vs
   the fused 3.1–3.6x, and checkpoint wire costs erase the ratio position.

### Generalization

Per the research report's stated value of this spike: the splice result
generalizes to candidates #3–#6 (dietpz rANS lane, num-G, gpu-ibwt, pz2-GA)
— every one of them either feeds this same splice or was already
Pareto-rejected on the bandwidth-sharing argument this measurement confirms.
**On unified-memory Apple silicon, where 18 CPU cores already convert the
shared ~0.5 TB/s into 18+ GB/s of pz2 decode, a GPU block decoder loses
end-to-end even with a 65 GB/s entropy kernel.** The GPU-decode question for
libpz on this hardware class is closed — with data, not extrapolation.

What would reopen it (record for the future, not a recommendation):
a persistent-process embedding on a *discrete*-GPU machine (where the GPU has
its own bandwidth pool and the CPU comparison is over PCIe), or a workload
where output stays GPU-resident (decode-into-texture/buffer for rendering or
GPU analytics) so the CPU path would have to pay an upload the GPU path
skips. Both change the denominator, not the kernel.

## Artifacts

- `kernels/pz2_g32_lit.metal`, `kernels/pz2_g32_splice.metal` — the kernels
  (MSL; `SKIP_PHASE_B` compile flag isolates phase A for attribution)
- `examples/pz2_g32_metal.rs` — host probe (`--reps/--inner/--tile-lits/--dup`),
  builds GPU data from real pz2 blocks, NEON + CPU-all-cores baselines,
  byte-exact round-trip verification, DVFS-stable GPU timing
- `examples/pz2_g32_cpu_simd.rs` — gate-1 NEON/portable decoders + probe
- `transcode_g32`/`decode_g32` in `src/pz2.rs` (stage 1, spike-only — no
  wire or default changes shipped)
- Full-corpus log: 9 reps/config, all spreads ≤0.2% (GPU); raw per-rep times
  in the probe output

## Verdict ledger

| Gate | Question | Verdict |
|---|---|---|
| 1 | Is the 32-lane wire a CPU win via NEON? | **No** (0.55–0.61x shipped 8-lane ST) |
| 2 | Can Metal decode the literal phase > 5 GB/s? | **PASS — 64.75 GB/s** |
| 3 | Does end-to-end GPU block decode beat CPU all-cores? | **KILL — 0.21x** |
