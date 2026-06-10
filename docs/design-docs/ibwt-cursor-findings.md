# iBWT cursor-chase spike findings (gpu-ibwt, candidate #5)

> **Status (2026-06-10): the CPU K=8 rider is SHIPPED** for the `bw` pipeline
> (versioned block header behind `BW_CURSORS_FLAG`; encode-time samples from
> the suffix array; legacy streams decode via the serial fallback). Measured
> on the shipped CLI: bw blob decode 2.87→1.85 s single-thread (1.55x) and
> 0.41→0.155 s all-cores (2.66x), at +28 B per 1 MiB block (+0.0027% raw).
> The whole-pipeline numbers are lower than the iBWT-stage-only projections
> below because FSE/MTF/zRLE decode stages dilute the win. `bbw` (per-factor
> sampling / factor interleaving) is an open follow-up. The spike harness
> stays on the spike branch `claude/spike-ibwt-cursors`.

**Date:** 2026-06-10 · **Branch:** `claude/spike-ibwt-cursors` · **Harness:** `examples/ibwt_cursor_spike.rs` (spike branch only)
**Hardware:** Apple M5 Max (6P+12E, 18 threads, 128 GB unified), Metal backend via wgpu 27.
**Corpus:** 96 real 1 MiB BWT blocks from Silesia — dickens (9), webster (39), mozilla (48).
All configurations verified round-trip against the original bytes.

This closes the question raised in `gpu-path-research.md` §#5 with M5 data instead of
A100 bandwidth extrapolation, and measures the CPU multi-cursor "rider" first.

## TL;DR

- **GPU verdict: PASS on its own kill gate** — at K=1024 cursors/block the WGSL chase
  hits **8.8 GB/s**, 2.48x the best CPU all-cores chase (3.57 GB/s, multi-cursor).
  The bandwidth-scaling prediction (~0.6 GB/s) was wrong by >10x: the M5 GPU genuinely
  hides pointer-chase latency (SLC residency + massive thread-level parallelism).
- **But the GPU win does not survive end-to-end accounting:** the CPU-side LF build
  (5.9 GB/s all-cores) bottlenecks a GPU pipeline to ~2.6 GB/s — only **1.15x** the
  multi-cursor CPU decoder's end-to-end 2.2 GB/s, before counting the FSE/MTF/zRLE
  stages that stay on CPU anyway. The pareto reject in the research report stands.
- **The CPU rider is the real win and should ship:** K=8 interleaved cursors per block
  give **5.5x chase throughput both single-thread (142→788 MB/s) and all-cores
  (582→3188 MB/s)**, end-to-end iBWT (LF build + chase) **2.6x ST / ~4x all-cores**,
  at a wire cost of 28 bytes per 1 MiB block (0.003%).
- **Deriving the cursor start positions at decode time is pointless — measured:** the
  derivation *is* a full serial chase (172 MB/s ST, the exact bottleneck being removed).
  The start samples must be stored on the wire (a bw container format bump).

## Methodology and noise

- `chase single (shipped)` replicates the loop in `src/bwt/mod.rs::decode_to_buf`
  exactly (safe indexing, separate `bwt[]` + `lf[]` arrays).
- `chase multi K` uses a packed LF array — `(bwt_byte << 24) | lf` in one u32, valid for
  n ≤ 2^24 — so each step is one dependent load instead of two, with K interleaved
  cursors each writing a contiguous 1/K output segment backwards. K=1 isolates the
  packing+unchecked-indexing effect from the memory-level-parallelism effect.
- LF-array build and start-derivation are timed as separate variants.
- 7 reps per configuration, variants interleaved within each rep; GPU sweep run twice
  back-to-back. Median + min/max reported.
- **Noise:** this machine is shared with other agents. Early runs (load avg ~13) showed
  spreads up to 115% on single-thread CPU configs; those runs are discarded. The
  canonical run below was taken at load avg ~4–6 with spreads ≤ ~11% on every
  CPU config that drives a conclusion (and ≤ 2.5% on GPU configs except where flagged).
  Medians across all 5 runs were directionally consistent — the rankings never changed,
  only the absolute numbers (busy-machine numbers were up to ~45% lower).

## Part A — CPU multi-cursor iBWT (the rider)

96 blocks × 1 MiB = 101 MB decoded per pass. MB/s = 1e6 bytes/s.

### Single thread

| variant | med MB/s | min | max | spread |
|---|---|---|---|---|
| lf-build (packed) | 418 | 401 | 419 | 4.5% |
| chase single (shipped) | **142** | 139 | 143 | 3.4% |
| chase multi K=1 (packed) | 168 | 163 | 173 | 6.3% |
| chase multi K=2 | 284 | 273 | 288 | 5.3% |
| chase multi K=4 | 481 | 466 | 486 | 4.2% |
| chase multi K=8 | **788** | 765 | 790 | 3.3% |
| chase multi K=16 | 220 | 219 | 221 | 0.5% |
| chase multi K=32 | 271 | 268 | 272 | 1.5% |
| derive-starts K=16 (serial chase) | 172 | 170 | 174 | 2.6% |

### 18 threads (block-parallel, work-stealing — same shape as the shipped decoder)

| variant | med MB/s | min | max | spread |
|---|---|---|---|---|
| lf-build (packed) | 5937 | 5891 | 5961 | 1.2% |
| chase single (shipped) | **582** | 560 | 603 | 7.4% |
| chase multi K=1 (packed) | 854 | 827 | 902 | 8.6% |
| chase multi K=2 | 1313 | 1277 | 1333 | 4.4% |
| chase multi K=4 | 2108 | 1900 | 2112 | 11.1% |
| chase multi K=8 | **3188** | 3122 | 3260 | 4.3% |
| chase multi K=16 | **3555** | 3514 | 3646 | 3.7% |
| chase multi K=32 | 3566 | 3127 | 3646 | 16.2% |
| derive-starts K=16 (serial chase) | 890 | 755 | 925 | 21.7% |

### Reading

- **Interleaved cursors convert load latency into MLP exactly as hypothesized:**
  near-linear scaling K=1→8 single-thread (168→788, 4.7x), and it survives all-cores
  (582→3188/3555) — the chase is latency-bound, not bandwidth-bound, even with 18
  threads × 8 cursors = 144 concurrent chains.
- **The K=16/32 single-thread cliff (788 → 220) is structural, not noise** (0.5%
  spread, reproduced on 5/5 runs). Best hypothesis: 16 × usize cursor state + loop
  bookkeeping exceeds aarch64's 31 GPRs and the cursor array spills to the stack,
  adding a store+load to every dependent step. K=8 stays fully in registers.
  (All-cores K=16 still edges K=8 — 3555 vs 3188 — presumably because E-cores have a
  different MLP/issue profile; not investigated further.)
- **Packing the BWT byte into the LF word matters:** K=1 packed (one load/step) is
  1.18x the shipped two-array loop; it's also what makes each extra cursor cost only
  one outstanding load. No wire impact — the LF array is built at decode time anyway.
- **End-to-end iBWT (LF build + chase), the shippable comparison:**
  - single-thread: shipped 1/(1/418+1/142) = **106 MB/s** → K=8 **273 MB/s** (**2.6x**)
  - 18 threads: shipped **530 MB/s** → K=8 **2076 MB/s**, K=16 **2224 MB/s** (**3.9–4.2x**)
  - At K=8 the single-thread bottleneck flips: LF build (418 MB/s) is now slower than
    the chase (788 MB/s). The next CPU lever is the LF-build pass, not more cursors.

### Wire cost — and why decode-time derivation is dead

Deriving the K start states from the LF array requires knowing the chase state at K
output positions — which is obtained by... running the full serial chase (measured:
172 MB/s ST, i.e. *slower than the thing being replaced*). There is no shortcut: the
state at output position t is defined by t applications of LF from the primary index.
**So the rider requires storing the samples on the wire**, which is a bw container
format change:

- K=8: (K−1) × 4 bytes = **28 B per 1 MiB block = 0.0027%** of raw, ~0.01% of the
  compressed block at bw's ~28% ratio. (The last segment's start *is* the primary
  index, already on the wire.) Entirely negligible; can be version-gated with a
  single-cursor fallback when absent.
- Samples are ≤ 20-bit values for 1 MiB blocks; bit-packing could shave this further,
  not worth the complexity at 28 bytes.

## Part B — GPU cursor-chase kernel (WGSL, Metal)

Packed LF arrays for all 96 blocks uploaded once to a persistent storage buffer
(384 MiB); one thread per cursor; cursor j of block b chases L = n/K steps and writes
its output segment backwards, packing 4 bytes per u32 store. Timed: encoder + dispatch
+ submit + poll-wait, after a warmup dispatch. Output verified for every K.

**Excluded from chase timing, reported separately:**
- one-time LF upload: 43.6 ms for 384 MiB (**9.2 GB/s**)
- CPU LF build: **5.9 GB/s** all-cores (table above) — the GPU does not dodge this cost

| K (cursors/block) | total threads | med MB/s | min | max | spread | wire cost/block |
|---|---|---|---|---|---|---|
| 64 | 6,144 | 6,109 | 6,100 | 6,114 | 0.2% | 256 B (0.024%) |
| 256 | 24,576 | 6,628 | 6,618 | 6,641 | 0.4% | 1 KiB (0.098%) |
| 512 | 49,152 | 7,239 | 7,225 | 7,284 | 0.8% | 2 KiB (0.20%) |
| 1024 | 98,304 | **8,829** | 8,735 | 8,846 | 1.3% | 4 KiB (0.39%) |
| 2048 | 196,608 | 11,322 | 11,300 | 11,346 | 0.4% | 8 KiB (0.78%) |
| 4096 | 393,216 | 15,882 | 15,841 | 15,925 | 0.5% | 16 KiB (1.6%) |
| 8192 | 786,432 | 19,660 | 19,399 | 19,810 | 2.1% | 32 KiB (3.1%) |
| 16384 | 1,572,864 | 39,550 | 26,464* | 39,897 | flagged* | 64 KiB (6.3%) |

\* K=16384 was bimodal across the two sweeps (one rep at 26.5 GB/s, rest at ~39.6;
50% spread) — treat as "roughly 26–40 GB/s", unreliable per the noise rule. Everything
else had ≤ 2.5% spread. A 41-block run reproduced the same curve shifted down slightly
(K=64: 4.3 GB/s — total thread count B×K, not K alone, is what buys throughput).

### Kill criterion

> KILL if GPU chase < 2x best CPU all-cores result (3,566 MB/s ⇒ gate = 7,132 MB/s)

| K | ratio vs best CPU | verdict |
|---|---|---|
| 64 | 1.71x | below gate |
| 256 | 1.86x | below gate |
| 1024 | **2.48x** | **PASS** |

**The GPU path formally PASSES its kill gate at K=1024** (and scales far beyond at
impractical K). The ~0.6 GB/s bandwidth-scaled prediction from the research report was
off by an order of magnitude: random 4-byte loads into a 4 MiB-per-block working set do
not pay 32x cache-line amplification on M5 because the wave-active blocks' LF arrays
sit in the GPU's cache hierarchy, and the GPU scheduler hides the remaining latency
with thousands of in-flight chains — precisely the latency-hiding the CPU's ~8-deep
MLP window cannot reach.

### Why PASS still doesn't make it worth building

The gate measured the chase in isolation; the system does not improve proportionally:

1. **LF build doesn't go away.** A GPU pipeline (CPU LF build 5.9 GB/s → upload
   9.2 GB/s → chase 8.8 GB/s) nets ~2.6 GB/s — **1.15x** the all-CPU rider (2.2 GB/s).
   Closing that requires porting LF build (a 256-bucket counting sort + stable scatter)
   to GPU too, i.e. the full bzip2gpu architecture, plus keeping the FSE/MTF/zRLE
   stages fed — a project, not a spike, for a pipeline whose pareto problems are
   ratio margin and *encode* speed (research report §#5), neither of which this touches.
2. **Wire cost at the passing K is no longer free:** 0.39% of raw ≈ +0.39pp on bw's
   ratio — which erases bw's hairline 27.8% vs zstd-9 27.9% edge, the one axis it wins.
   (The CPU rider's K=8 cost, 0.003%, does not.)
3. The CLI cold-start economics (~260 ms device init vs ~45 ms to decode 101 MB on
   CPU) remain unaddressed, as for every GPU-decode candidate.

**Verdict: GPU path PASSES the spike gate but is rejected on pareto grounds — file it
as "latency-hiding confirmed on M5, available if a persistent-process customer with
GPU-resident data ever materializes."** The M5 datapoint is valuable beyond this
candidate: it falsifies naive bandwidth-scaling for latency-bound pointer-chase
kernels on this hardware (relevant to research-report candidates #1/#3/#6).

## Recommendation

1. **Ship the CPU multi-cursor rider (K=8) for the bw pipeline.** ~5.5x chase, ~2.6x
   ST / ~4x all-cores end-to-end iBWT, for 28 B/block of wire and ~40 lines of decoder.
   Since iBWT dominates bw decode (~1.12 GB/s all-cores today, decode is
   "inverse-BWT-bound" per the block-size findings), this should materially lift the
   whole pipeline's decode wall. Requires a versioned bw container tweak (per-block
   K−1 u32 samples; fall back to single-cursor when absent). Keep K=8, not 16: K=16
   regresses single-thread 3.6x (register spill cliff) for a 11% all-cores gain.
2. **Do not build the GPU iBWT.** Record the latency-hiding datapoint (this doc).
3. Follow-ups if the rider ships: packed-LF single-pass build is the new ST bottleneck
   (418 MB/s — try unrolling/splitting the counting pass); bbw factors need their own
   sampling scheme (factor boundaries already partition the chase, so bbw may get the
   MLP win for free by interleaving *factors* instead of cursors — unmeasured).

## Raw artifacts

- Harness: `examples/ibwt_cursor_spike.rs`
  (`cargo run --release [--no-default-features] --example ibwt_cursor_spike -- cpu|gpu [--cap N]`)
- Canonical run captured 2026-06-10, load avg 4–6; 4 additional noisier runs agreed
  directionally (rankings stable, absolute numbers up to ~45% lower under load 13+).
