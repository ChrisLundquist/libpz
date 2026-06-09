# BWT + Context-Mixing Range Coder — Spike Findings (deferred)

**Date:** 2026-06-09
**Status:** ⚠️ Deferred (not a dead end). Code parked, not on master.
**Code:** branch `spike/bwt-cm`, tag `spike/bwt-cm-2026-06-09` (SHA `112e438`).
Retrieve with `git show spike/bwt-cm:src/bwt_cm.rs` (+ `examples/bwt_cm_eval.rs`).

## What it tested

The [future-pipelines roadmap](../design-plans/2026-06-09-future-pipelines-roadmap.md)'s
**#1 bet**: replace `pz bw`'s `MTF → zrle → FSE` entropy tail with a bzip3-style
**order-1 context-mixing binary range coder**. The thesis: `pz bw` is already
block-parallel and beats zstd-9 on ratio, so a CM tail could make it a ~25-27%
codec that still decodes block-parallel — a point in a tier (xz/bzip3/ppmd/zpaq)
that otherwise decodes single-threaded. And building the range coder once yields the
substrate for the whole high-ratio family (LZMA-class, full CM).

## What was built

A standalone `src/bwt_cm.rs` (872 lines, `pub mod bwt_cm`, 8 unit tests):

- **Carry-safe binary range coder** — `RangeEncoder`/`RangeDecoder`, 33-bit `low` +
  cache/`0xFF`-run carry chain (LZMA-style). Correct on the first full implementation;
  **360 fuzz + adversarial round-trips clean**, including 1 MiB all-`0xFF` carry chains
  and real BWT output. This is the reusable asset.
- **Order-1 context-mixing bit model** — each byte coded MSB-first through a 255-node
  bit-tree of per-`(context, node)` 12-bit predictors, context = previous byte. Four
  operating points: **full-CM** (o0+o1+o2 combined by a per-prev-byte adaptive logistic
  mixer), **MID** (o0+o1 logistic), **BLEND** (o1 confidence-weighted against o0 in the
  linear domain, no mixer), **FAST** (direct o1, no mixing).
- An apples-to-apples eval harness (`examples/bwt_cm_eval.rs`) comparing the CM-coded
  size of the **raw BWT byte stream** against the current `MTF → zrle → FSE` tail on the
  same `bwt::encode` output.

## Results (post-BWT byte stream: CM size / current-tail size; lower is better)

| File | CM/tail ratio | decode (MB/s/core) | note |
|---|---|---|---|
| dickens (text) | **0.892** @1 MiB block / 0.906 @512 KB | ~11 | the ratio gate (≤0.90) passes at ≥1 MiB, *misses* at the 512 KB default |
| dickens @2 MiB | 0.882 | ~11 | ratio is block-size-monotonic |
| x-ray (binary) | **0.831** (43.1% absolute) | ~11 | **−17%** — kills the byte-255 `zrle` fallback cliff |
| silesia mid-slice | 0.882 | ~11 | |

**Pareto frontier (dickens, 1 MiB, ratio @ decode MB/s/core):**
full-CM `0.892 @ 11` · MID `0.903 @ 14` · BLEND `0.964 @ 25` · FAST `0.972 @ 40`.

## Verdict: deferred, CPU-only if ever pursued

The ratio win is **real and large** (text BWT stream −9 to −12%, x-ray −17%), but it
fails the spike's gate (`ratio ≤ 0.90 AND decode ≥ 20 MB/s/core`) for three reasons:

1. **The per-core decode wall is structural (~11 MB/s).** A per-bit adaptive range
   coder is more latency-bound than rANS, and **the logistic mixer is the wall, not the
   cache**: shrinking the order-2 table 64× (8 MiB → 128 KiB) barely moved ratio
   (0.892 → 0.893) *or* speed (10.0 → 11.3 MB/s). Order-2 was near-worthless for ratio.
   The mixer arithmetic (2 stretch lookups + squash + dot + 2 weight updates per bit) is
   a serial dependency chain; removing it (FAST/BLEND) triples decode to 25–40 MB/s but
   **erases ~80% of the ratio edge** (0.89 → 0.96–0.97). There is no cheap middle.
2. **No config clears both gate halves** — ratio-winning configs decode 11–14 MB/s;
   speed-passing configs (≥20) sit at ratio 0.96–0.97.
3. **The numeric pipeline (`Num`, roadmap #3) independently took x-ray/sao** — CM's
   best wins. That narrows CM's remaining justification to *text* BWT streams at
   11 MB/s/core, a much weaker case.

This is **not a dead end** — it's a high-ratio-at-parallel-aggregate play that the
hardware can't price into a default tier. It would only ship as an explicit opt-in
backend, and only after: (a) bumping the BW block size to ≥1 MiB for the CM path so the
ratio gate passes, (b) validating *aggregate* (all-cores) decode vs zstd-9/xz, and
(c) heavy `cargo fuzz`/proptest on the range coder before it touches default-reachable
code (360 round-trips is a strong start, not sufficient for ship).

**CM is GPU-hostile.** Serial per-bit predict→code→update feedback + adaptive RMW
context hash tables = exactly the patterns the repo's dead-ends prove fail on GPU
(serial entropy dependency; atomics lose hash-table recency). The only parallelism is
across independent blocks — which pz already does on the CPU. CM is CPU-only.

## The reusable asset

Regardless of the CM decision, **keep the carry-safe `RangeEncoder`/`RangeDecoder`**
(`spike/bwt-cm:src/bwt_cm.rs`). It is correct and fuzzed, and it is the load-bearing
primitive for every future high-ratio (range-coded) family — LZMA-class context
literals, brotli-style modeling, or a full ICM-ISSE CM chain. The next high-ratio
spike should start from this coder rather than rebuild it.

## Incidental correction

`pz bw`/`bbw` use `DEFAULT_BW_BLOCK_SIZE = 512 * 1024` (`src/pipeline/mod.rs:143`), **not
1 MiB** as some session notes claimed. This matters for any adaptive-model entropy tail:
ratio amortizes model warm-up over the block, so it improves with larger blocks (the CM
gate literally flips between 512 KB and 1 MiB).
