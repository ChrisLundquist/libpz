# GPU Compression Experiments — Wave 2

**Status:** Proposed
**Last updated:** 2026-03-05
**Depends on:** [gpu-strategy.md](gpu-strategy.md), [experiments.md](experiments.md)

## Purpose

Four experiments designed to answer fundamental architectural questions about GPU compression. Unlike wave 1 (which optimized existing algorithms), these experiments each isolate a single variable to produce clean measurements. **The measurements matter more than the compression ratios.** A clean negative result that tells us *why* something doesn't work is more valuable than a sloppy implementation that compresses well.

Each experiment is named by a letter (C/D/E/F) and a short pipeline name usable via `pz -p <name>`.

---

## Experiment D: Bit-Plane Decomposition (`bitplane`)

### Question answered

**What is the GPU throughput ceiling for compression workloads?**

### Why this matters

This pipeline has zero serial stages, zero data-dependent branching, zero cross-position dependencies. Every operation is embarrassingly parallel. The throughput number it produces is the upper bound — every other pipeline's throughput should be measured against it. If another pipeline achieves 60% of bitplane's throughput, we know 40% is lost to serial dependencies and data-dependent work, not to dispatch overhead or transfer costs.

### What it teaches us

1. **The transfer/dispatch floor.** If bitplane throughput is only 2x faster than the BW pipeline despite having no serial stages, then dispatch overhead and PCIe transfer dominate. This would mean algorithmic optimization is low-leverage — we should focus on reducing round-trips instead.

2. **Where GPU parallelism is naturally strong.** On structured numeric data (integers, floats, fixed-width records), bit-plane decomposition captures redundancy that BWT and LZ miss entirely. On text, it will be terrible. The gap between "best file" and "worst file" quantifies how much a content-adaptive pipeline selector would be worth.

3. **Baseline for GPU memory bandwidth.** The bit transpose is a pure memory-bandwidth operation (read N bytes, write N bytes rearranged). This measures our effective GPU memory bandwidth utilization, which bounds all other GPU work.

### Pipeline design

```
Input → Bit transpose (N×8 → 8×N) → RLE each binary stream → FSE encode
```

- **Bit transpose:** Matrix transpose where input is N×8 bit matrix, output is 8 streams of N bits. GPU processes in tiles: each workgroup takes a 256-byte tile, each thread reads one byte and scatters bits to 8 shared-memory buffers. Well-studied GPU operation (AOS→SOA bit-level transpose).
- **RLE on binary streams:** Run-length encode sequences of 0s and 1s. Can be GPU-parallel (XOR adjacent bits → prefix sum to number runs → segment boundaries) or CPU (each stream is only N/8 bytes).
- **Entropy coding:** Existing FSE encoder on run-length values. Each of 8 streams independent.

### Key measurements

- Total throughput (MB/s) including GPU transfer — **primary output**
- Per-stage timing: transpose, RLE, entropy code
- Compression ratio per Canterbury file
- Which files compress well and why (e.g., "kennedy.xls has zero-heavy integer fields, bit-planes 4-7 are mostly zero")

### Expected results

- Worse than gzip on text files (bit planes of ASCII text have low redundancy)
- Possibly competitive on binary/numeric files (kennedy.xls, ptt5)
- If throughput is lower than BW pipeline despite no serial stages → bottleneck is dispatch/transfer overhead (important to quantify)

---

## Experiment F: Fixed-Window Sort Transform (`fwst`)

### Question answered

**Does full suffix sorting earn its cost, or do the first few bytes of context capture most of the compression?**

### Why this matters

BWT sorts all suffixes — sort keys can be arbitrarily long. On repetitive data, the GPU radix sort does many doubling passes comparing long keys, causing warp divergence and unpredictable work. A fixed-window sort caps the key length at `w` bytes, guaranteeing exactly `w` radix passes with fully predictable work. This directly measures the value of each additional byte of sort context.

### What it teaches us

1. **The effective context depth of real data.** If `w=8` achieves 95% of full BWT's compression, then most data has only ~8 bytes of useful context. If `w=32` is still noticeably worse, the data has deep repetitive structure requiring full suffix comparison. This varies by data type — the per-file results are the deliverable.

2. **Whether we can speed up BWT by capping sort depth.** If `w=8` is within 3% of full BWT, we can replace our expensive O(n log²n) suffix sort with a fixed-depth O(n·w) radix sort. The speedup would be significant and the compression loss negligible.

3. **Informing context-segmented BWT design.** The effective context depth directly tells us what `k` parameter to use in any future context-segmented BWT. If most data saturates at 8-12 bytes, there's no point building infrastructure for deeper context.

### Pipeline design

Minimal modification to existing BWT pipeline — only the sort changes:

```
Input → Extract w-byte key per position → Radix sort by key → BWT readoff → MTF → RLE → FSE
```

- **Key extraction:** For position `i`, key is `input[i..i+w]`. Pad with zeros near end. Exactly `w` bytes per position for uniform radix sort.
- **Radix sort:** Stop after `w` passes (vs full suffix sort's log²n doubling steps). Break ties by position (stable sort) for deterministic output.
- **Everything downstream unchanged** — same MTF → RLE → FSE as existing BW pipeline.

### Parameters to sweep

`w` values: 2, 4, 6, 8, 10, 12, 16, 24, 32, ∞ (full BWT baseline).

### Key measurements

For each `w` × each Canterbury file:
- Compression ratio
- Radix sort time (should decrease with smaller `w` — verify)
- Total pipeline time
- **The ratio-vs-w curve** — the shape is the main deliverable
- "Context value" metric: `(ratio_at_w - ratio_at_{w+2}) / ratio_at_full_bwt` — where this drops below 1% is the effective context depth

### Expected results

- **Strong result:** `w=8` within 3% of full BWT on text → immediate practical value, we can speed up our sort
- **Informative negative:** `w=32` still noticeably worse on E.coli or ptt5 → these files have deep repetitive structure requiring full suffix comparison. This is useful information about data characteristics, not a failure.

---

## Experiment E: Parallel-Parse LZ (`parlz`)

### Question answered

**What is the exact compression ratio cost of removing serial parsing from LZ?**

### Why this matters

This is the most strategically important experiment. Every LZ implementation has a serial parsing step: decide to take a match, advance past it, decide again. This is the hardest stage to parallelize. Nobody has published the actual compression ratio cost of removing this serial dependency.

If the gap is < 3%, fully GPU-parallel LZ is viable and we should invest in it. If the gap is > 15%, serial parsing is doing essential work and GPU LZ will always need a CPU parsing step (or blocking into independent chunks, which is what LzSeqR already does).

### What it teaches us

1. **Whether fully-parallel GPU LZ is architecturally viable.** This is a go/no-go decision for the project's GPU strategy. A small ratio gap means we can build a competitive all-GPU LZ pipeline. A large gap means we should focus on the hybrid GPU-match-finding + CPU-parsing approach we already have.

2. **The cost of match overlap conflicts.** In serial parsing, position 12 is never evaluated because the parser jumped from position 10 to 16. In parallel, both positions find matches independently and we must resolve conflicts. The number of suppressed matches and the resulting ratio loss directly quantify this cost.

3. **Which data types suffer most from parallel parsing.** Files with dense overlapping match candidates (many nearby positions finding matches at similar offsets) will have larger ratio gaps. Files with sparse, non-overlapping matches will have small gaps. The per-file breakdown tells us where parallel LZ works and where it doesn't.

### Pipeline design

```
Input → GPU match finding (all positions) → Parallel match selection → Conflict resolution → Entropy encode
```

- **Match finding:** Existing GPU match finder. For each position, produce best match `(length, distance)` or "no match."
- **Parallel match selection:** Each position independently picks longest match. One thread per position, no communication.
- **Conflict resolution via forward max-propagation scan:**
  - Create `coverage[p] = p + match_length[p]` for match positions, `coverage[p] = p` otherwise
  - Parallel prefix-max scan: `coverage[p] = max(coverage[p], coverage[p-1])`
  - Position `p` is suppressed if `coverage[p-1] > p`
  - O(n) work, O(log n) depth, standard GPU primitive
- **Classification:** Each position becomes match-start, covered (inside a match), or literal.

**CPU greedy baseline** using same GPU match data, for direct comparison:
- Scan left to right; if match ≥ min_length, take it and advance; else emit literal and advance by 1.

### Key measurements

For each Canterbury file:
- Parallel-parse compressed size
- Greedy-parse compressed size (same matches from GPU)
- **Ratio gap: `(parallel_size - greedy_size) / greedy_size`** — primary deliverable, report as percentage
- Match suppression statistics: how many matches found but suppressed by conflict resolution?
- Coverage fraction: what % of input bytes are covered by matches (parallel vs greedy)?
- Comparison against gzip -6 and zstd -3 for context

### Expected results

- **Strong result (< 5% gap on text):** Fully-parallel LZ is viable — invest in all-GPU pipeline
- **Moderate result (5-15% gap):** Parallel LZ is useful for throughput-sensitive workloads willing to trade ratio
- **Large gap (> 15%):** Serial parsing genuinely matters. Focus on hybrid GPU-match + CPU-parse strategy
- **If max-propagation scan is too aggressive** (suppresses so many matches that ratio is worse than raw FSE on literals): try "longer match wins" tiebreaking instead of "earlier position wins"

---

## Experiment C: Re-Pair Grammar Compression (`repair`)

### Question answered

**Is iterative GPU kernel dispatch viable for compression algorithms?**

### Why this matters

Re-Pair requires many rounds (potentially hundreds), each round being: compute bigram frequencies → find most frequent → replace all occurrences. Every operation within a round is GPU-parallel, but the rounds are sequential. This tests the practical limit of `dispatch_latency × iteration_count`. The answer generalizes to any iterative refinement algorithm on GPU — not just Re-Pair.

### What it teaches us

1. **The GPU dispatch overhead budget.** If per-dispatch overhead is 50μs and Re-Pair needs 200 rounds with 3 dispatches each, that's 30ms of pure overhead before any computation. If compute per round is also 30ms, dispatch overhead is 50% — borderline viable. If compute is 3ms, dispatch overhead is 91% — unviable. The ratio `total_dispatch_overhead / total_compute_time` is the key number.

2. **Whether grammar compression captures redundancy that LZ and BWT miss.** Grammar compression finds hierarchical repetition — repeated phrases built from repeated sub-phrases. This is structurally different from LZ (flat substring matching) or BWT (context clustering). On data with hierarchical structure (JSON, XML, source code), grammar compression may find patterns invisible to other approaches.

3. **Diminishing returns in iterative compression.** The rounds-to-ratio curve shows whether the first 20 rounds do most of the work with the remaining 180 providing marginal improvement. If so, early stopping makes the dispatch overhead problem manageable — we only need 20 rounds × 3 dispatches = 60 dispatches, not 600.

4. **The minimum useful work per GPU dispatch.** The per-dispatch cost in microseconds directly tells us the minimum computation that justifies a GPU kernel launch. This is valuable for all future GPU algorithm design in libpz, not just Re-Pair.

### Pipeline design

Each iteration has 3 GPU-parallel stages:

```
Repeat until no bigram occurs > threshold:
  1. Bigram frequency count (parallel histogram or sort-based counting)
  2. Find maximum (parallel reduction)
  3. Replace occurrences (parallel scan + conflict resolution + stream compaction)
```

- **Alphabet growth:** Each round adds a new symbol. Switch from 2D histogram to sort-based counting when alphabet exceeds ~512 symbols.
- **Batching optimization:** Find top-K most frequent bigrams per round, replace all simultaneously. Reduces round count by K× at cost of more complex conflict resolution.
- **Output:** Dictionary of rules `(new_symbol, left, right)` + FSE-encoded final string.

### Key measurements

For each Canterbury file:
- Final compression ratio (dictionary + compressed string)
- Number of rounds to convergence
- **Per-round timing: compute time vs dispatch overhead** — primary deliverable
- **Rounds-to-ratio curve:** ratio achieved after N rounds (where do diminishing returns kick in?)
- Dictionary size as fraction of total output
- Final alphabet size
- `total_dispatch_overhead / total_compute_time` ratio

### Expected results

- **Viable (dispatch overhead < 30% of wall time):** Iterative GPU algorithms are practical. Opens door to other iterative approaches (EM algorithms, iterative refinement, multi-pass analysis).
- **Marginal (dispatch overhead 30-60%):** Viable only with aggressive batching (top-K per round) to keep round count < 50.
- **Unviable (dispatch overhead > 60%):** Iterative GPU algorithms need fundamentally different dispatch models (persistent kernels, indirect dispatch). Document per-dispatch cost for future reference.
- **Ratio pattern:** If grammar compression achieves distinctly different ratios by file type than BWT and LZ, that motivates content-adaptive pipeline selection even if Re-Pair isn't competitive overall.

---

## Cross-Experiment Synthesis

### The four questions form a complete picture

| Experiment | Question | Informs |
|-----------|----------|---------|
| D (bitplane) | What's the GPU throughput ceiling? | Whether algorithmic optimization or dispatch optimization has more leverage |
| F (fwst) | How deep is data's statistical structure? | BWT sort depth, context-segmented BWT design |
| E (parlz) | Can we remove serial LZ parsing? | Go/no-go for all-GPU LZ pipeline |
| C (repair) | Can we iterate on GPU? | Viability of any multi-pass GPU compression algorithm |

### How results combine

- If **D shows high throughput** but **E shows large ratio gap**: GPU is fast but needs CPU parsing assistance. Invest in hybrid overlap.
- If **D shows low throughput** (close to BW pipeline): dispatch/transfer overhead dominates everything. Fix infrastructure before algorithms.
- If **F shows w=8 is sufficient**: simplify BWT sort, save GPU cycles for other stages.
- If **C shows dispatch overhead < 30%**: iterative GPU algorithms become a viable design pattern for future work.
- If **E shows < 5% gap** AND **D shows high throughput**: combine them into a fully GPU-parallel LZ pipeline. This would be the project's strongest result.

### Implementation priority

1. **D (bitplane)** — Simplest, establishes throughput ceiling. Calibration data for everything else.
2. **F (fwst)** — Small modification to existing BWT. Quick to implement, direct practical value.
3. **E (parlz)** — Most strategically important. Answers whether all-GPU LZ is viable.
4. **C (repair)** — Most complex. Answers GPU dispatch viability. Exploratory.

If time is limited, **D and F alone** provide high-value calibration data. **E** is the most strategically important for the project's direction. **C** is exploratory but produces insights that generalize beyond compression.

---

## Benchmark matrix

All experiments must produce results comparable via a single CSV:

```
file,size,pipeline,params,compressed_size,ratio,throughput_mbs,time_total_ms,time_transfer_ms,time_compute_ms,notes
```

Run all 4 experimental pipelines + all existing pipelines + external tools (gzip -6, zstd -3, zstd -9) on every Canterbury corpus file. The summary table showing ratio per file per pipeline is the primary artifact.

---

## Relationship to existing learnings

These experiments build on insights from wave 1 (documented in [experiments.md](experiments.md)):

- **Hash table failure** (wave 1) → Experiment E uses probe-based matching, not hash tables
- **GPU entropy is slower than CPU** (wave 1) → Experiment D measures whether the bottleneck is entropy or dispatch
- **Ring-buffer streaming** (wave 1) → All experiments benefit from existing ring-buffer infrastructure
- **Cooperative-stitch kernel** (wave 1) → Experiment E reuses existing match finding, adds parallel parsing on top
- **Blelloch prefix sum overhead** (wave 1) → Experiment C must track per-dispatch overhead to avoid the same trap at scale

### Key principle from wave 1 that applies here

> GPU algorithms benefit from thread cooperation for data reuse. Serial data dependencies kill GPU throughput. But the bottleneck isn't always where you expect — profiling identified buffer allocation (35% of kernel time) as the real hotspot, not compute.

These experiments are designed to systematically identify where the bottlenecks actually are, rather than assuming.
