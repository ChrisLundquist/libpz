# GPU Compression Experiments — Wave 2

**Status:** In progress (CPU scaffolding complete, GPU kernels pending)
**Last updated:** 2026-03-05
**Depends on:** [gpu-strategy.md](gpu-strategy.md), [experiments.md](experiments.md)

## Purpose

Four experiments designed to answer fundamental architectural questions about GPU compression. Unlike wave 1 (which optimized existing algorithms), these experiments each isolate a single variable to produce clean measurements. **The measurements matter more than the compression ratios.** A clean negative result that tells us *why* something doesn't work is more valuable than a sloppy implementation that compresses well.

Each experiment is named by a letter (C/D/E/F) and a short pipeline name usable via `pz -p <name>`.

### Implementation status

CPU reference implementations exist for all 4 pipelines (`src/bitplane.rs`, `src/fwst.rs`, `src/parlz.rs`, `src/repair.rs`). These serve as correctness scaffolding and decompression paths. **The GPU kernels are the actual experiment** — the CPU versions exist only to validate round-trip correctness and provide baseline measurements. The experiments cannot answer their architectural questions without GPU-native implementations.

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

### GPU kernel design (`bitplane_transpose.wgsl`)

**Bit transpose kernel:**
- Workgroup size: 256 threads (one per input byte in a tile)
- Shared memory: 8 × 32 bytes = 256 bytes (8 bit-plane output buffers per tile)
- Each thread reads 1 input byte, extracts 8 bits, atomically ORs each bit into the corresponding shared memory plane buffer
- After barrier, threads cooperatively write the 8 plane buffers to global memory
- For CPU fallback: process 8 bytes at a time using shifts and masks (no SIMD needed)

**GPU RLE kernel (optional — CPU RLE may suffice since each stream is N/8 bytes):**
- XOR adjacent bits to mark transitions → produces a binary "change" array
- Prefix sum over the change array to number the runs
- Compute run lengths from segment boundaries (run_end[i] - run_start[i])
- Uses existing `run_inclusive_prefix_sum` infrastructure from `src/webgpu/bwt.rs`

### Key measurements

- Total throughput (MB/s) including GPU transfer — **primary output**
- Per-stage timing: transpose, RLE, entropy code, total
- Compression ratio per Canterbury file
- Which files compress well and why (e.g., "kennedy.xls has zero-heavy integer fields, bit-planes 4-7 are mostly zero")

### Pass/fail criteria

- **Pass:** Pipeline runs correctly on all Canterbury files, compresses and decompresses losslessly, and we have clean throughput numbers.
- **Expected ratio:** Worse than gzip on text files, possibly competitive on binary/numeric files. If it beats gzip on kennedy.xls or ptt5, that's a notable result worth highlighting.
- **Failure worth investigating:** If throughput is lower than existing BW pipeline despite having no serial stages, the bottleneck is dispatch/transfer overhead, which is important to quantify.

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

- **Key extraction:** For position `i`, key is `input[i..i+w]`. Pad with zeros near end of input. Exactly `w` bytes per position for uniform radix sort.
- **Radix sort:** Stop after `w` radix passes (vs full suffix sort's log²n doubling steps). Break ties by position (stable sort) for deterministic output.
- **Tie-breaking matters for compression quality.** When two positions share a `w`-byte context, the BWT characters at those positions get placed adjacent in the output, but their relative order is determined by tie-breaking rather than by deeper context. Random tie-breaking produces worse context clustering than position-based tie-breaking. Use stable sort (tie-break by position).
- **Everything downstream unchanged** — same MTF → RLE → FSE as existing BW pipeline.

### GPU kernel design

**Modify existing `bwt_radix.wgsl`** to accept a max-pass parameter:
- Existing BWT GPU sort uses prefix-doubling: start with 1-byte keys, double to 2, 4, 8, ... until all suffixes are unique
- For FWST: replace prefix-doubling with fixed `w` radix passes over the `w`-byte key at each position
- **Key change:** Instead of `run_radix_sort` with convergence check, do exactly `w` passes of 8-bit radix sort on the concatenated key bytes
- Reuse existing radix sort infrastructure: histogram → prefix sum → scatter (all already GPU-accelerated in `src/webgpu/bwt.rs`)
- The `run_inclusive_prefix_sum` and histogram kernels need no changes — only the outer loop and key computation change

### Parameters to sweep

`w` values: 2, 4, 6, 8, 10, 12, 16, 24, 32, ∞ (full BWT baseline).

### Key measurements

For each `w` × each Canterbury file:
- Compression ratio (compressed_size / original_size)
- Radix sort time (this should decrease with smaller `w` — verify this)
- Total pipeline time
- **The compression ratio curve as a function of `w`** — plot this. The shape of this curve is the main deliverable. We expect diminishing returns: most of the compression comes from the first few bytes of context, with a long tail.
- "Context value" metric: `(ratio_at_w - ratio_at_{w+2}) / ratio_at_full_bwt` — report where this drops below 1%. That's the effective context depth.

### Pass/fail criteria

- **Pass:** We have a clean ratio-vs-w curve for every Canterbury file and can identify the effective context depth for each data type.
- **Strong pass:** `w=8` achieves within 3% of full BWT ratio on text files. If true, this has immediate practical value — we can speed up our BWT sort and barely lose compression.
- **Strong pass (throughput):** Sort time at `w=8` is measurably faster than full suffix sort. Quantify the speedup.
- **Informative failure:** If even `w=32` is noticeably worse than full BWT on some file, that file has deep repetitive structure that requires full suffix comparison. Identify which files these are and why (likely E.coli or ptt5). This is useful information, not a failure of the experiment.

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

- **Match finding:** Use existing GPU cooperative-stitch match finder (`lz77_coop.wgsl`). For each position, produce best match `(length, distance)` or "no match." This is already GPU-accelerated.
- **Parallel match selection:** Each position independently picks longest match. One thread per position, no communication.
- **Conflict resolution via forward max-propagation scan:**
  - Create `coverage[p] = p + match_length[p]` for match positions, `coverage[p] = p` otherwise
  - Parallel prefix-max scan: `coverage[p] = max(coverage[p], coverage[p-1])`
  - Position `p` is suppressed if `coverage[p-1] > p`
  - O(n) work, O(log n) depth, standard GPU primitive
- **Classification:** Each position becomes match-start, covered (inside a match), or literal.

**CPU greedy baseline** using same GPU match data, for direct comparison:
- Scan left to right; if match ≥ min_length, take it and advance; else emit literal and advance by 1.

### GPU kernel design (`parlz_resolve.wgsl`)

**Conflict resolution kernel — the new and interesting part:**

The forward max-propagation scan is a parallel prefix-max, structurally identical to the existing `run_inclusive_prefix_sum` but using `max` instead of `+`. Implementation:

1. **Initialize coverage array** (trivially parallel, 1 thread per position):
   ```
   coverage[p] = (matches[p].length > 0) ? p + matches[p].length : p
   ```

2. **Parallel prefix-max scan** (reuse Blelloch scan infrastructure from `src/webgpu/bwt.rs`):
   - Per-workgroup local scan with `max` operator
   - Recursive block-sum scan
   - Propagation phase
   - Replace `+` with `max` in the existing `run_inclusive_prefix_sum` kernel

3. **Classify positions** (trivially parallel, 1 thread per position):
   ```
   is_match_start[p] = (matches[p].length > 0) && (p == 0 || coverage[p-1] <= p)
   is_literal[p] = !is_match_start[p] && (p == 0 || coverage[p-1] <= p)
   is_covered[p] = (p > 0) && coverage[p-1] > p
   ```

4. **Stream compaction** (prefix sum over match-start/literal flags, then scatter):
   - Compact match-starts and literals into a dense token stream
   - Uses existing prefix-sum + scatter pattern from radix sort

### Key measurements

For each Canterbury file:
- Parallel-parse compressed size
- Greedy-parse compressed size (using same matches from GPU)
- **Ratio gap: `(parallel_size - greedy_size) / greedy_size`** — primary deliverable, report as percentage for every file
- Match suppression statistics: how many matches found but suppressed by conflict resolution? How many "wasted" match computations (matches found but suppressed)?
- Coverage fraction: what % of input bytes are covered by matches (parallel vs greedy)? If parallel covers significantly fewer bytes, conflict resolution is too aggressive.
- Comparison against gzip -6 and zstd -3 for context

### Pass/fail criteria

- **Pass:** We have clean ratio-gap numbers for every Canterbury file. The experiment succeeds as measurement even if the ratios are bad.
- **Strong pass:** Ratio gap < 5% on text files. This would mean fully-parallel LZ is viable.
- **Informative result:** Ratio gap varies significantly by file type. Report which data types suffer most from parallel parsing and hypothesize why. Likely: files with many overlapping match candidates at nearby positions (dense match environments) will have larger gaps.
- **Failure worth documenting:** If the max-propagation scan suppresses so many matches that compression ratio is worse than raw FSE on the literal stream, the conflict resolution strategy is too aggressive. Try alternative strategies: shorter-match-yields-to-longer rather than earlier-wins-on-ties.

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

Re-Pair (Larsson & Moffat, 1999):

```
while any bigram occurs more than once:
    find the most frequent bigram (a, b)
    create a new symbol S representing (a, b)
    replace every non-overlapping occurrence of (a, b) with S
    record the rule S → (a, b)
```

Output: dictionary of rules + FSE-encoded final string with all replacements applied.

### GPU kernel design (3 kernels per round)

**1. Bigram frequency count kernel (`repair_histogram.wgsl`):**
- For each position `i`, compute bigram `(symbol[i], symbol[i+1])`
- When alphabet is small (< 512 symbols): use 2D histogram in shared memory (alphabet² entries). At 456 symbols → 208K entries, fits in shared memory. At 1000 symbols → doesn't fit.
- When alphabet is large (≥ 512 symbols): switch to sort-based counting — sort all bigrams, then count run lengths using existing radix sort infrastructure.
- Each workgroup processes a tile of positions, accumulates local histogram, then atomically adds to global histogram.

**2. Find maximum kernel (`repair_argmax.wgsl`):**
- Parallel reduction to find bigram with highest frequency.
- Standard two-pass reduction: per-workgroup local max → global max.
- Trivially parallel, ~microseconds of work.

**3. Replace + compact kernel (`repair_replace.wgsl`):**
- Scan array; at each position, if `(symbol[i], symbol[i+1])` equals target bigram, replace `symbol[i]` with new symbol and mark `symbol[i+1]` for deletion.
- Handle non-overlapping constraint: if positions `i` and `i+1` both start a match, only take `i` (parallel conflict resolution, same technique as Experiment E).
- Stream compaction to remove deleted positions: prefix sum over "kept" flags → scatter to compacted output. Uses existing prefix-sum infrastructure.

### Practical concerns

- **Alphabet growth:** Each round adds a new symbol. After 200 rounds, the alphabet has 456+ symbols. The 2D bigram histogram is alphabet² in size. Switch to sort-based counting when alphabet exceeds ~512 symbols.
- **Array compaction cost:** Each round removes ~(frequency_of_best_bigram) positions and needs a compaction pass. Consider batching: find top-K most frequent bigrams per round and replace all simultaneously. This reduces round count by K× at cost of more complex conflict resolution.
- **Memory budget:** Working array starts at N symbols and shrinks each round. Need space for: symbol array, bigram pairs, frequency histogram, compaction indices. Budget ~4× input size for workspace.
- **Stopping condition:** Stop when no bigram occurs more than threshold (e.g., 2). Each additional round after this creates a rule that saves 1 byte and costs ~2 bytes of dictionary overhead. Optimal stopping minimizes `compressed_string_size + dictionary_size`.

### Dispatch overhead tracking — most important measurement

Time each kernel dispatch separately. Track:
- Actual GPU compute time per round (via GPU timestamps/events from wgpu-profiler)
- Dispatch overhead per round (total wall time minus compute time)
- Number of rounds until convergence
- `total_dispatch_overhead / total_compute_time` ratio — if this exceeds 1.0, GPU is spending more time launching kernels than doing work

### Key measurements

For each Canterbury file:
- Final compression ratio (dictionary + compressed string)
- Number of rounds to convergence
- **Per-round timing: compute time vs dispatch overhead** — primary deliverable
- **Rounds-to-ratio curve:** ratio achieved after N rounds (where do diminishing returns kick in? Are the first 20 rounds doing most of the work?)
- Dictionary size as fraction of total compressed output
- Alphabet size at convergence
- `total_dispatch_overhead / total_compute_time` ratio
- Comparison against all existing pipelines and external tools

### Pass/fail criteria

- **Pass:** We have clean per-round timing breakdowns and know the compute-to-dispatch ratio. The experiment succeeds as measurement regardless of compression performance.
- **Strong pass (ratio):** Compression ratio within 10% of gzip on text files, and we identify a reasonable early-stopping point (< 50 rounds) that captures > 90% of compression benefit.
- **Strong pass (throughput):** Batching top-K replacements per round reduces round count below 30 while maintaining ratio. This makes it a practical algorithm.
- **Informative failure (dispatch):** If dispatch overhead dominates (> 50% of wall time), report the per-dispatch cost in microseconds. This directly tells us the minimum useful work per GPU dispatch on our target platforms and is valuable for all future GPU algorithm design.
- **Informative failure (ratio):** If grammar compression achieves distinctly different ratios across file types than BWT and LZ (e.g., much better on fields.c, much worse on E.coli), document the pattern. This motivates content-adaptive pipeline selection even if Re-Pair itself isn't competitive.

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
2. **F (fwst)** — Small modification to existing BWT GPU sort. Quick to implement, direct practical value.
3. **E (parlz)** — Most strategically important. Answers whether all-GPU LZ is viable.
4. **C (repair)** — Most complex, most novel. Answers GPU dispatch viability question.

If time is limited, **D and F alone** provide high-value calibration data. **E** is the most strategically important for the project's direction. **C** is exploratory but produces insights that generalize beyond compression.

---

## Benchmark matrix

All experiments must produce results comparable via a single CSV:

```
file,size,pipeline,params,compressed_size,ratio,throughput_mbs,time_total_ms,time_transfer_ms,time_compute_ms,notes
```

Run all 4 experimental pipelines + all existing pipelines + external tools (gzip -6, zstd -3, zstd -9) on every Canterbury corpus file. The summary table showing ratio per file per pipeline is the primary artifact.

Script: `scripts/run_experiments.sh`

---

## Relationship to existing learnings

These experiments build on insights from wave 1 (documented in [experiments.md](experiments.md)):

- **Hash table failure** (wave 1) → Experiment E uses existing cooperative-stitch match finder, not hash tables
- **GPU entropy is slower than CPU** (wave 1) → Experiment D measures whether the bottleneck is entropy or dispatch
- **Ring-buffer streaming** (wave 1) → All experiments benefit from existing ring-buffer infrastructure
- **Cooperative-stitch kernel** (wave 1) → Experiment E reuses existing match finding, adds parallel parsing on top
- **Blelloch prefix sum overhead** (wave 1) → Experiment C must track per-dispatch overhead to avoid the same trap at scale
- **Shared memory tiling** (wave 1) → Experiment D's bit-transpose kernel uses the same cooperative tile-loading pattern

### Reusable GPU primitives from existing infrastructure

| Primitive | Source | Used by |
|-----------|--------|---------|
| Prefix sum (inclusive, Blelloch) | `src/webgpu/bwt.rs` → `run_inclusive_prefix_sum` | E (prefix-max scan), C (stream compaction) |
| Radix sort (8-bit LSB-first) | `src/webgpu/bwt.rs` → `run_radix_sort` | F (fixed-window sort), C (sort-based bigram counting) |
| Histogram (atomic, column-major) | `bwt_radix.wgsl` | C (bigram frequency count) |
| Cooperative-stitch match finding | `lz77_coop.wgsl` | E (match finding stage) |
| Stream compaction (prefix sum + scatter) | `bwt_radix.wgsl` scatter phase | C (array compaction after replacement) |
| GPU profiling (wgpu-profiler timestamps) | `src/webgpu/mod.rs` | C (per-round dispatch overhead measurement) |

### Key principle from wave 1 that applies here

> GPU algorithms benefit from thread cooperation for data reuse. Serial data dependencies kill GPU throughput. But the bottleneck isn't always where you expect — profiling identified buffer allocation (35% of kernel time) as the real hotspot, not compute.

These experiments are designed to systematically identify where the bottlenecks actually are, rather than assuming.
