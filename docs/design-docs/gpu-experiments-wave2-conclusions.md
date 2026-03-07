# GPU Compression Experiments — Wave 2 Conclusions

**Date:** 2026-03-06
**Branch:** `claude/funny-vaughan`
**Corpus:** Canterbury (11 files) + Large (3 files), 13.97 MB total
**Platform:** Windows 11, WebGPU (wgpu)

---

## Summary Table

| Exp | Pipeline | Ratio | gzip Ratio | vs gzip | Throughput | Verdict |
|-----|----------|-------|------------|---------|------------|---------|
| B | sortlz | 39.6% | 28.6% | +11.0pp | 1.3 MB/s | Best experimental; beats deflate/lzf |
| A | csbwt | 47.8% | 28.6% | +19.2pp | 1.2 MB/s | Worse than plain BW; not viable |
| E | parlz | 59.6% | 28.6% | +31.0pp | 1.9 MB/s | Large ratio gap; parallel parsing costly |
| C | repair | 73.0% | 28.6% | +44.4pp | 0.4 MB/s | Dispatch overhead dominates |
| D | bitplane | 136.1% | 28.6% | expands | 2.0 MB/s | Expands data; wrong for general use |
| F | fwst | 433.4% | 28.6% | expands | 1.2 MB/s | Permutation overhead kills it |

Baseline comparison (same GPU-enabled build):

| Pipeline | Ratio | Throughput |
|----------|-------|------------|
| gzip | 28.6% | 10.1 MB/s |
| **bw** (GPU BWT) | 32.7% | 1.2 MB/s |
| **lzr** | 41.6% | 2.0 MB/s |
| **deflate** | 43.4% | 2.0 MB/s |
| **lzf** | 43.3% | 2.0 MB/s |

---

## Experiment D: Bitplane — "What is the GPU throughput ceiling?"

### Result: INFORMATIVE FAILURE

**Ratio:** 136.1% (expands data on 13/14 files)

Per-file breakdown:
| File | Ratio | Notes |
|------|-------|-------|
| ptt5 | 63.0% | Only file that compresses (CCITT fax, bit-structured) |
| E.coli | 101.2% | Near break-even (4-symbol DNA has redundant bit planes) |
| kennedy.xls | 115.3% | Structured binary, slight expansion |
| All text files | 163-260% | Massive expansion |

### Conclusions

1. **The GPU throughput ceiling question is not meaningfully answered.** Bitplane throughput (2.0 MB/s) is the same as deflate/lzf because the `pz` CLI's per-process overhead dominates. Profiling shows ~110ms process startup + ~260ms wgpu device init/shader compilation as fixed costs. Actual GPU compute scales modestly with input size (~120ms for 4MB). The pipeline has zero serial stages as designed, but the measurement tells us about device init, not compute ceiling.

2. **Bit-plane decomposition is wrong for general compression.** Text files have no redundancy in bit planes. RLE on the binary streams produces more bytes than the original. The only file that compresses (ptt5) has inherent bit-plane structure (CCITT fax encoding).

3. **Key takeaway for GPU strategy:** The ~260ms device init overhead is a per-process cost, fully amortizable in a long-running process or library API. It is NOT a per-dispatch cost. Kernel dispatch itself is fast (<1ms). The bench.sh numbers are misleading because they fork a new `pz` process per file, paying device init every time.

### What we learned about the throughput ceiling

The real constraint is **wgpu device initialization (~260ms)**, not dispatch overhead. In a persistent process (library, daemon, streaming), this cost is paid once. Effective GPU throughput for compute-only work is better estimated from the 3.7KB→4MB scaling: ~120ms of actual GPU work for 4MB, implying ~33 MB/s effective compute throughput (before accounting for CPU-side entropy coding).

---

## Experiment F: FWST — "Does full suffix sorting earn its cost?"

### Result: ARCHITECTURAL DEAD END

**Ratio:** 433.4% (massive expansion)

### Root Cause

The window-capped sort breaks BWT's LF-mapping property. Without LF-mapping, the decoder cannot reconstruct the original from the transformed data alone — it needs the full sorted permutation stored in the wire format. At 4 bytes per position, this adds O(4n) bytes to the output, which dwarfs any compression benefit from the transform itself.

For alice29.txt (152KB): the permutation alone is 608KB before FSE compression. Even after FSE-encoding the permutation, the overhead is ~550KB. The BWT-like transformed data compresses well through MTF+RLE+FSE, but the permutation cost makes the total output 4-5x larger than the input.

### Conclusions

1. **Full suffix sorting does earn its cost — and it's structurally necessary, not optional.** The value of full suffix sort isn't just better context clustering. It's that full suffix sort produces a unique permutation that is invertible via LF-mapping without storing the permutation. This is a qualitative, not quantitative, property — you either have it or you don't.

2. **There is no useful "ratio vs window depth" curve.** The experiment design assumed FWST could use the same decode path as BWT. It can't. The permutation overhead swamps any signal about context depth. The question "how deep is data's statistical structure?" cannot be answered with this experimental design.

3. **Alternative approach needed:** To measure effective context depth, compare BWT with different numbers of radix sort passes (all using LF-mapping decode). The convergence point of the suffix sort iteration count would be a better proxy for context depth.

---

## Experiment E: Parlz — "What is the compression ratio cost of removing serial parsing?"

### Result: LARGE RATIO GAP — PARALLEL PARSING NOT VIABLE

**Ratio:** 59.6% (vs 43.3% for greedy LZ = **37.6% gap**)

Per-file ratio gap (parlz vs lzf):
| File | parlz | lzf | Gap |
|------|-------|-----|-----|
| xargs.1 | 100.9% | 90.5% | +11.5% |
| plrabn12.txt | 74.8% | 62.8% | +19.2% |
| asyoulik.txt | 73.4% | 60.3% | +21.6% |
| alice29.txt | 71.5% | 57.6% | +24.1% |
| grammar.lsp | 109.2% | 84.8% | +28.9% |
| lcet10.txt | 73.8% | 56.0% | +31.7% |
| cp.html | 72.4% | 50.2% | +44.2% |
| fields.c | 79.1% | 53.7% | +47.3% |
| sum | 76.9% | 49.7% | +54.6% |
| ptt5 | 31.7% | 13.2% | +140.9% |
| kennedy.xls | 94.3% | 20.9% | +350.8% |

### Conclusions

1. **Parallel parsing has a large, consistent cost.** The ratio gap ranges from 11.5% to 350.8%, with text files clustering around 20-50% and binary/structured files much worse. This is well above the 5% threshold that would make fully-parallel GPU LZ viable.

2. **The conflict resolution strategy (forward max-propagation) is too aggressive.** On kennedy.xls (structured binary with many long matches), parlz achieves 94.3% ratio vs lzf's 20.9% — a 350% gap. The parallel scan suppresses nearly all matches in dense match environments. The forward-max-propagation scan gives absolute priority to earlier positions, preventing later positions from using better matches.

3. **GO/NO-GO decision: NO-GO for fully-parallel GPU LZ.** The ratio gap is too large for a competitive compressor. The project should continue with the hybrid approach: GPU match finding + CPU serial parsing. This is what the existing lzr/lzf/lzseqr pipelines already do.

4. **Data type dependency confirms the theory.** Dense match environments (kennedy.xls: many overlapping match candidates) suffer far more than sparse environments (xargs.1: few matches). This matches the predicted failure mode from the experiment design.

---

## Experiment C: Repair — "Is iterative GPU kernel dispatch viable?"

### Result: DISPATCH OVERHEAD DOMINATES — NOT VIABLE AT CURRENT SCALE

**Ratio:** 73.0% overall, 0.4 MB/s GPU throughput

Notable per-file results:
| File | Ratio | Notes |
|------|-------|-------|
| ptt5 | 24.6% | Best — grammar compression finds CCITT structure |
| kennedy.xls | 39.5% | Good — hierarchical structure in spreadsheet data |
| alice29.txt | 87.4% | Poor on text |
| E.coli | 71.0% | Moderate on DNA (simple alphabet, limited bigram diversity) |

### Conclusions

1. **Iterative GPU dispatch is slow due to per-round buffer allocation and readback.** Each round creates new buffers (histogram, scratch, replace output, prefix sum, compacted) and reads back the argmax result and compacted symbols. With 100+ rounds, this per-round host overhead (buffer creation + readback synchronization) accumulates. Actual wall time for the full corpus was 33.3 seconds at 0.4 MB/s.

2. **Grammar compression finds different redundancy than LZ/BWT.** Repair's 24.6% on ptt5 (vs lzf's 13.2%) and 39.5% on kennedy.xls (vs lzf's 20.9%) show it captures some structure, but it's consistently worse than LZ on every file. Grammar compression's advantage (hierarchical patterns) doesn't overcome its disadvantage (greedy bigram selection without global optimization).

3. **Per-round host overhead dominates.** The 0.4 MB/s throughput for repair vs 2.0 MB/s for single-dispatch pipelines reflects the cost of hundreds of GPU↔CPU synchronization points (buffer readbacks for argmax results and compacted arrays). Reducing round count via batched top-K replacement, or eliminating readbacks via persistent GPU-side state, would significantly improve throughput.

4. **Early stopping opportunity:** The algorithm likely does most useful work in the first 20-50 rounds (when high-frequency bigrams are being replaced). After that, diminishing returns set in but dispatch costs remain constant. Batching multiple bigram replacements per round would reduce round count.

---

## Experiment B: SortLZ — "Is GPU radix-sort LZ77 competitive?"

### Result: BEST EXPERIMENTAL PIPELINE — BEATS DEFLATE/LZF BASELINES

**Ratio:** 39.6% (vs deflate 43.4%, lzf 43.3%)

Per-file comparison:
| File | sortlz | lzf | deflate | bw |
|------|--------|-----|---------|-----|
| ptt5 | 14.9% | 13.2% | 13.5% | 18.7% |
| kennedy.xls | 27.5% | 20.9% | 21.2% | 17.5% |
| alice29.txt | 44.1% | 57.6% | 58.2% | 61.3% |
| E.coli | 44.4% | 39.7% | 39.6% | 26.8% |
| bible.txt | 39.3% | 48.1% | 48.0% | 39.1% |
| world192.txt | 37.7% | 49.4% | 49.8% | 34.5% |

### Conclusions

1. **SortLZ produces significantly better compression ratios than deflate/lzf on large text files.** On bible.txt: 39.3% vs 48.1% (18% better). On world192.txt: 37.7% vs 49.4% (24% better). The global sort-based match finding discovers longer and better matches than the sliding-window hash approach.

2. **SortLZ approaches BW pipeline quality on text.** Bible.txt: 39.3% vs 39.1% (BW). World192.txt: 37.7% vs 34.5%. This is remarkable because SortLZ is an LZ77 variant (not a BWT variant) — it achieves BWT-competitive ratios through exhaustive match finding.

3. **GPU throughput is bottlenecked by per-process device init.** At 1.3 MB/s end-to-end (bench.sh), SortLZ is slower than CPU lzf. But ~260ms of this is one-time wgpu device init + shader compilation, not per-dispatch cost. In a persistent process, the radix sort + verify compute would dominate.

4. **Most promising direction:** In a library/daemon context where device init is amortized, SortLZ's match quality advantage becomes a real differentiator. The algorithm is naturally GPU-parallel (radix sort + verification are both embarrassingly parallel).

---

## Experiment A: CSBWT — "Does context segmentation enable parallel entropy coding?"

### Result: WORSE THAN PLAIN BW — NOT VIABLE

**Ratio:** 47.8% (vs BW 32.7% = **15.1pp worse**)

### Conclusions

1. **Context segmentation fragments BWT context clusters, losing compression.** BWT groups symbols by their following context. Segmenting and encoding independently breaks cross-segment statistical dependencies. Each segment's FSE table has less data to model, producing worse entropy estimates.

2. **The adaptive accuracy log helps but doesn't compensate.** Per-segment FSE with tuned accuracy_log is still worse than full-array MTF+RLE+FSE because the segment boundaries interrupt run-length patterns.

3. **Not useful as a parallelization strategy.** The GPU BWT is already the expensive step (correctly offloaded). The CPU MTF+RLE+FSE on the full BWT output is fast enough that parallelizing entropy coding via segmentation isn't worth the quality loss.

---

## Cross-Experiment Synthesis

### The four original questions answered

| Question | Answer |
|----------|--------|
| **D: What's the GPU throughput ceiling?** | Unmeasurable via CLI — ~260ms wgpu device init dominates. Kernel dispatch itself is fast (<1ms). Effective compute throughput ~33 MB/s on 4MB input. |
| **F: How deep is data's statistical structure?** | Cannot measure — FWST's permutation overhead prevents meaningful comparison. Full suffix sort is structurally required for BWT invertibility. |
| **E: Can we remove serial LZ parsing?** | **No.** 37.6% ratio gap is far too large. Hybrid GPU match finding + CPU parsing is the correct approach. |
| **C: Can we iterate on GPU?** | Repair's iterative pattern is slow (0.4 MB/s) but the bottleneck is hundreds of rounds of buffer allocation + readback, not dispatch latency per se. Batching and persistent buffers would help. |

### GPU strategy implications

1. **Device init, not dispatch, is the CLI bottleneck.** The ~260ms wgpu device init is a per-process cost. In a library or daemon where the device persists, this vanishes. Kernel dispatch is fast. The bench.sh numbers are misleading because they fork a fresh process per file.

2. **Hybrid is correct.** GPU for embarrassingly-parallel stages (BWT sort, LZ match finding), CPU for serial stages (parsing, entropy coding). The existing lzr/lzf/lzseqr architecture is on the right track.

3. **SortLZ's match quality is notable.** It achieves BWT-competitive ratios via LZ77, suggesting a "sort-based match finder → CPU greedy/lazy parse → entropy code" pipeline could compete with BWT on both ratio and speed.

4. **Grammar compression and bit-plane decomposition are dead ends** for general-purpose compression. Repair finds some niche structure but can't compete with LZ/BWT. Bitplane is only useful for specific binary formats with redundant bit planes.

### Recommended next steps

1. **Benchmark with persistent GPU device.** Write an in-process benchmark (Rust `criterion` or similar) that initializes wgpu once and compresses multiple files. This will reveal actual GPU compute throughput without device init noise.

2. **Integrate SortLZ match finding into the LZ pipeline.** Use GPU radix sort for match finding, then CPU greedy/lazy parsing, then existing entropy coding. This combines SortLZ's match quality with serial parsing's compression efficiency.

3. **Drop FWST, bitplane, CSBWT, repair from further GPU work.** These experiments have answered their questions — the answers are negative.

4. **Keep parlz as a measurement baseline.** While its ratios are too poor for production, it's useful for measuring the "parallel parsing gap" on new data types.

### Overhead breakdown (measured)

```
Component          Time      Notes
───────────────    ───────   ─────────────────────────
Process startup    ~110ms    Rust binary init, file I/O (same for CPU and GPU)
wgpu device init   ~260ms    One-time per process; adapter + device + shader compile
GPU compute (4MB)  ~120ms    Scales with input size; actual kernel work
Kernel dispatch    <1ms      Per-dispatch cost is negligible
PCIe transfer      <1ms      4MB at PCIe 5 speeds
```
