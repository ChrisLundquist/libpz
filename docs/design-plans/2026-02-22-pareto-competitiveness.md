# Pareto-Competitiveness Strategy Design

## Summary

libpz is a Rust compression library that assembles algorithms — LZ77 match finding, BWT, Huffman, rANS, FSE — into composable pipelines that can run on the CPU, the GPU, or both simultaneously. The goal of this effort is to make libpz's best pipelines Pareto-competitive with gzip: for every point on gzip's speed-versus-ratio curve (levels 1 through 9), libpz should have a configuration that is either faster at the same compression ratio, or achieves a better ratio at the same speed.

The strategy is built around the LzSeqR pipeline as the primary compressor, with seven phases of work covering measurement, match-finding quality, optimal parsing, entropy throughput, heterogeneous scheduling, GPU-native transform pipelines, and auto-selection tuning. Rather than building a single monolithic codec, libpz differentiates by composing stages dynamically — assigning match finding to the CPU (where LZ77's sequential dependency rules out GPU parallelism) and entropy coding to the GPU when block sizes make the transfer worthwhile. An auto-selector analyzes each input and chooses the pipeline and compute backend per workload. Phase 8 validates the complete result against gzip, zlib-ng, lz4, and zstd across standard corpora.

## Definition of Done

1. **Pareto-competitive with gzip**: libpz's best pipeline(s) sit on or above gzip's speed-vs-ratio curve when tested on standard corpora (Canterbury, Silesia, etc.). For any point on gzip's curve (levels 1-9), libpz has a pipeline/config that is either faster at the same ratio, or better ratio at the same speed.

2. **Single-thread CPU baseline**: One CPU thread of libpz achieves throughput and ratio comparable to gzip -6 (default). This is the minimum bar.

3. **Multi-thread and GPU scaling**: With full parallelism (CPU threads + GPU), libpz significantly outperforms gzip on throughput while maintaining competitive ratios. GPU contributes meaningfully to at least some pipeline stages.

4. **Deliverable**: A design document with a concrete strategy covering encoding efficiency, pipeline composition, CPU optimization, GPU acceleration approach, and auto-selection — broken into implementable phases.

## Acceptance Criteria

### pareto-competitiveness.AC1: Pareto-competitive with gzip
- **pareto-competitiveness.AC1.1 Success:** LzSeqR single-thread achieves compression ratio within 2% of gzip -6 on Canterbury corpus
- **pareto-competitiveness.AC1.2 Success:** LzSeqR single-thread achieves compression ratio within 2% of gzip -6 on Silesia corpus
- **pareto-competitiveness.AC1.3 Success:** LzSeqH achieves higher throughput than gzip -1 at comparable or better ratio
- **pareto-competitiveness.AC1.4 Success:** LzSeqR + optimal parsing achieves better ratio than gzip -9 on at least 50% of corpus files
- **pareto-competitiveness.AC1.5 Success:** For each gzip level 1-9, there exists a libpz configuration (pipeline + quality level) that is faster at the same ratio or achieves better ratio at the same speed
- **pareto-competitiveness.AC1.6 Failure:** Random/incompressible data does not crash or produce output larger than input + 1% overhead

### pareto-competitiveness.AC2: Single-thread CPU baseline
- **pareto-competitiveness.AC2.1 Success:** `pz -t 1` with LzSeqR compresses Canterbury corpus at throughput >= gzip -6 throughput
- **pareto-competitiveness.AC2.2 Success:** `pz -t 1` with LzSeqR achieves ratio within 2% of gzip -6 on Canterbury corpus
- **pareto-competitiveness.AC2.3 Success:** `pz -t 1` decompression throughput >= gzip decompression throughput
- **pareto-competitiveness.AC2.4 Edge:** Single-thread mode works correctly for inputs < 1KB (no block-parallel overhead)

### pareto-competitiveness.AC3: Multi-thread and GPU scaling
- **pareto-competitiveness.AC3.1 Success:** Multi-thread CPU compression achieves near-linear speedup up to 4 threads on 1MB+ inputs
- **pareto-competitiveness.AC3.2 Success:** GPU entropy encoding for LzSeqR achieves higher throughput than CPU-only path on blocks >= 256KB
- **pareto-competitiveness.AC3.3 Success:** Heterogeneous scheduling (CPU match + GPU entropy) achieves higher throughput than pure CPU on 512KB+ inputs with GPU available
- **pareto-competitiveness.AC3.4 Success:** GPU-native Bbw pipeline achieves higher throughput than CPU on BWT-suitable data >= 256KB
- **pareto-competitiveness.AC3.5 Failure:** No GPU available — pure CPU path works with zero overhead (no GPU initialization, no fallback penalty)
- **pareto-competitiveness.AC3.6 Failure:** GPU device lost mid-compression — graceful fallback to CPU, no data corruption

### pareto-competitiveness.AC4: Measurement and tooling
- **pareto-competitiveness.AC4.1 Success:** Benchmark harness produces single-thread Pareto comparison table against gzip, zlib-ng, lz4, zstd
- **pareto-competitiveness.AC4.2 Success:** Bit-budget analysis decomposes compressed output into literal/match/entropy/header costs
- **pareto-competitiveness.AC4.3 Success:** Auto-selection picks pipeline within 5% of best-available ratio in >90% of corpus files

## Glossary

- **LzSeq**: libpz's primary match-finding layer, built on LZ77 with zstd-style code+extra-bits token encoding, repeat offset tracking, and a 6-stream demux. Specific to this codebase.
- **LzSeqR / LzSeqH**: The flagship pipelines. LzSeqR pairs LzSeq with rANS entropy coding (balanced/quality). LzSeqH pairs it with Huffman (speed-oriented).
- **Pareto curve**: A set of configurations where no option is strictly better on both speed and ratio. A point is Pareto-competitive if no competitor beats it on both dimensions simultaneously.
- **rANS (range Asymmetric Numeral Systems)**: An entropy coder achieving near-Shannon-optimal compression with fast decoding. libpz implements an interleaved N-way variant.
- **FSE (Finite State Entropy)**: A table-driven entropy coder derived from ANS, implemented as a finite state machine. GPU-friendly.
- **BWT (Burrows-Wheeler Transform)**: A reversible text permutation that groups similar characters, improving subsequent compression. libpz uses SA-IS on CPU and radix sort on GPU.
- **Bijective BWT (BBWT)**: A BWT variant that is its own inverse, avoiding the end-of-string sentinel. Enables cleaner GPU implementation.
- **Bbw pipeline**: Bijective BWT → MTF → RLE → FSE. Suited for highly repetitive data.
- **Optimal parsing**: Dynamic programming over LZ77 match candidates to minimize total encoded cost, rather than greedy longest-match selection.
- **Repeat offset**: In LzSeq/zstd-style encoding, recently used match offsets can be referenced with a short code rather than a full offset value, saving bits.
- **Hash chain**: LZ77 match-finding data structure linking positions with the same hash prefix. Chain search depth trades speed for match quality.
- **Demux**: libpz's pattern for splitting LZ77/LzSeq output into separate symbol streams (offsets, lengths, literals) before entropy coding.
- **Heterogeneous scheduling**: Running different compression stages on different backends (CPU and GPU) simultaneously within the same job.
- **Ring-buffered dispatch**: A scheduling pattern where CPU submits GPU work in a circular buffer, allowing GPU compute to overlap with CPU readback.
- **Auto-selection**: The subsystem that inspects input characteristics (entropy, match density, run ratio) and selects the best pipeline and compute backend.
- **Canterbury / Silesia corpus**: Standard benchmark file collections for comparing compression algorithms.
- **Bit-budget analysis**: Decomposing compressed output to attribute bits to each cost source (literals, match offsets, match lengths, entropy tables, headers).
- **zlib-ng**: SIMD-optimized fork of zlib. Competitor in the balanced Pareto zone.
- **pigz**: Parallel gzip using multiple CPU threads. Competitor in the throughput zone.
- **zstd**: Facebook's Zstandard compressor. The modern reference for ratio+speed. Uses repeat offsets, FSE, and optimal parsing.
- **lz4**: Extremely fast compressor prioritizing throughput over ratio. Reference point for the speed end.
- **WebGPU / wgpu**: Cross-platform GPU API. wgpu is the Rust implementation (v27). Kernels written in WGSL.
- **Shannon entropy**: Theoretical lower bound on bits per symbol given a probability distribution. Used as cost estimate in optimal parsing.

## Architecture

### Strategic Approach: LzSeq-First with Heterogeneous Scheduling

The strategy concentrates effort on the LzSeqR pipeline as the flagship compressor while leveraging libpz's composable stage architecture for heterogeneous CPU+GPU execution. The competitive landscape includes gzip, zlib-ng, pigz, lz4, and zstd. Rather than cloning any single compressor, libpz differentiates through composable pipelines, auto-selection, and heterogeneous scheduling — capabilities no existing compressor offers.

### Competitive Positioning

The Pareto curve has three zones libpz must cover:

| Zone | Competitors | libpz Pipeline | Strategy |
|------|------------|----------------|----------|
| **Speed** | lz4, zlib-ng -1 | LzSeqH (Huffman decode) | Fast match finding, lightweight entropy |
| **Balanced** | zlib-ng -6, pigz | LzSeqR (default config) | Match gzip-6 ratio at higher throughput |
| **Quality** | zstd -3 to -9 | LzSeqR + optimal parsing | Repeat-offset-aware DP for best ratio |

libpz's differentiator is **composable, heterogeneous pipelines with auto-selection** — not a single monolithic codec. Different pipelines target different speed/ratio tradeoffs. Auto-selection picks the best pipeline and compute backend for each workload.

### Key Components

**LzSeq match finding** (`src/lzseq.rs`, `src/lz77.rs`): Hash-chain match finder with repeat offset tracking. Currently uses 3-byte hash prefix, fixed 64-entry bucket cap, and 32KB/128KB windows. Improvements target match quality and optimal parsing integration.

**Optimal parser** (`src/optimal.rs`): Backward DP over top-K match candidates (K=4). Cost model uses Shannon entropy weighting. Currently does not track repeat offset state — the biggest ratio gap vs zstd-style encoding.

**Entropy coders** (`src/rans.rs`, `src/huffman.rs`, `src/fse.rs`): rANS (interleaved N-way), Huffman, and FSE backends. rANS SIMD decode path exists in `src/simd.rs` but is not wired to the hot decode loop. GPU backends in `src/webgpu/{rans,huffman,fse}.rs` with kernels in `kernels/`.

**Pipeline orchestration** (`src/pipeline/mod.rs`, `src/pipeline/parallel.rs`): Block-parallel compression with per-block pipeline selection. Container format supports mixed pipelines per block (each block header records pipeline ID). GPU batching via ring-buffered handoff in `src/webgpu/lz77.rs`.

**Auto-selection** (`src/pipeline/mod.rs:select_pipeline`): Heuristic-based using entropy, match density, run ratio from `src/analysis.rs`. Trial mode (`select_pipeline_trial`) compresses sample with all candidates. Both need empirical tuning.

**GPU-friendly transforms**: Bijective BWT (`src/webgpu/bwt.rs`, `kernels/bwt_radix.wgsl`), RLE, delta encoding — transforms without LZ77's sequential dependency. These enable fully GPU-accelerated pipeline paths for suitable workloads.

### Data Flow

```
Input → Analysis Profiler → Auto-Selector
                                ↓
                    ┌───────────┴───────────┐
                    │                       │
              LzSeq path               BWT/Transform path
              (general data)           (repetitive data)
                    │                       │
           CPU Match Finding          GPU Transforms
           (hash chain / optimal)     (BBWT, RLE, delta)
                    │                       │
              ┌─────┴─────┐                 │
              │           │                 │
         CPU Entropy  GPU Entropy      GPU Entropy
         (small blk)  (large blk)     (rANS/FSE/Huffman)
              │           │                 │
              └─────┬─────┘                 │
                    │                       │
                Block Assembly ←────────────┘
                    │
                Container Output
```

### Heterogeneous Scheduling

Blocks are independent and stages within a block can run on either CPU or GPU. The scheduler distributes work based on:

- **Per-block pipeline**: Data-dependent, chosen by auto-selection (analysis profiling)
- **Per-stage backend**: Resource-dependent, based on GPU availability and block size thresholds (GPU entropy wins at 256KB+, loses below 128KB)
- **Cross-block distribution**: Load-dependent, filling both CPU and GPU queues. CPU compresses block N+1 while GPU entropy-encodes block N.

The container format already supports mixed pipelines per block. The decoder doesn't care which backend compressed each block.

## Existing Patterns

Investigation found established patterns this design follows:

**Composable stages**: Each algorithm (`src/{algorithm}.rs`) exposes `encode()`/`decode()` returning `PzResult<T>`. Pipelines compose stages via the demuxer (`src/pipeline/demux.rs`). This design extends the pattern — it doesn't replace it.

**Multi-stream demux**: LZ77 pipelines split into 3 streams (offsets, lengths, literals). LzSeq already splits into 6 streams. The demux architecture supports per-stream entropy coding naturally.

**GPU fallback**: GPU feature is gated behind `webgpu` cargo feature. All GPU paths fall back to CPU when no device is available. This design maintains that pattern — GPU is an accelerator, not a requirement.

**Ring-buffered GPU batching**: `src/webgpu/lz77.rs` implements ring-buffered dispatch that overlaps GPU compute with CPU readback. The heterogeneous scheduler extends this pattern to entropy coding stages.

**Block parallelism**: `src/pipeline/parallel.rs` compresses blocks independently via scoped threads. The scheduler builds on this — it adds backend assignment per block, not a new parallelism model.

**Auto-selection heuristic**: `src/pipeline/mod.rs:select_pipeline` uses `src/analysis.rs` profiling. This design tunes existing thresholds and adds trial-mode improvements, not a replacement architecture.

**Divergence from existing patterns:**

**Optimal parsing as default**: Currently optional (`ParseStrategy::Auto` uses lazy). This design makes optimal parsing the default for LzSeqR quality mode. Justified by the 4-6% ratio improvement measured in `ARCHITECTURE.md` and the need to close the ratio gap.

**Repeat-offset-aware cost model**: The current cost model in `src/optimal.rs` does not track repeat offset state. This design adds state tracking to the DP, which is a novel extension of the existing backward-DP approach.

## Implementation Phases

<!-- START_PHASE_1 -->
### Phase 1: Measurement Baseline

**Goal:** Establish where libpz sits on the Pareto curve and identify the top ratio sinks.

**Components:**
- Benchmark harness extension in `scripts/bench.sh` — add `-t 1` single-thread mode as a named preset, add competitor runners for zlib-ng, lz4, zstd
- Bit-budget analysis tool — new `pz --analyze` mode or `scripts/analyze-ratio.sh` that decomposes compressed output into literal bytes, match offset bits, match length bits, entropy table overhead, container headers
- LzSeqR and LzSeqH added to default benchmark pipeline set alongside deflate, lzr, lzf

**Dependencies:** None (first phase)

**Done when:** Can produce a single-thread Pareto comparison table (ratio vs throughput) for libpz pipelines against gzip, zlib-ng, lz4, zstd on Canterbury+Silesia. Bit-budget analysis shows where libpz loses ratio bits relative to gzip.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: LzSeq Match Finding Quality

**Goal:** Improve match finding to close the compression ratio gap.

**Components:**
- Hash chain improvements in `src/lz77.rs` — adaptive chain depth based on data compressibility (deeper chains on compressible data, shallower on random), 4-byte hash prefix option for fewer collisions, adaptive bucket cap (currently fixed at 64)
- Match profitability tuning in `src/lzseq.rs` — review `min_profitable_length()` thresholds, ensure short matches at large distances are correctly rejected
- Window size exploration — test 64KB, 128KB, 256KB windows for LzSeq and measure ratio vs speed tradeoff

**Dependencies:** Phase 1 (measurement baseline to quantify improvements)

**Done when:** LzSeqR single-thread compression ratio improves measurably on Canterbury+Silesia relative to Phase 1 baseline. Bit-budget analysis confirms ratio bits shifted from wasted overhead to useful match encoding. Tests pass for all modified match-finding paths.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Repeat-Offset-Aware Optimal Parsing

**Goal:** Integrate optimal parsing into LzSeqR with a cost model that exploits repeat offsets.

**Components:**
- Cost model extension in `src/optimal.rs` — track repeat offset state through the backward DP. When evaluating match candidates, account for whether the match offset matches a recent offset (cost reduction) or sets up a future repeat (forward-looking heuristic)
- LzSeqR integration — make optimal parsing the default for LzSeqR quality mode, lazy matching for speed mode
- Configuration — expose quality levels (speed/default/quality) that control parse strategy and chain depth

**Dependencies:** Phase 2 (improved match candidates feed into optimal parser)

**Done when:** LzSeqR with optimal parsing achieves compression ratio competitive with gzip -6 on Canterbury+Silesia. Repeat-offset-aware cost model demonstrably selects matches that improve ratio vs the current cost model. Tests verify round-trip correctness for all parse strategies.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Entropy Throughput (rANS SIMD + GPU Entropy)

**Goal:** Improve encode/decode throughput on the speed axis of the Pareto curve.

**Components:**
- rANS SIMD decode wiring in `src/simd.rs` and `src/rans.rs` — connect existing SSE2/AVX2 intrinsics to the hot decode loop for interleaved rANS
- GPU entropy encode path for LzSeqR — extend `src/webgpu/rans.rs` and `kernels/rans_encode.wgsl` to accept LzSeq's 6-stream output, triggered for blocks >= 256KB
- Huffman atomic contention fix in `kernels/huffman_encode.wgsl` — chunk-based packing to replace per-bit atomic_or

**Dependencies:** Phase 3 (optimal parsing determines the match stream that entropy coders process)

**Done when:** rANS decode throughput improves measurably with SIMD. GPU entropy encoding for LzSeqR works at 256KB+ block sizes with throughput exceeding CPU-only path. Huffman GPU encode scales better after contention fix. All round-trip tests pass including GPU-encode → CPU-decode and CPU-encode → GPU-decode cross-paths.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Heterogeneous Block Scheduling

**Goal:** CPU and GPU cooperatively compress blocks, with per-stage backend assignment.

**Components:**
- Block scheduler in `src/pipeline/parallel.rs` — extends existing block-parallel loop to assign backends per stage. CPU does match finding; for large blocks, GPU does entropy encoding. For small blocks or no GPU, pure CPU path with zero overhead.
- Ring-buffered handoff generalization — extend the existing ring buffer pattern from `src/webgpu/lz77.rs` to handle LzSeq-match → GPU-entropy handoff
- Load balancing — CPU continues compressing block N+1 while GPU entropy-encodes block N. Dynamic work distribution based on queue depth.

**Dependencies:** Phase 4 (GPU entropy encoding for LzSeqR must work before scheduling can assign it)

**Done when:** Multi-block compression on GPU-equipped machines achieves higher throughput than pure CPU path for inputs >= 512KB. CPU-only fallback works with zero overhead when no GPU. Tests verify correct output regardless of scheduling decisions.
<!-- END_PHASE_5 -->

<!-- START_PHASE_6 -->
### Phase 6: GPU-Native Transform Pipelines

**Goal:** Optimize non-LZ77 pipelines for full GPU execution on suitable workloads.

**Components:**
- Bijective BWT pipeline optimization — ensure Bbw (BBWT → MTF → RLE → FSE) can run fully on GPU with existing kernels (`kernels/bwt_radix.wgsl`, `kernels/bwt_rank.wgsl`, `kernels/fse_encode.wgsl`)
- GPU-friendly transforms — evaluate RLE and delta encoding as GPU-accelerated stages; implement GPU kernels if beneficial
- Pipeline-level GPU dispatch — for Bbw and other transform-heavy pipelines, dispatch entire pipeline to GPU rather than stage-by-stage handoff

**Dependencies:** Phase 5 (heterogeneous scheduler provides the framework for backend assignment)

**Done when:** Bbw pipeline achieves higher throughput on GPU than CPU for blocks >= 256KB on BWT-suitable data (high repetition). GPU transform stages produce identical output to CPU equivalents. Tests cover GPU-only and mixed GPU/CPU execution paths.
<!-- END_PHASE_6 -->

<!-- START_PHASE_7 -->
### Phase 7: Auto-Selection Tuning

**Goal:** Auto-selector picks the best pipeline and compute backend per workload.

**Components:**
- Empirical threshold tuning in `src/pipeline/mod.rs` — run all pipelines on Canterbury + Silesia + expanded corpus, record which pipeline wins for each file, retune heuristic thresholds in `select_pipeline()`
- Lightweight trial mode — compress 4-8KB sample with top 3 heuristic candidates instead of all 8, reducing trial overhead
- Backend-aware selection — auto-selector considers GPU availability and input size when recommending pipeline. Prefer GPU-friendly pipelines (Bbw) when GPU is available and data is BWT-suitable; prefer LzSeqR otherwise
- Analysis module improvements in `src/analysis.rs` — add any profiling features needed to distinguish BWT-suitable vs LZ-suitable data

**Dependencies:** Phases 2-6 (all pipeline optimizations must be complete before tuning selection)

**Done when:** Auto-selection picks a pipeline that is within 5% of the best-available pipeline's ratio on Canterbury+Silesia in >90% of cases. Trial mode overhead is < 10% of total compression time. Tests verify auto-selection produces valid output for all data profiles.
<!-- END_PHASE_7 -->

<!-- START_PHASE_8 -->
### Phase 8: Pareto Validation

**Goal:** Verify Pareto-competitiveness against all target competitors.

**Components:**
- Full Pareto benchmark using Phase 1 infrastructure — run libpz (all quality levels, single-thread and multi-thread, with and without GPU) against gzip levels 1-9, zlib-ng, pigz, lz4, zstd levels 1-9 on Canterbury + Silesia + expanded corpus
- Pareto curve visualization — script or tool that plots speed-vs-ratio for all compressors and identifies where libpz sits
- Gap analysis — identify remaining areas where competitors dominate and document as future work

**Dependencies:** Phase 7 (auto-selection must be tuned before final validation)

**Done when:** libpz demonstrates Pareto-competitive results: for each point on gzip's speed-vs-ratio curve, libpz has a configuration that is faster at the same ratio or achieves better ratio at the same speed. Results documented with reproducible benchmarks.
<!-- END_PHASE_8 -->

## Additional Considerations

**GPU LZ77 match finding is a confirmed dead end.** Two attempts (global hash table atomics, shared-memory local kernel) produced near-zero matches due to LZ77's sequential lookup-then-update requirement. nvcomp also does not GPU-accelerate LZ77 match finding. This design deliberately avoids GPU match finding and focuses GPU contribution on entropy coding and parallelizable transforms.

**Compression format stability.** Changes to match encoding, optimal parsing, or entropy coding may change the compressed output format. The container format version field must be incremented if the bitstream changes in ways that break backward compatibility. Existing test corpus compressed files must be re-validated.

**Decompression performance.** While this design focuses on compression-side improvements, several changes directly improve decompression: rANS SIMD decode (Phase 4), GPU entropy decode paths, and heterogeneous scheduling for decode. Decompression throughput should be tracked alongside compression in benchmarks.
