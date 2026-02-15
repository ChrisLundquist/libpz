# Competitive Roadmap Execution Plan (CPU-First)

**Created:** 2026-02-15  
**Status:** Planned  
**Priority:** P0  
**Owner:** Engineering team

## Problem

libpz has strong composability and promising GPU acceleration, but is not yet competitive with zstd-class CPU performance or nvCOMP-class GPU throughput. Current priorities are to:

1. Establish a repeatable baseline.
2. Optimize CPU path first (especially rANS/LZR and LZ/FSE path quality-speed balance).
3. Use GPU where it clearly wins on large workloads.
4. Preserve composability and pipeline modularity.

## Targets

### End-State Targets

1. CPU mode:
   - Reach within 10-20% of `zstd -1` encode throughput at similar ratio on mixed corpus.
2. GPU-assisted mode:
   - Achieve >=2x speedup vs libpz CPU mode on 1MB+ block workloads.
   - Reach >=60-70% of nvCOMP-class throughput on comparable batch-oriented tests.
3. Architectural:
   - Keep algorithm modules standalone and pipeline-composable.
   - Keep CPU/GPU output byte-identical per algorithm path.

### Guardrails

1. No correctness regressions (round-trip, validation corpus, cross CPU/GPU decode).
2. No pipeline format break without explicit versioning.
3. No optimization work proceeds without baseline data first.

## Baseline Protocol (Must Happen First)

### A. Corpus-Level Baseline (gzip comparison)

Use existing script and keep results in `docs/generated/` for each run.

```bash
./scripts/bench.sh -n 5 -p deflate,lzr,lzf
```

If needed, run pipeline-specific baselines:

```bash
./scripts/bench.sh -n 5 -p lzr
./scripts/bench.sh -n 5 -p lzf
./scripts/bench.sh -n 5 -p deflate
```

### B. Stage-Level Baseline (rANS-focused)

Use Criterion stage bench and filter to `rans` group.

```bash
cargo bench --bench stages -- rans
```

Capture encode/decode throughput and regression deltas before any rANS changes.

### C. Throughput Baseline (pipeline end-to-end, CPU)

```bash
cargo bench --bench throughput
```

### D. Baseline Artifacts

For each baseline cycle, record:

1. Git commit hash.
2. CPU model + thread count.
3. Command lines used.
4. Size ratio + encode/decode MB/s by pipeline (`deflate`, `lzr`, `lzf`).
5. Stage MB/s for `rans` encode/decode.

Store as dated markdown in `docs/generated/` (for example, `2026-02-15-cpu-baseline.md`).

## Execution Phases

### Phase 0: Measurement Hardening (Week 1)

**Goal:** Make baseline and comparison repeatable.

Tasks:

1. Define standard benchmark profile:
   - `ITERATIONS=5`, default corpora (`samples/cantrbry`, `samples/large`), fixed pipelines.
2. Add a simple benchmark results template in `docs/generated/`.
3. Document pass/fail thresholds:
   - Any >3% throughput regression or >1% ratio regression must be explained.

Acceptance criteria:

1. Two back-to-back runs show stable metrics (expected run-to-run variance bounded).
2. Baseline artifact committed and linked from this plan.

### Phase 1: CPU rANS Performance Pass (Weeks 1-3)

**Goal:** Improve LZR competitiveness by fixing known rANS CPU gaps first.

Tasks:

1. Wire and verify SIMD decode paths in runtime dispatch (SSE2/AVX2).
2. Validate reciprocal multiply path correctness and performance impact.
3. Re-run:
   - `cargo bench --bench stages -- rans`
   - `./scripts/bench.sh -n 5 -p lzr`
4. Add/expand tests for SIMD path parity (scalar vs SIMD output equivalence).

Acceptance criteria:

1. rANS decode throughput improvement is measurable and repeatable.
2. LZR end-to-end throughput improves with no ratio regression.
3. All correctness tests pass.

### Phase 2: CPU Pipeline Competitiveness Pass (Weeks 3-6)

**Goal:** Make CPU default paths materially stronger vs gzip and closer to zstd-class throughput/ratio tradeoffs.

Tasks:

1. Tune LZ77 parser cost model for LZF/LZR (quality/speed).
2. Re-evaluate auto-selection and trial mode thresholds using collected corpus stats.
3. Add focused benchmarks for 256KB, 1MB, 4MB blocks to reflect crossover behavior.
4. Benchmark against external baselines in existing throughput suite workflow.

Acceptance criteria:

1. `lzf` and/or `deflate` consistently beat gzip ratio and throughput on corpus.
2. `lzr` no longer materially lags `lzf` on decode-heavy workloads.
3. No degradation on small-input latency path.

### Phase 3: GPU Scheduling and Crossover Optimization (Weeks 6-9)

**Goal:** Only use GPU where it wins, and lower crossover point safely.

Tasks:

1. Refine scheduler heuristics (input size, batch size, device class, memory pressure).
2. Improve overlap in ring-buffer pipeline (submission/readback pacing).
3. Re-benchmark with:
   - `./scripts/bench.sh --webgpu -n 5 -p deflate,lzr,lzf`
   - `cargo bench --bench throughput --features webgpu`

Acceptance criteria:

1. GPU path shows clear gain at large blocks (1MB+).
2. No regressions on CPU fallback behavior.
3. GPU/CPU output equivalence tests remain green.

### Phase 4: nvCOMP-Style Batch Track (Weeks 9-12)

**Goal:** Improve real GPU batch throughput on independent-block workloads.

Tasks:

1. Define nvCOMP-style benchmark scenarios (many independent blocks, large payloads).
2. Optimize for persistent device residency where feasible between stages.
3. Compare libpz GPU throughput against internal CPU baseline and external GPU baseline methodology.

Acceptance criteria:

1. >=2x speedup over libpz CPU baseline on target batch scenarios.
2. Quantified progress toward nvCOMP-class throughput target.
3. No composability regressions in pipeline architecture.

## Risk Register

1. **Risk:** Chasing GPU micro-optimizations before CPU baseline maturity.
   - **Mitigation:** CPU-first gate; Phase 3 starts only after Phase 2 acceptance.
2. **Risk:** Benchmark noise causes false conclusions.
   - **Mitigation:** Fixed protocol, repeated runs, artifact logging.
3. **Risk:** Performance wins with hidden correctness debt.
   - **Mitigation:** Round-trip + validation + cross CPU/GPU checks required each phase.

## Immediate Next Actions

1. Run CPU baseline suite:
   - `./scripts/bench.sh -n 5 -p deflate,lzr,lzf`
   - `cargo bench --bench stages -- rans`
   - `cargo bench --bench throughput`
2. Save baseline artifact in `docs/generated/`.
3. Start Phase 1 implementation branch focused on rANS CPU decode performance.

## Execution Log

### 2026-02-15

1. Completed baseline runs:
   - `./scripts/bench.sh -n 5 -p deflate,lzr,lzf`
   - `cargo bench --bench stages -- rans`
2. Baseline report created:
   - `docs/generated/2026-02-15-cpu-baseline.md`
3. Reliability fix applied to `scripts/bench.sh`:
   - Use `cp -f` in decompression setup overwrite paths to avoid permission failures on read-only temp files.
4. `cargo bench --bench throughput` was started but intentionally interrupted; not included in baseline report.
5. Attempted CPU rANS micro-optimizations (single-stream encode/decode hot path) were benchmarked and then reverted.
   - Result quality was inconsistent under current run conditions (high variance and contradictory shifts).
   - Decision: keep baseline implementation unchanged until a clean, controlled benchmark environment is used for Phase 1 tuning.
6. Clean rerun after removing background workload:
   - `./scripts/bench.sh -n 5 -p lzr` => `pz-lzr` compression 28.9 MB/s, decompression 37.8 MB/s.
7. Used profiling harness directly (since `scripts/profile.sh`/samply failed in sandbox with `Unknown(1100)`):
   - `rans` encode: 184.0 MB/s
   - `rans` decode: 157.8 MB/s
   - `lzr` encode: 46.4 MB/s
   - `lzr` decode: 103.2 MB/s
   - `lz77` encode: 67.1 MB/s
   - `lz77` decode: 453.8 MB/s
8. Bottleneck callout: for current CPU 256KB workload, `lzr` encode appears more LZ77-limited than rANS-limited; prioritize LZ77 CPU optimization before further rANS micro-tuning.
9. Added benchmark control in `scripts/bench.sh`:
   - New option: `-t, --threads N` (passes `-t N` to `pz`).
   - Enables explicit single-thread (`-t 1`) vs auto-thread (`-t 0`) roadmap measurements.
10. Clean `lzr` thread-mode comparison (`n=3`, same corpus):
   - `-t 1`: encode 18.1 MB/s, decode 28.0 MB/s
   - `-t 0`: encode 17.1 MB/s, decode 34.5 MB/s
   - Takeaway: keep auto-thread decode as the primary competitive path; measure both modes in reports.
11. Implemented first accepted Phase 1 CPU optimization:
   - `LzDemuxer::Lz77` CPU `Auto/Lazy` fast path now demuxes directly from `lz77::Match` values (skips intermediate serialize-then-split byte buffer).
   - GPU and optimal parse paths unchanged.
12. Validation and measured impact:
   - Tests: `cargo test lzr` + `cargo test pipeline::tests` passed.
   - Profile harness (256KB):
     - `lzr` encode: 46.4 → 51.1 MB/s
     - `lzr` decode: 103.2 → 120.8 MB/s
   - Corpus benchmark (`./scripts/bench.sh -n 3 -p lzr -t 0`):
     - compression: 17.1 → 26.9 MB/s
     - decompression: 34.5 → 38.6 MB/s
13. Implemented second low-risk LZ77 hot-path improvement:
    - `find_match` / `find_top_k` switched to `hash3_unchecked` after existing bounds guards.
    - Removed unused checked hash helper.
14. Validation and measured impact (sequential):
    - Tests: `cargo test lz77` + `cargo test lzr` passed.
    - Profile harness (256KB):
      - `lz77` encode: 65.1 → 73.2 MB/s
      - `lzr` encode: 51.1 → 54.8 MB/s
    - Corpus benchmark (`./scripts/bench.sh -n 3 -p lzr -t 0`):
      - compression: 26.9 → 26.4 MB/s
      - decompression: 38.6 → 37.1 MB/s
    - Takeaway: stage-level gain is clear; end-to-end impact is small/noisy and should be rechecked with repeated runs.
15. Reverted unsafe hash optimization on request and remeasured:
    - Restored safe `hash3` in `src/lz77.rs` and removed `hash3_unchecked` usage.
    - Sequential measurements:
      - `lz77` encode: 73.2 → 72.0 MB/s
      - `lzr` encode: 54.8 → 50.5 MB/s
      - `bench.sh -n 3 -p lzr -t 0`: compression 26.4 → 24.7 MB/s, decompression 37.1 → 37.9 MB/s
    - Decision: keep safe code path and continue with non-unsafe optimization work.
16. New Samply-driven LZ77 hotspot pass:
    - Captured fresh profile: `profiling/e5b3290-dirty/lz77_encode_1MB.json.gz`.
    - Although saved JSON remained unsymbolicated, frame addresses were mapped to symbols via `nm -n target/profiling/examples/profile`.
    - Dominant hotspots identified: `pz::simd::compare_bytes_avx2` and `pz::lz77::HashChainFinder::find_match`.
17. Implemented hotspot-targeted safe optimizations in `src/lz77.rs`:
    - Added best-length probe prefilter in `find_match` to skip futile SIMD compares.
    - Added early chain-exit when no longer possible to improve useful match length.
    - Replaced `% MAX_WINDOW` with `& WINDOW_MASK` ring indexing in hot loops.
18. Validation and measurements after patch:
    - `cargo test lz77` passed.
    - `cargo test lzr` passed.
    - `cargo bench --bench stages_lz77` showed statistically significant throughput improvements across all tested sizes.
    - `./scripts/bench.sh -n 3 -p lzr -t 0` summary:
      - `pz-lzr` compression: 26.1 MB/s
      - `pz-lzr` decompression: 37.3 MB/s
19. Executed Objective 3 (parse-mode chain-depth tuning on CPU):
    - Added parse-mode-aware chain limit selection in `src/lz77.rs`:
      - `Auto` on large inputs uses reduced chain depth (`MAX_CHAIN_AUTO=48`).
      - `Lazy` keeps full chain depth (`MAX_CHAIN=64`).
    - Wired through CPU LZ77 backend and demux fast path via new helper in `src/pipeline/mod.rs`.
20. Executed Objective 4 (compare-loop overhead reduction):
    - Added `Dispatcher::compare_bytes_ptr` in `src/simd.rs`.
    - Updated `HashChainFinder::find_match` and `find_top_k` to use precomputed pointers and compare limit, removing per-candidate slice/min setup overhead.
21. Attempts and outcome notes:
    - Two intermediate safe optimizations (short AVX2 branch path and hash-collision triple-byte gate) were benchmarked and showed regressions; both were removed.
    - Kept only the pointer-compare optimization for Objective 4.
22. Validation and benchmarks:
    - Tests: `cargo test lz77`, `cargo test lzr` passed.
    - `cargo bench --bench stages_lz77` (pointer-compare run):
      - `compress_lazy` improved across tested sizes (notably 64KB and 4MB).
      - `decompress` mostly neutral to improved depending on size.
    - `./scripts/bench.sh -n 3 -p lzr -t 0` after Objective 3+4:
      - `pz-lzr` compression: 26.7 MB/s
      - `pz-lzr` decompression: 36.4 MB/s
      - ratio unchanged at 40.6% (Canterbury set in script).
