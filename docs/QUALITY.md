# Quality Status

**Last Updated:** 2026-02-14
**Owner:** Engineering team

## Purpose

Tracks quality grades for each module and feature in libpz. Grades reflect actual test coverage, validation status, and performance benchmarks - not aspirations.

## Grading Criteria

Each module is graded on five dimensions:

- **Correctness** - All tests pass, validation corpus tests pass
- **Test Coverage** - Unit tests, round-trip tests, edge cases, cross-decompression
- **GPU Implementation** - GPU acceleration status (Full/Partial/None/N/A)
- **Performance** - Benchmark results vs gzip and CPU baseline
- **Documentation** - API docs, design docs, examples

**Grades:**
- ✅ **A** - Excellent (production-ready)
- ⚠️ **B** - Good (minor gaps)
- ⚠️ **C** - Fair (notable gaps, usable)
- ❌ **D** - Poor (significant issues)
- ❌ **F** - Failing (not functional)

## Core Algorithms

### lz77 (LZ77 Compression)
**Overall Grade:** ✅ A

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ✅ 95%+ lines covered
  - Unit tests: empty, small, large inputs
  - Round-trip tests: all variants (brute, hashchain, lazy, parallel, GPU)
  - Validation corpus: Canterbury + large corpus
- **GPU:** ✅ Full GPU acceleration + CPU fallback
  - Hash-table kernel (primary)
  - Batch/per-position kernels (legacy)
  - Top-K kernel for optimal parsing
  - Break-even: ~256KB blocks
- **Performance:** ✅ Beats gzip on 256KB+ blocks
  - Throughput: 150-200 MB/s (CPU), 300-500 MB/s (GPU batched)
- **Documentation:** ⚠️ B - Missing optimal parsing explanation
  - API docs complete
  - GPU kernels documented
  - Optimal parsing (backward DP) needs design doc

**Gaps:** Optimal parsing design doc

---

### huffman (Huffman Coding)
**Overall Grade:** ✅ A

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ✅ 95%+ lines covered
  - Canonical Huffman encoding/decoding
  - GPU two-pass encoding (Blelloch prefix sum)
  - Edge cases: single symbol, all zeros
- **GPU:** ✅ Full GPU encode, ⚠️ CPU-only decode
  - GPU encode: two-pass (count + encode)
  - GPU decode: TODO (sync decode path, see exec-plans/active/TODO-huffman-sync-decode.md)
  - Break-even: ~128KB blocks
- **Performance:** ✅ Near-optimal entropy coding
  - Encode: 200+ MB/s (CPU), 400+ MB/s (GPU)
  - Decode: 300+ MB/s (CPU only)
- **Documentation:** ✅ API docs and design complete

**Gaps:** GPU decode not implemented (low priority, CPU decode is fast)

---

### fse (Finite State Entropy)
**Overall Grade:** ✅ A

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ✅ 90%+ lines covered
  - tANS encoding/decoding
  - Frequency table normalization
  - Round-trip tests
- **GPU:** ❌ CPU-only (table lookups not GPU-friendly)
- **Performance:** ✅ Matches FSE reference implementation
  - Encode/decode: 150-250 MB/s
  - Better compression than Huffman (1-2%)
- **Documentation:** ✅ Complete

**Gaps:** None (GPU not applicable for FSE table lookups)

---

### rans (Range ANS)
**Overall Grade:** ⚠️ B

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ✅ 90%+ lines covered
- **GPU:** ❌ CPU-only
- **Performance:** ⚠️ B - Slower than FSE/Huffman
  - Encode: 80-120 MB/s
  - Decode: 100-150 MB/s
  - SIMD decode not wired (see ARCHITECTURE.md)
- **Documentation:** ✅ Complete

**Gaps:**
- SIMD decode paths (SSE2 4-way, AVX2 8-way) declared but not wired
- Reciprocal multiplication for GPU (documented as future optimization)

---

### bwt (Burrows-Wheeler Transform)
**Overall Grade:** ⚠️ B

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ✅ 95%+ lines covered
  - CPU: SA-IS (linear time)
  - GPU: Radix sort + prefix-doubling
  - Cross-decompression: GPU BWT → CPU decode, vice versa
- **GPU:** ⚠️ Partial - GPU slower than CPU at all sizes
  - GPU uses radix sort (not SA-IS)
  - 7-14x faster than previous bitonic sort
  - Still slower than CPU SA-IS
- **Performance:** ⚠️ C - GPU not competitive
  - CPU: 50-80 MB/s (SA-IS)
  - GPU: 20-40 MB/s (radix sort)
  - Break-even: never (GPU always slower)
- **Documentation:** ✅ Complete (ARCHITECTURE.md has detailed notes)

**Gaps:**
- GPU BWT needs algorithmic improvement (SA-IS port or better radix sort)
- Current GPU path kept for research, not production use

---

### mtf (Move-to-Front)
**Overall Grade:** ✅ A

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ✅ 90%+ lines covered
- **GPU:** ❌ N/A (inherently sequential)
- **Performance:** ✅ Fast enough (MTF is cheap)
  - 200+ MB/s
- **Documentation:** ✅ Complete

**Gaps:** None

---

### rle (Run-Length Encoding)
**Overall Grade:** ✅ A

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ✅ 90%+ lines covered
- **GPU:** ❌ N/A (simple scan, not worth GPU overhead)
- **Performance:** ✅ Very fast (300+ MB/s)
- **Documentation:** ✅ Complete

**Gaps:** None

---

### optimal (Optimal LZ77 Parsing)
**Overall Grade:** ⚠️ B

- **Correctness:** ✅ All validation tests pass
- **Test Coverage:** ⚠️ B - 80%+ lines covered
  - Backward DP cost model
  - GPU top-K match table → CPU DP
- **GPU:** ✅ Partial (top-K on GPU, DP on CPU)
- **Performance:** ✅ 4-6% better compression than greedy
  - Throughput: 30-50% of greedy speed (expected trade-off)
- **Documentation:** ❌ D - Missing design doc

**Gaps:**
- No design doc explaining backward DP algorithm
- Cost model tuning not documented

---

## Pipelines

### deflate (LZ77 + Huffman)
**Overall Grade:** ✅ A

- **Correctness:** ✅ All validation tests pass
- **Multi-stream:** ✅ 16% better compression than single-stream
- **GPU:** ✅ Full GPU pipeline (LZ77→Huffman on device)
- **Performance:** ✅ Beats gzip on ratio and speed
- **Documentation:** ✅ Complete

**Gaps:** None

---

### lzf (LZ77 + FSE)
**Overall Grade:** ✅ A

- **Correctness:** ✅ All validation tests pass
- **Multi-stream:** ✅ 17.6% better compression than single-stream
- **GPU:** ✅ GPU LZ77, CPU FSE (FSE not GPU-friendly)
- **Performance:** ✅ Best compression ratio of all pipelines
- **Documentation:** ✅ Complete

**Gaps:** None

---

### lzr (LZ77 + rANS)
**Overall Grade:** ⚠️ B

- **Correctness:** ✅ All validation tests pass
- **Multi-stream:** ✅ Similar to Lzf
- **GPU:** ⚠️ Partial (GPU LZ77, CPU rANS)
- **Performance:** ⚠️ B - Slower than Lzf/Deflate due to rANS decode
- **Documentation:** ✅ Complete

**Gaps:** rANS SIMD decode not wired

---

### bw (BWT + MTF + RLE + FSE)
**Overall Grade:** ⚠️ B

- **Correctness:** ✅ All validation tests pass
- **GPU:** ⚠️ Partial (BWT GPU slower than CPU)
- **Performance:** ⚠️ C - Slow due to BWT (200-400ms for 1MB)
- **Documentation:** ✅ Complete

**Gaps:** BWT GPU performance (see bwt section)

---

## Infrastructure

### pipeline (Container Format)
**Overall Grade:** ✅ A

- **V2 Format:** ✅ Block-parallel, pipeline-parallel, multi-stream
- **Auto-selection:** ✅ Heuristic and trial-based selection
- **Parallel paths:** ✅ Block-parallel, pipeline-parallel, GPU-batched
- **Documentation:** ✅ Complete (design-docs/pipeline-architecture.md)

**Gaps:** None

---

### webgpu (GPU Backend)
**Overall Grade:** ✅ A

- **Device Init:** ✅ Robust fallback to CPU
- **Memory Management:** ✅ Ring buffer with backpressure
- **Kernels:** ✅ LZ77, Huffman encode, BWT radix sort
- **Testing:** ✅ Cross-decompression validation
- **Documentation:** ✅ Complete (design-docs/gpu-batching.md)

**Gaps:** None

---

### validation (Cross-Module Tests)
**Overall Grade:** ✅ A

- **Corpus Tests:** ✅ Canterbury + large corpus (13.3 MB)
- **Cross-Decompression:** ✅ GPU↔CPU validation for all algorithms
- **Coverage:** ✅ All pipelines, all algorithms
- **Documentation:** ✅ Complete

**Gaps:** None

---

### analysis (Data Profiling)
**Overall Grade:** ✅ A

- **Metrics:** ✅ Entropy, autocorrelation, match density, run ratio
- **Sampling:** ✅ Fast sampling for large inputs
- **Pipeline Selection:** ✅ Heuristic and trial-based
- **Documentation:** ✅ Complete

**Gaps:** None

---

## Tooling

### CLI (`pz`)
**Overall Grade:** ✅ A

- **Features:** ✅ Auto-selection, trial mode, all pipelines
- **Testing:** ✅ Tested on all validation corpus files
- **Documentation:** ✅ README.md

**Gaps:** None

---

### C FFI
**Overall Grade:** ⚠️ B

- **API:** ✅ Compress/decompress bindings
- **Testing:** ⚠️ Basic tests only
- **Documentation:** ⚠️ Minimal

**Gaps:**
- No comprehensive FFI test suite
- No example C programs

---

### Benchmarks (Criterion)
**Overall Grade:** ✅ A

- **Coverage:** ✅ All algorithms, all pipelines
- **Metrics:** ✅ Throughput, compression ratio
- **GPU:** ✅ CPU and GPU paths benchmarked
- **Documentation:** ✅ CLAUDE.md

**Gaps:** None

---

### Scripts
**Overall Grade:** ✅ A

- **bench.sh:** ✅ pz vs gzip comparison
- **profile.sh:** ✅ samply profiling wrapper
- **gpu-meminfo.sh:** ✅ GPU memory analysis
- **trace-pipeline.sh:** ✅ Flow diagram generator
- **Documentation:** ✅ CLAUDE.md

**Gaps:** None

---

## Known Issues

See `exec-plans/tech-debt-tracker.md` for the full prioritized list of known gaps and their action items.

## Summary Statistics

- **Total tests:** 651 passing, 0 failing
- **Overall quality:** 11/12 milestones complete
- **GPU readiness:** 80% (LZ77, Huffman encode fully GPU-accelerated)
- **Documentation coverage:** 90% (missing optimal parsing, some FFI docs)

---

**Next quality review:** After M5.3 (fuzz testing) completion
