# CLAUDE.md — libpz development guide

## Build & test commands
```bash
cargo build                    # compile
cargo test                     # run all tests
cargo test fse                 # run tests for a specific module
cargo fmt                      # format code (must match CI)
cargo clippy --all-targets     # lint (must pass with zero warnings)
cargo build --features opencl  # compile with GPU support
cargo test --features opencl   # run all tests including GPU (skips gracefully if no device)
cargo bench                    # run CPU benchmarks
cargo bench --features opencl  # run CPU + GPU benchmarks
```

## Git hooks setup
Enable the pre-commit hook (auto-formats and lints before each commit):
```bash
git config core.hooksPath .githooks
```

## Pre-commit checklist
Before every commit, **always** run (the pre-commit hook handles 1 and 2 automatically):
1. `cargo fmt` — format all code
2. `cargo clippy --all-targets` — fix all warnings, zero warnings policy
3. `cargo test` — all tests must pass

## Project layout
- `src/lib.rs` — crate root, module declarations
- `src/{algorithm}.rs` — one file per composable algorithm (bwt, deflate, fse, huffman, lz77, mtf, rangecoder, rle)
- `src/analysis.rs` — data profiling for pipeline auto-selection (entropy, match density, etc.)
- `src/optimal.rs` — optimal LZ77 parsing via backward DP (match table + cost model)
- `src/pipeline.rs` — multi-stage compression pipelines (Deflate, Bw, Lza), auto-selection
- `src/frequency.rs` — shared frequency table used by entropy coders (SIMD-accelerated counting)
- `src/simd.rs` — SIMD-accelerated primitives (SSE2/AVX2 for x86_64, NEON stubs for aarch64)
- `src/opencl.rs` — OpenCL GPU backend (feature-gated behind `opencl`)
- `src/ffi.rs` — C FFI bindings
- `src/validation.rs` — cross-module integration tests
- `kernels/*.cl` — OpenCL kernel source (lz77.cl, lz77_batch.cl, lz77_topk.cl, lz77_hash.cl, bwt_radix.cl, bwt_rank.cl, bwt_sort.cl, huffman_encode.cl)
- `benches/throughput.rs` — end-to-end pipeline benchmarks vs gzip/pigz/zstd, parallel scaling
- `benches/stages.rs` — per-algorithm scaling benchmarks, GPU vs CPU comparisons

## Conventions
- Public API: `encode()` / `decode()` returning `PzResult<T>`
- Provide `_to_buf` variants for caller-allocated output buffers
- Algorithms must be composable and standalone — not tied to a specific pipeline
- GPU-friendly: prefer table-driven, branchless designs (no data-dependent divisions)
- Error type: `PzError` (InvalidInput, BufferTooSmall, Unsupported, InternalError)
- Tests go in `#[cfg(test)] mod tests` at the bottom of each module file

## Project status (as of 2026-02-11)

### Completed (11/12 milestones)
- **Algorithms:** LZ77 (brute, hashchain, lazy, parallel), Huffman, BWT (SA-IS), MTF, RLE, RangeCoder, FSE
- **Pipelines:** Deflate (LZ77+Huffman), Bw (BWT+MTF+RLE+RC), Lza (LZ77+RC)
- **Auto-selection:** Heuristic (`select_pipeline`) and trial-based (`select_pipeline_trial`) pipeline selection using data analysis (entropy, match density, run ratio, autocorrelation)
- **Data analysis:** `src/analysis.rs` — statistical profiling (Shannon entropy, autocorrelation, run ratio, match density, distribution shape) with sampling support
- **Optimal parsing:** GPU top-K match table → CPU backward DP (4-6% better compression)
- **Multi-threading:** Block-parallel and pipeline-parallel via V2 container format; within-block parallel LZ77 match finding (`compress_lazy_parallel`)
- **GPU kernels:** LZ77 hash-table (fast), LZ77 batch/per-position (legacy), LZ77 top-K, BWT radix sort + parallel rank assignment, Huffman encode (two-pass with Blelloch prefix sum), GPU Deflate chaining (LZ77→Huffman on device)
- **Tooling:** CLI (`pz` with `-a`/`--auto` and `--trial` flags), C FFI, Criterion benchmarks, CI (3 OS + OpenCL)

### BWT implementation
- **CPU:** Uses SA-IS (Suffix Array by Induced Sorting) — O(n) linear time via doubled-text-with-sentinel strategy.
- **GPU:** Uses LSB-first 8-bit radix sort with prefix-doubling for suffix array construction. Replaced earlier bitonic sort (PR #21). Features adaptive key width (skip zero-digit radix passes) and event chain batching (one host sync per doubling step). Rank assignment runs on GPU via Blelloch prefix sum + scatter. Still slower than CPU SA-IS at all sizes but dramatically improved from bitonic sort (7-14x faster). The GPU uses circular comparison `(sa[i]+k) % n` vs CPU SA-IS's doubled-text approach — both produce valid BWTs that round-trip correctly.

### SIMD acceleration
`src/simd.rs` provides runtime-dispatched SIMD for CPU hot paths:
- **Byte frequency counting** — 4-bank histogramming with AVX2 merge, integrated into `FrequencyTable::count()`
- **LZ77 match comparison** — SSE2 (16 bytes/cycle) or AVX2 (32 bytes/cycle) `compare_bytes`, integrated into `HashChainFinder::find_match()` and `find_top_k()`
- **u32 array summation** — widened u64 accumulator lanes for overflow-safe SIMD sum

| Architecture | Baseline | Extended | Status |
|-------------|----------|----------|--------|
| x86_64      | SSE2     | AVX2     | Implemented + integrated |
| aarch64     | NEON     | SVE      | Stubs (dispatch to scalar) |

Runtime detection via `Dispatcher::new()` caches the best ISA level at first call. All SIMD implementations are verified against scalar reference in tests.

### GPU stage chaining
The Deflate GPU path chains LZ77 → Huffman on the GPU with minimized transfers:
1. GPU: LZ77 hash-table kernel → download match array → CPU dedupe + serialize
2. GPU: upload LZ77 bytes once → `ByteHistogram` kernel → download only 256×u32 (1KB)
3. CPU: build Huffman tree from histogram, produce code LUT
4. GPU: Huffman encode (reusing LZ77 buffer) with Blelloch prefix sum
5. GPU: download final encoded bitstream

The `ByteHistogram` kernel eliminates the need to scan LZ77 data on CPU for frequency counting — only 1KB of histogram data is transferred instead of the full LZ77 stream.

This is activated automatically when `Backend::OpenCl` is selected and input ≥ `MIN_GPU_INPUT_SIZE`.

### Parallel LZ77
`compress_lazy_parallel(input, num_threads)` pre-computes matches in parallel (each thread builds its own hash chain), then serializes sequentially with lazy evaluation. Thresholds:
- `MIN_PARALLEL_SIZE = 256KB` — below this, single-threaded is faster
- `MIN_SEGMENT_SIZE = 128KB` — caps thread count to amortize hash chain warmup

### Not started
- M5.3: Fuzz testing (`cargo-fuzz`)
- M12: Vulkan compute backend

## GPU benchmark results (AMD gfx1201 / RDNA3)

### LZ77 GPU

| Size | CPU hashchain | CPU lazy | GPU hash | GPU vs CPU hashchain |
|------|--------------|----------|----------|---------------------|
| 1KB  | 14µs (71 MiB/s) | 6µs (164 MiB/s) | 1.1ms (1 MiB/s) | 65x slower |
| 10KB | 57µs (171 MiB/s) | 42µs (231 MiB/s) | 1.4ms (7 MiB/s) | 20x slower |
| 64KB | 1.3ms (48 MiB/s) | 611µs (102 MiB/s) | 1.7ms (36 MiB/s) | 1.3x slower |
| 256KB | 6.2ms (40 MiB/s) | 2.6ms (97 MiB/s) | 3.4ms (74 MiB/s) | **2x faster** |
| 1MB | 20ms (50 MiB/s) | 16.7ms (60 MiB/s) | 9.0ms (111 MiB/s) | **2x faster** |

### Huffman GPU (encode only)

| Size | CPU | GPU + CPU scan | GPU + GPU scan | Best GPU vs CPU |
|------|-----|----------------|----------------|-----------------|
| 10KB | 23µs (418 MiB/s) | 312µs (31 MiB/s) | 518µs (19 MiB/s) | CPU 13x faster |
| 64KB | 432µs (145 MiB/s) | 926µs (68 MiB/s) | 1.45ms (43 MiB/s) | CPU 2x faster |
| 256KB | 1.85ms (135 MiB/s) | 999µs (250 MiB/s) | **543µs (460 MiB/s)** | **GPU 3.4x faster** |

GPU Huffman with Blelloch prefix sum crosses over ~128KB. At 256KB the GPU scan path is 3.4x faster than CPU.

### Deflate chained (GPU LZ77 → GPU Huffman)

| Size | CPU 1-thread | GPU chained | Speedup |
|------|-------------|-------------|---------|
| 64KB | 1.63ms (38 MiB/s) | 3.01ms (21 MiB/s) | CPU 1.8x faster |
| 256KB | 6.06ms (41 MiB/s) | 4.93ms (51 MiB/s) | **GPU 1.2x faster** |
| 1MB | 23.5ms (43 MiB/s) | 18.3ms (55 MiB/s) | **GPU 1.3x faster** |

### BWT GPU (radix sort)

| Size | GPU radix | Throughput | Old bitonic | Speedup vs bitonic |
|------|-----------|------------|-------------|--------------------|
| 1KB  | 3.4ms | 295 KiB/s | 23ms | **6.8x** |
| 10KB | 5.9ms | 1.6 MiB/s | 42ms | **7.1x** |
| 64KB | 4.1ms | 15.3 MiB/s | 56ms | **13.7x** |
| 256KB | 11.6ms | 21.6 MiB/s | — | — |
| 4MB | 333ms | 12.0 MiB/s | — | — |
| 16MB | 1.73s | 9.2 MiB/s | — | — |

GPU BWT radix sort is 7-14x faster than the old bitonic sort. Still slower than CPU SA-IS at small sizes but becoming competitive at 64KB+ (CPU SA-IS ~1ms at 64KB vs GPU 4.1ms). The gap narrows at larger sizes where GPU parallelism helps more.

## Remaining GPU bottlenecks

1. **GPU BWT still slower than CPU SA-IS** — Radix sort improved 7-14x over bitonic
   sort, but CPU SA-IS (O(n)) remains faster at small/medium sizes. GPU catches up
   at 64KB+ but prefix-doubling's O(n log n) work is inherently more than SA-IS's O(n).

2. **No shared memory usage** — LZ77 hash kernel uses only global memory.
   Loading hash buckets into `__local` memory could help at larger sizes.

3. **Hash bucket overflow** — Fixed BUCKET_CAP=64 means highly repetitive data
   may miss good matches. Adaptive bucket sizing could help.

4. **Huffman WriteCodes atomic contention** — Per-bit atomic_or on the output
   buffer limits scaling. Chunk-based packing could reduce contention.

5. **LZ77 match array still downloaded for dedupe** — GPU match dedup is sequential
   and runs on CPU. Keeping serialized LZ77 bytes on GPU for histogram+Huffman
   is already done (ByteHistogram optimization), but the match download is unavoidable.

## Next steps

### Priority 1: Use local/shared memory in LZ77 hash kernel
- Load hash buckets into `__local` memory for faster repeated access
- Could improve GPU LZ77 performance at mid-range sizes (64KB-256KB)
- May lower the GPU crossover point from 256KB toward 64-128KB

### Priority 3: Chunk-based Huffman bit packing
- Replace per-bit `atomic_or` in WriteCodes with work-group-local packing
- Each work-group packs its chunk into local memory (no atomics within WG)
- Single copy from local → global per chunk
- Could 5-10x GPU Huffman throughput

### Priority 4: Fuzz testing
- Set up `cargo-fuzz` for round-trip correctness on all pipelines
- Target edge cases in LZ77, Huffman, BWT decode paths

### Priority 5: Auto-selection threshold tuning
- Run all 3 pipelines on Canterbury + Silesia corpora
- Measure actual compression ratios vs analysis metrics
- Tune heuristic decision tree thresholds empirically

### Priority 6: aarch64 NEON/SVE SIMD implementation
- Replace scalar stubs in `src/simd.rs` with actual NEON intrinsics
- `compare_bytes`: `vceqq_u8` + `vmovn_u16` for 16-byte comparison
- `byte_frequencies`: 4-bank unrolled (NEON lacks efficient gather/scatter)
- `sum_u32`: `vld1q_u32` + `vaddq_u32` + `vaddvq_u32`
- SVE for ARMv8.2+ (variable-length vectors, predicated operations)
- Requires aarch64 hardware for benchmarking
