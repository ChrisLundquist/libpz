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
- `src/frequency.rs` — shared frequency table used by entropy coders
- `src/opencl.rs` — OpenCL GPU backend (feature-gated behind `opencl`)
- `src/ffi.rs` — C FFI bindings
- `src/validation.rs` — cross-module integration tests
- `kernels/*.cl` — OpenCL kernel source (lz77.cl, lz77_batch.cl, lz77_topk.cl, lz77_hash.cl, bwt_sort.cl, huffman_encode.cl)
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
- **GPU kernels:** LZ77 hash-table (fast), LZ77 batch/per-position (legacy), LZ77 top-K, BWT bitonic sort, Huffman encode (two-pass with Blelloch prefix sum), GPU Deflate chaining (LZ77→Huffman on device)
- **Tooling:** CLI (`pz` with `-a`/`--auto` and `--trial` flags), C FFI, Criterion benchmarks, CI (3 OS + OpenCL)

### BWT implementation
- **CPU:** Uses SA-IS (Suffix Array by Induced Sorting) — O(n) linear time via doubled-text-with-sentinel strategy. Replaced earlier O(n log²n) naive sort.
- **GPU:** Still uses bitonic sort for suffix array construction — O(n log²n) with many kernel launches. Significantly slower than CPU SA-IS at all tested sizes. The GPU BWT path is not used in the default pipeline.

### GPU stage chaining
The Deflate GPU path now chains LZ77 → Huffman on the GPU:
1. GPU: LZ77 hash-table kernel → download LZ77 output
2. CPU: build Huffman tree from LZ77 output (fast)
3. GPU: Huffman encode with Blelloch prefix sum (no host round-trip for scan)
4. CPU: download final encoded data

This is activated automatically when `Backend::OpenCl` is selected and input ≥ `MIN_GPU_INPUT_SIZE`.

### Parallel LZ77
`compress_lazy_parallel(input, num_threads)` pre-computes matches in parallel (each thread builds its own hash chain), then serializes sequentially with lazy evaluation. Thresholds:
- `MIN_PARALLEL_SIZE = 256KB` — below this, single-threaded is faster
- `MIN_SEGMENT_SIZE = 128KB` — caps thread count to amortize hash chain warmup

### Not started
- M5.3: Fuzz testing (`cargo-fuzz`)
- M12: Vulkan compute backend

### Known issue
- `test_gpu_bwt_all_same` fails (pre-existing): GPU BWT produces wrong
  primary_index for all-same-byte inputs due to bitonic sort tie-breaking

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

### BWT GPU (bitonic sort — not recommended, SA-IS CPU is faster)

| Size | GPU bitonic | CPU SA-IS |
|------|------------|-----------|
| 1KB  | 30ms (33 KiB/s) | ~3µs |
| 10KB | 42ms (240 KiB/s) | ~50µs |
| 64KB | 63ms (1 MiB/s) | ~1ms |

GPU BWT is 30-10,000x slower than CPU SA-IS. The bitonic sort approach has O(log²n) kernel launches with host sync between steps, making it fundamentally unsuitable. CPU SA-IS is O(n) and dominates at all sizes.

## Remaining GPU bottlenecks

1. **Bitonic sort for BWT** — O(log²n) kernel launches per doubling step, with
   host↔device sync between steps. CPU SA-IS is O(n) and vastly faster.
   The GPU BWT path exists but should not be used; it remains as a reference
   implementation.

2. **No shared memory usage** — LZ77 hash kernel uses only global memory.
   Loading hash buckets into `__local` memory could help at larger sizes.

3. **Hash bucket overflow** — Fixed BUCKET_CAP=64 means highly repetitive data
   may miss good matches. Adaptive bucket sizing could help.

4. **Huffman WriteCodes atomic contention** — Per-bit atomic_or on the output
   buffer limits scaling. Chunk-based packing could reduce contention.

## Next steps

### Priority 1: Optimize GPU BWT or remove it
- The GPU BWT bitonic sort path is 10,000x slower than CPU SA-IS at small sizes
- Options: (a) implement GPU-friendly radix sort, (b) adapt SA-IS for GPU, or (c) remove the GPU BWT path entirely and always use CPU SA-IS
- Fix `test_gpu_bwt_all_same` bug if keeping the GPU path

### Priority 2: Use local/shared memory in hash kernel
- Load hash buckets into `__local` memory for faster repeated access
- Could improve GPU LZ77 performance at mid-range sizes (64KB-256KB)

### Priority 3: Benchmark at even larger sizes
- Add 4MB, 16MB tiers — GPU advantage should grow with size
- Profile Deflate chained path at larger sizes

### Priority 4: Fuzz testing
- Set up `cargo-fuzz` for round-trip correctness on all pipelines
- Target edge cases in LZ77, Huffman, BWT decode paths
