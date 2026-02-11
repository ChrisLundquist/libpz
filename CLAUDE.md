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
- `src/optimal.rs` — optimal LZ77 parsing via backward DP (match table + cost model)
- `src/pipeline.rs` — multi-stage compression pipelines (Deflate, Bw, Lza)
- `src/frequency.rs` — shared frequency table used by entropy coders
- `src/opencl.rs` — OpenCL GPU backend (feature-gated behind `opencl`)
- `src/ffi.rs` — C FFI bindings
- `src/validation.rs` — cross-module integration tests
- `kernels/*.cl` — OpenCL kernel source (lz77.cl, lz77_batch.cl, lz77_topk.cl, bwt_sort.cl)
- `benches/throughput.rs` — end-to-end pipeline benchmarks vs gzip/pigz/zstd
- `benches/stages.rs` — per-algorithm scaling benchmarks (1KB/10KB/64KB)

## Conventions
- Public API: `encode()` / `decode()` returning `PzResult<T>`
- Provide `_to_buf` variants for caller-allocated output buffers
- Algorithms must be composable and standalone — not tied to a specific pipeline
- GPU-friendly: prefer table-driven, branchless designs (no data-dependent divisions)
- Error type: `PzError` (InvalidInput, BufferTooSmall, Unsupported, InternalError)
- Tests go in `#[cfg(test)] mod tests` at the bottom of each module file

## Project status (as of 2026-02-10)

### Completed (11/12 milestones)
- **Algorithms:** LZ77 (brute, hashchain, lazy), Huffman, BWT, MTF, RLE, RangeCoder, FSE
- **Pipelines:** Deflate (LZ77+Huffman), Bw (BWT+MTF+RLE+RC), Lza (LZ77+RC)
- **Optimal parsing:** GPU top-K match table → CPU backward DP (4-6% better compression)
- **Multi-threading:** Block-parallel and pipeline-parallel via V2 container format
- **GPU kernels:** LZ77 per-position, LZ77 batch, LZ77 top-K, BWT bitonic sort
- **Tooling:** CLI (`pz`), C FFI, Criterion benchmarks, CI (3 OS + OpenCL)

### Blocked: M8 — GPU must outperform CPU on large inputs
The GPU backend is functionally correct but slower than CPU at all tested sizes.
Root causes (see "GPU bottlenecks" below) are algorithmic, not infrastructure.

### Not started
- M5.3: Fuzz testing (`cargo-fuzz`)
- M12: Vulkan compute backend

## GPU bottlenecks (why GPU is currently slower)

1. **Brute-force LZ77 search** — GPU kernels scan the entire 32KB window linearly
   per position (O(n*w) total). CPU hash-chain is O(n) amortized. The GPU does ~1000x
   more comparisons, relying on parallelism to compensate — but it can't.

2. **Bitonic sort for BWT** — O(log^2 n) kernel launches per doubling step, with
   host↔device sync between steps. At 64KB: ~4000 kernel dispatches.

3. **No shared memory usage** — Kernels use only global memory. Loading the sliding
   window into `__local` memory would cut bandwidth cost significantly.

4. **Benchmarks too small** — Only tested at 1KB-64KB. GPU overhead (PCIe latency,
   kernel launch) dominates. Need 256KB-16MB tests to find the crossover point.

5. **Host round-trips between stages** — Each pipeline stage transfers data back to
   host. Chaining GPU stages would eliminate redundant transfers.

## Next steps to achieve GPU-accelerated compression

### Priority 1: Fix the LZ77 kernel algorithm
- Build a hash table on the GPU (parallel hash construction is well-studied)
- Search via hash chains instead of brute-force window scan
- This alone could yield 10-100x improvement in kernel efficiency
- Reference: nvcomp's approach, GPU LZSS papers (Ozsoy & Swany 2011)

### Priority 2: Benchmark at realistic sizes
- Add 256KB, 1MB, 4MB, 16MB size tiers to `benches/stages.rs`
- Find the actual CPU/GPU crossover point
- The current 64KB ceiling may be below the break-even size

### Priority 3: Optimize BWT GPU path
- Replace bitonic sort with GPU parallel radix sort (fewer kernel launches)
- Or adapt SA-IS (linear time suffix array) for GPU
- Move rank assignment to GPU to avoid host↔device round-trips per step

### Priority 4: Use local/shared memory in kernels
- Load sliding window chunks into `__local` memory
- Reduces global memory bandwidth by 10-100x for repeated access patterns

### Priority 5: Chain GPU stages
- Keep data on device between LZ77 → Huffman encode (Deflate pipeline)
- Keep data on device between BWT → MTF (Bw pipeline, though MTF is sequential)
- Eliminate redundant host↔device transfers between pipeline stages
