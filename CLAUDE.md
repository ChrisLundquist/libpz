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
- **GPU kernels:** LZ77 hash-table (fast), LZ77 batch/per-position (legacy), LZ77 top-K, BWT bitonic sort
- **Tooling:** CLI (`pz`), C FFI, Criterion benchmarks, CI (3 OS + OpenCL)

### M8 progress: GPU LZ77 now outperforms CPU at 256KB+
Hash-table GPU kernel (`lz77_hash.cl`) achieves 2x over CPU hashchain at 1MB.
Crossover point is ~128KB on AMD RDNA3. BWT GPU path still slower than CPU.

### Not started
- M5.3: Fuzz testing (`cargo-fuzz`)
- M12: Vulkan compute backend

### Known issue
- `test_gpu_bwt_all_same` fails (pre-existing): GPU BWT produces wrong
  primary_index for all-same-byte inputs due to bitonic sort tie-breaking

## GPU benchmark results (AMD gfx1201 / RDNA3)

| Size | CPU hashchain | CPU lazy | GPU hash | GPU vs CPU hashchain |
|------|--------------|----------|----------|---------------------|
| 1KB  | 14µs (71 MiB/s) | 6µs (164 MiB/s) | 885µs (1.1 MiB/s) | 65x slower |
| 10KB | 57µs (171 MiB/s) | 42µs (231 MiB/s) | 1.2ms (8 MiB/s) | 20x slower |
| 64KB | 1.3ms (48 MiB/s) | 611µs (102 MiB/s) | 1.6ms (38 MiB/s) | 1.3x slower |
| 256KB | 6.2ms (40 MiB/s) | 2.6ms (97 MiB/s) | 3.1ms (82 MiB/s) | **2x faster** |
| 1MB | 20ms (50 MiB/s) | 16.7ms (60 MiB/s) | 9.6ms (104 MiB/s) | **2x faster** |

## Remaining GPU bottlenecks

1. **Bitonic sort for BWT** — O(log^2 n) kernel launches per doubling step, with
   host↔device sync between steps. At 64KB: ~4000 kernel dispatches.

2. **No shared memory usage** — LZ77 hash kernel uses only global memory.
   Loading hash buckets into `__local` memory could help at larger sizes.

3. **Host round-trips between stages** — Each pipeline stage transfers data back to
   host. Chaining GPU stages would eliminate redundant transfers.

4. **Hash bucket overflow** — Fixed BUCKET_CAP=64 means highly repetitive data
   may miss good matches. Adaptive bucket sizing could help.

## Next steps

### Priority 1: Optimize BWT GPU path
- Replace bitonic sort with GPU parallel radix sort (fewer kernel launches)
- Or adapt SA-IS (linear time suffix array) for GPU
- Fix all-same-byte BWT bug
- Move rank assignment to GPU to avoid host↔device round-trips per step

### Priority 2: Chain GPU stages
- Keep data on device between LZ77 → Huffman encode (Deflate pipeline)
- Eliminate redundant host↔device transfers between pipeline stages

### Priority 3: Use local/shared memory in hash kernel
- Load hash buckets into `__local` memory for faster repeated access
- Could improve performance at mid-range sizes (64KB-256KB)

### Priority 4: Benchmark at even larger sizes
- Add 4MB, 16MB tiers — GPU advantage should grow with size
- End-to-end pipeline throughput benchmarks with GPU
