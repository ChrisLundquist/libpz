# CLAUDE.md — libpz development guide

## First-time setup
Extract test sample data (needed for validation/benchmark tests):
```bash
cd samples && mkdir -p cantrbry large && tar -xzf cantrbry.tar.gz -C cantrbry && tar -xzf large.tar.gz -C large && cd ..
```

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
- `kernels/*.cl` — OpenCL kernel source
- `benches/throughput.rs` — end-to-end pipeline benchmarks
- `benches/stages.rs` — per-algorithm scaling benchmarks

## Conventions
- Public API: `encode()` / `decode()` returning `PzResult<T>`
- Provide `_to_buf` variants for caller-allocated output buffers
- Algorithms must be composable and standalone — not tied to a specific pipeline
- GPU-friendly: prefer table-driven, branchless designs (no data-dependent divisions)
- Error type: `PzError` (InvalidInput, BufferTooSmall, Unsupported, InternalError)
- Tests go in `#[cfg(test)] mod tests` at the bottom of each module file

## Project status
11 of 12 milestones complete. All core algorithms, pipelines, GPU kernels, auto-selection, optimal parsing, multi-threading, and tooling are implemented. Not started: fuzz testing (M5.3), Vulkan backend (M12).

For detailed GPU benchmarks, architecture notes, and roadmap see `ARCHITECTURE.md`.
