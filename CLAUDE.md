# CLAUDE.md — libpz development guide

## Build & test commands
```bash
cargo build                    # compile
cargo test                     # run all tests
cargo test fse                 # run tests for a specific module
cargo fmt                      # format code (must match CI)
cargo clippy --all-targets     # lint (must pass with zero warnings)
```

## Pre-commit checklist
Before every commit, **always** run:
1. `cargo fmt` — format all code
2. `cargo clippy --all-targets` — fix all warnings, zero warnings policy
3. `cargo test` — all tests must pass

## Project layout
- `src/lib.rs` — crate root, module declarations
- `src/{algorithm}.rs` — one file per composable algorithm (bwt, deflate, fse, huffman, lz77, mtf, rangecoder, rle)
- `src/pipeline.rs` — multi-stage compression pipelines (Deflate, Bw, Lza)
- `src/frequency.rs` — shared frequency table used by entropy coders
- `src/ffi.rs` — C FFI bindings
- `src/validation.rs` — cross-module integration tests

## Conventions
- Public API: `encode()` / `decode()` returning `PzResult<T>`
- Provide `_to_buf` variants for caller-allocated output buffers
- Algorithms must be composable and standalone — not tied to a specific pipeline
- GPU-friendly: prefer table-driven, branchless designs (no data-dependent divisions)
- Error type: `PzError` (InvalidInput, BufferTooSmall, Unsupported, InternalError)
- Tests go in `#[cfg(test)] mod tests` at the bottom of each module file
