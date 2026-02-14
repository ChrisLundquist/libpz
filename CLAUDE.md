# CLAUDE.md — libpz development guide

This is the entry point for day-to-day development. For detailed documentation, see:
- **docs/DESIGN.md** - Design principles and patterns
- **docs/QUALITY.md** - Quality status per module/feature
- **docs/design-docs/** - Detailed design documentation (GPU, pipelines, etc.)
- **docs/exec-plans/** - Active execution plans and tech debt tracker
- **ARCHITECTURE.md** - Technical architecture, benchmarks, roadmap

## First-time setup
Extract test sample data (needed for validation/benchmark tests):
```bash
./scripts/setup.sh
```
This is called automatically by `bench.sh` and other scripts, so manual extraction is rarely needed.

## Build & test commands

### Quick verification (recommended)
```bash
./scripts/test.sh              # Run full test suite (fmt, clippy, build, test)
./scripts/test.sh --quick      # Skip build step, just lint + test
./scripts/test.sh --fix        # Auto-fix fmt + clippy before checking
./scripts/test.sh --webgpu     # Include WebGPU-specific tests
./scripts/test.sh --all        # Test all feature combinations
```

### Individual commands
```bash
cargo build                          # compile (WebGPU enabled by default)
cargo test                           # run all tests (includes GPU tests, skips if no device)
cargo test fse                       # run tests for a specific module
cargo fmt                            # format code (must match CI)
cargo clippy --all-targets           # lint (must pass with zero warnings)
cargo build --no-default-features    # compile CPU-only (no GPU)
cargo test --no-default-features     # run tests without GPU features
```

## Git hooks setup
Enable the pre-commit hook (auto-formats and lints before each commit):
```bash
git config core.hooksPath .githooks
```

## Pre-commit checklist
Before every commit, **always** run:
```bash
./scripts/test.sh --quick    # Recommended: runs fmt, clippy, test
```
Or manually (the pre-commit hook handles 1 and 2 automatically):
1. `cargo fmt` — format all code
2. `cargo clippy --all-targets` — fix all warnings, zero warnings policy
3. `cargo test` — all tests must pass

## Benchmarking & profiling

### Quick comparison vs gzip (end-to-end, real files)
```bash
./scripts/bench.sh                              # all Canterbury + large corpus (quiet, summary only)
./scripts/bench.sh myfile.bin                   # specific files
./scripts/bench.sh -p deflate,lzf               # subset of pipelines
./scripts/bench.sh -n 10                        # more iterations
./scripts/bench.sh -v                           # verbose mode (per-file breakdown)
./scripts/bench.sh --help                       # full usage info
```

### Criterion microbenchmarks
```bash
cargo bench                             # all benchmarks including GPU (~10 min)
cargo bench --bench throughput          # end-to-end pipeline throughput only
cargo bench --bench stages              # per-algorithm stage benchmarks only
cargo bench -- fse                      # filter to specific algorithm
cargo bench -- compress                 # filter to compress group
cargo bench --no-default-features       # CPU-only benchmarks (no GPU)
```

### Profiling with samply
```bash
cargo install samply                                        # one-time setup
./scripts/profile.sh                                        # → profiling/a1b2c3d/lzf_encode_256KB.json.gz
./scripts/profile.sh --pipeline deflate --decompress
./scripts/profile.sh --stage lz77                           # single algorithm
./scripts/profile.sh --stage fse --size 1048576             # 1MB input
./scripts/profile.sh --web --pipeline lzf                   # open browser UI
./scripts/profile.sh --no-default-features --pipeline lzf   # pure CPU (no GPU)
./scripts/profile.sh --features webgpu --pipeline lzf       # WebGPU GPU backend
samply load profiling/a1b2c3d/lz77_encode_256KB.json.gz     # view saved profile later
```

### Optimization workflow
1. **Measure** — `./scripts/bench.sh` to get baseline vs gzip
2. **Identify** — `./scripts/profile.sh --stage <stage>` to find hotspots
3. **Change** — edit the algorithm
4. **Validate** — `cargo test <module>` to verify correctness
5. **Re-measure** — `cargo bench -- <stage>` for precise before/after
6. **Confirm** — `./scripts/bench.sh` to verify end-to-end improvement

### GPU memory analysis
```bash
./scripts/gpu-meminfo.sh                    # overview table for standard block sizes
./scripts/gpu-meminfo.sh -b 262144          # detailed breakdown for 256KB blocks
./scripts/gpu-meminfo.sh -b 1048576 --explain  # 1MB blocks with formulas and sources
```
Shows per-block GPU memory costs and recommended batch sizes for different GPU memory budgets. Parses actual buffer allocations from `src/webgpu/lz77.rs` to compute ring buffer depth (2-3) or fallback to per-block allocation.

### Pipeline flow tracing
```bash
./scripts/trace-pipeline.sh                 # deflate pipeline (text format)
./scripts/trace-pipeline.sh -p bw           # BWT pipeline
./scripts/trace-pipeline.sh -p lzfi --format mermaid  # FSE interleaved (mermaid)
```
Generates visual flow diagrams showing:
- Call path from `compress_block()` through each stage
- File locations (file:line) for each function
- Data transformations (`StageBlock.data` vs `StageBlock.streams`)
- Stream counts and demuxer types

Use `--format mermaid` for diagrams you can paste into [mermaid.live](https://mermaid.live) or GitHub.

## Project layout
- `src/lib.rs` — crate root, module declarations
- `src/{algorithm}.rs` — one file per composable algorithm (bwt, deflate, fse, huffman, lz77, mtf, rans, rle)
- `src/analysis.rs` — data profiling for pipeline auto-selection (entropy, match density, etc.)
- `src/optimal.rs` — optimal LZ77 parsing via backward DP (match table + cost model)
- `src/pipeline/` — multi-stage compression pipelines, auto-selection
  - `mod.rs` — container format (V2 header), compress/decompress entry points
  - `blocks.rs` — single-block compress/decompress via `demuxer_for_pipeline()` dispatch
  - `demux.rs` — `StreamDemuxer` trait, `LzDemuxer` enum, `demuxer_for_pipeline()`
  - `stages.rs` — per-stage functions (`stage_demux_compress`, `stage_huffman_encode_gpu`, etc.)
  - `parallel.rs` — block-parallel, pipeline-parallel, and GPU-batched multi-block paths
- `src/frequency.rs` — shared frequency table used by entropy coders (SIMD-accelerated counting)
- `src/simd.rs` — SIMD-accelerated primitives (SSE2/AVX2 for x86_64, NEON stubs for aarch64)
- `src/webgpu/` — WebGPU backend (feature-gated behind `webgpu`)
- `src/ffi.rs` — C FFI bindings
- `src/validation.rs` — cross-module integration tests (including GPU↔CPU cross-decompression)
- `kernels/*.wgsl` — WebGPU kernel source
- `benches/throughput.rs` — end-to-end pipeline benchmarks
- `benches/stages.rs` — per-algorithm scaling benchmarks
- `scripts/test.sh` — comprehensive test suite (fmt, clippy, build, test, feature combinations)
- `scripts/setup.sh` — extract sample archives (idempotent, called automatically by other scripts)
- `scripts/bench.sh` — pz vs gzip comparison (ratio, throughput, all pipelines)
- `scripts/profile.sh` — samply profiling wrapper (headless by default, `--web` for browser)
- `scripts/gpu-meminfo.sh` — GPU memory cost calculator and batch size recommender
- `scripts/trace-pipeline.sh` — pipeline flow diagram generator (text or mermaid format)
- `profiling/` — saved profiles, organized as `<7-char-sha>/<description>.json.gz` (e.g. `a1b2c3d-dirty/`)
- `examples/profile.rs` — profiling harness (pipeline or individual stage loops)
- `docs/` — structured documentation (progressive disclosure):
  - `DESIGN.md` — design principles and patterns
  - `QUALITY.md` — quality status per module/feature
  - `design-docs/` — detailed design docs (core-beliefs.md, gpu-batching.md, pipeline-architecture.md, etc.)
  - `exec-plans/` — active and completed execution plans, tech debt tracker
  - `references/` — third-party API docs (compression algorithms, wgpu, criterion, etc.)
  - `generated/` — auto-generated docs (GPU memory formulas, etc.)
- `.claude/agents/` — custom Claude Code subagents:
  - `historian.md` — research project history and git archaeology (Haiku)
  - `tooling.md` — build scripts to minimize context usage and streamline workflows (Sonnet)
  - `benchmarker.md` — run benchmarks and generate detailed performance reports (Haiku)
  - `tester.md` — run tests with programmatic autofixes, diagnose remaining errors (Haiku)
- `.claude/friction/` — friction backlog for the tooling agent (other agents file reports here; the tooling agent consumes them)

## Key Conventions

For detailed design principles and patterns, see **docs/DESIGN.md**.
For agent-first operating principles, see **docs/design-docs/core-beliefs.md**.

**Quick reference:**
- Public API: `encode()` / `decode()` returning `PzResult<T>`, plus `_to_buf` variants
- Tests go in `#[cfg(test)] mod tests` at bottom of each module file
- GPU feature enabled by default, skip gracefully if no device available
- Zero warnings policy: `cargo clippy --all-targets` must pass clean
- Commit at every logical completion point (run `./scripts/test.sh --quick` first)

## Project Status

**11 of 12 milestones complete.** See **docs/QUALITY.md** for module-by-module status and **docs/exec-plans/tech-debt-tracker.md** for known gaps.

For detailed architecture, GPU benchmarks, and roadmap see **ARCHITECTURE.md**.
