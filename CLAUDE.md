# CLAUDE.md — libpz development guide

## First-time setup
Extract test sample data (needed for validation/benchmark tests):
```bash
./scripts/setup.sh
```
This is called automatically by `bench.sh` and other scripts, so manual extraction is rarely needed.

## Build & test commands
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
Before every commit, **always** run (the pre-commit hook handles 1 and 2 automatically):
1. `cargo fmt` — format all code
2. `cargo clippy --all-targets` — fix all warnings, zero warnings policy
3. `cargo test` — all tests must pass

## Benchmarking & profiling

### Quick comparison vs gzip (end-to-end, real files)
```bash
./scripts/bench.sh                              # all Canterbury + large corpus
./scripts/bench.sh myfile.bin                   # specific files
./scripts/bench.sh -p deflate,lza               # subset of pipelines
./scripts/bench.sh -n 10                        # more iterations
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
- `scripts/setup.sh` — extract sample archives (idempotent, called automatically by other scripts)
- `scripts/bench.sh` — pz vs gzip comparison (ratio, throughput, all pipelines)
- `scripts/profile.sh` — samply profiling wrapper (headless by default, `--web` for browser)
- `profiling/` — saved profiles, organized as `<7-char-sha>/<description>.json.gz` (e.g. `a1b2c3d-dirty/`)
- `examples/profile.rs` — profiling harness (pipeline or individual stage loops)
- `.claude/agents/` — custom Claude Code subagents (e.g., `historian.md` for project history research)
- `.claude/friction/` — friction reports documenting workflow impediments and tool limitations

## Conventions
- Public API: `encode()` / `decode()` returning `PzResult<T>`
- Provide `_to_buf` variants for caller-allocated output buffers
- Algorithms must be composable and standalone — not tied to a specific pipeline
- GPU-friendly: prefer table-driven, branchless designs (no data-dependent divisions)
- Error type: `PzError` (InvalidInput, BufferTooSmall, Unsupported, InternalError)
- Tests go in `#[cfg(test)] mod tests` at the bottom of each module file

## Feature flags
- **Default:** WebGPU GPU acceleration is enabled by default via wgpu (Vulkan/Metal/DX12). Tests gracefully skip if no GPU device available.
- **`--no-default-features`:** Pure CPU build (disables WebGPU). Useful for minimal builds or platforms without GPU support.
- **`webgpu`:** GPU acceleration feature (enabled by default). Always compile-check GPU changes: `cargo build --features webgpu`
- GPU is **not faster for small inputs** — LZ77 breaks even ~256KB, Huffman ~128KB. Don't optimize GPU paths for small data.

## Understanding GPU resource usage
- **Never trust memory estimates in comments or plans — read the actual `create_buffer` / `Buffer::create` calls.** Buffer sizes are computed from input length at runtime; the only source of truth is the Rust code that allocates them. Staging buffers, padding, and alignment are easy to miss.
- The GPU batched path (`compress_parallel_gpu_batched` in `parallel.rs`) and the pipeline-parallel path (`compress_pipeline_parallel`) solve the same problem differently. Understand both before modifying either.

## Tracing data flow through pipelines
- To understand how a pipeline actually works, trace a single block from `compress_block()` in `blocks.rs` through each stage. The `StageBlock.streams` field is the key handoff point — it's `None` before demux and `Some(Vec<Vec<u8>>)` after. Entropy encoders consume `streams`, not `data`.
- When GPU and CPU paths produce different output for the same input, the divergence is almost always in the demux (stream splitting), not in the entropy coder. Check stream count and byte ordering first.
- Adding a new LZ-based pipeline only requires an entry in `demuxer_for_pipeline()` and the `entropy_encode()`/`entropy_decode()` dispatch in `blocks.rs` — no new block function needed.

## Common pitfalls
1. **Using `--no-default-features` unintentionally** — This disables WebGPU. If GPU tests/benchmarks aren't running, verify you haven't disabled the default features.
2. **Multi-stream format changes are subtle** — LZ-based pipelines demux into independent streams (LZ77: 3 streams for offsets/lengths/literals, LZSS: 4, LZ78: 1). The per-block multistream container header is `[num_streams: u8][pre_entropy_len: u32][meta_len: u16][meta]`. Don't change without understanding round-trip implications.
3. **Pre-commit hook auto-reformats and re-stages files** — `cargo fmt` runs automatically and modifies files in-place. If a commit fails on clippy, fix the warning and make a new commit (don't amend).
4. **Use dedicated tools instead of shell pipelines** — Prefer Grep/Glob tools over `grep | cut | sort | uniq` shell pipelines. Dedicated tools are faster, don't need permission approval, and produce better-structured output.
5. **All algorithms must be composable** — New algorithms must work both standalone and in pipelines. See `src/validation.rs` for test patterns.
6. **In a worktree, run git from the worktree directory** — Never `cd` to the main repo to run git commands. Commits will land on the wrong branch.

## Shell command style
- **Never use `git -C <path>`** — always run git commands from the repo root directly.
- **Don't chain git commands with `echo`, `printf`, or `bash -c`** — these create compound commands that don't match permission allow-lists. Instead, run git commands as standalone Bash calls and use separate tool calls for any follow-up.
- **Prefer multiple sequential tool calls** over `&&`-chained shell commands when the chain mixes git with non-git commands.
- **Use HEREDOC syntax for multi-line commit messages** (see the commit instructions in your system prompt).

## Commit discipline
- **Commit at every logical completion point** — don't let work accumulate uncommitted.
- A "logical completion point" is any self-contained change: a bug fix, a new feature, a refactor, a test addition, a docs update, etc.
- Run the pre-commit checklist (`fmt`, `clippy`, `test`) before each commit.
- If a task has multiple independent parts, commit each part separately rather than one giant commit at the end.

## Documenting friction points
When you encounter obstacles, bugs, or workflow impediments during development, document them in `.claude/friction/`:

**What to document:**
- Permission prompts that shouldn't require approval (e.g., patterns not matching in settings.json)
- Tool limitations or unexpected behavior (e.g., pattern matching edge cases)
- Bugs in dependencies or external tools
- Confusing error messages that needed investigation
- Workarounds required for common tasks
- Missing features that would improve the workflow

**Format:**
Create a new file named `YYYY-MM-DD-short-description.md` with:
```markdown
# Short Description of Issue

**Date:** YYYY-MM-DD
**Agent/User:** (who encountered this)
**Severity:** Low | Medium | High

## Problem
Clear description of the friction point and how it impeded work.

## Steps to Reproduce
1. Step-by-step reproduction if applicable

## Workaround
What was done to work around the issue (if any).

## Suggested Fix
Ideas for permanently resolving the issue.
```

**When to document:**
- After completing a task where you hit an obstacle
- When you notice a pattern of repeated friction across sessions
- When a workaround feels hacky or unsatisfying
- When you spend >5 minutes debugging a tool or permission issue

These reports help identify patterns and prioritize tooling improvements.

## Project status
11 of 12 milestones complete. All core algorithms, pipelines, GPU kernels (WebGPU), auto-selection, optimal parsing, multi-threading, and tooling are implemented. Not started: fuzz testing (M5.3).

For detailed GPU benchmarks, architecture notes, and roadmap see `ARCHITECTURE.md`.
