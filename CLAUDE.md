# CLAUDE.md — libpz development guide

For detailed documentation, see `docs/DESIGN.md`, `docs/QUALITY.md`, `docs/design-docs/`, `docs/exec-plans/`, and `ARCHITECTURE.md`.

## Build & test

```bash
./scripts/test.sh              # Full suite: fmt, clippy, build, test
./scripts/test.sh --quick      # Skip build step, just lint + test
./scripts/test.sh --fix        # Auto-fix fmt + clippy before checking
./scripts/test.sh --all        # Test all feature combinations
./scripts/test-targets.sh ...  # Run multiple cargo test targets sequentially
```

The pre-commit hook (auto-configured by `scripts/setup.sh`) runs fmt, clippy, and tests before every commit. Use `--no-default-features` for CPU-only builds/tests. Prefer delegating test runs to the **tester** agent to keep your context clean.

## Benchmarking & profiling

```bash
./scripts/bench.sh             # pz vs gzip comparison (all pipelines, quiet)
./scripts/profile.sh           # samply profiling (see --help for options)
./scripts/samply-top-symbols.sh --profile ... --binary ...  # hotspot mapping for unsymbolicated save-only JSON
./scripts/gpu-meminfo.sh       # GPU memory cost calculator
./scripts/trace-pipeline.sh    # pipeline flow diagrams (text or mermaid)
./scripts/webgpu_profile.sh   # GPU vs CPU per-stage timing comparisons
```

All scripts support `--help`. Optimization workflow: measure (`bench.sh`) → identify (`profile.sh --stage <stage>`) → change → validate (`cargo test`) → re-measure (`cargo bench -- <stage>`) → confirm (`bench.sh`). Prefer delegating benchmark runs to the **benchmarker** agent.

**Benchmark caveat:** `bench.sh` runs `pz` as a subprocess, so results include ~260ms wgpu device init per invocation. Criterion benchmarks use pre-allocated buffers with zero I/O overhead, so they report up to ~18x higher throughput. Neither is wrong — they measure different things. When comparing, be explicit about which you're using.

## Agents

Specialized agents in `.claude/agents/` run on cheaper models and keep verbose output out of your context:
- **tester** — run tests, autofix, diagnose failures (Haiku)
- **benchmarker** — run benchmarks, generate comparison reports (Haiku)
- **historian** — git archaeology, research past attempts (Haiku)
- **tooling** — build scripts and workflow automation (Sonnet; consumes `.claude/friction/`)
- **maintainer** — review feedback backlog, update CLAUDE.md, delegate improvements (Opus; consumes `.claude/feedback/`)

## Architecture overview

All LZ-based pipelines share a unified token architecture (since PR #118):

```
input → tokenize() → Vec<LzToken> → TokenEncoder::encode() → multi-stream → entropy_encode()
```

**Three pluggable wire encoders** select how tokens map to byte streams:
- **LzSeqEncoder** (6 streams) — log2-coded offsets/lengths with repeat-offset tracking. Used by Lzf, LzSeqR, LzSeqH, SortLz. Best ratio.
- **LzssEncoder** (4 streams) — flag-bit based, raw u16 offsets+lengths. Used by Lzfi, LzssR. Faster decode, worse ratio.
- **Lz77Encoder** (3 streams) — DEFLATE-style. Legacy, no active pipeline uses it.

**Active pipelines:**

| Pipeline | Encoder | Entropy | Notes |
|----------|---------|---------|-------|
| **Lzf** (default) | LzSeq | FSE | General-purpose, good ratio |
| **LzSeqR** | LzSeq | rANS | Fastest overall, ratio matches Lzf |
| **LzSeqH** | LzSeq | Huffman | Fast decode |
| **Lzfi** | LZSS | interleaved FSE | Fastest algorithm but poor ratio (46% vs 34%) |
| **LzssR** | LZSS | rANS | Dominated by Lzfi, removal candidate |
| **SortLz** | LzSeq (internal) | FSE | Deterministic GPU radix-sort matching |
| **Bw** / **Bbw** | — | FSE | BWT-based, no LZ tokens |

**Removed pipelines:** Deflate (#117), Lzr (#118), Lz78R (#116), Parlz (ratio loss)

**CLI path:** `pz` always uses `streaming::compress_stream`, not `pipeline::compress_with_options`. The streaming path uses block-by-block parallelism with bounded memory.

### Silesia corpus benchmarks (211MB, CLI end-to-end, verified round-trip)

| Method | Compress | Ratio | Comp MB/s | Decomp MB/s |
|---------|----------|-------|-----------|-------------|
| gzip | 4.69s | 32.2% | 45 | 225 |
| pz lzf | 1.90s | 34.6% | 111 | 133 |
| pz lzseqr | 1.62s | 34.4% | 131 | 154 |
| pz lzfi | 1.70s | 46.0% | 125 | 143 |

pz compresses 2.5–7x faster than gzip with ~2pp ratio gap. Decompress is faster per-file but slower on many small files due to ~90ms per-invocation startup overhead. Criterion (pure algorithm, no I/O) measures 333–543 MB/s — the 3–4x CLI gap is in the streaming path, not the compressor.

**Benchmark corpus:** `./scripts/fetch-silesia.sh` downloads the 211MB Silesia corpus to `samples/silesia/`.

## Project layout

- `src/lib.rs` — crate root, `PzError`/`PzResult` types
- `src/lz_token.rs` — universal `LzToken` type, `TokenEncoder` trait, three encoder implementations
- `src/{algorithm}.rs` — one file per composable algorithm (bwt, crc32, fse, huffman, lz77, lzseq, lzss, lz_token, mtf, rans, rle, sortlz, recoil)
- `src/analysis.rs` — data profiling (entropy, match density, run ratio, autocorrelation)
- `src/optimal.rs` — optimal parsing (GPU top-K + backward DP)
- `src/simd.rs` — SIMD decode paths for rANS
- `src/streaming.rs` — streaming compression interface (CLI entry point)
- `src/ffi.rs` — C FFI bindings
- `src/pipeline/` — multi-stage compression pipelines, auto-selection, block parallelism, demux
- `src/bin/pz.rs` — CLI binary (`pz` with `-a`/`--auto` and `--trial` flags)
- `src/webgpu/` — WebGPU backend (feature-gated behind `webgpu`)
- `kernels/*.wgsl` — WebGPU kernel source
- `scripts/` — test, bench, profile, setup, and analysis tools
- `docs/` — design docs, quality status, exec plans, references

## Known dead ends

Before optimizing GPU code paths, read this first — multiple agents have spent full sessions rediscovering these:

- **GPU entropy (rANS/FSE) is slower than CPU** — 0.77x on encode, 0.54x on decode. This has been proven across 500+ optimization iterations. The serial state dependency in rANS limits GPU to ~300 threads; saturation needs ~8K-16K. Do not attempt to batch, parallelize, or "pipeline" GPU entropy encoding.
- **The parallel scheduler (`compress_with_options`) is CPU-only** — the GPU coordinator was removed because it serialized entropy encoding on one thread, bottlenecking at 28 MiB/s. GPU-accelerated compression uses the streaming path (`compress_stream`) which has a dedicated GPU coordinator with adaptive backpressure. Do not re-add a GPU coordinator to the parallel path.
- **The CLI uses `streaming::compress_stream`, not `pipeline::compress_with_options`** — the streaming path handles GPU match-finding via a coordinator thread with adaptive backpressure that decrements on batch completion. Workers use CPU for entropy.
- **The real GPU win (ring-buffered LZ77 batching) is already shipped** — delivers +7-17% throughput. See `docs/design-docs/gpu-strategy.md`.
- **GPU device init time skews throughput benchmarks** — first-call GPU init adds significant overhead that `bench.sh` captures but Criterion amortizes across iterations. When comparing GPU vs CPU throughput, use Criterion (`cargo bench`) for apples-to-apples; `bench.sh` reflects real-world cold-start cost. Don't chase "GPU is slower" regressions that are really just init time.
- **Compression ratio is limited by wire encoding overhead, not match quality** — the LZ match-finder finds good matches. The legacy Lz77Encoder (5-byte per match) was the worst offender; LzSeqEncoder (log2-coded, 6 streams) is much better but still ~2pp behind gzip on Silesia (34.4% vs 32.2%). Further ratio gains require encoding format work, not matcher tuning.
- **GPU Huffman is a dead end** — Huffman coding requires bit-level alignment, but GPU throughput depends on byte-aligned memory access patterns. This is a fundamental architectural mismatch; do not attempt to port Huffman to GPU.
- **GPU hash tables for LZ matching don't work** — GPU atomics don't preserve insertion order, so hash chains lose recency information. Match quality collapses to ~6% vs CPU's 99.6% on repetitive data. Tried twice (global atomics + shared-memory variant), both catastrophically failed. See `docs/design-docs/experiments.md`.
- **SSE2 rANS decode is 32% slower than scalar** — scalar 4-lane decode gets good ILP from out-of-order execution. SSE2 extract operations serialize and lose that parallelism. Proper SIMD rANS would need merged slot-indexed tables and SSE4.1+. The dispatch is disabled; don't re-enable it.
- **Fully parallel GPU LZ parsing (ParlZ) has unacceptable ratio loss** — 37.6% compression gap vs serial parsing. Forward-max-propagation conflict resolution is too aggressive. Hybrid GPU match-finding + CPU serial parsing is the correct architecture.
- **Iterative GPU algorithms have quadratic host overhead** — Repair grammar compression hit 0.4 MB/s due to 100+ rounds of buffer alloc + readback. Avoid per-round GPU↔CPU synchronization; prefer single-dispatch or persistent-buffer designs.
- **Window-capped suffix sorts break BWT invertibility** — FWST produced 433% ratio (massive expansion). Full suffix sort is structurally required for LF-mapping; there's no shortcut.

- **Streaming path is the CLI bottleneck, not the compressor** — Criterion measures 333–543 MB/s for raw algorithms, but CLI delivers 111–131 MB/s on the same data. The 3–4x gap is in `streaming::compress_stream`, not the encoder. Pipeline-level algorithmic speed differences (e.g., Lzfi vs LzSeqR) are largely invisible at the CLI level because streaming overhead dominates.
- **LzSeqR parallel encode used to route to incompatible GPU rANS** — `run_compress_stage` in `stages.rs` sent LzSeqR entropy to `stage_rans_encode_webgpu` (GPU chunked payload format), while the single-block path used standard CPU rANS. The chunked format was incompatible with all decoders. Fixed in PR #120 by routing to CPU rANS. Don't re-enable GPU rANS for LzSeqR without fixing the wire format compatibility.

For detailed history of all failed experiments, see `docs/design-docs/gpu-experiments-wave2-conclusions.md` and `docs/design-docs/experiments.md`.

## Key conventions

See **docs/DESIGN.md** for full design principles and **docs/design-docs/core-beliefs.md** for agent-first operating principles.

- Public API: `encode()` / `decode()` returning `PzResult<T>`, plus `_to_buf` variants
- Tests go in `#[cfg(test)] mod tests` at bottom of each module file
- GPU feature enabled by default, skip gracefully if no device available
- Zero warnings policy: `cargo clippy --all-targets` must pass clean
- Commit at every logical completion point (run `./scripts/test.sh --quick` first)

## Agent feedback loops

**Friction** (something blocked you or wasted time): Write a short report to `.claude/friction/YYYY-MM-DD-short-description.md` describing the problem, then move on. The tooling agent consumes the backlog and builds durable fixes.

**Feedback** (insights worth preserving): Write a short note to `.claude/feedback/YYYY-MM-DD-short-description.md` when you:
- Discover something that should be in CLAUDE.md (a convention, gotcha, or pattern not documented here)
- Find something in CLAUDE.md that was wrong, stale, or unhelpful
- Learn a non-obvious insight about the codebase that would save future agents time

Keep notes brief (a few lines). The **maintainer** agent consumes the backlog: evaluates reports, promotes worthy insights into CLAUDE.md, delegates fixes, and discards noise.
