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
| **Num** | — | per-plane FSE | Numeric/binary front-end (byte-plane transpose + per-plane gated delta/zigzag). Opt-in (`-p num`); for the worst numeric files — x-ray 55%→48% (beats xz/bzip2), sao 74%→63%. Block-parallel, O(n) inverse. `src/numeric.rs` |
| **Pz2** | sequences (own wire) | 8-lane Huffman | Decode-first clean-slate codec (`-p pz2`): LzSeq parse re-expressed as zstd-style sequences, 8-lane Huffman literals + fused splice, 2 MiB blocks (window = block), auto-greedy parse (greedy won 11/12 Silesia files on this wire; near-random blocks stay lazy). **Pareto-superior to pzstd -3: blob 31.0% vs 31.4% at 1.36x faster all-cores decode** (16.8 ms / 12.0 GiB/s; ST 1405 MB/s = 3.45x lzf). Encode 64 MiB/s all-cores is the deliberate decode-first trade (4 MiB blocks rejected — concurrent 16 MiB chain walks collapse encode 5.5x). `src/pz2.rs`, `docs/design-docs/clean-slate-codec.md` |
| **Pz2d** | sequences (own wire) | 8-lane Huffman | Pz2 dict tier (`-p pz2d`): 32 MiB segments, first 16 MiB doubles as shared dict (frozen-finder encode, 2-wave arena decode). **Blob 30.5%** — best LZ-family ratio in pz, 0.9pp under pzstd-3 — at 36.6 ms decode (v2 arena decode; wall is DRAM-latency-bound on random dict reads, NOT traffic-bound — do not chase further copy elimination or the two-region splice, see §11c) and 23.7 s encode (concurrent dict-walk cache inflation 2.9x — the open lever). Opt-in max-ratio tier; ledger in clean-slate-codec.md §11b-§11c. |

**Removed pipelines:** Deflate (#117), Lzr (#118), Lz78R (#116), Parlz (ratio loss)

**Deferred (parked in history, reverted out of the tree):** BWT+CM order-1 context-mixing range coder — real ratio win (dickens BWT −9–12%, x-ray −17%) but ~11 MB/s/core decode. See `docs/design-docs/bwt-cm-findings.md` (it points to the parked commit). The carry-safe range coder there is the reusable substrate for future high-ratio families.

**CLI path:** `pz` always uses `streaming::compress_stream`, not `pipeline::compress_with_options`. The streaming path uses block-by-block parallelism with bounded memory.

### Silesia corpus benchmarks (202MB blob, CLI end-to-end, all cores, verified round-trip)

Measured on Apple M5 Max (6P + 12E, 128GB unified), zstd 1.5.7. These supersede
earlier figures that were ~7x low on pz throughput (those reflected effectively
single-threaded runs; the CLI uses all cores by default).

| Method | Ratio | Comp MB/s | Decomp MB/s |
|---------|-------|-----------|-------------|
| gzip -6 | 32.2% | 45 | 225 |
| zstd -1 | 34.6% | 5900 | 1500 |
| zstd -3 (default) | 31.2% | 2860 | 1350 |
| zstd -9 | 27.9% | 695 | 1460 |
| pz lzf | 32.2% | 740 | 2760 |
| pz lzseqr | 32.2% | 930 | 3690 |
| pz lzfi | 46.0% | 1130 | 3130 |
| pz bw | 27.8% | 166 | 1120 |

**pz2 is not in the table above** (different methodology: warm-cache
hyperfine to /dev/null, 2026-06-10): blob **31.0%** at **64 MiB/s** compress /
**12.0 GiB/s** decompress all-cores — better ratio than zstd-3 AND 1.36x
faster parallel decode than pzstd -3, i.e. Pareto-superior to the parallel-
zstd frontier on (ratio, decode). Encode speed is its one conceded axis.
See `docs/design-docs/clean-slate-codec.md` §8-§10.

**Read the decode column carefully — it is the most misunderstood number in this repo.**
The pz "Decomp MB/s" above is **all-cores** throughput; zstd's is **single-threaded** (the
zstd CLI does not parallelize decode). On an apples-to-apples **single-thread** basis,
**zstd decodes ~3.5x FASTER per core** (zstd-3 ~1500 MB/s vs pz lzf ~430 MB/s, measured),
because zstd's entropy decoder is hand-tuned (huff0 4-stream literals + 2-state FSE + BMI2)
and pz's FSE decode is latency-bound (4-way interleaving buys only ~1.15x on this CPU).
So pz's genuine decode edge is **parallel/bulk throughput** (it scales across cores while
zstd's CLI doesn't), **not** per-core speed. Do not repeat the old "2.4–2.7x faster than
zstd" framing — it compared pz-all-cores to zstd-one-core.

On ratio, **zstd-3 (31.2%) still beats the pz default (32.2%) by ~0.9pp**, and zstd-9
(27.9%) ≈ ties pz bw (27.8%) while decoding far faster per core — so pz is **not**
single-thread Pareto-optimal. Its defensible position is (good ratio + high parallel decode
throughput). zstd -T0 compress (2.8–5.9 GB/s) is unbeatable on the compress axis. Compare
against zstd's frontier (`-1`/`-3`/`-6`/`-9`), not `-19` (~50 MB/s, not a speed competitor).

The pz ratios above are the **lazy default** (the shipped parser): a 1 MiB match window +
repeat-offset-aware parsing cut lzseqr/lzf ~1.6pp from the old 33.8/34.0%, and lzseqr now
beats zstd-1 on structured files (xml 11.8%, nci 8.1%). `--greedy` is an **opt-in** parse
mode that trades ~0.9pp better ratio on text (lzseqr blob → 31.3%, ~tied with zstd-3) for
~37% slower encode and a regression on structured/record data, so it is not the default;
making lazy ≥ greedy everywhere is an open follow-up. The FSE decode-only fix also lifted
single-thread `lzf` decode ~+72%, and two silent `bbw` corruption bugs (SA-IS rotation
mis-sort + u16 per-block factor-count overflow) are fixed and regression-tested.

**Entropy/BWT ratio work (all decode-cost-free):** FSE now picks `accuracy_log` adaptively
instead of using a fixed value — a full sweep for the BWT pipelines (bw −1.9pp, bbw −2.6pp,
decode is inverse-BWT-bound so it's free) and a bounded `center..center+2` window for the LZ
literal stream (lzf 32.4→32.2%). The BWT pipelines also gained **RUNA/RUNB** bzip2-style
zero-run coding (`src/zrle.rs`): MTF zero-runs flow through the FSE model in bijective base 2,
worth bw −0.5 to −1.4pp on text (dickens bw now 28.9%, beating zstd-9 there), with a legacy-RLE
fallback for all-256-value blocks. Decode hardening: overlapping LZ match copies now grow
exponentially (O(log) memmoves) instead of byte-at-a-time, so small-offset long matches no
longer decode quadratically (8–13x faster on repetitive input; Silesia unaffected).
`DEFAULT_BW_BLOCK_SIZE` is now 1 MiB, ratifying what the CLI streaming path already did
(it never applied the old 512KB adjustment — the library path was 0.54pp worse); 2–4 MiB
was swept and rejected (blob saturates at 27.38%, a zstd-12-dominated point, while
aggregate decode halves — see `docs/design-docs/bw-blocksize-findings.md`).

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
- **Ratio is part encoding overhead, part parse/window — the old "encoding only" claim was half-right** — LzSeqEncoder's offset/length coding is already tight (baseline+extra-bits, like zstd), so the encoding-is-everything framing held for the legacy Lz77Encoder era. But the bigger lever turned out to be parse-side: the match window was too small (128KB window / 256KB block) and the repeat-offset cache was almost never used (<2% on text vs zstd's 30–50%). Raising the default window to 1 MiB (1 MiB blocks) + repeat-offset-aware parsing cut Silesia ratio ~1.6pp (lzseqr 33.8→32.2%) and now beats zstd-1 on structured files (xml, nci). Remaining encoding-side wins still open: flag-stream→literal-length sequences, order-1 literals.
- **GPU Huffman is a dead end** — Huffman coding requires bit-level alignment, but GPU throughput depends on byte-aligned memory access patterns. This is a fundamental architectural mismatch; do not attempt to port Huffman to GPU.
- **GPU hash tables for LZ matching don't work** — GPU atomics don't preserve insertion order, so hash chains lose recency information. Match quality collapses to ~6% vs CPU's 99.6% on repetitive data. Tried twice (global atomics + shared-memory variant), both catastrophically failed. See `docs/design-docs/experiments.md`.
- **SSE2 rANS decode is 32% slower than scalar** — scalar 4-lane decode gets good ILP from out-of-order execution. SSE2 extract operations serialize and lose that parallelism. Proper SIMD rANS would need merged slot-indexed tables and SSE4.1+. The dispatch is disabled; don't re-enable it.
- **Fully parallel GPU LZ parsing (ParlZ) has unacceptable ratio loss** — 37.6% compression gap vs serial parsing. Forward-max-propagation conflict resolution is too aggressive. Hybrid GPU match-finding + CPU serial parsing is the correct architecture.
- **Iterative GPU algorithms have quadratic host overhead** — Repair grammar compression hit 0.4 MB/s due to 100+ rounds of buffer alloc + readback. Avoid per-round GPU↔CPU synchronization; prefer single-dispatch or persistent-buffer designs.
- **Window-capped suffix sorts break BWT invertibility** — FWST produced 433% ratio (massive expansion). Full suffix sort is structurally required for LF-mapping; there's no shortcut.
- **Order-1 context-bucketed literals are a measured dead end (Stage-0 spike, 2026-06)** — bucketing the post-LZ literal stream by previous byte into per-context FSE tables nets only +0.1–0.6pp on Silesia text at 1 MiB blocks (best: 16 coarse contexts); full 256-context bucketing is **negative** (−2.2 to −5.7pp — per-block table headers swamp the gain). The header-free ceiling (perfect prev-byte conditioning) is only +1.1–2.7pp, below the roadmap's 2–3pp gate, so no static context map at any granularity can pass — and dodging header costs requires adaptive per-bit coding, the proven ~11 MB/s/core latency wall. Kills the Brotli-style context-literal path; the LZMA-class range coder stays parked per its own decision rule. Numbers were measured with the *optimistic* 32 KiB-window parse (more surviving literals than the shipped parse). See `docs/design-docs/ctx-literals-stage0-findings.md` + `examples/ctx_literals_probe.rs`.

- **Streaming overhead is NOT a 3–4x bottleneck (corrected on M5 Max)** — earlier notes claimed a 3–4x CLI-vs-Criterion gap inside `streaming::compress_stream`. That was a measurement artifact: the "333–543 MB/s Criterion" number was all-cores on repetitive tiled data, compared against a single-threaded CLI run. Measured properly, the streaming path is within ~3% of the raw single-block compressor, and the CLI does 740–1130 MB/s compress / 2700–3700 MB/s decode (all cores, 13.4x thread scaling). The real speed limiter is the **per-core compressor** (~50–70 MB/s single-thread, ~6–9x behind zstd's per-core), driven by `find_best` hash-chain walking — an algorithmic cost, not streaming and not (per wave-2 experiments) GPU-addressable on this hardware.
- **LzSeqR parallel encode used to route to incompatible GPU rANS** — `run_compress_stage` in `stages.rs` sent LzSeqR entropy to `stage_rans_encode_webgpu` (GPU chunked payload format), while the single-block path used standard CPU rANS. The chunked format was incompatible with all decoders. Fixed in PR #120 by routing to CPU rANS. Don't re-enable GPU rANS for LzSeqR without fixing the wire format compatibility.
- **No cheap per-core DECODE win exists; the gap to zstd is structural** — profiled lzf single-thread decode (samply): ~49% FSE entropy + ~47% LZ match-copy. FSE decode is **latency-bound** (~4.7 cyc/byte ≈ the `decode_table[state]` load latency); single-stream micro-opts are futile and **4-way interleaving buys only ~1.15x on the M5** (its wide OoO core already hides the latency — measured in isolation, ratio-neutral). Match-copy is already `memcpy`/`extend_from_within`. So pz decodes ~3.5x slower per core than zstd (432 vs 1513 MB/s) because zstd's entropy decoder is hand-tuned (huff0 4-stream + 2-state FSE + BMI2); closing it needs a ground-up entropy-decode rewrite, not an incremental tweak. **Profiling gotcha:** `scripts/samply-top-symbols.sh` flat `nm`/`atos` leaf mapping misattributes unsymbolized libstd copy code to the nearest exported symbol — it labeled the match-copy memmove as `std::sync::mpmc::Channel::recv`, which does NOT exist in the single-thread decode path. Trust samply's `stackTable` call chain, not the flat leaf→symbol map.

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
