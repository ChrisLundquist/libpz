Provenance: 9c5d6b8-dirty
# CPU Baseline Report

**Date:** 2026-02-15  
**Commit:** `46bef2b`  
**Host:** `x86_64-apple-darwin` (`uname -a`), 16 logical CPUs (`getconf _NPROCESSORS_ONLN`)  
**Rust:** `rustc 1.93.0 (254b59607 2026-01-19)`

## Commands Executed

1. `LC_ALL=C LANG=C ./scripts/bench.sh -n 5 -p deflate,lzr,lzf`
2. `LC_ALL=C LANG=C cargo bench --bench stages -- rans`
3. `LC_ALL=C LANG=C cargo bench --bench throughput` (stopped before completion)

## Artifacts

1. `docs/generated/2026-02-15-bench-sh.txt`
2. `docs/generated/2026-02-15-stages-rans.txt`
3. `docs/generated/2026-02-15-throughput.txt` (not part of this baseline; run interrupted)

## Benchmark Script Reliability Fix

`scripts/bench.sh` was failing in decompression setup when copying over read-only temp files (permission denied on `cp`).  
Fix applied: use `cp -f` in overwrite paths in decompression preparation.

## Corpus-Level Baseline (`bench.sh`, n=5)

Corpus: 14 files, 13.32 MB total (`samples/cantrbry` + `samples/large`)

### Compression

1. `gzip`: 3.81 MB (28.6%), 2269.1 ms, 5.9 MB/s
2. `pz-deflate`: 5.54 MB (41.6%), 664.9 ms, 20.0 MB/s
3. `pz-lzr`: 5.41 MB (40.6%), 567.1 ms, 23.5 MB/s
4. `pz-lzf`: 5.55 MB (41.7%), 628.3 ms, 21.2 MB/s

### Decompression

1. `gzip`: 396.6 ms, 33.6 MB/s
2. `pz-deflate`: 519.9 ms, 25.6 MB/s
3. `pz-lzr`: 503.3 ms, 26.5 MB/s
4. `pz-lzf`: 539.2 ms, 24.7 MB/s

## rANS Stage Baseline (`cargo bench --bench stages -- rans`)

### Encode Throughput

1. `8192`: 136.94-139.07 MiB/s
2. `65536`: 129.95-130.15 MiB/s
3. `4194304`: 123.97-124.88 MiB/s

### Decode Throughput

1. `8192`: 119.73-120.03 MiB/s
2. `65536`: 108.50-109.06 MiB/s
3. `4194304`: 114.88-115.02 MiB/s

## Notes

1. The full end-to-end Criterion throughput baseline was intentionally interrupted and is not considered part of this baseline snapshot.
2. No WebGPU device was available in this run context for GPU benchmark groups.
3. `scripts/profile.sh` could not complete in this sandbox because `samply record` failed with `Unknown(1100)`. Equivalent measurements were captured by running the same profiling harness directly (`cargo run --profile profiling --example profile`).

## Clean Rerun (after removing background workload)

After clearing host contention, a clean `lzr` corpus rerun and profiling harness pass were captured.

### `bench.sh` clean rerun (`./scripts/bench.sh -n 5 -p lzr`)

1. Compression: `pz-lzr` 5.41 MB (40.6%), 460.9 ms, 28.9 MB/s
2. Decompression: `pz-lzr` 352.1 ms, 37.8 MB/s

Artifact: `docs/generated/2026-02-15-bench-lzr-clean-rerun.txt`

### Thread-mode comparison (`bench.sh --threads`)

Using the new `scripts/bench.sh --threads` passthrough:

1. `-t 1` (single-threaded): compression 18.1 MB/s, decompression 28.0 MB/s
2. `-t 0` (auto threads): compression 17.1 MB/s, decompression 34.5 MB/s

Artifacts:

1. `docs/generated/2026-02-15-bench-lzr-t1-clean.txt`
2. `docs/generated/2026-02-15-bench-lzr-t0-clean.txt`

Interpretation:

1. Compression difference between `-t 1` and `-t 0` is small on this corpus.
2. Decompression is materially better with auto threads.
3. For roadmap baselines, include both thread settings to avoid overfitting optimization to one mode.

### Profile harness measurements (`examples/profile`)

1. `rans` encode (262144 bytes, 400 iterations): 184.0 MB/s
2. `rans` decode (262144 bytes, 400 iterations): 157.8 MB/s
3. `lzr` encode (262144 bytes, 200 iterations): 46.4 MB/s
4. `lzr` decode (262144 bytes, 200 iterations): 103.2 MB/s
5. `lz77` encode (262144 bytes, 200 iterations): 67.1 MB/s
6. `lz77` decode (262144 bytes, 200 iterations): 453.8 MB/s

Artifacts:

1. `docs/generated/2026-02-15-profile-harness-rans-encode.txt`
2. `docs/generated/2026-02-15-profile-harness-rans-decode.txt`
3. `docs/generated/2026-02-15-profile-harness-lzr-encode.txt`
4. `docs/generated/2026-02-15-profile-harness-lzr-decode.txt`
5. `docs/generated/2026-02-15-profile-harness-lz77-encode.txt`
6. `docs/generated/2026-02-15-profile-harness-lz77-decode.txt`

### Bottleneck inference

For 256KB CPU runs, `lzr` encode throughput is much closer to `lz77` encode throughput than `rans` encode throughput (`46.4` vs `67.1` vs `184.0` MB/s). This indicates the highest-ROI CPU optimization path for `lzr` encode is LZ77-side work before additional rANS micro-optimization.

## Phase 1 Optimization Result (LZ77 Demux Fast Path)

Change:

1. In `src/pipeline/demux.rs` for `LzDemuxer::Lz77`, CPU `Auto/Lazy` path now demuxes directly from `Vec<lz77::Match>` instead of serializing to LZ bytes and immediately re-splitting into streams.
2. GPU and `ParseStrategy::Optimal` paths still use the existing byte-buffer path.

Validation:

1. `cargo test lzr` passed.
2. `cargo test pipeline::tests` passed (101 tests).

Performance snapshots (clean sequential runs):

1. Profile harness (`examples/profile`, 256KB):
   - `lzr` encode: `46.4` → `51.1` MB/s
   - `lzr` decode: `103.2` → `120.8` MB/s
2. Corpus benchmark (`./scripts/bench.sh -n 3 -p lzr -t 0`):
   - `pz-lzr` compression throughput: `17.1` → `26.9` MB/s
   - `pz-lzr` decompression throughput: `34.5` → `38.6` MB/s

Artifacts:

1. `docs/generated/2026-02-15-profile-harness-lzr-encode-after-demux-fastpath.txt`
2. `docs/generated/2026-02-15-profile-harness-lzr-decode-after-demux-fastpath.txt`
3. `docs/generated/2026-02-15-bench-lzr-t0-after-demux-fastpath.txt`

## Phase 1 Optimization Result #2 (Unchecked Hash in LZ77 Hot Path)

Change:

1. `lz77::HashChainFinder::find_match` and `find_top_k` now use `hash3_unchecked` after existing `remaining >= 3` guards.
2. Removed now-unused checked `hash3` helper.

Validation:

1. `cargo test lz77` passed.
2. `cargo test lzr` passed.

Performance snapshots (sequential runs):

1. Profile harness (256KB):
   - `lz77` encode: `65.1` → `73.2` MB/s
   - `lzr` encode: `51.1` → `54.8` MB/s
2. Corpus benchmark (`./scripts/bench.sh -n 3 -p lzr -t 0`):
   - `pz-lzr` compression throughput: `26.9` → `26.4` MB/s
   - `pz-lzr` decompression throughput: `38.6` → `37.1` MB/s

Interpretation:

1. Stage-level encode speed improved measurably.
2. End-to-end corpus throughput stayed roughly flat (small regression within likely run noise on this host).
3. Keep this change for now because it is low-risk and consistently improves the LZ77 stage; continue validating with larger repeated runs.

Artifacts:

1. `docs/generated/2026-02-15-profile-harness-lz77-encode-after-unchecked-hash-seq.txt`
2. `docs/generated/2026-02-15-profile-harness-lzr-encode-after-unchecked-hash-seq.txt`
3. `docs/generated/2026-02-15-bench-lzr-t0-after-unchecked-hash-seq.txt`

## Unsafe Removal Recheck (Safe Hash Restored)

The `hash3_unchecked` change was removed and safe `hash3` restored in `src/lz77.rs`, then remeasured sequentially.

Comparison (unsafe vs safe, same harness/settings):

1. `lz77` encode (profile harness, 256KB): `73.2` (unsafe) vs `72.0` MB/s (safe)
2. `lzr` encode (profile harness, 256KB): `54.8` (unsafe) vs `50.5` MB/s (safe)
3. Corpus benchmark (`./scripts/bench.sh -n 3 -p lzr -t 0`):
   - compression: `26.4` (unsafe) vs `24.7` MB/s (safe)
   - decompression: `37.1` (unsafe) vs `37.9` MB/s (safe)

Decision:

1. Keep safe hash path (unsafe removed), per preference.
2. Accept the observed encode-side regression and continue optimizing via safe structural changes.

Artifacts:

1. `docs/generated/2026-02-15-profile-harness-lz77-encode-safe-hash-seq.txt`
2. `docs/generated/2026-02-15-profile-harness-lzr-encode-safe-hash-seq.txt`
3. `docs/generated/2026-02-15-bench-lzr-t0-safe-hash-seq.txt`

## Next Step (Phase 1 Start)

Begin CPU rANS performance pass:

1. Verify and tighten SIMD decode dispatch coverage (`rans` decode paths).
2. Re-run `cargo bench --bench stages -- rans` after each optimization change.
3. Re-run `./scripts/bench.sh -n 5 -p lzr` for end-to-end pipeline impact validation.
