Provenance: e5b3290-dirty

# Samply LZ77 Hotspots and Execution Plan

Date: 2026-02-15
Profile: `profiling/e5b3290-dirty/lz77_encode_1MB.json.gz`
Workload: `./scripts/profile.sh --stage lz77 --size 1048576 --iterations 120`

## What We Found

Saved Samply JSON remained unsymbolicated (`"symbolicated": false`), so hotspots were recovered by mapping sampled frame addresses to symbols from:
- `target/profiling/examples/profile`
- `nm -n` address map (+ Mach-O base `0x100000000`)

Top leaf hotspots (sampled):
1. `pz::simd::compare_bytes_avx2` (dominant leaf addresses around `0x37de0..0x37e21`)
2. `pz::lz77::HashChainFinder::find_match` (leaf addresses around `0x4009a..0x401af`)
3. `pz::lz77::compress_lazy_to_matches_with_limit` (secondary)

Conclusion: CPU encode bottleneck is match-finder candidate filtering + compare loop dispatch, not entropy stage.

## Plan From This Profile

1. Match-candidate prefiltering in `find_match`:
- If candidate mismatches at `best_length` probe byte, skip SIMD compare (cannot beat current best).

2. Early-exit once maximum useful match is reached:
- Break chain walk when `best_length + 1 >= remaining` (cannot improve while preserving required literal byte).

3. Small hot-path arithmetic cleanup:
- Replace `% MAX_WINDOW` with `& WINDOW_MASK` for ring indexing.

4. Validate and remeasure:
- `cargo test lz77`
- `cargo test lzr`
- `cargo bench --bench stages_lz77`
- `./scripts/bench.sh -n 3 -p lzr -t 0`

## Execution Status

Implemented: Steps 1-3 in `src/lz77.rs`.

Measured impact:
- `cargo bench --bench stages_lz77`: significant throughput improvements across all tested sizes for both encode and decode.
- `./scripts/bench.sh -n 3 -p lzr -t 0`:
  - `pz-lzr` compression: 26.1 MB/s
  - `pz-lzr` decompression: 37.3 MB/s

## Next Optimization Candidates

1. Add bounded chain-depth heuristics by parse mode (fast vs quality) with ratio guardrails.
2. Add a cheap 32-bit prefix gate before deeper compare for high-collision buckets.
3. Investigate `compare_bytes_avx2` alignment and tail handling to reduce overhead per candidate.
