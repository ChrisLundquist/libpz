# TODO: Benchmark Lzfi vs LzssR — consolidation candidate

## Question

Are both Lzfi and LzssR worth keeping? They use the same demuxer (LZSS, 4
streams) and differ only in entropy coder (interleaved FSE vs rANS).

## Current state

| Property | Lzfi | LzssR |
|----------|------|-------|
| Demuxer | LzssEncoder (4 streams) | LzssEncoder (4 streams) |
| Entropy | Interleaved FSE | rANS |
| Auto-selected | Yes (high entropy + matches) | Never |
| Pipeline ID | 5 | 6 |
| GPU entropy | Yes (interleaved FSE) | Yes (rANS Recoil) |

## Known data

- FSE decode is ~2.2x faster than rANS decode (596 vs 266 MB/s, Criterion)
- FSE encode is comparable to rANS encode (~357 vs 359 MB/s)
- Lzfi auto-selected when: match_density > 0.4 + byte_entropy > 6.0,
  or match_density > 0.2 + byte_entropy > 5.0
- LzssR is only exercised via trial compression or explicit user selection

## Action items

1. Run `./scripts/bench.sh` comparing Lzfi vs LzssR on Canterbury+Silesia corpus
2. Run Criterion benchmarks: `cargo bench -- lzfi lzssr` for per-stage timing
3. If LzssR shows no ratio or throughput advantage over Lzfi, consider removing
   it to reduce pipeline surface area (similar to Lzr removal)
4. If rANS interleaved or Recoil decode gives LzssR better GPU decode throughput,
   document the use case and keep it

## Files

- `src/pipeline/stages.rs` — stage dispatch for both pipelines
- `src/pipeline/mod.rs` — `auto_select_pipeline`, `select_pipeline_trial`
- `src/pipeline/blocks.rs` — entropy encode/decode dispatch
