# LZR Pareto-Frontier Improvement Plan

Date: 2026-02-15
Owner: libpz engineering
Scope: `lzr` pipeline ratio/speed frontier

## Goal

Move `lzr` default behavior toward a stronger Pareto frontier (better ratio at equal or near-equal speed), while preserving composability and backward compatibility.

## 1. Define optimization modes and targets

- Add `CompressionProfile`:
  - `Balanced` (default)
  - `Fast`
  - `Ratio`
- Set guardrails:
  - `Balanced`: maintain or improve current speed, improve ratio.
  - `Ratio`: maximize ratio with acceptable slowdown.
- Track on fixed workloads:
  - 8KB, 64KB, 1MB, 4MB
  - `bench.sh` corpus runs

## 2. Replace 5-byte LZ77 token shape with semantic tokens

- Introduce internal token representation:
  - `Literal(u8)`
  - `Match { len, dist }`
- Remove mandatory `next` literal coupling from match emission.
- Keep legacy decoder path; gate new token format via stream version.

## 3. Entropy-code semantic symbols (not raw field bytes)

- Add DEFLATE-like symbolization:
  - length class + extra bits
  - distance class + extra bits
  - literal symbols
- Build symbol streams:
  - literal/length symbols
  - distance symbols
  - extra bits stream
- Entropy-code these symbol streams.

## 4. Add entropy-cost model for parsing decisions

- Add cost model API: estimated bits per token from current symbol stats.
- Parse objectives:
  - `Fast`: lazy + low chain depth.
  - `Balanced`: lazy + moderate chain + entropy-aware tie-breakers.
  - `Ratio`: minimize estimated bits.

## 5. Implement bounded optimal parse for ratio mode

- Reuse top-K candidate infrastructure.
- Add bounded DP/optimal parser for `Ratio` mode.
- Keep hard caps on K/lookahead/memory.

## 6. Rework demux/remux around semantic streams

- Update demux in `src/pipeline/demux.rs` for v2 semantic streams.
- Preserve stage composability:
  - LZ stage emits semantic token streams.
  - Entropy stage consumes symbol streams.

## 7. Versioning and compatibility

- Add explicit format version in header/container metadata.
- Decoder supports:
  - v1 legacy 5-byte match stream
  - v2 semantic token stream
- Add cross-version tests and fixtures.

## 8. Benchmark matrix and acceptance criteria

Run for each meaningful change:

- `cargo bench --bench stages_lz77`
- `cargo bench --bench stages_rans`
- `./scripts/bench.sh -p lzr -t 1`
- `./scripts/bench.sh -p lzr -t 0`

Acceptance:

- `Balanced`: ratio improves vs current `40.6%` with no major speed regression.
- `Ratio`: first milestone <= `35%` ratio on current Canterbury+large set.

## 9. Execution phases

- Phase A: token format + versioning scaffold.
- Phase B: semantic entropy streams (parser unchanged) to isolate coding wins.
- Phase C: entropy-aware parse tie-breakers.
- Phase D: bounded optimal parser for `Ratio`.
- Phase E: tune defaults and make `Balanced` Pareto-oriented default.

## 10. Immediate next task

- Implement Phase A skeleton:
  - semantic token enum
  - v2 stream header/version plumbing
  - encode/decode round-trip for v2
  - v1 fallback decode path retained
