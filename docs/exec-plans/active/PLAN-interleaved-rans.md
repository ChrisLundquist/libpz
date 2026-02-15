# Interleaved rANS Execution Plan (CPU-First, GPU-Ready)

**Created:** 2026-02-15  
**Status:** Planned  
**Priority:** P1  
**Owner:** Engineering team

## Problem

Single-stream rANS appears near the practical limit of safe inner-loop micro-optimization on current CPU path. We need higher-leverage throughput gains that preserve correctness and format stability.

## Objective

1. Introduce interleaved rANS as the high-throughput path for medium/large streams.
2. Preserve backward compatibility with existing single-stream rANS frames.
3. Keep design GPU-scalable by framing data so independent lanes/chunks can be scheduled in parallel.

## Clarification: Existing Mux vs Proposed Partitioning

Current pipeline already performs token-level stream splitting (for example, next/symbol/offset/length streams depending on demux format).

This plan adds a second layer:

1. Keep existing semantic stream split unchanged.
2. Optionally partition each semantic stream into entropy chunks/lanes for interleaved rANS scheduling.

## Scope

### In scope

1. Interleaved rANS stage encode/decode in pipeline stages.
2. Versioned/frame-tagged entropy payload support.
3. Policy and threshold selection for single vs interleaved rANS.
4. Benchmark and validation matrix for encode/decode and end-to-end pipelines.

### Out of scope (initial)

1. Mandatory GPU interleaved rANS decode kernel implementation.
2. Replacing all entropy paths with interleaved mode by default on day one.

## API and Format Plan

1. Add/confirm an explicit rANS frame mode tag:
   - `single_stream`
   - `interleaved_n`
2. Interleaved payload includes:
   - `num_states`
   - per-lane final states
   - per-lane word counts
   - lane word streams
3. Decode path remains backward-compatible:
   - old single-stream frames decode unchanged
   - interleaved frames decode through new path

## Implementation Phases

### Phase A: Stage-level Integration

1. Add `stage_rans_encode_interleaved` and `stage_rans_decode_interleaved`.
2. Keep existing `stage_rans_encode` / `stage_rans_decode`.
3. Add option flag in pipeline config for interleaved selection.

Acceptance:

1. Round-trip parity tests pass for all affected pipelines.
2. No regressions in single-stream decode compatibility.

### Phase B: Policy/Threshold Rollout

1. Add policy:
   - small streams: single-stream rANS
   - medium/large streams: interleaved rANS
2. Initial conservative defaults:
   - `num_states = 4`
   - threshold in `64KB-128KB` range
3. Feature/option kill-switch for instant fallback.

Acceptance:

1. Decode parity remains green.
2. Stage-level throughput improves for target sizes.

### Phase C: Tunable Entropy Partitioning

1. Add tunables:
   - `entropy_lane_count`
   - `entropy_chunk_bytes`
2. Apply tunables within each existing semantic stream (not replacing demux semantics).
3. Keep defaults conservative and disabled unless policy selects interleaved mode.

Acceptance:

1. No ratio regression from partitioning.
2. Measurable throughput gains on 1MB+ data.

### Phase D: GPU-Ready Framing Validation

1. Ensure interleaved framing is lane/chunk schedulable for future GPU decode.
2. Add CPU encode → (future) GPU decode compatibility tests scaffolding.
3. Keep CPU decode as correctness reference.

Acceptance:

1. Framing supports independent lane/chunk processing.
2. CPU reference decode remains byte-identical.

## Benchmark and Validation Protocol

1. Stage benchmarks:
   - `cargo bench --bench stages_rans -- encode`
   - `cargo bench --bench stages_rans -- decode`
2. Pipeline benchmarks:
   - `cargo bench --bench throughput_lzr`
   - `cargo bench --bench throughput_deflate`
3. Profile harness:
   - `./scripts/profile.sh --stage rans --size 1048576 --iterations 300`
   - optionally `--decompress` for decode hotspots
4. Correctness:
   - `cargo test rans`
   - pipeline round-trip tests for LZR/LZSSR/LZ78R where applicable

## Guardrails

1. No format break without explicit mode/version signaling.
2. No default flip without decode-focused benchmark improvement.
3. Any throughput regression >3% must be explained and either reverted or gated.

## Risks

1. Extra framing overhead hurts small blocks.
2. Lane imbalance on skewed distributions reduces gains.
3. Benchmark noise causes false wins/losses.

## Immediate Next Actions

1. Add stage-level interleaved encode/decode path behind option flag.
2. Add frame mode signaling and compatibility tests.
3. Run baseline A/B for `num_states=4` with threshold sweep (`64KB`, `96KB`, `128KB`).
4. Decide default policy only after decode and end-to-end criteria are met.
