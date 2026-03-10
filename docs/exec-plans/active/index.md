# Active Execution Plans

**Last Updated:** 2026-03-10

## Active Plans

### [PLAN-competitive-roadmap.md](PLAN-competitive-roadmap.md)
**Status:** In Progress (Phases 0–1 complete, Phase 2 active; Phases 3–4 blocked on GPU entropy) | **Priority:** P0

### [PLAN-unified-scheduler-perf-validation.md](PLAN-unified-scheduler-perf-validation.md)
**Status:** In Progress (Phases 0-1 landed; local baseline captured; Phase 2 optimization started) | **Priority:** P0

## Investigation TODOs

### [TODO-gpu-rans-6stream-bug.md](TODO-gpu-rans-6stream-bug.md)
**Status:** Open — GPU rANS interleaved decode fails with 6-stream LzSeqR; works with 4-stream LzssR | **Priority:** P1

### [TODO-benchmark-lzfi-vs-lzssr.md](TODO-benchmark-lzfi-vs-lzssr.md)
**Status:** Open — Benchmark whether LzssR is worth keeping vs Lzfi consolidation | **Priority:** P2

### [TODO-huffman-sync-decode.md](TODO-huffman-sync-decode.md)
**Status:** PARKED — valid approach, zero implementation progress, awaiting LzSeq encoding work | **Priority:** P2

## Completed Plans (in ../completed/)

- `PLAN-p0a-gpu-rans-vertical-slice.md` — GPU chunked rANS vertical slice (CLOSED: structural dead end)
- `PLAN-unified-scheduler-north-star.md` — Unified scheduler north star (PARKED: GPU entropy blocked)
- `PLAN-interleaved-rans.md` — Interleaved rANS (PARKED: Phase A merged, Phase D cancelled)
- `agent-harness-implementation.md` — Agent harness (PARKED: Phase 1 complete, rest deferred)
- `PLAN-gpu-backpressure-impl.md` — GPU ring buffer batching
- `lz77_merge.md` — Cooperative-stitch kernel consolidation
- `upgrade-wgpu-to-27.md` — wgpu 24→27 upgrade
- `integrate-wgpu-profiler.md` — wgpu-profiler for GPU timestamps
- `agent-harness-analysis.md` — OpenAI agent harness gap analysis

## Adding New Plans

1. Use template: Problem → Approach → Tasks → Acceptance Criteria
2. Add entry to this index
3. Update `tech-debt-tracker.md` if resolves known issue

## Moving Completed Plans

1. Move file to `../completed/`
2. Update this index (move entry to completed list)
