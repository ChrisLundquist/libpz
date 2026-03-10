# Active Execution Plans

**Last Updated:** 2026-03-09

## Active Plans

### [PLAN-competitive-roadmap.md](PLAN-competitive-roadmap.md)
**Status:** In Progress (Phases 0–1 complete, Phase 2 active; Phases 3–4 blocked on GPU entropy) | **Priority:** P0

### [PLAN-unified-scheduler-perf-validation.md](PLAN-unified-scheduler-perf-validation.md)
**Status:** In Progress (Phases 0-1 landed; local baseline captured; Phase 2 optimization started) | **Priority:** P0

## Parked Plans

### [PLAN-interleaved-rans.md](PLAN-interleaved-rans.md)
**Status:** PARKED — Phase A merged (PR #91); Phase D cancelled (GPU rANS dead end); Phases B–C need new owner | **Priority:** P1

### [PLAN-unified-scheduler-north-star.md](PLAN-unified-scheduler-north-star.md)
**Status:** PARKED — Phases 3–4 done and in production; Phases 2+5 blocked indefinitely (GPU entropy not competitive) | **Priority:** P1

### [TODO-huffman-sync-decode.md](TODO-huffman-sync-decode.md)
**Status:** PARKED — valid approach, zero implementation progress, awaiting LzSeq encoding work | **Priority:** P2

### [agent-harness-implementation.md](agent-harness-implementation.md)
**Status:** PARKED — Phase 1 complete; Phases 2–8 deferred | **Priority:** P1

## Closed Plans

### [PLAN-p0a-gpu-rans-vertical-slice.md](PLAN-p0a-gpu-rans-vertical-slice.md)
**Status:** CLOSED — Slice 4 perf gate failed; GPU rANS 0.54–0.77x CPU after 29+ iterations; structural dead end | **Priority:** was P0

## Completed Plans (in ../completed/)

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
