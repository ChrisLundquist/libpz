# Active Execution Plans

**Last Updated:** 2026-02-22

## Active Plans

### [PLAN-competitive-roadmap.md](PLAN-competitive-roadmap.md)
**Status:** In Progress (Phases 0–1 complete, Phase 2 active) | **Priority:** P0

### [PLAN-p0a-gpu-rans-vertical-slice.md](PLAN-p0a-gpu-rans-vertical-slice.md)
**Status:** In Progress (Slices 0–3 complete; Slice 4 perf gate open as of 2026-02-17) | **Priority:** P0

### [PLAN-interleaved-rans.md](PLAN-interleaved-rans.md)
**Status:** In Progress (Phase A merged in PR #91) | **Priority:** P1

### [PLAN-unified-scheduler-north-star.md](PLAN-unified-scheduler-north-star.md)
**Status:** In Progress (prototype merged; GPU Phases 1–5 pending P0-A perf gate) | **Priority:** P1

### [agent-harness-implementation.md](agent-harness-implementation.md)
**Status:** In Progress (Phase 1 complete 2026-02-14, Phases 2–8 deferred) | **Priority:** P1

### [TODO-huffman-sync-decode.md](TODO-huffman-sync-decode.md)
**Status:** Planned | **Priority:** P2

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
