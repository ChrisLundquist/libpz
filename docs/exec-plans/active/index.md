# Active Execution Plans

**Last Updated:** 2026-02-14

## Purpose

This directory contains active (in-progress or planned) execution plans for implementing features, fixing bugs, or addressing technical debt.

## Active Plans

### [agent-harness-implementation.md](agent-harness-implementation.md)
**Status:** 🟡 In Progress (Phase 1 complete)
**Priority:** P1
**Owner:** Engineering team
**Estimated completion:** 8 weeks (started 2026-02-14)

Implement agent-first development practices from OpenAI's methodology:
- ✅ Phase 1: Documentation structure (DONE)
- 🔄 Phase 2: Quality & debt tracking (CURRENT)
- ⏳ Phase 3-8: Design principles, references, automation, linters, agents

**Dependencies:** None
**Blocks:** Documentation validation CI

---

### [PLAN-gpu-backpressure-impl.md](PLAN-gpu-backpressure-impl.md)
**Status:** ✅ Completed (kept for reference)
**Priority:** P1
**Owner:** Engineering team
**Completion date:** 2026-02-12

GPU batching with ring buffer and backpressure. Prevents GPU memory exhaustion during large-file compression.

**Result:** Implemented in `src/pipeline/parallel.rs`
**Documentation:** `docs/design-docs/gpu-batching.md`

**Note:** Move to ../completed/ directory

---

### [upgrade-wgpu-to-27.md](upgrade-wgpu-to-27.md)
**Status:** ⏳ Planned
**Priority:** P2
**Owner:** Unassigned
**Estimated effort:** 1-2 days

Upgrade wgpu from 22.x to 27.x (current stable). Requires API migration and testing.

**Dependencies:** None
**Risks:** API changes, potential kernel rewrites

---

### [integrate-wgpu-profiler.md](integrate-wgpu-profiler.md)
**Status:** ⏳ Planned
**Priority:** P2
**Owner:** Unassigned
**Estimated effort:** 1 day

Integrate wgpu-profiler for GPU kernel timing. Currently use samply for CPU profiling, need GPU-specific metrics.

**Dependencies:** None
**Benefits:** Identify GPU kernel bottlenecks

---

### [TODO-huffman-sync-decode.md](TODO-huffman-sync-decode.md)
**Status:** ⏳ Planned
**Priority:** P2
**Owner:** Unassigned
**Estimated effort:** 3-5 days

Implement GPU Huffman decode (sync decode kernel). Currently CPU-only.

**Dependencies:** None
**Decision criteria:** Implement if >2x speedup on large blocks, otherwise close

---

### [lz77_merge.md](lz77_merge.md)
**Status:** ✅ Completed (kept for reference)
**Priority:** P1
**Owner:** Engineering team
**Completion date:** 2026-02-13

Merge LZ77 GPU kernels (batch, per-position, hash-table). Consolidate redundant code.

**Result:** Hash-table kernel is primary, others marked legacy
**Note:** Move to ../completed/ directory

---

### [agent-harness-analysis.md](agent-harness-analysis.md)
**Status:** ✅ Completed (analysis only)
**Priority:** P1
**Owner:** Engineering team
**Completion date:** 2026-02-14

Analysis of OpenAI agent harness article. Identified 10 applicable gaps for libpz.

**Result:** Created agent-harness-implementation.md plan
**Note:** Move to ../completed/ directory

---

## Plan Status Legend

- 🟢 **Ready** - Can start immediately
- 🟡 **In Progress** - Actively being worked on
- 🔴 **Blocked** - Waiting on dependencies
- ⏳ **Planned** - Not yet started, no blockers
- ✅ **Completed** - Done (should move to ../completed/)

## Dependencies

```
agent-harness-implementation
  └─> (no dependencies)

upgrade-wgpu-to-27
  └─> (no dependencies, but blocks integrate-wgpu-profiler)

integrate-wgpu-profiler
  └─> upgrade-wgpu-to-27 (optional, but recommended)

TODO-huffman-sync-decode
  └─> (no dependencies)
```

## Adding New Plans

When creating a new execution plan:
1. Use template: Problem → Approach → Tasks → Acceptance Criteria
2. Add entry to this index with status, priority, owner, effort estimate
3. Link dependencies and blocked-by relationships
4. Set realistic completion estimate
5. Update tech-debt-tracker.md if resolves known issue

## Moving Completed Plans

When a plan is completed:
1. Update status to ✅ Completed
2. Set completion date
3. Link to implementation (PR, files, design doc)
4. Move to ../completed/ directory
5. Update this index (mark as completed or remove entry)

## Related Documents

- **../tech-debt-tracker.md** - Known issues (many have plans here)
- **../../QUALITY.md** - Quality grades (gaps may need plans)
- **../../design-docs/** - Completed work often produces design docs
