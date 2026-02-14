# Design Documentation Index

**Last Updated:** 2026-02-14

## Purpose

This directory contains detailed design documentation for libpz's architecture, algorithms, and implementation decisions.

## Core Documents

### [core-beliefs.md](core-beliefs.md)
**Status:** ✅ Active
**Last reviewed:** 2026-02-14

Agent-first operating principles specific to libpz. Actionable rules for AI agents working in this codebase.

**Key topics:**
- Read code before changing it
- GPU compilation requirements
- Buffer allocations as source of truth
- Commit discipline
- Tool usage (Grep/Glob vs shell pipelines)

---

## Technical Design Docs

### [gpu-batching.md](gpu-batching.md)
**Status:** ✅ Active
**Last reviewed:** 2026-02-14

GPU memory management and batching strategy. Ring buffer with backpressure, batch size recommendations.

**Key topics:**
- Ring buffer depth (2-3 blocks)
- Backpressure mechanism
- Memory budgets for different GPU sizes
- Per-block vs batched allocation

---

### [pipeline-architecture.md](pipeline-architecture.md)
**Status:** ✅ Active
**Last reviewed:** 2026-02-14

Pipeline composition and demuxer pattern. How algorithms compose into multi-stage pipelines.

**Key topics:**
- V2 container format
- StreamDemuxer trait
- LzDemuxer enum
- Multi-stream entropy coding
- Adding new pipelines

---

### [lz77-gpu.md](lz77-gpu.md)
**Status:** ✅ Active
**Last reviewed:** 2026-02-14

LZ77 GPU kernel implementation details. Hash-table kernel, batch kernel, top-K kernel.

**Key topics:**
- Hash-table construction on GPU
- Match finding strategies
- Top-K for optimal parsing
- Break-even points (~256KB)

---

## Research and Experiments

### [experiments.md](experiments.md)
**Status:** ⚠️ Historical
**Last reviewed:** 2026-02-14

Experimental results and dead ends. Kept for historical reference.

**Key topics:**
- Algorithm experiments that didn't ship
- Performance comparisons
- Why certain approaches were rejected

---

### [research-log.md](research-log.md)
**Status:** ⚠️ Historical
**Last reviewed:** 2026-02-14

Development research log. Early design decisions and exploration.

**Key topics:**
- Initial design explorations
- Algorithm selection rationale
- Performance investigation notes

---

## Needed Documentation

### optimal-parsing.md (Missing - P1)
**Status:** ❌ Not started

Design documentation for `src/optimal.rs`. Should explain:
- Backward DP algorithm
- Cost model (why these weights?)
- GPU top-K → CPU DP handoff
- Parameter tuning guidance

**Owner:** Unassigned
**Priority:** P1 (high)
**Effort:** 1 day

---

### multi-stream-format.md (Missing - P2)
**Status:** ❌ Not started

Detailed specification of multi-stream container format:
- Stream demuxing logic
- Stream count determination
- Fallback to single-stream criteria
- Cross-pipeline consistency

**Owner:** Unassigned
**Priority:** P2 (medium)
**Effort:** 0.5 day

---

## Document Status Legend

- ✅ **Active** - Current, reviewed, accurate
- ⚠️ **Historical** - Kept for reference, may be outdated
- ❌ **Missing** - Planned but not created

---

## Contributing

When adding new design docs:
1. Add entry to this index with status and last-reviewed date
2. Link from relevant code comments (e.g., `// See docs/design-docs/foo.md`)
3. Update CLAUDE.md if agents should proactively consult it
4. Set owner and review schedule

When updating existing docs:
1. Update last-reviewed date in this index
2. Mark as Historical if superseded (don't delete)
3. Update cross-references in other docs

---

## Related Documents

- **../DESIGN.md** - High-level design principles
- **../QUALITY.md** - Quality grades (tracks documentation gaps)
- **../exec-plans/tech-debt-tracker.md** - Missing docs are tracked as tech debt
