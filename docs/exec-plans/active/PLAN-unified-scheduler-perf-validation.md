# Unified Scheduler Perf Validation and Overhead Reduction

**Created:** 2026-03-01
**Status:** In Progress (Phases 0-1 landed; Phase 2 optimization started)
**Last Updated:** 2026-03-02
**Priority:** P0
**Owner:** Next implementation agent
**Base Branch:** `claude/confident-faraday`
**Implementation Branch:** `codex/unified-scheduler-perf-plan`

## Problem

The current direction is a single unified scheduler path for CPU/GPU compression. We need to make scheduler overhead demonstrably minimal and reduce it where needed, without introducing alternate fast paths.

Automated CI perf checks are not reliable due to hardware variability. Validation is done by agent-on-commit on the same laptop, so we need a reproducible local perf-gate workflow with clear pass/fail criteria.

## Non-Goals

1. No separate fast path outside the unified scheduler.
2. No CI-hardware-dependent perf gating.
3. No container format changes.

## Goals

1. Quantify unified scheduler overhead in CPU-only and WebGPU runs.
2. Reduce overhead in the unified scheduler where telemetry shows it is dominant.
3. Add a repeatable local perf-gate procedure run by agent on each commit.
4. Preserve correctness and compression behavior.

## Baseline Metrics and Definitions

All metrics are collected from inside `compress_parallel_unified`.

1. `total_ns`: wall-clock time spent in unified scheduler for a compression call.
2. `stage_compute_ns`: summed time spent inside `run_compress_stage` on CPU workers and GPU coordinator.
3. `queue_wait_ns`: time waiting for queue lock/condvar.
4. `queue_admin_ns`: time inside queue bookkeeping (`push_back`, pop, pending/closed/failed updates).
5. `gpu_handoff_ns`: time spent preparing/sending GPU requests.
6. `gpu_try_send_full_count`: number of full-channel handoff fallbacks.
7. `gpu_try_send_disconnected_count`: number of disconnected-channel fallbacks.

Derived:

1. `scheduler_overhead_ns = queue_wait_ns + queue_admin_ns + gpu_handoff_ns`
2. `tracked_thread_time_ns = stage_compute_ns + scheduler_overhead_ns`
3. `scheduler_overhead_pct = scheduler_overhead_ns / tracked_thread_time_ns`

## Acceptance Targets

Targets are evaluated on the same laptop with fixed commands and medians across repeated runs.

1. CPU-only workloads: `scheduler_overhead_pct <= 5%` for large multiblock runs.
2. WebGPU workloads: `scheduler_overhead_pct <= 8%` for large multiblock runs.
3. No throughput regression > 3% versus recorded local baseline unless explicitly accepted in plan notes.
   1. Current `scripts/perf-gate.sh` default is 4% to reduce false failures from local run-to-run variance; use `--throughput-regression-pct 3` for strict runs.
4. No correctness regressions (`cargo test`, round-trip tests, existing pipeline tests).

If a target is missed, the commit may still land only with an explicit follow-up task and updated baseline rationale.

## Progress Snapshot

1. Phase 0 completed:
   1. Unified scheduler telemetry added (disabled by default).
   2. Profile harness prints machine-readable `SCHEDULER_STATS` via `--print-scheduler-stats`.
2. Phase 1 in place:
   1. Local perf gate script added: `scripts/perf-gate.sh`.
   2. Baseline captured in `docs/generated/perf-gate-baseline.tsv`.
   3. Latest passing run artifact: `docs/generated/2026-03-01-161450-perf-gate-run.tsv`.
3. WebGPU note for current environment:
   1. `profile --gpu` currently exits with `webgpu requested but unavailable`.
   2. Perf gate handles this by skipping GPU matrix when unavailable.
4. Phase 2 progress:
   1. Removed extra Stage0/Fused payload cloning across worker -> GPU coordinator handoff.
   2. `GpuRequest::Stage0`/`GpuRequest::Fused` now pass block indices and coordinator reads block slices directly.
   3. Collapsed per-task completion bookkeeping from two queue locks to one in both CPU worker and GPU completion paths (pending replacement semantics preserved).
   4. A/B (same-laptop, alternating order, 1MB, 20 iters, 3 repeats) vs `c502fb5` showed lower scheduler overhead on `deflate/lzr/lzf` with throughput deltas within gate tolerance in median; `lzseqr` overhead change was small (+0.0032 abs).
5. Phase 3 progress:
   1. Increased GPU request channel depth from fixed `4` to adaptive `min(num_blocks, 2*worker_count)` clamped to `1..16` to reduce transient `try_send(Full)` fallback pressure.
   2. Reordered GPU coordinator servicing to process StageN and fused requests before Stage0 batches for better downstream fairness.
   3. Validation: `cargo test pipeline::parallel::tests` and targeted WebGPU-feature interchangeability tests pass.

## Execution Phases

### Phase 0: Instrumentation and Measurement Surface

Files:

1. `src/pipeline/parallel.rs`
2. `examples/profile.rs`

Tasks:

1. Add a lightweight `SchedulerStats` struct and timing/counter collection in unified scheduler code paths.
2. Gate stats emission behind an option/env flag so default runtime overhead stays negligible.
3. Extend `examples/profile.rs` with a flag to print scheduler stats in machine-readable form (JSON line or TSV row).

Acceptance:

1. Stats output is available in profile runs and stable across repeated invocations.
2. With stats disabled, no measurable regression in quick smoke bench.

### Phase 1: Local Perf-Gate Workflow

Files:

1. `scripts/perf-gate.sh` (new)
2. `docs/generated/` (results snapshots)
3. `docs/exec-plans/active/` (summary notes)

Tasks:

1. Create `scripts/perf-gate.sh` that runs a fixed matrix on this laptop:
   1. CPU: `deflate,lzr,lzf,lzseqr` with `-t 0` and fixed sizes.
   2. WebGPU (if available): same matrix with `--webgpu`.
2. Record medians for throughput and scheduler overhead metrics.
3. Compare to baseline file and exit non-zero on threshold breach.

Acceptance:

1. Running the script twice gives near-identical medians within expected run-to-run noise.
2. Script can be used as "agent on commit" gate.

### Phase 2: Scheduler Overhead Reduction (Unified Path Only)

Files:

1. `src/pipeline/parallel.rs`

Tasks:

1. Reduce queue lock hold times (tighten critical sections).
2. Reduce queue churn where possible without changing scheduling semantics.
3. Keep pending/closed/failed invariants explicit and unchanged in behavior.
4. Avoid introducing alternate execution pathways.

Acceptance:

1. `scheduler_overhead_pct` improves vs baseline on CPU and WebGPU workloads.
2. No regression in correctness tests.

### Phase 3: GPU Handoff Copy and Backpressure Improvements

Files:

1. `src/pipeline/parallel.rs`
2. `src/webgpu/mod.rs` (if helper surface is required)

Tasks:

1. Remove avoidable block payload copying for Stage0/fused requests where safe.
2. Keep fallback behavior intact when GPU channel is full/disconnected.
3. Tune coordinator batching fairness to avoid starvation between Stage0, StageN, and fused work.

Acceptance:

1. Lower `gpu_handoff_ns` and lower fallback counts under equivalent load where expected.
2. Throughput improvement on WebGPU matrix with no ratio/correctness regression.

### Phase 4: Adaptive Auto Routing (No New Path)

Files:

1. `src/pipeline/parallel.rs`
2. `src/pipeline/tests.rs`

Tasks:

1. For `BackendAssignment::Auto` only, add adaptive routing heuristics using observed timings.
2. Preserve strict behavior for explicit `Cpu` and `Gpu` assignments.
3. Add deterministic tests for routing decisions and fallback semantics.

Acceptance:

1. Auto-routing median throughput improves or remains neutral across the local matrix.
2. Tests validate that explicit backend assignments are never overridden.

## Validation Protocol (Agent-On-Commit, Same Laptop)

Run on each perf-affecting commit:

1. `cargo test`
2. `cargo test --features webgpu pipeline::parallel::tests` (when WebGPU feature/device is available)
3. `./scripts/perf-gate.sh`
4. `./scripts/profile.sh --pipeline lzr --size 1048576 --iterations 120 --features webgpu` (if WebGPU available)

Attach generated artifacts to `docs/generated/` with date-stamped names and summarize deltas in this plan file or companion notes.

## Handoff Checklist for Implementing Agent

1. Work only on unified scheduler behavior; do not add non-unified fast paths.
2. Land Phase 0 before optimization phases.
3. Keep each phase as a separate commit for clean rollback and measurement.
4. Include before/after metrics table in commit message or companion note.
5. If a regression is accepted, document rationale and next action in this plan.

## Risks and Mitigations

1. Instrumentation overhead may pollute measurements.
   1. Mitigation: compile-time/runtime gating and quick A/B check with instrumentation off.
2. Hardware noise can obscure small gains.
   1. Mitigation: fixed local matrix, repeated runs, median reporting.
3. Queue changes can introduce subtle concurrency bugs.
   1. Mitigation: preserve invariants, keep tests broad, ship in small steps.
