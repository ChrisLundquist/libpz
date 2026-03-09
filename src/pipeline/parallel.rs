//! Multi-block parallel compression and decompression.
//!
//! **Unified scheduler**: a single shared work queue executes all pipeline stages from a
//! worker pool. Supports N-stage pipelines (2-stage LZ, 4-stage BWT) and GPU dispatch
//! via a dedicated GPU coordinator thread when a WebGPU device is available.
//!
//! GPU and CPU encoders are interchangeable: the same queue handles both, with the
//! coordinator thread dispatching GPU work via `run_compress_stage` and feeding
//! results back into the queue for subsequent stages.

use crate::{PzError, PzResult};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};
use std::time::Instant;

use super::stages::{run_compress_stage, StageBlock, StageMetadata};
use super::telemetry::{
    LocalSchedulerStats, SchedulerRunRecorder, UNIFIED_SCHEDULER_STATS_ENABLED,
};
use super::{write_header, CompressOptions, DecompressOptions, Pipeline, BLOCK_HEADER_SIZE};

/// Multi-block parallel compression.
///
/// All pipelines flow through the unified scheduler. When a WebGPU device is
/// available, a dedicated GPU coordinator thread handles ring-buffered GPU
/// submissions while the CPU worker pool continues processing other blocks.
///
/// Format after the 8-byte container header:
/// - num_blocks: u32 LE
/// - block_table: [compressed_len: u32 LE, original_len: u32 LE] * num_blocks
/// - block_data: concatenated compressed block bytes
pub(crate) fn compress_parallel(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
    num_threads: usize,
) -> PzResult<Vec<u8>> {
    let block_size = options.block_size;
    let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
    compress_parallel_unified(input, pipeline, options, num_threads, &blocks)
}

fn assemble_multiblock_container(
    pipeline: Pipeline,
    original_len: usize,
    orig_block_lens: &[usize],
    block_data: &[&[u8]],
) -> Vec<u8> {
    let num_blocks = block_data.len();
    debug_assert_eq!(orig_block_lens.len(), num_blocks);

    let mut output = Vec::new();
    write_header(&mut output, pipeline, original_len);
    output.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    for (i, compressed) in block_data.iter().enumerate() {
        let orig_block_len = orig_block_lens[i] as u32;
        let comp_block_len = compressed.len() as u32;
        output.extend_from_slice(&comp_block_len.to_le_bytes());
        output.extend_from_slice(&orig_block_len.to_le_bytes());
    }
    for compressed in block_data {
        output.extend_from_slice(compressed);
    }
    output
}

#[derive(Copy, Clone)]
#[cfg_attr(not(feature = "webgpu"), allow(dead_code))]
enum UnifiedTask {
    Stage(usize, usize),    // (stage_idx, block_idx) — CPU execution
    StageGpu(usize, usize), // (stage_idx, block_idx) — route to GPU coordinator
    /// Fused GPU execution: run stages start..=end on the GPU coordinator
    /// without intermediate queue round-trips. (stage_start, stage_end, block_idx)
    FusedGpu(usize, usize, usize),
}

/// Number of compression stages for a pipeline in the unified scheduler.
fn unified_stage_count(pipeline: Pipeline) -> usize {
    match pipeline {
        Pipeline::Bw | Pipeline::Bbw => 4,
        Pipeline::SortLz | Pipeline::Parlz => 1,
        _ => 2,
    }
}

#[derive(Default)]
struct UnifiedQueueState {
    queue: VecDeque<UnifiedTask>,
    pending_tasks: usize,
    closed: bool,
    failed: bool,
}

fn should_route_block_to_gpu_entropy_with_backpressure(
    block_len: usize,
    stage1_backend: super::BackendAssignment,
    has_gpu_entropy: bool,
    auto_backpressure_score: usize,
    auto_backpressure_limit: usize,
) -> bool {
    #[cfg(feature = "webgpu")]
    {
        use super::{BackendAssignment, GPU_ENTROPY_THRESHOLD};
        match stage1_backend {
            BackendAssignment::Gpu => has_gpu_entropy,
            BackendAssignment::Cpu => false,
            BackendAssignment::Auto => {
                has_gpu_entropy
                    && block_len >= GPU_ENTROPY_THRESHOLD
                    && auto_backpressure_score < auto_backpressure_limit
            }
        }
    }
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = block_len;
        let _ = stage1_backend;
        let _ = has_gpu_entropy;
        let _ = auto_backpressure_score;
        let _ = auto_backpressure_limit;
        false
    }
}

#[cfg(feature = "webgpu")]
#[inline]
fn pressure_inc(score: &AtomicUsize, delta: usize) {
    score.fetch_add(delta, Ordering::Relaxed);
}

#[cfg(feature = "webgpu")]
#[inline]
fn pressure_dec(score: &AtomicUsize) {
    let _ = score.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        Some(v.saturating_sub(1))
    });
}

/// Determine whether to route a block's Stage 0 (LZ77/LzSeq match-finding) to GPU.
///
/// Returns true for LZ77-based and LzSeq-based pipelines when the GPU backend
/// is active, the block meets size constraints, and the stage0_backend allows it.
#[cfg(feature = "webgpu")]
fn should_route_block_to_gpu_stage0(
    block: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> bool {
    use super::BackendAssignment;

    // Only LZ-based pipelines have GPU Stage 0 support
    let is_gpu_stage0_pipeline = matches!(
        pipeline,
        Pipeline::Deflate
            | Pipeline::Lzr
            | Pipeline::Lzf
            | Pipeline::Lzfi
            | Pipeline::LzssR
            | Pipeline::LzSeqR
            | Pipeline::LzSeqH
    );

    if !is_gpu_stage0_pipeline {
        return false;
    }

    // Optimal parse uses GPU top-K which runs through run_compress_stage already
    if options.parse_strategy == super::ParseStrategy::Optimal {
        return false;
    }

    match options.stage0_backend {
        BackendAssignment::Gpu => options.webgpu_engine.is_some(),
        BackendAssignment::Cpu => false,
        BackendAssignment::Auto => {
            if let Some(ref engine) = options.webgpu_engine {
                block.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE
                    && block.len() <= engine.max_dispatch_input_size()
            } else {
                false
            }
        }
    }
}

/// Determine whether a pipeline can fuse all its stages on GPU.
///
/// Returns `Some((start, end))` when every stage in `start..=end` has a GPU
/// implementation, allowing the GPU coordinator to run them sequentially
/// without intermediate queue round-trips or CPU readback.
#[cfg(feature = "webgpu")]
fn gpu_fused_span(pipeline: Pipeline) -> Option<(usize, usize)> {
    match pipeline {
        // Both stages have GPU paths: LZ77 match-finding + rANS encode
        Pipeline::Lzr => Some((0, 1)),
        // Both stages have GPU paths: LzSeq fused match+demux + rANS encode
        Pipeline::LzSeqR => Some((0, 1)),
        _ => None,
    }
}

/// Message sent from unified workers to the GPU coordinator thread.
#[cfg(feature = "webgpu")]
enum GpuRequest {
    /// Stage 0: LZ77/LzSeq match-finding on GPU.
    /// Carries only `block_idx`: the coordinator reads immutable input slices
    /// directly from the parent-scoped `blocks` array to avoid payload copies.
    Stage0(usize),
    /// Stage 1+: entropy encoding on GPU.
    /// (stage_idx, block_idx, stage_block)
    StageN(usize, usize, StageBlock),
    /// Fused: run stages start..=end on GPU without queue round-trips.
    /// (stage_start, stage_end, block_idx)
    Fused(usize, usize, usize),
}

/// Apply unified queue completion semantics for one finished task.
///
/// Returns `(mark_invalid_after_lock, should_return_worker)`.
fn complete_task_lifecycle(
    guard: &mut UnifiedQueueState,
    queue_cv: &Condvar,
    next_task: Option<UnifiedTask>,
    stage_failed: bool,
    mark_invalid_on_failed_final: bool,
) -> (bool, bool) {
    let mut mark_invalid_after_lock = false;
    let mut should_return = false;

    if stage_failed {
        if !guard.failed {
            guard.failed = true;
            let dropped = guard.queue.len();
            guard.queue.clear();
            guard.pending_tasks = guard.pending_tasks.saturating_sub(dropped);
            queue_cv.notify_all();
        }
        debug_assert!(guard.pending_tasks > 0);
        guard.pending_tasks -= 1;
    } else if let Some(task) = next_task {
        if !guard.failed {
            // Current task transitions into next stage; pending is unchanged.
            guard.queue.push_back(task);
            queue_cv.notify_one();
        } else {
            // Scheduler already failed; retire this task.
            debug_assert!(guard.pending_tasks > 0);
            guard.pending_tasks -= 1;
            mark_invalid_after_lock = true;
        }
    } else {
        if mark_invalid_on_failed_final && guard.failed {
            mark_invalid_after_lock = true;
        }
        // Final-stage success retires one pending task.
        debug_assert!(guard.pending_tasks > 0);
        guard.pending_tasks -= 1;
    }

    if guard.pending_tasks == 0 {
        guard.closed = true;
        queue_cv.notify_all();
        should_return = true;
    }

    (mark_invalid_after_lock, should_return)
}

fn compress_parallel_unified(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
    num_threads: usize,
    blocks: &[&[u8]],
) -> PzResult<Vec<u8>> {
    let num_blocks = blocks.len();
    let num_stages = unified_stage_count(pipeline);
    let last_stage = num_stages - 1;
    let worker_count = num_threads.min(num_blocks).max(1);
    let stats_local = if UNIFIED_SCHEDULER_STATS_ENABLED.load(Ordering::Relaxed) {
        Some(Arc::new(LocalSchedulerStats::default()))
    } else {
        None
    };
    let _stats_run = SchedulerRunRecorder::new(stats_local.clone());

    let mut resolved_options = options.clone();
    if resolved_options.max_match_len.is_none() {
        resolved_options.max_match_len = Some(super::resolve_max_match_len(pipeline, options));
    }

    // Per-block intermediate slot: holds the StageBlock between stages.
    // A block is only in one stage at a time, so one slot per block suffices.
    let intermediate_slots: Vec<Mutex<Option<StageBlock>>> =
        (0..num_blocks).map(|_| Mutex::new(None)).collect();
    let results: Vec<Mutex<Option<PzResult<Vec<u8>>>>> =
        (0..num_blocks).map(|_| Mutex::new(None)).collect();

    // Determine if any GPU routing is possible for initial task enqueue.
    #[cfg(feature = "webgpu")]
    let has_gpu =
        matches!(options.backend, super::Backend::WebGpu) && options.webgpu_engine.is_some();
    #[cfg(not(feature = "webgpu"))]
    let has_gpu = false;
    #[cfg(feature = "webgpu")]
    let gpu_auto_backpressure = if has_gpu {
        Some(Arc::new(AtomicUsize::new(0)))
    } else {
        None
    };
    #[cfg(not(feature = "webgpu"))]
    let gpu_auto_backpressure: Option<Arc<AtomicUsize>> = None;
    #[cfg(feature = "webgpu")]
    let gpu_auto_backpressure_limit = worker_count.saturating_mul(2).max(4);
    #[cfg(not(feature = "webgpu"))]
    let gpu_auto_backpressure_limit = 0usize;

    // Determine if this pipeline can fuse all GPU stages.
    #[cfg(feature = "webgpu")]
    let fused_span = if has_gpu {
        gpu_fused_span(pipeline)
    } else {
        None
    };
    #[cfg(not(feature = "webgpu"))]
    let fused_span: Option<(usize, usize)> = None;

    // Build initial task queue with GPU routing for Stage 0 where applicable.
    let initial_tasks: VecDeque<UnifiedTask> = (0..num_blocks)
        .map(|i| {
            #[cfg(feature = "webgpu")]
            if has_gpu && should_route_block_to_gpu_stage0(blocks[i], pipeline, &resolved_options) {
                // If the pipeline supports fusion and entropy also qualifies for GPU,
                // fuse all stages into a single GPU coordinator dispatch.
                if let Some((start, end)) = fused_span {
                    if should_route_block_to_gpu_entropy_with_backpressure(
                        blocks[i].len(),
                        resolved_options.stage1_backend,
                        resolved_options.webgpu_engine.is_some(),
                        0,
                        gpu_auto_backpressure_limit,
                    ) {
                        return UnifiedTask::FusedGpu(start, end, i);
                    }
                }
                return UnifiedTask::StageGpu(0, i);
            }
            let _ = has_gpu;
            let _ = fused_span;
            UnifiedTask::Stage(0, i)
        })
        .collect();

    let queue = Mutex::new(UnifiedQueueState {
        queue: initial_tasks,
        pending_tasks: num_blocks,
        closed: false,
        failed: false,
    });
    let queue_cv = Condvar::new();

    // GPU channel: created before the scope so workers can borrow the sender.
    // The GPU coordinator receives from `gpu_rx`, workers send via `gpu_tx`.
    #[cfg(feature = "webgpu")]
    let (gpu_tx, gpu_rx) = if has_gpu {
        // Use a modestly deeper channel than the previous fixed depth=4 to
        // reduce transient try_send(Full) fallbacks under bursty mixed-stage
        // loads without unbounded buffering.
        let ring_depth = num_blocks
            .min(worker_count.saturating_mul(2).max(1))
            .clamp(1, 16);
        let (tx, rx) = std::sync::mpsc::sync_channel::<GpuRequest>(ring_depth);
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    #[cfg(not(feature = "webgpu"))]
    let gpu_tx: Option<()> = None;

    std::thread::scope(|scope| {
        // GPU coordinator thread: spawned only when GPU is available.
        // Receives GPU work via a bounded channel, manages ring buffers,
        // and feeds completed results back into the unified queue.
        #[cfg(feature = "webgpu")]
        if let Some(rx) = gpu_rx {
            let queue_ref = &queue;
            let cv_ref = &queue_cv;
            let slots_ref = &intermediate_slots;
            let results_ref = &results;
            let opts = resolved_options.clone();
            let stats_ref = stats_local.clone();
            let gpu_pressure_ref = gpu_auto_backpressure.clone();
            let gpu_pressure_limit = gpu_auto_backpressure_limit;

            scope.spawn(move || {
                let engine = opts.webgpu_engine.as_ref().unwrap();
                let uses_lz77_demux =
                    matches!(pipeline, Pipeline::Deflate | Pipeline::Lzr | Pipeline::Lzf);
                let uses_sortlz_match_finder =
                    opts.match_finder == super::MatchFinder::SortLz && uses_lz77_demux;

                while let Ok(first) = rx.recv() {
                    // Batch-collect: drain additional pending requests.
                    let mut stage0_batch: Vec<usize> = Vec::new();
                    let mut stage_n_queue: Vec<(usize, usize, StageBlock)> = Vec::new();
                    let mut fused_queue: Vec<(usize, usize, usize)> = Vec::new();

                    // Classify the first request.
                    match first {
                        GpuRequest::Stage0(b) => stage0_batch.push(b),
                        GpuRequest::StageN(s, b, sb) => stage_n_queue.push((s, b, sb)),
                        GpuRequest::Fused(s, e, b) => fused_queue.push((s, e, b)),
                    }

                    // Non-blocking drain of additional requests.
                    while let Ok(req) = rx.try_recv() {
                        match req {
                            GpuRequest::Stage0(b) => stage0_batch.push(b),
                            GpuRequest::StageN(s, b, sb) => stage_n_queue.push((s, b, sb)),
                            GpuRequest::Fused(s, e, b) => fused_queue.push((s, e, b)),
                        }
                    }

                    // Process Stage N requests first (fairness): these are
                    // downstream continuations and completing them reduces
                    // in-flight work / pending pressure.
                    for (stage_idx, block_idx, sb) in stage_n_queue {
                        let t0 = Instant::now();
                        let result = run_compress_stage(pipeline, stage_idx, sb, &opts);
                        if let Some(stats) = stats_ref.as_ref() {
                            stats.add_stage_compute(t0.elapsed());
                        }
                        if result.is_err() {
                            // GPU entropy failed — restart from stage 0 on CPU.
                            // The intermediate StageBlock is consumed so we cannot
                            // retry just this stage; re-enqueue from scratch.
                            eprintln!(
                                "[pz-gpu] stage {stage_idx} failed for block {block_idx}; \
                                 retrying from stage 0 on CPU"
                            );
                            let lock_start = Instant::now();
                            let mut guard = queue_ref.lock().expect("unified queue poisoned");
                            if let Some(stats) = stats_ref.as_ref() {
                                stats.add_queue_wait(lock_start.elapsed());
                            }
                            let admin_start = Instant::now();
                            if !guard.failed {
                                guard.queue.push_back(UnifiedTask::Stage(0, block_idx));
                                cv_ref.notify_one();
                            }
                            if let Some(stats) = stats_ref.as_ref() {
                                stats.add_queue_admin(admin_start.elapsed());
                            }
                            drop(guard);
                        } else {
                            complete_gpu_stage(
                                result,
                                stage_idx,
                                block_idx,
                                last_stage,
                                blocks,
                                &opts,
                                slots_ref,
                                results_ref,
                                queue_ref,
                                cv_ref,
                                stats_ref.as_deref(),
                                gpu_pressure_ref.as_deref(),
                                gpu_pressure_limit,
                            );
                        }
                    }

                    // Process fused requests next: run stages start..=end sequentially
                    // on GPU without intermediate queue round-trips.
                    for (stage_start, stage_end, block_idx) in fused_queue {
                        let block = StageBlock {
                            block_index: block_idx,
                            original_len: blocks[block_idx].len(),
                            data: blocks[block_idx].to_vec(),
                            streams: None,
                            metadata: StageMetadata::default(),
                        };
                        let mut result: PzResult<StageBlock> = Ok(block);
                        let mut final_stage = stage_start;
                        for stage in stage_start..=stage_end {
                            match result {
                                Ok(sb) => {
                                    let t0 = Instant::now();
                                    result = run_compress_stage(pipeline, stage, sb, &opts);
                                    if let Some(stats) = stats_ref.as_ref() {
                                        stats.add_stage_compute(t0.elapsed());
                                    }
                                    final_stage = stage;
                                }
                                Err(_) => break,
                            }
                        }
                        if result.is_err() {
                            // GPU fused path failed — fall back to per-stage CPU.
                            // Re-enqueue from stage 0 since intermediate data is consumed.
                            eprintln!(
                                "[pz-gpu] fused stages {stage_start}..={stage_end} failed \
                                 for block {block_idx}; retrying on CPU"
                            );
                            let lock_start = Instant::now();
                            let mut guard = queue_ref.lock().expect("unified queue poisoned");
                            if let Some(stats) = stats_ref.as_ref() {
                                stats.add_queue_wait(lock_start.elapsed());
                            }
                            let admin_start = Instant::now();
                            if !guard.failed {
                                guard.queue.push_back(UnifiedTask::Stage(0, block_idx));
                                cv_ref.notify_one();
                            }
                            if let Some(stats) = stats_ref.as_ref() {
                                stats.add_queue_admin(admin_start.elapsed());
                            }
                            drop(guard);
                        } else {
                            complete_gpu_stage(
                                result,
                                final_stage,
                                block_idx,
                                last_stage,
                                blocks,
                                &opts,
                                slots_ref,
                                results_ref,
                                queue_ref,
                                cv_ref,
                                stats_ref.as_deref(),
                                gpu_pressure_ref.as_deref(),
                                gpu_pressure_limit,
                            );
                        }
                    }

                    // Process Stage 0 batch last to avoid starving queued StageN/Fused
                    // continuations when bursts arrive together.
                    if !stage0_batch.is_empty() && uses_sortlz_match_finder {
                        // SortLZ GPU match finding: per-block dispatch with LZ77 conversion
                        let sortlz_config = crate::sortlz::SortLzConfig::for_lz77(
                            opts.max_match_len.unwrap_or(crate::lz77::DEFLATE_MAX_MATCH),
                        );
                        for block_idx in stage0_batch {
                            let t0 = Instant::now();
                            let result = engine
                                .sortlz_find_matches(blocks[block_idx], &sortlz_config)
                                .map(|raw_matches| {
                                    let lz_matches = crate::sortlz::matches_to_lz77_lazy(
                                        blocks[block_idx],
                                        &raw_matches,
                                    );
                                    let demux = super::demux::demux_lz77_matches(lz_matches);
                                    StageBlock {
                                        block_index: block_idx,
                                        original_len: blocks[block_idx].len(),
                                        data: Vec::new(),
                                        streams: Some(demux.streams),
                                        metadata: StageMetadata {
                                            pre_entropy_len: Some(demux.pre_entropy_len),
                                            demux_meta: demux.meta,
                                            ..StageMetadata::default()
                                        },
                                    }
                                });
                            if let Some(stats) = stats_ref.as_ref() {
                                stats.add_stage_compute(t0.elapsed());
                            }
                            complete_gpu_stage(
                                result,
                                0,
                                block_idx,
                                last_stage,
                                blocks,
                                &opts,
                                slots_ref,
                                results_ref,
                                queue_ref,
                                cv_ref,
                                stats_ref.as_deref(),
                                gpu_pressure_ref.as_deref(),
                                gpu_pressure_limit,
                            );
                        }
                    } else if !stage0_batch.is_empty() && uses_lz77_demux {
                        let batch_blocks: Vec<&[u8]> =
                            stage0_batch.iter().map(|&b| blocks[b]).collect();
                        let t0 = Instant::now();
                        let batch_results = engine.find_matches_batched(&batch_blocks);
                        if let Some(stats) = stats_ref.as_ref() {
                            stats.add_stage_compute(t0.elapsed());
                        }
                        match batch_results {
                            Ok(all_matches) => {
                                for (matches, block_idx) in
                                    all_matches.into_iter().zip(stage0_batch.iter().copied())
                                {
                                    let demux = super::demux::demux_lz77_matches(matches);
                                    let sb = StageBlock {
                                        block_index: block_idx,
                                        original_len: blocks[block_idx].len(),
                                        data: Vec::new(),
                                        streams: Some(demux.streams),
                                        metadata: StageMetadata {
                                            pre_entropy_len: Some(demux.pre_entropy_len),
                                            demux_meta: demux.meta,
                                            ..StageMetadata::default()
                                        },
                                    };
                                    complete_gpu_stage(
                                        Ok(sb),
                                        0,
                                        block_idx,
                                        last_stage,
                                        blocks,
                                        &opts,
                                        slots_ref,
                                        results_ref,
                                        queue_ref,
                                        cv_ref,
                                        stats_ref.as_deref(),
                                        gpu_pressure_ref.as_deref(),
                                        gpu_pressure_limit,
                                    );
                                }
                            }
                            Err(e) => {
                                // GPU batch failed — fall back to CPU for each block.
                                eprintln!(
                                    "[pz-gpu] find_matches_batched failed: {e}; \
                                     retrying {} blocks on CPU",
                                    stage0_batch.len()
                                );
                                let lock_start = Instant::now();
                                let mut guard = queue_ref.lock().expect("unified queue poisoned");
                                if let Some(stats) = stats_ref.as_ref() {
                                    stats.add_queue_wait(lock_start.elapsed());
                                }
                                let admin_start = Instant::now();
                                if !guard.failed {
                                    for block_idx in &stage0_batch {
                                        guard.queue.push_back(UnifiedTask::Stage(0, *block_idx));
                                    }
                                    cv_ref.notify_all();
                                }
                                if let Some(stats) = stats_ref.as_ref() {
                                    stats.add_queue_admin(admin_start.elapsed());
                                }
                                drop(guard);
                            }
                        }
                    } else {
                        // Non-LZ77 pipelines (LzSeq, LZSS): dispatch individually
                        // through run_compress_stage which calls lzseq_encode_gpu etc.
                        for block_idx in stage0_batch {
                            let block = StageBlock {
                                block_index: block_idx,
                                original_len: blocks[block_idx].len(),
                                data: blocks[block_idx].to_vec(),
                                streams: None,
                                metadata: StageMetadata::default(),
                            };
                            let t0 = Instant::now();
                            let result = run_compress_stage(pipeline, 0, block, &opts);
                            if let Some(stats) = stats_ref.as_ref() {
                                stats.add_stage_compute(t0.elapsed());
                            }
                            if result.is_err() {
                                // GPU stage 0 failed — retry on CPU.
                                eprintln!(
                                    "[pz-gpu] stage 0 failed for block {block_idx}; \
                                     retrying on CPU"
                                );
                                let lock_start = Instant::now();
                                let mut guard = queue_ref.lock().expect("unified queue poisoned");
                                if let Some(stats) = stats_ref.as_ref() {
                                    stats.add_queue_wait(lock_start.elapsed());
                                }
                                let admin_start = Instant::now();
                                if !guard.failed {
                                    guard.queue.push_back(UnifiedTask::Stage(0, block_idx));
                                    cv_ref.notify_one();
                                }
                                if let Some(stats) = stats_ref.as_ref() {
                                    stats.add_queue_admin(admin_start.elapsed());
                                }
                                drop(guard);
                            } else {
                                complete_gpu_stage(
                                    result,
                                    0,
                                    block_idx,
                                    last_stage,
                                    blocks,
                                    &opts,
                                    slots_ref,
                                    results_ref,
                                    queue_ref,
                                    cv_ref,
                                    stats_ref.as_deref(),
                                    gpu_pressure_ref.as_deref(),
                                    gpu_pressure_limit,
                                );
                            }
                        }
                    }
                }
            });
        }

        // CPU worker threads — each gets a clone of the GPU sender (if any).
        // Clones are cheap (Arc internally) and avoid borrow issues with scoped threads.
        for _ in 0..worker_count {
            let queue_ref = &queue;
            let cv_ref = &queue_cv;
            let slots_ref = &intermediate_slots;
            let results_ref = &results;
            let opts = resolved_options.clone();
            let stats_ref = stats_local.clone();
            let gpu_pressure_ref = gpu_auto_backpressure.clone();
            let gpu_pressure_limit = gpu_auto_backpressure_limit;
            #[cfg(feature = "webgpu")]
            let gpu_tx_clone = gpu_tx.clone();

            scope.spawn(move || {
                // Local continuation state for this worker.
                // When set, process the next stage directly instead of
                // round-tripping through the shared queue.
                let mut local_task: Option<(usize, usize, Option<StageBlock>)> = None;

                loop {
                    let (stage_idx, block_idx, inline_block) = if let Some(task) = local_task.take()
                    {
                        task
                    } else {
                        let task = {
                            let lock_start = Instant::now();
                            let mut guard = queue_ref.lock().expect("unified queue poisoned");
                            if let Some(stats) = stats_ref.as_ref() {
                                stats.add_queue_wait(lock_start.elapsed());
                            }
                            loop {
                                let admin_start = Instant::now();
                                if let Some(task) = guard.queue.pop_front() {
                                    if let Some(stats) = stats_ref.as_ref() {
                                        stats.add_queue_admin(admin_start.elapsed());
                                    }
                                    break task;
                                }
                                if let Some(stats) = stats_ref.as_ref() {
                                    stats.add_queue_admin(admin_start.elapsed());
                                }
                                if guard.closed || (guard.failed && guard.queue.is_empty()) {
                                    return;
                                }
                                let wait_start = Instant::now();
                                guard = cv_ref.wait(guard).expect("unified queue wait poisoned");
                                if let Some(stats) = stats_ref.as_ref() {
                                    stats.add_queue_wait(wait_start.elapsed());
                                }
                            }
                        };

                        match task {
                            UnifiedTask::Stage(s, b) => (s, b, None),
                            #[allow(unused_variables)]
                            UnifiedTask::FusedGpu(start, end, b) => {
                                // Route to GPU coordinator for fused multi-stage execution.
                                #[cfg(feature = "webgpu")]
                                if let Some(ref tx) = gpu_tx_clone {
                                    let handoff_start = Instant::now();
                                    let request = GpuRequest::Fused(start, end, b);
                                    match tx.try_send(request) {
                                        Ok(()) => {
                                            if let Some(stats) = stats_ref.as_ref() {
                                                stats.add_gpu_handoff(handoff_start.elapsed());
                                            }
                                            if let Some(score) = gpu_pressure_ref.as_ref() {
                                                pressure_dec(score);
                                            }
                                            continue;
                                        }
                                        Err(std::sync::mpsc::TrySendError::Full(_)) => {
                                            if let Some(stats) = stats_ref.as_ref() {
                                                stats.add_gpu_handoff(handoff_start.elapsed());
                                                stats.inc_gpu_try_send_full();
                                            }
                                            if let Some(score) = gpu_pressure_ref.as_ref() {
                                                pressure_inc(score, 2);
                                            }
                                            // Channel full — fall through to CPU stage start.
                                        }
                                        Err(std::sync::mpsc::TrySendError::Disconnected(_)) => {
                                            if let Some(stats) = stats_ref.as_ref() {
                                                stats.add_gpu_handoff(handoff_start.elapsed());
                                                stats.inc_gpu_try_send_disconnected();
                                            }
                                            if let Some(score) = gpu_pressure_ref.as_ref() {
                                                pressure_inc(score, 1);
                                            }
                                            // Channel closed — fall through to CPU stage start.
                                        }
                                    }
                                }
                                (start, b, None)
                            }
                            UnifiedTask::StageGpu(s, b) => {
                                #[cfg(feature = "webgpu")]
                                {
                                    // Route to GPU coordinator — send and continue to next task.
                                    if let Some(ref tx) = gpu_tx_clone {
                                        let request = if s == 0 {
                                            GpuRequest::Stage0(b)
                                        } else {
                                            let stage_block = slots_ref[b]
                                                .lock()
                                                .expect("intermediate slot poisoned")
                                                .take()
                                                .expect("intermediate result missing");
                                            GpuRequest::StageN(s, b, stage_block)
                                        };
                                        let handoff_start = Instant::now();
                                        let inline_stage_block = match tx.try_send(request) {
                                            Ok(()) => {
                                                if let Some(stats) = stats_ref.as_ref() {
                                                    stats.add_gpu_handoff(handoff_start.elapsed());
                                                }
                                                if let Some(score) = gpu_pressure_ref.as_ref() {
                                                    pressure_dec(score);
                                                }
                                                continue;
                                            }
                                            Err(std::sync::mpsc::TrySendError::Full(req)) => {
                                                if let Some(stats) = stats_ref.as_ref() {
                                                    stats.add_gpu_handoff(handoff_start.elapsed());
                                                    stats.inc_gpu_try_send_full();
                                                }
                                                if let Some(score) = gpu_pressure_ref.as_ref() {
                                                    pressure_inc(score, 2);
                                                }
                                                match req {
                                                    // Keep StageN payload local for CPU fallback
                                                    // to avoid slot round-trips under pressure.
                                                    GpuRequest::StageN(_, _, sb) => Some(sb),
                                                    _ => None,
                                                }
                                            }
                                            Err(std::sync::mpsc::TrySendError::Disconnected(
                                                req,
                                            )) => {
                                                if let Some(stats) = stats_ref.as_ref() {
                                                    stats.add_gpu_handoff(handoff_start.elapsed());
                                                    stats.inc_gpu_try_send_disconnected();
                                                }
                                                if let Some(score) = gpu_pressure_ref.as_ref() {
                                                    pressure_inc(score, 1);
                                                }
                                                match req {
                                                    GpuRequest::StageN(_, _, sb) => Some(sb),
                                                    _ => None,
                                                }
                                            }
                                        };
                                        if let Some(sb) = inline_stage_block {
                                            (s, b, Some(sb))
                                        } else {
                                            (s, b, None)
                                        }
                                    } else {
                                        (s, b, None)
                                    }
                                }
                                #[cfg(not(feature = "webgpu"))]
                                {
                                    (s, b, None)
                                }
                            }
                        }
                    };

                    // Build or retrieve the StageBlock for this stage.
                    let block = if let Some(sb) = inline_block {
                        sb
                    } else if stage_idx == 0 {
                        StageBlock {
                            block_index: block_idx,
                            original_len: blocks[block_idx].len(),
                            data: blocks[block_idx].to_vec(),
                            streams: None,
                            metadata: StageMetadata::default(),
                        }
                    } else {
                        slots_ref[block_idx]
                            .lock()
                            .expect("intermediate slot poisoned")
                            .take()
                            .expect("intermediate result missing")
                    };

                    let mut stage_failed = false;
                    let t0 = Instant::now();
                    let result = run_compress_stage(pipeline, stage_idx, block, &opts);
                    if let Some(stats) = stats_ref.as_ref() {
                        stats.add_stage_compute(t0.elapsed());
                    }

                    match result {
                        Ok(stage_block) => {
                            if stage_idx == last_stage {
                                // Final stage: store compressed bytes and retire this block.
                                *results_ref[block_idx].lock().expect("result slot poisoned") =
                                    Some(Ok(stage_block.data));
                            } else {
                                let next_stage = stage_idx + 1;
                                let backpressure_score = gpu_pressure_ref
                                    .as_ref()
                                    .map_or(0usize, |s| s.load(Ordering::Relaxed));
                                let has_gpu = {
                                    #[cfg(feature = "webgpu")]
                                    {
                                        opts.webgpu_engine.is_some()
                                    }
                                    #[cfg(not(feature = "webgpu"))]
                                    {
                                        false
                                    }
                                };
                                let route_next_to_gpu = next_stage == last_stage
                                    && should_route_block_to_gpu_entropy_with_backpressure(
                                        blocks[block_idx].len(),
                                        opts.stage1_backend,
                                        has_gpu,
                                        backpressure_score,
                                        gpu_pressure_limit,
                                    );

                                if route_next_to_gpu {
                                    // Directly hand off StageN to GPU coordinator from this
                                    // worker, avoiding queue and slot round-trips.
                                    #[cfg(feature = "webgpu")]
                                    if let Some(ref tx) = gpu_tx_clone {
                                        let handoff_start = Instant::now();
                                        let request =
                                            GpuRequest::StageN(next_stage, block_idx, stage_block);
                                        match tx.try_send(request) {
                                            Ok(()) => {
                                                if let Some(stats) = stats_ref.as_ref() {
                                                    stats.add_gpu_handoff(handoff_start.elapsed());
                                                }
                                                if let Some(score) = gpu_pressure_ref.as_ref() {
                                                    pressure_dec(score);
                                                }
                                                continue;
                                            }
                                            Err(std::sync::mpsc::TrySendError::Full(req)) => {
                                                if let Some(stats) = stats_ref.as_ref() {
                                                    stats.add_gpu_handoff(handoff_start.elapsed());
                                                    stats.inc_gpu_try_send_full();
                                                }
                                                if let Some(score) = gpu_pressure_ref.as_ref() {
                                                    pressure_inc(score, 2);
                                                }
                                                if let GpuRequest::StageN(_, _, sb) = req {
                                                    local_task =
                                                        Some((next_stage, block_idx, Some(sb)));
                                                    continue;
                                                }
                                                unreachable!("StageN request expected");
                                            }
                                            Err(std::sync::mpsc::TrySendError::Disconnected(
                                                req,
                                            )) => {
                                                if let Some(stats) = stats_ref.as_ref() {
                                                    stats.add_gpu_handoff(handoff_start.elapsed());
                                                    stats.inc_gpu_try_send_disconnected();
                                                }
                                                if let Some(score) = gpu_pressure_ref.as_ref() {
                                                    pressure_inc(score, 1);
                                                }
                                                if let GpuRequest::StageN(_, _, sb) = req {
                                                    local_task =
                                                        Some((next_stage, block_idx, Some(sb)));
                                                    continue;
                                                }
                                                unreachable!("StageN request expected");
                                            }
                                        }
                                    }
                                }

                                // CPU continuation: keep processing the same block locally.
                                local_task = Some((next_stage, block_idx, Some(stage_block)));
                                continue;
                            }
                        }
                        Err(e) => {
                            *results_ref[block_idx].lock().expect("result slot poisoned") =
                                Some(Err(e));
                            stage_failed = true;
                        }
                    }

                    // Single completion lock when a block retires (final success/error).
                    let lock_start = Instant::now();
                    let mut guard = queue_ref.lock().expect("unified queue poisoned");
                    if let Some(stats) = stats_ref.as_ref() {
                        stats.add_queue_wait(lock_start.elapsed());
                    }
                    let admin_start = Instant::now();
                    let (mark_invalid_after_lock, should_return) =
                        complete_task_lifecycle(&mut guard, cv_ref, None, stage_failed, true);
                    if let Some(stats) = stats_ref.as_ref() {
                        stats.add_queue_admin(admin_start.elapsed());
                    }
                    drop(guard);

                    if mark_invalid_after_lock {
                        *results_ref[block_idx].lock().expect("result slot poisoned") =
                            Some(Err(PzError::InvalidInput));
                    }
                    if should_return {
                        return;
                    }
                }
            });
        }

        // Drop the original sender so it doesn't keep the channel open.
        // Workers hold clones; when the last worker exits the scope,
        // all clones drop, closing the channel and unblocking the
        // GPU coordinator's recv().
        #[cfg(feature = "webgpu")]
        drop(gpu_tx);
        #[cfg(not(feature = "webgpu"))]
        let _ = gpu_tx;
    });

    let mut block_data_vec = Vec::with_capacity(num_blocks);
    for slot in results {
        let result = slot
            .into_inner()
            .expect("result slot poisoned")
            .unwrap_or(Err(PzError::InvalidInput))?;
        block_data_vec.push(result);
    }

    let orig_block_lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();
    let block_slices: Vec<&[u8]> = block_data_vec.iter().map(|v| v.as_slice()).collect();
    Ok(assemble_multiblock_container(
        pipeline,
        input.len(),
        &orig_block_lens,
        &block_slices,
    ))
}

/// Handle the result of a GPU stage execution: store results or errors,
/// enqueue the next stage, and manage queue lifecycle (pending_tasks, closed, failed).
///
/// Shared by all GPU request types to avoid duplicating the stage-completion logic.
#[cfg(feature = "webgpu")]
#[allow(clippy::too_many_arguments)]
fn complete_gpu_stage(
    result: PzResult<StageBlock>,
    stage_idx: usize,
    block_idx: usize,
    last_stage: usize,
    blocks: &[&[u8]],
    options: &CompressOptions,
    intermediate_slots: &[Mutex<Option<StageBlock>>],
    results: &[Mutex<Option<PzResult<Vec<u8>>>>],
    queue: &Mutex<UnifiedQueueState>,
    queue_cv: &Condvar,
    stats: Option<&LocalSchedulerStats>,
    gpu_pressure: Option<&AtomicUsize>,
    gpu_pressure_limit: usize,
) {
    let mut next_task: Option<UnifiedTask> = None;
    let mut stage_failed = false;

    match result {
        Ok(sb) => {
            if stage_idx == last_stage {
                *results[block_idx].lock().expect("result slot poisoned") = Some(Ok(sb.data));
            } else {
                *intermediate_slots[block_idx]
                    .lock()
                    .expect("intermediate slot poisoned") = Some(sb);
                let next_stage = stage_idx + 1;
                let backpressure_score = gpu_pressure.map_or(0usize, |s| s.load(Ordering::Relaxed));
                let has_gpu = {
                    #[cfg(feature = "webgpu")]
                    {
                        options.webgpu_engine.is_some()
                    }
                    #[cfg(not(feature = "webgpu"))]
                    {
                        false
                    }
                };
                next_task = Some(
                    if next_stage == last_stage
                        && should_route_block_to_gpu_entropy_with_backpressure(
                            blocks[block_idx].len(),
                            options.stage1_backend,
                            has_gpu,
                            backpressure_score,
                            gpu_pressure_limit,
                        )
                    {
                        UnifiedTask::StageGpu(next_stage, block_idx)
                    } else {
                        UnifiedTask::Stage(next_stage, block_idx)
                    },
                );
            }
        }
        Err(e) => {
            *results[block_idx].lock().expect("result slot poisoned") = Some(Err(e));
            stage_failed = true;
        }
    }

    // Single completion lock per GPU-finished task.
    let lock_start = Instant::now();
    let mut guard = queue.lock().expect("unified queue poisoned");
    if let Some(stats) = stats {
        stats.add_queue_wait(lock_start.elapsed());
    }
    let admin_start = Instant::now();
    let (mark_invalid_after_lock, _) =
        complete_task_lifecycle(&mut guard, queue_cv, next_task, stage_failed, false);
    if let Some(stats) = stats {
        stats.add_queue_admin(admin_start.elapsed());
    }
    drop(guard);

    if mark_invalid_after_lock {
        *results[block_idx].lock().expect("result slot poisoned") =
            Some(Err(PzError::InvalidInput));
    }
}

/// Multi-block parallel decompression.
pub(crate) fn decompress_parallel(
    payload: &[u8],
    pipeline: Pipeline,
    orig_len: usize,
    num_blocks: usize,
    options: &DecompressOptions,
) -> PzResult<Vec<u8>> {
    let num_threads = super::resolve_thread_count(options.threads);

    // Parse block table (starts after num_blocks field)
    let table_start = 4; // skip num_blocks u32
    let table_size = num_blocks * BLOCK_HEADER_SIZE;
    if payload.len() < table_start + table_size {
        return Err(PzError::InvalidInput);
    }

    let mut block_entries: Vec<(usize, usize)> = Vec::with_capacity(num_blocks); // (compressed_len, original_len)
    let mut total_orig = 0usize;
    for i in 0..num_blocks {
        let offset = table_start + i * BLOCK_HEADER_SIZE;
        let comp_len = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]) as usize;
        let orig_block_len = u32::from_le_bytes([
            payload[offset + 4],
            payload[offset + 5],
            payload[offset + 6],
            payload[offset + 7],
        ]) as usize;
        block_entries.push((comp_len, orig_block_len));
        total_orig += orig_block_len;
    }

    if total_orig != orig_len {
        return Err(PzError::InvalidInput);
    }

    // Locate each block's compressed data
    let data_start = table_start + table_size;
    let mut block_slices: Vec<(&[u8], usize)> = Vec::with_capacity(num_blocks); // (compressed_data, original_len)
    let mut pos = data_start;
    for &(comp_len, orig_block_len) in &block_entries {
        if pos + comp_len > payload.len() {
            return Err(PzError::InvalidInput);
        }
        block_slices.push((&payload[pos..pos + comp_len], orig_block_len));
        pos += comp_len;
    }

    // Decompress blocks in parallel
    let decompressed_blocks: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        let max_concurrent = num_threads.min(num_blocks);
        let mut handles: Vec<std::thread::ScopedJoinHandle<PzResult<Vec<u8>>>> =
            Vec::with_capacity(max_concurrent);
        let mut results: Vec<PzResult<Vec<u8>>> = Vec::with_capacity(num_blocks);

        for &(comp_data, orig_block_len) in &block_slices {
            if handles.len() >= max_concurrent {
                let handle = handles.remove(0);
                results.push(handle.join().unwrap_or(Err(PzError::InvalidInput)));
            }
            let opts = options.clone();
            handles.push(scope.spawn(move || {
                super::blocks::decompress_block(comp_data, pipeline, orig_block_len, &opts)
            }));
        }

        for handle in handles {
            results.push(handle.join().unwrap_or(Err(PzError::InvalidInput)));
        }

        results
    });

    // Concatenate results in order
    let mut output = Vec::with_capacity(orig_len);
    for result in decompressed_blocks {
        output.extend_from_slice(&result?);
    }

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod tests;
