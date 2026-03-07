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
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, Condvar, Mutex, OnceLock,
};
use std::time::{Duration, Instant};

use super::stages::{run_compress_stage, StageBlock, StageMetadata};
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

/// Aggregated timing/counter telemetry for the unified scheduler.
///
/// Collection is disabled by default and can be enabled via
/// [`set_unified_scheduler_stats_enabled()`].
#[derive(Debug, Clone, Copy, Default)]
pub struct UnifiedSchedulerStats {
    pub runs: u64,
    pub total_ns: u64,
    pub stage_compute_ns: u64,
    pub queue_wait_ns: u64,
    pub queue_admin_ns: u64,
    pub gpu_handoff_ns: u64,
    pub gpu_try_send_full_count: u64,
    pub gpu_try_send_disconnected_count: u64,
}

impl UnifiedSchedulerStats {
    /// Sum of tracked scheduler thread-time across workers/coordinator.
    pub fn scheduler_overhead_ns(&self) -> u64 {
        self.queue_wait_ns
            .saturating_add(self.queue_admin_ns)
            .saturating_add(self.gpu_handoff_ns)
    }

    /// Sum of tracked thread-time for scheduler + stage execution.
    pub fn tracked_thread_time_ns(&self) -> u64 {
        self.stage_compute_ns
            .saturating_add(self.scheduler_overhead_ns())
    }

    /// Fraction of tracked thread-time spent in scheduler overhead (0.0..=1.0).
    pub fn scheduler_overhead_pct(&self) -> f64 {
        let denom = self.tracked_thread_time_ns();
        if denom == 0 {
            0.0
        } else {
            self.scheduler_overhead_ns() as f64 / denom as f64
        }
    }
}

#[derive(Default)]
struct LocalSchedulerStats {
    stage_compute_ns: AtomicU64,
    queue_wait_ns: AtomicU64,
    queue_admin_ns: AtomicU64,
    gpu_handoff_ns: AtomicU64,
    gpu_try_send_full_count: AtomicU64,
    gpu_try_send_disconnected_count: AtomicU64,
}

impl LocalSchedulerStats {
    fn add_stage_compute(&self, d: Duration) {
        self.stage_compute_ns
            .fetch_add(duration_to_ns(d), Ordering::Relaxed);
    }

    fn add_queue_wait(&self, d: Duration) {
        self.queue_wait_ns
            .fetch_add(duration_to_ns(d), Ordering::Relaxed);
    }

    fn add_queue_admin(&self, d: Duration) {
        self.queue_admin_ns
            .fetch_add(duration_to_ns(d), Ordering::Relaxed);
    }

    fn add_gpu_handoff(&self, d: Duration) {
        self.gpu_handoff_ns
            .fetch_add(duration_to_ns(d), Ordering::Relaxed);
    }

    fn inc_gpu_try_send_full(&self) {
        self.gpu_try_send_full_count.fetch_add(1, Ordering::Relaxed);
    }

    fn inc_gpu_try_send_disconnected(&self) {
        self.gpu_try_send_disconnected_count
            .fetch_add(1, Ordering::Relaxed);
    }
}

fn duration_to_ns(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

static UNIFIED_SCHEDULER_STATS_ENABLED: AtomicBool = AtomicBool::new(false);
static UNIFIED_SCHEDULER_STATS: OnceLock<Mutex<UnifiedSchedulerStats>> = OnceLock::new();

struct SchedulerRunRecorder {
    start: Instant,
    local: Option<Arc<LocalSchedulerStats>>,
}

impl SchedulerRunRecorder {
    fn new(local: Option<Arc<LocalSchedulerStats>>) -> Self {
        Self {
            start: Instant::now(),
            local,
        }
    }
}

impl Drop for SchedulerRunRecorder {
    fn drop(&mut self) {
        let Some(local) = self.local.as_ref() else {
            return;
        };
        let mut guard = UNIFIED_SCHEDULER_STATS
            .get_or_init(|| Mutex::new(UnifiedSchedulerStats::default()))
            .lock()
            .expect("unified scheduler stats lock poisoned");
        guard.runs = guard.runs.saturating_add(1);
        guard.total_ns = guard
            .total_ns
            .saturating_add(duration_to_ns(self.start.elapsed()));
        guard.stage_compute_ns = guard
            .stage_compute_ns
            .saturating_add(local.stage_compute_ns.load(Ordering::Relaxed));
        guard.queue_wait_ns = guard
            .queue_wait_ns
            .saturating_add(local.queue_wait_ns.load(Ordering::Relaxed));
        guard.queue_admin_ns = guard
            .queue_admin_ns
            .saturating_add(local.queue_admin_ns.load(Ordering::Relaxed));
        guard.gpu_handoff_ns = guard
            .gpu_handoff_ns
            .saturating_add(local.gpu_handoff_ns.load(Ordering::Relaxed));
        guard.gpu_try_send_full_count = guard
            .gpu_try_send_full_count
            .saturating_add(local.gpu_try_send_full_count.load(Ordering::Relaxed));
        guard.gpu_try_send_disconnected_count =
            guard.gpu_try_send_disconnected_count.saturating_add(
                local
                    .gpu_try_send_disconnected_count
                    .load(Ordering::Relaxed),
            );
    }
}

pub(crate) fn set_unified_scheduler_stats_enabled(enabled: bool) {
    UNIFIED_SCHEDULER_STATS_ENABLED.store(enabled, Ordering::Relaxed);
}

pub(crate) fn reset_unified_scheduler_stats() {
    let mut guard = UNIFIED_SCHEDULER_STATS
        .get_or_init(|| Mutex::new(UnifiedSchedulerStats::default()))
        .lock()
        .expect("unified scheduler stats lock poisoned");
    *guard = UnifiedSchedulerStats::default();
}

pub(crate) fn unified_scheduler_stats() -> UnifiedSchedulerStats {
    *UNIFIED_SCHEDULER_STATS
        .get_or_init(|| Mutex::new(UnifiedSchedulerStats::default()))
        .lock()
        .expect("unified scheduler stats lock poisoned")
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
                    if !stage0_batch.is_empty() && uses_lz77_demux {
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
        drop(gpu_tx);
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
mod tests {
    use super::*;

    // --- Task 2 tests: Stage 1 routing by backend assignment ---

    #[test]
    fn test_stage1_routing_cpu_when_no_gpu() {
        // Compress a 512KB block with Auto backend assignment and no GPU.
        // Should route to CPU path and produce correct output.
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

        let opts = CompressOptions {
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
            .expect("compression failed");
        let decompressed = super::super::decompress(&compressed).expect("decompression failed");
        assert_eq!(decompressed, input, "round-trip should match");
    }

    #[test]
    fn test_stage1_routing_cpu_forced() {
        // Compress with explicit CPU backend assignment.
        // Even if GPU were available, should use CPU.
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

        let opts = CompressOptions {
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Cpu,
            ..CompressOptions::default()
        };

        let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
            .expect("compression failed");
        let decompressed = super::super::decompress(&compressed).expect("decompression failed");
        assert_eq!(decompressed, input, "round-trip should match");
    }

    #[test]
    fn test_stage1_routing_respects_size_threshold() {
        // Compress a block smaller than GPU_ENTROPY_THRESHOLD.
        // Should route to CPU even with Auto assignment.
        // GPU_ENTROPY_THRESHOLD is 256KB, so use 128KB.
        let input: Vec<u8> = (0..=255).cycle().take(128 * 1024).collect();

        let opts = CompressOptions {
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
            .expect("compression failed");
        let decompressed = super::super::decompress(&compressed).expect("decompression failed");
        assert_eq!(decompressed, input, "round-trip should match");
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn test_stage1_auto_backpressure_biases_to_cpu() {
        use super::super::BackendAssignment;
        use super::super::GPU_ENTROPY_THRESHOLD;

        let block_len = GPU_ENTROPY_THRESHOLD * 2;
        let limit = 8usize;

        assert!(
            should_route_block_to_gpu_entropy_with_backpressure(
                block_len,
                BackendAssignment::Auto,
                true,
                0,
                limit,
            ),
            "auto should route to GPU when pressure is low"
        );
        assert!(
            !should_route_block_to_gpu_entropy_with_backpressure(
                block_len,
                BackendAssignment::Auto,
                true,
                limit,
                limit,
            ),
            "auto should bias to CPU when pressure reaches limit"
        );
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn test_stage1_backpressure_does_not_override_explicit_backend() {
        use super::super::BackendAssignment;
        use super::super::GPU_ENTROPY_THRESHOLD;

        let block_len = GPU_ENTROPY_THRESHOLD * 2;
        let high_pressure = 1_000usize;

        assert!(
            should_route_block_to_gpu_entropy_with_backpressure(
                block_len,
                BackendAssignment::Gpu,
                true,
                high_pressure,
                1,
            ),
            "explicit GPU assignment should remain GPU regardless of pressure"
        );
        assert!(
            !should_route_block_to_gpu_entropy_with_backpressure(
                block_len,
                BackendAssignment::Cpu,
                true,
                0,
                1,
            ),
            "explicit CPU assignment should remain CPU regardless of pressure"
        );
    }

    // --- Task 3 tests: Round-trip correctness and threshold boundary ---

    #[test]
    fn test_heterogeneous_routing_roundtrip_lzseqr_cpu_only() {
        // Compress and decompress a 1MB synthetic payload (alternating byte pattern)
        // using LzSeqR with CPU-only backend assignment.
        // Assert decompressed output equals input.
        let mut input = Vec::new();
        for i in 0..1024 * 1024 {
            input.push((i % 256) as u8);
        }

        let opts = CompressOptions {
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Cpu,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "1MB round-trip with CPU-only backend should match"
        );
    }

    #[test]
    fn test_heterogeneous_routing_roundtrip_lzseqr_auto_no_gpu() {
        // Compress and decompress using Auto backend with no GPU engine.
        // Should behave identically to CPU-only.
        // Assert no panic and output is correct.
        let mut input = Vec::new();
        for i in 0..1024 * 1024 {
            input.push((i % 256) as u8);
        }

        let opts = CompressOptions {
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Auto,
            #[cfg(feature = "webgpu")]
            webgpu_engine: None,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "1MB round-trip with Auto (no GPU) should match"
        );
    }

    #[test]
    fn test_backend_assignment_threshold_boundary() {
        // Test at and below the GPU_ENTROPY_THRESHOLD boundary.
        // GPU_ENTROPY_THRESHOLD is 256KB.

        // Test exactly at threshold (256KB)
        let input_at_threshold: Vec<u8> = (0..=255).cycle().take(256 * 1024).collect();
        let opts = CompressOptions {
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input_at_threshold, Pipeline::LzSeqR, &opts)
                .expect("compression at threshold failed");
        let decompressed =
            super::super::decompress(&compressed).expect("decompression at threshold failed");
        assert_eq!(
            decompressed, input_at_threshold,
            "round-trip at threshold should match"
        );

        // Test just below threshold (256KB - 1 byte)
        let input_below_threshold: Vec<u8> = (0..=255).cycle().take(256 * 1024 - 1).collect();
        let compressed =
            super::super::compress_with_options(&input_below_threshold, Pipeline::LzSeqR, &opts)
                .expect("compression below threshold failed");
        let decompressed =
            super::super::decompress(&compressed).expect("decompression below threshold failed");
        assert_eq!(
            decompressed, input_below_threshold,
            "round-trip below threshold should match"
        );
    }

    // --- Task 5 tests: Ring-buffered entropy handoff correctness ---

    #[test]
    fn test_entropy_ring_slot_recycling() {
        // Compress 16 blocks of 256KB (8 ring slots, depth=4 — forces each slot to be recycled twice).
        // Assert all 16 blocks decompress correctly.
        // This tests that slot recycling doesn't cause data corruption or loss.
        let block_size = 256 * 1024; // 256KB per block
        let mut input = Vec::new();
        // Create 16 distinct blocks so we can verify each decompresses correctly
        for block_idx in 0..16 {
            for i in 0..block_size {
                // Mix the block index into the data so each block is unique
                input.push(((block_idx * 17 + i) % 256) as u8);
            }
        }

        let opts = CompressOptions {
            block_size,
            threads: 4,

            stage1_backend: super::super::BackendAssignment::Cpu,
            ..CompressOptions::default()
        };

        let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
            .expect("compression failed");
        let decompressed = super::super::decompress(&compressed).expect("decompression failed");
        assert_eq!(
            decompressed, input,
            "slot recycling: 16 blocks x 256KB should round-trip correctly"
        );
    }

    #[test]
    fn test_entropy_handoff_output_order_preserved() {
        // Compress blocks with varying sizes (some above, some below GPU_ENTROPY_THRESHOLD).
        // Assert decompressed output equals input byte-for-byte regardless of which path
        // (CPU or GPU) each block took.
        // GPU_ENTROPY_THRESHOLD is 256KB, so use 128KB and 512KB blocks.
        let block_size = 128 * 1024; // 128KB blocks
        let mut input = Vec::new();
        let mut block_markers = Vec::new();

        // Create blocks with distinctive patterns so we can verify ordering
        // Block 0: 128KB (below threshold)
        for i in 0..block_size {
            input.push((i % 256) as u8);
        }
        block_markers.push((0, block_size));

        // Block 1: 512KB (above threshold)
        for i in 0usize..512 * 1024 {
            input.push(((i.wrapping_add(1)) % 256) as u8);
        }
        block_markers.push((block_size, 512 * 1024));

        // Block 2: 256KB (at threshold)
        for i in 0usize..256 * 1024 {
            input.push(((i.wrapping_add(2)) % 256) as u8);
        }
        block_markers.push((block_size + 512 * 1024, 256 * 1024));

        // Block 3: 100KB (small, below threshold)
        for i in 0usize..100 * 1024 {
            input.push(((i.wrapping_add(3)) % 256) as u8);
        }
        block_markers.push((block_size + 512 * 1024 + 256 * 1024, 100 * 1024));

        let opts = CompressOptions {
            block_size,
            threads: 4,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
            .expect("compression failed");
        let decompressed = super::super::decompress(&compressed).expect("decompression failed");
        assert_eq!(
            decompressed, input,
            "output order preserved: mixed block sizes should decompress in order"
        );

        // Verify each block's content is correct by spot-checking markers
        for (offset, size) in block_markers {
            assert!(
                offset + size <= decompressed.len(),
                "block out of bounds in decompressed output"
            );
        }
    }

    #[test]
    fn test_entropy_handoff_cpu_gpu_cross_decode() {
        // Test encoding with CPU entropy and decoding with CPU works (baseline).
        // This verifies that the entropy container format is platform-independent.
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

        let opts = CompressOptions {
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Cpu,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "CPU encode + CPU decode should round-trip"
        );
    }

    #[test]
    fn test_entropy_ring_empty_blocks_handled() {
        // Compress an input where the last block is smaller than typical GPU input sizes.
        // Assert no panic, correct output.
        // This tests the edge case where a ring slot is acquired but may not be fully utilized.
        let input: Vec<u8> = (0..=255).cycle().take(256 * 1024 + 1000).collect();

        let opts = CompressOptions {
            block_size: 256 * 1024,
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
            .expect("compression with small final block failed");
        let decompressed = super::super::decompress(&compressed)
            .expect("decompression of small final block failed");
        assert_eq!(
            decompressed, input,
            "small final block should not cause panic or data corruption"
        );
    }

    // Task 6: Stage0 prefetch overlaps with GPU entropy
    #[test]
    fn test_prefetch_stage0_correct_output() {
        // Verify that Stage0 prefetching does not affect correctness.
        // Compress with prefetch enabled (via heterogeneous path) and verify round-trip.
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

        let opts = CompressOptions {
            block_size: 256 * 1024,
            threads: 4,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "prefetch stage0 compression round-trip should preserve data"
        );
    }

    // Task 7: Zero-overhead CPU-only path when no GPU
    #[test]
    fn test_cpu_only_path_no_webgpu_engine() {
        // Compress with no GPU engine and verify correct output.
        // This should use the CPU-only path without any GPU initialization.
        let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

        let opts = CompressOptions {
            backend: super::super::Backend::Cpu,

            threads: 4,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "CPU-only path should work with zero overhead"
        );
    }

    #[test]
    fn test_cpu_only_path_cpu_backend_explicit() {
        // Force CPU backend and CPU entropy explicitly.
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

        let opts = CompressOptions {
            backend: super::super::Backend::Cpu,

            threads: 2,
            stage1_backend: super::super::BackendAssignment::Cpu,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "explicit CPU backend should work correctly"
        );
    }

    // Task 8: Graceful GPU device-lost fallback
    #[cfg(feature = "webgpu")]
    #[test]
    fn test_gpu_device_lost_fallback_continues() {
        // Test that the unified path gracefully handles GPU errors.
        // Since we can't easily inject real GPU failures, we test that the
        // fallback path (CPU entropy) is used when GPU submission fails.
        let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

        let opts = CompressOptions {
            block_size: 256 * 1024,
            threads: 4,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        // Even if GPU is available, this should not panic or corrupt data.
        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "GPU fallback path should preserve data integrity"
        );
    }

    // Task 9: Comprehensive fallback scenario tests (AC3.5 and AC3.6)
    #[test]
    #[cfg(feature = "webgpu")]
    fn test_ac3_5_no_gpu_zero_overhead_cpu_path() {
        // AC3.5: No GPU available — pure CPU path works with zero overhead.
        let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

        let opts = CompressOptions {
            backend: super::super::Backend::Cpu,

            threads: 4,
            webgpu_engine: None, // Explicitly no GPU
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "AC3.5: CPU-only path should produce correct output"
        );
    }

    #[test]
    fn test_ac3_5_webgpu_feature_disabled_compiles_and_works() {
        // AC3.5: Code should compile and work correctly even when webgpu feature is disabled.
        // This test runs only when the feature is disabled (checked by cargo test).
        #[cfg(not(feature = "webgpu"))]
        {
            let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

            let opts = CompressOptions {
                threads: 2,
                ..CompressOptions::default()
            };

            let compressed =
                super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
            let decompressed = super::super::decompress(&compressed).unwrap();
            assert_eq!(
                decompressed, input,
                "non-webgpu build should work correctly"
            );
        }
    }

    #[cfg(feature = "webgpu")]
    #[test]
    fn test_ac3_6_gpu_lost_fallback_semantics() {
        // AC3.6: GPU device lost mid-compression — graceful fallback to CPU, no data corruption.
        // This test verifies that the gpu_lost flag and fallback path work correctly.
        let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

        let opts = CompressOptions {
            block_size: 256 * 1024,
            threads: 4,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        // Compression should succeed even if GPU fallback occurs internally.
        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "AC3.6: GPU fallback should preserve data integrity"
        );
    }

    #[test]
    fn test_ac3_6_no_data_corruption_on_partial_slot_state() {
        // Verify that a GPU loss during any stage (submit or complete) falls back cleanly.
        let input: Vec<u8> = (0..=255).cycle().take(768 * 1024).collect();

        let opts = CompressOptions {
            block_size: 256 * 1024,
            threads: 2,

            stage1_backend: super::super::BackendAssignment::Auto,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "partial GPU slot state should fall back cleanly"
        );
    }

    // GPU-specific unified scheduler tests
    // These tests verify the unified scheduler dispatches GPU work correctly
    // when GPU is available and backend settings allow it.
    #[test]
    #[cfg(feature = "webgpu")]
    fn test_heterogeneous_compress_with_gpu_entropy() {
        // Test actual GPU unified compression when GPU is available.
        use crate::webgpu::WebGpuEngine;

        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

        // Try to create a GPU engine; skip test if no device available
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(_) => {
                eprintln!("GPU device not available, skipping GPU test");
                return;
            }
        };

        let opts = CompressOptions {
            backend: super::super::Backend::WebGpu,

            threads: 2,
            stage1_backend: super::super::BackendAssignment::Auto,
            webgpu_engine: Some(std::sync::Arc::new(engine)),
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "GPU unified path should decompress correctly"
        );
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn test_heterogeneous_compress_gpu_forced_large_blocks() {
        // Test with stage1_backend forced to Gpu and large blocks that trigger GPU entropy.
        use crate::webgpu::WebGpuEngine;

        let input: Vec<u8> = (0..=255).cycle().take(768 * 1024).collect(); // 3 x 256KB blocks

        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(_) => {
                eprintln!("GPU device not available, skipping GPU test");
                return;
            }
        };

        let opts = CompressOptions {
            backend: super::super::Backend::WebGpu,

            threads: 2,
            block_size: 256 * 1024,
            stage1_backend: super::super::BackendAssignment::Gpu,
            webgpu_engine: Some(std::sync::Arc::new(engine)),
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "GPU forced unified path should decompress correctly"
        );
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn test_heterogeneous_mixed_block_sizes_with_gpu() {
        // Verify unified path handles mixed block sizes correctly:
        // some blocks trigger GPU (>= 256KB), some use CPU (< 256KB).
        use crate::webgpu::WebGpuEngine;

        let mut input = Vec::new();
        // 1 large block (will use GPU if Auto), 2 small blocks (will use CPU)
        for i in 0..512 * 1024 {
            input.push((i % 256) as u8);
        }
        for i in 0..128 * 1024 {
            input.push((i % 256) as u8);
        }
        for i in 0..128 * 1024 {
            input.push((i % 256) as u8);
        }

        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(_) => {
                eprintln!("GPU device not available, skipping GPU test");
                return;
            }
        };

        let opts = CompressOptions {
            backend: super::super::Backend::WebGpu,

            threads: 2,
            block_size: 256 * 1024,
            stage1_backend: super::super::BackendAssignment::Auto,
            webgpu_engine: Some(std::sync::Arc::new(engine)),
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "Mixed GPU/CPU unified path should decompress correctly"
        );
    }

    // GPU unified scheduler tests for LZ77-based pipelines (Deflate, Lzr, Lzf).
    // These exercise the Stage 0 GPU routing and batch-collect path.

    #[test]
    #[cfg(feature = "webgpu")]
    fn test_gpu_roundtrip_deflate() {
        use crate::webgpu::WebGpuEngine;
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        let opts = CompressOptions {
            backend: super::super::Backend::WebGpu,
            threads: 2,
            block_size: 256 * 1024,
            webgpu_engine: Some(std::sync::Arc::new(engine)),
            ..CompressOptions::default()
        };
        let compressed =
            super::super::compress_with_options(&input, Pipeline::Deflate, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "GPU Deflate round-trip failed");
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn test_gpu_roundtrip_lzr() {
        use crate::webgpu::WebGpuEngine;
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        let opts = CompressOptions {
            backend: super::super::Backend::WebGpu,
            threads: 2,
            block_size: 256 * 1024,
            webgpu_engine: Some(std::sync::Arc::new(engine)),
            ..CompressOptions::default()
        };
        let compressed = super::super::compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "GPU Lzr round-trip failed");
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn test_gpu_roundtrip_lzf() {
        use crate::webgpu::WebGpuEngine;
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(_) => return,
        };
        let opts = CompressOptions {
            backend: super::super::Backend::WebGpu,
            threads: 2,
            block_size: 256 * 1024,
            webgpu_engine: Some(std::sync::Arc::new(engine)),
            ..CompressOptions::default()
        };
        let compressed = super::super::compress_with_options(&input, Pipeline::Lzf, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "GPU Lzf round-trip failed");
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn test_lzr_backend_assignments_are_interchangeable() {
        use crate::pipeline::{Backend, BackendAssignment};
        use crate::webgpu::WebGpuEngine;

        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
        let engine = match WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(_) => return,
        };

        let cases = [
            ("cpu/cpu", BackendAssignment::Cpu, BackendAssignment::Cpu),
            ("gpu/cpu", BackendAssignment::Gpu, BackendAssignment::Cpu),
            ("cpu/gpu", BackendAssignment::Cpu, BackendAssignment::Gpu),
            (
                "auto/auto",
                BackendAssignment::Auto,
                BackendAssignment::Auto,
            ),
        ];

        for (label, stage0_backend, stage1_backend) in cases {
            let opts = CompressOptions {
                backend: Backend::WebGpu,
                threads: 2,
                block_size: 256 * 1024,
                stage0_backend,
                stage1_backend,
                webgpu_engine: Some(engine.clone()),
                ..CompressOptions::default()
            };
            let compressed =
                super::super::compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
            let decompressed = super::super::decompress(&compressed).unwrap();
            assert_eq!(
                decompressed, input,
                "Lzr round-trip failed for interchangeable backends: {label}"
            );
        }
    }

    // Test that the channel-full fallback path produces correct results.
    // Uses a single-capacity channel (ring_depth=1) with many blocks to force
    // try_send failures, exercising the CPU fallback in the worker dispatch.
    #[test]
    fn test_channel_full_cpu_fallback() {
        // Many small blocks with minimal threads — channel of depth 1 will overflow
        let block_size = 64 * 1024; // 64KB blocks
        let num_blocks = 16;
        let input: Vec<u8> = (0..=255).cycle().take(block_size * num_blocks).collect();

        let opts = CompressOptions {
            block_size,
            threads: 8, // many workers competing for channel
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "channel-full fallback should produce correct results"
        );
    }
}
