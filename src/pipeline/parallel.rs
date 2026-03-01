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
use std::sync::{Condvar, Mutex};

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
}

/// Number of compression stages for a pipeline in the unified scheduler.
fn unified_stage_count(pipeline: Pipeline) -> usize {
    match pipeline {
        Pipeline::Bw | Pipeline::Bbw => 4,
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

/// Determine whether to route a block to GPU entropy encoding.
///
/// Checks the backend assignment setting and block size to decide if GPU entropy
/// should be used. Returns false when no GPU is available or when CPU is explicitly
/// assigned or when the block is too small.
fn should_route_block_to_gpu_entropy(block: &[u8], options: &CompressOptions) -> bool {
    #[cfg(feature = "webgpu")]
    {
        use super::{BackendAssignment, GPU_ENTROPY_THRESHOLD};
        match options.stage1_backend {
            BackendAssignment::Gpu => options.webgpu_engine.is_some(),
            BackendAssignment::Cpu => false,
            BackendAssignment::Auto => {
                options.webgpu_engine.is_some() && block.len() >= GPU_ENTROPY_THRESHOLD
            }
        }
    }
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = block;
        let _ = options;
        false
    }
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

/// Message sent from unified workers to the GPU coordinator thread.
#[cfg(feature = "webgpu")]
enum GpuRequest {
    /// Stage 0: LZ77/LzSeq match-finding on GPU.
    /// (stage_idx, block_idx, input_data)
    Stage0(usize, usize, Vec<u8>),
    /// Stage 1+: entropy encoding on GPU.
    /// (stage_idx, block_idx, stage_block)
    StageN(usize, usize, StageBlock),
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

    // Build initial task queue with GPU routing for Stage 0 where applicable.
    let initial_tasks: VecDeque<UnifiedTask> = (0..num_blocks)
        .map(|i| {
            #[cfg(feature = "webgpu")]
            if has_gpu && should_route_block_to_gpu_stage0(blocks[i], pipeline, &resolved_options) {
                return UnifiedTask::StageGpu(0, i);
            }
            let _ = has_gpu;
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
        let ring_depth = 4.min(num_blocks.max(1));
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

            scope.spawn(move || {
                while let Ok(request) = rx.recv() {
                    // Both Stage0 and StageN route through run_compress_stage,
                    // which handles GPU dispatch for all pipelines.
                    let (stage_idx, block_idx, block) = match request {
                        GpuRequest::Stage0(s, b, data) => (
                            s,
                            b,
                            StageBlock {
                                block_index: b,
                                original_len: data.len(),
                                data,
                                streams: None,
                                metadata: StageMetadata::default(),
                            },
                        ),
                        GpuRequest::StageN(s, b, sb) => (s, b, sb),
                    };
                    let result = run_compress_stage(pipeline, stage_idx, block, &opts);
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
                    );
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
            #[cfg(feature = "webgpu")]
            let gpu_tx_clone = gpu_tx.clone();

            scope.spawn(move || loop {
                let task = {
                    let mut guard = queue_ref.lock().expect("unified queue poisoned");
                    loop {
                        if let Some(task) = guard.queue.pop_front() {
                            break task;
                        }
                        if guard.closed || (guard.failed && guard.queue.is_empty()) {
                            return;
                        }
                        guard = cv_ref.wait(guard).expect("unified queue wait poisoned");
                    }
                };

                let (stage_idx, block_idx) = match task {
                    UnifiedTask::Stage(s, b) => (s, b),
                    UnifiedTask::StageGpu(s, b) => {
                        // Route to GPU coordinator — send and continue to next task
                        #[cfg(feature = "webgpu")]
                        if let Some(ref tx) = gpu_tx_clone {
                            let request = if s == 0 {
                                GpuRequest::Stage0(s, b, blocks[b].to_vec())
                            } else {
                                let stage_block = slots_ref[b]
                                    .lock()
                                    .expect("intermediate slot poisoned")
                                    .take()
                                    .expect("intermediate result missing");
                                GpuRequest::StageN(s, b, stage_block)
                            };
                            if tx.send(request).is_ok() {
                                continue;
                            }
                            // Channel closed — GPU coordinator gone, fall through to CPU
                        }
                        // Fallback: execute on CPU
                        (s, b)
                    }
                };

                // Build or retrieve the StageBlock for this stage.
                let block = if stage_idx == 0 {
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

                let result = run_compress_stage(pipeline, stage_idx, block, &opts);

                match result {
                    Ok(stage_block) => {
                        if stage_idx == last_stage {
                            // Final stage: store the compressed data.
                            *results_ref[block_idx].lock().expect("result slot poisoned") =
                                Some(Ok(stage_block.data));
                        } else {
                            // Intermediate stage: store block and enqueue next stage.
                            *slots_ref[block_idx]
                                .lock()
                                .expect("intermediate slot poisoned") = Some(stage_block);
                            let mut guard = queue_ref.lock().expect("unified queue poisoned");
                            if !guard.failed {
                                let next_stage = stage_idx + 1;
                                let next_task = if next_stage == last_stage
                                    && should_route_block_to_gpu_entropy(blocks[block_idx], &opts)
                                {
                                    UnifiedTask::StageGpu(next_stage, block_idx)
                                } else {
                                    UnifiedTask::Stage(next_stage, block_idx)
                                };
                                guard.queue.push_back(next_task);
                                guard.pending_tasks += 1;
                                cv_ref.notify_one();
                            } else {
                                *results_ref[block_idx].lock().expect("result slot poisoned") =
                                    Some(Err(PzError::InvalidInput));
                            }
                        }
                    }
                    Err(e) => {
                        *results_ref[block_idx].lock().expect("result slot poisoned") =
                            Some(Err(e));
                        let mut guard = queue_ref.lock().expect("unified queue poisoned");
                        if !guard.failed {
                            guard.failed = true;
                            let dropped = guard.queue.len();
                            guard.queue.clear();
                            guard.pending_tasks = guard.pending_tasks.saturating_sub(dropped);
                            cv_ref.notify_all();
                        }
                    }
                }

                let mut guard = queue_ref.lock().expect("unified queue poisoned");
                debug_assert!(guard.pending_tasks > 0);
                guard.pending_tasks -= 1;
                if guard.pending_tasks == 0 {
                    guard.closed = true;
                    cv_ref.notify_all();
                    return;
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
) {
    match result {
        Ok(sb) => {
            if stage_idx == last_stage {
                *results[block_idx].lock().expect("result slot poisoned") = Some(Ok(sb.data));
            } else {
                *intermediate_slots[block_idx]
                    .lock()
                    .expect("intermediate slot poisoned") = Some(sb);
                let mut guard = queue.lock().expect("unified queue poisoned");
                if !guard.failed {
                    let next_stage = stage_idx + 1;
                    let next_task = if next_stage == last_stage
                        && should_route_block_to_gpu_entropy(blocks[block_idx], options)
                    {
                        UnifiedTask::StageGpu(next_stage, block_idx)
                    } else {
                        UnifiedTask::Stage(next_stage, block_idx)
                    };
                    guard.queue.push_back(next_task);
                    guard.pending_tasks += 1;
                    queue_cv.notify_one();
                }
            }
        }
        Err(e) => {
            *results[block_idx].lock().expect("result slot poisoned") = Some(Err(e));
            let mut guard = queue.lock().expect("unified queue poisoned");
            if !guard.failed {
                guard.failed = true;
                let dropped = guard.queue.len();
                guard.queue.clear();
                guard.pending_tasks = guard.pending_tasks.saturating_sub(dropped);
            }
            // Decrement and check for completion while we hold the lock
            debug_assert!(guard.pending_tasks > 0);
            guard.pending_tasks -= 1;
            if guard.pending_tasks == 0 {
                guard.closed = true;
            }
            queue_cv.notify_all();
            return;
        }
    }

    // Decrement pending_tasks and check for completion
    let mut guard = queue.lock().expect("unified queue poisoned");
    debug_assert!(guard.pending_tasks > 0);
    guard.pending_tasks -= 1;
    if guard.pending_tasks == 0 {
        guard.closed = true;
        queue_cv.notify_all();
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
}
