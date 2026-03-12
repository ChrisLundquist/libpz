//! Multi-block parallel compression and decompression.
//!
//! **Unified scheduler**: a single shared work queue executes all pipeline stages from a
//! CPU worker pool. Workers process blocks through all stages using local continuation,
//! only round-tripping through the queue when stealing work from other blocks.
//!
//! GPU-accelerated compression is handled by the **streaming** path
//! (`streaming::compress_stream`), which has a dedicated GPU coordinator thread with
//! adaptive backpressure. The in-memory parallel path is CPU-only by design — it
//! achieves higher throughput (6+ GiB/s) by avoiding GPU dispatch overhead.

use crate::{PzError, PzResult};
use std::collections::VecDeque;
use std::sync::{atomic::Ordering, Arc, Condvar, Mutex};
use std::time::Instant;

use super::stages::{run_compress_stage, StageBlock, StageMetadata};
use super::telemetry::{
    LocalSchedulerStats, SchedulerRunRecorder, UNIFIED_SCHEDULER_STATS_ENABLED,
};
use super::{write_header, CompressOptions, DecompressOptions, Pipeline, BLOCK_HEADER_SIZE};

/// Multi-block parallel compression.
///
/// All pipelines flow through the unified scheduler with CPU-only workers.
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

/// A task in the unified work queue: `(stage_idx, block_idx)`.
type UnifiedTask = (usize, usize);

/// Number of compression stages for a pipeline in the unified scheduler.
fn unified_stage_count(pipeline: Pipeline) -> usize {
    match pipeline {
        Pipeline::Bw | Pipeline::Bbw => 4,
        Pipeline::SortLz => 1,
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

/// Retire one task from the unified queue after a block's final stage completes.
///
/// Returns `(mark_invalid_after_lock, should_return_worker)`.
fn complete_task_lifecycle(
    guard: &mut UnifiedQueueState,
    queue_cv: &Condvar,
    stage_failed: bool,
) -> (bool, bool) {
    let mut mark_invalid_after_lock = false;

    if stage_failed && !guard.failed {
        guard.failed = true;
        let dropped = guard.queue.len();
        guard.queue.clear();
        guard.pending_tasks = guard.pending_tasks.saturating_sub(dropped);
        queue_cv.notify_all();
    }

    if guard.failed && !stage_failed {
        // Block completed successfully but scheduler already failed elsewhere.
        mark_invalid_after_lock = true;
    }

    debug_assert!(guard.pending_tasks > 0);
    guard.pending_tasks -= 1;

    let should_return = if guard.pending_tasks == 0 {
        guard.closed = true;
        queue_cv.notify_all();
        true
    } else {
        false
    };

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

    // Safety net: compress_with_options routes GPU requests through
    // compress_stream before reaching here, but force CPU as defense-in-depth.
    #[cfg(feature = "webgpu")]
    {
        resolved_options.backend = super::Backend::Cpu;
        resolved_options.webgpu_engine = None;
    }

    // Per-block intermediate slot: holds the StageBlock between stages.
    // A block is only in one stage at a time, so one slot per block suffices.
    let intermediate_slots: Vec<Mutex<Option<StageBlock>>> =
        (0..num_blocks).map(|_| Mutex::new(None)).collect();
    let results: Vec<Mutex<Option<PzResult<Vec<u8>>>>> =
        (0..num_blocks).map(|_| Mutex::new(None)).collect();

    // All blocks start at Stage 0 on CPU.
    let initial_tasks: VecDeque<UnifiedTask> = (0..num_blocks).map(|i| (0, i)).collect();

    let queue = Mutex::new(UnifiedQueueState {
        queue: initial_tasks,
        pending_tasks: num_blocks,
        closed: false,
        failed: false,
    });
    let queue_cv = Condvar::new();

    std::thread::scope(|scope| {
        // CPU worker threads.
        for _ in 0..worker_count {
            let queue_ref = &queue;
            let cv_ref = &queue_cv;
            let slots_ref = &intermediate_slots;
            let results_ref = &results;
            let opts = resolved_options.clone();
            let stats_ref = stats_local.clone();

            scope.spawn(move || {
                // Local continuation state for this worker.
                // When set, process the next stage directly instead of
                // round-tripping through the shared queue.
                let mut local_task: Option<(usize, usize, StageBlock)> = None;

                loop {
                    let (stage_idx, block_idx, inline_block) = if let Some((s, b, sb)) =
                        local_task.take()
                    {
                        (s, b, Some(sb))
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

                        let (s, b) = task;
                        (s, b, None)
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
                                // CPU continuation: keep processing the same block locally.
                                let next_stage = stage_idx + 1;
                                local_task = Some((next_stage, block_idx, stage_block));
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
                        complete_task_lifecycle(&mut guard, cv_ref, stage_failed);
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
