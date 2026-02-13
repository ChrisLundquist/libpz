//! Multi-block parallel compression and decompression.
//!
//! Two strategies:
//! - **Block-parallel**: one thread per block, each running all stages (default).
//! - **Pipeline-parallel**: one thread per stage, blocks flow through channels.

use crate::{PzError, PzResult};

use super::blocks::compress_block;
use super::stages::{
    pipeline_stage_count, run_compress_stage, run_decompress_stage, StageBlock, StageMetadata,
};
use super::{write_header, CompressOptions, DecompressOptions, Pipeline, BLOCK_HEADER_SIZE};

/// Multi-block parallel compression.
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
    // Try streaming or batched GPU LZ77 path for LZ77-based pipelines with WebGPU backend
    #[cfg(feature = "webgpu")]
    {
        if let super::Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                let is_lz77_pipeline = matches!(
                    pipeline,
                    Pipeline::Deflate | Pipeline::Lzr | Pipeline::Lzf | Pipeline::Lzfi
                );
                let is_batchable =
                    is_lz77_pipeline && options.parse_strategy != super::ParseStrategy::Optimal;
                if is_batchable {
                    // Multi-block: use double-buffered streaming for GPU/CPU overlap
                    let block_size = options.block_size;
                    if input.len() > block_size {
                        if let Some(ring) = engine.create_lz77_ring(block_size) {
                            return compress_streaming_gpu(
                                input,
                                pipeline,
                                options,
                                num_threads,
                                engine,
                                ring,
                            );
                        }
                    }
                    // Single-block or insufficient GPU memory: use existing batched path
                    return compress_parallel_gpu_batched(
                        input,
                        pipeline,
                        options,
                        num_threads,
                        engine,
                    );
                }
            }
        }
    }

    let block_size = options.block_size;

    // Split input into blocks
    let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
    let num_blocks = blocks.len();

    // Compress blocks in parallel using scoped threads
    let compressed_blocks: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        // Launch threads in batches to cap concurrency
        let max_concurrent = num_threads.min(num_blocks);
        let mut handles: Vec<std::thread::ScopedJoinHandle<PzResult<Vec<u8>>>> =
            Vec::with_capacity(max_concurrent);
        let mut results: Vec<PzResult<Vec<u8>>> = Vec::with_capacity(num_blocks);

        for block in &blocks {
            if handles.len() >= max_concurrent {
                // Wait for the earliest thread to finish
                let handle = handles.remove(0);
                results.push(handle.join().unwrap_or(Err(PzError::InvalidInput)));
            }
            let opts = options.clone();
            handles.push(scope.spawn(move || compress_block(block, pipeline, &opts)));
        }

        // Collect remaining results
        for handle in handles {
            results.push(handle.join().unwrap_or(Err(PzError::InvalidInput)));
        }

        results
    });

    // Check for errors
    let mut block_data_vec: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for result in compressed_blocks {
        block_data_vec.push(result?);
    }

    // Build output: V2 header + num_blocks + block_table + block_data
    let mut output = Vec::new();
    write_header(&mut output, pipeline, input.len());

    // num_blocks
    output.extend_from_slice(&(num_blocks as u32).to_le_bytes());

    // Block table
    for (i, compressed) in block_data_vec.iter().enumerate() {
        let orig_block_len = blocks[i].len() as u32;
        let comp_block_len = compressed.len() as u32;
        output.extend_from_slice(&comp_block_len.to_le_bytes());
        output.extend_from_slice(&orig_block_len.to_le_bytes());
    }

    // Block data
    for compressed in &block_data_vec {
        output.extend_from_slice(compressed);
    }

    Ok(output)
}

/// Batched GPU LZ77 compression for multi-block inputs.
///
/// Pipelines GPU LZ77 matching and CPU entropy encoding: while CPU threads
/// entropy-encode the current batch, the GPU is already processing the next
/// batch. This overlaps GPU and CPU work, improving throughput on large inputs.
///
/// This is used for LZ77-based pipelines (Deflate, Lzr, Lzf, Lzfi) when the
/// WebGPU backend is active and there are multiple blocks.
#[cfg(feature = "webgpu")]
fn compress_parallel_gpu_batched(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
    num_threads: usize,
    engine: &crate::webgpu::WebGpuEngine,
) -> PzResult<Vec<u8>> {
    let block_size = options.block_size;
    let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
    let num_blocks = blocks.len();

    // Pipeline GPU and CPU work: submit GPU batch N+1 while CPU processes batch N.
    //
    // For small inputs (≤1 batch), this degrades to the simple submit→wait→encode
    // path with no overhead.
    let gpu_batch_size = engine.lz77_batch_size(block_size);
    let block_chunks: Vec<&[&[u8]]> = blocks.chunks(gpu_batch_size).collect();

    let compressed_blocks: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        let max_concurrent = num_threads.min(num_blocks);
        let mut handles: Vec<(usize, std::thread::ScopedJoinHandle<PzResult<Vec<u8>>>)> =
            Vec::with_capacity(max_concurrent);
        let mut results: Vec<Option<PzResult<Vec<u8>>>> = (0..num_blocks).map(|_| None).collect();

        let mut global_block_idx = 0usize;

        for chunk in &block_chunks {
            // GPU: batch LZ77 for this chunk
            let match_vecs = match engine.find_matches_batched(chunk) {
                Ok(v) => v,
                Err(e) => {
                    // Propagate error for all blocks in this chunk
                    for local_i in 0..chunk.len() {
                        let block_idx = global_block_idx + local_i;
                        results[block_idx] = Some(Err(e.clone()));
                    }
                    global_block_idx += chunk.len();
                    continue;
                }
            };

            // CPU: entropy-encode each block (while GPU buffers are freed for next batch)
            for (local_i, matches) in match_vecs.into_iter().enumerate() {
                let block_idx = global_block_idx + local_i;

                // Drain finished handles to maintain concurrency limit
                while handles.len() >= max_concurrent {
                    let (idx, handle) = handles.remove(0);
                    results[idx] = Some(handle.join().unwrap_or(Err(PzError::InvalidInput)));
                }

                let block_input = blocks[block_idx];
                handles.push((
                    block_idx,
                    scope.spawn(move || {
                        entropy_encode_lz77_block(&matches, block_input.len(), pipeline)
                    }),
                ));
            }

            global_block_idx += chunk.len();
        }

        // Collect remaining handles
        for (idx, handle) in handles {
            results[idx] = Some(handle.join().unwrap_or(Err(PzError::InvalidInput)));
        }

        results.into_iter().map(|r| r.unwrap()).collect::<Vec<_>>()
    });

    // Check for errors and collect
    let mut block_data_vec: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for result in compressed_blocks {
        block_data_vec.push(result?);
    }

    // Build output: V2 header + num_blocks + block_table + block_data
    let mut output = Vec::new();
    write_header(&mut output, pipeline, input.len());
    output.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    for (i, compressed) in block_data_vec.iter().enumerate() {
        let orig_block_len = blocks[i].len() as u32;
        let comp_block_len = compressed.len() as u32;
        output.extend_from_slice(&comp_block_len.to_le_bytes());
        output.extend_from_slice(&orig_block_len.to_le_bytes());
    }
    for compressed in &block_data_vec {
        output.extend_from_slice(compressed);
    }

    Ok(output)
}

/// Double-buffered streaming GPU compression for multi-block inputs.
///
/// Overlaps GPU LZ77 matching with CPU entropy encoding using a ring of
/// pre-allocated GPU buffer slots. While the GPU processes block N+1,
/// the CPU entropy-encodes block N's results. With triple buffering,
/// upload/compute/download can all overlap on different slots.
///
/// # Threading model
///
/// - **GPU coordinator** (one thread): round-robins through buffer slots,
///   submits work and reads back completed results.
/// - **CPU workers** (N-1 threads): receive matches from channels, run
///   demux + entropy encoding.
/// - **Collector** (calling thread): gathers indexed results in order.
#[cfg(feature = "webgpu")]
fn compress_streaming_gpu(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
    num_threads: usize,
    engine: &crate::webgpu::WebGpuEngine,
    mut ring: crate::webgpu::lz77::BufferRing<crate::webgpu::lz77::Lz77BufferSlot>,
) -> PzResult<Vec<u8>> {
    let block_size = options.block_size;
    let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
    let num_blocks = blocks.len();
    let ring_depth = ring.depth();
    let max_dispatch = engine.max_dispatch_input_size();

    // Per-worker channels: GPU coordinator distributes work round-robin
    let cpu_workers = (num_threads.saturating_sub(1)).max(1);

    let compressed_blocks: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        use std::sync::mpsc;

        // Per-worker input channels: (block_index, matches, original_len)
        let mut worker_txs: Vec<mpsc::SyncSender<(usize, Vec<crate::lz77::Match>, usize)>> =
            Vec::with_capacity(cpu_workers);
        // Single output channel from all workers: (block_index, result)
        let (result_tx, result_rx) = mpsc::sync_channel::<(usize, PzResult<Vec<u8>>)>(num_blocks);

        // Spawn CPU worker threads
        for _ in 0..cpu_workers {
            let (tx, rx) =
                mpsc::sync_channel::<(usize, Vec<crate::lz77::Match>, usize)>(ring_depth);
            worker_txs.push(tx);
            let rtx = result_tx.clone();
            scope.spawn(move || {
                while let Ok((block_idx, matches, original_len)) = rx.recv() {
                    let result = entropy_encode_lz77_block(&matches, original_len, pipeline);
                    let _ = rtx.send((block_idx, result));
                }
            });
        }
        drop(result_tx); // Workers hold their own clones

        // GPU coordinator thread — owns worker_txs so it can signal
        // completion by dropping all senders when done.
        scope.spawn(move || {
            let worker_txs = worker_txs; // move into closure
            let mut slot_inflight: Vec<Option<(usize, Vec<u8>)>> = vec![None; ring_depth];
            let mut next_worker = 0usize;

            for (block_idx, block) in blocks.iter().enumerate() {
                let slot_idx = ring.acquire();

                // If this slot has previous in-flight work, complete it first
                if let Some((prev_idx, prev_input)) = slot_inflight[slot_idx].take() {
                    engine.poll_wait();
                    let matches = engine
                        .complete_lz77_from_slot(&ring.slots[slot_idx], &prev_input)
                        .unwrap_or_default();
                    let _ = worker_txs[next_worker % cpu_workers].send((
                        prev_idx,
                        matches,
                        prev_input.len(),
                    ));
                    next_worker += 1;
                }

                // Submit new block to this slot (or CPU fallback for edge cases)
                if block.is_empty()
                    || block.len() < crate::webgpu::MIN_GPU_INPUT_SIZE
                    || block.len() > max_dispatch
                {
                    // CPU fallback: compute matches on CPU, send directly
                    let matches = crate::lz77::compress_lazy_to_matches(block).unwrap_or_default();
                    let _ = worker_txs[next_worker % cpu_workers].send((
                        block_idx,
                        matches,
                        block.len(),
                    ));
                    next_worker += 1;
                } else if let Err(_e) = engine.submit_lz77_to_slot(block, &ring.slots[slot_idx]) {
                    // GPU submission failed — fall back to CPU
                    let matches = crate::lz77::compress_lazy_to_matches(block).unwrap_or_default();
                    let _ = worker_txs[next_worker % cpu_workers].send((
                        block_idx,
                        matches,
                        block.len(),
                    ));
                    next_worker += 1;
                } else {
                    slot_inflight[slot_idx] = Some((block_idx, block.to_vec()));
                }
            }

            // Drain remaining in-flight slots
            for slot_idx in 0..ring_depth {
                if let Some((prev_idx, prev_input)) = slot_inflight[slot_idx].take() {
                    engine.poll_wait();
                    let matches = engine
                        .complete_lz77_from_slot(&ring.slots[slot_idx], &prev_input)
                        .unwrap_or_default();
                    let _ = worker_txs[next_worker % cpu_workers].send((
                        prev_idx,
                        matches,
                        prev_input.len(),
                    ));
                    next_worker += 1;
                }
            }

            // worker_txs dropped here → workers see channel closed → exit
        });

        // Collector: gather results indexed by block_index
        let mut results: Vec<Option<PzResult<Vec<u8>>>> = (0..num_blocks).map(|_| None).collect();
        for (idx, result) in result_rx {
            results[idx] = Some(result);
        }
        results
            .into_iter()
            .map(|r| r.unwrap_or(Err(PzError::InvalidInput)))
            .collect::<Vec<_>>()
    });

    // Check for errors and collect
    let mut block_data_vec: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for result in compressed_blocks {
        block_data_vec.push(result?);
    }

    // Build output: V2 header + num_blocks + block_table + block_data
    let mut output = Vec::new();
    write_header(&mut output, pipeline, input.len());
    output.extend_from_slice(&(num_blocks as u32).to_le_bytes());
    let block_chunks: Vec<&[u8]> = input.chunks(block_size).collect();
    for (i, compressed) in block_data_vec.iter().enumerate() {
        let orig_block_len = block_chunks[i].len() as u32;
        let comp_block_len = compressed.len() as u32;
        output.extend_from_slice(&comp_block_len.to_le_bytes());
        output.extend_from_slice(&orig_block_len.to_le_bytes());
    }
    for compressed in &block_data_vec {
        output.extend_from_slice(compressed);
    }

    Ok(output)
}

/// Demux LZ77 matches into 3 streams (offsets, lengths, literals) and apply
/// the appropriate entropy encoder for the given pipeline.
#[cfg(feature = "webgpu")]
fn entropy_encode_lz77_block(
    matches: &[crate::lz77::Match],
    original_len: usize,
    pipeline: Pipeline,
) -> PzResult<Vec<u8>> {
    use super::stages::{
        stage_fse_encode, stage_fse_interleaved_encode, stage_huffman_encode, stage_rans_encode,
    };
    use crate::lz77;

    // Serialize matches
    let match_size = lz77::Match::SERIALIZED_SIZE;
    let num_matches = matches.len();
    let lz_len = num_matches * match_size;

    // Demux into 3 streams: offsets (2 bytes each), lengths (2 bytes each), literals (1 byte each)
    let mut offsets = Vec::with_capacity(num_matches * 2);
    let mut lengths = Vec::with_capacity(num_matches * 2);
    let mut literals = Vec::with_capacity(num_matches);

    for m in matches {
        let bytes = m.to_bytes();
        offsets.push(bytes[0]);
        offsets.push(bytes[1]);
        lengths.push(bytes[2]);
        lengths.push(bytes[3]);
        literals.push(bytes[4]);
    }

    let streams = vec![offsets, lengths, literals];

    // Build a StageBlock with the demuxed streams
    let mut block = StageBlock {
        block_index: 0,
        original_len,
        data: Vec::new(),
        streams: Some(streams),
        metadata: StageMetadata::default(),
    };
    block.metadata.pre_entropy_len = Some(lz_len);

    // Apply the appropriate entropy encoder
    let block = match pipeline {
        Pipeline::Deflate => stage_huffman_encode(block)?,
        Pipeline::Lzr => stage_rans_encode(block)?,
        Pipeline::Lzf => stage_fse_encode(block)?,
        Pipeline::Lzfi => stage_fse_interleaved_encode(block)?,
        _ => return Err(PzError::Unsupported),
    };

    Ok(block.data)
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

/// Pipeline-parallel compression: one thread per stage, connected by channels.
pub(crate) fn compress_pipeline_parallel(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    let block_size = options.block_size;
    let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
    let num_blocks = blocks.len();
    let stage_count = pipeline_stage_count(pipeline);

    // Pre-resolve max_match_len so pipeline-parallel stages see the correct
    // limit (Deflate → 258, other LZ → u16::MAX).  The block-parallel path
    // resolves this inside compress_block(), but the pipeline-parallel path
    // bypasses compress_block() and feeds directly into run_compress_stage().
    let mut resolved_options = options.clone();
    if resolved_options.max_match_len.is_none() {
        resolved_options.max_match_len = Some(super::resolve_max_match_len(pipeline, options));
    }

    // Capture original block lengths before `blocks` is moved into the scope.
    let orig_block_lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();

    let results: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        use std::sync::mpsc;

        // Build channel chain: producer → stage[0] → stage[1] → ... → collector
        let (tx_in, mut prev_rx) = mpsc::sync_channel::<PzResult<StageBlock>>(2);

        for stage_idx in 0..stage_count {
            let (tx_out, rx_next) = mpsc::sync_channel::<PzResult<StageBlock>>(2);
            let rx = prev_rx;
            let opts = resolved_options.clone();

            scope.spawn(move || {
                while let Ok(result) = rx.recv() {
                    let output = match result {
                        Ok(block) => run_compress_stage(pipeline, stage_idx, block, &opts),
                        Err(e) => Err(e),
                    };
                    if tx_out.send(output).is_err() {
                        break;
                    }
                }
            });

            prev_rx = rx_next;
        }

        let final_rx = prev_rx;

        // Producer: feed blocks into the first channel.
        // Must run on its own thread so the collector can drain the final
        // channel concurrently — otherwise we deadlock once the bounded
        // channels fill up.
        scope.spawn(move || {
            for (i, chunk) in blocks.iter().enumerate() {
                let block = StageBlock {
                    block_index: i,
                    original_len: chunk.len(),
                    data: chunk.to_vec(),
                    streams: None,
                    metadata: StageMetadata::default(),
                };
                if tx_in.send(Ok(block)).is_err() {
                    break;
                }
            }
            // tx_in dropped here → signals completion
        });

        // Collector: gather results in order (FIFO channels preserve ordering)
        let mut results = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            match final_rx.recv() {
                Ok(Ok(block)) => results.push(Ok(block.data)),
                Ok(Err(e)) => results.push(Err(e)),
                Err(_) => results.push(Err(PzError::InvalidInput)),
            }
        }
        results
    });

    // Build container from collected block data
    let mut block_data_vec: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for result in results {
        block_data_vec.push(result?);
    }

    let mut output = Vec::new();
    write_header(&mut output, pipeline, input.len());
    output.extend_from_slice(&(num_blocks as u32).to_le_bytes());

    for (i, compressed) in block_data_vec.iter().enumerate() {
        let orig_block_len = orig_block_lens[i] as u32;
        let comp_block_len = compressed.len() as u32;
        output.extend_from_slice(&comp_block_len.to_le_bytes());
        output.extend_from_slice(&orig_block_len.to_le_bytes());
    }

    for compressed in &block_data_vec {
        output.extend_from_slice(compressed);
    }

    Ok(output)
}

/// Pipeline-parallel decompression: one thread per stage, connected by channels.
pub(crate) fn decompress_pipeline_parallel(
    payload: &[u8],
    pipeline: Pipeline,
    orig_len: usize,
    num_blocks: usize,
    options: &DecompressOptions,
) -> PzResult<Vec<u8>> {
    let stage_count = pipeline_stage_count(pipeline);

    // Parse block table
    let table_start = 4;
    let table_size = num_blocks * BLOCK_HEADER_SIZE;
    if payload.len() < table_start + table_size {
        return Err(PzError::InvalidInput);
    }

    let mut block_entries: Vec<(usize, usize)> = Vec::with_capacity(num_blocks);
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
    let mut block_slices: Vec<(&[u8], usize)> = Vec::with_capacity(num_blocks);
    let mut pos = data_start;
    for &(comp_len, orig_block_len) in &block_entries {
        if pos + comp_len > payload.len() {
            return Err(PzError::InvalidInput);
        }
        block_slices.push((&payload[pos..pos + comp_len], orig_block_len));
        pos += comp_len;
    }

    let results: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        use std::sync::mpsc;

        let (tx_in, mut prev_rx) = mpsc::sync_channel::<PzResult<StageBlock>>(2);

        for stage_idx in 0..stage_count {
            let (tx_out, rx_next) = mpsc::sync_channel::<PzResult<StageBlock>>(2);
            let rx = prev_rx;
            let opts = options.clone();

            scope.spawn(move || {
                while let Ok(result) = rx.recv() {
                    let output = match result {
                        Ok(block) => run_decompress_stage(pipeline, stage_idx, block, &opts),
                        Err(e) => Err(e),
                    };
                    if tx_out.send(output).is_err() {
                        break;
                    }
                }
            });

            prev_rx = rx_next;
        }

        let final_rx = prev_rx;

        // Producer: feed compressed blocks.
        // Must run on its own thread to avoid deadlock with the collector
        // (bounded channels would block the producer before the collector starts).
        scope.spawn(move || {
            for (i, &(comp_data, orig_block_len)) in block_slices.iter().enumerate() {
                let block = StageBlock {
                    block_index: i,
                    original_len: orig_block_len,
                    data: comp_data.to_vec(),
                    streams: None,
                    metadata: StageMetadata::default(),
                };
                if tx_in.send(Ok(block)).is_err() {
                    break;
                }
            }
            // tx_in dropped here → signals completion
        });

        let mut results = Vec::with_capacity(num_blocks);
        for _ in 0..num_blocks {
            match final_rx.recv() {
                Ok(Ok(block)) => results.push(Ok(block.data)),
                Ok(Err(e)) => results.push(Err(e)),
                Err(_) => results.push(Err(PzError::InvalidInput)),
            }
        }
        results
    });

    let mut output = Vec::with_capacity(orig_len);
    for result in results {
        output.extend_from_slice(&result?);
    }

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}
