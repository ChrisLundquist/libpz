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
use super::{write_header, CompressOptions, Pipeline, BLOCK_HEADER_SIZE};

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
    // Try batched GPU LZ77 path for LZ77-based pipelines with WebGPU backend
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
/// Submits all blocks to the GPU for LZ77 matching in one batch (hiding
/// transfer latency), then entropy-encodes each block on CPU threads.
///
/// This is used for LZ77-based pipelines (Deflate, Lzr, Lzf) when the
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

    // Phase 1: Batch GPU LZ77 matching (submit all → single sync → readback all)
    let match_vecs = engine.find_matches_batched(&blocks)?;

    // Phase 2: Demux + entropy encode each block in parallel on CPU threads
    let compressed_blocks: Vec<PzResult<Vec<u8>>> =
        std::thread::scope(|scope| {
            let max_concurrent = num_threads.min(num_blocks);
            let mut handles: Vec<std::thread::ScopedJoinHandle<PzResult<Vec<u8>>>> =
                Vec::with_capacity(max_concurrent);
            let mut results: Vec<PzResult<Vec<u8>>> = Vec::with_capacity(num_blocks);

            for (i, matches) in match_vecs.into_iter().enumerate() {
                if handles.len() >= max_concurrent {
                    let handle = handles.remove(0);
                    results.push(handle.join().unwrap_or(Err(PzError::InvalidInput)));
                }
                let block_input = blocks[i];
                handles.push(scope.spawn(move || {
                    entropy_encode_lz77_block(&matches, block_input.len(), pipeline)
                }));
            }

            for handle in handles {
                results.push(handle.join().unwrap_or(Err(PzError::InvalidInput)));
            }

            results
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
    threads: usize,
) -> PzResult<Vec<u8>> {
    let num_threads = super::resolve_thread_count(threads);

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
            handles.push(scope.spawn(move || {
                super::blocks::decompress_block(comp_data, pipeline, orig_block_len)
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

            scope.spawn(move || {
                while let Ok(result) = rx.recv() {
                    let output = match result {
                        Ok(block) => run_decompress_stage(pipeline, stage_idx, block),
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
