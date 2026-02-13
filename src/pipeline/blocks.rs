//! Per-pipeline single-block compress and decompress implementations.
//!
//! LZ-based pipelines (Deflate, Lzr, Lzf, Lzfi, LzssR, Lz78R) use a unified path:
//!   compress:   demux → entropy_encode
//!   decompress: entropy_decode → demux
//!
//! BWT-based pipelines (Bw, Bbw) have their own structure and are handled separately.

use crate::bwt;
use crate::fse;
use crate::mtf;
use crate::rle;
use crate::{PzError, PzResult};

use super::demux::{demuxer_for_pipeline, LzDemuxer};
use super::stages::*;
#[cfg(any(feature = "opencl", feature = "webgpu"))]
use super::Backend;
use super::{resolve_max_match_len, CompressOptions, Pipeline};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Compress a single block using the appropriate pipeline (no container header).
pub(crate) fn compress_block(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    // Resolve max match length for this pipeline (Deflate=258, others=u16::MAX).
    // Clone options only when we need to override the default.
    let resolved;
    let opts = if options.max_match_len.is_none() && demuxer_for_pipeline(pipeline).is_some() {
        resolved = CompressOptions {
            max_match_len: Some(resolve_max_match_len(pipeline, options)),
            ..options.clone()
        };
        &resolved
    } else {
        options
    };

    match demuxer_for_pipeline(pipeline) {
        Some(demuxer) => compress_block_lz(input, pipeline, &demuxer, opts),
        None => match pipeline {
            Pipeline::Bw => compress_block_bw(input, opts),
            Pipeline::Bbw => compress_block_bbw(input, opts),
            _ => Err(PzError::Unsupported),
        },
    }
}

/// Decompress a single block using the appropriate pipeline (no container header).
pub(crate) fn decompress_block(
    payload: &[u8],
    pipeline: Pipeline,
    orig_len: usize,
) -> PzResult<Vec<u8>> {
    match demuxer_for_pipeline(pipeline) {
        Some(demuxer) => decompress_block_lz(payload, pipeline, &demuxer, orig_len),
        None => match pipeline {
            Pipeline::Bw => decompress_block_bw(payload, orig_len),
            Pipeline::Bbw => decompress_block_bbw(payload, orig_len),
            _ => Err(PzError::Unsupported),
        },
    }
}

// ---------------------------------------------------------------------------
// Unified LZ-based pipeline path
// ---------------------------------------------------------------------------

/// Compress a single block for any LZ-based pipeline.
///
/// All LZ pipelines share the same structure:
///   input → stage_demux_compress(demuxer) → entropy_encode(pipeline) → output
fn compress_block_lz(
    input: &[u8],
    pipeline: Pipeline,
    demuxer: &LzDemuxer,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_demux_compress(block, demuxer, options)?;
    let block = entropy_encode(block, pipeline, input.len(), options)?;
    Ok(block.data)
}

/// Decompress a single block for any LZ-based pipeline.
///
/// All LZ pipelines share the same structure:
///   payload → entropy_decode(pipeline) → stage_demux_decompress(demuxer) → output
fn decompress_block_lz(
    payload: &[u8],
    pipeline: Pipeline,
    demuxer: &LzDemuxer,
    orig_len: usize,
) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: orig_len,
        data: payload.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = entropy_decode(block, pipeline)?;
    let block = stage_demux_decompress(block, demuxer)?;
    Ok(block.data)
}

// ---------------------------------------------------------------------------
// Entropy encode/decode dispatch
// ---------------------------------------------------------------------------

/// Dispatch to the correct entropy encoder for a pipeline.
///
/// For Huffman (Deflate), GPU variants are used when a GPU backend is active.
fn entropy_encode(
    block: StageBlock,
    pipeline: Pipeline,
    input_len: usize,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    match pipeline {
        Pipeline::Deflate => {
            // GPU Huffman when available, otherwise CPU Huffman.
            #[cfg(feature = "opencl")]
            {
                if let Backend::OpenCl = options.backend {
                    if let Some(ref engine) = options.opencl_engine {
                        return stage_huffman_encode_gpu(block, engine);
                    }
                }
            }

            // Note: WebGPU Huffman is intentionally NOT used here.
            // Profiling shows CPU Huffman (~0.5ms/256KB) is faster than the
            // WebGPU path (~2ms) due to CPU↔GPU round-trips for bit-length
            // computation and prefix-sum. The GPU LZ77 path provides the
            // parallelism win; entropy encoding is faster on the CPU.
            #[cfg(feature = "webgpu")]
            let _ = &options;

            // Suppress unused variable warning when no GPU features are enabled
            let _ = (input_len, options);
            stage_huffman_encode(block)
        }
        Pipeline::Lzr | Pipeline::LzssR | Pipeline::Lz78R => {
            let _ = (input_len, options);
            stage_rans_encode(block)
        }
        Pipeline::Lzf => {
            let _ = (input_len, options);
            stage_fse_encode(block)
        }
        Pipeline::Lzfi => {
            #[cfg(feature = "opencl")]
            {
                if let Backend::OpenCl = options.backend {
                    if let Some(ref engine) = options.opencl_engine {
                        return stage_fse_interleaved_encode_gpu(block, engine);
                    }
                }
            }
            #[cfg(feature = "webgpu")]
            {
                if let Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        return stage_fse_interleaved_encode_webgpu(block, engine);
                    }
                }
            }
            let _ = (input_len, options);
            stage_fse_interleaved_encode(block)
        }
        _ => Err(PzError::Unsupported),
    }
}

/// Dispatch to the correct entropy decoder for a pipeline.
fn entropy_decode(block: StageBlock, pipeline: Pipeline) -> PzResult<StageBlock> {
    match pipeline {
        Pipeline::Deflate => stage_huffman_decode(block),
        Pipeline::Lzr | Pipeline::LzssR | Pipeline::Lz78R => stage_rans_decode(block),
        Pipeline::Lzf => stage_fse_decode(block),
        Pipeline::Lzfi => stage_fse_interleaved_decode(block),
        _ => Err(PzError::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// BW pipeline: BWT + MTF + RLE + FSE
// ---------------------------------------------------------------------------

/// Compress a single block using the BW pipeline (no container header).
fn compress_block_bw(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_bwt_encode(block, options)?;
    let block = stage_mtf_encode(block)?;
    let block = stage_rle_encode(block)?;
    let block = stage_fse_encode_bw(block)?;
    Ok(block.data)
}

/// Decompress a single BW block (no container header).
fn decompress_block_bw(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 8 {
        return Err(PzError::InvalidInput);
    }

    let primary_index = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let rle_len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;

    let entropy_data = &payload[8..];

    // Stage 1: FSE decoder
    let rle_data = fse::decode(entropy_data, rle_len)?;

    // Stage 2: RLE decode
    let mtf_data = rle::decode(&rle_data)?;

    // Stage 3: Inverse MTF
    let bwt_data = mtf::decode(&mtf_data);

    // Stage 4: Inverse BWT
    let output = bwt::decode(&bwt_data, primary_index)?;

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// BBW pipeline: Bijective BWT + MTF + RLE + FSE
// ---------------------------------------------------------------------------

/// Compress a single block using the Bbw pipeline (no container header).
fn compress_block_bbw(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_bbwt_encode(block, options)?;
    let block = stage_mtf_encode(block)?;
    let block = stage_rle_encode(block)?;
    let block = stage_fse_encode_bbw(block)?;
    Ok(block.data)
}

/// Decompress a single Bbw block (no container header).
fn decompress_block_bbw(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 6 {
        return Err(PzError::InvalidInput);
    }

    // Parse header: [num_factors: u16] [factor_lengths: u32 × k] [rle_len: u32]
    let num_factors = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let header_len = 2 + num_factors * 4 + 4;
    if payload.len() < header_len {
        return Err(PzError::InvalidInput);
    }

    let mut factor_lengths = Vec::with_capacity(num_factors);
    for i in 0..num_factors {
        let offset = 2 + i * 4;
        let fl = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]) as usize;
        factor_lengths.push(fl);
    }

    let rle_offset = 2 + num_factors * 4;
    let rle_len = u32::from_le_bytes([
        payload[rle_offset],
        payload[rle_offset + 1],
        payload[rle_offset + 2],
        payload[rle_offset + 3],
    ]) as usize;

    let entropy_data = &payload[header_len..];

    // Stage 1: FSE decode
    let rle_data = fse::decode(entropy_data, rle_len)?;

    // Stage 2: RLE decode
    let mtf_data = rle::decode(&rle_data)?;

    // Stage 3: Inverse MTF
    let bwt_data = mtf::decode(&mtf_data);

    // Stage 4: Inverse bijective BWT
    let output = bwt::decode_bijective(&bwt_data, &factor_lengths)?;

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}
