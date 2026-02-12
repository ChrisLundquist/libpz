//! Per-pipeline single-block compress and decompress implementations.

use crate::bwt;
use crate::fse;
use crate::mtf;
use crate::rle;
use crate::{PzError, PzResult};

use super::demux::LzDemuxer;
use super::stages::*;
#[cfg(any(feature = "opencl", feature = "webgpu"))]
use super::Backend;
use super::CompressOptions;
use super::Pipeline;

/// Compress a single block using the appropriate pipeline (no container header).
pub(crate) fn compress_block(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    match pipeline {
        Pipeline::Deflate => compress_block_deflate(input, options),
        Pipeline::Bw => compress_block_bw(input, options),
        Pipeline::Bbw => compress_block_bbw(input, options),
        Pipeline::Lzr => compress_block_lzr(input, options),
        Pipeline::Lzf => compress_block_lzf(input, options),
        Pipeline::LzssR => compress_block_lzssr(input),
        Pipeline::Lz78R => compress_block_lz78r(input),
    }
}

/// Decompress a single block using the appropriate pipeline (no container header).
pub(crate) fn decompress_block(
    payload: &[u8],
    pipeline: Pipeline,
    orig_len: usize,
) -> PzResult<Vec<u8>> {
    match pipeline {
        Pipeline::Deflate => decompress_block_deflate(payload, orig_len),
        Pipeline::Bw => decompress_block_bw(payload, orig_len),
        Pipeline::Bbw => decompress_block_bbw(payload, orig_len),
        Pipeline::Lzr => decompress_block_lzr(payload, orig_len),
        Pipeline::Lzf => decompress_block_lzf(payload, orig_len),
        Pipeline::LzssR => decompress_block_lzssr(payload, orig_len),
        Pipeline::Lz78R => decompress_block_lz78r(payload, orig_len),
    }
}

// --- DEFLATE pipeline: LZ77 + Huffman ---

/// Compress a single block using the Deflate pipeline (no container header).
/// Returns pipeline-specific metadata + compressed data.
fn compress_block_deflate(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    // GPU chained path: LZ77 + Huffman on GPU without host round-trip
    #[cfg(feature = "opencl")]
    {
        if let Backend::OpenCl = options.backend {
            if let Some(ref engine) = options.opencl_engine {
                if !engine.is_cpu_device() && input.len() >= crate::opencl::MIN_GPU_INPUT_SIZE {
                    return engine.deflate_chained(input);
                }
            }
        }
    }

    #[cfg(feature = "webgpu")]
    {
        if let Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                if !engine.is_cpu_device()
                    && input.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE
                    && input.len() <= engine.max_dispatch_input_size()
                {
                    return engine.deflate_chained(input);
                }
            }
        }
    }

    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_demux_compress(block, &LzDemuxer::Lz77, options)?;
    let block = stage_huffman_encode(block)?;
    Ok(block.data)
}

/// Decompress a single Deflate block (no container header).
fn decompress_block_deflate(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: orig_len,
        data: payload.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_huffman_decode(block)?;
    let block = stage_demux_decompress(block, &LzDemuxer::Lz77)?;
    Ok(block.data)
}

// --- BW pipeline: BWT + MTF + RLE + FSE ---

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

// --- BBW pipeline: Bijective BWT + MTF + RLE + FSE ---

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

// --- LZR pipeline: LZ77 + rANS ---

/// Compress a single block using the Lzr pipeline (no container header).
fn compress_block_lzr(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_demux_compress(block, &LzDemuxer::Lz77, options)?;
    let block = stage_rans_encode(block)?;
    Ok(block.data)
}

/// Decompress a single Lzr block (no container header).
fn decompress_block_lzr(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: orig_len,
        data: payload.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_rans_decode(block)?;
    let block = stage_demux_decompress(block, &LzDemuxer::Lz77)?;
    Ok(block.data)
}

// --- LZF pipeline: LZ77 + FSE ---

/// Compress a single block using the Lzf pipeline (no container header).
fn compress_block_lzf(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_demux_compress(block, &LzDemuxer::Lz77, options)?;
    let block = stage_fse_encode(block)?;
    Ok(block.data)
}

/// Decompress a single Lzf block (no container header).
fn decompress_block_lzf(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: orig_len,
        data: payload.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_fse_decode(block)?;
    let block = stage_demux_decompress(block, &LzDemuxer::Lz77)?;
    Ok(block.data)
}

// --- LZSS+rANS pipeline: LZSS + rANS ---

/// Compress a single block using the LzssR pipeline (no container header).
fn compress_block_lzssr(input: &[u8]) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_demux_compress(block, &LzDemuxer::Lzss, &CompressOptions::default())?;
    let block = stage_rans_encode(block)?;
    Ok(block.data)
}

/// Decompress a single LzssR block (no container header).
fn decompress_block_lzssr(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: orig_len,
        data: payload.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_rans_decode(block)?;
    let block = stage_demux_decompress(block, &LzDemuxer::Lzss)?;
    Ok(block.data)
}

// --- LZ78+rANS pipeline: LZ78 + rANS ---

/// Compress a single block using the Lz78R pipeline (no container header).
fn compress_block_lz78r(input: &[u8]) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_demux_compress(block, &LzDemuxer::Lz78, &CompressOptions::default())?;
    let block = stage_rans_encode(block)?;
    Ok(block.data)
}

/// Decompress a single Lz78R block (no container header).
fn decompress_block_lz78r(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: orig_len,
        data: payload.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_rans_decode(block)?;
    let block = stage_demux_decompress(block, &LzDemuxer::Lz78)?;
    Ok(block.data)
}
