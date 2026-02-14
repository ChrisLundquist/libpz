//! Pipeline stage functions: individual compress/decompress transforms and
//! the multi-stream entropy container format.

use crate::bwt;
use crate::fse;
use crate::huffman::HuffmanTree;
use crate::mtf;
use crate::rans;
use crate::rle;
use crate::{PzError, PzResult};

use super::demux::{LzDemuxer, StreamDemuxer};
use super::{CompressOptions, Pipeline};

// ---------------------------------------------------------------------------
// StageBlock — data flowing through pipeline stages
// ---------------------------------------------------------------------------

/// A block moving through the pipeline stages.
///
/// Each stage transforms `data` in place (replacing it with output)
/// and may attach metadata. The `block_index` preserves ordering
/// for final reassembly.
pub(crate) struct StageBlock {
    /// Monotonically increasing index for ordered reassembly.
    /// Not read at runtime (FIFO channels preserve order), but useful for debugging.
    // TODO: Wire into debug logging or tracing spans when adding observability (M5.3/fuzz).
    #[allow(dead_code)]
    pub block_index: usize,
    /// The original uncompressed length of this block.
    pub original_len: usize,
    /// The current data payload. Each stage replaces this with its output.
    pub data: Vec<u8>,
    /// Optional multi-stream payload. When `Some`, downstream stages encode/decode
    /// each stream independently instead of using `data`.
    pub streams: Option<Vec<Vec<u8>>>,
    /// Accumulated metadata from prior stages.
    pub metadata: StageMetadata,
}

/// Metadata accumulated across pipeline stages.
#[derive(Default)]
pub(crate) struct StageMetadata {
    /// BWT primary index (Bw pipeline, set by BWT stage).
    pub bwt_primary_index: Option<u32>,
    /// Bijective BWT factor lengths (Bbw pipeline, set by BBWT stage).
    pub bbwt_factor_lengths: Option<Vec<usize>>,
    /// Length of data before entropy coding (RLE output for Bw/Bbw, LZ output for LZ pipelines).
    pub pre_entropy_len: Option<usize>,
    /// Opaque metadata from the demuxer that must round-trip through the entropy container.
    /// E.g., LZSS num_tokens (4 LE bytes). Empty for formats that don't need it.
    pub demux_meta: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Generic demux stage functions
// ---------------------------------------------------------------------------

/// Generic compress stage: compress with a demuxer, populating streams + metadata.
pub(crate) fn stage_demux_compress(
    mut block: StageBlock,
    demuxer: &LzDemuxer,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    let output = demuxer.compress_and_demux(&block.data, options)?;
    block.metadata.pre_entropy_len = Some(output.pre_entropy_len);
    block.metadata.demux_meta = output.meta;
    block.streams = Some(output.streams);
    block.data.clear();
    Ok(block)
}

/// Generic decompress stage: reinterleave streams with a demuxer.
///
/// Validates that the number of decoded streams matches the demuxer's
/// expected `stream_count()` before passing them to `remux_and_decompress`.
pub(crate) fn stage_demux_decompress(
    mut block: StageBlock,
    demuxer: &LzDemuxer,
) -> PzResult<StageBlock> {
    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    if streams.len() != demuxer.stream_count() {
        return Err(PzError::InvalidInput);
    }
    let decoded =
        demuxer.remux_and_decompress(streams, &block.metadata.demux_meta, block.original_len)?;
    block.data = decoded;
    Ok(block)
}

// ---------------------------------------------------------------------------
// Multi-stream entropy container format
// ---------------------------------------------------------------------------
//
// Wire format:
//   [num_streams: u8]
//   [pre_entropy_len: u32 LE]
//   [meta_len: u16 LE]
//   [meta: meta_len bytes]
//   for each stream: encoder-specific framing (see encode_one/decode_one closures)

/// Multi-stream header size: num_streams(1) + pre_entropy_len(4) + meta_len(2) = 7
const MULTISTREAM_HEADER_SIZE: usize = 7;

/// Encode N streams into a multi-stream container.
///
/// `encode_one` encodes a single byte stream and returns (compressed_data, per_stream_header).
/// For rANS/FSE: per_stream_header is [orig_len: u32][compressed_len: u32], data is the encoded bytes.
/// For Huffman: per_stream_header includes freq table + total_bits.
fn encode_multistream(
    streams: &[Vec<u8>],
    pre_entropy_len: usize,
    meta: &[u8],
    mut encode_one: impl FnMut(&[u8], &mut Vec<u8>) -> PzResult<()>,
) -> PzResult<Vec<u8>> {
    let mut output = Vec::new();
    output.push(streams.len() as u8);
    output.extend_from_slice(&(pre_entropy_len as u32).to_le_bytes());
    output.extend_from_slice(&(meta.len() as u16).to_le_bytes());
    output.extend_from_slice(meta);

    for stream in streams {
        encode_one(stream, &mut output)?;
    }

    Ok(output)
}

/// Decode a multi-stream container back to streams + metadata.
///
/// `decode_one` reads one stream from `data[pos..]` and returns (decoded_bytes, bytes_consumed).
fn decode_multistream(
    data: &[u8],
    mut decode_one: impl FnMut(&[u8]) -> PzResult<(Vec<u8>, usize)>,
) -> PzResult<(Vec<Vec<u8>>, usize, Vec<u8>)> {
    if data.len() < MULTISTREAM_HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }
    let num_streams = data[0] as usize;
    let pre_entropy_len = u32::from_le_bytes([data[1], data[2], data[3], data[4]]) as usize;
    let meta_len = u16::from_le_bytes([data[5], data[6]]) as usize;

    let meta_end = MULTISTREAM_HEADER_SIZE + meta_len;
    if data.len() < meta_end {
        return Err(PzError::InvalidInput);
    }
    let meta = data[MULTISTREAM_HEADER_SIZE..meta_end].to_vec();

    let mut pos = meta_end;
    let mut decoded_streams = Vec::with_capacity(num_streams);

    for _ in 0..num_streams {
        if pos > data.len() {
            return Err(PzError::InvalidInput);
        }
        let (decoded, consumed) = decode_one(&data[pos..])?;
        decoded_streams.push(decoded);
        pos += consumed;
    }

    Ok((decoded_streams, pre_entropy_len, meta))
}

// ---------------------------------------------------------------------------
// Entropy stage functions — Huffman
// ---------------------------------------------------------------------------

/// Huffman encoding stage: encode each stream independently with Huffman coding.
///
/// Per-stream framing:
///   [stream_data_len: u32] [total_bits: u32] [freq_table: 256×u32] [huffman_data]
pub(crate) fn stage_huffman_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    let pre_entropy_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;

    block.data = encode_multistream(
        &streams,
        pre_entropy_len,
        &block.metadata.demux_meta,
        |stream, output| {
            let tree = HuffmanTree::from_data(stream).ok_or(PzError::InvalidInput)?;
            let (huffman_data, total_bits) = tree.encode(stream)?;
            let freq_table = tree.serialize_frequencies();

            output.extend_from_slice(&(huffman_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&(total_bits as u32).to_le_bytes());
            for &freq in &freq_table {
                output.extend_from_slice(&freq.to_le_bytes());
            }
            output.extend_from_slice(&huffman_data);
            Ok(())
        },
    )?;

    Ok(block)
}

/// GPU Huffman encoding stage (WebGPU): same output format as [`stage_huffman_encode()`]
/// but uses the WebGPU backend for histogram computation and Huffman encoding
/// via [`DeviceBuf`].
///
/// Each stream is uploaded to the GPU once, then both the histogram and encoding
/// run on-device — no extra PCI transfers. The output is byte-identical to the
/// CPU path, so the same decoder works for both.
///
/// Falls back to CPU for empty streams where GPU overhead dominates.
#[cfg(feature = "webgpu")]
#[allow(dead_code)] // Available but not currently called; CPU Huffman is faster (see blocks.rs)
pub(crate) fn stage_huffman_encode_webgpu(
    mut block: StageBlock,
    engine: &crate::webgpu::WebGpuEngine,
) -> PzResult<StageBlock> {
    use crate::webgpu::DeviceBuf;

    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    let pre_entropy_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;

    block.data = encode_multistream(
        &streams,
        pre_entropy_len,
        &block.metadata.demux_meta,
        |stream, output| {
            // Upload stream to GPU once
            let device_buf = DeviceBuf::from_host(engine, stream)?;

            // GPU histogram (no re-upload)
            let histogram = engine.byte_histogram_on_device(&device_buf)?;

            let mut freq = crate::frequency::FrequencyTable::new();
            for (i, &count) in histogram.iter().enumerate() {
                freq.byte[i] = count;
            }
            freq.total = freq.byte.iter().map(|&c| c as u64).sum();
            freq.used = freq.byte.iter().filter(|&&c| c > 0).count() as u32;

            let tree = HuffmanTree::from_frequency_table(&freq).ok_or(PzError::InvalidInput)?;
            let freq_table = tree.serialize_frequencies();

            let mut code_lut = [0u32; 256];
            for byte in 0..=255u8 {
                let (codeword, bits) = tree.get_code(byte);
                code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
            }

            // GPU Huffman encode on same device buffer (no re-upload)
            let (huffman_data, total_bits) =
                engine.huffman_encode_on_device(&device_buf, &code_lut)?;

            output.extend_from_slice(&(huffman_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&(total_bits as u32).to_le_bytes());
            for &freq_val in &freq_table {
                output.extend_from_slice(&freq_val.to_le_bytes());
            }
            output.extend_from_slice(&huffman_data);
            Ok(())
        },
    )?;

    Ok(block)
}

/// Huffman decoding stage: parse multi-stream container + Huffman decode each stream.
///
/// Per-stream framing:
///   [stream_data_len: u32] [total_bits: u32] [freq_table: 256×u32] [huffman_data]
pub(crate) fn stage_huffman_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, |data| {
        // Per-stream: [stream_data_len: u32][total_bits: u32][freq_table: 256×u32][huffman_data]
        if data.len() < 1032 {
            return Err(PzError::InvalidInput);
        }
        let stream_data_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let total_bits = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        let mut freq_table = crate::frequency::FrequencyTable::new();
        for i in 0..256 {
            let off = 8 + i * 4;
            freq_table.byte[i] =
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        }
        freq_table.total = freq_table.byte.iter().map(|&f| f as u64).sum();
        freq_table.used = freq_table.byte.iter().filter(|&&f| f > 0).count() as u32;

        let huff_start = 1032;
        if huff_start + stream_data_len > data.len() {
            return Err(PzError::InvalidInput);
        }
        let tree = HuffmanTree::from_frequency_table(&freq_table).ok_or(PzError::InvalidInput)?;
        let decoded = tree.decode(&data[huff_start..huff_start + stream_data_len], total_bits)?;

        let consumed = huff_start + stream_data_len;
        Ok((decoded, consumed))
    })?;

    block.metadata.pre_entropy_len = Some(pre_entropy_len);
    block.metadata.demux_meta = meta;
    block.streams = Some(streams);
    block.data.clear();
    Ok(block)
}

// ---------------------------------------------------------------------------
// Entropy stage functions — rANS
// ---------------------------------------------------------------------------

/// rANS encoding stage: encode each stream independently with rANS.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [rans_data]
pub(crate) fn stage_rans_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    let pre_entropy_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;

    block.data = encode_multistream(
        &streams,
        pre_entropy_len,
        &block.metadata.demux_meta,
        |stream, output| {
            let rans_data = rans::encode(stream);
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            output.extend_from_slice(&(rans_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&rans_data);
            Ok(())
        },
    )?;

    Ok(block)
}

/// rANS decoding stage: parse multi-stream container + rANS decode each stream.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [rans_data]
pub(crate) fn stage_rans_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, |data| {
        if data.len() < 8 {
            return Err(PzError::InvalidInput);
        }
        let orig_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let comp_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if 8 + comp_len > data.len() {
            return Err(PzError::InvalidInput);
        }
        let decoded = rans::decode(&data[8..8 + comp_len], orig_len)?;
        Ok((decoded, 8 + comp_len))
    })?;

    block.metadata.pre_entropy_len = Some(pre_entropy_len);
    block.metadata.demux_meta = meta;
    block.streams = Some(streams);
    block.data.clear();
    Ok(block)
}

// ---------------------------------------------------------------------------
// Entropy stage functions — FSE (multi-stream, LZ-based pipelines)
// ---------------------------------------------------------------------------

/// Choose FSE accuracy_log based on the number of distinct symbols in a stream.
///
/// FSE needs enough table slots to represent all active symbols with reasonable
/// precision. With accuracy_log=7 (128 slots) and 256 active symbols, most get
/// frequency=1 and the coder degrades to raw output. This function ensures
/// at least ~4 table slots per distinct symbol, clamped to the FSE range [5, 12].
fn adaptive_accuracy_log(data: &[u8]) -> u8 {
    if data.is_empty() {
        return fse::DEFAULT_ACCURACY_LOG;
    }
    let mut seen = [false; 256];
    for &b in data {
        seen[b as usize] = true;
    }
    let distinct = seen.iter().filter(|&&s| s).count() as u32;

    // Target: table_size >= 4 * distinct_symbols
    // table_size = 1 << accuracy_log
    // So: accuracy_log = ceil(log2(4 * distinct))
    let target = 4 * distinct;
    let log = if target <= 1 {
        0
    } else {
        32 - (target - 1).leading_zeros() // ceil(log2(target))
    };
    log.clamp(fse::MIN_ACCURACY_LOG as u32, fse::MAX_ACCURACY_LOG as u32) as u8
}

/// FSE encoding stage: encode each stream independently with adaptive FSE.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [fse_data]
pub(crate) fn stage_fse_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    let pre_entropy_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;

    block.data = encode_multistream(
        &streams,
        pre_entropy_len,
        &block.metadata.demux_meta,
        |stream, output| {
            let acc = adaptive_accuracy_log(stream);
            let fse_data = fse::encode_with_accuracy(stream, acc);
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&fse_data);
            Ok(())
        },
    )?;

    Ok(block)
}

/// FSE decoding stage: parse multi-stream container + FSE decode each stream.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [fse_data]
pub(crate) fn stage_fse_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, |data| {
        if data.len() < 8 {
            return Err(PzError::InvalidInput);
        }
        let orig_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let comp_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if 8 + comp_len > data.len() {
            return Err(PzError::InvalidInput);
        }
        let decoded = fse::decode(&data[8..8 + comp_len], orig_len)?;
        Ok((decoded, 8 + comp_len))
    })?;

    block.metadata.pre_entropy_len = Some(pre_entropy_len);
    block.metadata.demux_meta = meta;
    block.streams = Some(streams);
    block.data.clear();
    Ok(block)
}

// ---------------------------------------------------------------------------
// Entropy stage functions — Interleaved FSE (GPU-decodable, LZ-based pipelines)
// ---------------------------------------------------------------------------

/// Interleaved FSE encoding stage: encode each stream with N-way interleaved FSE.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [interleaved_fse_data]
///
/// The interleaved format splits each stream into N independent FSE lanes,
/// enabling parallel decode on GPU (one workgroup per lane).
pub(crate) fn stage_fse_interleaved_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    let pre_entropy_len = block.metadata.pre_entropy_len.unwrap();

    block.data = encode_multistream(
        &streams,
        pre_entropy_len,
        &block.metadata.demux_meta,
        |stream, output| {
            let fse_data = fse::encode_interleaved(stream);
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&fse_data);
            Ok(())
        },
    )?;

    Ok(block)
}

/// GPU interleaved FSE encoding stage (WebGPU): encode each stream with GPU-accelerated
/// N-way interleaved FSE via WebGPU.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [interleaved_fse_data]
///
/// The output wire format is identical to [`stage_fse_interleaved_encode()`], so
/// the same decoder works for both CPU and GPU encoded data.
#[cfg(feature = "webgpu")]
pub(crate) fn stage_fse_interleaved_encode_webgpu(
    mut block: StageBlock,
    engine: &crate::webgpu::WebGpuEngine,
) -> PzResult<StageBlock> {
    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    let pre_entropy_len = block.metadata.pre_entropy_len.unwrap();

    block.data = encode_multistream(
        &streams,
        pre_entropy_len,
        &block.metadata.demux_meta,
        |stream, output| {
            // Match CPU's DEFAULT_INTERLEAVE=4 so the decoder's SIMD fast path activates
            let num_states = 4;
            let fse_data =
                engine.fse_encode_interleaved_gpu(stream, num_states, fse::DEFAULT_ACCURACY_LOG)?;
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&fse_data);
            Ok(())
        },
    )?;

    Ok(block)
}

/// Interleaved FSE decoding stage (CPU): parse multi-stream container + decode each stream.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [interleaved_fse_data]
pub(crate) fn stage_fse_interleaved_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, |data| {
        if data.len() < 8 {
            return Err(PzError::InvalidInput);
        }
        let orig_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let comp_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if 8 + comp_len > data.len() {
            return Err(PzError::InvalidInput);
        }
        let decoded = fse::decode_interleaved(&data[8..8 + comp_len], orig_len)?;
        Ok((decoded, 8 + comp_len))
    })?;

    block.metadata.pre_entropy_len = Some(pre_entropy_len);
    block.metadata.demux_meta = meta;
    block.streams = Some(streams);
    block.data.clear();
    Ok(block)
}

/// GPU interleaved FSE decoding stage (WebGPU): parse multi-stream container
/// + decode each stream's interleaved FSE data on the GPU.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [interleaved_fse_data]
///
/// The input wire format is identical to [`stage_fse_interleaved_decode()`], so
/// GPU decode works on data produced by either CPU or GPU encode.
#[cfg(feature = "webgpu")]
pub(crate) fn stage_fse_interleaved_decode_webgpu(
    mut block: StageBlock,
    engine: &crate::webgpu::WebGpuEngine,
) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, |data| {
        if data.len() < 8 {
            return Err(PzError::InvalidInput);
        }
        let orig_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let comp_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
        if 8 + comp_len > data.len() {
            return Err(PzError::InvalidInput);
        }
        let decoded = engine.fse_decode(&data[8..8 + comp_len], orig_len)?;
        Ok((decoded, 8 + comp_len))
    })?;

    block.metadata.pre_entropy_len = Some(pre_entropy_len);
    block.metadata.demux_meta = meta;
    block.streams = Some(streams);
    block.data.clear();
    Ok(block)
}

// ---------------------------------------------------------------------------
// BWT pipeline stages (Bw)
// ---------------------------------------------------------------------------

/// Bw stage 0: BWT encoding.
pub(crate) fn stage_bwt_encode(
    mut block: StageBlock,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    let bwt_result = super::bwt_encode_with_backend(&block.data, options)?;
    block.metadata.bwt_primary_index = Some(bwt_result.primary_index);
    block.data = bwt_result.data;
    Ok(block)
}

/// Bw stage 1: MTF encoding.
pub(crate) fn stage_mtf_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = mtf::encode(&block.data);
    Ok(block)
}

/// Bw stage 2: RLE encoding.
pub(crate) fn stage_rle_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = rle::encode(&block.data);
    block.metadata.pre_entropy_len = Some(block.data.len());
    Ok(block)
}

/// Bw stage 3: FSE encoding + serialization.
pub(crate) fn stage_fse_encode_bw(mut block: StageBlock) -> PzResult<StageBlock> {
    let primary_index = block
        .metadata
        .bwt_primary_index
        .ok_or(PzError::InvalidInput)?;
    let rle_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;
    let fse_data = fse::encode(&block.data);

    let mut output = Vec::new();
    output.extend_from_slice(&primary_index.to_le_bytes());
    output.extend_from_slice(&(rle_len as u32).to_le_bytes());
    output.extend_from_slice(&fse_data);

    block.data = output;
    Ok(block)
}

/// Bw decompress stage 0: parse metadata + FSE decode.
pub(crate) fn stage_fse_decode_bw(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 8 {
        return Err(PzError::InvalidInput);
    }
    let primary_index =
        u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);
    let rle_len =
        u32::from_le_bytes([block.data[4], block.data[5], block.data[6], block.data[7]]) as usize;

    block.metadata.bwt_primary_index = Some(primary_index);
    block.data = fse::decode(&block.data[8..], rle_len)?;
    Ok(block)
}

/// Bw decompress stage 1: RLE decode.
pub(crate) fn stage_rle_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = rle::decode(&block.data)?;
    Ok(block)
}

/// Bw decompress stage 2: MTF decode.
pub(crate) fn stage_mtf_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = mtf::decode(&block.data);
    Ok(block)
}

/// Bw decompress stage 3: BWT decode.
pub(crate) fn stage_bwt_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let primary_index = block
        .metadata
        .bwt_primary_index
        .ok_or(PzError::InvalidInput)?;
    block.data = bwt::decode(&block.data, primary_index)?;
    Ok(block)
}

// ---------------------------------------------------------------------------
// BBW pipeline stages (Bbw — bijective BWT)
// ---------------------------------------------------------------------------

/// Bbw stage 0: bijective BWT encoding.
pub(crate) fn stage_bbwt_encode(
    mut block: StageBlock,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    let (bwt_data, factor_lengths) = super::bbwt_encode_with_backend(&block.data, options)?;
    block.metadata.bbwt_factor_lengths = Some(factor_lengths);
    block.data = bwt_data;
    Ok(block)
}

/// Bbw stage 3: FSE encoding + serialization.
///
/// Format: [num_factors: u16] [factor_lengths: u32 × num_factors] [rle_len: u32] [fse_data...]
pub(crate) fn stage_fse_encode_bbw(mut block: StageBlock) -> PzResult<StageBlock> {
    let factor_lengths = block
        .metadata
        .bbwt_factor_lengths
        .take()
        .ok_or(PzError::InvalidInput)?;
    let rle_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;
    let fse_data = fse::encode(&block.data);

    let mut output = Vec::new();
    output.extend_from_slice(&(factor_lengths.len() as u16).to_le_bytes());
    for &fl in &factor_lengths {
        output.extend_from_slice(&(fl as u32).to_le_bytes());
    }
    output.extend_from_slice(&(rle_len as u32).to_le_bytes());
    output.extend_from_slice(&fse_data);

    block.data = output;
    Ok(block)
}

/// Bbw decompress stage 0: parse metadata + FSE decode.
pub(crate) fn stage_fse_decode_bbw(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 2 {
        return Err(PzError::InvalidInput);
    }
    let num_factors = u16::from_le_bytes([block.data[0], block.data[1]]) as usize;
    let header_len = 2 + num_factors * 4 + 4; // num_factors(2) + factor_lengths(4*k) + rle_len(4)
    if block.data.len() < header_len {
        return Err(PzError::InvalidInput);
    }

    let mut factor_lengths = Vec::with_capacity(num_factors);
    for i in 0..num_factors {
        let offset = 2 + i * 4;
        let fl = u32::from_le_bytes([
            block.data[offset],
            block.data[offset + 1],
            block.data[offset + 2],
            block.data[offset + 3],
        ]) as usize;
        factor_lengths.push(fl);
    }

    let rle_offset = 2 + num_factors * 4;
    let rle_len = u32::from_le_bytes([
        block.data[rle_offset],
        block.data[rle_offset + 1],
        block.data[rle_offset + 2],
        block.data[rle_offset + 3],
    ]) as usize;

    block.metadata.bbwt_factor_lengths = Some(factor_lengths);
    block.data = fse::decode(&block.data[header_len..], rle_len)?;
    Ok(block)
}

/// Bbw decompress stage 3: bijective BWT decode.
pub(crate) fn stage_bbwt_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let factor_lengths = block
        .metadata
        .bbwt_factor_lengths
        .take()
        .ok_or(PzError::InvalidInput)?;
    block.data = bwt::decode_bijective(&block.data, &factor_lengths)?;
    Ok(block)
}

// ---------------------------------------------------------------------------
// Stage dispatch — maps (pipeline, stage_idx) to the appropriate function
// ---------------------------------------------------------------------------

/// Number of compression stages in a pipeline.
pub(crate) fn pipeline_stage_count(pipeline: Pipeline) -> usize {
    match pipeline {
        Pipeline::Deflate => 2,
        Pipeline::Bw => 4,
        Pipeline::Bbw => 4,
        Pipeline::Lzr => 2,
        Pipeline::Lzf | Pipeline::Lzfi => 2,
        Pipeline::LzssR | Pipeline::Lz78R => 2,
    }
}

/// Dispatch to the appropriate compression stage function.
pub(crate) fn run_compress_stage(
    pipeline: Pipeline,
    stage_idx: usize,
    block: StageBlock,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    match (pipeline, stage_idx) {
        (Pipeline::Deflate, 0) => stage_demux_compress(block, &LzDemuxer::Lz77, options),
        (Pipeline::Deflate, 1) => stage_huffman_encode(block),
        (Pipeline::Bw, 0) => stage_bwt_encode(block, options),
        (Pipeline::Bw, 1) => stage_mtf_encode(block),
        (Pipeline::Bw, 2) => stage_rle_encode(block),
        (Pipeline::Bw, 3) => stage_fse_encode_bw(block),
        (Pipeline::Bbw, 0) => stage_bbwt_encode(block, options),
        (Pipeline::Bbw, 1) => stage_mtf_encode(block),
        (Pipeline::Bbw, 2) => stage_rle_encode(block),
        (Pipeline::Bbw, 3) => stage_fse_encode_bbw(block),
        (Pipeline::Lzr, 0) => stage_demux_compress(block, &LzDemuxer::Lz77, options),
        (Pipeline::Lzr, 1) => stage_rans_encode(block),
        (Pipeline::Lzf, 0) => stage_demux_compress(block, &LzDemuxer::Lz77, options),
        (Pipeline::Lzf, 1) => stage_fse_encode(block),
        (Pipeline::Lzfi, 0) => stage_demux_compress(block, &LzDemuxer::Lzss, options),
        (Pipeline::Lzfi, 1) => {
            #[cfg(feature = "webgpu")]
            {
                if let super::Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        return stage_fse_interleaved_encode_webgpu(block, engine);
                    }
                }
            }
            stage_fse_interleaved_encode(block)
        }
        (Pipeline::LzssR, 0) => stage_demux_compress(block, &LzDemuxer::Lzss, options),
        (Pipeline::LzssR, 1) => stage_rans_encode(block),
        (Pipeline::Lz78R, 0) => stage_demux_compress(block, &LzDemuxer::Lz78, options),
        (Pipeline::Lz78R, 1) => stage_rans_encode(block),
        _ => Err(PzError::Unsupported),
    }
}

/// Dispatch to the appropriate decompression stage function.
pub(crate) fn run_decompress_stage(
    pipeline: Pipeline,
    stage_idx: usize,
    block: StageBlock,
    options: &super::DecompressOptions,
) -> PzResult<StageBlock> {
    match (pipeline, stage_idx) {
        // Deflate: Huffman decode(0) → LZ77 decompress(1)
        (Pipeline::Deflate, 0) => stage_huffman_decode(block),
        (Pipeline::Deflate, 1) => stage_demux_decompress(block, &LzDemuxer::Lz77),
        // Bw: FSE decode(0) → RLE decode(1) → MTF decode(2) → BWT decode(3)
        (Pipeline::Bw, 0) => stage_fse_decode_bw(block),
        (Pipeline::Bw, 1) => stage_rle_decode(block),
        (Pipeline::Bw, 2) => stage_mtf_decode(block),
        (Pipeline::Bw, 3) => stage_bwt_decode(block),
        // Bbw: FSE decode(0) → RLE decode(1) → MTF decode(2) → BBWT decode(3)
        (Pipeline::Bbw, 0) => stage_fse_decode_bbw(block),
        (Pipeline::Bbw, 1) => stage_rle_decode(block),
        (Pipeline::Bbw, 2) => stage_mtf_decode(block),
        (Pipeline::Bbw, 3) => stage_bbwt_decode(block),
        // Lzr: rANS decode(0) → LZ77 decompress(1)
        (Pipeline::Lzr, 0) => stage_rans_decode(block),
        (Pipeline::Lzr, 1) => stage_demux_decompress(block, &LzDemuxer::Lz77),
        // Lzf: FSE decode(0) → LZ77 decompress(1)
        (Pipeline::Lzf, 0) => stage_fse_decode(block),
        (Pipeline::Lzf, 1) => stage_demux_decompress(block, &LzDemuxer::Lz77),
        // Lzfi: interleaved FSE decode(0) → LZSS decompress(1)
        (Pipeline::Lzfi, 0) => {
            #[cfg(feature = "webgpu")]
            {
                if let super::Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        return stage_fse_interleaved_decode_webgpu(block, engine);
                    }
                }
            }
            let _ = options;
            stage_fse_interleaved_decode(block)
        }
        (Pipeline::Lzfi, 1) => stage_demux_decompress(block, &LzDemuxer::Lzss),
        // LzssR: rANS decode(0) → LZSS decompress(1)
        (Pipeline::LzssR, 0) => stage_rans_decode(block),
        (Pipeline::LzssR, 1) => stage_demux_decompress(block, &LzDemuxer::Lzss),
        // Lz78R: rANS decode(0) → LZ78 decompress(1)
        (Pipeline::Lz78R, 0) => stage_rans_decode(block),
        (Pipeline::Lz78R, 1) => stage_demux_decompress(block, &LzDemuxer::Lz78),
        _ => Err(PzError::Unsupported),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PzError;

    #[test]
    fn test_demux_decompress_wrong_stream_count() {
        // LZ77 expects 3 streams; provide 2 → should fail with InvalidInput
        let block = StageBlock {
            block_index: 0,
            original_len: 100,
            data: Vec::new(),
            streams: Some(vec![vec![0u8; 10], vec![0u8; 10]]), // 2 streams, need 3
            metadata: StageMetadata::default(),
        };
        let result = stage_demux_decompress(block, &LzDemuxer::Lz77);
        assert!(matches!(result, Err(PzError::InvalidInput)));
    }

    #[test]
    fn test_demux_decompress_no_streams() {
        // No streams at all → should fail with InvalidInput
        let block = StageBlock {
            block_index: 0,
            original_len: 100,
            data: Vec::new(),
            streams: None,
            metadata: StageMetadata::default(),
        };
        let result = stage_demux_decompress(block, &LzDemuxer::Lz77);
        assert!(matches!(result, Err(PzError::InvalidInput)));
    }
}
