//! Pipeline stage functions: individual compress/decompress transforms and
//! the multi-stream entropy container format.

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
#[derive(Clone)]
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
#[derive(Clone, Default)]
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
/// High-bit flag on per-stream compressed length signaling interleaved rANS payload.
const RANS_INTERLEAVED_FLAG: u32 = 1 << 31;
/// Bit-30 flag signaling Recoil split-point metadata is appended to the payload.
const RANS_RECOIL_FLAG: u32 = 1 << 30;
/// Bit-29 flag signaling shared-stream rANS payload (ryg_rans-style).
const RANS_SHARED_STREAM_FLAG: u32 = 1 << 29;
/// Mask to extract the actual compressed length (bits 0-28, max 512 MiB).
const RANS_COMP_LEN_MASK: u32 =
    !(RANS_INTERLEAVED_FLAG | RANS_RECOIL_FLAG | RANS_SHARED_STREAM_FLAG);

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
            // Handle empty streams: emit empty huffman data with zero frequency table
            if stream.is_empty() {
                output.extend_from_slice(&0u32.to_le_bytes()); // huffman_data.len() = 0
                output.extend_from_slice(&0u32.to_le_bytes()); // total_bits = 0
                                                               // Emit empty frequency table (256 zeros)
                for _ in 0..256 {
                    output.extend_from_slice(&0u32.to_le_bytes());
                }
                return Ok(());
            }

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

/// Huffman decoding stage: parse multi-stream container + Huffman decode each stream.
///
/// Per-stream framing:
///   [stream_data_len: u32] [total_bits: u32] [freq_table: 256×u32] [huffman_data]
pub(crate) fn stage_huffman_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, |data| {
        // Per-stream: [stream_data_len: u32][total_bits: u32][freq_table: 256×u32][huffman_data]
        if data.len() < 8 {
            return Err(PzError::InvalidInput);
        }
        let stream_data_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let total_bits = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;

        // Handle empty streams: freq table still exists but all zeros
        if stream_data_len == 0 && total_bits == 0 {
            if data.len() < 1032 {
                return Err(PzError::InvalidInput);
            }
            return Ok((Vec::new(), 1032));
        }

        if data.len() < 1032 {
            return Err(PzError::InvalidInput);
        }
        let mut freq_table = crate::frequency::FrequencyTable::new();
        for i in 0..256 {
            let off = 8 + i * 4;
            freq_table.byte[i] =
                u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        }
        freq_table.recompute_totals();

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

/// rANS encoding stage with policy options.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32 with optional interleaved flag] [rans_data]
pub(crate) fn stage_rans_encode_with_options(
    mut block: StageBlock,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
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
            let (payload, flagged_len) = if options.rans_shared_stream
                && options.rans_interleaved
                && stream.len() >= options.rans_interleaved_min_bytes
            {
                // Shared-stream rANS (ryg_rans-style): all lanes share one word stream.
                // Incompatible with Recoil (shared-stream has no per-lane word boundaries).
                let data = rans::encode_shared_stream_n(
                    stream,
                    options.rans_interleaved_states,
                    rans::DEFAULT_SCALE_BITS,
                );
                if data.len() as u64 >= (1u64 << 29) {
                    return Err(PzError::InvalidInput);
                }
                let len = (data.len() as u32) | RANS_SHARED_STREAM_FLAG;
                (data, len)
            } else if options.rans_interleaved && stream.len() >= options.rans_interleaved_min_bytes
            {
                let data = rans::encode_interleaved_n(
                    stream,
                    options.rans_interleaved_states,
                    rans::DEFAULT_SCALE_BITS,
                );

                if options.rans_recoil {
                    // Generate Recoil split metadata and append to payload.
                    let meta = crate::recoil::recoil_generate_splits(
                        &data,
                        stream.len(),
                        options.rans_recoil_splits,
                    )?;
                    let meta_bytes = meta.serialize();
                    let mut combined = Vec::with_capacity(4 + data.len() + meta_bytes.len());
                    combined.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
                    combined.extend_from_slice(&data);
                    combined.extend_from_slice(&meta_bytes);
                    if combined.len() as u64 >= (1u64 << 29) {
                        return Err(PzError::InvalidInput);
                    }
                    let len = (combined.len() as u32) | RANS_INTERLEAVED_FLAG | RANS_RECOIL_FLAG;
                    (combined, len)
                } else {
                    if data.len() >= (1usize << 29) {
                        return Err(PzError::InvalidInput);
                    }
                    let len = (data.len() as u32) | RANS_INTERLEAVED_FLAG;
                    (data, len)
                }
            } else {
                let data = rans::encode(stream);
                if data.len() >= (1usize << 29) {
                    return Err(PzError::InvalidInput);
                }
                let len = data.len() as u32;
                (data, len)
            };
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            output.extend_from_slice(&flagged_len.to_le_bytes());
            output.extend_from_slice(&payload);
            Ok(())
        },
    )?;

    Ok(block)
}

/// Parsed rANS per-stream header fields.
struct RansStreamHeader<'a> {
    orig_len: usize,
    is_interleaved: bool,
    is_recoil: bool,
    is_shared_stream: bool,
    payload: &'a [u8],
}

/// Parse a rANS per-stream header: [orig_len: u32] [compressed_len: u32 | flags].
///
/// Returns parsed header with flags and payload slice.
fn parse_rans_stream_header(data: &[u8]) -> PzResult<RansStreamHeader<'_>> {
    if data.len() < 8 {
        return Err(PzError::InvalidInput);
    }
    let orig_len = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let comp_field = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let is_interleaved = (comp_field & RANS_INTERLEAVED_FLAG) != 0;
    let is_recoil = (comp_field & RANS_RECOIL_FLAG) != 0;
    let is_shared_stream = (comp_field & RANS_SHARED_STREAM_FLAG) != 0;
    let comp_len = (comp_field & RANS_COMP_LEN_MASK) as usize;
    if 8 + comp_len > data.len() {
        return Err(PzError::InvalidInput);
    }
    Ok(RansStreamHeader {
        orig_len,
        is_interleaved,
        is_recoil,
        is_shared_stream,
        payload: &data[8..8 + comp_len],
    })
}

/// Parse a Recoil payload: [meta_len:u32][rans_data][recoil_metadata].
///
/// Returns (rans_data, recoil_metadata).
fn parse_recoil_payload(payload: &[u8]) -> PzResult<(&[u8], crate::recoil::RecoilMetadata)> {
    if payload.len() < 4 {
        return Err(PzError::InvalidInput);
    }
    let meta_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if 4 + meta_len > payload.len() {
        return Err(PzError::InvalidInput);
    }
    let rans_data = &payload[4..payload.len() - meta_len];
    let recoil_meta_bytes = &payload[payload.len() - meta_len..];
    let recoil_meta = crate::recoil::RecoilMetadata::deserialize(recoil_meta_bytes)?;
    Ok((rans_data, recoil_meta))
}

/// Decode a Recoil payload using CPU parallel threads.
fn decode_recoil_payload(
    rans_data: &[u8],
    recoil_meta: &crate::recoil::RecoilMetadata,
    orig_len: usize,
) -> PzResult<Vec<u8>> {
    let num_threads = super::resolve_thread_count(0);
    crate::recoil::decode_recoil_parallel(rans_data, recoil_meta, orig_len, num_threads)
}

/// Decode a single rANS stream using the appropriate method (basic/interleaved/shared/recoil).
fn decode_rans_stream_cpu(data: &[u8]) -> PzResult<(Vec<u8>, usize)> {
    let hdr = parse_rans_stream_header(data)?;
    let decoded = if hdr.is_shared_stream {
        // Shared-stream decode (ryg_rans-style, single word pointer)
        rans::decode_shared_stream(hdr.payload, hdr.orig_len)?
    } else if hdr.is_interleaved && hdr.is_recoil {
        let (rans_data, recoil_meta) = parse_recoil_payload(hdr.payload)?;
        decode_recoil_payload(rans_data, &recoil_meta, hdr.orig_len)?
    } else if hdr.is_interleaved {
        rans::decode_interleaved(hdr.payload, hdr.orig_len)?
    } else {
        rans::decode(hdr.payload, hdr.orig_len)?
    };
    Ok((decoded, 8 + hdr.payload.len()))
}

/// rANS decoding stage: parse multi-stream container + rANS decode each stream.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32] [rans_data]
pub(crate) fn stage_rans_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, decode_rans_stream_cpu)?;

    block.metadata.pre_entropy_len = Some(pre_entropy_len);
    block.metadata.demux_meta = meta;
    block.streams = Some(streams);
    block.data.clear();
    Ok(block)
}

/// GPU-accelerated rANS decode stage that routes Recoil payloads to GPU.
///
/// Falls back to CPU for non-Recoil streams.
#[cfg(feature = "webgpu")]
pub(crate) fn stage_rans_decode_webgpu(
    mut block: StageBlock,
    engine: &crate::webgpu::WebGpuEngine,
) -> PzResult<StageBlock> {
    let (streams, pre_entropy_len, meta) = decode_multistream(&block.data, |data| {
        let hdr = parse_rans_stream_header(data)?;
        let decoded = if hdr.is_shared_stream {
            // Shared-stream: CPU decode (no GPU path yet)
            rans::decode_shared_stream(hdr.payload, hdr.orig_len)?
        } else if hdr.is_interleaved && hdr.is_recoil {
            let (rans_data, recoil_meta) = parse_recoil_payload(hdr.payload)?;
            engine.rans_decode_recoil_gpu(rans_data, &recoil_meta, hdr.orig_len)?
        } else if hdr.is_interleaved {
            rans::decode_interleaved(hdr.payload, hdr.orig_len)?
        } else {
            rans::decode(hdr.payload, hdr.orig_len)?
        };
        Ok((decoded, 8 + hdr.payload.len()))
    })?;

    block.metadata.pre_entropy_len = Some(pre_entropy_len);
    block.metadata.demux_meta = meta;
    block.streams = Some(streams);
    block.data.clear();
    Ok(block)
}

/// batched GPU dispatch with ring-buffered submit/readback overlap.
///
/// Per-stream framing: [orig_len: u32] [compressed_len: u32 | flags] [rans_data]
///
/// Streams below `rans_interleaved_min_bytes` fall back to CPU basic rANS.
/// The output wire format is identical to [`stage_rans_encode_with_options()`],
/// so the same decoder works for both CPU and GPU encoded data.
#[cfg(feature = "webgpu")]
pub(crate) fn stage_rans_encode_webgpu(
    mut block: StageBlock,
    engine: &crate::webgpu::WebGpuEngine,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    let streams = block.streams.take().ok_or(PzError::InvalidInput)?;
    let pre_entropy_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;
    let meta = &block.metadata.demux_meta;

    // Phase 1: batch-encode all GPU-eligible streams in one call.
    // The batched API uses a ring buffer internally to overlap GPU
    // compute with readback across streams.
    let min_bytes = options.rans_interleaved_min_bytes;
    let mut gpu_inputs: Vec<&[u8]> = Vec::new();
    let mut gpu_indices: Vec<usize> = Vec::new();
    for (i, stream) in streams.iter().enumerate() {
        if stream.len() >= min_bytes {
            gpu_inputs.push(stream);
            gpu_indices.push(i);
        }
    }

    let batch_results = if !gpu_inputs.is_empty() {
        engine.rans_encode_chunked_payload_gpu_batched(
            &gpu_inputs,
            options.rans_interleaved_states,
            rans::DEFAULT_SCALE_BITS,
            256,
        )?
    } else {
        Vec::new()
    };

    // Index batch results by original stream position.
    let mut gpu_results: Vec<Option<Vec<u8>>> = vec![None; streams.len()];
    for ((data, _used_chunked), &stream_idx) in batch_results.into_iter().zip(&gpu_indices) {
        gpu_results[stream_idx] = Some(data);
    }

    // Phase 2: assemble the multi-stream container.
    let mut output = Vec::new();
    output.push(streams.len() as u8);
    output.extend_from_slice(&(pre_entropy_len as u32).to_le_bytes());
    output.extend_from_slice(&(meta.len() as u16).to_le_bytes());
    output.extend_from_slice(meta);

    for (i, stream) in streams.iter().enumerate() {
        // GPU path: always interleaved (engine uses encode_interleaved_n
        // even when chunked encoding is not possible).
        // CPU path: basic rANS for small streams.
        let (rans_data, is_interleaved) = if let Some(data) = gpu_results[i].take() {
            (data, true)
        } else {
            (rans::encode(stream), false)
        };

        if rans_data.len() >= (1usize << 31) {
            return Err(PzError::InvalidInput);
        }

        let flagged_len = if is_interleaved {
            (rans_data.len() as u32) | RANS_INTERLEAVED_FLAG
        } else {
            rans_data.len() as u32
        };
        output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        output.extend_from_slice(&flagged_len.to_le_bytes());
        output.extend_from_slice(&rans_data);
    }

    block.data = output;
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
    let pre_entropy_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;

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
    let pre_entropy_len = block
        .metadata
        .pre_entropy_len
        .ok_or(PzError::InvalidInput)?;

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

/// FSE accuracy log for BWT-based pipelines.
///
/// Post-BWT data (after MTF+RLE) has a highly skewed distribution that
/// benefits from larger state tables. Accuracy log 10 (1024-entry table)
/// gives ~10pp better compression than the default 7 (128-entry table),
/// with negligible decode speed impact (table fits in L1 cache).
const BW_ACCURACY_LOG: u8 = 10;

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
    let fse_data = fse::encode_with_accuracy(&block.data, BW_ACCURACY_LOG);

    let mut output = Vec::new();
    output.extend_from_slice(&primary_index.to_le_bytes());
    output.extend_from_slice(&(rle_len as u32).to_le_bytes());
    output.extend_from_slice(&fse_data);

    block.data = output;
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
    let fse_data = fse::encode_with_accuracy(&block.data, BW_ACCURACY_LOG);

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

// ---------------------------------------------------------------------------
// Stage dispatch — maps (pipeline, stage_idx) to the appropriate function
// ---------------------------------------------------------------------------

/// Dispatch to the appropriate compression stage function.
pub(crate) fn run_compress_stage(
    pipeline: Pipeline,
    stage_idx: usize,
    block: StageBlock,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    match (pipeline, stage_idx) {
        (Pipeline::Bw, 0) => stage_bwt_encode(block, options),
        (Pipeline::Bw, 1) => stage_mtf_encode(block),
        (Pipeline::Bw, 2) => stage_rle_encode(block),
        (Pipeline::Bw, 3) => stage_fse_encode_bw(block),
        (Pipeline::Bbw, 0) => stage_bbwt_encode(block, options),
        (Pipeline::Bbw, 1) => stage_mtf_encode(block),
        (Pipeline::Bbw, 2) => stage_rle_encode(block),
        (Pipeline::Bbw, 3) => stage_fse_encode_bbw(block),
        (Pipeline::Lzf, 0) => stage_demux_compress(block, &LzDemuxer::LzSeq, options),
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
        (Pipeline::LzssR, 1) => stage_rans_encode_with_options(block, options),
        (Pipeline::LzSeqR, 0) => stage_demux_compress(block, &LzDemuxer::LzSeq, options),
        (Pipeline::LzSeqR, 1) => {
            #[cfg(feature = "webgpu")]
            {
                if let super::Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        return stage_rans_encode_webgpu(block, engine, options);
                    }
                }
            }
            stage_rans_encode_with_options(block, options)
        }
        (Pipeline::LzSeqH, 0) => stage_demux_compress(block, &LzDemuxer::LzSeq, options),
        (Pipeline::LzSeqH, 1) => stage_huffman_encode(block),
        (Pipeline::SortLz, 0) => stage_sortlz_compress(block),
        _ => Err(PzError::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// SortLZ pipeline stage (single-stage: does everything)
// ---------------------------------------------------------------------------

/// SortLZ single-stage compression: sort-based LZ77 + FSE.
pub(crate) fn stage_sortlz_compress(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = crate::sortlz::compress(&block.data, &crate::sortlz::SortLzConfig::default())?;
    Ok(block)
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
