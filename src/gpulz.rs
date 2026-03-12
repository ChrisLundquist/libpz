/// GpuLz codec: LzSeq + Huffman with sync-point parallel decode.
///
/// Standalone block-level compress/decompress functions for benchmarking
/// the GpuLz codec design before pipeline integration. The format stores
/// 6 LzSeq streams, each Huffman-encoded with periodic sync points that
/// enable per-segment GPU-parallel decode.
///
/// ## Wire format (per block)
///
/// ```text
/// BLOCK HEADER:
///   [meta_len: u16 LE]              // LzSeq metadata length
///   [meta: meta_len bytes]          // num_tokens(u32) + num_matches(u32)
///   [num_streams: u8]               // always 6
///
/// PER STREAM (×6):
///   [orig_len: u32 LE]              // decompressed stream byte count
///   [total_bits: u32 LE]            // Huffman bitstream total bits
///   [code_lengths: 256 bytes]       // canonical Huffman code lengths
///   [num_sync_points: u16 LE]       // count (includes sentinel)
///   [sync_points: (bit_offset: u32, symbol_index: u32) × num_sync_points]
///   [huffman_data: ceil(total_bits/8) bytes]
/// ```
use crate::huffman::{HuffmanTree, SyncPoint};
use crate::lz_token::{LzSeqEncoder, TokenEncoder};
use crate::{PzError, PzResult};

/// Default sync-point interval (symbols between checkpoints).
pub const DEFAULT_SYNC_INTERVAL: u32 = 1024;

/// Compress a block using the GpuLz codec.
///
/// Uses lazy LZ77 match finding → LzSeq tokenization → per-stream
/// Huffman encoding with sync points. No pipeline integration.
pub fn compress_block(input: &[u8]) -> PzResult<Vec<u8>> {
    compress_block_with_interval(input, DEFAULT_SYNC_INTERVAL)
}

/// Compress a block with a configurable sync-point interval.
pub fn compress_block_with_interval(input: &[u8], interval: u32) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    // Stage 0: LZ match finding + LzSeq tokenization.
    let matches = crate::lz77::compress_lazy_to_matches(input)?;
    let tokens = crate::lz_token::matches_to_tokens(&matches);
    let encoder = LzSeqEncoder::default();
    let encoded = encoder.encode(input, &tokens)?;

    // Stage 1: Huffman encode each stream with sync points.
    let mut output = Vec::new();

    // Block header: meta.
    let meta = &encoded.meta;
    output.extend_from_slice(&(meta.len() as u16).to_le_bytes());
    output.extend_from_slice(meta);
    output.push(encoded.streams.len() as u8);

    for stream in &encoded.streams {
        if stream.is_empty() {
            // Empty stream: orig_len=0, total_bits=0, code_lengths=zeros,
            // 1 sync point (sentinel), no huffman data.
            output.extend_from_slice(&0u32.to_le_bytes()); // orig_len
            output.extend_from_slice(&0u32.to_le_bytes()); // total_bits
            output.extend_from_slice(&[0u8; 256]); // code_lengths
            output.extend_from_slice(&1u16.to_le_bytes()); // num_sync_points (sentinel only)
            output.extend_from_slice(&0u32.to_le_bytes()); // sentinel bit_offset
            output.extend_from_slice(&0u32.to_le_bytes()); // sentinel symbol_index
            continue;
        }

        let mut tree = HuffmanTree::from_data(stream).ok_or(PzError::InvalidInput)?;

        // Canonicalize so that code_lengths alone can reconstruct the tree.
        tree.canonicalize();

        let result = tree.encode_with_sync_points(stream, interval)?;

        // orig_len
        output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        // total_bits
        output.extend_from_slice(&(result.total_bits as u32).to_le_bytes());
        // code_lengths (256 bytes)
        output.extend_from_slice(&tree.code_lengths());
        // num_sync_points
        output.extend_from_slice(&(result.sync_points.len() as u16).to_le_bytes());
        // sync_points
        for sp in &result.sync_points {
            output.extend_from_slice(&sp.bit_offset.to_le_bytes());
            output.extend_from_slice(&sp.symbol_index.to_le_bytes());
        }
        // huffman_data
        output.extend_from_slice(&result.data);
    }

    Ok(output)
}

/// Decompress a GpuLz block using per-segment tiled Huffman decode.
///
/// This is the CPU simulation of what the GPU decode pipeline will do:
/// each segment between sync points is decoded independently.
pub fn decompress_block(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 3 {
        return Err(PzError::InvalidInput);
    }

    let mut pos = 0;

    // Parse meta.
    let meta_len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
    pos += 2;
    if pos + meta_len >= payload.len() {
        return Err(PzError::InvalidInput);
    }
    let meta = &payload[pos..pos + meta_len];
    pos += meta_len;

    let num_streams = payload[pos] as usize;
    pos += 1;
    if num_streams != 6 {
        return Err(PzError::InvalidInput);
    }

    // Parse and decode each stream.
    let mut streams = Vec::with_capacity(num_streams);

    for _ in 0..num_streams {
        if pos + 8 + 256 + 2 > payload.len() {
            return Err(PzError::InvalidInput);
        }

        let stream_orig_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        let total_bits = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        // Code lengths.
        let code_lengths: [u8; 256] = payload[pos..pos + 256]
            .try_into()
            .map_err(|_| PzError::InvalidInput)?;
        pos += 256;

        let num_sync_points = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
        pos += 2;

        // Sync points.
        let sync_bytes = num_sync_points * 8;
        if pos + sync_bytes > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let mut sync_points = Vec::with_capacity(num_sync_points);
        for _ in 0..num_sync_points {
            let bit_offset = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            let symbol_index = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            sync_points.push(SyncPoint {
                bit_offset,
                symbol_index,
            });
        }

        if stream_orig_len == 0 {
            streams.push(Vec::new());
            continue;
        }

        // Huffman data.
        let huffman_bytes = total_bits.div_ceil(8);
        if pos + huffman_bytes > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let huffman_data = &payload[pos..pos + huffman_bytes];
        pos += huffman_bytes;

        // Rebuild Huffman tree from code lengths.
        let mut freq = crate::frequency::FrequencyTable::new();
        for (sym, &len) in code_lengths.iter().enumerate() {
            if len > 0 {
                // Assign a synthetic weight so the tree rebuilds with the same codes.
                // We need the tree to produce the same code_lengths. The simplest way
                // is to rebuild from the original data, but we don't have it.
                // Instead, use the decode_tiled path which only needs the decode table.
                freq.byte[sym] = 1;
                freq.used += 1;
            }
        }

        // Build tree from code lengths for decode. We need the decode table.
        // The HuffmanTree builds from frequencies which may not produce identical
        // code lengths. Instead, build the decode table directly from code_lengths.
        let tree = rebuild_tree_from_code_lengths(&code_lengths)?;

        let decoded = tree.decode_tiled(huffman_data, total_bits, &sync_points)?;
        if decoded.len() != stream_orig_len {
            return Err(PzError::InvalidInput);
        }
        streams.push(decoded);
    }

    // Stage 2: LzSeq reconstruct from 6 decoded streams.
    if meta.len() < 8 {
        return Err(PzError::InvalidInput);
    }
    let num_tokens = u32::from_le_bytes(meta[..4].try_into().unwrap());
    let num_matches = u32::from_le_bytes(meta[4..8].try_into().unwrap());

    crate::lzseq::decode(
        &streams[0], // flags
        &streams[1], // literals
        &streams[2], // offset_codes
        &streams[3], // offset_extra
        &streams[4], // length_codes
        &streams[5], // length_extra
        num_tokens,
        num_matches,
        orig_len,
    )
}

/// Timing breakdown for GPU decompress (returned by `decompress_block_gpu_timed`).
#[cfg(feature = "webgpu")]
#[derive(Debug, Clone)]
pub struct GpuDecompressTimings {
    /// Parse wire format + build decode LUTs (µs).
    pub parse_us: u64,
    /// GPU Huffman decode: buffer creation + dispatch + readback (µs).
    pub gpu_huffman_us: u64,
    /// GPU buffer creation (µs) — subset of gpu_huffman_us.
    pub gpu_buffers_us: u64,
    /// GPU command record + submit (µs) — subset of gpu_huffman_us.
    pub gpu_submit_us: u64,
    /// GPU poll + readback (µs) — subset of gpu_huffman_us.
    pub gpu_readback_us: u64,
    /// CPU LzSeq reconstruct (µs).
    pub lzseq_us: u64,
    /// Total wall time (µs).
    pub total_us: u64,
}

/// Decompress a GpuLz block using GPU-accelerated Huffman decode.
///
/// Parses the same wire format as [`decompress_block()`], but uses the GPU
/// for sync-point parallel Huffman decode of all 6 streams in a single
/// batched submission. The LzSeq reconstruct phase still runs on the CPU.
#[cfg(feature = "webgpu")]
pub fn decompress_block_gpu(
    engine: &crate::webgpu::WebGpuEngine,
    payload: &[u8],
    orig_len: usize,
) -> PzResult<Vec<u8>> {
    decompress_block_gpu_timed(engine, payload, orig_len).map(|(data, _)| data)
}

/// Same as [`decompress_block_gpu`] but also returns timing breakdown.
#[cfg(feature = "webgpu")]
pub fn decompress_block_gpu_timed(
    engine: &crate::webgpu::WebGpuEngine,
    payload: &[u8],
    orig_len: usize,
) -> PzResult<(Vec<u8>, GpuDecompressTimings)> {
    use crate::webgpu::HuffmanDecodeStream;
    use std::time::Instant;

    let t_start = Instant::now();

    if payload.len() < 3 {
        return Err(PzError::InvalidInput);
    }

    let mut pos = 0;

    // Parse meta (identical to CPU decompress_block).
    let meta_len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
    pos += 2;
    if pos + meta_len >= payload.len() {
        return Err(PzError::InvalidInput);
    }
    let meta = &payload[pos..pos + meta_len];
    pos += meta_len;

    let num_streams = payload[pos] as usize;
    pos += 1;
    if num_streams != 6 {
        return Err(PzError::InvalidInput);
    }

    // Parse all stream headers and build GPU decode descriptors.
    // Track ranges in payload for each stream's huffman data.
    struct StreamInfo {
        orig_len: usize,
        huff_start: usize,
        huff_end: usize,
        sync_points: Vec<SyncPoint>,
        decode_lut: Box<[u32; 4096]>,
    }

    let mut stream_infos: Vec<Option<StreamInfo>> = Vec::with_capacity(num_streams);

    for _ in 0..num_streams {
        if pos + 8 + 256 + 2 > payload.len() {
            return Err(PzError::InvalidInput);
        }

        let stream_orig_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        let total_bits = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        let code_lengths: [u8; 256] = payload[pos..pos + 256]
            .try_into()
            .map_err(|_| PzError::InvalidInput)?;
        pos += 256;

        let num_sync_points = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
        pos += 2;

        let sync_bytes = num_sync_points * 8;
        if pos + sync_bytes > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let mut sync_points = Vec::with_capacity(num_sync_points);
        for _ in 0..num_sync_points {
            let bit_offset = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            let symbol_index = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            sync_points.push(SyncPoint {
                bit_offset,
                symbol_index,
            });
        }

        if stream_orig_len == 0 {
            stream_infos.push(None);
            continue;
        }

        let huffman_bytes = total_bits.div_ceil(8);
        if pos + huffman_bytes > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let huff_start = pos;
        pos += huffman_bytes;

        // Build decode LUT from code lengths.
        let tree = rebuild_tree_from_code_lengths(&code_lengths)?;
        let lut = tree.build_gpu_decode_lut();
        let lut_box: Box<[u32; 4096]> = lut
            .into_boxed_slice()
            .try_into()
            .map_err(|_| PzError::InvalidInput)?;

        stream_infos.push(Some(StreamInfo {
            orig_len: stream_orig_len,
            huff_start,
            huff_end: pos,
            sync_points,
            decode_lut: lut_box,
        }));
    }

    let t_parse = Instant::now();
    let parse_us = t_parse.duration_since(t_start).as_micros() as u64;

    // Build batched decode descriptors referencing payload slices.
    let decode_streams: Vec<HuffmanDecodeStream> = stream_infos
        .into_iter()
        .map(|info| match info {
            Some(si) => HuffmanDecodeStream {
                huffman_data: &payload[si.huff_start..si.huff_end],
                decode_lut: si.decode_lut,
                sync_points: si.sync_points,
                output_len: si.orig_len,
            },
            None => HuffmanDecodeStream {
                huffman_data: &[],
                decode_lut: Box::new([0u32; 4096]),
                sync_points: Vec::new(),
                output_len: 0,
            },
        })
        .collect();

    // Single batched GPU dispatch for all 6 streams.
    let (streams, gpu_timings) = engine.huffman_decode_gpu_batched_timed(&decode_streams)?;
    let t_gpu = Instant::now();
    let gpu_huffman_us = t_gpu.duration_since(t_parse).as_micros() as u64;

    if streams.len() != 6 {
        return Err(PzError::InvalidInput);
    }

    // Stage 2: LzSeq reconstruct (CPU).
    if meta.len() < 8 {
        return Err(PzError::InvalidInput);
    }
    let num_tokens = u32::from_le_bytes(meta[..4].try_into().unwrap());
    let num_matches = u32::from_le_bytes(meta[4..8].try_into().unwrap());

    let result = crate::lzseq::decode(
        &streams[0],
        &streams[1],
        &streams[2],
        &streams[3],
        &streams[4],
        &streams[5],
        num_tokens,
        num_matches,
        orig_len,
    )?;
    let t_lzseq = Instant::now();
    let lzseq_us = t_lzseq.duration_since(t_gpu).as_micros() as u64;
    let total_us = t_lzseq.duration_since(t_start).as_micros() as u64;

    Ok((
        result,
        GpuDecompressTimings {
            parse_us,
            gpu_huffman_us,
            gpu_buffers_us: gpu_timings.buffer_create_us,
            gpu_submit_us: gpu_timings.submit_us,
            gpu_readback_us: gpu_timings.readback_us,
            lzseq_us,
            total_us,
        },
    ))
}

/// Rebuild a HuffmanTree from code lengths for decoding.
///
/// Constructs canonical Huffman codes from the lengths and builds
/// the fast 12-bit decode table. This matches how the GPU would
/// reconstruct the decode LUT from the wire format.
fn rebuild_tree_from_code_lengths(code_lengths: &[u8; 256]) -> PzResult<HuffmanTree> {
    // Build canonical codes from lengths: assign codes in order of
    // (length, symbol) using the standard canonical Huffman algorithm.
    let mut lookup = [(0u32, 0u8); 256];

    // Count codes per length.
    let mut bl_count = [0u32; 16];
    for &len in code_lengths.iter() {
        if len > 0 && (len as usize) < bl_count.len() {
            bl_count[len as usize] += 1;
        }
    }

    // Compute first code for each length.
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols.
    for sym in 0..256 {
        let len = code_lengths[sym];
        if len > 0 && (len as usize) < next_code.len() {
            lookup[sym] = (next_code[len as usize], len);
            next_code[len as usize] += 1;
        }
    }

    // Build decode table from the lookup.
    let decode_table = HuffmanTree::build_decode_table_from_lookup(&lookup);

    // Count leaves.
    let leaf_count = code_lengths.iter().filter(|&&l| l > 0).count() as u32;

    // Build minimal tree structure (we only need decode table, not the full tree).
    // Use a single-node tree as placeholder; decode_tiled uses decode_table only.
    let mut nodes = Vec::with_capacity(257);
    for i in 0..256u16 {
        nodes.push(crate::huffman::HuffmanNode {
            weight: 0,
            value: i as u8,
            codeword: lookup[i as usize].0,
            code_bits: lookup[i as usize].1,
            left: None,
            right: None,
        });
    }
    // Root node (minimal).
    nodes.push(crate::huffman::HuffmanNode {
        weight: 0,
        value: 0,
        codeword: 0,
        code_bits: 0,
        left: Some(0),
        right: if leaf_count > 1 { Some(1) } else { None },
    });

    Ok(HuffmanTree::from_parts(
        nodes,
        Some(256),
        lookup,
        leaf_count,
        decode_table,
    ))
}

// ---------------------------------------------------------------------------
// Multi-block GPU decompress with parallel LzSeq
// ---------------------------------------------------------------------------

/// Parsed block ready for GPU Huffman decode + CPU LzSeq.
#[cfg(feature = "webgpu")]
struct ParsedBlock {
    num_tokens: u32,
    num_matches: u32,
    orig_len: usize,
    /// Per-stream: (orig_len, huff_data_range, sync_points, decode_lut).
    streams: Vec<ParsedStream>,
}

#[cfg(feature = "webgpu")]
struct ParsedStream {
    orig_len: usize,
    huff_start: usize,
    huff_end: usize,
    sync_points: Vec<SyncPoint>,
    decode_lut: Box<[u32; 4096]>,
}

/// Parse a compressed block without decoding — extract all Huffman stream
/// metadata for batched GPU dispatch.
#[cfg(feature = "webgpu")]
fn parse_block(payload: &[u8], orig_len: usize) -> PzResult<ParsedBlock> {
    if payload.len() < 3 {
        return Err(PzError::InvalidInput);
    }

    let mut pos = 0;
    let meta_len = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
    pos += 2;
    if pos + meta_len >= payload.len() {
        return Err(PzError::InvalidInput);
    }
    let meta = &payload[pos..pos + meta_len];
    pos += meta_len;

    let num_streams = payload[pos] as usize;
    pos += 1;
    if num_streams != 6 {
        return Err(PzError::InvalidInput);
    }

    if meta.len() < 8 {
        return Err(PzError::InvalidInput);
    }
    let num_tokens = u32::from_le_bytes(meta[..4].try_into().unwrap());
    let num_matches = u32::from_le_bytes(meta[4..8].try_into().unwrap());

    let mut streams = Vec::with_capacity(6);

    for _ in 0..num_streams {
        if pos + 8 + 256 + 2 > payload.len() {
            return Err(PzError::InvalidInput);
        }

        let stream_orig_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        let total_bits = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        let code_lengths: [u8; 256] = payload[pos..pos + 256]
            .try_into()
            .map_err(|_| PzError::InvalidInput)?;
        pos += 256;

        let num_sync_points = u16::from_le_bytes([payload[pos], payload[pos + 1]]) as usize;
        pos += 2;

        let sync_bytes = num_sync_points * 8;
        if pos + sync_bytes > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let mut sync_points = Vec::with_capacity(num_sync_points);
        for _ in 0..num_sync_points {
            let bit_offset = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            let symbol_index = u32::from_le_bytes([
                payload[pos],
                payload[pos + 1],
                payload[pos + 2],
                payload[pos + 3],
            ]);
            pos += 4;
            sync_points.push(SyncPoint {
                bit_offset,
                symbol_index,
            });
        }

        if stream_orig_len == 0 {
            streams.push(ParsedStream {
                orig_len: 0,
                huff_start: pos,
                huff_end: pos,
                sync_points: Vec::new(),
                decode_lut: Box::new([0u32; 4096]),
            });
            continue;
        }

        let huffman_bytes = total_bits.div_ceil(8);
        if pos + huffman_bytes > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let huff_start = pos;
        pos += huffman_bytes;

        let tree = rebuild_tree_from_code_lengths(&code_lengths)?;
        let lut = tree.build_gpu_decode_lut();
        let lut_box: Box<[u32; 4096]> = lut
            .into_boxed_slice()
            .try_into()
            .map_err(|_| PzError::InvalidInput)?;

        streams.push(ParsedStream {
            orig_len: stream_orig_len,
            huff_start,
            huff_end: pos,
            sync_points,
            decode_lut: lut_box,
        });
    }

    Ok(ParsedBlock {
        num_tokens,
        num_matches,
        orig_len,
        streams,
    })
}

/// Timing breakdown for multi-block GPU decompress.
#[cfg(feature = "webgpu")]
#[derive(Debug, Clone)]
pub struct MultiBlockTimings {
    /// Parse all blocks (µs).
    pub parse_us: u64,
    /// GPU Huffman decode for all blocks combined (µs).
    pub gpu_huffman_us: u64,
    /// Parallel CPU LzSeq for all blocks (µs, wall time).
    pub lzseq_us: u64,
    /// Total wall time (µs).
    pub total_us: u64,
    /// Number of blocks.
    pub num_blocks: usize,
    /// Number of GPU streams dispatched.
    pub num_gpu_streams: usize,
}

/// Decompress multiple GpuLz blocks with batched GPU Huffman + parallel CPU LzSeq.
///
/// All Huffman decodes for all blocks are batched into a single GPU submission.
/// Then LzSeq reconstruct for each block runs in parallel on separate threads.
#[cfg(feature = "webgpu")]
pub fn decompress_blocks_gpu(
    engine: &crate::webgpu::WebGpuEngine,
    blocks: &[(&[u8], usize)], // (payload, orig_len) per block
) -> PzResult<(Vec<Vec<u8>>, MultiBlockTimings)> {
    use crate::webgpu::HuffmanDecodeStream;
    use std::time::Instant;

    let t_start = Instant::now();

    // Phase 1: Parse all blocks and extract Huffman stream descriptors.
    let mut parsed: Vec<ParsedBlock> = Vec::with_capacity(blocks.len());
    for &(payload, orig_len) in blocks {
        parsed.push(parse_block(payload, orig_len)?);
    }

    let t_parse = Instant::now();
    let parse_us = t_parse.duration_since(t_start).as_micros() as u64;

    // Phase 2: Build one big batch of all Huffman decode streams across all blocks.
    // Track which decoded streams belong to which block.
    let mut all_decode_streams = Vec::new();
    let mut block_ranges: Vec<(usize, usize)> = Vec::with_capacity(blocks.len());

    for (parsed_block, &(payload, _)) in parsed.iter().zip(blocks.iter()) {
        let start = all_decode_streams.len();
        for ps in &parsed_block.streams {
            if ps.orig_len == 0 {
                all_decode_streams.push(HuffmanDecodeStream {
                    huffman_data: &[],
                    decode_lut: Box::new([0u32; 4096]),
                    sync_points: Vec::new(),
                    output_len: 0,
                });
            } else {
                all_decode_streams.push(HuffmanDecodeStream {
                    huffman_data: &payload[ps.huff_start..ps.huff_end],
                    decode_lut: ps.decode_lut.clone(),
                    sync_points: ps.sync_points.clone(),
                    output_len: ps.orig_len,
                });
            }
        }
        let end = all_decode_streams.len();
        block_ranges.push((start, end));
    }

    let num_gpu_streams = all_decode_streams.len();

    // Single GPU submission for all streams across all blocks.
    let all_decoded = engine.huffman_decode_gpu_batched(&all_decode_streams)?;

    let t_gpu = Instant::now();
    let gpu_huffman_us = t_gpu.duration_since(t_parse).as_micros() as u64;

    // Phase 3: Parallel CPU LzSeq reconstruct — one thread per block.
    let results: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(blocks.len());

        for (bi, pb) in parsed.iter().enumerate() {
            let (start, end) = block_ranges[bi];
            let block_streams = &all_decoded[start..end];
            let num_tokens = pb.num_tokens;
            let num_matches = pb.num_matches;
            let orig_len = pb.orig_len;

            handles.push(scope.spawn(move || {
                if block_streams.len() != 6 {
                    return Err(PzError::InvalidInput);
                }
                crate::lzseq::decode(
                    &block_streams[0],
                    &block_streams[1],
                    &block_streams[2],
                    &block_streams[3],
                    &block_streams[4],
                    &block_streams[5],
                    num_tokens,
                    num_matches,
                    orig_len,
                )
            }));
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let t_lzseq = Instant::now();
    let lzseq_us = t_lzseq.duration_since(t_gpu).as_micros() as u64;
    let total_us = t_lzseq.duration_since(t_start).as_micros() as u64;

    // Collect results, propagating first error.
    let mut output = Vec::with_capacity(blocks.len());
    for r in results {
        output.push(r?);
    }

    Ok((
        output,
        MultiBlockTimings {
            parse_us,
            gpu_huffman_us,
            lzseq_us,
            total_us,
            num_blocks: blocks.len(),
            num_gpu_streams,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple() {
        let input = b"hello world! this is a test of the gpulz codec. hello world!";
        let compressed = compress_block(input).unwrap();
        let decompressed = decompress_block(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_repeated() {
        let input = b"abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabc";
        let compressed = compress_block(input).unwrap();
        let decompressed = decompress_block(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_longer() {
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
        }
        let compressed = compress_block(&input).unwrap();
        let decompressed = decompress_block(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_sync_intervals() {
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(b"testing various sync intervals with gpulz codec! ");
        }
        for &interval in &[512, 1024, 2048] {
            let compressed = compress_block_with_interval(&input, interval).unwrap();
            let decompressed = decompress_block(&compressed, input.len()).unwrap();
            assert_eq!(decompressed, input, "failed at interval={interval}");
        }
    }

    #[test]
    fn test_roundtrip_high_entropy() {
        // Simulate data with high entropy (like binary/compressed sections of a tar).
        // Use a pseudo-random sequence with enough structure for LZ to find some matches.
        let mut input = Vec::with_capacity(131072);
        let mut state = 12345u64;
        for _ in 0..131072 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let byte = (state >> 33) as u8;
            input.push(byte);
        }
        // Inject some repeated patterns so LZ finds matches.
        let pattern: Vec<u8> = input[..256].to_vec();
        for i in (0..input.len()).step_by(4096) {
            let end = (i + 256).min(input.len());
            let copy_len = end - i;
            input[i..end].copy_from_slice(&pattern[..copy_len]);
        }
        let compressed = compress_block(&input).unwrap();
        let decompressed = decompress_block(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_all_256_bytes() {
        // Every possible byte value in a 128KB block.
        let mut input = Vec::with_capacity(131072);
        for _ in 0..512 {
            for b in 0..=255u8 {
                input.push(b);
            }
        }
        let compressed = compress_block(&input).unwrap();
        let decompressed = decompress_block(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_all_same_byte() {
        let input = vec![0x42u8; 1000];
        let compressed = compress_block(&input).unwrap();
        let decompressed = decompress_block(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_128k() {
        // Use a realistic 128KB block size with varied data.
        let mut input = Vec::new();
        let patterns: &[&[u8]] = &[
            b"The quick brown fox jumps over the lazy dog. ",
            b"Pack my box with five dozen liquor jugs. ",
            b"How vexingly quick daft zebras jump! ",
        ];
        let mut pi = 0;
        while input.len() < 131072 {
            input.extend_from_slice(patterns[pi % patterns.len()]);
            pi += 1;
        }
        input.truncate(131072);
        let compressed = compress_block(&input).unwrap();
        let decompressed = decompress_block(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_multi_block_sliced() {
        // Simulate slicing a larger dataset into 128KB blocks.
        let mut big_data = Vec::new();
        // Mix in some binary-ish data too.
        for i in 0u32..200_000 {
            big_data.push((i % 256) as u8);
            if i % 37 == 0 {
                big_data.extend_from_slice(b"pattern ");
            }
        }
        let block_size = 131072;
        let num_blocks = big_data.len() / block_size;
        for i in 0..num_blocks {
            let block = &big_data[i * block_size..(i + 1) * block_size];
            let compressed = compress_block(block).unwrap();
            let decompressed = decompress_block(&compressed, block_size).unwrap();
            assert_eq!(decompressed, block, "block {i} round-trip failed");
        }
    }
}
