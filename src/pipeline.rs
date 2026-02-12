/// Compression pipeline orchestrator.
///
/// Chains algorithm stages together to form complete compression pipelines.
/// Each pipeline defines a sequence of transforms and entropy coding
/// stages that are applied in order for compression, and in reverse for
/// decompression.
///
/// **Supported pipelines:**
///
/// | Pipeline      | Stages                           | Similar to |
/// |---------------|----------------------------------|------------|
/// | `Deflate`     | LZ77 → Huffman                   | gzip       |
/// | `Bw`          | BWT → MTF → RLE → FSE            | bzip2      |
/// | `Lzr`         | LZ77 → rANS                      | fast ANS   |
/// | `Lzf`         | LZ77 → FSE                       | zstd-like  |
///
/// **Container format (V2, multi-block):**
/// Each compressed stream starts with a header:
/// - Magic bytes: `PZ` (2 bytes)
/// - Version: 2 (1 byte)
/// - Pipeline ID: 0=Deflate, 1=Bw, 3=Lzr, 4=Lzf (1 byte)
/// - Original length: u32 little-endian (4 bytes)
/// - num_blocks: u32 little-endian (4 bytes)
/// - Block table: \[compressed_len: u32, original_len: u32\] * num_blocks
/// - Block data: concatenated compressed block bytes
use crate::bwt;
use crate::fse;
use crate::huffman::HuffmanTree;
use crate::lz77;
use crate::mtf;
use crate::rans;
use crate::rle;
use crate::{PzError, PzResult};

/// Compute backend selection for pipeline stages.
///
/// Each pipeline stage can run on different backends depending on
/// hardware availability and input characteristics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    /// Single-threaded CPU reference implementation (always available).
    #[default]
    Cpu,
    /// OpenCL GPU backend (requires `opencl` feature and a GPU device).
    #[cfg(feature = "opencl")]
    OpenCl,
    /// WebGPU backend via wgpu (requires `webgpu` feature).
    #[cfg(feature = "webgpu")]
    WebGpu,
}

/// LZ77 match selection strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParseStrategy {
    /// Auto: choose the best strategy based on backend and input size.
    /// - GPU backend + large input: GPU hash-table kernel
    /// - CPU backend: lazy matching (best speed + compression)
    /// - Equivalent to Lazy on CPU, HashTable on GPU.
    #[default]
    Auto,
    /// Lazy: check if the next position has a longer match (gzip-style).
    Lazy,
    /// Optimal: backward DP to minimize total encoding cost.
    Optimal,
}

/// Default block size for multi-threaded compression (256KB).
const DEFAULT_BLOCK_SIZE: usize = 256 * 1024;

/// Options controlling pipeline compression behavior.
#[derive(Debug, Clone)]
pub struct CompressOptions {
    /// Which backend to use for GPU-amenable stages (LZ77 match finding,
    /// BWT suffix array, Huffman encoding).
    pub backend: Backend,
    /// Number of threads to use. 0 = auto (use all available cores),
    /// 1 = single-threaded.
    pub threads: usize,
    /// Block size for multi-threaded compression. Input is split into
    /// blocks of this size, each compressed independently.
    pub block_size: usize,
    /// LZ77 match selection strategy (greedy, lazy, or optimal).
    pub parse_strategy: ParseStrategy,
    /// OpenCL engine handle, required when `backend` is `Backend::OpenCl`.
    #[cfg(feature = "opencl")]
    pub opencl_engine: Option<std::sync::Arc<crate::opencl::OpenClEngine>>,
    /// WebGPU engine handle, required when `backend` is `Backend::WebGpu`.
    #[cfg(feature = "webgpu")]
    pub webgpu_engine: Option<std::sync::Arc<crate::webgpu::WebGpuEngine>>,
}

impl Default for CompressOptions {
    fn default() -> Self {
        CompressOptions {
            backend: Backend::Cpu,
            threads: 0,
            block_size: DEFAULT_BLOCK_SIZE,
            parse_strategy: ParseStrategy::Auto,
            #[cfg(feature = "opencl")]
            opencl_engine: None,
            #[cfg(feature = "webgpu")]
            webgpu_engine: None,
        }
    }
}

/// Magic bytes for the libpz container format.
const MAGIC: [u8; 2] = [b'P', b'Z'];
/// Format version for multi-block streams.
const VERSION: u8 = 2;

/// Minimum header size: magic(2) + version(1) + pipeline(1) + orig_len(4) = 8
const MIN_HEADER_SIZE: usize = 8;

/// Compression pipeline types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pipeline {
    /// LZ77 + Huffman (gzip-like)
    Deflate = 0,
    /// BWT + MTF + RLE + FSE (bzip2-like)
    Bw = 1,
    /// LZ77 + rANS (fast entropy coding, SIMD/GPU friendly)
    Lzr = 3,
    /// LZ77 + FSE (finite state entropy, zstd-style)
    Lzf = 4,
}

impl TryFrom<u8> for Pipeline {
    type Error = PzError;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Deflate),
            1 => Ok(Self::Bw),
            3 => Ok(Self::Lzr),
            4 => Ok(Self::Lzf),
            _ => Err(PzError::Unsupported),
        }
    }
}

/// Compress data using the specified pipeline (CPU backend).
///
/// Returns a self-contained compressed stream including the header.
pub fn compress(input: &[u8], pipeline: Pipeline) -> PzResult<Vec<u8>> {
    compress_with_options(input, pipeline, &CompressOptions::default())
}

/// Compress data using the specified pipeline and backend options.
///
/// When `options.threads` is 0 (auto) or > 1, input larger than one block
/// is split into blocks and compressed in parallel using scoped threads.
/// When `threads` is 1 or the input fits in a single block, a single block
/// is compressed without thread overhead.
///
/// The output always uses the multi-block container format (V2) with a
/// block table, even for single-block streams.
///
/// When `options.backend` is `Backend::OpenCl` and an engine is provided,
/// GPU-amenable stages (e.g., LZ77 match finding) run on the GPU.
/// Other stages and decompression always use the CPU.
pub fn compress_with_options(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let num_threads = resolve_thread_count(options.threads);
    let block_size = options.block_size;

    // Use single-block path if single-threaded or input fits in one block
    if num_threads <= 1 || input.len() <= block_size {
        let block_data = compress_block(input, pipeline, options)?;
        let mut output = Vec::new();
        write_header(&mut output, pipeline, input.len());
        output.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        output.extend_from_slice(&(block_data.len() as u32).to_le_bytes()); // compressed_len
        output.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original_len
        output.extend_from_slice(&block_data);
        return Ok(output);
    }

    // Multi-block compression: choose between pipeline-parallel and block-parallel.
    // Pipeline parallelism (one thread per stage) when enough blocks to fill the pipeline.
    // Block parallelism (one thread per block) otherwise.
    let num_blocks = input.len().div_ceil(block_size);
    let stage_count = pipeline_stage_count(pipeline);

    if num_blocks >= stage_count {
        compress_pipeline_parallel(input, pipeline, options)
    } else {
        compress_parallel(input, pipeline, options, num_threads)
    }
}

/// Decompress data produced by `compress`.
///
/// Reads the header to determine the pipeline, then applies the
/// inverse stages. Decompresses blocks in parallel when applicable.
pub fn decompress(input: &[u8]) -> PzResult<Vec<u8>> {
    decompress_with_threads(input, 0)
}

/// Decompress data with an explicit thread count.
///
/// `threads`: 0 = auto, 1 = single-threaded, N = use up to N threads.
pub fn decompress_with_threads(input: &[u8], threads: usize) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if input.len() < MIN_HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    // Parse header
    if input[0] != MAGIC[0] || input[1] != MAGIC[1] {
        return Err(PzError::InvalidInput);
    }

    let version = input[2];
    if version != VERSION {
        return Err(PzError::Unsupported);
    }

    let pipeline = Pipeline::try_from(input[3])?;
    let orig_len = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;

    if orig_len == 0 {
        return Ok(Vec::new());
    }

    let payload = &input[MIN_HEADER_SIZE..];

    if payload.len() < 4 {
        return Err(PzError::InvalidInput);
    }
    let num_blocks = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if num_blocks == 0 {
        return Err(PzError::InvalidInput);
    }

    let num_threads = resolve_thread_count(threads);
    let stage_count = pipeline_stage_count(pipeline);

    if num_threads > 1 && num_blocks >= stage_count {
        return decompress_pipeline_parallel(payload, pipeline, orig_len, num_blocks);
    }
    decompress_parallel(payload, pipeline, orig_len, num_blocks, threads)
}

/// Write the container header to output.
fn write_header(output: &mut Vec<u8>, pipeline: Pipeline, orig_len: usize) {
    output.extend_from_slice(&MAGIC);
    output.push(VERSION);
    output.push(pipeline as u8);
    output.extend_from_slice(&(orig_len as u32).to_le_bytes());
}

/// Resolve thread count: 0 = auto (available_parallelism), otherwise use the given value.
fn resolve_thread_count(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

/// Compress a single block using the appropriate pipeline (no container header).
fn compress_block(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    match pipeline {
        Pipeline::Deflate => compress_block_deflate(input, options),
        Pipeline::Bw => compress_block_bw(input, options),
        Pipeline::Lzr => compress_block_lzr(input, options),
        Pipeline::Lzf => compress_block_lzf(input, options),
    }
}

/// Decompress a single block using the appropriate pipeline (no container header).
fn decompress_block(payload: &[u8], pipeline: Pipeline, orig_len: usize) -> PzResult<Vec<u8>> {
    match pipeline {
        Pipeline::Deflate => decompress_block_deflate(payload, orig_len),
        Pipeline::Bw => decompress_block_bw(payload, orig_len),
        Pipeline::Lzr => decompress_block_lzr(payload, orig_len),
        Pipeline::Lzf => decompress_block_lzf(payload, orig_len),
    }
}

/// Size of a block table entry: compressed_len(4) + original_len(4) = 8 bytes.
const BLOCK_HEADER_SIZE: usize = 8;

/// Multi-block parallel compression.
///
/// Format after the 8-byte container header:
/// - num_blocks: u32 LE
/// - block_table: [compressed_len: u32 LE, original_len: u32 LE] * num_blocks
/// - block_data: concatenated compressed block bytes
fn compress_parallel(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
    num_threads: usize,
) -> PzResult<Vec<u8>> {
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

/// Multi-block parallel decompression.
fn decompress_parallel(
    payload: &[u8],
    pipeline: Pipeline,
    orig_len: usize,
    num_blocks: usize,
    threads: usize,
) -> PzResult<Vec<u8>> {
    let num_threads = resolve_thread_count(threads);

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
            handles
                .push(scope.spawn(move || decompress_block(comp_data, pipeline, orig_block_len)));
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

// --- Pipeline stage parallelism types and functions ---

/// A block moving through the pipeline stages.
///
/// Each stage transforms `data` in place (replacing it with output)
/// and may attach metadata. The `block_index` preserves ordering
/// for final reassembly.
struct StageBlock {
    /// Monotonically increasing index for ordered reassembly.
    /// Not read at runtime (FIFO channels preserve order), but useful for debugging.
    #[allow(dead_code)]
    block_index: usize,
    /// The original uncompressed length of this block.
    original_len: usize,
    /// The current data payload. Each stage replaces this with its output.
    data: Vec<u8>,
    /// Optional multi-stream payload. When `Some`, downstream stages encode/decode
    /// each stream independently instead of using `data`.
    streams: Option<Vec<Vec<u8>>>,
    /// Accumulated metadata from prior stages.
    metadata: StageMetadata,
}

/// Metadata accumulated across pipeline stages.
#[derive(Default)]
struct StageMetadata {
    /// BWT primary index (Bw pipeline, set by BWT stage).
    bwt_primary_index: Option<u32>,
    /// Length of data before entropy coding.
    /// Bw: RLE output length.
    pre_entropy_len: Option<usize>,
    /// Deflate: LZ77 output length.
    lz_len: Option<usize>,
}

/// Number of compression stages in a pipeline.
fn pipeline_stage_count(pipeline: Pipeline) -> usize {
    match pipeline {
        Pipeline::Deflate => 2,
        Pipeline::Bw => 4,
        Pipeline::Lzr => 2,
        Pipeline::Lzf => 2,
    }
}

// --- Compression stage functions ---

/// LZ-pipeline stage 0: LZ77 compression with multi-stream deinterleaving.
///
/// After LZ77 compression, the flat match stream (5 bytes per match:
/// offset_lo, offset_hi, length_lo, length_hi, next) is split into 3
/// independent streams for better entropy coding:
///   - Stream 0: offsets (2 bytes per match, little-endian u16)
///   - Stream 1: lengths (2 bytes per match, little-endian u16)
///   - Stream 2: literals (1 byte per match, the `next` field)
fn stage_lz77_compress(mut block: StageBlock, options: &CompressOptions) -> PzResult<StageBlock> {
    let lz_data = lz77_compress_with_backend(&block.data, options)?;
    let lz_len = lz_data.len();
    block.metadata.lz_len = Some(lz_len);

    // Deinterleave into 3 streams
    let match_size = lz77::Match::SERIALIZED_SIZE; // 5
    let num_matches = lz_len / match_size;

    let mut offsets = Vec::with_capacity(num_matches * 2);
    let mut lengths = Vec::with_capacity(num_matches * 2);
    let mut literals = Vec::with_capacity(num_matches);

    for i in 0..num_matches {
        let base = i * match_size;
        offsets.push(lz_data[base]);
        offsets.push(lz_data[base + 1]);
        lengths.push(lz_data[base + 2]);
        lengths.push(lz_data[base + 3]);
        literals.push(lz_data[base + 4]);
    }

    block.streams = Some(vec![offsets, lengths, literals]);
    block.data.clear();
    Ok(block)
}

/// Deflate stage 1: Huffman encoding + serialization.
///
/// Multi-stream format (when `block.streams` is `Some`):
///   [0u32: sentinel] [num_streams: u8] [lz_len: u32]
///   For each stream:
///     [stream_data_len: u32] [total_bits: u32] [freq_table: 256×u32] [huffman_data]
///
/// Single-stream format (legacy, when `block.streams` is `None`):
///   [lz_len: u32] [total_bits: u32] [freq_table: 256×u32] [huffman_data]
fn stage_huffman_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let lz_len = block.metadata.lz_len.unwrap();

    if let Some(streams) = block.streams.take() {
        // Multi-stream encoding: encode each stream independently
        let mut output = Vec::new();
        // Sentinel: 0u32 distinguishes from single-stream (lz_len > 0)
        output.extend_from_slice(&0u32.to_le_bytes());
        output.push(streams.len() as u8);
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());

        for stream in &streams {
            let tree = HuffmanTree::from_data(stream).ok_or(PzError::InvalidInput)?;
            let (huffman_data, total_bits) = tree.encode(stream)?;
            let freq_table = tree.serialize_frequencies();

            output.extend_from_slice(&(huffman_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&(total_bits as u32).to_le_bytes());
            for &freq in &freq_table {
                output.extend_from_slice(&freq.to_le_bytes());
            }
            output.extend_from_slice(&huffman_data);
        }

        block.data = output;
    } else {
        // Single-stream (legacy) path
        let tree = HuffmanTree::from_data(&block.data).ok_or(PzError::InvalidInput)?;
        let (huffman_data, total_bits) = tree.encode(&block.data)?;
        let freq_table = tree.serialize_frequencies();

        let mut output = Vec::new();
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());
        output.extend_from_slice(&(total_bits as u32).to_le_bytes());
        for &freq in &freq_table {
            output.extend_from_slice(&freq.to_le_bytes());
        }
        output.extend_from_slice(&huffman_data);

        block.data = output;
    }

    Ok(block)
}

/// Bw stage 0: BWT encoding.
fn stage_bwt_encode(mut block: StageBlock, options: &CompressOptions) -> PzResult<StageBlock> {
    let bwt_result = bwt_encode_with_backend(&block.data, options)?;
    block.metadata.bwt_primary_index = Some(bwt_result.primary_index);
    block.data = bwt_result.data;
    Ok(block)
}

/// Bw stage 1: MTF encoding.
fn stage_mtf_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = mtf::encode(&block.data);
    Ok(block)
}

/// Bw stage 2: RLE encoding.
fn stage_rle_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = rle::encode(&block.data);
    block.metadata.pre_entropy_len = Some(block.data.len());
    Ok(block)
}

/// Bw stage 3: FSE encoding + serialization.
fn stage_fse_encode_bw(mut block: StageBlock) -> PzResult<StageBlock> {
    let primary_index = block.metadata.bwt_primary_index.unwrap();
    let rle_len = block.metadata.pre_entropy_len.unwrap();
    let fse_data = fse::encode(&block.data);

    let mut output = Vec::new();
    output.extend_from_slice(&primary_index.to_le_bytes());
    output.extend_from_slice(&(rle_len as u32).to_le_bytes());
    output.extend_from_slice(&fse_data);

    block.data = output;
    Ok(block)
}

// --- Decompression stage functions ---

/// Bw decompress stage 0: parse metadata + FSE decode.
fn stage_fse_decode_bw(mut block: StageBlock) -> PzResult<StageBlock> {
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
fn stage_rle_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = rle::decode(&block.data)?;
    Ok(block)
}

/// Bw decompress stage 2: MTF decode.
fn stage_mtf_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    block.data = mtf::decode(&block.data);
    Ok(block)
}

/// Bw decompress stage 3: BWT decode.
fn stage_bwt_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    let primary_index = block.metadata.bwt_primary_index.unwrap();
    block.data = bwt::decode(&block.data, primary_index)?;
    Ok(block)
}

/// Deflate decompress stage 0: parse metadata + Huffman decode.
///
/// Detects multi-stream vs single-stream format by checking the first u32:
///   - 0 → multi-stream (sentinel), followed by num_streams, lz_len, per-stream data
///   - non-zero → single-stream (legacy), first u32 is lz_len
fn stage_huffman_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let first_u32 =
        u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);

    if first_u32 == 0 {
        // Multi-stream format
        // Header: [0u32: sentinel][num_streams: u8][lz_len: u32]
        if block.data.len() < 9 {
            return Err(PzError::InvalidInput);
        }
        let num_streams = block.data[4] as usize;
        let lz_len =
            u32::from_le_bytes([block.data[5], block.data[6], block.data[7], block.data[8]])
                as usize;
        block.metadata.lz_len = Some(lz_len);

        let mut pos = 9;
        let mut decoded_streams = Vec::with_capacity(num_streams);

        for _ in 0..num_streams {
            // Per-stream: [stream_data_len: u32][total_bits: u32][freq_table: 256×u32][huffman_data]
            if pos + 1032 > block.data.len() {
                return Err(PzError::InvalidInput);
            }
            let stream_data_len = u32::from_le_bytes([
                block.data[pos],
                block.data[pos + 1],
                block.data[pos + 2],
                block.data[pos + 3],
            ]) as usize;
            let total_bits = u32::from_le_bytes([
                block.data[pos + 4],
                block.data[pos + 5],
                block.data[pos + 6],
                block.data[pos + 7],
            ]) as usize;

            let mut freq_table = crate::frequency::FrequencyTable::new();
            for i in 0..256 {
                let off = pos + 8 + i * 4;
                freq_table.byte[i] = u32::from_le_bytes([
                    block.data[off],
                    block.data[off + 1],
                    block.data[off + 2],
                    block.data[off + 3],
                ]);
            }
            freq_table.total = freq_table.byte.iter().map(|&f| f as u64).sum();
            freq_table.used = freq_table.byte.iter().filter(|&&f| f > 0).count() as u32;

            let huff_start = pos + 1032;
            if huff_start + stream_data_len > block.data.len() {
                return Err(PzError::InvalidInput);
            }
            let tree =
                HuffmanTree::from_frequency_table(&freq_table).ok_or(PzError::InvalidInput)?;
            let decoded = tree.decode(
                &block.data[huff_start..huff_start + stream_data_len],
                total_bits,
            )?;

            decoded_streams.push(decoded);
            pos = huff_start + stream_data_len;
        }

        block.streams = Some(decoded_streams);
        block.data.clear();
    } else {
        // Single-stream (legacy) format
        if block.data.len() < 1032 {
            return Err(PzError::InvalidInput);
        }
        let lz_len = first_u32 as usize;
        let total_bits =
            u32::from_le_bytes([block.data[4], block.data[5], block.data[6], block.data[7]])
                as usize;

        let mut freq_table = crate::frequency::FrequencyTable::new();
        for i in 0..256 {
            let offset = 8 + i * 4;
            freq_table.byte[i] = u32::from_le_bytes([
                block.data[offset],
                block.data[offset + 1],
                block.data[offset + 2],
                block.data[offset + 3],
            ]);
        }
        freq_table.total = freq_table.byte.iter().map(|&f| f as u64).sum();
        freq_table.used = freq_table.byte.iter().filter(|&&f| f > 0).count() as u32;

        let tree = HuffmanTree::from_frequency_table(&freq_table).ok_or(PzError::InvalidInput)?;
        let mut lz_data = vec![0u8; lz_len];
        let decoded_len = tree.decode_to_buf(&block.data[1032..], total_bits, &mut lz_data)?;
        if decoded_len != lz_len {
            return Err(PzError::InvalidInput);
        }

        block.metadata.lz_len = Some(lz_len);
        block.data = lz_data;
    }

    Ok(block)
}

/// LZ-pipeline decompress: LZ77 decompress.
///
/// If `block.streams` is `Some` with 3 streams (offsets, lengths, literals),
/// reinterleaves them into the flat LzMatch byte stream before decompressing.
fn stage_lz77_decompress(mut block: StageBlock) -> PzResult<StageBlock> {
    if let Some(streams) = block.streams.take() {
        // Multi-stream: reinterleave offsets, lengths, literals into flat LzMatch bytes
        if streams.len() != 3 {
            return Err(PzError::InvalidInput);
        }
        let offsets = &streams[0];
        let lengths = &streams[1];
        let literals = &streams[2];

        // offsets and lengths are 2 bytes per match, literals are 1 byte per match
        if offsets.len() != lengths.len() || offsets.len() != literals.len() * 2 {
            return Err(PzError::InvalidInput);
        }
        let num_matches = literals.len();
        let match_size = lz77::Match::SERIALIZED_SIZE; // 5
        let mut lz_data = Vec::with_capacity(num_matches * match_size);

        for i in 0..num_matches {
            lz_data.push(offsets[i * 2]);
            lz_data.push(offsets[i * 2 + 1]);
            lz_data.push(lengths[i * 2]);
            lz_data.push(lengths[i * 2 + 1]);
            lz_data.push(literals[i]);
        }

        let mut output = vec![0u8; block.original_len];
        let out_len = lz77::decompress_to_buf(&lz_data, &mut output)?;
        if out_len != block.original_len {
            return Err(PzError::InvalidInput);
        }
        block.data = output;
        block.streams = None;
    } else {
        // Single-stream (legacy) path
        let mut output = vec![0u8; block.original_len];
        let out_len = lz77::decompress_to_buf(&block.data, &mut output)?;
        if out_len != block.original_len {
            return Err(PzError::InvalidInput);
        }
        block.data = output;
    }
    Ok(block)
}

/// Dispatch to the appropriate compression stage function.
fn run_compress_stage(
    pipeline: Pipeline,
    stage_idx: usize,
    block: StageBlock,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    match (pipeline, stage_idx) {
        (Pipeline::Deflate, 0) => stage_lz77_compress(block, options),
        (Pipeline::Deflate, 1) => stage_huffman_encode(block),
        (Pipeline::Bw, 0) => stage_bwt_encode(block, options),
        (Pipeline::Bw, 1) => stage_mtf_encode(block),
        (Pipeline::Bw, 2) => stage_rle_encode(block),
        (Pipeline::Bw, 3) => stage_fse_encode_bw(block),
        (Pipeline::Lzr, 0) => stage_lz77_compress(block, options),
        (Pipeline::Lzr, 1) => stage_rans_encode(block),
        (Pipeline::Lzf, 0) => stage_lz77_compress(block, options),
        (Pipeline::Lzf, 1) => stage_fse_encode(block),
        _ => Err(PzError::Unsupported),
    }
}

/// Dispatch to the appropriate decompression stage function.
fn run_decompress_stage(
    pipeline: Pipeline,
    stage_idx: usize,
    block: StageBlock,
) -> PzResult<StageBlock> {
    match (pipeline, stage_idx) {
        // Deflate: Huffman decode(0) → LZ77 decompress(1)
        (Pipeline::Deflate, 0) => stage_huffman_decode(block),
        (Pipeline::Deflate, 1) => stage_lz77_decompress(block),
        // Bw: FSE decode(0) → RLE decode(1) → MTF decode(2) → BWT decode(3)
        (Pipeline::Bw, 0) => stage_fse_decode_bw(block),
        (Pipeline::Bw, 1) => stage_rle_decode(block),
        (Pipeline::Bw, 2) => stage_mtf_decode(block),
        (Pipeline::Bw, 3) => stage_bwt_decode(block),
        // Lzr: rANS decode(0) → LZ77 decompress(1)
        (Pipeline::Lzr, 0) => stage_rans_decode(block),
        (Pipeline::Lzr, 1) => stage_lz77_decompress(block),
        // Lzf: FSE decode(0) → LZ77 decompress(1)
        (Pipeline::Lzf, 0) => stage_fse_decode(block),
        (Pipeline::Lzf, 1) => stage_lz77_decompress(block),
        _ => Err(PzError::Unsupported),
    }
}

/// Pipeline-parallel compression: one thread per stage, connected by channels.
fn compress_pipeline_parallel(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    let block_size = options.block_size;
    let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
    let num_blocks = blocks.len();
    let stage_count = pipeline_stage_count(pipeline);

    // Capture original block lengths before `blocks` is moved into the scope.
    let orig_block_lens: Vec<usize> = blocks.iter().map(|b| b.len()).collect();

    let results: Vec<PzResult<Vec<u8>>> = std::thread::scope(|scope| {
        use std::sync::mpsc;

        // Build channel chain: producer → stage[0] → stage[1] → ... → collector
        let (tx_in, mut prev_rx) = mpsc::sync_channel::<PzResult<StageBlock>>(2);

        for stage_idx in 0..stage_count {
            let (tx_out, rx_next) = mpsc::sync_channel::<PzResult<StageBlock>>(2);
            let rx = prev_rx;
            let opts = options.clone();

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
fn decompress_pipeline_parallel(
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

// --- LZ77 helper: select GPU or CPU backend for match finding ---

/// Run LZ77 compression using the configured backend and parse strategy.
///
/// Backend/strategy selection logic:
/// - `Auto` on GPU: hash-table kernel (best throughput at ≥256KB)
/// - `Auto` on CPU: lazy matching (best compression ratio)
/// - `Optimal` on GPU: GPU top-K match table → CPU backward DP
/// - `Optimal` on CPU: CPU match table → CPU backward DP
/// - GPU backend falls back to CPU when input < MIN_GPU_INPUT_SIZE
fn lz77_compress_with_backend(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    #[cfg(feature = "opencl")]
    {
        if let Backend::OpenCl = options.backend {
            if let Some(ref engine) = options.opencl_engine {
                if input.len() >= crate::opencl::MIN_GPU_INPUT_SIZE {
                    if options.parse_strategy == ParseStrategy::Optimal {
                        // GPU top-K match table + CPU optimal parse DP
                        let table = engine.find_topk_matches(input)?;
                        return crate::optimal::compress_optimal_with_table(input, &table);
                    }
                    // GPU hash-table kernel for Auto/Greedy/Lazy
                    return engine.lz77_compress(input, crate::opencl::KernelVariant::HashTable);
                }
                // Input too small for GPU — fall through to CPU paths
            }
        }
    }

    #[cfg(feature = "webgpu")]
    {
        if let Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                if input.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE
                    && input.len() <= engine.max_dispatch_input_size()
                {
                    if options.parse_strategy == ParseStrategy::Optimal {
                        let table = engine.find_topk_matches(input)?;
                        return crate::optimal::compress_optimal_with_table(input, &table);
                    }
                    return engine.lz77_compress(input);
                }
            }
        }
    }

    // CPU paths — lazy is the default (fastest single-thread + best ratio).
    // Multi-threading happens at the pipeline block level, not inside LZ77.
    match options.parse_strategy {
        ParseStrategy::Auto | ParseStrategy::Lazy => lz77::compress_lazy(input),
        ParseStrategy::Optimal => crate::optimal::compress_optimal(input),
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
    let block = stage_lz77_compress(block, options)?;
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
    let block = stage_lz77_decompress(block)?;
    Ok(block.data)
}

// --- BWT helper: select GPU or CPU backend for suffix array construction ---

/// Run BWT encoding using the configured backend.
fn bwt_encode_with_backend(input: &[u8], options: &CompressOptions) -> PzResult<bwt::BwtResult> {
    #[cfg(feature = "opencl")]
    {
        if let Backend::OpenCl = options.backend {
            if let Some(ref engine) = options.opencl_engine {
                // Skip GPU BWT on CPU OpenCL devices — bitonic sort is O(log^2 n)
                // kernel launches which is much slower than CPU's O(n) SA-IS.
                if !engine.is_cpu_device() && input.len() >= crate::opencl::MIN_GPU_BWT_SIZE {
                    return engine.bwt_encode(input);
                }
            }
        }
    }

    #[cfg(feature = "webgpu")]
    {
        if let Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                if !engine.is_cpu_device()
                    && input.len() >= crate::webgpu::MIN_GPU_BWT_SIZE
                    && input.len() <= engine.max_dispatch_input_size()
                {
                    return engine.bwt_encode(input);
                }
            }
        }
    }

    #[cfg(not(any(feature = "opencl", feature = "webgpu")))]
    let _ = options;

    bwt::encode(input).ok_or(PzError::InvalidInput)
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

// --- LZR pipeline: LZ77 + rANS ---

/// Lzr stage 1: rANS encoding + serialization.
///
/// Multi-stream format (when `block.streams` is `Some`):
///   [0u32: sentinel] [num_streams: u8] [lz_len: u32]
///   For each stream:
///     [original_stream_len: u32] [compressed_stream_len: u32] [rans_data]
///
/// Single-stream format (legacy, when `block.streams` is `None`):
///   [lz_len: u32] [rans_data]
fn stage_rans_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let lz_len = block.metadata.lz_len.unwrap();

    if let Some(streams) = block.streams.take() {
        // Multi-stream encoding: encode each stream independently with rANS
        let mut output = Vec::new();
        output.extend_from_slice(&0u32.to_le_bytes()); // sentinel
        output.push(streams.len() as u8);
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());

        for stream in &streams {
            let rans_data = rans::encode(stream);
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            output.extend_from_slice(&(rans_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&rans_data);
        }

        block.data = output;
    } else {
        // Single-stream (legacy) path
        let rans_data = rans::encode(&block.data);
        let mut output = Vec::new();
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());
        output.extend_from_slice(&rans_data);
        block.data = output;
    }

    Ok(block)
}

/// Lzr decompress stage 0: parse metadata + rANS decode.
///
/// Detects multi-stream vs single-stream format by checking the first u32:
///   - 0 → multi-stream (sentinel)
///   - non-zero → single-stream (legacy), first u32 is lz_len
fn stage_rans_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let first_u32 =
        u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);

    if first_u32 == 0 {
        // Multi-stream format
        if block.data.len() < 9 {
            return Err(PzError::InvalidInput);
        }
        let num_streams = block.data[4] as usize;
        let lz_len =
            u32::from_le_bytes([block.data[5], block.data[6], block.data[7], block.data[8]])
                as usize;
        block.metadata.lz_len = Some(lz_len);

        let mut pos = 9;
        let mut decoded_streams = Vec::with_capacity(num_streams);

        for _ in 0..num_streams {
            if pos + 8 > block.data.len() {
                return Err(PzError::InvalidInput);
            }
            let orig_stream_len = u32::from_le_bytes([
                block.data[pos],
                block.data[pos + 1],
                block.data[pos + 2],
                block.data[pos + 3],
            ]) as usize;
            let comp_stream_len = u32::from_le_bytes([
                block.data[pos + 4],
                block.data[pos + 5],
                block.data[pos + 6],
                block.data[pos + 7],
            ]) as usize;
            pos += 8;

            if pos + comp_stream_len > block.data.len() {
                return Err(PzError::InvalidInput);
            }
            let decoded = rans::decode(&block.data[pos..pos + comp_stream_len], orig_stream_len)?;
            decoded_streams.push(decoded);
            pos += comp_stream_len;
        }

        block.streams = Some(decoded_streams);
        block.data.clear();
    } else {
        // Single-stream (legacy) format
        let lz_len = first_u32 as usize;
        block.metadata.lz_len = Some(lz_len);
        block.data = rans::decode(&block.data[4..], lz_len)?;
    }

    Ok(block)
}

/// Compress a single block using the Lzr pipeline (no container header).
fn compress_block_lzr(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_lz77_compress(block, options)?;
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
    let block = stage_lz77_decompress(block)?;
    Ok(block.data)
}

// --- LZF pipeline: LZ77 + FSE ---

/// Lzf stage 1: FSE encoding + serialization.
///
/// Multi-stream format (when `block.streams` is `Some`):
///   [0u32: sentinel] [num_streams: u8] [lz_len: u32]
///   For each stream:
///     [original_stream_len: u32] [compressed_stream_len: u32] [fse_data]
///
/// Single-stream format (legacy, when `block.streams` is `None`):
///   [lz_len: u32] [fse_data]
fn stage_fse_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let lz_len = block.metadata.lz_len.unwrap();

    if let Some(streams) = block.streams.take() {
        // Multi-stream encoding: encode each stream independently with FSE
        let mut output = Vec::new();
        output.extend_from_slice(&0u32.to_le_bytes()); // sentinel
        output.push(streams.len() as u8);
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());

        for stream in &streams {
            let fse_data = fse::encode(stream);
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
            output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&fse_data);
        }

        block.data = output;
    } else {
        // Single-stream (legacy) path
        let fse_data = fse::encode(&block.data);
        let mut output = Vec::new();
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());
        output.extend_from_slice(&fse_data);
        block.data = output;
    }

    Ok(block)
}

/// Lzf decompress stage 0: parse metadata + FSE decode.
///
/// Detects multi-stream vs single-stream format by checking the first u32:
///   - 0 → multi-stream (sentinel)
///   - non-zero → single-stream (legacy), first u32 is lz_len
fn stage_fse_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let first_u32 =
        u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);

    if first_u32 == 0 {
        // Multi-stream format
        if block.data.len() < 9 {
            return Err(PzError::InvalidInput);
        }
        let num_streams = block.data[4] as usize;
        let lz_len =
            u32::from_le_bytes([block.data[5], block.data[6], block.data[7], block.data[8]])
                as usize;
        block.metadata.lz_len = Some(lz_len);

        let mut pos = 9;
        let mut decoded_streams = Vec::with_capacity(num_streams);

        for _ in 0..num_streams {
            if pos + 8 > block.data.len() {
                return Err(PzError::InvalidInput);
            }
            let orig_stream_len = u32::from_le_bytes([
                block.data[pos],
                block.data[pos + 1],
                block.data[pos + 2],
                block.data[pos + 3],
            ]) as usize;
            let comp_stream_len = u32::from_le_bytes([
                block.data[pos + 4],
                block.data[pos + 5],
                block.data[pos + 6],
                block.data[pos + 7],
            ]) as usize;
            pos += 8;

            if pos + comp_stream_len > block.data.len() {
                return Err(PzError::InvalidInput);
            }
            let decoded = fse::decode(&block.data[pos..pos + comp_stream_len], orig_stream_len)?;
            decoded_streams.push(decoded);
            pos += comp_stream_len;
        }

        block.streams = Some(decoded_streams);
        block.data.clear();
    } else {
        // Single-stream (legacy) format
        let lz_len = first_u32 as usize;
        block.metadata.lz_len = Some(lz_len);
        block.data = fse::decode(&block.data[4..], lz_len)?;
    }

    Ok(block)
}

/// Compress a single block using the Lzf pipeline (no container header).
fn compress_block_lzf(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_lz77_compress(block, options)?;
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
    let block = stage_lz77_decompress(block)?;
    Ok(block.data)
}

// --- Pipeline auto-selection ---

/// Automatically select the best pipeline for the given input data.
///
/// Uses statistical analysis (byte entropy, match density, run ratio,
/// distribution shape) to predict which pipeline will perform best.
/// The heuristic is fast (O(n) on a 64KB sample) and suitable for
/// use as a default when the caller doesn't know the data characteristics.
pub fn select_pipeline(input: &[u8]) -> Pipeline {
    use crate::analysis::{self, DistributionShape};

    if input.is_empty() {
        return Pipeline::Deflate;
    }

    let profile = analysis::analyze(input);

    // Near-random data: use fastest pipeline, won't compress much
    if profile.byte_entropy > 7.5 && profile.match_density < 0.1 {
        return Pipeline::Deflate;
    }

    // High run ratio: BWT+RLE excels
    if profile.run_ratio > 0.3 {
        return Pipeline::Bw;
    }

    // Low entropy with skewed or constant distribution: BWT handles well
    if profile.byte_entropy < 3.0
        && matches!(
            profile.distribution_shape,
            DistributionShape::Skewed | DistributionShape::Constant
        )
    {
        return Pipeline::Bw;
    }

    // Good match density: LZ-based pipelines
    if profile.match_density > 0.4 {
        if profile.byte_entropy > 6.0 {
            // High entropy: FSE handles better than Huffman
            return Pipeline::Lzf;
        }
        return Pipeline::Deflate;
    }

    // Moderate match density with high entropy
    if profile.match_density > 0.2 && profile.byte_entropy > 5.0 {
        return Pipeline::Lzf;
    }

    // Default: Deflate (fast, decent compression)
    Pipeline::Deflate
}

/// Select the best pipeline by trial compression.
///
/// Compresses the first `sample_size` bytes with each candidate pipeline,
/// measures compressed size, and picks the one with the best ratio.
/// Falls back to heuristic selection if all trials fail.
pub fn select_pipeline_trial(
    input: &[u8],
    options: &CompressOptions,
    sample_size: usize,
) -> Pipeline {
    if input.is_empty() {
        return Pipeline::Deflate;
    }

    let sample_len = input.len().min(sample_size);
    let sample = &input[..sample_len];

    // Force single-threaded for trial (avoid overhead)
    let trial_opts = CompressOptions {
        threads: 1,
        ..options.clone()
    };

    let candidates = [
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Lzr,
        Pipeline::Lzf,
    ];
    let mut best_pipeline = Pipeline::Deflate;
    let mut best_size = usize::MAX;

    for &pipeline in &candidates {
        if let Ok(compressed) = compress_with_options(sample, pipeline, &trial_opts) {
            if compressed.len() < best_size {
                best_size = compressed.len();
                best_pipeline = pipeline;
            }
        }
    }

    best_pipeline
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Deflate pipeline tests ---

    #[test]
    fn test_deflate_empty() {
        let result = compress(&[], Pipeline::Deflate).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_deflate_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_deflate_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        // Use enough repetitions to overcome the ~1KB frequency table overhead
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        // Should actually compress (input is ~4.5KB, overhead is ~1KB)
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_deflate_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- BW pipeline tests ---

    #[test]
    fn test_bw_empty() {
        let result = compress(&[], Pipeline::Bw).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_bw_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_bw_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_bw_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_bw_round_trip_all_same() {
        let input = vec![b'x'; 200];
        let compressed = compress(&input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- Header / format tests ---

    #[test]
    fn test_invalid_magic() {
        let result = decompress(&[0, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_invalid_version() {
        let result = decompress(&[b'P', b'Z', 99, 0, 0, 0, 0, 0]);
        assert_eq!(result, Err(PzError::Unsupported));
    }

    #[test]
    fn test_invalid_pipeline() {
        let result = decompress(&[b'P', b'Z', VERSION, 99, 0, 0, 0, 0]);
        assert_eq!(result, Err(PzError::Unsupported));
    }

    #[test]
    fn test_too_short_input() {
        let result = decompress(b"PZ");
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_zero_original_length() {
        let result = decompress(&[b'P', b'Z', VERSION, 0, 0, 0, 0, 0]);
        assert_eq!(result.unwrap(), Vec::<u8>::new());
    }

    // --- Cross-pipeline tests ---

    #[test]
    fn test_all_pipelines_banana() {
        let input = b"banana banana banana banana banana";
        for &pipeline in &[
            Pipeline::Deflate,
            Pipeline::Bw,
            Pipeline::Lzr,
            Pipeline::Lzf,
        ] {
            let compressed = compress(input, pipeline).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(decompressed, input, "failed for pipeline {:?}", pipeline);
        }
    }

    #[test]
    fn test_all_pipelines_medium_text() {
        let mut input = Vec::new();
        for _ in 0..10 {
            input.extend(b"abcdefghij klmnopqrstuvwxyz 0123456789 ");
        }
        for &pipeline in &[
            Pipeline::Deflate,
            Pipeline::Bw,
            Pipeline::Lzr,
            Pipeline::Lzf,
        ] {
            let compressed = compress(&input, pipeline).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(decompressed, input, "failed for pipeline {:?}", pipeline);
        }
    }

    // --- Multi-block parallel compression tests ---

    /// Helper: compress with explicit thread count and block size.
    fn compress_mt(
        input: &[u8],
        pipeline: Pipeline,
        threads: usize,
        block_size: usize,
    ) -> PzResult<Vec<u8>> {
        let options = CompressOptions {
            threads,
            block_size,
            ..Default::default()
        };
        compress_with_options(input, pipeline, &options)
    }

    #[test]
    fn test_multiblock_round_trip_all_pipelines() {
        // 2KB input with 512-byte blocks = 4 blocks
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }

        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            let compressed = compress_mt(&input, pipeline, 4, 512).unwrap();
            assert_eq!(compressed[2], VERSION, "expected V2 for {:?}", pipeline);
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(decompressed, input, "round-trip failed for {:?}", pipeline);
        }
    }

    #[test]
    fn test_multiblock_single_block_fallback() {
        // Input smaller than block_size → single block in V2 format
        let input = b"small input data";
        let compressed = compress_mt(input, Pipeline::Deflate, 4, 65536).unwrap();
        assert_eq!(compressed[2], VERSION, "expected V2 for small input");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multiblock_single_thread() {
        // threads=1 → single block in V2 format
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Bw, 1, 512).unwrap();
        assert_eq!(compressed[2], VERSION, "expected V2 for single-threaded");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multiblock_various_block_sizes() {
        let pattern = b"ABCDEFGHIJ0123456789";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }

        for block_size in [256, 512, 1024, 2048] {
            for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
                let compressed = compress_mt(&input, pipeline, 4, block_size).unwrap();
                let decompressed = decompress(&compressed).unwrap();
                assert_eq!(
                    decompressed, input,
                    "failed for {:?} block_size={}",
                    pipeline, block_size
                );
            }
        }
    }

    #[test]
    fn test_multiblock_exact_one_block() {
        // Input exactly equal to block_size → single block in V2 format
        let input = vec![b'x'; 1024];
        let compressed = compress_mt(&input, Pipeline::Bw, 4, 1024).unwrap();
        assert_eq!(compressed[2], VERSION, "exact one block should be V2");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multiblock_empty_input() {
        let compressed = compress_mt(&[], Pipeline::Deflate, 4, 512).unwrap();
        assert!(compressed.is_empty());
    }

    #[test]
    fn test_multiblock_2_threads() {
        // 4KB input, 1KB blocks = 4 blocks, but only 2 threads
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Lzf, 2, 1024).unwrap();
        assert_eq!(compressed[2], VERSION);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multiblock_decompress_with_threads() {
        // Verify decompress_with_threads works
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Bw, 4, 512).unwrap();
        let decompressed = decompress_with_threads(&compressed, 2).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multiblock_large_input() {
        // ~100KB input, 16KB blocks = ~6 blocks
        let pattern = b"Compression test data with some repetition. ";
        let mut input = Vec::new();
        for _ in 0..2500 {
            input.extend_from_slice(pattern);
        }
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            let compressed = compress_mt(&input, pipeline, 4, 16384).unwrap();
            assert_eq!(compressed[2], VERSION);
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(decompressed, input, "large input failed for {:?}", pipeline);
        }
    }

    // --- Pipeline-parallel specific tests ---

    #[test]
    fn test_pipeline_parallel_round_trip_all_pipelines() {
        // Force pipeline-parallel path: need num_blocks >= stage_count.
        // Bw has 4 stages, so we need >= 4 blocks. Use small block_size.
        let pattern = b"Pipeline parallelism test data with repetitions. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        // 5000 bytes / 512 byte blocks = ~10 blocks, enough for any pipeline
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            let stage_count = pipeline_stage_count(pipeline);
            let block_size = 512;
            let num_blocks = input.len().div_ceil(block_size);
            assert!(
                num_blocks >= stage_count,
                "test setup: need {} blocks >= {} stages for {:?}",
                num_blocks,
                stage_count,
                pipeline
            );

            let compressed = compress_mt(&input, pipeline, 4, block_size).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(
                decompressed, input,
                "pipeline-parallel round-trip failed for {:?}",
                pipeline
            );
        }
    }

    #[test]
    fn test_pipeline_parallel_matches_single_threaded() {
        // Verify pipeline-parallel produces output that single-threaded can decompress,
        // and vice versa.
        let pattern = b"Cross-check between single-threaded and pipeline-parallel. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }

        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            // Single-threaded compress
            let compressed_st = compress(input.as_slice(), pipeline).unwrap();
            let decompressed_st = decompress(&compressed_st).unwrap();
            assert_eq!(decompressed_st, input);

            // Multi-threaded (pipeline-parallel) compress
            let compressed_mt = compress_mt(&input, pipeline, 4, 512).unwrap();
            let decompressed_mt = decompress(&compressed_mt).unwrap();
            assert_eq!(decompressed_mt, input);

            // Both decompress to the same data
            assert_eq!(
                decompressed_st, decompressed_mt,
                "single-threaded and pipeline-parallel disagree for {:?}",
                pipeline
            );
        }
    }

    #[test]
    fn test_pipeline_parallel_many_small_blocks() {
        // Stress test: many blocks flowing through the pipeline.
        // 10KB input / 128 byte blocks = 80 blocks
        let input: Vec<u8> = (0..10240).map(|i| (i % 251) as u8).collect();
        let compressed = compress_mt(&input, Pipeline::Bw, 4, 128).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_pipeline_parallel_decompress_with_threads() {
        // Compress with pipeline parallelism, then decompress with explicit threads.
        let pattern = b"Decompress parallelism test. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }

        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            let compressed = compress_mt(&input, pipeline, 4, 512).unwrap();

            // Decompress with various thread counts
            for threads in [0, 1, 2, 4] {
                let decompressed = decompress_with_threads(&compressed, threads).unwrap();
                assert_eq!(
                    decompressed, input,
                    "decompress failed for {:?} threads={}",
                    pipeline, threads
                );
            }
        }
    }

    #[test]
    fn test_pipeline_parallel_error_propagation() {
        // Feed corrupt data that will fail during decompression stages.
        // Create valid V2 header but corrupt block data.
        let input = vec![b'A'; 2048];
        let compressed = compress_mt(&input, Pipeline::Bw, 4, 512).unwrap();

        // Corrupt the last byte of the block data
        let mut corrupt = compressed.clone();
        let last = corrupt.len() - 1;
        corrupt[last] ^= 0xFF;

        // Should fail (either error or wrong data)
        let result = decompress(&corrupt);
        // We just verify it doesn't hang/panic — either error or wrong data is acceptable
        // for this corruption test.
        match result {
            Ok(_data) => {
                // If it "succeeds", corruption wasn't detected
                // (may not always be detected depending on where it lands)
            }
            Err(_) => {
                // Expected: corruption detected
            }
        }
    }

    // --- Optimal parsing pipeline tests ---

    #[test]
    fn test_optimal_deflate_round_trip() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let opts = CompressOptions {
            parse_strategy: ParseStrategy::Optimal,
            threads: 1,
            ..Default::default()
        };
        let compressed = compress_with_options(&input, Pipeline::Deflate, &opts).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_optimal_lza_round_trip() {
        let pattern = b"abcdefghij abcdefghij ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let opts = CompressOptions {
            parse_strategy: ParseStrategy::Optimal,
            threads: 1,
            ..Default::default()
        };
        let compressed = compress_with_options(&input, Pipeline::Lzf, &opts).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_optimal_multiblock_round_trip() {
        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..500 {
            input.extend_from_slice(pattern);
        }
        let opts = CompressOptions {
            parse_strategy: ParseStrategy::Optimal,
            threads: 2,
            block_size: 4096,
            ..Default::default()
        };
        let compressed = compress_with_options(&input, Pipeline::Deflate, &opts).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- Auto-selection tests ---

    #[test]
    fn test_select_pipeline_empty() {
        assert_eq!(select_pipeline(&[]), Pipeline::Deflate);
    }

    #[test]
    fn test_select_pipeline_constant_data() {
        // All-same bytes → high run ratio → should select Bw
        let input = vec![0xAA; 10000];
        let pipeline = select_pipeline(&input);
        assert_eq!(pipeline, Pipeline::Bw);
    }

    #[test]
    fn test_select_pipeline_text() {
        // Repetitive text → good match density, moderate entropy → Deflate or Lzf
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let pipeline = select_pipeline(&input);
        assert!(
            pipeline == Pipeline::Deflate || pipeline == Pipeline::Lzf,
            "expected Deflate or Lzf, got {:?}",
            pipeline
        );
    }

    #[test]
    fn test_select_pipeline_random() {
        // Pseudo-random → high entropy, low match density → Deflate (fastest)
        let mut input = vec![0u8; 10000];
        let mut state: u32 = 12345;
        for byte in &mut input {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }
        let pipeline = select_pipeline(&input);
        assert_eq!(pipeline, Pipeline::Deflate);
    }

    #[test]
    fn test_select_pipeline_trial_round_trip() {
        // Trial-selected pipeline must produce valid compressed output
        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let opts = CompressOptions::default();
        let pipeline = select_pipeline_trial(&input, &opts, 4096);
        let compressed = compress(&input, pipeline).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_select_pipeline_trial_small_sample() {
        // Trial with very small input should not panic
        let input = b"tiny";
        let opts = CompressOptions::default();
        let pipeline = select_pipeline_trial(input, &opts, 32);
        // Should still produce a valid pipeline
        let compressed = compress(input, pipeline).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input.as_slice());
    }

    #[test]
    fn test_select_pipeline_auto_vs_explicit_round_trip() {
        // Every auto-selected pipeline must produce valid output
        let test_inputs: Vec<Vec<u8>> = vec![
            vec![0xFF; 5000],                            // constant
            (0..=255u8).cycle().take(5000).collect(),    // uniform
            b"banana banana banana banana ".repeat(100), // text
        ];
        for input in &test_inputs {
            let pipeline = select_pipeline(input);
            let compressed = compress(input, pipeline).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(&decompressed, input);
        }
    }

    // --- Multi-stream Deflate tests ---

    #[test]
    fn test_multistream_deflate_round_trip_small() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multistream_deflate_round_trip_medium() {
        // 10KB of repetitive text
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..222 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_multistream_deflate_round_trip_large() {
        // 1MB of pseudo-random + repeated data
        let mut input = Vec::with_capacity(1 << 20);
        let pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        while input.len() < (1 << 20) {
            input.extend_from_slice(pattern);
        }
        input.truncate(1 << 20);
        // Sprinkle pseudo-random bytes for variety
        let mut state: u32 = 42;
        for i in (0..input.len()).step_by(37) {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            input[i] = (state >> 16) as u8;
        }

        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multistream_deflate_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(2048).collect();
        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multistream_deflate_multiblock() {
        // Multi-block pipeline-parallel path
        let pattern = b"Multi-stream multi-block test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Deflate, 4, 1024).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multistream_deflate_compression_ratio() {
        // Multi-stream should compress well on structured/repetitive data
        // because offset/length/literal distributions are each tighter
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..500 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        assert!(
            compressed.len() < input.len() / 2,
            "expected significant compression: compressed {} vs input {}",
            compressed.len(),
            input.len()
        );
    }

    // --- Multi-stream LZF tests ---

    #[test]
    fn test_multistream_lzf_round_trip_small() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multistream_lzf_round_trip_medium() {
        let pattern = b"abcdefghij abcdefghij ";
        let mut input = Vec::new();
        for _ in 0..500 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_multistream_lzf_round_trip_large() {
        // 1MB input
        let mut input = Vec::with_capacity(1 << 20);
        let pattern = b"LZF multi-stream test data with some repetition. ";
        while input.len() < (1 << 20) {
            input.extend_from_slice(pattern);
        }
        input.truncate(1 << 20);

        let compressed = compress(&input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multistream_lzf_multiblock() {
        let pattern = b"LZF multi-stream multi-block test. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Lzf, 4, 1024).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multistream_all_pipelines_round_trip() {
        // Verify all pipelines still produce correct round-trips
        let pattern = b"Cross-pipeline multi-stream validation. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzr] {
            let compressed = compress(&input, pipeline).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(decompressed, input, "round-trip failed for {:?}", pipeline);
        }
    }

    #[test]
    fn test_multistream_stage_deinterleave_reinterleave() {
        // Unit test: verify LZ77 deinterleave/reinterleave round-trips correctly
        // by going through the stage functions directly
        let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
        let opts = CompressOptions::default();
        let block = StageBlock {
            block_index: 0,
            original_len: input.len(),
            data: input.to_vec(),
            streams: None,
            metadata: StageMetadata::default(),
        };

        // LZ77 compress → produces streams
        let block = stage_lz77_compress(block, &opts).unwrap();
        assert!(block.streams.is_some());
        let streams = block.streams.as_ref().unwrap();
        assert_eq!(streams.len(), 3);
        // offsets and lengths are 2 bytes per match, literals 1 byte per match
        assert_eq!(streams[0].len(), streams[1].len());
        assert_eq!(streams[0].len(), streams[2].len() * 2);

        // Huffman encode → serializes streams into data
        let block = stage_huffman_encode(block).unwrap();
        assert!(block.streams.is_none());
        assert!(!block.data.is_empty());
        // Verify multi-stream sentinel
        let sentinel =
            u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);
        assert_eq!(sentinel, 0, "expected multi-stream sentinel");
        assert_eq!(block.data[4], 3, "expected 3 streams");

        // Huffman decode → restores streams
        let block = stage_huffman_decode(block).unwrap();
        assert!(block.streams.is_some());
        let streams = block.streams.as_ref().unwrap();
        assert_eq!(streams.len(), 3);

        // LZ77 decompress → reinterleaves and decompresses
        let block = stage_lz77_decompress(block).unwrap();
        assert!(block.streams.is_none());
        assert_eq!(block.data, input);
    }

    // --- Lzr pipeline tests ---

    #[test]
    fn test_lzr_empty() {
        let result = compress(&[], Pipeline::Lzr).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_lzr_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Lzr).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzr_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Lzr).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_lzr_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Lzr).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzr_round_trip_all_same() {
        let input = vec![0xAA_u8; 500];
        let compressed = compress(&input, Pipeline::Lzr).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzr_multiblock_round_trip() {
        let pattern = b"Lzr multi-block test data with repetition. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Lzr, 4, 1024).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzr_multiblock_large() {
        let mut input = Vec::with_capacity(1 << 20);
        let pattern = b"Lzr multi-stream test data with some repetition. ";
        while input.len() < (1 << 20) {
            input.extend_from_slice(pattern);
        }
        input.truncate(1 << 20);

        let compressed = compress(&input, Pipeline::Lzr).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzr_multistream_deinterleave_reinterleave() {
        // Verify LZ77 deinterleave → rANS encode → rANS decode → reinterleave
        let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
        let opts = CompressOptions::default();
        let block = StageBlock {
            block_index: 0,
            original_len: input.len(),
            data: input.to_vec(),
            streams: None,
            metadata: StageMetadata::default(),
        };

        // LZ77 compress → produces streams
        let block = stage_lz77_compress(block, &opts).unwrap();
        assert!(block.streams.is_some());
        let streams = block.streams.as_ref().unwrap();
        assert_eq!(streams.len(), 3);

        // rANS encode → serializes streams into data
        let block = stage_rans_encode(block).unwrap();
        assert!(block.streams.is_none());
        assert!(!block.data.is_empty());
        // Verify multi-stream sentinel
        let sentinel =
            u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);
        assert_eq!(sentinel, 0, "expected multi-stream sentinel");
        assert_eq!(block.data[4], 3, "expected 3 streams");

        // rANS decode → restores streams
        let block = stage_rans_decode(block).unwrap();
        assert!(block.streams.is_some());
        let streams = block.streams.as_ref().unwrap();
        assert_eq!(streams.len(), 3);

        // LZ77 decompress → reinterleaves and decompresses
        let block = stage_lz77_decompress(block).unwrap();
        assert!(block.streams.is_none());
        assert_eq!(block.data, input);
    }

    #[test]
    fn test_lzr_pipeline_parallel() {
        let pattern = b"Lzr pipeline-parallel test data. ";
        let mut input = Vec::new();
        for _ in 0..300 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Lzr, 4, 512).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzr_trial_selection_candidate() {
        // Verify Lzr is included in trial selection
        let pattern = b"Trial selection test data for rANS pipeline. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let opts = CompressOptions {
            threads: 1,
            ..CompressOptions::default()
        };
        // Just verify it doesn't crash — Lzr may or may not win
        let _pipeline = select_pipeline_trial(&input, &opts, 2048);
    }

    // --- Lzf pipeline tests ---

    #[test]
    fn test_lzf_empty() {
        let result = compress(&[], Pipeline::Lzf).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_lzf_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzf_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_lzf_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzf_round_trip_all_same() {
        let input = vec![0xAA_u8; 500];
        let compressed = compress(&input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzf_multiblock_round_trip() {
        let pattern = b"Lzf multi-block test data with repetition. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Lzf, 4, 1024).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzf_multiblock_large() {
        let mut input = Vec::with_capacity(1 << 20);
        let pattern = b"Lzf multi-stream test data with some repetition. ";
        while input.len() < (1 << 20) {
            input.extend_from_slice(pattern);
        }
        input.truncate(1 << 20);

        let compressed = compress(&input, Pipeline::Lzf).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzf_multistream_deinterleave_reinterleave() {
        // Verify LZ77 deinterleave → FSE encode → FSE decode → reinterleave
        let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
        let opts = CompressOptions::default();
        let block = StageBlock {
            block_index: 0,
            original_len: input.len(),
            data: input.to_vec(),
            streams: None,
            metadata: StageMetadata::default(),
        };

        // LZ77 compress → produces streams
        let block = stage_lz77_compress(block, &opts).unwrap();
        assert!(block.streams.is_some());
        let streams = block.streams.as_ref().unwrap();
        assert_eq!(streams.len(), 3);

        // FSE encode → serializes streams into data
        let block = stage_fse_encode(block).unwrap();
        assert!(block.streams.is_none());
        assert!(!block.data.is_empty());
        // Verify multi-stream sentinel
        let sentinel =
            u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);
        assert_eq!(sentinel, 0, "expected multi-stream sentinel");
        assert_eq!(block.data[4], 3, "expected 3 streams");

        // FSE decode → restores streams
        let block = stage_fse_decode(block).unwrap();
        assert!(block.streams.is_some());
        let streams = block.streams.as_ref().unwrap();
        assert_eq!(streams.len(), 3);

        // LZ77 decompress → reinterleaves and decompresses
        let block = stage_lz77_decompress(block).unwrap();
        assert!(block.streams.is_none());
        assert_eq!(block.data, input);
    }

    #[test]
    fn test_lzf_pipeline_parallel() {
        let pattern = b"Lzf pipeline-parallel test data. ";
        let mut input = Vec::new();
        for _ in 0..300 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Lzf, 4, 512).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lzf_trial_selection_candidate() {
        // Verify Lzf is included in trial selection
        let pattern = b"Trial selection test data for FSE pipeline. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let opts = CompressOptions {
            threads: 1,
            ..CompressOptions::default()
        };
        // Just verify it doesn't crash — Lzf may or may not win
        let _pipeline = select_pipeline_trial(&input, &opts, 2048);
    }
}
