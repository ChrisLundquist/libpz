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
/// | `Bw`          | BWT → MTF → RLE → Range coder    | bzip2      |
/// | `Lza`         | LZ77 → Range coder               | lzma-like  |
///
/// **Container format:**
/// Each compressed stream starts with a header:
/// - Magic bytes: `PZ` (2 bytes)
/// - Version: 1 (1 byte)
/// - Pipeline ID: 0=Deflate, 1=Bw, 2=Lza (1 byte)
/// - Original length: u32 little-endian (4 bytes)
/// - Pipeline-specific metadata (variable)
/// - Compressed data
use crate::bwt;
use crate::huffman::HuffmanTree;
use crate::lz77;
use crate::mtf;
use crate::rangecoder;
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
}

/// LZ77 match selection strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParseStrategy {
    /// Auto: choose the best strategy based on backend and input size.
    /// - GPU backend + large input: GPU hash-table kernel
    /// - CPU backend: lazy matching (best compression)
    /// - Equivalent to Lazy on CPU, HashTable on GPU.
    #[default]
    Auto,
    /// Greedy: hash-chain match finder, pick the longest match.
    Greedy,
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
    /// 1 = single-threaded (produces legacy single-block format).
    pub threads: usize,
    /// Block size for multi-threaded compression. Input is split into
    /// blocks of this size, each compressed independently.
    pub block_size: usize,
    /// LZ77 match selection strategy (greedy, lazy, or optimal).
    pub parse_strategy: ParseStrategy,
    /// OpenCL engine handle, required when `backend` is `Backend::OpenCl`.
    #[cfg(feature = "opencl")]
    pub opencl_engine: Option<std::sync::Arc<crate::opencl::OpenClEngine>>,
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
        }
    }
}

/// Magic bytes for the libpz container format.
const MAGIC: [u8; 2] = [b'P', b'Z'];
/// Format version for single-block streams (legacy).
const VERSION_V1: u8 = 1;
/// Format version for multi-block streams (parallel compression).
const VERSION_V2: u8 = 2;

/// Minimum header size: magic(2) + version(1) + pipeline(1) + orig_len(4) = 8
const MIN_HEADER_SIZE: usize = 8;

/// Compression pipeline types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pipeline {
    /// LZ77 + Huffman (gzip-like)
    Deflate = 0,
    /// BWT + MTF + RLE + Range coder (bzip2-like)
    Bw = 1,
    /// LZ77 + Range coder (lzma-like)
    Lza = 2,
}

impl TryFrom<u8> for Pipeline {
    type Error = PzError;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Deflate),
            1 => Ok(Self::Bw),
            2 => Ok(Self::Lza),
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
/// The output uses a multi-block container format. When `threads` is 1 or
/// the input fits in a single block, the legacy single-block format is used.
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
        return match pipeline {
            Pipeline::Deflate => compress_deflate_with_options(input, options),
            Pipeline::Bw => compress_bw_with_options(input, options),
            Pipeline::Lza => compress_lza_with_options(input, options),
        };
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
/// inverse stages. Automatically detects multi-block format and
/// decompresses blocks in parallel when applicable.
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
    if version != VERSION_V1 && version != VERSION_V2 {
        return Err(PzError::Unsupported);
    }

    let pipeline = Pipeline::try_from(input[3])?;
    let orig_len = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;

    if orig_len == 0 {
        return Ok(Vec::new());
    }

    let payload = &input[MIN_HEADER_SIZE..];

    if version == VERSION_V2 {
        // V2 multi-block format
        if payload.len() < 4 {
            return Err(PzError::InvalidInput);
        }
        let num_blocks =
            u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
        if num_blocks == 0 {
            return Err(PzError::InvalidInput);
        }

        let num_threads = resolve_thread_count(threads);
        let stage_count = pipeline_stage_count(pipeline);

        if num_threads > 1 && num_blocks >= stage_count {
            return decompress_pipeline_parallel(payload, pipeline, orig_len, num_blocks);
        }
        return decompress_parallel(payload, pipeline, orig_len, num_blocks, threads);
    }

    // V1 single-block format
    match pipeline {
        Pipeline::Deflate => decompress_deflate(payload, orig_len),
        Pipeline::Bw => decompress_bw(payload, orig_len),
        Pipeline::Lza => decompress_lza(payload, orig_len),
    }
}

/// Write the standard header to output (V1 single-block format).
fn write_header(output: &mut Vec<u8>, pipeline: Pipeline, orig_len: usize) {
    output.extend_from_slice(&MAGIC);
    output.push(VERSION_V1);
    output.push(pipeline as u8);
    output.extend_from_slice(&(orig_len as u32).to_le_bytes());
}

/// Write the V2 multi-block header to output.
fn write_header_v2(output: &mut Vec<u8>, pipeline: Pipeline, orig_len: usize) {
    output.extend_from_slice(&MAGIC);
    output.push(VERSION_V2);
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
        Pipeline::Lza => compress_block_lza(input, options),
    }
}

/// Decompress a single block using the appropriate pipeline (no container header).
fn decompress_block(payload: &[u8], pipeline: Pipeline, orig_len: usize) -> PzResult<Vec<u8>> {
    match pipeline {
        Pipeline::Deflate => decompress_block_deflate(payload, orig_len),
        Pipeline::Bw => decompress_block_bw(payload, orig_len),
        Pipeline::Lza => decompress_block_lza(payload, orig_len),
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
    write_header_v2(&mut output, pipeline, input.len());

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
    /// Accumulated metadata from prior stages.
    metadata: StageMetadata,
}

/// Metadata accumulated across pipeline stages.
#[derive(Default)]
struct StageMetadata {
    /// BWT primary index (Bw pipeline, set by BWT stage).
    bwt_primary_index: Option<u32>,
    /// Length of data before entropy coding.
    /// Bw: RLE output length. Lza: LZ77 output length.
    pre_entropy_len: Option<usize>,
    /// Deflate: LZ77 output length.
    lz_len: Option<usize>,
}

/// Number of compression stages in a pipeline.
fn pipeline_stage_count(pipeline: Pipeline) -> usize {
    match pipeline {
        Pipeline::Deflate => 2,
        Pipeline::Bw => 4,
        Pipeline::Lza => 2,
    }
}

// --- Compression stage functions ---

/// Deflate stage 0: LZ77 compression.
fn stage_lz77_compress(mut block: StageBlock, options: &CompressOptions) -> PzResult<StageBlock> {
    block.data = lz77_compress_with_backend(&block.data, options)?;
    block.metadata.lz_len = Some(block.data.len());
    Ok(block)
}

/// Deflate stage 1: Huffman encoding + serialization.
fn stage_huffman_encode(mut block: StageBlock) -> PzResult<StageBlock> {
    let lz_len = block.metadata.lz_len.unwrap();
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

/// Bw stage 3: Range coder encoding + serialization.
fn stage_rangecoder_encode_bw(mut block: StageBlock) -> PzResult<StageBlock> {
    let primary_index = block.metadata.bwt_primary_index.unwrap();
    let rle_len = block.metadata.pre_entropy_len.unwrap();
    let rc_data = rangecoder::encode(&block.data);

    let mut output = Vec::new();
    output.extend_from_slice(&primary_index.to_le_bytes());
    output.extend_from_slice(&(rle_len as u32).to_le_bytes());
    output.extend_from_slice(&rc_data);

    block.data = output;
    Ok(block)
}

/// Lza stage 1: Range coder encoding + serialization.
fn stage_rangecoder_encode_lza(mut block: StageBlock) -> PzResult<StageBlock> {
    let lz_len = block.metadata.lz_len.unwrap();
    let rc_data = rangecoder::encode(&block.data);

    let mut output = Vec::new();
    output.extend_from_slice(&(lz_len as u32).to_le_bytes());
    output.extend_from_slice(&rc_data);

    block.data = output;
    Ok(block)
}

// --- Decompression stage functions ---

/// Bw decompress stage 0: parse metadata + range decode.
fn stage_rangecoder_decode_bw(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 8 {
        return Err(PzError::InvalidInput);
    }
    let primary_index =
        u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]);
    let rle_len =
        u32::from_le_bytes([block.data[4], block.data[5], block.data[6], block.data[7]]) as usize;

    block.metadata.bwt_primary_index = Some(primary_index);
    block.data = rangecoder::decode(&block.data[8..], rle_len)?;
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
fn stage_huffman_decode(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 1032 {
        return Err(PzError::InvalidInput);
    }
    let lz_len =
        u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]) as usize;
    let total_bits =
        u32::from_le_bytes([block.data[4], block.data[5], block.data[6], block.data[7]]) as usize;

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
    Ok(block)
}

/// Deflate/Lza decompress: LZ77 decompress.
fn stage_lz77_decompress(mut block: StageBlock) -> PzResult<StageBlock> {
    let mut output = vec![0u8; block.original_len];
    let out_len = lz77::decompress_to_buf(&block.data, &mut output)?;
    if out_len != block.original_len {
        return Err(PzError::InvalidInput);
    }
    block.data = output;
    Ok(block)
}

/// Lza decompress stage 0: parse metadata + range decode.
fn stage_rangecoder_decode_lza(mut block: StageBlock) -> PzResult<StageBlock> {
    if block.data.len() < 4 {
        return Err(PzError::InvalidInput);
    }
    let lz_len =
        u32::from_le_bytes([block.data[0], block.data[1], block.data[2], block.data[3]]) as usize;
    block.metadata.lz_len = Some(lz_len);
    block.data = rangecoder::decode(&block.data[4..], lz_len)?;
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
        (Pipeline::Bw, 3) => stage_rangecoder_encode_bw(block),
        (Pipeline::Lza, 0) => stage_lz77_compress(block, options),
        (Pipeline::Lza, 1) => stage_rangecoder_encode_lza(block),
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
        // Bw: RC decode(0) → RLE decode(1) → MTF decode(2) → BWT decode(3)
        (Pipeline::Bw, 0) => stage_rangecoder_decode_bw(block),
        (Pipeline::Bw, 1) => stage_rle_decode(block),
        (Pipeline::Bw, 2) => stage_mtf_decode(block),
        (Pipeline::Bw, 3) => stage_bwt_decode(block),
        // Lza: RC decode(0) → LZ77 decompress(1)
        (Pipeline::Lza, 0) => stage_rangecoder_decode_lza(block),
        (Pipeline::Lza, 1) => stage_lz77_decompress(block),
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

    // Build V2 container from collected block data
    let mut block_data_vec: Vec<Vec<u8>> = Vec::with_capacity(num_blocks);
    for result in results {
        block_data_vec.push(result?);
    }

    let mut output = Vec::new();
    write_header_v2(&mut output, pipeline, input.len());
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

    // CPU paths
    match options.parse_strategy {
        ParseStrategy::Auto | ParseStrategy::Lazy => lz77::compress_lazy(input),
        ParseStrategy::Greedy => lz77::compress_hashchain(input),
        ParseStrategy::Optimal => crate::optimal::compress_optimal(input),
    }
}

// --- DEFLATE pipeline: LZ77 + Huffman ---

/// Compress a single block using the Deflate pipeline (no container header).
/// Returns pipeline-specific metadata + compressed data.
fn compress_block_deflate(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        metadata: StageMetadata::default(),
    };
    let block = stage_lz77_compress(block, options)?;
    let block = stage_huffman_encode(block)?;
    Ok(block.data)
}

fn compress_deflate_with_options(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block_data = compress_block_deflate(input, options)?;
    let mut output = Vec::new();
    write_header(&mut output, Pipeline::Deflate, input.len());
    output.extend_from_slice(&block_data);
    Ok(output)
}

/// Decompress a single Deflate block (no container header).
fn decompress_block_deflate(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    // Parse metadata: lz_len(4) + total_bits(4) + freq_table(1024) = 1032 bytes
    if payload.len() < 1032 {
        return Err(PzError::InvalidInput);
    }

    let lz_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let total_bits = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;

    // Reconstruct frequency table
    let mut freq_table = crate::frequency::FrequencyTable::new();
    for i in 0..256 {
        let offset = 8 + i * 4;
        freq_table.byte[i] = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
    }
    freq_table.total = freq_table.byte.iter().map(|&f| f as u64).sum();
    freq_table.used = freq_table.byte.iter().filter(|&&f| f > 0).count() as u32;

    let huffman_data = &payload[1032..];

    // Stage 1: Huffman decode
    let tree = HuffmanTree::from_frequency_table(&freq_table).ok_or(PzError::InvalidInput)?;
    let mut lz_data = vec![0u8; lz_len];
    let decoded_len = tree.decode_to_buf(huffman_data, total_bits, &mut lz_data)?;
    if decoded_len != lz_len {
        return Err(PzError::InvalidInput);
    }

    // Stage 2: LZ77 decompress
    let mut output = vec![0u8; orig_len];
    let out_len = lz77::decompress_to_buf(&lz_data, &mut output)?;
    if out_len != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

fn decompress_deflate(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    decompress_block_deflate(payload, orig_len)
}

// --- BWT helper: select GPU or CPU backend for suffix array construction ---

/// Run BWT encoding using the configured backend.
fn bwt_encode_with_backend(input: &[u8], options: &CompressOptions) -> PzResult<bwt::BwtResult> {
    #[cfg(feature = "opencl")]
    {
        if let Backend::OpenCl = options.backend {
            if let Some(ref engine) = options.opencl_engine {
                if input.len() >= crate::opencl::MIN_GPU_BWT_SIZE {
                    return engine.bwt_encode(input);
                }
            }
        }
    }

    #[cfg(not(feature = "opencl"))]
    let _ = options;

    bwt::encode(input).ok_or(PzError::InvalidInput)
}

// --- BW pipeline: BWT + MTF + RLE + Range coder ---

/// Compress a single block using the BW pipeline (no container header).
fn compress_block_bw(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        metadata: StageMetadata::default(),
    };
    let block = stage_bwt_encode(block, options)?;
    let block = stage_mtf_encode(block)?;
    let block = stage_rle_encode(block)?;
    let block = stage_rangecoder_encode_bw(block)?;
    Ok(block.data)
}

fn compress_bw_with_options(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block_data = compress_block_bw(input, options)?;
    let mut output = Vec::new();
    write_header(&mut output, Pipeline::Bw, input.len());
    output.extend_from_slice(&block_data);
    Ok(output)
}

/// Decompress a single BW block (no container header).
fn decompress_block_bw(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 8 {
        return Err(PzError::InvalidInput);
    }

    let primary_index = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let rle_len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;

    let rc_data = &payload[8..];

    // Stage 1: Range decoder
    let rle_data = rangecoder::decode(rc_data, rle_len)?;

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

fn decompress_bw(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    decompress_block_bw(payload, orig_len)
}

// --- LZA pipeline: LZ77 + Range coder ---

/// Compress a single block using the LZA pipeline (no container header).
fn compress_block_lza(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        metadata: StageMetadata::default(),
    };
    let block = stage_lz77_compress(block, options)?;
    let block = stage_rangecoder_encode_lza(block)?;
    Ok(block.data)
}

fn compress_lza_with_options(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block_data = compress_block_lza(input, options)?;
    let mut output = Vec::new();
    write_header(&mut output, Pipeline::Lza, input.len());
    output.extend_from_slice(&block_data);
    Ok(output)
}

/// Decompress a single LZA block (no container header).
fn decompress_block_lza(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let lz_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let rc_data = &payload[4..];

    // Stage 1: Range decode
    let lz_data = rangecoder::decode(rc_data, lz_len)?;

    // Stage 2: LZ77 decompress
    let mut output = vec![0u8; orig_len];
    let out_len = lz77::decompress_to_buf(&lz_data, &mut output)?;
    if out_len != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

fn decompress_lza(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    decompress_block_lza(payload, orig_len)
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

    // --- LZA pipeline tests ---

    #[test]
    fn test_lza_empty() {
        let result = compress(&[], Pipeline::Lza).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_lza_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Lza).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lza_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Lza).unwrap();
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
    fn test_lza_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Lza).unwrap();
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
        let result = decompress(&[b'P', b'Z', 1, 99, 0, 0, 0, 0]);
        assert_eq!(result, Err(PzError::Unsupported));
    }

    #[test]
    fn test_too_short_input() {
        let result = decompress(b"PZ");
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_zero_original_length() {
        let result = decompress(&[b'P', b'Z', 1, 0, 0, 0, 0, 0]);
        assert_eq!(result.unwrap(), Vec::<u8>::new());
    }

    // --- Cross-pipeline tests ---

    #[test]
    fn test_all_pipelines_banana() {
        let input = b"banana banana banana banana banana";
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
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
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
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

        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
            let compressed = compress_mt(&input, pipeline, 4, 512).unwrap();
            // Should be V2 format
            assert_eq!(compressed[2], VERSION_V2, "expected V2 for {:?}", pipeline);
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(decompressed, input, "round-trip failed for {:?}", pipeline);
        }
    }

    #[test]
    fn test_multiblock_single_block_fallback() {
        // Input smaller than block_size → should use single-block V1 format
        let input = b"small input data";
        let compressed = compress_mt(input, Pipeline::Deflate, 4, 65536).unwrap();
        assert_eq!(compressed[2], VERSION_V1, "expected V1 for small input");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multiblock_single_thread() {
        // threads=1 → always single-block V1 format
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_mt(&input, Pipeline::Bw, 1, 512).unwrap();
        assert_eq!(compressed[2], VERSION_V1, "expected V1 for single-threaded");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_multiblock_v1_backward_compat() {
        // Compress with V1 (single-threaded), verify decompression works
        let input = b"hello, world! hello, world!";
        let compressed_v1 = compress(input, Pipeline::Bw).unwrap();
        assert_eq!(compressed_v1[2], VERSION_V1);
        let decompressed = decompress(&compressed_v1).unwrap();
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
            for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
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
        // Input exactly equal to block_size → single block, should use V1
        let input = vec![b'x'; 1024];
        let compressed = compress_mt(&input, Pipeline::Bw, 4, 1024).unwrap();
        assert_eq!(compressed[2], VERSION_V1, "exact one block should be V1");
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
        let compressed = compress_mt(&input, Pipeline::Lza, 2, 1024).unwrap();
        assert_eq!(compressed[2], VERSION_V2);
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
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
            let compressed = compress_mt(&input, pipeline, 4, 16384).unwrap();
            assert_eq!(compressed[2], VERSION_V2);
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
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
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

        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
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

        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
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
        let compressed = compress_with_options(&input, Pipeline::Lza, &opts).unwrap();
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
}
