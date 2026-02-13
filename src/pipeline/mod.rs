//! Compression pipeline orchestrator.
//!
//! Chains algorithm stages together to form complete compression pipelines.
//! Each pipeline defines a sequence of transforms and entropy coding
//! stages that are applied in order for compression, and in reverse for
//! decompression.
//!
//! **Supported pipelines:**
//!
//! | Pipeline      | Stages                           | Similar to      |
//! |---------------|----------------------------------|-----------------|
//! | `Deflate`     | LZ77 → Huffman                   | gzip            |
//! | `Bw`          | BWT → MTF → RLE → FSE            | bzip2           |
//! | `Lzr`         | LZ77 → rANS                      | fast ANS        |
//! | `Lzf`         | LZ77 → FSE                       | zstd-like       |
//! | `Lzfi`        | LZ77 → interleaved FSE           | GPU FSE decode  |
//! | `LzssR`       | LZSS → rANS                      | experimental    |
//! | `Bwi`         | BWT → MTF → RLE → interleaved FSE| GPU FSE decode  |
//! | `Lz78R`       | LZ78 → rANS                      | experimental    |
//!
//! **Container format (V2, multi-block):**
//! Each compressed stream starts with a header:
//! - Magic bytes: `PZ` (2 bytes)
//! - Version: 2 (1 byte)
//! - Pipeline ID: 0=Deflate, 1=Bw, 3=Lzr, 4=Lzf, 5=Lzfi, 6=LzssR, 7=Bwi, 8=Lz78R (1 byte)
//! - Original length: u32 little-endian (4 bytes)
//! - num_blocks: u32 little-endian (4 bytes)
//! - Block table: \[compressed_len: u32, original_len: u32\] \* num_blocks
//! - Block data: concatenated compressed block bytes

mod blocks;
mod demux;
mod parallel;
mod stages;

use crate::bwt;
use crate::lz77;
use crate::{PzError, PzResult};

pub(crate) use blocks::{compress_block, decompress_block};
use parallel::{
    compress_parallel, compress_pipeline_parallel, decompress_parallel,
    decompress_pipeline_parallel,
};
use stages::pipeline_stage_count;

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
    /// Maximum match length for LZ77 compression.
    ///
    /// `None` = use the pipeline's default: 258 for Deflate (RFC 1951
    /// constraint), `u16::MAX` for other LZ77-based pipelines (Lzr, Lzf).
    /// Larger limits allow longer matches on repetitive data without
    /// penalizing short-match performance (SIMD short-circuits).
    pub max_match_len: Option<u16>,
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
            max_match_len: None,
            #[cfg(feature = "opencl")]
            opencl_engine: None,
            #[cfg(feature = "webgpu")]
            webgpu_engine: None,
        }
    }
}

/// Options for GPU-accelerated decompression.
///
/// By default, decompression is CPU-only. To enable GPU-accelerated FSE
/// decode (for `Lzfi`/`Bwi` pipelines), provide a WebGPU engine handle.
#[derive(Clone, Default)]
pub struct DecompressOptions {
    /// Which backend to use for GPU-amenable decode stages.
    pub backend: Backend,
    /// Number of threads for multi-block decompression. 0 = auto.
    pub threads: usize,
    /// WebGPU engine handle for GPU FSE decode.
    #[cfg(feature = "webgpu")]
    pub webgpu_engine: Option<std::sync::Arc<crate::webgpu::WebGpuEngine>>,
}

/// Resolve the effective max match length from options and pipeline type.
///
/// Deflate is hard-capped at 258 (RFC 1951). Other LZ77-based pipelines
/// default to `u16::MAX` for better compression on repetitive data.
pub(crate) fn resolve_max_match_len(pipeline: Pipeline, options: &CompressOptions) -> u16 {
    options.max_match_len.unwrap_or(match pipeline {
        Pipeline::Deflate => lz77::DEFLATE_MAX_MATCH,
        _ => lz77::DEFAULT_MAX_MATCH,
    })
}

/// Magic bytes for the libpz container format.
pub(crate) const MAGIC: [u8; 2] = [b'P', b'Z'];
/// Format version for multi-block streams.
pub(crate) const VERSION: u8 = 2;

/// Minimum header size: magic(2) + version(1) + pipeline(1) + orig_len(4) = 8
const MIN_HEADER_SIZE: usize = 8;

/// Sentinel value for `num_blocks` indicating framed (streaming) mode.
///
/// When `num_blocks == FRAMED_SENTINEL`, blocks are self-framing: each block
/// is preceded by its compressed_len and original_len, and the stream ends
/// with a 4-byte EOS sentinel (compressed_len = 0). This allows streaming
/// compression and decompression without knowing block count upfront.
pub(crate) const FRAMED_SENTINEL: u32 = 0xFFFF_FFFF;

/// Size of a block table entry: compressed_len(4) + original_len(4) = 8 bytes.
pub(crate) const BLOCK_HEADER_SIZE: usize = 8;

/// Compression pipeline types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pipeline {
    /// LZ77 + Huffman (gzip-like)
    Deflate = 0,
    /// BWT + MTF + RLE + FSE (bzip2-like)
    Bw = 1,
    /// Bijective BWT + MTF + RLE + FSE (parallelizable BWT variant)
    Bbw = 2,
    /// LZ77 + rANS (fast entropy coding, SIMD/GPU friendly)
    Lzr = 3,
    /// LZ77 + FSE (finite state entropy, zstd-style)
    Lzf = 4,
    /// LZ77 + interleaved FSE (N-way parallel FSE, GPU-decodable)
    Lzfi = 5,
    /// LZSS + rANS (flag-bit LZ + arithmetic ANS, experimental)
    LzssR = 6,
    /// BWT + MTF + RLE + interleaved FSE (GPU-decodable entropy)
    Bwi = 7,
    /// LZ78 + rANS (incremental trie + rANS, experimental)
    Lz78R = 8,
}

impl TryFrom<u8> for Pipeline {
    type Error = PzError;

    fn try_from(v: u8) -> Result<Self, Self::Error> {
        match v {
            0 => Ok(Self::Deflate),
            1 => Ok(Self::Bw),
            2 => Ok(Self::Bbw),
            3 => Ok(Self::Lzr),
            4 => Ok(Self::Lzf),
            5 => Ok(Self::Lzfi),
            6 => Ok(Self::LzssR),
            7 => Ok(Self::Bwi),
            8 => Ok(Self::Lz78R),
            _ => Err(PzError::Unsupported),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
    //
    // Block parallelism (one thread per block, each block runs all stages) scales
    // with available cores and is preferred when we have more threads than stages.
    //
    // Pipeline parallelism (one thread per stage, blocks flow through channels) is
    // only beneficial when the stage count is high enough to saturate available
    // threads (e.g., the 4-stage Bw pipeline on <=4 cores).
    let num_blocks = input.len().div_ceil(block_size);
    let stage_count = pipeline_stage_count(pipeline);

    if num_blocks > 1 && stage_count >= num_threads {
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
    decompress_with_options(input, &DecompressOptions::default())
}

/// Decompress data with an explicit thread count.
///
/// `threads`: 0 = auto, 1 = single-threaded, N = use up to N threads.
pub fn decompress_with_threads(input: &[u8], threads: usize) -> PzResult<Vec<u8>> {
    decompress_with_options(
        input,
        &DecompressOptions {
            threads,
            ..Default::default()
        },
    )
}

/// Decompress data with full options (GPU backend, thread count).
///
/// Use this to enable GPU-accelerated FSE decode for `Lzfi`/`Bwi` pipelines.
pub fn decompress_with_options(input: &[u8], options: &DecompressOptions) -> PzResult<Vec<u8>> {
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

    let payload = &input[MIN_HEADER_SIZE..];

    if payload.len() < 4 {
        // No payload at all: only valid if orig_len is zero (empty input).
        if orig_len == 0 {
            return Ok(Vec::new());
        }
        return Err(PzError::InvalidInput);
    }
    let num_blocks_raw = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);

    // Framed (streaming) mode: blocks are self-framing with inline headers.
    // Must check before the orig_len == 0 short-circuit because streaming
    // uses orig_len = 0 to mean "unknown length".
    if num_blocks_raw == FRAMED_SENTINEL {
        return decompress_framed(&payload[4..], pipeline, orig_len, options);
    }

    if orig_len == 0 {
        return Ok(Vec::new());
    }

    let num_blocks = num_blocks_raw as usize;
    if num_blocks == 0 {
        return Err(PzError::InvalidInput);
    }

    let num_threads = resolve_thread_count(options.threads);
    let stage_count = pipeline_stage_count(pipeline);

    if num_threads > 1 && num_blocks > 1 && stage_count >= num_threads {
        return decompress_pipeline_parallel(payload, pipeline, orig_len, num_blocks, options);
    }
    decompress_parallel(payload, pipeline, orig_len, num_blocks, options)
}

// ---------------------------------------------------------------------------
// Auto-selection
// ---------------------------------------------------------------------------

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
        Pipeline::Lzfi,
        Pipeline::LzssR,
        Pipeline::Bwi,
        Pipeline::Lz78R,
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

// ---------------------------------------------------------------------------
// Container format helpers
// ---------------------------------------------------------------------------

/// Write the container header to output.
pub(crate) fn write_header(output: &mut Vec<u8>, pipeline: Pipeline, orig_len: usize) {
    output.extend_from_slice(&MAGIC);
    output.push(VERSION);
    output.push(pipeline as u8);
    output.extend_from_slice(&(orig_len as u32).to_le_bytes());
}

/// Resolve thread count: 0 = auto (available_parallelism), otherwise use the given value.
pub(crate) fn resolve_thread_count(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        threads
    }
}

/// Decompress framed (streaming) data from an in-memory buffer.
///
/// Framed format: repeated `[compressed_len: u32][original_len: u32][data]`
/// terminated by a 4-byte EOS sentinel (compressed_len = 0).
fn decompress_framed(
    data: &[u8],
    pipeline: Pipeline,
    declared_orig_len: usize,
    options: &DecompressOptions,
) -> PzResult<Vec<u8>> {
    let mut output = Vec::new();
    let mut pos = 0;

    loop {
        if pos + 4 > data.len() {
            return Err(PzError::InvalidInput);
        }
        let compressed_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        // EOS sentinel
        if compressed_len == 0 {
            break;
        }

        if pos + 4 > data.len() {
            return Err(PzError::InvalidInput);
        }
        let original_len =
            u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;

        if pos + compressed_len > data.len() {
            return Err(PzError::InvalidInput);
        }
        let block_data = &data[pos..pos + compressed_len];
        pos += compressed_len;

        let decompressed = blocks::decompress_block(block_data, pipeline, original_len, options)?;
        output.extend_from_slice(&decompressed);
    }

    // Validate total length if declared (non-zero orig_len in header).
    if declared_orig_len > 0 && output.len() != declared_orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// GPU/CPU backend dispatch helpers
// ---------------------------------------------------------------------------

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
    let max_match = options.max_match_len.unwrap_or(lz77::DEFLATE_MAX_MATCH);
    match options.parse_strategy {
        ParseStrategy::Auto | ParseStrategy::Lazy => {
            lz77::compress_lazy_with_limit(input, max_match)
        }
        ParseStrategy::Optimal => crate::optimal::compress_optimal_with_limit(input, max_match),
    }
}

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

/// Run bijective BWT encoding using the configured backend.
fn bbwt_encode_with_backend(
    input: &[u8],
    options: &CompressOptions,
) -> PzResult<(Vec<u8>, Vec<usize>)> {
    #[cfg(feature = "webgpu")]
    {
        if let Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                if !engine.is_cpu_device()
                    && input.len() >= crate::webgpu::MIN_GPU_BWT_SIZE
                    && input.len() <= engine.max_dispatch_input_size()
                {
                    return engine.bwt_encode_bijective(input);
                }
            }
        }
    }

    #[cfg(not(feature = "webgpu"))]
    let _ = options;

    bwt::encode_bijective(input).ok_or(PzError::InvalidInput)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
