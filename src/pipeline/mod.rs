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
//! | `Lzfi`        | LZSS → interleaved FSE           | fast CPU decode |
//! | `LzssR`       | LZSS → rANS                      | experimental    |
//! | `Lz78R`       | LZ78 → rANS                      | experimental    |
//! | `LzSeqR`      | LzSeq → rANS                     | zstd-style      |
//! | `LzSeqH`      | LzSeq → Huffman                  | fast decode     |
//! | `SortLz`      | SortLZ → FSE                     | GPU match find  |
//!
//! **Match finder selection:** All LZ-based pipelines accept `MatchFinder::SortLz`
//! as an alternative to `MatchFinder::HashChain`. When SortLz is used as a match
//! finder, the wire format is that of the host pipeline (fully compatible).
//!
//! **Container format (V2, multi-block):**
//! Each compressed stream starts with a header:
//! - Magic bytes: `PZ` (2 bytes)
//! - Version: 2 (1 byte)
//! - Pipeline ID: 0=Deflate, 1=Bw, 3=Lzr, 4=Lzf, 5=Lzfi, 6=LzssR, 7=Lz78R, 8=LzSeqR, 9=LzSeqH, 10=SortLz (1 byte)
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
pub use parallel::UnifiedSchedulerStats;
use parallel::{compress_parallel, decompress_parallel};

/// Compute backend selection for pipeline stages.
///
/// Each pipeline stage can run on different backends depending on
/// hardware availability and input characteristics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Backend {
    /// Single-threaded CPU reference implementation (always available).
    #[default]
    Cpu,
    /// WebGPU backend via wgpu (requires `webgpu` feature).
    #[cfg(feature = "webgpu")]
    WebGpu,
}

/// Per-stage compute backend override.
///
/// Controls which backend executes a specific pipeline stage.
/// `Auto` lets the scheduler decide based on block size and GPU availability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BackendAssignment {
    /// Scheduler chooses: GPU entropy for blocks >= GPU_ENTROPY_THRESHOLD,
    /// CPU for smaller blocks or when no GPU is available.
    #[default]
    Auto,
    /// Force CPU execution for this stage (always available, zero-overhead).
    Cpu,
    /// Force GPU execution for this stage (requires `webgpu` feature and a device).
    #[cfg(feature = "webgpu")]
    Gpu,
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
    /// Greedy: take the longest match at every position.
    Greedy,
    /// Lazy: check if the next position has a longer match (gzip-style).
    Lazy,
    /// Optimal: backward DP to minimize total encoding cost.
    Optimal,
}

/// Match-finding algorithm selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MatchFinder {
    /// Hash-chain match finder (default). Uses sliding-window hash
    /// chains with SIMD-accelerated comparison.
    #[default]
    HashChain,
    /// Sort-based match finder (SortLZ). Uses radix sort of (hash, position)
    /// pairs followed by adjacent-pair verification. Fully deterministic,
    /// GPU-friendly. Minimum match length is 4 (vs 3 for hash-chain).
    SortLz,
}

/// Compression quality preset for LzSeq pipelines.
///
/// Maps to `parse_strategy` and hash chain depth. Higher quality = better ratio
/// at the cost of more CPU time. Only affects LzSeqR and LzSeqH.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QualityLevel {
    /// Speed mode: lazy matching, shallow chain depth.
    Speed,
    /// Default mode: optimal parsing with standard chain depth.
    #[default]
    Default,
    /// Quality mode: optimal parsing with deep hash chains.
    Quality,
}

/// Default block size for multi-threaded compression (256KB).
const DEFAULT_BLOCK_SIZE: usize = 256 * 1024;

/// Default block size for BWT-based pipelines (512KB).
///
/// BWT benefits from larger blocks (better context grouping), but our FSE
/// encoder degrades beyond ~1MB. 512KB balances BWT context quality against
/// FSE table precision. Empirically optimal across Canterbury+Silesia corpus.
const DEFAULT_BW_BLOCK_SIZE: usize = 512 * 1024;

/// Default block size for GPU LZ77 pipelines (128KB).
///
/// The GPU hash table produces significantly better matches at ≤128KB than at
/// larger sizes, because bucket overflow degrades match quality. Empirically,
/// GPU match quality matches CPU lazy at 128KB but collapses at 256KB+.
/// Using smaller blocks creates more work items for GPU/CPU overlap pipelining,
/// which also improves throughput on streaming GPU paths.
const DEFAULT_GPU_BLOCK_SIZE: usize = 128 * 1024;

/// Minimum block size for GPU entropy to win over CPU (empirical from Phase 4).
/// Applies to both individual blocks and total stream byte count.
/// 256KB = 262144 bytes (aligns with AC3.2 threshold).
pub const GPU_ENTROPY_THRESHOLD: usize = 256 * 1024;

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
    /// WebGPU engine handle, required when `backend` is `Backend::WebGpu`.
    #[cfg(feature = "webgpu")]
    pub webgpu_engine: Option<std::sync::Arc<crate::webgpu::WebGpuEngine>>,
    /// Enable interleaved rANS for rANS-based pipelines when stream size permits.
    ///
    /// When enabled, streams with length >= `rans_interleaved_min_bytes` are encoded
    /// with `rans::encode_interleaved_n` using `rans_interleaved_states`.
    pub rans_interleaved: bool,
    /// Minimum stream size for interleaved rANS encode.
    pub rans_interleaved_min_bytes: usize,
    /// Number of interleaved rANS states when interleaved mode is active.
    pub rans_interleaved_states: usize,
    /// Enable Recoil split-point metadata for parallel rANS decode.
    ///
    /// When enabled (and `rans_interleaved` is also enabled), the encoder
    /// generates split-point metadata allowing the decoder to parallelize
    /// across `rans_recoil_splits` independent segments without re-encoding.
    pub rans_recoil: bool,
    /// Number of Recoil split points to generate (default 64).
    pub rans_recoil_splits: usize,
    /// LzSeq sliding window size in bytes. Must be a power of 2.
    ///
    /// `None` = use the default (128KB). Only affects the `LzSeqR`/`LzSeqH` pipelines;
    /// other pipelines ignore this. Larger windows find longer-range matches
    /// at the cost of more memory (4 bytes per window position for the
    /// hash-chain `prev` array).
    pub seq_window_size: Option<usize>,
    /// Backend assignment for stage 0 (match finding / transform).
    /// Match finding is always CPU — LZ77's sequential dependency rules out GPU.
    /// For BWT-based pipelines, Auto routes to GPU.
    pub stage0_backend: BackendAssignment,
    /// Backend assignment for stage 1 (entropy coding).
    /// Auto routes to GPU when block size >= GPU_ENTROPY_THRESHOLD and GPU available.
    pub stage1_backend: BackendAssignment,
    /// Match-finding algorithm: HashChain (default) or SortLz.
    /// Applies to all LZ-based pipelines (Deflate, Lzr, Lzf, Lzfi, LzssR, LzSeqR, LzSeqH).
    pub match_finder: MatchFinder,
}

impl Default for CompressOptions {
    fn default() -> Self {
        CompressOptions {
            backend: Backend::Cpu,
            threads: 0,
            block_size: DEFAULT_BLOCK_SIZE,
            parse_strategy: ParseStrategy::Auto,
            max_match_len: None,
            #[cfg(feature = "webgpu")]
            webgpu_engine: None,
            rans_interleaved: false,
            rans_interleaved_min_bytes: 64 * 1024,
            rans_interleaved_states: crate::rans::DEFAULT_INTERLEAVE,
            rans_recoil: false,
            rans_recoil_splits: 64,
            seq_window_size: None,
            stage0_backend: BackendAssignment::Auto,
            stage1_backend: BackendAssignment::Auto,
            match_finder: MatchFinder::HashChain,
        }
    }
}

impl CompressOptions {
    /// Build options for the given quality level (LzSeq pipelines).
    pub fn for_quality(level: QualityLevel) -> Self {
        let mut opts = CompressOptions::default();
        match level {
            QualityLevel::Speed => {
                opts.parse_strategy = ParseStrategy::Lazy;
            }
            QualityLevel::Default => {
                opts.parse_strategy = ParseStrategy::Optimal;
            }
            QualityLevel::Quality => {
                opts.parse_strategy = ParseStrategy::Optimal;
                // Deep chain: use a larger window for quality mode
                opts.seq_window_size = Some(256 * 1024);
            }
        }
        opts
    }
}

/// Options controlling pipeline decompression behavior.
///
/// Carries the GPU backend and engine handles so that decompression stages
/// can dispatch to GPU decode kernels when available.
#[derive(Debug, Clone)]
pub struct DecompressOptions {
    /// Which backend to use for GPU-amenable decode stages.
    pub backend: Backend,
    /// Number of threads to use. 0 = auto, 1 = single-threaded.
    pub threads: usize,
    /// WebGPU engine handle, required when `backend` is `Backend::WebGpu`.
    #[cfg(feature = "webgpu")]
    pub webgpu_engine: Option<std::sync::Arc<crate::webgpu::WebGpuEngine>>,
}

impl Default for DecompressOptions {
    fn default() -> Self {
        DecompressOptions {
            backend: Backend::Cpu,
            threads: 0,
            #[cfg(feature = "webgpu")]
            webgpu_engine: None,
        }
    }
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
    /// LZSS + interleaved FSE (N-way parallel FSE, fast CPU decode)
    Lzfi = 5,
    /// LZSS + rANS (flag-bit LZ + arithmetic ANS, experimental)
    LzssR = 6,
    /// LZ78 + rANS (incremental trie + rANS, experimental)
    Lz78R = 7,
    /// LzSeq + rANS (code+extra-bits sequence encoding, zstd-style)
    LzSeqR = 8,
    /// LzSeq + Huffman (fast decode, simpler entropy coding)
    LzSeqH = 9,
    /// Sort-based LZ77 + FSE (deterministic GPU match finding, experimental)
    SortLz = 10,
    /// Parallel-parse LZ + FSE (serial parsing cost experiment)
    Parlz = 11,
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
            7 => Ok(Self::Lz78R),
            8 => Ok(Self::LzSeqR),
            9 => Ok(Self::LzSeqH),
            10 => Ok(Self::SortLz),
            11 => Ok(Self::Parlz),
            _ => Err(PzError::Unsupported),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Enable or disable unified scheduler telemetry collection.
///
/// Disabled by default to avoid profiling overhead in normal runs.
pub fn set_unified_scheduler_stats_enabled(enabled: bool) {
    parallel::set_unified_scheduler_stats_enabled(enabled);
}

/// Reset all aggregated unified scheduler telemetry counters/timers.
pub fn reset_unified_scheduler_stats() {
    parallel::reset_unified_scheduler_stats();
}

/// Snapshot the current aggregated unified scheduler telemetry.
pub fn unified_scheduler_stats() -> UnifiedSchedulerStats {
    parallel::unified_scheduler_stats()
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
/// When `options.backend` is `Backend::WebGpu` and an engine is provided,
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

    // Adjust block size for pipeline characteristics when the caller
    // is using defaults: BW→512KB, GPU LZ→128KB.
    let options = &adjusted_options(pipeline, options);

    let num_threads = resolve_thread_count(options.threads);
    let block_size = options.block_size;

    // Single-block fast path: input fits in one block.
    if input.len() <= block_size {
        let block_data = compress_block(input, pipeline, options)?;
        let mut output = Vec::new();
        write_header(&mut output, pipeline, input.len());
        output.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        output.extend_from_slice(&(block_data.len() as u32).to_le_bytes()); // compressed_len
        output.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original_len
        output.extend_from_slice(&block_data);
        return Ok(output);
    }

    // Single-threaded multi-block: split into blocks and compress sequentially.
    // This matches the streaming path's behavior and is critical for BWT-based
    // pipelines where FSE degrades on large inputs (>512KB).
    if num_threads <= 1 {
        let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
        let num_blocks = blocks.len();

        // Compress all blocks
        let mut compressed_blocks = Vec::with_capacity(num_blocks);
        for block in &blocks {
            compressed_blocks.push(compress_block(block, pipeline, options)?);
        }

        // Build output with V2 table-mode container
        let mut output = Vec::new();
        write_header(&mut output, pipeline, input.len());
        output.extend_from_slice(&(num_blocks as u32).to_le_bytes());

        // Block table
        for (i, cb) in compressed_blocks.iter().enumerate() {
            output.extend_from_slice(&(cb.len() as u32).to_le_bytes());
            output.extend_from_slice(&(blocks[i].len() as u32).to_le_bytes());
        }

        // Block data
        for cb in &compressed_blocks {
            output.extend_from_slice(cb);
        }

        return Ok(output);
    }

    // Multi-block parallel: unified scheduler dispatches all stages from a shared work queue.
    compress_parallel(input, pipeline, options, num_threads)
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
            ..DecompressOptions::default()
        },
    )
}

/// Decompress data with full options including GPU backend selection.
///
/// When `options.backend` is `Backend::WebGpu` with an engine,
/// GPU-amenable decode stages (e.g., interleaved FSE for Lzfi) run on the
/// GPU. Other stages and pipelines use the CPU.
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
            return Pipeline::Lzfi;
        }
        return Pipeline::Deflate;
    }

    // Moderate match density with high entropy
    if profile.match_density > 0.2 && profile.byte_entropy > 5.0 {
        return Pipeline::Lzfi;
    }

    // Default: Deflate (fast, decent compression)
    Pipeline::Deflate
}

/// Select the best match finder for the given input and options.
///
/// Returns `MatchFinder::SortLz` when data characteristics favor exhaustive
/// sort-based match finding (text, structured data with moderate-to-high
/// match density and moderate entropy). Returns `MatchFinder::HashChain`
/// for near-random data, very small inputs, or when match finding speed
/// matters more than ratio.
///
/// When a GPU is available and the input is large enough, always prefers
/// SortLz since the GPU radix sort amortizes its cost at scale.
pub fn select_match_finder(input: &[u8], options: &CompressOptions) -> MatchFinder {
    // Tiny inputs: hashchain is fine, SortLZ overhead not worth it
    if input.len() < 1024 {
        return MatchFinder::HashChain;
    }

    let profile = crate::analysis::analyze(input);

    // Near-random data: hashchain is fine, SortLZ adds cost without ratio benefit
    if profile.byte_entropy > 7.5 {
        return MatchFinder::HashChain;
    }

    // GPU available: use SortLZ when input is large enough to amortize dispatch
    #[cfg(feature = "webgpu")]
    if matches!(options.backend, Backend::WebGpu) {
        if let Some(ref engine) = options.webgpu_engine {
            if input.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE
                && input.len() <= engine.max_dispatch_input_size()
            {
                return MatchFinder::SortLz;
            }
        }
    }

    // CPU: SortLZ shines on text/structured data where exhaustive match
    // finding discovers long-range matches that hash-chains miss.
    // Thresholds derived from experiment B results:
    //   - bible.txt: match_density ~0.5, entropy ~4.6 → SortLZ 18% better
    //   - world192.txt: match_density ~0.4, entropy ~5.1 → SortLZ 24% better
    //   - random data: match_density <0.1, entropy >7.5 → no benefit
    if profile.match_density > 0.3 && profile.byte_entropy < 6.5 {
        return MatchFinder::SortLz;
    }

    // Large inputs with moderate structure: SortLZ's global sort still helps
    if input.len() >= 64 * 1024 && profile.match_density > 0.2 && profile.byte_entropy < 5.5 {
        return MatchFinder::SortLz;
    }

    // Suppress unused variable warning when webgpu is disabled
    let _ = options;

    MatchFinder::HashChain
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
        Pipeline::Lzfi,
        Pipeline::LzssR,
        Pipeline::Lz78R,
        Pipeline::LzSeqR,
        Pipeline::LzSeqH,
        Pipeline::SortLz,
    ];
    let mut best_pipeline = Pipeline::Deflate;
    let mut best_size = usize::MAX;

    // Also try SortLz match finder with LZ pipelines to find the best combo
    let match_finders = [MatchFinder::HashChain, MatchFinder::SortLz];

    for &pipeline in &candidates {
        // SortLz pipeline has its own match finder, only test default
        let finders: &[MatchFinder] = if matches!(
            pipeline,
            Pipeline::Bw | Pipeline::Bbw | Pipeline::Lz78R | Pipeline::SortLz
        ) {
            &[MatchFinder::HashChain]
        } else {
            &match_finders
        };

        for &finder in finders {
            let opts = CompressOptions {
                match_finder: finder,
                ..trial_opts.clone()
            };
            if let Ok(compressed) = compress_with_options(sample, pipeline, &opts) {
                if compressed.len() < best_size {
                    best_size = compressed.len();
                    best_pipeline = pipeline;
                }
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

/// Return options with pipeline-optimal block size.
///
/// Adjusts block size based on pipeline characteristics when the caller
/// is using the default (256KB):
/// - BWT pipelines (Bw, Bbw): use 512KB for better BWT context grouping
/// - GPU LZ77 pipelines: use 128KB for GPU hash table quality
///
/// If the caller explicitly set a non-default block size, their choice is
/// respected.
fn adjusted_options(pipeline: Pipeline, options: &CompressOptions) -> CompressOptions {
    if options.block_size != DEFAULT_BLOCK_SIZE {
        return options.clone();
    }

    let is_bw_pipeline = matches!(pipeline, Pipeline::Bw | Pipeline::Bbw);
    if is_bw_pipeline {
        let mut adjusted = options.clone();
        adjusted.block_size = DEFAULT_BW_BLOCK_SIZE;
        return adjusted;
    }

    let is_lz_pipeline = matches!(
        pipeline,
        Pipeline::Deflate
            | Pipeline::Lzr
            | Pipeline::Lzf
            | Pipeline::Lzfi
            | Pipeline::LzssR
            | Pipeline::LzSeqR
            | Pipeline::LzSeqH
    );
    let is_gpu = {
        #[allow(unused_mut)]
        let mut gpu = false;
        #[cfg(feature = "webgpu")]
        if matches!(options.backend, Backend::WebGpu) {
            gpu = true;
        }
        gpu
    };

    if is_lz_pipeline && is_gpu {
        let mut adjusted = options.clone();
        adjusted.block_size = DEFAULT_GPU_BLOCK_SIZE;
        adjusted
    } else {
        options.clone()
    }
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

/// Demux helpers can use this for parse-mode-aware LZ77 match generation.
///
/// Backend/strategy selection logic:
/// - `Auto` on GPU: hash-table kernel (best throughput at ≥256KB)
/// - `Auto` on CPU: lazy matching (best compression ratio)
/// - `Optimal` on GPU: GPU top-K match table → CPU backward DP
/// - `Optimal` on CPU: CPU match table → CPU backward DP
/// - GPU backend falls back to CPU when input < MIN_GPU_INPUT_SIZE
pub(crate) fn lz77_matches_with_backend(
    input: &[u8],
    options: &CompressOptions,
) -> PzResult<Vec<lz77::Match>> {
    let max_match = options.max_match_len.unwrap_or(lz77::DEFLATE_MAX_MATCH);

    // GPU path: use GPU kernels when available and input is in range.
    #[cfg(feature = "webgpu")]
    {
        if let Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                if input.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE
                    && input.len() <= engine.max_dispatch_input_size()
                    && options.match_finder != MatchFinder::SortLz
                {
                    if options.parse_strategy == ParseStrategy::Optimal {
                        let table = engine.find_topk_matches(input)?;
                        let bytes = crate::optimal::compress_optimal_with_table(input, &table)?;
                        return Ok(lz77::deserialize_matches(&bytes));
                    }
                    let bytes = engine.lz77_compress(input)?;
                    return Ok(lz77::deserialize_matches(&bytes));
                }
            }
        }
    }

    // SortLZ match finder: radix sort + adjacent-pair verification
    if options.match_finder == MatchFinder::SortLz {
        return sortlz_matches_with_strategy(input, options, max_match);
    }

    // Hash-chain match finder (default)
    match options.parse_strategy {
        ParseStrategy::Auto => {
            let max_chain = lz77::select_chain_depth(input.len(), true);
            lz77::compress_lazy_to_matches_with_limit_and_chain(input, max_match, max_chain)
        }
        ParseStrategy::Greedy => lz77::compress_greedy_to_matches_with_limit(input, max_match),
        ParseStrategy::Lazy => lz77::compress_lazy_to_matches_with_limit(input, max_match),
        ParseStrategy::Optimal => crate::optimal::optimal_matches_with_limit(input, max_match),
    }
}

/// SortLZ match finding with parse strategy dispatch.
///
/// Uses GPU radix sort when a WebGPU engine is available and input is large
/// enough. Falls back to CPU sort otherwise.
fn sortlz_matches_with_strategy(
    input: &[u8],
    options: &CompressOptions,
    max_match: u16,
) -> PzResult<Vec<lz77::Match>> {
    use crate::sortlz::{self, SortLzConfig};

    let config = SortLzConfig::for_lz77(max_match);

    // GPU path: use GPU radix sort for match finding when available.
    // The GPU kernel does the sort + verify in parallel, then we convert
    // to lz77::Match on CPU (parsing is inherently serial).
    #[cfg(feature = "webgpu")]
    if let Some(ref engine) = options.webgpu_engine {
        if input.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE
            && input.len() <= engine.max_dispatch_input_size()
        {
            let raw_matches = engine.sortlz_find_matches(input, &config)?;
            return match options.parse_strategy {
                ParseStrategy::Greedy => Ok(sortlz::matches_to_lz77_greedy(input, &raw_matches)),
                _ => Ok(sortlz::matches_to_lz77_lazy(input, &raw_matches)),
            };
        }
    }

    // CPU path.
    match options.parse_strategy {
        ParseStrategy::Optimal => {
            let table = sortlz::find_matches_topk(input, &config, crate::optimal::K);
            let freq = crate::frequency::get_frequency(input);
            let cost_model = crate::optimal::CostModel::from_frequencies(&freq);
            Ok(crate::optimal::optimal_parse(input, &table, &cost_model))
        }
        ParseStrategy::Greedy => {
            let matches = sortlz::find_matches(input, &config);
            Ok(sortlz::matches_to_lz77_greedy(input, &matches))
        }
        ParseStrategy::Auto | ParseStrategy::Lazy => {
            let matches = sortlz::find_matches(input, &config);
            Ok(sortlz::matches_to_lz77_lazy(input, &matches))
        }
    }
}

/// Run BWT encoding using the configured backend.
fn bwt_encode_with_backend(input: &[u8], options: &CompressOptions) -> PzResult<bwt::BwtResult> {
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

    #[cfg(not(feature = "webgpu"))]
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

/// Choose whether to use GPU entropy encoding for a set of streams.
///
/// Returns true when the GPU engine is available and the total stream
/// bytes exceed the GPU_ENTROPY_THRESHOLD.
pub fn should_use_gpu_entropy(streams: &[Vec<u8>], options: &CompressOptions) -> bool {
    #[cfg(feature = "webgpu")]
    {
        if options.backend != Backend::WebGpu {
            return false;
        }
        if options.webgpu_engine.is_none() {
            return false;
        }
        let total: usize = streams.iter().map(|s| s.len()).sum();
        total >= GPU_ENTROPY_THRESHOLD
    }
    #[cfg(not(feature = "webgpu"))]
    {
        let _ = streams;
        let _ = options;
        false
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
