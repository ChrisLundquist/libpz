//! OpenCL GPU backend for libpz.
//!
//! Provides GPU-accelerated implementations of compute-intensive
//! compression stages, primarily LZ77 match finding. The GPU finds
//! all possible matches in parallel; the CPU then selects the optimal
//! chain via dynamic programming (Phase 2).
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐     ┌──────────────────┐     ┌─────────────┐
//! │  Host (CPU)  │────▶│  OpenCL Device   │────▶│  Host (CPU) │
//! │  input data  │     │  parallel match  │     │  dedup +    │
//! │              │     │  finding kernel  │     │  serialize  │
//! └──────────────┘     └──────────────────┘     └─────────────┘
//! ```
//!
//! # Feature Gate
//!
//! This module is only available when compiled with the `opencl` feature:
//! ```bash
//! cargo build --features opencl
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! # #[cfg(feature = "opencl")]
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use pz::opencl::{OpenClEngine, KernelVariant};
//!
//! let engine = OpenClEngine::new()?;
//! println!("Using device: {}", engine.device_name());
//!
//! let input = b"hello world hello world";
//! let matches = engine.find_matches(input, KernelVariant::Batch)?;
//! # Ok(())
//! # }
//! ```

use crate::bwt::BwtResult;
use crate::lz77::Match;
use crate::{PzError, PzResult};

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
use opencl3::event::Event;
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_device_type, cl_uint, CL_BLOCKING};

use std::ptr;

/// Embedded OpenCL kernel source: one work-item per input position,
/// 128KB sliding window.
const LZ77_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77.cl");

/// Embedded OpenCL kernel source: batched variant where each work-item
/// processes STEP_SIZE (32) consecutive positions with a 32KB window.
const LZ77_BATCH_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_batch.cl");

/// Embedded OpenCL kernel source: top-K match finding for optimal parsing.
const LZ77_TOPK_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_topk.cl");

/// Embedded OpenCL kernel source: hash-table-based LZ77 match finding.
/// Two-pass: BuildHashTable scatters positions, FindMatches searches buckets.
const LZ77_HASH_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_hash.cl");

/// Embedded OpenCL kernel source: GPU rank assignment for BWT prefix-doubling.
const BWT_RANK_KERNEL_SOURCE: &str = include_str!("../../kernels/bwt_rank.cl");

/// Embedded OpenCL kernel source: radix sort for BWT prefix-doubling.
const BWT_RADIX_KERNEL_SOURCE: &str = include_str!("../../kernels/bwt_radix.cl");

/// Embedded OpenCL kernel source: GPU Huffman encoding.
/// Two-pass: ComputeBitLengths + WriteCodes, plus a ByteHistogram helper.
const HUFFMAN_ENCODE_KERNEL_SOURCE: &str = include_str!("../../kernels/huffman_encode.cl");

/// Step size used by the batch kernel (must match STEP_SIZE in lz77_batch.cl).
const BATCH_STEP_SIZE: usize = 32;

/// Number of candidates per position in the top-K kernel (must match K in lz77_topk.cl).
const TOPK_K: usize = 4;

/// Minimum input size below which GPU overhead exceeds benefit.
/// For small inputs, the CPU reference implementation is faster.
pub const MIN_GPU_INPUT_SIZE: usize = 64 * 1024; // 64KB

/// Minimum BWT input size for GPU acceleration.
/// Below this, GPU memory transfer overhead exceeds the sort speedup.
pub const MIN_GPU_BWT_SIZE: usize = 32 * 1024; // 32KB

/// GPU match struct matching the OpenCL kernel's `lz77_match_t` layout.
/// Must match the struct definition in the .cl files exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct GpuMatch {
    offset: cl_uint,
    length: cl_uint,
    next: u8,
    _pad: [u8; 3],
}

/// GPU candidate struct matching the OpenCL kernel's `lz77_candidate_t` layout.
/// Must match the struct definition in lz77_topk.cl exactly.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct GpuCandidate {
    offset: u16,
    length: u16,
}

/// Hash table bucket capacity (must match BUCKET_CAP in lz77_hash.cl).
const HASH_BUCKET_CAP: usize = 64;

/// Hash table size (must match HASH_SIZE in lz77_hash.cl).
const HASH_TABLE_SIZE: usize = 1 << 15; // 32768

/// Which LZ77 kernel variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    /// One work-item per position, 128KB window. Better compression
    /// quality but higher GPU memory and compute cost.
    PerPosition,
    /// Batched: each work-item handles STEP_SIZE positions, 32KB window.
    /// Lower overhead, better for large inputs.
    Batch,
    /// Hash-table-based: two-pass approach matching the CPU hash-chain
    /// strategy. Pass 1 builds a hash table, Pass 2 searches buckets.
    /// Much faster than brute-force for large inputs.
    HashTable,
}

/// Information about a discovered OpenCL device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Human-readable device name (e.g. "NVIDIA GeForce RTX 3080").
    pub name: String,
    /// Device vendor string.
    pub vendor: String,
    /// Whether this is a GPU device (vs CPU or accelerator).
    pub is_gpu: bool,
    /// Maximum work-group size supported by the device.
    pub max_work_group_size: usize,
    /// Global memory size in bytes.
    pub global_mem_size: u64,
}

/// Probe all available OpenCL devices without creating an engine.
///
/// Returns an empty vec if no OpenCL runtime is installed or no
/// devices are found (never errors).
pub fn probe_devices() -> Vec<DeviceInfo> {
    let device_ids = match get_all_devices(CL_DEVICE_TYPE_ALL) {
        Ok(ids) => ids,
        Err(_) => return Vec::new(),
    };

    device_ids
        .into_iter()
        .map(|id| {
            let dev = Device::new(id);
            let name = dev.name().unwrap_or_default();
            let vendor = dev.vendor().unwrap_or_default();
            let dev_type: cl_device_type = dev.dev_type().unwrap_or(0);
            let is_gpu = (dev_type & CL_DEVICE_TYPE_GPU) != 0;
            let max_wg = dev.max_work_group_size().unwrap_or(1);
            let global_mem = dev.global_mem_size().unwrap_or(0);
            DeviceInfo {
                name: name.trim().to_string(),
                vendor: vendor.trim().to_string(),
                is_gpu,
                max_work_group_size: max_wg,
                global_mem_size: global_mem,
            }
        })
        .collect()
}

/// Return the number of available OpenCL devices.
///
/// This is a lightweight probe that doesn't create contexts or compile
/// kernels. Returns 0 if OpenCL is not available.
pub fn device_count() -> usize {
    get_all_devices(CL_DEVICE_TYPE_ALL)
        .map(|ids| ids.len())
        .unwrap_or(0)
}

/// OpenCL compute engine.
///
/// Manages the device, context, command queue, and compiled kernels.
/// Create one engine at library init time and reuse it across calls.
///
/// Note: `Debug` is implemented manually because the OpenCL handle
/// types from `opencl3` don't implement `Debug`.
pub struct OpenClEngine {
    _device: Device,
    context: Context,
    queue: CommandQueue,
    /// Compiled per-position LZ77 kernel.
    kernel_per_pos: Kernel,
    /// Compiled batched LZ77 kernel.
    kernel_batch: Kernel,
    /// Compiled top-K LZ77 match finding kernel.
    kernel_topk: Kernel,
    /// Compiled hash-table build kernel.
    kernel_hash_build: Kernel,
    /// Compiled hash-table find-matches kernel.
    kernel_hash_find: Kernel,
    /// Compiled rank comparison kernel (BWT prefix-doubling phase 1).
    kernel_rank_compare: Kernel,
    /// Compiled per-workgroup prefix sum kernel (BWT phase 2a).
    kernel_prefix_sum_local: Kernel,
    /// Compiled prefix sum propagation kernel (BWT phase 2b).
    kernel_prefix_sum_propagate: Kernel,
    /// Compiled rank scatter kernel (BWT phase 3).
    kernel_rank_scatter: Kernel,
    /// Compiled radix sort key computation kernel.
    kernel_radix_compute_keys: Kernel,
    /// Compiled radix sort histogram kernel.
    kernel_radix_histogram: Kernel,
    /// Compiled radix sort scatter kernel.
    kernel_radix_scatter: Kernel,
    /// Compiled inclusive-to-exclusive prefix sum conversion kernel.
    kernel_inclusive_to_exclusive: Kernel,
    /// Workgroup size for prefix sum kernels (power of 2, capped at 256).
    scan_workgroup_size: usize,
    /// Compiled Huffman bit-length computation kernel.
    kernel_huffman_bit_lengths: Kernel,
    /// Compiled Huffman codeword writing kernel.
    kernel_huffman_write_codes: Kernel,
    /// Compiled byte histogram kernel.
    kernel_byte_histogram: Kernel,
    /// Compiled block-level prefix sum kernel (Blelloch scan).
    kernel_prefix_sum_block: Kernel,
    /// Compiled prefix sum apply-offsets kernel.
    kernel_prefix_sum_apply: Kernel,
    /// Device name for diagnostics.
    device_name: String,
    /// Maximum work-group size.
    max_work_group_size: usize,
    /// Whether the selected device is a CPU (vs GPU/accelerator).
    is_cpu: bool,
    /// Whether profiling is enabled (CL_QUEUE_PROFILING_ENABLE).
    profiling: bool,
}

// SAFETY: OpenCL 1.2+ guarantees thread safety for context, command queue, kernel,
// and memory objects. The raw pointers in opencl3 types are opaque handles to the
// OpenCL runtime, which serializes access internally.
unsafe impl Send for OpenClEngine {}
unsafe impl Sync for OpenClEngine {}

impl std::fmt::Debug for OpenClEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenClEngine")
            .field("device_name", &self.device_name)
            .field("max_work_group_size", &self.max_work_group_size)
            .finish_non_exhaustive()
    }
}

impl OpenClEngine {
    /// Create a new engine, selecting the best available GPU device.
    ///
    /// Prefers GPU devices over CPU/accelerator. Falls back to the first
    /// available device if no GPU is found. Returns an error if no OpenCL
    /// devices are available or kernel compilation fails.
    pub fn new() -> PzResult<Self> {
        Self::create(true, false)
    }

    /// Create a new engine with explicit GPU preference.
    ///
    /// If `prefer_gpu` is true, selects the first GPU device (falling back
    /// to any device). If false, selects the first available device regardless
    /// of type.
    pub fn with_device_preference(prefer_gpu: bool) -> PzResult<Self> {
        Self::create(prefer_gpu, false)
    }

    /// Create a new engine with profiling enabled.
    ///
    /// When profiling is on, `CL_QUEUE_PROFILING_ENABLE` is set on the
    /// command queue and major kernel dispatches / buffer transfers print
    /// timing via `eprintln!`.
    pub fn with_profiling(profiling: bool) -> PzResult<Self> {
        Self::create(true, profiling)
    }

    /// Internal constructor shared by all public constructors.
    fn create(prefer_gpu: bool, profiling: bool) -> PzResult<Self> {
        // Discover devices
        let all_ids = get_all_devices(CL_DEVICE_TYPE_ALL).map_err(|_| PzError::Unsupported)?;

        if all_ids.is_empty() {
            return Err(PzError::Unsupported);
        }

        // Select device: prefer discrete GPU over integrated.
        // Among GPUs, pick the one with the most global memory (a reliable
        // heuristic for discrete vs integrated — discrete GPUs have dedicated
        // VRAM while integrated GPUs share system RAM and report less).
        // Sort candidates best-first but try each in order, since on macOS
        // some GPUs (e.g. AMD via deprecated OpenCL) may fail context creation.
        let gpu_ids = if prefer_gpu {
            let mut ids = get_all_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
            // Sort by global memory descending (discrete > integrated)
            ids.sort_by(|a, b| {
                let mem_a = Device::new(*a).global_mem_size().unwrap_or(0);
                let mem_b = Device::new(*b).global_mem_size().unwrap_or(0);
                mem_b.cmp(&mem_a)
            });
            ids
        } else {
            Vec::new()
        };

        // Candidate list: sorted GPUs first, then all devices as fallback
        let candidates: Vec<_> = gpu_ids
            .iter()
            .copied()
            .chain(all_ids.iter().copied())
            .collect();

        // Use the OpenCL 1.2 API (create_default) instead of the 2.0
        // create_default_with_properties, because macOS only supports OpenCL 1.2.
        let queue_props = if profiling {
            CL_QUEUE_PROFILING_ENABLE
        } else {
            0
        };

        // Try each candidate device until one successfully creates a context,
        // command queue, and compiles a test kernel. On macOS, some GPUs
        // (e.g. AMD discrete) pass context/queue creation but fail kernel
        // compilation because Apple only ships an Intel OpenCL driver.
        let mut device = None;
        let mut context = None;
        let mut queue = None;
        for &id in &candidates {
            let dev = Device::new(id);
            let Ok(ctx) = Context::from_device(&dev) else {
                continue;
            };
            #[allow(deprecated)]
            let Ok(q) = CommandQueue::create_default(&ctx, queue_props) else {
                continue;
            };
            // Smoke-test: compile a real kernel to catch drivers that accept
            // context/queue creation but fail on actual kernel code (e.g. AMD
            // on macOS where Apple only ships an Intel OpenCL driver).
            if Program::create_and_build_from_source(&ctx, LZ77_KERNEL_SOURCE, "-Werror").is_err() {
                continue;
            }
            device = Some(dev);
            context = Some(ctx);
            queue = Some(q);
            break;
        }

        let device = device.ok_or(PzError::Unsupported)?;
        let context = context.ok_or(PzError::Unsupported)?;
        let queue = queue.ok_or(PzError::Unsupported)?;

        let device_name = device.name().unwrap_or_default().trim().to_string();
        let max_work_group_size = device.max_work_group_size().unwrap_or(1);
        let dev_type: cl_device_type = device.dev_type().unwrap_or(0);
        let is_cpu = (dev_type & CL_DEVICE_TYPE_GPU) == 0;

        // Compile both kernel variants
        let program_per_pos =
            Program::create_and_build_from_source(&context, LZ77_KERNEL_SOURCE, "-Werror")
                .map_err(|_| PzError::Unsupported)?;

        let program_batch =
            Program::create_and_build_from_source(&context, LZ77_BATCH_KERNEL_SOURCE, "-Werror")
                .map_err(|_| PzError::Unsupported)?;

        let kernel_per_pos =
            Kernel::create(&program_per_pos, "Encode").map_err(|_| PzError::Unsupported)?;

        let kernel_batch =
            Kernel::create(&program_batch, "Encode").map_err(|_| PzError::Unsupported)?;

        // Compile top-K LZ77 kernel for optimal parsing
        let program_topk =
            Program::create_and_build_from_source(&context, LZ77_TOPK_KERNEL_SOURCE, "-Werror")
                .map_err(|_| PzError::Unsupported)?;

        let kernel_topk =
            Kernel::create(&program_topk, "EncodeTopK").map_err(|_| PzError::Unsupported)?;

        // Compile hash-table-based LZ77 kernel (two entry points)
        let program_hash =
            Program::create_and_build_from_source(&context, LZ77_HASH_KERNEL_SOURCE, "-Werror")
                .map_err(|_| PzError::Unsupported)?;

        let kernel_hash_build =
            Kernel::create(&program_hash, "BuildHashTable").map_err(|_| PzError::Unsupported)?;

        let kernel_hash_find =
            Kernel::create(&program_hash, "FindMatches").map_err(|_| PzError::Unsupported)?;

        // Compile BWT rank assignment kernels with workgroup size define.
        // Cap at 256 and round down to the nearest power of 2 for portability.
        let capped = max_work_group_size.clamp(1, 256);
        let scan_workgroup_size = 1 << (usize::BITS - 1 - capped.leading_zeros());
        let rank_flags = format!("-Werror -DWORKGROUP_SIZE={scan_workgroup_size}");
        let program_bwt_rank =
            Program::create_and_build_from_source(&context, BWT_RANK_KERNEL_SOURCE, &rank_flags)
                .map_err(|_| PzError::Unsupported)?;

        let kernel_rank_compare =
            Kernel::create(&program_bwt_rank, "rank_compare").map_err(|_| PzError::Unsupported)?;
        let kernel_prefix_sum_local = Kernel::create(&program_bwt_rank, "prefix_sum_local")
            .map_err(|_| PzError::Unsupported)?;
        let kernel_prefix_sum_propagate = Kernel::create(&program_bwt_rank, "prefix_sum_propagate")
            .map_err(|_| PzError::Unsupported)?;
        let kernel_rank_scatter =
            Kernel::create(&program_bwt_rank, "rank_scatter").map_err(|_| PzError::Unsupported)?;

        // Compile radix sort kernels (same workgroup size define)
        let program_bwt_radix =
            Program::create_and_build_from_source(&context, BWT_RADIX_KERNEL_SOURCE, &rank_flags)
                .map_err(|_| PzError::Unsupported)?;

        let kernel_radix_compute_keys = Kernel::create(&program_bwt_radix, "radix_compute_keys")
            .map_err(|_| PzError::Unsupported)?;
        let kernel_radix_histogram = Kernel::create(&program_bwt_radix, "radix_histogram")
            .map_err(|_| PzError::Unsupported)?;
        let kernel_radix_scatter = Kernel::create(&program_bwt_radix, "radix_scatter")
            .map_err(|_| PzError::Unsupported)?;
        let kernel_inclusive_to_exclusive =
            Kernel::create(&program_bwt_radix, "inclusive_to_exclusive")
                .map_err(|_| PzError::Unsupported)?;

        // Compile Huffman encoding kernels
        let program_huffman = Program::create_and_build_from_source(
            &context,
            HUFFMAN_ENCODE_KERNEL_SOURCE,
            "-Werror",
        )
        .map_err(|_| PzError::Unsupported)?;

        let kernel_huffman_bit_lengths = Kernel::create(&program_huffman, "ComputeBitLengths")
            .map_err(|_| PzError::Unsupported)?;

        let kernel_huffman_write_codes =
            Kernel::create(&program_huffman, "WriteCodes").map_err(|_| PzError::Unsupported)?;

        let kernel_byte_histogram =
            Kernel::create(&program_huffman, "ByteHistogram").map_err(|_| PzError::Unsupported)?;

        let kernel_prefix_sum_block =
            Kernel::create(&program_huffman, "PrefixSumBlock").map_err(|_| PzError::Unsupported)?;

        let kernel_prefix_sum_apply =
            Kernel::create(&program_huffman, "PrefixSumApply").map_err(|_| PzError::Unsupported)?;

        Ok(OpenClEngine {
            _device: device,
            context,
            queue,
            kernel_per_pos,
            kernel_batch,
            kernel_topk,
            kernel_hash_build,
            kernel_hash_find,
            kernel_rank_compare,
            kernel_prefix_sum_local,
            kernel_prefix_sum_propagate,
            kernel_rank_scatter,
            kernel_radix_compute_keys,
            kernel_radix_histogram,
            kernel_radix_scatter,
            kernel_inclusive_to_exclusive,
            scan_workgroup_size,
            kernel_huffman_bit_lengths,
            kernel_huffman_write_codes,
            kernel_byte_histogram,
            kernel_prefix_sum_block,
            kernel_prefix_sum_apply,
            device_name,
            max_work_group_size,
            is_cpu,
            profiling,
        })
    }

    /// Return the name of the selected compute device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Return the maximum work-group size for the device.
    pub fn max_work_group_size(&self) -> usize {
        self.max_work_group_size
    }

    /// Check if the selected device is a CPU (not a GPU or accelerator).
    ///
    /// Useful for skipping GPU-optimized algorithms (like BWT bitonic sort)
    /// that perform poorly on CPU OpenCL devices.
    pub fn is_cpu_device(&self) -> bool {
        self.is_cpu
    }

    /// Whether profiling is enabled on this engine.
    pub fn profiling(&self) -> bool {
        self.profiling
    }

    /// Extract elapsed time in milliseconds from a completed OpenCL event.
    ///
    /// Requires the command queue to have been created with
    /// `CL_QUEUE_PROFILING_ENABLE`. Returns `None` if profiling is
    /// disabled or the event doesn't have timing data.
    pub fn event_elapsed_ms(event: &Event) -> Option<f64> {
        let start = event.profiling_command_start().ok()?;
        let end = event.profiling_command_end().ok()?;
        Some((end - start) as f64 / 1_000_000.0)
    }

    /// Log timing for a completed event when profiling is enabled.
    fn profile_event(&self, label: &str, event: &Event) {
        if self.profiling {
            if let Some(ms) = Self::event_elapsed_ms(event) {
                eprintln!("[pz-gpu] {label}: {ms:.3} ms");
            }
        }
    }
}

/// Deduplicate raw GPU match output into a non-overlapping match sequence.
///
/// The GPU kernel produces a match for every input position, but many of
/// these overlap (e.g., position i has a match of length 7, position i+1
/// has length 6, etc.). This function walks the match array, advancing
/// by `match.length + 1` each step to select a non-overlapping covering.
///
/// This is equivalent to the `DedupeMatches()` function from the C engine.
fn dedupe_gpu_matches(gpu_matches: &[GpuMatch], input: &[u8]) -> Vec<Match> {
    if gpu_matches.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let len = gpu_matches.len();
    let mut index = 0;

    while index < len {
        let gm = &gpu_matches[index];

        // Ensure the match length doesn't exceed remaining input
        let remaining = len - index;
        let mut match_length = gm.length as usize;
        if match_length >= remaining {
            match_length = if remaining > 0 { remaining - 1 } else { 0 };
        }

        let next = if index + match_length < input.len() {
            input[index + match_length]
        } else {
            gm.next
        };

        result.push(Match {
            offset: gm.offset as u16,
            length: match_length as u16,
            next,
        });

        index += match_length + 1;
    }

    result
}

mod bwt;
mod huffman;
mod lz77;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
