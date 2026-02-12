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
const LZ77_KERNEL_SOURCE: &str = include_str!("../kernels/lz77.cl");

/// Embedded OpenCL kernel source: batched variant where each work-item
/// processes STEP_SIZE (32) consecutive positions with a 32KB window.
const LZ77_BATCH_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_batch.cl");

/// Embedded OpenCL kernel source: top-K match finding for optimal parsing.
const LZ77_TOPK_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_topk.cl");

/// Embedded OpenCL kernel source: hash-table-based LZ77 match finding.
/// Two-pass: BuildHashTable scatters positions, FindMatches searches buckets.
const LZ77_HASH_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_hash.cl");

/// Embedded OpenCL kernel source: GPU rank assignment for BWT prefix-doubling.
const BWT_RANK_KERNEL_SOURCE: &str = include_str!("../kernels/bwt_rank.cl");

/// Embedded OpenCL kernel source: radix sort for BWT prefix-doubling.
const BWT_RADIX_KERNEL_SOURCE: &str = include_str!("../kernels/bwt_radix.cl");

/// Embedded OpenCL kernel source: GPU Huffman encoding.
/// Two-pass: ComputeBitLengths + WriteCodes, plus a ByteHistogram helper.
const HUFFMAN_ENCODE_KERNEL_SOURCE: &str = include_str!("../kernels/huffman_encode.cl");

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

        // Select device: prefer GPU if requested
        let selected_id = if prefer_gpu {
            let gpu_ids = get_all_devices(CL_DEVICE_TYPE_GPU).unwrap_or_default();
            if gpu_ids.is_empty() {
                all_ids[0]
            } else {
                gpu_ids[0]
            }
        } else {
            all_ids[0]
        };

        let device = Device::new(selected_id);
        let device_name = device.name().unwrap_or_default().trim().to_string();
        let max_work_group_size = device.max_work_group_size().unwrap_or(1);
        let dev_type: cl_device_type = device.dev_type().unwrap_or(0);
        let is_cpu = (dev_type & CL_DEVICE_TYPE_GPU) == 0;

        // Create context and command queue
        let context = Context::from_device(&device).map_err(|_| PzError::Unsupported)?;

        // Use the OpenCL 1.2 API (create_default) instead of the 2.0
        // create_default_with_properties, because macOS only supports OpenCL 1.2.
        let queue_props = if profiling {
            CL_QUEUE_PROFILING_ENABLE
        } else {
            0
        };
        #[allow(deprecated)]
        let queue = CommandQueue::create_default(&context, queue_props)
            .map_err(|_| PzError::Unsupported)?;

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

    /// Find LZ77 matches for the entire input using the GPU.
    ///
    /// Returns a vector of deduplicated `Match` structs compatible with
    /// the CPU LZ77 format, ready for serialization or further processing
    /// (e.g., optimal-parse DP in Phase 2).
    ///
    /// # Arguments
    ///
    /// * `input` - The data to compress. Must be non-empty.
    /// * `variant` - Which kernel to use (per-position or batched).
    pub fn find_matches(&self, input: &[u8], variant: KernelVariant) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Allocate device buffers
        let input_len = input.len();

        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let output_buf = unsafe {
            Buffer::<GpuMatch>::create(&self.context, CL_MEM_WRITE_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        // Write input to device
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_event.wait().map_err(|_| PzError::InvalidInput)?;
        self.profile_event("find_matches: upload input", &write_event);

        // Execute kernel
        match variant {
            KernelVariant::PerPosition => {
                self.run_per_position_kernel(&input_buf, &output_buf, input_len)?;
            }
            KernelVariant::Batch => {
                self.run_batch_kernel(&input_buf, &output_buf, input_len)?;
            }
            KernelVariant::HashTable => {
                self.run_hash_kernel(&input_buf, &output_buf, input_len)?;
            }
        }

        // Read back results
        let mut gpu_matches = vec![GpuMatch::default(); input_len];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut gpu_matches, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        read_event.wait().map_err(|_| PzError::InvalidInput)?;
        self.profile_event("find_matches: download matches", &read_event);

        // Deduplicate and convert to the Rust Match type
        let matches = dedupe_gpu_matches(&gpu_matches, input);
        Ok(matches)
    }

    /// Execute the per-position kernel (one work-item per byte).
    fn run_per_position_kernel(
        &self,
        input_buf: &Buffer<u8>,
        output_buf: &Buffer<GpuMatch>,
        input_len: usize,
    ) -> PzResult<()> {
        let count = input_len as cl_uint;
        let global_size = input_len;

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_per_pos)
                .set_arg(input_buf)
                .set_arg(output_buf)
                .set_arg(&count)
                .set_global_work_size(global_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };

        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 per-position kernel", &kernel_event);
        Ok(())
    }

    /// Execute the batch kernel (each work-item handles STEP_SIZE positions).
    fn run_batch_kernel(
        &self,
        input_buf: &Buffer<u8>,
        output_buf: &Buffer<GpuMatch>,
        input_len: usize,
    ) -> PzResult<()> {
        let count = input_len as cl_uint;
        // Round up to cover all positions
        let num_work_items = input_len.div_ceil(BATCH_STEP_SIZE);

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_batch)
                .set_arg(input_buf)
                .set_arg(output_buf)
                .set_arg(&count)
                .set_global_work_size(num_work_items)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };

        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 batch kernel", &kernel_event);
        Ok(())
    }

    /// Execute the two-pass hash-table kernel.
    ///
    /// Pass 1: BuildHashTable — each work-item hashes its 3-byte prefix
    /// and atomically appends its position to a bucket.
    /// Pass 2: FindMatches — each work-item searches its hash bucket
    /// for the best match (bounded by MAX_CHAIN).
    fn run_hash_kernel(
        &self,
        input_buf: &Buffer<u8>,
        output_buf: &Buffer<GpuMatch>,
        input_len: usize,
    ) -> PzResult<()> {
        let count = input_len as cl_uint;
        let table_entries = HASH_TABLE_SIZE * HASH_BUCKET_CAP;

        // Allocate hash table buffers on the device
        let mut hash_counts_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                HASH_TABLE_SIZE,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        let hash_table_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                table_entries,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Zero the hash counts
        let zeros = vec![0u32; HASH_TABLE_SIZE];
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut hash_counts_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_event.wait().map_err(|_| PzError::InvalidInput)?;

        // Pass 1: Build hash table
        let build_event = unsafe {
            ExecuteKernel::new(&self.kernel_hash_build)
                .set_arg(input_buf)
                .set_arg(&count)
                .set_arg(&hash_counts_buf)
                .set_arg(&hash_table_buf)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        build_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 hash: build table", &build_event);

        // Pass 2: Find matches
        let find_event = unsafe {
            ExecuteKernel::new(&self.kernel_hash_find)
                .set_arg(input_buf)
                .set_arg(&count)
                .set_arg(&hash_counts_buf)
                .set_arg(&hash_table_buf)
                .set_arg(output_buf)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        find_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 hash: find matches", &find_event);

        Ok(())
    }

    /// GPU-accelerated LZ77 compression.
    ///
    /// Uses the GPU to find matches, deduplicates them, and serializes
    /// to the same byte format as the CPU `lz77::compress_lazy()`.
    /// The output is decompressible by `lz77::decompress()`.
    pub fn lz77_compress(&self, input: &[u8], variant: KernelVariant) -> PzResult<Vec<u8>> {
        let matches = self.find_matches(input, variant)?;

        let mut output = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
        for m in &matches {
            output.extend_from_slice(&m.to_bytes());
        }
        Ok(output)
    }

    /// GPU-accelerated top-K match finding for optimal parsing.
    ///
    /// For each input position, finds the K best match candidates using
    /// the GPU. Returns a `MatchTable` ready for `optimal_parse()`.
    pub fn find_topk_matches(&self, input: &[u8]) -> PzResult<crate::optimal::MatchTable> {
        use crate::optimal::{MatchCandidate, MatchTable};

        if input.is_empty() {
            return Ok(MatchTable::new(0, TOPK_K));
        }

        let input_len = input.len();
        let output_len = input_len * TOPK_K;

        // Allocate device buffers
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let output_buf = unsafe {
            Buffer::<GpuCandidate>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                output_len,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Write input to device
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_event.wait().map_err(|_| PzError::InvalidInput)?;

        // Execute top-K kernel
        let count = input_len as cl_uint;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_topk)
                .set_arg(&input_buf)
                .set_arg(&output_buf)
                .set_arg(&count)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;

        // Read back results
        let mut gpu_candidates = vec![GpuCandidate::default(); output_len];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut gpu_candidates, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        read_event.wait().map_err(|_| PzError::InvalidInput)?;

        // Convert to MatchTable
        let mut table = MatchTable::new(input_len, TOPK_K);
        for pos in 0..input_len {
            let slot = table.at_mut(pos);
            for k in 0..TOPK_K {
                let gc = &gpu_candidates[pos * TOPK_K + k];
                slot[k] = MatchCandidate {
                    offset: gc.offset,
                    length: gc.length,
                };
            }
        }

        Ok(table)
    }

    /// GPU-accelerated BWT forward transform.
    ///
    /// Uses the GPU for the expensive suffix array sort steps (radix sort)
    /// and rank assignment (parallel prefix sum). Produces a valid BWT
    /// that round-trips correctly through `bwt::decode()`.
    pub fn bwt_encode(&self, input: &[u8]) -> PzResult<BwtResult> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        let sa = self.bwt_build_suffix_array(input)?;

        // Extract BWT from suffix array (same logic as CPU bwt::encode)
        let n = input.len();
        let mut bwt = Vec::with_capacity(n);
        let mut primary_index = 0u32;

        for (i, &sa_val) in sa.iter().enumerate() {
            if sa_val == 0 {
                primary_index = i as u32;
                bwt.push(input[n - 1]);
            } else {
                bwt.push(input[sa_val - 1]);
            }
        }

        Ok(BwtResult {
            data: bwt,
            primary_index,
        })
    }

    /// Build suffix array on the GPU using prefix-doubling with radix sort.
    ///
    /// Each doubling step sorts sa[] by (rank[sa[i]], rank[(sa[i]+k) % n]),
    /// then assigns new ranks via parallel prefix sum — all on the GPU.
    /// Only a single scalar (max_rank) is read back per step for convergence.
    fn bwt_build_suffix_array(&self, input: &[u8]) -> PzResult<Vec<usize>> {
        let n = input.len();
        if n <= 1 {
            return Ok(if n == 0 { Vec::new() } else { vec![0] });
        }

        // Pad to power-of-2 size. Sentinel entries have rank = UINT_MAX.
        let padded_n = n.next_power_of_two();

        // Initialize sa in descending order so that LSB-first stable radix sort
        // breaks ties by suffix index descending (matching CPU SA-IS behavior).
        let sa_host: Vec<cl_uint> = (0..padded_n as cl_uint).rev().collect();
        let mut rank_host: Vec<cl_uint> = vec![cl_uint::MAX; padded_n];
        for i in 0..n {
            rank_host[i] = input[i] as cl_uint;
        }

        // Allocate GPU buffers
        let mut sa_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut sa_buf_alt = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut rank_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        // Double-buffer for rank output (swap each iteration)
        let mut rank_buf_alt = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        // diff/prefix buffer for rank assignment
        let mut diff_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut prefix_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        // Keys buffer for radix sort (one u32 per element, holds 8-bit digit)
        let mut keys_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        // Radix sort histogram buffers
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let histogram_len = 256 * num_groups;
        let mut histogram_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                histogram_len.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut histogram_buf_scan = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                histogram_len.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Block sums buffers for multi-level prefix sum (rank assignment)
        let block_elems = wg * 2;
        let num_blocks_l0 = padded_n.div_ceil(block_elems);
        let mut block_sums_l0 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l0.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let num_blocks_l1 = num_blocks_l0.div_ceil(block_elems);
        let mut block_sums_l1 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l1.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let num_blocks_l2 = num_blocks_l1.div_ceil(block_elems);
        let mut block_sums_l2 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l2.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Block sums buffers for histogram prefix sum (radix sort)
        let hist_num_blocks_l0 = histogram_len.div_ceil(block_elems);
        let mut hist_block_sums_l0 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hist_num_blocks_l0.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let hist_num_blocks_l1 = hist_num_blocks_l0.div_ceil(block_elems);
        let mut hist_block_sums_l1 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hist_num_blocks_l1.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let hist_num_blocks_l2 = hist_num_blocks_l1.div_ceil(block_elems);
        let mut hist_block_sums_l2 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hist_num_blocks_l2.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Upload initial data
        let write_sa = unsafe {
            self.queue
                .enqueue_write_buffer(&mut sa_buf, CL_BLOCKING, 0, &sa_host, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_sa.wait().map_err(|_| PzError::InvalidInput)?;

        let write_rank = unsafe {
            self.queue
                .enqueue_write_buffer(&mut rank_buf, CL_BLOCKING, 0, &rank_host, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_rank.wait().map_err(|_| PzError::InvalidInput)?;

        let n_arg = n as cl_uint;
        let padded_n_arg = padded_n as cl_uint;

        // Prefix-doubling loop — all GPU work is event-chained, only
        // the convergence read blocks the host (once per doubling step).
        // Adaptive key width: initial ranks are bytes (0-255), so first
        // radix sort only needs 2 passes instead of 8.
        let mut max_rank: cl_uint = 255;
        let mut k_step: usize = 1;
        while k_step < n {
            let k_arg = k_step as cl_uint;

            // Phase 0: Radix sort sa[] by composite key (adaptive pass count)
            let sort_event = self.run_radix_sort(
                &mut sa_buf,
                &mut sa_buf_alt,
                &rank_buf,
                &mut keys_buf,
                &mut histogram_buf,
                &mut histogram_buf_scan,
                padded_n,
                n_arg,
                k_arg,
                &mut hist_block_sums_l0,
                &mut hist_block_sums_l1,
                &mut hist_block_sums_l2,
                hist_num_blocks_l0,
                hist_num_blocks_l1,
                max_rank,
                None, // initial buffer writes were CL_BLOCKING
            )?;

            // Phase 1: Compare consecutive composite keys → diff[]
            let compare_event = self.run_rank_compare(
                &sa_buf,
                &rank_buf,
                &mut diff_buf,
                n_arg,
                padded_n_arg,
                k_arg,
                Some(&sort_event),
            )?;

            // Phase 2: Inclusive prefix sum of diff[] → prefix[]
            let prefix_event = self.run_prefix_sum(
                &diff_buf,
                &mut prefix_buf,
                padded_n,
                &mut block_sums_l0,
                &mut block_sums_l1,
                &mut block_sums_l2,
                num_blocks_l0,
                num_blocks_l1,
                Some(&compare_event),
            )?;

            // Phase 3: Scatter ranks to position-indexed buffer
            let scatter_event = self.run_rank_scatter(
                &sa_buf,
                &prefix_buf,
                &mut rank_buf_alt,
                n_arg,
                padded_n_arg,
                Some(&prefix_event),
            )?;

            // Read convergence scalar: prefix[n-1] = max_rank among real entries.
            // This is the only host sync point per doubling step — the read waits
            // for the scatter event, then blocks until the data is available.
            let mut max_rank_host: [cl_uint; 1] = [0];
            let read_event = unsafe {
                self.queue
                    .enqueue_read_buffer(
                        &prefix_buf,
                        CL_BLOCKING,
                        (n - 1) * std::mem::size_of::<cl_uint>(),
                        &mut max_rank_host,
                        &[scatter_event.get()],
                    )
                    .map_err(|_| PzError::InvalidInput)?
            };
            read_event.wait().map_err(|_| PzError::InvalidInput)?;

            // Update max_rank for next iteration's adaptive pass selection
            max_rank = max_rank_host[0];

            // Swap: rank_buf_alt (new ranks) becomes rank_buf for next iteration
            std::mem::swap(&mut rank_buf, &mut rank_buf_alt);

            if max_rank as usize == n - 1 {
                break;
            }

            k_step *= 2;
        }

        // Read final sorted sa back to host
        let mut sa_host_final: Vec<cl_uint> = vec![0; padded_n];
        let read_sa = unsafe {
            self.queue
                .enqueue_read_buffer(&sa_buf, CL_BLOCKING, 0, &mut sa_host_final, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        read_sa.wait().map_err(|_| PzError::InvalidInput)?;

        // Extract the real suffix array (filter out sentinel entries)
        let sa: Vec<usize> = sa_host_final
            .iter()
            .filter(|&&v| (v as usize) < n)
            .map(|&v| v as usize)
            .collect();

        if sa.len() != n {
            return Err(PzError::InvalidInput);
        }

        Ok(sa)
    }

    /// Run an adaptive radix sort on the sa buffer by composite key.
    ///
    /// Performs LSB-first 8-bit radix sort, sorting sa[] by the 64-bit
    /// composite key (rank[sa[i]] << 32 | rank[(sa[i]+k) % n]).
    ///
    /// Adaptive: only sorts the bytes that contain nonzero data based on
    /// `max_rank`. For max_rank < 256, only 2 passes instead of 8.
    ///
    /// Stable sort: elements with equal keys preserve their input order.
    /// Combined with descending initial sa[], this matches CPU SA-IS tiebreaking.
    ///
    /// Returns the final event for chaining. All internal kernel dispatches
    /// are chained via events with no host waits.
    #[allow(clippy::too_many_arguments)]
    fn run_radix_sort(
        &self,
        sa_buf: &mut Buffer<cl_uint>,
        sa_buf_alt: &mut Buffer<cl_uint>,
        rank_buf: &Buffer<cl_uint>,
        keys_buf: &mut Buffer<cl_uint>,
        histogram_buf: &mut Buffer<cl_uint>,
        histogram_buf_scan: &mut Buffer<cl_uint>,
        padded_n: usize,
        n: cl_uint,
        k_doubling: cl_uint,
        hist_block_sums_l0: &mut Buffer<cl_uint>,
        hist_block_sums_l1: &mut Buffer<cl_uint>,
        hist_block_sums_l2: &mut Buffer<cl_uint>,
        hist_num_blocks_l0: usize,
        hist_num_blocks_l1: usize,
        max_rank: cl_uint,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let padded_n_arg = padded_n as cl_uint;
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let num_groups_arg = num_groups as cl_uint;
        let histogram_len = 256 * num_groups;
        let global_wg = num_groups * wg;

        // Adaptive pass selection: skip zero-byte passes.
        // Composite key = (r1 << 32) | r2, both in [0, max_rank].
        // Passes 0..bytes_needed sort r2 bytes, passes 4..4+bytes_needed sort r1.
        let bytes_needed: u32 = if max_rank < 256 {
            1
        } else if max_rank < 65536 {
            2
        } else if max_rank < 16_777_216 {
            3
        } else {
            4
        };
        // Build pass list: r2 low bytes then r1 low bytes
        let mut passes: Vec<u32> = (0..bytes_needed).chain(4..4 + bytes_needed).collect();
        // Ensure even count so sa_buf holds result (each pass swaps sa_buf ↔ sa_buf_alt)
        debug_assert!(passes.len().is_multiple_of(2), "pass count must be even");
        if !passes.len().is_multiple_of(2) {
            passes.push(bytes_needed); // no-op pass on zero byte
        }

        let mut prev_event: Option<Event> = None;

        for (i, &pass) in passes.iter().enumerate() {
            let pass_arg = pass as cl_uint;
            let wait_ref: Option<&Event> = if i == 0 {
                wait_event
            } else {
                prev_event.as_ref()
            };

            // Phase 1: Compute 8-bit digit for each element
            let key_event = unsafe {
                let mut exec = ExecuteKernel::new(&self.kernel_radix_compute_keys);
                exec.set_arg(sa_buf as &Buffer<cl_uint>)
                    .set_arg(rank_buf)
                    .set_arg(keys_buf)
                    .set_arg(&n)
                    .set_arg(&padded_n_arg)
                    .set_arg(&k_doubling)
                    .set_arg(&pass_arg)
                    .set_global_work_size(global_wg);
                if let Some(evt) = wait_ref {
                    exec.set_wait_event(evt);
                }
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Phase 2: Per-workgroup histogram
            let hist_event = unsafe {
                let mut exec = ExecuteKernel::new(&self.kernel_radix_histogram);
                exec.set_arg(keys_buf as &Buffer<cl_uint>)
                    .set_arg(histogram_buf)
                    .set_arg(&padded_n_arg)
                    .set_arg(&num_groups_arg)
                    .set_global_work_size(global_wg)
                    .set_local_work_size(wg)
                    .set_wait_event(&key_event);
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Phase 3: Inclusive prefix sum over histogram
            let prefix_event = self.run_prefix_sum(
                histogram_buf,
                histogram_buf_scan,
                histogram_len,
                hist_block_sums_l0,
                hist_block_sums_l1,
                hist_block_sums_l2,
                hist_num_blocks_l0,
                hist_num_blocks_l1,
                Some(&hist_event),
            )?;

            // Phase 3b: Convert inclusive to exclusive prefix sum
            let hist_len_arg = histogram_len as cl_uint;
            let excl_event = unsafe {
                let mut exec = ExecuteKernel::new(&self.kernel_inclusive_to_exclusive);
                exec.set_arg(histogram_buf_scan as &Buffer<cl_uint>)
                    .set_arg(histogram_buf)
                    .set_arg(&hist_len_arg)
                    .set_global_work_size(histogram_len)
                    .set_wait_event(&prefix_event);
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Phase 4: Scatter sa elements to sorted positions
            let scatter_event = unsafe {
                let mut exec = ExecuteKernel::new(&self.kernel_radix_scatter);
                exec.set_arg(sa_buf as &Buffer<cl_uint>)
                    .set_arg(keys_buf as &Buffer<cl_uint>)
                    .set_arg(histogram_buf)
                    .set_arg(sa_buf_alt)
                    .set_arg(&padded_n_arg)
                    .set_arg(&num_groups_arg)
                    .set_global_work_size(global_wg)
                    .set_local_work_size(wg)
                    .set_wait_event(&excl_event);
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Swap: sa_buf_alt (sorted output) becomes sa_buf for next pass
            std::mem::swap(sa_buf, sa_buf_alt);
            prev_event = Some(scatter_event);
        }

        // After an even number of passes, sa_buf holds the final sorted result
        Ok(prev_event.expect("at least 2 radix passes"))
    }

    /// Run the rank comparison kernel (BWT prefix-doubling phase 1).
    ///
    /// For each sorted position i, computes diff[i] = 1 if the composite key
    /// of sa[i] differs from sa[i-1], 0 otherwise.
    /// Returns the kernel completion event for chaining.
    #[allow(clippy::too_many_arguments)]
    fn run_rank_compare(
        &self,
        sa_buf: &Buffer<cl_uint>,
        rank_buf: &Buffer<cl_uint>,
        diff_buf: &mut Buffer<cl_uint>,
        n: cl_uint,
        padded_n: cl_uint,
        k: cl_uint,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(&self.kernel_rank_compare);
            exec.set_arg(sa_buf)
                .set_arg(rank_buf)
                .set_arg(diff_buf)
                .set_arg(&n)
                .set_arg(&padded_n)
                .set_arg(&k)
                .set_global_work_size(padded_n as usize);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }

    /// Run a multi-level inclusive prefix sum on the GPU.
    ///
    /// Reads from `input_buf`, writes inclusive prefix sums to `output_buf`.
    /// Uses a two- or three-level workgroup scan depending on input size.
    /// Returns the final event for chaining.
    ///
    /// Note: temporary buffers (`block_sums_l*_scanned`) are dropped on return,
    /// but the OpenCL runtime retains memory objects for enqueued kernels, so
    /// they remain valid until the returned event (and its dependencies) complete.
    #[allow(clippy::too_many_arguments)]
    fn run_prefix_sum(
        &self,
        input_buf: &Buffer<cl_uint>,
        output_buf: &mut Buffer<cl_uint>,
        count: usize,
        block_sums_l0: &mut Buffer<cl_uint>,
        block_sums_l1: &mut Buffer<cl_uint>,
        block_sums_l2: &mut Buffer<cl_uint>,
        num_blocks_l0: usize,
        num_blocks_l1: usize,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let wg = self.scan_workgroup_size;
        let block_elems = wg * 2;

        // Level 0: per-workgroup scan of input → output, block totals → block_sums_l0
        let evt_l0 =
            self.run_prefix_sum_local(input_buf, output_buf, block_sums_l0, count, wg, wait_event)?;

        if num_blocks_l0 <= 1 {
            // Single workgroup: output is already the final inclusive prefix sum
            return Ok(evt_l0);
        }

        // Level 1: scan block_sums_l0 → block_sums_l0_scanned
        let mut block_sums_l0_scanned = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l0.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let evt_l1 = self.run_prefix_sum_local(
            block_sums_l0,
            &mut block_sums_l0_scanned,
            block_sums_l1,
            num_blocks_l0,
            wg,
            Some(&evt_l0),
        )?;

        let evt_offsets = if num_blocks_l1 > 1 {
            // Level 2: scan block_sums_l1 → block_sums_l1_scanned
            let mut block_sums_l1_scanned = unsafe {
                Buffer::<cl_uint>::create(
                    &self.context,
                    CL_MEM_READ_WRITE,
                    num_blocks_l1.max(1),
                    ptr::null_mut(),
                )
                .map_err(|_| PzError::BufferTooSmall)?
            };
            let evt_l2 = self.run_prefix_sum_local(
                block_sums_l1,
                &mut block_sums_l1_scanned,
                block_sums_l2,
                num_blocks_l1,
                wg,
                Some(&evt_l1),
            )?;

            // Propagate L2 offsets into L1 scanned sums
            self.run_prefix_sum_propagate(
                &mut block_sums_l0_scanned,
                &block_sums_l1_scanned,
                num_blocks_l0,
                block_elems,
                Some(&evt_l2),
            )?
        } else {
            evt_l1
        };

        // Propagate L1 (or L2-fixed L1) offsets into the L0 output
        let evt_final = self.run_prefix_sum_propagate(
            output_buf,
            &block_sums_l0_scanned,
            count,
            block_elems,
            Some(&evt_offsets),
        )?;

        Ok(evt_final)
    }

    /// Dispatch a single-level per-workgroup prefix sum kernel.
    ///
    /// Returns the kernel completion event for chaining.
    fn run_prefix_sum_local(
        &self,
        input_buf: &Buffer<cl_uint>,
        output_buf: &mut Buffer<cl_uint>,
        block_sums_buf: &mut Buffer<cl_uint>,
        count: usize,
        wg_size: usize,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let count_arg = count as cl_uint;
        let global_size = count.div_ceil(wg_size * 2) * wg_size;
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(&self.kernel_prefix_sum_local);
            exec.set_arg(input_buf)
                .set_arg(output_buf)
                .set_arg(block_sums_buf)
                .set_arg(&count_arg)
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }

    /// Dispatch the prefix sum propagation kernel (add block offsets).
    ///
    /// Returns the kernel completion event for chaining.
    fn run_prefix_sum_propagate(
        &self,
        data_buf: &mut Buffer<cl_uint>,
        offsets_buf: &Buffer<cl_uint>,
        count: usize,
        block_elems: usize,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let count_arg = count as cl_uint;
        let _ = block_elems; // BLOCK_ELEMS is a compile-time constant in the kernel
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(&self.kernel_prefix_sum_propagate);
            exec.set_arg(data_buf)
                .set_arg(offsets_buf)
                .set_arg(&count_arg)
                .set_global_work_size(count);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }

    /// Run the rank scatter kernel (BWT prefix-doubling phase 3).
    ///
    /// Writes new_rank[sa[i]] = prefix[i] for real entries, UINT_MAX for sentinels.
    /// Returns the kernel completion event for chaining.
    fn run_rank_scatter(
        &self,
        sa_buf: &Buffer<cl_uint>,
        prefix_buf: &Buffer<cl_uint>,
        new_rank_buf: &mut Buffer<cl_uint>,
        n: cl_uint,
        padded_n: cl_uint,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(&self.kernel_rank_scatter);
            exec.set_arg(sa_buf)
                .set_arg(prefix_buf)
                .set_arg(new_rank_buf)
                .set_arg(&n)
                .set_arg(&padded_n)
                .set_global_work_size(padded_n as usize);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }

    /// Compute a byte histogram of the input data on the GPU.
    ///
    /// Returns a 256-element array of byte frequencies. This can be used
    /// to build a Huffman tree without downloading the full data to the CPU.
    pub fn byte_histogram(&self, input: &[u8]) -> PzResult<[u32; 256]> {
        if input.is_empty() {
            return Ok([0u32; 256]);
        }

        let n = input.len();

        // Create input buffer
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Create histogram buffer (256 uints, zeroed)
        let mut hist_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Upload input
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_event.wait().map_err(|_| PzError::Unsupported)?;

        // Zero the histogram buffer
        let zeros = vec![0u32; 256];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut hist_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        // Run histogram kernel
        let n_arg = n as cl_uint;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_byte_histogram)
                .set_arg(&input_buf)
                .set_arg(&hist_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download histogram
        let mut histogram = vec![0u32; 256];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&hist_buf, CL_BLOCKING, 0, &mut histogram, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_event.wait().map_err(|_| PzError::Unsupported)?;

        let mut result = [0u32; 256];
        result.copy_from_slice(&histogram);
        Ok(result)
    }

    /// Encode data using Huffman coding on the GPU.
    ///
    /// Takes a code lookup table (from a HuffmanTree) and the input symbols.
    /// Returns the encoded bytes and the total number of bits.
    ///
    /// The lookup table format: for each byte value 0-255,
    /// `code_lut[byte] = (bits << 24) | codeword` where codeword is at most 24 bits.
    pub fn huffman_encode(
        &self,
        input: &[u8],
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();

        // Create buffers
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        let mut lut_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        let bit_lengths_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Upload input and LUT
        let write_input = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_input.wait().map_err(|_| PzError::Unsupported)?;

        let write_lut = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, code_lut, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_lut.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 1: compute bit lengths per symbol
        let n_arg = n as cl_uint;
        let pass1_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_bit_lengths)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass1_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download bit lengths and compute prefix sum on CPU
        // (GPU prefix sum would be faster for very large inputs, but adds complexity)
        let mut bit_lengths = vec![0u32; n];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&bit_lengths_buf, CL_BLOCKING, 0, &mut bit_lengths, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_event.wait().map_err(|_| PzError::Unsupported)?;

        // CPU prefix sum: bit_offsets[i] = sum of bit_lengths[0..i)
        let mut bit_offsets = vec![0u32; n];
        let mut running_sum: u64 = 0;
        for i in 0..n {
            bit_offsets[i] = running_sum as u32;
            running_sum += bit_lengths[i] as u64;
        }
        let total_bits = running_sum as usize;

        // Allocate output buffer (as uint array for atomic OR)
        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let mut output_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_uints,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };

        // Zero the output buffer
        let zeros = vec![0u32; output_uints];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        // Upload bit offsets
        let mut offsets_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let write_offsets = unsafe {
            self.queue
                .enqueue_write_buffer(&mut offsets_buf, CL_BLOCKING, 0, &bit_offsets, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_offsets.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 2: write codewords at computed offsets
        let pass2_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_write_codes)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&offsets_buf)
                .set_arg(&output_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass2_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download output as uint array
        let mut output_data = vec![0u32; output_uints];
        let read_out = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_out.wait().map_err(|_| PzError::Unsupported)?;

        // Convert uint array to bytes (big-endian within each uint to match MSB-first packing)
        let output_bytes_len = total_bits.div_ceil(8);
        let mut output_bytes = vec![0u8; output_bytes_len];
        for (i, &val) in output_data.iter().enumerate() {
            let base = i * 4;
            let bytes = val.to_be_bytes();
            for (j, &b) in bytes.iter().enumerate() {
                if base + j < output_bytes_len {
                    output_bytes[base + j] = b;
                }
            }
        }

        Ok((output_bytes, total_bits))
    }

    /// Perform an exclusive prefix sum on a GPU buffer in-place.
    ///
    /// Uses Blelloch scan (work-efficient parallel prefix sum) with
    /// multi-level reduction for large arrays. Avoids downloading
    /// the buffer to the host for CPU prefix sum.
    pub fn prefix_sum_gpu(&self, buf: &mut Buffer<cl_uint>, n: usize) -> PzResult<()> {
        if n <= 1 {
            if n == 1 {
                // Single element: exclusive prefix sum is just 0
                let zero = vec![0u32; 1];
                let write_event = unsafe {
                    self.queue
                        .enqueue_write_buffer(buf, CL_BLOCKING, 0, &zero, &[])
                        .map_err(|_| PzError::Unsupported)?
                };
                write_event.wait().map_err(|_| PzError::Unsupported)?;
            }
            return Ok(());
        }

        // Use a work-group size that processes 2 elements each (Blelloch scan)
        let max_wg = self.max_work_group_size.min(256);
        let block_size = max_wg * 2; // each work-item handles 2 elements

        if n <= block_size {
            // Single work-group: no need for multi-level scan
            self.run_prefix_sum_block(buf, None, n, max_wg)?;
            return Ok(());
        }

        // Multi-level: split into blocks, scan each, collect block totals
        let num_blocks = n.div_ceil(block_size);

        // Allocate block sums buffer
        let mut block_sums_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Level 1: scan each block, output block totals
        self.run_prefix_sum_block(buf, Some(&mut block_sums_buf), n, max_wg)?;

        // Level 2: recursively scan block totals
        if num_blocks > 1 {
            self.prefix_sum_gpu(&mut block_sums_buf, num_blocks)?;
        }

        // Level 3: apply block offsets to elements
        self.run_prefix_sum_apply(buf, &block_sums_buf, n, max_wg)?;

        Ok(())
    }

    /// Run the PrefixSumBlock kernel on a single level.
    fn run_prefix_sum_block(
        &self,
        data_buf: &mut Buffer<cl_uint>,
        block_sums_buf: Option<&mut Buffer<cl_uint>>,
        n: usize,
        local_size: usize,
    ) -> PzResult<()> {
        let block_size = local_size * 2;
        let num_blocks = n.div_ceil(block_size);
        let global_size = num_blocks * local_size;
        let n_arg = n as cl_uint;

        let kernel_event = match block_sums_buf {
            Some(sums_buf) => unsafe {
                let local_mem_bytes = block_size * std::mem::size_of::<cl_uint>();
                ExecuteKernel::new(&self.kernel_prefix_sum_block)
                    .set_arg(data_buf)
                    .set_arg(sums_buf)
                    .set_arg(&n_arg)
                    .set_arg_local_buffer(local_mem_bytes)
                    .set_local_work_size(local_size)
                    .set_global_work_size(global_size)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            },
            None => unsafe {
                // Null block_sums pointer — kernel checks for NULL
                let null_ptr: *const cl_uint = ptr::null();
                let local_mem_bytes = block_size * std::mem::size_of::<cl_uint>();
                ExecuteKernel::new(&self.kernel_prefix_sum_block)
                    .set_arg(data_buf)
                    .set_arg(&null_ptr)
                    .set_arg(&n_arg)
                    .set_arg_local_buffer(local_mem_bytes)
                    .set_local_work_size(local_size)
                    .set_global_work_size(global_size)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            },
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        Ok(())
    }

    /// Run the PrefixSumApply kernel to add block offsets.
    fn run_prefix_sum_apply(
        &self,
        data_buf: &mut Buffer<cl_uint>,
        block_sums_buf: &Buffer<cl_uint>,
        n: usize,
        local_size: usize,
    ) -> PzResult<()> {
        let block_size = local_size * 2;
        let n_arg = n as cl_uint;
        let block_size_arg = block_size as cl_uint;

        // Use a local_work_size that doesn't exceed the max work group size.
        // The kernel computes block_id from gid / block_size, so any valid
        // local_work_size works.
        let apply_local = local_size.min(self.max_work_group_size);
        let global_size = n.div_ceil(apply_local) * apply_local;

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_prefix_sum_apply)
                .set_arg(data_buf)
                .set_arg(block_sums_buf)
                .set_arg(&n_arg)
                .set_arg(&block_size_arg)
                .set_local_work_size(apply_local)
                .set_global_work_size(global_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        Ok(())
    }

    /// Encode data using Huffman coding entirely on the GPU with GPU prefix sum.
    ///
    /// Same as `huffman_encode` but uses the GPU Blelloch scan for the prefix
    /// sum instead of downloading to host. Eliminates one host↔device round-trip.
    pub fn huffman_encode_gpu_scan(
        &self,
        input: &[u8],
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();

        // Create buffers
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        let mut lut_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // bit_lengths_buf will also serve as bit_offsets_buf after prefix sum
        let mut bit_lengths_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Upload input and LUT
        let write_input = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_input.wait().map_err(|_| PzError::Unsupported)?;

        let write_lut = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, code_lut, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_lut.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 1: compute bit lengths per symbol
        let n_arg = n as cl_uint;
        let pass1_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_bit_lengths)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass1_event.wait().map_err(|_| PzError::Unsupported)?;

        // We need the total bits before doing the scan.
        // Read the last element + its bit length to get the total.
        // First, save the last element before the scan overwrites it.
        let mut last_val = vec![0u32; 1];
        let read_last = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_val,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_last.wait().map_err(|_| PzError::Unsupported)?;
        let last_bit_length = last_val[0];

        // GPU prefix sum (exclusive): bit_lengths → bit_offsets
        self.prefix_sum_gpu(&mut bit_lengths_buf, n)?;

        // Read the last offset to compute total_bits
        let mut last_offset = vec![0u32; 1];
        let read_offset = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_offset,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_offset.wait().map_err(|_| PzError::Unsupported)?;
        let total_bits = (last_offset[0] + last_bit_length) as usize;

        // Allocate output buffer
        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let mut output_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_uints,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };

        // Zero the output buffer
        let zeros = vec![0u32; output_uints];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 2: write codewords at GPU-computed offsets
        let pass2_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_write_codes)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf) // now contains bit_offsets
                .set_arg(&output_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass2_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download output
        let mut output_data = vec![0u32; output_uints];
        let read_out = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_out.wait().map_err(|_| PzError::Unsupported)?;

        // Convert uint array to bytes (big-endian to match MSB-first packing)
        let output_bytes_len = total_bits.div_ceil(8);
        let mut output_bytes = vec![0u8; output_bytes_len];
        for (i, &val) in output_data.iter().enumerate() {
            let base = i * 4;
            let bytes = val.to_be_bytes();
            for (j, &b) in bytes.iter().enumerate() {
                if base + j < output_bytes_len {
                    output_bytes[base + j] = b;
                }
            }
        }

        Ok((output_bytes, total_bits))
    }

    /// GPU-chained Deflate compression: LZ77 + Huffman on GPU.
    ///
    /// Performs GPU LZ77 match finding followed by GPU Huffman encoding
    /// with minimal host↔device transfers. The LZ77 output is uploaded
    /// once and stays on the GPU: a `ByteHistogram` kernel computes byte
    /// frequencies on-device (downloading only 1KB of histogram data
    /// instead of the full LZ77 stream), and Huffman encoding reuses the
    /// same GPU buffer.
    ///
    /// **Data flow:**
    /// 1. GPU: LZ77 hash-table match finding → download match array
    /// 2. CPU: deduplicate + serialize matches (sequential, unavoidable)
    /// 3. GPU: upload LZ77 bytes once → run ByteHistogram → download 256×u32 (1KB)
    /// 4. CPU: build Huffman tree from histogram, produce code LUT
    /// 5. GPU: Huffman encode (reusing LZ77 buffer) with GPU prefix sum
    /// 6. GPU: download final encoded bitstream
    ///
    /// Returns the serialized Deflate block data (lz_len + total_bits +
    /// freq_table + huffman_data), ready for the pipeline container.
    pub fn deflate_chained(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        // Stage 1: GPU LZ77 compression (match finding + dedupe + serialize)
        let lz_data = self.lz77_compress(input, KernelVariant::HashTable)?;
        let lz_len = lz_data.len();

        if lz_data.is_empty() {
            return Err(PzError::InvalidInput);
        }

        let n = lz_data.len();

        // Upload LZ77 bytes to GPU once — this buffer is reused for both
        // histogram and Huffman encoding, eliminating redundant transfers.
        let mut lz_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lz_buf, CL_BLOCKING, 0, &lz_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: upload LZ77 data", &write_event);

        // Stage 2: GPU ByteHistogram — compute frequencies on-device.
        // Only 1KB (256×u32) is downloaded instead of the full LZ77 stream.
        let mut hist_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let zeros = vec![0u32; 256];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut hist_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        let n_arg = n as cl_uint;
        let hist_event = unsafe {
            ExecuteKernel::new(&self.kernel_byte_histogram)
                .set_arg(&lz_buf)
                .set_arg(&hist_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        hist_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: byte histogram kernel", &hist_event);

        let mut histogram = vec![0u32; 256];
        let read_hist = unsafe {
            self.queue
                .enqueue_read_buffer(&hist_buf, CL_BLOCKING, 0, &mut histogram, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_hist.wait().map_err(|_| PzError::Unsupported)?;

        // Build Huffman tree from GPU-computed histogram (CPU — tree construction is fast)
        let mut freq = crate::frequency::FrequencyTable::new();
        for (i, &count) in histogram.iter().enumerate() {
            freq.byte[i] = count;
        }
        freq.total = freq.byte.iter().map(|&c| c as u64).sum();
        freq.used = freq.byte.iter().filter(|&&c| c > 0).count() as u32;

        let tree = crate::huffman::HuffmanTree::from_frequency_table(&freq)
            .ok_or(PzError::InvalidInput)?;
        let freq_table = tree.serialize_frequencies();

        // Build the packed code LUT for GPU
        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        // Stage 3: GPU Huffman encoding with GPU prefix sum.
        // Reuses lz_buf (already on device) — no re-upload needed.
        let mut lut_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let write_lut = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, &code_lut, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_lut.wait().map_err(|_| PzError::Unsupported)?;

        // bit_lengths_buf also serves as bit_offsets_buf after prefix sum
        let mut bit_lengths_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Pass 1: compute bit lengths per symbol
        let pass1_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_bit_lengths)
                .set_arg(&lz_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass1_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: huffman bit lengths", &pass1_event);

        // Save last bit length before prefix sum overwrites it
        let mut last_val = vec![0u32; 1];
        let read_last = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_val,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_last.wait().map_err(|_| PzError::Unsupported)?;
        let last_bit_length = last_val[0];

        // GPU prefix sum (exclusive): bit_lengths → bit_offsets
        self.prefix_sum_gpu(&mut bit_lengths_buf, n)?;

        // Read the last offset to compute total_bits
        let mut last_offset = vec![0u32; 1];
        let read_offset = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_offset,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_offset.wait().map_err(|_| PzError::Unsupported)?;
        let total_bits = (last_offset[0] + last_bit_length) as usize;

        // Allocate and zero output buffer
        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Err(PzError::InvalidInput);
        }

        let mut output_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_uints,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let out_zeros = vec![0u32; output_uints];
        let zero_out = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &out_zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_out.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 2: write codewords at GPU-computed offsets
        let pass2_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_write_codes)
                .set_arg(&lz_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf) // now contains bit_offsets
                .set_arg(&output_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass2_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: huffman write codes", &pass2_event);

        // Download final encoded bitstream
        let mut output_data = vec![0u32; output_uints];
        let read_out = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_out.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: download bitstream", &read_out);

        // Convert uint array to bytes (big-endian to match MSB-first packing)
        let output_bytes_len = total_bits.div_ceil(8);
        let mut huffman_data = vec![0u8; output_bytes_len];
        for (i, &val) in output_data.iter().enumerate() {
            let base = i * 4;
            let bytes = val.to_be_bytes();
            for (j, &b) in bytes.iter().enumerate() {
                if base + j < output_bytes_len {
                    huffman_data[base + j] = b;
                }
            }
        }

        // Serialize in the same format as the CPU Deflate block
        let mut output = Vec::new();
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());
        output.extend_from_slice(&(total_bits as u32).to_le_bytes());
        for &freq in &freq_table {
            output.extend_from_slice(&freq.to_le_bytes());
        }
        output.extend_from_slice(&huffman_data);

        Ok(output)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_match_struct_size() {
        // GpuMatch must be 12 bytes to match the OpenCL kernel struct layout
        assert_eq!(std::mem::size_of::<GpuMatch>(), 12);
    }

    #[test]
    fn test_dedupe_all_literals() {
        // Simulate GPU output where no matches were found (all literals)
        let input = b"abcdef";
        let gpu_matches: Vec<GpuMatch> = input
            .iter()
            .map(|&b| GpuMatch {
                offset: 0,
                length: 0,
                next: b,
                _pad: [0; 3],
            })
            .collect();

        let result = dedupe_gpu_matches(&gpu_matches, input);
        assert_eq!(result.len(), 6);
        for (i, m) in result.iter().enumerate() {
            assert_eq!(m.offset, 0);
            assert_eq!(m.length, 0);
            assert_eq!(m.next, input[i]);
        }
    }

    #[test]
    fn test_dedupe_with_matches() {
        // Simulate GPU output: position 0 is literal 'a', positions 1-3 have
        // a match of length 3, etc.
        let input = b"abcabc";
        let gpu_matches = vec![
            GpuMatch {
                offset: 0,
                length: 0,
                next: b'a',
                _pad: [0; 3],
            },
            GpuMatch {
                offset: 0,
                length: 0,
                next: b'b',
                _pad: [0; 3],
            },
            GpuMatch {
                offset: 0,
                length: 0,
                next: b'c',
                _pad: [0; 3],
            },
            GpuMatch {
                offset: 3,
                length: 2,
                next: b'c',
                _pad: [0; 3],
            },
            GpuMatch {
                offset: 3,
                length: 1,
                next: b'c',
                _pad: [0; 3],
            }, // overlapping, skipped
            GpuMatch {
                offset: 3,
                length: 0,
                next: b'c',
                _pad: [0; 3],
            }, // overlapping, skipped
        ];

        let result = dedupe_gpu_matches(&gpu_matches, input);
        // Position 0: literal 'a' -> advance 1
        // Position 1: literal 'b' -> advance 1
        // Position 2: literal 'c' -> advance 1
        // Position 3: match(3,2) + literal 'c' -> advance 3
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].next, b'a');
        assert_eq!(result[3].offset, 3);
        assert_eq!(result[3].length, 2);
    }

    #[test]
    fn test_dedupe_empty() {
        let result = dedupe_gpu_matches(&[], &[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_probe_devices_does_not_panic() {
        // This should never panic, even without OpenCL runtime
        let devices = probe_devices();
        // We can't assert a specific count since it depends on the environment
        let _ = devices;
    }

    #[test]
    fn test_device_count_does_not_panic() {
        let count = device_count();
        let _ = count;
    }

    // Integration tests that require an actual OpenCL device.
    // These are gated on the device being available at runtime.

    #[test]
    fn test_engine_creation() {
        // This test will pass if OpenCL is available, skip otherwise
        match OpenClEngine::new() {
            Ok(engine) => {
                assert!(!engine.device_name().is_empty());
                assert!(engine.max_work_group_size() > 0);
            }
            Err(PzError::Unsupported) => {
                // No OpenCL device available, that's fine
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_gpu_lz77_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return, // skip
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello world";
        let compressed = engine
            .lz77_compress(input, KernelVariant::Batch)
            .expect("GPU compression failed");

        let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");

        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_gpu_lz77_per_position_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return, // skip
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
        let compressed = engine
            .lz77_compress(input, KernelVariant::PerPosition)
            .expect("GPU compression failed");

        let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");

        assert_eq!(&decompressed, &input[..]);
    }

    #[test]
    fn test_gpu_lz77_empty_input() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let result = engine.lz77_compress(b"", KernelVariant::Batch).unwrap();
        assert!(result.is_empty());
    }

    // --- Hash-table LZ77 GPU tests ---

    #[test]
    fn test_gpu_lz77_hash_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello world";
        let compressed = engine
            .lz77_compress(input, KernelVariant::HashTable)
            .expect("GPU hash compression failed");

        let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_gpu_lz77_hash_round_trip_longer() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
        let compressed = engine
            .lz77_compress(input, KernelVariant::HashTable)
            .expect("GPU hash compression failed");

        let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
        assert_eq!(&decompressed, &input[..]);
    }

    #[test]
    fn test_gpu_lz77_hash_round_trip_large() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = engine
            .lz77_compress(&input, KernelVariant::HashTable)
            .expect("GPU hash compression failed");

        let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gpu_lz77_hash_no_matches() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"abcdefgh";
        let compressed = engine
            .lz77_compress(input, KernelVariant::HashTable)
            .expect("GPU hash compression failed");

        let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_gpu_lz77_hash_binary_data() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let compressed = engine
            .lz77_compress(&input, KernelVariant::HashTable)
            .expect("GPU hash compression failed");

        let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
        assert_eq!(decompressed, input);
    }

    // --- BWT GPU tests ---

    #[test]
    fn test_gpu_bwt_banana() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"banana";
        let gpu_result = engine.bwt_encode(input).unwrap();
        let cpu_result = crate::bwt::encode(input).unwrap();

        assert_eq!(gpu_result.data, cpu_result.data, "BWT data mismatch");
        assert_eq!(
            gpu_result.primary_index, cpu_result.primary_index,
            "primary_index mismatch"
        );

        // Round-trip through CPU decode
        let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_gpu_bwt_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // GPU uses circular prefix-doubling (like naive SA) which may produce
        // a different rotation order than CPU SA-IS for periodic inputs, but
        // both are valid BWTs that round-trip correctly.
        let input = b"hello world hello world hello world";
        let gpu_result = engine.bwt_encode(input).unwrap();
        let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_gpu_bwt_binary_data() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input: Vec<u8> = (0..=255).collect();
        let gpu_result = engine.bwt_encode(&input).unwrap();
        let cpu_result = crate::bwt::encode(&input).unwrap();

        assert_eq!(gpu_result.data, cpu_result.data);
        assert_eq!(gpu_result.primary_index, cpu_result.primary_index);

        let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_gpu_bwt_medium_sizes() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // Test sizes that cross the multi-workgroup boundary.
        // Uses round-trip verification since the GPU (circular prefix-doubling)
        // may order identical rotations differently from CPU SA-IS on periodic inputs.
        for size in [257, 300, 400, 500, 512, 513, 768, 1024] {
            let mut input = Vec::with_capacity(size);
            for i in 0..size {
                input.push((i % 256) as u8);
            }
            let gpu_result = engine.bwt_encode(&input).unwrap_or_else(|e| {
                panic!("GPU BWT failed for size {}: {:?}", size, e);
            });
            let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
            assert_eq!(decoded, input, "Round-trip failed for size {}", size);
        }
    }

    #[test]
    fn test_gpu_bwt_single_byte() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"x";
        let gpu_result = engine.bwt_encode(input).unwrap();
        assert_eq!(gpu_result.data, vec![b'x']);
        assert_eq!(gpu_result.primary_index, 0);
    }

    #[test]
    fn test_gpu_bwt_all_same() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = vec![b'a'; 64];
        let gpu_result = engine.bwt_encode(&input).unwrap();
        let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_gpu_bwt_large_structured() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // 4KB of structured data — enough to exercise multi-level prefix sum
        // with small workgroup sizes and verify the GPU rank assignment pipeline.
        let mut input = Vec::new();
        for i in 0..256u16 {
            input.extend_from_slice(&i.to_le_bytes());
        }
        for _ in 0..80 {
            input.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
        }
        while input.len() < 4096 {
            input.push(b'x');
        }
        input.truncate(4096);

        let gpu_result = engine.bwt_encode(&input).unwrap();
        let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    // --- Top-K match finding tests ---

    #[test]
    fn test_gpu_candidate_struct_size() {
        // GpuCandidate must be 4 bytes to match the OpenCL kernel struct layout
        assert_eq!(std::mem::size_of::<GpuCandidate>(), 4);
    }

    #[test]
    fn test_gpu_topk_empty_input() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let table = engine.find_topk_matches(b"").unwrap();
        assert_eq!(table.input_len, 0);
    }

    #[test]
    fn test_gpu_topk_produces_valid_candidates() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello world";
        let table = engine.find_topk_matches(input).unwrap();

        assert_eq!(table.input_len, input.len());
        assert_eq!(table.k, TOPK_K);

        // Verify candidates are valid: offsets point to matching data
        for pos in 0..input.len() {
            for cand in table.at(pos) {
                if cand.length == 0 {
                    continue;
                }
                let offset = cand.offset as usize;
                let length = cand.length as usize;
                assert!(offset <= pos, "offset {} > pos {}", offset, pos);
                let match_start = pos - offset;
                for j in 0..length.min(input.len() - pos) {
                    assert_eq!(
                        input[match_start + j],
                        input[pos + j],
                        "mismatch at pos {} offset {} len {} byte {}",
                        pos,
                        offset,
                        length,
                        j
                    );
                }
            }
        }
    }

    #[test]
    fn test_gpu_topk_optimal_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
        let table = engine.find_topk_matches(input).unwrap();
        let compressed = crate::optimal::compress_optimal_with_table(input, &table).unwrap();
        let decompressed = crate::lz77::decompress(&compressed).unwrap();
        assert_eq!(&decompressed, &input[..]);
    }

    #[test]
    fn test_gpu_topk_optimal_large_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let table = engine.find_topk_matches(&input).unwrap();
        let compressed = crate::optimal::compress_optimal_with_table(&input, &table).unwrap();
        let decompressed = crate::lz77::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- GPU Huffman encoding tests ---

    #[test]
    fn test_gpu_byte_histogram() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"aabbccdd";
        let hist = engine.byte_histogram(input).unwrap();
        assert_eq!(hist[b'a' as usize], 2);
        assert_eq!(hist[b'b' as usize], 2);
        assert_eq!(hist[b'c' as usize], 2);
        assert_eq!(hist[b'd' as usize], 2);
        assert_eq!(hist[0], 0);
    }

    #[test]
    fn test_gpu_byte_histogram_empty() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let hist = engine.byte_histogram(&[]).unwrap();
        assert!(hist.iter().all(|&c| c == 0));
    }

    #[test]
    fn test_gpu_huffman_encode_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello world!";
        let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

        // Build the packed LUT: (bits << 24) | codeword
        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        let (gpu_encoded, gpu_bits) = engine.huffman_encode(input, &code_lut).unwrap();
        let (cpu_encoded, cpu_bits) = tree.encode(input).unwrap();

        assert_eq!(gpu_bits, cpu_bits, "bit counts differ");
        // The byte representations should match
        assert_eq!(
            gpu_encoded,
            cpu_encoded,
            "encoded data differs: gpu {} bytes, cpu {} bytes",
            gpu_encoded.len(),
            cpu_encoded.len()
        );

        // Verify round-trip by decoding with CPU decoder
        let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_gpu_huffman_encode_larger() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }

        let tree = crate::huffman::HuffmanTree::from_data(&input).unwrap();
        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        let (gpu_encoded, gpu_bits) = engine.huffman_encode(&input, &code_lut).unwrap();
        let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_gpu_is_cpu_device() {
        // Just verify the method doesn't panic
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };
        // The value depends on hardware — just check it returns a bool
        let _is_cpu = engine.is_cpu_device();
    }

    // --- GPU prefix sum tests ---

    #[test]
    fn test_gpu_prefix_sum_small() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = vec![1u32, 2, 3, 4, 5];
        let n = input.len();

        let mut buf = unsafe {
            Buffer::<cl_uint>::create(&engine.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .unwrap()
        };
        unsafe {
            engine
                .queue
                .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, &input, &[])
                .unwrap()
                .wait()
                .unwrap();
        }

        engine.prefix_sum_gpu(&mut buf, n).unwrap();

        let mut result = vec![0u32; n];
        unsafe {
            engine
                .queue
                .enqueue_read_buffer(&buf, CL_BLOCKING, 0, &mut result, &[])
                .unwrap()
                .wait()
                .unwrap();
        }

        // Exclusive prefix sum: [0, 1, 3, 6, 10]
        assert_eq!(result, vec![0, 1, 3, 6, 10]);
    }

    #[test]
    fn test_gpu_prefix_sum_large() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        // Large enough to require multi-level scan
        let n = 2048;
        let input: Vec<u32> = (0..n as u32).map(|i| (i % 10) + 1).collect();

        let mut buf = unsafe {
            Buffer::<cl_uint>::create(&engine.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .unwrap()
        };
        unsafe {
            engine
                .queue
                .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, &input, &[])
                .unwrap()
                .wait()
                .unwrap();
        }

        engine.prefix_sum_gpu(&mut buf, n).unwrap();

        let mut result = vec![0u32; n];
        unsafe {
            engine
                .queue
                .enqueue_read_buffer(&buf, CL_BLOCKING, 0, &mut result, &[])
                .unwrap()
                .wait()
                .unwrap();
        }

        // Verify against CPU prefix sum
        let mut expected = vec![0u32; n];
        let mut sum: u64 = 0;
        for i in 0..n {
            expected[i] = sum as u32;
            sum += input[i] as u64;
        }

        assert_eq!(result, expected);
    }

    // --- GPU Huffman with GPU scan tests ---

    #[test]
    fn test_gpu_huffman_encode_gpu_scan_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello world!";
        let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        let (gpu_encoded, gpu_bits) = engine.huffman_encode_gpu_scan(input, &code_lut).unwrap();
        let (cpu_encoded, cpu_bits) = tree.encode(input).unwrap();

        assert_eq!(gpu_bits, cpu_bits, "bit counts differ");
        assert_eq!(gpu_encoded, cpu_encoded, "encoded data differs");

        let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
        assert_eq!(decoded, input);
    }

    // --- GPU chained Deflate tests ---

    #[test]
    fn test_gpu_deflate_chained_round_trip() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
        let block_data = engine.deflate_chained(input).unwrap();

        // Decompress using the standard CPU Deflate decoder
        let decompressed = crate::pipeline::decompress(&{
            // Build a proper V2 PZ container around the block data
            let mut container = Vec::new();
            container.extend_from_slice(&[b'P', b'Z', 2, 0]); // magic + version=2 + pipeline=Deflate
            container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original length
            container.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
            container.extend_from_slice(&(block_data.len() as u32).to_le_bytes()); // compressed_len
            container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original_len
            container.extend_from_slice(&block_data);
            container
        })
        .unwrap();

        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gpu_deflate_chained_larger() {
        let engine = match OpenClEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("Unexpected error: {:?}", e),
        };

        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }

        let block_data = engine.deflate_chained(&input).unwrap();

        let decompressed = crate::pipeline::decompress(&{
            let mut container = Vec::new();
            container.extend_from_slice(&[b'P', b'Z', 2, 0]); // magic + version=2 + pipeline=Deflate
            container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original length
            container.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
            container.extend_from_slice(&(block_data.len() as u32).to_le_bytes()); // compressed_len
            container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original_len
            container.extend_from_slice(&block_data);
            container
        })
        .unwrap();

        assert_eq!(decompressed, input);
    }
}
