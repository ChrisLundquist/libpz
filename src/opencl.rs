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

/// Embedded OpenCL kernel source: bitonic sort step for BWT suffix array.
const BWT_SORT_KERNEL_SOURCE: &str = include_str!("../kernels/bwt_sort.cl");

/// Embedded OpenCL kernel source: top-K match finding for optimal parsing.
const LZ77_TOPK_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_topk.cl");

/// Embedded OpenCL kernel source: hash-table-based LZ77 match finding.
/// Two-pass: BuildHashTable scatters positions, FindMatches searches buckets.
const LZ77_HASH_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_hash.cl");

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
    /// Compiled BWT bitonic sort step kernel.
    kernel_bwt_sort_step: Kernel,
    /// Compiled top-K LZ77 match finding kernel.
    kernel_topk: Kernel,
    /// Compiled hash-table build kernel.
    kernel_hash_build: Kernel,
    /// Compiled hash-table find-matches kernel.
    kernel_hash_find: Kernel,
    /// Device name for diagnostics.
    device_name: String,
    /// Maximum work-group size.
    max_work_group_size: usize,
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
        Self::with_device_preference(true)
    }

    /// Create a new engine with explicit GPU preference.
    ///
    /// If `prefer_gpu` is true, selects the first GPU device (falling back
    /// to any device). If false, selects the first available device regardless
    /// of type.
    pub fn with_device_preference(prefer_gpu: bool) -> PzResult<Self> {
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

        // Create context and command queue
        let context = Context::from_device(&device).map_err(|_| PzError::Unsupported)?;

        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)
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

        // Compile BWT bitonic sort kernel
        let program_bwt =
            Program::create_and_build_from_source(&context, BWT_SORT_KERNEL_SOURCE, "-Werror")
                .map_err(|_| PzError::Unsupported)?;

        let kernel_bwt_sort_step =
            Kernel::create(&program_bwt, "bitonic_sort_step").map_err(|_| PzError::Unsupported)?;

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

        Ok(OpenClEngine {
            _device: device,
            context,
            queue,
            kernel_per_pos,
            kernel_batch,
            kernel_bwt_sort_step,
            kernel_topk,
            kernel_hash_build,
            kernel_hash_find,
            device_name,
            max_work_group_size,
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

        Ok(())
    }

    /// GPU-accelerated LZ77 compression.
    ///
    /// Uses the GPU to find matches, deduplicates them, and serializes
    /// to the same byte format as the CPU `lz77::compress()`.
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
    /// Uses the GPU for the expensive suffix array sort steps (bitonic sort),
    /// with CPU handling rank assignment (cheap O(n) sequential scan).
    /// Produces output identical to the CPU `bwt::encode()`.
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

    /// Build suffix array on the GPU using prefix-doubling with bitonic sort.
    ///
    /// Each doubling step sorts sa[] by (rank[sa[i]], rank[(sa[i]+k) % n]).
    /// The sort is done on the GPU via bitonic sort; rank assignment is done
    /// on the CPU (cheap O(n) scan after each sort).
    fn bwt_build_suffix_array(&self, input: &[u8]) -> PzResult<Vec<usize>> {
        let n = input.len();
        if n <= 1 {
            return Ok(if n == 0 { Vec::new() } else { vec![0] });
        }

        // Bitonic sort requires power-of-2 size. We pad sa with sentinel
        // values that sort to the end (max rank).
        let padded_n = n.next_power_of_two();

        // Initialize sa and rank arrays (as u32 for GPU)
        let mut sa_host: Vec<cl_uint> = (0..padded_n as cl_uint).collect();
        let mut rank_host: Vec<cl_uint> = vec![cl_uint::MAX; padded_n];
        for i in 0..n {
            rank_host[i] = input[i] as cl_uint;
        }
        // Sentinel entries: sa values >= n, rank = MAX so they sort to the end

        // Allocate GPU buffers
        let mut sa_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let mut rank_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
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

        // Prefix-doubling loop
        let mut k_step: usize = 1;
        while k_step < n {
            // Run bitonic sort on GPU
            self.run_bitonic_sort(
                &mut sa_buf,
                &rank_buf,
                padded_n,
                n as cl_uint,
                k_step as cl_uint,
            )?;

            // Read back sorted sa
            let read_sa = unsafe {
                self.queue
                    .enqueue_read_buffer(&sa_buf, CL_BLOCKING, 0, &mut sa_host, &[])
                    .map_err(|_| PzError::InvalidInput)?
            };
            read_sa.wait().map_err(|_| PzError::InvalidInput)?;

            // CPU rank assignment (O(n) sequential scan)
            let mut new_rank = vec![cl_uint::MAX; padded_n];
            // First real entry gets rank 0
            if (sa_host[0] as usize) < n {
                new_rank[sa_host[0] as usize] = 0;
            }

            let mut current_rank: cl_uint = 0;
            for i in 1..padded_n {
                let curr_sa = sa_host[i] as usize;
                let prev_sa = sa_host[i - 1] as usize;

                // Sentinel entries keep MAX rank
                if curr_sa >= n {
                    continue;
                }

                if prev_sa >= n {
                    // Previous was sentinel, this is a real entry
                    current_rank += 1;
                    new_rank[curr_sa] = current_rank;
                    continue;
                }

                // Compare composite keys
                let r1_curr = rank_host[curr_sa];
                let r2_curr = rank_host[(curr_sa + k_step) % n];
                let r1_prev = rank_host[prev_sa];
                let r2_prev = rank_host[(prev_sa + k_step) % n];

                if r1_curr != r1_prev || r2_curr != r2_prev {
                    current_rank += 1;
                }
                new_rank[curr_sa] = current_rank;
            }

            rank_host = new_rank;

            // Check convergence: all ranks unique when max_rank == n-1
            if current_rank as usize == n - 1 {
                break;
            }

            // Upload new ranks to GPU
            let write_rank = unsafe {
                self.queue
                    .enqueue_write_buffer(&mut rank_buf, CL_BLOCKING, 0, &rank_host, &[])
                    .map_err(|_| PzError::InvalidInput)?
            };
            write_rank.wait().map_err(|_| PzError::InvalidInput)?;

            k_step *= 2;
        }

        // Extract the real suffix array (filter out sentinel entries)
        let sa: Vec<usize> = sa_host
            .iter()
            .filter(|&&v| (v as usize) < n)
            .map(|&v| v as usize)
            .collect();

        if sa.len() != n {
            return Err(PzError::InvalidInput);
        }

        Ok(sa)
    }

    /// Run a complete bitonic sort on the sa buffer.
    ///
    /// Executes O(log^2 padded_n) kernel launches to fully sort the array.
    fn run_bitonic_sort(
        &self,
        sa_buf: &mut Buffer<cl_uint>,
        rank_buf: &Buffer<cl_uint>,
        padded_n: usize,
        n: cl_uint,
        k_doubling: cl_uint,
    ) -> PzResult<()> {
        let padded_n_arg = padded_n as cl_uint;

        // Bitonic sort: for each stage (power of 2 block size),
        // run sub-stages that halve the comparison distance.
        let mut k_sort: usize = 2;
        while k_sort <= padded_n {
            let mut j: usize = k_sort / 2;
            while j >= 1 {
                let kernel_event = unsafe {
                    ExecuteKernel::new(&self.kernel_bwt_sort_step)
                        .set_arg(sa_buf)
                        .set_arg(rank_buf)
                        .set_arg(&n)
                        .set_arg(&padded_n_arg)
                        .set_arg(&k_doubling)
                        .set_arg(&(j as cl_uint))
                        .set_arg(&(k_sort as cl_uint))
                        .set_global_work_size(padded_n)
                        .enqueue_nd_range(&self.queue)
                        .map_err(|_| PzError::Unsupported)?
                };
                kernel_event.wait().map_err(|_| PzError::Unsupported)?;

                j /= 2;
            }
            k_sort *= 2;
        }
        Ok(())
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
            offset: gm.offset,
            length: match_length as u32,
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

        let input = b"hello world hello world hello world";
        let gpu_result = engine.bwt_encode(input).unwrap();
        let cpu_result = crate::bwt::encode(input).unwrap();

        assert_eq!(gpu_result.data, cpu_result.data);
        assert_eq!(gpu_result.primary_index, cpu_result.primary_index);

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
        let cpu_result = crate::bwt::encode(&input).unwrap();

        assert_eq!(gpu_result.data, cpu_result.data);
        assert_eq!(gpu_result.primary_index, cpu_result.primary_index);
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
}
