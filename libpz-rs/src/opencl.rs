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
//! # {
//! use pz::opencl::{OpenClEngine, KernelVariant};
//!
//! let engine = OpenClEngine::new()?;
//! println!("Using device: {}", engine.device_name());
//!
//! let input = b"hello world hello world";
//! let matches = engine.find_matches(input, KernelVariant::Batch)?;
//! # }
//! ```

use crate::lz77::Match;
use crate::{PzError, PzResult};

use opencl3::command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE};
use opencl3::context::Context;
use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_GPU};
use opencl3::kernel::{ExecuteKernel, Kernel};
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{cl_device_type, cl_uint, CL_BLOCKING};

use std::ptr;

/// Embedded OpenCL kernel source: one work-item per input position,
/// 128KB sliding window.
const LZ77_KERNEL_SOURCE: &str = include_str!("../kernels/lz77.cl");

/// Embedded OpenCL kernel source: batched variant where each work-item
/// processes STEP_SIZE (32) consecutive positions with a 32KB window.
const LZ77_BATCH_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_batch.cl");

/// Step size used by the batch kernel (must match STEP_SIZE in lz77_batch.cl).
const BATCH_STEP_SIZE: usize = 32;

/// Minimum input size below which GPU overhead exceeds benefit.
/// For small inputs, the CPU reference implementation is faster.
pub const MIN_GPU_INPUT_SIZE: usize = 64 * 1024; // 64KB

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

/// Which LZ77 kernel variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    /// One work-item per position, 128KB window. Better compression
    /// quality but higher GPU memory and compute cost.
    PerPosition,
    /// Batched: each work-item handles STEP_SIZE positions, 32KB window.
    /// Lower overhead, better for large inputs.
    Batch,
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
    /// Device name for diagnostics.
    device_name: String,
    /// Maximum work-group size.
    max_work_group_size: usize,
}

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
        let all_ids = get_all_devices(CL_DEVICE_TYPE_ALL)
            .map_err(|_| PzError::Unsupported)?;

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
        let context = Context::from_device(&device)
            .map_err(|_| PzError::Unsupported)?;

        let queue = CommandQueue::create_default_with_properties(
            &context,
            CL_QUEUE_PROFILING_ENABLE,
            0,
        )
        .map_err(|_| PzError::Unsupported)?;

        // Compile both kernel variants
        let program_per_pos =
            Program::create_and_build_from_source(&context, LZ77_KERNEL_SOURCE, "-Werror")
                .map_err(|_| PzError::Unsupported)?;

        let program_batch =
            Program::create_and_build_from_source(&context, LZ77_BATCH_KERNEL_SOURCE, "-Werror")
                .map_err(|_| PzError::Unsupported)?;

        let kernel_per_pos = Kernel::create(&program_per_pos, "Encode")
            .map_err(|_| PzError::Unsupported)?;

        let kernel_batch = Kernel::create(&program_batch, "Encode")
            .map_err(|_| PzError::Unsupported)?;

        Ok(OpenClEngine {
            _device: device,
            context,
            queue,
            kernel_per_pos,
            kernel_batch,
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
            Buffer::<GpuMatch>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                input_len,
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

        // Execute kernel
        match variant {
            KernelVariant::PerPosition => {
                self.run_per_position_kernel(&input_buf, &output_buf, input_len)?;
            }
            KernelVariant::Batch => {
                self.run_batch_kernel(&input_buf, &output_buf, input_len)?;
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
            GpuMatch { offset: 0, length: 0, next: b'a', _pad: [0; 3] },
            GpuMatch { offset: 0, length: 0, next: b'b', _pad: [0; 3] },
            GpuMatch { offset: 0, length: 0, next: b'c', _pad: [0; 3] },
            GpuMatch { offset: 3, length: 2, next: b'c', _pad: [0; 3] },
            GpuMatch { offset: 3, length: 1, next: b'c', _pad: [0; 3] }, // overlapping, skipped
            GpuMatch { offset: 3, length: 0, next: b'c', _pad: [0; 3] }, // overlapping, skipped
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

        let decompressed =
            crate::lz77::decompress(&compressed).expect("decompression failed");

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

        let decompressed =
            crate::lz77::decompress(&compressed).expect("decompression failed");

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
}
