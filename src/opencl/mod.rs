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
use crate::gpu_cost::KernelCost;
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
use std::sync::OnceLock;

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

/// Embedded OpenCL kernel source: FSE (tANS) decode.
/// One work-item per interleaved stream.
const FSE_DECODE_KERNEL_SOURCE: &str = include_str!("../../kernels/fse_decode.cl");

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
pub(crate) struct GpuMatch {
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

// ---------------------------------------------------------------------------
// Lazy kernel group structs — one per OpenCL program compilation unit.
// Kernels are compiled on first use via OnceLock, not at engine creation.
// ---------------------------------------------------------------------------

/// LZ77 per-position kernel (1 kernel from lz77.cl).
struct Lz77PerPosKernels {
    per_pos: Kernel,
}

/// LZ77 batch kernel (1 kernel from lz77_batch.cl).
struct Lz77BatchKernels {
    batch: Kernel,
}

/// LZ77 top-K kernel (1 kernel from lz77_topk.cl).
struct Lz77TopkKernels {
    topk: Kernel,
}

/// LZ77 hash-table kernels (2 kernels from lz77_hash.cl).
struct Lz77HashKernels {
    build: Kernel,
    find: Kernel,
}

/// BWT rank kernels (4 kernels from bwt_rank.cl).
struct BwtRankKernels {
    rank_compare: Kernel,
    prefix_sum_local: Kernel,
    prefix_sum_propagate: Kernel,
    rank_scatter: Kernel,
}

/// BWT radix sort kernels (4 kernels from bwt_radix.cl).
struct BwtRadixKernels {
    compute_keys: Kernel,
    histogram: Kernel,
    scatter: Kernel,
    inclusive_to_exclusive: Kernel,
}

/// Huffman encoding kernels (5 kernels from huffman_encode.cl).
struct HuffmanKernels {
    bit_lengths: Kernel,
    write_codes: Kernel,
    byte_histogram: Kernel,
    prefix_sum_block: Kernel,
    prefix_sum_apply: Kernel,
}

/// FSE decode kernel (1 kernel from fse_decode.cl).
struct FseDecodeKernels {
    decode: Kernel,
}

/// OpenCL compute engine.
///
/// Manages the device, context, command queue, and lazily-compiled kernels.
/// Create one engine at library init time and reuse it across calls.
/// Kernels are compiled on first use to avoid paying startup cost for
/// pipelines that don't need them.
///
/// Note: `Debug` is implemented manually because the OpenCL handle
/// types from `opencl3` don't implement `Debug`.
pub struct OpenClEngine {
    _device: Device,
    context: Context,
    queue: CommandQueue,
    // Lazily-compiled kernel groups (one OnceLock per OpenCL program).
    lz77_per_pos: OnceLock<Lz77PerPosKernels>,
    lz77_batch: OnceLock<Lz77BatchKernels>,
    lz77_topk: OnceLock<Lz77TopkKernels>,
    lz77_hash: OnceLock<Lz77HashKernels>,
    bwt_rank: OnceLock<BwtRankKernels>,
    bwt_radix: OnceLock<BwtRadixKernels>,
    huffman: OnceLock<HuffmanKernels>,
    fse_decode: OnceLock<FseDecodeKernels>,
    /// Workgroup size for prefix sum kernels (power of 2, capped at 256).
    scan_workgroup_size: usize,
    /// Device name for diagnostics.
    device_name: String,
    /// Maximum work-group size.
    max_work_group_size: usize,
    /// Whether the selected device is a CPU (vs GPU/accelerator).
    is_cpu: bool,
    /// Whether profiling is enabled (CL_QUEUE_PROFILING_ENABLE).
    profiling: bool,
    /// Device global memory size in bytes.
    global_mem_size: u64,
    /// Parsed cost model for the LZ77 hash kernel (used for batch scheduling).
    cost_lz77_hash: KernelCost,
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
        let global_mem_size = device.global_mem_size().unwrap_or(0);

        // Compute scan workgroup size eagerly (needed for BWT kernel compile flags).
        let capped = max_work_group_size.clamp(1, 256);
        let scan_workgroup_size = 1 << (usize::BITS - 1 - capped.leading_zeros());

        // Parse kernel cost annotations for batch scheduling (cheap string parse).
        let cost_lz77_hash = KernelCost::parse(LZ77_HASH_KERNEL_SOURCE)
            .expect("lz77_hash.cl missing @pz_cost annotation");

        // All kernel compilation is deferred to first use via OnceLock.
        Ok(OpenClEngine {
            _device: device,
            context,
            queue,
            lz77_per_pos: OnceLock::new(),
            lz77_batch: OnceLock::new(),
            lz77_topk: OnceLock::new(),
            lz77_hash: OnceLock::new(),
            bwt_rank: OnceLock::new(),
            bwt_radix: OnceLock::new(),
            huffman: OnceLock::new(),
            fse_decode: OnceLock::new(),
            scan_workgroup_size,
            device_name,
            max_work_group_size,
            is_cpu,
            profiling,
            global_mem_size,
            cost_lz77_hash,
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

    /// Max blocks of `block_size` bytes in flight for a kernel without
    /// exceeding the GPU memory budget.
    pub(crate) fn max_in_flight(&self, kernel: &KernelCost, block_size: usize) -> usize {
        let per_block = kernel.memory_bytes(block_size);
        if per_block == 0 {
            return 8;
        }
        (self.gpu_memory_budget() / per_block).max(1)
    }

    /// Conservative GPU memory budget: 50% of global_mem_size.
    fn gpu_memory_budget(&self) -> usize {
        (self.global_mem_size as usize) / 2
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

    // -------------------------------------------------------------------
    // Lazy kernel accessors — compile on first use via OnceLock.
    // -------------------------------------------------------------------

    fn kernel_per_pos(&self) -> &Kernel {
        &self
            .lz77_per_pos
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let program = Program::create_and_build_from_source(
                    &self.context,
                    LZ77_KERNEL_SOURCE,
                    "-Werror",
                )
                .expect("failed to compile lz77.cl");
                let group = Lz77PerPosKernels {
                    per_pos: Kernel::create(&program, "Encode")
                        .expect("failed to create Encode kernel"),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile lz77.cl: {ms:.3} ms");
                }
                group
            })
            .per_pos
    }

    fn kernel_batch(&self) -> &Kernel {
        &self
            .lz77_batch
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let program = Program::create_and_build_from_source(
                    &self.context,
                    LZ77_BATCH_KERNEL_SOURCE,
                    "-Werror",
                )
                .expect("failed to compile lz77_batch.cl");
                let group = Lz77BatchKernels {
                    batch: Kernel::create(&program, "Encode")
                        .expect("failed to create Encode kernel"),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile lz77_batch.cl: {ms:.3} ms");
                }
                group
            })
            .batch
    }

    fn kernel_topk(&self) -> &Kernel {
        &self
            .lz77_topk
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let program = Program::create_and_build_from_source(
                    &self.context,
                    LZ77_TOPK_KERNEL_SOURCE,
                    "-Werror",
                )
                .expect("failed to compile lz77_topk.cl");
                let group = Lz77TopkKernels {
                    topk: Kernel::create(&program, "EncodeTopK")
                        .expect("failed to create EncodeTopK kernel"),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile lz77_topk.cl: {ms:.3} ms");
                }
                group
            })
            .topk
    }

    fn lz77_hash_kernels(&self) -> &Lz77HashKernels {
        self.lz77_hash.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let program = Program::create_and_build_from_source(
                &self.context,
                LZ77_HASH_KERNEL_SOURCE,
                "-Werror",
            )
            .expect("failed to compile lz77_hash.cl");
            let group = Lz77HashKernels {
                build: Kernel::create(&program, "BuildHashTable")
                    .expect("failed to create BuildHashTable kernel"),
                find: Kernel::create(&program, "FindMatches")
                    .expect("failed to create FindMatches kernel"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile lz77_hash.cl: {ms:.3} ms");
            }
            group
        })
    }

    fn kernel_hash_build(&self) -> &Kernel {
        &self.lz77_hash_kernels().build
    }

    fn kernel_hash_find(&self) -> &Kernel {
        &self.lz77_hash_kernels().find
    }

    fn bwt_rank_kernels(&self) -> &BwtRankKernels {
        self.bwt_rank.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let flags = format!("-Werror -DWORKGROUP_SIZE={}", self.scan_workgroup_size);
            let program = Program::create_and_build_from_source(
                &self.context,
                BWT_RANK_KERNEL_SOURCE,
                &flags,
            )
            .expect("failed to compile bwt_rank.cl");
            let group = BwtRankKernels {
                rank_compare: Kernel::create(&program, "rank_compare")
                    .expect("failed to create rank_compare kernel"),
                prefix_sum_local: Kernel::create(&program, "prefix_sum_local")
                    .expect("failed to create prefix_sum_local kernel"),
                prefix_sum_propagate: Kernel::create(&program, "prefix_sum_propagate")
                    .expect("failed to create prefix_sum_propagate kernel"),
                rank_scatter: Kernel::create(&program, "rank_scatter")
                    .expect("failed to create rank_scatter kernel"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile bwt_rank.cl: {ms:.3} ms");
            }
            group
        })
    }

    fn kernel_rank_compare(&self) -> &Kernel {
        &self.bwt_rank_kernels().rank_compare
    }

    fn kernel_prefix_sum_local(&self) -> &Kernel {
        &self.bwt_rank_kernels().prefix_sum_local
    }

    fn kernel_prefix_sum_propagate(&self) -> &Kernel {
        &self.bwt_rank_kernels().prefix_sum_propagate
    }

    fn kernel_rank_scatter(&self) -> &Kernel {
        &self.bwt_rank_kernels().rank_scatter
    }

    fn bwt_radix_kernels(&self) -> &BwtRadixKernels {
        self.bwt_radix.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let flags = format!("-Werror -DWORKGROUP_SIZE={}", self.scan_workgroup_size);
            let program = Program::create_and_build_from_source(
                &self.context,
                BWT_RADIX_KERNEL_SOURCE,
                &flags,
            )
            .expect("failed to compile bwt_radix.cl");
            let group = BwtRadixKernels {
                compute_keys: Kernel::create(&program, "radix_compute_keys")
                    .expect("failed to create radix_compute_keys kernel"),
                histogram: Kernel::create(&program, "radix_histogram")
                    .expect("failed to create radix_histogram kernel"),
                scatter: Kernel::create(&program, "radix_scatter")
                    .expect("failed to create radix_scatter kernel"),
                inclusive_to_exclusive: Kernel::create(&program, "inclusive_to_exclusive")
                    .expect("failed to create inclusive_to_exclusive kernel"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile bwt_radix.cl: {ms:.3} ms");
            }
            group
        })
    }

    fn kernel_radix_compute_keys(&self) -> &Kernel {
        &self.bwt_radix_kernels().compute_keys
    }

    fn kernel_radix_histogram(&self) -> &Kernel {
        &self.bwt_radix_kernels().histogram
    }

    fn kernel_radix_scatter(&self) -> &Kernel {
        &self.bwt_radix_kernels().scatter
    }

    fn kernel_inclusive_to_exclusive(&self) -> &Kernel {
        &self.bwt_radix_kernels().inclusive_to_exclusive
    }

    fn huffman_kernels(&self) -> &HuffmanKernels {
        self.huffman.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let program = Program::create_and_build_from_source(
                &self.context,
                HUFFMAN_ENCODE_KERNEL_SOURCE,
                "-Werror",
            )
            .expect("failed to compile huffman_encode.cl");
            let group = HuffmanKernels {
                bit_lengths: Kernel::create(&program, "ComputeBitLengths")
                    .expect("failed to create ComputeBitLengths kernel"),
                write_codes: Kernel::create(&program, "WriteCodes")
                    .expect("failed to create WriteCodes kernel"),
                byte_histogram: Kernel::create(&program, "ByteHistogram")
                    .expect("failed to create ByteHistogram kernel"),
                prefix_sum_block: Kernel::create(&program, "PrefixSumBlock")
                    .expect("failed to create PrefixSumBlock kernel"),
                prefix_sum_apply: Kernel::create(&program, "PrefixSumApply")
                    .expect("failed to create PrefixSumApply kernel"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile huffman_encode.cl: {ms:.3} ms");
            }
            group
        })
    }

    fn kernel_huffman_bit_lengths(&self) -> &Kernel {
        &self.huffman_kernels().bit_lengths
    }

    fn kernel_huffman_write_codes(&self) -> &Kernel {
        &self.huffman_kernels().write_codes
    }

    fn kernel_byte_histogram(&self) -> &Kernel {
        &self.huffman_kernels().byte_histogram
    }

    fn kernel_prefix_sum_block(&self) -> &Kernel {
        &self.huffman_kernels().prefix_sum_block
    }

    fn kernel_prefix_sum_apply(&self) -> &Kernel {
        &self.huffman_kernels().prefix_sum_apply
    }

    fn kernel_fse_decode(&self) -> &Kernel {
        &self
            .fse_decode
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let program = Program::create_and_build_from_source(
                    &self.context,
                    FSE_DECODE_KERNEL_SOURCE,
                    "-Werror",
                )
                .expect("failed to compile fse_decode.cl");
                let group = FseDecodeKernels {
                    decode: Kernel::create(&program, "FseDecode")
                        .expect("failed to create FseDecode kernel"),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile fse_decode.cl: {ms:.3} ms");
                }
                group
            })
            .decode
    }
}

// ---------------------------------------------------------------------------
// DeviceBuf — data residing on the GPU, not read back unless requested
// ---------------------------------------------------------------------------

/// A buffer residing on the GPU device.
///
/// Data stays on-device until explicitly downloaded via [`read_to_host()`].
/// This avoids unnecessary PCI-bus round-trips when one GPU stage feeds
/// directly into another (e.g., LZ77 output → Huffman histogram on the
/// same device buffer).
///
/// # Examples
///
/// ```rust,no_run
/// # #[cfg(feature = "opencl")]
/// # fn example() -> pz::PzResult<()> {
/// use pz::opencl::{OpenClEngine, DeviceBuf};
///
/// let engine = OpenClEngine::new()?;
/// let data = b"hello world";
/// let device_buf = DeviceBuf::from_host(&engine, data)?;
///
/// // Data lives on GPU — use it with on-device methods
/// let histogram = engine.byte_histogram_on_device(&device_buf)?;
///
/// // Only download when you actually need the bytes on the host
/// let host_data = device_buf.read_to_host(&engine)?;
/// assert_eq!(&host_data, data);
/// # Ok(())
/// # }
/// ```
pub struct DeviceBuf {
    pub(crate) buf: Buffer<u8>,
    pub(crate) len: usize,
}

impl DeviceBuf {
    /// Upload host data to the GPU, returning a device-resident buffer.
    pub fn from_host(engine: &OpenClEngine, data: &[u8]) -> PzResult<Self> {
        if data.is_empty() {
            // Allocate a 1-byte buffer to avoid zero-size allocation issues
            let buf = unsafe {
                Buffer::<u8>::create(&engine.context, CL_MEM_READ_WRITE, 1, ptr::null_mut())
                    .map_err(|_| PzError::BufferTooSmall)?
            };
            return Ok(DeviceBuf { buf, len: 0 });
        }

        let mut buf = unsafe {
            Buffer::<u8>::create(
                &engine.context,
                CL_MEM_READ_WRITE,
                data.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        let write_event = unsafe {
            engine
                .queue
                .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_event.wait().map_err(|_| PzError::Unsupported)?;

        Ok(DeviceBuf {
            buf,
            len: data.len(),
        })
    }

    /// Allocate a device buffer of the given size.
    ///
    /// **Note:** The buffer contents are *not* guaranteed to be zero-initialized.
    /// Callers that need zeroed memory should write zeros explicitly.
    pub fn alloc(engine: &OpenClEngine, len: usize) -> PzResult<Self> {
        let actual_len = len.max(1); // avoid zero-size allocation
        let buf = unsafe {
            Buffer::<u8>::create(
                &engine.context,
                CL_MEM_READ_WRITE,
                actual_len,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        Ok(DeviceBuf { buf, len })
    }

    /// Download the buffer contents from the GPU to host memory.
    pub fn read_to_host(&self, engine: &OpenClEngine) -> PzResult<Vec<u8>> {
        if self.len == 0 {
            return Ok(Vec::new());
        }

        let mut host_data = vec![0u8; self.len];
        let read_event = unsafe {
            engine
                .queue
                .enqueue_read_buffer(&self.buf, CL_BLOCKING, 0, &mut host_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_event.wait().map_err(|_| PzError::Unsupported)?;

        Ok(host_data)
    }

    /// The logical length of the data in this buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer is empty (zero-length data).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// An opaque GPU-resident buffer of LZ77 match results.
///
/// Produced by [`OpenClEngine::find_matches_to_device()`] and consumed by
/// [`OpenClEngine::download_and_dedupe()`]. The match data stays on the GPU
/// until explicitly downloaded, enabling zero-copy stage chaining.
pub struct GpuMatchBuf {
    pub(crate) buf: Buffer<GpuMatch>,
    pub(crate) input_len: usize,
}

impl GpuMatchBuf {
    /// The number of input positions this match buffer covers.
    pub fn input_len(&self) -> usize {
        self.input_len
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

        // Clamp offset to u16 range. The PerPosition kernel uses a 128KB
        // window which can produce offsets > u16::MAX; matches with such
        // offsets are unrepresentable, so emit a literal instead.
        let offset = gm.offset;
        if offset > u16::MAX as u32 && match_length > 0 {
            // Unrepresentable match — emit as literal (offset=0, length=0).
            result.push(Match {
                offset: 0,
                length: 0,
                next: input[index],
            });
            index += 1;
            continue;
        }

        result.push(Match {
            offset: offset as u16,
            length: match_length as u16,
            next,
        });

        index += match_length + 1;
    }

    result
}

mod bwt;
pub mod fse;
mod huffman;
pub mod lz77;
pub mod rans;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
