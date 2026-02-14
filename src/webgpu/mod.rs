//! WebGPU (wgpu) GPU backend for libpz.
//!
//! Provides GPU-accelerated implementations of compute-intensive
//! compression stages using the cross-platform wgpu library, which
//! supports Vulkan, Metal, DX12, and WebGPU backends.
//!
//! # Feature Gate
//!
//! This module is only available when compiled with the `webgpu` feature:
//! ```bash
//! cargo build --features webgpu
//! ```
//!
//! # Usage
//!
//! ```rust,no_run
//! # #[cfg(feature = "webgpu")]
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! use pz::webgpu::WebGpuEngine;
//!
//! let engine = WebGpuEngine::new()?;
//! println!("Using device: {}", engine.device_name());
//!
//! let input = b"hello world hello world";
//! let matches = engine.find_matches(input)?;
//! # Ok(())
//! # }
//! ```

use crate::bwt::BwtResult;
use crate::gpu_cost::KernelCost;
use crate::lz77::Match;
use crate::{PzError, PzResult};

use std::sync::OnceLock;

use wgpu::util::DeviceExt;

mod bwt;
mod fse;
mod huffman;
pub(crate) mod lz77;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

/// Embedded WGSL kernel source: top-K match finding for optimal parsing.
const LZ77_TOPK_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_topk.wgsl");

/// Embedded WGSL kernel source: hash-table-based LZ77 match finding.
const LZ77_HASH_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_hash.wgsl");

/// Embedded WGSL kernel source: LZ77 match finding with lazy matching emulation.
const LZ77_LAZY_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_lazy.wgsl");

/// Embedded WGSL kernel source: GPU rank assignment for BWT prefix-doubling.
const BWT_RANK_KERNEL_SOURCE: &str = include_str!("../../kernels/bwt_rank.wgsl");

/// Embedded WGSL kernel source: radix sort for BWT prefix-doubling.
const BWT_RADIX_KERNEL_SOURCE: &str = include_str!("../../kernels/bwt_radix.wgsl");

/// Embedded WGSL kernel source: GPU Huffman encoding.
const HUFFMAN_ENCODE_KERNEL_SOURCE: &str = include_str!("../../kernels/huffman_encode.wgsl");

/// Embedded WGSL kernel source: GPU FSE decode.
const FSE_DECODE_KERNEL_SOURCE: &str = include_str!("../../kernels/fse_decode.wgsl");

/// Embedded WGSL kernel source: GPU FSE multi-block decode.
const FSE_DECODE_BLOCKS_KERNEL_SOURCE: &str = include_str!("../../kernels/fse_decode_blocks.wgsl");

/// Embedded WGSL kernel source: GPU FSE encode.
const FSE_ENCODE_KERNEL_SOURCE: &str = include_str!("../../kernels/fse_encode.wgsl");

/// Number of candidates per position in the top-K kernel (must match K in lz77_topk.wgsl).
const TOPK_K: usize = 4;

/// Minimum input size below which GPU overhead exceeds benefit.
pub const MIN_GPU_INPUT_SIZE: usize = 64 * 1024; // 64KB

/// Minimum BWT input size for GPU acceleration.
pub const MIN_GPU_BWT_SIZE: usize = 32 * 1024; // 32KB

/// Hash table bucket capacity (must match BUCKET_CAP in lz77_hash.wgsl).
///
/// Increased from 64 to 256 to prevent bucket overflow on large inputs.
/// The GPU hash table uses bounded append — once a bucket fills, new
/// entries are silently dropped. With BUCKET_CAP=64 on 1MB input, hot
/// buckets overflowed causing 4-5x more literals than CPU matching.
const HASH_BUCKET_CAP: usize = 256;

/// Hash table size (must match HASH_SIZE in lz77_hash.wgsl / lz77_lazy.wgsl).
///
/// Increased from 1<<15 (32K) to 1<<17 (128K) to reduce hash collisions on
/// inputs larger than 64KB. The old 32K table caused severe bucket overflow
/// with inputs >128KB, producing 4-5x more matches than CPU lazy matching.
const HASH_TABLE_SIZE: usize = 1 << 17; // 131072

/// GPU match struct matching the WGSL kernel's Lz77Match layout.
/// 3 x u32 = 12 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct GpuMatch {
    offset: u32,
    length: u32,
    next: u32, // next byte in low 8 bits
}

// SAFETY: GpuMatch is repr(C) with all-u32 fields, which are Pod/Zeroable.
unsafe impl bytemuck::Pod for GpuMatch {}
unsafe impl bytemuck::Zeroable for GpuMatch {}

/// Handle to a submitted-but-not-yet-read-back GPU LZ77 computation.
///
/// Created by `submit_find_matches_lazy()`, consumed by `complete_find_matches_lazy()`.
/// Owns the staging buffer and metadata needed to read back and deduplicate results.
struct PendingLz77 {
    staging_buf: wgpu::Buffer,
    input_len: usize,
}

/// Information about a discovered WebGPU device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Human-readable device name.
    pub name: String,
    /// Device vendor string.
    pub vendor: String,
    /// Whether this is a discrete GPU device.
    pub is_gpu: bool,
    /// Maximum workgroup size.
    pub max_work_group_size: usize,
}

/// Probe all available WebGPU devices without creating an engine.
pub fn probe_devices() -> Vec<DeviceInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());
    adapters
        .into_iter()
        .map(|adapter| {
            let info = adapter.get_info();
            let limits = adapter.limits();
            DeviceInfo {
                name: info.name.clone(),
                vendor: format!("{:?}", info.vendor),
                is_gpu: matches!(
                    info.device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                ),
                max_work_group_size: limits.max_compute_workgroup_size_x as usize,
            }
        })
        .collect()
}

/// Return the number of available WebGPU devices.
pub fn device_count() -> usize {
    probe_devices().len()
}

// ---------------------------------------------------------------------------
// Lazy pipeline group structs — one per WGSL shader module.
// Pipelines are compiled on first use via OnceLock, not at engine creation.
// ---------------------------------------------------------------------------

/// LZ77 top-K pipeline (1 pipeline from lz77_topk.wgsl).
struct Lz77TopkPipelines {
    topk: wgpu::ComputePipeline,
}

/// LZ77 hash-table pipelines (2 pipelines from lz77_hash.wgsl).
struct Lz77HashPipelines {
    build: wgpu::ComputePipeline,
    find: wgpu::ComputePipeline,
}

/// LZ77 lazy-matching pipelines (3 pipelines from lz77_lazy.wgsl).
struct Lz77LazyPipelines {
    build: wgpu::ComputePipeline,
    find: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
}

/// BWT rank pipelines (4 pipelines from bwt_rank.wgsl).
struct BwtRankPipelines {
    rank_compare: wgpu::ComputePipeline,
    prefix_sum_local: wgpu::ComputePipeline,
    prefix_sum_propagate: wgpu::ComputePipeline,
    rank_scatter: wgpu::ComputePipeline,
}

/// BWT radix sort pipelines (4 pipelines from bwt_radix.wgsl).
struct BwtRadixPipelines {
    compute_keys: wgpu::ComputePipeline,
    histogram: wgpu::ComputePipeline,
    inclusive_to_exclusive: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
}

/// Huffman encoding pipelines (5 pipelines from huffman_encode.wgsl).
struct HuffmanPipelines {
    byte_histogram: wgpu::ComputePipeline,
    compute_bit_lengths: wgpu::ComputePipeline,
    write_codes: wgpu::ComputePipeline,
    prefix_sum_block: wgpu::ComputePipeline,
    prefix_sum_apply: wgpu::ComputePipeline,
}

/// FSE decode pipeline (1 pipeline from fse_decode.wgsl).
struct FseDecodePipelines {
    decode: wgpu::ComputePipeline,
}

/// FSE multi-block decode pipeline (1 pipeline from fse_decode_blocks.wgsl).
struct FseDecodeBlocksPipelines {
    decode_blocks: wgpu::ComputePipeline,
}

/// FSE encode pipeline (1 pipeline from fse_encode.wgsl).
struct FseEncodePipelines {
    encode: wgpu::ComputePipeline,
}

/// WebGPU compute engine.
///
/// Manages the wgpu device, queue, and lazily-compiled compute pipelines.
/// Create one engine at library init time and reuse it across calls.
/// Pipelines are compiled on first use to avoid paying startup cost for
/// shader modules that the requested pipeline doesn't need.
pub struct WebGpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Lazily-compiled pipeline groups (one OnceLock per WGSL shader module).
    lz77_topk: OnceLock<Lz77TopkPipelines>,
    lz77_hash: OnceLock<Lz77HashPipelines>,
    lz77_lazy: OnceLock<Lz77LazyPipelines>,
    bwt_rank: OnceLock<BwtRankPipelines>,
    bwt_radix: OnceLock<BwtRadixPipelines>,
    huffman: OnceLock<HuffmanPipelines>,
    fse_decode: OnceLock<FseDecodePipelines>,
    fse_decode_blocks: OnceLock<FseDecodeBlocksPipelines>,
    fse_encode: OnceLock<FseEncodePipelines>,
    /// Device name for diagnostics.
    device_name: String,
    /// Maximum compute workgroup size.
    max_work_group_size: usize,
    /// Maximum workgroups per dispatch dimension (device-queried, typically 65535).
    max_workgroups_per_dim: u32,
    /// Whether the selected device is a CPU (not GPU).
    is_cpu: bool,
    /// Scan workgroup size (power of 2, capped at 256).
    scan_workgroup_size: usize,
    /// Maximum storage buffer binding size in bytes (device limit).
    max_buffer_size: u32,
    /// Parsed cost model for the LZ77 lazy kernel (used for batch scheduling).
    cost_lz77_lazy: KernelCost,
    /// Whether profiling is enabled (timestamp queries).
    profiling: bool,
    /// GPU profiler for timestamp queries (None when profiling disabled or unsupported).
    /// Wrapped in Mutex because resolve_queries/end_frame/process_finished_frame
    /// require &mut self, but WebGpuEngine methods use &self (engine is behind Arc).
    profiler: Option<std::sync::Mutex<wgpu_profiler::GpuProfiler>>,
}

impl std::fmt::Debug for WebGpuEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebGpuEngine")
            .field("device_name", &self.device_name)
            .field("max_work_group_size", &self.max_work_group_size)
            .finish_non_exhaustive()
    }
}

impl WebGpuEngine {
    /// Create a new engine, selecting the best available GPU device.
    pub fn new() -> PzResult<Self> {
        Self::create(true, false)
    }

    /// Create a new engine with explicit GPU preference.
    pub fn with_device_preference(prefer_gpu: bool) -> PzResult<Self> {
        Self::create(prefer_gpu, false)
    }

    /// Create a new engine with profiling enabled.
    ///
    /// When profiling is on, `TIMESTAMP_QUERY` is requested on the device
    /// and GPU dispatches are timed via `wgpu-profiler`. Call
    /// [`profiler_end_frame()`] after a batch of work to collect results,
    /// and [`profiler_write_trace()`] to export a Chrome trace file.
    pub fn with_profiling(profiling: bool) -> PzResult<Self> {
        Self::create(true, profiling)
    }

    fn create(prefer_gpu: bool, profiling: bool) -> PzResult<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let power_pref = if prefer_gpu {
            wgpu::PowerPreference::HighPerformance
        } else {
            wgpu::PowerPreference::None
        };

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: power_pref,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(|_| PzError::Unsupported)?;

        let info = adapter.get_info();
        let device_name = info.name.clone();
        let is_cpu = matches!(info.device_type, wgpu::DeviceType::Cpu);

        // Reject software/CPU adapters (e.g. WARP on Windows) when a real GPU
        // was requested — they're too slow for compute workloads and can hang.
        if prefer_gpu && is_cpu {
            return Err(PzError::Unsupported);
        }

        let limits = adapter.limits();
        let max_work_group_size = limits.max_compute_workgroup_size_x as usize;
        let max_workgroups_per_dim = limits.max_compute_workgroups_per_dimension;
        let max_buffer_size = limits.max_storage_buffer_binding_size;

        // Request TIMESTAMP_QUERY when profiling is desired; fall back if unsupported.
        // profiling stays true regardless -- we'll use CPU-side wall-clock timing as fallback.
        let supports_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let use_timestamps = profiling && supports_timestamps;
        let required_features = if use_timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("libpz-webgpu"),
            required_features,
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|_| PzError::Unsupported)?;

        let capped = max_work_group_size.clamp(1, 256);
        let scan_workgroup_size = 1 << (usize::BITS - 1 - capped.leading_zeros());

        // Parse kernel cost annotations for batch scheduling (cheap string parse).
        let cost_lz77_lazy = KernelCost::parse(LZ77_LAZY_KERNEL_SOURCE)
            .expect("lz77_lazy.wgsl missing @pz_cost annotation");

        // Create wgpu-profiler when GPU timestamps are available.
        let profiler = if use_timestamps {
            match wgpu_profiler::GpuProfiler::new(
                &device,
                wgpu_profiler::GpuProfilerSettings::default(),
            ) {
                Ok(p) => Some(std::sync::Mutex::new(p)),
                Err(_) => None,
            }
        } else {
            None
        };

        // All pipeline compilation is deferred to first use via OnceLock.
        Ok(WebGpuEngine {
            device,
            queue,
            lz77_topk: OnceLock::new(),
            lz77_hash: OnceLock::new(),
            lz77_lazy: OnceLock::new(),
            bwt_rank: OnceLock::new(),
            bwt_radix: OnceLock::new(),
            huffman: OnceLock::new(),
            fse_decode: OnceLock::new(),
            fse_decode_blocks: OnceLock::new(),
            fse_encode: OnceLock::new(),
            device_name,
            max_work_group_size,
            max_workgroups_per_dim,
            is_cpu,
            scan_workgroup_size,
            max_buffer_size,
            cost_lz77_lazy,
            profiling,
            profiler,
        })
    }

    /// Return the name of the selected compute device.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Block the host until all submitted GPU work completes.
    pub(crate) fn poll_wait(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    /// Return the maximum work-group size for the device.
    pub fn max_work_group_size(&self) -> usize {
        self.max_work_group_size
    }

    /// Check if the selected device is a CPU (not a GPU or accelerator).
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

    /// GPU memory budget for concurrent in-flight work.
    ///
    /// Discrete GPUs have far more VRAM than `max_buffer_size` (which is a
    /// per-buffer WebGPU limit, typically 256MB-1GB). Use 4× max_buffer_size
    /// as a proxy for total available memory, which is still conservative for
    /// cards with 8-16GB VRAM but allows enough headroom to submit many
    /// blocks in parallel.
    fn gpu_memory_budget(&self) -> usize {
        if self.is_cpu {
            // Integrated/CPU devices: be conservative
            (self.max_buffer_size as usize) / 2
        } else {
            // Discrete GPU: allow 4× max_buffer_size (~1-4GB on modern cards)
            (self.max_buffer_size as usize) * 4
        }
    }

    /// Compute the batch size for LZ77 GPU dispatches, based on the cost model
    /// and memory budget. Exposed for the pipelined batched path in `parallel.rs`.
    pub(crate) fn lz77_batch_size(&self, block_size: usize) -> usize {
        const GPU_MAX_BATCH: usize = 64;
        let mem_limit = self.max_in_flight(&self.cost_lz77_lazy, block_size);
        mem_limit.min(GPU_MAX_BATCH)
    }

    /// Maximum input size (bytes) that fits in a single GPU dispatch.
    ///
    /// Bounded by both the 2D workgroup dispatch limit (`max^2 * workgroup_size`)
    /// and the maximum storage buffer binding size (the LZ77 match output
    /// buffer is `input_len * 12` bytes).
    pub fn max_dispatch_input_size(&self) -> usize {
        let dispatch_limit = {
            let max = self.max_workgroups_per_dim as usize;
            max * max * 64
        };
        // GpuMatch is 12 bytes per position; buffer must fit in max_buffer_size.
        let buffer_limit = self.max_buffer_size as usize / std::mem::size_of::<GpuMatch>();
        dispatch_limit.min(buffer_limit)
    }

    /// Compute the X dispatch width in invocations for 2D tiling.
    /// Kernels use `gid.x + gid.y * dispatch_width` to linearize.
    fn dispatch_width(&self, workgroups_x: u32, workgroup_size: u32) -> u32 {
        let max = self.max_workgroups_per_dim;
        let wx = if workgroups_x <= max {
            workgroups_x
        } else {
            max
        };
        wx * workgroup_size
    }

    // --- Helper: create buffer with data ---

    fn create_buffer_init(
        &self,
        label: &str,
        data: &[u8],
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage,
            })
    }

    fn create_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Read a buffer back to the CPU.
    fn read_buffer(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<u8> {
        let staging = self.create_buffer(
            "staging",
            size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_buffer"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .unwrap()
            .map_err(|_| PzError::Unsupported)
            .unwrap();

        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    /// Read a single u32 from a buffer at the given element offset.
    /// Only transfers 4 bytes instead of the entire buffer.
    fn read_buffer_scalar_u32(&self, buffer: &wgpu::Buffer, element_offset: usize) -> u32 {
        let byte_offset = (element_offset * 4) as u64;
        let staging = self.create_buffer(
            "scalar_staging",
            4,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_scalar"),
            });
        encoder.copy_buffer_to_buffer(buffer, byte_offset, &staging, 0, 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let val = u32::from_ne_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        staging.unmap();
        val
    }

    /// Compute 2D tiling dimensions for a given workgroup count.
    fn tile_workgroups(&self, workgroups_x: u32) -> PzResult<(u32, u32)> {
        let max = self.max_workgroups_per_dim;
        if workgroups_x <= max {
            Ok((workgroups_x, 1u32))
        } else {
            let wy = workgroups_x.div_ceil(max);
            if wy > max {
                return Err(PzError::Unsupported);
            }
            Ok((max, wy))
        }
    }

    /// Record a compute pass into an existing command encoder (no submit).
    fn record_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups_x: u32,
        label: &str,
    ) -> PzResult<()> {
        let (wx, wy) = self.tile_workgroups(workgroups_x)?;

        // Start a profiler query if the profiler is active.
        let profiler_query = self
            .profiler
            .as_ref()
            .map(|p| p.lock().unwrap().begin_pass_query(label, encoder));

        // Get timestamp writes from the profiler query (borrows the query).
        let timestamp_writes = profiler_query
            .as_ref()
            .and_then(|q| q.compute_pass_timestamp_writes());

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(wx, wy, 1);
        }

        // End the profiler query (consumes it).
        if let Some(query) = profiler_query {
            if let Some(p) = &self.profiler {
                p.lock().unwrap().end_query(encoder, query);
            }
        }

        Ok(())
    }

    /// Resolve profiler queries into the command encoder.
    /// Call before `encoder.finish()` when manually managing encoders.
    pub(crate) fn profiler_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let Some(p) = &self.profiler {
            p.lock().unwrap().resolve_queries(encoder);
        }
    }

    /// End the current profiler frame and collect timing results.
    ///
    /// Call after all GPU work for the frame has been submitted.
    /// Returns `None` if profiling is disabled or no results are ready.
    pub fn profiler_end_frame(&self) -> Option<Vec<wgpu_profiler::GpuTimerQueryResult>> {
        let p = self.profiler.as_ref()?;
        {
            p.lock().unwrap().end_frame().ok()?;
        }
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        p.lock()
            .unwrap()
            .process_finished_frame(self.queue.get_timestamp_period())
    }

    /// Write collected profiler results to a Chrome trace file.
    ///
    /// The resulting JSON file can be viewed at `chrome://tracing`,
    /// `edge://tracing`, or <https://ui.perfetto.dev/>.
    pub fn profiler_write_trace(
        path: &std::path::Path,
        results: &[wgpu_profiler::GpuTimerQueryResult],
    ) -> std::io::Result<()> {
        wgpu_profiler::chrometrace::write_chrometrace(path, results)
    }

    /// Record and immediately submit a single compute dispatch.
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups_x: u32,
        label: &str,
    ) -> PzResult<()> {
        let t0 = if self.profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        self.record_dispatch(&mut encoder, pipeline, bind_group, workgroups_x, label)?;
        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        // Wall-clock fallback when profiler is not available
        if self.profiling && self.profiler.is_none() {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            let ms = t0.unwrap().elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] {label}: {ms:.3} ms");
        }
        Ok(())
    }

    // --- Pad input to u32-aligned for WGSL byte reading ---

    fn pad_input_bytes(input: &[u8]) -> Vec<u8> {
        let mut padded = input.to_vec();
        // Pad to u32-aligned plus 4 extra bytes to allow safe read_u32_at()
        // across word boundaries at the end of the buffer.
        let target = ((input.len() + 3) & !3) + 4;
        padded.resize(target, 0);
        padded
    }

    // -------------------------------------------------------------------
    // Lazy pipeline accessors — compile on first use via OnceLock.
    // -------------------------------------------------------------------

    /// Helper: create a shader module + compute pipeline from WGSL source.
    fn make_pipeline(&self, label: &str, source: &str, entry: &str) -> wgpu::ComputePipeline {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    fn pipeline_lz77_topk(&self) -> &wgpu::ComputePipeline {
        &self
            .lz77_topk
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = Lz77TopkPipelines {
                    topk: self.make_pipeline("lz77_topk", LZ77_TOPK_KERNEL_SOURCE, "encode_topk"),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile lz77_topk.wgsl: {ms:.3} ms");
                }
                group
            })
            .topk
    }

    fn lz77_hash_pipelines(&self) -> &Lz77HashPipelines {
        self.lz77_hash.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lz77_hash"),
                    source: wgpu::ShaderSource::Wgsl(LZ77_HASH_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = Lz77HashPipelines {
                build: make("lz77_hash_build", "build_hash_table"),
                find: make("lz77_hash_find", "find_matches"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile lz77_hash.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    fn pipeline_lz77_hash_build(&self) -> &wgpu::ComputePipeline {
        &self.lz77_hash_pipelines().build
    }

    fn pipeline_lz77_hash_find(&self) -> &wgpu::ComputePipeline {
        &self.lz77_hash_pipelines().find
    }

    fn lz77_lazy_pipelines(&self) -> &Lz77LazyPipelines {
        self.lz77_lazy.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lz77_lazy"),
                    source: wgpu::ShaderSource::Wgsl(LZ77_LAZY_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = Lz77LazyPipelines {
                build: make("lz77_lazy_build", "build_hash_table"),
                find: make("lz77_lazy_find", "find_matches"),
                resolve: make("lz77_lazy_resolve", "resolve_lazy"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile lz77_lazy.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    fn pipeline_lz77_lazy_build(&self) -> &wgpu::ComputePipeline {
        &self.lz77_lazy_pipelines().build
    }

    fn pipeline_lz77_lazy_find(&self) -> &wgpu::ComputePipeline {
        &self.lz77_lazy_pipelines().find
    }

    fn pipeline_lz77_lazy_resolve(&self) -> &wgpu::ComputePipeline {
        &self.lz77_lazy_pipelines().resolve
    }

    fn bwt_rank_pipelines(&self) -> &BwtRankPipelines {
        self.bwt_rank.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("bwt_rank"),
                    source: wgpu::ShaderSource::Wgsl(BWT_RANK_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = BwtRankPipelines {
                rank_compare: make("rank_compare", "rank_compare"),
                prefix_sum_local: make("prefix_sum_local", "prefix_sum_local"),
                prefix_sum_propagate: make("prefix_sum_propagate", "prefix_sum_propagate"),
                rank_scatter: make("rank_scatter", "rank_scatter"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile bwt_rank.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    fn pipeline_rank_compare(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().rank_compare
    }

    fn pipeline_prefix_sum_local(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().prefix_sum_local
    }

    fn pipeline_prefix_sum_propagate(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().prefix_sum_propagate
    }

    fn pipeline_rank_scatter(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().rank_scatter
    }

    fn bwt_radix_pipelines(&self) -> &BwtRadixPipelines {
        self.bwt_radix.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("bwt_radix"),
                    source: wgpu::ShaderSource::Wgsl(BWT_RADIX_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = BwtRadixPipelines {
                compute_keys: make("radix_compute_keys", "radix_compute_keys"),
                histogram: make("radix_histogram", "radix_histogram"),
                inclusive_to_exclusive: make("inclusive_to_exclusive", "inclusive_to_exclusive"),
                scatter: make("radix_scatter", "radix_scatter"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile bwt_radix.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    fn pipeline_radix_compute_keys(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().compute_keys
    }

    fn pipeline_radix_histogram(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().histogram
    }

    fn pipeline_inclusive_to_exclusive(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().inclusive_to_exclusive
    }

    fn pipeline_radix_scatter(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().scatter
    }

    fn huffman_pipelines(&self) -> &HuffmanPipelines {
        self.huffman.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("huffman_encode"),
                    source: wgpu::ShaderSource::Wgsl(HUFFMAN_ENCODE_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = HuffmanPipelines {
                byte_histogram: make("byte_histogram", "byte_histogram"),
                compute_bit_lengths: make("compute_bit_lengths", "compute_bit_lengths"),
                write_codes: make("write_codes", "write_codes"),
                prefix_sum_block: make("prefix_sum_block", "prefix_sum_block"),
                prefix_sum_apply: make("prefix_sum_apply", "prefix_sum_apply"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile huffman_encode.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    fn pipeline_byte_histogram(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().byte_histogram
    }

    fn pipeline_compute_bit_lengths(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().compute_bit_lengths
    }

    fn pipeline_write_codes(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().write_codes
    }

    fn pipeline_prefix_sum_block(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().prefix_sum_block
    }

    fn pipeline_prefix_sum_apply(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().prefix_sum_apply
    }

    fn pipeline_fse_decode(&self) -> &wgpu::ComputePipeline {
        &self
            .fse_decode
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = FseDecodePipelines {
                    decode: self.make_pipeline(
                        "fse_decode",
                        FSE_DECODE_KERNEL_SOURCE,
                        "fse_decode",
                    ),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile fse_decode.wgsl: {ms:.3} ms");
                }
                group
            })
            .decode
    }

    fn pipeline_fse_decode_blocks(&self) -> &wgpu::ComputePipeline {
        &self
            .fse_decode_blocks
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = FseDecodeBlocksPipelines {
                    decode_blocks: self.make_pipeline(
                        "fse_decode_blocks",
                        FSE_DECODE_BLOCKS_KERNEL_SOURCE,
                        "fse_decode_blocks",
                    ),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile fse_decode_blocks.wgsl: {ms:.3} ms");
                }
                group
            })
            .decode_blocks
    }

    fn pipeline_fse_encode(&self) -> &wgpu::ComputePipeline {
        &self
            .fse_encode
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = FseEncodePipelines {
                    encode: self.make_pipeline(
                        "fse_encode",
                        FSE_ENCODE_KERNEL_SOURCE,
                        "fse_encode",
                    ),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile fse_encode.wgsl: {ms:.3} ms");
                }
                group
            })
            .encode
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
pub struct DeviceBuf {
    pub(crate) buf: wgpu::Buffer,
    pub(crate) len: usize,
}

impl DeviceBuf {
    /// Upload host data to the GPU, returning a device-resident buffer.
    ///
    /// The data is padded to u32-aligned + 4 bytes (matching WGSL's `array<u32>`
    /// byte reading convention). The `len` field stores the logical (unpadded)
    /// length.
    pub fn from_host(engine: &WebGpuEngine, data: &[u8]) -> PzResult<Self> {
        if data.is_empty() {
            let buf = engine.create_buffer(
                "device_buf_empty",
                4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            return Ok(DeviceBuf { buf, len: 0 });
        }

        let padded = WebGpuEngine::pad_input_bytes(data);
        let buf = engine.create_buffer_init(
            "device_buf",
            &padded,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        Ok(DeviceBuf {
            buf,
            len: data.len(),
        })
    }

    /// Allocate a device buffer of the given size.
    ///
    /// **Note:** The buffer contents are *not* guaranteed to be zero-initialized.
    /// Callers that need zeroed memory should write zeros explicitly.
    pub fn alloc(engine: &WebGpuEngine, len: usize) -> PzResult<Self> {
        let actual_len = len.max(4); // avoid zero-size allocation
        let buf = engine.create_buffer(
            "device_buf_alloc",
            actual_len as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        Ok(DeviceBuf { buf, len })
    }

    /// Download the buffer contents from the GPU to host memory.
    pub fn read_to_host(&self, engine: &WebGpuEngine) -> PzResult<Vec<u8>> {
        if self.len == 0 {
            return Ok(Vec::new());
        }

        // The buffer may be padded, so read the full buffer and truncate.
        let raw = engine.read_buffer(&self.buf, self.buf.size());
        Ok(raw[..self.len].to_vec())
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
/// Produced by [`WebGpuEngine::find_matches_to_device()`] and consumed by
/// [`WebGpuEngine::download_and_dedupe()`]. The match data stays on the GPU
/// until explicitly downloaded, enabling zero-copy stage chaining.
pub struct GpuMatchBuf {
    pub(crate) buf: wgpu::Buffer,
    pub(crate) input_len: usize,
}

impl GpuMatchBuf {
    /// The number of input positions this match buffer covers.
    pub fn input_len(&self) -> usize {
        self.input_len
    }
}

/// Deduplicate raw GPU match output into a non-overlapping match sequence.
fn dedupe_gpu_matches(gpu_matches: &[GpuMatch], input: &[u8]) -> Vec<Match> {
    if gpu_matches.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let len = gpu_matches.len();
    let mut index = 0;

    while index < len {
        let gm = &gpu_matches[index];

        let remaining = len - index;
        let mut match_length = gm.length as usize;
        if match_length >= remaining {
            match_length = if remaining > 0 { remaining - 1 } else { 0 };
        }
        // Cap to u16::MAX since Match.length is u16. The next iteration will
        // pick up a fresh per-position match from the GPU output.
        if match_length > u16::MAX as usize {
            match_length = u16::MAX as usize;
        }

        let next = if index + match_length < input.len() {
            input[index + match_length]
        } else {
            gm.next as u8
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
