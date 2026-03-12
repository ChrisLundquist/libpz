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

mod buffers;
mod bwt;
mod fse;
mod huffman;
mod kernels;
pub(crate) mod lz77;
pub(crate) mod lzseq;
mod pipelines;
pub(crate) mod rans;
mod sortlz;

use buffers::*;
pub use buffers::{DeviceBuf, GpuMatchBuf};
pub use huffman::HuffmanDecodeStream;
use kernels::*;
use pipelines::*;

#[cfg(test)]
mod tests;

/// Minimum input size below which GPU overhead exceeds benefit.
pub const MIN_GPU_INPUT_SIZE: usize = 64 * 1024; // 64KB

/// Minimum BWT input size for GPU acceleration.
pub const MIN_GPU_BWT_SIZE: usize = 32 * 1024; // 32KB

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
        backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
        flags: wgpu::InstanceFlags::from_env_or_default(),
        ..Default::default()
    });

    let backends = wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12;
    let adapters = instance.enumerate_adapters(backends);
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
    #[allow(dead_code)]
    lz77_lazy: OnceLock<Lz77LazyPipelines>,
    lz77_coop: OnceLock<Lz77CoopPipelines>,
    bwt_rank: OnceLock<BwtRankPipelines>,
    bwt_radix: OnceLock<BwtRadixPipelines>,
    huffman: OnceLock<HuffmanPipelines>,
    fse_decode: OnceLock<FseDecodePipelines>,
    fse_encode: OnceLock<FseEncodePipelines>,
    lz77_decode: OnceLock<Lz77DecodePipelines>,
    rans_decode: OnceLock<RansDecodePipelines>,
    rans_encode: OnceLock<RansEncodePipelines>,
    lzseq_demux: OnceLock<LzSeqPipelines>,
    huffman_decode: OnceLock<HuffmanDecodePipelines>,
    sortlz: OnceLock<SortLzPipelines>,
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
    /// Maximum storage buffers per shader stage (device limit).
    max_storage_buffers_per_shader_stage: u32,
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
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL | wgpu::Backends::DX12,
            flags: wgpu::InstanceFlags::from_env_or_default(),
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
        let max_storage_buffers_per_shader_stage = limits.max_storage_buffers_per_shader_stage;

        let mut required_limits = wgpu::Limits::downlevel_defaults();
        // rANS kernels use 5 storage bindings. Keep conservative defaults while
        // raising this single limit to what the adapter can support.
        required_limits.max_storage_buffers_per_shader_stage = required_limits
            .max_storage_buffers_per_shader_stage
            .max(5)
            .min(max_storage_buffers_per_shader_stage);

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
            required_limits,
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
            lz77_lazy: OnceLock::new(),
            lz77_coop: OnceLock::new(),
            bwt_rank: OnceLock::new(),
            bwt_radix: OnceLock::new(),
            huffman: OnceLock::new(),
            fse_decode: OnceLock::new(),
            fse_encode: OnceLock::new(),
            lz77_decode: OnceLock::new(),
            rans_decode: OnceLock::new(),
            rans_encode: OnceLock::new(),
            lzseq_demux: OnceLock::new(),
            huffman_decode: OnceLock::new(),
            sortlz: OnceLock::new(),
            device_name,
            max_work_group_size,
            max_workgroups_per_dim,
            is_cpu,
            scan_workgroup_size,
            max_buffer_size,
            max_storage_buffers_per_shader_stage,
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

    /// Return the max compute workgroups per dispatch dimension.
    pub fn max_workgroups_per_dimension(&self) -> u32 {
        self.max_workgroups_per_dim
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
        self.poll_wait();
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
        self.poll_wait();
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

    /// Emit a no-op profiler query to burn the first query slot.
    ///
    /// Workaround for AMD Vulkan drivers (RDNA 4 confirmed) that return
    /// zero timestamps for query pair index 0 in a query set. Call once
    /// at the start of a profiled frame before any real dispatches.
    pub fn profiler_warmup(&self) {
        if let Some(p) = &self.profiler {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("profiler_warmup"),
                });
            let mut profiler = p.lock().unwrap();
            let query = profiler.begin_pass_query("_warmup", &mut encoder);
            // Empty compute pass so the timestamp writes are valid.
            {
                let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("_warmup"),
                    timestamp_writes: query.compute_pass_timestamp_writes(),
                });
            }
            profiler.end_query(&mut encoder, query);
            profiler.resolve_queries(&mut encoder);
            drop(profiler);
            self.queue.submit(Some(encoder.finish()));
        }
    }

    /// End the current profiler frame and collect timing results.
    ///
    /// Call after all GPU work for the frame has been submitted.
    /// Returns `None` if profiling is disabled or no results are ready.
    /// Filters out internal `_warmup` queries automatically.
    pub fn profiler_end_frame(&self) -> Option<Vec<wgpu_profiler::GpuTimerQueryResult>> {
        let p = self.profiler.as_ref()?;
        p.lock().unwrap().end_frame().ok()?;
        self.poll_wait();
        let results = p
            .lock()
            .unwrap()
            .process_finished_frame(self.queue.get_timestamp_period());
        // Filter out internal warmup queries
        results.map(|v| {
            v.into_iter()
                .filter(|r| !r.label.starts_with('_'))
                .collect()
        })
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
            self.poll_wait();
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
}
