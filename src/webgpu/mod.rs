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
use crate::lz77::Match;
use crate::{PzError, PzResult};

use wgpu::util::DeviceExt;

mod bwt;
mod fse;
mod huffman;
mod lz77;

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

/// Number of candidates per position in the top-K kernel (must match K in lz77_topk.wgsl).
const TOPK_K: usize = 4;

/// Minimum input size below which GPU overhead exceeds benefit.
pub const MIN_GPU_INPUT_SIZE: usize = 64 * 1024; // 64KB

/// Minimum BWT input size for GPU acceleration.
pub const MIN_GPU_BWT_SIZE: usize = 32 * 1024; // 32KB

/// Hash table bucket capacity (must match BUCKET_CAP in lz77_hash.wgsl).
const HASH_BUCKET_CAP: usize = 64;

/// Hash table size (must match HASH_SIZE in lz77_hash.wgsl).
const HASH_TABLE_SIZE: usize = 1 << 15; // 32768

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

/// Maximum number of blocks to submit in a single GPU batch.
/// Limits GPU memory pressure: 8 x 256KB x ~36 bytes/pos ~ 72MB.
const MAX_GPU_BATCH_SIZE: usize = 8;

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

/// WebGPU compute engine.
///
/// Manages the wgpu device, queue, and compiled shader modules.
/// Create one engine at library init time and reuse it across calls.
pub struct WebGpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Cached compute pipelines (created once at init, like OpenCL kernel objects)
    pipeline_lz77_topk: wgpu::ComputePipeline,
    pipeline_lz77_hash_build: wgpu::ComputePipeline,
    pipeline_lz77_hash_find: wgpu::ComputePipeline,
    pipeline_lz77_lazy_build: wgpu::ComputePipeline,
    pipeline_lz77_lazy_find: wgpu::ComputePipeline,
    pipeline_lz77_lazy_resolve: wgpu::ComputePipeline,
    pipeline_rank_compare: wgpu::ComputePipeline,
    pipeline_prefix_sum_local: wgpu::ComputePipeline,
    pipeline_prefix_sum_propagate: wgpu::ComputePipeline,
    pipeline_rank_scatter: wgpu::ComputePipeline,
    pipeline_radix_compute_keys: wgpu::ComputePipeline,
    pipeline_radix_histogram: wgpu::ComputePipeline,
    pipeline_inclusive_to_exclusive: wgpu::ComputePipeline,
    pipeline_radix_scatter: wgpu::ComputePipeline,
    pipeline_byte_histogram: wgpu::ComputePipeline,
    pipeline_compute_bit_lengths: wgpu::ComputePipeline,
    pipeline_write_codes: wgpu::ComputePipeline,
    pipeline_prefix_sum_block: wgpu::ComputePipeline,
    pipeline_prefix_sum_apply: wgpu::ComputePipeline,
    pipeline_fse_decode: wgpu::ComputePipeline,
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
    /// Whether profiling is enabled (timestamp queries).
    profiling: bool,
    /// Query set for timestamp profiling (begin + end).
    query_set: Option<wgpu::QuerySet>,
    /// Buffer to resolve timestamp queries into.
    resolve_buf: Option<wgpu::Buffer>,
    /// Staging buffer for reading back resolved timestamps.
    staging_buf: Option<wgpu::Buffer>,
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
    /// and major kernel dispatches print timing via `eprintln!`.
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
        .ok_or(PzError::Unsupported)?;

        let info = adapter.get_info();
        let device_name = info.name.clone();
        let is_cpu = matches!(info.device_type, wgpu::DeviceType::Cpu);
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("libpz-webgpu"),
                required_features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|_| PzError::Unsupported)?;

        let capped = max_work_group_size.clamp(1, 256);
        let scan_workgroup_size = 1 << (usize::BITS - 1 - capped.leading_zeros());

        let lz77_topk_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lz77_topk"),
            source: wgpu::ShaderSource::Wgsl(LZ77_TOPK_KERNEL_SOURCE.into()),
        });

        let lz77_hash_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lz77_hash"),
            source: wgpu::ShaderSource::Wgsl(LZ77_HASH_KERNEL_SOURCE.into()),
        });

        let lz77_lazy_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lz77_lazy"),
            source: wgpu::ShaderSource::Wgsl(LZ77_LAZY_KERNEL_SOURCE.into()),
        });

        let bwt_rank_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bwt_rank"),
            source: wgpu::ShaderSource::Wgsl(BWT_RANK_KERNEL_SOURCE.into()),
        });

        let bwt_radix_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bwt_radix"),
            source: wgpu::ShaderSource::Wgsl(BWT_RADIX_KERNEL_SOURCE.into()),
        });

        let huffman_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("huffman_encode"),
            source: wgpu::ShaderSource::Wgsl(HUFFMAN_ENCODE_KERNEL_SOURCE.into()),
        });

        let fse_decode_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fse_decode"),
            source: wgpu::ShaderSource::Wgsl(FSE_DECODE_KERNEL_SOURCE.into()),
        });

        // Helper to create a compute pipeline from a module + entry point.
        let make_pipeline =
            |label: &str, module: &wgpu::ShaderModule, entry: &str| -> wgpu::ComputePipeline {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: None,
                    module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };

        // Cache all 18 compute pipelines at init time (mirroring OpenCL kernel caching).
        let pipeline_lz77_topk = make_pipeline("lz77_topk", &lz77_topk_module, "encode_topk");
        let pipeline_lz77_hash_build =
            make_pipeline("lz77_hash_build", &lz77_hash_module, "build_hash_table");
        let pipeline_lz77_hash_find =
            make_pipeline("lz77_hash_find", &lz77_hash_module, "find_matches");
        let pipeline_lz77_lazy_build =
            make_pipeline("lz77_lazy_build", &lz77_lazy_module, "build_hash_table");
        let pipeline_lz77_lazy_find =
            make_pipeline("lz77_lazy_find", &lz77_lazy_module, "find_matches");
        let pipeline_lz77_lazy_resolve =
            make_pipeline("lz77_lazy_resolve", &lz77_lazy_module, "resolve_lazy");
        let pipeline_rank_compare = make_pipeline("rank_compare", &bwt_rank_module, "rank_compare");
        let pipeline_prefix_sum_local =
            make_pipeline("prefix_sum_local", &bwt_rank_module, "prefix_sum_local");
        let pipeline_prefix_sum_propagate = make_pipeline(
            "prefix_sum_propagate",
            &bwt_rank_module,
            "prefix_sum_propagate",
        );
        let pipeline_rank_scatter = make_pipeline("rank_scatter", &bwt_rank_module, "rank_scatter");
        let pipeline_radix_compute_keys = make_pipeline(
            "radix_compute_keys",
            &bwt_radix_module,
            "radix_compute_keys",
        );
        let pipeline_radix_histogram =
            make_pipeline("radix_histogram", &bwt_radix_module, "radix_histogram");
        let pipeline_inclusive_to_exclusive = make_pipeline(
            "inclusive_to_exclusive",
            &bwt_radix_module,
            "inclusive_to_exclusive",
        );
        let pipeline_radix_scatter =
            make_pipeline("radix_scatter", &bwt_radix_module, "radix_scatter");
        let pipeline_byte_histogram =
            make_pipeline("byte_histogram", &huffman_module, "byte_histogram");
        let pipeline_compute_bit_lengths = make_pipeline(
            "compute_bit_lengths",
            &huffman_module,
            "compute_bit_lengths",
        );
        let pipeline_write_codes = make_pipeline("write_codes", &huffman_module, "write_codes");
        let pipeline_prefix_sum_block =
            make_pipeline("prefix_sum_block", &huffman_module, "prefix_sum_block");
        let pipeline_prefix_sum_apply =
            make_pipeline("prefix_sum_apply", &huffman_module, "prefix_sum_apply");
        let pipeline_fse_decode = make_pipeline("fse_decode", &fse_decode_module, "fse_decode");

        // Create profiling resources when GPU timestamps are available.
        let (query_set, resolve_buf, staging_buf) = if use_timestamps {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamp_query_set"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_resolve"),
                size: 2 * std::mem::size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("timestamp_staging"),
                size: 2 * std::mem::size_of::<u64>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (Some(qs), Some(resolve), Some(staging))
        } else {
            (None, None, None)
        };

        Ok(WebGpuEngine {
            device,
            queue,
            pipeline_lz77_topk,
            pipeline_lz77_hash_build,
            pipeline_lz77_hash_find,
            pipeline_lz77_lazy_build,
            pipeline_lz77_lazy_find,
            pipeline_lz77_lazy_resolve,
            pipeline_rank_compare,
            pipeline_prefix_sum_local,
            pipeline_prefix_sum_propagate,
            pipeline_rank_scatter,
            pipeline_radix_compute_keys,
            pipeline_radix_histogram,
            pipeline_inclusive_to_exclusive,
            pipeline_radix_scatter,
            pipeline_byte_histogram,
            pipeline_compute_bit_lengths,
            pipeline_write_codes,
            pipeline_prefix_sum_block,
            pipeline_prefix_sum_apply,
            pipeline_fse_decode,
            device_name,
            max_work_group_size,
            max_workgroups_per_dim,
            is_cpu,
            scan_workgroup_size,
            max_buffer_size,
            profiling,
            query_set,
            resolve_buf,
            staging_buf,
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
    pub fn is_cpu_device(&self) -> bool {
        self.is_cpu
    }

    /// Whether profiling is enabled on this engine.
    pub fn profiling(&self) -> bool {
        self.profiling
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
        self.device.poll(wgpu::Maintain::Wait);
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
        self.device.poll(wgpu::Maintain::Wait);
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
        {
            let timestamp_writes =
                self.query_set
                    .as_ref()
                    .map(|qs| wgpu::ComputePassTimestampWrites {
                        query_set: qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(wx, wy, 1);
        }
        Ok(())
    }

    /// Resolve timestamp queries, read back, and print elapsed time.
    fn read_and_print_timestamps(&self, label: &str) {
        let (Some(qs), Some(resolve), Some(staging)) =
            (&self.query_set, &self.resolve_buf, &self.staging_buf)
        else {
            return;
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("timestamp_resolve"),
            });
        encoder.resolve_query_set(qs, 0..2, resolve, 0);
        encoder.copy_buffer_to_buffer(resolve, 0, staging, 0, 16);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        if rx.recv().unwrap().is_ok() {
            let data = slice.get_mapped_range();
            let timestamps: &[u64] = bytemuck::cast_slice(&data);
            let start_ns = timestamps[0];
            let end_ns = timestamps[1];
            drop(data);
            staging.unmap();
            let elapsed_ns = end_ns.saturating_sub(start_ns);
            let ms = elapsed_ns as f64 / 1_000_000.0;
            eprintln!("[pz-gpu] {label}: {ms:.3} ms");
        } else {
            staging.unmap();
        }
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
        self.queue.submit(Some(encoder.finish()));
        if self.profiling {
            if self.query_set.is_some() {
                // GPU timestamps available -- use precise device timings.
                self.read_and_print_timestamps(label);
            } else {
                // Fall back to CPU wall-clock (includes submit + sync overhead).
                self.device.poll(wgpu::Maintain::Wait);
                let ms = t0.unwrap().elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] {label}: {ms:.3} ms");
            }
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

    /// Allocate a zero-initialized device buffer of the given size.
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
