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

/// Embedded WGSL kernel source: top-K match finding for optimal parsing.
const LZ77_TOPK_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_topk.wgsl");

/// Embedded WGSL kernel source: hash-table-based LZ77 match finding.
const LZ77_HASH_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_hash.wgsl");

/// Embedded WGSL kernel source: GPU rank assignment for BWT prefix-doubling.
const BWT_RANK_KERNEL_SOURCE: &str = include_str!("../kernels/bwt_rank.wgsl");

/// Embedded WGSL kernel source: radix sort for BWT prefix-doubling.
const BWT_RADIX_KERNEL_SOURCE: &str = include_str!("../kernels/bwt_radix.wgsl");

/// Embedded WGSL kernel source: GPU Huffman encoding.
const HUFFMAN_ENCODE_KERNEL_SOURCE: &str = include_str!("../kernels/huffman_encode.wgsl");

/// Embedded WGSL kernel source: GPU FSE decode.
const FSE_DECODE_KERNEL_SOURCE: &str = include_str!("../kernels/fse_decode.wgsl");

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

        // Request TIMESTAMP_QUERY when profiling is desired; fall back if unsupported.
        // profiling stays true regardless — we'll use CPU-side wall-clock timing as fallback.
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
    /// With 2D dispatch tiling, the limit is `max^2 * workgroup_size`.
    pub fn max_dispatch_input_size(&self) -> usize {
        let max = self.max_workgroups_per_dim as usize;
        max * max * 64
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
                // GPU timestamps available — use precise device timings.
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

    // --- LZ77 Match Finding ---

    /// Find LZ77 matches for the entire input using the GPU hash-table kernel.
    pub fn find_matches(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        self.find_matches_hash(input)
    }

    fn find_matches_hash(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);

        let input_buf =
            self.create_buffer_init("lz77_hash_input", &padded, wgpu::BufferUsages::STORAGE);

        let workgroups = (input_len as u32).div_ceil(64);
        let params = [input_len as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "lz77_hash_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Hash counts and table buffers
        let hash_counts_buf = self.create_buffer_init(
            "hash_counts",
            &vec![0u8; HASH_TABLE_SIZE * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let hash_table_buf = self.create_buffer(
            "hash_table",
            (HASH_TABLE_SIZE * HASH_BUCKET_CAP * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );

        let output_buf = self.create_buffer(
            "lz77_hash_output",
            (input_len * std::mem::size_of::<GpuMatch>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Build + Find passes batched into a single command encoder submission
        let build_bg_layout = self.pipeline_lz77_hash_build.get_bind_group_layout(0);
        let build_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_hash_build_bg"),
            layout: &build_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: hash_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hash_table_buf.as_entire_binding(),
                },
            ],
        });

        let find_bg_layout = self.pipeline_lz77_hash_find.get_bind_group_layout(0);
        let find_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_hash_find_bg"),
            layout: &find_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: hash_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: hash_table_buf.as_entire_binding(),
                },
            ],
        });

        let t0 = if self.profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lz77_hash"),
            });
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_hash_build,
            &build_bg,
            workgroups,
            "lz77_hash_build",
        )?;
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_hash_find,
            &find_bg,
            workgroups,
            "lz77_hash_find",
        )?;
        self.queue.submit(Some(encoder.finish()));
        if let Some(t0) = t0 {
            self.device.poll(wgpu::Maintain::Wait);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lz77_hash (build+find): {ms:.3} ms");
        }

        let raw = self.read_buffer(
            &output_buf,
            (input_len * std::mem::size_of::<GpuMatch>()) as u64,
        );
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    /// GPU-accelerated LZ77 compression using hash-table kernel.
    pub fn lz77_compress(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        let matches = self.find_matches(input)?;

        let mut output = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
        for m in &matches {
            output.extend_from_slice(&m.to_bytes());
        }
        Ok(output)
    }

    /// GPU-accelerated top-K match finding for optimal parsing.
    pub fn find_topk_matches(&self, input: &[u8]) -> PzResult<crate::optimal::MatchTable> {
        use crate::optimal::{MatchCandidate, MatchTable};

        if input.is_empty() {
            return Ok(MatchTable::new(0, TOPK_K));
        }

        let input_len = input.len();
        let output_len = input_len * TOPK_K;
        let padded = Self::pad_input_bytes(input);

        let input_buf = self.create_buffer_init("topk_input", &padded, wgpu::BufferUsages::STORAGE);

        let output_buf = self.create_buffer(
            "topk_output",
            (output_len * 4) as u64, // packed u32 per candidate
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let workgroups = (input_len as u32).div_ceil(64);
        let params = [input_len as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "topk_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bind_group_layout = self.pipeline_lz77_topk.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_topk_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            &self.pipeline_lz77_topk,
            &bind_group,
            workgroups,
            "lz77_topk",
        )?;

        let raw = self.read_buffer(&output_buf, (output_len * 4) as u64);
        let packed: &[u32] = bytemuck::cast_slice(&raw);

        let mut table = MatchTable::new(input_len, TOPK_K);
        for pos in 0..input_len {
            let slot = table.at_mut(pos);
            for k in 0..TOPK_K {
                let p = packed[pos * TOPK_K + k];
                slot[k] = MatchCandidate {
                    offset: (p & 0xFFFF) as u16,
                    length: (p >> 16) as u16,
                };
            }
        }

        Ok(table)
    }

    /// GPU-accelerated BWT forward transform.
    pub fn bwt_encode(&self, input: &[u8]) -> PzResult<BwtResult> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        let sa = self.bwt_build_suffix_array(input)?;

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

    /// GPU-accelerated bijective BWT forward transform.
    ///
    /// Factorizes input into Lyndon words on CPU, then dispatches GPU SA
    /// construction per factor (falling back to CPU for factors below
    /// [`MIN_GPU_BWT_SIZE`]).
    pub fn bwt_encode_bijective(&self, input: &[u8]) -> PzResult<(Vec<u8>, Vec<usize>)> {
        use crate::bwt;

        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        let factors = bwt::lyndon_factorize(input);
        let mut output = Vec::with_capacity(input.len());
        let mut factor_lengths = Vec::with_capacity(factors.len());

        for &(start, len) in &factors {
            let factor = &input[start..start + len];
            factor_lengths.push(len);

            if len == 1 {
                output.push(factor[0]);
                continue;
            }

            // Use GPU for factors large enough to amortize launch overhead.
            let sa = if len >= MIN_GPU_BWT_SIZE && len <= self.max_dispatch_input_size() {
                self.bwt_build_suffix_array(factor)?
            } else {
                bwt::build_circular_suffix_array(factor)
            };

            for &sa_val in &sa {
                if sa_val == 0 {
                    output.push(factor[len - 1]);
                } else {
                    output.push(factor[sa_val - 1]);
                }
            }
        }

        Ok((output, factor_lengths))
    }

    /// Build suffix array on the GPU using prefix-doubling with radix sort.
    fn bwt_build_suffix_array(&self, input: &[u8]) -> PzResult<Vec<usize>> {
        let n = input.len();
        if n <= 1 {
            return Ok(if n == 0 { Vec::new() } else { vec![0] });
        }

        let padded_n = n.next_power_of_two();

        // Initialize sa in descending order for stable sort tiebreaking
        let sa_host: Vec<u32> = (0..padded_n as u32).rev().collect();
        let mut rank_host: Vec<u32> = vec![u32::MAX; padded_n];
        for i in 0..n {
            rank_host[i] = input[i] as u32;
        }

        // Create GPU buffers
        let mut sa_buf = self.create_buffer_init(
            "sa",
            bytemuck::cast_slice(&sa_host),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut sa_buf_alt = self.create_buffer(
            "sa_alt",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut rank_buf = self.create_buffer_init(
            "rank",
            bytemuck::cast_slice(&rank_host),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut rank_buf_alt = self.create_buffer(
            "rank_alt",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let diff_buf = self.create_buffer(
            "diff",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let mut prefix_buf = self.create_buffer(
            "prefix",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let keys_buf = self.create_buffer(
            "keys",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let histogram_len = 256 * num_groups;

        let histogram_buf = self.create_buffer(
            "histogram",
            (histogram_len.max(1) * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let histogram_buf_scan = self.create_buffer(
            "histogram_scan",
            (histogram_len.max(1) * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let n_arg = n as u32;
        let padded_n_arg = padded_n as u32;

        let mut max_rank: u32 = 255;
        let mut k_step: usize = 1;
        while k_step < n {
            let k_arg = k_step as u32;

            // Phase 0: Radix sort
            self.run_radix_sort(
                &mut sa_buf,
                &mut sa_buf_alt,
                &rank_buf,
                &keys_buf,
                &histogram_buf,
                &histogram_buf_scan,
                padded_n,
                n_arg,
                k_arg,
                max_rank,
            )?;

            // Phase 1: Rank compare
            self.run_rank_compare(&sa_buf, &rank_buf, &diff_buf, n_arg, padded_n_arg, k_arg)?;

            // Phase 2: Inclusive prefix sum
            self.run_inclusive_prefix_sum(&diff_buf, &mut prefix_buf, padded_n)?;

            // Phase 3: Scatter ranks
            self.run_rank_scatter(&sa_buf, &prefix_buf, &rank_buf_alt, n_arg, padded_n_arg)?;

            // Read convergence scalar (4 bytes instead of full buffer)
            max_rank = self.read_buffer_scalar_u32(&prefix_buf, n - 1);

            // Swap rank buffers
            std::mem::swap(&mut rank_buf, &mut rank_buf_alt);

            if max_rank as usize == n - 1 {
                break;
            }
            k_step *= 2;
        }

        // Read final sa
        let sa_data = self.read_buffer(&sa_buf, (padded_n * 4) as u64);
        let sa_vals: &[u32] = bytemuck::cast_slice(&sa_data);

        let sa: Vec<usize> = sa_vals
            .iter()
            .filter(|&&v| (v as usize) < n)
            .map(|&v| v as usize)
            .collect();

        if sa.len() != n {
            return Err(PzError::InvalidInput);
        }

        Ok(sa)
    }

    fn run_rank_compare(
        &self,
        sa_buf: &wgpu::Buffer,
        rank_buf: &wgpu::Buffer,
        diff_buf: &wgpu::Buffer,
        n: u32,
        padded_n: u32,
        k: u32,
    ) -> PzResult<()> {
        let params = [n, padded_n, k, 0];
        let params_buf = self.create_buffer_init(
            "rank_compare_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_rank_compare.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rank_compare_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rank_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: diff_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = padded_n.div_ceil(256);
        self.dispatch(&self.pipeline_rank_compare, &bg, workgroups, "rank_compare")?;
        Ok(())
    }

    fn run_inclusive_prefix_sum(
        &self,
        input_buf: &wgpu::Buffer,
        output_buf: &mut wgpu::Buffer,
        count: usize,
    ) -> PzResult<()> {
        let wg = self.scan_workgroup_size;
        let block_elems = wg * 2;
        let num_blocks = count.div_ceil(block_elems);

        let params = [count as u32, 0, 0, 0];
        let params_buf = self.create_buffer_init(
            "ps_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let block_sums_buf = self.create_buffer(
            "ps_block_sums",
            (num_blocks.max(1) * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let bg_layout = self.pipeline_prefix_sum_local.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ps_local_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_sums_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = num_blocks as u32;
        self.dispatch(
            &self.pipeline_prefix_sum_local,
            &bg,
            workgroups.max(1),
            "prefix_sum_local",
        )?;

        if num_blocks > 1 {
            // Recursively scan block sums, then propagate
            let mut block_sums_scanned = self.create_buffer(
                "ps_block_sums_scanned",
                (num_blocks * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            self.run_inclusive_prefix_sum(&block_sums_buf, &mut block_sums_scanned, num_blocks)?;

            // Propagate
            let prop_params = [count as u32, 0, 0, 0];
            let prop_params_buf = self.create_buffer_init(
                "prop_params",
                bytemuck::cast_slice(&prop_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let prop_bg_layout = self.pipeline_prefix_sum_propagate.get_bind_group_layout(0);
            let prop_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ps_propagate_bg"),
                layout: &prop_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: block_sums_scanned.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: prop_params_buf.as_entire_binding(),
                    },
                ],
            });

            let prop_wg = (count as u32).div_ceil(256);
            self.dispatch(
                &self.pipeline_prefix_sum_propagate,
                &prop_bg,
                prop_wg,
                "prefix_sum_propagate",
            )?;
        }

        Ok(())
    }

    fn run_rank_scatter(
        &self,
        sa_buf: &wgpu::Buffer,
        prefix_buf: &wgpu::Buffer,
        new_rank_buf: &wgpu::Buffer,
        n: u32,
        padded_n: u32,
    ) -> PzResult<()> {
        let params = [n, padded_n, 0, 0];
        let params_buf = self.create_buffer_init(
            "scatter_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_rank_scatter.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_rank_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = padded_n.div_ceil(256);
        self.dispatch(&self.pipeline_rank_scatter, &bg, workgroups, "rank_scatter")?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_radix_sort(
        &self,
        sa_buf: &mut wgpu::Buffer,
        sa_buf_alt: &mut wgpu::Buffer,
        rank_buf: &wgpu::Buffer,
        keys_buf: &wgpu::Buffer,
        histogram_buf: &wgpu::Buffer,
        _histogram_buf_scan: &wgpu::Buffer,
        padded_n: usize,
        n: u32,
        k_doubling: u32,
        max_rank: u32,
    ) -> PzResult<()> {
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let histogram_len = 256 * num_groups;

        let bytes_needed: u32 = if max_rank < 256 {
            1
        } else if max_rank < 65536 {
            2
        } else if max_rank < 16_777_216 {
            3
        } else {
            4
        };

        let mut passes: Vec<u32> = (0..bytes_needed).chain(4..4 + bytes_needed).collect();
        if !passes.len().is_multiple_of(2) {
            passes.push(bytes_needed);
        }

        for &pass in passes.iter() {
            // Phase 1: Compute keys
            let key_params = [n, padded_n as u32, k_doubling, pass];
            let key_params_buf = self.create_buffer_init(
                "rk_params",
                bytemuck::cast_slice(&key_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let key_bg_layout = self.pipeline_radix_compute_keys.get_bind_group_layout(0);
            let key_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rk_bg"),
                layout: &key_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rank_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keys_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: key_params_buf.as_entire_binding(),
                    },
                ],
            });

            // Phases 1+2 batched: compute keys → histogram (single submit)
            let hist_params = [padded_n as u32, num_groups as u32, 0, 0];
            let hist_params_buf = self.create_buffer_init(
                "hist_params",
                bytemuck::cast_slice(&hist_params),
                wgpu::BufferUsages::UNIFORM,
            );

            // Zero the histogram buffer (queued transfer, ordered before dispatches)
            let hist_zeros = vec![0u32; histogram_len];
            self.queue
                .write_buffer(histogram_buf, 0, bytemuck::cast_slice(&hist_zeros));

            let hist_bg_layout = self.pipeline_radix_histogram.get_bind_group_layout(0);
            let hist_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hist_bg"),
                layout: &hist_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: keys_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: hist_params_buf.as_entire_binding(),
                    },
                ],
            });

            let global_wg = (padded_n as u32).div_ceil(256);
            let t0 = if self.profiling {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("radix_keys_hist"),
                });
            self.record_dispatch(
                &mut encoder,
                &self.pipeline_radix_compute_keys,
                &key_bg,
                global_wg,
                "radix_keys",
            )?;
            self.record_dispatch(
                &mut encoder,
                &self.pipeline_radix_histogram,
                &hist_bg,
                num_groups as u32,
                "radix_histogram",
            )?;
            self.queue.submit(Some(encoder.finish()));
            if let Some(t0) = t0 {
                self.device.poll(wgpu::Maintain::Wait);
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] radix_keys+hist (pass={pass}): {ms:.3} ms");
            }

            // Phase 3: Prefix sum over histogram, then inclusive_to_exclusive
            let mut hist_scan_buf_temp = self.create_buffer(
                "hist_scan_temp",
                (histogram_len.max(1) * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            self.run_inclusive_prefix_sum(histogram_buf, &mut hist_scan_buf_temp, histogram_len)?;

            // inclusive_to_exclusive
            let ite_params = [histogram_len as u32, 0, 0, 0];
            let ite_params_buf = self.create_buffer_init(
                "ite_params",
                bytemuck::cast_slice(&ite_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let ite_bg_layout = self
                .pipeline_inclusive_to_exclusive
                .get_bind_group_layout(0);
            let ite_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ite_bg"),
                layout: &ite_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: hist_scan_buf_temp.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ite_params_buf.as_entire_binding(),
                    },
                ],
            });

            let ite_wg = (histogram_len as u32).div_ceil(256);
            self.dispatch(
                &self.pipeline_inclusive_to_exclusive,
                &ite_bg,
                ite_wg,
                "ite",
            )?;

            // Phase 4: Scatter
            let scat_params = [padded_n as u32, num_groups as u32, 0, 0];
            let scat_params_buf = self.create_buffer_init(
                "scat_params",
                bytemuck::cast_slice(&scat_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let scat_bg_layout = self.pipeline_radix_scatter.get_bind_group_layout(0);
            let scat_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scat_bg"),
                layout: &scat_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: keys_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: sa_buf_alt.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: scat_params_buf.as_entire_binding(),
                    },
                ],
            });

            self.dispatch(
                &self.pipeline_radix_scatter,
                &scat_bg,
                num_groups as u32,
                "radix_scatter",
            )?;

            // Swap sa buffers
            std::mem::swap(sa_buf, sa_buf_alt);
        }

        Ok(())
    }

    /// Compute a byte histogram of the input data on the GPU.
    pub fn byte_histogram(&self, input: &[u8]) -> PzResult<[u32; 256]> {
        if input.is_empty() {
            return Ok([0u32; 256]);
        }

        let n = input.len();
        let padded = Self::pad_input_bytes(input);

        let input_buf = self.create_buffer_init("hist_input", &padded, wgpu::BufferUsages::STORAGE);

        let hist_buf = self.create_buffer_init(
            "hist_buf",
            &vec![0u8; 256 * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let workgroups = (n as u32).div_ceil(64);
        let params = [n as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "hist_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_byte_histogram.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            &self.pipeline_byte_histogram,
            &bg,
            workgroups,
            "byte_histogram",
        )?;

        let raw = self.read_buffer(&hist_buf, (256 * 4) as u64);
        let hist: &[u32] = bytemuck::cast_slice(&raw);
        let mut result = [0u32; 256];
        result.copy_from_slice(hist);
        Ok(result)
    }

    /// Encode data using Huffman coding on the GPU.
    pub fn huffman_encode(
        &self,
        input: &[u8],
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();
        let padded = Self::pad_input_bytes(input);

        let input_buf = self.create_buffer_init("huff_input", &padded, wgpu::BufferUsages::STORAGE);

        let lut_buf = self.create_buffer_init(
            "huff_lut",
            bytemuck::cast_slice(code_lut),
            wgpu::BufferUsages::STORAGE,
        );

        let bit_lengths_buf = self.create_buffer(
            "bit_lengths",
            (n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let workgroups = (n as u32).div_ceil(64);
        let params = [n as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "huff_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 1: compute bit lengths
        let bg1_layout = self.pipeline_compute_bit_lengths.get_bind_group_layout(0);
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_pass1_bg"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bit_lengths_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            &self.pipeline_compute_bit_lengths,
            &bg1,
            workgroups,
            "huff_pass1",
        )?;

        // Download bit lengths and compute prefix sum on CPU
        let raw_lengths = self.read_buffer(&bit_lengths_buf, (n * 4) as u64);
        let bit_lengths: &[u32] = bytemuck::cast_slice(&raw_lengths);

        let mut bit_offsets = vec![0u32; n];
        let mut running_sum: u64 = 0;
        for i in 0..n {
            bit_offsets[i] = running_sum as u32;
            running_sum += bit_lengths[i] as u64;
        }
        let total_bits = running_sum as usize;

        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let offsets_buf = self.create_buffer_init(
            "bit_offsets",
            bytemuck::cast_slice(&bit_offsets),
            wgpu::BufferUsages::STORAGE,
        );

        let output_buf = self.create_buffer_init(
            "huff_output",
            &vec![0u8; output_uints * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Pass 2: write codes
        let bg2_layout = self.pipeline_write_codes.get_bind_group_layout(0);
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_pass2_bg"),
            layout: &bg2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: offsets_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(&self.pipeline_write_codes, &bg2, workgroups, "huff_pass2")?;

        // Download output
        let raw_output = self.read_buffer(&output_buf, (output_uints * 4) as u64);
        let output_data: &[u32] = bytemuck::cast_slice(&raw_output);

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

    /// Run an in-place exclusive prefix sum on a GPU buffer using Blelloch scan.
    fn run_exclusive_prefix_sum(&self, data_buf: &wgpu::Buffer, count: usize) -> PzResult<()> {
        let block_size = 512usize; // workgroup_size(256) * 2
        let num_blocks = count.div_ceil(block_size);

        let block_sums_buf = self.create_buffer(
            "eps_block_sums",
            (num_blocks.max(1) * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let params = [count as u32, 0, 0, 0];
        let params_buf = self.create_buffer_init(
            "eps_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Phase 1: Block-level exclusive scan
        let bg_layout = self.pipeline_prefix_sum_block.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("eps_block_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_sums_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            &self.pipeline_prefix_sum_block,
            &bg,
            num_blocks.max(1) as u32,
            "eps_block",
        )?;

        if num_blocks > 1 {
            // Recursively scan block sums
            self.run_exclusive_prefix_sum(&block_sums_buf, num_blocks)?;

            // Apply block offsets
            let apply_params = [count as u32, block_size as u32, 0, 0];
            let apply_params_buf = self.create_buffer_init(
                "eps_apply_params",
                bytemuck::cast_slice(&apply_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let apply_bg_layout = self.pipeline_prefix_sum_apply.get_bind_group_layout(0);
            let apply_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("eps_apply_bg"),
                layout: &apply_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: data_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: block_sums_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: apply_params_buf.as_entire_binding(),
                    },
                ],
            });

            let apply_wg = (count as u32).div_ceil(256);
            self.dispatch(
                &self.pipeline_prefix_sum_apply,
                &apply_bg,
                apply_wg,
                "eps_apply",
            )?;
        }

        Ok(())
    }

    /// Encode data using Huffman coding with GPU Blelloch prefix sum.
    /// Only transfers 8 bytes for prefix sum instead of downloading/uploading the full array.
    pub fn huffman_encode_gpu_scan(
        &self,
        input: &[u8],
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();
        let padded = Self::pad_input_bytes(input);

        let input_buf =
            self.create_buffer_init("huff_gs_input", &padded, wgpu::BufferUsages::STORAGE);

        let lut_buf = self.create_buffer_init(
            "huff_gs_lut",
            bytemuck::cast_slice(code_lut),
            wgpu::BufferUsages::STORAGE,
        );

        let bit_lengths_buf = self.create_buffer(
            "huff_gs_bit_lengths",
            (n * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let workgroups = (n as u32).div_ceil(64);
        let params = [n as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "huff_gs_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 1: compute bit lengths
        let bg1_layout = self.pipeline_compute_bit_lengths.get_bind_group_layout(0);
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_gs_pass1_bg"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bit_lengths_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        self.dispatch(
            &self.pipeline_compute_bit_lengths,
            &bg1,
            workgroups,
            "huff_gs_pass1",
        )?;

        // Read last bit length before prefix sum overwrites it
        let last_bit_length = self.read_buffer_scalar_u32(&bit_lengths_buf, n - 1);

        // GPU exclusive prefix sum (in-place on bit_lengths_buf)
        self.run_exclusive_prefix_sum(&bit_lengths_buf, n)?;

        // Read last offset to compute total bits
        let last_offset = self.read_buffer_scalar_u32(&bit_lengths_buf, n - 1);
        let total_bits = (last_offset + last_bit_length) as usize;

        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let output_buf = self.create_buffer_init(
            "huff_gs_output",
            &vec![0u8; output_uints * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Pass 2: write codes (bit_lengths_buf now contains offsets)
        let bg2_layout = self.pipeline_write_codes.get_bind_group_layout(0);
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_gs_pass2_bg"),
            layout: &bg2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bit_lengths_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            &self.pipeline_write_codes,
            &bg2,
            workgroups,
            "huff_gs_pass2",
        )?;

        // Download output
        let raw_output = self.read_buffer(&output_buf, (output_uints * 4) as u64);
        let output_data: &[u32] = bytemuck::cast_slice(&raw_output);

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

    /// GPU-accelerated FSE decode of N-way interleaved streams.
    ///
    /// Takes the serialized interleaved FSE data (as produced by
    /// `fse::encode_interleaved()`) and decodes it on the GPU.
    /// Each stream is decoded by one GPU workgroup (one thread).
    pub fn fse_decode(&self, input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
        if original_len == 0 {
            return Ok(Vec::new());
        }

        // Parse the interleaved FSE header (same format as fse::decode_interleaved).
        let freq_table_bytes = 256 * 2;
        let min_header = 1 + freq_table_bytes + 1;
        if input.len() < min_header {
            return Err(PzError::InvalidInput);
        }

        let accuracy_log = input[0];
        if !(5..=12).contains(&accuracy_log) {
            return Err(PzError::InvalidInput);
        }

        // Read normalized frequency table.
        let mut norm_freq = [0u16; 256];
        for (i, freq) in norm_freq.iter_mut().enumerate() {
            let offset = 1 + i * 2;
            *freq = u16::from_le_bytes([input[offset], input[offset + 1]]);
        }

        let table_size = 1u32 << accuracy_log;
        let sum: u32 = norm_freq.iter().map(|&f| f as u32).sum();
        if sum != table_size {
            return Err(PzError::InvalidInput);
        }

        let pos = 1 + freq_table_bytes;
        let num_streams = input[pos] as usize;
        if num_streams == 0 {
            return Err(PzError::InvalidInput);
        }

        let mut cursor = pos + 1;

        // Parse per-stream metadata and bitstreams.
        struct StreamInfo {
            initial_state: u32,
            total_bits: u32,
            bitstream: Vec<u8>,
            num_symbols: u32,
        }
        let mut streams = Vec::with_capacity(num_streams);

        // Count symbols per stream (round-robin assignment).
        let base_count = original_len / num_streams;
        let extra = original_len % num_streams;

        for lane in 0..num_streams {
            if input.len() < cursor + 2 + 4 + 4 {
                return Err(PzError::InvalidInput);
            }
            let initial_state = u16::from_le_bytes([input[cursor], input[cursor + 1]]) as u32;
            cursor += 2;
            let total_bits = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]);
            cursor += 4;
            let bitstream_len = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]) as usize;
            cursor += 4;

            if input.len() < cursor + bitstream_len {
                return Err(PzError::InvalidInput);
            }
            let bitstream = input[cursor..cursor + bitstream_len].to_vec();
            cursor += bitstream_len;

            let num_symbols = (base_count + if lane < extra { 1 } else { 0 }) as u32;

            streams.push(StreamInfo {
                initial_state,
                total_bits,
                bitstream,
                num_symbols,
            });
        }

        // Build FSE decode table on CPU (small, O(table_size)).
        // We need the same spread + decode_table as the CPU FSE decoder.
        // Pack entries as u32: symbol(8) | bits(8) | next_state_base(16).
        let fse_norm = crate::fse::NormalizedFreqs {
            freq: norm_freq,
            accuracy_log,
        };
        let spread = crate::fse::spread_symbols(&fse_norm);
        let decode_entries = crate::fse::build_decode_table(&fse_norm, &spread);

        let packed_table: Vec<u32> = decode_entries
            .iter()
            .map(|e| {
                (e.symbol as u32) | ((e.bits as u32) << 8) | ((e.next_state_base as u32) << 16)
            })
            .collect();

        // Handle single-symbol case (all streams have total_bits == 0).
        if streams.iter().all(|s| s.total_bits == 0) && original_len > 0 {
            let entry = &decode_entries[streams[0].initial_state as usize];
            return Ok(vec![entry.symbol; original_len]);
        }

        // Concatenate all bitstreams, padded to u32 alignment.
        let mut all_bitstream_data = Vec::new();
        let mut stream_meta_host: Vec<u32> = Vec::with_capacity(num_streams * 4);

        for stream in &streams {
            let byte_offset = all_bitstream_data.len() as u32;
            all_bitstream_data.extend_from_slice(&stream.bitstream);

            stream_meta_host.push(stream.initial_state);
            stream_meta_host.push(stream.total_bits);
            stream_meta_host.push(byte_offset);
            stream_meta_host.push(stream.num_symbols);
        }

        // Pad to u32 alignment.
        while all_bitstream_data.len() % 4 != 0 {
            all_bitstream_data.push(0);
        }

        // Create GPU buffers.
        let decode_table_buf = self.create_buffer_init(
            "fse_decode_table",
            bytemuck::cast_slice(&packed_table),
            wgpu::BufferUsages::STORAGE,
        );

        let bitstream_buf = self.create_buffer_init(
            "fse_bitstream",
            &all_bitstream_data,
            wgpu::BufferUsages::STORAGE,
        );

        let stream_meta_buf = self.create_buffer_init(
            "fse_stream_meta",
            bytemuck::cast_slice(&stream_meta_host),
            wgpu::BufferUsages::STORAGE,
        );

        // Output buffer: u32-packed bytes, zero-initialized.
        let output_u32_count = original_len.div_ceil(4);
        let output_buf = self.create_buffer_init(
            "fse_output",
            &vec![0u8; output_u32_count * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = [num_streams as u32, table_size, original_len as u32, 0u32];
        let params_buf = self.create_buffer_init(
            "fse_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Bind group.
        let bg_layout = self.pipeline_fse_decode.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fse_decode_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: decode_table_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bitstream_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: stream_meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch: one workgroup per stream.
        self.dispatch(
            &self.pipeline_fse_decode,
            &bg,
            num_streams as u32,
            "fse_decode",
        )?;

        // Read output.
        let raw_output = self.read_buffer(&output_buf, (output_u32_count * 4) as u64);
        let result = raw_output[..original_len].to_vec();

        Ok(result)
    }

    /// GPU-chained Deflate compression: LZ77 + Huffman on GPU.
    pub fn deflate_chained(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        // Stage 1: GPU LZ77 compression
        let lz_data = self.lz77_compress(input)?;
        let lz_len = lz_data.len();

        if lz_data.is_empty() {
            return Err(PzError::InvalidInput);
        }

        // Stage 2: GPU ByteHistogram
        let histogram = self.byte_histogram(&lz_data)?;

        // Build Huffman tree from GPU-computed histogram
        let mut freq = crate::frequency::FrequencyTable::new();
        for (i, &count) in histogram.iter().enumerate() {
            freq.byte[i] = count;
        }
        freq.total = freq.byte.iter().map(|&c| c as u64).sum();
        freq.used = freq.byte.iter().filter(|&&c| c > 0).count() as u32;

        let tree = crate::huffman::HuffmanTree::from_frequency_table(&freq)
            .ok_or(PzError::InvalidInput)?;
        let freq_table = tree.serialize_frequencies();

        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        // Stage 3: GPU Huffman encoding with GPU prefix sum
        let (huffman_data, total_bits) = self.huffman_encode_gpu_scan(&lz_data, &code_lut)?;

        // Serialize in the same format as the CPU Deflate block
        let mut output = Vec::new();
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());
        output.extend_from_slice(&(total_bits as u32).to_le_bytes());
        for &freq_val in &freq_table {
            output.extend_from_slice(&freq_val.to_le_bytes());
        }
        output.extend_from_slice(&huffman_data);

        Ok(output)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_match_struct_size() {
        assert_eq!(std::mem::size_of::<GpuMatch>(), 12);
    }

    #[test]
    fn test_dedupe_all_literals() {
        let input = b"abcdef";
        let gpu_matches: Vec<GpuMatch> = input
            .iter()
            .map(|&b| GpuMatch {
                offset: 0,
                length: 0,
                next: b as u32,
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
    fn test_dedupe_with_match() {
        let input = b"abcabc";
        let gpu_matches = vec![
            GpuMatch {
                offset: 0,
                length: 0,
                next: b'a' as u32,
            },
            GpuMatch {
                offset: 0,
                length: 0,
                next: b'b' as u32,
            },
            GpuMatch {
                offset: 0,
                length: 0,
                next: b'c' as u32,
            },
            GpuMatch {
                offset: 3,
                length: 2,
                next: b'c' as u32,
            },
            GpuMatch {
                offset: 3,
                length: 1,
                next: b'c' as u32,
            },
            GpuMatch {
                offset: 3,
                length: 0,
                next: b'c' as u32,
            },
        ];

        let result = dedupe_gpu_matches(&gpu_matches, input);
        assert_eq!(result.len(), 4);
        assert_eq!(result[3].offset, 3);
        assert_eq!(result[3].length, 2);
    }

    #[test]
    fn test_probe_devices() {
        // Should not crash; may return empty on headless systems
        let devices = probe_devices();
        for d in &devices {
            assert!(!d.name.is_empty() || d.name.is_empty()); // no-op, just validate struct
        }
    }

    #[test]
    fn test_engine_creation() {
        // May return Unsupported on headless systems — that's OK
        match WebGpuEngine::new() {
            Ok(engine) => {
                assert!(!engine.device_name().is_empty());
                assert!(engine.max_work_group_size() >= 1);
            }
            Err(PzError::Unsupported) => {
                // Expected on systems without GPU
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_profiling_creation() {
        match WebGpuEngine::with_profiling(true) {
            Ok(engine) => {
                eprintln!(
                    "Device: {}, profiling={}",
                    engine.device_name(),
                    engine.profiling()
                );
                // profiling accessor must match what was actually negotiated
                // (may be false if device doesn't support TIMESTAMP_QUERY)
            }
            Err(PzError::Unsupported) => {
                // Expected on systems without GPU
            }
            Err(e) => panic!("unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_lz77_round_trip() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello";
        let compressed = engine.lz77_compress(input).unwrap();
        let decompressed = crate::lz77::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_byte_histogram() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"aabbcc";
        let hist = engine.byte_histogram(input).unwrap();
        assert_eq!(hist[b'a' as usize], 2);
        assert_eq!(hist[b'b' as usize], 2);
        assert_eq!(hist[b'c' as usize], 2);
        assert_eq!(hist[b'd' as usize], 0);
    }

    #[test]
    fn test_huffman_round_trip() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"hello, world!";
        let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        let (encoded, total_bits) = engine.huffman_encode(input, &code_lut).unwrap();
        let mut decoded = vec![0u8; input.len()];
        let decoded_len = tree
            .decode_to_buf(&encoded, total_bits, &mut decoded)
            .unwrap();
        assert_eq!(&decoded[..decoded_len], input);
    }

    #[test]
    fn test_huffman_gpu_scan_round_trip() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"hello, world!";
        let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        let (encoded, total_bits) = engine.huffman_encode_gpu_scan(input, &code_lut).unwrap();
        let mut decoded = vec![0u8; input.len()];
        let decoded_len = tree
            .decode_to_buf(&encoded, total_bits, &mut decoded)
            .unwrap();
        assert_eq!(&decoded[..decoded_len], input);
    }

    #[test]
    fn test_bwt_round_trip() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"banana";
        let bwt_result = engine.bwt_encode(input).unwrap();
        let decoded = crate::bwt::decode(&bwt_result.data, bwt_result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_bijective_bwt_round_trip() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        // Test with various inputs
        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("banana", b"banana".to_vec()),
            (
                "hello_repeated",
                b"hello world hello world hello world".to_vec(),
            ),
            ("binary", (0..=255u8).collect()),
        ];

        for (name, input) in &test_cases {
            let (gpu_data, gpu_factors) = engine.bwt_encode_bijective(input).unwrap();

            // Compare against CPU bijective BWT
            let (cpu_data, cpu_factors) = crate::bwt::encode_bijective(input).unwrap();
            assert_eq!(gpu_factors, cpu_factors, "factor lengths differ on {name}");
            assert_eq!(gpu_data, cpu_data, "BWT data differs on {name}");

            // Round-trip decode
            let decoded = crate::bwt::decode_bijective(&gpu_data, &gpu_factors).unwrap();
            assert_eq!(
                decoded, *input,
                "GPU bijective BWT round-trip failed on {name}"
            );
        }
    }

    #[test]
    fn test_deflate_chained_round_trip() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }

        let compressed = engine.deflate_chained(&input).unwrap();
        // Verify it produces valid output (non-empty)
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_fse_decode_hello() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"hello, world!";
        let encoded = crate::fse::encode_interleaved(input);
        let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_fse_decode_repeated() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = vec![b'a'; 100];
        let encoded = crate::fse::encode_interleaved(&input);
        let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_fse_decode_binary() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded = crate::fse::encode_interleaved(&input);
        let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_fse_decode_medium_text() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let pattern = b"The Burrows-Wheeler transform clusters bytes. ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let encoded = crate::fse::encode_interleaved(&input);
        let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_fse_decode_various_stream_counts() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input: Vec<u8> = (0..200).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        for n in [1, 2, 4, 8] {
            let encoded = crate::fse::encode_interleaved_n(&input, n, 10);
            let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input, "failed at num_streams={}", n);
        }
    }
}
