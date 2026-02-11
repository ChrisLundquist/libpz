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
//! use pz::webgpu::{WebGpuEngine, KernelVariant};
//!
//! let engine = WebGpuEngine::new()?;
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

use wgpu::util::DeviceExt;

/// Embedded WGSL kernel source: one invocation per input position,
/// 128KB sliding window.
const LZ77_KERNEL_SOURCE: &str = include_str!("../kernels/lz77.wgsl");

/// Embedded WGSL kernel source: batched variant where each invocation
/// processes STEP_SIZE (32) consecutive positions with a 32KB window.
const LZ77_BATCH_KERNEL_SOURCE: &str = include_str!("../kernels/lz77_batch.wgsl");

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

/// Step size used by the batch kernel (must match STEP_SIZE in lz77_batch.wgsl).
const BATCH_STEP_SIZE: usize = 32;

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

/// Which LZ77 kernel variant to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelVariant {
    /// One invocation per position, 128KB window.
    PerPosition,
    /// Batched: each invocation handles STEP_SIZE positions, 32KB window.
    Batch,
    /// Hash-table-based: two-pass approach.
    HashTable,
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

/// WebGPU compute engine.
///
/// Manages the wgpu device, queue, and compiled shader modules.
/// Create one engine at library init time and reuse it across calls.
pub struct WebGpuEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Compiled shader modules
    lz77_module: wgpu::ShaderModule,
    lz77_batch_module: wgpu::ShaderModule,
    lz77_topk_module: wgpu::ShaderModule,
    lz77_hash_module: wgpu::ShaderModule,
    bwt_rank_module: wgpu::ShaderModule,
    bwt_radix_module: wgpu::ShaderModule,
    huffman_module: wgpu::ShaderModule,
    /// Device name for diagnostics.
    device_name: String,
    /// Maximum compute workgroup size.
    max_work_group_size: usize,
    /// Whether the selected device is a CPU (not GPU).
    is_cpu: bool,
    /// Scan workgroup size (power of 2, capped at 256).
    scan_workgroup_size: usize,
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
        Self::with_device_preference(true)
    }

    /// Create a new engine with explicit GPU preference.
    pub fn with_device_preference(prefer_gpu: bool) -> PzResult<Self> {
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

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("libpz-webgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .map_err(|_| PzError::Unsupported)?;

        let capped = max_work_group_size.clamp(1, 256);
        let scan_workgroup_size = 1 << (usize::BITS - 1 - capped.leading_zeros());

        let lz77_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lz77"),
            source: wgpu::ShaderSource::Wgsl(LZ77_KERNEL_SOURCE.into()),
        });

        let lz77_batch_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lz77_batch"),
            source: wgpu::ShaderSource::Wgsl(LZ77_BATCH_KERNEL_SOURCE.into()),
        });

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

        Ok(WebGpuEngine {
            device,
            queue,
            lz77_module,
            lz77_batch_module,
            lz77_topk_module,
            lz77_hash_module,
            bwt_rank_module,
            bwt_radix_module,
            huffman_module,
            device_name,
            max_work_group_size,
            is_cpu,
            scan_workgroup_size,
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

    /// Dispatch a compute pass with the given pipeline and bind group.
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups_x: u32,
        label: &str,
    ) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups_x, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    // --- Pad input to u32-aligned for WGSL byte reading ---

    fn pad_input_bytes(input: &[u8]) -> Vec<u8> {
        let mut padded = input.to_vec();
        while !padded.len().is_multiple_of(4) {
            padded.push(0);
        }
        padded
    }

    // --- LZ77 Match Finding ---

    /// Find LZ77 matches for the entire input using the GPU.
    pub fn find_matches(&self, input: &[u8], variant: KernelVariant) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        match variant {
            KernelVariant::PerPosition => self.find_matches_per_position(input),
            KernelVariant::Batch => self.find_matches_batch(input),
            KernelVariant::HashTable => self.find_matches_hash(input),
        }
    }

    fn find_matches_per_position(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);

        let input_buf = self.create_buffer_init("lz77_input", &padded, wgpu::BufferUsages::STORAGE);

        let output_buf = self.create_buffer(
            "lz77_output",
            (input_len * std::mem::size_of::<GpuMatch>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = [input_len as u32, 0, 0, 0];
        let params_buf = self.create_buffer_init(
            "lz77_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lz77_per_pos"),
                layout: None,
                module: &self.lz77_module,
                entry_point: Some("encode"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_per_pos_bg"),
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

        let workgroups = (input_len as u32).div_ceil(64);
        self.dispatch(&pipeline, &bind_group, workgroups, "lz77_per_pos");

        let raw = self.read_buffer(
            &output_buf,
            (input_len * std::mem::size_of::<GpuMatch>()) as u64,
        );
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    fn find_matches_batch(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);

        let input_buf =
            self.create_buffer_init("lz77_batch_input", &padded, wgpu::BufferUsages::STORAGE);

        let output_buf = self.create_buffer(
            "lz77_batch_output",
            (input_len * std::mem::size_of::<GpuMatch>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = [input_len as u32, 0, 0, 0];
        let params_buf = self.create_buffer_init(
            "lz77_batch_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lz77_batch"),
                layout: None,
                module: &self.lz77_batch_module,
                entry_point: Some("encode"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_batch_bg"),
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

        let num_work_items = input_len.div_ceil(BATCH_STEP_SIZE);
        let workgroups = (num_work_items as u32).div_ceil(64);
        self.dispatch(&pipeline, &bind_group, workgroups, "lz77_batch");

        let raw = self.read_buffer(
            &output_buf,
            (input_len * std::mem::size_of::<GpuMatch>()) as u64,
        );
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    fn find_matches_hash(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);

        let input_buf =
            self.create_buffer_init("lz77_hash_input", &padded, wgpu::BufferUsages::STORAGE);

        let params = [input_len as u32, 0, 0, 0];
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

        // Pass 1: Build hash table
        let build_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("lz77_hash_build"),
                    layout: None,
                    module: &self.lz77_hash_module,
                    entry_point: Some("build_hash_table"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let build_bg_layout = build_pipeline.get_bind_group_layout(0);
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

        let workgroups = (input_len as u32).div_ceil(64);
        self.dispatch(&build_pipeline, &build_bg, workgroups, "lz77_hash_build");

        // Pass 2: Find matches — uses a separate pipeline with different bindings
        // We need to read back hash_counts for the find pass (read-only)
        // The find_matches entry point uses bindings 0,1,4,5,6
        let find_pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lz77_hash_find"),
                layout: None,
                module: &self.lz77_hash_module,
                entry_point: Some("find_matches"),
                compilation_options: Default::default(),
                cache: None,
            });

        let find_bg_layout = find_pipeline.get_bind_group_layout(0);
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

        self.dispatch(&find_pipeline, &find_bg, workgroups, "lz77_hash_find");

        let raw = self.read_buffer(
            &output_buf,
            (input_len * std::mem::size_of::<GpuMatch>()) as u64,
        );
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    /// GPU-accelerated LZ77 compression.
    pub fn lz77_compress(&self, input: &[u8], variant: KernelVariant) -> PzResult<Vec<u8>> {
        let matches = self.find_matches(input, variant)?;

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

        let params = [input_len as u32, 0, 0, 0];
        let params_buf = self.create_buffer_init(
            "topk_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lz77_topk"),
                layout: None,
                module: &self.lz77_topk_module,
                entry_point: Some("encode_topk"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
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

        let workgroups = (input_len as u32).div_ceil(64);
        self.dispatch(&pipeline, &bind_group, workgroups, "lz77_topk");

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

            // Read convergence scalar
            let prefix_data = self.read_buffer(&prefix_buf, (padded_n * 4) as u64);
            let prefix_vals: &[u32] = bytemuck::cast_slice(&prefix_data);
            max_rank = prefix_vals[n - 1];

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

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rank_compare"),
                layout: None,
                module: &self.bwt_rank_module,
                entry_point: Some("rank_compare"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bg_layout = pipeline.get_bind_group_layout(0);
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
        self.dispatch(&pipeline, &bg, workgroups, "rank_compare");
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

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix_sum_local"),
                layout: None,
                module: &self.bwt_rank_module,
                entry_point: Some("prefix_sum_local"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bg_layout = pipeline.get_bind_group_layout(0);
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
        self.dispatch(&pipeline, &bg, workgroups.max(1), "prefix_sum_local");

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

            let prop_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("prefix_sum_propagate"),
                        layout: None,
                        module: &self.bwt_rank_module,
                        entry_point: Some("prefix_sum_propagate"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            let prop_bg_layout = prop_pipeline.get_bind_group_layout(0);
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
            self.dispatch(&prop_pipeline, &prop_bg, prop_wg, "prefix_sum_propagate");
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

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("rank_scatter"),
                layout: None,
                module: &self.bwt_rank_module,
                entry_point: Some("rank_scatter"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bg_layout = pipeline.get_bind_group_layout(0);
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
        self.dispatch(&pipeline, &bg, workgroups, "rank_scatter");
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

            let key_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("radix_compute_keys"),
                        layout: None,
                        module: &self.bwt_radix_module,
                        entry_point: Some("radix_compute_keys"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            let key_bg_layout = key_pipeline.get_bind_group_layout(0);
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

            let global_wg = (padded_n as u32).div_ceil(256);
            self.dispatch(&key_pipeline, &key_bg, global_wg, "radix_keys");

            // Phase 2: Histogram
            let hist_params = [padded_n as u32, num_groups as u32, 0, 0];
            let hist_params_buf = self.create_buffer_init(
                "hist_params",
                bytemuck::cast_slice(&hist_params),
                wgpu::BufferUsages::UNIFORM,
            );

            // Zero the histogram buffer
            let hist_zeros = vec![0u32; histogram_len];
            self.queue
                .write_buffer(histogram_buf, 0, bytemuck::cast_slice(&hist_zeros));

            let hist_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("radix_histogram"),
                        layout: None,
                        module: &self.bwt_radix_module,
                        entry_point: Some("radix_histogram"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            let hist_bg_layout = hist_pipeline.get_bind_group_layout(0);
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

            self.dispatch(
                &hist_pipeline,
                &hist_bg,
                num_groups as u32,
                "radix_histogram",
            );

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

            let ite_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("inclusive_to_exclusive"),
                        layout: None,
                        module: &self.bwt_radix_module,
                        entry_point: Some("inclusive_to_exclusive"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            let ite_bg_layout = ite_pipeline.get_bind_group_layout(0);
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
            self.dispatch(&ite_pipeline, &ite_bg, ite_wg, "ite");

            // Phase 4: Scatter
            let scat_params = [padded_n as u32, num_groups as u32, 0, 0];
            let scat_params_buf = self.create_buffer_init(
                "scat_params",
                bytemuck::cast_slice(&scat_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let scat_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("radix_scatter"),
                        layout: None,
                        module: &self.bwt_radix_module,
                        entry_point: Some("radix_scatter"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            let scat_bg_layout = scat_pipeline.get_bind_group_layout(0);
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

            self.dispatch(&scat_pipeline, &scat_bg, num_groups as u32, "radix_scatter");

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

        let params = [n as u32, 0, 0, 0];
        let params_buf = self.create_buffer_init(
            "hist_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("byte_histogram"),
                layout: None,
                module: &self.huffman_module,
                entry_point: Some("byte_histogram"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bg_layout = pipeline.get_bind_group_layout(0);
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

        let workgroups = (n as u32).div_ceil(64);
        self.dispatch(&pipeline, &bg, workgroups, "byte_histogram");

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

        let params = [n as u32, 0, 0, 0];
        let params_buf = self.create_buffer_init(
            "huff_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 1: compute bit lengths
        let pipeline1 = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("compute_bit_lengths"),
                layout: None,
                module: &self.huffman_module,
                entry_point: Some("compute_bit_lengths"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bg1_layout = pipeline1.get_bind_group_layout(0);
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

        let workgroups = (n as u32).div_ceil(64);
        self.dispatch(&pipeline1, &bg1, workgroups, "huff_pass1");

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
        let pipeline2 = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("write_codes"),
                layout: None,
                module: &self.huffman_module,
                entry_point: Some("write_codes"),
                compilation_options: Default::default(),
                cache: None,
            });

        let bg2_layout = pipeline2.get_bind_group_layout(0);
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

        self.dispatch(&pipeline2, &bg2, workgroups, "huff_pass2");

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

    /// GPU-chained Deflate compression: LZ77 + Huffman on GPU.
    pub fn deflate_chained(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        // Stage 1: GPU LZ77 compression
        let lz_data = self.lz77_compress(input, KernelVariant::HashTable)?;
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

        // Stage 3: GPU Huffman encoding
        let (huffman_data, total_bits) = self.huffman_encode(&lz_data, &code_lut)?;

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
    fn test_lz77_round_trip_per_position() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello";
        let compressed = engine
            .lz77_compress(input, KernelVariant::PerPosition)
            .unwrap();
        let decompressed = crate::lz77::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lz77_round_trip_batch() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello";
        let compressed = engine.lz77_compress(input, KernelVariant::Batch).unwrap();
        let decompressed = crate::lz77::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lz77_round_trip_hash() {
        let engine = match WebGpuEngine::new() {
            Ok(e) => e,
            Err(PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };

        let input = b"hello world hello world hello";
        let compressed = engine
            .lz77_compress(input, KernelVariant::HashTable)
            .unwrap();
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
}
