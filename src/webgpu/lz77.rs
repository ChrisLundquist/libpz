//! LZ77 GPU match finding kernels.

use super::*;

// ---------------------------------------------------------------------------
// Buffer ring for double/triple-buffered GPU streaming
// ---------------------------------------------------------------------------

pub(crate) use crate::gpu_common::BufferRing;

/// Pre-allocated GPU buffer set for one block's LZ77 lazy computation.
///
/// Reusing pre-allocated slots across blocks avoids per-block buffer
/// creation overhead and enables overlapped GPU/CPU execution: while
/// the GPU computes on slot N, the CPU reads back results from slot N-1.
pub(crate) struct Lz77BufferSlot {
    pub(crate) input_buf: wgpu::Buffer,
    pub(crate) params_buf: wgpu::Buffer,
    /// Separate params for the resolve pass (different dispatch width).
    pub(crate) params_resolve_buf: wgpu::Buffer,
    pub(crate) raw_match_buf: wgpu::Buffer,
    pub(crate) resolved_buf: wgpu::Buffer,
    pub(crate) staging_buf: wgpu::Buffer,
    pub(crate) capacity: usize,
}

impl WebGpuEngine {
    // --- LZ77 Match Finding ---

    /// Find LZ77 matches for the entire input using the GPU.
    ///
    /// Uses the per-workgroup shared-memory hash table kernel (lz77_local.wgsl)
    /// for high throughput. Each workgroup processes an independent 4KB block
    /// with its own hash table in shared memory, followed by lazy resolution.
    ///
    /// For higher quality (larger match window), use [`find_matches_coop()`].
    pub fn find_matches(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        self.find_matches_local_impl(input)
    }

    /// Find LZ77 matches using the per-workgroup shared-memory hash table kernel.
    ///
    /// Each workgroup processes an independent 4KB block using a 4096-slot
    /// hash table in workgroup-local memory (16KB). This gives O(1) hash
    /// lookups per position instead of 1788 brute-force probes (coop) or
    /// global atomic contention (hash). Match window is limited to 4KB.
    pub fn find_matches_local(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        self.find_matches_local_impl(input)
    }

    /// Find LZ77 matches using the cooperative-stitch kernel.
    ///
    /// Uses a cooperative search strategy: each thread in a 64-thread workgroup
    /// searches a distinct offset band, shares top-K discoveries via shared
    /// memory, then all threads re-test all discovered offsets from their own
    /// positions. Covers [1, 33792] effective lookback with 1788 probes
    /// per thread (vs 4896 for brute-force scan of the same range).
    ///
    /// Higher quality (32KB window) but lower throughput than [`find_matches_local()`].
    pub fn find_matches_coop(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        self.find_matches_coop_impl(input)
    }

    /// Block size for the per-workgroup local hash table kernel (must match BLOCK_SIZE in lz77_local.wgsl).
    const LZ77_LOCAL_BLOCK_SIZE: u32 = 4096;

    /// Single-dispatch LZ77 for the full input, split into per-block match vectors.
    ///
    /// Instead of dispatching per pipeline block (N round-trips), this does:
    /// 1. One GPU dispatch on the entire input (all blocks at once)
    /// 2. One readback of the full GpuMatch array
    /// 3. CPU-side split by `block_size` boundaries + per-block deduplication
    ///
    /// This eliminates per-block submission overhead, bind group creation,
    /// and PCIe round-trips. The local kernel already processes independent
    /// 4KB sub-blocks internally, so pipeline block boundaries are respected.
    pub fn find_matches_bulk(&self, input: &[u8], block_size: usize) -> PzResult<Vec<Vec<Match>>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

        let input_buf =
            self.create_buffer_init("lz77_bulk_input", &padded, wgpu::BufferUsages::STORAGE);

        // Pass 1 dispatch: one workgroup per 4KB block across entire input
        let num_blocks = (input_len as u32).div_ceil(Self::LZ77_LOCAL_BLOCK_SIZE);
        let dw_find = self.dispatch_width(num_blocks, 64);
        let params_find = [input_len as u32, 0, 0, dw_find];
        let params_find_buf = self.create_buffer_init(
            "lz77_bulk_params_find",
            bytemuck::cast_slice(&params_find),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 2 dispatch: one thread per position (resolve_lazy)
        let wgs_resolve = (input_len as u32).div_ceil(64);
        let dw_resolve = self.dispatch_width(wgs_resolve, 64);
        let params_resolve = [input_len as u32, 0, 0, dw_resolve];
        let params_resolve_buf = self.create_buffer_init(
            "lz77_bulk_params_resolve",
            bytemuck::cast_slice(&params_resolve),
            wgpu::BufferUsages::UNIFORM,
        );

        let raw_match_buf = self.create_buffer(
            "bulk_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let resolved_buf = self.create_buffer(
            "bulk_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let staging_buf = self.create_buffer(
            "lz77_bulk_staging",
            match_buf_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Pass 1 bind group: find_matches_local
        let find_bg_layout = self.pipeline_lz77_local_find().get_bind_group_layout(0);
        let find_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_bulk_find_bg"),
            layout: &find_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_find_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Pass 2 bind group: resolve_lazy
        let resolve_bg_layout = self.pipeline_lz77_coop_resolve().get_bind_group_layout(0);
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_bulk_resolve_bg"),
            layout: &resolve_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_resolve_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resolved_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Single command encoder for both passes + staging copy
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lz77_bulk_submit"),
            });
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_local_find(),
            &find_bg,
            num_blocks,
            "lz77_bulk_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_coop_resolve(),
            &resolve_bg,
            wgs_resolve,
            "lz77_bulk_resolve",
        )?;
        encoder.copy_buffer_to_buffer(&resolved_buf, 0, &staging_buf, 0, match_buf_size);
        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        self.poll_wait();

        // Single readback
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.poll_wait();
        rx.recv().unwrap().map_err(|_| PzError::Unsupported)?;

        let raw = slice.get_mapped_range().to_vec();
        staging_buf.unmap();
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        // Split by pipeline block boundaries and dedupe each independently
        let blocks: Vec<&[u8]> = input.chunks(block_size).collect();
        let mut results = Vec::with_capacity(blocks.len());
        let mut offset = 0;
        for block in &blocks {
            let end = offset + block.len();
            let block_matches = &gpu_matches[offset..end];
            results.push(dedupe_gpu_matches(block_matches, block));
            offset = end;
        }

        Ok(results)
    }

    /// Internal: per-workgroup local hash table match finding.
    fn find_matches_local_impl(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

        let input_buf =
            self.create_buffer_init("lz77_local_input", &padded, wgpu::BufferUsages::STORAGE);

        // Pass 1 dispatch: one workgroup per 4KB block
        let num_blocks = (input_len as u32).div_ceil(Self::LZ77_LOCAL_BLOCK_SIZE);
        let dw_find = self.dispatch_width(num_blocks, 64);
        let params_find = [input_len as u32, 0, 0, dw_find];
        let params_find_buf = self.create_buffer_init(
            "lz77_local_params_find",
            bytemuck::cast_slice(&params_find),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 2 dispatch: one thread per position (resolve_lazy)
        let wgs_resolve = (input_len as u32).div_ceil(64);
        let dw_resolve = self.dispatch_width(wgs_resolve, 64);
        let params_resolve = [input_len as u32, 0, 0, dw_resolve];
        let params_resolve_buf = self.create_buffer_init(
            "lz77_local_params_resolve",
            bytemuck::cast_slice(&params_resolve),
            wgpu::BufferUsages::UNIFORM,
        );

        let raw_match_buf = self.create_buffer(
            "local_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let resolved_buf = self.create_buffer(
            "local_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let staging_buf = self.create_buffer(
            "lz77_local_staging",
            match_buf_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Pass 1 bind group: find_matches_local
        let find_bg_layout = self.pipeline_lz77_local_find().get_bind_group_layout(0);
        let find_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_local_find_bg"),
            layout: &find_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_find_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Pass 2 bind group: resolve_lazy (reuse from coop kernel)
        let resolve_bg_layout = self.pipeline_lz77_coop_resolve().get_bind_group_layout(0);
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_local_resolve_bg"),
            layout: &resolve_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_resolve_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resolved_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        if self.profiler.is_some() {
            // Separate encoders for profiling (AMD timestamp workaround)
            self.dispatch(
                self.pipeline_lz77_local_find(),
                &find_bg,
                num_blocks,
                "lz77_local_find",
            )?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lz77_local_resolve"),
                });
            self.record_dispatch(
                &mut encoder,
                self.pipeline_lz77_coop_resolve(),
                &resolve_bg,
                wgs_resolve,
                "lz77_local_resolve",
            )?;
            encoder.copy_buffer_to_buffer(&resolved_buf, 0, &staging_buf, 0, match_buf_size);
            self.profiler_resolve(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
        } else {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lz77_local_submit"),
                });
            self.record_dispatch(
                &mut encoder,
                self.pipeline_lz77_local_find(),
                &find_bg,
                num_blocks,
                "lz77_local_find",
            )?;
            self.record_dispatch(
                &mut encoder,
                self.pipeline_lz77_coop_resolve(),
                &resolve_bg,
                wgs_resolve,
                "lz77_local_resolve",
            )?;
            encoder.copy_buffer_to_buffer(&resolved_buf, 0, &staging_buf, 0, match_buf_size);
            self.profiler_resolve(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
        }

        self.poll_wait();
        if self.profiling {
            eprintln!("[pz-gpu] lz77_local (find+resolve): submitted");
        }

        // Readback (same as coop/lazy path)
        self.complete_find_matches_lazy(
            PendingLz77 {
                staging_buf,
                input_len,
            },
            input,
        )
    }

    /// Internal: cooperative-stitch match finding with submit/complete pattern.
    fn find_matches_coop_impl(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let pending = self.submit_find_matches_coop(input)?;
        self.poll_wait();
        if self.profiling {
            eprintln!("[pz-gpu] lz77_coop (find+resolve): submitted");
        }
        self.complete_find_matches_coop(pending, input)
    }

    /// Submit GPU LZ77 cooperative matching work without blocking for results.
    ///
    /// Creates buffers, encodes 2 compute passes (find_matches_coop +
    /// resolve_lazy) + staging copy in one command buffer, and submits.
    /// Returns a handle to retrieve results later with
    /// `complete_find_matches_coop()`.
    ///
    /// The caller must call `device.poll(Wait)` before completing.
    fn submit_find_matches_coop(&self, input: &[u8]) -> PzResult<PendingLz77> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

        let input_buf =
            self.create_buffer_init("lz77_coop_input", &padded, wgpu::BufferUsages::STORAGE);

        let workgroups = (input_len as u32).div_ceil(64);
        let params = [input_len as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "lz77_coop_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let raw_match_buf = self.create_buffer(
            "coop_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let resolved_buf = self.create_buffer(
            "coop_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let staging_buf = self.create_buffer(
            "lz77_coop_staging",
            match_buf_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Pass 1: cooperative match finding
        let find_bg_layout = self.pipeline_lz77_coop_find().get_bind_group_layout(0);
        let find_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_coop_find_bg"),
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
                    binding: 2,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Pass 2: resolve_lazy
        let resolve_bg_layout = self.pipeline_lz77_coop_resolve().get_bind_group_layout(0);
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_coop_resolve_bg"),
            layout: &resolve_bg_layout,
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
                    resource: resolved_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        if self.profiler.is_some() {
            // When profiling, use separate encoders per dispatch so each gets
            // its own resolve_query_set. AMD Vulkan drivers (RDNA 4 confirmed)
            // return zero timestamps for the 2nd dispatch in a multi-pass encoder.
            self.dispatch(
                self.pipeline_lz77_coop_find(),
                &find_bg,
                workgroups,
                "lz77_coop_find",
            )?;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lz77_coop_resolve"),
                });
            self.record_dispatch(
                &mut encoder,
                self.pipeline_lz77_coop_resolve(),
                &resolve_bg,
                workgroups,
                "lz77_coop_resolve",
            )?;
            encoder.copy_buffer_to_buffer(&resolved_buf, 0, &staging_buf, 0, match_buf_size);
            self.profiler_resolve(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
        } else {
            // Non-profiling: single encoder for both passes + staging copy (less overhead).
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("lz77_coop_submit"),
                });
            self.record_dispatch(
                &mut encoder,
                self.pipeline_lz77_coop_find(),
                &find_bg,
                workgroups,
                "lz77_coop_find",
            )?;
            self.record_dispatch(
                &mut encoder,
                self.pipeline_lz77_coop_resolve(),
                &resolve_bg,
                workgroups,
                "lz77_coop_resolve",
            )?;
            encoder.copy_buffer_to_buffer(&resolved_buf, 0, &staging_buf, 0, match_buf_size);
            self.profiler_resolve(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
        }

        Ok(PendingLz77 {
            staging_buf,
            input_len,
        })
    }

    /// Complete a previously submitted GPU LZ77 cooperative computation.
    ///
    /// The caller must ensure `device.poll(Wait)` has been called after
    /// submitting all pending work.
    fn complete_find_matches_coop(
        &self,
        pending: PendingLz77,
        input: &[u8],
    ) -> PzResult<Vec<Match>> {
        // Readback is identical to the lazy path — same PendingLz77 struct.
        self.complete_find_matches_lazy(pending, input)
    }

    /// Find LZ77 matches using the original greedy hash-table kernel (no lazy).
    pub fn find_matches_greedy(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        self.find_matches_hash(input)
    }

    /// Find LZ77 matches on the GPU, keeping the match buffer on-device.
    ///
    /// Unlike [`find_matches()`] which downloads and deduplicates immediately,
    /// this returns a [`GpuMatchBuf`] that stays on the GPU. The caller can:
    /// - Download later via [`download_and_dedupe()`] for CPU processing
    /// - (future) Pass to a GPU demux kernel without any PCI transfer
    ///
    /// This is the building block for zero-copy GPU pipeline composition.
    ///
    /// Uses the per-workgroup local hash table kernel for fast matching,
    /// keeping results in GPU memory instead of downloading to CPU.
    pub fn find_matches_to_device(&self, input: &[u8]) -> PzResult<GpuMatchBuf> {
        if input.is_empty() {
            let buf = self.create_buffer(
                "lz77_empty_matches",
                std::mem::size_of::<GpuMatch>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let input_buf = self.create_buffer("lz77_empty_input", 4, wgpu::BufferUsages::STORAGE);
            return Ok(GpuMatchBuf {
                buf,
                input_len: 0,
                input_buf,
            });
        }

        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

        let input_buf =
            self.create_buffer_init("lz77_tod_input", &padded, wgpu::BufferUsages::STORAGE);

        // Pass 1 params: one workgroup per BLOCK_SIZE block
        let num_blocks = (input_len as u32).div_ceil(Self::LZ77_LOCAL_BLOCK_SIZE);
        let dw_find = self.dispatch_width(num_blocks, 64);
        let params_find = [input_len as u32, 0, 0, dw_find];
        let params_find_buf = self.create_buffer_init(
            "lz77_tod_params_find",
            bytemuck::cast_slice(&params_find),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 2 params: one thread per position
        let wgs_resolve = (input_len as u32).div_ceil(64);
        let dw_resolve = self.dispatch_width(wgs_resolve, 64);
        let params_resolve = [input_len as u32, 0, 0, dw_resolve];
        let params_resolve_buf = self.create_buffer_init(
            "lz77_tod_params_resolve",
            bytemuck::cast_slice(&params_resolve),
            wgpu::BufferUsages::UNIFORM,
        );

        // Raw match output (pass 1 writes, pass 2 reads)
        let raw_match_buf = self.create_buffer(
            "tod_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Resolved match output (pass 2 writes) — this stays on GPU
        let resolved_buf = self.create_buffer(
            "tod_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // --- Pass 1 bind group: find_matches_local ---
        let find_bg_layout = self.pipeline_lz77_local_find().get_bind_group_layout(0);
        let find_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_tod_find_bg"),
            layout: &find_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_find_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // --- Pass 2 bind group: resolve_lazy ---
        let resolve_bg_layout = self.pipeline_lz77_coop_resolve().get_bind_group_layout(0);
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_tod_resolve_bg"),
            layout: &resolve_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_resolve_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: resolved_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Submit both passes in a single command encoder
        let t0 = if self.profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lz77_to_device"),
            });
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_local_find(),
            &find_bg,
            num_blocks,
            "lz77_tod_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_coop_resolve(),
            &resolve_bg,
            wgs_resolve,
            "lz77_tod_resolve",
        )?;
        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        if let Some(t0) = t0 {
            self.poll_wait();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lz77_to_device (find+resolve): {ms:.3} ms");
        }

        Ok(GpuMatchBuf {
            buf: resolved_buf,
            input_len,
            input_buf,
        })
    }

    /// Download a device-resident match buffer and deduplicate into `Match` structs.
    ///
    /// This is the download counterpart to [`find_matches_to_device()`].
    pub fn download_and_dedupe(
        &self,
        match_buf: &GpuMatchBuf,
        input: &[u8],
    ) -> PzResult<Vec<Match>> {
        if match_buf.input_len == 0 {
            return Ok(Vec::new());
        }

        let buf_size = (match_buf.input_len * std::mem::size_of::<GpuMatch>()) as u64;
        let raw = self.read_buffer(&match_buf.buf, buf_size);
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    fn find_matches_hash(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

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
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffer for readback (baked into same command buffer)
        let staging_buf = self.create_buffer(
            "lz77_hash_staging",
            match_buf_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Build + Find passes batched into a single command encoder submission
        let build_bg_layout = self.pipeline_lz77_hash_build().get_bind_group_layout(0);
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

        let find_bg_layout = self.pipeline_lz77_hash_find().get_bind_group_layout(0);
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
            self.pipeline_lz77_hash_build(),
            &build_bg,
            workgroups,
            "lz77_hash_build",
        )?;
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_hash_find(),
            &find_bg,
            workgroups,
            "lz77_hash_find",
        )?;
        // Bake staging copy into the same command buffer
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging_buf, 0, match_buf_size);
        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        // Wait for all work including the copy
        self.poll_wait();
        if let Some(t0) = t0 {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lz77_hash (build+find): {ms:.3} ms");
        }

        // Map the staging buffer directly (no extra submission needed)
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.poll_wait();
        rx.recv().unwrap().map_err(|_| PzError::Unsupported)?;

        let raw = slice.get_mapped_range().to_vec();
        staging_buf.unmap();
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    /// Find LZ77 matches using the 2-pass lazy matching kernel.
    ///
    /// Pass 1: Near brute-force scan (parallel, no hash table).
    /// Pass 2: Lazy resolve -- demote positions where pos+1 has a longer match.
    #[allow(dead_code)]
    pub(crate) fn find_matches_lazy(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let pending = self.submit_find_matches_lazy(input)?;
        self.poll_wait();
        if self.profiling {
            eprintln!("[pz-gpu] lz77_lazy (find+resolve): submitted");
        }
        self.complete_find_matches_lazy(pending, input)
    }

    /// Submit GPU LZ77 lazy matching work without blocking for results.
    ///
    /// Creates buffers, encodes 2 compute passes + staging copy in one
    /// command buffer, and submits. Returns a handle to retrieve results
    /// later with `complete_find_matches_lazy()`.
    ///
    /// The caller must call `device.poll(Wait)` before completing.
    #[allow(dead_code)]
    fn submit_find_matches_lazy(&self, input: &[u8]) -> PzResult<PendingLz77> {
        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

        let input_buf =
            self.create_buffer_init("lz77_lazy_input", &padded, wgpu::BufferUsages::STORAGE);

        let workgroups = (input_len as u32).div_ceil(64);
        let params = [input_len as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "lz77_lazy_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let raw_match_buf = self.create_buffer(
            "lazy_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let resolved_buf = self.create_buffer(
            "lazy_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Staging buffer for readback (copy encoded in same command buffer)
        let staging_buf = self.create_buffer(
            "lz77_lazy_staging",
            match_buf_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        // Pass 1: find_matches (near brute-force scan, no hash table needed)
        let find_bg_layout = self.pipeline_lz77_lazy_find().get_bind_group_layout(0);
        let find_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_lazy_find_bg"),
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
                    binding: 2,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Pass 2: resolve_lazy
        let resolve_bg_layout = self.pipeline_lz77_lazy_resolve().get_bind_group_layout(0);
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_lazy_resolve_bg"),
            layout: &resolve_bg_layout,
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
                    resource: resolved_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Encode 2 compute passes + staging copy in one command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lz77_lazy_submit"),
            });
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_lazy_find(),
            &find_bg,
            workgroups,
            "lz77_lazy_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_lazy_resolve(),
            &resolve_bg,
            workgroups,
            "lz77_lazy_resolve",
        )?;
        // Copy resolved matches to staging buffer (in same command buffer)
        encoder.copy_buffer_to_buffer(&resolved_buf, 0, &staging_buf, 0, match_buf_size);

        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        Ok(PendingLz77 {
            staging_buf,
            input_len,
        })
    }

    /// Complete a previously submitted GPU LZ77 computation.
    ///
    /// The caller must ensure `device.poll(Wait)` has been called after
    /// submitting all pending work.
    fn complete_find_matches_lazy(
        &self,
        pending: PendingLz77,
        input: &[u8],
    ) -> PzResult<Vec<Match>> {
        debug_assert_eq!(pending.input_len, input.len());

        let slice = pending.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.poll_wait();
        rx.recv().unwrap().map_err(|_| PzError::Unsupported)?;

        let raw = slice.get_mapped_range().to_vec();
        pending.staging_buf.unmap();

        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();
        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    /// GPU-accelerated LZ77 match finding for multiple blocks.
    ///
    /// Uses a ring of pre-allocated buffer slots to avoid per-block buffer
    /// creation overhead. While the GPU computes on slot N, the CPU reads
    /// back results from slot N-1 (double-buffered streaming).
    ///
    /// Falls back to CPU lazy matching for blocks that are too small or
    /// too large for GPU.
    pub fn find_matches_batched(&self, blocks: &[&[u8]]) -> PzResult<Vec<Vec<Match>>> {
        if blocks.is_empty() {
            return Ok(Vec::new());
        }

        if blocks.len() == 1 {
            return Ok(vec![self.find_matches(blocks[0])?]);
        }

        let max_dispatch = self.max_dispatch_input_size();
        let max_block = blocks.iter().map(|b| b.len()).max().unwrap_or(256 * 1024);

        // Try to allocate a ring of pre-allocated slots
        if let Some(mut ring) = self.create_lz77_ring(max_block) {
            return self.find_matches_batched_ring(blocks, max_dispatch, &mut ring);
        }

        // Fallback: no ring available (insufficient GPU memory), use per-block alloc
        self.find_matches_batched_alloc(blocks, max_dispatch)
    }

    /// Ring-based batched match finding: double/triple-buffered streaming.
    ///
    /// Pre-allocated buffer slots cycle through blocks. While the GPU
    /// computes on one slot, the CPU reads back from a previously
    /// completed slot, avoiding per-block buffer allocation overhead.
    fn find_matches_batched_ring(
        &self,
        blocks: &[&[u8]],
        max_dispatch: usize,
        ring: &mut BufferRing<Lz77BufferSlot>,
    ) -> PzResult<Vec<Vec<Match>>> {
        let ring_depth = ring.depth();
        let mut all_results: Vec<Option<Vec<Match>>> = (0..blocks.len()).map(|_| None).collect();

        // slot_inflight[slot_idx] = Some(block_idx) if that slot has pending GPU work
        let mut slot_inflight: Vec<Option<usize>> = vec![None; ring_depth];

        for (block_idx, block) in blocks.iter().enumerate() {
            // CPU fallback for edge cases
            if block.is_empty() || block.len() < MIN_GPU_INPUT_SIZE || block.len() > max_dispatch {
                if block.is_empty() {
                    all_results[block_idx] = Some(Vec::new());
                } else {
                    all_results[block_idx] = Some(crate::lz77::compress_lazy_to_matches(block)?);
                }
                continue;
            }

            let slot_idx = ring.acquire();

            // If this slot has previous in-flight work, complete it first
            if let Some(prev_idx) = slot_inflight[slot_idx].take() {
                self.poll_wait();
                all_results[prev_idx] =
                    Some(self.complete_lz77_from_slot(&ring.slots[slot_idx], blocks[prev_idx])?);
            }

            // Submit new block to this slot
            self.submit_lz77_to_slot(block, &ring.slots[slot_idx])?;
            slot_inflight[slot_idx] = Some(block_idx);
        }

        // Drain remaining in-flight slots
        for (slot_idx, inflight) in slot_inflight.iter_mut().enumerate() {
            if let Some(prev_idx) = inflight.take() {
                self.poll_wait();
                all_results[prev_idx] =
                    Some(self.complete_lz77_from_slot(&ring.slots[slot_idx], blocks[prev_idx])?);
            }
        }

        if self.profiling {
            eprintln!(
                "[pz-gpu] lz77_batched_ring: {} blocks, ring depth {}",
                blocks.len(),
                ring_depth
            );
        }

        Ok(all_results
            .into_iter()
            .map(|r| r.unwrap_or_default())
            .collect())
    }

    /// Fallback batched match finding with per-block buffer allocation.
    ///
    /// Used when the ring can't be allocated (insufficient GPU memory).
    fn find_matches_batched_alloc(
        &self,
        blocks: &[&[u8]],
        max_dispatch: usize,
    ) -> PzResult<Vec<Vec<Match>>> {
        let mut all_results: Vec<Vec<Match>> = Vec::with_capacity(blocks.len());

        let block_size = blocks.first().map(|b| b.len()).unwrap_or(256 * 1024);
        let mem_limit = self.max_in_flight(&self.cost_lz77_lazy, block_size);
        let batch_size = mem_limit.min(64);

        for chunk in blocks.chunks(batch_size) {
            let mut pending: Vec<Option<PendingLz77>> = Vec::with_capacity(chunk.len());

            for block in chunk {
                if block.is_empty()
                    || block.len() < MIN_GPU_INPUT_SIZE
                    || block.len() > max_dispatch
                {
                    pending.push(None);
                } else {
                    pending.push(Some(self.submit_find_matches_coop(block)?));
                }
            }

            self.poll_wait();

            for (i, p) in pending.into_iter().enumerate() {
                match p {
                    Some(pending_lz77) => {
                        all_results.push(self.complete_find_matches_coop(pending_lz77, chunk[i])?);
                    }
                    None => {
                        if chunk[i].is_empty() {
                            all_results.push(Vec::new());
                        } else {
                            all_results.push(crate::lz77::compress_lazy_to_matches(chunk[i])?);
                        }
                    }
                }
            }
        }

        if self.profiling {
            eprintln!(
                "[pz-gpu] lz77_batched_alloc: {} blocks processed",
                blocks.len()
            );
        }

        Ok(all_results)
    }

    /// GPU-accelerated batched LZ77 compression (serialized output).
    pub fn lz77_compress_batched(&self, blocks: &[&[u8]]) -> PzResult<Vec<Vec<u8>>> {
        let match_vecs = self.find_matches_batched(blocks)?;
        let mut results = Vec::with_capacity(match_vecs.len());
        for matches in &match_vecs {
            let mut output = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
            for m in matches {
                output.extend_from_slice(&m.to_bytes());
            }
            results.push(output);
        }
        Ok(results)
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

        let bind_group_layout = self.pipeline_lz77_topk().get_bind_group_layout(0);
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
            self.pipeline_lz77_topk(),
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
                    offset: p & 0xFFFF,
                    length: p >> 16,
                };
            }
        }

        Ok(table)
    }

    // --- Streaming buffer slot management ---

    /// Allocate a single pre-allocated buffer slot for LZ77 lazy matching.
    ///
    /// The slot contains 5 GPU buffers needed for one block: input,
    /// params, raw matches, resolved matches, and a staging buffer
    /// for CPU readback.
    pub(crate) fn alloc_lz77_slot(&self, max_block_size: usize) -> Lz77BufferSlot {
        let padded_size = ((max_block_size + 3) & !3) + 4;
        let match_buf_size = (max_block_size * std::mem::size_of::<GpuMatch>()) as u64;

        let input_buf = self.create_buffer(
            "slot_input",
            padded_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let params_buf = self.create_buffer(
            "slot_params",
            16, // 4 x u32
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let params_resolve_buf = self.create_buffer(
            "slot_params_resolve",
            16, // 4 x u32
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let raw_match_buf = self.create_buffer(
            "slot_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let resolved_buf = self.create_buffer(
            "slot_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging_buf = self.create_buffer(
            "slot_staging",
            match_buf_size,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        Lz77BufferSlot {
            input_buf,
            params_buf,
            params_resolve_buf,
            raw_match_buf,
            resolved_buf,
            staging_buf,
            capacity: max_block_size,
        }
    }

    /// Create a buffer ring for streaming LZ77 with depth based on GPU memory.
    ///
    /// Uses the `@pz_cost` model to estimate per-slot memory and allocates
    /// 2 slots (double buffer) or 3 slots (triple buffer) depending on
    /// available GPU memory. Returns `None` if even 2 slots don't fit.
    pub(crate) fn create_lz77_ring(&self, block_size: usize) -> Option<BufferRing<Lz77BufferSlot>> {
        let per_slot = self.cost_lz77_lazy.memory_bytes(block_size);
        if per_slot == 0 {
            return None;
        }
        // Reserve 25% headroom for non-ring allocations
        let budget = (self.gpu_memory_budget() * 3) / 4;
        let max_slots = budget / per_slot;
        if max_slots < 2 {
            return None;
        }
        let depth = max_slots.clamp(2, 3);

        let slots = (0..depth)
            .map(|_| self.alloc_lz77_slot(block_size))
            .collect();
        Some(BufferRing::new(slots))
    }

    /// Submit LZ77 matching work to a pre-allocated buffer slot.
    ///
    /// Uses the per-workgroup local hash table kernel (pass 1) followed by
    /// resolve_lazy (pass 2). Writes input data, encodes both compute passes
    /// and a staging copy into one command buffer, and submits without blocking.
    /// Call `complete_lz77_from_slot()` after `device.poll(Wait)` to
    /// read back results.
    pub(crate) fn submit_lz77_to_slot(&self, input: &[u8], slot: &Lz77BufferSlot) -> PzResult<()> {
        assert!(
            input.len() <= slot.capacity,
            "input {} exceeds slot capacity {}",
            input.len(),
            slot.capacity
        );

        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

        // Write input data to the slot's buffers (non-blocking GPU-side enqueue)
        self.queue.write_buffer(&slot.input_buf, 0, &padded);

        // Pass 1: local hash table — one workgroup per BLOCK_SIZE block
        let num_blocks = (input_len as u32).div_ceil(Self::LZ77_LOCAL_BLOCK_SIZE);
        let dw_find = self.dispatch_width(num_blocks, 64);
        let params_find = [input_len as u32, 0, 0, dw_find];
        self.queue
            .write_buffer(&slot.params_buf, 0, bytemuck::cast_slice(&params_find));

        // Pass 2: resolve_lazy — one thread per position
        let wgs_resolve = (input_len as u32).div_ceil(64);
        let dw_resolve = self.dispatch_width(wgs_resolve, 64);
        let params_resolve = [input_len as u32, 0, 0, dw_resolve];
        self.queue.write_buffer(
            &slot.params_resolve_buf,
            0,
            bytemuck::cast_slice(&params_resolve),
        );

        // Pass 1: find_matches_local (per-workgroup shared-memory hash table)
        let find_bg_layout = self.pipeline_lz77_local_find().get_bind_group_layout(0);
        let find_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_find_bg"),
            layout: &find_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: slot.input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slot.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: slot.raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Pass 2: resolve_lazy (reused from coop kernel)
        let resolve_bg_layout = self.pipeline_lz77_coop_resolve().get_bind_group_layout(0);
        let resolve_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("slot_resolve_bg"),
            layout: &resolve_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: slot.input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: slot.params_resolve_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: slot.resolved_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: slot.raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // Encode 2 compute passes + staging copy in one command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("slot_lz77_local"),
            });
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_local_find(),
            &find_bg,
            num_blocks,
            "slot_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_coop_resolve(),
            &resolve_bg,
            wgs_resolve,
            "slot_resolve",
        )?;
        encoder.copy_buffer_to_buffer(&slot.resolved_buf, 0, &slot.staging_buf, 0, match_buf_size);

        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// Complete a previously submitted slot-based LZ77 computation.
    ///
    /// Maps the staging buffer, reads back matches, unmaps, and deduplicates.
    /// The caller must call `poll_wait()` before this method to ensure GPU
    /// compute and the staging copy have completed. This method issues its
    /// own `map_async` + `poll_wait` internally to process the buffer mapping
    /// callback — that second poll is not redundant with the caller's.
    pub(crate) fn complete_lz77_from_slot(
        &self,
        slot: &Lz77BufferSlot,
        input: &[u8],
    ) -> PzResult<Vec<Match>> {
        let input_len = input.len();
        let match_buf_size = input_len * std::mem::size_of::<GpuMatch>();

        let slice = slot.staging_buf.slice(..match_buf_size as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.poll_wait();
        rx.recv().unwrap().map_err(|_| PzError::Unsupported)?;

        let raw = slice.get_mapped_range().to_vec();
        slot.staging_buf.unmap();

        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();
        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    /// GPU-accelerated LZ77 block-parallel decompression.
    ///
    /// Decompresses multiple independently-compressed LZ77 blocks in a single
    /// GPU dispatch. Each workgroup (64 threads) handles one block using a
    /// leader-follower pattern: thread 0 parses matches sequentially while
    /// all threads cooperate on back-reference copying.
    ///
    /// # Arguments
    ///
    /// * `block_data` - Concatenated serialized LZ77 match data for all blocks.
    /// * `block_meta` - Per-block metadata: `(match_data_offset, num_matches, decompressed_size)`.
    ///
    /// Returns the concatenated decompressed data from all blocks.
    pub fn lz77_decompress_blocks(
        &self,
        block_data: &[u8],
        block_meta: &[(usize, usize, usize)],
    ) -> PzResult<Vec<u8>> {
        if block_data.is_empty() || block_meta.is_empty() {
            return Ok(Vec::new());
        }

        let num_blocks = block_meta.len();

        // Compute output offsets via prefix sum of decompressed sizes
        let mut gpu_meta = Vec::with_capacity(num_blocks * 3);
        let mut output_offset = 0usize;
        for &(data_offset, num_matches, decompressed_size) in block_meta {
            gpu_meta.push(data_offset as u32);
            gpu_meta.push(num_matches as u32);
            gpu_meta.push(output_offset as u32);
            output_offset += decompressed_size;
        }
        let total_output_len = output_offset;

        if total_output_len == 0 {
            return Ok(Vec::new());
        }

        // Pad match data to u32-aligned for WGSL byte reading
        let mut padded_data = block_data.to_vec();
        while !padded_data.len().is_multiple_of(4) {
            padded_data.push(0);
        }

        // Create GPU buffers
        let match_data_buf = self.create_buffer_init(
            "lz77_dec_match_data",
            &padded_data,
            wgpu::BufferUsages::STORAGE,
        );

        let meta_buf = self.create_buffer_init(
            "lz77_dec_meta",
            bytemuck::cast_slice(&gpu_meta),
            wgpu::BufferUsages::STORAGE,
        );

        let output_u32_count = total_output_len.div_ceil(4);
        let output_buf = self.create_buffer_init(
            "lz77_dec_output",
            &vec![0u8; output_u32_count * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = [num_blocks as u32, total_output_len as u32, 0u32, 0u32];
        let params_buf = self.create_buffer_init(
            "lz77_dec_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_lz77_decode().get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_dec_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: match_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            self.pipeline_lz77_decode(),
            &bg,
            num_blocks as u32,
            "lz77_decode",
        )?;

        // Read output
        let raw_output = self.read_buffer(&output_buf, (output_u32_count * 4) as u64);
        Ok(raw_output[..total_output_len].to_vec())
    }
}
