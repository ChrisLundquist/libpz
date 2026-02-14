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
    pub(crate) raw_match_buf: wgpu::Buffer,
    pub(crate) resolved_buf: wgpu::Buffer,
    pub(crate) staging_buf: wgpu::Buffer,
    pub(crate) capacity: usize,
}

impl WebGpuEngine {
    // --- LZ77 Match Finding ---

    /// Find LZ77 matches for the entire input using the GPU lazy matching kernel.
    ///
    /// Uses a 3-pass approach: hash table build, per-position greedy matching,
    /// then parallel lazy resolution (demoting positions where the next position
    /// has a longer match). This produces compression quality comparable to
    /// CPU lazy matching while retaining full GPU parallelism.
    pub fn find_matches(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        self.find_matches_lazy(input)
    }

    /// Find LZ77 matches using the cooperative-stitch kernel.
    ///
    /// Uses a cooperative search strategy: each thread in a 64-thread workgroup
    /// searches a distinct offset band, shares top-K discoveries via shared
    /// memory, then all threads re-test all discovered offsets from their own
    /// positions. Covers [1, 4288] effective lookback with only 572 probes
    /// per thread (vs 4896 for brute-force scan of the same range).
    pub fn find_matches_coop(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

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

        // Encode 2 compute passes + staging copy
        let t0 = if self.profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lz77_coop"),
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

        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        if let Some(t0) = t0 {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lz77_coop (find+resolve): {ms:.3} ms");
        }

        // Read back and deduplicate
        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().map_err(|_| PzError::Unsupported)?;

        let raw = slice.get_mapped_range().to_vec();
        staging_buf.unmap();
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
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
    pub fn find_matches_to_device(&self, input: &[u8]) -> PzResult<GpuMatchBuf> {
        if input.is_empty() {
            let buf = self.create_buffer(
                "lz77_empty_matches",
                std::mem::size_of::<GpuMatch>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            return Ok(GpuMatchBuf { buf, input_len: 0 });
        }

        let input_len = input.len();
        let padded = Self::pad_input_bytes(input);
        let match_buf_size = (input_len * std::mem::size_of::<GpuMatch>()) as u64;

        let input_buf =
            self.create_buffer_init("lz77_tod_input", &padded, wgpu::BufferUsages::STORAGE);

        let workgroups = (input_len as u32).div_ceil(64);
        let params = [input_len as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "lz77_tod_params",
            bytemuck::cast_slice(&params),
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

        // --- Pass 1 bind group: find_matches (near brute-force scan) ---
        let find_bg_layout = self.pipeline_lz77_lazy_find().get_bind_group_layout(0);
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
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: raw_match_buf.as_entire_binding(),
                },
            ],
        });

        // --- Pass 2 bind group: resolve_lazy ---
        let resolve_bg_layout = self.pipeline_lz77_lazy_resolve().get_bind_group_layout(0);
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
            self.pipeline_lz77_lazy_find(),
            &find_bg,
            workgroups,
            "lz77_tod_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_lazy_resolve(),
            &resolve_bg,
            workgroups,
            "lz77_tod_resolve",
        )?;
        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        if let Some(t0) = t0 {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lz77_to_device (find+resolve): {ms:.3} ms");
        }

        Ok(GpuMatchBuf {
            buf: resolved_buf,
            input_len,
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
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
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
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
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
    fn find_matches_lazy(&self, input: &[u8]) -> PzResult<Vec<Match>> {
        let pending = self.submit_find_matches_lazy(input)?;
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
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
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().map_err(|_| PzError::Unsupported)?;

        let raw = slice.get_mapped_range().to_vec();
        pending.staging_buf.unmap();

        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();
        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    /// GPU-accelerated LZ77 match finding for multiple blocks.
    ///
    /// Submits GPU work for all blocks before reading back any results,
    /// hiding GPU-CPU transfer latency. Falls back to CPU lazy matching
    /// for blocks that are too small or too large for GPU.
    ///
    /// # Phases
    /// 1. **Submit**: Encode and submit all block dispatches (non-blocking).
    /// 2. **Sync**: Single `device.poll(Wait)` for all blocks.
    /// 3. **Readback**: Map each staging buffer, dedup matches.
    pub fn find_matches_batched(&self, blocks: &[&[u8]]) -> PzResult<Vec<Vec<Match>>> {
        if blocks.is_empty() {
            return Ok(Vec::new());
        }

        if blocks.len() == 1 {
            return Ok(vec![self.find_matches(blocks[0])?]);
        }

        let max_dispatch = self.max_dispatch_input_size();
        let mut all_results: Vec<Vec<Match>> = Vec::with_capacity(blocks.len());

        // Compute batch size from the kernel cost model and device memory budget.
        // Allow large batches to maximize GPU utilization — submitting all blocks
        // in one batch lets the GPU pipeline work across blocks while hiding
        // per-dispatch overhead. On discrete GPUs (e.g. 16GB VRAM) the memory
        // budget easily accommodates 32+ concurrent 256KB blocks (~18MB each).
        const GPU_MAX_BATCH: usize = 64;
        let block_size = blocks.first().map(|b| b.len()).unwrap_or(256 * 1024);
        let mem_limit = self.max_in_flight(&self.cost_lz77_lazy, block_size);
        let batch_size = mem_limit.min(GPU_MAX_BATCH);

        // Process in batches to cap GPU memory usage
        for chunk in blocks.chunks(batch_size) {
            // Phase 1: Submit all blocks in this batch
            let mut pending: Vec<Option<PendingLz77>> = Vec::with_capacity(chunk.len());

            for block in chunk {
                if block.is_empty()
                    || block.len() < MIN_GPU_INPUT_SIZE
                    || block.len() > max_dispatch
                {
                    pending.push(None); // CPU fallback
                } else {
                    pending.push(Some(self.submit_find_matches_lazy(block)?));
                }
            }

            // Phase 2: Wait for ALL GPU work in this batch to complete
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

            // Phase 3: Read back + dedup
            for (i, p) in pending.into_iter().enumerate() {
                match p {
                    Some(pending_lz77) => {
                        all_results.push(self.complete_find_matches_lazy(pending_lz77, chunk[i])?);
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
                "[pz-gpu] lz77_lazy_batched: {} blocks processed",
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
                    offset: (p & 0xFFFF) as u16,
                    length: (p >> 16) as u16,
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

    /// Submit LZ77 lazy matching work to a pre-allocated buffer slot.
    ///
    /// Writes input data, encodes 2 compute passes and a staging copy
    /// into one command buffer, and submits without blocking.
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

        let workgroups = (input_len as u32).div_ceil(64);
        let params = [input_len as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        self.queue
            .write_buffer(&slot.params_buf, 0, bytemuck::cast_slice(&params));

        // Pass 1: find_matches (near brute-force scan)
        let find_bg_layout = self.pipeline_lz77_lazy_find().get_bind_group_layout(0);
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

        // Pass 2: resolve_lazy
        let resolve_bg_layout = self.pipeline_lz77_lazy_resolve().get_bind_group_layout(0);
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
                    resource: slot.params_buf.as_entire_binding(),
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
                label: Some("slot_lz77_lazy"),
            });
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_lazy_find(),
            &find_bg,
            workgroups,
            "slot_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            self.pipeline_lz77_lazy_resolve(),
            &resolve_bg,
            workgroups,
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
    /// The caller must ensure `device.poll(Wait)` has been called after
    /// submitting to guarantee GPU work is complete.
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
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
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
