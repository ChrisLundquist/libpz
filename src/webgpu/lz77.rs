//! LZ77 GPU match finding kernels.

use super::*;

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

        // Hash counts and table buffers (pass 1)
        let hash_counts_buf = self.create_buffer_init(
            "tod_hash_counts",
            &vec![0u8; HASH_TABLE_SIZE * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let hash_table_buf = self.create_buffer(
            "tod_hash_table",
            (HASH_TABLE_SIZE * HASH_BUCKET_CAP * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );

        // Raw match output (pass 2 writes, pass 3 reads)
        let raw_match_buf = self.create_buffer(
            "tod_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Resolved match output (pass 3 writes) — this stays on GPU
        let resolved_buf = self.create_buffer(
            "tod_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // --- Pass 1 bind group: build_hash_table ---
        let build_bg_layout = self.pipeline_lz77_lazy_build.get_bind_group_layout(0);
        let build_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_tod_build_bg"),
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

        // --- Pass 2 bind group: find_matches ---
        let find_bg_layout = self.pipeline_lz77_lazy_find.get_bind_group_layout(0);
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hash_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hash_table_buf.as_entire_binding(),
                },
            ],
        });

        // --- Pass 3 bind group: resolve_lazy ---
        let resolve_bg_layout = self.pipeline_lz77_lazy_resolve.get_bind_group_layout(0);
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

        // Submit all 3 passes in a single command encoder
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
            &self.pipeline_lz77_lazy_build,
            &build_bg,
            workgroups,
            "lz77_tod_build",
        )?;
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_find,
            &find_bg,
            workgroups,
            "lz77_tod_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_resolve,
            &resolve_bg,
            workgroups,
            "lz77_tod_resolve",
        )?;
        self.queue.submit(Some(encoder.finish()));
        if let Some(t0) = t0 {
            self.device.poll(wgpu::Maintain::Wait);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lz77_to_device (build+find+resolve): {ms:.3} ms");
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

    /// Find LZ77 matches using the 3-pass lazy matching kernel.
    ///
    /// Pass 1: Build hash table (parallel hash insertion with atomics).
    /// Pass 2: Find best match per position (greedy, parallel).
    /// Pass 3: Lazy resolve -- demote positions where pos+1 has a longer match.
    fn find_matches_lazy(&self, input: &[u8]) -> PzResult<Vec<Match>> {
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

        // Hash counts and table buffers (pass 1)
        let hash_counts_buf = self.create_buffer_init(
            "lazy_hash_counts",
            &vec![0u8; HASH_TABLE_SIZE * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let hash_table_buf = self.create_buffer(
            "lazy_hash_table",
            (HASH_TABLE_SIZE * HASH_BUCKET_CAP * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );

        // Raw match output (pass 2 writes, pass 3 reads)
        let raw_match_buf = self.create_buffer(
            "lazy_raw_matches",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Resolved match output (pass 3 writes)
        let resolved_buf = self.create_buffer(
            "lazy_resolved",
            match_buf_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // --- Pass 1 bind group: build_hash_table ---
        let build_bg_layout = self.pipeline_lz77_lazy_build.get_bind_group_layout(0);
        let build_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_lazy_build_bg"),
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

        // --- Pass 2 bind group: find_matches ---
        let find_bg_layout = self.pipeline_lz77_lazy_find.get_bind_group_layout(0);
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hash_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hash_table_buf.as_entire_binding(),
                },
            ],
        });

        // --- Pass 3 bind group: resolve_lazy ---
        let resolve_bg_layout = self.pipeline_lz77_lazy_resolve.get_bind_group_layout(0);
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

        // Submit all 3 passes in a single command encoder
        let t0 = if self.profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lz77_lazy"),
            });
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_build,
            &build_bg,
            workgroups,
            "lz77_lazy_build",
        )?;
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_find,
            &find_bg,
            workgroups,
            "lz77_lazy_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_resolve,
            &resolve_bg,
            workgroups,
            "lz77_lazy_resolve",
        )?;
        self.queue.submit(Some(encoder.finish()));
        if let Some(t0) = t0 {
            self.device.poll(wgpu::Maintain::Wait);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lz77_lazy (build+find+resolve): {ms:.3} ms");
        }

        let raw = self.read_buffer(&resolved_buf, match_buf_size);
        let gpu_matches: Vec<GpuMatch> = bytemuck::cast_slice(&raw).to_vec();

        Ok(dedupe_gpu_matches(&gpu_matches, input))
    }

    /// Submit GPU LZ77 lazy matching work without blocking for results.
    ///
    /// Creates buffers, encodes 3 compute passes + staging copy in one
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

        let hash_counts_buf = self.create_buffer_init(
            "lazy_hash_counts",
            &vec![0u8; HASH_TABLE_SIZE * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let hash_table_buf = self.create_buffer(
            "lazy_hash_table",
            (HASH_TABLE_SIZE * HASH_BUCKET_CAP * 4) as u64,
            wgpu::BufferUsages::STORAGE,
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

        // Build bind groups (same as find_matches_lazy)
        let build_bg_layout = self.pipeline_lz77_lazy_build.get_bind_group_layout(0);
        let build_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lz77_lazy_build_bg"),
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

        let find_bg_layout = self.pipeline_lz77_lazy_find.get_bind_group_layout(0);
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: hash_counts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: hash_table_buf.as_entire_binding(),
                },
            ],
        });

        let resolve_bg_layout = self.pipeline_lz77_lazy_resolve.get_bind_group_layout(0);
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

        // Encode 3 compute passes + staging copy in one command buffer
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("lz77_lazy_submit"),
            });
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_build,
            &build_bg,
            workgroups,
            "lz77_lazy_build",
        )?;
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_find,
            &find_bg,
            workgroups,
            "lz77_lazy_find",
        )?;
        self.record_dispatch(
            &mut encoder,
            &self.pipeline_lz77_lazy_resolve,
            &resolve_bg,
            workgroups,
            "lz77_lazy_resolve",
        )?;
        // Copy resolved matches to staging buffer (in same command buffer)
        encoder.copy_buffer_to_buffer(&resolved_buf, 0, &staging_buf, 0, match_buf_size);

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
        self.device.poll(wgpu::Maintain::Wait);
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
        // Cap at GPU_PREFETCH_DEPTH to limit latency while hiding dispatch overhead.
        const GPU_PREFETCH_DEPTH: usize = 3;
        let block_size = blocks.first().map(|b| b.len()).unwrap_or(256 * 1024);
        let mem_limit = self.max_in_flight(&self.cost_lz77_lazy, block_size);
        let batch_size = mem_limit.min(GPU_PREFETCH_DEPTH);

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
            self.device.poll(wgpu::Maintain::Wait);

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
}
