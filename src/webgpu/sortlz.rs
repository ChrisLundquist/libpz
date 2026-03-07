//! GPU sort-based LZ77 match finding for SortLZ pipeline.
//!
//! Uses GPU radix sort to sort (hash, position) pairs, then GPU-parallel
//! adjacent-pair match verification. Fully deterministic, zero atomics in
//! the sort path, making it a true GPU-native LZ alternative.
//!
//! The radix sort is batched into a single command encoder submission
//! (4 passes × 4 phases = 16 dispatches, 1 submit), following the same
//! pattern as the BWT radix sort in `bwt.rs`.

use super::*;

impl WebGpuEngine {
    /// GPU-accelerated SortLZ match finding.
    ///
    /// Returns best match `(offset, length)` per position, or `None`.
    /// Uses GPU radix sort + adjacent-pair verification.
    pub fn sortlz_find_matches(
        &self,
        input: &[u8],
        config: &crate::sortlz::SortLzConfig,
    ) -> PzResult<Vec<Option<(u16, u16)>>> {
        let n = input.len();
        if n < 4 {
            return Ok(vec![None; n]);
        }

        let num_hashes = n.saturating_sub(3);
        let padded_n = num_hashes.next_power_of_two();
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let histogram_len = 256 * num_groups;

        // Step 1: Compute hashes on CPU (trivial, one pass).
        let hashes: Vec<u32> = (0..num_hashes)
            .map(|i| u32::from_le_bytes([input[i], input[i + 1], input[i + 2], input[i + 3]]))
            .collect();
        // Pad hashes to padded_n (pad with 0xFFFFFFFF so they sort to end).
        let mut hashes_padded = hashes.clone();
        hashes_padded.resize(padded_n, 0xFFFF_FFFF);

        // Step 2: Upload buffers.
        let input_padded = Self::pad_input_bytes(input);
        let input_buf =
            self.create_buffer_init("sortlz_input", &input_padded, wgpu::BufferUsages::STORAGE);

        let hashes_buf = self.create_buffer_init(
            "sortlz_hashes",
            bytemuck::cast_slice(&hashes_padded),
            wgpu::BufferUsages::STORAGE,
        );

        // Initialize SA: descending for stable sort tiebreaking.
        let sa_init: Vec<u32> = (0..padded_n as u32).rev().collect();
        let mut sa_buf = self.create_buffer_init(
            "sortlz_sa",
            bytemuck::cast_slice(&sa_init),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let mut sa_buf_alt = self.create_buffer(
            "sortlz_sa_alt",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let keys_buf = self.create_buffer(
            "sortlz_keys",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let histogram_buf = self.create_buffer(
            "sortlz_histogram",
            (histogram_len.max(1) * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // Step 3: 4-pass radix sort — batched into a single encoder.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sortlz_radix_sort"),
            });

        self.record_sortlz_radix_sort(
            &mut encoder,
            &mut sa_buf,
            &mut sa_buf_alt,
            &hashes_buf,
            &keys_buf,
            &histogram_buf,
            padded_n,
            num_hashes as u32,
        )?;

        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        // Step 4: Match verification (separate submit — needs sort results).
        let best_buf = self.create_buffer_init(
            "sortlz_best",
            &vec![0u8; n * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let vm_params = [
            n as u32,
            config.max_window as u32,
            config.max_candidates as u32,
            padded_n as u32,
        ];
        let vm_params_buf = self.create_buffer_init(
            "sortlz_vm_params",
            bytemuck::cast_slice(&vm_params),
            wgpu::BufferUsages::UNIFORM,
        );

        let vm_bg_layout = self
            .pipeline_sortlz_verify_matches()
            .get_bind_group_layout(0);
        let vm_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sortlz_verify_bg"),
            layout: &vm_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: hashes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: best_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: vm_params_buf.as_entire_binding(),
                },
            ],
        });

        let verify_wg = (padded_n as u32).div_ceil(256);
        self.dispatch(
            self.pipeline_sortlz_verify_matches(),
            &vm_bg,
            verify_wg,
            "sortlz_verify",
        )?;

        // Step 5: Read back results.
        let best_bytes = self.read_buffer(&best_buf, (n * 4) as u64);
        let best_u32: &[u32] = bytemuck::cast_slice(&best_bytes);

        let mut matches = Vec::with_capacity(n);
        for &packed in best_u32 {
            if packed == 0 {
                matches.push(None);
            } else {
                let length = (packed >> 16) as u16;
                let offset = (packed & 0xFFFF) as u16;
                if length >= config.min_match as u16 {
                    matches.push(Some((offset, length)));
                } else {
                    matches.push(None);
                }
            }
        }

        Ok(matches)
    }

    /// Record all 4 radix sort passes into a single command encoder.
    ///
    /// This batches 16 dispatches (4 passes × {key extraction, histogram,
    /// prefix sum, inclusive-to-exclusive, scatter}) into one GPU submission,
    /// eliminating per-pass CPU↔GPU synchronization overhead.
    #[allow(clippy::too_many_arguments)]
    fn record_sortlz_radix_sort(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        sa_buf: &mut wgpu::Buffer,
        sa_buf_alt: &mut wgpu::Buffer,
        hashes_buf: &wgpu::Buffer,
        keys_buf: &wgpu::Buffer,
        histogram_buf: &wgpu::Buffer,
        padded_n: usize,
        num_hashes: u32,
    ) -> PzResult<()> {
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let histogram_len = 256 * num_groups;

        for pass in 0u32..4 {
            // Phase 1: Key extraction.
            let key_params = [num_hashes, padded_n as u32, pass, 0u32];
            let key_params_buf = self.create_buffer_init(
                "sortlz_key_params",
                bytemuck::cast_slice(&key_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let key_bg_layout = self.pipeline_sortlz_compute_keys().get_bind_group_layout(0);
            let key_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sortlz_key_bg"),
                layout: &key_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: hashes_buf.as_entire_binding(),
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

            // Phase 2: Histogram — zero inline via encoder (no CPU↔GPU sync).
            encoder.clear_buffer(histogram_buf, 0, None);

            let hist_params = [padded_n as u32, num_groups as u32, 0u32, 0u32];
            let hist_params_buf = self.create_buffer_init(
                "sortlz_hist_params",
                bytemuck::cast_slice(&hist_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let hist_bg_layout = self.pipeline_radix_histogram().get_bind_group_layout(0);
            let hist_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sortlz_hist_bg"),
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

            // Dispatch key extraction + histogram.
            let global_wg = (padded_n as u32).div_ceil(256);
            self.record_dispatch(
                encoder,
                self.pipeline_sortlz_compute_keys(),
                &key_bg,
                global_wg,
                "sortlz_keys",
            )?;
            self.record_dispatch(
                encoder,
                self.pipeline_radix_histogram(),
                &hist_bg,
                num_groups as u32,
                "sortlz_histogram",
            )?;

            // Phase 3: Prefix sum + inclusive-to-exclusive (recorded, not submitted).
            let mut hist_scan_buf_temp = self.create_buffer(
                "sortlz_hist_scan_temp",
                (histogram_len.max(1) * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            self.record_inclusive_prefix_sum(
                encoder,
                histogram_buf,
                &mut hist_scan_buf_temp,
                histogram_len,
            )?;

            let ite_params = [histogram_len as u32, 0u32, 0u32, 0u32];
            let ite_params_buf = self.create_buffer_init(
                "sortlz_ite_params",
                bytemuck::cast_slice(&ite_params),
                wgpu::BufferUsages::UNIFORM,
            );
            let ite_bg_layout = self
                .pipeline_inclusive_to_exclusive()
                .get_bind_group_layout(0);
            let ite_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sortlz_ite_bg"),
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
            self.record_dispatch(
                encoder,
                self.pipeline_inclusive_to_exclusive(),
                &ite_bg,
                ite_wg,
                "sortlz_ite",
            )?;

            // Phase 4: Scatter.
            let scat_params = [padded_n as u32, num_groups as u32, 0u32, 0u32];
            let scat_params_buf = self.create_buffer_init(
                "sortlz_scat_params",
                bytemuck::cast_slice(&scat_params),
                wgpu::BufferUsages::UNIFORM,
            );
            let scat_bg_layout = self.pipeline_radix_scatter().get_bind_group_layout(0);
            let scat_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sortlz_scat_bg"),
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
            self.record_dispatch(
                encoder,
                self.pipeline_radix_scatter(),
                &scat_bg,
                num_groups as u32,
                "sortlz_scatter",
            )?;

            std::mem::swap(sa_buf, sa_buf_alt);
        }

        Ok(())
    }

    /// GPU-accelerated SortLZ compression.
    pub fn sortlz_compress(
        &self,
        input: &[u8],
        config: &crate::sortlz::SortLzConfig,
    ) -> PzResult<Vec<u8>> {
        let matches = self.sortlz_find_matches(input, config)?;
        crate::sortlz::compress_with_matches(input, matches, config)
    }
}
