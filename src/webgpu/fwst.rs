//! GPU fixed-window sort transform for Experiment F.
//!
//! Reuses existing GPU radix sort infrastructure but caps the sort depth
//! at `w` bytes instead of running until convergence (full suffix sort).

use super::*;

impl WebGpuEngine {
    /// GPU-accelerated fixed-window sort transform.
    ///
    /// Like BWT, but sorts by only the first `w` bytes of context at each position.
    /// Uses the existing GPU radix sort, running exactly `w` passes instead of
    /// iterating until convergence.
    pub fn fwst_encode(&self, input: &[u8], window: usize) -> PzResult<crate::fwst::FwstResult> {
        let n = input.len();
        if n == 0 {
            return Err(PzError::InvalidInput);
        }

        let w = window.min(n);

        // If window >= input length, use full BWT (equivalent).
        if w >= n {
            let bwt_result = self.bwt_encode(input)?;
            return Ok(crate::fwst::FwstResult {
                data: bwt_result.data,
                positions: Vec::new(), // empty = use BWT inverse
                primary_index: bwt_result.primary_index,
            });
        }

        // For inputs below GPU threshold, fall back to CPU.
        if n < MIN_GPU_BWT_SIZE {
            return crate::fwst::encode(input, &crate::fwst::FwstConfig { window })
                .ok_or(PzError::InvalidInput);
        }

        // GPU path: use radix sort with exactly `w` passes.
        // The key at each position i is input[i..i+w] (wrapping at end).
        let sa = self.fwst_build_sorted_positions(input, w)?;

        // BWT readoff: last character of each rotation.
        let mut result = Vec::with_capacity(n);
        let mut primary_index = 0u32;

        for (i, &pos) in sa.iter().enumerate() {
            if pos == 0 {
                primary_index = i as u32;
                result.push(input[n - 1]);
            } else {
                result.push(input[pos - 1]);
            }
        }

        Ok(crate::fwst::FwstResult {
            data: result,
            positions: sa.iter().map(|&p| p as u32).collect(),
            primary_index,
        })
    }

    /// Build sorted position array using GPU radix sort with fixed window depth.
    ///
    /// Performs exactly `w` passes of 8-bit radix sort (one per byte of the key),
    /// processing from the last byte of the key to the first (LSB-first).
    /// Uses the FWST-specific key extraction kernel with the existing BWT radix
    /// sort infrastructure (histogram, prefix sum, scatter).
    fn fwst_build_sorted_positions(&self, input: &[u8], w: usize) -> PzResult<Vec<usize>> {
        let n = input.len();
        let padded_n = n.next_power_of_two();
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let histogram_len = 256 * num_groups;

        // Upload input data as packed u32s.
        let input_padded = Self::pad_input_bytes(input);
        let input_buf =
            self.create_buffer_init("fwst_input", &input_padded, wgpu::BufferUsages::STORAGE);

        // Initialize SA: descending order for stable sort tiebreaking.
        let sa_init: Vec<u32> = (0..padded_n as u32).rev().collect();
        let mut sa_buf = self.create_buffer_init(
            "fwst_sa",
            bytemuck::cast_slice(&sa_init),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let mut sa_buf_alt = self.create_buffer(
            "fwst_sa_alt",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Allocate keys and histogram buffers (reused across passes).
        let keys_buf = self.create_buffer(
            "fwst_keys",
            (padded_n * 4) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let histogram_buf = self.create_buffer(
            "fwst_histogram",
            (histogram_len.max(1) * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        // LSB-first: sort by byte w-1 first, then w-2, ..., then 0.
        for pass in (0..w).rev() {
            // Phase 1: FWST key extraction — key = input[(sa[i] + pass) % n].
            let key_params = [n as u32, padded_n as u32, pass as u32, 0u32];
            let key_params_buf = self.create_buffer_init(
                "fwst_key_params",
                bytemuck::cast_slice(&key_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let key_bg_layout = self.pipeline_fwst_compute_keys().get_bind_group_layout(0);
            let key_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fwst_key_bg"),
                layout: &key_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sa_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buf.as_entire_binding(),
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

            // Phases 1+2 batched: compute keys → histogram.
            let hist_params = [padded_n as u32, num_groups as u32, 0u32, 0u32];
            let hist_params_buf = self.create_buffer_init(
                "fwst_hist_params",
                bytemuck::cast_slice(&hist_params),
                wgpu::BufferUsages::UNIFORM,
            );

            // Zero the histogram buffer.
            let hist_zeros = vec![0u32; histogram_len];
            self.queue
                .write_buffer(&histogram_buf, 0, bytemuck::cast_slice(&hist_zeros));

            let hist_bg_layout = self.pipeline_radix_histogram().get_bind_group_layout(0);
            let hist_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fwst_hist_bg"),
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
                    label: Some("fwst_keys_hist"),
                });
            self.record_dispatch(
                &mut encoder,
                self.pipeline_fwst_compute_keys(),
                &key_bg,
                global_wg,
                "fwst_keys",
            )?;
            self.record_dispatch(
                &mut encoder,
                self.pipeline_radix_histogram(),
                &hist_bg,
                num_groups as u32,
                "fwst_histogram",
            )?;
            self.profiler_resolve(&mut encoder);
            self.queue.submit(Some(encoder.finish()));
            if let Some(t0) = t0 {
                self.poll_wait();
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] fwst_keys+hist (pass={pass}): {ms:.3} ms");
            }

            // Phase 3: Prefix sum over histogram, then inclusive_to_exclusive.
            let mut hist_scan_buf_temp = self.create_buffer(
                "fwst_hist_scan_temp",
                (histogram_len.max(1) * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            self.run_inclusive_prefix_sum(&histogram_buf, &mut hist_scan_buf_temp, histogram_len)?;

            let ite_params = [histogram_len as u32, 0u32, 0u32, 0u32];
            let ite_params_buf = self.create_buffer_init(
                "fwst_ite_params",
                bytemuck::cast_slice(&ite_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let ite_bg_layout = self
                .pipeline_inclusive_to_exclusive()
                .get_bind_group_layout(0);
            let ite_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fwst_ite_bg"),
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
                self.pipeline_inclusive_to_exclusive(),
                &ite_bg,
                ite_wg,
                "fwst_ite",
            )?;

            // Phase 4: Scatter.
            let scat_params = [padded_n as u32, num_groups as u32, 0u32, 0u32];
            let scat_params_buf = self.create_buffer_init(
                "fwst_scat_params",
                bytemuck::cast_slice(&scat_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let scat_bg_layout = self.pipeline_radix_scatter().get_bind_group_layout(0);
            let scat_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fwst_scat_bg"),
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
                self.pipeline_radix_scatter(),
                &scat_bg,
                num_groups as u32,
                "fwst_scatter",
            )?;

            // Swap SA buffers.
            std::mem::swap(&mut sa_buf, &mut sa_buf_alt);
        }

        // Read back sorted SA and filter out padding entries.
        let sa_bytes = self.read_buffer(&sa_buf, (padded_n * 4) as u64);
        let sa_u32: &[u32] = bytemuck::cast_slice(&sa_bytes);

        let result: Vec<usize> = sa_u32
            .iter()
            .filter(|&&v| (v as usize) < n)
            .map(|&v| v as usize)
            .collect();

        if result.len() != n {
            return Err(PzError::InvalidInput);
        }

        Ok(result)
    }

    /// GPU-accelerated FWST compression.
    pub fn fwst_compress(
        &self,
        input: &[u8],
        config: &crate::fwst::FwstConfig,
    ) -> PzResult<Vec<u8>> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        // Use the shared compress function which handles wire format correctly.
        let fwst_result = self.fwst_encode(input, config.window)?;
        crate::fwst::compress_from_result(&fwst_result, config)
    }
}
