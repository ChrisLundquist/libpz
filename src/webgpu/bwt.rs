//! BWT GPU suffix array construction.

use super::*;

impl WebGpuEngine {
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

            // Phases 1+2 batched: compute keys -> histogram (single submit)
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
}
