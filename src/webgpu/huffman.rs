//! GPU Huffman encoding: byte histogram and Huffman coding via GPU prefix sum.

use super::*;

impl WebGpuEngine {
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

    /// Compute a byte histogram from data already on the GPU device.
    ///
    /// Same as [`byte_histogram()`] but avoids uploading the input — the data
    /// is already in a [`DeviceBuf`]. This saves one host→device transfer
    /// when chaining GPU stages.
    pub fn byte_histogram_on_device(&self, input: &DeviceBuf) -> PzResult<[u32; 256]> {
        if input.is_empty() {
            return Ok([0u32; 256]);
        }

        let n = input.len();

        let hist_buf = self.create_buffer_init(
            "hist_od_buf",
            &vec![0u8; 256 * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let workgroups = (n as u32).div_ceil(64);
        let params = [n as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "hist_od_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_byte_histogram.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hist_od_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buf.as_entire_binding(),
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
            "byte_histogram_on_device",
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

        // Compute last bit length on the CPU (avoids GPU->CPU readback sync).
        // The kernel writes: bit_lengths[i] = code_lut[input[i]] >> 24
        let last_bit_length = code_lut[input[n - 1] as usize] >> 24;

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

    /// Encode data already on the GPU using Huffman coding with GPU prefix sum.
    ///
    /// Same as [`huffman_encode_gpu_scan()`] but the input is a [`DeviceBuf`]
    /// that's already resident on the GPU — no host→device upload for the
    /// symbol data. This saves one PCI transfer per call when chaining stages.
    pub fn huffman_encode_on_device(
        &self,
        input: &DeviceBuf,
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();

        let lut_buf = self.create_buffer_init(
            "huff_od_lut",
            bytemuck::cast_slice(code_lut),
            wgpu::BufferUsages::STORAGE,
        );

        let bit_lengths_buf = self.create_buffer(
            "huff_od_bit_lengths",
            (n * 4) as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        let workgroups = (n as u32).div_ceil(64);
        let params = [n as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "huff_od_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 1: compute bit lengths (input buffer is on-device)
        let bg1_layout = self.pipeline_compute_bit_lengths.get_bind_group_layout(0);
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_od_pass1_bg"),
            layout: &bg1_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buf.as_entire_binding(),
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
            "huff_od_pass1",
        )?;

        // We need the last input byte to compute last_bit_length on the CPU.
        // Read the u32 word containing the last byte from the device buffer.
        let last_byte_word_index = (n - 1) / 4;
        let last_byte_word = self.read_buffer_scalar_u32(&input.buf, last_byte_word_index);
        let last_byte_in_word = (n - 1) % 4;
        let last_byte = ((last_byte_word >> (last_byte_in_word * 8)) & 0xFF) as u8;
        let last_bit_length = code_lut[last_byte as usize] >> 24;

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
            "huff_od_output",
            &vec![0u8; output_uints * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Pass 2: write codes (bit_lengths_buf now contains offsets, input is on-device)
        let bg2_layout = self.pipeline_write_codes.get_bind_group_layout(0);
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_od_pass2_bg"),
            layout: &bg2_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buf.as_entire_binding(),
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
            "huff_od_pass2",
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
}
