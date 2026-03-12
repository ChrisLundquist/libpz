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

        let bg_layout = self.pipeline_byte_histogram().get_bind_group_layout(0);
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
            self.pipeline_byte_histogram(),
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

        let bg_layout = self.pipeline_byte_histogram().get_bind_group_layout(0);
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
            self.pipeline_byte_histogram(),
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
        let bg1_layout = self.pipeline_compute_bit_lengths().get_bind_group_layout(0);
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
            self.pipeline_compute_bit_lengths(),
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

        // Pass 2: write codes (now chunk-based, one thread per output u32 word)
        let wc_workgroups = (output_uints as u32).div_ceil(64);
        let wc_params = [
            n as u32,
            total_bits as u32,
            0,
            self.dispatch_width(wc_workgroups, 64),
        ];
        let wc_params_buf = self.create_buffer_init(
            "huff_wc_params",
            bytemuck::cast_slice(&wc_params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg2_layout = self.pipeline_write_codes().get_bind_group_layout(0);
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
                    resource: wc_params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            self.pipeline_write_codes(),
            &bg2,
            wc_workgroups,
            "huff_pass2",
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

    /// Encode data using Huffman coding with GPU compute + CPU prefix sum.
    /// Pass 1 (ComputeBitLengths) and Pass 2 (WriteCodes) run on GPU;
    /// the prefix sum between them runs on CPU to avoid recursive kernel overhead.
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
        let bg1_layout = self.pipeline_compute_bit_lengths().get_bind_group_layout(0);
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
            self.pipeline_compute_bit_lengths(),
            &bg1,
            workgroups,
            "huff_gs_pass1",
        )?;

        // Download bit_lengths and compute prefix sum on CPU.
        // This is faster than the GPU Blelloch scan because the recursive
        // multi-level decomposition generates many kernel launches whose
        // overhead dominates the actual O(n) sequential scan.
        let raw_lengths = self.read_buffer(&bit_lengths_buf, (n * 4) as u64);
        let bit_lengths: &[u32] = bytemuck::cast_slice(&raw_lengths);

        let mut bit_offsets = vec![0u32; n];
        let mut running_sum: u64 = 0;
        for i in 0..n {
            bit_offsets[i] = running_sum as u32;
            running_sum += bit_lengths[i] as u64;
        }
        let total_bits = running_sum as usize;

        // Upload bit_offsets back to GPU (overwrite bit_lengths_buf)
        self.queue
            .write_buffer(&bit_lengths_buf, 0, bytemuck::cast_slice(&bit_offsets));

        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let output_buf = self.create_buffer_init(
            "huff_gs_output",
            &vec![0u8; output_uints * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Pass 2: write codes (bit_lengths_buf now contains offsets, one thread per output u32 word)
        let wc_workgroups = (output_uints as u32).div_ceil(64);
        let wc_params = [
            n as u32,
            total_bits as u32,
            0,
            self.dispatch_width(wc_workgroups, 64),
        ];
        let wc_params_buf = self.create_buffer_init(
            "huff_gs_wc_params",
            bytemuck::cast_slice(&wc_params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg2_layout = self.pipeline_write_codes().get_bind_group_layout(0);
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
                    resource: wc_params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            self.pipeline_write_codes(),
            &bg2,
            wc_workgroups,
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
        let bg1_layout = self.pipeline_compute_bit_lengths().get_bind_group_layout(0);
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
            self.pipeline_compute_bit_lengths(),
            &bg1,
            workgroups,
            "huff_od_pass1",
        )?;

        // Download bit_lengths and compute prefix sum on CPU.
        // This is faster than the GPU Blelloch scan because the recursive
        // multi-level decomposition generates many kernel launches whose
        // overhead dominates the actual O(n) sequential scan.
        let raw_lengths = self.read_buffer(&bit_lengths_buf, (n * 4) as u64);
        let bit_lengths: &[u32] = bytemuck::cast_slice(&raw_lengths);

        let mut bit_offsets = vec![0u32; n];
        let mut running_sum: u64 = 0;
        for i in 0..n {
            bit_offsets[i] = running_sum as u32;
            running_sum += bit_lengths[i] as u64;
        }
        let total_bits = running_sum as usize;

        // Upload bit_offsets back to GPU (overwrite bit_lengths_buf)
        self.queue
            .write_buffer(&bit_lengths_buf, 0, bytemuck::cast_slice(&bit_offsets));

        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let output_buf = self.create_buffer_init(
            "huff_od_output",
            &vec![0u8; output_uints * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Pass 2: write codes (bit_lengths_buf now contains offsets, input is on-device, one thread per output u32 word)
        let wc_workgroups = (output_uints as u32).div_ceil(64);
        let wc_params = [
            n as u32,
            total_bits as u32,
            0,
            self.dispatch_width(wc_workgroups, 64),
        ];
        let wc_params_buf = self.create_buffer_init(
            "huff_od_wc_params",
            bytemuck::cast_slice(&wc_params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg2_layout = self.pipeline_write_codes().get_bind_group_layout(0);
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
                    resource: wc_params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            self.pipeline_write_codes(),
            &bg2,
            wc_workgroups,
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

    /// Perform an exclusive prefix sum on a GPU buffer in-place.
    ///
    /// Uses Blelloch scan (work-efficient parallel prefix sum) with
    /// multi-level reduction for large arrays. The WGSL kernel uses
    /// workgroup size 256, processing 512 elements per block.
    pub fn prefix_sum_gpu(&self, buf: &wgpu::Buffer, n: usize) -> PzResult<()> {
        if n == 0 {
            return Ok(());
        }

        const LOCAL_SIZE: usize = 256;
        const BLOCK_SIZE: usize = LOCAL_SIZE * 2; // 512

        if n <= BLOCK_SIZE {
            // Single block: no need for multi-level scan
            // Create a dummy block_sums buffer (kernel writes to binding 1)
            let block_sums_buf =
                self.create_buffer("ps_dummy_sums", 4, wgpu::BufferUsages::STORAGE);
            let params = [n as u32, 0u32, 0u32, 0u32];
            let params_buf = self.create_buffer_init(
                "ps_params",
                bytemuck::cast_slice(&params),
                wgpu::BufferUsages::UNIFORM,
            );

            let bg_layout = self.pipeline_prefix_sum_block().get_bind_group_layout(0);
            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ps_block_bg"),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buf.as_entire_binding(),
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
                self.pipeline_prefix_sum_block(),
                &bg,
                1,
                "prefix_sum_single",
            )?;
            self.poll_wait();
            return Ok(());
        }

        // Multi-level: split into blocks, scan each, collect block totals
        let num_blocks = n.div_ceil(BLOCK_SIZE);

        let block_sums_buf = self.create_buffer(
            "ps_block_sums",
            (num_blocks * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Level 1: scan each block, output block totals
        let params = [n as u32, 0u32, 0u32, 0u32];
        let params_buf = self.create_buffer_init(
            "ps_l1_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_prefix_sum_block().get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ps_l1_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
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
            self.pipeline_prefix_sum_block(),
            &bg,
            num_blocks as u32,
            "prefix_sum_l1",
        )?;
        self.poll_wait();

        // Level 2: recursively scan block totals
        if num_blocks > 1 {
            self.prefix_sum_gpu(&block_sums_buf, num_blocks)?;
        }

        // Level 3: apply block offsets to elements
        let apply_params = [n as u32, BLOCK_SIZE as u32, 0u32, 0u32];
        let apply_params_buf = self.create_buffer_init(
            "ps_apply_params",
            bytemuck::cast_slice(&apply_params),
            wgpu::BufferUsages::UNIFORM,
        );

        let apply_layout = self.pipeline_prefix_sum_apply().get_bind_group_layout(0);
        let apply_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ps_apply_bg"),
            layout: &apply_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
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

        let apply_workgroups = (n as u32).div_ceil(LOCAL_SIZE as u32);
        self.dispatch(
            self.pipeline_prefix_sum_apply(),
            &apply_bg,
            apply_workgroups,
            "prefix_sum_apply",
        )?;
        self.poll_wait();

        Ok(())
    }

    /// Encode data already on the GPU using Huffman coding with a fully
    /// on-device prefix sum (no CPU round-trip for the scan).
    ///
    /// Chains compute_bit_lengths → prefix_sum_gpu → write_codes entirely
    /// on the GPU. This is the zero-round-trip variant for kernel fusion.
    pub fn huffman_encode_fully_on_device(
        &self,
        input: &DeviceBuf,
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();

        let lut_buf = self.create_buffer_init(
            "huff_fod_lut",
            bytemuck::cast_slice(code_lut),
            wgpu::BufferUsages::STORAGE,
        );

        let bit_lengths_buf = self.create_buffer(
            "huff_fod_bit_lengths",
            (n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let workgroups = (n as u32).div_ceil(64);
        let params = [n as u32, 0, 0, self.dispatch_width(workgroups, 64)];
        let params_buf = self.create_buffer_init(
            "huff_fod_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Pass 1: compute bit lengths (on-device input)
        let bg1_layout = self.pipeline_compute_bit_lengths().get_bind_group_layout(0);
        let bg1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_fod_pass1_bg"),
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
            self.pipeline_compute_bit_lengths(),
            &bg1,
            workgroups,
            "huff_fod_pass1",
        )?;
        self.poll_wait();

        // Pass 2: GPU prefix sum on bit_lengths → bit_offsets (in-place)
        // First, read the last element to compute total_bits
        let last_length = self.read_buffer_scalar_u32(&bit_lengths_buf, n - 1);

        // Run exclusive prefix sum (converts bit_lengths → bit_offsets in-place)
        self.prefix_sum_gpu(&bit_lengths_buf, n)?;

        // Total bits = last_offset + last_length
        let last_offset = self.read_buffer_scalar_u32(&bit_lengths_buf, n - 1);
        let total_bits = (last_offset as usize) + (last_length as usize);

        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let output_buf = self.create_buffer_init(
            "huff_fod_output",
            &vec![0u8; output_uints * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Pass 3: write codes (bit_lengths_buf now contains offsets, one thread per output u32 word)
        let wc_workgroups = (output_uints as u32).div_ceil(64);
        let wc_params = [
            n as u32,
            total_bits as u32,
            0,
            self.dispatch_width(wc_workgroups, 64),
        ];
        let wc_params_buf = self.create_buffer_init(
            "huff_fod_wc_params",
            bytemuck::cast_slice(&wc_params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg2_layout = self.pipeline_write_codes().get_bind_group_layout(0);
        let bg2 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_fod_pass2_bg"),
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
                    resource: wc_params_buf.as_entire_binding(),
                },
            ],
        });

        self.dispatch(
            self.pipeline_write_codes(),
            &bg2,
            wc_workgroups,
            "huff_fod_pass2",
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

    /// Decode a Huffman-encoded bitstream on the GPU using sync-point parallel decode.
    ///
    /// Each sync-point segment is decoded by an independent GPU thread using a
    /// 12-bit lookup table. This is the GPU equivalent of [`HuffmanTree::decode_tiled()`].
    ///
    /// # Arguments
    /// * `huffman_data` — MSB-first packed bitstream bytes (as produced by `encode_with_sync_points`)
    /// * `total_bits` — number of valid bits in the bitstream
    /// * `decode_lut` — 4096-entry decode table: `(symbol << 8) | code_bits`
    /// * `sync_points` — sync-point array including sentinel at the end
    /// * `output_len` — expected number of decoded symbols (bytes)
    pub fn huffman_decode_gpu(
        &self,
        huffman_data: &[u8],
        _total_bits: usize,
        decode_lut: &[u32; 4096],
        sync_points: &[crate::huffman::SyncPoint],
        output_len: usize,
    ) -> PzResult<Vec<u8>> {
        if output_len == 0 || sync_points.len() < 2 {
            return Ok(Vec::new());
        }

        let num_segments = sync_points.len() - 1; // last entry is sentinel

        // Upload bitstream as big-endian u32 words.
        // The CPU encode produces MSB-first byte order; WGSL reads u32 words,
        // so we must convert bytes to big-endian u32 to preserve bit layout.
        // Pad by at least 4 bytes so peek_12_msb can safely read word_idx + 1.
        let padded_len = huffman_data.len().div_ceil(4) + 1; // +1 word padding
        let mut bitstream_words = vec![0u32; padded_len];
        for (i, chunk) in huffman_data.chunks(4).enumerate() {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            bitstream_words[i] = u32::from_be_bytes(bytes);
        }

        let bitstream_buf = self.create_buffer_init(
            "huff_dec_bitstream",
            bytemuck::cast_slice(&bitstream_words),
            wgpu::BufferUsages::STORAGE,
        );

        // Upload decode LUT (4096 × u32 = 16KB).
        let lut_buf = self.create_buffer_init(
            "huff_dec_lut",
            bytemuck::cast_slice(decode_lut),
            wgpu::BufferUsages::STORAGE,
        );

        // Upload sync points as flat u32 pairs: [bit_offset, symbol_index, ...].
        let sp_flat: Vec<u32> = sync_points
            .iter()
            .flat_map(|sp| [sp.bit_offset, sp.symbol_index])
            .collect();
        let sp_buf = self.create_buffer_init(
            "huff_dec_sync_points",
            bytemuck::cast_slice(&sp_flat),
            wgpu::BufferUsages::STORAGE,
        );

        // Output buffer: zero-initialized, padded to u32 alignment.
        let output_words = output_len.div_ceil(4);
        let output_buf = self.create_buffer_init(
            "huff_dec_output",
            &vec![0u8; output_words * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Params: num_segments, output_len, 0, dispatch_width
        let workgroups = (num_segments as u32).div_ceil(64);
        let params = [
            num_segments as u32,
            output_len as u32,
            0u32,
            self.dispatch_width(workgroups, 64),
        ];
        let params_buf = self.create_buffer_init(
            "huff_dec_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Create bind group.
        let bg_layout = self.pipeline_huffman_decode().get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("huff_dec_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bitstream_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lut_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sp_buf.as_entire_binding(),
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

        // Dispatch.
        self.dispatch(
            self.pipeline_huffman_decode(),
            &bg,
            workgroups,
            "huffman_sync_decode",
        )?;

        // Read back output and truncate to output_len.
        let raw = self.read_buffer(&output_buf, (output_words * 4) as u64);
        Ok(raw[..output_len].to_vec())
    }

    /// Decode multiple Huffman-encoded streams in a single GPU submission.
    ///
    /// This batches all stream decodes into one command encoder → one submit → one
    /// readback, amortizing per-dispatch overhead (~0.8ms) that dominates at small sizes.
    /// Used by [`crate::gpulz::decompress_block_gpu`] to decode all 6 LzSeq streams.
    pub fn huffman_decode_gpu_batched(
        &self,
        streams: &[HuffmanDecodeStream],
    ) -> PzResult<Vec<Vec<u8>>> {
        self.huffman_decode_gpu_batched_timed(streams)
            .map(|(data, _)| data)
    }

    /// Same as [`huffman_decode_gpu_batched`] but returns per-phase timing.
    #[allow(clippy::type_complexity)]
    pub fn huffman_decode_gpu_batched_timed(
        &self,
        streams: &[HuffmanDecodeStream],
    ) -> PzResult<(Vec<Vec<u8>>, GpuBatchedTimings)> {
        use std::time::Instant;
        let t0 = Instant::now();

        if streams.is_empty() {
            return Ok((
                Vec::new(),
                GpuBatchedTimings {
                    buffer_create_us: 0,
                    submit_us: 0,
                    readback_us: 0,
                    total_us: 0,
                },
            ));
        }

        let pipeline = self.pipeline_huffman_decode();
        let bg_layout = pipeline.get_bind_group_layout(0);

        // Build all buffers and bind groups upfront.
        struct StreamGpu {
            bg: wgpu::BindGroup,
            workgroups: u32,
            output_buf: wgpu::Buffer,
            output_words: usize,
            output_len: usize,
        }

        let mut stream_gpus = Vec::with_capacity(streams.len());

        for (i, s) in streams.iter().enumerate() {
            if s.output_len == 0 || s.sync_points.len() < 2 {
                stream_gpus.push(None);
                continue;
            }

            let num_segments = s.sync_points.len() - 1;

            // Bitstream → big-endian u32 words.
            let padded_len = s.huffman_data.len().div_ceil(4) + 1;
            let mut bitstream_words = vec![0u32; padded_len];
            for (j, chunk) in s.huffman_data.chunks(4).enumerate() {
                let mut bytes = [0u8; 4];
                bytes[..chunk.len()].copy_from_slice(chunk);
                bitstream_words[j] = u32::from_be_bytes(bytes);
            }
            let bitstream_buf = self.create_buffer_init(
                &format!("huff_dec_bs_{i}"),
                bytemuck::cast_slice(&bitstream_words),
                wgpu::BufferUsages::STORAGE,
            );

            let lut_buf = self.create_buffer_init(
                &format!("huff_dec_lut_{i}"),
                bytemuck::cast_slice(s.decode_lut.as_ref()),
                wgpu::BufferUsages::STORAGE,
            );

            let sp_flat: Vec<u32> = s
                .sync_points
                .iter()
                .flat_map(|sp| [sp.bit_offset, sp.symbol_index])
                .collect();
            let sp_buf = self.create_buffer_init(
                &format!("huff_dec_sp_{i}"),
                bytemuck::cast_slice(&sp_flat),
                wgpu::BufferUsages::STORAGE,
            );

            let output_words = s.output_len.div_ceil(4);
            let output_buf = self.create_buffer_init(
                &format!("huff_dec_out_{i}"),
                &vec![0u8; output_words * 4],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );

            let workgroups = (num_segments as u32).div_ceil(64);
            let params = [
                num_segments as u32,
                s.output_len as u32,
                0u32,
                self.dispatch_width(workgroups, 64),
            ];
            let params_buf = self.create_buffer_init(
                &format!("huff_dec_params_{i}"),
                bytemuck::cast_slice(&params),
                wgpu::BufferUsages::UNIFORM,
            );

            let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("huff_dec_bg_{i}")),
                layout: &bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bitstream_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lut_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: sp_buf.as_entire_binding(),
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

            stream_gpus.push(Some(StreamGpu {
                bg,
                workgroups,
                output_buf,
                output_words,
                output_len: s.output_len,
            }));
        }

        let t_buffers = Instant::now();
        let buffer_create_us = t_buffers.duration_since(t0).as_micros() as u64;

        // Record all dispatches into a single command encoder.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("huff_dec_batched"),
            });

        for sg in stream_gpus.iter().flatten() {
            self.record_dispatch(
                &mut encoder,
                pipeline,
                &sg.bg,
                sg.workgroups,
                "huffman_sync_decode",
            )?;
        }

        // Copy all outputs to staging buffers in the same encoder.
        let mut staging_bufs = Vec::with_capacity(streams.len());
        for sg in &stream_gpus {
            if let Some(sg) = sg {
                let staging = self.create_buffer(
                    "staging",
                    (sg.output_words * 4) as u64,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                );
                encoder.copy_buffer_to_buffer(
                    &sg.output_buf,
                    0,
                    &staging,
                    0,
                    (sg.output_words * 4) as u64,
                );
                staging_bufs.push(Some(staging));
            } else {
                staging_bufs.push(None);
            }
        }

        self.profiler_resolve(&mut encoder);
        self.queue.submit(Some(encoder.finish()));

        let t_submit = Instant::now();
        let submit_us = t_submit.duration_since(t_buffers).as_micros() as u64;

        // Map all staging buffers and read back.
        let mut results = Vec::with_capacity(streams.len());
        for (i, staging) in staging_bufs.iter().enumerate() {
            if let Some(staging) = staging {
                let slice = staging.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |r| {
                    tx.send(r).unwrap();
                });
                self.poll_wait();
                rx.recv().unwrap().map_err(|_| PzError::Unsupported)?;
                let data = slice.get_mapped_range().to_vec();
                staging.unmap();
                let output_len = stream_gpus[i].as_ref().unwrap().output_len;
                results.push(data[..output_len].to_vec());
            } else {
                results.push(Vec::new());
            }
        }

        let t_readback = Instant::now();
        let readback_us = t_readback.duration_since(t_submit).as_micros() as u64;
        let total_us = t_readback.duration_since(t0).as_micros() as u64;

        Ok((
            results,
            GpuBatchedTimings {
                buffer_create_us,
                submit_us,
                readback_us,
                total_us,
            },
        ))
    }
}

/// Per-phase timing for batched GPU Huffman decode.
#[derive(Debug, Clone)]
pub struct GpuBatchedTimings {
    /// Buffer creation + bind group setup (µs).
    pub buffer_create_us: u64,
    /// Command record + submit (µs).
    pub submit_us: u64,
    /// Poll + readback (µs).
    pub readback_us: u64,
    /// Total wall time (µs).
    pub total_us: u64,
}

/// Descriptor for one Huffman stream to decode on the GPU.
pub struct HuffmanDecodeStream<'a> {
    /// MSB-first packed bitstream bytes.
    pub huffman_data: &'a [u8],
    /// 4096-entry decode LUT: `(symbol << 8) | code_bits`.
    pub decode_lut: Box<[u32; 4096]>,
    /// Sync-point array including sentinel.
    pub sync_points: Vec<crate::huffman::SyncPoint>,
    /// Expected number of decoded symbols.
    pub output_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to get or skip tests if no GPU available.
    fn get_engine() -> Option<WebGpuEngine> {
        WebGpuEngine::new().ok()
    }

    #[test]
    fn test_huffman_gpu_encode_chunked_matches_cpu() {
        // Verify that write_codes_chunked produces output that round-trips correctly.
        let engine = match get_engine() {
            Some(e) => e,
            None => return, // skip if no GPU available
        };

        for input_size in [1024usize, 32768, 131072] {
            let input: Vec<u8> = (0..input_size)
                .map(|i| (i % 128) as u8) // 128 distinct symbols
                .collect();

            let cpu_tree = crate::huffman::HuffmanTree::from_data(&input)
                .expect("Huffman tree build must succeed");
            let (cpu_encoded, cpu_bits) = cpu_tree
                .encode(&input)
                .expect("CPU Huffman encode must succeed");

            // GPU encode via gpu_scan (uses write_codes_chunked)
            let (gpu_encoded_bytes, gpu_bits) = engine
                .huffman_encode_gpu_scan(&input, &cpu_tree.code_lut())
                .expect("GPU Huffman encode must succeed");

            // Both should decode to the original input
            let cpu_decoded = cpu_tree
                .decode(&cpu_encoded, cpu_bits)
                .unwrap_or_else(|e| panic!("CPU decode failed for size {}: {:?}", input_size, e));
            let gpu_decoded = cpu_tree
                .decode(&gpu_encoded_bytes, gpu_bits)
                .unwrap_or_else(|e| {
                    panic!(
                        "CPU decode of GPU-encoded failed for size {}: {:?}",
                        input_size, e
                    )
                });

            assert_eq!(
                cpu_decoded, input,
                "CPU Huffman round-trip failed for size {}",
                input_size
            );
            assert_eq!(
                gpu_decoded, input,
                "GPU Huffman chunked round-trip failed for size {}",
                input_size
            );
        }
    }

    #[test]
    fn test_huffman_gpu_encode_chunked_boundary_symbols() {
        // Deliberately construct input where many codewords straddle u32
        // boundaries to stress the boundary-symbol atomic path.
        let engine = match get_engine() {
            Some(e) => e,
            None => return,
        };

        // Input with 3 distinct symbols: each gets a ~10-bit code (long codes
        // mean more boundary crossings per 32-bit word).
        let input: Vec<u8> = (0..65536)
            .map(|i| match i % 3 {
                0 => 0u8,
                1 => 1,
                _ => 2,
            })
            .collect();

        let tree = crate::huffman::HuffmanTree::from_data(&input)
            .expect("Huffman tree build must succeed");

        let (gpu_encoded_bytes, gpu_bits) = engine
            .huffman_encode_gpu_scan(&input, &tree.code_lut())
            .expect("GPU Huffman encode must succeed for boundary-stress input");

        let decoded = tree
            .decode(&gpu_encoded_bytes, gpu_bits)
            .expect("decode of boundary-stress GPU output must succeed");
        assert_eq!(
            decoded, input,
            "boundary-symbol stress test round-trip mismatch"
        );
    }

    #[test]
    fn test_huffman_gpu_encode_chunked_single_symbol() {
        // Edge case: all bytes are the same symbol. Huffman assigns a 1-bit code.
        // All writes land in distinct u32 words with no boundary crossings.
        let engine = match get_engine() {
            Some(e) => e,
            None => return,
        };

        let input = vec![42u8; 8192];
        let tree = crate::huffman::HuffmanTree::from_data(&input)
            .expect("Huffman tree build must succeed");

        let (gpu_encoded_bytes, gpu_bits) = engine
            .huffman_encode_gpu_scan(&input, &tree.code_lut())
            .expect("GPU Huffman encode must succeed for single-symbol input");

        let decoded = tree
            .decode(&gpu_encoded_bytes, gpu_bits)
            .expect("decode of single-symbol GPU output must succeed");
        assert_eq!(decoded, input, "single-symbol round-trip mismatch");
    }
}
