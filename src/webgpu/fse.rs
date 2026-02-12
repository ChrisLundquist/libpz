//! FSE decode GPU kernel.

use super::*;

impl WebGpuEngine {
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
}
