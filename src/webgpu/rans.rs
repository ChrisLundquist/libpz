//! GPU rANS interleaved decode via WebGPU.
//!
//! Single-stream and multi-block rANS decode using wrapping u32 arithmetic
//! (no u64 needed — the low 32 bits of `freq * (state >> scale_bits)` are
//! identical whether computed in u32 or u64).

use super::*;

impl WebGpuEngine {
    /// GPU-accelerated N-way interleaved rANS decode (single stream).
    ///
    /// Parses the interleaved rANS wire format, builds the decode tables on
    /// the CPU, uploads everything to the GPU, and decodes all lanes in
    /// parallel (one workgroup per lane).
    pub fn rans_decode_interleaved(&self, input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
        if original_len == 0 {
            return Ok(Vec::new());
        }

        // Parse header: scale_bits + freq table + num_states
        if input.len() < 1 + 256 * 2 + 1 {
            return Err(PzError::InvalidInput);
        }

        let scale_bits = input[0];
        if !(crate::rans::MIN_SCALE_BITS..=crate::rans::MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }

        let norm = crate::rans::deserialize_freq_table(&input[1..], scale_bits)?;
        let lookup = crate::rans::build_symbol_lookup(&norm);

        let mut cursor = 1 + 256 * 2;
        let num_lanes = input[cursor] as usize;
        if num_lanes == 0 {
            return Err(PzError::InvalidInput);
        }
        cursor += 1;

        // Read initial states
        if input.len() < cursor + num_lanes * 4 {
            return Err(PzError::InvalidInput);
        }
        let mut initial_states = Vec::with_capacity(num_lanes);
        for _ in 0..num_lanes {
            let state = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]);
            initial_states.push(state);
            cursor += 4;
        }

        // Read word counts per lane
        if input.len() < cursor + num_lanes * 4 {
            return Err(PzError::InvalidInput);
        }
        let mut word_counts = Vec::with_capacity(num_lanes);
        for _ in 0..num_lanes {
            let count = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]) as usize;
            word_counts.push(count);
            cursor += 4;
        }

        // Collect all word streams into a single concatenated u16 buffer
        let total_words: usize = word_counts.iter().sum();
        let mut all_words: Vec<u16> = Vec::with_capacity(total_words);
        let mut word_offsets = Vec::with_capacity(num_lanes);

        for &count in &word_counts {
            if input.len() < cursor + count * 2 {
                return Err(PzError::InvalidInput);
            }
            word_offsets.push(all_words.len());
            let ws = crate::rans::bytes_as_u16_le(&input[cursor..], count);
            all_words.extend_from_slice(&ws);
            cursor += count * 2;
        }

        // Build per-lane metadata: [initial_state, num_words, word_offset]
        let mut lane_meta = Vec::with_capacity(num_lanes * 3);
        for i in 0..num_lanes {
            lane_meta.push(initial_states[i]);
            lane_meta.push(word_counts[i] as u32);
            lane_meta.push(word_offsets[i] as u32);
        }

        let table_size = 1u32 << scale_bits;

        // Build combined tables buffer:
        // [cum2sym packed 4-per-u32] [freq_cum packed u32 (low16=freq, high16=cum)]
        let cum2sym_u32_count = (table_size as usize) / 4;
        let mut tables = Vec::with_capacity(cum2sym_u32_count + 256);

        // Pack cum2sym: 4 bytes per u32
        for chunk in lookup.chunks(4) {
            let mut word = 0u32;
            for (j, &byte) in chunk.iter().enumerate() {
                word |= (byte as u32) << (j * 8);
            }
            tables.push(word);
        }
        // Pad to exact size if lookup isn't multiple of 4
        while tables.len() < cum2sym_u32_count {
            tables.push(0);
        }

        // Pack freq/cum: low16 = freq, high16 = cum
        for i in 0..256 {
            tables.push((norm.freq[i] as u32) | ((norm.cum[i] as u32) << 16));
        }

        // Pack word data as u32 (2 u16 per u32)
        let word_u32_count = all_words.len().div_ceil(2);
        let mut word_data_packed = Vec::with_capacity(word_u32_count);
        for chunk in all_words.chunks(2) {
            let lo = chunk[0] as u32;
            let hi = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            word_data_packed.push(lo | (hi << 16));
        }
        if word_data_packed.is_empty() {
            word_data_packed.push(0);
        }

        // Create GPU buffers
        let tables_buf = self.create_buffer_init(
            "rans_tables",
            bytemuck::cast_slice(&tables),
            wgpu::BufferUsages::STORAGE,
        );

        let word_data_buf = self.create_buffer_init(
            "rans_words",
            bytemuck::cast_slice(&word_data_packed),
            wgpu::BufferUsages::STORAGE,
        );

        let lane_meta_buf = self.create_buffer_init(
            "rans_lane_meta",
            bytemuck::cast_slice(&lane_meta),
            wgpu::BufferUsages::STORAGE,
        );

        let output_u32_count = original_len.div_ceil(4);
        let output_buf = self.create_buffer_init(
            "rans_output",
            &vec![0u8; output_u32_count * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = [
            num_lanes as u32,
            original_len as u32,
            scale_bits as u32,
            table_size,
        ];
        let params_buf = self.create_buffer_init(
            "rans_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_rans_decode().get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_decode_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tables_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: word_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lane_meta_buf.as_entire_binding(),
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
            self.pipeline_rans_decode(),
            &bg,
            num_lanes as u32,
            "rans_decode",
        )?;

        // Read output
        let raw_output = self.read_buffer(&output_buf, (output_u32_count * 4) as u64);
        Ok(raw_output[..original_len].to_vec())
    }

    /// GPU-accelerated multi-block rANS decode.
    ///
    /// Decodes multiple independently-encoded rANS blocks in a single GPU
    /// dispatch. Each workgroup handles one block with K interleaved lanes.
    /// All blocks must share the same frequency table.
    pub fn rans_decode_interleaved_blocks(
        &self,
        encoded_blocks: &[(&[u8], usize)],
    ) -> PzResult<Vec<u8>> {
        if encoded_blocks.is_empty() {
            return Ok(Vec::new());
        }

        // Parse the first block's frequency table (all blocks share it).
        let first_input = encoded_blocks[0].0;
        if first_input.len() < 1 + 256 * 2 + 1 {
            return Err(PzError::InvalidInput);
        }

        let scale_bits = first_input[0];
        if !(crate::rans::MIN_SCALE_BITS..=crate::rans::MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }

        let norm = crate::rans::deserialize_freq_table(&first_input[1..], scale_bits)?;
        let lookup = crate::rans::build_symbol_lookup(&norm);

        let table_size = 1u32 << scale_bits;

        // Build combined tables buffer (same layout as single-stream)
        let cum2sym_u32_count = (table_size as usize) / 4;
        let mut tables = Vec::with_capacity(cum2sym_u32_count + 256);

        for chunk in lookup.chunks(4) {
            let mut word = 0u32;
            for (j, &byte) in chunk.iter().enumerate() {
                word |= (byte as u32) << (j * 8);
            }
            tables.push(word);
        }
        while tables.len() < cum2sym_u32_count {
            tables.push(0);
        }
        for i in 0..256 {
            tables.push((norm.freq[i] as u32) | ((norm.cum[i] as u32) << 16));
        }

        // Parse each block and collect metadata.
        let mut all_words: Vec<u16> = Vec::new();
        let mut all_lane_meta: Vec<u32> = Vec::new();
        let mut block_metas: Vec<u32> = Vec::new(); // 5 entries per block
        let mut total_output = 0usize;

        for &(input, original_len) in encoded_blocks {
            let mut cursor = 1 + 256 * 2;
            if input.len() < cursor + 1 {
                return Err(PzError::InvalidInput);
            }
            let num_lanes = input[cursor] as usize;
            if num_lanes == 0 {
                return Err(PzError::InvalidInput);
            }
            cursor += 1;

            // Read initial states
            if input.len() < cursor + num_lanes * 4 {
                return Err(PzError::InvalidInput);
            }
            let mut initial_states = Vec::with_capacity(num_lanes);
            for _ in 0..num_lanes {
                let state = u32::from_le_bytes([
                    input[cursor],
                    input[cursor + 1],
                    input[cursor + 2],
                    input[cursor + 3],
                ]);
                initial_states.push(state);
                cursor += 4;
            }

            // Read word counts
            if input.len() < cursor + num_lanes * 4 {
                return Err(PzError::InvalidInput);
            }
            let mut word_counts = Vec::with_capacity(num_lanes);
            for _ in 0..num_lanes {
                let count = u32::from_le_bytes([
                    input[cursor],
                    input[cursor + 1],
                    input[cursor + 2],
                    input[cursor + 3],
                ]) as usize;
                word_counts.push(count);
                cursor += 4;
            }

            // Block metadata
            let lane_meta_offset = all_lane_meta.len() / 3;
            block_metas.push(total_output as u32); // output_offset
            block_metas.push(original_len as u32); // output_len
            block_metas.push(num_lanes as u32); // lanes_per_block
            block_metas.push(0u32); // unused
            block_metas.push(lane_meta_offset as u32); // lane_meta_offset

            // Read word streams and build lane metadata
            for i in 0..num_lanes {
                let count = word_counts[i];
                if input.len() < cursor + count * 2 {
                    return Err(PzError::InvalidInput);
                }
                let word_offset = all_words.len();
                let ws = crate::rans::bytes_as_u16_le(&input[cursor..], count);
                all_words.extend_from_slice(&ws);
                cursor += count * 2;

                all_lane_meta.push(initial_states[i]);
                all_lane_meta.push(count as u32);
                all_lane_meta.push(word_offset as u32);
            }

            total_output += original_len;
        }

        if total_output == 0 {
            return Ok(Vec::new());
        }

        let num_blocks = encoded_blocks.len();

        // Combine block_meta and lane_meta into single buffer
        let bm_count = block_metas.len();
        let mut combined_meta = block_metas;
        combined_meta.extend_from_slice(&all_lane_meta);

        // Pack word data as u32 (2 u16 per u32)
        let mut word_data_packed: Vec<u32> = Vec::with_capacity(all_words.len().div_ceil(2));
        for chunk in all_words.chunks(2) {
            let lo = chunk[0] as u32;
            let hi = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            word_data_packed.push(lo | (hi << 16));
        }
        if word_data_packed.is_empty() {
            word_data_packed.push(0);
        }

        // Create GPU buffers
        let tables_buf = self.create_buffer_init(
            "rans_blk_tables",
            bytemuck::cast_slice(&tables),
            wgpu::BufferUsages::STORAGE,
        );

        let word_data_buf = self.create_buffer_init(
            "rans_blk_words",
            bytemuck::cast_slice(&word_data_packed),
            wgpu::BufferUsages::STORAGE,
        );

        let metadata_buf = self.create_buffer_init(
            "rans_blk_metadata",
            bytemuck::cast_slice(&combined_meta),
            wgpu::BufferUsages::STORAGE,
        );

        let output_u32_count = total_output.div_ceil(4);
        let output_buf = self.create_buffer_init(
            "rans_blk_output",
            &vec![0u8; output_u32_count * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let params = [
            num_blocks as u32,
            scale_bits as u32,
            table_size,
            bm_count as u32,
        ];
        let params_buf = self.create_buffer_init(
            "rans_blk_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_rans_decode_blocks().get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_blk_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tables_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: word_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: metadata_buf.as_entire_binding(),
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
            self.pipeline_rans_decode_blocks(),
            &bg,
            num_blocks as u32,
            "rans_decode_blocks",
        )?;

        // Read output
        let raw_output = self.read_buffer(&output_buf, (output_u32_count * 4) as u64);
        Ok(raw_output[..total_output].to_vec())
    }
}
