//! GPU-accelerated rANS host-side orchestration.

use super::*;
use crate::frequency::FrequencyTable;
use crate::rans::{
    self, build_symbol_lookup, bytes_as_u16_le, deserialize_freq_table, normalize_frequencies,
    MAX_SCALE_BITS, MIN_SCALE_BITS,
};
use wgpu::util::DeviceExt;

fn write_packed_u16(words: &mut [u32], idx_u16: usize, value: u16) {
    let word_idx = idx_u16 / 2;
    let half = idx_u16 % 2;
    if half == 0 {
        words[word_idx] = (words[word_idx] & 0xFFFF_0000) | u32::from(value);
    } else {
        words[word_idx] = (words[word_idx] & 0x0000_FFFF) | (u32::from(value) << 16);
    }
}

fn build_tables_words(norm: &rans::NormalizedFreqs, scale_bits: u8) -> Vec<u32> {
    let symbol_lookup = build_symbol_lookup(norm);
    let table_size = 1usize << scale_bits;
    let lookup_words = table_size.div_ceil(4);

    // [0..255]   freq
    // [256..511] cum
    // [512..]    cum2sym packed 4 symbols per u32
    let mut tables_words = vec![0u32; 512 + lookup_words];
    for (i, &f) in norm.freq.iter().enumerate() {
        tables_words[i] = u32::from(f);
    }
    for (i, &c) in norm.cum.iter().enumerate() {
        tables_words[256 + i] = u32::from(c);
    }
    for (i, &sym) in symbol_lookup.iter().enumerate() {
        let word = 512 + (i / 4);
        let shift = (i % 4) * 8;
        tables_words[word] |= u32::from(sym) << shift;
    }
    tables_words
}

/// Parameters for chunked GPU rANS decode.
#[derive(Debug, Clone, Copy)]
pub struct RansChunkedDecodeParams {
    pub num_lanes: usize,
    pub scale_bits: u8,
    pub chunk_size: usize,
}

impl WebGpuEngine {
    /// GPU-accelerated chunked rANS encoder.
    ///
    /// Dispatches one workgroup per chunk, with `num_lanes` threads per workgroup.
    pub fn rans_encode_chunked_gpu(
        &self,
        input: &[u8],
        num_lanes: usize,
        scale_bits: u8,
        chunk_size: usize,
    ) -> PzResult<(DeviceBuf, DeviceBuf, wgpu::Buffer)> {
        let num_lanes = num_lanes.max(1);
        let chunk_size = chunk_size.max(1);
        if num_lanes > 4 {
            return Err(PzError::Unsupported);
        }
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        let shader = &self.rans_pipelines().encode;

        // Calculate frequency tables on CPU
        let mut freq = FrequencyTable::new();
        freq.count(input);
        let norm = normalize_frequencies(&freq, scale_bits)?;
        let table_size = 1usize << scale_bits;
        let tables_words = build_tables_words(&norm, scale_bits);

        let tables = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rANS tables"),
                contents: bytemuck::cast_slice(&tables_words),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let chunks: Vec<_> = input.chunks(chunk_size).collect();
        let num_chunks = chunks.len();
        let mut chunk_word_offsets = Vec::with_capacity(num_chunks);
        let mut total_words_u16 = 0usize;
        for chunk in &chunks {
            chunk_word_offsets.push(total_words_u16);
            let symbols_per_lane_max = chunk.len().div_ceil(num_lanes);
            let max_words_per_lane = symbols_per_lane_max * 2 + 4;
            total_words_u16 += num_lanes * max_words_per_lane;
        }

        // Prepare buffers.
        let input_buf = DeviceBuf::from_host(self, input)?;
        let words_bytes = (total_words_u16 * 2).max(4);
        let words_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rANS words"),
                contents: &vec![0u8; words_bytes],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let output_words = DeviceBuf {
            buf: words_buf,
            len: total_words_u16 * 2,
        };
        let states_bytes = (num_chunks * num_lanes * 2 * std::mem::size_of::<u32>()).max(4);
        let output_states = DeviceBuf::alloc(self, states_bytes)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rANS encode encoder"),
            });

        for (i, &chunk) in chunks.iter().enumerate() {
            let chunk_meta_data_arr = [
                (chunk.as_ptr() as usize - input.as_ptr() as usize) as u32, // input offset in bytes
                chunk.len() as u32,
                chunk_word_offsets[i] as u32, // output word-stream offset (u16 units)
                i as u32,                     // chunk id
            ];
            let chunk_meta = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Chunk Meta"),
                    contents: bytemuck::cast_slice(&chunk_meta_data_arr),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let params_data_arr = [num_lanes as u32, scale_bits as u32, table_size as u32, 0];
            let params = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rANS Params"),
                    contents: bytemuck::cast_slice(&params_data_arr),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rANS encode bind group"),
                layout: &shader.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: chunk_meta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buf.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_words.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_states.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tables.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params.as_entire_binding(),
                    },
                ],
            });

            self.record_dispatch(&mut encoder, shader, &bind_group, 1, "rANS encode pass")?;
        }
        self.queue.submit(Some(encoder.finish()));

        Ok((output_words, output_states, tables))
    }

    /// GPU-accelerated chunked rANS decoder.
    pub fn rans_decode_chunked_gpu(
        &self,
        words: &DeviceBuf,
        states: &DeviceBuf,
        tables: &wgpu::Buffer,
        output_len: usize,
        params: RansChunkedDecodeParams,
    ) -> PzResult<DeviceBuf> {
        let RansChunkedDecodeParams {
            num_lanes,
            scale_bits,
            chunk_size,
        } = params;
        let num_lanes = num_lanes.max(1);
        let chunk_size = chunk_size.max(1);
        if num_lanes > 4 {
            return Err(PzError::Unsupported);
        }
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }
        let shader = &self.rans_pipelines().decode;

        let num_chunks = output_len.div_ceil(chunk_size);
        let mut chunk_word_offsets = Vec::with_capacity(num_chunks);
        let mut running_words_u16 = 0usize;
        for i in 0..num_chunks {
            let chunk_output_len = if i == num_chunks - 1 {
                output_len - i * chunk_size
            } else {
                chunk_size
            };
            chunk_word_offsets.push(running_words_u16);
            let symbols_per_lane_max = chunk_output_len.div_ceil(num_lanes);
            let max_words_per_lane = symbols_per_lane_max * 2 + 4;
            running_words_u16 += num_lanes * max_words_per_lane;
        }

        let padded_output_len = output_len.max(4).div_ceil(4) * 4;
        let output_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rANS decode output"),
                contents: &vec![0u8; padded_output_len],
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let output = DeviceBuf {
            buf: output_buf,
            len: output_len,
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rANS decode encoder"),
            });

        for (i, &chunk_words_offset) in chunk_word_offsets.iter().enumerate() {
            let chunk_output_len = if i == num_chunks - 1 {
                output_len - i * chunk_size
            } else {
                chunk_size
            };

            let chunk_meta_data_arr = [
                (i * chunk_size) as u32, // output offset in bytes
                chunk_output_len as u32,
                chunk_words_offset as u32, // word-stream offset (u16 units)
                i as u32,                  // chunk id
            ];
            let chunk_meta = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Chunk Meta"),
                    contents: bytemuck::cast_slice(&chunk_meta_data_arr),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let params_data_arr = [
                num_lanes as u32,
                scale_bits as u32,
                (1 << scale_bits) as u32,
                0,
            ];
            let params = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rANS Params"),
                    contents: bytemuck::cast_slice(&params_data_arr),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rANS decode bind group"),
                layout: &shader.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: chunk_meta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: words.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: states.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output.buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: tables.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: params.as_entire_binding(),
                    },
                ],
            });
            self.record_dispatch(&mut encoder, shader, &bind_group, 1, "rANS decode pass")?;
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(output)
    }

    /// Decode interleaved rANS payload on GPU when compatible.
    ///
    /// Uses the existing interleaved wire format (`encode_interleaved_n`) and
    /// maps it into the chunked-kernel metadata/buffer layout (single chunk).
    /// Falls back should be handled by callers on `PzError::Unsupported`.
    pub fn rans_decode_interleaved_gpu(
        &self,
        input: &[u8],
        original_len: usize,
    ) -> PzResult<Vec<u8>> {
        if original_len == 0 {
            return Ok(Vec::new());
        }
        if input.len() < 1 + (rans::NUM_SYMBOLS * 2) + 1 {
            return Err(PzError::InvalidInput);
        }

        let scale_bits = input[0];
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }
        let norm = deserialize_freq_table(&input[1..], scale_bits)?;
        let mut cursor = 1 + rans::NUM_SYMBOLS * 2;

        let num_lanes = input[cursor] as usize;
        cursor += 1;
        if num_lanes == 0 {
            return Err(PzError::InvalidInput);
        }
        if num_lanes > 4 {
            return Err(PzError::Unsupported);
        }

        if input.len() < cursor + (num_lanes * 4) {
            return Err(PzError::InvalidInput);
        }
        let mut initial_states = vec![0u32; num_lanes];
        for state in &mut initial_states {
            *state = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]);
            cursor += 4;
        }

        if input.len() < cursor + (num_lanes * 4) {
            return Err(PzError::InvalidInput);
        }
        let mut word_counts = vec![0usize; num_lanes];
        for count in &mut word_counts {
            *count = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]) as usize;
            cursor += 4;
        }

        let symbols_per_lane_max = original_len.div_ceil(num_lanes);
        let max_words_per_lane = symbols_per_lane_max * 2 + 4;
        let total_words_u16 = num_lanes * max_words_per_lane;
        let mut words_packed = vec![0u32; total_words_u16.div_ceil(2).max(1)];

        for (lane, &count) in word_counts.iter().enumerate().take(num_lanes) {
            if count > max_words_per_lane || input.len() < cursor + (count * 2) {
                return Err(PzError::InvalidInput);
            }
            let stream = bytes_as_u16_le(&input[cursor..], count);
            cursor += count * 2;

            let lane_start_u16 = lane * max_words_per_lane;
            let write_start_u16 = lane_start_u16 + (max_words_per_lane - count);
            for (j, &word) in stream.iter().enumerate() {
                write_packed_u16(&mut words_packed, write_start_u16 + j, word);
            }
        }

        if cursor != input.len() {
            return Err(PzError::InvalidInput);
        }

        let words_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rANS interleaved words"),
                contents: bytemuck::cast_slice(&words_packed),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let words_dev = DeviceBuf {
            buf: words_buf,
            len: total_words_u16 * 2,
        };

        let mut state_words = vec![0u32; num_lanes * 2];
        for (lane, (&state, &count)) in initial_states
            .iter()
            .zip(word_counts.iter())
            .enumerate()
            .take(num_lanes)
        {
            state_words[lane] = state;
            state_words[num_lanes + lane] = count as u32;
        }
        let states_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rANS interleaved states"),
                contents: bytemuck::cast_slice(&state_words),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let states_dev = DeviceBuf {
            buf: states_buf,
            len: state_words.len() * std::mem::size_of::<u32>(),
        };

        let tables_words = build_tables_words(&norm, scale_bits);
        let tables = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rANS interleaved tables"),
                contents: bytemuck::cast_slice(&tables_words),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let decoded_dev = self.rans_decode_chunked_gpu(
            &words_dev,
            &states_dev,
            &tables,
            original_len,
            RansChunkedDecodeParams {
                num_lanes,
                scale_bits,
                chunk_size: original_len,
            },
        )?;
        decoded_dev.read_to_host(self)
    }
}
