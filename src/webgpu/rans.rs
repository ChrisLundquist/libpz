//! GPU-accelerated rANS host-side orchestration.
//!
//! Slice 1 implementation focuses on decode-first parity using the CPU wire
//! format as the compatibility contract.

use super::*;

use crate::frequency::FrequencyTable;
use crate::gpu_common::BufferRing;
use crate::rans::{
    self, build_symbol_lookup, bytes_as_u16_le, deserialize_freq_table, normalize_frequencies,
    serialize_freq_table, MAX_SCALE_BITS, MIN_SCALE_BITS, NUM_SYMBOLS,
};

const RANS_STORAGE_BINDINGS_PER_STAGE: u32 = 5;

fn write_packed_u16(words: &mut [u32], idx_u16: usize, value: u16) {
    let word_idx = idx_u16 / 2;
    let half = idx_u16 % 2;
    if half == 0 {
        words[word_idx] = (words[word_idx] & 0xFFFF_0000) | u32::from(value);
    } else {
        words[word_idx] = (words[word_idx] & 0x0000_FFFF) | (u32::from(value) << 16);
    }
}

fn read_u16_le(input: &[u8], cursor: &mut usize) -> PzResult<u16> {
    if input.len() < *cursor + 2 {
        return Err(PzError::InvalidInput);
    }
    let v = u16::from_le_bytes([input[*cursor], input[*cursor + 1]]);
    *cursor += 2;
    Ok(v)
}

fn read_u32_le(input: &[u8], cursor: &mut usize) -> PzResult<u32> {
    if input.len() < *cursor + 4 {
        return Err(PzError::InvalidInput);
    }
    let v = u32::from_le_bytes([
        input[*cursor],
        input[*cursor + 1],
        input[*cursor + 2],
        input[*cursor + 3],
    ]);
    *cursor += 4;
    Ok(v)
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

fn can_encode_chunked(input_len: usize, chunk_bytes: usize) -> bool {
    const MAX_CHUNK_META: usize = u16::MAX as usize;
    if input_len == 0 {
        return true;
    }
    if chunk_bytes == 0 || chunk_bytes > MAX_CHUNK_META {
        return false;
    }
    input_len.div_ceil(chunk_bytes) <= MAX_CHUNK_META
}

fn validate_chunk_grid(chunk_lens: &[usize], chunk_size: usize, original_len: usize) -> bool {
    if chunk_lens.is_empty() || chunk_size == 0 {
        return false;
    }
    for (chunk_idx, &len) in chunk_lens.iter().enumerate() {
        let expected = if chunk_idx + 1 == chunk_lens.len() {
            original_len - chunk_idx * chunk_size
        } else {
            chunk_size
        };
        if len != expected {
            return false;
        }
    }
    true
}

/// Parameters for chunked GPU rANS decode.
#[derive(Debug, Clone, Copy)]
pub struct RansChunkedDecodeParams {
    pub num_lanes: usize,
    pub scale_bits: u8,
    pub chunk_size: usize,
}

struct PendingRansChunkedPayloadEncode {
    words_dev: DeviceBuf,
    states_dev: DeviceBuf,
    _tables: wgpu::Buffer,
    norm: rans::NormalizedFreqs,
    chunk_lens: Vec<usize>,
    num_lanes: usize,
    input_len: usize,
}

struct PreparedRansChunkedPayloadDecode {
    words_dev: DeviceBuf,
    states_dev: DeviceBuf,
    tables: wgpu::Buffer,
    params: RansChunkedDecodeParams,
}

struct PendingRansChunkedPayloadDecode {
    output_dev: DeviceBuf,
    _words_dev: DeviceBuf,
    _states_dev: DeviceBuf,
    _tables: wgpu::Buffer,
}

impl WebGpuEngine {
    fn rans_encode_chunked_gpu_with_norm(
        &self,
        input: &[u8],
        num_lanes: usize,
        chunk_size: usize,
        norm: &rans::NormalizedFreqs,
    ) -> PzResult<(DeviceBuf, DeviceBuf, wgpu::Buffer)> {
        if self.max_storage_buffers_per_shader_stage < RANS_STORAGE_BINDINGS_PER_STAGE {
            return Err(PzError::Unsupported);
        }
        if input.is_empty() || num_lanes == 0 || num_lanes > 64 || chunk_size == 0 {
            return Err(PzError::InvalidInput);
        }

        let effective_scale_bits = norm.scale_bits;
        let tables_words = build_tables_words(norm, effective_scale_bits);
        let tables_buf = self.create_buffer_init(
            "rans_encode_tables",
            bytemuck::cast_slice(&tables_words),
            wgpu::BufferUsages::STORAGE,
        );

        let input_buf = DeviceBuf::from_host(self, input)?;
        let num_chunks = input.len().div_ceil(chunk_size);
        let mut chunk_meta_words = Vec::with_capacity(num_chunks * 4);
        let mut running_words_u16 = 0usize;
        for chunk_idx in 0..num_chunks {
            let chunk_input_len = if chunk_idx + 1 == num_chunks {
                input.len() - chunk_idx * chunk_size
            } else {
                chunk_size
            };
            let symbols_per_lane_max = chunk_input_len.div_ceil(num_lanes);
            let max_words_per_lane = symbols_per_lane_max
                .checked_mul(2)
                .and_then(|v| v.checked_add(4))
                .ok_or(PzError::InvalidInput)?;
            let state_offset = chunk_idx
                .checked_mul(num_lanes)
                .and_then(|v| v.checked_mul(2))
                .ok_or(PzError::InvalidInput)?;

            chunk_meta_words.extend_from_slice(&[
                (chunk_idx
                    .checked_mul(chunk_size)
                    .ok_or(PzError::InvalidInput)?) as u32,
                chunk_input_len as u32,
                running_words_u16 as u32,
                state_offset as u32,
            ]);

            running_words_u16 = running_words_u16
                .checked_add(
                    num_lanes
                        .checked_mul(max_words_per_lane)
                        .ok_or(PzError::InvalidInput)?,
                )
                .ok_or(PzError::InvalidInput)?;
        }

        let words_u32_count = running_words_u16.div_ceil(2).max(1);
        let words_buf = self.create_buffer_init(
            "rans_encode_words",
            &vec![0u8; words_u32_count * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let words_dev = DeviceBuf {
            buf: words_buf,
            len: running_words_u16 * 2,
        };

        let states_u32_count = num_chunks
            .checked_mul(num_lanes)
            .and_then(|v| v.checked_mul(2))
            .ok_or(PzError::InvalidInput)?;
        let states_buf = self.create_buffer_init(
            "rans_encode_states",
            &vec![0u8; states_u32_count * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let states_dev = DeviceBuf {
            buf: states_buf,
            len: states_u32_count * std::mem::size_of::<u32>(),
        };

        let chunk_meta_buf = self.create_buffer_init(
            "rans_encode_chunk_meta",
            bytemuck::cast_slice(&chunk_meta_words),
            wgpu::BufferUsages::STORAGE,
        );
        // [num_chunks, num_lanes, scale_bits, chunk_size]
        let params_words = [
            num_chunks as u32,
            num_lanes as u32,
            effective_scale_bits as u32,
            chunk_size as u32,
        ];
        let params_buf = self.create_buffer_init(
            "rans_encode_params",
            bytemuck::cast_slice(&params_words),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_rans_encode().get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_encode_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunk_meta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buf.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: words_dev.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: states_dev.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tables_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = u32::try_from(num_chunks).map_err(|_| PzError::Unsupported)?;
        self.dispatch(
            self.pipeline_rans_encode(),
            &bg,
            workgroups_x,
            "rans_encode_chunked",
        )?;

        Ok((words_dev, states_dev, tables_buf))
    }

    fn submit_rans_chunked_payload_encode(
        &self,
        input: &[u8],
        num_lanes: usize,
        scale_bits: u8,
        chunk_size: usize,
    ) -> PzResult<PendingRansChunkedPayloadEncode> {
        let mut freq = FrequencyTable::new();
        freq.count(input);
        let norm = normalize_frequencies(&freq, scale_bits)?;
        let num_chunks = input.len().div_ceil(chunk_size);
        let chunk_lens: Vec<usize> = (0..num_chunks)
            .map(|chunk_idx| {
                if chunk_idx + 1 == num_chunks {
                    input.len() - chunk_idx * chunk_size
                } else {
                    chunk_size
                }
            })
            .collect();

        let (words_dev, states_dev, tables) =
            self.rans_encode_chunked_gpu_with_norm(input, num_lanes, chunk_size, &norm)?;

        Ok(PendingRansChunkedPayloadEncode {
            words_dev,
            states_dev,
            _tables: tables,
            norm,
            chunk_lens,
            num_lanes,
            input_len: input.len(),
        })
    }

    fn read_two_buffers(&self, a: &wgpu::Buffer, b: &wgpu::Buffer) -> PzResult<(Vec<u8>, Vec<u8>)> {
        let size_a = a.size();
        let size_b = b.size();

        let staging_a = self.create_buffer(
            "rans_readback_a",
            size_a,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let staging_b = self.create_buffer(
            "rans_readback_b",
            size_b,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rans_readback"),
            });
        encoder.copy_buffer_to_buffer(a, 0, &staging_a, 0, size_a);
        encoder.copy_buffer_to_buffer(b, 0, &staging_b, 0, size_b);
        self.queue.submit(Some(encoder.finish()));

        let slice_a = staging_a.slice(..);
        let slice_b = staging_b.slice(..);

        let (tx_a, rx_a) = std::sync::mpsc::channel();
        let (tx_b, rx_b) = std::sync::mpsc::channel();
        slice_a.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx_a.send(result);
        });
        slice_b.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx_b.send(result);
        });

        self.poll_wait();
        rx_a.recv()
            .map_err(|_| PzError::Unsupported)?
            .map_err(|_| PzError::Unsupported)?;
        rx_b.recv()
            .map_err(|_| PzError::Unsupported)?
            .map_err(|_| PzError::Unsupported)?;

        let data_a = slice_a.get_mapped_range().to_vec();
        let data_b = slice_b.get_mapped_range().to_vec();
        staging_a.unmap();
        staging_b.unmap();

        Ok((data_a, data_b))
    }

    fn complete_rans_chunked_payload_encode(
        &self,
        pending: PendingRansChunkedPayloadEncode,
    ) -> PzResult<Vec<u8>> {
        let PendingRansChunkedPayloadEncode {
            words_dev,
            states_dev,
            _tables,
            norm,
            chunk_lens,
            num_lanes,
            input_len,
        } = pending;

        let (words_raw, states_raw) = self.read_two_buffers(&words_dev.buf, &states_dev.buf)?;
        let words_u32: &[u32] = bytemuck::cast_slice(&words_raw);
        let words_u16: &[u16] = bytemuck::cast_slice(words_u32);
        let states_u32: &[u32] = bytemuck::cast_slice(&states_raw);

        let num_chunks = chunk_lens.len();
        let header_size = 1 + NUM_SYMBOLS * 2 + 2 + num_chunks * 2;
        let mut output = Vec::with_capacity(header_size + input_len);
        output.push(norm.scale_bits);
        serialize_freq_table(&norm, &mut output);
        output.extend_from_slice(&(num_chunks as u16).to_le_bytes());
        for &chunk_len in &chunk_lens {
            output.extend_from_slice(&(chunk_len as u16).to_le_bytes());
        }

        let mut running_words_u16 = 0usize;
        for (chunk_idx, &chunk_len) in chunk_lens.iter().enumerate() {
            let symbols_per_lane_max = chunk_len.div_ceil(num_lanes);
            let max_words_per_lane = symbols_per_lane_max
                .checked_mul(2)
                .and_then(|v| v.checked_add(4))
                .ok_or(PzError::InvalidInput)?;
            let chunk_base = chunk_idx
                .checked_mul(num_lanes)
                .and_then(|v| v.checked_mul(2))
                .ok_or(PzError::InvalidInput)?;

            if states_u32.len() < chunk_base + num_lanes * 2 {
                return Err(PzError::InvalidInput);
            }

            output.push(num_lanes as u8);
            for lane in 0..num_lanes {
                output.extend_from_slice(&states_u32[chunk_base + lane].to_le_bytes());
            }
            for lane in 0..num_lanes {
                output.extend_from_slice(&states_u32[chunk_base + num_lanes + lane].to_le_bytes());
            }

            for lane in 0..num_lanes {
                let count = usize::try_from(states_u32[chunk_base + num_lanes + lane])
                    .map_err(|_| PzError::InvalidInput)?;
                if count > max_words_per_lane {
                    return Err(PzError::InvalidInput);
                }
                let lane_start_u16 = running_words_u16
                    .checked_add(
                        lane.checked_mul(max_words_per_lane)
                            .ok_or(PzError::InvalidInput)?,
                    )
                    .ok_or(PzError::InvalidInput)?;
                let read_start_u16 = lane_start_u16 + (max_words_per_lane - count);
                for j in 0..count {
                    let word = words_u16
                        .get(read_start_u16 + j)
                        .copied()
                        .ok_or(PzError::InvalidInput)?;
                    output.extend_from_slice(&word.to_le_bytes());
                }
            }

            running_words_u16 = running_words_u16
                .checked_add(
                    num_lanes
                        .checked_mul(max_words_per_lane)
                        .ok_or(PzError::InvalidInput)?,
                )
                .ok_or(PzError::InvalidInput)?;
        }

        Ok(output)
    }

    /// GPU-accelerated chunked rANS encoder.
    ///
    /// Returns GPU-resident lane word streams + state metadata and the table
    /// buffer needed for decode. The layout matches `rans_decode_chunked_gpu`.
    pub fn rans_encode_chunked_gpu(
        &self,
        input: &[u8],
        num_lanes: usize,
        scale_bits: u8,
        chunk_size: usize,
    ) -> PzResult<(DeviceBuf, DeviceBuf, wgpu::Buffer)> {
        if input.is_empty() || num_lanes == 0 || num_lanes > 64 || chunk_size == 0 {
            return Err(PzError::InvalidInput);
        }
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }

        let mut freq = FrequencyTable::new();
        freq.count(input);
        let norm = normalize_frequencies(&freq, scale_bits)?;
        self.rans_encode_chunked_gpu_with_norm(input, num_lanes, chunk_size, &norm)
    }

    /// GPU-accelerated chunked rANS decoder.
    ///
    /// The input buffers must match the packed lane layout produced by
    /// `rans_decode_interleaved_gpu()`/`rans_decode_chunked_payload_gpu()`.
    pub fn rans_decode_chunked_gpu(
        &self,
        words: &DeviceBuf,
        states: &DeviceBuf,
        tables: &wgpu::Buffer,
        output_len: usize,
        params: RansChunkedDecodeParams,
    ) -> PzResult<DeviceBuf> {
        if self.max_storage_buffers_per_shader_stage < RANS_STORAGE_BINDINGS_PER_STAGE {
            return Err(PzError::Unsupported);
        }
        let RansChunkedDecodeParams {
            num_lanes,
            scale_bits,
            chunk_size,
        } = params;
        if output_len == 0 || num_lanes == 0 || num_lanes > 64 || chunk_size == 0 {
            return Err(PzError::InvalidInput);
        }
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }

        let num_chunks = output_len.div_ceil(chunk_size);
        let mut chunk_meta_words = Vec::with_capacity(num_chunks * 4);
        let mut running_words_u16 = 0usize;
        for chunk_idx in 0..num_chunks {
            let chunk_output_len = if chunk_idx + 1 == num_chunks {
                output_len - chunk_idx * chunk_size
            } else {
                chunk_size
            };
            let symbols_per_lane_max = chunk_output_len.div_ceil(num_lanes);
            let max_words_per_lane = symbols_per_lane_max
                .checked_mul(2)
                .and_then(|v| v.checked_add(4))
                .ok_or(PzError::InvalidInput)?;
            let state_offset = chunk_idx
                .checked_mul(num_lanes)
                .and_then(|v| v.checked_mul(2))
                .ok_or(PzError::InvalidInput)?;

            chunk_meta_words.extend_from_slice(&[
                (chunk_idx
                    .checked_mul(chunk_size)
                    .ok_or(PzError::InvalidInput)?) as u32,
                chunk_output_len as u32,
                running_words_u16 as u32,
                state_offset as u32,
            ]);

            running_words_u16 = running_words_u16
                .checked_add(
                    num_lanes
                        .checked_mul(max_words_per_lane)
                        .ok_or(PzError::InvalidInput)?,
                )
                .ok_or(PzError::InvalidInput)?;
        }

        let required_word_bytes = running_words_u16
            .checked_mul(2)
            .ok_or(PzError::InvalidInput)?;
        if words.len() < required_word_bytes {
            return Err(PzError::InvalidInput);
        }

        let required_state_bytes = num_chunks
            .checked_mul(num_lanes)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_mul(std::mem::size_of::<u32>()))
            .ok_or(PzError::InvalidInput)?;
        if states.len() < required_state_bytes {
            return Err(PzError::InvalidInput);
        }

        let output_u32_words = output_len.div_ceil(4).max(1);
        let output_buf = self.create_buffer_init(
            "rans_decode_output",
            &vec![0u8; output_u32_words * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output = DeviceBuf {
            buf: output_buf,
            len: output_len,
        };

        let chunk_meta_buf = self.create_buffer_init(
            "rans_decode_chunk_meta",
            bytemuck::cast_slice(&chunk_meta_words),
            wgpu::BufferUsages::STORAGE,
        );

        // [num_chunks, num_lanes, scale_bits, chunk_size]
        let params_words = [
            num_chunks as u32,
            num_lanes as u32,
            scale_bits as u32,
            chunk_size as u32,
        ];
        let params_buf = self.create_buffer_init(
            "rans_decode_params",
            bytemuck::cast_slice(&params_words),
            wgpu::BufferUsages::UNIFORM,
        );

        let bg_layout = self.pipeline_rans_decode().get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_decode_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: chunk_meta_buf.as_entire_binding(),
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
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups_x = u32::try_from(num_chunks).map_err(|_| PzError::Unsupported)?;
        self.dispatch(
            self.pipeline_rans_decode(),
            &bg,
            workgroups_x,
            "rans_decode_chunked",
        )?;

        Ok(output)
    }

    /// Decode interleaved rANS payload on GPU when compatible.
    ///
    /// Uses CPU-compatible interleaved wire format (`encode_interleaved_n`).
    pub fn rans_decode_interleaved_gpu(
        &self,
        input: &[u8],
        original_len: usize,
    ) -> PzResult<Vec<u8>> {
        if original_len == 0 {
            return Ok(Vec::new());
        }
        if input.len() < 1 + NUM_SYMBOLS * 2 + 1 {
            return Err(PzError::InvalidInput);
        }

        let scale_bits = input[0];
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }
        let norm = deserialize_freq_table(&input[1..], scale_bits)?;

        let mut cursor = 1 + NUM_SYMBOLS * 2;
        let num_lanes = input[cursor] as usize;
        cursor += 1;
        if num_lanes == 0 || num_lanes > 64 {
            return Err(PzError::Unsupported);
        }

        if input.len() < cursor + num_lanes * 4 {
            return Err(PzError::InvalidInput);
        }
        let mut initial_states = vec![0u32; num_lanes];
        for state in &mut initial_states {
            *state = read_u32_le(input, &mut cursor)?;
        }

        if input.len() < cursor + num_lanes * 4 {
            return Err(PzError::InvalidInput);
        }
        let mut word_counts = vec![0u32; num_lanes];
        for count in &mut word_counts {
            *count = read_u32_le(input, &mut cursor)?;
        }

        let symbols_per_lane_max = original_len.div_ceil(num_lanes);
        let max_words_per_lane = symbols_per_lane_max
            .checked_mul(2)
            .and_then(|v| v.checked_add(4))
            .ok_or(PzError::InvalidInput)?;
        let total_words_u16 = num_lanes
            .checked_mul(max_words_per_lane)
            .ok_or(PzError::InvalidInput)?;
        let mut words_packed = vec![0u32; total_words_u16.div_ceil(2).max(1)];

        for (lane, &count_u32) in word_counts.iter().enumerate().take(num_lanes) {
            let count = usize::try_from(count_u32).map_err(|_| PzError::InvalidInput)?;
            if count > max_words_per_lane || input.len() < cursor + count * 2 {
                return Err(PzError::InvalidInput);
            }

            let stream = bytes_as_u16_le(&input[cursor..], count);
            cursor += count * 2;

            let lane_start_u16 = lane
                .checked_mul(max_words_per_lane)
                .ok_or(PzError::InvalidInput)?;
            let write_start_u16 = lane_start_u16 + (max_words_per_lane - count);
            for (j, &word) in stream.iter().enumerate() {
                write_packed_u16(&mut words_packed, write_start_u16 + j, word);
            }
        }

        if cursor != input.len() {
            return Err(PzError::InvalidInput);
        }

        let words_buf = self.create_buffer_init(
            "rans_interleaved_words",
            bytemuck::cast_slice(&words_packed),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let words_dev = DeviceBuf {
            buf: words_buf,
            len: total_words_u16 * 2,
        };

        let mut state_words = vec![0u32; num_lanes * 2];
        state_words[..num_lanes].copy_from_slice(&initial_states);
        state_words[num_lanes..(num_lanes + num_lanes)].copy_from_slice(&word_counts);
        let states_buf = self.create_buffer_init(
            "rans_interleaved_states",
            bytemuck::cast_slice(&state_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let states_dev = DeviceBuf {
            buf: states_buf,
            len: state_words.len() * std::mem::size_of::<u32>(),
        };

        let tables_words = build_tables_words(&norm, scale_bits);
        let tables = self.create_buffer_init(
            "rans_interleaved_tables",
            bytemuck::cast_slice(&tables_words),
            wgpu::BufferUsages::STORAGE,
        );

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

    fn prepare_rans_chunked_payload_decode(
        &self,
        input: &[u8],
        original_len: usize,
    ) -> PzResult<PreparedRansChunkedPayloadDecode> {
        if input.len() < 1 + NUM_SYMBOLS * 2 + 2 {
            return Err(PzError::InvalidInput);
        }

        let scale_bits = input[0];
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }
        let norm = deserialize_freq_table(&input[1..], scale_bits)?;

        let mut cursor = 1 + NUM_SYMBOLS * 2;
        let num_chunks = read_u16_le(input, &mut cursor)? as usize;
        if num_chunks == 0 {
            return Err(PzError::InvalidInput);
        }
        if input.len() < cursor + num_chunks * 2 {
            return Err(PzError::InvalidInput);
        }

        let mut chunk_lens = Vec::with_capacity(num_chunks);
        let mut total_chunk_len = 0usize;
        for _ in 0..num_chunks {
            let len = read_u16_le(input, &mut cursor)? as usize;
            total_chunk_len = total_chunk_len
                .checked_add(len)
                .ok_or(PzError::InvalidInput)?;
            chunk_lens.push(len);
        }
        if total_chunk_len != original_len {
            return Err(PzError::InvalidInput);
        }
        if input.len() <= cursor {
            return Err(PzError::InvalidInput);
        }

        let num_lanes = input[cursor] as usize;
        if num_lanes == 0 || num_lanes > 64 {
            return Err(PzError::Unsupported);
        }
        let chunk_size = chunk_lens[0].max(1);
        if !validate_chunk_grid(&chunk_lens, chunk_size, original_len) {
            // Current GPU path assumes fixed-size chunk grid (last chunk may be short).
            return Err(PzError::Unsupported);
        }

        let mut total_words_u16 = 0usize;
        for &chunk_len in &chunk_lens {
            let symbols_per_lane_max = chunk_len.div_ceil(num_lanes);
            let max_words_per_lane = symbols_per_lane_max
                .checked_mul(2)
                .and_then(|v| v.checked_add(4))
                .ok_or(PzError::InvalidInput)?;
            total_words_u16 = total_words_u16
                .checked_add(
                    num_lanes
                        .checked_mul(max_words_per_lane)
                        .ok_or(PzError::InvalidInput)?,
                )
                .ok_or(PzError::InvalidInput)?;
        }

        let mut words_packed = vec![0u32; total_words_u16.div_ceil(2).max(1)];
        let mut state_words = vec![0u32; num_chunks * num_lanes * 2];
        let mut running_words_u16 = 0usize;

        for (chunk_idx, &chunk_len) in chunk_lens.iter().enumerate() {
            if input.len() < cursor + 1 {
                return Err(PzError::InvalidInput);
            }
            let lane_count = input[cursor] as usize;
            cursor += 1;
            if lane_count != num_lanes {
                return Err(PzError::Unsupported);
            }

            let chunk_base = chunk_idx * num_lanes * 2;
            for lane in 0..num_lanes {
                state_words[chunk_base + lane] = read_u32_le(input, &mut cursor)?;
            }

            let symbols_per_lane_max = chunk_len.div_ceil(num_lanes);
            let max_words_per_lane = symbols_per_lane_max
                .checked_mul(2)
                .and_then(|v| v.checked_add(4))
                .ok_or(PzError::InvalidInput)?;

            let mut word_counts = vec![0usize; num_lanes];
            for lane in 0..num_lanes {
                let count_u32 = read_u32_le(input, &mut cursor)?;
                let count = usize::try_from(count_u32).map_err(|_| PzError::InvalidInput)?;
                if count > max_words_per_lane {
                    return Err(PzError::InvalidInput);
                }
                word_counts[lane] = count;
                state_words[chunk_base + num_lanes + lane] = count_u32;
            }

            for (lane, &count) in word_counts.iter().enumerate().take(num_lanes) {
                let bytes = count.checked_mul(2).ok_or(PzError::InvalidInput)?;
                if input.len() < cursor + bytes {
                    return Err(PzError::InvalidInput);
                }
                let stream = bytes_as_u16_le(&input[cursor..], count);
                cursor += bytes;

                let lane_start_u16 = running_words_u16
                    .checked_add(
                        lane.checked_mul(max_words_per_lane)
                            .ok_or(PzError::InvalidInput)?,
                    )
                    .ok_or(PzError::InvalidInput)?;
                let write_start_u16 = lane_start_u16 + (max_words_per_lane - count);
                for (j, &word) in stream.iter().enumerate() {
                    write_packed_u16(&mut words_packed, write_start_u16 + j, word);
                }
            }

            running_words_u16 = running_words_u16
                .checked_add(
                    num_lanes
                        .checked_mul(max_words_per_lane)
                        .ok_or(PzError::InvalidInput)?,
                )
                .ok_or(PzError::InvalidInput)?;
        }

        if cursor != input.len() || running_words_u16 != total_words_u16 {
            return Err(PzError::InvalidInput);
        }

        let words_buf = self.create_buffer_init(
            "rans_chunked_words",
            bytemuck::cast_slice(&words_packed),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let words_dev = DeviceBuf {
            buf: words_buf,
            len: total_words_u16 * 2,
        };

        let states_buf = self.create_buffer_init(
            "rans_chunked_states",
            bytemuck::cast_slice(&state_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let states_dev = DeviceBuf {
            buf: states_buf,
            len: state_words.len() * std::mem::size_of::<u32>(),
        };

        let tables_words = build_tables_words(&norm, scale_bits);
        let tables = self.create_buffer_init(
            "rans_chunked_tables",
            bytemuck::cast_slice(&tables_words),
            wgpu::BufferUsages::STORAGE,
        );

        Ok(PreparedRansChunkedPayloadDecode {
            words_dev,
            states_dev,
            tables,
            params: RansChunkedDecodeParams {
                num_lanes,
                scale_bits,
                chunk_size,
            },
        })
    }

    fn submit_rans_chunked_payload_decode(
        &self,
        input: &[u8],
        original_len: usize,
    ) -> PzResult<PendingRansChunkedPayloadDecode> {
        let prepared = self.prepare_rans_chunked_payload_decode(input, original_len)?;
        let output_dev = self.rans_decode_chunked_gpu(
            &prepared.words_dev,
            &prepared.states_dev,
            &prepared.tables,
            original_len,
            prepared.params,
        )?;
        Ok(PendingRansChunkedPayloadDecode {
            output_dev,
            _words_dev: prepared.words_dev,
            _states_dev: prepared.states_dev,
            _tables: prepared.tables,
        })
    }

    fn complete_rans_chunked_payload_decode(
        &self,
        pending: PendingRansChunkedPayloadDecode,
    ) -> PzResult<Vec<u8>> {
        pending.output_dev.read_to_host(self)
    }

    /// Decode chunked interleaved rANS payload (`rans::encode_chunked_n`) on GPU.
    ///
    /// Preserves CPU wire compatibility and rejects malformed metadata.
    pub fn rans_decode_chunked_payload_gpu(
        &self,
        input: &[u8],
        original_len: usize,
    ) -> PzResult<Vec<u8>> {
        if original_len == 0 {
            return Ok(Vec::new());
        }
        let pending = self.submit_rans_chunked_payload_decode(input, original_len)?;
        self.complete_rans_chunked_payload_decode(pending)
    }

    fn rans_decode_pending_ring_depth(&self, inputs: &[(&[u8], usize)]) -> usize {
        let max_depth = inputs.len().max(1);
        let max_payload = inputs
            .iter()
            .map(|(payload, _)| payload.len())
            .max()
            .unwrap_or(0);
        let max_output = inputs
            .iter()
            .map(|(_, output_len)| *output_len)
            .max()
            .unwrap_or(0);
        let per_slot = max_payload.saturating_add(max_output);
        if per_slot == 0 {
            return 1;
        }

        // Mirror the LZ77 streaming policy: reserve headroom, clamp to 3 slots.
        let budget = (self.gpu_memory_budget() * 3) / 4;
        (budget / per_slot).clamp(1, 3).min(max_depth)
    }

    /// Batched GPU chunked rANS decode with ring-buffered submit/readback.
    ///
    /// Inputs are `(payload, original_len)` tuples where `payload` is in
    /// `rans::encode_chunked_n` wire format.
    pub fn rans_decode_chunked_payload_gpu_batched(
        &self,
        inputs: &[(&[u8], usize)],
    ) -> PzResult<Vec<Vec<u8>>> {
        let ring_depth = self.rans_decode_pending_ring_depth(inputs);
        let mut results: Vec<Option<Vec<u8>>> = vec![None; inputs.len()];
        let mut ring = BufferRing::new(
            (0..ring_depth)
                .map(|_| None::<(usize, PendingRansChunkedPayloadDecode)>)
                .collect(),
        );

        for (idx, &(input, original_len)) in inputs.iter().enumerate() {
            let slot_idx = ring.acquire();
            if let Some((done_idx, done)) = ring.slots[slot_idx].take() {
                let decoded = self.complete_rans_chunked_payload_decode(done)?;
                results[done_idx] = Some(decoded);
            }

            if original_len == 0 {
                results[idx] = Some(Vec::new());
                continue;
            }

            let submitted = self.submit_rans_chunked_payload_decode(input, original_len)?;
            ring.slots[slot_idx] = Some((idx, submitted));
        }

        for slot in &mut ring.slots {
            if let Some((done_idx, done)) = slot.take() {
                let decoded = self.complete_rans_chunked_payload_decode(done)?;
                results[done_idx] = Some(decoded);
            }
        }

        Ok(results
            .into_iter()
            .map(|r| r.expect("all batched rans decode results must be populated"))
            .collect())
    }

    /// Encode chunked interleaved rANS payload on GPU using CPU wire format.
    ///
    /// Returns `(encoded, used_chunked)` with fallback behavior that mirrors
    /// `rans::encode_chunked_n` metadata limits.
    pub fn rans_encode_chunked_payload_gpu(
        &self,
        input: &[u8],
        num_lanes: usize,
        scale_bits: u8,
        chunk_size: usize,
    ) -> PzResult<(Vec<u8>, bool)> {
        if input.is_empty() {
            return Ok((Vec::new(), true));
        }

        let lanes_clamped = num_lanes.clamp(1, u8::MAX as usize);
        let scale_bits = scale_bits.clamp(MIN_SCALE_BITS, MAX_SCALE_BITS);

        if !can_encode_chunked(input.len(), chunk_size) {
            let encoded = rans::encode_interleaved_n(input, lanes_clamped, scale_bits);
            return Ok((encoded, false));
        }

        if lanes_clamped > 64 {
            return Err(PzError::Unsupported);
        }

        let pending =
            self.submit_rans_chunked_payload_encode(input, lanes_clamped, scale_bits, chunk_size)?;
        Ok((self.complete_rans_chunked_payload_encode(pending)?, true))
    }

    fn rans_encode_pending_ring_depth(
        &self,
        inputs: &[&[u8]],
        num_lanes: usize,
        chunk_size: usize,
    ) -> usize {
        let max_depth = inputs.len().max(1);
        let max_input = inputs.iter().map(|b| b.len()).max().unwrap_or(0);
        if max_input == 0 || chunk_size == 0 || num_lanes == 0 {
            return 1;
        }

        let chunks = max_input.div_ceil(chunk_size);
        let symbols_per_lane = chunk_size.div_ceil(num_lanes);
        let max_words_per_lane = symbols_per_lane.saturating_mul(2).saturating_add(4);
        let words_bytes = chunks
            .saturating_mul(num_lanes)
            .saturating_mul(max_words_per_lane)
            .saturating_mul(2);
        let states_bytes = chunks
            .saturating_mul(num_lanes)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<u32>());
        let input_bytes = max_input;

        let per_slot = input_bytes
            .saturating_add(words_bytes)
            .saturating_add(states_bytes);
        if per_slot == 0 {
            return 1;
        }

        // Mirror the LZ77 streaming policy: reserve headroom, clamp to 3 slots.
        let budget = (self.gpu_memory_budget() * 3) / 4;
        (budget / per_slot).clamp(1, 3).min(max_depth)
    }

    /// Batched GPU chunked rANS encode with ring-buffered submit/readback.
    ///
    /// This is intended for CPU+GPU overlap scenarios in higher-level schedulers.
    pub fn rans_encode_chunked_payload_gpu_batched(
        &self,
        inputs: &[&[u8]],
        num_lanes: usize,
        scale_bits: u8,
        chunk_size: usize,
    ) -> PzResult<Vec<(Vec<u8>, bool)>> {
        let lanes_clamped = num_lanes.clamp(1, u8::MAX as usize);
        let scale_bits = scale_bits.clamp(MIN_SCALE_BITS, MAX_SCALE_BITS);
        let ring_depth = self.rans_encode_pending_ring_depth(inputs, lanes_clamped, chunk_size);
        let mut results: Vec<Option<(Vec<u8>, bool)>> = vec![None; inputs.len()];
        let mut ring = BufferRing::new(
            (0..ring_depth)
                .map(|_| None::<(usize, PendingRansChunkedPayloadEncode)>)
                .collect(),
        );

        for (idx, input) in inputs.iter().enumerate() {
            let slot_idx = ring.acquire();
            if let Some((done_idx, done)) = ring.slots[slot_idx].take() {
                let encoded = self.complete_rans_chunked_payload_encode(done)?;
                results[done_idx] = Some((encoded, true));
            }

            if input.is_empty() {
                results[idx] = Some((Vec::new(), true));
                continue;
            }

            if !can_encode_chunked(input.len(), chunk_size) {
                let encoded = rans::encode_interleaved_n(input, lanes_clamped, scale_bits);
                results[idx] = Some((encoded, false));
                continue;
            }

            if lanes_clamped > 64 {
                return Err(PzError::Unsupported);
            }

            let submitted = self.submit_rans_chunked_payload_encode(
                input,
                lanes_clamped,
                scale_bits,
                chunk_size,
            )?;
            ring.slots[slot_idx] = Some((idx, submitted));
        }

        for slot in &mut ring.slots {
            if let Some((done_idx, done)) = slot.take() {
                let encoded = self.complete_rans_chunked_payload_encode(done)?;
                results[done_idx] = Some((encoded, true));
            }
        }

        Ok(results
            .into_iter()
            .map(|r| r.expect("all batched rans results must be populated"))
            .collect())
    }
}
