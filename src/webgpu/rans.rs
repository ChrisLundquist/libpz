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
const RANS_MAX_PENDING_RING_DEPTH: usize = 8;
const RANS_PACKED_SHARED_DECODE_MIN_PAYLOADS: usize = 8;

fn write_packed_u16_slice(words: &mut [u32], mut idx_u16: usize, mut values: &[u16]) {
    if values.is_empty() {
        return;
    }

    // Align to a u32 boundary when the write starts on the upper 16-bit lane.
    if idx_u16 % 2 == 1 {
        let word_idx = idx_u16 / 2;
        words[word_idx] = (words[word_idx] & 0x0000_FFFF) | (u32::from(values[0]) << 16);
        idx_u16 += 1;
        values = &values[1..];
        if values.is_empty() {
            return;
        }
    }

    let mut word_idx = idx_u16 / 2;
    let mut pairs = values.chunks_exact(2);
    for pair in &mut pairs {
        words[word_idx] = u32::from(pair[0]) | (u32::from(pair[1]) << 16);
        word_idx += 1;
    }

    if let Some(&tail) = pairs.remainder().first() {
        words[word_idx] = (words[word_idx] & 0xFFFF_0000) | u32::from(tail);
    }
}

fn append_u16_words_le(output: &mut Vec<u8>, words: &[u16]) {
    #[cfg(target_endian = "little")]
    {
        output.extend_from_slice(bytemuck::cast_slice(words));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &word in words {
            output.extend_from_slice(&word.to_le_bytes());
        }
    }
}

fn append_u32_words_le(output: &mut Vec<u8>, words: &[u32]) {
    #[cfg(target_endian = "little")]
    {
        output.extend_from_slice(bytemuck::cast_slice(words));
    }
    #[cfg(not(target_endian = "little"))]
    {
        for &word in words {
            output.extend_from_slice(&word.to_le_bytes());
        }
    }
}

fn copy_u16_words_from_le_bytes(dst: &mut [u16], src: &[u8]) -> PzResult<()> {
    let expected = dst.len().checked_mul(2).ok_or(PzError::InvalidInput)?;
    if src.len() != expected {
        return Err(PzError::InvalidInput);
    }
    #[cfg(target_endian = "little")]
    {
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(dst);
        dst_bytes.copy_from_slice(src);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for (word, bytes) in dst.iter_mut().zip(src.chunks_exact(2)) {
            *word = u16::from_le_bytes([bytes[0], bytes[1]]);
        }
    }
    Ok(())
}

fn pack_u16_words(words_u16: &[u16]) -> Vec<u32> {
    let mut packed = vec![0u32; words_u16.len().div_ceil(2).max(1)];
    #[cfg(target_endian = "little")]
    {
        let src_bytes: &[u8] = bytemuck::cast_slice(words_u16);
        let dst_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut packed);
        dst_bytes[..src_bytes.len()].copy_from_slice(src_bytes);
    }
    #[cfg(not(target_endian = "little"))]
    {
        for (idx, &word) in words_u16.iter().enumerate() {
            let out_idx = idx / 2;
            if idx % 2 == 0 {
                packed[out_idx] = (packed[out_idx] & 0xFFFF_0000) | u32::from(word);
            } else {
                packed[out_idx] = (packed[out_idx] & 0x0000_FFFF) | (u32::from(word) << 16);
            }
        }
    }
    packed
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

fn max_words_per_lane_for_chunk(chunk_len: usize, num_lanes: usize) -> PzResult<usize> {
    let symbols_per_lane_max = chunk_len.div_ceil(num_lanes);
    symbols_per_lane_max
        .checked_mul(2)
        .and_then(|v| v.checked_add(4))
        .ok_or(PzError::InvalidInput)
}

fn total_words_u16_for_chunks(chunk_lens: &[usize], num_lanes: usize) -> PzResult<usize> {
    chunk_lens.iter().try_fold(0usize, |acc, &chunk_len| {
        let max_words_per_lane = max_words_per_lane_for_chunk(chunk_len, num_lanes)?;
        acc.checked_add(
            num_lanes
                .checked_mul(max_words_per_lane)
                .ok_or(PzError::InvalidInput)?,
        )
        .ok_or(PzError::InvalidInput)
    })
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

struct ParsedRansChunkedPayloadDecode {
    words_packed: Vec<u32>,
    state_words: Vec<u32>,
    scale_bits: u8,
    num_lanes: usize,
    chunk_lens: Vec<usize>,
    chunk_size: usize,
    norm: Option<rans::NormalizedFreqs>,
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

        let pipeline = self.pipeline_rans_encode_for_lanes(num_lanes);
        let bg_layout = pipeline.get_bind_group_layout(0);
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
        self.dispatch(pipeline, &bg, workgroups_x, "rans_encode_chunked")?;

        Ok((words_dev, states_dev, tables_buf))
    }

    fn submit_rans_chunked_payload_encode_with_norm(
        &self,
        input: &[u8],
        num_lanes: usize,
        chunk_size: usize,
        norm: &rans::NormalizedFreqs,
    ) -> PzResult<PendingRansChunkedPayloadEncode> {
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
            self.rans_encode_chunked_gpu_with_norm(input, num_lanes, chunk_size, norm)?;

        Ok(PendingRansChunkedPayloadEncode {
            words_dev,
            states_dev,
            _tables: tables,
            norm: norm.clone(),
            chunk_lens,
            num_lanes,
            input_len: input.len(),
        })
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
        self.submit_rans_chunked_payload_encode_with_norm(input, num_lanes, chunk_size, &norm)
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
        let (words_raw, states_raw) =
            self.read_two_buffers(&pending.words_dev.buf, &pending.states_dev.buf)?;
        self.finish_rans_chunked_payload_encode_from_raw(pending, words_raw, states_raw)
    }

    fn finish_rans_chunked_payload_encode_from_raw(
        &self,
        pending: PendingRansChunkedPayloadEncode,
        words_raw: Vec<u8>,
        states_raw: Vec<u8>,
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

        let _ = words_dev;
        let _ = states_dev;
        let words_u32: &[u32] = bytemuck::cast_slice(&words_raw);
        let words_u16: &[u16] = bytemuck::cast_slice(words_u32);
        let states_u32: &[u32] = bytemuck::cast_slice(&states_raw);

        let num_chunks = chunk_lens.len();
        let header_size = 1 + NUM_SYMBOLS * 2 + 2 + 1 + num_chunks * 2;
        let mut output = Vec::with_capacity(header_size + input_len);
        output.push(norm.scale_bits);
        serialize_freq_table(&norm, &mut output);
        output.extend_from_slice(&(num_chunks as u16).to_le_bytes());
        output.push(num_lanes as u8);
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

            let state_words = &states_u32[chunk_base..chunk_base + num_lanes * 2];
            append_u32_words_le(&mut output, state_words);

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
                let read_end_u16 = read_start_u16
                    .checked_add(count)
                    .ok_or(PzError::InvalidInput)?;
                let words = words_u16
                    .get(read_start_u16..read_end_u16)
                    .ok_or(PzError::InvalidInput)?;
                append_u16_words_le(&mut output, words);
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

        let pipeline = self.pipeline_rans_decode_for_lanes(num_lanes);
        let bg_layout = pipeline.get_bind_group_layout(0);
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
        self.dispatch(pipeline, &bg, workgroups_x, "rans_decode_chunked")?;

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
            write_packed_u16_slice(&mut words_packed, write_start_u16, &stream);
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

    fn parse_rans_chunked_payload_decode(
        &self,
        input: &[u8],
        original_len: usize,
        shared_scale_bits: Option<u8>,
    ) -> PzResult<ParsedRansChunkedPayloadDecode> {
        if input.len() < 1 + NUM_SYMBOLS * 2 + 2 {
            return Err(PzError::InvalidInput);
        }

        let scale_bits = input[0];
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }
        if let Some(shared_scale_bits) = shared_scale_bits {
            if scale_bits != shared_scale_bits {
                return Err(PzError::InvalidInput);
            }
        }
        let norm = match shared_scale_bits {
            Some(_) => None,
            None => Some(deserialize_freq_table(&input[1..], scale_bits)?),
        };

        let mut cursor = 1 + NUM_SYMBOLS * 2;
        let num_chunks = read_u16_le(input, &mut cursor)? as usize;
        if num_chunks == 0 {
            return Err(PzError::InvalidInput);
        }
        if input.len() < cursor + 1 {
            return Err(PzError::InvalidInput);
        }
        let num_lanes = input[cursor] as usize;
        cursor += 1;
        if num_lanes == 0 || num_lanes > 64 {
            return Err(PzError::Unsupported);
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
        let chunk_size = chunk_lens[0].max(1);
        if !validate_chunk_grid(&chunk_lens, chunk_size, original_len) {
            // Current GPU path assumes fixed-size chunk grid (last chunk may be short).
            return Err(PzError::Unsupported);
        }

        let total_words_u16 = total_words_u16_for_chunks(&chunk_lens, num_lanes)?;
        let mut words_u16 = vec![0u16; total_words_u16];
        let mut state_words = vec![0u32; num_chunks * num_lanes * 2];
        let mut running_words_u16 = 0usize;

        for (chunk_idx, &chunk_len) in chunk_lens.iter().enumerate() {
            let chunk_base = chunk_idx * num_lanes * 2;
            for lane in 0..num_lanes {
                state_words[chunk_base + lane] = read_u32_le(input, &mut cursor)?;
            }

            let max_words_per_lane = max_words_per_lane_for_chunk(chunk_len, num_lanes)?;
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
                let stream_bytes = &input[cursor..cursor + bytes];
                cursor += bytes;

                let lane_start_u16 = running_words_u16
                    .checked_add(
                        lane.checked_mul(max_words_per_lane)
                            .ok_or(PzError::InvalidInput)?,
                    )
                    .ok_or(PzError::InvalidInput)?;
                let write_start_u16 = lane_start_u16 + (max_words_per_lane - count);
                let dst = words_u16
                    .get_mut(write_start_u16..write_start_u16 + count)
                    .ok_or(PzError::InvalidInput)?;
                copy_u16_words_from_le_bytes(dst, stream_bytes)?;
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

        Ok(ParsedRansChunkedPayloadDecode {
            words_packed: pack_u16_words(&words_u16),
            state_words,
            scale_bits,
            num_lanes,
            chunk_lens,
            chunk_size,
            norm,
        })
    }

    fn prepare_rans_chunked_payload_decode_with_shared_table(
        &self,
        input: &[u8],
        original_len: usize,
        shared_tables: Option<(&wgpu::Buffer, u8)>,
    ) -> PzResult<PreparedRansChunkedPayloadDecode> {
        let parsed = self.parse_rans_chunked_payload_decode(
            input,
            original_len,
            shared_tables.map(|(_, scale_bits)| scale_bits),
        )?;
        let total_words_u16 = total_words_u16_for_chunks(&parsed.chunk_lens, parsed.num_lanes)?;

        let words_buf = self.create_buffer_init(
            "rans_chunked_words",
            bytemuck::cast_slice(&parsed.words_packed),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let words_dev = DeviceBuf {
            buf: words_buf,
            len: total_words_u16 * 2,
        };

        let states_buf = self.create_buffer_init(
            "rans_chunked_states",
            bytemuck::cast_slice(&parsed.state_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let states_dev = DeviceBuf {
            buf: states_buf,
            len: parsed.state_words.len() * std::mem::size_of::<u32>(),
        };

        let tables = match shared_tables {
            Some((shared_buf, _)) => shared_buf.clone(),
            None => {
                let norm = parsed.norm.as_ref().ok_or(PzError::InvalidInput)?;
                let tables_words = build_tables_words(norm, parsed.scale_bits);
                self.create_buffer_init(
                    "rans_chunked_tables",
                    bytemuck::cast_slice(&tables_words),
                    wgpu::BufferUsages::STORAGE,
                )
            }
        };

        Ok(PreparedRansChunkedPayloadDecode {
            words_dev,
            states_dev,
            tables,
            params: RansChunkedDecodeParams {
                num_lanes: parsed.num_lanes,
                scale_bits: parsed.scale_bits,
                chunk_size: parsed.chunk_size,
            },
        })
    }

    fn prepare_rans_chunked_payload_decode(
        &self,
        input: &[u8],
        original_len: usize,
    ) -> PzResult<PreparedRansChunkedPayloadDecode> {
        self.prepare_rans_chunked_payload_decode_with_shared_table(input, original_len, None)
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

    fn submit_rans_chunked_payload_decode_with_shared_table(
        &self,
        input: &[u8],
        original_len: usize,
        shared_tables: &wgpu::Buffer,
        shared_scale_bits: u8,
    ) -> PzResult<PendingRansChunkedPayloadDecode> {
        let prepared = self.prepare_rans_chunked_payload_decode_with_shared_table(
            input,
            original_len,
            Some((shared_tables, shared_scale_bits)),
        )?;
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

    fn complete_rans_chunked_payload_decode_batch(
        &self,
        batch: Vec<(usize, PendingRansChunkedPayloadDecode)>,
        results: &mut [Option<Vec<u8>>],
    ) -> PzResult<()> {
        if batch.is_empty() {
            return Ok(());
        }

        let mut staging = Vec::with_capacity(batch.len());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rans_decode_batched_readback"),
            });

        for (_, pending) in &batch {
            let size = pending.output_dev.buf.size();
            let staging_buf = self.create_buffer(
                "rans_decode_batched_staging",
                size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            encoder.copy_buffer_to_buffer(&pending.output_dev.buf, 0, &staging_buf, 0, size);
            staging.push(staging_buf);
        }
        self.queue.submit(Some(encoder.finish()));

        let mut waits = Vec::with_capacity(staging.len());
        for staging_buf in &staging {
            let slice = staging_buf.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            waits.push(rx);
        }
        self.poll_wait();
        for rx in waits {
            rx.recv()
                .map_err(|_| PzError::Unsupported)?
                .map_err(|_| PzError::Unsupported)?;
        }

        for ((idx, pending), staging_buf) in batch.into_iter().zip(staging.into_iter()) {
            let slice = staging_buf.slice(..);
            let mapped = slice.get_mapped_range();
            let len = pending.output_dev.len();
            let decoded = mapped.get(..len).ok_or(PzError::InvalidInput)?.to_vec();
            drop(mapped);
            staging_buf.unmap();
            results[idx] = Some(decoded);
        }

        Ok(())
    }

    fn complete_rans_chunked_payload_encode_batch(
        &self,
        batch: Vec<(usize, PendingRansChunkedPayloadEncode)>,
        results: &mut [Option<(Vec<u8>, bool)>],
    ) -> PzResult<()> {
        if batch.is_empty() {
            return Ok(());
        }

        struct EncodeReadbackSlot {
            idx: usize,
            pending: PendingRansChunkedPayloadEncode,
            staging_words: wgpu::Buffer,
            staging_states: wgpu::Buffer,
        }

        let mut slots = Vec::with_capacity(batch.len());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rans_encode_batched_readback"),
            });
        for (idx, pending) in batch {
            let words_size = pending.words_dev.buf.size();
            let states_size = pending.states_dev.buf.size();
            let staging_words = self.create_buffer(
                "rans_encode_batched_staging_words",
                words_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            let staging_states = self.create_buffer(
                "rans_encode_batched_staging_states",
                states_size,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            );
            encoder.copy_buffer_to_buffer(&pending.words_dev.buf, 0, &staging_words, 0, words_size);
            encoder.copy_buffer_to_buffer(
                &pending.states_dev.buf,
                0,
                &staging_states,
                0,
                states_size,
            );
            slots.push(EncodeReadbackSlot {
                idx,
                pending,
                staging_words,
                staging_states,
            });
        }
        self.queue.submit(Some(encoder.finish()));

        let mut waits = Vec::with_capacity(slots.len() * 2);
        for slot in &slots {
            let words_slice = slot.staging_words.slice(..);
            let states_slice = slot.staging_states.slice(..);
            let (tx_words, rx_words) = std::sync::mpsc::channel();
            let (tx_states, rx_states) = std::sync::mpsc::channel();
            words_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx_words.send(result);
            });
            states_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx_states.send(result);
            });
            waits.push(rx_words);
            waits.push(rx_states);
        }
        self.poll_wait();
        for rx in waits {
            rx.recv()
                .map_err(|_| PzError::Unsupported)?
                .map_err(|_| PzError::Unsupported)?;
        }

        for slot in slots {
            let words_slice = slot.staging_words.slice(..);
            let states_slice = slot.staging_states.slice(..);
            let words_raw = words_slice.get_mapped_range().to_vec();
            let states_raw = states_slice.get_mapped_range().to_vec();
            slot.staging_words.unmap();
            slot.staging_states.unmap();

            let encoded = self.finish_rans_chunked_payload_encode_from_raw(
                slot.pending,
                words_raw,
                states_raw,
            )?;
            results[slot.idx] = Some((encoded, true));
        }

        Ok(())
    }

    /// Decode chunked interleaved rANS payload (`rans::encode_chunked`) on GPU.
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

        // Reserve headroom but allow deeper in-flight queues than LZ77 so rANS
        // payload upload/readback can overlap across more batches.
        let budget = (self.gpu_memory_budget() * 3) / 4;
        (budget / per_slot)
            .clamp(1, RANS_MAX_PENDING_RING_DEPTH)
            .min(max_depth)
    }

    fn rans_decode_chunked_payload_gpu_batched_impl(
        &self,
        inputs: &[(&[u8], usize)],
        shared_tables: Option<(&wgpu::Buffer, u8)>,
    ) -> PzResult<Vec<Vec<u8>>> {
        let ring_depth = self.rans_decode_pending_ring_depth(inputs);
        let mut results: Vec<Option<Vec<u8>>> = vec![None; inputs.len()];
        let mut done_batch: Vec<(usize, PendingRansChunkedPayloadDecode)> =
            Vec::with_capacity(ring_depth);
        let mut ring = BufferRing::new(
            (0..ring_depth)
                .map(|_| None::<(usize, PendingRansChunkedPayloadDecode)>)
                .collect(),
        );

        for (idx, &(input, original_len)) in inputs.iter().enumerate() {
            let slot_idx = ring.acquire();
            if let Some((done_idx, done)) = ring.slots[slot_idx].take() {
                done_batch.push((done_idx, done));
            }

            if original_len == 0 {
                results[idx] = Some(Vec::new());
                if done_batch.len() >= ring_depth {
                    let batch = std::mem::take(&mut done_batch);
                    self.complete_rans_chunked_payload_decode_batch(batch, &mut results)?;
                }
                continue;
            }

            let submitted = if let Some((shared_tables, shared_scale_bits)) = shared_tables {
                let payload_scale_bits = *input.first().ok_or(PzError::InvalidInput)?;
                if payload_scale_bits != shared_scale_bits {
                    return Err(PzError::InvalidInput);
                }
                self.submit_rans_chunked_payload_decode_with_shared_table(
                    input,
                    original_len,
                    shared_tables,
                    shared_scale_bits,
                )?
            } else {
                self.submit_rans_chunked_payload_decode(input, original_len)?
            };
            ring.slots[slot_idx] = Some((idx, submitted));

            if done_batch.len() >= ring_depth {
                let batch = std::mem::take(&mut done_batch);
                self.complete_rans_chunked_payload_decode_batch(batch, &mut results)?;
            }
        }

        for slot in &mut ring.slots {
            if let Some((done_idx, done)) = slot.take() {
                done_batch.push((done_idx, done));
            }
        }
        if !done_batch.is_empty() {
            self.complete_rans_chunked_payload_decode_batch(done_batch, &mut results)?;
        }

        Ok(results
            .into_iter()
            .map(|r| r.expect("all batched rans decode results must be populated"))
            .collect())
    }

    fn rans_decode_chunked_gpu_with_chunk_meta(
        &self,
        words: &DeviceBuf,
        states: &DeviceBuf,
        tables: &wgpu::Buffer,
        chunk_meta_words: &[u32],
        output_len: usize,
        params: RansChunkedDecodeParams,
    ) -> PzResult<DeviceBuf> {
        let RansChunkedDecodeParams {
            num_lanes,
            scale_bits,
            chunk_size: _,
        } = params;
        if self.max_storage_buffers_per_shader_stage < RANS_STORAGE_BINDINGS_PER_STAGE {
            return Err(PzError::Unsupported);
        }
        if output_len == 0
            || chunk_meta_words.is_empty()
            || num_lanes == 0
            || num_lanes > 64
            || !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits)
        {
            return Err(PzError::InvalidInput);
        }
        if !chunk_meta_words.len().is_multiple_of(4) {
            return Err(PzError::InvalidInput);
        }

        let num_chunks = chunk_meta_words.len() / 4;
        let output_u32_words = output_len.div_ceil(4).max(1);
        let output_buf = self.create_buffer_init(
            "rans_decode_output_packed",
            &vec![0u8; output_u32_words * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let output = DeviceBuf {
            buf: output_buf,
            len: output_len,
        };

        let chunk_meta_buf = self.create_buffer_init(
            "rans_decode_chunk_meta_packed",
            bytemuck::cast_slice(chunk_meta_words),
            wgpu::BufferUsages::STORAGE,
        );
        let params_words = [num_chunks as u32, num_lanes as u32, scale_bits as u32, 0];
        let params_buf = self.create_buffer_init(
            "rans_decode_params_packed",
            bytemuck::cast_slice(&params_words),
            wgpu::BufferUsages::UNIFORM,
        );

        let pipeline = self.pipeline_rans_decode_for_lanes(num_lanes);
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rans_decode_bg_packed"),
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
        self.dispatch(pipeline, &bg, workgroups_x, "rans_decode_chunked_packed")?;
        Ok(output)
    }

    fn rans_decode_chunked_payload_gpu_batched_shared_table_packed(
        &self,
        inputs: &[(&[u8], usize)],
        shared_scale_bits: u8,
        shared_tables: &wgpu::Buffer,
    ) -> PzResult<Option<Vec<Vec<u8>>>> {
        struct PackedDecodeInput {
            idx: usize,
            output_len: usize,
            chunk_lens: Vec<usize>,
            words_u16_len: usize,
            words_packed: Vec<u32>,
            state_words: Vec<u32>,
        }

        let mut results: Vec<Option<Vec<u8>>> = vec![None; inputs.len()];
        let mut packed_inputs = Vec::new();
        let mut expected_lanes = None::<usize>;
        let mut total_chunks = 0usize;
        let mut total_words_u16 = 0usize;
        let mut total_states_u32 = 0usize;
        let mut total_output_len = 0usize;

        for (idx, &(payload, output_len)) in inputs.iter().enumerate() {
            if output_len == 0 {
                results[idx] = Some(Vec::new());
                continue;
            }

            let parsed = self.parse_rans_chunked_payload_decode(
                payload,
                output_len,
                Some(shared_scale_bits),
            )?;
            if let Some(num_lanes) = expected_lanes {
                if parsed.num_lanes != num_lanes {
                    return Ok(None);
                }
            } else {
                expected_lanes = Some(parsed.num_lanes);
            }

            let words_u16_len = total_words_u16_for_chunks(&parsed.chunk_lens, parsed.num_lanes)?;
            let parsed_words_u16_len = parsed
                .words_packed
                .len()
                .checked_mul(2)
                .ok_or(PzError::InvalidInput)?;
            if parsed_words_u16_len < words_u16_len {
                return Err(PzError::InvalidInput);
            }

            total_chunks = total_chunks
                .checked_add(parsed.chunk_lens.len())
                .ok_or(PzError::InvalidInput)?;
            total_words_u16 = total_words_u16
                .checked_add(words_u16_len)
                .ok_or(PzError::InvalidInput)?;
            total_states_u32 = total_states_u32
                .checked_add(parsed.state_words.len())
                .ok_or(PzError::InvalidInput)?;
            total_output_len = total_output_len
                .checked_add(output_len)
                .ok_or(PzError::InvalidInput)?;
            packed_inputs.push(PackedDecodeInput {
                idx,
                output_len,
                chunk_lens: parsed.chunk_lens,
                words_u16_len,
                words_packed: parsed.words_packed,
                state_words: parsed.state_words,
            });
        }

        if packed_inputs.is_empty() {
            return Ok(Some(
                results
                    .into_iter()
                    .map(|r| r.expect("all packed rans decode results must be populated"))
                    .collect(),
            ));
        }

        let num_lanes = expected_lanes.ok_or(PzError::InvalidInput)?;
        let mut chunk_meta_words = Vec::with_capacity(total_chunks * 4);
        let mut all_words_u16 = Vec::with_capacity(total_words_u16);
        let mut all_state_words = Vec::with_capacity(total_states_u32);

        let mut running_output_offset = 0usize;
        let mut running_words_u16 = 0usize;
        let mut running_state_u32 = 0usize;
        for packed in &packed_inputs {
            let mut block_output_offset = 0usize;
            let mut block_words_u16 = 0usize;
            let mut block_state_u32 = 0usize;
            for &chunk_len in &packed.chunk_lens {
                let max_words_per_lane = max_words_per_lane_for_chunk(chunk_len, num_lanes)?;
                let chunk_words_u16 = num_lanes
                    .checked_mul(max_words_per_lane)
                    .ok_or(PzError::InvalidInput)?;
                let chunk_state_u32 = num_lanes.checked_mul(2).ok_or(PzError::InvalidInput)?;

                chunk_meta_words.extend_from_slice(&[
                    (running_output_offset + block_output_offset) as u32,
                    chunk_len as u32,
                    (running_words_u16 + block_words_u16) as u32,
                    (running_state_u32 + block_state_u32) as u32,
                ]);

                block_output_offset = block_output_offset
                    .checked_add(chunk_len)
                    .ok_or(PzError::InvalidInput)?;
                block_words_u16 = block_words_u16
                    .checked_add(chunk_words_u16)
                    .ok_or(PzError::InvalidInput)?;
                block_state_u32 = block_state_u32
                    .checked_add(chunk_state_u32)
                    .ok_or(PzError::InvalidInput)?;
            }
            if block_output_offset != packed.output_len || block_words_u16 != packed.words_u16_len {
                return Err(PzError::InvalidInput);
            }

            let words_u16: &[u16] = bytemuck::cast_slice(&packed.words_packed);
            let words_slice = words_u16
                .get(..packed.words_u16_len)
                .ok_or(PzError::InvalidInput)?;
            all_words_u16.extend_from_slice(words_slice);
            all_state_words.extend_from_slice(&packed.state_words);

            running_output_offset = running_output_offset
                .checked_add(packed.output_len)
                .ok_or(PzError::InvalidInput)?;
            running_words_u16 = running_words_u16
                .checked_add(packed.words_u16_len)
                .ok_or(PzError::InvalidInput)?;
            running_state_u32 = running_state_u32
                .checked_add(packed.state_words.len())
                .ok_or(PzError::InvalidInput)?;
        }

        if running_output_offset != total_output_len
            || running_words_u16 != total_words_u16
            || running_state_u32 != total_states_u32
        {
            return Err(PzError::InvalidInput);
        }

        let all_words_packed = pack_u16_words(&all_words_u16);

        let words_buf = self.create_buffer_init(
            "rans_decode_words_packed_batch",
            bytemuck::cast_slice(&all_words_packed),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let words_dev = DeviceBuf {
            buf: words_buf,
            len: total_words_u16 * 2,
        };
        let states_buf = self.create_buffer_init(
            "rans_decode_states_packed_batch",
            bytemuck::cast_slice(&all_state_words),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let states_dev = DeviceBuf {
            buf: states_buf,
            len: total_states_u32 * std::mem::size_of::<u32>(),
        };

        let output = self.rans_decode_chunked_gpu_with_chunk_meta(
            &words_dev,
            &states_dev,
            shared_tables,
            &chunk_meta_words,
            total_output_len,
            RansChunkedDecodeParams {
                num_lanes,
                scale_bits: shared_scale_bits,
                chunk_size: 0,
            },
        )?;
        let decoded_all = output.read_to_host(self)?;

        let mut cursor = 0usize;
        for packed in packed_inputs {
            let end = cursor
                .checked_add(packed.output_len)
                .ok_or(PzError::InvalidInput)?;
            let block = decoded_all.get(cursor..end).ok_or(PzError::InvalidInput)?;
            results[packed.idx] = Some(block.to_vec());
            cursor = end;
        }
        if cursor != decoded_all.len() {
            return Err(PzError::InvalidInput);
        }

        Ok(Some(
            results
                .into_iter()
                .map(|r| r.expect("all packed rans decode results must be populated"))
                .collect(),
        ))
    }

    /// Batched GPU chunked rANS decode with ring-buffered submit/readback.
    ///
    /// Inputs are `(payload, original_len)` tuples where `payload` is in
    /// `rans::encode_chunked` wire format.
    pub fn rans_decode_chunked_payload_gpu_batched(
        &self,
        inputs: &[(&[u8], usize)],
    ) -> PzResult<Vec<Vec<u8>>> {
        self.rans_decode_chunked_payload_gpu_batched_impl(inputs, None)
    }

    /// Batched GPU chunked rANS decode that reuses one precomputed table
    /// across all chunked payloads in the batch.
    ///
    /// Callers must ensure payloads were encoded with a normalization that
    /// matches `shared_table_seed` and share the same `scale_bits`.
    pub fn rans_decode_chunked_payload_gpu_batched_shared_table(
        &self,
        inputs: &[(&[u8], usize)],
        shared_table_seed: &[u8],
    ) -> PzResult<Vec<Vec<u8>>> {
        if shared_table_seed.is_empty() {
            return self.rans_decode_chunked_payload_gpu_batched(inputs);
        }
        let _ = shared_table_seed;

        let (first_payload, _) =
            if let Some((payload, output_len)) = inputs.iter().find(|(_, len)| *len > 0) {
                (*payload, *output_len)
            } else {
                return Ok(vec![Vec::new(); inputs.len()]);
            };
        let shared_scale_bits = *first_payload.first().ok_or(PzError::InvalidInput)?;
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&shared_scale_bits) {
            return Err(PzError::InvalidInput);
        }
        let norm = deserialize_freq_table(&first_payload[1..], shared_scale_bits)?;
        let tables_words = build_tables_words(&norm, shared_scale_bits);
        let tables = self.create_buffer_init(
            "rans_chunked_tables_shared_decode",
            bytemuck::cast_slice(&tables_words),
            wgpu::BufferUsages::STORAGE,
        );
        let non_empty_payloads = inputs
            .iter()
            .filter(|(_, output_len)| *output_len > 0)
            .count();
        if non_empty_payloads >= RANS_PACKED_SHARED_DECODE_MIN_PAYLOADS {
            if let Some(packed) = self.rans_decode_chunked_payload_gpu_batched_shared_table_packed(
                inputs,
                shared_scale_bits,
                &tables,
            )? {
                return Ok(packed);
            }
        }
        self.rans_decode_chunked_payload_gpu_batched_impl(
            inputs,
            Some((&tables, shared_scale_bits)),
        )
    }

    /// Encode chunked interleaved rANS payload on GPU using CPU wire format.
    ///
    /// Returns `(encoded, used_chunked)` with fallback behavior that mirrors
    /// `rans::encode_chunked` metadata limits.
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

        // Reserve headroom but allow deeper in-flight queues than LZ77 so rANS
        // payload upload/readback can overlap across more batches.
        let budget = (self.gpu_memory_budget() * 3) / 4;
        (budget / per_slot)
            .clamp(1, RANS_MAX_PENDING_RING_DEPTH)
            .min(max_depth)
    }

    fn rans_encode_chunked_payload_gpu_batched_impl(
        &self,
        inputs: &[&[u8]],
        num_lanes: usize,
        scale_bits: u8,
        chunk_size: usize,
        shared_norm: Option<&rans::NormalizedFreqs>,
    ) -> PzResult<Vec<(Vec<u8>, bool)>> {
        let lanes_clamped = num_lanes.clamp(1, u8::MAX as usize);
        let scale_bits = scale_bits.clamp(MIN_SCALE_BITS, MAX_SCALE_BITS);
        let ring_depth = self.rans_encode_pending_ring_depth(inputs, lanes_clamped, chunk_size);
        let mut results: Vec<Option<(Vec<u8>, bool)>> = vec![None; inputs.len()];
        let mut done_batch: Vec<(usize, PendingRansChunkedPayloadEncode)> =
            Vec::with_capacity(ring_depth);
        let mut ring = BufferRing::new(
            (0..ring_depth)
                .map(|_| None::<(usize, PendingRansChunkedPayloadEncode)>)
                .collect(),
        );

        for (idx, input) in inputs.iter().enumerate() {
            let slot_idx = ring.acquire();
            if let Some((done_idx, done)) = ring.slots[slot_idx].take() {
                done_batch.push((done_idx, done));
            }

            if input.is_empty() {
                results[idx] = Some((Vec::new(), true));
                if done_batch.len() >= ring_depth {
                    let batch = std::mem::take(&mut done_batch);
                    self.complete_rans_chunked_payload_encode_batch(batch, &mut results)?;
                }
                continue;
            }

            if !can_encode_chunked(input.len(), chunk_size) {
                let encoded = rans::encode_interleaved_n(input, lanes_clamped, scale_bits);
                results[idx] = Some((encoded, false));
                if done_batch.len() >= ring_depth {
                    let batch = std::mem::take(&mut done_batch);
                    self.complete_rans_chunked_payload_encode_batch(batch, &mut results)?;
                }
                continue;
            }

            if lanes_clamped > 64 {
                return Err(PzError::Unsupported);
            }

            let submitted = if let Some(norm) = shared_norm {
                self.submit_rans_chunked_payload_encode_with_norm(
                    input,
                    lanes_clamped,
                    chunk_size,
                    norm,
                )?
            } else {
                self.submit_rans_chunked_payload_encode(
                    input,
                    lanes_clamped,
                    scale_bits,
                    chunk_size,
                )?
            };
            ring.slots[slot_idx] = Some((idx, submitted));

            if done_batch.len() >= ring_depth {
                let batch = std::mem::take(&mut done_batch);
                self.complete_rans_chunked_payload_encode_batch(batch, &mut results)?;
            }
        }

        for slot in &mut ring.slots {
            if let Some((done_idx, done)) = slot.take() {
                done_batch.push((done_idx, done));
            }
        }
        if !done_batch.is_empty() {
            self.complete_rans_chunked_payload_encode_batch(done_batch, &mut results)?;
        }

        Ok(results
            .into_iter()
            .map(|r| r.expect("all batched rans results must be populated"))
            .collect())
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
        self.rans_encode_chunked_payload_gpu_batched_impl(
            inputs, num_lanes, scale_bits, chunk_size, None,
        )
    }

    /// Batched GPU chunked rANS encode that reuses one normalized table for all
    /// chunked payloads in the batch.
    ///
    /// Useful for nvCOMP-style independent-block probes where setup overhead
    /// dominates and blocks share a common input distribution.
    pub fn rans_encode_chunked_payload_gpu_batched_shared_table(
        &self,
        inputs: &[&[u8]],
        shared_table_seed: &[u8],
        num_lanes: usize,
        scale_bits: u8,
        chunk_size: usize,
    ) -> PzResult<Vec<(Vec<u8>, bool)>> {
        if shared_table_seed.is_empty() {
            return self.rans_encode_chunked_payload_gpu_batched(
                inputs, num_lanes, scale_bits, chunk_size,
            );
        }

        let scale_bits = scale_bits.clamp(MIN_SCALE_BITS, MAX_SCALE_BITS);
        let mut freq = FrequencyTable::new();
        freq.count(shared_table_seed);
        let norm = normalize_frequencies(&freq, scale_bits)?;
        self.rans_encode_chunked_payload_gpu_batched_impl(
            inputs,
            num_lanes,
            scale_bits,
            chunk_size,
            Some(&norm),
        )
    }
}
