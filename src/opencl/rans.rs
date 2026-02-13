//! GPU rANS interleaved decode.

use super::*;

impl OpenClEngine {
    /// GPU-accelerated N-way interleaved rANS decode (single stream).
    ///
    /// Parses the interleaved rANS wire format, builds the decode table on
    /// the CPU, uploads everything to the GPU, and decodes all lanes in
    /// parallel (one work-item per lane).
    ///
    /// Uses 64-bit intermediate arithmetic in the kernel to avoid u32
    /// overflow on the `freq * (state >> scale_bits)` state transition.
    /// This is why this experiment is OpenCL-only (WGSL lacks u64).
    ///
    /// The cum2sym lookup table is cached in `__local` (shared) memory
    /// for fast repeated access during the decode loop.
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

        // Compile kernel at runtime (scale_bits varies)
        let source = include_str!("../../kernels/rans_decode.cl");
        let program = Program::create_and_build_from_source(&self.context, source, "-Werror")
            .map_err(|_| PzError::Unsupported)?;
        let kernel = Kernel::create(&program, "RansDecode").map_err(|_| PzError::Unsupported)?;

        let lookup_size = lookup.len().max(1);
        let freq_count = 256usize;
        let words_count = all_words.len().max(1);
        let meta_count = lane_meta.len().max(1);

        // Upload lookup table
        let mut lookup_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                lookup_size,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lookup_buf, CL_BLOCKING, 0, &lookup, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("rans_decode: upload lookup", &ev);

        // Upload freq table (as u16)
        let mut freq_buf = unsafe {
            Buffer::<u16>::create(&self.context, CL_MEM_READ_ONLY, freq_count, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut freq_buf, CL_BLOCKING, 0, &norm.freq, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload cum table (as u16)
        let mut cum_buf = unsafe {
            Buffer::<u16>::create(&self.context, CL_MEM_READ_ONLY, freq_count, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut cum_buf, CL_BLOCKING, 0, &norm.cum, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload word data
        let mut word_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                words_count,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        if !all_words.is_empty() {
            let ev = unsafe {
                self.queue
                    .enqueue_write_buffer(&mut word_buf, CL_BLOCKING, 0, &all_words, &[])
                    .map_err(|_| PzError::Unsupported)?
            };
            ev.wait().map_err(|_| PzError::Unsupported)?;
        }

        // Upload lane metadata
        let mut meta_buf = unsafe {
            Buffer::<u32>::create(&self.context, CL_MEM_READ_ONLY, meta_count, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut meta_buf, CL_BLOCKING, 0, &lane_meta, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Allocate output buffer
        let output_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                original_len,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Dispatch with workgroup size for __local memory cooperative load
        let num_lanes_arg = num_lanes as cl_uint;
        let total_out_arg = original_len as cl_uint;
        let scale_bits_arg = scale_bits as cl_uint;
        let local_table_size = 1usize << scale_bits;

        // Use a workgroup size >= num_lanes, rounded up to allow cooperative load
        let wg_size = num_lanes.min(self.max_work_group_size).max(1);
        let global_size = num_lanes.div_ceil(wg_size) * wg_size;

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&lookup_buf)
                .set_arg(&freq_buf)
                .set_arg(&cum_buf)
                .set_arg(&word_buf)
                .set_arg(&meta_buf)
                .set_arg(&output_buf)
                .set_arg(&num_lanes_arg)
                .set_arg(&total_out_arg)
                .set_arg(&scale_bits_arg)
                .set_arg_local_buffer(local_table_size)
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("rans_decode: decode kernel", &kernel_event);

        // Download result
        let mut result = vec![0u8; original_len];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("rans_decode: download output", &ev);

        Ok(result)
    }

    /// GPU-accelerated multi-block rANS decode.
    ///
    /// Splits input into independent blocks, each encoded with K-way
    /// interleaved rANS, then decodes all blocks in a single kernel
    /// launch. Total work-items = num_blocks × lanes_per_block,
    /// achieving full GPU utilization.
    ///
    /// The cum2sym lookup table is cached in `__local` memory per
    /// workgroup for fast repeated access.
    ///
    /// # Arguments
    ///
    /// * `encoded_blocks` - Slice of (encoded_data, original_len) per block.
    ///
    /// Returns concatenated decoded output from all blocks.
    pub fn rans_decode_interleaved_blocks(
        &self,
        encoded_blocks: &[(&[u8], usize)],
    ) -> PzResult<Vec<u8>> {
        if encoded_blocks.is_empty() {
            return Ok(Vec::new());
        }

        // Parse the first block's frequency table (all blocks share the same one).
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

        // Parse each block and collect metadata.
        let mut all_words: Vec<u16> = Vec::new();
        let mut all_lane_meta: Vec<u32> = Vec::new();
        let mut block_metas: Vec<u32> = Vec::new(); // 5 entries per block
        let mut total_output = 0usize;
        let mut max_lanes = 0usize;

        for &(input, original_len) in encoded_blocks {
            // Skip scale_bits + freq table
            let mut cursor = 1 + 256 * 2;
            if input.len() < cursor + 1 {
                return Err(PzError::InvalidInput);
            }
            let num_lanes = input[cursor] as usize;
            if num_lanes == 0 {
                return Err(PzError::InvalidInput);
            }
            cursor += 1;
            max_lanes = max_lanes.max(num_lanes);

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

            // Record block metadata
            let lane_meta_offset = all_lane_meta.len() / 3;
            block_metas.push(total_output as u32); // output_offset
            block_metas.push(original_len as u32); // output_len
            block_metas.push(num_lanes as u32); // lanes_per_block
            block_metas.push(0u32); // word_data_offset (unused)
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

        // Compile multi-block kernel
        let source = include_str!("../../kernels/rans_decode_blocks.cl");
        let program = Program::create_and_build_from_source(&self.context, source, "-Werror")
            .map_err(|_| PzError::Unsupported)?;
        let kernel =
            Kernel::create(&program, "RansDecodeBlocks").map_err(|_| PzError::Unsupported)?;

        let lookup_size = lookup.len().max(1);
        let freq_count = 256usize;
        let words_count = all_words.len().max(1);
        let lane_meta_count = all_lane_meta.len().max(1);
        let block_meta_count = block_metas.len().max(1);

        // Upload lookup table
        let mut lookup_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                lookup_size,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lookup_buf, CL_BLOCKING, 0, &lookup, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("rans_decode_blocks: upload lookup", &ev);

        // Upload freq + cum tables
        let mut freq_buf = unsafe {
            Buffer::<u16>::create(&self.context, CL_MEM_READ_ONLY, freq_count, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut freq_buf, CL_BLOCKING, 0, &norm.freq, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        let mut cum_buf = unsafe {
            Buffer::<u16>::create(&self.context, CL_MEM_READ_ONLY, freq_count, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut cum_buf, CL_BLOCKING, 0, &norm.cum, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload word data
        let mut word_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                words_count,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        if !all_words.is_empty() {
            let ev = unsafe {
                self.queue
                    .enqueue_write_buffer(&mut word_buf, CL_BLOCKING, 0, &all_words, &[])
                    .map_err(|_| PzError::Unsupported)?
            };
            ev.wait().map_err(|_| PzError::Unsupported)?;
        }

        // Upload block metadata
        let mut block_meta_buf = unsafe {
            Buffer::<u32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                block_meta_count,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut block_meta_buf, CL_BLOCKING, 0, &block_metas, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload lane metadata
        let mut lane_meta_buf = unsafe {
            Buffer::<u32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                lane_meta_count,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lane_meta_buf, CL_BLOCKING, 0, &all_lane_meta, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Allocate output buffer
        let output_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                total_output,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Dispatch: one workgroup per block, max_lanes work-items per workgroup
        let wg_size = max_lanes.min(self.max_work_group_size).max(1);
        let global_size = num_blocks * wg_size;
        let num_blocks_arg = num_blocks as cl_uint;
        let scale_bits_arg = scale_bits as cl_uint;
        let local_table_size = 1usize << scale_bits;

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&lookup_buf)
                .set_arg(&freq_buf)
                .set_arg(&cum_buf)
                .set_arg(&word_buf)
                .set_arg(&block_meta_buf)
                .set_arg(&lane_meta_buf)
                .set_arg(&output_buf)
                .set_arg(&num_blocks_arg)
                .set_arg(&scale_bits_arg)
                .set_arg_local_buffer(local_table_size)
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("rans_decode_blocks: decode kernel", &kernel_event);

        // Download result
        let mut result = vec![0u8; total_output];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("rans_decode_blocks: download output", &ev);

        Ok(result)
    }
}

/// Encode input into independent rANS blocks with a shared frequency table.
///
/// Computes frequency statistics across the entire input, then encodes
/// each `block_size` chunk independently using the same normalized
/// frequency table. This ensures all blocks' encoded data can be decoded
/// with a single set of GPU-resident tables.
///
/// Returns a vector of (encoded_data, original_block_len) per block.
pub fn rans_encode_blocks(
    input: &[u8],
    block_size: usize,
    num_states: usize,
    scale_bits: u8,
) -> PzResult<Vec<(Vec<u8>, usize)>> {
    let slices: Vec<&[u8]> = input.chunks(block_size).collect();
    rans_encode_block_slices(&slices, num_states, scale_bits)
}

/// Encode pre-split blocks into independent rANS streams with a shared
/// frequency table.
///
/// Like [`rans_encode_blocks`], but accepts pre-split variable-sized
/// blocks instead of splitting by a fixed block_size. Useful when the
/// block boundaries don't follow a uniform size (e.g., LZ77 match data
/// from independent compression blocks).
///
/// Returns a vector of (encoded_data, original_block_len) per block.
pub fn rans_encode_block_slices(
    blocks: &[&[u8]],
    num_states: usize,
    scale_bits: u8,
) -> PzResult<Vec<(Vec<u8>, usize)>> {
    use crate::frequency::FrequencyTable;
    use crate::rans::{
        normalize_frequencies, rans_encode_interleaved, serialize_freq_table,
        serialize_u16_le_bulk, NUM_SYMBOLS,
    };

    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    let num_states = num_states.max(1);
    let scale_bits = scale_bits.clamp(crate::rans::MIN_SCALE_BITS, crate::rans::MAX_SCALE_BITS);

    // Build shared frequency table across all blocks.
    // Note: FrequencyTable::count() replaces (not accumulates), so we must
    // concatenate all data first to get correct combined frequencies.
    let total_len: usize = blocks.iter().map(|b| b.len()).sum();
    let mut combined = Vec::with_capacity(total_len);
    for block in blocks {
        combined.extend_from_slice(block);
    }
    let mut freq = FrequencyTable::new();
    freq.count(&combined);

    let mut sb = scale_bits;
    while (1u32 << sb) < freq.used {
        sb += 1;
        if sb > crate::rans::MAX_SCALE_BITS {
            break;
        }
    }

    let norm = normalize_frequencies(&freq, sb)?;

    // Encode each block independently using the shared frequency table.
    let mut result = Vec::new();
    for chunk in blocks {
        let (word_streams, final_states) = rans_encode_interleaved(chunk, &norm, num_states);

        // Serialize interleaved format (same wire format as encode_interleaved_n)
        let total_words: usize = word_streams.iter().map(|s| s.len()).sum();
        let header_size = 1 + NUM_SYMBOLS * 2 + 1 + num_states * 4 + num_states * 4;
        let mut output = Vec::with_capacity(header_size + total_words * 2);

        output.push(sb);
        serialize_freq_table(&norm, &mut output);
        output.push(num_states as u8);

        for &state in &final_states {
            output.extend_from_slice(&state.to_le_bytes());
        }
        for stream in &word_streams {
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        }
        for stream in &word_streams {
            serialize_u16_le_bulk(stream, &mut output);
        }

        result.push((output, chunk.len()));
    }

    Ok(result)
}
