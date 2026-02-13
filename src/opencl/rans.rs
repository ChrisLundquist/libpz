//! GPU rANS interleaved decode.

use super::*;

impl OpenClEngine {
    /// GPU-accelerated N-way interleaved rANS decode.
    ///
    /// Parses the interleaved rANS wire format, builds the decode table on
    /// the CPU, uploads everything to the GPU, and decodes all lanes in
    /// parallel (one work-item per lane).
    ///
    /// Uses 64-bit intermediate arithmetic in the kernel to avoid u32
    /// overflow on the `freq * (state >> scale_bits)` state transition.
    /// This is why this experiment is OpenCL-only (WGSL lacks u64).
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

        // Dispatch: one work-item per lane
        let num_lanes_arg = num_lanes as cl_uint;
        let total_out_arg = original_len as cl_uint;
        let scale_bits_arg = scale_bits as cl_uint;

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
                .set_global_work_size(num_lanes)
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
}
