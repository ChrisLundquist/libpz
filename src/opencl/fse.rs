//! FSE (tANS) GPU decode via OpenCL.
//!
//! Port of the WebGPU FSE decode kernel. Each work-item decodes one
//! interleaved stream — parallelism comes from decoding N streams
//! simultaneously.

use super::*;

impl OpenClEngine {
    /// GPU-accelerated FSE decode of N-way interleaved streams.
    ///
    /// Takes the serialized interleaved FSE data (as produced by
    /// `fse::encode_interleaved()`) and decodes it on the GPU.
    /// Each stream is decoded by one GPU work-item.
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

        // Concatenate all bitstreams.
        let mut all_bitstream_data: Vec<u8> = Vec::new();
        let mut stream_meta_host: Vec<u32> = Vec::with_capacity(num_streams * 4);

        for stream in &streams {
            let byte_offset = all_bitstream_data.len() as u32;
            all_bitstream_data.extend_from_slice(&stream.bitstream);

            stream_meta_host.push(stream.initial_state);
            stream_meta_host.push(stream.total_bits);
            stream_meta_host.push(byte_offset);
            stream_meta_host.push(stream.num_symbols);
        }

        // Pad bitstream to avoid out-of-bounds reads.
        all_bitstream_data.extend_from_slice(&[0u8; 4]);

        // Upload decode table
        let mut decode_table_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                packed_table.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut decode_table_buf, CL_BLOCKING, 0, &packed_table, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload bitstream data
        let mut bitstream_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                all_bitstream_data.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut bitstream_buf, CL_BLOCKING, 0, &all_bitstream_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload stream metadata
        let mut stream_meta_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                stream_meta_host.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut stream_meta_buf, CL_BLOCKING, 0, &stream_meta_host, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Output buffer: u32-packed bytes, zero-initialized
        let output_u32_count = original_len.div_ceil(4);
        let mut output_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_u32_count,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let zeros = vec![0u32; output_u32_count];
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Dispatch kernel: one work-item per stream
        let num_streams_arg = num_streams as cl_uint;
        let total_output_len_arg = original_len as cl_uint;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_fse_decode)
                .set_arg(&decode_table_buf)
                .set_arg(&bitstream_buf)
                .set_arg(&stream_meta_buf)
                .set_arg(&output_buf)
                .set_arg(&num_streams_arg)
                .set_arg(&total_output_len_arg)
                .set_global_work_size(num_streams)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("fse_decode", &kernel_event);

        // Download output as u32 array and extract bytes
        let mut output_data = vec![0u32; output_u32_count];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Extract bytes from u32 words (little-endian packed)
        let mut result = Vec::with_capacity(original_len);
        for &word in &output_data {
            let bytes = word.to_le_bytes();
            for &b in &bytes {
                if result.len() < original_len {
                    result.push(b);
                }
            }
        }

        Ok(result)
    }
}
