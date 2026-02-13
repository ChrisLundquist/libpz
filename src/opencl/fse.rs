//! FSE (tANS) GPU encode and decode via OpenCL.
//!
//! **Decode**: Port of the WebGPU FSE decode kernel. Each work-item
//! decodes one interleaved stream — parallelism comes from decoding
//! N streams simultaneously.
//!
//! **Encode**: Each work-item encodes one lane of N-way interleaved
//! FSE. Two-phase per lane: reverse symbol scan → forward bit-pack.
//! All operations are 32-bit (no u64), matching the WebGPU kernel.

use super::*;

impl OpenClEngine {
    /// GPU-accelerated FSE decode of N-way interleaved streams.
    ///
    /// Takes the serialized interleaved FSE data (as produced by
    /// `fse::encode_interleaved()`) and decodes it on the GPU.
    /// Each stream is decoded by one GPU work-item.
    ///
    /// The decode table is cached in `__local` (shared) memory when it
    /// fits (≤4096 entries = 16KB), significantly reducing global memory
    /// traffic in the hot decode loop.
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

        // Dispatch kernel: one work-item per stream, with __local decode table
        let num_streams_arg = num_streams as cl_uint;
        let total_output_len_arg = original_len as cl_uint;
        let table_size_arg = table_size;
        // __local memory: table_size entries × 4 bytes each (u32 packed)
        let local_table_bytes = (table_size as usize) * std::mem::size_of::<u32>();
        // Workgroup size for cooperative __local load (at least num_streams, capped)
        let wg_size = num_streams.min(self.max_work_group_size).max(1);
        let global_size = num_streams.div_ceil(wg_size) * wg_size;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_fse_decode)
                .set_arg(&decode_table_buf)
                .set_arg(&bitstream_buf)
                .set_arg(&stream_meta_buf)
                .set_arg(&output_buf)
                .set_arg(&num_streams_arg)
                .set_arg(&total_output_len_arg)
                .set_arg(&table_size_arg)
                .set_arg_local_buffer(local_table_bytes)
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size)
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

    /// GPU-accelerated multi-block FSE decode in a single kernel launch.
    ///
    /// All blocks must share the same decode table (encoded with a shared
    /// frequency table via [`fse_encode_block_slices`]). Each workgroup
    /// handles one block, with one work-item per interleaved stream.
    ///
    /// Total work-items = num_blocks × streams_per_block.
    ///
    /// # Arguments
    ///
    /// * `encoded_blocks` - Slice of (encoded_data, original_len) per block.
    ///
    /// Returns concatenated decoded output from all blocks.
    pub fn fse_decode_blocks(&self, encoded_blocks: &[(&[u8], usize)]) -> PzResult<Vec<u8>> {
        if encoded_blocks.is_empty() {
            return Ok(Vec::new());
        }

        // Parse first block's freq table (all blocks share the same one).
        let first = encoded_blocks[0].0;
        let freq_table_bytes = 256 * 2;
        let min_header = 1 + freq_table_bytes + 1;
        if first.len() < min_header {
            return Err(PzError::InvalidInput);
        }

        let accuracy_log = first[0];
        if !(5..=12).contains(&accuracy_log) {
            return Err(PzError::InvalidInput);
        }

        let mut norm_freq = [0u16; 256];
        for (i, freq) in norm_freq.iter_mut().enumerate() {
            let offset = 1 + i * 2;
            *freq = u16::from_le_bytes([first[offset], first[offset + 1]]);
        }

        let table_size = 1u32 << accuracy_log;
        let sum: u32 = norm_freq.iter().map(|&f| f as u32).sum();
        if sum != table_size {
            return Err(PzError::InvalidInput);
        }

        // Build decode table on CPU.
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

        // Parse each block: collect stream metadata and bitstreams.
        let mut all_bitstream_data: Vec<u8> = Vec::new();
        let mut all_stream_meta: Vec<u32> = Vec::new();
        let mut block_metas: Vec<u32> = Vec::new(); // 3 entries per block
        let mut total_output = 0usize;
        let mut max_streams = 0usize;

        for &(input, original_len) in encoded_blocks {
            // Skip accuracy_log + freq table
            let pos = 1 + freq_table_bytes;
            if input.len() < pos + 1 {
                return Err(PzError::InvalidInput);
            }
            let num_streams = input[pos] as usize;
            if num_streams == 0 {
                return Err(PzError::InvalidInput);
            }
            max_streams = max_streams.max(num_streams);

            let stream_meta_offset = all_stream_meta.len() / 4;

            // Block metadata: [output_offset, output_len, stream_meta_offset]
            block_metas.push(total_output as u32);
            block_metas.push(original_len as u32);
            block_metas.push(stream_meta_offset as u32);

            let base_count = original_len / num_streams;
            let extra = original_len % num_streams;

            let mut cursor = pos + 1;
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
                let byte_offset = all_bitstream_data.len() as u32;
                all_bitstream_data.extend_from_slice(&input[cursor..cursor + bitstream_len]);
                cursor += bitstream_len;

                let num_symbols = (base_count + if lane < extra { 1 } else { 0 }) as u32;

                // Stream metadata: [initial_state, total_bits, bitstream_byte_offset, num_symbols]
                all_stream_meta.push(initial_state);
                all_stream_meta.push(total_bits);
                all_stream_meta.push(byte_offset);
                all_stream_meta.push(num_symbols);
            }

            total_output += original_len;
        }

        if total_output == 0 {
            return Ok(Vec::new());
        }

        // Pad bitstream to avoid OOB reads.
        all_bitstream_data.extend_from_slice(&[0u8; 4]);

        let num_blocks = encoded_blocks.len();

        // Compile multi-block kernel at runtime.
        let source = include_str!("../../kernels/fse_decode_blocks.cl");
        let program = Program::create_and_build_from_source(&self.context, source, "-Werror")
            .map_err(|_| PzError::Unsupported)?;
        let kernel =
            Kernel::create(&program, "FseDecodeBlocks").map_err(|_| PzError::Unsupported)?;

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
        let bs_len = all_bitstream_data.len().max(1);
        let mut bitstream_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, bs_len, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        if !all_bitstream_data.is_empty() {
            let ev = unsafe {
                self.queue
                    .enqueue_write_buffer(
                        &mut bitstream_buf,
                        CL_BLOCKING,
                        0,
                        &all_bitstream_data,
                        &[],
                    )
                    .map_err(|_| PzError::Unsupported)?
            };
            ev.wait().map_err(|_| PzError::Unsupported)?;
        }

        // Upload block metadata
        let bm_len = block_metas.len().max(1);
        let mut block_meta_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, bm_len, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut block_meta_buf, CL_BLOCKING, 0, &block_metas, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload stream metadata
        let sm_len = all_stream_meta.len().max(1);
        let mut stream_meta_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, sm_len, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut stream_meta_buf, CL_BLOCKING, 0, &all_stream_meta, &[])
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
            .map_err(|_| PzError::Unsupported)?
        };

        // Dispatch: one workgroup per block, max_streams work-items per workgroup
        let wg_size = max_streams.min(self.max_work_group_size).max(1);
        let global_size = num_blocks * wg_size;
        let num_blocks_arg = num_blocks as cl_uint;
        let max_streams_arg = max_streams as cl_uint;
        let table_size_arg = table_size;
        let local_table_bytes = (table_size as usize) * std::mem::size_of::<u32>();

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&decode_table_buf)
                .set_arg(&bitstream_buf)
                .set_arg(&block_meta_buf)
                .set_arg(&stream_meta_buf)
                .set_arg(&output_buf)
                .set_arg(&num_blocks_arg)
                .set_arg(&max_streams_arg)
                .set_arg(&table_size_arg)
                .set_arg_local_buffer(local_table_bytes)
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("fse_decode_blocks: decode kernel", &kernel_event);

        // Download result
        let mut result = vec![0u8; total_output];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        Ok(result)
    }

    /// GPU-accelerated N-way interleaved FSE encode (single stream).
    ///
    /// Computes frequency normalization and encode table on the CPU (small
    /// O(table_size) work), then dispatches the GPU kernel where each
    /// work-item encodes one lane.
    ///
    /// Returns the serialized interleaved FSE wire format (identical to
    /// `fse::encode_interleaved_n()`), decodable by both CPU
    /// `fse::decode_interleaved()` and GPU `fse_decode()`.
    pub fn fse_encode_interleaved_gpu(
        &self,
        input: &[u8],
        num_states: usize,
        accuracy_log: u8,
    ) -> PzResult<Vec<u8>> {
        use crate::frequency::FrequencyTable;
        use crate::fse::{
            build_gpu_encode_table, normalize_frequencies, FREQ_TABLE_BYTES, MAX_ACCURACY_LOG,
            MIN_ACCURACY_LOG,
        };

        if input.is_empty() {
            return Ok(Vec::new());
        }

        let num_states = num_states.max(1);
        let accuracy_log = accuracy_log.clamp(MIN_ACCURACY_LOG, MAX_ACCURACY_LOG);

        // CPU: frequency counting + normalization + encode table
        let mut freq = FrequencyTable::new();
        freq.count(input);

        let mut al = accuracy_log;
        while (1u32 << al) < freq.used {
            al += 1;
            if al > MAX_ACCURACY_LOG {
                break;
            }
        }

        let norm = normalize_frequencies(&freq, al)?;
        let table_size = 1u32 << al;
        let packed_encode_table = build_gpu_encode_table(&norm);

        // Compute worst-case output: each symbol produces at most
        // accuracy_log bits. Worst case bytes ≈ (symbols_per_lane * al + 7) / 8.
        // Add margin for chunk storage in phase 1 (4 bytes per symbol).
        let symbols_per_lane = (input.len() + num_states - 1) / num_states;
        // Phase 1 stores chunks as u32 in the output buffer, phase 2 overwrites
        // with the actual bitstream. Need max(4 * symbols_per_lane, bitstream).
        let max_output_bytes_per_lane =
            (symbols_per_lane * 4).max((symbols_per_lane * al as usize).div_ceil(8) + 16);

        // Compile kernel
        let source = include_str!("../../kernels/fse_encode.cl");
        let program = Program::create_and_build_from_source(&self.context, source, "-Werror")
            .map_err(|_| PzError::Unsupported)?;
        let kernel = Kernel::create(&program, "FseEncode").map_err(|_| PzError::Unsupported)?;

        // Upload input symbols
        let mut symbols_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                input.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut symbols_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload encode table
        let mut encode_table_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                packed_encode_table.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(
                    &mut encode_table_buf,
                    CL_BLOCKING,
                    0,
                    &packed_encode_table,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Allocate output buffer (per-lane sections)
        let total_output_bytes = num_states * max_output_bytes_per_lane;
        let output_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                total_output_bytes,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Allocate lane results: 3 u32 per lane
        let lane_results_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                num_states * 3,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Dispatch kernel
        let wg_size = num_states.min(self.max_work_group_size).max(1);
        let global_size = num_states;

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&symbols_buf)
                .set_arg(&encode_table_buf)
                .set_arg(&output_buf)
                .set_arg(&lane_results_buf)
                .set_arg(&(num_states as cl_uint))
                .set_arg(&(input.len() as cl_uint))
                .set_arg(&table_size)
                .set_arg(&(max_output_bytes_per_lane as cl_uint))
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("fse_encode: encode kernel", &kernel_event);

        // Download lane results
        let mut lane_results = vec![0u32; num_states * 3];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&lane_results_buf, CL_BLOCKING, 0, &mut lane_results, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Download output bitstreams
        let mut output_data = vec![0u8; total_output_bytes];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Serialize to interleaved wire format (same as encode_interleaved_n)
        let mut total_bitstream_bytes = 0usize;
        for lane in 0..num_states {
            total_bitstream_bytes += lane_results[lane * 3 + 2] as usize;
        }

        let header_size = 1 + FREQ_TABLE_BYTES + 1 + num_states * (2 + 4 + 4);
        let mut output = Vec::with_capacity(header_size + total_bitstream_bytes);

        output.push(al);
        for &f in &norm.freq {
            output.extend_from_slice(&f.to_le_bytes());
        }
        output.push(num_states as u8);

        for lane in 0..num_states {
            let initial_state = lane_results[lane * 3] as u16;
            let total_bits = lane_results[lane * 3 + 1];
            let byte_len = lane_results[lane * 3 + 2] as usize;

            // Extract this lane's bitstream from the output buffer
            let lane_offset = lane * max_output_bytes_per_lane;
            let bitstream = &output_data[lane_offset..lane_offset + byte_len];

            output.extend_from_slice(&initial_state.to_le_bytes());
            output.extend_from_slice(&total_bits.to_le_bytes());
            output.extend_from_slice(&(byte_len as u32).to_le_bytes());
            output.extend_from_slice(bitstream);
        }

        Ok(output)
    }
}

/// Encode pre-split blocks into independent FSE interleaved streams with a
/// shared frequency table.
///
/// Computes frequency statistics across all blocks, then encodes each block
/// independently using the same normalized frequency table. This ensures all
/// blocks can be decoded with a single GPU-resident decode table.
///
/// Returns a vector of (encoded_data, original_block_len) per block.
pub fn fse_encode_block_slices(
    blocks: &[&[u8]],
    num_streams: usize,
    accuracy_log: u8,
) -> PzResult<Vec<(Vec<u8>, usize)>> {
    use crate::frequency::FrequencyTable;
    use crate::fse::{fse_encode_interleaved, normalize_frequencies, FseTable, FREQ_TABLE_BYTES};

    if blocks.is_empty() {
        return Ok(Vec::new());
    }

    let num_streams = num_streams.max(1);
    let accuracy_log =
        accuracy_log.clamp(crate::fse::MIN_ACCURACY_LOG, crate::fse::MAX_ACCURACY_LOG);

    // Build shared frequency table across all blocks.
    let total_len: usize = blocks.iter().map(|b| b.len()).sum();
    let mut combined = Vec::with_capacity(total_len);
    for block in blocks {
        combined.extend_from_slice(block);
    }
    let mut freq = FrequencyTable::new();
    freq.count(&combined);

    let mut al = accuracy_log;
    while (1u32 << al) < freq.used {
        al += 1;
        if al > crate::fse::MAX_ACCURACY_LOG {
            break;
        }
    }

    let norm = normalize_frequencies(&freq, al)?;
    let table = FseTable::from_normalized(&norm);

    // Encode each block independently.
    let mut result = Vec::new();
    for chunk in blocks {
        let stream_results = fse_encode_interleaved(chunk, &table, num_streams);

        // Serialize: same wire format as encode_interleaved_n
        let total_bitstream_bytes: usize = stream_results.iter().map(|(bs, _, _)| bs.len()).sum();
        let header_size = 1 + FREQ_TABLE_BYTES + 1 + num_streams * (2 + 4 + 4);
        let mut output = Vec::with_capacity(header_size + total_bitstream_bytes);

        output.push(al);
        for &f in &norm.freq {
            output.extend_from_slice(&f.to_le_bytes());
        }
        output.push(num_streams as u8);

        for (bitstream, initial_state, total_bits) in &stream_results {
            output.extend_from_slice(&initial_state.to_le_bytes());
            output.extend_from_slice(&total_bits.to_le_bytes());
            output.extend_from_slice(&(bitstream.len() as u32).to_le_bytes());
            output.extend_from_slice(bitstream);
        }

        result.push((output, chunk.len()));
    }

    Ok(result)
}
