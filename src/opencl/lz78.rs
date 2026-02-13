//! Static-dictionary LZW GPU decode.
//!
//! Three-pass approach:
//!   Pass 1 (DecodeLengths): each work-item reads one code, writes its expanded length
//!   Pass 2 (prefix_sum_gpu): compute output offsets from lengths
//!   Pass 3 (WriteOutput): each work-item copies its dictionary entry to the output

use super::*;

impl OpenClEngine {
    /// Decode static-dictionary LZW-encoded data on the GPU.
    ///
    /// Takes a frozen dictionary and packed code bitstream, returns the
    /// decoded byte sequence. The decode is embarrassingly parallel:
    /// each code maps to a fixed dictionary entry via pure table lookup.
    pub fn lzw_decode_static(
        &self,
        packed_codes: &[u8],
        num_codes: u32,
        dict: &crate::lz78_static::FrozenDict,
        original_len: usize,
    ) -> PzResult<Vec<u8>> {
        if num_codes == 0 {
            return Ok(Vec::new());
        }

        let (flat_entries, lengths) = dict.to_gpu_layout();
        let stride = dict.max_entry_len().max(1) as u32;
        let code_bits = dict.code_bits as u32;

        // Upload dictionary entries (flat layout)
        let mut dict_entries_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                flat_entries.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut dict_entries_buf, CL_BLOCKING, 0, &flat_entries, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload dictionary lengths (u16 per entry)
        let mut dict_lengths_buf = unsafe {
            Buffer::<u16>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                lengths.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut dict_lengths_buf, CL_BLOCKING, 0, &lengths, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Upload packed codes bitstream (pad to avoid out-of-bounds reads)
        let padded_len = packed_codes.len() + 4; // extra bytes for safe word-boundary reads
        let mut padded_codes = vec![0u8; padded_len];
        padded_codes[..packed_codes.len()].copy_from_slice(packed_codes);

        let mut codes_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, padded_len, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut codes_buf, CL_BLOCKING, 0, &padded_codes, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Allocate output_lengths buffer (u32 per code)
        let mut output_lengths_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_codes as usize,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };

        // Pass 1: DecodeLengths — each work-item reads its code and writes the entry length
        let num_codes_arg = num_codes as cl_uint;
        let code_bits_arg = code_bits as cl_uint;
        let pass1_event = unsafe {
            ExecuteKernel::new(&self.kernel_lzw_decode_lengths)
                .set_arg(&codes_buf)
                .set_arg(&dict_lengths_buf)
                .set_arg(&output_lengths_buf)
                .set_arg(&num_codes_arg)
                .set_arg(&code_bits_arg)
                .set_global_work_size(num_codes as usize)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass1_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lzw_decode_lengths", &pass1_event);

        // Read the last length before prefix sum overwrites it
        let mut last_len = vec![0u32; 1];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &output_lengths_buf,
                    CL_BLOCKING,
                    (num_codes as usize - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_len,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        let last_entry_len = last_len[0];

        // Pass 2: GPU prefix sum (exclusive) on output_lengths → output_offsets
        self.prefix_sum_gpu(&mut output_lengths_buf, num_codes as usize)?;

        // Read the last offset to compute total output size
        let mut last_offset = vec![0u32; 1];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &output_lengths_buf,
                    CL_BLOCKING,
                    (num_codes as usize - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_offset,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        let total_output = (last_offset[0] + last_entry_len) as usize;

        // Allocate output buffer
        let output_size = total_output.max(1); // avoid zero-size allocation
        let mut output_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                output_size,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };

        // Pass 3: WriteOutput — each work-item copies its dictionary entry to the output
        let stride_arg = stride as cl_uint;
        let pass3_event = unsafe {
            ExecuteKernel::new(&self.kernel_lzw_write_output)
                .set_arg(&codes_buf)
                .set_arg(&dict_entries_buf)
                .set_arg(&dict_lengths_buf)
                .set_arg(&output_lengths_buf) // now contains output_offsets
                .set_arg(&output_buf)
                .set_arg(&num_codes_arg)
                .set_arg(&code_bits_arg)
                .set_arg(&stride_arg)
                .set_global_work_size(num_codes as usize)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass3_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lzw_write_output", &pass3_event);

        // Download output
        let download_len = total_output.min(output_size);
        let mut result = vec![0u8; download_len];
        if download_len > 0 {
            let ev = unsafe {
                self.queue
                    .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut result, &[])
                    .map_err(|_| PzError::Unsupported)?
            };
            ev.wait().map_err(|_| PzError::Unsupported)?;
        }

        result.truncate(original_len);
        Ok(result)
    }
}
