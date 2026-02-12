//! Huffman encoding and deflate chained GPU pipeline.

use super::*;

impl OpenClEngine {
    /// Compute a byte histogram of the input data on the GPU.
    ///
    /// Returns a 256-element array of byte frequencies. This can be used
    /// to build a Huffman tree without downloading the full data to the CPU.
    pub fn byte_histogram(&self, input: &[u8]) -> PzResult<[u32; 256]> {
        if input.is_empty() {
            return Ok([0u32; 256]);
        }

        let n = input.len();

        // Create input buffer
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Create histogram buffer (256 uints, zeroed)
        let mut hist_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Upload input
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_event.wait().map_err(|_| PzError::Unsupported)?;

        // Zero the histogram buffer
        let zeros = vec![0u32; 256];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut hist_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        // Run histogram kernel
        let n_arg = n as cl_uint;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_byte_histogram)
                .set_arg(&input_buf)
                .set_arg(&hist_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download histogram
        let mut histogram = vec![0u32; 256];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&hist_buf, CL_BLOCKING, 0, &mut histogram, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_event.wait().map_err(|_| PzError::Unsupported)?;

        let mut result = [0u32; 256];
        result.copy_from_slice(&histogram);
        Ok(result)
    }

    /// Encode data using Huffman coding on the GPU.
    ///
    /// Takes a code lookup table (from a HuffmanTree) and the input symbols.
    /// Returns the encoded bytes and the total number of bits.
    ///
    /// The lookup table format: for each byte value 0-255,
    /// `code_lut[byte] = (bits << 24) | codeword` where codeword is at most 24 bits.
    pub fn huffman_encode(
        &self,
        input: &[u8],
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();

        // Create buffers
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        let mut lut_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        let bit_lengths_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Upload input and LUT
        let write_input = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_input.wait().map_err(|_| PzError::Unsupported)?;

        let write_lut = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, code_lut, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_lut.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 1: compute bit lengths per symbol
        let n_arg = n as cl_uint;
        let pass1_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_bit_lengths)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass1_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download bit lengths and compute prefix sum on CPU
        // (GPU prefix sum would be faster for very large inputs, but adds complexity)
        let mut bit_lengths = vec![0u32; n];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&bit_lengths_buf, CL_BLOCKING, 0, &mut bit_lengths, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_event.wait().map_err(|_| PzError::Unsupported)?;

        // CPU prefix sum: bit_offsets[i] = sum of bit_lengths[0..i)
        let mut bit_offsets = vec![0u32; n];
        let mut running_sum: u64 = 0;
        for i in 0..n {
            bit_offsets[i] = running_sum as u32;
            running_sum += bit_lengths[i] as u64;
        }
        let total_bits = running_sum as usize;

        // Allocate output buffer (as uint array for atomic OR)
        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let mut output_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_uints,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };

        // Zero the output buffer
        let zeros = vec![0u32; output_uints];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        // Upload bit offsets
        let mut offsets_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let write_offsets = unsafe {
            self.queue
                .enqueue_write_buffer(&mut offsets_buf, CL_BLOCKING, 0, &bit_offsets, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_offsets.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 2: write codewords at computed offsets
        let pass2_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_write_codes)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&offsets_buf)
                .set_arg(&output_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass2_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download output as uint array
        let mut output_data = vec![0u32; output_uints];
        let read_out = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_out.wait().map_err(|_| PzError::Unsupported)?;

        // Convert uint array to bytes (big-endian within each uint to match MSB-first packing)
        let output_bytes_len = total_bits.div_ceil(8);
        let mut output_bytes = vec![0u8; output_bytes_len];
        for (i, &val) in output_data.iter().enumerate() {
            let base = i * 4;
            let bytes = val.to_be_bytes();
            for (j, &b) in bytes.iter().enumerate() {
                if base + j < output_bytes_len {
                    output_bytes[base + j] = b;
                }
            }
        }

        Ok((output_bytes, total_bits))
    }

    /// Perform an exclusive prefix sum on a GPU buffer in-place.
    ///
    /// Uses Blelloch scan (work-efficient parallel prefix sum) with
    /// multi-level reduction for large arrays. Avoids downloading
    /// the buffer to the host for CPU prefix sum.
    pub fn prefix_sum_gpu(&self, buf: &mut Buffer<cl_uint>, n: usize) -> PzResult<()> {
        if n <= 1 {
            if n == 1 {
                // Single element: exclusive prefix sum is just 0
                let zero = vec![0u32; 1];
                let write_event = unsafe {
                    self.queue
                        .enqueue_write_buffer(buf, CL_BLOCKING, 0, &zero, &[])
                        .map_err(|_| PzError::Unsupported)?
                };
                write_event.wait().map_err(|_| PzError::Unsupported)?;
            }
            return Ok(());
        }

        // Use a work-group size that processes 2 elements each (Blelloch scan)
        let max_wg = self.max_work_group_size.min(256);
        let block_size = max_wg * 2; // each work-item handles 2 elements

        if n <= block_size {
            // Single work-group: no need for multi-level scan
            self.run_prefix_sum_block(buf, None, n, max_wg)?;
            return Ok(());
        }

        // Multi-level: split into blocks, scan each, collect block totals
        let num_blocks = n.div_ceil(block_size);

        // Allocate block sums buffer
        let mut block_sums_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Level 1: scan each block, output block totals
        self.run_prefix_sum_block(buf, Some(&mut block_sums_buf), n, max_wg)?;

        // Level 2: recursively scan block totals
        if num_blocks > 1 {
            self.prefix_sum_gpu(&mut block_sums_buf, num_blocks)?;
        }

        // Level 3: apply block offsets to elements
        self.run_prefix_sum_apply(buf, &block_sums_buf, n, max_wg)?;

        Ok(())
    }

    /// Run the PrefixSumBlock kernel on a single level.
    fn run_prefix_sum_block(
        &self,
        data_buf: &mut Buffer<cl_uint>,
        block_sums_buf: Option<&mut Buffer<cl_uint>>,
        n: usize,
        local_size: usize,
    ) -> PzResult<()> {
        let block_size = local_size * 2;
        let num_blocks = n.div_ceil(block_size);
        let global_size = num_blocks * local_size;
        let n_arg = n as cl_uint;

        let kernel_event = match block_sums_buf {
            Some(sums_buf) => unsafe {
                let local_mem_bytes = block_size * std::mem::size_of::<cl_uint>();
                ExecuteKernel::new(&self.kernel_prefix_sum_block)
                    .set_arg(data_buf)
                    .set_arg(sums_buf)
                    .set_arg(&n_arg)
                    .set_arg_local_buffer(local_mem_bytes)
                    .set_local_work_size(local_size)
                    .set_global_work_size(global_size)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            },
            None => unsafe {
                // Null block_sums pointer — kernel checks for NULL
                let null_ptr: *const cl_uint = ptr::null();
                let local_mem_bytes = block_size * std::mem::size_of::<cl_uint>();
                ExecuteKernel::new(&self.kernel_prefix_sum_block)
                    .set_arg(data_buf)
                    .set_arg(&null_ptr)
                    .set_arg(&n_arg)
                    .set_arg_local_buffer(local_mem_bytes)
                    .set_local_work_size(local_size)
                    .set_global_work_size(global_size)
                    .enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            },
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        Ok(())
    }

    /// Run the PrefixSumApply kernel to add block offsets.
    fn run_prefix_sum_apply(
        &self,
        data_buf: &mut Buffer<cl_uint>,
        block_sums_buf: &Buffer<cl_uint>,
        n: usize,
        local_size: usize,
    ) -> PzResult<()> {
        let block_size = local_size * 2;
        let n_arg = n as cl_uint;
        let block_size_arg = block_size as cl_uint;

        // Use a local_work_size that doesn't exceed the max work group size.
        // The kernel computes block_id from gid / block_size, so any valid
        // local_work_size works.
        let apply_local = local_size.min(self.max_work_group_size);
        let global_size = n.div_ceil(apply_local) * apply_local;

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_prefix_sum_apply)
                .set_arg(data_buf)
                .set_arg(block_sums_buf)
                .set_arg(&n_arg)
                .set_arg(&block_size_arg)
                .set_local_work_size(apply_local)
                .set_global_work_size(global_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        Ok(())
    }

    /// Encode data using Huffman coding entirely on the GPU with GPU prefix sum.
    ///
    /// Same as `huffman_encode` but uses the GPU Blelloch scan for the prefix
    /// sum instead of downloading to host. Eliminates one host↔device round-trip.
    pub fn huffman_encode_gpu_scan(
        &self,
        input: &[u8],
        code_lut: &[u32; 256],
    ) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let n = input.len();

        // Create buffers
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        let mut lut_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // bit_lengths_buf will also serve as bit_offsets_buf after prefix sum
        let mut bit_lengths_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Upload input and LUT
        let write_input = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_input.wait().map_err(|_| PzError::Unsupported)?;

        let write_lut = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, code_lut, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_lut.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 1: compute bit lengths per symbol
        let n_arg = n as cl_uint;
        let pass1_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_bit_lengths)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass1_event.wait().map_err(|_| PzError::Unsupported)?;

        // We need the total bits before doing the scan.
        // Read the last element + its bit length to get the total.
        // First, save the last element before the scan overwrites it.
        let mut last_val = vec![0u32; 1];
        let read_last = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_val,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_last.wait().map_err(|_| PzError::Unsupported)?;
        let last_bit_length = last_val[0];

        // GPU prefix sum (exclusive): bit_lengths → bit_offsets
        self.prefix_sum_gpu(&mut bit_lengths_buf, n)?;

        // Read the last offset to compute total_bits
        let mut last_offset = vec![0u32; 1];
        let read_offset = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_offset,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_offset.wait().map_err(|_| PzError::Unsupported)?;
        let total_bits = (last_offset[0] + last_bit_length) as usize;

        // Allocate output buffer
        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Ok((Vec::new(), 0));
        }

        let mut output_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_uints,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };

        // Zero the output buffer
        let zeros = vec![0u32; output_uints];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 2: write codewords at GPU-computed offsets
        let pass2_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_write_codes)
                .set_arg(&input_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf) // now contains bit_offsets
                .set_arg(&output_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass2_event.wait().map_err(|_| PzError::Unsupported)?;

        // Download output
        let mut output_data = vec![0u32; output_uints];
        let read_out = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_out.wait().map_err(|_| PzError::Unsupported)?;

        // Convert uint array to bytes (big-endian to match MSB-first packing)
        let output_bytes_len = total_bits.div_ceil(8);
        let mut output_bytes = vec![0u8; output_bytes_len];
        for (i, &val) in output_data.iter().enumerate() {
            let base = i * 4;
            let bytes = val.to_be_bytes();
            for (j, &b) in bytes.iter().enumerate() {
                if base + j < output_bytes_len {
                    output_bytes[base + j] = b;
                }
            }
        }

        Ok((output_bytes, total_bits))
    }

    /// GPU-chained Deflate compression: LZ77 + Huffman on GPU.
    ///
    /// Performs GPU LZ77 match finding followed by GPU Huffman encoding
    /// with minimal host↔device transfers. The LZ77 output is uploaded
    /// once and stays on the GPU: a `ByteHistogram` kernel computes byte
    /// frequencies on-device (downloading only 1KB of histogram data
    /// instead of the full LZ77 stream), and Huffman encoding reuses the
    /// same GPU buffer.
    ///
    /// **Data flow:**
    /// 1. GPU: LZ77 hash-table match finding → download match array
    /// 2. CPU: deduplicate + serialize matches (sequential, unavoidable)
    /// 3. GPU: upload LZ77 bytes once → run ByteHistogram → download 256×u32 (1KB)
    /// 4. CPU: build Huffman tree from histogram, produce code LUT
    /// 5. GPU: Huffman encode (reusing LZ77 buffer) with GPU prefix sum
    /// 6. GPU: download final encoded bitstream
    ///
    /// Returns the serialized Deflate block data (lz_len + total_bits +
    /// freq_table + huffman_data), ready for the pipeline container.
    pub fn deflate_chained(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        // Stage 1: GPU LZ77 compression (match finding + dedupe + serialize)
        let lz_data = self.lz77_compress(input, KernelVariant::HashTable)?;
        let lz_len = lz_data.len();

        if lz_data.is_empty() {
            return Err(PzError::InvalidInput);
        }

        let n = lz_data.len();

        // Upload LZ77 bytes to GPU once — this buffer is reused for both
        // histogram and Huffman encoding, eliminating redundant transfers.
        let mut lz_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lz_buf, CL_BLOCKING, 0, &lz_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: upload LZ77 data", &write_event);

        // Stage 2: GPU ByteHistogram — compute frequencies on-device.
        // Only 1KB (256×u32) is downloaded instead of the full LZ77 stream.
        let mut hist_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let zeros = vec![0u32; 256];
        let zero_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut hist_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_event.wait().map_err(|_| PzError::Unsupported)?;

        let n_arg = n as cl_uint;
        let hist_event = unsafe {
            ExecuteKernel::new(&self.kernel_byte_histogram)
                .set_arg(&lz_buf)
                .set_arg(&hist_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        hist_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: byte histogram kernel", &hist_event);

        let mut histogram = vec![0u32; 256];
        let read_hist = unsafe {
            self.queue
                .enqueue_read_buffer(&hist_buf, CL_BLOCKING, 0, &mut histogram, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_hist.wait().map_err(|_| PzError::Unsupported)?;

        // Build Huffman tree from GPU-computed histogram (CPU — tree construction is fast)
        let mut freq = crate::frequency::FrequencyTable::new();
        for (i, &count) in histogram.iter().enumerate() {
            freq.byte[i] = count;
        }
        freq.total = freq.byte.iter().map(|&c| c as u64).sum();
        freq.used = freq.byte.iter().filter(|&&c| c > 0).count() as u32;

        let tree = crate::huffman::HuffmanTree::from_frequency_table(&freq)
            .ok_or(PzError::InvalidInput)?;
        let freq_table = tree.serialize_frequencies();

        // Build the packed code LUT for GPU
        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        // Stage 3: GPU Huffman encoding with GPU prefix sum.
        // Reuses lz_buf (already on device) — no re-upload needed.
        let mut lut_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_ONLY, 256, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };
        let write_lut = unsafe {
            self.queue
                .enqueue_write_buffer(&mut lut_buf, CL_BLOCKING, 0, &code_lut, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        write_lut.wait().map_err(|_| PzError::Unsupported)?;

        // bit_lengths_buf also serves as bit_offsets_buf after prefix sum
        let mut bit_lengths_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, n, ptr::null_mut())
                .map_err(|_| PzError::Unsupported)?
        };

        // Pass 1: compute bit lengths per symbol
        let pass1_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_bit_lengths)
                .set_arg(&lz_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass1_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: huffman bit lengths", &pass1_event);

        // Save last bit length before prefix sum overwrites it
        let mut last_val = vec![0u32; 1];
        let read_last = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_val,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_last.wait().map_err(|_| PzError::Unsupported)?;
        let last_bit_length = last_val[0];

        // GPU prefix sum (exclusive): bit_lengths → bit_offsets
        self.prefix_sum_gpu(&mut bit_lengths_buf, n)?;

        // Read the last offset to compute total_bits
        let mut last_offset = vec![0u32; 1];
        let read_offset = unsafe {
            self.queue
                .enqueue_read_buffer(
                    &bit_lengths_buf,
                    CL_BLOCKING,
                    (n - 1) * std::mem::size_of::<cl_uint>(),
                    &mut last_offset,
                    &[],
                )
                .map_err(|_| PzError::Unsupported)?
        };
        read_offset.wait().map_err(|_| PzError::Unsupported)?;
        let total_bits = (last_offset[0] + last_bit_length) as usize;

        // Allocate and zero output buffer
        let output_uints = total_bits.div_ceil(32);
        if output_uints == 0 {
            return Err(PzError::InvalidInput);
        }

        let mut output_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                output_uints,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::Unsupported)?
        };
        let out_zeros = vec![0u32; output_uints];
        let zero_out = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &out_zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        zero_out.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 2: write codewords at GPU-computed offsets
        let pass2_event = unsafe {
            ExecuteKernel::new(&self.kernel_huffman_write_codes)
                .set_arg(&lz_buf)
                .set_arg(&lut_buf)
                .set_arg(&bit_lengths_buf) // now contains bit_offsets
                .set_arg(&output_buf)
                .set_arg(&n_arg)
                .set_global_work_size(n)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        pass2_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: huffman write codes", &pass2_event);

        // Download final encoded bitstream
        let mut output_data = vec![0u32; output_uints];
        let read_out = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut output_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        read_out.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("deflate_chained: download bitstream", &read_out);

        // Convert uint array to bytes (big-endian to match MSB-first packing)
        let output_bytes_len = total_bits.div_ceil(8);
        let mut huffman_data = vec![0u8; output_bytes_len];
        for (i, &val) in output_data.iter().enumerate() {
            let base = i * 4;
            let bytes = val.to_be_bytes();
            for (j, &b) in bytes.iter().enumerate() {
                if base + j < output_bytes_len {
                    huffman_data[base + j] = b;
                }
            }
        }

        // Serialize in the same format as the CPU Deflate block
        let mut output = Vec::new();
        output.extend_from_slice(&(lz_len as u32).to_le_bytes());
        output.extend_from_slice(&(total_bits as u32).to_le_bytes());
        for &freq in &freq_table {
            output.extend_from_slice(&freq.to_le_bytes());
        }
        output.extend_from_slice(&huffman_data);

        Ok(output)
    }
}
