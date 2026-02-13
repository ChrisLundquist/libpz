//! LZ77 GPU match finding kernels.

use super::*;

impl OpenClEngine {
    /// Find LZ77 matches for the entire input using the GPU.
    ///
    /// Returns a vector of deduplicated `Match` structs compatible with
    /// the CPU LZ77 format, ready for serialization or further processing
    /// (e.g., optimal-parse DP in Phase 2).
    ///
    /// # Arguments
    ///
    /// * `input` - The data to compress. Must be non-empty.
    /// * `variant` - Which kernel to use (per-position or batched).
    pub fn find_matches(&self, input: &[u8], variant: KernelVariant) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        // Allocate device buffers
        let input_len = input.len();

        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let output_buf = unsafe {
            Buffer::<GpuMatch>::create(&self.context, CL_MEM_WRITE_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        // Write input to device
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_event.wait().map_err(|_| PzError::InvalidInput)?;
        self.profile_event("find_matches: upload input", &write_event);

        // Execute kernel
        match variant {
            KernelVariant::PerPosition => {
                self.run_per_position_kernel(&input_buf, &output_buf, input_len)?;
            }
            KernelVariant::Batch => {
                self.run_batch_kernel(&input_buf, &output_buf, input_len)?;
            }
            KernelVariant::HashTable => {
                self.run_hash_kernel(&input_buf, &output_buf, input_len)?;
            }
        }

        // Read back results
        let mut gpu_matches = vec![GpuMatch::default(); input_len];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut gpu_matches, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        read_event.wait().map_err(|_| PzError::InvalidInput)?;
        self.profile_event("find_matches: download matches", &read_event);

        // Deduplicate and convert to the Rust Match type
        let matches = dedupe_gpu_matches(&gpu_matches, input);
        Ok(matches)
    }

    /// Find LZ77 matches on the GPU, keeping the match buffer on-device.
    ///
    /// Unlike [`find_matches()`] which downloads and deduplicates immediately,
    /// this returns a [`GpuMatchBuf`] that stays on the GPU. The caller can:
    /// - Download later via [`download_and_dedupe()`] for CPU processing
    /// - (future) Pass to a GPU demux kernel without any PCI transfer
    ///
    /// This is the building block for zero-copy GPU pipeline composition.
    pub fn find_matches_to_device(
        &self,
        input: &[u8],
        variant: KernelVariant,
    ) -> PzResult<GpuMatchBuf> {
        if input.is_empty() {
            return Ok(GpuMatchBuf {
                buf: unsafe {
                    Buffer::<GpuMatch>::create(&self.context, CL_MEM_WRITE_ONLY, 1, ptr::null_mut())
                        .map_err(|_| PzError::BufferTooSmall)?
                },
                input_len: 0,
            });
        }

        let input_len = input.len();

        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let output_buf = unsafe {
            Buffer::<GpuMatch>::create(&self.context, CL_MEM_WRITE_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        // Write input to device
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_event.wait().map_err(|_| PzError::InvalidInput)?;
        self.profile_event("find_matches_to_device: upload input", &write_event);

        // Execute kernel
        match variant {
            KernelVariant::PerPosition => {
                self.run_per_position_kernel(&input_buf, &output_buf, input_len)?;
            }
            KernelVariant::Batch => {
                self.run_batch_kernel(&input_buf, &output_buf, input_len)?;
            }
            KernelVariant::HashTable => {
                self.run_hash_kernel(&input_buf, &output_buf, input_len)?;
            }
        }

        Ok(GpuMatchBuf {
            buf: output_buf,
            input_len,
        })
    }

    /// Download a device-resident match buffer and deduplicate into `Match` structs.
    ///
    /// This is the download counterpart to [`find_matches_to_device()`].
    pub fn download_and_dedupe(
        &self,
        match_buf: &GpuMatchBuf,
        input: &[u8],
    ) -> PzResult<Vec<Match>> {
        if match_buf.input_len == 0 {
            return Ok(Vec::new());
        }

        let mut gpu_matches = vec![GpuMatch::default(); match_buf.input_len];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&match_buf.buf, CL_BLOCKING, 0, &mut gpu_matches, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        read_event.wait().map_err(|_| PzError::InvalidInput)?;
        self.profile_event("download_and_dedupe: download matches", &read_event);

        let matches = dedupe_gpu_matches(&gpu_matches, input);
        Ok(matches)
    }

    /// Execute the per-position kernel (one work-item per byte).
    fn run_per_position_kernel(
        &self,
        input_buf: &Buffer<u8>,
        output_buf: &Buffer<GpuMatch>,
        input_len: usize,
    ) -> PzResult<()> {
        let count = input_len as cl_uint;
        let global_size = input_len;

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_per_pos)
                .set_arg(input_buf)
                .set_arg(output_buf)
                .set_arg(&count)
                .set_global_work_size(global_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };

        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 per-position kernel", &kernel_event);
        Ok(())
    }

    /// Execute the batch kernel (each work-item handles STEP_SIZE positions).
    fn run_batch_kernel(
        &self,
        input_buf: &Buffer<u8>,
        output_buf: &Buffer<GpuMatch>,
        input_len: usize,
    ) -> PzResult<()> {
        let count = input_len as cl_uint;
        // Round up to cover all positions
        let num_work_items = input_len.div_ceil(BATCH_STEP_SIZE);

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_batch)
                .set_arg(input_buf)
                .set_arg(output_buf)
                .set_arg(&count)
                .set_global_work_size(num_work_items)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };

        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 batch kernel", &kernel_event);
        Ok(())
    }

    /// Execute the two-pass hash-table kernel.
    ///
    /// Pass 1: BuildHashTable — each work-item hashes its 3-byte prefix
    /// and atomically appends its position to a bucket.
    /// Pass 2: FindMatches — each work-item searches its hash bucket
    /// for the best match (bounded by MAX_CHAIN).
    fn run_hash_kernel(
        &self,
        input_buf: &Buffer<u8>,
        output_buf: &Buffer<GpuMatch>,
        input_len: usize,
    ) -> PzResult<()> {
        let count = input_len as cl_uint;
        let table_entries = HASH_TABLE_SIZE * HASH_BUCKET_CAP;

        // Allocate hash table buffers on the device
        let mut hash_counts_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                HASH_TABLE_SIZE,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        let hash_table_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                table_entries,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Zero the hash counts
        let zeros = vec![0u32; HASH_TABLE_SIZE];
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut hash_counts_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_event.wait().map_err(|_| PzError::InvalidInput)?;

        // Pass 1: Build hash table
        let build_event = unsafe {
            ExecuteKernel::new(&self.kernel_hash_build)
                .set_arg(input_buf)
                .set_arg(&count)
                .set_arg(&hash_counts_buf)
                .set_arg(&hash_table_buf)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        build_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 hash: build table", &build_event);

        // Pass 2: Find matches
        let find_event = unsafe {
            ExecuteKernel::new(&self.kernel_hash_find)
                .set_arg(input_buf)
                .set_arg(&count)
                .set_arg(&hash_counts_buf)
                .set_arg(&hash_table_buf)
                .set_arg(output_buf)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        find_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 hash: find matches", &find_event);

        Ok(())
    }

    /// GPU-accelerated LZ77 compression.
    ///
    /// Uses the GPU to find matches, deduplicates them, and serializes
    /// to the same byte format as the CPU `lz77::compress_lazy()`.
    /// The output is decompressible by `lz77::decompress()`.
    pub fn lz77_compress(&self, input: &[u8], variant: KernelVariant) -> PzResult<Vec<u8>> {
        let matches = self.find_matches(input, variant)?;

        let mut output = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
        for m in &matches {
            output.extend_from_slice(&m.to_bytes());
        }
        Ok(output)
    }

    /// GPU-accelerated top-K match finding for optimal parsing.
    ///
    /// For each input position, finds the K best match candidates using
    /// the GPU. Returns a `MatchTable` ready for `optimal_parse()`.
    pub fn find_topk_matches(&self, input: &[u8]) -> PzResult<crate::optimal::MatchTable> {
        use crate::optimal::{MatchCandidate, MatchTable};

        if input.is_empty() {
            return Ok(MatchTable::new(0, TOPK_K));
        }

        let input_len = input.len();
        let output_len = input_len * TOPK_K;

        // Allocate device buffers
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let output_buf = unsafe {
            Buffer::<GpuCandidate>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                output_len,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Write input to device
        let write_event = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_event.wait().map_err(|_| PzError::InvalidInput)?;

        // Execute top-K kernel
        let count = input_len as cl_uint;
        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel_topk)
                .set_arg(&input_buf)
                .set_arg(&output_buf)
                .set_arg(&count)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;

        // Read back results
        let mut gpu_candidates = vec![GpuCandidate::default(); output_len];
        let read_event = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut gpu_candidates, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        read_event.wait().map_err(|_| PzError::InvalidInput)?;

        // Convert to MatchTable
        let mut table = MatchTable::new(input_len, TOPK_K);
        for pos in 0..input_len {
            let slot = table.at_mut(pos);
            for k in 0..TOPK_K {
                let gc = &gpu_candidates[pos * TOPK_K + k];
                slot[k] = MatchCandidate {
                    offset: gc.offset,
                    length: gc.length,
                };
            }
        }

        Ok(table)
    }

    /// GPU-accelerated block-parallel LZ77 decompression.
    ///
    /// Decompresses multiple independently-compressed LZ77 blocks in parallel
    /// on the GPU. Each block is assigned to one workgroup; thread 0 parses
    /// matches sequentially while all threads cooperate on back-reference copies.
    ///
    /// # Arguments
    ///
    /// * `block_data` - Concatenated serialized LZ77 match data for all blocks.
    /// * `block_meta` - Per-block metadata: `(match_data_offset, num_matches, decompressed_size)`.
    /// * `cooperative_threads` - Workgroup size (threads per block). Must be
    ///   a power of 2, clamped to device max.
    ///
    /// Returns the concatenated decompressed data from all blocks.
    pub fn lz77_decompress_blocks(
        &self,
        block_data: &[u8],
        block_meta: &[(usize, usize, usize)],
        cooperative_threads: usize,
    ) -> PzResult<Vec<u8>> {
        if block_data.is_empty() || block_meta.is_empty() {
            return Ok(Vec::new());
        }

        let num_blocks = block_meta.len();

        // Compute output offsets via prefix sum of decompressed sizes
        let mut gpu_meta = Vec::with_capacity(num_blocks * 3);
        let mut output_offset = 0usize;
        for &(data_offset, num_matches, decompressed_size) in block_meta {
            gpu_meta.push(data_offset as u32);
            gpu_meta.push(num_matches as u32);
            gpu_meta.push(output_offset as u32);
            output_offset += decompressed_size;
        }
        let total_output_len = output_offset;

        if total_output_len == 0 {
            return Ok(Vec::new());
        }

        // Compile kernel with the requested workgroup size
        let wg_size = cooperative_threads
            .next_power_of_two()
            .min(self.max_work_group_size)
            .max(1);
        let defines = format!("-Werror -DWG_SIZE={wg_size}u");
        let source = include_str!("../../kernels/lz77_decode.cl");
        let program = Program::create_and_build_from_source(&self.context, source, &defines)
            .map_err(|_| PzError::Unsupported)?;
        let kernel =
            Kernel::create(&program, "Lz77DecodeBlock").map_err(|_| PzError::Unsupported)?;

        // Upload match data
        let mut match_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                block_data.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut match_buf, CL_BLOCKING, 0, block_data, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77_decompress_blocks: upload matches", &ev);

        // Upload block metadata
        let mut meta_buf = unsafe {
            Buffer::<u32>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                gpu_meta.len(),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut meta_buf, CL_BLOCKING, 0, &gpu_meta, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Allocate output buffer (zeroed)
        let mut output_buf = unsafe {
            Buffer::<u8>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                total_output_len,
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let zeros = vec![0u8; total_output_len];
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut output_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Dispatch: one workgroup per block
        let num_blocks_arg = num_blocks as cl_uint;
        let total_out_arg = total_output_len as cl_uint;
        let global_size = num_blocks * wg_size;

        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&match_buf)
                .set_arg(&meta_buf)
                .set_arg(&output_buf)
                .set_arg(&num_blocks_arg)
                .set_arg(&total_out_arg)
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        kernel_event.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77_decompress_blocks: decode kernel", &kernel_event);

        // Download result
        let mut result = vec![0u8; total_output_len];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut result, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77_decompress_blocks: download output", &ev);

        Ok(result)
    }
}

/// Per-block metadata: `(data_offset, num_matches, decompressed_size)`.
pub type BlockMeta = Vec<(usize, usize, usize)>;

/// Compress input into independently-decompressible LZ77 blocks.
///
/// Each block of `block_size` bytes is compressed with a fresh LZ77 window,
/// so blocks can be decompressed in any order without cross-block dependencies.
///
/// Returns:
/// - `block_data`: concatenated serialized match data for all blocks
/// - `block_meta`: per-block `(data_offset, num_matches, decompressed_size)`
pub fn lz77_compress_blocks(input: &[u8], block_size: usize) -> PzResult<(Vec<u8>, BlockMeta)> {
    use crate::lz77;

    let mut block_data = Vec::new();
    let mut block_meta = Vec::new();

    for chunk in input.chunks(block_size) {
        let compressed = lz77::compress_lazy(chunk)?;
        let data_offset = block_data.len();
        let num_matches = compressed.len() / Match::SERIALIZED_SIZE;
        block_meta.push((data_offset, num_matches, chunk.len()));
        block_data.extend_from_slice(&compressed);
    }

    Ok((block_data, block_meta))
}
