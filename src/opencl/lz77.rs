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

    /// Find matches using the hash-table kernel with custom parameters.
    ///
    /// Compiles a parameterized kernel at runtime with the given hash table
    /// size, chain depth, and bucket capacity. Used for parameter sweep
    /// experiments (Experiment 3).
    pub fn find_matches_with_params(
        &self,
        input: &[u8],
        hash_bits: u32,
        max_chain: u32,
        bucket_cap: u32,
    ) -> PzResult<Vec<Match>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }

        let input_len = input.len();
        let hash_size = 1usize << hash_bits;
        let table_entries = hash_size * bucket_cap as usize;

        // Compile parameterized kernel at runtime
        let defines = format!(
            "-Werror -DSWEEP_HASH_BITS={hash_bits}u -DSWEEP_MAX_CHAIN={max_chain}u -DSWEEP_BUCKET_CAP={bucket_cap}u"
        );
        let source = include_str!("../../kernels/lz77_hash_sweep.cl");
        let program = opencl3::program::Program::create_and_build_from_source(
            &self.context,
            source,
            &defines,
        )
        .map_err(|_| PzError::Unsupported)?;

        let kernel_build =
            Kernel::create(&program, "BuildHashTable").map_err(|_| PzError::Unsupported)?;
        let kernel_find =
            Kernel::create(&program, "FindMatches").map_err(|_| PzError::Unsupported)?;

        // Allocate buffers
        let mut input_buf = unsafe {
            Buffer::<u8>::create(&self.context, CL_MEM_READ_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let output_buf = unsafe {
            Buffer::<GpuMatch>::create(&self.context, CL_MEM_WRITE_ONLY, input_len, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        let mut hash_counts_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, hash_size, ptr::null_mut())
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

        // Upload input
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut input_buf, CL_BLOCKING, 0, input, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Zero hash counts
        let zeros = vec![0u32; hash_size];
        let ev = unsafe {
            self.queue
                .enqueue_write_buffer(&mut hash_counts_buf, CL_BLOCKING, 0, &zeros, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        // Pass 1: Build hash table
        let count = input_len as cl_uint;
        let build_ev = unsafe {
            ExecuteKernel::new(&kernel_build)
                .set_arg(&input_buf)
                .set_arg(&count)
                .set_arg(&hash_counts_buf)
                .set_arg(&hash_table_buf)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        build_ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 hash sweep: build table", &build_ev);

        // Pass 2: Find matches
        let find_ev = unsafe {
            ExecuteKernel::new(&kernel_find)
                .set_arg(&input_buf)
                .set_arg(&count)
                .set_arg(&hash_counts_buf)
                .set_arg(&hash_table_buf)
                .set_arg(&output_buf)
                .set_global_work_size(input_len)
                .enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        find_ev.wait().map_err(|_| PzError::Unsupported)?;
        self.profile_event("lz77 hash sweep: find matches", &find_ev);

        // Read back results
        let mut gpu_matches = vec![GpuMatch::default(); input_len];
        let ev = unsafe {
            self.queue
                .enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut gpu_matches, &[])
                .map_err(|_| PzError::Unsupported)?
        };
        ev.wait().map_err(|_| PzError::Unsupported)?;

        let matches = dedupe_gpu_matches(&gpu_matches, input);
        Ok(matches)
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
}
