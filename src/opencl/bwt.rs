//! BWT GPU suffix array construction.

use super::*;

impl OpenClEngine {
    /// GPU-accelerated BWT forward transform.
    ///
    /// Uses the GPU for the expensive suffix array sort steps (radix sort)
    /// and rank assignment (parallel prefix sum). Produces a valid BWT
    /// that round-trips correctly through `bwt::decode()`.
    pub fn bwt_encode(&self, input: &[u8]) -> PzResult<BwtResult> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        let sa = self.bwt_build_suffix_array(input)?;

        // Extract BWT from suffix array (same logic as CPU bwt::encode)
        let n = input.len();
        let mut bwt = Vec::with_capacity(n);
        let mut primary_index = 0u32;

        for (i, &sa_val) in sa.iter().enumerate() {
            if sa_val == 0 {
                primary_index = i as u32;
                bwt.push(input[n - 1]);
            } else {
                bwt.push(input[sa_val - 1]);
            }
        }

        Ok(BwtResult {
            data: bwt,
            primary_index,
        })
    }

    /// Build suffix array on the GPU using prefix-doubling with radix sort.
    ///
    /// Each doubling step sorts sa[] by (rank[sa[i]], rank[(sa[i]+k) % n]),
    /// then assigns new ranks via parallel prefix sum — all on the GPU.
    /// Only a single scalar (max_rank) is read back per step for convergence.
    fn bwt_build_suffix_array(&self, input: &[u8]) -> PzResult<Vec<usize>> {
        let n = input.len();
        if n <= 1 {
            return Ok(if n == 0 { Vec::new() } else { vec![0] });
        }

        // Pad to power-of-2 size. Sentinel entries have rank = UINT_MAX.
        let padded_n = n.next_power_of_two();

        // Initialize sa in descending order so that LSB-first stable radix sort
        // breaks ties by suffix index descending (matching CPU SA-IS behavior).
        let sa_host: Vec<cl_uint> = (0..padded_n as cl_uint).rev().collect();
        let mut rank_host: Vec<cl_uint> = vec![cl_uint::MAX; padded_n];
        for i in 0..n {
            rank_host[i] = input[i] as cl_uint;
        }

        // Allocate GPU buffers
        let mut sa_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut sa_buf_alt = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut rank_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        // Double-buffer for rank output (swap each iteration)
        let mut rank_buf_alt = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        // diff/prefix buffer for rank assignment
        let mut diff_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut prefix_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };
        // Keys buffer for radix sort (one u32 per element, holds 8-bit digit)
        let mut keys_buf = unsafe {
            Buffer::<cl_uint>::create(&self.context, CL_MEM_READ_WRITE, padded_n, ptr::null_mut())
                .map_err(|_| PzError::BufferTooSmall)?
        };

        // Radix sort histogram buffers
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let histogram_len = 256 * num_groups;
        let mut histogram_buf = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                histogram_len.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let mut histogram_buf_scan = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                histogram_len.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Block sums buffers for multi-level prefix sum (rank assignment)
        let block_elems = wg * 2;
        let num_blocks_l0 = padded_n.div_ceil(block_elems);
        let mut block_sums_l0 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l0.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let num_blocks_l1 = num_blocks_l0.div_ceil(block_elems);
        let mut block_sums_l1 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l1.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let num_blocks_l2 = num_blocks_l1.div_ceil(block_elems);
        let mut block_sums_l2 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l2.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Block sums buffers for histogram prefix sum (radix sort)
        let hist_num_blocks_l0 = histogram_len.div_ceil(block_elems);
        let mut hist_block_sums_l0 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hist_num_blocks_l0.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let hist_num_blocks_l1 = hist_num_blocks_l0.div_ceil(block_elems);
        let mut hist_block_sums_l1 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hist_num_blocks_l1.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let hist_num_blocks_l2 = hist_num_blocks_l1.div_ceil(block_elems);
        let mut hist_block_sums_l2 = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                hist_num_blocks_l2.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };

        // Upload initial data
        let write_sa = unsafe {
            self.queue
                .enqueue_write_buffer(&mut sa_buf, CL_BLOCKING, 0, &sa_host, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_sa.wait().map_err(|_| PzError::InvalidInput)?;

        let write_rank = unsafe {
            self.queue
                .enqueue_write_buffer(&mut rank_buf, CL_BLOCKING, 0, &rank_host, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        write_rank.wait().map_err(|_| PzError::InvalidInput)?;

        let n_arg = n as cl_uint;
        let padded_n_arg = padded_n as cl_uint;

        // Prefix-doubling loop — all GPU work is event-chained, only
        // the convergence read blocks the host (once per doubling step).
        // Adaptive key width: initial ranks are bytes (0-255), so first
        // radix sort only needs 2 passes instead of 8.
        let mut max_rank: cl_uint = 255;
        let mut k_step: usize = 1;
        while k_step < n {
            let k_arg = k_step as cl_uint;

            // Phase 0: Radix sort sa[] by composite key (adaptive pass count)
            let sort_event = self.run_radix_sort(
                &mut sa_buf,
                &mut sa_buf_alt,
                &rank_buf,
                &mut keys_buf,
                &mut histogram_buf,
                &mut histogram_buf_scan,
                padded_n,
                n_arg,
                k_arg,
                &mut hist_block_sums_l0,
                &mut hist_block_sums_l1,
                &mut hist_block_sums_l2,
                hist_num_blocks_l0,
                hist_num_blocks_l1,
                max_rank,
                None, // initial buffer writes were CL_BLOCKING
            )?;

            // Phase 1: Compare consecutive composite keys → diff[]
            let compare_event = self.run_rank_compare(
                &sa_buf,
                &rank_buf,
                &mut diff_buf,
                n_arg,
                padded_n_arg,
                k_arg,
                Some(&sort_event),
            )?;

            // Phase 2: Inclusive prefix sum of diff[] → prefix[]
            let prefix_event = self.run_prefix_sum(
                &diff_buf,
                &mut prefix_buf,
                padded_n,
                &mut block_sums_l0,
                &mut block_sums_l1,
                &mut block_sums_l2,
                num_blocks_l0,
                num_blocks_l1,
                Some(&compare_event),
            )?;

            // Phase 3: Scatter ranks to position-indexed buffer
            let scatter_event = self.run_rank_scatter(
                &sa_buf,
                &prefix_buf,
                &mut rank_buf_alt,
                n_arg,
                padded_n_arg,
                Some(&prefix_event),
            )?;

            // Read convergence scalar: prefix[n-1] = max_rank among real entries.
            // This is the only host sync point per doubling step — the read waits
            // for the scatter event, then blocks until the data is available.
            let mut max_rank_host: [cl_uint; 1] = [0];
            let read_event = unsafe {
                self.queue
                    .enqueue_read_buffer(
                        &prefix_buf,
                        CL_BLOCKING,
                        (n - 1) * std::mem::size_of::<cl_uint>(),
                        &mut max_rank_host,
                        &[scatter_event.get()],
                    )
                    .map_err(|_| PzError::InvalidInput)?
            };
            read_event.wait().map_err(|_| PzError::InvalidInput)?;

            // Update max_rank for next iteration's adaptive pass selection
            max_rank = max_rank_host[0];

            // Swap: rank_buf_alt (new ranks) becomes rank_buf for next iteration
            std::mem::swap(&mut rank_buf, &mut rank_buf_alt);

            if max_rank as usize == n - 1 {
                break;
            }

            k_step *= 2;
        }

        // Read final sorted sa back to host
        let mut sa_host_final: Vec<cl_uint> = vec![0; padded_n];
        let read_sa = unsafe {
            self.queue
                .enqueue_read_buffer(&sa_buf, CL_BLOCKING, 0, &mut sa_host_final, &[])
                .map_err(|_| PzError::InvalidInput)?
        };
        read_sa.wait().map_err(|_| PzError::InvalidInput)?;

        // Extract the real suffix array (filter out sentinel entries)
        let sa: Vec<usize> = sa_host_final
            .iter()
            .filter(|&&v| (v as usize) < n)
            .map(|&v| v as usize)
            .collect();

        if sa.len() != n {
            return Err(PzError::InvalidInput);
        }

        Ok(sa)
    }

    /// Run an adaptive radix sort on the sa buffer by composite key.
    ///
    /// Performs LSB-first 8-bit radix sort, sorting sa[] by the 64-bit
    /// composite key (rank[sa[i]] << 32 | rank[(sa[i]+k) % n]).
    ///
    /// Adaptive: only sorts the bytes that contain nonzero data based on
    /// `max_rank`. For max_rank < 256, only 2 passes instead of 8.
    ///
    /// Stable sort: elements with equal keys preserve their input order.
    /// Combined with descending initial sa[], this matches CPU SA-IS tiebreaking.
    ///
    /// Returns the final event for chaining. All internal kernel dispatches
    /// are chained via events with no host waits.
    #[allow(clippy::too_many_arguments)]
    fn run_radix_sort(
        &self,
        sa_buf: &mut Buffer<cl_uint>,
        sa_buf_alt: &mut Buffer<cl_uint>,
        rank_buf: &Buffer<cl_uint>,
        keys_buf: &mut Buffer<cl_uint>,
        histogram_buf: &mut Buffer<cl_uint>,
        histogram_buf_scan: &mut Buffer<cl_uint>,
        padded_n: usize,
        n: cl_uint,
        k_doubling: cl_uint,
        hist_block_sums_l0: &mut Buffer<cl_uint>,
        hist_block_sums_l1: &mut Buffer<cl_uint>,
        hist_block_sums_l2: &mut Buffer<cl_uint>,
        hist_num_blocks_l0: usize,
        hist_num_blocks_l1: usize,
        max_rank: cl_uint,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let padded_n_arg = padded_n as cl_uint;
        let wg = self.scan_workgroup_size;
        let num_groups = padded_n.div_ceil(wg);
        let num_groups_arg = num_groups as cl_uint;
        let histogram_len = 256 * num_groups;
        let global_wg = num_groups * wg;

        // Adaptive pass selection: skip zero-byte passes.
        // Composite key = (r1 << 32) | r2, both in [0, max_rank].
        // Passes 0..bytes_needed sort r2 bytes, passes 4..4+bytes_needed sort r1.
        let bytes_needed: u32 = if max_rank < 256 {
            1
        } else if max_rank < 65536 {
            2
        } else if max_rank < 16_777_216 {
            3
        } else {
            4
        };
        // Build pass list: r2 low bytes then r1 low bytes
        let mut passes: Vec<u32> = (0..bytes_needed).chain(4..4 + bytes_needed).collect();
        // Ensure even count so sa_buf holds result (each pass swaps sa_buf ↔ sa_buf_alt)
        debug_assert!(passes.len().is_multiple_of(2), "pass count must be even");
        if !passes.len().is_multiple_of(2) {
            passes.push(bytes_needed); // no-op pass on zero byte
        }

        let mut prev_event: Option<Event> = None;

        for (i, &pass) in passes.iter().enumerate() {
            let pass_arg = pass as cl_uint;
            let wait_ref: Option<&Event> = if i == 0 {
                wait_event
            } else {
                prev_event.as_ref()
            };

            // Phase 1: Compute 8-bit digit for each element
            let key_event = unsafe {
                let mut exec = ExecuteKernel::new(self.kernel_radix_compute_keys());
                exec.set_arg(sa_buf as &Buffer<cl_uint>)
                    .set_arg(rank_buf)
                    .set_arg(keys_buf)
                    .set_arg(&n)
                    .set_arg(&padded_n_arg)
                    .set_arg(&k_doubling)
                    .set_arg(&pass_arg)
                    .set_global_work_size(global_wg);
                if let Some(evt) = wait_ref {
                    exec.set_wait_event(evt);
                }
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Phase 2: Per-workgroup histogram
            let hist_event = unsafe {
                let mut exec = ExecuteKernel::new(self.kernel_radix_histogram());
                exec.set_arg(keys_buf as &Buffer<cl_uint>)
                    .set_arg(histogram_buf)
                    .set_arg(&padded_n_arg)
                    .set_arg(&num_groups_arg)
                    .set_global_work_size(global_wg)
                    .set_local_work_size(wg)
                    .set_wait_event(&key_event);
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Phase 3: Inclusive prefix sum over histogram
            let prefix_event = self.run_prefix_sum(
                histogram_buf,
                histogram_buf_scan,
                histogram_len,
                hist_block_sums_l0,
                hist_block_sums_l1,
                hist_block_sums_l2,
                hist_num_blocks_l0,
                hist_num_blocks_l1,
                Some(&hist_event),
            )?;

            // Phase 3b: Convert inclusive to exclusive prefix sum
            let hist_len_arg = histogram_len as cl_uint;
            let excl_event = unsafe {
                let mut exec = ExecuteKernel::new(self.kernel_inclusive_to_exclusive());
                exec.set_arg(histogram_buf_scan as &Buffer<cl_uint>)
                    .set_arg(histogram_buf)
                    .set_arg(&hist_len_arg)
                    .set_global_work_size(histogram_len)
                    .set_wait_event(&prefix_event);
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Phase 4: Scatter sa elements to sorted positions
            let scatter_event = unsafe {
                let mut exec = ExecuteKernel::new(self.kernel_radix_scatter());
                exec.set_arg(sa_buf as &Buffer<cl_uint>)
                    .set_arg(keys_buf as &Buffer<cl_uint>)
                    .set_arg(histogram_buf)
                    .set_arg(sa_buf_alt)
                    .set_arg(&padded_n_arg)
                    .set_arg(&num_groups_arg)
                    .set_global_work_size(global_wg)
                    .set_local_work_size(wg)
                    .set_wait_event(&excl_event);
                exec.enqueue_nd_range(&self.queue)
                    .map_err(|_| PzError::Unsupported)?
            };

            // Swap: sa_buf_alt (sorted output) becomes sa_buf for next pass
            std::mem::swap(sa_buf, sa_buf_alt);
            prev_event = Some(scatter_event);
        }

        // After an even number of passes, sa_buf holds the final sorted result
        Ok(prev_event.expect("at least 2 radix passes"))
    }

    /// Run the rank comparison kernel (BWT prefix-doubling phase 1).
    ///
    /// For each sorted position i, computes diff[i] = 1 if the composite key
    /// of sa[i] differs from sa[i-1], 0 otherwise.
    /// Returns the kernel completion event for chaining.
    #[allow(clippy::too_many_arguments)]
    fn run_rank_compare(
        &self,
        sa_buf: &Buffer<cl_uint>,
        rank_buf: &Buffer<cl_uint>,
        diff_buf: &mut Buffer<cl_uint>,
        n: cl_uint,
        padded_n: cl_uint,
        k: cl_uint,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(self.kernel_rank_compare());
            exec.set_arg(sa_buf)
                .set_arg(rank_buf)
                .set_arg(diff_buf)
                .set_arg(&n)
                .set_arg(&padded_n)
                .set_arg(&k)
                .set_global_work_size(padded_n as usize);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }

    /// Run a multi-level inclusive prefix sum on the GPU.
    ///
    /// Reads from `input_buf`, writes inclusive prefix sums to `output_buf`.
    /// Uses a two- or three-level workgroup scan depending on input size.
    /// Returns the final event for chaining.
    ///
    /// Note: temporary buffers (`block_sums_l*_scanned`) are dropped on return,
    /// but the OpenCL runtime retains memory objects for enqueued kernels, so
    /// they remain valid until the returned event (and its dependencies) complete.
    #[allow(clippy::too_many_arguments)]
    fn run_prefix_sum(
        &self,
        input_buf: &Buffer<cl_uint>,
        output_buf: &mut Buffer<cl_uint>,
        count: usize,
        block_sums_l0: &mut Buffer<cl_uint>,
        block_sums_l1: &mut Buffer<cl_uint>,
        block_sums_l2: &mut Buffer<cl_uint>,
        num_blocks_l0: usize,
        num_blocks_l1: usize,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let wg = self.scan_workgroup_size;
        let block_elems = wg * 2;

        // Level 0: per-workgroup scan of input → output, block totals → block_sums_l0
        let evt_l0 =
            self.run_prefix_sum_local(input_buf, output_buf, block_sums_l0, count, wg, wait_event)?;

        if num_blocks_l0 <= 1 {
            // Single workgroup: output is already the final inclusive prefix sum
            return Ok(evt_l0);
        }

        // Level 1: scan block_sums_l0 → block_sums_l0_scanned
        let mut block_sums_l0_scanned = unsafe {
            Buffer::<cl_uint>::create(
                &self.context,
                CL_MEM_READ_WRITE,
                num_blocks_l0.max(1),
                ptr::null_mut(),
            )
            .map_err(|_| PzError::BufferTooSmall)?
        };
        let evt_l1 = self.run_prefix_sum_local(
            block_sums_l0,
            &mut block_sums_l0_scanned,
            block_sums_l1,
            num_blocks_l0,
            wg,
            Some(&evt_l0),
        )?;

        let evt_offsets = if num_blocks_l1 > 1 {
            // Level 2: scan block_sums_l1 → block_sums_l1_scanned
            let mut block_sums_l1_scanned = unsafe {
                Buffer::<cl_uint>::create(
                    &self.context,
                    CL_MEM_READ_WRITE,
                    num_blocks_l1.max(1),
                    ptr::null_mut(),
                )
                .map_err(|_| PzError::BufferTooSmall)?
            };
            let evt_l2 = self.run_prefix_sum_local(
                block_sums_l1,
                &mut block_sums_l1_scanned,
                block_sums_l2,
                num_blocks_l1,
                wg,
                Some(&evt_l1),
            )?;

            // Propagate L2 offsets into L1 scanned sums
            self.run_prefix_sum_propagate(
                &mut block_sums_l0_scanned,
                &block_sums_l1_scanned,
                num_blocks_l0,
                block_elems,
                Some(&evt_l2),
            )?
        } else {
            evt_l1
        };

        // Propagate L1 (or L2-fixed L1) offsets into the L0 output
        let evt_final = self.run_prefix_sum_propagate(
            output_buf,
            &block_sums_l0_scanned,
            count,
            block_elems,
            Some(&evt_offsets),
        )?;

        Ok(evt_final)
    }

    /// Dispatch a single-level per-workgroup prefix sum kernel.
    ///
    /// Returns the kernel completion event for chaining.
    fn run_prefix_sum_local(
        &self,
        input_buf: &Buffer<cl_uint>,
        output_buf: &mut Buffer<cl_uint>,
        block_sums_buf: &mut Buffer<cl_uint>,
        count: usize,
        wg_size: usize,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let count_arg = count as cl_uint;
        let global_size = count.div_ceil(wg_size * 2) * wg_size;
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(self.kernel_prefix_sum_local());
            exec.set_arg(input_buf)
                .set_arg(output_buf)
                .set_arg(block_sums_buf)
                .set_arg(&count_arg)
                .set_global_work_size(global_size)
                .set_local_work_size(wg_size);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }

    /// Dispatch the prefix sum propagation kernel (add block offsets).
    ///
    /// Returns the kernel completion event for chaining.
    fn run_prefix_sum_propagate(
        &self,
        data_buf: &mut Buffer<cl_uint>,
        offsets_buf: &Buffer<cl_uint>,
        count: usize,
        block_elems: usize,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let count_arg = count as cl_uint;
        let _ = block_elems; // BLOCK_ELEMS is a compile-time constant in the kernel
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(self.kernel_prefix_sum_propagate());
            exec.set_arg(data_buf)
                .set_arg(offsets_buf)
                .set_arg(&count_arg)
                .set_global_work_size(count);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }

    /// Run the rank scatter kernel (BWT prefix-doubling phase 3).
    ///
    /// Writes new_rank[sa[i]] = prefix[i] for real entries, UINT_MAX for sentinels.
    /// Returns the kernel completion event for chaining.
    fn run_rank_scatter(
        &self,
        sa_buf: &Buffer<cl_uint>,
        prefix_buf: &Buffer<cl_uint>,
        new_rank_buf: &mut Buffer<cl_uint>,
        n: cl_uint,
        padded_n: cl_uint,
        wait_event: Option<&Event>,
    ) -> PzResult<Event> {
        let kernel_event = unsafe {
            let mut exec = ExecuteKernel::new(self.kernel_rank_scatter());
            exec.set_arg(sa_buf)
                .set_arg(prefix_buf)
                .set_arg(new_rank_buf)
                .set_arg(&n)
                .set_arg(&padded_n)
                .set_global_work_size(padded_n as usize);
            if let Some(evt) = wait_event {
                exec.set_wait_event(evt);
            }
            exec.enqueue_nd_range(&self.queue)
                .map_err(|_| PzError::Unsupported)?
        };
        Ok(kernel_event)
    }
}
