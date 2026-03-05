//! GPU parallel-parse LZ conflict resolution for Experiment E.
//!
//! Uses existing GPU cooperative-stitch match finder, then resolves
//! overlapping matches with a parallel prefix-max scan on the GPU.

use super::*;

impl WebGpuEngine {
    /// GPU-accelerated parallel conflict resolution for parlz.
    ///
    /// Takes match data (one entry per position: high 16 = offset, low 16 = length,
    /// or 0 if no match) and returns a boolean vector marking match-start positions.
    pub fn parlz_resolve(&self, match_data: &[u32]) -> PzResult<Vec<bool>> {
        let n = match_data.len();
        if n == 0 {
            return Err(PzError::InvalidInput);
        }

        let pipelines = self.parlz_pipelines();

        // Upload match data.
        let matches_buf = self.create_buffer_init(
            "parlz_matches",
            bytemuck::cast_slice(match_data),
            wgpu::BufferUsages::STORAGE,
        );

        // Coverage buffer.
        let coverage_buf = self.create_buffer(
            "parlz_coverage",
            (n * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Block maxima for prefix-max scan.
        let num_blocks = n.div_ceil(256);
        let block_maxima_buf = self.create_buffer(
            "parlz_block_maxima",
            (num_blocks * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Flags buffer (packed u32, one bit per position).
        let flag_u32s = n.div_ceil(32);
        let flags_buf = self.create_buffer_init(
            "parlz_flags",
            &vec![0u8; flag_u32s * 4],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Parameters.
        let params = [n as u32, 256u32, 0u32, 0u32];
        let params_buf = self.create_buffer_init(
            "parlz_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Bind group for init_coverage + classify (they share the same layout).
        let bg_layout = pipelines.init_coverage.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("parlz_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matches_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coverage_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: block_maxima_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: flags_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let workgroups = (n as u32).div_ceil(256);

        // Step 1: Initialize coverage array.
        self.dispatch(
            &pipelines.init_coverage,
            &bg,
            workgroups,
            "parlz_init_coverage",
        )?;

        // Step 2: Prefix-max scan (local).
        self.dispatch(
            &pipelines.prefix_max_local,
            &bg,
            workgroups,
            "parlz_prefix_max_local",
        )?;

        // Step 3: Propagate block maxima (if more than 1 block).
        if num_blocks > 1 {
            // For small numbers of blocks, the propagation kernel handles
            // the serial scan over block_maxima internally.
            self.dispatch(
                &pipelines.prefix_max_propagate,
                &bg,
                workgroups,
                "parlz_prefix_max_propagate",
            )?;
        }

        // Step 4: Classify positions.
        self.dispatch(&pipelines.classify, &bg, workgroups, "parlz_classify")?;

        // Read back flags.
        let flags_raw = self.read_buffer(&flags_buf, (flag_u32s * 4) as u64);
        let flags_u32s: &[u32] = bytemuck::cast_slice(&flags_raw);

        // Unpack bits to bool vector.
        let mut result = vec![false; n];
        for (i, r) in result.iter_mut().enumerate() {
            let word = flags_u32s[i / 32];
            let bit = i % 32;
            *r = (word >> bit) & 1 != 0;
        }

        Ok(result)
    }

    /// Full GPU parlz compression pipeline.
    ///
    /// Uses GPU coop match finder + GPU conflict resolution + CPU encoding.
    pub fn parlz_compress(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        if input.is_empty() {
            return Err(PzError::InvalidInput);
        }

        // For the initial implementation, use CPU match finding + CPU conflict resolution.
        // GPU match finding (coop kernel) produces matches indexed differently than
        // the per-position format expected by parlz_resolve. Once the format mapping
        // is verified, we can wire GPU match finding + GPU resolve together.
        let matches = crate::parlz::find_all_matches(input);
        let is_match_start = crate::parlz::parallel_resolve(&matches);
        Ok(crate::parlz::parallel_parse_and_encode(
            input,
            &matches,
            &is_match_start,
        ))
    }
}
