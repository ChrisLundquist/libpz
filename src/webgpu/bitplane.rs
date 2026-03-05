//! GPU bit-plane transpose for Experiment D.
//!
//! Transposes input bytes into 8 bit-plane streams on the GPU.
//! This is the GPU throughput ceiling benchmark — zero serial stages,
//! zero data-dependent branching.

use super::*;

impl WebGpuEngine {
    /// GPU-accelerated bit-plane transpose.
    ///
    /// Returns 8 byte vectors, one per bit plane. Each plane contains
    /// ceil(n/8) bytes with bits packed MSB-first.
    pub fn bitplane_transpose(&self, input: &[u8]) -> PzResult<[Vec<u8>; 8]> {
        let n = input.len();
        if n == 0 {
            return Err(PzError::InvalidInput);
        }

        let plane_bytes = n.div_ceil(8);
        let plane_u32s = plane_bytes.div_ceil(4);
        let total_output_u32s = 8 * plane_u32s;

        // Pad input to u32-aligned.
        let padded = Self::pad_input_bytes(input);
        let input_buf =
            self.create_buffer_init("bitplane_input", &padded, wgpu::BufferUsages::STORAGE);

        // Output buffer: 8 planes × plane_u32s × 4 bytes, initialized to zero.
        let output_size = (total_output_u32s * 4) as u64;
        let output_buf = self.create_buffer_init(
            "bitplane_output",
            &vec![0u8; output_size as usize],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Parameters.
        let params = [n as u32, plane_bytes as u32, 0u32, 0u32];
        let params_buf = self.create_buffer_init(
            "bitplane_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Create bind group.
        let pipeline = self.pipeline_bitplane_transpose();
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bitplane_transpose_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        // Dispatch: one thread per input byte, workgroup size 256.
        let workgroups = (n as u32).div_ceil(256);
        self.dispatch(pipeline, &bg, workgroups, "bitplane_transpose")?;

        // Read back output.
        let raw = self.read_buffer(&output_buf, output_size);
        let raw_u32s: &[u32] = bytemuck::cast_slice(&raw);

        // Split into 8 planes.
        let mut planes: [Vec<u8>; 8] = std::array::from_fn(|_| Vec::new());
        for (i, plane) in planes.iter_mut().enumerate() {
            let start = i * plane_u32s;
            let end = start + plane_u32s;
            let plane_data: Vec<u8> = raw_u32s[start..end]
                .iter()
                .flat_map(|w| w.to_ne_bytes())
                .take(plane_bytes)
                .collect();
            *plane = plane_data;
        }

        Ok(planes)
    }

    /// GPU-accelerated bitplane compression.
    ///
    /// Runs the full pipeline: GPU transpose → CPU RLE per plane → CPU FSE encode.
    /// This measures the GPU throughput ceiling including transfer overhead.
    pub fn bitplane_compress(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        let n = input.len();
        if n == 0 {
            return Err(PzError::InvalidInput);
        }

        // Step 1: GPU bit transpose.
        let planes = self.bitplane_transpose(input)?;

        // Steps 2-3: CPU RLE + FSE per plane (same as CPU path).
        // The GPU experiment measures transpose throughput; RLE+FSE stays on CPU
        // since each plane is only N/8 bytes (too small for GPU benefit).
        let mut output = Vec::new();
        output.extend_from_slice(&(n as u32).to_le_bytes());

        for plane in &planes {
            let rle_data = crate::bitplane::rle_binary_for_gpu(plane, n);
            let rle_raw_len = rle_data.len();
            let fse_data = crate::fse::encode(&rle_data);

            output.extend_from_slice(&(rle_raw_len as u32).to_le_bytes());
            output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
            output.extend_from_slice(&fse_data);
        }

        Ok(output)
    }
}
