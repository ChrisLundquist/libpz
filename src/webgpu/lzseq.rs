//! GPU-accelerated LzSeq demux: match buffer → 6 compressed streams.
//!
//! Chains GPU match finding (cooperative-stitch) with a serial demux kernel
//! that walks the per-position match buffer and produces the 6 LzSeq output
//! streams (flags, literals, offset_codes, offset_extra, length_codes,
//! length_extra) entirely on-device.
//!
//! Eliminates the PCIe bottleneck of downloading 12 bytes/position from
//! the GPU match buffer. The compressed output streams are typically <10%
//! of the match buffer size.

use super::*;
use crate::lzseq::SeqEncoded;

impl WebGpuEngine {
    /// GPU-accelerated LzSeq encoding: match finding + demux on-device.
    ///
    /// Chains `find_matches_to_device()` (cooperative-stitch kernel) with
    /// the `lzseq_demux` kernel in back-to-back queue submissions. Only
    /// the small compressed streams are downloaded to CPU.
    ///
    /// No repeat offsets are used on GPU — all offset codes are literal
    /// (shifted by 3). The decoder handles this transparently.
    pub fn lzseq_encode_gpu(&self, input: &[u8]) -> PzResult<SeqEncoded> {
        if input.is_empty() {
            return Ok(SeqEncoded {
                flags: Vec::new(),
                literals: Vec::new(),
                offset_codes: Vec::new(),
                offset_extra: Vec::new(),
                length_codes: Vec::new(),
                length_extra: Vec::new(),
                num_tokens: 0,
                num_matches: 0,
            });
        }

        let input_len = input.len();

        // Step 1: GPU match finding (stays on device, also retains input buffer)
        let match_buf = self.find_matches_to_device(input)?;

        // Step 2: Compute output buffer layout (worst-case sizes)
        let layout = OutputLayout::new(input_len);

        // Allocate output buffer (driver zero-initialized, no host upload needed)
        let output_buf = self.create_buffer(
            "lzseq_output",
            (layout.total_words * 4) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        // Step 3: Params uniform (section offsets within output buffer)
        let params = [
            input_len as u32,
            layout.flags_offset as u32,
            layout.literals_offset as u32,
            layout.offset_codes_offset as u32,
            layout.offset_extra_offset as u32,
            layout.length_codes_offset as u32,
            layout.length_extra_offset as u32,
            0u32, // padding
        ];
        let params_buf = self.create_buffer_init(
            "lzseq_params",
            bytemuck::cast_slice(&params),
            wgpu::BufferUsages::UNIFORM,
        );

        // Step 4: Create bind group and dispatch
        // Reuse the input buffer from match finding (no second upload)
        let pipeline = self.pipeline_lzseq_demux();
        let bg_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lzseq_demux_bg"),
            layout: &bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: match_buf.input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: match_buf.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buf.as_entire_binding(),
                },
            ],
        });

        let t0 = if self.profiling {
            Some(std::time::Instant::now())
        } else {
            None
        };

        // Single workgroup, single thread — the demux walk is inherently serial
        self.dispatch(pipeline, &bind_group, 1, "lzseq_demux")?;

        if let Some(t0) = t0 {
            self.poll_wait();
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            eprintln!("[pz-gpu] lzseq_demux: {ms:.3} ms");
        }

        // Step 5: Read back output and extract streams
        let raw = self.read_buffer(&output_buf, (layout.total_words * 4) as u64);

        extract_lzseq_streams(&raw, &layout)
    }
}

/// Extract the 6 LzSeq streams from the raw GPU output buffer.
///
/// Returns `Err(InvalidInput)` if counter values exceed buffer bounds.
fn extract_lzseq_streams(raw: &[u8], layout: &OutputLayout) -> PzResult<SeqEncoded> {
    if raw.len() < layout.total_words * 4 || raw.len() < 20 {
        return Err(PzError::InvalidInput);
    }
    let output_u32: &[u32] = bytemuck::cast_slice(raw);

    // Read counters
    let num_tokens = output_u32[0];
    let num_matches = output_u32[1];
    let num_literals = output_u32[2];
    let off_extra_total_bits = output_u32[3];
    let len_extra_total_bits = output_u32[4];

    // Extract flags (packed MSB-first, ceil(num_tokens/8) bytes)
    let flags_bytes = (num_tokens as usize).div_ceil(8);
    let flags_start = layout.flags_offset * 4;
    let flags_end = flags_start + flags_bytes;
    if flags_end > raw.len() {
        return Err(PzError::InvalidInput);
    }
    let flags = raw[flags_start..flags_end].to_vec();

    // Extract literals (1 byte per literal token)
    let literals_start = layout.literals_offset * 4;
    let literals_end = literals_start + num_literals as usize;
    if literals_end > raw.len() {
        return Err(PzError::InvalidInput);
    }
    let literals = raw[literals_start..literals_end].to_vec();

    // Extract offset codes (1 byte per match)
    let oc_start = layout.offset_codes_offset * 4;
    let oc_end = oc_start + num_matches as usize;
    if oc_end > raw.len() {
        return Err(PzError::InvalidInput);
    }
    let offset_codes = raw[oc_start..oc_end].to_vec();

    // Extract offset extra bits (packed LSB-first bitstream)
    let oextra_bytes = (off_extra_total_bits as usize).div_ceil(8);
    let oextra_start = layout.offset_extra_offset * 4;
    let oextra_end = oextra_start + oextra_bytes;
    if oextra_end > raw.len() {
        return Err(PzError::InvalidInput);
    }
    let offset_extra = raw[oextra_start..oextra_end].to_vec();

    // Extract length codes (1 byte per match)
    let lc_start = layout.length_codes_offset * 4;
    let lc_end = lc_start + num_matches as usize;
    if lc_end > raw.len() {
        return Err(PzError::InvalidInput);
    }
    let length_codes = raw[lc_start..lc_end].to_vec();

    // Extract length extra bits (packed LSB-first bitstream)
    let lextra_bytes = (len_extra_total_bits as usize).div_ceil(8);
    let lextra_start = layout.length_extra_offset * 4;
    let lextra_end = lextra_start + lextra_bytes;
    if lextra_end > raw.len() {
        return Err(PzError::InvalidInput);
    }
    let length_extra = raw[lextra_start..lextra_end].to_vec();

    Ok(SeqEncoded {
        flags,
        literals,
        offset_codes,
        offset_extra,
        length_codes,
        length_extra,
        num_tokens,
        num_matches,
    })
}

/// Output buffer layout: section offsets (in u32 words) for each stream.
///
/// The GPU kernel writes all 6 streams into a single contiguous buffer
/// at these pre-computed offsets. Worst-case sizes are used to avoid
/// any output overflow.
struct OutputLayout {
    /// Offset for flags section (u32 words, after 8-word counter region)
    flags_offset: usize,
    /// Offset for literals section
    literals_offset: usize,
    /// Offset for offset_codes section
    offset_codes_offset: usize,
    /// Offset for offset_extra section (bit-packed)
    offset_extra_offset: usize,
    /// Offset for length_codes section
    length_codes_offset: usize,
    /// Offset for length_extra section (bit-packed)
    length_extra_offset: usize,
    /// Total size of the output buffer in u32 words
    total_words: usize,
}

impl OutputLayout {
    fn new(input_len: usize) -> Self {
        // Worst-case sizes:
        // - num_tokens ≤ input_len (all literals)
        // - num_matches ≤ input_len / MIN_MATCH (each match ≥ 3 bytes)
        // - extra bits per match ≤ 20 (offset or length code up to ~20)
        let max_matches = input_len / 3 + 1;

        // Counter region: 5 u32 counters, padded to 8 words for alignment
        let counters_words = 8;

        // Flags: ceil(input_len / 32) u32 words (1 bit per token)
        let flags_words = input_len.div_ceil(32);

        // Literals: ceil(input_len / 4) u32 words (worst case: all literals)
        let literals_words = input_len.div_ceil(4);

        // Offset codes: ceil(max_matches / 4) u32 words (1 byte per match)
        let codes_words = max_matches.div_ceil(4);

        // Extra bits: ceil(max_matches * 20 / 32) u32 words (worst case: 20 bits each)
        let extra_words = (max_matches * 20).div_ceil(32) + 1; // +1 for bit-spanning writes

        let flags_offset = counters_words;
        let literals_offset = flags_offset + flags_words;
        let offset_codes_offset = literals_offset + literals_words;
        let offset_extra_offset = offset_codes_offset + codes_words;
        let length_codes_offset = offset_extra_offset + extra_words;
        let length_extra_offset = length_codes_offset + codes_words;
        let total_words = length_extra_offset + extra_words;

        OutputLayout {
            flags_offset,
            literals_offset,
            offset_codes_offset,
            offset_extra_offset,
            length_codes_offset,
            length_extra_offset,
            total_words,
        }
    }
}
