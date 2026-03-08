//! GPU-resident buffer types for zero-copy stage chaining.

use super::WebGpuEngine;
use crate::lz77::Match;
use crate::PzResult;

/// GPU match struct matching the WGSL kernel's Lz77Match layout.
/// 3 x u32 = 12 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct GpuMatch {
    pub(super) offset: u32,
    pub(super) length: u32,
    pub(super) next: u32, // next byte in low 8 bits
}

// SAFETY: GpuMatch is repr(C) with all-u32 fields, which are Pod/Zeroable.
unsafe impl bytemuck::Pod for GpuMatch {}
unsafe impl bytemuck::Zeroable for GpuMatch {}

/// A buffer residing on the GPU device.
///
/// Data stays on-device until explicitly downloaded via [`read_to_host()`].
/// This avoids unnecessary PCI-bus round-trips when one GPU stage feeds
/// directly into another (e.g., LZ77 output → Huffman histogram on the
/// same device buffer).
pub struct DeviceBuf {
    pub(crate) buf: wgpu::Buffer,
    pub(crate) len: usize,
}

impl DeviceBuf {
    /// Upload host data to the GPU, returning a device-resident buffer.
    ///
    /// The data is padded to u32-aligned + 4 bytes (matching WGSL's `array<u32>`
    /// byte reading convention). The `len` field stores the logical (unpadded)
    /// length.
    pub fn from_host(engine: &WebGpuEngine, data: &[u8]) -> PzResult<Self> {
        if data.is_empty() {
            let buf = engine.create_buffer(
                "device_buf_empty",
                4,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            return Ok(DeviceBuf { buf, len: 0 });
        }

        let padded = WebGpuEngine::pad_input_bytes(data);
        let buf = engine.create_buffer_init(
            "device_buf",
            &padded,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        Ok(DeviceBuf {
            buf,
            len: data.len(),
        })
    }

    /// Allocate a device buffer of the given size.
    ///
    /// **Note:** The buffer contents are *not* guaranteed to be zero-initialized.
    /// Callers that need zeroed memory should write zeros explicitly.
    pub fn alloc(engine: &WebGpuEngine, len: usize) -> PzResult<Self> {
        let actual_len = len.max(4); // avoid zero-size allocation
        let buf = engine.create_buffer(
            "device_buf_alloc",
            actual_len as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        Ok(DeviceBuf { buf, len })
    }

    /// Download the buffer contents from the GPU to host memory.
    pub fn read_to_host(&self, engine: &WebGpuEngine) -> PzResult<Vec<u8>> {
        if self.len == 0 {
            return Ok(Vec::new());
        }

        // The buffer may be padded, so read the full buffer and truncate.
        let raw = engine.read_buffer(&self.buf, self.buf.size());
        Ok(raw[..self.len].to_vec())
    }

    /// The logical length of the data in this buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether this buffer is empty (zero-length data).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// An opaque GPU-resident buffer of LZ77 match results.
///
/// Produced by [`WebGpuEngine::find_matches_to_device()`] and consumed by
/// [`WebGpuEngine::download_and_dedupe()`]. The match data stays on the GPU
/// until explicitly downloaded, enabling zero-copy stage chaining.
pub struct GpuMatchBuf {
    pub(crate) buf: wgpu::Buffer,
    pub(crate) input_len: usize,
    /// The original input bytes on-device (reusable by downstream kernels).
    pub(crate) input_buf: wgpu::Buffer,
}

impl GpuMatchBuf {
    /// The number of input positions this match buffer covers.
    pub fn input_len(&self) -> usize {
        self.input_len
    }
}

/// Deduplicate raw GPU match output into a non-overlapping match sequence.
pub(super) fn dedupe_gpu_matches(gpu_matches: &[GpuMatch], input: &[u8]) -> Vec<Match> {
    if gpu_matches.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let len = gpu_matches.len();
    let mut index = 0;

    while index < len {
        let gm = &gpu_matches[index];

        let remaining = len - index;
        let mut match_length = gm.length as usize;
        if match_length >= remaining {
            match_length = if remaining > 0 { remaining - 1 } else { 0 };
        }
        // Cap to u16::MAX since Match.length is u16. The next iteration will
        // pick up a fresh per-position match from the GPU output.
        if match_length > u16::MAX as usize {
            match_length = u16::MAX as usize;
        }

        let next = if index + match_length < input.len() {
            input[index + match_length]
        } else {
            gm.next as u8
        };

        result.push(Match {
            offset: gm.offset as u16,
            length: match_length as u16,
            next,
        });

        index += match_length + 1;
    }

    result
}
