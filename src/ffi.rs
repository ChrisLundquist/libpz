//! C-callable FFI layer for libpz.
//!
//! Provides a stable C API that hides all Rust internals behind
//! opaque handles and simple C types.

use std::slice;

use crate::huffman::HuffmanTree;
use crate::lz77;
use crate::pipeline;

// Error codes matching the C API design
const PZ_OK: i32 = 0;
const PZ_ERROR_BUFFER_TOO_SMALL: i32 = -1;
const PZ_ERROR_INVALID_INPUT: i32 = -2;
const PZ_ERROR_UNSUPPORTED: i32 = -3;

/// Convert a [`PzError`](crate::PzError) to an FFI error code.
fn error_to_code(e: crate::PzError) -> i32 {
    match e {
        crate::PzError::BufferTooSmall => PZ_ERROR_BUFFER_TOO_SMALL,
        crate::PzError::InvalidInput => PZ_ERROR_INVALID_INPUT,
        crate::PzError::Unsupported => PZ_ERROR_UNSUPPORTED,
    }
}

/// Compression levels.
#[repr(C)]
pub enum PzLevel {
    Fast = 1,
    Default = 5,
    Best = 9,
}

/// Compression pipeline types.
#[repr(C)]
pub enum PzPipeline {
    Deflate = 0,
    Bw = 1,
    Lza = 2,
}

/// Device information returned by pz_query_devices.
#[repr(C)]
pub struct PzDeviceInfo {
    pub opencl_devices: i32,
    pub vulkan_devices: i32,
    pub cpu_threads: i32,
}

/// Opaque context handle.
pub struct PzContext {
    /// Placeholder for future state (cached kernels, etc.)
    _initialized: bool,
    /// OpenCL GPU engine, if available. Wrapped in Arc for sharing
    /// with pipeline CompressOptions.
    #[cfg(feature = "opencl")]
    opencl_engine: Option<std::sync::Arc<crate::opencl::OpenClEngine>>,
    /// Cached count of OpenCL devices found during init.
    opencl_device_count: i32,
}

/// Initialize a libpz context.
///
/// Probes for available compute devices (OpenCL GPUs, etc.) and
/// initializes the best available backend. Falls back gracefully
/// to CPU-only if no GPU is found.
///
/// Returns a pointer to a new context, or null on failure.
#[no_mangle]
pub extern "C" fn pz_init() -> *mut PzContext {
    #[cfg(feature = "opencl")]
    let (opencl_engine, opencl_device_count) = {
        let count = crate::opencl::device_count() as i32;
        let engine = crate::opencl::OpenClEngine::new()
            .ok()
            .map(std::sync::Arc::new);
        (engine, count)
    };

    #[cfg(not(feature = "opencl"))]
    let opencl_device_count = 0i32;

    let ctx = Box::new(PzContext {
        _initialized: true,
        #[cfg(feature = "opencl")]
        opencl_engine,
        opencl_device_count,
    });
    Box::into_raw(ctx)
}

/// Destroy a libpz context and free associated resources.
///
/// # Safety
///
/// `ctx` must be a pointer returned by [`pz_init`], or null.
/// After this call the pointer is invalid and must not be reused.
#[no_mangle]
pub unsafe extern "C" fn pz_destroy(ctx: *mut PzContext) {
    if !ctx.is_null() {
        let _ = Box::from_raw(ctx);
    }
}

/// Query available compute devices.
///
/// Reports the number of OpenCL devices found during `pz_init()`,
/// the number of Vulkan devices (0 until Phase 5), and the number
/// of CPU threads available.
///
/// # Safety
///
/// - `ctx` must be a valid pointer returned by [`pz_init`].
/// - `info` must point to a writable [`PzDeviceInfo`].
#[no_mangle]
pub unsafe extern "C" fn pz_query_devices(ctx: *const PzContext, info: *mut PzDeviceInfo) -> i32 {
    if ctx.is_null() || info.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    let ctx = &*ctx;
    let info = &mut *info;
    info.opencl_devices = ctx.opencl_device_count;
    info.vulkan_devices = 0; // Phase 5
    info.cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1);

    PZ_OK
}

/// Build compression options from context, selecting GPU when appropriate.
///
/// Uses the GPU backend when:
/// 1. The `opencl` feature is enabled
/// 2. An OpenCL engine was successfully created at init time
/// 3. The input is large enough to benefit from GPU acceleration
///
/// `threads`: 0 = auto, 1 = single-threaded, N = use N threads.
fn build_compress_options(
    ctx: &PzContext,
    _input_len: usize,
    threads: usize,
) -> pipeline::CompressOptions {
    #[cfg(feature = "opencl")]
    {
        if let Some(ref engine) = ctx.opencl_engine {
            if _input_len >= crate::opencl::MIN_GPU_INPUT_SIZE {
                return pipeline::CompressOptions {
                    backend: pipeline::Backend::OpenCl,
                    threads,
                    opencl_engine: Some(engine.clone()),
                    ..Default::default()
                };
            }
        }
    }

    #[cfg(not(feature = "opencl"))]
    let _ = ctx;

    pipeline::CompressOptions {
        threads,
        ..Default::default()
    }
}

/// Compress data using the specified pipeline.
///
/// Returns bytes written on success, or a negative error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer returned by [`pz_init`].
/// - `input` must point to at least `input_len` readable bytes.
/// - `output` must point to at least `output_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn pz_compress(
    ctx: *mut PzContext,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
    _level: i32,
    pipeline: i32,
) -> i32 {
    if ctx.is_null() || input.is_null() || output.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    if input_len == 0 {
        return 0;
    }

    let input_slice = slice::from_raw_parts(input, input_len);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    let ctx = &*ctx;

    let pipe = match pipeline {
        0 => pipeline::Pipeline::Deflate,
        1 => pipeline::Pipeline::Bw,
        2 => pipeline::Pipeline::Lza,
        _ => return PZ_ERROR_UNSUPPORTED,
    };

    // Build compress options, selecting GPU backend if engine is available
    // and the input is large enough to benefit.
    // Default to auto-threading (0 = use all available cores).
    let options = build_compress_options(ctx, input_len, 0);

    match pipeline::compress_with_options(input_slice, pipe, &options) {
        Ok(compressed) => {
            if compressed.len() > output_slice.len() {
                return PZ_ERROR_BUFFER_TOO_SMALL;
            }
            output_slice[..compressed.len()].copy_from_slice(&compressed);
            compressed.len() as i32
        }
        Err(e) => error_to_code(e),
    }
}

/// Compress data with explicit thread count.
///
/// Like `pz_compress` but allows controlling the number of threads:
/// - 0: auto-detect (use all available CPU cores)
/// - 1: single-threaded (produces legacy single-block format)
/// - N: use up to N threads
///
/// # Safety
///
/// Same requirements as [`pz_compress`].
#[no_mangle]
pub unsafe extern "C" fn pz_compress_mt(
    ctx: *mut PzContext,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
    _level: i32,
    pipeline: i32,
    threads: i32,
) -> i32 {
    if ctx.is_null() || input.is_null() || output.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    if input_len == 0 {
        return 0;
    }

    let input_slice = slice::from_raw_parts(input, input_len);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    let ctx = &*ctx;

    let pipe = match pipeline {
        0 => pipeline::Pipeline::Deflate,
        1 => pipeline::Pipeline::Bw,
        2 => pipeline::Pipeline::Lza,
        _ => return PZ_ERROR_UNSUPPORTED,
    };

    let num_threads = if threads < 0 { 0 } else { threads as usize };
    let options = build_compress_options(ctx, input_len, num_threads);

    match pipeline::compress_with_options(input_slice, pipe, &options) {
        Ok(compressed) => {
            if compressed.len() > output_slice.len() {
                return PZ_ERROR_BUFFER_TOO_SMALL;
            }
            output_slice[..compressed.len()].copy_from_slice(&compressed);
            compressed.len() as i32
        }
        Err(e) => error_to_code(e),
    }
}

/// Decompress data produced by pz_compress.
///
/// Automatically detects the pipeline from the stream header.
/// Returns bytes written on success, or a negative error code on failure.
///
/// # Safety
///
/// - `ctx` must be a valid pointer returned by [`pz_init`].
/// - `input` must point to at least `input_len` readable bytes.
/// - `output` must point to at least `output_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn pz_decompress(
    ctx: *mut PzContext,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    if ctx.is_null() || input.is_null() || output.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    if input_len == 0 {
        return 0;
    }

    let input_slice = slice::from_raw_parts(input, input_len);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    match pipeline::decompress(input_slice) {
        Ok(decompressed) => {
            if decompressed.len() > output_slice.len() {
                return PZ_ERROR_BUFFER_TOO_SMALL;
            }
            output_slice[..decompressed.len()].copy_from_slice(&decompressed);
            decompressed.len() as i32
        }
        Err(e) => error_to_code(e),
    }
}

/// Get the maximum possible compressed size for a given input length.
///
/// This is a conservative upper bound; actual compressed output is
/// typically much smaller.
#[no_mangle]
pub extern "C" fn pz_compress_bound(input_len: usize) -> usize {
    // Worst case: every byte is a literal, each producing a Match struct
    // (SERIALIZED_SIZE bytes per input byte)
    input_len * lz77::Match::SERIALIZED_SIZE
}

// --- LZ77-specific FFI functions ---

/// Compress data using LZ77 only.
///
/// Returns bytes written on success, or a negative error code.
///
/// # Safety
///
/// - `input` must point to at least `input_len` readable bytes.
/// - `output` must point to at least `output_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn pz_lz77_compress(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    if input.is_null() || output.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }
    if input_len == 0 {
        return 0;
    }

    let input_slice = slice::from_raw_parts(input, input_len);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    match lz77::compress_to_buf(input_slice, output_slice) {
        Ok(bytes_written) => bytes_written as i32,
        Err(e) => error_to_code(e),
    }
}

/// Decompress LZ77-compressed data.
///
/// Returns bytes written on success, or a negative error code.
///
/// # Safety
///
/// - `input` must point to at least `input_len` readable bytes.
/// - `output` must point to at least `output_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn pz_lz77_decompress(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    if input.is_null() || output.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }
    if input_len == 0 {
        return 0;
    }

    let input_slice = slice::from_raw_parts(input, input_len);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    match lz77::decompress_to_buf(input_slice, output_slice) {
        Ok(bytes_written) => bytes_written as i32,
        Err(e) => error_to_code(e),
    }
}

// --- Huffman-specific FFI functions ---

/// Create a Huffman tree from input data.
///
/// Returns an opaque handle to the tree, or null on failure.
///
/// # Safety
///
/// - `input` must point to at least `input_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn pz_huffman_new(input: *const u8, input_len: usize) -> *mut HuffmanTree {
    if input.is_null() || input_len == 0 {
        return std::ptr::null_mut();
    }

    let input_slice = slice::from_raw_parts(input, input_len);
    match HuffmanTree::from_data(input_slice) {
        Some(tree) => Box::into_raw(Box::new(tree)),
        None => std::ptr::null_mut(),
    }
}

/// Free a Huffman tree.
///
/// # Safety
///
/// `tree` must be a pointer returned by [`pz_huffman_new`], or null.
/// After this call the pointer is invalid and must not be reused.
#[no_mangle]
pub unsafe extern "C" fn pz_huffman_free(tree: *mut HuffmanTree) {
    if !tree.is_null() {
        let _ = Box::from_raw(tree);
    }
}

/// Encode data using a Huffman tree.
///
/// Returns the number of bits written, or a negative error code.
/// The caller must provide `bits_out` to receive the total bit count.
///
/// # Safety
///
/// - `tree` must be a valid pointer returned by [`pz_huffman_new`].
/// - `input` must point to at least `input_len` readable bytes.
/// - `output` must point to at least `output_len` writable bytes.
/// - `bits_out` must point to a writable `usize`.
#[no_mangle]
pub unsafe extern "C" fn pz_huffman_encode(
    tree: *const HuffmanTree,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
    bits_out: *mut usize,
) -> i32 {
    if tree.is_null() || input.is_null() || output.is_null() || bits_out.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    let tree = &*tree;
    let input_slice = slice::from_raw_parts(input, input_len);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    // Zero the output buffer first
    output_slice.fill(0);

    match tree.encode_to_buf(input_slice, output_slice) {
        Ok(bits) => {
            *bits_out = bits;
            PZ_OK
        }
        Err(e) => error_to_code(e),
    }
}

/// Decode Huffman-encoded data.
///
/// Returns the number of bytes written, or a negative error code.
///
/// # Safety
///
/// - `tree` must be a valid pointer returned by [`pz_huffman_new`].
/// - `input` must point to at least `ceil(total_bits / 8)` readable bytes.
/// - `output` must point to at least `output_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn pz_huffman_decode(
    tree: *const HuffmanTree,
    input: *const u8,
    total_bits: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    if tree.is_null() || input.is_null() || output.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    let tree = &*tree;
    let input_bytes = total_bits.div_ceil(8);
    let input_slice = slice::from_raw_parts(input, input_bytes);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    match tree.decode_to_buf(input_slice, total_bits, output_slice) {
        Ok(bytes_written) => bytes_written as i32,
        Err(e) => error_to_code(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_destroy() {
        unsafe {
            let ctx = pz_init();
            assert!(!ctx.is_null());
            pz_destroy(ctx);
        }
    }

    #[test]
    fn test_query_devices() {
        unsafe {
            let ctx = pz_init();
            let mut info = PzDeviceInfo {
                opencl_devices: -1,
                vulkan_devices: -1,
                cpu_threads: -1,
            };
            let result = pz_query_devices(ctx, &mut info);
            assert_eq!(result, PZ_OK);
            assert!(info.opencl_devices >= 0);
            assert_eq!(info.vulkan_devices, 0);
            assert!(info.cpu_threads >= 1);
            pz_destroy(ctx);
        }
    }

    #[test]
    fn test_compress_decompress_ffi() {
        unsafe {
            let ctx = pz_init();
            // Use longer input to overcome Deflate pipeline overhead (~1KB freq table)
            let pattern = b"hello, world! this is a test of compression. ";
            let mut input = Vec::new();
            for _ in 0..50 {
                input.extend_from_slice(pattern);
            }
            let mut compressed = vec![0u8; input.len() * 2 + 2048];
            let mut decompressed = vec![0u8; input.len() + 1024];

            let comp_size = pz_compress(
                ctx,
                input.as_ptr(),
                input.len(),
                compressed.as_mut_ptr(),
                compressed.len(),
                PzLevel::Default as i32,
                PzPipeline::Deflate as i32,
            );
            assert!(comp_size > 0, "compression failed: {}", comp_size);

            let decomp_size = pz_decompress(
                ctx,
                compressed.as_ptr(),
                comp_size as usize,
                decompressed.as_mut_ptr(),
                decompressed.len(),
            );
            assert!(decomp_size > 0, "decompression failed: {}", decomp_size);
            assert_eq!(&decompressed[..decomp_size as usize], &input[..]);

            pz_destroy(ctx);
        }
    }

    #[test]
    fn test_compress_decompress_all_pipelines_ffi() {
        unsafe {
            let ctx = pz_init();
            let pattern = b"The quick brown fox jumps over the lazy dog. ";
            let mut input = Vec::new();
            for _ in 0..50 {
                input.extend_from_slice(pattern);
            }

            for pipeline_id in 0..3i32 {
                let mut compressed = vec![0u8; input.len() * 2 + 2048];
                let mut decompressed = vec![0u8; input.len() + 1024];

                let comp_size = pz_compress(
                    ctx,
                    input.as_ptr(),
                    input.len(),
                    compressed.as_mut_ptr(),
                    compressed.len(),
                    PzLevel::Default as i32,
                    pipeline_id,
                );
                assert!(
                    comp_size > 0,
                    "compression failed for pipeline {}: {}",
                    pipeline_id,
                    comp_size
                );

                let decomp_size = pz_decompress(
                    ctx,
                    compressed.as_ptr(),
                    comp_size as usize,
                    decompressed.as_mut_ptr(),
                    decompressed.len(),
                );
                assert!(
                    decomp_size > 0,
                    "decompression failed for pipeline {}: {}",
                    pipeline_id,
                    decomp_size
                );
                assert_eq!(&decompressed[..decomp_size as usize], &input[..]);
            }

            pz_destroy(ctx);
        }
    }

    #[test]
    fn test_null_safety() {
        unsafe {
            assert_eq!(
                pz_compress(
                    std::ptr::null_mut(),
                    b"x".as_ptr(),
                    1,
                    [0u8; 100].as_mut_ptr(),
                    100,
                    1,
                    0
                ),
                PZ_ERROR_INVALID_INPUT
            );

            let ctx = pz_init();
            assert_eq!(
                pz_compress(ctx, std::ptr::null(), 1, [0u8; 100].as_mut_ptr(), 100, 1, 0),
                PZ_ERROR_INVALID_INPUT
            );
            pz_destroy(ctx);
        }
    }

    #[test]
    fn test_lz77_ffi_round_trip() {
        unsafe {
            let input = b"abcabcabc";
            let mut compressed = vec![0u8; pz_compress_bound(input.len())];
            let mut decompressed = vec![0u8; input.len() + 100];

            let comp_size = pz_lz77_compress(
                input.as_ptr(),
                input.len(),
                compressed.as_mut_ptr(),
                compressed.len(),
            );
            assert!(comp_size > 0);

            let decomp_size = pz_lz77_decompress(
                compressed.as_ptr(),
                comp_size as usize,
                decompressed.as_mut_ptr(),
                decompressed.len(),
            );
            assert!(decomp_size > 0);
            assert_eq!(&decompressed[..decomp_size as usize], input);
        }
    }

    #[test]
    fn test_huffman_ffi_round_trip() {
        unsafe {
            let input = b"hello, world!";
            let tree = pz_huffman_new(input.as_ptr(), input.len());
            assert!(!tree.is_null());

            let mut encoded = vec![0u8; input.len() * 2];
            let mut bits_out: usize = 0;
            let result = pz_huffman_encode(
                tree,
                input.as_ptr(),
                input.len(),
                encoded.as_mut_ptr(),
                encoded.len(),
                &mut bits_out,
            );
            assert_eq!(result, PZ_OK);
            assert!(bits_out > 0);

            let mut decoded = vec![0u8; input.len() + 100];
            let decoded_size = pz_huffman_decode(
                tree,
                encoded.as_ptr(),
                bits_out,
                decoded.as_mut_ptr(),
                decoded.len(),
            );
            assert!(decoded_size > 0);
            assert_eq!(&decoded[..decoded_size as usize], input);

            pz_huffman_free(tree);
        }
    }

    #[test]
    fn test_compress_bound() {
        assert_eq!(pz_compress_bound(0), 0);
        assert_eq!(pz_compress_bound(1), lz77::Match::SERIALIZED_SIZE);
        assert_eq!(pz_compress_bound(100), 100 * lz77::Match::SERIALIZED_SIZE);
    }
}
