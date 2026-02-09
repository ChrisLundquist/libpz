//! C-callable FFI layer for libpz.
//!
//! Provides a stable C API that hides all Rust internals behind
//! opaque handles and simple C types.
//!
//! # Safety
//!
//! All `unsafe extern "C"` functions in this module require:
//! - Non-null pointers where documented
//! - Valid pointer/length pairs (caller must ensure the pointed-to memory
//!   is valid for the specified length)
//! - Proper alignment for the pointed-to types
#![allow(clippy::missing_safety_doc)]

use std::slice;

use crate::huffman::HuffmanTree;
use crate::lz77;

// Error codes matching the C API design
const PZ_OK: i32 = 0;
const PZ_ERROR_BUFFER_TOO_SMALL: i32 = -1;
const PZ_ERROR_INVALID_INPUT: i32 = -2;
const PZ_ERROR_UNSUPPORTED: i32 = -3;

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
    /// Placeholder for future state (GPU devices, cached kernels, etc.)
    _initialized: bool,
}

/// Initialize a libpz context.
///
/// Returns a pointer to a new context, or null on failure.
#[no_mangle]
pub extern "C" fn pz_init() -> *mut PzContext {
    let ctx = Box::new(PzContext {
        _initialized: true,
    });
    Box::into_raw(ctx)
}

/// Destroy a libpz context and free associated resources.
#[no_mangle]
pub unsafe extern "C" fn pz_destroy(ctx: *mut PzContext) {
    if !ctx.is_null() {
        let _ = Box::from_raw(ctx);
    }
}

/// Query available compute devices.
#[no_mangle]
pub unsafe extern "C" fn pz_query_devices(
    ctx: *const PzContext,
    info: *mut PzDeviceInfo,
) -> i32 {
    if ctx.is_null() || info.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    // For Phase 1, only CPU is available
    let info = &mut *info;
    info.opencl_devices = 0;
    info.vulkan_devices = 0;
    info.cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(1);

    PZ_OK
}

/// Compress data using the specified pipeline.
///
/// Returns bytes written on success, or a negative error code on failure.
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

    // For Phase 1, only DEFLATE-like pipeline (LZ77) is implemented
    match pipeline {
        0 => {
            // PZ_DEFLATE: LZ77 + Huffman
            // For now, just LZ77 (Huffman integration in future)
            match lz77::compress_to_buf(input_slice, output_slice) {
                Ok(bytes_written) => bytes_written as i32,
                Err(crate::PzError::BufferTooSmall) => PZ_ERROR_BUFFER_TOO_SMALL,
                Err(crate::PzError::InvalidInput) => PZ_ERROR_INVALID_INPUT,
                Err(crate::PzError::Unsupported) => PZ_ERROR_UNSUPPORTED,
            }
        }
        _ => PZ_ERROR_UNSUPPORTED,
    }
}

/// Decompress data.
///
/// Returns bytes written on success, or a negative error code on failure.
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

    match lz77::decompress_to_buf(input_slice, output_slice) {
        Ok(bytes_written) => bytes_written as i32,
        Err(crate::PzError::BufferTooSmall) => PZ_ERROR_BUFFER_TOO_SMALL,
        Err(crate::PzError::InvalidInput) => PZ_ERROR_INVALID_INPUT,
        Err(crate::PzError::Unsupported) => PZ_ERROR_UNSUPPORTED,
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
        Err(crate::PzError::BufferTooSmall) => PZ_ERROR_BUFFER_TOO_SMALL,
        Err(crate::PzError::InvalidInput) => PZ_ERROR_INVALID_INPUT,
        Err(crate::PzError::Unsupported) => PZ_ERROR_UNSUPPORTED,
    }
}

/// Decompress LZ77-compressed data.
///
/// Returns bytes written on success, or a negative error code.
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
        Err(crate::PzError::BufferTooSmall) => PZ_ERROR_BUFFER_TOO_SMALL,
        Err(crate::PzError::InvalidInput) => PZ_ERROR_INVALID_INPUT,
        Err(crate::PzError::Unsupported) => PZ_ERROR_UNSUPPORTED,
    }
}

// --- Huffman-specific FFI functions ---

/// Create a Huffman tree from input data.
///
/// Returns an opaque handle to the tree, or null on failure.
#[no_mangle]
pub unsafe extern "C" fn pz_huffman_new(
    input: *const u8,
    input_len: usize,
) -> *mut HuffmanTree {
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
    for byte in output_slice.iter_mut() {
        *byte = 0;
    }

    match tree.encode_to_buf(input_slice, output_slice) {
        Ok(bits) => {
            *bits_out = bits;
            PZ_OK
        }
        Err(crate::PzError::BufferTooSmall) => PZ_ERROR_BUFFER_TOO_SMALL,
        Err(crate::PzError::InvalidInput) => PZ_ERROR_INVALID_INPUT,
        Err(crate::PzError::Unsupported) => PZ_ERROR_UNSUPPORTED,
    }
}

/// Decode Huffman-encoded data.
///
/// Returns the number of bytes written, or a negative error code.
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
        Err(crate::PzError::BufferTooSmall) => PZ_ERROR_BUFFER_TOO_SMALL,
        Err(crate::PzError::InvalidInput) => PZ_ERROR_INVALID_INPUT,
        Err(crate::PzError::Unsupported) => PZ_ERROR_UNSUPPORTED,
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
            assert_eq!(info.opencl_devices, 0);
            assert_eq!(info.vulkan_devices, 0);
            assert!(info.cpu_threads >= 1);
            pz_destroy(ctx);
        }
    }

    #[test]
    fn test_compress_decompress_ffi() {
        unsafe {
            let ctx = pz_init();
            let input = b"hello, world! hello, world!";
            let mut compressed = vec![0u8; pz_compress_bound(input.len())];
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
            assert_eq!(&decompressed[..decomp_size as usize], input);

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

            pz_huffman_free(tree as *mut HuffmanTree);
        }
    }

    #[test]
    fn test_compress_bound() {
        assert_eq!(pz_compress_bound(0), 0);
        assert_eq!(pz_compress_bound(1), lz77::Match::SERIALIZED_SIZE);
        assert_eq!(pz_compress_bound(100), 100 * lz77::Match::SERIALIZED_SIZE);
    }
}
