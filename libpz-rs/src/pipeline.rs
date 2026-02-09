/// Compression pipeline orchestrator.
///
/// Chains algorithm stages together to form complete compression pipelines.
/// Each pipeline defines a sequence of transforms and entropy coding
/// stages that are applied in order for compression, and in reverse for
/// decompression.
///
/// **Supported pipelines:**
///
/// | Pipeline      | Stages                           | Similar to |
/// |---------------|----------------------------------|------------|
/// | `Deflate`     | LZ77 → Huffman                   | gzip       |
/// | `Bw`          | BWT → MTF → RLE → Range coder    | bzip2      |
/// | `Lza`         | LZ77 → Range coder               | lzma-like  |
///
/// **Container format:**
/// Each compressed stream starts with a header:
/// - Magic bytes: `PZ` (2 bytes)
/// - Version: 1 (1 byte)
/// - Pipeline ID: 0=Deflate, 1=Bw, 2=Lza (1 byte)
/// - Original length: u32 little-endian (4 bytes)
/// - Pipeline-specific metadata (variable)
/// - Compressed data

use crate::bwt;
use crate::huffman::HuffmanTree;
use crate::lz77;
use crate::mtf;
use crate::rangecoder;
use crate::rle;
use crate::{PzError, PzResult};

/// Magic bytes for the libpz container format.
const MAGIC: [u8; 2] = [b'P', b'Z'];
/// Format version.
const VERSION: u8 = 1;

/// Minimum header size: magic(2) + version(1) + pipeline(1) + orig_len(4) = 8
const MIN_HEADER_SIZE: usize = 8;

/// Compression pipeline types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Pipeline {
    /// LZ77 + Huffman (gzip-like)
    Deflate = 0,
    /// BWT + MTF + RLE + Range coder (bzip2-like)
    Bw = 1,
    /// LZ77 + Range coder (lzma-like)
    Lza = 2,
}

impl Pipeline {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Pipeline::Deflate),
            1 => Some(Pipeline::Bw),
            2 => Some(Pipeline::Lza),
            _ => None,
        }
    }
}

/// Compress data using the specified pipeline.
///
/// Returns a self-contained compressed stream including the header.
pub fn compress(input: &[u8], pipeline: Pipeline) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    match pipeline {
        Pipeline::Deflate => compress_deflate(input),
        Pipeline::Bw => compress_bw(input),
        Pipeline::Lza => compress_lza(input),
    }
}

/// Decompress data produced by `compress`.
///
/// Reads the header to determine the pipeline, then applies the
/// inverse stages.
pub fn decompress(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if input.len() < MIN_HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    // Parse header
    if input[0] != MAGIC[0] || input[1] != MAGIC[1] {
        return Err(PzError::InvalidInput);
    }
    if input[2] != VERSION {
        return Err(PzError::Unsupported);
    }

    let pipeline = Pipeline::from_u8(input[3]).ok_or(PzError::Unsupported)?;
    let orig_len = u32::from_le_bytes([input[4], input[5], input[6], input[7]]) as usize;

    if orig_len == 0 {
        return Ok(Vec::new());
    }

    let payload = &input[MIN_HEADER_SIZE..];

    match pipeline {
        Pipeline::Deflate => decompress_deflate(payload, orig_len),
        Pipeline::Bw => decompress_bw(payload, orig_len),
        Pipeline::Lza => decompress_lza(payload, orig_len),
    }
}

/// Write the standard header to output.
fn write_header(output: &mut Vec<u8>, pipeline: Pipeline, orig_len: usize) {
    output.extend_from_slice(&MAGIC);
    output.push(VERSION);
    output.push(pipeline as u8);
    output.extend_from_slice(&(orig_len as u32).to_le_bytes());
}

// --- DEFLATE pipeline: LZ77 + Huffman ---

fn compress_deflate(input: &[u8]) -> PzResult<Vec<u8>> {
    // Stage 1: LZ77 compression (using hash-chain for speed)
    let lz_data = lz77::compress_hashchain(input)?;

    // Stage 2: Huffman encoding of the LZ77 output
    let tree = HuffmanTree::from_data(&lz_data).ok_or(PzError::InvalidInput)?;
    let (huffman_data, total_bits) = tree.encode(&lz_data)?;

    // Serialize: header + freq_table + total_bits + huffman_data
    let freq_table = tree.serialize_frequencies();
    let mut output = Vec::new();
    write_header(&mut output, Pipeline::Deflate, input.len());

    // Write LZ77 output length (needed for Huffman decode allocation)
    output.extend_from_slice(&(lz_data.len() as u32).to_le_bytes());
    // Write total bits
    output.extend_from_slice(&(total_bits as u32).to_le_bytes());
    // Write frequency table (256 * 4 bytes = 1024 bytes)
    for &freq in &freq_table {
        output.extend_from_slice(&freq.to_le_bytes());
    }
    // Write Huffman-encoded data
    output.extend_from_slice(&huffman_data);

    Ok(output)
}

fn decompress_deflate(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    // Parse metadata: lz_len(4) + total_bits(4) + freq_table(1024) = 1032 bytes
    if payload.len() < 1032 {
        return Err(PzError::InvalidInput);
    }

    let lz_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let total_bits =
        u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;

    // Reconstruct frequency table
    let mut freq_table = crate::frequency::FrequencyTable::new();
    for i in 0..256 {
        let offset = 8 + i * 4;
        freq_table.byte[i] = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]);
    }
    freq_table.total = freq_table.byte.iter().map(|&f| f as u64).sum();
    freq_table.used = freq_table.byte.iter().filter(|&&f| f > 0).count() as u32;

    let huffman_data = &payload[1032..];

    // Stage 1: Huffman decode
    let tree =
        HuffmanTree::from_frequency_table(&freq_table).ok_or(PzError::InvalidInput)?;
    let mut lz_data = vec![0u8; lz_len];
    let decoded_len = tree.decode_to_buf(huffman_data, total_bits, &mut lz_data)?;
    if decoded_len != lz_len {
        return Err(PzError::InvalidInput);
    }

    // Stage 2: LZ77 decompress
    let mut output = vec![0u8; orig_len];
    let out_len = lz77::decompress_to_buf(&lz_data, &mut output)?;
    if out_len != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

// --- BW pipeline: BWT + MTF + RLE + Range coder ---

fn compress_bw(input: &[u8]) -> PzResult<Vec<u8>> {
    // Stage 1: BWT
    let bwt_result = bwt::encode(input).ok_or(PzError::InvalidInput)?;

    // Stage 2: MTF
    let mtf_data = mtf::encode(&bwt_result.data);

    // Stage 3: RLE
    let rle_data = rle::encode(&mtf_data);

    // Stage 4: Range coder
    let rc_data = rangecoder::encode(&rle_data);

    // Serialize: header + primary_index + rle_len + rc_data
    let mut output = Vec::new();
    write_header(&mut output, Pipeline::Bw, input.len());
    output.extend_from_slice(&bwt_result.primary_index.to_le_bytes());
    output.extend_from_slice(&(rle_data.len() as u32).to_le_bytes());
    output.extend_from_slice(&rc_data);

    Ok(output)
}

fn decompress_bw(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 8 {
        return Err(PzError::InvalidInput);
    }

    let primary_index =
        u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let rle_len =
        u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;

    let rc_data = &payload[8..];

    // Stage 1: Range decoder
    let rle_data = rangecoder::decode(rc_data, rle_len)?;

    // Stage 2: RLE decode
    let mtf_data = rle::decode(&rle_data)?;

    // Stage 3: Inverse MTF
    let bwt_data = mtf::decode(&mtf_data);

    // Stage 4: Inverse BWT
    let output = bwt::decode(&bwt_data, primary_index)?;

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

// --- LZA pipeline: LZ77 + Range coder ---

fn compress_lza(input: &[u8]) -> PzResult<Vec<u8>> {
    // Stage 1: LZ77 compression (using hash-chain)
    let lz_data = lz77::compress_hashchain(input)?;

    // Stage 2: Range coder
    let rc_data = rangecoder::encode(&lz_data);

    // Serialize: header + lz_len + rc_data
    let mut output = Vec::new();
    write_header(&mut output, Pipeline::Lza, input.len());
    output.extend_from_slice(&(lz_data.len() as u32).to_le_bytes());
    output.extend_from_slice(&rc_data);

    Ok(output)
}

fn decompress_lza(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let lz_len =
        u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let rc_data = &payload[4..];

    // Stage 1: Range decode
    let lz_data = rangecoder::decode(rc_data, lz_len)?;

    // Stage 2: LZ77 decompress
    let mut output = vec![0u8; orig_len];
    let out_len = lz77::decompress_to_buf(&lz_data, &mut output)?;
    if out_len != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Deflate pipeline tests ---

    #[test]
    fn test_deflate_empty() {
        let result = compress(&[], Pipeline::Deflate).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_deflate_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_deflate_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        // Use enough repetitions to overcome the ~1KB frequency table overhead
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        // Should actually compress (input is ~4.5KB, overhead is ~1KB)
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_deflate_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Deflate).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- BW pipeline tests ---

    #[test]
    fn test_bw_empty() {
        let result = compress(&[], Pipeline::Bw).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_bw_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_bw_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_bw_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_bw_round_trip_all_same() {
        let input = vec![b'x'; 200];
        let compressed = compress(&input, Pipeline::Bw).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- LZA pipeline tests ---

    #[test]
    fn test_lza_empty() {
        let result = compress(&[], Pipeline::Lza).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_lza_round_trip_hello() {
        let input = b"hello, world! hello, world!";
        let compressed = compress(input, Pipeline::Lza).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lza_round_trip_repeating() {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input, Pipeline::Lza).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
        assert!(
            compressed.len() < input.len(),
            "compressed {} >= input {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_lza_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, Pipeline::Lza).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- Header / format tests ---

    #[test]
    fn test_invalid_magic() {
        let result = decompress(&[0, 0, 1, 0, 0, 0, 0, 0]);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_invalid_version() {
        let result = decompress(&[b'P', b'Z', 99, 0, 0, 0, 0, 0]);
        assert_eq!(result, Err(PzError::Unsupported));
    }

    #[test]
    fn test_invalid_pipeline() {
        let result = decompress(&[b'P', b'Z', 1, 99, 0, 0, 0, 0]);
        assert_eq!(result, Err(PzError::Unsupported));
    }

    #[test]
    fn test_too_short_input() {
        let result = decompress(&[b'P', b'Z']);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_zero_original_length() {
        let result = decompress(&[b'P', b'Z', 1, 0, 0, 0, 0, 0]);
        assert_eq!(result.unwrap(), Vec::<u8>::new());
    }

    // --- Cross-pipeline tests ---

    #[test]
    fn test_all_pipelines_banana() {
        let input = b"banana banana banana banana banana";
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
            let compressed = compress(input, pipeline).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(
                decompressed, input,
                "failed for pipeline {:?}",
                pipeline
            );
        }
    }

    #[test]
    fn test_all_pipelines_medium_text() {
        let mut input = Vec::new();
        for _ in 0..10 {
            input.extend(b"abcdefghij klmnopqrstuvwxyz 0123456789 ");
        }
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
            let compressed = compress(&input, pipeline).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(
                decompressed, input,
                "failed for pipeline {:?}",
                pipeline
            );
        }
    }
}
