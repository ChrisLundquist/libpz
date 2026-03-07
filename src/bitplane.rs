/// Bit-Plane Decomposition pipeline (`bitplane`).
///
/// Splits each byte into 8 bit-streams (one per bit position), then
/// run-length encodes each binary stream independently and FSE-encodes
/// the RLE output. This pipeline has zero serial stages, zero data-dependent
/// branching, and zero cross-position dependencies — making it useful as
/// a GPU throughput ceiling benchmark.
///
/// **Pipeline:**
/// ```text
/// Input bytes
///   → Bit transpose: split each byte into 8 bit-streams
///   → RLE each binary stream independently
///   → FSE encode each RLE stream
/// Output: 8 compressed bit-streams + header
/// ```
use crate::fse;
use crate::{PzError, PzResult};

/// Transpose N bytes into 8 bit-plane streams.
///
/// For each bit position b (0-7), stream b contains the b-th bit of every
/// input byte, packed into bytes (MSB-first within each output byte).
fn bit_transpose(input: &[u8]) -> [Vec<u8>; 8] {
    let n = input.len();
    let plane_bytes = n.div_ceil(8);
    let mut planes: [Vec<u8>; 8] = std::array::from_fn(|_| vec![0u8; plane_bytes]);

    for (i, &byte) in input.iter().enumerate() {
        let out_byte_idx = i / 8;
        let out_bit_pos = 7 - (i % 8); // MSB-first packing
        for bit in 0..8u8 {
            if byte & (1 << (7 - bit)) != 0 {
                planes[bit as usize][out_byte_idx] |= 1 << out_bit_pos;
            }
        }
    }

    planes
}

/// Inverse bit transpose: reconstruct N bytes from 8 bit-plane streams.
fn bit_untranspose(planes: &[Vec<u8>; 8], n: usize) -> Vec<u8> {
    let mut output = vec![0u8; n];

    for (i, out_byte) in output.iter_mut().enumerate() {
        let in_byte_idx = i / 8;
        let in_bit_pos = 7 - (i % 8);
        for bit in 0..8u8 {
            if in_byte_idx < planes[bit as usize].len()
                && planes[bit as usize][in_byte_idx] & (1 << in_bit_pos) != 0
            {
                *out_byte |= 1 << (7 - bit);
            }
        }
    }

    output
}

/// Public wrapper for GPU path: RLE encode a single bit-plane.
#[cfg(feature = "webgpu")]
pub fn rle_binary_for_gpu(data: &[u8], num_bits: usize) -> Vec<u8> {
    rle_binary(data, num_bits)
}

/// Run-length encode a binary bit-stream (packed bytes).
///
/// Encodes runs of identical bits. Output format:
/// Sequence of (run_length: u16 LE) values. First run is always for bit 0.
/// If the stream starts with bit 1, a zero-length run for bit 0 is emitted first.
fn rle_binary(data: &[u8], num_bits: usize) -> Vec<u8> {
    if num_bits == 0 {
        return vec![0, 0]; // single zero-length run
    }

    let mut runs: Vec<u16> = Vec::new();

    // Read individual bits
    let get_bit = |i: usize| -> u8 {
        if i / 8 >= data.len() {
            return 0;
        }
        (data[i / 8] >> (7 - (i % 8))) & 1
    };

    let first_bit = get_bit(0);
    // If first bit is 1, emit a zero-length run for bit 0
    if first_bit == 1 {
        runs.push(0);
    }

    let mut current_bit = first_bit;
    let mut run_len: u16 = 1;

    for i in 1..num_bits {
        let bit = get_bit(i);
        if bit == current_bit && run_len < u16::MAX {
            run_len += 1;
        } else if bit != current_bit {
            // Bit value changed — emit current run and start new one.
            runs.push(run_len);
            current_bit = bit;
            run_len = 1;
        } else {
            // Same bit but run_len hit u16::MAX — split the run.
            // Emit the full run, then a zero-length run for the opposite bit
            // to keep the alternating protocol in sync.
            runs.push(run_len);
            runs.push(0); // zero-length run for opposite bit
            run_len = 1;
        }
    }
    runs.push(run_len);

    // Serialize runs as u16 LE
    let mut output = Vec::with_capacity(runs.len() * 2);
    for r in &runs {
        output.extend_from_slice(&r.to_le_bytes());
    }
    output
}

/// Decode RLE binary stream back to packed bits.
fn rle_binary_decode(rle_data: &[u8], num_bits: usize) -> PzResult<Vec<u8>> {
    let plane_bytes = num_bits.div_ceil(8);
    let mut output = vec![0u8; plane_bytes];

    if !rle_data.len().is_multiple_of(2) {
        return Err(PzError::InvalidInput);
    }

    let mut bit_pos = 0;
    let mut current_value: u8 = 0; // starts with bit 0

    let set_bit = |output: &mut [u8], pos: usize| {
        if pos / 8 < output.len() {
            output[pos / 8] |= 1 << (7 - (pos % 8));
        }
    };

    let mut i = 0;
    while i + 1 < rle_data.len() && bit_pos < num_bits {
        let run_len = u16::from_le_bytes([rle_data[i], rle_data[i + 1]]) as usize;
        i += 2;

        if current_value == 1 {
            for j in 0..run_len {
                if bit_pos + j < num_bits {
                    set_bit(&mut output, bit_pos + j);
                }
            }
        }
        bit_pos += run_len;
        current_value ^= 1;
    }

    Ok(output)
}

/// Compress input using bit-plane decomposition.
///
/// Wire format:
/// ```text
/// [orig_len: u32 LE]
/// For each of 8 planes:
///   [rle_raw_len: u32 LE] [fse_compressed_len: u32 LE] [fse_data: ...]
/// ```
pub fn compress(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let n = input.len();
    let num_bits = n; // one bit per input byte per plane

    // Step 1: Bit transpose
    let planes = bit_transpose(input);

    // Step 2 & 3: RLE + FSE each plane
    let mut output = Vec::new();
    output.extend_from_slice(&(n as u32).to_le_bytes());

    for plane in &planes {
        let rle_data = rle_binary(plane, num_bits);
        let rle_raw_len = rle_data.len();
        let fse_data = fse::encode(&rle_data);

        output.extend_from_slice(&(rle_raw_len as u32).to_le_bytes());
        output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
        output.extend_from_slice(&fse_data);
    }

    Ok(output)
}

/// Decompress bit-plane data back to the original input.
pub fn decompress(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let stored_len = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if stored_len != orig_len {
        return Err(PzError::InvalidInput);
    }

    let num_bits = orig_len;
    let mut pos = 4;
    let mut planes: [Vec<u8>; 8] = std::array::from_fn(|_| Vec::new());

    for plane in &mut planes {
        if pos + 8 > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let rle_raw_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;
        let fse_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        if pos + fse_len > payload.len() {
            return Err(PzError::InvalidInput);
        }

        let rle_data = fse::decode(&payload[pos..pos + fse_len], rle_raw_len)?;
        pos += fse_len;

        *plane = rle_binary_decode(&rle_data, num_bits)?;
    }

    let output = bit_untranspose(&planes, orig_len);
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let input = b"Hello, World! This is a test of the bit-plane decomposition pipeline.";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_all_zeros() {
        let input = vec![0u8; 256];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_all_ones() {
        let input = vec![0xFFu8; 256];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_binary_data() {
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_small() {
        let input = b"ab";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_repetitive_text() {
        // Highly repetitive data — triggers edge cases in RLE binary coding
        let input: Vec<u8> = b"The quick brown fox. ".repeat(500);
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_all_same_large() {
        let input = vec![0x41u8; 10000]; // 10KB of 'A'
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn bit_transpose_roundtrip() {
        let input: Vec<u8> = (0..=255).collect();
        let planes = bit_transpose(&input);
        let recovered = bit_untranspose(&planes, input.len());
        assert_eq!(recovered, input);
    }
}
