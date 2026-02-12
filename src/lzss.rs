/// LZSS (Lempel-Ziv-Storer-Szymanski) compression and decompression.
///
/// A variation of LZ77 that uses flag bits to distinguish literals from
/// match references, eliminating the overhead of always emitting a 5-byte
/// (offset, length, next) triple. Each token is either:
/// - A literal byte (1 byte, flag bit = 1)
/// - A match reference (4 bytes: offset u16 + length u16, flag bit = 0)
///
/// Uses the same hash-chain match finder and lazy matching strategy as LZ77.
///
/// Wire format:
/// ```text
/// [original_len: u32 LE]   [num_tokens: u32 LE]   [flag_bytes_len: u32 LE]
/// [flag_bits: packed bytes, MSB-first]
/// [token_data: literals (1B) and matches (4B) concatenated]
/// ```
use crate::lz77::{HashChainFinder, MIN_MATCH};
use crate::{PzError, PzResult};

/// Header size: original_len (4) + num_tokens (4) + flag_bytes_len (4).
const HEADER_SIZE: usize = 12;

/// Match length threshold above which lazy evaluation is skipped.
const LAZY_SKIP_THRESHOLD: u16 = 32;

/// Maximum hash insertion count per match.
const MAX_INSERT_LEN: usize = 128;

/// Pack boolean flags into bytes, MSB-first.
fn pack_flags(flags: &[bool]) -> Vec<u8> {
    let num_bytes = flags.len().div_ceil(8);
    let mut bytes = vec![0u8; num_bytes];
    for (i, &flag) in flags.iter().enumerate() {
        if flag {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    bytes
}

/// Unpack boolean flags from bytes, MSB-first.
fn unpack_flags(bytes: &[u8], count: usize) -> Vec<bool> {
    (0..count)
        .map(|i| bytes[i / 8] & (1 << (7 - (i % 8))) != 0)
        .collect()
}

/// Compress input using LZSS with lazy matching.
///
/// Reuses the LZ77 hash-chain match finder but encodes output more
/// efficiently: literals cost 1 byte instead of 5.
pub fn encode(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut finder = HashChainFinder::new();
    let mut flags: Vec<bool> = Vec::new();
    let mut token_data: Vec<u8> = Vec::new();
    let mut pos: usize = 0;

    while pos < input.len() {
        let m = finder.find_match(input, pos);
        finder.insert(input, pos);

        // Lazy matching: check if next position has a longer match
        if m.length >= MIN_MATCH && m.length < LAZY_SKIP_THRESHOLD && pos + 1 < input.len() {
            finder.insert(input, pos + 1);
            let next_m = finder.find_match(input, pos + 1);

            if next_m.length > m.length {
                // Emit literal for current position, use the better match
                flags.push(true);
                token_data.push(input[pos]);
                pos += 1;

                // Emit the match from pos (which was pos+1 before increment)
                flags.push(false);
                token_data.extend_from_slice(&next_m.offset.to_le_bytes());
                token_data.extend_from_slice(&next_m.length.to_le_bytes());

                // Insert covered positions into hash chains
                let advance = next_m.length as usize;
                let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
                for i in 1..insert_count {
                    finder.insert(input, pos + i);
                }
                pos += advance;
                continue;
            }
        }

        if m.length >= MIN_MATCH {
            // Emit match
            flags.push(false);
            token_data.extend_from_slice(&m.offset.to_le_bytes());
            token_data.extend_from_slice(&m.length.to_le_bytes());

            let advance = m.length as usize;
            let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
            for i in 1..insert_count {
                finder.insert(input, pos + i);
            }
            pos += advance;
        } else {
            // Emit literal
            flags.push(true);
            token_data.push(input[pos]);
            pos += 1;
        }
    }

    // Serialize: header + packed flags + token data
    let flag_bytes = pack_flags(&flags);
    let total_size = HEADER_SIZE + flag_bytes.len() + token_data.len();
    let mut output = Vec::with_capacity(total_size);
    output.extend_from_slice(&(input.len() as u32).to_le_bytes());
    output.extend_from_slice(&(flags.len() as u32).to_le_bytes());
    output.extend_from_slice(&(flag_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(&flag_bytes);
    output.extend_from_slice(&token_data);
    Ok(output)
}

/// Decompress LZSS-compressed data.
pub fn decode(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if input.len() < HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    let original_len = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    let num_tokens = u32::from_le_bytes(input[4..8].try_into().unwrap()) as usize;
    let flag_bytes_len = u32::from_le_bytes(input[8..12].try_into().unwrap()) as usize;

    if HEADER_SIZE + flag_bytes_len > input.len() {
        return Err(PzError::InvalidInput);
    }

    let flag_bytes = &input[HEADER_SIZE..HEADER_SIZE + flag_bytes_len];
    let flags = unpack_flags(flag_bytes, num_tokens);
    let token_data = &input[HEADER_SIZE + flag_bytes_len..];

    let mut output = Vec::with_capacity(original_len);
    let mut td_pos: usize = 0;

    for &is_literal in &flags {
        if is_literal {
            if td_pos >= token_data.len() {
                return Err(PzError::InvalidInput);
            }
            output.push(token_data[td_pos]);
            td_pos += 1;
        } else {
            if td_pos + 4 > token_data.len() {
                return Err(PzError::InvalidInput);
            }
            let offset = u16::from_le_bytes([token_data[td_pos], token_data[td_pos + 1]]) as usize;
            let length =
                u16::from_le_bytes([token_data[td_pos + 2], token_data[td_pos + 3]]) as usize;
            td_pos += 4;

            if offset == 0 || offset > output.len() {
                return Err(PzError::InvalidInput);
            }

            let copy_start = output.len() - offset;
            for j in 0..length {
                let byte = output[copy_start + j];
                output.push(byte);
            }
        }
    }

    // Sanity check: decoded length must match original
    if output.len() != original_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

/// Compress into a caller-allocated buffer. Returns bytes written.
pub fn encode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    let encoded = encode(input)?;
    if encoded.len() > output.len() {
        return Err(PzError::BufferTooSmall);
    }
    output[..encoded.len()].copy_from_slice(&encoded);
    Ok(encoded.len())
}

/// Decompress into a caller-allocated buffer. Returns bytes written.
pub fn decode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    if input.len() < HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    let original_len = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    if original_len > output.len() {
        return Err(PzError::BufferTooSmall);
    }

    let decoded = decode(input)?;
    output[..decoded.len()].copy_from_slice(&decoded);
    Ok(decoded.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag_packing() {
        let flags = vec![true, false, true, true, false, false, true, false, true];
        let packed = pack_flags(&flags);
        let unpacked = unpack_flags(&packed, flags.len());
        assert_eq!(unpacked, flags);
    }

    #[test]
    fn test_flag_packing_empty() {
        let flags: Vec<bool> = Vec::new();
        let packed = pack_flags(&flags);
        assert!(packed.is_empty());
        let unpacked = unpack_flags(&packed, 0);
        assert!(unpacked.is_empty());
    }

    #[test]
    fn test_flag_packing_all_true() {
        let flags = vec![true; 16];
        let packed = pack_flags(&flags);
        assert_eq!(packed, vec![0xFF, 0xFF]);
        let unpacked = unpack_flags(&packed, 16);
        assert_eq!(unpacked, flags);
    }

    #[test]
    fn test_round_trip_empty() {
        let input: Vec<u8> = Vec::new();
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_single_byte() {
        let input = vec![42u8];
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_no_matches() {
        // Short unique data — all literals, no matches
        let input = b"abcdefgh".to_vec();
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_all_same() {
        let input = vec![b'x'; 200];
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_repeats() {
        let pattern = b"the quick brown fox ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_longer_text() {
        let input = b"To be, or not to be, that is the question: \
            Whether 'tis nobler in the mind to suffer \
            The slings and arrows of outrageous fortune, \
            Or to take arms against a sea of troubles"
            .to_vec();
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_large() {
        let pattern = b"abcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        assert!(input.len() > 7500);
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_binary() {
        let input: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_window_boundary() {
        // Data larger than 32KB window to exercise boundary handling
        let mut input = Vec::new();
        let pattern = b"window boundary test pattern! ";
        for _ in 0..2000 {
            input.extend_from_slice(pattern);
        }
        assert!(input.len() > 32768);
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_decode_invalid_short() {
        // Too short to contain a valid header
        let result = decode(&[0, 1, 2]);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_invalid_truncated() {
        // Valid header but truncated flag data
        let mut data = vec![0u8; HEADER_SIZE];
        // original_len = 10, num_tokens = 5, flag_bytes_len = 100 (too large)
        data[0..4].copy_from_slice(&10u32.to_le_bytes());
        data[4..8].copy_from_slice(&5u32.to_le_bytes());
        data[8..12].copy_from_slice(&100u32.to_le_bytes());
        let result = decode(&data);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_encode_to_buf() {
        let input = b"hello hello hello".to_vec();
        let encoded = encode(&input).unwrap();
        let mut buf = vec![0u8; encoded.len() + 10];
        let written = encode_to_buf(&input, &mut buf).unwrap();
        assert_eq!(written, encoded.len());
        assert_eq!(&buf[..written], &encoded[..]);
    }

    #[test]
    fn test_encode_to_buf_too_small() {
        let input = vec![b'a'; 100];
        let mut buf = vec![0u8; 1]; // way too small
        let result = encode_to_buf(&input, &mut buf);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_decode_to_buf() {
        let input = b"test data for decode_to_buf".to_vec();
        let encoded = encode(&input).unwrap();
        let mut buf = vec![0u8; input.len() + 10];
        let written = decode_to_buf(&encoded, &mut buf).unwrap();
        assert_eq!(written, input.len());
        assert_eq!(&buf[..written], &input[..]);
    }

    #[test]
    fn test_decode_to_buf_too_small() {
        let input = vec![b'z'; 100];
        let encoded = encode(&input).unwrap();
        let mut buf = vec![0u8; 1]; // too small
        let result = decode_to_buf(&encoded, &mut buf);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_two_bytes() {
        let input = vec![0u8, 255];
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compresses_repetitive_data() {
        // LZSS should compress repetitive text data
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        let compressed = encode(&input).unwrap();
        assert!(
            compressed.len() < input.len(),
            "LZSS should compress repeated text: {} >= {}",
            compressed.len(),
            input.len()
        );
    }
}
