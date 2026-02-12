/// LZ78 (Lempel-Ziv 1978) compression and decompression.
///
/// Builds an incremental dictionary (trie) during compression. At each step,
/// finds the longest prefix of the remaining input that exists in the
/// dictionary, emits a (dictionary_index, next_byte) token, and adds the
/// extended string to the dictionary.
///
/// Wire format:
/// ```text
/// [original_len: u32 LE]  [max_dict_size: u16 LE]  [num_tokens: u32 LE]
/// [tokens: (index: u16 LE, next: u8) × num_tokens]
/// ```
///
/// Each token is 3 bytes. Dictionary index 0 represents the root (empty string).
/// When the dictionary fills to max_dict_size, no new entries are added (freeze).
use std::collections::HashMap;

use crate::{PzError, PzResult};

/// Default maximum dictionary entries.
const DEFAULT_DICT_SIZE: u16 = 16384;

/// Header size: original_len (4) + max_dict_size (2) + num_tokens (4).
const HEADER_SIZE: usize = 10;

/// Bytes per serialized token: index (2) + next (1).
const TOKEN_SIZE: usize = 3;

/// Compress input using LZ78 with the default dictionary size (16384).
pub fn encode(input: &[u8]) -> PzResult<Vec<u8>> {
    encode_with_dict_size(input, DEFAULT_DICT_SIZE)
}

/// Compress input using LZ78 with a specific maximum dictionary size.
///
/// `max_dict_size` must be >= 2 (at least root + one entry).
pub fn encode_with_dict_size(input: &[u8], max_dict_size: u16) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if max_dict_size < 2 {
        return Err(PzError::InvalidInput);
    }

    // Trie: (parent_index, byte) -> child_index
    // Index 0 = root (empty string)
    let mut trie: HashMap<(u16, u8), u16> = HashMap::new();
    let mut next_index: u16 = 1;
    let mut tokens: Vec<(u16, u8)> = Vec::new();
    let mut pos = 0;

    while pos < input.len() {
        let mut current_index: u16 = 0; // start at root

        // Walk the trie, finding the longest known prefix
        while pos < input.len() {
            let byte = input[pos];
            if let Some(&child) = trie.get(&(current_index, byte)) {
                current_index = child;
                pos += 1;
            } else {
                break;
            }
        }

        if pos < input.len() {
            let byte = input[pos];
            tokens.push((current_index, byte));

            // Add new entry to dictionary if not full
            if next_index < max_dict_size {
                trie.insert((current_index, byte), next_index);
                next_index += 1;
            }
            pos += 1;
        } else {
            // Reached end of input while matching a prefix.
            // Emit token with a dummy next byte; decoder uses original_len
            // to truncate output to the correct length.
            tokens.push((current_index, 0));
        }
    }

    // Serialize
    let total_size = HEADER_SIZE + tokens.len() * TOKEN_SIZE;
    let mut output = Vec::with_capacity(total_size);
    output.extend_from_slice(&(input.len() as u32).to_le_bytes());
    output.extend_from_slice(&max_dict_size.to_le_bytes());
    output.extend_from_slice(&(tokens.len() as u32).to_le_bytes());

    for &(index, next) in &tokens {
        output.extend_from_slice(&index.to_le_bytes());
        output.push(next);
    }

    Ok(output)
}

/// Decompress LZ78-compressed data.
pub fn decode(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if input.len() < HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    let original_len = u32::from_le_bytes(input[0..4].try_into().unwrap()) as usize;
    let max_dict_size = u16::from_le_bytes(input[4..6].try_into().unwrap()) as usize;
    let num_tokens = u32::from_le_bytes(input[6..10].try_into().unwrap()) as usize;

    let expected_data_len = num_tokens * TOKEN_SIZE;
    if input.len() < HEADER_SIZE + expected_data_len {
        return Err(PzError::InvalidInput);
    }

    // Dictionary: index -> full string
    // Index 0 = root = empty string
    let mut dict: Vec<Vec<u8>> = Vec::with_capacity(max_dict_size.min(65536));
    dict.push(Vec::new()); // root

    let mut output = Vec::with_capacity(original_len);
    let token_data = &input[HEADER_SIZE..];

    for i in 0..num_tokens {
        let base = i * TOKEN_SIZE;
        let index = u16::from_le_bytes([token_data[base], token_data[base + 1]]) as usize;
        let next = token_data[base + 2];

        if index >= dict.len() {
            return Err(PzError::InvalidInput);
        }

        // Output = dict[index] + next_byte
        let prefix = &dict[index];
        output.extend_from_slice(prefix);
        output.push(next);

        // Add new dictionary entry: prefix + next
        if dict.len() < max_dict_size {
            let mut new_entry = prefix.clone();
            new_entry.push(next);
            dict.push(new_entry);
        }
    }

    // Truncate to original length (handles end-of-stream edge case)
    output.truncate(original_len);
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
    fn test_round_trip_short() {
        let input = b"abcabc".to_vec();
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
    fn test_round_trip_text() {
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
    fn test_round_trip_two_bytes() {
        let input = vec![0u8, 255];
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_small_dict() {
        let input = b"aaabbbaaabbbaaabbb".to_vec();
        let compressed = encode_with_dict_size(&input, 8).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_large_dict() {
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        let compressed = encode_with_dict_size(&input, 65535).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_dict_fills_up() {
        // With a tiny dictionary, it should fill up quickly but still work
        let mut input = Vec::new();
        for i in 0..500u16 {
            input.push((i % 256) as u8);
            input.push(((i / 256) % 256) as u8);
        }
        let compressed = encode_with_dict_size(&input, 16).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_invalid_dict_size() {
        let input = b"hello".to_vec();
        let result = encode_with_dict_size(&input, 1);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_invalid_short() {
        let result = decode(&[0, 1, 2]);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_invalid_truncated() {
        let mut data = vec![0u8; HEADER_SIZE];
        // original_len = 10, max_dict_size = 256, num_tokens = 100 (too many for data)
        data[0..4].copy_from_slice(&10u32.to_le_bytes());
        data[4..6].copy_from_slice(&256u16.to_le_bytes());
        data[6..10].copy_from_slice(&100u32.to_le_bytes());
        let result = decode(&data);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_invalid_index() {
        // Craft a token with an out-of-range dictionary index
        let mut data = vec![0u8; HEADER_SIZE + TOKEN_SIZE];
        data[0..4].copy_from_slice(&1u32.to_le_bytes()); // original_len = 1
        data[4..6].copy_from_slice(&256u16.to_le_bytes()); // max_dict_size
        data[6..10].copy_from_slice(&1u32.to_le_bytes()); // num_tokens = 1
                                                          // Token: index = 999 (invalid), next = 'a'
        data[10..12].copy_from_slice(&999u16.to_le_bytes());
        data[12] = b'a';
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
        let mut buf = vec![0u8; 1];
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
        let mut buf = vec![0u8; 1];
        let result = decode_to_buf(&encoded, &mut buf);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_compresses_repetitive_data() {
        let input = vec![b'a'; 1000];
        let compressed = encode(&input).unwrap();
        assert!(
            compressed.len() < input.len(),
            "LZ78 should compress repeated data: {} >= {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_repeating_pattern() {
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }
}
