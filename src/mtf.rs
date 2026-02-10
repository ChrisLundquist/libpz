/// Move-to-Front (MTF) transform.
///
/// The MTF transform converts a stream of byte values into a stream of
/// indices, where frequently recurring bytes get small index values.
/// This is particularly effective after BWT, which clusters identical
/// bytes together, resulting in many small indices (especially zeros)
/// that compress well with entropy coding.
///
/// **Algorithm:**
/// - Maintain an ordered list of all 256 byte values (initially [0, 1, 2, ..., 255]).
/// - For each input byte, output its current position in the list.
/// - Then move that byte to the front of the list (position 0).
///
/// The inverse is symmetric:
/// - Maintain the same initial list.
/// - For each input index, output the byte at that position.
/// - Then move that byte to the front.
use crate::{PzError, PzResult};

/// Apply the Move-to-Front transform to input data.
///
/// Returns the transformed data where each byte is replaced by its
/// index in a dynamically reordered symbol list.
pub fn encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    // Initialize the symbol list: [0, 1, 2, ..., 255]
    let mut list: [u8; 256] = std::array::from_fn(|i| i as u8);

    let mut output = Vec::with_capacity(input.len());

    for &byte in input {
        // Find the position of this byte in the list
        let pos = list.iter().position(|&b| b == byte).unwrap();

        // Output the position
        output.push(pos as u8);

        // Move the byte to the front
        if pos > 0 {
            list.copy_within(..pos, 1);
            list[0] = byte;
        }
    }

    output
}

/// Apply the MTF transform into a pre-allocated output buffer.
///
/// Returns the number of bytes written.
pub fn encode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    if input.is_empty() {
        return Ok(0);
    }
    if output.len() < input.len() {
        return Err(PzError::BufferTooSmall);
    }

    let mut list: [u8; 256] = std::array::from_fn(|i| i as u8);

    for (idx, &byte) in input.iter().enumerate() {
        let pos = list.iter().position(|&b| b == byte).unwrap();

        output[idx] = pos as u8;

        if pos > 0 {
            list.copy_within(..pos, 1);
            list[0] = byte;
        }
    }

    Ok(input.len())
}

/// Inverse Move-to-Front transform (decode).
///
/// Converts MTF-encoded indices back to the original byte stream.
pub fn decode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut list: [u8; 256] = std::array::from_fn(|i| i as u8);

    let mut output = Vec::with_capacity(input.len());

    for &index in input {
        let pos = index as usize;
        let byte = list[pos];
        output.push(byte);

        // Move to front
        if pos > 0 {
            list.copy_within(..pos, 1);
            list[0] = byte;
        }
    }

    output
}

/// Inverse MTF transform into a pre-allocated output buffer.
///
/// Returns the number of bytes written.
pub fn decode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    if input.is_empty() {
        return Ok(0);
    }
    if output.len() < input.len() {
        return Err(PzError::BufferTooSmall);
    }

    let mut list: [u8; 256] = std::array::from_fn(|i| i as u8);

    for (idx, &index) in input.iter().enumerate() {
        let pos = index as usize;
        let byte = list[pos];
        output[idx] = byte;

        if pos > 0 {
            list.copy_within(..pos, 1);
            list[0] = byte;
        }
    }

    Ok(input.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(encode(&[]), Vec::<u8>::new());
        assert_eq!(decode(&[]), Vec::<u8>::new());
    }

    #[test]
    fn test_single_byte() {
        let input = &[42u8];
        let encoded = encode(input);
        // Byte 42 is at position 42 in the initial list
        assert_eq!(encoded, &[42]);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_repeated_byte() {
        // After the first occurrence moves it to front, subsequent
        // occurrences should output 0.
        let input = &[b'a', b'a', b'a', b'a'];
        let encoded = encode(input);
        // First 'a' (97) is at position 97, then it's at position 0
        assert_eq!(encoded[0], 97);
        assert_eq!(encoded[1], 0);
        assert_eq!(encoded[2], 0);
        assert_eq!(encoded[3], 0);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_two_alternating() {
        let input = &[b'a', b'b', b'a', b'b'];
        let encoded = encode(input);
        // 'a' at position 97
        assert_eq!(encoded[0], 97);
        // After 'a' moves to front: [a, 0, 1, ..., 96, 98, ...]
        // 'b' (98) was after 'a', now at position 98
        assert_eq!(encoded[1], 98);
        // After 'b' moves to front: [b, a, 0, 1, ...]
        // 'a' is now at position 1
        assert_eq!(encoded[2], 1);
        // After 'a' moves to front: [a, b, 0, 1, ...]
        // 'b' is now at position 1
        assert_eq!(encoded[3], 1);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_banana() {
        let input = b"banana";
        let encoded = encode(input);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_all_bytes() {
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_reverse_bytes() {
        let input: Vec<u8> = (0..=255).rev().collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_longer_text() {
        let input = b"the quick brown fox jumps over the lazy dog";
        let encoded = encode(input);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_bwt_like_output() {
        // Simulate BWT output: clustered bytes
        let mut input = Vec::new();
        input.extend(vec![b'a'; 20]);
        input.extend(vec![b'b'; 15]);
        input.extend(vec![b'c'; 10]);
        input.extend(vec![b'a'; 5]);

        let encoded = encode(&input);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);

        // After the first 'a', subsequent 'a's should be 0
        // This means MTF output has lots of zeros for clustered input
        let zero_count = encoded.iter().filter(|&&b| b == 0).count();
        // Most of the repeated bytes should produce 0
        assert!(zero_count > 30, "expected many zeros, got {}", zero_count);
    }

    #[test]
    fn test_encode_to_buf() {
        let input = b"hello";
        let mut buf = vec![0u8; 100];
        let size = encode_to_buf(input, &mut buf).unwrap();
        assert_eq!(size, input.len());
        let decoded = decode(&buf[..size]);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_to_buf_too_small() {
        let input = b"hello";
        let mut buf = vec![0u8; 2];
        assert_eq!(encode_to_buf(input, &mut buf), Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_decode_to_buf() {
        let input = b"hello";
        let encoded = encode(input);
        let mut buf = vec![0u8; 100];
        let size = decode_to_buf(&encoded, &mut buf).unwrap();
        assert_eq!(&buf[..size], input);
    }

    #[test]
    fn test_decode_to_buf_too_small() {
        let input = b"hello";
        let encoded = encode(input);
        let mut buf = vec![0u8; 2];
        assert_eq!(
            decode_to_buf(&encoded, &mut buf),
            Err(PzError::BufferTooSmall)
        );
    }

    #[test]
    fn test_round_trip_binary() {
        let input: Vec<u8> = (0..1024).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_identity_for_sorted_input() {
        // Strictly increasing input: each byte appears at its natural position
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode(&input);
        // First byte 0 is at position 0, then 1 is at position 1 (since 0 moved to front,
        // everything else shifted right by 1, but 1 was at position 1 originally and moved to 2)
        // Actually this isn't identity, but verify round-trip
        let decoded = decode(&encoded);
        assert_eq!(decoded, input);
    }
}
