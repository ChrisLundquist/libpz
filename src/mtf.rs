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
///
/// Uses an inverse index array for O(1) position lookup instead of
/// O(256) linear search.
pub fn encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    // list[rank] = symbol at that rank
    let mut list: [u8; 256] = std::array::from_fn(|i| i as u8);
    // inv[symbol] = rank of that symbol (inverse of list)
    let mut inv: [u8; 256] = std::array::from_fn(|i| i as u8);

    let mut output = Vec::with_capacity(input.len());

    for &byte in input {
        // O(1) lookup of position
        let pos = inv[byte as usize] as usize;

        output.push(pos as u8);

        // Move the byte to the front
        if pos > 0 {
            // Update inverse index for all symbols that shift right by 1
            for i in 0..pos {
                inv[list[i] as usize] += 1;
            }
            list.copy_within(..pos, 1);
            list[0] = byte;
            inv[byte as usize] = 0;
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
    let mut inv: [u8; 256] = std::array::from_fn(|i| i as u8);

    for (idx, &byte) in input.iter().enumerate() {
        let pos = inv[byte as usize] as usize;

        output[idx] = pos as u8;

        if pos > 0 {
            for i in 0..pos {
                inv[list[i] as usize] += 1;
            }
            list.copy_within(..pos, 1);
            list[0] = byte;
            inv[byte as usize] = 0;
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

/// Apply the MTF-1 (delayed promotion) transform to input data.
///
/// MTF-1 is a variant of the Move-to-Front transform that uses a delayed
/// promotion strategy. Instead of always moving an accessed symbol to
/// position 0, the promotion rule is:
///
/// - If the symbol is at position 0: leave it there (output 0).
/// - If the symbol is at position 1: move it to position 0 (swap with
///   the symbol currently at position 0). Output 1.
/// - If the symbol is at position p >= 2: move it to position 1, shifting
///   symbols at positions 1..p-1 right by one. Output p.
///
/// This prevents transient (one-off) symbols from polluting position 0,
/// which can improve compression on BWT output where the dominant symbol
/// in a run should hold position 0 without being displaced by occasional
/// outliers.
///
/// Uses an inverse index array for O(1) position lookup, matching the
/// optimization used by `encode`.
pub fn encode_mtf1(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    // list[rank] = symbol at that rank
    let mut list: [u8; 256] = std::array::from_fn(|i| i as u8);
    // inv[symbol] = rank of that symbol (inverse of list)
    let mut inv: [u8; 256] = std::array::from_fn(|i| i as u8);

    let mut output = Vec::with_capacity(input.len());

    for &byte in input {
        // O(1) lookup of position
        let pos = inv[byte as usize] as usize;

        output.push(pos as u8);

        // Apply MTF-1 promotion rule
        match pos {
            0 => {
                // Already at position 0, nothing to do.
            }
            1 => {
                // At position 1: swap with position 0 (promote to front).
                let front = list[0];
                list[0] = byte;
                list[1] = front;
                inv[byte as usize] = 0;
                inv[front as usize] = 1;
            }
            _ => {
                // At position p >= 2: move to position 1.
                // Shift positions 1..p-1 right by one to make room at position 1.
                // Position 0 is untouched.

                // Update inverse indices for symbols that shift right by 1
                // (those at positions 1..p-1).
                for i in 1..pos {
                    inv[list[i] as usize] += 1;
                }

                // Shift the list: copy [1..pos) to [2..pos+1)
                list.copy_within(1..pos, 2);
                list[1] = byte;
                inv[byte as usize] = 1;
            }
        }
    }

    output
}

/// Inverse MTF-1 (delayed promotion) transform (decode).
///
/// Converts MTF-1-encoded indices back to the original byte stream.
/// Applies the same promotion rule as `encode_mtf1`:
///
/// - Index 0: symbol at position 0, leave in place.
/// - Index 1: symbol at position 1, swap with position 0.
/// - Index p >= 2: symbol at position p, move to position 1.
pub fn decode_mtf1(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut list: [u8; 256] = std::array::from_fn(|i| i as u8);

    let mut output = Vec::with_capacity(input.len());

    for &index in input {
        let pos = index as usize;
        let byte = list[pos];
        output.push(byte);

        // Apply MTF-1 promotion rule
        match pos {
            0 => {
                // Already at position 0, nothing to do.
            }
            1 => {
                // At position 1: swap with position 0.
                list[1] = list[0];
                list[0] = byte;
            }
            _ => {
                // At position p >= 2: move to position 1.
                // Shift positions 1..p-1 right by one, leave position 0 alone.
                list.copy_within(1..pos, 2);
                list[1] = byte;
            }
        }
    }

    output
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
        let input: &[u8] = b"aaaa";
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
        let input: &[u8] = b"abab";
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
    fn test_encode_to_buf_too_small() {
        let input = b"hello";
        let mut buf = vec![0u8; 2];
        assert_eq!(encode_to_buf(input, &mut buf), Err(PzError::BufferTooSmall));
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

    // ---- MTF-1 tests ----

    #[test]
    fn test_mtf1_empty() {
        assert_eq!(encode_mtf1(&[]), Vec::<u8>::new());
        assert_eq!(decode_mtf1(&[]), Vec::<u8>::new());
    }

    #[test]
    fn test_mtf1_single_byte() {
        let input = &[42u8];
        let encoded = encode_mtf1(input);
        // Byte 42 is at position 42 in the initial list
        assert_eq!(encoded, &[42]);
        let decoded = decode_mtf1(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_mtf1_roundtrip_repeated() {
        let input: &[u8] = b"aaaa";
        let encoded = encode_mtf1(input);
        // First 'a' (97) at position 97 -> moves to position 1
        // Second 'a' at position 1 -> swap with position 0 -> moves to position 0
        // Third 'a' at position 0 -> stays
        // Fourth 'a' at position 0 -> stays
        assert_eq!(encoded[0], 97);
        assert_eq!(encoded[1], 1);
        assert_eq!(encoded[2], 0);
        assert_eq!(encoded[3], 0);
        let decoded = decode_mtf1(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_mtf1_roundtrip_alternating() {
        let input: &[u8] = b"abababab";
        let encoded = encode_mtf1(input);
        let decoded = decode_mtf1(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_mtf1_roundtrip_all_bytes() {
        // Test with all 256 byte values
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode_mtf1(&input);
        let decoded = decode_mtf1(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_mtf1_roundtrip_random_pattern() {
        // A longer pseudo-random pattern
        let input: Vec<u8> = (0..1000).map(|i| ((i * 7 + 13) % 256) as u8).collect();
        let encoded = encode_mtf1(&input);
        let decoded = decode_mtf1(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_mtf1_roundtrip_bwt_like() {
        // Simulate BWT output: clustered bytes
        let mut input = Vec::new();
        input.extend(vec![b'a'; 50]);
        input.extend(vec![b'b'; 30]);
        input.extend(vec![b'c'; 20]);
        input.extend(vec![b'a'; 40]);
        input.extend(vec![b'd'; 10]);

        let encoded = encode_mtf1(&input);
        let decoded = decode_mtf1(&encoded);
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_mtf1_differs_from_mtf() {
        // MTF-1 should produce different output than standard MTF
        // for inputs where the delayed promotion matters.
        let input: &[u8] = b"abcabc";
        let mtf_encoded = encode(input);
        let mtf1_encoded = encode_mtf1(input);
        assert_ne!(
            mtf_encoded, mtf1_encoded,
            "MTF-1 should differ from standard MTF on non-trivial input"
        );
    }

    #[test]
    fn test_mtf1_more_zeros_on_clustered_data() {
        // On BWT-like clustered data with frequent interruptions, MTF-1
        // should produce more zeros than standard MTF because transient
        // symbols don't displace the dominant symbol from position 0.
        //
        // With standard MTF on "aaa b aaa b ...":
        //   Each 'b' moves to position 0, pushing 'a' to position 1.
        //   The next 'a' must output 1 to reclaim position 0.
        //
        // With MTF-1 on "aaa b aaa b ...":
        //   Each 'b' only goes to position 1, so 'a' stays at position 0.
        //   The next 'a' still outputs 0.
        //
        // The advantage grows with more frequent interruptions by diverse
        // transient symbols (typical of real BWT output).
        let mut input = Vec::new();
        // Dominant 'a' with frequent, diverse interrupters
        for i in 0..50 {
            input.extend(vec![b'a'; 3]);
            // Use different interrupter symbols to prevent them from
            // accumulating at position 1 in standard MTF
            input.push(b'b' + (i % 5));
        }

        let mtf_encoded = encode(&input);
        let mtf1_encoded = encode_mtf1(&input);

        let mtf_zeros = mtf_encoded.iter().filter(|&&b| b == 0).count();
        let mtf1_zeros = mtf1_encoded.iter().filter(|&&b| b == 0).count();

        assert!(
            mtf1_zeros > mtf_zeros,
            "MTF-1 should produce more zeros than MTF on clustered data \
             (MTF zeros: {}, MTF-1 zeros: {})",
            mtf_zeros,
            mtf1_zeros,
        );
    }

    #[test]
    fn test_mtf1_position_mechanics() {
        // Manually verify the MTF-1 list state transitions.
        // Initial list: [0, 1, 2, 3, ...]
        //
        // Encode byte 3:
        //   pos = 3, output 3. Move to position 1.
        //   List: [0, 3, 1, 2, 4, 5, ...]
        //
        // Encode byte 3:
        //   pos = 1, output 1. Swap with position 0.
        //   List: [3, 0, 1, 2, 4, 5, ...]
        //
        // Encode byte 3:
        //   pos = 0, output 0. Stay.
        //   List: [3, 0, 1, 2, 4, 5, ...]
        //
        // Encode byte 5:
        //   pos = 5 (byte 5 is at index 5), output 5. Move to position 1.
        //   List: [3, 5, 0, 1, 2, 4, 6, ...]
        //
        // Encode byte 3:
        //   pos = 0, output 0. Stay.
        let input: &[u8] = &[3, 3, 3, 5, 3];
        let encoded = encode_mtf1(input);
        assert_eq!(encoded, &[3, 1, 0, 5, 0]);
        let decoded = decode_mtf1(&encoded);
        assert_eq!(decoded, input);
    }
}
