/// Run-Length Encoding (RLE).
///
/// Uses a bzip2-style encoding scheme optimized for BWT+MTF output
/// where long runs of identical bytes (especially zeros) are common.
///
/// **Encoding format:**
/// - Bytes are emitted literally.
/// - When a byte appears 4 or more times consecutively, the first 4
///   occurrences are emitted literally, followed by a count byte indicating
///   how many additional repetitions follow (0..255, so runs of 4..259).
///
/// This is efficient for BWT+MTF output because:
/// - Short runs (1-3) have zero overhead
/// - Long runs are compactly represented
/// - The format is unambiguous and simple to decode

use crate::{PzError, PzResult};

/// Maximum additional run length after the initial 4 bytes.
const MAX_RUN_EXTRA: usize = 255;

/// Encode data using run-length encoding.
///
/// Returns the RLE-encoded data.
pub fn encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(input.len());
    let mut i = 0;

    while i < input.len() {
        let byte = input[i];

        // Count the run length
        let mut run_len = 1;
        while i + run_len < input.len() && input[i + run_len] == byte {
            run_len += 1;
        }

        // Emit in chunks of MAX_RUN_TOTAL
        let mut remaining = run_len;
        while remaining > 0 {
            if remaining >= 4 {
                // Emit 4 literal bytes + count of additional
                output.push(byte);
                output.push(byte);
                output.push(byte);
                output.push(byte);
                let extra = std::cmp::min(remaining - 4, MAX_RUN_EXTRA);
                output.push(extra as u8);
                remaining -= 4 + extra;
            } else {
                // Short run: emit literally
                for _ in 0..remaining {
                    output.push(byte);
                }
                remaining = 0;
            }
        }

        i += run_len;
    }

    output
}

/// Encode data into a pre-allocated output buffer.
///
/// Returns the number of bytes written.
pub fn encode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut out_pos = 0;
    let mut i = 0;

    while i < input.len() {
        let byte = input[i];

        let mut run_len = 1;
        while i + run_len < input.len() && input[i + run_len] == byte {
            run_len += 1;
        }

        let mut remaining = run_len;
        while remaining > 0 {
            if remaining >= 4 {
                if out_pos + 5 > output.len() {
                    return Err(PzError::BufferTooSmall);
                }
                output[out_pos] = byte;
                output[out_pos + 1] = byte;
                output[out_pos + 2] = byte;
                output[out_pos + 3] = byte;
                let extra = std::cmp::min(remaining - 4, MAX_RUN_EXTRA);
                output[out_pos + 4] = extra as u8;
                out_pos += 5;
                remaining -= 4 + extra;
            } else {
                if out_pos + remaining > output.len() {
                    return Err(PzError::BufferTooSmall);
                }
                for _ in 0..remaining {
                    output[out_pos] = byte;
                    out_pos += 1;
                }
                remaining = 0;
            }
        }

        i += run_len;
    }

    Ok(out_pos)
}

/// Decode RLE-encoded data.
///
/// Returns the decoded data.
pub fn decode(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(input.len());
    let mut i = 0;

    while i < input.len() {
        let byte = input[i];
        output.push(byte);
        i += 1;

        // Check for run of 4 identical bytes
        let mut run_count = 1;
        while run_count < 3 && i < input.len() && input[i] == byte {
            output.push(byte);
            run_count += 1;
            i += 1;
        }

        // If we saw 3 more of the same byte (4 total), next byte is run count
        if run_count == 3 && i < input.len() && input[i] == byte {
            output.push(byte);
            i += 1;

            // Next byte is the extra count
            if i >= input.len() {
                return Err(PzError::InvalidInput);
            }
            let extra = input[i] as usize;
            i += 1;

            for _ in 0..extra {
                output.push(byte);
            }
        }
    }

    Ok(output)
}

/// Decode RLE-encoded data into a pre-allocated output buffer.
///
/// Returns the number of bytes written.
pub fn decode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut out_pos = 0;
    let mut i = 0;

    while i < input.len() {
        let byte = input[i];
        if out_pos >= output.len() {
            return Err(PzError::BufferTooSmall);
        }
        output[out_pos] = byte;
        out_pos += 1;
        i += 1;

        let mut run_count = 1;
        while run_count < 3 && i < input.len() && input[i] == byte {
            if out_pos >= output.len() {
                return Err(PzError::BufferTooSmall);
            }
            output[out_pos] = byte;
            out_pos += 1;
            run_count += 1;
            i += 1;
        }

        if run_count == 3 && i < input.len() && input[i] == byte {
            if out_pos >= output.len() {
                return Err(PzError::BufferTooSmall);
            }
            output[out_pos] = byte;
            out_pos += 1;
            i += 1;

            if i >= input.len() {
                return Err(PzError::InvalidInput);
            }
            let extra = input[i] as usize;
            i += 1;

            if out_pos + extra > output.len() {
                return Err(PzError::BufferTooSmall);
            }
            for _ in 0..extra {
                output[out_pos] = byte;
                out_pos += 1;
            }
        }
    }

    Ok(out_pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(encode(&[]), Vec::<u8>::new());
        assert_eq!(decode(&[]).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_no_runs() {
        let input = b"abcdef";
        let encoded = encode(input);
        // No runs >= 4, so output should equal input
        assert_eq!(encoded, input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_short_run() {
        // Runs of 1-3 are emitted literally
        let input = b"aabbc";
        let encoded = encode(input);
        assert_eq!(encoded, input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_run_of_exactly_4() {
        let input = b"aaaa";
        let encoded = encode(input);
        // 4 bytes of 'a' + count 0 (no extra)
        assert_eq!(encoded, &[b'a', b'a', b'a', b'a', 0]);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_run_of_5() {
        let input = b"aaaaa";
        let encoded = encode(input);
        // 4 bytes of 'a' + count 1 (1 extra)
        assert_eq!(encoded, &[b'a', b'a', b'a', b'a', 1]);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_run_of_10() {
        let input = vec![b'x'; 10];
        let encoded = encode(&input);
        // 4 bytes + count 6
        assert_eq!(encoded, &[b'x', b'x', b'x', b'x', 6]);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_max_run_259() {
        let input = vec![b'z'; 4 + MAX_RUN_EXTRA];
        let encoded = encode(&input);
        // 4 bytes + count 255
        assert_eq!(encoded, &[b'z', b'z', b'z', b'z', 255]);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_run_exceeding_max() {
        // 260 bytes = one chunk of 259 + 1 leftover literal
        let input = vec![b'a'; 260];
        let encoded = encode(&input);
        // First chunk: a a a a 255, second chunk: a (literal)
        assert_eq!(encoded, &[b'a', b'a', b'a', b'a', 255, b'a']);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_large_run_520() {
        // 520 = 259 + 259 + 2
        let input = vec![b'b'; 520];
        let encoded = encode(&input);
        // Two full chunks + 2 literal
        assert_eq!(
            encoded,
            &[b'b', b'b', b'b', b'b', 255, b'b', b'b', b'b', b'b', 255, b'b', b'b']
        );
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_mixed_runs_and_literals() {
        let mut input = Vec::new();
        input.extend(b"abc");          // literals
        input.extend(vec![b'd'; 7]);   // run of 7
        input.extend(b"ef");           // literals
        input.extend(vec![b'g'; 4]);   // run of 4

        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_single_byte() {
        let input = b"x";
        let encoded = encode(input);
        assert_eq!(encoded, input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_three_same_bytes() {
        let input = b"aaa";
        let encoded = encode(input);
        assert_eq!(encoded, input); // no RLE for < 4
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_binary() {
        // Binary data with natural runs
        let mut input = Vec::new();
        for i in 0..256u16 {
            for _ in 0..(i % 7 + 1) {
                input.push(i as u8);
            }
        }
        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_all_zeros() {
        // Simulates BWT+MTF output with lots of zeros
        let input = vec![0u8; 1000];
        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
        // Should compress significantly
        assert!(encoded.len() < input.len());
    }

    #[test]
    fn test_encode_to_buf() {
        let input = vec![b'a'; 10];
        let mut buf = vec![0u8; 100];
        let size = encode_to_buf(&input, &mut buf).unwrap();
        let decoded = decode(&buf[..size]).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_to_buf_too_small() {
        let input = vec![b'a'; 100];
        let mut buf = vec![0u8; 2]; // too small
        let result = encode_to_buf(&input, &mut buf);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_decode_to_buf() {
        let input = vec![b'x'; 20];
        let encoded = encode(&input);
        let mut buf = vec![0u8; 100];
        let size = decode_to_buf(&encoded, &mut buf).unwrap();
        assert_eq!(&buf[..size], &input[..]);
    }

    #[test]
    fn test_decode_to_buf_too_small() {
        let input = vec![b'x'; 20];
        let encoded = encode(&input);
        let mut buf = vec![0u8; 2]; // too small
        let result = decode_to_buf(&encoded, &mut buf);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_alternating_runs() {
        // aaaa bbbb cccc
        let mut input = Vec::new();
        input.extend(vec![b'a'; 5]);
        input.extend(vec![b'b'; 6]);
        input.extend(vec![b'c'; 4]);

        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }
}
