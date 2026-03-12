use super::*;

// --- Frequency normalization ---

#[test]
fn test_normalize_frequencies_basic() {
    let mut freq = FrequencyTable::new();
    freq.count(b"aaabbc");
    let norm = normalize_frequencies(&freq, 12).unwrap();
    let sum: u32 = norm.freq.iter().map(|&f| f as u32).sum();
    assert_eq!(sum, 4096);
    assert!(norm.freq[b'a' as usize] > 0);
    assert!(norm.freq[b'b' as usize] > 0);
    assert!(norm.freq[b'c' as usize] > 0);
    // Cumulative must be monotonically non-decreasing
    for i in 1..NUM_SYMBOLS {
        assert!(norm.cum[i] >= norm.cum[i - 1]);
    }
}

#[test]
fn test_normalize_single_symbol() {
    let mut freq = FrequencyTable::new();
    freq.count(&[42u8; 100]);
    let norm = normalize_frequencies(&freq, 12).unwrap();
    assert_eq!(norm.freq[42], 4096);
    assert_eq!(norm.cum[42], 0);
}

#[test]
fn test_normalize_preserves_order() {
    let mut freq = FrequencyTable::new();
    freq.count(b"aaaaabbbcc");
    let norm = normalize_frequencies(&freq, 12).unwrap();
    assert!(norm.freq[b'a' as usize] > norm.freq[b'b' as usize]);
    assert!(norm.freq[b'b' as usize] > norm.freq[b'c' as usize]);
}

#[test]
fn test_normalize_empty_fails() {
    let freq = FrequencyTable::new();
    assert_eq!(normalize_frequencies(&freq, 12), Err(PzError::InvalidInput));
}

// --- Symbol lookup ---

#[test]
fn test_symbol_lookup_coverage() {
    let mut freq = FrequencyTable::new();
    freq.count(b"aaabbc");
    let norm = normalize_frequencies(&freq, 12).unwrap();
    let lookup = build_symbol_lookup(&norm);
    assert_eq!(lookup.len(), 4096);
    // Every entry should map to a valid present symbol
    for &s in &lookup {
        assert!(norm.freq[s as usize] > 0);
    }
}

// --- Division helper ---

#[test]
fn test_rans_div_rcp_correctness() {
    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    // Test across all supported scale_bits
    for sb in MIN_SCALE_BITS..=MAX_SCALE_BITS {
        let mut freq = FrequencyTable::new();
        freq.count(&input);
        let norm = normalize_frequencies(&freq, sb).unwrap();
        let rcp_table = ReciprocalTable::from_normalized(&norm);

        for sym in 0..NUM_SYMBOLS {
            if norm.freq[sym] == 0 {
                continue;
            }
            let f = norm.freq[sym] as u32;
            let rcp = rcp_table.rcp[sym];
            // Test across the full rANS state range
            let test_vals: Vec<u32> = vec![
                1,
                f,
                f + 1,
                RANS_L - 1,
                RANS_L,
                RANS_L + 1,
                RANS_L * 2,
                0xFFFF,
                0x1_0000,
                0x7FFF_FFFF,
                0xFFFF_FFFE,
                0xFFFF_FFFF,
            ];
            for x in test_vals {
                if x == 0 {
                    continue;
                }
                let (q, r) = rans_div_rcp(x, f, rcp);
                assert_eq!(
                    (q, r),
                    (x / f, x % f),
                    "rans_div_rcp wrong for x={x}, f={f}, rcp={rcp}, sb={sb}"
                );
            }
        }
    }
}

// --- Single-stream round-trip tests ---

#[test]
fn test_empty() {
    assert_eq!(encode(&[]), Vec::<u8>::new());
    assert_eq!(decode(&[], 0).unwrap(), Vec::<u8>::new());
}

#[test]
fn test_single_byte() {
    let input = &[42u8];
    let encoded = encode(input);
    let decoded = decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_repeated_byte() {
    let input = vec![b'a'; 100];
    let encoded = encode(&input);
    let decoded = decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_round_trip_all_bytes() {
    let input: Vec<u8> = (0..=255).collect();
    let encoded = encode(&input);
    let decoded = decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_round_trip_longer_text() {
    let input =
        b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
    let encoded = encode(input);
    let decoded = decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_round_trip_binary() {
    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let encoded = encode(&input);
    let decoded = decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

// --- Compression effectiveness ---

#[test]
fn test_compresses_repeated() {
    let input = vec![0u8; 2000];
    let encoded = encode(&input);
    let decoded = decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
    assert!(
        encoded.len() < input.len(),
        "encoded {} bytes, expected < {}",
        encoded.len(),
        input.len()
    );
}

// --- Scale bits variants ---

#[test]
fn test_all_scale_bits() {
    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    for sb in MIN_SCALE_BITS..=MAX_SCALE_BITS {
        let encoded = encode_with_scale(&input, sb);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input, "failed at scale_bits={}", sb);
    }
}

// --- Error handling ---

#[test]
fn test_decode_too_short() {
    let result = decode(&[0u8; 10], 5);
    assert_eq!(result, Err(PzError::InvalidInput));
}

#[test]
fn test_decode_invalid_scale_bits() {
    let mut bad = vec![0u8; HEADER_SIZE + 10];
    bad[0] = 15; // > MAX_SCALE_BITS
    assert_eq!(decode(&bad, 5), Err(PzError::InvalidInput));
}

#[test]
fn test_decode_to_buf_too_small() {
    let input = b"hello, world!";
    let encoded = encode(input);
    let mut buf = vec![0u8; 2];
    assert_eq!(
        decode_to_buf(&encoded, input.len(), &mut buf),
        Err(PzError::BufferTooSmall)
    );
}

// --- Interleaved round-trip tests ---

#[test]
fn test_interleaved_empty() {
    assert_eq!(encode_interleaved(&[]), Vec::<u8>::new());
    assert_eq!(decode_interleaved(&[], 0).unwrap(), Vec::<u8>::new());
}

#[test]
fn test_interleaved_single_byte() {
    let input = &[42u8];
    let encoded = encode_interleaved(input);
    let decoded = decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_interleaved_repeated() {
    let input = vec![b'x'; 500];
    let encoded = encode_interleaved(&input);
    let decoded = decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_interleaved_all_bytes() {
    let input: Vec<u8> = (0..=255).collect();
    let encoded = encode_interleaved(&input);
    let decoded = decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_interleaved_8way() {
    let input: Vec<u8> = (0..2000).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let encoded = encode_interleaved_n(&input, 8, DEFAULT_SCALE_BITS);
    let decoded = decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_interleaved_1way_matches_single() {
    // 1-way interleaved should produce the same decoded output as single-stream
    let input = b"the quick brown fox jumps over the lazy dog";
    let decoded_single = {
        let encoded = encode(input);
        decode(&encoded, input.len()).unwrap()
    };
    let decoded_interleaved = {
        let encoded = encode_interleaved_n(input, 1, DEFAULT_SCALE_BITS);
        decode_interleaved(&encoded, input.len()).unwrap()
    };
    assert_eq!(decoded_single, decoded_interleaved);
}

mod chunked_tests {
    use super::*;

    const NUM_STATES: usize = 4;
    const SCALE_BITS: u8 = DEFAULT_SCALE_BITS;
    const CHUNK_SIZE: usize = 1024;

    #[test]
    fn test_chunked_empty() {
        let encoded = encode_chunked(&[], NUM_STATES, SCALE_BITS, CHUNK_SIZE);
        assert_eq!(encoded, Vec::<u8>::new());
        let decoded = decode_chunked(&encoded).unwrap();
        assert_eq!(decoded, Vec::<u8>::new());
    }

    #[test]
    fn test_chunked_single_byte() {
        let input = &[42u8];
        let encoded = encode_chunked(input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
        let decoded = decode_chunked(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_chunked_multiple_chunks_exact() {
        let mut input = Vec::new();
        for i in 0..3 {
            for j in 0..CHUNK_SIZE {
                input.push((i * CHUNK_SIZE + j) as u8);
            }
        }
        let encoded = encode_chunked(&input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
        let decoded = decode_chunked(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_chunked_multiple_chunks_partial() {
        let input: Vec<u8> = (0..CHUNK_SIZE + 100).map(|i| i as u8).collect();
        let encoded = encode_chunked(&input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
        let decoded = decode_chunked(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_chunked_binary_data() {
        let input: Vec<u8> = (0..5000).map(|i| ((i * 41 + 61) % 256) as u8).collect();
        let encoded = encode_chunked(&input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
        let decoded = decode_chunked(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_chunked_different_params() {
        let input: Vec<u8> = (0..3000).map(|i| ((i * 41 + 61) % 256) as u8).collect();
        let encoded = encode_chunked(&input, 8, 13, 512);
        let decoded = decode_chunked(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_chunked_decode_invalid_input() {
        assert_eq!(decode_chunked(&[0; 10]), Err(PzError::InvalidInput));
        // Valid header but truncated data
        let input = b"some data";
        let valid_encoded = encode_chunked(input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
        assert_eq!(
            decode_chunked(&valid_encoded[..valid_encoded.len() - 10]),
            Err(PzError::InvalidInput)
        );
    }

    #[test]
    #[should_panic(expected = "num_states")]
    fn test_chunked_num_states_overflow_panics() {
        let input = vec![0u8; 1024];
        let _ = encode_chunked(&input, 256, SCALE_BITS, CHUNK_SIZE);
    }

    #[test]
    #[should_panic(expected = "chunk count")]
    fn test_chunked_chunk_count_overflow_panics() {
        // 65536 chunks (one byte each) exceeds u16 chunk-count header field.
        let input = vec![0u8; (u16::MAX as usize) + 1];
        let _ = encode_chunked(&input, NUM_STATES, SCALE_BITS, 1);
    }

    #[test]
    #[should_panic(expected = "chunk length")]
    fn test_chunked_chunk_len_overflow_panics() {
        // Single chunk length exceeds u16 per-chunk-len header field.
        let input = vec![0u8; (u16::MAX as usize) + 1];
        let _ = encode_chunked(&input, NUM_STATES, SCALE_BITS, (u16::MAX as usize) + 1);
    }
}

// --- Shared-stream round-trip tests ---

mod shared_stream_tests {
    use super::*;

    #[test]
    fn test_shared_stream_empty() {
        assert_eq!(encode_shared_stream(&[]), Vec::<u8>::new());
        assert_eq!(decode_shared_stream(&[], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_shared_stream_single_byte() {
        let input = &[42u8];
        let encoded = encode_shared_stream(input);
        let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_shared_stream_repeated() {
        let input = vec![b'x'; 500];
        let encoded = encode_shared_stream(&input);
        let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_shared_stream_all_bytes() {
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode_shared_stream(&input);
        let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_shared_stream_binary() {
        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded = encode_shared_stream(&input);
        let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_shared_stream_large() {
        let input: Vec<u8> = (0..100_000).map(|i| ((i * 41 + 61) % 256) as u8).collect();
        let encoded = encode_shared_stream(&input);
        let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_shared_stream_skewed_distribution() {
        // Heavy skew: 90% one symbol, 10% spread
        let mut input = vec![0u8; 9000];
        for i in 0..1000 {
            input.push(((i * 7 + 3) % 255 + 1) as u8);
        }
        let encoded = encode_shared_stream(&input);
        let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_shared_stream_all_same() {
        let input = vec![42u8; 10_000];
        let encoded = encode_shared_stream(&input);
        let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_shared_stream_small_sizes() {
        // Test sizes around and below num_states to exercise remainder logic
        for len in 1..=20 {
            let input: Vec<u8> = (0..len).map(|i| (i * 3) as u8).collect();
            let encoded = encode_shared_stream(&input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input, "round-trip failed at len={}", len);
        }
    }

    #[test]
    fn test_shared_stream_all_scale_bits() {
        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        for sb in MIN_SCALE_BITS..=MAX_SCALE_BITS {
            let encoded = encode_shared_stream_n(&input, 4, sb);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input, "failed at scale_bits={}", sb);
        }
    }

    #[test]
    fn test_shared_stream_matches_interleaved_output() {
        // Shared-stream and interleaved should decode to the same original data
        let input: Vec<u8> = (0..2048).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded_shared = encode_shared_stream(&input);
        let encoded_interleaved = encode_interleaved(&input);
        let decoded_shared = decode_shared_stream(&encoded_shared, input.len()).unwrap();
        let decoded_interleaved = decode_interleaved(&encoded_interleaved, input.len()).unwrap();
        assert_eq!(decoded_shared, input);
        assert_eq!(decoded_interleaved, input);
        assert_eq!(decoded_shared, decoded_interleaved);
    }

    // --- Sparse frequency tables ---

    #[test]
    fn test_sparse_freq_roundtrip() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaabbc");
        let norm = normalize_frequencies(&freq, 12).unwrap();

        let mut buf = Vec::new();
        serialize_freq_table_sparse(&norm, &mut buf);
        // 3 symbols: 1 + 3 + 6 = 10 bytes (vs 512 for dense)
        assert_eq!(buf.len(), 10);

        let (norm2, consumed) = deserialize_freq_table_sparse(&buf, 12).unwrap();
        assert_eq!(consumed, 10);
        assert_eq!(norm, norm2);
    }

    #[test]
    fn test_sparse_freq_all_symbols() {
        // All 256 symbols present
        let input: Vec<u8> = (0..=255).collect();
        let mut freq = FrequencyTable::new();
        freq.count(&input);
        let norm = normalize_frequencies(&freq, 12).unwrap();

        let mut buf = Vec::new();
        serialize_freq_table_sparse(&norm, &mut buf);
        // 256 symbols: 1 + 256 + 512 = 769 bytes (vs 512 for dense, slightly larger)
        // But this is the rare case; most streams have far fewer symbols.
        let (norm2, _) = deserialize_freq_table_sparse(&buf, 12).unwrap();
        assert_eq!(norm, norm2);
    }

    #[test]
    fn test_sparse_encode_decode_roundtrip() {
        let input = b"hello world! this is a test of sparse rANS encoding.";
        let encoded = encode_sparse(input, DEFAULT_SCALE_BITS);
        let decoded = decode_sparse(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input.as_slice());
    }

    #[test]
    fn test_sparse_smaller_than_dense() {
        // With few distinct symbols, sparse should be smaller
        let input: Vec<u8> = vec![0u8; 500]
            .into_iter()
            .chain(vec![1u8; 300])
            .chain(vec![2u8; 200])
            .collect();
        let dense = encode_with_scale(&input, DEFAULT_SCALE_BITS);
        let sparse = encode_sparse(&input, DEFAULT_SCALE_BITS);
        assert!(
            sparse.len() < dense.len(),
            "sparse {} should be < dense {}",
            sparse.len(),
            dense.len()
        );
        // Verify both decode correctly
        let dec_dense = decode(&dense, input.len()).unwrap();
        let dec_sparse = decode_sparse(&sparse, input.len()).unwrap();
        assert_eq!(dec_dense, input);
        assert_eq!(dec_sparse, input);
    }
}
