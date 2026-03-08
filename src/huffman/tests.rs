use super::*;

#[test]
fn test_build_from_empty() {
    let tree = HuffmanTree::from_data(&[]);
    assert!(tree.is_none());
}

#[test]
fn test_build_single_symbol() {
    let input = vec![b'a'; 10];
    let tree = HuffmanTree::from_data(&input).unwrap();
    assert_eq!(tree.leaf_count, 1);
    let (_, bits) = tree.get_code(b'a');
    assert_eq!(bits, 1);
}

#[test]
fn test_build_two_symbols() {
    let input = b"aabb";
    let tree = HuffmanTree::from_data(input).unwrap();
    assert_eq!(tree.leaf_count, 2);
    let (_, bits_a) = tree.get_code(b'a');
    let (_, bits_b) = tree.get_code(b'b');
    // Both should have 1-bit codes
    assert_eq!(bits_a, 1);
    assert_eq!(bits_b, 1);
}

#[test]
fn test_prefix_free() {
    // Verify no codeword is a prefix of another
    let input = b"aaabbbccddeef";
    let tree = HuffmanTree::from_data(input).unwrap();

    let mut codes: Vec<(u32, u8)> = Vec::new();
    for i in 0..=255u8 {
        let (cw, bits) = tree.get_code(i);
        if bits > 0 {
            codes.push((cw, bits));
        }
    }

    // Check no code is a prefix of another
    for i in 0..codes.len() {
        for j in 0..codes.len() {
            if i == j {
                continue;
            }
            let (cw_i, bits_i) = codes[i];
            let (cw_j, bits_j) = codes[j];
            if bits_i <= bits_j {
                let shifted = cw_j >> (bits_j - bits_i);
                assert_ne!(shifted, cw_i, "code {} is prefix of code {}", i, j);
            }
        }
    }
}

#[test]
fn test_encode_decode_round_trip() {
    let input = b"hello, world!";
    let tree = HuffmanTree::from_data(input).unwrap();
    let (encoded, total_bits) = tree.encode(input).unwrap();
    let decoded = tree.decode(&encoded, total_bits).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_encode_decode_all_bytes() {
    let input: Vec<u8> = (0..=255).collect();
    let tree = HuffmanTree::from_data(&input).unwrap();
    let (encoded, total_bits) = tree.encode(&input).unwrap();
    let decoded = tree.decode(&encoded, total_bits).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_encode_decode_skewed_distribution() {
    // 'a' appears much more often, should get shorter code
    let mut input = vec![b'a'; 100];
    input.push(b'b');
    input.push(b'c');

    let tree = HuffmanTree::from_data(&input).unwrap();
    let (_, bits_a) = tree.get_code(b'a');
    let (_, bits_b) = tree.get_code(b'b');
    assert!(
        bits_a <= bits_b,
        "more frequent symbol should have shorter code: a={}, b={}",
        bits_a,
        bits_b
    );

    let (encoded, total_bits) = tree.encode(&input).unwrap();
    let decoded = tree.decode(&encoded, total_bits).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_encode_buf_too_small() {
    let input = b"hello, world! this is a longer string";
    let tree = HuffmanTree::from_data(input).unwrap();
    let mut buf = vec![0u8; 1]; // too small
    let result = tree.encode_to_buf(input, &mut buf);
    assert_eq!(result, Err(PzError::BufferTooSmall));
}

#[test]
fn test_frequency_reconstruction() {
    let input = b"aaabbcc";
    let tree = HuffmanTree::from_data(input).unwrap();
    let freqs = tree.serialize_frequencies();

    // Reconstruct tree from frequencies
    let mut freq_table = FrequencyTable::new();
    freq_table.byte = freqs;
    freq_table.total = freqs.iter().map(|&f| f as u64).sum();
    freq_table.used = freqs.iter().filter(|&&f| f > 0).count() as u32;

    let tree2 = HuffmanTree::from_frequency_table(&freq_table).unwrap();

    // Both trees should produce identical encoding
    let (encoded1, bits1) = tree.encode(input).unwrap();
    let (encoded2, bits2) = tree2.encode(input).unwrap();
    assert_eq!(encoded1, encoded2);
    assert_eq!(bits1, bits2);
}

#[test]
fn test_round_trip_binary_data() {
    // Pseudo-random binary data
    let input: Vec<u8> = (0..500).map(|i| ((i * 17 + 31) % 256) as u8).collect();
    let tree = HuffmanTree::from_data(&input).unwrap();
    let (encoded, total_bits) = tree.encode(&input).unwrap();
    let decoded = tree.decode(&encoded, total_bits).unwrap();
    assert_eq!(decoded, input);
}

// --- HuffTable (canonical decode from code lengths) tests ---

#[test]
fn test_hufftable_from_lengths() {
    // DEFLATE fixed literal table: 0-143 → 8 bits, 144-255 → 9 bits,
    // 256-279 → 7 bits, 280-287 → 8 bits
    let mut lengths = [0u8; 288];
    for len in &mut lengths[0..=143] {
        *len = 8;
    }
    for len in &mut lengths[144..=255] {
        *len = 9;
    }
    for len in &mut lengths[256..=279] {
        *len = 7;
    }
    for len in &mut lengths[280..=287] {
        *len = 8;
    }

    let table = HuffTable::from_lengths(&lengths).unwrap();
    assert_eq!(table.symbol_count(), 288);
}

#[test]
fn test_hufftable_decode_two_symbols() {
    // Two symbols: A=0 (1 bit code "0"), B=1 (1 bit code "1")
    let lengths = [1u8, 1];
    let table = HuffTable::from_lengths(&lengths).unwrap();

    // Decode symbol from bit 0 → should be symbol 0
    let mut bits = [0u32].into_iter();
    let sym = table
        .decode(&mut || bits.next().ok_or(PzError::InvalidInput))
        .unwrap();
    assert_eq!(sym, 0);

    // Decode symbol from bit 1 → should be symbol 1
    let mut bits = [1u32].into_iter();
    let sym = table
        .decode(&mut || bits.next().ok_or(PzError::InvalidInput))
        .unwrap();
    assert_eq!(sym, 1);
}

#[test]
fn test_hufftable_reject_overlength() {
    let lengths = [16u8]; // exceeds MAX_CODE_BITS
    assert!(HuffTable::from_lengths(&lengths).is_err());
}
