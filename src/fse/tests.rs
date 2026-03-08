use super::*;

// --- Frequency normalization ---

#[test]
fn test_normalize_frequencies_basic() {
    let mut freq = FrequencyTable::new();
    freq.count(b"aaabbc");
    let norm = normalize_frequencies(&freq, 5).unwrap();
    let sum: u32 = norm.freq.iter().map(|&f| f as u32).sum();
    assert_eq!(sum, 32);
    assert!(norm.freq[b'a' as usize] > 0);
    assert!(norm.freq[b'b' as usize] > 0);
    assert!(norm.freq[b'c' as usize] > 0);
}

#[test]
fn test_normalize_single_symbol() {
    let mut freq = FrequencyTable::new();
    freq.count(&[42u8; 100]);
    let norm = normalize_frequencies(&freq, 8).unwrap();
    assert_eq!(norm.freq[42], 256);
}

#[test]
fn test_normalize_preserves_order() {
    let mut freq = FrequencyTable::new();
    freq.count(b"aaaaabbbcc");
    let norm = normalize_frequencies(&freq, 5).unwrap();
    assert!(norm.freq[b'a' as usize] > norm.freq[b'b' as usize]);
    assert!(norm.freq[b'b' as usize] > norm.freq[b'c' as usize]);
}

#[test]
fn test_normalize_empty_fails() {
    let freq = FrequencyTable::new();
    assert_eq!(normalize_frequencies(&freq, 8), Err(PzError::InvalidInput));
}

// --- Symbol spread ---

#[test]
fn test_spread_all_filled() {
    let mut freq = FrequencyTable::new();
    freq.count(b"aaabbc");
    let norm = normalize_frequencies(&freq, 5).unwrap();
    let spread = spread_symbols(&norm);
    assert_eq!(spread.len(), 32);
    assert!(spread.iter().all(|&s| s != 255));
}

#[test]
fn test_spread_counts_match() {
    let mut freq = FrequencyTable::new();
    freq.count(b"aaabbc");
    let norm = normalize_frequencies(&freq, 5).unwrap();
    let spread = spread_symbols(&norm);
    for (sym, &expected) in norm.freq.iter().enumerate() {
        if expected > 0 {
            let actual = spread.iter().filter(|&&s| s == sym as u8).count();
            assert_eq!(actual, expected as usize);
        }
    }
}

// --- Decode table ---

#[test]
fn test_decode_table_valid_symbols() {
    let mut freq = FrequencyTable::new();
    freq.count(b"hello, world!");
    let norm = normalize_frequencies(&freq, 8).unwrap();
    let spread = spread_symbols(&norm);
    let decode_table = build_decode_table(&norm, &spread);
    for entry in &decode_table {
        assert!(
            norm.freq[entry.symbol as usize] > 0,
            "decode table references absent symbol {}",
            entry.symbol
        );
    }
}

// --- BitWriter / BitReader ---

#[test]
fn test_bitwriter_reader_roundtrip() {
    let mut writer = BitWriter::new();
    writer.write_bits(0b101, 3);
    writer.write_bits(0b1100, 4);
    writer.write_bits(0b1, 1);
    writer.write_bits(0b11010, 5);
    let (data, total_bits) = writer.finish();
    assert_eq!(total_bits, 13);

    let mut reader = BitReader::new(&data);
    assert_eq!(reader.read_bits(3), 0b101);
    assert_eq!(reader.read_bits(4), 0b1100);
    assert_eq!(reader.read_bits(1), 0b1);
    assert_eq!(reader.read_bits(5), 0b11010);
}

#[test]
fn test_bitwriter_reader_large_values() {
    let mut writer = BitWriter::new();
    writer.write_bits(0xDEAD, 16);
    writer.write_bits(0xBEEF, 16);
    let (data, total_bits) = writer.finish();
    assert_eq!(total_bits, 32);

    let mut reader = BitReader::new(&data);
    assert_eq!(reader.read_bits(16), 0xDEAD);
    assert_eq!(reader.read_bits(16), 0xBEEF);
}

#[test]
fn test_bitwriter_zero_bits() {
    let mut writer = BitWriter::new();
    writer.write_bits(0, 0);
    let (data, total_bits) = writer.finish();
    assert_eq!(total_bits, 0);
    assert!(data.is_empty());
}

// --- Round-trip tests ---

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

// --- Accuracy log variants ---

#[test]
fn test_all_accuracy_logs() {
    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    for al in MIN_ACCURACY_LOG..=MAX_ACCURACY_LOG {
        let encoded = encode_with_accuracy(&input, al);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input, "failed at accuracy_log={}", al);
    }
}

// --- Error handling ---

#[test]
fn test_decode_too_short() {
    let result = decode(&[0u8; 10], 5);
    assert_eq!(result, Err(PzError::InvalidInput));
}

#[test]
fn test_decode_invalid_accuracy_log() {
    let mut bad = vec![0u8; HEADER_SIZE + 10];
    bad[0] = 15; // > MAX_ACCURACY_LOG
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
fn test_interleaved_repeated_byte() {
    let input = vec![b'a'; 100];
    let encoded = encode_interleaved(&input);
    let decoded = decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_interleaved_two_bytes() {
    let input = b"ab";
    let encoded = encode_interleaved(input);
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
fn test_interleaved_n_various() {
    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    for n in [1, 2, 4, 8] {
        let encoded = encode_interleaved_n(&input, n, DEFAULT_ACCURACY_LOG);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input, "failed at num_states={}", n);
    }
}

#[test]
fn test_interleaved_all_accuracy_logs() {
    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    for al in MIN_ACCURACY_LOG..=MAX_ACCURACY_LOG {
        let encoded = encode_interleaved_n(&input, 4, al);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input, "failed at accuracy_log={}", al);
    }
}

#[test]
fn test_interleaved_to_buf_too_small() {
    let input = b"hello, world!";
    let encoded = encode_interleaved(input);
    let mut buf = vec![0u8; 2];
    assert_eq!(
        decode_interleaved_to_buf(&encoded, input.len(), &mut buf),
        Err(PzError::BufferTooSmall)
    );
}

#[test]
fn test_interleaved_skewed() {
    let mut input = vec![0u8; 2000];
    input.push(1);
    input.push(2);
    let encoded = encode_interleaved(&input);
    let decoded = decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

// --- GPU encode table ---

#[test]
fn test_build_gpu_encode_table_matches_cpu() {
    // Verify that the packed GPU encode table produces the same
    // encode results as the CPU SymbolEncodeTable for all valid
    // (symbol, state) pairs.
    let mut freq = FrequencyTable::new();
    freq.count(b"abracadabra alakazam");
    let norm = normalize_frequencies(&freq, 7).unwrap();
    let table_size = 1usize << norm.accuracy_log;

    let fse_table = FseTable::from_normalized(&norm);
    let packed = build_gpu_encode_table(&norm);

    assert_eq!(packed.len(), 256 * table_size);

    // Check every present symbol at every state.
    for sym in 0..256usize {
        if norm.freq[sym] == 0 {
            continue;
        }
        for state in 0..table_size {
            let cpu_mapping = fse_table.encode_tables[sym].find(state);
            let gpu_entry = packed[sym * table_size + state];

            let gpu_compressed = (gpu_entry & 0xFFF) as u16;
            let gpu_bits = ((gpu_entry >> 12) & 0xF) as u8;
            let gpu_base = (gpu_entry >> 16) as u16;

            assert_eq!(
                gpu_compressed, cpu_mapping.compressed_state,
                "compressed_state mismatch: sym={sym}, state={state}"
            );
            assert_eq!(
                gpu_bits, cpu_mapping.bits,
                "bits mismatch: sym={sym}, state={state}"
            );
            assert_eq!(
                gpu_base, cpu_mapping.base,
                "base mismatch: sym={sym}, state={state}"
            );
        }
    }
}
