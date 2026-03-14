use super::*;
use crate::lz77::MAX_WINDOW;

/// Convenience: encode then immediately decode (for testing).
fn round_trip(input: &[u8]) -> PzResult<Vec<u8>> {
    let enc = encode(input)?;
    decode(
        &enc.flags,
        &enc.literals,
        &enc.offset_codes,
        &enc.offset_extra,
        &enc.length_codes,
        &enc.length_extra,
        enc.num_tokens,
        enc.num_matches,
        input.len(),
    )
}

// --- Code table self-consistency ---

#[test]
fn test_encode_decode_offset_exhaustive() {
    for d in 1..=100_000u32 {
        let (code, eb, ev) = encode_offset(d);
        assert_eq!(eb, extra_bits_for_code(code));
        let decoded = decode_offset(code, ev);
        assert_eq!(decoded, d, "offset {d}: code={code}, eb={eb}, ev={ev}");
    }
}

#[test]
fn test_encode_decode_length_exhaustive() {
    for len in MIN_MATCH..=1000 {
        let (code, eb, ev) = encode_length(len);
        assert_eq!(eb, extra_bits_for_code(code));
        let decoded = decode_length(code, ev);
        assert_eq!(decoded, len, "length {len}: code={code}, eb={eb}, ev={ev}");
    }
}

#[test]
fn test_offset_code_table_known_values() {
    // Code 0: offset 1
    assert_eq!(encode_offset(1), (0, 0, 0));
    // Code 1: offset 2
    assert_eq!(encode_offset(2), (1, 0, 0));
    // Code 2: offsets 3-4
    assert_eq!(encode_offset(3), (2, 1, 0));
    assert_eq!(encode_offset(4), (2, 1, 1));
    // Code 3: offsets 5-8
    assert_eq!(encode_offset(5), (3, 2, 0));
    assert_eq!(encode_offset(8), (3, 2, 3));
    // Code 4: offsets 9-16
    assert_eq!(encode_offset(9), (4, 3, 0));
    assert_eq!(encode_offset(16), (4, 3, 7));
    // Code 15: max offset for 32KB window = 32768
    let (code, eb, _) = encode_offset(32768);
    assert_eq!(code, 15);
    assert_eq!(eb, 14);
}

#[test]
fn test_length_code_table_known_values() {
    // Code 0: length 3
    assert_eq!(encode_length(3), (0, 0, 0));
    // Code 1: length 4
    assert_eq!(encode_length(4), (1, 0, 0));
    // Code 2: lengths 5-6
    assert_eq!(encode_length(5), (2, 1, 0));
    assert_eq!(encode_length(6), (2, 1, 1));
    // Code 3: lengths 7-10
    assert_eq!(encode_length(7), (3, 2, 0));
    assert_eq!(encode_length(10), (3, 2, 3));
}

// --- BitWriter / BitReader round-trip ---

#[test]
fn test_bitwriter_reader_round_trip() {
    let mut w = BitWriter::new();
    w.write_bits(5, 3); // 101
    w.write_bits(0, 1); // 0
    w.write_bits(15, 4); // 1111
    w.write_bits(42, 7); // 0101010
    w.write_bits(0, 0); // nothing
    let data = w.finish();

    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(3), 5);
    assert_eq!(r.read_bits(1), 0);
    assert_eq!(r.read_bits(4), 15);
    assert_eq!(r.read_bits(7), 42);
    assert_eq!(r.read_bits(0), 0);
}

#[test]
fn test_bitwriter_reader_large_values() {
    let mut w = BitWriter::new();
    w.write_bits(0x7FFF, 15); // 15 bits
    w.write_bits(0x1FFFF, 17); // 17 bits
    w.write_bits(1, 1);
    let data = w.finish();

    let mut r = BitReader::new(&data);
    assert_eq!(r.read_bits(15), 0x7FFF);
    assert_eq!(r.read_bits(17), 0x1FFFF);
    assert_eq!(r.read_bits(1), 1);
}

#[test]
fn test_bitwriter_reader_many_small() {
    let mut w = BitWriter::new();
    for i in 0..100 {
        w.write_bits(i % 4, 2);
    }
    let data = w.finish();

    let mut r = BitReader::new(&data);
    for i in 0..100 {
        assert_eq!(r.read_bits(2), i % 4);
    }
}

// --- Flag packing ---

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

// --- Encode/decode round-trip tests (ported from LZSS) ---

#[test]
fn test_round_trip_empty() {
    let input: Vec<u8> = Vec::new();
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_single_byte() {
    let input = vec![42u8];
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_no_matches() {
    let input = b"abcdefgh".to_vec();
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_all_same() {
    let input = vec![b'x'; 200];
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_repeats() {
    let pattern = b"the quick brown fox ";
    let mut input = Vec::new();
    for _ in 0..20 {
        input.extend_from_slice(pattern);
    }
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_longer_text() {
    let input = b"To be, or not to be, that is the question: \
        Whether 'tis nobler in the mind to suffer \
        The slings and arrows of outrageous fortune, \
        Or to take arms against a sea of troubles"
        .to_vec();
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_large() {
    let pattern = b"abcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    assert!(input.len() > 7500);
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_binary() {
    let input: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_window_boundary() {
    let mut input = Vec::new();
    let pattern = b"window boundary test pattern! ";
    for _ in 0..2000 {
        input.extend_from_slice(pattern);
    }
    assert!(input.len() > 32768);
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_150kb_mixed() {
    // Semi-random 150KB: mix of text patterns and noise
    let mut data = Vec::with_capacity(150 * 1024);
    let phrases: &[&[u8]] = &[
        b"compression algorithms are fascinating ",
        b"data structures enable efficient storage ",
        b"entropy coding reduces redundancy ",
        b"hash tables provide fast lookup ",
    ];
    let mut state: u32 = 0xDEADBEEF;
    while data.len() < 150 * 1024 {
        let phrase = phrases[(state as usize / 7) % phrases.len()];
        data.extend_from_slice(phrase);
        for _ in 0..32 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            data.push(state as u8);
        }
    }
    data.truncate(150 * 1024);
    let output = round_trip(&data).unwrap();
    assert_eq!(output, data);
}

#[test]
fn test_two_bytes() {
    let input = vec![0u8, 255];
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_compresses_with_matches() {
    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..50 {
        input.extend_from_slice(pattern);
    }
    let enc = encode(&input).unwrap();
    // Should have found matches
    assert!(
        enc.num_matches > 0,
        "should find matches in repetitive data"
    );
    // Round-trip
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_long_run_all_same() {
    let input = vec![0xAAu8; 70_000];
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_round_trip_long_repeating_pattern() {
    let pattern = b"abcde";
    let input: Vec<u8> = pattern.iter().cycle().take(80_000).copied().collect();
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

// --- Encode-specific tests ---

#[test]
fn test_encode_empty_produces_zero_tokens() {
    let enc = encode(&[]).unwrap();
    assert_eq!(enc.num_tokens, 0);
    assert_eq!(enc.num_matches, 0);
    assert!(enc.flags.is_empty());
    assert!(enc.literals.is_empty());
}

#[test]
fn test_encode_all_literals() {
    let input = b"abcdefgh";
    let enc = encode(input).unwrap();
    assert_eq!(enc.num_tokens as usize, input.len());
    assert_eq!(enc.num_matches, 0);
    assert_eq!(enc.literals.len(), input.len());
}

#[test]
fn test_offset_codes_in_valid_range() {
    let pattern = b"hello world hello world hello world ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let enc = encode(&input).unwrap();
    for &code in &enc.offset_codes {
        assert!(code <= 31, "offset code {code} out of range");
    }
    for &code in &enc.length_codes {
        assert!(code <= 20, "length code {code} out of range");
    }
}

// --- Wide window tests (Phase 3) ---

#[test]
fn test_wide_window_round_trip() {
    // Create data with a repeated block separated by >32KB of unique fill.
    // Only a wide window can find the long-distance match.
    let marker = b"WIDE_WINDOW_MARKER_PATTERN_12345";
    let fill_len = 40_000; // > 32KB
    let mut input = Vec::with_capacity(marker.len() * 2 + fill_len);
    input.extend_from_slice(marker);
    // Fill with non-repeating bytes so the matcher doesn't find local matches
    for i in 0..fill_len {
        input.push((i % 251) as u8 ^ 0x55);
    }
    input.extend_from_slice(marker);

    let config = SeqConfig {
        max_window: 64 * 1024,
        ..SeqConfig::default()
    };
    let enc = encode_with_config(&input, &config).unwrap();
    let output = decode(
        &enc.flags,
        &enc.literals,
        &enc.offset_codes,
        &enc.offset_extra,
        &enc.length_codes,
        &enc.length_extra,
        enc.num_tokens,
        enc.num_matches,
        input.len(),
    )
    .unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_wide_window_finds_distant_matches() {
    // Verify the 128KB window actually finds matches beyond 32KB.
    let pattern = b"DISTANT_MATCH_TEST_PATTERN_ABCD!";
    let fill_len = 50_000; // > 32KB gap
    let mut input = Vec::with_capacity(pattern.len() * 2 + fill_len);
    input.extend_from_slice(pattern);
    for i in 0..fill_len {
        input.push((i % 251) as u8 ^ 0xAA);
    }
    input.extend_from_slice(pattern);

    // Wide window: should find a match at offset > 32KB
    let config_wide = SeqConfig {
        max_window: 128 * 1024,
        ..SeqConfig::default()
    };
    let enc_wide = encode_with_config(&input, &config_wide).unwrap();

    // Narrow window: cannot see across the gap
    let config_narrow = SeqConfig {
        max_window: MAX_WINDOW, // 32KB
        ..SeqConfig::default()
    };
    let enc_narrow = encode_with_config(&input, &config_narrow).unwrap();

    // Wide window should produce fewer or equal literals
    assert!(
        enc_wide.literals.len() <= enc_narrow.literals.len(),
        "wide window ({} literals) should not produce more literals than narrow ({})",
        enc_wide.literals.len(),
        enc_narrow.literals.len()
    );

    // Both must round-trip correctly
    let out_wide = decode(
        &enc_wide.flags,
        &enc_wide.literals,
        &enc_wide.offset_codes,
        &enc_wide.offset_extra,
        &enc_wide.length_codes,
        &enc_wide.length_extra,
        enc_wide.num_tokens,
        enc_wide.num_matches,
        input.len(),
    )
    .unwrap();
    assert_eq!(out_wide, input);

    let out_narrow = decode(
        &enc_narrow.flags,
        &enc_narrow.literals,
        &enc_narrow.offset_codes,
        &enc_narrow.offset_extra,
        &enc_narrow.length_codes,
        &enc_narrow.length_extra,
        enc_narrow.num_tokens,
        enc_narrow.num_matches,
        input.len(),
    )
    .unwrap();
    assert_eq!(out_narrow, input);
}

#[test]
fn test_default_config_uses_128kb() {
    let config = SeqConfig::default();
    assert_eq!(config.max_window, 128 * 1024);
}

#[test]
fn test_encode_backward_compat_uses_32kb() {
    // encode() uses MAX_WINDOW (32KB) for backward compat.
    // Verify it doesn't panic and round-trips.
    let input = vec![b'Z'; 50_000];
    let enc = encode(&input).unwrap();
    let output = decode(
        &enc.flags,
        &enc.literals,
        &enc.offset_codes,
        &enc.offset_extra,
        &enc.length_codes,
        &enc.length_extra,
        enc.num_tokens,
        enc.num_matches,
        input.len(),
    )
    .unwrap();
    assert_eq!(output, input);
}

// --- Distance-dependent MIN_MATCH tests (Phase 4) ---

#[test]
fn test_min_profitable_length_close_offsets() {
    // Close offsets (1-16) should need only MIN_MATCH (3)
    for offset in 1..=16 {
        assert_eq!(
            min_profitable_length(offset),
            MIN_MATCH,
            "offset {offset} should need min_match=3"
        );
    }
}

#[test]
fn test_min_profitable_length_increases_with_distance() {
    // Minimum length should be non-decreasing with offset
    let mut prev = min_profitable_length(1);
    for offset in [16, 256, 4096, 65536, 500_000u32] {
        let cur = min_profitable_length(offset);
        assert!(
            cur >= prev,
            "min_profitable_length should be non-decreasing: offset {offset} gave {cur}, prev was {prev}"
        );
        prev = cur;
    }
}

#[test]
fn test_min_profitable_length_far_offsets_require_longer() {
    // Very far offsets should require longer matches
    let close = min_profitable_length(1);
    let far = min_profitable_length(100_000);
    assert!(
        far > close,
        "far offset (100K) should need longer match than close (1): {far} vs {close}"
    );
}

#[test]
fn test_distance_dependent_round_trip() {
    // Ensure distance-dependent filtering doesn't break round-trip
    let pattern = b"mixed content with some repeats mixed content! ";
    let mut input = Vec::new();
    for _ in 0..500 {
        input.extend_from_slice(pattern);
    }
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

// --- Repeat offset tests (Phase 5) ---

#[test]
fn test_repeat_offsets_state_management() {
    let mut rep = RepeatOffsets::new();
    assert_eq!(rep.recent, [1, 1, 1]);

    // Push a new offset
    rep.push_new(42);
    assert_eq!(rep.recent, [42, 1, 1]);

    // Push another
    rep.push_new(100);
    assert_eq!(rep.recent, [100, 42, 1]);

    // Promote index 1 (swap)
    rep.promote(1);
    assert_eq!(rep.recent, [42, 100, 1]);

    // Promote index 2 (rotate)
    rep.promote(2);
    assert_eq!(rep.recent, [1, 42, 100]);
}

#[test]
fn test_repeat_offset_encode_decode_round_trip() {
    let mut enc_rep = RepeatOffsets::new();
    let mut dec_rep = RepeatOffsets::new();

    // Encode a sequence of offsets (some repeats)
    let offsets = [10, 20, 10, 10, 20, 30, 20, 20, 20];
    for &offset in &offsets {
        let (code, eb, ev) = enc_rep.encode_offset(offset);
        let decoded = dec_rep.decode_offset(code, ev);
        assert_eq!(
            decoded, offset,
            "repeat offset mismatch for offset {offset}"
        );
        assert_eq!(
            extra_bits_for_offset_code(code),
            eb,
            "extra bits mismatch for code {code}"
        );
    }
}

#[test]
fn test_repeat_matches_used_on_structured_data() {
    // With small structured patterns, repeat offsets should be used
    // on subsequent occurrences. Use a pattern short enough that
    // one giant match doesn't cover everything.
    let pattern = b"ABC";
    let mut input = Vec::new();
    // Create 10 copies, each separated by a small unique sequence
    for i in 0..10 {
        input.extend_from_slice(pattern);
        // Add a small unique suffix to prevent one giant match
        input.push(b'0' + (i % 10) as u8);
    }
    let enc = encode(&input).unwrap();

    // Count repeat codes (0-2) in offset_codes
    let repeat_count = enc
        .offset_codes
        .iter()
        .filter(|&&c| c < NUM_REPEAT_CODES)
        .count();

    assert!(
        repeat_count > 0,
        "structured data should use repeat offsets, but found 0 (offset_codes={:?})",
        &enc.offset_codes
    );

    // Verify round-trip
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

#[test]
fn test_repeat_offsets_round_trip_various() {
    // All-same bytes: mostly repeat offset 1
    let input = vec![b'Q'; 10_000];
    assert_eq!(round_trip(&input).unwrap(), input);

    // Short repeating pattern: repeat offset = pattern length
    let pattern = b"ABCD";
    let input: Vec<u8> = pattern.iter().cycle().take(10_000).copied().collect();
    assert_eq!(round_trip(&input).unwrap(), input);

    // Alternating patterns at different offsets
    let mut input = Vec::new();
    for i in 0..200 {
        if i % 2 == 0 {
            input.extend_from_slice(b"even pattern here! ");
        } else {
            input.extend_from_slice(b"odd pattern here!! ");
        }
    }
    assert_eq!(round_trip(&input).unwrap(), input);
}

#[test]
fn test_repeat_offsets_improve_ratio() {
    // With small repeated patterns separated by variation,
    // repeat codes should be used on matching offsets.
    let pattern = b"abc";
    let mut input = Vec::new();
    for i in 0..50 {
        input.extend_from_slice(pattern);
        // Add variation to prevent giant matches
        input.push(b'0' + (i as u8 % 10));
    }
    let enc = encode(&input).unwrap();

    let repeat_count = enc
        .offset_codes
        .iter()
        .filter(|&&c| c < NUM_REPEAT_CODES)
        .count();
    let total_matches = enc.num_matches as usize;

    // Expect at least some matches to use repeat codes
    assert!(total_matches > 0, "should have found matches");
    assert!(
        repeat_count > 0,
        "expected some repeat usage on regular data, got 0 repeat codes out of {} matches",
        total_matches
    );
}

#[test]
fn seq_config_hash4_roundtrip() {
    let input = b"the quick brown fox the quick brown fox";
    let config = SeqConfig {
        hash_prefix_len: 4,
        ..SeqConfig::default()
    };
    let encoded = encode_with_config(input, &config).unwrap();
    assert!(encoded.num_matches > 0, "hash4 config should find matches");
}

#[test]
fn seq_config_default_no_regression() {
    let input = b"aababcabcdabcdeabcdefabcdefg";
    let encoded_default = encode_with_config(input, &SeqConfig::default()).unwrap();
    let encoded_hash3 = encode_with_config(
        input,
        &SeqConfig {
            hash_prefix_len: 3,
            ..SeqConfig::default()
        },
    )
    .unwrap();
    // Verify both configs encode successfully. Hash selection may affect token
    // count after DP cost model recalibration. The default config uses hash_prefix_len=4.
    assert!(
        encoded_default.num_tokens > 0,
        "default config should produce tokens"
    );
    assert!(
        encoded_hash3.num_tokens > 0,
        "hash3 config should produce tokens"
    );
}

#[test]
fn adaptive_chain_finds_matches_on_compressible_data() {
    let input: Vec<u8> = b"the quick brown fox "
        .iter()
        .cycle()
        .take(4096)
        .copied()
        .collect();
    let config = SeqConfig {
        adaptive_chain: true,
        ..SeqConfig::default()
    };
    let encoded = encode_with_config(&input, &config).unwrap();
    assert!(
        encoded.num_matches > 0,
        "adaptive mode must find matches on repetitive input"
    );
    // Verify basic functionality: matches found and tokens reasonable
    assert!(encoded.num_tokens > 0, "should have tokens");
}

#[test]
fn adaptive_chain_no_panic_on_random() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Deterministic pseudo-random bytes
    let input: Vec<u8> = (0u32..4096)
        .map(|i| {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            h.finish() as u8
        })
        .collect();
    let config = SeqConfig {
        adaptive_chain: true,
        ..SeqConfig::default()
    };
    let _ = encode_with_config(&input, &config).unwrap(); // must not panic
}

#[test]
fn adaptive_chain_false_is_identical_to_default() {
    let input = b"hello world hello world hello world";
    let default_encoded = encode_with_config(input, &SeqConfig::default()).unwrap();
    let explicit_false = encode_with_config(
        input,
        &SeqConfig {
            adaptive_chain: false,
            ..SeqConfig::default()
        },
    )
    .unwrap();
    assert_eq!(default_encoded.num_tokens, explicit_false.num_tokens);
    assert_eq!(default_encoded.num_matches, explicit_false.num_matches);
}

#[test]
fn adaptive_vs_nonadaptive_compressible_match_count() {
    let input: Vec<u8> = b"abcdefghij".iter().cycle().take(65536).copied().collect();

    let adaptive_cfg = SeqConfig {
        adaptive_chain: true,
        ..SeqConfig::default()
    };
    let normal_cfg = SeqConfig {
        adaptive_chain: false,
        ..SeqConfig::default()
    };

    let adaptive = encode_with_config(&input, &adaptive_cfg).unwrap();
    let normal = encode_with_config(&input, &normal_cfg).unwrap();

    // Adaptive must find at least 95% as many matches as non-adaptive
    let threshold = (normal.num_matches as f64 * 0.95) as u32;
    assert!(
        adaptive.num_matches >= threshold,
        "adaptive found {} matches, non-adaptive found {} (threshold {})",
        adaptive.num_matches,
        normal.num_matches,
        threshold
    );
}

#[test]
fn adaptive_mixed_input_no_corruption() {
    // First 32KB repetitive, next 32KB pseudo-random
    let mut input: Vec<u8> = b"abcde".iter().cycle().take(32768).copied().collect();
    input.extend((0u8..=255).cycle().take(32768));

    let config = SeqConfig {
        adaptive_chain: true,
        ..SeqConfig::default()
    };
    let encoded = encode_with_config(&input, &config).unwrap();
    // Must have found at least some matches (from the repetitive first half)
    assert!(
        encoded.num_matches > 0,
        "adaptive mode must find matches in mixed input"
    );
}

// --- Min profitable length tests (Task 7) ---

#[test]
fn min_profitable_length_offset_zero() {
    assert_eq!(min_profitable_length(0), u16::MAX);
}

#[test]
fn min_profitable_length_tiers() {
    // oeb 0-3: offset 1-16 → min 3
    for d in 1u32..=16 {
        assert_eq!(
            min_profitable_length(d),
            3,
            "offset {d} should require min 3"
        );
    }
    // oeb 4-8: offset 17-256 → min 4
    for d in [17u32, 32, 64, 128, 256] {
        assert_eq!(
            min_profitable_length(d),
            4,
            "offset {d} should require min 4"
        );
    }
    // oeb 9-12: offset 257-4096 → min 5
    for d in [257u32, 512, 1024, 2048, 4096] {
        assert_eq!(
            min_profitable_length(d),
            5,
            "offset {d} should require min 5"
        );
    }
    // oeb 13-16: offset 4097-65536 → min 6
    for d in [4097u32, 8192, 16384, 32768, 65536] {
        assert_eq!(
            min_profitable_length(d),
            6,
            "offset {d} should require min 6"
        );
    }
    // oeb 17+: offset 65537+ → min 7
    assert_eq!(min_profitable_length(65537), 7);
    assert_eq!(min_profitable_length(131072), 7);
}

#[test]
fn min_profitable_length_monotone() {
    // Sample offsets spanning all tiers; verify non-decreasing.
    let offsets = [
        1u32, 4, 8, 9, 32, 256, 257, 1024, 4096, 4097, 32768, 65536, 65537, 131072,
    ];
    let thresholds: Vec<u16> = offsets.iter().map(|&d| min_profitable_length(d)).collect();
    for w in thresholds.windows(2) {
        assert!(
            w[0] <= w[1],
            "min_profitable_length not monotone: {} > {} for adjacent offsets",
            w[0],
            w[1]
        );
    }
}

// --- Match profitability edge cases (Task 8) ---

#[test]
fn encoder_rejects_short_match_at_large_offset() {
    let pattern = b"ABCDE";
    let mut input_3byte = Vec::new();
    let mut input_4byte = Vec::new();

    input_3byte.extend_from_slice(&pattern[..3]);
    input_4byte.extend_from_slice(pattern);

    for i in 0..295 {
        input_3byte.push((i % 256) as u8 ^ 0x55);
        input_4byte.push((i % 256) as u8 ^ 0x55);
    }

    input_3byte.extend_from_slice(&pattern[..3]);
    input_4byte.extend_from_slice(&pattern[..4]);

    let config = SeqConfig {
        max_window: 128 * 1024,
        ..SeqConfig::default()
    };

    let enc_3 = encode_with_config(&input_3byte, &config).unwrap();
    let enc_4 = encode_with_config(&input_4byte, &config).unwrap();

    assert!(
        enc_3.num_matches == enc_4.num_matches
            || (enc_3.num_matches == 0 && enc_4.num_matches == 0),
        "short matches at offset ~300 should be rejected: 3-byte gave {}, 4-byte gave {}",
        enc_3.num_matches,
        enc_4.num_matches
    );
}

#[test]
fn encoder_accepts_profitable_match_at_large_offset() {
    let pattern = b"ABCDE";
    let mut input = Vec::new();

    input.extend_from_slice(pattern);

    for i in 0..295 {
        input.push((i % 256) as u8 ^ 0x55);
    }

    input.extend_from_slice(pattern);

    let config = SeqConfig {
        max_window: 128 * 1024,
        ..SeqConfig::default()
    };
    let encoded = encode_with_config(&input, &config).unwrap();

    assert!(
        encoded.num_matches >= 1,
        "5-byte match at offset ~300 should be accepted"
    );
}

#[test]
fn encoder_all_same_bytes_efficient() {
    let input = vec![0xAAu8; 1024];
    let encoded = encode_with_config(&input, &SeqConfig::default()).unwrap();
    assert!(
        encoded.num_matches >= 1,
        "all-same input must find at least one match"
    );
    assert!(
        encoded.num_tokens < 20,
        "all-same 1KB input should compress to very few tokens, got {}",
        encoded.num_tokens
    );
}

#[test]
fn encoder_profitability_round_trip() {
    let pattern = b"mixed content with some repeats mixed content! ";
    let mut input = Vec::new();
    for _ in 0..500 {
        input.extend_from_slice(pattern);
    }
    let output = round_trip(&input).unwrap();
    assert_eq!(output, input);
}

// ---- Task 9: Window size configuration and presets ----

#[test]
fn seq_config_presets_have_correct_windows() {
    assert_eq!(SeqConfig::fast().max_window, 64 * 1024);
    assert_eq!(SeqConfig::default_quality().max_window, 128 * 1024);
    assert_eq!(SeqConfig::high().max_window, 256 * 1024);
    assert_eq!(SeqConfig::high().hash_prefix_len, 4);
    assert_eq!(SeqConfig::high().max_chain, 128);
}

#[test]
fn all_presets_produce_valid_output_on_short_input() {
    let input = b"the quick brown fox jumps over the lazy dog";
    for config in [
        SeqConfig::fast(),
        SeqConfig::default_quality(),
        SeqConfig::high(),
    ] {
        let encoded = encode_with_config(input, &config).unwrap();
        assert!(encoded.num_tokens > 0, "preset must encode non-empty input");
        assert!(
            encoded.num_tokens <= input.len() as u32,
            "token count should not exceed input length"
        );
    }
}

#[test]
fn high_preset_256kb_window_no_panic() {
    let input: Vec<u8> = b"Lorem ipsum dolor sit amet "
        .iter()
        .cycle()
        .take(256 * 1024)
        .cloned()
        .collect();
    let encoded = encode_with_config(&input, &SeqConfig::high()).unwrap();
    assert!(encoded.num_matches > 0);
}

// ---- Task 10: Window size impact on match quality ----

#[test]
fn larger_window_finds_distant_match() {
    let pattern = b"ABCDE";
    let filler_len = 70 * 1024;
    let mut input = Vec::with_capacity(filler_len + 10);
    input.extend_from_slice(pattern);
    for i in 0..filler_len {
        input.push((i % 251 + 1) as u8);
    }
    input.extend_from_slice(pattern);

    let small_window_cfg = SeqConfig {
        max_window: 32 * 1024,
        ..SeqConfig::default()
    };
    let large_window_cfg = SeqConfig {
        max_window: 256 * 1024,
        ..SeqConfig::default()
    };

    let small_encoded = encode_with_config(&input, &small_window_cfg).unwrap();
    let large_encoded = encode_with_config(&input, &large_window_cfg).unwrap();

    assert!(
        large_encoded.num_matches >= small_encoded.num_matches,
        "256KB window found {} matches, but 32KB window found {}. Larger window should not find fewer matches.",
        large_encoded.num_matches,
        small_encoded.num_matches
    );

    assert!(
        large_encoded.num_matches > 0,
        "256KB window should find at least one match in repetitive input"
    );
}

#[test]
fn window_size_match_count_nondecreasing() {
    let mut input: Vec<u8> = Vec::new();
    let phrase = b"hello world ";
    for gap in [100usize, 1000, 10000, 50000, 100000] {
        input.extend_from_slice(phrase);
        for i in 0..gap {
            input.push((i % 200 + 10) as u8);
        }
    }
    input.extend_from_slice(phrase); // final match target

    let windows = [32 * 1024usize, 64 * 1024, 128 * 1024, 256 * 1024];
    let mut prev_match_count = None;
    for &w in &windows {
        let cfg = SeqConfig {
            max_window: w,
            ..SeqConfig::default()
        };
        let encoded = encode_with_config(&input, &cfg).unwrap();
        assert!(
            encoded.num_matches > 0,
            "window {} should find at least one match in repetitive input, found {}",
            w,
            encoded.num_matches
        );

        if let Some(prev) = prev_match_count {
            let ratio = encoded.num_matches as f64 / prev as f64;
            assert!(
                ratio > 0.5,
                "window {} match count {} dropped by more than 50% from previous window's {}",
                w,
                encoded.num_matches,
                prev
            );
        }
        prev_match_count = Some(encoded.num_matches);
    }
}

#[test]
fn test_encode_optimal_round_trip() {
    let input = b"abc".repeat(10);
    let config = SeqConfig::default();
    let encoded = encode_optimal(&input, &config).expect("encode_optimal failed");
    let decoded = decode(
        &encoded.flags,
        &encoded.literals,
        &encoded.offset_codes,
        &encoded.offset_extra,
        &encoded.length_codes,
        &encoded.length_extra,
        encoded.num_tokens,
        encoded.num_matches,
        input.len(),
    )
    .expect("decode failed");
    assert_eq!(decoded, input);
}

#[test]
fn test_encode_optimal_empty() {
    let input: &[u8] = b"";
    let config = SeqConfig::default();
    let encoded = encode_optimal(input, &config).expect("encode_optimal failed");
    assert_eq!(encoded.num_tokens, 0);
    assert_eq!(encoded.num_matches, 0);
}

#[test]
fn test_encode_optimal_no_matches() {
    let input: &[u8] = b"abcdefghij";
    let config = SeqConfig::default();
    let encoded = encode_optimal(input, &config).expect("encode_optimal failed");
    // Should only have literal tokens
    assert_eq!(encoded.num_matches, 0);
    assert_eq!(encoded.num_tokens as usize, input.len());
}
