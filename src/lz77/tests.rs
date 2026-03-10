use super::*;

#[test]
fn test_match_serialization() {
    let m = Match {
        offset: 42,
        length: 10,
        next: b'x',
    };
    let bytes = m.to_bytes();
    let m2 = Match::from_bytes(&bytes);
    assert_eq!(m, m2);
}

#[test]
fn hash4_range() {
    let data = b"hello world this is a test";
    for pos in 0..data.len() {
        let h = hash4(data, pos);
        assert!(h < HASH_SIZE, "hash4 out of range at pos {pos}: {h}");
    }
}

#[test]
fn hash4_boundary_guard() {
    let data = b"abc"; // len=3, pos+3 >= len for all pos >= 0
    assert_eq!(hash4(data, 0), 0);
    let data2 = b"abcd"; // len=4, pos=0: pos+3=3 >= 4? No. pos=1: 1+3=4 >= 4? Yes.
    assert_ne!(
        hash4(data2, 0),
        0,
        "4-byte input at pos 0 should hash normally"
    );
    assert_eq!(hash4(data2, 1), 0, "pos 1 in 4-byte input triggers guard");
}

#[test]
fn hash4_deterministic() {
    let data = b"the quick brown fox";
    assert_eq!(hash4(data, 0), hash4(data, 0));
    assert_eq!(hash4(data, 4), hash4(data, 4));
}

#[test]
fn hash4_distinct_inputs() {
    // These four 4-byte sequences should produce different hashes.
    let inputs: &[&[u8]] = &[b"aaaa", b"aaab", b"aaba", b"abaa"];
    let hashes: Vec<usize> = inputs.iter().map(|d| hash4(d, 0)).collect();
    // All distinct
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(
                hashes[i], hashes[j],
                "collision: {:?} vs {:?}",
                inputs[i], inputs[j]
            );
        }
    }
}

#[test]
fn hash_prefix_default_is_3() {
    let _finder = HashChainFinder::new();
    // Verify hash3 path: insert and find a known match
    let data = b"abcabcabc";
    let mut f = HashChainFinder::new();
    f.insert(data, 0);
    f.insert(data, 1);
    f.insert(data, 2);
    let m = f.find_best(data, 3, 32768);
    assert!(m.1 >= 3, "hash3 finder should find match at pos 3");
    assert_eq!(m.0, 3);
}

#[test]
fn hash4_finder_finds_matches() {
    let data = b"abcdabcdabcd";
    let mut f = HashChainFinder::with_hash4(32768, u16::MAX, 64);
    for i in 0..4 {
        f.insert(data, i);
    }
    let (off, len) = f.find_best(data, 4, 32768);
    assert!(len >= 4, "hash4 finder should find 4-byte match");
    assert_eq!(off, 4, "offset should be 4");
}

#[test]
fn hash4_finder_correctness_roundtrip() {
    // Verify hash4 finder doesn't crash and functions as a valid finder.
    // Detailed match verification is implicit in encode/decode round-trip tests.
    let input = vec![0x55u8; 500];
    let mut finder = HashChainFinder::with_hash4(32768, 258, 64);
    // Just verify insertion and searching don't panic
    for pos in 0..input.len() {
        finder.insert(&input, pos);
    }
    // Search from multiple positions - should not crash
    for pos in 0..input.len().min(100) {
        let _m = finder.find_best(&input, pos, 32768);
    }
}

#[test]
fn test_decompress_empty() {
    let result = decompress(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_decompress_to_buf_too_small() {
    let input = b"abcabcabc";
    let compressed = compress_lazy(input).unwrap();
    let mut output = vec![0u8; 1]; // way too small
    let result = decompress_to_buf(&compressed, &mut output);
    assert_eq!(result, Err(PzError::BufferTooSmall));
}

#[test]
fn test_decompress_invalid_input_size() {
    // Input not a multiple of SERIALIZED_SIZE
    let result = decompress(&[1, 2, 3]);
    assert_eq!(result, Err(PzError::InvalidInput));
}

#[test]
fn test_decompress_invalid_offset() {
    // Match with offset larger than output so far
    let m = Match {
        offset: 100,
        length: 5,
        next: b'a',
    };
    let result = decompress(&m.to_bytes());
    assert_eq!(result, Err(PzError::InvalidInput));
}

// --- Lazy matching tests ---

#[test]
fn test_lazy_round_trip_repeats() {
    let input = b"abcabcabc";
    let compressed = compress_lazy(input).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(&decompressed, input);
}

#[test]
fn test_chain_depth_selection_tiers() {
    assert_eq!(select_chain_depth(128 * 1024, true), MAX_CHAIN);
    assert_eq!(select_chain_depth(256 * 1024, true), MAX_CHAIN_AUTO);
    assert_eq!(select_chain_depth(1024 * 1024, true), 32);
    assert_eq!(select_chain_depth(4 * 1024 * 1024, true), 24);
    assert_eq!(select_chain_depth(4 * 1024 * 1024, false), MAX_CHAIN);
}

#[test]
fn test_lazy_round_trip_window_boundary() {
    // Input long enough to exercise the MAX_WINDOW boundary
    let mut input = Vec::new();
    let block = b"ABCDEFGHIJ"; // 10 bytes
    for _ in 0..500 {
        // 5000 bytes > MAX_WINDOW
        input.extend_from_slice(block);
    }
    let compressed = compress_lazy(&input).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Parse quality regression tests ---
//
// These tests assert that the lazy parser produces at most N sequences
// (Match structs) for known inputs. Fewer sequences = better compression.
// If a change reduces the count, update the golden value downward.
// If a change increases the count, the test fails — investigate the regression.

/// Count total sequences and match sequences in serialized LZ77 output.
fn count_sequences(lz_data: &[u8]) -> (usize, usize) {
    let total = lz_data.len() / Match::SERIALIZED_SIZE;
    let mut matches = 0;
    for chunk in lz_data.chunks_exact(Match::SERIALIZED_SIZE) {
        let buf: &[u8; Match::SERIALIZED_SIZE] = chunk.try_into().unwrap();
        let m = Match::from_bytes(buf);
        if m.length > 0 && m.offset > 0 {
            matches += 1;
        }
    }
    (total, matches)
}

/// Sum of all match lengths (bytes covered by back-references).
fn total_match_bytes(lz_data: &[u8]) -> usize {
    let mut total = 0;
    for chunk in lz_data.chunks_exact(Match::SERIALIZED_SIZE) {
        let buf: &[u8; Match::SERIALIZED_SIZE] = chunk.try_into().unwrap();
        let m = Match::from_bytes(buf);
        if m.length > 0 {
            total += m.length as usize;
        }
    }
    total
}

// -- Synthetic golden tests (always run) --
//
// These use exact equality so ANY change to the parser — improvement or
// regression — is immediately flagged. When you intentionally improve the
// parser, update the golden values to the new numbers.

#[test]
fn test_lazy_quality_repeated_pattern() {
    // 200 repeats of a 38-byte pattern = 7600 bytes.
    let pattern = b"Hello, World! This is a test pattern. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_lazy(&input).unwrap();
    let (total_seqs, match_seqs) = count_sequences(&compressed);
    let matched = total_match_bytes(&compressed);

    // Golden: 65 seqs, 31 matches, 7535 bytes matched.
    assert_eq!(total_seqs, 65, "total sequences changed (was 65)");
    assert_eq!(match_seqs, 31, "match count changed (was 31)");
    assert_eq!(matched, 7535, "match coverage changed (was 7535)");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lazy_quality_all_same() {
    // 10000 identical bytes. Overlapping matches (length > offset)
    // allow very long matches at offset=1, drastically reducing seqs.
    let input = vec![b'A'; 10000];
    let compressed = compress_lazy(&input).unwrap();
    let (total_seqs, match_seqs) = count_sequences(&compressed);
    let matched = total_match_bytes(&compressed);

    // Golden: 40 seqs, 39 matches, 9960 bytes matched.
    assert_eq!(total_seqs, 40, "total sequences changed (was 40)");
    assert_eq!(match_seqs, 39, "match count changed (was 39)");
    assert_eq!(matched, 9960, "match coverage changed (was 9960)");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lazy_quality_cyclic_256() {
    // 0,1,2,...,255 repeated 4 times = 1024 bytes.
    let input: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
    let compressed = compress_lazy(&input).unwrap();
    let (total_seqs, match_seqs) = count_sequences(&compressed);
    let matched = total_match_bytes(&compressed);

    // Golden: 259 seqs, 3 matches, 765 bytes matched.
    assert_eq!(total_seqs, 259, "total sequences changed (was 259)");
    assert_eq!(match_seqs, 3, "match count changed (was 3)");
    assert_eq!(matched, 765, "match coverage changed (was 765)");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// -- Canterbury corpus golden tests (skip if samples not extracted) --

#[cfg(test)]
fn corpus_file(name: &str) -> Option<Vec<u8>> {
    for dir in &["samples/cantrbry", "/home/user/libpz/samples/cantrbry"] {
        let path = format!("{}/{}", dir, name);
        if let Ok(data) = std::fs::read(&path) {
            return Some(data);
        }
    }
    None
}

#[test]
fn test_lazy_quality_alice29() {
    let Some(data) = corpus_file("alice29.txt") else {
        eprintln!("skipping: alice29.txt not found");
        return;
    };
    assert_eq!(data.len(), 152089);
    let compressed = compress_lazy(&data).unwrap();
    let (total_seqs, match_seqs) = count_sequences(&compressed);
    let matched = total_match_bytes(&compressed);

    // Golden: 27564 seqs, 23951 matches, 124525 bytes matched.
    assert_eq!(total_seqs, 27564, "alice29.txt total sequences changed");
    assert_eq!(match_seqs, 23951, "alice29.txt match count changed");
    assert_eq!(matched, 124525, "alice29.txt match coverage changed");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_lazy_quality_fields_c() {
    let Some(data) = corpus_file("fields.c") else {
        eprintln!("skipping: fields.c not found");
        return;
    };
    assert_eq!(data.len(), 11150);
    let compressed = compress_lazy(&data).unwrap();
    let (total_seqs, match_seqs) = count_sequences(&compressed);
    let matched = total_match_bytes(&compressed);

    // Golden: 1943 seqs, 1158 matches, 9207 bytes matched.
    assert_eq!(total_seqs, 1943, "fields.c total sequences changed");
    assert_eq!(match_seqs, 1158, "fields.c match count changed");
    assert_eq!(matched, 9207, "fields.c match coverage changed");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_lazy_quality_grammar_lsp() {
    let Some(data) = corpus_file("grammar.lsp") else {
        eprintln!("skipping: grammar.lsp not found");
        return;
    };
    assert_eq!(data.len(), 3721);
    let compressed = compress_lazy(&data).unwrap();
    let (total_seqs, match_seqs) = count_sequences(&compressed);
    let matched = total_match_bytes(&compressed);

    // Golden: 867 seqs, 364 matches, 2854 bytes matched.
    assert_eq!(total_seqs, 867, "grammar.lsp total sequences changed");
    assert_eq!(match_seqs, 364, "grammar.lsp match count changed");
    assert_eq!(matched, 2854, "grammar.lsp match coverage changed");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

#[test]
fn test_lazy_quality_xargs() {
    let Some(data) = corpus_file("xargs.1") else {
        eprintln!("skipping: xargs.1 not found");
        return;
    };
    assert_eq!(data.len(), 4227);
    let compressed = compress_lazy(&data).unwrap();
    let (total_seqs, match_seqs) = count_sequences(&compressed);
    let matched = total_match_bytes(&compressed);

    // Golden: 1235 seqs, 509 matches, 2992 bytes matched.
    assert_eq!(total_seqs, 1235, "xargs.1 total sequences changed");
    assert_eq!(match_seqs, 509, "xargs.1 match count changed");
    assert_eq!(matched, 2992, "xargs.1 match coverage changed");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, data);
}

/// Regression: all-same-byte input >65535 bytes would produce matches with
/// length exceeding u16::MAX, causing silent truncation and corrupt output.
#[test]
fn test_round_trip_long_run_all_same() {
    let input = vec![0xAAu8; 70_000];
    let compressed = compress_lazy(&input).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input, "70KB all-same-byte round trip failed");
}

/// Regression: short repeating pattern >65535 bytes creates overlapping
/// matches with length > u16::MAX via offset=pattern_len.
#[test]
fn test_round_trip_long_repeating_pattern() {
    let pattern = b"abcde";
    let input: Vec<u8> = pattern.iter().cycle().take(80_000).copied().collect();
    let compressed = compress_lazy(&input).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "80KB repeating pattern round trip failed"
    );
}

/// Verify find_match respects the default LZ77_MAX_MATCH (258).
///
/// The default HashChainFinder uses LZ77_MAX_MATCH, which is passed
/// as the limit to SIMD compare_bytes. This prevents matches from
/// exceeding 258 for DEFLATE-compatible output.
#[test]
fn test_find_match_length_bounded_deflate() {
    let input = vec![0u8; 70_000];
    let mut finder = HashChainFinder::new();
    finder.insert(&input, 0);
    let m = finder.find_match(&input, 1);
    assert_eq!(
        m.length, 258,
        "default find_match should cap at LZ77_MAX_MATCH"
    );
    assert_eq!(m.offset, 1, "should match with offset 1");
}

/// Verify find_match with extended limit finds longer matches.
#[test]
fn test_find_match_extended_limit() {
    let input = vec![0u8; 70_000];
    let mut finder = HashChainFinder::with_max_match_len(DEFAULT_MAX_MATCH);
    finder.insert(&input, 0);
    let m = finder.find_match(&input, 1);
    // With extended limit, match should be much longer than 258.
    // Capped by remaining bytes (70000 - 1 - 1 for next byte = 69998)
    // and u16::MAX (65535).
    assert!(
        m.length > 258,
        "extended limit should find matches > 258, got {}",
        m.length
    );
    assert_eq!(m.offset, 1, "should match with offset 1");
}

/// Verify compress_lazy_with_limit produces longer matches and fewer tokens.
#[test]
fn test_compress_lazy_with_limit_extended() {
    let input = vec![0xCCu8; 100_000];
    let deflate_matches = compress_lazy_to_matches(&input).unwrap();
    let extended_matches = compress_lazy_to_matches_with_limit(&input, DEFAULT_MAX_MATCH).unwrap();

    // Extended limit should produce far fewer matches (longer each)
    assert!(
        extended_matches.len() < deflate_matches.len(),
        "extended ({}) should need fewer matches than deflate ({})",
        extended_matches.len(),
        deflate_matches.len()
    );

    // Verify extended matches have lengths > 258
    let max_len = extended_matches.iter().map(|m| m.length).max().unwrap_or(0);
    assert!(
        max_len > 258,
        "extended limit should find matches > 258, got max {}",
        max_len
    );

    // Round-trip must still work
    let compressed = compress_lazy_with_limit(&input, DEFAULT_MAX_MATCH).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

/// Verify compress_lazy_with_limit round-trips various data patterns.
#[test]
fn test_compress_lazy_with_limit_round_trip_patterns() {
    // Repeating pattern
    let pattern = b"hello world! ";
    let input: Vec<u8> = pattern.iter().cycle().take(50_000).copied().collect();
    let compressed = compress_lazy_with_limit(&input, DEFAULT_MAX_MATCH).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input, "repeating pattern round trip failed");

    // Mixed data — shouldn't regress
    let input: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
    let compressed = compress_lazy_with_limit(&input, DEFAULT_MAX_MATCH).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input, "sequential bytes round trip failed");
}

#[test]
fn with_tuning_deep_chain() {
    let data = b"abcabcabcabcabcabcabc";
    let mut finder = HashChainFinder::with_tuning(258, 128);
    for i in 0..3 {
        finder.insert(data, i);
    }
    let m = finder.find_match(data, 3);
    // At position 3, we have the first repeat of "abc" after positions 0-2
    assert!(
        m.length >= 3,
        "deep chain finder should find match at pos 3"
    );
}

#[test]
fn with_tuning_chain_depth_1() {
    let data = b"aaaaaaaaaa";
    let mut finder = HashChainFinder::with_tuning(258, 1);
    for i in 0..data.len() {
        finder.insert(data, i);
    }
    let m = finder.find_match(data, 5);
    // Chain depth 1 may miss some matches but must not panic or return invalid offset
    assert!(m.offset as usize <= 5 || m.length == 0);
}

#[test]
fn with_tuning_chain_depth_256_no_panic() {
    let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
    let mut finder = HashChainFinder::with_tuning(u16::MAX, 256);
    for i in 0..data.len() {
        finder.insert(&data, i);
        let _ = finder.find_match(&data, i);
    }
}
