use super::super::*;

#[test]
fn test_lz77_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello";
    let compressed = engine.lz77_compress(input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz77_lazy_round_trip_repeats() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"abcabcabcabcabcabc";
    let compressed = engine.lz77_compress(input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz77_lazy_round_trip_all_same() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = vec![b'x'; 500];
    let compressed = engine.lz77_compress(&input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz77_lazy_round_trip_large() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"Hello, World! This is a test pattern. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = engine.lz77_compress(&input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz77_lazy_round_trip_binary() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
    let compressed = engine.lz77_compress(&input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz77_lazy_improves_over_greedy() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Repetitive text where lazy matching should produce fewer sequences
    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..50 {
        input.extend_from_slice(pattern);
    }

    let lazy_compressed = engine.lz77_compress(&input).unwrap();
    let greedy_compressed = {
        let matches = engine.find_matches_greedy(&input).unwrap();
        let mut out = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
        for m in &matches {
            out.extend_from_slice(&m.to_bytes());
        }
        out
    };

    // Both should round-trip correctly
    let lazy_dec = crate::lz77::decompress(&lazy_compressed).unwrap();
    let greedy_dec = crate::lz77::decompress(&greedy_compressed).unwrap();
    assert_eq!(lazy_dec, input);
    assert_eq!(greedy_dec, input);

    // Lazy should produce <= sequences than greedy (fewer = better)
    let lazy_seqs = lazy_compressed.len() / Match::SERIALIZED_SIZE;
    let greedy_seqs = greedy_compressed.len() / Match::SERIALIZED_SIZE;
    assert!(
        lazy_seqs <= greedy_seqs,
        "lazy ({lazy_seqs}) should produce <= sequences than greedy ({greedy_seqs})"
    );
}

#[test]
fn test_find_matches_batched_empty() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };
    let result = engine.find_matches_batched(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_find_matches_batched_single_block() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"hello world hello world hello world hello ";
    let big_input: Vec<u8> = pattern
        .iter()
        .cycle()
        .take(MIN_GPU_INPUT_SIZE + 1024)
        .copied()
        .collect();

    let batched = engine.find_matches_batched(&[&big_input]).unwrap();
    assert_eq!(batched.len(), 1);

    // Verify the batched result round-trips correctly
    let mut compressed = Vec::new();
    for m in &batched[0] {
        compressed.extend_from_slice(&m.to_bytes());
    }
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, big_input);
}

#[test]
fn test_find_matches_batched_multi_block() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let block1: Vec<u8> = (0..MIN_GPU_INPUT_SIZE + 1024)
        .map(|i| (i % 251) as u8)
        .collect();
    let block2: Vec<u8> = (0..MIN_GPU_INPUT_SIZE + 2048)
        .map(|i| ((i * 7 + 13) % 251) as u8)
        .collect();

    let batched = engine.find_matches_batched(&[&block1, &block2]).unwrap();
    assert_eq!(batched.len(), 2);

    // Verify both blocks round-trip correctly
    for (i, (matches, original)) in batched.iter().zip([&block1, &block2]).enumerate() {
        let mut compressed = Vec::new();
        for m in matches {
            compressed.extend_from_slice(&m.to_bytes());
        }
        let decompressed = crate::lz77::decompress(&compressed).unwrap();
        assert_eq!(decompressed, *original, "block {i} round-trip failed");
    }
}

#[test]
fn test_find_matches_batched_with_small_block_fallback() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let big: Vec<u8> = (0..MIN_GPU_INPUT_SIZE + 1024)
        .map(|i| (i % 251) as u8)
        .collect();
    let small = b"too small for GPU";

    let batched = engine.find_matches_batched(&[&big, small]).unwrap();
    assert_eq!(batched.len(), 2);

    // Both should produce valid matches that round-trip
    let big_compressed = {
        let mut out = Vec::new();
        for m in &batched[0] {
            out.extend_from_slice(&m.to_bytes());
        }
        out
    };
    let dec = crate::lz77::decompress(&big_compressed).unwrap();
    assert_eq!(dec, big);
}

#[test]
fn test_lz77_compress_batched_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    let block1: Vec<u8> = pattern
        .iter()
        .cycle()
        .take(MIN_GPU_INPUT_SIZE + 512)
        .copied()
        .collect();
    let block2: Vec<u8> = pattern
        .iter()
        .cycle()
        .take(MIN_GPU_INPUT_SIZE + 4096)
        .copied()
        .collect();

    let compressed = engine.lz77_compress_batched(&[&block1, &block2]).unwrap();
    assert_eq!(compressed.len(), 2);

    let dec1 = crate::lz77::decompress(&compressed[0]).unwrap();
    let dec2 = crate::lz77::decompress(&compressed[1]).unwrap();
    assert_eq!(dec1, block1);
    assert_eq!(dec2, block2);
}

#[test]
fn test_lz77_single_large_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Repetitive pattern > MIN_GPU_INPUT_SIZE to exercise u16 clamping
    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    let input: Vec<u8> = pattern
        .iter()
        .cycle()
        .take(MIN_GPU_INPUT_SIZE + 512)
        .copied()
        .collect();
    let compressed = engine.lz77_compress(&input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

/// Regression: all-same-byte input >65535 bytes produces matches with
/// length exceeding u16::MAX in the raw GPU output. dedupe_gpu_matches
/// must clamp to u16::MAX to prevent silent truncation.
#[test]
fn test_lz77_gpu_all_same_byte_large() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = vec![0xAAu8; MIN_GPU_INPUT_SIZE + 4096];
    let compressed = engine.lz77_compress(&input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Tests ported from OpenCL test suite ---

#[test]
fn test_lz77_empty_input() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let compressed = engine.lz77_compress(b"").unwrap();
    assert!(compressed.is_empty());
}

#[test]
fn test_lz77_no_matches() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"abcdefgh";
    let compressed = engine.lz77_compress(input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(&decompressed, input);
}

#[test]
fn test_lz77_hash_round_trip_longer() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let compressed = engine.lz77_compress(input).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(&decompressed, &input[..]);
}

// --- Top-K match finding tests ---

#[test]
fn test_topk_empty_input() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let table = engine.find_topk_matches(b"").unwrap();
    assert_eq!(table.input_len, 0);
}

#[test]
fn test_topk_produces_valid_candidates() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world";
    let table = engine.find_topk_matches(input).unwrap();

    assert_eq!(table.input_len, input.len());
    assert_eq!(table.k, TOPK_K);

    // Verify candidates are valid: offsets point to matching data
    for pos in 0..input.len() {
        for cand in table.at(pos) {
            if cand.length == 0 {
                continue;
            }
            let offset = cand.offset as usize;
            let length = cand.length as usize;
            assert!(offset <= pos, "offset {} > pos {}", offset, pos);
            let match_start = pos - offset;
            for j in 0..length.min(input.len() - pos) {
                assert_eq!(
                    input[match_start + j],
                    input[pos + j],
                    "mismatch at pos {} offset {} len {} byte {}",
                    pos,
                    offset,
                    length,
                    j
                );
            }
        }
    }
}

#[test]
fn test_topk_optimal_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let table = engine.find_topk_matches(input).unwrap();
    let compressed = crate::optimal::compress_optimal_with_table(input, &table).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(&decompressed, &input[..]);
}

#[test]
fn test_topk_optimal_large_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"Hello, World! This is a test pattern. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let table = engine.find_topk_matches(&input).unwrap();
    let compressed = crate::optimal::compress_optimal_with_table(&input, &table).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- GPU LZ77 match quality regression tests ---
//
// These parallel the CPU golden tests in lz77.rs, but use bound-based
// assertions since the GPU hash table uses parallel atomics and may
// produce slightly different (but valid) results across runs.
//
// Each test asserts:
//   1. Round-trip correctness (exact).
//   2. total_seqs ≤ MAX_SEQS (regression ceiling — if GPU quality improves,
//      lower the bound; if it gets worse, the test catches it).
//   3. matched bytes ≥ MIN_MATCHED (most of the input is covered by matches).
//   4. Serialized output ≤ input size (GPU actually compresses, not expands).

/// Count total and match-only sequences from a Vec<Match>.
fn gpu_count_sequences(matches: &[Match]) -> (usize, usize) {
    let total = matches.len();
    let match_seqs = matches
        .iter()
        .filter(|m| m.length > 0 && m.offset > 0)
        .count();
    (total, match_seqs)
}

/// Sum of all match lengths.
fn gpu_total_match_bytes(matches: &[Match]) -> usize {
    matches.iter().map(|m| m.length as usize).sum()
}

/// Round-trip verify GPU matches against original input.
fn gpu_verify_round_trip(matches: &[Match], input: &[u8], label: &str) {
    let mut serialized = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
    for m in matches {
        serialized.extend_from_slice(&m.to_bytes());
    }
    let decompressed = crate::lz77::decompress(&serialized).unwrap();
    assert_eq!(
        decompressed.len(),
        input.len(),
        "{label}: GPU round-trip length mismatch"
    );
    assert_eq!(decompressed, input, "{label}: GPU round-trip data mismatch");
}

#[test]
fn test_gpu_lazy_quality_repeated_pattern() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // 200 repeats of a 38-byte pattern = 7600 bytes.
    // CPU golden: 65 seqs, 31 matches, 7535 bytes matched.
    let pattern = b"Hello, World! This is a test pattern. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    let matches = engine.find_matches(&input).unwrap();
    gpu_verify_round_trip(&matches, &input, "repeated_pattern");

    let (total_seqs, _match_seqs) = gpu_count_sequences(&matches);
    let matched = gpu_total_match_bytes(&matches);
    let serialized_size = total_seqs * Match::SERIALIZED_SIZE;

    // GPU should compress this well — very repetitive pattern.
    // Bounds are generous to absorb GPU non-determinism from atomic hash inserts.
    assert!(
        total_seqs <= 200,
        "repeated_pattern: too many seqs ({total_seqs} > 200), quality regression"
    );
    assert!(
        matched >= 6000,
        "repeated_pattern: too few matched bytes ({matched} < 6000)"
    );
    assert!(
        serialized_size < input.len(),
        "repeated_pattern: GPU output ({serialized_size}) >= input ({}), not compressing",
        input.len()
    );
}

#[test]
fn test_gpu_lazy_quality_all_same() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // 10000 identical bytes.
    // CPU golden: 40 seqs, 39 matches, 9960 bytes matched.
    // GPU is non-deterministic here (2-98 seqs across runs) due to atomics.
    let input = vec![b'A'; 10000];

    let matches = engine.find_matches(&input).unwrap();
    gpu_verify_round_trip(&matches, &input, "all_same");

    let (total_seqs, _match_seqs) = gpu_count_sequences(&matches);
    let matched = gpu_total_match_bytes(&matches);
    let serialized_size = total_seqs * Match::SERIALIZED_SIZE;

    // All-same byte: GPU should find very long matches.
    // Non-deterministic: observed 2-98 seqs across runs due to atomics.
    // Bounds are generous to absorb worst-case GPU scheduling.
    assert!(
        total_seqs <= 500,
        "all_same: too many seqs ({total_seqs} > 500), quality regression"
    );
    assert!(
        matched >= 8000,
        "all_same: too few matched bytes ({matched} < 8000)"
    );
    assert!(
        serialized_size < input.len(),
        "all_same: GPU output ({serialized_size}) >= input ({}), not compressing",
        input.len()
    );
}

#[test]
fn test_gpu_lazy_quality_pattern_64kb() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // 64KB repetitive text — this is the GPU's sweet spot.
    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    while input.len() < 65536 {
        input.extend_from_slice(pattern);
    }
    input.truncate(65536);

    let matches = engine.find_matches(&input).unwrap();
    gpu_verify_round_trip(&matches, &input, "pattern_64KB");

    let (total_seqs, _) = gpu_count_sequences(&matches);
    let matched = gpu_total_match_bytes(&matches);
    let serialized_size = total_seqs * Match::SERIALIZED_SIZE;

    // 64KB repetitive: GPU should compress very well (ratio < 1.0).
    assert!(
        total_seqs <= 500,
        "pattern_64KB: too many seqs ({total_seqs} > 500), quality regression"
    );
    assert!(
        matched >= 55000,
        "pattern_64KB: too few matched bytes ({matched} < 55000)"
    );
    assert!(
        serialized_size < input.len(),
        "pattern_64KB: GPU output ({serialized_size}) >= input ({}), not compressing",
        input.len()
    );
}

#[test]
fn test_gpu_lazy_quality_vs_cpu_64kb() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Compare GPU vs CPU match quality on 64KB text data.
    // After the hash table fix, GPU should be within 2x of CPU seqs at this size.
    let pattern = b"the quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    while input.len() < 65536 {
        input.extend_from_slice(pattern);
    }
    input.truncate(65536);

    let gpu_matches = engine.find_matches(&input).unwrap();
    let cpu_matches = crate::lz77::compress_lazy_to_matches(&input).unwrap();

    gpu_verify_round_trip(&gpu_matches, &input, "vs_cpu_64KB");

    let (gpu_seqs, _) = gpu_count_sequences(&gpu_matches);
    let (cpu_seqs, _) = gpu_count_sequences(&cpu_matches);

    // GPU should produce no more than 3x the CPU's sequence count at 64KB.
    // With the improved hash table, it's typically much closer to 1x.
    let ratio = gpu_seqs as f64 / cpu_seqs as f64;
    assert!(
        ratio <= 3.0,
        "GPU/CPU seq ratio {ratio:.2} > 3.0 at 64KB (GPU={gpu_seqs}, CPU={cpu_seqs})"
    );
}

#[test]
fn test_gpu_lazy_quality_vs_cpu_128kb() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Use realistic text data (diverse byte distribution) rather than a
    // short repeating pattern which concentrates all positions into a
    // handful of hash buckets.
    // Multiple distinct sentences → many unique 3-byte prefixes → even hash spread.
    let sentences = [
        b"the quick brown fox jumps over the lazy dog. " as &[u8],
        b"pack my box with five dozen liquor jugs now. ",
        b"how vexingly quick daft zebras jump high up! ",
        b"the five boxing wizards jump quickly at dawn. ",
        b"sphinx of black quartz, judge my vow today!! ",
        b"two driven jocks help fax my big quiz plan. ",
        b"crazy frederick bought many very exquisite. ",
        b"we promptly judged antique ivory buckles ok. ",
    ];
    let mut input = Vec::new();
    let mut i = 0;
    while input.len() < 131072 {
        input.extend_from_slice(sentences[i % sentences.len()]);
        i += 1;
    }
    input.truncate(131072);

    let gpu_matches = engine.find_matches(&input).unwrap();
    let cpu_matches = crate::lz77::compress_lazy_to_matches(&input).unwrap();

    gpu_verify_round_trip(&gpu_matches, &input, "vs_cpu_128KB");

    let (gpu_seqs, _) = gpu_count_sequences(&gpu_matches);
    let (cpu_seqs, _) = gpu_count_sequences(&cpu_matches);

    // At 128KB with diverse data, GPU should be within 5x of CPU.
    let ratio = gpu_seqs as f64 / cpu_seqs as f64;
    assert!(
        ratio <= 5.0,
        "GPU/CPU seq ratio {ratio:.2} > 5.0 at 128KB (GPU={gpu_seqs}, CPU={cpu_seqs})"
    );
}

fn corpus_file_for_gpu(name: &str) -> Option<Vec<u8>> {
    for dir in &["samples/cantrbry", "/home/user/libpz/samples/cantrbry"] {
        let path = format!("{}/{}", dir, name);
        if let Ok(data) = std::fs::read(&path) {
            return Some(data);
        }
    }
    None
}

#[test]
fn test_gpu_lazy_quality_alice29() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let Some(data) = corpus_file_for_gpu("alice29.txt") else {
        eprintln!("skipping: alice29.txt not found");
        return;
    };
    assert_eq!(data.len(), 152089);

    let gpu_matches = engine.find_matches(&data).unwrap();
    let cpu_matches = crate::lz77::compress_lazy_to_matches(&data).unwrap();
    gpu_verify_round_trip(&gpu_matches, &data, "alice29.txt");

    let (gpu_seqs, _) = gpu_count_sequences(&gpu_matches);
    let (cpu_seqs, _) = gpu_count_sequences(&cpu_matches);
    let gpu_matched = gpu_total_match_bytes(&gpu_matches);

    // alice29.txt is 152KB — GPU hash table has some pressure.
    // CPU golden: 27564 seqs. GPU should be within 6x.
    let ratio = gpu_seqs as f64 / cpu_seqs as f64;
    assert!(
        ratio <= 6.0,
        "alice29.txt: GPU/CPU seq ratio {ratio:.2} > 6.0 (GPU={gpu_seqs}, CPU={cpu_seqs})"
    );
    assert!(
        gpu_matched >= 80000,
        "alice29.txt: too few GPU matched bytes ({gpu_matched} < 80000)"
    );
}

#[test]
fn test_gpu_lazy_quality_fields_c() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let Some(data) = corpus_file_for_gpu("fields.c") else {
        eprintln!("skipping: fields.c not found");
        return;
    };
    assert_eq!(data.len(), 11150);

    let gpu_matches = engine.find_matches(&data).unwrap();
    let cpu_matches = crate::lz77::compress_lazy_to_matches(&data).unwrap();
    gpu_verify_round_trip(&gpu_matches, &data, "fields.c");

    let (gpu_seqs, _) = gpu_count_sequences(&gpu_matches);
    let (cpu_seqs, _) = gpu_count_sequences(&cpu_matches);
    let gpu_matched = gpu_total_match_bytes(&gpu_matches);

    // fields.c is only 11KB — should be well within GPU hash table capacity.
    // CPU golden: 1943 seqs. GPU should be within 3x.
    let ratio = gpu_seqs as f64 / cpu_seqs as f64;
    assert!(
        ratio <= 3.0,
        "fields.c: GPU/CPU seq ratio {ratio:.2} > 3.0 (GPU={gpu_seqs}, CPU={cpu_seqs})"
    );
    assert!(
        gpu_matched >= 7000,
        "fields.c: too few GPU matched bytes ({gpu_matched} < 7000)"
    );
}

// ---------------------------------------------------------------------------
// LZ77 GPU Decompression tests
// ---------------------------------------------------------------------------

/// Helper: compress input into blocks and return (block_data, block_meta).
/// block_meta entries are (match_data_offset, num_matches, decompressed_size).
fn lz77_compress_blocks(input: &[u8], block_size: usize) -> (Vec<u8>, Vec<(usize, usize, usize)>) {
    let mut block_data = Vec::new();
    let mut block_meta = Vec::new();

    for chunk in input.chunks(block_size) {
        let compressed = crate::lz77::compress_lazy(chunk).unwrap();
        let data_offset = block_data.len();
        let num_matches = compressed.len() / crate::lz77::Match::SERIALIZED_SIZE;
        block_meta.push((data_offset, num_matches, chunk.len()));
        block_data.extend_from_slice(&compressed);
    }

    (block_data, block_meta)
}

#[test]
fn test_lz77_decompress_blocks_small() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Small repetitive input — fits in one block
    let input = b"abcabcabcabcabcabcabcabc";
    let (block_data, block_meta) = lz77_compress_blocks(input, 1024);

    let gpu_result = engine
        .lz77_decompress_blocks(&block_data, &block_meta)
        .unwrap();
    assert_eq!(gpu_result, input);
}

#[test]
fn test_lz77_decompress_blocks_multiblock() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Larger input split into multiple 4KB blocks
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..400 {
        input.extend_from_slice(pattern);
    }

    let (block_data, block_meta) = lz77_compress_blocks(&input, 4096);
    assert!(block_meta.len() > 1, "should have multiple blocks");

    let gpu_result = engine
        .lz77_decompress_blocks(&block_data, &block_meta)
        .unwrap();
    assert_eq!(gpu_result, input);
}

#[test]
fn test_lz77_decompress_blocks_vs_cpu() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Test that GPU decompression matches CPU decompression exactly
    let pattern = b"hello world compression test data ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    let (block_data, block_meta) = lz77_compress_blocks(&input, 2048);

    // CPU decompress each block independently
    let mut cpu_result = Vec::new();
    for &(data_offset, num_matches, _decompressed_size) in &block_meta {
        let block_bytes = &block_data
            [data_offset..data_offset + num_matches * crate::lz77::Match::SERIALIZED_SIZE];
        let decoded = crate::lz77::decompress(block_bytes).unwrap();
        cpu_result.extend_from_slice(&decoded);
    }
    assert_eq!(cpu_result, input, "CPU decompress sanity check");

    // GPU decompress
    let gpu_result = engine
        .lz77_decompress_blocks(&block_data, &block_meta)
        .unwrap();
    assert_eq!(gpu_result, cpu_result, "GPU vs CPU mismatch");
}

#[test]
fn test_lz77_decompress_blocks_overlapping_backref() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Create input with overlapping back-references (offset < length)
    // "aaaa..." repeated pattern triggers offset=1, length=N
    let input: Vec<u8> = vec![b'a'; 200];

    let (block_data, block_meta) = lz77_compress_blocks(&input, 1024);

    let gpu_result = engine
        .lz77_decompress_blocks(&block_data, &block_meta)
        .unwrap();
    assert_eq!(gpu_result, input);
}

// --- Cooperative-stitch kernel tests ---

#[test]
fn test_lz77_coop_round_trip_repeats() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"abcabcabcabcabcabc";
    let matches = engine.find_matches_coop(input).unwrap();
    let mut compressed = Vec::new();
    for m in &matches {
        compressed.extend_from_slice(&m.to_bytes());
    }
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input.to_vec());
}

#[test]
fn test_lz77_coop_round_trip_all_same() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = vec![b'x'; 500];
    let matches = engine.find_matches_coop(&input).unwrap();
    let mut compressed = Vec::new();
    for m in &matches {
        compressed.extend_from_slice(&m.to_bytes());
    }
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz77_coop_round_trip_large() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::with_capacity(256 * 1024);
    while input.len() < 256 * 1024 {
        let chunk = (256 * 1024 - input.len()).min(pattern.len());
        input.extend_from_slice(&pattern[..chunk]);
    }
    let matches = engine.find_matches_coop(&input).unwrap();
    let mut compressed = Vec::new();
    for m in &matches {
        compressed.extend_from_slice(&m.to_bytes());
    }
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

/// Exercises the lazy kernel path directly to prevent bitrot.
/// The lazy kernel is kept for A/B benchmarking against the default coop kernel.
#[test]
fn test_lz77_lazy_kernel_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::with_capacity(8 * 1024);
    while input.len() < 8 * 1024 {
        let chunk = (8 * 1024 - input.len()).min(pattern.len());
        input.extend_from_slice(&pattern[..chunk]);
    }

    let matches = engine.find_matches_lazy(&input).unwrap();
    assert!(!matches.is_empty(), "lazy kernel should produce matches");

    let mut compressed = Vec::new();
    for m in &matches {
        compressed.extend_from_slice(&m.to_bytes());
    }
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input, "lazy kernel round-trip failed");
}
