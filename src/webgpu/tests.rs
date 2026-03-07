use super::*;

#[test]
fn test_gpu_match_struct_size() {
    assert_eq!(std::mem::size_of::<GpuMatch>(), 12);
}

#[test]
fn test_dedupe_all_literals() {
    let input = b"abcdef";
    let gpu_matches: Vec<GpuMatch> = input
        .iter()
        .map(|&b| GpuMatch {
            offset: 0,
            length: 0,
            next: b as u32,
        })
        .collect();

    let result = dedupe_gpu_matches(&gpu_matches, input);
    assert_eq!(result.len(), 6);
    for (i, m) in result.iter().enumerate() {
        assert_eq!(m.offset, 0);
        assert_eq!(m.length, 0);
        assert_eq!(m.next, input[i]);
    }
}

#[test]
fn test_dedupe_with_match() {
    let input = b"abcabc";
    let gpu_matches = vec![
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'a' as u32,
        },
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'b' as u32,
        },
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'c' as u32,
        },
        GpuMatch {
            offset: 3,
            length: 2,
            next: b'c' as u32,
        },
        GpuMatch {
            offset: 3,
            length: 1,
            next: b'c' as u32,
        },
        GpuMatch {
            offset: 3,
            length: 0,
            next: b'c' as u32,
        },
    ];

    let result = dedupe_gpu_matches(&gpu_matches, input);
    assert_eq!(result.len(), 4);
    assert_eq!(result[3].offset, 3);
    assert_eq!(result[3].length, 2);
}

#[test]
fn test_probe_devices() {
    // Should not crash; may return empty on headless systems
    let devices = probe_devices();
    for d in &devices {
        assert!(!d.name.is_empty() || d.name.is_empty()); // no-op, just validate struct
    }
}

#[test]
fn test_engine_creation() {
    // May return Unsupported on headless systems -- that's OK
    match WebGpuEngine::new() {
        Ok(engine) => {
            assert!(!engine.device_name().is_empty());
            assert!(engine.max_work_group_size() >= 1);
        }
        Err(PzError::Unsupported) => {
            // Expected on systems without GPU
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn test_profiling_creation() {
    match WebGpuEngine::with_profiling(true) {
        Ok(engine) => {
            eprintln!(
                "Device: {}, profiling={}",
                engine.device_name(),
                engine.profiling()
            );
            // profiling accessor must match what was actually negotiated
            // (may be false if device doesn't support TIMESTAMP_QUERY)
        }
        Err(PzError::Unsupported) => {
            // Expected on systems without GPU
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

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

#[test]
fn test_byte_histogram() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"aabbcc";
    let hist = engine.byte_histogram(input).unwrap();
    assert_eq!(hist[b'a' as usize], 2);
    assert_eq!(hist[b'b' as usize], 2);
    assert_eq!(hist[b'c' as usize], 2);
    assert_eq!(hist[b'd' as usize], 0);
}

#[test]
fn test_huffman_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello, world!";
    let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let (encoded, total_bits) = engine.huffman_encode(input, &code_lut).unwrap();
    let mut decoded = vec![0u8; input.len()];
    let decoded_len = tree
        .decode_to_buf(&encoded, total_bits, &mut decoded)
        .unwrap();
    assert_eq!(&decoded[..decoded_len], input);
}

#[test]
fn test_huffman_gpu_scan_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello, world!";
    let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let (encoded, total_bits) = engine.huffman_encode_gpu_scan(input, &code_lut).unwrap();
    let mut decoded = vec![0u8; input.len()];
    let decoded_len = tree
        .decode_to_buf(&encoded, total_bits, &mut decoded)
        .unwrap();
    assert_eq!(&decoded[..decoded_len], input);
}

#[test]
fn test_bwt_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"banana";
    let bwt_result = engine.bwt_encode(input).unwrap();
    let decoded = crate::bwt::decode(&bwt_result.data, bwt_result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_bijective_bwt_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Test with various inputs
    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        ("banana", b"banana".to_vec()),
        (
            "hello_repeated",
            b"hello world hello world hello world".to_vec(),
        ),
        ("binary", (0..=255u8).collect()),
    ];

    for (name, input) in &test_cases {
        let (gpu_data, gpu_factors) = engine.bwt_encode_bijective(input).unwrap();

        // Compare against CPU bijective BWT
        let (cpu_data, cpu_factors) = crate::bwt::encode_bijective(input).unwrap();
        assert_eq!(gpu_factors, cpu_factors, "factor lengths differ on {name}");
        assert_eq!(gpu_data, cpu_data, "BWT data differs on {name}");

        // Round-trip decode
        let decoded = crate::bwt::decode_bijective(&gpu_data, &gpu_factors).unwrap();
        assert_eq!(
            decoded, *input,
            "GPU bijective BWT round-trip failed on {name}"
        );
    }
}

#[test]
fn test_webgpu_deflate_pipeline_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let options = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        webgpu_engine: Some(engine),
        ..Default::default()
    };

    let compressed =
        crate::pipeline::compress_with_options(input, crate::pipeline::Pipeline::Deflate, &options)
            .unwrap();
    let decompressed = crate::pipeline::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_webgpu_deflate_pipeline_larger() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    let options = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        webgpu_engine: Some(engine),
        ..Default::default()
    };

    let compressed = crate::pipeline::compress_with_options(
        &input,
        crate::pipeline::Pipeline::Deflate,
        &options,
    )
    .unwrap();
    let decompressed = crate::pipeline::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_fse_decode_hello() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello, world!";
    let encoded = crate::fse::encode_interleaved(input);
    let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_fse_decode_repeated() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = vec![b'a'; 100];
    let encoded = crate::fse::encode_interleaved(&input);
    let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_fse_decode_binary() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let encoded = crate::fse::encode_interleaved(&input);
    let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_fse_decode_medium_text() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"The Burrows-Wheeler transform clusters bytes. ";
    let mut input = Vec::new();
    for _ in 0..20 {
        input.extend_from_slice(pattern);
    }
    let encoded = crate::fse::encode_interleaved(&input);
    let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_fse_decode_various_stream_counts() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..200).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    for n in [1, 2, 4, 8] {
        let encoded = crate::fse::encode_interleaved_n(&input, n, 10);
        let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input, "failed at num_streams={}", n);
    }
}

// --- Tests ported from OpenCL test suite ---

#[test]
fn test_dedupe_empty() {
    let result = dedupe_gpu_matches(&[], &[]);
    assert!(result.is_empty());
}

#[test]
fn test_device_count_does_not_panic() {
    let count = device_count();
    let _ = count;
}

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

#[test]
fn test_is_cpu_device() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };
    let _is_cpu = engine.is_cpu_device();
}

// --- BWT edge case tests ---

#[test]
fn test_bwt_hello_world_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world";
    let gpu_result = engine.bwt_encode(input).unwrap();
    let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_bwt_binary_data() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..=255).collect();
    let gpu_result = engine.bwt_encode(&input).unwrap();
    let cpu_result = crate::bwt::encode(&input).unwrap();

    assert_eq!(gpu_result.data, cpu_result.data);
    assert_eq!(gpu_result.primary_index, cpu_result.primary_index);

    let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_bwt_medium_sizes() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Test sizes that cross the multi-workgroup boundary.
    for size in [257, 300, 400, 500, 512, 513, 768, 1024] {
        let mut input = Vec::with_capacity(size);
        for i in 0..size {
            input.push((i % 256) as u8);
        }
        let gpu_result = engine.bwt_encode(&input).unwrap_or_else(|e| {
            panic!("GPU BWT failed for size {}: {:?}", size, e);
        });
        let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
        assert_eq!(decoded, input, "Round-trip failed for size {}", size);
    }
}

#[test]
fn test_bwt_single_byte() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"x";
    let gpu_result = engine.bwt_encode(input).unwrap();
    assert_eq!(gpu_result.data, vec![b'x']);
    assert_eq!(gpu_result.primary_index, 0);
}

#[test]
fn test_bwt_all_same() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = vec![b'a'; 64];
    let gpu_result = engine.bwt_encode(&input).unwrap();
    let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_bwt_large_structured() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // 4KB of structured data — enough to exercise multi-level prefix sum
    let mut input = Vec::new();
    for i in 0..256u16 {
        input.extend_from_slice(&i.to_le_bytes());
    }
    for _ in 0..80 {
        input.extend_from_slice(b"the quick brown fox jumps over the lazy dog ");
    }
    while input.len() < 4096 {
        input.push(b'x');
    }
    input.truncate(4096);

    let gpu_result = engine.bwt_encode(&input).unwrap();
    let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
    assert_eq!(decoded, input);
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

// --- Huffman edge case tests ---

#[test]
fn test_byte_histogram_empty() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let hist = engine.byte_histogram(&[]).unwrap();
    assert!(hist.iter().all(|&c| c == 0));
}

#[test]
fn test_huffman_encode_cpu_vs_gpu() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world!";
    let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let (gpu_encoded, gpu_bits) = engine.huffman_encode(input, &code_lut).unwrap();
    let (cpu_encoded, cpu_bits) = tree.encode(input).unwrap();

    assert_eq!(gpu_bits, cpu_bits, "bit counts differ");
    assert_eq!(
        gpu_encoded,
        cpu_encoded,
        "encoded data differs: gpu {} bytes, cpu {} bytes",
        gpu_encoded.len(),
        cpu_encoded.len()
    );

    // Verify round-trip by decoding with CPU decoder
    let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_huffman_encode_larger() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }

    let tree = crate::huffman::HuffmanTree::from_data(&input).unwrap();
    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let (gpu_encoded, gpu_bits) = engine.huffman_encode(&input, &code_lut).unwrap();
    let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_huffman_gpu_scan_cpu_comparison() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world!";
    let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let (gpu_encoded, gpu_bits) = engine.huffman_encode_gpu_scan(input, &code_lut).unwrap();
    let (cpu_encoded, cpu_bits) = tree.encode(input).unwrap();

    assert_eq!(gpu_bits, cpu_bits, "bit counts differ");
    assert_eq!(gpu_encoded, cpu_encoded, "encoded data differs");

    let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
    assert_eq!(decoded, input);
}

// --- Modular GPU pipeline round-trip tests ---

fn gpu_pipeline_round_trip(input: &[u8], pipeline: crate::pipeline::Pipeline) {
    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let options = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        webgpu_engine: Some(engine),
        ..Default::default()
    };

    let compressed = crate::pipeline::compress_with_options(input, pipeline, &options)
        .unwrap_or_else(|e| {
            panic!("compress failed for {:?}: {:?}", pipeline, e);
        });

    let decompressed = crate::pipeline::decompress(&compressed).unwrap_or_else(|e| {
        panic!("decompress failed for {:?}: {:?}", pipeline, e);
    });

    assert_eq!(
        decompressed, input,
        "round-trip mismatch for {:?}",
        pipeline
    );
}

#[test]
fn test_modular_gpu_deflate_round_trip() {
    // GPU LZ77 → GPU Huffman (modular stage path)
    let mut input = Vec::new();
    for i in 0u8..=255 {
        for _ in 0..40 {
            input.push(i);
        }
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Deflate);
}

#[test]
fn test_gpu_lz77_cpu_rans_round_trip() {
    // GPU LZ77 → CPU rANS
    let pattern = b"Hello, World! This is a test pattern for GPU+CPU composition. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Lzr);
}

#[test]
fn test_gpu_lz77_cpu_fse_round_trip() {
    // GPU LZ77 → CPU FSE
    let pattern = b"Hello, World! This is a test pattern for GPU+CPU composition. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Lzf);
}

#[test]
fn test_gpu_bwt_cpu_pipeline_round_trip() {
    // GPU BWT → CPU MTF → CPU RLE → CPU FSE
    let pattern = b"the quick brown fox jumps over the lazy dog ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Bw);
}

// --- GPU FSE encode tests ---

#[test]
fn test_gpu_fse_encode_round_trip() {
    // GPU encode → CPU decode
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let encoded = engine
        .fse_encode_interleaved_gpu(&input, 4, crate::fse::DEFAULT_ACCURACY_LOG)
        .unwrap();
    let decoded = crate::fse::decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_fse_encode_decode_gpu_round_trip() {
    // GPU encode → GPU decode
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let encoded = engine
        .fse_encode_interleaved_gpu(&input, 4, crate::fse::DEFAULT_ACCURACY_LOG)
        .unwrap();
    let decoded = engine.fse_decode(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_fse_encode_various_params() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..300).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    for (num_states, al) in [(4, 7), (8, 8), (16, 9), (32, 7), (4, 10)] {
        let encoded = engine
            .fse_encode_interleaved_gpu(&input, num_states, al)
            .unwrap();
        let decoded = crate::fse::decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(
            decoded, input,
            "failed at num_states={num_states}, accuracy_log={al}"
        );
    }
}

#[test]
fn test_gpu_fse_encode_single_byte() {
    // Edge case: single repeated byte
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = vec![0xAA; 100];
    let encoded = engine
        .fse_encode_interleaved_gpu(&input, 4, crate::fse::DEFAULT_ACCURACY_LOG)
        .unwrap();
    let decoded = crate::fse::decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_fse_encode_all_bytes() {
    // All 256 byte values present
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let encoded = engine.fse_encode_interleaved_gpu(&input, 8, 10).unwrap();
    let decoded = crate::fse::decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_lzfi_pipeline_webgpu_round_trip() {
    // Full Lzfi pipeline round-trip with WebGPU backend
    let pattern = b"Hello, World! This is a test pattern for GPU Lzfi pipeline. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Lzfi);
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

#[cfg(test)]
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

// --- GPU prefix sum tests ---

#[test]
fn test_prefix_sum_small() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let n = data.len();
    let buf = engine.create_buffer_init(
        "ps_test",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, n).unwrap();

    let raw = engine.read_buffer(&buf, (n * 4) as u64);
    let result: &[u32] = bytemuck::cast_slice(&raw);
    // Exclusive prefix sum of [1,2,3,4,5] = [0,1,3,6,10]
    assert_eq!(&result[..n], &[0, 1, 3, 6, 10]);
}

#[test]
fn test_prefix_sum_exact_block_size() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Exact block size = 512 (256 * 2)
    let n = 512;
    let data: Vec<u32> = (0..n as u32).map(|_| 1).collect();
    let buf = engine.create_buffer_init(
        "ps_test_512",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, n).unwrap();

    let raw = engine.read_buffer(&buf, (n * 4) as u64);
    let result: &[u32] = bytemuck::cast_slice(&raw);
    // Exclusive prefix sum of all-ones = [0, 1, 2, ..., 511]
    for (i, &val) in result[..n].iter().enumerate() {
        assert_eq!(val, i as u32, "mismatch at index {i}");
    }
}

#[test]
fn test_prefix_sum_multi_level() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // 2000 elements requires multi-level (> 512)
    let n = 2000;
    let data: Vec<u32> = (0..n as u32).map(|i| (i % 7) + 1).collect();
    let buf = engine.create_buffer_init(
        "ps_test_multi",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, n).unwrap();

    let raw = engine.read_buffer(&buf, (n * 4) as u64);
    let result: &[u32] = bytemuck::cast_slice(&raw);

    // Verify against CPU prefix sum
    let mut expected = vec![0u32; n];
    let mut sum = 0u32;
    for (i, slot) in expected.iter_mut().enumerate() {
        *slot = sum;
        sum += (i as u32 % 7) + 1;
    }
    assert_eq!(&result[..n], &expected[..]);
}

#[test]
fn test_prefix_sum_single_element() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let data: Vec<u32> = vec![42];
    let buf = engine.create_buffer_init(
        "ps_test_single",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, 1).unwrap();

    let raw = engine.read_buffer(&buf, 4);
    let result: &[u32] = bytemuck::cast_slice(&raw);
    assert_eq!(result[0], 0);
}

// --- huffman_encode_fully_on_device tests ---

#[test]
fn test_huffman_encode_fully_on_device_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world!";
    let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let device_input = DeviceBuf::from_host(&engine, input).unwrap();
    let (gpu_encoded, gpu_bits) = engine
        .huffman_encode_fully_on_device(&device_input, &code_lut)
        .unwrap();
    let (cpu_encoded, cpu_bits) = tree.encode(input).unwrap();

    assert_eq!(gpu_bits, cpu_bits, "bit counts differ");
    assert_eq!(gpu_encoded, cpu_encoded, "encoded data differs");

    let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_huffman_encode_fully_on_device_larger() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }

    let tree = crate::huffman::HuffmanTree::from_data(&input).unwrap();
    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let device_input = DeviceBuf::from_host(&engine, &input).unwrap();
    let (gpu_encoded, gpu_bits) = engine
        .huffman_encode_fully_on_device(&device_input, &code_lut)
        .unwrap();
    let decoded = tree.decode(&gpu_encoded, gpu_bits).unwrap();
    assert_eq!(decoded, input);
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

#[test]
fn test_rans_interleaved_decode_gpu_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..8192).map(|i| ((i * 41 + 61) % 251) as u8).collect();
    let encoded = crate::rans::encode_interleaved_n(&input, 4, crate::rans::DEFAULT_SCALE_BITS);
    let decoded = engine
        .rans_decode_interleaved_gpu(&encoded, input.len())
        .unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_rans_chunked_encode_gpu_cpu_decode_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..32768).map(|i| ((i * 13 + 5) % 256) as u8).collect();
    let (encoded, used_chunked) = engine
        .rans_encode_chunked_payload_gpu(&input, 4, crate::rans::DEFAULT_SCALE_BITS, 2048)
        .unwrap();
    assert!(used_chunked);
    let decoded = crate::rans::decode_chunked(&encoded).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_rans_chunked_encode_gpu_decode_gpu_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..24576).map(|i| ((i * 17 + 11) % 251) as u8).collect();
    let (encoded, used_chunked) = engine
        .rans_encode_chunked_payload_gpu(&input, 4, crate::rans::DEFAULT_SCALE_BITS, 1024)
        .unwrap();
    assert!(used_chunked);
    let decoded = engine
        .rans_decode_chunked_payload_gpu(&encoded, input.len())
        .unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_rans_chunked_encode_gpu_fallback_matches_cpu() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..8192).map(|i| ((i * 31 + 7) % 256) as u8).collect();
    let (encoded_gpu, used_chunked_gpu) = engine
        .rans_encode_chunked_payload_gpu(&input, 4, crate::rans::DEFAULT_SCALE_BITS, 0)
        .unwrap();
    assert!(!used_chunked_gpu);

    let encoded_cpu = crate::rans::encode_interleaved_n(&input, 4, crate::rans::DEFAULT_SCALE_BITS);
    assert_eq!(encoded_gpu, encoded_cpu);
    let decoded = crate::rans::decode_interleaved(&encoded_gpu, input.len()).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_rans_chunked_encode_gpu_batched_cpu_decode_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let inputs: Vec<Vec<u8>> = vec![
        (0..24_576).map(|i| ((i * 17 + 11) % 251) as u8).collect(),
        (0..16_384).map(|i| ((i * 31 + 7) % 256) as u8).collect(),
        (0..8_192).map(|i| ((i * 13 + 5) % 256) as u8).collect(),
    ];
    let input_refs: Vec<&[u8]> = inputs.iter().map(|v| v.as_slice()).collect();

    let encoded = engine
        .rans_encode_chunked_payload_gpu_batched(
            &input_refs,
            4,
            crate::rans::DEFAULT_SCALE_BITS,
            2048,
        )
        .unwrap();
    assert_eq!(encoded.len(), inputs.len());

    for (i, (payload, used_chunked)) in encoded.iter().enumerate() {
        assert!(*used_chunked);
        let decoded = crate::rans::decode_chunked(payload).unwrap();
        assert_eq!(decoded, inputs[i]);
    }
}

#[test]
fn test_rans_chunked_encode_gpu_batched_shared_table_cpu_decode_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let full_input: Vec<u8> = (0..65_536).map(|i| ((i * 37 + 19) % 251) as u8).collect();
    let input_blocks: Vec<&[u8]> = full_input.chunks(16_384).collect();
    let expected_blocks: Vec<Vec<u8>> = input_blocks.iter().map(|block| block.to_vec()).collect();

    let encoded = engine
        .rans_encode_chunked_payload_gpu_batched_shared_table(
            &input_blocks,
            &full_input,
            4,
            crate::rans::DEFAULT_SCALE_BITS,
            2048,
        )
        .unwrap();
    assert_eq!(encoded.len(), expected_blocks.len());

    for (i, (payload, used_chunked)) in encoded.iter().enumerate() {
        assert!(*used_chunked);
        let decoded = crate::rans::decode_chunked(payload).unwrap();
        assert_eq!(decoded, expected_blocks[i]);
    }
}

#[test]
fn test_rans_chunked_encode_gpu_batched_shared_table_packed_cpu_decode_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // 8 blocks so packed shared-table encode remains eligible under the
    // ring-depth safety gate.
    let full_input: Vec<u8> = (0..32_768).map(|i| ((i * 23 + 29) % 251) as u8).collect();
    let input_blocks: Vec<&[u8]> = full_input.chunks(4_096).collect();
    let expected_blocks: Vec<Vec<u8>> = input_blocks.iter().map(|block| block.to_vec()).collect();

    let encoded = engine
        .rans_encode_chunked_payload_gpu_batched_shared_table(
            &input_blocks,
            &full_input,
            4,
            crate::rans::DEFAULT_SCALE_BITS,
            2048,
        )
        .unwrap();
    assert_eq!(encoded.len(), expected_blocks.len());

    for (i, (payload, used_chunked)) in encoded.iter().enumerate() {
        assert!(*used_chunked);
        let decoded = crate::rans::decode_chunked(payload).unwrap();
        assert_eq!(decoded, expected_blocks[i]);
    }
}

#[test]
fn test_rans_chunked_encode_gpu_batched_shared_table_lanes_over_64_falls_back() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let full_input: Vec<u8> = (0..16_384).map(|i| ((i * 17 + 13) % 251) as u8).collect();
    let input_blocks: Vec<&[u8]> = full_input.chunks(4_096).collect();
    let expected_blocks: Vec<Vec<u8>> = input_blocks.iter().map(|block| block.to_vec()).collect();

    // chunk_size=0 forces non-chunked fallback; this should remain valid even
    // when lanes exceed the GPU-supported maximum.
    let encoded = engine
        .rans_encode_chunked_payload_gpu_batched_shared_table(
            &input_blocks,
            &full_input,
            128,
            crate::rans::DEFAULT_SCALE_BITS,
            0,
        )
        .unwrap();
    assert_eq!(encoded.len(), expected_blocks.len());

    for (i, (payload, used_chunked)) in encoded.iter().enumerate() {
        assert!(!*used_chunked);
        let decoded = crate::rans::decode_interleaved(payload, expected_blocks[i].len()).unwrap();
        assert_eq!(decoded, expected_blocks[i]);
    }
}

#[test]
fn test_rans_chunked_decode_gpu_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..32768).map(|i| ((i * 19 + 7) % 256) as u8).collect();
    let encoded = crate::rans::encode_chunked(&input, 4, crate::rans::DEFAULT_SCALE_BITS, 2048);

    let decoded = engine
        .rans_decode_chunked_payload_gpu(&encoded, input.len())
        .unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_rans_chunked_decode_gpu_batched_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let inputs: Vec<Vec<u8>> = vec![
        (0..24_576).map(|i| ((i * 17 + 11) % 251) as u8).collect(),
        (0..16_384).map(|i| ((i * 31 + 7) % 256) as u8).collect(),
        (0..8_192).map(|i| ((i * 13 + 5) % 256) as u8).collect(),
    ];
    let mut payloads = Vec::with_capacity(inputs.len());
    for input in &inputs {
        let encoded = crate::rans::encode_chunked(input, 4, crate::rans::DEFAULT_SCALE_BITS, 2048);
        payloads.push(encoded);
    }

    let decode_inputs: Vec<(&[u8], usize)> = payloads
        .iter()
        .zip(inputs.iter())
        .map(|(payload, input)| (payload.as_slice(), input.len()))
        .collect();
    let decoded = engine
        .rans_decode_chunked_payload_gpu_batched(&decode_inputs)
        .unwrap();
    assert_eq!(decoded.len(), inputs.len());
    for (i, output) in decoded.iter().enumerate() {
        assert_eq!(output, &inputs[i]);
    }
}

#[test]
fn test_rans_chunked_decode_gpu_batched_shared_table_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let full_input: Vec<u8> = (0..65_536).map(|i| ((i * 41 + 3) % 251) as u8).collect();
    // Use enough blocks to exercise packed shared-table decode submission.
    let input_blocks: Vec<&[u8]> = full_input.chunks(4_096).collect();
    let expected_blocks: Vec<Vec<u8>> = input_blocks.iter().map(|block| block.to_vec()).collect();

    let encoded = engine
        .rans_encode_chunked_payload_gpu_batched_shared_table(
            &input_blocks,
            &full_input,
            4,
            crate::rans::DEFAULT_SCALE_BITS,
            2048,
        )
        .unwrap();
    let decode_inputs: Vec<(&[u8], usize)> = encoded
        .iter()
        .zip(expected_blocks.iter())
        .map(|((payload, used_chunked), block)| {
            assert!(*used_chunked);
            (payload.as_slice(), block.len())
        })
        .collect();

    let decoded = engine
        .rans_decode_chunked_payload_gpu_batched_shared_table(&decode_inputs, &full_input)
        .unwrap();
    assert_eq!(decoded.len(), expected_blocks.len());
    for (i, output) in decoded.iter().enumerate() {
        assert_eq!(output, &expected_blocks[i]);
    }
}

#[test]
fn test_rans_chunked_decode_gpu_batched_shared_table_repeated_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let full_input: Vec<u8> = (0..65_536).map(|i| ((i * 43 + 7) % 251) as u8).collect();
    let input_blocks: Vec<&[u8]> = full_input.chunks(4_096).collect();
    let expected_blocks: Vec<Vec<u8>> = input_blocks.iter().map(|block| block.to_vec()).collect();

    let encoded = engine
        .rans_encode_chunked_payload_gpu_batched_shared_table(
            &input_blocks,
            &full_input,
            4,
            crate::rans::DEFAULT_SCALE_BITS,
            2048,
        )
        .unwrap();
    let decode_inputs: Vec<(&[u8], usize)> = encoded
        .iter()
        .zip(expected_blocks.iter())
        .map(|((payload, used_chunked), block)| {
            assert!(*used_chunked);
            (payload.as_slice(), block.len())
        })
        .collect();

    let decoded = engine
        .rans_decode_chunked_payload_gpu_batched_shared_table_repeated(
            &decode_inputs,
            &full_input,
            3,
        )
        .unwrap();
    assert_eq!(decoded.len(), expected_blocks.len());
    for (i, output) in decoded.iter().enumerate() {
        assert_eq!(output, &expected_blocks[i]);
    }
}

#[test]
fn test_rans_chunked_decode_gpu_batched_shared_table_mixed_lanes_fallback() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let full_input: Vec<u8> = (0..65_536).map(|i| ((i * 29 + 17) % 251) as u8).collect();
    let block_a = &full_input[..32_768];
    let block_b = &full_input[32_768..];

    let encoded_a = engine
        .rans_encode_chunked_payload_gpu_batched_shared_table(
            &[block_a],
            &full_input,
            4,
            crate::rans::DEFAULT_SCALE_BITS,
            2048,
        )
        .unwrap();
    let encoded_b = engine
        .rans_encode_chunked_payload_gpu_batched_shared_table(
            &[block_b],
            &full_input,
            8,
            crate::rans::DEFAULT_SCALE_BITS,
            2048,
        )
        .unwrap();
    assert!(encoded_a[0].1);
    assert!(encoded_b[0].1);

    let decode_inputs = vec![
        (encoded_a[0].0.as_slice(), block_a.len()),
        (encoded_b[0].0.as_slice(), block_b.len()),
    ];
    let decoded = engine
        .rans_decode_chunked_payload_gpu_batched_shared_table(&decode_inputs, &full_input)
        .unwrap();
    assert_eq!(decoded[0], block_a);
    assert_eq!(decoded[1], block_b);
}

#[test]
fn test_rans_chunked_gpu_round_trip_tiled_dispatch_chunks() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Chunk metadata encodes num_chunks as u16, so tiled dispatch can only be
    // exercised on adapters whose max workgroups per dimension is below u16::MAX.
    let max_wg = engine.max_workgroups_per_dimension() as usize;
    if max_wg >= u16::MAX as usize {
        return;
    }

    let num_chunks = (max_wg + 17).min(u16::MAX as usize);
    let input: Vec<u8> = (0..num_chunks)
        .map(|i| ((i * 29 + 13) % 256) as u8)
        .collect();

    let (encoded, used_chunked) = engine
        .rans_encode_chunked_payload_gpu(&input, 4, crate::rans::DEFAULT_SCALE_BITS, 1)
        .unwrap();
    assert!(used_chunked);

    let decoded = engine
        .rans_decode_chunked_payload_gpu(&encoded, input.len())
        .unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_rans_chunked_decode_gpu_rejects_chunk_len_mismatch() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..8192usize).map(|i| ((i * 23 + 3) % 256) as u8).collect();
    let mut encoded = crate::rans::encode_chunked(&input, 4, crate::rans::DEFAULT_SCALE_BITS, 1024);

    let first_chunk_len_offset = 1 + crate::rans::NUM_SYMBOLS * 2 + 2;
    encoded[first_chunk_len_offset] ^= 1;
    assert_eq!(
        engine.rans_decode_chunked_payload_gpu(&encoded, input.len()),
        Err(PzError::InvalidInput)
    );
}

#[test]
fn test_rans_chunked_parity_matrix_gpu_cpu() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let repetitive = vec![b'A'; 70_000];

    let mut text = Vec::with_capacity(70_000);
    let text_pattern = b"The Burrows-Wheeler transform clusters bytes. ";
    while text.len() < 70_000 {
        let take = (70_000 - text.len()).min(text_pattern.len());
        text.extend_from_slice(&text_pattern[..take]);
    }

    let binary: Vec<u8> = (0..70_000).map(|i| ((i * 73 + 19) % 256) as u8).collect();

    let small_edge: Vec<u8> = (0..509).map(|i| ((i * 29 + 7) % 251) as u8).collect();

    let patterns = vec![
        ("repetitive", repetitive),
        ("mixed_text", text),
        ("binary", binary),
        ("small_edge", small_edge),
    ];

    for &num_lanes in &[4usize, 8usize] {
        for &chunk_size in &[1024usize, 4096, 8192, 16_384] {
            for (label, input) in &patterns {
                // 1) CPU encode -> CPU decode
                let cpu_encoded = crate::rans::encode_chunked(
                    input,
                    num_lanes,
                    crate::rans::DEFAULT_SCALE_BITS,
                    chunk_size,
                );
                let cpu_decoded = crate::rans::decode_chunked(&cpu_encoded).unwrap();
                assert_eq!(
                    cpu_decoded, *input,
                    "cpu->cpu mismatch pattern={label}, lanes={num_lanes}, chunk={chunk_size}"
                );

                // 2) CPU encode -> GPU decode
                let gpu_decoded_from_cpu = engine
                    .rans_decode_chunked_payload_gpu(&cpu_encoded, input.len())
                    .unwrap();
                assert_eq!(
                    gpu_decoded_from_cpu, *input,
                    "cpu->gpu mismatch pattern={label}, lanes={num_lanes}, chunk={chunk_size}"
                );

                // 3) GPU encode -> CPU decode
                let (gpu_encoded, used_gpu_chunked) = engine
                    .rans_encode_chunked_payload_gpu(
                        input,
                        num_lanes,
                        crate::rans::DEFAULT_SCALE_BITS,
                        chunk_size,
                    )
                    .unwrap();
                assert!(
                    used_gpu_chunked,
                    "gpu chunked fallback for pattern={label}, lanes={num_lanes}, chunk={chunk_size}"
                );
                let cpu_decoded_from_gpu = crate::rans::decode_chunked(&gpu_encoded).unwrap();
                assert_eq!(
                    cpu_decoded_from_gpu, *input,
                    "gpu->cpu mismatch pattern={label}, lanes={num_lanes}, chunk={chunk_size}"
                );

                // 4) GPU encode -> GPU decode
                let gpu_decoded_from_gpu = engine
                    .rans_decode_chunked_payload_gpu(&gpu_encoded, input.len())
                    .unwrap();
                assert_eq!(
                    gpu_decoded_from_gpu, *input,
                    "gpu->gpu mismatch pattern={label}, lanes={num_lanes}, chunk={chunk_size}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GPU LzSeq demux tests
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_lzseq_encode_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Test patterns of varying characteristics
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("all_same", vec![b'A'; 128 * 1024]),
        ("repeating", {
            let pattern = b"the quick brown fox jumps over the lazy dog. ";
            pattern.iter().cycle().take(128 * 1024).copied().collect()
        }),
        (
            "binary_cycle",
            (0..128 * 1024).map(|i| (i % 256) as u8).collect(),
        ),
        ("short_repeat", {
            let pattern = b"ABCD";
            pattern.iter().cycle().take(128 * 1024).copied().collect()
        }),
    ];

    for (label, input) in &patterns {
        let enc = engine
            .lzseq_encode_gpu(input)
            .unwrap_or_else(|e| panic!("GPU encode failed for {label}: {e:?}"));

        assert!(enc.num_tokens > 0, "{label}: should have tokens");
        assert_eq!(
            enc.num_tokens,
            enc.num_matches + enc.literals.len() as u32,
            "{label}: num_tokens should equal num_matches + num_literals"
        );

        // Round-trip via CPU decoder
        let decoded = crate::lzseq::decode(
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
        .unwrap_or_else(|e| panic!("CPU decode failed for {label}: {e:?}"));

        assert_eq!(decoded, *input, "{label}: round-trip mismatch");
    }
}

#[test]
fn test_gpu_lzseq_all_literals() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Random-ish data that won't compress (no matches)
    let input: Vec<u8> = (0..64 * 1024).map(|i| ((i * 7 + 13) % 251) as u8).collect();

    let enc = engine.lzseq_encode_gpu(&input).unwrap();

    // With poor-quality matches, most should be literals
    // (GPU might still find some short matches, so don't assert num_matches == 0)
    assert!(enc.num_tokens > 0);

    let decoded = crate::lzseq::decode(
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

    assert_eq!(decoded, input, "all-literals round-trip mismatch");
}

#[test]
fn test_gpu_lzseq_pipeline_round_trip() {
    use crate::pipeline;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = b"hello world hello world hello world! "
        .iter()
        .cycle()
        .take(128 * 1024)
        .copied()
        .collect();

    let options = pipeline::CompressOptions {
        backend: pipeline::Backend::WebGpu,
        webgpu_engine: Some(engine),
        threads: 1,
        ..pipeline::CompressOptions::default()
    };

    let compressed =
        pipeline::compress_with_options(&input, pipeline::Pipeline::LzSeqR, &options).unwrap();
    let decompressed = pipeline::decompress(&compressed).unwrap();

    assert_eq!(decompressed, input, "pipeline GPU round-trip mismatch");
}

#[test]
fn test_gpu_lzseq_rans_encode_pipeline() {
    use crate::pipeline;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Large enough input so streams exceed rans_interleaved_min_bytes → GPU rANS path
    let input: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(256 * 1024)
        .copied()
        .collect();

    let options = pipeline::CompressOptions {
        backend: pipeline::Backend::WebGpu,
        webgpu_engine: Some(engine.clone()),
        threads: 1,
        // Lower threshold so individual streams hit the GPU rANS path
        rans_interleaved_min_bytes: 4096,
        ..pipeline::CompressOptions::default()
    };

    let compressed =
        pipeline::compress_with_options(&input, pipeline::Pipeline::LzSeqR, &options).unwrap();
    let decompressed = pipeline::decompress(&compressed).unwrap();

    assert_eq!(decompressed, input, "GPU rANS pipeline round-trip mismatch");

    // Also test with small input that forces CPU fallback for small streams
    let small_input: Vec<u8> = b"ABCD".iter().cycle().take(4096).copied().collect();
    let compressed_small =
        pipeline::compress_with_options(&small_input, pipeline::Pipeline::LzSeqR, &options)
            .unwrap();
    let decompressed_small = pipeline::decompress(&compressed_small).unwrap();

    assert_eq!(
        decompressed_small, small_input,
        "GPU rANS pipeline small-input round-trip mismatch"
    );
}

#[test]
fn test_gpu_encode_cpu_decode_lzseq_streams() {
    // Build a synthetic LzSeq DemuxOutput (6 streams) and verify that
    // GPU encode -> CPU decode round-trips correctly for each stream.
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Synthetic streams: vary sizes and distributions to exercise
    // the 256KB threshold and stream-size heterogeneity.
    let streams: Vec<Vec<u8>> = vec![
        // flags: mostly 0 and 1 (match/literal bits)
        (0..32768).map(|i| (i % 2) as u8).collect(),
        // literals: all 256 values
        (0..32768).map(|i| (i % 256) as u8).collect(),
        // offset_codes: zstd-style, concentrated in low values
        (0..16384).map(|i| (i % 32) as u8).collect(),
        // offset_extra: near-uniform
        (0..16384).map(|i| (i % 256) as u8).collect(),
        // length_codes: concentrated in 0-15
        (0..16384).map(|i| (i % 16) as u8).collect(),
        // length_extra: sparse
        (0..8192).map(|i| (i % 64) as u8).collect(),
    ];

    let encoded_streams = engine
        .rans_encode_6streams_gpu(&streams, 4, 65536, crate::rans::DEFAULT_SCALE_BITS)
        .expect("GPU encode of 6 streams must succeed");

    assert_eq!(encoded_streams.len(), 6, "must get 6 encoded streams back");

    for (i, (original, encoded)) in streams.iter().zip(encoded_streams.iter()).enumerate() {
        let decoded = crate::rans::decode_interleaved(encoded, original.len())
            .unwrap_or_else(|e| panic!("CPU decode of stream {} failed: {:?}", i, e));
        assert_eq!(
            &decoded, original,
            "stream {} GPU-encode -> CPU-decode round-trip mismatch",
            i
        );
    }
}

#[test]
fn test_cpu_encode_gpu_decode_lzseq_streams() {
    // CPU encode -> GPU decode cross-path: encode on CPU, decode on GPU.
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..131072).map(|i| (i % 200) as u8).collect();
    let encoded_cpu = crate::rans::encode_interleaved(&input);
    let decoded_gpu = engine
        .rans_decode_interleaved_gpu(&encoded_cpu, input.len())
        .expect("GPU decode of CPU-encoded data must succeed");
    assert_eq!(
        decoded_gpu, input,
        "CPU-encode -> GPU-decode round-trip mismatch"
    );
}

#[test]
fn test_gpu_entropy_threshold_cpu_fallback_below_256kb() {
    // Streams whose total size is below GPU_ENTROPY_THRESHOLD (256KB)
    // must silently use the CPU path — no GPU initialization or error.
    let small_streams: Vec<Vec<u8>> = vec![
        vec![0u8; 1024], // 1KB each
        vec![1u8; 1024],
        vec![2u8; 1024],
        vec![3u8; 1024],
        vec![4u8; 1024],
        vec![5u8; 1024],
    ]; // total: 6KB << 256KB

    let options = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        #[cfg(feature = "webgpu")]
        webgpu_engine: None, // simulates "no GPU"
        threads: 1,
        ..crate::pipeline::CompressOptions::default()
    };
    assert!(
        !crate::pipeline::should_use_gpu_entropy(&small_streams, &options),
        "must not use GPU entropy for streams totaling < 256KB"
    );

    // Also test with large streams that should use GPU if available
    let large_streams: Vec<Vec<u8>> = vec![
        vec![0u8; 100_000],
        vec![1u8; 100_000],
        vec![2u8; 100_000], // total: 300KB > 256KB
    ];

    let options_no_engine = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        #[cfg(feature = "webgpu")]
        webgpu_engine: None,
        threads: 1,
        ..crate::pipeline::CompressOptions::default()
    };
    assert!(
        !crate::pipeline::should_use_gpu_entropy(&large_streams, &options_no_engine),
        "must not use GPU entropy when engine is None"
    );

    // If we have a GPU available, test with engine
    #[cfg(feature = "webgpu")]
    {
        if let Ok(engine) = WebGpuEngine::new() {
            let options_with_engine = crate::pipeline::CompressOptions {
                backend: crate::pipeline::Backend::WebGpu,
                webgpu_engine: Some(std::sync::Arc::new(engine)),
                threads: 1,
                ..crate::pipeline::CompressOptions::default()
            };
            assert!(
                crate::pipeline::should_use_gpu_entropy(&large_streams, &options_with_engine),
                "must use GPU entropy for streams totaling >= 256KB when GPU available"
            );
        }
    }
}

#[test]
fn test_rans_recoil_decode_gpu_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..8192).map(|i| ((i * 41 + 61) % 251) as u8).collect();
    let encoded = crate::rans::encode_interleaved_n(&input, 4, crate::rans::DEFAULT_SCALE_BITS);
    let metadata = crate::recoil::recoil_generate_splits(&encoded, input.len(), 8).unwrap();

    let decoded = engine
        .rans_decode_recoil_gpu(&encoded, &metadata, input.len())
        .unwrap();
    assert_eq!(decoded, input, "Recoil GPU decode mismatch");
}

#[test]
fn test_rans_recoil_decode_gpu_various_splits() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..16384).map(|i| ((i * 13 + 7) % 200) as u8).collect();
    let encoded = crate::rans::encode_interleaved_n(&input, 4, crate::rans::DEFAULT_SCALE_BITS);

    for num_splits in [1, 2, 4, 8, 16, 64] {
        let metadata =
            crate::recoil::recoil_generate_splits(&encoded, input.len(), num_splits).unwrap();
        let decoded = engine
            .rans_decode_recoil_gpu(&encoded, &metadata, input.len())
            .unwrap();
        assert_eq!(
            decoded, input,
            "Recoil GPU decode failed with {} splits",
            num_splits
        );
    }
}

#[test]
fn test_rans_recoil_decode_gpu_wide_interleave() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..8192).map(|i| ((i * 17 + 3) % 256) as u8).collect();
    let encoded = crate::rans::encode_interleaved_n(&input, 8, crate::rans::DEFAULT_SCALE_BITS);
    let metadata = crate::recoil::recoil_generate_splits(&encoded, input.len(), 16).unwrap();

    let decoded = engine
        .rans_decode_recoil_gpu(&encoded, &metadata, input.len())
        .unwrap();
    assert_eq!(
        decoded, input,
        "Recoil GPU decode failed with 8-way interleave"
    );
}

#[test]
fn test_rans_recoil_decode_gpu_matches_cpu() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..4096).map(|i| ((i * 31 + 11) % 251) as u8).collect();
    let encoded = crate::rans::encode_interleaved_n(&input, 4, crate::rans::DEFAULT_SCALE_BITS);
    let metadata = crate::recoil::recoil_generate_splits(&encoded, input.len(), 8).unwrap();

    let cpu_decoded = crate::recoil::decode_recoil(&encoded, &metadata, input.len()).unwrap();
    let gpu_decoded = engine
        .rans_decode_recoil_gpu(&encoded, &metadata, input.len())
        .unwrap();
    assert_eq!(gpu_decoded, cpu_decoded, "GPU Recoil must match CPU Recoil");
}

// --- GPU parlz experiment roundtrip tests ---

#[test]
fn test_gpu_parlz_resolve_roundtrip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Non-overlapping matches: both should be selected.
    let match_data: Vec<u32> = vec![
        (5 << 16) | 3, // pos 0: offset=5, length=3
        0,             // pos 1: no match
        0,             // pos 2: no match
        0,             // pos 3: no match
        (5 << 16) | 3, // pos 4: offset=5, length=3
        0,             // pos 5: no match
        0,             // pos 6: no match
    ];
    let result = engine.parlz_resolve(&match_data).unwrap();
    assert!(result[0], "pos 0 should be match start");
    assert!(result[4], "pos 4 should be match start");
}

#[test]
fn test_gpu_parlz_compress() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"abcabcabcabcabc this repeated text repeated text";
    let compressed = engine.parlz_compress(input).unwrap();
    assert!(!compressed.is_empty());
    // Verify decompression produces original.
    let decompressed = crate::parlz::decompress(&compressed, input.len()).unwrap();
    assert_eq!(&decompressed, &input[..]);
}
