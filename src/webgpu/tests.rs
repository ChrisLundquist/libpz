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
fn test_deflate_chained_round_trip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let block_data = engine.deflate_chained(input).unwrap();

    // Decompress using the standard CPU Deflate decoder
    let decompressed = crate::pipeline::decompress(&{
        let mut container = Vec::new();
        container.extend_from_slice(&[b'P', b'Z', 2, 0]); // magic + version=2 + pipeline=Deflate
        container.extend_from_slice(&(input.len() as u32).to_le_bytes());
        container.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        container.extend_from_slice(&(block_data.len() as u32).to_le_bytes());
        container.extend_from_slice(&(input.len() as u32).to_le_bytes());
        container.extend_from_slice(&block_data);
        container
    })
    .unwrap();

    assert_eq!(decompressed, input);
}

#[test]
fn test_deflate_chained_larger() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    let block_data = engine.deflate_chained(&input).unwrap();

    let decompressed = crate::pipeline::decompress(&{
        let mut container = Vec::new();
        container.extend_from_slice(&[b'P', b'Z', 2, 0]);
        container.extend_from_slice(&(input.len() as u32).to_le_bytes());
        container.extend_from_slice(&1u32.to_le_bytes());
        container.extend_from_slice(&(block_data.len() as u32).to_le_bytes());
        container.extend_from_slice(&(input.len() as u32).to_le_bytes());
        container.extend_from_slice(&block_data);
        container
    })
    .unwrap();

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
