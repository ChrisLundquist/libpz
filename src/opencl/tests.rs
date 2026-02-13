use super::*;

#[test]
fn test_gpu_match_struct_size() {
    // GpuMatch must be 12 bytes to match the OpenCL kernel struct layout
    assert_eq!(std::mem::size_of::<GpuMatch>(), 12);
}

#[test]
fn test_dedupe_all_literals() {
    // Simulate GPU output where no matches were found (all literals)
    let input = b"abcdef";
    let gpu_matches: Vec<GpuMatch> = input
        .iter()
        .map(|&b| GpuMatch {
            offset: 0,
            length: 0,
            next: b,
            _pad: [0; 3],
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
fn test_dedupe_with_matches() {
    // Simulate GPU output: position 0 is literal 'a', positions 1-3 have
    // a match of length 3, etc.
    let input = b"abcabc";
    let gpu_matches = vec![
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'a',
            _pad: [0; 3],
        },
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'b',
            _pad: [0; 3],
        },
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'c',
            _pad: [0; 3],
        },
        GpuMatch {
            offset: 3,
            length: 2,
            next: b'c',
            _pad: [0; 3],
        },
        GpuMatch {
            offset: 3,
            length: 1,
            next: b'c',
            _pad: [0; 3],
        }, // overlapping, skipped
        GpuMatch {
            offset: 3,
            length: 0,
            next: b'c',
            _pad: [0; 3],
        }, // overlapping, skipped
    ];

    let result = dedupe_gpu_matches(&gpu_matches, input);
    // Position 0: literal 'a' -> advance 1
    // Position 1: literal 'b' -> advance 1
    // Position 2: literal 'c' -> advance 1
    // Position 3: match(3,2) + literal 'c' -> advance 3
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].next, b'a');
    assert_eq!(result[3].offset, 3);
    assert_eq!(result[3].length, 2);
}

#[test]
fn test_dedupe_empty() {
    let result = dedupe_gpu_matches(&[], &[]);
    assert!(result.is_empty());
}

#[test]
fn test_probe_devices_does_not_panic() {
    // This should never panic, even without OpenCL runtime
    let devices = probe_devices();
    // We can't assert a specific count since it depends on the environment
    let _ = devices;
}

#[test]
fn test_device_count_does_not_panic() {
    let count = device_count();
    let _ = count;
}

// Integration tests that require an actual OpenCL device.
// These are gated on the device being available at runtime.

#[test]
fn test_engine_creation() {
    // This test will pass if OpenCL is available, skip otherwise
    match OpenClEngine::new() {
        Ok(engine) => {
            assert!(!engine.device_name().is_empty());
            assert!(engine.max_work_group_size() > 0);
        }
        Err(PzError::Unsupported) => {
            // No OpenCL device available, that's fine
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[test]
fn test_gpu_lz77_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return, // skip
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world";
    let compressed = engine
        .lz77_compress(input, KernelVariant::Batch)
        .expect("GPU compression failed");

    let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");

    assert_eq!(&decompressed, input);
}

#[test]
fn test_gpu_lz77_per_position_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return, // skip
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let compressed = engine
        .lz77_compress(input, KernelVariant::PerPosition)
        .expect("GPU compression failed");

    let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");

    assert_eq!(&decompressed, &input[..]);
}

#[test]
fn test_gpu_lz77_empty_input() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let result = engine.lz77_compress(b"", KernelVariant::Batch).unwrap();
    assert!(result.is_empty());
}

// --- find_matches_to_device tests ---

#[test]
fn test_find_matches_to_device_matches_find_matches() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world";

    // Direct path: find_matches downloads and dedupes in one call
    let direct = engine
        .find_matches(input, KernelVariant::HashTable)
        .unwrap();

    // Device path: keep on GPU, then download
    let match_buf = engine
        .find_matches_to_device(input, KernelVariant::HashTable)
        .unwrap();
    assert_eq!(match_buf.input_len(), input.len());
    let device = engine.download_and_dedupe(&match_buf, input).unwrap();

    // Both paths should produce identical match sequences
    assert_eq!(direct.len(), device.len(), "match count differs");
    for (i, (d, v)) in direct.iter().zip(device.iter()).enumerate() {
        assert_eq!(d.offset, v.offset, "offset mismatch at match {}", i);
        assert_eq!(d.length, v.length, "length mismatch at match {}", i);
        assert_eq!(d.next, v.next, "next mismatch at match {}", i);
    }
}

#[test]
fn test_find_matches_to_device_empty() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let match_buf = engine
        .find_matches_to_device(b"", KernelVariant::HashTable)
        .unwrap();
    assert_eq!(match_buf.input_len(), 0);

    let matches = engine.download_and_dedupe(&match_buf, b"").unwrap();
    assert!(matches.is_empty());
}

#[test]
fn test_find_matches_to_device_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let match_buf = engine
        .find_matches_to_device(input, KernelVariant::Batch)
        .unwrap();
    let matches = engine.download_and_dedupe(&match_buf, input).unwrap();

    // Serialize matches and verify LZ77 round-trip
    let mut compressed = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
    for m in &matches {
        compressed.extend_from_slice(&m.to_bytes());
    }
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(&decompressed, &input[..]);
}

// --- Hash-table LZ77 GPU tests ---

#[test]
fn test_gpu_lz77_hash_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world";
    let compressed = engine
        .lz77_compress(input, KernelVariant::HashTable)
        .expect("GPU hash compression failed");

    let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
    assert_eq!(&decompressed, input);
}

#[test]
fn test_gpu_lz77_hash_round_trip_longer() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let compressed = engine
        .lz77_compress(input, KernelVariant::HashTable)
        .expect("GPU hash compression failed");

    let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
    assert_eq!(&decompressed, &input[..]);
}

#[test]
fn test_gpu_lz77_hash_round_trip_large() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let pattern = b"Hello, World! This is a test pattern. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = engine
        .lz77_compress(&input, KernelVariant::HashTable)
        .expect("GPU hash compression failed");

    let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
    assert_eq!(decompressed, input);
}

#[test]
fn test_gpu_lz77_hash_no_matches() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"abcdefgh";
    let compressed = engine
        .lz77_compress(input, KernelVariant::HashTable)
        .expect("GPU hash compression failed");

    let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
    assert_eq!(&decompressed, input);
}

#[test]
fn test_gpu_lz77_hash_binary_data() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
    let compressed = engine
        .lz77_compress(&input, KernelVariant::HashTable)
        .expect("GPU hash compression failed");

    let decompressed = crate::lz77::decompress(&compressed).expect("decompression failed");
    assert_eq!(decompressed, input);
}

// --- BWT GPU tests ---

#[test]
fn test_gpu_bwt_banana() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"banana";
    let gpu_result = engine.bwt_encode(input).unwrap();
    let cpu_result = crate::bwt::encode(input).unwrap();

    assert_eq!(gpu_result.data, cpu_result.data, "BWT data mismatch");
    assert_eq!(
        gpu_result.primary_index, cpu_result.primary_index,
        "primary_index mismatch"
    );

    // Round-trip through CPU decode
    let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_bwt_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    // GPU uses circular prefix-doubling (like naive SA) which may produce
    // a different rotation order than CPU SA-IS for periodic inputs, but
    // both are valid BWTs that round-trip correctly.
    let input = b"hello world hello world hello world";
    let gpu_result = engine.bwt_encode(input).unwrap();
    let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_bwt_binary_data() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
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
fn test_gpu_bwt_medium_sizes() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    // Test sizes that cross the multi-workgroup boundary.
    // Uses round-trip verification since the GPU (circular prefix-doubling)
    // may order identical rotations differently from CPU SA-IS on periodic inputs.
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
fn test_gpu_bwt_single_byte() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"x";
    let gpu_result = engine.bwt_encode(input).unwrap();
    assert_eq!(gpu_result.data, vec![b'x']);
    assert_eq!(gpu_result.primary_index, 0);
}

#[test]
fn test_gpu_bwt_all_same() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = vec![b'a'; 64];
    let gpu_result = engine.bwt_encode(&input).unwrap();
    let decoded = crate::bwt::decode(&gpu_result.data, gpu_result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_bwt_large_structured() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    // 4KB of structured data — enough to exercise multi-level prefix sum
    // with small workgroup sizes and verify the GPU rank assignment pipeline.
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
fn test_gpu_candidate_struct_size() {
    // GpuCandidate must be 4 bytes to match the OpenCL kernel struct layout
    assert_eq!(std::mem::size_of::<GpuCandidate>(), 4);
}

#[test]
fn test_gpu_topk_empty_input() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let table = engine.find_topk_matches(b"").unwrap();
    assert_eq!(table.input_len, 0);
}

#[test]
fn test_gpu_topk_produces_valid_candidates() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
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
fn test_gpu_topk_optimal_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let table = engine.find_topk_matches(input).unwrap();
    let compressed = crate::optimal::compress_optimal_with_table(input, &table).unwrap();
    let decompressed = crate::lz77::decompress(&compressed).unwrap();
    assert_eq!(&decompressed, &input[..]);
}

#[test]
fn test_gpu_topk_optimal_large_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
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

// --- DeviceBuf tests ---

#[test]
fn test_device_buf_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let data = b"hello world this is a test of device buffers";
    let device_buf = DeviceBuf::from_host(&engine, data).unwrap();
    assert_eq!(device_buf.len(), data.len());
    assert!(!device_buf.is_empty());

    let host_data = device_buf.read_to_host(&engine).unwrap();
    assert_eq!(&host_data, data);
}

#[test]
fn test_device_buf_empty() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let device_buf = DeviceBuf::from_host(&engine, &[]).unwrap();
    assert_eq!(device_buf.len(), 0);
    assert!(device_buf.is_empty());

    let host_data = device_buf.read_to_host(&engine).unwrap();
    assert!(host_data.is_empty());
}

#[test]
fn test_byte_histogram_on_device_matches_host() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"aabbccdd hello world aabbccdd";

    // Host-upload path
    let hist_host = engine.byte_histogram(input).unwrap();

    // Device-buffer path
    let device_buf = DeviceBuf::from_host(&engine, input).unwrap();
    let hist_device = engine.byte_histogram_on_device(&device_buf).unwrap();

    assert_eq!(
        hist_host, hist_device,
        "histogram mismatch between host and device paths"
    );
}

#[test]
fn test_byte_histogram_on_device_empty() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let device_buf = DeviceBuf::from_host(&engine, &[]).unwrap();
    let hist = engine.byte_histogram_on_device(&device_buf).unwrap();
    assert!(hist.iter().all(|&c| c == 0));
}

// --- GPU Huffman encoding tests ---

#[test]
fn test_gpu_byte_histogram() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"aabbccdd";
    let hist = engine.byte_histogram(input).unwrap();
    assert_eq!(hist[b'a' as usize], 2);
    assert_eq!(hist[b'b' as usize], 2);
    assert_eq!(hist[b'c' as usize], 2);
    assert_eq!(hist[b'd' as usize], 2);
    assert_eq!(hist[0], 0);
}

#[test]
fn test_gpu_byte_histogram_empty() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let hist = engine.byte_histogram(&[]).unwrap();
    assert!(hist.iter().all(|&c| c == 0));
}

#[test]
fn test_gpu_huffman_encode_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world!";
    let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

    // Build the packed LUT: (bits << 24) | codeword
    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    let (gpu_encoded, gpu_bits) = engine.huffman_encode(input, &code_lut).unwrap();
    let (cpu_encoded, cpu_bits) = tree.encode(input).unwrap();

    assert_eq!(gpu_bits, cpu_bits, "bit counts differ");
    // The byte representations should match
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
fn test_gpu_huffman_encode_larger() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
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
fn test_gpu_is_cpu_device() {
    // Just verify the method doesn't panic
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };
    // The value depends on hardware — just check it returns a bool
    let _is_cpu = engine.is_cpu_device();
}

// --- GPU prefix sum tests ---

#[test]
fn test_gpu_prefix_sum_small() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = vec![1u32, 2, 3, 4, 5];
    let n = input.len();

    let mut buf = unsafe {
        Buffer::<cl_uint>::create(&engine.context, CL_MEM_READ_WRITE, n, ptr::null_mut()).unwrap()
    };
    unsafe {
        engine
            .queue
            .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, &input, &[])
            .unwrap()
            .wait()
            .unwrap();
    }

    engine.prefix_sum_gpu(&mut buf, n).unwrap();

    let mut result = vec![0u32; n];
    unsafe {
        engine
            .queue
            .enqueue_read_buffer(&buf, CL_BLOCKING, 0, &mut result, &[])
            .unwrap()
            .wait()
            .unwrap();
    }

    // Exclusive prefix sum: [0, 1, 3, 6, 10]
    assert_eq!(result, vec![0, 1, 3, 6, 10]);
}

#[test]
fn test_gpu_prefix_sum_large() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    // Large enough to require multi-level scan
    let n = 2048;
    let input: Vec<u32> = (0..n as u32).map(|i| (i % 10) + 1).collect();

    let mut buf = unsafe {
        Buffer::<cl_uint>::create(&engine.context, CL_MEM_READ_WRITE, n, ptr::null_mut()).unwrap()
    };
    unsafe {
        engine
            .queue
            .enqueue_write_buffer(&mut buf, CL_BLOCKING, 0, &input, &[])
            .unwrap()
            .wait()
            .unwrap();
    }

    engine.prefix_sum_gpu(&mut buf, n).unwrap();

    let mut result = vec![0u32; n];
    unsafe {
        engine
            .queue
            .enqueue_read_buffer(&buf, CL_BLOCKING, 0, &mut result, &[])
            .unwrap()
            .wait()
            .unwrap();
    }

    // Verify against CPU prefix sum
    let mut expected = vec![0u32; n];
    let mut sum: u64 = 0;
    for i in 0..n {
        expected[i] = sum as u32;
        sum += input[i] as u64;
    }

    assert_eq!(result, expected);
}

// --- GPU Huffman on-device tests ---

#[test]
fn test_gpu_huffman_encode_on_device_matches_gpu_scan() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"hello world hello world hello world!";
    let tree = crate::huffman::HuffmanTree::from_data(input).unwrap();

    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    // Host-upload path
    let (scan_encoded, scan_bits) = engine.huffman_encode_gpu_scan(input, &code_lut).unwrap();

    // Device-buffer path
    let device_buf = DeviceBuf::from_host(&engine, input).unwrap();
    let (device_encoded, device_bits) = engine
        .huffman_encode_on_device(&device_buf, &code_lut)
        .unwrap();

    assert_eq!(scan_bits, device_bits, "bit counts differ");
    assert_eq!(scan_encoded, device_encoded, "encoded data differs");

    // Verify round-trip
    let decoded = tree.decode(&device_encoded, device_bits).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_gpu_huffman_encode_on_device_larger() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
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

    let device_buf = DeviceBuf::from_host(&engine, &input).unwrap();
    let (device_encoded, device_bits) = engine
        .huffman_encode_on_device(&device_buf, &code_lut)
        .unwrap();

    let decoded = tree.decode(&device_encoded, device_bits).unwrap();
    assert_eq!(decoded, input);
}

// --- GPU Huffman with GPU scan tests ---

#[test]
fn test_gpu_huffman_encode_gpu_scan_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
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

// --- GPU Deflate pipeline round-trip tests (modular stage path) ---

#[test]
fn test_gpu_deflate_pipeline_round_trip() {
    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    gpu_pipeline_round_trip(input, crate::pipeline::Pipeline::Deflate);
}

#[test]
fn test_gpu_deflate_pipeline_larger() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Deflate);
}

#[test]
fn test_gpu_deflate_pipeline_binary() {
    // Binary data with repeating patterns — exercises all byte values
    let mut input = Vec::new();
    for i in 0u8..=255 {
        for _ in 0..40 {
            input.push(i);
        }
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Deflate);
}

// --- Modular GPU pipeline composition tests ---

/// Helper: compress/decompress via the pipeline API with OpenCL backend.
fn gpu_pipeline_round_trip(input: &[u8], pipeline: crate::pipeline::Pipeline) {
    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let options = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::OpenCl,
        opencl_engine: Some(engine),
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
    // GPU LZ77 → GPU Huffman (modular stage path, not monolithic deflate_chained)
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Deflate);
}

#[test]
fn test_gpu_lz77_cpu_rans_round_trip() {
    // GPU LZ77 → CPU rANS (previously impossible without modular stages)
    let pattern = b"Hello, World! This is a test pattern for GPU+CPU composition. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Lzr);
}

#[test]
fn test_gpu_lz77_cpu_fse_round_trip() {
    // GPU LZ77 → CPU FSE (previously impossible without modular stages)
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

#[test]
fn test_gpu_rans_decode_interleaved() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let pattern = b"abracadabra alakazam abracadabra ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }

    let encoded = crate::rans::encode_interleaved(&input);
    let cpu_decoded = crate::rans::decode_interleaved(&encoded, input.len()).unwrap();
    assert_eq!(cpu_decoded, input);

    let gpu_decoded = engine
        .rans_decode_interleaved(&encoded, input.len())
        .unwrap();
    assert_eq!(
        gpu_decoded, input,
        "GPU rANS decode should match CPU decode"
    );
}

#[test]
fn test_gpu_rans_decode_various_interleave() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let pattern = b"the quick brown fox jumps over the lazy dog ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    // Test with different interleave counts and scale bits
    for (num_states, scale_bits) in [(4, 12), (8, 11), (16, 10), (32, 12)] {
        let encoded = crate::rans::encode_interleaved_n(&input, num_states, scale_bits);
        let gpu_decoded = engine
            .rans_decode_interleaved(&encoded, input.len())
            .unwrap();
        assert_eq!(
            gpu_decoded, input,
            "mismatch with num_states={num_states}, scale_bits={scale_bits}"
        );
    }
}

#[test]
fn test_gpu_lz77_block_decompress_small() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    // Small repetitive input — fits in one block
    let input = b"abcabcabcabcabcabcabcabc";
    let (block_data, block_meta) = crate::opencl::lz77::lz77_compress_blocks(input, 1024).unwrap();

    let gpu_result = engine
        .lz77_decompress_blocks(&block_data, &block_meta, 32)
        .unwrap();
    assert_eq!(gpu_result, input);
}

#[test]
fn test_gpu_lz77_block_decompress_multiblock() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    // Larger input split into multiple 4KB blocks
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..400 {
        input.extend_from_slice(pattern);
    }

    let (block_data, block_meta) = crate::opencl::lz77::lz77_compress_blocks(&input, 4096).unwrap();
    assert!(block_meta.len() > 1, "should have multiple blocks");

    let gpu_result = engine
        .lz77_decompress_blocks(&block_data, &block_meta, 32)
        .unwrap();
    assert_eq!(gpu_result, input);
}

#[test]
fn test_gpu_lz77_block_decompress_various_threads() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let pattern = b"hello world compression test data ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    let (block_data, block_meta) = crate::opencl::lz77::lz77_compress_blocks(&input, 2048).unwrap();

    // Test with different cooperative thread counts
    for threads in [1, 8, 32, 64] {
        let result = engine
            .lz77_decompress_blocks(&block_data, &block_meta, threads)
            .unwrap();
        assert_eq!(result, input, "mismatch with cooperative_threads={threads}");
    }
}

#[test]
fn test_gpu_rans_decode_interleaved_blocks() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    // Create test data and split into independent blocks
    let pattern = b"abracadabra alakazam ";
    let mut input = Vec::new();
    for _ in 0..500 {
        input.extend_from_slice(pattern);
    }

    let block_size = 2048;
    let k = 8;
    let scale_bits = 11u8;

    // Encode all blocks with a shared frequency table (required for GPU batch decode)
    let encoded_blocks =
        crate::opencl::rans::rans_encode_blocks(&input, block_size, k, scale_bits).unwrap();

    // Verify CPU decode of each block works
    let mut cpu_result = Vec::new();
    for (enc, orig_len) in &encoded_blocks {
        let decoded = crate::rans::decode_interleaved(enc, *orig_len).unwrap();
        cpu_result.extend_from_slice(&decoded);
    }
    assert_eq!(
        cpu_result, input,
        "CPU round-trip with shared freq table failed"
    );

    // Build the encoded_blocks slice for the GPU method
    let block_refs: Vec<(&[u8], usize)> = encoded_blocks
        .iter()
        .map(|(enc, len)| (enc.as_slice(), *len))
        .collect();

    let gpu_result = engine.rans_decode_interleaved_blocks(&block_refs).unwrap();
    assert_eq!(
        gpu_result, input,
        "multi-block GPU rANS decode should match original"
    );
}

#[test]
fn test_gpu_combined_rans_lz77_pipeline() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    // Create repetitive test data that compresses well with LZ77.
    // Use a larger corpus so LZ77 blocks have enough match data to avoid
    // rANS encoder buffer underflow on small blocks.
    let pattern = b"the quick brown fox jumps over the lazy dog and then repeats ";
    let mut input = Vec::new();
    for _ in 0..2000 {
        input.extend_from_slice(pattern);
    }

    // Larger blocks → each block's LZ77 match data is large enough for
    // rANS to work correctly (avoids per-lane buffer underflow).
    let block_size = 16384;

    // Stage 1: CPU LZ77 independent-block compression
    let (block_data, block_meta) =
        crate::opencl::lz77::lz77_compress_blocks(&input, block_size).unwrap();

    // Stage 2: CPU rANS encode with shared freq table (for GPU batched decode)
    // Use 4 interleaved states and scale_bits=11 for safe encoding of binary match data
    let match_slices: Vec<&[u8]> = block_meta
        .iter()
        .map(|&(offset, num_matches, _)| {
            let end = offset + num_matches * 5;
            &block_data[offset..end]
        })
        .collect();
    let rans_encoded = crate::opencl::rans::rans_encode_block_slices(&match_slices, 4, 11).unwrap();

    // Stage 3: GPU rANS batched decode (all blocks in one kernel launch)
    let block_refs: Vec<(&[u8], usize)> = rans_encoded
        .iter()
        .map(|(enc, len)| (enc.as_slice(), *len))
        .collect();
    let all_lz77 = engine.rans_decode_interleaved_blocks(&block_refs).unwrap();

    // Verify rANS decode matches the original LZ77 match data
    let mut expected_lz77 = Vec::new();
    for &(offset, num_matches, _) in &block_meta {
        let end = offset + num_matches * 5;
        expected_lz77.extend_from_slice(&block_data[offset..end]);
    }
    assert_eq!(
        all_lz77, expected_lz77,
        "GPU rANS batched decode should reproduce LZ77 match data"
    );

    // Stage 4: GPU LZ77 block-parallel decompress
    // Rebuild metadata for the decoded LZ77 data (contiguous, not offset-based)
    let mut decoded_meta = Vec::new();
    let mut lz77_offset = 0usize;
    for &(_, num_matches, decompressed_size) in &block_meta {
        decoded_meta.push((lz77_offset, num_matches, decompressed_size));
        lz77_offset += num_matches * 5;
    }

    let result = engine
        .lz77_decompress_blocks(&all_lz77, &decoded_meta, 32)
        .unwrap();
    assert_eq!(
        result, input,
        "GPU combined rANS+LZ77 pipeline should reproduce original data"
    );
}

#[test]
fn test_gpu_fse_decode_blocks() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    // Create test data
    let pattern = b"abracadabra alakazam open sesame ";
    let mut input = Vec::new();
    for _ in 0..500 {
        input.extend_from_slice(pattern);
    }

    let block_size = 2048;
    let num_streams = 4;
    let accuracy_log = 10u8;

    // Encode all blocks with shared frequency table
    let slices: Vec<&[u8]> = input.chunks(block_size).collect();
    let encoded_blocks =
        crate::opencl::fse::fse_encode_block_slices(&slices, num_streams, accuracy_log).unwrap();

    // Verify CPU decode of each block
    let mut cpu_result = Vec::new();
    for (enc, orig_len) in &encoded_blocks {
        let decoded = crate::fse::decode_interleaved(enc, *orig_len).unwrap();
        cpu_result.extend_from_slice(&decoded);
    }
    assert_eq!(
        cpu_result, input,
        "CPU FSE round-trip with shared freq table failed"
    );

    // GPU batched decode
    let block_refs: Vec<(&[u8], usize)> = encoded_blocks
        .iter()
        .map(|(enc, len)| (enc.as_slice(), *len))
        .collect();
    let gpu_result = engine.fse_decode_blocks(&block_refs).unwrap();
    assert_eq!(
        gpu_result, input,
        "GPU multi-block FSE decode should match original"
    );
}
