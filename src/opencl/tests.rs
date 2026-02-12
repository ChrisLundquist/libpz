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

// --- GPU chained Deflate tests ---

#[test]
fn test_gpu_deflate_chained_round_trip() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox.";
    let block_data = engine.deflate_chained(input).unwrap();

    // Decompress using the standard CPU Deflate decoder
    let decompressed = crate::pipeline::decompress(&{
        // Build a proper V2 PZ container around the block data
        let mut container = Vec::new();
        container.extend_from_slice(&[b'P', b'Z', 2, 0]); // magic + version=2 + pipeline=Deflate
        container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original length
        container.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        container.extend_from_slice(&(block_data.len() as u32).to_le_bytes()); // compressed_len
        container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original_len
        container.extend_from_slice(&block_data);
        container
    })
    .unwrap();

    assert_eq!(decompressed, input);
}

#[test]
fn test_gpu_deflate_chained_larger() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    let block_data = engine.deflate_chained(&input).unwrap();

    let decompressed = crate::pipeline::decompress(&{
        let mut container = Vec::new();
        container.extend_from_slice(&[b'P', b'Z', 2, 0]); // magic + version=2 + pipeline=Deflate
        container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original length
        container.extend_from_slice(&1u32.to_le_bytes()); // num_blocks = 1
        container.extend_from_slice(&(block_data.len() as u32).to_le_bytes()); // compressed_len
        container.extend_from_slice(&(input.len() as u32).to_le_bytes()); // original_len
        container.extend_from_slice(&block_data);
        container
    })
    .unwrap();

    assert_eq!(decompressed, input);
}

#[test]
fn test_gpu_deflate_chained_binary() {
    let engine = match OpenClEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    // Binary data with repeating patterns — exercises all byte values
    let mut input = Vec::new();
    for i in 0u8..=255 {
        for _ in 0..40 {
            input.push(i);
        }
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
