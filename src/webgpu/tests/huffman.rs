use super::super::*;

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

// --- GPU Huffman sync-point decode tests ---

#[test]
fn test_huffman_decode_gpu_roundtrip() {
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

    let mut tree = crate::huffman::HuffmanTree::from_data(&input).unwrap();
    tree.canonicalize();

    let result = tree.encode_with_sync_points(&input, 1024).unwrap();
    let lut = tree.build_gpu_decode_lut();
    let lut_array: &[u32; 4096] = lut.as_slice().try_into().unwrap();

    let decoded = engine
        .huffman_decode_gpu(
            &result.data,
            result.total_bits,
            lut_array,
            &result.sync_points,
            input.len(),
        )
        .unwrap();

    assert_eq!(decoded, input, "GPU Huffman decode round-trip mismatch");
}

#[test]
fn test_huffman_decode_gpu_vs_cpu() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let pattern = b"abcdefghijklmnopqrstuvwxyz0123456789 ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    let mut tree = crate::huffman::HuffmanTree::from_data(&input).unwrap();
    tree.canonicalize();

    let result = tree.encode_with_sync_points(&input, 512).unwrap();
    let lut = tree.build_gpu_decode_lut();
    let lut_array: &[u32; 4096] = lut.as_slice().try_into().unwrap();

    // CPU tiled decode.
    let cpu_decoded = tree
        .decode_tiled(&result.data, result.total_bits, &result.sync_points)
        .unwrap();

    // GPU decode.
    let gpu_decoded = engine
        .huffman_decode_gpu(
            &result.data,
            result.total_bits,
            lut_array,
            &result.sync_points,
            input.len(),
        )
        .unwrap();

    assert_eq!(
        gpu_decoded, cpu_decoded,
        "GPU decode differs from CPU tiled decode"
    );
    assert_eq!(gpu_decoded, input, "GPU decode differs from original input");
}

#[test]
fn test_huffman_decode_gpu_single_symbol() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let input = vec![42u8; 4096];
    let mut tree = crate::huffman::HuffmanTree::from_data(&input).unwrap();
    tree.canonicalize();

    let result = tree.encode_with_sync_points(&input, 1024).unwrap();
    let lut = tree.build_gpu_decode_lut();
    let lut_array: &[u32; 4096] = lut.as_slice().try_into().unwrap();

    let decoded = engine
        .huffman_decode_gpu(
            &result.data,
            result.total_bits,
            lut_array,
            &result.sync_points,
            input.len(),
        )
        .unwrap();

    assert_eq!(decoded, input, "single-symbol GPU decode mismatch");
}

#[test]
fn test_gpulz_decompress_gpu_roundtrip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
    }

    let compressed = crate::gpulz::compress_block(&input).unwrap();

    // CPU decompress.
    let cpu_decoded = crate::gpulz::decompress_block(&compressed, input.len()).unwrap();
    assert_eq!(cpu_decoded, input, "CPU decompress mismatch");

    // GPU decompress.
    let gpu_decoded =
        crate::gpulz::decompress_block_gpu(&engine, &compressed, input.len()).unwrap();
    assert_eq!(gpu_decoded, input, "GPU decompress mismatch");
    assert_eq!(gpu_decoded, cpu_decoded, "GPU differs from CPU");
}

#[test]
fn test_gpulz_multiblock_gpu_roundtrip() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Create 4 blocks of different content.
    let block_size = 4096;
    let mut blocks_data: Vec<Vec<u8>> = Vec::new();
    for i in 0..4u8 {
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(
                &format!("block {i}: the quick brown fox jumps over the lazy dog. ").into_bytes(),
            );
        }
        data.truncate(block_size);
        blocks_data.push(data);
    }

    // Compress each block.
    let compressed: Vec<Vec<u8>> = blocks_data
        .iter()
        .map(|d| crate::gpulz::compress_block(d).unwrap())
        .collect();

    // Multi-block GPU decompress.
    let block_refs: Vec<(&[u8], usize)> = compressed
        .iter()
        .map(|c| (c.as_slice(), block_size))
        .collect();
    let (gpu_results, _timings) =
        crate::gpulz::decompress_blocks_gpu(&engine, &block_refs).unwrap();

    // Verify each block.
    for (i, (gpu_out, orig)) in gpu_results.iter().zip(blocks_data.iter()).enumerate() {
        assert_eq!(gpu_out, orig, "block {i} GPU decompress mismatch");
    }
}

#[test]
fn test_merged_decode_matches_batched() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Create multiple distinct streams.
    let patterns: &[&[u8]] = &[
        b"The quick brown fox jumps over the lazy dog. ",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
        b"aaabbbcccdddeeefffggghhhiiijjjkkklll ",
    ];

    let mut all_streams = Vec::new();
    for pattern in patterns {
        let data: Vec<u8> = pattern.iter().copied().cycle().take(4096).collect();
        let mut tree = crate::huffman::HuffmanTree::from_data(&data).unwrap();
        tree.canonicalize();
        let result = tree.encode_with_sync_points(&data, 1024).unwrap();
        let lut = tree.build_gpu_decode_lut();
        let lut_box: Box<[u32; 4096]> = lut.into_boxed_slice().try_into().unwrap();
        all_streams.push(HuffmanDecodeStream {
            huffman_data: Box::leak(result.data.into_boxed_slice()),
            decode_lut: lut_box,
            sync_points: result.sync_points,
            output_len: data.len(),
        });
    }

    // Decode with both approaches.
    let batched_results = engine.huffman_decode_gpu_batched(&all_streams).unwrap();
    let merged_results = engine.huffman_decode_gpu_merged(&all_streams).unwrap();

    assert_eq!(batched_results.len(), merged_results.len());
    for (i, (batched, merged)) in batched_results
        .iter()
        .zip(merged_results.iter())
        .enumerate()
    {
        assert_eq!(
            batched,
            merged,
            "stream {i}: merged decode doesn't match batched decode (len {} vs {})",
            batched.len(),
            merged.len(),
        );
    }
}

#[test]
fn test_hybrid_decode_matches_batched() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Create multiple distinct streams (same as merged test).
    let patterns: &[&[u8]] = &[
        b"The quick brown fox jumps over the lazy dog. ",
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
        b"aaabbbcccdddeeefffggghhhiiijjjkkklll ",
    ];

    let mut all_streams = Vec::new();
    for pattern in patterns {
        let data: Vec<u8> = pattern.iter().copied().cycle().take(4096).collect();
        let mut tree = crate::huffman::HuffmanTree::from_data(&data).unwrap();
        tree.canonicalize();
        let result = tree.encode_with_sync_points(&data, 1024).unwrap();
        let lut = tree.build_gpu_decode_lut();
        let lut_box: Box<[u32; 4096]> = lut.into_boxed_slice().try_into().unwrap();
        all_streams.push(HuffmanDecodeStream {
            huffman_data: Box::leak(result.data.into_boxed_slice()),
            decode_lut: lut_box,
            sync_points: result.sync_points,
            output_len: data.len(),
        });
    }

    // Decode with both approaches.
    let batched_results = engine.huffman_decode_gpu_batched(&all_streams).unwrap();
    let hybrid_results = engine.huffman_decode_gpu_hybrid(&all_streams).unwrap();

    assert_eq!(batched_results.len(), hybrid_results.len());
    for (i, (batched, hybrid)) in batched_results
        .iter()
        .zip(hybrid_results.iter())
        .enumerate()
    {
        assert_eq!(
            batched,
            hybrid,
            "stream {i}: hybrid decode doesn't match batched decode (len {} vs {})",
            batched.len(),
            hybrid.len(),
        );
    }
}
