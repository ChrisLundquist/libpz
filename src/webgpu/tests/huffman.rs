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
