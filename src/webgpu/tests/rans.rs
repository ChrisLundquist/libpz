use super::super::*;

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

// --- rANS recoil decode GPU tests ---

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
