use super::super::*;

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
