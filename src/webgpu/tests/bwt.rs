use super::super::*;

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
