use super::super::*;

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
