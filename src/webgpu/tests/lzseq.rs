use super::super::*;

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
