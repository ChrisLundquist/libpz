use super::super::*;

// --- Modular GPU pipeline round-trip tests ---

fn gpu_pipeline_round_trip(input: &[u8], pipeline: crate::pipeline::Pipeline) {
    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let options = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        webgpu_engine: Some(engine),
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
fn test_gpu_lz77_cpu_rans_round_trip() {
    // GPU LZ77 → CPU rANS
    let pattern = b"Hello, World! This is a test pattern for GPU+CPU composition. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::LzSeqR);
}

#[test]
fn test_gpu_lz77_cpu_fse_round_trip() {
    // GPU LZ77 → CPU FSE
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
fn test_gpu_lzfi_pipeline_webgpu_round_trip() {
    // Full Lzfi pipeline round-trip with WebGPU backend
    let pattern = b"Hello, World! This is a test pattern for GPU Lzfi pipeline. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    gpu_pipeline_round_trip(&input, crate::pipeline::Pipeline::Lzfi);
}

#[test]
fn test_gpu_entropy_threshold_cpu_fallback_below_256kb() {
    // Streams whose total size is below GPU_ENTROPY_THRESHOLD (256KB)
    // must silently use the CPU path — no GPU initialization or error.
    let small_streams: Vec<Vec<u8>> = vec![
        vec![0u8; 1024], // 1KB each
        vec![1u8; 1024],
        vec![2u8; 1024],
        vec![3u8; 1024],
        vec![4u8; 1024],
        vec![5u8; 1024],
    ]; // total: 6KB << 256KB

    let options = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        #[cfg(feature = "webgpu")]
        webgpu_engine: None, // simulates "no GPU"
        threads: 1,
        ..crate::pipeline::CompressOptions::default()
    };
    assert!(
        !crate::pipeline::should_use_gpu_entropy(&small_streams, &options),
        "must not use GPU entropy for streams totaling < 256KB"
    );

    // Also test with large streams that should use GPU if available
    let large_streams: Vec<Vec<u8>> = vec![
        vec![0u8; 100_000],
        vec![1u8; 100_000],
        vec![2u8; 100_000], // total: 300KB > 256KB
    ];

    let options_no_engine = crate::pipeline::CompressOptions {
        backend: crate::pipeline::Backend::WebGpu,
        #[cfg(feature = "webgpu")]
        webgpu_engine: None,
        threads: 1,
        ..crate::pipeline::CompressOptions::default()
    };
    assert!(
        !crate::pipeline::should_use_gpu_entropy(&large_streams, &options_no_engine),
        "must not use GPU entropy when engine is None"
    );

    // If we have a GPU available, test with engine
    #[cfg(feature = "webgpu")]
    {
        if let Ok(engine) = WebGpuEngine::new() {
            let options_with_engine = crate::pipeline::CompressOptions {
                backend: crate::pipeline::Backend::WebGpu,
                webgpu_engine: Some(std::sync::Arc::new(engine)),
                threads: 1,
                ..crate::pipeline::CompressOptions::default()
            };
            assert!(
                crate::pipeline::should_use_gpu_entropy(&large_streams, &options_with_engine),
                "must use GPU entropy for streams totaling >= 256KB when GPU available"
            );
        }
    }
}
