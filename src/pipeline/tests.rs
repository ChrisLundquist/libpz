use super::*;
use demux::LzDemuxer;
use stages::{
    pipeline_stage_count, stage_demux_compress, stage_demux_decompress, stage_fse_decode,
    stage_fse_encode, stage_huffman_decode, stage_huffman_encode, stage_rans_decode,
    stage_rans_encode, StageBlock, StageMetadata,
};

// --- Deflate pipeline tests ---

#[test]
fn test_deflate_empty() {
    let result = compress(&[], Pipeline::Deflate).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_deflate_round_trip_hello() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Deflate).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_deflate_round_trip_repeating() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    // Use enough repetitions to overcome the ~1KB frequency table overhead
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Deflate).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
    // Should actually compress (input is ~4.5KB, overhead is ~1KB)
    assert!(
        compressed.len() < input.len(),
        "compressed {} >= input {}",
        compressed.len(),
        input.len()
    );
}

#[test]
fn test_deflate_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let compressed = compress(&input, Pipeline::Deflate).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- BW pipeline tests ---

#[test]
fn test_bw_empty() {
    let result = compress(&[], Pipeline::Bw).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_bw_round_trip_hello() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Bw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_bw_round_trip_repeating() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..20 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Bw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
    assert!(
        compressed.len() < input.len(),
        "compressed {} >= input {}",
        compressed.len(),
        input.len()
    );
}

#[test]
fn test_bw_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let compressed = compress(&input, Pipeline::Bw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_bw_round_trip_all_same() {
    let input = vec![b'x'; 200];
    let compressed = compress(&input, Pipeline::Bw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Bbw pipeline round-trip tests ---

#[test]
fn test_bbw_round_trip_hello() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Bbw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_bbw_round_trip_repeating() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Bbw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
    assert!(
        compressed.len() < input.len(),
        "compressed {} >= input {}",
        compressed.len(),
        input.len()
    );
}

#[test]
fn test_bbw_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let compressed = compress(&input, Pipeline::Bbw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_bbw_round_trip_all_same() {
    let input = vec![b'x'; 200];
    let compressed = compress(&input, Pipeline::Bbw).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Header / format tests ---

#[test]
fn test_invalid_magic() {
    let result = decompress(&[0, 0, 1, 0, 0, 0, 0, 0]);
    assert_eq!(result, Err(PzError::InvalidInput));
}

#[test]
fn test_invalid_version() {
    let result = decompress(&[b'P', b'Z', 99, 0, 0, 0, 0, 0]);
    assert_eq!(result, Err(PzError::Unsupported));
}

#[test]
fn test_invalid_pipeline() {
    let result = decompress(&[b'P', b'Z', VERSION, 99, 0, 0, 0, 0]);
    assert_eq!(result, Err(PzError::Unsupported));
}

#[test]
fn test_too_short_input() {
    let result = decompress(b"PZ");
    assert_eq!(result, Err(PzError::InvalidInput));
}

#[test]
fn test_zero_original_length() {
    let result = decompress(&[b'P', b'Z', VERSION, 0, 0, 0, 0, 0]);
    assert_eq!(result.unwrap(), Vec::<u8>::new());
}

// --- Cross-pipeline tests ---

#[test]
fn test_all_pipelines_banana() {
    let input = b"banana banana banana banana banana";
    for &pipeline in &[
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Lzr,
        Pipeline::Lzf,
    ] {
        let compressed = compress(input, pipeline).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "failed for pipeline {:?}", pipeline);
    }
}

#[test]
fn test_all_pipelines_medium_text() {
    let mut input = Vec::new();
    for _ in 0..10 {
        input.extend(b"abcdefghij klmnopqrstuvwxyz 0123456789 ");
    }
    for &pipeline in &[
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Lzr,
        Pipeline::Lzf,
    ] {
        let compressed = compress(&input, pipeline).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "failed for pipeline {:?}", pipeline);
    }
}

// --- Multi-block parallel compression tests ---

/// Helper: compress with explicit thread count and block size.
fn compress_mt(
    input: &[u8],
    pipeline: Pipeline,
    threads: usize,
    block_size: usize,
) -> PzResult<Vec<u8>> {
    let options = CompressOptions {
        threads,
        block_size,
        ..Default::default()
    };
    compress_with_options(input, pipeline, &options)
}

#[test]
fn test_multiblock_round_trip_all_pipelines() {
    // 2KB input with 512-byte blocks = 4 blocks
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..50 {
        input.extend_from_slice(pattern);
    }

    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
        let compressed = compress_mt(&input, pipeline, 4, 512).unwrap();
        assert_eq!(compressed[2], VERSION, "expected V2 for {:?}", pipeline);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "round-trip failed for {:?}", pipeline);
    }
}

#[test]
fn test_multiblock_single_block_fallback() {
    // Input smaller than block_size → single block in V2 format
    let input = b"small input data";
    let compressed = compress_mt(input, Pipeline::Deflate, 4, 65536).unwrap();
    assert_eq!(compressed[2], VERSION, "expected V2 for small input");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multiblock_single_thread() {
    // threads=1 → single block in V2 format
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..50 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Bw, 1, 512).unwrap();
    assert_eq!(compressed[2], VERSION, "expected V2 for single-threaded");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multiblock_various_block_sizes() {
    let pattern = b"ABCDEFGHIJ0123456789";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    for block_size in [256, 512, 1024, 2048] {
        for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            let compressed = compress_mt(&input, pipeline, 4, block_size).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(
                decompressed, input,
                "failed for {:?} block_size={}",
                pipeline, block_size
            );
        }
    }
}

#[test]
fn test_multiblock_exact_one_block() {
    // Input exactly equal to block_size → single block in V2 format
    let input = vec![b'x'; 1024];
    let compressed = compress_mt(&input, Pipeline::Bw, 4, 1024).unwrap();
    assert_eq!(compressed[2], VERSION, "exact one block should be V2");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multiblock_empty_input() {
    let compressed = compress_mt(&[], Pipeline::Deflate, 4, 512).unwrap();
    assert!(compressed.is_empty());
}

#[test]
fn test_multiblock_2_threads() {
    // 4KB input, 1KB blocks = 4 blocks, but only 2 threads
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Lzf, 2, 1024).unwrap();
    assert_eq!(compressed[2], VERSION);
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multiblock_decompress_with_threads() {
    // Verify decompress_with_threads works
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..50 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Bw, 4, 512).unwrap();
    let decompressed = decompress_with_threads(&compressed, 2).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multiblock_large_input() {
    // ~100KB input, 16KB blocks = ~6 blocks
    let pattern = b"Compression test data with some repetition. ";
    let mut input = Vec::new();
    for _ in 0..2500 {
        input.extend_from_slice(pattern);
    }
    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
        let compressed = compress_mt(&input, pipeline, 4, 16384).unwrap();
        assert_eq!(compressed[2], VERSION);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "large input failed for {:?}", pipeline);
    }
}

// --- Pipeline-parallel specific tests ---

#[test]
fn test_pipeline_parallel_round_trip_all_pipelines() {
    // Force pipeline-parallel path: need num_blocks >= stage_count.
    // Bw has 4 stages, so we need >= 4 blocks. Use small block_size.
    let pattern = b"Pipeline parallelism test data with repetitions. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    // 5000 bytes / 512 byte blocks = ~10 blocks, enough for any pipeline
    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
        let stage_count = pipeline_stage_count(pipeline);
        let block_size = 512;
        let num_blocks = input.len().div_ceil(block_size);
        assert!(
            num_blocks >= stage_count,
            "test setup: need {} blocks >= {} stages for {:?}",
            num_blocks,
            stage_count,
            pipeline
        );

        let compressed = compress_mt(&input, pipeline, 4, block_size).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "pipeline-parallel round-trip failed for {:?}",
            pipeline
        );
    }
}

#[test]
fn test_pipeline_parallel_matches_single_threaded() {
    // Verify pipeline-parallel produces output that single-threaded can decompress,
    // and vice versa.
    let pattern = b"Cross-check between single-threaded and pipeline-parallel. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }

    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
        // Single-threaded compress
        let compressed_st = compress(input.as_slice(), pipeline).unwrap();
        let decompressed_st = decompress(&compressed_st).unwrap();
        assert_eq!(decompressed_st, input);

        // Multi-threaded (pipeline-parallel) compress
        let compressed_mt = compress_mt(&input, pipeline, 4, 512).unwrap();
        let decompressed_mt = decompress(&compressed_mt).unwrap();
        assert_eq!(decompressed_mt, input);

        // Both decompress to the same data
        assert_eq!(
            decompressed_st, decompressed_mt,
            "single-threaded and pipeline-parallel disagree for {:?}",
            pipeline
        );
    }
}

#[test]
fn test_pipeline_parallel_many_small_blocks() {
    // Stress test: many blocks flowing through the pipeline.
    // 10KB input / 128 byte blocks = 80 blocks
    let input: Vec<u8> = (0..10240).map(|i| (i % 251) as u8).collect();
    let compressed = compress_mt(&input, Pipeline::Bw, 4, 128).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_pipeline_parallel_decompress_with_threads() {
    // Compress with pipeline parallelism, then decompress with explicit threads.
    let pattern = b"Decompress parallelism test. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }

    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
        let compressed = compress_mt(&input, pipeline, 4, 512).unwrap();

        // Decompress with various thread counts
        for threads in [0, 1, 2, 4] {
            let decompressed = decompress_with_threads(&compressed, threads).unwrap();
            assert_eq!(
                decompressed, input,
                "decompress failed for {:?} threads={}",
                pipeline, threads
            );
        }
    }
}

#[test]
fn test_pipeline_parallel_error_propagation() {
    // Feed corrupt data that will fail during decompression stages.
    // Create valid V2 header but corrupt block data.
    let input = vec![b'A'; 2048];
    let compressed = compress_mt(&input, Pipeline::Bw, 4, 512).unwrap();

    // Corrupt the last byte of the block data
    let mut corrupt = compressed.clone();
    let last = corrupt.len() - 1;
    corrupt[last] ^= 0xFF;

    // Should fail (either error or wrong data)
    let result = decompress(&corrupt);
    // We just verify it doesn't hang/panic — either error or wrong data is acceptable
    // for this corruption test.
    match result {
        Ok(_data) => {
            // If it "succeeds", corruption wasn't detected
            // (may not always be detected depending on where it lands)
        }
        Err(_) => {
            // Expected: corruption detected
        }
    }
}

// --- Optimal parsing pipeline tests ---

#[test]
fn test_optimal_deflate_round_trip() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let opts = CompressOptions {
        parse_strategy: ParseStrategy::Optimal,
        threads: 1,
        ..Default::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Deflate, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_optimal_lza_round_trip() {
    let pattern = b"abcdefghij abcdefghij ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let opts = CompressOptions {
        parse_strategy: ParseStrategy::Optimal,
        threads: 1,
        ..Default::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Lzf, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_optimal_multiblock_round_trip() {
    let pattern = b"Hello, World! This is a test pattern. ";
    let mut input = Vec::new();
    for _ in 0..500 {
        input.extend_from_slice(pattern);
    }
    let opts = CompressOptions {
        parse_strategy: ParseStrategy::Optimal,
        threads: 2,
        block_size: 4096,
        ..Default::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Deflate, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Auto-selection tests ---

#[test]
fn test_select_pipeline_empty() {
    assert_eq!(select_pipeline(&[]), Pipeline::Deflate);
}

#[test]
fn test_select_pipeline_constant_data() {
    // All-same bytes → high run ratio → should select Bw
    let input = vec![0xAA; 10000];
    let pipeline = select_pipeline(&input);
    assert_eq!(pipeline, Pipeline::Bw);
}

#[test]
fn test_select_pipeline_text() {
    // Repetitive text → good match density, moderate entropy → Deflate or Lzf
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let pipeline = select_pipeline(&input);
    assert!(
        pipeline == Pipeline::Deflate || pipeline == Pipeline::Lzf,
        "expected Deflate or Lzf, got {:?}",
        pipeline
    );
}

#[test]
fn test_select_pipeline_random() {
    // Pseudo-random → high entropy, low match density → Deflate (fastest)
    let mut input = vec![0u8; 10000];
    let mut state: u32 = 12345;
    for byte in &mut input {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        *byte = (state >> 16) as u8;
    }
    let pipeline = select_pipeline(&input);
    assert_eq!(pipeline, Pipeline::Deflate);
}

#[test]
fn test_select_pipeline_trial_round_trip() {
    // Trial-selected pipeline must produce valid compressed output
    let pattern = b"Hello, World! This is a test pattern. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let opts = CompressOptions::default();
    let pipeline = select_pipeline_trial(&input, &opts, 4096);
    let compressed = compress(&input, pipeline).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_select_pipeline_trial_small_sample() {
    // Trial with very small input should not panic
    let input = b"tiny";
    let opts = CompressOptions::default();
    let pipeline = select_pipeline_trial(input, &opts, 32);
    // Should still produce a valid pipeline
    let compressed = compress(input, pipeline).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input.as_slice());
}

#[test]
fn test_select_pipeline_auto_vs_explicit_round_trip() {
    // Every auto-selected pipeline must produce valid output
    let test_inputs: Vec<Vec<u8>> = vec![
        vec![0xFF; 5000],                            // constant
        (0..=255u8).cycle().take(5000).collect(),    // uniform
        b"banana banana banana banana ".repeat(100), // text
    ];
    for input in &test_inputs {
        let pipeline = select_pipeline(input);
        let compressed = compress(input, pipeline).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }
}

// --- Multi-stream Deflate tests ---

#[test]
fn test_multistream_deflate_round_trip_small() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Deflate).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multistream_deflate_round_trip_medium() {
    // 10KB of repetitive text
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..222 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Deflate).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
    assert!(
        compressed.len() < input.len(),
        "compressed {} >= input {}",
        compressed.len(),
        input.len()
    );
}

#[test]
fn test_multistream_deflate_round_trip_large() {
    // 1MB of pseudo-random + repeated data
    let mut input = Vec::with_capacity(1 << 20);
    let pattern = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    while input.len() < (1 << 20) {
        input.extend_from_slice(pattern);
    }
    input.truncate(1 << 20);
    // Sprinkle pseudo-random bytes for variety
    let mut state: u32 = 42;
    for i in (0..input.len()).step_by(37) {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        input[i] = (state >> 16) as u8;
    }

    let compressed = compress(&input, Pipeline::Deflate).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multistream_deflate_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(2048).collect();
    let compressed = compress(&input, Pipeline::Deflate).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multistream_deflate_multiblock() {
    // Multi-block pipeline-parallel path
    let pattern = b"Multi-stream multi-block test pattern. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Deflate, 4, 1024).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multistream_deflate_compression_ratio() {
    // Multi-stream should compress well on structured/repetitive data
    // because offset/length/literal distributions are each tighter
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..500 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Deflate).unwrap();
    assert!(
        compressed.len() < input.len() / 2,
        "expected significant compression: compressed {} vs input {}",
        compressed.len(),
        input.len()
    );
}

// --- Multi-stream LZF tests ---

#[test]
fn test_multistream_lzf_round_trip_small() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multistream_lzf_round_trip_medium() {
    let pattern = b"abcdefghij abcdefghij ";
    let mut input = Vec::new();
    for _ in 0..500 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
    assert!(
        compressed.len() < input.len(),
        "compressed {} >= input {}",
        compressed.len(),
        input.len()
    );
}

#[test]
fn test_multistream_lzf_round_trip_large() {
    // 1MB input
    let mut input = Vec::with_capacity(1 << 20);
    let pattern = b"LZF multi-stream test data with some repetition. ";
    while input.len() < (1 << 20) {
        input.extend_from_slice(pattern);
    }
    input.truncate(1 << 20);

    let compressed = compress(&input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multistream_lzf_multiblock() {
    let pattern = b"LZF multi-stream multi-block test. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Lzf, 4, 1024).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_multistream_all_pipelines_round_trip() {
    // Verify all pipelines still produce correct round-trips
    let pattern = b"Cross-pipeline multi-stream validation. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzr] {
        let compressed = compress(&input, pipeline).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "round-trip failed for {:?}", pipeline);
    }
}

#[test]
fn test_multistream_stage_deinterleave_reinterleave() {
    // Unit test: verify LZ77 deinterleave/reinterleave round-trips correctly
    // by going through the stage functions directly
    let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
    let opts = CompressOptions::default();
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };

    // LZ77 compress → produces streams
    let block = stage_demux_compress(block, &LzDemuxer::Lz77, &opts).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);
    // offsets and lengths are 2 bytes per match, literals 1 byte per match
    assert_eq!(streams[0].len(), streams[1].len());
    assert_eq!(streams[0].len(), streams[2].len() * 2);

    // Huffman encode → serializes streams into data
    let block = stage_huffman_encode(block).unwrap();
    assert!(block.streams.is_none());
    assert!(!block.data.is_empty());
    // Verify multi-stream header: [num_streams: u8][pre_entropy_len: u32][meta_len: u16]
    assert_eq!(block.data[0], 3, "expected 3 streams");

    // Huffman decode → restores streams
    let block = stage_huffman_decode(block).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);

    // LZ77 decompress → reinterleaves and decompresses
    let block = stage_demux_decompress(block, &LzDemuxer::Lz77).unwrap();
    assert!(block.streams.is_none());
    assert_eq!(block.data, input);
}

// --- Lzr pipeline tests ---

#[test]
fn test_lzr_empty() {
    let result = compress(&[], Pipeline::Lzr).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_lzr_round_trip_hello() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Lzr).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzr_round_trip_repeating() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Lzr).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
    assert!(
        compressed.len() < input.len(),
        "compressed {} >= input {}",
        compressed.len(),
        input.len()
    );
}

#[test]
fn test_lzr_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let compressed = compress(&input, Pipeline::Lzr).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzr_round_trip_all_same() {
    let input = vec![0xAA_u8; 500];
    let compressed = compress(&input, Pipeline::Lzr).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzr_multiblock_round_trip() {
    let pattern = b"Lzr multi-block test data with repetition. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Lzr, 4, 1024).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzr_multiblock_large() {
    let mut input = Vec::with_capacity(1 << 20);
    let pattern = b"Lzr multi-stream test data with some repetition. ";
    while input.len() < (1 << 20) {
        input.extend_from_slice(pattern);
    }
    input.truncate(1 << 20);

    let compressed = compress(&input, Pipeline::Lzr).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzr_multistream_deinterleave_reinterleave() {
    // Verify LZ77 deinterleave → rANS encode → rANS decode → reinterleave
    let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
    let opts = CompressOptions::default();
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };

    // LZ77 compress → produces streams
    let block = stage_demux_compress(block, &LzDemuxer::Lz77, &opts).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);

    // rANS encode → serializes streams into data
    let block = stage_rans_encode(block).unwrap();
    assert!(block.streams.is_none());
    assert!(!block.data.is_empty());
    // Verify multi-stream header: [num_streams: u8][pre_entropy_len: u32][meta_len: u16]
    assert_eq!(block.data[0], 3, "expected 3 streams");

    // rANS decode → restores streams
    let block = stage_rans_decode(block).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);

    // LZ77 decompress → reinterleaves and decompresses
    let block = stage_demux_decompress(block, &LzDemuxer::Lz77).unwrap();
    assert!(block.streams.is_none());
    assert_eq!(block.data, input);
}

#[test]
fn test_lzr_pipeline_parallel() {
    let pattern = b"Lzr pipeline-parallel test data. ";
    let mut input = Vec::new();
    for _ in 0..300 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Lzr, 4, 512).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzr_trial_selection_candidate() {
    // Verify Lzr is included in trial selection
    let pattern = b"Trial selection test data for rANS pipeline. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let opts = CompressOptions {
        threads: 1,
        ..CompressOptions::default()
    };
    // Just verify it doesn't crash — Lzr may or may not win
    let _pipeline = select_pipeline_trial(&input, &opts, 2048);
}

// --- Lzf pipeline tests ---

#[test]
fn test_lzf_empty() {
    let result = compress(&[], Pipeline::Lzf).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_lzf_round_trip_hello() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzf_round_trip_repeating() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
    assert!(
        compressed.len() < input.len(),
        "compressed {} >= input {}",
        compressed.len(),
        input.len()
    );
}

#[test]
fn test_lzf_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let compressed = compress(&input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzf_round_trip_all_same() {
    let input = vec![0xAA_u8; 500];
    let compressed = compress(&input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzf_multiblock_round_trip() {
    let pattern = b"Lzf multi-block test data with repetition. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Lzf, 4, 1024).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzf_multiblock_large() {
    let mut input = Vec::with_capacity(1 << 20);
    let pattern = b"Lzf multi-stream test data with some repetition. ";
    while input.len() < (1 << 20) {
        input.extend_from_slice(pattern);
    }
    input.truncate(1 << 20);

    let compressed = compress(&input, Pipeline::Lzf).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzf_multistream_deinterleave_reinterleave() {
    // Verify LZ77 deinterleave → FSE encode → FSE decode → reinterleave
    let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
    let opts = CompressOptions::default();
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };

    // LZ77 compress → produces streams
    let block = stage_demux_compress(block, &LzDemuxer::Lz77, &opts).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);

    // FSE encode → serializes streams into data
    let block = stage_fse_encode(block).unwrap();
    assert!(block.streams.is_none());
    assert!(!block.data.is_empty());
    // Verify multi-stream header: [num_streams: u8][pre_entropy_len: u32][meta_len: u16]
    assert_eq!(block.data[0], 3, "expected 3 streams");

    // FSE decode → restores streams
    let block = stage_fse_decode(block).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);

    // LZ77 decompress → reinterleaves and decompresses
    let block = stage_demux_decompress(block, &LzDemuxer::Lz77).unwrap();
    assert!(block.streams.is_none());
    assert_eq!(block.data, input);
}

#[test]
fn test_lzf_pipeline_parallel() {
    let pattern = b"Lzf pipeline-parallel test data. ";
    let mut input = Vec::new();
    for _ in 0..300 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Lzf, 4, 512).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzf_trial_selection_candidate() {
    // Verify Lzf is included in trial selection
    let pattern = b"Trial selection test data for FSE pipeline. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let opts = CompressOptions {
        threads: 1,
        ..CompressOptions::default()
    };
    // Just verify it doesn't crash — Lzf may or may not win
    let _pipeline = select_pipeline_trial(&input, &opts, 2048);
}

// --- LzssR pipeline tests ---

#[test]
fn test_lzssr_empty() {
    let result = compress(&[], Pipeline::LzssR).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_lzssr_round_trip_hello() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::LzssR).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzssr_round_trip_repeating() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::LzssR).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzssr_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let compressed = compress(&input, Pipeline::LzssR).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Lz78R pipeline tests ---

#[test]
fn test_lz78r_empty() {
    let result = compress(&[], Pipeline::Lz78R).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_lz78r_round_trip_hello() {
    let input = b"hello, world! hello, world!";
    let compressed = compress(input, Pipeline::Lz78R).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz78r_round_trip_repeating() {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress(&input, Pipeline::Lz78R).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz78r_round_trip_binary() {
    let input: Vec<u8> = (0..=255).cycle().take(512).collect();
    let compressed = compress(&input, Pipeline::Lz78R).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lz78r_round_trip_all_same() {
    let input = vec![0xAA_u8; 500];
    let compressed = compress(&input, Pipeline::Lz78R).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Trial selection with new pipelines ---

#[test]
fn test_trial_selection_includes_experimental() {
    let pattern = b"Trial selection test data with experimental pipelines. ";
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend_from_slice(pattern);
    }
    let opts = CompressOptions {
        threads: 1,
        ..CompressOptions::default()
    };
    // Verify trial selection doesn't crash with new pipelines in candidates
    let _pipeline = select_pipeline_trial(&input, &opts, 2048);
}

#[cfg(feature = "webgpu")]
mod gpu_batched_tests {
    use super::*;

    fn make_webgpu_options() -> Option<CompressOptions> {
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(_) => return None,
        };
        Some(CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            // Small block size to force multiple blocks within GPU dispatch range
            block_size: 64 * 1024,
            threads: 2,
            ..CompressOptions::default()
        })
    }

    #[test]
    fn test_gpu_batched_deflate_round_trip() {
        let opts = match make_webgpu_options() {
            Some(o) => o,
            None => return,
        };

        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(256 * 1024).copied().collect();
        let compressed = compress_with_options(&input, Pipeline::Deflate, &opts).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gpu_batched_lzr_round_trip() {
        let opts = match make_webgpu_options() {
            Some(o) => o,
            None => return,
        };

        let input: Vec<u8> = (0..256 * 1024).map(|i| (i % 251) as u8).collect();
        let compressed = compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gpu_batched_lzf_round_trip() {
        let opts = match make_webgpu_options() {
            Some(o) => o,
            None => return,
        };

        let input: Vec<u8> = (0..256 * 1024)
            .map(|i| ((i * 7 + 13) % 251) as u8)
            .collect();
        let compressed = compress_with_options(&input, Pipeline::Lzf, &opts).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }
}
