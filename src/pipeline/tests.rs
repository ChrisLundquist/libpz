use super::*;
use demux::LzDemuxer;
use stages::{
    self, stage_demux_compress, stage_demux_decompress, stage_fse_decode, stage_fse_encode,
    stage_huffman_decode, stage_huffman_encode, stage_rans_decode, stage_rans_encode_with_options,
    StageBlock, StageMetadata,
};

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
        Pipeline::Bbw,
        Pipeline::Lzr,
        Pipeline::Lzf,
        Pipeline::LzssR,
        Pipeline::Lzfi,
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
        Pipeline::Bbw,
        Pipeline::Lzr,
        Pipeline::Lzf,
        Pipeline::LzssR,
        Pipeline::Lzfi,
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

    for &pipeline in &[
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Lzf,
        Pipeline::Lzfi,
        Pipeline::LzSeqR,
        Pipeline::LzSeqH,
    ] {
        let compressed = compress_mt(&input, pipeline, 4, 512).unwrap();
        assert_eq!(compressed[2], VERSION, "expected V2 for {:?}", pipeline);
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "round-trip failed for {:?}", pipeline);
    }
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

// --- Multi-stream tests ---

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
    let block = stage_rans_encode_with_options(block, &CompressOptions::default()).unwrap();
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
fn test_lzr_unified_scheduler_multiblock_round_trip() {
    let pattern = b"Lzr unified scheduler multiblock round-trip test. ";
    let mut input = Vec::new();
    for _ in 0..400 {
        input.extend_from_slice(pattern);
    }

    let opts = CompressOptions {
        threads: 4,
        block_size: 512,
        ..CompressOptions::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
    assert_eq!(compressed[2], VERSION, "expected V2 multi-block container");
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

// --- Lzf pipeline tests ---

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

// --- Lzfi pipeline tests ---

#[test]
fn test_lzfi_multiblock_round_trip() {
    let pattern = b"Lzfi multi-block test data with repetition. ";
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(pattern);
    }
    let compressed = compress_mt(&input, Pipeline::Lzfi, 4, 1024).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzfi_multistream_deinterleave_reinterleave() {
    // Verify LZ77 deinterleave → interleaved FSE encode → decode → reinterleave
    let input = b"The quick brown fox jumps over the lazy dog. The quick brown fox.";
    let opts = CompressOptions::default();
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };

    let block = stage_demux_compress(block, &LzDemuxer::Lz77, &opts).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);

    let block = stages::stage_fse_interleaved_encode(block).unwrap();
    assert!(block.streams.is_none());
    assert!(!block.data.is_empty());
    assert_eq!(block.data[0], 3, "expected 3 streams");

    let block = stages::stage_fse_interleaved_decode(block).unwrap();
    assert!(block.streams.is_some());
    let streams = block.streams.as_ref().unwrap();
    assert_eq!(streams.len(), 3);

    let block = stage_demux_decompress(block, &LzDemuxer::Lz77).unwrap();
    assert!(block.streams.is_none());
    assert_eq!(block.data, input);
}

// --- Extended match length tests ---

/// Verify Lzr pipeline benefits from extended match lengths on repetitive data.
#[test]
fn test_lzr_extended_match_length() {
    let input = vec![0xAAu8; 100_000];

    // Deflate should use 258-byte max matches
    let deflate_compressed = compress(&input, Pipeline::Deflate).unwrap();

    // Lzr should use extended matches (u16::MAX) and compress better
    let lzr_compressed = compress(&input, Pipeline::Lzr).unwrap();

    // Both must decompress correctly
    let deflate_decompressed = decompress(&deflate_compressed).unwrap();
    let lzr_decompressed = decompress(&lzr_compressed).unwrap();
    assert_eq!(deflate_decompressed, input);
    assert_eq!(lzr_decompressed, input);

    // Lzr with extended matches should produce smaller output on highly
    // repetitive data (fewer matches needed = fewer tokens = better ratio)
    assert!(
        lzr_compressed.len() < deflate_compressed.len(),
        "Lzr ({} bytes) should compress better than Deflate ({} bytes) on repetitive data",
        lzr_compressed.len(),
        deflate_compressed.len()
    );
}

#[test]
fn test_lzr_rans_interleaved_round_trip() {
    let mut input = Vec::new();
    for _ in 0..2048 {
        input.extend_from_slice(b"interleaved-rans-round-trip-");
    }

    let opts = CompressOptions {
        rans_interleaved: true,
        rans_interleaved_min_bytes: 0,
        rans_interleaved_states: 4,
        ..Default::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzssr_rans_interleaved_round_trip() {
    let mut input = Vec::new();
    for _ in 0..1024 {
        input.extend_from_slice(b"lzssr-interleaved-rans-round-trip-");
    }

    let opts = CompressOptions {
        rans_interleaved: true,
        rans_interleaved_min_bytes: 0,
        rans_interleaved_states: 4,
        ..Default::default()
    };
    let compressed = compress_with_options(&input, Pipeline::LzssR, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

/// Verify LZR pipeline round-trips with Recoil parallel rANS decode.
#[test]
fn test_lzr_recoil_round_trip() {
    let mut input = Vec::new();
    for _ in 0..2048 {
        input.extend_from_slice(b"recoil-parallel-rans-decode-test-");
    }

    let opts = CompressOptions {
        rans_interleaved: true,
        rans_interleaved_min_bytes: 0,
        rans_interleaved_states: 4,
        rans_recoil: true,
        rans_recoil_splits: 8,
        ..Default::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

/// Verify Recoil with 8-way interleaved rANS (wider interleave).
#[test]
fn test_lzr_recoil_wide_interleave_round_trip() {
    let mut input = Vec::new();
    for _ in 0..2048 {
        input.extend_from_slice(b"recoil-wide-interleave-test-data-");
    }

    let opts = CompressOptions {
        rans_interleaved: true,
        rans_interleaved_min_bytes: 0,
        rans_interleaved_states: 8,
        rans_recoil: true,
        rans_recoil_splits: 16,
        ..Default::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

/// Verify Lzf pipeline benefits from extended match lengths on repetitive data.
#[test]
fn test_lzf_extended_match_length() {
    let input = vec![0xBBu8; 100_000];

    let deflate_compressed = compress(&input, Pipeline::Deflate).unwrap();
    let lzf_compressed = compress(&input, Pipeline::Lzf).unwrap();

    // Both must decompress correctly
    assert_eq!(decompress(&deflate_compressed).unwrap(), input);
    assert_eq!(decompress(&lzf_compressed).unwrap(), input);

    // Lzf with extended matches should produce smaller output
    assert!(
        lzf_compressed.len() < deflate_compressed.len(),
        "Lzf ({} bytes) should compress better than Deflate ({} bytes) on repetitive data",
        lzf_compressed.len(),
        deflate_compressed.len()
    );
}

/// Verify the max_match_len option is respected when explicitly set.
#[test]
fn test_explicit_max_match_len_option() {
    use crate::lz77;

    let input = vec![0xCCu8; 100_000];

    // Force Lzr to use Deflate-style 258 limit
    let opts_limited = CompressOptions {
        max_match_len: Some(lz77::DEFLATE_MAX_MATCH),
        threads: 1,
        ..Default::default()
    };
    let limited = compress_with_options(&input, Pipeline::Lzr, &opts_limited).unwrap();

    // Use default (extended) limit
    let opts_extended = CompressOptions {
        threads: 1,
        ..Default::default()
    };
    let extended = compress_with_options(&input, Pipeline::Lzr, &opts_extended).unwrap();

    // Both must decompress correctly
    assert_eq!(decompress(&limited).unwrap(), input);
    assert_eq!(decompress(&extended).unwrap(), input);

    // Extended should compress better on highly repetitive data
    assert!(
        extended.len() < limited.len(),
        "extended ({} bytes) should be smaller than limited ({} bytes)",
        extended.len(),
        limited.len()
    );
}

/// Verify extended matches round-trip with various data patterns.
#[test]
fn test_extended_match_round_trip_patterns() {
    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("all_same", vec![0xDD; 50_000]),
        (
            "repeating_short",
            b"abc".iter().cycle().take(50_000).copied().collect(),
        ),
        (
            "repeating_long",
            b"The quick brown fox jumps over the lazy dog. "
                .iter()
                .cycle()
                .take(50_000)
                .copied()
                .collect(),
        ),
    ];

    for (name, input) in &patterns {
        for pipeline in [Pipeline::Lzr, Pipeline::Lzf] {
            let compressed = compress(input, pipeline).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(
                &decompressed, input,
                "{:?} round-trip failed for pattern '{}'",
                pipeline, name
            );
        }
    }
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

    #[test]
    fn test_gpu_rans_interleaved_decode_round_trip() {
        let mut opts = match make_webgpu_options() {
            Some(o) => o,
            None => return,
        };
        opts.rans_interleaved = true;
        opts.rans_interleaved_min_bytes = 0;
        opts.rans_interleaved_states = 4;

        let input: Vec<u8> = (0..192 * 1024)
            .map(|i| ((i * 13 + 97) % 251) as u8)
            .collect();
        let compressed = compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();

        let dec_opts = DecompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: opts.webgpu_engine.clone(),
            threads: 0,
        };
        let decompressed = decompress_with_options(&compressed, &dec_opts).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_gpu_batched_lzfi_round_trip() {
        let opts = match make_webgpu_options() {
            Some(o) => o,
            None => return,
        };

        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = pattern.iter().cycle().take(256 * 1024).copied().collect();
        let compressed = compress_with_options(&input, Pipeline::Lzfi, &opts).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    /// Multi-block pipeline pipelining regression test.
    ///
    /// Creates input large enough to produce many blocks (>= 8), then
    /// compresses with GPU and verifies the round-trip. This exercises
    /// the full GPU coordinator pipeline: ring-buffered LZ77 stage 0
    /// batching, StageN entropy encoding, and result assembly.
    ///
    /// Correctness failures here indicate pipelining bugs (e.g., staging
    /// buffer reuse before readback, ring slot cross-contamination, or
    /// incomplete synchronization between GPU compute and CPU readback).
    #[test]
    fn test_gpu_pipeline_multiblock_correctness() {
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(_) => return,
        };
        let opts = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            block_size: 64 * 1024, // 64KB blocks → 8 blocks for 512KB input
            threads: 2,
            ..CompressOptions::default()
        };

        // 512KB input with distinct pattern per 64KB region to detect
        // cross-contamination between blocks.
        let mut input = Vec::with_capacity(512 * 1024);
        let patterns: Vec<&[u8]> = vec![
            b"alpha pattern block one data here. ",
            b"beta different content for block two. ",
            b"gamma third block uses gamma pattern. ",
            b"delta fourth block with delta stuff. ",
            b"epsilon five five five five five. ",
            b"zeta block six zeta zeta zeta. ",
            b"eta seventh block eta eta eta. ",
            b"theta eighth block theta theta. ",
        ];
        for p in &patterns {
            let block: Vec<u8> = p.iter().cycle().take(64 * 1024).copied().collect();
            input.extend_from_slice(&block);
        }

        for pipeline in [Pipeline::Deflate, Pipeline::Lzr, Pipeline::LzSeqR] {
            let compressed = compress_with_options(&input, pipeline, &opts).unwrap();
            let decompressed = decompress(&compressed).unwrap();
            assert_eq!(
                decompressed, input,
                "{:?} multi-block GPU pipeline round-trip failed",
                pipeline
            );
        }
    }
}

// --- LzSeqR optimal parsing tests (Task 6) ---

#[test]
fn test_lzseq_r_optimal_round_trip_short() {
    let input = b"abc".repeat(30);
    let opts = CompressOptions {
        parse_strategy: ParseStrategy::Optimal,
        ..CompressOptions::default()
    };
    let compressed = compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input.as_slice());
}

#[test]
fn test_lzseq_r_optimal_round_trip_large() {
    // Larger data to exercise optimal parsing
    let pattern = b"compression and decompression with optimal parsing ";
    let input: Vec<u8> = pattern.iter().cycle().take(64 * 1024).copied().collect();
    let opts = CompressOptions {
        parse_strategy: ParseStrategy::Optimal,
        ..CompressOptions::default()
    };
    let compressed = compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_lzseq_r_quality_level_default_is_optimal() {
    let opts = CompressOptions::for_quality(QualityLevel::Default);
    assert_eq!(opts.parse_strategy, ParseStrategy::Optimal);
}

#[test]
fn test_lzseq_r_quality_level_speed_is_lazy() {
    let opts = CompressOptions::for_quality(QualityLevel::Speed);
    assert_eq!(opts.parse_strategy, ParseStrategy::Lazy);
}

#[test]
fn test_lzseq_r_quality_level_quality_uses_larger_window() {
    let opts = CompressOptions::for_quality(QualityLevel::Quality);
    assert_eq!(opts.parse_strategy, ParseStrategy::Optimal);
    assert!(
        opts.seq_window_size.unwrap_or(0) > 128 * 1024,
        "quality mode should use a window larger than 128KB"
    );
}

#[test]
fn test_lzseq_r_optimal_better_than_lazy_on_structured_data() {
    // Verify both parsing strategies round-trip correctly on structured data.
    // NOTE: We don't assert optimal < lazy because they make different tradeoffs:
    // lazy matching is simpler and sometimes better on specific patterns, while
    // optimal parsing is better on average. The key fix here is correct handling
    // of the "next" byte in the Match token (see encode_match_sequence).
    let pattern = b"aaaaaabcbcbcbcbcbcbcbcbcbc";
    let input: Vec<u8> = pattern.iter().cycle().take(64 * 1024).copied().collect();

    let lazy_opts = CompressOptions::for_quality(QualityLevel::Speed);
    let optimal_opts = CompressOptions::for_quality(QualityLevel::Default);

    let lazy_compressed = compress_with_options(&input, Pipeline::LzSeqR, &lazy_opts).unwrap();
    let optimal_compressed =
        compress_with_options(&input, Pipeline::LzSeqR, &optimal_opts).unwrap();

    // Both must round-trip correctly
    assert_eq!(
        decompress(&lazy_compressed).unwrap(),
        input,
        "lazy round-trip"
    );
    assert_eq!(
        decompress(&optimal_compressed).unwrap(),
        input,
        "optimal round-trip"
    );
}

// --- BackendAssignment tests (Task 1) ---

#[test]
fn test_backend_assignment_default_is_auto() {
    let opts = CompressOptions::default();
    assert_eq!(opts.stage0_backend, BackendAssignment::Auto);
    assert_eq!(opts.stage1_backend, BackendAssignment::Auto);
}

#[test]
fn test_backend_assignment_cpu_variant_always_available() {
    // This test ensures the Cpu variant compiles and is always available,
    // regardless of feature flags
    let cpu_assign = BackendAssignment::Cpu;
    assert_eq!(cpu_assign, BackendAssignment::Cpu);

    // Create options with explicit CPU assignment
    let opts = CompressOptions {
        stage0_backend: BackendAssignment::Cpu,
        stage1_backend: BackendAssignment::Cpu,
        ..CompressOptions::default()
    };

    assert_eq!(opts.stage0_backend, BackendAssignment::Cpu);
    assert_eq!(opts.stage1_backend, BackendAssignment::Cpu);
}

// --- Shared-stream rANS pipeline tests ---

#[test]
fn test_shared_stream_lzseqr_round_trip() {
    let input: Vec<u8> = b"the quick brown fox jumps over the lazy dog. "
        .iter()
        .cycle()
        .take(45_000)
        .copied()
        .collect();
    let opts = CompressOptions {
        rans_interleaved: true,
        rans_shared_stream: true,
        threads: 1,
        ..CompressOptions::default()
    };
    let compressed = compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_shared_stream_lzr_round_trip() {
    let input: Vec<u8> = (0..10_000).map(|i| ((i * 37 + 13) % 256) as u8).collect();
    let opts = CompressOptions {
        rans_interleaved: true,
        rans_shared_stream: true,
        threads: 1,
        ..CompressOptions::default()
    };
    let compressed = compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_shared_stream_backward_compat() {
    // Data encoded with interleaved (non-shared) must still decode correctly
    let input = b"backward compatibility test data ".repeat(500);
    let opts_interleaved = CompressOptions {
        rans_interleaved: true,
        rans_shared_stream: false,
        threads: 1,
        ..CompressOptions::default()
    };
    let compressed = compress_with_options(&input, Pipeline::LzSeqR, &opts_interleaved).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}

#[test]
fn test_shared_stream_small_input() {
    // Input smaller than rans_interleaved_min_bytes → falls back to basic rANS
    let input = b"small";
    let opts = CompressOptions {
        rans_interleaved: true,
        rans_shared_stream: true,
        threads: 1,
        ..CompressOptions::default()
    };
    let compressed = compress_with_options(input, Pipeline::Lzr, &opts).unwrap();
    let decompressed = decompress(&compressed).unwrap();
    assert_eq!(decompressed, input);
}
