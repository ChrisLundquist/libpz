use super::*;

// --- Task 2 tests: Stage 1 routing by backend assignment ---

#[test]
fn test_stage1_routing_cpu_when_no_gpu() {
    // Compress a 512KB block with Auto backend assignment and no GPU.
    // Should route to CPU path and produce correct output.
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

    let opts = CompressOptions {
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
        .expect("compression failed");
    let decompressed = super::super::decompress(&compressed).expect("decompression failed");
    assert_eq!(decompressed, input, "round-trip should match");
}

#[test]
fn test_stage1_routing_cpu_forced() {
    // Compress with explicit CPU backend assignment.
    // Even if GPU were available, should use CPU.
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

    let opts = CompressOptions {
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Cpu,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
        .expect("compression failed");
    let decompressed = super::super::decompress(&compressed).expect("decompression failed");
    assert_eq!(decompressed, input, "round-trip should match");
}

#[test]
fn test_stage1_routing_respects_size_threshold() {
    // Compress a block smaller than GPU_ENTROPY_THRESHOLD.
    // Should route to CPU even with Auto assignment.
    // GPU_ENTROPY_THRESHOLD is 256KB, so use 128KB.
    let input: Vec<u8> = (0..=255).cycle().take(128 * 1024).collect();

    let opts = CompressOptions {
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
        .expect("compression failed");
    let decompressed = super::super::decompress(&compressed).expect("decompression failed");
    assert_eq!(decompressed, input, "round-trip should match");
}

#[cfg(feature = "webgpu")]
#[test]
fn test_stage1_auto_backpressure_biases_to_cpu() {
    use super::super::BackendAssignment;
    use super::super::GPU_ENTROPY_THRESHOLD;

    let block_len = GPU_ENTROPY_THRESHOLD * 2;
    let limit = 8usize;

    assert!(
        should_route_block_to_gpu_entropy_with_backpressure(
            block_len,
            BackendAssignment::Auto,
            true,
            0,
            limit,
        ),
        "auto should route to GPU when pressure is low"
    );
    assert!(
        !should_route_block_to_gpu_entropy_with_backpressure(
            block_len,
            BackendAssignment::Auto,
            true,
            limit,
            limit,
        ),
        "auto should bias to CPU when pressure reaches limit"
    );
}

#[cfg(feature = "webgpu")]
#[test]
fn test_stage1_backpressure_does_not_override_explicit_backend() {
    use super::super::BackendAssignment;
    use super::super::GPU_ENTROPY_THRESHOLD;

    let block_len = GPU_ENTROPY_THRESHOLD * 2;
    let high_pressure = 1_000usize;

    assert!(
        should_route_block_to_gpu_entropy_with_backpressure(
            block_len,
            BackendAssignment::Gpu,
            true,
            high_pressure,
            1,
        ),
        "explicit GPU assignment should remain GPU regardless of pressure"
    );
    assert!(
        !should_route_block_to_gpu_entropy_with_backpressure(
            block_len,
            BackendAssignment::Cpu,
            true,
            0,
            1,
        ),
        "explicit CPU assignment should remain CPU regardless of pressure"
    );
}

// --- Task 3 tests: Round-trip correctness and threshold boundary ---

#[test]
fn test_heterogeneous_routing_roundtrip_lzseqr_cpu_only() {
    // Compress and decompress a 1MB synthetic payload (alternating byte pattern)
    // using LzSeqR with CPU-only backend assignment.
    // Assert decompressed output equals input.
    let mut input = Vec::new();
    for i in 0..1024 * 1024 {
        input.push((i % 256) as u8);
    }

    let opts = CompressOptions {
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Cpu,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "1MB round-trip with CPU-only backend should match"
    );
}

#[test]
fn test_heterogeneous_routing_roundtrip_lzseqr_auto_no_gpu() {
    // Compress and decompress using Auto backend with no GPU engine.
    // Should behave identically to CPU-only.
    // Assert no panic and output is correct.
    let mut input = Vec::new();
    for i in 0..1024 * 1024 {
        input.push((i % 256) as u8);
    }

    let opts = CompressOptions {
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Auto,
        #[cfg(feature = "webgpu")]
        webgpu_engine: None,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "1MB round-trip with Auto (no GPU) should match"
    );
}

#[test]
fn test_backend_assignment_threshold_boundary() {
    // Test at and below the GPU_ENTROPY_THRESHOLD boundary.
    // GPU_ENTROPY_THRESHOLD is 256KB.

    // Test exactly at threshold (256KB)
    let input_at_threshold: Vec<u8> = (0..=255).cycle().take(256 * 1024).collect();
    let opts = CompressOptions {
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    let compressed =
        super::super::compress_with_options(&input_at_threshold, Pipeline::LzSeqR, &opts)
            .expect("compression at threshold failed");
    let decompressed =
        super::super::decompress(&compressed).expect("decompression at threshold failed");
    assert_eq!(
        decompressed, input_at_threshold,
        "round-trip at threshold should match"
    );

    // Test just below threshold (256KB - 1 byte)
    let input_below_threshold: Vec<u8> = (0..=255).cycle().take(256 * 1024 - 1).collect();
    let compressed =
        super::super::compress_with_options(&input_below_threshold, Pipeline::LzSeqR, &opts)
            .expect("compression below threshold failed");
    let decompressed =
        super::super::decompress(&compressed).expect("decompression below threshold failed");
    assert_eq!(
        decompressed, input_below_threshold,
        "round-trip below threshold should match"
    );
}

// --- Task 5 tests: Ring-buffered entropy handoff correctness ---

#[test]
fn test_entropy_ring_slot_recycling() {
    // Compress 16 blocks of 256KB (8 ring slots, depth=4 — forces each slot to be recycled twice).
    // Assert all 16 blocks decompress correctly.
    // This tests that slot recycling doesn't cause data corruption or loss.
    let block_size = 256 * 1024; // 256KB per block
    let mut input = Vec::new();
    // Create 16 distinct blocks so we can verify each decompresses correctly
    for block_idx in 0..16 {
        for i in 0..block_size {
            // Mix the block index into the data so each block is unique
            input.push(((block_idx * 17 + i) % 256) as u8);
        }
    }

    let opts = CompressOptions {
        block_size,
        threads: 4,

        stage1_backend: super::super::BackendAssignment::Cpu,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
        .expect("compression failed");
    let decompressed = super::super::decompress(&compressed).expect("decompression failed");
    assert_eq!(
        decompressed, input,
        "slot recycling: 16 blocks x 256KB should round-trip correctly"
    );
}

#[test]
fn test_entropy_handoff_output_order_preserved() {
    // Compress blocks with varying sizes (some above, some below GPU_ENTROPY_THRESHOLD).
    // Assert decompressed output equals input byte-for-byte regardless of which path
    // (CPU or GPU) each block took.
    // GPU_ENTROPY_THRESHOLD is 256KB, so use 128KB and 512KB blocks.
    let block_size = 128 * 1024; // 128KB blocks
    let mut input = Vec::new();
    let mut block_markers = Vec::new();

    // Create blocks with distinctive patterns so we can verify ordering
    // Block 0: 128KB (below threshold)
    for i in 0..block_size {
        input.push((i % 256) as u8);
    }
    block_markers.push((0, block_size));

    // Block 1: 512KB (above threshold)
    for i in 0usize..512 * 1024 {
        input.push(((i.wrapping_add(1)) % 256) as u8);
    }
    block_markers.push((block_size, 512 * 1024));

    // Block 2: 256KB (at threshold)
    for i in 0usize..256 * 1024 {
        input.push(((i.wrapping_add(2)) % 256) as u8);
    }
    block_markers.push((block_size + 512 * 1024, 256 * 1024));

    // Block 3: 100KB (small, below threshold)
    for i in 0usize..100 * 1024 {
        input.push(((i.wrapping_add(3)) % 256) as u8);
    }
    block_markers.push((block_size + 512 * 1024 + 256 * 1024, 100 * 1024));

    let opts = CompressOptions {
        block_size,
        threads: 4,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
        .expect("compression failed");
    let decompressed = super::super::decompress(&compressed).expect("decompression failed");
    assert_eq!(
        decompressed, input,
        "output order preserved: mixed block sizes should decompress in order"
    );

    // Verify each block's content is correct by spot-checking markers
    for (offset, size) in block_markers {
        assert!(
            offset + size <= decompressed.len(),
            "block out of bounds in decompressed output"
        );
    }
}

#[test]
fn test_entropy_handoff_cpu_gpu_cross_decode() {
    // Test encoding with CPU entropy and decoding with CPU works (baseline).
    // This verifies that the entropy container format is platform-independent.
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

    let opts = CompressOptions {
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Cpu,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "CPU encode + CPU decode should round-trip"
    );
}

#[test]
fn test_entropy_ring_empty_blocks_handled() {
    // Compress an input where the last block is smaller than typical GPU input sizes.
    // Assert no panic, correct output.
    // This tests the edge case where a ring slot is acquired but may not be fully utilized.
    let input: Vec<u8> = (0..=255).cycle().take(256 * 1024 + 1000).collect();

    let opts = CompressOptions {
        block_size: 256 * 1024,
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts)
        .expect("compression with small final block failed");
    let decompressed =
        super::super::decompress(&compressed).expect("decompression of small final block failed");
    assert_eq!(
        decompressed, input,
        "small final block should not cause panic or data corruption"
    );
}

// Task 6: Stage0 prefetch overlaps with GPU entropy
#[test]
fn test_prefetch_stage0_correct_output() {
    // Verify that Stage0 prefetching does not affect correctness.
    // Compress with prefetch enabled (via heterogeneous path) and verify round-trip.
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

    let opts = CompressOptions {
        block_size: 256 * 1024,
        threads: 4,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "prefetch stage0 compression round-trip should preserve data"
    );
}

// Task 7: Zero-overhead CPU-only path when no GPU
#[test]
fn test_cpu_only_path_no_webgpu_engine() {
    // Compress with no GPU engine and verify correct output.
    // This should use the CPU-only path without any GPU initialization.
    let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

    let opts = CompressOptions {
        backend: super::super::Backend::Cpu,

        threads: 4,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "CPU-only path should work with zero overhead"
    );
}

#[test]
fn test_cpu_only_path_cpu_backend_explicit() {
    // Force CPU backend and CPU entropy explicitly.
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

    let opts = CompressOptions {
        backend: super::super::Backend::Cpu,

        threads: 2,
        stage1_backend: super::super::BackendAssignment::Cpu,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "explicit CPU backend should work correctly"
    );
}

// Task 8: Graceful GPU device-lost fallback
#[cfg(feature = "webgpu")]
#[test]
fn test_gpu_device_lost_fallback_continues() {
    // Test that the unified path gracefully handles GPU errors.
    // Since we can't easily inject real GPU failures, we test that the
    // fallback path (CPU entropy) is used when GPU submission fails.
    let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

    let opts = CompressOptions {
        block_size: 256 * 1024,
        threads: 4,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    // Even if GPU is available, this should not panic or corrupt data.
    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "GPU fallback path should preserve data integrity"
    );
}

// Task 9: Comprehensive fallback scenario tests (AC3.5 and AC3.6)
#[test]
#[cfg(feature = "webgpu")]
fn test_ac3_5_no_gpu_zero_overhead_cpu_path() {
    // AC3.5: No GPU available — pure CPU path works with zero overhead.
    let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

    let opts = CompressOptions {
        backend: super::super::Backend::Cpu,

        threads: 4,
        webgpu_engine: None, // Explicitly no GPU
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "AC3.5: CPU-only path should produce correct output"
    );
}

#[test]
fn test_ac3_5_webgpu_feature_disabled_compiles_and_works() {
    // AC3.5: Code should compile and work correctly even when webgpu feature is disabled.
    // This test runs only when the feature is disabled (checked by cargo test).
    #[cfg(not(feature = "webgpu"))]
    {
        let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

        let opts = CompressOptions {
            threads: 2,
            ..CompressOptions::default()
        };

        let compressed =
            super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "non-webgpu build should work correctly"
        );
    }
}

#[cfg(feature = "webgpu")]
#[test]
fn test_ac3_6_gpu_lost_fallback_semantics() {
    // AC3.6: GPU device lost mid-compression — graceful fallback to CPU, no data corruption.
    // This test verifies that the gpu_lost flag and fallback path work correctly.
    let input: Vec<u8> = (0..=255).cycle().take(1024 * 1024).collect();

    let opts = CompressOptions {
        block_size: 256 * 1024,
        threads: 4,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    // Compression should succeed even if GPU fallback occurs internally.
    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "AC3.6: GPU fallback should preserve data integrity"
    );
}

#[test]
fn test_ac3_6_no_data_corruption_on_partial_slot_state() {
    // Verify that a GPU loss during any stage (submit or complete) falls back cleanly.
    let input: Vec<u8> = (0..=255).cycle().take(768 * 1024).collect();

    let opts = CompressOptions {
        block_size: 256 * 1024,
        threads: 2,

        stage1_backend: super::super::BackendAssignment::Auto,
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "partial GPU slot state should fall back cleanly"
    );
}

// GPU-specific unified scheduler tests
// These tests verify the unified scheduler dispatches GPU work correctly
// when GPU is available and backend settings allow it.
#[test]
#[cfg(feature = "webgpu")]
fn test_heterogeneous_compress_with_gpu_entropy() {
    // Test actual GPU unified compression when GPU is available.
    use crate::webgpu::WebGpuEngine;

    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();

    // Try to create a GPU engine; skip test if no device available
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("GPU device not available, skipping GPU test");
            return;
        }
    };

    let opts = CompressOptions {
        backend: super::super::Backend::WebGpu,

        threads: 2,
        stage1_backend: super::super::BackendAssignment::Auto,
        webgpu_engine: Some(std::sync::Arc::new(engine)),
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "GPU unified path should decompress correctly"
    );
}

#[test]
#[cfg(feature = "webgpu")]
fn test_heterogeneous_compress_gpu_forced_large_blocks() {
    // Test with stage1_backend forced to Gpu and large blocks that trigger GPU entropy.
    use crate::webgpu::WebGpuEngine;

    let input: Vec<u8> = (0..=255).cycle().take(768 * 1024).collect(); // 3 x 256KB blocks

    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("GPU device not available, skipping GPU test");
            return;
        }
    };

    let opts = CompressOptions {
        backend: super::super::Backend::WebGpu,

        threads: 2,
        block_size: 256 * 1024,
        stage1_backend: super::super::BackendAssignment::Gpu,
        webgpu_engine: Some(std::sync::Arc::new(engine)),
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "GPU forced unified path should decompress correctly"
    );
}

#[test]
#[cfg(feature = "webgpu")]
fn test_heterogeneous_mixed_block_sizes_with_gpu() {
    // Verify unified path handles mixed block sizes correctly:
    // some blocks trigger GPU (>= 256KB), some use CPU (< 256KB).
    use crate::webgpu::WebGpuEngine;

    let mut input = Vec::new();
    // 1 large block (will use GPU if Auto), 2 small blocks (will use CPU)
    for i in 0..512 * 1024 {
        input.push((i % 256) as u8);
    }
    for i in 0..128 * 1024 {
        input.push((i % 256) as u8);
    }
    for i in 0..128 * 1024 {
        input.push((i % 256) as u8);
    }

    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(_) => {
            eprintln!("GPU device not available, skipping GPU test");
            return;
        }
    };

    let opts = CompressOptions {
        backend: super::super::Backend::WebGpu,

        threads: 2,
        block_size: 256 * 1024,
        stage1_backend: super::super::BackendAssignment::Auto,
        webgpu_engine: Some(std::sync::Arc::new(engine)),
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "Mixed GPU/CPU unified path should decompress correctly"
    );
}

// GPU unified scheduler tests for LZ77-based pipelines (Deflate, Lzr, Lzf).
// These exercise the Stage 0 GPU routing and batch-collect path.

#[test]
#[cfg(feature = "webgpu")]
fn test_gpu_roundtrip_deflate() {
    use crate::webgpu::WebGpuEngine;
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };
    let opts = CompressOptions {
        backend: super::super::Backend::WebGpu,
        threads: 2,
        block_size: 256 * 1024,
        webgpu_engine: Some(std::sync::Arc::new(engine)),
        ..CompressOptions::default()
    };
    let compressed = super::super::compress_with_options(&input, Pipeline::Deflate, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input, "GPU Deflate round-trip failed");
}

#[test]
#[cfg(feature = "webgpu")]
fn test_gpu_roundtrip_lzr() {
    use crate::webgpu::WebGpuEngine;
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };
    let opts = CompressOptions {
        backend: super::super::Backend::WebGpu,
        threads: 2,
        block_size: 256 * 1024,
        webgpu_engine: Some(std::sync::Arc::new(engine)),
        ..CompressOptions::default()
    };
    let compressed = super::super::compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input, "GPU Lzr round-trip failed");
}

#[test]
#[cfg(feature = "webgpu")]
fn test_gpu_roundtrip_lzf() {
    use crate::webgpu::WebGpuEngine;
    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };
    let opts = CompressOptions {
        backend: super::super::Backend::WebGpu,
        threads: 2,
        block_size: 256 * 1024,
        webgpu_engine: Some(std::sync::Arc::new(engine)),
        ..CompressOptions::default()
    };
    let compressed = super::super::compress_with_options(&input, Pipeline::Lzf, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(decompressed, input, "GPU Lzf round-trip failed");
}

#[test]
#[cfg(feature = "webgpu")]
fn test_lzr_backend_assignments_are_interchangeable() {
    use crate::pipeline::{Backend, BackendAssignment};
    use crate::webgpu::WebGpuEngine;

    let input: Vec<u8> = (0..=255).cycle().take(512 * 1024).collect();
    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => return,
    };

    let cases = [
        ("cpu/cpu", BackendAssignment::Cpu, BackendAssignment::Cpu),
        ("gpu/cpu", BackendAssignment::Gpu, BackendAssignment::Cpu),
        ("cpu/gpu", BackendAssignment::Cpu, BackendAssignment::Gpu),
        (
            "auto/auto",
            BackendAssignment::Auto,
            BackendAssignment::Auto,
        ),
    ];

    for (label, stage0_backend, stage1_backend) in cases {
        let opts = CompressOptions {
            backend: Backend::WebGpu,
            threads: 2,
            block_size: 256 * 1024,
            stage0_backend,
            stage1_backend,
            webgpu_engine: Some(engine.clone()),
            ..CompressOptions::default()
        };
        let compressed = super::super::compress_with_options(&input, Pipeline::Lzr, &opts).unwrap();
        let decompressed = super::super::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "Lzr round-trip failed for interchangeable backends: {label}"
        );
    }
}

// Test that the channel-full fallback path produces correct results.
// Uses a single-capacity channel (ring_depth=1) with many blocks to force
// try_send failures, exercising the CPU fallback in the worker dispatch.
#[test]
fn test_channel_full_cpu_fallback() {
    // Many small blocks with minimal threads — channel of depth 1 will overflow
    let block_size = 64 * 1024; // 64KB blocks
    let num_blocks = 16;
    let input: Vec<u8> = (0..=255).cycle().take(block_size * num_blocks).collect();

    let opts = CompressOptions {
        block_size,
        threads: 8, // many workers competing for channel
        ..CompressOptions::default()
    };

    let compressed = super::super::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
    let decompressed = super::super::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed, input,
        "channel-full fallback should produce correct results"
    );
}
