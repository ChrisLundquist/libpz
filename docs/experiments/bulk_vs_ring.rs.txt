//! Compare single-dispatch bulk LZ77 vs per-block ring/batched dispatch.
//!
//! Usage: cargo run --release --example bulk_vs_ring

#[cfg(feature = "webgpu")]
fn main() {
    use pz::webgpu::WebGpuEngine;
    use std::time::Instant;

    let engine = WebGpuEngine::new().expect("WebGPU init failed");

    // Generate 4MB test data with some compressible patterns
    let size = 4 * 1024 * 1024;
    let mut data = vec![0u8; size];
    for (i, b) in data.iter_mut().enumerate() {
        *b = ((i * 7 + i / 256) % 256) as u8;
    }

    let block_size = 128 * 1024; // 128KB GPU pipeline block size
    let blocks: Vec<&[u8]> = data.chunks(block_size).collect();
    let num_blocks = blocks.len();

    println!(
        "Input: {} bytes ({} x {}KB blocks)",
        size,
        num_blocks,
        block_size / 1024
    );
    println!();

    // Warmup
    let _ = engine.find_matches(&data[..block_size]);

    // --- 1. Single dispatch on full input (find_matches) ---
    let iters = 5;
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let matches = engine.find_matches(&data).unwrap();
        let elapsed = t0.elapsed();
        times.push(elapsed.as_secs_f64());
        std::hint::black_box(matches);
    }
    let median = median_f64(&mut times);
    println!(
        "find_matches (full {}MB, single dispatch):  {:.1} ms  ({:.1} MB/s)",
        size / (1024 * 1024),
        median * 1000.0,
        size as f64 / median / 1e6
    );

    // --- 2. Bulk dispatch (single GPU round-trip, CPU-side split) ---
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let match_vecs = engine.find_matches_bulk(&data, block_size).unwrap();
        let elapsed = t0.elapsed();
        times.push(elapsed.as_secs_f64());
        std::hint::black_box(match_vecs);
    }
    let median = median_f64(&mut times);
    println!(
        "find_matches_bulk ({} blocks, single dispatch): {:.1} ms  ({:.1} MB/s)",
        num_blocks,
        median * 1000.0,
        size as f64 / median / 1e6
    );

    // --- 3. Batched dispatch (per-block ring or alloc path) ---
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let match_vecs = engine.find_matches_batched(&blocks).unwrap();
        let elapsed = t0.elapsed();
        times.push(elapsed.as_secs_f64());
        std::hint::black_box(match_vecs);
    }
    let median = median_f64(&mut times);
    println!(
        "find_matches_batched ({} blocks, per-block dispatch): {:.1} ms  ({:.1} MB/s)",
        num_blocks,
        median * 1000.0,
        size as f64 / median / 1e6
    );

    // --- 4. Full pipeline: compress_with_options (tests the wired-in path) ---
    {
        use pz::pipeline::{self, Backend, CompressOptions, Pipeline};

        let options = CompressOptions {
            backend: Backend::WebGpu,
            threads: 4,
            block_size,
            parse_strategy: pipeline::ParseStrategy::Auto,
            webgpu_engine: Some(std::sync::Arc::new(
                WebGpuEngine::new().expect("WebGPU init"),
            )),
            ..Default::default()
        };

        // Warmup
        let _ = pipeline::compress_with_options(&data, Pipeline::Lzf, &options);

        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let compressed =
                pipeline::compress_with_options(&data, Pipeline::Lzf, &options).unwrap();
            let elapsed = t0.elapsed();
            times.push(elapsed.as_secs_f64());
            std::hint::black_box(compressed);
        }
        let median = median_f64(&mut times);
        println!(
            "pipeline Lzf compress (4 threads, {}KB blocks): {:.1} ms  ({:.1} MB/s)",
            block_size / 1024,
            median * 1000.0,
            size as f64 / median / 1e6
        );
    }
}

fn median_f64(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[cfg(not(feature = "webgpu"))]
fn main() {
    eprintln!("This example requires the 'webgpu' feature");
}
