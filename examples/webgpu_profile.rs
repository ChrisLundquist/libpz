//! WebGPU performance profiling: breaks down time per stage.
//!
//! Usage:
//!   cargo build --release --features webgpu --example webgpu_profile
//!   ./target/release/examples/webgpu_profile

fn main() {
    #[cfg(feature = "webgpu")]
    run();

    #[cfg(not(feature = "webgpu"))]
    eprintln!("Requires --features webgpu");
}

#[cfg(feature = "webgpu")]
fn run() {
    use pz::pipeline::{self, Backend, CompressOptions, Pipeline};
    use pz::webgpu::WebGpuEngine;
    use std::sync::Arc;
    use std::time::Instant;

    // Load test data
    let data = load_data(256 * 1024); // 256KB - one block
    let large_data = load_data(4 * 1024 * 1024); // 4MB - multi-block

    eprintln!("=== WebGPU Performance Profile ===\n");

    // Phase 1: Engine creation time
    let t0 = Instant::now();
    let engine = WebGpuEngine::new().expect("No WebGPU device");
    let engine_time = t0.elapsed();
    eprintln!("Device: {}", engine.device_name());
    eprintln!(
        "Engine creation: {:.1} ms\n",
        engine_time.as_secs_f64() * 1000.0
    );

    let engine = Arc::new(engine);

    // Phase 2: Single-block LZ77 match finding
    eprintln!("--- Single block LZ77 (256KB) ---");
    // Warmup
    let _ = engine.find_matches(&data).unwrap();

    let iters = 20;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(engine.find_matches(&data).unwrap());
    }
    let lz77_single = t0.elapsed() / iters;
    let mbps = data.len() as f64 / lz77_single.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "GPU LZ77 (lazy): {:.2} ms ({:.1} MB/s)",
        lz77_single.as_secs_f64() * 1000.0,
        mbps
    );

    // CPU comparison
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(pz::lz77::compress_lazy_to_matches(&data).unwrap());
    }
    let cpu_lz77 = t0.elapsed() / iters;
    let cpu_mbps = data.len() as f64 / cpu_lz77.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU LZ77 (lazy): {:.2} ms ({:.1} MB/s)",
        cpu_lz77.as_secs_f64() * 1000.0,
        cpu_mbps
    );
    eprintln!(
        "GPU/CPU ratio:   {:.2}x",
        cpu_lz77.as_secs_f64() / lz77_single.as_secs_f64()
    );

    // Phase 3: Batched LZ77 (multi-block)
    eprintln!("\n--- Batched LZ77 (4MB = 16 blocks) ---");
    let blocks: Vec<&[u8]> = large_data.chunks(256 * 1024).collect();
    let nblocks = blocks.len();

    // Warmup
    let _ = engine.find_matches_batched(&blocks).unwrap();

    let iters = 5;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(engine.find_matches_batched(&blocks).unwrap());
    }
    let batch_time = t0.elapsed() / iters;
    let batch_mbps = large_data.len() as f64 / batch_time.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "GPU batched ({nblocks} blocks): {:.2} ms ({:.1} MB/s)",
        batch_time.as_secs_f64() * 1000.0,
        batch_mbps
    );
    eprintln!(
        "Per-block avg:                {:.2} ms",
        batch_time.as_secs_f64() * 1000.0 / nblocks as f64
    );

    // Phase 4: Huffman stages
    eprintln!("\n--- Huffman encode (256KB) ---");
    let _matches = engine.find_matches(&data).unwrap();

    // Build frequency table + tree
    let mut freq = pz::frequency::FrequencyTable::new();
    freq.count(&data);
    let tree = pz::huffman::HuffmanTree::from_frequency_table(&freq).unwrap();
    let mut code_lut = [0u32; 256];
    for byte in 0..=255u8 {
        let (codeword, bits) = tree.get_code(byte);
        code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
    }

    // GPU histogram
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(engine.byte_histogram(&data).unwrap());
    }
    let hist_time = t0.elapsed() / iters;
    eprintln!(
        "GPU histogram:    {:.2} ms",
        hist_time.as_secs_f64() * 1000.0
    );

    // GPU Huffman encode
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(engine.huffman_encode_gpu_scan(&data, &code_lut).unwrap());
    }
    let huff_time = t0.elapsed() / iters;
    eprintln!(
        "GPU Huffman enc:  {:.2} ms",
        huff_time.as_secs_f64() * 1000.0
    );

    // CPU Huffman
    let t0 = Instant::now();
    for _ in 0..20 {
        let _ = std::hint::black_box(tree.encode(&data).unwrap());
    }
    let cpu_huff = t0.elapsed() / 20;
    eprintln!(
        "CPU Huffman enc:  {:.2} ms",
        cpu_huff.as_secs_f64() * 1000.0
    );

    // Phase 5: Full pipeline end-to-end (deflate)
    eprintln!("\n--- Full pipeline compress (4MB, deflate) ---");
    let opts_gpu = CompressOptions {
        backend: Backend::WebGpu,
        webgpu_engine: Some(engine.clone()),
        ..Default::default()
    };
    let opts_cpu = CompressOptions::default();

    // Warmup
    let _ = pipeline::compress_with_options(&large_data, Pipeline::Deflate, &opts_gpu).unwrap();

    let iters = 3;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(
            pipeline::compress_with_options(&large_data, Pipeline::Deflate, &opts_gpu).unwrap(),
        );
    }
    let gpu_full = t0.elapsed() / iters;
    let gpu_full_mbps = large_data.len() as f64 / gpu_full.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "GPU deflate: {:.0} ms ({:.1} MB/s)",
        gpu_full.as_secs_f64() * 1000.0,
        gpu_full_mbps
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(
            pipeline::compress_with_options(&large_data, Pipeline::Deflate, &opts_cpu).unwrap(),
        );
    }
    let cpu_full = t0.elapsed() / iters;
    let cpu_full_mbps = large_data.len() as f64 / cpu_full.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU deflate: {:.0} ms ({:.1} MB/s)",
        cpu_full.as_secs_f64() * 1000.0,
        cpu_full_mbps
    );
    eprintln!(
        "GPU/CPU:     {:.2}x",
        cpu_full.as_secs_f64() / gpu_full.as_secs_f64()
    );

    // Phase 5b: Lzfi pipeline (LZ77 + interleaved FSE) — full WebGPU acceleration
    eprintln!("\n--- Full pipeline compress (4MB, lzfi) ---");
    // Warmup
    let _ = pipeline::compress_with_options(&large_data, Pipeline::Lzfi, &opts_gpu).unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(
            pipeline::compress_with_options(&large_data, Pipeline::Lzfi, &opts_gpu).unwrap(),
        );
    }
    let gpu_lzfi = t0.elapsed() / iters;
    let gpu_lzfi_mbps = large_data.len() as f64 / gpu_lzfi.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "GPU lzfi:    {:.0} ms ({:.1} MB/s)",
        gpu_lzfi.as_secs_f64() * 1000.0,
        gpu_lzfi_mbps
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(
            pipeline::compress_with_options(&large_data, Pipeline::Lzfi, &opts_cpu).unwrap(),
        );
    }
    let cpu_lzfi = t0.elapsed() / iters;
    let cpu_lzfi_mbps = large_data.len() as f64 / cpu_lzfi.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU lzfi:    {:.0} ms ({:.1} MB/s)",
        cpu_lzfi.as_secs_f64() * 1000.0,
        cpu_lzfi_mbps
    );
    eprintln!(
        "GPU/CPU:     {:.2}x",
        cpu_lzfi.as_secs_f64() / gpu_lzfi.as_secs_f64()
    );

    // Phase 5c: Lzfi decompress
    eprintln!("\n--- Full pipeline decompress (4MB, lzfi) ---");
    let compressed_lzfi_gpu =
        pipeline::compress_with_options(&large_data, Pipeline::Lzfi, &opts_gpu).unwrap();
    let compressed_lzfi_cpu =
        pipeline::compress_with_options(&large_data, Pipeline::Lzfi, &opts_cpu).unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(pipeline::decompress(&compressed_lzfi_gpu).unwrap());
    }
    let dec_lzfi = t0.elapsed() / iters;
    let dec_lzfi_mbps = large_data.len() as f64 / dec_lzfi.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU decompress (gpu-compressed): {:.0} ms ({:.1} MB/s)",
        dec_lzfi.as_secs_f64() * 1000.0,
        dec_lzfi_mbps
    );

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(pipeline::decompress(&compressed_lzfi_cpu).unwrap());
    }
    let dec_lzfi_cpu = t0.elapsed() / iters;
    let dec_lzfi_cpu_mbps =
        large_data.len() as f64 / dec_lzfi_cpu.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU decompress (cpu-compressed): {:.0} ms ({:.1} MB/s)",
        dec_lzfi_cpu.as_secs_f64() * 1000.0,
        dec_lzfi_cpu_mbps
    );

    // Phase 5d: LzssR pipeline (CPU-only, for comparison)
    eprintln!("\n--- Full pipeline compress (4MB, lzssr) ---");
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(
            pipeline::compress_with_options(&large_data, Pipeline::LzssR, &opts_cpu).unwrap(),
        );
    }
    let cpu_lzssr = t0.elapsed() / iters;
    let cpu_lzssr_mbps = large_data.len() as f64 / cpu_lzssr.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU lzssr:   {:.0} ms ({:.1} MB/s)",
        cpu_lzssr.as_secs_f64() * 1000.0,
        cpu_lzssr_mbps
    );

    let compressed_lzssr =
        pipeline::compress_with_options(&large_data, Pipeline::LzssR, &opts_cpu).unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(pipeline::decompress(&compressed_lzssr).unwrap());
    }
    let dec_lzssr = t0.elapsed() / iters;
    let dec_lzssr_mbps = large_data.len() as f64 / dec_lzssr.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU decompress lzssr: {:.0} ms ({:.1} MB/s)",
        dec_lzssr.as_secs_f64() * 1000.0,
        dec_lzssr_mbps
    );

    // Compression ratio comparison
    eprintln!("\n--- Compression ratios (4MB) ---");
    let compressed_deflate =
        pipeline::compress_with_options(&large_data, Pipeline::Deflate, &opts_cpu).unwrap();
    eprintln!(
        "deflate: {:.2}%",
        compressed_deflate.len() as f64 / large_data.len() as f64 * 100.0
    );
    eprintln!(
        "lzfi:    {:.2}%",
        compressed_lzfi_cpu.len() as f64 / large_data.len() as f64 * 100.0
    );
    eprintln!(
        "lzssr:   {:.2}%",
        compressed_lzssr.len() as f64 / large_data.len() as f64 * 100.0
    );

    // Phase 6: Buffer allocation overhead
    eprintln!("\n--- Buffer allocation overhead ---");
    let t0 = Instant::now();
    for _ in 0..100 {
        let _buf = engine.find_matches_to_device(&data).unwrap();
    }
    let to_device = t0.elapsed() / 100;
    eprintln!(
        "find_matches_to_device (256KB): {:.2} ms",
        to_device.as_secs_f64() * 1000.0
    );

    // Phase 7: Single-threaded CPU comparison
    eprintln!("\n--- Single-threaded CPU deflate (4MB) ---");
    let opts_cpu_1t = CompressOptions {
        threads: 1,
        ..Default::default()
    };

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = std::hint::black_box(
            pipeline::compress_with_options(&large_data, Pipeline::Deflate, &opts_cpu_1t).unwrap(),
        );
    }
    let cpu_1t = t0.elapsed() / iters;
    let cpu_1t_mbps = large_data.len() as f64 / cpu_1t.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU 1-thread: {:.0} ms ({:.1} MB/s)",
        cpu_1t.as_secs_f64() * 1000.0,
        cpu_1t_mbps
    );

    // Phase 8: Larger inputs for GPU
    eprintln!("\n--- Large input LZ77 (1MB single block) ---");
    let big_block = load_data(1024 * 1024);
    // Warmup
    let _ = engine.find_matches(&big_block).unwrap();
    let iters_big = 10;
    let t0 = Instant::now();
    for _ in 0..iters_big {
        let _ = std::hint::black_box(engine.find_matches(&big_block).unwrap());
    }
    let gpu_1m = t0.elapsed() / iters_big;
    let gpu_1m_mbps = big_block.len() as f64 / gpu_1m.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "GPU LZ77 1MB:   {:.2} ms ({:.1} MB/s)",
        gpu_1m.as_secs_f64() * 1000.0,
        gpu_1m_mbps
    );

    let t0 = Instant::now();
    for _ in 0..iters_big {
        let _ = std::hint::black_box(pz::lz77::compress_lazy_to_matches(&big_block).unwrap());
    }
    let cpu_1m = t0.elapsed() / iters_big;
    let cpu_1m_mbps = big_block.len() as f64 / cpu_1m.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!(
        "CPU LZ77 1MB:   {:.2} ms ({:.1} MB/s)",
        cpu_1m.as_secs_f64() * 1000.0,
        cpu_1m_mbps
    );

    // Phase 10: Readback overhead
    eprintln!("\n--- Match readback overhead ---");
    let match_buf = engine.find_matches_to_device(&data).unwrap();
    let t0 = Instant::now();
    for _ in 0..50 {
        let _ = std::hint::black_box(engine.download_and_dedupe(&match_buf, &data).unwrap());
    }
    let readback = t0.elapsed() / 50;
    eprintln!(
        "download_and_dedupe (256KB): {:.2} ms",
        readback.as_secs_f64() * 1000.0
    );

    // Phase 11: Measure submit vs readback time separately
    eprintln!("\n--- Submit vs Readback breakdown (256KB) ---");
    // Time just the GPU submission (no readback)
    let t0 = Instant::now();
    for _ in 0..20 {
        let _ = std::hint::black_box(engine.find_matches_to_device(&data).unwrap());
    }
    let submit_time = t0.elapsed() / 20;
    eprintln!(
        "submit (build+find+resolve): {:.2} ms",
        submit_time.as_secs_f64() * 1000.0
    );
    eprintln!(
        "download_and_dedupe:         {:.2} ms",
        readback.as_secs_f64() * 1000.0
    );
    eprintln!(
        "total find_matches:          {:.2} ms",
        lz77_single.as_secs_f64() * 1000.0
    );
    eprintln!(
        "submit % of total:           {:.0}%",
        submit_time.as_secs_f64() / lz77_single.as_secs_f64() * 100.0
    );
    eprintln!(
        "readback % of total:         {:.0}%",
        readback.as_secs_f64() / lz77_single.as_secs_f64() * 100.0
    );

    // Phase 12: Hash table buffer allocation cost
    eprintln!("\n--- Hash table buffer size ---");
    let hash_table_bytes = 32768 * 64 * 4; // HASH_TABLE_SIZE * BUCKET_CAP * sizeof(u32)
    let hash_counts_bytes = 32768 * 4;
    let match_buf_bytes = 256 * 1024 * 12; // input_len * sizeof(GpuMatch)
    eprintln!(
        "hash_table:   {} MB",
        hash_table_bytes as f64 / 1024.0 / 1024.0
    );
    eprintln!("hash_counts:  {} KB", hash_counts_bytes / 1024);
    eprintln!(
        "match_buf:    {} MB",
        match_buf_bytes as f64 / 1024.0 / 1024.0
    );
    eprintln!(
        "total/block:  {:.1} MB",
        (hash_table_bytes + hash_counts_bytes + match_buf_bytes * 2 + 256 * 1024) as f64
            / 1024.0
            / 1024.0
    );

    eprintln!("\n=== Summary ===");
    eprintln!("-- LZ77 stage --");
    eprintln!(
        "GPU LZ77 single block (256KB): {:.2} ms ({:.1} MB/s)",
        lz77_single.as_secs_f64() * 1000.0,
        mbps
    );
    eprintln!(
        "CPU LZ77 single block (256KB): {:.2} ms ({:.1} MB/s)",
        cpu_lz77.as_secs_f64() * 1000.0,
        cpu_mbps
    );
    eprintln!(
        "GPU LZ77 single block (1MB):   {:.2} ms ({:.1} MB/s)",
        gpu_1m.as_secs_f64() * 1000.0,
        gpu_1m_mbps
    );
    eprintln!(
        "CPU LZ77 single block (1MB):   {:.2} ms ({:.1} MB/s)",
        cpu_1m.as_secs_f64() * 1000.0,
        cpu_1m_mbps
    );
    eprintln!(
        "GPU batched 16 blocks (4MB):   {:.2} ms ({:.1} MB/s)",
        batch_time.as_secs_f64() * 1000.0,
        batch_mbps
    );
    eprintln!("\n-- Full pipelines (4MB) --");
    eprintln!(
        "GPU deflate:  {:.0} ms ({:.1} MB/s)",
        gpu_full.as_secs_f64() * 1000.0,
        gpu_full_mbps
    );
    eprintln!(
        "CPU deflate:  {:.0} ms ({:.1} MB/s)",
        cpu_full.as_secs_f64() * 1000.0,
        cpu_full_mbps
    );
    eprintln!(
        "CPU 1T defl:  {:.0} ms ({:.1} MB/s)",
        cpu_1t.as_secs_f64() * 1000.0,
        cpu_1t_mbps
    );
    eprintln!(
        "GPU lzfi:     {:.0} ms ({:.1} MB/s)",
        gpu_lzfi.as_secs_f64() * 1000.0,
        gpu_lzfi_mbps
    );
    eprintln!(
        "CPU lzfi:     {:.0} ms ({:.1} MB/s)",
        cpu_lzfi.as_secs_f64() * 1000.0,
        cpu_lzfi_mbps
    );
    eprintln!(
        "CPU lzssr:    {:.0} ms ({:.1} MB/s)",
        cpu_lzssr.as_secs_f64() * 1000.0,
        cpu_lzssr_mbps
    );

    // --- GPU Profiling Trace ---
    // Create a separate profiling engine to collect one clean trace without
    // benchmark overhead. Each GPU dispatch is timed via wgpu-profiler.
    eprintln!("\n--- GPU Profiling Trace ---");
    let profile_engine = WebGpuEngine::with_profiling(true).expect("No WebGPU device");
    eprintln!(
        "Profiler active: {}",
        if profile_engine.profiling() {
            "yes"
        } else {
            "no"
        }
    );

    // Run key GPU operations once for the trace
    let _ = profile_engine.find_matches(&data).unwrap();
    let _ = profile_engine.byte_histogram(&data).unwrap();
    let _ = profile_engine
        .huffman_encode_gpu_scan(&data, &code_lut)
        .unwrap();

    // Collect and write profiler results
    if let Some(results) = profile_engine.profiler_end_frame() {
        let trace_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("profiling");
        std::fs::create_dir_all(&trace_dir).ok();
        let trace_path = trace_dir.join("webgpu_trace.json");
        WebGpuEngine::profiler_write_trace(&trace_path, &results).unwrap();
        eprintln!("Chrome trace written to {}", trace_path.display());
        eprintln!("View at chrome://tracing or https://ui.perfetto.dev/");
    } else {
        eprintln!("GPU timestamps not available (profiler returned no results)");
    }
}

#[cfg(feature = "webgpu")]
fn load_data(size: usize) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("samples")
        .join("cantrbry.tar.gz");
    if path.exists() {
        if let Ok(gz_data) = std::fs::read(&path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                if decompressed.len() >= size {
                    return decompressed[..size].to_vec();
                }
                let mut data = Vec::with_capacity(size);
                while data.len() < size {
                    let remaining = size - data.len();
                    let chunk = remaining.min(decompressed.len());
                    data.extend_from_slice(&decompressed[..chunk]);
                }
                return data;
            }
        }
    }
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let full = pattern.repeat((size / pattern.len()) + 1);
    full[..size].to_vec()
}
