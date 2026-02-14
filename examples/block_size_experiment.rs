/// Block-size experiment: many small blocks vs. few large blocks.
///
/// Tests the hypothesis that issuing many small blocks to the GPU (e.g.,
/// 64×32KB) achieves higher throughput than fewer large blocks (e.g., 1×2MB),
/// because smaller blocks enable better GPU/CPU overlap and PCI pipelining.
///
/// Measures wall-clock throughput (MB/s) and compression ratio across a sweep
/// of block sizes, for both CPU and GPU backends.
///
/// Usage:
///   cargo run --example block_size_experiment --release --features webgpu
///   cargo run --example block_size_experiment --release  # CPU-only baseline
use std::path::Path;
use std::time::Instant;

use pz::pipeline::{self, CompressOptions, Pipeline};

/// Number of timed iterations per configuration (median reported).
const ITERS: usize = 5;

/// Total input sizes to test.
const INPUT_SIZES: &[usize] = &[2_097_152, 4_194_304]; // 2MB, 4MB

/// Block sizes to sweep (bytes).
const BLOCK_SIZES: &[usize] = &[
    16 * 1024,   // 16KB
    32 * 1024,   // 32KB
    64 * 1024,   // 64KB
    128 * 1024,  // 128KB
    256 * 1024,  // 256KB (current default)
    512 * 1024,  // 512KB
    1024 * 1024, // 1MB
    2048 * 1024, // 2MB (single block for 2MB input)
    4096 * 1024, // 4MB (single block for 4MB input)
];

/// Pipelines to test (LZ77-based, GPU-eligible).
const PIPELINES: &[Pipeline] = &[Pipeline::Deflate, Pipeline::Lzf];

fn load_data(size: usize) -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Try Canterbury corpus (realistic, mixed-content data)
    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                if decompressed.len() >= size {
                    return decompressed[..size].to_vec();
                }
                // Tile to fill requested size
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

    // Fallback: synthetic repetitive text
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let full = pattern.repeat((size / pattern.len()) + 1);
    full[..size].to_vec()
}

struct TrialResult {
    block_size: usize,
    input_size: usize,
    num_blocks: usize,
    compressed_size: usize,
    ratio: f64,
    median_us: f64,
    throughput_mbs: f64,
    backend: &'static str,
    pipeline: Pipeline,
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{}MB", bytes / 1_048_576)
    } else if bytes >= 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{}B", bytes)
    }
}

fn run_trial(
    data: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
    backend_name: &'static str,
) -> TrialResult {
    let block_size = options.block_size;
    let input_size = data.len();
    let num_blocks = input_size.div_ceil(block_size);

    // Warmup
    let compressed = pipeline::compress_with_options(data, pipeline, options).unwrap();
    let compressed_size = compressed.len();

    // Verify round-trip correctness
    let decompressed = pipeline::decompress(&compressed).unwrap();
    assert_eq!(
        decompressed.len(),
        data.len(),
        "Round-trip length mismatch! block_size={}, pipeline={:?}, backend={}",
        block_size,
        pipeline,
        backend_name,
    );
    assert_eq!(
        decompressed, data,
        "Round-trip data mismatch! block_size={}, pipeline={:?}, backend={}",
        block_size, pipeline, backend_name,
    );

    // Timed iterations
    let mut times_us = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let _ =
            std::hint::black_box(pipeline::compress_with_options(data, pipeline, options).unwrap());
        times_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[ITERS / 2];
    let throughput_mbs = (input_size as f64) / median_us; // bytes/us = MB/s
    let ratio = compressed_size as f64 / input_size as f64;

    TrialResult {
        block_size,
        input_size,
        num_blocks,
        compressed_size,
        ratio,
        median_us,
        throughput_mbs,
        backend: backend_name,
        pipeline,
    }
}

fn print_results(results: &[TrialResult]) {
    println!(
        "\n{:<10} {:<8} {:<8} {:<7} {:<12} {:<10} {:<12} {:<10}",
        "Pipeline", "Backend", "Input", "BlkSz", "Blocks", "Ratio", "Time(us)", "MB/s"
    );
    println!("{}", "=".repeat(88));

    let mut last_pipeline = None;
    let mut last_backend = None;
    let mut last_input = None;

    for r in results {
        // Print separator between groups
        let group_key = (
            format!("{:?}", r.pipeline),
            r.backend.to_string(),
            r.input_size,
        );
        let prev_key = (
            last_pipeline.clone().unwrap_or_default(),
            last_backend.clone().unwrap_or_default(),
            last_input.unwrap_or(0),
        );
        if last_pipeline.is_some() && group_key != prev_key {
            println!("{}", "-".repeat(88));
        }
        last_pipeline = Some(format!("{:?}", r.pipeline));
        last_backend = Some(r.backend.to_string());
        last_input = Some(r.input_size);

        println!(
            "{:<10} {:<8} {:<8} {:<7} {:<12} {:<10.4} {:<12.0} {:<10.1}",
            format!("{:?}", r.pipeline),
            r.backend,
            format_size(r.input_size),
            format_size(r.block_size),
            r.num_blocks,
            r.ratio,
            r.median_us,
            r.throughput_mbs,
        );
    }
}

fn print_csv(results: &[TrialResult]) {
    println!("\n--- CSV ---");
    println!("pipeline,backend,input_bytes,block_size_bytes,num_blocks,compressed_bytes,ratio,median_us,throughput_mbs");
    for r in results {
        println!(
            "{:?},{},{},{},{},{},{:.6},{:.1},{:.2}",
            r.pipeline,
            r.backend,
            r.input_size,
            r.block_size,
            r.num_blocks,
            r.compressed_size,
            r.ratio,
            r.median_us,
            r.throughput_mbs,
        );
    }
}

fn main() {
    println!("=== Block Size Experiment ===");
    println!("Hypothesis: many small GPU blocks > few large GPU blocks for throughput");
    println!("Iterations per config: {ITERS} (reporting median)");
    println!();

    let mut all_results = Vec::new();

    // --- CPU baseline ---
    println!("Running CPU baseline...");
    for &input_size in INPUT_SIZES {
        let data = load_data(input_size);
        for &pipeline in PIPELINES {
            for &block_size in BLOCK_SIZES {
                if block_size > input_size * 2 {
                    continue; // skip nonsensical configs
                }
                let options = CompressOptions {
                    backend: pipeline::Backend::Cpu,
                    threads: 0, // auto (use all cores)
                    block_size,
                    ..Default::default()
                };
                let r = run_trial(&data, pipeline, &options, "CPU");
                eprint!(
                    "  {:?} {} blk={} => {:.1} MB/s ratio={:.4}\r",
                    pipeline,
                    format_size(input_size),
                    format_size(block_size),
                    r.throughput_mbs,
                    r.ratio,
                );
                all_results.push(r);
            }
        }
    }
    eprintln!();

    // --- WebGPU ---
    #[cfg(feature = "webgpu")]
    {
        use pz::webgpu::WebGpuEngine;
        match WebGpuEngine::new() {
            Ok(engine) => {
                let engine = std::sync::Arc::new(engine);
                println!(
                    "Running WebGPU trials (device: {})...",
                    engine.device_name()
                );
                for &input_size in INPUT_SIZES {
                    let data = load_data(input_size);
                    for &pipeline in PIPELINES {
                        for &block_size in BLOCK_SIZES {
                            if block_size > input_size * 2 {
                                continue;
                            }
                            let options = CompressOptions {
                                backend: pipeline::Backend::WebGpu,
                                threads: 0,
                                block_size,
                                webgpu_engine: Some(engine.clone()),
                                ..Default::default()
                            };
                            let r = run_trial(&data, pipeline, &options, "WebGPU");
                            eprint!(
                                "  {:?} {} blk={} => {:.1} MB/s ratio={:.4}\r",
                                pipeline,
                                format_size(input_size),
                                format_size(block_size),
                                r.throughput_mbs,
                                r.ratio,
                            );
                            all_results.push(r);
                        }
                    }
                }
                eprintln!();
            }
            Err(e) => {
                eprintln!("WebGPU unavailable: {:?}", e);
            }
        }
    }

    // --- Results ---
    print_results(&all_results);
    print_csv(&all_results);

    // --- Analysis: find optimal block size per (pipeline, backend, input_size) ---
    println!("\n=== Optimal Block Size per Configuration ===");
    println!(
        "{:<10} {:<8} {:<8} {:<10} {:<10} {:<10}",
        "Pipeline", "Backend", "Input", "Best BlkSz", "MB/s", "Ratio"
    );
    println!("{}", "-".repeat(60));

    let mut groups: std::collections::BTreeMap<(String, String, usize), Vec<&TrialResult>> =
        std::collections::BTreeMap::new();
    for r in &all_results {
        let key = (
            format!("{:?}", r.pipeline),
            r.backend.to_string(),
            r.input_size,
        );
        groups.entry(key).or_default().push(r);
    }
    for (key, trials) in &groups {
        if let Some(best) = trials
            .iter()
            .max_by(|a, b| a.throughput_mbs.partial_cmp(&b.throughput_mbs).unwrap())
        {
            println!(
                "{:<10} {:<8} {:<8} {:<10} {:<10.1} {:<10.4}",
                key.0,
                key.1,
                format_size(key.2),
                format_size(best.block_size),
                best.throughput_mbs,
                best.ratio,
            );
        }
    }

    println!("\nDone.");
}
