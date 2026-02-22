//! Compare local kernel vs coop kernel: speed and compression ratio.
//!
//! Usage: cargo run --release --example local_vs_coop

#[cfg(feature = "webgpu")]
fn main() {
    use pz::webgpu::WebGpuEngine;
    use std::time::Instant;

    let engine = WebGpuEngine::new().expect("WebGPU init failed");

    // Test with multiple data types
    let test_cases: Vec<(&str, Vec<u8>)> = vec![
        ("synthetic_4MB", {
            let size = 4 * 1024 * 1024;
            let mut data = vec![0u8; size];
            for (i, b) in data.iter_mut().enumerate() {
                *b = ((i * 7 + i / 256) % 256) as u8;
            }
            data
        }),
        ("random_64KB", {
            let size = 64 * 1024;
            let mut data = vec![0u8; size];
            let mut state: u32 = 0xDEADBEEF;
            for b in data.iter_mut() {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                *b = state as u8;
            }
            data
        }),
        ("mixed_128KB", {
            // Semi-compressible: mix of repeated text and varying data
            let size = 128 * 1024;
            let mut data = Vec::with_capacity(size);
            let phrases = [
                b"compression is important ".as_slice(),
                b"data structures matter ".as_slice(),
                b"algorithms are fun ".as_slice(),
            ];
            let mut idx = 0;
            while data.len() < size {
                data.extend_from_slice(phrases[idx % phrases.len()]);
                // Add some noise between phrases
                for j in 0..16 {
                    data.push((idx * 37 + j * 13) as u8);
                }
                idx += 1;
            }
            data.truncate(size);
            data
        }),
    ];

    // Warmup
    let warmup_data = vec![0u8; 8192];
    let _ = engine.find_matches(&warmup_data);
    let _ = engine.find_matches_coop(&warmup_data);

    println!(
        "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Dataset", "Local ms", "Coop ms", "Speedup", "Local #M", "Coop #M", "Ratio diff"
    );
    println!("{}", "-".repeat(95));

    for (name, data) in &test_cases {
        let iters = 5;

        // Local kernel (new default)
        let mut local_times = Vec::with_capacity(iters);
        let mut local_matches = 0usize;
        let mut local_match_bytes = 0usize;
        for i in 0..iters {
            let t0 = Instant::now();
            let matches = engine.find_matches(data).unwrap();
            let elapsed = t0.elapsed();
            local_times.push(elapsed.as_secs_f64());
            if i == 0 {
                local_match_bytes = matches.iter().map(|m| m.length as usize).sum::<usize>();
                local_matches = matches.len();
            }
        }
        local_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let local_median = local_times[iters / 2];

        // Coop kernel (old default)
        let mut coop_times = Vec::with_capacity(iters);
        let mut coop_matches = 0usize;
        let mut coop_match_bytes = 0usize;
        for i in 0..iters {
            let t0 = Instant::now();
            let matches = engine.find_matches_coop(data).unwrap();
            let elapsed = t0.elapsed();
            coop_times.push(elapsed.as_secs_f64());
            if i == 0 {
                coop_match_bytes = matches.iter().map(|m| m.length as usize).sum::<usize>();
                coop_matches = matches.len();
            }
        }
        coop_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let coop_median = coop_times[iters / 2];

        let speedup = coop_median / local_median;
        // match_bytes / input_len = fraction of input covered by matches (higher = better compression)
        let local_coverage = local_match_bytes as f64 / data.len() as f64 * 100.0;
        let coop_coverage = coop_match_bytes as f64 / data.len() as f64 * 100.0;

        println!(
            "{:<25} {:>9.1} {:>9.1} {:>9.2}x {:>9} {:>9} {:>8.1}pp",
            name,
            local_median * 1000.0,
            coop_median * 1000.0,
            speedup,
            local_matches,
            coop_matches,
            local_coverage - coop_coverage,
        );
    }

    println!();
    println!("Local = per-workgroup 4KB shared-memory hash table (lz77_local.wgsl)");
    println!("Coop  = cooperative 32KB window with 1788 probes (lz77_coop.wgsl)");
    println!("#M    = number of match+literal tokens emitted");
    println!("Ratio diff = local match coverage - coop match coverage (negative = coop better)");

    // Also show end-to-end pipeline compression ratio
    println!();
    println!("--- End-to-end pipeline compression ratio (Lzf) ---");
    {
        use pz::pipeline::{self, Backend, CompressOptions, Pipeline};

        for (name, data) in &test_cases {
            // GPU local (current default)
            let options_gpu = CompressOptions {
                backend: Backend::WebGpu,
                threads: 1,
                webgpu_engine: Some(std::sync::Arc::new(
                    WebGpuEngine::new().expect("WebGPU init"),
                )),
                ..Default::default()
            };
            let compressed_gpu =
                pipeline::compress_with_options(data, Pipeline::Lzf, &options_gpu).unwrap();

            // CPU lazy
            let options_cpu = CompressOptions {
                backend: Backend::Cpu,
                threads: 1,
                ..Default::default()
            };
            let compressed_cpu =
                pipeline::compress_with_options(data, Pipeline::Lzf, &options_cpu).unwrap();

            let ratio_gpu = compressed_gpu.len() as f64 / data.len() as f64 * 100.0;
            let ratio_cpu = compressed_cpu.len() as f64 / data.len() as f64 * 100.0;

            println!(
                "{:<25} GPU: {:.1}% ({} bytes)  CPU: {:.1}% ({} bytes)  delta: {:.1}pp",
                name,
                ratio_gpu,
                compressed_gpu.len(),
                ratio_cpu,
                compressed_cpu.len(),
                ratio_gpu - ratio_cpu,
            );
        }
    }
}

#[cfg(not(feature = "webgpu"))]
fn main() {
    eprintln!("This example requires the 'webgpu' feature");
}
