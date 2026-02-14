//! WebGPU LZ77 diagnostic: match quality and speed vs CPU.
//!
//! ```bash
//! cargo run --example webgpu_diag --release --features webgpu
//! ```

#[cfg(feature = "webgpu")]
use std::time::Instant;

#[cfg(feature = "webgpu")]
const ITERS: usize = 5;

#[cfg(feature = "webgpu")]
fn make_text_data(size: usize) -> Vec<u8> {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = remaining.min(pattern.len());
        data.extend_from_slice(&pattern[..chunk]);
    }
    data
}

#[cfg(feature = "webgpu")]
fn make_binary_data(size: usize) -> Vec<u8> {
    // Pseudo-random with some repeating structure
    let mut data = vec![0u8; size];
    let mut state: u32 = 0xDEADBEEF;
    for i in 0..size {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        // Mix in some repeats: every 1024 bytes, copy from 512 bytes ago
        if i >= 512 && i % 1024 < 128 {
            data[i] = data[i - 512];
        } else {
            data[i] = (state & 0xFF) as u8;
        }
    }
    data
}

#[cfg(feature = "webgpu")]
fn analyze_matches(matches: &[pz::lz77::Match], input_len: usize) -> (usize, usize, usize, f64) {
    let total = matches.len();
    let num_matches = matches.iter().filter(|m| m.length > 0).count();
    let matched_bytes: usize = matches.iter().map(|m| m.length as usize).sum();
    let ratio = matched_bytes as f64 / input_len as f64;
    (total, num_matches, matched_bytes, ratio)
}

#[cfg(feature = "webgpu")]
fn median_us<F: FnMut()>(mut f: F) -> f64 {
    // Warmup
    f();

    let mut times = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[ITERS / 2]
}

#[cfg(feature = "webgpu")]
fn run() {
    use pz::lz77;
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("No WebGPU device: {:?}", e);
            return;
        }
    };

    println!("WebGPU LZ77 Diagnostic: Match Quality & Speed");
    println!("==============================================\n");

    let sizes = [64 * 1024, 256 * 1024, 512 * 1024, 1024 * 1024];

    for &(label, make_data) in &[
        ("text (repeating)", make_text_data as fn(usize) -> Vec<u8>),
        (
            "binary (semi-random)",
            make_binary_data as fn(usize) -> Vec<u8>,
        ),
    ] {
        println!("\n--- {label} ---");
        println!(
            "{:<10} {:>8} {:>8} {:>8} {:>10} | {:>8} {:>8} {:>8} {:>10} | {:>8} {:>8}",
            "Size",
            "CPU seq",
            "CPU mtch",
            "CPU byts",
            "CPU ratio",
            "GPU seq",
            "GPU mtch",
            "GPU byts",
            "GPU ratio",
            "CPU us",
            "GPU us"
        );
        println!("{}", "-".repeat(130));

        for size in sizes {
            let data = make_data(size);

            // CPU lazy matches
            let cpu_matches = lz77::compress_lazy_to_matches(&data).unwrap();
            let (cpu_total, cpu_nmatch, cpu_bytes, cpu_ratio) =
                analyze_matches(&cpu_matches, data.len());

            // GPU lazy matches
            let gpu_matches = engine.find_matches(&data).unwrap();
            let (gpu_total, gpu_nmatch, gpu_bytes, gpu_ratio) =
                analyze_matches(&gpu_matches, data.len());

            // Speed benchmarks
            let cpu_us = {
                let data_ref = &data;
                median_us(|| {
                    let _ = lz77::compress_lazy_to_matches(data_ref).unwrap();
                })
            };

            let gpu_us = {
                let data_ref = &data;
                median_us(|| {
                    let _ = engine.find_matches(data_ref).unwrap();
                })
            };

            let size_label = if size >= 1024 * 1024 {
                format!("{}MB", size / (1024 * 1024))
            } else {
                format!("{}KB", size / 1024)
            };

            println!(
                "{:<10} {:>8} {:>8} {:>8} {:>9.4} | {:>8} {:>8} {:>8} {:>9.4} | {:>7.0} {:>7.0}",
                size_label,
                cpu_total,
                cpu_nmatch,
                cpu_bytes,
                cpu_ratio,
                gpu_total,
                gpu_nmatch,
                gpu_bytes,
                gpu_ratio,
                cpu_us,
                gpu_us,
            );
        }
    }

    // Also test with real file data if available
    let sample_path = "samples/canterbury/alice29.txt";
    if let Ok(data) = std::fs::read(sample_path) {
        println!("\n--- alice29.txt ({} bytes) ---", data.len());

        let cpu_matches = lz77::compress_lazy_to_matches(&data).unwrap();
        let (cpu_total, cpu_nmatch, cpu_bytes, cpu_ratio) =
            analyze_matches(&cpu_matches, data.len());

        let gpu_matches = engine.find_matches(&data).unwrap();
        let (gpu_total, gpu_nmatch, gpu_bytes, gpu_ratio) =
            analyze_matches(&gpu_matches, data.len());

        let cpu_us = {
            let data_ref = &data;
            median_us(|| {
                let _ = lz77::compress_lazy_to_matches(data_ref).unwrap();
            })
        };
        let gpu_us = {
            let data_ref = &data;
            median_us(|| {
                let _ = engine.find_matches(data_ref).unwrap();
            })
        };

        println!(
            "CPU: {} seqs, {} matches, {} bytes matched, ratio {:.4}, {:.0} us",
            cpu_total, cpu_nmatch, cpu_bytes, cpu_ratio, cpu_us
        );
        println!(
            "GPU: {} seqs, {} matches, {} bytes matched, ratio {:.4}, {:.0} us",
            gpu_total, gpu_nmatch, gpu_bytes, gpu_ratio, gpu_us
        );

        // Verify round-trip
        let cpu_compressed = lz77::compress_lazy(&data).unwrap();
        let gpu_compressed = {
            let mut output = Vec::with_capacity(gpu_matches.len() * lz77::Match::SERIALIZED_SIZE);
            for m in &gpu_matches {
                output.extend_from_slice(&m.to_bytes());
            }
            output
        };
        println!("CPU compressed: {} bytes", cpu_compressed.len());
        println!("GPU compressed: {} bytes", gpu_compressed.len());

        // Verify GPU round-trip
        let gpu_decoded = lz77::decompress(&gpu_compressed).unwrap();
        assert_eq!(gpu_decoded, data, "GPU round-trip FAILED!");
        println!("GPU round-trip: OK");
    }

    // Profiling data
    println!("\n--- GPU Profiling (256KB text) ---");
    let engine = WebGpuEngine::with_profiling(true).unwrap();
    let data = make_text_data(256 * 1024);
    // warmup
    let _ = engine.find_matches(&data);
    // profiled run
    let _ = engine.find_matches(&data);
    if let Some(results) = engine.profiler_end_frame() {
        let path = std::path::Path::new("/tmp/pz_webgpu_lz77_trace.json");
        WebGpuEngine::profiler_write_trace(path, &results).ok();
        println!("Trace written to /tmp/pz_webgpu_lz77_trace.json");
        // Print timing summary
        for r in &results {
            if let Some(ref time) = r.time {
                let dur_us = (time.end - time.start) * 1_000_000.0;
                println!("  {}: {:.1} us", r.label, dur_us);
            } else {
                println!("  {}: (no timestamp)", r.label);
            }
        }
    } else {
        println!("  (no profiler results — timestamps may not be supported)");
    }
}

fn main() {
    #[cfg(feature = "webgpu")]
    run();

    #[cfg(not(feature = "webgpu"))]
    eprintln!("Requires --features webgpu");
}
