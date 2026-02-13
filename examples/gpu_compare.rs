//! Side-by-side OpenCL vs WebGPU GPU backend comparison.
//!
//! Runs the same operations on both backends and reports throughput.
//!
//! ```bash
//! cargo run --example gpu_compare --release --features opencl,webgpu
//! ```

#[cfg(all(feature = "opencl", feature = "webgpu"))]
use std::time::Instant;

/// Number of timed iterations per benchmark.
#[cfg(all(feature = "opencl", feature = "webgpu"))]
const ITERS: usize = 5;

/// Input sizes to test (bytes).
#[cfg(all(feature = "opencl", feature = "webgpu"))]
const SIZES: &[usize] = &[8_192, 65_536, 262_144, 1_048_576];

#[cfg(all(feature = "opencl", feature = "webgpu"))]
fn make_test_data(size: usize) -> Vec<u8> {
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = remaining.min(pattern.len());
        data.extend_from_slice(&pattern[..chunk]);
    }
    data
}

#[cfg(all(feature = "opencl", feature = "webgpu"))]
struct BenchResult {
    size: usize,
    median_us: f64,
}

#[cfg(all(feature = "opencl", feature = "webgpu"))]
impl BenchResult {
    fn throughput_mbs(&self) -> f64 {
        if self.median_us == 0.0 {
            return 0.0;
        }
        (self.size as f64) / self.median_us // bytes/us = MB/s
    }
}

#[cfg(all(feature = "opencl", feature = "webgpu"))]
fn bench<F: FnMut()>(_label: &str, size: usize, mut f: F) -> BenchResult {
    // Warmup
    f();

    let mut times_us = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        f();
        times_us.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }
    times_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_us = times_us[ITERS / 2];

    BenchResult { size, median_us }
}

#[cfg(all(feature = "opencl", feature = "webgpu"))]
fn print_comparison(group: &str, results: &[(BenchResult, BenchResult)]) {
    println!("\n=== {group} ===");
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12} {:>8}",
        "Size", "OpenCL (us)", "OpenCL MB/s", "WebGPU (us)", "WebGPU MB/s", "Winner"
    );
    println!("{}", "-".repeat(80));
    for (cl, wg) in results {
        let winner = if wg.median_us < cl.median_us {
            "WebGPU"
        } else {
            "OpenCL"
        };
        let ratio = if wg.median_us > 0.0 {
            cl.median_us / wg.median_us
        } else {
            0.0
        };
        println!(
            "{:<12} {:>12.0} {:>12.1} {:>12.0} {:>12.1} {:>8} ({:.2}x)",
            format_size(cl.size),
            cl.median_us,
            cl.throughput_mbs(),
            wg.median_us,
            wg.throughput_mbs(),
            winner,
            ratio,
        );
    }
}

#[cfg(all(feature = "opencl", feature = "webgpu"))]
fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{}MB", bytes / 1_048_576)
    } else if bytes >= 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{}B", bytes)
    }
}

fn main() {
    #[cfg(not(all(feature = "opencl", feature = "webgpu")))]
    {
        eprintln!("This example requires both opencl and webgpu features:");
        eprintln!("  cargo run --example gpu_compare --release --features opencl,webgpu");
        std::process::exit(1);
    }

    #[cfg(all(feature = "opencl", feature = "webgpu"))]
    run_comparison();
}

#[cfg(all(feature = "opencl", feature = "webgpu"))]
fn run_comparison() {
    use pz::opencl::{KernelVariant, OpenClEngine};
    use pz::webgpu::WebGpuEngine;
    use std::sync::Arc;

    let cl_engine = match OpenClEngine::new() {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("OpenCL unavailable: {:?}", e);
            return;
        }
    };

    let wg_engine = match WebGpuEngine::new() {
        Ok(e) => Arc::new(e),
        Err(e) => {
            eprintln!("WebGPU unavailable: {:?}", e);
            return;
        }
    };

    println!("OpenCL device: {}", cl_engine.device_name());
    println!("WebGPU device: {}", wg_engine.device_name());
    println!("Iterations per bench: {ITERS}  (reporting median)");

    // --- LZ77 Hash ---
    {
        let mut results = Vec::new();
        for &size in SIZES {
            let data = make_test_data(size);
            let cl = cl_engine.clone();
            let cl_res = bench("opencl", size, || {
                cl.lz77_compress(&data, KernelVariant::HashTable).unwrap();
            });
            let wg = wg_engine.clone();
            let wg_res = bench("webgpu", size, || {
                // WebGPU's greedy == OpenCL's hash (both 2-pass hash-table)
                let matches = wg.find_matches_greedy(&data).unwrap();
                let mut out = Vec::with_capacity(matches.len() * pz::lz77::Match::SERIALIZED_SIZE);
                for m in &matches {
                    out.extend_from_slice(&m.to_bytes());
                }
            });
            results.push((cl_res, wg_res));
        }
        print_comparison("LZ77 Hash (greedy, apples-to-apples)", &results);
    }

    // --- LZ77: OpenCL hash vs WebGPU lazy ---
    {
        let mut results = Vec::new();
        for &size in SIZES {
            let data = make_test_data(size);
            let cl = cl_engine.clone();
            let cl_res = bench("opencl", size, || {
                cl.lz77_compress(&data, KernelVariant::HashTable).unwrap();
            });
            let wg = wg_engine.clone();
            let wg_res = bench("webgpu", size, || {
                wg.lz77_compress(&data).unwrap();
            });
            results.push((cl_res, wg_res));
        }
        print_comparison("LZ77: OpenCL hash vs WebGPU lazy (best-of-each)", &results);
    }

    // --- BWT ---
    {
        let mut results = Vec::new();
        for &size in SIZES {
            let data = make_test_data(size);
            let cl = cl_engine.clone();
            let cl_res = bench("opencl", size, || {
                cl.bwt_encode(&data).unwrap();
            });
            let wg = wg_engine.clone();
            let wg_res = bench("webgpu", size, || {
                wg.bwt_encode(&data).unwrap();
            });
            results.push((cl_res, wg_res));
        }
        print_comparison("BWT Encode", &results);
    }

    // --- Huffman (GPU scan) ---
    {
        let mut results = Vec::new();
        for &size in SIZES {
            let data = make_test_data(size);

            let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();
            let mut code_lut = [0u32; 256];
            for byte in 0..=255u8 {
                let (codeword, bits) = tree.get_code(byte);
                code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
            }

            let cl = cl_engine.clone();
            let lut1 = code_lut;
            let cl_res = bench("opencl", size, || {
                cl.huffman_encode_gpu_scan(&data, &lut1).unwrap();
            });
            let wg = wg_engine.clone();
            let lut2 = code_lut;
            let wg_res = bench("webgpu", size, || {
                wg.huffman_encode_gpu_scan(&data, &lut2).unwrap();
            });
            results.push((cl_res, wg_res));
        }
        print_comparison("Huffman Encode (GPU scan)", &results);
    }

    // --- Byte Histogram ---
    {
        let mut results = Vec::new();
        for &size in SIZES {
            let data = make_test_data(size);
            let cl = cl_engine.clone();
            let cl_res = bench("opencl", size, || {
                cl.byte_histogram(&data).unwrap();
            });
            let wg = wg_engine.clone();
            let wg_res = bench("webgpu", size, || {
                wg.byte_histogram(&data).unwrap();
            });
            results.push((cl_res, wg_res));
        }
        print_comparison("Byte Histogram", &results);
    }

    // --- Deflate Chained (LZ77 + Huffman, end-to-end) ---
    {
        let mut results = Vec::new();
        for &size in &[65_536, 262_144, 1_048_576] {
            let data = make_test_data(size);

            // Build Huffman lookup table from the data
            let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();
            let mut code_lut = [0u32; 256];
            for byte in 0..=255u8 {
                let (codeword, bits) = tree.get_code(byte);
                code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
            }

            let cl = cl_engine.clone();
            let lut1 = code_lut;
            let cl_res = bench("opencl", size, || {
                let lz_data = cl.lz77_compress(&data, KernelVariant::HashTable).unwrap();
                let _ = cl.huffman_encode_gpu_scan(&lz_data, &lut1).unwrap();
            });
            let wg = wg_engine.clone();
            let lut2 = code_lut;
            let wg_res = bench("webgpu", size, || {
                let lz_data = wg.lz77_compress(&data).unwrap();
                let _ = wg.huffman_encode_gpu_scan(&lz_data, &lut2).unwrap();
            });
            results.push((cl_res, wg_res));
        }
        print_comparison("Deflate Chained (LZ77 + Huffman, end-to-end)", &results);
    }

    println!("\nDone.");
}
