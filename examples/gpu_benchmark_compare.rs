use pz::*;
use std::fs;
use std::sync::Arc;
use std::time::Instant;

fn main() {
    println!("=== GPU Backend Comparison Benchmark ===\n");

    // Enable profiling to see what's happening
    std::env::set_var("PZ_GPU_PROFILE", "1");

    // Test sizes - larger inputs to benefit from GPU batching
    let sizes = vec![(4 * 1024 * 1024, "4MB"), (16 * 1024 * 1024, "16MB")];

    // Initialize GPU engines
    #[cfg(feature = "opencl")]
    let opencl_engine = match opencl::OpenClEngine::new() {
        Ok(e) => {
            println!("✓ OpenCL initialized: {}", e.device_name());
            Some(Arc::new(e))
        }
        Err(e) => {
            println!("✗ OpenCL not available: {}", e);
            None
        }
    };
    #[cfg(not(feature = "opencl"))]
    let _opencl_engine: Option<Arc<()>> = None;

    #[cfg(feature = "webgpu")]
    let webgpu_engine = match webgpu::WebGpuEngine::new() {
        Ok(e) => {
            println!("✓ WebGPU initialized: {}", e.device_name());
            Some(Arc::new(e))
        }
        Err(e) => {
            println!("✗ WebGPU not available: {}", e);
            None
        }
    };
    #[cfg(not(feature = "webgpu"))]
    let _webgpu_engine: Option<Arc<()>> = None;

    println!();

    // Test pipelines to compare
    let pipelines = vec![
        (pipeline::Pipeline::Deflate, "Deflate (LZ77+Huffman)"),
        (pipeline::Pipeline::Lzr, "Lzr (LZ77+rANS)"),
        (pipeline::Pipeline::Lzf, "Lzf (LZ77+FSE)"),
    ];

    println!(
        "{:>10} | {:25} | {:>14} | {:>14} | {:>14} | {:>14}",
        "Size", "Pipeline", "CPU (ST)", "CPU (MT)", "OpenCL", "WebGPU"
    );
    println!("{}", "-".repeat(115));

    // Run benchmarks for each size and pipeline
    for (size, label) in &sizes {
        for (pipeline_type, pipeline_name) in &pipelines {
            // Generate test data (moderately compressible)
            let data: Vec<u8> = (0..*size).map(|i| (i % 251) as u8).collect();

            // CPU (single-threaded)
            let cpu_st_throughput = {
                let start = Instant::now();
                let iterations = 3;
                for _ in 0..iterations {
                    let _ = pipeline::compress(&data, *pipeline_type).unwrap();
                }
                let elapsed = start.elapsed();
                (*size as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0)
            };

            // CPU (multi-threaded)
            let cpu_mt_throughput = {
                let opts = pipeline::CompressOptions {
                    threads: 0,
                    block_size: 256 * 1024,
                    ..Default::default()
                };

                let start = Instant::now();
                let iterations = 3;
                for _ in 0..iterations {
                    let _ = pipeline::compress_with_options(&data, *pipeline_type, &opts).unwrap();
                }
                let elapsed = start.elapsed();
                (*size as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0)
            };

            // OpenCL (multi-threaded for GPU batching)
            #[cfg(feature = "opencl")]
            let opencl_throughput = if let Some(ref engine) = opencl_engine {
                let mut opts = pipeline::CompressOptions::default();
                opts.backend = pipeline::Backend::OpenCl;
                opts.opencl_engine = Some(engine.clone());
                opts.threads = 0; // Auto-detect threads for GPU batching
                opts.block_size = 256 * 1024; // 256KB blocks for GPU batching

                let start = Instant::now();
                let iterations = 3;
                let mut success = true;
                for _ in 0..iterations {
                    if pipeline::compress_with_options(&data, *pipeline_type, &opts).is_err() {
                        success = false;
                        break;
                    }
                }
                if success {
                    let elapsed = start.elapsed();
                    Some(
                        (*size as f64 * iterations as f64)
                            / elapsed.as_secs_f64()
                            / (1024.0 * 1024.0),
                    )
                } else {
                    None
                }
            } else {
                None
            };
            #[cfg(not(feature = "opencl"))]
            let opencl_throughput: Option<f64> = None;

            // WebGPU (multi-threaded for GPU batching)
            #[cfg(feature = "webgpu")]
            let webgpu_throughput = if let Some(ref engine) = webgpu_engine {
                let mut opts = pipeline::CompressOptions::default();
                opts.backend = pipeline::Backend::WebGpu;
                opts.webgpu_engine = Some(engine.clone());
                opts.threads = 0; // Auto-detect threads for GPU batching
                opts.block_size = 2 * 1024 * 1024; // 2MB blocks to amortize GPU overhead

                let start = Instant::now();
                let iterations = 3;
                let mut success = true;
                for _ in 0..iterations {
                    if pipeline::compress_with_options(&data, *pipeline_type, &opts).is_err() {
                        success = false;
                        break;
                    }
                }
                if success {
                    let elapsed = start.elapsed();
                    Some(
                        (*size as f64 * iterations as f64)
                            / elapsed.as_secs_f64()
                            / (1024.0 * 1024.0),
                    )
                } else {
                    None
                }
            } else {
                None
            };
            #[cfg(not(feature = "webgpu"))]
            let webgpu_throughput: Option<f64> = None;

            let opencl_str =
                opencl_throughput.map_or("N/A".to_string(), |t| format!("{:.1} MB/s", t));
            let webgpu_str =
                webgpu_throughput.map_or("N/A".to_string(), |t| format!("{:.1} MB/s", t));

            println!(
                "{:>10} | {:25} | {:>12.1} MB/s | {:>12.1} MB/s | {:>14} | {:>14}",
                label, pipeline_name, cpu_st_throughput, cpu_mt_throughput, opencl_str, webgpu_str
            );
        }
        println!("{}", "-".repeat(115));
    }

    // Test with real file if available
    if let Ok(file_data) = fs::read("data/enwik8") {
        let size = file_data.len();
        println!(
            "\n=== Real file: enwik8 ({:.1} MB) ===",
            size as f64 / (1024.0 * 1024.0)
        );

        for (pipeline_type, pipeline_name) in &pipelines {
            // CPU
            let cpu_time = {
                let start = Instant::now();
                let _ = pipeline::compress(&file_data, *pipeline_type).unwrap();
                start.elapsed().as_secs_f64()
            };
            let cpu_throughput = size as f64 / cpu_time / (1024.0 * 1024.0);

            // OpenCL (with GPU batching)
            #[cfg(feature = "opencl")]
            let (opencl_throughput, opencl_speedup) = if let Some(ref engine) = opencl_engine {
                let mut opts = pipeline::CompressOptions::default();
                opts.backend = pipeline::Backend::OpenCl;
                opts.opencl_engine = Some(engine.clone());
                opts.threads = 0; // Auto-detect for batching
                opts.block_size = 256 * 1024;

                let start = Instant::now();
                let _ = pipeline::compress_with_options(&file_data, *pipeline_type, &opts).unwrap();
                let opencl_time = start.elapsed().as_secs_f64();
                let throughput = size as f64 / opencl_time / (1024.0 * 1024.0);
                (Some(throughput), Some(cpu_time / opencl_time))
            } else {
                (None, None)
            };
            #[cfg(not(feature = "opencl"))]
            let (opencl_throughput, opencl_speedup): (Option<f64>, Option<f64>) = (None, None);

            // WebGPU (with GPU batching)
            #[cfg(feature = "webgpu")]
            let (webgpu_throughput, webgpu_speedup) = if let Some(ref engine) = webgpu_engine {
                let mut opts = pipeline::CompressOptions::default();
                opts.backend = pipeline::Backend::WebGpu;
                opts.webgpu_engine = Some(engine.clone());
                opts.threads = 0; // Auto-detect for batching
                opts.block_size = 2 * 1024 * 1024; // 2MB blocks

                let start = Instant::now();
                let _ = pipeline::compress_with_options(&file_data, *pipeline_type, &opts).unwrap();
                let webgpu_time = start.elapsed().as_secs_f64();
                let throughput = size as f64 / webgpu_time / (1024.0 * 1024.0);
                (Some(throughput), Some(cpu_time / webgpu_time))
            } else {
                (None, None)
            };
            #[cfg(not(feature = "webgpu"))]
            let (webgpu_throughput, webgpu_speedup): (Option<f64>, Option<f64>) = (None, None);

            println!("\n{}:", pipeline_name);
            println!("  CPU:    {:7.2} MB/s", cpu_throughput);
            if let Some(throughput) = opencl_throughput {
                println!(
                    "  OpenCL: {:7.2} MB/s  ({:.2}x)",
                    throughput,
                    opencl_speedup.unwrap()
                );
            }
            if let Some(throughput) = webgpu_throughput {
                println!(
                    "  WebGPU: {:7.2} MB/s  ({:.2}x)",
                    throughput,
                    webgpu_speedup.unwrap()
                );
            }
        }
    }
}
