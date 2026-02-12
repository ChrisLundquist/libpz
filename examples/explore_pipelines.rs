/// Pipeline exploration: compare pz pipelines vs gzip on real data.
///
/// Measures compression ratio and throughput for each pipeline/strategy
/// combination across Canterbury corpus + large corpus files.
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use pz::pipeline::{self, Backend, CompressOptions, ParseStrategy, Pipeline};

/// Maximum time allowed for a single compress call before we disqualify it.
const COMPRESS_TIMEOUT: Duration = Duration::from_secs(10);

struct Result {
    file: String,
    file_size: usize,
    method: String,
    compressed_size: usize,
    ratio: f64,
    compress_us: u64,
    decompress_us: u64,
    compress_mbps: f64,
    roundtrip_ok: bool,
}

/// Holds optional GPU engine handles.
#[allow(dead_code)]
struct GpuEngines {
    #[cfg(feature = "opencl")]
    opencl: Option<std::sync::Arc<pz::opencl::OpenClEngine>>,
    #[cfg(feature = "webgpu")]
    webgpu: Option<std::sync::Arc<pz::webgpu::WebGpuEngine>>,
}

/// Pipe data through an external command, writing stdin from a separate thread
/// to avoid pipe deadlocks on large inputs.
fn pipe_through(data: &[u8], cmd: &str, args: &[&str]) -> Option<Vec<u8>> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    let mut stdin = child.stdin.take()?;
    let data_owned = data.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&data_owned);
    });

    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    out.status.success().then_some(out.stdout)
}

fn gzip_compress(data: &[u8], level: &str) -> Option<Vec<u8>> {
    pipe_through(data, "gzip", &[level, "-c"])
}

fn gzip_decompress(data: &[u8]) -> Option<Vec<u8>> {
    pipe_through(data, "gzip", &["-dc"])
}

fn measure_pz(
    data: &[u8],
    file_name: &str,
    pipe: Pipeline,
    strategy: ParseStrategy,
    threads: usize,
    backend: Backend,
    _engines: &GpuEngines,
) -> Option<Result> {
    let backend_tag = match backend {
        Backend::Cpu => "",
        #[cfg(feature = "opencl")]
        Backend::OpenCl => "/OpenCL",
        #[cfg(feature = "webgpu")]
        Backend::WebGpu => "/WebGPU",
    };
    let label = format!(
        "pz/{:?}/{:?}/{}t{}",
        pipe,
        strategy,
        if threads == 0 {
            "auto".into()
        } else {
            threads.to_string()
        },
        backend_tag,
    );

    let opts = CompressOptions {
        parse_strategy: strategy,
        threads,
        backend,
        #[cfg(feature = "opencl")]
        opencl_engine: _engines.opencl.clone(),
        #[cfg(feature = "webgpu")]
        webgpu_engine: _engines.webgpu.clone(),
        ..Default::default()
    };

    // Single compress with timeout check (catch panics from GPU backends)
    let t = Instant::now();
    let compress_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pipeline::compress_with_options(data, pipe, &opts)
    }));
    let compressed = match compress_result {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            eprintln!("    ERROR: {} failed: {}", label, e);
            return None;
        }
        Err(_) => {
            eprintln!("    PANIC: {} panicked, skipping", label);
            return None;
        }
    };
    let elapsed = t.elapsed();
    if elapsed > COMPRESS_TIMEOUT {
        eprintln!(
            "    TIMEOUT: {} took {:.1}s (>{:.0}s limit), disqualified",
            label,
            elapsed.as_secs_f64(),
            COMPRESS_TIMEOUT.as_secs_f64()
        );
        return None;
    }
    let compress_us = elapsed.as_micros() as u64;

    // Second run for better timing (if fast enough for another)
    let best_compress = if elapsed < COMPRESS_TIMEOUT / 3 {
        let t2 = Instant::now();
        let _ = pipeline::compress_with_options(data, pipe, &opts).unwrap();
        compress_us.min(t2.elapsed().as_micros() as u64)
    } else {
        compress_us
    };

    // Decompress
    let t = Instant::now();
    let decompressed = pipeline::decompress(&compressed).unwrap();
    let decompress_us = t.elapsed().as_micros() as u64;
    let roundtrip_ok = decompressed == data;

    let ratio = compressed.len() as f64 / data.len() as f64;
    let mbps = data.len() as f64 / (best_compress as f64 / 1_000_000.0) / (1024.0 * 1024.0);

    Some(Result {
        file: file_name.to_string(),
        file_size: data.len(),
        method: label,
        compressed_size: compressed.len(),
        ratio,
        compress_us: best_compress,
        decompress_us,
        compress_mbps: mbps,
        roundtrip_ok,
    })
}

fn measure_gzip(data: &[u8], file_name: &str, level: &str) -> Result {
    let label = format!("gzip {}", level);

    let mut best_compress = u64::MAX;
    let mut compressed = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        compressed = gzip_compress(data, level).unwrap();
        let elapsed = t.elapsed().as_micros() as u64;
        best_compress = best_compress.min(elapsed);
    }

    let mut best_decompress = u64::MAX;
    let mut roundtrip_ok = false;
    for _ in 0..3 {
        let t = Instant::now();
        let decompressed = gzip_decompress(&compressed).unwrap();
        let elapsed = t.elapsed().as_micros() as u64;
        best_decompress = best_decompress.min(elapsed);
        roundtrip_ok = decompressed == data;
    }

    let ratio = compressed.len() as f64 / data.len() as f64;
    let mbps = data.len() as f64 / (best_compress as f64 / 1_000_000.0) / (1024.0 * 1024.0);

    Result {
        file: file_name.to_string(),
        file_size: data.len(),
        method: label,
        compressed_size: compressed.len(),
        ratio,
        compress_us: best_compress,
        decompress_us: best_decompress,
        compress_mbps: mbps,
        roundtrip_ok,
    }
}

fn print_results(results: &[Result]) {
    println!(
        "{:<20} {:>8} {:<32} {:>10} {:>7} {:>10} {:>10} {:>10} {:>4}",
        "File",
        "Size",
        "Method",
        "Compressed",
        "Ratio",
        "Comp(us)",
        "Decomp(us)",
        "Comp MB/s",
        "OK"
    );
    println!("{}", "-".repeat(120));
    for r in results {
        println!(
            "{:<20} {:>8} {:<32} {:>10} {:>6.3}x {:>10} {:>10} {:>10.1} {:>4}",
            r.file,
            r.file_size,
            r.method,
            r.compressed_size,
            r.ratio,
            r.compress_us,
            r.decompress_us,
            r.compress_mbps,
            if r.roundtrip_ok { "yes" } else { "FAIL" }
        );
    }
}

/// Check if a config should be skipped for large files (too slow in prototype).
fn should_skip_for_large(pipe: Pipeline, strategy: ParseStrategy, data_len: usize) -> bool {
    if data_len < 200_000 {
        return false;
    }
    // Optimal parsing is O(n^2)-ish, skip for anything above ~200KB
    if strategy == ParseStrategy::Optimal && data_len > 200_000 {
        return true;
    }
    // BWT on >500KB is very slow in prototype
    if pipe == Pipeline::Bw && data_len > 500_000 {
        return true;
    }
    false
}

fn main() {
    // Initialize GPU engines (if feature-enabled and hardware available)
    #[cfg(feature = "opencl")]
    let has_opencl;
    #[cfg(feature = "webgpu")]
    let has_webgpu;

    let engines = GpuEngines {
        #[cfg(feature = "opencl")]
        opencl: match pz::opencl::OpenClEngine::new() {
            Ok(e) => {
                eprintln!("OpenCL engine initialized: {}", e.device_name());
                has_opencl = true;
                Some(std::sync::Arc::new(e))
            }
            Err(e) => {
                eprintln!("OpenCL not available: {}", e);
                has_opencl = false;
                None
            }
        },
        #[cfg(feature = "webgpu")]
        webgpu: match pz::webgpu::WebGpuEngine::new() {
            Ok(e) => {
                eprintln!("WebGPU engine initialized: {}", e.device_name());
                has_webgpu = true;
                Some(std::sync::Arc::new(e))
            }
            Err(e) => {
                eprintln!("WebGPU not available: {}", e);
                has_webgpu = false;
                None
            }
        },
    };

    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    // Collect test files
    let mut test_files: Vec<(String, Vec<u8>)> = Vec::new();

    // Canterbury corpus (smaller files, diverse types)
    let cantrbry = manifest.join("samples").join("cantrbry");
    if cantrbry.exists() {
        for name in &[
            "alice29.txt",
            "asyoulik.txt",
            "cp.html",
            "fields.c",
            "kennedy.xls",
            "lcet10.txt",
            "ptt5",
            "sum",
        ] {
            let path = cantrbry.join(name);
            if let Ok(data) = std::fs::read(&path) {
                test_files.push((name.to_string(), data));
            }
        }
    }

    // Large corpus — truncated to 1MB to keep exploration practical
    let large = manifest.join("samples").join("large");
    if large.exists() {
        for name in &["bible.txt", "E.coli", "world192.txt"] {
            let path = large.join(name);
            if let Ok(data) = std::fs::read(&path) {
                let truncated = data[..data.len().min(1_000_000)].to_vec();
                test_files.push((format!("{}(1MB)", name), truncated));
            }
        }
    }

    if test_files.is_empty() {
        eprintln!("No test files found! Extract samples first.");
        std::process::exit(1);
    }

    println!("=== Pipeline Exploration: pz vs gzip ===\n");

    // Pipeline/strategy combinations to test
    #[allow(unused_mut)]
    let mut pz_configs: Vec<(Pipeline, ParseStrategy, usize, Backend)> = vec![
        // Single-threaded CPU variants
        (Pipeline::Deflate, ParseStrategy::Lazy, 1, Backend::Cpu),
        (Pipeline::Deflate, ParseStrategy::Optimal, 1, Backend::Cpu),
        (Pipeline::Lza, ParseStrategy::Lazy, 1, Backend::Cpu),
        (Pipeline::Lza, ParseStrategy::Optimal, 1, Backend::Cpu),
        (Pipeline::Bw, ParseStrategy::Auto, 1, Backend::Cpu),
        // Multi-threaded CPU
        (Pipeline::Deflate, ParseStrategy::Lazy, 0, Backend::Cpu),
        (Pipeline::Lza, ParseStrategy::Lazy, 0, Backend::Cpu),
        (Pipeline::Bw, ParseStrategy::Auto, 0, Backend::Cpu),
    ];

    // OpenCL GPU variants (if available)
    #[cfg(feature = "opencl")]
    if has_opencl {
        pz_configs.extend([
            (Pipeline::Deflate, ParseStrategy::Auto, 1, Backend::OpenCl),
            (Pipeline::Lza, ParseStrategy::Auto, 1, Backend::OpenCl),
            (Pipeline::Bw, ParseStrategy::Auto, 1, Backend::OpenCl),
        ]);
    }

    // WebGPU GPU variants (if available)
    #[cfg(feature = "webgpu")]
    if has_webgpu {
        pz_configs.extend([
            (Pipeline::Deflate, ParseStrategy::Auto, 1, Backend::WebGpu),
            (Pipeline::Lza, ParseStrategy::Auto, 1, Backend::WebGpu),
            (Pipeline::Bw, ParseStrategy::Auto, 1, Backend::WebGpu),
        ]);
    }

    let gzip_levels = ["-1", "-6", "-9"];

    // Accumulate win counts for aggregate
    let mut win_counts: std::collections::HashMap<String, (u32, u32, u32)> =
        std::collections::HashMap::new();
    let mut total_files_tested: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();

    for (file_name, data) in &test_files {
        println!("\n### {} ({} bytes) ###\n", file_name, data.len());
        let mut results = Vec::new();

        // gzip baselines
        for level in &gzip_levels {
            results.push(measure_gzip(data, file_name, level));
        }

        // pz pipelines
        for &(pipe, strategy, threads, backend) in &pz_configs {
            if should_skip_for_large(pipe, strategy, data.len()) {
                continue;
            }
            let backend_tag = match backend {
                Backend::Cpu => "",
                #[cfg(feature = "opencl")]
                Backend::OpenCl => "/OpenCL",
                #[cfg(feature = "webgpu")]
                Backend::WebGpu => "/WebGPU",
            };
            eprint!(
                "  testing {:?}/{:?}/{}t{} ... ",
                pipe, strategy, threads, backend_tag
            );
            if let Some(r) = measure_pz(data, file_name, pipe, strategy, threads, backend, &engines)
            {
                eprintln!("{:.3}x @ {:.1} MB/s", r.ratio, r.compress_mbps);
                results.push(r);
            }
        }

        // Sort by compression ratio (best first)
        results.sort_by(|a, b| a.ratio.partial_cmp(&b.ratio).unwrap());
        print_results(&results);

        // Summary: which pz methods beat gzip -6?
        let gzip6_ratio = results
            .iter()
            .find(|r| r.method == "gzip -6")
            .map(|r| r.ratio)
            .unwrap_or(1.0);
        let gzip6_speed = results
            .iter()
            .find(|r| r.method == "gzip -6")
            .map(|r| r.compress_mbps)
            .unwrap_or(0.0);

        println!(
            "\n  vs gzip -6 ({:.3}x, {:.1} MB/s):",
            gzip6_ratio, gzip6_speed
        );
        for r in results.iter().filter(|r| r.method.starts_with("pz/")) {
            let ratio_icon = if r.ratio < gzip6_ratio { "+" } else { "-" };
            let speed_icon = if r.compress_mbps > gzip6_speed {
                "+"
            } else {
                "-"
            };
            println!(
                "    [ratio:{} speed:{}] {} -> {:.3}x @ {:.1} MB/s",
                ratio_icon, speed_icon, r.method, r.ratio, r.compress_mbps
            );

            // Accumulate wins
            let entry = win_counts.entry(r.method.clone()).or_default();
            let count = total_files_tested.entry(r.method.clone()).or_default();
            *count += 1;
            if r.ratio < gzip6_ratio {
                entry.0 += 1;
            }
            if r.compress_mbps > gzip6_speed {
                entry.1 += 1;
            }
            if r.ratio < gzip6_ratio && r.compress_mbps > gzip6_speed {
                entry.2 += 1;
            }
        }
    }

    // Final aggregate summary
    println!("\n\n========================================");
    println!("  AGGREGATE SUMMARY vs gzip -6");
    println!("========================================\n");

    println!(
        "{:<35} {:>14} {:>14} {:>14}",
        "Method", "Ratio wins", "Speed wins", "Both wins"
    );
    println!("{}", "-".repeat(80));
    let mut sorted: Vec<_> = win_counts.iter().collect();
    sorted.sort_by(|a, b| b.1 .2.cmp(&a.1 .2).then(b.1 .0.cmp(&a.1 .0)));
    for (method, (ratio_wins, speed_wins, both)) in &sorted {
        let n = total_files_tested[*method];
        println!(
            "{:<35} {:>5}/{:<7} {:>5}/{:<7} {:>5}/{:<7}",
            method, ratio_wins, n, speed_wins, n, both, n
        );
    }

    println!("\nKey findings:");
    println!("  - Ratio win  = compressed smaller than gzip -6");
    println!("  - Speed win  = faster compression throughput than gzip -6");
    println!("  - Both       = simultaneously better ratio AND faster");
}
