//! End-to-end pipeline throughput benchmarks and external tool comparison.
//!
//! Measures compression and decompression throughput in MB/s for each
//! pipeline, and compares against external tools (gzip, pigz, zstd)
//! when available on the system PATH.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use pz::pipeline::{self, Pipeline};

/// Load test data from the Canterbury corpus, or fall back to synthetic data.
fn get_test_data() -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Try extracted corpus files
    for name in &["alice29.txt", "cantrbry.tar"] {
        let path = manifest.join("samples").join(name);
        if path.exists() {
            if let Ok(data) = std::fs::read(&path) {
                if !data.is_empty() {
                    return data;
                }
            }
        }
    }

    // Try decompressing the tar.gz to get the tar (use pz's own gzip support)
    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                return decompressed;
            }
        }
    }

    // Fallback: synthetic repetitive text (~135KB)
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    pattern.repeat(3000)
}

/// Check if a command exists on PATH.
fn command_exists(cmd: &str) -> bool {
    Command::new(cmd)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

/// Compress data by piping through an external tool.
fn shell_compress(data: &[u8], cmd: &str, args: &[&str]) -> Option<Vec<u8>> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;

    child.stdin.take()?.write_all(data).ok()?;
    let output = child.wait_with_output().ok()?;
    if output.status.success() {
        Some(output.stdout)
    } else {
        None
    }
}

/// Decompress data by piping through an external tool.
fn shell_decompress(data: &[u8], cmd: &str, args: &[&str]) -> Option<Vec<u8>> {
    shell_compress(data, cmd, args) // same mechanism
}

fn bench_compress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("compress");
    group.throughput(Throughput::Bytes(data.len() as u64));

    // pz pipelines
    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
        group.bench_with_input(
            BenchmarkId::new("pz", format!("{:?}", pipeline)),
            &data,
            |b, data| {
                b.iter(|| pipeline::compress(data, pipeline).unwrap());
            },
        );
    }

    // External: gzip
    if command_exists("gzip") {
        group.bench_with_input(BenchmarkId::new("external", "gzip"), &data, |b, data| {
            b.iter(|| shell_compress(data, "gzip", &["-c"]).unwrap());
        });
    }

    // External: pigz
    if command_exists("pigz") {
        group.bench_with_input(BenchmarkId::new("external", "pigz"), &data, |b, data| {
            b.iter(|| shell_compress(data, "pigz", &["-c"]).unwrap());
        });
    }

    // External: zstd
    if command_exists("zstd") {
        group.bench_with_input(BenchmarkId::new("external", "zstd"), &data, |b, data| {
            b.iter(|| shell_compress(data, "zstd", &["-c", "-"]).unwrap());
        });
    }

    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("decompress");
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Pre-compress lazily so filtered-out benchmarks don't pay the cost.
    // Criterion calls all group functions during enumeration even when a
    // filter is active; eagerly compressing with BWT/SA-IS on large data
    // would block for minutes.
    for &pipeline in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
        group.bench_function(
            BenchmarkId::new("pz", format!("{:?}", pipeline)),
            |b| {
                let compressed = pipeline::compress(&data, pipeline).unwrap();
                b.iter(|| pipeline::decompress(&compressed).unwrap());
            },
        );
    }

    // External: gzip decompress
    if command_exists("gzip") {
        group.bench_function(BenchmarkId::new("external", "gzip"), |b| {
            let gz_data = shell_compress(&data, "gzip", &["-c"]).unwrap();
            b.iter(|| shell_decompress(&gz_data, "gzip", &["-dc"]).unwrap());
        });
    }

    // External: pigz decompress
    if command_exists("pigz") {
        group.bench_function(BenchmarkId::new("external", "pigz"), |b| {
            let gz_data = shell_compress(&data, "pigz", &["-c"]).unwrap();
            b.iter(|| shell_decompress(&gz_data, "pigz", &["-dc"]).unwrap());
        });
    }

    // External: zstd decompress
    if command_exists("zstd") {
        group.bench_function(BenchmarkId::new("external", "zstd"), |b| {
            let zst_data = shell_compress(&data, "zstd", &["-c", "-"]).unwrap();
            b.iter(|| shell_decompress(&zst_data, "zstd", &["-dc"]).unwrap());
        });
    }

    group.finish();
}

#[cfg(feature = "opencl")]
fn bench_compress_gpu(c: &mut Criterion) {
    use pz::opencl::OpenClEngine;
    use pz::pipeline::{Backend, CompressOptions};

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("throughput: no OpenCL device, skipping GPU benchmarks");
            return;
        }
    };

    eprintln!("throughput: GPU device: {}", engine.device_name());

    let data = get_test_data();
    let mut group = c.benchmark_group("compress_gpu");
    group.throughput(Throughput::Bytes(data.len() as u64));

    let options = CompressOptions {
        backend: Backend::OpenCl,
        opencl_engine: Some(engine),
    };

    for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
        let opts = options.clone();
        group.bench_with_input(
            BenchmarkId::new("pz_gpu", format!("{:?}", pipe)),
            &data,
            move |b, data| {
                b.iter(|| pipeline::compress_with_options(data, pipe, &opts).unwrap());
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_compress_gpu(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_compress_gpu
);
criterion_main!(benches);
