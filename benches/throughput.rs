//! End-to-end pipeline throughput benchmarks.
//!
//! Measures compression and decompression throughput in MB/s for each
//! pipeline (CPU, GPU, multi-threaded).
//!
//! Size tiers: default corpus (~135KB), 4MB, 16MB.
//!
//! All groups enforce warm_up_time(2s) + measurement_time(5s) + sample_size(10)
//! to keep total runtime bounded.
//!
//! Note: External tool comparison (gzip, pigz, zstd) is done via `scripts/bench.sh`
//! instead, since subprocess-based benchmarks are unreliable in Criterion.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::Path;
use std::time::Duration;

use pz::pipeline::{self, Pipeline};

/// Apply standard timeout caps to a benchmark group.
fn cap(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);
}

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

/// Load test data, repeated to fill the requested size.
fn get_test_data_sized(size: usize) -> Vec<u8> {
    let base = get_test_data();
    if base.len() >= size {
        return base[..size].to_vec();
    }
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = remaining.min(base.len());
        data.extend_from_slice(&base[..chunk]);
    }
    data
}

fn bench_compress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("compress");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    // pz pipelines
    for &pipeline in &[
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Bbw,
        Pipeline::Lzr,
        Pipeline::Lzf,
        Pipeline::Lzfi,
        Pipeline::Bwi,
    ] {
        group.bench_with_input(
            BenchmarkId::new("pz", format!("{:?}", pipeline)),
            &data,
            |b, data| {
                b.iter(|| pipeline::compress(data, pipeline).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("decompress");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Pre-compress lazily so filtered-out benchmarks don't pay the cost.
    // Criterion calls all group functions during enumeration even when a
    // filter is active; eagerly compressing with BWT/SA-IS on large data
    // would block for minutes.
    for &pipeline in &[
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Bbw,
        Pipeline::Lzr,
        Pipeline::Lzf,
        Pipeline::Lzfi,
        Pipeline::Bwi,
    ] {
        group.bench_function(BenchmarkId::new("pz", format!("{:?}", pipeline)), |b| {
            let compressed = pipeline::compress(&data, pipeline).unwrap();
            b.iter(|| pipeline::decompress(&compressed).unwrap());
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
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    let options = CompressOptions {
        backend: Backend::OpenCl,
        threads: 1,
        block_size: 0,
        parse_strategy: pz::pipeline::ParseStrategy::Auto,
        opencl_engine: Some(engine),
        #[cfg(feature = "webgpu")]
        webgpu_engine: None,
        ..Default::default()
    };

    for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
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

fn bench_compress_parallel(c: &mut Criterion) {
    use pz::pipeline::CompressOptions;

    let data = get_test_data();
    let mut group = c.benchmark_group("compress_parallel");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    // Multi-threaded (auto thread count) — compare against single-threaded
    // results in the `compress` group.
    for &pipe in &[
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Bbw,
        Pipeline::Lzr,
        Pipeline::Lzf,
    ] {
        group.bench_with_input(
            BenchmarkId::new("pz_mt", format!("{:?}", pipe)),
            &data,
            |b, data| {
                let opts = CompressOptions {
                    threads: 0, // auto
                    ..Default::default()
                };
                b.iter(|| pipeline::compress_with_options(data, pipe, &opts).unwrap());
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_compress_gpu(_c: &mut Criterion) {}

/// End-to-end pipeline throughput at 4MB and 16MB (CPU).
fn bench_compress_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_large");
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));

        for &pipe in &[Pipeline::Deflate, Pipeline::Lzr, Pipeline::Lzf] {
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", pipe), size),
                &data,
                |b, data| {
                    b.iter(|| pipeline::compress(data, pipe).unwrap());
                },
            );
        }
    }
    group.finish();
}

/// End-to-end pipeline decompression at 4MB and 16MB (CPU).
fn bench_decompress_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_large");
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));

        for &pipe in &[Pipeline::Deflate, Pipeline::Lzr, Pipeline::Lzf] {
            group.bench_function(BenchmarkId::new(format!("{:?}", pipe), size), |b| {
                let compressed = pipeline::compress(&data, pipe).unwrap();
                b.iter(|| pipeline::decompress(&compressed).unwrap());
            });
        }
    }
    group.finish();
}

/// End-to-end GPU pipeline throughput at 4MB and 16MB.
#[cfg(feature = "opencl")]
fn bench_compress_gpu_large(c: &mut Criterion) {
    use pz::opencl::OpenClEngine;
    use pz::pipeline::{Backend, CompressOptions};

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("throughput: no OpenCL device, skipping large GPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("compress_gpu_large");
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));

        let options = CompressOptions {
            backend: Backend::OpenCl,
            opencl_engine: Some(engine.clone()),
            ..CompressOptions::default()
        };

        for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            let opts = options.clone();
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_gpu", pipe), size),
                &data,
                move |b, data| {
                    b.iter(|| pipeline::compress_with_options(data, pipe, &opts).unwrap());
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_compress_gpu_large(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn bench_compress_webgpu(c: &mut Criterion) {
    use pz::pipeline::{Backend, CompressOptions};
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("throughput: no WebGPU device, skipping WebGPU benchmarks");
            return;
        }
    };

    eprintln!("throughput: WebGPU device: {}", engine.device_name());

    let data = get_test_data();
    let mut group = c.benchmark_group("compress_webgpu");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    let options = CompressOptions {
        backend: Backend::WebGpu,
        threads: 1,
        block_size: 0,
        parse_strategy: pz::pipeline::ParseStrategy::Auto,
        #[cfg(feature = "opencl")]
        opencl_engine: None,
        webgpu_engine: Some(engine),
        ..Default::default()
    };

    for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
        let opts = options.clone();
        group.bench_with_input(
            BenchmarkId::new("pz_webgpu", format!("{:?}", pipe)),
            &data,
            move |b, data| {
                b.iter(|| pipeline::compress_with_options(data, pipe, &opts).unwrap());
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_compress_webgpu(_c: &mut Criterion) {}

/// End-to-end WebGPU pipeline throughput at 4MB and 16MB.
#[cfg(feature = "webgpu")]
fn bench_compress_webgpu_large(c: &mut Criterion) {
    use pz::pipeline::{Backend, CompressOptions};
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("throughput: no WebGPU device, skipping large WebGPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("compress_webgpu_large");
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));

        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine.clone()),
            ..CompressOptions::default()
        };

        for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lzf] {
            let opts = options.clone();
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}_webgpu", pipe), size),
                &data,
                move |b, data| {
                    b.iter(|| pipeline::compress_with_options(data, pipe, &opts).unwrap());
                },
            );
        }
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_compress_webgpu_large(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_compress_parallel,
    bench_compress_large,
    bench_decompress_large,
    bench_compress_gpu,
    bench_compress_gpu_large,
    bench_compress_webgpu,
    bench_compress_webgpu_large
);
criterion_main!(benches);
