#[path = "throughput_common.rs"]
mod throughput_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pz::pipeline::{self, Pipeline};
use throughput_common::{cap, get_test_data, get_test_data_sized};

fn bench_compress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("compress_bw");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(BenchmarkId::new("pz", "Bw"), &data, |b, data| {
        b.iter(|| pipeline::compress(data, Pipeline::Bw).unwrap());
    });
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("decompress_bw");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function(BenchmarkId::new("pz", "Bw"), |b| {
        let compressed = pipeline::compress(&data, Pipeline::Bw).unwrap();
        b.iter(|| pipeline::decompress(&compressed).unwrap());
    });
    group.finish();
}

fn bench_compress_parallel(c: &mut Criterion) {
    use pz::pipeline::CompressOptions;

    let data = get_test_data();
    let mut group = c.benchmark_group("compress_parallel_bw");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(BenchmarkId::new("pz_mt", "Bw"), &data, |b, data| {
        let opts = CompressOptions {
            threads: 0,
            ..Default::default()
        };
        b.iter(|| pipeline::compress_with_options(data, Pipeline::Bw, &opts).unwrap());
    });

    group.finish();
}

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

    let data = get_test_data();
    let mut group = c.benchmark_group("compress_webgpu_bw");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    let options = CompressOptions {
        backend: Backend::WebGpu,
        threads: 1,
        block_size: 0,
        parse_strategy: pz::pipeline::ParseStrategy::Auto,
        webgpu_engine: Some(engine),
        ..Default::default()
    };

    group.bench_with_input(
        BenchmarkId::new("pz_webgpu", "Bw"),
        &data,
        move |b, data| {
            b.iter(|| pipeline::compress_with_options(data, Pipeline::Bw, &options).unwrap());
        },
    );

    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_compress_webgpu(_c: &mut Criterion) {}

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

    let mut group = c.benchmark_group("compress_webgpu_large_bw");
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));

        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine.clone()),
            ..CompressOptions::default()
        };

        group.bench_with_input(
            BenchmarkId::new("Bw_webgpu", size),
            &data,
            move |b, data| {
                b.iter(|| pipeline::compress_with_options(data, Pipeline::Bw, &options).unwrap());
            },
        );
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
    bench_compress_webgpu,
    bench_compress_webgpu_large
);
criterion_main!(benches);
