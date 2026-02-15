use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use pz::pipeline::{self, Pipeline};
use std::path::Path;
use std::time::Duration;

pub struct ThroughputBenchSpec {
    pub id: &'static str,
    pub pipeline: Pipeline,
    pub parallel: bool,
    pub large: bool,
    pub decompress_large: bool,
    pub webgpu: bool,
    pub webgpu_large: bool,
}

pub fn run_throughput_benches(c: &mut Criterion, spec: &ThroughputBenchSpec) {
    bench_compress(c, spec);
    bench_decompress(c, spec);

    if spec.parallel {
        bench_compress_parallel(c, spec);
    }

    if spec.large {
        bench_compress_large(c, spec);
    }

    if spec.decompress_large {
        bench_decompress_large(c, spec);
    }

    if spec.webgpu {
        bench_compress_webgpu(c, spec);
    }

    if spec.webgpu_large {
        bench_compress_webgpu_large(c, spec);
    }
}

fn cap(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);
}

fn get_test_data() -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

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

    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                return decompressed;
            }
        }
    }

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    pattern.repeat(3000)
}

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

fn bench_compress(c: &mut Criterion, spec: &ThroughputBenchSpec) {
    let data = get_test_data();
    let mut group = c.benchmark_group(format!("compress_{}", spec.id));
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("pz", format!("{:?}", spec.pipeline)),
        &data,
        |b, data| {
            b.iter(|| pipeline::compress(data, spec.pipeline).unwrap());
        },
    );

    group.finish();
}

fn bench_decompress(c: &mut Criterion, spec: &ThroughputBenchSpec) {
    let data = get_test_data();
    let mut group = c.benchmark_group(format!("decompress_{}", spec.id));
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function(
        BenchmarkId::new("pz", format!("{:?}", spec.pipeline)),
        |b| {
            let compressed = pipeline::compress(&data, spec.pipeline).unwrap();
            b.iter(|| pipeline::decompress(&compressed).unwrap());
        },
    );

    group.finish();
}

fn bench_compress_parallel(c: &mut Criterion, spec: &ThroughputBenchSpec) {
    use pz::pipeline::CompressOptions;

    let data = get_test_data();
    let mut group = c.benchmark_group(format!("compress_parallel_{}", spec.id));
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("pz_mt", format!("{:?}", spec.pipeline)),
        &data,
        |b, data| {
            let opts = CompressOptions {
                threads: 0,
                ..Default::default()
            };
            b.iter(|| pipeline::compress_with_options(data, spec.pipeline, &opts).unwrap());
        },
    );

    group.finish();
}

fn bench_compress_large(c: &mut Criterion, spec: &ThroughputBenchSpec) {
    let mut group = c.benchmark_group(format!("compress_large_{}", spec.id));
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("{:?}", spec.pipeline), size),
            &data,
            |b, data| {
                b.iter(|| pipeline::compress(data, spec.pipeline).unwrap());
            },
        );
    }

    group.finish();
}

fn bench_decompress_large(c: &mut Criterion, spec: &ThroughputBenchSpec) {
    let mut group = c.benchmark_group(format!("decompress_large_{}", spec.id));
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(
            BenchmarkId::new(format!("{:?}", spec.pipeline), size),
            |b| {
                let compressed = pipeline::compress(&data, spec.pipeline).unwrap();
                b.iter(|| pipeline::decompress(&compressed).unwrap());
            },
        );
    }

    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_compress_webgpu(c: &mut Criterion, spec: &ThroughputBenchSpec) {
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
    let mut group = c.benchmark_group(format!("compress_webgpu_{}", spec.id));
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
        BenchmarkId::new("pz_webgpu", format!("{:?}", spec.pipeline)),
        &data,
        move |b, data| {
            b.iter(|| pipeline::compress_with_options(data, spec.pipeline, &options).unwrap());
        },
    );

    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_compress_webgpu(_c: &mut Criterion, _spec: &ThroughputBenchSpec) {}

#[cfg(feature = "webgpu")]
fn bench_compress_webgpu_large(c: &mut Criterion, spec: &ThroughputBenchSpec) {
    use pz::pipeline::{Backend, CompressOptions};
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("throughput: no WebGPU device, skipping large WebGPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group(format!("compress_webgpu_large_{}", spec.id));
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
            BenchmarkId::new(format!("{:?}_webgpu", spec.pipeline), size),
            &data,
            move |b, data| {
                b.iter(|| pipeline::compress_with_options(data, spec.pipeline, &options).unwrap());
            },
        );
    }

    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_compress_webgpu_large(_c: &mut Criterion, _spec: &ThroughputBenchSpec) {}
