#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

const SIZES_ALL: &[usize] = &[8192, 65536, 4_194_304];

fn bench_lz77(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz77");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("compress_lazy", size), &data, |b, data| {
            b.iter(|| pz::lz77::compress_lazy(data).unwrap());
        });

        let compressed = pz::lz77::compress_lazy(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &compressed,
            |b, compressed| {
                b.iter(|| pz::lz77::decompress(compressed).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_lz77_webgpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lz77_webgpu");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("compress_webgpu_lazy", size),
            &data,
            move |b, data| {
                b.iter(|| eng.lz77_compress(data).unwrap());
            },
        );

        let eng2 = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("compress_webgpu_greedy", size),
            &data,
            move |b, data| {
                b.iter(|| {
                    let matches = eng2.find_matches_greedy(data).unwrap();
                    let mut out =
                        Vec::with_capacity(matches.len() * pz::lz77::Match::SERIALIZED_SIZE);
                    for m in &matches {
                        out.extend_from_slice(&m.to_bytes());
                    }
                    out
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compress_cpu_lazy", size),
            &data,
            |b, data| {
                b.iter(|| pz::lz77::compress_lazy(data).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_lz77_webgpu(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn bench_lz77_webgpu_batched(c: &mut Criterion) {
    use pz::pipeline::{Backend, CompressOptions, Pipeline};
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no WebGPU device, skipping batched LZ77 benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("lz77_webgpu_batched");
    cap(&mut group);
    for &size in &[1_048_576, 4_194_304] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("gpu_batched_deflate", size),
            &data,
            move |b, data| {
                let opts = CompressOptions {
                    backend: Backend::WebGpu,
                    webgpu_engine: Some(eng.clone()),
                    threads: 4,
                    ..Default::default()
                };
                b.iter(|| {
                    pz::pipeline::compress_with_options(data, Pipeline::Deflate, &opts).unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cpu_parallel_deflate", size),
            &data,
            |b, data| {
                let opts = CompressOptions {
                    threads: 4,
                    ..Default::default()
                };
                b.iter(|| {
                    pz::pipeline::compress_with_options(data, Pipeline::Deflate, &opts).unwrap()
                });
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_lz77_webgpu_batched(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_lz77,
    bench_lz77_webgpu,
    bench_lz77_webgpu_batched
);
criterion_main!(benches);
