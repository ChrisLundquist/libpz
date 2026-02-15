#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

#[cfg(feature = "webgpu")]
fn bench_deflate_webgpu_chained(c: &mut Criterion) {
    use pz::pipeline::CompressOptions;
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no WebGPU device, skipping WebGPU Deflate chained benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("deflate_webgpu_chained");
    cap(&mut group);
    for &size in &[65536, 262_144, 1_048_576, 4_194_304, 16_777_216] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("cpu_1t", size), &data, |b, data| {
            let opts = CompressOptions {
                threads: 1,
                ..Default::default()
            };
            b.iter(|| {
                pz::pipeline::compress_with_options(data, pz::pipeline::Pipeline::Deflate, &opts)
                    .unwrap()
            });
        });

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("webgpu_modular", size),
            &data,
            move |b, data| {
                let opts = CompressOptions {
                    backend: pz::pipeline::Backend::WebGpu,
                    threads: 1,
                    webgpu_engine: Some(eng.clone()),
                    ..Default::default()
                };
                b.iter(|| {
                    pz::pipeline::compress_with_options(
                        data,
                        pz::pipeline::Pipeline::Deflate,
                        &opts,
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_deflate_webgpu_chained(_c: &mut Criterion) {}

criterion_group!(benches, bench_deflate_webgpu_chained);
criterion_main!(benches);
