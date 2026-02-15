#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data, SIZES_ALL, SIZES_LARGE, SIZES_SMALL};

fn bench_bwt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bwt");
    cap(&mut group);
    for &size in SIZES_SMALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode(data).unwrap());
        });

        let encoded = pz::bwt::encode(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode(&enc.data, enc.primary_index).unwrap());
        });
    }
    group.finish();
}

fn bench_bwt_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("bwt_large");
    cap(&mut group);
    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode(data).unwrap());
        });

        let encoded = pz::bwt::encode(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode(&enc.data, enc.primary_index).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_bwt_webgpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no WebGPU device, skipping WebGPU BWT benchmarks");
            return;
        }
    };

    eprintln!("stages: WebGPU device: {}", engine.device_name());

    let mut group = c.benchmark_group("bwt_webgpu");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("encode_webgpu", size),
            &data,
            move |b, data| {
                b.iter(|| eng.bwt_encode(data).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_bwt_webgpu(_c: &mut Criterion) {}

criterion_group!(benches, bench_bwt, bench_bwt_large, bench_bwt_webgpu);
criterion_main!(benches);
