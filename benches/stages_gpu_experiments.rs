#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

const SIZES: &[usize] = &[8192, 65536, 262_144, 4_194_304];

/// GPU BWT vs CPU BWT — persistent device, no device-init noise.
#[cfg(feature = "webgpu")]
fn bench_bwt_gpu_vs_cpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("no WebGPU device, skipping GPU experiment benchmarks");
            return;
        }
    };

    eprintln!("GPU device: {}", engine.device_name());

    let mut group = c.benchmark_group("bwt_gpu_vs_cpu");
    cap(&mut group);
    for &size in SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("cpu", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode(data).unwrap());
        });

        let eng = engine.clone();
        group.bench_with_input(BenchmarkId::new("gpu", size), &data, move |b, data| {
            b.iter(|| eng.bwt_encode(data).unwrap());
        });
    }
    group.finish();
}

/// GPU SortLZ — persistent device, measures match finding + compression.
#[cfg(feature = "webgpu")]
fn bench_sortlz_gpu(c: &mut Criterion) {
    use pz::sortlz::SortLzConfig;
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("no WebGPU device, skipping SortLZ benchmarks");
            return;
        }
    };

    let config = SortLzConfig::default();

    let mut group = c.benchmark_group("sortlz_gpu");
    cap(&mut group);
    for &size in SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Full pipeline: GPU match finding + CPU greedy parse + entropy coding
        let eng = engine.clone();
        let cfg = config.clone();
        group.bench_with_input(BenchmarkId::new("compress", size), &data, move |b, data| {
            b.iter(|| eng.sortlz_compress(data, &cfg).unwrap());
        });

        // Match finding only: isolates the GPU radix sort + verify stage
        let eng = engine.clone();
        let cfg = config.clone();
        group.bench_with_input(
            BenchmarkId::new("find_matches", size),
            &data,
            move |b, data| {
                b.iter(|| eng.sortlz_find_matches(data, &cfg).unwrap());
            },
        );
    }
    group.finish();
}

/// CPU SortLZ baseline for comparison.
fn bench_sortlz_cpu(c: &mut Criterion) {
    use pz::sortlz::{compress, SortLzConfig};

    let mut group = c.benchmark_group("sortlz_cpu");
    cap(&mut group);
    for &size in SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("compress", size), &data, |b, data| {
            b.iter(|| compress(data, &SortLzConfig::default()).unwrap());
        });
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_bwt_gpu_vs_cpu(_c: &mut Criterion) {}

#[cfg(not(feature = "webgpu"))]
fn bench_sortlz_gpu(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_bwt_gpu_vs_cpu,
    bench_sortlz_gpu,
    bench_sortlz_cpu
);
criterion_main!(benches);
