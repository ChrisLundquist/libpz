#[path = "throughput_common.rs"]
mod throughput_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pz::pipeline::{self, Pipeline};
use throughput_common::{cap, get_test_data};

fn bench_compress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("compress_bbw");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(BenchmarkId::new("pz", "Bbw"), &data, |b, data| {
        b.iter(|| pipeline::compress(data, Pipeline::Bbw).unwrap());
    });
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("decompress_bbw");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function(BenchmarkId::new("pz", "Bbw"), |b| {
        let compressed = pipeline::compress(&data, Pipeline::Bbw).unwrap();
        b.iter(|| pipeline::decompress(&compressed).unwrap());
    });
    group.finish();
}

fn bench_compress_parallel(c: &mut Criterion) {
    use pz::pipeline::CompressOptions;

    let data = get_test_data();
    let mut group = c.benchmark_group("compress_parallel_bbw");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(BenchmarkId::new("pz_mt", "Bbw"), &data, |b, data| {
        let opts = CompressOptions {
            threads: 0,
            ..Default::default()
        };
        b.iter(|| pipeline::compress_with_options(data, Pipeline::Bbw, &opts).unwrap());
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_compress_parallel
);
criterion_main!(benches);
