#[path = "throughput_common.rs"]
mod throughput_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pz::pipeline::{self, Pipeline};
use throughput_common::{cap, get_test_data, get_test_data_sized};

fn bench_compress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("compress_lzr");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(BenchmarkId::new("pz", "Lzr"), &data, |b, data| {
        b.iter(|| pipeline::compress(data, Pipeline::Lzr).unwrap());
    });
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("decompress_lzr");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function(BenchmarkId::new("pz", "Lzr"), |b| {
        let compressed = pipeline::compress(&data, Pipeline::Lzr).unwrap();
        b.iter(|| pipeline::decompress(&compressed).unwrap());
    });
    group.finish();
}

fn bench_compress_parallel(c: &mut Criterion) {
    use pz::pipeline::CompressOptions;

    let data = get_test_data();
    let mut group = c.benchmark_group("compress_parallel_lzr");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(BenchmarkId::new("pz_mt", "Lzr"), &data, |b, data| {
        let opts = CompressOptions {
            threads: 0,
            ..Default::default()
        };
        b.iter(|| pipeline::compress_with_options(data, Pipeline::Lzr, &opts).unwrap());
    });

    group.finish();
}

fn bench_compress_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("compress_large_lzr");
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("Lzr", size), &data, |b, data| {
            b.iter(|| pipeline::compress(data, Pipeline::Lzr).unwrap());
        });
    }

    group.finish();
}

fn bench_decompress_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("decompress_large_lzr");
    cap(&mut group);

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(BenchmarkId::new("Lzr", size), |b| {
            let compressed = pipeline::compress(&data, Pipeline::Lzr).unwrap();
            b.iter(|| pipeline::decompress(&compressed).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_compress,
    bench_decompress,
    bench_compress_parallel,
    bench_compress_large,
    bench_decompress_large
);
criterion_main!(benches);
