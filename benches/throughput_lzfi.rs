#[path = "throughput_common.rs"]
mod throughput_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pz::pipeline::{self, Pipeline};
use throughput_common::{cap, get_test_data};

fn bench_compress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("compress_lzfi");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_with_input(BenchmarkId::new("pz", "Lzfi"), &data, |b, data| {
        b.iter(|| pipeline::compress(data, Pipeline::Lzfi).unwrap());
    });
    group.finish();
}

fn bench_decompress(c: &mut Criterion) {
    let data = get_test_data();
    let mut group = c.benchmark_group("decompress_lzfi");
    cap(&mut group);
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function(BenchmarkId::new("pz", "Lzfi"), |b| {
        let compressed = pipeline::compress(&data, Pipeline::Lzfi).unwrap();
        b.iter(|| pipeline::decompress(&compressed).unwrap());
    });
    group.finish();
}

criterion_group!(benches, bench_compress, bench_decompress);
criterion_main!(benches);
