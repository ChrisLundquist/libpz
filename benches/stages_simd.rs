#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::get_test_data;

fn bench_simd(c: &mut Criterion) {
    use pz::simd::{scalar, Dispatcher};

    let mut group = c.benchmark_group("simd");
    for &size in &[8192, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("byte_freq_scalar", size),
            &data,
            |b, data| {
                b.iter(|| scalar::byte_frequencies(data));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("byte_freq_simd", size),
            &data,
            |b, data| {
                let d = Dispatcher::new();
                b.iter(|| d.byte_frequencies(data));
            },
        );

        let mut data2 = data.clone();
        let mismatch_pos = size / 2;
        data2[mismatch_pos] ^= 0xFF;

        group.bench_with_input(
            BenchmarkId::new("compare_scalar", size),
            &data,
            |b, data| {
                b.iter(|| scalar::compare_bytes(data, &data2, data.len().min(258)));
            },
        );

        group.bench_with_input(BenchmarkId::new("compare_simd", size), &data, |b, data| {
            let d = Dispatcher::new();
            b.iter(|| d.compare_bytes(data, &data2, 258));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_simd);
criterion_main!(benches);
