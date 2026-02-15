#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data, SIZES_LARGE, SIZES_SMALL};

fn bench_bbwt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbwt");
    cap(&mut group);
    for &size in SIZES_SMALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode_bijective(data).unwrap());
        });

        let (encoded, factor_lengths) = pz::bwt::encode_bijective(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode_bijective(enc, &factor_lengths).unwrap());
        });
    }
    group.finish();
}

fn bench_bbwt_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbwt_large");
    cap(&mut group);
    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode_bijective(data).unwrap());
        });

        let (encoded, factor_lengths) = pz::bwt::encode_bijective(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode_bijective(enc, &factor_lengths).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_bbwt, bench_bbwt_large);
criterion_main!(benches);
