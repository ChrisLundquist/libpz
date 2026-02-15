#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

fn bench_lz_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz_comparison");
    cap(&mut group);

    let size = 65536;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(BenchmarkId::new("lz77_compress", size), &data, |b, data| {
        b.iter(|| pz::lz77::compress_lazy(data).unwrap());
    });
    group.bench_with_input(BenchmarkId::new("lzss_compress", size), &data, |b, data| {
        b.iter(|| pz::lzss::encode(data).unwrap());
    });
    group.bench_with_input(BenchmarkId::new("lz78_compress", size), &data, |b, data| {
        b.iter(|| pz::lz78::encode(data).unwrap());
    });

    let lz77_c = pz::lz77::compress_lazy(&data).unwrap();
    let lzss_c = pz::lzss::encode(&data).unwrap();
    let lz78_c = pz::lz78::encode(&data).unwrap();

    group.bench_with_input(
        BenchmarkId::new("lz77_decompress", size),
        &lz77_c,
        |b, c| {
            b.iter(|| pz::lz77::decompress(c).unwrap());
        },
    );
    group.bench_with_input(
        BenchmarkId::new("lzss_decompress", size),
        &lzss_c,
        |b, c| {
            b.iter(|| pz::lzss::decode(c).unwrap());
        },
    );
    group.bench_with_input(
        BenchmarkId::new("lz78_decompress", size),
        &lz78_c,
        |b, c| {
            b.iter(|| pz::lz78::decode(c).unwrap());
        },
    );

    group.finish();
}

criterion_group!(benches, bench_lz_comparison);
criterion_main!(benches);
