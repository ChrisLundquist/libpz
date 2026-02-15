#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

fn bench_lz_plus_fse(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz_plus_fse");
    cap(&mut group);

    let size = 65536;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(
        BenchmarkId::new("lz77_fse_compress", size),
        &data,
        |b, data| {
            b.iter(|| {
                let lz = pz::lz77::compress_lazy(data).unwrap();
                pz::fse::encode(&lz)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("lzss_fse_compress", size),
        &data,
        |b, data| {
            b.iter(|| {
                let lz = pz::lzss::encode(data).unwrap();
                pz::fse::encode(&lz)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("lz78_fse_compress", size),
        &data,
        |b, data| {
            b.iter(|| {
                let lz = pz::lz78::encode(data).unwrap();
                pz::fse::encode(&lz)
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_lz_plus_fse);
criterion_main!(benches);
