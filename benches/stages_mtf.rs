#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data, SIZES_ALL};

fn bench_mtf(c: &mut Criterion) {
    let mut group = c.benchmark_group("mtf");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::mtf::encode(data));
        });

        let encoded = pz::mtf::encode(&data);
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::mtf::decode(enc));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_mtf);
criterion_main!(benches);
