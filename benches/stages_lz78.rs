#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data, SIZES_ALL};

fn bench_lz78(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz78");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::lz78::encode(data).unwrap());
        });

        let compressed = pz::lz78::encode(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode", size),
            &compressed,
            |b, compressed| {
                b.iter(|| pz::lz78::decode(compressed).unwrap());
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_lz78);
criterion_main!(benches);
