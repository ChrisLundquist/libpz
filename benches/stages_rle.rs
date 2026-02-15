#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

const SIZES_ALL: &[usize] = &[8192, 65536, 4_194_304];

fn bench_rle(c: &mut Criterion) {
    let mut group = c.benchmark_group("rle");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::rle::encode(data));
        });

        let encoded = pz::rle::encode(&data);
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::rle::decode(enc).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rle);
criterion_main!(benches);
