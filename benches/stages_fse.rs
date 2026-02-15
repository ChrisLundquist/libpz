#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

const SIZES_ALL: &[usize] = &[8192, 65536, 4_194_304];

fn bench_fse(c: &mut Criterion) {
    let mut group = c.benchmark_group("fse");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::fse::encode(data));
        });

        let encoded = pz::fse::encode(&data);
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::fse::decode(enc, size).unwrap());
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fse);
criterion_main!(benches);
