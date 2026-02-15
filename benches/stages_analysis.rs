#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::get_test_data;

fn bench_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis");
    for &size in &[8192, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("analyze", size), &data, |b, data| {
            b.iter(|| pz::analysis::analyze(data));
        });

        group.bench_with_input(
            BenchmarkId::new("analyze_sample_4k", size),
            &data,
            |b, data| {
                b.iter(|| pz::analysis::analyze_with_sample(data, 4096));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_analysis);
criterion_main!(benches);
