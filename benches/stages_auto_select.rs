#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

fn bench_auto_select(c: &mut Criterion) {
    use pz::pipeline::{self, CompressOptions};

    let mut group = c.benchmark_group("auto_select");
    cap(&mut group);
    for &size in &[8192, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("heuristic", size), &data, |b, data| {
            b.iter(|| pipeline::select_pipeline(data));
        });

        group.bench_with_input(BenchmarkId::new("trial_4k", size), &data, |b, data| {
            let opts = CompressOptions::default();
            b.iter(|| pipeline::select_pipeline_trial(data, &opts, 4096));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_auto_select);
criterion_main!(benches);
