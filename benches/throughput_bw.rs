#[path = "throughput_common.rs"]
mod throughput_common;

use criterion::{criterion_group, criterion_main, Criterion};
use pz::pipeline::Pipeline;
use throughput_common::{run_throughput_benches, ThroughputBenchSpec};

const SPEC: ThroughputBenchSpec = ThroughputBenchSpec {
    id: "bw",
    pipeline: Pipeline::Bw,
    parallel: true,
    large: false,
    decompress_large: false,
    webgpu: true,
    webgpu_large: true,
};

fn bench(c: &mut Criterion) {
    run_throughput_benches(c, &SPEC);
}

criterion_group!(benches, bench);
criterion_main!(benches);
