#[path = "stages_common.rs"]
#[allow(dead_code)]
mod stages_common;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

fn bench_rans_decode_4way(c: &mut Criterion) {
    // Create a 1MB input with skewed distribution (100 distinct symbols)
    let input_1mb: Vec<u8> = (0..1_048_576)
        .map(|i| (i % 100) as u8) // skewed: 100 symbols, not 256
        .collect();

    // Use the public interleaved API: encode the input to get the format
    // that 4-way decode will operate on
    let encoded = pz::rans::encode_interleaved(&input_1mb);

    let mut g = c.benchmark_group("rans_decode_4way");
    g.throughput(Throughput::Bytes(input_1mb.len() as u64));

    // Benchmark the decode path (which internally uses either scalar or SSE2 4-way)
    g.bench_function("decode_interleaved", |b| {
        b.iter(|| pz::rans::decode_interleaved(&encoded, input_1mb.len()).unwrap())
    });

    g.finish();
}

fn bench_rans_decode_shared_stream(c: &mut Criterion) {
    // Same 1MB input as above — matches bench_rans_decode_4way for direct comparison
    let input_1mb: Vec<u8> = (0..1_048_576)
        .map(|i| (i % 100) as u8) // skewed: 100 symbols, not 256
        .collect();

    let encoded = pz::rans::encode_shared_stream(&input_1mb);
    let len = input_1mb.len();

    let mut g = c.benchmark_group("rans_decode_shared_stream");
    g.throughput(Throughput::Bytes(len as u64));

    // Side-by-side comparison of all implementations
    g.bench_function("scalar", |b| {
        b.iter(|| pz::rans::decode_shared_stream_force_scalar(&encoded, len).unwrap())
    });

    #[cfg(target_arch = "x86_64")]
    {
        g.bench_function("ssse3_pshufb", |b| {
            b.iter(|| pz::rans::decode_shared_stream_force_ssse3(&encoded, len).unwrap())
        });

        g.bench_function("avx2_gather", |b| {
            b.iter(|| pz::rans::decode_shared_stream_force_avx2(&encoded, len).unwrap())
        });
    }

    // Auto-dispatch (what the user gets by default)
    g.bench_function("auto", |b| {
        b.iter(|| pz::rans::decode_shared_stream(&encoded, len).unwrap())
    });

    g.finish();
}

criterion_group!(
    benches,
    bench_rans_decode_4way,
    bench_rans_decode_shared_stream
);
criterion_main!(benches);
