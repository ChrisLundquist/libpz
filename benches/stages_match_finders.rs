#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

use pz::lz77;
use pz::sortlz::{self, SortLzConfig};

const SIZES: &[usize] = &[8192, 65536, 262_144];

/// Raw match-finding throughput: hash-chain vs sortlz.
fn bench_match_finding(c: &mut Criterion) {
    let mut group = c.benchmark_group("match_finding");
    cap(&mut group);

    for &size in SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("hashchain_lazy", size),
            &data,
            |b, data| {
                b.iter(|| lz77::compress_lazy_to_matches(data).unwrap());
            },
        );

        group.bench_with_input(BenchmarkId::new("sortlz", size), &data, |b, data| {
            b.iter(|| sortlz::find_matches(data, &SortLzConfig::default()));
        });
    }
    group.finish();
}

/// Pipeline roundtrip: {hashchain, sortlz} x {greedy, lazy, optimal} on Lzr.
fn bench_pipeline_lzr(c: &mut Criterion) {
    use pz::pipeline::{self, CompressOptions, MatchFinder, ParseStrategy, Pipeline};

    let mut group = c.benchmark_group("lzr_match_finders");
    cap(&mut group);

    let size = 65536;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    for (finder_name, finder) in [
        ("hashchain", MatchFinder::HashChain),
        ("sortlz", MatchFinder::SortLz),
    ] {
        for (strategy_name, strategy) in [
            ("greedy", ParseStrategy::Greedy),
            ("lazy", ParseStrategy::Lazy),
            ("optimal", ParseStrategy::Optimal),
        ] {
            let label = format!("{finder_name}_{strategy_name}");
            let opts = CompressOptions {
                match_finder: finder,
                parse_strategy: strategy,
                threads: 1,
                ..Default::default()
            };
            group.bench_with_input(BenchmarkId::new(&label, size), &data, |b, data| {
                b.iter(|| pipeline::compress_with_options(data, Pipeline::Lzr, &opts).unwrap());
            });
        }
    }
    group.finish();
}

/// Compression ratio comparison (not timed — measures output size).
fn bench_ratio_comparison(c: &mut Criterion) {
    use pz::pipeline::{self, CompressOptions, MatchFinder, ParseStrategy, Pipeline};

    let mut group = c.benchmark_group("ratio_comparison");
    cap(&mut group);

    let size = 65536;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    let configs: Vec<(&str, MatchFinder, ParseStrategy)> = vec![
        ("hc_greedy", MatchFinder::HashChain, ParseStrategy::Greedy),
        ("hc_lazy", MatchFinder::HashChain, ParseStrategy::Lazy),
        ("hc_optimal", MatchFinder::HashChain, ParseStrategy::Optimal),
        ("slz_greedy", MatchFinder::SortLz, ParseStrategy::Greedy),
        ("slz_lazy", MatchFinder::SortLz, ParseStrategy::Lazy),
        ("slz_optimal", MatchFinder::SortLz, ParseStrategy::Optimal),
    ];

    for (name, finder, strategy) in configs {
        let opts = CompressOptions {
            match_finder: finder,
            parse_strategy: strategy,
            threads: 1,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::new(name, size), &data, |b, data| {
            b.iter(|| pipeline::compress_with_options(data, Pipeline::Lzr, &opts).unwrap());
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_match_finding,
    bench_pipeline_lzr,
    bench_ratio_comparison
);
criterion_main!(benches);
