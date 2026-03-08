#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

use pz::lz77;
use pz::sortlz::{self, SortLzConfig};

const SIZES: &[usize] = &[8192, 65536, 262_144];
#[cfg(feature = "webgpu")]
const GPU_SIZES: &[usize] = &[8192, 65536, 262_144, 4_194_304];

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

/// GPU SortLZ match finding vs CPU hashchain — persistent device, no init noise.
///
/// This is the key benchmark for the library/daemon use case where device init
/// is amortized. Compares raw match-finding throughput and full pipeline
/// throughput across sizes.
#[cfg(feature = "webgpu")]
fn bench_gpu_match_finding(c: &mut Criterion) {
    use pz::pipeline::{self, Backend, CompressOptions, MatchFinder, ParseStrategy, Pipeline};
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("no WebGPU device, skipping GPU match-finding benchmarks");
            return;
        }
    };
    eprintln!("GPU device: {}", engine.device_name());

    // --- Part 1: Raw match-finding throughput (GPU sortlz vs CPU hashchain vs CPU sortlz) ---
    let mut group = c.benchmark_group("match_finding_gpu");
    cap(&mut group);

    for &size in GPU_SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // CPU hashchain
        group.bench_with_input(BenchmarkId::new("cpu_hashchain", size), &data, |b, data| {
            b.iter(|| lz77::compress_lazy_to_matches(data).unwrap());
        });

        // CPU sortlz
        group.bench_with_input(BenchmarkId::new("cpu_sortlz", size), &data, |b, data| {
            b.iter(|| sortlz::find_matches(data, &SortLzConfig::default()));
        });

        // GPU sortlz
        let eng = engine.clone();
        let cfg = SortLzConfig::default();
        group.bench_with_input(
            BenchmarkId::new("gpu_sortlz", size),
            &data,
            move |b, data| {
                b.iter(|| eng.sortlz_find_matches(data, &cfg).unwrap());
            },
        );
    }
    group.finish();

    // --- Part 2: Full Lzr pipeline with GPU sortlz match finder ---
    let mut group = c.benchmark_group("lzr_gpu_match_finders");
    cap(&mut group);

    for &size in GPU_SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // CPU hashchain + Lzr
        let opts = CompressOptions {
            parse_strategy: ParseStrategy::Lazy,
            threads: 1,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::new("cpu_hashchain", size), &data, |b, data| {
            b.iter(|| pipeline::compress_with_options(data, Pipeline::Lzr, &opts).unwrap());
        });

        // CPU sortlz + Lzr
        let opts = CompressOptions {
            match_finder: MatchFinder::SortLz,
            parse_strategy: ParseStrategy::Lazy,
            threads: 1,
            ..Default::default()
        };
        group.bench_with_input(BenchmarkId::new("cpu_sortlz", size), &data, |b, data| {
            b.iter(|| pipeline::compress_with_options(data, Pipeline::Lzr, &opts).unwrap());
        });

        // GPU sortlz + Lzr (GPU match finding, CPU parse + entropy)
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("gpu_sortlz", size),
            &data,
            move |b, data| {
                let opts = CompressOptions {
                    backend: Backend::WebGpu,
                    webgpu_engine: Some(eng.clone()),
                    match_finder: MatchFinder::SortLz,
                    parse_strategy: ParseStrategy::Lazy,
                    threads: 1,
                    ..Default::default()
                };
                b.iter(|| pipeline::compress_with_options(data, Pipeline::Lzr, &opts).unwrap());
            },
        );
    }
    group.finish();

    // --- Part 3: Cross-pipeline GPU sortlz (Deflate, Lzr, Lzf) at 256K ---
    let mut group = c.benchmark_group("gpu_sortlz_pipelines");
    cap(&mut group);

    let size = 262_144;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    for (name, pipeline) in [
        ("deflate", Pipeline::Deflate),
        ("lzr", Pipeline::Lzr),
        ("lzf", Pipeline::Lzf),
    ] {
        // CPU hashchain baseline
        let opts = CompressOptions {
            parse_strategy: ParseStrategy::Lazy,
            threads: 1,
            ..Default::default()
        };
        group.bench_with_input(
            BenchmarkId::new(format!("{name}_cpu_hc"), size),
            &data,
            |b, data| {
                b.iter(|| pipeline::compress_with_options(data, pipeline, &opts).unwrap());
            },
        );

        // GPU sortlz
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new(format!("{name}_gpu_slz"), size),
            &data,
            move |b, data| {
                let opts = CompressOptions {
                    backend: Backend::WebGpu,
                    webgpu_engine: Some(eng.clone()),
                    match_finder: MatchFinder::SortLz,
                    parse_strategy: ParseStrategy::Lazy,
                    threads: 1,
                    ..Default::default()
                };
                b.iter(|| pipeline::compress_with_options(data, pipeline, &opts).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_gpu_match_finding(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_match_finding,
    bench_pipeline_lzr,
    bench_ratio_comparison,
    bench_gpu_match_finding
);
criterion_main!(benches);
