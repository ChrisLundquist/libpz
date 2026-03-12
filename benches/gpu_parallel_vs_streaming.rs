/// Criterion benchmark: GPU parallel (in-memory) vs GPU streaming for Lzf.
///
/// Compares four paths at 4 MB and 16 MB:
///   1. parallel CPU-only  (compress_with_options, threads=0)
///   2. parallel GPU       (compress_with_options, Backend::WebGpu, threads=0)
///   3. streaming CPU-only (compress_stream, threads=0)
///   4. streaming GPU      (compress_stream, Backend::WebGpu, threads=0)
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pz::pipeline::{self, CompressOptions, Pipeline};
use pz::streaming;
use std::io::Cursor;
use std::path::Path;
use std::time::Duration;

fn get_test_data_sized(size: usize) -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let base = ["alice29.txt", "cantrbry.tar"]
        .iter()
        .filter_map(|name| {
            let p = manifest.join("samples").join(name);
            std::fs::read(&p).ok().filter(|d| !d.is_empty())
        })
        .next()
        .unwrap_or_else(|| b"The quick brown fox jumps over the lazy dog. ".repeat(3000));

    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        data.extend_from_slice(&base[..remaining.min(base.len())]);
    }
    data
}

fn bench_parallel_vs_streaming(c: &mut Criterion) {
    let pipeline = Pipeline::Lzf;

    #[cfg(feature = "webgpu")]
    let engine = {
        use pz::webgpu::WebGpuEngine;
        match WebGpuEngine::new() {
            Ok(e) => Some(std::sync::Arc::new(e)),
            Err(e) => {
                eprintln!("gpu_parallel_vs_streaming: no GPU device ({e}), GPU benches skipped");
                None
            }
        }
    };

    for &size in &[4_194_304usize, 16_777_216] {
        let data = get_test_data_sized(size);
        let label = format!("{}MB", size / (1024 * 1024));

        let mut group = c.benchmark_group(format!("gpu_par_vs_stream_{label}"));
        group.warm_up_time(Duration::from_secs(3));
        group.measurement_time(Duration::from_secs(8));
        group.sample_size(10);
        group.throughput(Throughput::Bytes(size as u64));

        // --- parallel CPU ---
        {
            let opts = CompressOptions {
                threads: 0,
                ..Default::default()
            };
            group.bench_with_input(
                BenchmarkId::new("parallel_cpu", &label),
                &data,
                |b, data| {
                    b.iter(|| pipeline::compress_with_options(data, pipeline, &opts).unwrap());
                },
            );
        }

        // --- streaming CPU ---
        {
            let opts = CompressOptions {
                threads: 0,
                ..Default::default()
            };
            group.bench_with_input(
                BenchmarkId::new("streaming_cpu", &label),
                &data,
                |b, data| {
                    b.iter(|| {
                        let input = Cursor::new(data);
                        let mut output = Vec::with_capacity(data.len());
                        streaming::compress_stream(input, &mut output, pipeline, &opts).unwrap();
                        output
                    });
                },
            );
        }

        // --- parallel GPU ---
        #[cfg(feature = "webgpu")]
        if let Some(ref engine) = engine {
            use pz::pipeline::Backend;
            let opts = CompressOptions {
                backend: Backend::WebGpu,
                threads: 0,
                webgpu_engine: Some(engine.clone()),
                ..Default::default()
            };
            group.bench_with_input(
                BenchmarkId::new("parallel_gpu", &label),
                &data,
                |b, data| {
                    b.iter(|| pipeline::compress_with_options(data, pipeline, &opts).unwrap());
                },
            );
        }

        // --- streaming GPU ---
        #[cfg(feature = "webgpu")]
        if let Some(ref engine) = engine {
            use pz::pipeline::Backend;
            let opts = CompressOptions {
                backend: Backend::WebGpu,
                threads: 0,
                webgpu_engine: Some(engine.clone()),
                ..Default::default()
            };
            group.bench_with_input(
                BenchmarkId::new("streaming_gpu", &label),
                &data,
                |b, data| {
                    b.iter(|| {
                        let input = Cursor::new(data);
                        let mut output = Vec::with_capacity(data.len());
                        streaming::compress_stream(input, &mut output, pipeline, &opts).unwrap();
                        output
                    });
                },
            );
        }

        group.finish();
    }
}

criterion_group!(benches, bench_parallel_vs_streaming);
criterion_main!(benches);
