//! Per-algorithm stage benchmarks.
//!
//! Benchmarks each compression primitive individually at multiple input
//! sizes to identify bottlenecks and show scaling behavior. BWT's
//! O(n log^2 n) suffix array construction is expected to dominate.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::Path;

/// Load test data, truncated to the requested size.
fn get_test_data(size: usize) -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Try decompressing cantrbry.tar.gz using pz's own gzip support
    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                if decompressed.len() >= size {
                    return decompressed[..size].to_vec();
                }
                // Repeat to fill if needed
                let mut data = Vec::with_capacity(size);
                while data.len() < size {
                    let remaining = size - data.len();
                    let chunk = remaining.min(decompressed.len());
                    data.extend_from_slice(&decompressed[..chunk]);
                }
                return data;
            }
        }
    }

    // Fallback: synthetic data
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let full = pattern.repeat((size / pattern.len()) + 1);
    full[..size].to_vec()
}

fn bench_bwt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bwt");
    for &size in &[1024, 10240, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode(data).unwrap());
        });

        // Pre-encode for decode benchmark
        let encoded = pz::bwt::encode(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode(&enc.data, enc.primary_index).unwrap());
        });
    }
    group.finish();
}

fn bench_lz77(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz77");
    for &size in &[1024, 10240, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(
            BenchmarkId::new("compress_hashchain", size),
            &data,
            |b, data| {
                b.iter(|| pz::lz77::compress_hashchain(data).unwrap());
            },
        );

        group.bench_with_input(
            BenchmarkId::new("compress_lazy", size),
            &data,
            |b, data| {
                b.iter(|| pz::lz77::compress_lazy(data).unwrap());
            },
        );

        // Decompress benchmark
        let compressed = pz::lz77::compress_hashchain(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decompress", size),
            &compressed,
            |b, compressed| {
                b.iter(|| pz::lz77::decompress(compressed).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_huffman(c: &mut Criterion) {
    let mut group = c.benchmark_group("huffman");
    for &size in &[1024, 10240, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| tree.encode(data).unwrap());
        });

        let (encoded, total_bits) = tree.encode(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode", size),
            &encoded,
            |b, encoded| {
                let mut out = vec![0u8; size];
                b.iter(|| tree.decode_to_buf(encoded, total_bits, &mut out).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_mtf(c: &mut Criterion) {
    let mut group = c.benchmark_group("mtf");
    for &size in &[1024, 10240, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::mtf::encode(data));
        });

        let encoded = pz::mtf::encode(&data);
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::mtf::decode(enc));
        });
    }
    group.finish();
}

fn bench_rle(c: &mut Criterion) {
    let mut group = c.benchmark_group("rle");
    for &size in &[1024, 10240, 65536] {
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

fn bench_rangecoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("rangecoder");
    for &size in &[1024, 10240, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::rangecoder::encode(data));
        });

        let encoded = pz::rangecoder::encode(&data);
        group.bench_with_input(
            BenchmarkId::new("decode", size),
            &encoded,
            |b, enc| {
                b.iter(|| pz::rangecoder::decode(enc, size).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "opencl")]
fn bench_bwt_gpu(c: &mut Criterion) {
    use pz::opencl::OpenClEngine;

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no OpenCL device, skipping GPU BWT benchmarks");
            return;
        }
    };

    eprintln!("stages: GPU device: {}", engine.device_name());

    let mut group = c.benchmark_group("bwt_gpu");
    for &size in &[1024, 10240, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(BenchmarkId::new("encode_gpu", size), &data, move |b, data| {
            b.iter(|| eng.bwt_encode(data).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "opencl")]
fn bench_lz77_gpu(c: &mut Criterion) {
    use pz::opencl::{KernelVariant, OpenClEngine};

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lz77_gpu");
    for &size in &[1024, 10240, 65536] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("compress_gpu_batch", size),
            &data,
            move |b, data| {
                b.iter(|| eng.lz77_compress(data, KernelVariant::Batch).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_bwt_gpu(_c: &mut Criterion) {}

#[cfg(not(feature = "opencl"))]
fn bench_lz77_gpu(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_bwt,
    bench_lz77,
    bench_huffman,
    bench_mtf,
    bench_rle,
    bench_rangecoder,
    bench_bwt_gpu,
    bench_lz77_gpu
);
criterion_main!(benches);
