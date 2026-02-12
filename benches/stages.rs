//! Per-algorithm stage benchmarks.
//!
//! Benchmarks each compression primitive individually at multiple input
//! sizes to identify bottlenecks and show scaling behavior. BWT's
//! O(n log^2 n) suffix array construction is expected to dominate.
//!
//! Size tiers:
//!   - Small:  1KB, 10KB, 64KB       — all algorithms
//!   - Medium: 256KB                  — GPU crossover region
//!   - Large:  4MB, 16MB              — GPU advantage expected
//!
//! All groups enforce warm_up_time(5s) + measurement_time(10s) + sample_size(10)
//! to keep total runtime bounded (~60s per slow benchmark).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::Path;
use std::time::Duration;

/// Small/medium sizes benchmarked for all algorithms.
const SIZES_SMALL: &[usize] = &[1024, 10240, 65536];
/// All sizes including large tiers for fast algorithms (LZ77, Huffman, etc.).
const SIZES_ALL: &[usize] = &[1024, 10240, 65536, 262_144, 4_194_304, 16_777_216];
/// Large sizes only — for targeted GPU-vs-CPU comparisons.
const SIZES_LARGE: &[usize] = &[262_144, 4_194_304];

/// Apply standard timeout caps to a benchmark group.
fn cap(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    group.warm_up_time(Duration::from_secs(5));
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);
}

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
    cap(&mut group);
    for &size in SIZES_SMALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode(data).unwrap());
        });

        let encoded = pz::bwt::encode(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode(&enc.data, enc.primary_index).unwrap());
        });
    }
    group.finish();
}

fn bench_bwt_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("bwt_large");
    cap(&mut group);
    // Cap at 4MB — 16MB BWT CPU encode is ~6s/iter, too slow for criterion.
    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode(data).unwrap());
        });

        let encoded = pz::bwt::encode(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode(&enc.data, enc.primary_index).unwrap());
        });
    }
    group.finish();
}

fn bench_lz77(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz77");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("compress_lazy", size), &data, |b, data| {
            b.iter(|| pz::lz77::compress_lazy(data).unwrap());
        });

        // Decompress benchmark
        let compressed = pz::lz77::compress_lazy(&data).unwrap();
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
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| tree.encode(data).unwrap());
        });

        let (encoded, total_bits) = tree.encode(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, encoded| {
            let mut out = vec![0u8; size];
            b.iter(|| tree.decode_to_buf(encoded, total_bits, &mut out).unwrap());
        });
    }
    group.finish();
}

fn bench_mtf(c: &mut Criterion) {
    let mut group = c.benchmark_group("mtf");
    cap(&mut group);
    for &size in SIZES_ALL {
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

fn bench_rangecoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("rangecoder");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::rangecoder::encode(data));
        });

        let encoded = pz::rangecoder::encode(&data);
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::rangecoder::decode(enc, size).unwrap());
        });
    }
    group.finish();
}

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

fn bench_rans(c: &mut Criterion) {
    let mut group = c.benchmark_group("rans");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::rans::encode(data));
        });

        let encoded = pz::rans::encode(&data);
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::rans::decode(enc, size).unwrap());
        });
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
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("encode_gpu", size),
            &data,
            move |b, data| {
                b.iter(|| eng.bwt_encode(data).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "opencl")]
fn bench_huffman_gpu(c: &mut Criterion) {
    use pz::opencl::OpenClEngine;

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no OpenCL device, skipping GPU Huffman benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("huffman_gpu");
    cap(&mut group);
    for &size in &[10240, 65536, 262_144, 4_194_304, 16_777_216] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Build tree and LUT
        let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();
        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        // CPU baseline
        let tree_clone = tree.clone();
        group.bench_with_input(
            BenchmarkId::new("encode_cpu", size),
            &data,
            move |b, data| {
                b.iter(|| tree_clone.encode(data).unwrap());
            },
        );

        // GPU with CPU prefix sum
        let eng1 = engine.clone();
        let lut1 = code_lut;
        group.bench_with_input(
            BenchmarkId::new("encode_gpu_cpu_scan", size),
            &data,
            move |b, data| {
                b.iter(|| eng1.huffman_encode(data, &lut1).unwrap());
            },
        );

        // GPU with GPU prefix sum
        let eng2 = engine.clone();
        let lut2 = code_lut;
        group.bench_with_input(
            BenchmarkId::new("encode_gpu_gpu_scan", size),
            &data,
            move |b, data| {
                b.iter(|| eng2.huffman_encode_gpu_scan(data, &lut2).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "opencl")]
fn bench_deflate_gpu_chained(c: &mut Criterion) {
    use pz::opencl::OpenClEngine;
    use pz::pipeline::CompressOptions;

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no OpenCL device, skipping GPU Deflate chained benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("deflate_gpu_chained");
    cap(&mut group);
    for &size in &[65536, 262_144, 1_048_576, 4_194_304, 16_777_216] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // CPU Deflate (single-threaded)
        group.bench_with_input(BenchmarkId::new("cpu_1t", size), &data, |b, data| {
            let opts = CompressOptions {
                threads: 1,
                ..Default::default()
            };
            b.iter(|| {
                pz::pipeline::compress_with_options(data, pz::pipeline::Pipeline::Deflate, &opts)
                    .unwrap()
            });
        });

        // GPU chained Deflate
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("gpu_chained", size),
            &data,
            move |b, data| {
                b.iter(|| eng.deflate_chained(data).unwrap());
            },
        );
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
    cap(&mut group);
    for &size in SIZES_ALL {
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

        let eng2 = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("compress_gpu_hash", size),
            &data,
            move |b, data| {
                b.iter(|| eng2.lz77_compress(data, KernelVariant::HashTable).unwrap());
            },
        );
    }
    group.finish();
}

// bench_lz77_parallel removed: parallel LZ77 match-finding was slower than
// single-threaded lazy. Multi-threading now happens at the pipeline block level.

fn bench_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("analysis");
    for &size in &[1024, 10240, 65536, 262144] {
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

fn bench_simd(c: &mut Criterion) {
    use pz::simd::{scalar, Dispatcher};

    let mut group = c.benchmark_group("simd");
    for &size in &[1024, 10240, 65536, 262_144, 1_048_576] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Byte frequency counting: scalar baseline
        group.bench_with_input(
            BenchmarkId::new("byte_freq_scalar", size),
            &data,
            |b, data| {
                b.iter(|| scalar::byte_frequencies(data));
            },
        );

        // Byte frequency counting: SIMD dispatch
        group.bench_with_input(
            BenchmarkId::new("byte_freq_simd", size),
            &data,
            |b, data| {
                let d = Dispatcher::new();
                b.iter(|| d.byte_frequencies(data));
            },
        );

        // Match comparison: create two copies with a known mismatch
        let mut data2 = data.clone();
        let mismatch_pos = size / 2;
        data2[mismatch_pos] ^= 0xFF;

        // Compare bytes: scalar baseline
        group.bench_with_input(
            BenchmarkId::new("compare_scalar", size),
            &data,
            |b, data| {
                b.iter(|| scalar::compare_bytes(data, &data2, data.len().min(258)));
            },
        );

        // Compare bytes: SIMD dispatch
        group.bench_with_input(BenchmarkId::new("compare_simd", size), &data, |b, data| {
            let d = Dispatcher::new();
            b.iter(|| d.compare_bytes(data, &data2));
        });
    }
    group.finish();
}

fn bench_auto_select(c: &mut Criterion) {
    use pz::pipeline::{self, CompressOptions};

    let mut group = c.benchmark_group("auto_select");
    for &size in &[10240, 65536, 262144] {
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

#[cfg(not(feature = "opencl"))]
fn bench_bwt_gpu(_c: &mut Criterion) {}

#[cfg(not(feature = "opencl"))]
fn bench_lz77_gpu(_c: &mut Criterion) {}

#[cfg(not(feature = "opencl"))]
fn bench_huffman_gpu(_c: &mut Criterion) {}

#[cfg(not(feature = "opencl"))]
fn bench_deflate_gpu_chained(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn bench_lz77_webgpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lz77_webgpu");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("compress_webgpu_hash", size),
            &data,
            move |b, data| {
                b.iter(|| eng.lz77_compress(data).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_deflate_webgpu_chained(c: &mut Criterion) {
    use pz::pipeline::CompressOptions;
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no WebGPU device, skipping WebGPU Deflate chained benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("deflate_webgpu_chained");
    cap(&mut group);
    for &size in &[65536, 262_144, 1_048_576, 4_194_304, 16_777_216] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // CPU Deflate (single-threaded) baseline
        group.bench_with_input(BenchmarkId::new("cpu_1t", size), &data, |b, data| {
            let opts = CompressOptions {
                threads: 1,
                ..Default::default()
            };
            b.iter(|| {
                pz::pipeline::compress_with_options(data, pz::pipeline::Pipeline::Deflate, &opts)
                    .unwrap()
            });
        });

        // WebGPU chained Deflate
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("webgpu_chained", size),
            &data,
            move |b, data| {
                b.iter(|| eng.deflate_chained(data).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_bwt_webgpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no WebGPU device, skipping WebGPU BWT benchmarks");
            return;
        }
    };

    eprintln!("stages: WebGPU device: {}", engine.device_name());

    let mut group = c.benchmark_group("bwt_webgpu");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("encode_webgpu", size),
            &data,
            move |b, data| {
                b.iter(|| eng.bwt_encode(data).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_bwt_webgpu(_c: &mut Criterion) {}

#[cfg(feature = "webgpu")]
fn bench_huffman_webgpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no WebGPU device, skipping WebGPU Huffman benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("huffman_webgpu");
    cap(&mut group);
    for &size in &[10240, 65536, 262_144, 4_194_304, 16_777_216] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Build tree and LUT
        let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();
        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        // CPU baseline
        let tree_clone = tree.clone();
        group.bench_with_input(
            BenchmarkId::new("encode_cpu", size),
            &data,
            move |b, data| {
                b.iter(|| tree_clone.encode(data).unwrap());
            },
        );

        // WebGPU encode (CPU prefix sum)
        let eng = engine.clone();
        let lut = code_lut;
        group.bench_with_input(
            BenchmarkId::new("encode_webgpu", size),
            &data,
            move |b, data| {
                b.iter(|| eng.huffman_encode(data, &lut).unwrap());
            },
        );

        // WebGPU encode (GPU prefix sum — Blelloch scan)
        let eng2 = engine.clone();
        let lut2 = code_lut;
        group.bench_with_input(
            BenchmarkId::new("encode_webgpu_gpu_scan", size),
            &data,
            move |b, data| {
                b.iter(|| eng2.huffman_encode_gpu_scan(data, &lut2).unwrap());
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_huffman_webgpu(_c: &mut Criterion) {}

#[cfg(not(feature = "webgpu"))]
fn bench_lz77_webgpu(_c: &mut Criterion) {}

#[cfg(not(feature = "webgpu"))]
fn bench_deflate_webgpu_chained(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_bwt,
    bench_bwt_large,
    bench_lz77,
    bench_huffman,
    bench_mtf,
    bench_rle,
    bench_rangecoder,
    bench_fse,
    bench_rans,
    bench_simd,
    bench_analysis,
    bench_auto_select,
    bench_bwt_gpu,
    bench_lz77_gpu,
    bench_huffman_gpu,
    bench_deflate_gpu_chained,
    bench_bwt_webgpu,
    bench_lz77_webgpu,
    bench_huffman_webgpu,
    bench_deflate_webgpu_chained
);
criterion_main!(benches);
