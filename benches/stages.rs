//! Per-algorithm stage benchmarks.
//!
//! Benchmarks each compression primitive individually at multiple input
//! sizes to identify bottlenecks and show scaling behavior. BWT's
//! O(n log^2 n) suffix array construction is expected to dominate.
//!
//! Size tiers:
//!   - Small:  8KB, 64KB             — all algorithms
//!   - Large:  4MB                   — fast algorithms, scaling behavior
//!   - GPU:    256KB, 4MB            — GPU crossover region
//!
//! All groups enforce warm_up_time(2s) + measurement_time(5s) + sample_size(10)
//! to keep total runtime bounded.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::path::Path;
use std::time::Duration;

/// Small sizes benchmarked for slow algorithms (BWT).
const SIZES_SMALL: &[usize] = &[8192, 65536];
/// Representative sizes for fast algorithms (LZ77, Huffman, entropy coders, etc.).
const SIZES_ALL: &[usize] = &[8192, 65536, 4_194_304];
/// Large sizes only — for targeted GPU-vs-CPU comparisons.
const SIZES_LARGE: &[usize] = &[262_144, 4_194_304];

/// Apply standard timeout caps to a benchmark group.
fn cap(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>) {
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
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

fn bench_bbwt(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbwt");
    cap(&mut group);
    for &size in SIZES_SMALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode_bijective(data).unwrap());
        });

        let (encoded, factor_lengths) = pz::bwt::encode_bijective(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode_bijective(enc, &factor_lengths).unwrap());
        });
    }
    group.finish();
}

fn bench_bbwt_large(c: &mut Criterion) {
    let mut group = c.benchmark_group("bbwt_large");
    cap(&mut group);
    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::bwt::encode_bijective(data).unwrap());
        });

        let (encoded, factor_lengths) = pz::bwt::encode_bijective(&data).unwrap();
        group.bench_with_input(BenchmarkId::new("decode", size), &encoded, |b, enc| {
            b.iter(|| pz::bwt::decode_bijective(enc, &factor_lengths).unwrap());
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

        // GPU modular Deflate (GPU LZ77 → GPU Huffman via composable stages)
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("gpu_modular", size),
            &data,
            move |b, data| {
                let opts = CompressOptions {
                    backend: pz::pipeline::Backend::OpenCl,
                    threads: 1,
                    opencl_engine: Some(eng.clone()),
                    ..Default::default()
                };
                b.iter(|| {
                    pz::pipeline::compress_with_options(
                        data,
                        pz::pipeline::Pipeline::Deflate,
                        &opts,
                    )
                    .unwrap()
                });
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

#[cfg(feature = "opencl")]
fn bench_lz77_gpu_params(c: &mut Criterion) {
    use pz::opencl::OpenClEngine;

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lz77_gpu_params");
    cap(&mut group);

    // Use 256KB input — the GPU crossover region
    let size = 262_144;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    // Parameter sweep: hash_bits x max_chain x bucket_cap
    for &hash_bits in &[14u32, 15, 16, 18] {
        for &max_chain in &[16u32, 64, 128] {
            for &bucket_cap in &[16u32, 32, 64] {
                let label = format!("h{hash_bits}_c{max_chain}_b{bucket_cap}");
                let eng = engine.clone();
                group.bench_with_input(BenchmarkId::new(&label, size), &data, move |b, data| {
                    b.iter(|| {
                        eng.find_matches_with_params(data, hash_bits, max_chain, bucket_cap)
                            .unwrap()
                    });
                });
            }
        }
    }

    // Also measure compression ratio for a few key configurations
    // (not timed, just ratio comparison)
    let cpu_compressed = pz::lz77::compress_lazy(&data).unwrap();
    let cpu_ratio = cpu_compressed.len() as f64 / data.len() as f64;

    for &(hash_bits, max_chain, bucket_cap) in
        &[(14, 64, 64), (15, 64, 64), (16, 64, 64), (18, 128, 64)]
    {
        let matches = engine
            .find_matches_with_params(&data, hash_bits, max_chain, bucket_cap)
            .unwrap();
        let compressed_size = matches.len() * 5; // 5 bytes per match
        let gpu_ratio = compressed_size as f64 / data.len() as f64;
        eprintln!(
            "  h{hash_bits}_c{max_chain}_b{bucket_cap}: ratio={gpu_ratio:.3} (cpu_lazy={cpu_ratio:.3})"
        );
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_lz77_gpu_params(_c: &mut Criterion) {}

#[cfg(feature = "opencl")]
fn bench_fse_gpu(c: &mut Criterion) {
    use pz::opencl::OpenClEngine;

    let engine = match OpenClEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => return,
    };

    let mut group = c.benchmark_group("fse_gpu");
    cap(&mut group);

    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Encode with interleaved FSE at various K values
        for &num_states in &[4usize, 8, 16, 32] {
            let encoded = pz::fse::encode_interleaved_n(&data, num_states, 7);

            // CPU decode baseline
            let label_cpu = format!("decode_cpu_k{num_states}");
            group.bench_with_input(
                BenchmarkId::new(&label_cpu, size),
                &encoded,
                |b, encoded| {
                    b.iter(|| pz::fse::decode_interleaved(encoded, data.len()).unwrap());
                },
            );

            // GPU decode
            let label_gpu = format!("decode_gpu_k{num_states}");
            let eng = engine.clone();
            let orig_len = data.len();
            group.bench_with_input(
                BenchmarkId::new(&label_gpu, size),
                &encoded,
                move |b, encoded| {
                    b.iter(|| eng.fse_decode(encoded, orig_len).unwrap());
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_fse_gpu(_c: &mut Criterion) {}

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

fn bench_simd(c: &mut Criterion) {
    use pz::simd::{scalar, Dispatcher};

    let mut group = c.benchmark_group("simd");
    for &size in &[8192, 65536] {
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
            b.iter(|| d.compare_bytes(data, &data2, 258));
        });
    }
    group.finish();
}

fn bench_auto_select(c: &mut Criterion) {
    use pz::pipeline::{self, CompressOptions};

    let mut group = c.benchmark_group("auto_select");
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

fn bench_lzss(c: &mut Criterion) {
    let mut group = c.benchmark_group("lzss");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::lzss::encode(data).unwrap());
        });

        let compressed = pz::lzss::encode(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode", size),
            &compressed,
            |b, compressed| {
                b.iter(|| pz::lzss::decode(compressed).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_lz78(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz78");
    cap(&mut group);
    for &size in SIZES_ALL {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        group.bench_with_input(BenchmarkId::new("encode", size), &data, |b, data| {
            b.iter(|| pz::lz78::encode(data).unwrap());
        });

        let compressed = pz::lz78::encode(&data).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode", size),
            &compressed,
            |b, compressed| {
                b.iter(|| pz::lz78::decode(compressed).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_lz_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz_comparison");
    cap(&mut group);

    let size = 65536;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    // Compression
    group.bench_with_input(BenchmarkId::new("lz77_compress", size), &data, |b, data| {
        b.iter(|| pz::lz77::compress_lazy(data).unwrap());
    });
    group.bench_with_input(BenchmarkId::new("lzss_compress", size), &data, |b, data| {
        b.iter(|| pz::lzss::encode(data).unwrap());
    });
    group.bench_with_input(BenchmarkId::new("lz78_compress", size), &data, |b, data| {
        b.iter(|| pz::lz78::encode(data).unwrap());
    });

    // Decompression
    let lz77_c = pz::lz77::compress_lazy(&data).unwrap();
    let lzss_c = pz::lzss::encode(&data).unwrap();
    let lz78_c = pz::lz78::encode(&data).unwrap();

    group.bench_with_input(
        BenchmarkId::new("lz77_decompress", size),
        &lz77_c,
        |b, c| {
            b.iter(|| pz::lz77::decompress(c).unwrap());
        },
    );
    group.bench_with_input(
        BenchmarkId::new("lzss_decompress", size),
        &lzss_c,
        |b, c| {
            b.iter(|| pz::lzss::decode(c).unwrap());
        },
    );
    group.bench_with_input(
        BenchmarkId::new("lz78_decompress", size),
        &lz78_c,
        |b, c| {
            b.iter(|| pz::lz78::decode(c).unwrap());
        },
    );

    group.finish();
}

fn bench_lz_plus_fse(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz_plus_fse");
    cap(&mut group);

    let size = 65536;
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    group.bench_with_input(
        BenchmarkId::new("lz77_fse_compress", size),
        &data,
        |b, data| {
            b.iter(|| {
                let lz = pz::lz77::compress_lazy(data).unwrap();
                pz::fse::encode(&lz)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("lzss_fse_compress", size),
        &data,
        |b, data| {
            b.iter(|| {
                let lz = pz::lzss::encode(data).unwrap();
                pz::fse::encode(&lz)
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("lz78_fse_compress", size),
        &data,
        |b, data| {
            b.iter(|| {
                let lz = pz::lz78::encode(data).unwrap();
                pz::fse::encode(&lz)
            });
        },
    );

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

        // GPU lazy matching (default — 3-pass with lazy resolution)
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("compress_webgpu_lazy", size),
            &data,
            move |b, data| {
                b.iter(|| eng.lz77_compress(data).unwrap());
            },
        );

        // GPU greedy matching (original 2-pass, for comparison)
        let eng2 = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("compress_webgpu_greedy", size),
            &data,
            move |b, data| {
                b.iter(|| {
                    let matches = eng2.find_matches_greedy(data).unwrap();
                    let mut out =
                        Vec::with_capacity(matches.len() * pz::lz77::Match::SERIALIZED_SIZE);
                    for m in &matches {
                        out.extend_from_slice(&m.to_bytes());
                    }
                    out
                });
            },
        );

        // CPU lazy matching (baseline)
        group.bench_with_input(
            BenchmarkId::new("compress_cpu_lazy", size),
            &data,
            |b, data| {
                b.iter(|| pz::lz77::compress_lazy(data).unwrap());
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

        // WebGPU modular Deflate (WebGPU LZ77 → WebGPU Huffman via composable stages)
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("webgpu_modular", size),
            &data,
            move |b, data| {
                let opts = CompressOptions {
                    backend: pz::pipeline::Backend::WebGpu,
                    threads: 1,
                    webgpu_engine: Some(eng.clone()),
                    ..Default::default()
                };
                b.iter(|| {
                    pz::pipeline::compress_with_options(
                        data,
                        pz::pipeline::Pipeline::Deflate,
                        &opts,
                    )
                    .unwrap()
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_lz77_webgpu_batched(c: &mut Criterion) {
    use pz::pipeline::{Backend, CompressOptions, Pipeline};
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("stages: no WebGPU device, skipping batched LZ77 benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("lz77_webgpu_batched");
    cap(&mut group);
    // Multi-block sizes: 4 blocks × 256KB = 1MB, 16 blocks × 256KB = 4MB
    for &size in &[1_048_576, 4_194_304] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // GPU batched: submit all blocks → single sync → readback → CPU entropy
        let eng = engine.clone();
        group.bench_with_input(
            BenchmarkId::new("gpu_batched_deflate", size),
            &data,
            move |b, data| {
                let opts = CompressOptions {
                    backend: Backend::WebGpu,
                    webgpu_engine: Some(eng.clone()),
                    threads: 4,
                    ..Default::default()
                };
                b.iter(|| {
                    pz::pipeline::compress_with_options(data, Pipeline::Deflate, &opts).unwrap()
                });
            },
        );

        // CPU multi-threaded baseline (same thread count, no GPU)
        group.bench_with_input(
            BenchmarkId::new("cpu_parallel_deflate", size),
            &data,
            |b, data| {
                let opts = CompressOptions {
                    threads: 4,
                    ..Default::default()
                };
                b.iter(|| {
                    pz::pipeline::compress_with_options(data, Pipeline::Deflate, &opts).unwrap()
                });
            },
        );
    }
    group.finish();
}

#[cfg(not(feature = "webgpu"))]
fn bench_lz77_webgpu_batched(_c: &mut Criterion) {}

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

fn bench_lz78_static(c: &mut Criterion) {
    let mut group = c.benchmark_group("lz78_static");
    cap(&mut group);

    // Test with various dict sizes at a fixed input size
    let size = 262_144; // 256KB — large enough for dictionary overhead to amortize
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    for &dict_size in &[512u16, 1024, 2048, 4096] {
        for &max_entry_len in &[16u16, 32, 64] {
            let label = format!("encode_d{dict_size}_l{max_entry_len}");
            let ds = dict_size;
            let mel = max_entry_len;
            group.bench_with_input(BenchmarkId::new(&label, size), &data, |b, data| {
                b.iter(|| pz::lz78_static::encode_with_params(data, ds, mel).unwrap());
            });
        }
    }

    // Decode benchmarks: encode once with default params, benchmark decode
    let compressed = pz::lz78_static::encode(&data).unwrap();
    group.bench_with_input(
        BenchmarkId::new("decode_cpu", size),
        &compressed,
        |b, compressed| {
            b.iter(|| pz::lz78_static::decode(compressed).unwrap());
        },
    );

    group.finish();
}

#[cfg(feature = "opencl")]
fn bench_lz78_static_gpu(c: &mut Criterion) {
    let engine = match pz::opencl::OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lz78_static_gpu");
    cap(&mut group);

    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Encode on CPU, then benchmark GPU decode vs CPU decode
        let compressed = pz::lz78_static::encode(&data).unwrap();
        let (dict, packed_codes, num_codes, original_len) =
            pz::lz78_static::parse_for_gpu(&compressed).unwrap();

        // CPU decode baseline
        group.bench_with_input(
            BenchmarkId::new("decode_cpu", size),
            &compressed,
            |b, compressed| {
                b.iter(|| pz::lz78_static::decode(compressed).unwrap());
            },
        );

        // GPU decode
        group.bench_function(BenchmarkId::new("decode_gpu", size), |b| {
            b.iter(|| {
                engine
                    .lzw_decode_static(packed_codes, num_codes, &dict, original_len)
                    .unwrap()
            });
        });
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_lz78_static_gpu(_c: &mut Criterion) {}

/// Experiment 4: LZ77 block-parallel GPU decompression.
///
/// Sweeps block_size (4KB-256KB) and cooperative_threads (1, 8, 32, 64).
/// Compares GPU block-parallel decode vs CPU sequential decode.
#[cfg(feature = "opencl")]
fn bench_lz77_decompress_blocks(c: &mut Criterion) {
    let engine = match pz::opencl::OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lz77_decompress_blocks");
    cap(&mut group);

    let block_sizes: &[usize] = &[4096, 16384, 32768, 65536, 262144];
    let thread_counts: &[usize] = &[1, 8, 32, 64];

    // Use 1MB test data
    let data = get_test_data(1_048_576);

    for &block_size in block_sizes {
        // Compress into independent blocks
        let (block_data, block_meta) =
            pz::opencl::lz77::lz77_compress_blocks(&data, block_size).unwrap();

        let label_bs = if block_size >= 1024 {
            format!("{}KB", block_size / 1024)
        } else {
            format!("{}B", block_size)
        };

        // CPU baseline: decompress each block independently
        group.throughput(Throughput::Bytes(data.len() as u64));
        group.bench_function(BenchmarkId::new("cpu", &label_bs), |b| {
            b.iter(|| {
                let mut result = Vec::with_capacity(data.len());
                for &(offset, num_matches, _decompressed_size) in &block_meta {
                    let end = offset + num_matches * 5;
                    let block_compressed = &block_data[offset..end];
                    let decoded = pz::lz77::decompress(block_compressed).unwrap();
                    result.extend_from_slice(&decoded);
                }
                result
            });
        });

        // GPU: sweep cooperative thread counts
        for &threads in thread_counts {
            group.bench_function(
                BenchmarkId::new(format!("gpu_t{threads}"), &label_bs),
                |b| {
                    b.iter(|| {
                        engine
                            .lz77_decompress_blocks(&block_data, &block_meta, threads)
                            .unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_lz77_decompress_blocks(_c: &mut Criterion) {}

/// Experiment 1: rANS interleaved GPU decode.
///
/// Sweeps K (interleaved lanes: 4, 8, 16, 32, 64) and scale_bits (10, 11, 12).
/// Compares GPU decode vs CPU decode throughput.
#[cfg(feature = "opencl")]
fn bench_rans_gpu(c: &mut Criterion) {
    let engine = match pz::opencl::OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("rans_gpu_decode");
    cap(&mut group);

    let interleave_counts: &[usize] = &[4, 8, 16, 32, 64];
    let scale_values: &[u8] = &[10, 11, 12];

    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        let label_size = if size >= 1_048_576 {
            format!("{}MB", size / 1_048_576)
        } else {
            format!("{}KB", size / 1024)
        };

        group.throughput(Throughput::Bytes(size as u64));

        for &scale_bits in scale_values {
            for &k in interleave_counts {
                let encoded = pz::rans::encode_interleaved_n(&data, k, scale_bits);
                let label = format!("K{k}_s{scale_bits}_{label_size}");

                // CPU baseline
                group.bench_function(BenchmarkId::new("cpu", &label), |b| {
                    b.iter(|| pz::rans::decode_interleaved(&encoded, data.len()).unwrap());
                });

                // GPU
                group.bench_function(BenchmarkId::new("gpu", &label), |b| {
                    b.iter(|| {
                        engine
                            .rans_decode_interleaved(&encoded, data.len())
                            .unwrap()
                    });
                });
            }
        }
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_rans_gpu(_c: &mut Criterion) {}

/// Experiment 6: Combined LZ77 + entropy GPU decode pipeline.
///
/// Measures end-to-end GPU decode of independently-compressed LZ77 blocks
/// with FSE or rANS entropy coding, compared to CPU sequential decode.
/// Demonstrates the combined speedup potential of GPU entropy + LZ77 decode.
#[cfg(feature = "opencl")]
fn bench_combined_gpu_decode(c: &mut Criterion) {
    let engine = match pz::opencl::OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("combined_gpu_decode");
    cap(&mut group);

    let block_sizes: &[usize] = &[16384, 65536];

    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        let label_size = if size >= 1_048_576 {
            format!("{}MB", size / 1_048_576)
        } else {
            format!("{}KB", size / 1024)
        };

        group.throughput(Throughput::Bytes(size as u64));

        for &block_size in block_sizes {
            let label_bs = format!("{}KB", block_size / 1024);

            // Compress into independent blocks then entropy-code each block's LZ77 output
            let (block_data, block_meta) =
                pz::opencl::lz77::lz77_compress_blocks(&data, block_size).unwrap();

            // Also compress with FSE for entropy decode benchmarking
            let mut fse_encoded_blocks: Vec<Vec<u8>> = Vec::new();
            let mut fse_interleaved_blocks: Vec<Vec<u8>> = Vec::new();
            let mut fse_original_lens: Vec<usize> = Vec::new();
            for &(offset, num_matches, _decompressed_size) in &block_meta {
                let end = offset + num_matches * 5;
                let block_compressed = &block_data[offset..end];
                fse_encoded_blocks.push(pz::fse::encode(block_compressed));
                fse_interleaved_blocks.push(pz::fse::encode_interleaved(block_compressed));
                fse_original_lens.push(block_compressed.len());
            }

            // CPU baseline: FSE decode + LZ77 decompress per block
            group.bench_function(
                BenchmarkId::new("cpu_fse_lz77", format!("{label_size}_{label_bs}")),
                |b| {
                    b.iter(|| {
                        let mut result = Vec::with_capacity(data.len());
                        for (i, fse_block) in fse_encoded_blocks.iter().enumerate() {
                            let lz77_data =
                                pz::fse::decode(fse_block, fse_original_lens[i]).unwrap();
                            let decoded = pz::lz77::decompress(&lz77_data).unwrap();
                            result.extend_from_slice(&decoded);
                        }
                        result
                    });
                },
            );

            // GPU: LZ77 block-parallel decompress only (from pre-decoded LZ77 data)
            group.bench_function(
                BenchmarkId::new("gpu_lz77_only", format!("{label_size}_{label_bs}")),
                |b| {
                    b.iter(|| {
                        engine
                            .lz77_decompress_blocks(&block_data, &block_meta, 32)
                            .unwrap()
                    });
                },
            );

            // GPU: FSE decode (per-block) + LZ77 decompress (batched)
            group.bench_function(
                BenchmarkId::new("gpu_fse_lz77", format!("{label_size}_{label_bs}")),
                |b| {
                    b.iter(|| {
                        // Stage 1: GPU FSE decode each block
                        let mut all_lz77 = Vec::with_capacity(block_data.len());
                        let mut decoded_meta = Vec::new();
                        for (i, fse_block) in fse_interleaved_blocks.iter().enumerate() {
                            let lz77_data =
                                engine.fse_decode(fse_block, fse_original_lens[i]).unwrap();
                            let num_matches = lz77_data.len() / 5;
                            decoded_meta.push((all_lz77.len(), num_matches, block_meta[i].2));
                            all_lz77.extend_from_slice(&lz77_data);
                        }

                        // Stage 2: GPU LZ77 block-parallel decompress
                        engine
                            .lz77_decompress_blocks(&all_lz77, &decoded_meta, 32)
                            .unwrap()
                    });
                },
            );

            // Collect each block's LZ77 match data as separate slices for encoding
            let match_slices: Vec<&[u8]> = block_meta
                .iter()
                .map(|&(offset, num_matches, _)| {
                    let end = offset + num_matches * 5;
                    &block_data[offset..end]
                })
                .collect();
            // GPU: batched FSE decode (single kernel launch) + LZ77 decompress
            let fse_batched_result =
                pz::opencl::fse::fse_encode_block_slices(&match_slices, 4, 8).unwrap();
            let fse_batched_refs: Vec<(&[u8], usize)> = fse_batched_result
                .iter()
                .map(|(enc, len)| (enc.as_slice(), *len))
                .collect();

            group.bench_function(
                BenchmarkId::new("gpu_fse_batched_lz77", format!("{label_size}_{label_bs}")),
                |b| {
                    b.iter(|| {
                        // Stage 1: GPU batched FSE decode (all blocks in one launch)
                        let all_lz77 = engine.fse_decode_blocks(&fse_batched_refs).unwrap();

                        // Build metadata for LZ77 decompress
                        let mut decoded_meta = Vec::new();
                        let mut lz77_offset = 0usize;
                        for (i, &orig_len) in fse_original_lens.iter().enumerate() {
                            let num_matches = orig_len / 5;
                            decoded_meta.push((lz77_offset, num_matches, block_meta[i].2));
                            lz77_offset += orig_len;
                        }

                        // Stage 2: GPU LZ77 block-parallel decompress
                        engine
                            .lz77_decompress_blocks(&all_lz77, &decoded_meta, 32)
                            .unwrap()
                    });
                },
            );

            let rans_encoded_result =
                pz::opencl::rans::rans_encode_block_slices(&match_slices, 4, 12).unwrap();

            // GPU: batched rANS decode (single kernel launch) + LZ77 decompress
            let rans_block_refs: Vec<(&[u8], usize)> = rans_encoded_result
                .iter()
                .map(|(enc, len)| (enc.as_slice(), *len))
                .collect();

            group.bench_function(
                BenchmarkId::new("gpu_rans_lz77", format!("{label_size}_{label_bs}")),
                |b| {
                    b.iter(|| {
                        // Stage 1: GPU batched rANS decode (all blocks in one launch)
                        let all_lz77 = engine
                            .rans_decode_interleaved_blocks(&rans_block_refs)
                            .unwrap();

                        // Build metadata for LZ77 decompress
                        let mut decoded_meta = Vec::new();
                        let mut lz77_offset = 0usize;
                        for (i, &orig_len) in fse_original_lens.iter().enumerate() {
                            let num_matches = orig_len / 5;
                            decoded_meta.push((lz77_offset, num_matches, block_meta[i].2));
                            lz77_offset += orig_len;
                        }

                        // Stage 2: GPU LZ77 block-parallel decompress
                        engine
                            .lz77_decompress_blocks(&all_lz77, &decoded_meta, 32)
                            .unwrap()
                    });
                },
            );
        }
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_combined_gpu_decode(_c: &mut Criterion) {}

/// Experiment 1 (gap fix): Multi-block rANS GPU decode.
///
/// Splits input into independent blocks (e.g., 16KB each), encodes each
/// with K-way interleaved rANS, then decodes all blocks in a single GPU
/// kernel launch. Total work-items = num_blocks × K.
#[cfg(feature = "opencl")]
fn bench_rans_gpu_blocks(c: &mut Criterion) {
    let engine = match pz::opencl::OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("rans_gpu_blocks");
    cap(&mut group);

    let block_sizes: &[usize] = &[4096, 16384, 65536];
    let interleave_counts: &[usize] = &[8, 16, 32];

    for &size in SIZES_LARGE {
        let data = get_test_data(size);
        let label_size = if size >= 1_048_576 {
            format!("{}MB", size / 1_048_576)
        } else {
            format!("{}KB", size / 1024)
        };
        group.throughput(Throughput::Bytes(size as u64));

        for &block_size in block_sizes {
            for &k in interleave_counts {
                let label_bs = format!("{}KB", block_size / 1024);
                let scale_bits = 11u8;

                // Encode all blocks with shared frequency table
                let encoded_blocks =
                    pz::opencl::rans::rans_encode_blocks(&data, block_size, k, scale_bits).unwrap();
                let num_blocks = encoded_blocks.len();

                let label = format!("K{k}_bs{label_bs}_{label_size}_n{num_blocks}");

                // CPU baseline: decode each block sequentially
                group.bench_function(BenchmarkId::new("cpu", &label), |b| {
                    b.iter(|| {
                        let mut result = Vec::with_capacity(data.len());
                        for (enc, orig_len) in &encoded_blocks {
                            let decoded = pz::rans::decode_interleaved(enc, *orig_len).unwrap();
                            result.extend_from_slice(&decoded);
                        }
                        result
                    });
                });

                // GPU: batched multi-block decode
                let block_refs: Vec<(&[u8], usize)> = encoded_blocks
                    .iter()
                    .map(|(enc, len)| (enc.as_slice(), *len))
                    .collect();
                group.bench_function(BenchmarkId::new("gpu_blocks", &label), |b| {
                    b.iter(|| engine.rans_decode_interleaved_blocks(&block_refs).unwrap());
                });
            }
        }
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_rans_gpu_blocks(_c: &mut Criterion) {}

/// Experiment 3 (gap fix): Variant B short-chain hash match finder.
///
/// Compares Variant A (default 64-entry buckets) vs Variant B (2-4 entry
/// short buckets) in terms of compression ratio and throughput.
#[cfg(feature = "opencl")]
fn bench_lz77_gpu_short_chain(c: &mut Criterion) {
    let engine = match pz::opencl::OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lz77_short_chain");
    cap(&mut group);

    let size = 262144; // 256KB
    let data = get_test_data(size);
    group.throughput(Throughput::Bytes(size as u64));

    // Variant A baseline (default params: hash_bits=15, max_chain=64, bucket_cap=64)
    group.bench_function("variant_a_default", |b| {
        b.iter(|| engine.find_matches_with_params(&data, 15, 64, 64).unwrap());
    });

    // Variant B: short buckets
    for &bucket_cap in &[2u32, 3, 4] {
        group.bench_function(
            BenchmarkId::new("variant_b", format!("b{bucket_cap}")),
            |b| {
                b.iter(|| {
                    engine
                        .find_matches_short_chain(&data, 15, bucket_cap)
                        .unwrap()
                });
            },
        );
    }

    // Compare compression ratios
    let matches_a = engine.find_matches_with_params(&data, 15, 64, 64).unwrap();
    let ratio_a = {
        let mut comp = Vec::new();
        for m in &matches_a {
            comp.extend_from_slice(&m.offset.to_le_bytes());
            comp.extend_from_slice(&m.length.to_le_bytes());
            comp.push(m.next);
        }
        data.len() as f64 / comp.len() as f64
    };

    for &bucket_cap in &[2u32, 3, 4] {
        let matches_b = engine
            .find_matches_short_chain(&data, 15, bucket_cap)
            .unwrap();
        let ratio_b = {
            let mut comp = Vec::new();
            for m in &matches_b {
                comp.extend_from_slice(&m.offset.to_le_bytes());
                comp.extend_from_slice(&m.length.to_le_bytes());
                comp.push(m.next);
            }
            data.len() as f64 / comp.len() as f64
        };
        eprintln!(
            "[variant_b b{bucket_cap}] ratio: {ratio_b:.3} (vs variant_a: {ratio_a:.3}, diff: {:.1}%)",
            (ratio_b - ratio_a) / ratio_a * 100.0
        );
    }

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_lz77_gpu_short_chain(_c: &mut Criterion) {}

/// Experiment 5 (gap fix): Out-of-domain LZW benchmark.
///
/// Trains a dictionary on one dataset (alice29.txt) and encodes a different
/// dataset (synthetic random). Measures how domain mismatch affects ratio.
#[cfg(feature = "opencl")]
fn bench_lzw_out_of_domain(c: &mut Criterion) {
    let engine = match pz::opencl::OpenClEngine::new() {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("lzw_out_of_domain");
    cap(&mut group);

    let size = 262144; // 256KB

    // Training data: repetitive structured data
    let training_pattern = b"the quick brown fox jumps over the lazy dog ";
    let mut training_data = Vec::with_capacity(size);
    while training_data.len() < size {
        training_data.extend_from_slice(training_pattern);
    }
    training_data.truncate(size);

    // In-domain test: similar to training data
    let in_domain_pattern = b"a quick red fox leaps across the sleeping cat ";
    let mut in_domain = Vec::with_capacity(size);
    while in_domain.len() < size {
        in_domain.extend_from_slice(in_domain_pattern);
    }
    in_domain.truncate(size);

    // Out-of-domain test: random bytes (maximally different)
    let mut out_of_domain = vec![0u8; size];
    let mut rng_state = 42u64;
    for b in out_of_domain.iter_mut() {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *b = (rng_state >> 33) as u8;
    }

    group.throughput(Throughput::Bytes(size as u64));

    // Train dictionary on training data
    let dict = pz::lz78_static::FrozenDict::train(&training_data, 4096).unwrap();

    // In-domain encode + GPU decode
    let in_encoded = pz::lz78_static::encode_with_dict(&in_domain, &dict).unwrap();
    let (in_dict, in_codes, in_num_codes, in_orig_len) =
        pz::lz78_static::parse_for_gpu(&in_encoded).unwrap();
    let in_ratio = in_domain.len() as f64 / in_encoded.len() as f64;

    group.bench_function("in_domain_gpu", |b| {
        b.iter(|| {
            engine
                .lzw_decode_static(in_codes, in_num_codes, &in_dict, in_orig_len)
                .unwrap()
        });
    });

    // Out-of-domain encode + GPU decode
    let out_encoded = pz::lz78_static::encode_with_dict(&out_of_domain, &dict).unwrap();
    let (out_dict, out_codes, out_num_codes, out_orig_len) =
        pz::lz78_static::parse_for_gpu(&out_encoded).unwrap();
    let out_ratio = out_of_domain.len() as f64 / out_encoded.len() as f64;

    group.bench_function("out_of_domain_gpu", |b| {
        b.iter(|| {
            engine
                .lzw_decode_static(out_codes, out_num_codes, &out_dict, out_orig_len)
                .unwrap()
        });
    });

    eprintln!(
        "[lzw out-of-domain] in-domain ratio: {in_ratio:.3}, out-of-domain ratio: {out_ratio:.3}, \
         degradation: {:.1}%",
        (in_ratio - out_ratio) / in_ratio * 100.0
    );

    group.finish();
}

#[cfg(not(feature = "opencl"))]
fn bench_lzw_out_of_domain(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_bwt,
    bench_bwt_large,
    bench_bbwt,
    bench_bbwt_large,
    bench_lz77,
    bench_lzss,
    bench_lz78,
    bench_lz_comparison,
    bench_lz_plus_fse,
    bench_huffman,
    bench_mtf,
    bench_rle,
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
    bench_lz77_webgpu_batched,
    bench_huffman_webgpu,
    bench_deflate_webgpu_chained,
    bench_lz78_static,
    bench_lz78_static_gpu,
    bench_lz77_gpu_params,
    bench_fse_gpu,
    bench_lz77_decompress_blocks,
    bench_rans_gpu,
    bench_combined_gpu_decode,
    bench_rans_gpu_blocks,
    bench_lz77_gpu_short_chain,
    bench_lzw_out_of_domain
);
criterion_main!(benches);
