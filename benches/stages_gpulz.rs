#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

const SIZES: &[usize] = &[8192, 65536, 131072, 4_194_304];

// ---------------------------------------------------------------------------
// Experiment 1: Huffman sync-point encode overhead
// ---------------------------------------------------------------------------

fn bench_huffman_encode_sync(c: &mut Criterion) {
    let mut group = c.benchmark_group("huffman_encode_sync");
    cap(&mut group);

    for &size in SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Build tree (same for all variants).
        let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();
        let mut tree_canon = tree.clone();
        tree_canon.canonicalize();

        // Baseline: plain encode.
        let tree_b = tree.clone();
        group.bench_with_input(BenchmarkId::new("baseline", size), &data, move |b, data| {
            b.iter(|| tree_b.encode(data).unwrap());
        });

        // Sync-point encode at various intervals.
        for &interval in &[512u32, 1024, 2048] {
            let tree_s = tree_canon.clone();
            group.bench_with_input(
                BenchmarkId::new(format!("sync_{interval}"), size),
                &data,
                move |b, data| {
                    b.iter(|| tree_s.encode_with_sync_points(data, interval).unwrap());
                },
            );
        }
    }
    group.finish();
}

/// Report sync-point overhead: compressed sizes.
fn bench_huffman_sync_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("huffman_sync_ratio");
    cap(&mut group);

    for &size in &[65536usize, 131072, 4_194_304] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();
        let mut tree_canon = tree.clone();
        tree_canon.canonicalize();

        // Baseline compressed size.
        let (baseline_enc, baseline_bits) = tree.encode(&data).unwrap();
        let baseline_bytes = baseline_enc.len();

        // Sync-point sizes (bitstream is the same, overhead is sync point metadata).
        for &interval in &[512u32, 1024, 2048] {
            let result = tree_canon.encode_with_sync_points(&data, interval).unwrap();
            let sync_metadata = result.sync_points.len() * 8; // 8 bytes per sync point
            let total_wire = result.data.len() + 256 + 2 + sync_metadata; // code_lengths + num_sp + sp + data
            let baseline_wire = baseline_bytes + 256; // code_lengths for fair comparison

            // This benchmark just reports the ratio — the "work" is trivial.
            let interval_copy = interval;
            group.bench_with_input(
                BenchmarkId::new(format!("overhead_{interval_copy}"), size),
                &data,
                move |b, _data| {
                    b.iter(|| {
                        // Return sizes for criterion to measure (tiny work).
                        (
                            baseline_wire,
                            total_wire,
                            baseline_bits,
                            result.total_bits,
                            sync_metadata,
                        )
                    });
                },
            );

            // Print ratio info (visible in bench output).
            let overhead_pct = (total_wire as f64 / baseline_wire as f64 - 1.0) * 100.0;
            eprintln!(
                "  [{size}B, interval={interval}] baseline={baseline_wire}B, sync={total_wire}B, \
                 overhead=+{sync_metadata}B (+{overhead_pct:.2}%), \
                 sync_points={}",
                result.sync_points.len()
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Experiment 2: Tiled Huffman decode throughput
// ---------------------------------------------------------------------------

fn bench_huffman_decode_tiled(c: &mut Criterion) {
    let mut group = c.benchmark_group("huffman_decode_tiled");
    cap(&mut group);

    for &size in SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Build tree with decode table.
        let freq = pz::frequency::get_frequency(&data);
        let tree = pz::huffman::HuffmanTree::from_frequency_table(&freq).unwrap();
        let mut tree_canon = tree.clone();
        tree_canon.canonicalize();

        // Baseline: monolithic decode.
        let (encoded, total_bits) = tree.encode(&data).unwrap();
        let tree_d = tree.clone();
        let encoded_d = encoded.clone();
        group.bench_with_input(
            BenchmarkId::new("baseline", size),
            &data,
            move |b, _data| {
                let mut out = vec![0u8; size];
                b.iter(|| {
                    tree_d
                        .decode_to_buf(&encoded_d, total_bits, &mut out)
                        .unwrap()
                });
            },
        );

        // Tiled decode at various intervals.
        for &interval in &[512u32, 1024, 2048] {
            let result = tree_canon.encode_with_sync_points(&data, interval).unwrap();

            // Rebuild decoder from code lengths (as the GPU would do).
            let code_lengths = tree_canon.code_lengths();
            let decoder = rebuild_from_lengths(&code_lengths);

            let enc = result.data.clone();
            let tb = result.total_bits;
            let sps = result.sync_points.clone();
            group.bench_with_input(
                BenchmarkId::new(format!("tiled_{interval}"), size),
                &data,
                move |b, _data| {
                    b.iter(|| decoder.decode_tiled(&enc, tb, &sps).unwrap());
                },
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Experiment 3: Full GpuLz block encode/decode vs baselines
// ---------------------------------------------------------------------------

fn bench_gpulz_block(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpulz_block");
    cap(&mut group);

    for &size in SIZES {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // --- Compression ---

        // GpuLz compress.
        group.bench_with_input(
            BenchmarkId::new("compress_gpulz", size),
            &data,
            |b, data| {
                b.iter(|| pz::gpulz::compress_block(data).unwrap());
            },
        );

        // LzSeqH compress (baseline).
        group.bench_with_input(
            BenchmarkId::new("compress_lzseqh", size),
            &data,
            |b, data| {
                b.iter(|| {
                    pz::pipeline::compress_with_options(
                        data,
                        pz::pipeline::Pipeline::LzSeqH,
                        &pz::pipeline::CompressOptions {
                            threads: 1,
                            ..Default::default()
                        },
                    )
                    .unwrap()
                });
            },
        );

        // Lzf compress (baseline).
        group.bench_with_input(BenchmarkId::new("compress_lzf", size), &data, |b, data| {
            b.iter(|| {
                pz::pipeline::compress_with_options(
                    data,
                    pz::pipeline::Pipeline::Lzf,
                    &pz::pipeline::CompressOptions {
                        threads: 1,
                        ..Default::default()
                    },
                )
                .unwrap()
            });
        });

        // --- Decompression ---

        // GpuLz decompress.
        let gpulz_compressed = pz::gpulz::compress_block(&data).unwrap();
        let gpulz_ratio = gpulz_compressed.len() as f64 / data.len() as f64 * 100.0;
        eprintln!(
            "  [{size}B] gpulz ratio: {gpulz_ratio:.1}% ({} -> {} bytes)",
            data.len(),
            gpulz_compressed.len()
        );

        group.bench_with_input(
            BenchmarkId::new("decompress_gpulz", size),
            &data,
            |b, _data| {
                b.iter(|| pz::gpulz::decompress_block(&gpulz_compressed, size).unwrap());
            },
        );

        // LzSeqH decompress (baseline).
        let lzseqh_compressed = pz::pipeline::compress_with_options(
            &data,
            pz::pipeline::Pipeline::LzSeqH,
            &pz::pipeline::CompressOptions {
                threads: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let lzseqh_ratio = lzseqh_compressed.len() as f64 / data.len() as f64 * 100.0;
        eprintln!(
            "  [{size}B] lzseqh ratio: {lzseqh_ratio:.1}% ({} -> {} bytes)",
            data.len(),
            lzseqh_compressed.len()
        );

        group.bench_with_input(
            BenchmarkId::new("decompress_lzseqh", size),
            &data,
            |b, _data| {
                b.iter(|| pz::pipeline::decompress(&lzseqh_compressed).unwrap());
            },
        );

        // Lzf decompress (baseline).
        let lzf_compressed = pz::pipeline::compress_with_options(
            &data,
            pz::pipeline::Pipeline::Lzf,
            &pz::pipeline::CompressOptions {
                threads: 1,
                ..Default::default()
            },
        )
        .unwrap();
        let lzf_ratio = lzf_compressed.len() as f64 / data.len() as f64 * 100.0;
        eprintln!(
            "  [{size}B] lzf ratio: {lzf_ratio:.1}% ({} -> {} bytes)",
            data.len(),
            lzf_compressed.len()
        );

        group.bench_with_input(
            BenchmarkId::new("decompress_lzf", size),
            &data,
            |b, _data| {
                b.iter(|| pz::pipeline::decompress(&lzf_compressed).unwrap());
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Experiment 4: GPU Huffman decode + full GpuLz decompress
// ---------------------------------------------------------------------------

#[cfg(feature = "webgpu")]
fn bench_huffman_decode_gpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("no WebGPU device, skipping GPU Huffman decode benchmarks");
            return;
        }
    };
    eprintln!("GPU device: {}", engine.device_name());

    let mut group = c.benchmark_group("huffman_decode_gpu");
    cap(&mut group);

    for &size in &[65536usize, 131072, 4_194_304] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        // Build canonical tree and encode with sync points.
        let freq = pz::frequency::get_frequency(&data);
        let tree = pz::huffman::HuffmanTree::from_frequency_table(&freq).unwrap();
        let mut tree_canon = tree.clone();
        tree_canon.canonicalize();

        let result = tree_canon.encode_with_sync_points(&data, 1024).unwrap();

        // Rebuild decoder from code lengths.
        let code_lengths = tree_canon.code_lengths();
        let decoder = rebuild_from_lengths(&code_lengths);

        // CPU monolithic baseline.
        {
            let (encoded, total_bits) = tree.encode(&data).unwrap();
            let tree_d = tree.clone();
            let encoded_d = encoded.clone();
            group.bench_with_input(BenchmarkId::new("cpu_mono", size), &data, move |b, _| {
                let mut out = vec![0u8; size];
                b.iter(|| {
                    tree_d
                        .decode_to_buf(&encoded_d, total_bits, &mut out)
                        .unwrap()
                });
            });
        }

        // CPU tiled baseline.
        {
            let enc = result.data.clone();
            let tb = result.total_bits;
            let sps = result.sync_points.clone();
            let dec = decoder.clone();
            group.bench_with_input(
                BenchmarkId::new("cpu_tiled_1024", size),
                &data,
                move |b, _| {
                    b.iter(|| dec.decode_tiled(&enc, tb, &sps).unwrap());
                },
            );
        }

        // GPU sync-point decode.
        {
            let lut = tree_canon.build_gpu_decode_lut();
            let lut_array: Box<[u32; 4096]> = lut.into_boxed_slice().try_into().unwrap();
            let enc = result.data.clone();
            let tb = result.total_bits;
            let sps = result.sync_points.clone();
            let eng = engine.clone();
            group.bench_with_input(BenchmarkId::new("gpu_sync", size), &data, move |b, _| {
                b.iter(|| {
                    eng.huffman_decode_gpu(&enc, tb, &lut_array, &sps, size)
                        .unwrap()
                });
            });
        }
    }
    group.finish();
}

#[cfg(feature = "webgpu")]
fn bench_gpulz_decompress_gpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("no WebGPU device, skipping GPU GpuLz decompress benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("gpulz_decompress_gpu");
    cap(&mut group);

    for &size in &[65536usize, 131072, 4_194_304] {
        let data = get_test_data(size);
        group.throughput(Throughput::Bytes(size as u64));

        let compressed = pz::gpulz::compress_block(&data).unwrap();
        let ratio = compressed.len() as f64 / data.len() as f64 * 100.0;
        eprintln!("  [{size}B] gpulz ratio: {ratio:.1}%");

        // CPU decompress baseline.
        {
            let comp = compressed.clone();
            group.bench_with_input(BenchmarkId::new("cpu", size), &data, move |b, _| {
                b.iter(|| pz::gpulz::decompress_block(&comp, size).unwrap());
            });
        }

        // GPU decompress (GPU Huffman + CPU LzSeq reconstruct).
        {
            let comp = compressed.clone();
            let eng = engine.clone();
            group.bench_with_input(BenchmarkId::new("gpu", size), &data, move |b, _| {
                b.iter(|| pz::gpulz::decompress_block_gpu(&eng, &comp, size).unwrap());
            });
        }
    }
    group.finish();
}

/// Report per-phase GPU decompress timing breakdown (not a throughput benchmark).
#[cfg(feature = "webgpu")]
fn bench_gpulz_gpu_timing(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("no WebGPU device, skipping GPU timing breakdown");
            return;
        }
    };

    let mut group = c.benchmark_group("gpulz_gpu_timing");
    cap(&mut group);

    for &size in &[65536usize, 131072, 4_194_304] {
        let data = get_test_data(size);
        let compressed = pz::gpulz::compress_block(&data).unwrap();

        // Warmup + collect timings.
        let mut timings = Vec::new();
        for _ in 0..20 {
            let (_, t) = pz::gpulz::decompress_block_gpu_timed(&engine, &compressed, size).unwrap();
            timings.push(t);
        }

        // Report median timings.
        let n = timings.len();
        let mut parse: Vec<u64> = timings.iter().map(|t| t.parse_us).collect();
        let mut gpu: Vec<u64> = timings.iter().map(|t| t.gpu_huffman_us).collect();
        let mut gpu_buf: Vec<u64> = timings.iter().map(|t| t.gpu_buffers_us).collect();
        let mut gpu_sub: Vec<u64> = timings.iter().map(|t| t.gpu_submit_us).collect();
        let mut gpu_rb: Vec<u64> = timings.iter().map(|t| t.gpu_readback_us).collect();
        let mut lzseq: Vec<u64> = timings.iter().map(|t| t.lzseq_us).collect();
        let mut total: Vec<u64> = timings.iter().map(|t| t.total_us).collect();
        parse.sort();
        gpu.sort();
        gpu_buf.sort();
        gpu_sub.sort();
        gpu_rb.sort();
        lzseq.sort();
        total.sort();

        let med = n / 2;
        eprintln!(
            "  [{size}B] median: parse={parse}µs  gpu={gpu}µs [buf={buf}µs sub={sub}µs rb={rb}µs]  \
             lzseq={lzseq}µs  total={total}µs  gpu%={gpu_pct:.0}%  lzseq%={lzseq_pct:.0}%",
            parse = parse[med],
            gpu = gpu[med],
            buf = gpu_buf[med],
            sub = gpu_sub[med],
            rb = gpu_rb[med],
            lzseq = lzseq[med],
            total = total[med],
            gpu_pct = gpu[med] as f64 / total[med] as f64 * 100.0,
            lzseq_pct = lzseq[med] as f64 / total[med] as f64 * 100.0,
        );

        // Dummy bench so criterion doesn't complain about empty group.
        let comp = compressed.clone();
        let eng = engine.clone();
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("gpu_timed", size), &data, move |b, _| {
            b.iter(|| pz::gpulz::decompress_block_gpu(&eng, &comp, size).unwrap());
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Experiment 5: Multi-block batched GPU decompress
// ---------------------------------------------------------------------------

/// Multi-block batched GPU decompress with parallel CPU LzSeq.
///
/// Simulates the target pipeline architecture: N pipeline blocks' Huffman
/// decodes batched into a single GPU submission, then LzSeq reconstructed
/// in parallel on CPU threads.
#[cfg(feature = "webgpu")]
fn bench_gpulz_multiblock_gpu(c: &mut Criterion) {
    use pz::webgpu::WebGpuEngine;

    let engine = match WebGpuEngine::new() {
        Ok(e) => std::sync::Arc::new(e),
        Err(_) => {
            eprintln!("no WebGPU device, skipping multi-block GPU benchmarks");
            return;
        }
    };

    let mut group = c.benchmark_group("gpulz_multiblock_gpu");
    cap(&mut group);

    // Test with fixed block size (128KB) and varying block count.
    // Use Canterbury corpus block 0 (realistic ~46% ratio), replicated N times.
    let block_size = 131072usize; // 128KB
    let block_data = get_test_data(block_size);
    let reference_compressed = pz::gpulz::compress_block(&block_data).unwrap();
    let _reference_ratio = reference_compressed.len() as f64 / block_data.len() as f64 * 100.0;
    pz::gpulz::decompress_block(&reference_compressed, block_size)
        .expect("reference block round-trip must work");

    for &num_blocks in &[1usize, 4, 8, 16, 32] {
        let total_size = block_size * num_blocks;
        let test_data = block_data.repeat(num_blocks);

        // Replicate the same compressed block N times.
        let compressed_blocks: Vec<Vec<u8>> = (0..num_blocks)
            .map(|_| reference_compressed.clone())
            .collect();

        // Report compressed size.
        let total_compressed: usize = compressed_blocks.iter().map(|b| b.len()).sum();
        let ratio = total_compressed as f64 / total_size as f64 * 100.0;
        eprintln!(
            "  [{}×128KB = {}KB] gpulz ratio: {ratio:.1}%  ({num_blocks} blocks)",
            num_blocks,
            total_size / 1024,
        );

        // CPU single-threaded decompress baseline (serial over all blocks).
        {
            let blocks_c = compressed_blocks.clone();
            group.throughput(Throughput::Bytes(total_size as u64));
            group.bench_with_input(
                BenchmarkId::new("cpu_serial", num_blocks),
                &test_data,
                move |b, _| {
                    b.iter(|| {
                        for bc in &blocks_c {
                            pz::gpulz::decompress_block(bc, block_size).unwrap();
                        }
                    });
                },
            );
        }

        // CPU multi-threaded decompress baseline (parallel over blocks).
        if num_blocks > 1 {
            let blocks_c = compressed_blocks.clone();
            group.throughput(Throughput::Bytes(total_size as u64));
            group.bench_with_input(
                BenchmarkId::new("cpu_parallel", num_blocks),
                &test_data,
                move |b, _| {
                    b.iter(|| {
                        std::thread::scope(|scope| {
                            let handles: Vec<_> = blocks_c
                                .iter()
                                .map(|bc| {
                                    scope.spawn(|| {
                                        pz::gpulz::decompress_block(bc, block_size).unwrap()
                                    })
                                })
                                .collect();
                            handles
                                .into_iter()
                                .map(|h| h.join().unwrap())
                                .collect::<Vec<_>>()
                        })
                    });
                },
            );
        }

        // GPU batched Huffman + parallel CPU LzSeq.
        {
            let blocks_c = compressed_blocks.clone();
            let eng = engine.clone();
            group.throughput(Throughput::Bytes(total_size as u64));
            group.bench_with_input(
                BenchmarkId::new("gpu_parallel", num_blocks),
                &test_data,
                move |b, _| {
                    b.iter(|| {
                        let block_refs: Vec<(&[u8], usize)> = blocks_c
                            .iter()
                            .map(|bc| (bc.as_slice(), block_size))
                            .collect();
                        pz::gpulz::decompress_blocks_gpu(&eng, &block_refs).unwrap()
                    });
                },
            );
        }

        // Report timing breakdown for this configuration.
        {
            let block_refs: Vec<(&[u8], usize)> = compressed_blocks
                .iter()
                .map(|bc| (bc.as_slice(), block_size))
                .collect();
            // Warmup.
            for _ in 0..5 {
                let _ = pz::gpulz::decompress_blocks_gpu(&engine, &block_refs).unwrap();
            }
            let mut timings = Vec::new();
            for _ in 0..10 {
                let (_, t) = pz::gpulz::decompress_blocks_gpu(&engine, &block_refs).unwrap();
                timings.push(t);
            }
            let n = timings.len();
            let mut total: Vec<u64> = timings.iter().map(|t| t.total_us).collect();
            let mut gpu: Vec<u64> = timings.iter().map(|t| t.gpu_huffman_us).collect();
            let mut gpu_buf: Vec<u64> = timings.iter().map(|t| t.gpu_buf_us).collect();
            let mut gpu_sub: Vec<u64> = timings.iter().map(|t| t.gpu_submit_us).collect();
            let mut gpu_rb: Vec<u64> = timings.iter().map(|t| t.gpu_readback_us).collect();
            let mut lzseq: Vec<u64> = timings.iter().map(|t| t.lzseq_us).collect();
            total.sort();
            gpu.sort();
            gpu_buf.sort();
            gpu_sub.sort();
            gpu_rb.sort();
            lzseq.sort();
            let med = n / 2;
            let throughput = total_size as f64 / total[med] as f64; // bytes/µs = MB/s
            eprintln!(
                "    → {num_blocks}×128KB: gpu={gpu}µs [buf={buf}µs sub={sub}µs rb={rb}µs]  lzseq={lzseq}µs  total={total}µs  = {tp:.0} MiB/s  ({streams} GPU streams)",
                gpu = gpu[med],
                buf = gpu_buf[med],
                sub = gpu_sub[med],
                rb = gpu_rb[med],
                lzseq = lzseq[med],
                total = total[med],
                tp = throughput / 1.048576,
                streams = timings[0].num_gpu_streams,
            );
        }
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rebuild_from_lengths(code_lengths: &[u8; 256]) -> pz::huffman::HuffmanTree {
    // Build canonical codes from lengths.
    let mut lookup = [(0u32, 0u8); 256];
    let mut bl_count = [0u32; 16];
    for &len in code_lengths.iter() {
        if len > 0 && (len as usize) < bl_count.len() {
            bl_count[len as usize] += 1;
        }
    }
    let mut next_code = [0u32; 16];
    let mut code = 0u32;
    for bits in 1..16 {
        code = (code + bl_count[bits - 1]) << 1;
        next_code[bits] = code;
    }
    for sym in 0..256 {
        let len = code_lengths[sym];
        if len > 0 && (len as usize) < next_code.len() {
            lookup[sym] = (next_code[len as usize], len);
            next_code[len as usize] += 1;
        }
    }
    let decode_table = pz::huffman::HuffmanTree::build_decode_table_from_lookup(&lookup);
    let leaf_count = code_lengths.iter().filter(|&&l| l > 0).count() as u32;

    let mut nodes = Vec::with_capacity(257);
    for i in 0..256u16 {
        nodes.push(pz::huffman::HuffmanNode {
            weight: 0,
            value: i as u8,
            codeword: lookup[i as usize].0,
            code_bits: lookup[i as usize].1,
            left: None,
            right: None,
        });
    }
    nodes.push(pz::huffman::HuffmanNode {
        weight: 0,
        value: 0,
        codeword: 0,
        code_bits: 0,
        left: Some(0),
        right: if leaf_count > 1 { Some(1) } else { None },
    });

    pz::huffman::HuffmanTree::from_parts(nodes, Some(256), lookup, leaf_count, decode_table)
}

#[cfg(feature = "webgpu")]
criterion_group!(
    benches,
    bench_huffman_encode_sync,
    bench_huffman_sync_ratio,
    bench_huffman_decode_tiled,
    bench_gpulz_block,
    bench_huffman_decode_gpu,
    bench_gpulz_decompress_gpu,
    bench_gpulz_gpu_timing,
    bench_gpulz_multiblock_gpu,
);

#[cfg(not(feature = "webgpu"))]
criterion_group!(
    benches,
    bench_huffman_encode_sync,
    bench_huffman_sync_ratio,
    bench_huffman_decode_tiled,
    bench_gpulz_block,
);

criterion_main!(benches);
