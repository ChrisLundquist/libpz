#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

const SIZES_ALL: &[usize] = &[8192, 65536, 4_194_304];

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

        let tree = pz::huffman::HuffmanTree::from_data(&data).unwrap();
        let mut code_lut = [0u32; 256];
        for byte in 0..=255u8 {
            let (codeword, bits) = tree.get_code(byte);
            code_lut[byte as usize] = ((bits as u32) << 24) | codeword;
        }

        let tree_clone = tree.clone();
        group.bench_with_input(
            BenchmarkId::new("encode_cpu", size),
            &data,
            move |b, data| {
                b.iter(|| tree_clone.encode(data).unwrap());
            },
        );

        let eng = engine.clone();
        let lut = code_lut;
        group.bench_with_input(
            BenchmarkId::new("encode_webgpu", size),
            &data,
            move |b, data| {
                b.iter(|| eng.huffman_encode(data, &lut).unwrap());
            },
        );

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

criterion_group!(benches, bench_huffman, bench_huffman_webgpu);
criterion_main!(benches);
