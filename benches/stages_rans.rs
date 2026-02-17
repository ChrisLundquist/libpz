#[path = "stages_common.rs"]
mod stages_common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use stages_common::{cap, get_test_data};

const SIZES_ALL: &[usize] = &[8192, 65536, 4_194_304];

fn bench_rans(c: &mut Criterion) {
    #[cfg(feature = "webgpu")]
    let gpu_engine = pz::webgpu::WebGpuEngine::new().ok();

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

        let (chunked_cpu, used_chunked_cpu) = pz::rans::encode_chunked_n(
            &data,
            pz::rans::DEFAULT_INTERLEAVE,
            pz::rans::DEFAULT_SCALE_BITS,
            pz::rans::DEFAULT_CHUNK_BYTES,
        );
        if used_chunked_cpu {
            group.bench_with_input(
                BenchmarkId::new("encode_chunked_cpu", size),
                &data,
                |b, data| {
                    b.iter(|| {
                        pz::rans::encode_chunked_n(
                            data,
                            pz::rans::DEFAULT_INTERLEAVE,
                            pz::rans::DEFAULT_SCALE_BITS,
                            pz::rans::DEFAULT_CHUNK_BYTES,
                        )
                    });
                },
            );

            group.bench_with_input(
                BenchmarkId::new("decode_chunked_cpu", size),
                &chunked_cpu,
                |b, enc| {
                    b.iter(|| pz::rans::decode_chunked(enc, size).unwrap());
                },
            );
        }

        #[cfg(feature = "webgpu")]
        if let Some(engine) = &gpu_engine {
            let (gpu_encoded, used_chunked_gpu) = engine
                .rans_encode_chunked_payload_gpu(
                    &data,
                    pz::rans::DEFAULT_INTERLEAVE,
                    pz::rans::DEFAULT_SCALE_BITS,
                    pz::rans::DEFAULT_CHUNK_BYTES,
                )
                .expect("gpu rans encode");
            if used_chunked_gpu {
                group.bench_with_input(
                    BenchmarkId::new("encode_chunked_gpu", size),
                    &data,
                    |b, data| {
                        b.iter(|| {
                            engine
                                .rans_encode_chunked_payload_gpu(
                                    data,
                                    pz::rans::DEFAULT_INTERLEAVE,
                                    pz::rans::DEFAULT_SCALE_BITS,
                                    pz::rans::DEFAULT_CHUNK_BYTES,
                                )
                                .unwrap()
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new("decode_chunked_gpu", size),
                    &gpu_encoded,
                    |b, enc| {
                        b.iter(|| engine.rans_decode_chunked_payload_gpu(enc, size).unwrap());
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_rans);
criterion_main!(benches);
