/// Benchmark Recoil parallel rANS decode: CPU vs GPU vs standard interleaved.
///
/// Two modes:
/// 1. Raw rANS decode (isolates entropy decode overhead)
/// 2. Full pipeline decode (end-to-end including LZ77 + rANS)
use std::time::Instant;

use pz::pipeline::{self, CompressOptions, Pipeline};

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn bench_decode_fn(
    label: &str,
    original_len: usize,
    iterations: usize,
    warmup: usize,
    decode: &dyn Fn() -> Vec<u8>,
) {
    for _ in 0..warmup {
        let _ = std::hint::black_box(decode());
    }

    let mut mbps_samples = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..iterations {
            let result = std::hint::black_box(decode());
            assert_eq!(result.len(), original_len);
        }
        let elapsed = start.elapsed();
        let mbps =
            (original_len as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
        mbps_samples.push(mbps);
    }

    let med = median(&mut mbps_samples);
    let min = mbps_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = mbps_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    eprintln!(
        "  {:<50} median {:>8.1} MB/s  (min {:>7.1}, max {:>7.1})",
        label, med, min, max,
    );
}

fn main() {
    let sizes: Vec<usize> = vec![65_536, 262_144, 1_048_576, 4_194_304];

    let iterations_for_size = |size: usize| -> usize {
        if size >= 4_000_000 {
            20
        } else if size >= 1_000_000 {
            50
        } else if size >= 200_000 {
            100
        } else {
            200
        }
    };

    let base_pattern =
        b"The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. ";
    let mut full_data = Vec::new();
    while full_data.len() < *sizes.last().unwrap() {
        full_data.extend_from_slice(base_pattern);
    }

    #[cfg(feature = "webgpu")]
    let gpu_engine = pz::webgpu::WebGpuEngine::new().ok();
    #[cfg(not(feature = "webgpu"))]
    let gpu_engine: Option<()> = None;

    eprintln!(
        "GPU: {}",
        if gpu_engine.is_some() {
            "available"
        } else {
            "not available"
        }
    );

    // =========================================================
    // Part 1: Raw rANS decode (entropy decoding only)
    // =========================================================
    eprintln!("\n========== RAW rANS DECODE (entropy only) ==========");

    for &size in &sizes {
        let data = &full_data[..size];
        let iterations = iterations_for_size(size);
        let warmup = (iterations / 4).max(2);

        eprintln!(
            "\n--- {} bytes ({:.0} KB), {} iterations ---",
            size,
            size as f64 / 1024.0,
            iterations
        );

        let encoded = pz::rans::encode_interleaved_n(data, 4, pz::rans::DEFAULT_SCALE_BITS);
        let enc_ref = &encoded;

        bench_decode_fn(
            "CPU interleaved (baseline)",
            size,
            iterations,
            warmup,
            &|| pz::rans::decode_interleaved(enc_ref, size).unwrap(),
        );

        #[cfg(feature = "webgpu")]
        if let Some(ref engine) = gpu_engine {
            bench_decode_fn(
                "GPU interleaved (1 chunk, no recoil)",
                size,
                iterations,
                warmup,
                &|| engine.rans_decode_interleaved_gpu(enc_ref, size).unwrap(),
            );
        }

        for num_splits in [4, 16, 64] {
            let metadata = pz::recoil::recoil_generate_splits(&encoded, size, num_splits).unwrap();
            let meta_ref = &metadata;

            bench_decode_fn(
                &format!("CPU Recoil ({} splits, sequential)", num_splits),
                size,
                iterations,
                warmup,
                &|| pz::recoil::decode_recoil(enc_ref, meta_ref, size).unwrap(),
            );

            #[cfg(feature = "webgpu")]
            if let Some(ref engine) = gpu_engine {
                bench_decode_fn(
                    &format!("GPU Recoil ({} splits)", num_splits),
                    size,
                    iterations,
                    warmup,
                    &|| {
                        engine
                            .rans_decode_recoil_gpu(enc_ref, meta_ref, size)
                            .unwrap()
                    },
                );
            }
        }
    }

    // =========================================================
    // Part 2: Full pipeline decode (LZR: LZ77 + rANS)
    // =========================================================
    eprintln!("\n========== FULL PIPELINE DECODE (LZR: LZ77 + rANS) ==========");

    #[cfg(feature = "webgpu")]
    let gpu_dec_opts = pipeline::DecompressOptions {
        backend: pipeline::Backend::WebGpu,
        webgpu_engine: gpu_engine
            .as_ref()
            .map(|_| std::sync::Arc::new(pz::webgpu::WebGpuEngine::new().unwrap())),
        ..Default::default()
    };

    for &size in &sizes {
        let data = &full_data[..size];
        let iterations = iterations_for_size(size);
        let warmup = (iterations / 4).max(2);

        eprintln!(
            "\n--- {} bytes ({:.0} KB), {} iterations ---",
            size,
            size as f64 / 1024.0,
            iterations
        );

        // Standard (no recoil)
        let opts_std = CompressOptions {
            rans_interleaved: true,
            rans_interleaved_min_bytes: 0,
            rans_interleaved_states: 4,
            rans_recoil: false,
            ..Default::default()
        };
        let compressed_std =
            pipeline::compress_with_options(data, Pipeline::Lzr, &opts_std).unwrap();
        let cref = &compressed_std;
        eprintln!(
            "    compressed: {:.1} KB (ratio {:.1}%)",
            compressed_std.len() as f64 / 1024.0,
            compressed_std.len() as f64 / size as f64 * 100.0
        );

        bench_decode_fn(
            "LZR standard (no recoil)",
            size,
            iterations,
            warmup,
            &|| pipeline::decompress(cref).unwrap(),
        );

        // With recoil (CPU decode via pipeline)
        let opts_recoil = CompressOptions {
            rans_interleaved: true,
            rans_interleaved_min_bytes: 0,
            rans_interleaved_states: 4,
            rans_recoil: true,
            rans_recoil_splits: 16,
            ..Default::default()
        };
        let compressed_recoil =
            pipeline::compress_with_options(data, Pipeline::Lzr, &opts_recoil).unwrap();
        let cref_r = &compressed_recoil;
        bench_decode_fn(
            "LZR + Recoil (16 splits, CPU thread::scope)",
            size,
            iterations,
            warmup,
            &|| pipeline::decompress(cref_r).unwrap(),
        );

        // With recoil (GPU decode via pipeline)
        #[cfg(feature = "webgpu")]
        if gpu_engine.is_some() {
            let dec_ref = &gpu_dec_opts;
            bench_decode_fn(
                "LZR + Recoil (16 splits, GPU decode)",
                size,
                iterations,
                warmup,
                &|| pipeline::decompress_with_options(cref_r, dec_ref).unwrap(),
            );
        }
    }
}
