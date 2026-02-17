/// Profiling harness for samply / Instruments / perf.
///
/// Runs a pipeline compress/decompress loop so profilers can collect samples.
///
/// Usage:
///   cargo build --profile profiling --example profile
///   samply record ./target/profiling/examples/profile --pipeline lzf
///   samply record ./target/profiling/examples/profile --pipeline deflate --decompress
///   samply record ./target/profiling/examples/profile --stage lz77
///   samply record ./target/profiling/examples/profile --stage fse --size 1048576
use std::path::Path;
use std::time::Instant;

use pz::pipeline::{self, CompressOptions, Pipeline};

fn usage() {
    eprintln!("profile - run compression in a loop for profiling");
    eprintln!();
    eprintln!("Usage: profile [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --pipeline P    Pipeline: deflate, bw, bbw, lzr, lzf, lzfi, lzssr (default: lzf)");
    eprintln!("  --stage S       Profile a single stage instead of full pipeline:");
    eprintln!("                    lz77, huffman, bwt, mtf, rle, fse, rans");
    eprintln!("  --decompress    Profile decompression instead of compression");
    eprintln!("  --iterations N  Number of iterations (default: 200)");
    eprintln!("  --size N        Input data size in bytes (default: 262144)");
    eprintln!(
        "  --rans-interleaved            Enable interleaved rANS for pipeline and rANS stage"
    );
    eprintln!(
        "  --rans-interleaved-min-bytes  Min stream bytes for interleaved rANS (default: 65536)"
    );
    eprintln!("  --rans-interleaved-states N   Interleaved rANS state count (default: 4)");
    eprintln!("  --rans-chunked                Enable chunked interleaved rANS");
    eprintln!(
        "  --rans-chunked-min-bytes N    Min stream bytes for chunked interleaved rANS (default: 262144)"
    );
    eprintln!(
        "  --rans-chunk-bytes N          Chunk size for chunked interleaved rANS (default: {})",
        pz::rans::DEFAULT_CHUNK_BYTES
    );
    eprintln!("  --unified-scheduler           Enable prototype mixed-task scheduler");
    eprintln!("  --help          Show this help");
}

#[derive(Clone, Copy)]
struct RansProfileOptions {
    interleaved: bool,
    interleaved_states: usize,
    chunked: bool,
    chunk_bytes: usize,
}

fn load_data(size: usize) -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

    // Try Canterbury corpus
    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                if decompressed.len() >= size {
                    return decompressed[..size].to_vec();
                }
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

    // Fallback
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let full = pattern.repeat((size / pattern.len()) + 1);
    full[..size].to_vec()
}

fn profile_pipeline(
    data: &[u8],
    pipe: Pipeline,
    decompress: bool,
    iterations: usize,
    opts: &CompressOptions,
) {
    if decompress {
        let compressed = pipeline::compress_with_options(data, pipe, opts).unwrap();
        eprintln!(
            "profiling {:?} decompress: {} bytes compressed -> {} bytes, {} iterations",
            pipe,
            compressed.len(),
            data.len(),
            iterations
        );
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = std::hint::black_box(pipeline::decompress(&compressed).unwrap());
        }
        let elapsed = start.elapsed();
        let mbps =
            (data.len() as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!("done: {:.1}s, {:.1} MB/s", elapsed.as_secs_f64(), mbps);
    } else {
        eprintln!(
            "profiling {:?} compress: {} bytes, {} iterations",
            pipe,
            data.len(),
            iterations
        );
        let start = Instant::now();
        for _ in 0..iterations {
            let _ =
                std::hint::black_box(pipeline::compress_with_options(data, pipe, opts).unwrap());
        }
        let elapsed = start.elapsed();
        let mbps =
            (data.len() as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!("done: {:.1}s, {:.1} MB/s", elapsed.as_secs_f64(), mbps);
    }
}

fn profile_stage(
    data: &[u8],
    stage: &str,
    decompress: bool,
    iterations: usize,
    rans: RansProfileOptions,
) {
    eprintln!(
        "profiling stage {} {}: {} bytes, {} iterations",
        stage,
        if decompress { "decode" } else { "encode" },
        data.len(),
        iterations
    );
    let start = Instant::now();
    match (stage, decompress) {
        ("lz77", false) => {
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::lz77::compress_lazy(data).unwrap());
            }
        }
        ("lz77", true) => {
            let compressed = pz::lz77::compress_lazy(data).unwrap();
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::lz77::decompress(&compressed).unwrap());
            }
        }
        ("huffman", false) => {
            let tree = pz::huffman::HuffmanTree::from_data(data).unwrap();
            for _ in 0..iterations {
                let _ = std::hint::black_box(tree.encode(data).unwrap());
            }
        }
        ("huffman", true) => {
            let tree = pz::huffman::HuffmanTree::from_data(data).unwrap();
            let (encoded, total_bits) = tree.encode(data).unwrap();
            let mut out = vec![0u8; data.len()];
            for _ in 0..iterations {
                let _ = std::hint::black_box(
                    tree.decode_to_buf(&encoded, total_bits, &mut out).unwrap(),
                );
            }
        }
        ("bwt", false) => {
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::bwt::encode(data).unwrap());
            }
        }
        ("bwt", true) => {
            let enc = pz::bwt::encode(data).unwrap();
            for _ in 0..iterations {
                let _ =
                    std::hint::black_box(pz::bwt::decode(&enc.data, enc.primary_index).unwrap());
            }
        }
        ("mtf", false) => {
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::mtf::encode(data));
            }
        }
        ("mtf", true) => {
            let enc = pz::mtf::encode(data);
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::mtf::decode(&enc));
            }
        }
        ("rle", false) => {
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::rle::encode(data));
            }
        }
        ("rle", true) => {
            let enc = pz::rle::encode(data);
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::rle::decode(&enc).unwrap());
            }
        }
        ("fse", false) => {
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::fse::encode(data));
            }
        }
        ("fse", true) => {
            let enc = pz::fse::encode(data);
            let len = data.len();
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::fse::decode(&enc, len).unwrap());
            }
        }
        ("rans", false) => {
            if !profile_rans_stage_gpu(data, false, iterations, rans) {
                for _ in 0..iterations {
                    if rans.chunked {
                        let _ = std::hint::black_box(
                            pz::rans::encode_chunked_n(
                                data,
                                rans.interleaved_states,
                                pz::rans::DEFAULT_SCALE_BITS,
                                rans.chunk_bytes,
                            )
                            .0,
                        );
                    } else if rans.interleaved {
                        let _ = std::hint::black_box(pz::rans::encode_interleaved_n(
                            data,
                            rans.interleaved_states,
                            pz::rans::DEFAULT_SCALE_BITS,
                        ));
                    } else {
                        let _ = std::hint::black_box(pz::rans::encode(data));
                    }
                }
            }
        }
        ("rans", true) => {
            if !profile_rans_stage_gpu(data, true, iterations, rans) {
                let (enc, encoded_chunked) = if rans.chunked {
                    pz::rans::encode_chunked_n(
                        data,
                        rans.interleaved_states,
                        pz::rans::DEFAULT_SCALE_BITS,
                        rans.chunk_bytes,
                    )
                } else if rans.interleaved {
                    (
                        pz::rans::encode_interleaved_n(
                            data,
                            rans.interleaved_states,
                            pz::rans::DEFAULT_SCALE_BITS,
                        ),
                        false,
                    )
                } else {
                    (pz::rans::encode(data), false)
                };
                let len = data.len();
                for _ in 0..iterations {
                    if encoded_chunked {
                        let _ = std::hint::black_box(pz::rans::decode_chunked(&enc, len).unwrap());
                    } else if rans.interleaved {
                        let _ =
                            std::hint::black_box(pz::rans::decode_interleaved(&enc, len).unwrap());
                    } else {
                        let _ = std::hint::black_box(pz::rans::decode(&enc, len).unwrap());
                    }
                }
            }
        }
        _ => {
            eprintln!("unknown stage: {}", stage);
            eprintln!("valid stages: lz77, huffman, bwt, mtf, rle, fse, rans");
            std::process::exit(1);
        }
    }
    let elapsed = start.elapsed();
    let mbps = (data.len() as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
    eprintln!("done: {:.1}s, {:.1} MB/s", elapsed.as_secs_f64(), mbps);
}

#[cfg(feature = "webgpu")]
fn profile_rans_stage_gpu(
    data: &[u8],
    decompress: bool,
    iterations: usize,
    rans: RansProfileOptions,
) -> bool {
    let engine = match pz::webgpu::WebGpuEngine::new() {
        Ok(e) => e,
        Err(_) => return false,
    };

    let num_lanes = if rans.interleaved {
        rans.interleaved_states.max(1)
    } else {
        pz::rans::DEFAULT_INTERLEAVE
    };
    let chunk_bytes = if rans.chunked {
        rans.chunk_bytes
    } else {
        pz::rans::DEFAULT_CHUNK_BYTES
    };
    let scale_bits = pz::rans::DEFAULT_SCALE_BITS;

    eprintln!(
        "using webgpu chunked rANS path (lanes={}, chunk={})",
        num_lanes, chunk_bytes
    );

    if decompress {
        let (enc, used_chunked) = match engine.rans_encode_chunked_payload_gpu(
            data,
            num_lanes,
            scale_bits,
            chunk_bytes,
        ) {
            Ok(v) => v,
            Err(_) => return false,
        };
        if !used_chunked {
            return false;
        }
        let len = data.len();
        for _ in 0..iterations {
            let _ =
                std::hint::black_box(engine.rans_decode_chunked_payload_gpu(&enc, len).unwrap());
        }
    } else {
        // Use a small batch so the GPU path can overlap submit/readback.
        const GPU_BATCH: usize = 8;
        let batch_inputs: Vec<&[u8]> = vec![data; GPU_BATCH];
        let full_batches = iterations / GPU_BATCH;
        for _ in 0..full_batches {
            let _ = std::hint::black_box(
                engine
                    .rans_encode_chunked_payload_gpu_batched(
                        &batch_inputs,
                        num_lanes,
                        scale_bits,
                        chunk_bytes,
                    )
                    .unwrap(),
            );
        }
        for _ in 0..(iterations % GPU_BATCH) {
            let _ = std::hint::black_box(
                engine
                    .rans_encode_chunked_payload_gpu(data, num_lanes, scale_bits, chunk_bytes)
                    .unwrap(),
            );
        }
    }
    true
}

#[cfg(not(feature = "webgpu"))]
fn profile_rans_stage_gpu(
    _data: &[u8],
    _decompress: bool,
    _iterations: usize,
    _rans: RansProfileOptions,
) -> bool {
    false
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pipeline_name = "lzf".to_string();
    let mut stage: Option<String> = None;
    let mut decompress = false;
    let mut iterations = 200usize;
    let mut size = 262_144usize;
    let mut rans_interleaved = false;
    let mut rans_interleaved_min_bytes = 65_536usize;
    let mut rans_interleaved_states = 4usize;
    let mut rans_chunked = false;
    let mut rans_chunked_min_bytes = 262_144usize;
    let mut rans_chunk_bytes = pz::rans::DEFAULT_CHUNK_BYTES;
    let mut unified_scheduler = false;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--pipeline" | "-p" => {
                i += 1;
                pipeline_name = args[i].clone();
            }
            "--stage" | "-s" => {
                i += 1;
                stage = Some(args[i].clone());
            }
            "--decompress" | "-d" => decompress = true,
            "--iterations" | "-n" => {
                i += 1;
                iterations = args[i].parse().expect("invalid iterations");
            }
            "--size" => {
                i += 1;
                size = args[i].parse().expect("invalid size");
            }
            "--rans-interleaved" => rans_interleaved = true,
            "--rans-interleaved-min-bytes" => {
                i += 1;
                rans_interleaved_min_bytes = args[i]
                    .parse()
                    .expect("invalid --rans-interleaved-min-bytes");
            }
            "--rans-interleaved-states" => {
                i += 1;
                rans_interleaved_states =
                    args[i].parse().expect("invalid --rans-interleaved-states");
            }
            "--rans-chunked" => rans_chunked = true,
            "--rans-chunked-min-bytes" => {
                i += 1;
                rans_chunked_min_bytes = args[i].parse().expect("invalid --rans-chunked-min-bytes");
            }
            "--rans-chunk-bytes" => {
                i += 1;
                rans_chunk_bytes = args[i].parse().expect("invalid --rans-chunk-bytes");
                if rans_chunk_bytes == 0 {
                    panic!("--rans-chunk-bytes must be > 0");
                }
            }
            "--unified-scheduler" => unified_scheduler = true,
            "--help" | "-h" => {
                usage();
                return;
            }
            other => {
                eprintln!("unknown option: {}", other);
                usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let data = load_data(size);
    let rans_profile_opts = RansProfileOptions {
        interleaved: rans_interleaved,
        interleaved_states: rans_interleaved_states,
        chunked: rans_chunked,
        chunk_bytes: rans_chunk_bytes,
    };

    if let Some(ref stage_name) = stage {
        profile_stage(&data, stage_name, decompress, iterations, rans_profile_opts);
    } else {
        let pipe = match pipeline_name.as_str() {
            "deflate" => Pipeline::Deflate,
            "bw" => Pipeline::Bw,
            "bbw" => Pipeline::Bbw,
            "lzr" => Pipeline::Lzr,
            "lzf" => Pipeline::Lzf,
            "lzfi" => Pipeline::Lzfi,
            "lzssr" => Pipeline::LzssR,
            other => {
                eprintln!("unknown pipeline: {}", other);
                eprintln!("valid pipelines: deflate, bw, bbw, lzr, lzf, lzfi, lzssr");
                std::process::exit(1);
            }
        };

        // Warm up once
        let opts = CompressOptions {
            rans_interleaved,
            rans_interleaved_min_bytes,
            rans_interleaved_states,
            rans_chunked,
            rans_chunked_min_bytes,
            rans_chunk_bytes,
            unified_scheduler,
            ..CompressOptions::default()
        };
        let _ = pipeline::compress_with_options(&data, pipe, &opts).unwrap();

        profile_pipeline(&data, pipe, decompress, iterations, &opts);
    }
}
