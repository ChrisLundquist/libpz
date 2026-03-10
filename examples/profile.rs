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
use std::time::{Duration, Instant};

use pz::pipeline::{self, CompressOptions, Pipeline};

const DEFAULT_RANS_CHUNK_BYTES: usize = 8192;
#[cfg(feature = "webgpu")]
const DEFAULT_RANS_GPU_CHUNK_BYTES: usize = 2048;
const DEFAULT_RANS_GPU_BATCH: usize = 6;

fn usage() {
    eprintln!("profile - run compression in a loop for profiling");
    eprintln!();
    eprintln!("Usage: profile [OPTIONS]");
    eprintln!();
    eprintln!("Options:");
    eprintln!(
        "  --pipeline P    Pipeline: deflate, bw, bbw, lzf, lzfi, lzssr, lzseqr (default: lzf)"
    );
    eprintln!("  --stage S       Profile a single stage instead of full pipeline:");
    eprintln!("                    lz77, huffman, bwt, mtf, rle, fse, rans");
    eprintln!("  --decompress    Profile decompression instead of compression");
    eprintln!("  --gpu           Use WebGPU backend for full-pipeline profiling");
    eprintln!("  --threads N     Threads for pipeline mode (0=auto, default: 0)");
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
        DEFAULT_RANS_CHUNK_BYTES
    );
    eprintln!(
        "  --rans-gpu-batch N            Batch size for GPU rANS profiling (default: {})",
        DEFAULT_RANS_GPU_BATCH
    );
    eprintln!(
        "  --rans-independent-block-bytes N  Split input into independent blocks for nvCOMP-style stage profiling (default: disabled)"
    );
    eprintln!("  --print-scheduler-stats       Print unified scheduler telemetry after run");
    eprintln!("  --help          Show this help");
}

fn print_profile_stats(mbps: f64, elapsed: Duration) {
    eprintln!(
        "PROFILE_STATS\tmbps={:.6}\telapsed_ns={}",
        mbps,
        elapsed.as_nanos()
    );
}

#[derive(Clone, Copy)]
struct RansProfileOptions {
    interleaved: bool,
    interleaved_states: usize,
    chunked: bool,
    chunked_min_bytes: usize,
    chunk_bytes: usize,
    #[cfg_attr(not(feature = "webgpu"), allow(dead_code))]
    gpu_batch: usize,
    #[cfg_attr(not(feature = "webgpu"), allow(dead_code))]
    independent_block_bytes: usize,
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

#[cfg(feature = "webgpu")]
fn split_independent_blocks(data: &[u8], block_bytes: usize) -> Vec<&[u8]> {
    if block_bytes == 0 || block_bytes >= data.len() {
        return vec![data];
    }
    data.chunks(block_bytes).collect()
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
            let _ = std::hint::black_box(
                pipeline::decompress_with_threads(&compressed, opts.threads).unwrap(),
            );
        }
        let elapsed = start.elapsed();
        let mbps =
            (data.len() as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!("done: {:.1}s, {:.1} MB/s", elapsed.as_secs_f64(), mbps);
        print_profile_stats(mbps, elapsed);
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
        print_profile_stats(mbps, elapsed);
    }
}

fn profile_stage(
    data: &[u8],
    stage: &str,
    decompress: bool,
    iterations: usize,
    rans: RansProfileOptions,
) {
    let chunked_enabled = rans.chunked && data.len() >= rans.chunked_min_bytes;
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
                    if chunked_enabled {
                        let _ = std::hint::black_box(pz::rans::encode_chunked(
                            data,
                            rans.interleaved_states,
                            pz::rans::DEFAULT_SCALE_BITS,
                            rans.chunk_bytes,
                        ));
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
                let (enc, encoded_chunked) = if chunked_enabled {
                    (
                        pz::rans::encode_chunked(
                            data,
                            rans.interleaved_states,
                            pz::rans::DEFAULT_SCALE_BITS,
                            rans.chunk_bytes,
                        ),
                        true,
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
                        let _ = std::hint::black_box(pz::rans::decode_chunked(&enc).unwrap());
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
    print_profile_stats(mbps, elapsed);
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
        DEFAULT_RANS_GPU_CHUNK_BYTES
    };
    let scale_bits = pz::rans::DEFAULT_SCALE_BITS;

    eprintln!(
        "using webgpu chunked rANS path (lanes={}, chunk={})",
        num_lanes, chunk_bytes
    );
    let independent_blocks = split_independent_blocks(data, rans.independent_block_bytes);
    let use_independent_blocks = independent_blocks.len() > 1;
    if use_independent_blocks {
        eprintln!(
            "using independent-block batch split (blocks={}, block_bytes={})",
            independent_blocks.len(),
            rans.independent_block_bytes
        );
        eprintln!("using shared frequency-table seed from full input");
    }

    if decompress {
        if use_independent_blocks {
            let encoded_blocks = match engine.rans_encode_chunked_payload_gpu_batched_shared_table(
                &independent_blocks,
                data,
                num_lanes,
                scale_bits,
                chunk_bytes,
            ) {
                Ok(v) => v,
                Err(_) => return false,
            };
            if encoded_blocks
                .iter()
                .any(|(_, used_chunked)| !*used_chunked)
            {
                return false;
            }

            let decode_inputs: Vec<(&[u8], usize)> = encoded_blocks
                .iter()
                .zip(independent_blocks.iter())
                .map(|((payload, _), block)| (payload.as_slice(), block.len()))
                .collect();
            let _ = std::hint::black_box(
                engine
                    .rans_decode_chunked_payload_gpu_batched_shared_table_repeated(
                        &decode_inputs,
                        data,
                        iterations,
                    )
                    .unwrap(),
            );
        } else {
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
            let gpu_batch = rans.gpu_batch.max(1);
            let batch_inputs: Vec<(&[u8], usize)> = vec![(enc.as_slice(), len); gpu_batch];
            let full_batches = iterations / gpu_batch;
            for _ in 0..full_batches {
                let _ = std::hint::black_box(
                    engine
                        .rans_decode_chunked_payload_gpu_batched(&batch_inputs)
                        .unwrap(),
                );
            }
            for _ in 0..(iterations % gpu_batch) {
                let _ = std::hint::black_box(
                    engine.rans_decode_chunked_payload_gpu(&enc, len).unwrap(),
                );
            }
        }
    } else if use_independent_blocks {
        for _ in 0..iterations {
            let _ = std::hint::black_box(
                engine
                    .rans_encode_chunked_payload_gpu_batched_shared_table(
                        &independent_blocks,
                        data,
                        num_lanes,
                        scale_bits,
                        chunk_bytes,
                    )
                    .unwrap(),
            );
        }
    } else {
        // Use a small batch so the GPU path can overlap submit/readback.
        let gpu_batch = rans.gpu_batch.max(1);
        let batch_inputs: Vec<&[u8]> = vec![data; gpu_batch];
        let full_batches = iterations / gpu_batch;
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
        for _ in 0..(iterations % gpu_batch) {
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
    let mut use_gpu = false;
    let mut threads = 0usize;
    let mut iterations = 200usize;
    let mut size = 262_144usize;
    let mut rans_interleaved = false;
    let mut rans_interleaved_min_bytes = 65_536usize;
    let mut rans_interleaved_states = 4usize;
    let mut rans_chunked = false;
    let mut rans_chunked_min_bytes = 262_144usize;
    let mut rans_chunk_bytes = DEFAULT_RANS_CHUNK_BYTES;
    let mut rans_gpu_batch = DEFAULT_RANS_GPU_BATCH;
    let mut rans_independent_block_bytes = 0usize;
    let mut print_scheduler_stats = false;

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
            "--gpu" => use_gpu = true,
            "--threads" | "-t" => {
                i += 1;
                threads = args[i].parse().expect("invalid threads");
            }
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
            "--rans-gpu-batch" => {
                i += 1;
                rans_gpu_batch = args[i].parse().expect("invalid --rans-gpu-batch");
                if rans_gpu_batch == 0 {
                    panic!("--rans-gpu-batch must be > 0");
                }
            }
            "--rans-independent-block-bytes" => {
                i += 1;
                rans_independent_block_bytes = args[i]
                    .parse()
                    .expect("invalid --rans-independent-block-bytes");
            }
            "--print-scheduler-stats" => {
                print_scheduler_stats = true;
            }
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
        chunked_min_bytes: rans_chunked_min_bytes,
        chunk_bytes: rans_chunk_bytes,
        gpu_batch: rans_gpu_batch,
        independent_block_bytes: rans_independent_block_bytes,
    };

    pipeline::set_unified_scheduler_stats_enabled(print_scheduler_stats);
    if print_scheduler_stats {
        pipeline::reset_unified_scheduler_stats();
    }

    if let Some(ref stage_name) = stage {
        if use_gpu {
            eprintln!(
                "note: --gpu applies to full-pipeline mode; stage mode uses stage-specific paths"
            );
        }
        if threads != 0 {
            eprintln!("note: --threads applies to full-pipeline mode; stage mode ignores it");
        }
        profile_stage(&data, stage_name, decompress, iterations, rans_profile_opts);
    } else {
        let pipe = match pipeline_name.as_str() {
            "deflate" => Pipeline::Deflate,
            "bw" => Pipeline::Bw,
            "bbw" => Pipeline::Bbw,
            "lzf" => Pipeline::Lzf,
            "lzfi" => Pipeline::Lzfi,
            "lzssr" => Pipeline::LzssR,
            "lzseqr" => Pipeline::LzSeqR,
            other => {
                eprintln!("unknown pipeline: {}", other);
                eprintln!("valid pipelines: deflate, bw, bbw, lzf, lzfi, lzssr, lzseqr");
                std::process::exit(1);
            }
        };

        // Warm up once
        #[allow(unused_mut)]
        let mut opts = CompressOptions {
            threads,
            rans_interleaved,
            rans_interleaved_min_bytes,
            rans_interleaved_states,
            ..CompressOptions::default()
        };
        #[cfg(feature = "webgpu")]
        if use_gpu {
            let engine = match pz::webgpu::WebGpuEngine::new() {
                Ok(e) => e,
                Err(e) => {
                    eprintln!("webgpu requested but unavailable: {e}");
                    std::process::exit(2);
                }
            };
            eprintln!("using webgpu device: {}", engine.device_name());
            opts.backend = pz::pipeline::Backend::WebGpu;
            opts.webgpu_engine = Some(std::sync::Arc::new(engine));
        }
        #[cfg(not(feature = "webgpu"))]
        if use_gpu {
            eprintln!("webgpu requested but this binary was built without `webgpu` feature");
            std::process::exit(2);
        }
        let _ = pipeline::compress_with_options(&data, pipe, &opts).unwrap();
        if print_scheduler_stats {
            pipeline::reset_unified_scheduler_stats();
        }

        profile_pipeline(&data, pipe, decompress, iterations, &opts);
    }

    if print_scheduler_stats {
        let stats = pipeline::unified_scheduler_stats();
        println!(
            "SCHEDULER_STATS\truns={}\ttotal_ns={}\ttracked_thread_time_ns={}\tstage_compute_ns={}\tqueue_wait_ns={}\tqueue_admin_ns={}\tgpu_handoff_ns={}\tgpu_try_send_full_count={}\tgpu_try_send_disconnected_count={}\tscheduler_overhead_ns={}\tscheduler_overhead_pct={:.6}",
            stats.runs,
            stats.total_ns,
            stats.tracked_thread_time_ns(),
            stats.stage_compute_ns,
            stats.queue_wait_ns,
            stats.queue_admin_ns,
            stats.gpu_handoff_ns,
            stats.gpu_try_send_full_count,
            stats.gpu_try_send_disconnected_count,
            stats.scheduler_overhead_ns(),
            stats.scheduler_overhead_pct()
        );
    }
}
