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
    eprintln!("  --pipeline P    Pipeline: deflate, bw, lzr, lzf (default: lzf)");
    eprintln!("  --stage S       Profile a single stage instead of full pipeline:");
    eprintln!("                    lz77, huffman, bwt, mtf, rle, fse, rans");
    eprintln!("  --decompress    Profile decompression instead of compression");
    eprintln!("  --iterations N  Number of iterations (default: 200)");
    eprintln!("  --size N        Input data size in bytes (default: 262144)");
    eprintln!("  --help          Show this help");
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

fn profile_pipeline(data: &[u8], pipe: Pipeline, decompress: bool, iterations: usize) {
    if decompress {
        let compressed = pipeline::compress(data, pipe).unwrap();
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
            let _ = std::hint::black_box(pipeline::compress(data, pipe).unwrap());
        }
        let elapsed = start.elapsed();
        let mbps =
            (data.len() as f64 * iterations as f64) / elapsed.as_secs_f64() / (1024.0 * 1024.0);
        eprintln!("done: {:.1}s, {:.1} MB/s", elapsed.as_secs_f64(), mbps);
    }
}

fn profile_stage(data: &[u8], stage: &str, decompress: bool, iterations: usize) {
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
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::rans::encode(data));
            }
        }
        ("rans", true) => {
            let enc = pz::rans::encode(data);
            let len = data.len();
            for _ in 0..iterations {
                let _ = std::hint::black_box(pz::rans::decode(&enc, len).unwrap());
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pipeline_name = "lzf".to_string();
    let mut stage: Option<String> = None;
    let mut decompress = false;
    let mut iterations = 200usize;
    let mut size = 262_144usize;

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

    if let Some(ref stage_name) = stage {
        profile_stage(&data, stage_name, decompress, iterations);
    } else {
        let pipe = match pipeline_name.as_str() {
            "deflate" => Pipeline::Deflate,
            "bw" => Pipeline::Bw,
            "bbw" => Pipeline::Bbw,
            "lzr" => Pipeline::Lzr,
            "lzf" => Pipeline::Lzf,
            other => {
                eprintln!("unknown pipeline: {}", other);
                eprintln!("valid pipelines: deflate, bw, bbw, lzr, lzf");
                std::process::exit(1);
            }
        };

        // Warm up once
        let opts = CompressOptions::default();
        let _ = pipeline::compress_with_options(&data, pipe, &opts).unwrap();

        profile_pipeline(&data, pipe, decompress, iterations);
    }
}
