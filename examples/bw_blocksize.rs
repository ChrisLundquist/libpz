// Sweep the BW pipeline block size: compression ratio + decode throughput.
//
// This harness produced docs/design-docs/bw-blocksize-findings.md (2026-06):
// ratio is block-size-monotonic but the blob saturates at 2M on a
// zstd-12-dominated point, and aggregate decode scaling collapses with block
// size (concurrent inverse-BWT working sets outgrow shared cache), so the
// default landed at 1 MiB. Decode is measured both single-threaded (per-core
// cost) and all-cores (the parallel-decode moat: bigger blocks = fewer work
// units AND more cache pressure per worker).
//
// Usage: cargo run --release --no-default-features --example bw_blocksize -- files...

use std::time::Instant;

use pz::pipeline::{self, CompressOptions, Pipeline};

fn sweep(data: &[u8], block_size: usize) {
    let opts = CompressOptions {
        threads: 0, // ratio is thread-independent; use all cores to sweep fast
        block_size,
        ..Default::default()
    };
    let t0 = Instant::now();
    let compressed = pipeline::compress_with_options(data, Pipeline::Bw, &opts).unwrap();
    let enc_s = t0.elapsed().as_secs_f64();

    // Single-thread decode (per-core cost).
    let t1 = Instant::now();
    let out = pipeline::decompress_with_threads(&compressed, 1).unwrap();
    let dec1_s = t1.elapsed().as_secs_f64();
    assert_eq!(out, data, "round-trip mismatch at block={block_size}");

    // All-cores decode (the parallel moat).
    let t2 = Instant::now();
    let out = pipeline::decompress_with_threads(&compressed, 0).unwrap();
    let dec_all_s = t2.elapsed().as_secs_f64();
    assert_eq!(out, data);

    let mb = data.len() as f64 / 1048576.0;
    println!(
        "  block={:>5}KB  ratio={:6.3}%  enc(all)={:6.1} MB/s  dec(1t)={:6.1} MB/s  dec(all)={:7.1} MB/s",
        block_size / 1024,
        100.0 * compressed.len() as f64 / data.len() as f64,
        mb / enc_s,
        mb / dec1_s,
        mb / dec_all_s
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: bw_blocksize <files...>");
        std::process::exit(2);
    }
    let block_sizes = [
        256 * 1024,
        512 * 1024, // shipped default
        // NOT 1 MiB exactly: adjusted_options can't distinguish an explicit
        // 1 MiB (== DEFAULT_BLOCK_SIZE) from "unset" and substitutes
        // DEFAULT_BW_BLOCK_SIZE. Harmless now that both are 1 MiB, but +4K
        // keeps this row honest if the defaults ever diverge again.
        1024 * 1024 + 4096,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
    ];
    for path in &args {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path);
        println!("{} ({:.1} MB):", name, data.len() as f64 / 1048576.0);
        for &bs in &block_sizes {
            sweep(&data, bs);
        }
        println!();
    }
}
