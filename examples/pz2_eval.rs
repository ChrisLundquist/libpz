//! pz2 prototype gate evaluation (clean-slate-codec.md §5).
//!
//! Per file: split into 1 MiB blocks, pz2-encode each, then measure
//! single-thread decode throughput (best of 3 passes over all blocks,
//! round-trip-verified). Baseline: the shipped Lzf pipeline on the same data
//! (threads=1 both directions — the honest per-core comparison; same parse).
//!
//! Gate: pz2 ST decode ≥ 2× Lzf ST decode at ratio within ~1pp.
//!
//! Usage: cargo run --release --no-default-features --example pz2_eval -- files...

use std::time::Instant;

use pz::pipeline::{self, CompressOptions, Pipeline};

const BLOCK: usize = 1024 * 1024;
const PASSES: usize = 3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: pz2_eval <files...>");
        std::process::exit(2);
    }
    println!(
        "{:>12} {:>9} | {:>8} {:>9} | {:>8} {:>9} | {:>7}",
        "file", "MB", "pz2 %", "dec MB/s", "lzf %", "dec MB/s", "speedup"
    );

    for path in &args {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path);
        let mb = data.len() as f64 / 1048576.0;

        // --- pz2: encode all blocks ---
        let blocks: Vec<&[u8]> = data.chunks(BLOCK).collect();
        let encoded: Vec<Vec<u8>> = blocks
            .iter()
            .map(|b| pz::pz2::encode(b).expect("pz2 encode"))
            .collect();
        let pz2_size: usize = encoded.iter().map(Vec::len).sum();

        // Verify once, then time decode passes.
        for (enc, blk) in encoded.iter().zip(blocks.iter()) {
            let dec = pz::pz2::decode(enc, blk.len()).expect("pz2 decode");
            assert_eq!(&dec, blk, "pz2 round-trip mismatch");
        }
        let mut best = f64::INFINITY;
        for _ in 0..PASSES {
            let t = Instant::now();
            for (enc, blk) in encoded.iter().zip(blocks.iter()) {
                let dec = pz::pz2::decode(enc, blk.len()).expect("pz2 decode");
                std::hint::black_box(&dec);
            }
            best = best.min(t.elapsed().as_secs_f64());
        }
        let pz2_mbs = mb / best;

        // --- Lzf baseline (same parse, shipped wire + FSE) ---
        let opts = CompressOptions {
            threads: 1,
            ..Default::default()
        };
        let lzf = pipeline::compress_with_options(&data, Pipeline::Lzf, &opts).expect("lzf");
        let dec = pipeline::decompress_with_threads(&lzf, 1).expect("lzf decode");
        assert_eq!(dec, data, "lzf round-trip mismatch");
        let mut best_lzf = f64::INFINITY;
        for _ in 0..PASSES {
            let t = Instant::now();
            let dec = pipeline::decompress_with_threads(&lzf, 1).expect("lzf decode");
            std::hint::black_box(&dec);
            best_lzf = best_lzf.min(t.elapsed().as_secs_f64());
        }
        let lzf_mbs = mb / best_lzf;

        println!(
            "{:>12} {:>9.1} | {:>8.3} {:>9.0} | {:>8.3} {:>9.0} | {:>6.2}x",
            name,
            mb,
            100.0 * pz2_size as f64 / data.len() as f64,
            pz2_mbs,
            100.0 * lzf.len() as f64 / data.len() as f64,
            lzf_mbs,
            pz2_mbs / lzf_mbs
        );
    }
}
