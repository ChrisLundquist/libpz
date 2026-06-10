//! pz2 block/window size sweep: ratio vs single-thread decode.
//!
//! pz2's match window is capped by the block size (each block parses cold),
//! so block size IS the window lever. This measures ratio + ST decode at
//! 1/2/4/8 MiB blocks with `max_window` = block size, to find the Pareto
//! point for the streaming default. (The bw 2-4 MiB rejection was
//! inverse-BWT cache physics — LZ decode is sequential-write and expected
//! to be size-neutral; this verifies that.)
//!
//! Usage: cargo run --release --example pz2_block_sweep -- <files...>

use std::time::Instant;

use pz::lzseq::SeqConfig;

const PASSES: usize = 3;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: pz2_block_sweep <files...>");
        std::process::exit(2);
    }
    println!(
        "{:>14} {:>7} | {:>8} {:>10} | {:>9}",
        "file", "block", "ratio %", "dec MB/s", "enc MB/s"
    );

    for path in &args {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path);
        let mb = data.len() as f64 / 1048576.0;

        for shift in [20usize, 21, 22, 23] {
            let block = 1usize << shift;
            let config = SeqConfig {
                max_window: block,
                ..SeqConfig::default()
            };

            let blocks: Vec<&[u8]> = data.chunks(block).collect();
            let t = Instant::now();
            let encoded: Vec<Vec<u8>> = blocks
                .iter()
                .map(|b| pz::pz2::encode_with_config(b, &config).expect("encode"))
                .collect();
            let enc_mbs = mb / t.elapsed().as_secs_f64();
            let size: usize = encoded.iter().map(Vec::len).sum();

            for (enc, blk) in encoded.iter().zip(blocks.iter()) {
                let dec = pz::pz2::decode(enc, blk.len()).expect("decode");
                assert_eq!(&dec, blk, "round-trip mismatch at block {block}");
            }
            let mut best = f64::INFINITY;
            for _ in 0..PASSES {
                let t = Instant::now();
                for (enc, blk) in encoded.iter().zip(blocks.iter()) {
                    let dec = pz::pz2::decode(enc, blk.len()).expect("decode");
                    std::hint::black_box(&dec);
                }
                best = best.min(t.elapsed().as_secs_f64());
            }

            println!(
                "{:>14} {:>5}Mi | {:>8.3} {:>10.0} | {:>9.0}",
                name,
                block >> 20,
                100.0 * size as f64 / data.len() as f64,
                mb / best,
                enc_mbs
            );
        }
    }
}
