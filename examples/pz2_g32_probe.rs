//! pz2-G32 stage-1 probe: format cost + scalar decode of the 32-lane layout.
//!
//! Encodes each input with the shipped pz2 block recipe (2 MiB blocks,
//! window = block, auto-greedy parse — the `-p pz2` defaults), transcodes
//! every block's literal section into the GDeflate-style 32-lane
//! word-interleaved layout (`pz2::transcode_g32`, same Huffman table,
//! sequence section byte-identical), then reports:
//!
//! - PRIMARY (deterministic): compressed bytes pz2 vs G32, per file and
//!   aggregate. Kill criterion for stage 1: aggregate delta > 0.5pp.
//! - Round-trip: every G32 block decoded with the scalar `decode_g32` and
//!   compared byte-exact against the original block.
//! - SECONDARY (indicative only — shared machine, noisy): single-thread
//!   decode time of all blocks, pz2 vs G32, median + min/max over N reps.
//!
//! Usage:
//!   cargo run --release --no-default-features --example pz2_g32_probe -- \
//!     [--reps N] <files...>

use std::time::Instant;

use pz::lzseq::SeqConfig;

const BLOCK: usize = 2 * 1024 * 1024; // shipped DEFAULT_PZ2_BLOCK_SIZE

/// Mirror of the shipped `pz2_auto_greedy` (pub(crate) in the pipeline):
/// greedy unless the block is near-random.
fn auto_greedy(block: &[u8]) -> bool {
    let p = pz::analysis::analyze(block);
    !(p.byte_entropy > 7.5 && p.match_density < 0.1)
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn main() {
    let mut reps = 5usize;
    let mut files: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--reps" {
            reps = args.next().and_then(|v| v.parse().ok()).expect("--reps N");
        } else {
            files.push(a);
        }
    }
    if files.is_empty() {
        eprintln!("usage: pz2_g32_probe [--reps N] <files...>");
        std::process::exit(2);
    }

    println!(
        "{:>10} {:>11} {:>11} {:>11} {:>8} {:>8} {:>7} | {:>9} {:>9} {:>7}",
        "file",
        "orig",
        "pz2 B",
        "g32 B",
        "pz2 %",
        "g32 %",
        "Δpp",
        "pz2 MB/s",
        "g32 MB/s",
        "g32/pz2"
    );

    let (mut tot_orig, mut tot_pz2, mut tot_g32) = (0usize, 0usize, 0usize);
    let mut all_ok = true;

    for path in &files {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path);
        let mb = data.len() as f64 / 1e6;

        // Shipped pz2 recipe: 2 MiB blocks, window = block, auto-greedy.
        let blocks: Vec<&[u8]> = data.chunks(BLOCK).collect();
        let encoded: Vec<Vec<u8>> = blocks
            .iter()
            .map(|b| {
                let config = SeqConfig {
                    max_window: BLOCK,
                    greedy: auto_greedy(b),
                    ..SeqConfig::default()
                };
                pz::pz2::encode_with_config(b, &config).expect("encode")
            })
            .collect();
        let transcoded: Vec<Vec<u8>> = encoded
            .iter()
            .map(|e| pz::pz2::transcode_g32(e).expect("transcode"))
            .collect();

        let pz2_size: usize = encoded.iter().map(Vec::len).sum();
        let g32_size: usize = transcoded.iter().map(Vec::len).sum();

        // Round-trip verification (byte-exact), both decoders.
        for ((enc, g32), blk) in encoded.iter().zip(&transcoded).zip(&blocks) {
            let d = pz::pz2::decode(enc, blk.len()).expect("pz2 decode");
            if &d != blk {
                println!("{name}: pz2 ROUND-TRIP MISMATCH");
                all_ok = false;
            }
            let d = pz::pz2::decode_g32(g32, blk.len()).expect("g32 decode");
            if &d != blk {
                println!("{name}: g32 ROUND-TRIP MISMATCH");
                all_ok = false;
            }
        }

        // Timing: full-corpus ST decode, `reps` repetitions, median.
        let mut t_pz2 = Vec::with_capacity(reps);
        let mut t_g32 = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            for (enc, blk) in encoded.iter().zip(&blocks) {
                std::hint::black_box(pz::pz2::decode(enc, blk.len()).expect("decode"));
            }
            t_pz2.push(t.elapsed().as_secs_f64());
            let t = Instant::now();
            for (g32, blk) in transcoded.iter().zip(&blocks) {
                std::hint::black_box(pz::pz2::decode_g32(g32, blk.len()).expect("decode"));
            }
            t_g32.push(t.elapsed().as_secs_f64());
        }
        let m_pz2 = median(&mut t_pz2);
        let m_g32 = median(&mut t_g32);

        let pct_pz2 = 100.0 * pz2_size as f64 / data.len() as f64;
        let pct_g32 = 100.0 * g32_size as f64 / data.len() as f64;
        println!(
            "{:>10} {:>11} {:>11} {:>11} {:>8.3} {:>8.3} {:>+7.4} | {:>9.0} {:>9.0} {:>6.2}x",
            name,
            data.len(),
            pz2_size,
            g32_size,
            pct_pz2,
            pct_g32,
            pct_g32 - pct_pz2,
            mb / m_pz2,
            mb / m_g32,
            m_pz2 / m_g32,
        );
        eprintln!(
            "  [{name} timing spread over {reps} reps] pz2 {:.1}-{:.1} ms, g32 {:.1}-{:.1} ms",
            1e3 * t_pz2.first().unwrap(),
            1e3 * t_pz2.last().unwrap(),
            1e3 * t_g32.first().unwrap(),
            1e3 * t_g32.last().unwrap(),
        );

        tot_orig += data.len();
        tot_pz2 += pz2_size;
        tot_g32 += g32_size;
    }

    let pct_pz2 = 100.0 * tot_pz2 as f64 / tot_orig as f64;
    let pct_g32 = 100.0 * tot_g32 as f64 / tot_orig as f64;
    println!(
        "\nAGGREGATE: orig {} | pz2 {} ({:.4}%) | g32 {} ({:.4}%) | delta {:+.4}pp | round-trip {}",
        tot_orig,
        tot_pz2,
        pct_pz2,
        tot_g32,
        pct_g32,
        pct_g32 - pct_pz2,
        if all_ok { "OK" } else { "FAILED" }
    );
    println!(
        "VERDICT (0.5pp kill line): {}",
        if !all_ok {
            "FAIL (round-trip)"
        } else if pct_g32 - pct_pz2 > 0.5 {
            "KILL"
        } else {
            "PASS"
        }
    );
}
