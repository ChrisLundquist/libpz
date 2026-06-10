//! pz2-G32 stage-2 gate 1: CPU-SIMD (NEON) baseline for the 32-lane layout.
//!
//! Per the gpu-path-research.md verifier finding, the honest denominator for
//! any GPU decode number on unified memory is a CPU-SIMD decoder of the SAME
//! GPU-friendly wire — not the shipped wire. This probe measures the literal
//! Huffman phase in isolation (the phase the stage-2 Metal kernel decodes)
//! and the full block decode (literal + splice), single-thread and
//! all-cores, across four decoders:
//!
//! - `pz2 8L`  — shipped 8-lane decoder over the SHIPPED pz2 wire
//! - `g32 sc`  — stage-1 scalar decoder over the G32 wire (per-symbol branch)
//! - `g32 rd`  — round-based portable decoder over the G32 wire
//! - `g32 nn`  — NEON decoder over the G32 wire (aarch64; = `rd` elsewhere)
//!
//! All decoders are round-trip verified against the original input before
//! timing. Timing is median of `--reps` (default 7) with min-max spread.
//!
//! Usage:
//!   cargo run --release --no-default-features --example pz2_g32_cpu_simd -- \
//!     [--reps N] <files...>

use std::time::Instant;

use pz::lzseq::SeqConfig;

const BLOCK: usize = 2 * 1024 * 1024; // shipped DEFAULT_PZ2_BLOCK_SIZE

/// Mirror of the shipped `pz2_auto_greedy` (pub(crate) in the pipeline).
fn auto_greedy(block: &[u8]) -> bool {
    let p = pz::analysis::analyze(block);
    !(p.byte_entropy > 7.5 && p.match_density < 0.1)
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Time `f()` `reps` times; return (median, min, max) seconds.
fn bench(reps: usize, mut f: impl FnMut()) -> (f64, f64, f64) {
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        f();
        ts.push(t.elapsed().as_secs_f64());
    }
    let (min, max) = (
        ts.iter().cloned().fold(f64::INFINITY, f64::min),
        ts.iter().cloned().fold(0.0, f64::max),
    );
    (median(&mut ts), min, max)
}

/// Run `f(block_index)` for every block across all cores (one chunk of the
/// block list per thread), mirroring the container's block-parallel decode.
fn par_blocks(n_blocks: usize, threads: usize, f: impl Fn(usize) + Sync) {
    let per = n_blocks.div_ceil(threads);
    std::thread::scope(|s| {
        for t in 0..threads {
            let f = &f;
            let lo = t * per;
            let hi = ((t + 1) * per).min(n_blocks);
            if lo < hi {
                s.spawn(move || {
                    for i in lo..hi {
                        f(i);
                    }
                });
            }
        }
    });
}

struct Corpus {
    name: String,
    blocks: Vec<Vec<u8>>,  // original input blocks
    encoded: Vec<Vec<u8>>, // shipped pz2 wire
    g32: Vec<Vec<u8>>,     // G32-transcoded wire
    lit_bytes: usize,      // total literal bytes (the literal-phase volume)
    huff_lit_bytes: usize, // literal bytes in LIT_HUFF blocks only
}

fn main() {
    let mut reps = 7usize;
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
        eprintln!("usage: pz2_g32_cpu_simd [--reps N] <files...>");
        std::process::exit(2);
    }
    let threads = std::thread::available_parallelism().map_or(8, |n| n.get());
    println!("threads for all-cores runs: {threads}\n");

    let mut corpora = Vec::new();
    for path in &files {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path).to_string();
        let blocks: Vec<Vec<u8>> = data.chunks(BLOCK).map(<[u8]>::to_vec).collect();
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
        let g32: Vec<Vec<u8>> = encoded
            .iter()
            .map(|e| pz::pz2::transcode_g32(e).expect("transcode"))
            .collect();

        // Round-trip + variant agreement, and literal-volume accounting.
        let mut lit_bytes = 0usize;
        let mut huff_lit_bytes = 0usize;
        for ((enc, g), blk) in encoded.iter().zip(&g32).zip(&blocks) {
            let lits = pz::pz2::spike_decode_lits_pz2(enc).expect("lits pz2");
            for v in 0..3u8 {
                let l = pz::pz2::spike_decode_lits_g32(g, v).expect("lits g32");
                assert_eq!(l, lits, "{name}: g32 variant {v} literal mismatch");
            }
            let d = pz::pz2::decode_g32_simd(g, blk.len()).expect("decode_g32_simd");
            assert_eq!(&d, blk, "{name}: decode_g32_simd round-trip mismatch");
            lit_bytes += lits.len();
            huff_lit_bytes += match pz::pz2::spike_lit_materials(enc).expect("materials") {
                Some((_, l)) => l.len(),
                None => 0,
            };
        }

        corpora.push(Corpus {
            name,
            blocks,
            encoded,
            g32,
            lit_bytes,
            huff_lit_bytes,
        });
    }

    // ---- Literal phase, single thread ----
    println!("LITERAL PHASE (entropy decode only), single thread, median MB/s of {reps} reps");
    println!(
        "{:>10} {:>8} {:>9} {:>9} {:>9} {:>9} {:>8} {:>8}",
        "file", "lit MB", "pz2 8L", "g32 sc", "g32 rd", "g32 nn", "nn/8L", "spread%"
    );
    for c in &corpora {
        let mb = c.lit_bytes as f64 / 1e6;
        let run = |variant: i8| {
            bench(reps, || {
                for (enc, g) in c.encoded.iter().zip(&c.g32) {
                    let l = if variant < 0 {
                        pz::pz2::spike_decode_lits_pz2(enc).expect("lits")
                    } else {
                        pz::pz2::spike_decode_lits_g32(g, variant as u8).expect("lits")
                    };
                    std::hint::black_box(l);
                }
            })
        };
        let (m8, _, _) = run(-1);
        let (msc, _, _) = run(0);
        let (mrd, _, _) = run(1);
        let (mnn, lo, hi) = run(2);
        println!(
            "{:>10} {:>8.1} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>7.2}x {:>7.1}%",
            c.name,
            mb,
            mb / m8,
            mb / msc,
            mb / mrd,
            mb / mnn,
            m8 / mnn,
            100.0 * (hi - lo) / mnn,
        );
    }

    // ---- Literal phase, all cores (all corpora pooled into one block list) ----
    let all: Vec<(&Corpus, usize)> = corpora
        .iter()
        .flat_map(|c| (0..c.encoded.len()).map(move |i| (c, i)))
        .collect();
    let pooled_lit_mb: f64 = corpora.iter().map(|c| c.lit_bytes as f64).sum::<f64>() / 1e6;
    println!(
        "\nLITERAL PHASE, all cores ({threads} threads), pooled corpus {pooled_lit_mb:.1} lit MB"
    );
    for (label, variant) in [
        ("pz2 8L", -1i8),
        ("g32 sc", 0),
        ("g32 rd", 1),
        ("g32 nn", 2),
    ] {
        let (m, lo, hi) = bench(reps, || {
            par_blocks(all.len(), threads, |i| {
                let (c, b) = all[i];
                let l = if variant < 0 {
                    pz::pz2::spike_decode_lits_pz2(&c.encoded[b]).expect("lits")
                } else {
                    pz::pz2::spike_decode_lits_g32(&c.g32[b], variant as u8).expect("lits")
                };
                std::hint::black_box(l);
            });
        });
        println!(
            "  {label}: {:>7.0} MB/s lit  ({:.2} ms median, spread {:.1}%)",
            pooled_lit_mb / m,
            1e3 * m,
            100.0 * (hi - lo) / m
        );
    }

    // ---- Full block decode (literal + splice), ST and all-cores ----
    let pooled_mb: f64 = corpora
        .iter()
        .map(|c| c.blocks.iter().map(Vec::len).sum::<usize>() as f64)
        .sum::<f64>()
        / 1e6;
    println!("\nFULL BLOCK DECODE (entropy + splice), pooled corpus {pooled_mb:.1} MB");
    for (label, st) in [("single-thread", true), ("all-cores", false)] {
        for (dec, is_g32) in [("pz2", false), ("g32 simd", true)] {
            let one = |c: &Corpus, b: usize| {
                let blk_len = c.blocks[b].len();
                let d = if is_g32 {
                    pz::pz2::decode_g32_simd(&c.g32[b], blk_len).expect("decode")
                } else {
                    pz::pz2::decode(&c.encoded[b], blk_len).expect("decode")
                };
                std::hint::black_box(d);
            };
            let (m, lo, hi) = bench(reps, || {
                if st {
                    for &(c, b) in &all {
                        one(c, b);
                    }
                } else {
                    par_blocks(all.len(), threads, |i| {
                        let (c, b) = all[i];
                        one(c, b);
                    });
                }
            });
            println!(
                "  {label:>13} {dec:>9}: {:>7.0} MB/s  ({:.2} ms median, spread {:.1}%)",
                pooled_mb / m,
                1e3 * m,
                100.0 * (hi - lo) / m
            );
        }
    }

    let huff_share: f64 = 100.0 * corpora.iter().map(|c| c.huff_lit_bytes as f64).sum::<f64>()
        / corpora
            .iter()
            .map(|c| c.lit_bytes as f64)
            .sum::<f64>()
            .max(1.0);
    println!("\n(LIT_HUFF share of literal bytes: {huff_share:.1}% — the rest is LIT_RAW memcpy)");
}
