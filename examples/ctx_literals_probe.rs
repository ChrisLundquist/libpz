//! Stage-0 spike probe: order-1 (previous-byte) context-bucketed literals
//! through the EXISTING FSE coder — no new entropy coder.
//!
//! Question (future-pipelines roadmap, LZMA-class family): do the literals
//! that SURVIVE LZ matching retain enough order-1 structure that splitting
//! the literal stream into per-context FSE buckets pays for the extra
//! per-bucket table headers? Gate: >=2-3pp of input on Silesia text ->
//! pursue a Brotli-style context-literal path; <1pp -> kill the branch.
//!
//! Method: per 1 MiB block (the shipped block size), lazy-parse with
//! `lz77::compress_lazy_to_matches` (32 KiB window — yields MORE surviving
//! literals than the shipped 1 MiB-window lzseq parse, so the measured gain
//! is an OPTIMISTIC bound: a fail here is a fail everywhere). Each literal's
//! context is the previous byte of the original data (available to any LZ
//! decoder at emit time). Compare, summed over blocks:
//!   - baseline: one `fse::encode_best` stream per block (what lzf does)
//!   - C4/C16/C64/C256: per-(prev>>6 / prev>>4 / prev>>2 / prev) buckets,
//!     each its own `fse::encode_best` (real table-header cost included)
//!   - ideal O1: conditional entropy H(lit | prev byte) — the ceiling for
//!     ANY prev-byte context scheme, header-free.
//!
//! Usage: cargo run --release --no-default-features --example ctx_literals_probe -- files...

use std::fs;

const BLOCK: usize = 1024 * 1024;

fn entropy_bits(counts: &[u32], total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut h = 0.0;
    for &c in counts {
        if c > 0 {
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h * n
}

/// (literal, prev-original-byte) pairs for one block, via the lazy parse.
fn block_literals(block: &[u8]) -> Vec<(u8, u8)> {
    let matches = pz::lz77::compress_lazy_to_matches(block).expect("parse");
    let mut out = Vec::new();
    let mut pos = 0usize;
    for m in &matches {
        if m.length > 0 {
            pos += m.length as usize;
        }
        if pos < block.len() {
            assert_eq!(block[pos], m.next, "cursor walk desynced at {pos}");
            let ctx = if pos == 0 { 0 } else { block[pos - 1] };
            out.push((m.next, ctx));
            pos += 1;
        }
    }
    assert_eq!(pos, block.len(), "parse did not cover block");
    out
}

fn bucket_cost(pairs: &[(u8, u8)], shift: u8) -> usize {
    let nb = 256usize >> shift;
    let mut buckets: Vec<Vec<u8>> = vec![Vec::new(); nb];
    for &(lit, ctx) in pairs {
        buckets[(ctx >> shift) as usize].push(lit);
    }
    buckets
        .iter()
        .filter(|b| !b.is_empty())
        .map(|b| pz::fse::encode_best(b).len())
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: ctx_literals_probe <files...>");
        std::process::exit(2);
    }
    println!(
        "{:>10} {:>9} {:>6} | {:>9} {:>9} {:>9} {:>9} {:>9} | {:>9} {:>9}",
        "file", "lits", "share", "base", "C4", "C16", "C64", "C256", "idealO0", "idealO1"
    );

    for path in &args {
        let data = fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path);

        let mut lits_total = 0u64;
        let (mut base, mut c4, mut c16, mut c64, mut c256) = (0usize, 0, 0, 0, 0);
        let (mut ideal_o0, mut ideal_o1) = (0f64, 0f64);

        for block in data.chunks(BLOCK) {
            let pairs = block_literals(block);
            lits_total += pairs.len() as u64;
            let lits: Vec<u8> = pairs.iter().map(|&(l, _)| l).collect();

            base += pz::fse::encode_best(&lits).len();
            c4 += bucket_cost(&pairs, 6);
            c16 += bucket_cost(&pairs, 4);
            c64 += bucket_cost(&pairs, 2);
            c256 += bucket_cost(&pairs, 0);

            // Ideal (header-free) order-0 and order-1 conditional entropy.
            let mut h0 = [0u32; 256];
            let mut joint = vec![[0u32; 256]; 256];
            let mut ctx_n = [0u64; 256];
            for &(l, c) in &pairs {
                h0[l as usize] += 1;
                joint[c as usize][l as usize] += 1;
                ctx_n[c as usize] += 1;
            }
            ideal_o0 += entropy_bits(&h0, pairs.len() as u64) / 8.0;
            for c in 0..256 {
                ideal_o1 += entropy_bits(&joint[c], ctx_n[c]) / 8.0;
            }
        }

        let pp = |bytes: usize| -> f64 { 100.0 * (base as f64 - bytes as f64) / data.len() as f64 };
        println!(
            "{:>10} {:>9} {:>5.1}% | {:>9} {:>9} {:>9} {:>9} {:>9} | {:>9.0} {:>9.0}",
            name,
            lits_total,
            100.0 * lits_total as f64 / data.len() as f64,
            base,
            c4,
            c16,
            c64,
            c256,
            ideal_o0,
            ideal_o1
        );
        println!(
            "{:>27}   pp vs base: {:>7} {:+8.3} {:+9.3} {:+9.3} {:+9.3} | {:+9.3} {:+9.3}",
            "",
            "",
            pp(c4),
            pp(c16),
            pp(c64),
            pp(c256),
            pp(ideal_o0.round() as usize),
            pp(ideal_o1.round() as usize)
        );
    }
}
