//! Entropy-accounting probe for pz2's Huffman lanes vs tANS alternatives
//! (task: "would tANS or another entropy coder beat Huffman if we had
//! global state tables?"). Pure histogram math — no codec changes, no
//! wire formats, just bit-exact pricing of each scenario:
//!
//!   A  shipped:    per-block ≤11-bit package-merge Huffman, shipped
//!                  headers and CONST/RAW fallbacks mirrored exactly
//!   B  tANS/blk:   per-block Shannon ideal + the SAME header as A —
//!                  isolates the fractional-bit (sub-1-bit) modeling win,
//!                  which is all tANS adds over optimal Huffman
//!   C  tANS/seg:   one global table per 32 MiB segment per lane
//!                  (cross-entropy of each block against the segment
//!                  histogram), header amortized; C' lets each block
//!                  choose min(A, C) + 1 mode byte (the deployable form)
//!   D  o1/seg:     order-1 (previous symbol in lane) segment-global
//!                  tables, seq-code lanes only (order-1 literals are the
//!                  measured ctx-literals dead end)
//!
//! Layout mirrors pz2: independent 2 MiB blocks, greedy parse, 2 MiB
//! window; segments are 32 MiB groups of blocks. Gate (roadmap): a
//! scenario must beat shipped by ≥0.15-0.2pp of input to justify a wire
//! change.
//!
//! Usage: pz2_entropy_probe FILE [FILE...]

use pz::lzseq::SeqConfig;
use pz::pz2;

const BLOCK: usize = 2 * 1024 * 1024;
const SEGMENT: usize = 32 * 1024 * 1024;
const NUM_LANES: usize = 8; // pz2 literal lanes (wire constant)

/// Per-block lane streams.
struct BlockStreams {
    lits: Vec<u8>,
    ll: Vec<u8>,
    of: Vec<u8>,
    ml: Vec<u8>,
}

/// Selects one lane's stream out of a block.
type LanePick = fn(&BlockStreams) -> &Vec<u8>;

fn hist(data: &[u8]) -> [u64; 256] {
    let mut h = [0u64; 256];
    for &b in data {
        h[b as usize] += 1;
    }
    h
}

/// Shannon information content of `data` under distribution `p` (counts
/// `ph` summing to `pn`), in bits. `ph` must cover every symbol in `data`.
fn cross_entropy_bits(data: &[u8], ph: &[u64; 256], pn: u64) -> f64 {
    let h = hist(data);
    let mut bits = 0.0;
    for s in 0..256 {
        if h[s] > 0 {
            bits += h[s] as f64 * ((pn as f64) / (ph[s] as f64)).log2();
        }
    }
    bits
}

/// Shipped cost of one literal section in bits (mode byte + table + lane
/// length words + Huffman payload, RAW fallback mirrored).
fn lit_shipped_bits(lits: &[u8]) -> f64 {
    let h = hist(lits);
    let distinct = h.iter().filter(|&&c| c > 0).count();
    let raw = 8.0 * (1 + lits.len()) as f64;
    if distinct < 2 {
        return raw;
    }
    let mut counts32 = [0u32; 256];
    for s in 0..256 {
        counts32[s] = h[s] as u32;
    }
    let lengths = pz2::probe_huffman_lengths(&counts32);
    let payload: u64 = (0..256).map(|s| h[s] * lengths[s] as u64).sum();
    // Lane split rounds each of the 8 lanes up to whole bytes.
    let huff =
        8.0 * (1 + 128 + 4 * NUM_LANES) as f64 + payload as f64 + 8.0 * NUM_LANES as f64 / 2.0;
    huff.min(raw)
}

/// Shipped cost of one sequence-code stream in bits (CONST or Huffman,
/// exactly as encode_code_stream prices it).
fn seq_shipped_bits(codes: &[u8]) -> f64 {
    if codes.is_empty() {
        return 0.0;
    }
    let h = hist(codes);
    if h.iter().filter(|&&c| c > 0).count() == 1 {
        return 16.0; // CODES_CONST: mode + value
    }
    let mut counts32 = [0u32; 256];
    for s in 0..256 {
        counts32[s] = h[s] as u32;
    }
    let lengths = pz2::probe_huffman_lengths(&counts32);
    let payload: u64 = (0..256).map(|s| h[s] * lengths[s] as u64).sum();
    8.0 * (1 + 16 + 4) as f64 + payload as f64 + 4.0
}

/// Per-block Shannon ideal + the same header the shipped coder pays.
fn shannon_block_bits(codes: &[u8], header_bits: f64, const_ok: bool) -> f64 {
    if codes.is_empty() {
        return 0.0;
    }
    let h = hist(codes);
    if const_ok && h.iter().filter(|&&c| c > 0).count() == 1 {
        return 16.0;
    }
    let n = codes.len() as u64;
    let ideal = cross_entropy_bits(codes, &h, n);
    header_bits + ideal
}

/// Order-1 cross-entropy of `codes` against segment-global conditional
/// histograms `ctx[prev][sym]` (which must include this block's counts).
fn o1_cross_bits(codes: &[u8], ctx: &[[u64; 32]; 32], ctx_tot: &[u64; 32]) -> f64 {
    let mut bits = 0.0;
    let mut prev = 0usize;
    for &c in codes {
        let s = c as usize;
        bits += ((ctx_tot[prev] as f64) / (ctx[prev][s] as f64)).log2();
        prev = s;
    }
    bits
}

#[derive(Default, Clone)]
struct LaneTotals {
    shipped: f64,    // A
    tans_block: f64, // B
    tans_seg: f64,   // C
    best_seg: f64,   // C' = per-block min(A, C) + mode byte
    o1_seg: f64,     // D (seq lanes only)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: pz2_entropy_probe FILE [FILE...]");
        std::process::exit(2);
    }

    let config = SeqConfig {
        max_window: BLOCK.next_power_of_two().max(1 << 20),
        greedy: true,
        ..SeqConfig::default()
    };

    for path in &args {
        let data = std::fs::read(path).expect("read input");
        let mut lanes: [LaneTotals; 4] = Default::default(); // lit, ll, of, ml
        let lane_names = ["lit", "ll ", "of ", "ml "];

        for seg in data.chunks(SEGMENT) {
            // Parse every block in the segment first (segment-global
            // scenarios need the full histograms before pricing blocks).
            let blocks: Vec<BlockStreams> = seg
                .chunks(BLOCK)
                .map(|b| {
                    let (lits, ll, of, ml) =
                        pz2::probe_lane_streams(b, &config).expect("parse block");
                    BlockStreams { lits, ll, of, ml }
                })
                .collect();

            let pickers: [(usize, LanePick); 4] = [
                (0, |b| &b.lits),
                (1, |b| &b.ll),
                (2, |b| &b.of),
                (3, |b| &b.ml),
            ];
            for (li, pick) in pickers {
                let is_lit = li == 0;
                // Segment-global order-0 histogram.
                let mut gh = [0u64; 256];
                let mut gn = 0u64;
                for b in &blocks {
                    for &s in pick(b) {
                        gh[s as usize] += 1;
                    }
                    gn += pick(b).len() as u64;
                }
                // Segment-global order-1 (seq lanes; 32-symbol alphabet).
                let mut ctx = [[0u64; 32]; 32];
                let mut ctx_tot = [0u64; 32];
                if !is_lit {
                    for b in &blocks {
                        let mut prev = 0usize;
                        for &c in pick(b) {
                            ctx[prev][c as usize] += 1;
                            ctx_tot[prev] += 1;
                            prev = c as usize;
                        }
                    }
                }

                // Segment header charges: order-0 table once, order-1
                // table once (32x32 nibble-class ≈ 512 B).
                let seg_hdr = if is_lit { 8.0 * 128.0 } else { 8.0 * 16.0 };
                let o1_hdr = 8.0 * 512.0;
                let t = &mut lanes[li];
                if gn > 0 {
                    t.tans_seg += seg_hdr;
                    t.best_seg += seg_hdr;
                    if !is_lit {
                        t.o1_seg += o1_hdr;
                    }
                }

                for b in &blocks {
                    let s = pick(b);
                    let shipped = if is_lit {
                        lit_shipped_bits(s)
                    } else {
                        seq_shipped_bits(s)
                    };
                    let hdr = if is_lit {
                        8.0 * (1 + 128 + 4 * NUM_LANES) as f64
                    } else {
                        8.0 * 21.0
                    };
                    let tans_blk = shannon_block_bits(s, hdr, !is_lit).min(shipped);
                    let cross = if s.is_empty() {
                        0.0
                    } else {
                        8.0 + cross_entropy_bits(s, &gh, gn) // mode byte
                    };
                    t.shipped += shipped;
                    t.tans_block += tans_blk;
                    t.tans_seg += cross;
                    t.best_seg += 8.0 + cross.min(shipped); // choice + mode byte
                    if !is_lit && !s.is_empty() {
                        t.o1_seg += 8.0 + o1_cross_bits(s, &ctx, &ctx_tot);
                    }
                }
            }
        }

        let total_in = data.len() as f64;
        let pp = |bits: f64| 100.0 * bits / 8.0 / total_in;
        println!("\n=== {} ({} bytes) ===", path, data.len());
        println!(
            "{:<5} {:>10} {:>18} {:>18} {:>18} {:>18}",
            "lane", "A shipped", "B tANS/blk", "C tANS/seg", "C' min(A,C)", "D o1/seg"
        );
        let mut tot = LaneTotals::default();
        for (li, name) in lane_names.iter().enumerate() {
            let t = &lanes[li];
            let d = |x: f64| format!("{:8.4}pp ({:+.4})", pp(x), pp(x) - pp(t.shipped));
            println!(
                "{:<5} {:>9.4}pp {:>18} {:>18} {:>18} {:>18}",
                name,
                pp(t.shipped),
                d(t.tans_block),
                d(t.tans_seg),
                d(t.best_seg),
                if li == 0 {
                    "-".to_string()
                } else {
                    d(t.o1_seg)
                }
            );
            tot.shipped += t.shipped;
            tot.tans_block += t.tans_block;
            tot.tans_seg += t.tans_seg;
            tot.best_seg += t.best_seg;
            tot.o1_seg += if li == 0 { t.shipped } else { t.o1_seg };
        }
        println!(
            "TOTAL A {:.4}pp | B {:+.4}pp | C {:+.4}pp | C' {:+.4}pp | D(seq)+A(lit) {:+.4}pp",
            pp(tot.shipped),
            pp(tot.tans_block) - pp(tot.shipped),
            pp(tot.tans_seg) - pp(tot.shipped),
            pp(tot.best_seg) - pp(tot.shipped),
            pp(tot.o1_seg) - pp(tot.shipped),
        );
        println!("gate: any scenario must reach -0.15 to -0.20pp vs A to justify a wire change");
    }
}
