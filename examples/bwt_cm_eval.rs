// SPIKE #1 evaluation: BWT + order-1 context-mixing binary range coder vs the
// current bw entropy tail (MTF -> zrle/RLE -> FSE), compared on the SAME raw
// BWT byte stream. Reports post-BWT sizes, decode MB/s, and round-trip status.
//
//   cargo run --release --no-default-features --example bwt_cm_eval -- <file> [block_bytes]
//
// If <file> is "SLICE", a 1 MiB slice from the middle of /tmp/silesia.blob is used.

use pz::{bwt, bwt_cm, fse, mtf, rle, zrle};
use std::time::Instant;

/// Encode raw BWT bytes through the current bw tail (MTF -> zrle/RLE -> FSE).
/// Returns (entropy_bytes_len, zrle_used). This is exactly the post-header size
/// the bw pipeline emits (it adds an 8-byte primary_index+len header on top).
fn bw_tail_size(bwt_bytes: &[u8]) -> (usize, bool) {
    let mtf_out = mtf::encode(bwt_bytes);
    let (rle_out, zrle_used) = match zrle::encode(&mtf_out) {
        Some(e) => (e, true),
        None => (rle::encode(&mtf_out), false),
    };
    let fse_out = fse::encode_best(&rle_out);
    (fse_out.len(), zrle_used)
}

fn run(label: &str, input: &[u8]) {
    let n = input.len();
    let bwt_res = match bwt::encode(input) {
        Some(r) => r,
        None => {
            println!("{label}: empty input, skipped");
            return;
        }
    };
    let bwt_bytes = &bwt_res.data;

    // --- current bw tail ---
    let (tail_len, zrle_used) = bw_tail_size(bwt_bytes);

    // --- CM tail (full context-mixing) ---
    let t0 = Instant::now();
    let cm = bwt_cm::encode(bwt_bytes);
    let cm_enc_s = t0.elapsed().as_secs_f64();

    // round-trip + decode timing (best of 3)
    let mut best_dec_s = f64::INFINITY;
    let mut ok = true;
    for _ in 0..3 {
        let t = Instant::now();
        let dec = bwt_cm::decode(&cm, bwt_bytes.len()).expect("cm decode");
        let s = t.elapsed().as_secs_f64();
        if &dec != bwt_bytes {
            ok = false;
        }
        if s < best_dec_s {
            best_dec_s = s;
        }
    }

    // --- MID tail (order-0 + order-1 mix, no order-2) ---
    let mid = bwt_cm::encode_mid(bwt_bytes);
    let mut best_mid_dec_s = f64::INFINITY;
    let mut mid_ok = true;
    for _ in 0..3 {
        let t = Instant::now();
        let dec = bwt_cm::decode_mid(&mid, bwt_bytes.len()).expect("mid decode");
        let s = t.elapsed().as_secs_f64();
        if &dec != bwt_bytes {
            mid_ok = false;
        }
        if s < best_mid_dec_s {
            best_mid_dec_s = s;
        }
    }
    let mid_dec_mbs = (n as f64 / 1e6) / best_mid_dec_s.max(1e-9);
    let mid_over_tail = mid.len() as f64 / tail_len as f64;

    // --- BLEND tail (order-1 ⊕ order-0 linear blend, no mixer) ---
    let blend = bwt_cm::encode_blend(bwt_bytes);
    let mut best_blend_dec_s = f64::INFINITY;
    let mut blend_ok = true;
    for _ in 0..3 {
        let t = Instant::now();
        let dec = bwt_cm::decode_blend(&blend, bwt_bytes.len()).expect("blend decode");
        let s = t.elapsed().as_secs_f64();
        if &dec != bwt_bytes {
            blend_ok = false;
        }
        if s < best_blend_dec_s {
            best_blend_dec_s = s;
        }
    }
    let blend_dec_mbs = (n as f64 / 1e6) / best_blend_dec_s.max(1e-9);
    let blend_over_tail = blend.len() as f64 / tail_len as f64;

    // --- FAST tail (direct order-1, no mixing) ---
    let fast = bwt_cm::encode_fast(bwt_bytes);
    let mut best_fast_dec_s = f64::INFINITY;
    let mut fast_ok = true;
    for _ in 0..3 {
        let t = Instant::now();
        let dec = bwt_cm::decode_fast(&fast, bwt_bytes.len()).expect("fast decode");
        let s = t.elapsed().as_secs_f64();
        if &dec != bwt_bytes {
            fast_ok = false;
        }
        if s < best_fast_dec_s {
            best_fast_dec_s = s;
        }
    }
    let fast_dec_mbs = (n as f64 / 1e6) / best_fast_dec_s.max(1e-9);
    let fast_over_tail = fast.len() as f64 / tail_len as f64;

    let cm_ratio_vs_input = cm.len() as f64 / n as f64 * 100.0;
    let tail_ratio_vs_input = tail_len as f64 / n as f64 * 100.0;
    let cm_over_tail = cm.len() as f64 / tail_len as f64;
    // decode MB/s expressed over the ORIGINAL input bytes (what end users feel),
    // and over the BWT byte stream (what the coder actually processes — same here
    // since bwt is length-preserving).
    let dec_mbs = (n as f64 / 1e6) / best_dec_s.max(1e-9);
    let enc_mbs = (n as f64 / 1e6) / cm_enc_s.max(1e-9);

    println!("=== {label} (input {} bytes) ===", n);
    println!(
        "  bw tail (MTF->{}->FSE): {:>9} bytes  ({:5.2}% of input)",
        if zrle_used { "zrle" } else { "RLE" },
        tail_len,
        tail_ratio_vs_input
    );
    println!(
        "  CM (order-1 range):     {:>9} bytes  ({:5.2}% of input)",
        cm.len(),
        cm_ratio_vs_input
    );
    println!(
        "  CM / bw-tail ratio:     {:.4}   (gate: <= 0.9000 on dickens)",
        cm_over_tail
    );
    println!(
        "  CM encode: {:7.1} MB/s   CM decode: {:7.1} MB/s (gate: >= 20)  round-trip: {}",
        enc_mbs,
        dec_mbs,
        if ok { "OK" } else { "*** MISMATCH ***" }
    );
    println!(
        "  MID (o0+o1 mix):        {:>9} bytes  ({:5.2}% of input)  CM/tail={:.4}",
        mid.len(),
        mid.len() as f64 / n as f64 * 100.0,
        mid_over_tail
    );
    println!(
        "  MID decode:  {:7.1} MB/s (gate: >= 20)  round-trip: {}",
        mid_dec_mbs,
        if mid_ok { "OK" } else { "*** MISMATCH ***" }
    );
    println!(
        "  BLEND (o1+o0 linear):   {:>9} bytes  ({:5.2}% of input)  CM/tail={:.4}",
        blend.len(),
        blend.len() as f64 / n as f64 * 100.0,
        blend_over_tail
    );
    println!(
        "  BLEND decode:{:7.1} MB/s (gate: >= 20)  round-trip: {}",
        blend_dec_mbs,
        if blend_ok { "OK" } else { "*** MISMATCH ***" }
    );
    println!(
        "  FAST (order-1, no mix): {:>9} bytes  ({:5.2}% of input)  CM/tail={:.4}",
        fast.len(),
        fast.len() as f64 / n as f64 * 100.0,
        fast_over_tail
    );
    println!(
        "  FAST decode: {:7.1} MB/s (gate: >= 20)  round-trip: {}",
        fast_dec_mbs,
        if fast_ok { "OK" } else { "*** MISMATCH ***" }
    );
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bwt_cm_eval <file|SLICE> [block_bytes]");
        std::process::exit(2);
    }
    let block_bytes: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1 << 20);

    let (label, data): (String, Vec<u8>) = if args[1] == "SLICE" {
        let blob = std::fs::read("/tmp/silesia.blob").expect("read silesia.blob");
        let mid = blob.len() / 2;
        let end = (mid + block_bytes).min(blob.len());
        (
            format!("silesia.blob[{mid}..{end}]"),
            blob[mid..end].to_vec(),
        )
    } else {
        let full = std::fs::read(&args[1]).expect("read file");
        let take = block_bytes.min(full.len());
        (
            format!("{} (first {} bytes)", args[1], take),
            full[..take].to_vec(),
        )
    };

    run(&label, &data);
}
