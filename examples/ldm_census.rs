//! Long-range dedup census probe (CPU-only spike).
//!
//! Measures how much *verified* duplicate content exists in an input at
//! offsets larger than a threshold (default 1 MiB — beyond what any pz
//! block codec can see). Method:
//!
//! 1. Sample an exact 8-byte fingerprint (raw little-endian u64) at a fixed
//!    stride (default 32) over the whole input.
//! 2. Sort (fingerprint, position) pairs.
//! 3. Within each equal-fingerprint group, pair every sample with the
//!    *nearest earlier* sample at distance >= min-dist (two-pointer scan).
//! 4. Verify each candidate by actual byte comparison, greedily extending
//!    in both directions (distance is constant under extension, so the
//!    long-range property is preserved).
//! 5. Count bytes covered by verified matches of length >= min-match using
//!    a bitmap, so overlapping matches are never double-counted.
//!
//! Because the fingerprint is the raw bytes (not a hash), every candidate
//! is guaranteed to verify to at least 8 bytes; min-match is what makes a
//! match "worth a long-range copy" (zstd LDM default min match is 64).
//!
//! Usage:
//!   cargo run --release --no-default-features --example ldm_census -- \
//!       <file> [--stride 32] [--min-dist 1048576] [--min-match 64]

use std::env;
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

const DEFAULT_STRIDE: usize = 32;
const DEFAULT_MIN_DIST: usize = 1 << 20; // 1 MiB
const DEFAULT_MIN_MATCH: usize = 64;

struct Bitmap {
    words: Vec<u64>,
}

impl Bitmap {
    fn new(bits: usize) -> Self {
        Bitmap {
            words: vec![0u64; bits.div_ceil(64)],
        }
    }

    #[inline]
    fn get(&self, i: usize) -> bool {
        self.words[i >> 6] & (1u64 << (i & 63)) != 0
    }

    /// Set bits [start, end), returning how many were newly set.
    fn set_range_count_new(&mut self, start: usize, end: usize) -> usize {
        let mut new_bits = 0usize;
        let mut i = start;
        while i < end {
            let w = i >> 6;
            let bit_lo = i & 63;
            let span = (end - i).min(64 - bit_lo);
            let mask = if span == 64 {
                u64::MAX
            } else {
                ((1u64 << span) - 1) << bit_lo
            };
            let old = self.words[w];
            new_bits += (mask & !old).count_ones() as usize;
            self.words[w] = old | mask;
            i += span;
        }
        new_bits
    }
}

/// Length of common prefix of data[a..] and data[b..], capped at n - b (b > a).
fn match_forward(data: &[u8], a: usize, b: usize) -> usize {
    let n = data.len();
    let max = n - b;
    let mut k = 0usize;
    while k + 8 <= max {
        let x = u64::from_le_bytes(data[a + k..a + k + 8].try_into().unwrap());
        let y = u64::from_le_bytes(data[b + k..b + k + 8].try_into().unwrap());
        let diff = x ^ y;
        if diff != 0 {
            return k + (diff.trailing_zeros() / 8) as usize;
        }
        k += 8;
    }
    while k < max && data[a + k] == data[b + k] {
        k += 1;
    }
    k
}

/// Length of common suffix of data[..a] and data[..b] (bytes before a and b).
fn match_backward(data: &[u8], a: usize, b: usize) -> usize {
    let max = a; // a < b, so a bounds backward extension
    let mut k = 0usize;
    while k + 8 <= max {
        let x = u64::from_le_bytes(data[a - k - 8..a - k].try_into().unwrap());
        let y = u64::from_le_bytes(data[b - k - 8..b - k].try_into().unwrap());
        let diff = x ^ y;
        if diff != 0 {
            return k + (diff.leading_zeros() / 8) as usize;
        }
        k += 8;
    }
    while k < max && data[a - k - 1] == data[b - k - 1] {
        k += 1;
    }
    k
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let mut path: Option<String> = None;
    let mut stride = DEFAULT_STRIDE;
    let mut min_dist = DEFAULT_MIN_DIST;
    let mut min_match = DEFAULT_MIN_MATCH;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--stride" => {
                stride = args[i + 1].parse().expect("bad --stride");
                i += 2;
            }
            "--min-dist" => {
                min_dist = args[i + 1].parse().expect("bad --min-dist");
                i += 2;
            }
            "--min-match" => {
                min_match = args[i + 1].parse().expect("bad --min-match");
                i += 2;
            }
            other => {
                if path.is_some() {
                    eprintln!("unexpected arg: {other}");
                    return ExitCode::FAILURE;
                }
                path = Some(other.to_string());
                i += 1;
            }
        }
    }
    let Some(path) = path else {
        eprintln!("usage: ldm_census <file> [--stride N] [--min-dist BYTES] [--min-match BYTES]");
        return ExitCode::FAILURE;
    };

    let t0 = Instant::now();
    let data = fs::read(&path).expect("read input");
    let n = data.len();
    assert!(n < u32::MAX as usize, "probe uses u32 positions (< 4 GiB)");
    if n < min_dist + 8 {
        eprintln!("input smaller than min-dist; nothing to measure");
        return ExitCode::FAILURE;
    }

    // 1. Sample exact 8-byte fingerprints at stride.
    let mut samples: Vec<(u64, u32)> = Vec::with_capacity(n / stride + 1);
    let mut pos = 0usize;
    while pos + 8 <= n {
        let fp = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        samples.push((fp, pos as u32));
        pos += stride;
    }
    let n_samples = samples.len();

    // 2. Sort by (fingerprint, position).
    samples.sort_unstable();
    let t_sorted = t0.elapsed();

    // 3. Candidate pairs: nearest earlier same-fingerprint sample at
    //    distance >= min_dist.
    let mut candidates: Vec<(u32, u32)> = Vec::new(); // (src, dst)
    let mut g = 0usize;
    while g < n_samples {
        let fp = samples[g].0;
        let mut g_end = g + 1;
        while g_end < n_samples && samples[g_end].0 == fp {
            g_end += 1;
        }
        if g_end - g > 1 {
            let mut j = g;
            for k in g + 1..g_end {
                let pk = samples[k].1;
                while j + 1 < k && pk - samples[j + 1].1 >= min_dist as u32 {
                    j += 1;
                }
                if pk - samples[j].1 >= min_dist as u32 {
                    candidates.push((samples[j].1, pk));
                }
            }
        }
        g = g_end;
    }
    drop(samples);
    let n_candidates = candidates.len();

    // 4./5. Verify in dst order, mark coverage, never double-count.
    candidates.sort_unstable_by_key(|&(_, dst)| dst);
    let mut bitmap = Bitmap::new(n);
    let mut covered = 0usize;
    let mut verified_matches = 0usize;
    // log2 histograms over match length and distance (verified, >= min_match)
    let mut len_hist = [0usize; 33];
    let mut dist_hist = [0usize; 33];

    for &(src, dst) in &candidates {
        let (src, dst) = (src as usize, dst as usize);
        if bitmap.get(dst) {
            continue; // this region is already explained by an earlier match
        }
        let fwd = match_forward(&data, src, dst);
        debug_assert!(fwd >= 8);
        let back = match_backward(&data, src, dst);
        let len = fwd + back;
        if len < min_match {
            continue;
        }
        verified_matches += 1;
        len_hist[(len.ilog2() as usize).min(32)] += 1;
        dist_hist[((dst - src).ilog2() as usize).min(32)] += 1;
        covered += bitmap.set_range_count_new(dst - back, dst + fwd);
    }
    let t_total = t0.elapsed();

    println!("file:              {path}");
    println!("input bytes:       {n}");
    println!("params:            stride={stride} min_dist={min_dist} min_match={min_match}");
    println!("samples:           {n_samples}");
    println!("candidate pairs:   {n_candidates}");
    println!("verified matches:  {verified_matches}");
    println!(
        "covered bytes:     {covered} ({:.2}% of input)",
        covered as f64 * 100.0 / n as f64
    );
    println!(
        "time:              sort {:.1}s, total {:.1}s (indicative only)",
        t_sorted.as_secs_f64(),
        t_total.as_secs_f64()
    );
    println!("match length histogram (log2 buckets, verified matches):");
    for (b, &c) in len_hist.iter().enumerate() {
        if c > 0 {
            println!("  2^{b:>2} ({:>10}..): {c}", 1usize << b);
        }
    }
    println!("match distance histogram (log2 buckets, verified matches):");
    for (b, &c) in dist_hist.iter().enumerate() {
        if c > 0 {
            println!("  2^{b:>2} ({:>10}..): {c}", 1usize << b);
        }
    }
    ExitCode::SUCCESS
}
