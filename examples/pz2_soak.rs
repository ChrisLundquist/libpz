//! pz2 decode-safety soak (clean-slate-codec.md §7 hardening gate).
//!
//! Miri is unavailable on this box (no nightly toolchain, rustup shims
//! broken), so this is the compensating control for the unsafe splice /
//! wildcopy paths in `pz2::decode`: a time-budgeted, deterministic,
//! seed-reportable soak over three attack surfaces:
//!
//! 1. **Round-trip**: structured random inputs (splice-of-history, periodic
//!    at every small period, runs, random, text-ish, u16 walks, block-size
//!    boundary cases) must encode → decode byte-identically.
//! 2. **Mutation**: valid encodings with bit flips, byte stomps,
//!    truncations and length-field corruption must never panic or hang —
//!    they either error or return garbage of the right length (the codec
//!    has no checksum; the container CRC owns integrity).
//! 3. **Garbage**: random byte soup fed to `decode` with arbitrary
//!    `orig_len` must error or return, never panic.
//!
//! Every decode runs under `catch_unwind`; a panic is a failure and prints
//! the reproducing seed. Run in debug too (checked arithmetic) — sizes are
//! scaled down so a debug pass stays fast.
//!
//! Usage: pz2_soak [seconds]   (default 30)

use std::panic::{self, AssertUnwindSafe};
use std::time::{Duration, Instant};

struct Rng(u64);

impl Rng {
    #[inline]
    fn next(&mut self) -> u32 {
        // xorshift64* — deterministic, full-period, no deps.
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u32
    }
    #[inline]
    fn below(&mut self, m: usize) -> usize {
        (self.next() as usize) % m.max(1)
    }
}

fn gen_input(rng: &mut Rng, max_size: usize) -> Vec<u8> {
    let kind = rng.below(8);
    let target = match rng.below(4) {
        // Bias toward small (header/tail paths) but include block-scale.
        0 => 1 + rng.below(64),
        1 => 64 + rng.below(4096),
        2 => 4096 + rng.below(max_size / 4),
        _ => max_size.saturating_sub(rng.below(64)).max(1),
    };
    let mut out = Vec::with_capacity(target + 64);
    match kind {
        // Splice-of-history: literals + matches at random offsets/lengths.
        0 | 1 => {
            while out.len() < target {
                if out.is_empty() || rng.below(2) == 0 {
                    for _ in 0..=rng.below(16) {
                        out.push(rng.next() as u8);
                    }
                } else {
                    let off = 1 + rng.below(out.len().min(70_000));
                    let len = 1 + rng.below(300);
                    for _ in 0..len {
                        let b = out[out.len() - off];
                        out.push(b);
                    }
                }
            }
        }
        // Periodic: every overlap-copy path (offset 1, 2..16, >=16).
        2 => {
            let period = 1 + rng.below(48);
            let pattern: Vec<u8> = (0..period).map(|_| rng.next() as u8).collect();
            while out.len() < target {
                out.extend_from_slice(&pattern);
            }
        }
        // Long runs with interruptions (offset-1 splat + rep offsets).
        3 => {
            while out.len() < target {
                let b = rng.next() as u8;
                let run = 1 + rng.below(5000);
                out.extend(std::iter::repeat_n(b, run));
                out.push(rng.next() as u8);
            }
        }
        // Random (raw-literal mode, lane tails).
        4 => {
            for _ in 0..target {
                out.push(rng.next() as u8);
            }
        }
        // Text-ish: skewed literal distribution (deep Huffman trees).
        5 => {
            const WORDS: &[&str] = &["the ", "of ", "and ", "compression ", "entropy ", "a "];
            while out.len() < target {
                out.extend_from_slice(WORDS[rng.below(WORDS.len())].as_bytes());
            }
        }
        // u16 random walk (numeric-shaped, moderate entropy).
        6 => {
            let mut v: u16 = rng.next() as u16;
            while out.len() < target {
                v = v.wrapping_add((rng.below(33) as i32 - 16) as u16);
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        // Near-constant with rare deviations (CODES_CONST paths).
        _ => {
            let b = rng.next() as u8;
            for i in 0..target {
                out.push(if i % 977 == 0 { rng.next() as u8 } else { b });
            }
        }
    }
    out.truncate(target);
    out
}

/// Decode under catch_unwind; panics are failures, results are ignored.
fn must_not_panic(seed: u64, what: &str, data: &[u8], orig_len: usize) {
    let r = panic::catch_unwind(AssertUnwindSafe(|| {
        let _ = pz::pz2::decode(data, orig_len);
    }));
    if r.is_err() {
        eprintln!("PANIC in pz2::decode ({what}), seed {seed:#x}, orig_len {orig_len}");
        std::process::exit(1);
    }
}

fn main() {
    let secs: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(30);
    let budget = Duration::from_secs(secs);
    // Debug builds are ~20x slower on encode; shrink block sizes so the
    // soak still cycles many cases.
    let max_size = if cfg!(debug_assertions) {
        192 * 1024
    } else {
        1 << 20
    };

    let t0 = Instant::now();
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let (mut rt, mut mu, mut gb) = (0u64, 0u64, 0u64);

    while t0.elapsed() < budget {
        let seed = rng.0;

        // 1. Round-trip.
        let input = gen_input(&mut rng, max_size);
        let enc = pz::pz2::encode(&input).expect("encode must succeed");
        let dec = pz::pz2::decode(&enc, input.len())
            .unwrap_or_else(|e| panic!("decode of valid stream failed: {e:?}, seed {seed:#x}"));
        assert_eq!(dec, input, "round-trip mismatch, seed {seed:#x}");
        rt += 1;

        // 2. Mutations of the valid encoding (and wrong orig_len).
        for _ in 0..40 {
            let mut bad = enc.clone();
            match rng.below(5) {
                0 => {
                    let i = rng.below(bad.len());
                    bad[i] ^= 1 << rng.below(8);
                }
                1 => {
                    let i = rng.below(bad.len());
                    bad[i] = rng.next() as u8;
                }
                2 => bad.truncate(rng.below(bad.len() + 1)),
                3 => {
                    // Stomp an early length/header field hard.
                    let i = rng.below(bad.len().min(48));
                    let v = rng.next().to_le_bytes();
                    for (k, &b) in v.iter().enumerate() {
                        if i + k < bad.len() {
                            bad[i + k] = b;
                        }
                    }
                }
                _ => {
                    // Duplicate a random slice over another (structure-preserving-ish).
                    if bad.len() >= 8 {
                        let src = rng.below(bad.len() - 4);
                        let dst = rng.below(bad.len() - 4);
                        let n = 1 + rng.below((bad.len() - src.max(dst)).min(64));
                        bad.copy_within(src..src + n, dst);
                    }
                }
            }
            let ol = match rng.below(4) {
                0 => input.len(),
                1 => input.len().saturating_sub(1 + rng.below(64)),
                2 => input.len() + 1 + rng.below(64),
                _ => rng.below(2 * max_size),
            };
            must_not_panic(seed, "mutated", &bad, ol);
            mu += 1;
        }

        // 3. Pure garbage.
        for _ in 0..8 {
            let n = rng.below(2048);
            let junk: Vec<u8> = (0..n).map(|_| rng.next() as u8).collect();
            must_not_panic(seed, "garbage", &junk, rng.below(2 * max_size));
            gb += 1;
        }
    }

    println!(
        "pz2 soak PASS: {rt} round-trips, {mu} mutated decodes, {gb} garbage decodes in {:.1}s ({})",
        t0.elapsed().as_secs_f64(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    );
}
