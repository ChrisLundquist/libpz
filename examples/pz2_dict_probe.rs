//! Cross-block dictionary spike (future-pipelines-roadmap §3 spike #2,
//! clean-slate-codec P2 phase 2): measure the ratio CEILING of letting each
//! 2 MiB pz2 block reference a sliding prefix of all preceding bytes
//! (capped at D), with the shipped greedy parse.
//!
//! This is the measurement that decides whether a dict tier is worth
//! container complexity. Decision rule from the roadmap: if the delta is
//! small everywhere, the 2 MiB window already captured the redundancy —
//! kill the tier. Note this measures the ceiling (full sliding prefix);
//! a production fixed-dict region captures less.
//!
//! Encode here is deliberately wasteful (re-tokenizes the prefix for every
//! block — spike only); round-trips are verified via decode_with_prefix.
//!
//! Usage: cargo run --release --example pz2_dict_probe -- <files...>

use std::time::Instant;

use pz::lzseq::SeqConfig;

const BLOCK: usize = 2 * 1024 * 1024;
const DICTS: [usize; 3] = [0, 4 << 20, 16 << 20];

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // --head: fixed head-of-file dictionary (the parallel-decode-friendly
    // production shape: one Arc-shared region, 2-wave decode) instead of
    // the sliding prefix (the serializing ceiling measurement).
    let head_mode = if let Some(i) = args.iter().position(|a| a == "--head") {
        args.remove(i);
        true
    } else {
        false
    };
    // --seg: like --head but the dict is the first D bytes of each 32 MiB
    // SEGMENT (P2's segment tier) instead of the file head — heterogeneous
    // inputs (the blob) get locally-relevant dicts.
    let seg_mode = if let Some(i) = args.iter().position(|a| a == "--seg") {
        args.remove(i);
        true
    } else {
        false
    };
    // --frozen: per-segment head dict via the production frozen finder
    // (FrozenDict built once per segment, parse starts at the dict
    // boundary) instead of the spike's full dict+block re-parse. Same wire,
    // same decoder; encode cost is the point.
    let frozen_mode = if let Some(i) = args.iter().position(|a| a == "--frozen") {
        args.remove(i);
        true
    } else {
        false
    };
    const SEG: usize = 32 << 20;
    if args.is_empty() {
        eprintln!("usage: pz2_dict_probe [--head] <files...>");
        std::process::exit(2);
    }
    println!(
        "{:>14} {:>7} | {:>8} {:>7} | {:>8}   ({})",
        "file",
        "dict",
        "ratio %",
        "delta",
        "enc s",
        if frozen_mode {
            "per-segment FROZEN finder"
        } else if seg_mode {
            "per-segment head dict"
        } else if head_mode {
            "fixed head dict"
        } else {
            "sliding prefix"
        }
    );

    for path in &args {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path);
        let mut base_pct = 0.0f64;

        for d in DICTS {
            let config = SeqConfig {
                max_window: (d + BLOCK).next_power_of_two(),
                greedy: true,
                ..SeqConfig::default()
            };
            let t = Instant::now();
            let mut size = 0usize;
            let mut start = 0usize;
            // Frozen mode: build each segment's dict tables ONCE (the
            // production shape — Arc-shared across workers).
            let mut frozen: Option<(usize, std::sync::Arc<pz::lz77::FrozenDict>, usize)> = None;
            while start < data.len() {
                let end = (start + BLOCK).min(data.len());
                let (enc, prefix): (Vec<u8>, &[u8]) = if frozen_mode {
                    let seg_base = start - (start % SEG);
                    let dlen = d.min(start - seg_base);
                    let dict = &data[seg_base..seg_base + dlen];
                    let reuse = matches!(frozen, Some((b, _, l)) if b == seg_base && l == dlen);
                    if !reuse {
                        frozen = Some((
                            seg_base,
                            std::sync::Arc::new(pz::lz77::FrozenDict::build(
                                dict,
                                config.hash_prefix_len,
                            )),
                            dlen,
                        ));
                    }
                    let tables = &frozen.as_ref().unwrap().1;
                    let mut arena = Vec::with_capacity(dlen + (end - start));
                    arena.extend_from_slice(dict);
                    arena.extend_from_slice(&data[start..end]);
                    (
                        pz::pz2::encode_with_frozen_dict(&arena, tables, &config).expect("encode"),
                        dict,
                    )
                } else if head_mode || seg_mode {
                    // Dict = first min(d, start - base) bytes of the file
                    // (--head) or of the block's 32 MiB segment (--seg);
                    // blocks inside the dict region parse cold (they ARE
                    // the dict).
                    let base = if seg_mode { start - (start % SEG) } else { 0 };
                    let dlen = d.min(start - base);
                    let dict = &data[base..base + dlen];
                    let mut buf = Vec::with_capacity(dlen + (end - start));
                    buf.extend_from_slice(dict);
                    buf.extend_from_slice(&data[start..end]);
                    (
                        pz::pz2::encode_with_prefix(&buf, dlen, &config).expect("encode"),
                        dict,
                    )
                } else {
                    let pstart = start.saturating_sub(d);
                    (
                        pz::pz2::encode_with_prefix(&data[pstart..end], start - pstart, &config)
                            .expect("encode"),
                        &data[pstart..start],
                    )
                };
                let dec = pz::pz2::decode_with_prefix(&enc, prefix, end - start).expect("decode");
                assert_eq!(dec, &data[start..end], "round-trip mismatch at {start}");
                size += enc.len();
                start = end;
            }
            let pct = 100.0 * size as f64 / data.len() as f64;
            if d == 0 {
                base_pct = pct;
            }
            println!(
                "{:>14} {:>5}Mi | {:>8.3} {:>+7.3} | {:>8.1}",
                name,
                d >> 20,
                pct,
                pct - base_pct,
                t.elapsed().as_secs_f64()
            );
        }
    }
}
