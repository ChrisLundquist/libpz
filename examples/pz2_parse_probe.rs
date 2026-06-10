//! pz2 lazy-vs-greedy probe: per-file ratio under both parses (2 MiB
//! blocks, window = block) alongside the analysis profile, to derive a
//! per-block auto-parse discriminator from data instead of guesswork.
//!
//! Usage: cargo run --release --example pz2_parse_probe -- <files...>

use std::time::Instant;

use pz::lzseq::SeqConfig;

const BLOCK: usize = 2 * 1024 * 1024;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: pz2_parse_probe <files...>");
        std::process::exit(2);
    }
    println!(
        "{:>10} | {:>8} {:>8} {:>7} | {:>8} {:>8} | {:>5} {:>5} {:>5} {:>5}",
        "file", "lazy %", "greedy %", "delta", "lazyMB/s", "grdyMB/s", "ent", "match", "run", "ac1"
    );

    for path in &args {
        let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let name = path.rsplit('/').next().unwrap_or(path);
        let mb = data.len() as f64 / 1048576.0;
        let p = pz::analysis::analyze(&data);

        let mut sizes = [0usize; 2];
        let mut speeds = [0f64; 2];
        for (i, greedy) in [false, true].into_iter().enumerate() {
            let config = SeqConfig {
                max_window: BLOCK,
                greedy,
                ..SeqConfig::default()
            };
            let t = Instant::now();
            sizes[i] = data
                .chunks(BLOCK)
                .map(|b| {
                    pz::pz2::encode_with_config(b, &config)
                        .expect("encode")
                        .len()
                })
                .sum();
            speeds[i] = mb / t.elapsed().as_secs_f64();
        }

        let lazy_pct = 100.0 * sizes[0] as f64 / data.len() as f64;
        let greedy_pct = 100.0 * sizes[1] as f64 / data.len() as f64;
        println!(
            "{:>10} | {:>8.3} {:>8.3} {:>+7.3} | {:>8.0} {:>8.0} | {:>5.2} {:>5.2} {:>5.2} {:>5.2}",
            name,
            lazy_pct,
            greedy_pct,
            greedy_pct - lazy_pct,
            speeds[0],
            speeds[1],
            p.byte_entropy,
            p.match_density,
            p.run_ratio,
            p.autocorrelation_lag1,
        );
    }
}
