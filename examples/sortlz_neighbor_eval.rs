/// Evaluate 8-byte sort key + neighbor scanning vs 4-byte same-cluster matching.
///
/// Compares match quality and compression ratio across several strategies:
/// - 4b-c8:   Current baseline (4-byte sort, 8 candidates, same-cluster)
/// - 4b-c32:  More candidates (4-byte sort, 32 candidates, same-cluster)
/// - 8b-n16:  8-byte sort, unbounded neighbor radius 16
/// - 8b-n32:  8-byte sort, unbounded neighbor radius 32
/// - 8b-b16:  8-byte sort, bounded to 4-byte cluster, radius 16
/// - 8b-b32:  8-byte sort, bounded to 4-byte cluster, radius 32
/// - 8b-b64:  8-byte sort, bounded to 4-byte cluster, radius 64
use std::time::Instant;

use pz::sortlz::{self, SortLzConfig};

struct MatchStats {
    num_matches: usize,
    coverage_pct: f64,
    avg_match_len: f64,
}

fn compute_stats(matches: &[Option<(u16, u16)>]) -> MatchStats {
    let mut num_matches = 0usize;
    let mut matched_bytes = 0u64;
    for (_offset, length) in matches.iter().flatten() {
        num_matches += 1;
        matched_bytes += *length as u64;
    }
    let coverage_pct = matched_bytes as f64 / matches.len() as f64 * 100.0;
    let avg_match_len = if num_matches > 0 {
        matched_bytes as f64 / num_matches as f64
    } else {
        0.0
    };
    MatchStats {
        num_matches,
        coverage_pct,
        avg_match_len,
    }
}

struct VariantResult {
    name: &'static str,
    stats: MatchStats,
    ratio: f64,
    find_ms: f64,
}

/// Variant config: (name, max_candidates, sort_8byte, neighbor_radius, cluster_bounded)
type VariantConfig = (&'static str, usize, bool, usize, bool);

fn run_variant(input: &[u8], cfg: &VariantConfig) -> Option<VariantResult> {
    let (name, max_candidates, sort_8byte, neighbor_radius, cluster_bounded) = *cfg;

    let config = SortLzConfig {
        max_candidates,
        ..SortLzConfig::default()
    };

    // Find matches
    let t = Instant::now();
    let matches = if sort_8byte {
        sortlz::find_matches_neighbor(input, &config, neighbor_radius, cluster_bounded)
    } else {
        sortlz::find_matches(input, &config)
    };
    let find_ms = t.elapsed().as_secs_f64() * 1000.0;

    let stats = compute_stats(&matches);

    // Compress using SortLZ pipeline with these matches
    let compressed = sortlz::compress_with_matches(input, matches, &config).ok()?;
    let ratio = compressed.len() as f64 / input.len() as f64 * 100.0;

    // Verify roundtrip
    let decoded = sortlz::decompress(&compressed, input.len()).ok()?;
    assert_eq!(decoded.len(), input.len(), "roundtrip length mismatch");
    assert_eq!(decoded, input, "roundtrip data mismatch for {}", name);

    Some(VariantResult {
        name,
        stats,
        ratio,
        find_ms,
    })
}

fn main() {
    let files: Vec<(&str, &str)> = vec![
        ("samples/cantrbry/alice29.txt", "alice29"),
        ("samples/silesia/dickens", "dickens"),
        ("samples/large/E.coli", "E.coli"),
        ("samples/large/world192.txt", "world192"),
    ];

    // (name, max_candidates, sort_8byte, neighbor_radius, cluster_bounded)
    let variants: Vec<VariantConfig> = vec![
        ("4b-c8", 8, false, 0, false),
        ("4b-c32", 32, false, 0, false),
        ("4b-c64", 64, false, 0, false),
        ("4b-c128", 128, false, 0, false),
        ("8b-n16", 8, true, 16, false),
        ("8b-b8", 8, true, 65536, true),
        ("8b-b32", 32, true, 65536, true),
        ("8b-b64", 64, true, 65536, true),
        ("8b-b128", 128, true, 65536, true),
    ];

    for (path, file_label) in &files {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                println!("{}: {} (skipping)", file_label, e);
                continue;
            }
        };

        let kb = data.len() as f64 / 1024.0;
        println!("\n{} ({:.0} KB, {:.1} MB):", file_label, kb, kb / 1024.0);
        println!(
            "  {:<10} {:>7} {:>8} {:>8} {:>8} {:>9}",
            "variant", "ratio", "matches", "cover%", "avg_len", "find_ms"
        );
        println!("  {}", "-".repeat(55));

        for cfg in &variants {
            match run_variant(&data, cfg) {
                Some(r) => {
                    println!(
                        "  {:<10} {:>6.1}% {:>8} {:>7.1}% {:>8.1} {:>8.1}",
                        r.name,
                        r.ratio,
                        r.stats.num_matches,
                        r.stats.coverage_pct,
                        r.stats.avg_match_len,
                        r.find_ms,
                    );
                }
                None => println!("  {:<10} FAILED", cfg.0),
            }
        }
    }

    println!("\nLegend:");
    println!("  4b-cN  = 4-byte sort, N max candidates, same-cluster only");
    println!("  8b-nN  = 8-byte sort, N neighbor radius, unbounded (cross-cluster)");
    println!("  8b-bN  = 8-byte sort, N max candidates, bounded to 4-byte cluster");
    println!("  ratio  = compressed/original (lower = better)");
    println!("  cover% = matched bytes / total bytes");
    println!("  avg_len = average match length in bytes");
}
