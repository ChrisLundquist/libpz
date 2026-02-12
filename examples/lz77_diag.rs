/// LZ77 diagnostic: compare lazy vs optimal parsing quality on corpus files.
///
/// For each file, reports:
/// - Number of sequences (Match structs emitted)
/// - Matches vs literals breakdown
/// - Average match length, bytes covered by matches
/// - Raw LZ77 output size (before entropy coding)
/// - gzip compressed size (reference)
///
/// Usage: cargo run --release --example lz77_diag [file ...]
/// If no files given, runs on Canterbury + large corpus.
use std::path::PathBuf;
use std::process::Command;

use pz::lz77::{self, Match};
use pz::optimal;

/// Parse a serialized Match stream and compute statistics.
struct ParseStats {
    total_sequences: usize,
    match_sequences: usize,
    literal_sequences: usize,
    total_match_bytes: usize,
    total_literal_bytes: usize, // each sequence always has 1 literal (the `next` byte)
    match_length_sum: usize,
    min_match_len: u16,
    max_match_len: u16,
    min_offset: u16,
    max_offset: u16,
    offset_sum: u64,
    /// Distribution of match lengths (index = length, value = count)
    length_dist: Vec<usize>,
}

impl ParseStats {
    fn from_lz77_bytes(data: &[u8]) -> Self {
        let num = data.len() / Match::SERIALIZED_SIZE;
        let mut stats = ParseStats {
            total_sequences: num,
            match_sequences: 0,
            literal_sequences: 0,
            total_match_bytes: 0,
            total_literal_bytes: 0,
            match_length_sum: 0,
            min_match_len: u16::MAX,
            max_match_len: 0,
            min_offset: u16::MAX,
            max_offset: 0,
            offset_sum: 0,
            length_dist: vec![0; 260],
        };

        for chunk in data.chunks_exact(Match::SERIALIZED_SIZE) {
            let buf: &[u8; Match::SERIALIZED_SIZE] = chunk.try_into().unwrap();
            let m = Match::from_bytes(buf);

            // Every sequence contributes 1 literal byte (the `next` field)
            stats.total_literal_bytes += 1;

            if m.length > 0 && m.offset > 0 {
                stats.match_sequences += 1;
                stats.total_match_bytes += m.length as usize;
                stats.match_length_sum += m.length as usize;
                if m.length < stats.min_match_len {
                    stats.min_match_len = m.length;
                }
                if m.length > stats.max_match_len {
                    stats.max_match_len = m.length;
                }
                if m.offset < stats.min_offset {
                    stats.min_offset = m.offset;
                }
                if m.offset > stats.max_offset {
                    stats.max_offset = m.offset;
                }
                stats.offset_sum += m.offset as u64;
                let idx = (m.length as usize).min(stats.length_dist.len() - 1);
                stats.length_dist[idx] += 1;
            } else {
                stats.literal_sequences += 1;
            }
        }

        stats
    }

    fn avg_match_len(&self) -> f64 {
        if self.match_sequences == 0 {
            0.0
        } else {
            self.match_length_sum as f64 / self.match_sequences as f64
        }
    }

    fn avg_offset(&self) -> f64 {
        if self.match_sequences == 0 {
            0.0
        } else {
            self.offset_sum as f64 / self.match_sequences as f64
        }
    }

    fn match_coverage(&self, input_len: usize) -> f64 {
        self.total_match_bytes as f64 / input_len as f64 * 100.0
    }
}

fn gzip_size(data: &[u8]) -> usize {
    use std::io::Write;
    let mut child = Command::new("gzip")
        .args(["-c", "-6"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("gzip not found");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(data)
        .expect("write to gzip");
    let output = child.wait_with_output().expect("gzip failed");
    output.stdout.len()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let files: Vec<PathBuf> = if args.is_empty() {
        // Default: Canterbury + large corpus
        let mut f = Vec::new();
        for dir in &["samples/cantrbry", "samples/large"] {
            if let Ok(entries) = std::fs::read_dir(dir) {
                let mut paths: Vec<PathBuf> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_file())
                    .collect();
                paths.sort();
                f.extend(paths);
            }
        }
        f
    } else {
        args.iter().map(PathBuf::from).collect()
    };

    if files.is_empty() {
        eprintln!("No files found. Extract samples first: see CLAUDE.md");
        std::process::exit(1);
    }

    // Header
    println!(
        "{:<20} {:>8} | {:>7} {:>7} {:>6} {:>6} {:>7} {:>7} | {:>7} {:>7} {:>6} {:>6} {:>7} {:>7} | {:>7}",
        "FILE", "SIZE",
        "L-SEQS", "L-MTCH", "L-LIT", "L-AVG", "L-COV%", "L-RAW",
        "O-SEQS", "O-MTCH", "O-LIT", "O-AVG", "O-COV%", "O-RAW",
        "GZIP",
    );
    println!("{}", "-".repeat(170));

    let mut total_input = 0usize;
    let mut total_lazy_raw = 0usize;
    let mut total_opt_raw = 0usize;
    let mut total_gzip = 0usize;
    let mut total_lazy_seqs = 0usize;
    let mut total_opt_seqs = 0usize;

    for path in &files {
        let data = std::fs::read(path).expect("read file");
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let input_len = data.len();

        // Run lazy
        let lazy_out = lz77::compress_lazy(&data).unwrap();
        let lazy_stats = ParseStats::from_lz77_bytes(&lazy_out);

        // Run optimal
        let opt_out = optimal::compress_optimal(&data).unwrap();
        let opt_stats = ParseStats::from_lz77_bytes(&opt_out);

        // gzip reference
        let gz = gzip_size(&data);

        println!(
            "{:<20} {:>8} | {:>7} {:>7} {:>6} {:>6.1} {:>6.1}% {:>7} | {:>7} {:>7} {:>6} {:>6.1} {:>6.1}% {:>7} | {:>7}",
            name,
            input_len,
            lazy_stats.total_sequences,
            lazy_stats.match_sequences,
            lazy_stats.literal_sequences,
            lazy_stats.avg_match_len(),
            lazy_stats.match_coverage(input_len),
            lazy_out.len(),
            opt_stats.total_sequences,
            opt_stats.match_sequences,
            opt_stats.literal_sequences,
            opt_stats.avg_match_len(),
            opt_stats.match_coverage(input_len),
            opt_out.len(),
            gz,
        );

        total_input += input_len;
        total_lazy_raw += lazy_out.len();
        total_opt_raw += opt_out.len();
        total_gzip += gz;
        total_lazy_seqs += lazy_stats.total_sequences;
        total_opt_seqs += opt_stats.total_sequences;
    }

    println!("{}", "-".repeat(170));
    println!(
        "{:<20} {:>8} | {:>7} {:>33} {:>7} | {:>7} {:>33} {:>7} | {:>7}",
        "TOTAL",
        total_input,
        total_lazy_seqs,
        "",
        total_lazy_raw,
        total_opt_seqs,
        "",
        total_opt_raw,
        total_gzip,
    );

    println!();
    println!("Legend:");
    println!("  L-* = Lazy parser, O-* = Optimal (backward DP) parser");
    println!("  SEQS  = total Match structs emitted");
    println!("  MTCH  = sequences with length>0 (back-references)");
    println!("  LIT   = sequences with length=0 (literal-only)");
    println!("  AVG   = average match length");
    println!("  COV%  = % of input bytes covered by matches");
    println!("  RAW   = raw LZ77 output bytes (SEQS * 5, before entropy coding)");
    println!("  GZIP  = gzip -6 compressed size (reference)");

    // Detailed match-length distribution for largest file
    if let Some(path) = files
        .iter()
        .max_by_key(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
    {
        let data = std::fs::read(path).unwrap();
        let name = path.file_name().unwrap_or_default().to_string_lossy();

        let lazy_out = lz77::compress_lazy(&data).unwrap();
        let lazy_stats = ParseStats::from_lz77_bytes(&lazy_out);
        let opt_out = optimal::compress_optimal(&data).unwrap();
        let opt_stats = ParseStats::from_lz77_bytes(&opt_out);

        println!();
        println!("=== Match length distribution for {} ===", name);
        println!("{:>6}  {:>8}  {:>8}", "LEN", "LAZY", "OPTIMAL");
        for len in 3..=20 {
            let lc = lazy_stats.length_dist.get(len).copied().unwrap_or(0);
            let oc = opt_stats.length_dist.get(len).copied().unwrap_or(0);
            if lc > 0 || oc > 0 {
                println!("{:>6}  {:>8}  {:>8}", len, lc, oc);
            }
        }
        // Bucket the rest
        let lazy_long: usize = lazy_stats.length_dist.iter().skip(21).sum();
        let opt_long: usize = opt_stats.length_dist.iter().skip(21).sum();
        if lazy_long > 0 || opt_long > 0 {
            println!("{:>6}  {:>8}  {:>8}", "21+", lazy_long, opt_long);
        }

        println!();
        println!("=== Offset stats for {} ===", name);
        println!(
            "  Lazy:    min={}, max={}, avg={:.0}",
            lazy_stats.min_offset,
            lazy_stats.max_offset,
            lazy_stats.avg_offset()
        );
        println!(
            "  Optimal: min={}, max={}, avg={:.0}",
            opt_stats.min_offset,
            opt_stats.max_offset,
            opt_stats.avg_offset()
        );
    }
}
