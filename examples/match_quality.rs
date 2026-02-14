/// Match quality experiment: GPU lazy vs CPU lazy LZ77 match finding.
///
/// The block_size_experiment showed WebGPU has dramatically worse compression
/// ratios than CPU. Both paths use "lazy" matching, but the GPU's parallel
/// lazy emulation may produce worse matches than the CPU's sequential lazy.
///
/// This experiment compares:
///   - Number of matches (fewer = better, each match covers more input)
///   - Average match length (longer = better)
///   - Total literals emitted (fewer = better)
///   - Total serialized size (smaller = better compression)
///
/// Usage:
///   cargo run --example match_quality --release --features webgpu
use std::path::Path;

fn load_data(size: usize) -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                if decompressed.len() >= size {
                    return decompressed[..size].to_vec();
                }
                let mut data = Vec::with_capacity(size);
                while data.len() < size {
                    let remaining = size - data.len();
                    let chunk = remaining.min(decompressed.len());
                    data.extend_from_slice(&decompressed[..chunk]);
                }
                return data;
            }
        }
    }
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let full = pattern.repeat((size / pattern.len()) + 1);
    full[..size].to_vec()
}

struct MatchStats {
    label: String,
    input_size: usize,
    num_matches: usize,
    total_match_bytes: u64,
    total_literal_bytes: u64,
    avg_match_length: f64,
    max_match_length: u16,
    serialized_size: usize,
    ratio: f64,
    // Distribution: how many matches at each length bucket
    len_0: usize,        // literals only (length=0)
    len_3_7: usize,      // short matches
    len_8_31: usize,     // medium matches
    len_32_127: usize,   // long matches
    len_128_plus: usize, // very long matches
}

fn analyze_matches(label: &str, matches: &[pz::lz77::Match], input_size: usize) -> MatchStats {
    let num_matches = matches.len();
    let mut total_match_bytes: u64 = 0;
    let mut total_literal_bytes: u64 = 0;
    let mut max_match_length: u16 = 0;
    let mut len_0 = 0usize;
    let mut len_3_7 = 0usize;
    let mut len_8_31 = 0usize;
    let mut len_32_127 = 0usize;
    let mut len_128_plus = 0usize;

    for m in matches {
        if m.length == 0 {
            total_literal_bytes += 1;
            len_0 += 1;
        } else {
            total_match_bytes += m.length as u64;
            total_literal_bytes += 1; // the 'next' byte
            if m.length > max_match_length {
                max_match_length = m.length;
            }
            match m.length {
                0..=2 => len_0 += 1, // shouldn't happen
                3..=7 => len_3_7 += 1,
                8..=31 => len_8_31 += 1,
                32..=127 => len_32_127 += 1,
                _ => len_128_plus += 1,
            }
        }
    }

    let serialized_size = num_matches * pz::lz77::Match::SERIALIZED_SIZE;
    let avg_match_length = if num_matches > len_0 {
        total_match_bytes as f64 / (num_matches - len_0) as f64
    } else {
        0.0
    };

    MatchStats {
        label: label.to_string(),
        input_size,
        num_matches,
        total_match_bytes,
        total_literal_bytes,
        avg_match_length,
        max_match_length,
        serialized_size,
        ratio: serialized_size as f64 / input_size as f64,
        len_0,
        len_3_7,
        len_8_31,
        len_32_127,
        len_128_plus,
    }
}

fn print_stats(stats: &[MatchStats]) {
    println!(
        "\n{:<20} {:<8} {:<10} {:<10} {:<10} {:<10} {:<8} {:<8} {:<8}",
        "Method", "Input", "Matches", "AvgLen", "MaxLen", "Literals", "Ratio", "Short%", "Long%"
    );
    println!("{}", "=".repeat(100));

    for s in stats {
        let non_literal = s.num_matches - s.len_0;
        let short_pct = if non_literal > 0 {
            100.0 * s.len_3_7 as f64 / non_literal as f64
        } else {
            0.0
        };
        let long_pct = if non_literal > 0 {
            100.0 * (s.len_32_127 + s.len_128_plus) as f64 / non_literal as f64
        } else {
            0.0
        };
        println!(
            "{:<20} {:<8} {:<10} {:<10.1} {:<10} {:<10} {:<8.4} {:<8.1} {:<8.1}",
            s.label,
            format_size(s.input_size),
            s.num_matches,
            s.avg_match_length,
            s.max_match_length,
            s.total_literal_bytes,
            s.ratio,
            short_pct,
            long_pct,
        );
    }
}

fn print_distribution(stats: &[MatchStats]) {
    println!(
        "\n{:<20} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10}",
        "Method", "Input", "Lit(0)", "Short(3-7)", "Med(8-31)", "Long(32+)", "VLong(128+)"
    );
    println!("{}", "=".repeat(80));

    for s in stats {
        println!(
            "{:<20} {:<8} {:<10} {:<10} {:<10} {:<10} {:<10}",
            s.label,
            format_size(s.input_size),
            s.len_0,
            s.len_3_7,
            s.len_8_31,
            s.len_32_127,
            s.len_128_plus,
        );
    }
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1_048_576 {
        format!("{}MB", bytes / 1_048_576)
    } else if bytes >= 1024 {
        format!("{}KB", bytes / 1024)
    } else {
        format!("{}B", bytes)
    }
}

fn main() {
    println!("=== LZ77 Match Quality: GPU vs CPU ===");
    println!();

    let sizes = [65536, 131072, 262144, 524288, 1048576];
    let mut all_stats = Vec::new();

    for &size in &sizes {
        let data = load_data(size);

        // --- CPU lazy ---
        let cpu_matches = pz::lz77::compress_lazy_to_matches(&data).unwrap();

        // Verify round-trip
        {
            let mut serialized = Vec::with_capacity(cpu_matches.len() * 5);
            for m in &cpu_matches {
                serialized.extend_from_slice(&m.to_bytes());
            }
            let decompressed = pz::lz77::decompress(&serialized).unwrap();
            assert_eq!(
                decompressed.len(),
                data.len(),
                "CPU round-trip length mismatch"
            );
            assert_eq!(&decompressed[..], &data[..], "CPU round-trip data mismatch");
        }

        all_stats.push(analyze_matches("CPU lazy", &cpu_matches, size));

        // --- WebGPU lazy ---
        #[cfg(feature = "webgpu")]
        {
            use pz::webgpu::WebGpuEngine;
            // Re-create engine per size to avoid stale state
            match WebGpuEngine::new() {
                Ok(engine) => {
                    if size >= pz::webgpu::MIN_GPU_INPUT_SIZE {
                        let gpu_matches = engine.find_matches(&data).unwrap();

                        // Verify round-trip
                        {
                            let mut serialized = Vec::with_capacity(gpu_matches.len() * 5);
                            for m in &gpu_matches {
                                serialized.extend_from_slice(&m.to_bytes());
                            }
                            let decompressed = pz::lz77::decompress(&serialized).unwrap();
                            assert_eq!(
                                decompressed.len(),
                                data.len(),
                                "GPU lazy round-trip length mismatch at size {}",
                                size
                            );
                            assert_eq!(
                                &decompressed[..],
                                &data[..],
                                "GPU lazy round-trip data mismatch at size {}",
                                size
                            );
                        }

                        all_stats.push(analyze_matches("GPU lazy", &gpu_matches, size));

                        // Also test GPU greedy for comparison
                        let gpu_greedy = engine.find_matches_greedy(&data).unwrap();
                        {
                            let mut serialized = Vec::with_capacity(gpu_greedy.len() * 5);
                            for m in &gpu_greedy {
                                serialized.extend_from_slice(&m.to_bytes());
                            }
                            let decompressed = pz::lz77::decompress(&serialized).unwrap();
                            assert_eq!(
                                decompressed.len(),
                                data.len(),
                                "GPU greedy round-trip length mismatch at size {}",
                                size
                            );
                        }
                        all_stats.push(analyze_matches("GPU greedy", &gpu_greedy, size));
                    }
                }
                Err(e) => {
                    eprintln!("WebGPU unavailable: {:?}", e);
                }
            }
        }
    }

    print_stats(&all_stats);
    print_distribution(&all_stats);

    // Print CSV
    println!("\n--- CSV ---");
    println!("method,input_bytes,num_matches,avg_match_len,max_match_len,total_match_bytes,total_literal_bytes,serialized_bytes,ratio,lit_only,short_3_7,med_8_31,long_32_127,vlong_128_plus");
    for s in &all_stats {
        println!(
            "{},{},{},{:.2},{},{},{},{},{:.6},{},{},{},{},{}",
            s.label,
            s.input_size,
            s.num_matches,
            s.avg_match_length,
            s.max_match_length,
            s.total_match_bytes,
            s.total_literal_bytes,
            s.serialized_size,
            s.ratio,
            s.len_0,
            s.len_3_7,
            s.len_8_31,
            s.len_32_127,
            s.len_128_plus,
        );
    }

    println!("\nDone.");
}
