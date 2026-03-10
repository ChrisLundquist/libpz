use pz::pipeline::{self, Pipeline};
use std::time::Instant;

fn main() {
    println!("\n=== Pipeline Compression Comparison ===\n");

    // Test with different data patterns
    test_pipeline_comparison("repetitive", generate_repetitive(256 * 1024));
    test_pipeline_comparison("sequential", generate_sequential(256 * 1024));
    test_pipeline_comparison("structured", generate_structured(256 * 1024));
}

fn test_pipeline_comparison(name: &str, data: Vec<u8>) {
    println!("Test data: {} ({} bytes)", name, data.len());
    println!("{:-<80}", "");

    let pipelines = vec![
        ("Lzf (LzSeq+FSE)", Pipeline::Lzf),
        ("LzSeqR (LzSeq+rANS)", Pipeline::LzSeqR),
        ("Deflate (LZ77+Huffman)", Pipeline::Deflate),
    ];

    println!(
        "{:<25} {:<15} {:<15} {:<15}",
        "Pipeline", "Compressed", "Ratio %", "Time (ms)"
    );
    println!("{:-<70}", "");

    for (label, pipeline) in pipelines {
        let start = Instant::now();
        match pipeline::compress(&data, pipeline) {
            Ok(compressed) => {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let ratio = (compressed.len() as f64 / data.len() as f64) * 100.0;
                println!(
                    "{:<25} {:<15} {:<15.2} {:<15.3}",
                    label,
                    format!("{} bytes", compressed.len()),
                    ratio,
                    elapsed
                );
            }
            Err(e) => {
                println!("{:<25} ERROR: {}", label, e);
            }
        }
    }
    println!();
}

fn generate_repetitive(size: usize) -> Vec<u8> {
    let pattern =
        b"The quick brown fox jumps over the lazy dog. This is a test pattern for compression. ";
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = remaining.min(pattern.len());
        data.extend_from_slice(&pattern[..chunk]);
    }
    data
}

fn generate_sequential(size: usize) -> Vec<u8> {
    (0..size).map(|i| (i % 256) as u8).collect()
}

fn generate_structured(size: usize) -> Vec<u8> {
    let json_pattern = br#"{"key":"value","number":123,"nested":{"field":"data"}}"#;
    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = remaining.min(json_pattern.len());
        data.extend_from_slice(&json_pattern[..chunk]);
    }
    data
}
