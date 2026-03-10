// Evaluate SortLZ+LzSeq hybrid pipeline vs alternatives.
// Compares: LzSeqR (hashchain), LzSeqR (sortlz), LzSeqH (hashchain), LzSeqH (sortlz),
//           SortLz (native), Bw
use std::time::Instant;

use pz::pipeline::{self, CompressOptions, MatchFinder, Pipeline};

struct Result {
    label: &'static str,
    ratio: f64,
    compress_mbs: f64,
    decompress_mbs: f64,
}

fn bench_pipeline(
    data: &[u8],
    pipeline: Pipeline,
    match_finder: MatchFinder,
    label: &'static str,
) -> Option<Result> {
    let opts = CompressOptions {
        threads: 1,
        match_finder,
        ..Default::default()
    };

    let t = Instant::now();
    let compressed = pipeline::compress_with_options(data, pipeline, &opts).ok()?;
    let compress_ms = t.elapsed().as_secs_f64() * 1000.0;

    let t = Instant::now();
    let decompressed = pipeline::decompress(&compressed).ok()?;
    let decompress_ms = t.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(
        decompressed.len(),
        data.len(),
        "roundtrip mismatch for {}",
        label
    );

    let mb = data.len() as f64 / 1048576.0;
    Some(Result {
        label,
        ratio: compressed.len() as f64 / data.len() as f64 * 100.0,
        compress_mbs: mb / (compress_ms / 1000.0),
        decompress_mbs: mb / (decompress_ms / 1000.0),
    })
}

fn main() {
    let files = [
        ("samples/silesia/dickens", "dickens"),
        ("samples/cantrbry/alice29.txt", "alice29"),
        ("samples/large/E.coli", "E.coli"),
        ("samples/large/world192.txt", "world192"),
    ];

    let configs: Vec<(Pipeline, MatchFinder, &str)> = vec![
        (Pipeline::Bw, MatchFinder::HashChain, "bw"),
        (Pipeline::LzSeqR, MatchFinder::HashChain, "lzseqr-hc"),
        (Pipeline::LzSeqR, MatchFinder::SortLz, "lzseqr-slz"),
        (Pipeline::LzSeqH, MatchFinder::HashChain, "lzseqh-hc"),
        (Pipeline::LzSeqH, MatchFinder::SortLz, "lzseqh-slz"),
        (Pipeline::SortLz, MatchFinder::SortLz, "sortlz"),
    ];

    for (path, file_label) in &files {
        match std::fs::read(path) {
            Ok(data) => {
                let mb = data.len() as f64 / 1048576.0;
                println!("{} ({:.1} MB):", file_label, mb);
                println!(
                    "  {:<14} {:>7} {:>10} {:>10}",
                    "pipeline", "ratio", "comp MB/s", "dec MB/s"
                );
                println!("  {}", "-".repeat(45));

                for (pipeline, mf, label) in &configs {
                    match bench_pipeline(&data, *pipeline, *mf, label) {
                        Some(r) => println!(
                            "  {:<14} {:>6.1}% {:>9.1} {:>9.1}",
                            r.label, r.ratio, r.compress_mbs, r.decompress_mbs
                        ),
                        None => println!("  {:<14} FAILED", label),
                    }
                }
                println!();
            }
            Err(e) => println!("{}: {}", file_label, e),
        }
    }
}
