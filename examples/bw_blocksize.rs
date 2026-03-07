// Test BW pipeline compression ratio at different block sizes.
use pz::pipeline::{self, CompressOptions, Pipeline};

fn test_block_size(data: &[u8], block_size: usize, _label: &str) {
    let opts = CompressOptions {
        threads: 1,
        block_size,
        ..Default::default()
    };
    let compressed = pipeline::compress_with_options(data, Pipeline::Bw, &opts).unwrap();
    let ratio = compressed.len() as f64 / data.len() as f64 * 100.0;
    println!(
        "  block={:>6}KB  ratio={:5.2}%  size={:>8}",
        block_size / 1024,
        ratio,
        compressed.len()
    );
}

fn main() {
    let files = [
        ("samples/silesia/dickens", "dickens"),
        ("samples/cantrbry/alice29.txt", "alice29"),
        ("samples/large/E.coli", "E.coli"),
        ("samples/large/world192.txt", "world192"),
    ];

    let block_sizes = [
        64 * 1024,
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
        2 * 1024 * 1024,
        4 * 1024 * 1024,
    ];

    for (path, label) in &files {
        match std::fs::read(path) {
            Ok(data) => {
                println!("{} ({:.1} MB):", label, data.len() as f64 / 1048576.0);
                for &bs in &block_sizes {
                    if bs <= data.len() + 1024 {
                        test_block_size(&data, bs, label);
                    }
                }
                println!();
            }
            Err(e) => println!("{}: {}", label, e),
        }
    }
}
