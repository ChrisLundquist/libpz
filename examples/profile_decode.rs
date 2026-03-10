// Profile decode stages: measure time per stage in BW and LZR pipelines.
use std::time::Instant;

fn profile_bw_decode(data: &[u8], label: &str) {
    use pz::{bwt, fse, mtf, rle};

    // First compress to get compressed data
    let bwt_result = pz::bwt::encode(data).unwrap();
    let mtf_out = mtf::encode(&bwt_result.data);
    let rle_out = rle::encode(&mtf_out);
    let fse_out = fse::encode(&rle_out);

    // Now profile decode stages
    let iters = 10;
    let mut fse_ns = 0u128;
    let mut rle_ns = 0u128;
    let mut mtf_ns = 0u128;
    let mut bwt_ns = 0u128;

    for _ in 0..iters {
        let t = Instant::now();
        let rle_data = fse::decode(&fse_out, rle_out.len()).unwrap();
        fse_ns += t.elapsed().as_nanos();

        let t = Instant::now();
        let mtf_data = rle::decode(&rle_data).unwrap();
        rle_ns += t.elapsed().as_nanos();

        let t = Instant::now();
        let bwt_data = mtf::decode(&mtf_data);
        mtf_ns += t.elapsed().as_nanos();

        let t = Instant::now();
        let _output = bwt::decode(&bwt_data, bwt_result.primary_index).unwrap();
        bwt_ns += t.elapsed().as_nanos();
    }

    let total = fse_ns + rle_ns + mtf_ns + bwt_ns;
    let mb = data.len() as f64 / 1048576.0;
    let total_ms = total as f64 / iters as f64 / 1_000_000.0;
    let tp = mb / (total_ms / 1000.0);

    println!(
        "BW decode {} ({:.1} MB) — {:.1} ms ({:.1} MB/s)",
        label, mb, total_ms, tp
    );
    println!(
        "  FSE:  {:6.2} ms ({:4.1}%)  RLE: {:6.2} ms ({:4.1}%)  MTF: {:6.2} ms ({:4.1}%)  BWT: {:6.2} ms ({:4.1}%)",
        fse_ns as f64 / iters as f64 / 1e6,
        fse_ns as f64 / total as f64 * 100.0,
        rle_ns as f64 / iters as f64 / 1e6,
        rle_ns as f64 / total as f64 * 100.0,
        mtf_ns as f64 / iters as f64 / 1e6,
        mtf_ns as f64 / total as f64 * 100.0,
        bwt_ns as f64 / iters as f64 / 1e6,
        bwt_ns as f64 / total as f64 * 100.0,
    );
}

fn profile_lzseqr_decode(data: &[u8], label: &str) {
    use pz::pipeline::{self, CompressOptions, Pipeline};

    let opts = CompressOptions {
        threads: 1,
        ..Default::default()
    };
    let compressed = pipeline::compress_with_options(data, Pipeline::LzSeqR, &opts).unwrap();

    let iters = 10;
    let mut total_ns = 0u128;
    for _ in 0..iters {
        let t = Instant::now();
        let _out = pipeline::decompress(&compressed).unwrap();
        total_ns += t.elapsed().as_nanos();
    }

    let mb = data.len() as f64 / 1048576.0;
    let avg_ms = total_ns as f64 / iters as f64 / 1e6;
    let tp = mb / (avg_ms / 1000.0);
    println!(
        "LZR decode {} ({:.1} MB) — {:.1} ms ({:.1} MB/s)",
        label, mb, avg_ms, tp
    );
}

fn profile_deflate_decode(data: &[u8], label: &str) {
    use pz::pipeline::{self, CompressOptions, Pipeline};

    let opts = CompressOptions {
        threads: 1,
        ..Default::default()
    };
    let compressed = pipeline::compress_with_options(data, Pipeline::Deflate, &opts).unwrap();

    let iters = 10;
    let mut total_ns = 0u128;
    for _ in 0..iters {
        let t = Instant::now();
        let _out = pipeline::decompress(&compressed).unwrap();
        total_ns += t.elapsed().as_nanos();
    }

    let mb = data.len() as f64 / 1048576.0;
    let avg_ms = total_ns as f64 / iters as f64 / 1e6;
    let tp = mb / (avg_ms / 1000.0);
    println!(
        "Deflate decode {} ({:.1} MB) — {:.1} ms ({:.1} MB/s)",
        label, mb, avg_ms, tp
    );
}

fn main() {
    let files = [
        ("samples/silesia/dickens", "dickens"),
        ("samples/large/E.coli", "E.coli"),
        ("samples/large/world192.txt", "world192"),
    ];

    for (path, label) in &files {
        match std::fs::read(path) {
            Ok(data) => {
                profile_bw_decode(&data, label);
                profile_lzseqr_decode(&data, label);
                profile_deflate_decode(&data, label);
                println!();
            }
            Err(e) => println!("{}: {}", label, e),
        }
    }
}
