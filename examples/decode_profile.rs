//! Profile LzSeqR vs LzSeqH decode: isolate entropy decode vs LzSeq decode time.
//!
//! Usage: cargo run --release --example decode_profile

fn main() {
    use pz::pipeline::{self, CompressOptions, Pipeline};
    use std::time::Instant;

    // Use Canterbury corpus alice29.txt if available, else generate semi-random data
    let data = if let Ok(d) = std::fs::read("samples/cantrbry/alice29.txt") {
        d
    } else {
        // Semi-random 150KB: mix of text patterns and noise (typical ~40% ratio)
        let mut d = Vec::with_capacity(150 * 1024);
        let phrases = [
            b"compression algorithms are fascinating ".as_slice(),
            b"data structures enable efficient storage ".as_slice(),
            b"entropy coding reduces redundancy ".as_slice(),
            b"hash tables provide fast lookup ".as_slice(),
        ];
        let mut state: u32 = 0xDEADBEEF;
        while d.len() < 150 * 1024 {
            let phrase = phrases[(state as usize / 7) % phrases.len()];
            d.extend_from_slice(phrase);
            // Add noise between phrases
            for _ in 0..32 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                d.push(state as u8);
            }
        }
        d.truncate(150 * 1024);
        d
    };

    let iters = 20;

    for &(name, pipeline) in &[("LzSeqR", Pipeline::LzSeqR), ("LzSeqH", Pipeline::LzSeqH)] {
        let options = CompressOptions::default();
        let compressed = pipeline::compress_with_options(&data, pipeline, &options).unwrap();
        let ratio = compressed.len() as f64 / data.len() as f64 * 100.0;

        // Warmup + verify
        let warmup = pipeline::decompress(&compressed)
            .unwrap_or_else(|e| panic!("{name}: decompress failed: {e:?}"));
        assert_eq!(warmup.len(), data.len(), "{name}: length mismatch");
        assert_eq!(warmup, data, "{name}: data mismatch");

        // Decode benchmark
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let decoded = pipeline::decompress(&compressed)
                .unwrap_or_else(|e| panic!("{name}: decompress failed: {e:?}"));
            let elapsed = t0.elapsed();
            times.push(elapsed.as_secs_f64());
            assert_eq!(decoded.len(), data.len());
            std::hint::black_box(decoded);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[iters / 2];
        let throughput = data.len() as f64 / median / 1e6;

        println!(
            "{name:10} decode: {:.2} ms ({:.1} MB/s)  ratio: {ratio:.1}%  compressed: {} bytes",
            median * 1000.0,
            throughput,
            compressed.len(),
        );
    }

    // Now test raw LzSeq encode/decode without entropy coding to isolate it
    println!();
    println!("--- Raw LzSeq (no entropy coding) ---");
    {
        let enc = pz::lzseq::encode(&data).unwrap();
        let total_stream_size = enc.flags.len()
            + enc.literals.len()
            + enc.offset_codes.len()
            + enc.offset_extra.len()
            + enc.length_codes.len()
            + enc.length_extra.len();
        println!(
            "Encoded: {} tokens ({} matches), {} total stream bytes",
            enc.num_tokens, enc.num_matches, total_stream_size
        );

        // Time raw LzSeq decode (no entropy)
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let decoded = pz::lzseq::decode(
                &enc.flags,
                &enc.literals,
                &enc.offset_codes,
                &enc.offset_extra,
                &enc.length_codes,
                &enc.length_extra,
                enc.num_tokens,
                enc.num_matches,
                data.len(),
            )
            .unwrap();
            let elapsed = t0.elapsed();
            times.push(elapsed.as_secs_f64());
            assert_eq!(decoded.len(), data.len());
            std::hint::black_box(decoded);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[iters / 2];
        let throughput = data.len() as f64 / median / 1e6;
        println!(
            "Raw LzSeq decode: {:.2} ms ({:.1} MB/s)",
            median * 1000.0,
            throughput
        );
    }

    // Time raw rANS and Huffman encode/decode to isolate entropy speed
    println!();
    println!("--- Raw entropy coding (1MB random-ish data) ---");
    {
        let enc = pz::lzseq::encode(&data).unwrap();

        // Use the literals stream as test data (most realistic)
        let stream = &enc.literals;
        println!("Test stream: {} bytes (literals)", stream.len());

        // rANS
        let rans_encoded = pz::rans::encode(stream);
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let decoded = pz::rans::decode(&rans_encoded, stream.len()).unwrap();
            let elapsed = t0.elapsed();
            times.push(elapsed.as_secs_f64());
            std::hint::black_box(decoded);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[iters / 2];
        let throughput = stream.len() as f64 / median / 1e6;
        println!(
            "rANS decode:    {:.2} ms ({:.1} MB/s) for {} bytes",
            median * 1000.0,
            throughput,
            stream.len()
        );

        // Huffman
        let tree = pz::huffman::HuffmanTree::from_data(stream).unwrap();
        let (huff_data, total_bits) = tree.encode(stream).unwrap();
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t0 = Instant::now();
            let decoded = tree.decode(&huff_data, total_bits).unwrap();
            let elapsed = t0.elapsed();
            times.push(elapsed.as_secs_f64());
            std::hint::black_box(decoded);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times[iters / 2];
        let throughput = stream.len() as f64 / median / 1e6;
        println!(
            "Huffman decode: {:.2} ms ({:.1} MB/s) for {} bytes",
            median * 1000.0,
            throughput,
            stream.len()
        );
    }
}
