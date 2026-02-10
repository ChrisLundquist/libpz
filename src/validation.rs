/// Validation tests for the reference implementation.
///
/// These tests verify:
/// 1. **Round-trip correctness** for every individual algorithm and every pipeline
/// 2. **Cross-module composition** - stages can be chained arbitrarily
/// 3. **Corpus testing** - Canterbury and large corpus files
/// 4. **Algorithmic properties** - entropy bounds, BWT clustering, MTF locality
/// 5. **Edge cases** - adversarial patterns, boundary conditions
/// 6. **Equivalence** - different LZ77 strategies produce decompressible output
#[cfg(test)]
mod tests {
    use crate::bwt;
    use crate::frequency;
    use crate::huffman::HuffmanTree;
    use crate::lz77;
    use crate::mtf;
    use crate::pipeline::{self, Pipeline};
    use crate::rangecoder;
    use crate::rle;

    // ---------------------------------------------------------------
    // Helper: generate diverse test vectors
    // ---------------------------------------------------------------

    /// Highly compressible: single byte repeated.
    fn data_all_zeros(n: usize) -> Vec<u8> {
        vec![0u8; n]
    }

    /// Incompressible: every byte value once (uniform distribution, 8 bits entropy).
    fn data_uniform() -> Vec<u8> {
        (0..=255u8).collect()
    }

    /// Skewed distribution: 90% one byte, 10% another.
    fn data_skewed(n: usize) -> Vec<u8> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(if i % 10 == 0 { 1 } else { 0 });
        }
        v
    }

    /// Repetitive text with structure.
    fn data_repeating_text() -> Vec<u8> {
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut v = Vec::new();
        for _ in 0..100 {
            v.extend_from_slice(pattern);
        }
        v
    }

    /// Binary data with some structure (sawtooth).
    fn data_sawtooth(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i % 256) as u8).collect()
    }

    /// Run-heavy data (simulates post-BWT+MTF output).
    fn data_runs() -> Vec<u8> {
        let mut v = Vec::new();
        for i in 0..50u8 {
            for _ in 0..(256 - i as usize * 4).max(1) {
                v.push(i);
            }
        }
        v
    }

    // ---------------------------------------------------------------
    // 1. Individual algorithm round-trip validation
    // ---------------------------------------------------------------

    /// Verify every individual algorithm with a broad set of test vectors.
    macro_rules! round_trip_test {
        ($name:ident, $data:expr) => {
            mod $name {
                use super::*;

                #[test]
                fn lz77_brute() {
                    let input = $data;
                    let compressed = lz77::compress(&input).unwrap();
                    let decompressed = lz77::decompress(&compressed).unwrap();
                    assert_eq!(decompressed, input, "lz77 brute-force round-trip failed");
                }

                #[test]
                fn lz77_hashchain() {
                    let input = $data;
                    let compressed = lz77::compress_hashchain(&input).unwrap();
                    let decompressed = lz77::decompress(&compressed).unwrap();
                    assert_eq!(decompressed, input, "lz77 hash-chain round-trip failed");
                }

                #[test]
                fn lz77_lazy() {
                    let input = $data;
                    let compressed = lz77::compress_lazy(&input).unwrap();
                    let decompressed = lz77::decompress(&compressed).unwrap();
                    assert_eq!(decompressed, input, "lz77 lazy round-trip failed");
                }

                #[test]
                fn huffman() {
                    let input = $data;
                    if input.is_empty() {
                        return;
                    }
                    let tree = HuffmanTree::from_data(&input).unwrap();
                    let (encoded, bits) = tree.encode(&input).unwrap();
                    let decoded = tree.decode(&encoded, bits).unwrap();
                    assert_eq!(decoded, input, "huffman round-trip failed");
                }

                #[test]
                fn rangecoder() {
                    let input = $data;
                    let encoded = rangecoder::encode(&input);
                    let decoded = rangecoder::decode(&encoded, input.len()).unwrap();
                    assert_eq!(decoded, input, "rangecoder round-trip failed");
                }

                #[test]
                fn bwt() {
                    let input = $data;
                    if input.is_empty() {
                        return;
                    }
                    let result = bwt::encode(&input).unwrap();
                    let decoded = bwt::decode(&result.data, result.primary_index).unwrap();
                    assert_eq!(decoded, input, "bwt round-trip failed");
                }

                #[test]
                fn mtf() {
                    let input = $data;
                    let encoded = mtf::encode(&input);
                    let decoded = mtf::decode(&encoded);
                    assert_eq!(decoded, input, "mtf round-trip failed");
                }

                #[test]
                fn rle() {
                    let input = $data;
                    let encoded = rle::encode(&input);
                    let decoded = rle::decode(&encoded).unwrap();
                    assert_eq!(decoded, input, "rle round-trip failed");
                }
            }
        };
    }

    round_trip_test!(rt_zeros_100, data_all_zeros(100));
    round_trip_test!(rt_zeros_5000, data_all_zeros(5000));
    round_trip_test!(rt_uniform, data_uniform());
    round_trip_test!(rt_skewed_1000, data_skewed(1000));
    round_trip_test!(rt_repeating_text, data_repeating_text());
    round_trip_test!(rt_sawtooth_1024, data_sawtooth(1024));
    round_trip_test!(rt_runs, data_runs());
    round_trip_test!(rt_single_byte, vec![42u8]);
    round_trip_test!(rt_two_bytes, vec![0u8, 255]);

    // ---------------------------------------------------------------
    // 2. Cross-module composition tests
    // ---------------------------------------------------------------

    /// Verify that any combination of transform stages composes correctly.
    /// This is the key modular architecture test: stages are independent
    /// building blocks that can be chained in any order.
    mod composition {
        use super::*;

        #[test]
        fn bwt_then_mtf() {
            let input = data_repeating_text();
            let bwt_result = bwt::encode(&input).unwrap();
            let mtf_data = mtf::encode(&bwt_result.data);
            // Inverse
            let inv_mtf = mtf::decode(&mtf_data);
            let inv_bwt = bwt::decode(&inv_mtf, bwt_result.primary_index).unwrap();
            assert_eq!(inv_bwt, input);
        }

        #[test]
        fn bwt_then_mtf_then_rle() {
            let input = data_repeating_text();
            let bwt_result = bwt::encode(&input).unwrap();
            let mtf_data = mtf::encode(&bwt_result.data);
            let rle_data = rle::encode(&mtf_data);
            // Inverse
            let inv_rle = rle::decode(&rle_data).unwrap();
            let inv_mtf = mtf::decode(&inv_rle);
            let inv_bwt = bwt::decode(&inv_mtf, bwt_result.primary_index).unwrap();
            assert_eq!(inv_bwt, input);
        }

        #[test]
        fn bwt_mtf_rle_rangecoder() {
            let input = data_repeating_text();
            let bwt_result = bwt::encode(&input).unwrap();
            let mtf_data = mtf::encode(&bwt_result.data);
            let rle_data = rle::encode(&mtf_data);
            let rc_data = rangecoder::encode(&rle_data);
            // Inverse
            let inv_rc = rangecoder::decode(&rc_data, rle_data.len()).unwrap();
            assert_eq!(inv_rc, rle_data, "rangecoder decode mismatch");
            let inv_rle = rle::decode(&inv_rc).unwrap();
            assert_eq!(inv_rle, mtf_data, "rle decode mismatch");
            let inv_mtf = mtf::decode(&inv_rle);
            assert_eq!(inv_mtf, bwt_result.data, "mtf decode mismatch");
            let inv_bwt = bwt::decode(&inv_mtf, bwt_result.primary_index).unwrap();
            assert_eq!(inv_bwt, input, "bwt decode mismatch");
        }

        #[test]
        fn lz77_then_huffman() {
            let input = data_repeating_text();
            let lz_data = lz77::compress_hashchain(&input).unwrap();
            let tree = HuffmanTree::from_data(&lz_data).unwrap();
            let (huff_data, bits) = tree.encode(&lz_data).unwrap();
            // Inverse
            let inv_huff = tree.decode(&huff_data, bits).unwrap();
            assert_eq!(inv_huff, lz_data);
            let inv_lz = lz77::decompress(&inv_huff).unwrap();
            assert_eq!(inv_lz, input);
        }

        #[test]
        fn lz77_then_rangecoder() {
            let input = data_repeating_text();
            let lz_data = lz77::compress_hashchain(&input).unwrap();
            let rc_data = rangecoder::encode(&lz_data);
            // Inverse
            let inv_rc = rangecoder::decode(&rc_data, lz_data.len()).unwrap();
            assert_eq!(inv_rc, lz_data);
            let inv_lz = lz77::decompress(&inv_rc).unwrap();
            assert_eq!(inv_lz, input);
        }

        #[test]
        fn mtf_then_rle_then_huffman() {
            // MTF + RLE + Huffman (without BWT)
            let input = data_runs();
            let mtf_data = mtf::encode(&input);
            let rle_data = rle::encode(&mtf_data);
            let tree = HuffmanTree::from_data(&rle_data).unwrap();
            let (huff_data, bits) = tree.encode(&rle_data).unwrap();
            // Inverse
            let inv_huff = tree.decode(&huff_data, bits).unwrap();
            let inv_rle = rle::decode(&inv_huff).unwrap();
            let inv_mtf = mtf::decode(&inv_rle);
            assert_eq!(inv_mtf, input);
        }
    }

    // ---------------------------------------------------------------
    // 3. Pipeline round-trip with all test vectors
    // ---------------------------------------------------------------

    mod pipeline_validation {
        use super::*;

        fn assert_pipeline_round_trip(input: &[u8], pipe: Pipeline) {
            let compressed = pipeline::compress(input, pipe).unwrap();
            let decompressed = pipeline::decompress(&compressed).unwrap();
            assert_eq!(
                decompressed,
                input,
                "pipeline {:?} round-trip failed for {} bytes",
                pipe,
                input.len()
            );
        }

        #[test]
        fn all_pipelines_zeros() {
            let input = data_all_zeros(500);
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_uniform() {
            let input = data_uniform();
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_skewed() {
            let input = data_skewed(2000);
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_text() {
            let input = data_repeating_text();
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_sawtooth() {
            let input = data_sawtooth(2048);
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_runs() {
            let input = data_runs();
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_single_byte() {
            let input = vec![42u8];
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                assert_pipeline_round_trip(&input, p);
            }
        }
    }

    // ---------------------------------------------------------------
    // 4. Algorithmic property validation
    // ---------------------------------------------------------------

    mod properties {
        use super::*;

        #[test]
        fn entropy_bounds() {
            // Shannon entropy of uniform distribution = 8.0 bits
            let uniform = data_uniform();
            let freq = frequency::get_frequency(&uniform);
            let e = freq.entropy();
            assert!((e - 8.0).abs() < 0.01, "uniform entropy = {}", e);

            // Single-byte data = 0.0 bits
            let zeros = data_all_zeros(100);
            let freq = frequency::get_frequency(&zeros);
            assert_eq!(freq.entropy(), 0.0);

            // 50/50 split = 1.0 bit
            let mut half = vec![0u8; 100];
            half.extend(vec![1u8; 100]);
            let freq = frequency::get_frequency(&half);
            assert!((freq.entropy() - 1.0).abs() < 0.01);
        }

        #[test]
        fn bwt_clusters_bytes() {
            // BWT should reduce the number of byte transitions (runs)
            let input = data_repeating_text();
            let result = bwt::encode(&input).unwrap();

            let count_runs = |data: &[u8]| -> usize {
                if data.len() <= 1 {
                    return data.len();
                }
                1 + data.windows(2).filter(|w| w[0] != w[1]).count()
            };

            let input_runs = count_runs(&input);
            let bwt_runs = count_runs(&result.data);
            assert!(
                bwt_runs <= input_runs,
                "BWT should cluster: input_runs={}, bwt_runs={}",
                input_runs,
                bwt_runs
            );
        }

        #[test]
        fn mtf_produces_small_values_after_bwt() {
            // After BWT+MTF, the output should be dominated by small index values
            let input = data_repeating_text();
            let bwt_result = bwt::encode(&input).unwrap();
            let mtf_data = mtf::encode(&bwt_result.data);

            let small_count = mtf_data.iter().filter(|&&b| b < 16).count();
            let ratio = small_count as f64 / mtf_data.len() as f64;
            assert!(
                ratio > 0.3,
                "expected >30% small values after BWT+MTF, got {:.1}%",
                ratio * 100.0
            );
        }

        #[test]
        fn rle_compresses_runs() {
            // RLE should shrink data with long runs
            let input = data_all_zeros(1000);
            let encoded = rle::encode(&input);
            assert!(
                encoded.len() < input.len() / 10,
                "RLE should compress 1000 zeros significantly, got {} bytes",
                encoded.len()
            );
        }

        #[test]
        fn huffman_beats_uniform_encoding() {
            // Huffman should encode skewed data in fewer bits than 8 bits/symbol
            let input = data_skewed(1000);
            let tree = HuffmanTree::from_data(&input).unwrap();
            let (encoded, _bits) = tree.encode(&input).unwrap();
            assert!(
                encoded.len() < input.len(),
                "huffman should compress skewed data: {} >= {}",
                encoded.len(),
                input.len()
            );
        }

        #[test]
        fn rangecoder_better_than_huffman_on_skewed() {
            // Range coder achieves fractional bits, so it should beat Huffman
            // on highly skewed data (where Huffman wastes due to integer code lengths)
            let input = data_all_zeros(1000);
            let rc_encoded = rangecoder::encode(&input);

            let tree = HuffmanTree::from_data(&input).unwrap();
            let (huff_encoded, _) = tree.encode(&input).unwrap();

            // Range coder should produce smaller output for single-symbol data
            // (Huffman needs 1 bit/symbol minimum, range coder can go below)
            assert!(
                rc_encoded.len() <= huff_encoded.len(),
                "rangecoder {} > huffman {} on uniform zeros",
                rc_encoded.len(),
                huff_encoded.len()
            );
        }

        #[test]
        fn lz77_finds_matches_in_repetitive_data() {
            // LZ77 should produce fewer matches than input bytes for repetitive data
            let input = data_repeating_text();
            let compressed = lz77::compress_hashchain(&input).unwrap();
            let num_matches = compressed.len() / lz77::Match::SERIALIZED_SIZE;
            // Each match covers at least 1 byte (literal), many cover more
            assert!(
                num_matches < input.len() / 2,
                "expected fewer matches than half the input: {} matches for {} bytes",
                num_matches,
                input.len()
            );
        }

        #[test]
        fn all_three_lz77_strategies_decompress_equivalently() {
            // All LZ77 strategies produce different compressed representations
            // but must all decompress back to the original
            let input = data_repeating_text();
            let brute = lz77::compress(&input).unwrap();
            let hashchain = lz77::compress_hashchain(&input).unwrap();
            let lazy = lz77::compress_lazy(&input).unwrap();

            let d_brute = lz77::decompress(&brute).unwrap();
            let d_hashchain = lz77::decompress(&hashchain).unwrap();
            let d_lazy = lz77::decompress(&lazy).unwrap();

            assert_eq!(d_brute, input);
            assert_eq!(d_hashchain, input);
            assert_eq!(d_lazy, input);
        }
    }

    // ---------------------------------------------------------------
    // 5. Corpus tests (Canterbury + large)
    // ---------------------------------------------------------------

    mod corpus {
        use super::*;
        use std::fs;
        use std::path::Path;

        /// Run all pipeline round-trips on a file if it exists.
        fn test_file_all_pipelines(path: &str) {
            let p = Path::new(path);
            if !p.exists() {
                // Skip if corpus not extracted (CI environments)
                eprintln!("SKIP: corpus file not found: {}", path);
                return;
            }
            let input = fs::read(p).unwrap();
            if input.is_empty() {
                return;
            }

            for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(&input, pipe).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(
                    decompressed, input,
                    "corpus round-trip failed: {:?} on {}",
                    pipe, path
                );
            }
        }

        /// Test individual algorithm round-trips on a file.
        fn test_file_individual_algorithms(path: &str) {
            let p = Path::new(path);
            if !p.exists() {
                eprintln!("SKIP: corpus file not found: {}", path);
                return;
            }
            let input = fs::read(p).unwrap();
            if input.is_empty() {
                return;
            }

            // LZ77 (all strategies)
            for compress_fn in &[
                lz77::compress as fn(&[u8]) -> crate::PzResult<Vec<u8>>,
                lz77::compress_hashchain,
                lz77::compress_lazy,
            ] {
                let compressed = compress_fn(&input).unwrap();
                let decompressed = lz77::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "LZ77 failed on {}", path);
            }

            // Huffman
            let tree = HuffmanTree::from_data(&input).unwrap();
            let (encoded, bits) = tree.encode(&input).unwrap();
            let decoded = tree.decode(&encoded, bits).unwrap();
            assert_eq!(decoded, input, "Huffman failed on {}", path);

            // Range coder
            let rc_encoded = rangecoder::encode(&input);
            let rc_decoded = rangecoder::decode(&rc_encoded, input.len()).unwrap();
            assert_eq!(rc_decoded, input, "Range coder failed on {}", path);

            // BWT + MTF + RLE chain
            let bwt_result = bwt::encode(&input).unwrap();
            let mtf_data = mtf::encode(&bwt_result.data);
            let rle_data = rle::encode(&mtf_data);
            let inv_rle = rle::decode(&rle_data).unwrap();
            let inv_mtf = mtf::decode(&inv_rle);
            let inv_bwt = bwt::decode(&inv_mtf, bwt_result.primary_index).unwrap();
            assert_eq!(inv_bwt, input, "BWT+MTF+RLE chain failed on {}", path);
        }

        // Canterbury corpus files
        static CANTRBRY_DIR: &str = "/home/user/libpz/samples/cantrbry";

        #[test]
        fn canterbury_alice() {
            test_file_all_pipelines(&format!("{}/alice29.txt", CANTRBRY_DIR));
        }

        #[test]
        fn canterbury_asyoulik() {
            test_file_all_pipelines(&format!("{}/asyoulik.txt", CANTRBRY_DIR));
        }

        #[test]
        fn canterbury_cp_html() {
            test_file_all_pipelines(&format!("{}/cp.html", CANTRBRY_DIR));
        }

        #[test]
        fn canterbury_fields_c() {
            test_file_all_pipelines(&format!("{}/fields.c", CANTRBRY_DIR));
        }

        #[test]
        fn canterbury_grammar_lsp() {
            test_file_all_pipelines(&format!("{}/grammar.lsp", CANTRBRY_DIR));
        }

        #[test]
        fn canterbury_xargs() {
            test_file_all_pipelines(&format!("{}/xargs.1", CANTRBRY_DIR));
        }

        #[test]
        fn canterbury_algorithms_individual() {
            // Test individual algorithms on a representative text file
            test_file_individual_algorithms(&format!("{}/alice29.txt", CANTRBRY_DIR));
        }

        // Large corpus files (longer running, test individual algos on smaller slices)
        static LARGE_DIR: &str = "/home/user/libpz/samples/large";

        #[test]
        fn large_bible_first_64k() {
            let path = format!("{}/bible.txt", LARGE_DIR);
            let p = Path::new(&path);
            if !p.exists() {
                eprintln!("SKIP: {}", path);
                return;
            }
            let full = fs::read(p).unwrap();
            let input = &full[..full.len().min(65536)];
            for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(input, pipe).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(
                    decompressed, input,
                    "large corpus round-trip failed: {:?}",
                    pipe
                );
            }
        }

        #[test]
        fn large_ecoli_first_64k() {
            let path = format!("{}/E.coli", LARGE_DIR);
            let p = Path::new(&path);
            if !p.exists() {
                eprintln!("SKIP: {}", path);
                return;
            }
            let full = fs::read(p).unwrap();
            let input = &full[..full.len().min(65536)];
            for &pipe in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(input, pipe).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(
                    decompressed, input,
                    "large corpus round-trip failed: {:?}",
                    pipe
                );
            }
        }
    }

    // ---------------------------------------------------------------
    // 6. Edge cases and adversarial patterns
    // ---------------------------------------------------------------

    mod edge_cases {
        use super::*;

        #[test]
        fn pipeline_two_bytes() {
            // Smallest non-trivial input
            let input = vec![0u8, 1];
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn alternating_bytes() {
            // Worst case for RLE (no runs), but structured for LZ77
            let input: Vec<u8> = (0..1000).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn all_256_byte_values() {
            // Every byte value appears exactly once
            let input: Vec<u8> = (0..=255).collect();
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn max_byte_runs() {
            // 259 zeros (max single RLE chunk), then 259 ones, etc.
            let mut input = Vec::new();
            for byte in 0..4u8 {
                input.extend(vec![byte; 259]);
            }
            let encoded = rle::encode(&input);
            let decoded = rle::decode(&encoded).unwrap();
            assert_eq!(decoded, input);

            // Full pipeline test
            for &p in &[Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn descending_bytes() {
            let input: Vec<u8> = (0..=255).rev().collect();
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn repeated_short_pattern() {
            // "ab" repeated 500 times - good for LZ77
            let input: Vec<u8> = b"ab".iter().copied().cycle().take(1000).collect();
            for &p in &[Pipeline::Deflate, Pipeline::Bw, Pipeline::Lza] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn long_match_boundary() {
            // Force a match that spans the MAX_WINDOW boundary
            let mut input = vec![0u8; 4096]; // fill window
            input.extend(vec![1u8; 100]); // different data
            input.extend(vec![0u8; 200]); // should NOT match (too far back)
            let compressed = lz77::compress_hashchain(&input).unwrap();
            let decompressed = lz77::decompress(&compressed).unwrap();
            assert_eq!(decompressed, input);
        }
    }
}
