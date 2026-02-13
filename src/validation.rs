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
    use crate::lz78;
    use crate::lzss;
    use crate::mtf;
    use crate::pipeline::{self, Pipeline};
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

                #[test]
                fn lzss() {
                    let input = $data;
                    let compressed = lzss::encode(&input).unwrap();
                    let decompressed = lzss::decode(&compressed).unwrap();
                    assert_eq!(decompressed, input, "lzss round-trip failed");
                }

                #[test]
                fn lz78() {
                    let input = $data;
                    let compressed = lz78::encode(&input).unwrap();
                    let decompressed = lz78::decode(&compressed).unwrap();
                    assert_eq!(decompressed, input, "lz78 round-trip failed");
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
        fn bwt_mtf_rle_fse() {
            use crate::fse;
            let input = data_repeating_text();
            let bwt_result = bwt::encode(&input).unwrap();
            let mtf_data = mtf::encode(&bwt_result.data);
            let rle_data = rle::encode(&mtf_data);
            let fse_data = fse::encode(&rle_data);
            // Inverse
            let inv_fse = fse::decode(&fse_data, rle_data.len()).unwrap();
            assert_eq!(inv_fse, rle_data, "fse decode mismatch");
            let inv_rle = rle::decode(&inv_fse).unwrap();
            assert_eq!(inv_rle, mtf_data, "rle decode mismatch");
            let inv_mtf = mtf::decode(&inv_rle);
            assert_eq!(inv_mtf, bwt_result.data, "mtf decode mismatch");
            let inv_bwt = bwt::decode(&inv_mtf, bwt_result.primary_index).unwrap();
            assert_eq!(inv_bwt, input, "bwt decode mismatch");
        }

        #[test]
        fn lz77_then_huffman() {
            let input = data_repeating_text();
            let lz_data = lz77::compress_lazy(&input).unwrap();
            let tree = HuffmanTree::from_data(&lz_data).unwrap();
            let (huff_data, bits) = tree.encode(&lz_data).unwrap();
            // Inverse
            let inv_huff = tree.decode(&huff_data, bits).unwrap();
            assert_eq!(inv_huff, lz_data);
            let inv_lz = lz77::decompress(&inv_huff).unwrap();
            assert_eq!(inv_lz, input);
        }

        #[test]
        fn lz77_then_fse() {
            use crate::fse;
            let input = data_repeating_text();
            let lz_data = lz77::compress_lazy(&input).unwrap();
            let fse_data = fse::encode(&lz_data);
            // Inverse
            let inv_fse = fse::decode(&fse_data, lz_data.len()).unwrap();
            assert_eq!(inv_fse, lz_data);
            let inv_lz = lz77::decompress(&inv_fse).unwrap();
            assert_eq!(inv_lz, input);
        }

        #[test]
        fn lzss_then_fse() {
            use crate::fse;
            let input = data_repeating_text();
            let lzss_data = lzss::encode(&input).unwrap();
            let fse_data = fse::encode(&lzss_data);
            // Inverse
            let inv_fse = fse::decode(&fse_data, lzss_data.len()).unwrap();
            assert_eq!(inv_fse, lzss_data);
            let inv_lzss = lzss::decode(&inv_fse).unwrap();
            assert_eq!(inv_lzss, input);
        }

        #[test]
        fn lz78_then_fse() {
            use crate::fse;
            let input = data_repeating_text();
            let lz78_data = lz78::encode(&input).unwrap();
            let fse_data = fse::encode(&lz78_data);
            // Inverse
            let inv_fse = fse::decode(&fse_data, lz78_data.len()).unwrap();
            assert_eq!(inv_fse, lz78_data);
            let inv_lz78 = lz78::decode(&inv_fse).unwrap();
            assert_eq!(inv_lz78, input);
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
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::Lzfi,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_uniform() {
            let input = data_uniform();
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::Lzfi,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_skewed() {
            let input = data_skewed(2000);
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::Lzfi,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_text() {
            let input = data_repeating_text();
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::Lzfi,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_sawtooth() {
            let input = data_sawtooth(2048);
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::Lzfi,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_runs() {
            let input = data_runs();
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::Lzfi,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                assert_pipeline_round_trip(&input, p);
            }
        }

        #[test]
        fn all_pipelines_single_byte() {
            let input = vec![42u8];
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::Lzfi,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
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
        fn lz77_finds_matches_in_repetitive_data() {
            // LZ77 should produce fewer matches than input bytes for repetitive data
            let input = data_repeating_text();
            let compressed = lz77::compress_lazy(&input).unwrap();
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
        fn lz77_lazy_decompresses_correctly() {
            let input = data_repeating_text();
            let lazy = lz77::compress_lazy(&input).unwrap();
            let d_lazy = lz77::decompress(&lazy).unwrap();
            assert_eq!(d_lazy, input);
        }

        #[test]
        fn lzss_more_compact_on_literal_heavy_data() {
            // LZSS should produce smaller output than LZ77 on data with many
            // literals, because literals cost 1 byte instead of 5.
            let input: Vec<u8> = (0..=255).collect(); // 256 unique bytes, mostly literals
            let lz77_size = lz77::compress_lazy(&input).unwrap().len();
            let lzss_size = lzss::encode(&input).unwrap().len();
            assert!(
                lzss_size <= lz77_size,
                "LZSS should be <= LZ77 on literal-heavy data: {} vs {}",
                lzss_size,
                lz77_size
            );
        }

        #[test]
        fn lz78_dictionary_grows() {
            let input = data_repeating_text();
            let compressed = lz78::encode(&input).unwrap();
            assert!(
                compressed.len() < input.len(),
                "LZ78 should compress repetitive text: {} >= {}",
                compressed.len(),
                input.len()
            );
        }
    }

    // ---------------------------------------------------------------
    // Compression ratio comparison (LZ77 vs LZSS vs LZ78)
    // ---------------------------------------------------------------

    mod lz_comparison {
        use super::*;
        use crate::fse;

        fn ratio_report(name: &str, input: &[u8]) {
            let lz77_raw = lz77::compress_lazy(input).unwrap();
            let lzss_raw = lzss::encode(input).unwrap();
            let lz78_raw = lz78::encode(input).unwrap();

            let lz77_fse = fse::encode(&lz77_raw);
            let lzss_fse = fse::encode(&lzss_raw);
            let lz78_fse = fse::encode(&lz78_raw);

            eprintln!(
                "  {:20} {:>6}B | LZ77 {:>6} ({:5.1}%) | LZSS {:>6} ({:5.1}%) | LZ78 {:>6} ({:5.1}%)",
                name,
                input.len(),
                lz77_raw.len(),
                100.0 * lz77_raw.len() as f64 / input.len() as f64,
                lzss_raw.len(),
                100.0 * lzss_raw.len() as f64 / input.len() as f64,
                lz78_raw.len(),
                100.0 * lz78_raw.len() as f64 / input.len() as f64,
            );
            eprintln!(
                "  {:20} {:>6}  | +FSE {:>6} ({:5.1}%) | +FSE {:>6} ({:5.1}%) | +FSE {:>6} ({:5.1}%)",
                "",
                "",
                lz77_fse.len(),
                100.0 * lz77_fse.len() as f64 / input.len() as f64,
                lzss_fse.len(),
                100.0 * lzss_fse.len() as f64 / input.len() as f64,
                lz78_fse.len(),
                100.0 * lz78_fse.len() as f64 / input.len() as f64,
            );
        }

        #[test]
        fn lz_compression_ratio_comparison() {
            eprintln!();
            eprintln!("=== LZ Algorithm Compression Ratio Comparison ===");
            eprintln!();

            ratio_report("repeating_text", &data_repeating_text());
            ratio_report("zeros_5000", &data_all_zeros(5000));
            ratio_report("skewed_2000", &data_skewed(2000));
            ratio_report("sawtooth_2048", &data_sawtooth(2048));
            ratio_report("uniform_256", &data_uniform());

            eprintln!();
        }
    }

    // ---------------------------------------------------------------
    // 5. Corpus tests (Canterbury + large)
    // ---------------------------------------------------------------

    mod corpus {
        use super::*;
        use std::fs;
        use std::path::{Path, PathBuf};

        /// Build a path relative to the crate root (CARGO_MANIFEST_DIR).
        fn samples_dir(subdir: &str) -> PathBuf {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("samples")
                .join(subdir)
        }

        /// Run all pipeline round-trips on a file if it exists.
        fn test_file_all_pipelines(path: &Path) {
            if !path.exists() {
                // Skip if corpus not extracted (CI environments)
                eprintln!("SKIP: corpus file not found: {}", path.display());
                return;
            }
            let input = fs::read(path).unwrap();
            if input.is_empty() {
                return;
            }

            for &pipe in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                let compressed = pipeline::compress(&input, pipe).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(
                    decompressed,
                    input,
                    "corpus round-trip failed: {:?} on {}",
                    pipe,
                    path.display()
                );
            }
        }

        /// Test individual algorithm round-trips on a file.
        fn test_file_individual_algorithms(path: &Path) {
            if !path.exists() {
                eprintln!("SKIP: corpus file not found: {}", path.display());
                return;
            }
            let input = fs::read(path).unwrap();
            if input.is_empty() {
                return;
            }

            // LZ77 (lazy matching)
            let compressed = lz77::compress_lazy(&input).unwrap();
            let decompressed = lz77::decompress(&compressed).unwrap();
            assert_eq!(decompressed, input, "LZ77 failed on {}", path.display());

            // Huffman
            let tree = HuffmanTree::from_data(&input).unwrap();
            let (encoded, bits) = tree.encode(&input).unwrap();
            let decoded = tree.decode(&encoded, bits).unwrap();
            assert_eq!(decoded, input, "Huffman failed on {}", path.display());

            // BWT + MTF + RLE chain
            let bwt_result = bwt::encode(&input).unwrap();
            let mtf_data = mtf::encode(&bwt_result.data);
            let rle_data = rle::encode(&mtf_data);
            let inv_rle = rle::decode(&rle_data).unwrap();
            let inv_mtf = mtf::decode(&inv_rle);
            let inv_bwt = bwt::decode(&inv_mtf, bwt_result.primary_index).unwrap();
            assert_eq!(
                inv_bwt,
                input,
                "BWT+MTF+RLE chain failed on {}",
                path.display()
            );
        }

        // Canterbury corpus files — path built from CARGO_MANIFEST_DIR

        #[test]
        fn canterbury_alice() {
            test_file_all_pipelines(&samples_dir("cantrbry").join("alice29.txt"));
        }

        #[test]
        fn canterbury_asyoulik() {
            test_file_all_pipelines(&samples_dir("cantrbry").join("asyoulik.txt"));
        }

        #[test]
        fn canterbury_cp_html() {
            test_file_all_pipelines(&samples_dir("cantrbry").join("cp.html"));
        }

        #[test]
        fn canterbury_fields_c() {
            test_file_all_pipelines(&samples_dir("cantrbry").join("fields.c"));
        }

        #[test]
        fn canterbury_grammar_lsp() {
            test_file_all_pipelines(&samples_dir("cantrbry").join("grammar.lsp"));
        }

        #[test]
        fn canterbury_xargs() {
            test_file_all_pipelines(&samples_dir("cantrbry").join("xargs.1"));
        }

        #[test]
        fn canterbury_algorithms_individual() {
            // Test individual algorithms on a representative text file
            test_file_individual_algorithms(&samples_dir("cantrbry").join("alice29.txt"));
        }

        // Large corpus files (longer running, test individual algos on smaller slices)

        #[test]
        fn large_bible_first_64k() {
            let path = samples_dir("large").join("bible.txt");
            if !path.exists() {
                eprintln!("SKIP: {}", path.display());
                return;
            }
            let full = fs::read(&path).unwrap();
            let input = &full[..full.len().min(65536)];
            for &pipe in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
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
            let path = samples_dir("large").join("E.coli");
            if !path.exists() {
                eprintln!("SKIP: {}", path.display());
                return;
            }
            let full = fs::read(&path).unwrap();
            let input = &full[..full.len().min(65536)];
            for &pipe in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
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
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn alternating_bytes() {
            // Worst case for RLE (no runs), but structured for LZ77
            let input: Vec<u8> = (0..1000).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn all_256_byte_values() {
            // Every byte value appears exactly once
            let input: Vec<u8> = (0..=255).collect();
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
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
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn descending_bytes() {
            let input: Vec<u8> = (0..=255).rev().collect();
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
                let compressed = pipeline::compress(&input, p).unwrap();
                let decompressed = pipeline::decompress(&compressed).unwrap();
                assert_eq!(decompressed, input, "pipeline {:?}", p);
            }
        }

        #[test]
        fn repeated_short_pattern() {
            // "ab" repeated 500 times - good for LZ77
            let input: Vec<u8> = b"ab".iter().copied().cycle().take(1000).collect();
            for &p in &[
                Pipeline::Deflate,
                Pipeline::Bw,
                Pipeline::Bbw,
                Pipeline::Lzr,
                Pipeline::Lzf,
                Pipeline::LzssR,
                Pipeline::Lz78R,
            ] {
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
            let compressed = lz77::compress_lazy(&input).unwrap();
            let decompressed = lz77::decompress(&compressed).unwrap();
            assert_eq!(decompressed, input);
        }
    }

    // -----------------------------------------------------------------------
    // GPU → CPU cross-decompression tests
    //
    // These verify the critical invariant: compressed bytes from a GPU backend
    // are decompressible by the standard CPU path. This is the exact scenario
    // that breaks when GPU and CPU demux/mux formats diverge.
    // -----------------------------------------------------------------------

    /// Generate test input large enough to trigger GPU dispatch (≥64KB).
    #[cfg(any(feature = "opencl", feature = "webgpu"))]
    fn gpu_test_input() -> Vec<u8> {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let target_size = 80 * 1024; // 80KB, above MIN_GPU_INPUT_SIZE (64KB)
        let mut input = Vec::with_capacity(target_size);
        while input.len() < target_size {
            input.extend_from_slice(pattern);
        }
        input.truncate(target_size);
        input
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn opencl_compress_cpu_decompress_deflate() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::opencl::OpenClEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_test_input();
        let options = CompressOptions {
            backend: Backend::OpenCl,
            opencl_engine: Some(engine),
            ..Default::default()
        };
        let compressed =
            pipeline::compress_with_options(&input, Pipeline::Deflate, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "OpenCL Deflate GPU→CPU round-trip");
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn opencl_compress_cpu_decompress_lzr() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::opencl::OpenClEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_test_input();
        let options = CompressOptions {
            backend: Backend::OpenCl,
            opencl_engine: Some(engine),
            ..Default::default()
        };
        let compressed = pipeline::compress_with_options(&input, Pipeline::Lzr, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "OpenCL Lzr GPU→CPU round-trip");
    }

    #[test]
    #[cfg(feature = "opencl")]
    fn opencl_compress_cpu_decompress_lzf() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::opencl::OpenClEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_test_input();
        let options = CompressOptions {
            backend: Backend::OpenCl,
            opencl_engine: Some(engine),
            ..Default::default()
        };
        let compressed = pipeline::compress_with_options(&input, Pipeline::Lzf, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "OpenCL Lzf GPU→CPU round-trip");
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn webgpu_compress_cpu_decompress_deflate() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_test_input();
        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            ..Default::default()
        };
        let compressed =
            pipeline::compress_with_options(&input, Pipeline::Deflate, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "WebGPU Deflate GPU→CPU round-trip");
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn webgpu_compress_cpu_decompress_lzr() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_test_input();
        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            ..Default::default()
        };
        let compressed = pipeline::compress_with_options(&input, Pipeline::Lzr, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "WebGPU Lzr GPU→CPU round-trip");
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn webgpu_compress_cpu_decompress_lzf() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_test_input();
        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            ..Default::default()
        };
        let compressed = pipeline::compress_with_options(&input, Pipeline::Lzf, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "WebGPU Lzf GPU→CPU round-trip");
    }

    // -----------------------------------------------------------------------
    // GPU streaming pipeline round-trip tests
    //
    // These use multi-block inputs (>1 block) to exercise the double-buffered
    // streaming GPU path. A small block size forces multiple blocks even on
    // moderately sized inputs, ensuring the ring/slot logic is exercised.
    // -----------------------------------------------------------------------

    /// Generate multi-block test input: ~512KB of structured text, guaranteeing
    /// at least 2 blocks at default block size (256KB).
    #[cfg(feature = "webgpu")]
    fn gpu_streaming_test_input() -> Vec<u8> {
        let pattern = b"The quick brown fox jumps over the lazy dog. ";
        let target_size = 512 * 1024; // 512KB = 2 blocks at 256KB default
        let mut input = Vec::with_capacity(target_size);
        while input.len() < target_size {
            input.extend_from_slice(pattern);
        }
        input.truncate(target_size);
        input
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn webgpu_streaming_round_trip_deflate() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_streaming_test_input();
        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            threads: 2,
            ..Default::default()
        };
        let compressed =
            pipeline::compress_with_options(&input, Pipeline::Deflate, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "WebGPU streaming Deflate round-trip (multi-block)"
        );
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn webgpu_streaming_round_trip_lzf() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        let input = gpu_streaming_test_input();
        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            threads: 2,
            ..Default::default()
        };
        let compressed = pipeline::compress_with_options(&input, Pipeline::Lzf, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "WebGPU streaming Lzf round-trip (multi-block)"
        );
    }

    #[test]
    #[cfg(feature = "webgpu")]
    fn webgpu_streaming_round_trip_small_blocks() {
        use crate::pipeline::{Backend, CompressOptions};
        let engine = match crate::webgpu::WebGpuEngine::new() {
            Ok(e) => std::sync::Arc::new(e),
            Err(crate::PzError::Unsupported) => return,
            Err(e) => panic!("unexpected error: {:?}", e),
        };
        // Use a smaller input with small block size to create many blocks
        let input = gpu_test_input(); // 80KB
        let options = CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(engine),
            block_size: 16 * 1024, // 16KB blocks → 5 blocks from 80KB
            threads: 2,
            ..Default::default()
        };
        // This should go through the streaming path (5 blocks > 1)
        // or batched path, both exercising multi-block GPU code
        let compressed =
            pipeline::compress_with_options(&input, Pipeline::Deflate, &options).unwrap();
        let decompressed = pipeline::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "WebGPU streaming Deflate round-trip (small blocks)"
        );
    }
}
