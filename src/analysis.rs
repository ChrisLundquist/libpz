//! Data analysis and profiling for pipeline auto-selection.
//!
//! Computes statistical metrics on input data to predict which compression
//! pipeline will perform best. Metrics include byte entropy, autocorrelation,
//! run ratio, LZ77 match density, and distribution shape classification.

use crate::frequency;

/// Default sample size for large inputs (64KB).
const DEFAULT_SAMPLE_SIZE: usize = 64 * 1024;

/// Hash table size for match density estimation.
const MATCH_HASH_SIZE: usize = 8192;
const MATCH_HASH_MASK: usize = MATCH_HASH_SIZE - 1;

/// Window size for match density estimation (smaller than full LZ77 window for speed).
const MATCH_WINDOW: usize = 4096;

/// Statistical profile of input data, used for pipeline auto-selection.
#[derive(Debug, Clone)]
pub struct DataProfile {
    /// Shannon entropy of raw byte distribution (0.0 to 8.0 bits per symbol).
    pub byte_entropy: f32,
    /// Pearson correlation between consecutive bytes (range -1.0 to 1.0).
    /// High positive value indicates sequential similarity (structured data).
    pub autocorrelation_lag1: f32,
    /// Fraction of input bytes that are part of runs of 4+ identical bytes.
    /// High run_ratio predicts BWT+RLE benefit.
    pub run_ratio: f32,
    /// Estimated fraction of positions where a 3+ byte match exists
    /// within a nearby window. High match_density predicts LZ77 benefit.
    pub match_density: f32,
    /// Classification of the byte value distribution shape.
    pub distribution_shape: DistributionShape,
    /// Number of distinct byte values present (0-256).
    pub distinct_bytes: u32,
    /// Total input length that was analyzed (may be a sample).
    pub input_len: usize,
    /// Entropy (bits/byte) saved by the best byte-plane split + per-plane
    /// gated delta, over `byte_entropy`. High values indicate fixed-stride
    /// numeric/record data (e.g. 16-bit samples, fixed-width records) that
    /// the `Num` pipeline compresses far better than LZ or BWT.
    pub numeric_gain: f32,
    /// The stride that produced `numeric_gain` (0 when no stride applies).
    pub numeric_stride: u32,
}

/// Byte value distribution shape classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistributionShape {
    /// Nearly all bytes are the same value (entropy < 0.5).
    Constant,
    /// Heavily skewed toward few values (entropy < 3.0, distinct < 32).
    Skewed,
    /// Roughly uniform across all byte values (entropy > 7.0).
    Uniform,
    /// Two distinct peaks in the frequency histogram.
    Bimodal,
    /// General distribution not fitting other categories.
    General,
}

/// Analyze input data and return a statistical profile.
///
/// For large inputs, samples the first 64KB to keep analysis fast.
pub fn analyze(input: &[u8]) -> DataProfile {
    analyze_with_sample(input, DEFAULT_SAMPLE_SIZE)
}

/// Analyze input data with an explicit sample size.
///
/// If `input.len() > sample_size`, only the first `sample_size` bytes are used.
pub fn analyze_with_sample(input: &[u8], sample_size: usize) -> DataProfile {
    if input.is_empty() {
        return DataProfile {
            byte_entropy: 0.0,
            autocorrelation_lag1: 0.0,
            run_ratio: 0.0,
            match_density: 0.0,
            distribution_shape: DistributionShape::Constant,
            distinct_bytes: 0,
            input_len: 0,
            numeric_gain: 0.0,
            numeric_stride: 0,
        };
    }

    let sample = if input.len() > sample_size && sample_size > 0 {
        &input[..sample_size]
    } else {
        input
    };

    let freq = frequency::get_frequency(sample);
    let byte_entropy = freq.entropy();
    let distinct_bytes = freq.used;
    let autocorrelation = autocorrelation_lag1(sample);
    let run_rat = run_ratio(sample);
    let match_dens = match_density_estimate(sample);
    let shape = classify_distribution(&freq, byte_entropy);
    let (numeric_gain, numeric_stride) = stride_decorrelation(sample, byte_entropy, distinct_bytes);

    DataProfile {
        byte_entropy,
        autocorrelation_lag1: autocorrelation,
        run_ratio: run_rat,
        match_density: match_dens,
        distribution_shape: shape,
        distinct_bytes,
        input_len: sample.len(),
        numeric_gain,
        numeric_stride,
    }
}

/// Pearson correlation between consecutive byte values.
///
/// Measures how similar `input[i]` is to `input[i+1]` on average.
/// Returns 0.0 for inputs shorter than 2 bytes.
fn autocorrelation_lag1(input: &[u8]) -> f32 {
    if input.len() < 2 {
        return 0.0;
    }

    let n = (input.len() - 1) as f64;

    // Compute means of x = input[0..n-1] and y = input[1..n].
    // Since x and y share input[1..n-1], the means are very similar
    // for large n. Compute exactly for correctness.
    let mut sum_x: u64 = 0;
    let mut sum_y: u64 = 0;
    for i in 0..input.len() - 1 {
        sum_x += input[i] as u64;
        sum_y += input[i + 1] as u64;
    }
    let mean_x = sum_x as f64 / n;
    let mean_y = sum_y as f64 / n;

    // Compute Pearson correlation: sum((xi - mx)(yi - my)) / sqrt(sum((xi-mx)^2) * sum((yi-my)^2))
    let mut sum_xy: f64 = 0.0;
    let mut sum_xx: f64 = 0.0;
    let mut sum_yy: f64 = 0.0;
    for i in 0..input.len() - 1 {
        let dx = input[i] as f64 - mean_x;
        let dy = input[i + 1] as f64 - mean_y;
        sum_xy += dx * dy;
        sum_xx += dx * dx;
        sum_yy += dy * dy;
    }

    let denom = (sum_xx * sum_yy).sqrt();
    if denom < 1e-12 {
        return 0.0; // constant data
    }

    (sum_xy / denom) as f32
}

/// Fraction of input bytes that are part of runs of 4+ identical bytes.
///
/// This matches the RLE encoding threshold used in `rle.rs`.
fn run_ratio(input: &[u8]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    let mut run_bytes: usize = 0;
    let mut i = 0;

    while i < input.len() {
        // Count the length of the current run
        let start = i;
        let val = input[i];
        i += 1;
        while i < input.len() && input[i] == val {
            i += 1;
        }
        let run_len = i - start;
        if run_len >= 4 {
            run_bytes += run_len;
        }
    }

    run_bytes as f32 / input.len() as f32
}

/// Quick LZ77 match density estimation.
///
/// Uses a simplified hash table (single entry per hash slot, no chains)
/// to estimate what fraction of positions have a match of length >= 3
/// within a nearby window. Runs in O(n) with a very small constant.
fn match_density_estimate(input: &[u8]) -> f32 {
    if input.len() < 3 {
        return 0.0;
    }

    let mut hash_table = vec![0u32; MATCH_HASH_SIZE];
    let mut match_count: usize = 0;
    let positions = input.len() - 2; // positions where we can compute hash3

    for i in 0..positions {
        let h = match_hash3(input, i);
        let prev = hash_table[h] as usize;

        // Check if previous position with same hash is within window
        // and actually matches 3 bytes
        if prev > 0 {
            let p = prev - 1; // stored as 1-indexed to distinguish from "empty"
            if i > p
                && i - p <= MATCH_WINDOW
                && input[p] == input[i]
                && input[p + 1] == input[i + 1]
            {
                // Already matched 2 bytes via hash + first check, verify third
                if p + 2 < input.len() && input[p + 2] == input[i + 2] {
                    match_count += 1;
                }
            }
        }

        // Store position (1-indexed)
        hash_table[h] = (i + 1) as u32;
    }

    match_count as f32 / positions as f32
}

/// Compute a hash for 3 bytes at the given position.
/// Same algorithm as `lz77::hash3` but with our local hash table size.
fn match_hash3(data: &[u8], pos: usize) -> usize {
    let h = (data[pos] as usize) << 10 ^ (data[pos + 1] as usize) << 5 ^ (data[pos + 2] as usize);
    h & MATCH_HASH_MASK
}

/// Minimum records per stride for the decorrelation estimate to be meaningful.
/// Below this, per-plane histograms are too sparse for even the bias-corrected
/// entropy estimate to be trustworthy.
const STRIDE_MIN_RECORDS: usize = 64;

/// Bias-corrected Shannon entropy (bits/symbol) of a 256-bin count histogram.
///
/// Applies the Miller-Madow correction `(K-1) / (2N ln 2)` (K = observed
/// symbols, N = total count): the empirical plug-in estimate systematically
/// *under*-estimates entropy on small samples, and a byte-plane split divides
/// the sample by the stride — without the correction, large strides would show
/// phantom decorrelation gains on random data.
fn entropy_from_counts(counts: &[u32; 256], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut h = 0.0;
    let mut observed = 0u32;
    for &c in counts.iter() {
        if c > 0 {
            observed += 1;
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h + (observed.saturating_sub(1)) as f64 / (2.0 * n * std::f64::consts::LN_2)
}

/// Estimate how much entropy the `Num` byte-plane transform would remove.
///
/// For each candidate stride `S` (the same set `numeric::encode` sweeps), the
/// sample is viewed as `S` interleaved byte-planes. Each plane is scored as
/// `min(H(plane), H(delta(plane)))` — exactly mirroring the per-plane gated
/// transform `Num` applies — and the length-weighted average is compared
/// against the pooled order-0 entropy. The best (gain, stride) pair over all
/// candidates is returned, with gain clamped to zero.
///
/// Fixed-stride numeric/record data (16-bit samples, fixed-width records)
/// scores a large gain (≥1 bit/byte) because separating the planes deshuffles
/// distinct column statistics and delta collapses smooth fields. Text, LZ-y
/// and random data score near zero: both sides of the comparison are
/// Miller-Madow bias-corrected (see [`entropy_from_counts`]), so the split
/// itself does not manufacture gain out of thinner histograms.
fn stride_decorrelation(sample: &[u8], byte_entropy: f32, distinct_bytes: u32) -> (f32, u32) {
    let n = sample.len();
    let mut best_gain = 0.0f64;
    let mut best_stride = 0u32;

    // Bias-correct the pooled order-0 entropy to match the per-plane side.
    let pooled_entropy = byte_entropy as f64
        + (distinct_bytes.saturating_sub(1)) as f64 / (2.0 * n as f64 * std::f64::consts::LN_2);

    for &s in &crate::numeric::CANDIDATE_STRIDES {
        if n < s * STRIDE_MIN_RECORDS {
            continue;
        }

        // Per-plane raw and lag-S delta histograms, built in one pass without
        // materializing the planes. The first record's delta is the raw byte,
        // matching `numeric::delta_fwd` (out[0] = p[0]).
        let mut raw = vec![[0u32; 256]; s];
        let mut delta = vec![[0u32; 256]; s];
        for (i, &b) in sample.iter().enumerate() {
            let k = i % s;
            raw[k][b as usize] += 1;
            let d = if i >= s {
                b.wrapping_sub(sample[i - s])
            } else {
                b
            };
            delta[k][d as usize] += 1;
        }

        // Length-weighted per-plane gated entropy, in bits/byte.
        let mut bits = 0.0f64;
        for k in 0..s {
            let nk = n / s + usize::from(k < n % s);
            let h = entropy_from_counts(&raw[k], nk).min(entropy_from_counts(&delta[k], nk));
            bits += nk as f64 * h;
        }
        let plane_entropy = bits / n as f64;

        let gain = pooled_entropy - plane_entropy;
        if gain > best_gain {
            best_gain = gain;
            best_stride = s as u32;
        }
    }

    (best_gain as f32, best_stride)
}

/// Classify the distribution shape from frequency data and entropy.
fn classify_distribution(freq: &frequency::FrequencyTable, entropy: f32) -> DistributionShape {
    if freq.used <= 1 || entropy < 0.1 {
        return DistributionShape::Constant;
    }

    if entropy > 7.0 {
        return DistributionShape::Uniform;
    }

    // Check bimodal before skewed: a bimodal distribution with 2 symbols
    // would otherwise match the skewed criteria.
    if is_bimodal(freq) {
        return DistributionShape::Bimodal;
    }

    if entropy < 3.0 && freq.used < 32 {
        return DistributionShape::Skewed;
    }

    DistributionShape::General
}

/// Detect bimodal distribution: two dominant peaks far apart.
fn is_bimodal(freq: &frequency::FrequencyTable) -> bool {
    if freq.total == 0 {
        return false;
    }

    let threshold = (freq.total as f64 * 0.05) as u32;

    // Find the two byte values with highest frequency
    let mut top1_idx: usize = 0;
    let mut top1_count: u32 = 0;
    let mut top2_idx: usize = 0;
    let mut top2_count: u32 = 0;

    for (i, &count) in freq.byte.iter().enumerate() {
        if count > top1_count {
            top2_idx = top1_idx;
            top2_count = top1_count;
            top1_idx = i;
            top1_count = count;
        } else if count > top2_count {
            top2_idx = i;
            top2_count = count;
        }
    }

    // Both peaks must be above threshold and separated by >= 64 byte values
    let separation = top1_idx.abs_diff(top2_idx);

    top1_count >= threshold && top2_count >= threshold && separation >= 64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let profile = analyze(&[]);
        assert_eq!(profile.input_len, 0);
        assert_eq!(profile.byte_entropy, 0.0);
        assert_eq!(profile.distinct_bytes, 0);
        assert_eq!(profile.distribution_shape, DistributionShape::Constant);
    }

    #[test]
    fn test_single_byte() {
        let profile = analyze(&[42]);
        assert_eq!(profile.input_len, 1);
        assert_eq!(profile.byte_entropy, 0.0);
        assert_eq!(profile.distinct_bytes, 1);
        assert_eq!(profile.autocorrelation_lag1, 0.0);
        assert_eq!(profile.run_ratio, 0.0); // run of 1, not >= 4
    }

    #[test]
    fn test_constant_data() {
        let input = vec![0xAA; 1000];
        let profile = analyze(&input);
        assert_eq!(profile.byte_entropy, 0.0);
        assert_eq!(profile.distinct_bytes, 1);
        assert_eq!(profile.distribution_shape, DistributionShape::Constant);
        assert!((profile.run_ratio - 1.0).abs() < 0.01);
        // Autocorrelation of constant data is 0 (no variance)
        assert_eq!(profile.autocorrelation_lag1, 0.0);
    }

    #[test]
    fn test_uniform_data() {
        // All 256 byte values equally represented
        let input: Vec<u8> = (0..=255u8).cycle().take(2560).collect();
        let profile = analyze(&input);
        assert!(
            profile.byte_entropy > 7.9,
            "entropy was {}",
            profile.byte_entropy
        );
        assert_eq!(profile.distinct_bytes, 256);
        assert_eq!(profile.distribution_shape, DistributionShape::Uniform);
        assert!(profile.run_ratio < 0.01);
    }

    #[test]
    fn test_ascending_data() {
        // Perfectly correlated consecutive bytes
        let input: Vec<u8> = (0..200).collect();
        let profile = analyze(&input);
        // High autocorrelation — consecutive bytes differ by exactly 1
        assert!(
            profile.autocorrelation_lag1 > 0.99,
            "autocorrelation was {}",
            profile.autocorrelation_lag1
        );
    }

    #[test]
    fn test_run_heavy_data() {
        // Runs of 10 of each byte value
        let mut input = Vec::new();
        for byte in 0..=25u8 {
            input.extend(std::iter::repeat_n(byte, 10));
        }
        let profile = analyze(&input);
        // Every byte is in a run of 10, so run_ratio should be 1.0
        assert!(
            profile.run_ratio > 0.95,
            "run_ratio was {}",
            profile.run_ratio
        );
    }

    #[test]
    fn test_no_runs() {
        // Alternating bytes — no runs of 4+
        let input: Vec<u8> = (0..500).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        let profile = analyze(&input);
        assert!(
            profile.run_ratio < 0.01,
            "run_ratio was {}",
            profile.run_ratio
        );
    }

    #[test]
    fn test_skewed_distribution() {
        // 90% one byte, 10% another
        let mut input = vec![0u8; 900];
        input.extend(vec![1u8; 100]);
        let profile = analyze(&input);
        assert_eq!(profile.distribution_shape, DistributionShape::Skewed);
    }

    #[test]
    fn test_bimodal_distribution() {
        // Two peaks separated by >= 64 byte values, each > 5% of total
        let mut input = vec![0u8; 500]; // peak at byte 0
        input.extend(vec![200u8; 500]); // peak at byte 200 (separation = 200)
        let profile = analyze(&input);
        assert_eq!(
            profile.distribution_shape,
            DistributionShape::Bimodal,
            "profile: {:?}",
            profile
        );
    }

    #[test]
    fn test_match_density_repetitive() {
        // Highly repetitive data should have high match density
        let pattern = b"abcdefgh";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let profile = analyze(&input);
        assert!(
            profile.match_density > 0.5,
            "match_density was {}",
            profile.match_density
        );
    }

    #[test]
    fn test_match_density_random() {
        // Pseudo-random data should have low match density
        // Use a simple LCG for deterministic "random" data
        let mut input = vec![0u8; 4096];
        let mut state: u32 = 12345;
        for byte in &mut input {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }
        let profile = analyze(&input);
        assert!(
            profile.match_density < 0.1,
            "match_density was {}",
            profile.match_density
        );
    }

    #[test]
    fn test_text_data() {
        let input = b"The quick brown fox jumps over the lazy dog. \
                       The quick brown fox jumps over the lazy dog. \
                       The quick brown fox jumps over the lazy dog. ";
        let profile = analyze(input);
        // Text typically has moderate entropy (3-5 bits)
        assert!(profile.byte_entropy > 2.0 && profile.byte_entropy < 6.0);
        // Text may classify as General or Bimodal depending on character distribution
        assert!(
            profile.distribution_shape == DistributionShape::General
                || profile.distribution_shape == DistributionShape::Bimodal,
            "shape was {:?}",
            profile.distribution_shape
        );
        // Repeated text should show some match density
        assert!(profile.match_density > 0.1);
    }

    #[test]
    fn test_sampling() {
        // Large input analyzed with small sample should give similar results
        let pattern = b"Hello, world! This is a test of the compression library. ";
        let mut input = Vec::new();
        for _ in 0..10000 {
            input.extend_from_slice(pattern);
        }

        let full = analyze_with_sample(&input, input.len());
        let sampled = analyze_with_sample(&input, 4096);

        // Entropy should be similar (within 1.0 bit)
        assert!(
            (full.byte_entropy - sampled.byte_entropy).abs() < 1.0,
            "full={} sampled={}",
            full.byte_entropy,
            sampled.byte_entropy
        );

        // Sampled length should be the sample size
        assert_eq!(sampled.input_len, 4096);
    }

    #[test]
    fn test_analyze_with_sample_larger_than_input() {
        let input = b"short";
        let profile = analyze_with_sample(input, 100000);
        assert_eq!(profile.input_len, 5);
    }

    /// Deterministic LCG byte stream (same generator as the match-density test).
    fn lcg_stream(n: usize, mut state: u32) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for byte in &mut out {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *byte = (state >> 16) as u8;
        }
        out
    }

    /// Bounded random-walk u16 LE samples: pooled byte entropy is high (the
    /// low bytes look near-uniform) but a stride-2 plane split + delta
    /// collapses it — the x-ray shape.
    fn u16_walk_samples(samples: usize) -> Vec<u8> {
        let mut input = Vec::with_capacity(samples * 2);
        let mut v: u16 = 30000;
        let mut state: u32 = 99;
        for _ in 0..samples {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let step = ((state >> 16) % 65) as i32 - 32; // [-32, +32]
            v = v.wrapping_add(step as u16);
            input.extend_from_slice(&v.to_le_bytes());
        }
        input
    }

    #[test]
    fn test_numeric_gain_u16_samples() {
        let input = u16_walk_samples(16384);
        let profile = analyze(&input);
        assert!(
            profile.byte_entropy > 6.0,
            "expected high pooled entropy, got {}",
            profile.byte_entropy
        );
        assert!(
            profile.numeric_gain > 1.5,
            "numeric_gain was {}",
            profile.numeric_gain
        );
        assert_eq!(profile.numeric_stride, 2, "profile: {:?}", profile);
    }

    #[test]
    fn test_numeric_stride_28_records() {
        // 28-byte records: constant magic, incrementing counter, slow
        // per-column walks, two noise columns. Only S=28 isolates the
        // columns (every other candidate stride mixes 7+ of them).
        let records = 2048;
        let mut input = Vec::with_capacity(records * 28);
        let mut counter: u16 = 0;
        let mut cols = [128u8; 20];
        let mut state: u32 = 7;
        for _ in 0..records {
            input.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
            input.extend_from_slice(&counter.to_le_bytes());
            counter = counter.wrapping_add(7);
            for _ in 0..2 {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                input.push((state >> 16) as u8);
            }
            for col in &mut cols {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                let step = ((state >> 16) % 5) as i32 - 2; // [-2, +2]
                *col = col.wrapping_add(step as u8);
                input.push(*col);
            }
        }
        let profile = analyze(&input);
        assert!(
            profile.numeric_gain > 2.0,
            "numeric_gain was {}",
            profile.numeric_gain
        );
        assert_eq!(profile.numeric_stride, 28, "profile: {:?}", profile);
    }

    #[test]
    fn test_numeric_gain_random_near_zero() {
        // Bias-corrected estimate must not manufacture gain on random data.
        let input = lcg_stream(65536, 12345);
        let profile = analyze(&input);
        assert!(
            profile.numeric_gain < 0.3,
            "spurious numeric_gain {} on random data",
            profile.numeric_gain
        );
    }

    #[test]
    fn test_numeric_gain_text_low() {
        // Text has no fixed-stride structure: gain stays under the routing
        // threshold at every candidate stride.
        let sentence = b"The quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::with_capacity(65536 + sentence.len());
        while input.len() < 65536 {
            input.extend_from_slice(sentence);
        }
        let profile = analyze(&input);
        assert!(
            profile.numeric_gain < 0.75,
            "numeric_gain was {} on text",
            profile.numeric_gain
        );
    }

    #[test]
    fn test_numeric_gain_short_input_skipped() {
        // Inputs below STRIDE_MIN_RECORDS * stride produce no estimate at all
        // rather than a noise-driven one.
        let profile = analyze(&lcg_stream(64, 5));
        assert_eq!(profile.numeric_gain, 0.0);
        assert_eq!(profile.numeric_stride, 0);
    }
}
