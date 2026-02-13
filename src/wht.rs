/// Walsh-Hadamard Transform (WHT) for spectral decorrelation.
///
/// Implements the fast Walsh-Hadamard Transform in-place using a butterfly
/// network. The WHT is a real-valued orthogonal transform that decomposes
/// a signal into Walsh functions (rectangular waveforms taking values ±1).
///
/// # Properties
///
/// - **Self-inverse**: WHT(WHT(x)) = N·x (exact in integers, no rounding).
/// - **Integer-only**: requires only addition and subtraction — no multiplies,
///   no twiddle factors, no modular arithmetic.
/// - **O(N log N)**: same asymptotic cost as FFT, but with trivial operations.
/// - **Decorrelation**: for structured data (smooth signals, sensor readings,
///   gradients), the WHT concentrates energy into a small number of
///   coefficients, producing a sparser representation that entropy coders
///   (rANS, FSE) can exploit.
///
/// # Block size
///
/// Fixed at 256 points (8 butterfly stages). This keeps values manageable
/// in i32 (max magnitude 128 × 256 = 32,768), minimizes shared memory
/// pressure for future GPU shaders, and provides 16 independent sub-blocks
/// per 4KB page for parallelism.
///
/// # Usage
///
/// ```
/// use pz::wht;
///
/// let input = vec![100i32; 256];
/// let mut coeffs = input.clone();
/// wht::forward(&mut coeffs);
///
/// // Inverse: apply forward again, then divide by N
/// wht::inverse(&mut coeffs);
/// assert_eq!(coeffs, input);
/// ```
use crate::{PzError, PzResult};

/// Block size for WHT: 256 points = 8 butterfly stages.
pub const BLOCK_SIZE: usize = 256;

/// log2(BLOCK_SIZE)
const LOG2_N: u32 = 8;

/// Apply the forward Walsh-Hadamard Transform in-place.
///
/// Input must have length equal to `BLOCK_SIZE` (256).
/// After forward transform, coefficients are unnormalized (scaled by 1,
/// not 1/√N). The DC component (index 0) equals the sum of all inputs.
pub fn forward(data: &mut [i32]) {
    assert_eq!(
        data.len(),
        BLOCK_SIZE,
        "WHT block must be exactly {BLOCK_SIZE} elements"
    );
    butterfly(data);
}

/// Apply the inverse Walsh-Hadamard Transform in-place.
///
/// Since WHT is self-inverse up to a factor of N, this applies the
/// butterfly network then divides every element by N. This is exact
/// because forward WHT of integer input always produces values divisible
/// by 1 (and iWHT(forward(x)) = N·x, so dividing by N recovers x exactly).
pub fn inverse(data: &mut [i32]) {
    assert_eq!(
        data.len(),
        BLOCK_SIZE,
        "WHT block must be exactly {BLOCK_SIZE} elements"
    );
    butterfly(data);
    for x in data.iter_mut() {
        *x /= BLOCK_SIZE as i32;
    }
}

/// Core butterfly network: 8 stages for 256-point WHT.
///
/// Each stage s operates on pairs separated by `half = N >> (s+1)`.
/// For each pair (a, b):
///   a' = a + b
///   b' = a - b
///
/// This is the "natural order" (Hadamard-ordered) WHT.
#[inline]
fn butterfly(data: &mut [i32]) {
    let mut half = BLOCK_SIZE >> 1; // 128, 64, 32, 16, 8, 4, 2, 1
    for _ in 0..LOG2_N {
        let step = half << 1;
        let mut j = 0;
        while j < BLOCK_SIZE {
            for k in 0..half {
                let a = data[j + k];
                let b = data[j + k + half];
                data[j + k] = a + b;
                data[j + k + half] = a - b;
            }
            j += step;
        }
        half >>= 1;
    }
}

/// Lift a byte slice into centered i32 values for WHT processing.
///
/// Each byte b is mapped to (b as i32) - 128, centering the range
/// around zero: [0, 255] → [-128, 127].
pub fn lift_bytes(input: &[u8], output: &mut [i32]) {
    assert_eq!(input.len(), output.len());
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = src as i32 - 128;
    }
}

/// Convert WHT coefficients back to bytes after inverse transform.
///
/// Adds 128 to each value and clamps to [0, 255]. For a lossless
/// round-trip (forward → inverse), clamping should never activate.
pub fn unlift_bytes(input: &[i32], output: &mut [u8]) {
    assert_eq!(input.len(), output.len());
    for (dst, &src) in output.iter_mut().zip(input.iter()) {
        *dst = (src + 128).clamp(0, 255) as u8;
    }
}

/// Encode a byte buffer using WHT decorrelation.
///
/// Pads the input to a multiple of `BLOCK_SIZE`, applies the forward WHT
/// to each block, and maps coefficients back to bytes for entropy coding.
///
/// Returns a header (original length as 4 bytes LE) followed by the
/// WHT coefficient bytes.
pub fn encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let num_blocks = input.len().div_ceil(BLOCK_SIZE);
    let padded_len = num_blocks * BLOCK_SIZE;

    // Header: original length (4 bytes LE)
    let mut output = Vec::with_capacity(4 + padded_len);
    output.extend_from_slice(&(input.len() as u32).to_le_bytes());

    let mut block = [0i32; BLOCK_SIZE];

    for b in 0..num_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(input.len());
        let chunk = &input[start..end];

        // Lift bytes to centered i32
        lift_bytes(chunk, &mut block[..chunk.len()]);
        // Zero-pad if this is the last partial block
        for x in &mut block[chunk.len()..] {
            *x = -128; // padding byte 0 centered = -128
        }

        // Forward WHT
        forward(&mut block);

        // Map coefficients to bytes for entropy coding.
        // Coefficients are in [-32768, 32767] range for 256-point WHT of byte data.
        // Store as 2 bytes each (i16 LE) since they don't fit in u8.
        for &coeff in &block {
            output.extend_from_slice(&(coeff as i16).to_le_bytes());
        }
    }

    output
}

/// Decode a WHT-encoded buffer back to the original bytes.
///
/// Reads the header, applies inverse WHT to each block, and unlifts
/// back to bytes.
pub fn decode(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if input.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let original_len = u32::from_le_bytes([input[0], input[1], input[2], input[3]]) as usize;
    let coeff_data = &input[4..];

    let num_blocks = original_len.div_ceil(BLOCK_SIZE);
    let expected_coeff_bytes = num_blocks * BLOCK_SIZE * 2; // 2 bytes per i16 coefficient

    if coeff_data.len() < expected_coeff_bytes {
        return Err(PzError::InvalidInput);
    }

    let mut output = Vec::with_capacity(original_len);
    let mut block = [0i32; BLOCK_SIZE];
    let mut byte_block = [0u8; BLOCK_SIZE];

    for b in 0..num_blocks {
        let coeff_start = b * BLOCK_SIZE * 2;

        // Read i16 coefficients
        for (i, slot) in block.iter_mut().enumerate() {
            let offset = coeff_start + i * 2;
            let val = i16::from_le_bytes([coeff_data[offset], coeff_data[offset + 1]]);
            *slot = val as i32;
        }

        // Inverse WHT
        inverse(&mut block);

        // Unlift back to bytes
        unlift_bytes(&block, &mut byte_block);

        let remaining = original_len - output.len();
        let take = remaining.min(BLOCK_SIZE);
        output.extend_from_slice(&byte_block[..take]);
    }

    Ok(output)
}

/// Compute Shannon entropy of WHT coefficients stored as i16 values.
///
/// This measures the entropy over the full i16 symbol space, which tells
/// us how compressible the coefficient stream is. For comparison with
/// raw byte entropy, divide by 2 (since each coefficient is 2 bytes)
/// or compare bits-per-original-byte directly.
pub fn coefficient_entropy(coefficients: &[i16]) -> f32 {
    if coefficients.is_empty() {
        return 0.0;
    }

    // Count frequencies of each i16 value using a HashMap (sparse distribution)
    use std::collections::HashMap;
    let mut freq: HashMap<i16, u32> = HashMap::new();
    for &c in coefficients {
        *freq.entry(c).or_insert(0) += 1;
    }

    let total = coefficients.len() as f32;
    freq.values()
        .map(|&count| {
            let p = count as f32 / total;
            -p * p.log2()
        })
        .sum()
}

/// Analyze spectral properties of a data block after WHT.
///
/// Returns a tuple of:
/// - `energy_ratio_top_k`: fraction of total energy in the top K coefficients
/// - `near_zero_fraction`: fraction of coefficients with |value| <= threshold
///
/// This helps determine whether WHT decorrelation is useful for a given
/// data class.
pub fn spectral_sparsity(data: &[u8], energy_top_k: usize, zero_threshold: i32) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    let num_blocks = data.len().div_ceil(BLOCK_SIZE);
    let mut total_energy: f64 = 0.0;
    let mut top_k_energy: f64 = 0.0;
    let mut near_zero_count: usize = 0;
    let mut total_coeffs: usize = 0;

    let mut block = [0i32; BLOCK_SIZE];

    for b in 0..num_blocks {
        let start = b * BLOCK_SIZE;
        let end = (start + BLOCK_SIZE).min(data.len());
        let chunk = &data[start..end];

        lift_bytes(chunk, &mut block[..chunk.len()]);
        for x in &mut block[chunk.len()..] {
            *x = -128;
        }

        forward(&mut block);

        // Compute energy (sum of squares) for this block
        let mut energies: Vec<f64> = block.iter().map(|&c| (c as f64) * (c as f64)).collect();
        let block_energy: f64 = energies.iter().sum();
        total_energy += block_energy;

        // Top-K energy
        energies.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let k = energy_top_k.min(BLOCK_SIZE);
        top_k_energy += energies[..k].iter().sum::<f64>();

        // Near-zero count
        for &c in &block {
            if c.abs() <= zero_threshold {
                near_zero_count += 1;
            }
        }
        total_coeffs += BLOCK_SIZE;
    }

    let energy_ratio = if total_energy > 0.0 {
        (top_k_energy / total_energy) as f32
    } else {
        1.0
    };
    let near_zero_frac = near_zero_count as f32 / total_coeffs as f32;

    (energy_ratio, near_zero_frac)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_inverse_roundtrip() {
        // WHT(WHT(x)) = N*x, so inverse(forward(x)) = x
        let original: Vec<i32> = (0..BLOCK_SIZE as i32).map(|i| i - 128).collect();
        let mut data = original.clone();

        forward(&mut data);
        // Coefficients should differ from original
        assert_ne!(data, original);

        inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_roundtrip_all_zeros() {
        let mut data = [0i32; BLOCK_SIZE];
        let original = data;
        forward(&mut data);
        inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_roundtrip_constant() {
        let mut data = [42i32; BLOCK_SIZE];
        let original = data;
        forward(&mut data);
        // DC component should be 42 * 256 = 10752, all others 0
        assert_eq!(data[0], 42 * BLOCK_SIZE as i32);
        for &c in &data[1..] {
            assert_eq!(c, 0);
        }
        inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_roundtrip_alternating() {
        let mut data: Vec<i32> = (0..BLOCK_SIZE)
            .map(|i| if i % 2 == 0 { 1 } else { -1 })
            .collect();
        let original = data.clone();
        forward(&mut data);
        inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_roundtrip_random_values() {
        // Deterministic pseudo-random
        let mut state: u32 = 54321;
        let mut data = [0i32; BLOCK_SIZE];
        for x in &mut data {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *x = ((state >> 16) as i32 % 256) - 128;
        }
        let original = data;
        forward(&mut data);
        inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_energy_conservation() {
        // Parseval's theorem: sum(x^2) = sum(X^2) / N for normalized WHT
        // For unnormalized: sum(X^2) = N * sum(x^2)
        let mut data: Vec<i32> = (0..BLOCK_SIZE as i32).map(|i| i - 128).collect();
        let input_energy: i64 = data.iter().map(|&x| (x as i64) * (x as i64)).sum();

        forward(&mut data);
        let spectral_energy: i64 = data.iter().map(|&x| (x as i64) * (x as i64)).sum();

        assert_eq!(spectral_energy, input_energy * BLOCK_SIZE as i64);
    }

    #[test]
    fn test_byte_encode_decode_roundtrip() {
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_byte_encode_decode_short() {
        // Shorter than one block
        let input = b"Hello, Walsh-Hadamard!";
        let encoded = encode(input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_byte_encode_decode_multi_block() {
        // Multiple blocks
        let input: Vec<u8> = (0..700).map(|i| (i % 256) as u8).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_byte_encode_decode_empty() {
        let encoded = encode(&[]);
        let decoded = decode(&encoded).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_invalid_input() {
        assert_eq!(decode(&[0, 0]).unwrap_err(), PzError::InvalidInput);
    }

    #[test]
    fn test_coefficient_entropy_constant() {
        let coeffs = vec![0i16; 256];
        assert_eq!(coefficient_entropy(&coeffs), 0.0);
    }

    #[test]
    fn test_spectral_sparsity_constant_data() {
        // Constant data should be maximally sparse (all energy in DC)
        let data = vec![128u8; 256];
        let (energy_top1, near_zero) = spectral_sparsity(&data, 1, 0);
        assert!(
            (energy_top1 - 1.0).abs() < 0.01,
            "energy_top1={energy_top1}"
        );
        // All non-DC coefficients should be zero
        assert!(near_zero > 0.99, "near_zero={near_zero}");
    }

    /// Entropy analysis: compare raw byte entropy vs WHT coefficient entropy.
    ///
    /// This is the key experiment. If WHT reduces entropy for structured data
    /// but not for random data, the spectral decorrelation hypothesis holds.
    #[test]
    fn test_entropy_analysis_synthetic() {
        use crate::frequency;

        println!("\n=== WHT Entropy Analysis (Synthetic Data) ===\n");
        println!(
            "{:<25} {:>10} {:>10} {:>10} {:>10}",
            "Data type", "H_raw", "H_wht", "Delta", "Bits/orig"
        );

        // Test different data patterns
        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            // Smooth ramp — high autocorrelation, WHT should help
            ("smooth_ramp", (0..=255).cycle().take(256).collect()),
            // Sawtooth wave
            (
                "sawtooth_16",
                (0..256).map(|i| ((i % 16) * 16) as u8).collect(),
            ),
            // Slow sine-like (quantized)
            (
                "sine_approx",
                (0..256)
                    .map(|i| {
                        let t = i as f64 * std::f64::consts::TAU / 256.0;
                        ((t.sin() * 100.0) + 128.0).clamp(0.0, 255.0) as u8
                    })
                    .collect(),
            ),
            // Step function (two levels)
            ("step_function", {
                let mut v = vec![50u8; 128];
                v.extend(vec![200u8; 128]);
                v
            }),
            // Text-like (ASCII range, moderate correlation)
            ("ascii_text", {
                let text = b"The quick brown fox jumps over the lazy dog. ";
                text.iter().cycle().take(256).copied().collect()
            }),
            // Pseudo-random (WHT should NOT help)
            ("pseudo_random", {
                let mut state: u32 = 12345;
                (0..256)
                    .map(|_| {
                        state = state.wrapping_mul(1103515245).wrapping_add(12345);
                        (state >> 16) as u8
                    })
                    .collect()
            }),
            // Constant (trivial case)
            ("constant_128", vec![128u8; 256]),
            // Sensor-like: slow drift with noise
            (
                "sensor_drift",
                (0..256)
                    .map(|i| {
                        let base = (i as f64 * 0.1).sin() * 30.0 + 128.0;
                        let noise = ((i as u32).wrapping_mul(2654435761) >> 28) as f64;
                        (base + noise).clamp(0.0, 255.0) as u8
                    })
                    .collect(),
            ),
        ];

        for (name, data) in &test_cases {
            // Raw byte entropy
            let raw_freq = frequency::get_frequency(data);
            let h_raw = raw_freq.entropy();

            // WHT coefficient entropy
            let mut block = [0i32; BLOCK_SIZE];
            lift_bytes(data, &mut block);
            forward(&mut block);

            // Convert to i16 for coefficient entropy measurement
            let coeffs: Vec<i16> = block.iter().map(|&c| c as i16).collect();
            let h_wht = coefficient_entropy(&coeffs);

            // Bits per original byte: each coefficient is 2 bytes, so
            // total bits for the block = h_wht * 256 coefficients
            // bits per original byte = (h_wht * 256) / 256 = h_wht
            // But each coeff symbol is from a larger alphabet, so we
            // need bits_total = h_wht * num_coefficients
            // bits_per_orig_byte = bits_total / num_orig_bytes
            let bits_per_orig = h_wht * 2.0; // 2 bytes of coefficients per 1 byte of input

            let delta = h_raw - bits_per_orig;

            println!(
                "{:<25} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
                name, h_raw, h_wht, delta, bits_per_orig
            );
        }

        // The test passes — it's informational. We assert basic sanity only.
        // Constant data should have 0 raw entropy and 0 WHT entropy
        let const_data = vec![128u8; 256];
        let raw_freq = frequency::get_frequency(&const_data);
        assert_eq!(raw_freq.entropy(), 0.0);
    }

    /// Test entropy on real corpus files (Canterbury corpus).
    ///
    /// This is the critical experiment: does WHT reduce effective entropy
    /// on real-world data that this library actually compresses?
    #[test]
    fn test_entropy_analysis_corpus() {
        use crate::frequency;

        let corpus_dir = std::path::Path::new("samples/cantrbry");
        if !corpus_dir.exists() {
            eprintln!("Skipping corpus test: samples/cantrbry not found (run scripts/setup.sh)");
            return;
        }

        println!("\n=== WHT Entropy Analysis (Canterbury Corpus) ===\n");
        println!(
            "{:<20} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12}",
            "File", "Size", "H_raw", "H_wht_sym", "Bits/orig", "Delta", "Sparse90%"
        );

        let mut files: Vec<_> = std::fs::read_dir(corpus_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        files.sort_by_key(|e| e.file_name());

        for entry in &files {
            let path = entry.path();
            let data = std::fs::read(&path).unwrap();
            if data.is_empty() {
                continue;
            }

            let file_name = path.file_name().unwrap().to_str().unwrap();

            // Raw byte entropy
            let raw_freq = frequency::get_frequency(&data);
            let h_raw = raw_freq.entropy();

            // WHT analysis: process all blocks, aggregate coefficient statistics
            let num_blocks = data.len().div_ceil(BLOCK_SIZE);
            let mut all_coeffs: Vec<i16> = Vec::with_capacity(num_blocks * BLOCK_SIZE);
            let mut block = [0i32; BLOCK_SIZE];

            for b in 0..num_blocks {
                let start = b * BLOCK_SIZE;
                let end = (start + BLOCK_SIZE).min(data.len());
                let chunk = &data[start..end];

                lift_bytes(chunk, &mut block[..chunk.len()]);
                for x in &mut block[chunk.len()..] {
                    *x = -128;
                }
                forward(&mut block);

                for &c in &block {
                    all_coeffs.push(c as i16);
                }
            }

            let h_wht = coefficient_entropy(&all_coeffs);
            let bits_per_orig = h_wht * 2.0;
            let delta = h_raw - bits_per_orig;

            // Spectral sparsity: energy in top 10% of coefficients
            let (energy_90, _) = spectral_sparsity(&data, BLOCK_SIZE / 10, 0);

            println!(
                "{:<20} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>12.3} {:>12.1}%",
                file_name,
                data.len(),
                h_raw,
                h_wht,
                bits_per_orig,
                delta,
                energy_90 * 100.0
            );
        }

        // Informational test — always passes
    }
}
