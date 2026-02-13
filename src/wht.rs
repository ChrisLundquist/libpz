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

// ---------------------------------------------------------------------------
// Haar wavelet transform (reversed butterfly ordering)
// ---------------------------------------------------------------------------

/// Apply the forward Haar wavelet transform in-place.
///
/// Uses the standard "in-place lifting" Haar wavelet:
/// - At each scale, pairs of values are replaced by (sum, difference)
/// - The sums are packed into the left half, differences into the right
/// - Next scale operates only on the left half (sums from previous scale)
///
/// After all stages, `data[0]` contains the global sum (DC), and the
/// remaining entries hold detail coefficients at successively finer scales:
/// `data[1]` = coarsest detail, `data[2..4]` = next level, etc.
///
/// This is fundamentally different from WHT: Haar operates on a shrinking
/// subarray at each stage, while WHT always operates on the full array.
/// The result is a multi-resolution decomposition where most energy
/// concentrates in the low-index (coarse) coefficients for smooth signals.
pub fn haar_forward(data: &mut [i32]) {
    assert_eq!(
        data.len(),
        BLOCK_SIZE,
        "Haar block must be exactly {BLOCK_SIZE} elements"
    );

    // Process at each scale: n = 256, 128, 64, ..., 2
    let mut n = BLOCK_SIZE;
    while n >= 2 {
        let half = n / 2;
        // Compute sums and differences for pairs in data[0..n]
        // Use a temporary buffer to avoid aliasing
        let mut temp = vec![0i32; n];
        for i in 0..half {
            let a = data[2 * i];
            let b = data[2 * i + 1];
            temp[i] = a + b; // sum (goes to left half)
            temp[half + i] = a - b; // difference (goes to right half)
        }
        data[..n].copy_from_slice(&temp);
        n /= 2;
    }
}

/// Apply the inverse Haar wavelet transform in-place.
///
/// Reverses the forward Haar lifting. At each stage, reconstruct pairs
/// from (sum, difference): a = (s + d) / 2, b = (s - d) / 2.
/// Since forward used (a+b, a-b) without scaling, inverse needs the /2.
/// After all LOG2_N stages, divide by remaining scale factor.
///
/// The total scale factor from the forward transform is 2^LOG2_N = N,
/// but each inverse stage already divides by 2, so after LOG2_N stages
/// of /2, we've divided by 2^LOG2_N = N total. No final division needed.
pub fn haar_inverse(data: &mut [i32]) {
    assert_eq!(
        data.len(),
        BLOCK_SIZE,
        "Haar block must be exactly {BLOCK_SIZE} elements"
    );

    // Process at each scale: n = 2, 4, 8, ..., 256
    let mut n = 2;
    while n <= BLOCK_SIZE {
        let half = n / 2;
        let mut temp = vec![0i32; n];
        for i in 0..half {
            let s = data[i]; // sum
            let d = data[half + i]; // difference
                                    // a + b = s, a - b = d  =>  a = (s+d)/2, b = (s-d)/2
            temp[2 * i] = (s + d) / 2;
            temp[2 * i + 1] = (s - d) / 2;
        }
        data[..n].copy_from_slice(&temp);
        n *= 2;
    }
    // No final division: the /2 in each of the LOG2_N stages
    // accounts for the factor of 2^LOG2_N = N from the forward transform.
}

/// Apply sparse thresholding using the Haar transform instead of WHT.
///
/// Unlike [`sparse_threshold_block`], the threshold here is an absolute
/// value (not scaled by BLOCK_SIZE) because Haar coefficients at different
/// decomposition levels have different magnitudes. The finest-level
/// detail coefficients (indices 128..256) are single byte-pair differences,
/// while the DC coefficient (index 0) is the global sum.
///
/// Coefficients are stored as i32 LE (4 bytes) since Haar's DC can be
/// as large as 256 * 127 = 32,512 which fits i16, but intermediate
/// sums at coarser levels can exceed i16 range. For safety and
/// round-trip correctness, we use i32 throughout.
pub fn haar_sparse_threshold_block(input: &[u8], threshold: i32) -> SparseBlock {
    assert!(input.len() <= BLOCK_SIZE);

    let mut block = [0i32; BLOCK_SIZE];
    lift_bytes(input, &mut block[..input.len()]);
    for x in &mut block[input.len()..] {
        *x = -128;
    }

    haar_forward(&mut block);

    // Absolute threshold — not scaled by BLOCK_SIZE
    let mut sparse_data = Vec::new();
    let mut approx_coeffs = [0i32; BLOCK_SIZE];

    for (i, &coeff) in block.iter().enumerate() {
        if coeff.abs() >= threshold {
            sparse_data.push(i as u8);
            // Store as i32 LE (4 bytes) for lossless round-trip
            let bytes = coeff.to_le_bytes();
            sparse_data.push(bytes[0]);
            sparse_data.push(bytes[1]);
            sparse_data.push(bytes[2]);
            sparse_data.push(bytes[3]);
            approx_coeffs[i] = coeff;
        }
    }

    let num_significant = sparse_data.len() / 5; // 1 byte index + 4 bytes value

    haar_inverse(&mut approx_coeffs);

    let mut approx_bytes = [0u8; BLOCK_SIZE];
    unlift_bytes(&approx_coeffs, &mut approx_bytes);

    let mut residual = vec![0u8; input.len()];
    for (i, r) in residual.iter_mut().enumerate() {
        *r = input[i].wrapping_sub(approx_bytes[i]);
    }

    SparseBlock {
        num_significant,
        sparse_data,
        residual,
    }
}

/// Reconstruct a block from a Haar-based SparseBlock.
pub fn haar_reconstruct_sparse_block(sb: &SparseBlock, output: &mut [u8]) {
    let mut coeffs = [0i32; BLOCK_SIZE];

    // Unpack: 5 bytes per coefficient (1 byte index + 4 bytes i32 LE)
    for i in 0..sb.num_significant {
        let base = i * 5;
        let idx = sb.sparse_data[base] as usize;
        let val = i32::from_le_bytes([
            sb.sparse_data[base + 1],
            sb.sparse_data[base + 2],
            sb.sparse_data[base + 3],
            sb.sparse_data[base + 4],
        ]);
        coeffs[idx] = val;
    }

    haar_inverse(&mut coeffs);

    let mut approx_bytes = [0u8; BLOCK_SIZE];
    unlift_bytes(&coeffs, &mut approx_bytes);

    for (i, &r) in sb.residual.iter().enumerate() {
        output[i] = approx_bytes[i].wrapping_add(r);
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

/// Result of sparse thresholding on a single WHT block.
///
/// After forward WHT, coefficients above the threshold are stored explicitly
/// as (index, value) pairs. The residual is computed as:
///   residual = original_block - iWHT(sparse_coefficients_only)
/// This is exact and lossless.
#[derive(Debug, Clone)]
pub struct SparseBlock {
    /// Number of significant coefficients kept.
    pub num_significant: usize,
    /// Packed sparse data: for each significant coeff, [index: u8, value_lo: u8, value_hi: u8].
    /// Total size = num_significant * 3 bytes.
    pub sparse_data: Vec<u8>,
    /// Residual: original bytes minus the approximation from sparse coefficients.
    /// These values cluster tightly around zero, ideal for entropy coding.
    pub residual: Vec<u8>,
}

/// Apply sparse thresholding to a single 256-byte block.
///
/// 1. Lift bytes → centered i32, forward WHT
/// 2. Partition coefficients by |coeff| >= threshold * N (unnormalized domain)
/// 3. Reconstruct approximation from significant coefficients via iWHT
/// 4. Compute exact residual = original - approximation
///
/// The threshold operates in the *normalized* domain conceptually:
/// a coefficient is "significant" if |coeff| >= threshold * BLOCK_SIZE.
/// We work in unnormalized domain to avoid any rounding.
pub fn sparse_threshold_block(input: &[u8], threshold: i32) -> SparseBlock {
    assert!(input.len() <= BLOCK_SIZE);

    let mut block = [0i32; BLOCK_SIZE];
    lift_bytes(input, &mut block[..input.len()]);
    for x in &mut block[input.len()..] {
        *x = -128;
    }

    forward(&mut block);

    // The threshold in unnormalized domain
    let thresh_unnorm = threshold * BLOCK_SIZE as i32;

    // Partition: significant vs zeroed
    let mut sparse_data = Vec::new();
    let mut approx_coeffs = [0i32; BLOCK_SIZE];

    for (i, &coeff) in block.iter().enumerate() {
        if coeff.abs() >= thresh_unnorm {
            // Store as (index: u8, value: i16 LE)
            sparse_data.push(i as u8);
            let val = coeff as i16;
            sparse_data.push(val as u8);
            sparse_data.push((val >> 8) as u8);
            approx_coeffs[i] = coeff;
        }
    }

    let num_significant = sparse_data.len() / 3;

    // Reconstruct approximation via inverse WHT
    inverse(&mut approx_coeffs);

    // Compute residual in byte domain
    let mut approx_bytes = [0u8; BLOCK_SIZE];
    unlift_bytes(&approx_coeffs, &mut approx_bytes);

    // Residual = original - approximation, stored as wrapping byte difference
    let mut residual = vec![0u8; input.len()];
    for (i, r) in residual.iter_mut().enumerate() {
        *r = input[i].wrapping_sub(approx_bytes[i]);
    }

    SparseBlock {
        num_significant,
        sparse_data,
        residual,
    }
}

/// Reconstruct the original block from a SparseBlock.
///
/// 1. Unpack sparse coefficients into a 256-element array (rest zero)
/// 2. Inverse WHT → approximation bytes
/// 3. Add residual to get exact original
pub fn reconstruct_sparse_block(sb: &SparseBlock, output: &mut [u8]) {
    let mut coeffs = [0i32; BLOCK_SIZE];

    // Unpack sparse data
    for i in 0..sb.num_significant {
        let base = i * 3;
        let idx = sb.sparse_data[base] as usize;
        let val = i16::from_le_bytes([sb.sparse_data[base + 1], sb.sparse_data[base + 2]]);
        coeffs[idx] = val as i32;
    }

    inverse(&mut coeffs);

    let mut approx_bytes = [0u8; BLOCK_SIZE];
    unlift_bytes(&coeffs, &mut approx_bytes);

    // Add residual
    for (i, &r) in sb.residual.iter().enumerate() {
        output[i] = approx_bytes[i].wrapping_add(r);
    }
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

    #[test]
    fn test_sparse_block_roundtrip() {
        // Verify sparse thresholding + reconstruction is lossless
        let input: Vec<u8> = (0..=255).collect();
        for threshold in [1, 2, 4, 8, 16, 32] {
            let sb = sparse_threshold_block(&input, threshold);
            let mut output = [0u8; BLOCK_SIZE];
            reconstruct_sparse_block(&sb, &mut output);
            assert_eq!(
                &output[..],
                &input[..],
                "round-trip failed at threshold={threshold}"
            );
        }
    }

    #[test]
    fn test_sparse_block_roundtrip_text() {
        let text = b"The quick brown fox jumps over the lazy dog. ";
        let input: Vec<u8> = text.iter().cycle().take(256).copied().collect();
        for threshold in [1, 4, 16, 64] {
            let sb = sparse_threshold_block(&input, threshold);
            let mut output = [0u8; BLOCK_SIZE];
            reconstruct_sparse_block(&sb, &mut output);
            assert_eq!(
                &output[..],
                &input[..],
                "round-trip failed at threshold={threshold}"
            );
        }
    }

    #[test]
    fn test_sparse_sparsity_increases_with_threshold() {
        let input: Vec<u8> = (0..=255).collect();
        let sb_low = sparse_threshold_block(&input, 1);
        let sb_high = sparse_threshold_block(&input, 16);
        // Higher threshold should keep fewer coefficients
        assert!(
            sb_high.num_significant <= sb_low.num_significant,
            "low={} high={}",
            sb_low.num_significant,
            sb_high.num_significant
        );
    }

    /// The critical experiment: actual rANS compression comparison.
    ///
    /// For each file in the Canterbury corpus, at various thresholds, measure:
    /// - raw_rans: rANS(original_bytes) — the baseline
    /// - sparse_total: sparse_header + sparse_data + rANS(residual_bytes)
    ///
    /// If sparse_total < raw_rans for any threshold, the spectral
    /// sparsification is earning its keep.
    #[test]
    fn test_sparse_rans_comparison_corpus() {
        use crate::rans;

        let corpus_dir = std::path::Path::new("samples/cantrbry");
        if !corpus_dir.exists() {
            eprintln!("Skipping: samples/cantrbry not found");
            return;
        }

        let thresholds = [1, 2, 4, 8, 16, 32, 64];

        println!("\n=== WHT Sparse + rANS vs Raw rANS (Canterbury Corpus) ===\n");
        println!(
            "{:<16} {:>7} {:>10} | {:>5} {:>6} {:>10} {:>10} {:>7}",
            "File", "Size", "rANS raw", "Thr", "Kept", "Sparse", "rANS res", "Total"
        );
        println!("{}", "-".repeat(90));

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

            // Baseline: rANS on raw bytes
            let raw_rans = rans::encode(&data);
            let raw_rans_size = raw_rans.len();

            // Verify raw rANS round-trips
            let raw_decoded = rans::decode(&raw_rans, data.len()).unwrap();
            assert_eq!(
                raw_decoded, data,
                "raw rANS round-trip failed for {file_name}"
            );

            let mut best_total = usize::MAX;
            let mut best_thresh = 0;

            for &threshold in &thresholds {
                let num_blocks = data.len().div_ceil(BLOCK_SIZE);
                let mut total_sparse_bytes: usize = 0;
                let mut total_kept: usize = 0;
                let mut all_residuals = Vec::with_capacity(data.len());

                // 4 bytes: original length header
                total_sparse_bytes += 4;
                // 1 byte per block: num_significant count
                total_sparse_bytes += num_blocks;

                for b in 0..num_blocks {
                    let start = b * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(data.len());
                    let chunk = &data[start..end];

                    let sb = sparse_threshold_block(chunk, threshold);

                    // Verify lossless round-trip
                    let mut reconstructed = vec![0u8; chunk.len()];
                    reconstruct_sparse_block(&sb, &mut reconstructed);
                    assert_eq!(
                        &reconstructed[..],
                        chunk,
                        "sparse round-trip failed: file={file_name} block={b} threshold={threshold}"
                    );

                    total_sparse_bytes += sb.sparse_data.len(); // 3 bytes per kept coeff
                    total_kept += sb.num_significant;
                    all_residuals.extend_from_slice(&sb.residual);
                }

                // Compress residuals with rANS
                let residual_rans = rans::encode(&all_residuals);
                let residual_rans_size = residual_rans.len();

                // Verify residual rANS round-trips
                let res_decoded = rans::decode(&residual_rans, all_residuals.len()).unwrap();
                assert_eq!(res_decoded, all_residuals, "residual rANS failed");

                let total = total_sparse_bytes + residual_rans_size;
                let avg_kept = total_kept as f64 / num_blocks as f64;

                if total < best_total {
                    best_total = total;
                    best_thresh = threshold;
                }

                println!(
                    "{:<16} {:>7} {:>10} | {:>5} {:>6.1} {:>10} {:>10} {:>7}",
                    if threshold == thresholds[0] {
                        file_name
                    } else {
                        ""
                    },
                    if threshold == thresholds[0] {
                        format!("{}", data.len())
                    } else {
                        String::new()
                    },
                    if threshold == thresholds[0] {
                        format!("{raw_rans_size}")
                    } else {
                        String::new()
                    },
                    threshold,
                    avg_kept,
                    total_sparse_bytes,
                    residual_rans_size,
                    total
                );
            }

            let ratio = best_total as f64 / raw_rans_size as f64;
            let winner = if best_total < raw_rans_size {
                "WHT WINS"
            } else {
                "raw wins"
            };
            println!(
                "{:>16} best: thr={:<3} total={:<8} ratio={:.3} {}",
                "", best_thresh, best_total, ratio, winner
            );
            println!();
        }
    }

    /// Same experiment on synthetic data where WHT showed promise.
    #[test]
    fn test_sparse_rans_comparison_synthetic() {
        use crate::rans;

        println!("\n=== WHT Sparse + rANS vs Raw rANS (Synthetic Data) ===\n");
        println!(
            "{:<20} {:>8} {:>10} | {:>5} {:>6} {:>10} {:>10} {:>8} {:>8}",
            "Data type", "Size", "rANS raw", "Thr", "Kept", "Sparse", "rANS res", "Total", "Ratio"
        );
        println!("{}", "-".repeat(105));

        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            // Smooth ramp — WHT concentrates energy heavily
            ("smooth_ramp_1k", (0..=255u8).cycle().take(1024).collect()),
            // Sawtooth
            (
                "sawtooth_1k",
                (0..1024).map(|i| ((i % 16) * 16) as u8).collect(),
            ),
            // Step function
            ("step_1k", {
                let mut v = vec![50u8; 512];
                v.extend(vec![200u8; 512]);
                v
            }),
            // Sine wave
            (
                "sine_1k",
                (0..1024)
                    .map(|i| {
                        let t = i as f64 * std::f64::consts::TAU / 256.0;
                        ((t.sin() * 100.0) + 128.0).clamp(0.0, 255.0) as u8
                    })
                    .collect(),
            ),
            // Text
            ("text_1k", {
                let text = b"The quick brown fox jumps over the lazy dog. ";
                text.iter().cycle().take(1024).copied().collect()
            }),
            // Pseudo-random (control — WHT should not help)
            ("random_1k", {
                let mut state: u32 = 12345;
                (0..1024)
                    .map(|_| {
                        state = state.wrapping_mul(1103515245).wrapping_add(12345);
                        (state >> 16) as u8
                    })
                    .collect()
            }),
            // Mostly constant with occasional spikes
            ("sparse_signal_1k", {
                let mut v = vec![128u8; 1024];
                for i in (0..1024).step_by(64) {
                    v[i] = 255;
                }
                v
            }),
            // Slowly varying gradient
            (
                "gradient_4k",
                (0..4096).map(|i| (i * 256 / 4096) as u8).collect(),
            ),
        ];

        let thresholds = [1, 2, 4, 8, 16, 32, 64];

        for (name, data) in &test_cases {
            let raw_rans = rans::encode(data);
            let raw_rans_size = raw_rans.len();

            let mut best_total = usize::MAX;
            let mut best_thresh = 0;

            for &threshold in &thresholds {
                let num_blocks = data.len().div_ceil(BLOCK_SIZE);
                let mut total_sparse_bytes: usize = 4 + num_blocks; // header + per-block counts
                let mut total_kept: usize = 0;
                let mut all_residuals = Vec::with_capacity(data.len());

                for b in 0..num_blocks {
                    let start = b * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(data.len());
                    let chunk = &data[start..end];

                    let sb = sparse_threshold_block(chunk, threshold);
                    total_sparse_bytes += sb.sparse_data.len();
                    total_kept += sb.num_significant;
                    all_residuals.extend_from_slice(&sb.residual);
                }

                let residual_rans = rans::encode(&all_residuals);
                let total = total_sparse_bytes + residual_rans.len();
                let avg_kept = total_kept as f64 / num_blocks as f64;

                if total < best_total {
                    best_total = total;
                    best_thresh = threshold;
                }

                let ratio = total as f64 / raw_rans_size as f64;
                println!(
                    "{:<20} {:>8} {:>10} | {:>5} {:>6.1} {:>10} {:>10} {:>8} {:>8.3}",
                    if threshold == thresholds[0] {
                        name
                    } else {
                        &""
                    },
                    if threshold == thresholds[0] {
                        format!("{}", data.len())
                    } else {
                        String::new()
                    },
                    if threshold == thresholds[0] {
                        format!("{raw_rans_size}")
                    } else {
                        String::new()
                    },
                    threshold,
                    avg_kept,
                    total_sparse_bytes,
                    residual_rans.len(),
                    total,
                    ratio
                );
            }

            let best_ratio = best_total as f64 / raw_rans_size as f64;
            let winner = if best_total < raw_rans_size {
                "WHT WINS"
            } else {
                "raw wins"
            };
            println!(
                "{:>20} best: thr={:<3} ratio={:.3} {}",
                "", best_thresh, best_ratio, winner
            );
            println!();
        }
    }

    // -----------------------------------------------------------------------
    // Haar transform tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_haar_forward_inverse_roundtrip() {
        let original: Vec<i32> = (0..BLOCK_SIZE as i32).map(|i| i - 128).collect();
        let mut data = original.clone();
        haar_forward(&mut data);
        assert_ne!(data, original);
        haar_inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_haar_roundtrip_constant() {
        let mut data = [42i32; BLOCK_SIZE];
        let original = data;
        haar_forward(&mut data);
        // DC component should be 42 * 256 = 10752, all others 0
        assert_eq!(data[0], 42 * BLOCK_SIZE as i32);
        for &c in &data[1..] {
            assert_eq!(c, 0);
        }
        haar_inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_haar_roundtrip_random() {
        let mut state: u32 = 98765;
        let mut data = [0i32; BLOCK_SIZE];
        for x in &mut data {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *x = ((state >> 16) as i32 % 256) - 128;
        }
        let original = data;
        haar_forward(&mut data);
        haar_inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_haar_energy_scaling() {
        // Haar's lifting scheme packs sums and differences at different
        // scales, so the energy relationship differs from WHT.
        // Verify that the transform is still exactly invertible (the key property).
        let original: Vec<i32> = (0..BLOCK_SIZE as i32).map(|i| i - 128).collect();
        let mut data = original.clone();
        let input_energy: i64 = data.iter().map(|&x| (x as i64) * (x as i64)).sum();

        haar_forward(&mut data);
        let spectral_energy: i64 = data.iter().map(|&x| (x as i64) * (x as i64)).sum();

        // Energy is NOT preserved 1:1 with Haar (unlike WHT).
        // The ratio depends on the data. Just verify it's nonzero and
        // the transform round-trips correctly.
        assert!(spectral_energy > 0);
        assert!(input_energy > 0);

        haar_inverse(&mut data);
        assert_eq!(data, original, "Haar round-trip must be exact");
    }

    #[test]
    fn test_haar_sparse_roundtrip() {
        let input: Vec<u8> = (0..=255).collect();
        // Absolute thresholds for Haar (not scaled by N)
        for threshold in [1, 4, 16, 64, 256, 1024] {
            let sb = haar_sparse_threshold_block(&input, threshold);
            let mut output = [0u8; BLOCK_SIZE];
            haar_reconstruct_sparse_block(&sb, &mut output);
            assert_eq!(
                &output[..],
                &input[..],
                "Haar sparse round-trip failed at threshold={threshold}"
            );
        }
    }

    /// Diagnostic: examine whether Haar and WHT produce different coefficient
    /// distributions when applied to the same data.
    #[test]
    fn test_haar_vs_wht_coefficient_analysis() {
        // Use non-periodic data: first 256 bytes of a gradient
        let input: Vec<u8> = (0..256)
            .map(|i| {
                let smooth = (i as f64 * 0.05).sin() * 80.0 + 128.0;
                let noise = ((i as u32).wrapping_mul(2654435761) >> 28) as f64;
                (smooth + noise).clamp(0.0, 255.0) as u8
            })
            .collect();

        let mut wht_block = [0i32; BLOCK_SIZE];
        let mut haar_block = [0i32; BLOCK_SIZE];
        lift_bytes(&input, &mut wht_block);
        lift_bytes(&input, &mut haar_block);

        forward(&mut wht_block);
        haar_forward(&mut haar_block);

        // Energy differs between WHT and Haar (different normalization per level)
        let wht_energy: i64 = wht_block.iter().map(|&x| (x as i64) * (x as i64)).sum();
        let haar_energy: i64 = haar_block.iter().map(|&x| (x as i64) * (x as i64)).sum();
        println!("WHT total energy: {wht_energy}, Haar total energy: {haar_energy}");

        // Check if coefficient arrays differ, or are just permutations
        let coeffs_differ = wht_block != haar_block;

        let mut wht_sorted: Vec<i32> = wht_block.iter().map(|x| x.abs()).collect();
        let mut haar_sorted: Vec<i32> = haar_block.iter().map(|x| x.abs()).collect();
        wht_sorted.sort_unstable();
        haar_sorted.sort_unstable();
        let sorted_mags_match = wht_sorted == haar_sorted;

        println!("\n=== Haar vs WHT Coefficient Analysis ===\n");
        println!("Coefficients differ:          {coeffs_differ}");
        println!("Sorted magnitudes match:      {sorted_mags_match}");

        // Count near-zero at various thresholds
        println!(
            "\n{:<12} {:>12} {:>12}",
            "Threshold", "WHT near-0", "Haar near-0"
        );
        for thresh in [1, 2, 4, 8, 16, 32, 64] {
            let t = thresh * BLOCK_SIZE as i32;
            let wht_near = wht_block.iter().filter(|&&c| c.abs() < t).count();
            let haar_near = haar_block.iter().filter(|&&c| c.abs() < t).count();
            println!("{:<12} {:>12} {:>12}", thresh, wht_near, haar_near);
        }

        // The critical insight: if sorted magnitudes always match, then
        // Haar and WHT are just permutations of each other for any input,
        // and magnitude-based thresholding cannot distinguish them.
        // In that case, the difference would only matter for:
        // 1. Position-dependent coding (run-length on coefficient indices)
        // 2. Coefficient prediction / DPCM
        // 3. Subband-specific quantization (Haar has meaningful subbands)
        if sorted_mags_match {
            println!("\nHaar and WHT produce the same multiset of coefficient magnitudes.");
            println!("Magnitude-based thresholding cannot distinguish them.");
            println!("To exploit Haar's locality, need subband-aware coding.");
        }
    }

    /// Head-to-head: Haar vs WHT vs raw rANS on Canterbury corpus.
    ///
    /// Tests whether Haar's multi-scale decomposition gives better
    /// energy compaction than WHT's global Walsh-function basis for
    /// real-world byte data.
    #[test]
    fn test_haar_vs_wht_vs_raw_corpus() {
        use crate::rans;

        let corpus_dir = std::path::Path::new("samples/cantrbry");
        if !corpus_dir.exists() {
            eprintln!("Skipping: samples/cantrbry not found");
            return;
        }

        println!("\n=== Haar vs WHT vs Raw rANS (Canterbury Corpus) ===\n");
        println!(
            "{:<16} {:>7} {:>10} {:>10} {:>6} {:>10} {:>6} {:>8}",
            "File", "Size", "Raw rANS", "WHT best", "ratio", "Haar best", "ratio", "Winner"
        );
        println!("{}", "-".repeat(95));

        // WHT thresholds are scaled by BLOCK_SIZE in sparse_threshold_block
        let wht_thresholds = [1, 2, 4, 8, 16, 32, 64];
        // Haar thresholds are absolute values (not scaled)
        // Fine details are byte diffs (~0-50), coarse are larger
        let haar_thresholds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

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

            let raw_rans_size = rans::encode(&data).len();

            // WHT sweep
            let mut wht_best = usize::MAX;
            for &threshold in &wht_thresholds {
                let num_blocks = data.len().div_ceil(BLOCK_SIZE);
                let mut sparse_bytes: usize = 4 + num_blocks;
                let mut all_residuals = Vec::with_capacity(data.len());

                for b in 0..num_blocks {
                    let start = b * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(data.len());
                    let sb = sparse_threshold_block(&data[start..end], threshold);
                    sparse_bytes += sb.sparse_data.len();
                    all_residuals.extend_from_slice(&sb.residual);
                }
                let total = sparse_bytes + rans::encode(&all_residuals).len();
                wht_best = wht_best.min(total);
            }

            // Haar sweep (absolute thresholds, 5 bytes per kept coeff)
            let mut haar_best = usize::MAX;
            for &threshold in &haar_thresholds {
                let num_blocks = data.len().div_ceil(BLOCK_SIZE);
                let mut sparse_bytes: usize = 4 + num_blocks;
                let mut all_residuals = Vec::with_capacity(data.len());

                for b in 0..num_blocks {
                    let start = b * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(data.len());
                    let sb = haar_sparse_threshold_block(&data[start..end], threshold);
                    sparse_bytes += sb.sparse_data.len();
                    all_residuals.extend_from_slice(&sb.residual);
                }
                let total = sparse_bytes + rans::encode(&all_residuals).len();
                haar_best = haar_best.min(total);
            }

            let wht_ratio = wht_best as f64 / raw_rans_size as f64;
            let haar_ratio = haar_best as f64 / raw_rans_size as f64;
            let winner = if haar_best < wht_best && haar_best < raw_rans_size {
                "Haar"
            } else if wht_best < haar_best && wht_best < raw_rans_size {
                "WHT"
            } else {
                "Raw"
            };

            println!(
                "{:<16} {:>7} {:>10} {:>10} {:>6.3} {:>10} {:>6.3} {:>8}",
                file_name,
                data.len(),
                raw_rans_size,
                wht_best,
                wht_ratio,
                haar_best,
                haar_ratio,
                winner
            );
        }
    }

    /// Head-to-head on synthetic data.
    #[test]
    fn test_haar_vs_wht_vs_raw_synthetic() {
        use crate::rans;

        println!("\n=== Haar vs WHT vs Raw rANS (Synthetic Data) ===\n");
        println!(
            "{:<20} {:>8} {:>10} {:>10} {:>6} {:>10} {:>6} {:>8}",
            "Data type", "Size", "Raw rANS", "WHT best", "ratio", "Haar best", "ratio", "Winner"
        );
        println!("{}", "-".repeat(100));

        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("smooth_ramp_1k", (0..=255u8).cycle().take(1024).collect()),
            (
                "sawtooth_1k",
                (0..1024).map(|i| ((i % 16) * 16) as u8).collect(),
            ),
            ("step_1k", {
                let mut v = vec![50u8; 512];
                v.extend(vec![200u8; 512]);
                v
            }),
            (
                "sine_1k",
                (0..1024)
                    .map(|i| {
                        let t = i as f64 * std::f64::consts::TAU / 256.0;
                        ((t.sin() * 100.0) + 128.0).clamp(0.0, 255.0) as u8
                    })
                    .collect(),
            ),
            ("text_1k", {
                let text = b"The quick brown fox jumps over the lazy dog. ";
                text.iter().cycle().take(1024).copied().collect()
            }),
            ("random_1k", {
                let mut state: u32 = 12345;
                (0..1024)
                    .map(|_| {
                        state = state.wrapping_mul(1103515245).wrapping_add(12345);
                        (state >> 16) as u8
                    })
                    .collect()
            }),
            ("sparse_signal_1k", {
                let mut v = vec![128u8; 1024];
                for i in (0..1024).step_by(64) {
                    v[i] = 255;
                }
                v
            }),
            (
                "gradient_4k",
                (0..4096).map(|i| (i * 256 / 4096) as u8).collect(),
            ),
            // New: piecewise-constant (staircase) — Haar's sweet spot
            (
                "staircase_1k",
                (0..1024).map(|i| ((i / 32) * 8 + 16) as u8).collect(),
            ),
            // New: smooth + noise
            (
                "smooth_noisy_1k",
                (0..1024)
                    .map(|i| {
                        let smooth = (i as f64 * 0.025).sin() * 80.0 + 128.0;
                        let noise = ((i as u32).wrapping_mul(2654435761) >> 27) as f64;
                        (smooth + noise).clamp(0.0, 255.0) as u8
                    })
                    .collect(),
            ),
        ];

        let wht_thresholds = [1, 2, 4, 8, 16, 32, 64];
        let haar_thresholds = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

        for (name, data) in &test_cases {
            let raw_rans_size = rans::encode(data).len();

            let mut wht_best = usize::MAX;
            for &threshold in &wht_thresholds {
                let num_blocks = data.len().div_ceil(BLOCK_SIZE);
                let mut sparse_bytes: usize = 4 + num_blocks;
                let mut all_residuals = Vec::with_capacity(data.len());
                for b in 0..num_blocks {
                    let start = b * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(data.len());
                    let sb = sparse_threshold_block(&data[start..end], threshold);
                    sparse_bytes += sb.sparse_data.len();
                    all_residuals.extend_from_slice(&sb.residual);
                }
                wht_best = wht_best.min(sparse_bytes + rans::encode(&all_residuals).len());
            }

            let mut haar_best = usize::MAX;
            for &threshold in &haar_thresholds {
                let num_blocks = data.len().div_ceil(BLOCK_SIZE);
                let mut sparse_bytes: usize = 4 + num_blocks;
                let mut all_residuals = Vec::with_capacity(data.len());
                for b in 0..num_blocks {
                    let start = b * BLOCK_SIZE;
                    let end = (start + BLOCK_SIZE).min(data.len());
                    let sb = haar_sparse_threshold_block(&data[start..end], threshold);
                    sparse_bytes += sb.sparse_data.len();
                    all_residuals.extend_from_slice(&sb.residual);
                }
                haar_best = haar_best.min(sparse_bytes + rans::encode(&all_residuals).len());
            }

            let wht_ratio = wht_best as f64 / raw_rans_size as f64;
            let haar_ratio = haar_best as f64 / raw_rans_size as f64;
            let winner = if haar_best < wht_best && haar_best < raw_rans_size {
                "Haar"
            } else if wht_best < haar_best && wht_best < raw_rans_size {
                "WHT"
            } else if haar_best <= raw_rans_size || wht_best <= raw_rans_size {
                if haar_ratio <= wht_ratio {
                    "Haar"
                } else {
                    "WHT"
                }
            } else {
                "Raw"
            };

            println!(
                "{:<20} {:>8} {:>10} {:>10} {:>6.3} {:>10} {:>6.3} {:>8}",
                name,
                data.len(),
                raw_rans_size,
                wht_best,
                wht_ratio,
                haar_best,
                haar_ratio,
                winner
            );
        }
    }
}
