/// Context-Segmented BWT (CSBWT) pipeline.
///
/// Replaces MTF + RLE (serial) with context segmentation + per-context
/// entropy coding (parallel). After BWT construction, the suffix array
/// is used to detect context boundaries. Adjacent BWT positions whose
/// suffixes share the same k-byte prefix belong to the same context
/// segment. Each segment is then independently FSE-encoded.
///
/// **Pipeline:**
/// ```text
/// Input
///   → BWT via suffix array (reuse existing)
///   → Context segmentation from suffix array
///   → Per-segment histogram computation
///   → Small-segment merging
///   → Multi-stream FSE encoding (one stream per merged segment)
/// Output
/// ```
use crate::bwt;
use crate::fse;
use crate::{PzError, PzResult};

/// Default context order (number of prefix bytes to compare).
const DEFAULT_CONTEXT_ORDER: usize = 1;

/// Default minimum segment size after merging.
const DEFAULT_MIN_SEGMENT_SIZE: usize = 64;

/// Result of the CSBWT forward transform.
#[derive(Debug, Clone)]
pub struct CsbwtResult {
    /// BWT primary index (for inverse BWT).
    pub primary_index: u32,
    /// BWT-transformed data.
    pub bwt_data: Vec<u8>,
    /// Segment boundaries (start indices into bwt_data) after merging.
    /// Always starts with 0. Length = num_segments + 1 (last entry = bwt_data.len()).
    pub segment_boundaries: Vec<usize>,
}

/// Configuration for the CSBWT pipeline.
#[derive(Debug, Clone)]
pub struct CsbwtConfig {
    /// Context order k: compare first k bytes of suffixes to detect boundaries.
    pub context_order: usize,
    /// Minimum segment size after merging.
    pub min_segment_size: usize,
}

impl Default for CsbwtConfig {
    fn default() -> Self {
        CsbwtConfig {
            context_order: DEFAULT_CONTEXT_ORDER,
            min_segment_size: DEFAULT_MIN_SEGMENT_SIZE,
        }
    }
}

/// Perform CSBWT forward transform: BWT + context segmentation.
///
/// Returns the BWT data with segment boundaries that can be used for
/// per-segment entropy coding.
pub fn encode(input: &[u8], config: &CsbwtConfig) -> PzResult<CsbwtResult> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let n = input.len();

    // Step 1: Build suffix array and BWT simultaneously.
    let sa = bwt::build_suffix_array_public(input);
    let mut bwt_data = Vec::with_capacity(n);
    let mut primary_index = 0u32;

    for (i, &sa_val) in sa.iter().enumerate() {
        if sa_val == 0 {
            primary_index = i as u32;
            bwt_data.push(input[n - 1]);
        } else {
            bwt_data.push(input[sa_val - 1]);
        }
    }

    // Step 2: Context boundary detection.
    // For adjacent SA entries, compare the first k characters of the suffixes
    // they point to. If they differ, mark a context boundary.
    let k = config.context_order;
    let mut boundaries = Vec::new();
    boundaries.push(0usize); // first segment always starts at 0

    for i in 1..n {
        let sa_prev = sa[i - 1];
        let sa_curr = sa[i];
        let differs = suffix_prefix_differs(input, sa_prev, sa_curr, k);
        if differs {
            boundaries.push(i);
        }
    }
    boundaries.push(n); // sentinel: end of last segment

    // Step 3: Merge small segments.
    let merged = merge_small_segments(&boundaries, config.min_segment_size);

    Ok(CsbwtResult {
        primary_index,
        bwt_data,
        segment_boundaries: merged,
    })
}

/// Compare first k characters of two suffixes (treating input as circular).
/// Returns true if they differ in any of the first k positions.
fn suffix_prefix_differs(input: &[u8], sa_a: usize, sa_b: usize, k: usize) -> bool {
    let n = input.len();
    for j in 0..k {
        let a = input[(sa_a + j) % n];
        let b = input[(sa_b + j) % n];
        if a != b {
            return true;
        }
    }
    false
}

/// Public wrapper for merge_small_segments (used by GPU path).
pub fn merge_small_segments_public(boundaries: &[usize], min_size: usize) -> Vec<usize> {
    merge_small_segments(boundaries, min_size)
}

/// Merge adjacent segments until each has at least `min_size` symbols.
///
/// Scans left to right, accumulating segments until the cumulative size
/// meets the threshold. Returns the merged boundary list.
fn merge_small_segments(boundaries: &[usize], min_size: usize) -> Vec<usize> {
    if boundaries.len() <= 2 {
        return boundaries.to_vec();
    }

    let mut merged = Vec::new();
    merged.push(0);

    let num_segments = boundaries.len() - 1;
    let mut accum = 0usize;

    for i in 0..num_segments {
        let seg_size = boundaries[i + 1] - boundaries[i];
        accum += seg_size;

        // Emit a merged boundary when we have enough data,
        // unless this is the last segment (always emit the final boundary).
        if accum >= min_size && i + 1 < num_segments {
            merged.push(boundaries[i + 1]);
            accum = 0;
        }
    }

    merged.push(*boundaries.last().unwrap());
    merged
}

/// Compress using the CSBWT pipeline: BWT + context segmentation + per-segment FSE.
///
/// Wire format:
/// ```text
/// [primary_index: u32]
/// [num_segments: u32]
/// [segment_sizes: u32 × num_segments]  (original segment sizes)
/// [per-segment FSE data]:
///   for each segment:
///     [fse_len: u32] [fse_data: fse_len bytes]
/// ```
pub fn compress(input: &[u8], config: &CsbwtConfig) -> PzResult<Vec<u8>> {
    let result = encode(input, config)?;
    let num_segments = result.segment_boundaries.len() - 1;

    let mut output = Vec::new();
    output.extend_from_slice(&result.primary_index.to_le_bytes());
    output.extend_from_slice(&(num_segments as u32).to_le_bytes());

    // Write segment sizes (original, before FSE).
    for i in 0..num_segments {
        let seg_size = result.segment_boundaries[i + 1] - result.segment_boundaries[i];
        output.extend_from_slice(&(seg_size as u32).to_le_bytes());
    }

    // Per-segment FSE encode.
    for i in 0..num_segments {
        let start = result.segment_boundaries[i];
        let end = result.segment_boundaries[i + 1];
        let segment = &result.bwt_data[start..end];

        let acc = adaptive_accuracy_log(segment);
        let fse_data = fse::encode_with_accuracy(segment, acc);
        output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
        output.extend_from_slice(&fse_data);
    }

    Ok(output)
}

/// Decompress CSBWT data back to the original input.
pub fn decompress(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 8 {
        return Err(PzError::InvalidInput);
    }

    let primary_index = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let num_segments =
        u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;

    let header_size = 8 + num_segments * 4;
    if payload.len() < header_size {
        return Err(PzError::InvalidInput);
    }

    // Read segment sizes.
    let mut segment_sizes = Vec::with_capacity(num_segments);
    for i in 0..num_segments {
        let off = 8 + i * 4;
        let size = u32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]) as usize;
        segment_sizes.push(size);
    }

    // Decode each segment's FSE data and reconstruct BWT output.
    let mut bwt_data = Vec::with_capacity(orig_len);
    let mut pos = header_size;

    for &seg_size in &segment_sizes {
        if pos + 4 > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let fse_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;

        if pos + fse_len > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let segment = fse::decode(&payload[pos..pos + fse_len], seg_size)?;
        bwt_data.extend_from_slice(&segment);
        pos += fse_len;
    }

    if bwt_data.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    // Inverse BWT to recover original data.
    bwt::decode(&bwt_data, primary_index)
}

/// Public wrapper for adaptive_accuracy_log (used by GPU path).
pub fn adaptive_accuracy_log_public(data: &[u8]) -> u8 {
    adaptive_accuracy_log(data)
}

/// Choose FSE accuracy_log based on distinct symbol count.
fn adaptive_accuracy_log(data: &[u8]) -> u8 {
    if data.is_empty() {
        return fse::DEFAULT_ACCURACY_LOG;
    }
    let mut seen = [false; 256];
    for &b in data {
        seen[b as usize] = true;
    }
    let distinct = seen.iter().filter(|&&s| s).count() as u32;
    let target = 4 * distinct;
    let log = if target <= 1 {
        0
    } else {
        32 - (target - 1).leading_zeros()
    };
    log.clamp(fse::MIN_ACCURACY_LOG as u32, fse::MAX_ACCURACY_LOG as u32) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple() {
        let input = b"banana";
        let config = CsbwtConfig::default();
        let compressed = compress(input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_repeated() {
        let input = b"abcabcabcabcabcabcabcabcabcabcabcabc";
        let config = CsbwtConfig::default();
        let compressed = compress(input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_longer() {
        let mut input = Vec::new();
        for i in 0..1000 {
            input.push((i % 256) as u8);
        }
        let config = CsbwtConfig::default();
        let compressed = compress(&input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_context_order_2() {
        let input = b"abracadabra";
        let config = CsbwtConfig {
            context_order: 2,
            min_segment_size: 2,
        };
        let compressed = compress(input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_merge_small_segments() {
        // 10 elements, boundaries every 2
        let boundaries = vec![0, 2, 4, 6, 8, 10];
        let merged = merge_small_segments(&boundaries, 5);
        // Should merge into segments of at least 5
        for i in 0..merged.len() - 1 {
            let size = merged[i + 1] - merged[i];
            // Last segment might be smaller
            if i + 1 < merged.len() - 1 {
                assert!(size >= 5, "segment {i} too small: {size}");
            }
        }
        assert_eq!(*merged.first().unwrap(), 0);
        assert_eq!(*merged.last().unwrap(), 10);
    }
}
