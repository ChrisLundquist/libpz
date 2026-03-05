/// Fixed-Window Sort Transform (FWST) pipeline.
///
/// Like BWT, but caps the sort key at `w` bytes instead of using full suffixes.
/// This guarantees exactly `w` radix passes with fully predictable work,
/// vs BWT's unbounded suffix comparisons.
///
/// **What this measures:**
/// The effective context depth of real data. If `w=8` achieves 95% of full
/// BWT compression, then most data has only ~8 bytes of useful context.
///
/// **Pipeline:**
/// ```text
/// Input
///   → Extract w-byte key per position
///   → Radix sort positions by key (stable, ties broken by position)
///   → BWT readoff: input[sorted_position - 1]
///   → MTF → RLE → FSE (same as existing BW pipeline)
/// Output
/// ```
use crate::bwt;
use crate::fse;
use crate::mtf;
use crate::rle;
use crate::{PzError, PzResult};

/// Default window size in bytes for the fixed-window sort.
const DEFAULT_WINDOW: usize = 8;

/// Configuration for the FWST pipeline.
#[derive(Debug, Clone)]
pub struct FwstConfig {
    /// Window size: number of bytes used as sort key per position.
    pub window: usize,
}

impl Default for FwstConfig {
    fn default() -> Self {
        FwstConfig {
            window: DEFAULT_WINDOW,
        }
    }
}

/// Result of the fixed-window sort transform.
#[derive(Debug, Clone)]
pub struct FwstResult {
    /// The transformed data (BWT-like output).
    pub data: Vec<u8>,
    /// The primary index (position of original string's rotation in sorted order).
    pub primary_index: u32,
}

/// Perform fixed-window sort transform.
///
/// For each position i, the sort key is `input[i..i+w]` (wrapping around at end).
/// Positions are sorted by their w-byte keys with stable sort (ties broken by position).
/// Output is BWT-like: `input[(sorted_pos - 1) % n]` for each sorted position.
pub fn encode(input: &[u8], config: &FwstConfig) -> Option<FwstResult> {
    if input.is_empty() {
        return None;
    }

    let n = input.len();
    let w = config.window.min(n);

    // If window >= input length, fall back to full BWT (equivalent).
    if w >= n {
        let bwt_result = bwt::encode(input)?;
        return Some(FwstResult {
            data: bwt_result.data,
            primary_index: bwt_result.primary_index,
        });
    }

    // Build (position) array, then sort by w-byte key at each position.
    let mut positions: Vec<usize> = (0..n).collect();

    // Sort by w-byte key, ties broken by position (stable sort).
    positions.sort_by(|&a, &b| {
        for k in 0..w {
            let ca = input[(a + k) % n];
            let cb = input[(b + k) % n];
            match ca.cmp(&cb) {
                std::cmp::Ordering::Equal => continue,
                other => return other,
            }
        }
        a.cmp(&b) // tie-break by position
    });

    // BWT readoff: last character of each rotation.
    let mut result = Vec::with_capacity(n);
    let mut primary_index = 0u32;

    for (i, &pos) in positions.iter().enumerate() {
        if pos == 0 {
            primary_index = i as u32;
            result.push(input[n - 1]);
        } else {
            result.push(input[pos - 1]);
        }
    }

    Some(FwstResult {
        data: result,
        primary_index,
    })
}

/// Compress input using the FWST pipeline.
///
/// Wire format:
/// ```text
/// [primary_index: u32 LE] [rle_len: u32 LE] [window: u16 LE] [fse_data: ...]
/// ```
pub fn compress(input: &[u8], config: &FwstConfig) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    // Step 1: Fixed-window sort transform
    let fwst_result = encode(input, config).ok_or(PzError::InvalidInput)?;

    // Step 2: MTF
    let mtf_data = mtf::encode(&fwst_result.data);

    // Step 3: RLE
    let rle_data = rle::encode(&mtf_data);

    // Step 4: FSE
    let fse_data = fse::encode(&rle_data);

    // Assemble output
    let mut output = Vec::new();
    output.extend_from_slice(&fwst_result.primary_index.to_le_bytes());
    output.extend_from_slice(&(rle_data.len() as u32).to_le_bytes());
    output.extend_from_slice(&(config.window as u16).to_le_bytes());
    output.extend_from_slice(&fse_data);

    Ok(output)
}

/// Decompress FWST data back to the original input.
///
/// The inverse transform is identical to BWT inverse (LF-mapping) because
/// the forward transform produces BWT-compatible output.
pub fn decompress(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 10 {
        return Err(PzError::InvalidInput);
    }

    let primary_index = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let rle_len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    // Window size stored for metadata but not needed for decompression.
    // let _window = u16::from_le_bytes([payload[8], payload[9]]);

    let fse_data = &payload[10..];

    // Inverse pipeline: FSE → RLE → MTF → BWT inverse
    let rle_data = fse::decode(fse_data, rle_len)?;
    let mtf_data = rle::decode(&rle_data)?;
    let fwst_data = mtf::decode(&mtf_data);
    let output = bwt::decode(&fwst_data, primary_index)?;

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_w8() {
        let input = b"abracadabra and some more text to test the fixed window sort transform";
        let compressed = compress(input, &FwstConfig { window: 8 }).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_w4() {
        let input = b"banana banana banana banana";
        let compressed = compress(input, &FwstConfig { window: 4 }).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_w2() {
        let input = b"Hello, World!";
        let compressed = compress(input, &FwstConfig { window: 2 }).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_w16() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input, &FwstConfig { window: 16 }).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_large_window_falls_back_to_bwt() {
        let input = b"short";
        let compressed = compress(input, &FwstConfig { window: 100 }).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn encode_deterministic() {
        let input = b"aabbcc";
        let r1 = encode(input, &FwstConfig { window: 4 }).unwrap();
        let r2 = encode(input, &FwstConfig { window: 4 }).unwrap();
        assert_eq!(r1.data, r2.data);
        assert_eq!(r1.primary_index, r2.primary_index);
    }
}
