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
///
/// **Decode note:** Unlike BWT, the window-capped sort does not guarantee
/// valid LF-mapping properties. Decode reconstructs the original by storing
/// the sorted permutation in the wire format and inverting it directly.
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
    /// The sorted position array (permutation). Stored for decoding
    /// since window-capped sort doesn't guarantee valid LF-mapping.
    pub positions: Vec<u32>,
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
        let positions: Vec<u32> = Vec::new(); // empty = use BWT inverse
        return Some(FwstResult {
            data: bwt_result.data,
            positions,
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
        positions: positions.iter().map(|&p| p as u32).collect(),
        primary_index,
    })
}

/// Compress from an already-computed FwstResult. Used by the GPU path
/// to share wire format logic.
pub fn compress_from_result(fwst_result: &FwstResult, config: &FwstConfig) -> PzResult<Vec<u8>> {
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

    // Encode permutation (if present, i.e. w < n).
    if fwst_result.positions.is_empty() {
        // Full BWT mode: no permutation needed.
        output.extend_from_slice(&0u32.to_le_bytes());
    } else {
        let perm_bytes: Vec<u8> = fwst_result
            .positions
            .iter()
            .flat_map(|&p| p.to_le_bytes())
            .collect();
        let perm_raw_len = perm_bytes.len();
        let perm_fse = fse::encode(&perm_bytes);
        output.extend_from_slice(&(perm_raw_len as u32).to_le_bytes());
        output.extend_from_slice(&(perm_fse.len() as u32).to_le_bytes());
        output.extend_from_slice(&perm_fse);
    }

    output.extend_from_slice(&fse_data);

    Ok(output)
}

/// Compress input using the FWST pipeline.
///
/// Wire format:
/// ```text
/// [primary_index: u32 LE] [rle_len: u32 LE] [window: u16 LE]
/// [perm_len: u32 LE] [perm_fse_data: ...] [fse_data: ...]
/// ```
///
/// When `w >= n`, perm_len is 0 and decode uses standard BWT inverse.
/// When `w < n`, the sorted permutation is FSE-encoded for decode.
pub fn compress(input: &[u8], config: &FwstConfig) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let fwst_result = encode(input, config).ok_or(PzError::InvalidInput)?;
    compress_from_result(&fwst_result, config)
}

/// Decompress FWST data back to the original input.
///
/// When the permutation is stored (w < n), inverts the permutation directly.
/// When no permutation is stored (w >= n), uses standard BWT inverse.
pub fn decompress(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 14 {
        return Err(PzError::InvalidInput);
    }

    let primary_index = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let rle_len = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    // Window size stored for metadata.
    // let _window = u16::from_le_bytes([payload[8], payload[9]]);

    let perm_raw_len =
        u32::from_le_bytes([payload[10], payload[11], payload[12], payload[13]]) as usize;

    let mut pos = 14;

    // Read permutation if present.
    let positions: Option<Vec<u32>> = if perm_raw_len == 0 {
        None
    } else {
        if pos + 4 > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let perm_fse_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        pos += 4;
        if pos + perm_fse_len > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let perm_bytes = fse::decode(&payload[pos..pos + perm_fse_len], perm_raw_len)?;
        pos += perm_fse_len;
        let perm: Vec<u32> = perm_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Some(perm)
    };

    let fse_data = &payload[pos..];

    // Inverse pipeline: FSE → RLE → MTF
    let rle_data = fse::decode(fse_data, rle_len)?;
    let mtf_data = rle::decode(&rle_data)?;
    let fwst_data = mtf::decode(&mtf_data);

    let output = if let Some(positions) = positions {
        // Invert the permutation: fwst_data[i] = input[positions[i] - 1],
        // so input[positions[i] - 1] = fwst_data[i].
        let n = fwst_data.len();
        if positions.len() != n {
            return Err(PzError::InvalidInput);
        }
        let mut result = vec![0u8; n];
        for (i, &p) in positions.iter().enumerate() {
            let orig_pos = if p == 0 { n - 1 } else { p as usize - 1 };
            result[orig_pos] = fwst_data[i];
        }
        result
    } else {
        // Full BWT mode: use standard LF-mapping inverse.
        bwt::decode(&fwst_data, primary_index)?
    };

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

    #[test]
    fn roundtrip_repetitive_with_newline() {
        // Regression: window-capped sort produces non-BWT permutation on
        // data with many tied w-byte windows. Decode must not rely on LF-mapping.
        let input = b"Hello World! Repeated text Repeated text Repeated text\n";
        let compressed = compress(input, &FwstConfig { window: 8 }).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }
}
