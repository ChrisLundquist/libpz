/// Sort-based LZ77 match finding (SortLZ).
///
/// Replaces hash-table-based LZ77 match finding with sort-based matching:
/// 1. Hash every 4-byte window
/// 2. Radix sort (hash, position) pairs
/// 3. Adjacent-pair match verification (extend matches)
/// 4. Best-match selection per position
/// 5. Greedy or lazy parsing
///
/// The approach is fully deterministic and uses no atomic operations,
/// making it ideal for GPU execution.
use crate::fse;
use crate::lz_token::{LzSeqEncoder, LzToken, TokenEncoder};
use crate::{PzError, PzResult};

/// Minimum match length.
const MIN_MATCH: usize = 4;

/// Maximum match candidates to consider per position.
const MAX_CANDIDATES: usize = 8;

/// Default maximum window size for back-references.
const DEFAULT_MAX_WINDOW: usize = 65535;

/// Satisficing threshold: once a position has a match this long,
/// skip further improvement attempts. Greatly reduces verification
/// work on repetitive data where most positions have very long matches.
/// 256 covers the full Deflate match range (258) and provides
/// diminishing returns beyond this point.
const SATISFICE_LEN: u16 = 256;

/// Configuration for the SortLZ pipeline.
#[derive(Debug, Clone)]
pub struct SortLzConfig {
    /// Maximum back-reference distance.
    pub max_window: usize,
    /// Whether to use lazy parsing (1-step lookahead).
    pub lazy_parsing: bool,
    /// Minimum match length.
    pub min_match: usize,
    /// Maximum match candidates per position.
    pub max_candidates: usize,
    /// Maximum match length (e.g., 258 for Deflate, u16::MAX for others).
    pub max_match_len: u16,
}

impl Default for SortLzConfig {
    fn default() -> Self {
        SortLzConfig {
            max_window: DEFAULT_MAX_WINDOW,
            lazy_parsing: true,
            min_match: MIN_MATCH,
            max_candidates: MAX_CANDIDATES,
            max_match_len: u16::MAX,
        }
    }
}

impl SortLzConfig {
    /// Build config for use as a match source in LZ77 pipelines.
    ///
    /// Uses the LZ77 window size (32KB) and match constraints.
    /// Note: SortLZ minimum match is 4 (hash-based), not LZ77's 3.
    pub fn for_lz77(max_match_len: u16) -> Self {
        SortLzConfig {
            max_window: crate::lz77::MAX_WINDOW,
            lazy_parsing: false,  // parsing handled by caller
            min_match: MIN_MATCH, // 4 (SortLZ hash minimum)
            max_candidates: MAX_CANDIDATES,
            max_match_len,
        }
    }
}

/// Find matches using sort-based approach.
///
/// Steps:
/// 1. Compute 4-byte hashes for all positions.
/// 2. Sort (hash, position) pairs by hash.
/// 3. For adjacent entries with equal hashes, verify and extend matches.
/// 4. For each position, select the best match.
pub fn find_matches(input: &[u8], config: &SortLzConfig) -> Vec<Option<(u16, u16)>> {
    let n = input.len();
    if n < 4 {
        return vec![None; n];
    }

    // Step 1: Hash every 4-byte window. Use the 4 bytes directly as a u32 hash
    // (no collisions for 4-byte matches).
    let num_hashes = n.saturating_sub(3);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(num_hashes);
    for i in 0..num_hashes {
        let hash = u32::from_le_bytes([input[i], input[i + 1], input[i + 2], input[i + 3]]);
        pairs.push((hash, i as u32));
    }

    // Step 2: Radix sort by hash value.
    radix_sort_pairs(&mut pairs);

    // Step 3: Adjacent-pair match verification.
    // For each position, collect the best match candidates.
    let mut best_match: Vec<Option<(u16, u16)>> = vec![None; n]; // (offset, length)

    for window_start in 0..pairs.len() {
        let (hash_a, pos_a_raw) = pairs[window_start];
        let pos_a = pos_a_raw as usize;

        // Skip if this position already has a good-enough match.
        if let Some((_, len)) = best_match[pos_a] {
            if len >= SATISFICE_LEN {
                continue;
            }
        }

        // Look at adjacent entries with the same hash.
        let mut candidates_checked = 0;
        for pair in pairs.iter().skip(window_start + 1) {
            if pair.0 != hash_a {
                break;
            }
            if candidates_checked >= config.max_candidates {
                break;
            }

            let pos_b = pair.1 as usize;

            // Ensure earlier position is source for valid back-reference.
            let (src, dst) = if pos_a < pos_b {
                (pos_a, pos_b)
            } else {
                (pos_b, pos_a)
            };

            let distance = dst - src;
            if distance > config.max_window || distance == 0 {
                candidates_checked += 1;
                continue;
            }

            // Skip if destination already has a good-enough match.
            if let Some((_, existing_len)) = best_match[dst] {
                if existing_len >= SATISFICE_LEN {
                    candidates_checked += 1;
                    continue;
                }
            }

            // Skip offsets that overflow u16 (lz77::Match.offset is u16).
            if distance > u16::MAX as usize {
                candidates_checked += 1;
                continue;
            }

            // Verify and extend match.
            let match_len = extend_match(input, src, dst, config.max_match_len);
            if match_len >= config.min_match {
                let offset = distance as u16;
                let length = match_len as u16; // already capped at max_match_len in extend_match

                // Update best match for the destination position.
                if let Some((_, existing_len)) = best_match[dst] {
                    if length > existing_len {
                        best_match[dst] = Some((offset, length));
                    }
                } else {
                    best_match[dst] = Some((offset, length));
                }
            }

            candidates_checked += 1;
        }
    }

    best_match
}

/// Find top-K match candidates per position using sort-based approach.
///
/// Like `find_matches()`, but returns up to `k` candidates per position
/// sorted by length descending. Adjacent entries in the sorted array
/// naturally provide multiple candidates when they share the same hash.
///
/// Returns a `MatchTable` compatible with `optimal::optimal_parse()`.
pub fn find_matches_topk(
    input: &[u8],
    config: &SortLzConfig,
    k: usize,
) -> crate::optimal::MatchTable {
    use crate::optimal::{MatchCandidate, MatchTable};

    let n = input.len();
    let mut table = MatchTable::new(n, k);

    if n < 4 {
        return table;
    }

    // Step 1: Hash every 4-byte window
    let num_hashes = n.saturating_sub(3);
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(num_hashes);
    for i in 0..num_hashes {
        let hash = u32::from_le_bytes([input[i], input[i + 1], input[i + 2], input[i + 3]]);
        pairs.push((hash, i as u32));
    }

    // Step 2: Radix sort by hash value
    radix_sort_pairs(&mut pairs);

    // Step 3: Adjacent-pair match verification with top-K collection
    for window_start in 0..pairs.len() {
        let (hash_a, _) = pairs[window_start];

        let mut candidates_checked = 0;
        for j in (window_start + 1)..pairs.len() {
            if pairs[j].0 != hash_a {
                break;
            }
            if candidates_checked >= config.max_candidates {
                break;
            }

            let pos_a = pairs[window_start].1 as usize;
            let pos_b = pairs[j].1 as usize;

            let (src, dst) = if pos_a < pos_b {
                (pos_a, pos_b)
            } else {
                (pos_b, pos_a)
            };

            let distance = dst - src;
            if distance > config.max_window || distance == 0 {
                candidates_checked += 1;
                continue;
            }

            let match_len = extend_match(input, src, dst, config.max_match_len);
            if match_len >= config.min_match {
                let offset = distance as u32;
                let length = match_len as u32; // already capped at max_match_len in extend_match

                // Insert into top-K slot for the destination position,
                // maintaining sorted-by-length-desc order.
                let slot = table.at_mut(dst);
                insert_topk_candidate(slot, MatchCandidate { offset, length });
            }

            candidates_checked += 1;
        }
    }

    table
}

/// Insert a candidate into a top-K slot, maintaining length-descending order.
/// If the slot is full and the new candidate is shorter than all existing ones,
/// it is discarded.
fn insert_topk_candidate(
    slot: &mut [crate::optimal::MatchCandidate],
    candidate: crate::optimal::MatchCandidate,
) {
    // Find insertion point (sorted by length descending)
    let k = slot.len();

    // Skip if duplicate offset (keep longer one)
    for existing in slot.iter() {
        if existing.length == 0 {
            break;
        }
        if existing.offset == candidate.offset {
            return; // same offset already present with >= length
        }
    }

    // Find position to insert (first slot with shorter length)
    let mut insert_pos = k;
    for (i, existing) in slot.iter().enumerate() {
        if existing.length < candidate.length {
            insert_pos = i;
            break;
        }
    }

    if insert_pos < k {
        // Shift shorter candidates down, dropping the last one
        for i in (insert_pos + 1..k).rev() {
            slot[i] = slot[i - 1];
        }
        slot[insert_pos] = candidate;
    }
}

/// Extend a match starting at positions src and dst, return total match length.
///
/// Uses u64 chunk comparisons for 4-8x speedup on long matches.
/// Capped at `u16::MAX` to avoid excessive scanning on highly repetitive data.
fn extend_match(input: &[u8], src: usize, dst: usize, max_match_len: u16) -> usize {
    let max_len = input.len() - dst;
    let max_len = max_len.min(input.len() - src).min(max_match_len as usize);

    // Fast path: compare 8 bytes at a time.
    let mut len = 0;
    let chunks = max_len / 8;
    let src_ptr = &input[src..];
    let dst_ptr = &input[dst..];
    for _ in 0..chunks {
        let a = u64::from_le_bytes(src_ptr[len..len + 8].try_into().unwrap());
        let b = u64::from_le_bytes(dst_ptr[len..len + 8].try_into().unwrap());
        if a != b {
            // Find first differing byte within the u64.
            let diff = a ^ b;
            len += (diff.trailing_zeros() / 8) as usize;
            return len;
        }
        len += 8;
    }

    // Tail: compare remaining bytes one at a time.
    while len < max_len && src_ptr[len] == dst_ptr[len] {
        len += 1;
    }
    len
}

/// Parse matches into tokens using greedy or lazy strategy.
pub(crate) fn parse_matches(
    input: &[u8],
    matches: &[Option<(u16, u16)>],
    lazy: bool,
) -> Vec<LzToken> {
    let n = input.len();
    let mut tokens = Vec::new();
    let mut pos = 0;

    while pos < n {
        if let Some((offset, length)) = matches[pos] {
            if lazy && pos + 1 < n {
                // Lazy: check if next position has a longer match.
                if let Some((_, next_len)) = matches[pos + 1] {
                    if next_len > length {
                        // Emit current position as literal, take next match.
                        tokens.push(LzToken::Literal(input[pos]));
                        pos += 1;
                        continue;
                    }
                }
            }

            tokens.push(LzToken::Match {
                offset: offset as u32,
                length: length as u32,
            });
            pos += length as usize;
        } else {
            tokens.push(LzToken::Literal(input[pos]));
            pos += 1;
        }
    }

    tokens
}

// ---------------------------------------------------------------------------
// SortLZ → LZ77 Match conversion (for feeding into LZ77 pipelines)
// ---------------------------------------------------------------------------

/// Convert position-indexed matches to an LZ77 match sequence using greedy parsing.
///
/// Takes the longest match at every position, emitting `lz77::Match` structs
/// with the `next` byte. Produces the same format as `lz77::compress_lazy_to_matches()`.
pub fn matches_to_lz77_greedy(
    input: &[u8],
    matches: &[Option<(u16, u16)>],
) -> Vec<crate::lz77::Match> {
    let n = input.len();
    let mut result = Vec::with_capacity(n / 4);
    let mut pos = 0;

    while pos < n {
        if let Some((offset, length)) = matches.get(pos).copied().flatten() {
            let end = pos + length as usize;
            if end < n {
                result.push(crate::lz77::Match {
                    offset,
                    length,
                    next: input[end],
                });
                pos = end + 1;
            } else {
                // Match extends to or past end of input; truncate to leave room for next byte
                let adj_len = (n - 1 - pos) as u16;
                if adj_len >= crate::lz77::MIN_MATCH {
                    result.push(crate::lz77::Match {
                        offset,
                        length: adj_len,
                        next: input[pos + adj_len as usize],
                    });
                    pos = pos + adj_len as usize + 1;
                } else {
                    result.push(crate::lz77::Match {
                        offset: 0,
                        length: 0,
                        next: input[pos],
                    });
                    pos += 1;
                }
            }
        } else {
            result.push(crate::lz77::Match {
                offset: 0,
                length: 0,
                next: input[pos],
            });
            pos += 1;
        }
    }

    result
}

/// Convert position-indexed matches to an LZ77 match sequence using lazy parsing.
///
/// If the next position has a longer match, emits a literal for the current
/// position and takes the longer match instead (gzip-style lazy evaluation).
pub fn matches_to_lz77_lazy(
    input: &[u8],
    matches: &[Option<(u16, u16)>],
) -> Vec<crate::lz77::Match> {
    let n = input.len();
    let mut result = Vec::with_capacity(n / 4);
    let mut pos = 0;

    while pos < n {
        if let Some((offset, length)) = matches.get(pos).copied().flatten() {
            // Lazy check: if next position has a longer match, emit literal here
            if pos + 1 < n {
                if let Some((_, next_len)) = matches.get(pos + 1).copied().flatten() {
                    if next_len > length {
                        result.push(crate::lz77::Match {
                            offset: 0,
                            length: 0,
                            next: input[pos],
                        });
                        pos += 1;
                        continue;
                    }
                }
            }

            let end = pos + length as usize;
            if end < n {
                result.push(crate::lz77::Match {
                    offset,
                    length,
                    next: input[end],
                });
                pos = end + 1;
            } else {
                let adj_len = (n - 1 - pos) as u16;
                if adj_len >= crate::lz77::MIN_MATCH {
                    result.push(crate::lz77::Match {
                        offset,
                        length: adj_len,
                        next: input[pos + adj_len as usize],
                    });
                    pos = pos + adj_len as usize + 1;
                } else {
                    result.push(crate::lz77::Match {
                        offset: 0,
                        length: 0,
                        next: input[pos],
                    });
                    pos += 1;
                }
            }
        } else {
            result.push(crate::lz77::Match {
                offset: 0,
                length: 0,
                next: input[pos],
            });
            pos += 1;
        }
    }

    result
}

/// Compress using the SortLZ pipeline.
///
/// Wire format (v2 — LzSeq-encoded streams + FSE):
/// ```text
/// [meta_len: u16 LE] [meta: meta_len bytes]
/// [num_streams: u8]
/// per stream: [orig_len: u32 LE] [fse_len: u32 LE] [fse_data]
/// ```
pub fn compress(input: &[u8], config: &SortLzConfig) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let matches = find_matches(input, config);
    compress_with_matches(input, matches, config)
}

/// Compress using pre-computed matches (shared by CPU and GPU paths).
pub fn compress_with_matches(
    input: &[u8],
    matches: Vec<Option<(u16, u16)>>,
    config: &SortLzConfig,
) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let tokens = parse_matches(input, &matches, config.lazy_parsing);
    let encoder = LzSeqEncoder {
        max_window: config.max_window,
    };
    let encoded = encoder.encode(input, &tokens)?;

    // Assemble output: [meta_len:u16] [meta] [num_streams:u8]
    // per stream: [orig_len:u32] [fse_len:u32] [fse_data]
    let mut output = Vec::new();
    output.extend_from_slice(&(encoded.meta.len() as u16).to_le_bytes());
    output.extend_from_slice(&encoded.meta);
    output.push(encoded.streams.len() as u8);

    for stream in &encoded.streams {
        let fse_data = fse::encode(stream);
        output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
        output.extend_from_slice(&fse_data);
    }

    Ok(output)
}

/// Decompress SortLZ data back to the original input.
///
/// Wire format (v2): [meta_len:u16] [meta] [num_streams:u8]
/// per stream: [orig_len:u32] [fse_len:u32] [fse_data]
pub fn decompress(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 3 {
        return Err(PzError::InvalidInput);
    }

    let meta_len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    let mut pos = 2;
    if pos + meta_len > payload.len() {
        return Err(PzError::InvalidInput);
    }
    let meta = payload[pos..pos + meta_len].to_vec();
    pos += meta_len;

    if pos >= payload.len() {
        return Err(PzError::InvalidInput);
    }
    let num_streams = payload[pos] as usize;
    pos += 1;

    let mut streams = Vec::with_capacity(num_streams);
    for _ in 0..num_streams {
        if pos + 8 > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let stream_orig_len = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]) as usize;
        let fse_len = u32::from_le_bytes([
            payload[pos + 4],
            payload[pos + 5],
            payload[pos + 6],
            payload[pos + 7],
        ]) as usize;
        pos += 8;

        if pos + fse_len > payload.len() {
            return Err(PzError::InvalidInput);
        }
        let decoded = if stream_orig_len == 0 {
            Vec::new()
        } else {
            fse::decode(&payload[pos..pos + fse_len], stream_orig_len)?
        };
        pos += fse_len;
        streams.push(decoded);
    }

    let encoder = LzSeqEncoder {
        max_window: DEFAULT_MAX_WINDOW,
    };
    encoder.decode(streams, &meta, orig_len)
}

/// Radix sort (hash, position) pairs by hash value.
/// Uses 4-pass 8-bit radix sort (LSB first) with double-buffering
/// to eliminate per-pass copies. Skips passes where all values share
/// the same byte (single-bucket optimization).
type PairSliceRefs<'a> = (&'a [(u32, u32)], &'a mut [(u32, u32)]);

fn radix_sort_pairs(pairs: &mut [(u32, u32)]) {
    if pairs.len() <= 1 {
        return;
    }

    let n = pairs.len();
    let mut buf = vec![(0u32, 0u32); n];

    // Double-buffer: alternate src/dst to avoid copies.
    // After 4 passes (even), result is back in `pairs`.
    let mut in_buf = false; // false = pairs is source, true = buf is source

    for byte_idx in 0..4u32 {
        let shift = byte_idx * 8;
        let (src, dst): PairSliceRefs = if in_buf {
            (buf.as_slice(), pairs)
        } else {
            (pairs, buf.as_mut_slice())
        };

        // Count.
        let mut counts = [0u32; 256];
        for &(hash, _) in src.iter() {
            let bucket = ((hash >> shift) & 0xFF) as usize;
            counts[bucket] += 1;
        }

        // Skip pass if all values are in one bucket (no reordering needed).
        let mut nonzero = 0u32;
        for &c in &counts {
            nonzero += u32::from(c > 0);
        }
        if nonzero <= 1 {
            // All same byte — no scatter needed, keep src as-is.
            continue;
        }

        // Prefix sum.
        let mut offsets = [0u32; 256];
        let mut sum = 0u32;
        for i in 0..256 {
            offsets[i] = sum;
            sum += counts[i];
        }

        // Scatter.
        for &pair in src.iter() {
            let bucket = ((pair.0 >> shift) & 0xFF) as usize;
            dst[offsets[bucket] as usize] = pair;
            offsets[bucket] += 1;
        }

        in_buf = !in_buf;
    }

    // If final result ended up in buf, copy back to pairs.
    if in_buf {
        pairs.copy_from_slice(&buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple() {
        let input = b"abcabcabcabc";
        let config = SortLzConfig::default();
        let compressed = compress(input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_repeated() {
        let input = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let config = SortLzConfig::default();
        let compressed = compress(input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_no_matches() {
        // Short input with no 4-byte repeats.
        let input = b"abcdefghijklmnop";
        let config = SortLzConfig::default();
        let compressed = compress(input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_roundtrip_longer() {
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(b"hello world! this is a test. ");
        }
        let config = SortLzConfig::default();
        let compressed = compress(&input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_greedy_parsing() {
        let input = b"abcabcabcabcabcabcabcabc";
        let config = SortLzConfig {
            lazy_parsing: false,
            ..SortLzConfig::default()
        };
        let compressed = compress(input, &config).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_radix_sort() {
        let mut pairs = vec![(5u32, 0), (1, 1), (3, 2), (2, 3), (4, 4)];
        radix_sort_pairs(&mut pairs);
        let hashes: Vec<u32> = pairs.iter().map(|p| p.0).collect();
        assert_eq!(hashes, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_determinism() {
        let input = b"the quick brown fox jumps over the lazy dog the quick brown fox";
        let config = SortLzConfig::default();
        let c1 = compress(input, &config).unwrap();
        let c2 = compress(input, &config).unwrap();
        assert_eq!(c1, c2, "SortLZ must be deterministic");
    }

    // -----------------------------------------------------------------------
    // SortLZ → LZ77 pipeline integration tests
    // -----------------------------------------------------------------------

    fn test_data() -> Vec<u8> {
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(b"hello world! this is a test of sortlz match finding. ");
        }
        input
    }

    #[test]
    fn test_lz77_greedy_roundtrip() {
        let input = test_data();
        let config = SortLzConfig::for_lz77(crate::lz77::LZ77_MAX_MATCH);
        let matches = find_matches(&input, &config);
        let lz_matches = matches_to_lz77_greedy(&input, &matches);

        // Verify matches reconstruct the input via LZ77 decompress
        let mut lz_bytes = Vec::new();
        for m in &lz_matches {
            lz_bytes.extend_from_slice(&m.to_bytes());
        }
        let decoded = crate::lz77::decompress(&lz_bytes).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_lz77_lazy_roundtrip() {
        let input = test_data();
        let config = SortLzConfig::for_lz77(crate::lz77::LZ77_MAX_MATCH);
        let matches = find_matches(&input, &config);
        let lz_matches = matches_to_lz77_lazy(&input, &matches);

        let mut lz_bytes = Vec::new();
        for m in &lz_matches {
            lz_bytes.extend_from_slice(&m.to_bytes());
        }
        let decoded = crate::lz77::decompress(&lz_bytes).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_topk_roundtrip() {
        let input = test_data();
        let config = SortLzConfig::for_lz77(crate::lz77::LZ77_MAX_MATCH);
        let table = find_matches_topk(&input, &config, 4);

        // Verify table has valid candidates
        let mut found_match = false;
        for pos in 0..input.len() {
            let candidates = table.at(pos);
            if candidates[0].length >= 4 {
                found_match = true;
                assert!(candidates[0].offset > 0);
                // Verify candidates are sorted by length descending
                for i in 1..candidates.len() {
                    if candidates[i].length == 0 {
                        break;
                    }
                    assert!(candidates[i].length <= candidates[i - 1].length);
                }
            }
        }
        assert!(
            found_match,
            "should find at least one match in repetitive data"
        );

        // Verify optimal parse produces valid LZ77
        let freq = crate::frequency::get_frequency(&input);
        let cost_model = crate::optimal::CostModel::from_frequencies(&freq);
        let lz_matches = crate::optimal::optimal_parse(&input, &table, &cost_model);

        let mut lz_bytes = Vec::new();
        for m in &lz_matches {
            lz_bytes.extend_from_slice(&m.to_bytes());
        }
        let decoded = crate::lz77::decompress(&lz_bytes).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_pipeline_roundtrip_lzf_sortlz() {
        use crate::pipeline::{self, CompressOptions, MatchFinder, ParseStrategy, Pipeline};
        let input = test_data();

        for strategy in [ParseStrategy::Greedy, ParseStrategy::Lazy] {
            let opts = CompressOptions {
                match_finder: MatchFinder::SortLz,
                parse_strategy: strategy,
                threads: 1,
                ..Default::default()
            };
            let compressed = pipeline::compress_with_options(&input, Pipeline::Lzf, &opts).unwrap();
            let decoded = pipeline::decompress(&compressed).unwrap();
            assert_eq!(
                decoded, input,
                "Lzf + SortLz + {:?} roundtrip failed",
                strategy
            );
        }
    }

    #[test]
    fn test_pipeline_roundtrip_lzseqr_sortlz() {
        use crate::pipeline::{self, CompressOptions, MatchFinder, Pipeline};
        let input = test_data();

        let opts = CompressOptions {
            match_finder: MatchFinder::SortLz,
            threads: 1,
            ..Default::default()
        };
        let compressed = pipeline::compress_with_options(&input, Pipeline::LzSeqR, &opts).unwrap();
        let decoded = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decoded, input, "LzSeqR + SortLz roundtrip failed");
    }

    #[test]
    fn test_pipeline_roundtrip_lzseqh_sortlz() {
        use crate::pipeline::{self, CompressOptions, MatchFinder, Pipeline};
        let input = test_data();

        let opts = CompressOptions {
            match_finder: MatchFinder::SortLz,
            threads: 1,
            ..Default::default()
        };
        let compressed = pipeline::compress_with_options(&input, Pipeline::LzSeqH, &opts).unwrap();
        let decoded = pipeline::decompress(&compressed).unwrap();
        assert_eq!(decoded, input, "LzSeqH + SortLz roundtrip failed");
    }
}
