/// LZ77 compression and decompression.
///
/// This implements a correct LZ77 compressor/decompressor, fixing bugs
/// BUG-02, BUG-03, and BUG-06 from the C reference implementation:
/// - BUG-02: Spot-check reads past search buffer (added bounds check)
/// - BUG-03: Inner loop bounded by wrong size variable (correct bounds)
/// - BUG-06: Decompressor buffer overflow on literal write (check length+1)
///
/// Three match-finding strategies are available:
/// - **Brute-force** (`compress`): O(n*w) worst case, simple and correct.
/// - **Hash-chain** (`compress_hashchain`): O(n) average, uses a hash table
///   to quickly find candidate positions. Much faster on large inputs.
/// - **Lazy matching** (`compress_lazy`): Uses hash-chain finder with
///   gzip-style lazy evaluation: if the next position has a longer match,
///   emit a literal and use the longer match instead. Better compression ratio.
use crate::{PzError, PzResult};

/// Maximum sliding window size for match finding.
const MAX_WINDOW: usize = 4096;

/// Minimum match length to consider (shorter matches aren't worth encoding).
const MIN_MATCH: u32 = 3;

/// Hash table size for hash-chain match finder (power of 2).
const HASH_SIZE: usize = 1 << 15; // 32768
const HASH_MASK: usize = HASH_SIZE - 1;

/// Maximum number of chain links to follow per position.
const MAX_CHAIN: usize = 64;

/// An LZ77 match: a (offset, length, next) triple.
///
/// - `offset`: distance back from current position to match start (0 = no match)
/// - `length`: number of matching bytes
/// - `next`: the literal byte following the match
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Match {
    pub offset: u32,
    pub length: u32,
    pub next: u8,
}

impl Match {
    /// Size of a serialized match in bytes.
    pub const SERIALIZED_SIZE: usize = 9; // 4 + 4 + 1

    /// Serialize this match to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[0..4].copy_from_slice(&self.offset.to_le_bytes());
        buf[4..8].copy_from_slice(&self.length.to_le_bytes());
        buf[8] = self.next;
        buf
    }

    /// Deserialize a match from bytes (little-endian).
    pub fn from_bytes(buf: &[u8; Self::SERIALIZED_SIZE]) -> Self {
        Match {
            offset: u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]),
            length: u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]),
            next: buf[8],
        }
    }
}

/// Find the best match for `target` in the `search` window.
///
/// This is the corrected version of FindMatchClassic:
/// - BUG-02 fix: bounds check before spot-check access
/// - BUG-03 fix: correct loop bounds for both search and target buffers
fn find_best_match(search: &[u8], target: &[u8]) -> Match {
    if target.is_empty() {
        return Match {
            offset: 0,
            length: 0,
            next: 0,
        };
    }

    let search_size = search.len();
    let target_size = target.len();
    let mut best_offset: u32 = 0;
    let mut best_length: u32 = 0;

    for i in 0..search_size {
        // BUG-02 fix: bounds check before spot-check
        if best_length > 0 && (best_length as usize) < target_size {
            let spot = i + best_length as usize;
            if spot >= search_size {
                continue;
            }
            if search[spot] != target[best_length as usize] {
                continue;
            }
        }

        let mut match_len: u32 = 0;
        let mut si = i;

        // BUG-03 fix: properly bound both indices
        // si must stay within search buffer, match_len must stay within target
        while si < search_size && (match_len as usize) < target_size {
            if search[si] != target[match_len as usize] {
                break;
            }
            match_len += 1;
            si += 1;
        }

        if match_len > best_length {
            best_offset = (search_size - i) as u32;
            best_length = match_len;
        }
    }

    // Truncate match so that `next` points to a valid byte in target.
    // If the match covers the entire target, reduce it so there's room
    // for the literal `next` byte.
    while best_length as usize >= target_size && best_length > 0 {
        best_length -= 1;
    }

    let next = if (best_length as usize) < target_size {
        target[best_length as usize]
    } else {
        0
    };

    Match {
        offset: best_offset,
        length: best_length,
        next,
    }
}

/// Compress input data using LZ77.
///
/// Returns the compressed data as a vector of serialized matches.
pub fn compress(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let mut pos: usize = 0;

    while pos < input.len() {
        // Determine the search window
        let window_start = pos.saturating_sub(MAX_WINDOW);
        let search = &input[window_start..pos];
        let target = &input[pos..];

        let m = find_best_match(search, target);
        output.extend_from_slice(&m.to_bytes());
        pos += m.length as usize + 1;
    }

    Ok(output)
}

/// Compress input data into a pre-allocated output buffer.
///
/// Returns the number of bytes written to `output`.
pub fn compress_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    let mut pos: usize = 0;
    let mut out_pos: usize = 0;

    while pos < input.len() {
        let window_start = pos.saturating_sub(MAX_WINDOW);
        let search = &input[window_start..pos];
        let target = &input[pos..];

        let m = find_best_match(search, target);

        if out_pos + Match::SERIALIZED_SIZE > output.len() {
            return Err(PzError::BufferTooSmall);
        }

        let bytes = m.to_bytes();
        output[out_pos..out_pos + Match::SERIALIZED_SIZE].copy_from_slice(&bytes);
        out_pos += Match::SERIALIZED_SIZE;
        pos += m.length as usize + 1;
    }

    Ok(out_pos)
}

/// Decompress LZ77-compressed data.
///
/// The input must be a sequence of serialized Match structs.
///
/// BUG-06 fix: checks buffer space for both the match copy AND the
/// literal `next` byte (length + 1).
pub fn decompress(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if !input.len().is_multiple_of(Match::SERIALIZED_SIZE) {
        return Err(PzError::InvalidInput);
    }

    let num_matches = input.len() / Match::SERIALIZED_SIZE;
    let mut output = Vec::new();

    for i in 0..num_matches {
        let start = i * Match::SERIALIZED_SIZE;
        let buf: [u8; Match::SERIALIZED_SIZE] =
            input[start..start + Match::SERIALIZED_SIZE]
                .try_into()
                .unwrap();
        let m = Match::from_bytes(&buf);

        // Copy back-referenced bytes
        if m.length > 0 {
            if m.offset as usize > output.len() {
                return Err(PzError::InvalidInput);
            }
            let copy_start = output.len() - m.offset as usize;
            for j in 0..m.length as usize {
                let byte = output[copy_start + j];
                output.push(byte);
            }
        }

        // Append the literal byte
        output.push(m.next);
    }

    Ok(output)
}

/// Decompress LZ77-compressed data into a pre-allocated output buffer.
///
/// Returns the number of bytes written to `output`.
///
/// BUG-06 fix: checks buffer space for both the match copy AND the
/// literal `next` byte (length + 1).
pub fn decompress_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    if input.is_empty() {
        return Ok(0);
    }

    if !input.len().is_multiple_of(Match::SERIALIZED_SIZE) {
        return Err(PzError::InvalidInput);
    }

    let num_matches = input.len() / Match::SERIALIZED_SIZE;
    let mut out_pos: usize = 0;

    for i in 0..num_matches {
        let start = i * Match::SERIALIZED_SIZE;
        let buf: [u8; Match::SERIALIZED_SIZE] =
            input[start..start + Match::SERIALIZED_SIZE]
                .try_into()
                .unwrap();
        let m = Match::from_bytes(&buf);

        // BUG-06 fix: check for length + 1 (match bytes + literal byte)
        if out_pos + m.length as usize + 1 > output.len() {
            return Err(PzError::BufferTooSmall);
        }

        // Copy back-referenced bytes
        if m.length > 0 {
            if m.offset as usize > out_pos {
                return Err(PzError::InvalidInput);
            }
            let copy_start = out_pos - m.offset as usize;
            for j in 0..m.length as usize {
                output[out_pos] = output[copy_start + j];
                out_pos += 1;
            }
        }

        // Append the literal byte
        output[out_pos] = m.next;
        out_pos += 1;
    }

    Ok(out_pos)
}

// --- Hash-chain match finder ---

/// Compute a hash for 3 bytes at the given position.
fn hash3(data: &[u8], pos: usize) -> usize {
    if pos + 2 >= data.len() {
        return 0;
    }
    let h = (data[pos] as usize) << 10
        ^ (data[pos + 1] as usize) << 5
        ^ (data[pos + 2] as usize);
    h & HASH_MASK
}

/// Hash-chain based match finder.
///
/// Maintains a hash table mapping 3-byte prefixes to positions,
/// with chains for collision resolution. Average O(n) complexity.
struct HashChainFinder {
    /// head[hash] = most recent position with this hash, or 0
    head: Vec<u32>,
    /// prev[pos % MAX_WINDOW] = previous position in the chain
    prev: Vec<u32>,
}

impl HashChainFinder {
    fn new() -> Self {
        HashChainFinder {
            head: vec![0; HASH_SIZE],
            prev: vec![0; MAX_WINDOW],
        }
    }

    /// Find the best match at `pos` in `input`, looking back up to MAX_WINDOW bytes.
    fn find_match(&self, input: &[u8], pos: usize) -> Match {
        let remaining = input.len() - pos;
        if remaining < 3 {
            // Not enough bytes for a match prefix
            return Match {
                offset: 0,
                length: 0,
                next: if pos < input.len() { input[pos] } else { 0 },
            };
        }

        let h = hash3(input, pos);
        let mut chain_pos = self.head[h] as usize;
        let mut best_offset: u32 = 0;
        let mut best_length: u32 = 0;
        let min_pos = pos.saturating_sub(MAX_WINDOW);
        let mut chain_count = 0;

        while chain_pos >= min_pos && chain_pos < pos && chain_count < MAX_CHAIN {
            // Compare bytes
            let mut match_len = 0u32;
            let max_len = remaining.min(pos - chain_pos) as u32;
            while match_len < max_len
                && input[chain_pos + match_len as usize] == input[pos + match_len as usize]
            {
                match_len += 1;
            }

            if match_len > best_length && match_len >= MIN_MATCH {
                best_length = match_len;
                best_offset = (pos - chain_pos) as u32;
            }

            // Follow chain
            let prev_pos = self.prev[chain_pos % MAX_WINDOW] as usize;
            if prev_pos >= chain_pos || prev_pos < min_pos {
                break;
            }
            chain_pos = prev_pos;
            chain_count += 1;
        }

        // Ensure room for the literal `next` byte
        while best_length as usize >= remaining && best_length > 0 {
            best_length -= 1;
        }

        let next = if (best_length as usize) < remaining {
            input[pos + best_length as usize]
        } else {
            0
        };

        Match {
            offset: best_offset,
            length: best_length,
            next,
        }
    }

    /// Insert position `pos` into the hash chain.
    fn insert(&mut self, input: &[u8], pos: usize) {
        if pos + 2 >= input.len() {
            return;
        }
        let h = hash3(input, pos);
        self.prev[pos % MAX_WINDOW] = self.head[h];
        self.head[h] = pos as u32;
    }
}

/// Compress input using hash-chain match finder.
///
/// O(n) average time complexity vs O(n*w) for brute-force.
/// Produces the same output format (serialized Match structs).
pub fn compress_hashchain(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let mut finder = HashChainFinder::new();
    let mut pos: usize = 0;

    while pos < input.len() {
        let m = finder.find_match(input, pos);

        // Insert all positions we're about to skip
        let advance = m.length as usize + 1;
        for i in 0..advance.min(input.len() - pos) {
            finder.insert(input, pos + i);
        }

        output.extend_from_slice(&m.to_bytes());
        pos += advance;
    }

    Ok(output)
}

/// Compress input using lazy matching (gzip-style).
///
/// After finding a match at position P, checks if position P+1 has a
/// longer match. If so, emits a literal for P and uses the longer match.
/// This produces better compression ratios than greedy matching.
pub fn compress_lazy(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let mut finder = HashChainFinder::new();
    let mut pos: usize = 0;

    while pos < input.len() {
        let m = finder.find_match(input, pos);
        finder.insert(input, pos);

        // If we found a match, check if the next position has a longer one
        if m.length >= MIN_MATCH && pos + 1 < input.len() {
            finder.insert(input, pos + 1);
            let next_m = finder.find_match(input, pos + 1);

            if next_m.length > m.length {
                // Emit current position as a literal, use the next match
                let literal = Match {
                    offset: 0,
                    length: 0,
                    next: input[pos],
                };
                output.extend_from_slice(&literal.to_bytes());
                pos += 1;

                // Insert positions covered by next_m
                let advance = next_m.length as usize + 1;
                for i in 1..advance.min(input.len() - pos) {
                    finder.insert(input, pos + i);
                }

                output.extend_from_slice(&next_m.to_bytes());
                pos += advance;
                continue;
            }
        }

        // Use the original match
        let advance = m.length as usize + 1;
        for i in 1..advance.min(input.len() - pos) {
            finder.insert(input, pos + i);
        }

        output.extend_from_slice(&m.to_bytes());
        pos += advance;
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_serialization() {
        let m = Match {
            offset: 42,
            length: 10,
            next: b'x',
        };
        let bytes = m.to_bytes();
        let m2 = Match::from_bytes(&bytes);
        assert_eq!(m, m2);
    }

    #[test]
    fn test_compress_empty() {
        let result = compress(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_decompress_empty() {
        let result = decompress(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_round_trip_single_byte() {
        let input = b"a";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_round_trip_no_matches() {
        let input = b"abcdefgh";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_round_trip_with_repeats() {
        let input = b"abcabcabc";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_round_trip_all_same() {
        let input = vec![b'x'; 100];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_banana() {
        let input = b"banana";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_round_trip_longer_text() {
        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, &input[..]);
    }

    #[test]
    fn test_round_trip_binary_data() {
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_round_trip_large_repeating() {
        // Large input with lots of repetition to exercise window limits
        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_compress_to_buf() {
        let input = b"abcabc";
        let mut output = vec![0u8; 1024];
        let size = compress_to_buf(input, &mut output).unwrap();
        let decompressed = decompress(&output[..size]).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_compress_to_buf_too_small() {
        let input = b"abcdefghijklmnop";
        let mut output = vec![0u8; 1]; // way too small
        let result = compress_to_buf(input, &mut output);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_decompress_to_buf() {
        let input = b"abcabc";
        let compressed = compress(input).unwrap();
        let mut output = vec![0u8; 1024];
        let size = decompress_to_buf(&compressed, &mut output).unwrap();
        assert_eq!(&output[..size], input);
    }

    #[test]
    fn test_decompress_to_buf_too_small() {
        let input = b"abcabcabc";
        let compressed = compress(input).unwrap();
        let mut output = vec![0u8; 1]; // way too small
        let result = decompress_to_buf(&compressed, &mut output);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_decompress_invalid_input_size() {
        // Input not a multiple of SERIALIZED_SIZE
        let result = decompress(&[1, 2, 3]);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decompress_invalid_offset() {
        // Match with offset larger than output so far
        let m = Match {
            offset: 100,
            length: 5,
            next: b'a',
        };
        let result = decompress(&m.to_bytes());
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_find_match_no_window() {
        // When there's no search window (start of input), should find no match
        let target = b"abc";
        let m = find_best_match(&[], target);
        assert_eq!(m.offset, 0);
        assert_eq!(m.length, 0);
        assert_eq!(m.next, b'a');
    }

    #[test]
    fn test_find_match_exact() {
        let search = b"abc";
        let target = b"abcd";
        let m = find_best_match(search, target);
        assert_eq!(m.length, 3);
        assert_eq!(m.next, b'd');
    }

    #[test]
    fn test_round_trip_window_boundary() {
        // Input long enough to exercise the MAX_WINDOW boundary
        let mut input = Vec::new();
        let block = b"ABCDEFGHIJ"; // 10 bytes
        for _ in 0..500 {
            // 5000 bytes > MAX_WINDOW (4096)
            input.extend_from_slice(block);
        }
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- Hash-chain match finder tests ---

    #[test]
    fn test_hashchain_round_trip_empty() {
        let result = compress_hashchain(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_hashchain_round_trip_single() {
        let input = b"a";
        let compressed = compress_hashchain(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_hashchain_round_trip_no_matches() {
        let input = b"abcdefgh";
        let compressed = compress_hashchain(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_hashchain_round_trip_repeats() {
        let input = b"abcabcabc";
        let compressed = compress_hashchain(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_hashchain_round_trip_all_same() {
        let input = vec![b'x'; 200];
        let compressed = compress_hashchain(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_hashchain_round_trip_longer_text() {
        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
        let compressed = compress_hashchain(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, &input[..]);
    }

    #[test]
    fn test_hashchain_round_trip_large() {
        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_hashchain(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_hashchain_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let compressed = compress_hashchain(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- Lazy matching tests ---

    #[test]
    fn test_lazy_round_trip_empty() {
        let result = compress_lazy(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_lazy_round_trip_single() {
        let input = b"a";
        let compressed = compress_lazy(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_lazy_round_trip_no_matches() {
        let input = b"abcdefgh";
        let compressed = compress_lazy(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_lazy_round_trip_repeats() {
        let input = b"abcabcabc";
        let compressed = compress_lazy(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn test_lazy_round_trip_all_same() {
        let input = vec![b'x'; 200];
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lazy_round_trip_longer_text() {
        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
        let compressed = compress_lazy(input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(&decompressed, &input[..]);
    }

    #[test]
    fn test_lazy_round_trip_large() {
        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lazy_correctness_vs_greedy() {
        // Both greedy and lazy should produce correct (decompressible) output.
        // Note: In our fixed-size match format, lazy may produce more output
        // bytes (each literal emits a full 9-byte Match struct). The benefit
        // of lazy matching appears when combined with entropy coding (Huffman
        // or arithmetic), where a literal + longer match can be cheaper.
        let pattern = b"abcdefg abcxyz abcdefg abcxyz ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        let greedy = compress_hashchain(&input).unwrap();
        let lazy = compress_lazy(&input).unwrap();
        // Both must decompress correctly
        assert_eq!(decompress(&greedy).unwrap(), input);
        assert_eq!(decompress(&lazy).unwrap(), input);
    }

    #[test]
    fn test_lazy_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }
}
