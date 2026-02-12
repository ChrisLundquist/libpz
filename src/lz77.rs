/// LZ77 compression and decompression.
///
/// This implements a correct LZ77 compressor/decompressor, fixing bugs
/// BUG-02, BUG-03, and BUG-06 from the C reference implementation:
/// - BUG-02: Spot-check reads past search buffer (added bounds check)
/// - BUG-03: Inner loop bounded by wrong size variable (correct bounds)
/// - BUG-06: Decompressor buffer overflow on literal write (check length+1)
///
/// Match-finding strategy:
/// - **Lazy matching** (`compress_lazy`): Uses hash-chain finder with
///   gzip-style lazy evaluation: if the next position has a longer match,
///   emit a literal and use the longer match instead. Best speed + ratio.
use crate::{PzError, PzResult};

/// Maximum sliding window size for match finding.
/// 32KB matches the gzip standard and provides much better compression
/// than the previous 4KB window, especially on repetitive data.
pub(crate) const MAX_WINDOW: usize = 32768;

/// Minimum match length to consider (shorter matches aren't worth encoding).
pub(crate) const MIN_MATCH: u16 = 3;

/// Hash table size for hash-chain match finder (power of 2).
pub(crate) const HASH_SIZE: usize = 1 << 15; // 32768
pub(crate) const HASH_MASK: usize = HASH_SIZE - 1;

/// Maximum number of chain links to follow per position.
pub(crate) const MAX_CHAIN: usize = 64;

/// An LZ77 match: a (offset, length, next) triple.
///
/// - `offset`: distance back from current position to match start (0 = no match)
/// - `length`: number of matching bytes
/// - `next`: the literal byte following the match
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Match {
    pub offset: u16,
    pub length: u16,
    pub next: u8,
}

impl Match {
    /// Size of a serialized match in bytes.
    pub const SERIALIZED_SIZE: usize = 5; // 2 + 2 + 1

    /// Serialize this match to bytes (little-endian).
    pub fn to_bytes(&self) -> [u8; Self::SERIALIZED_SIZE] {
        let mut buf = [0u8; Self::SERIALIZED_SIZE];
        buf[0..2].copy_from_slice(&self.offset.to_le_bytes());
        buf[2..4].copy_from_slice(&self.length.to_le_bytes());
        buf[4] = self.next;
        buf
    }

    /// Deserialize a match from bytes (little-endian).
    pub fn from_bytes(buf: &[u8; Self::SERIALIZED_SIZE]) -> Self {
        Self {
            offset: u16::from_le_bytes([buf[0], buf[1]]),
            length: u16::from_le_bytes([buf[2], buf[3]]),
            next: buf[4],
        }
    }
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

    let mut output = Vec::new();

    for chunk in input.chunks_exact(Match::SERIALIZED_SIZE) {
        let buf: &[u8; Match::SERIALIZED_SIZE] = chunk.try_into().unwrap();
        let m = Match::from_bytes(buf);

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

    let mut out_pos: usize = 0;

    for chunk in input.chunks_exact(Match::SERIALIZED_SIZE) {
        let buf: &[u8; Match::SERIALIZED_SIZE] = chunk.try_into().unwrap();
        let m = Match::from_bytes(buf);

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
#[inline(always)]
pub(crate) fn hash3(data: &[u8], pos: usize) -> usize {
    if pos + 2 >= data.len() {
        return 0;
    }
    let h = (data[pos] as usize) << 10 ^ (data[pos + 1] as usize) << 5 ^ (data[pos + 2] as usize);
    h & HASH_MASK
}

/// Unchecked hash for the hot path where caller guarantees `pos + 2 < data.len()`.
///
/// # Safety
/// Caller must ensure `pos + 2 < data.len()`.
#[inline(always)]
unsafe fn hash3_unchecked(data: &[u8], pos: usize) -> usize {
    let h = (*data.get_unchecked(pos) as usize) << 10
        ^ (*data.get_unchecked(pos + 1) as usize) << 5
        ^ (*data.get_unchecked(pos + 2) as usize);
    h & HASH_MASK
}

/// Maximum hash insertion count per match (longer matches cap insertion
/// to avoid spending time on positions that will soon leave the window).
const MAX_INSERT_LEN: usize = 128;

/// Match length threshold above which lazy evaluation is skipped.
/// A match this long is unlikely to be beaten by the next position.
const LAZY_SKIP_THRESHOLD: u16 = 32;

/// Hash-chain based match finder.
///
/// Maintains a hash table mapping 3-byte prefixes to positions,
/// with chains for collision resolution. Average O(n) complexity.
pub(crate) struct HashChainFinder {
    /// head[hash] = most recent position with this hash, or 0
    head: Vec<u32>,
    /// prev[pos % MAX_WINDOW] = previous position in the chain
    prev: Vec<u32>,
    /// Cached SIMD dispatcher — resolved once, avoids per-call feature detection.
    dispatcher: crate::simd::Dispatcher,
}

impl HashChainFinder {
    pub(crate) fn new() -> Self {
        Self {
            head: vec![0; HASH_SIZE],
            prev: vec![0; MAX_WINDOW],
            dispatcher: crate::simd::Dispatcher::new(),
        }
    }

    /// Find the best match at `pos` in `input`, looking back up to MAX_WINDOW bytes.
    ///
    /// Uses SIMD-accelerated byte comparison (SSE2: 16 bytes/cycle,
    /// AVX2: 32 bytes/cycle) for the inner match extension loop.
    pub(crate) fn find_match(&self, input: &[u8], pos: usize) -> Match {
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
            // SIMD-accelerated byte comparison.
            // max_len is capped only by remaining bytes (not by offset distance),
            // allowing overlapping matches where length > offset. This enables
            // efficient encoding of repeated-byte runs (e.g., offset=1, length=999
            // for 1000 identical bytes). The decompressor's byte-by-byte copy loop
            // already handles the overlap correctly.
            let max_len = remaining;
            let match_len =
                self.dispatcher
                    .compare_bytes(&input[chain_pos..], &input[pos..]) as u32;
            let match_len = match_len.min(max_len as u32);

            if match_len > best_length && match_len >= MIN_MATCH as u32 {
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

        // Cap to u16::MAX since Match.length is u16
        if best_length > u16::MAX as u32 {
            best_length = u16::MAX as u32;
        }

        let next = if (best_length as usize) < remaining {
            input[pos + best_length as usize]
        } else {
            0
        };

        Match {
            offset: best_offset as u16,
            length: best_length as u16,
            next,
        }
    }

    /// Insert position `pos` into the hash chain.
    #[inline(always)]
    pub(crate) fn insert(&mut self, input: &[u8], pos: usize) {
        if pos + 2 >= input.len() {
            return;
        }
        // SAFETY: bounds check above guarantees pos + 2 < input.len()
        let h = unsafe { hash3_unchecked(input, pos) };
        self.prev[pos % MAX_WINDOW] = self.head[h];
        self.head[h] = pos as u32;
    }

    /// Find top-K match candidates at `pos`.
    ///
    /// For each distinct match length found (>= MIN_MATCH), keeps the
    /// candidate with the smallest offset. Returns up to `k` candidates
    /// sorted by length descending.
    ///
    /// Uses SIMD-accelerated byte comparison for match extension.
    pub(crate) fn find_top_k(&self, input: &[u8], pos: usize, k: usize) -> Vec<(u16, u16)> {
        let remaining = input.len() - pos;
        if remaining < MIN_MATCH as usize {
            return Vec::new();
        }

        let h = hash3(input, pos);
        let mut chain_pos = self.head[h] as usize;
        let min_pos = pos.saturating_sub(MAX_WINDOW);
        let mut chain_count = 0;

        // For each distinct length, keep the match with the smallest offset.
        // Use a simple vec of (length, offset) pairs; K is small.
        let mut found: Vec<(u16, u16)> = Vec::new();

        while chain_pos >= min_pos && chain_pos < pos && chain_count < MAX_CHAIN {
            // Allow overlapping matches (length > offset) for run compression.
            let max_len = remaining;
            let match_len =
                self.dispatcher
                    .compare_bytes(&input[chain_pos..], &input[pos..]) as u32;
            let match_len = match_len.min(max_len as u32);

            if match_len >= MIN_MATCH as u32 {
                let offset = (pos - chain_pos) as u16;
                let length = match_len.min(u16::MAX as u32) as u16;

                // Check if we already have a candidate with this length
                let existing = found.iter_mut().find(|(l, _)| *l == length);
                match existing {
                    Some((_, ref mut o)) => {
                        // Keep the smaller offset for this length
                        if offset < *o {
                            *o = offset;
                        }
                    }
                    None => {
                        found.push((length, offset));
                    }
                }
            }

            // Follow chain
            let prev_pos = self.prev[chain_pos % MAX_WINDOW] as usize;
            if prev_pos >= chain_pos || prev_pos < min_pos {
                break;
            }
            chain_pos = prev_pos;
            chain_count += 1;
        }

        // Sort by length descending, take top K
        found.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        found.truncate(k);
        found
    }
}

/// Compress input using lazy matching, returning the match sequence.
///
/// After finding a match at position P, checks if position P+1 has a
/// longer match. If so, emits a literal for P and uses the longer match.
/// This produces the best compression ratios of the greedy strategies,
/// and is also faster than greedy hash-chain due to skipping matched
/// positions during hash insertion.
pub fn compress_lazy_to_matches(input: &[u8]) -> PzResult<Vec<Match>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut matches = Vec::with_capacity(input.len() / 4);
    let mut finder = HashChainFinder::new();
    let mut pos: usize = 0;

    while pos < input.len() {
        let m = finder.find_match(input, pos);
        finder.insert(input, pos);

        // If we found a match, check if the next position has a longer one
        // (skip lazy check for long matches — unlikely to be beaten)
        if m.length >= MIN_MATCH && m.length < LAZY_SKIP_THRESHOLD && pos + 1 < input.len() {
            finder.insert(input, pos + 1);
            let next_m = finder.find_match(input, pos + 1);

            if next_m.length > m.length {
                // Emit current position as a literal, use the next match
                matches.push(Match {
                    offset: 0,
                    length: 0,
                    next: input[pos],
                });
                pos += 1;

                // Insert positions covered by next_m (capped for long matches)
                let advance = next_m.length as usize + 1;
                let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
                for i in 1..insert_count {
                    finder.insert(input, pos + i);
                }

                matches.push(next_m);
                pos += advance;
                continue;
            }
        }

        // Use the original match
        let advance = m.length as usize + 1;
        let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
        for i in 1..insert_count {
            finder.insert(input, pos + i);
        }

        matches.push(m);
        pos += advance;
    }

    Ok(matches)
}

/// Compress input using lazy matching (gzip-style), returning serialized bytes.
///
/// This is the standard entry point for LZ77 compression. Uses
/// `compress_lazy_to_matches` internally and serializes the result.
pub fn compress_lazy(input: &[u8]) -> PzResult<Vec<u8>> {
    let matches = compress_lazy_to_matches(input)?;
    let mut output = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
    for m in &matches {
        output.extend_from_slice(&m.to_bytes());
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
    fn test_decompress_empty() {
        let result = decompress(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_decompress_to_buf() {
        let input = b"abcabc";
        let compressed = compress_lazy(input).unwrap();
        let mut output = vec![0u8; 1024];
        let size = decompress_to_buf(&compressed, &mut output).unwrap();
        assert_eq!(&output[..size], input);
    }

    #[test]
    fn test_decompress_to_buf_too_small() {
        let input = b"abcabcabc";
        let compressed = compress_lazy(input).unwrap();
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
    fn test_lazy_round_trip_binary() {
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lazy_round_trip_window_boundary() {
        // Input long enough to exercise the MAX_WINDOW boundary
        let mut input = Vec::new();
        let block = b"ABCDEFGHIJ"; // 10 bytes
        for _ in 0..500 {
            // 5000 bytes > MAX_WINDOW
            input.extend_from_slice(block);
        }
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // --- Parse quality regression tests ---
    //
    // These tests assert that the lazy parser produces at most N sequences
    // (Match structs) for known inputs. Fewer sequences = better compression.
    // If a change reduces the count, update the golden value downward.
    // If a change increases the count, the test fails — investigate the regression.

    /// Count total sequences and match sequences in serialized LZ77 output.
    fn count_sequences(lz_data: &[u8]) -> (usize, usize) {
        let total = lz_data.len() / Match::SERIALIZED_SIZE;
        let mut matches = 0;
        for chunk in lz_data.chunks_exact(Match::SERIALIZED_SIZE) {
            let buf: &[u8; Match::SERIALIZED_SIZE] = chunk.try_into().unwrap();
            let m = Match::from_bytes(buf);
            if m.length > 0 && m.offset > 0 {
                matches += 1;
            }
        }
        (total, matches)
    }

    /// Sum of all match lengths (bytes covered by back-references).
    fn total_match_bytes(lz_data: &[u8]) -> usize {
        let mut total = 0;
        for chunk in lz_data.chunks_exact(Match::SERIALIZED_SIZE) {
            let buf: &[u8; Match::SERIALIZED_SIZE] = chunk.try_into().unwrap();
            let m = Match::from_bytes(buf);
            if m.length > 0 {
                total += m.length as usize;
            }
        }
        total
    }

    // -- Synthetic golden tests (always run) --
    //
    // These use exact equality so ANY change to the parser — improvement or
    // regression — is immediately flagged. When you intentionally improve the
    // parser, update the golden values to the new numbers.

    #[test]
    fn test_lazy_quality_repeated_pattern() {
        // 200 repeats of a 38-byte pattern = 7600 bytes.
        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = compress_lazy(&input).unwrap();
        let (total_seqs, match_seqs) = count_sequences(&compressed);
        let matched = total_match_bytes(&compressed);

        // Golden: 65 seqs, 31 matches, 7535 bytes matched.
        assert_eq!(total_seqs, 65, "total sequences changed (was 65)");
        assert_eq!(match_seqs, 31, "match count changed (was 31)");
        assert_eq!(matched, 7535, "match coverage changed (was 7535)");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lazy_quality_all_same() {
        // 10000 identical bytes. Overlapping matches (length > offset)
        // allow very long matches at offset=1, drastically reducing seqs.
        let input = vec![b'A'; 10000];
        let compressed = compress_lazy(&input).unwrap();
        let (total_seqs, match_seqs) = count_sequences(&compressed);
        let matched = total_match_bytes(&compressed);

        // Golden: 40 seqs, 39 matches, 9960 bytes matched.
        assert_eq!(total_seqs, 40, "total sequences changed (was 40)");
        assert_eq!(match_seqs, 39, "match count changed (was 39)");
        assert_eq!(matched, 9960, "match coverage changed (was 9960)");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_lazy_quality_cyclic_256() {
        // 0,1,2,...,255 repeated 4 times = 1024 bytes.
        let input: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
        let compressed = compress_lazy(&input).unwrap();
        let (total_seqs, match_seqs) = count_sequences(&compressed);
        let matched = total_match_bytes(&compressed);

        // Golden: 259 seqs, 3 matches, 765 bytes matched.
        assert_eq!(total_seqs, 259, "total sequences changed (was 259)");
        assert_eq!(match_seqs, 3, "match count changed (was 3)");
        assert_eq!(matched, 765, "match coverage changed (was 765)");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    // -- Canterbury corpus golden tests (skip if samples not extracted) --

    #[cfg(test)]
    fn corpus_file(name: &str) -> Option<Vec<u8>> {
        for dir in &["samples/cantrbry", "/home/user/libpz/samples/cantrbry"] {
            let path = format!("{}/{}", dir, name);
            if let Ok(data) = std::fs::read(&path) {
                return Some(data);
            }
        }
        None
    }

    #[test]
    fn test_lazy_quality_alice29() {
        let Some(data) = corpus_file("alice29.txt") else {
            eprintln!("skipping: alice29.txt not found");
            return;
        };
        assert_eq!(data.len(), 152089);
        let compressed = compress_lazy(&data).unwrap();
        let (total_seqs, match_seqs) = count_sequences(&compressed);
        let matched = total_match_bytes(&compressed);

        // Golden: 27564 seqs, 23951 matches, 124525 bytes matched.
        assert_eq!(total_seqs, 27564, "alice29.txt total sequences changed");
        assert_eq!(match_seqs, 23951, "alice29.txt match count changed");
        assert_eq!(matched, 124525, "alice29.txt match coverage changed");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lazy_quality_fields_c() {
        let Some(data) = corpus_file("fields.c") else {
            eprintln!("skipping: fields.c not found");
            return;
        };
        assert_eq!(data.len(), 11150);
        let compressed = compress_lazy(&data).unwrap();
        let (total_seqs, match_seqs) = count_sequences(&compressed);
        let matched = total_match_bytes(&compressed);

        // Golden: 1943 seqs, 1158 matches, 9207 bytes matched.
        assert_eq!(total_seqs, 1943, "fields.c total sequences changed");
        assert_eq!(match_seqs, 1158, "fields.c match count changed");
        assert_eq!(matched, 9207, "fields.c match coverage changed");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lazy_quality_grammar_lsp() {
        let Some(data) = corpus_file("grammar.lsp") else {
            eprintln!("skipping: grammar.lsp not found");
            return;
        };
        assert_eq!(data.len(), 3721);
        let compressed = compress_lazy(&data).unwrap();
        let (total_seqs, match_seqs) = count_sequences(&compressed);
        let matched = total_match_bytes(&compressed);

        // Golden: 867 seqs, 364 matches, 2854 bytes matched.
        assert_eq!(total_seqs, 867, "grammar.lsp total sequences changed");
        assert_eq!(match_seqs, 364, "grammar.lsp match count changed");
        assert_eq!(matched, 2854, "grammar.lsp match coverage changed");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_lazy_quality_xargs() {
        let Some(data) = corpus_file("xargs.1") else {
            eprintln!("skipping: xargs.1 not found");
            return;
        };
        assert_eq!(data.len(), 4227);
        let compressed = compress_lazy(&data).unwrap();
        let (total_seqs, match_seqs) = count_sequences(&compressed);
        let matched = total_match_bytes(&compressed);

        // Golden: 1235 seqs, 509 matches, 2992 bytes matched.
        assert_eq!(total_seqs, 1235, "xargs.1 total sequences changed");
        assert_eq!(match_seqs, 509, "xargs.1 match count changed");
        assert_eq!(matched, 2992, "xargs.1 match coverage changed");
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    /// Regression: all-same-byte input >65535 bytes would produce matches with
    /// length exceeding u16::MAX, causing silent truncation and corrupt output.
    #[test]
    fn test_round_trip_long_run_all_same() {
        let input = vec![0xAAu8; 70_000];
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "70KB all-same-byte round trip failed");
    }

    /// Regression: short repeating pattern >65535 bytes creates overlapping
    /// matches with length > u16::MAX via offset=pattern_len.
    #[test]
    fn test_round_trip_long_repeating_pattern() {
        let pattern = b"abcde";
        let input: Vec<u8> = pattern.iter().cycle().take(80_000).copied().collect();
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(
            decompressed, input,
            "80KB repeating pattern round trip failed"
        );
    }

    /// Verify find_match respects MAX_COMPARE_LEN (258) from SIMD comparisons.
    ///
    /// The SIMD compare_bytes function caps at MAX_COMPARE_LEN=258 (matching
    /// DEFLATE's max match length). This prevents CPU matches from exceeding
    /// u16::MAX. The u16 cap in find_match is defensive (the SIMD cap makes
    /// it unreachable on CPU, but the GPU path needs it).
    #[test]
    fn test_find_match_length_bounded() {
        let input = vec![0u8; 70_000];
        let mut finder = HashChainFinder::new();
        finder.insert(&input, 0);
        let m = finder.find_match(&input, 1);
        // SIMD compare_bytes caps at MAX_COMPARE_LEN = 258
        assert_eq!(
            m.length, 258,
            "find_match should be capped by MAX_COMPARE_LEN"
        );
        assert_eq!(m.offset, 1, "should match with offset 1");
    }

    /// Verify compress_lazy round-trips 100KB of identical bytes.
    ///
    /// Each match is capped at MAX_COMPARE_LEN=258 by the SIMD comparator,
    /// so many sequential matches are needed.
    #[test]
    fn test_compress_lazy_to_matches_large_all_same() {
        let input = vec![0xBBu8; 100_000];
        let matches = compress_lazy_to_matches(&input).unwrap();
        // With 258-byte max matches, need ~100000/259 ≈ 386 entries
        assert!(
            matches.len() > 100,
            "100KB all-same should need many matches, got {}",
            matches.len()
        );
        let compressed = compress_lazy(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }
}
