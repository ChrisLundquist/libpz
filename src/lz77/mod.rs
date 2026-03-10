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
const WINDOW_MASK: usize = MAX_WINDOW - 1;

/// Minimum match length to consider (shorter matches aren't worth encoding).
pub(crate) const MIN_MATCH: u16 = 3;

/// Hash table size for hash-chain match finder (power of 2).
pub(crate) const HASH_SIZE: usize = 1 << 15; // 32768
pub(crate) const HASH_MASK: usize = HASH_SIZE - 1;

/// Maximum number of chain links to follow per position.
pub(crate) const MAX_CHAIN: usize = 64;
/// Reduced chain depth used by auto/speed-biased parsing on large inputs.
const MAX_CHAIN_AUTO: usize = 48;

/// Maximum match length for DEFLATE-compatible pipelines (RFC 1951).
pub const DEFLATE_MAX_MATCH: u16 = 258;

/// Default maximum match length for non-DEFLATE pipelines.
/// Uses full u16 range since Match.length is u16.
pub const DEFAULT_MAX_MATCH: u16 = u16::MAX;

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

/// Serialize a slice of matches to bytes.
pub(crate) fn serialize_matches(matches: &[Match]) -> Vec<u8> {
    let mut out = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
    for m in matches {
        out.extend_from_slice(&m.to_bytes());
    }
    out
}

/// Deserialize bytes back into a vector of matches.
#[cfg(feature = "webgpu")]
pub(crate) fn deserialize_matches(data: &[u8]) -> Vec<Match> {
    let num = data.len() / Match::SERIALIZED_SIZE;
    let mut matches = Vec::with_capacity(num);
    for i in 0..num {
        let base = i * Match::SERIALIZED_SIZE;
        matches.push(Match::from_bytes(
            data[base..base + Match::SERIALIZED_SIZE]
                .try_into()
                .unwrap(),
        ));
    }
    matches
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

/// Compute a hash for 4 bytes at the given position.
///
/// Uses a multiply-shift construction for better avalanche than XOR.
/// Falls back to 0 (no hash) when fewer than 4 bytes remain.
#[inline(always)]
pub(crate) fn hash4(data: &[u8], pos: usize) -> usize {
    if pos + 3 >= data.len() {
        return 0;
    }
    let v = (data[pos] as u32)
        | ((data[pos + 1] as u32) << 8)
        | ((data[pos + 2] as u32) << 16)
        | ((data[pos + 3] as u32) << 24);
    // Multiply-shift: multiply by a large prime, take top 15 bits.
    // 0x9E37_79B9 is the Fibonacci/golden-ratio constant used by Knuth.
    ((v.wrapping_mul(0x9E37_79B9) >> 17) as usize) & HASH_MASK
}

/// Maximum hash insertion count per match (longer matches cap insertion
/// to avoid spending time on positions that will soon leave the window).
const MAX_INSERT_LEN: usize = 128;

/// Match length threshold above which lazy evaluation is skipped.
/// A match this long is unlikely to be beaten by the next position.
const LAZY_SKIP_THRESHOLD: u16 = 32;

/// Wide match result with u32 offset for finders with windows > 32KB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WideMatch {
    pub offset: u32,
    pub length: u16,
}

/// Hash-chain based match finder.
///
/// Maintains a hash table mapping 3-byte or 4-byte prefixes to positions,
/// with chains for collision resolution. Average O(n) complexity.
pub(crate) struct HashChainFinder {
    /// head[hash] = most recent position with this hash, or 0
    head: Vec<u32>,
    /// prev[pos % max_window] = previous position in the chain
    prev: Vec<u32>,
    /// Cached SIMD dispatcher — resolved once, avoids per-call feature detection.
    dispatcher: crate::simd::Dispatcher,
    /// Maximum match length to find. Deflate pipelines use 258 (RFC 1951);
    /// other pipelines can use larger values (up to u16::MAX) for better
    /// compression on repetitive data.
    max_match_len: usize,
    /// Maximum number of chain links to walk for each match search.
    max_chain: usize,
    /// Sliding window size. Defaults to MAX_WINDOW (32KB).
    /// Larger values enable longer-distance matches (requires `find_match_wide`).
    max_window: usize,
    /// Bitmask for modular indexing into `prev` (max_window - 1).
    window_mask: usize,
    /// Number of bytes used for hashing. 3 = current XOR hash, 4 = multiply-shift.
    /// Using 4 reduces hash collisions at the cost of requiring 4 bytes of lookahead.
    hash_prefix_len: u8,
}

impl HashChainFinder {
    /// Create a match finder with the DEFLATE-standard max match length (258).
    pub(crate) fn new() -> Self {
        Self::with_max_match_len(DEFLATE_MAX_MATCH)
    }

    /// Create a match finder with a caller-specified max match length.
    pub(crate) fn with_max_match_len(max_match_len: u16) -> Self {
        Self::with_tuning(max_match_len, MAX_CHAIN)
    }

    /// Create a match finder with explicit chain-depth tuning.
    pub(crate) fn with_tuning(max_match_len: u16, max_chain: usize) -> Self {
        Self {
            head: vec![0; HASH_SIZE],
            prev: vec![0; MAX_WINDOW],
            dispatcher: crate::simd::Dispatcher::new(),
            max_match_len: max_match_len as usize,
            max_chain: max_chain.clamp(1, MAX_CHAIN * 4),
            max_window: MAX_WINDOW,
            window_mask: WINDOW_MASK,
            hash_prefix_len: 3,
        }
    }

    /// Create a match finder with a custom window size.
    ///
    /// `max_window` must be a power of 2. The `prev` array scales with
    /// the window size (4 bytes per position). Use `find_match_wide()` to
    /// get u32 offsets for windows > 32KB.
    ///
    /// Memory: 128KB window = 512KB prev array, 256KB = 1MB, etc.
    #[allow(dead_code)]
    pub(crate) fn with_window(max_window: usize, max_match_len: u16) -> Self {
        debug_assert!(
            max_window.is_power_of_two(),
            "max_window must be power of 2"
        );
        Self {
            head: vec![0; HASH_SIZE],
            prev: vec![0; max_window],
            dispatcher: crate::simd::Dispatcher::new(),
            max_match_len: max_match_len as usize,
            max_chain: MAX_CHAIN,
            max_window,
            window_mask: max_window - 1,
            hash_prefix_len: 3,
        }
    }

    /// Create a match finder with a custom window size and chain depth.
    pub(crate) fn with_window_and_chain(
        max_window: usize,
        max_match_len: u16,
        max_chain: usize,
    ) -> Self {
        debug_assert!(
            max_window.is_power_of_two(),
            "max_window must be power of 2"
        );
        Self {
            head: vec![0; HASH_SIZE],
            prev: vec![0; max_window],
            dispatcher: crate::simd::Dispatcher::new(),
            max_match_len: max_match_len as usize,
            max_chain: max_chain.clamp(1, MAX_CHAIN * 4),
            max_window,
            window_mask: max_window - 1,
            hash_prefix_len: 3,
        }
    }

    /// Create a match finder with 4-byte hashing and a custom window size.
    ///
    /// 4-byte hashing reduces hash collisions at the cost of needing 4 bytes
    /// of lookahead (positions near EOF fall back to hash3 internally).
    pub(crate) fn with_hash4(max_window: usize, max_match_len: u16, max_chain: usize) -> Self {
        debug_assert!(
            max_window.is_power_of_two(),
            "max_window must be power of 2"
        );
        Self {
            head: vec![0; HASH_SIZE],
            prev: vec![0; max_window],
            dispatcher: crate::simd::Dispatcher::new(),
            max_match_len: max_match_len as usize,
            max_chain: max_chain.clamp(1, MAX_CHAIN * 4),
            max_window,
            window_mask: max_window - 1,
            hash_prefix_len: 4,
        }
    }

    /// Dispatch to hash3 or hash4 based on configuration.
    #[inline(always)]
    fn hash_at(&self, data: &[u8], pos: usize) -> usize {
        if self.hash_prefix_len == 4 {
            hash4(data, pos)
        } else {
            hash3(data, pos)
        }
    }

    /// Dynamically adjust chain depth. Used by adaptive encoding loops.
    /// Clamps to 1..=MAX_CHAIN*4.
    pub(crate) fn set_max_chain(&mut self, max_chain: usize) {
        self.max_chain = max_chain.clamp(1, MAX_CHAIN * 4);
    }

    /// Core match-finding loop. Returns (best_offset, best_length) as raw u32.
    ///
    /// Does not cap offset/length or ensure room for a literal `next` byte.
    /// Callers post-process the result for their specific needs.
    #[inline]
    fn find_best(&self, input: &[u8], pos: usize, max_lookback: usize) -> (u32, u32) {
        let remaining = input.len() - pos;
        if remaining < 3 {
            return (0, 0);
        }

        let h = self.hash_at(input, pos);
        let mut chain_pos = self.head[h] as usize;
        let mut best_offset: u32 = 0;
        let mut best_length: u32 = 0;
        let mut best_probe_byte: u8 = 0;
        let min_pos = pos.saturating_sub(max_lookback);
        let mut chain_count = 0;
        let cmp_limit = remaining.min(self.max_match_len);
        let input_ptr = input.as_ptr();
        // SAFETY: `pos < input.len()` whenever `remaining > 0`.
        let pos_ptr = unsafe { input_ptr.add(pos) };
        let prev = &self.prev;
        let window_mask = self.window_mask;

        while chain_pos >= min_pos && chain_pos < pos && chain_count < self.max_chain {
            // If a candidate differs at the current best-length probe point,
            // it cannot beat the current best match. Skip the SIMD compare.
            if best_length >= MIN_MATCH as u32 {
                let probe = best_length as usize;
                debug_assert!(probe < remaining);
                // SAFETY: chain_pos < pos and probe < remaining guarantees in-bounds.
                let candidate_probe = unsafe { *input_ptr.add(chain_pos + probe) };
                if candidate_probe != best_probe_byte {
                    let prev_pos = prev[chain_pos & window_mask] as usize;
                    if prev_pos >= chain_pos || prev_pos < min_pos {
                        break;
                    }
                    chain_pos = prev_pos;
                    chain_count += 1;
                    continue;
                }
            }

            // SIMD-accelerated byte comparison.
            // max_len is capped only by remaining bytes (not by offset distance),
            // allowing overlapping matches where length > offset. This enables
            // efficient encoding of repeated-byte runs (e.g., offset=1, length=999
            // for 1000 identical bytes). The decompressor's byte-by-byte copy loop
            // already handles the overlap correctly.
            let match_len = unsafe {
                self.dispatcher
                    .compare_bytes_ptr(input_ptr.add(chain_pos), pos_ptr, cmp_limit)
            } as u32;

            if match_len > best_length && match_len >= MIN_MATCH as u32 {
                best_length = match_len;
                best_offset = (pos - chain_pos) as u32;
                if best_length as usize >= cmp_limit {
                    break;
                }
                // SAFETY: best_length < cmp_limit <= remaining, so pos + best_length is valid.
                best_probe_byte = unsafe { *input_ptr.add(pos + best_length as usize) };
            }

            // Follow chain
            let prev_pos = prev[chain_pos & window_mask] as usize;
            if prev_pos >= chain_pos || prev_pos < min_pos {
                break;
            }
            chain_pos = prev_pos;
            chain_count += 1;
        }

        (best_offset, best_length)
    }

    /// Find the best match at `pos` in `input`, looking back up to MAX_WINDOW bytes.
    ///
    /// Uses SIMD-accelerated byte comparison (SSE2: 16 bytes/cycle,
    /// AVX2: 32 bytes/cycle) for the inner match extension loop.
    ///
    /// Returns a `Match` with u16 offset (safe for windows up to 32KB).
    /// For wider windows, use `find_match_wide()`.
    pub(crate) fn find_match(&self, input: &[u8], pos: usize) -> Match {
        let remaining = input.len() - pos;
        if remaining < 3 {
            return Match {
                offset: 0,
                length: 0,
                next: if pos < input.len() { input[pos] } else { 0 },
            };
        }

        // Cap lookback to u16::MAX so offset fits in Match.offset (u16).
        let max_lookback = self.max_window.min(u16::MAX as usize);
        let (best_offset, mut best_length) = self.find_best(input, pos, max_lookback);

        // Ensure room for the literal `next` byte (required by LZ77 triple format)
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

    /// Find the best match with u32 offset, for wide-window finders.
    ///
    /// Unlike `find_match()`, this:
    /// - Returns a u32 offset (supports windows > 32KB)
    /// - Does NOT reduce length to leave room for a literal `next` byte
    ///   (suitable for LZSS/LzSeq formats that separate literals from matches)
    pub(crate) fn find_match_wide(&self, input: &[u8], pos: usize) -> WideMatch {
        let (best_offset, best_length) = self.find_best(input, pos, self.max_window);

        WideMatch {
            offset: best_offset,
            length: best_length.min(u16::MAX as u32) as u16,
        }
    }

    /// Insert position `pos` into the hash chain.
    #[inline(always)]
    pub(crate) fn insert(&mut self, input: &[u8], pos: usize) {
        if pos + 2 >= input.len() {
            return;
        }
        let h = self.hash_at(input, pos);
        self.prev[pos & self.window_mask] = self.head[h];
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

        let h = self.hash_at(input, pos);
        let mut chain_pos = self.head[h] as usize;
        let min_pos = pos.saturating_sub(self.max_window);
        let mut chain_count = 0;
        let cmp_limit = remaining.min(self.max_match_len);
        let input_ptr = input.as_ptr();
        // SAFETY: `pos < input.len()` whenever `remaining > 0`.
        let pos_ptr = unsafe { input_ptr.add(pos) };

        // For each distinct length, keep the match with the smallest offset.
        // Use a simple vec of (length, offset) pairs; K is small.
        let mut found: Vec<(u16, u16)> = Vec::new();

        while chain_pos >= min_pos && chain_pos < pos && chain_count < self.max_chain {
            // Allow overlapping matches (length > offset) for run compression.
            let match_len = unsafe {
                self.dispatcher
                    .compare_bytes_ptr(input_ptr.add(chain_pos), pos_ptr, cmp_limit)
            } as u32;

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
            let prev_pos = self.prev[chain_pos & self.window_mask] as usize;
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
///
/// Uses `DEFLATE_MAX_MATCH` (258) as the maximum match length.
/// For configurable max match length, use `compress_lazy_to_matches_with_limit`.
pub fn compress_lazy_to_matches(input: &[u8]) -> PzResult<Vec<Match>> {
    compress_lazy_to_matches_with_limit(input, DEFLATE_MAX_MATCH)
}

/// Compress input using greedy matching: always take the longest match at
/// each position without lookahead. Faster than lazy but slightly worse ratio.
pub(crate) fn compress_greedy_to_matches_with_limit(
    input: &[u8],
    max_match_len: u16,
) -> PzResult<Vec<Match>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut matches = Vec::with_capacity(input.len() / 4);
    let mut finder = HashChainFinder::with_max_match_len(max_match_len);
    let mut pos: usize = 0;

    while pos < input.len() {
        let m = finder.find_match(input, pos);
        finder.insert(input, pos);

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

/// Like `compress_lazy_to_matches` but with a caller-specified max match length.
///
/// Non-Deflate pipelines (Lzf, LzSeqR, etc.) can pass `DEFAULT_MAX_MATCH` to find
/// longer matches on repetitive data without being constrained by DEFLATE.
pub(crate) fn compress_lazy_to_matches_with_limit(
    input: &[u8],
    max_match_len: u16,
) -> PzResult<Vec<Match>> {
    compress_lazy_to_matches_with_limit_and_chain(input, max_match_len, MAX_CHAIN)
}

/// Like `compress_lazy_to_matches_with_limit`, but with caller-controlled
/// chain depth for speed/ratio tuning.
pub(crate) fn compress_lazy_to_matches_with_limit_and_chain(
    input: &[u8],
    max_match_len: u16,
    max_chain: usize,
) -> PzResult<Vec<Match>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut matches = Vec::with_capacity(input.len() / 4);
    let mut finder = HashChainFinder::with_tuning(max_match_len, max_chain);
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
/// Uses `DEFLATE_MAX_MATCH` (258) as the maximum match length.
pub fn compress_lazy(input: &[u8]) -> PzResult<Vec<u8>> {
    compress_lazy_with_limit(input, DEFLATE_MAX_MATCH)
}

/// Like `compress_lazy` but with a caller-specified max match length.
pub(crate) fn compress_lazy_with_limit(input: &[u8], max_match_len: u16) -> PzResult<Vec<u8>> {
    let matches = compress_lazy_to_matches_with_limit(input, max_match_len)?;
    Ok(serialize_matches(&matches))
}

// ---------------------------------------------------------------------------
// Token-producing wrappers (for pluggable wire encoders)
// ---------------------------------------------------------------------------

/// Compress using greedy matching, returning a universal token stream.
pub(crate) fn compress_greedy_to_tokens_with_limit(
    input: &[u8],
    max_match_len: u16,
) -> PzResult<Vec<crate::lz_token::LzToken>> {
    let matches = compress_greedy_to_matches_with_limit(input, max_match_len)?;
    Ok(crate::lz_token::matches_to_tokens(&matches))
}

/// Compress using lazy matching, returning a universal token stream.
pub(crate) fn compress_lazy_to_tokens_with_limit(
    input: &[u8],
    max_match_len: u16,
) -> PzResult<Vec<crate::lz_token::LzToken>> {
    let matches = compress_lazy_to_matches_with_limit(input, max_match_len)?;
    Ok(crate::lz_token::matches_to_tokens(&matches))
}

/// Compress using lazy matching with tunable chain depth, returning tokens.
pub(crate) fn compress_lazy_to_tokens_with_limit_and_chain(
    input: &[u8],
    max_match_len: u16,
    max_chain: usize,
) -> PzResult<Vec<crate::lz_token::LzToken>> {
    let matches = compress_lazy_to_matches_with_limit_and_chain(input, max_match_len, max_chain)?;
    Ok(crate::lz_token::matches_to_tokens(&matches))
}

/// Parse-mode-aware chain-depth heuristic.
///
/// `prefer_speed = true` is used for auto/default CPU mode on large blocks.
pub(crate) fn select_chain_depth(input_len: usize, prefer_speed: bool) -> usize {
    if !prefer_speed {
        return MAX_CHAIN;
    }
    // Keep full search on small blocks. On larger inputs, progressively trim
    // chain depth where CPU time grows super-linearly but ratio gains flatten.
    if input_len >= 4 * 1024 * 1024 {
        24
    } else if input_len >= 1024 * 1024 {
        32
    } else if input_len >= 256 * 1024 {
        MAX_CHAIN_AUTO
    } else {
        MAX_CHAIN
    }
}

#[cfg(test)]
mod tests;
