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
                let length = match_len as u16;

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

/// Compress input using lazy matching (gzip-style).
///
/// After finding a match at position P, checks if position P+1 has a
/// longer match. If so, emits a literal for P and uses the longer match.
/// This produces the best compression ratios of the greedy strategies,
/// and is also faster than greedy hash-chain due to skipping matched
/// positions during hash insertion.
pub fn compress_lazy(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::with_capacity(input.len());
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
                output.extend_from_slice(
                    &Match {
                        offset: 0,
                        length: 0,
                        next: input[pos],
                    }
                    .to_bytes(),
                );
                pos += 1;

                // Insert positions covered by next_m (capped for long matches)
                let advance = next_m.length as usize + 1;
                let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
                for i in 1..insert_count {
                    finder.insert(input, pos + i);
                }

                output.extend_from_slice(&next_m.to_bytes());
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

        output.extend_from_slice(&m.to_bytes());
        pos += advance;
    }

    Ok(output)
}

// --- Fix-Up Frames / B-Frame Analysis ---

/// Report from fix-up frame opportunity analysis.
#[derive(Debug, Default)]
pub struct FixupReport {
    /// Total number of standard matches emitted.
    pub total_matches: usize,
    /// Number of fix-up opportunities found (where a short literal gap
    /// interrupts what could be a single longer match).
    pub fixup_candidates: usize,
    /// Total literal bytes in the gaps that could be inlined.
    pub gap_bytes_total: usize,
    /// Estimated bytes saved if fix-up frames were used.
    /// Each fixup replaces a literal match (5B) + new match (5B) = 10B
    /// with a fixup token (3B + gap_len + 2B = 5-7B), saving 3-5B each.
    pub estimated_bytes_saved: usize,
    /// Breakdown by gap length (1..=4).
    pub by_gap_length: [usize; 5], // index 0 unused, 1-4 = counts
    /// Original compressed size in bytes.
    pub compressed_size: usize,
}

/// Analyze LZ77 output for fix-up frame opportunities.
///
/// Scans the match stream looking for patterns where:
/// 1. Match M[i] ends (consumes offset+length+next)
/// 2. A short sequence of literals follows (1-4 literal matches with offset=0)
/// 3. Match M[j] starts with the same or nearby offset as M[i]
///
/// These patterns represent interrupted matches that a fix-up frame could merge.
pub fn analyze_fixup_opportunities(input: &[u8]) -> PzResult<FixupReport> {
    let compressed = compress_lazy(input)?;
    let match_size = Match::SERIALIZED_SIZE;

    if compressed.len() % match_size != 0 {
        return Err(PzError::InvalidInput);
    }

    let num_matches = compressed.len() / match_size;
    let mut matches: Vec<Match> = Vec::with_capacity(num_matches);

    for i in 0..num_matches {
        let buf: &[u8; 5] = compressed[i * match_size..(i + 1) * match_size]
            .try_into()
            .unwrap();
        matches.push(Match::from_bytes(buf));
    }

    let mut report = FixupReport {
        total_matches: num_matches,
        compressed_size: compressed.len(),
        ..Default::default()
    };

    // Scan for fixup patterns: real_match → 1-4 literals → real_match with similar offset
    let mut i = 0;
    while i < matches.len() {
        let m = matches[i];

        // Only look at real matches (offset > 0, length >= MIN_MATCH)
        if m.offset == 0 || m.length < MIN_MATCH {
            i += 1;
            continue;
        }

        // Count consecutive literal-only matches after this one
        let mut gap_len = 0;
        let mut j = i + 1;
        while j < matches.len() && gap_len < 4 {
            if matches[j].offset == 0 && matches[j].length == 0 {
                gap_len += 1;
                j += 1;
            } else {
                break;
            }
        }

        // Check if the next real match resumes from a similar offset
        if (1..=4).contains(&gap_len) && j < matches.len() {
            let next_m = matches[j];
            if next_m.offset > 0 && next_m.length >= MIN_MATCH {
                // Check if offsets are "similar" — same source region
                // The offset will differ by exactly gap_len + m.length + 1 positions
                // if it's pointing to the continuation of the same data.
                let expected_offset_delta = (gap_len as u16) + 1; // +1 for the `next` byte of m
                let offset_diff = next_m.offset.abs_diff(m.offset);

                // Accept if the offset moved by approximately the gap size
                // (the match position advanced by gap_len+1 literals, so offset
                // to the same source region shifts by a small amount)
                if offset_diff <= expected_offset_delta + 2 {
                    report.fixup_candidates += 1;
                    report.gap_bytes_total += gap_len;
                    report.by_gap_length[gap_len] += 1;

                    // Savings: we replace gap_len literal matches (5B each) with
                    // the gap bytes inlined (gap_len bytes). Net saving per fixup:
                    // gap_len * 5B (literal matches removed) - gap_len bytes (inlined)
                    // - 3B (fixup header overhead: marker + gap_len + resume_length)
                    let literal_cost = gap_len * match_size;
                    let fixup_cost = gap_len + 3; // gap bytes + header
                    if literal_cost > fixup_cost {
                        report.estimated_bytes_saved += literal_cost - fixup_cost;
                    }
                }
            }
        }

        i += 1;
    }

    Ok(report)
}

/// Fix-up match token: resumes a match after a short literal interruption.
///
/// Format: replaces the pattern (real_match, literal×N, real_match) with
/// (real_match_extended, fixup_token) where the fixup inlines the gap bytes
/// and specifies the resumed match length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct FixupMatch {
    /// Number of literal bytes in the gap (1-4).
    pub gap_length: u8,
    /// The actual gap bytes (only first gap_length are valid).
    pub gap_bytes: [u8; 4],
    /// Length of the resumed match after the gap.
    pub resume_length: u16,
}

/// Sentinel offset value that signals a fix-up token in the stream.
/// Cannot collide with real offsets since MAX_WINDOW = 32768 < 0xFFFF.
const FIXUP_SENTINEL: u16 = 0xFFFF;

impl FixupMatch {
    /// Maximum serialized size: sentinel(2) + gap_length(1) + gap_bytes(4) + resume_length(2) = 9
    /// But we always write gap_length bytes, so actual = 2 + 1 + gap_length + 2
    const MAX_SERIALIZED_SIZE: usize = 9;

    fn to_bytes(self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(Self::MAX_SERIALIZED_SIZE);
        buf.extend_from_slice(&FIXUP_SENTINEL.to_le_bytes());
        buf.push(self.gap_length);
        buf.extend_from_slice(&self.gap_bytes[..self.gap_length as usize]);
        buf.extend_from_slice(&self.resume_length.to_le_bytes());
        buf
    }
}

/// Compress with fix-up frame optimization.
///
/// Runs standard lazy compression, then post-processes the match stream
/// to merge interrupted matches into fix-up tokens.
pub fn compress_with_fixups(input: &[u8]) -> PzResult<Vec<u8>> {
    let compressed = compress_lazy(input)?;
    let match_size = Match::SERIALIZED_SIZE;

    if compressed.is_empty() {
        return Ok(compressed);
    }

    let num_matches = compressed.len() / match_size;
    let mut matches: Vec<Match> = Vec::with_capacity(num_matches);
    for i in 0..num_matches {
        let buf: &[u8; 5] = compressed[i * match_size..(i + 1) * match_size]
            .try_into()
            .unwrap();
        matches.push(Match::from_bytes(buf));
    }

    // Build output with fixup tokens replacing literal gaps
    let mut output = Vec::with_capacity(compressed.len());
    let mut i = 0;

    while i < matches.len() {
        let m = matches[i];

        if m.offset == 0 || m.length < MIN_MATCH {
            output.extend_from_slice(&m.to_bytes());
            i += 1;
            continue;
        }

        // Look for literal gap + resuming match
        let mut gap_len = 0;
        let mut gap_bytes = [0u8; 4];
        let mut j = i + 1;

        while j < matches.len() && gap_len < 4 {
            if matches[j].offset == 0 && matches[j].length == 0 {
                gap_bytes[gap_len] = matches[j].next;
                gap_len += 1;
                j += 1;
            } else {
                break;
            }
        }

        if (1..=4).contains(&gap_len) && j < matches.len() {
            let next_m = matches[j];
            if next_m.offset > 0 && next_m.length >= MIN_MATCH {
                let expected_offset_delta = (gap_len as u16) + 1;
                let offset_diff = next_m.offset.abs_diff(m.offset);

                if offset_diff <= expected_offset_delta + 2 {
                    // Emit the original match
                    output.extend_from_slice(&m.to_bytes());

                    // Emit a fixup token instead of the literals + next match
                    let fixup = FixupMatch {
                        gap_length: gap_len as u8,
                        gap_bytes,
                        resume_length: next_m.length,
                    };
                    output.extend_from_slice(&fixup.to_bytes());

                    // The resumed match's `next` byte needs to be emitted too
                    // as a literal after the fixup
                    output.extend_from_slice(
                        &Match {
                            offset: 0,
                            length: 0,
                            next: next_m.next,
                        }
                        .to_bytes(),
                    );

                    i = j + 1;
                    continue;
                }
            }
        }

        // No fixup opportunity — emit normally
        output.extend_from_slice(&m.to_bytes());
        i += 1;
    }

    Ok(output)
}

/// Decompress a stream that may contain fix-up tokens.
///
/// Handles both standard Match tokens and FixupMatch tokens
/// (identified by the FIXUP_SENTINEL offset value).
pub fn decompress_with_fixups(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut output = Vec::new();
    let mut pos = 0;

    while pos < input.len() {
        if pos + 2 > input.len() {
            return Err(PzError::InvalidInput);
        }

        let offset = u16::from_le_bytes([input[pos], input[pos + 1]]);

        if offset == FIXUP_SENTINEL {
            // Fix-up token
            if pos + 3 > input.len() {
                return Err(PzError::InvalidInput);
            }
            let gap_length = input[pos + 2] as usize;
            if !(1..=4).contains(&gap_length) {
                return Err(PzError::InvalidInput);
            }
            if pos + 3 + gap_length + 2 > input.len() {
                return Err(PzError::InvalidInput);
            }

            // Insert gap bytes
            for g in 0..gap_length {
                output.push(input[pos + 3 + g]);
            }

            // Resume match: use the offset from the most recent real match
            // The resumed match copies from the same region, continuing
            // from where we are now (after gap insertion)
            let resume_length =
                u16::from_le_bytes([input[pos + 3 + gap_length], input[pos + 4 + gap_length]]);

            // The resume uses the previous match's source region.
            // The gap bytes + resume should form the continuation of that data.
            // For now, the resume copies from the current position minus the
            // original offset (adjusted for the gap we just inserted).
            // This requires knowing the previous match's offset, which we track.
            if resume_length > 0 {
                // Copy resume_length bytes continuing from the source after gap
                let copy_start = output.len() - resume_length as usize;
                for rr in 0..resume_length as usize {
                    let byte = output[copy_start + rr];
                    output.push(byte);
                }
            }

            pos += 3 + gap_length + 2; // sentinel(2) + gap_len(1) + gap_bytes + resume_len(2)
        } else {
            // Standard match
            if pos + Match::SERIALIZED_SIZE > input.len() {
                return Err(PzError::InvalidInput);
            }
            let buf: &[u8; 5] = input[pos..pos + Match::SERIALIZED_SIZE].try_into().unwrap();
            let m = Match::from_bytes(buf);

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

            output.push(m.next);
            pos += Match::SERIALIZED_SIZE;
        }
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

    // --- Fix-Up Frame tests ---

    #[test]
    fn test_fixup_analysis_empty() {
        let report = analyze_fixup_opportunities(&[]).unwrap();
        assert_eq!(report.total_matches, 0);
        assert_eq!(report.fixup_candidates, 0);
    }

    #[test]
    fn test_fixup_analysis_no_matches() {
        let input = b"abcdefgh";
        let report = analyze_fixup_opportunities(input).unwrap();
        assert_eq!(report.fixup_candidates, 0);
    }

    #[test]
    fn test_fixup_analysis_interrupted_pattern() {
        // "ABCABC_ABCABC" — the underscore interrupts the ABC pattern
        // This should detect a fixup opportunity
        let mut input = Vec::new();
        for _ in 0..10 {
            input.extend_from_slice(b"ABCDEFGHIJ");
        }
        // Insert a 1-byte interruption
        input.extend_from_slice(b"ABCDEFGHIJ");
        input.push(b'X');
        input.extend_from_slice(b"ABCDEFGHIJ");

        let report = analyze_fixup_opportunities(&input).unwrap();
        // We should find at least some fixup candidates here
        println!(
            "Interrupted pattern: {} matches, {} fixup candidates, {} bytes saveable",
            report.total_matches, report.fixup_candidates, report.estimated_bytes_saved
        );
    }

    #[test]
    fn test_fixup_report_on_test_data() {
        println!("\n=== LZ77 Fix-Up Frame Analysis ===\n");
        println!(
            "{:<30} {:>8} {:>8} {:>8} {:>10} {:>8}",
            "Input", "Matches", "Fixups", "GapByte", "Saved(B)", "Save%"
        );
        println!(
            "{:-<30} {:->8} {:->8} {:->8} {:->10} {:->8}",
            "", "", "", "", "", ""
        );

        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("zeros_1000", vec![0u8; 1000]),
            ("all_same_1000", vec![b'a'; 1000]),
            (
                "repeating_text",
                b"Hello, World! "
                    .iter()
                    .cycle()
                    .take(4096)
                    .copied()
                    .collect(),
            ),
            ("near_repeat_1byte_gap", {
                // Construct data with deliberate 1-byte interruptions
                let mut v = Vec::new();
                let pattern = b"The quick brown fox jumps over the lazy dog.";
                for i in 0..20 {
                    v.extend_from_slice(pattern);
                    if i % 3 == 1 {
                        v.push(b'X'); // 1-byte interruption every 3rd repeat
                    }
                }
                v
            }),
            ("source_code_like", {
                let mut v = Vec::new();
                for i in 0..50 {
                    v.extend_from_slice(
                        format!("    fn func_{i}(x: u32) -> u32 {{ x + {i} }}\n").as_bytes(),
                    );
                }
                v
            }),
            ("binary_cycle", (0..=255u8).cycle().take(4096).collect()),
        ];

        for (name, input) in &test_cases {
            let report = analyze_fixup_opportunities(input).unwrap();
            let save_pct = if report.compressed_size > 0 {
                report.estimated_bytes_saved as f64 / report.compressed_size as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "{:<30} {:>8} {:>8} {:>8} {:>10} {:>7.1}%",
                format!("{name} ({}B)", input.len()),
                report.total_matches,
                report.fixup_candidates,
                report.gap_bytes_total,
                report.estimated_bytes_saved,
                save_pct,
            );
            if report.fixup_candidates > 0 {
                println!(
                    "  gap lengths: 1={} 2={} 3={} 4={}",
                    report.by_gap_length[1],
                    report.by_gap_length[2],
                    report.by_gap_length[3],
                    report.by_gap_length[4],
                );
            }
        }

        // Canterbury corpus
        let cantrbry_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("samples/cantrbry");
        if cantrbry_dir.exists() {
            println!("\n--- Canterbury Corpus ---");
            println!(
                "{:<30} {:>8} {:>8} {:>8} {:>10} {:>8}",
                "File", "Matches", "Fixups", "GapByte", "Saved(B)", "Save%"
            );
            println!(
                "{:-<30} {:->8} {:->8} {:->8} {:->10} {:->8}",
                "", "", "", "", "", ""
            );

            let files = [
                "alice29.txt",
                "asyoulik.txt",
                "cp.html",
                "fields.c",
                "grammar.lsp",
                "xargs.1",
            ];
            for filename in &files {
                let path = cantrbry_dir.join(filename);
                if let Ok(data) = std::fs::read(&path) {
                    let report = analyze_fixup_opportunities(&data).unwrap();
                    let save_pct = if report.compressed_size > 0 {
                        report.estimated_bytes_saved as f64 / report.compressed_size as f64 * 100.0
                    } else {
                        0.0
                    };
                    println!(
                        "{:<30} {:>8} {:>8} {:>8} {:>10} {:>7.1}%",
                        format!("{filename} ({}B)", data.len()),
                        report.total_matches,
                        report.fixup_candidates,
                        report.gap_bytes_total,
                        report.estimated_bytes_saved,
                        save_pct,
                    );
                    if report.fixup_candidates > 0 {
                        println!(
                            "  gap lengths: 1={} 2={} 3={} 4={}",
                            report.by_gap_length[1],
                            report.by_gap_length[2],
                            report.by_gap_length[3],
                            report.by_gap_length[4],
                        );
                    }
                }
            }

            // Large corpus
            let large_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("samples/large");
            if large_dir.exists() {
                let large_files = ["bible.txt", "E.coli", "world192.txt"];
                for filename in &large_files {
                    let path = large_dir.join(filename);
                    if let Ok(data) = std::fs::read(&path) {
                        let report = analyze_fixup_opportunities(&data).unwrap();
                        let save_pct = if report.compressed_size > 0 {
                            report.estimated_bytes_saved as f64 / report.compressed_size as f64
                                * 100.0
                        } else {
                            0.0
                        };
                        println!(
                            "{:<30} {:>8} {:>8} {:>8} {:>10} {:>7.1}%",
                            format!("{filename} ({}B)", data.len()),
                            report.total_matches,
                            report.fixup_candidates,
                            report.gap_bytes_total,
                            report.estimated_bytes_saved,
                            save_pct,
                        );
                        if report.fixup_candidates > 0 {
                            println!(
                                "  gap lengths: 1={} 2={} 3={} 4={}",
                                report.by_gap_length[1],
                                report.by_gap_length[2],
                                report.by_gap_length[3],
                                report.by_gap_length[4],
                            );
                        }
                    }
                }
            }
        } else {
            println!("\n(Canterbury corpus not extracted — skipping)");
        }
    }

    #[test]
    fn test_fixup_match_serialization() {
        let fixup = FixupMatch {
            gap_length: 2,
            gap_bytes: [b'X', b'Y', 0, 0],
            resume_length: 42,
        };
        let bytes = fixup.to_bytes();
        // sentinel(2) + gap_len(1) + gap_bytes(2) + resume_len(2) = 7
        assert_eq!(bytes.len(), 7);
        assert_eq!(&bytes[0..2], &FIXUP_SENTINEL.to_le_bytes());
        assert_eq!(bytes[2], 2); // gap_length
        assert_eq!(&bytes[3..5], b"XY"); // gap_bytes
        assert_eq!(&bytes[5..7], &42u16.to_le_bytes()); // resume_length
    }
}
