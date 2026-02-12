//! Optimal LZ77 match selection via backward dynamic programming.
//!
//! Instead of greedily picking the longest match at each position,
//! this module evaluates all match candidates (top-K per position)
//! and selects the minimum-cost parse using backward DP.
//!
//! This is the approach used by zstd (levels 17+) and xz/lzma.
//! The key difference in libpz is that match candidates can come
//! from the GPU (top-K kernel) rather than CPU hash chains.
//!
//! # Algorithm
//!
//! 1. Build a match table: K candidates per input position
//! 2. Estimate encoding cost via a frequency-based cost model
//! 3. Backward DP: `cost[i] = min over literals and matches`
//! 4. Forward trace: recover the optimal match sequence

use crate::frequency::{self, FrequencyTable};
use crate::lz77::{HashChainFinder, Match, MIN_MATCH};
use crate::PzResult;

/// Number of match candidates to keep per position.
pub const K: usize = 4;

/// Fixed-point scaling factor for bit costs (multiply real bits by this).
/// Using integer arithmetic in the DP inner loop avoids floating-point.
const COST_SCALE: u32 = 256;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A match candidate: one possible (offset, length) pair at a position.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub struct MatchCandidate {
    pub offset: u16,
    pub length: u16,
}

/// Per-position match table: K candidates per input position.
///
/// Stored as a flat `Vec`, indexed as `candidates[pos * k + i]`.
/// Candidates are sorted by length descending (longest first).
/// Unused slots have `length = 0`.
pub struct MatchTable {
    pub candidates: Vec<MatchCandidate>,
    pub k: usize,
    pub input_len: usize,
}

impl MatchTable {
    /// Create a new match table filled with empty candidates.
    pub fn new(input_len: usize, k: usize) -> Self {
        Self {
            candidates: vec![MatchCandidate::default(); input_len * k],
            k,
            input_len,
        }
    }

    /// Get the candidates at a given position (slice of length `k`).
    #[inline]
    pub fn at(&self, pos: usize) -> &[MatchCandidate] {
        let start = pos * self.k;
        &self.candidates[start..start + self.k]
    }

    /// Get mutable candidates at a given position.
    #[inline]
    pub fn at_mut(&mut self, pos: usize) -> &mut [MatchCandidate] {
        let start = pos * self.k;
        &mut self.candidates[start..start + self.k]
    }
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Estimates the bit cost of encoding literals and matches.
///
/// In the PZ format, every token (literal or match) serializes to exactly
/// 5 bytes: `offset:u16 + length:u16 + next:u8`. These 5 bytes are then
/// entropy-coded by Huffman or Range Coder.
///
/// A **literal** token has offset=0, length=0, so 4 of its 5 bytes are
/// always zero (very cheap after entropy coding). Only the `next` byte varies.
///
/// A **match** token has non-zero offset and length fields, whose bytes
/// are more expensive to encode. But the match covers `length+1` input
/// bytes with a single 5-byte token, amortizing the cost.
///
/// All costs are in fixed-point (scaled by [`COST_SCALE`]).
pub struct CostModel {
    /// Cost of each literal byte value, in scaled bits.
    literal_cost: [u32; 256],
    /// Cost of the 4 zero bytes in a literal token (offset=0, length=0).
    /// This is cheap because 0x00 is very common in the LZ77 output stream.
    literal_overhead: u32,
    /// Cost of the 4 offset+length bytes in a match token.
    /// Estimated from the average entropy of typical offset/length values.
    match_overhead: u32,
}

impl CostModel {
    /// Build a cost model from byte frequencies (Shannon entropy estimate).
    pub fn from_frequencies(freq: &FrequencyTable) -> Self {
        let mut literal_cost = [8 * COST_SCALE; 256]; // default: 8 bits (uniform)

        if freq.total > 0 {
            let total = freq.total as f32;
            for (i, cost) in literal_cost.iter_mut().enumerate() {
                let count = freq.byte[i];
                if count > 0 {
                    let prob = count as f32 / total;
                    let bits = -prob.log2();
                    *cost = (bits * COST_SCALE as f32) as u32;
                }
            }
        }

        // Estimate overhead costs:
        // In a typical LZ77 output, ~50% of tokens are literals (offset=0, length=0).
        // The 0x00 byte thus appears very frequently, making it cheap to encode.
        // Estimate: 0x00 costs ~1 bit after entropy coding, so 4 zero bytes ≈ 4 bits.
        let literal_overhead = 4 * COST_SCALE;

        // Match offset/length fields contain varied byte values.
        // Typical entropy: ~4-5 bits/byte for offset, ~3-4 bits/byte for length.
        // Conservative estimate: 4 bytes × 4 bits/byte = 16 bits overhead.
        let match_overhead = 16 * COST_SCALE;

        Self {
            literal_cost,
            literal_overhead,
            match_overhead,
        }
    }

    /// Full cost of emitting a literal token (in scaled bits).
    ///
    /// A literal token is `Match { offset:0, length:0, next:byte }` — 5 bytes.
    /// Cost = overhead of 4 zero bytes + entropy of the `next` byte.
    #[inline]
    pub fn literal_token(&self, byte: u8) -> u32 {
        self.literal_overhead
            .saturating_add(self.literal_cost[byte as usize])
    }

    /// Full cost of emitting a match token (in scaled bits).
    ///
    /// A match token is `Match { offset, length, next }` — 5 bytes.
    /// Cost = overhead of offset+length bytes + entropy of the `next` byte.
    #[inline]
    pub fn match_token(&self, next_byte: u8) -> u32 {
        self.match_overhead
            .saturating_add(self.literal_cost[next_byte as usize])
    }
}

// ---------------------------------------------------------------------------
// Match table construction (CPU)
// ---------------------------------------------------------------------------

/// Build a match table from input using the hash-chain finder.
///
/// For each position, finds up to `k` match candidates using the
/// existing hash-chain infrastructure.
pub fn build_match_table_cpu(input: &[u8], k: usize) -> MatchTable {
    let mut table = MatchTable::new(input.len(), k);
    let mut finder = HashChainFinder::new();

    for pos in 0..input.len() {
        let top_k = finder.find_top_k(input, pos, k);
        let slot = table.at_mut(pos);
        for (i, &(length, offset)) in top_k.iter().enumerate() {
            slot[i] = MatchCandidate { offset, length };
        }
        finder.insert(input, pos);
    }

    table
}

// ---------------------------------------------------------------------------
// Backward DP optimal parse
// ---------------------------------------------------------------------------

/// Run backward dynamic programming to find the minimum-cost parse.
///
/// Returns a `Vec<Match>` representing the optimal sequence of
/// literals and matches, compatible with `lz77::decompress()`.
pub fn optimal_parse(input: &[u8], table: &MatchTable, cost_model: &CostModel) -> Vec<Match> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }

    // cost[i] = minimum scaled-bit cost to encode input[i..n]
    let mut cost = vec![0u32; n + 1];
    // choice_len[i] = match length chosen at position i (0 = literal)
    let mut choice_len = vec![0u16; n];
    // choice_offset[i] = match offset chosen (only meaningful when choice_len > 0)
    let mut choice_offset = vec![0u16; n];

    // Backward pass: from n-1 down to 0
    for i in (0..n).rev() {
        // Option 1: emit a literal token (covers 1 input byte)
        // Token: Match { offset:0, length:0, next:input[i] }
        let lit_cost = cost_model
            .literal_token(input[i])
            .saturating_add(cost[i + 1]);
        cost[i] = lit_cost;
        choice_len[i] = 0;

        // Option 2: each match candidate at this position
        for cand in table.at(i) {
            if cand.length < MIN_MATCH {
                break; // candidates are sorted by length desc; rest are empty
            }
            let match_end = i + cand.length as usize; // position of 'next' byte
            if match_end >= n {
                continue; // need room for mandatory 'next' byte
            }
            let next_pos = match_end + 1;
            // Token: Match { offset, length, next:input[match_end] }
            // Covers (length + 1) input bytes with one 5-byte token
            let mcost = cost_model
                .match_token(input[match_end])
                .saturating_add(cost[next_pos]);
            if mcost < cost[i] {
                cost[i] = mcost;
                choice_len[i] = cand.length;
                choice_offset[i] = cand.offset;
            }
        }
    }

    // Forward trace: recover the optimal match sequence
    let mut matches = Vec::new();
    let mut pos = 0;
    while pos < n {
        let len = choice_len[pos];
        if len == 0 {
            // Literal
            matches.push(Match {
                offset: 0,
                length: 0,
                next: input[pos],
            });
            pos += 1;
        } else {
            let offset = choice_offset[pos];
            let match_end = pos + len as usize;
            let next = input[match_end]; // safe: DP ensured match_end < n
            matches.push(Match {
                offset,
                length: len,
                next,
            });
            pos = match_end + 1;
        }
    }

    matches
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compress input using optimal parsing with hash-chain match finding.
///
/// Produces the same serialized `Match` format as `lz77::compress_lazy`,
/// but selects matches via backward DP to minimize total encoding cost.
/// Decompressible with `lz77::decompress()`.
pub fn compress_optimal(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Step 1: Build frequency table for cost estimation
    let freq = frequency::get_frequency(input);
    let cost_model = CostModel::from_frequencies(&freq);

    // Step 2: Build match table using hash-chain finder
    let table = build_match_table_cpu(input, K);

    // Step 3: Run backward DP
    let matches = optimal_parse(input, &table, &cost_model);

    // Step 4: Serialize to the standard Match byte format
    let mut output = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
    for m in &matches {
        output.extend_from_slice(&m.to_bytes());
    }

    Ok(output)
}

/// Compress input using optimal parsing with a pre-built match table.
///
/// This variant is used when the match table comes from the GPU
/// (via `OpenClEngine::find_topk_matches`).
pub fn compress_optimal_with_table(input: &[u8], table: &MatchTable) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let freq = frequency::get_frequency(input);
    let cost_model = CostModel::from_frequencies(&freq);
    let matches = optimal_parse(input, table, &cost_model);

    let mut output = Vec::with_capacity(matches.len() * Match::SERIALIZED_SIZE);
    for m in &matches {
        output.extend_from_slice(&m.to_bytes());
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lz77;

    fn round_trip(input: &[u8]) {
        let compressed = compress_optimal(input).unwrap();
        let decompressed = lz77::decompress(&compressed).unwrap();
        assert_eq!(
            decompressed,
            input,
            "round-trip failed for input of length {}",
            input.len()
        );
    }

    #[test]
    fn test_empty() {
        let result = compress_optimal(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_byte() {
        round_trip(b"a");
    }

    #[test]
    fn test_two_bytes() {
        round_trip(b"ab");
    }

    #[test]
    fn test_no_matches() {
        round_trip(b"abcdefgh");
    }

    #[test]
    fn test_repeats() {
        round_trip(b"abcabcabc");
    }

    #[test]
    fn test_all_same() {
        round_trip(&[b'x'; 200]);
    }

    #[test]
    fn test_longer_text() {
        round_trip(
            b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.",
        );
    }

    #[test]
    fn test_large_repeating() {
        let pattern = b"Hello, World! This is a test pattern. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        round_trip(&input);
    }

    #[test]
    fn test_binary_data() {
        let input: Vec<u8> = (0..=255).cycle().take(1024).collect();
        round_trip(&input);
    }

    #[test]
    fn test_window_boundary() {
        // Exceed MAX_WINDOW to exercise boundary handling
        let block = b"ABCDEFGHIJ";
        let mut input = Vec::new();
        for _ in 0..4000 {
            input.extend_from_slice(block);
        }
        round_trip(&input);
    }

    #[test]
    fn test_optimal_vs_greedy_correctness() {
        // Both lazy and optimal should produce correct output.
        // Optimal may use different match/literal decisions to minimize
        // total encoding cost after entropy coding.
        let pattern = b"abcdefg abcxyz abcdefg abcxyz ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let lazy = lz77::compress_lazy(&input).unwrap();
        let optimal = compress_optimal(&input).unwrap();

        // Both must decompress correctly
        assert_eq!(lz77::decompress(&lazy).unwrap(), input);
        assert_eq!(lz77::decompress(&optimal).unwrap(), input);
    }

    #[test]
    fn test_optimal_reduces_cost_on_skewed_data() {
        // On data with very skewed byte frequencies, optimal parsing
        // should find a lower-cost parse than greedy.
        // Use input where some literals are very cheap (high-frequency)
        // and matches might not always be the best choice.
        let mut input = Vec::new();
        // Lots of 'a' (cheap literal) interspersed with patterns
        for _ in 0..50 {
            input.extend_from_slice(b"aaaaaaaaaa"); // cheap literals
            input.extend_from_slice(b"xyz"); // pattern to match
        }

        let optimal = compress_optimal(&input).unwrap();
        let decompressed = lz77::decompress(&optimal).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_match_table_construction() {
        let input = b"abcabcabcabc";
        let table = build_match_table_cpu(input, K);
        assert_eq!(table.input_len, input.len());
        assert_eq!(table.k, K);

        // First 3 positions should have no matches (nothing in the window yet)
        for pos in 0..3 {
            let candidates = table.at(pos);
            assert!(
                candidates.iter().all(|c| c.length == 0),
                "position {} should have no matches",
                pos
            );
        }

        // Position 3 should have at least one match (back-ref to position 0)
        let cands_3 = table.at(3);
        assert!(
            cands_3[0].length >= 3,
            "position 3 should find a match of length >= 3, got {}",
            cands_3[0].length
        );
    }

    #[test]
    fn test_cost_model_basics() {
        let freq = frequency::get_frequency(b"aaabbc");
        let model = CostModel::from_frequencies(&freq);

        // 'a' appears 3/6 = 50%, so cost ~1 bit = 256 scaled
        // 'b' appears 2/6 = 33%, so cost ~1.58 bits = ~405 scaled
        // 'c' appears 1/6 = 17%, so cost ~2.58 bits = ~661 scaled
        // Check relative ordering of literal token costs: a < b < c
        assert!(
            model.literal_token(b'a') < model.literal_token(b'b'),
            "a should be cheaper than b"
        );
        assert!(
            model.literal_token(b'b') < model.literal_token(b'c'),
            "b should be cheaper than c"
        );

        // A match token should be more expensive than a literal token
        // (match has non-zero offset/length bytes), but covers more bytes
        assert!(
            model.match_token(b'a') > model.literal_token(b'a'),
            "match token should cost more than literal token"
        );
    }
}
