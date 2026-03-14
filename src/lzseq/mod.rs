/// LzSeq: zstd-style code+extra-bits sequence encoding for LZ matches.
///
/// Uses log2-based code tables for offsets and lengths, with variable-width
/// extra bits packed into separate bitstreams. Close matches cost 2-3 bytes,
/// far matches cost proportionally to log2(distance).
///
/// ## Repeat offsets
///
/// The encoder/decoder track the 3 most recently used offsets. Offset codes
/// 0-2 signal repeat offsets (0 extra bits each). Codes 3+ are literal
/// offsets shifted by 3 from the base table.
///
/// ## Output: 6 independent streams for entropy coding
///
/// - flags: 1 bit/token packed MSB-first (literal=1, match=0)
/// - literals: 1 byte per literal token
/// - offset_codes: 1 byte per match (0-2 = repeat, 3+ = literal offset)
/// - offset_extra: packed bitstream (LSB-first, 0 bits for repeats)
/// - length_codes: 1 byte per match (log2-based code)
/// - length_extra: packed bitstream (LSB-first)
///
/// ## Base code table (log2-based, applied after repeat offset shift)
///
/// Offset codes (raw, before +3 shift for repeat reservation):
///   Code 0: offset 1       (0 extra bits)
///   Code 1: offset 2       (0 extra bits)
///   Code 2: offset 3-4     (1 extra bit)
///   Code 3: offset 5-8     (2 extra bits)
///   Code N (N>=2): base = 1 + 2^(N-1), extra = N-1 bits
///
/// Length codes (MIN_MATCH=3 bias, same structure):
///   Code 0: length 3       (0 extra bits)
///   Code 1: length 4       (0 extra bits)
///   Code 2: length 5-6     (1 extra bit)
///   Code 3: length 7-10    (2 extra bits)
use crate::lz77::{HashChainFinder, MIN_MATCH};
use crate::{PzError, PzResult};

/// Match length threshold above which lazy evaluation is skipped.
const LAZY_SKIP_THRESHOLD: u16 = 32;

/// Maximum hash insertion count per match.
const MAX_INSERT_LEN: usize = 128;

/// Configuration for LzSeq encoding.
pub struct SeqConfig {
    /// Maximum lookback window size in bytes. Must be a power of 2.
    /// Default: 128KB. Use larger values for better compression on data
    /// with long-range repeats.
    pub max_window: usize,
    /// Hash prefix length for match finding: 3 (XOR, default) or 4 (multiply-shift).
    /// Using 4 reduces hash collisions and may improve ratio on large inputs at
    /// a small speed cost (one extra byte of lookahead required per position).
    pub hash_prefix_len: u8,
    /// Maximum hash chain depth. Higher values improve ratio at the cost of speed.
    /// Default: 64. Clamped to 1..=256 internally.
    pub max_chain: usize,
    /// When true, reduce chain depth on blocks with low match density.
    /// Speeds up encoding on incompressible data with minimal ratio cost.
    /// Default: false.
    pub adaptive_chain: bool,
    /// Maximum match length. Default: `u16::MAX` (extended matches).
    /// Set to 258 to emulate DEFLATE constraints.
    pub max_match_len: u16,
}

impl Default for SeqConfig {
    fn default() -> Self {
        SeqConfig {
            max_window: 128 * 1024,
            hash_prefix_len: 4,
            max_chain: crate::lz77::MAX_CHAIN,
            adaptive_chain: true,
            max_match_len: crate::lz77::DEFAULT_MAX_MATCH,
        }
    }
}

impl SeqConfig {
    /// Fast preset: 64KB window, chain 32. Faster than default, lower ratio.
    pub fn fast() -> Self {
        SeqConfig {
            max_window: 64 * 1024,
            hash_prefix_len: 3,
            max_chain: 32,
            adaptive_chain: false,
            max_match_len: crate::lz77::DEFAULT_MAX_MATCH,
        }
    }

    /// Default preset: 128KB window, chain 64.
    pub fn default_quality() -> Self {
        SeqConfig::default()
    }

    /// High preset: 256KB window, chain 128, 4-byte hash.
    /// Better ratio at the cost of higher memory and time.
    /// Note: Uses ~1MB of memory for the hash finder (256KB `prev` array).
    pub fn high() -> Self {
        SeqConfig {
            max_window: 256 * 1024,
            hash_prefix_len: 4,
            max_chain: 128,
            adaptive_chain: false,
            max_match_len: crate::lz77::DEFAULT_MAX_MATCH,
        }
    }
}

/// Encoded output: 6 independent streams ready for entropy coding.
pub struct SeqEncoded {
    /// Packed flag bits, MSB-first. 1=literal, 0=match.
    pub flags: Vec<u8>,
    /// One byte per literal token.
    pub literals: Vec<u8>,
    /// One byte per match: log2-based offset code (0-31).
    pub offset_codes: Vec<u8>,
    /// Packed extra bits for offsets (LSB-first bitstream).
    pub offset_extra: Vec<u8>,
    /// One byte per match: log2-based length code (0-20).
    pub length_codes: Vec<u8>,
    /// Packed extra bits for lengths (LSB-first bitstream).
    pub length_extra: Vec<u8>,
    /// Total number of tokens (literal + match).
    pub num_tokens: u32,
    /// Number of match tokens.
    pub num_matches: u32,
}

// ---------------------------------------------------------------------------
// Code tables
// ---------------------------------------------------------------------------

/// Encode a 1-based positive integer to (code, extra_bits_count, extra_value).
///
/// Code 0: value 1 (0 extra bits)
/// Code 1: value 2 (0 extra bits)
/// Code N (N>=2): base = 1 + 2^(N-1), extra_bits = N-1
#[inline]
pub(crate) fn encode_value(value: u32) -> (u8, u8, u32) {
    debug_assert!(value >= 1);
    match value {
        1 => (0, 0, 0),
        2 => (1, 0, 0),
        v => {
            let code = 32 - (v - 1).leading_zeros(); // = floor(log2(v-1)) + 1
            let extra_bits = code - 1;
            let base = 1 + (1u32 << (code - 1));
            let extra_value = v - base;
            (code as u8, extra_bits as u8, extra_value)
        }
    }
}

/// Decode from (code, extra_value) back to 1-based value.
#[inline]
pub(crate) fn decode_value(code: u8, extra_value: u32) -> u32 {
    match code {
        0 => 1,
        1 => 2,
        _ => {
            let base = 1 + (1u32 << (code as u32 - 1));
            base + extra_value
        }
    }
}

/// Number of extra bits for a given code.
#[inline]
pub(crate) fn extra_bits_for_code(code: u8) -> u8 {
    if code < 2 {
        0
    } else {
        code - 1
    }
}

/// Encode an offset (1-based distance) to (code, extra_bits, extra_value).
/// This is the raw (non-repeat) encoding used by the cost model.
#[inline]
pub(crate) fn encode_offset(offset: u32) -> (u8, u8, u32) {
    encode_value(offset)
}

/// Decode an offset from a raw (non-repeat) code + extra_value.
/// Used in tests and by the cost model; the decoder uses `RepeatOffsets::decode_offset`.
#[cfg(test)]
#[inline]
pub(crate) fn decode_offset(code: u8, extra_value: u32) -> u32 {
    decode_value(code, extra_value)
}

/// Encode a match length to (code, extra_bits, extra_value).
/// Applies MIN_MATCH bias: length 3 → value 1, length 4 → value 2, etc.
#[inline]
pub(crate) fn encode_length(length: u16) -> (u8, u8, u32) {
    let adj = (length - MIN_MATCH) as u32 + 1;
    encode_value(adj)
}

/// Decode a match length from code + extra_value.
#[inline]
pub(crate) fn decode_length(code: u8, extra_value: u32) -> u16 {
    let adj = decode_value(code, extra_value);
    (adj - 1 + MIN_MATCH as u32) as u16
}

// ---------------------------------------------------------------------------
// Repeat offsets
// ---------------------------------------------------------------------------

/// Number of reserved repeat offset codes (0, 1, 2).
pub(crate) const NUM_REPEAT_CODES: u8 = 3;

/// Tracks the 3 most recently used offsets for repeat-offset encoding.
///
/// Encoder and decoder maintain identical state. Matches that reuse a
/// recent offset encode with code 0-2 (0 extra bits), saving the full
/// offset encoding cost.
pub(crate) struct RepeatOffsets {
    pub(crate) recent: [u32; 3],
}

impl RepeatOffsets {
    pub(crate) fn new() -> Self {
        // Initialize with common small offsets. Encoder and decoder must match.
        RepeatOffsets { recent: [1, 1, 1] }
    }

    /// Encode an offset using repeat codes. Returns (code, extra_bits, extra_value).
    ///
    /// Codes 0-2: repeat offset (0 extra bits).
    /// Code 3+: literal offset (shifted from base table).
    #[inline]
    pub(crate) fn encode_offset(&mut self, offset: u32) -> (u8, u8, u32) {
        // Check repeat offsets (cheapest encoding: 0 extra bits)
        for i in 0..3 {
            if offset == self.recent[i] {
                self.promote(i);
                return (i as u8, 0, 0);
            }
        }
        // Literal offset: shift code by NUM_REPEAT_CODES
        let (code, eb, ev) = encode_value(offset);
        self.push_new(offset);
        (code + NUM_REPEAT_CODES, eb, ev)
    }

    /// Decode an offset from code + extra_value, updating repeat state.
    #[inline]
    pub(crate) fn decode_offset(&mut self, code: u8, extra_value: u32) -> u32 {
        if code < NUM_REPEAT_CODES {
            let offset = self.recent[code as usize];
            self.promote(code as usize);
            offset
        } else {
            let offset = decode_value(code - NUM_REPEAT_CODES, extra_value);
            self.push_new(offset);
            offset
        }
    }

    /// Promote repeat index `i` to most-recent position.
    #[inline]
    pub(crate) fn promote(&mut self, i: usize) {
        match i {
            0 => {}                           // already most recent
            1 => self.recent.swap(0, 1),      // swap 1↔0
            2 => self.recent.rotate_right(1), // [2,0,1]
            _ => unreachable!(),
        }
    }

    /// Push a new (non-repeat) offset, evicting the oldest.
    #[inline]
    pub(crate) fn push_new(&mut self, offset: u32) {
        self.recent[2] = self.recent[1];
        self.recent[1] = self.recent[0];
        self.recent[0] = offset;
    }
}

/// Number of extra bits for a repeat-aware offset code.
#[inline]
pub(crate) fn extra_bits_for_offset_code(code: u8) -> u8 {
    if code < NUM_REPEAT_CODES {
        0
    } else {
        extra_bits_for_code(code - NUM_REPEAT_CODES)
    }
}

/// Check how long a match extends at `pos` with the given offset.
/// Returns 0 if no valid match (offset too large or no bytes match).
#[inline]
fn check_repeat_match(input: &[u8], pos: usize, offset: u32, max_match: usize) -> u16 {
    if offset == 0 || offset as usize > pos {
        return 0;
    }
    let max_len = (input.len() - pos).min(max_match);
    let mut src_idx = pos - offset as usize;
    let mut dst_idx = pos;
    let end = pos + max_len;
    while dst_idx < end && input[src_idx] == input[dst_idx] {
        src_idx += 1;
        dst_idx += 1;
    }
    (dst_idx - pos) as u16
}

// ---------------------------------------------------------------------------
// BitWriter / BitReader for extra-bits streams (LSB-first, u64 container)
// ---------------------------------------------------------------------------

pub(crate) struct BitWriter {
    buffer: Vec<u8>,
    container: u64,
    bit_pos: u32,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        BitWriter {
            buffer: Vec::new(),
            container: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    pub(crate) fn write_bits(&mut self, value: u32, nb_bits: u8) {
        debug_assert!(nb_bits <= 32);
        if nb_bits == 0 {
            return;
        }
        self.container |= (value as u64) << self.bit_pos;
        self.bit_pos += nb_bits as u32;
        while self.bit_pos >= 8 {
            self.buffer.push(self.container as u8);
            self.container >>= 8;
            self.bit_pos -= 8;
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buffer.push(self.container as u8);
        }
        self.buffer
    }
}

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    container: u64,
    bits_available: u32,
    /// Total bits consumed so far (for overflow detection).
    bits_consumed: u64,
    /// Total bits available in the underlying data.
    total_bits: u64,
    /// Set if a read consumed more bits than available in the data.
    pub overflow: bool,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        let mut r = BitReader {
            data,
            byte_pos: 0,
            container: 0,
            bits_available: 0,
            bits_consumed: 0,
            total_bits: data.len() as u64 * 8,
            overflow: false,
        };
        r.refill();
        r
    }

    #[inline]
    pub(crate) fn read_bits(&mut self, nb_bits: u8) -> u32 {
        if nb_bits == 0 {
            return 0;
        }
        self.refill();
        let mask = (1u64 << nb_bits) - 1;
        let value = (self.container & mask) as u32;
        self.container >>= nb_bits;
        self.bits_available = self.bits_available.saturating_sub(nb_bits as u32);
        self.bits_consumed += nb_bits as u64;
        if self.bits_consumed > self.total_bits {
            self.overflow = true;
        }
        value
    }

    fn refill(&mut self) {
        // Fast path: bulk-load when at least 8 bytes remain in buffer.
        // Loads a full u64, masks to only the bytes that fit without
        // overflow, then shifts into position.
        if self.bits_available <= 56 && self.byte_pos < self.data.len() {
            if self.byte_pos + 8 <= self.data.len() {
                let chunk = u64::from_le_bytes(
                    self.data[self.byte_pos..self.byte_pos + 8]
                        .try_into()
                        .unwrap(),
                );
                // Only load whole bytes that fit (bits_available may not be
                // byte-aligned after reading variable-width extra bits).
                let full_bytes = ((64 - self.bits_available) / 8) as usize;
                let keep_mask = if full_bytes >= 8 {
                    u64::MAX
                } else {
                    (1u64 << (full_bytes * 8)) - 1
                };
                self.container |= (chunk & keep_mask) << self.bits_available;
                self.byte_pos += full_bytes;
                self.bits_available += full_bytes as u32 * 8;
            } else {
                // Tail: fewer than 8 bytes remain, load byte-by-byte.
                while self.bits_available <= 56 && self.byte_pos < self.data.len() {
                    self.container |= (self.data[self.byte_pos] as u64) << self.bits_available;
                    self.byte_pos += 1;
                    self.bits_available += 8;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flag packing (MSB-first)
// ---------------------------------------------------------------------------

/// Pack boolean flags into bytes, MSB-first.
fn pack_flags(flags: &[bool]) -> Vec<u8> {
    let num_bytes = flags.len().div_ceil(8);
    let mut bytes = vec![0u8; num_bytes];
    for (i, &flag) in flags.iter().enumerate() {
        if flag {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }
    bytes
}

/// Unpack boolean flags from bytes, MSB-first.
///
/// # Panics (debug only)
/// Panics if `bytes` is too short to hold `count` flags.
/// Callers must validate length before calling.
#[cfg(test)]
fn unpack_flags(bytes: &[u8], count: usize) -> Vec<bool> {
    debug_assert!(
        bytes.len() >= count.div_ceil(8),
        "unpack_flags: need {} bytes for {} flags, got {}",
        count.div_ceil(8),
        count,
        bytes.len()
    );
    (0..count)
        .map(|i| bytes[i / 8] & (1 << (7 - (i % 8))) != 0)
        .collect()
}

// ---------------------------------------------------------------------------
// Distance-dependent minimum match length
// ---------------------------------------------------------------------------

/// Minimum profitable match length at a given offset.
///
/// Short matches at large distances aren't worth encoding because the
/// code+extra-bits cost exceeds the literal cost. This function returns
/// the minimum match length that's worth emitting for a given offset.
///
/// Exact cost model (match cost in bytes vs literal savings):
/// - Each literal: 1 byte (flag bit + literal byte, entropy coded)
/// - Each match overhead: 2 bytes (offset_code + length_code, entropy coded)
///   + extra_bits / 8 bytes (raw extra bits for offset + length)
///
/// A match of length L saves L literal bytes and costs match overhead + extra bits.
/// Break-even: L >= ceil(2 + oeb/8). Close offsets have few extra bits (cheap),
/// far offsets have many extra bits (expensive).
///
/// Exact thresholds per offset bit-cost (oeb = extra bits for offset code):
/// | oeb | Offset range       | Match cost | Min profitable L |
/// |-----|--------------------|-----------| ------------------|
/// | 0   | 1-2                | 2 bytes   | 3 (= MIN_MATCH)   |
/// | 1   | 3-4                | 2 bytes   | 3                 |
/// | 2   | 5-8                | 2 bytes   | 3                 |
/// | 3   | 9-16               | 2 bytes   | 3                 |
/// | 4   | 17-32              | 3 bytes   | 4                 |
/// | 5   | 33-64              | 3 bytes   | 4                 |
/// | 6   | 65-128             | 3 bytes   | 4                 |
/// | 7   | 129-256            | 3 bytes   | 4                 |
/// | 8   | 257-512            | 4 bytes   | 5                 |
/// | 9   | 513-1024           | 4 bytes   | 5                 |
/// | 10  | 1025-2048          | 4 bytes   | 5                 |
/// | 11  | 2049-4096          | 4 bytes   | 5                 |
/// | 12  | 4097-8192          | 5 bytes   | 6                 |
/// | 13  | 8193-16384         | 5 bytes   | 6                 |
/// | 14  | 16385-32768        | 5 bytes   | 6                 |
/// | 15  | 32769-65536        | 5 bytes   | 6                 |
/// | 16  | 65537-131072       | 6 bytes   | 7                 |
#[inline]
pub(crate) fn min_profitable_length(offset: u32) -> u16 {
    if offset == 0 {
        return u16::MAX; // No valid match
    }
    let (oc, _, _) = encode_offset(offset);
    let oeb = extra_bits_for_code(oc);
    // Exact cost model based on offset extra bits:
    //   oeb 0-3  (offset 1-16):        min 3 (= MIN_MATCH)
    //   oeb 4-7  (offset 17-256):      min 4
    //   oeb 8-11 (offset 257-4096):    min 5
    //   oeb 12-15 (offset 4097-65536): min 6
    //   oeb 16+  (offset 65537+):      min 7
    MIN_MATCH
        + match oeb {
            0..=3 => 0,
            4..=7 => 1,
            8..=11 => 2,
            12..=15 => 3,
            _ => 4,
        }
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Compress input using LzSeq with lazy matching (32KB window).
///
/// Uses the same HashChainFinder and lazy matching strategy as LZSS,
/// but encodes matches with log2-based codes + extra bits instead of
/// fixed-width offset:u16 + length:u16.
///
/// For wider windows (128KB+), use `encode_with_config`.
pub fn encode(input: &[u8]) -> PzResult<SeqEncoded> {
    encode_with_config(input, &SeqConfig::default())
}

/// Select the best match considering both hash-chain and repeat-offset candidates.
///
/// Returns (offset, length, is_repeat) of the best match. Prefers repeat-offset
/// matches when they're competitive with the hash-chain match, because repeat
/// offsets encode with 0 extra bits for the offset component.
fn select_best_match(
    input: &[u8],
    pos: usize,
    hash_offset: u32,
    hash_length: u16,
    repeats: &RepeatOffsets,
    max_match_len: usize,
) -> (u32, u16, bool) {
    // Find the best repeat-offset match. Deduplicate comparisons when the
    // repeat set contains identical offsets (common in early stream positions).
    let [rep0, rep1, rep2] = repeats.recent;
    let len0 = check_repeat_match(input, pos, rep0, max_match_len);
    let len1 = if rep1 == rep0 {
        len0
    } else {
        check_repeat_match(input, pos, rep1, max_match_len)
    };
    let len2 = if rep2 == rep0 {
        len0
    } else if rep2 == rep1 {
        len1
    } else {
        check_repeat_match(input, pos, rep2, max_match_len)
    };

    let mut best_rep_offset = rep0;
    let mut best_rep_len = len0;
    if len1 > best_rep_len {
        best_rep_len = len1;
        best_rep_offset = rep1;
    }
    if len2 > best_rep_len {
        best_rep_len = len2;
        best_rep_offset = rep2;
    }

    // Decide: hash-chain vs repeat
    if best_rep_len >= MIN_MATCH {
        if hash_length < MIN_MATCH {
            // No hash match — use repeat
            return (best_rep_offset, best_rep_len, true);
        }
        // Repeat saves the full offset encoding cost (~1 code byte + extra bits).
        // Accept a repeat match that's shorter by up to the offset savings.
        let (_, oeb, _) = encode_offset(hash_offset);
        let offset_savings = 1 + (oeb as u16) / 8; // bytes saved by repeat
        if best_rep_len.saturating_add(offset_savings) >= hash_length {
            return (best_rep_offset, best_rep_len, true);
        }
    }

    // Use hash-chain match (or no match if hash_length < MIN_MATCH)
    (hash_offset, hash_length, false)
}

/// Emit a match token into the output streams.
#[allow(clippy::too_many_arguments)]
fn emit_match(
    offset: u32,
    length: u16,
    repeats: &mut RepeatOffsets,
    flags_vec: &mut Vec<bool>,
    offset_codes: &mut Vec<u8>,
    offset_extra_writer: &mut BitWriter,
    length_codes: &mut Vec<u8>,
    length_extra_writer: &mut BitWriter,
) {
    flags_vec.push(false);
    let (oc, oeb, oev) = repeats.encode_offset(offset);
    offset_codes.push(oc);
    offset_extra_writer.write_bits(oev, oeb);
    let (lc, leb, lev) = encode_length(length);
    length_codes.push(lc);
    length_extra_writer.write_bits(lev, leb);
}

/// Encode a universal `LzToken` stream into LzSeq's 6-stream format.
///
/// Like `encode_match_sequence` but takes `LzToken` directly instead of
/// `lz77::Match`. Used by the `LzSeqEncoder` wire encoder.
pub(crate) fn encode_from_tokens(
    tokens: &[crate::lz_token::LzToken],
    config: &SeqConfig,
) -> PzResult<SeqEncoded> {
    use crate::lz_token::LzToken;

    let max_len = config.max_match_len;
    let mut repeats = RepeatOffsets::new();
    let mut flags_vec: Vec<bool> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut offset_codes: Vec<u8> = Vec::new();
    let mut length_codes: Vec<u8> = Vec::new();
    let mut offset_extra_writer = BitWriter::new();
    let mut length_extra_writer = BitWriter::new();

    for token in tokens {
        match token {
            LzToken::Literal(b) => {
                flags_vec.push(true);
                literals.push(*b);
            }
            LzToken::Match { offset, length } => {
                emit_match(
                    *offset,
                    (*length).min(max_len as u32) as u16,
                    &mut repeats,
                    &mut flags_vec,
                    &mut offset_codes,
                    &mut offset_extra_writer,
                    &mut length_codes,
                    &mut length_extra_writer,
                );
            }
        }
    }

    let num_tokens = flags_vec.len() as u32;
    let num_matches = offset_codes.len() as u32;
    let flags = pack_flags(&flags_vec);

    Ok(SeqEncoded {
        flags,
        literals,
        offset_codes,
        offset_extra: offset_extra_writer.finish(),
        length_codes,
        length_extra: length_extra_writer.finish(),
        num_tokens,
        num_matches,
    })
}

/// Compress input using LzSeq with optimal parsing.
///
/// Uses backward DP (`optimal.rs`) to select matches, then encodes them with
/// the same LzSeq token format as `encode_with_config`. The DP cost model
/// accounts for repeat offset savings, so it selects matches that set up
/// future cheap repeat encodings.
///
/// Slower than `encode_with_config` (lazy) but produces better ratios.
/// Used by LzSeqR quality mode.
pub fn encode_optimal(input: &[u8], config: &SeqConfig) -> PzResult<SeqEncoded> {
    if input.is_empty() {
        return Ok(SeqEncoded {
            flags: Vec::new(),
            literals: Vec::new(),
            offset_codes: Vec::new(),
            offset_extra: Vec::new(),
            length_codes: Vec::new(),
            length_extra: Vec::new(),
            num_tokens: 0,
            num_matches: 0,
        });
    }

    // Build match table and run repeat-offset-aware optimal parse.
    // Use LZ77_MAX_MATCH (258) for reasonable performance.
    // Searching for extremely long matches (u16::MAX) is prohibitively slow.
    let max_match = crate::lz77::LZ77_MAX_MATCH;
    let table = crate::optimal::build_match_table_cpu_with_config(
        input,
        crate::optimal::K,
        max_match,
        config.max_window,
        config.max_chain,
    );
    let matches = crate::optimal::optimal_parse_lzseq(input, &table)?;

    // Convert optimal parse output to universal tokens, then encode.
    let tokens = crate::lz_token::matches_to_tokens(&matches);
    encode_from_tokens(&tokens, config)
}

/// Compress input using LzSeq with lazy matching and configurable window.
///
/// Uses `find_match_wide` to support u32 offsets for windows larger than 32KB.
/// The offset code table naturally handles any offset up to ~1MB (code 20).
pub fn encode_with_config(input: &[u8], config: &SeqConfig) -> PzResult<SeqEncoded> {
    if input.is_empty() {
        return Ok(SeqEncoded {
            flags: Vec::new(),
            literals: Vec::new(),
            offset_codes: Vec::new(),
            offset_extra: Vec::new(),
            length_codes: Vec::new(),
            length_extra: Vec::new(),
            num_tokens: 0,
            num_matches: 0,
        });
    }

    let match_limit = config.max_match_len;
    let mut finder = if config.hash_prefix_len == 4 {
        HashChainFinder::with_hash4(config.max_window, match_limit, config.max_chain)
    } else {
        HashChainFinder::with_window_and_chain(config.max_window, match_limit, config.max_chain)
    };
    let mut repeats = RepeatOffsets::new();
    let mut flags_vec: Vec<bool> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut offset_codes: Vec<u8> = Vec::new();
    let mut length_codes: Vec<u8> = Vec::new();
    let mut offset_extra_writer = BitWriter::new();
    let mut length_extra_writer = BitWriter::new();
    let mut pos: usize = 0;
    let max_match_len = match_limit as usize;

    // Adaptive chain depth tracking
    let base_chain = config.max_chain;
    let mut adapt_pos_counter: usize = 0; // positions since last check
    let mut adapt_match_counter: usize = 0; // matches found in current window
    let adapt_check_interval: usize = 64; // check every 64 positions
    let adapt_low_threshold: usize = 2; // fewer than 2 matches in 64 = low compressibility
    let adapt_penalty_positions: usize = 256; // how long to use reduced chain
    let mut adapt_penalty_remaining: usize = 0;

    while pos < input.len() {
        if config.adaptive_chain {
            adapt_pos_counter += 1;
            if adapt_penalty_remaining > 0 {
                adapt_penalty_remaining -= 1;
                if adapt_penalty_remaining == 0 {
                    finder.set_max_chain(base_chain);
                }
            }
            if adapt_pos_counter >= adapt_check_interval {
                adapt_pos_counter = 0;
                if adapt_match_counter < adapt_low_threshold {
                    // Low compressibility: halve chain depth for next window
                    finder.set_max_chain((base_chain / 2).max(1));
                    adapt_penalty_remaining = adapt_penalty_positions;
                } else {
                    // Restore full chain depth
                    finder.set_max_chain(base_chain);
                }
                adapt_match_counter = 0;
            }
        }

        let m = finder.find_match_wide(input, pos);
        finder.insert(input, pos);

        // Check repeat offsets: matches at recent offsets encode with 0 extra
        // bits for the offset, making them much cheaper than hash-chain matches.
        let (best_offset, best_length, is_repeat) =
            select_best_match(input, pos, m.offset, m.length, &repeats, max_match_len);

        // Distance-dependent minimum match length: reject short matches
        // at large distances where code+extra-bits cost exceeds literal cost.
        // Repeat matches always use MIN_MATCH (they're essentially free).
        let effective_min = if best_length >= MIN_MATCH && !is_repeat {
            min_profitable_length(best_offset)
        } else {
            MIN_MATCH
        };

        // Lazy matching: check if next position has a longer match
        if best_length >= effective_min
            && best_length < LAZY_SKIP_THRESHOLD
            && pos + 1 < input.len()
        {
            finder.insert(input, pos + 1);
            let next_m = finder.find_match_wide(input, pos + 1);
            let (next_offset, next_length, next_is_repeat) = select_best_match(
                input,
                pos + 1,
                next_m.offset,
                next_m.length,
                &repeats,
                max_match_len,
            );
            let next_effective_min = if next_length >= MIN_MATCH && !next_is_repeat {
                if next_offset > 0 {
                    min_profitable_length(next_offset)
                } else {
                    u16::MAX
                }
            } else {
                MIN_MATCH
            };

            if next_length >= next_effective_min && next_length > best_length {
                // Emit literal for current position, use the better match
                flags_vec.push(true);
                literals.push(input[pos]);
                pos += 1;

                // Emit match from next position
                emit_match(
                    next_offset,
                    next_length,
                    &mut repeats,
                    &mut flags_vec,
                    &mut offset_codes,
                    &mut offset_extra_writer,
                    &mut length_codes,
                    &mut length_extra_writer,
                );

                if config.adaptive_chain {
                    adapt_match_counter += 1;
                }

                // Insert covered positions into hash chains
                let advance = next_length as usize;
                let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
                for i in 1..insert_count {
                    finder.insert(input, pos + i);
                }
                pos += advance;
                continue;
            }
        }

        if best_length >= effective_min {
            // Emit match
            emit_match(
                best_offset,
                best_length,
                &mut repeats,
                &mut flags_vec,
                &mut offset_codes,
                &mut offset_extra_writer,
                &mut length_codes,
                &mut length_extra_writer,
            );

            if config.adaptive_chain {
                adapt_match_counter += 1;
            }

            let advance = best_length as usize;
            let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
            for i in 1..insert_count {
                finder.insert(input, pos + i);
            }
            pos += advance;
        } else {
            // Emit literal
            flags_vec.push(true);
            literals.push(input[pos]);
            pos += 1;
        }
    }

    let num_tokens = flags_vec.len() as u32;
    let num_matches = offset_codes.len() as u32;
    let flags = pack_flags(&flags_vec);

    Ok(SeqEncoded {
        flags,
        literals,
        offset_codes,
        offset_extra: offset_extra_writer.finish(),
        length_codes,
        length_extra: length_extra_writer.finish(),
        num_tokens,
        num_matches,
    })
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Decompress LzSeq-encoded data from 6 separate streams.
#[allow(clippy::too_many_arguments)]
pub fn decode(
    flags: &[u8],
    literals: &[u8],
    offset_codes: &[u8],
    offset_extra: &[u8],
    length_codes: &[u8],
    length_extra: &[u8],
    num_tokens: u32,
    num_matches: u32,
    original_len: usize,
) -> PzResult<Vec<u8>> {
    if num_tokens == 0 {
        return Ok(Vec::new());
    }

    // Validate stream lengths to avoid panics on malformed input.
    let num_tokens = num_tokens as usize;
    let num_matches = num_matches as usize;
    let required_flag_bytes = num_tokens.div_ceil(8);
    if flags.len() < required_flag_bytes {
        return Err(PzError::InvalidInput);
    }
    if offset_codes.len() < num_matches || length_codes.len() < num_matches {
        return Err(PzError::InvalidInput);
    }

    let mut lit_pos = 0usize;
    let mut match_idx = 0usize;
    let mut off_extra_reader = BitReader::new(offset_extra);
    let mut len_extra_reader = BitReader::new(length_extra);
    let mut repeats = RepeatOffsets::new();
    let mut output = Vec::with_capacity(original_len);

    // Iterate packed flag bytes directly instead of allocating a Vec<bool>.
    let mut flag_byte_idx = 0usize;
    let mut flag_bit_mask = 0x80u8;
    let mut token_idx = 0usize;

    while token_idx < num_tokens {
        let is_literal = flags[flag_byte_idx] & flag_bit_mask != 0;
        flag_bit_mask >>= 1;
        if flag_bit_mask == 0 {
            flag_bit_mask = 0x80;
            flag_byte_idx += 1;
        }
        token_idx += 1;

        if is_literal {
            if lit_pos >= literals.len() {
                return Err(PzError::InvalidInput);
            }
            output.push(literals[lit_pos]);
            lit_pos += 1;
        } else {
            if match_idx >= num_matches {
                return Err(PzError::InvalidInput);
            }
            let oc = offset_codes[match_idx];
            let oeb = extra_bits_for_offset_code(oc);
            let oev = off_extra_reader.read_bits(oeb);
            let offset = repeats.decode_offset(oc, oev) as usize;

            let lc = length_codes[match_idx];
            let leb = extra_bits_for_code(lc);
            let lev = len_extra_reader.read_bits(leb);
            let length = decode_length(lc, lev) as usize;

            match_idx += 1;

            // Check for truncated extra-bits streams.
            if off_extra_reader.overflow || len_extra_reader.overflow {
                return Err(PzError::InvalidInput);
            }

            if offset == 0 || offset > output.len() {
                return Err(PzError::InvalidInput);
            }
            if output.len() + length > original_len {
                return Err(PzError::InvalidInput);
            }

            let copy_start = output.len() - offset;
            if offset >= length {
                // Non-overlapping: one bulk copy (compiles to memcpy).
                output.extend_from_within(copy_start..copy_start + length);
            } else {
                // Overlapping: copy in offset-sized chunks to amortize.
                let mut remaining = length;
                while remaining > 0 {
                    let chunk = remaining.min(offset);
                    let start = output.len() - offset;
                    output.extend_from_within(start..start + chunk);
                    remaining -= chunk;
                }
            }
        }
    }

    if output.len() != original_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

#[cfg(test)]
mod tests;
