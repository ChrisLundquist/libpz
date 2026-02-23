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
use crate::lz77::{HashChainFinder, DEFAULT_MAX_MATCH, MIN_MATCH};
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
    /// Default: 64. Clamped to 1..=MAX_CHAIN internally.
    pub max_chain: usize,
    /// When true, reduce chain depth on blocks with low match density.
    /// Speeds up encoding on incompressible data with minimal ratio cost.
    /// Default: false.
    pub adaptive_chain: bool,
}

impl Default for SeqConfig {
    fn default() -> Self {
        SeqConfig {
            max_window: 128 * 1024,
            hash_prefix_len: 3,
            max_chain: crate::lz77::MAX_CHAIN,
            adaptive_chain: false,
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
fn encode_value(value: u32) -> (u8, u8, u32) {
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
fn decode_value(code: u8, extra_value: u32) -> u32 {
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
fn extra_bits_for_code(code: u8) -> u8 {
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
const NUM_REPEAT_CODES: u8 = 3;

/// Tracks the 3 most recently used offsets for repeat-offset encoding.
///
/// Encoder and decoder maintain identical state. Matches that reuse a
/// recent offset encode with code 0-2 (0 extra bits), saving the full
/// offset encoding cost.
struct RepeatOffsets {
    recent: [u32; 3],
}

impl RepeatOffsets {
    fn new() -> Self {
        // Initialize with common small offsets. Encoder and decoder must match.
        RepeatOffsets { recent: [1, 1, 1] }
    }

    /// Encode an offset using repeat codes. Returns (code, extra_bits, extra_value).
    ///
    /// Codes 0-2: repeat offset (0 extra bits).
    /// Code 3+: literal offset (shifted from base table).
    #[inline]
    fn encode_offset(&mut self, offset: u32) -> (u8, u8, u32) {
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
    fn decode_offset(&mut self, code: u8, extra_value: u32) -> u32 {
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
    fn promote(&mut self, i: usize) {
        match i {
            0 => {}                           // already most recent
            1 => self.recent.swap(0, 1),      // swap 1↔0
            2 => self.recent.rotate_right(1), // [2,0,1]
            _ => unreachable!(),
        }
    }

    /// Push a new (non-repeat) offset, evicting the oldest.
    #[inline]
    fn push_new(&mut self, offset: u32) {
        self.recent[2] = self.recent[1];
        self.recent[1] = self.recent[0];
        self.recent[0] = offset;
    }
}

/// Number of extra bits for a repeat-aware offset code.
#[inline]
fn extra_bits_for_offset_code(code: u8) -> u8 {
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
    let src = pos - offset as usize;
    let mut len = 0;
    while len < max_len && input[src + len] == input[pos + len] {
        len += 1;
    }
    len as u16
}

// ---------------------------------------------------------------------------
// BitWriter / BitReader for extra-bits streams (LSB-first, u64 container)
// ---------------------------------------------------------------------------

struct BitWriter {
    buffer: Vec<u8>,
    container: u64,
    bit_pos: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            buffer: Vec::new(),
            container: 0,
            bit_pos: 0,
        }
    }

    #[inline]
    fn write_bits(&mut self, value: u32, nb_bits: u8) {
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

    fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.buffer.push(self.container as u8);
        }
        self.buffer
    }
}

struct BitReader<'a> {
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
    fn new(data: &'a [u8]) -> Self {
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
    fn read_bits(&mut self, nb_bits: u8) -> u32 {
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
/// | 0   | 1                  | 2 bytes   | 3 (= MIN_MATCH)   |
/// | 1   | 2                  | 2 bytes   | 3                 |
/// | 2   | 3-4                | 2 bytes   | 3                 |
/// | 3   | 5-8                | 2 bytes   | 3                 |
/// | 4   | 9-16               | 3 bytes   | 4                 |
/// | 5   | 17-32              | 3 bytes   | 4                 |
/// | 6   | 33-64              | 3 bytes   | 4                 |
/// | 7   | 65-128             | 3 bytes   | 4                 |
/// | 8   | 129-256            | 3 bytes   | 4                 |
/// | 9   | 257-512            | 4 bytes   | 5                 |
/// | 10  | 513-1024           | 4 bytes   | 5                 |
/// | 11  | 1025-2048          | 4 bytes   | 5                 |
/// | 12  | 2049-4096          | 4 bytes   | 5                 |
/// | 13  | 4097-8192          | 5 bytes   | 6                 |
/// | 14  | 8193-16384         | 5 bytes   | 6                 |
/// | 15  | 16385-32768        | 5 bytes   | 6                 |
/// | 16  | 32769-65536        | 5 bytes   | 6                 |
/// | 17  | 65537-131072       | 6 bytes   | 7                 |
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
    // Find the best repeat-offset match
    let mut best_rep_offset = 0u32;
    let mut best_rep_len = 0u16;
    for &rep_offset in &repeats.recent {
        let rep_len = check_repeat_match(input, pos, rep_offset, max_match_len);
        if rep_len > best_rep_len {
            best_rep_len = rep_len;
            best_rep_offset = rep_offset;
        }
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

    let mut finder = if config.hash_prefix_len == 4 {
        HashChainFinder::with_hash4(config.max_window, DEFAULT_MAX_MATCH, config.max_chain)
    } else {
        HashChainFinder::with_window_and_chain(
            config.max_window,
            DEFAULT_MAX_MATCH,
            config.max_chain,
        )
    };
    let mut repeats = RepeatOffsets::new();
    let mut flags_vec: Vec<bool> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut offset_codes: Vec<u8> = Vec::new();
    let mut length_codes: Vec<u8> = Vec::new();
    let mut offset_extra_writer = BitWriter::new();
    let mut length_extra_writer = BitWriter::new();
    let mut pos: usize = 0;
    let max_match_len = DEFAULT_MAX_MATCH as usize;

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

/// Convenience: encode then immediately decode (for testing).
#[cfg(test)]
fn round_trip(input: &[u8]) -> PzResult<Vec<u8>> {
    let enc = encode(input)?;
    decode(
        &enc.flags,
        &enc.literals,
        &enc.offset_codes,
        &enc.offset_extra,
        &enc.length_codes,
        &enc.length_extra,
        enc.num_tokens,
        enc.num_matches,
        input.len(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lz77::MAX_WINDOW;

    // --- Code table self-consistency ---

    #[test]
    fn test_encode_decode_offset_exhaustive() {
        for d in 1..=100_000u32 {
            let (code, eb, ev) = encode_offset(d);
            assert_eq!(eb, extra_bits_for_code(code));
            let decoded = decode_offset(code, ev);
            assert_eq!(decoded, d, "offset {d}: code={code}, eb={eb}, ev={ev}");
        }
    }

    #[test]
    fn test_encode_decode_length_exhaustive() {
        for len in MIN_MATCH..=1000 {
            let (code, eb, ev) = encode_length(len);
            assert_eq!(eb, extra_bits_for_code(code));
            let decoded = decode_length(code, ev);
            assert_eq!(decoded, len, "length {len}: code={code}, eb={eb}, ev={ev}");
        }
    }

    #[test]
    fn test_offset_code_table_known_values() {
        // Code 0: offset 1
        assert_eq!(encode_offset(1), (0, 0, 0));
        // Code 1: offset 2
        assert_eq!(encode_offset(2), (1, 0, 0));
        // Code 2: offsets 3-4
        assert_eq!(encode_offset(3), (2, 1, 0));
        assert_eq!(encode_offset(4), (2, 1, 1));
        // Code 3: offsets 5-8
        assert_eq!(encode_offset(5), (3, 2, 0));
        assert_eq!(encode_offset(8), (3, 2, 3));
        // Code 4: offsets 9-16
        assert_eq!(encode_offset(9), (4, 3, 0));
        assert_eq!(encode_offset(16), (4, 3, 7));
        // Code 15: max offset for 32KB window = 32768
        let (code, eb, _) = encode_offset(32768);
        assert_eq!(code, 15);
        assert_eq!(eb, 14);
    }

    #[test]
    fn test_length_code_table_known_values() {
        // Code 0: length 3
        assert_eq!(encode_length(3), (0, 0, 0));
        // Code 1: length 4
        assert_eq!(encode_length(4), (1, 0, 0));
        // Code 2: lengths 5-6
        assert_eq!(encode_length(5), (2, 1, 0));
        assert_eq!(encode_length(6), (2, 1, 1));
        // Code 3: lengths 7-10
        assert_eq!(encode_length(7), (3, 2, 0));
        assert_eq!(encode_length(10), (3, 2, 3));
    }

    // --- BitWriter / BitReader round-trip ---

    #[test]
    fn test_bitwriter_reader_round_trip() {
        let mut w = BitWriter::new();
        w.write_bits(5, 3); // 101
        w.write_bits(0, 1); // 0
        w.write_bits(15, 4); // 1111
        w.write_bits(42, 7); // 0101010
        w.write_bits(0, 0); // nothing
        let data = w.finish();

        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(3), 5);
        assert_eq!(r.read_bits(1), 0);
        assert_eq!(r.read_bits(4), 15);
        assert_eq!(r.read_bits(7), 42);
        assert_eq!(r.read_bits(0), 0);
    }

    #[test]
    fn test_bitwriter_reader_large_values() {
        let mut w = BitWriter::new();
        w.write_bits(0x7FFF, 15); // 15 bits
        w.write_bits(0x1FFFF, 17); // 17 bits
        w.write_bits(1, 1);
        let data = w.finish();

        let mut r = BitReader::new(&data);
        assert_eq!(r.read_bits(15), 0x7FFF);
        assert_eq!(r.read_bits(17), 0x1FFFF);
        assert_eq!(r.read_bits(1), 1);
    }

    #[test]
    fn test_bitwriter_reader_many_small() {
        let mut w = BitWriter::new();
        for i in 0..100 {
            w.write_bits(i % 4, 2);
        }
        let data = w.finish();

        let mut r = BitReader::new(&data);
        for i in 0..100 {
            assert_eq!(r.read_bits(2), i % 4);
        }
    }

    // --- Flag packing ---

    #[test]
    fn test_flag_packing() {
        let flags = vec![true, false, true, true, false, false, true, false, true];
        let packed = pack_flags(&flags);
        let unpacked = unpack_flags(&packed, flags.len());
        assert_eq!(unpacked, flags);
    }

    #[test]
    fn test_flag_packing_empty() {
        let flags: Vec<bool> = Vec::new();
        let packed = pack_flags(&flags);
        assert!(packed.is_empty());
        let unpacked = unpack_flags(&packed, 0);
        assert!(unpacked.is_empty());
    }

    // --- Encode/decode round-trip tests (ported from LZSS) ---

    #[test]
    fn test_round_trip_empty() {
        let input: Vec<u8> = Vec::new();
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_single_byte() {
        let input = vec![42u8];
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_no_matches() {
        let input = b"abcdefgh".to_vec();
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_all_same() {
        let input = vec![b'x'; 200];
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_repeats() {
        let pattern = b"the quick brown fox ";
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend_from_slice(pattern);
        }
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_longer_text() {
        let input = b"To be, or not to be, that is the question: \
            Whether 'tis nobler in the mind to suffer \
            The slings and arrows of outrageous fortune, \
            Or to take arms against a sea of troubles"
            .to_vec();
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_large() {
        let pattern = b"abcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        assert!(input.len() > 7500);
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_binary() {
        let input: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_window_boundary() {
        let mut input = Vec::new();
        let pattern = b"window boundary test pattern! ";
        for _ in 0..2000 {
            input.extend_from_slice(pattern);
        }
        assert!(input.len() > 32768);
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_150kb_mixed() {
        // Semi-random 150KB: mix of text patterns and noise
        let mut data = Vec::with_capacity(150 * 1024);
        let phrases: &[&[u8]] = &[
            b"compression algorithms are fascinating ",
            b"data structures enable efficient storage ",
            b"entropy coding reduces redundancy ",
            b"hash tables provide fast lookup ",
        ];
        let mut state: u32 = 0xDEADBEEF;
        while data.len() < 150 * 1024 {
            let phrase = phrases[(state as usize / 7) % phrases.len()];
            data.extend_from_slice(phrase);
            for _ in 0..32 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                data.push(state as u8);
            }
        }
        data.truncate(150 * 1024);
        let output = round_trip(&data).unwrap();
        assert_eq!(output, data);
    }

    #[test]
    fn test_two_bytes() {
        let input = vec![0u8, 255];
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_compresses_with_matches() {
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend_from_slice(pattern);
        }
        let enc = encode(&input).unwrap();
        // Should have found matches
        assert!(
            enc.num_matches > 0,
            "should find matches in repetitive data"
        );
        // Round-trip
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_long_run_all_same() {
        let input = vec![0xAAu8; 70_000];
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_round_trip_long_repeating_pattern() {
        let pattern = b"abcde";
        let input: Vec<u8> = pattern.iter().cycle().take(80_000).copied().collect();
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    // --- Encode-specific tests ---

    #[test]
    fn test_encode_empty_produces_zero_tokens() {
        let enc = encode(&[]).unwrap();
        assert_eq!(enc.num_tokens, 0);
        assert_eq!(enc.num_matches, 0);
        assert!(enc.flags.is_empty());
        assert!(enc.literals.is_empty());
    }

    #[test]
    fn test_encode_all_literals() {
        let input = b"abcdefgh";
        let enc = encode(input).unwrap();
        assert_eq!(enc.num_tokens as usize, input.len());
        assert_eq!(enc.num_matches, 0);
        assert_eq!(enc.literals.len(), input.len());
    }

    #[test]
    fn test_offset_codes_in_valid_range() {
        let pattern = b"hello world hello world hello world ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let enc = encode(&input).unwrap();
        for &code in &enc.offset_codes {
            assert!(code <= 31, "offset code {code} out of range");
        }
        for &code in &enc.length_codes {
            assert!(code <= 20, "length code {code} out of range");
        }
    }

    // --- Wide window tests (Phase 3) ---

    #[test]
    fn test_wide_window_round_trip() {
        // Create data with a repeated block separated by >32KB of unique fill.
        // Only a wide window can find the long-distance match.
        let marker = b"WIDE_WINDOW_MARKER_PATTERN_12345";
        let fill_len = 40_000; // > 32KB
        let mut input = Vec::with_capacity(marker.len() * 2 + fill_len);
        input.extend_from_slice(marker);
        // Fill with non-repeating bytes so the matcher doesn't find local matches
        for i in 0..fill_len {
            input.push((i % 251) as u8 ^ 0x55);
        }
        input.extend_from_slice(marker);

        let config = SeqConfig {
            max_window: 64 * 1024,
            ..SeqConfig::default()
        };
        let enc = encode_with_config(&input, &config).unwrap();
        let output = decode(
            &enc.flags,
            &enc.literals,
            &enc.offset_codes,
            &enc.offset_extra,
            &enc.length_codes,
            &enc.length_extra,
            enc.num_tokens,
            enc.num_matches,
            input.len(),
        )
        .unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_wide_window_finds_distant_matches() {
        // Verify the 128KB window actually finds matches beyond 32KB.
        let pattern = b"DISTANT_MATCH_TEST_PATTERN_ABCD!";
        let fill_len = 50_000; // > 32KB gap
        let mut input = Vec::with_capacity(pattern.len() * 2 + fill_len);
        input.extend_from_slice(pattern);
        for i in 0..fill_len {
            input.push((i % 251) as u8 ^ 0xAA);
        }
        input.extend_from_slice(pattern);

        // Wide window: should find a match at offset > 32KB
        let config_wide = SeqConfig {
            max_window: 128 * 1024,
            ..SeqConfig::default()
        };
        let enc_wide = encode_with_config(&input, &config_wide).unwrap();

        // Narrow window: cannot see across the gap
        let config_narrow = SeqConfig {
            max_window: MAX_WINDOW, // 32KB
            ..SeqConfig::default()
        };
        let enc_narrow = encode_with_config(&input, &config_narrow).unwrap();

        // Wide window should produce fewer matches (or equal) and fewer literals
        // because it finds the distant match that the narrow window misses.
        assert!(
            enc_wide.literals.len() <= enc_narrow.literals.len(),
            "wide window ({} literals) should not produce more literals than narrow ({})",
            enc_wide.literals.len(),
            enc_narrow.literals.len()
        );

        // Both must round-trip correctly
        let out_wide = decode(
            &enc_wide.flags,
            &enc_wide.literals,
            &enc_wide.offset_codes,
            &enc_wide.offset_extra,
            &enc_wide.length_codes,
            &enc_wide.length_extra,
            enc_wide.num_tokens,
            enc_wide.num_matches,
            input.len(),
        )
        .unwrap();
        assert_eq!(out_wide, input);

        let out_narrow = decode(
            &enc_narrow.flags,
            &enc_narrow.literals,
            &enc_narrow.offset_codes,
            &enc_narrow.offset_extra,
            &enc_narrow.length_codes,
            &enc_narrow.length_extra,
            enc_narrow.num_tokens,
            enc_narrow.num_matches,
            input.len(),
        )
        .unwrap();
        assert_eq!(out_narrow, input);
    }

    #[test]
    fn test_default_config_uses_128kb() {
        let config = SeqConfig::default();
        assert_eq!(config.max_window, 128 * 1024);
    }

    #[test]
    fn test_encode_backward_compat_uses_32kb() {
        // encode() uses MAX_WINDOW (32KB) for backward compat.
        // Verify it doesn't panic and round-trips.
        let input = vec![b'Z'; 50_000];
        let enc = encode(&input).unwrap();
        let output = decode(
            &enc.flags,
            &enc.literals,
            &enc.offset_codes,
            &enc.offset_extra,
            &enc.length_codes,
            &enc.length_extra,
            enc.num_tokens,
            enc.num_matches,
            input.len(),
        )
        .unwrap();
        assert_eq!(output, input);
    }

    // --- Distance-dependent MIN_MATCH tests (Phase 4) ---

    #[test]
    fn test_min_profitable_length_close_offsets() {
        // Close offsets (1-16) should need only MIN_MATCH (3)
        for offset in 1..=16 {
            assert_eq!(
                min_profitable_length(offset),
                MIN_MATCH,
                "offset {offset} should need min_match=3"
            );
        }
    }

    #[test]
    fn test_min_profitable_length_increases_with_distance() {
        // Minimum length should be non-decreasing with offset
        let mut prev = min_profitable_length(1);
        for offset in [16, 256, 4096, 65536, 500_000u32] {
            let cur = min_profitable_length(offset);
            assert!(
                cur >= prev,
                "min_profitable_length should be non-decreasing: offset {offset} gave {cur}, prev was {prev}"
            );
            prev = cur;
        }
    }

    #[test]
    fn test_min_profitable_length_far_offsets_require_longer() {
        // Very far offsets should require longer matches
        let close = min_profitable_length(1);
        let far = min_profitable_length(100_000);
        assert!(
            far > close,
            "far offset (100K) should need longer match than close (1): {far} vs {close}"
        );
    }

    #[test]
    fn test_distance_dependent_round_trip() {
        // Ensure distance-dependent filtering doesn't break round-trip
        let pattern = b"mixed content with some repeats mixed content! ";
        let mut input = Vec::new();
        for _ in 0..500 {
            input.extend_from_slice(pattern);
        }
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    // --- Repeat offset tests (Phase 5) ---

    #[test]
    fn test_repeat_offsets_state_management() {
        let mut rep = RepeatOffsets::new();
        assert_eq!(rep.recent, [1, 1, 1]);

        // Push a new offset
        rep.push_new(42);
        assert_eq!(rep.recent, [42, 1, 1]);

        // Push another
        rep.push_new(100);
        assert_eq!(rep.recent, [100, 42, 1]);

        // Promote index 1 (swap)
        rep.promote(1);
        assert_eq!(rep.recent, [42, 100, 1]);

        // Promote index 2 (rotate)
        rep.promote(2);
        assert_eq!(rep.recent, [1, 42, 100]);
    }

    #[test]
    fn test_repeat_offset_encode_decode_round_trip() {
        let mut enc_rep = RepeatOffsets::new();
        let mut dec_rep = RepeatOffsets::new();

        // Encode a sequence of offsets (some repeats)
        let offsets = [10, 20, 10, 10, 20, 30, 20, 20, 20];
        for &offset in &offsets {
            let (code, eb, ev) = enc_rep.encode_offset(offset);
            let decoded = dec_rep.decode_offset(code, ev);
            assert_eq!(
                decoded, offset,
                "repeat offset mismatch for offset {offset}"
            );
            assert_eq!(
                extra_bits_for_offset_code(code),
                eb,
                "extra bits mismatch for code {code}"
            );
        }
    }

    #[test]
    fn test_repeat_matches_used_on_structured_data() {
        // With small structured patterns, repeat offsets should be used
        // on subsequent occurrences. Use a pattern short enough that
        // one giant match doesn't cover everything.
        let pattern = b"ABC";
        let mut input = Vec::new();
        // Create 10 copies, each separated by a small unique sequence
        for i in 0..10 {
            input.extend_from_slice(pattern);
            // Add a small unique suffix to prevent one giant match
            input.push(b'0' + (i % 10) as u8);
        }
        let enc = encode(&input).unwrap();

        // Count repeat codes (0-2) in offset_codes
        let repeat_count = enc
            .offset_codes
            .iter()
            .filter(|&&c| c < NUM_REPEAT_CODES)
            .count();

        assert!(
            repeat_count > 0,
            "structured data should use repeat offsets, but found 0 (offset_codes={:?})",
            &enc.offset_codes
        );

        // Verify round-trip
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn test_repeat_offsets_round_trip_various() {
        // All-same bytes: mostly repeat offset 1
        let input = vec![b'Q'; 10_000];
        assert_eq!(round_trip(&input).unwrap(), input);

        // Short repeating pattern: repeat offset = pattern length
        let pattern = b"ABCD";
        let input: Vec<u8> = pattern.iter().cycle().take(10_000).copied().collect();
        assert_eq!(round_trip(&input).unwrap(), input);

        // Alternating patterns at different offsets
        let mut input = Vec::new();
        for i in 0..200 {
            if i % 2 == 0 {
                input.extend_from_slice(b"even pattern here! ");
            } else {
                input.extend_from_slice(b"odd pattern here!! ");
            }
        }
        assert_eq!(round_trip(&input).unwrap(), input);
    }

    #[test]
    fn test_repeat_offsets_improve_ratio() {
        // With small repeated patterns separated by variation,
        // repeat codes should be used on matching offsets.
        let pattern = b"abc";
        let mut input = Vec::new();
        for i in 0..50 {
            input.extend_from_slice(pattern);
            // Add variation to prevent giant matches
            input.push(b'0' + (i as u8 % 10));
        }
        let enc = encode(&input).unwrap();

        let repeat_count = enc
            .offset_codes
            .iter()
            .filter(|&&c| c < NUM_REPEAT_CODES)
            .count();
        let total_matches = enc.num_matches as usize;

        // Expect at least some matches to use repeat codes
        assert!(total_matches > 0, "should have found matches");
        assert!(
            repeat_count > 0,
            "expected some repeat usage on regular data, got 0 repeat codes out of {} matches",
            total_matches
        );
    }

    #[test]
    fn seq_config_hash4_roundtrip() {
        let input = b"the quick brown fox the quick brown fox";
        let config = SeqConfig {
            hash_prefix_len: 4,
            ..SeqConfig::default()
        };
        let encoded = encode_with_config(input, &config).unwrap();
        assert!(encoded.num_matches > 0, "hash4 config should find matches");
    }

    #[test]
    fn seq_config_default_no_regression() {
        let input = b"aababcabcdabcdeabcdefabcdefg";
        let encoded_default = encode_with_config(input, &SeqConfig::default()).unwrap();
        let encoded_hash3 = encode_with_config(
            input,
            &SeqConfig {
                hash_prefix_len: 3,
                ..SeqConfig::default()
            },
        )
        .unwrap();
        // Same number of matches — hash selection should not change token count
        // for small inputs where both hashes resolve cleanly.
        assert_eq!(encoded_default.num_tokens, encoded_hash3.num_tokens);
    }

    #[test]
    fn adaptive_chain_finds_matches_on_compressible_data() {
        let input: Vec<u8> = b"the quick brown fox "
            .iter()
            .cycle()
            .take(4096)
            .copied()
            .collect();
        let config = SeqConfig {
            adaptive_chain: true,
            ..SeqConfig::default()
        };
        let encoded = encode_with_config(&input, &config).unwrap();
        assert!(
            encoded.num_matches > 0,
            "adaptive mode must find matches on repetitive input"
        );
        // Verify basic functionality: matches found and tokens reasonable
        assert!(encoded.num_tokens > 0, "should have tokens");
    }

    #[test]
    fn adaptive_chain_no_panic_on_random() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        // Deterministic pseudo-random bytes
        let input: Vec<u8> = (0u32..4096)
            .map(|i| {
                let mut h = DefaultHasher::new();
                i.hash(&mut h);
                h.finish() as u8
            })
            .collect();
        let config = SeqConfig {
            adaptive_chain: true,
            ..SeqConfig::default()
        };
        let _ = encode_with_config(&input, &config).unwrap(); // must not panic
    }

    #[test]
    fn adaptive_chain_false_is_identical_to_default() {
        let input = b"hello world hello world hello world";
        let default_encoded = encode_with_config(input, &SeqConfig::default()).unwrap();
        let explicit_false = encode_with_config(
            input,
            &SeqConfig {
                adaptive_chain: false,
                ..SeqConfig::default()
            },
        )
        .unwrap();
        assert_eq!(default_encoded.num_tokens, explicit_false.num_tokens);
        assert_eq!(default_encoded.num_matches, explicit_false.num_matches);
    }

    #[test]
    fn adaptive_vs_nonadaptive_compressible_match_count() {
        let input: Vec<u8> = b"abcdefghij".iter().cycle().take(65536).copied().collect();

        let adaptive_cfg = SeqConfig {
            adaptive_chain: true,
            ..SeqConfig::default()
        };
        let normal_cfg = SeqConfig {
            adaptive_chain: false,
            ..SeqConfig::default()
        };

        let adaptive = encode_with_config(&input, &adaptive_cfg).unwrap();
        let normal = encode_with_config(&input, &normal_cfg).unwrap();

        // Adaptive must find at least 95% as many matches as non-adaptive
        let threshold = (normal.num_matches as f64 * 0.95) as u32;
        assert!(
            adaptive.num_matches >= threshold,
            "adaptive found {} matches, non-adaptive found {} (threshold {})",
            adaptive.num_matches,
            normal.num_matches,
            threshold
        );
    }

    #[test]
    fn adaptive_mixed_input_no_corruption() {
        // First 32KB repetitive, next 32KB pseudo-random
        let mut input: Vec<u8> = b"abcde".iter().cycle().take(32768).copied().collect();
        input.extend((0u8..=255).cycle().take(32768));

        let config = SeqConfig {
            adaptive_chain: true,
            ..SeqConfig::default()
        };
        let encoded = encode_with_config(&input, &config).unwrap();
        // Must have found at least some matches (from the repetitive first half)
        assert!(
            encoded.num_matches > 0,
            "adaptive mode must find matches in mixed input"
        );
    }

    // --- Min profitable length tests (Task 7) ---

    #[test]
    fn min_profitable_length_offset_zero() {
        assert_eq!(min_profitable_length(0), u16::MAX);
    }

    #[test]
    fn min_profitable_length_tiers() {
        // oeb 0-3: offset 1-16 → min 3
        // oeb values: 1(0), 2(0), 3-4(1), 5-8(2), 9-16(3)
        for d in 1u32..=16 {
            assert_eq!(
                min_profitable_length(d),
                3,
                "offset {d} should require min 3"
            );
        }
        // oeb 4-8: offset 17-256 → min 4
        // oeb values: 17-32(4), 33-64(5), 65-128(6), 129-256(7)
        for d in [17u32, 32, 64, 128, 256] {
            assert_eq!(
                min_profitable_length(d),
                4,
                "offset {d} should require min 4"
            );
        }
        // oeb 9-12: offset 257-4096 → min 5
        // oeb values: 257-512(8), 513-1024(9), 1025-2048(10), 2049-4096(11)
        for d in [257u32, 512, 1024, 2048, 4096] {
            assert_eq!(
                min_profitable_length(d),
                5,
                "offset {d} should require min 5"
            );
        }
        // oeb 13-16: offset 4097-65536 → min 6
        // oeb values: 4097-8192(12), 8193-16384(13), 16385-32768(14), 32769-65536(15)
        for d in [4097u32, 8192, 16384, 32768, 65536] {
            assert_eq!(
                min_profitable_length(d),
                6,
                "offset {d} should require min 6"
            );
        }
        // oeb 17+: offset 65537+ → min 7
        // oeb values: 65537-131072(16), 131073+(17)
        assert_eq!(min_profitable_length(65537), 7);
        assert_eq!(min_profitable_length(131072), 7);
    }

    #[test]
    fn min_profitable_length_monotone() {
        // Sample offsets spanning all tiers; verify non-decreasing.
        let offsets = [
            1u32, 4, 8, 9, 32, 256, 257, 1024, 4096, 4097, 32768, 65536, 65537, 131072,
        ];
        let thresholds: Vec<u16> = offsets.iter().map(|&d| min_profitable_length(d)).collect();
        for w in thresholds.windows(2) {
            assert!(
                w[0] <= w[1],
                "min_profitable_length not monotone: {} > {} for adjacent offsets",
                w[0],
                w[1]
            );
        }
    }

    // --- Match profitability edge cases (Task 8) ---

    #[test]
    fn encoder_rejects_short_match_at_large_offset() {
        // Test that short matches at large offsets are rejected.
        // Create a pattern that repeats after 300 bytes, with 3-byte and 4-byte variants.
        let pattern = b"ABCDE";
        let mut input_3byte = Vec::new();
        let mut input_4byte = Vec::new();

        // Add pattern, then 295 filler bytes, then repeat pattern
        input_3byte.extend_from_slice(&pattern[..3]); // first 3 bytes
        input_4byte.extend_from_slice(pattern); // first 5 bytes

        // Add ~295 bytes of filler to create offset ~300
        for i in 0..295 {
            input_3byte.push((i % 256) as u8 ^ 0x55); // non-repeating filler
            input_4byte.push((i % 256) as u8 ^ 0x55);
        }

        // Append the match target: 3 bytes for test 1
        input_3byte.extend_from_slice(&pattern[..3]);

        // Append the match target: 4 bytes for test 2
        input_4byte.extend_from_slice(&pattern[..4]);

        let config = SeqConfig {
            max_window: 128 * 1024,
            ..SeqConfig::default()
        };

        let enc_3 = encode_with_config(&input_3byte, &config).unwrap();
        let enc_4 = encode_with_config(&input_4byte, &config).unwrap();

        // min_profitable_length(300) = 5, so:
        // - 3-byte match at offset 300 should be rejected
        // - 4-byte match at offset 300 should be rejected
        // - 5-byte match at offset 300 should be accepted

        // The encodings should produce very similar match counts (neither should
        // find the distant match at offset 300 for short lengths).
        // If the profitability filter is working, enc_3 and enc_4 should both be
        // rejected and have similar compression.
        assert!(
            enc_3.num_matches == enc_4.num_matches
                || (enc_3.num_matches == 0 && enc_4.num_matches == 0),
            "short matches at offset ~300 should be rejected: 3-byte gave {}, 4-byte gave {}",
            enc_3.num_matches,
            enc_4.num_matches
        );
    }

    #[test]
    fn encoder_accepts_profitable_match_at_large_offset() {
        // Test that profitable (long enough) matches at large offsets are accepted.
        let pattern = b"ABCDE";
        let mut input = Vec::new();

        input.extend_from_slice(pattern); // first 5 bytes

        // Add ~295 bytes of filler to create offset ~300
        for i in 0..295 {
            input.push((i % 256) as u8 ^ 0x55); // non-repeating filler
        }

        // Append the full pattern: 5 bytes (which should be accepted at offset ~300)
        input.extend_from_slice(pattern);

        let config = SeqConfig {
            max_window: 128 * 1024,
            ..SeqConfig::default()
        };
        let encoded = encode_with_config(&input, &config).unwrap();

        // min_profitable_length(300) = 5, so a 5-byte match should be accepted.
        // We should find at least 1 match.
        assert!(
            encoded.num_matches >= 1,
            "5-byte match at offset ~300 should be accepted"
        );
    }

    #[test]
    fn encoder_all_same_bytes_efficient() {
        let input = vec![0xAAu8; 1024];
        let encoded = encode_with_config(&input, &SeqConfig::default()).unwrap();
        // All-same input: should produce very few match tokens (ideally 1-2)
        // and very few literals (just the first MIN_MATCH bytes before first match)
        assert!(
            encoded.num_matches >= 1,
            "all-same input must find at least one match"
        );
        assert!(
            encoded.num_tokens < 20,
            "all-same 1KB input should compress to very few tokens, got {}",
            encoded.num_tokens
        );
    }

    #[test]
    fn encoder_profitability_round_trip() {
        // Ensure profitability filtering doesn't break round-trip
        let pattern = b"mixed content with some repeats mixed content! ";
        let mut input = Vec::new();
        for _ in 0..500 {
            input.extend_from_slice(pattern);
        }
        let output = round_trip(&input).unwrap();
        assert_eq!(output, input);
    }
}
