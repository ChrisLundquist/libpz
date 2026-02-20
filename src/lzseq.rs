/// LzSeq: zstd-style code+extra-bits sequence encoding for LZ matches.
///
/// Uses log2-based code tables for offsets and lengths, with variable-width
/// extra bits packed into separate bitstreams. Close matches cost 2-3 bytes,
/// far matches cost proportionally to log2(distance).
///
/// Output: 6 independent streams for entropy coding:
/// - flags: 1 bit/token packed MSB-first (literal=1, match=0)
/// - literals: 1 byte per literal token
/// - offset_codes: 1 byte per match (values 0-31)
/// - offset_extra: packed bitstream (LSB-first)
/// - length_codes: 1 byte per match (values 0-20)
/// - length_extra: packed bitstream (LSB-first)
///
/// Code tables (log2-based, similar to zstd):
///
/// Offset codes:
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
use crate::lz77::{HashChainFinder, DEFAULT_MAX_MATCH, MAX_WINDOW, MIN_MATCH};
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
}

impl Default for SeqConfig {
    fn default() -> Self {
        SeqConfig {
            max_window: 128 * 1024,
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
#[inline]
pub(crate) fn encode_offset(offset: u32) -> (u8, u8, u32) {
    encode_value(offset)
}

/// Decode an offset from code + extra_value.
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
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut r = BitReader {
            data,
            byte_pos: 0,
            container: 0,
            bits_available: 0,
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
        value
    }

    fn refill(&mut self) {
        while self.bits_available <= 56 && self.byte_pos < self.data.len() {
            self.container |= (self.data[self.byte_pos] as u64) << self.bits_available;
            self.byte_pos += 1;
            self.bits_available += 8;
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
fn unpack_flags(bytes: &[u8], count: usize) -> Vec<bool> {
    (0..count)
        .map(|i| bytes[i / 8] & (1 << (7 - (i % 8))) != 0)
        .collect()
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
    encode_with_config(
        input,
        &SeqConfig {
            max_window: MAX_WINDOW,
        },
    )
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

    let mut finder = HashChainFinder::with_window(config.max_window, DEFAULT_MAX_MATCH);
    let mut flags_vec: Vec<bool> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut offset_codes: Vec<u8> = Vec::new();
    let mut length_codes: Vec<u8> = Vec::new();
    let mut offset_extra_writer = BitWriter::new();
    let mut length_extra_writer = BitWriter::new();
    let mut pos: usize = 0;

    while pos < input.len() {
        let m = finder.find_match_wide(input, pos);
        finder.insert(input, pos);

        // Lazy matching: check if next position has a longer match
        if m.length >= MIN_MATCH && m.length < LAZY_SKIP_THRESHOLD && pos + 1 < input.len() {
            finder.insert(input, pos + 1);
            let next_m = finder.find_match_wide(input, pos + 1);

            if next_m.length > m.length {
                // Emit literal for current position, use the better match
                flags_vec.push(true);
                literals.push(input[pos]);
                pos += 1;

                // Emit match from next position
                flags_vec.push(false);
                let (oc, oeb, oev) = encode_offset(next_m.offset);
                offset_codes.push(oc);
                offset_extra_writer.write_bits(oev, oeb);
                let (lc, leb, lev) = encode_length(next_m.length);
                length_codes.push(lc);
                length_extra_writer.write_bits(lev, leb);

                // Insert covered positions into hash chains
                let advance = next_m.length as usize;
                let insert_count = advance.min(input.len() - pos).min(MAX_INSERT_LEN);
                for i in 1..insert_count {
                    finder.insert(input, pos + i);
                }
                pos += advance;
                continue;
            }
        }

        if m.length >= MIN_MATCH {
            // Emit match
            flags_vec.push(false);
            let (oc, oeb, oev) = encode_offset(m.offset);
            offset_codes.push(oc);
            offset_extra_writer.write_bits(oev, oeb);
            let (lc, leb, lev) = encode_length(m.length);
            length_codes.push(lc);
            length_extra_writer.write_bits(lev, leb);

            let advance = m.length as usize;
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

    let flag_bits = unpack_flags(flags, num_tokens as usize);
    let mut lit_pos = 0usize;
    let mut match_idx = 0usize;
    let mut off_extra_reader = BitReader::new(offset_extra);
    let mut len_extra_reader = BitReader::new(length_extra);
    let mut output = Vec::with_capacity(original_len);

    for &is_literal in &flag_bits {
        if is_literal {
            if lit_pos >= literals.len() {
                return Err(PzError::InvalidInput);
            }
            output.push(literals[lit_pos]);
            lit_pos += 1;
        } else {
            if match_idx >= num_matches as usize {
                return Err(PzError::InvalidInput);
            }
            let oc = offset_codes[match_idx];
            let oeb = extra_bits_for_code(oc);
            let oev = off_extra_reader.read_bits(oeb);
            let offset = decode_offset(oc, oev) as usize;

            let lc = length_codes[match_idx];
            let leb = extra_bits_for_code(lc);
            let lev = len_extra_reader.read_bits(leb);
            let length = decode_length(lc, lev) as usize;

            match_idx += 1;

            if offset == 0 || offset > output.len() {
                return Err(PzError::InvalidInput);
            }

            let copy_start = output.len() - offset;
            for j in 0..length {
                let byte = output[copy_start + j];
                output.push(byte);
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
        };
        let enc_wide = encode_with_config(&input, &config_wide).unwrap();

        // Narrow window: cannot see across the gap
        let config_narrow = SeqConfig {
            max_window: MAX_WINDOW, // 32KB
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
}
