/// Finite State Entropy (FSE) encoder/decoder.
///
/// Implements tANS (table-based Asymmetric Numeral Systems), a near-optimal
/// entropy coder that uses table lookups instead of arithmetic operations.
/// All critical paths are pure table lookups and bit shifts — no divisions
/// or data-dependent branches — making FSE ideal for GPU/SIMD execution.
///
/// FSE approaches Shannon entropy (like arithmetic/range coding) but decodes
/// faster due to the table-driven state machine.
///
/// # Algorithm
///
/// **Encoding** (reverse-order):
/// 1. Build normalized frequency table (power-of-2 sum).
/// 2. Spread symbols across state table using quasi-random pattern.
/// 3. Process input in reverse: table lookup + bit output per symbol.
/// 4. Serialize table and final state with the bitstream.
///
/// **Decoding** (forward-order):
/// 1. Deserialize table and initial state from stream.
/// 2. For each position: `table[state]` → (symbol, bits_to_read, next_base).
/// 3. Read bits, compute next state, emit symbol.
///
/// # GPU Suitability
///
/// The decode loop is a tight sequence of table lookups with no
/// data-dependent branches or divisions. Multiple independent streams
/// can be decoded in parallel on GPU compute units. The encode
/// direction is inherently sequential per-stream, but can be
/// parallelized across independent blocks.
use crate::frequency::FrequencyTable;
use crate::{PzError, PzResult};

/// Default accuracy log (table size = 1 << 7 = 128 entries).
///
/// Benchmarks show accuracy_log 7 gives equivalent compression ratio to 9
/// on typical data, with ~2× faster encode/decode due to smaller tables.
pub const DEFAULT_ACCURACY_LOG: u8 = 7;

/// Minimum supported accuracy log. Below this, table is too small for
/// reasonable compression of byte data.
pub const MIN_ACCURACY_LOG: u8 = 5;

/// Maximum supported accuracy log. Above this, tables consume excessive
/// memory for diminishing returns.
pub const MAX_ACCURACY_LOG: u8 = 12;

/// Number of symbols in the byte alphabet.
const NUM_SYMBOLS: usize = 256;

/// Size of the serialized frequency table in the header (256 x u16 LE).
const FREQ_TABLE_BYTES: usize = NUM_SYMBOLS * 2;

/// Fixed header size: accuracy_log(1) + freq_table(512) + initial_state(2) + total_bits(4).
const HEADER_SIZE: usize = 1 + FREQ_TABLE_BYTES + 2 + 4;

// ---------------------------------------------------------------------------
// Normalized frequency table
// ---------------------------------------------------------------------------

/// Normalized frequency table where frequencies sum to a power of 2.
#[derive(Debug, Clone, PartialEq)]
struct NormalizedFreqs {
    /// Normalized frequency for each symbol. Sum = 1 << accuracy_log.
    freq: [u16; NUM_SYMBOLS],
    /// The accuracy log. table_size = 1 << accuracy_log.
    accuracy_log: u8,
}

/// Normalize raw frequencies so they sum to exactly `1 << accuracy_log`.
///
/// Every symbol with a nonzero raw count is guaranteed at least 1 in the
/// normalized table. Rounding remainder is distributed to the symbols
/// with the largest raw counts.
fn normalize_frequencies(raw: &FrequencyTable, accuracy_log: u8) -> PzResult<NormalizedFreqs> {
    let table_size = 1u32 << accuracy_log;
    let total = raw.total;

    if total == 0 || raw.used == 0 {
        return Err(PzError::InvalidInput);
    }

    // Single symbol: it gets the entire table.
    if raw.used == 1 {
        let mut freq = [0u16; NUM_SYMBOLS];
        for (i, &count) in raw.byte.iter().enumerate() {
            if count > 0 {
                freq[i] = table_size as u16;
                break;
            }
        }
        return Ok(NormalizedFreqs { freq, accuracy_log });
    }

    // Need table_size >= num_symbols (each present symbol needs >= 1 slot).
    if table_size < raw.used {
        return Err(PzError::InvalidInput);
    }

    let mut norm_freq = [0u16; NUM_SYMBOLS];
    let mut distributed = 0u32;

    // Indices of present symbols, sorted by raw count descending.
    let mut present: Vec<usize> = (0..NUM_SYMBOLS).filter(|&i| raw.byte[i] > 0).collect();
    present.sort_by(|&a, &b| raw.byte[b].cmp(&raw.byte[a]));

    // Proportional scaling with floor, ensuring minimum of 1.
    for &i in &present {
        let scaled = ((raw.byte[i] as u64 * table_size as u64) / total).max(1) as u16;
        norm_freq[i] = scaled;
        distributed += scaled as u32;
    }

    // Adjust to hit exact sum == table_size.
    let mut diff = table_size as i32 - distributed as i32;

    if diff > 0 {
        let mut idx = 0;
        while diff > 0 {
            norm_freq[present[idx % present.len()]] += 1;
            diff -= 1;
            idx += 1;
        }
    } else {
        let mut idx = 0;
        while diff < 0 {
            let sym = present[idx % present.len()];
            if norm_freq[sym] > 1 {
                norm_freq[sym] -= 1;
                diff += 1;
            }
            idx += 1;
        }
    }

    debug_assert_eq!(norm_freq.iter().map(|&f| f as u32).sum::<u32>(), table_size);

    Ok(NormalizedFreqs {
        freq: norm_freq,
        accuracy_log,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Floor of log2 for a positive integer.
fn highest_bit_set(x: u32) -> u8 {
    debug_assert!(x > 0);
    31 - x.leading_zeros() as u8
}

// ---------------------------------------------------------------------------
// Symbol spread
// ---------------------------------------------------------------------------

/// Spread symbols across the state table using the classic tANS pattern.
///
/// Uses step = `(table_size >> 1) + (table_size >> 3) + 3` to produce
/// quasi-random interleaving.
fn spread_symbols(norm: &NormalizedFreqs) -> Vec<u8> {
    let table_size = 1usize << norm.accuracy_log;
    let mask = table_size - 1;

    // Use a step that is coprime with table_size (which is a power of 2).
    // Any odd number works. The classic FSE step provides good interleaving.
    let step = (table_size >> 1) + (table_size >> 3) + 3;
    // step is always odd (sum of even + even + 3), so gcd(step, 2^n) = 1.

    let mut table = vec![255u8; table_size];

    // Generate the full permutation sequence first. Since gcd(step, table_size)=1,
    // the sequence pos, pos+step, pos+2*step, ... (mod table_size) visits every
    // position exactly once before repeating.
    let mut positions = Vec::with_capacity(table_size);
    let mut pos = 0usize;
    for _ in 0..table_size {
        positions.push(pos);
        pos = (pos + step) & mask;
    }

    // Assign symbols to positions: symbol 0 gets the first freq[0] positions,
    // symbol 1 gets the next freq[1] positions, etc.
    let mut idx = 0;
    for (symbol, &freq) in norm.freq.iter().enumerate() {
        for _ in 0..freq {
            table[positions[idx]] = symbol as u8;
            idx += 1;
        }
    }

    debug_assert_eq!(idx, table_size);
    table
}

// ---------------------------------------------------------------------------
// Decode table
// ---------------------------------------------------------------------------

/// Entry in the FSE decoding table. Indexed directly by state.
///
/// Decode step (hot path — no divisions, no branches on data):
/// 1. `symbol = table[state].symbol`
/// 2. Read `table[state].bits` bits from bitstream → `value`
/// 3. `state = table[state].next_state_base + value`
/// 4. Emit `symbol`
#[derive(Debug, Clone, Copy, Default)]
struct DecodeEntry {
    symbol: u8,
    bits: u8,
    next_state_base: u16,
}

/// Build the decode table from the spread symbol assignment.
///
/// For each state, tracks a per-symbol destination state starting at
/// `freq[s]` and incrementing. Bits and base are derived from this.
fn build_decode_table(norm: &NormalizedFreqs, spread: &[u8]) -> Vec<DecodeEntry> {
    let table_size = 1usize << norm.accuracy_log;
    let mut decode = vec![DecodeEntry::default(); table_size];

    let mut next_state = [0u16; NUM_SYMBOLS];
    next_state.copy_from_slice(&norm.freq);

    for state in 0..table_size {
        let symbol = spread[state] as usize;
        let dest = next_state[symbol] as u32;
        next_state[symbol] += 1;

        if dest == 0 {
            // Should not happen if spread is correct.
            continue;
        }

        let high_bit = highest_bit_set(dest) as usize;
        let bits = norm.accuracy_log as usize - high_bit;
        let next_state_base = ((dest as usize) << bits) - table_size;

        decode[state] = DecodeEntry {
            symbol: symbol as u8,
            bits: bits as u8,
            next_state_base: next_state_base as u16,
        };
    }

    decode
}

// ---------------------------------------------------------------------------
// Encode table (inverse of decode)
// ---------------------------------------------------------------------------

/// Encode table entry: for a given encoding state and symbol, tells us
/// what bits to output and what new state to transition to.
///
/// Derived by inverting the decode table: for each decode entry at index
/// `c`, decoding `c` produces symbol `s`, reads `bits` bits to get
/// `value`, and transitions to `base + value`. So encoding symbol `s`
/// from state `new_state = base + value` outputs `value` as `bits` bits
/// and transitions to compressed state `c`.
///
/// The per-symbol encode table is a flat array indexed directly by the
/// encoding state for O(1) lookup.
#[derive(Debug, Clone, Copy, Default)]
struct EncodeMapping {
    /// The compressed (decode table) state to transition to.
    compressed_state: u16,
    /// Number of bits to output.
    bits: u8,
    /// Base value to subtract from encoding state to get the output value.
    base: u16,
}

/// Per-symbol encode info: flat direct-lookup table for O(1) state mapping.
///
/// During encoding, the state is always in `[0, table_size)`. For each
/// present symbol we pre-expand the mapping ranges into a flat array so
/// that `lookup[state]` gives the correct `EncodeMapping` without any
/// search. Absent symbols have an empty lookup (never queried).
#[derive(Debug, Clone, Default)]
struct SymbolEncodeTable {
    /// Direct lookup indexed by encoding state. Length is `table_size`
    /// for present symbols, 0 for absent symbols.
    lookup: Vec<EncodeMapping>,
}

impl SymbolEncodeTable {
    /// Find the mapping for a given encoding state (O(1) table lookup).
    #[inline]
    fn find(&self, state: usize) -> &EncodeMapping {
        &self.lookup[state]
    }
}

/// Build per-symbol encode tables by inverting the decode table.
///
/// For each present symbol, expands the decode-table-derived mapping
/// ranges into a flat lookup array of size `table_size` for O(1) access.
fn build_encode_tables(
    norm: &NormalizedFreqs,
    decode_table: &[DecodeEntry],
) -> Vec<SymbolEncodeTable> {
    let table_size = 1usize << norm.accuracy_log;

    // Collect mappings per symbol from the decode table.
    let mut mappings_per_sym: Vec<Vec<EncodeMapping>> = vec![Vec::new(); NUM_SYMBOLS];
    for (c, entry) in decode_table.iter().enumerate() {
        let s = entry.symbol as usize;
        mappings_per_sym[s].push(EncodeMapping {
            compressed_state: c as u16,
            bits: entry.bits,
            base: entry.next_state_base,
        });
    }

    // Expand each present symbol's mappings into a flat lookup array.
    let tables = mappings_per_sym
        .iter()
        .enumerate()
        .map(|(sym, sym_mappings)| {
            if norm.freq[sym] == 0 {
                return SymbolEncodeTable::default();
            }
            let mut lookup = vec![EncodeMapping::default(); table_size];
            for m in sym_mappings {
                let range_start = m.base as usize;
                let range_end = range_start + (1usize << m.bits);
                for entry in lookup
                    .iter_mut()
                    .take(range_end.min(table_size))
                    .skip(range_start)
                {
                    *entry = *m;
                }
            }
            SymbolEncodeTable { lookup }
        })
        .collect();

    tables
}

// ---------------------------------------------------------------------------
// Combined FSE table
// ---------------------------------------------------------------------------

/// Complete FSE table for encoding and decoding.
struct FseTable {
    decode_table: Vec<DecodeEntry>,
    encode_tables: Vec<SymbolEncodeTable>,
    table_size: usize,
}

impl FseTable {
    fn from_normalized(norm: &NormalizedFreqs) -> Self {
        let table_size = 1usize << norm.accuracy_log;
        let spread = spread_symbols(norm);
        let decode_table = build_decode_table(norm, &spread);
        let encode_tables = build_encode_tables(norm, &decode_table);

        FseTable {
            decode_table,
            encode_tables,
            table_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Bit I/O (LSB-first)
// ---------------------------------------------------------------------------

/// Bitstream writer that packs bits LSB-first into a byte buffer.
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

    fn write_bits(&mut self, value: u32, nb_bits: u32) {
        debug_assert!(nb_bits <= 32);
        if nb_bits == 0 {
            return;
        }
        self.container |= (value as u64) << self.bit_pos;
        self.bit_pos += nb_bits;
        self.flush();
    }

    fn flush(&mut self) {
        while self.bit_pos >= 8 {
            self.buffer.push(self.container as u8);
            self.container >>= 8;
            self.bit_pos -= 8;
        }
    }

    fn total_bits(&self) -> u32 {
        self.buffer.len() as u32 * 8 + self.bit_pos
    }

    fn finish(mut self) -> (Vec<u8>, u32) {
        let total = self.total_bits();
        if self.bit_pos > 0 {
            self.buffer.push(self.container as u8);
        }
        (self.buffer, total)
    }
}

/// Bitstream reader that reads bits LSB-first.
struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    container: u64,
    bits_available: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        let mut reader = BitReader {
            data,
            byte_pos: 0,
            container: 0,
            bits_available: 0,
        };
        reader.refill();
        reader
    }

    fn read_bits(&mut self, nb_bits: u32) -> u32 {
        debug_assert!(nb_bits <= 32);
        if nb_bits == 0 {
            return 0;
        }
        self.refill();
        let mask = (1u64 << nb_bits) - 1;
        let value = (self.container & mask) as u32;
        self.container >>= nb_bits;
        self.bits_available = self.bits_available.saturating_sub(nb_bits);
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
// Internal encode / decode
// ---------------------------------------------------------------------------

/// Encode input using FSE tables.
///
/// Returns (bitstream, initial_decoder_state, total_bits).
///
/// The encoder processes symbols in **reverse** order (last symbol first).
/// Bits are pushed LSB-first to the bitstream. The decoder reads the
/// bitstream forward (LSB-first) and emits symbols in forward order.
///
/// For each symbol, we invert the decode step:
/// - Decode: from compressed_state `c`, emit symbol, read `bits` bits → value,
///   next_state = base + value.
/// - Encode: from encoding_state `next_state`, find `c` for this symbol where
///   `base <= next_state < base + 2^bits`, output `next_state - base` as
///   `bits` bits, transition to `c`.
fn fse_encode_internal(input: &[u8], table: &FseTable) -> (Vec<u8>, u16, u32) {
    // Initial encoding state: 0. After encoding all symbols in reverse,
    // the final state becomes the decoder's initial state.
    let mut state: usize = 0;

    // Pre-allocate and index directly: chunks[j] holds the bit-chunk for
    // input[j]. The backward pass fills from the end, so after the loop
    // chunks[0..n] is already in forward order — no reverse needed.
    let mut chunks: Vec<(u32, u32)> = vec![(0, 0); input.len()]; // (value, nb_bits)

    for (j, &byte) in input.iter().enumerate().rev() {
        let s = byte as usize;
        let mapping = table.encode_tables[s].find(state);
        let value = state as u32 - mapping.base as u32;
        chunks[j] = (value, mapping.bits as u32);
        state = mapping.compressed_state as usize;
    }

    let mut writer = BitWriter::new();
    for &(value, nb_bits) in &chunks {
        writer.write_bits(value, nb_bits);
    }

    let (bitstream, total_bits) = writer.finish();
    (bitstream, state as u16, total_bits)
}

/// Decode FSE-encoded bitstream. Returns the decoded byte sequence.
fn fse_decode_internal(
    bitstream: &[u8],
    total_bits: u32,
    initial_state: u16,
    table: &FseTable,
    original_len: usize,
) -> PzResult<Vec<u8>> {
    if total_bits == 0 && original_len > 0 {
        // Single-symbol case: no bits needed, just repeat the symbol.
        if (initial_state as usize) >= table.table_size {
            return Err(PzError::InvalidInput);
        }
        let entry = &table.decode_table[initial_state as usize];
        return Ok(vec![entry.symbol; original_len]);
    }

    let mut reader = BitReader::new(bitstream);
    let mut state = initial_state as usize;
    let mut output = Vec::with_capacity(original_len);

    for _ in 0..original_len {
        if state >= table.table_size {
            return Err(PzError::InvalidInput);
        }
        let entry = &table.decode_table[state];
        output.push(entry.symbol);
        let bits = reader.read_bits(entry.bits as u32);
        state = entry.next_state_base as usize + bits as usize;
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Encode data using Finite State Entropy with the default accuracy log (9).
///
/// Returns self-contained compressed data including the serialized
/// frequency table. The original length must be known by the decoder
/// (stored externally).
pub fn encode(input: &[u8]) -> Vec<u8> {
    encode_with_accuracy(input, DEFAULT_ACCURACY_LOG)
}

/// Encode data using FSE with a specific accuracy log (5..12).
///
/// Higher `accuracy_log` gives better compression but larger tables.
/// Lower values give smaller tables but worse compression.
pub fn encode_with_accuracy(input: &[u8], accuracy_log: u8) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let accuracy_log = accuracy_log.clamp(MIN_ACCURACY_LOG, MAX_ACCURACY_LOG);

    // Count frequencies.
    let mut freq = FrequencyTable::new();
    freq.count(input);

    // If too many distinct symbols for the table size, bump accuracy_log.
    let mut al = accuracy_log;
    while (1u32 << al) < freq.used {
        al += 1;
        if al > MAX_ACCURACY_LOG {
            break;
        }
    }

    let norm = normalize_frequencies(&freq, al).expect("valid non-empty input");
    let table = FseTable::from_normalized(&norm);
    let (bitstream, initial_state, total_bits) = fse_encode_internal(input, &table);

    // Serialize: header + bitstream.
    let mut output = Vec::with_capacity(HEADER_SIZE + bitstream.len());
    output.push(al);
    for &f in &norm.freq {
        output.extend_from_slice(&f.to_le_bytes());
    }
    output.extend_from_slice(&initial_state.to_le_bytes());
    output.extend_from_slice(&total_bits.to_le_bytes());
    output.extend_from_slice(&bitstream);

    output
}

/// Decode FSE-encoded data.
///
/// `original_len` is the number of bytes in the original uncompressed data.
pub fn decode(input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
    if original_len == 0 {
        return Ok(Vec::new());
    }
    if input.len() < HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    // Parse header.
    let accuracy_log = input[0];
    if !(MIN_ACCURACY_LOG..=MAX_ACCURACY_LOG).contains(&accuracy_log) {
        return Err(PzError::InvalidInput);
    }

    // Read normalized frequency table.
    let mut norm_freq = [0u16; NUM_SYMBOLS];
    for (i, freq) in norm_freq.iter_mut().enumerate() {
        let offset = 1 + i * 2;
        *freq = u16::from_le_bytes([input[offset], input[offset + 1]]);
    }

    // Validate: sum must equal table_size.
    let table_size = 1u32 << accuracy_log;
    let sum: u32 = norm_freq.iter().map(|&f| f as u32).sum();
    if sum != table_size {
        return Err(PzError::InvalidInput);
    }

    let norm = NormalizedFreqs {
        freq: norm_freq,
        accuracy_log,
    };

    let header_end = 1 + FREQ_TABLE_BYTES;
    let initial_state = u16::from_le_bytes([input[header_end], input[header_end + 1]]);
    let total_bits = u32::from_le_bytes([
        input[header_end + 2],
        input[header_end + 3],
        input[header_end + 4],
        input[header_end + 5],
    ]);

    let bitstream = &input[header_end + 6..];

    let table = FseTable::from_normalized(&norm);
    fse_decode_internal(bitstream, total_bits, initial_state, &table, original_len)
}

/// Decode FSE-encoded data into a pre-allocated buffer.
///
/// Returns the number of bytes written.
pub fn decode_to_buf(input: &[u8], original_len: usize, output: &mut [u8]) -> PzResult<usize> {
    if original_len == 0 {
        return Ok(0);
    }
    if output.len() < original_len {
        return Err(PzError::BufferTooSmall);
    }
    let decoded = decode(input, original_len)?;
    output[..original_len].copy_from_slice(&decoded);
    Ok(original_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Frequency normalization ---

    #[test]
    fn test_normalize_frequencies_basic() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaabbc");
        let norm = normalize_frequencies(&freq, 5).unwrap();
        let sum: u32 = norm.freq.iter().map(|&f| f as u32).sum();
        assert_eq!(sum, 32);
        assert!(norm.freq[b'a' as usize] > 0);
        assert!(norm.freq[b'b' as usize] > 0);
        assert!(norm.freq[b'c' as usize] > 0);
    }

    #[test]
    fn test_normalize_single_symbol() {
        let mut freq = FrequencyTable::new();
        freq.count(&[42u8; 100]);
        let norm = normalize_frequencies(&freq, 8).unwrap();
        assert_eq!(norm.freq[42], 256);
    }

    #[test]
    fn test_normalize_preserves_order() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaaaabbbcc");
        let norm = normalize_frequencies(&freq, 5).unwrap();
        assert!(norm.freq[b'a' as usize] > norm.freq[b'b' as usize]);
        assert!(norm.freq[b'b' as usize] > norm.freq[b'c' as usize]);
    }

    #[test]
    fn test_normalize_empty_fails() {
        let freq = FrequencyTable::new();
        assert_eq!(normalize_frequencies(&freq, 8), Err(PzError::InvalidInput));
    }

    // --- Symbol spread ---

    #[test]
    fn test_spread_all_filled() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaabbc");
        let norm = normalize_frequencies(&freq, 5).unwrap();
        let spread = spread_symbols(&norm);
        assert_eq!(spread.len(), 32);
        assert!(spread.iter().all(|&s| s != 255));
    }

    #[test]
    fn test_spread_counts_match() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaabbc");
        let norm = normalize_frequencies(&freq, 5).unwrap();
        let spread = spread_symbols(&norm);
        for (sym, &expected) in norm.freq.iter().enumerate() {
            if expected > 0 {
                let actual = spread.iter().filter(|&&s| s == sym as u8).count();
                assert_eq!(actual, expected as usize);
            }
        }
    }

    // --- Decode table ---

    #[test]
    fn test_decode_table_valid_symbols() {
        let mut freq = FrequencyTable::new();
        freq.count(b"hello, world!");
        let norm = normalize_frequencies(&freq, 8).unwrap();
        let spread = spread_symbols(&norm);
        let decode_table = build_decode_table(&norm, &spread);
        for entry in &decode_table {
            assert!(
                norm.freq[entry.symbol as usize] > 0,
                "decode table references absent symbol {}",
                entry.symbol
            );
        }
    }

    // --- BitWriter / BitReader ---

    #[test]
    fn test_bitwriter_reader_roundtrip() {
        let mut writer = BitWriter::new();
        writer.write_bits(0b101, 3);
        writer.write_bits(0b1100, 4);
        writer.write_bits(0b1, 1);
        writer.write_bits(0b11010, 5);
        let (data, total_bits) = writer.finish();
        assert_eq!(total_bits, 13);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(3), 0b101);
        assert_eq!(reader.read_bits(4), 0b1100);
        assert_eq!(reader.read_bits(1), 0b1);
        assert_eq!(reader.read_bits(5), 0b11010);
    }

    #[test]
    fn test_bitwriter_reader_large_values() {
        let mut writer = BitWriter::new();
        writer.write_bits(0xDEAD, 16);
        writer.write_bits(0xBEEF, 16);
        let (data, total_bits) = writer.finish();
        assert_eq!(total_bits, 32);

        let mut reader = BitReader::new(&data);
        assert_eq!(reader.read_bits(16), 0xDEAD);
        assert_eq!(reader.read_bits(16), 0xBEEF);
    }

    #[test]
    fn test_bitwriter_zero_bits() {
        let mut writer = BitWriter::new();
        writer.write_bits(0, 0);
        let (data, total_bits) = writer.finish();
        assert_eq!(total_bits, 0);
        assert!(data.is_empty());
    }

    // --- Round-trip tests ---

    #[test]
    fn test_empty() {
        assert_eq!(encode(&[]), Vec::<u8>::new());
        assert_eq!(decode(&[], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_single_byte() {
        let input = &[42u8];
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_repeated_byte() {
        let input = vec![b'a'; 100];
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_two_bytes() {
        let input = b"ab";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_hello() {
        let input = b"hello, world!";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_banana() {
        let input = b"banana";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_all_bytes() {
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_longer_text() {
        let input =
            b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_binary() {
        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    // --- Compression effectiveness ---

    #[test]
    fn test_compression_skewed() {
        let mut input = vec![0u8; 2000];
        input.push(1);
        input.push(2);
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
        assert!(
            encoded.len() < input.len(),
            "encoded {} bytes, expected < {}",
            encoded.len(),
            input.len()
        );
    }

    #[test]
    fn test_compresses_repeated() {
        let input = vec![0u8; 2000];
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
        assert!(
            encoded.len() < input.len(),
            "encoded {} bytes, expected < {}",
            encoded.len(),
            input.len()
        );
    }

    // --- Accuracy log variants ---

    #[test]
    fn test_accuracy_log_5() {
        let input = b"the quick brown fox jumps over the lazy dog";
        let encoded = encode_with_accuracy(input, 5);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_accuracy_log_12() {
        let input = b"the quick brown fox jumps over the lazy dog. test test test.";
        let encoded = encode_with_accuracy(input, 12);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_all_accuracy_logs() {
        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        for al in MIN_ACCURACY_LOG..=MAX_ACCURACY_LOG {
            let encoded = encode_with_accuracy(&input, al);
            let decoded = decode(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input, "failed at accuracy_log={}", al);
        }
    }

    // --- Error handling ---

    #[test]
    fn test_decode_too_short() {
        let result = decode(&[0u8; 10], 5);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_invalid_accuracy_log() {
        let mut bad = vec![0u8; HEADER_SIZE + 10];
        bad[0] = 15; // > MAX_ACCURACY_LOG
        assert_eq!(decode(&bad, 5), Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_to_buf_basic() {
        let input = b"hello, world!";
        let encoded = encode(input);
        let mut buf = vec![0u8; 100];
        let size = decode_to_buf(&encoded, input.len(), &mut buf).unwrap();
        assert_eq!(&buf[..size], input);
    }

    #[test]
    fn test_decode_to_buf_too_small() {
        let input = b"hello, world!";
        let encoded = encode(input);
        let mut buf = vec![0u8; 2];
        assert_eq!(
            decode_to_buf(&encoded, input.len(), &mut buf),
            Err(PzError::BufferTooSmall)
        );
    }

    // --- Medium data ---

    #[test]
    fn test_round_trip_medium() {
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend(b"The Burrows-Wheeler transform clusters bytes. ");
        }
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_large_repeated_pattern() {
        let pattern: Vec<u8> = (0..=127).collect();
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend(&pattern);
        }
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }
}
