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
pub(crate) const FREQ_TABLE_BYTES: usize = NUM_SYMBOLS * 2;

/// Fixed header size: accuracy_log(1) + freq_table(512) + initial_state(2) + total_bits(4).
const HEADER_SIZE: usize = 1 + FREQ_TABLE_BYTES + 2 + 4;

// ---------------------------------------------------------------------------
// Normalized frequency table
// ---------------------------------------------------------------------------

/// Normalized frequency table where frequencies sum to a power of 2.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NormalizedFreqs {
    /// Normalized frequency for each symbol. Sum = 1 << accuracy_log.
    pub(crate) freq: [u16; NUM_SYMBOLS],
    /// The accuracy log. table_size = 1 << accuracy_log.
    pub(crate) accuracy_log: u8,
}

/// Normalize raw frequencies so they sum to exactly `1 << accuracy_log`.
///
/// Every symbol with a nonzero raw count is guaranteed at least 1 in the
/// normalized table. Rounding remainder is distributed to the symbols
/// with the largest raw counts.
pub(crate) fn normalize_frequencies(
    raw: &FrequencyTable,
    accuracy_log: u8,
) -> PzResult<NormalizedFreqs> {
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
pub(crate) fn spread_symbols(norm: &NormalizedFreqs) -> Vec<u8> {
    let table_size = 1usize << norm.accuracy_log;
    let mask = table_size - 1;

    // Use a step that is coprime with table_size (which is a power of 2).
    // Any odd number works. The classic FSE step provides good interleaving.
    let step = (table_size >> 1) + (table_size >> 3) + 3;
    // step is always odd (sum of even + even + 3), so gcd(step, 2^n) = 1.

    let mut table = vec![255u8; table_size];

    // Assign symbols directly while walking the permutation cycle. This keeps
    // the classic spread pattern but avoids the intermediate positions buffer.
    let mut pos = 0usize;
    let mut written = 0usize;
    for (symbol, &freq) in norm.freq.iter().enumerate() {
        for _ in 0..freq {
            table[pos] = symbol as u8;
            pos = (pos + step) & mask;
            written += 1;
        }
    }

    debug_assert_eq!(written, table_size);
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
pub(crate) struct DecodeEntry {
    pub(crate) symbol: u8,
    pub(crate) bits: u8,
    pub(crate) next_state_base: u16,
}

/// Build the decode table from the spread symbol assignment.
///
/// For each state, tracks a per-symbol destination state starting at
/// `freq[s]` and incrementing. Bits and base are derived from this.
pub(crate) fn build_decode_table(norm: &NormalizedFreqs, spread: &[u8]) -> Vec<DecodeEntry> {
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
                let range_end = range_start.saturating_add(1usize << m.bits).min(table_size);
                if range_start < range_end {
                    lookup[range_start..range_end].fill(*m);
                }
            }
            SymbolEncodeTable { lookup }
        })
        .collect();

    tables
}

/// Build a flat GPU-friendly encode table packed as `u32` values.
///
/// Returns a `Vec<u32>` of length `256 * table_size`, indexed by
/// `symbol * table_size + state`. Each entry is packed as:
///
/// ```text
/// bits  0..11  = compressed_state (12 bits, max 4096 for accuracy_log=12)
/// bits 12..15  = bits_to_output   (4 bits, max 12)
/// bits 16..31  = base             (16 bits)
/// ```
///
/// Absent symbols (frequency == 0) have all-zero entries (never accessed
/// during encoding because no input byte maps to them).
#[allow(dead_code)] // Used by GPU FSE encode (webgpu feature)
pub(crate) fn build_gpu_encode_table(norm: &NormalizedFreqs) -> Vec<u32> {
    let table_size = 1usize << norm.accuracy_log;
    let spread = spread_symbols(norm);
    let decode_table = build_decode_table(norm, &spread);
    let encode_tables = build_encode_tables(norm, &decode_table);

    let mut packed = vec![0u32; NUM_SYMBOLS * table_size];
    for (sym, sym_table) in encode_tables.iter().enumerate() {
        if sym_table.lookup.is_empty() {
            continue;
        }
        let base_idx = sym * table_size;
        for (state, m) in sym_table.lookup.iter().enumerate() {
            packed[base_idx + state] = (m.compressed_state as u32 & 0xFFF)
                | ((m.bits as u32 & 0xF) << 12)
                | ((m.base as u32) << 16);
        }
    }
    packed
}

// ---------------------------------------------------------------------------
// Combined FSE table
// ---------------------------------------------------------------------------

/// Complete FSE table for encoding and decoding.
pub(crate) struct FseTable {
    pub(crate) decode_table: Vec<DecodeEntry>,
    encode_tables: Vec<SymbolEncodeTable>,
    pub(crate) table_size: usize,
}

impl FseTable {
    pub(crate) fn from_normalized(norm: &NormalizedFreqs) -> Self {
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

    #[inline]
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

    /// Refill the bit container from the byte stream.
    ///
    /// Fast path: single unaligned u64 load when ≥8 bytes remain,
    /// replacing up to 7 byte-at-a-time loop iterations with one `mov`.
    /// Falls back to byte-at-a-time for the last <8 bytes.
    #[inline]
    fn refill(&mut self) {
        if self.bits_available <= 56 {
            if self.byte_pos + 8 <= self.data.len() {
                // Bulk load: one unaligned u64 read (single `mov` on x86).
                let raw = u64::from_le_bytes(
                    self.data[self.byte_pos..self.byte_pos + 8]
                        .try_into()
                        .unwrap(),
                );
                self.container |= raw << self.bits_available;
                let bytes_consumed = ((64 - self.bits_available) / 8) as usize;
                self.byte_pos += bytes_consumed;
                self.bits_available += (bytes_consumed as u32) * 8;
            } else {
                // Tail: byte-at-a-time fallback for last <8 bytes.
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

    // Store bit values and bit counts in separate arrays to reduce memory
    // bandwidth during the write pass.
    let mut bit_values = vec![0u32; input.len()];
    let mut bit_counts = vec![0u8; input.len()];

    for (j, &byte) in input.iter().enumerate().rev() {
        let s = byte as usize;
        let mapping = table.encode_tables[s].find(state);
        let value = state as u32 - mapping.base as u32;
        bit_values[j] = value;
        bit_counts[j] = mapping.bits;
        state = mapping.compressed_state as usize;
    }

    let mut writer = BitWriter::new();
    for (&value, &nb_bits) in bit_values.iter().zip(bit_counts.iter()) {
        writer.write_bits(value, nb_bits as u32);
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

// ---------------------------------------------------------------------------
// Interleaved N-way FSE encode / decode
// ---------------------------------------------------------------------------

/// Default number of interleaved streams.
const DEFAULT_INTERLEAVE: usize = 4;

/// Encode input using N interleaved FSE streams.
///
/// Symbol `i` is processed by state `i % num_states`. Each state produces
/// its own independent bitstream. This enables N-way parallelism on decode:
/// each stream can be decoded independently (on separate GPU threads, etc.)
/// with zero data dependencies.
///
/// Returns per-stream (bitstream, initial_state, total_bits).
pub(crate) fn fse_encode_interleaved(
    input: &[u8],
    table: &FseTable,
    num_states: usize,
) -> Vec<(Vec<u8>, u16, u32)> {
    let mut states: Vec<usize> = vec![0; num_states];

    // Pre-allocate per-lane chunk buffers.
    let cap = input.len().div_ceil(num_states);
    let mut lane_chunks: Vec<Vec<(u32, u32)>> =
        (0..num_states).map(|_| vec![(0u32, 0u32); cap]).collect();
    let mut lane_counts = vec![0usize; num_states];

    // Count how many symbols each lane will process, to fill chunks
    // at the correct indices when processing in reverse.
    for i in 0..input.len() {
        let lane = i % num_states;
        lane_counts[lane] += 1;
    }

    // Cursor per lane, starts at end, decrements as we process in reverse.
    let mut lane_cursors: Vec<usize> = lane_counts.clone();

    // Encode in reverse: for each symbol, find the encode mapping for
    // this lane's current state, record the bit-chunk, transition state.
    for (i, &byte) in input.iter().enumerate().rev() {
        let lane = i % num_states;
        let s = byte as usize;
        let mapping = table.encode_tables[s].find(states[lane]);
        let value = states[lane] as u32 - mapping.base as u32;
        lane_cursors[lane] -= 1;
        lane_chunks[lane][lane_cursors[lane]] = (value, mapping.bits as u32);
        states[lane] = mapping.compressed_state as usize;
    }

    // Write each lane's chunks into its own BitWriter (forward order).
    let mut results = Vec::with_capacity(num_states);
    for lane in 0..num_states {
        let mut writer = BitWriter::new();
        for &(value, nb_bits) in lane_chunks[lane].iter().take(lane_counts[lane]) {
            writer.write_bits(value, nb_bits);
        }
        let (bitstream, total_bits) = writer.finish();
        results.push((bitstream, states[lane] as u16, total_bits));
    }

    results
}

/// Decode N interleaved FSE streams.
///
/// Symbol `i` is decoded from stream `i % num_states`. Each stream is
/// independent, enabling N-way parallelism.
fn fse_decode_interleaved(
    streams: &[(Vec<u8>, u16, u32)],
    table: &FseTable,
    original_len: usize,
) -> PzResult<Vec<u8>> {
    let num_states = streams.len();
    if num_states == 0 {
        return Err(PzError::InvalidInput);
    }

    // Fast path: 4-way batched decode.
    if num_states == 4 {
        let bitstreams: [&[u8]; 4] = [&streams[0].0, &streams[1].0, &streams[2].0, &streams[3].0];
        let initial_states: [u16; 4] = [streams[0].1, streams[1].1, streams[2].1, streams[3].1];
        return crate::simd::fse_decode_4way(
            &bitstreams,
            &initial_states,
            &table.decode_table,
            table.table_size,
            original_len,
        )
        .ok_or(PzError::InvalidInput);
    }

    // Generic N-way path.

    // Initialize per-stream readers and states.
    let mut readers: Vec<BitReader> = streams
        .iter()
        .map(|(bs, _, _)| BitReader::new(bs))
        .collect();
    let mut states: Vec<usize> = streams.iter().map(|(_, st, _)| *st as usize).collect();

    let mut output = Vec::with_capacity(original_len);

    for i in 0..original_len {
        let lane = i % num_states;

        if states[lane] >= table.table_size {
            return Err(PzError::InvalidInput);
        }
        let entry = &table.decode_table[states[lane]];
        output.push(entry.symbol);
        let bits = readers[lane].read_bits(entry.bits as u32);
        states[lane] = entry.next_state_base as usize + bits as usize;
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Public API — interleaved N-way
// ---------------------------------------------------------------------------

/// Encode data using 4-way interleaved FSE (default).
pub fn encode_interleaved(input: &[u8]) -> Vec<u8> {
    encode_interleaved_n(input, DEFAULT_INTERLEAVE, DEFAULT_ACCURACY_LOG)
}

/// Encode data using N-way interleaved FSE with configurable parameters.
///
/// `num_states`: number of interleaved FSE states (typically 4).
/// `accuracy_log`: table accuracy (5..12).
pub fn encode_interleaved_n(input: &[u8], num_states: usize, accuracy_log: u8) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let num_states = num_states.max(1);
    let accuracy_log = accuracy_log.clamp(MIN_ACCURACY_LOG, MAX_ACCURACY_LOG);

    let mut freq = FrequencyTable::new();
    freq.count(input);

    let mut al = accuracy_log;
    while (1u32 << al) < freq.used {
        al += 1;
        if al > MAX_ACCURACY_LOG {
            break;
        }
    }

    let norm = normalize_frequencies(&freq, al).expect("valid non-empty input");
    let table = FseTable::from_normalized(&norm);
    let stream_results = fse_encode_interleaved(input, &table, num_states);

    // Serialize interleaved format:
    // [accuracy_log: u8] [freq_table: 512B] [num_states: u8]
    // per-state: [initial_state: u16] [total_bits: u32] [bitstream_len: u32] [bitstream_data]
    let total_bitstream_bytes: usize = stream_results.iter().map(|(bs, _, _)| bs.len()).sum();
    let header_size = 1 + FREQ_TABLE_BYTES + 1 + num_states * (2 + 4 + 4);
    let mut output = Vec::with_capacity(header_size + total_bitstream_bytes);

    output.push(al);
    for &f in &norm.freq {
        output.extend_from_slice(&f.to_le_bytes());
    }
    output.push(num_states as u8);

    for (bitstream, initial_state, total_bits) in &stream_results {
        output.extend_from_slice(&initial_state.to_le_bytes());
        output.extend_from_slice(&total_bits.to_le_bytes());
        output.extend_from_slice(&(bitstream.len() as u32).to_le_bytes());
        output.extend_from_slice(bitstream);
    }

    output
}

/// Decode N-way interleaved FSE-encoded data.
pub fn decode_interleaved(input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
    if original_len == 0 {
        return Ok(Vec::new());
    }

    // Minimum header: accuracy_log(1) + freq_table(512) + num_states(1)
    let min_header = 1 + FREQ_TABLE_BYTES + 1;
    if input.len() < min_header {
        return Err(PzError::InvalidInput);
    }

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

    let table_size = 1u32 << accuracy_log;
    let sum: u32 = norm_freq.iter().map(|&f| f as u32).sum();
    if sum != table_size {
        return Err(PzError::InvalidInput);
    }

    let norm = NormalizedFreqs {
        freq: norm_freq,
        accuracy_log,
    };

    let pos = 1 + FREQ_TABLE_BYTES;
    let num_states = input[pos] as usize;
    if num_states == 0 {
        return Err(PzError::InvalidInput);
    }

    let mut cursor = pos + 1;

    // Read per-stream metadata and bitstreams.
    let mut streams = Vec::with_capacity(num_states);
    for _ in 0..num_states {
        if input.len() < cursor + 2 + 4 + 4 {
            return Err(PzError::InvalidInput);
        }
        let initial_state = u16::from_le_bytes([input[cursor], input[cursor + 1]]);
        cursor += 2;
        let total_bits = u32::from_le_bytes([
            input[cursor],
            input[cursor + 1],
            input[cursor + 2],
            input[cursor + 3],
        ]);
        cursor += 4;
        let bitstream_len = u32::from_le_bytes([
            input[cursor],
            input[cursor + 1],
            input[cursor + 2],
            input[cursor + 3],
        ]) as usize;
        cursor += 4;

        if input.len() < cursor + bitstream_len {
            return Err(PzError::InvalidInput);
        }
        let bitstream = input[cursor..cursor + bitstream_len].to_vec();
        cursor += bitstream_len;

        streams.push((bitstream, initial_state, total_bits));
    }

    let table = FseTable::from_normalized(&norm);

    // Handle single-symbol case: all streams have total_bits == 0.
    if streams.iter().all(|(_, _, tb)| *tb == 0) && original_len > 0 {
        let state = streams[0].1 as usize;
        if state >= table.table_size {
            return Err(PzError::InvalidInput);
        }
        let entry = &table.decode_table[state];
        return Ok(vec![entry.symbol; original_len]);
    }

    fse_decode_interleaved(&streams, &table, original_len)
}

/// Decode N-way interleaved FSE-encoded data into a pre-allocated buffer.
pub fn decode_interleaved_to_buf(
    input: &[u8],
    original_len: usize,
    output: &mut [u8],
) -> PzResult<usize> {
    if original_len == 0 {
        return Ok(0);
    }
    if output.len() < original_len {
        return Err(PzError::BufferTooSmall);
    }
    let decoded = decode_interleaved(input, original_len)?;
    output[..original_len].copy_from_slice(&decoded);
    Ok(original_len)
}

#[cfg(test)]
mod tests;
