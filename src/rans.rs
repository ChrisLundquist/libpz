/// Range ANS (rANS) encoder/decoder.
///
/// Implements streaming rANS, a fast entropy coder using arithmetic
/// (multiply + shift) state transitions instead of table lookups.
/// rANS approaches Shannon entropy (like arithmetic/range coding) but
/// decodes faster: the hot path is a single multiply-add with one
/// predictable branch for renormalization.
///
/// # Why rANS
///
/// Compared to the library's other entropy coders:
///
/// | Property          | Huffman | FSE (tANS) | rANS       |
/// |-------------------|---------|------------|------------|
/// | Decode operation  | Bit-level tree walk | Table lookup | Multiply + lookup |
/// | I/O granularity   | Bits    | Bits       | 16-bit words |
/// | Branch predict    | Poor    | Good       | Good       |
/// | State independence| N/A     | Awkward    | Interleave N states |
/// | GPU shared memory | Large trees | Large tables | Small freq tables |
///
/// The key unlock is **interleaving**: maintain N independent rANS states
/// (N=4 for SSE2, N=8 for AVX2, N=32+ for GPU). Symbols are assigned
/// round-robin across states, so all N decode chains run in parallel
/// with zero data dependencies between them.
///
/// # Variants
///
/// - [`encode`] / [`decode`]: Single-stream rANS (reference implementation).
/// - [`encode_interleaved`] / [`decode_interleaved`]: N-way interleaved rANS
///   for SIMD/GPU parallelism. Default N=4.
///
/// # Format
///
/// **Single-stream:**
/// ```text
/// [scale_bits: u8] [freq_table: 256 × u16 LE] [final_state: u32 LE]
/// [num_words: u32 LE] [words: num_words × u16 LE]
/// ```
///
/// **Interleaved N-way:**
/// ```text
/// [scale_bits: u8] [freq_table: 256 × u16 LE] [num_states: u8]
/// [final_states: N × u32 LE] [num_words: N × u32 LE]
/// [stream_0_words] [stream_1_words] ... [stream_N-1_words]
/// ```
use crate::frequency::FrequencyTable;
use crate::{PzError, PzResult};

/// Default scale bits (frequency table sums to 1 << 12 = 4096).
///
/// 12-bit precision is the sweet spot for rANS on byte data: close to
/// arithmetic coding quality (~0.01-0.03 bits/byte above Shannon) with
/// fast single-multiply decode. Higher values give diminishing returns
/// while increasing the cum2sym lookup table size.
pub const DEFAULT_SCALE_BITS: u8 = 12;

/// Minimum supported scale bits. Below this, frequency resolution is
/// too coarse for reasonable compression of byte data.
pub const MIN_SCALE_BITS: u8 = 9;

/// Maximum supported scale bits.
pub const MAX_SCALE_BITS: u8 = 14;

/// Number of symbols in the byte alphabet.
const NUM_SYMBOLS: usize = 256;

/// Lower bound of the normalized rANS state.
///
/// State invariant: after each encode/decode step, state ∈ [RANS_L, RANS_L << IO_BITS).
/// With RANS_L = 2^16 and IO_BITS = 16, state ∈ [2^16, 2^32).
const RANS_L: u32 = 1 << 16;

/// I/O granularity: stream 16-bit words (not individual bits).
/// Word-aligned I/O is what makes rANS GPU/SIMD friendly.
const IO_BITS: u32 = 16;

/// Default number of interleaved states.
pub const DEFAULT_INTERLEAVE: usize = 4;

/// Header size for single-stream:
/// scale_bits(1) + freq_table(512) + final_state(4) + num_words(4) = 521
const HEADER_SIZE: usize = 1 + NUM_SYMBOLS * 2 + 4 + 4;

// ---------------------------------------------------------------------------
// Normalized frequency table
// ---------------------------------------------------------------------------

/// Normalized frequency table with cumulative frequencies.
///
/// Frequencies sum to exactly `1 << scale_bits`. The cumulative table
/// enables O(1) symbol lookup during decode via a direct-index array.
#[derive(Debug, Clone, PartialEq)]
struct NormalizedFreqs {
    /// Normalized frequency for each symbol. Sum = 1 << scale_bits.
    freq: [u16; NUM_SYMBOLS],
    /// Cumulative frequency: cum[i] = sum of freq[0..i].
    cum: [u16; NUM_SYMBOLS],
    /// The scale bits. table_size = 1 << scale_bits.
    scale_bits: u8,
}

/// Normalize raw frequencies so they sum to exactly `1 << scale_bits`.
///
/// Every symbol with a nonzero raw count is guaranteed at least 1 in the
/// normalized table. Rounding remainder is distributed to the symbols
/// with the largest raw counts.
fn normalize_frequencies(raw: &FrequencyTable, scale_bits: u8) -> PzResult<NormalizedFreqs> {
    let table_size = 1u32 << scale_bits;
    let total = raw.total;

    if total == 0 || raw.used == 0 {
        return Err(PzError::InvalidInput);
    }

    // Single symbol: it gets the entire table.
    if raw.used == 1 {
        let mut freq = [0u16; NUM_SYMBOLS];
        let mut cum = [0u16; NUM_SYMBOLS];
        let mut cumulative = 0u16;
        for (i, &count) in raw.byte.iter().enumerate() {
            cum[i] = cumulative;
            if count > 0 && cumulative == 0 {
                freq[i] = table_size as u16;
            }
            cumulative += freq[i];
        }
        return Ok(NormalizedFreqs {
            freq,
            cum,
            scale_bits,
        });
    }

    // Need table_size >= num_present_symbols (each needs >= 1 slot).
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

    // Build cumulative frequency table.
    let mut cum = [0u16; NUM_SYMBOLS];
    let mut cumulative = 0u16;
    for i in 0..NUM_SYMBOLS {
        cum[i] = cumulative;
        cumulative += norm_freq[i];
    }

    debug_assert_eq!(cumulative as u32, table_size);

    Ok(NormalizedFreqs {
        freq: norm_freq,
        cum,
        scale_bits,
    })
}

// ---------------------------------------------------------------------------
// Symbol lookup table (decode acceleration)
// ---------------------------------------------------------------------------

/// Build a direct-index lookup table for O(1) symbol resolution during decode.
///
/// Given the low `scale_bits` bits of the rANS state (which equal the
/// cumulative frequency slot), this table maps directly to the symbol.
/// Size: `1 << scale_bits` bytes.
fn build_symbol_lookup(norm: &NormalizedFreqs) -> Vec<u8> {
    let table_size = 1usize << norm.scale_bits;
    let mut lookup = vec![0u8; table_size];

    for sym in 0..NUM_SYMBOLS {
        let freq = norm.freq[sym] as usize;
        let start = norm.cum[sym] as usize;
        for entry in lookup.iter_mut().skip(start).take(freq) {
            *entry = sym as u8;
        }
    }

    lookup
}

// ---------------------------------------------------------------------------
// Division helpers
// ---------------------------------------------------------------------------

/// Pre-computed reciprocals for division-free rANS encoding.
///
/// For each symbol frequency f, stores `rcp = ceil(2^32 / f)` so that
/// division can be approximated by `q = hi32(x * rcp)` with a single
/// correction step. This is critical for GPU kernels where hardware
/// division is 10-30x slower than multiply.
struct ReciprocalTable {
    rcp: [u32; NUM_SYMBOLS],
}

impl ReciprocalTable {
    fn from_normalized(norm: &NormalizedFreqs) -> Self {
        let mut rcp = [0u32; NUM_SYMBOLS];
        for (i, &f) in norm.freq.iter().enumerate() {
            if f > 0 {
                // rcp = floor(2^32 / freq)
                // Using floor avoids overestimating the quotient, so only a
                // single upward correction is ever needed.
                rcp[i] = ((1u64 << 32) / f as u64) as u32;
            }
        }
        ReciprocalTable { rcp }
    }
}

/// Division via reciprocal multiply: (x / freq, x % freq).
///
/// Uses a precomputed reciprocal to avoid hardware division.
/// The reciprocal is floor(2^32 / freq), so the estimate q may be
/// too low by at most 1. A single correction step suffices.
///
/// Special case: freq=1 has rcp=0 (2^32 doesn't fit in u32),
/// but division by 1 is trivial.
#[inline]
fn rans_div_rcp(x: u32, freq: u32, rcp: u32) -> (u32, u32) {
    if freq == 1 {
        return (x, 0);
    }
    // q = floor(x * floor(2^32/freq) / 2^32)
    // This can underestimate the true quotient by at most 1.
    let q = ((x as u64 * rcp as u64) >> 32) as u32;
    let r = x - q * freq;
    // Correct if remainder is still >= freq (quotient was 1 too low)
    if r >= freq {
        (q + 1, r - freq)
    } else {
        (q, r)
    }
}

// ---------------------------------------------------------------------------
// Core rANS encode / decode (single-stream)
// ---------------------------------------------------------------------------

/// Encode input using rANS. Returns (word_stream, final_state).
///
/// Processes symbols in **reverse** order (last symbol first).
/// Words are streamed out during normalization (16 bits at a time).
/// The decoder reads words forward and emits symbols in forward order.
fn rans_encode_internal(input: &[u8], norm: &NormalizedFreqs) -> (Vec<u16>, u32) {
    let scale_bits = norm.scale_bits as u32;
    let rcp_table = ReciprocalTable::from_normalized(norm);
    let mut state: u32 = RANS_L;
    let mut words: Vec<u16> = Vec::with_capacity(input.len());

    for &byte in input.iter().rev() {
        let s = byte as usize;
        let freq = norm.freq[s] as u32;
        let cum = norm.cum[s] as u32;

        // Renormalize: output low 16 bits until state fits.
        // x_max = ((RANS_L >> scale_bits) << IO_BITS) * freq
        // Compute in u64 to avoid overflow (can reach 2^32 for max freq).
        let x_max = ((RANS_L as u64 >> scale_bits) << IO_BITS) * freq as u64;
        while (state as u64) >= x_max {
            words.push(state as u16);
            state >>= IO_BITS;
        }

        // Encode: state = (state / freq) << scale_bits + state % freq + cum
        let (q, r) = rans_div_rcp(state, freq, rcp_table.rcp[s]);
        state = (q << scale_bits) + r + cum;
    }

    // Reverse so decoder reads in forward order.
    words.reverse();

    (words, state)
}

/// Decode rANS-encoded word stream. Returns the decoded byte sequence.
///
/// The decode hot path per symbol is:
/// ```text
/// slot  = state & mask          // extract cumulative frequency slot
/// sym   = cum2sym[slot]         // O(1) table lookup
/// state = freq[sym] * (state >> scale_bits) + slot - cum[sym]  // multiply-add
/// if state < RANS_L: state = (state << 16) | read_u16()        // renormalize
/// ```
/// No divisions. One predictable branch. Word-aligned I/O.
fn rans_decode_internal(
    words: &[u16],
    initial_state: u32,
    norm: &NormalizedFreqs,
    lookup: &[u8],
    original_len: usize,
) -> PzResult<Vec<u8>> {
    let scale_bits = norm.scale_bits as u32;
    let scale_mask = (1u32 << scale_bits) - 1;
    let mut state = initial_state;
    let mut word_pos = 0;
    let mut output = Vec::with_capacity(original_len);

    for _ in 0..original_len {
        // Decode symbol
        let slot = state & scale_mask;
        if slot as usize >= lookup.len() {
            return Err(PzError::InvalidInput);
        }
        let s = lookup[slot as usize];
        let freq = norm.freq[s as usize] as u32;
        let cum = norm.cum[s as usize] as u32;

        // Advance state: multiply-add (no division)
        state = freq * (state >> scale_bits) + slot - cum;

        // Renormalize: read one 16-bit word if state dropped below RANS_L.
        // With RANS_L = 2^16 and IO_BITS = 16, a single step always suffices:
        // after decode, state >= 0 and state < 2^32; shifting left by 16 and
        // OR-ing a 16-bit word guarantees state >= 2^16 = RANS_L.
        if state < RANS_L && word_pos < words.len() {
            state = (state << IO_BITS) | words[word_pos] as u32;
            word_pos += 1;
        }

        output.push(s);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Interleaved N-way rANS encode / decode
// ---------------------------------------------------------------------------

/// Encode input using N interleaved rANS streams.
///
/// Symbol `i` is processed by state `i % num_states`. Each state produces
/// its own independent word stream. This enables N-way SIMD parallelism
/// on decode: all N states can be advanced simultaneously with zero data
/// dependencies.
///
/// Returns (per-stream word vectors, per-stream final states).
fn rans_encode_interleaved(
    input: &[u8],
    norm: &NormalizedFreqs,
    num_states: usize,
) -> (Vec<Vec<u16>>, Vec<u32>) {
    let scale_bits = norm.scale_bits as u32;
    let rcp_table = ReciprocalTable::from_normalized(norm);

    let mut states = vec![RANS_L; num_states];
    let mut word_streams: Vec<Vec<u16>> =
        vec![Vec::with_capacity(input.len() / num_states); num_states];

    // Assign symbols to states round-robin. Process in reverse.
    for (i, &byte) in input.iter().enumerate().rev() {
        let lane = i % num_states;
        let s = byte as usize;
        let freq = norm.freq[s] as u32;
        let cum = norm.cum[s] as u32;

        // Renormalize this lane's state (u64 to avoid overflow)
        let x_max = ((RANS_L as u64 >> scale_bits) << IO_BITS) * freq as u64;
        while (states[lane] as u64) >= x_max {
            word_streams[lane].push(states[lane] as u16);
            states[lane] >>= IO_BITS;
        }

        // Encode
        let (q, r) = rans_div_rcp(states[lane], freq, rcp_table.rcp[s]);
        states[lane] = (q << scale_bits) + r + cum;
    }

    // Reverse each stream for forward reading by decoder.
    for stream in &mut word_streams {
        stream.reverse();
    }

    (word_streams, states)
}

/// Decode N interleaved rANS streams.
///
/// Symbol `i` is decoded from state `i % num_states`, enabling N-way
/// SIMD parallelism. Each state reads from its own word stream.
///
/// When `num_states == 4`, dispatches to the batched 4-way decode path
/// which processes all 4 lanes per iteration for better register usage
/// and reduced loop overhead.
fn rans_decode_interleaved(
    word_streams: &[&[u16]],
    initial_states: &[u32],
    norm: &NormalizedFreqs,
    lookup: &[u8],
    original_len: usize,
) -> PzResult<Vec<u8>> {
    let num_states = initial_states.len();
    if word_streams.len() != num_states || num_states == 0 {
        return Err(PzError::InvalidInput);
    }

    // Fast path: 4-way batched decode
    if num_states == 4 {
        let streams_arr: [&[u16]; 4] = [
            word_streams[0],
            word_streams[1],
            word_streams[2],
            word_streams[3],
        ];
        let states_arr: [u32; 4] = [
            initial_states[0],
            initial_states[1],
            initial_states[2],
            initial_states[3],
        ];
        return crate::simd::rans_decode_4way(
            &streams_arr,
            &states_arr,
            &norm.freq,
            &norm.cum,
            lookup,
            norm.scale_bits as u32,
            original_len,
        )
        .ok_or(PzError::InvalidInput);
    }

    // Generic N-way path
    let scale_bits = norm.scale_bits as u32;
    let scale_mask = (1u32 << scale_bits) - 1;

    let mut states: Vec<u32> = initial_states.to_vec();
    let mut word_positions: Vec<usize> = vec![0; num_states];
    let mut output = Vec::with_capacity(original_len);

    for i in 0..original_len {
        let lane = i % num_states;

        // Decode symbol from this lane's state
        let slot = states[lane] & scale_mask;
        if slot as usize >= lookup.len() {
            return Err(PzError::InvalidInput);
        }
        let s = lookup[slot as usize];
        let freq = norm.freq[s as usize] as u32;
        let cum = norm.cum[s as usize] as u32;

        // Advance state
        states[lane] = freq * (states[lane] >> scale_bits) + slot - cum;

        // Renormalize: single step suffices (32-bit state, 16-bit I/O)
        if states[lane] < RANS_L && word_positions[lane] < word_streams[lane].len() {
            states[lane] =
                (states[lane] << IO_BITS) | word_streams[lane][word_positions[lane]] as u32;
            word_positions[lane] += 1;
        }

        output.push(s);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Result of zero-copy word slice access: either borrowed or owned.
enum WordSlice<'a> {
    Borrowed(&'a [u16]),
    Owned(Vec<u16>),
}

impl<'a> std::ops::Deref for WordSlice<'a> {
    type Target = [u16];
    #[inline]
    fn deref(&self) -> &[u16] {
        match self {
            WordSlice::Borrowed(s) => s,
            WordSlice::Owned(v) => v,
        }
    }
}

/// Reinterpret a byte slice as a slice of little-endian u16 values.
///
/// On little-endian platforms (x86_64, aarch64) this is a zero-copy pointer
/// cast when alignment permits, returning a borrowed slice. Falls back to
/// byte-at-a-time parsing only when the slice is misaligned or on big-endian.
#[inline]
fn bytes_as_u16_le(data: &[u8], count: usize) -> WordSlice<'_> {
    debug_assert!(data.len() >= count * 2);

    #[cfg(target_endian = "little")]
    {
        // Fast path: try zero-copy via align_to.
        // SAFETY: u16 from LE bytes is valid for all bit patterns, and
        // align_to returns the maximal aligned middle slice.
        let (prefix, aligned, suffix) = unsafe { data[..count * 2].align_to::<u16>() };

        if prefix.is_empty() && suffix.is_empty() && aligned.len() == count {
            return WordSlice::Borrowed(aligned);
        }
    }

    // Fallback: byte-at-a-time parsing
    let mut words = Vec::with_capacity(count);
    for i in 0..count {
        let off = i * 2;
        words.push(u16::from_le_bytes([data[off], data[off + 1]]));
    }
    WordSlice::Owned(words)
}

/// Serialize a slice of u16 values as little-endian bytes in bulk.
///
/// On little-endian platforms, this is a single memcpy when aligned.
#[inline]
fn serialize_u16_le_bulk(words: &[u16], output: &mut Vec<u8>) {
    #[cfg(target_endian = "little")]
    {
        let byte_len = words.len() * 2;
        let start = output.len();
        output.reserve(byte_len);
        // SAFETY: we just reserved enough space, and u16→u8 reinterpret is
        // always valid on little-endian. We set the length after the copy.
        unsafe {
            let src = words.as_ptr() as *const u8;
            let dst = output.as_mut_ptr().add(start);
            std::ptr::copy_nonoverlapping(src, dst, byte_len);
            output.set_len(start + byte_len);
        }
    }
    #[cfg(target_endian = "big")]
    {
        for &w in words {
            output.extend_from_slice(&w.to_le_bytes());
        }
    }
}

/// Serialize a normalized frequency table (256 × u16 LE).
fn serialize_freq_table(norm: &NormalizedFreqs, output: &mut Vec<u8>) {
    for &f in &norm.freq {
        output.extend_from_slice(&f.to_le_bytes());
    }
}

/// Deserialize a normalized frequency table and validate sum.
fn deserialize_freq_table(input: &[u8], scale_bits: u8) -> PzResult<NormalizedFreqs> {
    if input.len() < NUM_SYMBOLS * 2 {
        return Err(PzError::InvalidInput);
    }

    let mut freq = [0u16; NUM_SYMBOLS];
    for (i, f) in freq.iter_mut().enumerate() {
        let offset = i * 2;
        *f = u16::from_le_bytes([input[offset], input[offset + 1]]);
    }

    let table_size = 1u32 << scale_bits;
    let sum: u32 = freq.iter().map(|&f| f as u32).sum();
    if sum != table_size {
        return Err(PzError::InvalidInput);
    }

    // Build cumulative table.
    let mut cum = [0u16; NUM_SYMBOLS];
    let mut cumulative = 0u16;
    for i in 0..NUM_SYMBOLS {
        cum[i] = cumulative;
        cumulative += freq[i];
    }

    Ok(NormalizedFreqs {
        freq,
        cum,
        scale_bits,
    })
}

// ---------------------------------------------------------------------------
// Public API — single-stream
// ---------------------------------------------------------------------------

/// Encode data using rANS with the default scale (12-bit precision).
///
/// Returns self-contained compressed data including the serialized
/// frequency table. The original length must be known by the decoder
/// (stored externally, consistent with other entropy coders in libpz).
pub fn encode(input: &[u8]) -> Vec<u8> {
    encode_with_scale(input, DEFAULT_SCALE_BITS)
}

/// Encode data using rANS with a specific scale (9..14 bits).
///
/// Higher `scale_bits` gives better compression (finer frequency
/// resolution) but larger cum2sym lookup table on decode.
pub fn encode_with_scale(input: &[u8], scale_bits: u8) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let scale_bits = scale_bits.clamp(MIN_SCALE_BITS, MAX_SCALE_BITS);

    let mut freq = FrequencyTable::new();
    freq.count(input);

    // Bump scale_bits if too many distinct symbols for the table size.
    let mut sb = scale_bits;
    while (1u32 << sb) < freq.used {
        sb += 1;
        if sb > MAX_SCALE_BITS {
            break;
        }
    }

    let norm = normalize_frequencies(&freq, sb).expect("valid non-empty input");
    let (words, final_state) = rans_encode_internal(input, &norm);

    // Serialize: header + word stream.
    let mut output = Vec::with_capacity(HEADER_SIZE + words.len() * 2);
    output.push(sb);
    serialize_freq_table(&norm, &mut output);
    output.extend_from_slice(&final_state.to_le_bytes());
    output.extend_from_slice(&(words.len() as u32).to_le_bytes());
    serialize_u16_le_bulk(&words, &mut output);

    output
}

/// Decode rANS-encoded data.
///
/// `original_len` is the number of bytes in the original uncompressed data.
pub fn decode(input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
    if original_len == 0 {
        return Ok(Vec::new());
    }
    if input.len() < HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    let scale_bits = input[0];
    if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
        return Err(PzError::InvalidInput);
    }

    let norm = deserialize_freq_table(&input[1..], scale_bits)?;

    let header_end = 1 + NUM_SYMBOLS * 2;
    let initial_state = u32::from_le_bytes([
        input[header_end],
        input[header_end + 1],
        input[header_end + 2],
        input[header_end + 3],
    ]);
    let num_words = u32::from_le_bytes([
        input[header_end + 4],
        input[header_end + 5],
        input[header_end + 6],
        input[header_end + 7],
    ]) as usize;

    let words_start = header_end + 8;
    if input.len() < words_start + num_words * 2 {
        return Err(PzError::InvalidInput);
    }

    let words = bytes_as_u16_le(&input[words_start..], num_words);

    let lookup = build_symbol_lookup(&norm);
    rans_decode_internal(&words, initial_state, &norm, &lookup, original_len)
}

/// Decode rANS-encoded data into a pre-allocated buffer.
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
// Public API — interleaved N-way
// ---------------------------------------------------------------------------

/// Encode data using 4-way interleaved rANS (default).
pub fn encode_interleaved(input: &[u8]) -> Vec<u8> {
    encode_interleaved_n(input, DEFAULT_INTERLEAVE, DEFAULT_SCALE_BITS)
}

/// Encode data using N-way interleaved rANS with configurable parameters.
///
/// `num_states`: number of interleaved rANS states (typically 4 or 8).
/// `scale_bits`: frequency precision (9..14).
pub fn encode_interleaved_n(input: &[u8], num_states: usize, scale_bits: u8) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let num_states = num_states.max(1);
    let scale_bits = scale_bits.clamp(MIN_SCALE_BITS, MAX_SCALE_BITS);

    let mut freq = FrequencyTable::new();
    freq.count(input);

    let mut sb = scale_bits;
    while (1u32 << sb) < freq.used {
        sb += 1;
        if sb > MAX_SCALE_BITS {
            break;
        }
    }

    let norm = normalize_frequencies(&freq, sb).expect("valid non-empty input");
    let (word_streams, final_states) = rans_encode_interleaved(input, &norm, num_states);

    // Serialize interleaved format:
    // [scale_bits: u8] [freq_table: 512] [num_states: u8]
    // [final_states: N × u32 LE] [num_words: N × u32 LE]
    // [stream_0_words] [stream_1_words] ...
    let total_words: usize = word_streams.iter().map(|s| s.len()).sum();
    let header_size = 1 + NUM_SYMBOLS * 2 + 1 + num_states * 4 + num_states * 4;
    let mut output = Vec::with_capacity(header_size + total_words * 2);

    output.push(sb);
    serialize_freq_table(&norm, &mut output);
    output.push(num_states as u8);

    for &state in &final_states {
        output.extend_from_slice(&state.to_le_bytes());
    }
    for stream in &word_streams {
        output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
    }
    for stream in &word_streams {
        serialize_u16_le_bulk(stream, &mut output);
    }

    output
}

/// Decode N-way interleaved rANS-encoded data.
pub fn decode_interleaved(input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
    if original_len == 0 {
        return Ok(Vec::new());
    }

    // Minimum header: scale_bits(1) + freq_table(512) + num_states(1)
    if input.len() < 1 + NUM_SYMBOLS * 2 + 1 {
        return Err(PzError::InvalidInput);
    }

    let scale_bits = input[0];
    if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
        return Err(PzError::InvalidInput);
    }

    let norm = deserialize_freq_table(&input[1..], scale_bits)?;

    let pos = 1 + NUM_SYMBOLS * 2;
    let num_states = input[pos] as usize;
    if num_states == 0 {
        return Err(PzError::InvalidInput);
    }

    let mut cursor = pos + 1;

    // Read final states
    if input.len() < cursor + num_states * 4 {
        return Err(PzError::InvalidInput);
    }
    let mut initial_states = Vec::with_capacity(num_states);
    for _ in 0..num_states {
        let state = u32::from_le_bytes([
            input[cursor],
            input[cursor + 1],
            input[cursor + 2],
            input[cursor + 3],
        ]);
        initial_states.push(state);
        cursor += 4;
    }

    // Read word counts per stream
    if input.len() < cursor + num_states * 4 {
        return Err(PzError::InvalidInput);
    }
    let mut word_counts = Vec::with_capacity(num_states);
    for _ in 0..num_states {
        let count = u32::from_le_bytes([
            input[cursor],
            input[cursor + 1],
            input[cursor + 2],
            input[cursor + 3],
        ]) as usize;
        word_counts.push(count);
        cursor += 4;
    }

    // Read word streams (zero-copy when aligned on little-endian)
    let mut word_slices: Vec<WordSlice<'_>> = Vec::with_capacity(num_states);
    for &count in &word_counts {
        if input.len() < cursor + count * 2 {
            return Err(PzError::InvalidInput);
        }
        word_slices.push(bytes_as_u16_le(&input[cursor..], count));
        cursor += count * 2;
    }
    // Collect &[u16] references for the decode function
    let word_streams: Vec<&[u16]> = word_slices.iter().map(|ws| &**ws).collect();

    let lookup = build_symbol_lookup(&norm);
    rans_decode_interleaved(&word_streams, &initial_states, &norm, &lookup, original_len)
}

/// Decode N-way interleaved rANS-encoded data into a pre-allocated buffer.
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Frequency normalization ---

    #[test]
    fn test_normalize_frequencies_basic() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaabbc");
        let norm = normalize_frequencies(&freq, 12).unwrap();
        let sum: u32 = norm.freq.iter().map(|&f| f as u32).sum();
        assert_eq!(sum, 4096);
        assert!(norm.freq[b'a' as usize] > 0);
        assert!(norm.freq[b'b' as usize] > 0);
        assert!(norm.freq[b'c' as usize] > 0);
        // Cumulative must be monotonically non-decreasing
        for i in 1..NUM_SYMBOLS {
            assert!(norm.cum[i] >= norm.cum[i - 1]);
        }
    }

    #[test]
    fn test_normalize_single_symbol() {
        let mut freq = FrequencyTable::new();
        freq.count(&[42u8; 100]);
        let norm = normalize_frequencies(&freq, 12).unwrap();
        assert_eq!(norm.freq[42], 4096);
        assert_eq!(norm.cum[42], 0);
    }

    #[test]
    fn test_normalize_preserves_order() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaaaabbbcc");
        let norm = normalize_frequencies(&freq, 12).unwrap();
        assert!(norm.freq[b'a' as usize] > norm.freq[b'b' as usize]);
        assert!(norm.freq[b'b' as usize] > norm.freq[b'c' as usize]);
    }

    #[test]
    fn test_normalize_empty_fails() {
        let freq = FrequencyTable::new();
        assert_eq!(normalize_frequencies(&freq, 12), Err(PzError::InvalidInput));
    }

    // --- Symbol lookup ---

    #[test]
    fn test_symbol_lookup_coverage() {
        let mut freq = FrequencyTable::new();
        freq.count(b"aaabbc");
        let norm = normalize_frequencies(&freq, 12).unwrap();
        let lookup = build_symbol_lookup(&norm);
        assert_eq!(lookup.len(), 4096);
        // Every entry should map to a valid present symbol
        for &s in &lookup {
            assert!(norm.freq[s as usize] > 0);
        }
    }

    // --- Division helper ---

    #[test]
    fn test_rans_div_rcp_correctness() {
        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        // Test across all supported scale_bits
        for sb in MIN_SCALE_BITS..=MAX_SCALE_BITS {
            let mut freq = FrequencyTable::new();
            freq.count(&input);
            let norm = normalize_frequencies(&freq, sb).unwrap();
            let rcp_table = ReciprocalTable::from_normalized(&norm);

            for sym in 0..NUM_SYMBOLS {
                if norm.freq[sym] == 0 {
                    continue;
                }
                let f = norm.freq[sym] as u32;
                let rcp = rcp_table.rcp[sym];
                // Test across the full rANS state range
                let test_vals: Vec<u32> = vec![
                    1,
                    f,
                    f + 1,
                    RANS_L - 1,
                    RANS_L,
                    RANS_L + 1,
                    RANS_L * 2,
                    0xFFFF,
                    0x1_0000,
                    0x7FFF_FFFF,
                    0xFFFF_FFFE,
                    0xFFFF_FFFF,
                ];
                for x in test_vals {
                    if x == 0 {
                        continue;
                    }
                    let (q, r) = rans_div_rcp(x, f, rcp);
                    assert_eq!(
                        (q, r),
                        (x / f, x % f),
                        "rans_div_rcp wrong for x={x}, f={f}, rcp={rcp}, sb={sb}"
                    );
                }
            }
        }
    }

    // --- Single-stream round-trip tests ---

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

    // --- Scale bits variants ---

    #[test]
    fn test_all_scale_bits() {
        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        for sb in MIN_SCALE_BITS..=MAX_SCALE_BITS {
            let encoded = encode_with_scale(&input, sb);
            let decoded = decode(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input, "failed at scale_bits={}", sb);
        }
    }

    // --- Error handling ---

    #[test]
    fn test_decode_too_short() {
        let result = decode(&[0u8; 10], 5);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_invalid_scale_bits() {
        let mut bad = vec![0u8; HEADER_SIZE + 10];
        bad[0] = 15; // > MAX_SCALE_BITS
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

    // --- Interleaved round-trip tests ---

    #[test]
    fn test_interleaved_empty() {
        assert_eq!(encode_interleaved(&[]), Vec::<u8>::new());
        assert_eq!(decode_interleaved(&[], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_interleaved_single_byte() {
        let input = &[42u8];
        let encoded = encode_interleaved(input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_hello() {
        let input = b"hello, world!";
        let encoded = encode_interleaved(input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_repeated() {
        let input = vec![b'x'; 500];
        let encoded = encode_interleaved(&input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_all_bytes() {
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode_interleaved(&input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_longer_text() {
        let input =
            b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
        let encoded = encode_interleaved(input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_binary() {
        let input: Vec<u8> = (0..1000).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded = encode_interleaved(&input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_8way() {
        let input: Vec<u8> = (0..2000).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded = encode_interleaved_n(&input, 8, DEFAULT_SCALE_BITS);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_1way_matches_single() {
        // 1-way interleaved should produce the same decoded output as single-stream
        let input = b"the quick brown fox jumps over the lazy dog";
        let decoded_single = {
            let encoded = encode(input);
            decode(&encoded, input.len()).unwrap()
        };
        let decoded_interleaved = {
            let encoded = encode_interleaved_n(input, 1, DEFAULT_SCALE_BITS);
            decode_interleaved(&encoded, input.len()).unwrap()
        };
        assert_eq!(decoded_single, decoded_interleaved);
    }

    #[test]
    fn test_interleaved_to_buf() {
        let input = b"hello, world!";
        let encoded = encode_interleaved(input);
        let mut buf = vec![0u8; 100];
        let size = decode_interleaved_to_buf(&encoded, input.len(), &mut buf).unwrap();
        assert_eq!(&buf[..size], input);
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

    #[test]
    fn test_interleaved_medium() {
        let mut input = Vec::new();
        for _ in 0..50 {
            input.extend(b"rANS interleaved encode/decode test data. ");
        }
        let encoded = encode_interleaved(&input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_interleaved_large() {
        let pattern: Vec<u8> = (0..=127).collect();
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend(&pattern);
        }
        let encoded = encode_interleaved(&input);
        let decoded = decode_interleaved(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_compression_vs_huffman_quality() {
        // rANS with 12-bit precision should compress at least as well as
        // a naive frequency header for skewed data
        let mut input = vec![b'a'; 1000];
        input.extend(vec![b'b'; 100]);
        input.extend(vec![b'c'; 10]);
        input.push(b'd');

        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
        // Should compress (accounting for 521-byte header overhead)
        assert!(
            encoded.len() < input.len(),
            "encoded {} bytes, expected < {}",
            encoded.len(),
            input.len()
        );
    }
}
