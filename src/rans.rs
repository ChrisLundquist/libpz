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
pub(crate) const NUM_SYMBOLS: usize = 256;

/// Lower bound of the normalized rANS state.
///
/// State invariant: after each encode/decode step, state ∈ [RANS_L, RANS_L << IO_BITS).
/// With RANS_L = 2^16 and IO_BITS = 16, state ∈ [2^16, 2^32).
pub(crate) const RANS_L: u32 = 1 << 16;

/// I/O granularity: stream 16-bit words (not individual bits).
/// Word-aligned I/O is what makes rANS GPU/SIMD friendly.
pub(crate) const IO_BITS: u32 = 16;

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
pub(crate) struct NormalizedFreqs {
    /// Normalized frequency for each symbol. Sum = 1 << scale_bits.
    pub(crate) freq: [u16; NUM_SYMBOLS],
    /// Cumulative frequency: cum[i] = sum of freq[0..i].
    pub(crate) cum: [u16; NUM_SYMBOLS],
    /// The scale bits. table_size = 1 << scale_bits.
    pub(crate) scale_bits: u8,
}

/// Normalize raw frequencies so they sum to exactly `1 << scale_bits`.
///
/// Every symbol with a nonzero raw count is guaranteed at least 1 in the
/// normalized table. Rounding remainder is distributed to the symbols
/// with the largest raw counts.
pub(crate) fn normalize_frequencies(
    raw: &FrequencyTable,
    scale_bits: u8,
) -> PzResult<NormalizedFreqs> {
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
pub(crate) fn build_symbol_lookup(norm: &NormalizedFreqs) -> Vec<u8> {
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
// Merged slot-indexed table (ryg_rans-style)
// ---------------------------------------------------------------------------

/// Merged slot-indexed decode table entry.
///
/// Combines frequency and bias into a single `u32`, eliminating the 2-hop
/// gather pattern (slot → symbol → freq/cum).  The decode step becomes:
///
/// ```text
/// slot = state & scale_mask
/// sym = slot2sym[slot]
/// freq = slot_table[slot].freq_bias & 0xFFFF
/// bias = slot_table[slot].freq_bias >> 16
/// new_state = freq * (state >> scale_bits) + bias
/// ```
///
/// Where `bias = slot - cum[sym]`, precomputed per slot.
#[derive(Copy, Clone)]
pub struct SlotEntry {
    /// Low 16 bits: frequency.  High 16 bits: bias (= slot position
    /// within this symbol's frequency range, i.e. `slot - cum[sym]`).
    pub freq_bias: u32,
}

/// Build the merged slot-indexed decode table.
///
/// Returns `(slot2sym, slot_table)` where:
/// - `slot2sym[slot]` = symbol byte (same as [`build_symbol_lookup`])
/// - `slot_table[slot].freq_bias` = `freq | (bias << 16)`
pub(crate) fn build_slot_table(norm: &NormalizedFreqs) -> (Vec<u8>, Vec<SlotEntry>) {
    let table_size = 1usize << norm.scale_bits;
    let mut slot2sym = vec![0u8; table_size];
    let mut slot_table = vec![SlotEntry { freq_bias: 0 }; table_size];

    for sym in 0..NUM_SYMBOLS {
        let freq = norm.freq[sym] as u32;
        let start = norm.cum[sym] as usize;
        for offset in 0..freq as usize {
            let slot = start + offset;
            slot2sym[slot] = sym as u8;
            slot_table[slot] = SlotEntry {
                freq_bias: freq | ((offset as u32) << 16),
            };
        }
    }

    (slot2sym, slot_table)
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
pub struct ReciprocalTable {
    pub rcp: [u32; NUM_SYMBOLS],
}

impl ReciprocalTable {
    pub(crate) fn from_normalized(norm: &NormalizedFreqs) -> Self {
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

    // Pre-allocate from the end: worst case is ~2 words per symbol
    // (when freq=1 at scale_bits=14). We fill backwards and truncate.
    let capacity = input.len() * 2;
    let mut words = vec![0u16; capacity];
    let mut cursor = capacity; // write position, decrements

    for &byte in input.iter().rev() {
        let s = byte as usize;
        let freq = norm.freq[s] as u32;
        let cum = norm.cum[s] as u32;

        // Renormalize: output low 16 bits until state fits.
        let x_max = ((RANS_L as u64 >> scale_bits) << IO_BITS) * freq as u64;
        while (state as u64) >= x_max {
            cursor -= 1;
            words[cursor] = state as u16;
            state >>= IO_BITS;
        }

        // Encode: state = (state / freq) << scale_bits + state % freq + cum
        let (q, r) = rans_div_rcp(state, freq, rcp_table.rcp[s]);
        state = (q << scale_bits) + r + cum;
    }

    // Return only the filled portion — already in forward order, no reverse needed.
    let result = words[cursor..].to_vec();
    (result, state)
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
/// Core single-stream rANS decode loop, writing directly into `output`.
///
/// Caller must ensure `output.len() >= original_len`.
fn rans_decode_to_slice(
    words: &[u16],
    initial_state: u32,
    norm: &NormalizedFreqs,
    lookup: &[u8],
    output: &mut [u8],
    original_len: usize,
) -> PzResult<()> {
    let scale_bits = norm.scale_bits as u32;
    let scale_mask = (1u32 << scale_bits) - 1;
    let mut state = initial_state;
    let mut word_pos = 0;

    for out in output.iter_mut().take(original_len) {
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

        *out = s;
    }

    Ok(())
}

/// Decode rANS-encoded word stream, returning a new Vec.
fn rans_decode_internal(
    words: &[u16],
    initial_state: u32,
    norm: &NormalizedFreqs,
    lookup: &[u8],
    original_len: usize,
) -> PzResult<Vec<u8>> {
    let mut output = vec![0u8; original_len];
    rans_decode_to_slice(
        words,
        initial_state,
        norm,
        lookup,
        &mut output,
        original_len,
    )?;
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
pub(crate) fn rans_encode_interleaved(
    input: &[u8],
    norm: &NormalizedFreqs,
    num_states: usize,
) -> (Vec<Vec<u16>>, Vec<u32>) {
    let scale_bits = norm.scale_bits as u32;
    let rcp_table = ReciprocalTable::from_normalized(norm);

    let mut states = vec![RANS_L; num_states];

    // Pre-allocate per-lane buffers and fill backwards to avoid reversals.
    let per_lane_cap = (input.len() / num_states) * 2 + 4;
    let mut word_bufs: Vec<Vec<u16>> = (0..num_states).map(|_| vec![0u16; per_lane_cap]).collect();
    let mut cursors: Vec<usize> = vec![per_lane_cap; num_states];

    // Assign symbols to states round-robin. Process in reverse.
    for (i, &byte) in input.iter().enumerate().rev() {
        let lane = i % num_states;
        let s = byte as usize;
        let freq = norm.freq[s] as u32;
        let cum = norm.cum[s] as u32;

        // Renormalize this lane's state (u64 to avoid overflow)
        let x_max = ((RANS_L as u64 >> scale_bits) << IO_BITS) * freq as u64;
        while (states[lane] as u64) >= x_max {
            cursors[lane] -= 1;
            word_bufs[lane][cursors[lane]] = states[lane] as u16;
            states[lane] >>= IO_BITS;
        }

        // Encode
        let (q, r) = rans_div_rcp(states[lane], freq, rcp_table.rcp[s]);
        states[lane] = (q << scale_bits) + r + cum;
    }

    // Extract the filled portions — already in forward order, no reverse needed.
    let word_streams: Vec<Vec<u16>> = word_bufs
        .into_iter()
        .zip(cursors.iter())
        .map(|(buf, &cursor)| buf[cursor..].to_vec())
        .collect();

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

        // Use merged slot-indexed table for faster decode (eliminates 2-hop
        // gather: slot→symbol→freq/cum → single slot→freq_bias lookup).
        let (slot2sym, slot_table) = build_slot_table(norm);
        return crate::simd::rans_decode_4way_slot(
            &streams_arr,
            &states_arr,
            &slot2sym,
            &slot_table,
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

/// Like [`rans_decode_interleaved`] but writes into a pre-allocated output buffer.
fn rans_decode_interleaved_into(
    word_streams: &[&[u16]],
    initial_states: &[u32],
    norm: &NormalizedFreqs,
    lookup: &[u8],
    original_len: usize,
    output: &mut [u8],
) -> PzResult<()> {
    let num_states = initial_states.len();
    if word_streams.len() != num_states || num_states == 0 {
        return Err(PzError::InvalidInput);
    }

    // Fast path: 4-way batched decode (note: _into variant not yet SSE2-optimized,
    // keeping scalar implementation for now)
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
        return crate::simd::rans_decode_4way_into(
            &streams_arr,
            &states_arr,
            &norm.freq,
            &norm.cum,
            lookup,
            norm.scale_bits as u32,
            original_len,
            output,
        )
        .ok_or(PzError::InvalidInput);
    }

    // Generic N-way path
    let scale_bits = norm.scale_bits as u32;
    let scale_mask = (1u32 << scale_bits) - 1;

    let mut states: Vec<u32> = initial_states.to_vec();
    let mut word_positions: Vec<usize> = vec![0; num_states];

    for (i, out) in output.iter_mut().enumerate() {
        let lane = i % num_states;
        let slot = states[lane] & scale_mask;
        if slot as usize >= lookup.len() {
            return Err(PzError::InvalidInput);
        }
        let s = lookup[slot as usize];
        let freq = norm.freq[s as usize] as u32;
        let cum = norm.cum[s as usize] as u32;

        states[lane] = freq * (states[lane] >> scale_bits) + slot - cum;

        if states[lane] < RANS_L && word_positions[lane] < word_streams[lane].len() {
            states[lane] =
                (states[lane] << IO_BITS) | word_streams[lane][word_positions[lane]] as u32;
            word_positions[lane] += 1;
        }

        *out = s;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Result of zero-copy word slice access: either borrowed or owned.
pub(crate) enum WordSlice<'a> {
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
pub(crate) fn bytes_as_u16_le(data: &[u8], count: usize) -> WordSlice<'_> {
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
pub(crate) fn serialize_u16_le_bulk(words: &[u16], output: &mut Vec<u8>) {
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
pub(crate) fn serialize_freq_table(norm: &NormalizedFreqs, output: &mut Vec<u8>) {
    for &f in &norm.freq {
        output.extend_from_slice(&f.to_le_bytes());
    }
}

/// Deserialize a normalized frequency table and validate sum.
pub(crate) fn deserialize_freq_table(input: &[u8], scale_bits: u8) -> PzResult<NormalizedFreqs> {
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

/// Parsed single-stream header: everything needed to start decoding.
struct SingleStreamHeader<'a> {
    norm: NormalizedFreqs,
    initial_state: u32,
    words: WordSlice<'a>,
}

/// Parse the single-stream header (shared by decode and decode_to_buf).
fn parse_single_stream_header(input: &[u8]) -> PzResult<SingleStreamHeader<'_>> {
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

    Ok(SingleStreamHeader {
        norm,
        initial_state,
        words,
    })
}

/// Decode rANS-encoded data.
///
/// `original_len` is the number of bytes in the original uncompressed data.
pub fn decode(input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
    if original_len == 0 {
        return Ok(Vec::new());
    }
    let hdr = parse_single_stream_header(input)?;
    let lookup = build_symbol_lookup(&hdr.norm);
    rans_decode_internal(
        &hdr.words,
        hdr.initial_state,
        &hdr.norm,
        &lookup,
        original_len,
    )
}

/// Decode rANS-encoded data into a pre-allocated buffer.
///
/// Decodes directly into the output buffer without intermediate allocation.
/// Returns the number of bytes written.
pub fn decode_to_buf(input: &[u8], original_len: usize, output: &mut [u8]) -> PzResult<usize> {
    if original_len == 0 {
        return Ok(0);
    }
    if output.len() < original_len {
        return Err(PzError::BufferTooSmall);
    }
    let hdr = parse_single_stream_header(input)?;
    let lookup = build_symbol_lookup(&hdr.norm);
    rans_decode_to_slice(
        &hdr.words,
        hdr.initial_state,
        &hdr.norm,
        &lookup,
        output,
        original_len,
    )?;
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
// Public API — chunked N-way
// ---------------------------------------------------------------------------

/// Encode data using chunked N-way interleaved rANS.
///
/// This splits the input into chunks of `chunk_size` bytes, and encodes
/// each chunk independently with N-way interleaved rANS. This allows for
/// block-level parallelism on decode, which is critical for GPU saturation.
///
/// The format shares a single frequency table across all chunks.
pub fn encode_chunked(
    input: &[u8],
    num_states: usize,
    scale_bits: u8,
    chunk_size: usize,
) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    const MAX_CHUNKED_NUM_STATES: usize = u8::MAX as usize;
    const MAX_CHUNKED_NUM_CHUNKS: usize = u16::MAX as usize;
    const MAX_CHUNKED_CHUNK_LEN: usize = u16::MAX as usize;

    let num_states = num_states.max(1);
    assert!(
        num_states <= MAX_CHUNKED_NUM_STATES,
        "encode_chunked: num_states {} exceeds on-wire u8 limit {}",
        num_states,
        MAX_CHUNKED_NUM_STATES
    );

    let scale_bits = scale_bits.clamp(MIN_SCALE_BITS, MAX_SCALE_BITS);
    let chunk_size = chunk_size.max(1);

    // Global frequency table for the whole input.
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

    // Encode each chunk.
    let chunks: Vec<_> = input.chunks(chunk_size).collect();
    assert!(
        chunks.len() <= MAX_CHUNKED_NUM_CHUNKS,
        "encode_chunked: chunk count {} exceeds on-wire u16 limit {}",
        chunks.len(),
        MAX_CHUNKED_NUM_CHUNKS
    );
    for chunk in &chunks {
        assert!(
            chunk.len() <= MAX_CHUNKED_CHUNK_LEN,
            "encode_chunked: chunk length {} exceeds on-wire u16 limit {}",
            chunk.len(),
            MAX_CHUNKED_CHUNK_LEN
        );
    }

    let mut encoded_chunks = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let (word_streams, final_states) = rans_encode_interleaved(chunk, &norm, num_states);
        encoded_chunks.push((word_streams, final_states));
    }

    // Serialize chunked format:
    // [scale_bits: u8] [freq_table: 512] [num_chunks: u16] [num_states: u8]
    // [chunk_0_original_len: u16] [chunk_1_original_len: u16] ...
    // per chunk:
    //   [final_states: N × u32] [num_words: N × u32]
    //   [stream_0_words] [stream_1_words] ...
    let mut output = Vec::new();
    output.push(sb);
    serialize_freq_table(&norm, &mut output);
    output.extend_from_slice(&(chunks.len() as u16).to_le_bytes());
    output.push(num_states as u8);

    for chunk in &chunks {
        output.extend_from_slice(&(chunk.len() as u16).to_le_bytes());
    }

    for (word_streams, final_states) in &encoded_chunks {
        for &state in final_states {
            output.extend_from_slice(&state.to_le_bytes());
        }
        for stream in word_streams {
            output.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        }
        for stream in word_streams {
            serialize_u16_le_bulk(stream, &mut output);
        }
    }

    output
}

/// Decode chunked N-way interleaved rANS-encoded data.
pub fn decode_chunked(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut cursor = 0;

    // --- Global Header ---
    // [scale_bits: u8]
    if input.len() < cursor + 1 {
        return Err(PzError::InvalidInput);
    }
    let scale_bits = input[cursor];
    cursor += 1;
    if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
        return Err(PzError::InvalidInput);
    }

    // [freq_table: 512]
    if input.len() < cursor + NUM_SYMBOLS * 2 {
        return Err(PzError::InvalidInput);
    }
    let norm = deserialize_freq_table(&input[cursor..], scale_bits)?;
    let lookup = build_symbol_lookup(&norm);
    cursor += NUM_SYMBOLS * 2;

    // [num_chunks: u16] [num_states: u8]
    if input.len() < cursor + 3 {
        return Err(PzError::InvalidInput);
    }
    let num_chunks = u16::from_le_bytes([input[cursor], input[cursor + 1]]) as usize;
    cursor += 2;
    let num_states = input[cursor] as usize;
    cursor += 1;
    if num_states == 0 {
        return Err(PzError::InvalidInput);
    }

    // [chunk_original_lens: num_chunks × u16]
    if input.len() < cursor + num_chunks * 2 {
        return Err(PzError::InvalidInput);
    }
    let mut chunk_original_lens = Vec::with_capacity(num_chunks);
    for _ in 0..num_chunks {
        chunk_original_lens.push(u16::from_le_bytes([input[cursor], input[cursor + 1]]) as usize);
        cursor += 2;
    }

    // --- Pre-allocate output and reusable per-chunk buffers ---
    let total_len: usize = chunk_original_lens.iter().sum();
    let mut decoded_output = vec![0u8; total_len];
    let mut output_offset = 0;

    let mut initial_states = vec![0u32; num_states];
    let mut word_counts = vec![0usize; num_states];
    let mut word_slices: Vec<WordSlice<'_>> = Vec::with_capacity(num_states);

    for &original_len in &chunk_original_lens {
        // [final_states: N × u32]
        if input.len() < cursor + num_states * 4 {
            return Err(PzError::InvalidInput);
        }
        for state in initial_states.iter_mut() {
            *state = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]);
            cursor += 4;
        }

        // [num_words: N × u32]
        if input.len() < cursor + num_states * 4 {
            return Err(PzError::InvalidInput);
        }
        for wc in word_counts.iter_mut() {
            *wc = u32::from_le_bytes([
                input[cursor],
                input[cursor + 1],
                input[cursor + 2],
                input[cursor + 3],
            ]) as usize;
            cursor += 4;
        }

        // Word streams
        word_slices.clear();
        for &count in &word_counts {
            if input.len() < cursor + count * 2 {
                return Err(PzError::InvalidInput);
            }
            word_slices.push(bytes_as_u16_le(&input[cursor..], count));
            cursor += count * 2;
        }
        let word_streams: Vec<&[u16]> = word_slices.iter().map(|ws| &**ws).collect();

        // Decode directly into output buffer
        rans_decode_interleaved_into(
            &word_streams,
            &initial_states,
            &norm,
            &lookup,
            original_len,
            &mut decoded_output[output_offset..output_offset + original_len],
        )?;
        output_offset += original_len;
    }

    Ok(decoded_output)
}

// ---------------------------------------------------------------------------
// Shared-stream N-way rANS encode / decode (ryg_rans-style)
// ---------------------------------------------------------------------------
//
// In the shared-stream format, all N interleaved lanes push renormalization
// words to a single shared word stream instead of N separate streams. This
// is the prerequisite for PSHUFB branchless SIMD renormalization: the
// decoder loads consecutive words from one pointer and routes them to the
// correct lanes via a shuffle mask indexed by the renorm bitmask.
//
// Wire format:
// ```text
// [scale_bits: u8] [freq_table: 256 × u16 LE] [num_states: u8]
// [final_states: N × u32 LE] [total_words: u32 LE]
// [shared_word_stream: total_words × u16 LE]
// [padding: 8 bytes of zeros]  ← safe over-read zone for SIMD loads
// ```
//
// The 8-byte zero padding at the end allows `_mm_loadl_epi64` to safely
// read up to 8 bytes past the last valid word without bounds checking.

/// Encode input using N interleaved rANS streams with a single shared
/// word stream.
///
/// Unlike [`rans_encode_interleaved`] which produces N separate word
/// streams, this encoder pushes all renormalization words from all lanes
/// into one shared buffer. The word order in the shared stream is
/// determined by the reverse-order processing: the last symbol's lane
/// pushes first, and so on backwards.
///
/// Returns (shared_word_stream, per-lane final_states).
pub(crate) fn rans_encode_shared_stream(
    input: &[u8],
    norm: &NormalizedFreqs,
    num_states: usize,
) -> (Vec<u16>, Vec<u32>) {
    let scale_bits = norm.scale_bits as u32;
    let rcp_table = ReciprocalTable::from_normalized(norm);

    let mut states = vec![RANS_L; num_states];

    // All lanes push to a single shared buffer, filled backwards.
    let shared_cap = input.len() * 2 + 4;
    let mut shared_words = vec![0u16; shared_cap];
    let mut cursor = shared_cap; // write position, decrements

    // Process symbols in reverse (same as interleaved encoder)
    for (i, &byte) in input.iter().enumerate().rev() {
        let lane = i % num_states;
        let s = byte as usize;
        let freq = norm.freq[s] as u32;
        let cum = norm.cum[s] as u32;

        // Renormalize this lane's state — push to the SHARED buffer
        let x_max = ((RANS_L as u64 >> scale_bits) << IO_BITS) * freq as u64;
        while (states[lane] as u64) >= x_max {
            cursor -= 1;
            shared_words[cursor] = states[lane] as u16;
            states[lane] >>= IO_BITS;
        }

        // Encode
        let (q, r) = rans_div_rcp(states[lane], freq, rcp_table.rcp[s]);
        states[lane] = (q << scale_bits) + r + cum;
    }

    // Extract the filled portion — already in forward order
    let result = shared_words[cursor..].to_vec();
    (result, states)
}

/// Encode data using 4-way shared-stream rANS (default).
pub fn encode_shared_stream(input: &[u8]) -> Vec<u8> {
    encode_shared_stream_n(input, DEFAULT_INTERLEAVE, DEFAULT_SCALE_BITS)
}

/// Encode data using N-way shared-stream rANS with configurable parameters.
///
/// `num_states`: number of interleaved rANS states (typically 4).
/// `scale_bits`: frequency precision (9..14).
pub fn encode_shared_stream_n(input: &[u8], num_states: usize, scale_bits: u8) -> Vec<u8> {
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
    let (shared_words, final_states) = rans_encode_shared_stream(input, &norm, num_states);

    // Serialize shared-stream format:
    // [scale_bits: u8] [freq_table: 512] [num_states: u8]
    // [final_states: N × u32 LE] [total_words: u32 LE]
    // [shared_word_stream] [8 bytes zero padding]
    let header_size = 1 + NUM_SYMBOLS * 2 + 1 + num_states * 4 + 4;
    let padding = 8; // safe over-read zone for SIMD
    let mut output = Vec::with_capacity(header_size + shared_words.len() * 2 + padding);

    output.push(sb);
    serialize_freq_table(&norm, &mut output);
    output.push(num_states as u8);

    for &state in &final_states {
        output.extend_from_slice(&state.to_le_bytes());
    }
    output.extend_from_slice(&(shared_words.len() as u32).to_le_bytes());
    serialize_u16_le_bulk(&shared_words, &mut output);

    // Zero padding for safe SIMD over-reads at end of stream
    output.extend_from_slice(&[0u8; 8]);

    output
}

/// Decode shared-stream rANS-encoded data.
///
/// `original_len` is the number of bytes in the original uncompressed data.
pub fn decode_shared_stream(input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
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

    // Read total word count
    if input.len() < cursor + 4 {
        return Err(PzError::InvalidInput);
    }
    let total_words = u32::from_le_bytes([
        input[cursor],
        input[cursor + 1],
        input[cursor + 2],
        input[cursor + 3],
    ]) as usize;
    cursor += 4;

    // Read shared word stream (+ 8 bytes padding that follows)
    if input.len() < cursor + total_words * 2 {
        return Err(PzError::InvalidInput);
    }
    let word_slice = bytes_as_u16_le(&input[cursor..], total_words);

    // Build slot table for fast decode
    let (slot2sym, slot_table) = build_slot_table(&norm);

    // SIMD dispatch: SSSE3+SSE4.1 PSHUFB path.
    //
    // Currently DISABLED: the scalar batched 4-way path (1.18 GiB/s) is 2.7x
    // faster than the SIMD path (432 MiB/s) because the scalar gather
    // (slot2sym + slot_table lookups) dominates, and SIMD pack/unpack for
    // the gather adds overhead that exceeds the branchless renorm savings.
    //
    // The SIMD path will become viable when:
    // - AVX2 _mm256_i32gather_epi32 eliminates the scalar gather bottleneck
    // - 8-way interleaving amortizes SIMD overhead across more lanes
    //
    // The implementation is preserved and tested for future AVX2 work.
    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::overly_complex_bool_expr)]
    if false && num_states == 4 && is_x86_feature_detected!("sse4.1") {
        // SAFETY: feature detection checked above; shared_words has 8 bytes
        // of zero padding for safe _mm_loadl_epi64 over-read at end
        if let Some(result) = unsafe {
            crate::simd::rans_decode_4way_shared_ssse3(
                &word_slice,
                &initial_states,
                &slot2sym,
                &slot_table,
                norm.scale_bits as u32,
                original_len,
                num_states,
            )
        } {
            return Ok(result);
        }
    }

    // Scalar batched 4-way decode (1.18 GiB/s — faster than SIMD due to
    // gather bottleneck, better than interleaved 1.14 GiB/s due to single
    // pointer cache locality)
    rans_decode_shared_stream_scalar(
        &word_slice,
        &initial_states,
        &slot2sym,
        &slot_table,
        norm.scale_bits as u32,
        original_len,
        num_states,
    )
}

/// Scalar shared-stream rANS decoder with batched 4-way inner loop.
///
/// All lanes read from a single shared word pointer. Processes 4 symbols
/// per iteration (one per lane) for better ILP, matching the structure of
/// `rans_decode_4way_slot` which achieves 1.14 GiB/s.
fn rans_decode_shared_stream_scalar(
    shared_words: &[u16],
    initial_states: &[u32],
    slot2sym: &[u8],
    slot_table: &[SlotEntry],
    scale_bits: u32,
    original_len: usize,
    num_states: usize,
) -> PzResult<Vec<u8>> {
    let scale_mask = (1u32 << scale_bits) - 1;
    let table_len = slot2sym.len();

    let mut output = vec![0u8; original_len];

    // Fast path: 4-way batched (same structure as rans_decode_4way_slot)
    if num_states == 4 && initial_states.len() == 4 {
        let mut states = [
            initial_states[0],
            initial_states[1],
            initial_states[2],
            initial_states[3],
        ];
        let mut word_pos: usize = 0;
        let mut out_pos = 0;

        let full_quads = original_len / 4;
        let remainder = original_len % 4;

        for _ in 0..full_quads {
            let slot0 = states[0] & scale_mask;
            let slot1 = states[1] & scale_mask;
            let slot2 = states[2] & scale_mask;
            let slot3 = states[3] & scale_mask;

            if (slot0 as usize | slot1 as usize | slot2 as usize | slot3 as usize) >= table_len {
                return Err(PzError::InvalidInput);
            }

            let s0 = slot2sym[slot0 as usize];
            let s1 = slot2sym[slot1 as usize];
            let s2 = slot2sym[slot2 as usize];
            let s3 = slot2sym[slot3 as usize];

            let e0 = slot_table[slot0 as usize].freq_bias;
            let e1 = slot_table[slot1 as usize].freq_bias;
            let e2 = slot_table[slot2 as usize].freq_bias;
            let e3 = slot_table[slot3 as usize].freq_bias;

            states[0] = (e0 & 0xFFFF) * (states[0] >> scale_bits) + (e0 >> 16);
            states[1] = (e1 & 0xFFFF) * (states[1] >> scale_bits) + (e1 >> 16);
            states[2] = (e2 & 0xFFFF) * (states[2] >> scale_bits) + (e2 >> 16);
            states[3] = (e3 & 0xFFFF) * (states[3] >> scale_bits) + (e3 >> 16);

            // Renorm from SHARED stream (sequential — word_pos advances serially)
            if states[0] < RANS_L && word_pos < shared_words.len() {
                states[0] = (states[0] << IO_BITS) | shared_words[word_pos] as u32;
                word_pos += 1;
            }
            if states[1] < RANS_L && word_pos < shared_words.len() {
                states[1] = (states[1] << IO_BITS) | shared_words[word_pos] as u32;
                word_pos += 1;
            }
            if states[2] < RANS_L && word_pos < shared_words.len() {
                states[2] = (states[2] << IO_BITS) | shared_words[word_pos] as u32;
                word_pos += 1;
            }
            if states[3] < RANS_L && word_pos < shared_words.len() {
                states[3] = (states[3] << IO_BITS) | shared_words[word_pos] as u32;
                word_pos += 1;
            }

            output[out_pos] = s0;
            output[out_pos + 1] = s1;
            output[out_pos + 2] = s2;
            output[out_pos + 3] = s3;
            out_pos += 4;
        }

        // Remainder
        for state in states.iter_mut().take(remainder) {
            let slot = *state & scale_mask;
            if slot as usize >= table_len {
                return Err(PzError::InvalidInput);
            }
            let s = slot2sym[slot as usize];
            let e = slot_table[slot as usize].freq_bias;
            *state = (e & 0xFFFF) * (*state >> scale_bits) + (e >> 16);
            if *state < RANS_L && word_pos < shared_words.len() {
                *state = (*state << IO_BITS) | shared_words[word_pos] as u32;
                word_pos += 1;
            }
            output[out_pos] = s;
            out_pos += 1;
        }

        return Ok(output);
    }

    // Generic N-way fallback (1 symbol per iteration)
    let mut states: Vec<u32> = initial_states.to_vec();
    let mut word_pos: usize = 0;

    for (i, out) in output.iter_mut().enumerate() {
        let lane = i % num_states;

        let slot = states[lane] & scale_mask;
        if slot as usize >= table_len {
            return Err(PzError::InvalidInput);
        }

        let s = slot2sym[slot as usize];
        let e = slot_table[slot as usize].freq_bias;
        states[lane] = (e & 0xFFFF) * (states[lane] >> scale_bits) + (e >> 16);

        if states[lane] < RANS_L && word_pos < shared_words.len() {
            states[lane] = (states[lane] << IO_BITS) | shared_words[word_pos] as u32;
            word_pos += 1;
        }

        *out = s;
    }

    Ok(output)
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

    mod chunked_tests {
        use super::*;

        const NUM_STATES: usize = 4;
        const SCALE_BITS: u8 = DEFAULT_SCALE_BITS;
        const CHUNK_SIZE: usize = 1024;

        #[test]
        fn test_chunked_empty() {
            let encoded = encode_chunked(&[], NUM_STATES, SCALE_BITS, CHUNK_SIZE);
            assert_eq!(encoded, Vec::<u8>::new());
            let decoded = decode_chunked(&encoded).unwrap();
            assert_eq!(decoded, Vec::<u8>::new());
        }

        #[test]
        fn test_chunked_single_byte() {
            let input = &[42u8];
            let encoded = encode_chunked(input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
            let decoded = decode_chunked(&encoded).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_chunked_multiple_chunks_exact() {
            let mut input = Vec::new();
            for i in 0..3 {
                for j in 0..CHUNK_SIZE {
                    input.push((i * CHUNK_SIZE + j) as u8);
                }
            }
            let encoded = encode_chunked(&input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
            let decoded = decode_chunked(&encoded).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_chunked_multiple_chunks_partial() {
            let input: Vec<u8> = (0..CHUNK_SIZE + 100).map(|i| i as u8).collect();
            let encoded = encode_chunked(&input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
            let decoded = decode_chunked(&encoded).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_chunked_binary_data() {
            let input: Vec<u8> = (0..5000).map(|i| ((i * 41 + 61) % 256) as u8).collect();
            let encoded = encode_chunked(&input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
            let decoded = decode_chunked(&encoded).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_chunked_different_params() {
            let input: Vec<u8> = (0..3000).map(|i| ((i * 41 + 61) % 256) as u8).collect();
            let encoded = encode_chunked(&input, 8, 13, 512);
            let decoded = decode_chunked(&encoded).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_chunked_decode_invalid_input() {
            assert_eq!(decode_chunked(&[0; 10]), Err(PzError::InvalidInput));
            // Valid header but truncated data
            let input = b"some data";
            let valid_encoded = encode_chunked(input, NUM_STATES, SCALE_BITS, CHUNK_SIZE);
            assert_eq!(
                decode_chunked(&valid_encoded[..valid_encoded.len() - 10]),
                Err(PzError::InvalidInput)
            );
        }

        #[test]
        #[should_panic(expected = "num_states")]
        fn test_chunked_num_states_overflow_panics() {
            let input = vec![0u8; 1024];
            let _ = encode_chunked(&input, 256, SCALE_BITS, CHUNK_SIZE);
        }

        #[test]
        #[should_panic(expected = "chunk count")]
        fn test_chunked_chunk_count_overflow_panics() {
            // 65536 chunks (one byte each) exceeds u16 chunk-count header field.
            let input = vec![0u8; (u16::MAX as usize) + 1];
            let _ = encode_chunked(&input, NUM_STATES, SCALE_BITS, 1);
        }

        #[test]
        #[should_panic(expected = "chunk length")]
        fn test_chunked_chunk_len_overflow_panics() {
            // Single chunk length exceeds u16 per-chunk-len header field.
            let input = vec![0u8; (u16::MAX as usize) + 1];
            let _ = encode_chunked(&input, NUM_STATES, SCALE_BITS, (u16::MAX as usize) + 1);
        }
    }

    // --- Shared-stream round-trip tests ---

    mod shared_stream_tests {
        use super::*;

        #[test]
        fn test_shared_stream_empty() {
            assert_eq!(encode_shared_stream(&[]), Vec::<u8>::new());
            assert_eq!(decode_shared_stream(&[], 0).unwrap(), Vec::<u8>::new());
        }

        #[test]
        fn test_shared_stream_single_byte() {
            let input = &[42u8];
            let encoded = encode_shared_stream(input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_shared_stream_repeated() {
            let input = vec![b'x'; 500];
            let encoded = encode_shared_stream(&input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_shared_stream_all_bytes() {
            let input: Vec<u8> = (0..=255).collect();
            let encoded = encode_shared_stream(&input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_shared_stream_binary() {
            let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            let encoded = encode_shared_stream(&input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_shared_stream_large() {
            let input: Vec<u8> = (0..100_000).map(|i| ((i * 41 + 61) % 256) as u8).collect();
            let encoded = encode_shared_stream(&input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_shared_stream_skewed_distribution() {
            // Heavy skew: 90% one symbol, 10% spread
            let mut input = vec![0u8; 9000];
            for i in 0..1000 {
                input.push(((i * 7 + 3) % 255 + 1) as u8);
            }
            let encoded = encode_shared_stream(&input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_shared_stream_all_same() {
            let input = vec![42u8; 10_000];
            let encoded = encode_shared_stream(&input);
            let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
            assert_eq!(decoded, input);
        }

        #[test]
        fn test_shared_stream_small_sizes() {
            // Test sizes around and below num_states to exercise remainder logic
            for len in 1..=20 {
                let input: Vec<u8> = (0..len).map(|i| (i * 3) as u8).collect();
                let encoded = encode_shared_stream(&input);
                let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
                assert_eq!(decoded, input, "round-trip failed at len={}", len);
            }
        }

        #[test]
        fn test_shared_stream_all_scale_bits() {
            let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            for sb in MIN_SCALE_BITS..=MAX_SCALE_BITS {
                let encoded = encode_shared_stream_n(&input, 4, sb);
                let decoded = decode_shared_stream(&encoded, input.len()).unwrap();
                assert_eq!(decoded, input, "failed at scale_bits={}", sb);
            }
        }

        #[test]
        fn test_shared_stream_matches_interleaved_output() {
            // Shared-stream and interleaved should decode to the same original data
            let input: Vec<u8> = (0..2048).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            let encoded_shared = encode_shared_stream(&input);
            let encoded_interleaved = encode_interleaved(&input);
            let decoded_shared = decode_shared_stream(&encoded_shared, input.len()).unwrap();
            let decoded_interleaved =
                decode_interleaved(&encoded_interleaved, input.len()).unwrap();
            assert_eq!(decoded_shared, input);
            assert_eq!(decoded_interleaved, input);
            assert_eq!(decoded_shared, decoded_interleaved);
        }
    }
}
