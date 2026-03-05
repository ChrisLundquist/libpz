/// Recoil: parallel rANS decoding with decoder-adaptive scalability.
///
/// Implements a variant of the Recoil algorithm (Lin et al., ICPP 2023) which
/// enables parallel decoding of a *single* interleaved rANS bitstream by
/// recording split-point metadata during a post-encode pass. The decoder uses
/// this metadata to start decoding from any split point in parallel.
///
/// Unlike the conventional chunked approach ([`crate::rans::encode_chunked`]),
/// Recoil does **not** modify the rANS bitstream. The encoder produces one
/// standard interleaved rANS stream, and a splitting pass records where a
/// decoder can resume decoding. Metadata entries can be dropped to reduce
/// parallelism — no re-encoding needed.
///
/// # Exact-state splits (no catchup needed)
///
/// The original Recoil paper uses approximate split points that require a
/// "catchup" phase where the decoder discards symbols until it resynchronizes
/// with the bitstream. Our implementation avoids this: the splitting pass
/// performs a full forward decode simulation and records **exact** decoder
/// state (rANS states + per-lane word positions) at each split boundary.
/// This means each split can begin decoding immediately — no catchup, no
/// overlap, no discarded symbols.
///
/// Each split decodes its range `[symbol_index, next_split.symbol_index)`
/// directly into the output buffer.
use crate::rans::{
    self, build_symbol_lookup, bytes_as_u16_le, deserialize_freq_table, NormalizedFreqs, WordSlice,
    IO_BITS, MAX_SCALE_BITS, MIN_SCALE_BITS, NUM_SYMBOLS, RANS_L,
};
use crate::{PzError, PzResult};

/// Magic byte identifying serialized Recoil metadata.
const RECOIL_MAGIC: u8 = 0xEC;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Metadata for a single Recoil split point.
///
/// Records the decoder state at a point in the interleaved rANS stream,
/// enabling a decoder to start decoding from this position.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RecoilSplit {
    /// Symbol index where this split begins producing output.
    pub symbol_index: u32,
    /// Intermediate rANS state for each interleaved decoder at this split.
    /// Length = num_states (K).
    pub states: Vec<u32>,
    /// Per-lane word-stream consumption position (in u16 words).
    /// `word_positions[lane]` = how many words lane has consumed from its
    /// word stream up to this split point.
    pub word_positions: Vec<u32>,
}

/// Recoil metadata — side-channel alongside an interleaved rANS bitstream.
///
/// This metadata does not modify the rANS bitstream. It records split points
/// that allow parallel decoding. Split points can be combined (dropped) to
/// reduce parallelism without re-encoding.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct RecoilMetadata {
    /// Number of interleaved coders (K). Matches the encode's num_states.
    pub num_states: u8,
    /// Frequency precision bits. Matches the encode's scale_bits.
    pub scale_bits: u8,
    /// Total number of symbols in the original input.
    ///
    /// Stored as `u32`, limiting Recoil metadata to inputs < 4 GiB.
    /// The wire format (`symbol_index: u32`) shares this constraint.
    pub total_symbols: u32,
    /// Split points, sorted by symbol_index. The first split always has
    /// symbol_index = 0.
    pub splits: Vec<RecoilSplit>,
}

// ---------------------------------------------------------------------------
// Parsing the interleaved wire format (shared between split generation and
// decode — extracts header fields and word stream references)
// ---------------------------------------------------------------------------

/// Parsed interleaved rANS header + word stream references.
struct ParsedInterleaved<'a> {
    norm: NormalizedFreqs,
    initial_states: Vec<u32>,
    word_slices: Vec<WordSlice<'a>>,
}

/// Parse the standard interleaved rANS wire format header.
///
/// Returns the frequency table, initial states, and per-lane word stream
/// slices. This is factored out so both `recoil_generate_splits` and
/// `decode_recoil` can share it.
fn parse_interleaved_header(encoded: &[u8]) -> PzResult<ParsedInterleaved<'_>> {
    // Minimum: scale_bits(1) + freq_table(512) + num_states(1)
    if encoded.len() < 1 + NUM_SYMBOLS * 2 + 1 {
        return Err(PzError::InvalidInput);
    }

    let scale_bits = encoded[0];
    if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
        return Err(PzError::InvalidInput);
    }

    let norm = deserialize_freq_table(&encoded[1..], scale_bits)?;

    let pos = 1 + NUM_SYMBOLS * 2;
    let num_states = encoded[pos] as usize;
    if num_states == 0 {
        return Err(PzError::InvalidInput);
    }

    let mut cursor = pos + 1;

    // Read final/initial states
    if encoded.len() < cursor + num_states * 4 {
        return Err(PzError::InvalidInput);
    }
    let mut initial_states = Vec::with_capacity(num_states);
    for _ in 0..num_states {
        let state = u32::from_le_bytes([
            encoded[cursor],
            encoded[cursor + 1],
            encoded[cursor + 2],
            encoded[cursor + 3],
        ]);
        initial_states.push(state);
        cursor += 4;
    }

    // Read word counts per stream
    if encoded.len() < cursor + num_states * 4 {
        return Err(PzError::InvalidInput);
    }
    let mut word_counts = Vec::with_capacity(num_states);
    for _ in 0..num_states {
        let count = u32::from_le_bytes([
            encoded[cursor],
            encoded[cursor + 1],
            encoded[cursor + 2],
            encoded[cursor + 3],
        ]) as usize;
        word_counts.push(count);
        cursor += 4;
    }

    // Read word streams (zero-copy when aligned on little-endian)
    let mut word_slices: Vec<WordSlice<'_>> = Vec::with_capacity(num_states);
    for &count in &word_counts {
        if encoded.len() < cursor + count * 2 {
            return Err(PzError::InvalidInput);
        }
        word_slices.push(bytes_as_u16_le(&encoded[cursor..], count));
        cursor += count * 2;
    }

    Ok(ParsedInterleaved {
        norm,
        initial_states,
        word_slices,
    })
}

// ---------------------------------------------------------------------------
// Splitting pass (Phase 1)
// ---------------------------------------------------------------------------

/// Generate Recoil split-point metadata from an interleaved rANS stream.
///
/// This is a **post-encode** step. It simulates a full decode of the
/// interleaved rANS stream and records split points at regular intervals.
/// The rANS bitstream is **not modified**.
///
/// `encoded` must be a standard interleaved rANS wire format payload
/// (as produced by [`rans::encode_interleaved_n`]).
///
/// `original_len` is the number of symbols that were encoded.
///
/// `num_splits` is the desired number of split points (including the
/// initial split at symbol_index=0). Must be >= 1.
pub(crate) fn recoil_generate_splits(
    encoded: &[u8],
    original_len: usize,
    num_splits: usize,
) -> PzResult<RecoilMetadata> {
    if original_len == 0 {
        return Ok(RecoilMetadata {
            num_states: 1,
            scale_bits: rans::DEFAULT_SCALE_BITS,
            total_symbols: 0,
            splits: vec![],
        });
    }

    // Split metadata stores symbol indices as u32; reject inputs that overflow.
    if original_len > u32::MAX as usize {
        return Err(PzError::InvalidInput);
    }

    let num_splits = num_splits.max(1);

    let parsed = parse_interleaved_header(encoded)?;
    let num_states = parsed.initial_states.len();
    let lookup = build_symbol_lookup(&parsed.norm);
    let scale_bits_u32 = parsed.norm.scale_bits as u32;
    let scale_mask = (1u32 << scale_bits_u32) - 1;

    // Simulate decode, recording splits at intervals.
    let split_interval = if num_splits >= original_len {
        1
    } else {
        original_len / num_splits
    };

    let mut states: Vec<u32> = parsed.initial_states.clone();
    let mut word_positions: Vec<u32> = vec![0; num_states];
    let mut splits = Vec::with_capacity(num_splits);

    // Record the initial split at symbol_index=0.
    splits.push(RecoilSplit {
        symbol_index: 0,
        states: states.clone(),
        word_positions: word_positions.clone(),
    });

    let word_streams: Vec<&[u16]> = parsed.word_slices.iter().map(|ws| &**ws).collect();

    for i in 0..original_len {
        let lane = i % num_states;

        // Decode one symbol (we don't need the actual symbol value here,
        // but we need to advance the state correctly)
        let slot = states[lane] & scale_mask;
        if slot as usize >= lookup.len() {
            return Err(PzError::InvalidInput);
        }
        let s = lookup[slot as usize];
        let freq = parsed.norm.freq[s as usize] as u32;
        let cum = parsed.norm.cum[s as usize] as u32;

        states[lane] = freq * (states[lane] >> scale_bits_u32) + slot - cum;

        // Renormalize
        if states[lane] < RANS_L && (word_positions[lane] as usize) < word_streams[lane].len() {
            states[lane] = (states[lane] << IO_BITS)
                | word_streams[lane][word_positions[lane] as usize] as u32;
            word_positions[lane] += 1;
        }

        // Record split after processing the last symbol before the boundary.
        // The split's symbol_index is the *next* symbol (i+1).
        let sym_after = i + 1;
        if sym_after < original_len && sym_after % split_interval == 0 && splits.len() < num_splits
        {
            splits.push(RecoilSplit {
                symbol_index: sym_after as u32,
                states: states.clone(),
                word_positions: word_positions.clone(),
            });
        }
    }

    Ok(RecoilMetadata {
        num_states: num_states as u8,
        scale_bits: parsed.norm.scale_bits,
        total_symbols: original_len as u32,
        splits,
    })
}

// ---------------------------------------------------------------------------
// Adaptive scaling
// ---------------------------------------------------------------------------

impl RecoilMetadata {
    /// Reduce the number of splits to `target_count` by dropping intermediate
    /// entries. The first split (symbol_index=0) is always preserved.
    ///
    /// If `target_count >= self.splits.len()`, returns a clone unchanged.
    /// This is an O(S) operation — the bitstream is not touched.
    pub fn combine_splits(&self, target_count: usize) -> RecoilMetadata {
        if target_count == 0 || self.splits.is_empty() {
            return self.clone();
        }
        if target_count >= self.splits.len() {
            return self.clone();
        }

        let total = self.splits.len();
        let mut kept = Vec::with_capacity(target_count);

        // Always keep the first split.
        kept.push(self.splits[0].clone());

        // Evenly space the remaining target_count-1 splits across the rest.
        if target_count > 1 {
            let remaining = target_count - 1;
            for i in 1..=remaining {
                let idx = (i * (total - 1)) / remaining;
                if idx > 0 && idx < total {
                    kept.push(self.splits[idx].clone());
                }
            }
        }

        // Deduplicate by symbol_index (shouldn't happen, but be safe).
        kept.dedup_by_key(|s| s.symbol_index);

        RecoilMetadata {
            num_states: self.num_states,
            scale_bits: self.scale_bits,
            total_symbols: self.total_symbols,
            splits: kept,
        }
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

impl RecoilMetadata {
    /// Serialize to a compact binary format.
    ///
    /// Wire format:
    /// ```text
    /// [magic: u8 = 0xEC]
    /// [num_states: u8]
    /// [scale_bits: u8]
    /// [total_symbols: u32 LE]
    /// [num_splits: u16 LE]
    /// Per split:
    ///   [symbol_index: u32 LE]
    ///   [states: K × u32 LE]
    ///   [word_positions: K × u32 LE]
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let k = self.num_states as usize;
        let per_split = 4 + k * 4 + k * 4;
        let header = 1 + 1 + 1 + 4 + 2; // magic + num_states + scale_bits + total_symbols + num_splits
        let mut out = Vec::with_capacity(header + self.splits.len() * per_split);

        out.push(RECOIL_MAGIC);
        out.push(self.num_states);
        out.push(self.scale_bits);
        out.extend_from_slice(&self.total_symbols.to_le_bytes());
        out.extend_from_slice(&(self.splits.len() as u16).to_le_bytes());

        for split in &self.splits {
            out.extend_from_slice(&split.symbol_index.to_le_bytes());
            for &state in &split.states {
                out.extend_from_slice(&state.to_le_bytes());
            }
            for &wp in &split.word_positions {
                out.extend_from_slice(&wp.to_le_bytes());
            }
        }

        out
    }

    /// Deserialize from the compact binary format produced by [`serialize`].
    pub fn deserialize(data: &[u8]) -> PzResult<RecoilMetadata> {
        let header_len = 1 + 1 + 1 + 4 + 2;
        if data.len() < header_len {
            return Err(PzError::InvalidInput);
        }

        if data[0] != RECOIL_MAGIC {
            return Err(PzError::InvalidInput);
        }

        let num_states = data[1];
        if num_states == 0 {
            return Err(PzError::InvalidInput);
        }
        let scale_bits = data[2];
        if !(MIN_SCALE_BITS..=MAX_SCALE_BITS).contains(&scale_bits) {
            return Err(PzError::InvalidInput);
        }

        let total_symbols = u32::from_le_bytes([data[3], data[4], data[5], data[6]]);
        let num_splits = u16::from_le_bytes([data[7], data[8]]) as usize;

        let k = num_states as usize;
        let per_split = 4 + k * 4 + k * 4;
        if data.len() < header_len + num_splits * per_split {
            return Err(PzError::InvalidInput);
        }

        let mut splits = Vec::with_capacity(num_splits);
        let mut cursor = header_len;

        for _ in 0..num_splits {
            let symbol_index = u32::from_le_bytes([
                data[cursor],
                data[cursor + 1],
                data[cursor + 2],
                data[cursor + 3],
            ]);
            cursor += 4;

            let mut states = Vec::with_capacity(k);
            for _ in 0..k {
                states.push(u32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]));
                cursor += 4;
            }

            let mut word_positions = Vec::with_capacity(k);
            for _ in 0..k {
                word_positions.push(u32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]));
                cursor += 4;
            }

            splits.push(RecoilSplit {
                symbol_index,
                states,
                word_positions,
            });
        }

        Ok(RecoilMetadata {
            num_states,
            scale_bits,
            total_symbols,
            splits,
        })
    }
}

// ---------------------------------------------------------------------------
// Single-threaded Recoil decode (Phase 2)
// ---------------------------------------------------------------------------

/// Decode an interleaved rANS stream using Recoil split-point metadata.
///
/// This is a single-threaded decode that processes all splits sequentially.
/// For multi-threaded decode, see [`decode_recoil_parallel`].
///
/// `encoded` is the standard interleaved rANS wire format payload.
/// `metadata` provides the split points generated by [`recoil_generate_splits`].
/// `original_len` is the expected number of output symbols.
pub(crate) fn decode_recoil(
    encoded: &[u8],
    metadata: &RecoilMetadata,
    original_len: usize,
) -> PzResult<Vec<u8>> {
    if original_len == 0 {
        return Ok(Vec::new());
    }

    if metadata.splits.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let parsed = parse_interleaved_header(encoded)?;
    let lookup = build_symbol_lookup(&parsed.norm);
    let word_streams: Vec<&[u16]> = parsed.word_slices.iter().map(|ws| &**ws).collect();
    let ctx = DecodeContext {
        word_streams: &word_streams,
        norm: &parsed.norm,
        lookup: &lookup,
    };

    let mut output = vec![0u8; original_len];

    for (split_idx, split) in metadata.splits.iter().enumerate() {
        let sym_start = split.symbol_index as usize;
        let sym_end = if split_idx + 1 < metadata.splits.len() {
            metadata.splits[split_idx + 1].symbol_index as usize
        } else {
            original_len
        };

        if sym_start >= original_len || sym_end > original_len {
            return Err(PzError::InvalidInput);
        }

        decode_split_range(
            &ctx,
            &split.states,
            &split.word_positions,
            sym_start,
            sym_end,
            &mut output,
        )?;
    }

    Ok(output)
}

/// Shared decode context for split-range decoding.
struct DecodeContext<'a> {
    word_streams: &'a [&'a [u16]],
    norm: &'a NormalizedFreqs,
    lookup: &'a [u8],
}

/// Decode a single split's range of symbols into the output buffer.
///
/// For split 0 (symbol_index=0), no catchup is needed — the states and
/// word positions are the stream's initial values.
///
/// For subsequent splits, the saved states come from the splitting pass
/// and are already synchronized (they were recorded mid-decode), so
/// **no catchup phase is needed** — the states and word positions precisely
/// capture the decoder state at that symbol boundary.
#[allow(clippy::needless_range_loop)] // `i` drives both lane assignment and output indexing
fn decode_split_range(
    ctx: &DecodeContext<'_>,
    initial_states: &[u32],
    initial_word_positions: &[u32],
    sym_start: usize,
    sym_end: usize,
    output: &mut [u8],
) -> PzResult<()> {
    let num_states = initial_states.len();
    let scale_bits = ctx.norm.scale_bits as u32;
    let scale_mask = (1u32 << scale_bits) - 1;

    let mut states: Vec<u32> = initial_states.to_vec();
    let mut word_positions: Vec<usize> =
        initial_word_positions.iter().map(|&p| p as usize).collect();

    for (idx, output_ref) in output
        .iter_mut()
        .enumerate()
        .skip(sym_start)
        .take(sym_end - sym_start)
    {
        let lane = idx % num_states;

        let slot = states[lane] & scale_mask;
        if slot as usize >= ctx.lookup.len() {
            return Err(PzError::InvalidInput);
        }
        let s = ctx.lookup[slot as usize];
        let freq = ctx.norm.freq[s as usize] as u32;
        let cum = ctx.norm.cum[s as usize] as u32;

        states[lane] = freq * (states[lane] >> scale_bits) + slot - cum;

        if states[lane] < RANS_L && word_positions[lane] < ctx.word_streams[lane].len() {
            states[lane] =
                (states[lane] << IO_BITS) | ctx.word_streams[lane][word_positions[lane]] as u32;
            word_positions[lane] += 1;
        }

        *output_ref = s;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Multi-threaded Recoil decode (Phase 3)
// ---------------------------------------------------------------------------

/// Decode an interleaved rANS stream using Recoil metadata with multiple threads.
///
/// Uses `std::thread::scope` for zero-overhead scoped threading.
/// Each thread decodes one split range into a non-overlapping slice of the
/// output buffer — no synchronization needed during the hot loop.
///
/// If `num_threads` is less than the number of splits, splits are combined
/// (via [`RecoilMetadata::combine_splits`]) to match.
pub(crate) fn decode_recoil_parallel(
    encoded: &[u8],
    metadata: &RecoilMetadata,
    original_len: usize,
    num_threads: usize,
) -> PzResult<Vec<u8>> {
    let num_threads = num_threads.max(1);

    // Fall back to single-threaded if only 1 thread requested or 1 split.
    if num_threads == 1 || metadata.splits.len() <= 1 {
        return decode_recoil(encoded, metadata, original_len);
    }

    if original_len == 0 {
        return Ok(Vec::new());
    }

    // Adapt split count to thread count.
    let active_meta = if metadata.splits.len() > num_threads {
        metadata.combine_splits(num_threads)
    } else {
        metadata.clone()
    };

    let parsed = parse_interleaved_header(encoded)?;
    let lookup = build_symbol_lookup(&parsed.norm);
    let word_streams: Vec<&[u16]> = parsed.word_slices.iter().map(|ws| &**ws).collect();
    let ctx = DecodeContext {
        word_streams: &word_streams,
        norm: &parsed.norm,
        lookup: &lookup,
    };

    let mut output = vec![0u8; original_len];
    let num_splits = active_meta.splits.len();

    // Build a list of (split, sym_start, sym_end) tasks.
    let mut tasks: Vec<(&RecoilSplit, usize, usize)> = Vec::with_capacity(num_splits);
    for (idx, split) in active_meta.splits.iter().enumerate() {
        let sym_start = split.symbol_index as usize;
        let sym_end = if idx + 1 < num_splits {
            active_meta.splits[idx + 1].symbol_index as usize
        } else {
            original_len
        };
        if sym_start > original_len || sym_end > original_len || sym_start > sym_end {
            return Err(PzError::InvalidInput);
        }
        tasks.push((split, sym_start, sym_end));
    }

    // Split the output buffer into non-overlapping mutable slices using
    // split_at_mut, then dispatch each to a scoped thread.
    let mut slices: Vec<&mut [u8]> = Vec::with_capacity(num_splits);
    {
        let mut remainder = output.as_mut_slice();
        let mut prev_start = 0usize;
        for &(_, _, sym_end) in &tasks {
            let split_at = sym_end - prev_start;
            let (chunk, rest) = remainder.split_at_mut(split_at);
            slices.push(chunk);
            remainder = rest;
            prev_start = sym_end;
        }
    }

    let result: PzResult<()> = std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(tasks.len());

        for (task_idx, &(split, sym_start, sym_end)) in tasks.iter().enumerate() {
            let states = &split.states;
            let word_pos = &split.word_positions;
            let slice = std::mem::take(&mut slices[task_idx]);
            let ctx_ref = &ctx;

            let handle = s.spawn(move || {
                decode_split_range_local(ctx_ref, states, word_pos, sym_start, sym_end, slice)
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().map_err(|_| PzError::InvalidInput)??;
        }

        Ok(())
    });

    result?;
    Ok(output)
}

/// Like `decode_split_range` but writes to a local slice starting at offset 0.
/// `sym_start`/`sym_end` are the global symbol indices (used for lane assignment).
fn decode_split_range_local(
    ctx: &DecodeContext<'_>,
    initial_states: &[u32],
    initial_word_positions: &[u32],
    sym_start: usize,
    sym_end: usize,
    output: &mut [u8],
) -> PzResult<()> {
    let num_states = initial_states.len();
    let scale_bits = ctx.norm.scale_bits as u32;
    let scale_mask = (1u32 << scale_bits) - 1;

    let mut states: Vec<u32> = initial_states.to_vec();
    let mut word_positions: Vec<usize> =
        initial_word_positions.iter().map(|&p| p as usize).collect();

    for i in sym_start..sym_end {
        let lane = i % num_states;
        let out_idx = i - sym_start;

        let slot = states[lane] & scale_mask;
        if slot as usize >= ctx.lookup.len() {
            return Err(PzError::InvalidInput);
        }
        let s = ctx.lookup[slot as usize];
        let freq = ctx.norm.freq[s as usize] as u32;
        let cum = ctx.norm.cum[s as usize] as u32;

        states[lane] = freq * (states[lane] >> scale_bits) + slot - cum;

        if states[lane] < RANS_L && word_positions[lane] < ctx.word_streams[lane].len() {
            states[lane] =
                (states[lane] << IO_BITS) | ctx.word_streams[lane][word_positions[lane]] as u32;
            word_positions[lane] += 1;
        }

        output[out_idx] = s;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rans;

    const NUM_STATES: usize = 4;
    const SCALE_BITS: u8 = 12;

    /// Encode helper: interleaved rANS with given parameters.
    fn encode(input: &[u8]) -> Vec<u8> {
        rans::encode_interleaved_n(input, NUM_STATES, SCALE_BITS)
    }

    // ----- Phase 1 tests: splitting pass + serialization -----

    #[test]
    fn test_generate_splits_empty() {
        let encoded = encode(&[]);
        let meta = recoil_generate_splits(&encoded, 0, 4).unwrap();
        assert_eq!(meta.total_symbols, 0);
        assert!(meta.splits.is_empty());
    }

    #[test]
    fn test_generate_splits_single_byte() {
        let input = [42u8];
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 1).unwrap();
        assert_eq!(meta.total_symbols, 1);
        assert_eq!(meta.splits.len(), 1);
        assert_eq!(meta.splits[0].symbol_index, 0);
    }

    #[test]
    fn test_generate_splits_basic() {
        let input: Vec<u8> = (0..256u16).map(|i| (i % 256) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 4).unwrap();

        assert_eq!(meta.total_symbols, 256);
        assert_eq!(meta.num_states as usize, NUM_STATES);
        assert_eq!(meta.scale_bits, SCALE_BITS);
        assert!(meta.splits.len() >= 2); // at least first + one more
        assert_eq!(meta.splits[0].symbol_index, 0);

        // Splits should be sorted by symbol_index.
        for w in meta.splits.windows(2) {
            assert!(w[0].symbol_index < w[1].symbol_index);
        }

        // Each split should have the right number of states and word positions.
        for split in &meta.splits {
            assert_eq!(split.states.len(), NUM_STATES);
            assert_eq!(split.word_positions.len(), NUM_STATES);
        }
    }

    #[test]
    fn test_generate_splits_many() {
        let input: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 16).unwrap();

        assert_eq!(meta.total_symbols as usize, input.len());
        // Should have roughly 16 splits (may be slightly less if input doesn't
        // divide evenly).
        assert!(meta.splits.len() >= 2);
        assert!(meta.splits.len() <= 16);
    }

    #[test]
    fn test_combine_splits() {
        let input: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 16).unwrap();
        let original_len = meta.splits.len();

        let combined = meta.combine_splits(4);
        assert!(combined.splits.len() <= 4);
        assert!(!combined.splits.is_empty());
        assert_eq!(combined.splits[0].symbol_index, 0);

        // Combining to more splits than we have should return a clone.
        let no_change = meta.combine_splits(original_len + 10);
        assert_eq!(no_change.splits.len(), original_len);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let input: Vec<u8> = (0..1024).map(|i| (i % 200) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 8).unwrap();

        let serialized = meta.serialize();
        let deserialized = RecoilMetadata::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.num_states, meta.num_states);
        assert_eq!(deserialized.scale_bits, meta.scale_bits);
        assert_eq!(deserialized.total_symbols, meta.total_symbols);
        assert_eq!(deserialized.splits.len(), meta.splits.len());

        for (a, b) in meta.splits.iter().zip(deserialized.splits.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_serialize_invalid_input() {
        // Too short
        assert_eq!(RecoilMetadata::deserialize(&[]), Err(PzError::InvalidInput));
        assert_eq!(
            RecoilMetadata::deserialize(&[0; 5]),
            Err(PzError::InvalidInput)
        );

        // Bad magic
        let mut data = vec![0u8; 9];
        data[0] = 0xFF; // wrong magic
        data[1] = 4; // num_states
        data[2] = 12; // scale_bits
        assert_eq!(
            RecoilMetadata::deserialize(&data),
            Err(PzError::InvalidInput)
        );
    }

    // ----- Phase 2 tests: single-threaded decode -----

    #[test]
    fn test_decode_recoil_empty() {
        let encoded = encode(&[]);
        let meta = recoil_generate_splits(&encoded, 0, 4).unwrap();
        let decoded = decode_recoil(&encoded, &meta, 0).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_decode_recoil_single_byte() {
        let input = [42u8];
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 1).unwrap();
        let decoded = decode_recoil(&encoded, &meta, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_single_split() {
        let input: Vec<u8> = (0..256u16).map(|i| (i % 256) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 1).unwrap();
        let decoded = decode_recoil(&encoded, &meta, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_multi_split() {
        let input: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&input);

        for num_splits in [2, 4, 8, 16] {
            let meta = recoil_generate_splits(&encoded, input.len(), num_splits).unwrap();
            let decoded = decode_recoil(&encoded, &meta, input.len()).unwrap();
            assert_eq!(
                decoded, input,
                "Recoil decode failed with {} splits",
                num_splits
            );
        }
    }

    #[test]
    fn test_decode_recoil_all_same_byte() {
        let input = vec![0xAA; 2048];
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 8).unwrap();
        let decoded = decode_recoil(&encoded, &meta, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_short_input() {
        // Input shorter than num_states
        let input = vec![1, 2, 3];
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 2).unwrap();
        let decoded = decode_recoil(&encoded, &meta, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_combined_splits() {
        let input: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 16).unwrap();

        // Combine to 4 splits and verify decode still works.
        let combined = meta.combine_splits(4);
        let decoded = decode_recoil(&encoded, &combined, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_cross_validate_with_interleaved() {
        let input: Vec<u8> = (0..1024).map(|i| (i % 200) as u8).collect();
        let encoded = encode(&input);

        // Standard interleaved decode
        let expected = rans::decode_interleaved(&encoded, input.len()).unwrap();

        // Recoil decode with 1 split (should be identical path)
        let meta = recoil_generate_splits(&encoded, input.len(), 1).unwrap();
        let recoil_result = decode_recoil(&encoded, &meta, input.len()).unwrap();

        assert_eq!(recoil_result, expected);
    }

    #[test]
    fn test_decode_recoil_various_num_states() {
        let input: Vec<u8> = (0..2048).map(|i| (i % 200) as u8).collect();

        for num_states in [1, 2, 4, 8] {
            let encoded = rans::encode_interleaved_n(&input, num_states, SCALE_BITS);
            let meta = recoil_generate_splits(&encoded, input.len(), 8).unwrap();
            let decoded = decode_recoil(&encoded, &meta, input.len()).unwrap();
            assert_eq!(
                decoded, input,
                "Recoil failed with num_states={}",
                num_states
            );
        }
    }

    // ----- Phase 3 tests: multi-threaded decode -----

    #[test]
    fn test_decode_recoil_parallel_single_thread() {
        let input: Vec<u8> = (0..2048).map(|i| (i % 200) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 8).unwrap();

        let decoded = decode_recoil_parallel(&encoded, &meta, input.len(), 1).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_parallel_multi_thread() {
        let input: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 16).unwrap();

        for num_threads in [2, 4, 8] {
            let decoded =
                decode_recoil_parallel(&encoded, &meta, input.len(), num_threads).unwrap();
            assert_eq!(
                decoded, input,
                "Parallel decode failed with {} threads",
                num_threads
            );
        }
    }

    #[test]
    fn test_decode_recoil_parallel_more_threads_than_splits() {
        let input: Vec<u8> = (0..1024).map(|i| (i % 200) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 2).unwrap();

        // 8 threads but only 2 splits — should still work
        let decoded = decode_recoil_parallel(&encoded, &meta, input.len(), 8).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_parallel_more_splits_than_threads() {
        let input: Vec<u8> = (0..8192).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 256).unwrap();

        // 4 threads with many splits — should combine and decode correctly
        let decoded = decode_recoil_parallel(&encoded, &meta, input.len(), 4).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_recoil_parallel_matches_sequential() {
        let input: Vec<u8> = (0..4096).map(|i| (i % 251) as u8).collect();
        let encoded = encode(&input);
        let meta = recoil_generate_splits(&encoded, input.len(), 16).unwrap();

        let sequential = decode_recoil(&encoded, &meta, input.len()).unwrap();
        let parallel = decode_recoil_parallel(&encoded, &meta, input.len(), 4).unwrap();
        assert_eq!(sequential, parallel);
    }
}
