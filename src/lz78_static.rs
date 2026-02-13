/// Static-dictionary LZW compression and decompression.
///
/// Unlike standard LZ78 (which builds the dictionary incrementally during
/// both encode and decode), this variant freezes the dictionary after a
/// training pass. Encoding uses only codes present in the frozen dictionary,
/// falling back to single-byte literals for unmatched input.
///
/// The frozen dictionary enables embarrassingly-parallel GPU decoding:
/// each code maps to a fixed byte sequence via pure table lookup, with
/// no inter-code dependencies.
///
/// Wire format (fixed-width codes):
/// ```text
/// [num_entries: u16 LE]
/// [max_entry_len: u16 LE]
/// [code_bits: u8]
/// [original_len: u32 LE]
/// [num_codes: u32 LE]
/// [dict_data: num_entries * max_entry_len bytes]   (padded flat layout)
/// [dict_lengths: num_entries * 2 bytes]             (u16 LE per entry)
/// [codes: ceil(num_codes * code_bits / 8) bytes]   (packed bitstream)
/// ```
use std::collections::HashMap;

use crate::{PzError, PzResult};

/// Header size: num_entries(2) + max_entry_len(2) + code_bits(1) + original_len(4) + num_codes(4)
const HEADER_SIZE: usize = 13;

/// A frozen LZW dictionary built from training data.
///
/// Each entry stores a byte sequence. Entry 0 is the root (empty string).
/// Entries 1..=255 are single-byte literals (always present).
/// Entries 256.. are learned phrases from the training data.
pub struct FrozenDict {
    /// Per-entry metadata: (offset_into_data, length).
    entries: Vec<(u32, u16)>,
    /// Concatenated byte sequences for all entries.
    data: Vec<u8>,
    /// Number of entries (including root and single-byte entries).
    num_entries: u16,
    /// Maximum entry length across all entries.
    max_entry_len: u16,
    /// Bits per code: ceil(log2(num_entries)).
    pub code_bits: u8,
}

impl FrozenDict {
    /// Build a frozen dictionary from training data using LZ78-style trie building.
    ///
    /// The dictionary always contains:
    /// - Entry 0: root (empty string)
    /// - Entries 1..=255: single-byte literals
    /// - Entries 256..max_entries: learned phrases
    ///
    /// `max_entries` must be >= 257 (256 single-byte entries + root).
    /// `max_entry_len` caps the depth of trie entries to limit GPU memory.
    pub fn train(training_data: &[u8], max_entries: u16) -> PzResult<Self> {
        Self::train_with_max_len(training_data, max_entries, 64)
    }

    /// Build a frozen dictionary with explicit max entry length.
    pub fn train_with_max_len(
        training_data: &[u8],
        max_entries: u16,
        max_entry_len: u16,
    ) -> PzResult<Self> {
        if max_entries < 257 {
            return Err(PzError::InvalidInput);
        }

        // Build trie from training data
        // (parent_index, byte) -> child_index
        let mut trie: HashMap<(u16, u8), u16> = HashMap::new();
        // Track depth of each entry for max_entry_len enforcement
        let mut entry_depth: Vec<u16> = Vec::new();

        // Initialize single-byte entries (1..=255 map to each byte value)
        // Entry 0 = root (empty), depth 0
        entry_depth.push(0); // root
        for b in 0u16..256 {
            trie.insert((0, b as u8), b + 1);
            entry_depth.push(1); // single-byte entries have depth 1
        }
        let mut next_index: u16 = 257; // first multi-byte entry

        // Build entries by scanning training data
        let mut pos = 0;
        while pos < training_data.len() && next_index < max_entries {
            let mut current_index: u16 = 0;
            let mut depth: u16 = 0;

            // Walk the trie, finding the longest known prefix
            while pos < training_data.len() {
                let byte = training_data[pos];
                if let Some(&child) = trie.get(&(current_index, byte)) {
                    current_index = child;
                    depth = entry_depth[current_index as usize];
                    pos += 1;
                } else {
                    break;
                }
            }

            if pos < training_data.len() {
                let byte = training_data[pos];
                // Only add new entry if it wouldn't exceed max_entry_len
                if next_index < max_entries && depth < max_entry_len {
                    trie.insert((current_index, byte), next_index);
                    entry_depth.push(depth + 1);
                    next_index += 1;
                }
                pos += 1;
            }
        }

        // Now reconstruct the byte sequences for each entry.
        // We need to walk the trie to find what each entry expands to.
        // Build parent/byte arrays from the trie.
        let num_entries = next_index;

        // parent[i] = parent index, byte_val[i] = byte at this node
        let mut parent = vec![0u16; num_entries as usize];
        let mut byte_val = vec![0u8; num_entries as usize];

        // Single-byte entries: parent is root (0), byte is the value
        for b in 0u16..256 {
            let idx = (b + 1) as usize;
            parent[idx] = 0;
            byte_val[idx] = b as u8;
        }

        // Multi-byte entries: set from trie
        for (&(par, b), &child) in &trie {
            if child >= 257 {
                parent[child as usize] = par;
                byte_val[child as usize] = b;
            }
        }

        // Reconstruct each entry's byte sequence by walking up to root
        let mut entries = Vec::with_capacity(num_entries as usize);
        let mut data = Vec::new();
        let mut max_entry_len: u16 = 0;

        for i in 0..num_entries {
            if i == 0 {
                // Root: empty string
                entries.push((data.len() as u32, 0u16));
                continue;
            }

            // Walk up to root, collecting bytes in reverse
            let mut bytes = Vec::new();
            let mut cur = i;
            while cur != 0 {
                bytes.push(byte_val[cur as usize]);
                cur = parent[cur as usize];
            }
            bytes.reverse();

            let len = bytes.len() as u16;
            if len > max_entry_len {
                max_entry_len = len;
            }
            entries.push((data.len() as u32, len));
            data.extend_from_slice(&bytes);
        }

        // Compute code_bits
        let code_bits = if num_entries <= 1 {
            1
        } else {
            (u16::BITS - (num_entries - 1).leading_zeros()) as u8
        };

        Ok(FrozenDict {
            entries,
            data,
            num_entries,
            max_entry_len,
            code_bits,
        })
    }

    /// Look up an entry, returning its byte sequence.
    pub fn lookup(&self, index: u16) -> &[u8] {
        if index as usize >= self.entries.len() {
            return &[];
        }
        let (offset, len) = self.entries[index as usize];
        &self.data[offset as usize..offset as usize + len as usize]
    }

    /// Number of entries in the dictionary.
    pub fn num_entries(&self) -> u16 {
        self.num_entries
    }

    /// Maximum entry length.
    pub fn max_entry_len(&self) -> u16 {
        self.max_entry_len
    }

    /// Flatten to GPU-friendly layout.
    ///
    /// Returns `(flat_entries, lengths)` where:
    /// - `flat_entries[i * stride .. (i+1) * stride]` contains entry i's bytes
    ///   (zero-padded to `stride = max_entry_len`)
    /// - `lengths[i]` is the actual length of entry i
    pub fn to_gpu_layout(&self) -> (Vec<u8>, Vec<u16>) {
        let stride = self.max_entry_len.max(1) as usize;
        let mut flat = vec![0u8; self.num_entries as usize * stride];
        let mut lengths = vec![0u16; self.num_entries as usize];

        for (i, (length, &(offset, len))) in lengths.iter_mut().zip(self.entries.iter()).enumerate()
        {
            *length = len;
            let src = &self.data[offset as usize..offset as usize + len as usize];
            let dst_start = i * stride;
            flat[dst_start..dst_start + len as usize].copy_from_slice(src);
        }

        (flat, lengths)
    }

    /// Build a trie from the frozen dictionary for fast encoding lookups.
    ///
    /// Returns a map from (parent_entry_index, byte) -> child_entry_index.
    fn build_encode_trie(&self) -> HashMap<(u16, u8), u16> {
        let mut trie = HashMap::new();

        for i in 1..self.num_entries {
            let bytes = self.lookup(i);
            if bytes.is_empty() {
                continue;
            }

            if bytes.len() == 1 {
                // Single-byte entry: parent is root
                trie.insert((0u16, bytes[0]), i);
            } else {
                // Find parent: the entry matching bytes[..bytes.len()-1]
                // Walk trie from root to find it
                let mut parent = 0u16;
                for &b in &bytes[..bytes.len() - 1] {
                    if let Some(&child) = trie.get(&(parent, b)) {
                        parent = child;
                    } else {
                        // Shouldn't happen if dictionary is well-formed
                        break;
                    }
                }
                trie.insert((parent, bytes[bytes.len() - 1]), i);
            }
        }

        trie
    }
}

/// Encode input using a frozen dictionary.
///
/// Greedily matches the longest prefix in the dictionary at each position.
/// When no match is found (shouldn't happen since single-byte entries exist),
/// falls back to emitting the single-byte code.
///
/// Returns `(packed_codes, num_codes)` where packed_codes is a bitstream
/// of fixed-width codes (`dict.code_bits` bits each).
pub fn encode_static(input: &[u8], dict: &FrozenDict) -> PzResult<(Vec<u8>, u32)> {
    if input.is_empty() {
        return Ok((Vec::new(), 0));
    }

    let trie = dict.build_encode_trie();
    let code_bits = dict.code_bits as u32;
    let mut codes: Vec<u16> = Vec::new();

    let mut pos = 0;
    while pos < input.len() {
        let mut current_index: u16 = 0;
        let mut last_match: u16 = 0;
        let mut match_len: usize = 0;

        // Greedily find the longest prefix in the dictionary
        let mut p = pos;
        while p < input.len() {
            if let Some(&child) = trie.get(&(current_index, input[p])) {
                current_index = child;
                last_match = current_index;
                match_len = p - pos + 1;
                p += 1;
            } else {
                break;
            }
        }

        if match_len == 0 {
            // Fallback: emit single-byte literal code
            let code = input[pos] as u16 + 1; // entries 1..=255 are single bytes
            codes.push(code);
            pos += 1;
        } else {
            codes.push(last_match);
            pos += match_len;
        }
    }

    // Pack codes into a bitstream
    let num_codes = codes.len() as u32;
    let total_bits = num_codes as usize * code_bits as usize;
    let packed_len = total_bits.div_ceil(8);
    let mut packed = vec![0u8; packed_len];

    for (i, &code) in codes.iter().enumerate() {
        let bit_offset = i * code_bits as usize;
        write_bits(&mut packed, bit_offset, code as u32, code_bits);
    }

    Ok((packed, num_codes))
}

/// Decode static-dictionary LZW on CPU (reference implementation).
pub fn decode_static(
    packed_codes: &[u8],
    num_codes: u32,
    dict: &FrozenDict,
    original_len: usize,
) -> PzResult<Vec<u8>> {
    if num_codes == 0 {
        return Ok(Vec::new());
    }

    let code_bits = dict.code_bits as u32;
    let mask = (1u32 << code_bits) - 1;
    let mut output = Vec::with_capacity(original_len);

    for i in 0..num_codes as usize {
        let bit_offset = i * code_bits as usize;
        let code = read_bits(packed_codes, bit_offset, code_bits) & mask;

        if code as u16 >= dict.num_entries {
            return Err(PzError::InvalidInput);
        }

        let bytes = dict.lookup(code as u16);
        output.extend_from_slice(bytes);
    }

    output.truncate(original_len);
    Ok(output)
}

/// Full encode: train dictionary on input prefix, then static-encode everything.
pub fn encode(input: &[u8]) -> PzResult<Vec<u8>> {
    encode_with_params(input, 4096, 64)
}

/// Encode with configurable dictionary parameters.
pub fn encode_with_params(input: &[u8], max_entries: u16, max_entry_len: u16) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let max_entries = max_entries.max(257);
    let max_entry_len = max_entry_len.max(1);

    // Train on the full input (in-domain training)
    let dict = FrozenDict::train_with_max_len(input, max_entries, max_entry_len)?;

    // Encode
    let (packed_codes, num_codes) = encode_static(input, &dict)?;

    // Get GPU-friendly layout for serialization
    let (flat_entries, lengths) = dict.to_gpu_layout();

    // Serialize
    let dict_data_len = flat_entries.len();
    let dict_lengths_len = lengths.len() * 2;
    let total_size = HEADER_SIZE + dict_data_len + dict_lengths_len + packed_codes.len();

    let mut output = Vec::with_capacity(total_size);

    // Header
    output.extend_from_slice(&dict.num_entries.to_le_bytes());
    output.extend_from_slice(&dict.max_entry_len.to_le_bytes());
    output.push(dict.code_bits);
    output.extend_from_slice(&(input.len() as u32).to_le_bytes());
    output.extend_from_slice(&num_codes.to_le_bytes());

    // Dictionary data (flat layout)
    output.extend_from_slice(&flat_entries);

    // Dictionary lengths
    for &len in &lengths {
        output.extend_from_slice(&len.to_le_bytes());
    }

    // Packed codes
    output.extend_from_slice(&packed_codes);

    Ok(output)
}

/// Encode data using a pre-trained frozen dictionary.
///
/// This is used for out-of-domain experiments: train on one dataset,
/// encode a different dataset. The dictionary is serialized into the
/// output alongside the encoded data.
pub fn encode_with_dict(input: &[u8], dict: &FrozenDict) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let (packed_codes, num_codes) = encode_static(input, dict)?;
    let (flat_entries, lengths) = dict.to_gpu_layout();

    let dict_data_len = flat_entries.len();
    let dict_lengths_len = lengths.len() * 2;
    let total_size = HEADER_SIZE + dict_data_len + dict_lengths_len + packed_codes.len();

    let mut output = Vec::with_capacity(total_size);

    // Header
    output.extend_from_slice(&dict.num_entries.to_le_bytes());
    output.extend_from_slice(&dict.max_entry_len.to_le_bytes());
    output.push(dict.code_bits);
    output.extend_from_slice(&(input.len() as u32).to_le_bytes());
    output.extend_from_slice(&num_codes.to_le_bytes());

    // Dictionary data (flat layout)
    output.extend_from_slice(&flat_entries);

    // Dictionary lengths
    for &len in &lengths {
        output.extend_from_slice(&len.to_le_bytes());
    }

    // Packed codes
    output.extend_from_slice(&packed_codes);

    Ok(output)
}

/// Decompress static-dictionary LZW data.
pub fn decode(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    if input.len() < HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    // Parse header
    let num_entries = u16::from_le_bytes([input[0], input[1]]);
    let max_entry_len = u16::from_le_bytes([input[2], input[3]]);
    let code_bits = input[4];
    let original_len = u32::from_le_bytes([input[5], input[6], input[7], input[8]]) as usize;
    let num_codes = u32::from_le_bytes([input[9], input[10], input[11], input[12]]);

    let stride = max_entry_len.max(1) as usize;
    let dict_data_len = num_entries as usize * stride;
    let dict_lengths_len = num_entries as usize * 2;
    let codes_start = HEADER_SIZE + dict_data_len + dict_lengths_len;

    if input.len() < codes_start {
        return Err(PzError::InvalidInput);
    }

    // Reconstruct dictionary from flat layout
    let flat_entries = &input[HEADER_SIZE..HEADER_SIZE + dict_data_len];
    let lengths_data = &input[HEADER_SIZE + dict_data_len..codes_start];
    let packed_codes = &input[codes_start..];

    // Build FrozenDict from serialized data
    let mut entries = Vec::with_capacity(num_entries as usize);
    let mut data = Vec::new();

    for i in 0..num_entries as usize {
        let len = u16::from_le_bytes([lengths_data[i * 2], lengths_data[i * 2 + 1]]);
        let src_start = i * stride;
        let offset = data.len() as u32;
        data.extend_from_slice(&flat_entries[src_start..src_start + len as usize]);
        entries.push((offset, len));
    }

    let dict = FrozenDict {
        entries,
        data,
        num_entries,
        max_entry_len,
        code_bits,
    };

    decode_static(packed_codes, num_codes, &dict, original_len)
}

/// Compress into a caller-allocated buffer. Returns bytes written.
pub fn encode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    let encoded = encode(input)?;
    if encoded.len() > output.len() {
        return Err(PzError::BufferTooSmall);
    }
    output[..encoded.len()].copy_from_slice(&encoded);
    Ok(encoded.len())
}

/// Decompress into a caller-allocated buffer. Returns bytes written.
pub fn decode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    let decoded = decode(input)?;
    if decoded.len() > output.len() {
        return Err(PzError::BufferTooSmall);
    }
    output[..decoded.len()].copy_from_slice(&decoded);
    Ok(decoded.len())
}

/// Parse the header of a static-dictionary LZW compressed stream.
///
/// Returns `(dict, packed_codes, num_codes, original_len)` for GPU decode.
pub fn parse_for_gpu(input: &[u8]) -> PzResult<(FrozenDict, &[u8], u32, usize)> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    if input.len() < HEADER_SIZE {
        return Err(PzError::InvalidInput);
    }

    let num_entries = u16::from_le_bytes([input[0], input[1]]);
    let max_entry_len = u16::from_le_bytes([input[2], input[3]]);
    let code_bits = input[4];
    let original_len = u32::from_le_bytes([input[5], input[6], input[7], input[8]]) as usize;
    let num_codes = u32::from_le_bytes([input[9], input[10], input[11], input[12]]);

    let stride = max_entry_len.max(1) as usize;
    let dict_data_len = num_entries as usize * stride;
    let dict_lengths_len = num_entries as usize * 2;
    let codes_start = HEADER_SIZE + dict_data_len + dict_lengths_len;

    if input.len() < codes_start {
        return Err(PzError::InvalidInput);
    }

    let flat_entries = &input[HEADER_SIZE..HEADER_SIZE + dict_data_len];
    let lengths_data = &input[HEADER_SIZE + dict_data_len..codes_start];
    let packed_codes = &input[codes_start..];

    let mut entries = Vec::with_capacity(num_entries as usize);
    let mut data = Vec::new();

    for i in 0..num_entries as usize {
        let len = u16::from_le_bytes([lengths_data[i * 2], lengths_data[i * 2 + 1]]);
        let src_start = i * stride;
        let offset = data.len() as u32;
        data.extend_from_slice(&flat_entries[src_start..src_start + len as usize]);
        entries.push((offset, len));
    }

    let dict = FrozenDict {
        entries,
        data,
        num_entries,
        max_entry_len,
        code_bits,
    };

    Ok((dict, packed_codes, num_codes, original_len))
}

// --- Bit I/O helpers ---

/// Write `num_bits` bits of `value` into `buf` starting at `bit_offset`.
/// LSB-first packing.
fn write_bits(buf: &mut [u8], bit_offset: usize, value: u32, num_bits: u32) {
    let mut remaining = num_bits;
    let mut val = value;
    let mut bit_pos = bit_offset;

    while remaining > 0 {
        let byte_idx = bit_pos / 8;
        let bit_in_byte = bit_pos % 8;
        let bits_this_byte = remaining.min((8 - bit_in_byte) as u32);
        let mask = ((1u32 << bits_this_byte) - 1) as u8;

        buf[byte_idx] |= ((val as u8) & mask) << bit_in_byte;

        val >>= bits_this_byte;
        bit_pos += bits_this_byte as usize;
        remaining -= bits_this_byte;
    }
}

/// Read `num_bits` bits from `buf` starting at `bit_offset`.
/// LSB-first packing.
fn read_bits(buf: &[u8], bit_offset: usize, num_bits: u32) -> u32 {
    let byte_offset = bit_offset / 8;
    let bit_in_byte = bit_offset % 8;

    // Read up to 4 bytes to cover any code_bits <= 16
    let mut raw: u32 = 0;
    for i in 0..4 {
        let idx = byte_offset + i;
        if idx < buf.len() {
            raw |= (buf[idx] as u32) << (i * 8);
        }
    }

    (raw >> bit_in_byte) & ((1u32 << num_bits) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frozen_dict_train_basic() {
        let training = b"aaabbbccc";
        let dict = FrozenDict::train(training, 512).unwrap();
        // Should have 257 single-byte + some multi-byte entries
        assert!(dict.num_entries >= 257);
        // Entry 0 is root (empty)
        assert_eq!(dict.lookup(0), &[]);
        // Entry 1 is byte 0x00
        assert_eq!(dict.lookup(1), &[0u8]);
        // Entry for 'a' (0x61) should be at index 0x61 + 1 = 98
        assert_eq!(dict.lookup(b'a' as u16 + 1), b"a");
    }

    #[test]
    fn test_frozen_dict_train_min_entries() {
        let result = FrozenDict::train(b"test", 256);
        assert_eq!(result.err(), Some(PzError::InvalidInput));
    }

    #[test]
    fn test_encode_decode_static_round_trip() {
        let input = b"aaabbbaaabbb";
        let dict = FrozenDict::train(input, 512).unwrap();
        let (packed, num_codes) = encode_static(input, &dict).unwrap();
        let decoded = decode_static(&packed, num_codes, &dict, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_full_round_trip_text() {
        let input = b"To be, or not to be, that is the question: \
            Whether 'tis nobler in the mind to suffer \
            The slings and arrows of outrageous fortune, \
            Or to take arms against a sea of troubles"
            .to_vec();
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_full_round_trip_binary() {
        let input: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_full_round_trip_repetitive() {
        let input = vec![b'x'; 2000];
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_full_round_trip_empty() {
        let input: Vec<u8> = Vec::new();
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_full_round_trip_single_byte() {
        let input = vec![42u8];
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_full_round_trip_large_repetitive() {
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..100 {
            input.extend_from_slice(pattern);
        }
        let compressed = encode(&input).unwrap();
        let decompressed = decode(&compressed).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn test_various_dict_sizes() {
        let input = b"abcabcabcabcabcabcabcabc".to_vec();
        for &dict_size in &[257u16, 512, 1024, 2048, 4096, 8192] {
            let compressed = encode_with_params(&input, dict_size, 64).unwrap();
            let decompressed = decode(&compressed).unwrap();
            assert_eq!(decompressed, input, "Failed with dict_size={dict_size}");
        }
    }

    #[test]
    fn test_gpu_layout() {
        let input = b"aabbcc";
        let dict = FrozenDict::train(input, 512).unwrap();
        let (flat, lengths) = dict.to_gpu_layout();

        // Verify flat layout
        let stride = dict.max_entry_len.max(1) as usize;
        assert_eq!(flat.len(), dict.num_entries as usize * stride);
        assert_eq!(lengths.len(), dict.num_entries as usize);

        // Entry 0 (root) should have length 0
        assert_eq!(lengths[0], 0);

        // Single-byte entries should have length 1
        for i in 1..=255u16 {
            assert_eq!(lengths[i as usize], 1);
        }
    }

    #[test]
    fn test_bit_io_round_trip() {
        let mut buf = vec![0u8; 16];
        // Write various values
        write_bits(&mut buf, 0, 0b101, 3);
        write_bits(&mut buf, 3, 0b1100, 4);
        write_bits(&mut buf, 7, 0xFF, 8);
        write_bits(&mut buf, 15, 0b1, 1);

        assert_eq!(read_bits(&buf, 0, 3), 0b101);
        assert_eq!(read_bits(&buf, 3, 4), 0b1100);
        assert_eq!(read_bits(&buf, 7, 8), 0xFF);
        assert_eq!(read_bits(&buf, 15, 1), 0b1);
    }

    #[test]
    fn test_parse_for_gpu() {
        let input = b"hello world hello world hello world".to_vec();
        let compressed = encode(&input).unwrap();
        let (dict, packed_codes, num_codes, original_len) = parse_for_gpu(&compressed).unwrap();

        assert_eq!(original_len, input.len());
        assert!(num_codes > 0);

        // Decode should match
        let decoded = decode_static(packed_codes, num_codes, &dict, original_len).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_compresses_repetitive_data() {
        // Static-dictionary format has fixed overhead for the dictionary:
        // num_entries * max_entry_len bytes. Use a small dictionary (512
        // entries) to keep overhead low, and a large enough input to
        // outweigh it.
        let pattern = b"the quick brown fox jumps over the lazy dog. ";
        let mut input = Vec::new();
        for _ in 0..200 {
            input.extend_from_slice(pattern);
        }
        let compressed = encode_with_params(&input, 512, 32).unwrap();
        assert!(
            compressed.len() < input.len(),
            "Should compress repetitive data with small dict: {} >= {}",
            compressed.len(),
            input.len()
        );
    }

    #[test]
    fn test_encode_to_buf() {
        let input = b"hello hello hello".to_vec();
        let encoded = encode(&input).unwrap();
        let mut buf = vec![0u8; encoded.len() + 10];
        let written = encode_to_buf(&input, &mut buf).unwrap();
        assert_eq!(written, encoded.len());
        assert_eq!(&buf[..written], &encoded[..]);
    }

    #[test]
    fn test_decode_to_buf() {
        let input = b"test data for decode_to_buf".to_vec();
        let encoded = encode(&input).unwrap();
        let mut buf = vec![0u8; input.len() + 10];
        let written = decode_to_buf(&encoded, &mut buf).unwrap();
        assert_eq!(written, input.len());
        assert_eq!(&buf[..written], &input[..]);
    }

    #[test]
    fn test_decode_invalid_short() {
        let result = decode(&[0, 1, 2]);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_code_bits_calculation() {
        let dict = FrozenDict::train(b"test", 257).unwrap();
        // 257 entries needs ceil(log2(257)) = 9 bits
        assert_eq!(dict.code_bits, 9);

        let dict = FrozenDict::train(b"test data with more entries", 512).unwrap();
        // Up to 512 entries needs ceil(log2(512)) = 9 bits
        assert!(dict.code_bits <= 10);
    }
}
