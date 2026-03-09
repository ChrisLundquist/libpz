/// Huffman coding: tree construction, encoding, and decoding.
///
/// Provides two complementary Huffman primitives:
///
/// - [`HuffmanTree`]: Full encoder/decoder built from byte frequency data.
///   Used by the library's own compression pipelines (PZ format).
///
/// - [`HuffTable`]: Decode-only canonical Huffman table built from code
///   lengths. Used for standard formats like DEFLATE/gzip where the
///   bitstream transmits code lengths rather than frequencies.
///
/// Bug fixes from the C reference:
/// - BUG-01: Priority queue sift-down (fixed in pqueue module)
/// - BUG-05: Signed char as array index (Rust uses u8, no issue)
/// - BUG-08: Encode actually writes to output buffer (implemented)
/// - BUG-09: Decode is fully implemented (tree walk)
/// - BUG-10: Dead code in merge_nodes (removed)
/// - BUG-12: Output buffer bounds checking (implemented)
use crate::frequency::FrequencyTable;
use crate::pqueue::MinHeap;
use crate::{PzError, PzResult};

/// A node in the Huffman tree.
#[derive(Debug, Clone)]
pub struct HuffmanNode {
    /// Frequency weight of this node (or subtree).
    pub weight: u32,
    /// Byte value (only meaningful for leaf nodes).
    pub value: u8,
    /// The Huffman codeword assigned to this node.
    pub codeword: u32,
    /// Number of bits in the codeword.
    pub code_bits: u8,
    /// Left child index (None for leaves).
    pub left: Option<usize>,
    /// Right child index (None for leaves).
    pub right: Option<usize>,
}

/// Bits used for the fast decode lookup table.
const DECODE_TABLE_BITS: u8 = 12;
/// Number of entries in the fast decode table.
const DECODE_TABLE_SIZE: usize = 1 << DECODE_TABLE_BITS;

/// A Huffman tree for encoding and decoding byte streams.
#[derive(Debug, Clone)]
pub struct HuffmanTree {
    /// All nodes stored in a flat vector. Internal nodes are appended
    /// after the initial 256 leaf slots.
    nodes: Vec<HuffmanNode>,
    /// Index of the root node in `nodes`.
    root: Option<usize>,
    /// Lookup table: for each byte value, (codeword, code_bits).
    /// Indexed by byte value (0-255).
    lookup: [(u32, u8); 256],
    /// Number of distinct symbols in the tree.
    pub leaf_count: u32,
    /// Fast decode table: peek DECODE_TABLE_BITS MSB bits → (symbol, code_length).
    /// Codes longer than DECODE_TABLE_BITS fall back to tree walk.
    decode_table: Vec<(u8, u8)>,
}

impl HuffmanTree {
    /// Build a Huffman tree from input data.
    ///
    /// Computes byte frequencies, builds the tree via a min-heap priority
    /// queue, and generates the codeword lookup table.
    pub fn from_data(input: &[u8]) -> Option<Self> {
        if input.is_empty() {
            return None;
        }

        let mut freq = FrequencyTable::new();
        freq.count(input);

        Self::from_frequency_table_encode_only(&freq)
    }

    /// Build a Huffman tree from a pre-computed frequency table.
    ///
    /// Includes the fast decode table for O(1) per-symbol decoding.
    pub fn from_frequency_table(freq: &FrequencyTable) -> Option<Self> {
        let mut tree = Self::build_tree(freq)?;
        tree.decode_table = Self::build_decode_table(&tree.lookup);
        Some(tree)
    }

    /// Build a Huffman tree from a frequency table, encode-only (no decode table).
    ///
    /// Skips the 4096-entry decode lookup table allocation. The resulting tree
    /// can only be used for encoding. Decode methods will fall back to tree walk.
    pub fn from_frequency_table_encode_only(freq: &FrequencyTable) -> Option<Self> {
        Self::build_tree(freq)
    }

    /// Shared tree construction: builds nodes, lookup table, but no decode table.
    fn build_tree(freq: &FrequencyTable) -> Option<Self> {
        if freq.used == 0 {
            return None;
        }

        // Initialize nodes: 256 leaf slots (one per byte value)
        let mut nodes: Vec<HuffmanNode> = Vec::with_capacity(512);
        for i in 0..256u16 {
            nodes.push(HuffmanNode {
                weight: freq.byte[i as usize],
                value: i as u8,
                codeword: 0,
                code_bits: 0,
                left: None,
                right: None,
            });
        }

        // Build the tree using a min-heap priority queue
        let mut heap: MinHeap<usize> = MinHeap::new();
        for (i, node) in nodes.iter().enumerate().take(256) {
            if node.weight > 0 {
                heap.push(node.weight, i);
            }
        }

        // Special case: only one distinct symbol
        if freq.used == 1 {
            let leaf_idx = heap.pop().unwrap();
            let root_idx = nodes.len();
            nodes.push(HuffmanNode {
                weight: nodes[leaf_idx].weight,
                value: 0,
                codeword: 0,
                code_bits: 0,
                left: Some(leaf_idx),
                right: None,
            });

            let mut lookup = [(0u32, 0u8); 256];
            nodes[leaf_idx].codeword = 0;
            nodes[leaf_idx].code_bits = 1;
            lookup[nodes[leaf_idx].value as usize] = (0, 1);

            return Some(HuffmanTree {
                nodes,
                root: Some(root_idx),
                lookup,
                leaf_count: 1,
                decode_table: Vec::new(),
            });
        }

        // Standard Huffman construction: merge two lowest-weight nodes
        while heap.len() > 1 {
            let left_idx = heap.pop().unwrap();
            let right_idx = heap.pop().unwrap();

            let combined_weight = nodes[left_idx].weight + nodes[right_idx].weight;
            let new_idx = nodes.len();
            nodes.push(HuffmanNode {
                weight: combined_weight,
                value: 0,
                codeword: 0,
                code_bits: 0,
                left: Some(left_idx),
                right: Some(right_idx),
            });
            heap.push(combined_weight, new_idx);
        }

        let root_idx = heap.pop().unwrap();

        // Generate codewords via tree traversal
        let mut lookup = [(0u32, 0u8); 256];
        Self::generate_codes(&mut nodes, root_idx, 0, 0, &mut lookup);

        Some(HuffmanTree {
            nodes,
            root: Some(root_idx),
            lookup,
            leaf_count: freq.used,
            decode_table: Vec::new(),
        })
    }

    /// Recursively assign codewords to all nodes in the tree.
    fn generate_codes(
        nodes: &mut [HuffmanNode],
        idx: usize,
        prefix: u32,
        depth: u8,
        lookup: &mut [(u32, u8); 256],
    ) {
        nodes[idx].codeword = prefix;
        nodes[idx].code_bits = depth;

        let left = nodes[idx].left;
        let right = nodes[idx].right;

        if let Some(left_idx) = left {
            Self::generate_codes(nodes, left_idx, prefix << 1, depth + 1, lookup);
        }
        if let Some(right_idx) = right {
            Self::generate_codes(nodes, right_idx, (prefix << 1) | 1, depth + 1, lookup);
        }

        // If this is a leaf node, store in lookup table
        if left.is_none() && right.is_none() {
            lookup[nodes[idx].value as usize] = (nodes[idx].codeword, nodes[idx].code_bits);
        }
    }

    /// Build a flat decode lookup table for fast O(1) symbol resolution.
    ///
    /// For each symbol with code_bits <= DECODE_TABLE_BITS, fills
    /// 2^(TABLE_BITS - code_bits) entries so that peeking TABLE_BITS
    /// bits from the MSB of the bitstream directly yields (symbol, length).
    fn build_decode_table(lookup: &[(u32, u8); 256]) -> Vec<(u8, u8)> {
        let mut table = vec![(0u8, 0u8); DECODE_TABLE_SIZE];

        for sym in 0..256u16 {
            let (codeword, code_bits) = lookup[sym as usize];
            if code_bits == 0 || code_bits > DECODE_TABLE_BITS {
                continue;
            }
            // The codeword occupies the top code_bits bits of the peek window.
            // Fill all entries where the top code_bits bits match.
            let num_entries = 1usize << (DECODE_TABLE_BITS - code_bits);
            let base = (codeword as usize) << (DECODE_TABLE_BITS - code_bits);
            for i in 0..num_entries {
                table[base + i] = (sym as u8, code_bits);
            }
        }

        table
    }

    /// Return the code lookup table for use with GPU encoding.
    ///
    /// Each entry is a u32 where bits [31:24] are the code length (in bits)
    /// and bits [23:0] are the codeword value (MSB-first). This format matches
    /// the GPU kernel's expectations.
    pub fn code_lut(&self) -> [u32; 256] {
        let mut result = [0u32; 256];
        for (i, &(codeword, code_bits)) in self.lookup.iter().enumerate() {
            // Encode as: length in top 8 bits, codeword in bottom 24 bits
            result[i] = ((code_bits as u32) << 24) | (codeword & 0x00FFFFFF);
        }
        result
    }

    /// Returns a tuple of (encoded_bytes, total_bits_encoded).
    ///
    /// Uses a 64-bit accumulator to pack entire codewords per symbol
    /// (MSB-first), flushing complete bytes from the top. This avoids
    /// the per-bit loop and branch that the naive implementation requires.
    pub fn encode(&self, input: &[u8]) -> PzResult<(Vec<u8>, usize)> {
        if input.is_empty() {
            return Ok((Vec::new(), 0));
        }

        // Calculate worst-case output size
        let mut total_bits: usize = 0;
        for &byte in input {
            let (_, bits) = self.lookup[byte as usize];
            if bits == 0 {
                return Err(PzError::InvalidInput);
            }
            total_bits += bits as usize;
        }

        let output_len = total_bits.div_ceil(8);
        let mut output = vec![0u8; output_len];
        let mut byte_pos: usize = 0;
        // 64-bit accumulator: bits packed from MSB downward.
        // bits_in_acc counts valid bits from the MSB side.
        let mut accumulator: u64 = 0;
        let mut bits_in_acc: u32 = 0;

        for &byte in input {
            let (codeword, code_bits) = self.lookup[byte as usize];
            // Pack entire codeword into accumulator in one operation.
            // Place it at position (64 - bits_in_acc - code_bits) from the LSB,
            // which is right-justified under the existing bits.
            let shift = 64 - bits_in_acc - code_bits as u32;
            accumulator |= (codeword as u64) << shift;
            bits_in_acc += code_bits as u32;

            // Flush complete bytes from the MSB side
            while bits_in_acc >= 8 {
                output[byte_pos] = (accumulator >> 56) as u8;
                accumulator <<= 8;
                bits_in_acc -= 8;
                byte_pos += 1;
            }
        }

        // Flush any remaining partial byte
        if bits_in_acc > 0 {
            output[byte_pos] = (accumulator >> 56) as u8;
        }

        Ok((output, total_bits))
    }

    /// Encode input bytes into a pre-allocated output buffer.
    ///
    /// Returns the number of bits written.
    pub fn encode_to_buf(&self, input: &[u8], output: &mut [u8]) -> PzResult<usize> {
        if input.is_empty() {
            return Ok(0);
        }

        let mut byte_pos: usize = 0;
        let mut accumulator: u64 = 0;
        let mut bits_in_acc: u32 = 0;
        let mut total_bits: usize = 0;

        for &byte in input {
            let (codeword, code_bits) = self.lookup[byte as usize];
            if code_bits == 0 {
                return Err(PzError::InvalidInput);
            }

            total_bits += code_bits as usize;

            // BUG-12 fix: check output buffer has room
            if total_bits.div_ceil(8) > output.len() {
                return Err(PzError::BufferTooSmall);
            }

            let shift = 64 - bits_in_acc - code_bits as u32;
            accumulator |= (codeword as u64) << shift;
            bits_in_acc += code_bits as u32;

            while bits_in_acc >= 8 {
                output[byte_pos] = (accumulator >> 56) as u8;
                accumulator <<= 8;
                bits_in_acc -= 8;
                byte_pos += 1;
            }
        }

        if bits_in_acc > 0 {
            output[byte_pos] = (accumulator >> 56) as u8;
        }

        Ok(total_bits)
    }

    /// Decode Huffman-encoded data back to the original bytes.
    ///
    /// `total_bits` is the number of valid bits in `input` (needed because
    /// the last byte may have padding bits).
    ///
    /// Uses a flat lookup table for O(1) per-symbol decode. Peek
    /// DECODE_TABLE_BITS from the MSB accumulator, look up (symbol, length),
    /// advance by length bits. Codes longer than DECODE_TABLE_BITS fall back
    /// to tree walk (extremely rare in practice).
    pub fn decode(&self, input: &[u8], total_bits: usize) -> PzResult<Vec<u8>> {
        if self.root.is_none() {
            return Err(PzError::InvalidInput);
        }

        let mut output = Vec::new();
        // MSB-first bit accumulator: new bytes enter at the bottom,
        // we peek/consume from the top.
        let mut accumulator: u64 = 0;
        let mut bits_avail: u32 = 0;
        let mut byte_pos: usize = 0;
        let mut bits_consumed: usize = 0;

        // Refill the accumulator from the MSB side
        while bits_avail <= 56 && byte_pos < input.len() {
            accumulator |= (input[byte_pos] as u64) << (56 - bits_avail);
            bits_avail += 8;
            byte_pos += 1;
        }

        while bits_consumed < total_bits {
            // Refill when we might not have enough for a table peek
            while bits_avail <= 56 && byte_pos < input.len() {
                accumulator |= (input[byte_pos] as u64) << (56 - bits_avail);
                bits_avail += 8;
                byte_pos += 1;
            }

            if !self.decode_table.is_empty() && bits_avail >= DECODE_TABLE_BITS as u32 {
                // Fast path: peek top DECODE_TABLE_BITS bits
                let peek = (accumulator >> (64 - DECODE_TABLE_BITS as u32)) as usize;
                let (sym, len) = self.decode_table[peek];
                if len > 0 {
                    output.push(sym);
                    accumulator <<= len as u32;
                    bits_avail -= len as u32;
                    bits_consumed += len as usize;
                    continue;
                }
            }

            // Slow path: tree walk for codes > DECODE_TABLE_BITS,
            // when accumulator is nearly empty, or no decode table
            return self.decode_tree_walk(input, total_bits, bits_consumed, output);
        }

        Ok(output)
    }

    /// Decode Huffman-encoded data into a pre-allocated output buffer.
    ///
    /// Returns the number of bytes written to `output`.
    pub fn decode_to_buf(
        &self,
        input: &[u8],
        total_bits: usize,
        output: &mut [u8],
    ) -> PzResult<usize> {
        if self.root.is_none() {
            return Err(PzError::InvalidInput);
        }

        let mut out_pos: usize = 0;
        let mut accumulator: u64 = 0;
        let mut bits_avail: u32 = 0;
        let mut byte_pos: usize = 0;
        let mut bits_consumed: usize = 0;

        while bits_avail <= 56 && byte_pos < input.len() {
            accumulator |= (input[byte_pos] as u64) << (56 - bits_avail);
            bits_avail += 8;
            byte_pos += 1;
        }

        while bits_consumed < total_bits {
            while bits_avail <= 56 && byte_pos < input.len() {
                accumulator |= (input[byte_pos] as u64) << (56 - bits_avail);
                bits_avail += 8;
                byte_pos += 1;
            }

            if !self.decode_table.is_empty() && bits_avail >= DECODE_TABLE_BITS as u32 {
                let peek = (accumulator >> (64 - DECODE_TABLE_BITS as u32)) as usize;
                let (sym, len) = self.decode_table[peek];
                if len > 0 {
                    if out_pos >= output.len() {
                        return Err(PzError::BufferTooSmall);
                    }
                    output[out_pos] = sym;
                    out_pos += 1;
                    accumulator <<= len as u32;
                    bits_avail -= len as u32;
                    bits_consumed += len as usize;
                    continue;
                }
            }

            // Slow path: tree walk for remaining symbols or no decode table
            let remaining = self.decode_tree_walk(input, total_bits, bits_consumed, Vec::new())?;
            for &sym in &remaining {
                if out_pos >= output.len() {
                    return Err(PzError::BufferTooSmall);
                }
                output[out_pos] = sym;
                out_pos += 1;
            }
            return Ok(out_pos);
        }

        Ok(out_pos)
    }

    /// Fallback tree-walk decoder for codes longer than DECODE_TABLE_BITS.
    fn decode_tree_walk(
        &self,
        input: &[u8],
        total_bits: usize,
        start_bit: usize,
        mut output: Vec<u8>,
    ) -> PzResult<Vec<u8>> {
        let root_idx = self.root.unwrap();
        let mut bit_pos = start_bit;
        let mut node_idx = root_idx;

        while bit_pos < total_bits {
            let byte_idx = bit_pos / 8;
            if byte_idx >= input.len() {
                return Err(PzError::InvalidInput);
            }
            let bit_offset = 7 - (bit_pos % 8);
            let bit = (input[byte_idx] >> bit_offset) & 1;
            bit_pos += 1;

            let next = if bit == 0 {
                self.nodes[node_idx].left
            } else {
                self.nodes[node_idx].right
            };

            match next {
                Some(child_idx) => {
                    let child = &self.nodes[child_idx];
                    if child.left.is_none() && child.right.is_none() {
                        output.push(child.value);
                        node_idx = root_idx;
                    } else {
                        node_idx = child_idx;
                    }
                }
                None => {
                    if self.nodes[node_idx].left.is_none() && self.nodes[node_idx].right.is_none() {
                        output.push(self.nodes[node_idx].value);
                        node_idx = root_idx;
                    } else {
                        return Err(PzError::InvalidInput);
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get the codeword and number of bits for a given byte value.
    pub fn get_code(&self, byte: u8) -> (u32, u8) {
        self.lookup[byte as usize]
    }

    /// Serialize the tree for transmission (frequency table format).
    ///
    /// Returns a 256-entry frequency table that can be used to reconstruct
    /// the tree on the decoder side.
    pub fn serialize_frequencies(&self) -> [u32; 256] {
        let mut freqs = [0u32; 256];
        for (freq, node) in freqs.iter_mut().zip(self.nodes.iter().take(256)) {
            *freq = node.weight;
        }
        freqs
    }
}

// ---------------------------------------------------------------------------
// HuffTable – canonical Huffman decoder from code lengths
// ---------------------------------------------------------------------------

/// Maximum code length for canonical Huffman tables.
///
/// DEFLATE allows up to 15 bits. This limit is suitable for most
/// canonical Huffman applications.
pub const MAX_CODE_BITS: usize = 15;

/// Canonical Huffman decode table built from code lengths.
///
/// This is the standard approach used by formats that transmit code
/// lengths in the bitstream (DEFLATE, PNG, etc.) rather than
/// frequencies. Supports arbitrary symbol counts (not limited to 256).
///
/// Decoding uses the count-based canonical algorithm — no tree
/// structure is needed, just a sorted symbol list and per-length
/// counts.
pub struct HuffTable {
    /// Number of codes of each length (index 0 unused).
    counts: [u16; MAX_CODE_BITS + 1],
    /// Symbols sorted by (code_length, canonical_order).
    symbols: Vec<u16>,
}

impl HuffTable {
    /// Build a decode table from an array of code lengths (one per symbol).
    ///
    /// Symbols with length 0 are not coded. Returns an error if any
    /// code length exceeds [`MAX_CODE_BITS`].
    pub fn from_lengths(lengths: &[u8]) -> PzResult<Self> {
        let mut counts = [0u16; MAX_CODE_BITS + 1];
        for &len in lengths {
            if len as usize > MAX_CODE_BITS {
                return Err(PzError::InvalidInput);
            }
            counts[len as usize] += 1;
        }

        // Compute offsets for sorting.
        let mut offsets = [0u16; MAX_CODE_BITS + 1];
        for i in 1..MAX_CODE_BITS {
            offsets[i + 1] = offsets[i] + counts[i];
        }

        let num_symbols: usize = counts[1..].iter().map(|&c| c as usize).sum();
        let mut symbols = vec![0u16; num_symbols];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                let idx = offsets[len as usize] as usize;
                symbols[idx] = sym as u16;
                offsets[len as usize] += 1;
            }
        }

        Ok(HuffTable { counts, symbols })
    }

    /// Decode one symbol by reading bits via the provided closure.
    ///
    /// `read_bit` must return the next bit (0 or 1) from the
    /// bitstream each time it is called. This keeps the table
    /// independent of any particular bit-reader implementation.
    pub fn decode(&self, read_bit: &mut impl FnMut() -> PzResult<u32>) -> PzResult<u16> {
        let mut code: u32 = 0;
        let mut first: u32 = 0;
        let mut index: usize = 0;

        for len in 1..=MAX_CODE_BITS {
            code = (code << 1) | read_bit()?;
            let count = self.counts[len] as u32;
            if code < first + count {
                return Ok(self.symbols[index + (code - first) as usize]);
            }
            index += count as usize;
            first = (first + count) << 1;
        }

        Err(PzError::InvalidInput)
    }

    /// Number of coded symbols in this table.
    pub fn symbol_count(&self) -> usize {
        self.symbols.len()
    }
}

#[cfg(test)]
mod tests;
