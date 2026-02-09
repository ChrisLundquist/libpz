/// Huffman coding: tree construction, encoding, and decoding.
///
/// This is a complete implementation, fixing bugs from the C reference:
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

        Self::from_frequency_table(&freq)
    }

    /// Build a Huffman tree from a pre-computed frequency table.
    pub fn from_frequency_table(freq: &FrequencyTable) -> Option<Self> {
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
            // Find the one leaf with nonzero weight
            let leaf_idx = heap.pop().unwrap();
            // Create a dummy internal node as root with the leaf as left child
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
            // Assign code 0 with 1 bit to the single symbol
            nodes[leaf_idx].codeword = 0;
            nodes[leaf_idx].code_bits = 1;
            lookup[nodes[leaf_idx].value as usize] = (0, 1);

            return Some(HuffmanTree {
                nodes,
                root: Some(root_idx),
                lookup,
                leaf_count: 1,
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

    /// Encode input bytes using this Huffman tree.
    ///
    /// Returns a tuple of (encoded_bytes, total_bits_encoded).
    ///
    /// This is the complete implementation that fixes BUG-08 (the C version
    /// never wrote to the output buffer) and BUG-12 (no bounds checking).
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
        let mut bit_pos: usize = 0;

        for &byte in input {
            let (codeword, code_bits) = self.lookup[byte as usize];
            // Write codeword bits into output, MSB first
            for bit_idx in (0..code_bits).rev() {
                let bit = (codeword >> bit_idx) & 1;
                if bit == 1 {
                    let byte_idx = bit_pos / 8;
                    let bit_offset = 7 - (bit_pos % 8);
                    output[byte_idx] |= 1 << bit_offset;
                }
                bit_pos += 1;
            }
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

        let mut bit_pos: usize = 0;

        for &byte in input {
            let (codeword, code_bits) = self.lookup[byte as usize];
            if code_bits == 0 {
                return Err(PzError::InvalidInput);
            }

            // BUG-12 fix: check output buffer has room
            if (bit_pos + code_bits as usize).div_ceil(8) > output.len() {
                return Err(PzError::BufferTooSmall);
            }

            // Write codeword bits into output, MSB first
            for bit_idx in (0..code_bits).rev() {
                let bit = (codeword >> bit_idx) & 1;
                if bit == 1 {
                    let byte_idx = bit_pos / 8;
                    let bit_offset = 7 - (bit_pos % 8);
                    output[byte_idx] |= 1 << bit_offset;
                }
                bit_pos += 1;
            }
        }

        Ok(bit_pos)
    }

    /// Decode Huffman-encoded data back to the original bytes.
    ///
    /// `total_bits` is the number of valid bits in `input` (needed because
    /// the last byte may have padding bits).
    ///
    /// This is the complete implementation that fixes BUG-09 (the C version
    /// was an unimplemented stub).
    pub fn decode(&self, input: &[u8], total_bits: usize) -> PzResult<Vec<u8>> {
        let root_idx = match self.root {
            Some(idx) => idx,
            None => return Err(PzError::InvalidInput),
        };

        let mut output = Vec::new();
        let mut bit_pos: usize = 0;
        let mut node_idx = root_idx;

        while bit_pos < total_bits {
            // Read one bit
            let byte_idx = bit_pos / 8;
            if byte_idx >= input.len() {
                return Err(PzError::InvalidInput);
            }
            let bit_offset = 7 - (bit_pos % 8);
            let bit = (input[byte_idx] >> bit_offset) & 1;
            bit_pos += 1;

            // Traverse: left on 0, right on 1
            let next = if bit == 0 {
                self.nodes[node_idx].left
            } else {
                self.nodes[node_idx].right
            };

            match next {
                Some(child_idx) => {
                    let child = &self.nodes[child_idx];
                    if child.left.is_none() && child.right.is_none() {
                        // Leaf node: emit byte value
                        output.push(child.value);
                        node_idx = root_idx;
                    } else {
                        node_idx = child_idx;
                    }
                }
                None => {
                    // We're at a leaf already (single-symbol case) or invalid
                    // For the single-symbol tree, the root has only a left child
                    // which is the leaf. If we get None, it means the current
                    // node is a leaf.
                    if self.nodes[node_idx].left.is_none()
                        && self.nodes[node_idx].right.is_none()
                    {
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

    /// Decode Huffman-encoded data into a pre-allocated output buffer.
    ///
    /// Returns the number of bytes written to `output`.
    pub fn decode_to_buf(
        &self,
        input: &[u8],
        total_bits: usize,
        output: &mut [u8],
    ) -> PzResult<usize> {
        let root_idx = match self.root {
            Some(idx) => idx,
            None => return Err(PzError::InvalidInput),
        };

        let mut out_pos: usize = 0;
        let mut bit_pos: usize = 0;
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
                        if out_pos >= output.len() {
                            return Err(PzError::BufferTooSmall);
                        }
                        output[out_pos] = child.value;
                        out_pos += 1;
                        node_idx = root_idx;
                    } else {
                        node_idx = child_idx;
                    }
                }
                None => {
                    if self.nodes[node_idx].left.is_none()
                        && self.nodes[node_idx].right.is_none()
                    {
                        if out_pos >= output.len() {
                            return Err(PzError::BufferTooSmall);
                        }
                        output[out_pos] = self.nodes[node_idx].value;
                        out_pos += 1;
                        node_idx = root_idx;
                    } else {
                        return Err(PzError::InvalidInput);
                    }
                }
            }
        }

        Ok(out_pos)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_from_empty() {
        let tree = HuffmanTree::from_data(&[]);
        assert!(tree.is_none());
    }

    #[test]
    fn test_build_single_symbol() {
        let input = vec![b'a'; 10];
        let tree = HuffmanTree::from_data(&input).unwrap();
        assert_eq!(tree.leaf_count, 1);
        let (_, bits) = tree.get_code(b'a');
        assert_eq!(bits, 1);
    }

    #[test]
    fn test_build_two_symbols() {
        let input = b"aabb";
        let tree = HuffmanTree::from_data(input).unwrap();
        assert_eq!(tree.leaf_count, 2);
        let (_, bits_a) = tree.get_code(b'a');
        let (_, bits_b) = tree.get_code(b'b');
        // Both should have 1-bit codes
        assert_eq!(bits_a, 1);
        assert_eq!(bits_b, 1);
    }

    #[test]
    fn test_prefix_free() {
        // Verify no codeword is a prefix of another
        let input = b"aaabbbccddeef";
        let tree = HuffmanTree::from_data(input).unwrap();

        let mut codes: Vec<(u32, u8)> = Vec::new();
        for i in 0..=255u8 {
            let (cw, bits) = tree.get_code(i);
            if bits > 0 {
                codes.push((cw, bits));
            }
        }

        // Check no code is a prefix of another
        for i in 0..codes.len() {
            for j in 0..codes.len() {
                if i == j {
                    continue;
                }
                let (cw_i, bits_i) = codes[i];
                let (cw_j, bits_j) = codes[j];
                if bits_i <= bits_j {
                    let shifted = cw_j >> (bits_j - bits_i);
                    assert_ne!(
                        shifted, cw_i,
                        "code {} is prefix of code {}",
                        i, j
                    );
                }
            }
        }
    }

    #[test]
    fn test_encode_decode_round_trip() {
        let input = b"hello, world!";
        let tree = HuffmanTree::from_data(input).unwrap();
        let (encoded, total_bits) = tree.encode(input).unwrap();
        let decoded = tree.decode(&encoded, total_bits).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_decode_single_symbol() {
        let input = vec![b'x'; 50];
        let tree = HuffmanTree::from_data(&input).unwrap();
        let (encoded, total_bits) = tree.encode(&input).unwrap();
        let decoded = tree.decode(&encoded, total_bits).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_decode_two_symbols() {
        let input = b"ababababab";
        let tree = HuffmanTree::from_data(input).unwrap();
        let (encoded, total_bits) = tree.encode(input).unwrap();
        let decoded = tree.decode(&encoded, total_bits).unwrap();
        assert_eq!(&decoded, input);
    }

    #[test]
    fn test_encode_decode_all_bytes() {
        let input: Vec<u8> = (0..=255).collect();
        let tree = HuffmanTree::from_data(&input).unwrap();
        let (encoded, total_bits) = tree.encode(&input).unwrap();
        let decoded = tree.decode(&encoded, total_bits).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_decode_skewed_distribution() {
        // 'a' appears much more often, should get shorter code
        let mut input = vec![b'a'; 100];
        input.push(b'b');
        input.push(b'c');

        let tree = HuffmanTree::from_data(&input).unwrap();
        let (_, bits_a) = tree.get_code(b'a');
        let (_, bits_b) = tree.get_code(b'b');
        assert!(
            bits_a <= bits_b,
            "more frequent symbol should have shorter code: a={}, b={}",
            bits_a,
            bits_b
        );

        let (encoded, total_bits) = tree.encode(&input).unwrap();
        let decoded = tree.decode(&encoded, total_bits).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_encode_to_buf() {
        let input = b"hello";
        let tree = HuffmanTree::from_data(input).unwrap();
        let mut buf = vec![0u8; 1024];
        let bits = tree.encode_to_buf(input, &mut buf).unwrap();
        let decoded = tree.decode(&buf, bits).unwrap();
        assert_eq!(&decoded, input);
    }

    #[test]
    fn test_decode_to_buf() {
        let input = b"hello";
        let tree = HuffmanTree::from_data(input).unwrap();
        let (encoded, total_bits) = tree.encode(input).unwrap();
        let mut buf = vec![0u8; 1024];
        let size = tree.decode_to_buf(&encoded, total_bits, &mut buf).unwrap();
        assert_eq!(&buf[..size], input);
    }

    #[test]
    fn test_encode_buf_too_small() {
        let input = b"hello, world! this is a longer string";
        let tree = HuffmanTree::from_data(input).unwrap();
        let mut buf = vec![0u8; 1]; // too small
        let result = tree.encode_to_buf(input, &mut buf);
        assert_eq!(result, Err(PzError::BufferTooSmall));
    }

    #[test]
    fn test_frequency_reconstruction() {
        let input = b"aaabbcc";
        let tree = HuffmanTree::from_data(input).unwrap();
        let freqs = tree.serialize_frequencies();

        // Reconstruct tree from frequencies
        let mut freq_table = FrequencyTable::new();
        freq_table.byte = freqs;
        freq_table.total = freqs.iter().map(|&f| f as u64).sum();
        freq_table.used = freqs.iter().filter(|&&f| f > 0).count() as u32;

        let tree2 = HuffmanTree::from_frequency_table(&freq_table).unwrap();

        // Both trees should produce identical encoding
        let (encoded1, bits1) = tree.encode(input).unwrap();
        let (encoded2, bits2) = tree2.encode(input).unwrap();
        assert_eq!(encoded1, encoded2);
        assert_eq!(bits1, bits2);
    }

    #[test]
    fn test_compression_ratio() {
        // Skewed data should compress well
        let mut input = vec![b'a'; 1000];
        input.extend(vec![b'b'; 10]);
        input.extend(vec![b'c'; 5]);

        let tree = HuffmanTree::from_data(&input).unwrap();
        let (encoded, _) = tree.encode(&input).unwrap();

        // Encoded should be much smaller than input
        assert!(
            encoded.len() < input.len(),
            "encoded {} bytes, input {} bytes",
            encoded.len(),
            input.len()
        );
    }

    #[test]
    fn test_round_trip_binary_data() {
        // Pseudo-random binary data
        let input: Vec<u8> = (0..500)
            .map(|i| ((i * 17 + 31) % 256) as u8)
            .collect();
        let tree = HuffmanTree::from_data(&input).unwrap();
        let (encoded, total_bits) = tree.encode(&input).unwrap();
        let decoded = tree.decode(&encoded, total_bits).unwrap();
        assert_eq!(decoded, input);
    }
}
