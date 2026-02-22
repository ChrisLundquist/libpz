#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::huffman::HuffmanTree;

/// Huffman encode/decode roundtrip and decode crash resistance.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Build tree and roundtrip
    let tree = HuffmanTree::from_frequencies(&pz::frequency::byte_frequencies(input));
    let (encoded, total_bits) = match tree.encode(input) {
        Ok(r) => r,
        Err(_) => return,
    };
    let decoded = tree
        .decode(&encoded, total_bits)
        .expect("Huffman decode failed on valid encoded data");
    assert_eq!(input, decoded.as_slice(), "Huffman roundtrip mismatch");
});
