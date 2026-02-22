#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::rle;

/// RLE encode/decode roundtrip and decode crash resistance.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Roundtrip
    let encoded = rle::encode(input);
    let decoded = rle::decode(&encoded)
        .expect("RLE decode failed on valid encoded data");
    assert_eq!(input, decoded.as_slice(), "RLE roundtrip mismatch");

    // Crash resistance
    let _ = rle::decode(data);
});
