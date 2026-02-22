#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::mtf;

/// MTF encode/decode roundtrip (MTF is always bijective, never fails).
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    let encoded = mtf::encode(input);
    let decoded = mtf::decode(&encoded);
    assert_eq!(input, decoded.as_slice(), "MTF roundtrip mismatch");
});
