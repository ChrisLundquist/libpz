#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::lzss;

/// LZSS encode/decode roundtrip and decode crash resistance.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Roundtrip
    let encoded = match lzss::encode(input) {
        Ok(e) => e,
        Err(_) => return,
    };
    let decoded = lzss::decode(&encoded)
        .expect("LZSS decode failed on valid encoded data");
    assert_eq!(input, decoded.as_slice(), "LZSS roundtrip mismatch");

    // Crash resistance
    let _ = lzss::decode(data);
});
