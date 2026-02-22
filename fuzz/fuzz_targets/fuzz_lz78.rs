#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::lz78;

/// LZ78 encode/decode roundtrip and decode crash resistance.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Roundtrip
    let encoded = match lz78::encode(input) {
        Ok(e) => e,
        Err(_) => return,
    };
    let decoded = lz78::decode(&encoded)
        .expect("LZ78 decode failed on valid encoded data");
    assert_eq!(input, decoded.as_slice(), "LZ78 roundtrip mismatch");

    // Crash resistance
    let _ = lz78::decode(data);
});
