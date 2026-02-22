#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::fse;

/// FSE encode/decode roundtrip (basic + interleaved) and crash resistance.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Basic FSE roundtrip
    let encoded = fse::encode(input);
    let decoded = fse::decode(&encoded, input.len())
        .expect("FSE decode failed on valid encoded data");
    assert_eq!(input, decoded.as_slice(), "FSE roundtrip mismatch");

    // Interleaved FSE roundtrip
    let encoded = fse::encode_interleaved(input);
    let decoded = fse::decode_interleaved(&encoded, input.len())
        .expect("interleaved FSE decode failed");
    assert_eq!(input, decoded.as_slice(), "interleaved FSE roundtrip mismatch");

    // Arbitrary bytes to decode — must not panic
    let _ = fse::decode(data, data.len());
    let _ = fse::decode_interleaved(data, data.len());
});
