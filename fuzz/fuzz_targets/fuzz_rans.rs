#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::rans;

/// rANS encode/decode roundtrip (basic + interleaved + chunked) and
/// decode of arbitrary bytes (crash resistance).
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Basic rANS roundtrip
    let encoded = rans::encode(input);
    let decoded = rans::decode(&encoded, input.len())
        .expect("rANS decode failed on valid encoded data");
    assert_eq!(input, decoded.as_slice(), "rANS roundtrip mismatch");

    // Interleaved rANS roundtrip
    let encoded = rans::encode_interleaved(input);
    let decoded = rans::decode_interleaved(&encoded, input.len())
        .expect("interleaved rANS decode failed");
    assert_eq!(input, decoded.as_slice(), "interleaved rANS roundtrip mismatch");

    // Chunked rANS roundtrip
    let encoded = rans::encode_chunked(input, 4, rans::DEFAULT_SCALE_BITS, 1024);
    let decoded = rans::decode_chunked(&encoded)
        .expect("chunked rANS decode failed");
    assert_eq!(input, decoded.as_slice(), "chunked rANS roundtrip mismatch");

    // Arbitrary bytes to decode — must not panic
    let _ = rans::decode(data, data.len());
    let _ = rans::decode_interleaved(data, data.len());
    let _ = rans::decode_chunked(data);
});
