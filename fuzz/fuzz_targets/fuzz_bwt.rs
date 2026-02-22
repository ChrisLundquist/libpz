#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::bwt;

/// BWT encode/decode roundtrip + bijective variant.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > 64 * 1024 {
        return;
    }

    // Standard BWT roundtrip
    if let Some(result) = bwt::encode(data) {
        let decoded = bwt::decode(&result.output, result.primary_index)
            .expect("BWT decode failed on valid encoded data");
        assert_eq!(data, decoded.as_slice(), "BWT roundtrip mismatch");
    }

    // Bijective BWT roundtrip
    if let Some((encoded, factor_lengths)) = bwt::encode_bijective(data) {
        let decoded = bwt::decode_bijective(&encoded, &factor_lengths)
            .expect("bijective BWT decode failed on valid encoded data");
        assert_eq!(data, decoded.as_slice(), "bijective BWT roundtrip mismatch");
    }
});
