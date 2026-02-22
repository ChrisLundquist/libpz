#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::lzseq;

/// LzSeq encode/decode roundtrip and decode crash resistance.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Roundtrip: encode then decode
    let encoded = match lzseq::encode(input) {
        Ok(e) => e,
        Err(_) => return,
    };
    let decoded = lzseq::decode(
        &encoded.flags,
        &encoded.literals,
        &encoded.offset_codes,
        &encoded.offset_extra,
        &encoded.length_codes,
        &encoded.length_extra,
        encoded.num_tokens,
        encoded.num_matches,
        input.len(),
    )
    .expect("LzSeq decode failed on valid encoded data");
    assert_eq!(input, decoded.as_slice(), "LzSeq roundtrip mismatch");

    // Crash resistance: feed arbitrary bytes to decode
    // Use first byte as num_tokens proxy, split rest across streams
    if data.len() >= 7 {
        let num_tokens = u32::from(data[0]).wrapping_mul(4);
        let num_matches = num_tokens / 2;
        let chunk = data.len() / 6;
        let _ = lzseq::decode(
            &data[..chunk],
            &data[chunk..chunk * 2],
            &data[chunk * 2..chunk * 3],
            &data[chunk * 3..chunk * 4],
            &data[chunk * 4..chunk * 5],
            &data[chunk * 5..],
            num_tokens,
            num_matches,
            1024,
        );
    }
});
