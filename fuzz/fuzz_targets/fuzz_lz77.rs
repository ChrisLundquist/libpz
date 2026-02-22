#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::lz77;

/// LZ77 compress/decompress roundtrip and decompress crash resistance.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    // Roundtrip
    let compressed = match lz77::compress_lazy(input) {
        Ok(c) => c,
        Err(_) => return,
    };
    let decompressed = lz77::decompress(&compressed)
        .expect("LZ77 decompress failed on valid compressed data");
    assert_eq!(input, decompressed.as_slice(), "LZ77 roundtrip mismatch");

    // Crash resistance: feed arbitrary bytes to decompress
    let _ = lz77::decompress(data);
});
