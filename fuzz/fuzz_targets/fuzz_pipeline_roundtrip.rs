#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::pipeline::{compress, decompress, Pipeline};

/// Fuzz all pipelines: compress random input, then decompress and verify roundtrip.
fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }
    // Cap input size to avoid timeouts on very large inputs.
    let input = if data.len() > 64 * 1024 { &data[..64 * 1024] } else { data };

    let pipelines = [
        Pipeline::Deflate,
        Pipeline::Bw,
        Pipeline::Bbw,
        Pipeline::Lzf,
        Pipeline::Lzfi,
        Pipeline::LzssR,
        Pipeline::LzSeqR,
    ];

    for pipeline in pipelines {
        let compressed = match compress(input, pipeline) {
            Ok(c) => c,
            Err(_) => continue,
        };
        let decompressed = match decompress(&compressed) {
            Ok(d) => d,
            Err(e) => panic!(
                "roundtrip failure for {:?}: decompress returned {:?}",
                pipeline, e
            ),
        };
        assert_eq!(
            input,
            decompressed.as_slice(),
            "roundtrip mismatch for {:?}",
            pipeline
        );
    }
});
