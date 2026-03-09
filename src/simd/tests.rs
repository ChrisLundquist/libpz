use super::*;

#[test]
fn test_dispatcher_creation() {
    let d = Dispatcher::new();
    // Just verify it doesn't panic
    let _ = d.level();
}

#[test]
fn test_scalar_byte_frequencies() {
    let data = b"hello world";
    let freqs = scalar::byte_frequencies(data);
    assert_eq!(freqs[b'h' as usize], 1);
    assert_eq!(freqs[b'l' as usize], 3);
    assert_eq!(freqs[b'o' as usize], 2);
    assert_eq!(freqs[b' ' as usize], 1);
}

#[test]
fn test_scalar_compare_bytes() {
    let a = b"hello world";
    let b = b"hello there";
    assert_eq!(scalar::compare_bytes(a, b, a.len()), 6);

    let c = b"hello world";
    assert_eq!(scalar::compare_bytes(a, c, a.len()), a.len());
}

#[test]
fn test_scalar_sum_u32() {
    let data = vec![1u32, 2, 3, 4, 5];
    assert_eq!(scalar::sum_u32(&data), 15);
}

#[test]
fn test_dispatcher_byte_frequencies_matches_scalar() {
    let d = Dispatcher::new();
    // Test with various sizes to exercise SIMD paths and scalar tails
    for size in [0, 1, 3, 15, 16, 17, 31, 32, 33, 63, 64, 100, 1000, 65536] {
        let data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        let simd_result = d.byte_frequencies(&data);
        let scalar_result = scalar::byte_frequencies(&data);
        assert_eq!(simd_result, scalar_result, "mismatch at size {}", size);
    }
}

#[test]
fn test_dispatcher_compare_bytes_matches_scalar() {
    let d = Dispatcher::new();

    // Identical slices of various lengths (using legacy 258 limit)
    for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 100, 258] {
        let a: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
        let b = a.clone();
        let max = a.len().min(b.len()).min(MAX_COMPARE_LEN);
        let simd_result = d.compare_bytes(&a, &b, MAX_COMPARE_LEN);
        assert_eq!(simd_result, max, "identical mismatch at len {}", len);
    }

    // Mismatch at specific positions
    for mismatch_pos in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 100] {
        let len = mismatch_pos + 10;
        let a: Vec<u8> = (0..len).map(|i| (i % 200) as u8).collect();
        let mut b = a.clone();
        b[mismatch_pos] = 255; // force mismatch
        let simd_result = d.compare_bytes(&a, &b, MAX_COMPARE_LEN);
        let scalar_result = scalar::compare_bytes(&a, &b, a.len().min(MAX_COMPARE_LEN));
        assert_eq!(
            simd_result, scalar_result,
            "mismatch_pos={} expected={}",
            mismatch_pos, scalar_result
        );
    }
}

#[test]
fn test_compare_bytes_extended_limit() {
    let d = Dispatcher::new();

    // With a large limit, identical slices should match fully
    for len in [500, 1000, 8192, 65535] {
        let a: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
        let b = a.clone();
        let result = d.compare_bytes(&a, &b, u16::MAX as usize);
        assert_eq!(result, len, "full match expected at len {}", len);
    }

    // Mismatch at position 1000 with large limit
    let a: Vec<u8> = vec![0xAA; 8192];
    let mut b = a.clone();
    b[1000] = 0xBB;
    let result = d.compare_bytes(&a, &b, u16::MAX as usize);
    assert_eq!(result, 1000, "should stop at mismatch pos 1000");

    // Small limit still caps even on matching data
    let result = d.compare_bytes(&a, &a, 258);
    assert_eq!(result, 258, "limit=258 should cap at 258");
}

#[test]
fn test_dispatcher_sum_u32_matches_scalar() {
    let d = Dispatcher::new();

    for size in [0, 1, 3, 4, 7, 8, 9, 15, 16, 100, 1000] {
        let data: Vec<u32> = (0..size).map(|i| (i * 7 + 3) as u32).collect();
        let simd_result = d.sum_u32(&data);
        let scalar_result = scalar::sum_u32(&data);
        assert_eq!(simd_result, scalar_result, "mismatch at size {}", size);
    }
}

#[test]
fn test_dispatcher_sum_u32_large_values() {
    let d = Dispatcher::new();
    // Test with values near u32::MAX to verify u64 accumulation
    let data = vec![u32::MAX; 100];
    let result = d.sum_u32(&data);
    assert_eq!(result, u32::MAX as u64 * 100);
}

#[test]
fn test_dispatcher_byte_frequencies_all_same() {
    let d = Dispatcher::new();
    let data = vec![42u8; 10000];
    let freqs = d.byte_frequencies(&data);
    assert_eq!(freqs[42], 10000);
    for (i, &f) in freqs.iter().enumerate() {
        if i != 42 {
            assert_eq!(f, 0);
        }
    }
}

// -----------------------------------------------------------------------
// rANS 4-way SIMD decode tests
// -----------------------------------------------------------------------

#[test]
fn test_rans_decode_4way_sse2_matches_scalar() {
    // Encode a known sequence with the CPU rANS encoder, then decode
    // both paths and assert byte-for-byte identical output.
    use crate::rans;

    let input: Vec<u8> = (0..1024).map(|i| (i % 26 + b'a' as usize) as u8).collect();
    let freq_table = crate::frequency::get_frequency(&input);
    let norm = rans::normalize_frequencies(&freq_table, rans::DEFAULT_SCALE_BITS).unwrap();
    let (word_streams, final_states) = rans::rans_encode_interleaved(&input, &norm, 4);

    let lookup = rans::build_symbol_lookup(&norm);
    let streams_arr: [&[u16]; 4] = [
        &word_streams[0],
        &word_streams[1],
        &word_streams[2],
        &word_streams[3],
    ];
    let states_arr: [u32; 4] = [
        final_states[0],
        final_states[1],
        final_states[2],
        final_states[3],
    ];

    // Scalar path
    let scalar_out = rans_decode_4way(
        &streams_arr,
        &states_arr,
        &norm.freq,
        &norm.cum,
        &lookup,
        norm.scale_bits as u32,
        input.len(),
    )
    .unwrap();

    // SSE2 path (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    {
        let sse2_out = unsafe {
            rans_decode_4way_sse2(
                &streams_arr,
                &states_arr,
                &norm.freq,
                &norm.cum,
                &lookup,
                norm.scale_bits as u32,
                input.len(),
            )
        }
        .unwrap();
        assert_eq!(
            scalar_out, sse2_out,
            "SSE2 and scalar rANS decode must produce identical output"
        );
    }

    assert_eq!(scalar_out, input);
}

#[test]
fn test_rans_decode_4way_sse2_round_trips_varied_distributions() {
    // Test with skewed distributions (few high-freq symbols) and flat
    // distributions (all 256 symbols equally likely), as these exercise
    // different freq/cum table patterns.
    use crate::rans;

    for seed in [0u8, 42, 128, 255] {
        let input: Vec<u8> = (0..4096)
            .map(|i| {
                let val = ((i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(seed as u64))
                    >> 56;
                val as u8
            })
            .collect();
        let freq_table = crate::frequency::get_frequency(&input);
        let norm = rans::normalize_frequencies(&freq_table, rans::DEFAULT_SCALE_BITS).unwrap();
        let (word_streams, final_states) = rans::rans_encode_interleaved(&input, &norm, 4);

        let lookup = rans::build_symbol_lookup(&norm);
        let streams_arr: [&[u16]; 4] = [
            &word_streams[0],
            &word_streams[1],
            &word_streams[2],
            &word_streams[3],
        ];
        let states_arr: [u32; 4] = [
            final_states[0],
            final_states[1],
            final_states[2],
            final_states[3],
        ];

        let scalar_out = rans_decode_4way(
            &streams_arr,
            &states_arr,
            &norm.freq,
            &norm.cum,
            &lookup,
            norm.scale_bits as u32,
            input.len(),
        )
        .unwrap();
        assert_eq!(scalar_out, input, "round-trip failed for seed {}", seed);

        #[cfg(target_arch = "x86_64")]
        {
            let sse2_out = unsafe {
                rans_decode_4way_sse2(
                    &streams_arr,
                    &states_arr,
                    &norm.freq,
                    &norm.cum,
                    &lookup,
                    norm.scale_bits as u32,
                    input.len(),
                )
            }
            .unwrap();
            assert_eq!(sse2_out, input, "SSE2 round-trip failed for seed {}", seed);
        }
    }
}
