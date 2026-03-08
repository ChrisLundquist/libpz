use super::*;

#[test]
fn test_encode_empty() {
    assert!(encode(&[]).is_none());
}

#[test]
fn test_single_byte() {
    let result = encode(b"a").unwrap();
    assert_eq!(result.data, vec![b'a']);
    assert_eq!(result.primary_index, 0);
    let decoded = decode(&result.data, result.primary_index).unwrap();
    assert_eq!(decoded, vec![b'a']);
}

#[test]
fn test_banana() {
    let input = b"banana";
    let result = encode(input).unwrap();
    // The BWT of "banana" should be "nnbaaa" with the standard rotation sort
    // Rotations sorted:
    // 0: abanana -> a at end = a
    // 1: ananab  -> b at end
    // 2: anaban  -> n at end
    // 3: banana  -> a at end
    // 4: nabana  -> a at end
    // 5: nanaba  -> a at end
    // Wait, let me re-derive this properly.
    // "banana" rotations:
    // banana -> last = a
    // ananab -> last = b
    // nanaba -> last = a
    // anaban -> last = n
    // nabana -> last = a
    // abanan -> last = n
    // Sorted: abanan, anaban, ananab, banana, nabana, nanaba
    // Last column: n, n, b, a, a, a = "nnbaaa"
    // Original "banana" is at position 3
    assert_eq!(result.data, b"nnbaaa");
    assert_eq!(result.primary_index, 3);

    let decoded = decode(&result.data, result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_round_trip_all_same() {
    let input = vec![b'x'; 10];
    let result = encode(&input).unwrap();
    assert_eq!(result.data, input); // BWT of all-same is all-same
    let decoded = decode(&result.data, result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_round_trip_longer_text() {
    let input = b"the quick brown fox jumps over the lazy dog";
    let result = encode(input).unwrap();
    let decoded = decode(&result.data, result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_round_trip_binary_data() {
    let input: Vec<u8> = (0..=255).collect();
    let result = encode(&input).unwrap();
    let decoded = decode(&result.data, result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_decode_invalid_primary_index() {
    let result = decode(&[1, 2, 3], 5);
    assert_eq!(result, Err(PzError::InvalidInput));
}

#[test]
fn test_decode_to_buf_too_small() {
    let input = b"banana";
    let result = encode(input).unwrap();
    let mut buf = vec![0u8; 2];
    assert_eq!(
        decode_to_buf(&result.data, result.primary_index, &mut buf),
        Err(PzError::BufferTooSmall)
    );
}

#[test]
fn test_bwt_clusters_bytes() {
    // BWT should cluster identical bytes together
    let input = b"abracadabra";
    let result = encode(input).unwrap();
    let decoded = decode(&result.data, result.primary_index).unwrap();
    assert_eq!(decoded, input);

    // Count runs in BWT output vs input
    fn count_runs(data: &[u8]) -> usize {
        if data.is_empty() {
            return 0;
        }
        let mut runs = 1;
        for i in 1..data.len() {
            if data[i] != data[i - 1] {
                runs += 1;
            }
        }
        runs
    }

    let input_runs = count_runs(input);
    let bwt_runs = count_runs(&result.data);
    // BWT output should have fewer or equal runs (better clustering)
    assert!(
        bwt_runs <= input_runs,
        "BWT should cluster: input_runs={}, bwt_runs={}",
        input_runs,
        bwt_runs
    );
}

#[test]
fn test_suffix_array_basic() {
    let sa = build_suffix_array(b"banana");
    // For circular rotations of "banana":
    // Sorted: abanan(5), anaban(3), ananab(1), banana(0), nabana(4), nanaba(2)
    assert_eq!(sa, vec![5, 3, 1, 0, 4, 2]);
}

#[test]
fn test_sais_and_naive_both_produce_valid_bwt() {
    // SA-IS (doubled+sentinel) and naive (circular) may produce different
    // suffix array orderings for texts with repeated substrings, but both
    // must produce valid BWT outputs that round-trip correctly.
    let test_cases: Vec<&[u8]> = vec![
        b"banana",
        b"abcabc",
        b"aaaaaa",
        b"abcdefghijklmnopqrstuvwxyz",
        b"the quick brown fox",
        b"abababababab",
        b"abracadabra",
        b"mississippi",
    ];
    for input in test_cases {
        // Test with SA-IS (current build_suffix_array)
        let result = encode(input).unwrap();
        let decoded = decode(&result.data, result.primary_index).unwrap();
        assert_eq!(
            decoded,
            input,
            "SA-IS BWT round-trip failed on {:?}",
            std::str::from_utf8(input).unwrap_or("<binary>")
        );

        // Test with naive (old build_suffix_array_naive)
        let naive_sa = build_suffix_array_naive(input);
        let n = input.len();
        let mut naive_bwt = Vec::with_capacity(n);
        let mut naive_primary = 0u32;
        for (i, &sa_val) in naive_sa.iter().enumerate() {
            if sa_val == 0 {
                naive_primary = i as u32;
                naive_bwt.push(input[n - 1]);
            } else {
                naive_bwt.push(input[sa_val - 1]);
            }
        }
        let naive_decoded = decode(&naive_bwt, naive_primary).unwrap();
        assert_eq!(
            naive_decoded,
            input,
            "Naive BWT round-trip failed on {:?}",
            std::str::from_utf8(input).unwrap_or("<binary>")
        );
    }
}

#[test]
fn test_sais_matches_naive_on_distinct_text() {
    // For texts where all rotations are distinct, SA-IS and naive
    // must produce the exact same suffix array.
    let input: Vec<u8> = (0..=255).collect();
    let sais = build_suffix_array(&input);
    let naive = build_suffix_array_naive(&input);
    assert_eq!(
        sais, naive,
        "SA-IS and naive disagree on all-bytes (distinct rotations)"
    );
}

#[test]
fn test_round_trip_large() {
    // Test with a larger input to exercise SA-IS performance
    let mut input = Vec::new();
    for _ in 0..100 {
        input.extend(b"The quick brown fox jumps over the lazy dog. ");
    }
    let result = encode(&input).unwrap();
    let decoded = decode(&result.data, result.primary_index).unwrap();
    assert_eq!(decoded, input);
}

// --- Lyndon factorization tests ---

#[test]
fn test_lyndon_factorize_basic() {
    // "abab" → factors: "ab", "ab" (non-increasing Lyndon words)
    let factors = lyndon_factorize(b"abab");
    let words: Vec<&[u8]> = factors.iter().map(|&(s, l)| &b"abab"[s..s + l]).collect();
    assert_eq!(words, vec![b"ab" as &[u8], b"ab"]);
}

#[test]
fn test_lyndon_factorize_single_char() {
    let factors = lyndon_factorize(b"aaaa");
    // Each 'a' is its own Lyndon word (can't be further decomposed)
    assert_eq!(factors.len(), 4);
    for &(_, len) in &factors {
        assert_eq!(len, 1);
    }
}

#[test]
fn test_lyndon_factorize_descending() {
    // "dcba" → each char is a factor (strictly descending = all length 1)
    let factors = lyndon_factorize(b"dcba");
    assert_eq!(factors.len(), 4);
}

#[test]
fn test_lyndon_factorize_lyndon_word() {
    // "abc" is itself a Lyndon word (lex smallest rotation)
    let factors = lyndon_factorize(b"abc");
    assert_eq!(factors, vec![(0, 3)]);
}

#[test]
fn test_lyndon_factorize_banana() {
    let input = b"banana";
    let factors = lyndon_factorize(input);
    // Verify: concatenation of factors equals input
    let mut reconstructed = Vec::new();
    for &(start, len) in &factors {
        reconstructed.extend_from_slice(&input[start..start + len]);
    }
    assert_eq!(reconstructed, input);

    // Verify non-increasing property
    for i in 1..factors.len() {
        let prev = &input[factors[i - 1].0..factors[i - 1].0 + factors[i - 1].1];
        let curr = &input[factors[i].0..factors[i].0 + factors[i].1];
        assert!(
            prev >= curr,
            "Lyndon factors not non-increasing: {:?} < {:?}",
            prev,
            curr
        );
    }
}

// --- Bijective BWT tests ---

#[test]
fn test_bijective_bwt_empty() {
    assert!(encode_bijective(&[]).is_none());
}

#[test]
fn test_bijective_bwt_single() {
    let (data, factors) = encode_bijective(b"a").unwrap();
    assert_eq!(data, vec![b'a']);
    assert_eq!(factors, vec![1]);
    let decoded = decode_bijective(&data, &factors).unwrap();
    assert_eq!(decoded, b"a");
}

#[test]
fn test_bijective_bwt_round_trip_simple() {
    let test_cases: &[&[u8]] = &[b"ab", b"abc", b"abcabc", b"hello", b"banana"];
    for input in test_cases {
        let (bwt_data, factor_lens) = encode_bijective(input).unwrap();
        let decoded = decode_bijective(&bwt_data, &factor_lens).unwrap();
        assert_eq!(
            &decoded,
            *input,
            "Bijective BWT round-trip failed on {:?}",
            std::str::from_utf8(input).unwrap_or("<binary>")
        );
    }
}

#[test]
fn test_bijective_decode_to_buf_too_small() {
    let input = b"hello";
    let (bwt_data, factor_lens) = encode_bijective(input).unwrap();
    let mut buf = vec![0u8; 2];
    assert_eq!(
        decode_bijective_to_buf(&bwt_data, &factor_lens, &mut buf),
        Err(PzError::BufferTooSmall)
    );
}

#[test]
fn test_circular_sa_produces_valid_bwt() {
    let test_cases: Vec<&[u8]> = vec![
        b"banana",
        b"abcabc",
        b"abcdefghijklmnopqrstuvwxyz",
        b"the quick brown fox",
        b"abracadabra",
        b"mississippi",
        b"ab",
        b"abc",
        b"hello world hello world",
    ];
    for input in test_cases {
        let sa = build_circular_suffix_array(input);
        let n = input.len();

        let mut bwt_data = Vec::with_capacity(n);
        let mut primary_index = 0u32;
        for (i, &sa_val) in sa.iter().enumerate() {
            if sa_val == 0 {
                primary_index = i as u32;
                bwt_data.push(input[n - 1]);
            } else {
                bwt_data.push(input[sa_val - 1]);
            }
        }

        let decoded = decode(&bwt_data, primary_index).unwrap();
        assert_eq!(
            decoded,
            input,
            "Circular SA BWT round-trip failed on {:?}",
            std::str::from_utf8(input).unwrap_or("<binary>")
        );
    }
}

#[test]
fn test_circular_sa_matches_on_distinct_rotations() {
    let input: Vec<u8> = (0..=255).collect();
    let sa_doubled = build_suffix_array(&input);
    let sa_circular = build_circular_suffix_array(&input);
    assert_eq!(
        sa_doubled, sa_circular,
        "SA mismatch on all-bytes input (all rotations distinct)"
    );
}

#[test]
fn test_bijective_parallel_decode_matches_sequential() {
    let test_cases: &[&[u8]] = &[
        b"the quick brown fox jumps over the lazy dog",
        b"abracadabra abracadabra abracadabra",
        b"hello world hello world hello world hello world",
    ];
    for input in test_cases {
        let (bwt_data, factor_lens) = encode_bijective(input).unwrap();
        let sequential = decode_bijective(&bwt_data, &factor_lens).unwrap();
        let parallel = decode_bijective_parallel(&bwt_data, &factor_lens, 4).unwrap();
        assert_eq!(
            sequential,
            parallel,
            "Parallel decode differs from sequential on {:?}",
            std::str::from_utf8(input).unwrap_or("<binary>")
        );
        assert_eq!(&sequential, *input);
    }
}

#[test]
fn test_bijective_parallel_decode_large() {
    let mut input = Vec::new();
    for _ in 0..200 {
        input.extend_from_slice(b"The quick brown fox jumps over the lazy dog. ");
    }
    let (bwt_data, factor_lens) = encode_bijective(&input).unwrap();
    let decoded = decode_bijective_parallel(&bwt_data, &factor_lens, 4).unwrap();
    assert_eq!(decoded, input);
}

#[test]
fn test_bijective_small_factors() {
    let test_cases: &[&[u8]] = &[
        b"dcba",   // 4 single-char factors (descending)
        b"ab",     // one len-2 factor
        b"abc",    // one len-3 factor
        b"abcba",  // "abc" (len 3) + "b" (len 1) + "a" (len 1)
        b"abd",    // one len-3 factor
        b"zy",     // one len-2 factor (z < y? no, z > y, so two len-1 factors)
        b"yz",     // one len-2 factor (y < z, Lyndon word)
        b"abcabc", // "abcabc" or "abc","abc" — len-3 factors
        b"ba",     // two len-1 factors (descending)
        b"cba",    // three len-1 factors (descending)
    ];
    for input in test_cases {
        let (bwt_data, factor_lens) = encode_bijective(input).unwrap();
        let decoded = decode_bijective(&bwt_data, &factor_lens).unwrap();
        assert_eq!(
            &decoded,
            *input,
            "Small factor round-trip failed on {:?}",
            std::str::from_utf8(input).unwrap_or("<binary>")
        );

        let total: usize = factor_lens.iter().sum();
        assert_eq!(total, input.len());
    }
}
