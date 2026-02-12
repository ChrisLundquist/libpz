/// Burrows-Wheeler Transform (BWT).
///
/// The BWT is a reversible transformation that rearranges bytes to group
/// similar contexts together. When followed by MTF and RLE, the resulting
/// data compresses very well with entropy coding (similar to bzip2).
///
/// **Forward transform:**
/// 1. Form all rotations of the input (conceptually).
/// 2. Sort the rotations lexicographically.
/// 3. Output the last column of the sorted rotation matrix.
/// 4. Also output the index of the original string in the sorted order.
///
/// **Implementation:** Uses suffix array construction for efficiency.
/// The suffix array approach avoids materializing all rotations.
///
/// **Inverse transform:** Uses the "LF-mapping" technique:
/// 1. Sort the BWT output to get the first column.
/// 2. Build the LF-mapping: for each position in the last column,
///    find the corresponding position in the first column.
/// 3. Follow the mapping starting from the original index to recover the string.
use crate::{PzError, PzResult};

/// Result of a BWT forward transform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BwtResult {
    /// The transformed data (last column of sorted rotation matrix).
    pub data: Vec<u8>,
    /// The index of the original string in the sorted rotations.
    /// Needed for the inverse transform.
    pub primary_index: u32,
}

/// Perform the forward Burrows-Wheeler Transform.
///
/// Returns the transformed data and the primary index needed for inversion.
pub fn encode(input: &[u8]) -> Option<BwtResult> {
    if input.is_empty() {
        return None;
    }

    let n = input.len();

    // Build rotation suffix array using SA-IS O(n).
    //
    // SA-IS builds the suffix array of `text$` where `$` is a sentinel
    // smaller than all input bytes. The suffix array has n+1 entries.
    // Entry sa[0] == n (the sentinel suffix) is skipped.
    // The remaining n entries give the rotation order needed for BWT.
    //
    // For rotation i, the last character of the rotation is input[(i + n - 1) % n],
    // which equals input[i - 1] for i > 0, or input[n - 1] for i == 0.
    // The primary index is where sa[j] == 0.
    let sa = build_suffix_array(input);

    let mut bwt = Vec::with_capacity(n);
    let mut primary_index = 0u32;

    for (i, &sa_val) in sa.iter().enumerate() {
        if sa_val == 0 {
            primary_index = i as u32;
            bwt.push(input[n - 1]);
        } else {
            bwt.push(input[sa_val - 1]);
        }
    }

    Some(BwtResult {
        data: bwt,
        primary_index,
    })
}

/// Perform the inverse Burrows-Wheeler Transform.
///
/// Given the BWT output and primary index, recover the original data.
pub fn decode(bwt: &[u8], primary_index: u32) -> PzResult<Vec<u8>> {
    if bwt.is_empty() {
        return Ok(Vec::new());
    }

    let n = bwt.len();
    if primary_index as usize >= n {
        return Err(PzError::InvalidInput);
    }

    // Build the LF-mapping using counting sort.
    //
    // The first column F is the sorted version of the last column L (=bwt).
    // For each position i in L, LF[i] gives the corresponding row in F.
    //
    // Step 1: Count occurrences of each byte
    let mut counts = [0u32; 256];
    for &byte in bwt {
        counts[byte as usize] += 1;
    }

    // Step 2: Compute cumulative counts (starting positions in F)
    let mut cumul = [0u32; 256];
    let mut sum = 0u32;
    for (c, &count) in counts.iter().enumerate() {
        cumul[c] = sum;
        sum += count;
    }

    // Step 3: Build the LF-mapping (also called T-vector or transformation vector)
    // For each position i in L, LF[i] = cumul[L[i]]++
    let mut lf = vec![0u32; n];
    let mut running = cumul;
    for (i, &byte) in bwt.iter().enumerate() {
        lf[i] = running[byte as usize];
        running[byte as usize] += 1;
    }

    // Step 4: Follow the LF-mapping to recover the original string
    // Start at primary_index, follow lf[] for n steps (reading backwards)
    let mut output = vec![0u8; n];
    let mut idx = primary_index as usize;
    for i in (0..n).rev() {
        output[i] = bwt[idx];
        idx = lf[idx] as usize;
    }

    Ok(output)
}

/// Perform the inverse BWT into a pre-allocated output buffer.
///
/// Returns the number of bytes written.
pub fn decode_to_buf(bwt: &[u8], primary_index: u32, output: &mut [u8]) -> PzResult<usize> {
    if bwt.is_empty() {
        return Ok(0);
    }

    let n = bwt.len();
    if primary_index as usize >= n {
        return Err(PzError::InvalidInput);
    }
    if output.len() < n {
        return Err(PzError::BufferTooSmall);
    }

    let mut counts = [0u32; 256];
    for &byte in bwt {
        counts[byte as usize] += 1;
    }

    let mut cumul = [0u32; 256];
    let mut sum = 0u32;
    for (c, &count) in counts.iter().enumerate() {
        cumul[c] = sum;
        sum += count;
    }

    let mut lf = vec![0u32; n];
    let mut running = cumul;
    for (i, &byte) in bwt.iter().enumerate() {
        lf[i] = running[byte as usize];
        running[byte as usize] += 1;
    }

    let mut idx = primary_index as usize;
    for i in (0..n).rev() {
        output[i] = bwt[idx];
        idx = lf[idx] as usize;
    }

    Ok(n)
}

/// Build a rotation suffix array for the input using SA-IS.
///
/// Returns an array where sa[i] is the starting position of the i-th
/// smallest rotation of `input` (for BWT use).
///
/// Strategy: Build suffix array of `text + text + $` using SA-IS O(n),
/// then keep only entries where sa[i] < n. This gives the rotation
/// order because any suffix starting in the first copy of text has
/// enough characters to represent the full rotation, and the sentinel
/// breaks ties for equal rotations consistently.
fn build_suffix_array(input: &[u8]) -> Vec<usize> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Build text+text+$ by mapping bytes to 1..257 and appending sentinel 0.
    let mut doubled: Vec<usize> = Vec::with_capacity(2 * n + 1);
    for &b in input {
        doubled.push(b as usize + 1);
    }
    for &b in input {
        doubled.push(b as usize + 1);
    }
    doubled.push(0); // sentinel

    let alphabet_size = 257;
    let sa = sais_core(&doubled, alphabet_size);

    // Keep only entries where sa[i] < n (first copy of text).
    // Skip sa[0] which is the sentinel position (2*n).
    let mut result = Vec::with_capacity(n);
    for &s in &sa {
        if s < n {
            result.push(s);
        }
    }
    result
}

/// SA-IS core: build suffix array of integer array `text` with alphabet [0, alpha_size).
///
/// text must end with a unique sentinel (value 0) that is the smallest character.
fn sais_core(text: &[usize], alpha_size: usize) -> Vec<usize> {
    let n = text.len();
    if n <= 2 {
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![0];
        }
        // n == 2: sentinel is at [1], so SA = [1, 0]
        if text[0] > text[1] {
            return vec![1, 0];
        } else {
            return vec![0, 1];
        }
    }

    // Step 1: Classify each suffix as S-type or L-type.
    // A suffix i is S-type if text[i..] < text[i+1..], L-type otherwise.
    // The sentinel (last position) is always S-type.
    let mut is_s = vec![false; n];
    is_s[n - 1] = true; // sentinel is S-type
    for i in (0..n - 1).rev() {
        is_s[i] = if text[i] < text[i + 1] {
            true
        } else if text[i] > text[i + 1] {
            false
        } else {
            is_s[i + 1]
        };
    }

    // Step 2: Find LMS (Leftmost S-type) positions.
    // Position i is LMS if is_s[i] && !is_s[i-1] (i.e., S-type preceded by L-type).
    let mut lms_positions: Vec<usize> = Vec::new();
    for i in 1..n {
        if is_s[i] && !is_s[i - 1] {
            lms_positions.push(i);
        }
    }

    // Step 3: Compute bucket boundaries.
    let get_buckets = |alpha_size: usize, text: &[usize], end: bool| -> Vec<usize> {
        let mut buckets = vec![0usize; alpha_size];
        for &c in text {
            buckets[c] += 1;
        }
        let mut sum = 0;
        for b in buckets.iter_mut() {
            sum += *b;
            if end {
                *b = sum;
            } else {
                *b = sum - *b;
            }
        }
        buckets
    };

    // Step 4: Induced sort LMS suffixes.
    let mut sa = vec![usize::MAX; n];

    // Place LMS suffixes at the ends of their buckets (right to left).
    {
        let mut bucket_tails = get_buckets(alpha_size, text, true);
        for &lms in lms_positions.iter().rev() {
            let c = text[lms];
            bucket_tails[c] -= 1;
            sa[bucket_tails[c]] = lms;
        }
    }

    // Induce L-type suffixes (left to right scan).
    {
        let mut bucket_heads = get_buckets(alpha_size, text, false);
        for i in 0..n {
            if sa[i] == usize::MAX || sa[i] == 0 {
                continue;
            }
            let j = sa[i] - 1;
            if !is_s[j] {
                let c = text[j];
                sa[bucket_heads[c]] = j;
                bucket_heads[c] += 1;
            }
        }
    }

    // Induce S-type suffixes (right to left scan).
    {
        let mut bucket_tails = get_buckets(alpha_size, text, true);
        for i in (0..n).rev() {
            if sa[i] == usize::MAX || sa[i] == 0 {
                continue;
            }
            let j = sa[i] - 1;
            if is_s[j] {
                let c = text[j];
                bucket_tails[c] -= 1;
                sa[bucket_tails[c]] = j;
            }
        }
    }

    // Step 5: Compact sorted LMS suffixes and name them.
    // Extract sorted LMS positions from SA.
    let mut sorted_lms: Vec<usize> = Vec::with_capacity(lms_positions.len());
    for &s in &sa {
        if s != usize::MAX && s > 0 && is_s[s] && !is_s[s - 1] {
            sorted_lms.push(s);
        }
    }
    // Also include the sentinel if it's LMS (position 0 is never LMS by definition,
    // but position n-1 can be if n >= 2 and text[n-2] > text[n-1]).
    // Actually, the sentinel at n-1 is S-type, but is it LMS?
    // LMS requires i > 0 and is_s[i] && !is_s[i-1].
    // We already handled this in the loop above.

    // Assign names to LMS substrings.
    // Two LMS substrings are equal if they have the same characters
    // between their LMS positions (inclusive of both endpoints' LMS status).
    let mut name = 0usize;
    let mut lms_names = vec![usize::MAX; n]; // name for LMS position i
    let mut prev_lms = usize::MAX;

    for &pos in &sorted_lms {
        if prev_lms == usize::MAX || !lms_substrings_equal(text, &is_s, prev_lms, pos) {
            name += 1;
        }
        lms_names[pos] = name - 1;
        prev_lms = pos;
    }

    // Step 6: If names are not unique, recurse.
    if name < sorted_lms.len() {
        // Build the reduced string from LMS names in original text order.
        let mut reduced: Vec<usize> = Vec::with_capacity(lms_positions.len());
        for &lms in &lms_positions {
            reduced.push(lms_names[lms]);
        }

        // Recurse
        let reduced_sa = sais_core(&reduced, name);

        // Use reduced_sa to place LMS suffixes in the correct order.
        sa.fill(usize::MAX);
        {
            let mut bucket_tails = get_buckets(alpha_size, text, true);
            for i in (0..reduced_sa.len()).rev() {
                let lms = lms_positions[reduced_sa[i]];
                let c = text[lms];
                bucket_tails[c] -= 1;
                sa[bucket_tails[c]] = lms;
            }
        }
    } else {
        // Names are unique; we can directly place LMS suffixes.
        sa.fill(usize::MAX);
        {
            let mut bucket_tails = get_buckets(alpha_size, text, true);
            for &lms in lms_positions.iter().rev() {
                let c = text[lms];
                bucket_tails[c] -= 1;
                sa[bucket_tails[c]] = lms;
            }
        }
    }

    // Step 7: Final induced sort.
    // Induce L-type.
    {
        let mut bucket_heads = get_buckets(alpha_size, text, false);
        for i in 0..n {
            if sa[i] == usize::MAX || sa[i] == 0 {
                continue;
            }
            let j = sa[i] - 1;
            if !is_s[j] {
                let c = text[j];
                sa[bucket_heads[c]] = j;
                bucket_heads[c] += 1;
            }
        }
    }
    // Induce S-type.
    {
        let mut bucket_tails = get_buckets(alpha_size, text, true);
        for i in (0..n).rev() {
            if sa[i] == usize::MAX || sa[i] == 0 {
                continue;
            }
            let j = sa[i] - 1;
            if is_s[j] {
                let c = text[j];
                bucket_tails[c] -= 1;
                sa[bucket_tails[c]] = j;
            }
        }
    }

    sa
}

/// Check if two LMS substrings starting at positions `a` and `b` are equal.
///
/// An LMS substring is the substring from one LMS position to the next (inclusive).
fn lms_substrings_equal(text: &[usize], is_s: &[bool], a: usize, b: usize) -> bool {
    let n = text.len();
    let mut i = 0;
    loop {
        let ai = a + i;
        let bi = b + i;
        if ai >= n || bi >= n {
            return ai >= n && bi >= n;
        }
        if text[ai] != text[bi] || is_s[ai] != is_s[bi] {
            return false;
        }
        if i > 0 {
            // Check if we've reached the end of the LMS substring
            // (next LMS position or end of string).
            let a_is_lms = ai > 0 && is_s[ai] && !is_s[ai - 1];
            let b_is_lms = bi > 0 && is_s[bi] && !is_s[bi - 1];
            if a_is_lms || b_is_lms {
                return a_is_lms && b_is_lms;
            }
        }
        i += 1;
    }
}

/// Build suffix array using prefix-doubling (O(n log²n)).
///
/// Kept for cross-validation in tests. This was the original implementation
/// and produces the same rotation order as SA-IS for BWT.
#[cfg(test)]
fn build_suffix_array_naive(input: &[u8]) -> Vec<usize> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut rank = vec![0i64; n];
    let mut sa: Vec<usize> = (0..n).collect();

    for i in 0..n {
        rank[i] = input[i] as i64;
    }

    let mut tmp_rank = vec![0i64; n];
    let mut k = 1;

    while k < n {
        let r = &rank;
        sa.sort_by(|&a, &b| {
            let ra = (r[a], r[(a + k) % n]);
            let rb = (r[b], r[(b + k) % n]);
            ra.cmp(&rb)
        });

        tmp_rank[sa[0]] = 0;
        for i in 1..n {
            let prev = sa[i - 1];
            let curr = sa[i];
            if (rank[curr], rank[(curr + k) % n]) == (rank[prev], rank[(prev + k) % n]) {
                tmp_rank[curr] = tmp_rank[prev];
            } else {
                tmp_rank[curr] = tmp_rank[prev] + 1;
            }
        }

        rank.copy_from_slice(&tmp_rank);

        if rank[sa[n - 1]] as usize == n - 1 {
            break;
        }

        k *= 2;
    }

    sa
}

// --- Bijective BWT via Lyndon Factorization ---

/// Lyndon factorization using Duval's algorithm (O(n), O(1) extra space).
///
/// Returns a vector of (start, len) pairs representing the Lyndon factors.
/// By the Chen-Fox-Lyndon theorem, every string has a unique factorization
/// into non-increasing Lyndon words: w = l1 · l2 · ... · lk where
/// l1 >= l2 >= ... >= lk lexicographically.
pub fn lyndon_factorize(input: &[u8]) -> Vec<(usize, usize)> {
    let n = input.len();
    let mut factors = Vec::new();
    let mut i = 0;

    while i < n {
        let mut j = i;
        let mut k = i + 1;

        while k < n && input[j] <= input[k] {
            if input[j] < input[k] {
                j = i;
            } else {
                j += 1;
            }
            k += 1;
        }

        let period = k - j;
        while i + period <= k {
            factors.push((i, period));
            i += period;
        }
    }

    factors
}

/// Bijective BWT forward transform.
///
/// Instead of a single rotation sort on the whole input (which requires a
/// primary index to invert), bijective BWT:
/// 1. Factorizes input into Lyndon factors via Duval's algorithm
/// 2. For each factor, sorts all rotations and takes the last character
/// 3. Concatenates the per-factor BWTs
///
/// Returns (transformed_data, factor_lengths). The factor_lengths are needed
/// for the inverse transform.
pub fn encode_bijective(input: &[u8]) -> Option<(Vec<u8>, Vec<usize>)> {
    if input.is_empty() {
        return None;
    }

    let factors = lyndon_factorize(input);
    let mut output = Vec::with_capacity(input.len());
    let mut factor_lengths = Vec::with_capacity(factors.len());

    for &(start, len) in &factors {
        let factor = &input[start..start + len];
        factor_lengths.push(len);

        if len == 1 {
            // Single-byte factor: BWT is itself
            output.push(factor[0]);
            continue;
        }

        if len == 2 {
            // Two rotations: [a,b] and [b,a]. Sort and take last column.
            // For a Lyndon word, a < b always (otherwise it wouldn't be Lyndon).
            // Sorted: [a,b], [b,a] → last column: b, a
            output.push(factor[1]);
            output.push(factor[0]);
            continue;
        }

        if len == 3 {
            // Three rotations — sort by circular comparison, emit last column.
            let mut sa = [0usize, 1, 2];
            sa.sort_by(|&x, &y| {
                for i in 0..3 {
                    match factor[(x + i) % 3].cmp(&factor[(y + i) % 3]) {
                        std::cmp::Ordering::Equal => {}
                        ord => return ord,
                    }
                }
                std::cmp::Ordering::Equal
            });
            for &s in &sa {
                output.push(factor[(s + 2) % 3]);
            }
            continue;
        }

        // Build suffix array for this factor's rotations
        // For a Lyndon word, all rotations are distinct, so the SA is unique.
        let sa = build_suffix_array(factor);

        // Extract last column of sorted rotation matrix
        for &sa_val in &sa {
            if sa_val == 0 {
                output.push(factor[len - 1]);
            } else {
                output.push(factor[sa_val - 1]);
            }
        }
    }

    Some((output, factor_lengths))
}

/// Bijective BWT inverse transform.
///
/// Given the transformed data and the factor lengths from encode_bijective,
/// reconstructs the original input by inverting each factor's BWT independently.
///
/// Key insight: each factor is a Lyndon word, which by definition is the
/// lexicographically smallest rotation of itself. In the sorted rotation matrix,
/// the original string is always at row 0, so `primary_index = 0` for every factor.
/// This lets us reuse the standard O(n) LF-mapping decode.
pub fn decode_bijective(bwt: &[u8], factor_lengths: &[usize]) -> PzResult<Vec<u8>> {
    if bwt.is_empty() {
        return Ok(Vec::new());
    }

    // Verify total lengths match
    let total: usize = factor_lengths.iter().sum();
    if total != bwt.len() {
        return Err(PzError::InvalidInput);
    }

    let mut output = Vec::with_capacity(bwt.len());
    let mut offset = 0;

    for &flen in factor_lengths {
        let factor_bwt = &bwt[offset..offset + flen];
        offset += flen;

        if flen == 1 {
            output.push(factor_bwt[0]);
            continue;
        }

        // Lyndon word ⟹ original is lex-smallest rotation ⟹ primary_index = 0.
        // Use the standard LF-mapping decode, O(n).
        let decoded = decode(factor_bwt, 0)?;
        output.extend_from_slice(&decoded);
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
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
    fn test_round_trip_hello() {
        let input = b"hello";
        let result = encode(input).unwrap();
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
    fn test_round_trip_abcabc() {
        let input = b"abcabc";
        let result = encode(input).unwrap();
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
    fn test_round_trip_repeating_pattern() {
        let input = b"abababababab";
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
    fn test_round_trip_two_bytes() {
        let input = b"ab";
        let result = encode(input).unwrap();
        let decoded = decode(&result.data, result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_three_bytes() {
        let input = b"cab";
        let result = encode(input).unwrap();
        let decoded = decode(&result.data, result.primary_index).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_decode_invalid_primary_index() {
        let result = decode(&[1, 2, 3], 5);
        assert_eq!(result, Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_to_buf() {
        let input = b"banana";
        let result = encode(input).unwrap();
        let mut buf = vec![0u8; 100];
        let size = decode_to_buf(&result.data, result.primary_index, &mut buf).unwrap();
        assert_eq!(&buf[..size], input);
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
    fn test_round_trip_medium() {
        // Medium-length input to exercise the algorithm more thoroughly
        let mut input = Vec::new();
        for _ in 0..10 {
            input.extend(b"Hello, World! This tests the BWT. ");
        }
        let result = encode(&input).unwrap();
        let decoded = decode(&result.data, result.primary_index).unwrap();
        assert_eq!(decoded, input);
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
    fn test_bijective_bwt_round_trip_all_same() {
        let input = vec![b'x'; 20];
        let (bwt_data, factor_lens) = encode_bijective(&input).unwrap();
        let decoded = decode_bijective(&bwt_data, &factor_lens).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_bijective_bwt_round_trip_longer() {
        let input = b"the quick brown fox jumps over the lazy dog";
        let (bwt_data, factor_lens) = encode_bijective(input).unwrap();
        let decoded = decode_bijective(&bwt_data, &factor_lens).unwrap();
        assert_eq!(&decoded, &input[..]);
    }

    #[test]
    fn test_bijective_bwt_round_trip_binary() {
        let input: Vec<u8> = (0..=255).collect();
        let (bwt_data, factor_lens) = encode_bijective(&input).unwrap();
        let decoded = decode_bijective(&bwt_data, &factor_lens).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_bijective_bwt_round_trip_repeating() {
        let input = b"abababababab";
        let (bwt_data, factor_lens) = encode_bijective(input).unwrap();
        let decoded = decode_bijective(&bwt_data, &factor_lens).unwrap();
        assert_eq!(&decoded, &input[..]);
    }

    #[test]
    fn test_bijective_small_factors() {
        // Inputs that produce factors of various small lengths
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

            // Verify factor lengths match expectations
            let total: usize = factor_lens.iter().sum();
            assert_eq!(total, input.len());
        }
    }

    #[test]
    fn test_bijective_vs_standard_compression() {
        // Compare bijective vs standard BWT through MTF→RLE→FSE pipeline
        // and report compression ratios
        use crate::{fse, mtf, rle};

        println!("\n=== Bijective BWT vs Standard BWT Compression Comparison ===\n");
        println!(
            "{:<30} {:>8} {:>8} {:>8} {:>8}",
            "Input", "Std BWT", "Bij BWT", "Std Full", "Bij Full"
        );
        println!("{:-<30} {:->8} {:->8} {:->8} {:->8}", "", "", "", "", "");

        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("zeros_100", vec![0u8; 100]),
            ("all_same_1000", vec![b'a'; 1000]),
            (
                "repeating_text",
                b"Hello, World! "
                    .iter()
                    .cycle()
                    .take(1024)
                    .copied()
                    .collect(),
            ),
            (
                "abracadabra_x100",
                b"abracadabra ".iter().cycle().take(1200).copied().collect(),
            ),
            ("binary_256", (0..=255u8).collect()),
            ("sawtooth", (0..1024u16).map(|i| (i % 64) as u8).collect()),
        ];

        for (name, input) in &test_cases {
            // Standard BWT pipeline
            let std_result = encode(input).unwrap();
            let std_mtf = mtf::encode(&std_result.data);
            let std_rle = rle::encode(&std_mtf);
            let std_fse = fse::encode(&std_rle);

            // Bijective BWT pipeline
            let (bij_data, _bij_factors) = encode_bijective(input).unwrap();
            let bij_mtf = mtf::encode(&bij_data);
            let bij_rle = rle::encode(&bij_mtf);
            let bij_fse = fse::encode(&bij_rle);

            // Count runs in BWT output (clustering metric)
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

            let std_runs = count_runs(&std_result.data);
            let bij_runs = count_runs(&bij_data);

            println!(
                "{:<30} {:>5}r/{:>4}B {:>5}r/{:>4}B {:>8}B {:>8}B",
                format!("{name} ({}B)", input.len()),
                std_runs,
                std_result.data.len(),
                bij_runs,
                bij_data.len(),
                std_fse.len(),
                bij_fse.len(),
            );
        }

        // Canterbury corpus — with timing
        let cantrbry_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("samples/cantrbry");
        if cantrbry_dir.exists() {
            println!("\n--- Canterbury Corpus (compression ratio) ---");
            println!(
                "{:<25} {:>8} {:>8} {:>8} {:>8} {:>7}",
                "File", "Std Runs", "Bij Runs", "Std Size", "Bij Size", "Delta%"
            );
            println!(
                "{:-<25} {:->8} {:->8} {:->8} {:->8} {:->7}",
                "", "", "", "", "", ""
            );

            let files = [
                "alice29.txt",
                "asyoulik.txt",
                "cp.html",
                "fields.c",
                "grammar.lsp",
                "xargs.1",
            ];

            // First pass: compression ratios
            for filename in &files {
                let path = cantrbry_dir.join(filename);
                if let Ok(data) = std::fs::read(&path) {
                    let input = if data.len() > 65536 {
                        &data[..65536]
                    } else {
                        &data
                    };

                    let std_result = encode(input).unwrap();
                    let std_mtf = mtf::encode(&std_result.data);
                    let std_rle = rle::encode(&std_mtf);
                    let std_fse = fse::encode(&std_rle);

                    let (bij_data, _) = encode_bijective(input).unwrap();
                    let bij_mtf = mtf::encode(&bij_data);
                    let bij_rle = rle::encode(&bij_mtf);
                    let bij_fse = fse::encode(&bij_rle);

                    fn count_runs2(data: &[u8]) -> usize {
                        if data.is_empty() {
                            return 0;
                        }
                        data.windows(2).filter(|w| w[0] != w[1]).count() + 1
                    }

                    let std_runs = count_runs2(&std_result.data);
                    let bij_runs = count_runs2(&bij_data);
                    let delta_pct = (bij_fse.len() as f64 / std_fse.len() as f64 - 1.0) * 100.0;

                    println!(
                        "{:<25} {:>8} {:>8} {:>8} {:>8} {:>+6.1}%",
                        format!("{filename} ({}B)", input.len()),
                        std_runs,
                        bij_runs,
                        std_fse.len(),
                        bij_fse.len(),
                        delta_pct,
                    );
                }
            }

            // Second pass: timing (encode + decode, 3 iterations, report median-ish)
            println!("\n--- Canterbury Corpus (timing) ---");
            println!(
                "{:<25} {:>10} {:>10} {:>10} {:>10} {:>8} {:>10}",
                "File", "Std Enc", "Bij Enc", "Std Dec", "Bij Dec", "Factors", "Avg Flen"
            );
            println!(
                "{:-<25} {:->10} {:->10} {:->10} {:->10} {:->8} {:->10}",
                "", "", "", "", "", "", ""
            );

            for filename in &files {
                let path = cantrbry_dir.join(filename);
                if let Ok(data) = std::fs::read(&path) {
                    let input = if data.len() > 65536 {
                        &data[..65536]
                    } else {
                        &data
                    };

                    // Warm up + measure standard BWT encode (3 runs, take last)
                    let mut std_enc_us = 0u128;
                    let mut std_result = None;
                    for _ in 0..3 {
                        let t0 = std::time::Instant::now();
                        std_result = Some(encode(input).unwrap());
                        std_enc_us = t0.elapsed().as_micros();
                    }
                    let std_result = std_result.unwrap();

                    // Measure standard BWT decode
                    let mut std_dec_us = 0u128;
                    for _ in 0..3 {
                        let t0 = std::time::Instant::now();
                        let _ = decode(&std_result.data, std_result.primary_index).unwrap();
                        std_dec_us = t0.elapsed().as_micros();
                    }

                    // Measure bijective BWT encode
                    let mut bij_enc_us = 0u128;
                    let mut bij_result = None;
                    for _ in 0..3 {
                        let t0 = std::time::Instant::now();
                        bij_result = Some(encode_bijective(input).unwrap());
                        bij_enc_us = t0.elapsed().as_micros();
                    }
                    let (bij_data, bij_factors) = bij_result.unwrap();

                    // Measure bijective BWT decode
                    let mut bij_dec_us = 0u128;
                    for _ in 0..3 {
                        let t0 = std::time::Instant::now();
                        let _ = decode_bijective(&bij_data, &bij_factors).unwrap();
                        bij_dec_us = t0.elapsed().as_micros();
                    }

                    let num_factors = bij_factors.len();
                    let avg_flen = input.len() as f64 / num_factors as f64;

                    println!(
                        "{:<25} {:>8}us {:>8}us {:>8}us {:>8}us {:>8} {:>9.1}",
                        format!("{filename} ({}B)", input.len()),
                        std_enc_us,
                        bij_enc_us,
                        std_dec_us,
                        bij_dec_us,
                        num_factors,
                        avg_flen,
                    );
                }
            }
        } else {
            println!("\n(Canterbury corpus not extracted — skipping)");
        }
    }
}
