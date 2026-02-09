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

    // Build suffix array using prefix-doubling O(n log^2 n)
    let sa = build_suffix_array(input);

    // The BWT of input is constructed from the suffix array:
    // For the BWT, we need rotations, not suffixes. We treat the input as
    // circular by conceptually doubling it.
    //
    // BWT[i] = input[(sa[i] + n - 1) % n]
    // Primary index = position where sa[i] == 0

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

/// Build a suffix array for the input using prefix-doubling.
///
/// Returns an array where sa[i] is the starting position of the i-th
/// smallest suffix of `input` (treating input as circular/doubled).
///
/// Uses the prefix-doubling algorithm: O(n log^2 n) time, O(n) space.
/// This is suitable for a reference implementation and is the approach
/// that maps well to GPU parallelization (Phase 3).
fn build_suffix_array(input: &[u8]) -> Vec<usize> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // For BWT we need to sort rotations, not suffixes.
    // We do this by doubling the input and building a suffix array
    // on the doubled text, then filtering to keep only the first n suffixes.
    //
    // Alternative: treat suffixes as circular using modular indexing.
    // We use the circular approach for memory efficiency.

    // Initial ranking based on single characters
    let mut rank = vec![0i64; n];
    let mut sa: Vec<usize> = (0..n).collect();

    // Assign initial ranks based on byte value
    for i in 0..n {
        rank[i] = input[i] as i64;
    }

    let mut tmp_rank = vec![0i64; n];
    let mut k = 1;

    while k < n {
        // Sort by (rank[i], rank[(i + k) % n])
        let r = &rank;
        sa.sort_by(|&a, &b| {
            let ra = (r[a], r[(a + k) % n]);
            let rb = (r[b], r[(b + k) % n]);
            ra.cmp(&rb)
        });

        // Reassign ranks
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

        // If all ranks are unique, we're done
        if rank[sa[n - 1]] as usize == n - 1 {
            break;
        }

        k *= 2;
    }

    sa
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
        let result = encode(&[b'a']).unwrap();
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
        let size =
            decode_to_buf(&result.data, result.primary_index, &mut buf).unwrap();
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
}
