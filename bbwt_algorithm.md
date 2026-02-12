# Bijective BWT (BBWT) — Algorithm Summary for Implementation

## Overview

The BBWT is a bijective variant of the Burrows-Wheeler Transform. Unlike the standard BWT, it requires no sentinel character (`$`) and no primary index — the transform is a true bijection on strings of length n. It works by:

1. Factoring the input into Lyndon words (Chen–Fox–Lyndon theorem)
2. Computing the BWT of each factor independently (using circular rotations, not linear suffixes)
3. Interleaving the results into a single output string

---

## Algorithm 1: Lyndon Factorization (Duval's Algorithm)

**Purpose**: Decompose input string `T` into `T = w₁ w₂ ... wₖ` where each `wᵢ` is a Lyndon word and `w₁ ≥ w₂ ≥ ... ≥ wₖ` lexicographically.

**Properties of Lyndon words**:
- A string is Lyndon iff it is strictly smaller than all its proper suffixes
- Equivalently: strictly smaller than all its non-trivial rotations
- Equivalently: whenever split into two nonempty parts `uv`, we have `u < v`
- A Lyndon word is always primitive (not a repetition of a shorter string)

**Duval's algorithm** — O(n) time, O(1) extra space:

```
function lyndon_factorize(s):
    n = len(s)
    factors = []
    i = 0
    while i < n:
        j = i + 1  // candidate extension pointer
        k = i      // comparison pointer
        while j < n and s[k] <= s[j]:
            if s[k] < s[j]:
                k = i      // reset comparison to start (extending Lyndon word)
            else:
                k = k + 1  // equal: continue comparing (building repeated prefix)
            j = j + 1
        // Output factors of length (j - k)
        period = j - k
        while i <= k:
            factors.append((i, i + period))  // [start, end) of each factor
            i = i + period
    return factors
```

**Key insight**: The algorithm maintains a "pre-simple" string `s[i..j)` which is of the form `w^m · w_prefix` where `w` is Lyndon. When the extension fails (s[k] > s[j]), it emits complete copies of `w`.

**Expected factor counts from benchmarks**:
- English text (64KB): 2–3 factors (very large factors, ~20–30KB average)
- Source code (64KB): 8–13 factors (smaller, ~1–8KB average)
- The smallest character in the text (often `\n` = 0x0A) acts as a natural factor boundary since Lyndon words cannot contain a character smaller than their first character internally without breaking the Lyndon property

---

## Algorithm 2: BBWT Encoding (Per-Factor Approach — Current Prototype)

This is the simpler approach: build a suffix array per Lyndon factor.

```
function bbwt_encode(T):
    factors = lyndon_factorize(T)
    output = []
    for (start, end) in factors:
        factor = T[start..end)
        n = len(factor)
        // Build circular suffix array for this factor
        // Option A: Use SA-IS on factor·factor, then filter to indices < n
        // Option B: Use SA-IS on factor with circular comparison
        csa = circular_suffix_array(factor)
        // Last column of sorted rotation matrix
        for i in 0..n:
            output.append(factor[(csa[i] + n - 1) % n])
    return concat(output), factor_boundaries
```

**Circular suffix array via SA-IS on doubled string**:
```
function circular_suffix_array(w):
    n = len(w)
    doubled = w + w
    sa = sais(doubled)  // standard SA-IS on the doubled string
    // Filter: keep only entries where sa[i] < n
    csa = [x for x in sa if x < n]
    return csa
```

**Cost**: O(Σ|wᵢ|) = O(n) total, but with constant-factor overhead proportional to number of factors (each SA-IS call has setup cost). This explains the 0–21% slowdown observed.

---

## Algorithm 3: BBWT Encoding (Unified Linear-Time — Bannai et al. CPM 2021)

This is the optimal approach that eliminates per-factor overhead.

**Key ideas**:
1. Compute Lyndon factorization in O(n) via Duval
2. Build a compact representation `R` where duplicate Lyndon factors are stored only once
3. Run a single modified SA-IS that computes circular suffix arrays for all factors simultaneously

**Modified SA-IS for circular suffix arrays**:

The standard SA-IS algorithm sorts linear suffixes. For circular suffixes of a Lyndon word `w`, the crucial observation is:

- Since `w` is Lyndon, `w` itself is the lexicographically smallest rotation
- The circular suffixes of `w` can be mapped to suffixes of `ww` (the doubled string)
- But instead of literally doubling, the algorithm exploits the Lyndon structure:
  - For a Lyndon word `w`, suffixes of `w` and circular rotations of `w` have a known relationship
  - The induced sorting framework of SA-IS can be adapted to handle the wrap-around

**Sketch of the unified algorithm**:
```
function bbwt_encode_linear(T):
    factors = lyndon_factorize(T)  // O(n), Duval

    // Step 1: Deduplicate factors
    // Group identical factors. Since factors are non-increasing,
    // identical factors are always adjacent.
    unique_factors = deduplicate(factors)  // list of (factor_string, count)

    // Step 2: Build representation R
    // R = concatenation of unique factors with separator logic
    // The separators are handled implicitly by SA-IS type classification

    // Step 3: Modified SA-IS on R
    // This computes the circular suffix array for all unique factors at once.
    // The modification: at factor boundaries, instead of comparing with a sentinel,
    // the comparison wraps around to the start of the same factor.
    csa = modified_sais_circular(R, factor_boundaries)

    // Step 4: Read off BBWT
    // For each position in the circular suffix array, output the preceding character
    // (with wraparound within each factor)
    output = []
    for i in 0..len(csa):
        factor_id = which_factor(csa[i])
        pos_in_factor = position_within_factor(csa[i])
        factor_len = len(unique_factors[factor_id])
        output.append(R[(pos_in_factor + factor_len - 1) % factor_len + factor_start])
    // Expand duplicates: repeat output segments for factors that appeared multiple times
    return expand_duplicates(output, unique_factors)
```

**Why this is linear**: SA-IS is linear, Lyndon factorization is linear, and deduplication is a single pass since identical factors are adjacent in the non-increasing sequence. The key theoretical contribution is proving that the modified SA-IS correctly handles circular suffixes across multiple factors in one pass.

**Reference implementation**: Available at the repository linked in Bannai et al. (CPM 2021), written in C++.

---

## Algorithm 4: BBWT Decoding (Inversion)

**Key insight already discovered in the prototype**: For each Lyndon factor, `primary_index = 0` because the Lyndon word is by definition the lexicographically smallest rotation of itself.

**Per-factor inversion via LF-mapping**:
```
function bbwt_decode(bbwt_string, factor_lengths):
    output = []
    offset = 0
    for flen in factor_lengths:
        L = bbwt_string[offset .. offset + flen)  // last column for this factor

        // Build F (first column) by sorting L
        F = sorted(L)

        // Build LF-mapping
        // LF[i] = position in F that row i maps to
        // Standard construction: for each character c at position i in L,
        // LF[i] = C[c] + rank(L, c, i)
        // where C[c] = number of characters in L smaller than c
        // and rank(L, c, i) = number of occurrences of c in L[0..i)
        C = compute_C_array(L)
        lf = compute_LF_mapping(L, C)

        // Reconstruct: start from row 0 (since primary_index = 0 for Lyndon words)
        row = 0
        factor = []
        for _ in 0..flen:
            factor.append(L[row])  // or F[row], depending on convention
            row = lf[row]

        // The factor is read in reverse from L, or forward from F
        // Convention: starting at row 0 and following LF gives the original
        // string read backwards. So reverse it, OR:
        // Use FL-mapping (inverse of LF) to read forward.
        output.append(reverse(factor))  // check direction based on your convention

        offset += flen

    return concat(output)
```

**Important detail on reading direction**: The LF-mapping from row 0 gives characters in a specific order. The standard BWT inversion reads the string backward (last character first). Verify which direction your implementation uses:
- If you read `F[row]` at each step: you get the string forward
- If you read `L[row]` at each step: you get the string backward (then reverse)

**Reassembly**: After decoding all factors, concatenate them in the order they appeared. The Lyndon factorization is unique so no additional metadata is needed beyond factor lengths.

---

## Algorithm 5: Storing/Transmitting Factor Boundaries

The BBWT itself is bijective (length n → length n), but to invert it you need to know where the factor boundaries are. Options:

1. **Store factor lengths explicitly**: For k factors, store k integers. For English text k ≈ 2–3, so overhead is negligible (a few bytes for 64KB input).

2. **Recompute from BBWT**: This is possible but nontrivial. The BBWT of a Lyndon word has a specific structure — the first character of each factor's BWT region can be identified. Gil and Scott's original paper discusses this.

3. **For compression pipelines**: The factor count is typically so small relative to input size that storing it explicitly is the pragmatic choice.

---

## Algorithm 6: BWT ↔ BBWT Conversion (Köppl et al. 2020)

If you already have the standard BWT and want the BBWT (or vice versa):

- **In-place, O(n²) time**: Possible using the relationship between the Lyndon factorization and the BWT's structure.
- **Run-length compressed, O(n lg r / lg lg r) time, O(r lg n) bits**: More efficient when both transforms have few runs. Here r is the sum of runs in BWT and BBWT.

This is mainly of theoretical interest but could be useful if your pipeline already has one transform computed.

---

## Key Theoretical Results to Be Aware Of

### Run counts (compression quality)
- `r_B` = number of maximal character runs in BBWT
- `r` = number of maximal character runs in standard BWT
- `r_B = O(z · log² n)` where z = number of LZ77 factors
- There exist strings where `r_B = Ω(log n)` but `r = 2` (BBWT can be worse)
- The minimum `r_B` over all cyclic rotations of the input is always ≤ r (BBWT can match BWT with the right rotation)
- Reversing the string can cause a logarithmic increase in `r_B`

### Compression benchmarks (Gil & Scott, Calgary Corpus)
- BBWT output is slightly more compressible than BWT on most files
- The advantage comes from not needing to store the primary index
- On the transformed data alone (ignoring the index), standard BWT typically has slightly better clustering

### Pattern matching / indexing
- Backward search works on BBWT with O(|P| lg |P|) steps per pattern P
- This is slightly worse than BWT's O(|P|) backward search
- A self-index can be built on top of BBWT (Bannai et al., Köppl et al.)

---

## Optimization Opportunities

1. **Unified SA-IS**: Biggest win. Eliminates per-factor SA-IS overhead. Most impactful on source code / markup inputs with many factors.

2. **SIMD-accelerated LF-mapping**: The decode inner loop is simple enough for vectorization. Each factor's decode is independent → parallelizable.

3. **Factor-parallel decode**: Since factors are independent, decode all factors simultaneously on separate threads. Particularly effective when factors are roughly equal size.

4. **Wavelet tree for rank queries**: If building an index (not just compress/decompress), replace the simple rank arrays with a wavelet tree for O(lg σ) rank queries.

5. **Run-length encoding awareness**: If the BBWT output has many runs (which it should for compressible data), an RLE-aware representation can speed up both storage and subsequent compression stages.

---

## References

- Gil, Scott (2012). "A Bijective String Sorting Transform." arXiv:1201.3077.
- Bannai, Kärkkäinen, Köppl, Piątkowski (2021). "Constructing the Bijective and the Extended BWT in Linear Time." CPM 2021, LIPIcs vol. 191.
- Köppl, Hashimoto, Hendrian, Shinohara (2020). "In-Place Bijective Burrows-Wheeler Transforms." arXiv:2004.12590.
- Badkobeh, Bannai, Köppl (2024). "Bijective BWT Based Compression Schemes." SPIRE 2024.
- Biagi, Cenzato, Lipták, Romana (2024). "On the number of equal-letter runs of the BBWT." Theoretical Computer Science.
- Olbrich et al. (2025). "Fast and memory-efficient BWT construction of repetitive texts using Lyndon grammars." arXiv:2504.19123.
- Duval (1983). "Factorizing words over an ordered alphabet." J. Algorithms 4(4):363–381.
