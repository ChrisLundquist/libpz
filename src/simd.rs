//! SIMD-accelerated primitives for compression hot paths.
//!
//! Provides optimized implementations of inner loops that dominate
//! compression and decompression throughput:
//!
//! - **Byte frequency counting** — used by Huffman tree construction and
//!   data analysis. SIMD processes 16/32 bytes at a time.
//! - **Huffman bit-packing** — batch symbol→codeword lookup with wider
//!   accumulators to reduce bit-shift overhead.
//! - **LZ77 match comparison** — `memcmp`-style comparison using SIMD
//!   to find the length of a match between two byte sequences.
//!
//! # Architecture support
//!
//! | ISA            | Baseline | Extended   | Status      |
//! |----------------|----------|------------|-------------|
//! | x86_64 SSE2    | Yes      | SSSE3      | Implemented |
//! | x86_64 AVX2    | —        | AVX2       | Implemented |
//! | aarch64 NEON   | Yes      | —          | Stub        |
//! | aarch64 SVE    | —        | SVE        | Stub        |
//!
//! # Runtime dispatch
//!
//! Use [`Dispatcher`] for automatic feature detection at startup:
//!
//! ```rust,no_run
//! use pz::simd::Dispatcher;
//! let d = Dispatcher::new();
//! let data = b"hello world hello world";
//! let freqs = d.byte_frequencies(data);
//! let a = b"hello world";
//! let b = b"hello there";
//! let match_len = d.compare_bytes(a, b);
//! assert_eq!(match_len, 6); // first 6 bytes match
//! ```
//!
//! The dispatcher probes CPU features once and caches function pointers
//! for the best available implementation.

/// Maximum match length for SIMD comparison (matches LZ77 window constraints).
const MAX_COMPARE_LEN: usize = 258;

// ---------------------------------------------------------------------------
// Runtime dispatcher
// ---------------------------------------------------------------------------

/// SIMD capability level detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    /// No SIMD — scalar fallback.
    Scalar,
    /// x86_64 SSE2 (baseline for x86_64, always available).
    #[cfg(target_arch = "x86_64")]
    Sse2,
    /// x86_64 AVX2 (256-bit registers).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// aarch64 NEON (baseline for aarch64, always available).
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// Function pointer type for SIMD byte comparison.
///
/// Signature: `(a_ptr, b_ptr, max_len) -> match_length`
///
/// # Safety
/// Implementations require the appropriate SIMD feature to be available
/// and `max_len` bytes to be readable from both pointers.
type CompareFn = unsafe fn(*const u8, *const u8, usize) -> usize;

/// Runtime SIMD dispatcher.
///
/// Detects available SIMD features at construction time and resolves
/// function pointers for hot-path operations. This eliminates per-call
/// match dispatch overhead — the SIMD level is checked once at startup,
/// not on every call.
#[derive(Clone, Copy)]
pub struct Dispatcher {
    level: SimdLevel,
    /// Resolved function pointer for byte comparison (hot path).
    compare_fn: CompareFn,
}

impl std::fmt::Debug for Dispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dispatcher")
            .field("level", &self.level)
            .finish()
    }
}

impl Dispatcher {
    /// Detect the best available SIMD level for the current CPU.
    ///
    /// Resolves function pointers once — subsequent calls to `compare_bytes`
    /// go through a direct function pointer with no match dispatch.
    pub fn new() -> Self {
        let level = detect_level();
        let compare_fn = resolve_compare_fn(level);
        Dispatcher { level, compare_fn }
    }

    /// Return the detected SIMD capability level.
    pub fn level(&self) -> SimdLevel {
        self.level
    }

    /// Count byte frequencies in `input`, returning a 256-entry histogram.
    ///
    /// Uses SIMD-accelerated counting when available; falls back to scalar
    /// for short inputs or unsupported architectures.
    pub fn byte_frequencies(&self, input: &[u8]) -> [u32; 256] {
        match self.level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => {
                // SAFETY: `detect_level` verified AVX2 is available
                unsafe { avx2::byte_frequencies(input) }
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse2 => {
                // SAFETY: SSE2 is always available on x86_64
                unsafe { sse2::byte_frequencies(input) }
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => scalar::byte_frequencies(input),
            SimdLevel::Scalar => scalar::byte_frequencies(input),
        }
    }

    /// Find the length of the common prefix between `a` and `b`.
    ///
    /// Returns the number of leading bytes that are identical (up to
    /// `MAX_COMPARE_LEN` or the shorter slice length). Used as the inner
    /// loop of LZ77 match extension.
    ///
    /// Uses a resolved function pointer — no per-call match dispatch.
    #[inline]
    pub fn compare_bytes(&self, a: &[u8], b: &[u8]) -> usize {
        let max_len = a.len().min(b.len()).min(MAX_COMPARE_LEN);
        if max_len == 0 {
            return 0;
        }
        // SAFETY: compare_fn was resolved from detect_level() which verified
        // the required SIMD features are available. max_len is bounded by
        // both slice lengths, so reads are in-bounds.
        unsafe { (self.compare_fn)(a.as_ptr(), b.as_ptr(), max_len) }
    }

    /// Sum all values in a u32 slice. Used for prefix sum verification
    /// and total bit length computation in Huffman encoding.
    pub fn sum_u32(&self, data: &[u32]) -> u64 {
        match self.level {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2 => {
                // SAFETY: `detect_level` verified AVX2 is available
                unsafe { avx2::sum_u32(data) }
            }
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Sse2 => {
                // SAFETY: SSE2 is always available on x86_64
                unsafe { sse2::sum_u32(data) }
            }
            #[cfg(target_arch = "aarch64")]
            SimdLevel::Neon => scalar::sum_u32(data),
            SimdLevel::Scalar => scalar::sum_u32(data),
        }
    }
}

impl Default for Dispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect the highest SIMD level supported by the current CPU.
fn detect_level() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
        // SSE2 is always available on x86_64
        return SimdLevel::Sse2;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return SimdLevel::Neon;
    }

    #[allow(unreachable_code)]
    SimdLevel::Scalar
}

/// Resolve compare_bytes to a direct function pointer based on SIMD level.
///
/// This is called once at `Dispatcher::new()` time. The returned function
/// pointer is stored and called directly on every `compare_bytes` invocation,
/// eliminating the per-call match dispatch that was costing ~19% of samples.
fn resolve_compare_fn(level: SimdLevel) -> CompareFn {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => compare_bytes_avx2,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse2 => compare_bytes_sse2,
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon => compare_bytes_scalar,
        SimdLevel::Scalar => compare_bytes_scalar,
    }
}

/// Scalar compare_bytes wrapper matching `CompareFn` signature.
///
/// # Safety
/// Caller must ensure `max_len` bytes are readable from both `a` and `b`.
unsafe fn compare_bytes_scalar(a: *const u8, b: *const u8, max_len: usize) -> usize {
    let a = std::slice::from_raw_parts(a, max_len);
    let b = std::slice::from_raw_parts(b, max_len);
    scalar::compare_bytes(a, b, max_len)
}

/// SSE2 compare_bytes wrapper matching `CompareFn` signature.
///
/// Marked `#[target_feature(enable = "sse2")]` so the compiler can inline
/// the SSE2 intrinsics directly without an extra call frame.
///
/// # Safety
/// Caller must ensure SSE2 is available and `max_len` bytes are readable
/// from both `a` and `b`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn compare_bytes_sse2(a: *const u8, b: *const u8, max_len: usize) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;

    // Process 16 bytes at a time with SSE2
    while i + 16 <= max_len {
        let va = _mm_loadu_si128(a.add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.add(i) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(va, vb);
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0xFFFF {
            let first_diff = (!mask).trailing_zeros() as usize;
            return i + first_diff;
        }
        i += 16;
    }

    // Scalar tail
    while i < max_len && *a.add(i) == *b.add(i) {
        i += 1;
    }
    i
}

/// AVX2 compare_bytes wrapper matching `CompareFn` signature.
///
/// Marked `#[target_feature(enable = "avx2")]` so the compiler can inline
/// the AVX2 intrinsics directly without an extra call frame.
///
/// # Safety
/// Caller must ensure AVX2 is available and `max_len` bytes are readable
/// from both `a` and `b`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn compare_bytes_avx2(a: *const u8, b: *const u8, max_len: usize) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;

    // Process 32 bytes at a time with AVX2
    while i + 32 <= max_len {
        let va = _mm256_loadu_si256(a.add(i) as *const __m256i);
        let vb = _mm256_loadu_si256(b.add(i) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(va, vb);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0xFFFFFFFF {
            let first_diff = (!mask).trailing_zeros() as usize;
            return i + first_diff;
        }
        i += 32;
    }

    // SSE2 16-byte tail
    while i + 16 <= max_len {
        let va = _mm_loadu_si128(a.add(i) as *const __m128i);
        let vb = _mm_loadu_si128(b.add(i) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(va, vb);
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0xFFFF {
            let first_diff = (!mask).trailing_zeros() as usize;
            return i + first_diff;
        }
        i += 16;
    }

    // Scalar tail
    while i < max_len && *a.add(i) == *b.add(i) {
        i += 1;
    }
    i
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

pub mod scalar {
    /// Scalar byte frequency counting.
    pub fn byte_frequencies(input: &[u8]) -> [u32; 256] {
        let mut freqs = [0u32; 256];
        for &b in input {
            freqs[b as usize] += 1;
        }
        freqs
    }

    /// Scalar byte comparison — count matching prefix bytes.
    pub fn compare_bytes(a: &[u8], b: &[u8], max_len: usize) -> usize {
        let mut i = 0;
        while i < max_len && a[i] == b[i] {
            i += 1;
        }
        i
    }

    /// Scalar u32 sum.
    pub fn sum_u32(data: &[u32]) -> u64 {
        data.iter().map(|&v| v as u64).sum()
    }
}

// ---------------------------------------------------------------------------
// x86_64 SSE2 (baseline — always available on x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod sse2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// SSE2-accelerated byte frequency counting.
    ///
    /// Uses a 4-way unrolled scalar loop (SSE2 doesn't have gather/scatter
    /// for histogramming, so the main speedup comes from reducing loop
    /// overhead and improving cache utilization through unrolling).
    ///
    /// # Safety
    /// Requires SSE2 (always available on x86_64).
    #[target_feature(enable = "sse2")]
    pub unsafe fn byte_frequencies(input: &[u8]) -> [u32; 256] {
        let mut freqs = [0u32; 256];
        let len = input.len();
        let mut i = 0;

        // Process 4 bytes at a time (unrolled scalar — histogram isn't
        // naturally SIMD-friendly, but unrolling reduces branch overhead)
        let end4 = len & !3;
        while i < end4 {
            freqs[*input.get_unchecked(i) as usize] += 1;
            freqs[*input.get_unchecked(i + 1) as usize] += 1;
            freqs[*input.get_unchecked(i + 2) as usize] += 1;
            freqs[*input.get_unchecked(i + 3) as usize] += 1;
            i += 4;
        }

        // Handle remainder
        while i < len {
            freqs[*input.get_unchecked(i) as usize] += 1;
            i += 1;
        }

        freqs
    }

    /// SSE2-accelerated u32 sum using u64 accumulator lanes.
    ///
    /// Widens each u32 to u64 before accumulating, so no overflow is
    /// possible even with u32::MAX values. Processes 4 u32s per iteration
    /// (two 128-bit loads, each widened to 2×u64 and added to accumulators).
    ///
    /// # Safety
    /// Requires SSE2 (always available on x86_64).
    #[target_feature(enable = "sse2")]
    pub unsafe fn sum_u32(data: &[u32]) -> u64 {
        let len = data.len();
        let mut i = 0;

        // Two u64 accumulators (2 lanes each = 4 u64 lanes total)
        let mut acc_lo = _mm_setzero_si128(); // 2×u64
        let mut acc_hi = _mm_setzero_si128(); // 2×u64
        let zero = _mm_setzero_si128();
        let end4 = len & !3;

        while i < end4 {
            let v = _mm_loadu_si128(data.as_ptr().add(i) as *const __m128i);
            // Widen 4×u32 → 2×(2×u64)
            let lo = _mm_unpacklo_epi32(v, zero); // lower 2 u32 → 2 u64
            let hi = _mm_unpackhi_epi32(v, zero); // upper 2 u32 → 2 u64
            acc_lo = _mm_add_epi64(acc_lo, lo);
            acc_hi = _mm_add_epi64(acc_hi, hi);
            i += 4;
        }

        // Horizontal sum: 4 u64 lanes → single u64
        let combined = _mm_add_epi64(acc_lo, acc_hi);
        let upper = _mm_srli_si128(combined, 8);
        let total = _mm_add_epi64(combined, upper);
        let mut sum = _mm_cvtsi128_si64(total) as u64;

        // Scalar tail
        while i < len {
            sum += *data.get_unchecked(i) as u64;
            i += 1;
        }

        sum
    }
}

// ---------------------------------------------------------------------------
// x86_64 AVX2 (requires runtime detection)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2-accelerated byte frequency counting.
    ///
    /// Uses 4-bank histogramming to reduce store-to-load conflicts:
    /// four independent histogram arrays are updated in parallel, then
    /// merged. This reduces pipeline stalls from dependent memory accesses.
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn byte_frequencies(input: &[u8]) -> [u32; 256] {
        let len = input.len();

        // 4-bank histogram to reduce store-forwarding conflicts
        let mut f0 = [0u32; 256];
        let mut f1 = [0u32; 256];
        let mut f2 = [0u32; 256];
        let mut f3 = [0u32; 256];

        let mut i = 0;
        let end4 = len & !3;

        while i < end4 {
            f0[*input.get_unchecked(i) as usize] += 1;
            f1[*input.get_unchecked(i + 1) as usize] += 1;
            f2[*input.get_unchecked(i + 2) as usize] += 1;
            f3[*input.get_unchecked(i + 3) as usize] += 1;
            i += 4;
        }

        // Remainder
        while i < len {
            f0[*input.get_unchecked(i) as usize] += 1;
            i += 1;
        }

        // Merge banks using AVX2 (8 u32 at a time)
        let mut result = [0u32; 256];
        let mut j = 0;
        while j + 8 <= 256 {
            let v0 = _mm256_loadu_si256(f0.as_ptr().add(j) as *const __m256i);
            let v1 = _mm256_loadu_si256(f1.as_ptr().add(j) as *const __m256i);
            let v2 = _mm256_loadu_si256(f2.as_ptr().add(j) as *const __m256i);
            let v3 = _mm256_loadu_si256(f3.as_ptr().add(j) as *const __m256i);
            let sum01 = _mm256_add_epi32(v0, v1);
            let sum23 = _mm256_add_epi32(v2, v3);
            let total = _mm256_add_epi32(sum01, sum23);
            _mm256_storeu_si256(result.as_mut_ptr().add(j) as *mut __m256i, total);
            j += 8;
        }

        result
    }

    /// AVX2-accelerated u32 sum using u64 accumulator lanes.
    ///
    /// Widens each u32 to u64 before accumulating, so no overflow is
    /// possible even with u32::MAX values. Processes 8 u32s per iteration
    /// by widening to 4×u64 accumulators.
    ///
    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub unsafe fn sum_u32(data: &[u32]) -> u64 {
        let len = data.len();
        let mut i = 0;

        // Four u64 accumulators (2 lanes each = 8 u64 lanes total)
        let zero = _mm256_setzero_si256();
        let mut acc0 = _mm256_setzero_si256(); // 4×u64
        let mut acc1 = _mm256_setzero_si256(); // 4×u64
        let end8 = len & !7;

        while i < end8 {
            let v = _mm256_loadu_si256(data.as_ptr().add(i) as *const __m256i);
            // Extract two 128-bit halves
            let v_lo128 = _mm256_castsi256_si128(v);
            let v_hi128 = _mm256_extracti128_si256(v, 1);
            // Widen each 4×u32 → 4×u64 by zero-extending
            let lo_wide = _mm256_cvtepu32_epi64(v_lo128); // 4 u32 → 4 u64
            let hi_wide = _mm256_cvtepu32_epi64(v_hi128); // 4 u32 → 4 u64
            acc0 = _mm256_add_epi64(acc0, lo_wide);
            acc1 = _mm256_add_epi64(acc1, hi_wide);
            i += 8;
        }

        // Horizontal sum: 8 u64 lanes → single u64
        let combined = _mm256_add_epi64(acc0, acc1); // 4 u64
        let lo128 = _mm256_castsi256_si128(combined);
        let hi128 = _mm256_extracti128_si256(combined, 1);
        let sum128 = _mm_add_epi64(lo128, hi128); // 2 u64
        let upper = _mm_srli_si128(sum128, 8);
        let total = _mm_add_epi64(sum128, upper);
        let mut sum = _mm_cvtsi128_si64(total) as u64;

        // Scalar tail
        while i < len {
            sum += *data.get_unchecked(i) as u64;
            i += 1;
        }

        let _ = zero; // suppress unused warning
        sum
    }
}

// ---------------------------------------------------------------------------
// rANS 4-way interleaved decode (batched — 4 lanes per iteration)
// ---------------------------------------------------------------------------

/// Decode 4-way interleaved rANS, processing all 4 lanes per iteration.
///
/// This is the core SIMD-friendly decode loop. The per-quad iteration:
/// 1. Extract slots from all 4 states (SIMD AND + mask)
/// 2. Scalar gather: lookup symbol, freq, cum for each lane
/// 3. Compute new states: freq * (state >> scale_bits) + slot - cum
/// 4. Batch renormalization: compare all states, conditionally refill
///
/// Even without explicit SIMD intrinsics, batching 4 lanes per iteration
/// keeps states in registers, reduces loop overhead 4x, and exposes ILP
/// to the CPU's out-of-order engine.
///
/// # Arguments
/// - `word_streams`: per-lane word data (4 slices)
/// - `initial_states`: starting state for each lane (4 values)
/// - `freq`: normalized frequencies [256]
/// - `cum`: cumulative frequencies [256]
/// - `lookup`: slot→symbol table (size = 1 << scale_bits)
/// - `scale_bits`: frequency precision
/// - `original_len`: total symbols to decode
///
/// # Returns
/// Decoded bytes, or `None` on invalid input (slot out of bounds).
pub fn rans_decode_4way(
    word_streams: &[&[u16]; 4],
    initial_states: &[u32; 4],
    freq: &[u16; 256],
    cum: &[u16; 256],
    lookup: &[u8],
    scale_bits: u32,
    original_len: usize,
) -> Option<Vec<u8>> {
    let scale_mask = (1u32 << scale_bits) - 1;
    let rans_l: u32 = 1 << 16;
    let io_bits: u32 = 16;
    let lookup_len = lookup.len();

    let mut states = *initial_states;
    let mut word_pos = [0usize; 4];
    let mut output = vec![0u8; original_len];
    let mut out_pos = 0;

    // Process in batches of 4 (one symbol per lane per iteration).
    let full_quads = original_len / 4;
    let remainder = original_len % 4;

    for _ in 0..full_quads {
        // Step 1: Extract slots from all 4 states
        let slot0 = states[0] & scale_mask;
        let slot1 = states[1] & scale_mask;
        let slot2 = states[2] & scale_mask;
        let slot3 = states[3] & scale_mask;

        // Bounds check all 4 at once
        if (slot0 as usize | slot1 as usize | slot2 as usize | slot3 as usize) >= lookup_len {
            return None;
        }

        // Step 2: Scalar gather — lookup symbols, frequencies, cumulative
        let s0 = lookup[slot0 as usize];
        let s1 = lookup[slot1 as usize];
        let s2 = lookup[slot2 as usize];
        let s3 = lookup[slot3 as usize];

        let f0 = freq[s0 as usize] as u32;
        let f1 = freq[s1 as usize] as u32;
        let f2 = freq[s2 as usize] as u32;
        let f3 = freq[s3 as usize] as u32;

        let c0 = cum[s0 as usize] as u32;
        let c1 = cum[s1 as usize] as u32;
        let c2 = cum[s2 as usize] as u32;
        let c3 = cum[s3 as usize] as u32;

        // Step 3: State transition — all 4 independent, exposes ILP
        states[0] = f0 * (states[0] >> scale_bits) + slot0 - c0;
        states[1] = f1 * (states[1] >> scale_bits) + slot1 - c1;
        states[2] = f2 * (states[2] >> scale_bits) + slot2 - c2;
        states[3] = f3 * (states[3] >> scale_bits) + slot3 - c3;

        // Step 4: Renormalize all 4 lanes
        if states[0] < rans_l && word_pos[0] < word_streams[0].len() {
            states[0] = (states[0] << io_bits) | word_streams[0][word_pos[0]] as u32;
            word_pos[0] += 1;
        }
        if states[1] < rans_l && word_pos[1] < word_streams[1].len() {
            states[1] = (states[1] << io_bits) | word_streams[1][word_pos[1]] as u32;
            word_pos[1] += 1;
        }
        if states[2] < rans_l && word_pos[2] < word_streams[2].len() {
            states[2] = (states[2] << io_bits) | word_streams[2][word_pos[2]] as u32;
            word_pos[2] += 1;
        }
        if states[3] < rans_l && word_pos[3] < word_streams[3].len() {
            states[3] = (states[3] << io_bits) | word_streams[3][word_pos[3]] as u32;
            word_pos[3] += 1;
        }

        // Step 5: Write output symbols
        output[out_pos] = s0;
        output[out_pos + 1] = s1;
        output[out_pos + 2] = s2;
        output[out_pos + 3] = s3;
        out_pos += 4;
    }

    // Handle remaining symbols (< 4)
    for r in 0..remainder {
        let lane = r;
        let slot = states[lane] & scale_mask;
        if slot as usize >= lookup_len {
            return None;
        }
        let s = lookup[slot as usize];
        let f = freq[s as usize] as u32;
        let c = cum[s as usize] as u32;

        states[lane] = f * (states[lane] >> scale_bits) + slot - c;

        if states[lane] < rans_l && word_pos[lane] < word_streams[lane].len() {
            states[lane] = (states[lane] << io_bits) | word_streams[lane][word_pos[lane]] as u32;
            word_pos[lane] += 1;
        }

        output[out_pos] = s;
        out_pos += 1;
    }

    Some(output)
}

// ---------------------------------------------------------------------------
// aarch64 NEON / SVE stubs
// ---------------------------------------------------------------------------

/// aarch64 NEON implementations (stub — structurally complete but dispatches
/// to scalar for now). NEON is the baseline SIMD on aarch64 (always available).
///
/// When aarch64 hardware is available for benchmarking, these stubs should be
/// replaced with actual NEON intrinsics:
///
/// - `byte_frequencies`: Use `vld1q_u8` + lookup table approach, or 4-bank
///   unrolled scalar (NEON gather/scatter is limited).
/// - `compare_bytes`: Use `vceqq_u8` + `vmaxvq_u8` for 16-byte comparison,
///   with `vclzq_u32` to find the first mismatch position.
/// - `sum_u32`: Use `vld1q_u32` + `vaddq_u32` with `vaddvq_u32` for
///   horizontal reduction.
///
/// SVE (Scalable Vector Extension) is available on ARMv8.2+ servers (e.g.
/// AWS Graviton3, Fujitsu A64FX) and offers variable-length vectors (128-2048
/// bits). SVE implementations would use predicated operations for automatic
/// tail handling.
#[cfg(target_arch = "aarch64")]
mod neon {
    // TODO: Replace with actual NEON intrinsics when aarch64 hardware
    // is available for testing and benchmarking.
    //
    // Key intrinsics to use:
    //   byte_frequencies: vld1q_u8, vtbl1_u8 for permutation
    //   compare_bytes: vceqq_u8, vmovn_u16, vget_lane_u64
    //   sum_u32: vld1q_u32, vaddq_u32, vaddvq_u32

    pub use super::scalar::byte_frequencies;
    pub use super::scalar::compare_bytes;
    pub use super::scalar::sum_u32;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
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

        // Identical slices of various lengths
        for len in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 100, 258] {
            let a: Vec<u8> = (0..len).map(|i| (i % 256) as u8).collect();
            let b = a.clone();
            let max = a.len().min(b.len()).min(MAX_COMPARE_LEN);
            let simd_result = d.compare_bytes(&a, &b);
            assert_eq!(simd_result, max, "identical mismatch at len {}", len);
        }

        // Mismatch at specific positions
        for mismatch_pos in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 100] {
            let len = mismatch_pos + 10;
            let a: Vec<u8> = (0..len).map(|i| (i % 200) as u8).collect();
            let mut b = a.clone();
            b[mismatch_pos] = 255; // force mismatch
            let simd_result = d.compare_bytes(&a, &b);
            let scalar_result = scalar::compare_bytes(&a, &b, a.len().min(MAX_COMPARE_LEN));
            assert_eq!(
                simd_result, scalar_result,
                "mismatch_pos={} expected={}",
                mismatch_pos, scalar_result
            );
        }
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

    #[test]
    fn test_compare_bytes_empty() {
        let d = Dispatcher::new();
        assert_eq!(d.compare_bytes(&[], &[1, 2, 3]), 0);
        assert_eq!(d.compare_bytes(&[1, 2, 3], &[]), 0);
        assert_eq!(d.compare_bytes(&[], &[]), 0);
    }
}
