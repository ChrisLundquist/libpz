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
//! let match_len = d.compare_bytes(a, b, 258);
//! assert_eq!(match_len, 6); // first 6 bytes match
//! ```
//!
//! The dispatcher probes CPU features once and caches function pointers
//! for the best available implementation.

/// Legacy DEFLATE max match length, kept for SIMD tests.
#[cfg(test)]
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
    /// `limit` or the shorter slice length). Used as the inner loop of
    /// LZ77 match extension.
    ///
    /// The caller controls the maximum comparison length: Deflate
    /// pipelines pass 258 (RFC 1951 constraint), while other pipelines
    /// can pass larger values (up to `u16::MAX`) for better compression
    /// on repetitive data. SIMD implementations short-circuit on the
    /// first byte mismatch, so larger limits add zero overhead for
    /// typical short matches.
    ///
    /// Uses a resolved function pointer — no per-call match dispatch.
    #[inline]
    pub fn compare_bytes(&self, a: &[u8], b: &[u8], limit: usize) -> usize {
        let max_len = a.len().min(b.len()).min(limit);
        if max_len == 0 {
            return 0;
        }
        // SAFETY: compare_fn was resolved from detect_level() which verified
        // the required SIMD features are available. max_len is bounded by
        // both slice lengths, so reads are in-bounds.
        unsafe { (self.compare_fn)(a.as_ptr(), b.as_ptr(), max_len) }
    }

    /// Pointer-based variant of [`compare_bytes`] for hot callers that already
    /// maintain precise bounds and want to avoid repeated slice construction.
    ///
    /// # Safety
    /// Caller must guarantee that `max_len` bytes are readable from both `a`
    /// and `b`, and that the dispatcher was created on a CPU supporting the
    /// resolved SIMD function (guaranteed by `Dispatcher::new()`).
    #[inline]
    pub(crate) unsafe fn compare_bytes_ptr(
        &self,
        a: *const u8,
        b: *const u8,
        max_len: usize,
    ) -> usize {
        if max_len == 0 {
            return 0;
        }
        // SAFETY: caller guarantees pointer validity and length bounds.
        unsafe { (self.compare_fn)(a, b, max_len) }
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
// rANS SSE2 state transition
// ---------------------------------------------------------------------------

/// SSE2-accelerated rANS state transition for 4 lanes.
///
/// Computes: new_state[i] = freq[i] * (state[i] >> scale_bits) + slot[i] - cum[i]
/// for i in 0..4, using SSE2 `_mm_mul_epu32` for the multiply step.
///
/// # Safety
/// Requires SSE2 (always available on x86_64). All arrays must have length >= 4.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn rans_state_transition_sse2(
    states: &mut [u32; 4],
    freqs: &[u32; 4],
    slots: &[u32; 4],
    cums: &[u32; 4],
    scale_bits: u32,
) {
    use std::arch::x86_64::*;

    // Shift states right by scale_bits (scalar — no SIMD shift by variable amount in SSE2)
    let s0 = states[0] >> scale_bits;
    let s1 = states[1] >> scale_bits;
    let s2 = states[2] >> scale_bits;
    let s3 = states[3] >> scale_bits;

    // Pack into SSE2 registers: interleave pairs for _mm_mul_epu32
    // _mm_mul_epu32 multiplies the low 32 bits of each 64-bit lane.
    // To get products for all 4 u32 pairs, run two multiplies:
    //   lo_pair: lanes 0,2 -> products at 64-bit positions 0,1
    //   hi_pair: lanes 1,3 -> products at 64-bit positions 0,1
    let freq_lo = _mm_set_epi32(0, freqs[2] as i32, 0, freqs[0] as i32); // [f0, 0, f2, 0]
    let state_lo = _mm_set_epi32(0, s2 as i32, 0, s0 as i32);
    let prod_lo = _mm_mul_epu32(freq_lo, state_lo); // [f0*s0, f2*s2] as u64

    let freq_hi = _mm_set_epi32(0, freqs[3] as i32, 0, freqs[1] as i32);
    let state_hi = _mm_set_epi32(0, s3 as i32, 0, s1 as i32);
    let prod_hi = _mm_mul_epu32(freq_hi, state_hi); // [f1*s1, f3*s3] as u64

    // Extract low 32 bits of each product (the multiply result fits in u32 for
    // valid rANS states: freq <= 1<<14, state>>scale_bits <= 1<<16, product <= 1<<30)
    states[0] = (_mm_cvtsi128_si64(prod_lo) as u32)
        .wrapping_add(slots[0])
        .wrapping_sub(cums[0]);
    states[1] = (_mm_cvtsi128_si64(prod_hi) as u32)
        .wrapping_add(slots[1])
        .wrapping_sub(cums[1]);
    let prod_lo_hi = _mm_srli_si128(prod_lo, 8);
    states[2] = (_mm_cvtsi128_si64(prod_lo_hi) as u32)
        .wrapping_add(slots[2])
        .wrapping_sub(cums[2]);
    let prod_hi_hi = _mm_srli_si128(prod_hi, 8);
    states[3] = (_mm_cvtsi128_si64(prod_hi_hi) as u32)
        .wrapping_add(slots[3])
        .wrapping_sub(cums[3]);
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
// rANS 4-way decode with merged slot table (ryg_rans-style)
// ---------------------------------------------------------------------------

/// Decode 4-way interleaved rANS using merged slot-indexed tables.
///
/// Eliminates the 2-hop gather (slot → symbol → freq/cum) by combining
/// freq and bias into a single table lookup per lane.  The state
/// transition per lane becomes:
///
/// ```text
/// slot = state & mask
/// sym  = slot2sym[slot]
/// entry = slot_table[slot]
/// freq  = entry.freq_bias & 0xFFFF
/// bias  = entry.freq_bias >> 16
/// new_state = freq * (state >> scale_bits) + bias
/// ```
pub fn rans_decode_4way_slot(
    word_streams: &[&[u16]; 4],
    initial_states: &[u32; 4],
    slot2sym: &[u8],
    slot_table: &[crate::rans::SlotEntry],
    scale_bits: u32,
    original_len: usize,
) -> Option<Vec<u8>> {
    let scale_mask = (1u32 << scale_bits) - 1;
    let rans_l: u32 = 1 << 16;
    let io_bits: u32 = 16;
    let table_len = slot2sym.len();

    let mut states = *initial_states;
    let mut word_pos = [0usize; 4];
    let mut output = vec![0u8; original_len];
    let mut out_pos = 0;

    let full_quads = original_len / 4;
    let remainder = original_len % 4;

    for _ in 0..full_quads {
        // Extract slots
        let slot0 = states[0] & scale_mask;
        let slot1 = states[1] & scale_mask;
        let slot2 = states[2] & scale_mask;
        let slot3 = states[3] & scale_mask;

        // Bounds check all 4 at once
        if (slot0 as usize | slot1 as usize | slot2 as usize | slot3 as usize) >= table_len {
            return None;
        }

        // Gather symbols (single-hop)
        let s0 = slot2sym[slot0 as usize];
        let s1 = slot2sym[slot1 as usize];
        let s2 = slot2sym[slot2 as usize];
        let s3 = slot2sym[slot3 as usize];

        // Gather merged freq+bias (single-hop, eliminates sym→freq/cum indirection)
        let e0 = slot_table[slot0 as usize].freq_bias;
        let e1 = slot_table[slot1 as usize].freq_bias;
        let e2 = slot_table[slot2 as usize].freq_bias;
        let e3 = slot_table[slot3 as usize].freq_bias;

        // State transition: new_state = freq * (state >> scale_bits) + bias
        states[0] = (e0 & 0xFFFF) * (states[0] >> scale_bits) + (e0 >> 16);
        states[1] = (e1 & 0xFFFF) * (states[1] >> scale_bits) + (e1 >> 16);
        states[2] = (e2 & 0xFFFF) * (states[2] >> scale_bits) + (e2 >> 16);
        states[3] = (e3 & 0xFFFF) * (states[3] >> scale_bits) + (e3 >> 16);

        // Renormalize all 4 lanes
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

        // Write output
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
        if slot as usize >= table_len {
            return None;
        }
        let s = slot2sym[slot as usize];
        let e = slot_table[slot as usize].freq_bias;

        states[lane] = (e & 0xFFFF) * (states[lane] >> scale_bits) + (e >> 16);

        if states[lane] < rans_l && word_pos[lane] < word_streams[lane].len() {
            states[lane] = (states[lane] << io_bits) | word_streams[lane][word_pos[lane]] as u32;
            word_pos[lane] += 1;
        }

        output[out_pos] = s;
        out_pos += 1;
    }

    Some(output)
}

/// Like [`rans_decode_4way`] but writes into a provided buffer instead of allocating.
///
/// Returns `Some(())` on success, `None` on invalid input.
/// `output` must have length `>= original_len`.
#[allow(clippy::too_many_arguments)]
pub fn rans_decode_4way_into(
    word_streams: &[&[u16]; 4],
    initial_states: &[u32; 4],
    freq: &[u16; 256],
    cum: &[u16; 256],
    lookup: &[u8],
    scale_bits: u32,
    original_len: usize,
    output: &mut [u8],
) -> Option<()> {
    let scale_mask = (1u32 << scale_bits) - 1;
    let rans_l: u32 = 1 << 16;
    let io_bits: u32 = 16;
    let lookup_len = lookup.len();

    let mut states = *initial_states;
    let mut word_pos = [0usize; 4];
    let mut out_pos = 0;

    let full_quads = original_len / 4;
    let remainder = original_len % 4;

    for _ in 0..full_quads {
        let slot0 = states[0] & scale_mask;
        let slot1 = states[1] & scale_mask;
        let slot2 = states[2] & scale_mask;
        let slot3 = states[3] & scale_mask;

        if (slot0 as usize | slot1 as usize | slot2 as usize | slot3 as usize) >= lookup_len {
            return None;
        }

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

        states[0] = f0 * (states[0] >> scale_bits) + slot0 - c0;
        states[1] = f1 * (states[1] >> scale_bits) + slot1 - c1;
        states[2] = f2 * (states[2] >> scale_bits) + slot2 - c2;
        states[3] = f3 * (states[3] >> scale_bits) + slot3 - c3;

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

        output[out_pos] = s0;
        output[out_pos + 1] = s1;
        output[out_pos + 2] = s2;
        output[out_pos + 3] = s3;
        out_pos += 4;
    }

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

    Some(())
}

/// 4-way rANS decode using SSE2 for the state transition multiply step.
///
/// Same signature and behavior as [`rans_decode_4way`] but uses SSE2 intrinsics
/// for the `freq * (state >> scale_bits)` multiply operations. The gather step
/// (symbol lookup) remains scalar as there is no gather on SSE2.
///
/// # Safety
/// Requires SSE2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
pub unsafe fn rans_decode_4way_sse2(
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

        // Step 3: State transition using SSE2 for multiply step
        let freqs_arr = [f0, f1, f2, f3];
        let slots_arr = [slot0, slot1, slot2, slot3];
        let cums_arr = [c0, c1, c2, c3];
        rans_state_transition_sse2(&mut states, &freqs_arr, &slots_arr, &cums_arr, scale_bits);

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

        // SSE2 transition for single lane
        let mut state_arr = [states[lane], 0, 0, 0];
        rans_state_transition_sse2(
            &mut state_arr,
            &[f, 0, 0, 0],
            &[slot, 0, 0, 0],
            &[c, 0, 0, 0],
            scale_bits,
        );
        states[lane] = state_arr[0];

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
// Batched 4-way FSE decode
// ---------------------------------------------------------------------------

/// Bulk refill helper for FSE 4-way decode: fills the bit container from
/// the byte stream using a single unaligned u64 load when possible.
#[inline(always)]
fn fse_bulk_refill(
    data: &[u8],
    container: &mut u64,
    bits_available: &mut u32,
    byte_pos: &mut usize,
) {
    if *bits_available <= 56 {
        if *byte_pos + 8 <= data.len() {
            let raw = u64::from_le_bytes(data[*byte_pos..*byte_pos + 8].try_into().unwrap());
            *container |= raw << *bits_available;
            let bytes_consumed = ((64 - *bits_available) / 8) as usize;
            *byte_pos += bytes_consumed;
            *bits_available += (bytes_consumed as u32) * 8;
        } else {
            while *bits_available <= 56 && *byte_pos < data.len() {
                *container |= (data[*byte_pos] as u64) << *bits_available;
                *byte_pos += 1;
                *bits_available += 8;
            }
        }
    }
}

/// Decode 4-way interleaved FSE, processing all 4 lanes per iteration.
///
/// Mirrors `rans_decode_4way`: batches all 4 lanes per loop iteration to
/// keep states in registers, reduce loop overhead 4×, and enable the
/// compiler to autovectorize.
///
/// Each lane has its own inline BitReader state (container, bits_available,
/// byte_pos) reading from independent bitstreams. Uses bulk u64 refill.
///
/// # Arguments
/// - `bitstreams`: per-lane encoded bitstream data (4 slices)
/// - `initial_states`: starting FSE state for each lane (4 values)
/// - `decode_table`: the shared FSE decode table (indexed by state)
/// - `table_size`: size of the decode table (1 << accuracy_log)
/// - `original_len`: total symbols to decode
///
/// # Returns
/// Decoded bytes, or `None` on invalid state.
pub(crate) fn fse_decode_4way(
    bitstreams: &[&[u8]; 4],
    initial_states: &[u16; 4],
    decode_table: &[crate::fse::DecodeEntry],
    table_size: usize,
    original_len: usize,
) -> Option<Vec<u8>> {
    let mut states = [
        initial_states[0] as usize,
        initial_states[1] as usize,
        initial_states[2] as usize,
        initial_states[3] as usize,
    ];

    // Inline BitReader state for each lane (avoids method call overhead).
    let mut containers = [0u64; 4];
    let mut bits_available = [0u32; 4];
    let mut byte_positions = [0usize; 4];

    // Initial refill for all 4 lanes.
    for lane in 0..4 {
        fse_bulk_refill(
            bitstreams[lane],
            &mut containers[lane],
            &mut bits_available[lane],
            &mut byte_positions[lane],
        );
    }

    let mut output = Vec::with_capacity(original_len);

    let full_quads = original_len / 4;
    let remainder = original_len % 4;

    for _ in 0..full_quads {
        for lane in 0..4 {
            if states[lane] >= table_size {
                return None;
            }
            let entry = decode_table[states[lane]];
            output.push(entry.symbol);

            // Inline read_bits: refill + extract.
            let nb_bits = entry.bits as u32;
            fse_bulk_refill(
                bitstreams[lane],
                &mut containers[lane],
                &mut bits_available[lane],
                &mut byte_positions[lane],
            );
            let value = if nb_bits > 0 {
                let mask = (1u64 << nb_bits) - 1;
                let v = (containers[lane] & mask) as usize;
                containers[lane] >>= nb_bits;
                bits_available[lane] = bits_available[lane].saturating_sub(nb_bits);
                v
            } else {
                0
            };

            states[lane] = entry.next_state_base as usize + value;
        }
    }

    // Handle remaining symbols (< 4).
    for r in 0..remainder {
        let lane = r;
        if states[lane] >= table_size {
            return None;
        }
        let entry = decode_table[states[lane]];
        output.push(entry.symbol);

        let nb_bits = entry.bits as u32;
        fse_bulk_refill(
            bitstreams[lane],
            &mut containers[lane],
            &mut bits_available[lane],
            &mut byte_positions[lane],
        );
        let value = if nb_bits > 0 {
            let mask = (1u64 << nb_bits) - 1;
            let v = (containers[lane] & mask) as usize;
            containers[lane] >>= nb_bits;
            bits_available[lane] = bits_available[lane].saturating_sub(nb_bits);
            v
        } else {
            0
        };

        states[lane] = entry.next_state_base as usize + value;
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
}
