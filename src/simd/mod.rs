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
// rANS 4-way shared-stream SIMD decode (SSSE3 + SSE4.1, ryg_rans-style)
// ---------------------------------------------------------------------------

/// PSHUFB shuffle masks for branchless renormalization.
///
/// For each 4-bit renorm mask (which lanes need a word), maps consecutive
/// u16 words from the shared stream to the correct 32-bit lane positions.
/// Each word occupies the low 2 bytes of a 4-byte lane; upper 2 bytes are
/// zero-filled (0x80 in PSHUFB = zero output byte).
///
/// Index: 4-bit mask where bit i means lane i needs renormalization.
/// Entry: 16-byte PSHUFB control mask.
#[cfg(target_arch = "x86_64")]
#[repr(align(16))]
struct AlignedMask([u8; 16]);

#[cfg(target_arch = "x86_64")]
static SHUFFLE_MASKS: [AlignedMask; 16] = [
    // 0b0000: no lanes
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b0001: lane 0
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b0010: lane 1
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b0011: lanes 0,1
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b0100: lane 2
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b0101: lanes 0,2
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b0110: lanes 1,2
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b0111: lanes 0,1,2
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80,
        0x80,
    ]),
    // 0b1000: lane 3
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x80,
        0x80,
    ]),
    // 0b1001: lanes 0,3
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02, 0x03, 0x80,
        0x80,
    ]),
    // 0b1010: lanes 1,3
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02, 0x03, 0x80,
        0x80,
    ]),
    // 0b1011: lanes 0,1,3
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x04, 0x05, 0x80,
        0x80,
    ]),
    // 0b1100: lanes 2,3
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x80, 0x80, 0x02, 0x03, 0x80,
        0x80,
    ]),
    // 0b1101: lanes 0,2,3
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x04, 0x05, 0x80,
        0x80,
    ]),
    // 0b1110: lanes 1,2,3
    AlignedMask([
        0x80, 0x80, 0x80, 0x80, 0x00, 0x01, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x04, 0x05, 0x80,
        0x80,
    ]),
    // 0b1111: all lanes
    AlignedMask([
        0x00, 0x01, 0x80, 0x80, 0x02, 0x03, 0x80, 0x80, 0x04, 0x05, 0x80, 0x80, 0x06, 0x07, 0x80,
        0x80,
    ]),
];

/// Number of u16 words consumed for each 4-bit renorm mask (popcount).
#[cfg(target_arch = "x86_64")]
static WORD_COUNTS: [usize; 16] = [0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4];

/// SSSE3 + SSE4.1 branchless 4-way shared-stream rANS decode.
///
/// This is the high-performance decode path combining three ryg_rans techniques:
///
/// 1. **Merged slot table**: single `slot_table[slot].freq_bias` read per lane
///    (eliminates 2-hop gather: slot→symbol→freq/cum)
///
/// 2. **SSE4.1 `_mm_mullo_epi32`**: 4-lane u32 multiply in one instruction
///    (vs SSE2's awkward 2× `_mm_mul_epu32` with interleave)
///
/// 3. **PSHUFB branchless renorm**: `_mm_shuffle_epi8` routes consecutive words
///    from the shared stream to the correct lanes based on a 4-bit renorm mask.
///    No per-lane branches — the renorm pattern is a table lookup.
///
/// The inner loop is fully branchless: the only data-dependent value is the
/// 4-bit renorm mask (0..15), which selects from precomputed shuffle tables.
///
/// # Safety
/// Requires SSSE3 (for `_mm_shuffle_epi8`) and SSE4.1 (for `_mm_mullo_epi32`
/// and `_mm_blendv_epi8`). Caller must ensure these features are available.
/// `shared_words` must have at least 8 bytes of readable padding past the end
/// (the encode format provides 8 bytes of zero padding).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3", enable = "sse4.1")]
pub unsafe fn rans_decode_4way_shared_ssse3(
    shared_words: &[u16],
    initial_states: &[u32],
    slot2sym: &[u8],
    slot_table: &[crate::rans::SlotEntry],
    scale_bits: u32,
    original_len: usize,
    num_states: usize,
) -> Option<Vec<u8>> {
    use std::arch::x86_64::*;

    if num_states != 4 || initial_states.len() != 4 {
        return None;
    }

    let scale_mask = (1u32 << scale_bits) - 1;
    let table_len = slot2sym.len();

    // Load initial states into SSE register
    let mut states = _mm_set_epi32(
        initial_states[3] as i32,
        initial_states[2] as i32,
        initial_states[1] as i32,
        initial_states[0] as i32,
    );

    // Constants
    let sign_bit = _mm_set1_epi32(0x80000000u32 as i32);
    // RANS_L = 0x10000, threshold for signed compare = 0x10000 ^ 0x80000000 = 0x80010000
    let threshold = _mm_set1_epi32(0x80010000u32 as i32);
    // Vector shift count for _mm_srl_epi32 (runtime scale_bits → register operand)
    let scale_vec = _mm_set_epi64x(0, scale_bits as i64);

    let mut output = vec![0u8; original_len];
    let mut out_pos = 0;

    // Raw pointer into shared word stream (byte-addressed for _mm_loadl_epi64)
    let word_base = shared_words.as_ptr() as *const u8;
    let mut word_byte_pos: usize = 0;

    let full_quads = original_len / 4;
    let remainder = original_len % 4;

    // Temporary array for extracting states
    let mut state_arr = [0u32; 4];

    for _ in 0..full_quads {
        // Step 1: Extract individual states for scalar gather
        _mm_storeu_si128(state_arr.as_mut_ptr() as *mut __m128i, states);

        let slot0 = state_arr[0] & scale_mask;
        let slot1 = state_arr[1] & scale_mask;
        let slot2 = state_arr[2] & scale_mask;
        let slot3 = state_arr[3] & scale_mask;

        // Bounds check
        if (slot0 as usize | slot1 as usize | slot2 as usize | slot3 as usize) >= table_len {
            return None;
        }

        // Step 2: Scalar gather — symbols and merged freq+bias
        let s0 = *slot2sym.get_unchecked(slot0 as usize);
        let s1 = *slot2sym.get_unchecked(slot1 as usize);
        let s2 = *slot2sym.get_unchecked(slot2 as usize);
        let s3 = *slot2sym.get_unchecked(slot3 as usize);

        let e0 = slot_table.get_unchecked(slot0 as usize).freq_bias;
        let e1 = slot_table.get_unchecked(slot1 as usize).freq_bias;
        let e2 = slot_table.get_unchecked(slot2 as usize).freq_bias;
        let e3 = slot_table.get_unchecked(slot3 as usize).freq_bias;

        // Step 3: Pack freq_bias into SIMD, split into freq (low 16) and bias (high 16)
        let fb_vec = _mm_set_epi32(e3 as i32, e2 as i32, e1 as i32, e0 as i32);
        let freq_vec = _mm_and_si128(fb_vec, _mm_set1_epi32(0xFFFF));
        let bias_vec = _mm_srli_epi32::<16>(fb_vec);

        // Step 4: State transition — freq * (state >> scale_bits) + bias
        // _mm_srl_epi32 uses a register operand (not immediate), so scale_bits
        // can be a runtime value — single psrld instruction vs scalar fallback
        let shifted = _mm_srl_epi32(states, scale_vec);
        let products = _mm_mullo_epi32(freq_vec, shifted); // SSE4.1
        states = _mm_add_epi32(products, bias_vec);

        // Step 5: Unsigned compare — which lanes need renormalization?
        // state < RANS_L ⟺ (state ^ 0x80000000) < (RANS_L ^ 0x80000000) [signed]
        let xored = _mm_xor_si128(states, sign_bit);
        let cmp = _mm_cmplt_epi32(xored, threshold); // -1 where state < RANS_L
        let mask_4bit = _mm_movemask_ps(_mm_castsi128_ps(cmp)) as usize;

        // Step 6: Load words from shared stream and shuffle to correct lanes
        // _mm_loadl_epi64 loads 8 bytes (up to 4 u16 words) — safe due to padding
        let raw_words = _mm_loadl_epi64(word_base.add(word_byte_pos) as *const __m128i);
        let shuf_ctrl = _mm_load_si128(SHUFFLE_MASKS[mask_4bit].0.as_ptr() as *const __m128i);
        let word_vec = _mm_shuffle_epi8(raw_words, shuf_ctrl); // SSSE3

        // Step 7: Compute merged states — (state << 16) | word
        let shifted_states = _mm_slli_epi32(states, 16);
        let merged = _mm_or_si128(shifted_states, word_vec);

        // Step 8: Conditional update — blend merged where renorm needed
        states = _mm_blendv_epi8(states, merged, cmp); // SSE4.1

        // Step 9: Advance shared pointer by number of words consumed
        word_byte_pos += WORD_COUNTS[mask_4bit] * 2;

        // Step 10: Write output symbols
        *output.get_unchecked_mut(out_pos) = s0;
        *output.get_unchecked_mut(out_pos + 1) = s1;
        *output.get_unchecked_mut(out_pos + 2) = s2;
        *output.get_unchecked_mut(out_pos + 3) = s3;
        out_pos += 4;
    }

    // Scalar remainder for last < 4 symbols
    _mm_storeu_si128(state_arr.as_mut_ptr() as *mut __m128i, states);
    let mut wp = word_byte_pos / 2; // convert back to word index
    #[allow(clippy::needless_range_loop)] // lane indexes state_arr + drives slot lookups
    for lane in 0..remainder {
        let slot = state_arr[lane] & scale_mask;
        if slot as usize >= table_len {
            return None;
        }
        let s = *slot2sym.get_unchecked(slot as usize);
        let e = slot_table.get_unchecked(slot as usize).freq_bias;
        state_arr[lane] = (e & 0xFFFF) * (state_arr[lane] >> scale_bits) + (e >> 16);

        if state_arr[lane] < (1 << 16) && wp < shared_words.len() {
            state_arr[lane] = (state_arr[lane] << 16) | *shared_words.get_unchecked(wp) as u32;
            wp += 1;
        }

        *output.get_unchecked_mut(out_pos) = s;
        out_pos += 1;
    }

    Some(output)
}

// ---------------------------------------------------------------------------
// rANS 4-way shared-stream AVX2 decode (hardware gather)
// ---------------------------------------------------------------------------

/// AVX2 4-way shared-stream rANS decode with hardware gather.
///
/// Replaces the scalar gather bottleneck that made the SSSE3 path slower
/// than scalar (432 MiB/s vs 1.18 GiB/s). AVX2's `_mm_i32gather_epi32`
/// does all 4 slot_table lookups in a single instruction, eliminating
/// the extract→load→repack overhead.
///
/// Combines all four ryg_rans techniques:
///
/// 1. **Merged slot table**: single gather reads freq+bias per lane
/// 2. **AVX2 hardware gather**: `_mm_i32gather_epi32` replaces 4 scalar loads
/// 3. **SSE4.1 `_mm_mullo_epi32`**: 4-lane u32 multiply
/// 4. **PSHUFB branchless renorm**: SSSE3 shuffle routes shared-stream words
///
/// # Safety
/// Requires AVX2 (which implies SSSE3 + SSE4.1). `shared_words` must have
/// at least 8 bytes of readable padding past the end (the encode format
/// provides 8 bytes of zero padding).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn rans_decode_4way_shared_avx2(
    shared_words: &[u16],
    initial_states: &[u32],
    slot2sym: &[u8],
    slot_table: &[crate::rans::SlotEntry],
    scale_bits: u32,
    original_len: usize,
    num_states: usize,
) -> Option<Vec<u8>> {
    use std::arch::x86_64::*;

    if num_states != 4 || initial_states.len() != 4 {
        return None;
    }

    let scale_mask = (1u32 << scale_bits) - 1;
    let table_len = slot2sym.len();

    // Load initial states into SSE register
    let mut states = _mm_set_epi32(
        initial_states[3] as i32,
        initial_states[2] as i32,
        initial_states[1] as i32,
        initial_states[0] as i32,
    );

    // Constants
    let scale_mask_vec = _mm_set1_epi32(scale_mask as i32);
    let sign_bit = _mm_set1_epi32(0x80000000u32 as i32);
    let threshold = _mm_set1_epi32(0x80010000u32 as i32);
    let scale_vec = _mm_set_epi64x(0, scale_bits as i64);
    let lo16_mask = _mm_set1_epi32(0xFFFF);

    let mut output = vec![0u8; original_len];
    let mut out_pos = 0;

    // Raw pointer into shared word stream
    let word_base = shared_words.as_ptr() as *const u8;
    let mut word_byte_pos: usize = 0;

    // Slot table base pointer for gather (each entry is 4 bytes = 1 i32)
    let slot_base = slot_table.as_ptr() as *const i32;

    let full_quads = original_len / 4;
    let remainder = original_len % 4;

    // Temporary for extracting slots for symbol lookup
    let mut slot_arr = [0u32; 4];

    for _ in 0..full_quads {
        // Step 1: Compute slots = states & scale_mask (SIMD)
        let slots = _mm_and_si128(states, scale_mask_vec);

        // Step 2: Extract slots for bounds check + symbol lookup (scalar)
        _mm_storeu_si128(slot_arr.as_mut_ptr() as *mut __m128i, slots);

        // Bounds check all 4
        if (slot_arr[0] as usize
            | slot_arr[1] as usize
            | slot_arr[2] as usize
            | slot_arr[3] as usize)
            >= table_len
        {
            return None;
        }

        // Scalar symbol lookup (slot2sym is u8[], not worth SIMD-gathering)
        let s0 = *slot2sym.get_unchecked(slot_arr[0] as usize);
        let s1 = *slot2sym.get_unchecked(slot_arr[1] as usize);
        let s2 = *slot2sym.get_unchecked(slot_arr[2] as usize);
        let s3 = *slot2sym.get_unchecked(slot_arr[3] as usize);

        // Step 3: AVX2 hardware gather for freq_bias (THE key optimization)
        // SCALE=4: byte_offset = slot * 4, matching SlotEntry's 4-byte stride
        let fb_vec = _mm_i32gather_epi32::<4>(slot_base, slots);

        // Step 4: Split freq (low 16) and bias (high 16)
        let freq_vec = _mm_and_si128(fb_vec, lo16_mask);
        let bias_vec = _mm_srli_epi32::<16>(fb_vec);

        // Step 5: State transition — freq * (state >> scale_bits) + bias
        let shifted = _mm_srl_epi32(states, scale_vec);
        let products = _mm_mullo_epi32(freq_vec, shifted);
        states = _mm_add_epi32(products, bias_vec);

        // Step 6: Unsigned compare — which lanes need renormalization?
        let xored = _mm_xor_si128(states, sign_bit);
        let cmp = _mm_cmplt_epi32(xored, threshold);
        let mask_4bit = _mm_movemask_ps(_mm_castsi128_ps(cmp)) as usize;

        // Step 7: Load words from shared stream and shuffle to correct lanes
        let raw_words = _mm_loadl_epi64(word_base.add(word_byte_pos) as *const __m128i);
        let shuf_ctrl = _mm_load_si128(SHUFFLE_MASKS[mask_4bit].0.as_ptr() as *const __m128i);
        let word_vec = _mm_shuffle_epi8(raw_words, shuf_ctrl);

        // Step 8: Compute merged states — (state << 16) | word
        let shifted_states = _mm_slli_epi32(states, 16);
        let merged = _mm_or_si128(shifted_states, word_vec);

        // Step 9: Conditional update — blend merged where renorm needed
        states = _mm_blendv_epi8(states, merged, cmp);

        // Step 10: Advance shared pointer by number of words consumed
        word_byte_pos += WORD_COUNTS[mask_4bit] * 2;

        // Step 11: Write output symbols
        *output.get_unchecked_mut(out_pos) = s0;
        *output.get_unchecked_mut(out_pos + 1) = s1;
        *output.get_unchecked_mut(out_pos + 2) = s2;
        *output.get_unchecked_mut(out_pos + 3) = s3;
        out_pos += 4;
    }

    // Scalar remainder for last < 4 symbols
    let mut state_arr = [0u32; 4];
    _mm_storeu_si128(state_arr.as_mut_ptr() as *mut __m128i, states);
    let mut wp = word_byte_pos / 2;
    #[allow(clippy::needless_range_loop)] // lane indexes state_arr + drives slot lookups
    for lane in 0..remainder {
        let slot = state_arr[lane] & scale_mask;
        if slot as usize >= table_len {
            return None;
        }
        let s = *slot2sym.get_unchecked(slot as usize);
        let e = slot_table.get_unchecked(slot as usize).freq_bias;
        state_arr[lane] = (e & 0xFFFF) * (state_arr[lane] >> scale_bits) + (e >> 16);

        if state_arr[lane] < (1 << 16) && wp < shared_words.len() {
            state_arr[lane] = (state_arr[lane] << 16) | *shared_words.get_unchecked(wp) as u32;
            wp += 1;
        }

        *output.get_unchecked_mut(out_pos) = s;
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
mod tests;
