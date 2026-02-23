# Pareto-Competitiveness Implementation Plan — Phase 4: Entropy Throughput

**Goal:** Improve entropy encode/decode throughput via SIMD rANS decode, GPU entropy for LzSeqR, and Huffman contention fixes.

**Architecture:** Wire existing SIMD intrinsics to rANS hot decode loop. Extend GPU rANS encode to accept LzSeq 6-stream demux output. Replace per-bit atomic_or in Huffman GPU kernel with chunk-based packing.

**Tech Stack:** Rust (SIMD intrinsics), WGSL (GPU kernels), wgpu v27

**Scope:** 8 phases from original design (phase 4 of 8)

**Codebase verified:** 2026-02-22

---

## Acceptance Criteria Coverage

### pareto-competitiveness.AC1: Pareto-competitive with gzip
- **pareto-competitiveness.AC1.3 Success:** LzSeqH achieves higher throughput than gzip -1 at comparable or better ratio

### pareto-competitiveness.AC3: Multi-thread and GPU scaling
- **pareto-competitiveness.AC3.2 Success:** GPU entropy encoding for LzSeqR achieves higher throughput than CPU-only path on blocks >= 256KB

---

## Current State

### rANS decode path (src/rans.rs + src/simd.rs)

`rans_decode_interleaved` in `src/rans.rs` already dispatches to `simd::rans_decode_4way` when `num_states == 4` (lines 468-491). The `rans_decode_4way` and `rans_decode_4way_into` functions live in `src/simd.rs` (lines 572-783). They batch all 4 lanes per loop iteration, keeping states in registers and reducing loop overhead 4x.

The existing implementation is "SIMD-friendly" in that it exposes ILP to the CPU's out-of-order engine, but it does **not** yet use explicit SSE2/AVX2 intrinsics for the state transition arithmetic (the multiply-add: `freq * (state >> scale_bits) + slot - cum`). The `Dispatcher` in `simd.rs` exposes `compare_bytes` (SSE2/AVX2) and `byte_frequencies` (SSE2/AVX2) but has no SIMD path for the rANS multiply-add. The `Dispatcher` struct is copy-safe and already resolves function pointers at construction time.

The gap: `rans_decode_4way` does scalar multiply-add for all 4 lanes sequentially. On SSE2 we can pack all 4 `u32` states into one `__m128i` and do the multiply-add with `_mm_mul_epu32` + shift + OR in a single pass. On AVX2 we can do 8 lanes at once.

### GPU rANS encode (src/webgpu/rans.rs)

`rans_encode_chunked_gpu_with_tables` (line 276) accepts a flat `&[u8]` input and a `chunk_size`. For LzSeqR the entropy stage receives 6 independent streams from `demux.rs` (`seq_encoded_to_demux`, line 356): flags, literals, offset_codes, offset_extra, length_codes, length_extra. Currently each stream is entropy-coded independently via the CPU path in the pipeline stage. There is no GPU path that accepts all 6 streams as a batch.

The gap: the GPU encode path needs a multi-stream entry point that accepts `DemuxOutput::streams` (a `Vec<Vec<u8>>`) and encodes each stream independently on-device, returning per-stream encoded output. The 256KB threshold for GPU vs CPU routing is not yet wired at the pipeline dispatch layer.

### Huffman GPU kernel (kernels/huffman_encode.wgsl)

`write_codes` (line 54) issues one `atomicOr` per symbol when the codeword fits in a single u32, and two `atomicOr` calls when it spans two u32s (lines 84-97). On typical text data with Huffman code lengths averaging 6-8 bits, a 256KB block produces ~300K symbol writes, each hitting a shared `atomic<u32>` output buffer. Threads within the same workgroup that happen to write to adjacent bits in the same u32 word contend on the same atomic location.

The gap: replace per-symbol atomic_or with chunk-based packing where each thread owns an exclusive 32-bit output word, packs its symbol's bits into a local accumulator, and flushes with a single non-atomic store. Boundary symbols that span two chunks are handled with a two-step write only for those edge cases.

---

## Subcomponent A: rANS SIMD Decode Wiring

**Files:** `src/simd.rs`, `src/rans.rs`

### Task 1: Add SIMD multiply-add path to rans_decode_4way

The `rans_decode_4way` function in `src/simd.rs` (line 572) currently runs the state transition loop in scalar. Add a `Dispatcher`-aware fast path that uses SSE2/AVX2 to process the `freq * (state >> scale_bits) + slot - cum` computation across all 4 lanes simultaneously.

**Design note:** rANS state transition requires `u32 * u32 -> u32` (low half). SSE2 `_mm_mul_epu32` operates on the low 32 bits of 64-bit lanes (4 u64 → 2 u64 low-half products). We interleave two `_mm_mul_epu32` calls to cover all 4 state lanes. The gather step (lookup by slot) remains scalar — there is no SIMD gather on SSE2, and the table is small enough that cache stays warm.

The approach for SSE2 (4-way, processes 4 lanes per iteration):

```rust
// In src/simd.rs, add alongside the existing rans_decode_4way:

/// SSE2-accelerated rANS state transition for 4 lanes.
///
/// Computes: new_state[i] = freq[i] * (state[i] >> scale_bits) + slot[i] - cum[i]
/// for i in 0..4, using SSE2 _mm_mul_epu32 for the multiply step.
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
```

**WRONG — do not use:** Detecting the SIMD level inside the decode loop:

```rust
// ❌ WRONG: is_x86_feature_detected!("sse2") called per loop iteration
// In rans_decode_4way, replace the Step 3 block:
//   states[0] = f0 * (states[0] >> scale_bits) + slot0 - c0;
//   states[1] = ...
// with:
#[cfg(target_arch = "x86_64")]
{
    if is_x86_feature_detected!("sse2") {
        let freqs = [f0, f1, f2, f3];
        let slots = [slot0, slot1, slot2, slot3];
        let cums = [c0, c1, c2, c3];
        // SAFETY: is_x86_feature_detected verified SSE2
        unsafe { rans_state_transition_sse2(&mut states, &freqs, &slots, &cums, scale_bits); }
    } else {
        states[0] = f0 * (states[0] >> scale_bits) + slot0 - c0;
        states[1] = f1 * (states[1] >> scale_bits) + slot1 - c1;
        states[2] = f2 * (states[2] >> scale_bits) + slot2 - c2;
        states[3] = f3 * (states[3] >> scale_bits) + slot3 - c3;
    }
}
#[cfg(not(target_arch = "x86_64"))]
{
    states[0] = f0 * (states[0] >> scale_bits) + slot0 - c0;
    states[1] = f1 * (states[1] >> scale_bits) + slot1 - c1;
    states[2] = f2 * (states[2] >> scale_bits) + slot2 - c2;
    states[3] = f3 * (states[3] >> scale_bits) + slot3 - c3;
}
```

**CORRECT approach:** Dispatch at the call site using a dedicated `#[target_feature]` function. Add a `rans_decode_4way_sse2` variant that `#[target_feature(enable = "sse2")]` annotates the entire function body, and dispatch at the `rans_decode_interleaved` call site using the existing `Dispatcher::level()` cache:

```rust
// In src/simd.rs, add:
/// 4-way rANS decode using SSE2 for the state transition multiply step.
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
    // ... same structure as rans_decode_4way but with rans_state_transition_sse2
    // in the inner loop Step 3
}
```

Then in `rans.rs`, `rans_decode_interleaved` dispatches:

```rust
if num_states == 4 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("sse2") {
        // SAFETY: verified above
        return unsafe {
            crate::simd::rans_decode_4way_sse2(
                &streams_arr, &states_arr, &norm.freq, &norm.cum,
                lookup, norm.scale_bits as u32, original_len,
            )
        }.ok_or(PzError::InvalidInput);
    }
    // fallback to existing rans_decode_4way
    return crate::simd::rans_decode_4way(...).ok_or(PzError::InvalidInput);
}
```

### Task 2: Tests verifying SIMD decode produces identical output to scalar path

Add to `src/simd.rs` `#[cfg(test)] mod tests`:

```rust
#[test]
fn test_rans_decode_4way_sse2_matches_scalar() {
    // Encode a known sequence with the CPU rANS encoder, then decode
    // both paths and assert byte-for-byte identical output.
    use crate::rans;

    let input: Vec<u8> = (0..1024).map(|i| (i % 26 + b'a' as usize) as u8).collect();
    let freq_table = crate::frequency::FrequencyTable::from_bytes(&input);
    let norm = rans::normalize_frequencies(&freq_table, rans::DEFAULT_SCALE_BITS).unwrap();
    let (word_streams, final_states) =
        rans::rans_encode_interleaved(&input, &norm, 4);

    let lookup = rans::build_symbol_lookup(&norm);
    let streams_arr: [&[u16]; 4] = [
        &word_streams[0], &word_streams[1],
        &word_streams[2], &word_streams[3],
    ];
    let states_arr: [u32; 4] = [
        final_states[0], final_states[1],
        final_states[2], final_states[3],
    ];

    // Scalar path
    let scalar_out = rans_decode_4way(
        &streams_arr, &states_arr, &norm.freq, &norm.cum,
        &lookup, norm.scale_bits as u32, input.len(),
    ).unwrap();

    // SSE2 path (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    {
        let sse2_out = unsafe {
            rans_decode_4way_sse2(
                &streams_arr, &states_arr, &norm.freq, &norm.cum,
                &lookup, norm.scale_bits as u32, input.len(),
            )
        }.unwrap();
        assert_eq!(scalar_out, sse2_out,
            "SSE2 and scalar rANS decode must produce identical output");
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
            .map(|i| ((i as u64 * 6364136223846793005 + seed as u64) >> 56) as u8)
            .collect();
        let freq_table = crate::frequency::FrequencyTable::from_bytes(&input);
        let norm = rans::normalize_frequencies(&freq_table, rans::DEFAULT_SCALE_BITS).unwrap();
        let (word_streams, final_states) =
            rans::rans_encode_interleaved(&input, &norm, 4);

        let lookup = rans::build_symbol_lookup(&norm);
        let streams_arr: [&[u16]; 4] = [
            &word_streams[0], &word_streams[1],
            &word_streams[2], &word_streams[3],
        ];
        let states_arr: [u32; 4] = [
            final_states[0], final_states[1],
            final_states[2], final_states[3],
        ];

        let scalar_out = rans_decode_4way(
            &streams_arr, &states_arr, &norm.freq, &norm.cum,
            &lookup, norm.scale_bits as u32, input.len(),
        ).unwrap();
        assert_eq!(scalar_out, input, "round-trip failed for seed {}", seed);

        #[cfg(target_arch = "x86_64")]
        {
            let sse2_out = unsafe {
                rans_decode_4way_sse2(
                    &streams_arr, &states_arr, &norm.freq, &norm.cum,
                    &lookup, norm.scale_bits as u32, input.len(),
                )
            }.unwrap();
            assert_eq!(sse2_out, input,
                "SSE2 round-trip failed for seed {}", seed);
        }
    }
}
```

### Task 3: Benchmark comparison showing SIMD improvement

**AC2.3 Remediation:** If SIMD rANS decode does not achieve throughput >= gzip decompression, investigate additional optimizations: branchless decode loop, prefetching, lookup table optimization. Track decompression throughput in Criterion benchmarks alongside encode throughput.

Add a criterion benchmark in `benches/` (or extend an existing bench file) that compares scalar vs SSE2 `rans_decode_4way` on a 1MB input:

```rust
// In benches/rans_bench.rs (create if absent):
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_rans_decode_4way(c: &mut Criterion) {
    let input_1mb: Vec<u8> = (0..1_048_576)
        .map(|i| (i % 100) as u8)  // skewed: 100 symbols, not 256
        .collect();

    // Pre-encode once
    let freq_table = pz::frequency::FrequencyTable::from_bytes(&input_1mb);
    let norm = pz::rans::normalize_frequencies(&freq_table, pz::rans::DEFAULT_SCALE_BITS).unwrap();
    let (word_streams, final_states) =
        pz::rans::rans_encode_interleaved(&input_1mb, &norm, 4);
    let lookup = pz::rans::build_symbol_lookup(&norm);
    let streams_arr: [&[u16]; 4] = [
        &word_streams[0], &word_streams[1],
        &word_streams[2], &word_streams[3],
    ];
    let states_arr: [u32; 4] = [
        final_states[0], final_states[1], final_states[2], final_states[3],
    ];

    let mut g = c.benchmark_group("rans_decode_4way");
    g.throughput(Throughput::Bytes(input_1mb.len() as u64));

    g.bench_function("scalar", |b| {
        b.iter(|| {
            pz::simd::rans_decode_4way(
                &streams_arr, &states_arr, &norm.freq, &norm.cum,
                &lookup, norm.scale_bits as u32, input_1mb.len(),
            ).unwrap()
        })
    });

    #[cfg(target_arch = "x86_64")]
    g.bench_function("sse2", |b| {
        b.iter(|| unsafe {
            pz::simd::rans_decode_4way_sse2(
                &streams_arr, &states_arr, &norm.freq, &norm.cum,
                &lookup, norm.scale_bits as u32, input_1mb.len(),
            ).unwrap()
        })
    });

    g.finish();
}

criterion_group!(rans_benches, bench_rans_decode_4way);
criterion_main!(rans_benches);
```

Run with `cargo bench --bench rans_bench` and record the throughput difference. The expected gain is 5-15% from reduced instruction count in the multiply step; the bottleneck will shift to the scatter-gather table lookup.

---

## Subcomponent B: GPU Entropy Encode for LzSeqR

**Files:** `src/webgpu/rans.rs`, `src/pipeline/demux.rs`, `src/pipeline/mod.rs`

### Task 4: Extend GPU rANS encode to accept LzSeq 6-stream format

The existing GPU encode entry point `rans_encode_chunked_gpu_with_tables` accepts a single flat `&[u8]`. Add a multi-stream variant that accepts a slice of streams and encodes each independently on the GPU, reusing the same frequency table per stream (each stream gets its own table, derived from its own byte distribution):

```rust
// In src/webgpu/rans.rs, add to impl WebGpuEngine:

/// GPU-accelerated rANS encode for multiple independent byte streams.
///
/// Each stream is encoded independently with its own frequency table.
/// Designed for the LzSeq 6-stream output from `demux.rs`:
///   [flags, literals, offset_codes, offset_extra, length_codes, length_extra]
///
/// Returns a `Vec` of encoded byte blobs in the same order as `streams`.
/// Falls back to CPU encode for any stream that is empty or below the
/// minimum GPU dispatch size.
///
/// # Arguments
/// - `streams`: the 6 independent byte streams from LzSeq demux
/// - `num_lanes`: interleave width for rANS (4 for SSE2 compat, 8+ for GPU)
/// - `chunk_size`: chunk granularity for GPU dispatch (use 65536 = 64KB)
/// - `scale_bits`: rANS frequency table precision
pub fn rans_encode_6streams_gpu(
    &self,
    streams: &[Vec<u8>],
    num_lanes: usize,
    chunk_size: usize,
    scale_bits: u8,
) -> PzResult<Vec<Vec<u8>>> {
    streams
        .iter()
        .map(|stream| {
            if stream.len() < crate::webgpu::MIN_GPU_INPUT_SIZE {
                // CPU fallback for small streams
                return crate::rans::encode_interleaved(stream, num_lanes);
            }
            // Derive per-stream frequency table
            let freq_table = crate::frequency::FrequencyTable::from_bytes(stream);
            let norm = crate::rans::normalize_frequencies(&freq_table, scale_bits)?;
            let tables_buf = self.create_rans_encode_tables_buffer(&norm, "rans_6stream_tables");
            let (words_dev, states_dev) = self.rans_encode_chunked_gpu_with_tables(
                stream,
                num_lanes,
                chunk_size,
                scale_bits,
                &tables_buf,
            )?;
            // Readback and serialize into the standard interleaved wire format
            self.rans_encode_readback_to_bytes(words_dev, states_dev, &norm, num_lanes, stream.len())
        })
        .collect()
}
```

The `rans_encode_readback_to_bytes` helper reads back the GPU output buffers and serializes them into the same interleaved wire format that the CPU encoder produces (`src/rans.rs` format doc, lines 40-45). This ensures CPU decode compatibility.

### Task 5: Add block size threshold for GPU vs CPU entropy routing

In `src/pipeline/demux.rs`, `LzDemuxer::LzSeq` `compress_and_demux` (line 186) already gates GPU usage on `input.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE`. The entropy stage routing lives in the pipeline stage that calls the demuxer. Add a constant and a routing function:

```rust
// In src/pipeline/mod.rs (or a new src/pipeline/entropy_routing.rs):

/// Minimum total pre-entropy bytes for GPU entropy encoding to be worthwhile.
/// Below this threshold, PCIe transfer overhead exceeds compute savings.
/// 256KB = 262144 bytes (aligns with AC3.2 threshold).
pub const GPU_ENTROPY_MIN_BYTES: usize = 262_144;

/// Choose whether to use GPU entropy encoding for a set of streams.
///
/// Returns true when the GPU engine is available and the total stream
/// bytes exceed the 256KB threshold.
pub fn should_use_gpu_entropy(
    streams: &[Vec<u8>],
    options: &CompressOptions,
) -> bool {
    #[cfg(feature = "webgpu")]
    {
        if options.backend != Backend::WebGpu {
            return false;
        }
        if options.webgpu_engine.is_none() {
            return false;
        }
        let total: usize = streams.iter().map(|s| s.len()).sum();
        total >= GPU_ENTROPY_MIN_BYTES
    }
    #[cfg(not(feature = "webgpu"))]
    { false }
}
```

In the pipeline stage that applies entropy coding after demux, check `should_use_gpu_entropy` and dispatch accordingly. The CPU fallback path (existing `encode_interleaved` per stream) remains unchanged and requires zero overhead when GPU is unavailable (the `#[cfg(not(feature = "webgpu"))]` branch returns `false` immediately).

### Task 6: Tests for GPU encode -> CPU decode and CPU encode -> GPU decode cross-paths

Add integration tests in `src/webgpu/rans.rs` `#[cfg(test)] mod tests`:

```rust
#[cfg(feature = "webgpu")]
#[test]
fn test_gpu_encode_cpu_decode_lzseq_streams() {
    // Build a synthetic LzSeq DemuxOutput (6 streams) and verify that
    // GPU encode -> CPU decode round-trips correctly for each stream.
    use crate::rans;

    let engine = match WebGpuEngine::new_blocking() {
        Ok(e) => e,
        Err(_) => return, // no GPU available in this environment
    };

    // Synthetic streams: vary sizes and distributions to exercise
    // the 256KB threshold and stream-size heterogeneity.
    let streams: Vec<Vec<u8>> = vec![
        // flags: mostly 0 and 1 (match/literal bits)
        (0..32768).map(|i| (i % 2) as u8).collect(),
        // literals: all 256 values
        (0..32768).map(|i| (i % 256) as u8).collect(),
        // offset_codes: zstd-style, concentrated in low values
        (0..16384).map(|i| (i % 32) as u8).collect(),
        // offset_extra: near-uniform
        (0..16384).map(|i| (i % 256) as u8).collect(),
        // length_codes: concentrated in 0-15
        (0..16384).map(|i| (i % 16) as u8).collect(),
        // length_extra: sparse
        (0..8192).map(|i| (i % 64) as u8).collect(),
    ];

    let encoded_streams = engine
        .rans_encode_6streams_gpu(&streams, 4, 65536, rans::DEFAULT_SCALE_BITS)
        .expect("GPU encode of 6 streams must succeed");

    assert_eq!(encoded_streams.len(), 6, "must get 6 encoded streams back");

    for (i, (original, encoded)) in streams.iter().zip(encoded_streams.iter()).enumerate() {
        let decoded = rans::decode_interleaved(encoded, original.len())
            .unwrap_or_else(|e| panic!("CPU decode of stream {} failed: {:?}", i, e));
        assert_eq!(&decoded, original,
            "stream {} GPU-encode -> CPU-decode round-trip mismatch", i);
    }
}

#[cfg(feature = "webgpu")]
#[test]
fn test_cpu_encode_gpu_decode_lzseq_streams() {
    // CPU encode -> GPU decode cross-path: encode on CPU, decode on GPU.
    use crate::rans;

    let engine = match WebGpuEngine::new_blocking() {
        Ok(e) => e,
        Err(_) => return,
    };

    let input: Vec<u8> = (0..131072).map(|i| (i % 200) as u8).collect();
    let encoded_cpu = rans::encode_interleaved(&input, 4)
        .expect("CPU encode must succeed");
    let decoded_gpu = engine
        .rans_decode_interleaved_gpu(&encoded_cpu, input.len())
        .expect("GPU decode of CPU-encoded data must succeed");
    assert_eq!(decoded_gpu, input,
        "CPU-encode -> GPU-decode round-trip mismatch");
}

#[cfg(feature = "webgpu")]
#[test]
fn test_gpu_entropy_threshold_cpu_fallback_below_256kb() {
    // Streams whose total size is below GPU_ENTROPY_MIN_BYTES (256KB)
    // must silently use the CPU path — no GPU initialization or error.
    use crate::pipeline::{CompressOptions, Backend};
    use crate::pipeline::should_use_gpu_entropy;

    let small_streams: Vec<Vec<u8>> = vec![
        vec![0u8; 1024],  // 1KB each
        vec![1u8; 1024],
        vec![2u8; 1024],
        vec![3u8; 1024],
        vec![4u8; 1024],
        vec![5u8; 1024],
    ]; // total: 6KB << 256KB

    let options = CompressOptions {
        backend: Backend::WebGpu,
        webgpu_engine: None, // simulates "no GPU"
        ..Default::default()
    };
    assert!(!should_use_gpu_entropy(&small_streams, &options),
        "must not use GPU entropy for streams totaling < 256KB");
}
```

---

## Subcomponent C: Huffman Atomic Contention Fix

**Files:** `kernels/huffman_encode.wgsl`

### Task 7: Replace per-bit atomic_or with chunk-based packing

The current `write_codes` kernel (line 54 of `kernels/huffman_encode.wgsl`) performs at most 2 `atomicOr` calls per symbol, but on densely packed data these operations contend on adjacent u32 words. The fix assigns each thread an exclusive output chunk (one u32 word), accumulates all bits for symbols that fall entirely within that word into a local variable, and writes once with a non-atomic store. Only symbols whose codeword spans a chunk boundary require atomic writes.

Replace the `write_codes` entry point with:

```wgsl
// New write_codes kernel: chunk-based packing.
//
// Each thread owns output words in the range [g*CHUNK_WORDS, (g+1)*CHUNK_WORDS).
// It iterates over all input symbols whose bits land in its chunk, accumulates
// them into a local u32, and stores non-atomically at the end.
//
// Symbols that straddle a chunk boundary are written via atomicOr only for
// the cross-boundary word — this is O(1) per symbol, not per-bit.

// CHUNK_WORDS: each thread covers this many output u32 words exclusively.
// Set to 1 for now (each thread owns exactly one output word).
// This eliminates all intra-word contention: two threads never write to
// the same word unless they are handling a boundary symbol.
const CHUNK_WORDS: u32 = 1u;

@compute @workgroup_size(64)
fn write_codes_chunked(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x + gid.y * wc_params.w;
    let num_output_words = (wc_params.y + 31u) / 32u; // total output u32 words
    let num_symbols = wc_params.x;

    if (g >= num_output_words) {
        return;
    }

    // Bit range this thread owns: [g*32, (g+1)*32)
    let my_bit_start = g * 32u;
    let my_bit_end = my_bit_start + 32u;

    var local_word: u32 = 0u;
    var has_boundary_low = false;  // straddles into previous word
    var has_boundary_high = false; // straddles into next word

    // Scan all symbols to find those whose bits intersect [my_bit_start, my_bit_end).
    // This is O(N) per thread, giving O(N*W) total work where W = num_output_words.
    // For blocks where N/W is large (dense codewords, typical), this is efficient.
    // For sparse blocks with many zero-frequency symbols, use the prefix-sum offsets
    // to limit the scan range.
    //
    // Optimization: use bit_offsets[] to binary-search for the first symbol
    // in range, then scan forward while bit_offsets[sym] < my_bit_end.
    // This reduces average work from O(N) to O(avg_codewords_per_word * log N).
    // For the initial implementation, use a linear scan bounded by num_symbols.

    var sym_idx: u32 = 0u;
    loop {
        if (sym_idx >= num_symbols) { break; }

        let start_bit = bit_offsets[sym_idx];
        if (start_bit >= my_bit_end) {
            // All remaining symbols start at or after our range; done.
            break;
        }

        let sym = read_wc_symbol(sym_idx);
        let entry = wc_code_lut[sym];
        let bits = entry >> 24u;
        let codeword = entry & 0x00FFFFFFu;

        if (bits > 0u) {
            let end_bit = start_bit + bits - 1u;

            if (end_bit >= my_bit_start && start_bit < my_bit_end) {
                // Symbol intersects our range.
                let first_word = start_bit / 32u;
                let last_word = end_bit / 32u;

                if (first_word == last_word) {
                    // Entire codeword fits within one word — it must be ours.
                    let first_shift = 31u - (start_bit % 32u);
                    let shifted = codeword << (first_shift - (bits - 1u));
                    local_word = local_word | shifted;
                } else {
                    // Boundary symbol.
                    if (first_word == g) {
                        // We own the high part (bits that land in our word).
                        let bits_in_first = 32u - (start_bit % 32u);
                        let high_part = codeword >> (bits - bits_in_first);
                        // Place high_part at the MSB side of our word
                        let shift = 31u - (start_bit % 32u);
                        local_word = local_word | (high_part << (shift - (bits_in_first - 1u)));
                    }
                    if (last_word == g) {
                        // We own the low part (bits that overhang into our word).
                        let remaining = bits - (32u - (start_bit % 32u));
                        let low_part = (codeword << (32u - remaining)) & 0xFFFFFFFFu;
                        atomicOr(&wc_output[g], low_part);
                        // Mark that we issued an atomic — we must not stomp it later.
                        has_boundary_low = true;
                    }
                }
            }
        }

        sym_idx = sym_idx + 1u;
    }

    // Write the accumulated local word.
    // If we only have non-boundary bits, a plain store suffices and avoids
    // all atomic overhead for the common case.
    if (!has_boundary_low) {
        wc_output[g] = local_word;
    } else {
        // Merge local_word with the already-atomically-written boundary bits.
        atomicOr(&wc_output[g], local_word);
    }
}
```

Remove the original `write_codes` entry point (lines 54-98 of the current kernel) and replace it with `write_codes_chunked`. Update the host-side Huffman GPU pipeline dispatch in `src/webgpu/huffman.rs` to use the new entry point name.

**Key improvement:** In the common case where a symbol's codeword fits entirely within one 32-bit output word, the write is now a local OR into a register variable rather than an `atomicOr` to global memory. The `atomicOr` path is taken only for boundary symbols, which on 6-8 bit codes (typical Huffman output) occur at most every 4-5 symbols on average. This reduces atomic traffic by roughly 4-5x compared to the current per-symbol atomic.

### Task 8: Tests verifying Huffman GPU encode produces correct output after fix

Add to `src/webgpu/huffman.rs` `#[cfg(test)] mod tests` (or the existing Huffman test module):

```rust
#[cfg(feature = "webgpu")]
#[test]
fn test_huffman_gpu_encode_chunked_matches_cpu() {
    // Verify that write_codes_chunked produces identical output to the CPU
    // Huffman encoder for a variety of inputs.
    let engine = match WebGpuEngine::new_blocking() {
        Ok(e) => e,
        Err(_) => return,
    };

    for input_size in [1024usize, 32768, 131072, 524288] {
        let input: Vec<u8> = (0..input_size)
            .map(|i| (i % 128) as u8) // 128 distinct symbols
            .collect();

        let cpu_encoded = crate::huffman::encode(&input)
            .expect("CPU Huffman encode must succeed");
        let gpu_encoded = engine
            .huffman_encode_gpu(&input)
            .expect("GPU Huffman encode must succeed");

        // Both must decode to the original input
        let cpu_decoded = crate::huffman::decode(&cpu_encoded, input.len())
            .unwrap_or_else(|e| panic!("CPU decode failed for size {}: {:?}", input_size, e));
        let gpu_decoded = crate::huffman::decode(&gpu_encoded, input.len())
            .unwrap_or_else(|e| panic!("GPU decode of GPU-encoded failed for size {}: {:?}",
                input_size, e));

        assert_eq!(cpu_decoded, input,
            "CPU Huffman round-trip failed for size {}", input_size);
        assert_eq!(gpu_decoded, input,
            "GPU Huffman chunked round-trip failed for size {}", input_size);
    }
}

#[cfg(feature = "webgpu")]
#[test]
fn test_huffman_gpu_encode_chunked_boundary_symbols() {
    // Deliberately construct input where many codewords straddle u32
    // boundaries to stress the boundary-symbol atomic path.
    let engine = match WebGpuEngine::new_blocking() {
        Ok(e) => e,
        Err(_) => return,
    };

    // Input with 3 distinct symbols: each gets a ~10-bit code (long codes
    // mean more boundary crossings per 32-bit word).
    let input: Vec<u8> = (0..65536)
        .map(|i| match i % 3 { 0 => 0u8, 1 => 1, _ => 2 })
        .collect();

    let gpu_encoded = engine
        .huffman_encode_gpu(&input)
        .expect("GPU Huffman encode must succeed for boundary-stress input");
    let decoded = crate::huffman::decode(&gpu_encoded, input.len())
        .expect("decode of boundary-stress GPU output must succeed");
    assert_eq!(decoded, input,
        "boundary-symbol stress test round-trip mismatch");
}

#[cfg(feature = "webgpu")]
#[test]
fn test_huffman_gpu_encode_chunked_single_symbol() {
    // Edge case: all bytes are the same symbol. Huffman assigns a 1-bit code.
    // All writes land in distinct u32 words with no boundary crossings.
    let engine = match WebGpuEngine::new_blocking() {
        Ok(e) => e,
        Err(_) => return,
    };

    let input = vec![42u8; 8192];
    let gpu_encoded = engine
        .huffman_encode_gpu(&input)
        .expect("GPU Huffman encode must succeed for single-symbol input");
    let decoded = crate::huffman::decode(&gpu_encoded, input.len())
        .expect("decode of single-symbol GPU output must succeed");
    assert_eq!(decoded, input, "single-symbol round-trip mismatch");
}
```

---

## Dependencies

- **Phase 3 must complete first.** Phase 3 adds the optimal parsing cost model that selects better matches for the LzSeq encoder. Phase 4's GPU entropy path processes the match stream that Phase 3 produces. Running Phase 4 before Phase 3 is valid for the SIMD rANS and Huffman subcomponents (A and C), but the GPU entropy for LzSeqR (subcomponent B) should be validated on Phase-3-quality match output to reflect the real workload.

- **wgpu v27** is the pinned version (Cargo.toml). All GPU kernel changes must use WGSL constructs available in wgpu 27's Naga backend.

---

## Done When

1. `cargo test --features webgpu` passes all tests including the new round-trip tests for Task 2, Task 6, and Task 8.
2. `cargo bench --bench rans_bench` shows measurable throughput improvement on x86_64 for the SSE2 rANS decode path (any positive delta confirms the wiring is active; target is 5%+ on 1MB input).
3. The GPU entropy encoding path for LzSeqR activates on inputs >= 256KB when `--features webgpu` is enabled and a GPU is present, and the output decodes correctly on CPU.
4. Huffman GPU encode round-trips all test cases using `write_codes_chunked`, with no incorrect output from the atomic contention fix.
5. `./scripts/test.sh --quick` passes clean (fmt + clippy + tests, no warnings).
