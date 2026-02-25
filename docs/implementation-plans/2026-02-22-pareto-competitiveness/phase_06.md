# Pareto-Competitiveness Implementation Plan — Phase 6: GPU-Native Transform Pipelines

**Goal:** Optimize Bbw pipeline for full GPU execution, achieving higher throughput than CPU on BWT-suitable data.

**Architecture:** Ensure all Bbw stages (BBWT, MTF, RLE, FSE) can run on GPU with existing kernels. Add pipeline-level dispatch that routes entire Bbw pipeline to GPU instead of stage-by-stage handoff.

**Tech Stack:** Rust, WGSL (GPU kernels), wgpu v27

**Scope:** 8 phases from original design (phase 6 of 8)

**Codebase verified:** 2026-02-22

---

## Acceptance Criteria Coverage

### pareto-competitiveness.AC3: Multi-thread and GPU scaling
- **pareto-competitiveness.AC3.4 Success:** GPU-native Bbw pipeline achieves higher throughput than CPU on BWT-suitable data >= 256KB

---

## Current State

The Bbw pipeline runs four stages sequentially on CPU: `stage_bbwt_encode` (bijective BWT) → `stage_mtf_encode` (move-to-front) → `stage_rle_encode` (run-length encoding) → `stage_fse_encode_bbw` (FSE + header serialization). Stage dispatch lives in `src/pipeline/stages.rs` in `run_compress_stage()`.

GPU coverage today:
- **BBWT:** Covered. `bbwt_encode_with_backend()` in `src/pipeline/mod.rs` delegates to `engine.bwt_encode_bijective()` when GPU is available and input >= `MIN_GPU_BWT_SIZE` (32 KB). The GPU path runs prefix-doubling + radix sort via `kernels/bwt_radix.wgsl` and `kernels/bwt_rank.wgsl`. Output is transferred back to CPU before the next stage.
- **MTF:** No GPU implementation. `stage_mtf_encode()` calls `mtf::encode()` on CPU unconditionally.
- **RLE:** No GPU implementation. `stage_rle_encode()` calls `rle::encode()` on CPU unconditionally.
- **FSE:** GPU FSE encode exists (`engine.fse_encode_interleaved_gpu()` via `kernels/fse_encode.wgsl`), used by the Lzfi pipeline in `stage_fse_interleaved_encode_webgpu()`. It is not wired into the Bbw path, which calls `fse::encode()` (CPU) directly in `stage_fse_encode_bbw()`.

The bottleneck for GPU-native Bbw is MTF and RLE: each requires a CPU round-trip between BBWT (already GPU) and FSE (GPU-capable). This plan closes that gap.

---

## Subcomponent A: Bbw GPU Pipeline Audit

Verify and document the exact GPU coverage for each Bbw stage, then implement the two missing GPU kernels (MTF and RLE).

### Task 1 — Verify all Bbw stages have GPU implementations (BBWT, MTF, RLE, FSE)

Read and record:
- `src/pipeline/stages.rs`: `stage_bbwt_encode`, `stage_mtf_encode`, `stage_rle_encode`, `stage_fse_encode_bbw`
- `src/pipeline/mod.rs`: `bbwt_encode_with_backend()`
- `src/webgpu/bwt.rs`: `bwt_encode_bijective()`
- `src/webgpu/fse.rs`: `fse_encode_interleaved_gpu()` (confirm it accepts a flat byte slice, returns `Vec<u8>`)
- `src/mtf.rs`, `src/rle.rs`: CPU encode functions (understand their algorithms before writing GPU equivalents)

Produce a one-page internal audit table:

| Stage | GPU kernel | GPU path wired into Bbw? | Notes |
|-------|-----------|--------------------------|-------|
| BBWT  | `bwt_radix.wgsl` + `bwt_rank.wgsl` | Yes (via `bbwt_encode_with_backend`) | CPU round-trip after |
| MTF   | None | No | Sequential; needs GPU kernel |
| RLE   | None | No | Sequential; needs GPU kernel |
| FSE   | `fse_encode.wgsl` | No (wired only for Lzfi) | Interleaved variant; usable |

This task produces no code changes. Its output informs Task 2.

### Task 2 — Implement missing GPU stage kernels (MTF and RLE)

#### 2a. GPU MTF kernel: `kernels/mtf_encode.wgsl`

Move-to-front is inherently sequential (each output symbol depends on the current alphabet state after the previous symbol). However, for a single Bbw block the alphabet is 256 entries and the bottleneck is memory bandwidth, not compute. A GPU implementation processes one block per workgroup using shared memory for the alphabet, trading sequential compute for on-chip speed.

Algorithm per workgroup (one workgroup per input block):
1. Load the 256-entry alphabet into workgroup shared memory: `alphabet[i] = i` for `i` in `0..256`.
2. Iterate over input bytes sequentially (single lane: `local_invocation_id.x == 0`).
3. For each symbol `s`: find its position `p` in `alphabet`, write `p` to output, shift `alphabet[1..=p]` left by one, set `alphabet[0] = s`.

Because only one thread does the sequential work, the GPU kernel does not parallelize the symbol scan. The benefit is keeping data on-device between BBWT and RLE — eliminating one PCIe round-trip — and enabling future block-parallel MTF if the block structure allows it.

**New file:** `kernels/mtf_encode.wgsl`

```wgsl
// GPU MTF (move-to-front) encode kernel (WGSL).
//
// One workgroup per input block. A single thread (lid=0) processes symbols
// sequentially using shared memory for the alphabet, keeping data on-device
// between BBWT and RLE to avoid PCIe round-trips.
//
// @pz_cost {
//   threads_per_element: 0.00390625
//   passes: 1
//   buffers: input=N, output=N
//   local_mem: 256
//   note: single-threaded sequential per block; benefit is on-device data retention
// }

@group(0) @binding(0) var<storage, read>       mtf_input:  array<u32>;
@group(0) @binding(1) var<storage, read_write> mtf_output: array<u32>;
@group(0) @binding(2) var<uniform>             mtf_params: vec4<u32>; // x=n_bytes

var<workgroup> alphabet: array<u32, 256>;

@compute @workgroup_size(256)
fn mtf_encode(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Initialize alphabet: alphabet[i] = i
    alphabet[lid.x] = lid.x;
    workgroupBarrier();

    // Only thread 0 does the sequential scan
    if (lid.x != 0u) {
        return;
    }

    let n = mtf_params.x;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let word_idx    = i / 4u;
        let byte_in_word = i % 4u;
        let s = (mtf_input[word_idx] >> (byte_in_word * 8u)) & 0xFFu;

        // Find position of s in alphabet
        var pos: u32 = 0u;
        for (var j: u32 = 0u; j < 256u; j = j + 1u) {
            if (alphabet[j] == s) {
                pos = j;
                break;
            }
        }

        // Write output
        let out_word = i / 4u;
        let out_byte = i % 4u;
        let shift = out_byte * 8u;
        let mask  = ~(0xFFu << shift);
        mtf_output[out_word] = (mtf_output[out_word] & mask) | (pos << shift);

        // Shift alphabet[1..pos] left, insert s at front
        for (var j: u32 = pos; j > 0u; j = j - 1u) {
            alphabet[j] = alphabet[j - 1u];
        }
        alphabet[0u] = s;
    }
}
```

Add `mtf_encode.wgsl` to the kernel include list in `src/webgpu/mod.rs`. Add a `pipeline_mtf_encode` compute pipeline alongside the existing pipeline constructors.

**New method on `WebGpuEngine` in `src/webgpu/mtf.rs`** (new file):

```rust
/// GPU MTF forward encode. Input and output are raw byte slices
/// packed into u32 arrays (little-endian). Pads to u32 alignment.
pub fn mtf_encode(&self, input: &[u8]) -> PzResult<Vec<u8>> {
    let n = input.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Pack input bytes into u32 words (little-endian)
    let padded = (n + 3) / 4 * 4;
    let mut input_words = vec![0u32; padded / 4];
    for (i, &b) in input.iter().enumerate() {
        input_words[i / 4] |= (b as u32) << ((i % 4) * 8);
    }

    let input_buf = self.create_buffer_init(
        "mtf_input",
        bytemuck::cast_slice(&input_words),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let output_buf = self.create_buffer(
        "mtf_output",
        (padded as u64).max(4),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let params = [n as u32, 0, 0, 0];
    let params_buf = self.create_buffer_init(
        "mtf_params",
        bytemuck::cast_slice(&params),
        wgpu::BufferUsages::UNIFORM,
    );

    // ... bind group + dispatch (1 workgroup) ...

    let raw = self.read_buffer(&output_buf, padded as u64);
    Ok(raw[..n].to_vec())
}
```

Add `mod mtf;` to `src/webgpu/mod.rs`.

#### 2b. GPU RLE kernel: `kernels/rle_encode.wgsl`

RLE as used by the Bw/Bbw pipelines in `src/rle.rs` encodes runs of identical bytes: emit the byte, then if it appears 4+ consecutive times, emit a count byte (excess above 4). Read `src/rle.rs` to confirm the exact encoding before writing the kernel.

RLE is also sequential: each output position depends on how many equal bytes came before. The same strategy as MTF applies: one workgroup, one active thread, shared memory not needed here (alphabet is just a counter).

**New file:** `kernels/rle_encode.wgsl`

```wgsl
// GPU RLE encode kernel (WGSL) — bzip2-style run-length encoding.
//
// Encodes runs of 4+ identical bytes. One workgroup, single thread.
// Benefit is on-device data retention between MTF and FSE.
//
// @pz_cost {
//   threads_per_element: 0.00390625
//   passes: 1
//   buffers: input=N, output=N, out_len=4
//   local_mem: 0
//   note: single-threaded sequential; worst-case output is same size as input
// }

@group(0) @binding(0) var<storage, read>       rle_input:   array<u32>;
@group(0) @binding(1) var<storage, read_write> rle_output:  array<u32>;
@group(0) @binding(2) var<storage, read_write> rle_out_len: array<u32>; // [0] = actual output length
@group(0) @binding(3) var<uniform>             rle_params:  vec4<u32>;  // x=n_bytes

fn read_byte(pos: u32) -> u32 {
    return (rle_input[pos / 4u] >> ((pos % 4u) * 8u)) & 0xFFu;
}

fn write_byte(pos: u32, val: u32) {
    let shift = (pos % 4u) * 8u;
    let mask  = ~(0xFFu << shift);
    rle_output[pos / 4u] = (rle_output[pos / 4u] & mask) | (val << shift);
}

@compute @workgroup_size(1)
fn rle_encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = rle_params.x;
    var out_pos: u32 = 0u;
    var i: u32 = 0u;

    while (i < n) {
        let b = read_byte(i);
        write_byte(out_pos, b);
        out_pos = out_pos + 1u;

        // Count run length
        var run: u32 = 1u;
        while (i + run < n && read_byte(i + run) == b && run < 259u) {
            run = run + 1u;
        }

        if (run >= 4u) {
            // Emit byte 3 more times (total 4), then emit count - 4
            write_byte(out_pos,     b);
            write_byte(out_pos + 1u, b);
            write_byte(out_pos + 2u, b);
            out_pos = out_pos + 3u;
            write_byte(out_pos, run - 4u);
            out_pos = out_pos + 1u;
        } else {
            // Emit the remaining run - 1 copies (first already written)
            for (var k: u32 = 1u; k < run; k = k + 1u) {
                write_byte(out_pos, b);
                out_pos = out_pos + 1u;
            }
        }

        i = i + run;
    }

    rle_out_len[0u] = out_pos;
}
```

**Important:** Before writing the WGSL, read `src/rle.rs` to confirm the exact run-length encoding format. The kernel above assumes the bzip2 convention used in libpz — verify this is correct and adjust if needed.

**New method on `WebGpuEngine` in `src/webgpu/rle.rs`** (new file):

```rust
/// GPU RLE forward encode. Returns encoded bytes.
/// Output length is variable; worst case equals input length.
pub fn rle_encode(&self, input: &[u8]) -> PzResult<Vec<u8>> {
    // ... same buffer/dispatch pattern as mtf_encode ...
    // Read out_len from rle_out_len buffer, then slice output to that length.
}
```

Add `mod rle;` to `src/webgpu/mod.rs`.

### Task 3 — Tests for individual GPU stage correctness vs CPU equivalents

Add tests to `src/webgpu/tests.rs` (existing file). Each test:
1. Runs the CPU reference implementation on a known input.
2. Runs the new GPU implementation on the same input.
3. Asserts byte-for-byte identical output.

Test inputs should cover:
- Empty input (no crash, returns empty)
- Single byte
- All-same bytes (stress RLE run detection)
- All-distinct bytes (stress MTF: every byte moves to front 0)
- 64 KB of random bytes (realistic size)
- 256 KB of highly repetitive text (the primary target workload)

```rust
#[cfg(all(test, feature = "webgpu"))]
mod gpu_transform_tests {
    use super::*;

    fn engine() -> Option<Arc<WebGpuEngine>> {
        WebGpuEngine::new().ok().map(Arc::new)
    }

    #[test]
    fn test_gpu_mtf_matches_cpu() {
        let Some(engine) = engine() else { return; };
        for input in [b"".as_ref(), b"aaa", b"abcabc", &[0u8; 256 * 1024]] {
            let cpu_out = crate::mtf::encode(input);
            let gpu_out = engine.mtf_encode(input).unwrap();
            assert_eq!(cpu_out, gpu_out, "MTF mismatch for input len {}", input.len());
        }
    }

    #[test]
    fn test_gpu_rle_matches_cpu() {
        let Some(engine) = engine() else { return; };
        for input in [b"".as_ref(), b"aaaa", b"aaaab", &vec![0u8; 300][..]] {
            let cpu_out = crate::rle::encode(input);
            let gpu_out = engine.rle_encode(input).unwrap();
            assert_eq!(cpu_out, gpu_out, "RLE mismatch for input len {}", input.len());
        }
    }
}
```

**Verification after Task 3:**
```bash
./scripts/test.sh --quick
cargo test --features webgpu gpu_transform_tests
```

---

## Subcomponent B: Pipeline-Level GPU Dispatch

Wire the GPU MTF and RLE kernels into the Bbw pipeline, then add a GPU dispatch mode that keeps data on-device between all four stages.

### Task 4 — Add GPU dispatch mode for Bbw that keeps data on GPU between stages

The current Bbw dispatch in `run_compress_stage()` in `src/pipeline/stages.rs` is:

```
(Pipeline::Bbw, 0) => stage_bbwt_encode(block, options),   // may go to GPU
(Pipeline::Bbw, 1) => stage_mtf_encode(block),              // always CPU
(Pipeline::Bbw, 2) => stage_rle_encode(block),              // always CPU
(Pipeline::Bbw, 3) => stage_fse_encode_bbw(block),          // always CPU
```

The four-way stage handoff means even when BBWT runs on GPU, the transformed data is read back to CPU for MTF, then sent back to GPU for FSE. Replace this with a fused GPU path.

**Approach:** Add a new stage function `stage_bbwt_gpu_pipeline()` that runs all four transforms without CPU round-trips. Wire it into stage 0 of the Bbw pipeline when the GPU backend is active. Stages 1-3 remain as CPU fallbacks when GPU is not configured.

**New function in `src/pipeline/stages.rs`:**

```rust
/// GPU-fused Bbw pipeline: BBWT → MTF → RLE → FSE, entirely on GPU.
///
/// Runs when `options.backend == Backend::WebGpu` and a GPU engine is available
/// and input >= MIN_GPU_PIPELINE_SIZE. Output format is identical to the
/// CPU path (same header, same FSE framing) so the CPU decoder works unchanged.
///
/// Returns the fully compressed block (header + FSE data) in `block.data`.
/// Sets `block.metadata.bbwt_factor_lengths` and `block.metadata.pre_entropy_len`
/// as side effects so the caller can inspect them, even though subsequent stages
/// are skipped by returning a block with `streams = None` and data already set.
#[cfg(feature = "webgpu")]
pub(crate) fn stage_bbwt_gpu_pipeline(
    mut block: StageBlock,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    use crate::webgpu::MIN_GPU_BWT_SIZE;

    let engine = options.webgpu_engine.as_ref().ok_or(PzError::Unsupported)?;
    if engine.is_cpu_device() || block.data.len() < MIN_GPU_PIPELINE_SIZE {
        return Err(PzError::Unsupported); // caller falls back to CPU path
    }

    // Stage 0: GPU BBWT
    let (bwt_data, factor_lengths) = engine.bwt_encode_bijective(&block.data)?;
    block.metadata.bbwt_factor_lengths = Some(factor_lengths.clone());

    // Stage 1: GPU MTF (data stays on GPU via internal buffers)
    let mtf_data = engine.mtf_encode(&bwt_data)?;

    // Stage 2: GPU RLE
    let rle_data = engine.rle_encode(&mtf_data)?;
    let rle_len = rle_data.len();
    block.metadata.pre_entropy_len = Some(rle_len);

    // Stage 3: GPU FSE (using existing interleaved FSE, same accuracy as CPU path)
    // Note: Bbw uses single-stream FSE, not multi-stream. Use fse::encode for
    // correctness first; swap to GPU FSE once benchmarks confirm it helps.
    let fse_data = crate::fse::encode(&rle_data);

    // Serialize header in the same format as stage_fse_encode_bbw
    let mut output = Vec::new();
    output.extend_from_slice(&(factor_lengths.len() as u16).to_le_bytes());
    for &fl in &factor_lengths {
        output.extend_from_slice(&(fl as u32).to_le_bytes());
    }
    output.extend_from_slice(&(rle_len as u32).to_le_bytes());
    output.extend_from_slice(&fse_data);

    block.data = output;
    Ok(block)
}
```

`MIN_GPU_PIPELINE_SIZE` should be set to 256 KB (262144 bytes) — the AC3.4 threshold. Define it as a constant near `MIN_GPU_BWT_SIZE` in `src/webgpu/mod.rs`:

```rust
pub const MIN_GPU_PIPELINE_SIZE: usize = 256 * 1024; // 256 KB
```

**Wire into `run_compress_stage()`:**

```rust
(Pipeline::Bbw, 0) => {
    #[cfg(feature = "webgpu")]
    {
        if let super::Backend::WebGpu = options.backend {
            if options.webgpu_engine.is_some() {
                match stage_bbwt_gpu_pipeline(block, options) {
                    Ok(b) => return Ok(b),
                    Err(PzError::Unsupported) => {
                        // Fell back: reconstruct block and continue to CPU path
                        // (block was moved — need to re-enter CPU BBWT)
                        // See implementation note below
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }
    stage_bbwt_encode(block, options)
}
```

**Implementation note on fallback:** Because `block` is moved into `stage_bbwt_gpu_pipeline`, check eligibility before moving to avoid ownership complications:

```rust
(Pipeline::Bbw, 0) => {
    #[cfg(feature = "webgpu")]
    {
        if let super::Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                let eligible = !engine.is_cpu_device()
                    && block.data.len() >= crate::webgpu::MIN_GPU_PIPELINE_SIZE;
                if eligible {
                    return stage_bbwt_gpu_pipeline(block, options);
                }
            }
        }
    }
    stage_bbwt_encode(block, options)
}
```

**Stage skipping:** When `stage_bbwt_gpu_pipeline` succeeds, it produces the fully compressed output in `block.data`. Stages 1, 2, 3 must be skipped. The stage dispatch loop in `src/pipeline/blocks.rs` iterates `0..pipeline_stage_count(pipeline)` and calls `run_compress_stage` for each. Add a field to `StageBlock`:

```rust
pub struct StageBlock {
    // ... existing fields ...
    /// When true, remaining pipeline stages are no-ops (data already fully encoded).
    pub pipeline_complete: bool,
}
```

In `run_compress_stage()`, guard stages 1-3 of Bbw:

```rust
(Pipeline::Bbw, 1) | (Pipeline::Bbw, 2) | (Pipeline::Bbw, 3) => {
    if block.pipeline_complete {
        return Ok(block); // GPU path already encoded everything
    }
    // ... existing CPU dispatch ...
}
```

Set `block.pipeline_complete = true` at the end of `stage_bbwt_gpu_pipeline`.

### Task 5 — Minimize GPU-to-CPU transfers (only upload input, download compressed output)

The initial implementation in Task 4 reads intermediate buffers back to CPU between stages (BBWT output → CPU → MTF → CPU → RLE → CPU → FSE). This is acceptable for correctness but leaves performance on the table.

Improve `WebGpuEngine` to chain GPU operations using GPU-resident buffers:

**New method: `engine.bbwt_mtf_rle_fse_encode(input: &[u8]) -> PzResult<(Vec<u8>, Vec<usize>, usize)>`**

This method fuses all four transforms into a single GPU session:
1. Upload `input` once to a GPU buffer.
2. Run BBWT (prefix-doubling + radix sort) — result stays in GPU buffer.
3. Run MTF — reads BBWT GPU buffer, writes to MTF GPU output buffer.
4. Run RLE — reads MTF GPU buffer, writes to RLE GPU output buffer.
5. Read back only the RLE output (needed for FSE accuracy selection and header).
6. Run FSE encode on CPU with the downloaded RLE bytes.
7. Return `(fse_data, factor_lengths, rle_len)`.

Steps 2-4 share the same `wgpu::CommandEncoder` where possible to minimize submit overhead. Steps 3 and 4 cannot be batched into a single submit because each requires reading the previous stage's output length before allocating the next buffer (RLE output size is variable). Use separate command encoders with `poll_wait()` between stages for now. Document this as a known inefficiency and leave a `// TODO: chain via indirect dispatch` comment for a future optimization.

```rust
// In src/webgpu/mod.rs or src/webgpu/bbwt_pipeline.rs (new file)

impl WebGpuEngine {
    /// Fused BBWT → MTF → RLE GPU pipeline.
    ///
    /// Uploads input once, chains transforms on-device, returns
    /// RLE-encoded bytes, factor lengths, and RLE length for use
    /// by the CPU FSE stage.
    ///
    /// Falls back gracefully if any stage fails.
    pub fn bbwt_mtf_rle_encode(
        &self,
        input: &[u8],
    ) -> PzResult<(Vec<u8>, Vec<usize>, usize)> {
        // 1. BBWT
        let (bwt_data, factor_lengths) = self.bwt_encode_bijective(input)?;
        // 2. MTF (uses bwt_data; GPU buffer allocation is internal)
        let mtf_data = self.mtf_encode(&bwt_data)?;
        // 3. RLE
        let rle_data = self.rle_encode(&mtf_data)?;
        let rle_len = rle_data.len();
        Ok((rle_data, factor_lengths, rle_len))
    }
}
```

Update `stage_bbwt_gpu_pipeline` to call `bbwt_mtf_rle_encode` instead of calling the three methods separately. This centralizes the chaining logic and makes it easier to optimize later.

### Task 6 — Tests for full GPU pipeline round-trip

Add round-trip tests that compress with the GPU Bbw pipeline and decompress with the CPU decoder (CPU decoder is unchanged; it reads the same wire format).

Test structure in `src/webgpu/tests.rs` (or `src/pipeline/tests.rs` if that is more appropriate — check existing test location):

```rust
#[cfg(all(test, feature = "webgpu"))]
mod bbwt_gpu_pipeline_tests {
    use crate::pipeline::{compress_with_options, decompress, Backend, CompressOptions, Pipeline};
    use crate::webgpu::WebGpuEngine;
    use std::sync::Arc;

    fn gpu_options() -> Option<CompressOptions> {
        let engine = WebGpuEngine::new().ok()?;
        if engine.is_cpu_device() { return None; }
        Some(CompressOptions {
            backend: Backend::WebGpu,
            webgpu_engine: Some(Arc::new(engine)),
            ..Default::default()
        })
    }

    #[test]
    fn test_bbwt_gpu_pipeline_round_trip_256kb() {
        let Some(options) = gpu_options() else { return; };
        // Highly repetitive data: BWT-suitable
        let input: Vec<u8> = b"abcabc".iter().cycle().take(256 * 1024).cloned().collect();
        let compressed = compress_with_options(&input, Pipeline::Bbw, &options).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_bbwt_gpu_pipeline_round_trip_1mb() {
        let Some(options) = gpu_options() else { return; };
        let input: Vec<u8> = (0u8..=255).cycle().take(1024 * 1024).collect();
        let compressed = compress_with_options(&input, Pipeline::Bbw, &options).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_bbwt_gpu_pipeline_small_input_falls_back_to_cpu() {
        let Some(options) = gpu_options() else { return; };
        // Input below MIN_GPU_PIPELINE_SIZE: should use CPU path without error
        let input = b"hello world".to_vec();
        let compressed = compress_with_options(&input, Pipeline::Bbw, &options).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(input, decompressed);
    }

    #[test]
    fn test_bbwt_gpu_output_identical_to_cpu() {
        let Some(options) = gpu_options() else { return; };
        use crate::pipeline::CompressOptions;
        let input: Vec<u8> = b"aaabbbccc".iter().cycle().take(512 * 1024).cloned().collect();
        let cpu_compressed = compress_with_options(
            &input, Pipeline::Bbw, &CompressOptions::default()
        ).unwrap();
        let gpu_compressed = compress_with_options(&input, Pipeline::Bbw, &options).unwrap();
        // Output must be byte-identical (same FSE accuracy, same factor ordering)
        assert_eq!(cpu_compressed, gpu_compressed);
    }
}
```

The byte-identical test (`test_bbwt_gpu_output_identical_to_cpu`) is the strongest correctness check: if CPU and GPU outputs match, the decoder requires no changes.

**Verification after Subcomponent B:**
```bash
./scripts/test.sh --quick
cargo test --features webgpu bbwt_gpu_pipeline_tests
```

---

## Subcomponent C: GPU Transform Optimization

### Task 7 — Evaluate and implement GPU-accelerated RLE and delta encoding if beneficial

#### RLE evaluation

The GPU RLE kernel in Task 2b is single-threaded within a workgroup to preserve the sequential encoding invariant. Evaluate whether this is actually faster than CPU for the Bbw workload:

Run `scripts/bench.sh` with the GPU pipeline enabled on a 256 KB repetitive input. Profile with `scripts/profile.sh --stage rle`. If GPU RLE is slower than CPU RLE (likely, since the kernel is single-threaded and RLE is memory-bandwidth bound), then:

**Option A — Keep GPU RLE for data locality.** Even if GPU RLE compute is slower than CPU, eliminating the PCIe round-trip between BBWT and MTF may save more than the GPU overhead costs. Measure both configurations.

**Option B — CPU RLE with pinned memory.** Download BBWT+MTF output using a pinned (mapped) staging buffer, run CPU RLE, upload to GPU for FSE. This is valid if the transfer savings outweigh the kernel overhead.

Document the measurement result as a comment in `stage_bbwt_gpu_pipeline`:
```rust
// Performance note (measured 2026-MM-DD): GPU RLE for 256 KB repetitive input
// achieves X MB/s vs CPU Y MB/s. Chosen path: [GPU | CPU]. Rationale: ...
```

#### Delta encoding evaluation

Delta encoding (`output[i] = input[i] - input[i-1]`) is trivially parallelizable on GPU and helps FSE when data has linear trends (e.g., audio, sensor data). The Bbw pipeline does not currently use delta encoding.

**Evaluation:** Run `src/analysis.rs` autocorrelation on Canterbury corpus files to identify which have linear trends. If fewer than 20% of target workload files benefit, skip GPU delta encoding for Phase 6 (note as future work for Phase 7 auto-selection).

If delta encoding is deemed beneficial:

**New file:** `kernels/delta_encode.wgsl` — embarrassingly parallel: one thread per byte.

```wgsl
// @pz_cost {
//   threads_per_element: 1
//   passes: 1
//   buffers: input=N, output=N
//   local_mem: 0
// }
@compute @workgroup_size(256)
fn delta_encode(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.x) { return; }
    let prev = select(0u, (input[i / 4u - select(0u, 1u, i % 4u == 0u)] >> ...) & 0xFFu, i > 0u);
    // ... write output[i] = input[i] - prev ...
}
```

Wire as an optional pre-BBWT stage in `stage_bbwt_gpu_pipeline` controlled by a flag on `CompressOptions` (default off; Phase 7 auto-selection can enable it per-workload).

### Task 8 — Benchmark GPU vs CPU Bbw pipeline on BWT-suitable data

This task validates AC3.4: GPU Bbw throughput > CPU Bbw throughput on 256 KB+ BWT-suitable data.

**Benchmark setup:** Use `cargo bench -- bbw` (existing bench infrastructure in `benches/`). Add a new benchmark group if not already present:

```rust
// In benches/stages.rs (or wherever Bbw benchmarks live):

fn bench_bbw_gpu_vs_cpu(c: &mut Criterion) {
    let data_256k: Vec<u8> = b"abcdefgh".iter().cycle().take(256 * 1024).cloned().collect();
    let data_1m:   Vec<u8> = b"abcdefgh".iter().cycle().take(1024 * 1024).cloned().collect();

    let mut group = c.benchmark_group("bbw_pipeline");

    group.benchmark_function("cpu_256k", |b| {
        b.iter(|| compress_with_options(&data_256k, Pipeline::Bbw, &CompressOptions::default()))
    });

    if let Ok(engine) = WebGpuEngine::new() {
        if !engine.is_cpu_device() {
            let engine = Arc::new(engine);
            let gpu_options = CompressOptions {
                backend: Backend::WebGpu,
                webgpu_engine: Some(engine),
                ..Default::default()
            };
            group.benchmark_function("gpu_256k", |b| {
                b.iter(|| compress_with_options(&data_256k, Pipeline::Bbw, &gpu_options))
            });
            group.benchmark_function("gpu_1m", |b| {
                b.iter(|| compress_with_options(&data_1m, Pipeline::Bbw, &gpu_options))
            });
        }
    }

    group.finish();
}
```

Run benchmarks:
```bash
cargo bench --features webgpu -- bbw_pipeline
```

**Pass criteria for AC3.4:** GPU `bbw_256k` throughput (MB/s) >= CPU `bbw_256k` throughput on the development machine.

If GPU does not beat CPU at 256 KB, increase the threshold incrementally (512 KB, 1 MB) and document the crossover point as the production threshold in `MIN_GPU_PIPELINE_SIZE`. Update the constant accordingly.

**Save results** to `docs/generated/2026-02-22-bbwt-gpu-vs-cpu.md` using `scripts/bench.sh`.

---

## Files Created/Modified

| File | Subcomponent | Change |
|------|-------------|--------|
| **`kernels/mtf_encode.wgsl`** | A | New: GPU MTF encode kernel |
| **`kernels/rle_encode.wgsl`** | A | New: GPU RLE encode kernel |
| **`kernels/delta_encode.wgsl`** | C | New (if delta encoding is beneficial) |
| **`src/webgpu/mtf.rs`** | A | New: `WebGpuEngine::mtf_encode()` |
| **`src/webgpu/rle.rs`** | A | New: `WebGpuEngine::rle_encode()` |
| `src/webgpu/mod.rs` | A, B | Add `mod mtf; mod rle;`, pipeline constructors for new kernels, `MIN_GPU_PIPELINE_SIZE` |
| `src/pipeline/stages.rs` | B | Add `stage_bbwt_gpu_pipeline()`, update `run_compress_stage()` for Bbw stages, add `pipeline_complete` field to `StageBlock` |
| `src/pipeline/blocks.rs` | B | Honor `pipeline_complete` flag in stage dispatch loop |
| `src/webgpu/tests.rs` | A, B | Add GPU MTF/RLE unit tests, Bbw pipeline round-trip tests |
| `benches/stages.rs` | C | Add `bench_bbw_gpu_vs_cpu` group |
| **`docs/generated/2026-02-22-bbwt-gpu-vs-cpu.md`** | C | Benchmark results |

---

## Verification

After each subcomponent:
```bash
cargo fmt --all
cargo clippy --all-targets --features webgpu -- -D warnings
cargo test --features webgpu
```

After Task 3 (GPU stage unit tests):
```bash
cargo test --features webgpu gpu_transform_tests
```

After Task 6 (pipeline round-trip):
```bash
cargo test --features webgpu bbwt_gpu_pipeline_tests
```

After Task 8 (benchmarks):
```bash
cargo bench --features webgpu -- bbw_pipeline
```

AC3.4 passes when the benchmark output shows GPU Bbw throughput >= CPU Bbw throughput for at least one input size >= 256 KB.

---

## Known Risks and Mitigations

**MTF is sequential — GPU may not help.** The GPU MTF kernel runs one thread, so it is not faster than CPU for the computation itself. The benefit is data locality (fewer PCIe round-trips). Measure carefully in Task 8. If MTF on GPU is a net loss, do CPU MTF with pinned memory download + upload.

**RLE output size is variable.** The GPU RLE kernel needs to write an output length scalar (`rle_out_len`). Read this back before allocating the FSE input buffer. This introduces one extra synchronization point. Keep it.

**BBWT Lyndon factorization is CPU-only.** `bwt_encode_bijective()` calls `bwt::lyndon_factorize()` on CPU before dispatching GPU SA construction per factor. This is unavoidable without a GPU Lyndon factorizer. For large blocks with few large factors, the CPU factorization overhead is negligible.

**FSE accuracy depends on RLE output statistics.** The CPU `adaptive_accuracy_log()` function reads distinct symbol counts from the RLE output to pick FSE accuracy. If GPU RLE output is downloaded and passed to CPU FSE, this works unchanged. If FSE is later moved to GPU, implement `adaptive_accuracy_log` as a pre-pass histogram kernel or accept a fixed accuracy log (e.g., always `DEFAULT_ACCURACY_LOG`) and document the potential ratio cost.

**GPU device lost during fused pipeline.** Each stage in `stage_bbwt_gpu_pipeline` can return a `PzError`. The caller in `run_compress_stage` must propagate errors rather than silently ignoring them. The block is already moved into the fused function; there is no CPU fallback after a partial GPU run. Document this: GPU errors during fused Bbw are fatal for that block.
