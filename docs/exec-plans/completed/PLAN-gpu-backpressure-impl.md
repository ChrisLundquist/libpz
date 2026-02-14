# Implementation Plan: GPU Kernel Cost Annotations and Runtime Scheduling

**Source plan:** `PLAN-gpu-backpressure.md` (was in exciting-cori worktree, now lost).

**Goal:** Replace the hardcoded `MAX_GPU_BATCH_SIZE = 8` with a cost-model-driven scheduler that limits GPU in-flight work based on actual kernel memory requirements and device VRAM.

**Important context for the implementing agent:**
- This plan is independent of the GPU composability refactor on `compassionate-euler`. It only touches kernel comments, a new parser module, and the engine init/batching code. No conflicts expected.
- All kernel files are embedded via `include_str!()` — the parser runs on compile-time string constants, not files at runtime.
- The `MAX_GPU_BATCH_SIZE = 8` constant lives in `src/webgpu/mod.rs`. The batched path is in `src/pipeline/parallel.rs` (`compress_parallel_gpu_batched`). The actual buffer allocations are in `src/webgpu/lz77.rs` (`submit_find_matches_lazy`).

## Problem

When running GPU workloads, we see 80-100% GPU utilization, inconsistent benchmarks, and a GPU hang at 4MB. The root cause: the scheduler has no idea how expensive a kernel dispatch is, so it fire-and-forgets up to 8 blocks with zero backpressure.

`MAX_GPU_BATCH_SIZE = 8` doesn't account for:
- Which kernel is being dispatched (hash LZ77 uses ~160 MB per 4MB block, Huffman uses ~22 MB)
- The input size (a 256KB block is fine, a 4MB block blows the GPU memory budget)
- The device's actual VRAM (we already query `global_mem_size` / `max_buffer_size` but don't use it for scheduling)

## Core Idea

Annotate each kernel with its cost model as structured comments (`@pz_cost` blocks), parse those at engine init time to derive a per-dispatch memory budget. The scheduler uses this to decide how many blocks to keep in flight.

---

## Phase 1: Annotate all kernel files with `@pz_cost` blocks

**What:** Add structured cost comments to the top of every kernel file (after existing header comments, before kernel code). These are pure comments — zero impact on compiled kernels.

**Format:**
```
// @pz_cost {
//   threads_per_element: <float>
//   passes: <int>
//   buffers: <name>=<expr>, <name>=<expr>, ...
//   local_mem: <int>
// }
```

Where buffer expressions are: bare integer literals (e.g., `131072`), `N` (input length), or `N*<int>` (e.g., `N*12`).

**There are 15 kernel files.** Find them with `ls kernels/`. The plan document listed 14 — it missed `kernels/bwt_sort.cl` (the bitonic sort kernel).

**Critical: Verify buffer formulas against actual Rust allocation code.** The actual `submit_find_matches_lazy()` in `src/webgpu/lz77.rs` allocates a staging buffer (`N*12` bytes) that some estimates don't account for. For each kernel, before writing the annotation:

1. Search for the kernel's entry point name (e.g., `BuildHashTable`, `FindMatches`, `Encode`, `bitonic_sort_step`) in the Rust code
2. Find the function that dispatches that kernel
3. Read the buffer creation calls (`Buffer::create`, `create_buffer`, `create_buffer_init`)
4. Confirm the buffer sizes match the annotation formulas
5. Count the number of `enqueue_nd_range` / `dispatch` calls to verify `passes`

**Files and annotations:**

### `kernels/lz77_hash.cl` and `kernels/lz77_hash.wgsl`
Verify in `src/opencl/lz77.rs` (`KernelVariant::HashTable`) and `src/webgpu/lz77.rs` (hash path).
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, hash_counts=131072, hash_table=8388608, output=N*12
//   local_mem: 0
// }
```

### `kernels/lz77_lazy.wgsl`
Verify in `src/webgpu/lz77.rs` (`submit_find_matches_lazy`). Uses 3 passes (build hash → find matches → resolve lazy) and allocates raw_matches + resolved buffers.
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 3
//   buffers: input=N, hash_counts=131072, hash_table=8388608, raw_matches=N*12, resolved=N*12
//   local_mem: 0
// }
```

### `kernels/lz77.cl` (per-position brute-force)
Verify in `src/opencl/lz77.rs` (`KernelVariant::PerPosition`).
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 1
//   buffers: input=N, output=N*12
//   local_mem: 0
// }
```

### `kernels/lz77_batch.cl` (batched, 1 thread per 32 positions)
Verify in `src/opencl/lz77.rs` (`KernelVariant::Batch`).
```
// @pz_cost {
//   threads_per_element: 0.03125
//   passes: 1
//   buffers: input=N, output=N*12
//   local_mem: 0
// }
```

### `kernels/lz77_topk.cl` and `kernels/lz77_topk.wgsl`
Verify in `find_topk_matches` in both backends. Output is K=4 matches per position.
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, hash_counts=131072, hash_table=8388608, output=N*48
//   local_mem: 0
// }
```

### `kernels/huffman_encode.cl` and `kernels/huffman_encode.wgsl`
Verify in `huffman_encode` / `huffman_encode_gpu_scan` in both backends. Contains ByteHistogram + ComputeBitLengths + PrefixSum + WriteCodes.
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 4
//   buffers: input=N, lut=1024, bit_lengths=N*4, output=N, block_sums=32768
//   local_mem: 2048
// }
```

### `kernels/bwt_rank.cl` and `kernels/bwt_rank.wgsl`
Verify in BWT rank-doubling code. Note: called log2(N) times per BWT encode.
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 4
//   buffers: sa=N*8, rank=N*8, diff=N*4, prefix=N*4
//   local_mem: 2048
//   note: called per doubling step (log2(N) steps)
// }
```

### `kernels/bwt_radix.cl` and `kernels/bwt_radix.wgsl`
Verify in radix sort code in BWT modules.
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 3
//   buffers: keys=N*4, histogram=N*4, sa_in=N*4, sa_out=N*4
//   local_mem: 1024
//   note: 4 radix passes per doubling step
// }
```

### `kernels/bwt_sort.cl` (bitonic sort — was missing from original plan)
Verify in `bwt_encode` in `src/opencl/bwt.rs`. Uses N/2 comparators.
```
// @pz_cost {
//   threads_per_element: 0.5
//   passes: 1
//   buffers: sa=N*4, data=N
//   local_mem: 0
//   note: O(log^2 N) dispatches for full bitonic sort
// }
```
**Verify the 0.5 threads_per_element and buffer sizes against the actual code.**

### `kernels/fse_decode.wgsl`
Verify in `fse_decode_gpu` in the WebGPU module.
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 1
//   buffers: input=N, output=N, table=4096
//   local_mem: 0
// }
```
**Verify the table size and buffer layout against the actual Rust code.**

**Test:** `cargo test` and `cargo test --features opencl` and `cargo test --features webgpu` — all pass (annotations are comments).

**Commit:** "Add @pz_cost annotations to all GPU kernel files"

---

## Phase 2: Create `src/gpu_cost.rs` — parser and cost model

**What:** New file implementing `KernelCost`, `BufferFormula`, and a parser that extracts `@pz_cost` blocks from kernel source strings.

**File:** `src/gpu_cost.rs` (new)

**Also modify:** `src/lib.rs` — add `mod gpu_cost;` between the `simd` and `opencl` module declarations. NOT `pub` — crate-internal only.

**Structs:**

```rust
/// Cost model for a single GPU kernel, parsed from @pz_cost annotations
/// embedded as structured comments in kernel source files.
#[derive(Debug, Clone)]
pub(crate) struct KernelCost {
    /// GPU threads launched per input byte (e.g., 1.0 for per-position kernels).
    pub threads_per_element: f64,
    /// Number of sequential kernel dispatches.
    pub passes: usize,
    /// Buffer allocations: (name, formula for size in bytes).
    pub buffers: Vec<(String, BufferFormula)>,
    /// Workgroup-local memory in bytes.
    pub local_mem: usize,
}

/// Expression for buffer size as a function of input length N.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BufferFormula {
    /// Fixed size in bytes (e.g., `131072`).
    Fixed(usize),
    /// Linear in N: `scale * N + offset` (e.g., `N*12` → scale=12, offset=0).
    Linear { scale: usize, offset: usize },
}
```

**Methods on `BufferFormula`:**
```rust
impl BufferFormula {
    pub fn eval(&self, n: usize) -> usize {
        match self {
            BufferFormula::Fixed(size) => *size,
            BufferFormula::Linear { scale, offset } => scale * n + offset,
        }
    }
}
```

**Methods on `KernelCost`:**
```rust
impl KernelCost {
    /// Parse a @pz_cost block from a kernel source string.
    /// Returns None if no @pz_cost block is found.
    pub fn parse(source: &str) -> Option<Self> { ... }

    /// Total GPU memory for a given input size, in bytes.
    pub fn memory_bytes(&self, input_len: usize) -> usize {
        self.buffers.iter().map(|(_, f)| f.eval(input_len)).sum()
    }

    /// Total GPU threads across all passes.
    pub fn total_threads(&self, input_len: usize) -> usize {
        (self.threads_per_element * input_len as f64) as usize * self.passes
    }
}
```

**Parser implementation notes:**

Do NOT use the `regex` crate. Manual string parsing is sufficient:
1. Scan for a line containing `@pz_cost {`
2. Collect subsequent lines until finding `}`
3. Strip comment prefixes (`//` for both .cl and .wgsl)
4. Parse key-value pairs:
   - `threads_per_element: <float>` → f64
   - `passes: <int>` → usize
   - `local_mem: <int>` → usize
   - `buffers: <name>=<expr>, ...` → each expr:
     - Bare integer → `BufferFormula::Fixed(n)`
     - `N` → `BufferFormula::Linear { scale: 1, offset: 0 }`
     - `N*<int>` → `BufferFormula::Linear { scale: k, offset: 0 }`
   - `note: ...` → ignored

**Tests (in `#[cfg(test)] mod tests`):**

1. Parse a minimal annotation and verify `memory_bytes()` math
2. Parse every actual kernel source via `include_str!` — verify all 15 return `Some`
3. Verify `memory_bytes` at known input sizes against hand-calculated values
4. Parse with `note:` field (should be ignored)
5. No annotation → returns `None`

**Test:** `cargo test gpu_cost` — all parser tests pass.

**Commit:** "Add gpu_cost module: KernelCost parser and memory model"

---

## Phase 3: Wire cost model into engine init

**What:** Parse kernel cost annotations at engine creation time and store them on engine structs. Add `max_in_flight()` and `gpu_memory_budget()` methods.

### WebGPU (`src/webgpu/mod.rs`)

**Find:** `struct WebGpuEngine` — add cost fields for each kernel.

**Find:** The `create()` function — after creating compute pipelines, parse costs from the `include_str!` constants using `KernelCost::parse()`. Use `.expect()` so a missing annotation is a hard error (it means someone added a kernel without the cost block).

**Add methods:**
```rust
impl WebGpuEngine {
    /// Max blocks of `block_size` in flight for a kernel without exceeding budget.
    pub fn max_in_flight(&self, kernel: &KernelCost, block_size: usize) -> usize {
        let per_block = kernel.memory_bytes(block_size);
        if per_block == 0 { return 8; }
        (self.gpu_memory_budget() / per_block).max(1)
    }

    /// Conservative GPU memory budget: 50% of max_buffer_size.
    fn gpu_memory_budget(&self) -> usize {
        (self.max_buffer_size as usize) / 2
    }
}
```

### OpenCL (`src/opencl/mod.rs`)

**Find:** `struct OpenClEngine` — add cost fields AND `global_mem_size: u64`. The global memory IS queried during device selection (search for `global_mem_size`) but is NOT currently stored on the struct. Fix that.

**Same pattern:** Parse costs at init, add `max_in_flight()` using `self.global_mem_size / 2` as budget.

**Test:** `cargo test --features opencl` and `cargo test --features webgpu` — engine creation still works.

**Commit:** "Wire kernel cost models into GPU engine init"

---

## Phase 4: Replace `MAX_GPU_BATCH_SIZE` with dynamic scheduling

**What:** Use the cost model to compute batch size instead of the hardcoded constant.

### WebGPU batching (`src/webgpu/lz77.rs`)

**Find:** `find_matches_batched()` — currently uses `blocks.chunks(MAX_GPU_BATCH_SIZE)`.

**Replace with:**
```rust
const GPU_PREFETCH_DEPTH: usize = 3; // cap for latency hiding

let block_size = blocks.first().map(|b| b.len()).unwrap_or(256 * 1024);
let mem_limit = self.max_in_flight(&self.cost_lz77_lazy, block_size);
let batch_size = mem_limit.min(GPU_PREFETCH_DEPTH);
```

Change `blocks.chunks(MAX_GPU_BATCH_SIZE)` → `blocks.chunks(batch_size)`.

**Find:** `MAX_GPU_BATCH_SIZE` in `src/webgpu/mod.rs` — delete the constant.

**Test:**
- `cargo test --features webgpu` — all tests pass
- If GPU available: `cargo bench --bench stages --features webgpu -- lz77` at 4MB should NOT hang

**Commit:** "Replace MAX_GPU_BATCH_SIZE with cost-model-driven dynamic scheduling"

---

## Phase 5: Pipelined producer-consumer (stretch goal, can defer)

**Note:** Phases 1-4 fix the GPU hang by making batch size memory-aware. This phase adds GPU/CPU overlap for throughput — it's valuable but more complex and can be a separate PR.

**If implementing:**

**File:** `src/webgpu/lz77.rs` — add `find_matches_pipelined()`:
```rust
pub fn find_matches_pipelined<F>(
    &self,
    blocks: &[&[u8]],
    callback: F,
) -> PzResult<()>
where
    F: FnMut(usize, Vec<lz77::Match>) -> PzResult<()>,
```

Logic: sliding window of `in_flight_limit` blocks — submit next block as each completes, call callback for each result.

**File:** `src/pipeline/parallel.rs` — rewrite `compress_parallel_gpu_batched()` to use pipelined API with `sync_channel(GPU_PREFETCH_DEPTH)`. **Important:** The consumer side should use a thread pool for entropy encoding, not a single thread, otherwise you trade GPU saturation for CPU underutilization.

**Commit:** "Add pipelined GPU producer-consumer with backpressure"

---

## Files Created/Modified (Summary)

| File | Phase | Change |
|------|-------|--------|
| `kernels/*.cl`, `kernels/*.wgsl` (all 15) | 1 | Add `@pz_cost` annotation |
| **`src/gpu_cost.rs`** | 2 | **New file**: parser, KernelCost, BufferFormula |
| `src/lib.rs` | 2 | Add `mod gpu_cost;` |
| `src/webgpu/mod.rs` | 3, 4 | Cost fields on engine, parse at init, `max_in_flight()`, delete `MAX_GPU_BATCH_SIZE` |
| `src/opencl/mod.rs` | 3 | Cost fields + `global_mem_size` on engine, parse at init, `max_in_flight()` |
| `src/webgpu/lz77.rs` | 4, 5 | Dynamic batch size, `find_matches_pipelined()` (Phase 5) |
| `src/pipeline/parallel.rs` | 5 | Pipelined batched path (Phase 5) |

## Verification

After each phase:
1. `cargo fmt && cargo clippy --all-targets` — zero warnings
2. `cargo test` — all CPU tests pass
3. `cargo clippy --all-targets --features opencl` — GPU lint clean
4. `cargo clippy --all-targets --features webgpu` — GPU lint clean
5. `cargo test --features opencl` — GPU tests pass (skip if no device)
6. `cargo test --features webgpu` — GPU tests pass (skip if no device)

After Phase 4:
- The 4MB GPU hang should be fixed
- GPU memory usage should be lower and more predictable

After Phase 5:
- `./scripts/bench.sh` — end-to-end throughput same or better
- Activity Monitor GPU usage: bursty instead of sustained 80-100%
