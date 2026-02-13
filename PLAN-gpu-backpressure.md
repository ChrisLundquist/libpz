# GPU Backpressure: Kernel Cost Annotations and Runtime Scheduling

## Problem

When running GPU workloads, we see 80-100% GPU utilization, inconsistent benchmarks, and a GPU hang at 4MB. The root cause: **the scheduler has no idea how expensive a kernel dispatch is**, so it fire-and-forgets up to 8 blocks with zero backpressure.

Currently, `MAX_GPU_BATCH_SIZE = 8` is a magic constant. It doesn't account for:
- Which kernel is being dispatched (hash LZ77 uses 60 MB per block, Huffman uses 22 MB)
- The input size (a 256KB block is fine, a 4MB block blows the GPU memory budget)
- The device's actual VRAM (we already query `global_mem_size` / `max_buffer_size` but don't use it for scheduling)

## Idea

**Annotate each kernel with its cost model as structured comments, then parse those at init time to derive a per-dispatch memory/thread budget. The scheduler uses this to decide how many blocks to keep in flight.**

The kernels are already loaded as `include_str!` constants — parsing a few comment lines at engine init is essentially free.

## Kernel Cost Annotation Format

Add a `@pz_cost` block to the top of each kernel file. Use a simple key-value format that's easy to regex out of both `.cl` and `.wgsl` files:

```
// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, hash_counts=131072, hash_table=N*8, output=N*12
//   local_mem: 0
// }
```

Where `N` is the input length (substituted at runtime). The format is:

| Field | Meaning | Example |
|-------|---------|---------|
| `threads_per_element` | GPU threads launched per input byte | `1` (one thread per position) or `0.03125` (1 per 32 bytes for batch) |
| `passes` | Number of sequential dispatches | `2` (build + find) |
| `buffers` | Named buffer allocations as expressions of `N` | `input=N, output=N*12, hash_table=8388608` |
| `local_mem` | Workgroup-local memory in bytes | `1024` (for `__local uint[256]`) |

### Concrete annotations for each kernel

**`kernels/lz77_hash.cl` / `kernels/lz77_hash.wgsl`:**
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, hash_counts=131072, hash_table=8388608, output=N*12
//   local_mem: 0
// }
```
At N=256KB: threads=262,144/pass, memory = 256K + 128K + 8M + 3M ≈ **11.5 MB**
At N=4MB: threads=4,194,304/pass, memory = 4M + 128K + 8M + 48M ≈ **60 MB**

**`kernels/lz77_lazy.wgsl`:**
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 3
//   buffers: input=N, hash_counts=131072, hash_table=8388608, raw_matches=N*12, resolved=N*12
//   local_mem: 0
// }
```
At N=256KB: ≈ **14.5 MB**.  At N=4MB: ≈ **108 MB**

**`kernels/lz77_batch.cl`:**
```
// @pz_cost {
//   threads_per_element: 0.03125
//   passes: 1
//   buffers: input=N, output=N*12
//   local_mem: 0
//   note: O(n*window) per thread, high ALU cost
// }
```

**`kernels/huffman_encode.cl` / `kernels/huffman_encode.wgsl`:**
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 4
//   buffers: input=N, lut=1024, bit_lengths=N*4, output=N, block_sums=32768
//   local_mem: 2048
// }
```
At N=256KB: ≈ **1.5 MB**.  At N=4MB: ≈ **22 MB**

**`kernels/bwt_rank.cl` / `kernels/bwt_rank.wgsl`:**
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 4
//   buffers: sa=N*8, rank=N*8, diff=N*4, prefix=N*4
//   local_mem: 2048
//   note: called per doubling step (log2(N) steps)
// }
```

**`kernels/bwt_radix.cl` / `kernels/bwt_radix.wgsl`:**
```
// @pz_cost {
//   threads_per_element: 1
//   passes: 3
//   buffers: keys=N*4, histogram=N*4, sa_in=N*4, sa_out=N*4
//   local_mem: 1024
//   note: 4 radix passes per doubling step
// }
```

## Rust-Side Cost Model

### Step 1: Parser (`src/gpu_cost.rs`, new file)

A small struct and parser that extracts cost annotations from kernel source strings:

```rust
/// Cost model for a single GPU kernel, parsed from @pz_cost annotations.
#[derive(Debug, Clone)]
pub(crate) struct KernelCost {
    pub threads_per_element: f64,
    pub passes: usize,
    /// Buffer formulas: list of (name, bytes as function of input_len)
    pub buffers: Vec<(String, BufferFormula)>,
    pub local_mem: usize,
}

#[derive(Debug, Clone)]
pub(crate) enum BufferFormula {
    /// Fixed size in bytes (e.g. `131072`)
    Fixed(usize),
    /// Linear in N (e.g. `N*12` → scale=12, offset=0)
    Linear { scale: usize, offset: usize },
}

impl KernelCost {
    /// Parse from kernel source string. Returns None if no @pz_cost block found.
    pub fn parse(source: &str) -> Option<Self> { ... }

    /// Total GPU memory for a given input size, in bytes.
    pub fn memory_bytes(&self, input_len: usize) -> usize {
        self.buffers.iter().map(|(_, f)| f.eval(input_len)).sum()
    }

    /// Total threads across all passes.
    pub fn total_threads(&self, input_len: usize) -> usize {
        (self.threads_per_element * input_len as f64) as usize * self.passes
    }
}
```

The parser is intentionally simple: scan for `// @pz_cost {`, collect lines until `// }`, regex out key-value pairs. No complex expression parser — just `N`, `N*K`, and integer literals. This runs once at engine init on `include_str!` constants.

### Step 2: Scheduler (`KernelCost` used in engine init)

At `WebGpuEngine::create()` / `OpenClEngine::create()`, parse cost annotations from all kernel sources and store them:

```rust
struct WebGpuEngine {
    // ... existing fields ...
    cost_lz77_hash: KernelCost,
    cost_lz77_lazy: KernelCost,
    cost_huffman: KernelCost,
    cost_bwt_rank: KernelCost,
    cost_bwt_radix: KernelCost,
}
```

Then add a method that computes how many blocks can be in flight:

```rust
impl WebGpuEngine {
    /// How many blocks of `block_size` can be in flight for the given kernel
    /// without exceeding `memory_budget` bytes of GPU memory.
    pub fn max_in_flight(&self, kernel: &KernelCost, block_size: usize) -> usize {
        let per_block = kernel.memory_bytes(block_size);
        let budget = self.gpu_memory_budget();
        (budget / per_block).max(1)
    }

    /// Conservative GPU memory budget: fraction of device memory or max_buffer_size.
    fn gpu_memory_budget(&self) -> usize {
        // Use 50% of max_buffer_size as budget, leaving room for
        // other stages (Huffman, BWT) and the system.
        (self.max_buffer_size as usize) / 2
    }
}
```

### Step 3: Replace `MAX_GPU_BATCH_SIZE` with dynamic calculation

In `find_matches_batched()` / `find_matches_pipelined()`:

```rust
// Before: hardcoded
const MAX_GPU_BATCH_SIZE: usize = 8;

// After: derived from cost model
let max_batch = self.max_in_flight(&self.cost_lz77_lazy, block_size);
```

For a 256KB block on a GPU with 4GB max buffer:
- `cost_lz77_lazy.memory_bytes(262144)` = 262K + 128K + 8M + 3M + 3M ≈ 14.5 MB
- budget = 2GB → max_batch = 2048/14.5 ≈ **137 blocks** (plenty, memory isn't the bottleneck)

For a 4MB block:
- `cost_lz77_lazy.memory_bytes(4194304)` ≈ 108 MB
- budget = 2GB → max_batch = 2048/108 ≈ **18 blocks** (still safe)
- But with only 4GB VRAM total → budget = 2GB, and 18 × 108 MB = 1.9 GB (tight!)

The `GPU_PREFETCH_DEPTH` constant from the old plan becomes a *cap* on the dynamic value, giving both a memory-derived limit and a latency-overlap limit:

```rust
const GPU_PREFETCH_DEPTH: usize = 3;  // max blocks ahead for latency hiding

let mem_limit = self.max_in_flight(&self.cost_lz77_lazy, block_size);
let in_flight = mem_limit.min(GPU_PREFETCH_DEPTH);
```

## Files to Create/Modify

| File | Change |
|------|--------|
| `kernels/lz77_hash.cl` | Add `@pz_cost` annotation block |
| `kernels/lz77_hash.wgsl` | Add `@pz_cost` annotation block |
| `kernels/lz77_lazy.wgsl` | Add `@pz_cost` annotation block |
| `kernels/lz77.cl` | Add `@pz_cost` annotation block |
| `kernels/lz77_batch.cl` | Add `@pz_cost` annotation block |
| `kernels/lz77_topk.cl` | Add `@pz_cost` annotation block |
| `kernels/lz77_topk.wgsl` | Add `@pz_cost` annotation block |
| `kernels/huffman_encode.cl` | Add `@pz_cost` annotation block |
| `kernels/huffman_encode.wgsl` | Add `@pz_cost` annotation block |
| `kernels/bwt_rank.cl` | Add `@pz_cost` annotation block |
| `kernels/bwt_rank.wgsl` | Add `@pz_cost` annotation block |
| `kernels/bwt_radix.cl` | Add `@pz_cost` annotation block |
| `kernels/bwt_radix.wgsl` | Add `@pz_cost` annotation block |
| `kernels/fse_decode.wgsl` | Add `@pz_cost` annotation block |
| **`src/gpu_cost.rs`** | **New file**: `KernelCost` struct, `BufferFormula` enum, parser, `memory_bytes()`, `total_threads()` |
| `src/lib.rs` | Add `mod gpu_cost;` (crate-internal, not public) |
| `src/webgpu/mod.rs` | Parse costs at init, store in engine struct, add `max_in_flight()` and `gpu_memory_budget()` |
| `src/opencl/mod.rs` | Same: parse costs at init, store in engine struct |
| `src/webgpu/lz77.rs` | Replace `MAX_GPU_BATCH_SIZE` with `self.max_in_flight().min(GPU_PREFETCH_DEPTH)` in `find_matches_batched()` |
| `src/pipeline/parallel.rs` | Use engine's cost-derived batch size in `compress_gpu_batched()` |

## What NOT to Change

- Kernel logic itself — annotations are comments, zero impact on compiled kernels
- WGSL `@workgroup_size()` decorators — already correct
- `deflate_chained()` — single-block, scheduling is the pipeline's job
- `compress_pipeline_parallel()` — already has backpressure via `sync_channel(2)`

## Implementation Order

1. **Annotate all kernel files** with `@pz_cost` blocks — pure comments, no functional change
2. **Create `src/gpu_cost.rs`** — parser + cost model, with unit tests against the actual kernel source strings
3. **Wire into engine init** — parse costs, store on engine struct
4. **Replace `MAX_GPU_BATCH_SIZE`** with dynamic `max_in_flight().min(GPU_PREFETCH_DEPTH)`
5. **Add pipelined producer-consumer** (the Part B from before) using the dynamic batch size

Steps 1-2 are one commit (annotations + parser). Step 3-4 are a second commit (wiring). Step 5 is a third (backpressure pipeline).

## Testing

1. `cargo test` — `gpu_cost::tests` should parse every kernel source and verify `memory_bytes()` against hand-calculated values
2. `cargo test --features opencl,webgpu` — all GPU tests still pass (annotations are comments)
3. Verify `max_in_flight()` returns sensible values: e.g. for 256KB blocks, it should be >= 3 (not the bottleneck); for 4MB blocks on 8GB VRAM, it should be ~2-3
4. `cargo bench --bench stages --features opencl,webgpu -- lz77` — the 4MB GPU hang should be fixed because the scheduler limits in-flight blocks
5. `cargo run --example gpu_compare --release --features opencl,webgpu` — more consistent numbers
