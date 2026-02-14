# Task: Implement Cooperative-Stitch LZSS Match Finding Pipeline in WebGPU

## Context

We are building a GPU-accelerated LZSS compressor targeting laptop-class hardware via WebGPU. The goal is to emit few, high-quality matches quickly — optimizing match quality per latency. The output feeds a downstream encoding stage (not part of this task).

Prior profiling on the Canterbury corpus shows:
- 40–85% of matches fall within a 1K lookback window (data-dependent)
- 67–95% fall within 4K
- 80–97% fall within 8K
- Diminishing returns beyond 8K

This design uses a cooperative stitch technique: each thread in a workgroup directly searches a small offset band, records its top-K best matches into shared memory, then every thread re-tests all discovered offsets from its own position. This exploits the core property of LZ77: if offset `d` produces a good match at position `p`, it likely produces a good match at nearby positions too, because the underlying repetition at period `d` extends across multiple positions.

The result: 572 probes per thread covering 4K+ effective lookback, versus 4096+ probes for brute-force search of the same range. 8.6× fewer probes with comparable match quality on structured data.

## Core Algorithm: Why "Same Offset" Stitching Works

When thread `k` at position `B+k` finds that `bytes[B+k .. B+k+29]` match `bytes[B+k-200 .. B+k-171]` (offset 200, length 30), thread `m` at position `B+m` should try offset 200 from its own position — NOT the same absolute candidate position.

Why: bytes `B+m` vs `bytes[B+m-200]`. Since `m` is close to `k`, and the match at offset 200 reflects a structural repetition of period 200 in the data, `B+m` is likely inside the same repeated region. If `k=0` and `m=1`, we already know `bytes[B+1] == bytes[B-199]` because it's interior to thread 0's 30-byte match. Thread 1 gets length ≈ 29 at offset 200. Thread 15 gets length ≈ 15. The match degrades by ~1 byte per position of separation from the discovering thread.

This means the stitch pass is NOT speculative — for positions near the discovering thread, matches are near-certain. Quality degrades gracefully with distance and long matches (which matter most for compression) transfer the furthest.

## Architecture Overview

Two compute passes:

1. **Cooperative match finding** — single kernel, two phases separated by workgroupBarrier:
   - Phase A: each thread directly searches its assigned offset band + near region, writes top-K to shared memory
   - Phase B: each thread re-tests ALL offsets discovered by ALL threads in the workgroup, keeps best overall match
2. **Match selection** — segmented forward DP to pick the optimal non-overlapping match set, then backtrack to emit sparse output

## Data Layout

```
Input buffer:       Uint8Array packed as array<u32>, read-only, up to 1MB per dispatch
Match buffer:       array<u32>, 1 entry per input position
                    Encoding: upper 16 bits = offset (1–65535), lower 16 bits = length (0–258)
                    Value 0 means no match found (emit literal)
Cost buffer:        array<u32>, 1 entry per input position + 1, for DP costs
Choice buffer:      array<u32>, 1 entry per input position, for backtracking
Output buffer:      array<u32>, sparse triples of (position, offset, length)
Output count:       single u32 written by backtrack kernel
Params uniform:     see struct below
```

### Byte extraction helper (used by all shaders)

WebGPU storage buffers are u32-aligned. All byte reads use:

```wgsl
fn load_byte(index: u32) -> u32 {
    return (input[index / 4u] >> ((index % 4u) * 8u)) & 0xFFu;
}
```

### Params struct

```wgsl
struct Params {
    input_size: u32,
    min_match_len: u32,     // default 5
    max_match_len: u32,     // default 258
    window_size: u32,       // W, default 256 (direct search range per thread)
    stride: u32,            // S, default 64 (offset between adjacent threads' bands)
    near_range: u32,        // default 64 (near-region every thread searches)
    top_k: u32,             // default 4 (candidates per thread in shared memory)
    literal_cost: u32,      // default 9 (bits, for DP cost model)
    match_fixed_cost: u32,  // default 25 (bits, for DP cost model)
}
```

---

## Pass 1: Cooperative Match Finding

### Workgroup geometry

- 64 threads per workgroup (2 warps)
- Each workgroup processes 64 consecutive input positions: `[block_start, block_start + 64)`
- `block_start = workgroup_id * 64`
- Thread `t` (local_invocation_id.x) operates on position `pos = block_start + t`

### Phase A: Direct Search

Each thread performs two searches from its position `pos = block_start + t`:

**Near search:** offsets `[1, near_range]` (default [1, 64]). Every thread does this to ensure short-distance matches are never missed. This covers the most common match distances.

**Strided search:** offsets `[t * stride + 1, t * stride + window_size]` (default `[t*64+1, t*64+256]`). Thread 0 searches [1, 256], thread 1 searches [65, 320], thread 2 searches [129, 384], ..., thread 63 searches [4033, 4288]. Adjacent thread windows overlap by `window_size - stride = 192` positions in absolute candidate space. Total coverage with no gaps: [1, 4288].

Each thread keeps its top-K matches (by length, breaking ties by smallest offset) in registers during the search. After both searches complete, write top-K to shared memory.

**Probe count per thread:** `near_range + window_size = 64 + 256 = 320` (each probe is a full match-length comparison, early-exiting on mismatch or when max_match_len is reached).

### Shared memory layout

```wgsl
const WORKGROUP_SIZE: u32 = 64u;
const MAX_TOP_K: u32 = 4u;  // tunable, 8 for higher quality

// Top-K storage: each thread writes K entries of (offset: u16, length: u16) packed as u32
// Total: 64 threads × 4 entries × 4 bytes = 1024 bytes
var<workgroup> top_k_matches: array<u32, 256>;  // 64 * 4

// Index: top_k_matches[thread_id * MAX_TOP_K + k]
```

Total shared memory: 1024 bytes. Well under the 48KB limit on laptop GPUs. This leaves ample room for increasing K to 8 (2048 bytes) or even 16 (4096 bytes) if quality demands it.

### Phase B: Stitch

After `workgroupBarrier()`, every thread reads ALL top-K entries from ALL other threads in the workgroup. For each discovered offset `d`, the thread tests offset `d` from its own position `pos`:

```
Compare bytes[pos .. pos + max_match_len] vs bytes[pos - d .. pos - d + max_match_len]
```

The thread keeps the single best match across all stitch probes AND its own Phase A results.

**Stitch probe count per thread:** `(WORKGROUP_SIZE - 1) × top_k = 63 × 4 = 252` probes. Many will be duplicates (adjacent threads' bands overlap, so they discover similar offsets). Deduplicate by skipping offsets already tested in Phase A, or simply allow the duplicate work — 252 probes is cheap.

**Total probes per thread:** `320 (Phase A) + 252 (Phase B) = 572`

For comparison, brute-force search of the same 4288-position range would require 4288 probes. The cooperative stitch achieves 7.5× probe reduction.

### Output

Each thread writes its single best match (the winner across Phase A and Phase B) to the global match buffer at `matches[pos]`. If no match meets `min_match_len`, write 0.

### Pseudocode

```wgsl
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> matches: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 64u;
const MAX_TOP_K: u32 = 4u;

var<workgroup> shared_top_k: array<u32, 256>;  // WORKGROUP_SIZE * MAX_TOP_K

// Helper: compare bytes at pos vs pos-offset, return match length
fn match_length(pos: u32, offset: u32, max_len: u32) -> u32 {
    var len: u32 = 0u;
    while (len < max_len && load_byte(pos + len) == load_byte(pos - offset + len)) {
        len++;
    }
    return len;
}

@compute @workgroup_size(64)
fn cooperative_match_find(
    @builtin(workgroup_id) wg_id: vec3u,
    @builtin(local_invocation_id) lid: vec3u
) {
    let t = lid.x;
    let block_start = wg_id.x * WORKGROUP_SIZE;
    let pos = block_start + t;

    // ── Track top-K in registers ──
    // Use parallel arrays for offset and length, sorted by length descending
    var tk_offset: array<u32, 4>;  // MAX_TOP_K
    var tk_length: array<u32, 4>;
    for (var i: u32 = 0u; i < MAX_TOP_K; i++) {
        tk_offset[i] = 0u;
        tk_length[i] = 0u;
    }

    var best_offset: u32 = 0u;
    var best_length: u32 = 0u;

    if (pos < params.input_size) {
        let max_len = min(params.max_match_len, params.input_size - pos);

        // ── Phase A: Near search [1, near_range] ──
        let near_max = min(pos, params.near_range);
        for (var offset: u32 = 1u; offset <= near_max; offset++) {
            let len = match_length(pos, offset, max_len);
            if (len >= params.min_match_len) {
                insert_top_k(&tk_offset, &tk_length, offset, len);
                if (len > best_length) {
                    best_length = len;
                    best_offset = offset;
                }
            }
        }

        // ── Phase A: Strided search [t*stride+1, t*stride+window_size] ──
        let band_lo = t * params.stride + 1u;
        let band_hi = t * params.stride + params.window_size;
        let band_max = min(pos, band_hi);
        if (band_lo <= pos) {
            for (var offset: u32 = band_lo; offset <= band_max; offset++) {
                let len = match_length(pos, offset, max_len);
                if (len >= params.min_match_len) {
                    insert_top_k(&tk_offset, &tk_length, offset, len);
                    if (len > best_length) {
                        best_length = len;
                        best_offset = offset;
                    }
                }
            }
        }
    }

    // ── Write top-K to shared memory ──
    for (var k: u32 = 0u; k < MAX_TOP_K; k++) {
        if (tk_length[k] >= params.min_match_len) {
            shared_top_k[t * MAX_TOP_K + k] = (tk_offset[k] << 16u) | tk_length[k];
        } else {
            shared_top_k[t * MAX_TOP_K + k] = 0u;
        }
    }

    workgroupBarrier();

    // ── Phase B: Stitch — try all other threads' discovered offsets ──
    if (pos < params.input_size) {
        let max_len = min(params.max_match_len, params.input_size - pos);

        for (var other_t: u32 = 0u; other_t < WORKGROUP_SIZE; other_t++) {
            if (other_t == t) { continue; }  // skip own entries, already tested

            for (var k: u32 = 0u; k < MAX_TOP_K; k++) {
                let entry = shared_top_k[other_t * MAX_TOP_K + k];
                if (entry == 0u) { continue; }

                let stitch_offset = entry >> 16u;

                // Bounds check: offset must be <= pos (can't look before start of input)
                if (stitch_offset > pos) { continue; }

                // Optional: skip if this offset was already covered in Phase A
                // (i.e., within near_range or within this thread's band)
                // For simplicity, just re-test it — 252 probes is cheap

                let len = match_length(pos, stitch_offset, max_len);
                if (len > best_length) {
                    best_length = len;
                    best_offset = stitch_offset;
                }
            }
        }
    }

    // ── Write final best match to global memory ──
    if (pos < params.input_size) {
        if (best_length >= params.min_match_len) {
            matches[pos] = (best_offset << 16u) | best_length;
        } else {
            matches[pos] = 0u;
        }
    }
}

// ── Top-K insertion (maintain sorted by length descending) ──
// Replaces the smallest entry if len is larger
fn insert_top_k(
    offsets: ptr<function, array<u32, 4>>,
    lengths: ptr<function, array<u32, 4>>,
    offset: u32,
    len: u32
) {
    // Find the minimum entry
    var min_idx: u32 = 0u;
    var min_len: u32 = (*lengths)[0];
    for (var i: u32 = 1u; i < MAX_TOP_K; i++) {
        if ((*lengths)[i] < min_len) {
            min_len = (*lengths)[i];
            min_idx = i;
        }
    }
    // Replace if new entry is better
    if (len > min_len) {
        (*offsets)[min_idx] = offset;
        (*lengths)[min_idx] = len;
    }
}
```

### Dispatch

```javascript
const numBlocks = Math.ceil(inputSize / 64);  // one workgroup per 64 input positions
encoder.dispatchWorkgroups(numBlocks);
```

### Important notes

- **Phase A overlap is intentional.** Thread 0 searches [1, 256], thread 1 searches [65, 320]. The overlap of 192 means both threads independently discover matches in the [65, 256] offset range. This is redundant but harmless — it increases top-K diversity. The stitch phase benefits from multiple threads independently confirming the same offset.

- **Top-K diversity matters more than top-K size.** K=4 captures the best matches per thread, but if all 4 are at similar offsets, the stitch gets little diversity. A variant: store top-1 per 64-offset bucket instead of global top-4. This guarantees offset diversity. Implement this as a follow-up optimization if K=4 proves insufficient.

- **The near-region search [1, 64] is non-negotiable.** Without it, thread 63 (whose band starts at offset 4033) has no direct coverage of short distances. The stitch gives it offset hints from threads 0–1's top-K, but those may not cover the specific short-distance matches relevant to thread 63's position. The 64 extra probes are cheap insurance.

- **Deduplication in Phase B is optional.** A stitch offset that falls within the thread's own Phase A search range will be retested. This is wasted work but only ~10-20% of stitch probes on average (only threads whose band overlaps the discovering thread's band). The branch cost of checking for duplicates may exceed the cost of the redundant probe.

---

## Pass 2: Optimal Selection via Forward DP

### Concept

Pass 1 produces one best match per position. The selection pass decides which matches to keep and which to skip in favor of literals, minimizing total encoding cost.

This is the shortest-path / optimal parsing formulation:
- `cost[i]` = minimum cost to encode `input[0..i)`
- `cost[0] = 0`
- For each position `i`:
  - Emit literal: `cost[i+1] = min(cost[i+1], cost[i] + literal_cost)`
  - Use match of length `L`: `cost[i+L] = min(cost[i+L], cost[i] + match_fixed_cost)`
- The optimal parse is recovered by backtracking from `cost[input_size]`

### GPU strategy: segmented DP

Since pass 1 emits only 1 match per position (not 4 candidates), the DP inner loop is trivial — at each position, evaluate the literal transition and (if a match exists) one match transition. The sequential dependency is along positions; the parallelism comes from processing multiple segments simultaneously.

1. Divide input into segments of 4096 positions
2. Each workgroup processes one segment
3. Thread 0 advances the DP position-by-position (sequential within segment)
4. At each position: compare literal cost vs match cost, take the cheaper one, record choice

With only 1 candidate per position, there's no need for parallel candidate evaluation within each position — thread 0 alone handles it. The GPU utilization comes from running many segments concurrently.

### Implementation: Pass 2a — Forward DP

```wgsl
@group(0) @binding(0) var<storage, read> matches: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> cost: array<u32>;
@group(0) @binding(3) var<storage, read_write> choice: array<u32>;

const SEGMENT_SIZE: u32 = 4096u;
const OVERFLOW: u32 = 258u;  // max_match_len, for segment boundary spill

var<workgroup> local_cost: array<u32, 4354>;  // SEGMENT_SIZE + OVERFLOW

@compute @workgroup_size(1)  // one thread per segment
fn forward_dp(@builtin(workgroup_id) wg_id: vec3u) {
    let seg_start = wg_id.x * SEGMENT_SIZE;
    let seg_end = min(seg_start + SEGMENT_SIZE, params.input_size);

    // Initialize local cost array
    for (var i: u32 = 0u; i < SEGMENT_SIZE + OVERFLOW; i++) {
        local_cost[i] = 0xFFFFFFFFu;
    }

    // Seed: cost of reaching seg_start
    if (seg_start == 0u) {
        local_cost[0] = 0u;
    } else {
        local_cost[0] = cost[seg_start];
    }

    // Forward pass
    for (var rel_pos: u32 = 0u; rel_pos < SEGMENT_SIZE; rel_pos++) {
        let pos = seg_start + rel_pos;
        if (pos >= params.input_size) { break; }

        let current_cost = local_cost[rel_pos];
        if (current_cost == 0xFFFFFFFFu) { continue; }

        // Option 1: literal
        let lit_cost = current_cost + params.literal_cost;
        if (lit_cost < local_cost[rel_pos + 1u]) {
            local_cost[rel_pos + 1u] = lit_cost;
        }

        // Option 2: match (if one exists at this position)
        let m = matches[pos];
        if (m != 0u) {
            let length = m & 0xFFFFu;
            let match_cost = current_cost + params.match_fixed_cost;
            let target = rel_pos + length;
            if (target < SEGMENT_SIZE + OVERFLOW && match_cost < local_cost[target]) {
                local_cost[target] = match_cost;
            }
        }

        // Record choice: did we arrive here via literal or match?
        // Check: was this position's cost set by a literal from pos-1 or a match from some earlier pos?
        // Simpler approach: record at decision time, not arrival time.
        // We record what we EMIT from this position.
        // The backtrack pass will reconstruct which choices were actually on the optimal path.
    }

    // Write costs for positions that spill into the next segment
    for (var i: u32 = 0u; i < OVERFLOW; i++) {
        let global_pos = seg_end + i;
        if (global_pos <= params.input_size && local_cost[SEGMENT_SIZE + i] < cost[global_pos]) {
            cost[global_pos] = local_cost[SEGMENT_SIZE + i];
        }
    }

    // Write local costs back to global cost array for this segment
    for (var i: u32 = 0u; i < SEGMENT_SIZE; i++) {
        let global_pos = seg_start + i;
        if (global_pos < params.input_size) {
            cost[global_pos] = local_cost[i];
        }
    }
}
```

**Segment boundary handling:** A match near the end of segment K may land in segment K+1's range. The spill region (OVERFLOW = 258 positions past seg_end) captures this. After all segments run, the global cost array has correct values because:
- Segments process left-to-right (segment 0 first)
- Each segment reads its start cost from the global array (set by the previous segment's spill)
- This requires segments to run in order. **Dispatch segments sequentially** (or use a single workgroup with a loop if segment count is small). For inputs up to 1MB, that's 256 segments — fine for a single kernel dispatch with workgroup_size(1) and 256 workgroups, executed left-to-right.

**IMPORTANT:** If WebGPU does not guarantee left-to-right workgroup execution order (it doesn't), then this must be handled differently. Two options:

**Option A (recommended for simplicity):** Run the DP as a single-threaded kernel on the full input. With 1 candidate per position, the inner loop is 2 comparisons per position. On 1MB input that's ~1M iterations of trivial work — a few milliseconds on a GPU thread. This is the simplest correct implementation.

**Option B (for larger inputs):** Multi-pass segmented DP. Pass i processes even-numbered segments, pass i+1 processes odd-numbered segments using spill values from pass i. Converges in 2 passes for non-overlapping segments. More complex, only worthwhile above ~4MB input.

**Use Option A for the initial implementation.**

### Implementation: Pass 2b — Backtrack and Compact

Single-threaded kernel that walks the cost array to reconstruct the optimal parse.

```wgsl
@group(0) @binding(0) var<storage, read> matches: array<u32>;
@group(0) @binding(1) var<storage, read> cost: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_count: atomic<u32>;

@compute @workgroup_size(1)
fn backtrack_and_compact() {
    var pos: u32 = 0u;
    var out_idx: u32 = 0u;

    while (pos < params.input_size) {
        let current_cost = cost[pos];

        // Check if a match transition is optimal from this position
        let m = matches[pos];
        if (m != 0u) {
            let length = m & 0xFFFFu;
            let offset = m >> 16u;
            let match_target = pos + length;

            if (match_target <= params.input_size
                && cost[match_target] == current_cost + params.match_fixed_cost) {
                // Match was on the optimal path
                output[out_idx * 3u] = pos;
                output[out_idx * 3u + 1u] = offset;
                output[out_idx * 3u + 2u] = length;
                out_idx++;
                pos = match_target;
                continue;
            }
        }

        // Otherwise, literal
        pos += 1u;
    }

    atomicStore(&output_count, out_idx);
}
```

### Dispatch

```javascript
// Pass 2a: single-threaded DP (Option A)
encoder.dispatchWorkgroups(1);  // one workgroup, one thread

// Pass 2b: backtrack
encoder.dispatchWorkgroups(1);
```

---

## JavaScript / WebGPU Host Code

### Interface

```typescript
interface LZSSMatcherParams {
    minMatchLength?: number;    // default 5
    maxMatchLength?: number;    // default 258
    windowSize?: number;        // W, default 256
    stride?: number;            // S, default 64
    nearRange?: number;         // default 64
    topK?: number;              // default 4
    literalCost?: number;       // default 9 (bits)
    matchFixedCost?: number;    // default 25 (bits)
}

interface MatchResult {
    position: number;
    offset: number;
    length: number;
}

class LZSSMatcher {
    constructor(device: GPUDevice, params?: LZSSMatcherParams);

    // Compile shaders and create pipelines — call once
    async initialize(): Promise<void>;

    // Run all passes, return sparse match list
    async findMatches(input: Uint8Array): Promise<MatchResult[]>;

    // Cleanup GPU resources
    destroy(): void;
}
```

### Buffer allocation in `findMatches()`

```
input_buffer:      ceil(input.length / 4) * 4 bytes, STORAGE | COPY_DST
match_buffer:      input.length * 4 bytes, STORAGE
cost_buffer:       (input.length + 1) * 4 bytes, STORAGE, initialized to 0xFFFFFFFF except [0] = 0
output_buffer:     input.length * 3 * 4 bytes (pessimistic), STORAGE | COPY_SRC
output_count_buf:  4 bytes, STORAGE | COPY_SRC
params_uniform:    Params struct, UNIFORM | COPY_DST
staging_buffer:    for readback, MAP_READ | COPY_DST
```

### Dispatch sequence

```javascript
async findMatches(input: Uint8Array): Promise<MatchResult[]> {
    // Upload input, create buffers, set uniforms...

    const encoder = device.createCommandEncoder();

    // Pass 1: Cooperative match finding
    const pass1 = encoder.beginComputePass();
    pass1.setPipeline(this.cooperativePipeline);
    pass1.setBindGroup(0, this.pass1BindGroup);
    pass1.dispatchWorkgroups(Math.ceil(input.length / 64));
    pass1.end();

    // Pass 2a: Forward DP (single-threaded)
    const pass2a = encoder.beginComputePass();
    pass2a.setPipeline(this.dpPipeline);
    pass2a.setBindGroup(0, this.pass2aBindGroup);
    pass2a.dispatchWorkgroups(1);
    pass2a.end();

    // Pass 2b: Backtrack and compact
    const pass2b = encoder.beginComputePass();
    pass2b.setPipeline(this.backtrackPipeline);
    pass2b.setBindGroup(0, this.pass2bBindGroup);
    pass2b.dispatchWorkgroups(1);
    pass2b.end();

    // Copy output to staging buffer for readback
    encoder.copyBufferToBuffer(this.outputCountBuf, 0, this.stagingCount, 0, 4);
    // Read count first, then copy the right amount of output data

    device.queue.submit([encoder.finish()]);

    // Readback
    await this.stagingCount.mapAsync(GPUMapMode.READ);
    const count = new Uint32Array(this.stagingCount.getMappedRange())[0];
    this.stagingCount.unmap();

    // Now read output triples
    // (In practice, copy the full pessimistic output buffer and read only `count` triples,
    //  or do a second submit with the known count for a smaller copy)
    await this.stagingOutput.mapAsync(GPUMapMode.READ);
    const raw = new Uint32Array(this.stagingOutput.getMappedRange());
    const results: MatchResult[] = [];
    for (let i = 0; i < count; i++) {
        results.push({
            position: raw[i * 3],
            offset: raw[i * 3 + 1],
            length: raw[i * 3 + 2],
        });
    }
    this.stagingOutput.unmap();

    return results;
}
```

---

## Constraints

- Target WebGPU with valid WGSL shaders
- Must work on laptop-class GPUs: assume 48KB shared memory per workgroup, modest L2
- Do NOT load the full search window into shared memory — only the top-K entries (1–2KB). The input is read from the storage buffer, relying on L1/L2 cache for locality.
- Byte extraction from u32-packed storage required (WebGPU storage buffers are u32-aligned)
- All tunable parameters (window_size, stride, top_k, cost model, etc.) must come from the uniform buffer. Do not hardcode values in shaders. The exception is WORKGROUP_SIZE and MAX_TOP_K which must be compile-time constants in WGSL — define these as shader module constants and ensure the host code uses matching values.
- No dynamic memory allocation on GPU

## Evaluation Criteria

1. **Correctness**: every emitted match must be verifiable — `input[position + i] == input[position - offset + i]` for all `i` in `[0, length)`
2. **Match quality**: on `alice29.txt`, the total encoding cost (literal_cost × num_literals + match_fixed_cost × num_matches) should be within 5% of a CPU reference optimal parse with a 4K window
3. **Probe efficiency**: instrument pass 1 to count total byte comparisons per input byte. Should be approximately 572 (or less with early exits), NOT 4096+
4. **Latency**: total GPU time for all passes on 256KB input, measured via `performance.now()` around queue submission and readback
5. **Stitch value**: measure match quality with stitch disabled (Phase B skipped) vs enabled. On text files, stitch should improve total matched bytes by 30%+ (reflecting the 45% → 71% CDF jump from 1K to 4K)

## Suggested File Structure

```
src/
    lzss-matcher.ts             # Host code: LZSSMatcher class
    shaders/
        cooperative-match.wgsl  # Pass 1: cooperative search + stitch
        forward-dp.wgsl         # Pass 2a: segmented forward DP
        backtrack.wgsl           # Pass 2b: backtrack and compact
    test/
        test-matcher.ts         # Validation, quality comparison, probe counting
        reference-dp.ts         # CPU reference optimal parser for comparison
```

## Scaling Parameters

The architecture scales to larger lookback windows by adjusting S and W:

```
┌──────┬──────┬───────────┬──────────────┬───────────────────────────────────┐
│  S   │  W   │ Coverage  │ Probes/thread│ Notes                             │
├──────┼──────┼───────────┼──────────────┼───────────────────────────────────┤
│  64  │  256 │  4.3K     │  572         │ Default. Good for most data.      │
├──────┼──────┼───────────┼──────────────┼───────────────────────────────────┤
│  256 │  256 │ 16.3K     │  572         │ No overlap. Clean tiling.         │
├──────┼──────┼───────────┼──────────────┼───────────────────────────────────┤
│  512 │  512 │ 32.7K     │  828         │ Full deflate window. Best quality.│
├──────┼──────┼───────────┼──────────────┼───────────────────────────────────┤
│  512 │  256 │ 32.7K     │  572         │ Full range but 256-pos gaps.      │
└──────┴──────┴───────────┴──────────────┴───────────────────────────────────┘

Shared memory cost is dominated by top-K storage:
  64 threads × K entries × 4 bytes = 256K bytes (K=4) or 512K bytes (K=8)
All configurations fit comfortably in 48KB.
```

To scale, the host code only needs to change the uniform buffer values. No shader recompilation required (assuming MAX_TOP_K is set to the maximum K you'll use).

## Key Risks and Mitigations

**Risk: Top-K doesn't capture diverse enough offsets.** If a thread's 256-offset band has many matches at similar offsets, all K slots go to nearby offsets and the stitch provides poor coverage.
**Mitigation:** Change top-K strategy to "top-1 per bucket" — divide the offset band into K equal sub-ranges, keep the best match in each. This guarantees offset diversity at the cost of potentially missing the 2nd-best match in a strong sub-range. Implement as an optional mode selectable via a flag in params.

**Risk: Single-threaded DP is too slow on large inputs.** At 1MB input, the DP is ~1M iterations of trivial work — likely 2–5ms on a GPU thread. If this is too slow, move to multi-pass segmented DP (Option B described above).
**Mitigation:** Benchmark the single-threaded DP on target hardware before optimizing. It's almost certainly not the bottleneck compared to pass 1.

**Risk: Stitch is less effective on random/binary data.** Offsets that produce good matches for one position don't transfer to others in unstructured data.
**Mitigation:** This is inherent and acceptable. The CDF shows random-like data (kennedy.xls) already has 85% of matches within 1K, so the stitch's marginal value on such data is small. The architecture still works correctly — it just doesn't find many additional matches via stitch, falling back to Phase A's direct search quality.
