// Re-Pair grammar compression GPU kernels for Experiment C.
//
// Three kernels per round:
//   1. repair_histogram — count bigram frequencies
//   2. repair_argmax — find most frequent bigram via parallel reduction
//   3. repair_replace — replace occurrences + mark for compaction
//
// Tests the viability of iterative GPU kernel dispatch (dispatch_latency × round_count).

// Symbol array (u32 per symbol to support extended alphabet).
@group(0) @binding(0) var<storage, read_write> symbols: array<u32>;

// Bigram frequency histogram. For small alphabet (< 512), indexed by
// sym_a * max_alphabet + sym_b. For large alphabet, use sort-based counting.
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>>;

// Output / scratch buffer for argmax result and compaction indices.
@group(0) @binding(2) var<storage, read_write> scratch: array<u32>;

// Parameters: [n (current symbol count), max_alphabet, new_symbol, target_bigram_packed]
// target_bigram_packed = target_a | (target_b << 16)
@group(0) @binding(3) var<uniform> params: vec4<u32>;

// Shared memory for reductions.
var<workgroup> shared_max_val: array<u32, 256>;
var<workgroup> shared_max_idx: array<u32, 256>;

// Count bigram frequencies. Each thread processes one position.
// Bigram at position i is (symbols[i], symbols[i+1]).
@compute @workgroup_size(256)
fn repair_histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let max_alpha = params.y;
    let pos = gid.x + gid.y * 65535u * 256u;

    // Last position has no successor — no bigram.
    if pos >= n - 1u {
        return;
    }

    let sym_a = symbols[pos];
    let sym_b = symbols[pos + 1u];

    // Index into flat 2D histogram.
    let idx = sym_a * max_alpha + sym_b;
    atomicAdd(&histogram[idx], 1u);
}

// Parallel reduction to find the bigram with maximum frequency.
// Each workgroup finds its local max, writes to scratch buffer.
// A second pass over scratch finds the global max.
@compute @workgroup_size(256)
fn repair_argmax(@builtin(local_invocation_id) lid: vec3<u32>,
                 @builtin(workgroup_id) wgid: vec3<u32>,
                 @builtin(global_invocation_id) gid: vec3<u32>) {
    let total_entries = params.x;  // max_alphabet * max_alphabet
    let pos = gid.x + gid.y * 65535u * 256u;
    let tid = lid.x;

    // Load value.
    var val: u32 = 0u;
    var idx: u32 = pos;
    if pos < total_entries {
        val = atomicLoad(&histogram[pos]);
    }

    shared_max_val[tid] = val;
    shared_max_idx[tid] = idx;
    workgroupBarrier();

    // Tree reduction.
    for (var stride: u32 = 128u; stride > 0u; stride = stride / 2u) {
        if tid < stride {
            if shared_max_val[tid + stride] > shared_max_val[tid] {
                shared_max_val[tid] = shared_max_val[tid + stride];
                shared_max_idx[tid] = shared_max_idx[tid + stride];
            }
        }
        workgroupBarrier();
    }

    // Thread 0 writes block result.
    if tid == 0u {
        let block_idx = wgid.x + wgid.y * 65535u;
        // scratch layout: [max_freq, max_idx] pairs per block
        scratch[block_idx * 2u] = shared_max_val[0];
        scratch[block_idx * 2u + 1u] = shared_max_idx[0];
    }
}

// Replace target bigram with new symbol + mark positions for compaction.
// scratch[i] = 1 if position i is kept, 0 if deleted.
@compute @workgroup_size(256)
fn repair_replace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let new_symbol = params.z;
    let target_packed = params.w;
    let target_a = target_packed & 0xFFFFu;
    let target_b = (target_packed >> 16u) & 0xFFFFu;

    let pos = gid.x + gid.y * 65535u * 256u;
    if pos >= n {
        return;
    }

    // Default: position is kept.
    scratch[pos] = 1u;

    if pos >= n - 1u {
        return;
    }

    let sym_a = symbols[pos];
    let sym_b = symbols[pos + 1u];

    if sym_a == target_a && sym_b == target_b {
        // Check for non-overlapping: only replace if previous position didn't replace.
        // Use a simple even-position-first strategy:
        // If pos is even OR position pos-1 didn't match the target, we can replace.
        var can_replace: bool = true;
        if pos > 0u {
            // Check if previous position also matches — if so, only replace at even positions.
            let prev_a = symbols[pos - 1u];
            let prev_b = symbols[pos];  // = sym_a
            if prev_a == target_a && prev_b == target_b {
                // Both pos-1 and pos match. Take even positions only.
                can_replace = (pos % 2u) == 0u;
            }
        }

        if can_replace {
            // Replace: symbol at pos becomes new_symbol, pos+1 is deleted.
            symbols[pos] = new_symbol;
            scratch[pos + 1u] = 0u;  // mark for deletion
        }
    }
}

// Compact: move kept symbols to contiguous positions using prefix-sum indices.
// scratch[i] contains the new index for position i (from prefix sum of keep flags).
@compute @workgroup_size(256)
fn repair_compact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let pos = gid.x + gid.y * 65535u * 256u;
    if pos >= n {
        return;
    }

    // scratch now holds prefix-sum destinations (set by host between dispatches).
    // We need a two-buffer approach for compaction. For now, read from symbols
    // and write to a second region of scratch.
    // Actually, the host will handle the compaction step using the existing
    // prefix-sum infrastructure from bwt.rs. This kernel is a placeholder.
}
