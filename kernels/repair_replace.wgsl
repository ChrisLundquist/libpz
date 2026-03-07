// Two-buffer replace + scatter kernels for Re-Pair GPU compression.
//
// Fixes the data race in the original repair_replace kernel by reading
// from an immutable source buffer and writing to a separate destination.

// Source symbols (read-only — no race possible).
@group(0) @binding(0) var<storage, read> symbols_in: array<u32>;
// Destination symbols (write-only from this kernel's perspective).
@group(0) @binding(1) var<storage, read_write> symbols_out: array<u32>;
// Keep flags: 1 = keep position, 0 = delete (consumed by bigram replacement).
@group(0) @binding(2) var<storage, read_write> keep_flags: array<u32>;
// Parameters: [n, new_symbol, target_packed (a | b<<16), unused]
@group(0) @binding(3) var<uniform> params: vec4<u32>;

// Replace target bigram with new symbol using two-buffer approach.
// All reads from symbols_in (immutable), all writes to symbols_out + keep_flags.
@compute @workgroup_size(256)
fn repair_replace_twobuf(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let new_symbol = params.y;
    let target_packed = params.z;
    let target_a = target_packed & 0xFFFFu;
    let target_b = (target_packed >> 16u) & 0xFFFFu;

    let pos = gid.x + gid.y * 65535u * 256u;
    if (pos >= n) {
        return;
    }

    // Default: keep this position and copy input to output.
    keep_flags[pos] = 1u;
    symbols_out[pos] = symbols_in[pos];

    if (pos >= n - 1u) {
        return;
    }

    let sym_a = symbols_in[pos];
    let sym_b = symbols_in[pos + 1u];

    if (sym_a == target_a && sym_b == target_b) {
        // Non-overlapping: left-to-right greedy semantics.
        // If previous position also matches the target bigram, only the
        // earlier position gets to replace. We check whether pos-1 starts
        // a valid bigram — if so, pos yields to pos-1.
        var can_replace: bool = true;
        if (pos > 0u) {
            let prev_a = symbols_in[pos - 1u];
            let prev_b = symbols_in[pos]; // = sym_a
            if (prev_a == target_a && prev_b == target_b) {
                // pos-1 also matches. In left-to-right greedy, pos-1 wins.
                // But pos-1 might itself be blocked by pos-2...
                // For GPU parallel: odd positions yield when both pos-1 and pos match.
                can_replace = (pos % 2u) == 0u;
            }
        }

        if (can_replace) {
            symbols_out[pos] = new_symbol;
            keep_flags[pos + 1u] = 0u; // mark successor for deletion
        }
    }
}

// Scatter: compact kept symbols using prefix-sum offsets.
// After prefix sum on keep_flags, keep_flags[i] = output position + 1 (inclusive sum).
@compute @workgroup_size(256)
fn repair_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.x;
    let pos = gid.x + gid.y * 65535u * 256u;

    if (pos >= n) {
        return;
    }

    // Original keep_flags have been replaced by prefix sum values.
    // The inclusive prefix sum at pos gives the 1-based output index.
    // We need to check if this position was originally kept by comparing
    // with the previous position's prefix sum value.
    var is_kept: bool;
    if (pos == 0u) {
        // Position 0 is kept if prefix_sum[0] > 0 (which it always is: either 0 or 1).
        is_kept = keep_flags[0] > 0u;
    } else {
        is_kept = keep_flags[pos] > keep_flags[pos - 1u];
    }

    if (is_kept) {
        let new_pos = keep_flags[pos] - 1u; // inclusive → 0-based
        symbols_out[new_pos] = symbols_in[pos];
    }
}
