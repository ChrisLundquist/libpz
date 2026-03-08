// GPU kernels for SortLZ (sort-based LZ77 match finding).
//
// Two kernels:
//   1. sortlz_compute_keys — extract one byte of the hash for radix sort
//   2. sortlz_verify_matches — verify adjacent sorted pairs and find best match per position
//
// The radix sort itself reuses the BWT infrastructure (histogram, prefix sum, scatter).

// ---------------------------------------------------------------------------
// Kernel 1: Key extraction for radix sort
// ---------------------------------------------------------------------------

// Sorted position array (reordered by radix sort each pass).
@group(0) @binding(0) var<storage, read> sk_sa: array<u32>;
// Precomputed hashes (one u32 per position).
@group(0) @binding(1) var<storage, read> sk_hashes: array<u32>;
// Output keys (one byte per position, used by radix histogram + scatter).
@group(0) @binding(2) var<storage, read_write> sk_keys: array<u32>;
// Parameters: [n, padded_n, pass_idx (0-3), unused]
@group(0) @binding(3) var<uniform> sk_params: vec4<u32>;

@compute @workgroup_size(256)
fn sortlz_compute_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = sk_params.x;
    let padded_n = sk_params.y;
    let pass_idx = sk_params.z;

    if (i >= padded_n) {
        return;
    }

    let sa_i = sk_sa[i];
    if (sa_i >= n) {
        // Padded entries sort to end (0xFF for LSB-first sort).
        sk_keys[i] = 0xFFu;
        return;
    }

    // Extract byte `pass_idx` from the hash at position sa[i].
    let hash = sk_hashes[sa_i];
    sk_keys[i] = (hash >> (pass_idx * 8u)) & 0xFFu;
}

// ---------------------------------------------------------------------------
// Kernel 2: Match verification and best-match selection
// ---------------------------------------------------------------------------

// Sorted position array (after radix sort, adjacent entries share hashes).
@group(0) @binding(0) var<storage, read> vm_sa: array<u32>;
// Precomputed hashes.
@group(0) @binding(1) var<storage, read> vm_hashes: array<u32>;
// Input data as packed u32s (for byte-level comparison).
@group(0) @binding(2) var<storage, read> vm_input: array<u32>;
// Best match per position: packed as (length << 16) | offset.
// Uses atomicMax so longest match wins.
@group(0) @binding(3) var<storage, read_write> vm_best: array<atomic<u32>>;
// Parameters: [n, max_window, max_candidates, padded_n]
@group(0) @binding(4) var<uniform> vm_params: vec4<u32>;

// Read a single byte from the packed u32 input array.
fn read_byte(pos: u32) -> u32 {
    let word = vm_input[pos / 4u];
    return (word >> ((pos % 4u) * 8u)) & 0xFFu;
}

@compute @workgroup_size(256)
fn sortlz_verify_matches(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = vm_params.x;
    let max_window = vm_params.y;
    let max_candidates = vm_params.z;
    let padded_n = vm_params.w;

    let i = gid.x;
    if (i >= n) {
        return;
    }

    let pos_i = vm_sa[i];
    if (pos_i >= n) {
        return;
    }
    let hash_i = vm_hashes[pos_i];

    // Scan forward through adjacent entries with the same hash.
    for (var j: u32 = 1u; j <= max_candidates; j++) {
        if (i + j >= padded_n) {
            break;
        }

        let pos_j = vm_sa[i + j];
        if (pos_j >= n) {
            continue;
        }

        // Different hash → no more candidates in this group.
        if (vm_hashes[pos_j] != hash_i) {
            break;
        }

        // Determine source (earlier) and destination (later) positions.
        var src: u32;
        var dst: u32;
        if (pos_i < pos_j) {
            src = pos_i;
            dst = pos_j;
        } else {
            src = pos_j;
            dst = pos_i;
        }

        let distance = dst - src;
        if (distance == 0u || distance > max_window) {
            continue;
        }

        // Extend match by comparing bytes from position 0 (hash match does not
        // guarantee byte equality on collision).
        var match_len: u32 = 0u;
        let max_len = min(n - dst, min(n - src, 260u)); // cap extension at 260
        for (var k: u32 = 0u; k < max_len; k++) {
            if (read_byte(src + k) != read_byte(dst + k)) {
                break;
            }
            match_len = k + 1u;
        }

        // Require minimum match length of 4 (matches SortLzConfig::min_match).
        if (match_len < 4u) {
            continue;
        }

        // Pack (length, offset) into u32: length in high 16 bits so atomicMax prefers longer.
        let offset = min(distance, 65535u);
        let length = min(match_len, 65535u);
        let packed = (length << 16u) | offset;
        atomicMax(&vm_best[dst], packed);
    }
}
