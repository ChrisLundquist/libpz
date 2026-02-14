# Task: Implement GPU Huffman decode with sync points

## Context

The WebGPU backend has GPU Huffman **encode** (fully parallel via prefix sum) but no
GPU Huffman **decode**. The previous GPU entropy decode kernels (rANS, FSE multi-block)
were removed because they're inherently sequential per stream — workgroup_size(1) or
4-of-64-threads active. GPU entropy decode needs a fundamentally parallel approach.

The solution is **Huffman decode with sync points**: during encode, periodically
record `(bit_offset, symbol_count)` checkpoints. During decode, each GPU thread
independently decodes one segment between checkpoints. This is the approach used by
NVIDIA nvCOMP and GDeflate.

## What to implement

### 1. Modify Huffman encode to emit sync points

**File:** `src/huffman.rs` (CPU encode) and `src/webgpu/huffman.rs` (GPU encode)

During Huffman encoding, every N symbols (N=1024 recommended), record a sync point:
```
SyncPoint { bit_offset: u32, symbol_index: u32 }
```

The sync point table is stored as part of the encoded output, between the Huffman tree
header and the bitstream data. Format:
```
[tree header] [num_sync_points: u16] [sync_points: (bit_offset: u32, symbol_index: u32) × N] [bitstream]
```

The GPU encode path (`huffman_encode.wgsl`) already computes per-symbol bit offsets via
prefix sum. After the prefix sum, emit sync points by reading `bit_offsets[k*1024]` for
each k. This requires one small readback or a gather kernel.

The CPU encode path should emit identical sync points for format compatibility.

### 2. New WGSL kernel: `huffman_decode.wgsl`

Create a new kernel that decodes Huffman-encoded data in parallel using sync points.

**Design:**
- Workgroup size: 64
- Each thread decodes one segment (between two adjacent sync points)
- Dispatch: `ceil(num_sync_points / 64)` workgroups
- Each thread uses a lookup table (max codeword length L, so 2^L entries) to decode
  one symbol per table access in O(1)

**Bindings:**
- `@binding(0)` decode_table: `array<u32>` — LUT indexed by next L bits, gives (symbol, code_length)
- `@binding(1)` bitstream: `array<u32>` — packed bitstream data
- `@binding(2)` sync_points: `array<u32>` — packed (bit_offset, symbol_count) pairs
- `@binding(3)` output: `array<atomic<u32>>` — output bytes packed as u32
- `@binding(4)` params: `vec4<u32>` — (num_segments, total_symbols, max_code_length, 0)

**Decode table construction (CPU-side):**
For canonical Huffman with max code length L, build a 2^L entry LUT:
```
for each symbol with code (codeword, length):
    for all 2^(L-length) suffixes:
        lut[codeword | (suffix << length)] = (symbol << 8) | length
```
Each thread reads L bits, looks up (symbol, length), writes symbol, advances by length bits.

**Thread algorithm:**
```
segment_id = global_invocation_id.x
start_bit = sync_points[segment_id].bit_offset
start_sym = sync_points[segment_id].symbol_index
end_sym = sync_points[segment_id + 1].symbol_index  // or total_symbols for last segment

bit_pos = start_bit
for sym_idx in start_sym..end_sym:
    // Read L bits from bitstream at bit_pos
    bits = read_bits(bitstream, bit_pos, max_code_length)
    entry = decode_table[bits]
    symbol = entry >> 8
    code_len = entry & 0xFF
    write_output_byte(sym_idx, symbol)
    bit_pos += code_len
```

### 3. Host-side wiring: `src/webgpu/huffman.rs`

Add `huffman_decode_gpu()` method to `WebGpuEngine`:
- Parse tree header to reconstruct decode LUT on CPU (small, O(2^L))
- Parse sync point table from encoded data
- Upload LUT, bitstream, sync points to GPU
- Dispatch decode kernel
- Read back output

Also add pipeline wiring in `src/webgpu/mod.rs` (OnceLock + accessor, same pattern as
existing pipelines).

### 4. Integration with pipeline decode path

**File:** `src/pipeline/stages.rs`

Add a `stage_huffman_decode_gpu()` function that calls the new GPU decode when a WebGPU
engine is available and the block is large enough (>= 128KB, matching the existing
encode threshold).

Wire it into the block decompression path in `src/pipeline/blocks.rs` so that pipelines
using Huffman entropy coding (Deflate, Lzh) can decode on GPU.

### 5. Tests

Add to `src/webgpu/tests.rs`:
- `test_huffman_decode_gpu_round_trip` — encode with sync points, decode on GPU, verify
- `test_huffman_decode_gpu_vs_cpu` — verify GPU decode matches CPU decode exactly
- `test_huffman_decode_gpu_larger` — 64KB+ input to exercise many sync point segments
- `test_huffman_decode_gpu_single_symbol` — edge case: all same byte
- `test_huffman_encode_decode_fully_on_device` — encode on GPU → decode on GPU, no readback

### Constraints

- Stay within WebGPU's 4 storage buffer limit per stage (pack if needed)
- Use `array<atomic<u32>>` + `atomicOr` for byte-packed output (same pattern as LZ77 decode)
- The sync point interval (1024) should be a compile-time constant, tunable later
- The decode LUT max size is 2^16 = 64K entries (max Huffman code length 16), which fits
  in a storage buffer. If codes are ≤ 12 bits, the LUT is only 4K entries.
- Existing CPU Huffman decode must continue to work with both old (no sync points) and
  new (with sync points) formats. Add a format version byte or use the existing tree
  header to distinguish.

### Files to modify/create

| File | Action |
|------|--------|
| `kernels/huffman_decode.wgsl` | **Create** — new parallel decode kernel |
| `src/huffman.rs` | **Modify** — add sync point emission to CPU encode, LUT builder |
| `src/webgpu/huffman.rs` | **Modify** — add `huffman_decode_gpu()`, sync point GPU encode |
| `src/webgpu/mod.rs` | **Modify** — add pipeline struct + OnceLock for huffman_decode |
| `src/webgpu/tests.rs` | **Modify** — add decode tests |
| `src/pipeline/stages.rs` | **Modify** — add `stage_huffman_decode_gpu()` |
| `src/pipeline/blocks.rs` | **Modify** — wire GPU decode into block decompress |

### Reference

- Existing GPU Huffman encode: `src/webgpu/huffman.rs` (encode + prefix sum + write_codes)
- Existing WGSL kernel: `kernels/huffman_encode.wgsl` (has prefix_sum_block/apply entries)
- CPU Huffman: `src/huffman.rs` (HuffmanTree, encode/decode, canonical codes)
- Pipeline stage dispatch: `src/pipeline/stages.rs`
- nvCOMP approach: canonical Huffman with periodic bit-offset checkpoints, 1 thread/segment
