# Optimal Parsing via Backward Dynamic Programming

Design documentation for `src/optimal.rs` — the minimum-cost LZ parse selector.

**Last updated:** 2026-02-21

## Overview

Greedy/lazy LZ parsers make local decisions: pick the longest match, or defer one position if the next match is longer. These heuristics are fast but can miss globally better parses — for example, emitting a cheap literal now to set up a longer, more efficient match later.

Optimal parsing uses backward dynamic programming to evaluate *all* possible parse decisions and select the one with minimum total encoding cost. This is the approach used by zstd (levels 17+) and xz/lzma.

**Trade-off:** 4-6% better compression ratio, 30-50% slower encoding. Used when `ParseStrategy::Optimal` is selected.

## Architecture

```
Input bytes
    │
    ▼
┌───────────────────────┐
│  Match Finding (K=4)  │  ← GPU kernel or CPU hash chains
│  per-position top-K   │
└───────────┬───────────┘
            │ MatchTable (flat Vec, position-major)
            ▼
┌───────────────────────┐
│  Backward DP          │  ← Always CPU (sequential dependency)
│  cost[i] = min cost   │
│  to encode input[i..]│
└───────────┬───────────┘
            │ Vec<Match> (optimal sequence)
            ▼
┌───────────────────────┐
│  Serialize / Demux    │  ← Pipeline-specific (Deflate, Lzr, LzSeqR, etc.)
└───────────────────────┘
```

The split is deliberate: match finding is embarrassingly parallel (GPU excels), while the DP has a sequential data dependency (`cost[i]` depends on `cost[i+1]`) that makes it inherently serial.

## Match Finding

### CPU Path

`build_match_table_cpu(input, k)` — `src/optimal.rs:207`

Uses the existing `lz77::HashChainFinder` to find the top-K match candidates at each input position. Each call to `finder.find_top_k(input, pos, k)` returns up to K matches sorted by length descending.

### GPU Path

`WebGpuEngine::find_topk_matches(input)` — `src/webgpu/lz77.rs:879`
Kernel: `kernels/lz77_topk.wgsl`

One GPU thread per input position. Each thread:
1. Scans the 32KB search window for match candidates
2. Maintains a sorted top-K array in registers (K=4)
3. Uses a **spot-check optimization**: before computing full match length, checks whether `input[j + worst_top_k_len] == input[pos + worst_top_k_len]`. If not, skip — the candidate can't beat the worst entry.
4. Writes K packed `(length << 16 | offset)` values per position

Host-side unpacking converts packed u32 values to `MatchCandidate` structs.

### MatchTable Layout

```
candidates[pos * K + 0]  ← longest match at position pos
candidates[pos * K + 1]  ← second longest
candidates[pos * K + 2]  ← third
candidates[pos * K + 3]  ← shortest (or empty: length=0)
```

`MatchCandidate { offset: u32, length: u32 }` — u32 fields support both LZ77 (offset ≤ 32KB) and LzSeq (offset ≤ 1MB).

## Backward DP Algorithm

`optimal_parse(input, table, cost_model)` — `src/optimal.rs:250`

### Arrays

| Array | Type | Size | Purpose |
|-------|------|------|---------|
| `cost[i]` | `u32` | n+1 | Min scaled-bit cost to encode `input[i..n]` |
| `choice_len[i]` | `u16` | n | Match length at position i (0 = literal) |
| `choice_offset[i]` | `u16` | n | Match offset at position i |

### Backward Pass

For each position `i` from `n-1` down to `0`:

**Literal option** (always available):
```
Token: { offset:0, length:0, next:input[i] }
Covers 1 byte.
cost[i] = literal_token(input[i]) + cost[i+1]
```

**Match options** (for each candidate at position i):
```
For each candidate with length >= MIN_MATCH (3):
  match_end = i + length
  if match_end < n:  // need room for mandatory 'next' byte
    Token: { offset, length, next:input[match_end] }
    Covers (length + 1) bytes.
    mcost = match_cost(offset, length, input[match_end]) + cost[match_end + 1]
    if mcost < cost[i]:
      cost[i] = mcost
      choice_len[i] = length
      choice_offset[i] = offset
```

The literal option is set as the default, then each match candidate may improve it.

### Forward Trace

Walk forward from position 0, following `choice_len` / `choice_offset` to reconstruct the optimal match sequence:

```
pos = 0
while pos < n:
  if choice_len[pos] == 0:
    emit literal(input[pos])
    pos += 1
  else:
    emit match(choice_offset[pos], choice_len[pos], input[pos + choice_len[pos]])
    pos += choice_len[pos] + 1
```

### Complexity

- **Time:** O(n * K) — one pass over all positions, K candidates each
- **Space:** O(n) — three arrays of length n

## Cost Model

`CostModel` — `src/optimal.rs:99`

The cost model estimates the bit cost of encoding each token (literal or match) after entropy coding.

### Token Format Context

In the LZ77 wire format, every token is 5 bytes: `offset:u16 + length:u16 + next:u8`. Literal tokens have `offset=0, length=0` (4 zero bytes + the literal value). These 5 bytes are then entropy-coded by Huffman, FSE, or rANS.

### Cost Components

**`literal_cost[byte]`** — per-byte entropy estimate:
```
cost[b] = -log2(freq[b] / total) * COST_SCALE
```
Computed from input byte frequencies. Default: 8 * COST_SCALE (uniform) if no frequency data.

**`literal_overhead`** = 4 * COST_SCALE (4 bits):
The 4 zero bytes (`offset=0, length=0`) are very common in LZ77 output (~50% of tokens are literals), so `0x00` has low entropy. Estimated at ~1 bit per zero byte.

**`match_overhead`** = 16 * COST_SCALE (16 bits):
Offset and length fields contain varied values. Estimated at ~4 bits/byte average entropy, so 4 bytes * 4 bits = 16 bits.

### Cost Functions

**`literal_token(byte)`**: overhead (4 bits) + entropy of byte value.

**`match_token(next_byte)`**: overhead (16 bits) + entropy of next byte. Used for standard LZ77 pipelines where all matches have uniform overhead regardless of distance.

**`match_cost(offset, length, next_byte)`**: distance-aware cost for LzSeq pipelines.
Uses `lzseq::encode_offset()` and `lzseq::encode_length()` to compute the actual code+extra-bits cost:

```
(offset_code, offset_extra_bits, _) = encode_offset(offset)
(_, length_extra_bits, _)           = encode_length(length)

cost = 2 * (4 * COST_SCALE)           // 2 code bytes at ~4 bits each
     - discount(offset_code)           // offset codes 0,1 → 2-bit discount
     + (offset_extra_bits + length_extra_bits) * COST_SCALE
     + literal_cost[next_byte]
```

Closer matches are cheaper: offset 1 (code 0, 0 extra bits) costs far less than offset 50000 (code 16, 15 extra bits). This lets the DP prefer nearby matches when the compression gain is similar.

### Known Limitation

The cost model does not account for LzSeq repeat offsets. Repeat offset matches cost 0 extra bits for the offset, but the DP doesn't track which offsets are in the repeat buffer. Modeling this would require 3x more DP state (one cost array per repeat-offset configuration). Documented as a future enhancement.

## Pipeline Integration

### ParseStrategy Enum

`src/pipeline/mod.rs:63`

| Strategy | Behavior |
|----------|----------|
| `Auto` | GPU hash-table or CPU lazy (default, fastest) |
| `Lazy` | Force CPU lazy evaluation |
| `Optimal` | Backward DP, best compression |

### Dispatch Logic

`src/pipeline/mod.rs:636-687`

**With GPU backend** (input >= MIN_GPU_INPUT_SIZE):
- `Optimal` → `engine.find_topk_matches()` (GPU) → `compress_optimal_with_table()` (CPU DP)
- `Auto` → `engine.lz77_compress()` (GPU hash-table, greedy)

**CPU backend**:
- `Optimal` → `build_match_table_cpu_with_limit()` → `optimal_parse()`
- `Auto`/`Lazy` → `compress_lazy_with_chain()` / `compress_lazy_with_limit()`

### Demux Fast Path

`src/pipeline/demux.rs:77-100`

For CPU backends, `optimal_matches_with_limit()` returns `Vec<Match>` directly, avoiding the serialize-then-parse overhead of going through the byte wire format.

## Constants and Tuning

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `K` | 4 | Match candidates per position. Testing showed diminishing returns above 4: K=8 adds ~3% compression time for <0.5% improvement. |
| `COST_SCALE` | 256 | Fixed-point precision. Avoids floating-point in the DP inner loop while maintaining sufficient granularity for cost comparison. |
| `literal_overhead` | 4 bits | Empirical: ~50% of LZ77 tokens are literals, making 0x00 very frequent (~1 bit after entropy coding). |
| `match_overhead` | 16 bits | Conservative: offset/length bytes average ~4 bits each after entropy coding. |
| `MIN_MATCH` | 3 | Standard LZ77 minimum match length. |

## Source Files

| Component | Path | Key Lines |
|-----------|------|-----------|
| Core module | `src/optimal.rs` | Full file (567 lines) |
| Data structures | `src/optimal.rs` | 40-79 (`MatchCandidate`, `MatchTable`) |
| Cost model | `src/optimal.rs` | 99-197 (`CostModel`) |
| Backward DP | `src/optimal.rs` | 250-325 (`optimal_parse`) |
| Public API | `src/optimal.rs` | 336-392 |
| Tests | `src/optimal.rs` | 398-566 |
| Pipeline dispatch | `src/pipeline/mod.rs` | 63-74, 636-687 |
| Demux fast path | `src/pipeline/demux.rs` | 77-100 |
| LzSeq cost functions | `src/lzseq.rs` | 132-149 |
| GPU top-K kernel | `kernels/lz77_topk.wgsl` | 1-117 |
| GPU host code | `src/webgpu/lz77.rs` | 879-949 |
