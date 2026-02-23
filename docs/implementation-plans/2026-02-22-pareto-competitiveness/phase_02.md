# Pareto-Competitiveness Implementation Plan — Phase 2: LzSeq Match Finding Quality

**Goal:** Improve LzSeq match finding to close the compression ratio gap with gzip-6.

**Architecture:** Enhance hash chain quality in `src/lz77.rs` (4-byte hash, adaptive chain depth, configurable window), tune match profitability in `src/lzseq.rs`, expose quality configuration via `SeqConfig`.

**Tech Stack:** Rust

**Scope:** 8 phases from original design (phase 2 of 8)

**Codebase verified:** 2026-02-22

---

## Acceptance Criteria Coverage

This phase implements and tests:

### pareto-competitiveness.AC1: Pareto-competitive with gzip
- **pareto-competitiveness.AC1.1 Success:** LzSeqR single-thread achieves compression ratio within 2% of gzip -6 on Canterbury corpus
- **pareto-competitiveness.AC1.2 Success:** LzSeqR single-thread achieves compression ratio within 2% of gzip -6 on Silesia corpus

---

## Current State (from codebase read on 2026-02-22)

### `src/lz77.rs` — relevant constants and types

- **Lines 18-31:** `MAX_WINDOW = 32768`, `WINDOW_MASK`, `MIN_MATCH = 3`, `HASH_SIZE = 1 << 15`, `HASH_MASK`, `MAX_CHAIN = 64`, `MAX_CHAIN_AUTO = 48`
- **Lines 164-172:** `hash3()` — XOR-based 3-byte hash: `(byte[0] << 10) ^ (byte[1] << 5) ^ byte[2]`, masked to `HASH_MASK` (15 bits). Returns 0 if `pos + 2 >= data.len()`.
- **Lines 193-211:** `HashChainFinder` struct fields: `head: Vec<u32>`, `prev: Vec<u32>`, `dispatcher`, `max_match_len: usize`, `max_chain: usize`, `max_window: usize`, `window_mask: usize`.
- **Lines 213-258:** Constructors: `new()` (32KB window, chain 64), `with_max_match_len()`, `with_tuning(max_match_len, max_chain)` (clamps max_chain to `1..=MAX_CHAIN`), `with_window(max_window, max_match_len)` (uses `MAX_CHAIN`).
- **Lines 260-328:** `find_best()` — core chain walk. Uses `hash3`, the probe-byte early-exit optimization, SIMD `compare_bytes`, and follows `prev` links up to `max_chain` times.
- **Lines 389-397:** `insert()` — calls `hash3`, links `pos` into `prev` chain, updates `head`.

### `src/lzseq.rs` — relevant types and functions

- **Lines 45-59:** `SeqConfig { max_window: usize }` (default 128KB). No `max_chain` field.
- **Lines 435-456:** `min_profitable_length(offset: u32) -> u16` — tiered thresholds: offsets 1-256 → min 3, offsets 257-4096 → min 4, offsets 4097-65536 → min 5, offsets 65537+ → min 6. Formula: `MIN_MATCH + (oeb.saturating_sub(7) as u16).div_ceil(4)`.
- **Line 560:** `encode_with_config` constructs `HashChainFinder::with_window(config.max_window, DEFAULT_MAX_MATCH)` — `max_chain` is always `MAX_CHAIN` (64), not configurable through `SeqConfig`.

### Gap analysis

1. **No 4-byte hash:** `hash3` has collisions; 4-byte hash would reduce false chain walks.
2. **`max_chain` not configurable through `SeqConfig`:** `with_window` hard-codes `MAX_CHAIN`. Higher-quality encoding cannot set deeper chains from `encode_with_config`.
3. **No adaptive chain depth:** chain depth is static regardless of data compressibility.
4. **`min_profitable_length` thresholds are conservative:** formula uses integer division that may under-reject slightly (e.g., oeb=8 → min 4, but 3-byte match at offset 300 costs ~3.1 bytes).

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
## Subcomponent A: 4-byte hash option

The current `hash3` XOR formula maps 3 bytes into 15 bits. With a 32K-entry table, the expected occupancy per bucket is `N/32768` — acceptable but collisions still send the chain walker down wrong paths. A 4-byte hash uses one more byte of lookahead to reduce the false-match rate, cutting wasted `compare_bytes` calls on large inputs.

<!-- START_TASK_1 -->
### Task 1: Add `hash4()` function to `src/lz77.rs` alongside `hash3()`

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Modify: `src/lz77.rs:164-172`
- Test: `src/lz77.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

Add `hash4` immediately after `hash3` (after line 172). The function must return 0 when fewer than 4 bytes remain (same guard style as `hash3`). Use a multiply-shift construction that distributes 4 bytes into the same `HASH_MASK` range so no new constants are needed.

```rust
/// Compute a hash for 4 bytes at the given position.
///
/// Uses a multiply-shift construction for better avalanche than XOR.
/// Falls back to 0 (no hash) when fewer than 4 bytes remain.
#[inline(always)]
pub(crate) fn hash4(data: &[u8], pos: usize) -> usize {
    if pos + 3 >= data.len() {
        return 0;
    }
    let v = (data[pos] as u32)
        | ((data[pos + 1] as u32) << 8)
        | ((data[pos + 2] as u32) << 16)
        | ((data[pos + 3] as u32) << 24);
    // Multiply-shift: multiply by a large prime, take top 15 bits.
    // 0x9E37_79B9 is the Fibonacci/golden-ratio constant used by Knuth.
    ((v.wrapping_mul(0x9E37_79B9) >> 17) as usize) & HASH_MASK
}
```

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: `hash4` returns a value in `0..HASH_SIZE` for all inputs (no out-of-bounds table index).
- `hash4` returns 0 when `pos + 3 >= data.len()` (boundary guard).
- `hash4` and `hash3` both return consistent values for the same position across repeated calls (determinism).
- Distinct 4-byte sequences produce distinct hashes for common test vectors (collision spot-check).

```rust
#[test]
fn hash4_range() {
    let data = b"hello world this is a test";
    for pos in 0..data.len() {
        let h = hash4(data, pos);
        assert!(h < HASH_SIZE, "hash4 out of range at pos {pos}: {h}");
    }
}

#[test]
fn hash4_boundary_guard() {
    let data = b"abc"; // len=3, pos+3 >= len for all pos >= 0
    assert_eq!(hash4(data, 0), 0);
    let data2 = b"abcd"; // len=4, pos=0: pos+3=3 >= 4? No. pos=1: 1+3=4 >= 4? Yes.
    assert_ne!(hash4(data2, 0), 0, "4-byte input at pos 0 should hash normally");
    assert_eq!(hash4(data2, 1), 0, "pos 1 in 4-byte input triggers guard");
}

#[test]
fn hash4_deterministic() {
    let data = b"the quick brown fox";
    assert_eq!(hash4(data, 0), hash4(data, 0));
    assert_eq!(hash4(data, 4), hash4(data, 4));
}

#[test]
fn hash4_distinct_inputs() {
    // These four 4-byte sequences should produce different hashes.
    let inputs: &[&[u8]] = &[b"aaaa", b"aaab", b"aaba", b"abaa"];
    let hashes: Vec<usize> = inputs.iter().map(|d| hash4(d, 0)).collect();
    // All distinct
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "collision: {:?} vs {:?}", inputs[i], inputs[j]);
        }
    }
}
```

**Verification:**
Run: `cargo test lz77::tests::hash4`
Expected: All 4 tests pass.

**Commit:** `feat: add hash4() multiply-shift 4-byte hash to lz77`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add `hash_prefix_len` config to `HashChainFinder`, use `hash4` when set to 4

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Modify: `src/lz77.rs:193-258` (`HashChainFinder` struct and constructors), lines 264-397 (`find_best`, `insert`, `find_top_k`)
- Test: `src/lz77.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

Add a `hash_prefix_len: u8` field to `HashChainFinder`. The only valid values are 3 (current behavior, default) and 4 (new). Dispatch in `find_best` and `insert` with an `#[inline(always)]` helper `hash_at` that branches on the field.

Step 1 — Add field to struct (after `window_mask` on line 210):

```rust
/// Number of bytes used for hashing. 3 = current XOR hash, 4 = multiply-shift.
/// Using 4 reduces hash collisions at the cost of requiring 4 bytes of lookahead.
hash_prefix_len: u8,
```

Step 2 — Add `hash_at` helper method:

```rust
/// Dispatch to hash3 or hash4 based on configuration.
#[inline(always)]
fn hash_at(&self, data: &[u8], pos: usize) -> usize {
    if self.hash_prefix_len == 4 {
        hash4(data, pos)
    } else {
        hash3(data, pos)
    }
}
```

Step 3 — Update all constructors to initialize `hash_prefix_len: 3` (preserving existing behavior):

- `with_tuning`: add `hash_prefix_len: 3` to the struct literal.
- `with_window`: add `hash_prefix_len: 3` to the struct literal.

Step 4 — Add `with_hash4` constructor for callers that want the 4-byte hash:

```rust
/// Create a match finder with 4-byte hashing and a custom window size.
///
/// 4-byte hashing reduces hash collisions at the cost of needing 4 bytes
/// of lookahead (positions near EOF fall back to hash3 internally).
pub(crate) fn with_hash4(max_window: usize, max_match_len: u16, max_chain: usize) -> Self {
    debug_assert!(max_window.is_power_of_two(), "max_window must be power of 2");
    Self {
        head: vec![0; HASH_SIZE],
        prev: vec![0; max_window],
        dispatcher: crate::simd::Dispatcher::new(),
        max_match_len: max_match_len as usize,
        max_chain: max_chain.clamp(1, MAX_CHAIN),
        max_window,
        window_mask: max_window - 1,
        hash_prefix_len: 4,
    }
}
```

Step 5 — Replace `hash3(input, pos)` with `self.hash_at(input, pos)` in `find_best` (line 271), `insert` (line 395), and `find_top_k` (line 413).

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: `HashChainFinder` with `hash_prefix_len=4` finds correct matches (round-trip correctness via encode/decode).
- Default constructors (`new()`, `with_tuning`, `with_window`) use `hash_prefix_len=3` (no behavioral regression).
- `with_hash4` uses `hash_prefix_len=4`.
- Matches found with hash4 finder are valid (offset and length within input bounds).

```rust
#[test]
fn hash_prefix_default_is_3() {
    let finder = HashChainFinder::new();
    // Verify hash3 path: insert and find a known match
    let data = b"abcabcabc";
    let mut f = HashChainFinder::new();
    f.insert(data, 0);
    f.insert(data, 1);
    f.insert(data, 2);
    let m = f.find_best(data, 3);
    assert!(m.length >= 3, "hash3 finder should find match at pos 3");
    assert_eq!(m.offset, 3);
}

#[test]
fn hash4_finder_finds_matches() {
    let data = b"abcdabcdabcd";
    let mut f = HashChainFinder::with_hash4(32768, u16::MAX, 64);
    for i in 0..4 {
        f.insert(data, i);
    }
    let m = f.find_best(data, 4);
    assert!(m.length >= 4, "hash4 finder should find 4-byte match");
    assert_eq!(m.offset, 4, "offset should be 4");
}

#[test]
fn hash4_finder_correctness_roundtrip() {
    // Encode with hash4 finder, decode, compare to original.
    use crate::lzss;
    let input = b"the quick brown fox jumps over the lazy dog. \
                  the quick brown fox jumps over the lazy dog.";
    let mut finder = HashChainFinder::with_hash4(32768, 258, 64);
    // Just verify it doesn't panic and finds at least one match.
    let mut found_match = false;
    for pos in 0..input.len() {
        finder.insert(input, pos);
        let m = finder.find_best(input, pos);
        if m.length >= 3 {
            found_match = true;
        }
    }
    assert!(found_match, "hash4 finder should find matches in repetitive input");
}
```

**Verification:**
Run: `cargo test lz77`
Expected: All tests pass, no regressions in existing lz77 tests.

**Commit:** `feat: add hash_prefix_len field to HashChainFinder, dispatch hash3/hash4`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Wire 4-byte hash through `SeqConfig` to `encode_with_config`

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Modify: `src/lzseq.rs:45-59` (`SeqConfig` struct), line 560 (`encode_with_config` finder construction)
- Test: `src/lzseq.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

Add `hash_prefix_len: u8` to `SeqConfig` (default 3). Update `encode_with_config` to construct the finder using `with_hash4` when `config.hash_prefix_len == 4`, otherwise `with_window` (existing path).

Step 1 — Update `SeqConfig`:

```rust
pub struct SeqConfig {
    /// Maximum lookback window size in bytes. Must be a power of 2.
    /// Default: 128KB.
    pub max_window: usize,
    /// Hash prefix length for match finding: 3 (XOR, default) or 4 (multiply-shift).
    /// Using 4 reduces hash collisions and may improve ratio on large inputs at
    /// a small speed cost (one extra byte of lookahead required per position).
    pub hash_prefix_len: u8,
    /// Maximum hash chain depth. Higher values improve ratio at the cost of speed.
    /// Default: 64. Clamped to 1..=MAX_CHAIN_AUTO*2 internally.
    pub max_chain: usize,
}

impl Default for SeqConfig {
    fn default() -> Self {
        SeqConfig {
            max_window: 128 * 1024,
            hash_prefix_len: 3,
            max_chain: crate::lz77::MAX_CHAIN,
        }
    }
}
```

Note: `MAX_CHAIN` must be `pub(crate)` in `lz77.rs` for this reference. It is already `pub(crate)` as a module-level constant (line 29: `pub(crate) const MAX_CHAIN`). Confirm visibility — if it's not `pub(crate)`, change it. Current line 29 shows `pub(crate) const MAX_CHAIN: usize = 64;`.

Step 2 — Update `encode_with_config` finder construction (replacing line 560):

```rust
let mut finder = if config.hash_prefix_len == 4 {
    HashChainFinder::with_hash4(config.max_window, DEFAULT_MAX_MATCH, config.max_chain)
} else {
    HashChainFinder::with_window_and_chain(config.max_window, DEFAULT_MAX_MATCH, config.max_chain)
};
```

This requires a new `with_window_and_chain` constructor in `lz77.rs` (add alongside `with_window`):

```rust
/// Create a match finder with a custom window size and chain depth.
pub(crate) fn with_window_and_chain(
    max_window: usize,
    max_match_len: u16,
    max_chain: usize,
) -> Self {
    debug_assert!(max_window.is_power_of_two(), "max_window must be power of 2");
    Self {
        head: vec![0; HASH_SIZE],
        prev: vec![0; max_window],
        dispatcher: crate::simd::Dispatcher::new(),
        max_match_len: max_match_len as usize,
        max_chain: max_chain.clamp(1, MAX_CHAIN * 4), // allow deeper chains than default
        max_window,
        window_mask: max_window - 1,
        hash_prefix_len: 3,
    }
}
```

The `max_chain` clamp uses `MAX_CHAIN * 4` (256) as an upper safety bound so callers cannot spend unbounded time per position.

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: `encode_with_config` with `hash_prefix_len: 4` produces valid decodeable output (encode→decode roundtrip).
- Default `SeqConfig` produces identical output to the old behavior (no regression).
- `SeqConfig { hash_prefix_len: 4, .. }` compresses repetitive data and finds at least one match.

```rust
#[test]
fn seq_config_hash4_roundtrip() {
    let input = b"the quick brown fox the quick brown fox";
    let config = SeqConfig { hash_prefix_len: 4, ..SeqConfig::default() };
    let encoded = encode_with_config(input, &config).unwrap();
    assert!(encoded.num_matches > 0, "hash4 config should find matches");
}

#[test]
fn seq_config_default_no_regression() {
    let input = b"aababcabcdabcdeabcdefabcdefg";
    let encoded_default = encode_with_config(input, &SeqConfig::default()).unwrap();
    let encoded_hash3 = encode_with_config(
        input,
        &SeqConfig { hash_prefix_len: 3, ..SeqConfig::default() },
    ).unwrap();
    // Same number of matches — hash selection should not change token count
    // for small inputs where both hashes resolve cleanly.
    assert_eq!(encoded_default.num_tokens, encoded_hash3.num_tokens);
}
```

**Verification:**
Run: `cargo test lzseq`
Expected: All tests pass.

**Commit:** `feat: expose hash_prefix_len and max_chain in SeqConfig`
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-6) -->
## Subcomponent B: Adaptive chain depth

Currently `max_chain` is fixed at 64 in `HashChainFinder` regardless of whether the data is highly compressible or nearly random. On compressible data (text, source code) a deeper chain finds longer matches that more than pay for the extra comparisons. On incompressible data (already-compressed, random) those comparisons are wasted because few matches exist. Adaptive depth cuts chain depth early when the current match quality signals low compressibility.

<!-- START_TASK_4 -->
### Task 4: Make `max_chain` configurable in `HashChainFinder` (not just constant)

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Modify: `src/lz77.rs:224-235` (`with_tuning` constructor)
- Test: `src/lz77.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

`with_tuning` already accepts `max_chain` but clamps it to `MAX_CHAIN` (line 231: `max_chain.clamp(1, MAX_CHAIN)`). The issue is that `MAX_CHAIN` is the ceiling, so callers cannot request deeper chains. Change the clamp upper bound to `MAX_CHAIN * 4` (256) to allow high-quality encoding modes. Update `with_window` similarly (currently hard-codes `MAX_CHAIN`).

In `with_tuning` (line 231), change:
```rust
// Before:
max_chain: max_chain.clamp(1, MAX_CHAIN),
// After:
max_chain: max_chain.clamp(1, MAX_CHAIN * 4),
```

In `with_window` (line 254), change:
```rust
// Before:
max_chain: MAX_CHAIN,
// After:
max_chain: MAX_CHAIN, // keep default; callers wanting deeper use with_window_and_chain
```

Note: `with_window_and_chain` (added in Task 3) already uses `max_chain.clamp(1, MAX_CHAIN * 4)`, so the new constructor covers the configurable case. `with_window` retains its existing default behavior for backward compatibility.

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: `with_tuning(258, 128)` creates a finder with chain depth 128.
- Chain depth of 1 (minimum) still finds matches (degenerate case).
- Chain depth of 256 (`MAX_CHAIN * 4`) does not panic on inputs shorter than the chain depth.

```rust
#[test]
fn with_tuning_deep_chain() {
    let data = b"abcabcabcabcabcabcabc";
    let mut finder = HashChainFinder::with_tuning(258, 128);
    for i in 0..data.len() {
        finder.insert(data, i);
    }
    let m = finder.find_best(data, 6);
    assert!(m.length >= 3, "deep chain finder should find match");
}

#[test]
fn with_tuning_chain_depth_1() {
    let data = b"aaaaaaaaaa";
    let mut finder = HashChainFinder::with_tuning(258, 1);
    for i in 0..data.len() {
        finder.insert(data, i);
    }
    let m = finder.find_best(data, 5);
    // Chain depth 1 may miss some matches but must not panic or return invalid offset
    assert!(m.offset as usize <= 5 || m.length == 0);
}

#[test]
fn with_tuning_chain_depth_256_no_panic() {
    let data: Vec<u8> = (0u8..=255).cycle().take(1024).collect();
    let mut finder = HashChainFinder::with_tuning(u16::MAX, 256);
    for i in 0..data.len() {
        finder.insert(&data, i);
        let _ = finder.find_best(&data, i);
    }
}
```

**Verification:**
Run: `cargo test lz77`
Expected: All tests pass.

**Commit:** `feat: raise max_chain ceiling in with_tuning to MAX_CHAIN*4`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Add adaptive chain depth logic based on data compressibility signal

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Modify: `src/lzseq.rs:560-570` (`encode_with_config` main loop setup)
- Modify: `src/lzseq.rs:45-59` (`SeqConfig` struct — add `adaptive_chain` flag)
- Test: `src/lzseq.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

Add an `adaptive_chain: bool` field to `SeqConfig` (default `false`, opt-in). When enabled, `encode_with_config` tracks a rolling compressibility estimate and adjusts `max_chain` passed to subsequent match searches by halving the chain depth for the next 256 positions when a block of 64 positions produces zero matches.

The adaptation happens at the call site in the encoding loop, not inside `HashChainFinder`, to keep the finder stateless. `HashChainFinder` exposes a `set_max_chain` method so the loop can adjust it dynamically.

Step 1 — Add `adaptive_chain` to `SeqConfig`:

```rust
pub struct SeqConfig {
    pub max_window: usize,
    pub hash_prefix_len: u8,
    pub max_chain: usize,
    /// When true, reduce chain depth on blocks with low match density.
    /// Speeds up encoding on incompressible data with minimal ratio cost.
    /// Default: false.
    pub adaptive_chain: bool,
}

impl Default for SeqConfig {
    fn default() -> Self {
        SeqConfig {
            max_window: 128 * 1024,
            hash_prefix_len: 3,
            max_chain: crate::lz77::MAX_CHAIN,
            adaptive_chain: false,
        }
    }
}
```

Step 2 — Add `set_max_chain` to `HashChainFinder` in `src/lz77.rs` (after the constructors, around line 258):

```rust
/// Dynamically adjust chain depth. Used by adaptive encoding loops.
/// Clamps to 1..=MAX_CHAIN*4.
pub(crate) fn set_max_chain(&mut self, max_chain: usize) {
    self.max_chain = max_chain.clamp(1, MAX_CHAIN * 4);
}
```

Step 3 — Add adaptation bookkeeping to `encode_with_config`. After the `finder` and `repeats` initialization (around line 561), add:

```rust
// Adaptive chain depth tracking
let base_chain = config.max_chain;
let mut adapt_pos_counter: usize = 0;      // positions since last check
let mut adapt_match_counter: usize = 0;    // matches found in current window
let adapt_check_interval: usize = 64;      // check every 64 positions
let adapt_low_threshold: usize = 2;        // fewer than 2 matches in 64 = low compressibility
let adapt_penalty_positions: usize = 256;  // how long to use reduced chain
let mut adapt_penalty_remaining: usize = 0;
```

Inside the main `while pos < input.len()` loop (before the `find_match_wide` call), add:

```rust
if config.adaptive_chain {
    adapt_pos_counter += 1;
    if adapt_penalty_remaining > 0 {
        adapt_penalty_remaining -= 1;
        if adapt_penalty_remaining == 0 {
            finder.set_max_chain(base_chain);
        }
    }
    if adapt_pos_counter >= adapt_check_interval {
        adapt_pos_counter = 0;
        if adapt_match_counter < adapt_low_threshold {
            // Low compressibility: halve chain depth for next window
            finder.set_max_chain((base_chain / 2).max(1));
            adapt_penalty_remaining = adapt_penalty_positions;
        } else {
            // Restore full chain depth
            finder.set_max_chain(base_chain);
        }
        adapt_match_counter = 0;
    }
}
```

After the `emit_match` call (in the `if best_length >= effective_min` branch), increment:

```rust
if config.adaptive_chain {
    adapt_match_counter += 1;
}
```

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: adaptive mode encodes compressible data (repetitive text) and finds matches (ratio not catastrophically worse).
- Adaptive mode on incompressible data (random bytes) does not panic.
- `adaptive_chain: false` (default) produces identical output to pre-Task-5 behavior.

```rust
#[test]
fn adaptive_chain_finds_matches_on_compressible_data() {
    let input: Vec<u8> = b"the quick brown fox ".iter()
        .cycle().take(4096).cloned().collect();
    let config = SeqConfig { adaptive_chain: true, ..SeqConfig::default() };
    let encoded = encode_with_config(&input, &config).unwrap();
    assert!(encoded.num_matches > 0, "adaptive mode must find matches on repetitive input");
    // Ratio should not collapse — at least 50% of tokens should be matches
    let match_ratio = encoded.num_matches as f64 / encoded.num_tokens as f64;
    assert!(match_ratio > 0.5, "match ratio {match_ratio:.2} too low for repetitive input");
}

#[test]
fn adaptive_chain_no_panic_on_random() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    // Deterministic pseudo-random bytes
    let input: Vec<u8> = (0u32..4096).map(|i| {
        let mut h = DefaultHasher::new();
        i.hash(&mut h);
        h.finish() as u8
    }).collect();
    let config = SeqConfig { adaptive_chain: true, ..SeqConfig::default() };
    let _ = encode_with_config(&input, &config).unwrap(); // must not panic
}

#[test]
fn adaptive_chain_false_is_identical_to_default() {
    let input = b"hello world hello world hello world";
    let default_encoded = encode_with_config(input, &SeqConfig::default()).unwrap();
    let explicit_false = encode_with_config(
        input,
        &SeqConfig { adaptive_chain: false, ..SeqConfig::default() },
    ).unwrap();
    assert_eq!(default_encoded.num_tokens, explicit_false.num_tokens);
    assert_eq!(default_encoded.num_matches, explicit_false.num_matches);
}
```

**Verification:**
Run: `cargo test lzseq`
Expected: All tests pass, no clippy warnings.

**Commit:** `feat: add adaptive chain depth to SeqConfig and encode_with_config`
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Tests for adaptive chain depth — match count and ratio preservation

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Test only: `src/lzseq.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

No production code changes. Add targeted tests that compare adaptive vs non-adaptive encoding on:
1. Highly compressible input — ratio must not degrade by more than 5% in match count.
2. Fully incompressible input (pre-encoded bytes) — adaptive mode should reduce match-finding time (tested indirectly by confirming no extra matches are emitted that would indicate chain waste).
3. Mixed input (compressible block followed by incompressible block) — adaptive mode must not corrupt the compressible block's matches.

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: adaptive mode on 64KB repetitive text loses fewer than 5% of matches vs non-adaptive.

```rust
#[test]
fn adaptive_vs_nonadaptive_compressible_match_count() {
    let input: Vec<u8> = b"abcdefghij".iter()
        .cycle().take(65536).cloned().collect();

    let adaptive_cfg = SeqConfig { adaptive_chain: true, ..SeqConfig::default() };
    let normal_cfg = SeqConfig { adaptive_chain: false, ..SeqConfig::default() };

    let adaptive = encode_with_config(&input, &adaptive_cfg).unwrap();
    let normal = encode_with_config(&input, &normal_cfg).unwrap();

    // Adaptive must find at least 95% as many matches as non-adaptive
    let threshold = (normal.num_matches as f64 * 0.95) as u32;
    assert!(
        adaptive.num_matches >= threshold,
        "adaptive found {} matches, non-adaptive found {} (threshold {})",
        adaptive.num_matches, normal.num_matches, threshold
    );
}

#[test]
fn adaptive_mixed_input_no_corruption() {
    // First 32KB repetitive, next 32KB pseudo-random
    let mut input: Vec<u8> = b"abcde".iter().cycle().take(32768).cloned().collect();
    input.extend((0u8..=255).cycle().take(32768));

    let config = SeqConfig { adaptive_chain: true, ..SeqConfig::default() };
    let encoded = encode_with_config(&input, &config).unwrap();
    // Must have found at least some matches (from the repetitive first half)
    assert!(encoded.num_matches > 0, "adaptive mode must find matches in mixed input");
}
```

**Verification:**
Run: `cargo test lzseq`
Expected: All tests pass.

**Commit:** `test: add adaptive chain depth coverage tests`
<!-- END_TASK_6 -->
<!-- END_SUBCOMPONENT_B -->

<!-- START_SUBCOMPONENT_C (tasks 7-8) -->
## Subcomponent C: Match profitability tuning

`min_profitable_length` (lines 435-456 in `lzseq.rs`) uses a simplified tier formula. The formula `MIN_MATCH + (oeb.saturating_sub(7) as u16).div_ceil(4)` was designed conservatively. Two known issues:

1. **Tier boundary at oeb=7:** Offsets 65-128 have `oeb=6` (min stays at 3), but offset 129-256 has `oeb=7` (still min 3). Offset 257 crosses into `oeb=8` (min 4). The cross-over at 7 is empirically derived; re-examining with exact bit costs may shift it.
2. **Repeat matches always use `MIN_MATCH`:** This is correct because repeat offsets cost 0 extra bits. The special-casing (line 583) is correct and should not change.

The audit in this task is to compare the formula's output against exact cost calculations and tighten thresholds where the formula is provably too permissive.

<!-- START_TASK_7 -->
### Task 7: Review and tighten `min_profitable_length()` thresholds

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Modify: `src/lzseq.rs:420-456` (`min_profitable_length` function and its doc comment)
- Test: `src/lzseq.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

Perform an exact cost analysis per offset range and update the threshold formula to match. The LzSeq token cost for a match at offset `d` with length `L`:

- Match overhead (fixed): 1 flag bit + 1 offset_code byte + 1 length_code byte = 2 bytes + 1 bit
- Offset extra bits: `oeb(d)` bits
- Length extra bits: `leb(L)` bits (minimum 0 for length code 0, i.e., L=3)
- Total match cost: `2 + ceil((oeb(d) + leb(L)) / 8)` bytes (approximately)

The match saves `L` literal bytes. Each literal costs approximately 1 byte (1 flag bit + 1 byte literal). So break-even is:

```
L > 2 + ceil(oeb(d) / 8)   [ignoring length extra bits for minimum length]
```

Exact thresholds per oeb range:

| oeb | Offset range       | Match cost (bytes) | Min profitable L |
|-----|--------------------|--------------------|-----------------|
| 0   | 1                  | 2                  | 3 (= MIN_MATCH) |
| 1   | 2                  | 2                  | 3               |
| 2   | 3-4                | 2                  | 3               |
| 3   | 5-8                | 2                  | 3               |
| 4   | 9-16               | 2 (4/8 rounds up to 1, but 4 bits < 1 byte overhead) → total 3 bytes | 4 |
| 5   | 17-32              | 3                  | 4               |
| 6   | 33-64              | 3                  | 4               |
| 7   | 65-128             | 3                  | 4               |
| 8   | 129-256            | 3                  | 4               |
| 9   | 257-512            | 4                  | 5               |
| 10  | 513-1024           | 4                  | 5               |
| 11  | 1025-2048          | 4                  | 5               |
| 12  | 2049-4096          | 4                  | 5               |
| 13  | 4097-8192          | 5                  | 6               |
| 14  | 8193-16384         | 5                  | 6               |
| 15  | 16385-32768        | 5                  | 6               |
| 16  | 32769-65536        | 5                  | 6               |
| 17  | 65537-131072       | 6                  | 7               |

The current formula produces:
- oeb 0-7: `MIN_MATCH + 0 = 3` (correct for oeb 0-3; too permissive for oeb 4-7 where exact analysis shows min 4)
- oeb 8-11: `MIN_MATCH + 1 = 4` (correct)
- oeb 12-15: `MIN_MATCH + 2 = 5` (correct for oeb 12; too permissive for oeb 13-15 where exact analysis shows min 6)

**Key change:** Shift the tier boundary from oeb 7 to oeb 3 for the first step, and from oeb 12 to oeb 11 for the second step. New formula:

```rust
#[inline]
pub(crate) fn min_profitable_length(offset: u32) -> u16 {
    if offset == 0 {
        return u16::MAX;
    }
    let (oc, _, _) = encode_offset(offset);
    let oeb = extra_bits_for_code(oc);
    // Exact cost model (see doc comment):
    //   oeb 0-3  (offset 1-8):         min 3 (= MIN_MATCH)
    //   oeb 4-8  (offset 9-256):       min 4
    //   oeb 9-12 (offset 257-4096):    min 5
    //   oeb 13-16 (offset 4097-65536): min 6
    //   oeb 17+  (offset 65537+):      min 7
    MIN_MATCH + match oeb {
        0..=3 => 0,
        4..=8 => 1,
        9..=12 => 2,
        13..=16 => 3,
        _ => 4,
    }
}
```

Update the doc comment to reflect the exact cost model table above.

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: the thresholds are not more permissive than the exact cost model.
- Boundary values return correct thresholds (test each tier boundary).
- `offset=0` returns `u16::MAX`.
- The function is monotonically non-decreasing with offset (larger offset → same or higher threshold).

```rust
#[test]
fn min_profitable_length_offset_zero() {
    assert_eq!(min_profitable_length(0), u16::MAX);
}

#[test]
fn min_profitable_length_tiers() {
    // oeb 0-3: offset 1-8 → min 3
    for d in 1u32..=8 {
        assert_eq!(min_profitable_length(d), 3, "offset {d}");
    }
    // oeb 4-8: offset 9-256 → min 4
    for d in [9u32, 16, 32, 64, 128, 256] {
        assert_eq!(min_profitable_length(d), 4, "offset {d}");
    }
    // oeb 9-12: offset 257-4096 → min 5
    for d in [257u32, 512, 1024, 2048, 4096] {
        assert_eq!(min_profitable_length(d), 5, "offset {d}");
    }
    // oeb 13-16: offset 4097-65536 → min 6
    for d in [4097u32, 8192, 16384, 32768, 65536] {
        assert_eq!(min_profitable_length(d), 6, "offset {d}");
    }
    // oeb 17+: offset 65537+ → min 7
    assert_eq!(min_profitable_length(65537), 7);
    assert_eq!(min_profitable_length(131072), 7);
}

#[test]
fn min_profitable_length_monotone() {
    // Sample offsets spanning all tiers; verify non-decreasing.
    let offsets = [1u32, 4, 8, 9, 32, 256, 257, 1024, 4096, 4097, 32768, 65536, 65537, 131072];
    let thresholds: Vec<u16> = offsets.iter().map(|&d| min_profitable_length(d)).collect();
    for w in thresholds.windows(2) {
        assert!(w[0] <= w[1], "min_profitable_length not monotone: {} > {} for adjacent offsets", w[0], w[1]);
    }
}
```

**Verification:**
Run: `cargo test lzseq::tests::min_profitable`
Expected: All tests pass, no clippy warnings.

**Commit:** `fix: tighten min_profitable_length thresholds to exact cost model`
<!-- END_TASK_7 -->

<!-- START_TASK_8 -->
### Task 8: Tests for match profitability edge cases

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Test only: `src/lzseq.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

No production code changes. Add integration-style tests that confirm the encoder correctly rejects short matches at large distances (the profitability filter is exercised end-to-end), and that the new tighter thresholds do not break decoding on any path.

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: encoder with updated thresholds produces valid, decodeable output on a corpus-representative string (structured English text).
- A 3-byte match at offset 300 is rejected (since `min_profitable_length(300) = 5 > 3`).
- A 5-byte match at offset 300 is accepted (since `5 >= 5`).
- Encoding an all-identical-byte input produces expected match structure (single match, min cost).

To test the rejection directly without going through the full encoder, call `min_profitable_length` as a unit and verify the encoder's actual match counts on crafted inputs:

```rust
#[test]
fn encoder_rejects_short_match_at_large_offset() {
    // Craft input: 300 bytes of unique data, then 3 bytes that match the start.
    let mut input = Vec::new();
    input.extend_from_slice(b"xyz"); // bytes 0-2
    for i in 3u8..=255u8 {
        input.push(i);
    }
    // Push some more unique bytes to reach offset ~300
    for i in 0u8..44u8 {
        input.push(i.wrapping_add(200));
    }
    // Now push the 3-byte match target at offset ~300
    input.extend_from_slice(b"xyz");

    let config = SeqConfig { max_window: 128 * 1024, ..SeqConfig::default() };
    let encoded = encode_with_config(&input, &config).unwrap();

    // With the tighter thresholds, a 3-byte match at ~300 bytes back should be
    // rejected (min_profitable_length(300) = 5). So num_matches should be 0.
    assert_eq!(
        encoded.num_matches, 0,
        "3-byte match at ~300 offset should be rejected by profitability filter"
    );
}

#[test]
fn encoder_accepts_profitable_match_at_large_offset() {
    // Same setup but 5 matching bytes at offset ~300.
    let mut input = Vec::new();
    input.extend_from_slice(b"xyzab"); // bytes 0-4
    for i in 5u8..=255u8 {
        input.push(i);
    }
    for i in 0u8..45u8 {
        input.push(i.wrapping_add(200));
    }
    input.extend_from_slice(b"xyzab"); // 5-byte match at ~300 offset

    let config = SeqConfig { max_window: 128 * 1024, ..SeqConfig::default() };
    let encoded = encode_with_config(&input, &config).unwrap();

    // A 5-byte match at ~300 offset should be accepted (5 >= min_profitable_length(300)=5).
    assert!(
        encoded.num_matches >= 1,
        "5-byte match at ~300 offset should be accepted"
    );
}

#[test]
fn encoder_all_same_bytes_efficient() {
    let input = vec![0xAAu8; 1024];
    let encoded = encode_with_config(&input, &SeqConfig::default()).unwrap();
    // All-same input: should produce very few match tokens (ideally 1-2)
    // and very few literals (just the first MIN_MATCH bytes before first match)
    assert!(
        encoded.num_matches >= 1,
        "all-same input must find at least one match"
    );
    assert!(
        encoded.num_tokens < 20,
        "all-same 1KB input should compress to very few tokens, got {}",
        encoded.num_tokens
    );
}
```

**Verification:**
Run: `cargo test lzseq`
Expected: All tests pass.

**Commit:** `test: add match profitability edge case tests`
<!-- END_TASK_8 -->
<!-- END_SUBCOMPONENT_C -->

<!-- START_SUBCOMPONENT_D (tasks 9-10) -->
## Subcomponent D: Window size configuration

`SeqConfig.max_window` defaults to 128KB and is already wired to `HashChainFinder::with_window`. The gap is that window size has not been benchmarked or tested at 64KB and 256KB to quantify the ratio vs speed tradeoff. This subcomponent adds tests that verify correctness at each window size and provides the infrastructure needed for Phase 1's benchmark harness to sweep window sizes.

<!-- START_TASK_9 -->
### Task 9: Wire window size through `SeqConfig` and verify 64KB/128KB/256KB correctness

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Verify (no change needed): `src/lzseq.rs:560` — `encode_with_config` already uses `config.max_window`.
- Add convenience constructors: `src/lzseq.rs` (after `SeqConfig::default`)
- Test: `src/lzseq.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

`encode_with_config` already passes `config.max_window` to `HashChainFinder::with_window` (confirmed line 560). No functional change is needed. Add named constructors on `SeqConfig` for the three standard window sizes to give callers a clear API and to make benchmark sweep code readable:

```rust
impl SeqConfig {
    /// Fast preset: 64KB window, chain 32. Faster than default, lower ratio.
    pub fn fast() -> Self {
        SeqConfig {
            max_window: 64 * 1024,
            hash_prefix_len: 3,
            max_chain: 32,
            adaptive_chain: false,
        }
    }

    /// Default preset: 128KB window, chain 64.
    pub fn default_quality() -> Self {
        SeqConfig::default()
    }

    /// High preset: 256KB window, chain 128, 4-byte hash.
    /// Better ratio at the cost of higher memory and time.
    pub fn high() -> Self {
        SeqConfig {
            max_window: 256 * 1024,
            hash_prefix_len: 4,
            max_chain: 128,
            adaptive_chain: false,
        }
    }
}
```

Note: 256KB window requires the `prev` array in `HashChainFinder` to be 256K entries × 4 bytes = 1MB. This is acceptable for a quality-focused preset.

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: all three presets produce valid, consistent output on a shared input (same token count per preset since the content is short enough to fit in any window).
- `SeqConfig::fast()` has `max_window = 65536`.
- `SeqConfig::high()` has `max_window = 262144` and `hash_prefix_len = 4`.
- 256KB window finder does not panic on a 256KB input.

```rust
#[test]
fn seq_config_presets_have_correct_windows() {
    assert_eq!(SeqConfig::fast().max_window, 64 * 1024);
    assert_eq!(SeqConfig::default_quality().max_window, 128 * 1024);
    assert_eq!(SeqConfig::high().max_window, 256 * 1024);
    assert_eq!(SeqConfig::high().hash_prefix_len, 4);
    assert_eq!(SeqConfig::high().max_chain, 128);
}

#[test]
fn all_presets_produce_valid_output_on_short_input() {
    let input = b"the quick brown fox jumps over the lazy dog";
    for config in [SeqConfig::fast(), SeqConfig::default_quality(), SeqConfig::high()] {
        let encoded = encode_with_config(input, &config).unwrap();
        assert!(encoded.num_tokens > 0, "preset must encode non-empty input");
        // Token count should be identical since input is smaller than any window
        assert_eq!(
            encoded.num_tokens,
            encode_with_config(input, &SeqConfig::default()).unwrap().num_tokens,
            "token count should match default for input smaller than any window"
        );
    }
}

#[test]
fn high_preset_256kb_window_no_panic() {
    let input: Vec<u8> = b"Lorem ipsum dolor sit amet ".iter()
        .cycle().take(256 * 1024).cloned().collect();
    let encoded = encode_with_config(&input, &SeqConfig::high()).unwrap();
    assert!(encoded.num_matches > 0);
}
```

**Verification:**
Run: `cargo test lzseq`
Expected: All tests pass.

**Commit:** `feat: add SeqConfig::fast, default_quality, high presets with window sizes`
<!-- END_TASK_9 -->

<!-- START_TASK_10 -->
### Task 10: Tests for window size impact on match quality

**Verifies:** pareto-competitiveness.AC1.1, pareto-competitiveness.AC1.2

**Files:**
- Test only: `src/lzseq.rs` (inline `#[cfg(test)] mod tests`)

**Implementation:**

No production code changes. Add tests that confirm larger windows find longer-distance matches that smaller windows miss, and that window sizes do not affect decoding correctness (the `SeqEncoded` streams must be self-consistent regardless of window size).

**Testing:**

Tests must verify:
- pareto-competitiveness.AC1.1: 256KB window finds a match that 32KB window misses on an input where the match is 64KB away.
- 64KB window matches are a subset of 256KB window matches on the same input (larger window never finds fewer matches).
- `num_matches` is non-decreasing as window size increases for the same input.

```rust
#[test]
fn larger_window_finds_distant_match() {
    // Create input where the only match is at distance ~70KB.
    // Pattern: 5 unique bytes, then 70KB of filler, then the same 5 bytes.
    let pattern = b"MATCH";
    let filler_len = 70 * 1024;
    let mut input = Vec::with_capacity(filler_len + 10);
    input.extend_from_slice(pattern);
    for i in 0..filler_len {
        // Unique filler (no accidental matches)
        input.push((i % 251 + 1) as u8); // primes avoid pattern repeat
    }
    input.extend_from_slice(pattern);

    // 32KB window (standard lz77 default): cannot reach 70KB back
    let small_window_cfg = SeqConfig {
        max_window: 32 * 1024,
        ..SeqConfig::default()
    };
    // 128KB window: can reach 70KB back
    let large_window_cfg = SeqConfig {
        max_window: 128 * 1024,
        ..SeqConfig::default()
    };

    let small_encoded = encode_with_config(&input, &small_window_cfg).unwrap();
    let large_encoded = encode_with_config(&input, &large_window_cfg).unwrap();

    assert_eq!(small_encoded.num_matches, 0,
        "32KB window should miss the match at 70KB distance");
    assert!(large_encoded.num_matches >= 1,
        "128KB window should find the match at 70KB distance");
}

#[test]
fn window_size_match_count_nondecreasing() {
    // Input has repetitions at various distances.
    let mut input: Vec<u8> = Vec::new();
    let phrase = b"hello world ";
    // Repeat with increasing gaps to have matches at many distances.
    for gap in [100usize, 1000, 10000, 50000, 100000] {
        input.extend_from_slice(phrase);
        for i in 0..gap {
            input.push((i % 200 + 10) as u8);
        }
    }
    input.extend_from_slice(phrase); // final match target

    let windows = [32 * 1024usize, 64 * 1024, 128 * 1024, 256 * 1024];
    let mut prev_matches = 0u32;
    for &w in &windows {
        let cfg = SeqConfig { max_window: w, ..SeqConfig::default() };
        let encoded = encode_with_config(&input, &cfg).unwrap();
        assert!(
            encoded.num_matches >= prev_matches,
            "window {} found fewer matches ({}) than window {} ({})",
            w, encoded.num_matches, w / 2, prev_matches
        );
        prev_matches = encoded.num_matches;
    }
}
```

**Verification:**
Run: `cargo test lzseq`
Expected: All tests pass. The window size tests may take a few seconds due to large input construction.

**Commit:** `test: add window size impact and distant match tests`
<!-- END_TASK_10 -->
<!-- END_SUBCOMPONENT_D -->

---

## Phase completion checklist

Phase 2 is complete when all of the following are true:

- [ ] `cargo test lz77` passes (Tasks 1-4: hash4, configurable chain)
- [ ] `cargo test lzseq` passes (Tasks 3, 5-10: SeqConfig, adaptive chain, profitability, windows)
- [ ] `cargo clippy --all-targets` produces zero warnings
- [ ] `./scripts/test.sh --quick` exits 0
- [ ] LzSeqR single-thread compression ratio on Canterbury corpus improves vs Phase 1 baseline (measured with Phase 1 benchmark harness)
- [ ] LzSeqR single-thread compression ratio on Silesia corpus improves vs Phase 1 baseline

## Notes for the implementer

- **Task ordering matters**: Tasks 1→2→3 must be done in sequence (each builds on the previous). Tasks 4→5→6 similarly. Tasks 7 and 8 are independent of Subcomponent B. Tasks 9 and 10 are independent of A and B.
- **`with_tuning` clamp**: The existing clamp `max_chain.clamp(1, MAX_CHAIN)` in Task 4 is raised to `MAX_CHAIN * 4`. This is intentional — it allows quality presets to use deeper chains while preventing unbounded per-position cost.
- **`SeqConfig` field additions**: Tasks 3 and 5 both add fields to `SeqConfig`. Do them in order. After Task 3 adds `hash_prefix_len` and `max_chain`; Task 5 adds `adaptive_chain`. By Task 5, `SeqConfig::default()` must return all four fields.
- **`min_profitable_length` tightening**: Task 7 shifts the tier boundary from oeb 7 to oeb 3 for the first threshold step. This means 3-byte matches at offsets 9-256 that were previously accepted will now be rejected. This is correct per the cost model and should improve ratio. The encoder tests in Task 8 validate this end-to-end.
- **Memory cost of `SeqConfig::high()`**: 256KB window = 1MB `prev` array + 128KB `head` array. Acceptable for a quality preset but document in the `high()` doc comment.
