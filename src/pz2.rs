//! pz2 — clean-slate decode-first block codec (prototype).
//!
//! Implements the block format from `docs/design-docs/clean-slate-codec.md`:
//! the shipped LzSeq parse (via `lzseq::tokenize_with_config`, byte-identical
//! parse decisions to `Lzf`) re-expressed as zstd-style **sequences**
//! `(literal_run_len, match_len, offset)` with:
//!
//! - **Literals** in an 8-lane huff0-style canonical Huffman: one shared
//!   length-limited (≤11 bit) table, eight independent LSB bitstreams decoded
//!   with a flat 2048-entry L1-resident table, interleaved for ILP. No
//!   inter-symbol entropy state — this is the design's P3 ("shortest critical
//!   path wins"). 8 lanes measured best on the M5 (4/6/8/12/16 swept).
//! - **Sequence codes** (literal-run / match-len / offset log2 buckets) as
//!   three more small-alphabet Huffman lanes decoded fused with the splice.
//!   Offsets use the shipped `RepeatOffsets` rep-cache.
//! - **Extra bits** in one raw LSB bit lane.
//! - Decode is splice-shaped: bulk literal-run copies from the pre-decoded
//!   literal buffer + `extend_from_within` match copies with
//!   exponential-doubling overlap handling (P7, "tokens shaped like memcpy").
//!
//! The prototype gate (design doc §5): single-thread decode ≥ 2× `Lzf`
//! (≥ ~850 MB/s on the M5) at ratio within ~1pp of `Lzf` with the same parse.
//! Measured by `examples/pz2_eval.rs`.
//!
//! Like `numeric`, the decoder is told `orig_len` by the caller (the
//! container stores it in the block table).

use crate::lz_token::LzToken;
use crate::lzseq::{self, extra_bits_for_offset_code, RepeatOffsets, SeqConfig};
use crate::{PzError, PzResult};

/// Minimum match length the LzSeq parse emits (bias for match-length codes).
const MIN_MATCH: u32 = 3;

/// Maximum Huffman code length. 2^11-entry decode table = 4 KB (L1-resident).
const MAX_CODE_LEN: u32 = 11;

/// Number of independent literal bitstream lanes. Swept on the M5
/// (4/6/8/12/16): 8 is the knee (+7% over 4 on text; 12/16 regress
/// slightly from register spill). Part of the wire format.
const NUM_LANES: usize = 8;

/// Literal section modes.
const LIT_RAW: u8 = 0;
const LIT_HUFF: u8 = 1;

/// Sequence-code stream modes.
const CODES_CONST: u8 = 0;
const CODES_HUFF: u8 = 1;

/// Sequence code alphabets are small (≤ 32 symbols): literal-run and
/// match-length log2 codes top out around 22, offset codes around 27.
const MAX_SEQ_CODE: u8 = 31;

// ---------------------------------------------------------------------------
// LSB-first bit IO
// ---------------------------------------------------------------------------

/// LSB-first bit writer: first bit written is the lowest bit of the first byte.
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    nbits: u32,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter {
            out: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }

    #[inline]
    fn write(&mut self, value: u32, bits: u8) {
        debug_assert!(bits <= 32);
        debug_assert!(bits == 32 || (value as u64) < (1u64 << bits));
        self.acc |= (value as u64) << self.nbits;
        self.nbits += bits as u32;
        while self.nbits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.nbits -= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

/// LSB-first bit reader tuned for the per-sequence extras path: one
/// branchless whole-byte refill per sequence (clamped near stream end),
/// then mask/shift reads. Strict errors on exhaustion, never over-reads.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    nbits: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            acc: 0,
            nbits: 0,
        }
    }

    /// Top up to ≥56 live bits (or stream end). Call once per sequence.
    #[inline(always)]
    fn refill(&mut self) {
        if self.pos + 8 <= self.data.len() {
            let w = u64::from_le_bytes(self.data[self.pos..self.pos + 8].try_into().unwrap());
            self.acc |= w << self.nbits;
            self.pos += ((63 - self.nbits) >> 3) as usize;
            self.nbits |= 56;
        } else {
            while self.nbits <= 56 && self.pos < self.data.len() {
                self.acc |= (self.data[self.pos] as u64) << self.nbits;
                self.pos += 1;
                self.nbits += 8;
            }
        }
    }

    #[inline(always)]
    fn read(&mut self, bits: u8) -> PzResult<u32> {
        debug_assert!(bits <= 32);
        if (bits as u32) > self.nbits {
            self.refill();
            if (bits as u32) > self.nbits {
                return Err(PzError::InvalidInput);
            }
        }
        let v = (self.acc & ((1u64 << bits) - 1)) as u32;
        self.acc >>= bits;
        self.nbits -= bits as u32;
        Ok(v)
    }
}

// ---------------------------------------------------------------------------
// Literal-run / match-length value coding (log2 bucket + extra bits)
// ---------------------------------------------------------------------------

/// Code a value: 0 → code 0 (no extra); v in [2^(k-1), 2^k) → code k with
/// k-1 extra bits holding v - 2^(k-1).
#[inline]
fn vcode(v: u32) -> (u8, u8, u32) {
    if v == 0 {
        (0, 0, 0)
    } else {
        let k = 32 - v.leading_zeros();
        (k as u8, (k - 1) as u8, v - (1u32 << (k - 1)))
    }
}

/// Extra-bit count implied by a value code.
#[inline]
fn vbits(code: u8) -> u8 {
    code.saturating_sub(1)
}

/// Inverse of [`vcode`]. `code` must be ≤ 31 (validated by callers).
#[inline]
fn vdecode(code: u8, extra: u32) -> u32 {
    if code == 0 {
        0
    } else {
        (1u32 << (code - 1)) + extra
    }
}

// ---------------------------------------------------------------------------
// Length-limited canonical Huffman
// ---------------------------------------------------------------------------

/// Compute Huffman code lengths for `counts`, limited to MAX_CODE_LEN by
/// halving counts and rebuilding until the tree fits (slightly suboptimal in
/// rare deep-tree cases, always a valid Kraft-exact Huffman tree).
///
/// Requires ≥ 2 symbols with nonzero count (callers fall back to raw mode).
fn huffman_lengths(counts: &[u32; 256]) -> [u8; 256] {
    let mut work: Vec<u64> = counts.iter().map(|&c| c as u64).collect();
    loop {
        let lengths = heap_lengths(&work);
        let max = lengths.iter().copied().max().unwrap_or(0);
        if (max as u32) <= MAX_CODE_LEN {
            let mut out = [0u8; 256];
            out.copy_from_slice(&lengths);
            return out;
        }
        for c in work.iter_mut() {
            if *c > 0 {
                *c = (*c + 1) >> 1;
            }
        }
    }
}

/// Plain heap Huffman → per-symbol code lengths (0 = absent).
fn heap_lengths(counts: &[u64]) -> Vec<u8> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    // Node arena: leaves for nonzero symbols, then internal nodes.
    // parent[i] = parent node index (usize::MAX = root/none yet).
    let mut node_count: Vec<u64> = Vec::new();
    let mut parent: Vec<usize> = Vec::new();
    let mut leaf_of_sym: Vec<Option<usize>> = vec![None; counts.len()];

    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    for (sym, &c) in counts.iter().enumerate() {
        if c > 0 {
            let idx = node_count.len();
            node_count.push(c);
            parent.push(usize::MAX);
            leaf_of_sym[sym] = Some(idx);
            heap.push(Reverse((c, idx)));
        }
    }
    let num_leaves = node_count.len();
    assert!(num_leaves >= 2, "huffman_lengths requires >= 2 symbols");

    while heap.len() > 1 {
        let Reverse((c1, i1)) = heap.pop().unwrap();
        let Reverse((c2, i2)) = heap.pop().unwrap();
        let idx = node_count.len();
        node_count.push(c1 + c2);
        parent.push(usize::MAX);
        parent[i1] = idx;
        parent[i2] = idx;
        heap.push(Reverse((c1 + c2, idx)));
    }

    let mut lengths = vec![0u8; counts.len()];
    for (sym, leaf) in leaf_of_sym.iter().enumerate() {
        if let Some(mut idx) = *leaf {
            let mut depth = 0u32;
            while parent[idx] != usize::MAX {
                idx = parent[idx];
                depth += 1;
            }
            lengths[sym] = depth as u8;
        }
    }
    lengths
}

/// Canonical code assignment shared by encoder and decoder.
///
/// Returns per-symbol `(lsb_code, len)` where `lsb_code` is the canonical
/// code bit-reversed to `len` bits, ready for LSB-first IO.
fn canonical_codes(lengths: &[u8; 256]) -> PzResult<[(u16, u8); 256]> {
    let mut bl_count = [0u32; (MAX_CODE_LEN + 1) as usize];
    for &l in lengths.iter() {
        if l as u32 > MAX_CODE_LEN {
            return Err(PzError::InvalidInput);
        }
        if l > 0 {
            bl_count[l as usize] += 1;
        }
    }
    // Kraft sum must be exact (complete tree) so the flat decode table has
    // no holes and every 11-bit pattern resolves to a symbol.
    let kraft: u64 = (1..=MAX_CODE_LEN)
        .map(|l| (bl_count[l as usize] as u64) << (MAX_CODE_LEN - l))
        .sum();
    if kraft != 1u64 << MAX_CODE_LEN {
        return Err(PzError::InvalidInput);
    }

    let mut next_code = [0u32; (MAX_CODE_LEN + 1) as usize];
    let mut code = 0u32;
    for l in 1..=MAX_CODE_LEN as usize {
        code = (code + bl_count[l - 1]) << 1;
        next_code[l] = code;
    }

    let mut out = [(0u16, 0u8); 256];
    for sym in 0..256 {
        let l = lengths[sym];
        if l == 0 {
            continue;
        }
        let c = next_code[l as usize];
        next_code[l as usize] += 1;
        let rev = (c.reverse_bits() >> (32 - l as u32)) as u16;
        out[sym] = (rev, l);
    }
    Ok(out)
}

/// Build the flat LSB decode table: index by the low 11 bits of the
/// bitstream, get `(sym << 4) | len`. Fixed-size so the hot loop's masked
/// index needs no bounds check.
fn build_decode_table(lengths: &[u8; 256]) -> PzResult<Box<[u16; 1 << MAX_CODE_LEN]>> {
    let codes = canonical_codes(lengths)?;
    let mut table = Box::new([0u16; 1 << MAX_CODE_LEN]);
    for (sym, &(rev, l)) in codes.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let entry = ((sym as u16) << 4) | (l as u16);
        let step = 1usize << l;
        let mut idx = rev as usize;
        while idx < table.len() {
            table[idx] = entry;
            idx += step;
        }
    }
    Ok(table)
}

// ---------------------------------------------------------------------------
// Multi-lane literal codec
// ---------------------------------------------------------------------------

/// Contiguous near-equal lane lengths (first `total % NUM_LANES` lanes get
/// one extra byte).
#[inline]
fn lane_lengths(total: usize) -> [usize; NUM_LANES] {
    let base = total / NUM_LANES;
    let rem = total % NUM_LANES;
    let mut out = [base; NUM_LANES];
    for slot in out.iter_mut().take(rem) {
        *slot += 1;
    }
    out
}

/// Encode literals into 4 independent Huffman bitstream lanes.
fn encode_lanes(lits: &[u8], codes: &[(u16, u8); 256]) -> [Vec<u8>; NUM_LANES] {
    let lens = lane_lengths(lits.len());
    let mut lanes: [Vec<u8>; NUM_LANES] = Default::default();
    let mut start = 0usize;
    for (lane, &n) in lanes.iter_mut().zip(lens.iter()) {
        let mut w = BitWriter::new();
        for &b in &lits[start..start + n] {
            let (code, len) = codes[b as usize];
            w.write(code as u32, len);
        }
        *lane = w.finish();
        start += n;
    }
    lanes
}

/// Per-lane decode state for the interleaved hot loop.
#[derive(Clone, Copy, Default)]
struct LaneState {
    pos: usize,
    acc: u64,
    nbits: u32,
    written: usize,
}

impl LaneState {
    /// Branchless whole-byte refill: after this, 56-63 bits are live.
    /// Caller guarantees `pos + 8 <= data.len()`.
    #[inline(always)]
    fn refill(&mut self, data: &[u8]) {
        let w = u64::from_le_bytes(data[self.pos..self.pos + 8].try_into().unwrap());
        self.acc |= w << self.nbits;
        self.pos += ((63 - self.nbits) >> 3) as usize;
        self.nbits |= 56;
    }

    #[inline(always)]
    fn decode_one(&mut self, table: &[u16; 1 << MAX_CODE_LEN]) -> u8 {
        // Fixed-size table + masked index → no bounds check in the hot loop.
        let e = table[(self.acc & ((1 << MAX_CODE_LEN) - 1)) as usize];
        let len = (e & 0xF) as u32;
        self.acc >>= len;
        self.nbits -= len;
        (e >> 4) as u8
    }
}

/// Decode NUM_LANES Huffman lanes into one literal buffer.
///
/// The hot loop refills every lane (branchless 8-byte loads, gated by one
/// predictable per-round check against the real slice lengths) then decodes
/// 5 symbols per lane per round — NUM_LANES independent dependency chains
/// for the OoO core to overlap (the huff0 trick; 8 lanes measured best on
/// the M5, +7% over 4). The tail finishes each lane with a fully clamped
/// byte-wise refill, erroring (never panicking, never over-reading) on
/// corrupt streams.
///
/// A huff0-style dual-symbol ("X2") table was tried here and measured a dead
/// end on this core: with 8 lanes the loop is execution-throughput-bound,
/// not chain-latency-bound, so even 81% pair coverage (dickens) gained only
/// ~1.5% while low-coverage data (x-ray, 27-34%) paid -8.5% for the wider
/// entries. Same physics as the FSE "4-way interleave buys only 1.15x"
/// finding. See clean-slate-codec.md §7.
fn decode_lanes(
    table: &[u16; 1 << MAX_CODE_LEN],
    lanes: [&[u8]; NUM_LANES],
    out: &mut [u8],
) -> PzResult<()> {
    let lit_total = out.len();
    let lens = lane_lengths(lit_total);

    // Split the output into NUM_LANES disjoint regions.
    let mut regions: [&mut [u8]; NUM_LANES] = Default::default();
    let mut rest: &mut [u8] = out;
    for (region, &n) in regions.iter_mut().zip(lens.iter()) {
        let (head, tail) = std::mem::take(&mut rest).split_at_mut(n);
        *region = head;
        rest = tail;
    }

    let mut st = [LaneState::default(); NUM_LANES];

    // Hot rounds: 5 symbols per lane per refill (5 × 11 = 55 ≤ 56 live bits).
    // For valid streams the gate only fires within the last 8 bytes of a
    // lane; the remainder falls through to the tail loop.
    let min_len = lens.iter().copied().min().unwrap_or(0);
    let rounds = min_len / 5;
    // Raw write cursors so the NUM_LANES*5 stores per round carry no bounds
    // checks. SAFETY: each lane writes exactly 5 symbols per round, so
    // written stays < rounds * 5 ≤ min_len ≤ regions[lane].len(); the
    // pointers are only used inside the rounds loop, before `regions` is
    // touched again.
    let mut ptrs = [std::ptr::null_mut::<u8>(); NUM_LANES];
    for (p, region) in ptrs.iter_mut().zip(regions.iter_mut()) {
        *p = region.as_mut_ptr();
    }
    'rounds: for _ in 0..rounds {
        for lane in 0..NUM_LANES {
            if st[lane].pos + 8 > lanes[lane].len() {
                break 'rounds;
            }
        }
        // Const trip counts: the compiler fully unrolls the lane loops,
        // keeping NUM_LANES independent chains live for the OoO core.
        for lane in 0..NUM_LANES {
            st[lane].refill(lanes[lane]);
        }
        for _ in 0..5 {
            unsafe {
                for lane in 0..NUM_LANES {
                    *ptrs[lane].add(st[lane].written) = st[lane].decode_one(table);
                    st[lane].written += 1;
                }
            }
        }
    }

    // Tails: fully clamped byte-wise refill.
    for lane in 0..NUM_LANES {
        let region = &mut regions[lane];
        let s = &mut st[lane];
        let data = lanes[lane];
        while s.written < region.len() {
            while s.nbits < MAX_CODE_LEN && s.pos < data.len() {
                s.acc |= (data[s.pos] as u64) << s.nbits;
                s.pos += 1;
                s.nbits += 8;
            }
            let e = table[(s.acc & ((1 << MAX_CODE_LEN) - 1)) as usize];
            let len = (e & 0xF) as u32;
            if len > s.nbits {
                return Err(PzError::InvalidInput);
            }
            s.acc >>= len;
            s.nbits -= len;
            region[s.written] = (e >> 4) as u8;
            s.written += 1;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Sequences
// ---------------------------------------------------------------------------

struct Seq {
    lit_run: u32,
    match_len: u32,
    offset: u32,
}

/// Convert an LzToken stream into sequences + the literal buffer.
/// Trailing literals (after the last match) stay implicit: the decoder
/// derives them as `lit_total - Σ lit_run`.
fn build_sequences(tokens: &[LzToken]) -> (Vec<Seq>, Vec<u8>) {
    let mut seqs = Vec::new();
    let mut lits = Vec::new();
    let mut run: u32 = 0;
    for t in tokens {
        match *t {
            LzToken::Literal(b) => {
                lits.push(b);
                run += 1;
            }
            LzToken::Match { offset, length } => {
                seqs.push(Seq {
                    lit_run: run,
                    match_len: length,
                    offset,
                });
                run = 0;
            }
        }
    }
    (seqs, lits)
}

// ---------------------------------------------------------------------------
// Wire helpers
// ---------------------------------------------------------------------------

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn take<'a>(data: &mut &'a [u8], n: usize) -> PzResult<&'a [u8]> {
    if data.len() < n {
        return Err(PzError::InvalidInput);
    }
    let (head, rest) = data.split_at(n);
    *data = rest;
    Ok(head)
}

fn take_u32(data: &mut &[u8]) -> PzResult<u32> {
    let b = take(data, 4)?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Serialize 256 4-bit code lengths into 128 bytes.
fn pack_lengths(lengths: &[u8; 256]) -> [u8; 128] {
    let mut out = [0u8; 128];
    for i in 0..128 {
        out[i] = (lengths[2 * i] & 0xF) | (lengths[2 * i + 1] << 4);
    }
    out
}

fn unpack_lengths(packed: &[u8]) -> [u8; 256] {
    let mut out = [0u8; 256];
    for i in 0..128 {
        out[2 * i] = packed[i] & 0xF;
        out[2 * i + 1] = packed[i] >> 4;
    }
    out
}

// ---------------------------------------------------------------------------
// Sequence-code streams (small-alphabet Huffman lanes, fused decode)
// ---------------------------------------------------------------------------

/// Encode one sequence-code stream (values ≤ MAX_SEQ_CODE).
///
/// Wire: `[mode]` then either `[value]` (constant stream — common for offset
/// codes when everything is rep0) or `[16B nibble lengths for syms 0..32]
/// [lane_len: u32][lane bytes]`.
fn encode_code_stream(codes: &[u8], out: &mut Vec<u8>) {
    debug_assert!(!codes.is_empty());
    debug_assert!(codes.iter().all(|&c| c <= MAX_SEQ_CODE));
    let mut counts = [0u32; 256];
    for &c in codes {
        counts[c as usize] += 1;
    }
    if counts.iter().filter(|&&c| c > 0).count() == 1 {
        out.push(CODES_CONST);
        out.push(codes[0]);
        return;
    }
    let lengths = huffman_lengths(&counts);
    let codebook = canonical_codes(&lengths).expect("own lengths are valid");
    out.push(CODES_HUFF);
    for i in 0..16 {
        out.push((lengths[2 * i] & 0xF) | (lengths[2 * i + 1] << 4));
    }
    let mut w = BitWriter::new();
    for &c in codes {
        let (code, len) = codebook[c as usize];
        w.write(code as u32, len);
    }
    let lane = w.finish();
    put_u32(out, lane.len() as u32);
    out.extend_from_slice(&lane);
}

/// Decoder side of one sequence-code stream: either a constant or an
/// independent Huffman bit lane decoded one symbol per sequence inside the
/// splice loop (three of these run as independent chains — the same ILP
/// trick as the literal lanes, fused with the splice).
struct CodeLane<'a> {
    constant: Option<u8>,
    table: Option<Box<[u16; 1 << MAX_CODE_LEN]>>,
    data: &'a [u8],
    st: LaneState,
}

impl<'a> CodeLane<'a> {
    fn parse(p: &mut &'a [u8]) -> PzResult<Self> {
        let mode = take(p, 1)?[0];
        match mode {
            CODES_CONST => {
                let v = take(p, 1)?[0];
                if v > MAX_SEQ_CODE {
                    return Err(PzError::InvalidInput);
                }
                Ok(CodeLane {
                    constant: Some(v),
                    table: None,
                    data: &[],
                    st: LaneState::default(),
                })
            }
            CODES_HUFF => {
                let packed = take(p, 16)?;
                let mut lengths = [0u8; 256];
                for i in 0..16 {
                    lengths[2 * i] = packed[i] & 0xF;
                    lengths[2 * i + 1] = packed[i] >> 4;
                }
                // Symbols ≥ 32 have zero length by construction, so every
                // decoded symbol is ≤ MAX_SEQ_CODE once the table validates.
                let table = build_decode_table(&lengths)?;
                let lane_len = take_u32(p)? as usize;
                let data = take(p, lane_len)?;
                Ok(CodeLane {
                    constant: None,
                    table: Some(table),
                    data,
                    st: LaneState::default(),
                })
            }
            _ => Err(PzError::InvalidInput),
        }
    }

    /// Decode the next code. Clamped refill; errors on bit exhaustion.
    #[inline(always)]
    fn next(&mut self) -> PzResult<u8> {
        if let Some(v) = self.constant {
            return Ok(v);
        }
        let table = self.table.as_deref().expect("huff mode has table");
        let s = &mut self.st;
        if s.nbits < MAX_CODE_LEN {
            if s.pos + 8 <= self.data.len() {
                s.refill(self.data);
            } else {
                while s.nbits <= 56 && s.pos < self.data.len() {
                    s.acc |= (self.data[s.pos] as u64) << s.nbits;
                    s.pos += 1;
                    s.nbits += 8;
                }
            }
        }
        let e = table[(s.acc & ((1 << MAX_CODE_LEN) - 1)) as usize];
        let len = (e & 0xF) as u32;
        if len > s.nbits {
            return Err(PzError::InvalidInput);
        }
        s.acc >>= len;
        s.nbits -= len;
        Ok((e >> 4) as u8)
    }
}

// ---------------------------------------------------------------------------
// Public block codec API
// ---------------------------------------------------------------------------

/// Encode one block with the pz2 sequence format using the default parse.
///
/// Uses the shipped LzSeq lazy + repeat-aware parse (identical match
/// decisions to `Lzf`), then re-encodes the tokens in the decode-first wire
/// layout. The result decodes with [`decode`] given the original length.
pub fn encode(input: &[u8]) -> PzResult<Vec<u8>> {
    encode_with_config(input, &SeqConfig::default())
}

/// [`encode`] with an explicit parse config (window size, greedy/lazy,
/// max match length). The wire format does not depend on the config — any
/// pz2 stream decodes with [`decode`] regardless of parse settings.
pub fn encode_with_config(input: &[u8], config: &SeqConfig) -> PzResult<Vec<u8>> {
    let tokens = lzseq::tokenize_with_config(input, config)?;
    let (seqs, lits) = build_sequences(&tokens);

    let mut out = Vec::with_capacity(input.len() / 2 + 64);
    put_u32(&mut out, seqs.len() as u32);
    put_u32(&mut out, lits.len() as u32);

    // --- Literal section ---
    let mut wrote_huff = false;
    let mut counts = [0u32; 256];
    for &b in &lits {
        counts[b as usize] += 1;
    }
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    if distinct >= 2 {
        let lengths = huffman_lengths(&counts);
        let codes = canonical_codes(&lengths)?;
        let lanes = encode_lanes(&lits, &codes);
        let huff_size: usize = 1 + 128 + 4 * NUM_LANES + lanes.iter().map(Vec::len).sum::<usize>();
        if huff_size < 1 + lits.len() {
            out.push(LIT_HUFF);
            out.extend_from_slice(&pack_lengths(&lengths));
            for lane in &lanes {
                put_u32(&mut out, lane.len() as u32);
            }
            for lane in &lanes {
                out.extend_from_slice(lane);
            }
            wrote_huff = true;
        }
    }
    if !wrote_huff {
        out.push(LIT_RAW);
        out.extend_from_slice(&lits);
    }

    // --- Sequence section ---
    if !seqs.is_empty() {
        let n = seqs.len();
        let mut ll_codes = Vec::with_capacity(n);
        let mut of_codes = Vec::with_capacity(n);
        let mut ml_codes = Vec::with_capacity(n);
        let mut extras = BitWriter::new();
        let mut reps = RepeatOffsets::new();
        for s in &seqs {
            let (c, eb, ev) = vcode(s.lit_run);
            ll_codes.push(c);
            extras.write(ev, eb);
            let (c, eb, ev) = reps.encode_offset(s.offset);
            of_codes.push(c);
            extras.write(ev, eb);
            debug_assert!(s.match_len >= MIN_MATCH);
            let (c, eb, ev) = vcode(s.match_len - MIN_MATCH);
            ml_codes.push(c);
            extras.write(ev, eb);
        }
        for stream in [&ll_codes, &of_codes, &ml_codes] {
            encode_code_stream(stream, &mut out);
        }
        let extra_bytes = extras.finish();
        put_u32(&mut out, extra_bytes.len() as u32);
        out.extend_from_slice(&extra_bytes);
    }

    Ok(out)
}

/// Decode one pz2 block. `orig_len` comes from the container block table.
pub fn decode(data: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let mut p = data;
    let seq_count = take_u32(&mut p)? as usize;
    let lit_total = take_u32(&mut p)? as usize;
    if lit_total > orig_len {
        return Err(PzError::InvalidInput);
    }

    // --- Literal section ---
    // The literal buffer carries 16 initialized slack bytes so the splice
    // loop's 16-byte wild copies can read past a run's end without touching
    // uninitialized or unowned memory.
    let mut lits = vec![0u8; lit_total + WILD];
    let mode = take(&mut p, 1)?[0];
    match mode {
        LIT_RAW => lits[..lit_total].copy_from_slice(take(&mut p, lit_total)?),
        LIT_HUFF => {
            let packed = take(&mut p, 128)?;
            let lengths = unpack_lengths(packed);
            let table = build_decode_table(&lengths)?;
            let mut lane_lens = [0usize; NUM_LANES];
            for l in lane_lens.iter_mut() {
                *l = take_u32(&mut p)? as usize;
            }
            let mut lanes: [&[u8]; NUM_LANES] = [&[]; NUM_LANES];
            for (lane, &l) in lanes.iter_mut().zip(lane_lens.iter()) {
                *lane = take(&mut p, l)?;
            }
            decode_lanes(&table, lanes, &mut lits[..lit_total])?;
        }
        _ => return Err(PzError::InvalidInput),
    }

    // --- Sequence splice (wildcopy discipline) ---
    // Output is FULLY INITIALIZED (zeroed) with 2*WILD slack; every copy
    // below validates its logical bounds against `orig_len` BEFORE copying,
    // and wild 16-byte chunks may only spill into the initialized slack
    // (bounded by orig_len + WILD - 1 + WILD < out.len()). Garbage written
    // to slack is either overwritten by the next sequence (which starts at
    // the exact logical cursor) or removed by the final truncate.
    let mut out = vec![0u8; orig_len + 2 * WILD];
    let mut out_len = 0usize;
    let mut lit_pos = 0usize;
    if seq_count > 0 {
        let mut ll_lane = CodeLane::parse(&mut p)?;
        let mut of_lane = CodeLane::parse(&mut p)?;
        let mut ml_lane = CodeLane::parse(&mut p)?;
        let extra_len = take_u32(&mut p)? as usize;
        let extra_bytes = take(&mut p, extra_len)?;
        let mut extras = BitReader::new(extra_bytes);
        let mut reps = RepeatOffsets::new();

        let out_ptr = out.as_mut_ptr();
        let lit_ptr = lits.as_ptr();
        for _ in 0..seq_count {
            // Three independent Huffman chains + the extras lane, fused with
            // the splice — per-sequence work is a handful of table loads the
            // OoO core can overlap, instead of three upfront FSE passes.
            let llc = ll_lane.next()?;
            let ofc = of_lane.next()?;
            let mlc = ml_lane.next()?;
            if llc > 31 || mlc > 31 {
                return Err(PzError::InvalidInput);
            }
            // One branchless refill covers the common case (≤56 bits/seq);
            // read() self-refills for the rare longer ones.
            extras.refill();
            let ll = vdecode(llc, extras.read(vbits(llc))?) as usize;
            let of_extra = extras.read(extra_bits_for_offset_code(ofc))?;
            let offset = reps.decode_offset(ofc, of_extra) as usize;
            let ml = (vdecode(mlc, extras.read(vbits(mlc))?) + MIN_MATCH) as usize;

            // Validate EVERYTHING before any raw copy.
            if lit_pos + ll > lit_total
                || out_len + ll + ml > orig_len
                || offset == 0
                || offset > out_len + ll
            {
                return Err(PzError::InvalidInput);
            }

            // Literal run: 16-byte wild chunks. Reads stay inside
            // lits[..lit_total + WILD); writes inside out[..orig_len + WILD).
            unsafe {
                wild_copy(lit_ptr.add(lit_pos), out_ptr.add(out_len), ll);
            }
            lit_pos += ll;
            out_len += ll;

            // Match copy. offset ≤ out_len was validated above.
            unsafe {
                let dst = out_ptr.add(out_len);
                let src = out_ptr.add(out_len - offset);
                if offset >= WILD {
                    // Chunked: each 16-byte chunk is disjoint (dst-src ≥ 16)
                    // and sequenced after the chunks it may re-read.
                    wild_copy(src, dst, ml);
                } else if offset == 1 {
                    std::ptr::write_bytes(dst, *src, ml);
                } else if ml <= 2 * WILD {
                    // Short overlapping match: the byte loop beats memcpy
                    // dispatch overhead at these sizes.
                    for k in 0..ml {
                        *dst.add(k) = *src.add(k);
                    }
                } else {
                    // Long match at small offset (2..WILD): replicate the
                    // period by exponential doubling — O(log ml) bulk copies
                    // instead of an O(ml) byte chain (mirrors the shipped
                    // lzf overlap fix; 8-13x on repetitive input there).
                    //
                    // Invariant: after copying n bytes, dst[..n] extends the
                    // period. Each round copies cnt ≤ n bytes from dst to
                    // dst+n (disjoint), and n is always a multiple of
                    // `offset` when used as a copy distance, preserving
                    // periodicity.
                    std::ptr::copy_nonoverlapping(src, dst, offset);
                    let mut n = offset;
                    while n < ml {
                        let cnt = n.min(ml - n);
                        std::ptr::copy_nonoverlapping(dst, dst.add(n), cnt);
                        n += cnt;
                    }
                }
            }
            out_len += ml;
        }
    }

    // Trailing literals (exact copy, no wild spill needed).
    let trailing = lit_total - lit_pos;
    if out_len + trailing != orig_len {
        return Err(PzError::InvalidInput);
    }
    // SAFETY: disjoint buffers; lit_pos + trailing == lit_total ≤ lits.len()
    // and out_len + trailing == orig_len < out.len().
    unsafe {
        std::ptr::copy_nonoverlapping(
            lits.as_ptr().add(lit_pos),
            out.as_mut_ptr().add(out_len),
            trailing,
        );
    }
    out.truncate(orig_len);
    Ok(out)
}

/// Wild-copy granularity: copies round up to 16-byte chunks.
const WILD: usize = 16;

/// Copy `n` bytes in unconditional 16-byte chunks (over-copies up to 15
/// bytes past `n`).
///
/// # Safety
/// `src..src + n` rounded up to the next 16-byte multiple must be readable
/// and the corresponding `dst` range writable (both within initialized
/// allocations), and either the regions are fully disjoint or
/// `dst - src >= 16` (so each 16-byte chunk is disjoint and chunks only
/// re-read bytes already written by earlier chunks).
#[inline(always)]
unsafe fn wild_copy(src: *const u8, dst: *mut u8, n: usize) {
    let mut i = 0;
    while i < n {
        std::ptr::copy_nonoverlapping(src.add(i), dst.add(i), WILD);
        i += WILD;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_stream(n: usize, mut state: u32) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for b in &mut out {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            *b = (state >> 16) as u8;
        }
        out
    }

    fn round_trip(input: &[u8]) {
        let enc = encode(input).expect("encode");
        let dec = decode(&enc, input.len()).expect("decode");
        assert_eq!(dec, input, "round-trip mismatch (len {})", input.len());
    }

    #[test]
    fn test_bitio_round_trip() {
        let mut w = BitWriter::new();
        let values: Vec<(u32, u8)> = vec![
            (0, 0),
            (1, 1),
            (0b101, 3),
            (0x7FF, 11),
            (0, 7),
            (0xFFFF_FFFF, 32),
            (0x12345, 17),
            (1, 1),
        ];
        for &(v, b) in &values {
            w.write(v, b);
        }
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &(v, b) in &values {
            assert_eq!(r.read(b).unwrap(), v, "value {v} bits {b}");
        }
    }

    #[test]
    fn test_vcode_round_trip() {
        for v in 0..2048u32 {
            let (c, eb, ev) = vcode(v);
            assert_eq!(vbits(c), eb);
            assert_eq!(vdecode(c, ev), v, "v={v}");
        }
        for v in [4096u32, 65535, 65536, 1 << 20, (1 << 21) - 1, u32::MAX / 2] {
            let (c, eb, ev) = vcode(v);
            assert_eq!(vbits(c), eb);
            assert_eq!(vdecode(c, ev), v, "v={v}");
        }
    }

    #[test]
    fn test_huffman_lanes_round_trip() {
        // Mixed distribution including a depth-limit stress: near-Fibonacci
        // counts force deep trees that must clamp to MAX_CODE_LEN.
        let mut counts = [0u32; 256];
        let mut fib = (1u32, 1u32);
        for c in counts.iter_mut().take(32) {
            *c = fib.0;
            fib = (fib.1, fib.0.saturating_add(fib.1));
        }
        let mut lits = Vec::new();
        for (sym, &c) in counts.iter().enumerate() {
            for _ in 0..c.min(2000) {
                lits.push(sym as u8);
            }
        }
        // Deterministic shuffle so lanes see mixed symbols.
        let mut state = 7u32;
        for i in (1..lits.len()).rev() {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (state as usize) % (i + 1);
            lits.swap(i, j);
        }

        let mut real_counts = [0u32; 256];
        for &b in &lits {
            real_counts[b as usize] += 1;
        }
        let lengths = huffman_lengths(&real_counts);
        assert!(lengths.iter().all(|&l| (l as u32) <= MAX_CODE_LEN));
        let codes = canonical_codes(&lengths).unwrap();
        let table = build_decode_table(&lengths).unwrap();
        let lanes = encode_lanes(&lits, &codes);
        let lane_refs: [&[u8]; NUM_LANES] = std::array::from_fn(|i| &lanes[i][..]);
        let mut decoded = vec![0u8; lits.len()];
        decode_lanes(&table, lane_refs, &mut decoded).unwrap();
        assert_eq!(decoded, lits);
    }

    #[test]
    fn test_periodic_overlap_paths() {
        // Periodic inputs force long matches at every small offset, covering
        // the splice loop's three match-copy paths (offset==1 splat,
        // 2..WILD byte loop, ≥WILD wild chunks) through the real decoder.
        for period in 1usize..=24 {
            let pattern: Vec<u8> = (0..period as u8).map(|b| b.wrapping_mul(37)).collect();
            let input: Vec<u8> = pattern
                .iter()
                .copied()
                .cycle()
                .take(5000 + period)
                .collect();
            round_trip(&input);
        }
    }

    #[test]
    fn test_round_trip_suite() {
        round_trip(b"");
        round_trip(b"a");
        round_trip(b"ab");
        round_trip(b"banana banana banana banana banana");
        round_trip(&b"The quick brown fox jumps over the lazy dog. ".repeat(200));
        round_trip(&vec![0xAB; 100_000]); // offset-1 overlap path
        round_trip(&lcg_stream(65536, 42)); // incompressible → raw literals
        round_trip(&lcg_stream(63, 1));
        // Random-walk u16 samples.
        let mut walk = Vec::new();
        let mut v: u16 = 30000;
        let mut state = 9u32;
        for _ in 0..32768 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            v = v.wrapping_add((((state >> 16) % 33) as i32 - 16) as u16);
            walk.extend_from_slice(&v.to_le_bytes());
        }
        round_trip(&walk);
        // Ends exactly on a match.
        let mut m = b"abcdefgh".repeat(50);
        m.truncate(m.len() - 3);
        round_trip(&m);
    }

    #[test]
    fn test_round_trip_fuzz_lite() {
        // Random splice-of-history inputs: exercises literal runs, repeats,
        // long matches and overlaps together.
        let mut state = 0xC0FFEEu32;
        let mut next = |m: u32| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            (state >> 16) % m
        };
        for case in 0..200 {
            let mut input: Vec<u8> = Vec::new();
            let target = 64 + next(8192) as usize;
            while input.len() < target {
                if input.is_empty() || next(2) == 0 {
                    for _ in 0..=next(16) {
                        input.push(next(256) as u8);
                    }
                } else {
                    let off = 1 + next(input.len().min(4000) as u32) as usize;
                    let len = (1 + next(64)) as usize;
                    for _ in 0..len {
                        let b = input[input.len() - off];
                        input.push(b);
                    }
                }
            }
            let enc = encode(&input).expect("encode");
            let dec = decode(&enc, input.len()).expect("decode");
            assert_eq!(dec, input, "fuzz case {case} len {}", input.len());
        }
    }

    #[test]
    fn test_decode_rejects_garbage() {
        // Truncations of a valid block must error, never panic.
        let input = b"The quick brown fox jumps over the lazy dog. ".repeat(100);
        let enc = encode(&input).unwrap();
        for cut in [0, 1, 4, 8, 9, enc.len() / 2, enc.len() - 1] {
            let _ = decode(&enc[..cut], input.len());
        }
        // Bit-flips in headers must error or produce wrong-length output,
        // never panic.
        for i in 0..enc.len().min(64) {
            let mut bad = enc.clone();
            bad[i] ^= 0x55;
            let _ = decode(&bad, input.len());
        }
    }

    #[test]
    fn test_empty_and_tiny() {
        assert_eq!(decode(&encode(b"").unwrap(), 0).unwrap(), b"");
        // Wrong orig_len must error.
        let enc = encode(b"hello world hello world").unwrap();
        assert!(decode(&enc, 5).is_err());
    }
}
