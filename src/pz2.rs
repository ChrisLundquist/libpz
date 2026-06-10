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

/// Compute optimal MAX_CODE_LEN-limited Huffman code lengths for `counts`
/// via package-merge (Larmore-Hirschberg). Replaces the old halve-and-rebuild
/// heuristic, which distorted *all* frequencies whenever the unlimited tree
/// ran deeper than the limit (routine on skewed 1-4 MiB literal histograms).
///
/// The selected items always form a full binary tree, so the lengths are
/// Kraft-exact — required by `canonical_codes`' validator and the hole-free
/// flat decode table.
///
/// Requires ≥ 2 symbols with nonzero count (callers fall back to raw /
/// constant modes). Cost is O(alphabet × MAX_CODE_LEN) on ≤256 symbols —
/// noise next to the parse.
fn huffman_lengths(counts: &[u32; 256]) -> [u8; 256] {
    // (weight, leaf syms with multiplicity). Sorted ascending by weight,
    // ties by symbol for determinism.
    let mut leaves: Vec<(u64, Vec<u8>)> = counts
        .iter()
        .enumerate()
        .filter(|&(_, &c)| c > 0)
        .map(|(sym, &c)| (c as u64, vec![sym as u8]))
        .collect();
    leaves.sort_by_key(|&(w, ref syms)| (w, syms[0]));
    let n = leaves.len();
    assert!(n >= 2, "huffman_lengths requires >= 2 symbols");
    debug_assert!(n <= 1 << MAX_CODE_LEN);

    // I_1 = leaves; I_{k+1} = merge(leaves, package(I_k)). After
    // MAX_CODE_LEN rounds, each leaf's optimal length-limited code length is
    // its multiplicity among the first 2n-2 items of I_L.
    let mut list = leaves.clone();
    for _ in 1..MAX_CODE_LEN {
        let mut packages: Vec<(u64, Vec<u8>)> = Vec::with_capacity(list.len() / 2);
        let mut it = list.into_iter();
        while let (Some(a), Some(b)) = (it.next(), it.next()) {
            let mut syms = a.1;
            syms.extend_from_slice(&b.1);
            packages.push((a.0 + b.0, syms));
        }
        // Merge the (sorted) packages with the (sorted) fresh leaves.
        let mut merged = Vec::with_capacity(leaves.len() + packages.len());
        let (mut li, mut pi) = (0, 0);
        while li < leaves.len() || pi < packages.len() {
            // Leaves win ties: shorter codes for real symbols over packages.
            let take_leaf =
                pi >= packages.len() || (li < leaves.len() && leaves[li].0 <= packages[pi].0);
            if take_leaf {
                merged.push(leaves[li].clone());
                li += 1;
            } else {
                merged.push(std::mem::take(&mut packages[pi]));
                pi += 1;
            }
        }
        list = merged;
    }

    let mut out = [0u8; 256];
    for (_, syms) in list.iter().take(2 * n - 2) {
        for &s in syms {
            out[s as usize] += 1;
        }
    }
    out
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
    encode_with_prefix(input, 0, config)
}

/// Encode `data[prefix_len..]` as one pz2 block whose matches may reach
/// back into the dictionary prefix `data[..prefix_len]` (cross-block dict
/// tier, design doc P2). The prefix itself is not emitted; the stream
/// decodes with [`decode_with_prefix`] given the same prefix bytes.
///
/// With `prefix_len == 0` this is exactly [`encode_with_config`].
pub fn encode_with_prefix(data: &[u8], prefix_len: usize, config: &SeqConfig) -> PzResult<Vec<u8>> {
    assert!(prefix_len <= data.len());
    let input = &data[prefix_len..];
    // The parse runs over prefix + block so the match finder's window spans
    // the dictionary; tokens covering the prefix are then dropped (a match
    // straddling the boundary re-emits its in-block bytes as literals —
    // at most one straddle per block, negligible).
    let tokens = lzseq::tokenize_with_config(data, config)?;
    let (seqs, lits) = if prefix_len == 0 {
        build_sequences(&tokens)
    } else {
        let mut kept: Vec<LzToken> = Vec::new();
        let mut pos = 0usize;
        for t in &tokens {
            let len = match *t {
                LzToken::Literal(_) => 1,
                LzToken::Match { length, .. } => length as usize,
            };
            if pos >= prefix_len {
                kept.push(*t);
            } else if pos + len > prefix_len {
                // Straddling match: re-emit the in-block tail as literals.
                for &b in &data[prefix_len..pos + len] {
                    kept.push(LzToken::Literal(b));
                }
            }
            pos += len;
        }
        build_sequences(&kept)
    };
    encode_sequences(&seqs, &lits, input.len())
}

/// Encode one block from a worker arena (`dict ‖ block`) using frozen
/// dictionary chains built once via [`crate::lz77::FrozenDict::build`] and
/// shared across workers. Unlike [`encode_with_prefix`], the dict is NOT
/// re-parsed — the parse starts at the dict boundary and consults the
/// frozen chains, so per-block encode cost is block-sized. The stream
/// decodes with [`decode_with_prefix`] given the same dict bytes.
pub fn encode_with_frozen_dict(
    arena: &[u8],
    dict: &std::sync::Arc<crate::lz77::FrozenDict>,
    config: &SeqConfig,
) -> PzResult<Vec<u8>> {
    let dict_len = dict.len();
    assert!(dict_len <= arena.len());
    let tokens =
        lzseq::tokenize_with_dict(arena, dict_len, Some(std::sync::Arc::clone(dict)), config)?;
    let (seqs, lits) = build_sequences(&tokens);
    encode_sequences(&seqs, &lits, arena.len() - dict_len)
}

// ---------------------------------------------------------------------------
// Segment codec (Pz2d dict tier, clean-slate-codec.md §11)
// ---------------------------------------------------------------------------

/// Encode one segment as independent-decodable blocks sharing the segment's
/// head as a dictionary (the Pz2d shape). Returns `(orig_len, wire)` per
/// block, in order.
///
/// - Blocks inside the dict region (`seg[..dict_size]`) come from ONE parse
///   of the whole region, split at block boundaries (matches crossing a
///   boundary are split; each half keeps its offset, which stays valid
///   because the second half still references earlier region content).
///   Block `k` of the region decodes with prefix `seg[..k*block_size]`.
/// - Blocks after the dict region parse with the frozen finder
///   ([`encode_with_frozen_dict`]) and decode with prefix
///   `seg[..dict_size]`.
///
/// `config.max_window` should cover `dict_size + block_size` or reach is
/// left on the table. Decode with [`decode_segment`].
pub fn encode_segment(
    seg: &[u8],
    block_size: usize,
    dict_size: usize,
    config: &SeqConfig,
) -> PzResult<Vec<(usize, Vec<u8>)>> {
    assert!(block_size > 0);
    let dict_len = dict_size.min(seg.len());
    let mut out: Vec<(usize, Vec<u8>)> = Vec::new();

    // --- Dict region: one parse, split at block boundaries ---
    if dict_len > 0 {
        let region = &seg[..dict_len];
        let tokens = lzseq::tokenize_with_config(region, config)?;
        let num_blocks = dict_len.div_ceil(block_size);
        let mut per_block: Vec<Vec<LzToken>> = vec![Vec::new(); num_blocks];
        let mut pos = 0usize;
        for t in &tokens {
            match *t {
                LzToken::Literal(b) => {
                    per_block[pos / block_size].push(LzToken::Literal(b));
                    pos += 1;
                }
                LzToken::Match { offset, length } => {
                    let mut start = pos;
                    let mut rem = length as usize;
                    while rem > 0 {
                        let blk_end = ((start / block_size) + 1) * block_size;
                        let take = rem.min(blk_end - start);
                        let blk = &mut per_block[start / block_size];
                        if take >= MIN_MATCH as usize {
                            blk.push(LzToken::Match {
                                offset,
                                length: take as u32,
                            });
                        } else {
                            // A split remnant too short for a match: emit
                            // the region bytes as literals.
                            for &b in &region[start..start + take] {
                                blk.push(LzToken::Literal(b));
                            }
                        }
                        start += take;
                        rem -= take;
                    }
                    pos += length as usize;
                }
            }
        }
        debug_assert_eq!(pos, dict_len);
        for (k, blk_tokens) in per_block.iter().enumerate() {
            let blk_len = block_size.min(dict_len - k * block_size);
            let (seqs, lits) = build_sequences(blk_tokens);
            out.push((blk_len, encode_sequences(&seqs, &lits, blk_len)?));
        }
    }

    // --- Dicted blocks: frozen finder over the full dict region ---
    if seg.len() > dict_len {
        let frozen = std::sync::Arc::new(crate::lz77::FrozenDict::build(
            &seg[..dict_len],
            config.hash_prefix_len,
        ));
        let mut arena = Vec::with_capacity(dict_len + block_size);
        arena.extend_from_slice(&seg[..dict_len]);
        let mut start = dict_len;
        while start < seg.len() {
            let end = (start + block_size).min(seg.len());
            arena.truncate(dict_len);
            arena.extend_from_slice(&seg[start..end]);
            out.push((
                end - start,
                encode_with_frozen_dict(&arena, &frozen, config)?,
            ));
            start = end;
        }
    }

    Ok(out)
}

/// Decode a segment produced by [`encode_segment`]: blocks in order, each
/// `(orig_len, wire)`. The container's parallel path fans the same logic out
/// in waves (dict-region chain, then dicted blocks against the shared dict);
/// this reference implementation is sequential.
pub fn decode_segment(blocks: &[(usize, &[u8])], dict_size: usize) -> PzResult<Vec<u8>> {
    let total: usize = blocks.iter().map(|&(n, _)| n).sum();
    let mut out: Vec<u8> = Vec::with_capacity(total);
    let mut arena: Vec<u8> = Vec::new();
    for &(orig_len, wire) in blocks {
        if out.len() < dict_size {
            // Dict-region chain: each block's prefix is everything decoded
            // so far, so `out` itself is the arena.
            decode_into_arena(&mut out, wire, orig_len)?;
        } else {
            // Dicted block: prefix is exactly the first dict_size bytes.
            // Seed the side arena with them once, then truncate-and-reuse.
            if arena.len() < dict_size {
                arena.extend_from_slice(&out[arena.len()..dict_size]);
            }
            arena.truncate(dict_size);
            decode_into_arena(&mut arena, wire, orig_len)?;
            out.extend_from_slice(&arena[dict_size..]);
        }
    }
    Ok(out)
}

/// Shared wire writer: sequences + literals → the pz2 block format.
fn encode_sequences(seqs: &[Seq], lits: &[u8], block_len: usize) -> PzResult<Vec<u8>> {
    let mut out = Vec::with_capacity(block_len / 2 + 64);
    put_u32(&mut out, seqs.len() as u32);
    put_u32(&mut out, lits.len() as u32);

    // --- Literal section ---
    let mut wrote_huff = false;
    let mut counts = [0u32; 256];
    for &b in lits {
        counts[b as usize] += 1;
    }
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    if distinct >= 2 {
        let lengths = huffman_lengths(&counts);
        let codes = canonical_codes(&lengths)?;
        let lanes = encode_lanes(lits, &codes);
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
        out.extend_from_slice(lits);
    }

    // --- Sequence section ---
    if !seqs.is_empty() {
        let n = seqs.len();
        let mut ll_codes = Vec::with_capacity(n);
        let mut of_codes = Vec::with_capacity(n);
        let mut ml_codes = Vec::with_capacity(n);
        let mut extras = BitWriter::new();
        let mut reps = RepeatOffsets::new();
        for s in seqs {
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
    decode_with_prefix(data, &[], orig_len)
}

/// Decode one pz2 block whose matches may reach into `prefix` (a stream
/// produced by [`encode_with_prefix`] with the same prefix bytes). Returns
/// only the block's `orig_len` bytes.
///
/// Convenience wrapper over [`decode_into_arena`]; callers decoding many
/// blocks against one shared dict should use the arena form directly so the
/// dict bytes are not re-copied (and the output not re-zeroed) per block.
pub fn decode_with_prefix(data: &[u8], prefix: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    // Empty prefix keeps the arena fresh (capacity 0) so decode_into_arena
    // takes its calloc path — pre-zeroed pages, no explicit memset.
    let mut arena = if prefix.is_empty() {
        Vec::new()
    } else {
        let mut a = Vec::with_capacity(prefix.len() + orig_len + 2 * WILD);
        a.extend_from_slice(prefix);
        a
    };
    decode_into_arena(&mut arena, data, orig_len)?;
    if prefix.is_empty() {
        return Ok(arena);
    }
    Ok(arena.split_off(prefix.len()))
}

/// Decode one pz2 block into `arena`, which on entry holds the bytes the
/// block's matches may reach into (its dict/prefix; `arena.len()` is the
/// prefix length). On success the arena holds `prefix ‖ block`
/// (`arena.len()` = prefix len + `orig_len`).
///
/// The prefix region `arena[..pre]` is never written — every store below
/// lands at positions ≥ `pre` — so a caller decoding many blocks against one
/// shared dict can `truncate(dict_len)` between calls and reuse the arena
/// (and its grown capacity), paying one dict copy per worker instead of one
/// per block.
pub fn decode_into_arena(arena: &mut Vec<u8>, data: &[u8], orig_len: usize) -> PzResult<()> {
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

    splice_into_arena(arena, p, seq_count, &lits, lit_total, orig_len)
}

/// Sequence splice shared by [`decode_into_arena`] and the G32 spike
/// decoders: consumes the sequence section (`p` points just past the literal
/// section) and the pre-decoded literal buffer (`lits`, carrying `WILD`
/// slack bytes past `lit_total`), appending the block to `arena` (whose
/// current contents are the dict/prefix the block's matches may reach into).
fn splice_into_arena(
    arena: &mut Vec<u8>,
    mut p: &[u8],
    seq_count: usize,
    lits: &[u8],
    lit_total: usize,
    orig_len: usize,
) -> PzResult<()> {
    debug_assert!(lits.len() >= lit_total + WILD);
    let pre = arena.len();
    let total = pre + orig_len;
    // --- Sequence splice (wildcopy discipline) ---
    // Output is FULLY INITIALIZED with 2*WILD slack: arena[..pre] is the
    // caller's prefix (initialized by definition) and resize() zero-fills
    // [pre, total + 2*WILD). Every copy below validates its logical bounds
    // against `total` BEFORE copying, and wild 16-byte chunks may only
    // spill into the initialized slack (bounded by
    // total + WILD - 1 + WILD < arena.len()). Garbage written to slack is
    // either overwritten by the next sequence (which starts at the exact
    // logical cursor) or removed by the final truncate. The prefix occupies
    // arena[..pre], so match offsets may reach into it while the cursor
    // (out_len) starts at pre — no store ever targets a position < pre.
    if arena.capacity() == 0 {
        // Fresh arena (pre == 0): vec![0; n] gets pre-zeroed pages straight
        // from the allocator, skipping the explicit memset (and the double
        // page touch) that resize() would pay on a large new buffer.
        *arena = vec![0u8; total + 2 * WILD];
    } else {
        arena.resize(total + 2 * WILD, 0);
    }
    let mut out_len = pre;
    let mut lit_pos = 0usize;
    if seq_count > 0 {
        let mut ll_lane = CodeLane::parse(&mut p)?;
        let mut of_lane = CodeLane::parse(&mut p)?;
        let mut ml_lane = CodeLane::parse(&mut p)?;
        let extra_len = take_u32(&mut p)? as usize;
        let extra_bytes = take(&mut p, extra_len)?;
        let mut extras = BitReader::new(extra_bytes);
        let mut reps = RepeatOffsets::new();

        let out_ptr = arena.as_mut_ptr();
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
                || out_len + ll + ml > total
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
    if out_len + trailing != total {
        return Err(PzError::InvalidInput);
    }
    // SAFETY: disjoint buffers; lit_pos + trailing == lit_total ≤ lits.len()
    // and out_len + trailing == total < arena.len().
    unsafe {
        std::ptr::copy_nonoverlapping(
            lits.as_ptr().add(lit_pos),
            arena.as_mut_ptr().add(out_len),
            trailing,
        );
    }
    arena.truncate(total);
    Ok(())
}

/// Vec-returning splice for the G32 spike decoders (`decode_g32`,
/// `decode_g32_simd`): empty-prefix convenience over [`splice_into_arena`].
fn splice(
    p: &[u8],
    seq_count: usize,
    lits: &[u8],
    lit_total: usize,
    prefix: &[u8],
    orig_len: usize,
) -> PzResult<Vec<u8>> {
    let mut arena = if prefix.is_empty() {
        Vec::new()
    } else {
        let mut a = Vec::with_capacity(prefix.len() + orig_len + 2 * WILD);
        a.extend_from_slice(prefix);
        a
    };
    splice_into_arena(&mut arena, p, seq_count, lits, lit_total, orig_len)?;
    if prefix.is_empty() {
        return Ok(arena);
    }
    Ok(arena.split_off(prefix.len()))
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
// G32 spike: GDeflate-style 32-lane literal relayout (stage 1, CPU-only)
// ---------------------------------------------------------------------------
//
// Stage-1 probe for the pz2-G32 candidate (gpu-path-research.md #1): re-lay
// the literal section of a shipped pz2 block as Huffman codes across 32
// sub-streams pinned to SIMD lanes — literal `i` belongs to lane `i % 32`,
// so each decode round produces 32 contiguous output bytes (one simdgroup
// per tile, 32 symbols/round on a GPU). The shared canonical table is reused
// verbatim; only the bitstream framing changes:
//
// - Each lane's bits are packed LSB-first into 32-bit words.
// - The shared word stream interleaves lane words in EXACT decode-read
//   order, determined by simulating the decoder's refill schedule: before
//   each symbol, a lane with < 32 live bits fetches one word. Since a
//   literal symbol consumes ≤ MAX_CODE_LEN = 11 bits, a lane consumes at
//   most 1 word per round — within the GDeflate ≤2-words-per-round budget
//   (the second word is headroom for fusing extra-bits into lanes later).
// - Lane word counts are implicit (the decoder replays the same schedule);
//   fetches past a lane's real data are zero padding words emitted by the
//   encoder, ≤ 2 per lane beyond `ceil(lane_bits/32)`.
//
// Wire (G32 block): same header as a pz2 block; the LIT_HUFF literal section
// becomes `[128B packed lengths][word_bytes: u32][interleaved words]`.
// Sequence section is byte-identical to shipped pz2. NOT a shipping format —
// spike code measuring layout cost. See pz2-g32-stage1-findings.md.

/// Number of literal sub-streams in the G32 layout (Metal simdgroup width).
const G32_LANES: usize = 32;

/// Encode literals into the G32 interleaved word stream (round-robin lane
/// assignment, decode-read word order). Returns the word stream as bytes
/// (little-endian u32 words).
fn encode_lits_g32(lits: &[u8], codes: &[(u16, u8); 256]) -> Vec<u8> {
    // Per-lane word buffers: lane k holds the codes of literals k, k+32, ...
    #[derive(Default, Clone)]
    struct LaneEnc {
        words: Vec<u32>,
        acc: u64,
        nbits: u32,
    }
    let mut enc = vec![LaneEnc::default(); G32_LANES];
    for (i, &b) in lits.iter().enumerate() {
        let lane = &mut enc[i % G32_LANES];
        let (code, len) = codes[b as usize];
        lane.acc |= (code as u64) << lane.nbits;
        lane.nbits += len as u32;
        if lane.nbits >= 32 {
            lane.words.push(lane.acc as u32);
            lane.acc >>= 32;
            lane.nbits -= 32;
        }
    }
    for lane in &mut enc {
        if lane.nbits > 0 {
            lane.words.push(lane.acc as u32);
        }
    }

    // Simulate the decoder's refill schedule to emit words in decode-read
    // order. `live[k]` mirrors the decoder's bit count exactly.
    let total_words: usize = enc.iter().map(|l| l.words.len()).sum();
    let mut out = Vec::with_capacity((total_words + G32_LANES) * 4);
    let mut wpos = [0usize; G32_LANES];
    let mut live = [0u32; G32_LANES];
    for (i, &b) in lits.iter().enumerate() {
        let k = i % G32_LANES;
        if live[k] < 32 {
            let w = enc[k].words.get(wpos[k]).copied().unwrap_or(0);
            wpos[k] += 1;
            live[k] += 32;
            out.extend_from_slice(&w.to_le_bytes());
        }
        live[k] -= codes[b as usize].1 as u32;
    }
    out
}

/// Scalar decoder for the G32 literal layout: 32 lane states, one symbol per
/// lane per round, refill-when-below-32-bits replayed in the encoder's exact
/// word order. Each round writes 32 contiguous output bytes.
fn decode_lits_g32(table: &[u16; 1 << MAX_CODE_LEN], words: &[u8], out: &mut [u8]) -> PzResult<()> {
    let mut acc = [0u64; G32_LANES];
    let mut nbits = [0u32; G32_LANES];
    let mut wpos = 0usize;

    let mut chunks = out.chunks_exact_mut(G32_LANES);
    for round_out in chunks.by_ref() {
        for (k, slot) in round_out.iter_mut().enumerate() {
            if nbits[k] < 32 {
                if wpos + 4 > words.len() {
                    return Err(PzError::InvalidInput);
                }
                let w = u32::from_le_bytes(words[wpos..wpos + 4].try_into().unwrap());
                acc[k] |= (w as u64) << nbits[k];
                nbits[k] += 32;
                wpos += 4;
            }
            // Invariant: nbits[k] >= 32 >= MAX_CODE_LEN here, and the table
            // is validated hole-free (len in 1..=11), so no per-symbol check.
            let e = table[(acc[k] & ((1 << MAX_CODE_LEN) - 1)) as usize];
            let len = (e & 0xF) as u32;
            acc[k] >>= len;
            nbits[k] -= len;
            *slot = (e >> 4) as u8;
        }
    }
    // Tail round (< 32 symbols), same schedule.
    for (k, slot) in chunks.into_remainder().iter_mut().enumerate() {
        if nbits[k] < 32 {
            if wpos + 4 > words.len() {
                return Err(PzError::InvalidInput);
            }
            let w = u32::from_le_bytes(words[wpos..wpos + 4].try_into().unwrap());
            acc[k] |= (w as u64) << nbits[k];
            nbits[k] += 32;
            wpos += 4;
        }
        let e = table[(acc[k] & ((1 << MAX_CODE_LEN) - 1)) as usize];
        let len = (e & 0xF) as u32;
        acc[k] >>= len;
        nbits[k] -= len;
        *slot = (e >> 4) as u8;
    }
    Ok(())
}

/// Tail decode for the G32 layout: the final `< G32_LANES` symbols, same
/// refill schedule as the hot rounds (lane = position within the round).
/// Shared by the scalar, round-based, and NEON decoders.
#[inline]
fn decode_lits_g32_tail(
    table: &[u16; 1 << MAX_CODE_LEN],
    words: &[u8],
    tail: &mut [u8],
    acc: &mut [u64; G32_LANES],
    nbits: &mut [u32; G32_LANES],
    wpos: &mut usize,
) -> PzResult<()> {
    for (k, slot) in tail.iter_mut().enumerate() {
        if nbits[k] < 32 {
            if *wpos + 4 > words.len() {
                return Err(PzError::InvalidInput);
            }
            let w = u32::from_le_bytes(words[*wpos..*wpos + 4].try_into().unwrap());
            acc[k] |= (w as u64) << nbits[k];
            nbits[k] += 32;
            *wpos += 4;
        }
        let e = table[(acc[k] & ((1 << MAX_CODE_LEN) - 1)) as usize];
        let len = (e & 0xF) as u32;
        acc[k] >>= len;
        nbits[k] -= len;
        *slot = (e >> 4) as u8;
    }
    Ok(())
}

/// Round-based portable decoder for the G32 layout (stage-2 CPU baseline,
/// variant "rounds"): instead of a conditional refill branch per symbol, each
/// round (a) builds the 32-bit refill mask, (b) bounds-checks the word stream
/// ONCE for the whole round, (c) refills only the set lanes (sparse loop),
/// then (d) decodes 32 symbols branch-free. Identical schedule and output to
/// [`decode_lits_g32`]; restructured for ILP.
fn decode_lits_g32_rounds(
    table: &[u16; 1 << MAX_CODE_LEN],
    words: &[u8],
    out: &mut [u8],
) -> PzResult<()> {
    let mut acc = [0u64; G32_LANES];
    let mut nbits = [0u32; G32_LANES];
    let mut wpos = 0usize;

    let mut chunks = out.chunks_exact_mut(G32_LANES);
    for round_out in chunks.by_ref() {
        // (a) refill mask — independent compares the compiler vectorizes.
        let mut m: u32 = 0;
        for (k, &nb) in nbits.iter().enumerate() {
            m |= ((nb < 32) as u32) << k;
        }
        // (b) one bounds check per round.
        let need = m.count_ones() as usize;
        if wpos + 4 * need > words.len() {
            return Err(PzError::InvalidInput);
        }
        // (c) sparse refill in lane order (the wire's word order).
        let mut mm = m;
        while mm != 0 {
            let k = mm.trailing_zeros() as usize;
            let w = u32::from_le_bytes(words[wpos..wpos + 4].try_into().unwrap());
            acc[k] |= (w as u64) << nbits[k];
            nbits[k] += 32;
            wpos += 4;
            mm &= mm - 1;
        }
        // (d) 32 independent table lookups; no per-symbol branches. After a
        // refill every lane holds >= 32 live bits and len <= 11, so neither
        // acc nor nbits can underflow even on corrupt input.
        for (k, slot) in round_out.iter_mut().enumerate() {
            let e = table[(acc[k] & ((1 << MAX_CODE_LEN) - 1)) as usize];
            let len = (e & 0xF) as u32;
            acc[k] >>= len;
            nbits[k] -= len;
            *slot = (e >> 4) as u8;
        }
    }
    decode_lits_g32_tail(
        table,
        words,
        chunks.into_remainder(),
        &mut acc,
        &mut nbits,
        &mut wpos,
    )
}

/// NEON decoder for the G32 layout (aarch64, stage-2 CPU baseline, variant
/// "neon"). Per round: the refill mask and the `nbits += 32` / `nbits -= len`
/// updates are vector ops over two u8x16 lane-count registers (nbits <= 63
/// fits u8); the table gather is 32 independent scalar loads (NEON has no
/// gather) fused with the acc shift; symbol extraction (`e >> 4`) narrows
/// four u16x8 entry vectors straight into two 16-byte output stores.
#[cfg(target_arch = "aarch64")]
fn decode_lits_g32_neon(
    table: &[u16; 1 << MAX_CODE_LEN],
    words: &[u8],
    out: &mut [u8],
) -> PzResult<()> {
    use std::arch::aarch64::*;

    let full_rounds = out.len() / G32_LANES;
    let mut acc = [0u64; G32_LANES];
    let mut wpos = 0usize;
    let wlen = words.len();
    let wptr = words.as_ptr();

    // SAFETY: all loads/stores below are within `acc`/`e_arr`/`nbits_arr`
    // stack arrays, `out[..full_rounds * 32]`, or `words` after the explicit
    // per-round bounds check on `wpos + 4 * need`.
    unsafe {
        let mut nb0 = vdupq_n_u8(0); // lanes 0..16 live-bit counts
        let mut nb1 = vdupq_n_u8(0); // lanes 16..32
        let thresh = vdupq_n_u8(32);
        const BITS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let wbits = vld1q_u8(BITS.as_ptr());
        let lenmask = vdupq_n_u8(0xF);
        let mut nbits_arr = [0u8; G32_LANES];
        let mut e_arr = [0u16; G32_LANES];
        let out_ptr = out.as_mut_ptr();

        for r in 0..full_rounds {
            // Refill mask: bit k set iff lane k holds < 32 live bits.
            let lt0 = vcltq_u8(nb0, thresh);
            let lt1 = vcltq_u8(nb1, thresh);
            // Weights are disjoint per 8-lane group, so horizontal ADD == OR.
            let w0 = vandq_u8(lt0, wbits);
            let w1 = vandq_u8(lt1, wbits);
            let m = vaddv_u8(vget_low_u8(w0)) as u32
                | (vaddv_u8(vget_high_u8(w0)) as u32) << 8
                | (vaddv_u8(vget_low_u8(w1)) as u32) << 16
                | (vaddv_u8(vget_high_u8(w1)) as u32) << 24;

            if m != 0 {
                let need = m.count_ones() as usize;
                if wpos + 4 * need > wlen {
                    return Err(PzError::InvalidInput);
                }
                vst1q_u8(nbits_arr.as_mut_ptr(), nb0);
                vst1q_u8(nbits_arr.as_mut_ptr().add(16), nb1);
                let mut mm = m;
                while mm != 0 {
                    let k = mm.trailing_zeros() as usize;
                    let w = (wptr.add(wpos) as *const u32).read_unaligned().to_le();
                    *acc.get_unchecked_mut(k) |= (w as u64) << *nbits_arr.get_unchecked(k);
                    wpos += 4;
                    mm &= mm - 1;
                }
                nb0 = vaddq_u8(nb0, vandq_u8(lt0, thresh));
                nb1 = vaddq_u8(nb1, vandq_u8(lt1, thresh));
            }

            // Gather (scalar — no NEON gather) fused with the acc shift.
            // Post-refill every lane has >= 32 live bits and len <= 11, so
            // no underflow is possible even on corrupt input.
            for k in 0..G32_LANES {
                let a = *acc.get_unchecked(k);
                let e = *table.get_unchecked((a & ((1 << MAX_CODE_LEN) - 1)) as usize);
                *acc.get_unchecked_mut(k) = a >> (e & 0xF);
                *e_arr.get_unchecked_mut(k) = e;
            }

            // Vector epilogue: sym = e >> 4 (shift-right-narrow), len = low
            // nibble of the truncated entry; two 16B output stores.
            let e0 = vld1q_u16(e_arr.as_ptr());
            let e1 = vld1q_u16(e_arr.as_ptr().add(8));
            let e2 = vld1q_u16(e_arr.as_ptr().add(16));
            let e3 = vld1q_u16(e_arr.as_ptr().add(24));
            let sym01 = vcombine_u8(vshrn_n_u16(e0, 4), vshrn_n_u16(e1, 4));
            let sym23 = vcombine_u8(vshrn_n_u16(e2, 4), vshrn_n_u16(e3, 4));
            let len01 = vandq_u8(vcombine_u8(vmovn_u16(e0), vmovn_u16(e1)), lenmask);
            let len23 = vandq_u8(vcombine_u8(vmovn_u16(e2), vmovn_u16(e3)), lenmask);
            nb0 = vsubq_u8(nb0, len01);
            nb1 = vsubq_u8(nb1, len23);
            vst1q_u8(out_ptr.add(r * G32_LANES), sym01);
            vst1q_u8(out_ptr.add(r * G32_LANES + 16), sym23);
        }

        // Tail: hand the vector state back to the shared scalar tail.
        vst1q_u8(nbits_arr.as_mut_ptr(), nb0);
        vst1q_u8(nbits_arr.as_mut_ptr().add(16), nb1);
        let mut nbits = [0u32; G32_LANES];
        for (dst, &src) in nbits.iter_mut().zip(nbits_arr.iter()) {
            *dst = src as u32;
        }
        decode_lits_g32_tail(
            table,
            words,
            &mut out[full_rounds * G32_LANES..],
            &mut acc,
            &mut nbits,
            &mut wpos,
        )
    }
}

/// Best available CPU decoder for the G32 literal layout (NEON on aarch64,
/// round-based portable elsewhere).
fn decode_lits_g32_best(
    table: &[u16; 1 << MAX_CODE_LEN],
    words: &[u8],
    out: &mut [u8],
) -> PzResult<()> {
    #[cfg(target_arch = "aarch64")]
    return decode_lits_g32_neon(table, words, out);
    #[cfg(not(target_arch = "aarch64"))]
    return decode_lits_g32_rounds(table, words, out);
}

// ---------------------------------------------------------------------------
// Spike-only probe hooks (stage 2): literal-phase benchmark entry points and
// raw G32 literal-section access for the Metal probe. Not a shipping API.
// ---------------------------------------------------------------------------

/// Spike-only: decode just the literal section of a SHIPPED pz2 block with
/// the production 8-lane decoder, returning the literal bytes. This is the
/// honest CPU denominator for the G32 literal-phase comparisons.
#[doc(hidden)]
pub fn spike_decode_lits_pz2(block: &[u8]) -> PzResult<Vec<u8>> {
    let mut p = block;
    let _seq_count = take_u32(&mut p)?;
    let lit_total = take_u32(&mut p)? as usize;
    let mut lits = vec![0u8; lit_total];
    let mode = take(&mut p, 1)?[0];
    match mode {
        LIT_RAW => lits.copy_from_slice(take(&mut p, lit_total)?),
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
            decode_lanes(&table, lanes, &mut lits)?;
        }
        _ => return Err(PzError::InvalidInput),
    }
    Ok(lits)
}

/// Spike-only: decode just the literal section of a G32-transcoded block.
/// `variant`: 0 = scalar (stage-1), 1 = round-based portable, 2 = best
/// (NEON on aarch64).
#[doc(hidden)]
pub fn spike_decode_lits_g32(block: &[u8], variant: u8) -> PzResult<Vec<u8>> {
    let mut p = block;
    let _seq_count = take_u32(&mut p)?;
    let lit_total = take_u32(&mut p)? as usize;
    let mut lits = vec![0u8; lit_total];
    let mode = take(&mut p, 1)?[0];
    match mode {
        LIT_RAW => lits.copy_from_slice(take(&mut p, lit_total)?),
        LIT_HUFF => {
            let packed = take(&mut p, 128)?;
            let lengths = unpack_lengths(packed);
            let table = build_decode_table(&lengths)?;
            let word_bytes = take_u32(&mut p)? as usize;
            let words = take(&mut p, word_bytes)?;
            match variant {
                0 => decode_lits_g32(&table, words, &mut lits)?,
                1 => decode_lits_g32_rounds(&table, words, &mut lits)?,
                _ => decode_lits_g32_best(&table, words, &mut lits)?,
            }
        }
        _ => return Err(PzError::InvalidInput),
    }
    Ok(lits)
}

/// Spike-only: full block decode of a G32-transcoded block using the best
/// CPU literal decoder + the shared splice (counterpart of [`decode`]).
#[doc(hidden)]
pub fn decode_g32_simd(data: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let mut p = data;
    let seq_count = take_u32(&mut p)? as usize;
    let lit_total = take_u32(&mut p)? as usize;
    if lit_total > orig_len {
        return Err(PzError::InvalidInput);
    }
    let mut lits = vec![0u8; lit_total + WILD];
    let mode = take(&mut p, 1)?[0];
    match mode {
        LIT_RAW => lits[..lit_total].copy_from_slice(take(&mut p, lit_total)?),
        LIT_HUFF => {
            let packed = take(&mut p, 128)?;
            let lengths = unpack_lengths(packed);
            let table = build_decode_table(&lengths)?;
            let word_bytes = take_u32(&mut p)? as usize;
            let words = take(&mut p, word_bytes)?;
            decode_lits_g32_best(&table, words, &mut lits[..lit_total])?;
        }
        _ => return Err(PzError::InvalidInput),
    }
    splice(p, seq_count, &lits, lit_total, &[], orig_len)
}

/// Spike-only: extract the literal-section raw materials of a SHIPPED pz2
/// block for the Metal probe: `(code_lengths, literal_bytes)`. Returns
/// `None` for `LIT_RAW` blocks (on GPU that phase is a plain memcpy).
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub fn spike_lit_materials(block: &[u8]) -> PzResult<Option<(Box<[u8; 256]>, Vec<u8>)>> {
    let mut p = block;
    let _seq_count = take_u32(&mut p)?;
    let lit_total = take_u32(&mut p)? as usize;
    let mode = take(&mut p, 1)?[0];
    if mode != LIT_HUFF {
        return Ok(None);
    }
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
    let mut lits = vec![0u8; lit_total];
    decode_lanes(&table, lanes, &mut lits)?;
    Ok(Some((Box::new(lengths), lits)))
}

/// Spike-only: encode a literal slice into the G32 interleaved word stream
/// with the given canonical code lengths (the Metal probe uses this to cut
/// blocks into independent tiles).
#[doc(hidden)]
pub fn spike_g32_encode_words(lits: &[u8], lengths: &[u8; 256]) -> PzResult<Vec<u8>> {
    let codes = canonical_codes(lengths)?;
    Ok(encode_lits_g32(lits, &codes))
}

/// Spike-only: build the flat 2048-entry decode table (`sym << 4 | len`)
/// for upload to GPU memory.
#[doc(hidden)]
pub fn spike_g32_decode_table(lengths: &[u8; 256]) -> PzResult<Vec<u16>> {
    Ok(build_decode_table(lengths)?.to_vec())
}

/// Spike-only: CPU reference decode of a G32 word stream (best variant),
/// used by the Metal probe for round-trip verification.
#[doc(hidden)]
pub fn spike_g32_decode_words(
    words: &[u8],
    lengths: &[u8; 256],
    lit_total: usize,
) -> PzResult<Vec<u8>> {
    let table = build_decode_table(lengths)?;
    let mut out = vec![0u8; lit_total];
    decode_lits_g32_best(&table, words, &mut out)?;
    Ok(out)
}

/// Spike-only: like [`spike_g32_decode_words`] but with a pre-built flat
/// decode table (from [`spike_g32_decode_table`]), so per-tile CPU baselines
/// don't pay a table rebuild per call (the GPU kernel doesn't either).
#[doc(hidden)]
pub fn spike_g32_decode_words_with_table(
    words: &[u8],
    table: &[u16],
    out: &mut [u8],
) -> PzResult<()> {
    let table: &[u16; 1 << MAX_CODE_LEN] = table.try_into().map_err(|_| PzError::InvalidInput)?;
    decode_lits_g32_best(table, words, out)
}

/// Spike-only: one parsed sequence-code lane of a pz2 block (see
/// [`spike_seq_section`]).
#[doc(hidden)]
pub enum SpikeSeqLane {
    /// CODES_CONST: every sequence uses this code.
    Const(u8),
    /// CODES_HUFF: flat 2048-entry decode table + the lane's bitstream.
    Huff { table: Vec<u16>, bits: Vec<u8> },
}

/// Spike-only: the fully parsed sequence section of a pz2 block, in GPU
/// upload form (flat tables, raw bitstreams). Lanes are [ll, of, ml].
#[doc(hidden)]
pub struct SpikeSeqSection {
    pub seq_count: u32,
    pub lit_count: u32,
    pub lanes: [SpikeSeqLane; 3],
    pub extras: Vec<u8>,
}

/// Spike-only: parse a SHIPPED pz2 block's sequence section into GPU upload
/// form for the stage-3 Metal splice probe. The sequence wire is identical
/// between pz2 and G32 blocks.
#[doc(hidden)]
pub fn spike_seq_section(block: &[u8]) -> PzResult<SpikeSeqSection> {
    let mut p = block;
    let seq_count = take_u32(&mut p)?;
    let lit_total = take_u32(&mut p)? as usize;
    // Skip the literal section.
    let mode = take(&mut p, 1)?[0];
    match mode {
        LIT_RAW => {
            take(&mut p, lit_total)?;
        }
        LIT_HUFF => {
            take(&mut p, 128)?;
            let mut lane_lens = [0usize; NUM_LANES];
            for l in lane_lens.iter_mut() {
                *l = take_u32(&mut p)? as usize;
            }
            for &l in lane_lens.iter() {
                take(&mut p, l)?;
            }
        }
        _ => return Err(PzError::InvalidInput),
    }

    let mut parse_lane = |p: &mut &[u8]| -> PzResult<SpikeSeqLane> {
        let mode = take(p, 1)?[0];
        match mode {
            CODES_CONST => Ok(SpikeSeqLane::Const(take(p, 1)?[0])),
            CODES_HUFF => {
                let packed = take(p, 16)?;
                let mut lengths = [0u8; 256];
                for i in 0..16 {
                    lengths[2 * i] = packed[i] & 0xF;
                    lengths[2 * i + 1] = packed[i] >> 4;
                }
                let table = build_decode_table(&lengths)?.to_vec();
                let lane_len = take_u32(p)? as usize;
                let bits = take(p, lane_len)?.to_vec();
                Ok(SpikeSeqLane::Huff { table, bits })
            }
            _ => Err(PzError::InvalidInput),
        }
    };

    if seq_count == 0 {
        return Ok(SpikeSeqSection {
            seq_count,
            lit_count: lit_total as u32,
            lanes: [
                SpikeSeqLane::Const(0),
                SpikeSeqLane::Const(0),
                SpikeSeqLane::Const(0),
            ],
            extras: Vec::new(),
        });
    }
    let ll = parse_lane(&mut p)?;
    let of = parse_lane(&mut p)?;
    let ml = parse_lane(&mut p)?;
    let extra_len = take_u32(&mut p)? as usize;
    let extras = take(&mut p, extra_len)?.to_vec();
    Ok(SpikeSeqSection {
        seq_count,
        lit_count: lit_total as u32,
        lanes: [ll, of, ml],
        extras,
    })
}

/// Transcode a shipped pz2 block into the G32 literal layout. The Huffman
/// table (packed code lengths) and the entire sequence section are copied
/// verbatim; only the literal bitstream framing changes, so the size delta
/// is purely the cost of the GPU-friendly layout.
pub fn transcode_g32(block: &[u8]) -> PzResult<Vec<u8>> {
    let mut p = block;
    let seq_count = take_u32(&mut p)?;
    let lit_total = take_u32(&mut p)? as usize;
    let mode = take(&mut p, 1)?[0];

    let mut out = Vec::with_capacity(block.len() + 4 * G32_LANES);
    put_u32(&mut out, seq_count);
    put_u32(&mut out, lit_total as u32);
    out.push(mode);
    match mode {
        LIT_RAW => out.extend_from_slice(take(&mut p, lit_total)?),
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
            let mut lits = vec![0u8; lit_total];
            decode_lanes(&table, lanes, &mut lits)?;

            // Re-encode with the SAME canonical table in the 32-lane layout.
            let codes = canonical_codes(&lengths)?;
            let words = encode_lits_g32(&lits, &codes);
            out.extend_from_slice(packed);
            put_u32(&mut out, words.len() as u32);
            out.extend_from_slice(&words);
        }
        _ => return Err(PzError::InvalidInput),
    }
    // Sequence section: byte-identical.
    out.extend_from_slice(p);
    Ok(out)
}

/// Decode a G32-transcoded block (counterpart of [`decode`] for the spike
/// layout). `orig_len` comes from the container block table.
pub fn decode_g32(data: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let mut p = data;
    let seq_count = take_u32(&mut p)? as usize;
    let lit_total = take_u32(&mut p)? as usize;
    if lit_total > orig_len {
        return Err(PzError::InvalidInput);
    }

    let mut lits = vec![0u8; lit_total + WILD];
    let mode = take(&mut p, 1)?[0];
    match mode {
        LIT_RAW => lits[..lit_total].copy_from_slice(take(&mut p, lit_total)?),
        LIT_HUFF => {
            let packed = take(&mut p, 128)?;
            let lengths = unpack_lengths(packed);
            let table = build_decode_table(&lengths)?;
            let word_bytes = take_u32(&mut p)? as usize;
            let words = take(&mut p, word_bytes)?;
            decode_lits_g32(&table, words, &mut lits[..lit_total])?;
        }
        _ => return Err(PzError::InvalidInput),
    }

    splice(p, seq_count, &lits, lit_total, &[], orig_len)
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

    fn round_trip_g32(input: &[u8]) {
        let enc = encode(input).expect("encode");
        let g32 = transcode_g32(&enc).expect("transcode");
        let dec = decode_g32(&g32, input.len()).expect("decode_g32");
        assert_eq!(dec, input, "g32 round-trip mismatch (len {})", input.len());
        // Stage-2 variants must agree with the stage-1 scalar decoder.
        let dec_simd = decode_g32_simd(&g32, input.len()).expect("decode_g32_simd");
        assert_eq!(dec_simd, input, "g32 simd mismatch (len {})", input.len());
        let l0 = spike_decode_lits_g32(&g32, 0).expect("lits scalar");
        let l1 = spike_decode_lits_g32(&g32, 1).expect("lits rounds");
        let l2 = spike_decode_lits_g32(&g32, 2).expect("lits best");
        assert_eq!(l0, l1, "rounds variant lits mismatch");
        assert_eq!(l0, l2, "best variant lits mismatch");
        assert_eq!(
            l0,
            spike_decode_lits_pz2(&enc).expect("lits pz2"),
            "8-lane vs g32 lits mismatch"
        );
    }

    #[test]
    fn test_g32_round_trip_suite() {
        round_trip_g32(b"");
        round_trip_g32(b"a");
        round_trip_g32(b"banana banana banana banana banana");
        round_trip_g32(&b"The quick brown fox jumps over the lazy dog. ".repeat(200));
        round_trip_g32(&vec![0xAB; 100_000]);
        round_trip_g32(&lcg_stream(65536, 42)); // raw-literal mode passthrough
        round_trip_g32(&lcg_stream(63, 1));
        // Literal counts straddling lane-round boundaries (multiples of 32
        // and neighbors) exercise the tail round and padding-word schedule.
        for n in [31usize, 32, 33, 63, 64, 65, 1023, 1024, 1025] {
            let input: Vec<u8> = (0..n).map(|i| (i % 7) as u8 + b'a').collect();
            round_trip_g32(&input);
        }
        for period in 1usize..=24 {
            let pattern: Vec<u8> = (0..period as u8).map(|b| b.wrapping_mul(37)).collect();
            let input: Vec<u8> = pattern
                .iter()
                .copied()
                .cycle()
                .take(5000 + period)
                .collect();
            round_trip_g32(&input);
        }
    }

    #[test]
    fn test_g32_fuzz_lite() {
        // Same splice-of-history generator as test_round_trip_fuzz_lite,
        // routed through transcode + g32 decode.
        let mut state = 0xBEEFu32;
        let mut next = |m: u32| {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            (state >> 16) % m
        };
        for case in 0..100 {
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
            let g32 = transcode_g32(&enc).expect("transcode");
            let dec = decode_g32(&g32, input.len()).expect("decode_g32");
            assert_eq!(dec, input, "g32 fuzz case {case} len {}", input.len());
        }
    }

    #[test]
    fn test_g32_decode_rejects_garbage() {
        let input = b"The quick brown fox jumps over the lazy dog. ".repeat(100);
        let g32 = transcode_g32(&encode(&input).unwrap()).unwrap();
        for cut in [0, 1, 4, 8, 9, g32.len() / 2, g32.len() - 1] {
            let _ = decode_g32(&g32[..cut], input.len());
            let _ = decode_g32_simd(&g32[..cut], input.len());
        }
        for i in 0..g32.len().min(64) {
            let mut bad = g32.clone();
            bad[i] ^= 0x55;
            let _ = decode_g32(&bad, input.len());
            let _ = decode_g32_simd(&bad, input.len());
        }
    }

    #[test]
    fn test_prefix_round_trip() {
        // Block content repeats the prefix, so matches must reach into it.
        let prefix = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let block = b"the quick brown fox jumps over the lazy dog! ".repeat(80);
        let mut data = prefix.clone();
        data.extend_from_slice(&block);

        let config = SeqConfig::default();
        let enc = encode_with_prefix(&data, prefix.len(), &config).unwrap();
        let dec = decode_with_prefix(&enc, &prefix, block.len()).unwrap();
        assert_eq!(dec, block);

        // The dict reach must shrink the stream vs encoding the block cold.
        let cold = encode_with_config(&block, &config).unwrap();
        assert!(
            enc.len() < cold.len(),
            "prefix encode ({}) not smaller than cold ({})",
            enc.len(),
            cold.len()
        );

        // Decoding with the wrong prefix length must error or mismatch,
        // never panic.
        let _ = decode_with_prefix(&enc, &prefix[..prefix.len() / 2], block.len());

        // Empty prefix delegates to the plain path.
        let enc0 = encode_with_prefix(&block, 0, &config).unwrap();
        assert_eq!(decode_with_prefix(&enc0, &[], block.len()).unwrap(), block);
    }

    #[test]
    fn test_arena_reuse_round_trip() {
        // Two different blocks decoded against one shared dict through a
        // single truncate-and-reuse arena must match the wrapper path, and
        // the dict region must come through every decode untouched.
        let dict = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let block_a = b"the quick brown fox jumps over the lazy dog! ".repeat(80);
        let block_b = lcg_stream(4000, 7);

        let config = SeqConfig::default();
        let enc = |block: &[u8]| {
            let mut data = dict.clone();
            data.extend_from_slice(block);
            encode_with_prefix(&data, dict.len(), &config).unwrap()
        };
        let enc_a = enc(&block_a);
        let enc_b = enc(&block_b);

        let mut arena = dict.clone();
        for (enc, block) in [(&enc_a, &block_a), (&enc_b, &block_b.clone())] {
            arena.truncate(dict.len());
            decode_into_arena(&mut arena, enc, block.len()).unwrap();
            assert_eq!(&arena[..dict.len()], &dict[..], "dict region modified");
            assert_eq!(&arena[dict.len()..], &block[..]);
            assert_eq!(
                decode_with_prefix(enc, &dict, block.len()).unwrap(),
                block[..]
            );
        }

        // Empty-prefix arena decode matches plain decode.
        let enc0 = encode(&block_b).unwrap();
        let mut arena0 = Vec::new();
        decode_into_arena(&mut arena0, &enc0, block_b.len()).unwrap();
        assert_eq!(arena0, block_b);
    }

    #[test]
    fn test_frozen_dict_round_trip() {
        use crate::lz77::FrozenDict;
        use std::sync::Arc;

        let dict = b"the quick brown fox jumps over the lazy dog. ".repeat(64);
        let block = b"the quick brown fox jumps over the lazy dog! ".repeat(80);
        let mut arena = dict.clone();
        arena.extend_from_slice(&block);

        let config = SeqConfig::default();
        let frozen = Arc::new(FrozenDict::build(&dict, config.hash_prefix_len));
        let enc = encode_with_frozen_dict(&arena, &frozen, &config).unwrap();

        // Wire-compatible with the prefix decoder.
        let dec = decode_with_prefix(&enc, &dict, block.len()).unwrap();
        assert_eq!(dec, block);

        // The frozen chains must find the dict matches: at least as small
        // as the cold encode, in the same family as the re-parse spike.
        let cold = encode_with_config(&block, &config).unwrap();
        assert!(
            enc.len() < cold.len(),
            "frozen-dict encode ({}) not smaller than cold ({})",
            enc.len(),
            cold.len()
        );

        // Empty dict behaves like a plain encode.
        let empty = Arc::new(FrozenDict::build(&[], config.hash_prefix_len));
        let enc0 = encode_with_frozen_dict(&block, &empty, &config).unwrap();
        assert_eq!(decode(&enc0, block.len()).unwrap(), block);
    }

    #[test]
    fn test_segment_round_trip() {
        // Periodic-ish text forces matches that cross every block boundary
        // (exercising the match-splitting path) and reach into the dict.
        let mut seg = Vec::new();
        let mut state = 11u32;
        while seg.len() < 400_000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            seg.extend_from_slice(b"the quick brown fox jumps over the lazy dog. ");
            seg.push((state >> 16) as u8);
        }

        let config = SeqConfig::default();
        for (block, dict) in [
            (64 * 1024, 128 * 1024),
            (64 * 1024, 0),
            (100_000, 250_000),  // unaligned boundaries
            (1 << 20, 16 << 20), // dict larger than segment
            (37, 119),           // degenerate tiny blocks
        ] {
            let blocks = encode_segment(&seg, block, dict, &config).unwrap();
            let total: usize = blocks.iter().map(|(n, _)| n).sum();
            assert_eq!(total, seg.len());
            let refs: Vec<(usize, &[u8])> =
                blocks.iter().map(|(n, w)| (*n, w.as_slice())).collect();
            let dec = decode_segment(&refs, dict).unwrap();
            assert_eq!(dec, seg, "segment round-trip block={block} dict={dict}");
        }

        // Dict reach must beat independent cold blocks when the redundancy
        // lives BEYOND block reach: a 100 KB pseudorandom unit tiled 4x is
        // incompressible per 64 KB block but trivial against the dict.
        let unit = lcg_stream(100_000, 77);
        let mut tiled = Vec::new();
        for _ in 0..4 {
            tiled.extend_from_slice(&unit);
        }
        let cold: usize = tiled
            .chunks(64 * 1024)
            .map(|b| encode_with_config(b, &config).unwrap().len())
            .sum();
        let dicted: usize = encode_segment(&tiled, 64 * 1024, 128 * 1024, &config)
            .unwrap()
            .iter()
            .map(|(_, w)| w.len())
            .sum();
        assert!(
            dicted * 2 < cold,
            "segment encode ({dicted}) should be far smaller than cold blocks ({cold})"
        );
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
