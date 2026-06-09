//! BWT + order-1 context-mixing binary range coder (bzip3-style entropy tail).
//!
//! This is an alternative entropy tail for the BWT pipeline that replaces the
//! `MTF -> zrle(RUNA/RUNB) -> FSE` stack with a context-mixing binary range
//! coder operating directly on the raw BWT output bytes.
//!
//! **Why.** After the BWT groups similar contexts together, the byte stream is
//! locally very predictable: the previous byte is a strong predictor of the
//! next. bzip2/pz model this with MTF + RLE + a static-ish entropy coder, which
//! throws away the higher-order structure. A bzip3-style approach instead feeds
//! the BWT bytes to an adaptive order-1 (previous-byte) model coded bit-by-bit
//! through a binary range coder. The adaptivity + context mixing captures more
//! redundancy than MTF+FSE, at the cost of a more latency-bound decoder.
//!
//! **Coder.** A carry-safe binary range coder with a 33-bit `low` accumulator
//! and a 32-bit `range`. Renormalization emits one byte at a time once `range`
//! drops below 2^24. Carries that propagate out of `low` are handled with a
//! cache byte + a run-length of `0xFF` bytes (the classic LZMA-style range
//! encoder carry scheme), so the coder is exact for arbitrarily long streams.
//!
//! **Model.** For each byte we code its 8 bits MSB-first walking a binary tree
//! of 255 internal nodes. At each node we mix two predictors in the logistic
//! domain:
//!   * an order-0 predictor indexed by the tree node, and
//!   * an order-1 predictor indexed by `(prev_byte, tree_node)`.
//! The two stretched probabilities are combined with a 2-input adaptive logistic
//! mixer (one weight pair per `prev_byte` context). The mixed, squashed
//! probability drives the range coder; after coding the bit, both component
//! predictors and the mixer weights are updated toward the observed bit.
//!
//! All predictors are 12-bit probabilities (`p` in `[1, 4095]`, scale 4096).
//! Predictor update uses the standard shift rule
//! (`p += (4096 - p) >> RATE` on a 1-bit, `p -= p >> RATE` on a 0-bit).

use crate::{PzError, PzResult};

// ---------------------------------------------------------------------------
// Logistic-domain helpers (stretch / squash) for the mixer.
// ---------------------------------------------------------------------------

/// Probability scale: predictors live in `1..=PSCALE-1`, representing P(bit=1).
const PSCALE: i32 = 4096;
const PBITS: u32 = 12;

/// Predictor adaptation rate (shift). Larger = slower adaptation.
const RATE: u32 = 5;

/// Stretch/squash tables for the logistic mixer. Built once per coder.
struct LogisticTables {
    stretch: Vec<i32>, // index 0..=4096 -> stretched value (fixed point, domain [-2047,2047])
    squash: Vec<i32>,  // index 0..4096 (stretched+2048) -> probability 1..=4095
}

impl LogisticTables {
    fn new() -> Self {
        // squash(x) = 4096 / (1 + e^{-x/256}); domain x in [-2047, 2047].
        let mut squash = vec![0i32; 4096];
        for (i, slot) in squash.iter_mut().enumerate() {
            let x = (i as i32 - 2048) as f64 / 256.0;
            let v = 4096.0 / (1.0 + (-x).exp());
            let mut iv = v.round() as i32;
            if iv < 1 {
                iv = 1;
            }
            if iv > 4095 {
                iv = 4095;
            }
            *slot = iv;
        }
        // stretch is the inverse of squash: stretch[p] = x s.t. squash(x) ~= p.
        let mut stretch = vec![0i32; (PSCALE + 1) as usize];
        let mut pi = 0usize;
        for x in 0..4096i32 {
            let p = squash[x as usize] as usize;
            while pi <= p {
                stretch[pi] = x - 2048;
                pi += 1;
            }
        }
        while pi <= PSCALE as usize {
            stretch[pi] = 2047;
            pi += 1;
        }
        LogisticTables { stretch, squash }
    }

    #[inline]
    fn stretch(&self, p: i32) -> i32 {
        self.stretch[p.clamp(1, 4095) as usize]
    }

    #[inline]
    fn squash(&self, x: i32) -> i32 {
        let xc = x.clamp(-2047, 2047) + 2048;
        self.squash[xc as usize]
    }
}

// ---------------------------------------------------------------------------
// Carry-safe binary range encoder / decoder (LZMA-style).
// ---------------------------------------------------------------------------

struct RangeEncoder {
    low: u64, // 33+ bits of headroom for carry detection
    range: u32,
    cache: u8,
    cache_size: u64, // number of pending bytes (cache + run of 0xFF)
    out: Vec<u8>,
    started: bool,
}

impl RangeEncoder {
    fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: 0xFFFF_FFFF,
            cache: 0,
            cache_size: 1, // first shift_low emits the (dummy) cache byte
            out: Vec::new(),
            started: false,
        }
    }

    /// Encode one bit with probability `p1` (scaled to PSCALE) that bit == 1.
    #[inline]
    fn encode_bit(&mut self, p1: u32, bit: u32) {
        // Split the range: bound = range * p1 / PSCALE is the size of the "1"
        // sub-interval. 32x12 multiply fits in u64.
        let bound = ((self.range as u64 * p1 as u64) >> PBITS) as u32;
        if bit == 1 {
            self.range = bound;
        } else {
            self.low += bound as u64;
            self.range -= bound;
        }
        while self.range < (1 << 24) {
            self.shift_low();
            self.range <<= 8;
        }
    }

    #[inline]
    fn shift_low(&mut self) {
        let hi = (self.low >> 32) as u32; // carry bit (0 or 1)
        if self.low < 0xFF00_0000u64 || hi == 1 {
            let mut temp = self.cache;
            loop {
                if self.started {
                    self.out.push((temp as u32).wrapping_add(hi) as u8);
                } else {
                    // The very first emitted byte is a dummy (cache=0); skip it.
                    self.started = true;
                }
                temp = 0xFF;
                self.cache_size -= 1;
                if self.cache_size == 0 {
                    break;
                }
            }
            self.cache = ((self.low >> 24) & 0xFF) as u8;
        }
        self.cache_size += 1;
        self.low = (self.low << 8) & 0xFFFF_FFFF;
    }

    fn finish(mut self) -> Vec<u8> {
        for _ in 0..5 {
            self.shift_low();
        }
        self.out
    }
}

struct RangeDecoder<'a> {
    code: u32,
    range: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    fn new(input: &'a [u8]) -> Self {
        let mut d = RangeDecoder {
            code: 0,
            range: 0xFFFF_FFFF,
            input,
            pos: 0,
        };
        // Prime: read 4 code bytes. The encoder suppressed its dummy cache byte,
        // so the stream begins with the real high byte of `low`.
        for _ in 0..4 {
            d.code = (d.code << 8) | d.next_byte() as u32;
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u8 {
        let b = if self.pos < self.input.len() {
            self.input[self.pos]
        } else {
            0
        };
        self.pos += 1;
        b
    }

    #[inline]
    fn decode_bit(&mut self, p1: u32) -> u32 {
        let bound = ((self.range as u64 * p1 as u64) >> PBITS) as u32;
        let bit;
        if self.code < bound {
            self.range = bound;
            bit = 1;
        } else {
            self.code -= bound;
            self.range -= bound;
            bit = 0;
        }
        while self.range < (1 << 24) {
            self.code = (self.code << 8) | self.next_byte() as u32;
            self.range <<= 8;
        }
        bit
    }
}

// ---------------------------------------------------------------------------
// Order-1 context-mixing bit model.
// ---------------------------------------------------------------------------

/// A count-adaptive 12-bit bit predictor.
///
/// We pack the probability (12 bits) and a saturating hit count (4 bits) into a
/// single `u16`: `state = (count << 12) | p`. The update rate is fast while the
/// count is small (so a fresh context locks onto its bias in a few bits) and
/// settles to a slow steady-state rate, à la PAQ/zpaq state maps. This is worth
/// ~1–2% over a fixed shift on BWT data, for free at decode.
#[derive(Clone, Copy)]
struct Counter(u16);

impl Counter {
    #[inline]
    fn new() -> Self {
        Counter((PSCALE / 2) as u16)
    }
    #[inline]
    fn p(self) -> i32 {
        (self.0 & 0x0FFF) as i32
    }
    #[inline]
    fn count(self) -> u16 {
        (self.0 >> 12) & 0xF
    }
    #[inline]
    fn update(&mut self, bit: u32) {
        let cnt = (self.0 >> 12) & 0xF;
        let p = (self.0 & 0x0FFF) as i32;
        // Faster shift early (rate 2 at count 0) easing to RATE at saturation.
        let rate = (RATE as i32 - 3 + cnt as i32).clamp(2, RATE as i32) as u32;
        let np = if bit == 1 {
            p + ((PSCALE - p) >> rate)
        } else {
            p - (p >> rate)
        };
        let np = np.clamp(1, 4095) as u16;
        let ncnt = if cnt < 0xF { cnt + 1 } else { cnt };
        self.0 = (ncnt << 12) | np;
    }
}

const MIX_SHIFT: i32 = 16;
/// Mixer learning rate shift.
const MIX_LR: i32 = 10;
/// Number of model inputs to the logistic mixer (o0, o1, o2).
const NMODELS: usize = 3;
/// Order-2 hashed table size (power of two). 2^22 entries × 2 B = 8 MiB.
const O2_BITS_DEFAULT: u32 = 22;

#[inline]
fn o2_bits() -> u32 {
    // Sweep override for the spike harness; defaults to O2_BITS_DEFAULT.
    match std::env::var("PZ_O2_BITS") {
        Ok(s) => s.parse().unwrap_or(O2_BITS_DEFAULT),
        Err(_) => O2_BITS_DEFAULT,
    }
}

/// Context-mixing model: order-0, order-1 (prev byte), order-2 (hashed prev two
/// bytes), combined by a per-(prev byte) adaptive logistic mixer.
///
/// Memory: o1 `256*256` Counters (128 KiB), o0 `256`, o2 `2^22` (8 MiB), mixer
/// weights `256 * NMODELS` i32. The o2 table dominates the footprint; it is the
/// single biggest ratio lever on BWT output (captures bigram structure the
/// MTF+FSE tail discards).
struct CmModel {
    tables: LogisticTables,
    o0: Vec<Counter>, // [node]; node in 1..=255
    o1: Vec<Counter>, // [prev_byte * 256 + node]
    o2: Vec<Counter>, // [hash(prev2,prev1,node) & o2_mask]
    w: Vec<i32>,      // mixer weights <<16; [prev_byte*NMODELS + which]
    o2_bits: u32,
    o2_mask: usize,
}

/// Per-bit prediction state carried from `predict` to `update` so the decoder
/// reproduces the encoder exactly.
struct PredState {
    s: [i32; NMODELS], // stretched component inputs
    idx: [usize; NMODELS],
    wbase: usize,
    p_mixed: u32,
}

#[inline]
fn o2_index(prev2: u8, prev1: u8, node: usize, bits: u32, mask: usize) -> usize {
    // Cheap multiplicative hash of (prev2, prev1, node) into the o2 table.
    let key = ((prev2 as u32) << 16) ^ ((prev1 as u32) << 8) ^ (node as u32);
    (key.wrapping_mul(2654435761) >> (32 - bits)) as usize & mask
}

impl CmModel {
    fn new() -> Self {
        let bits = o2_bits();
        let size = 1usize << bits;
        CmModel {
            tables: LogisticTables::new(),
            o0: vec![Counter::new(); 256],
            o1: vec![Counter::new(); 256 * 256],
            o2: vec![Counter::new(); size],
            w: vec![1 << (MIX_SHIFT - 2); 256 * NMODELS], // ~0.25 each initially
            o2_bits: bits,
            o2_mask: size - 1,
        }
    }

    /// Predict P(bit=1) at tree `node` given prev-byte contexts.
    #[inline]
    fn predict(&self, prev1: u8, prev2: u8, node: usize) -> (u32, PredState) {
        let i0 = node;
        let i1 = (prev1 as usize) * 256 + node;
        let i2 = o2_index(prev2, prev1, node, self.o2_bits, self.o2_mask);
        let s0 = self.tables.stretch(self.o0[i0].p());
        let s1 = self.tables.stretch(self.o1[i1].p());
        let s2 = self.tables.stretch(self.o2[i2].p());
        let wbase = (prev1 as usize) * NMODELS;
        let dot =
            (self.w[wbase] * s0 + self.w[wbase + 1] * s1 + self.w[wbase + 2] * s2) >> MIX_SHIFT;
        let p = (self.tables.squash(dot) as u32).clamp(1, 4095);
        (
            p,
            PredState {
                s: [s0, s1, s2],
                idx: [i0, i1, i2],
                wbase,
                p_mixed: p,
            },
        )
    }

    /// Update component predictors and mixer weights toward `bit`.
    #[inline]
    fn update(&mut self, st: &PredState, bit: u32) {
        let err = ((bit as i32) << PBITS) - st.p_mixed as i32;
        for k in 0..NMODELS {
            self.w[st.wbase + k] += (err * st.s[k]) >> MIX_LR;
        }
        self.o0[st.idx[0]].update(bit);
        self.o1[st.idx[1]].update(bit);
        self.o2[st.idx[2]].update(bit);
    }
}

// ---------------------------------------------------------------------------
// Public API: encode / decode a raw byte stream (intended: BWT output bytes).
// ---------------------------------------------------------------------------

/// Encode `input` with the order-1 context-mixing binary range coder.
///
/// The output is a self-delimiting blob (no length prefix); the caller must
/// supply the original length to [`decode`]. Empty input yields empty output.
pub fn encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut model = CmModel::new();
    let mut enc = RangeEncoder::new();
    let mut prev1: u8 = 0;
    let mut prev2: u8 = 0;
    for &byte in input {
        let mut node = 1usize; // bit-tree root
        for k in (0..8).rev() {
            let bit = ((byte >> k) & 1) as u32;
            let (p, st) = model.predict(prev1, prev2, node);
            enc.encode_bit(p, bit);
            model.update(&st, bit);
            node = (node << 1) | bit as usize;
        }
        prev2 = prev1;
        prev1 = byte;
    }
    enc.finish()
}

/// Decode a blob produced by [`encode`], recovering `out_len` bytes.
pub fn decode(input: &[u8], out_len: usize) -> PzResult<Vec<u8>> {
    if out_len == 0 {
        return Ok(Vec::new());
    }
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }
    let mut model = CmModel::new();
    let mut dec = RangeDecoder::new(input);
    let mut out = Vec::with_capacity(out_len);
    let mut prev1: u8 = 0;
    let mut prev2: u8 = 0;
    for _ in 0..out_len {
        let mut node = 1usize;
        for _ in 0..8 {
            let (p, st) = model.predict(prev1, prev2, node);
            let bit = dec.decode_bit(p);
            model.update(&st, bit);
            node = (node << 1) | bit as usize;
        }
        let byte = (node & 0xFF) as u8;
        out.push(byte);
        prev2 = prev1;
        prev1 = byte;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// FAST variant: direct order-1 bit-tree, NO logistic mixing.
//
// A pure order-1 binary range coder: one adaptive predictor per (prev_byte,
// tree_node), coded directly (no order-0/order-2 mix, no stretch/squash). This
// is the speed ceiling for a serial per-bit range coder on this CPU — it trades
// the CM ratio gain for the cheapest possible inner loop. Kept as a probe / a
// faster operating point for the spike's decode-throughput question.
// ---------------------------------------------------------------------------

/// 2-input model: order-0 + order-1, adaptive logistic mixer, NO order-2.
///
/// The "middle" operating point. Dropping the 8 MiB order-2 table removes its
/// per-bit cache miss + update while keeping the cold-start robustness that the
/// order-0↔order-1 mix buys (a fresh prev-byte context falls back on order-0
/// instead of a flat 0.5). This recovers most of the FAST variant's speed while
/// keeping most of the full CM's ratio.
struct MidModel {
    tables: LogisticTables,
    o0: Vec<Counter>,
    o1: Vec<Counter>,
    w: Vec<i32>, // [prev_byte*2 + which]
}

struct MidState {
    s0: i32,
    s1: i32,
    i0: usize,
    i1: usize,
    wbase: usize,
    p: u32,
}

impl MidModel {
    fn new() -> Self {
        MidModel {
            tables: LogisticTables::new(),
            o0: vec![Counter::new(); 256],
            o1: vec![Counter::new(); 256 * 256],
            w: vec![1 << (MIX_SHIFT - 1); 256 * 2],
        }
    }
    #[inline]
    fn predict(&self, prev1: u8, node: usize) -> (u32, MidState) {
        let i0 = node;
        let i1 = (prev1 as usize) * 256 + node;
        let s0 = self.tables.stretch(self.o0[i0].p());
        let s1 = self.tables.stretch(self.o1[i1].p());
        let wbase = (prev1 as usize) * 2;
        let dot = (self.w[wbase] * s0 + self.w[wbase + 1] * s1) >> MIX_SHIFT;
        let p = (self.tables.squash(dot) as u32).clamp(1, 4095);
        (
            p,
            MidState {
                s0,
                s1,
                i0,
                i1,
                wbase,
                p,
            },
        )
    }
    #[inline]
    fn update(&mut self, st: &MidState, bit: u32) {
        let err = ((bit as i32) << PBITS) - st.p as i32;
        self.w[st.wbase] += (err * st.s0) >> MIX_LR;
        self.w[st.wbase + 1] += (err * st.s1) >> MIX_LR;
        self.o0[st.i0].update(bit);
        self.o1[st.i1].update(bit);
    }
}

/// Encode with the 2-input mid model (order-0 + order-1, no order-2).
pub fn encode_mid(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut model = MidModel::new();
    let mut enc = RangeEncoder::new();
    let mut prev1: u8 = 0;
    for &byte in input {
        let mut node = 1usize;
        for k in (0..8).rev() {
            let bit = ((byte >> k) & 1) as u32;
            let (p, st) = model.predict(prev1, node);
            enc.encode_bit(p, bit);
            model.update(&st, bit);
            node = (node << 1) | bit as usize;
        }
        prev1 = byte;
    }
    enc.finish()
}

/// Decode a blob produced by [`encode_mid`].
pub fn decode_mid(input: &[u8], out_len: usize) -> PzResult<Vec<u8>> {
    if out_len == 0 {
        return Ok(Vec::new());
    }
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }
    let mut model = MidModel::new();
    let mut dec = RangeDecoder::new(input);
    let mut out = Vec::with_capacity(out_len);
    let mut prev1: u8 = 0;
    for _ in 0..out_len {
        let mut node = 1usize;
        for _ in 0..8 {
            let (p, st) = model.predict(prev1, node);
            let bit = dec.decode_bit(p);
            model.update(&st, bit);
            node = (node << 1) | bit as usize;
        }
        let byte = (node & 0xFF) as u8;
        out.push(byte);
        prev1 = byte;
    }
    Ok(out)
}

/// BLEND model: order-1 ⊕ order-0 blended in the LINEAR probability domain by
/// order-1's saturating hit count — no stretch/squash, no adaptive mixer.
///
/// Rationale: the MID variant showed the logistic mixer (two stretches + squash
/// + dot + two weight updates per bit) is the decode bottleneck, not memory. A
/// confidence-weighted linear blend keeps the cold-start robustness (lean on
/// order-0 while the order-1 context is unseen) at a fraction of the cost — a
/// few adds/mults per bit. Counter packs a 4-bit count we reuse directly as the
/// blend weight.
struct BlendModel {
    o0: Vec<Counter>,
    o1: Vec<Counter>,
}

struct BlendState {
    i0: usize,
    i1: usize,
}

impl BlendModel {
    fn new() -> Self {
        BlendModel {
            o0: vec![Counter::new(); 256],
            o1: vec![Counter::new(); 256 * 256],
        }
    }
    #[inline]
    fn predict(&self, prev1: u8, node: usize) -> (u32, BlendState) {
        let i0 = node;
        let i1 = (prev1 as usize) * 256 + node;
        let c1 = self.o1[i1];
        let p1 = c1.p();
        let p0 = self.o0[i0].p();
        // Blend weight from order-1 confidence: count 0..15 -> weight 1..16/16.
        // Cold order-1 (count 0) leans ~94% on order-0; warm order-1 dominates.
        let w = (c1.count() as i32) + 1; // 1..=16
        let p = ((p1 * w + p0 * (16 - w)) >> 4).clamp(1, 4095) as u32;
        (p, BlendState { i0, i1 })
    }
    #[inline]
    fn update(&mut self, st: &BlendState, bit: u32) {
        self.o0[st.i0].update(bit);
        self.o1[st.i1].update(bit);
    }
}

/// Encode with the linear-blend model (order-1 ⊕ order-0 by confidence).
pub fn encode_blend(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut model = BlendModel::new();
    let mut enc = RangeEncoder::new();
    let mut prev1: u8 = 0;
    for &byte in input {
        let mut node = 1usize;
        for k in (0..8).rev() {
            let bit = ((byte >> k) & 1) as u32;
            let (p, st) = model.predict(prev1, node);
            enc.encode_bit(p, bit);
            model.update(&st, bit);
            node = (node << 1) | bit as usize;
        }
        prev1 = byte;
    }
    enc.finish()
}

/// Decode a blob produced by [`encode_blend`].
pub fn decode_blend(input: &[u8], out_len: usize) -> PzResult<Vec<u8>> {
    if out_len == 0 {
        return Ok(Vec::new());
    }
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }
    let mut model = BlendModel::new();
    let mut dec = RangeDecoder::new(input);
    let mut out = Vec::with_capacity(out_len);
    let mut prev1: u8 = 0;
    for _ in 0..out_len {
        let mut node = 1usize;
        for _ in 0..8 {
            let (p, st) = model.predict(prev1, node);
            let bit = dec.decode_bit(p);
            model.update(&st, bit);
            node = (node << 1) | bit as usize;
        }
        let byte = (node & 0xFF) as u8;
        out.push(byte);
        prev1 = byte;
    }
    Ok(out)
}

/// Encode with the fast direct order-1 model (no mixing).
pub fn encode_fast(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    // one predictor per (prev_byte, node); node in 1..=255.
    let mut t = vec![(PSCALE / 2) as u16; 256 * 256];
    let mut enc = RangeEncoder::new();
    let mut prev: usize = 0;
    for &byte in input {
        let base = prev * 256;
        let mut node = 1usize;
        for k in (0..8).rev() {
            let bit = ((byte >> k) & 1) as u32;
            let idx = base + node;
            let p = t[idx] as u32;
            enc.encode_bit(p, bit);
            // inline fixed-rate update
            if bit == 1 {
                t[idx] += ((PSCALE - t[idx] as i32) >> RATE) as u16;
            } else {
                t[idx] -= (t[idx] >> RATE) as u16;
            }
            node = (node << 1) | bit as usize;
        }
        prev = byte as usize;
    }
    enc.finish()
}

/// Decode a blob produced by [`encode_fast`].
pub fn decode_fast(input: &[u8], out_len: usize) -> PzResult<Vec<u8>> {
    if out_len == 0 {
        return Ok(Vec::new());
    }
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }
    let mut t = vec![(PSCALE / 2) as u16; 256 * 256];
    let mut dec = RangeDecoder::new(input);
    let mut out = Vec::with_capacity(out_len);
    let mut prev: usize = 0;
    for _ in 0..out_len {
        let base = prev * 256;
        let mut node = 1usize;
        for _ in 0..8 {
            let idx = base + node;
            let p = t[idx] as u32;
            let bit = dec.decode_bit(p);
            if bit == 1 {
                t[idx] += ((PSCALE - t[idx] as i32) >> RATE) as u16;
            } else {
                t[idx] -= (t[idx] >> RATE) as u16;
            }
            node = (node << 1) | bit as usize;
        }
        let byte = (node & 0xFF) as u8;
        out.push(byte);
        prev = byte as usize;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(data: &[u8]) {
        let enc = encode(data);
        let dec = decode(&enc, data.len()).expect("decode");
        assert_eq!(dec, data, "round-trip mismatch (len {})", data.len());
        // The fast variant must also round-trip on every case.
        let encf = encode_fast(data);
        let decf = decode_fast(&encf, data.len()).expect("decode_fast");
        assert_eq!(decf, data, "fast round-trip mismatch (len {})", data.len());
        // ...as must the mid (2-input) variant.
        let encm = encode_mid(data);
        let decm = decode_mid(&encm, data.len()).expect("decode_mid");
        assert_eq!(decm, data, "mid round-trip mismatch (len {})", data.len());
        // ...and the linear-blend variant.
        let encb = encode_blend(data);
        let decb = decode_blend(&encb, data.len()).expect("decode_blend");
        assert_eq!(decb, data, "blend round-trip mismatch (len {})", data.len());
    }

    #[test]
    fn empty() {
        assert!(encode(&[]).is_empty());
        assert_eq!(decode(&[], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn single_bytes() {
        for b in 0u16..=255 {
            roundtrip(&[b as u8]);
        }
    }

    #[test]
    fn short_patterns() {
        roundtrip(b"a");
        roundtrip(b"ab");
        roundtrip(b"hello world");
        roundtrip(b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        roundtrip(&[0u8; 1000]);
        roundtrip(&[255u8; 1000]);
    }

    #[test]
    fn adversarial() {
        roundtrip(&vec![0u8; 100_000]); // all-zeros
        roundtrip(&vec![255u8; 100_000]); // all-255
        let periodic: Vec<u8> = (0..100_000).map(|i| (i % 7) as u8).collect();
        roundtrip(&periodic);
        // alternating extremes (carry stress)
        let alt: Vec<u8> = (0..100_000)
            .map(|i| if i % 2 == 0 { 0 } else { 255 })
            .collect();
        roundtrip(&alt);
        let ramp: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
        roundtrip(&ramp);
    }

    #[test]
    fn random_fuzz() {
        let mut state = 0x12345678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..200 {
            let len = (next() % 4000) as usize;
            let data: Vec<u8> = (0..len).map(|_| (next() & 0xFF) as u8).collect();
            let enc = encode(&data);
            let dec = decode(&enc, data.len()).expect("decode");
            assert_eq!(dec, data, "fuzz trial {} len {}", trial, len);
            let encf = encode_fast(&data);
            let decf = decode_fast(&encf, data.len()).expect("decode_fast");
            assert_eq!(decf, data, "fast fuzz trial {} len {}", trial, len);
            let encm = encode_mid(&data);
            let decm = decode_mid(&encm, data.len()).expect("decode_mid");
            assert_eq!(decm, data, "mid fuzz trial {} len {}", trial, len);
            let encb = encode_blend(&data);
            let decb = decode_blend(&encb, data.len()).expect("decode_blend");
            assert_eq!(decb, data, "blend fuzz trial {} len {}", trial, len);
        }
    }

    #[test]
    fn biased_random_fuzz() {
        // Biased toward a few symbols (more BWT-like) to exercise the model's
        // high-probability paths and carry propagation.
        let mut state = 0xCAFEBABEu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..100 {
            let len = (next() % 8000) as usize;
            let data: Vec<u8> = (0..len)
                .map(|_| {
                    if next() % 10 < 8 {
                        0
                    } else {
                        (next() & 0xFF) as u8
                    }
                })
                .collect();
            roundtrip(&data);
            let _ = trial;
        }
    }

    #[test]
    fn bwt_output_fuzz() {
        // The real use case: round-trip the ACTUAL BWT output of random/biased
        // inputs through all four coders. Catches any interaction between the
        // BWT byte distribution and the model/carry logic that synthetic data
        // might miss.
        let mut state = 0x9E3779B9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..60 {
            let len = 1 + (next() % 20000) as usize;
            let mode = next() % 4;
            let data: Vec<u8> = (0..len)
                .map(|_| match mode {
                    0 => (next() & 0xFF) as u8,              // uniform
                    1 => (next() % 4) as u8,                 // low-cardinality
                    2 => b'a' + (next() % 6) as u8,          // text-like
                    _ => (((next() % 13) == 0) as u8) * 255, // sparse 255s
                })
                .collect();
            if let Some(bwt) = crate::bwt::encode(&data) {
                roundtrip(&bwt.data);
            }
        }
    }

    #[test]
    fn large_carry_stress() {
        // A long run of maximally-skewed predictions forces many consecutive
        // 0xFF carry bytes in the range encoder; verify the carry chain is exact
        // at scale across all coders. (~1 MiB.)
        let mut data = vec![0u8; 1 << 20];
        // sprinkle a few non-zero bytes so the model stays near-but-not-at p=max
        for i in (0..data.len()).step_by(4096) {
            data[i] = 0xAA;
        }
        roundtrip(&data);
        // also all-0xFF at scale
        roundtrip(&vec![0xFFu8; 1 << 20]);
    }
}
