//! Numeric decorrelation front-end (the `Num` pipeline).
//!
//! `Num` is a transform + entropy pipeline (like `Bw`/`Bbw`, with **no LZ
//! tokens**) aimed at fixed-width numeric / record data such as the Silesia
//! `x-ray` (16-bit medical image samples) and `sao` (star catalogue records)
//! files, where order-0 FSE on the raw byte stream leaves a lot on the table
//! because the *inter-byte* structure (low/high bytes of a sample, columns of a
//! record) is not modelled.
//!
//! The transform is a byte-plane split (transpose / struct-of-arrays): the input
//! is cut into `S` interleaved planes (plane `k` holds bytes at positions `k,
//! k+S, k+2S, …`), so each plane contains one "column" of the record and is far
//! more compressible on its own. Each plane is then independently passed through
//! a small **gated** transform — `Raw`, `Delta` (byte-wrapping first difference),
//! or `Delta`+`zigzag` — choosing whichever produces the smallest FSE-coded
//! output, and **entropy-coded as its own separate FSE stream**.
//!
//! Three load-bearing facts (each one found the hard way by the validating spike):
//!
//! 1. **Planes must be entropy-coded as *separate* FSE streams.** Concatenating
//!    the transformed planes into one blob before FSE smears the per-column
//!    statistics together and loses several points of ratio (`sao` in particular
//!    regresses badly). Each plane gets its own `fse::encode_best`.
//! 2. **Per-plane transform gating matters**, especially for `sao`: blanket delta
//!    is *worse* than transpose-only there; gating each plane to `min(raw, delta,
//!    delta+zigzag)` is what reaches the good ratio.
//! 3. **The stride is auto-picked.** The encoder tries `S ∈ {2,4,8,16,28,32}`,
//!    measures the total gated compressed size for each, and keeps the smallest
//!    (`x-ray` wins at `S=2`, `sao` at `S=28`).
//!
//! If even the best transform fails to beat storing the bytes verbatim (the data
//! is not numeric / already incompressible), the block falls back to a **STORE**
//! mode so `Num` never expands by more than a one-byte header.
//!
//! The inverse is `O(n)` (a prefix sum per plane plus an interleave), so the
//! pipeline preserves libpz's block-parallel decode property.
//!
//! ## Wire format (one block; the decoder is told `orig_len`)
//!
//! ```text
//! [stride: u8]                         // 0 = STORED raw; else S in {2,4,8,16,28,32}
//! if stride == 0:  [raw bytes]         // orig_len bytes verbatim
//! else, with full = orig_len / stride, rem = orig_len % stride:
//!   for each of `stride` planes:
//!      [transform_tag: u8]             // 0=Raw 1=Delta 2=DeltaZigzag
//!      [plane_comp_len: u32 LE]
//!      [plane_fse_bytes]               // fse::encode_best(transformed plane)
//!   [remainder: rem raw bytes]         // last `rem` bytes of input, verbatim
//! ```
//!
//! The per-plane transform functions (`split_planes`, `join_planes`,
//! `delta_fwd`/`delta_inv`, `zigzag_fwd`/`zigzag_inv`, `apply_fwd`/`apply_inv`)
//! were lifted verbatim from a fuzzed spike (63k random + adversarial cases, all
//! exact) and must not be re-derived.

use crate::fse;
use crate::{PzError, PzResult};

/// Strides the encoder sweeps when auto-picking the byte-plane width.
///
/// `x-ray` (16-bit samples) wins at `S=2`; `sao` (28-byte records) at `S=28`.
/// The set spans common power-of-two widths plus 28 for the `sao` record size.
///
/// Shared with `analysis::stride_decorrelation` so pipeline auto-selection
/// detects exactly the strides this encoder can exploit.
pub(crate) const CANDIDATE_STRIDES: [usize; 6] = [2, 4, 8, 16, 28, 32];

/// Sentinel stride value meaning "block is stored raw" (STORE fallback).
const STORE_STRIDE: u8 = 0;

// ---------------------------------------------------------------------------
// Transform primitives (lifted verbatim from the validated, fuzzed spike).
// ---------------------------------------------------------------------------

/// Split `data` into `s` byte-planes (transpose / SoA). Plane `k` collects bytes
/// at positions `k, k+s, k+2s, …`. The tail (`len % s` bytes) is returned as a
/// separate "remainder" buffer so the transform is exact for any length.
/// Returns `(planes, remainder)`.
fn split_planes(data: &[u8], s: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    let n = data.len();
    let full = n / s; // number of complete records
    let mut planes = vec![Vec::with_capacity(full); s];
    for rec in 0..full {
        let base = rec * s;
        for (k, plane) in planes.iter_mut().enumerate() {
            plane.push(data[base + k]);
        }
    }
    let remainder = data[full * s..].to_vec();
    (planes, remainder)
}

/// Inverse of [`split_planes`]: interleave planes back, append remainder.
fn join_planes(planes: &[Vec<u8>], remainder: &[u8], s: usize) -> Vec<u8> {
    assert_eq!(planes.len(), s);
    let full = if s == 0 { 0 } else { planes[0].len() };
    let mut out = Vec::with_capacity(full * s + remainder.len());
    for rec in 0..full {
        for plane in planes.iter() {
            out.push(plane[rec]);
        }
    }
    out.extend_from_slice(remainder);
    out
}

/// Forward delta (byte-wrapping). First byte unchanged.
fn delta_fwd(p: &[u8]) -> Vec<u8> {
    if p.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u8; p.len()];
    out[0] = p[0];
    for i in 1..p.len() {
        out[i] = p[i].wrapping_sub(p[i - 1]);
    }
    out
}

/// Inverse delta (prefix sum, byte-wrapping).
fn delta_inv(d: &[u8]) -> Vec<u8> {
    if d.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0u8; d.len()];
    out[0] = d[0];
    for i in 1..d.len() {
        out[i] = d[i].wrapping_add(out[i - 1]);
    }
    out
}

/// Zigzag-encode signed delta to unsigned (maps small +/- to small values).
/// Applied to an already-delta'd plane treated as signed `i8`.
fn zigzag_fwd(d: &[u8]) -> Vec<u8> {
    d.iter()
        .map(|&b| {
            let v = b as i8 as i16;
            ((v << 1) ^ (v >> 7)) as u8
        })
        .collect()
}

/// Inverse of [`zigzag_fwd`].
fn zigzag_inv(z: &[u8]) -> Vec<u8> {
    z.iter()
        .map(|&b| {
            let zz = b as u16;
            let v = ((zz >> 1) as i16) ^ -((zz & 1) as i16);
            (v as i8) as u8
        })
        .collect()
}

/// Per-plane transform choice (wire tag values are stable: do not renumber).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneXf {
    /// No transform: the plane bytes are FSE-coded directly.
    Raw = 0,
    /// Byte-wrapping first difference.
    Delta = 1,
    /// Byte-wrapping first difference, then zigzag (signed→unsigned).
    DeltaZigzag = 2,
}

impl PlaneXf {
    /// Wire tag for this transform.
    fn tag(self) -> u8 {
        self as u8
    }

    /// Parse a wire tag back into a transform.
    fn from_tag(tag: u8) -> PzResult<Self> {
        match tag {
            0 => Ok(PlaneXf::Raw),
            1 => Ok(PlaneXf::Delta),
            2 => Ok(PlaneXf::DeltaZigzag),
            _ => Err(PzError::InvalidInput),
        }
    }
}

/// Apply a plane transform forward.
fn apply_fwd(p: &[u8], xf: PlaneXf) -> Vec<u8> {
    match xf {
        PlaneXf::Raw => p.to_vec(),
        PlaneXf::Delta => delta_fwd(p),
        PlaneXf::DeltaZigzag => zigzag_fwd(&delta_fwd(p)),
    }
}

/// Invert a plane transform.
fn apply_inv(p: &[u8], xf: PlaneXf) -> Vec<u8> {
    match xf {
        PlaneXf::Raw => p.to_vec(),
        PlaneXf::Delta => delta_inv(p),
        PlaneXf::DeltaZigzag => delta_inv(&zigzag_inv(p)),
    }
}

// ---------------------------------------------------------------------------
// Per-plane gated FSE encode
// ---------------------------------------------------------------------------

/// Encode a single plane with its gated-best transform.
///
/// Tries `Raw`, `Delta`, and `Delta`+`zigzag`, FSE-encodes each candidate with
/// `fse::encode_best`, and returns the `(transform, fse_bytes)` pair with the
/// smallest encoded length. An empty plane encodes to `(Raw, empty)`.
fn encode_plane_gated(plane: &[u8]) -> (PlaneXf, Vec<u8>) {
    if plane.is_empty() {
        return (PlaneXf::Raw, Vec::new());
    }
    let mut best_xf = PlaneXf::Raw;
    let mut best_enc = fse::encode_best(plane);
    for xf in [PlaneXf::Delta, PlaneXf::DeltaZigzag] {
        let enc = fse::encode_best(&apply_fwd(plane, xf));
        if enc.len() < best_enc.len() {
            best_xf = xf;
            best_enc = enc;
        }
    }
    (best_xf, best_enc)
}

/// Encode a whole input at a fixed stride: per-plane gated FSE + remainder.
///
/// Returns the full wire-format body for `stride` (the leading stride byte,
/// then each plane's `[tag][len][fse]`, then the raw remainder).
fn encode_at_stride(input: &[u8], stride: usize) -> Vec<u8> {
    let (planes, remainder) = split_planes(input, stride);
    // Pre-size: 1 stride byte + per-plane (1 tag + 4 len) + payload + remainder.
    let mut out = Vec::with_capacity(1 + stride * 5 + input.len());
    out.push(stride as u8);
    for plane in &planes {
        let (xf, enc) = encode_plane_gated(plane);
        out.push(xf.tag());
        out.extend_from_slice(&(enc.len() as u32).to_le_bytes());
        out.extend_from_slice(&enc);
    }
    out.extend_from_slice(&remainder);
    out
}

// ---------------------------------------------------------------------------
// Stage-0 spike (num-bitpack): vertical bit-pack + zero-word plane coder.
//
// ndzip-style entropy-stage replacement candidate for the per-plane FSE above:
// each group of 32 consecutive plane bytes is bit-transposed into 8 bitplane
// words (word `b` holds bit `b` of all 32 bytes), all-zero words are dropped,
// and a 1-byte presence bitmap per group records which words survive.
//
// For byte planes this 32x8 transpose is the natural specialization of
// ndzip's 32x32 word transpose, at the same bitmap overhead (1 bit per 32
// payload bits = 3.125% floor) but strictly finer zero-word granularity.
//
// This is probe-only code: it is not wired into the wire format. See
// `examples/num_bitpack_probe.rs` and
// `docs/design-docs/num-bitpack-stage0-findings.md`.
// ---------------------------------------------------------------------------

pub mod bitpack {
    //! Vertical bit-packing + zero-word elimination for byte planes (spike).

    use crate::{PzError, PzResult};

    /// Values per transpose group (one presence-bitmap byte per group).
    const GROUP: usize = 32;

    /// Bit-transpose up to 32 bytes into 8 bitplane words. Missing tail bytes
    /// act as zero padding (their bits are simply never set).
    fn transpose_fwd(group: &[u8]) -> [u32; 8] {
        let mut words = [0u32; 8];
        for (i, &byte) in group.iter().enumerate() {
            let mut b = byte;
            for w in words.iter_mut() {
                *w |= ((b & 1) as u32) << i;
                b >>= 1;
            }
        }
        words
    }

    /// Encode one plane: per 32-byte group, a presence bitmap byte followed by
    /// the nonzero bitplane words (LE). Worst case is 33/32 of the input.
    pub fn encode(plane: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(plane.len() + plane.len() / GROUP + 1);
        for group in plane.chunks(GROUP) {
            let words = transpose_fwd(group);
            let mut bitmap = 0u8;
            for (b, &w) in words.iter().enumerate() {
                if w != 0 {
                    bitmap |= 1 << b;
                }
            }
            out.push(bitmap);
            for &w in &words {
                if w != 0 {
                    out.extend_from_slice(&w.to_le_bytes());
                }
            }
        }
        out
    }

    /// Exact inverse of [`encode`]; `len` is the original plane length.
    pub fn decode(data: &[u8], len: usize) -> PzResult<Vec<u8>> {
        let groups = len.div_ceil(GROUP);
        let mut out = Vec::with_capacity(groups * GROUP);
        let mut pos = 0usize;
        for _ in 0..groups {
            if pos >= data.len() {
                return Err(PzError::InvalidInput);
            }
            let bitmap = data[pos];
            pos += 1;
            let mut words = [0u32; 8];
            for (b, w) in words.iter_mut().enumerate() {
                if (bitmap >> b) & 1 == 1 {
                    if pos + 4 > data.len() {
                        return Err(PzError::InvalidInput);
                    }
                    *w = u32::from_le_bytes([
                        data[pos],
                        data[pos + 1],
                        data[pos + 2],
                        data[pos + 3],
                    ]);
                    pos += 4;
                }
            }
            for i in 0..GROUP {
                let mut byte = 0u8;
                for (b, &w) in words.iter().enumerate() {
                    byte |= (((w >> i) & 1) as u8) << b;
                }
                out.push(byte);
            }
        }
        if pos != data.len() {
            return Err(PzError::InvalidInput);
        }
        out.truncate(len);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Stage-0 spike probe: per-plane FSE vs bitpack byte counts.
// ---------------------------------------------------------------------------

/// Per-plane measurement from [`probe_block`] (num-bitpack stage-0 spike).
#[derive(Clone, Debug)]
pub struct PlaneProbe {
    /// Index of the plane within the chosen stride.
    pub plane_idx: usize,
    /// Untransformed plane length in bytes.
    pub plane_len: usize,
    /// Transform the shipping FSE gate chose for this plane.
    pub fse_xf: PlaneXf,
    /// Gated-best FSE size (exactly what ships today).
    pub fse_bytes: usize,
    /// Bitpack size of the FSE-chosen transform (front-end held identical).
    pub bp_same_bytes: usize,
    /// Bitpack size with its own transform gating (min over Raw/Delta/DeltaZigzag).
    pub bp_gated_bytes: usize,
    /// Transform the bitpack gate chose.
    pub bp_gated_xf: PlaneXf,
}

/// Block-level probe result: the stride the shipping encoder would pick for
/// this block (0 = STORE fallback, no planes), its per-plane measurements, and
/// the raw remainder length.
#[derive(Clone, Debug)]
pub struct BlockProbe {
    /// Chosen stride (0 means the block would be STORED raw).
    pub stride: usize,
    /// One entry per plane of the chosen stride.
    pub planes: Vec<PlaneProbe>,
    /// Raw remainder bytes (`len % stride`).
    pub remainder_len: usize,
}

/// Run the Num front-end on one block exactly as [`encode`] does (stride
/// sweep plus per-plane FSE-gated transform), then measure each plane of the
/// winning stride under both entropy stages. Probe-only; not on the wire.
pub fn probe_block(input: &[u8]) -> BlockProbe {
    // Replicate encode()'s stride selection: total gated FSE body size.
    let mut best: Option<(usize, usize)> = None; // (stride, body_len)
    for &s in &CANDIDATE_STRIDES {
        if s > input.len() {
            continue;
        }
        let (planes, remainder) = split_planes(input, s);
        let mut body = 1 + remainder.len();
        for plane in &planes {
            let (_, enc) = encode_plane_gated(plane);
            body += 5 + enc.len();
        }
        if best.is_none_or(|(_, b)| body < b) {
            best = Some((s, body));
        }
    }
    let store_len = 1 + input.len();
    let stride = match best {
        Some((s, body)) if body < store_len => s,
        _ => {
            return BlockProbe {
                stride: 0,
                planes: Vec::new(),
                remainder_len: input.len(),
            };
        }
    };

    let (planes, remainder) = split_planes(input, stride);
    let mut probes = Vec::with_capacity(stride);
    for (plane_idx, plane) in planes.iter().enumerate() {
        let (fse_xf, fse_enc) = encode_plane_gated(plane);
        let bp_same_bytes = bitpack::encode(&apply_fwd(plane, fse_xf)).len();
        let mut bp_gated_xf = PlaneXf::Raw;
        let mut bp_gated_bytes = usize::MAX;
        for xf in [PlaneXf::Raw, PlaneXf::Delta, PlaneXf::DeltaZigzag] {
            let n = bitpack::encode(&apply_fwd(plane, xf)).len();
            if n < bp_gated_bytes {
                bp_gated_bytes = n;
                bp_gated_xf = xf;
            }
        }
        probes.push(PlaneProbe {
            plane_idx,
            plane_len: plane.len(),
            fse_xf,
            fse_bytes: fse_enc.len(),
            bp_same_bytes,
            bp_gated_bytes,
            bp_gated_xf,
        });
    }
    BlockProbe {
        stride,
        planes: probes,
        remainder_len: remainder.len(),
    }
}

// ---------------------------------------------------------------------------
// Public codec API
// ---------------------------------------------------------------------------

/// Encode `input` with the numeric-decorrelation transform.
///
/// Sweeps the candidate strides, picks the one with the smallest total gated
/// FSE size, and emits the wire-format block. Falls back to a STORE block
/// (1-byte header + raw bytes) when no transform beats storing the bytes
/// verbatim, so the output never expands by more than the header.
///
/// The returned bytes are self-describing apart from the original length, which
/// the decoder must supply to [`decode`] (libpz stores it in the block table).
pub fn encode(input: &[u8]) -> Vec<u8> {
    // Empty input: a single STORE byte with no payload.
    if input.is_empty() {
        return vec![STORE_STRIDE];
    }

    let mut best: Option<Vec<u8>> = None;
    for &s in &CANDIDATE_STRIDES {
        // A stride larger than the input yields zero full records (all
        // remainder) — pointless, skip it.
        if s > input.len() {
            continue;
        }
        let body = encode_at_stride(input, s);
        if best.as_ref().is_none_or(|b| body.len() < b.len()) {
            best = Some(body);
        }
    }

    // STORE fallback: header byte + raw bytes. Chosen when no transform beat it
    // (or no stride was applicable, e.g. input shorter than the smallest stride).
    let store_len = 1 + input.len();
    match best {
        Some(body) if body.len() < store_len => body,
        _ => {
            let mut out = Vec::with_capacity(store_len);
            out.push(STORE_STRIDE);
            out.extend_from_slice(input);
            out
        }
    }
}

/// Encode `input` into a caller-provided buffer.
///
/// Returns the number of bytes written. Fails with [`PzError::BufferTooSmall`]
/// if `output` cannot hold the encoded block.
pub fn encode_to_buf(input: &[u8], output: &mut [u8]) -> PzResult<usize> {
    let encoded = encode(input);
    if output.len() < encoded.len() {
        return Err(PzError::BufferTooSmall);
    }
    output[..encoded.len()].copy_from_slice(&encoded);
    Ok(encoded.len())
}

/// Decode a numeric-decorrelation block produced by [`encode`].
///
/// `orig_len` is the length of the original (pre-transform) data; the decoder
/// derives `full = orig_len / stride` and `rem = orig_len % stride` from it.
/// Returns [`PzError::InvalidInput`] on any structural inconsistency — malformed
/// input must never panic.
pub fn decode(data: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if orig_len == 0 {
        // Valid encodings of empty input are a lone STORE byte or truly empty.
        return Ok(Vec::new());
    }
    if data.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let stride = data[0] as usize;
    let body = &data[1..];

    // STORE mode: the remaining bytes are the original verbatim.
    if data[0] == STORE_STRIDE {
        if body.len() != orig_len {
            return Err(PzError::InvalidInput);
        }
        return Ok(body.to_vec());
    }

    // Reject any stride the encoder would never emit (defensive: keeps the
    // accepted set tight and avoids absurd plane counts from corrupt input).
    if !CANDIDATE_STRIDES.contains(&stride) {
        return Err(PzError::InvalidInput);
    }

    let full = orig_len / stride;
    let rem = orig_len % stride;

    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(stride);
    let mut pos = 0usize;
    for _ in 0..stride {
        // Each plane header is [tag: u8][comp_len: u32 LE].
        if pos + 5 > body.len() {
            return Err(PzError::InvalidInput);
        }
        let xf = PlaneXf::from_tag(body[pos])?;
        pos += 1;
        let comp_len =
            u32::from_le_bytes([body[pos], body[pos + 1], body[pos + 2], body[pos + 3]]) as usize;
        pos += 4;

        if pos + comp_len > body.len() {
            return Err(PzError::InvalidInput);
        }
        let fse_bytes = &body[pos..pos + comp_len];
        pos += comp_len;

        // Every plane has exactly `full` elements. An empty plane (full == 0)
        // must carry an empty payload.
        if full == 0 {
            if comp_len != 0 {
                return Err(PzError::InvalidInput);
            }
            planes.push(Vec::new());
            continue;
        }
        let transformed = fse::decode(fse_bytes, full)?;
        if transformed.len() != full {
            return Err(PzError::InvalidInput);
        }
        planes.push(apply_inv(&transformed, xf));
    }

    // The remaining `rem` bytes are the stored remainder.
    let remainder = &body[pos..];
    if remainder.len() != rem {
        return Err(PzError::InvalidInput);
    }

    let out = join_planes(&planes, remainder, stride);
    if out.len() != orig_len {
        return Err(PzError::InvalidInput);
    }
    Ok(out)
}

/// Decode a numeric-decorrelation block into a caller-provided buffer.
///
/// Returns the number of bytes written (always `orig_len` on success). Fails
/// with [`PzError::BufferTooSmall`] if `output` is shorter than `orig_len`.
pub fn decode_to_buf(data: &[u8], orig_len: usize, output: &mut [u8]) -> PzResult<usize> {
    let decoded = decode(data, orig_len)?;
    if output.len() < decoded.len() {
        return Err(PzError::BufferTooSmall);
    }
    output[..decoded.len()].copy_from_slice(&decoded);
    Ok(decoded.len())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift PRNG (no external deps), matching the spike.
    struct XorShift(u64);
    impl XorShift {
        fn new() -> Self {
            XorShift(0x9E3779B97F4A7C15)
        }
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// The adversarial fixed inputs the spike used (extended a little).
    fn adversarial() -> Vec<Vec<u8>> {
        vec![
            vec![],
            vec![0],
            vec![255],
            vec![0, 0, 0, 0, 0],
            vec![255; 17],
            vec![0, 255, 0, 255, 128, 127],
            (0u8..=255).collect(),
            (0u8..=255).rev().collect(),
            vec![1, 2, 3, 4, 5, 6, 7], // prime length 7
            vec![7; 28 * 3 + 5],       // not a multiple of 28
            vec![42; 1],
            vec![3; 28],   // exactly one 28-byte record
            vec![9; 64],   // multiple of several strides
            vec![0; 1000], // long all-zero
        ]
    }

    // -- Transform-level fuzz: split/join + per-plane transforms, including a
    //    real FSE round-trip, exactly mirroring the validated spike's --fuzz. --

    #[test]
    fn fuzz_transforms_inverse_and_fse_roundtrip() {
        let mut inputs = adversarial();
        let mut rng = XorShift::new();
        for _ in 0..200 {
            let len = (rng.next() % 300) as usize;
            let v: Vec<u8> = (0..len).map(|_| (rng.next() & 0xff) as u8).collect();
            inputs.push(v);
        }

        let mut cases = 0usize;
        for data in &inputs {
            for &s in &[1usize, 2, 3, 4, 7, 8, 16, 28, 32] {
                // split/join identity
                let (planes, remainder) = split_planes(data, s);
                let rejoined = join_planes(&planes, &remainder, s);
                assert_eq!(&rejoined, data, "split/join FAIL s={s} len={}", data.len());

                for p in &planes {
                    for xf in [PlaneXf::Raw, PlaneXf::Delta, PlaneXf::DeltaZigzag] {
                        let f = apply_fwd(p, xf);
                        let inv = apply_inv(&f, xf);
                        cases += 1;
                        assert_eq!(&inv, p, "{xf:?} inverse FAIL s={s} plane_len={}", p.len());

                        // Must survive a real FSE encode_best → decode round-trip.
                        if !f.is_empty() {
                            let enc = fse::encode_best(&f);
                            let dec = fse::decode(&enc, f.len()).expect("fse decode");
                            assert_eq!(dec, f, "FSE roundtrip FAIL s={s} len={}", f.len());
                        }
                    }
                }
            }
        }
        assert!(cases > 0);
    }

    // -- Per-transform unit round-trips at several strides via encode/decode. --

    #[test]
    fn encode_decode_roundtrip_strides_and_modes() {
        // Construct inputs that should exercise each transform mode:
        //  - smooth ramp (delta-friendly)
        //  - 16-bit little-endian counter (transpose-friendly, S=2)
        //  - 4-column records with one slowly-varying column
        let mut inputs: Vec<Vec<u8>> = Vec::new();

        // 16-bit LE counter: low byte cycles, high byte ramps slowly.
        let mut le16 = Vec::new();
        for i in 0u16..4000 {
            le16.extend_from_slice(&i.to_le_bytes());
        }
        inputs.push(le16);

        // 4-byte records: [ramp, const, noise-ish, slow].
        let mut rec4 = Vec::new();
        let mut rng = XorShift::new();
        for i in 0u32..2000 {
            rec4.push((i & 0xff) as u8);
            rec4.push(0x55);
            rec4.push((rng.next() & 0xff) as u8);
            rec4.push((i / 16) as u8);
        }
        inputs.push(rec4);

        // 28-byte records (sao-like): a few smooth columns + filler.
        let mut rec28 = Vec::new();
        for i in 0u32..1500 {
            for c in 0..28u32 {
                rec28.push(((i.wrapping_mul(c + 1)) >> (c % 5)) as u8);
            }
        }
        inputs.push(rec28);

        for data in &inputs {
            let enc = encode(data);
            let dec = decode(&enc, data.len()).expect("decode");
            assert_eq!(
                &dec,
                data,
                "encode/decode roundtrip mismatch len={}",
                data.len()
            );
            // Encoder must not have chosen STORE for these compressible inputs.
            assert_ne!(enc[0], STORE_STRIDE, "expected a transform, got STORE");
        }
    }

    #[test]
    fn encode_decode_roundtrip_adversarial() {
        let mut inputs = adversarial();
        let mut rng = XorShift::new();
        for _ in 0..300 {
            let len = (rng.next() % 500) as usize;
            let v: Vec<u8> = (0..len).map(|_| (rng.next() & 0xff) as u8).collect();
            inputs.push(v);
        }
        for data in &inputs {
            let enc = encode(data);
            let dec = decode(&enc, data.len()).expect("decode");
            assert_eq!(
                &dec,
                data,
                "adversarial roundtrip mismatch len={}",
                data.len()
            );
        }
    }

    // -- STORE fallback: incompressible random data must round-trip and the
    //    output must not expand beyond the 1-byte header. --

    #[test]
    fn store_fallback_random_does_not_expand() {
        let mut rng = XorShift::new();
        // 8 KiB of high-entropy random bytes: no plane transform can help.
        let data: Vec<u8> = (0..8192).map(|_| (rng.next() & 0xff) as u8).collect();
        let enc = encode(&data);
        let dec = decode(&enc, data.len()).expect("decode");
        assert_eq!(dec, data, "random roundtrip mismatch");
        // STORE fallback guarantees at most a 1-byte header of overhead.
        assert!(
            enc.len() <= data.len() + 1,
            "expanded too much: {} -> {}",
            data.len(),
            enc.len()
        );
        assert_eq!(
            enc[0], STORE_STRIDE,
            "random data should hit STORE fallback"
        );
    }

    #[test]
    fn empty_and_tiny_inputs() {
        for data in [vec![], vec![0u8], vec![1u8, 2], vec![5u8; 3]] {
            let enc = encode(&data);
            let dec = decode(&enc, data.len()).expect("decode");
            assert_eq!(dec, data, "tiny roundtrip mismatch len={}", data.len());
            assert!(enc.len() <= data.len() + 1 + 64, "tiny output too large");
        }
        // Empty input encodes to a single STORE byte and decodes to empty.
        assert_eq!(encode(&[]), vec![STORE_STRIDE]);
        assert_eq!(decode(&[STORE_STRIDE], 0).unwrap(), Vec::<u8>::new());
        // Empty data buffer with orig_len 0 also decodes to empty.
        assert_eq!(decode(&[], 0).unwrap(), Vec::<u8>::new());
    }

    // -- Malformed input must error, never panic. --

    #[test]
    fn decode_rejects_malformed() {
        // Non-empty orig_len but empty data.
        assert_eq!(decode(&[], 10), Err(PzError::InvalidInput));
        // STORE stride but wrong body length.
        assert_eq!(
            decode(&[STORE_STRIDE, 1, 2, 3], 10),
            Err(PzError::InvalidInput)
        );
        // Stride the encoder never emits.
        assert_eq!(decode(&[3, 0, 0, 0, 0, 0], 6), Err(PzError::InvalidInput));
        // Valid stride byte but truncated plane header.
        assert_eq!(decode(&[2, 0], 100), Err(PzError::InvalidInput));
        // Valid stride, plane claims more bytes than present.
        let mut bad = vec![2u8, 0u8];
        bad.extend_from_slice(&(9999u32).to_le_bytes());
        assert_eq!(decode(&bad, 100), Err(PzError::InvalidInput));
        // Bad transform tag.
        let mut bad_tag = vec![2u8, 7u8];
        bad_tag.extend_from_slice(&(0u32).to_le_bytes());
        assert!(decode(&bad_tag, 0).is_ok()); // orig_len 0 short-circuits
        let mut bad_tag2 = vec![2u8, 9u8];
        bad_tag2.extend_from_slice(&(1u32).to_le_bytes());
        bad_tag2.push(0);
        assert_eq!(decode(&bad_tag2, 2), Err(PzError::InvalidInput));
    }

    // -- Stage-0 spike: bitpack coder must be an exact inverse on plane data. --

    #[test]
    fn bitpack_roundtrip_adversarial_and_fuzz() {
        let mut inputs = adversarial();
        let mut rng = XorShift::new();
        for _ in 0..300 {
            let len = (rng.next() % 700) as usize;
            // Mix of high-entropy and sparse (low-bit-only) bytes.
            let mask = if rng.next().is_multiple_of(2) {
                0xff
            } else {
                0x03
            };
            let v: Vec<u8> = (0..len).map(|_| (rng.next() as u8) & mask).collect();
            inputs.push(v);
        }
        for data in &inputs {
            let enc = bitpack::encode(data);
            let dec = bitpack::decode(&enc, data.len()).expect("bitpack decode");
            assert_eq!(&dec, data, "bitpack roundtrip FAIL len={}", data.len());
            // Worst case per group (even a partial tail group) is one bitmap
            // byte plus all 8 bitplane words: 33 bytes per ceil(len/32) groups.
            assert!(enc.len() <= data.len().div_ceil(32) * 33);
        }
        // Malformed input must error, never panic.
        assert_eq!(bitpack::decode(&[], 5), Err(PzError::InvalidInput));
        assert_eq!(
            bitpack::decode(&[0xff, 1, 2], 5),
            Err(PzError::InvalidInput)
        );
        assert_eq!(bitpack::decode(&[0, 0], 5), Err(PzError::InvalidInput));
    }

    /// Round-trip on *real* plane data: run the actual Num front-end (stride
    /// sweep + per-plane gated transform) on structured record data and check
    /// the bitpack coder exactly inverts every transformed plane.
    #[test]
    fn bitpack_roundtrip_real_front_end_planes() {
        // sao-like 28-byte records + a 16-bit LE ramp, as in the codec tests.
        let mut rec28 = Vec::new();
        for i in 0u32..1500 {
            for c in 0..28u32 {
                rec28.push(((i.wrapping_mul(c + 1)) >> (c % 5)) as u8);
            }
        }
        let le16: Vec<u8> = (0u16..4000).flat_map(|i| i.to_le_bytes()).collect();

        for data in [&rec28, &le16] {
            let probe = probe_block(data);
            assert_ne!(probe.stride, 0, "front-end should not STORE this input");
            let (planes, _) = split_planes(data, probe.stride);
            for (p, plane) in probe.planes.iter().zip(&planes) {
                for xf in [PlaneXf::Raw, PlaneXf::Delta, PlaneXf::DeltaZigzag] {
                    let t = apply_fwd(plane, xf);
                    let enc = bitpack::encode(&t);
                    let dec = bitpack::decode(&enc, t.len()).expect("decode");
                    assert_eq!(dec, t, "plane {} xf {:?} roundtrip FAIL", p.plane_idx, xf);
                }
                // Probe sizes must match a fresh encode of the same plane.
                assert_eq!(
                    p.bp_same_bytes,
                    bitpack::encode(&apply_fwd(plane, p.fse_xf)).len()
                );
            }
        }
    }

    #[test]
    fn to_buf_variants() {
        let data: Vec<u8> = (0u16..1000).flat_map(|i| i.to_le_bytes()).collect();
        let mut enc_buf = vec![0u8; data.len() + 64 + 6 * 32];
        let n = encode_to_buf(&data, &mut enc_buf).expect("encode_to_buf");
        assert_eq!(&enc_buf[..n], &encode(&data)[..]);

        let mut dec_buf = vec![0u8; data.len()];
        let m = decode_to_buf(&enc_buf[..n], data.len(), &mut dec_buf).expect("decode_to_buf");
        assert_eq!(m, data.len());
        assert_eq!(&dec_buf[..m], &data[..]);

        // Too-small buffers must error, not panic.
        let mut tiny = vec![0u8; 1];
        assert_eq!(
            encode_to_buf(&data, &mut tiny),
            Err(PzError::BufferTooSmall)
        );
        assert_eq!(
            decode_to_buf(&enc_buf[..n], data.len(), &mut tiny),
            Err(PzError::BufferTooSmall)
        );
    }
}
