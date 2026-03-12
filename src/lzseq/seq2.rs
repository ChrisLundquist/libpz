/// LzSeq2: literal-run-length sequence encoding (zstd-style).
///
/// Replaces per-token flag bits with per-sequence literal-run-length codes.
/// Each "sequence" is (literal_run_length, match_offset, match_length).
/// Trailing literals after the last match are implicit (remaining bytes
/// in the literals stream).
///
/// ## Output: 5 streams (4 entropy-coded + 1 raw)
///
/// - lit_run_codes: 1 byte per sequence (log2-coded literal run length, 0-based via +1 bias)
/// - offset_codes:  1 byte per sequence (repeat-offset-aware, same as LzSeq)
/// - length_codes:  1 byte per sequence (log2-coded match length, same as LzSeq)
/// - literals:      concatenated literal bytes
/// - packed_extra:  single raw bitstream — per sequence: lit_run_extra + offset_extra + length_extra
///
/// The first 4 streams are entropy-coded (rANS with sparse freq tables).
/// Stream 5 (packed_extra) is raw — near-uniform bits that rANS can't compress.
use crate::lz_token::LzToken;
use crate::{PzError, PzResult};

use super::{
    decode_value, encode_length, encode_value, extra_bits_for_code, extra_bits_for_offset_code,
    BitReader, BitWriter, RepeatOffsets,
};

/// Encoded output: 5 streams ready for entropy coding.
pub(crate) struct Seq2Encoded {
    /// Log2-coded literal run-length codes (1 byte per sequence).
    pub lit_run_codes: Vec<u8>,
    /// Log2-coded offset codes with repeat offsets (1 byte per sequence).
    pub offset_codes: Vec<u8>,
    /// Log2-coded match length codes (1 byte per sequence).
    pub length_codes: Vec<u8>,
    /// Concatenated literal byte values.
    pub literals: Vec<u8>,
    /// Combined packed extra bits (lit_run + offset + length per sequence).
    pub packed_extra: Vec<u8>,
    /// Number of sequences (matches). Trailing literals are implicit.
    pub num_sequences: u32,
}

/// Encode a literal run length (0-based) using the log2 code table.
///
/// Maps run_len 0 → value 1 → code 0, run_len 1 → value 2 → code 1, etc.
/// This reuses the existing code table with a +1 bias.
#[inline]
fn encode_lit_run(run_len: u32) -> (u8, u8, u32) {
    encode_value(run_len + 1)
}

/// Decode a literal run length from code + extra_value.
#[inline]
fn decode_lit_run(code: u8, extra_value: u32) -> u32 {
    decode_value(code, extra_value) - 1
}

/// Encode a universal `LzToken` stream into LzSeq2's 5-stream format.
pub(crate) fn encode_from_tokens(tokens: &[LzToken]) -> PzResult<Seq2Encoded> {
    let mut repeats = RepeatOffsets::new();
    let mut lit_run_codes: Vec<u8> = Vec::new();
    let mut offset_codes: Vec<u8> = Vec::new();
    let mut length_codes: Vec<u8> = Vec::new();
    let mut literals: Vec<u8> = Vec::new();
    let mut extra_writer = BitWriter::new();

    let mut pending_lits: u32 = 0;

    for token in tokens {
        match token {
            LzToken::Literal(b) => {
                literals.push(*b);
                pending_lits += 1;
            }
            LzToken::Match { offset, length } => {
                // Emit sequence: (lit_run_len, offset, length)
                let (lrc, lreb, lrev) = encode_lit_run(pending_lits);
                lit_run_codes.push(lrc);
                extra_writer.write_bits(lrev, lreb);

                let (oc, oeb, oev) = repeats.encode_offset(*offset);
                offset_codes.push(oc);
                extra_writer.write_bits(oev, oeb);

                let (lc, leb, lev) = encode_length(*length as u16);
                length_codes.push(lc);
                extra_writer.write_bits(lev, leb);

                pending_lits = 0;
            }
        }
    }

    // Trailing literals are implicit — they remain in the literals stream
    // and the decoder consumes them after all sequences.
    let num_sequences = offset_codes.len() as u32;

    Ok(Seq2Encoded {
        lit_run_codes,
        offset_codes,
        length_codes,
        literals,
        packed_extra: extra_writer.finish(),
        num_sequences,
    })
}

/// Decode LzSeq2 streams back to original bytes.
pub(crate) fn decode(
    lit_run_codes: &[u8],
    offset_codes: &[u8],
    length_codes: &[u8],
    literals: &[u8],
    packed_extra: &[u8],
    num_sequences: u32,
    original_len: usize,
) -> PzResult<Vec<u8>> {
    let ns = num_sequences as usize;
    if lit_run_codes.len() < ns || offset_codes.len() < ns || length_codes.len() < ns {
        return Err(PzError::InvalidInput);
    }

    let mut output = Vec::with_capacity(original_len);
    let mut repeats = RepeatOffsets::new();
    let mut extra_reader = BitReader::new(packed_extra);
    let mut lit_pos: usize = 0;

    for i in 0..ns {
        // Decode literal run length
        let lrc = lit_run_codes[i];
        let lreb = extra_bits_for_code(lrc);
        let lrev = extra_reader.read_bits(lreb);
        let lit_run = decode_lit_run(lrc, lrev) as usize;

        // Copy literals
        if lit_pos + lit_run > literals.len() {
            return Err(PzError::InvalidInput);
        }
        output.extend_from_slice(&literals[lit_pos..lit_pos + lit_run]);
        lit_pos += lit_run;

        // Decode offset
        let oc = offset_codes[i];
        let oeb = extra_bits_for_offset_code(oc);
        let oev = extra_reader.read_bits(oeb);
        let offset = repeats.decode_offset(oc, oev);

        // Decode match length
        let lc = length_codes[i];
        let leb = extra_bits_for_code(lc);
        let lev = extra_reader.read_bits(leb);
        let match_len = super::decode_length(lc, lev) as usize;

        // Copy match
        if offset == 0 || offset as usize > output.len() {
            return Err(PzError::InvalidInput);
        }
        for _ in 0..match_len {
            let src = output.len() - offset as usize;
            let b = output[src];
            output.push(b);
        }
    }

    // Trailing literals after the last match
    if lit_pos < literals.len() {
        output.extend_from_slice(&literals[lit_pos..]);
    }

    if output.len() != original_len {
        return Err(PzError::InvalidInput);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokens(input: &[u8]) -> Vec<LzToken> {
        let matches = crate::lz77::compress_lazy_to_matches(input).unwrap();
        crate::lz_token::matches_to_tokens(&matches)
    }

    #[test]
    fn roundtrip_basic() {
        let input = b"abcabcabcabcabcabc hello world hello world test";
        let tokens = make_tokens(input);
        let enc = encode_from_tokens(&tokens).unwrap();
        let dec = decode(
            &enc.lit_run_codes,
            &enc.offset_codes,
            &enc.length_codes,
            &enc.literals,
            &enc.packed_extra,
            enc.num_sequences,
            input.len(),
        )
        .unwrap();
        assert_eq!(dec, input);
    }

    #[test]
    fn roundtrip_all_literals() {
        let input: Vec<u8> = (0..=255).collect();
        let tokens: Vec<LzToken> = input.iter().map(|&b| LzToken::Literal(b)).collect();
        let enc = encode_from_tokens(&tokens).unwrap();
        assert_eq!(enc.num_sequences, 0);
        let dec = decode(
            &enc.lit_run_codes,
            &enc.offset_codes,
            &enc.length_codes,
            &enc.literals,
            &enc.packed_extra,
            enc.num_sequences,
            input.len(),
        )
        .unwrap();
        assert_eq!(dec, input);
    }

    #[test]
    fn roundtrip_empty() {
        let tokens: Vec<LzToken> = Vec::new();
        let enc = encode_from_tokens(&tokens).unwrap();
        let dec = decode(
            &enc.lit_run_codes,
            &enc.offset_codes,
            &enc.length_codes,
            &enc.literals,
            &enc.packed_extra,
            enc.num_sequences,
            0,
        )
        .unwrap();
        assert!(dec.is_empty());
    }

    #[test]
    fn roundtrip_consecutive_matches() {
        // Tokens: lit, lit, lit, match, match (consecutive matches = lit_run 0)
        let input = b"abcabcabc";
        let tokens = make_tokens(input);
        let enc = encode_from_tokens(&tokens).unwrap();
        let dec = decode(
            &enc.lit_run_codes,
            &enc.offset_codes,
            &enc.length_codes,
            &enc.literals,
            &enc.packed_extra,
            enc.num_sequences,
            input.len(),
        )
        .unwrap();
        assert_eq!(dec, input);
    }

    #[test]
    fn roundtrip_large_repetitive() {
        let input: Vec<u8> = b"the quick brown fox jumps over the lazy dog "
            .iter()
            .copied()
            .cycle()
            .take(10000)
            .collect();
        let tokens = make_tokens(&input);
        let enc = encode_from_tokens(&tokens).unwrap();
        let dec = decode(
            &enc.lit_run_codes,
            &enc.offset_codes,
            &enc.length_codes,
            &enc.literals,
            &enc.packed_extra,
            enc.num_sequences,
            input.len(),
        )
        .unwrap();
        assert_eq!(dec, input);
    }

    #[test]
    fn lit_run_code_table() {
        // Verify round-trip of literal run length coding
        for run in 0..1000u32 {
            let (code, eb, ev) = encode_lit_run(run);
            let decoded = decode_lit_run(code, ev);
            assert_eq!(
                decoded, run,
                "failed for run={run}, code={code}, eb={eb}, ev={ev}"
            );
            assert_eq!(eb, extra_bits_for_code(code));
        }
    }
}
