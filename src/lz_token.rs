/// Universal LZ token type and pluggable wire encoding trait.
///
/// Match finders and parsers produce `Vec<LzToken>`. Wire encoders convert
/// token streams into independent byte streams for entropy coding.
///
/// ## Encoders
///
/// - `Lz77Encoder`: DEFLATE-compatible (u16 offset, u16 length, u8 next).
///   3 streams. Legacy format (Deflate pipeline removed).
/// - `LzSeqEncoder`: log2-coded offsets/lengths with repeat offsets.
///   6 streams. Best ratio. Used by LzSeqR, LzSeqH, Lzf, SortLz.
/// - `LzssEncoder`: flag bits + raw u16 offsets/lengths.
///   4 streams. Used by Lzfi, LzssR.
use crate::{PzError, PzResult};

/// Universal LZ token — output of match finding + parsing.
///
/// Widened to u32 offset/length to support LzSeq's 128KB+ windows.
#[derive(Debug, Clone, Copy)]
pub(crate) enum LzToken {
    Literal(u8),
    Match { offset: u32, length: u32 },
}

/// Output of a token encoder — independent byte streams for entropy coding.
pub(crate) struct EncodedStreams {
    /// Independent byte streams (one per encoder channel).
    pub streams: Vec<Vec<u8>>,
    /// Opaque metadata that must round-trip through the entropy container.
    pub meta: Vec<u8>,
    /// Length of the pre-entropy data (for container metadata).
    pub pre_entropy_len: usize,
}

/// Pluggable wire encoding for LZ token streams.
pub(crate) trait TokenEncoder {
    /// Encode a token stream into independent byte streams.
    fn encode(&self, input: &[u8], tokens: &[LzToken]) -> PzResult<EncodedStreams>;
    /// Decode streams back to original bytes.
    fn decode(&self, streams: Vec<Vec<u8>>, meta: &[u8], original_len: usize) -> PzResult<Vec<u8>>;
}

// ---------------------------------------------------------------------------
// Lz77Encoder — DEFLATE-compatible (3 streams)
// ---------------------------------------------------------------------------

/// DEFLATE-compatible encoder: (u16 offset, u16 length, u8 next) triples
/// split into 3 streams (offsets, lengths, literals).
///
/// Every Match token must be followed by a Literal token. Pure literals
/// are encoded as offset=0, length=0, next=literal.
pub(crate) struct Lz77Encoder;

impl TokenEncoder for Lz77Encoder {
    fn encode(&self, input: &[u8], tokens: &[LzToken]) -> PzResult<EncodedStreams> {
        // Estimate: worst case every position is a literal → one triple per literal.
        let est = tokens.len();
        let mut offsets = Vec::with_capacity(est * 2);
        let mut lengths = Vec::with_capacity(est * 2);
        let mut literals = Vec::with_capacity(est);

        // Walk the token stream, pairing each Match with the following Literal.
        // Pure literals (no preceding match) become offset=0, length=0, next=byte.
        //
        // DEFLATE invariant: every match must be followed by a literal byte.
        // When two consecutive matches appear or a match ends the stream,
        // we shorten the match by 1 and use the actual byte at the boundary
        // (looked up from `input`) as the trailing literal.
        let mut input_pos = 0usize;
        let mut i = 0;
        while i < tokens.len() {
            debug_assert!(
                input_pos <= input.len(),
                "Lz77Encoder: input_pos ({input_pos}) overran input ({})",
                input.len()
            );
            match tokens[i] {
                LzToken::Match { offset, length } => {
                    let offset_u16 = offset.min(u16::MAX as u32) as u16;
                    let len = length as usize;

                    // Check if the next token is a Literal.
                    let next_is_literal =
                        i + 1 < tokens.len() && matches!(tokens[i + 1], LzToken::Literal(_));

                    if next_is_literal {
                        // Normal case: match paired with following literal.
                        let length_u16 = length.min(u16::MAX as u32) as u16;
                        offsets.extend_from_slice(&offset_u16.to_le_bytes());
                        lengths.extend_from_slice(&length_u16.to_le_bytes());
                        input_pos += len;
                        i += 1;
                        if let LzToken::Literal(b) = tokens[i] {
                            literals.push(b);
                            input_pos += 1;
                        }
                    } else {
                        // No following literal (consecutive match or end of stream).
                        // Shorten match by 1 and use the actual input byte as the literal.
                        let adj_len = len.saturating_sub(1);
                        offsets.extend_from_slice(&offset_u16.to_le_bytes());
                        lengths.extend_from_slice(
                            &(adj_len.min(u16::MAX as usize) as u16).to_le_bytes(),
                        );
                        input_pos += adj_len;
                        // Use the actual byte from input at the boundary.
                        if input_pos < input.len() {
                            literals.push(input[input_pos]);
                            input_pos += 1;
                        } else {
                            literals.push(0);
                        }
                    }
                }
                LzToken::Literal(b) => {
                    offsets.extend_from_slice(&0u16.to_le_bytes());
                    lengths.extend_from_slice(&0u16.to_le_bytes());
                    literals.push(b);
                    input_pos += 1;
                }
            }
            i += 1;
        }

        let num_triples = literals.len();
        let pre_entropy_len = num_triples * 5; // 2 + 2 + 1 per triple

        Ok(EncodedStreams {
            streams: vec![offsets, lengths, literals],
            meta: Vec::new(),
            pre_entropy_len,
        })
    }

    fn decode(
        &self,
        streams: Vec<Vec<u8>>,
        _meta: &[u8],
        original_len: usize,
    ) -> PzResult<Vec<u8>> {
        if streams.len() != 3 {
            return Err(PzError::InvalidInput);
        }
        let offsets = &streams[0];
        let lengths = &streams[1];
        let literals = &streams[2];

        if offsets.len() != lengths.len() || offsets.len() != literals.len() * 2 {
            return Err(PzError::InvalidInput);
        }
        let num_triples = literals.len();
        let mut output = vec![0u8; original_len];
        let mut out_pos = 0usize;

        for i in 0..num_triples {
            let offset = u16::from_le_bytes([offsets[i * 2], offsets[i * 2 + 1]]) as usize;
            let length = u16::from_le_bytes([lengths[i * 2], lengths[i * 2 + 1]]) as usize;
            let next = literals[i];

            if out_pos + length + 1 > output.len() {
                return Err(PzError::BufferTooSmall);
            }

            if length > 0 {
                if offset > out_pos {
                    return Err(PzError::InvalidInput);
                }
                for _ in 0..length {
                    let src = out_pos - offset;
                    output[out_pos] = output[src];
                    out_pos += 1;
                }
            }

            output[out_pos] = next;
            out_pos += 1;
        }

        if out_pos != original_len {
            return Err(PzError::InvalidInput);
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// LzSeqEncoder — log2-coded (6 streams)
// ---------------------------------------------------------------------------

/// Log2-coded offset/length encoder with repeat offsets.
/// Produces 6 streams: flags, literals, offset_codes, offset_extra,
/// length_codes, length_extra.
pub(crate) struct LzSeqEncoder {
    /// Maximum lookback window size (for SeqConfig).
    pub max_window: usize,
}

impl Default for LzSeqEncoder {
    fn default() -> Self {
        LzSeqEncoder {
            max_window: crate::lzseq::SeqConfig::default().max_window,
        }
    }
}

impl TokenEncoder for LzSeqEncoder {
    fn encode(&self, _input: &[u8], tokens: &[LzToken]) -> PzResult<EncodedStreams> {
        let config = crate::lzseq::SeqConfig {
            max_window: self.max_window,
            ..crate::lzseq::SeqConfig::default()
        };
        let enc = crate::lzseq::encode_from_tokens(tokens, &config)?;
        let pre_entropy_len = enc.flags.len()
            + enc.literals.len()
            + enc.offset_codes.len()
            + enc.offset_extra.len()
            + enc.length_codes.len()
            + enc.length_extra.len();
        let mut meta = Vec::with_capacity(8);
        meta.extend_from_slice(&enc.num_tokens.to_le_bytes());
        meta.extend_from_slice(&enc.num_matches.to_le_bytes());
        Ok(EncodedStreams {
            streams: vec![
                enc.flags,
                enc.literals,
                enc.offset_codes,
                enc.offset_extra,
                enc.length_codes,
                enc.length_extra,
            ],
            meta,
            pre_entropy_len,
        })
    }

    fn decode(&self, streams: Vec<Vec<u8>>, meta: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
        if streams.len() != 6 {
            return Err(PzError::InvalidInput);
        }
        if meta.len() < 8 {
            return Err(PzError::InvalidInput);
        }
        let num_tokens = u32::from_le_bytes(meta[..4].try_into().unwrap());
        let num_matches = u32::from_le_bytes(meta[4..8].try_into().unwrap());

        crate::lzseq::decode(
            &streams[0], // flags
            &streams[1], // literals
            &streams[2], // offset_codes
            &streams[3], // offset_extra
            &streams[4], // length_codes
            &streams[5], // length_extra
            num_tokens,
            num_matches,
            original_len,
        )
    }
}

// ---------------------------------------------------------------------------
// LzssEncoder — flags + raw u16 (4 streams)
// ---------------------------------------------------------------------------

/// LZSS encoder: flag bits (1=literal, 0=match) + raw u16 offsets/lengths.
/// Produces 4 streams: flags, literals, offsets, lengths.
pub(crate) struct LzssEncoder;

impl TokenEncoder for LzssEncoder {
    fn encode(&self, _input: &[u8], tokens: &[LzToken]) -> PzResult<EncodedStreams> {
        let num_tokens = tokens.len();
        let flag_bytes = num_tokens.div_ceil(8);
        let mut flags = vec![0u8; flag_bytes];
        let mut literals = Vec::new();
        let mut offsets = Vec::new();
        let mut lengths = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            match token {
                LzToken::Literal(b) => {
                    flags[i / 8] |= 1 << (7 - (i % 8));
                    literals.push(*b);
                }
                LzToken::Match { offset, length } => {
                    let offset_u16 = (*offset).min(u16::MAX as u32) as u16;
                    let length_u16 = (*length).min(u16::MAX as u32) as u16;
                    offsets.extend_from_slice(&offset_u16.to_le_bytes());
                    lengths.extend_from_slice(&length_u16.to_le_bytes());
                }
            }
        }

        let pre_entropy_len = flag_bytes + literals.len() + offsets.len() + lengths.len();
        let meta = (num_tokens as u32).to_le_bytes().to_vec();

        Ok(EncodedStreams {
            streams: vec![flags, literals, offsets, lengths],
            meta,
            pre_entropy_len,
        })
    }

    fn decode(&self, streams: Vec<Vec<u8>>, meta: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
        if streams.len() != 4 {
            return Err(PzError::InvalidInput);
        }
        if meta.len() < 4 {
            return Err(PzError::InvalidInput);
        }
        let num_tokens = u32::from_le_bytes(meta[..4].try_into().unwrap()) as usize;

        let flags = &streams[0];
        let literals = &streams[1];
        let offsets_raw = &streams[2];
        let lengths_raw = &streams[3];

        let required_flag_bytes = num_tokens.div_ceil(8);
        if required_flag_bytes > flags.len() {
            return Err(PzError::InvalidInput);
        }

        let mut output = Vec::with_capacity(original_len);
        let mut lit_idx = 0;
        let mut match_idx = 0;

        for i in 0..num_tokens {
            let is_literal = flags[i / 8] & (1 << (7 - (i % 8))) != 0;
            if is_literal {
                if lit_idx >= literals.len() {
                    return Err(PzError::InvalidInput);
                }
                output.push(literals[lit_idx]);
                lit_idx += 1;
            } else {
                let off_pos = match_idx * 2;
                if off_pos + 2 > offsets_raw.len() || off_pos + 2 > lengths_raw.len() {
                    return Err(PzError::InvalidInput);
                }
                let offset =
                    u16::from_le_bytes([offsets_raw[off_pos], offsets_raw[off_pos + 1]]) as usize;
                let length =
                    u16::from_le_bytes([lengths_raw[off_pos], lengths_raw[off_pos + 1]]) as usize;

                if offset == 0 || offset > output.len() {
                    return Err(PzError::InvalidInput);
                }

                for _ in 0..length {
                    let src = output.len() - offset;
                    let b = output[src];
                    output.push(b);
                }

                match_idx += 1;
            }
        }

        if output.len() != original_len {
            return Err(PzError::InvalidInput);
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert a sequence of `lz77::Match` structs to `LzToken`s.
///
/// Each Match{offset, length, next} becomes:
/// - If length > 0: Match{offset, length}, Literal(next)
/// - If length == 0: Literal(next)
pub(crate) fn matches_to_tokens(matches: &[crate::lz77::Match]) -> Vec<LzToken> {
    let mut tokens = Vec::with_capacity(matches.len() * 2);
    for m in matches {
        if m.length > 0 {
            tokens.push(LzToken::Match {
                offset: m.offset as u32,
                length: m.length as u32,
            });
        }
        tokens.push(LzToken::Literal(m.next));
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_input() -> Vec<u8> {
        b"abcabcabcabcabcabc hello world hello world test".to_vec()
    }

    fn make_tokens(input: &[u8]) -> Vec<LzToken> {
        // Use LZ77 match finder to get realistic tokens.
        let matches = crate::lz77::compress_lazy_to_matches(input).unwrap();
        matches_to_tokens(&matches)
    }

    #[test]
    fn lz77_encoder_roundtrip() {
        let input = test_input();
        let tokens = make_tokens(&input);
        let encoder = Lz77Encoder;
        let encoded = encoder.encode(&input, &tokens).unwrap();
        assert_eq!(encoded.streams.len(), 3);
        let decoded = encoder
            .decode(encoded.streams, &encoded.meta, input.len())
            .unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn lzseq_encoder_roundtrip() {
        let input = test_input();
        let tokens = make_tokens(&input);
        let encoder = LzSeqEncoder::default();
        let encoded = encoder.encode(&input, &tokens).unwrap();
        assert_eq!(encoded.streams.len(), 6);
        let decoded = encoder
            .decode(encoded.streams, &encoded.meta, input.len())
            .unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn lzss_encoder_roundtrip() {
        let input = test_input();
        let tokens = make_tokens(&input);
        let encoder = LzssEncoder;
        let encoded = encoder.encode(&input, &tokens).unwrap();
        assert_eq!(encoded.streams.len(), 4);
        let decoded = encoder
            .decode(encoded.streams, &encoded.meta, input.len())
            .unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn all_literals_roundtrip() {
        // Input with no matches (all unique bytes).
        let input: Vec<u8> = (0..=255).collect();
        let tokens: Vec<LzToken> = input.iter().map(|&b| LzToken::Literal(b)).collect();

        for encoder in [
            &Lz77Encoder as &dyn TokenEncoder,
            &LzSeqEncoder::default(),
            &LzssEncoder,
        ] {
            let encoded = encoder.encode(&input, &tokens).unwrap();
            let decoded = encoder
                .decode(encoded.streams, &encoded.meta, input.len())
                .unwrap();
            assert_eq!(decoded, input);
        }
    }

    #[test]
    fn matches_to_tokens_conversion() {
        let matches = vec![
            crate::lz77::Match {
                offset: 0,
                length: 0,
                next: b'a',
            },
            crate::lz77::Match {
                offset: 3,
                length: 5,
                next: b'b',
            },
        ];
        let tokens = matches_to_tokens(&matches);
        assert_eq!(tokens.len(), 3); // Literal(a), Match(3,5), Literal(b)
        assert!(matches!(tokens[0], LzToken::Literal(b'a')));
        assert!(matches!(
            tokens[1],
            LzToken::Match {
                offset: 3,
                length: 5
            }
        ));
        assert!(matches!(tokens[2], LzToken::Literal(b'b')));
    }
}
