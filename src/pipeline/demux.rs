//! Stream demuxer: splits pre-entropy stage output into independent byte streams
//! for entropy coding, and re-merges them on decompression.

use crate::lz77;
use crate::lz_token;
use crate::lzseq;
use crate::lzss;
use crate::{PzError, PzResult};

#[cfg(feature = "webgpu")]
use super::Backend;
use super::{CompressOptions, ParseStrategy};

/// Output from a demuxer's compress-and-split operation.
pub(crate) struct DemuxOutput {
    /// Independent byte streams for entropy coding.
    pub streams: Vec<Vec<u8>>,
    /// Length of the pre-entropy data (e.g., total LZ output before splitting).
    pub pre_entropy_len: usize,
    /// Opaque metadata that must round-trip through the entropy container.
    pub meta: Vec<u8>,
}

impl From<lz_token::EncodedStreams> for DemuxOutput {
    fn from(enc: lz_token::EncodedStreams) -> Self {
        DemuxOutput {
            streams: enc.streams,
            pre_entropy_len: enc.pre_entropy_len,
            meta: enc.meta,
        }
    }
}

/// Describes how a pre-entropy stage (LZ77, LZSS, LzSeq, etc.) splits its
/// output into independent byte streams for entropy coding, and merges
/// them back on decompression.
pub(crate) trait StreamDemuxer {
    /// Number of independent streams this format produces.
    fn stream_count(&self) -> usize;

    /// Compress input bytes and split into independent streams + metadata.
    fn compress_and_demux(&self, input: &[u8], options: &CompressOptions) -> PzResult<DemuxOutput>;

    /// Reinterleave decoded streams + metadata back into decompressed output.
    fn remux_and_decompress(
        &self,
        streams: Vec<Vec<u8>>,
        meta: &[u8],
        original_len: usize,
    ) -> PzResult<Vec<u8>>;
}

/// Concrete LZ demuxer variants (enum dispatch, no dyn/vtable overhead).
pub(crate) enum LzDemuxer {
    /// LZ77: 3 streams (offsets, lengths, literals).
    #[allow(dead_code)]
    Lz77,
    /// LZSS: 4 streams (flags, literals, offsets, lengths).
    Lzss,
    /// LzSeq: 6 streams (flags, literals, offset_codes, offset_extra, length_codes, length_extra).
    LzSeq,
}

/// Map a pipeline to its demuxer, if it uses one.
/// Returns `None` for BWT-based pipelines (Bw, Bbw).
pub(crate) fn demuxer_for_pipeline(pipeline: super::Pipeline) -> Option<LzDemuxer> {
    match pipeline {
        super::Pipeline::Lzf => Some(LzDemuxer::LzSeq),
        super::Pipeline::Lzfi | super::Pipeline::LzssR => Some(LzDemuxer::Lzss),
        super::Pipeline::LzSeqR | super::Pipeline::LzSeqH => Some(LzDemuxer::LzSeq),
        super::Pipeline::Bw | super::Pipeline::Bbw => None,
        super::Pipeline::SortLz => None,
    }
}

/// Get the TokenEncoder for a given demuxer variant.
fn encoder_for_demuxer(demuxer: &LzDemuxer) -> Box<dyn lz_token::TokenEncoder> {
    match demuxer {
        LzDemuxer::Lz77 => Box::new(lz_token::Lz77Encoder),
        LzDemuxer::Lzss => Box::new(lz_token::LzssEncoder),
        LzDemuxer::LzSeq => Box::new(lz_token::LzSeqEncoder::default()),
    }
}

/// Demux pre-computed LZ77 matches into encoder streams for the GPU coordinator.
///
/// Used by the batched LZ77 GPU path (`find_matches_batched`) which returns
/// `Vec<lz77::Match>`. Converts Match → LzToken → encoder.encode().
#[cfg(feature = "webgpu")]
pub(crate) fn demux_lz77_matches(
    input: &[u8],
    matches: Vec<lz77::Match>,
    pipeline: super::Pipeline,
) -> PzResult<DemuxOutput> {
    let tokens = lz_token::matches_to_tokens(&matches);
    demux_tokens(input, &tokens, pipeline)
}

/// Demux a universal token stream into encoder streams.
///
/// Used by GPU SortLZ coordinator and LzSeq SortLZ CPU path, which produce
/// `Vec<LzToken>` directly via `sortlz::parse_matches()`.
#[cfg(feature = "webgpu")]
pub(crate) fn demux_tokens(
    input: &[u8],
    tokens: &[lz_token::LzToken],
    pipeline: super::Pipeline,
) -> PzResult<DemuxOutput> {
    let demuxer = demuxer_for_pipeline(pipeline).ok_or(PzError::InvalidInput)?;
    let encoder = encoder_for_demuxer(&demuxer);
    Ok(encoder.encode(input, tokens)?.into())
}

impl StreamDemuxer for LzDemuxer {
    fn stream_count(&self) -> usize {
        match self {
            LzDemuxer::Lz77 => 3,
            LzDemuxer::Lzss => 4,
            LzDemuxer::LzSeq => 6,
        }
    }

    fn compress_and_demux(&self, input: &[u8], options: &CompressOptions) -> PzResult<DemuxOutput> {
        match self {
            LzDemuxer::Lz77 => {
                // Token-based path: tokenize → wire-encode.
                let tokens = super::tokenize(input, options)?;
                let encoder = encoder_for_demuxer(self);
                Ok(encoder.encode(input, &tokens)?.into())
            }
            LzDemuxer::Lzss => {
                let encoded = lzss::encode(input)?;
                if encoded.len() < 12 {
                    return Err(PzError::InvalidInput);
                }
                let num_tokens = u32::from_le_bytes(encoded[4..8].try_into().unwrap());
                let flag_bytes_len =
                    u32::from_le_bytes(encoded[8..12].try_into().unwrap()) as usize;

                let flags_data = &encoded[12..12 + flag_bytes_len];
                let token_data = &encoded[12 + flag_bytes_len..];

                let flags_stream = flags_data.to_vec();
                let mut literals = Vec::new();
                let mut offsets = Vec::new();
                let mut lengths = Vec::new();
                let mut td_pos = 0;

                let required_flag_bytes = (num_tokens as usize).div_ceil(8);
                if required_flag_bytes > flag_bytes_len {
                    return Err(PzError::InvalidInput);
                }

                for i in 0..num_tokens as usize {
                    let is_literal = flags_data[i / 8] & (1 << (7 - (i % 8))) != 0;
                    if is_literal {
                        if td_pos >= token_data.len() {
                            return Err(PzError::InvalidInput);
                        }
                        literals.push(token_data[td_pos]);
                        td_pos += 1;
                    } else {
                        if td_pos + 4 > token_data.len() {
                            return Err(PzError::InvalidInput);
                        }
                        offsets.extend_from_slice(&token_data[td_pos..td_pos + 2]);
                        lengths.extend_from_slice(&token_data[td_pos + 2..td_pos + 4]);
                        td_pos += 4;
                    }
                }

                Ok(DemuxOutput {
                    streams: vec![flags_stream, literals, offsets, lengths],
                    pre_entropy_len: encoded.len(),
                    meta: num_tokens.to_le_bytes().to_vec(),
                })
            }
            LzDemuxer::LzSeq => {
                // SortLz and Optimal strategies route through the shared
                // tokenize() entry point, which handles GPU/CPU dispatch,
                // SortLz match finding, and optimal parsing uniformly.
                let use_tokenize = options.match_finder == super::MatchFinder::SortLz
                    || options.parse_strategy == ParseStrategy::Optimal;

                if use_tokenize {
                    let tokens = super::tokenize(input, options)?;
                    let defaults = lzseq::SeqConfig::default();
                    let enc = lzseq::encode_from_tokens(
                        &tokens,
                        &lzseq::SeqConfig {
                            max_window: options.seq_window_size.unwrap_or(defaults.max_window),
                            max_match_len: options.max_match_len.unwrap_or(defaults.max_match_len),
                            ..defaults
                        },
                    )?;
                    return Ok(seq_encoded_to_demux(enc));
                }

                // GPU path: fused match finding + demux on-device
                #[cfg(feature = "webgpu")]
                if let Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        if input.len() >= crate::webgpu::MIN_GPU_INPUT_SIZE
                            && input.len() <= engine.max_dispatch_input_size()
                        {
                            let enc = engine.lzseq_encode_gpu(input)?;
                            return Ok(seq_encoded_to_demux(enc));
                        }
                    }
                }

                // Default CPU path: encode_with_config (tuned lazy matching
                // with repeat-offset awareness, adaptive chain depth, hash4).
                let defaults = lzseq::SeqConfig::default();
                let config = lzseq::SeqConfig {
                    max_window: options.seq_window_size.unwrap_or(defaults.max_window),
                    max_match_len: options.max_match_len.unwrap_or(defaults.max_match_len),
                    ..defaults
                };
                let enc = lzseq::encode_with_config(input, &config)?;
                Ok(seq_encoded_to_demux(enc))
            }
        }
    }

    fn remux_and_decompress(
        &self,
        streams: Vec<Vec<u8>>,
        meta: &[u8],
        original_len: usize,
    ) -> PzResult<Vec<u8>> {
        match self {
            LzDemuxer::Lz77 => {
                if streams.len() != 3 {
                    return Err(PzError::InvalidInput);
                }
                let offsets = &streams[0];
                let lengths = &streams[1];
                let literals = &streams[2];

                if offsets.len() != lengths.len() || offsets.len() != literals.len() * 2 {
                    return Err(PzError::InvalidInput);
                }
                let num_matches = literals.len();
                let mut output = vec![0u8; original_len];
                let mut out_pos = 0usize;

                for i in 0..num_matches {
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
            LzDemuxer::Lzss => {
                if streams.len() != 4 {
                    return Err(PzError::InvalidInput);
                }
                if meta.len() < 4 {
                    return Err(PzError::InvalidInput);
                }
                let num_tokens = u32::from_le_bytes(meta[..4].try_into().unwrap());

                let flags_stream = &streams[0];
                let literals = &streams[1];
                let offsets = &streams[2];
                let lengths = &streams[3];
                let flag_bytes_len = flags_stream.len();

                let required_flag_bytes = (num_tokens as usize).div_ceil(8);
                if required_flag_bytes > flag_bytes_len {
                    return Err(PzError::InvalidInput);
                }

                let mut token_data = Vec::new();
                let mut lit_pos = 0;
                let mut match_idx = 0;
                for i in 0..num_tokens as usize {
                    let is_literal = flags_stream[i / 8] & (1 << (7 - (i % 8))) != 0;
                    if is_literal {
                        if lit_pos >= literals.len() {
                            return Err(PzError::InvalidInput);
                        }
                        token_data.push(literals[lit_pos]);
                        lit_pos += 1;
                    } else {
                        let off_pos = match_idx * 2;
                        if off_pos + 2 > offsets.len() || off_pos + 2 > lengths.len() {
                            return Err(PzError::InvalidInput);
                        }
                        token_data.extend_from_slice(&offsets[off_pos..off_pos + 2]);
                        token_data.extend_from_slice(&lengths[off_pos..off_pos + 2]);
                        match_idx += 1;
                    }
                }

                let mut lzss_blob = Vec::with_capacity(12 + flag_bytes_len + token_data.len());
                lzss_blob.extend_from_slice(&(original_len as u32).to_le_bytes());
                lzss_blob.extend_from_slice(&num_tokens.to_le_bytes());
                lzss_blob.extend_from_slice(&(flag_bytes_len as u32).to_le_bytes());
                lzss_blob.extend_from_slice(flags_stream);
                lzss_blob.extend_from_slice(&token_data);

                let decoded = lzss::decode(&lzss_blob)?;
                if decoded.len() != original_len {
                    return Err(PzError::InvalidInput);
                }
                Ok(decoded)
            }
            LzDemuxer::LzSeq => {
                if streams.len() != 6 {
                    return Err(PzError::InvalidInput);
                }
                if meta.len() < 8 {
                    return Err(PzError::InvalidInput);
                }
                let num_tokens = u32::from_le_bytes(meta[..4].try_into().unwrap());
                let num_matches = u32::from_le_bytes(meta[4..8].try_into().unwrap());

                let decoded = lzseq::decode(
                    &streams[0], // flags
                    &streams[1], // literals
                    &streams[2], // offset_codes
                    &streams[3], // offset_extra
                    &streams[4], // length_codes
                    &streams[5], // length_extra
                    num_tokens,
                    num_matches,
                    original_len,
                )?;
                Ok(decoded)
            }
        }
    }
}

/// Convert a SeqEncoded into a DemuxOutput.
fn seq_encoded_to_demux(enc: lzseq::SeqEncoded) -> DemuxOutput {
    let pre_entropy_len = enc.flags.len()
        + enc.literals.len()
        + enc.offset_codes.len()
        + enc.offset_extra.len()
        + enc.length_codes.len()
        + enc.length_extra.len();
    let mut meta = Vec::with_capacity(8);
    meta.extend_from_slice(&enc.num_tokens.to_le_bytes());
    meta.extend_from_slice(&enc.num_matches.to_le_bytes());
    DemuxOutput {
        streams: vec![
            enc.flags,
            enc.literals,
            enc.offset_codes,
            enc.offset_extra,
            enc.length_codes,
            enc.length_extra,
        ],
        pre_entropy_len,
        meta,
    }
}
