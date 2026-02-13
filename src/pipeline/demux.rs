//! Stream demuxer: splits pre-entropy stage output into independent byte streams
//! for entropy coding, and re-merges them on decompression.

use crate::lz77;
use crate::lz78;
use crate::lzss;
use crate::{PzError, PzResult};

use super::CompressOptions;

/// Output from a demuxer's compress-and-split operation.
pub(crate) struct DemuxOutput {
    /// Independent byte streams for entropy coding.
    pub streams: Vec<Vec<u8>>,
    /// Length of the pre-entropy data (e.g., total LZ output before splitting).
    pub pre_entropy_len: usize,
    /// Opaque metadata that must round-trip through the entropy container.
    pub meta: Vec<u8>,
}

/// Describes how a pre-entropy stage (LZ77, LZSS, LZ78, etc.) splits its
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
    Lz77,
    /// LZSS: 4 streams (flags, literals, offsets, lengths).
    Lzss,
    /// LZ78: 1 stream (flat blob, no splitting).
    Lz78,
}

/// Map a pipeline to its demuxer, if it uses one.
/// Returns `None` for BWT-based pipelines (Bw, Bbw).
pub(crate) fn demuxer_for_pipeline(pipeline: super::Pipeline) -> Option<LzDemuxer> {
    match pipeline {
        super::Pipeline::Deflate | super::Pipeline::Lzr | super::Pipeline::Lzf => {
            Some(LzDemuxer::Lz77)
        }
        super::Pipeline::LzssR => Some(LzDemuxer::Lzss),
        super::Pipeline::Lz78R => Some(LzDemuxer::Lz78),
        super::Pipeline::Bw | super::Pipeline::Bbw => None,
    }
}

impl StreamDemuxer for LzDemuxer {
    fn stream_count(&self) -> usize {
        match self {
            LzDemuxer::Lz77 => 3,
            LzDemuxer::Lzss => 4,
            LzDemuxer::Lz78 => 1,
        }
    }

    fn compress_and_demux(&self, input: &[u8], options: &CompressOptions) -> PzResult<DemuxOutput> {
        match self {
            LzDemuxer::Lz77 => {
                let lz_data = super::lz77_compress_with_backend(input, options)?;
                let lz_len = lz_data.len();
                let match_size = lz77::Match::SERIALIZED_SIZE; // 5
                let num_matches = lz_len / match_size;

                let mut offsets = Vec::with_capacity(num_matches * 2);
                let mut lengths = Vec::with_capacity(num_matches * 2);
                let mut literals = Vec::with_capacity(num_matches);

                for i in 0..num_matches {
                    let base = i * match_size;
                    offsets.push(lz_data[base]);
                    offsets.push(lz_data[base + 1]);
                    lengths.push(lz_data[base + 2]);
                    lengths.push(lz_data[base + 3]);
                    literals.push(lz_data[base + 4]);
                }

                Ok(DemuxOutput {
                    streams: vec![offsets, lengths, literals],
                    pre_entropy_len: lz_len,
                    meta: Vec::new(),
                })
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

                for i in 0..num_tokens as usize {
                    let is_literal = flags_data[i / 8] & (1 << (7 - (i % 8))) != 0;
                    if is_literal {
                        literals.push(token_data[td_pos]);
                        td_pos += 1;
                    } else {
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
            LzDemuxer::Lz78 => {
                let encoded = lz78::encode(input)?;
                let pre_entropy_len = encoded.len();
                Ok(DemuxOutput {
                    streams: vec![encoded],
                    pre_entropy_len,
                    meta: Vec::new(),
                })
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
                let match_size = lz77::Match::SERIALIZED_SIZE;
                let mut lz_data = Vec::with_capacity(num_matches * match_size);

                for i in 0..num_matches {
                    lz_data.push(offsets[i * 2]);
                    lz_data.push(offsets[i * 2 + 1]);
                    lz_data.push(lengths[i * 2]);
                    lz_data.push(lengths[i * 2 + 1]);
                    lz_data.push(literals[i]);
                }

                let mut output = vec![0u8; original_len];
                let out_len = lz77::decompress_to_buf(&lz_data, &mut output)?;
                if out_len != original_len {
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
            LzDemuxer::Lz78 => {
                if streams.len() != 1 {
                    return Err(PzError::InvalidInput);
                }
                let decoded = lz78::decode(&streams[0])?;
                if decoded.len() != original_len {
                    return Err(PzError::InvalidInput);
                }
                Ok(decoded)
            }
        }
    }
}
