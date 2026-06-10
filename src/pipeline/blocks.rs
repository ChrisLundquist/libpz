//! Per-pipeline single-block compress and decompress implementations.
//!
//! LZ-based pipelines (Lzf, Lzfi, LzssR) use a unified path:
//!   compress:   demux → entropy_encode
//!   decompress: entropy_decode → demux
//!
//! BWT-based pipelines (Bw, Bbw) have their own structure and are handled separately.

use crate::bwt;
use crate::fse;
use crate::mtf;
use crate::rle;
use crate::zrle;
use crate::{PzError, PzResult};

use super::demux::{demuxer_for_pipeline, LzDemuxer};
use super::stages::*;
#[cfg(feature = "webgpu")]
use super::Backend;
use super::{resolve_max_match_len, CompressOptions, DecompressOptions, Pipeline};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Compress a single block using the appropriate pipeline (no container header).
pub(crate) fn compress_block(
    input: &[u8],
    pipeline: Pipeline,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    // Resolve max match length for this pipeline.
    // Clone options only when we need to override the default.
    let resolved;
    let opts = if options.max_match_len.is_none() && demuxer_for_pipeline(pipeline).is_some() {
        resolved = CompressOptions {
            max_match_len: Some(resolve_max_match_len(pipeline, options)),
            ..options.clone()
        };
        &resolved
    } else {
        options
    };

    match demuxer_for_pipeline(pipeline) {
        Some(demuxer) => compress_block_lz(input, pipeline, &demuxer, opts),
        None => match pipeline {
            Pipeline::Bw => compress_block_bw(input, opts),
            Pipeline::Bbw => compress_block_bbw(input, opts),
            Pipeline::SortLz => compress_block_sortlz(input, opts),
            Pipeline::Num => compress_block_num(input, opts),
            Pipeline::Pz2 => compress_block_pz2(input, opts),
            Pipeline::Pz2d => compress_block_pz2d(input, opts),
            _ => Err(PzError::Unsupported),
        },
    }
}

/// Compress a single block from pre-computed demux output (entropy-encode only).
///
/// Used by the GPU streaming coordinator: GPU match-finding produces matches,
/// CPU `demux_lz77_matches` converts to streams + meta, and this function
/// runs only the entropy stage. Skips match-finding entirely.
#[cfg(feature = "webgpu")]
pub(crate) fn compress_block_from_demux(
    pipeline: Pipeline,
    original_len: usize,
    streams: Vec<Vec<u8>>,
    pre_entropy_len: usize,
    demux_meta: Vec<u8>,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len,
        data: Vec::new(),
        streams: Some(streams),
        metadata: StageMetadata {
            pre_entropy_len: Some(pre_entropy_len),
            demux_meta,
            ..StageMetadata::default()
        },
    };
    let block = entropy_encode(block, pipeline, original_len, options)?;
    Ok(block.data)
}

/// Decompress a single block using the appropriate pipeline (no container header).
pub(crate) fn decompress_block(
    payload: &[u8],
    pipeline: Pipeline,
    orig_len: usize,
    options: &DecompressOptions,
) -> PzResult<Vec<u8>> {
    match demuxer_for_pipeline(pipeline) {
        Some(demuxer) => decompress_block_lz(payload, pipeline, &demuxer, orig_len, options),
        None => match pipeline {
            Pipeline::Bw => decompress_block_bw(payload, orig_len),
            Pipeline::Bbw => decompress_block_bbw(payload, orig_len),
            Pipeline::SortLz => decompress_block_sortlz(payload, orig_len),
            Pipeline::Num => decompress_block_num(payload, orig_len),
            Pipeline::Pz2 => decompress_block_pz2(payload, orig_len),
            Pipeline::Pz2d => decompress_block_pz2d(payload, orig_len),
            _ => Err(PzError::Unsupported),
        },
    }
}

// ---------------------------------------------------------------------------
// Unified LZ-based pipeline path
// ---------------------------------------------------------------------------

/// Compress a single block for any LZ-based pipeline.
///
/// All LZ pipelines share the same structure:
///   input → stage_demux_compress(demuxer) → entropy_encode(pipeline) → output
fn compress_block_lz(
    input: &[u8],
    pipeline: Pipeline,
    demuxer: &LzDemuxer,
    options: &CompressOptions,
) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_demux_compress(block, demuxer, options)?;
    let block = entropy_encode(block, pipeline, input.len(), options)?;
    Ok(block.data)
}

/// Decompress a single block for any LZ-based pipeline.
///
/// All LZ pipelines share the same structure:
///   payload → entropy_decode(pipeline) → stage_demux_decompress(demuxer) → output
fn decompress_block_lz(
    payload: &[u8],
    pipeline: Pipeline,
    demuxer: &LzDemuxer,
    orig_len: usize,
    options: &DecompressOptions,
) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: orig_len,
        data: payload.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = entropy_decode(block, pipeline, options)?;
    let block = stage_demux_decompress(block, demuxer)?;
    Ok(block.data)
}

// ---------------------------------------------------------------------------
// Entropy encode/decode dispatch
// ---------------------------------------------------------------------------

/// Dispatch to the correct entropy encoder for a pipeline.
fn entropy_encode(
    block: StageBlock,
    pipeline: Pipeline,
    input_len: usize,
    options: &CompressOptions,
) -> PzResult<StageBlock> {
    match pipeline {
        Pipeline::LzssR | Pipeline::LzSeqR => {
            let _ = (input_len, options);
            stage_rans_encode_with_options(block, options)
        }
        Pipeline::LzSeqH => {
            let _ = (input_len, options);
            stage_huffman_encode(block)
        }
        Pipeline::Lzf => {
            let _ = (input_len, options);
            stage_fse_encode(block)
        }
        Pipeline::Lzfi => {
            #[cfg(feature = "webgpu")]
            {
                if let Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        return stage_fse_interleaved_encode_webgpu(block, engine);
                    }
                }
            }
            let _ = (input_len, options);
            stage_fse_interleaved_encode(block)
        }
        Pipeline::LzSeq2R => {
            let _ = (input_len, options);
            stage_rans_encode_sparse(block, options)
        }
        _ => Err(PzError::Unsupported),
    }
}

/// Dispatch to the correct entropy decoder for a pipeline.
///
/// For interleaved FSE (Lzfi), GPU variants are used when a GPU backend is active.
fn entropy_decode(
    block: StageBlock,
    pipeline: Pipeline,
    options: &DecompressOptions,
) -> PzResult<StageBlock> {
    match pipeline {
        Pipeline::LzssR | Pipeline::LzSeqR => {
            #[cfg(feature = "webgpu")]
            {
                if let Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        return stage_rans_decode_webgpu(block, engine);
                    }
                }
            }
            let _ = options;
            stage_rans_decode(block)
        }
        Pipeline::LzSeqH => stage_huffman_decode(block),
        Pipeline::Lzf => stage_fse_decode(block),
        Pipeline::Lzfi => {
            #[cfg(feature = "webgpu")]
            {
                if let Backend::WebGpu = options.backend {
                    if let Some(ref engine) = options.webgpu_engine {
                        return stage_fse_interleaved_decode_webgpu(block, engine);
                    }
                }
            }
            let _ = options;
            stage_fse_interleaved_decode(block)
        }
        Pipeline::LzSeq2R => {
            let _ = options;
            stage_rans_decode_sparse(block)
        }
        _ => Err(PzError::Unsupported),
    }
}

// ---------------------------------------------------------------------------
// BW pipeline: BWT + MTF + RLE + FSE
// ---------------------------------------------------------------------------

/// Compress a single block using the BW pipeline (no container header).
fn compress_block_bw(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_bwt_encode(block, options)?;
    let block = stage_mtf_encode(block)?;
    let block = stage_rle_encode(block)?;
    let block = stage_fse_encode_bw(block)?;
    Ok(block.data)
}

/// Decompress a single BW block (no container header).
fn decompress_block_bw(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 8 {
        return Err(PzError::InvalidInput);
    }

    let primary_field = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]);
    let primary_index = primary_field & !BW_CURSORS_FLAG;
    let len_field = u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]);
    let zrle_used = (len_field & BW_ZRLE_FLAG) != 0;
    let rle_len = (len_field & !BW_ZRLE_FLAG) as usize;

    // Multi-cursor iBWT start samples (versioned: legacy blocks lack the flag
    // and decode through the serial single-cursor chase).
    let mut offset = 8usize;
    let cursor_samples = if (primary_field & BW_CURSORS_FLAG) != 0 {
        let end = offset + 4 * bwt::IBWT_WIRE_SAMPLES;
        if payload.len() < end {
            return Err(PzError::InvalidInput);
        }
        let samples: Vec<u32> = payload[offset..end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        offset = end;
        Some(samples)
    } else {
        None
    };

    let entropy_data = &payload[offset..];

    // Stage 1: FSE decoder
    let rle_data = fse::decode(entropy_data, rle_len)?;

    // Stage 2: zero-run decode (RUNA/RUNB or legacy RLE). The MTF/BWT stream
    // length equals the original block length.
    let mtf_data = if zrle_used {
        zrle::decode(&rle_data, orig_len)?
    } else {
        rle::decode(&rle_data)?
    };

    // Stage 3: Inverse MTF
    let bwt_data = mtf::decode(&mtf_data);

    // Stage 4: Inverse BWT (multi-cursor when the block carries samples).
    let output = match &cursor_samples {
        Some(samples) => bwt::decode_with_samples(&bwt_data, primary_index, samples)?,
        None => bwt::decode(&bwt_data, primary_index)?,
    };

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// BBW pipeline: Bijective BWT + MTF + RLE + FSE
// ---------------------------------------------------------------------------

/// Compress a single block using the Bbw pipeline (no container header).
fn compress_block_bbw(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let block = StageBlock {
        block_index: 0,
        original_len: input.len(),
        data: input.to_vec(),
        streams: None,
        metadata: StageMetadata::default(),
    };
    let block = stage_bbwt_encode(block, options)?;
    let block = stage_mtf_encode(block)?;
    let block = stage_rle_encode(block)?;
    let block = stage_fse_encode_bbw(block)?;
    Ok(block.data)
}

/// Decompress a single Bbw block (no container header).
fn decompress_block_bbw(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 8 {
        return Err(PzError::InvalidInput);
    }

    // Parse header: [num_factors: u32] [factor_lengths: u32 × k] [rle_len: u32]
    let num_factors = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let header_len = 4 + num_factors * 4 + 4;
    if payload.len() < header_len {
        return Err(PzError::InvalidInput);
    }

    let mut factor_lengths = Vec::with_capacity(num_factors);
    for i in 0..num_factors {
        let offset = 4 + i * 4;
        let fl = u32::from_le_bytes([
            payload[offset],
            payload[offset + 1],
            payload[offset + 2],
            payload[offset + 3],
        ]) as usize;
        factor_lengths.push(fl);
    }

    let rle_offset = 4 + num_factors * 4;
    let len_field = u32::from_le_bytes([
        payload[rle_offset],
        payload[rle_offset + 1],
        payload[rle_offset + 2],
        payload[rle_offset + 3],
    ]);
    let zrle_used = (len_field & BW_ZRLE_FLAG) != 0;
    let rle_len = (len_field & !BW_ZRLE_FLAG) as usize;

    let entropy_data = &payload[header_len..];

    // Stage 1: FSE decode
    let rle_data = fse::decode(entropy_data, rle_len)?;

    // Stage 2: zero-run decode (RUNA/RUNB or legacy RLE). The MTF/BWT stream
    // length equals the original block length.
    let mtf_data = if zrle_used {
        zrle::decode(&rle_data, orig_len)?
    } else {
        rle::decode(&rle_data)?
    };

    // Stage 3: Inverse MTF
    let bwt_data = mtf::decode(&mtf_data);

    // Stage 4: Inverse bijective BWT
    let output = bwt::decode_bijective(&bwt_data, &factor_lengths)?;

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// SortLZ pipeline: Sort-based LZ77 + FSE
// ---------------------------------------------------------------------------

/// Compress a single block using the SortLZ pipeline (no container header).
fn compress_block_sortlz(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    #[cfg(feature = "webgpu")]
    {
        if let Backend::WebGpu = options.backend {
            if let Some(ref engine) = options.webgpu_engine {
                return engine.sortlz_compress(input, &crate::sortlz::SortLzConfig::default());
            }
        }
    }
    let _ = options;
    crate::sortlz::compress(input, &crate::sortlz::SortLzConfig::default())
}

/// Decompress a single SortLZ block (no container header).
fn decompress_block_sortlz(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    crate::sortlz::decompress(payload, orig_len)
}

// ---------------------------------------------------------------------------
// Num pipeline: numeric decorrelation (byte-plane split + per-plane gated FSE)
// ---------------------------------------------------------------------------

/// Compress a single block using the Num pipeline (no container header).
///
/// The whole stride-sweep + per-plane gating + per-plane FSE + STORE fallback
/// lives in [`crate::numeric::encode`]; this is a thin adapter. `Num` is a pure
/// CPU transform pipeline with no GPU path and no LZ tokens.
fn compress_block_num(input: &[u8], _options: &CompressOptions) -> PzResult<Vec<u8>> {
    Ok(crate::numeric::encode(input))
}

/// Decompress a single Num block (no container header).
fn decompress_block_num(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    crate::numeric::decode(payload, orig_len)
}

// ---------------------------------------------------------------------------
// Pz2 pipeline: decode-first sequence codec (4-lane Huffman + fused splice)
// ---------------------------------------------------------------------------

/// Compress a single block using the Pz2 pipeline (no container header).
///
/// The whole wire format (sequence conversion, multi-lane Huffman literals,
/// sequence-code lanes, raw-literal fallback) lives in
/// [`crate::pz2::encode_with_config`]; this is a thin adapter like
/// `compress_block_num`, with the parse flags (`--greedy`, window size)
/// mapped exactly as the LzSeq demux path maps them.
fn compress_block_pz2(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let mut config = super::pz2_seq_config(options);
    if options.parse_strategy == super::ParseStrategy::Auto {
        config.greedy = super::pz2_auto_greedy(input);
    }
    crate::pz2::encode_with_config(input, &config)
}

/// Decompress a single Pz2 block (no container header).
fn decompress_block_pz2(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    crate::pz2::decode(payload, orig_len)
}

// ---------------------------------------------------------------------------
// Pz2d pipeline: Pz2 dict tier — one container block = one segment
// ---------------------------------------------------------------------------
//
// Segment payload layout (inner framing, all u32 LE):
//   [num_inner] then num_inner × [inner_orig_len][inner_comp_len],
//   then the inner pz2 wires concatenated in order.
// The inner structure (PZ2D_INNER_BLOCK blocks, PZ2D_DICT_SIZE dict) is a
// format constant; the decoder only needs the frame table.

/// Compress one segment with the Pz2d dict tier.
fn compress_block_pz2d(input: &[u8], options: &CompressOptions) -> PzResult<Vec<u8>> {
    let defaults = crate::lzseq::SeqConfig::default();
    // The window must span dict + inner block or dict reach is lost.
    let window = (super::PZ2D_DICT_SIZE + super::PZ2D_INNER_BLOCK)
        .next_power_of_two()
        .max(defaults.max_window);
    let greedy = match options.parse_strategy {
        super::ParseStrategy::Greedy => true,
        super::ParseStrategy::Lazy => false,
        // Same auto rule as Pz2, applied at segment granularity.
        _ => super::pz2_auto_greedy(input),
    };
    let config = crate::lzseq::SeqConfig {
        max_window: options.seq_window_size.unwrap_or(window),
        max_match_len: options.max_match_len.unwrap_or(defaults.max_match_len),
        greedy,
        ..defaults
    };
    let blocks = crate::pz2::encode_segment(
        input,
        super::PZ2D_INNER_BLOCK,
        super::PZ2D_DICT_SIZE,
        &config,
    )?;

    let mut out = Vec::with_capacity(input.len() / 2 + 64);
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for (orig, wire) in &blocks {
        out.extend_from_slice(&(*orig as u32).to_le_bytes());
        out.extend_from_slice(&(wire.len() as u32).to_le_bytes());
    }
    for (_, wire) in &blocks {
        out.extend_from_slice(wire);
    }
    Ok(out)
}

/// Decompress one Pz2d segment (no container header).
fn decompress_block_pz2d(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    let take_u32 = |p: &mut &[u8]| -> PzResult<u32> {
        if p.len() < 4 {
            return Err(PzError::InvalidInput);
        }
        let v = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
        *p = &p[4..];
        Ok(v)
    };
    let mut p = payload;
    let num = take_u32(&mut p)? as usize;
    // Sanity bound: every inner block covers ≥ 1 original byte.
    if num > orig_len.max(1) {
        return Err(PzError::InvalidInput);
    }
    let mut table = Vec::with_capacity(num);
    let mut total_orig = 0usize;
    for _ in 0..num {
        let o = take_u32(&mut p)? as usize;
        let c = take_u32(&mut p)? as usize;
        total_orig += o;
        table.push((o, c));
    }
    if total_orig != orig_len {
        return Err(PzError::InvalidInput);
    }
    let mut blocks: Vec<(usize, &[u8])> = Vec::with_capacity(num);
    for &(o, c) in &table {
        if p.len() < c {
            return Err(PzError::InvalidInput);
        }
        let (wire, rest) = p.split_at(c);
        blocks.push((o, wire));
        p = rest;
    }

    // 2-wave decode (clean-slate-codec.md §11): the dict region is a prefix
    // chain (block k needs blocks 0..k) and decodes sequentially; every
    // block after it depends only on the completed dict region, so those
    // fan out across scoped threads. The outer scheduler already
    // parallelizes across segments; the inner fan-out keeps a single
    // segment's wall close to dict-chain time instead of whole-segment
    // time.
    let dict_len = super::PZ2D_DICT_SIZE.min(orig_len);
    let mut out: Vec<u8> = Vec::with_capacity(orig_len);
    let mut wave2_start = blocks.len();
    for (i, &(o, wire)) in blocks.iter().enumerate() {
        if out.len() >= dict_len {
            wave2_start = i;
            break;
        }
        // Chain block's prefix is everything decoded so far: `out` is the
        // arena, so the chain appends in place with zero prefix copies.
        crate::pz2::decode_into_arena(&mut out, wire, o)?;
    }
    if out.len() > dict_len {
        // Inner frames must tile the dict region exactly.
        return Err(PzError::InvalidInput);
    }
    let wave2 = &blocks[wave2_start.min(blocks.len())..];
    if !wave2.is_empty() {
        // wave2 non-empty implies the chain filled the dict region exactly
        // (checked above), so out.len() == dict_len here.
        //
        // v1 spawned one thread per block and each decode re-zeroed an
        // 18 MiB buffer and re-copied the 16 MiB dict — ~3.3 GB of memory
        // traffic per segment that made decode memory-bound (§11b). Now a
        // bounded set of workers each seed ONE arena with the dict and
        // truncate-and-reuse it across their strided share of the blocks,
        // decoding into per-block slices of the final output.
        out.resize(orig_len, 0);
        let (dict, rest) = out.split_at_mut(dict_len);
        let dict: &[u8] = dict;
        let mut jobs: Vec<(&[u8], &mut [u8])> = Vec::with_capacity(wave2.len());
        let mut rem = rest;
        for &(o, wire) in wave2 {
            // total_orig == orig_len was validated, so the slices tile rest.
            let (dst, tail) = rem.split_at_mut(o);
            jobs.push((wire, dst));
            rem = tail;
        }
        // Narrow fan-out: the outer scheduler already decodes segments
        // concurrently, and each worker pays one 16 MiB dict copy — wide
        // fan-out here would put every block on its own worker and degrade
        // to v1's per-block dict traffic. A few workers per segment keep
        // the wave-2 tail short relative to the sequential dict chain
        // while the dict copy amortizes over each worker's block share.
        let nworkers = (std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            / 4)
        .clamp(1, jobs.len());
        let mut shares: Vec<Vec<(&[u8], &mut [u8])>> = (0..nworkers).map(|_| Vec::new()).collect();
        for (i, job) in jobs.into_iter().enumerate() {
            shares[i % nworkers].push(job);
        }
        let mut results: Vec<PzResult<()>> = Vec::new();
        std::thread::scope(|s| {
            let handles: Vec<_> = shares
                .into_iter()
                .map(|share| {
                    s.spawn(move || -> PzResult<()> {
                        // Reserve dict + largest block + wildcopy slack up
                        // front so decode_into_arena's resize never
                        // reallocates (which would re-copy the dict).
                        let max_block = share.iter().map(|(_, dst)| dst.len()).max().unwrap_or(0);
                        let mut arena: Vec<u8> = Vec::with_capacity(dict.len() + max_block + 64);
                        arena.extend_from_slice(dict);
                        for (wire, dst) in share {
                            arena.truncate(dict.len());
                            crate::pz2::decode_into_arena(&mut arena, wire, dst.len())?;
                            dst.copy_from_slice(&arena[dict.len()..]);
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in handles {
                results.push(h.join().unwrap_or(Err(PzError::InvalidInput)));
            }
        });
        for r in results {
            r?;
        }
    }
    if out.len() != orig_len {
        return Err(PzError::InvalidInput);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression: a ~1 MiB block of highly periodic data (`0^15 1`) produces
    /// more than 65535 single-char Lyndon factors. The per-block factor count was
    /// a u16 that silently truncated, producing an undecodable (corrupt) bbw block.
    /// The count is now u32. This drives the *block* path
    /// (`compress_block`/`decompress_block`) — the `encode_bijective`-only
    /// regression test in `bwt/tests.rs` cannot reach this framing-layer bug.
    #[test]
    fn test_bbw_block_over_65535_factors_roundtrip() {
        // ~1.05 MiB of "0^15 1" => ~65700 factors in a single block (> u16::MAX).
        let mut input = Vec::with_capacity(1_100_000);
        while input.len() < 1_100_000 {
            input.extend(std::iter::repeat_n(0u8, 15));
            input.push(1);
        }
        let copts = CompressOptions::default();
        let dopts = DecompressOptions::default();
        let payload = compress_block(&input, Pipeline::Bbw, &copts).unwrap();
        let out = decompress_block(&payload, Pipeline::Bbw, input.len(), &dopts).unwrap();
        assert!(
            out == input,
            "bbw block round-trip corrupted at >65535 factors (len {})",
            input.len()
        );
    }
}
