#!/usr/bin/env bash
# trace-pipeline.sh — Generate visual flow diagrams for compression pipelines
#
# Traces the call path from compress_block() through each stage, showing:
# - Function names and file locations (file:line)
# - Data transformations (StageBlock.data vs StageBlock.streams)
# - Stream counts and demuxer types

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
trace-pipeline.sh — Generate visual flow diagrams for compression pipelines

Traces the call path from compress_block() through each stage, showing
data transformations and file locations. Useful for understanding how
data flows through multi-stage pipelines.

USAGE:
    ./scripts/trace-pipeline.sh [OPTIONS]

OPTIONS:
    -p, --pipeline NAME     Pipeline to trace (default: deflate)
                            Options: deflate, lzr, lzf, lzfi, lzssr, lz78r, bw, bbw
    --format FORMAT         Output format: text (default) or mermaid
    -h, --help              Show this help

OUTPUT FORMATS:
    text                    Human-readable indented trace (default)
    mermaid                 Mermaid flowchart syntax (paste into mermaid.live)

EXAMPLES:
    ./scripts/trace-pipeline.sh                        # deflate pipeline (text)
    ./scripts/trace-pipeline.sh -p bw                  # BWT pipeline
    ./scripts/trace-pipeline.sh -p lzfi --format mermaid  # FSE interleaved (mermaid)

UNDERSTANDING THE OUTPUT:
    [data: N bytes]    - StageBlock.data contains N bytes
    [streams: K×...]   - StageBlock.streams contains K separate byte streams
    → stage_name()     - Function call with source location
EOF
}

PIPELINE="deflate"
FORMAT="text"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -p|--pipeline)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --pipeline requires an argument" >&2
                exit 1
            fi
            PIPELINE="$2"
            shift 2
            ;;
        --format)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --format requires an argument" >&2
                exit 1
            fi
            FORMAT="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown option '$1'" >&2
            echo "Run './scripts/trace-pipeline.sh --help' for usage." >&2
            exit 1
            ;;
    esac
done

# Validate pipeline
case "$PIPELINE" in
    deflate|lzr|lzf|lzfi|lzssr|lz78r|bw|bbw) ;;
    *)
        echo "ERROR: unknown pipeline '$PIPELINE'" >&2
        echo "Valid pipelines: deflate, lzr, lzf, lzfi, lzssr, lz78r, bw, bbw" >&2
        exit 1
        ;;
esac

# Validate format
case "$FORMAT" in
    text|mermaid) ;;
    *)
        echo "ERROR: unknown format '$FORMAT'" >&2
        echo "Valid formats: text, mermaid" >&2
        exit 1
        ;;
esac

# Pipeline metadata: stage count, demuxer type, entropy encoder
# These mappings are derived from:
# - demuxer_for_pipeline() in src/pipeline/demux.rs:52-61
# - entropy_encode()/entropy_decode() in src/pipeline/blocks.rs:124-194
# - compress_block_bw()/compress_block_bbw() in src/pipeline/blocks.rs:200-263

case "$PIPELINE" in
    deflate)
        DEMUXER="Lz77"
        STREAM_COUNT=3
        ENTROPY="Huffman"
        ;;
    lzr)
        DEMUXER="Lz77"
        STREAM_COUNT=3
        ENTROPY="rANS"
        ;;
    lzf)
        DEMUXER="Lz77"
        STREAM_COUNT=3
        ENTROPY="FSE"
        ;;
    lzfi)
        DEMUXER="Lzss"
        STREAM_COUNT=4
        ENTROPY="FSE-interleaved"
        ;;
    lzssr)
        DEMUXER="Lzss"
        STREAM_COUNT=4
        ENTROPY="rANS"
        ;;
    lz78r)
        DEMUXER="Lz78"
        STREAM_COUNT=1
        ENTROPY="rANS"
        ;;
    bw)
        DEMUXER="N/A"
        STREAM_COUNT=0
        ENTROPY="N/A"
        ;;
    bbw)
        DEMUXER="N/A"
        STREAM_COUNT=0
        ENTROPY="N/A"
        ;;
esac

# Output functions
emit_text() {
    echo "$1"
}

emit_mermaid() {
    echo "$1"
}

if [[ "$FORMAT" == "mermaid" ]]; then
    emit_mermaid "flowchart TD"
    emit_mermaid "    Start([\"Input: raw bytes\"]) --> CompressBlock"
fi

# Trace LZ-based pipelines (Deflate, Lzr, Lzf, Lzfi, LzssR, Lz78R)
trace_lz_pipeline() {
    local pipeline=$1
    local demuxer=$2
    local stream_count=$3
    local entropy=$4

    if [[ "$FORMAT" == "text" ]]; then
        emit_text "=== Pipeline Trace: $pipeline ==="
        emit_text ""
        emit_text "Entry point: compress_block()"
        emit_text "  src/pipeline/blocks.rs:26-52"
        emit_text "  [data: input.len() bytes, streams: None]"
        emit_text ""
        emit_text "↓ Unified LZ path: compress_block_lz()"
        emit_text "  src/pipeline/blocks.rs:75-95"
        emit_text "  Creates StageBlock with input data"
        emit_text ""
        emit_text "↓ Stage 1: stage_demux_compress()"
        emit_text "  src/pipeline/stages.rs:60-71"
        emit_text "  Calls demuxer.compress_and_demux()"
        emit_text "  Demuxer: $demuxer"
        emit_text "  src/pipeline/demux.rs:72-156"
        emit_text "  [data: cleared, streams: $stream_count streams]"

        case "$demuxer" in
            Lz77)
                emit_text ""
                emit_text "  LZ77 stream layout (3 streams):"
                emit_text "    [0]: offsets   (u16 × num_matches, LE)"
                emit_text "    [1]: lengths   (u16 × num_matches, LE)"
                emit_text "    [2]: literals  (u8 × num_matches)"
                emit_text "  See src/pipeline/demux.rs:74-97"
                ;;
            Lzss)
                emit_text ""
                emit_text "  LZSS stream layout (4 streams):"
                emit_text "    [0]: flags     (1 bit per token, packed into bytes)"
                emit_text "    [1]: literals  (u8 × num_literal_tokens)"
                emit_text "    [2]: offsets   (u16 × num_match_tokens, LE)"
                emit_text "    [3]: lengths   (u16 × num_match_tokens, LE)"
                emit_text "  metadata: num_tokens (u32 LE)"
                emit_text "  See src/pipeline/demux.rs:99-144"
                ;;
            Lz78)
                emit_text ""
                emit_text "  LZ78 stream layout (1 stream):"
                emit_text "    [0]: flat LZ78 encoded blob (no splitting)"
                emit_text "  See src/pipeline/demux.rs:146-155"
                ;;
        esac

        emit_text ""
        emit_text "↓ Stage 2: entropy_encode()"
        emit_text "  src/pipeline/blocks.rs:124-165"
        emit_text "  Dispatch to $entropy encoder"

        case "$entropy" in
            Huffman)
                emit_text "  → stage_huffman_encode()"
                emit_text "    src/pipeline/stages.rs:172-199"
                emit_text "    Encodes each stream independently"
                emit_text "    Container: multistream header + per-stream Huffman"
                ;;
            rANS)
                emit_text "  → stage_rans_encode()"
                emit_text "    src/pipeline/stages.rs:398-426"
                emit_text "    Encodes each stream independently"
                emit_text "    Container: multistream header + per-stream rANS"
                ;;
            FSE)
                emit_text "  → stage_fse_encode()"
                emit_text "    src/pipeline/stages.rs:441-469"
                emit_text "    Encodes each stream independently"
                emit_text "    Container: multistream header + per-stream FSE"
                ;;
            FSE-interleaved)
                emit_text "  → stage_fse_interleaved_encode()"
                emit_text "    src/pipeline/stages.rs:484-533"
                emit_text "    Interleaves 4 LZSS streams into FSE table"
                emit_text "    (GPU variant available: stage_fse_interleaved_encode_webgpu)"
                ;;
        esac

        emit_text "  [data: compressed bytes, streams: None]"
        emit_text ""
        emit_text "Return: compressed block data"
        emit_text ""
        emit_text "Key file locations:"
        emit_text "  Entry:         src/pipeline/blocks.rs:26 (compress_block)"
        emit_text "  LZ dispatch:   src/pipeline/blocks.rs:75 (compress_block_lz)"
        emit_text "  Demux stage:   src/pipeline/stages.rs:60 (stage_demux_compress)"
        emit_text "  Demuxer impl:  src/pipeline/demux.rs:63 (LzDemuxer trait)"
        emit_text "  Entropy stage: src/pipeline/blocks.rs:124 (entropy_encode)"
        emit_text "  Multistream:   src/pipeline/stages.rs:110 (encode_multistream)"

    else
        # Mermaid format
        emit_mermaid "    CompressBlock[\"compress_block()<br/>blocks.rs:26\"] --> CompressBlockLz"
        emit_mermaid "    CompressBlockLz[\"compress_block_lz()<br/>blocks.rs:75<br/>[data: input, streams: None]\"] --> DemuxCompress"
        emit_mermaid "    DemuxCompress[\"stage_demux_compress()<br/>stages.rs:60<br/>demuxer: $demuxer\"] --> DemuxImpl"
        emit_mermaid "    DemuxImpl[\"demuxer.compress_and_demux()<br/>demux.rs:72<br/>[data: cleared, streams: $stream_count]\"] --> EntropyEncode"
        emit_mermaid "    EntropyEncode[\"entropy_encode()<br/>blocks.rs:124<br/>encoder: $entropy\"] --> EntropyStage"

        case "$entropy" in
            Huffman)
                emit_mermaid "    EntropyStage[\"stage_huffman_encode()<br/>stages.rs:172\"] --> Output"
                ;;
            rANS)
                emit_mermaid "    EntropyStage[\"stage_rans_encode()<br/>stages.rs:398\"] --> Output"
                ;;
            FSE)
                emit_mermaid "    EntropyStage[\"stage_fse_encode()<br/>stages.rs:441\"] --> Output"
                ;;
            FSE-interleaved)
                emit_mermaid "    EntropyStage[\"stage_fse_interleaved_encode()<br/>stages.rs:484\"] --> Output"
                ;;
        esac

        emit_mermaid "    Output([\"Compressed block data<br/>[data: bytes, streams: None]\"])"
    fi
}

# Trace BWT-based pipelines (Bw, Bbw)
trace_bwt_pipeline() {
    local pipeline=$1

    if [[ "$FORMAT" == "text" ]]; then
        emit_text "=== Pipeline Trace: $pipeline ==="
        emit_text ""
        emit_text "Entry point: compress_block()"
        emit_text "  src/pipeline/blocks.rs:26-52"
        emit_text "  [data: input.len() bytes, streams: None]"
        emit_text ""

        if [[ "$pipeline" == "bw" ]]; then
            emit_text "↓ BW pipeline: compress_block_bw()"
            emit_text "  src/pipeline/blocks.rs:200-213"
            emit_text ""
            emit_text "↓ Stage 1: stage_bwt_encode()"
            emit_text "  src/pipeline/stages.rs:639-656"
            emit_text "  Calls bwt::encode() → BwtResult { data, primary_index }"
            emit_text "  metadata.bwt_primary_index = Some(primary_index)"
            emit_text "  [data: BWT-transformed bytes, streams: None]"
        else
            emit_text "↓ BBW pipeline: compress_block_bbw()"
            emit_text "  src/pipeline/blocks.rs:250-263"
            emit_text ""
            emit_text "↓ Stage 1: stage_bbwt_encode()"
            emit_text "  src/pipeline/stages.rs:674-691"
            emit_text "  Calls bwt::encode_bijective() → (data, factor_lengths)"
            emit_text "  metadata.bbwt_factor_lengths = Some(factor_lengths)"
            emit_text "  [data: Bijective BWT output, streams: None]"
        fi

        emit_text ""
        emit_text "↓ Stage 2: stage_mtf_encode()"
        emit_text "  src/pipeline/stages.rs:703-709"
        emit_text "  Calls mtf::encode()"
        emit_text "  [data: MTF-transformed bytes, streams: None]"
        emit_text ""
        emit_text "↓ Stage 3: stage_rle_encode()"
        emit_text "  src/pipeline/stages.rs:721-729"
        emit_text "  Calls rle::encode()"
        emit_text "  metadata.pre_entropy_len = Some(rle_output.len())"
        emit_text "  [data: RLE-compressed bytes, streams: None]"
        emit_text ""

        if [[ "$pipeline" == "bw" ]]; then
            emit_text "↓ Stage 4: stage_fse_encode_bw()"
            emit_text "  src/pipeline/stages.rs:741-760"
            emit_text "  Calls fse::encode()"
            emit_text "  Prepends header: [bwt_primary_index: u32][rle_len: u32]"
        else
            emit_text "↓ Stage 4: stage_fse_encode_bbw()"
            emit_text "  src/pipeline/stages.rs:775-800"
            emit_text "  Calls fse::encode()"
            emit_text "  Prepends header: [num_factors: u16][factor_lengths: u32×k][rle_len: u32]"
        fi

        emit_text "  [data: final compressed bytes, streams: None]"
        emit_text ""
        emit_text "Return: compressed block data"
        emit_text ""
        emit_text "Key file locations:"
        emit_text "  Entry:       src/pipeline/blocks.rs:26 (compress_block)"

        if [[ "$pipeline" == "bw" ]]; then
            emit_text "  BW dispatch: src/pipeline/blocks.rs:200 (compress_block_bw)"
            emit_text "  BWT stage:   src/pipeline/stages.rs:639 (stage_bwt_encode)"
        else
            emit_text "  BBW dispatch: src/pipeline/blocks.rs:250 (compress_block_bbw)"
            emit_text "  BBWT stage:   src/pipeline/stages.rs:674 (stage_bbwt_encode)"
        fi

        emit_text "  MTF stage:   src/pipeline/stages.rs:703 (stage_mtf_encode)"
        emit_text "  RLE stage:   src/pipeline/stages.rs:721 (stage_rle_encode)"

        if [[ "$pipeline" == "bw" ]]; then
            emit_text "  FSE stage:   src/pipeline/stages.rs:741 (stage_fse_encode_bw)"
        else
            emit_text "  FSE stage:   src/pipeline/stages.rs:775 (stage_fse_encode_bbw)"
        fi

    else
        # Mermaid format
        emit_mermaid "    CompressBlock[\"compress_block()<br/>blocks.rs:26\"] --> BwtPipeline"

        if [[ "$pipeline" == "bw" ]]; then
            emit_mermaid "    BwtPipeline[\"compress_block_bw()<br/>blocks.rs:200\"] --> BwtStage"
            emit_mermaid "    BwtStage[\"stage_bwt_encode()<br/>stages.rs:639<br/>bwt::encode()\"] --> MtfStage"
        else
            emit_mermaid "    BwtPipeline[\"compress_block_bbw()<br/>blocks.rs:250\"] --> BbwtStage"
            emit_mermaid "    BbwtStage[\"stage_bbwt_encode()<br/>stages.rs:674<br/>bwt::encode_bijective()\"] --> MtfStage"
        fi

        emit_mermaid "    MtfStage[\"stage_mtf_encode()<br/>stages.rs:703<br/>mtf::encode()\"] --> RleStage"
        emit_mermaid "    RleStage[\"stage_rle_encode()<br/>stages.rs:721<br/>rle::encode()\"] --> FseStage"

        if [[ "$pipeline" == "bw" ]]; then
            emit_mermaid "    FseStage[\"stage_fse_encode_bw()<br/>stages.rs:741<br/>fse::encode()\"] --> Output"
        else
            emit_mermaid "    FseStage[\"stage_fse_encode_bbw()<br/>stages.rs:775<br/>fse::encode()\"] --> Output"
        fi

        emit_mermaid "    Output([\"Compressed block data<br/>[data: bytes, streams: None]\"])"
    fi
}

# Main trace dispatch
case "$PIPELINE" in
    deflate|lzr|lzf|lzfi|lzssr|lz78r)
        trace_lz_pipeline "$PIPELINE" "$DEMUXER" "$STREAM_COUNT" "$ENTROPY"
        ;;
    bw|bbw)
        trace_bwt_pipeline "$PIPELINE"
        ;;
esac

if [[ "$FORMAT" == "text" ]]; then
    echo ""
    echo "─────────────────────────────────────────────────────────────"
    echo "Understanding the data flow:"
    echo "  - StageBlock is the container flowing through stages"
    echo "  - StageBlock.data: main byte vector (cleared after demux)"
    echo "  - StageBlock.streams: optional multi-stream payload"
    echo "  - StageBlock.metadata: carries BWT index, pre-entropy len, etc."
    echo ""
    echo "See CLAUDE.md 'Tracing data flow through pipelines' for details."
fi
