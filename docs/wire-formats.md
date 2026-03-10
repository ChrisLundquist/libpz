# Wire Formats Reference

Pre-release format (pre-1.0). All formats subject to change.

## Container Format (V2, multi-block)

```
[magic: 2 bytes = "PZ"]
[version: u8 = 2]
[pipeline_id: u8]              see Pipeline ID Table below
[original_len: u32 LE]         total uncompressed size
[num_blocks: u32 LE]           number of blocks
Block table (num_blocks entries):
  [compressed_len: u32 LE]
  [original_len: u32 LE]
Block data: concatenated compressed block bytes
```

**Framed (streaming) mode:** `num_blocks = 0xFFFFFFFF`, blocks are
length-delimited pairs `[compressed_len: u32][original_len: u32][data]`
terminated by `compressed_len = 0`.

## Per-Block: Multi-stream Entropy Container

Each block stores its pre-entropy streams in a multi-stream container:

```
[num_streams: u8]
[pre_entropy_len: u32 LE]      total pre-entropy byte count (for metadata)
[meta_len: u16 LE]
[meta: meta_len bytes]         encoder-specific metadata (round-trips through entropy)
Per-stream framing (depends on entropy coder):
  [orig_len: u32 LE]           uncompressed stream length
  [compressed_len: u32 LE]     compressed length (high bits may carry flags)
  [payload: compressed_len bytes]
```

### Per-stream flags (rANS pipelines)

The `compressed_len` field's high bits carry variant flags:

| Bit | Flag | Meaning |
|-----|------|---------|
| 31 | `RANS_INTERLEAVED_FLAG` | N-way interleaved rANS payload |
| 30 | `RANS_RECOIL_FLAG` | Recoil split-point metadata appended |
| 29 | `RANS_SHARED_STREAM_FLAG` | Shared-stream rANS (ryg_rans-style) |

## Pre-entropy Stream Formats (TokenEncoder)

### LzSeqEncoder (6 streams)

Used by: **Lzf**, **LzSeqR**, **LzSeqH**, **SortLz** (as MatchFinder)

Log2-coded offsets/lengths with repeat offset tracking. Best ratio.

| Stream | Contents |
|--------|----------|
| flags | Packed bits MSB-first (1=literal, 0=match) |
| literals | u8 per literal token |
| offset_codes | u8 per match (0-2 = repeat offset, 3+ = literal offset code) |
| offset_extra | LSB-first packed bitstream (extra bits per offset code) |
| length_codes | u8 per match |
| length_extra | LSB-first packed bitstream (extra bits per length code) |

**Meta** (8 bytes): `[num_tokens: u32 LE][num_matches: u32 LE]`

### LzssEncoder (4 streams)

Used by: **Lzfi**, **LzssR**

Flag bits + raw u16 offsets/lengths.

| Stream | Contents |
|--------|----------|
| flags | Packed bits MSB-first (1=literal, 0=match) |
| literals | u8 per literal token |
| offsets | u16 LE per match |
| lengths | u16 LE per match |

**Meta** (4 bytes): `[num_tokens: u32 LE]`

## Entropy Coders

### FSE (Finite State Entropy)

Used by: Lzf (stage 1), Lzfi (interleaved), Bw (stage 3), SortLz

Per-stream: `[orig_len: u32 LE][compressed_len: u32 LE][fse_data]`

### rANS (range ANS)

Used by: LzSeqR, LzssR

Single-stream format:
```
[scale_bits: u8] [freq_table: 256 x u16 LE] [final_state: u32 LE]
[num_words: u32 LE] [words: num_words x u16 LE]
```

Interleaved N-way format:
```
[scale_bits: u8] [freq_table: 256 x u16 LE] [num_states: u8]
[final_states: N x u32 LE] [num_words: N x u32 LE]
[stream_0_words] [stream_1_words] ... [stream_N-1_words]
```

### Huffman

Used by: LzSeqH

Per-stream: `[data_len: u32 LE][total_bits: u32 LE][freq_table: 256 x u32 LE][data]`

## SortLz Standalone Wire Format (v2)

Pipeline::SortLz (ID 10) uses its own framing with `LzSeqEncoder` + FSE:

```
[meta_len: u16 LE]
[meta: meta_len bytes]         LzSeq metadata (num_tokens + num_matches)
[num_streams: u8]              6 (LzSeq streams)
Per stream:
  [orig_len: u32 LE]           uncompressed stream length
  [fse_len: u32 LE]            FSE-compressed length
  [fse_data: fse_len bytes]
```

## Pipeline ID Table

| ID | Pipeline | Pre-entropy | Entropy | Streams |
|----|----------|-------------|---------|---------|
| 1 | Bw | BWT+MTF+RLE | FSE | 1 |
| 2 | Bbw | BBWT+MTF+RLE | FSE | 1 |
| 4 | Lzf | LzSeqEncoder | FSE | 6 |
| 5 | Lzfi | LzssEncoder | Interleaved FSE | 4 |
| 6 | LzssR | LzssEncoder | rANS | 4 |
| 8 | LzSeqR | LzSeqEncoder | rANS | 6 |
| 9 | LzSeqH | LzSeqEncoder | Huffman | 6 |
| 10 | SortLz | SortLz own | FSE | 6 |

**Retired IDs:** 0 (Deflate), 3 (Lzr), 7 (Lz78R)
