//! Embedded WGSL kernel sources and GPU configuration constants.

/// Embedded WGSL kernel source: top-K match finding for optimal parsing.
pub(super) const LZ77_TOPK_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_topk.wgsl");

/// Embedded WGSL kernel source: LZ77 match finding with lazy matching emulation.
pub(super) const LZ77_LAZY_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_lazy.wgsl");

/// Embedded WGSL kernel source: cooperative-stitch LZ77 match finding.
pub(super) const LZ77_COOP_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_coop.wgsl");

/// Embedded WGSL kernel source: GPU rank assignment for BWT prefix-doubling.
pub(super) const BWT_RANK_KERNEL_SOURCE: &str = include_str!("../../kernels/bwt_rank.wgsl");

/// Embedded WGSL kernel source: radix sort for BWT prefix-doubling.
pub(super) const BWT_RADIX_KERNEL_SOURCE: &str = include_str!("../../kernels/bwt_radix.wgsl");

/// Embedded WGSL kernel source: GPU Huffman encoding.
pub(super) const HUFFMAN_ENCODE_KERNEL_SOURCE: &str =
    include_str!("../../kernels/huffman_encode.wgsl");

/// Embedded WGSL kernel source: GPU FSE decode.
pub(super) const FSE_DECODE_KERNEL_SOURCE: &str = include_str!("../../kernels/fse_decode.wgsl");

/// Embedded WGSL kernel source: GPU FSE encode.
pub(super) const FSE_ENCODE_KERNEL_SOURCE: &str = include_str!("../../kernels/fse_encode.wgsl");

/// Embedded WGSL kernel source: GPU LZ77 block-parallel decompression.
pub(super) const LZ77_DECODE_KERNEL_SOURCE: &str = include_str!("../../kernels/lz77_decode.wgsl");

/// Embedded WGSL kernel source: GPU chunked rANS decode.
pub(super) const RANS_DECODE_KERNEL_SOURCE: &str = include_str!("../../kernels/rans_decode.wgsl");

/// Embedded WGSL kernel source: GPU chunked rANS encode.
pub(super) const RANS_ENCODE_KERNEL_SOURCE: &str = include_str!("../../kernels/rans_encode.wgsl");

/// Embedded WGSL kernel source: GPU LzSeq demux (match buffer → 6 streams).
pub(super) const LZSEQ_DEMUX_KERNEL_SOURCE: &str = include_str!("../../kernels/lzseq_demux.wgsl");

/// Embedded WGSL kernel source: GPU SortLZ operations (Experiment B).
pub(super) const SORTLZ_OPS_KERNEL_SOURCE: &str = include_str!("../../kernels/sortlz_ops.wgsl");

/// Embedded WGSL kernel source: GPU Huffman sync-point parallel decode.
pub(super) const HUFFMAN_DECODE_KERNEL_SOURCE: &str =
    include_str!("../../kernels/huffman_decode.wgsl");

/// Number of candidates per position in the top-K kernel (must match K in lz77_topk.wgsl).
pub(super) const TOPK_K: usize = 4;
