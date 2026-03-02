## LzSeq is the right pipeline family for compression ratio work

Pipeline ratio comparison (Canterbury corpus vs gzip 28.6%):
- PZ-Deflate: 43.4% — fixed-width 5-byte LZ77 tokens, huge gap
- PZ-LzR: 41.6% — same fixed-width tokens
- PZ-LzSeqH: 36.8% — code+extra-bits, Huffman entropy
- PZ-LzSeqR: 35.1% — code+extra-bits, rANS entropy

LzSeq's advantages: log2-based code tables, packed extra bits, 3 repeat offsets (0 extra bits each), 6 independent streams for good symbol density. It already has a GPU fused match+demux path (lzseq_encode_gpu).

The format is not yet released, so changes are free. Highest-leverage ratio improvements:
1. Zstd-style sequences (literal_run_length, offset, match_length) — eliminates flags stream
2. Combined literal/length alphabet (Deflate-style 286-symbol tree)
3. Entropy-coding the extra bits instead of raw packing
4. Larger repeat offset cache (4-8 instead of 3)

**For future agents**: Focus ratio work on LzSeq pipelines. Don't optimize LZ77-based pipelines (Deflate, Lzr, Lzf) for ratio — their 5-byte-per-match format is the fundamental bottleneck.
