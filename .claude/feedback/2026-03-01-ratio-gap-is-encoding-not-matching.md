## The ratio gap vs gzip is encoding efficiency, not match quality

**Wrong assumption**: GPU match quality is the bottleneck (fewer probes = worse matches = worse ratio).

**Reality**: GPU match quality is competitive with the CPU optimal DP parser. The gap is in how matches are *encoded*:
- LZ77 pipelines (Deflate, Lzr, Lzf) use 5 bytes per match regardless of distance/length
- gzip uses variable-length Huffman codes averaging ~2-3 bytes per match
- LzSeqR uses code+extra-bits encoding and is much closer to gzip (35.1% vs 28.6%)

**Implication**: Work on match quality (better probes, better lazy evaluation) has diminishing returns. Work on encoding efficiency (LzSeq improvements, zstd-style sequences) has the highest ratio leverage.
