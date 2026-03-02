## Per-stream rANS overhead: 512 bytes freq table per stream

Each rANS-encoded stream carries a 512-byte frequency table (256 × u16) in the wire format. LzSeqR has 4 rANS streams per block (flags, literals, offset_codes, length_codes — the extra-bits streams bypass rANS). That's 2 KB of tables per block.

Sparse frequency tables (only store nonzero entries) save ~1.3 KB/block for LzSeq but only close ~9% of the gzip gap on the Canterbury corpus. The savings are proportionally better on small files but the absolute numbers are small.

**For future agents estimating ratio improvements**: Don't overestimate framing/overhead wins. The gap to gzip is ~906 KB on 13.9 MB. Frequency table savings are ~83 KB total. The real gap is in per-match encoding cost across hundreds of thousands of tokens.
