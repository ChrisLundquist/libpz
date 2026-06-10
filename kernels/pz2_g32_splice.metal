// pz2-G32 stage-3 probe kernel: cooperative sequence splice.
//
// One threadgroup (= one simdgroup, 32 lanes) splices one pz2 block, cribbing
// the leader/cooperative pattern of kernels/lz77_decode.wgsl but restructured
// to keep the serial residue off the memory-latency critical path:
//
//  Phase A — lanes 0..2 decode the three independent sequence-code Huffman
//    lanes (ll, of, ml) into a device scratch array: three concurrent serial
//    chains instead of one. Tables live in threadgroup memory.
//  Phase B — rounds of 32 sequences. The extras bit-width of a sequence is a
//    pure function of its three codes, so each lane computes its own extras
//    bit offset with simd_prefix_exclusive_sum and extracts its (ll, ml,
//    raw offset) in parallel. The repeat-offset LRU is the one true serial
//    dependency: every lane replays the same 32-step ALU resolve loop
//    (uniform, no memory traffic) using simd_shuffle broadcasts. Output and
//    literal cursors come from two more prefix sums. The copies then execute
//    in sequence order: literal run and match copy are cooperative across
//    all 32 lanes, with simdgroup_barrier(device) between dependent copies.
//    Overlapping matches (offset < 32) read the pre-match period via modulo
//    indexing — every read lands before the match start, so no inner
//    barriers are needed.
//
// The literal stream is consumed from the lits buffer produced by the
// stage-2 literal kernel (LIT_HUFF blocks) or staged raw bytes (LIT_RAW).
// NOT a shipping kernel: stage-3 measurement probe.

#include <metal_stdlib>
using namespace metal;

constant uint LANE_CONST = 0xFFFFFFFFu; // table_off sentinel: constant lane

struct SpliceLane {
    uint table_off; // ushort index into seq_tables, or LANE_CONST
    uint const_val; // code value when LANE_CONST
    uint bits_off;  // byte offset into seq_bits
    uint bits_len;  // byte length
};

struct SpliceBlock {
    uint seq_count;
    uint lit_off;     // byte offset of this block's literals in `lits`
    uint lit_count;   // total literal bytes of the block
    uint out_off;     // byte offset of this block's output
    uint out_len;     // original block length
    uint scratch_off; // byte offset into `codes` scratch (3 * seq_count)
    SpliceLane ll;
    SpliceLane of;
    SpliceLane ml;
    uint ex_off; // byte offset into seq_bits (extras stream, 8B zero-padded)
    uint ex_len;
};

// vdecode: log2-bucket value coding for literal-run and match lengths.
inline uint vdecode(uint code, uint extra) {
    return code == 0 ? 0u : ((1u << (code - 1u)) + extra);
}
inline uint vbits(uint code) {
    return code == 0 ? 0u : code - 1u;
}
// Raw (non-repeat) offset value coding.
inline uint off_decode_raw(uint code, uint extra) {
    return code == 0 ? 1u : (code == 1u ? 2u : (1u + (1u << (code - 1u)) + extra));
}
inline uint off_extra_bits(uint code) {
    // code is the wire code (repeat codes 0..2 take 0 extra bits).
    if (code < 3u) {
        return 0u;
    }
    uint raw = code - 3u;
    return raw < 2u ? 0u : raw - 1u;
}

// Read `n` (<= 30) bits at absolute bit position `pos` from a byte stream
// that the host zero-padded by >= 8 bytes past its real end.
inline uint read_bits(device const uchar* base, uint pos, uint n) {
    if (n == 0u) {
        return 0u;
    }
    device const uchar* p = base + (pos >> 3u);
    ulong w = (ulong)p[0] | ((ulong)p[1] << 8) | ((ulong)p[2] << 16) | ((ulong)p[3] << 24)
        | ((ulong)p[4] << 32);
    return (uint)((w >> (pos & 7u)) & ((1ul << n) - 1ul));
}

kernel void g32_splice(
    device const uchar* seq_bits [[buffer(0)]],   // all lanes' bitstreams + extras
    device const ushort* seq_tables [[buffer(1)]],// flat 2048-entry tables
    device const SpliceBlock* blocks [[buffer(2)]],
    device const uchar* lits [[buffer(3)]],       // literal phase output
    device uchar* codes [[buffer(4)]],            // phase-A scratch
    device uchar* out [[buffer(5)]],
    constant uint& n_blocks [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup ushort tbl[3][2048];

    if (tg_id >= n_blocks) {
        return;
    }
    SpliceBlock blk = blocks[tg_id];

    // Cooperative table loads (constant lanes skip theirs).
    SpliceLane lanes3[3] = {blk.ll, blk.of, blk.ml};
    for (uint s = 0; s < 3; ++s) {
        if (lanes3[s].table_off != LANE_CONST) {
            for (uint i = lane; i < 2048u; i += 32u) {
                tbl[s][i] = seq_tables[lanes3[s].table_off + i];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint seq_count = blk.seq_count;

    // ---- Phase A: three concurrent serial Huffman chains (lanes 0..2);
    // constant lanes are filled cooperatively by all 32 lanes. ----
    for (uint s = 0; s < 3u; ++s) {
        if (lanes3[s].table_off == LANE_CONST) {
            device uchar* dst = codes + blk.scratch_off + s * seq_count;
            for (uint i = lane; i < seq_count; i += 32u) {
                dst[i] = (uchar)lanes3[s].const_val;
            }
        }
    }
    if (lane < 3u && seq_count > 0u && lanes3[lane].table_off != LANE_CONST) {
        SpliceLane L = lanes3[lane];
        device uchar* dst = codes + blk.scratch_off + lane * seq_count;
        device const uchar* bits = seq_bits + L.bits_off;
        uint len = L.bits_len;
        ulong acc = 0;
        uint nb = 0;
        uint pos = 0;
        for (uint i = 0; i < seq_count; ++i) {
            while (nb <= 56u && pos < len) {
                acc |= (ulong)bits[pos++] << nb;
                nb += 8u;
            }
            ushort e = tbl[lane][(uint)(acc & 2047ul)];
            uint l = e & 0xFu;
            acc >>= l;
            nb -= l;
            dst[i] = (uchar)(e >> 4);
        }
    }
    simdgroup_barrier(mem_flags::mem_device);

#ifdef SKIP_PHASE_B
    return; // diagnostic build: time phase A alone
#endif

    // ---- Phase B: rounds of 32 sequences ----
    device const uchar* cl = codes + blk.scratch_off;
    device const uchar* ex = seq_bits + blk.ex_off;
    device uchar* o = out + blk.out_off;
    device const uchar* li = lits + blk.lit_off;

    uint rep0 = 1u, rep1 = 1u, rep2 = 1u; // uniform across lanes
    uint out_pos = 0u;
    uint lit_pos = 0u;
    uint ex_bit = 0u;
    uint rounds = (seq_count + 31u) / 32u;

    for (uint r = 0; r < rounds; ++r) {
        uint i = r * 32u + lane;
        bool active = i < seq_count;
        uint llc = active ? (uint)cl[i] : 0u;
        uint ofc = active ? (uint)cl[seq_count + i] : 0u;
        uint mlc = active ? (uint)cl[2u * seq_count + i] : 0u;

        uint llb = active ? vbits(llc) : 0u;
        uint ofb = active ? off_extra_bits(ofc) : 0u;
        uint mlb = active ? vbits(mlc) : 0u;
        uint my_bits = llb + ofb + mlb;
        uint bit0 = ex_bit + simd_prefix_exclusive_sum(my_bits);
        ex_bit += simd_sum(my_bits);

        uint ll = vdecode(llc, read_bits(ex, bit0, llb));
        uint of_extra = read_bits(ex, bit0 + llb, ofb);
        uint ml = vdecode(mlc, read_bits(ex, bit0 + llb + ofb, mlb)) + 3u;
        uint raw_off = ofc >= 3u ? off_decode_raw(ofc - 3u, of_extra) : 0u;

        // Serial repeat-offset resolve: every lane replays the same uniform
        // ALU loop; lane j's resolved offset is captured when k == lane.
        uint n_in_round = min(32u, seq_count - r * 32u);
        uint my_off = 0u;
        for (uint k = 0; k < n_in_round; ++k) {
            uint c = simd_shuffle(ofc, k);
            uint rw = simd_shuffle(raw_off, k);
            uint off_k;
            if (c == 0u) {
                off_k = rep0;
            } else if (c == 1u) {
                off_k = rep1;
                rep1 = rep0;
                rep0 = off_k;
            } else if (c == 2u) {
                off_k = rep2;
                rep2 = rep1;
                rep1 = rep0;
                rep0 = off_k;
            } else {
                off_k = rw;
                rep2 = rep1;
                rep1 = rep0;
                rep0 = off_k;
            }
            if (lane == k) {
                my_off = off_k;
            }
        }

        // Position prefix sums (inactive lanes contribute 0).
        uint round_start = out_pos;
        uint adv_out = active ? ll + ml : 0u;
        uint adv_lit = active ? ll : 0u;
        uint my_out = out_pos + simd_prefix_exclusive_sum(adv_out);
        uint my_lit = lit_pos + simd_prefix_exclusive_sum(adv_lit);
        out_pos += simd_sum(adv_out);
        lit_pos += simd_sum(adv_lit);

        // ---- Copies, classified (the GDeflate-style fast path) ----
        // Literal runs are mutually independent (disjoint dst, sources in
        // the lits buffer): every lane copies its own. A match whose source
        // ends before this round's first write (src + ml <= round_start)
        // reads only finished output: its lane copies it immediately too.
        // Everything else (intra-round-dependent or oversized) falls to an
        // in-order cooperative pass. Per-sequence barriers vanish for the
        // common case; one barrier per round publishes the writes.
        uint dst = my_out + ll;
        uint src = dst - my_off;
        bool indep = active && (src + ml <= round_start);
        const uint BIG = 128u;
        bool big_lit = active && ll > BIG;
        bool coop_match = active && (!indep || ml > BIG);

        if (active && !big_lit) {
            for (uint t = 0; t < ll; ++t) {
                o[my_out + t] = li[my_lit + t];
            }
        }
        if (indep && ml <= BIG) {
            for (uint t = 0; t < ml; ++t) {
                o[dst + t] = o[src + t];
            }
        }

        uint coop = (uint)((simd_vote::vote_t)simd_ballot(big_lit || coop_match));
        if (coop != 0u) {
            simdgroup_barrier(mem_flags::mem_device);
            uint blits = (uint)((simd_vote::vote_t)simd_ballot(big_lit));
            uint match_mask = (uint)((simd_vote::vote_t)simd_ballot(coop_match));
            // Ascending sequence order preserves intra-round dependencies.
            for (uint m = coop; m != 0u; m &= m - 1u) {
                uint k = ctz(m);
                uint bll = simd_shuffle(ll, k);
                uint bml = simd_shuffle(ml, k);
                uint boff = simd_shuffle(my_off, k);
                uint bout = simd_shuffle(my_out, k);
                uint blit = simd_shuffle(my_lit, k);
                if ((blits >> k) & 1u) {
                    for (uint t = lane; t < bll; t += 32u) {
                        o[bout + t] = li[blit + t];
                    }
                    simdgroup_barrier(mem_flags::mem_device);
                }
                if ((match_mask >> k) & 1u) {
                    uint d2 = bout + bll;
                    uint s2 = d2 - boff;
                    if (boff >= bml) {
                        for (uint t = lane; t < bml; t += 32u) {
                            o[d2 + t] = o[s2 + t];
                        }
                    } else if (boff >= 32u) {
                        // Each 32-byte wave reads bytes finished by earlier
                        // waves; barrier per wave.
                        for (uint base = 0; base < bml; base += 32u) {
                            uint t = base + lane;
                            if (t < bml) {
                                o[d2 + t] = o[s2 + t];
                            }
                            simdgroup_barrier(mem_flags::mem_device);
                        }
                    } else {
                        // Overlapping: replicate the pre-match period; reads
                        // land strictly before d2.
                        for (uint t = lane; t < bml; t += 32u) {
                            o[d2 + t] = o[s2 + (t % boff)];
                        }
                    }
                    simdgroup_barrier(mem_flags::mem_device);
                }
            }
        }
        // Publish this round's writes for the next round's fast path.
        simdgroup_barrier(mem_flags::mem_device);
    }

    // Trailing literals.
    uint trailing = blk.lit_count - lit_pos;
    for (uint t = lane; t < trailing; t += 32u) {
        o[out_pos + t] = li[lit_pos + t];
    }
}
