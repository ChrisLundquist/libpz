// pz2-G32 stage-2 probe kernel: literal Huffman phase of the 32-lane layout.
//
// One simdgroup (32 lanes, guaranteed width on Apple GPUs) decodes one tile:
// literal i is pinned to lane i % 32, each round emits 32 contiguous output
// bytes. The interleaved word stream is consumed in the exact order the
// encoder emitted it: before each symbol, every lane holding < 32 live bits
// fetches one word; the lane's word index inside the round is its rank among
// the refilling lanes (simd_ballot + popcount prefix — the metadata-free
// GDeflate trick). Fetches past a lane's real data hit the encoder's zero
// padding words, which keeps the schedule implicit (clamping instead would
// desynchronize — see pz2-g32-stage1-findings.md).
//
// Threadgroups bundle SG_PER_TG simdgroups that share one 4 KB canonical
// decode table in threadgroup memory (the host groups tiles by block and
// pads with lit_count == 0 dummies so every tile in a group uses the same
// table). NOT a shipping kernel: stage-2 measurement probe.

#include <metal_stdlib>
using namespace metal;

#define SG_PER_TG 8
#define TG_THREADS (SG_PER_TG * 32)

struct Tile {
    uint word_off;  // index into `words` (uint units)
    uint out_off;   // byte offset into `out`
    uint lit_count; // literals in this tile (0 = padding tile)
    uint table_off; // index into `tables` (ushort units)
};

kernel void g32_lit_decode(
    device const uint* words [[buffer(0)]],
    device const ushort* tables [[buffer(1)]],
    device const Tile* tiles [[buffer(2)]],
    device uchar* out [[buffer(3)]],
    constant uint& n_tiles [[buffer(4)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup ushort table[2048];

    uint first_tile = tg_id * SG_PER_TG;
    // Cooperative table load: every tile in this group shares table_off.
    uint toff = tiles[min(first_tile, n_tiles - 1)].table_off;
    for (uint i = tid; i < 2048; i += TG_THREADS) {
        table[i] = tables[toff + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint t = first_tile + sg;
    if (t >= n_tiles) {
        return; // uniform per simdgroup
    }
    Tile d = tiles[t];
    if (d.lit_count == 0) {
        return; // padding tile, uniform per simdgroup
    }

    device const uint* w = words + d.word_off;
    device uchar* o = out + d.out_off;
    ulong acc = 0;
    uint nb = 0;
    uint wpos = 0;
    uint rounds = (d.lit_count + 31) / 32;
    for (uint r = 0; r < rounds; ++r) {
        uint idx = r * 32 + lane;
        bool active = idx < d.lit_count;
        bool want = active && nb < 32;
        uint mask = (uint)((simd_vote::vote_t)simd_ballot(want));
        if (want) {
            uint before = popcount(mask & ((1u << lane) - 1u));
            acc |= (ulong)w[wpos + before] << nb;
            nb += 32;
        }
        wpos += popcount(mask);
        if (active) {
            ushort e = table[(uint)(acc & 2047)];
            uint len = e & 0xFu;
            acc >>= len;
            nb -= len;
            o[idx] = (uchar)(e >> 4);
        }
    }
}
