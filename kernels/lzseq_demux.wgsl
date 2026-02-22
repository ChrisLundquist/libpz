// LzSeq GPU demux kernel.
//
// Single-threaded sequential walk through GPU match buffer, converting
// per-position matches into LzSeq's 6 output streams: flags, literals,
// offset_codes, offset_extra, length_codes, length_extra.
//
// Eliminates the PCIe download of 12 bytes/position by keeping match data
// on-device and producing only the small compressed streams (~5-10% of
// match buffer size).
//
// No repeat offsets on GPU — all offset codes are literal (shifted by
// NUM_REPEAT_CODES=3). The decoder handles this transparently.
//
// @pz_cost {
//   threads_per_element: 0
//   passes: 1
//   buffers: input=N+7, gpu_matches=N*12, params=32, output=N*4
//   local_mem: 0
//   note: single-thread serial walk, output is worst-case sized
// }

const MIN_MATCH: u32 = 3u;
const NUM_REPEAT_CODES: u32 = 3u;
// Maximum match length (must fit in u16 for decode_length compatibility).
const MAX_MATCH_LENGTH: u32 = 65535u;

// Counter indices in output buffer (first 8 u32s reserved for counters + padding)
const COUNTER_NUM_TOKENS: u32 = 0u;
const COUNTER_NUM_MATCHES: u32 = 1u;
const COUNTER_NUM_LITERALS: u32 = 2u;
const COUNTER_OFF_EXTRA_BITS: u32 = 3u;
const COUNTER_LEN_EXTRA_BITS: u32 = 4u;

struct Params {
    // p0.x = input_len
    // p0.y = flags section offset (u32 words)
    // p0.z = literals section offset (u32 words)
    // p0.w = offset_codes section offset (u32 words)
    p0: vec4<u32>,
    // p1.x = offset_extra section offset (u32 words)
    // p1.y = length_codes section offset (u32 words)
    // p1.z = length_extra section offset (u32 words)
    // p1.w = unused
    p1: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input_data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> gpu_matches: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

// ---------------------------------------------------------------------------
// Byte I/O helpers
// ---------------------------------------------------------------------------

fn read_input_byte(pos: u32) -> u32 {
    let word_idx = pos / 4u;
    let byte_idx = pos % 4u;
    return (input_data[word_idx] >> (byte_idx * 8u)) & 0xFFu;
}

fn read_match_offset(pos: u32) -> u32 {
    return gpu_matches[pos * 3u];
}

fn read_match_length(pos: u32) -> u32 {
    return gpu_matches[pos * 3u + 1u];
}

// Write a byte into a packed byte-array section (little-endian u32 words).
fn write_packed_byte(section_offset: u32, byte_index: u32, value: u32) {
    let word_idx = section_offset + byte_index / 4u;
    let byte_pos = byte_index % 4u;
    output[word_idx] = output[word_idx] | (value << (byte_pos * 8u));
}

// Set a flag bit (MSB-first packing, 1=literal, matching CPU pack_flags).
// In little-endian u32 layout, byte K occupies u32 bits [K*8 .. K*8+7].
fn set_flag_bit(token_index: u32) {
    let word_idx = params.p0.y + token_index / 32u;
    let t = token_index % 32u;
    let byte_in_u32 = t / 8u;
    let bit_in_byte = 7u - (t % 8u);
    let u32_bit = byte_in_u32 * 8u + bit_in_byte;
    output[word_idx] = output[word_idx] | (1u << u32_bit);
}

// Write bits into a packed bitstream (LSB-first, matching CPU BitWriter).
fn write_bits_lsb(section_offset: u32, bit_pos: u32, value: u32, nb_bits: u32) {
    if (nb_bits == 0u) {
        return;
    }
    let word_idx = section_offset + bit_pos / 32u;
    let bit_in_word = bit_pos % 32u;
    output[word_idx] = output[word_idx] | (value << bit_in_word);
    if (bit_in_word + nb_bits > 32u) {
        output[word_idx + 1u] = output[word_idx + 1u] | (value >> (32u - bit_in_word));
    }
}

// ---------------------------------------------------------------------------
// Code table (matches CPU encode_value / encode_offset / encode_length)
// ---------------------------------------------------------------------------

// Encode a 1-based positive integer to (code, extra_bits_count, extra_value).
// Code 0: value 1 (0 extra bits)
// Code 1: value 2 (0 extra bits)
// Code N (N>=2): base = 1 + 2^(N-1), extra_bits = N-1
fn encode_value(value: u32) -> vec3<u32> {
    if (value == 1u) {
        return vec3<u32>(0u, 0u, 0u);
    }
    if (value == 2u) {
        return vec3<u32>(1u, 0u, 0u);
    }
    let code = 32u - countLeadingZeros(value - 1u);
    let extra_bits = code - 1u;
    let base = 1u + (1u << (code - 1u));
    let extra_value = value - base;
    return vec3<u32>(code, extra_bits, extra_value);
}

// extra_bits_for_code: code < 2 → 0, else code - 1
fn extra_bits_for_code(code: u32) -> u32 {
    return select(code - 1u, 0u, code < 2u);
}

// Distance-dependent minimum profitable match length (matches CPU min_profitable_length).
fn min_profitable_length(offset: u32) -> u32 {
    if (offset == 0u) {
        return 0xFFFFu;
    }
    let v = encode_value(offset);
    let oeb = extra_bits_for_code(v.x);
    let excess = select(oeb - 7u, 0u, oeb <= 7u);
    return MIN_MATCH + (excess + 3u) / 4u;
}

// ---------------------------------------------------------------------------
// Main kernel: sequential demux walk
// ---------------------------------------------------------------------------

@compute @workgroup_size(1)
fn lzseq_demux(@builtin(global_invocation_id) gid: vec3<u32>) {
    let input_len = params.p0.x;
    if (input_len == 0u) {
        return;
    }

    var pos: u32 = 0u;
    var num_tokens: u32 = 0u;
    var num_matches: u32 = 0u;
    var num_literals: u32 = 0u;
    var off_extra_bit_pos: u32 = 0u;
    var len_extra_bit_pos: u32 = 0u;

    while (pos < input_len) {
        let match_offset = read_match_offset(pos);
        var match_length = read_match_length(pos);

        // Cap match length to remaining input and u16::MAX
        let remaining = input_len - pos;
        if (match_length >= remaining) {
            match_length = select(remaining - 1u, 0u, remaining > 0u);
        }
        if (match_length > MAX_MATCH_LENGTH) {
            match_length = MAX_MATCH_LENGTH;
        }

        // Distance-dependent minimum profitable length
        let min_len = min_profitable_length(match_offset);

        if (match_length >= MIN_MATCH && match_offset > 0u && match_length >= min_len) {
            // --- Match token (flag=0, default in zero-initialized output) ---

            // Encode offset (no repeat offsets: shift by NUM_REPEAT_CODES)
            let ov = encode_value(match_offset);
            let offset_code = ov.x + NUM_REPEAT_CODES;
            let offset_extra_bits = ov.y;
            let offset_extra_value = ov.z;

            // Encode length (MIN_MATCH bias: length 3 → value 1)
            let adj_len = match_length - MIN_MATCH + 1u;
            let lv = encode_value(adj_len);
            let length_code = lv.x;
            let length_extra_bits = lv.y;
            let length_extra_value = lv.z;

            // Write offset code byte
            write_packed_byte(params.p0.w, num_matches, offset_code);

            // Write offset extra bits (LSB-first)
            write_bits_lsb(params.p1.x, off_extra_bit_pos, offset_extra_value, offset_extra_bits);
            off_extra_bit_pos += offset_extra_bits;

            // Write length code byte
            write_packed_byte(params.p1.y, num_matches, length_code);

            // Write length extra bits (LSB-first)
            write_bits_lsb(params.p1.z, len_extra_bit_pos, length_extra_value, length_extra_bits);
            len_extra_bit_pos += length_extra_bits;

            num_matches += 1u;
            pos += match_length;
        } else {
            // --- Literal token (flag=1) ---
            set_flag_bit(num_tokens);
            write_packed_byte(params.p0.z, num_literals, read_input_byte(pos));
            num_literals += 1u;
            pos += 1u;
        }

        num_tokens += 1u;
    }

    // Write counters
    output[COUNTER_NUM_TOKENS] = num_tokens;
    output[COUNTER_NUM_MATCHES] = num_matches;
    output[COUNTER_NUM_LITERALS] = num_literals;
    output[COUNTER_OFF_EXTRA_BITS] = off_extra_bit_pos;
    output[COUNTER_LEN_EXTRA_BITS] = len_extra_bit_pos;
}
