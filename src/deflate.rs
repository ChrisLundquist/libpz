/// RFC 1951 DEFLATE decompression (inflate).
///
/// Supports all three block types:
/// - Type 0: Stored (no compression)
/// - Type 1: Fixed Huffman codes
/// - Type 2: Dynamic Huffman codes
///
/// This module is intentionally thin glue: it combines the reusable
/// [`HuffTable`](crate::huffman::HuffTable) primitive with
/// DEFLATE-specific bit ordering, block framing, and LZ77
/// length/distance tables per RFC 1951.
use crate::huffman::HuffTable;
use crate::{PzError, PzResult};

// ---------------------------------------------------------------------------
// Bit reader – DEFLATE-specific LSB-first bit stream
// ---------------------------------------------------------------------------

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize, // byte position
    bit: u8,    // bit position within current byte (0..8)
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            bit: 0,
        }
    }

    /// Read `n` bits (1..=25) and return them as a u32, LSB first.
    fn read_bits(&mut self, n: u8) -> PzResult<u32> {
        let mut result: u32 = 0;
        let mut shift = 0u8;
        let mut remaining = n;

        while remaining > 0 {
            if self.pos >= self.data.len() {
                return Err(PzError::InvalidInput);
            }

            let avail = 8 - self.bit;
            let take = remaining.min(avail);
            let mask = (1u32 << take) - 1;
            let bits = ((self.data[self.pos] >> self.bit) as u32) & mask;
            result |= bits << shift;

            shift += take;
            remaining -= take;
            self.bit += take;
            if self.bit >= 8 {
                self.bit = 0;
                self.pos += 1;
            }
        }

        Ok(result)
    }

    /// Align to next byte boundary (discard remaining bits in current byte).
    fn align(&mut self) {
        if self.bit > 0 {
            self.bit = 0;
            self.pos += 1;
        }
    }

    /// Read a u16 from the aligned byte stream (little-endian).
    fn read_u16_aligned(&mut self) -> PzResult<u16> {
        self.align();
        if self.pos + 2 > self.data.len() {
            return Err(PzError::InvalidInput);
        }
        let val = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(val)
    }

    /// Decode one Huffman symbol using the given table.
    fn decode_symbol(&mut self, table: &HuffTable) -> PzResult<u16> {
        table.decode(&mut || self.read_bits(1))
    }
}

// ---------------------------------------------------------------------------
// DEFLATE-specific constants (RFC 1951)
// ---------------------------------------------------------------------------

/// Maximum number of literal/length symbols.
const MAX_LIT_CODES: usize = 288;
/// Maximum number of distance symbols.
const MAX_DIST_CODES: usize = 32;

/// Base lengths for length codes 257..285.
static LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra bits for length codes 257..285.
static LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Base distances for distance codes 0..29.
static DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra bits for distance codes 0..29.
static DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Order of code length alphabet codes (RFC 1951 section 3.2.7).
static CODELEN_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

// ---------------------------------------------------------------------------
// Fixed Huffman tables (RFC 1951 section 3.2.6)
// ---------------------------------------------------------------------------

fn build_fixed_lit_table() -> PzResult<HuffTable> {
    let mut lengths = [0u8; MAX_LIT_CODES];
    lengths[..=143].fill(8);
    lengths[144..=255].fill(9);
    lengths[256..=279].fill(7);
    lengths[280..=287].fill(8);
    HuffTable::from_lengths(&lengths)
}

fn build_fixed_dist_table() -> PzResult<HuffTable> {
    let lengths = [5u8; MAX_DIST_CODES];
    HuffTable::from_lengths(&lengths)
}

// ---------------------------------------------------------------------------
// Dynamic Huffman tables (RFC 1951 section 3.2.7)
// ---------------------------------------------------------------------------

fn decode_dynamic_tables(reader: &mut BitReader) -> PzResult<(HuffTable, HuffTable)> {
    let hlit = reader.read_bits(5)? as usize + 257;
    let hdist = reader.read_bits(5)? as usize + 1;
    let hclen = reader.read_bits(4)? as usize + 4;

    if hlit > MAX_LIT_CODES || hdist > MAX_DIST_CODES {
        return Err(PzError::InvalidInput);
    }

    // Read code length alphabet code lengths
    let mut codelen_lengths = [0u8; 19];
    for i in 0..hclen {
        codelen_lengths[CODELEN_ORDER[i]] = reader.read_bits(3)? as u8;
    }

    let codelen_table = HuffTable::from_lengths(&codelen_lengths)?;

    // Decode literal/length + distance code lengths
    let total = hlit + hdist;
    let mut lengths = vec![0u8; total];
    let mut i = 0;

    while i < total {
        let sym = reader.decode_symbol(&codelen_table)?;
        match sym {
            0..=15 => {
                lengths[i] = sym as u8;
                i += 1;
            }
            16 => {
                if i == 0 {
                    return Err(PzError::InvalidInput);
                }
                let repeat = reader.read_bits(2)? as usize + 3;
                let prev = lengths[i - 1];
                for _ in 0..repeat {
                    if i >= total {
                        return Err(PzError::InvalidInput);
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                let repeat = reader.read_bits(3)? as usize + 3;
                for _ in 0..repeat {
                    if i >= total {
                        return Err(PzError::InvalidInput);
                    }
                    lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                let repeat = reader.read_bits(7)? as usize + 11;
                for _ in 0..repeat {
                    if i >= total {
                        return Err(PzError::InvalidInput);
                    }
                    lengths[i] = 0;
                    i += 1;
                }
            }
            _ => return Err(PzError::InvalidInput),
        }
    }

    let lit_table = HuffTable::from_lengths(&lengths[..hlit])?;
    let dist_table = HuffTable::from_lengths(&lengths[hlit..hlit + hdist])?;

    Ok((lit_table, dist_table))
}

// ---------------------------------------------------------------------------
// Block decompression
// ---------------------------------------------------------------------------

fn inflate_stored(reader: &mut BitReader, output: &mut Vec<u8>) -> PzResult<()> {
    let len = reader.read_u16_aligned()?;
    let nlen = reader.read_u16_aligned()?;

    if len != !nlen {
        return Err(PzError::InvalidInput);
    }

    let len = len as usize;
    if reader.pos + len > reader.data.len() {
        return Err(PzError::InvalidInput);
    }

    output.extend_from_slice(&reader.data[reader.pos..reader.pos + len]);
    reader.pos += len;

    Ok(())
}

fn inflate_huffman(
    reader: &mut BitReader,
    lit_table: &HuffTable,
    dist_table: &HuffTable,
    output: &mut Vec<u8>,
) -> PzResult<()> {
    loop {
        let sym = reader.decode_symbol(lit_table)?;

        match sym {
            0..=255 => {
                output.push(sym as u8);
            }
            256 => {
                return Ok(());
            }
            257..=285 => {
                let len_idx = (sym - 257) as usize;
                if len_idx >= LENGTH_BASE.len() {
                    return Err(PzError::InvalidInput);
                }
                let length = LENGTH_BASE[len_idx] as usize
                    + reader.read_bits(LENGTH_EXTRA[len_idx])? as usize;

                let dist_sym = reader.decode_symbol(dist_table)? as usize;
                if dist_sym >= DIST_BASE.len() {
                    return Err(PzError::InvalidInput);
                }
                let distance =
                    DIST_BASE[dist_sym] as usize + reader.read_bits(DIST_EXTRA[dist_sym])? as usize;

                if distance > output.len() {
                    return Err(PzError::InvalidInput);
                }

                let start = output.len() - distance;
                for i in 0..length {
                    let b = output[start + i];
                    output.push(b);
                }
            }
            _ => return Err(PzError::InvalidInput),
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Decompress a raw DEFLATE stream (no gzip/zlib wrapper).
pub fn inflate(data: &[u8]) -> PzResult<Vec<u8>> {
    let mut reader = BitReader::new(data);
    let mut output = Vec::new();

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => inflate_stored(&mut reader, &mut output)?,
            1 => {
                let lit_table = build_fixed_lit_table()?;
                let dist_table = build_fixed_dist_table()?;
                inflate_huffman(&mut reader, &lit_table, &dist_table, &mut output)?;
            }
            2 => {
                let (lit_table, dist_table) = decode_dynamic_tables(&mut reader)?;
                inflate_huffman(&mut reader, &lit_table, &dist_table, &mut output)?;
            }
            3 => return Err(PzError::InvalidInput),
            _ => unreachable!(),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inflate_stored_block() {
        let mut data = Vec::new();
        // bfinal=1, btype=00 → 0x01
        data.push(0x01);
        // LEN = 5, NLEN = !5
        data.push(0x05);
        data.push(0x00);
        data.push(0xFA);
        data.push(0xFF);
        data.extend_from_slice(b"hello");

        let result = inflate(&data).unwrap();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_inflate_fixed_huffman() {
        let lit_table = build_fixed_lit_table().unwrap();
        assert_eq!(lit_table.symbol_count(), 288);

        let dist_table = build_fixed_dist_table().unwrap();
        assert_eq!(dist_table.symbol_count(), 32);
    }

    #[test]
    fn test_bit_reader_basics() {
        let data = [0b10110100u8, 0b01101001u8];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0b0100);
        assert_eq!(reader.read_bits(4).unwrap(), 0b1011);
        assert_eq!(reader.read_bits(8).unwrap(), 0b01101001);
    }

    #[test]
    fn test_bit_reader_cross_byte() {
        let data = [0xFF, 0x00];
        let mut reader = BitReader::new(&data);

        assert_eq!(reader.read_bits(4).unwrap(), 0x0F);
        // bits 4-7 of 0xFF then bits 0-3 of 0x00, packed LSB-first → 0x0F
        assert_eq!(reader.read_bits(8).unwrap(), 0x0F);
    }
}
