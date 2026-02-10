/// Arithmetic (Range) Coder.
///
/// Implements a byte-level range coder with adaptive frequency model.
/// Unlike Huffman coding which assigns integer bit-lengths to symbols,
/// a range coder can use fractional bits per symbol, achieving compression
/// closer to the Shannon entropy limit.
///
/// Uses the Subbotin carryless range coder technique:
/// - 32-bit range maintained with TOP (1<<24) and BOTTOM (1<<16) bounds
/// - When the top byte is settled (low and low+range agree), output it
/// - When range is too small, clamp to prevent carry propagation
///
/// **Adaptive model:**
/// - Start with uniform frequencies (count 1 for each byte value).
/// - After encoding/decoding each symbol, increment its count.
/// - Periodically halve all counts to adapt to local statistics.
use crate::{PzError, PzResult};

/// Maximum total frequency before rescaling.
/// Must fit so that `range / total` doesn't underflow to zero.
/// With BOTTOM = 1<<16, we need total < BOTTOM, so 1<<14 is safe.
const MAX_TOTAL_FREQ: u32 = 1 << 14;

/// Number of symbols in the alphabet (bytes 0-255).
const NUM_SYMBOLS: usize = 256;

/// Top boundary: output a byte when the top 8 bits are settled.
const TOP: u32 = 1 << 24;
/// Bottom boundary: when range falls below this, force a renormalization
/// with range clamping to prevent carry ambiguity.
const BOTTOM: u32 = 1 << 16;

/// Adaptive frequency model for the range coder.
#[derive(Debug, Clone)]
struct AdaptiveModel {
    /// Per-symbol frequency counts.
    freq: [u32; NUM_SYMBOLS],
    /// Cumulative frequency table: cumul[i] = sum of freq[0..i].
    cumul: [u32; NUM_SYMBOLS + 1],
    /// Total of all frequencies.
    total: u32,
}

impl AdaptiveModel {
    fn new() -> Self {
        let mut model = AdaptiveModel {
            freq: [1u32; NUM_SYMBOLS],
            cumul: [0u32; NUM_SYMBOLS + 1],
            total: NUM_SYMBOLS as u32,
        };
        model.rebuild_cumulative();
        model
    }

    fn rebuild_cumulative(&mut self) {
        self.cumul[0] = 0;
        for i in 0..NUM_SYMBOLS {
            self.cumul[i + 1] = self.cumul[i] + self.freq[i];
        }
        self.total = self.cumul[NUM_SYMBOLS];
    }

    fn update(&mut self, symbol: u8) {
        self.freq[symbol as usize] += 1;
        self.total += 1;

        if self.total >= MAX_TOTAL_FREQ {
            self.rescale();
        }

        self.rebuild_cumulative();
    }

    fn rescale(&mut self) {
        self.total = 0;
        for i in 0..NUM_SYMBOLS {
            self.freq[i] = self.freq[i].div_ceil(2);
            if self.freq[i] == 0 {
                self.freq[i] = 1;
            }
            self.total += self.freq[i];
        }
    }

    fn get_range(&self, symbol: u8) -> (u32, u32, u32) {
        let s = symbol as usize;
        (self.cumul[s], self.cumul[s + 1], self.total)
    }

    fn find_symbol(&self, value: u32) -> (u8, u32, u32) {
        let mut lo = 0usize;
        let mut hi = NUM_SYMBOLS;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cumul[mid + 1] <= value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo as u8, self.cumul[lo], self.cumul[lo + 1])
    }
}

/// Subbotin-style carryless range encoder.
struct RangeEncoder {
    low: u32,
    range: u32,
    output: Vec<u8>,
}

impl RangeEncoder {
    fn new() -> Self {
        RangeEncoder {
            low: 0,
            range: u32::MAX,
            output: Vec::new(),
        }
    }

    fn encode_symbol(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let r = self.range / total;
        self.low = self.low.wrapping_add(cum_low.wrapping_mul(r));
        if cum_high < total {
            self.range = (cum_high - cum_low) * r;
        } else {
            self.range -= cum_low * r;
        }
        self.normalize();
    }

    fn normalize(&mut self) {
        // Subbotin's carryless normalization:
        // Output bytes when top byte is settled, clamp range when needed.
        while self.low ^ self.low.wrapping_add(self.range) < TOP || self.range < BOTTOM {
            if self.low ^ self.low.wrapping_add(self.range) >= TOP {
                // Top byte not settled, but range < BOTTOM: clamp range
                self.range = self.low.wrapping_neg() & (BOTTOM - 1);
            }
            self.output.push((self.low >> 24) as u8);
            self.low <<= 8;
            self.range <<= 8;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        // Flush remaining state
        for _ in 0..4 {
            self.output.push((self.low >> 24) as u8);
            self.low <<= 8;
        }
        self.output
    }
}

/// Subbotin-style carryless range decoder.
struct RangeDecoder<'a> {
    low: u32,
    range: u32,
    code: u32,
    input: &'a [u8],
    pos: usize,
}

impl<'a> RangeDecoder<'a> {
    fn new(input: &'a [u8]) -> Self {
        let mut dec = RangeDecoder {
            low: 0,
            range: u32::MAX,
            code: 0,
            input,
            pos: 0,
        };
        for _ in 0..4 {
            dec.code = (dec.code << 8) | dec.read_byte() as u32;
        }
        dec
    }

    fn read_byte(&mut self) -> u8 {
        if self.pos < self.input.len() {
            let byte = self.input[self.pos];
            self.pos += 1;
            byte
        } else {
            0
        }
    }

    fn get_freq(&self, total: u32) -> u32 {
        let r = self.range / total;
        let value = self.code.wrapping_sub(self.low) / r;
        std::cmp::min(value, total - 1)
    }

    fn decode_symbol(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let r = self.range / total;
        self.low = self.low.wrapping_add(cum_low.wrapping_mul(r));
        if cum_high < total {
            self.range = (cum_high - cum_low) * r;
        } else {
            self.range -= cum_low * r;
        }
        self.normalize();
    }

    fn normalize(&mut self) {
        while self.low ^ self.low.wrapping_add(self.range) < TOP || self.range < BOTTOM {
            if self.low ^ self.low.wrapping_add(self.range) >= TOP {
                self.range = self.low.wrapping_neg() & (BOTTOM - 1);
            }
            self.code = (self.code << 8) | self.read_byte() as u32;
            self.low <<= 8;
            self.range <<= 8;
        }
    }
}

/// Encode data using an adaptive range coder.
///
/// Returns the compressed data. The original length must be known
/// by the decoder (stored externally).
pub fn encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut model = AdaptiveModel::new();
    let mut enc = RangeEncoder::new();

    for &byte in input {
        let (cum_low, cum_high, total) = model.get_range(byte);
        enc.encode_symbol(cum_low, cum_high, total);
        model.update(byte);
    }

    enc.finish()
}

/// Decode range-coded data.
///
/// `original_len` is the number of bytes in the original uncompressed data.
pub fn decode(input: &[u8], original_len: usize) -> PzResult<Vec<u8>> {
    if original_len == 0 {
        return Ok(Vec::new());
    }
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let mut model = AdaptiveModel::new();
    let mut dec = RangeDecoder::new(input);
    let mut output = Vec::with_capacity(original_len);

    for _ in 0..original_len {
        let value = dec.get_freq(model.total);
        let (symbol, cum_low, cum_high) = model.find_symbol(value);
        dec.decode_symbol(cum_low, cum_high, model.total);
        output.push(symbol);
        model.update(symbol);
    }

    Ok(output)
}

/// Decode range-coded data into a pre-allocated buffer.
///
/// Returns the number of bytes written.
pub fn decode_to_buf(input: &[u8], original_len: usize, output: &mut [u8]) -> PzResult<usize> {
    if original_len == 0 {
        return Ok(0);
    }
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }
    if output.len() < original_len {
        return Err(PzError::BufferTooSmall);
    }

    let mut model = AdaptiveModel::new();
    let mut dec = RangeDecoder::new(input);

    for slot in output.iter_mut().take(original_len) {
        let value = dec.get_freq(model.total);
        let (symbol, cum_low, cum_high) = model.find_symbol(value);
        dec.decode_symbol(cum_low, cum_high, model.total);
        *slot = symbol;
        model.update(symbol);
    }

    Ok(original_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(encode(&[]), Vec::<u8>::new());
        assert_eq!(decode(&[], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_single_byte() {
        let input = &[42u8];
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_repeated_byte() {
        let input = vec![b'a'; 100];
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
        assert!(encoded.len() < input.len());
    }

    #[test]
    fn test_two_bytes() {
        let input = b"ab";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_hello() {
        let input = b"hello, world!";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_banana() {
        let input = b"banana";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_all_bytes() {
        let input: Vec<u8> = (0..=255).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_longer_text() {
        let input = b"the quick brown fox jumps over the lazy dog. the quick brown fox jumps over the lazy dog.";
        let encoded = encode(input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_round_trip_binary() {
        let input: Vec<u8> = (0..500).map(|i| ((i * 37 + 13) % 256) as u8).collect();
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_compression_skewed() {
        let mut input = vec![0u8; 1000];
        input.push(1);
        input.push(2);
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
        assert!(
            encoded.len() < input.len() / 2,
            "encoded {} bytes, expected < {}",
            encoded.len(),
            input.len() / 2
        );
    }

    #[test]
    fn test_adaptive_model_basic() {
        let mut model = AdaptiveModel::new();
        assert_eq!(model.total, 256);
        model.update(b'a');
        assert_eq!(model.freq[b'a' as usize], 2);
        assert_eq!(model.total, 257);
    }

    #[test]
    fn test_adaptive_model_rescale() {
        let mut model = AdaptiveModel::new();
        for _ in 0..MAX_TOTAL_FREQ {
            model.update(0);
        }
        assert!(model.total < MAX_TOTAL_FREQ);
    }

    #[test]
    fn test_adaptive_model_find_symbol() {
        let model = AdaptiveModel::new();
        let (sym, low, high) = model.find_symbol(0);
        assert_eq!(sym, 0);
        assert_eq!(low, 0);
        assert_eq!(high, 1);

        let (sym, low, high) = model.find_symbol(100);
        assert_eq!(sym, 100);
        assert_eq!(low, 100);
        assert_eq!(high, 101);
    }

    #[test]
    fn test_decode_to_buf() {
        let input = b"hello, world!";
        let encoded = encode(input);
        let mut buf = vec![0u8; 100];
        let size = decode_to_buf(&encoded, input.len(), &mut buf).unwrap();
        assert_eq!(&buf[..size], input);
    }

    #[test]
    fn test_decode_to_buf_too_small() {
        let input = b"hello, world!";
        let encoded = encode(input);
        let mut buf = vec![0u8; 2];
        assert_eq!(
            decode_to_buf(&encoded, input.len(), &mut buf),
            Err(PzError::BufferTooSmall)
        );
    }

    #[test]
    fn test_round_trip_medium() {
        let mut input = Vec::new();
        for _ in 0..20 {
            input.extend(b"The Burrows-Wheeler transform clusters bytes. ");
        }
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
    }

    #[test]
    fn test_compresses_repeated_data() {
        let input = vec![0u8; 500];
        let encoded = encode(&input);
        let decoded = decode(&encoded, input.len()).unwrap();
        assert_eq!(decoded, input);
        // Should compress significantly
        assert!(
            encoded.len() < 100,
            "encoded {} bytes, expected < 100",
            encoded.len()
        );
    }
}
