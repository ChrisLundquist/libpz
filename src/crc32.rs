/// CRC32 implementation (ISO 3309 / ITU-T V.42 / gzip).
///
/// Uses the standard polynomial 0xEDB88320 (bit-reversed 0x04C11DB7)
/// with a 256-entry lookup table for byte-at-a-time processing.
const CRC32_POLY: u32 = 0xEDB8_8320;

/// Pre-computed CRC32 lookup table (256 entries).
const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut crc = i;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ CRC32_POLY;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i as usize] = crc;
        i += 1;
    }
    table
}

static TABLE: [u32; 256] = make_table();

/// Compute CRC32 of a byte slice.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ TABLE[idx];
    }
    crc ^ 0xFFFF_FFFF
}

/// Incremental CRC32 state.
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self { state: 0xFFFF_FFFF }
    }
}

impl Crc32 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, data: &[u8]) {
        for &b in data {
            let idx = ((self.state ^ b as u32) & 0xFF) as usize;
            self.state = (self.state >> 8) ^ TABLE[idx];
        }
    }

    pub fn finalize(&self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(b""), 0x0000_0000);
    }

    #[test]
    fn test_crc32_known_vectors() {
        // "123456789" → 0xCBF43926 (standard test vector)
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_hello() {
        // Known value for "hello"
        assert_eq!(crc32(b"hello"), 0x3610_A686);
    }

    #[test]
    fn test_crc32_incremental() {
        let data = b"123456789";
        let mut c = Crc32::new();
        c.update(&data[..4]);
        c.update(&data[4..]);
        assert_eq!(c.finalize(), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_single_bytes() {
        let data = b"123456789";
        let mut c = Crc32::new();
        for &b in data {
            c.update(&[b]);
        }
        assert_eq!(c.finalize(), 0xCBF4_3926);
    }
}
