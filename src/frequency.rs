//! Frequency analysis for byte streams.
//!
//! Counts the occurrence of each byte value (0-255) in an input buffer
//! and computes Shannon entropy.

use std::sync::OnceLock;

/// A frequency table that tracks byte occurrence counts.
#[derive(Debug, Clone)]
pub struct FrequencyTable {
    /// Count of each byte value (index = byte value, value = count).
    pub byte: [u32; 256],
    /// Sum of all counts.
    pub total: u64,
    /// Number of distinct byte values with nonzero count.
    pub used: u32,
}

impl FrequencyTable {
    /// Create a new, zeroed frequency table.
    pub fn new() -> Self {
        Self {
            byte: [0u32; 256],
            total: 0,
            used: 0,
        }
    }

    /// Count byte frequencies in the input buffer.
    ///
    /// Populates `self.byte` with per-byte counts, and computes
    /// `self.total` and `self.used`.
    ///
    /// Uses SIMD-accelerated counting when available (AVX2 on x86_64
    /// with 4-bank histogramming, SSE2 with unrolled scalar).
    pub fn count(&mut self, input: &[u8]) {
        static DISPATCHER: OnceLock<crate::simd::Dispatcher> = OnceLock::new();
        let d = DISPATCHER.get_or_init(crate::simd::Dispatcher::new);
        self.byte = d.byte_frequencies(input);

        let mut total = 0u64;
        let mut used = 0u32;
        for &c in &self.byte {
            total += c as u64;
            used += (c > 0) as u32;
        }
        self.total = total;
        self.used = used;
    }

    /// Compute the Shannon entropy of the distribution (in bits per symbol).
    ///
    /// Returns 0.0 if the table is empty.
    pub fn entropy(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let total = self.total as f32;
        self.byte
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let prob = c as f32 / total;
                -prob * prob.log2()
            })
            .sum()
    }

    /// Get the count for a specific byte value.
    pub fn get(&self, byte: u8) -> u32 {
        self.byte[byte as usize]
    }
}

impl Default for FrequencyTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: compute a frequency table from input.
pub fn get_frequency(input: &[u8]) -> FrequencyTable {
    let mut table = FrequencyTable::new();
    table.count(input);
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_input() {
        let table = get_frequency(&[]);
        assert_eq!(table.total, 0);
        assert_eq!(table.used, 0);
        assert_eq!(table.entropy(), 0.0);
    }

    #[test]
    fn test_single_byte() {
        let table = get_frequency(&[42]);
        assert_eq!(table.total, 1);
        assert_eq!(table.used, 1);
        assert_eq!(table.get(42), 1);
        assert_eq!(table.entropy(), 0.0); // single symbol = 0 entropy
    }

    #[test]
    fn test_uniform_distribution() {
        // All 256 byte values, each appearing once
        let input: Vec<u8> = (0..=255).collect();
        let table = get_frequency(&input);
        assert_eq!(table.total, 256);
        assert_eq!(table.used, 256);
        // Entropy of uniform distribution over 256 symbols = 8.0 bits
        let entropy = table.entropy();
        assert!((entropy - 8.0).abs() < 0.01, "entropy was {}", entropy);
    }

    #[test]
    fn test_known_frequencies() {
        let input = b"aaabbc";
        let table = get_frequency(input);
        assert_eq!(table.get(b'a'), 3);
        assert_eq!(table.get(b'b'), 2);
        assert_eq!(table.get(b'c'), 1);
        assert_eq!(table.total, 6);
        assert_eq!(table.used, 3);
    }

    #[test]
    fn test_all_same_byte() {
        let input = vec![0xFFu8; 100];
        let table = get_frequency(&input);
        assert_eq!(table.total, 100);
        assert_eq!(table.used, 1);
        assert_eq!(table.get(0xFF), 100);
        assert_eq!(table.entropy(), 0.0);
    }

    #[test]
    fn test_two_equal_symbols() {
        // 50/50 split => 1 bit of entropy
        let mut input = vec![0u8; 50];
        input.extend(vec![1u8; 50]);
        let table = get_frequency(&input);
        assert_eq!(table.total, 100);
        assert_eq!(table.used, 2);
        let entropy = table.entropy();
        assert!((entropy - 1.0).abs() < 0.01, "entropy was {}", entropy);
    }
}
