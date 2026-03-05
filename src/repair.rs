/// Re-Pair Grammar Compression (`repair`) pipeline.
///
/// Iteratively replaces the most frequent bigram with a new symbol until
/// no bigram occurs more than once. Tests whether iterative GPU dispatch
/// is viable for compression (many short rounds of parallel work).
///
/// **Pipeline:**
/// ```text
/// Repeat until no bigram occurs > threshold:
///   1. Count bigram frequencies (parallel histogram)
///   2. Find most frequent bigram (parallel reduction)
///   3. Replace all non-overlapping occurrences (parallel scan + compact)
/// Output: dictionary of rules + FSE-encoded final string
/// ```
use crate::fse;
use crate::{PzError, PzResult};

/// Minimum frequency for a bigram replacement to be worthwhile.
/// Below this, the dictionary overhead exceeds the savings.
const MIN_FREQ: usize = 2;

/// Maximum number of replacement rounds (safety limit).
const MAX_ROUNDS: usize = 10_000;

/// A grammar rule: new_symbol → (left, right).
#[derive(Debug, Clone, Copy)]
struct Rule {
    left: u32,
    right: u32,
}

/// Find the most frequent bigram in the symbol sequence.
/// Returns `(left, right, frequency)` or None if no bigram occurs >= MIN_FREQ.
fn find_most_frequent_bigram(symbols: &[u32]) -> Option<(u32, u32, usize)> {
    if symbols.len() < 2 {
        return None;
    }

    // Use a hash map for bigram counting.
    let mut counts: std::collections::HashMap<(u32, u32), usize> = std::collections::HashMap::new();

    for w in symbols.windows(2) {
        *counts.entry((w[0], w[1])).or_insert(0) += 1;
    }

    let mut best: Option<(u32, u32, usize)> = None;
    for (&(left, right), &count) in &counts {
        if count >= MIN_FREQ {
            if let Some((_, _, best_count)) = best {
                if count > best_count
                    || (count == best_count
                        && (left, right) < best.map(|(l, r, _)| (l, r)).unwrap())
                {
                    best = Some((left, right, count));
                }
            } else {
                best = Some((left, right, count));
            }
        }
    }

    best
}

/// Replace all non-overlapping occurrences of bigram (left, right) with new_symbol.
/// Returns the compacted sequence.
fn replace_bigram(symbols: &[u32], left: u32, right: u32, new_symbol: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(symbols.len());
    let mut i = 0;

    while i < symbols.len() {
        if i + 1 < symbols.len() && symbols[i] == left && symbols[i + 1] == right {
            result.push(new_symbol);
            i += 2; // skip the bigram (non-overlapping)
        } else {
            result.push(symbols[i]);
            i += 1;
        }
    }

    result
}

/// Compress input using Re-Pair grammar compression.
///
/// Wire format:
/// ```text
/// [num_rules: u32 LE]
/// For each rule:
///   [left: u32 LE] [right: u32 LE]
/// [final_string_len: u32 LE]
/// [fse_compressed_len: u32 LE] [fse_data: ...]
/// ```
///
/// Note: rules are stored in order of creation. Rule i has symbol = 256 + i.
/// The final string uses u32 symbols; for FSE encoding, each u32 symbol is
/// split into 4 bytes (LE).
pub fn compress(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    // Initialize symbol sequence from input bytes.
    let mut symbols: Vec<u32> = input.iter().map(|&b| b as u32).collect();
    let mut rules: Vec<Rule> = Vec::new();
    let mut next_symbol: u32 = 256;

    // Iterative replacement.
    for _ in 0..MAX_ROUNDS {
        let Some((left, right, _freq)) = find_most_frequent_bigram(&symbols) else {
            break;
        };

        rules.push(Rule { left, right });
        symbols = replace_bigram(&symbols, left, right, next_symbol);
        next_symbol += 1;
    }

    // Serialize the final symbol sequence as bytes for FSE encoding.
    // Use varint-like encoding: symbols 0-255 stay as single bytes.
    // Symbols >= 256 are encoded as [0xFF escape] [symbol - 256 as u16 LE].
    let mut final_bytes = Vec::new();
    for &sym in &symbols {
        if sym < 255 {
            final_bytes.push(sym as u8);
        } else {
            final_bytes.push(0xFF);
            let idx = (sym.wrapping_sub(255)) as u16;
            final_bytes.extend_from_slice(&idx.to_le_bytes());
        }
    }

    let fse_data = fse::encode(&final_bytes);

    // Assemble output.
    let mut output = Vec::new();
    output.extend_from_slice(&(rules.len() as u32).to_le_bytes());
    for rule in &rules {
        output.extend_from_slice(&rule.left.to_le_bytes());
        output.extend_from_slice(&rule.right.to_le_bytes());
    }
    output.extend_from_slice(&(final_bytes.len() as u32).to_le_bytes());
    output.extend_from_slice(&(fse_data.len() as u32).to_le_bytes());
    output.extend_from_slice(&fse_data);

    Ok(output)
}

/// Decompress Re-Pair grammar data.
pub fn decompress(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 4 {
        return Err(PzError::InvalidInput);
    }

    let num_rules = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;

    let mut pos = 4;
    let rules_bytes = num_rules * 8; // 4 bytes left + 4 bytes right
    if pos + rules_bytes + 8 > payload.len() {
        return Err(PzError::InvalidInput);
    }

    let mut rules: Vec<Rule> = Vec::with_capacity(num_rules);
    for _ in 0..num_rules {
        let left = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]);
        pos += 4;
        let right = u32::from_le_bytes([
            payload[pos],
            payload[pos + 1],
            payload[pos + 2],
            payload[pos + 3],
        ]);
        pos += 4;
        rules.push(Rule { left, right });
    }

    let final_bytes_len = u32::from_le_bytes([
        payload[pos],
        payload[pos + 1],
        payload[pos + 2],
        payload[pos + 3],
    ]) as usize;
    pos += 4;

    let fse_len = u32::from_le_bytes([
        payload[pos],
        payload[pos + 1],
        payload[pos + 2],
        payload[pos + 3],
    ]) as usize;
    pos += 4;

    if pos + fse_len > payload.len() {
        return Err(PzError::InvalidInput);
    }

    let final_bytes = fse::decode(&payload[pos..pos + fse_len], final_bytes_len)?;

    // Parse the varint-encoded symbol sequence.
    let mut symbols: Vec<u32> = Vec::new();
    let mut i = 0;
    while i < final_bytes.len() {
        if final_bytes[i] == 0xFF {
            if i + 2 >= final_bytes.len() {
                return Err(PzError::InvalidInput);
            }
            let idx = u16::from_le_bytes([final_bytes[i + 1], final_bytes[i + 2]]) as u32;
            symbols.push(idx + 255);
            i += 3;
        } else {
            symbols.push(final_bytes[i] as u32);
            i += 1;
        }
    }

    // Expand grammar rules in reverse order.
    // Rule i: symbol 256+i → (left, right)
    for (rule_idx, rule) in rules.iter().enumerate().rev() {
        let sym = 256 + rule_idx as u32;
        let mut expanded = Vec::with_capacity(symbols.len());
        for &s in &symbols {
            if s == sym {
                expanded.push(rule.left);
                expanded.push(rule.right);
            } else {
                expanded.push(s);
            }
        }
        symbols = expanded;
    }

    // Convert back to bytes.
    let mut output = Vec::with_capacity(orig_len);
    for &sym in &symbols {
        if sym > 255 {
            return Err(PzError::InvalidInput);
        }
        output.push(sym as u8);
    }

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

/// Diagnostic: return per-round statistics.
///
/// Returns a vector of `(round, frequency, symbols_remaining, dictionary_size)`.
pub fn compression_stats(input: &[u8]) -> Vec<(usize, usize, usize, usize)> {
    let mut symbols: Vec<u32> = input.iter().map(|&b| b as u32).collect();
    let mut next_symbol: u32 = 256;
    let mut stats = Vec::new();

    for round in 0..MAX_ROUNDS {
        let Some((left, right, freq)) = find_most_frequent_bigram(&symbols) else {
            break;
        };

        symbols = replace_bigram(&symbols, left, right, next_symbol);
        next_symbol += 1;

        let num_rules = round + 1;
        let dict_size = num_rules * 8; // 4 bytes left + 4 bytes right per rule
        stats.push((round, freq, symbols.len(), dict_size));
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let input = b"abcabcabcabcabc this repeats abcabcabcabc";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_no_repeats() {
        let input: Vec<u8> = (0..200).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_highly_repetitive() {
        let input = b"ababababababababababababababababab";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_single_byte_repeated() {
        let input = vec![b'a'; 100];
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_all_bytes() {
        let input: Vec<u8> = (0..=255).cycle().take(512).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn stats_track_rounds() {
        let input = b"abcabcabcabcabcabcabcabc";
        let stats = compression_stats(input);
        assert!(!stats.is_empty());
        // Each round should reduce symbol count.
        for w in stats.windows(2) {
            assert!(w[1].2 <= w[0].2); // symbols_remaining should decrease or stay same
        }
    }

    #[test]
    fn replace_bigram_non_overlapping() {
        let symbols = vec![1, 2, 1, 2, 1, 2];
        let result = replace_bigram(&symbols, 1, 2, 256);
        assert_eq!(result, vec![256, 256, 256]);
    }
}
