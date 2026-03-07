/// Parallel-Parse LZ (`parlz`) pipeline.
///
/// Measures the exact compression ratio cost of removing serial parsing from LZ.
/// All positions find matches independently (parallel), then a conflict
/// resolution pass (forward max-propagation scan) suppresses overlapping matches.
///
/// The key output is the **ratio gap** between parallel parsing and serial
/// greedy parsing using the same match data.
///
/// **Pipeline:**
/// ```text
/// Input
///   → Match finding at every position (reuse existing hash-chain finder)
///   → Parallel match selection (each position picks best match)
///   → Conflict resolution via forward max-propagation scan
///   → Encode literals and match references with FSE
/// Output
/// ```
use crate::fse;
use crate::lz77;
use crate::{PzError, PzResult};

/// Minimum match length (must match lz77::MIN_MATCH).
const MIN_MATCH: usize = lz77::MIN_MATCH as usize;

/// A parsed LZ token.
#[derive(Debug, Clone, Copy)]
enum LzToken {
    Literal(u8),
    Match { offset: u16, length: u16 },
}

/// Find best match at every position using hash chains.
///
/// Delegates to `lz77::HashChainFinder::find_all()` to avoid duplicating
/// the hash-chain match finder logic.
///
/// Returns a vector where `matches[i]` is `Some((offset, length))` if a match
/// was found at position i, or `None` otherwise.
pub(crate) fn find_all_matches(input: &[u8]) -> Vec<Option<(u16, u16)>> {
    let n = input.len();
    if n < MIN_MATCH {
        return vec![None; n];
    }

    let mut finder = lz77::HashChainFinder::with_max_match_len(lz77::DEFAULT_MAX_MATCH);
    finder.find_all(input)
}

/// Parallel conflict resolution via forward max-propagation scan.
///
/// Rule: longer match wins; on ties, earlier position wins.
/// Uses a coverage array: `coverage[p] = p + match_length[p]` for match positions.
/// After prefix-max scan, position p is covered (suppressed) if `coverage[p-1] > p`.
///
/// Returns classification: true = match start, false = literal or covered.
pub(crate) fn parallel_resolve(matches: &[Option<(u16, u16)>]) -> Vec<bool> {
    let n = matches.len();
    let mut is_match_start = vec![false; n];

    // Build coverage array.
    let mut coverage = vec![0usize; n];
    for (i, m) in matches.iter().enumerate() {
        if let Some((_, length)) = m {
            coverage[i] = i + *length as usize;
        } else {
            coverage[i] = i;
        }
    }

    // Forward max-propagation scan (prefix-max).
    for i in 1..n {
        coverage[i] = coverage[i].max(coverage[i - 1]);
    }

    // Classify positions.
    // A position is a match start if:
    // 1. It has a match, AND
    // 2. It's not covered by an earlier match (i.e., coverage[i-1] <= i for i > 0)
    for i in 0..n {
        if matches[i].is_some() {
            let not_covered = i == 0 || coverage[i - 1] <= i;
            is_match_start[i] = not_covered;
        }
    }

    is_match_start
}

/// Greedy serial parse (baseline for comparison).
///
/// Standard left-to-right greedy: take first match found, advance by match length.
fn greedy_parse(input: &[u8], matches: &[Option<(u16, u16)>]) -> Vec<LzToken> {
    let n = input.len();
    let mut tokens = Vec::new();
    let mut pos = 0;

    while pos < n {
        if let Some((offset, length)) = matches[pos] {
            tokens.push(LzToken::Match { offset, length });
            pos += length as usize;
        } else {
            tokens.push(LzToken::Literal(input[pos]));
            pos += 1;
        }
    }

    tokens
}

/// Parallel parse: use conflict resolution results to build token stream.
fn parallel_parse(
    input: &[u8],
    matches: &[Option<(u16, u16)>],
    is_match_start: &[bool],
) -> Vec<LzToken> {
    let n = input.len();
    let mut tokens = Vec::new();
    let mut pos = 0;

    while pos < n {
        if is_match_start[pos] {
            let (offset, length) = matches[pos].unwrap();
            tokens.push(LzToken::Match { offset, length });
            pos += length as usize;
        } else {
            // Check if we're inside a match (covered by an earlier match start).
            // If so, skip — but we handle this via the match-start advancing pos.
            // If we reach here, it's a literal.
            tokens.push(LzToken::Literal(input[pos]));
            pos += 1;
        }
    }

    tokens
}

/// Encode tokens into wire format with FSE compression.
///
/// Wire format:
/// ```text
/// [num_tokens: u32 LE] [num_literals: u32 LE] [flag_bytes: u32 LE]
/// [flags: flag_bytes bytes]
/// [fse_literals_len: u32 LE] [fse_literals]
/// [fse_offsets_len: u32 LE] [fse_offsets]
/// [fse_lengths_len: u32 LE] [fse_lengths]
/// ```
fn encode_tokens(tokens: &[LzToken]) -> Vec<u8> {
    let num_tokens = tokens.len();
    let flag_bytes = num_tokens.div_ceil(8);
    let mut flags = vec![0u8; flag_bytes];
    let mut literals = Vec::new();
    let mut offsets_raw = Vec::new();
    let mut lengths_raw = Vec::new();

    for (i, token) in tokens.iter().enumerate() {
        match token {
            LzToken::Literal(b) => {
                flags[i / 8] |= 1 << (7 - (i % 8));
                literals.push(*b);
            }
            LzToken::Match { offset, length } => {
                offsets_raw.extend_from_slice(&offset.to_le_bytes());
                lengths_raw.extend_from_slice(&length.to_le_bytes());
            }
        }
    }

    let fse_literals = fse::encode(&literals);
    let fse_offsets = fse::encode(&offsets_raw);
    let fse_lengths = fse::encode(&lengths_raw);

    let mut output = Vec::new();
    output.extend_from_slice(&(num_tokens as u32).to_le_bytes());
    output.extend_from_slice(&(literals.len() as u32).to_le_bytes());
    output.extend_from_slice(&(flag_bytes as u32).to_le_bytes());
    output.extend_from_slice(&flags);

    output.extend_from_slice(&(fse_literals.len() as u32).to_le_bytes());
    output.extend_from_slice(&fse_literals);

    output.extend_from_slice(&(fse_offsets.len() as u32).to_le_bytes());
    output.extend_from_slice(&fse_offsets);

    output.extend_from_slice(&(fse_lengths.len() as u32).to_le_bytes());
    output.extend_from_slice(&fse_lengths);

    output
}

/// Compress input using parallel-parse LZ.
pub fn compress(input: &[u8]) -> PzResult<Vec<u8>> {
    if input.is_empty() {
        return Err(PzError::InvalidInput);
    }

    let matches = find_all_matches(input);
    let is_match_start = parallel_resolve(&matches);
    let tokens = parallel_parse(input, &matches, &is_match_start);
    Ok(encode_tokens(&tokens))
}

/// Decompress parallel-parse LZ data.
pub fn decompress(payload: &[u8], orig_len: usize) -> PzResult<Vec<u8>> {
    if payload.len() < 12 {
        return Err(PzError::InvalidInput);
    }

    let num_tokens = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let num_literals =
        u32::from_le_bytes([payload[4], payload[5], payload[6], payload[7]]) as usize;
    let flag_bytes =
        u32::from_le_bytes([payload[8], payload[9], payload[10], payload[11]]) as usize;

    let mut pos = 12;
    if pos + flag_bytes > payload.len() {
        return Err(PzError::InvalidInput);
    }
    let flags = &payload[pos..pos + flag_bytes];
    pos += flag_bytes;

    let literals = read_fse_stream(payload, &mut pos, num_literals)?;

    let num_matches = num_tokens - num_literals;
    let offsets_raw = read_fse_stream(payload, &mut pos, num_matches * 2)?;
    let lengths_raw = read_fse_stream(payload, &mut pos, num_matches * 2)?;

    // Reconstruct output.
    let mut output = Vec::with_capacity(orig_len);
    let mut lit_idx = 0;
    let mut match_idx = 0;

    for i in 0..num_tokens {
        let is_literal = flags[i / 8] & (1 << (7 - (i % 8))) != 0;
        if is_literal {
            if lit_idx >= literals.len() {
                return Err(PzError::InvalidInput);
            }
            output.push(literals[lit_idx]);
            lit_idx += 1;
        } else {
            let off_pos = match_idx * 2;
            if off_pos + 2 > offsets_raw.len() || off_pos + 2 > lengths_raw.len() {
                return Err(PzError::InvalidInput);
            }
            let offset =
                u16::from_le_bytes([offsets_raw[off_pos], offsets_raw[off_pos + 1]]) as usize;
            let length =
                u16::from_le_bytes([lengths_raw[off_pos], lengths_raw[off_pos + 1]]) as usize;

            if offset == 0 || offset > output.len() {
                return Err(PzError::InvalidInput);
            }

            for _ in 0..length {
                let src = output.len() - offset;
                let b = output[src];
                output.push(b);
            }

            match_idx += 1;
        }
    }

    if output.len() != orig_len {
        return Err(PzError::InvalidInput);
    }

    Ok(output)
}

/// Read an FSE-encoded stream from payload.
fn read_fse_stream(payload: &[u8], pos: &mut usize, expected_len: usize) -> PzResult<Vec<u8>> {
    if *pos + 4 > payload.len() {
        return Err(PzError::InvalidInput);
    }
    let fse_len = u32::from_le_bytes([
        payload[*pos],
        payload[*pos + 1],
        payload[*pos + 2],
        payload[*pos + 3],
    ]) as usize;
    *pos += 4;

    if *pos + fse_len > payload.len() {
        return Err(PzError::InvalidInput);
    }
    let data = fse::decode(&payload[*pos..*pos + fse_len], expected_len)?;
    *pos += fse_len;
    Ok(data)
}

/// Parallel parse + encode in one step (avoids exposing LzToken).
pub(crate) fn parallel_parse_and_encode(
    input: &[u8],
    matches: &[Option<(u16, u16)>],
    is_match_start: &[bool],
) -> Vec<u8> {
    let tokens = parallel_parse(input, matches, is_match_start);
    encode_tokens(&tokens)
}

/// Diagnostic: compute ratio gap between parallel and greedy parsing.
///
/// Returns `(parallel_compressed_size, greedy_compressed_size, ratio_gap_percent)`.
pub fn ratio_gap(input: &[u8]) -> Option<(usize, usize, f64)> {
    if input.is_empty() {
        return None;
    }

    let matches = find_all_matches(input);

    // Parallel parse
    let is_match_start = parallel_resolve(&matches);
    let par_tokens = parallel_parse(input, &matches, &is_match_start);
    let par_encoded = encode_tokens(&par_tokens);

    // Greedy parse (same match data)
    let greedy_tokens = greedy_parse(input, &matches);
    let greedy_encoded = encode_tokens(&greedy_tokens);

    let gap = (par_encoded.len() as f64 - greedy_encoded.len() as f64)
        / greedy_encoded.len() as f64
        * 100.0;

    Some((par_encoded.len(), greedy_encoded.len(), gap))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let input =
            b"abcabcabcabcabcabc this is a test with some repeated patterns repeated patterns";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn roundtrip_no_matches() {
        let input: Vec<u8> = (0..=255).collect();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_highly_repetitive() {
        let input = b"aaaaaaaaaaaabbbbbbbbbbbbcccccccccccc".repeat(10);
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(decompressed, input);
    }

    #[test]
    fn roundtrip_small() {
        let input = b"ab";
        let compressed = compress(input).unwrap();
        let decompressed = decompress(&compressed, input.len()).unwrap();
        assert_eq!(&decompressed, input);
    }

    #[test]
    fn parallel_resolve_no_overlap() {
        // Two non-overlapping matches.
        let matches = vec![
            Some((5u16, 3u16)),
            None,
            None,
            None,
            Some((5, 3)),
            None,
            None,
        ];
        let is_start = parallel_resolve(&matches);
        assert!(is_start[0]);
        assert!(is_start[4]);
    }

    #[test]
    fn parallel_resolve_overlap() {
        // Match at pos 0 (length 5) covers pos 2's match.
        let matches = vec![Some((10u16, 5u16)), None, Some((8, 3)), None, None, None];
        let is_start = parallel_resolve(&matches);
        assert!(is_start[0]);
        assert!(!is_start[2]); // covered by pos 0
    }

    #[test]
    fn ratio_gap_computable() {
        let input = b"the quick brown fox jumps over the lazy dog the quick brown fox jumps again";
        let result = ratio_gap(input);
        assert!(result.is_some());
        let (par, greedy, gap) = result.unwrap();
        assert!(par > 0);
        assert!(greedy > 0);
        // Gap should be finite.
        assert!(gap.is_finite());
    }
}
