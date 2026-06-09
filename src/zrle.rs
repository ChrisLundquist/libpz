//! Zero-run-length coding (bzip2-style RUNA/RUNB) for BWT+MTF output.
//!
//! After BWT + MTF, the data is dominated by long runs of zero (every repeated
//! input symbol becomes a 0 after MTF). The legacy [`crate::rle`] codes runs of
//! *any* byte as "4 literals + a count byte"; that count byte has a fairly flat
//! distribution and the 4 redundant literals waste entropy.
//!
//! RUNA/RUNB instead codes only the zero-runs, writing each run length in
//! *bijective base 2* using two dedicated symbols (RUNA = digit 1, RUNB = digit 2).
//! Those two symbols flow through the same FSE model as the data symbols, so the
//! entropy coder assigns them optimal (short) codes — zero-runs dominate, so RUNA
//! ends up very cheap. This is the single biggest ratio lever for the BWT pipeline.
//!
//! **Alphabet fit.** Removing 0 and adding {RUNA, RUNB} grows the alphabet by one.
//! To stay within FSE's 256-symbol (u8) range we shift every non-zero MTF value
//! `v` up by one (`v -> v + 1`), with RUNA = 0 and RUNB = 1 taking the freed low
//! slots. That overflows only when a value of 255 is present (255 -> 256), which
//! requires a block using all 256 byte values at maximal MTF distance — rare, and
//! a poor fit for BWT anyway. In that case [`encode`] returns `None` and the caller
//! falls back to [`crate::rle`].
//!
//! **Framing.** No in-band EOB symbol: the caller stores the decoded length, and a
//! trailing zero-run (RUNA/RUNB to end-of-input) decodes naturally.

use crate::{PzError, PzResult};

/// Bijective base-2 digit 1 — also reuses the freed MTF "0" slot.
pub const RUNA: u8 = 0;
/// Bijective base-2 digit 2.
pub const RUNB: u8 = 1;

/// Encode MTF output with RUNA/RUNB zero-run coding.
///
/// Returns `Some(encoded)` on success, or `None` if the input contains a byte
/// value of 255 (which cannot be shifted up by one without overflow); the caller
/// should fall back to [`crate::rle::encode`] for that block.
pub fn encode(input: &[u8]) -> Option<Vec<u8>> {
    if input.is_empty() {
        return Some(Vec::new());
    }
    // Reject blocks that can't take the +1 shift. Checking up front keeps the
    // hot loop branch-free of the overflow guard.
    if input.contains(&255) {
        return None;
    }

    let mut output = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        if input[i] == 0 {
            // Measure the maximal zero-run and emit it in bijective base 2.
            let start = i;
            while i < input.len() && input[i] == 0 {
                i += 1;
            }
            let mut run = i - start;
            // LSB-first: subtract 1, low bit selects RUNA(0)/RUNB(1), shift down.
            while run > 0 {
                run -= 1;
                output.push(if run & 1 == 0 { RUNA } else { RUNB });
                run >>= 1;
            }
        } else {
            // Non-zero MTF value: shift up into [2, 255].
            output.push(input[i] + 1);
            i += 1;
        }
    }
    Some(output)
}

/// Decode RUNA/RUNB-coded data back to the original MTF output.
///
/// `out_len` is the expected decoded length (the BWT/MTF stream length); it is
/// used to size the buffer and validate the result.
pub fn decode(input: &[u8], out_len: usize) -> PzResult<Vec<u8>> {
    let mut output = Vec::with_capacity(out_len);
    let mut i = 0;
    while i < input.len() {
        if input[i] <= RUNB {
            // Accumulate a bijective base-2 run: RUNA=+1×w, RUNB=+2×w, w doubles.
            let mut run: usize = 0;
            let mut weight: usize = 1;
            while i < input.len() && input[i] <= RUNB {
                run += if input[i] == RUNA { weight } else { 2 * weight };
                weight <<= 1;
                i += 1;
                if run > out_len {
                    return Err(PzError::InvalidInput);
                }
            }
            output.resize(output.len() + run, 0);
        } else {
            // Shifted non-zero value: undo the +1.
            output.push(input[i] - 1);
            i += 1;
        }
        if output.len() > out_len {
            return Err(PzError::InvalidInput);
        }
    }
    if output.len() != out_len {
        return Err(PzError::InvalidInput);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(mtf: &[u8]) {
        let enc = encode(mtf).expect("no 255 in test inputs");
        let dec = decode(&enc, mtf.len()).unwrap();
        assert_eq!(dec, mtf, "roundtrip mismatch for {mtf:?}");
    }

    #[test]
    fn test_empty() {
        assert_eq!(encode(&[]), Some(Vec::new()));
        assert_eq!(decode(&[], 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn test_no_zeros() {
        roundtrip(&[1, 2, 3, 7, 42, 200, 254]);
    }

    #[test]
    fn test_all_zeros_various_lengths() {
        for n in 1..=300usize {
            roundtrip(&vec![0u8; n]);
        }
    }

    #[test]
    fn test_mixed_runs() {
        roundtrip(&[0, 0, 0, 5, 0, 1, 1, 0, 0, 0, 0, 0, 9, 0]);
        roundtrip(&[5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]); // long interior run
        roundtrip(&[7, 0, 0, 0]); // trailing run (no EOB)
        roundtrip(&[0, 0, 0, 7]); // leading run
    }

    #[test]
    fn test_bijective_base2_values() {
        // Run of length N -> the documented bijective base-2 string.
        assert_eq!(encode(&[0]).unwrap(), vec![RUNA]); // 1 = "A"
        assert_eq!(encode(&[0, 0]).unwrap(), vec![RUNB]); // 2 = "B"
        assert_eq!(encode(&[0, 0, 0]).unwrap(), vec![RUNA, RUNA]); // 3 = "AA"
        assert_eq!(encode(&[0; 4]).unwrap(), vec![RUNB, RUNA]); // 4 = "BA"
        assert_eq!(encode(&[0; 5]).unwrap(), vec![RUNA, RUNB]); // 5 = "AB"
        assert_eq!(encode(&[0; 6]).unwrap(), vec![RUNB, RUNB]); // 6 = "BB"
        assert_eq!(encode(&[0; 7]).unwrap(), vec![RUNA, RUNA, RUNA]); // 7 = "AAA"
    }

    #[test]
    fn test_value_255_falls_back() {
        assert_eq!(encode(&[1, 2, 255, 3]), None);
        assert_eq!(encode(&[0, 0, 255]), None);
        // 254 is the highest value that still fits after the shift.
        roundtrip(&[254, 0, 0, 254]);
    }

    #[test]
    fn test_decode_rejects_overlong_run() {
        // A long RUNA/RUNB sequence that decodes past out_len must error, not OOM.
        let bomb = vec![RUNB; 40]; // ~2^40 zeros
        assert_eq!(decode(&bomb, 100), Err(PzError::InvalidInput));
    }

    #[test]
    fn test_decode_wrong_len_errors() {
        let enc = encode(&[0, 0, 5]).unwrap();
        assert_eq!(decode(&enc, 99), Err(PzError::InvalidInput));
    }
}
