//! Matrix RREF Journaling — experimental compression prototype.
//!
//! Folds input bytes into an N×N matrix over GF(256), performs Gaussian
//! elimination toward RREF, and journals the row operations. The hypothesis:
//! the operation journal + residual may be smaller than the original data,
//! especially on structured inputs.
//!
//! GF(256) arithmetic (the same field used by AES and Reed-Solomon) avoids
//! integer overflow — every nonzero element has a multiplicative inverse.

// --- GF(256) arithmetic ---
//
// Field: GF(2^8) with irreducible polynomial x^8 + x^4 + x^3 + x + 1 (0x11B).
// We use log/exp tables for fast multiply and divide.

const GF_POLY: u16 = 0x11B;

/// Precomputed exp table: EXP[i] = g^i where g=3 is a generator of GF(256)*.
/// EXP has 512 entries to avoid modular reduction in multiply.
const fn build_gf_tables() -> ([u8; 512], [u8; 256]) {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];
    let mut val: u16 = 1;
    let mut i = 0;
    while i < 255 {
        exp[i] = val as u8;
        exp[i + 255] = val as u8;
        log[val as usize] = i as u8;
        // multiply by generator 3
        val = (val << 1) ^ val; // val * 3 = val * 2 + val
        if val >= 256 {
            val ^= GF_POLY;
        }
        i += 1;
    }
    // exp[255] = 1 (wraps), log[0] is unused (0 has no log)
    exp[255] = 1;
    (exp, log)
}

const GF_TABLES: ([u8; 512], [u8; 256]) = build_gf_tables();

#[inline(always)]
fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let (exp, log) = &GF_TABLES;
    exp[log[a as usize] as usize + log[b as usize] as usize]
}

#[inline(always)]
fn gf_div(a: u8, b: u8) -> u8 {
    debug_assert!(b != 0, "GF(256) division by zero");
    if a == 0 {
        return 0;
    }
    let (exp, log) = &GF_TABLES;
    // a / b = g^(log(a) - log(b)) ; use +255 to keep positive
    exp[log[a as usize] as usize + 255 - log[b as usize] as usize]
}

#[inline(always)]
fn gf_add(a: u8, b: u8) -> u8 {
    a ^ b // addition in GF(2^k) is XOR
}

// In GF(2^k), addition == subtraction (both are XOR)
#[inline(always)]
fn gf_sub(a: u8, b: u8) -> u8 {
    a ^ b
}

// --- Row Operations and Journal ---

/// A single row operation in the Gaussian elimination journal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowOp {
    /// Swap rows i and j.
    Swap(u8, u8),
    /// Scale row i by factor (multiply every element).
    Scale(u8, u8),
    /// row[dst] ^= gf_mul(factor, row[src]) for each column.
    AddScaled(u8, u8, u8), // dst, src, factor
}

impl RowOp {
    /// Encode to bytes: [opcode, arg1, arg2, (arg3)]
    fn to_bytes(self) -> Vec<u8> {
        match self {
            RowOp::Swap(a, b) => vec![0x00, a, b],
            RowOp::Scale(r, f) => vec![0x01, r, f],
            RowOp::AddScaled(d, s, f) => vec![0x02, d, s, f],
        }
    }

    fn from_bytes(data: &[u8]) -> Option<(Self, usize)> {
        if data.is_empty() {
            return None;
        }
        match data[0] {
            0x00 if data.len() >= 3 => Some((RowOp::Swap(data[1], data[2]), 3)),
            0x01 if data.len() >= 3 => Some((RowOp::Scale(data[1], data[2]), 3)),
            0x02 if data.len() >= 4 => Some((RowOp::AddScaled(data[1], data[2], data[3]), 4)),
            _ => None,
        }
    }
}

// --- Matrix Operations ---

/// N×N matrix over GF(256), row-major.
struct GfMatrix {
    n: usize,
    data: Vec<u8>,
}

impl GfMatrix {
    fn new(n: usize) -> Self {
        Self {
            n,
            data: vec![0; n * n],
        }
    }

    fn from_bytes(input: &[u8], n: usize) -> Self {
        let mut m = Self::new(n);
        let copy_len = input.len().min(n * n);
        m.data[..copy_len].copy_from_slice(&input[..copy_len]);
        m
    }

    #[inline(always)]
    fn get(&self, row: usize, col: usize) -> u8 {
        self.data[row * self.n + col]
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn set(&mut self, row: usize, col: usize, val: u8) {
        self.data[row * self.n + col] = val;
    }

    fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let (start_a, start_b) = (a * self.n, b * self.n);
        for c in 0..self.n {
            self.data.swap(start_a + c, start_b + c);
        }
    }

    fn scale_row(&mut self, row: usize, factor: u8) {
        let start = row * self.n;
        for c in 0..self.n {
            self.data[start + c] = gf_mul(self.data[start + c], factor);
        }
    }

    /// row[dst] ^= gf_mul(factor, row[src])
    fn add_scaled_row(&mut self, dst: usize, src: usize, factor: u8) {
        let n = self.n;
        for c in 0..n {
            let src_val = self.data[src * n + c];
            self.data[dst * n + c] = gf_sub(self.data[dst * n + c], gf_mul(factor, src_val));
        }
    }

    /// Apply a row operation.
    fn apply(&mut self, op: &RowOp) {
        match *op {
            RowOp::Swap(a, b) => self.swap_rows(a as usize, b as usize),
            RowOp::Scale(r, f) => self.scale_row(r as usize, f),
            RowOp::AddScaled(d, s, f) => self.add_scaled_row(d as usize, s as usize, f),
        }
    }

    /// Apply the inverse of a row operation.
    fn apply_inverse(&mut self, op: &RowOp) {
        match *op {
            RowOp::Swap(a, b) => self.swap_rows(a as usize, b as usize), // swap is self-inverse
            RowOp::Scale(r, f) => self.scale_row(r as usize, gf_div(1, f)), // scale by 1/f
            RowOp::AddScaled(d, s, f) => self.add_scaled_row(d as usize, s as usize, f), // XOR is self-inverse
        }
    }

    fn to_bytes(&self) -> &[u8] {
        &self.data
    }
}

/// Perform Gaussian elimination toward RREF, returning the journal of operations.
fn gaussian_eliminate(matrix: &mut GfMatrix) -> Vec<RowOp> {
    let n = matrix.n;
    let mut journal = Vec::new();
    let mut pivot_row = 0;

    for col in 0..n {
        // Find pivot in column
        let mut found = None;
        for row in pivot_row..n {
            if matrix.get(row, col) != 0 {
                found = Some(row);
                break;
            }
        }
        let Some(prow) = found else {
            continue; // zero column, skip
        };

        // Swap to pivot position
        if prow != pivot_row {
            let op = RowOp::Swap(pivot_row as u8, prow as u8);
            matrix.apply(&op);
            journal.push(op);
        }

        // Scale pivot to 1
        let pivot_val = matrix.get(pivot_row, col);
        if pivot_val != 1 {
            let inv = gf_div(1, pivot_val);
            let op = RowOp::Scale(pivot_row as u8, inv);
            matrix.apply(&op);
            journal.push(op);
        }

        // Eliminate all other rows in this column
        for row in 0..n {
            if row == pivot_row {
                continue;
            }
            let val = matrix.get(row, col);
            if val != 0 {
                let op = RowOp::AddScaled(row as u8, pivot_row as u8, val);
                matrix.apply(&op);
                journal.push(op);
            }
        }

        pivot_row += 1;
    }

    journal
}

/// Result of matrix RREF compression analysis for one block.
#[derive(Debug)]
#[allow(dead_code)]
pub struct MatrixBlockResult {
    /// Size of the original input block (n*n bytes, possibly padded).
    pub original_size: usize,
    /// Number of journal operations.
    pub num_ops: usize,
    /// Raw journal size in bytes (before any entropy coding).
    pub journal_bytes: usize,
    /// Size of the residual matrix (non-identity portion).
    pub residual_bytes: usize,
    /// Size of journal after delta encoding.
    pub delta_journal_bytes: usize,
    /// Number of non-zero entries in the RREF matrix.
    pub rref_nonzeros: usize,
    /// Rank of the matrix (number of pivot rows found).
    pub rank: usize,
}

/// Encode: fold input into N×N matrix, RREF, return journal + residual.
///
/// Returns (journal_bytes, residual, rref_matrix, analysis).
fn encode_block(
    input: &[u8],
    fold_size: usize,
) -> (Vec<u8>, Vec<u8>, Vec<RowOp>, MatrixBlockResult) {
    let n = fold_size;
    let block_bytes = n * n;
    let original_size = input.len().min(block_bytes);

    let mut matrix = GfMatrix::from_bytes(input, n);
    let journal = gaussian_eliminate(&mut matrix);

    // Serialize journal
    let mut journal_bytes_raw = Vec::new();
    for op in &journal {
        journal_bytes_raw.extend_from_slice(&op.to_bytes());
    }

    // Delta-encode the journal stream: each byte becomes (byte - prev_byte)
    let delta_journal = delta_encode(&journal_bytes_raw);

    // Residual: the RREF matrix itself (mostly identity + zeros if full rank)
    let residual = matrix.to_bytes().to_vec();

    // Count non-zeros in RREF
    let rref_nonzeros = residual.iter().filter(|&&b| b != 0).count();

    // Count rank (number of leading 1s on diagonal)
    let mut rank = 0;
    for i in 0..n {
        if matrix.get(i, i) == 1 {
            rank += 1;
        }
    }

    let analysis = MatrixBlockResult {
        original_size,
        num_ops: journal.len(),
        journal_bytes: journal_bytes_raw.len(),
        residual_bytes: residual.len(),
        delta_journal_bytes: delta_journal.len(),
        rref_nonzeros,
        rank,
    };

    (journal_bytes_raw, residual, journal, analysis)
}

/// Decode: given a journal and RREF matrix, reconstruct the original block.
fn decode_block(journal: &[RowOp], rref_data: &[u8], fold_size: usize) -> Vec<u8> {
    let mut matrix = GfMatrix::from_bytes(rref_data, fold_size);

    // Replay journal in reverse, applying inverse operations
    for op in journal.iter().rev() {
        matrix.apply_inverse(op);
    }

    matrix.to_bytes().to_vec()
}

/// Delta encode a byte stream: output[i] = input[i] ^ input[i-1].
fn delta_encode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(input.len());
    out.push(input[0]);
    for i in 1..input.len() {
        out.push(input[i] ^ input[i - 1]);
    }
    out
}

/// Delta decode a byte stream (inverse of delta_encode).
fn delta_decode(input: &[u8]) -> Vec<u8> {
    if input.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(input.len());
    out.push(input[0]);
    for i in 1..input.len() {
        out.push(input[i] ^ out[i - 1]);
    }
    out
}

/// Shannon entropy in bits per symbol for a byte stream.
fn entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let total = data.len() as f64;
    let mut h = 0.0;
    for &c in &counts {
        if c > 0 {
            let p = c as f64 / total;
            h -= p * p.log2();
        }
    }
    h
}

/// Full analysis: try multiple fold sizes on input, report results.
pub fn analyze(input: &[u8]) -> Vec<(usize, MatrixBlockResult)> {
    let fold_sizes = [4, 8, 16, 32];
    let mut results = Vec::new();

    for &n in &fold_sizes {
        if input.len() < n * n {
            continue;
        }

        let (journal_raw, _residual, _journal_ops, analysis) = encode_block(&input[..n * n], n);

        // Also compute entropy of raw vs delta journal
        let _ = entropy(&journal_raw);
        let delta = delta_encode(&journal_raw);
        let _ = entropy(&delta);

        results.push((n, analysis));
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf256_basic() {
        // Identity: a * 1 = a
        for a in 0..=255u8 {
            assert_eq!(gf_mul(a, 1), a);
        }
        // Zero: a * 0 = 0
        for a in 0..=255u8 {
            assert_eq!(gf_mul(a, 0), 0);
        }
        // Inverse: a * (1/a) = 1 for a != 0
        for a in 1..=255u8 {
            let inv = gf_div(1, a);
            assert_eq!(gf_mul(a, inv), 1, "failed for a={a}");
        }
        // a / a = 1
        for a in 1..=255u8 {
            assert_eq!(gf_div(a, a), 1);
        }
    }

    #[test]
    fn test_gf256_add_sub() {
        // In GF(2^k), add == sub == XOR
        for a in 0..=255u8 {
            assert_eq!(gf_add(a, a), 0);
            assert_eq!(gf_sub(a, a), 0);
        }
    }

    #[test]
    fn test_delta_round_trip() {
        let data = vec![10, 20, 15, 30, 25, 5];
        let encoded = delta_encode(&data);
        let decoded = delta_decode(&encoded);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_delta_empty() {
        assert!(delta_encode(&[]).is_empty());
        assert!(delta_decode(&[]).is_empty());
    }

    #[test]
    fn test_matrix_round_trip_identity() {
        // An identity matrix should produce zero journal entries
        let n = 4;
        let mut input = vec![0u8; n * n];
        for i in 0..n {
            input[i * n + i] = 1;
        }
        let (_, _, _journal_ops, analysis) = encode_block(&input, n);
        assert_eq!(analysis.rank, n);
        assert_eq!(analysis.num_ops, 0); // identity needs no ops

        // RREF of identity is identity, so replaying empty journal gives identity
        // Test round-trip properly using the actual RREF
        let mut matrix = GfMatrix::from_bytes(&input, n);
        let ops = gaussian_eliminate(&mut matrix);
        let rref_data = matrix.to_bytes().to_vec();
        let recovered = decode_block(&ops, &rref_data, n);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_matrix_round_trip_random() {
        // Test that encode→decode is lossless
        let n = 8;
        // Deterministic "random" data
        let mut input = vec![0u8; n * n];
        let mut val: u8 = 42;
        for byte in input.iter_mut() {
            *byte = val;
            val = val.wrapping_mul(137).wrapping_add(73);
        }

        let mut matrix = GfMatrix::from_bytes(&input, n);
        let ops = gaussian_eliminate(&mut matrix);
        let rref_data = matrix.to_bytes().to_vec();
        let recovered = decode_block(&ops, &rref_data, n);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_matrix_round_trip_zeros() {
        // All-zero matrix: rank 0, identity journal should be empty
        let n = 4;
        let input = vec![0u8; n * n];
        let mut matrix = GfMatrix::from_bytes(&input, n);
        let ops = gaussian_eliminate(&mut matrix);
        let rref_data = matrix.to_bytes().to_vec();
        assert_eq!(rref_data, input); // all zeros stays all zeros
        let recovered = decode_block(&ops, &rref_data, n);
        assert_eq!(recovered, input);
    }

    #[test]
    fn test_matrix_round_trip_structured() {
        // Structured data that should have low rank
        let n = 8;
        let mut input = vec![0u8; n * n];
        // Row i = [i+1, i+1, i+1, ...] — rank 1 over GF(256)
        for i in 0..n {
            for j in 0..n {
                input[i * n + j] = (i + 1) as u8;
            }
        }

        let mut matrix = GfMatrix::from_bytes(&input, n);
        let ops = gaussian_eliminate(&mut matrix);
        let rref_data = matrix.to_bytes().to_vec();
        let recovered = decode_block(&ops, &rref_data, n);
        assert_eq!(recovered, input);
        // Should be rank 1
        let (_, _, _, analysis) = encode_block(&input, n);
        assert_eq!(analysis.rank, 1);
    }

    #[test]
    fn test_matrix_round_trip_various_sizes() {
        for n in [4, 8, 16, 32] {
            let mut input = vec![0u8; n * n];
            let mut val: u8 = 17;
            for byte in input.iter_mut() {
                *byte = val;
                val = val.wrapping_mul(179).wrapping_add(31);
            }

            let mut matrix = GfMatrix::from_bytes(&input, n);
            let ops = gaussian_eliminate(&mut matrix);
            let rref_data = matrix.to_bytes().to_vec();
            let recovered = decode_block(&ops, &rref_data, n);
            assert_eq!(recovered, input, "round-trip failed for n={n}");
        }
    }

    #[test]
    fn test_compression_report() {
        // Run analysis on some test data and print a report
        println!("\n=== Matrix RREF Compression Analysis ===\n");

        let test_cases: Vec<(&str, Vec<u8>)> = vec![
            ("zeros_1024", vec![0u8; 1024]),
            ("sequential_1024", (0..=255u8).cycle().take(1024).collect()),
            (
                "repeating_text",
                b"Hello, World! "
                    .iter()
                    .cycle()
                    .take(1024)
                    .copied()
                    .collect(),
            ),
            ("all_ones_1024", vec![1u8; 1024]),
            ("structured_rows", {
                let mut v = vec![0u8; 1024];
                for i in 0..32 {
                    for j in 0..32 {
                        v[i * 32 + j] = (i * 3 + j * 7) as u8;
                    }
                }
                v
            }),
        ];

        for (name, data) in &test_cases {
            println!("--- {name} ({} bytes) ---", data.len());
            let results = analyze(data);
            for (fold_size, analysis) in &results {
                let total_encoded = analysis.journal_bytes + analysis.residual_bytes;
                let total_with_delta = analysis.delta_journal_bytes + analysis.residual_bytes;
                let ratio = total_encoded as f64 / analysis.original_size as f64;
                let ratio_delta = total_with_delta as f64 / analysis.original_size as f64;

                let journal_raw_data = {
                    let (j, _, _, _) = encode_block(&data[..fold_size * fold_size], *fold_size);
                    j
                };
                let journal_entropy = entropy(&journal_raw_data);
                let delta_entropy = entropy(&delta_encode(&journal_raw_data));

                println!(
                    "  fold={fold_size:>2}: rank={:>2}/{fold_size}, ops={:>4}, \
                     journal={:>5}B, residual={:>4}B, total={:>5}B (ratio={ratio:.3}), \
                     delta_total={:>5}B (ratio={ratio_delta:.3}), \
                     journal_H={journal_entropy:.2}bps, delta_H={delta_entropy:.2}bps",
                    analysis.rank,
                    analysis.num_ops,
                    analysis.journal_bytes,
                    analysis.residual_bytes,
                    total_encoded,
                    total_with_delta,
                );
            }
            println!();
        }

        // Canterbury corpus (if available)
        let cantrbry_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("samples/cantrbry");
        if cantrbry_dir.exists() {
            println!("--- Canterbury Corpus ---");
            let files = [
                "alice29.txt",
                "asyoulik.txt",
                "cp.html",
                "fields.c",
                "grammar.lsp",
                "xargs.1",
            ];
            for filename in &files {
                let path = cantrbry_dir.join(filename);
                if let Ok(data) = std::fs::read(&path) {
                    println!("\n{filename} ({} bytes):", data.len());
                    let results = analyze(&data);
                    for (fold_size, analysis) in &results {
                        let total_encoded = analysis.journal_bytes + analysis.residual_bytes;
                        let total_with_delta =
                            analysis.delta_journal_bytes + analysis.residual_bytes;
                        let ratio = total_encoded as f64 / analysis.original_size as f64;
                        let ratio_delta = total_with_delta as f64 / analysis.original_size as f64;
                        println!(
                            "  fold={fold_size:>2}: rank={:>2}/{fold_size}, ops={:>4}, \
                             journal={:>6}B, residual={:>5}B, total={:>6}B ({ratio:.3}), \
                             delta={:>6}B ({ratio_delta:.3})",
                            analysis.rank,
                            analysis.num_ops,
                            analysis.journal_bytes,
                            analysis.residual_bytes,
                            total_encoded,
                            total_with_delta,
                        );
                    }
                }
            }
        } else {
            println!("(Canterbury corpus not extracted — skipping real-file analysis)");
            println!(
                "Run: cd samples && mkdir -p cantrbry && tar -xzf cantrbry.tar.gz -C cantrbry"
            );
        }
    }

    #[test]
    fn test_row_op_serialization() {
        let ops = vec![
            RowOp::Swap(3, 7),
            RowOp::Scale(2, 42),
            RowOp::AddScaled(5, 1, 200),
        ];
        for op in &ops {
            let bytes = op.to_bytes();
            let (recovered, len) = RowOp::from_bytes(&bytes).unwrap();
            assert_eq!(*op, recovered);
            assert_eq!(len, bytes.len());
        }
    }
}
