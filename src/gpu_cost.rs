//! GPU kernel cost model: parse `@pz_cost` annotations from kernel source
//! and estimate per-dispatch memory requirements.
//!
//! Kernel files embed structured cost comments that describe their buffer
//! allocations, thread counts, and dispatch passes. This module parses those
//! annotations at engine init time (from `include_str!` constants) so the
//! scheduler can compute how many blocks fit in GPU memory.

/// Expression for buffer size as a function of input length N.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BufferFormula {
    /// Fixed size in bytes (e.g., `131072`).
    Fixed(usize),
    /// Linear in N: `scale * N + offset` (e.g., `N*12` → scale=12, offset=0).
    Linear { scale: usize, offset: usize },
}

impl BufferFormula {
    /// Evaluate the formula for a given input length.
    pub fn eval(&self, n: usize) -> usize {
        match self {
            BufferFormula::Fixed(size) => *size,
            BufferFormula::Linear { scale, offset } => scale * n + offset,
        }
    }
}

/// Cost model for a single GPU kernel, parsed from `@pz_cost` annotations
/// embedded as structured comments in kernel source files.
#[derive(Debug, Clone)]
pub(crate) struct KernelCost {
    /// GPU threads launched per input byte (e.g., 1.0 for per-position kernels).
    pub threads_per_element: f64,
    /// Number of sequential kernel dispatches.
    pub passes: usize,
    /// Buffer allocations: (name, formula for size in bytes).
    pub buffers: Vec<(String, BufferFormula)>,
    /// Workgroup-local memory in bytes.
    pub local_mem: usize,
}

impl KernelCost {
    /// Parse a `@pz_cost` block from a kernel source string.
    /// Returns `None` if no `@pz_cost` block is found.
    pub fn parse(source: &str) -> Option<Self> {
        // Find the line containing @pz_cost {
        let mut lines = source.lines();
        let mut found = false;
        for line in &mut lines {
            let stripped = strip_comment_prefix(line);
            if stripped.contains("@pz_cost") && stripped.contains('{') {
                found = true;
                break;
            }
        }
        if !found {
            return None;
        }

        let mut threads_per_element = 1.0;
        let mut passes = 1;
        let mut buffers = Vec::new();
        let mut local_mem = 0;

        // Collect lines until closing brace
        for line in lines {
            let stripped = strip_comment_prefix(line);
            let stripped = stripped.trim();
            if stripped.starts_with('}') {
                break;
            }

            if let Some(rest) = stripped.strip_prefix("threads_per_element:") {
                if let Ok(v) = rest.trim().parse::<f64>() {
                    threads_per_element = v;
                }
            } else if let Some(rest) = stripped.strip_prefix("passes:") {
                if let Ok(v) = rest.trim().parse::<usize>() {
                    passes = v;
                }
            } else if let Some(rest) = stripped.strip_prefix("local_mem:") {
                if let Ok(v) = rest.trim().parse::<usize>() {
                    local_mem = v;
                }
            } else if let Some(rest) = stripped.strip_prefix("buffers:") {
                buffers = parse_buffers(rest.trim());
            }
            // `note:` and unknown keys are silently ignored.
        }

        Some(KernelCost {
            threads_per_element,
            passes,
            buffers,
            local_mem,
        })
    }

    /// Total GPU memory for a given input size, in bytes.
    pub fn memory_bytes(&self, input_len: usize) -> usize {
        self.buffers.iter().map(|(_, f)| f.eval(input_len)).sum()
    }

    /// Total GPU threads across all passes.
    pub fn total_threads(&self, input_len: usize) -> usize {
        (self.threads_per_element * input_len as f64) as usize * self.passes
    }
}

/// Strip leading `//` or `///` comment prefix (for both .cl and .wgsl files).
fn strip_comment_prefix(line: &str) -> &str {
    let trimmed = line.trim_start();
    if let Some(rest) = trimmed.strip_prefix("///") {
        rest
    } else if let Some(rest) = trimmed.strip_prefix("//") {
        rest
    } else {
        trimmed
    }
}

/// Parse a comma-separated list of `name=expr` buffer declarations.
///
/// Supported expr formats:
/// - Bare integer: `131072` → `BufferFormula::Fixed(131072)`
/// - `N` → `BufferFormula::Linear { scale: 1, offset: 0 }`
/// - `N*<int>` → `BufferFormula::Linear { scale: int, offset: 0 }`
fn parse_buffers(s: &str) -> Vec<(String, BufferFormula)> {
    let mut result = Vec::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mut kv = part.splitn(2, '=');
        let name = match kv.next() {
            Some(n) => n.trim().to_string(),
            None => continue,
        };
        let expr = match kv.next() {
            Some(e) => e.trim(),
            None => continue,
        };
        if let Some(formula) = parse_formula(expr) {
            result.push((name, formula));
        }
    }
    result
}

/// Parse a single buffer size expression.
fn parse_formula(expr: &str) -> Option<BufferFormula> {
    let expr = expr.trim();
    if expr == "N" {
        return Some(BufferFormula::Linear {
            scale: 1,
            offset: 0,
        });
    }
    if let Some(rest) = expr.strip_prefix("N*") {
        if let Ok(scale) = rest.trim().parse::<usize>() {
            return Some(BufferFormula::Linear { scale, offset: 0 });
        }
    }
    if let Ok(fixed) = expr.parse::<usize>() {
        return Some(BufferFormula::Fixed(fixed));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_annotation() {
        let src = r#"
// @pz_cost {
//   threads_per_element: 1
//   passes: 2
//   buffers: input=N, table=1024, output=N*12
//   local_mem: 256
// }
"#;
        let cost = KernelCost::parse(src).expect("should parse");
        assert_eq!(cost.threads_per_element, 1.0);
        assert_eq!(cost.passes, 2);
        assert_eq!(cost.local_mem, 256);
        assert_eq!(cost.buffers.len(), 3);
        assert_eq!(
            cost.buffers[0],
            (
                "input".to_string(),
                BufferFormula::Linear {
                    scale: 1,
                    offset: 0
                }
            )
        );
        assert_eq!(
            cost.buffers[1],
            ("table".to_string(), BufferFormula::Fixed(1024))
        );
        assert_eq!(
            cost.buffers[2],
            (
                "output".to_string(),
                BufferFormula::Linear {
                    scale: 12,
                    offset: 0
                }
            )
        );

        // memory_bytes at N=1024: 1024 + 1024 + 1024*12 = 14336
        assert_eq!(cost.memory_bytes(1024), 1024 + 1024 + 12288);
    }

    #[test]
    fn parse_with_note_field() {
        let src = r#"
// @pz_cost {
//   threads_per_element: 0.5
//   passes: 1
//   buffers: sa=N*4, rank=N*4
//   local_mem: 0
//   note: O(log^2 N) dispatches for full bitonic sort
// }
"#;
        let cost = KernelCost::parse(src).expect("should parse");
        assert_eq!(cost.threads_per_element, 0.5);
        assert_eq!(cost.passes, 1);
        assert_eq!(cost.buffers.len(), 2);
    }

    #[test]
    fn no_annotation_returns_none() {
        let src = "// Just a regular kernel\n__kernel void foo() {}\n";
        assert!(KernelCost::parse(src).is_none());
    }

    #[test]
    fn memory_bytes_at_known_sizes() {
        // lz77_lazy: input=N, hash_counts=131072, hash_table=8388608,
        //            raw_matches=N*12, resolved=N*12, staging=N*12
        let src = r#"
// @pz_cost {
//   threads_per_element: 1
//   passes: 3
//   buffers: input=N, hash_counts=131072, hash_table=8388608, raw_matches=N*12, resolved=N*12, staging=N*12
//   local_mem: 0
// }
"#;
        let cost = KernelCost::parse(src).unwrap();
        // At N=256KB (262144):
        // input=262144, hash_counts=131072, hash_table=8388608,
        // raw_matches=3145728, resolved=3145728, staging=3145728
        // Total = 262144 + 131072 + 8388608 + 3*3145728 = 18219008
        let n = 256 * 1024;
        let expected = n + 131072 + 8388608 + 3 * (n * 12);
        assert_eq!(cost.memory_bytes(n), expected);

        // At N=4MB (4194304):
        // input=4194304, hash_counts=131072, hash_table=8388608,
        // raw_matches=50331648, resolved=50331648, staging=50331648
        // Total = 4194304 + 131072 + 8388608 + 3*50331648 = 163689728 (~156MB)
        let n = 4 * 1024 * 1024;
        let expected = n + 131072 + 8388608 + 3 * (n * 12);
        assert_eq!(cost.memory_bytes(n), expected);
    }

    #[test]
    fn total_threads_calculation() {
        let src = r#"
// @pz_cost {
//   threads_per_element: 0.03125
//   passes: 1
//   buffers: input=N, output=N*12
//   local_mem: 0
// }
"#;
        let cost = KernelCost::parse(src).unwrap();
        // 0.03125 = 1/32, so 1M input → 32768 threads * 1 pass
        assert_eq!(cost.total_threads(1_048_576), 32768);
    }

    #[test]
    fn parse_triple_slash_comments() {
        let src = r#"
/// @pz_cost {
///   threads_per_element: 1
///   passes: 4
///   buffers: sa=N*4, rank=N*4, diff=N*4, prefix=N*4
///   local_mem: 2048
///   note: called per doubling step
/// }
"#;
        let cost = KernelCost::parse(src).expect("should parse triple-slash");
        assert_eq!(cost.passes, 4);
        assert_eq!(cost.local_mem, 2048);
        assert_eq!(cost.buffers.len(), 4);
    }

    // Parse every actual kernel source via include_str! to ensure all 15 return Some.
    #[cfg(feature = "webgpu")]
    #[test]
    fn parse_all_wgsl_kernels() {
        let sources = [
            ("lz77_hash.wgsl", include_str!("../kernels/lz77_hash.wgsl")),
            ("lz77_lazy.wgsl", include_str!("../kernels/lz77_lazy.wgsl")),
            ("lz77_topk.wgsl", include_str!("../kernels/lz77_topk.wgsl")),
            ("bwt_rank.wgsl", include_str!("../kernels/bwt_rank.wgsl")),
            ("bwt_radix.wgsl", include_str!("../kernels/bwt_radix.wgsl")),
            (
                "huffman_encode.wgsl",
                include_str!("../kernels/huffman_encode.wgsl"),
            ),
            (
                "fse_decode.wgsl",
                include_str!("../kernels/fse_decode.wgsl"),
            ),
        ];
        for (name, src) in &sources {
            assert!(
                KernelCost::parse(src).is_some(),
                "Failed to parse @pz_cost from {name}"
            );
        }
    }

    #[cfg(feature = "opencl")]
    #[test]
    fn parse_all_cl_kernels() {
        let sources = [
            ("lz77.cl", include_str!("../kernels/lz77.cl")),
            ("lz77_batch.cl", include_str!("../kernels/lz77_batch.cl")),
            ("lz77_hash.cl", include_str!("../kernels/lz77_hash.cl")),
            ("lz77_topk.cl", include_str!("../kernels/lz77_topk.cl")),
            ("bwt_rank.cl", include_str!("../kernels/bwt_rank.cl")),
            ("bwt_radix.cl", include_str!("../kernels/bwt_radix.cl")),
            ("bwt_sort.cl", include_str!("../kernels/bwt_sort.cl")),
            (
                "huffman_encode.cl",
                include_str!("../kernels/huffman_encode.cl"),
            ),
            ("rans_decode.cl", include_str!("../kernels/rans_decode.cl")),
            (
                "rans_decode_blocks.cl",
                include_str!("../kernels/rans_decode_blocks.cl"),
            ),
            ("fse_decode.cl", include_str!("../kernels/fse_decode.cl")),
            (
                "fse_decode_blocks.cl",
                include_str!("../kernels/fse_decode_blocks.cl"),
            ),
            ("lz77_decode.cl", include_str!("../kernels/lz77_decode.cl")),
        ];
        for (name, src) in &sources {
            assert!(
                KernelCost::parse(src).is_some(),
                "Failed to parse @pz_cost from {name}"
            );
        }
    }

    #[test]
    fn print_cost_table() {
        let kernels: &[(&str, &str)] = &[
            ("lz77.cl", include_str!("../kernels/lz77.cl")),
            ("lz77_batch.cl", include_str!("../kernels/lz77_batch.cl")),
            ("lz77_hash.cl", include_str!("../kernels/lz77_hash.cl")),
            ("lz77_lazy.wgsl", include_str!("../kernels/lz77_lazy.wgsl")),
            ("lz77_topk.cl", include_str!("../kernels/lz77_topk.cl")),
            (
                "huffman_encode.cl",
                include_str!("../kernels/huffman_encode.cl"),
            ),
            ("bwt_rank.cl", include_str!("../kernels/bwt_rank.cl")),
            ("bwt_radix.cl", include_str!("../kernels/bwt_radix.cl")),
            ("bwt_sort.cl", include_str!("../kernels/bwt_sort.cl")),
            (
                "fse_decode.wgsl",
                include_str!("../kernels/fse_decode.wgsl"),
            ),
            ("rans_decode.cl", include_str!("../kernels/rans_decode.cl")),
            (
                "rans_decode_blocks.cl",
                include_str!("../kernels/rans_decode_blocks.cl"),
            ),
            ("fse_decode.cl", include_str!("../kernels/fse_decode.cl")),
            (
                "fse_decode_blocks.cl",
                include_str!("../kernels/fse_decode_blocks.cl"),
            ),
            ("lz77_decode.cl", include_str!("../kernels/lz77_decode.cl")),
        ];
        let sizes: &[usize] = &[64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024];

        eprintln!();
        eprintln!(
            "{:<22} {:>5} {:>7}  {:>10}  {:>10}  {:>10}  {:>10}",
            "KERNEL", "PASS", "TPE", "64KB", "256KB", "1MB", "4MB"
        );
        eprintln!("{}", "-".repeat(88));
        for (name, src) in kernels {
            let cost = KernelCost::parse(src).unwrap();
            eprint!(
                "{:<22} {:>5} {:>7.5}",
                name, cost.passes, cost.threads_per_element
            );
            for &n in sizes {
                let mb = cost.memory_bytes(n) as f64 / (1024.0 * 1024.0);
                eprint!("  {:>8.1} MB", mb);
            }
            eprintln!();
        }
        eprintln!();

        // Show max_in_flight at 128MB budget (typical Apple M-series GPU)
        let budget: usize = 128 * 1024 * 1024;
        eprintln!("max_in_flight (128 MB budget):");
        eprintln!(
            "{:<22}  {:>6}  {:>6}  {:>6}  {:>6}",
            "KERNEL", "64KB", "256KB", "1MB", "4MB"
        );
        eprintln!("{}", "-".repeat(54));
        for (name, src) in kernels {
            let cost = KernelCost::parse(src).unwrap();
            eprint!("{:<22}", name);
            for &n in sizes {
                let per_block = cost.memory_bytes(n);
                let max = if per_block == 0 {
                    8
                } else {
                    (budget / per_block).max(1)
                };
                eprint!("  {:>6}", max);
            }
            eprintln!();
        }
        eprintln!();
    }

    // Non-feature-gated test that parses all 15 kernels (include_str! works without features)
    #[test]
    fn parse_all_kernels() {
        let sources: &[(&str, &str)] = &[
            ("lz77.cl", include_str!("../kernels/lz77.cl")),
            ("lz77_batch.cl", include_str!("../kernels/lz77_batch.cl")),
            ("lz77_hash.cl", include_str!("../kernels/lz77_hash.cl")),
            ("lz77_hash.wgsl", include_str!("../kernels/lz77_hash.wgsl")),
            ("lz77_lazy.wgsl", include_str!("../kernels/lz77_lazy.wgsl")),
            ("lz77_topk.cl", include_str!("../kernels/lz77_topk.cl")),
            ("lz77_topk.wgsl", include_str!("../kernels/lz77_topk.wgsl")),
            ("bwt_rank.cl", include_str!("../kernels/bwt_rank.cl")),
            ("bwt_rank.wgsl", include_str!("../kernels/bwt_rank.wgsl")),
            ("bwt_radix.cl", include_str!("../kernels/bwt_radix.cl")),
            ("bwt_radix.wgsl", include_str!("../kernels/bwt_radix.wgsl")),
            ("bwt_sort.cl", include_str!("../kernels/bwt_sort.cl")),
            (
                "huffman_encode.cl",
                include_str!("../kernels/huffman_encode.cl"),
            ),
            (
                "huffman_encode.wgsl",
                include_str!("../kernels/huffman_encode.wgsl"),
            ),
            (
                "fse_decode.wgsl",
                include_str!("../kernels/fse_decode.wgsl"),
            ),
            ("rans_decode.cl", include_str!("../kernels/rans_decode.cl")),
            (
                "rans_decode_blocks.cl",
                include_str!("../kernels/rans_decode_blocks.cl"),
            ),
            ("fse_decode.cl", include_str!("../kernels/fse_decode.cl")),
            (
                "fse_decode_blocks.cl",
                include_str!("../kernels/fse_decode_blocks.cl"),
            ),
            ("lz77_decode.cl", include_str!("../kernels/lz77_decode.cl")),
        ];
        for (name, src) in sources {
            assert!(
                KernelCost::parse(src).is_some(),
                "Failed to parse @pz_cost from {name}"
            );
        }
    }
}
