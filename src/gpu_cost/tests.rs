use super::*;

// ---------------------------------------------------------------------------
// Test helper: extract constants from kernel source
// ---------------------------------------------------------------------------

/// Extract an integer constant from WGSL `const NAME: TYPE = VALUE;` declarations.
///
/// Handles common patterns:
/// - `const FOO: u32 = 32768u;` -> Some(32768)
///
/// Returns `None` if the constant is not found or has an unparseable expression.
fn extract_constant(source: &str, name: &str) -> Option<usize> {
    for line in source.lines() {
        let trimmed = line.trim();

        // WGSL: const NAME: TYPE = VALUE;
        if let Some(rest) = trimmed.strip_prefix("const ") {
            let rest = rest.trim_start();
            if let Some(after_name) = rest.strip_prefix(name) {
                let after_name = after_name.trim_start();
                // Must be followed by `:` (type annotation)
                if after_name.starts_with(':') {
                    if let Some(eq_pos) = after_name.find('=') {
                        let val_str = after_name[eq_pos + 1..].trim();
                        return parse_constant_expr(val_str);
                    }
                }
            }
        }
    }
    None
}

/// Parse a constant expression from kernel source.
///
/// Supports:
/// - Plain integers: `42`, `42u`
/// - Shift expressions: `(1u << 15)`, `1u << 15`
fn parse_constant_expr(s: &str) -> Option<usize> {
    // Strip trailing comments (// ...) before parsing
    let s = if let Some(comment_pos) = s.find("//") {
        &s[..comment_pos]
    } else {
        s
    };
    let s = s.trim().trim_end_matches(';');
    let s = s.trim();

    // Strip parens: (1u << 15) -> 1u << 15
    let s = if s.starts_with('(') && s.ends_with(')') {
        &s[1..s.len() - 1]
    } else {
        s
    };
    let s = s.trim();

    // Shift expression: BASE << SHIFT
    if let Some(shift_pos) = s.find("<<") {
        let base_str = s[..shift_pos].trim().trim_end_matches('u');
        let shift_str = s[shift_pos + 2..].trim().trim_end_matches('u').trim();
        if let (Ok(base), Ok(shift)) = (base_str.parse::<usize>(), shift_str.parse::<usize>()) {
            return Some(base << shift);
        }
    }

    // Plain integer (with optional `u` suffix)
    let s = s.trim_end_matches('u');
    s.parse::<usize>().ok()
}

/// Count kernel entry points in WGSL shader source by counting `@compute` annotations.
fn count_entry_points(source: &str) -> usize {
    source.matches("@compute").count()
}

/// Look up a named buffer's formula in a KernelCost.
fn find_buffer<'a>(cost: &'a KernelCost, name: &str) -> Option<&'a BufferFormula> {
    cost.buffers.iter().find(|(n, _)| n == name).map(|(_, f)| f)
}

// ---------------------------------------------------------------------------
// Unit tests: annotation parsing
// ---------------------------------------------------------------------------

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
    // lz77_lazy: input=N, hash_counts=524288, hash_table=134217728,
    //            raw_matches=N*12, resolved=N*12, staging=N*12
    let src = r#"
// @pz_cost {
//   threads_per_element: 1
//   passes: 3
//   buffers: input=N, hash_counts=524288, hash_table=134217728, raw_matches=N*12, resolved=N*12, staging=N*12
//   local_mem: 0
// }
"#;
    let cost = KernelCost::parse(src).unwrap();
    // At N=256KB (262144):
    // input=262144, hash_counts=524288, hash_table=134217728,
    // raw_matches=3145728, resolved=3145728, staging=3145728
    // Total = 262144 + 524288 + 134217728 + 3*3145728 = 144401344
    let n = 256 * 1024;
    let expected = n + 524288 + 134217728 + 3 * (n * 12);
    assert_eq!(cost.memory_bytes(n), expected);

    // At N=4MB (4194304):
    // input=4194304, hash_counts=524288, hash_table=134217728,
    // raw_matches=50331648, resolved=50331648, staging=50331648
    // Total = 4194304 + 524288 + 134217728 + 3*50331648 = 289736960 (~276MB)
    let n = 4 * 1024 * 1024;
    let expected = n + 524288 + 134217728 + 3 * (n * 12);
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
    // 0.03125 = 1/32, so 1M input -> 32768 threads * 1 pass
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

// ---------------------------------------------------------------------------
// Parse every actual kernel source via include_str!
// ---------------------------------------------------------------------------

#[cfg(feature = "webgpu")]
#[test]
fn parse_all_wgsl_kernels() {
    let sources = [
        (
            "lz77_hash.wgsl",
            include_str!("../../kernels/lz77_hash.wgsl"),
        ),
        (
            "lz77_lazy.wgsl",
            include_str!("../../kernels/lz77_lazy.wgsl"),
        ),
        (
            "lz77_topk.wgsl",
            include_str!("../../kernels/lz77_topk.wgsl"),
        ),
        ("bwt_rank.wgsl", include_str!("../../kernels/bwt_rank.wgsl")),
        (
            "bwt_radix.wgsl",
            include_str!("../../kernels/bwt_radix.wgsl"),
        ),
        (
            "huffman_encode.wgsl",
            include_str!("../../kernels/huffman_encode.wgsl"),
        ),
        (
            "fse_decode.wgsl",
            include_str!("../../kernels/fse_decode.wgsl"),
        ),
        (
            "fse_encode.wgsl",
            include_str!("../../kernels/fse_encode.wgsl"),
        ),
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
        (
            "lz77_lazy.wgsl",
            include_str!("../../kernels/lz77_lazy.wgsl"),
        ),
        (
            "fse_decode.wgsl",
            include_str!("../../kernels/fse_decode.wgsl"),
        ),
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

#[test]
fn parse_all_kernels() {
    let sources: &[(&str, &str)] = &[
        (
            "lz77_hash.wgsl",
            include_str!("../../kernels/lz77_hash.wgsl"),
        ),
        (
            "lz77_lazy.wgsl",
            include_str!("../../kernels/lz77_lazy.wgsl"),
        ),
        (
            "lz77_topk.wgsl",
            include_str!("../../kernels/lz77_topk.wgsl"),
        ),
        ("bwt_rank.wgsl", include_str!("../../kernels/bwt_rank.wgsl")),
        (
            "bwt_radix.wgsl",
            include_str!("../../kernels/bwt_radix.wgsl"),
        ),
        (
            "huffman_encode.wgsl",
            include_str!("../../kernels/huffman_encode.wgsl"),
        ),
        (
            "fse_decode.wgsl",
            include_str!("../../kernels/fse_decode.wgsl"),
        ),
        (
            "fse_encode.wgsl",
            include_str!("../../kernels/fse_encode.wgsl"),
        ),
    ];
    for (name, src) in sources {
        assert!(
            KernelCost::parse(src).is_some(),
            "Failed to parse @pz_cost from {name}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test helpers: constant extraction
// ---------------------------------------------------------------------------

#[test]
fn extract_constant_wgsl_const() {
    let src = "const HASH_SIZE: u32 = 32768u;\nconst BUCKET_CAP: u32 = 64u;\n";
    assert_eq!(extract_constant(src, "HASH_SIZE"), Some(32768));
    assert_eq!(extract_constant(src, "BUCKET_CAP"), Some(64));
}

#[test]
fn extract_constant_shift_expr() {
    assert_eq!(parse_constant_expr("(1u << 15)"), Some(32768));
    assert_eq!(parse_constant_expr("1u << 15"), Some(32768));
    assert_eq!(parse_constant_expr("32768u"), Some(32768));
    assert_eq!(parse_constant_expr("32768u;"), Some(32768));
}

#[test]
fn count_entry_points_wgsl() {
    let src = include_str!("../../kernels/lz77_lazy.wgsl");
    assert_eq!(count_entry_points(src), 3); // build, find, resolve
}

// ---------------------------------------------------------------------------
// Cross-validation: verify @pz_cost annotations match kernel constants
// ---------------------------------------------------------------------------

#[test]
fn cross_validate_lz77_hash_wgsl() {
    let src = include_str!("../../kernels/lz77_hash.wgsl");
    let cost = KernelCost::parse(src).expect("parse lz77_hash.wgsl");

    let hash_size = extract_constant(src, "HASH_SIZE").expect("HASH_SIZE in lz77_hash.wgsl");
    let bucket_cap = extract_constant(src, "BUCKET_CAP").expect("BUCKET_CAP in lz77_hash.wgsl");

    let expected_hash_counts = hash_size * 4;
    assert_eq!(
        find_buffer(&cost, "hash_counts"),
        Some(&BufferFormula::Fixed(expected_hash_counts)),
        "lz77_hash.wgsl: hash_counts mismatch"
    );

    let expected_hash_table = hash_size * bucket_cap * 4;
    assert_eq!(
        find_buffer(&cost, "hash_table"),
        Some(&BufferFormula::Fixed(expected_hash_table)),
        "lz77_hash.wgsl: hash_table mismatch"
    );

    assert_eq!(
        find_buffer(&cost, "output"),
        Some(&BufferFormula::Linear {
            scale: 12,
            offset: 0
        }),
        "lz77_hash.wgsl: output should be N*12 (sizeof Lz77Match = 3×u32 = 12)"
    );

    let entry_points = count_entry_points(src);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_hash.wgsl: passes vs entry points"
    );
}

#[test]
fn cross_validate_lz77_lazy_wgsl() {
    let src = include_str!("../../kernels/lz77_lazy.wgsl");
    let cost = KernelCost::parse(src).expect("parse lz77_lazy.wgsl");

    let hash_size = extract_constant(src, "HASH_SIZE").expect("HASH_SIZE in lz77_lazy.wgsl");
    let bucket_cap = extract_constant(src, "BUCKET_CAP").expect("BUCKET_CAP in lz77_lazy.wgsl");

    let expected_hash_counts = hash_size * 4;
    assert_eq!(
        find_buffer(&cost, "hash_counts"),
        Some(&BufferFormula::Fixed(expected_hash_counts)),
        "lz77_lazy.wgsl: hash_counts mismatch"
    );

    let expected_hash_table = hash_size * bucket_cap * 4;
    assert_eq!(
        find_buffer(&cost, "hash_table"),
        Some(&BufferFormula::Fixed(expected_hash_table)),
        "lz77_lazy.wgsl: hash_table mismatch"
    );

    // Lazy kernel has three match buffers, all N*12
    for buf_name in &["raw_matches", "resolved", "staging"] {
        assert_eq!(
            find_buffer(&cost, buf_name),
            Some(&BufferFormula::Linear {
                scale: 12,
                offset: 0
            }),
            "lz77_lazy.wgsl: {buf_name} should be N*12"
        );
    }

    let entry_points = count_entry_points(src);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_lazy.wgsl: passes vs entry points"
    );
}

#[test]
fn cross_validate_lz77_topk_wgsl() {
    let src = include_str!("../../kernels/lz77_topk.wgsl");
    let cost = KernelCost::parse(src).expect("parse lz77_topk.wgsl");

    let k = extract_constant(src, "K").expect("K in lz77_topk.wgsl");
    // WGSL packs each candidate as a single u32 (offset in low 16, length in high 16)
    let sizeof_candidate = 4;
    let expected_output_scale = k * sizeof_candidate;

    assert_eq!(
        find_buffer(&cost, "output"),
        Some(&BufferFormula::Linear {
            scale: expected_output_scale,
            offset: 0
        }),
        "lz77_topk.wgsl: output should be N*{expected_output_scale}"
    );

    let entry_points = count_entry_points(src);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_topk.wgsl: passes vs entry points"
    );
}

#[test]
fn cross_validate_huffman_encode_wgsl() {
    let src = include_str!("../../kernels/huffman_encode.wgsl");
    let cost = KernelCost::parse(src).expect("parse huffman_encode.wgsl");

    assert_eq!(
        find_buffer(&cost, "lut"),
        Some(&BufferFormula::Fixed(1024)),
        "huffman_encode.wgsl: lut should be 1024"
    );

    assert_eq!(cost.passes, 2, "huffman_encode.wgsl: passes should be 2");
}

#[test]
fn cross_validate_bwt_rank_wgsl() {
    let src = include_str!("../../kernels/bwt_rank.wgsl");
    let cost = KernelCost::parse(src).expect("parse bwt_rank.wgsl");

    for buf_name in &["sa", "rank", "diff", "prefix"] {
        assert_eq!(
            find_buffer(&cost, buf_name),
            Some(&BufferFormula::Linear {
                scale: 4,
                offset: 0
            }),
            "bwt_rank.wgsl: {buf_name} should be N*4"
        );
    }

    let entry_points = count_entry_points(src);
    assert_eq!(
        cost.passes, entry_points,
        "bwt_rank.wgsl: passes vs entry points"
    );
}

#[test]
fn cross_validate_bwt_radix_wgsl() {
    let src = include_str!("../../kernels/bwt_radix.wgsl");
    let cost = KernelCost::parse(src).expect("parse bwt_radix.wgsl");

    for buf_name in &["keys", "histogram", "sa_in", "sa_out"] {
        assert_eq!(
            find_buffer(&cost, buf_name),
            Some(&BufferFormula::Linear {
                scale: 4,
                offset: 0
            }),
            "bwt_radix.wgsl: {buf_name} should be N*4"
        );
    }

    assert_eq!(cost.passes, 3, "bwt_radix.wgsl: passes should be 3");
}

/// Verify FSE decode kernel: simple 1-pass kernel.
#[test]
fn cross_validate_fse_decode_wgsl() {
    let src = include_str!("../../kernels/fse_decode.wgsl");
    let cost = KernelCost::parse(src).expect("parse fse_decode.wgsl");

    // Decode table: 1024 entries × 4 bytes = 4096
    assert_eq!(
        find_buffer(&cost, "table"),
        Some(&BufferFormula::Fixed(4096)),
        "fse_decode.wgsl: table should be 4096 (1024 × 4)"
    );

    let entry_points = count_entry_points(src);
    assert_eq!(
        cost.passes, entry_points,
        "fse_decode.wgsl: passes vs entry points"
    );
}
