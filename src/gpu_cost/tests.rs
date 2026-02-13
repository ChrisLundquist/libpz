use super::*;

// ---------------------------------------------------------------------------
// Test helper: extract constants from kernel source
// ---------------------------------------------------------------------------

/// Extract an integer constant from OpenCL `#define NAME VALUE` or
/// WGSL `const NAME: TYPE = VALUE;` declarations.
///
/// Handles common patterns:
/// - `#define FOO 42u`  → Some(42)
/// - `#define FOO (1u << 15)` → Some(32768)
/// - `const FOO: u32 = 32768u;` → Some(32768)
///
/// Returns `None` if the constant is not found or has an unparseable expression.
fn extract_constant(source: &str, name: &str) -> Option<usize> {
    for line in source.lines() {
        let trimmed = line.trim();

        // OpenCL: #define NAME VALUE
        if let Some(rest) = trimmed.strip_prefix("#define") {
            let rest = rest.trim_start();
            if let Some(after_name) = rest.strip_prefix(name) {
                // Ensure the name isn't a prefix of a longer identifier:
                // the next char must be whitespace or end-of-string.
                let next_char = after_name.chars().next();
                if next_char.is_none() {
                    continue; // #define NAME with no value
                }
                if next_char.unwrap().is_alphanumeric() || next_char.unwrap() == '_' {
                    continue; // e.g., matched "K" but line is "#define K_SORT ..."
                }
                return parse_constant_expr(after_name.trim_start());
            }
        }

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

    // Strip parens: (1u << 15) → 1u << 15
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

/// Count kernel entry points in shader source.
///
/// For OpenCL (.cl): counts `__kernel void` declarations.
/// For WGSL (.wgsl): counts `@compute` annotations.
fn count_entry_points(source: &str, is_wgsl: bool) -> usize {
    if is_wgsl {
        source.matches("@compute").count()
    } else {
        source.matches("__kernel void").count()
    }
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

#[cfg(feature = "opencl")]
#[test]
fn parse_all_cl_kernels() {
    let sources = [
        ("lz77.cl", include_str!("../../kernels/lz77.cl")),
        ("lz77_batch.cl", include_str!("../../kernels/lz77_batch.cl")),
        ("lz77_hash.cl", include_str!("../../kernels/lz77_hash.cl")),
        ("lz77_topk.cl", include_str!("../../kernels/lz77_topk.cl")),
        ("bwt_rank.cl", include_str!("../../kernels/bwt_rank.cl")),
        ("bwt_radix.cl", include_str!("../../kernels/bwt_radix.cl")),
        ("bwt_sort.cl", include_str!("../../kernels/bwt_sort.cl")),
        (
            "huffman_encode.cl",
            include_str!("../../kernels/huffman_encode.cl"),
        ),
        (
            "rans_decode.cl",
            include_str!("../../kernels/rans_decode.cl"),
        ),
        (
            "rans_decode_blocks.cl",
            include_str!("../../kernels/rans_decode_blocks.cl"),
        ),
        ("fse_decode.cl", include_str!("../../kernels/fse_decode.cl")),
        (
            "fse_decode_blocks.cl",
            include_str!("../../kernels/fse_decode_blocks.cl"),
        ),
        (
            "lz77_decode.cl",
            include_str!("../../kernels/lz77_decode.cl"),
        ),
        ("fse_encode.cl", include_str!("../../kernels/fse_encode.cl")),
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
        ("lz77.cl", include_str!("../../kernels/lz77.cl")),
        ("lz77_batch.cl", include_str!("../../kernels/lz77_batch.cl")),
        ("lz77_hash.cl", include_str!("../../kernels/lz77_hash.cl")),
        (
            "lz77_lazy.wgsl",
            include_str!("../../kernels/lz77_lazy.wgsl"),
        ),
        ("lz77_topk.cl", include_str!("../../kernels/lz77_topk.cl")),
        (
            "huffman_encode.cl",
            include_str!("../../kernels/huffman_encode.cl"),
        ),
        ("bwt_rank.cl", include_str!("../../kernels/bwt_rank.cl")),
        ("bwt_radix.cl", include_str!("../../kernels/bwt_radix.cl")),
        ("bwt_sort.cl", include_str!("../../kernels/bwt_sort.cl")),
        (
            "fse_decode.wgsl",
            include_str!("../../kernels/fse_decode.wgsl"),
        ),
        (
            "rans_decode.cl",
            include_str!("../../kernels/rans_decode.cl"),
        ),
        (
            "rans_decode_blocks.cl",
            include_str!("../../kernels/rans_decode_blocks.cl"),
        ),
        ("fse_decode.cl", include_str!("../../kernels/fse_decode.cl")),
        (
            "fse_decode_blocks.cl",
            include_str!("../../kernels/fse_decode_blocks.cl"),
        ),
        (
            "lz77_decode.cl",
            include_str!("../../kernels/lz77_decode.cl"),
        ),
        ("fse_encode.cl", include_str!("../../kernels/fse_encode.cl")),
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
        ("lz77.cl", include_str!("../../kernels/lz77.cl")),
        ("lz77_batch.cl", include_str!("../../kernels/lz77_batch.cl")),
        ("lz77_hash.cl", include_str!("../../kernels/lz77_hash.cl")),
        (
            "lz77_hash.wgsl",
            include_str!("../../kernels/lz77_hash.wgsl"),
        ),
        (
            "lz77_lazy.wgsl",
            include_str!("../../kernels/lz77_lazy.wgsl"),
        ),
        ("lz77_topk.cl", include_str!("../../kernels/lz77_topk.cl")),
        (
            "lz77_topk.wgsl",
            include_str!("../../kernels/lz77_topk.wgsl"),
        ),
        ("bwt_rank.cl", include_str!("../../kernels/bwt_rank.cl")),
        ("bwt_rank.wgsl", include_str!("../../kernels/bwt_rank.wgsl")),
        ("bwt_radix.cl", include_str!("../../kernels/bwt_radix.cl")),
        (
            "bwt_radix.wgsl",
            include_str!("../../kernels/bwt_radix.wgsl"),
        ),
        ("bwt_sort.cl", include_str!("../../kernels/bwt_sort.cl")),
        (
            "huffman_encode.cl",
            include_str!("../../kernels/huffman_encode.cl"),
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
            "rans_decode.cl",
            include_str!("../../kernels/rans_decode.cl"),
        ),
        (
            "rans_decode_blocks.cl",
            include_str!("../../kernels/rans_decode_blocks.cl"),
        ),
        ("fse_decode.cl", include_str!("../../kernels/fse_decode.cl")),
        (
            "fse_decode_blocks.cl",
            include_str!("../../kernels/fse_decode_blocks.cl"),
        ),
        (
            "lz77_decode.cl",
            include_str!("../../kernels/lz77_decode.cl"),
        ),
        ("fse_encode.cl", include_str!("../../kernels/fse_encode.cl")),
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
fn extract_constant_cl_define() {
    let src = "#define HASH_SIZE  (1u << HASH_BITS)\n#define HASH_BITS  15u\n";
    assert_eq!(extract_constant(src, "HASH_BITS"), Some(15));
}

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
fn count_entry_points_cl() {
    let src = include_str!("../../kernels/lz77_hash.cl");
    assert_eq!(count_entry_points(src, false), 2); // BuildHashTable, FindMatches
}

#[test]
fn count_entry_points_wgsl() {
    let src = include_str!("../../kernels/lz77_lazy.wgsl");
    assert_eq!(count_entry_points(src, true), 3); // build, find, resolve
}

// ---------------------------------------------------------------------------
// Cross-validation: verify @pz_cost annotations match kernel constants
// ---------------------------------------------------------------------------

/// Verify that @pz_cost annotations are consistent with kernel source
/// constants for all hash-table LZ77 kernels.
///
/// Checks:
/// - `hash_counts` fixed buffer == HASH_SIZE * sizeof(u32)
/// - `hash_table` fixed buffer == HASH_SIZE * BUCKET_CAP * sizeof(u32)
/// - Match output buffer scale == sizeof(match struct) per element
/// - `passes` == number of kernel entry points
#[test]
fn cross_validate_lz77_hash_cl() {
    let src = include_str!("../../kernels/lz77_hash.cl");
    let cost = KernelCost::parse(src).expect("parse lz77_hash.cl");

    // Extract constants from kernel source
    let hash_bits = extract_constant(src, "HASH_BITS").expect("HASH_BITS in lz77_hash.cl");
    let hash_size = 1usize << hash_bits;
    let bucket_cap = extract_constant(src, "BUCKET_CAP").expect("BUCKET_CAP in lz77_hash.cl");

    // hash_counts buffer: HASH_SIZE entries × 4 bytes each
    let expected_hash_counts = hash_size * 4;
    assert_eq!(
        find_buffer(&cost, "hash_counts"),
        Some(&BufferFormula::Fixed(expected_hash_counts)),
        "lz77_hash.cl: hash_counts should be HASH_SIZE({hash_size}) * 4 = {expected_hash_counts}"
    );

    // hash_table buffer: HASH_SIZE * BUCKET_CAP entries × 4 bytes each
    let expected_hash_table = hash_size * bucket_cap * 4;
    assert_eq!(
        find_buffer(&cost, "hash_table"),
        Some(&BufferFormula::Fixed(expected_hash_table)),
        "lz77_hash.cl: hash_table should be HASH_SIZE({hash_size}) * BUCKET_CAP({bucket_cap}) * 4 = {expected_hash_table}"
    );

    // output buffer: sizeof(lz77_match_t) = 12 bytes per position
    // (uint offset=4 + uint length=4 + uchar next=1 + uchar _pad[3]=3 = 12)
    assert_eq!(
        find_buffer(&cost, "output"),
        Some(&BufferFormula::Linear {
            scale: 12,
            offset: 0
        }),
        "lz77_hash.cl: output should be N*12 (sizeof lz77_match_t)"
    );

    // passes == number of __kernel entry points
    let entry_points = count_entry_points(src, false);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_hash.cl: passes({}) should match entry point count({entry_points})",
        cost.passes
    );
}

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

    let entry_points = count_entry_points(src, true);
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

    let entry_points = count_entry_points(src, true);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_lazy.wgsl: passes vs entry points"
    );
}

/// Verify lz77_batch.cl: threads_per_element == 1/STEP_SIZE, passes == entry points.
#[test]
fn cross_validate_lz77_batch_cl() {
    let src = include_str!("../../kernels/lz77_batch.cl");
    let cost = KernelCost::parse(src).expect("parse lz77_batch.cl");

    let step_size = extract_constant(src, "STEP_SIZE").expect("STEP_SIZE in lz77_batch.cl");
    let expected_tpe = 1.0 / step_size as f64;
    assert!(
        (cost.threads_per_element - expected_tpe).abs() < 1e-9,
        "lz77_batch.cl: threads_per_element({}) should be 1/STEP_SIZE(1/{step_size}) = {expected_tpe}",
        cost.threads_per_element
    );

    let entry_points = count_entry_points(src, false);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_batch.cl: passes vs entry points"
    );
}

/// Verify lz77_topk kernels: output scale == K * sizeof(candidate).
#[test]
fn cross_validate_lz77_topk_cl() {
    let src = include_str!("../../kernels/lz77_topk.cl");
    let cost = KernelCost::parse(src).expect("parse lz77_topk.cl");

    let k = extract_constant(src, "K").expect("K in lz77_topk.cl");
    // lz77_candidate_t = { unsigned short offset; unsigned short length; } = 4 bytes
    let sizeof_candidate = 4;
    let expected_output_scale = k * sizeof_candidate;

    assert_eq!(
        find_buffer(&cost, "output"),
        Some(&BufferFormula::Linear {
            scale: expected_output_scale,
            offset: 0
        }),
        "lz77_topk.cl: output should be N*{expected_output_scale} (K={k} * sizeof(candidate)={sizeof_candidate})"
    );

    let entry_points = count_entry_points(src, false);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_topk.cl: passes vs entry points"
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

    let entry_points = count_entry_points(src, true);
    assert_eq!(
        cost.passes, entry_points,
        "lz77_topk.wgsl: passes vs entry points"
    );
}

/// Verify lz77.cl (per-position brute-force): simple kernel, 1 pass.
#[test]
fn cross_validate_lz77_cl() {
    let src = include_str!("../../kernels/lz77.cl");
    let cost = KernelCost::parse(src).expect("parse lz77.cl");

    assert_eq!(
        find_buffer(&cost, "output"),
        Some(&BufferFormula::Linear {
            scale: 12,
            offset: 0
        }),
        "lz77.cl: output should be N*12 (sizeof lz77_match_t)"
    );

    let entry_points = count_entry_points(src, false);
    assert_eq!(cost.passes, entry_points, "lz77.cl: passes vs entry points");
}

/// Verify huffman_encode kernels: pass counts match entry points.
/// The lut buffer should be 256 entries × 4 bytes = 1024.
#[test]
fn cross_validate_huffman_encode_cl() {
    let src = include_str!("../../kernels/huffman_encode.cl");
    let cost = KernelCost::parse(src).expect("parse huffman_encode.cl");

    // LUT: 256 code entries × 4 bytes each = 1024
    assert_eq!(
        find_buffer(&cost, "lut"),
        Some(&BufferFormula::Fixed(1024)),
        "huffman_encode.cl: lut should be 1024 (256 × 4)"
    );

    // bit_lengths and offsets: one u32 per symbol = N*4
    assert_eq!(
        find_buffer(&cost, "bit_lengths"),
        Some(&BufferFormula::Linear {
            scale: 4,
            offset: 0
        }),
        "huffman_encode.cl: bit_lengths should be N*4"
    );
    assert_eq!(
        find_buffer(&cost, "offsets"),
        Some(&BufferFormula::Linear {
            scale: 4,
            offset: 0
        }),
        "huffman_encode.cl: offsets should be N*4"
    );

    // The @pz_cost passes count covers ComputeBitLengths + WriteCodes (the
    // two primary kernels). ByteHistogram, PrefixSumBlock, and PrefixSumApply
    // are auxiliary kernels with their own cost profiles, not counted here.
    assert_eq!(
        cost.passes, 2,
        "huffman_encode.cl: passes should be 2 (ComputeBitLengths + WriteCodes)"
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

/// Verify BWT rank kernels: 4 passes, buffer scales match sizeof(u32).
#[test]
fn cross_validate_bwt_rank_cl() {
    let src = include_str!("../../kernels/bwt_rank.cl");
    let cost = KernelCost::parse(src).expect("parse bwt_rank.cl");

    // All four buffers are N × sizeof(u32) = N*4
    for buf_name in &["sa", "rank", "diff", "prefix"] {
        assert_eq!(
            find_buffer(&cost, buf_name),
            Some(&BufferFormula::Linear {
                scale: 4,
                offset: 0
            }),
            "bwt_rank.cl: {buf_name} should be N*4"
        );
    }

    // 4 phases: rank_compare, prefix_sum_local, prefix_sum_propagate, rank_scatter
    let entry_points = count_entry_points(src, false);
    assert_eq!(
        cost.passes, entry_points,
        "bwt_rank.cl: passes vs entry points"
    );
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

    let entry_points = count_entry_points(src, true);
    assert_eq!(
        cost.passes, entry_points,
        "bwt_rank.wgsl: passes vs entry points"
    );
}

/// Verify BWT radix kernels: buffer scales and pass counts.
#[test]
fn cross_validate_bwt_radix_cl() {
    let src = include_str!("../../kernels/bwt_radix.cl");
    let cost = KernelCost::parse(src).expect("parse bwt_radix.cl");

    for buf_name in &["keys", "histogram", "sa_in", "sa_out"] {
        assert_eq!(
            find_buffer(&cost, buf_name),
            Some(&BufferFormula::Linear {
                scale: 4,
                offset: 0
            }),
            "bwt_radix.cl: {buf_name} should be N*4"
        );
    }

    // The annotation counts 3 primary passes (compute_keys, histogram, scatter).
    // inclusive_to_exclusive is a lightweight helper counted with histogram.
    assert_eq!(cost.passes, 3, "bwt_radix.cl: passes should be 3");
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

/// Verify BWT bitonic sort: 0.5 threads per element (each thread handles a pair).
#[test]
fn cross_validate_bwt_sort_cl() {
    let src = include_str!("../../kernels/bwt_sort.cl");
    let cost = KernelCost::parse(src).expect("parse bwt_sort.cl");

    assert_eq!(
        cost.threads_per_element, 0.5,
        "bwt_sort.cl: threads_per_element should be 0.5 (pair-based)"
    );

    for buf_name in &["sa", "rank"] {
        assert_eq!(
            find_buffer(&cost, buf_name),
            Some(&BufferFormula::Linear {
                scale: 4,
                offset: 0
            }),
            "bwt_sort.cl: {buf_name} should be N*4"
        );
    }

    let entry_points = count_entry_points(src, false);
    assert_eq!(
        cost.passes, entry_points,
        "bwt_sort.cl: passes vs entry points"
    );
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

    let entry_points = count_entry_points(src, true);
    assert_eq!(
        cost.passes, entry_points,
        "fse_decode.wgsl: passes vs entry points"
    );
}

/// Verify that CL and WGSL variants of the same kernel agree on cost.
///
/// This catches the case where one variant is updated but the other is not.
#[test]
fn cross_validate_cl_wgsl_parity() {
    let pairs: &[(&str, &str, &str)] = &[
        (
            "lz77_hash",
            include_str!("../../kernels/lz77_hash.cl"),
            include_str!("../../kernels/lz77_hash.wgsl"),
        ),
        (
            "lz77_topk",
            include_str!("../../kernels/lz77_topk.cl"),
            include_str!("../../kernels/lz77_topk.wgsl"),
        ),
        (
            "huffman_encode",
            include_str!("../../kernels/huffman_encode.cl"),
            include_str!("../../kernels/huffman_encode.wgsl"),
        ),
        (
            "bwt_rank",
            include_str!("../../kernels/bwt_rank.cl"),
            include_str!("../../kernels/bwt_rank.wgsl"),
        ),
        (
            "bwt_radix",
            include_str!("../../kernels/bwt_radix.cl"),
            include_str!("../../kernels/bwt_radix.wgsl"),
        ),
    ];

    for (name, cl_src, wgsl_src) in pairs {
        let cl_cost = KernelCost::parse(cl_src).unwrap_or_else(|| panic!("parse {name}.cl"));
        let wgsl_cost = KernelCost::parse(wgsl_src).unwrap_or_else(|| panic!("parse {name}.wgsl"));

        assert_eq!(
            cl_cost.threads_per_element, wgsl_cost.threads_per_element,
            "{name}: CL and WGSL threads_per_element mismatch"
        );
        assert_eq!(
            cl_cost.passes, wgsl_cost.passes,
            "{name}: CL and WGSL passes mismatch"
        );

        // Compare total memory at a representative size (256KB)
        let n = 256 * 1024;
        assert_eq!(
            cl_cost.memory_bytes(n),
            wgsl_cost.memory_bytes(n),
            "{name}: CL and WGSL memory_bytes at 256KB differ (CL={}, WGSL={})",
            cl_cost.memory_bytes(n),
            wgsl_cost.memory_bytes(n)
        );
    }
}
