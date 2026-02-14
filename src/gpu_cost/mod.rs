//! GPU kernel cost model: parse `@pz_cost` annotations from kernel source
//! and estimate per-dispatch memory requirements.
//!
//! Kernel files embed structured cost comments that describe their buffer
//! allocations, thread counts, and dispatch passes. This module parses those
//! annotations at engine init time (from `include_str!` constants) so the
//! scheduler can compute how many blocks fit in GPU memory.

#[cfg(test)]
mod tests;

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

/// Strip leading `//` or `///` comment prefix (for .wgsl files).
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
