//! GPU Re-Pair grammar compression for Experiment C.
//!
//! Tests the viability of iterative GPU kernel dispatch.
//! Each round dispatches up to 5 kernels: histogram, argmax, replace, prefix_sum, scatter.
//! The key measurement is dispatch_overhead / compute_time ratio.

use super::*;

impl WebGpuEngine {
    /// GPU-accelerated Re-Pair compression.
    ///
    /// Runs the iterative grammar compression loop on the GPU:
    /// while any bigram frequency > threshold:
    ///   1. GPU histogram: count all bigram frequencies
    ///   2. GPU argmax: find most frequent bigram
    ///   3. GPU replace (two-buffer): replace occurrences, mark deletions
    ///   4. GPU prefix sum: compute compaction offsets
    ///   5. GPU scatter: compact symbols
    ///
    /// Returns compressed data in the same wire format as the CPU implementation.
    pub fn repair_compress(&self, input: &[u8]) -> PzResult<Vec<u8>> {
        let n = input.len();
        if n == 0 {
            return Err(PzError::InvalidInput);
        }

        let pipelines = self.repair_pipelines();

        // Convert input to u32 symbol array.
        let mut symbols: Vec<u32> = input.iter().map(|&b| b as u32).collect();
        let mut next_symbol: u32 = 256;
        let mut rules: Vec<(u32, u32, u32)> = Vec::new(); // (new_sym, left, right)
        let max_rounds = 1000;
        let min_freq = 2u32;

        for _round in 0..max_rounds {
            let current_n = symbols.len();
            if current_n < 2 {
                break;
            }

            // Current alphabet size (for histogram sizing).
            let max_alphabet = next_symbol.max(256) as usize;

            // For large alphabets, histogram becomes too large for shared memory.
            // Fall back to CPU for alphabets > 512.
            if max_alphabet > 512 {
                let (best_a, best_b, best_freq) = cpu_find_best_bigram(&symbols);
                if best_freq < min_freq {
                    break;
                }
                cpu_replace_bigram(&mut symbols, best_a, best_b, next_symbol);
                rules.push((next_symbol, best_a, best_b));
                next_symbol += 1;
                continue;
            }

            let hist_size = max_alphabet * max_alphabet;

            // Upload current symbol array.
            let symbols_buf = self.create_buffer_init(
                "repair_symbols",
                bytemuck::cast_slice(&symbols),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );

            // Histogram buffer (zero-initialized).
            let histogram_buf = self.create_buffer_init(
                "repair_histogram",
                &vec![0u8; hist_size * 4],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );

            // Scratch buffer for argmax results.
            let scratch_size = current_n.max(hist_size / 256 * 2 + 2);
            let scratch_buf = self.create_buffer_init(
                "repair_scratch",
                &vec![0u8; scratch_size * 4],
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );

            // --- Kernel 1: Histogram ---
            let hist_params = [current_n as u32, max_alphabet as u32, 0u32, 0u32];
            let hist_params_buf = self.create_buffer_init(
                "repair_hist_params",
                bytemuck::cast_slice(&hist_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let hist_bg_layout = pipelines.histogram.get_bind_group_layout(0);
            let hist_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("repair_histogram_bg"),
                layout: &hist_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: symbols_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: scratch_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: hist_params_buf.as_entire_binding(),
                    },
                ],
            });

            let workgroups = (current_n as u32).div_ceil(256);
            self.dispatch(
                &pipelines.histogram,
                &hist_bg,
                workgroups,
                "repair_histogram",
            )?;

            // --- Kernel 2: Argmax ---
            let argmax_params = [hist_size as u32, max_alphabet as u32, 0u32, 0u32];
            let argmax_params_buf = self.create_buffer_init(
                "repair_argmax_params",
                bytemuck::cast_slice(&argmax_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let argmax_bg_layout = pipelines.argmax.get_bind_group_layout(0);
            let argmax_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("repair_argmax_bg"),
                layout: &argmax_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: symbols_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: scratch_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: argmax_params_buf.as_entire_binding(),
                    },
                ],
            });

            let argmax_workgroups = (hist_size as u32).div_ceil(256);
            self.dispatch(
                &pipelines.argmax,
                &argmax_bg,
                argmax_workgroups,
                "repair_argmax",
            )?;

            // Read back the argmax result.
            let num_argmax_blocks = (hist_size as u32).div_ceil(256) as usize;
            let argmax_data = self.read_buffer(&scratch_buf, (num_argmax_blocks * 2 * 4) as u64);
            let argmax_u32s: &[u32] = bytemuck::cast_slice(&argmax_data);

            let mut best_freq = 0u32;
            let mut best_idx = 0u32;
            for block in 0..num_argmax_blocks {
                let freq = argmax_u32s[block * 2];
                let idx = argmax_u32s[block * 2 + 1];
                if freq > best_freq {
                    best_freq = freq;
                    best_idx = idx;
                }
            }

            if best_freq < min_freq {
                break;
            }

            let best_a = best_idx / max_alphabet as u32;
            let best_b = best_idx % max_alphabet as u32;

            // --- Phase 3: GPU two-buffer replace + compact ---
            let replace_pips = self.repair_replace_pipelines();

            // symbols_buf is the input (read-only for replace).
            // Create output buffer for replaced symbols.
            let symbols_out_buf = self.create_buffer(
                "repair_symbols_out",
                (current_n * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );

            // Keep flags buffer (1 = keep, 0 = delete).
            let keep_flags_buf = self.create_buffer(
                "repair_keep_flags",
                (current_n * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );

            let target_packed = best_a | (best_b << 16);
            let replace_params = [current_n as u32, next_symbol, target_packed, 0u32];
            let replace_params_buf = self.create_buffer_init(
                "repair_replace_params",
                bytemuck::cast_slice(&replace_params),
                wgpu::BufferUsages::UNIFORM,
            );

            let replace_bg_layout = replace_pips.replace_twobuf.get_bind_group_layout(0);
            let replace_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("repair_replace_bg"),
                layout: &replace_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: symbols_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: symbols_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: keep_flags_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: replace_params_buf.as_entire_binding(),
                    },
                ],
            });

            self.dispatch(
                &replace_pips.replace_twobuf,
                &replace_bg,
                workgroups,
                "repair_replace_twobuf",
            )?;

            // --- Phase 4: Prefix sum on keep_flags for compaction offsets ---
            let mut prefix_sum_buf = self.create_buffer(
                "repair_prefix_sum",
                (current_n * 4) as u64,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            self.run_inclusive_prefix_sum(&keep_flags_buf, &mut prefix_sum_buf, current_n)?;

            // Read back last element of prefix sum to get new length.
            let new_n = self.read_buffer_scalar_u32(&prefix_sum_buf, current_n - 1) as usize;

            if new_n == 0 || new_n >= current_n {
                // No replacements happened or something went wrong — use CPU fallback.
                cpu_replace_bigram(&mut symbols, best_a, best_b, next_symbol);
                rules.push((next_symbol, best_a, best_b));
                next_symbol += 1;
                continue;
            }

            // --- Phase 5: Scatter compact ---
            let compacted_buf = self.create_buffer(
                "repair_compacted",
                (new_n * 4) as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );

            // Reuse params for scatter (n is current_n).
            let scatter_bg_layout = replace_pips.scatter.get_bind_group_layout(0);
            let scatter_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("repair_scatter_bg"),
                layout: &scatter_bg_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: symbols_out_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: compacted_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: prefix_sum_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: replace_params_buf.as_entire_binding(),
                    },
                ],
            });

            self.dispatch(
                &replace_pips.scatter,
                &scatter_bg,
                workgroups,
                "repair_scatter",
            )?;

            // Read back compacted symbols.
            let compacted_bytes = self.read_buffer(&compacted_buf, (new_n * 4) as u64);
            let compacted_u32s: &[u32] = bytemuck::cast_slice(&compacted_bytes);
            symbols = compacted_u32s.to_vec();

            rules.push((next_symbol, best_a, best_b));
            next_symbol += 1;
        }

        // Encode output (same format as CPU repair).
        Ok(crate::repair::encode_repair_output(&symbols, &rules))
    }
}

/// CPU fallback: find the most frequent bigram.
fn cpu_find_best_bigram(symbols: &[u32]) -> (u32, u32, u32) {
    use std::collections::HashMap;
    let mut counts: HashMap<(u32, u32), u32> = HashMap::new();
    for i in 0..symbols.len().saturating_sub(1) {
        *counts.entry((symbols[i], symbols[i + 1])).or_default() += 1;
    }
    let mut best = (0u32, 0u32, 0u32);
    for (&(a, b), &freq) in &counts {
        if freq > best.2 {
            best = (a, b, freq);
        }
    }
    best
}

/// CPU fallback: replace all non-overlapping occurrences of bigram.
fn cpu_replace_bigram(symbols: &mut Vec<u32>, target_a: u32, target_b: u32, new_symbol: u32) {
    let mut i = 0;
    let mut new_symbols = Vec::with_capacity(symbols.len());
    while i < symbols.len() {
        if i + 1 < symbols.len() && symbols[i] == target_a && symbols[i + 1] == target_b {
            new_symbols.push(new_symbol);
            i += 2;
        } else {
            new_symbols.push(symbols[i]);
            i += 1;
        }
    }
    *symbols = new_symbols;
}
