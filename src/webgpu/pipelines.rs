//! Lazily-compiled GPU compute pipeline groups.
//!
//! Each WGSL shader module has a corresponding pipeline group struct that holds
//! the compiled `wgpu::ComputePipeline` handles. Pipelines are compiled on
//! first use via `OnceLock`, not at engine creation time, so startup cost
//! scales with actual usage.

use super::kernels::*;
use super::WebGpuEngine;

// ---------------------------------------------------------------------------
// Pipeline group structs — one per WGSL shader module.
// ---------------------------------------------------------------------------

/// LZ77 top-K pipeline (1 pipeline from lz77_topk.wgsl).
pub(super) struct Lz77TopkPipelines {
    pub(super) topk: wgpu::ComputePipeline,
}

/// LZ77 hash-table pipelines (2 pipelines from lz77_hash.wgsl).
pub(super) struct Lz77HashPipelines {
    pub(super) build: wgpu::ComputePipeline,
    pub(super) find: wgpu::ComputePipeline,
}

/// LZ77 lazy-matching pipelines (2 pipelines from lz77_lazy.wgsl).
/// Retained for A/B benchmarking against the coop kernel.
#[allow(dead_code)]
pub(super) struct Lz77LazyPipelines {
    pub(super) find: wgpu::ComputePipeline,
    pub(super) resolve: wgpu::ComputePipeline,
}

/// LZ77 cooperative-stitch pipelines (2 pipelines from lz77_coop.wgsl).
pub(super) struct Lz77CoopPipelines {
    pub(super) find: wgpu::ComputePipeline,
    pub(super) resolve: wgpu::ComputePipeline,
}

/// BWT rank pipelines (4 pipelines from bwt_rank.wgsl).
pub(super) struct BwtRankPipelines {
    pub(super) rank_compare: wgpu::ComputePipeline,
    pub(super) prefix_sum_local: wgpu::ComputePipeline,
    pub(super) prefix_sum_propagate: wgpu::ComputePipeline,
    pub(super) rank_scatter: wgpu::ComputePipeline,
}

/// BWT radix sort pipelines (4 pipelines from bwt_radix.wgsl).
pub(super) struct BwtRadixPipelines {
    pub(super) compute_keys: wgpu::ComputePipeline,
    pub(super) histogram: wgpu::ComputePipeline,
    pub(super) inclusive_to_exclusive: wgpu::ComputePipeline,
    pub(super) scatter: wgpu::ComputePipeline,
}

/// Huffman encoding pipelines (5 pipelines from huffman_encode.wgsl).
pub(super) struct HuffmanPipelines {
    pub(super) byte_histogram: wgpu::ComputePipeline,
    pub(super) compute_bit_lengths: wgpu::ComputePipeline,
    pub(super) write_codes: wgpu::ComputePipeline,
    pub(super) prefix_sum_block: wgpu::ComputePipeline,
    pub(super) prefix_sum_apply: wgpu::ComputePipeline,
}

/// FSE decode pipeline (1 pipeline from fse_decode.wgsl).
pub(super) struct FseDecodePipelines {
    pub(super) decode: wgpu::ComputePipeline,
}

/// FSE encode pipeline (1 pipeline from fse_encode.wgsl).
pub(super) struct FseEncodePipelines {
    pub(super) encode: wgpu::ComputePipeline,
}

/// LZ77 block-parallel decode pipeline (1 pipeline from lz77_decode.wgsl).
pub(super) struct Lz77DecodePipelines {
    pub(super) decode: wgpu::ComputePipeline,
}

/// rANS decode pipelines (lane-specialized entries from rans_decode.wgsl).
pub(super) struct RansDecodePipelines {
    pub(super) decode_wg4: wgpu::ComputePipeline,
    pub(super) decode_wg8: wgpu::ComputePipeline,
    pub(super) decode_wg64: wgpu::ComputePipeline,
    pub(super) decode_packed: wgpu::ComputePipeline,
}

/// rANS encode pipelines (lane-specialized entries from rans_encode.wgsl).
pub(super) struct RansEncodePipelines {
    pub(super) encode_wg4: wgpu::ComputePipeline,
    pub(super) encode_wg8: wgpu::ComputePipeline,
    pub(super) encode_wg64: wgpu::ComputePipeline,
    pub(super) encode_packed: wgpu::ComputePipeline,
}

/// LzSeq demux pipeline (1 pipeline from lzseq_demux.wgsl).
pub(super) struct LzSeqPipelines {
    pub(super) demux: wgpu::ComputePipeline,
}

/// Parlz conflict resolution pipelines (Experiment E).
pub(super) struct ParlzPipelines {
    pub(super) init_coverage: wgpu::ComputePipeline,
    pub(super) prefix_max_local: wgpu::ComputePipeline,
    pub(super) prefix_max_propagate: wgpu::ComputePipeline,
    pub(super) classify: wgpu::ComputePipeline,
}

/// SortLZ radix sort + match verification pipelines (Experiment B).
pub(super) struct SortLzPipelines {
    pub(super) compute_keys: wgpu::ComputePipeline,
    pub(super) verify_matches: wgpu::ComputePipeline,
}

// ---------------------------------------------------------------------------
// Lazy pipeline accessors — compile on first use via OnceLock.
// ---------------------------------------------------------------------------

impl WebGpuEngine {
    /// Helper: create a shader module + compute pipeline from WGSL source.
    pub(super) fn make_pipeline(
        &self,
        label: &str,
        source: &str,
        entry: &str,
    ) -> wgpu::ComputePipeline {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
    }

    pub(super) fn pipeline_lz77_topk(&self) -> &wgpu::ComputePipeline {
        &self
            .lz77_topk
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = Lz77TopkPipelines {
                    topk: self.make_pipeline("lz77_topk", LZ77_TOPK_KERNEL_SOURCE, "encode_topk"),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile lz77_topk.wgsl: {ms:.3} ms");
                }
                group
            })
            .topk
    }

    pub(super) fn lz77_hash_pipelines(&self) -> &Lz77HashPipelines {
        self.lz77_hash.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lz77_hash"),
                    source: wgpu::ShaderSource::Wgsl(LZ77_HASH_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = Lz77HashPipelines {
                build: make("lz77_hash_build", "build_hash_table"),
                find: make("lz77_hash_find", "find_matches"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile lz77_hash.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    pub(super) fn pipeline_lz77_hash_build(&self) -> &wgpu::ComputePipeline {
        &self.lz77_hash_pipelines().build
    }

    pub(super) fn pipeline_lz77_hash_find(&self) -> &wgpu::ComputePipeline {
        &self.lz77_hash_pipelines().find
    }

    #[allow(dead_code)]
    pub(super) fn lz77_lazy_pipelines(&self) -> &Lz77LazyPipelines {
        self.lz77_lazy.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lz77_lazy"),
                    source: wgpu::ShaderSource::Wgsl(LZ77_LAZY_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = Lz77LazyPipelines {
                find: make("lz77_lazy_find", "find_matches"),
                resolve: make("lz77_lazy_resolve", "resolve_lazy"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile lz77_lazy.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    #[allow(dead_code)]
    pub(super) fn pipeline_lz77_lazy_find(&self) -> &wgpu::ComputePipeline {
        &self.lz77_lazy_pipelines().find
    }

    #[allow(dead_code)]
    pub(super) fn pipeline_lz77_lazy_resolve(&self) -> &wgpu::ComputePipeline {
        &self.lz77_lazy_pipelines().resolve
    }

    pub(super) fn lz77_coop_pipelines(&self) -> &Lz77CoopPipelines {
        self.lz77_coop.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("lz77_coop"),
                    source: wgpu::ShaderSource::Wgsl(LZ77_COOP_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = Lz77CoopPipelines {
                find: make("lz77_coop_find", "find_matches_coop"),
                resolve: make("lz77_coop_resolve", "resolve_lazy"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile lz77_coop.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    pub(super) fn pipeline_lz77_coop_find(&self) -> &wgpu::ComputePipeline {
        &self.lz77_coop_pipelines().find
    }

    pub(super) fn pipeline_lz77_coop_resolve(&self) -> &wgpu::ComputePipeline {
        &self.lz77_coop_pipelines().resolve
    }

    pub(super) fn bwt_rank_pipelines(&self) -> &BwtRankPipelines {
        self.bwt_rank.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("bwt_rank"),
                    source: wgpu::ShaderSource::Wgsl(BWT_RANK_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = BwtRankPipelines {
                rank_compare: make("rank_compare", "rank_compare"),
                prefix_sum_local: make("prefix_sum_local", "prefix_sum_local"),
                prefix_sum_propagate: make("prefix_sum_propagate", "prefix_sum_propagate"),
                rank_scatter: make("rank_scatter", "rank_scatter"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile bwt_rank.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    pub(super) fn pipeline_rank_compare(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().rank_compare
    }

    pub(super) fn pipeline_prefix_sum_local(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().prefix_sum_local
    }

    pub(super) fn pipeline_prefix_sum_propagate(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().prefix_sum_propagate
    }

    pub(super) fn pipeline_rank_scatter(&self) -> &wgpu::ComputePipeline {
        &self.bwt_rank_pipelines().rank_scatter
    }

    pub(super) fn bwt_radix_pipelines(&self) -> &BwtRadixPipelines {
        self.bwt_radix.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("bwt_radix"),
                    source: wgpu::ShaderSource::Wgsl(BWT_RADIX_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = BwtRadixPipelines {
                compute_keys: make("radix_compute_keys", "radix_compute_keys"),
                histogram: make("radix_histogram", "radix_histogram"),
                inclusive_to_exclusive: make("inclusive_to_exclusive", "inclusive_to_exclusive"),
                scatter: make("radix_scatter", "radix_scatter"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile bwt_radix.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    pub(super) fn pipeline_radix_compute_keys(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().compute_keys
    }

    pub(super) fn pipeline_radix_histogram(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().histogram
    }

    pub(super) fn pipeline_inclusive_to_exclusive(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().inclusive_to_exclusive
    }

    pub(super) fn pipeline_radix_scatter(&self) -> &wgpu::ComputePipeline {
        &self.bwt_radix_pipelines().scatter
    }

    pub(super) fn huffman_pipelines(&self) -> &HuffmanPipelines {
        self.huffman.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("huffman_encode"),
                    source: wgpu::ShaderSource::Wgsl(HUFFMAN_ENCODE_KERNEL_SOURCE.into()),
                });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: None,
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = HuffmanPipelines {
                byte_histogram: make("byte_histogram", "byte_histogram"),
                compute_bit_lengths: make("compute_bit_lengths", "compute_bit_lengths"),
                write_codes: make("write_codes", "write_codes"),
                prefix_sum_block: make("prefix_sum_block", "prefix_sum_block"),
                prefix_sum_apply: make("prefix_sum_apply", "prefix_sum_apply"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile huffman_encode.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    pub(super) fn pipeline_byte_histogram(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().byte_histogram
    }

    pub(super) fn pipeline_compute_bit_lengths(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().compute_bit_lengths
    }

    pub(super) fn pipeline_write_codes(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().write_codes
    }

    pub(super) fn pipeline_prefix_sum_block(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().prefix_sum_block
    }

    pub(super) fn pipeline_prefix_sum_apply(&self) -> &wgpu::ComputePipeline {
        &self.huffman_pipelines().prefix_sum_apply
    }

    pub(super) fn pipeline_fse_decode(&self) -> &wgpu::ComputePipeline {
        &self
            .fse_decode
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = FseDecodePipelines {
                    decode: self.make_pipeline(
                        "fse_decode",
                        FSE_DECODE_KERNEL_SOURCE,
                        "fse_decode",
                    ),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile fse_decode.wgsl: {ms:.3} ms");
                }
                group
            })
            .decode
    }

    pub(super) fn pipeline_lz77_decode(&self) -> &wgpu::ComputePipeline {
        &self
            .lz77_decode
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = Lz77DecodePipelines {
                    decode: self.make_pipeline(
                        "lz77_decode",
                        LZ77_DECODE_KERNEL_SOURCE,
                        "lz77_decode",
                    ),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile lz77_decode.wgsl: {ms:.3} ms");
                }
                group
            })
            .decode
    }

    pub(super) fn pipeline_fse_encode(&self) -> &wgpu::ComputePipeline {
        &self
            .fse_encode
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = FseEncodePipelines {
                    encode: self.make_pipeline(
                        "fse_encode",
                        FSE_ENCODE_KERNEL_SOURCE,
                        "fse_encode",
                    ),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile fse_encode.wgsl: {ms:.3} ms");
                }
                group
            })
            .encode
    }

    pub(super) fn pipeline_rans_decode_for_lanes(
        &self,
        num_lanes: usize,
    ) -> &wgpu::ComputePipeline {
        let group = self.rans_decode.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let group = RansDecodePipelines {
                decode_wg4: self.make_pipeline(
                    "rans_decode_wg4",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk_wg4",
                ),
                decode_wg8: self.make_pipeline(
                    "rans_decode_wg8",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk_wg8",
                ),
                decode_wg64: self.make_pipeline(
                    "rans_decode_wg64",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk",
                ),
                decode_packed: self.make_pipeline(
                    "rans_decode_packed",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk_packed",
                ),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile rans_decode.wgsl: {ms:.3} ms");
            }
            group
        });
        if num_lanes <= 4 {
            &group.decode_wg4
        } else if num_lanes <= 8 {
            &group.decode_wg8
        } else {
            &group.decode_wg64
        }
    }

    pub(super) fn pipeline_rans_decode_packed(&self) -> &wgpu::ComputePipeline {
        &self
            .rans_decode
            .get_or_init(|| RansDecodePipelines {
                decode_wg4: self.make_pipeline(
                    "rans_decode_wg4",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk_wg4",
                ),
                decode_wg8: self.make_pipeline(
                    "rans_decode_wg8",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk_wg8",
                ),
                decode_wg64: self.make_pipeline(
                    "rans_decode_wg64",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk",
                ),
                decode_packed: self.make_pipeline(
                    "rans_decode_packed",
                    RANS_DECODE_KERNEL_SOURCE,
                    "rans_decode_chunk_packed",
                ),
            })
            .decode_packed
    }

    pub(super) fn pipeline_rans_encode_for_lanes(
        &self,
        num_lanes: usize,
    ) -> &wgpu::ComputePipeline {
        let group = self.rans_encode.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let group = RansEncodePipelines {
                encode_wg4: self.make_pipeline(
                    "rans_encode_wg4",
                    RANS_ENCODE_KERNEL_SOURCE,
                    "rans_encode_chunk_wg4",
                ),
                encode_wg8: self.make_pipeline(
                    "rans_encode_wg8",
                    RANS_ENCODE_KERNEL_SOURCE,
                    "rans_encode_chunk_wg8",
                ),
                encode_wg64: self.make_pipeline(
                    "rans_encode_wg64",
                    RANS_ENCODE_KERNEL_SOURCE,
                    "rans_encode_chunk",
                ),
                encode_packed: self.make_pipeline(
                    "rans_encode_packed",
                    RANS_ENCODE_KERNEL_SOURCE,
                    "rans_encode_chunk_packed",
                ),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile rans_encode.wgsl: {ms:.3} ms");
            }
            group
        });
        if num_lanes <= 4 {
            &group.encode_wg4
        } else if num_lanes <= 8 {
            &group.encode_wg8
        } else {
            &group.encode_wg64
        }
    }

    pub(super) fn pipeline_rans_encode_packed(&self) -> &wgpu::ComputePipeline {
        &self
            .rans_encode
            .get_or_init(|| {
                // This path shouldn't normally be hit since pipeline_rans_encode_for_lanes
                // initializes the same OnceLock, but we need it for completeness.
                RansEncodePipelines {
                    encode_wg4: self.make_pipeline(
                        "rans_encode_wg4",
                        RANS_ENCODE_KERNEL_SOURCE,
                        "rans_encode_chunk_wg4",
                    ),
                    encode_wg8: self.make_pipeline(
                        "rans_encode_wg8",
                        RANS_ENCODE_KERNEL_SOURCE,
                        "rans_encode_chunk_wg8",
                    ),
                    encode_wg64: self.make_pipeline(
                        "rans_encode_wg64",
                        RANS_ENCODE_KERNEL_SOURCE,
                        "rans_encode_chunk",
                    ),
                    encode_packed: self.make_pipeline(
                        "rans_encode_packed",
                        RANS_ENCODE_KERNEL_SOURCE,
                        "rans_encode_chunk_packed",
                    ),
                }
            })
            .encode_packed
    }

    pub(crate) fn pipeline_lzseq_demux(&self) -> &wgpu::ComputePipeline {
        &self
            .lzseq_demux
            .get_or_init(|| {
                let t0 = std::time::Instant::now();
                let group = LzSeqPipelines {
                    demux: self.make_pipeline(
                        "lzseq_demux",
                        LZSEQ_DEMUX_KERNEL_SOURCE,
                        "lzseq_demux",
                    ),
                };
                if self.profiling {
                    let ms = t0.elapsed().as_secs_f64() * 1000.0;
                    eprintln!("[pz-gpu] compile lzseq_demux.wgsl: {ms:.3} ms");
                }
                group
            })
            .demux
    }

    // --- Experiment E: Parlz conflict resolution ---

    pub(super) fn parlz_pipelines(&self) -> &ParlzPipelines {
        self.parlz.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("parlz_resolve"),
                    source: wgpu::ShaderSource::Wgsl(PARLZ_RESOLVE_KERNEL_SOURCE.into()),
                });
            // Explicit bind group layout: all entry points share the same 5 bindings
            // even though individual entry points may not reference all of them.
            // Auto-derived layouts only include referenced bindings, causing mismatches.
            let bgl = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("parlz_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("parlz_pipeline_layout"),
                        bind_group_layouts: &[&bgl],
                        push_constant_ranges: &[],
                    });
            let make = |label, entry| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            };
            let group = ParlzPipelines {
                init_coverage: make("parlz_init_coverage", "init_coverage"),
                prefix_max_local: make("parlz_prefix_max_local", "prefix_max_local"),
                prefix_max_propagate: make("parlz_prefix_max_propagate", "prefix_max_propagate"),
                classify: make("parlz_classify", "classify"),
            };
            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile parlz_resolve.wgsl: {ms:.3} ms");
            }
            group
        })
    }

    // --- Experiment B: SortLZ radix sort + match verification ---

    pub(super) fn pipeline_sortlz_compute_keys(&self) -> &wgpu::ComputePipeline {
        &self.sortlz_pipelines().compute_keys
    }

    pub(super) fn pipeline_sortlz_verify_matches(&self) -> &wgpu::ComputePipeline {
        &self.sortlz_pipelines().verify_matches
    }

    pub(super) fn sortlz_pipelines(&self) -> &SortLzPipelines {
        self.sortlz.get_or_init(|| {
            let t0 = std::time::Instant::now();
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("sortlz_ops"),
                    source: wgpu::ShaderSource::Wgsl(SORTLZ_OPS_KERNEL_SOURCE.into()),
                });
            // Key extraction: same binding pattern as FWST (sa RO, hashes RO, keys RW, params).
            // Uses auto-derived layout (layout: None) since all bindings are referenced.
            let compute_keys =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("sortlz_compute_keys"),
                        layout: None,
                        module: &module,
                        entry_point: Some("sortlz_compute_keys"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            // Match verification needs 5 bindings with explicit layout
            // (sa RO, hashes RO, input RO, best RW+atomic, params).
            let vm_bgl = self
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("sortlz_vm_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
            let vm_layout = self
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("sortlz_vm_layout"),
                    bind_group_layouts: &[&vm_bgl],
                    push_constant_ranges: &[],
                });
            let verify_matches =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("sortlz_verify_matches"),
                        layout: Some(&vm_layout),
                        module: &module,
                        entry_point: Some("sortlz_verify_matches"),
                        compilation_options: Default::default(),
                        cache: None,
                    });

            if self.profiling {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                eprintln!("[pz-gpu] compile sortlz_ops.wgsl: {ms:.3} ms");
            }

            SortLzPipelines {
                compute_keys,
                verify_matches,
            }
        })
    }
}
