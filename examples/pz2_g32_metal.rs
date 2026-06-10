//! pz2-G32 stage-2 gate 2 + gate 3: Metal GPU decode of the 32-lane wire.
//!
//! Gate 2 — literal Huffman phase only (`kernels/pz2_g32_lit.metal`): one
//! simdgroup per tile, 32 symbols/round via simd_ballot + popcount-prefix
//! word fetch, 4 KB decode table in threadgroup memory, persistent shared
//! buffers (unified memory — no copies). Tiles are real Silesia pz2 blocks'
//! literal sections, re-cut into independent `--tile-lits` chunks (the same
//! per-block canonical table; GDeflate cuts tiles the same way). Kill line:
//! < 5 GB/s effective on-device. The decision-relevant comparison is the
//! NEON all-cores baseline over the SAME tiles, measured here too.
//!
//! Gate 3 — cooperative sequence splice (`kernels/pz2_g32_splice.metal`):
//! one threadgroup (one simdgroup) per pz2 block. Three concurrent serial
//! Huffman chains pre-decode the code lanes, then rounds of 32 sequences
//! resolve extras/positions with simd prefix sums and execute the copies
//! cooperatively. End-to-end GPU block decode = LIT_RAW blit + literal
//! dispatch + splice dispatch, vs CPU all-cores full pz2 decode.
//!
//! GPU timing: median + spread of `--reps` command buffers, each carrying
//! `--inner` back-to-back iterations (sustains DVFS clocks; GPUStartTime /
//! GPUEndTime via objc, so submission overhead is excluded); device init and
//! pipeline compile reported separately. Shared machine: per-rep sequences
//! are printed and spread > 15% is flagged.
//!
//! Usage:
//!   cargo run --release --no-default-features --example pz2_g32_metal -- \
//!     [--reps N] [--inner N] [--tile-lits N] <files...>

// objc's msg_send! internally tests cfg(feature = "cargo-clippy"), which this
// crate doesn't declare.
#![allow(unexpected_cfgs)]

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("pz2_g32_metal: macOS-only probe (Metal)");
}

#[cfg(target_os = "macos")]
fn main() {
    probe::run();
}

#[cfg(target_os = "macos")]
mod probe {
    use std::time::Instant;

    use metal::foreign_types::ForeignTypeRef;
    use metal::objc::runtime::Object;
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{Buffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize};
    use pz::lzseq::SeqConfig;
    use pz::pz2::SpikeSeqLane;

    const BLOCK: usize = 2 * 1024 * 1024; // shipped DEFAULT_PZ2_BLOCK_SIZE
    const SG_PER_TG: usize = 8; // must match kernels/pz2_g32_lit.metal
    const LANE_CONST: u32 = 0xFFFF_FFFF; // must match kernels/pz2_g32_splice.metal

    fn auto_greedy(block: &[u8]) -> bool {
        let p = pz::analysis::analyze(block);
        !(p.byte_entropy > 7.5 && p.match_density < 0.1)
    }

    fn median(xs: &mut [f64]) -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Tile {
        word_off: u32,  // index into words buffer (u32 units)
        out_off: u32,   // byte offset into lits buffer
        lit_count: u32, // 0 = padding tile
        table_off: u32, // index into tables buffer (u16 units)
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct SpliceLaneDesc {
        table_off: u32, // ushort index into seq tables, or LANE_CONST
        const_val: u32,
        bits_off: u32, // byte offset into seq_bits
        bits_len: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct SpliceBlockDesc {
        seq_count: u32,
        lit_off: u32,
        lit_count: u32,
        out_off: u32,
        out_len: u32,
        scratch_off: u32,
        ll: SpliceLaneDesc,
        of: SpliceLaneDesc,
        ml: SpliceLaneDesc,
        ex_off: u32,
        ex_len: u32,
    }

    /// GPU + wall times for a batch of command buffers (per-iteration secs).
    struct GpuTimes {
        gpu: Vec<f64>,
        wall: Vec<f64>,
    }

    enum Pass<'a> {
        /// (pipeline, buffers, n for set_bytes, threadgroups, threads/tg)
        Compute(&'a ComputePipelineState, Vec<&'a Buffer>, u32, usize, usize),
        /// (src, src_off, dst, dst_off, len)
        Blit(&'a Buffer, usize, &'a Buffer, usize, usize),
    }

    /// Run `inner` back-to-back iterations of `passes` per command buffer
    /// (sustains GPU load across DVFS), `reps` command buffers. Compute and
    /// blit passes get separate encoders, so Metal's hazard tracking orders
    /// them on shared buffers.
    fn run_passes(queue: &CommandQueue, passes: &[Pass], reps: usize, inner: usize) -> GpuTimes {
        let mut times = GpuTimes {
            gpu: Vec::new(),
            wall: Vec::new(),
        };
        for _ in 0..reps {
            let cmd = queue.new_command_buffer();
            for _ in 0..inner {
                for pass in passes {
                    match pass {
                        Pass::Compute(pipeline, buffers, n, groups, tg) => {
                            let enc = cmd.new_compute_command_encoder();
                            enc.set_compute_pipeline_state(pipeline);
                            for (i, b) in buffers.iter().enumerate() {
                                enc.set_buffer(i as u64, Some(b), 0);
                            }
                            enc.set_bytes(
                                buffers.len() as u64,
                                4,
                                n as *const u32 as *const std::ffi::c_void,
                            );
                            enc.dispatch_thread_groups(
                                MTLSize::new(*groups as u64, 1, 1),
                                MTLSize::new(*tg as u64, 1, 1),
                            );
                            enc.end_encoding();
                        }
                        Pass::Blit(src, so, dst, doff, len) => {
                            if *len > 0 {
                                let enc = cmd.new_blit_command_encoder();
                                enc.copy_from_buffer(
                                    src,
                                    *so as u64,
                                    dst,
                                    *doff as u64,
                                    *len as u64,
                                );
                                enc.end_encoding();
                            }
                        }
                    }
                }
            }
            let t = Instant::now();
            cmd.commit();
            cmd.wait_until_completed();
            times.wall.push(t.elapsed().as_secs_f64() / inner as f64);
            unsafe {
                let p = cmd.as_ptr() as *mut Object;
                let start: f64 = msg_send![p, GPUStartTime];
                let end: f64 = msg_send![p, GPUEndTime];
                times.gpu.push((end - start) / inner as f64);
            }
        }
        times
    }

    /// Print the per-rep sequence and the steady-state (last half) median.
    /// Returns the steady-state median seconds.
    fn report(label: &str, bytes: usize, times: &mut GpuTimes) -> f64 {
        let seq: Vec<String> = times
            .gpu
            .iter()
            .map(|t| format!("{:.2}", 1e3 * t))
            .collect();
        println!(
            "  [{label} per-rep GPU ms, submission order] {}",
            seq.join(" ")
        );
        let m_wall = median(&mut times.wall);
        let steady = times.gpu.len() / 2;
        let mut tail = times.gpu[steady..].to_vec();
        let m = median(&mut tail);
        let (lo, hi) = (
            tail.iter().cloned().fold(f64::INFINITY, f64::min),
            tail.iter().cloned().fold(0.0, f64::max),
        );
        let spread = 100.0 * (hi - lo) / m;
        println!(
            "  {label}: steady-state GPU {:.2} ms = {:.2} GB/s (wall median {:.2} ms), \
             spread {:.1}%{}",
            1e3 * m,
            bytes as f64 / m / 1e9,
            1e3 * m_wall,
            spread,
            if spread > 15.0 {
                "  [UNRELIABLE: spread > 15%]"
            } else {
                ""
            }
        );
        m
    }

    fn par_iter(n: usize, threads: usize, f: impl Fn(usize) + Sync) {
        let per = n.div_ceil(threads);
        std::thread::scope(|s| {
            for t in 0..threads {
                let f = &f;
                let lo = t * per;
                let hi = ((t + 1) * per).min(n);
                if lo < hi {
                    s.spawn(move || {
                        for i in lo..hi {
                            f(i);
                        }
                    });
                }
            }
        });
    }

    struct BlockRec {
        enc: Vec<u8>,
        orig_len: usize,
    }

    #[allow(clippy::too_many_lines)]
    pub fn run() {
        let mut reps = 9usize;
        let mut inner = 20usize;
        let mut tile_lits = 65536usize;
        let mut dup = 1usize;
        let mut files: Vec<String> = Vec::new();
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--reps" => reps = args.next().and_then(|v| v.parse().ok()).expect("--reps N"),
                "--inner" => inner = args.next().and_then(|v| v.parse().ok()).expect("--inner N"),
                "--tile-lits" => {
                    tile_lits = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .expect("--tile-lits N")
                }
                "--dup" => dup = args.next().and_then(|v| v.parse().ok()).expect("--dup N"),
                _ => files.push(a),
            }
        }
        if files.is_empty() {
            eprintln!("usage: pz2_g32_metal [--reps N] [--inner N] [--tile-lits N] <files...>");
            std::process::exit(2);
        }
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

        // ---- Build GPU data from real Silesia pz2 blocks ----
        // Literal phase: tiles over LIT_HUFF blocks. The lits buffer layout
        // is [all LIT_HUFF blocks' literals, packed][all LIT_RAW literals],
        // so the literal kernel writes the prefix and one blit stages the
        // raw tail.
        let mut corpus: Vec<u8> = Vec::new();
        let mut blocks: Vec<BlockRec> = Vec::new();
        let mut tables: Vec<u16> = Vec::new();
        let mut words: Vec<u32> = Vec::new();
        let mut tiles: Vec<Tile> = Vec::new();
        let mut huff_lits: Vec<u8> = Vec::new();
        // (block index, lengths, lits) deferred raw entries; lit_off assigned
        // after the huff prefix is final.
        let mut raw_lits: Vec<u8> = Vec::new();
        // Splice phase inputs.
        let mut seq_tables: Vec<u16> = Vec::new();
        let mut seq_bits: Vec<u8> = Vec::new();
        let mut descs: Vec<SpliceBlockDesc> = Vec::new();
        let mut lit_kind: Vec<(bool, u32)> = Vec::new(); // (is_huff, lit_off_within_kind)
        let mut scratch_bytes = 0usize;

        for path in &files {
            let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
            for blk in data.chunks(BLOCK) {
                let config = SeqConfig {
                    max_window: BLOCK,
                    greedy: auto_greedy(blk),
                    ..SeqConfig::default()
                };
                let enc = pz::pz2::encode_with_config(blk, &config).expect("encode");

                // Literal section.
                match pz::pz2::spike_lit_materials(&enc).expect("materials") {
                    Some((lengths, lits)) => {
                        let table_off = tables.len() as u32;
                        tables.extend(pz::pz2::spike_g32_decode_table(&lengths).expect("table"));
                        let block_first_tile = tiles.len();
                        lit_kind.push((true, huff_lits.len() as u32));
                        for chunk in lits.chunks(tile_lits.max(1)) {
                            let wb = pz::pz2::spike_g32_encode_words(chunk, &lengths)
                                .expect("encode g32");
                            let word_off = words.len() as u32;
                            words.extend(
                                wb.chunks_exact(4)
                                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                            );
                            tiles.push(Tile {
                                word_off,
                                out_off: huff_lits.len() as u32,
                                lit_count: chunk.len() as u32,
                                table_off,
                            });
                            huff_lits.extend_from_slice(chunk);
                        }
                        while !(tiles.len() - block_first_tile).is_multiple_of(SG_PER_TG) {
                            tiles.push(Tile {
                                word_off: 0,
                                out_off: 0,
                                lit_count: 0,
                                table_off,
                            });
                        }
                    }
                    None => {
                        let lits = pz::pz2::spike_decode_lits_pz2(&enc).expect("raw lits");
                        lit_kind.push((false, raw_lits.len() as u32));
                        raw_lits.extend_from_slice(&lits);
                    }
                }

                // Sequence section.
                let seq = pz::pz2::spike_seq_section(&enc).expect("seq section");
                let mut lane_descs = [SpliceLaneDesc::default(); 3];
                for (d, lane) in lane_descs.iter_mut().zip(seq.lanes.iter()) {
                    match lane {
                        SpikeSeqLane::Const(v) => {
                            d.table_off = LANE_CONST;
                            d.const_val = *v as u32;
                        }
                        SpikeSeqLane::Huff { table, bits } => {
                            d.table_off = seq_tables.len() as u32;
                            seq_tables.extend_from_slice(table);
                            d.bits_off = seq_bits.len() as u32;
                            d.bits_len = bits.len() as u32;
                            seq_bits.extend_from_slice(bits);
                        }
                    }
                }
                let ex_off = seq_bits.len() as u32;
                seq_bits.extend_from_slice(&seq.extras);
                descs.push(SpliceBlockDesc {
                    seq_count: seq.seq_count,
                    lit_off: 0, // patched below once huff prefix size is final
                    lit_count: seq.lit_count,
                    out_off: corpus.len() as u32,
                    out_len: blk.len() as u32,
                    scratch_off: scratch_bytes as u32,
                    ll: lane_descs[0],
                    of: lane_descs[1],
                    ml: lane_descs[2],
                    ex_off,
                    ex_len: seq.extras.len() as u32,
                });
                scratch_bytes += 3 * seq.seq_count as usize;

                blocks.push(BlockRec {
                    enc,
                    orig_len: blk.len(),
                });
                corpus.extend_from_slice(blk);
            }
        }
        seq_bits.extend_from_slice(&[0u8; 8]); // read_bits over-read pad
        let huff_lit_bytes = huff_lits.len();
        for (d, &(is_huff, off)) in descs.iter_mut().zip(lit_kind.iter()) {
            d.lit_off = if is_huff {
                off
            } else {
                huff_lit_bytes as u32 + off
            };
        }
        // --dup N: replicate every block descriptor with distinct output and
        // scratch regions (inputs shared). The 2 MiB-block wire caps a real
        // corpus at ~106 simdgroups, far below what the GPU needs to hide
        // its serial-chain latencies; dup measures the saturated throughput
        // a persistent-process embedding with more in-flight blocks would
        // see. CPU baseline work is scaled identically.
        let base_blocks = descs.len();
        let base_out = corpus.len();
        let base_scratch = scratch_bytes;
        for k in 1..dup {
            for i in 0..base_blocks {
                let mut d = descs[i];
                d.out_off += (k * base_out) as u32;
                d.scratch_off += (k * base_scratch) as u32;
                descs.push(d);
            }
        }
        scratch_bytes *= dup;
        let lits_total = huff_lit_bytes + raw_lits.len();
        let real_tiles = tiles.iter().filter(|t| t.lit_count > 0).count();
        let seq_total: u64 = descs.iter().map(|d| d.seq_count as u64).sum();
        println!(
            "corpus: {} blocks ({} LIT_HUFF), {real_tiles} tiles, {:.1} MB huff literals \
             (+{:.1} MB raw), {:.1} M sequences, {:.1} MB scratch",
            blocks.len(),
            lit_kind.iter().filter(|(h, _)| *h).count(),
            huff_lit_bytes as f64 / 1e6,
            raw_lits.len() as f64 / 1e6,
            seq_total as f64 / 1e6,
            scratch_bytes as f64 / 1e6,
        );
        assert!(real_tiles >= 100, "need >= 100 tiles for the gate");

        // ---- CPU NEON baseline over the SAME tiles (the honest denominator) ----
        let mut word_ends = vec![0u32; tiles.len()];
        {
            let mut next_end = words.len() as u32;
            for (i, t) in tiles.iter().enumerate().rev() {
                if t.lit_count > 0 {
                    word_ends[i] = next_end;
                    next_end = t.word_off;
                }
            }
        }
        let real_idx: Vec<usize> = tiles
            .iter()
            .enumerate()
            .filter(|(_, t)| t.lit_count > 0)
            .map(|(i, _)| i)
            .collect();
        let words_bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let out_neon: Vec<std::sync::Mutex<Vec<u8>>> = real_idx
            .iter()
            .map(|&i| std::sync::Mutex::new(vec![0u8; tiles[i].lit_count as usize]))
            .collect();
        let mut neon_ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            par_iter(real_idx.len(), threads, |i| {
                let tile = &tiles[real_idx[i]];
                let wlo = tile.word_off as usize * 4;
                let whi = word_ends[real_idx[i]] as usize * 4;
                let table = &tables[tile.table_off as usize..tile.table_off as usize + 2048];
                let mut out = out_neon[i].lock().unwrap();
                pz::pz2::spike_g32_decode_words_with_table(&words_bytes[wlo..whi], table, &mut out)
                    .expect("neon tile decode");
            });
            neon_ts.push(t.elapsed().as_secs_f64());
        }
        for (i, &ti) in real_idx.iter().enumerate() {
            let out = out_neon[i].lock().unwrap();
            let lo = tiles[ti].out_off as usize;
            assert_eq!(
                &out[..],
                &huff_lits[lo..lo + tiles[ti].lit_count as usize],
                "NEON tile {i} mismatch"
            );
        }
        let m_neon = median(&mut neon_ts);
        println!(
            "\nCPU NEON all-cores ({threads} threads), same tiles: {:.2} ms = {:.2} GB/s \
             (spread {:.1}%)",
            1e3 * m_neon,
            huff_lit_bytes as f64 / m_neon / 1e9,
            100.0
                * (neon_ts.iter().cloned().fold(0.0f64, f64::max)
                    - neon_ts.iter().cloned().fold(f64::INFINITY, f64::min))
                / m_neon,
        );

        // ---- CPU all-cores full pz2 decode (the gate-3 denominator) ----
        let work_bytes = corpus.len() * dup;
        assert!(
            work_bytes < u32::MAX as usize,
            "--dup too large for u32 offsets"
        );
        let mut cpu_ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            par_iter(blocks.len() * dup, threads, |i| {
                let b = &blocks[i % blocks.len()];
                let d = pz::pz2::decode(&b.enc, b.orig_len).expect("decode");
                std::hint::black_box(d);
            });
            cpu_ts.push(t.elapsed().as_secs_f64());
        }
        let m_cpu = median(&mut cpu_ts);
        println!(
            "CPU all-cores full pz2 decode (x{dup}): {:.2} ms = {:.2} GB/s (spread {:.1}%)",
            1e3 * m_cpu,
            work_bytes as f64 / m_cpu / 1e9,
            100.0
                * (cpu_ts.iter().cloned().fold(0.0f64, f64::max)
                    - cpu_ts.iter().cloned().fold(f64::INFINITY, f64::min))
                / m_cpu,
        );

        // ---- Metal setup (init timed separately) ----
        let t_init = Instant::now();
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let init_ms = 1e3 * t_init.elapsed().as_secs_f64();
        let t_compile = Instant::now();
        let compile = |src: &str, name: &str| -> ComputePipelineState {
            let lib = device
                .new_library_with_source(src, &metal::CompileOptions::new())
                .unwrap_or_else(|e| panic!("compile {name}: {e}"));
            let f = lib.get_function(name, None).expect("function");
            device
                .new_compute_pipeline_state_with_function(&f)
                .expect("pipeline")
        };
        let lit_pipe = compile(
            include_str!("../kernels/pz2_g32_lit.metal"),
            "g32_lit_decode",
        );
        let splice_pipe = compile(
            include_str!("../kernels/pz2_g32_splice.metal"),
            "g32_splice",
        );
        let compile_ms = 1e3 * t_compile.elapsed().as_secs_f64();
        println!(
            "\nMetal: device '{}' init {:.1} ms, kernel compile {:.1} ms (excluded from \
             steady-state numbers)",
            device.name(),
            init_ms,
            compile_ms
        );

        let opt = MTLResourceOptions::StorageModeShared;
        let as_bytes = |p: *const u8, len: usize| unsafe { std::slice::from_raw_parts(p, len) };
        let buf = |ptr: *const std::ffi::c_void, len: usize| -> Buffer {
            if len == 0 {
                device.new_buffer(4, opt)
            } else {
                device.new_buffer_with_data(ptr, len as u64, opt)
            }
        };
        let b_words = buf(words.as_ptr() as *const _, words.len() * 4);
        let b_tables = buf(tables.as_ptr() as *const _, tables.len() * 2);
        let b_tiles = buf(
            tiles.as_ptr() as *const _,
            std::mem::size_of_val(&tiles[..]),
        );
        let b_lits = device.new_buffer(lits_total.max(4) as u64, opt);
        let b_rawsrc = buf(raw_lits.as_ptr() as *const _, raw_lits.len());
        let b_seqbits = buf(seq_bits.as_ptr() as *const _, seq_bits.len());
        let b_seqtables = buf(seq_tables.as_ptr() as *const _, seq_tables.len() * 2);
        let b_descs = buf(
            descs.as_ptr() as *const _,
            std::mem::size_of_val(&descs[..]),
        );
        let b_codes = device.new_buffer(scratch_bytes.max(4) as u64, opt);
        let b_out = device.new_buffer(work_bytes as u64, opt);
        assert_eq!(std::mem::size_of::<SpliceBlockDesc>(), 80);

        let n_groups = tiles.len() / SG_PER_TG;
        let lit_pass = || {
            Pass::Compute(
                &lit_pipe,
                vec![&b_words, &b_tables, &b_tiles, &b_lits],
                tiles.len() as u32,
                n_groups,
                SG_PER_TG * 32,
            )
        };
        let splice_pass = || {
            Pass::Compute(
                &splice_pipe,
                vec![
                    &b_seqbits,
                    &b_seqtables,
                    &b_descs,
                    &b_lits,
                    &b_codes,
                    &b_out,
                ],
                descs.len() as u32,
                descs.len(),
                32,
            )
        };
        // Zero-length blits are invalid in Metal; substitute a no-op copy.
        let blit_pass = || {
            if raw_lits.is_empty() {
                Pass::Blit(&b_rawsrc, 0, &b_rawsrc, 0, 0)
            } else {
                Pass::Blit(&b_rawsrc, 0, &b_lits, huff_lit_bytes, raw_lits.len())
            }
        };

        // ---- Warm-up + round-trip verification (lit, then end-to-end) ----
        let _ = run_passes(&queue, &[lit_pass()], 1, 1);
        assert_eq!(
            as_bytes(b_lits.contents() as *const u8, huff_lit_bytes),
            &huff_lits[..],
            "GPU literal phase round-trip FAILED"
        );
        println!(
            "GPU literal round-trip: OK ({real_tiles} tiles, {:.1} MB byte-exact)",
            huff_lit_bytes as f64 / 1e6
        );
        let _ = run_passes(&queue, &[blit_pass(), lit_pass(), splice_pass()], 1, 1);
        assert_eq!(
            as_bytes(b_out.contents() as *const u8, corpus.len()),
            &corpus[..],
            "GPU end-to-end round-trip FAILED"
        );
        if dup > 1 {
            let last = as_bytes(
                (b_out.contents() as *const u8).wrapping_add((dup - 1) * corpus.len()),
                corpus.len(),
            );
            assert_eq!(last, &corpus[..], "GPU dup replica round-trip FAILED");
        }
        println!(
            "GPU end-to-end round-trip: OK ({} blocks, {:.1} MB byte-exact)",
            blocks.len(),
            corpus.len() as f64 / 1e6
        );

        // ---- Gate 2: literal phase ----
        println!(
            "\nGATE 2: literal Huffman phase, {n_groups} threadgroups x {} threads",
            SG_PER_TG * 32
        );
        let mut t2 = run_passes(&queue, &[lit_pass()], reps, inner);
        let m_lit = report("literal kernel", huff_lit_bytes, &mut t2);
        let gbs = huff_lit_bytes as f64 / m_lit / 1e9;
        println!(
            "  vs NEON all-cores same-wire: {:.2}x | vs 5 GB/s kill line: {}",
            m_neon / m_lit,
            if gbs < 5.0 { "KILL (< 5 GB/s)" } else { "PASS" }
        );

        // Diagnostic: phase A (3 serial Huffman chains) alone, to attribute
        // the splice cost between entropy decode and the copy loop.
        let phase_a_pipe = compile(
            &format!(
                "#define SKIP_PHASE_B 1\n{}",
                include_str!("../kernels/pz2_g32_splice.metal")
            ),
            "g32_splice",
        );
        let phase_a_pass = || {
            Pass::Compute(
                &phase_a_pipe,
                vec![
                    &b_seqbits,
                    &b_seqtables,
                    &b_descs,
                    &b_lits,
                    &b_codes,
                    &b_out,
                ],
                descs.len() as u32,
                descs.len(),
                32,
            )
        };
        let mut ta = run_passes(&queue, &[phase_a_pass()], reps.min(5), inner.min(5));
        let m_pa = report("splice phase A only", work_bytes, &mut ta);

        // ---- Gate 3: splice alone, then end-to-end ----
        println!(
            "\nGATE 3: cooperative splice, {} threadgroups x 32 threads",
            descs.len()
        );
        let splice_inner = inner.clamp(1, 5);
        let mut t3 = run_passes(&queue, &[splice_pass()], reps, splice_inner);
        let m_splice = report("splice kernel", work_bytes, &mut t3);
        let mut te = run_passes(
            &queue,
            &[blit_pass(), lit_pass(), splice_pass()],
            reps,
            splice_inner,
        );
        let m_e2e = report("end-to-end (blit+lit+splice)", work_bytes, &mut te);
        println!(
            "  splice/lit split: {:.2} ms / {:.2} ms (phase A alone {:.2} ms = {:.0}% of \
             splice) | GPU end-to-end vs CPU all-cores: {:.2}x (CPU {:.2} ms, GPU {:.2} ms)",
            1e3 * m_splice,
            1e3 * m_lit,
            1e3 * m_pa,
            100.0 * m_pa / m_splice,
            m_cpu / m_e2e,
            1e3 * m_cpu,
            1e3 * m_e2e,
        );
    }
}
