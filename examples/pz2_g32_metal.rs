//! pz2-G32 stage-2 gate 2 (+ gate 3): Metal GPU decode of the 32-lane wire.
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
//! one threadgroup per pz2 block; lane 0 walks the three Huffman sequence
//! lanes + extras serially (the wire's serial residue) while all 32 lanes
//! execute each literal-run/match copy cooperatively. End-to-end GPU block
//! decode = literal dispatch + splice dispatch, vs CPU all-cores pz2.
//!
//! GPU timing: median + spread of `--reps` command buffers (GPUStartTime /
//! GPUEndTime via objc, so submission overhead is excluded); device init and
//! pipeline compile reported separately. Run on a quiet machine if possible;
//! spread > 15% is flagged.
//!
//! Usage:
//!   cargo run --release --no-default-features --example pz2_g32_metal -- \
//!     [--reps N] [--tile-lits N] <files...>

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

    const BLOCK: usize = 2 * 1024 * 1024; // shipped DEFAULT_PZ2_BLOCK_SIZE
    const SG_PER_TG: usize = 8; // must match kernels/pz2_g32_lit.metal

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
        out_off: u32,   // byte offset into out buffer
        lit_count: u32, // 0 = padding tile
        table_off: u32, // index into tables buffer (u16 units)
    }

    struct GpuTimes {
        /// GPU-side seconds (GPUEndTime - GPUStartTime).
        gpu: Vec<f64>,
        /// Host wall seconds around commit + wait.
        wall: Vec<f64>,
    }

    fn run_kernel(
        queue: &CommandQueue,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        n_tiles: u32,
        n_groups: usize,
        tg_threads: usize,
        reps: usize,
        inner: usize,
    ) -> GpuTimes {
        let mut times = GpuTimes {
            gpu: Vec::new(),
            wall: Vec::new(),
        };
        for _ in 0..reps {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipeline);
            for (i, b) in buffers.iter().enumerate() {
                enc.set_buffer(i as u64, Some(b), 0);
            }
            enc.set_bytes(
                buffers.len() as u64,
                4,
                &n_tiles as *const u32 as *const std::ffi::c_void,
            );
            // `inner` back-to-back dispatches per command buffer: sustains
            // GPU load across DVFS so the per-dispatch time reflects a busy
            // device rather than a single short blip on a ramping clock.
            for _ in 0..inner {
                enc.dispatch_thread_groups(
                    MTLSize::new(n_groups as u64, 1, 1),
                    MTLSize::new(tg_threads as u64, 1, 1),
                );
            }
            enc.end_encoding();
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

    fn report(label: &str, bytes: usize, times: &mut GpuTimes) -> f64 {
        // GPU clocks ramp under load: print the rep sequence so ramp-up is
        // visible, then report the steady-state (last half) median.
        let seq: Vec<String> = times
            .gpu
            .iter()
            .map(|t| format!("{:.2}", 1e3 * t))
            .collect();
        println!(
            "  [{label} per-rep GPU ms, submission order] {}",
            seq.join(" ")
        );
        let steady = times.gpu.len() / 2;
        let mut tail_gpu = times.gpu[steady..].to_vec();
        let m_steady = median(&mut tail_gpu);
        let lo_s = tail_gpu.first().copied().unwrap_or(0.0);
        let hi_s = tail_gpu.last().copied().unwrap_or(0.0);
        println!(
            "  {label} steady-state (last {} reps): GPU {:.2} ms = {:.2} GB/s, spread {:.1}%{}",
            tail_gpu.len(),
            1e3 * m_steady,
            bytes as f64 / m_steady / 1e9,
            100.0 * (hi_s - lo_s) / m_steady,
            if (hi_s - lo_s) / m_steady > 0.15 {
                "  [UNRELIABLE: spread > 15%]"
            } else {
                ""
            }
        );
        let m_gpu = median(&mut times.gpu);
        let m_wall = median(&mut times.wall);
        let lo = times.gpu.first().copied().unwrap_or(0.0);
        let hi = times.gpu.last().copied().unwrap_or(0.0);
        let spread = 100.0 * (hi - lo) / m_gpu;
        println!(
            "  {label}: GPU {:>7.2} ms ({:>6.2} GB/s effective), wall {:>7.2} ms, spread {:.1}%{}",
            1e3 * m_gpu,
            bytes as f64 / m_gpu / 1e9,
            1e3 * m_wall,
            spread,
            if spread > 15.0 {
                "  [UNRELIABLE: spread > 15%]"
            } else {
                ""
            }
        );
        m_steady
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

    pub fn run() {
        let mut reps = 7usize;
        let mut inner = 20usize;
        let mut tile_lits = 65536usize;
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
                _ => files.push(a),
            }
        }
        if files.is_empty() {
            eprintln!("usage: pz2_g32_metal [--reps N] [--tile-lits N] <files...>");
            std::process::exit(2);
        }
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get());

        // ---- Build tiles from real Silesia pz2 blocks ----
        let mut tables: Vec<u16> = Vec::new();
        let mut words: Vec<u32> = Vec::new();
        let mut tiles: Vec<Tile> = Vec::new();
        let mut expected: Vec<u8> = Vec::new(); // concatenated tile outputs
        let mut n_blocks = 0usize;
        let mut raw_lit_bytes = 0usize;

        for path in &files {
            let data = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
            for blk in data.chunks(BLOCK) {
                let config = SeqConfig {
                    max_window: BLOCK,
                    greedy: auto_greedy(blk),
                    ..SeqConfig::default()
                };
                let enc = pz::pz2::encode_with_config(blk, &config).expect("encode");
                let Some((lengths, lits)) = pz::pz2::spike_lit_materials(&enc).expect("materials")
                else {
                    raw_lit_bytes += blk.len().min(BLOCK); // LIT_RAW: GPU memcpy phase
                    continue;
                };
                n_blocks += 1;
                let table_off = tables.len() as u32;
                tables.extend(pz::pz2::spike_g32_decode_table(&lengths).expect("table"));
                let block_first_tile = tiles.len();
                for chunk in lits.chunks(tile_lits) {
                    let wb = pz::pz2::spike_g32_encode_words(chunk, &lengths).expect("encode g32");
                    assert_eq!(wb.len() % 4, 0);
                    let word_off = words.len() as u32;
                    words.extend(
                        wb.chunks_exact(4)
                            .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
                    );
                    tiles.push(Tile {
                        word_off,
                        out_off: expected.len() as u32,
                        lit_count: chunk.len() as u32,
                        table_off,
                    });
                    expected.extend_from_slice(chunk);
                }
                // Pad this block's tile group to a SG_PER_TG multiple so every
                // threadgroup's tiles share one decode table.
                while (tiles.len() - block_first_tile) % SG_PER_TG != 0 {
                    tiles.push(Tile {
                        word_off: 0,
                        out_off: 0,
                        lit_count: 0,
                        table_off,
                    });
                }
            }
        }
        let lit_bytes = expected.len();
        let real_tiles = tiles.iter().filter(|t| t.lit_count > 0).count();
        println!(
            "corpus: {n_blocks} LIT_HUFF blocks, {real_tiles} tiles ({} padded slots), \
             {:.1} MB literals via Huffman ({:.1} MB LIT_RAW skipped), {:.1} MB words, \
             {} tables",
            tiles.len() - real_tiles,
            lit_bytes as f64 / 1e6,
            raw_lit_bytes as f64 / 1e6,
            words.len() as f64 * 4.0 / 1e6,
            tables.len() / 2048,
        );
        assert!(real_tiles >= 100, "need >= 100 tiles for the gate");

        // ---- CPU NEON baseline over the SAME tiles (the honest denominator) ----
        let real: Vec<&Tile> = tiles.iter().filter(|t| t.lit_count > 0).collect();
        // Per-tile word lengths: distance to next tile's word_off (tiles are
        // emitted in word order), last tile runs to the end.
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
        let mut neon_ts = Vec::with_capacity(reps);
        let out_neon: Vec<std::sync::Mutex<Vec<u8>>> = real
            .iter()
            .map(|t| std::sync::Mutex::new(vec![0u8; t.lit_count as usize]))
            .collect();
        for _ in 0..reps {
            let t = Instant::now();
            par_iter(real.len(), threads, |i| {
                let ti = real_idx[i];
                let tile = &tiles[ti];
                let wlo = tile.word_off as usize * 4;
                let whi = word_ends[ti] as usize * 4;
                let table = &tables[tile.table_off as usize..tile.table_off as usize + 2048];
                let mut out = out_neon[i].lock().unwrap();
                pz::pz2::spike_g32_decode_words_with_table(&words_bytes[wlo..whi], table, &mut out)
                    .expect("neon tile decode");
            });
            neon_ts.push(t.elapsed().as_secs_f64());
        }
        for (i, t) in real.iter().enumerate() {
            let out = out_neon[i].lock().unwrap();
            let lo = t.out_off as usize;
            assert_eq!(
                &out[..],
                &expected[lo..lo + t.lit_count as usize],
                "NEON tile {i} mismatch"
            );
        }
        let m_neon = median(&mut neon_ts);
        println!(
            "\nCPU NEON all-cores ({threads} threads), same tiles: {:.2} ms = {:.2} GB/s \
             (spread {:.1}%)",
            1e3 * m_neon,
            lit_bytes as f64 / m_neon / 1e9,
            100.0
                * (neon_ts.iter().cloned().fold(0.0f64, f64::max)
                    - neon_ts.iter().cloned().fold(f64::INFINITY, f64::min))
                / m_neon,
        );

        // ---- Metal setup (init timed separately) ----
        let t_init = Instant::now();
        let device = Device::system_default().expect("no Metal device");
        let queue = device.new_command_queue();
        let init_ms = 1e3 * t_init.elapsed().as_secs_f64();
        let t_compile = Instant::now();
        let src = include_str!("../kernels/pz2_g32_lit.metal");
        let lib = device
            .new_library_with_source(src, &metal::CompileOptions::new())
            .expect("compile MSL");
        let f = lib.get_function("g32_lit_decode", None).expect("function");
        let pipeline = device
            .new_compute_pipeline_state_with_function(&f)
            .expect("pipeline");
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
        let b_words =
            device.new_buffer_with_data(words.as_ptr() as *const _, (words.len() * 4) as u64, opt);
        let b_tables = device.new_buffer_with_data(
            tables.as_ptr() as *const _,
            (tables.len() * 2) as u64,
            opt,
        );
        let b_tiles = device.new_buffer_with_data(
            tiles.as_ptr() as *const _,
            std::mem::size_of_val(&tiles[..]) as u64,
            opt,
        );
        let b_out = device.new_buffer(lit_bytes as u64, opt);

        let n_groups = tiles.len() / SG_PER_TG;
        let n_tiles = tiles.len() as u32;

        // Warm-up + round-trip verification.
        let _ = run_kernel(
            &queue,
            &pipeline,
            &[&b_words, &b_tables, &b_tiles, &b_out],
            n_tiles,
            n_groups,
            SG_PER_TG * 32,
            1,
            1,
        );
        let gpu_out = as_bytes(b_out.contents() as *const u8, lit_bytes);
        assert_eq!(
            gpu_out,
            &expected[..],
            "GPU literal phase round-trip FAILED"
        );
        println!(
            "GPU literal round-trip: OK ({} tiles, {:.1} MB byte-exact)",
            real_tiles,
            lit_bytes as f64 / 1e6
        );

        // ---- Gate 2 measurement ----
        println!(
            "\nGATE 2: literal Huffman phase, {n_groups} threadgroups x {} threads, {reps} reps",
            SG_PER_TG * 32
        );
        let mut times = run_kernel(
            &queue,
            &pipeline,
            &[&b_words, &b_tables, &b_tiles, &b_out],
            n_tiles,
            n_groups,
            SG_PER_TG * 32,
            reps,
            inner,
        );
        times.gpu.sort_by(|a, b| a.partial_cmp(b).unwrap());
        times.wall.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m_gpu = report("literal kernel", lit_bytes, &mut times);
        let gbs = lit_bytes as f64 / m_gpu / 1e9;
        println!(
            "  vs NEON all-cores same-wire: {:.2}x | vs 5 GB/s kill line: {}",
            m_neon / m_gpu,
            if gbs < 5.0 { "KILL (< 5 GB/s)" } else { "PASS" }
        );
    }
}
