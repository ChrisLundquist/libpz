/// pz – CLI compression tool for libpz.
///
/// Works similar to gzip / zstd:
///   pz file.txt          → compress to file.txt.pz (removes original)
///   pz -d file.txt.pz    → decompress to file.txt (removes original)
///   pz -d file.txt.gz    → decompress gzip file to file.txt
///   pz -c file.txt       → compress to stdout
///   pz -k file.txt       → keep original after compress
///   pz -l file.txt.pz    → list info about compressed file
///   cat file | pz -c     → compress stdin to stdout
///   cat file | pz -dc    → decompress stdin to stdout
use std::env;
use std::fs;
use std::io::{self, BufReader, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::process::{self, ExitCode};

use pz::gzip;
use pz::pipeline::{self, CompressOptions, ParseStrategy, Pipeline};
use pz::streaming;

fn usage() {
    eprintln!("pz - lossless data compression tool");
    eprintln!();
    eprintln!("Usage: pz [OPTIONS] [FILE]...");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -d, --decompress   Decompress mode");
    eprintln!("  -c, --stdout       Write to stdout (don't remove original)");
    eprintln!("  -k, --keep         Keep original file");
    eprintln!("  -f, --force        Overwrite existing output files");
    eprintln!("  -l, --list         List info about compressed file");
    eprintln!("  -p, --pipeline P   Compression pipeline (default: deflate)");
    eprintln!("  --list-pipelines   List all available pipelines and exit");
    eprintln!("  -a, --auto         Auto-select best pipeline based on data analysis");
    eprintln!("  --trial            Auto-select by trial compression (slower, more accurate)");
    eprintln!("  -t, --threads N    Number of threads (0=auto, 1=single-threaded)");
    #[cfg(feature = "opencl")]
    eprintln!("  -g, --gpu          Use GPU (OpenCL) for compression");
    #[cfg(feature = "webgpu")]
    eprintln!("  --webgpu           Use GPU (WebGPU/wgpu) for compression");
    eprintln!("  -O, --optimal      Use optimal parsing (best compression, slowest)");
    eprintln!("  --lazy             Use lazy matching (good compression, default)");
    eprintln!("  --greedy           Use greedy matching (fastest, least compression)");
    eprintln!("  -q, --quiet        Suppress warnings");
    eprintln!("  -v, --verbose      Verbose output");
    eprintln!("  -h, --help         Show this help");
    eprintln!();
    eprintln!("If no FILE is given, reads from stdin and writes to stdout.");
    eprintln!("Compressed files use the .pz extension.");
    eprintln!("Gzip (.gz) files are auto-detected during decompression.");
}

fn list_pipelines() {
    println!("Available compression pipelines:");
    println!();
    println!("  NAME        ID  DESCRIPTION");
    println!("  ----        --  -----------");
    let pipelines: &[(&str, &str, &str)] = &[
        ("deflate", "0", "LZ77 + Huffman (gzip-like, default)"),
        ("bw", "1", "BWT + MTF + RLE + FSE (bzip2-like, best ratio)"),
        (
            "bbw",
            "2",
            "Bijective BWT + MTF + RLE + FSE (parallelizable BWT)",
        ),
        ("lzr", "3", "LZ77 + rANS (fastest compression)"),
        ("lzf", "4", "LZ77 + FSE (zstd-style entropy coding)"),
        ("lzssr", "6", "LZSS + rANS (experimental)"),
        ("lz78r", "8", "LZ78 + rANS (experimental)"),
    ];
    for (name, id, desc) in pipelines {
        println!("  {name:10} {id:>2}  {desc}");
    }
    println!();
    println!("Use -p <name> to select a pipeline, or -a for auto-selection.");
}

#[derive(Debug)]
struct Opts {
    decompress: bool,
    to_stdout: bool,
    keep: bool,
    force: bool,
    list: bool,
    verbose: bool,
    quiet: bool,
    gpu: bool,
    webgpu: bool,
    auto_select: bool,
    trial_mode: bool,
    parse_strategy: ParseStrategy,
    threads: usize,
    pipeline: Pipeline,
    files: Vec<String>,
}

fn parse_args() -> Opts {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut opts = Opts {
        decompress: false,
        to_stdout: false,
        keep: false,
        force: false,
        list: false,
        verbose: false,
        quiet: false,
        gpu: false,
        webgpu: false,
        auto_select: false,
        trial_mode: false,
        parse_strategy: ParseStrategy::Auto,
        threads: 0,
        pipeline: Pipeline::Deflate,
        files: Vec::new(),
    };

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        match arg.as_str() {
            "-d" | "--decompress" => opts.decompress = true,
            "-c" | "--stdout" | "--to-stdout" => opts.to_stdout = true,
            "-k" | "--keep" => opts.keep = true,
            "-f" | "--force" => opts.force = true,
            "-l" | "--list" => opts.list = true,
            "-v" | "--verbose" => opts.verbose = true,
            "-q" | "--quiet" => opts.quiet = true,
            "-g" | "--gpu" | "--opencl" => opts.gpu = true,
            "--webgpu" => opts.webgpu = true,
            "-a" | "--auto" => opts.auto_select = true,
            "--trial" => {
                opts.auto_select = true;
                opts.trial_mode = true;
            }
            "-O" | "--optimal" => opts.parse_strategy = ParseStrategy::Optimal,
            "--lazy" => opts.parse_strategy = ParseStrategy::Lazy,
            "--greedy" => opts.parse_strategy = ParseStrategy::Lazy, // greedy removed; lazy is strictly better
            "--list-pipelines" => {
                list_pipelines();
                process::exit(0);
            }
            "-h" | "--help" => {
                usage();
                process::exit(0);
            }
            "-p" | "--pipeline" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("pz: missing argument for -p");
                    process::exit(1);
                }
                opts.pipeline = match args[i].as_str() {
                    "deflate" | "0" => Pipeline::Deflate,
                    "bw" | "1" => Pipeline::Bw,
                    "bbw" | "2" => Pipeline::Bbw,
                    "lzr" | "3" => Pipeline::Lzr,
                    "lzf" | "4" => Pipeline::Lzf,
                    "lzssr" | "6" => Pipeline::LzssR,
                    "lz78r" | "8" => Pipeline::Lz78R,
                    other => {
                        eprintln!("pz: unknown pipeline '{other}'");
                        eprintln!("pz: run 'pz --list-pipelines' to see available pipelines");
                        process::exit(1);
                    }
                };
            }
            "-t" | "--threads" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("pz: missing argument for -t");
                    process::exit(1);
                }
                opts.threads = match args[i].parse::<usize>() {
                    Ok(n) => n,
                    Err(_) => {
                        eprintln!("pz: invalid thread count '{}'", args[i]);
                        process::exit(1);
                    }
                };
            }
            // Handle combined short flags like -dc, -kv, etc.
            s if s.starts_with('-') && !s.starts_with("--") && s.len() > 2 => {
                for ch in s[1..].chars() {
                    match ch {
                        'd' => opts.decompress = true,
                        'c' => opts.to_stdout = true,
                        'k' => opts.keep = true,
                        'f' => opts.force = true,
                        'l' => opts.list = true,
                        'v' => opts.verbose = true,
                        'q' => opts.quiet = true,
                        'g' => opts.gpu = true,
                        'a' => opts.auto_select = true,
                        'O' => opts.parse_strategy = ParseStrategy::Optimal,
                        _ => {
                            eprintln!("pz: unknown flag '-{ch}'");
                            process::exit(1);
                        }
                    }
                }
            }
            _ => {
                opts.files.push(arg.clone());
            }
        }
        i += 1;
    }

    opts
}

/// Detect the format of compressed data and return a descriptive string.
enum Format {
    Pz,
    Gzip,
    Unknown,
}

fn detect_format(data: &[u8]) -> Format {
    if data.len() >= 2 && data[0] == b'P' && data[1] == b'Z' {
        Format::Pz
    } else if gzip::is_gzip(data) {
        Format::Gzip
    } else {
        Format::Unknown
    }
}

/// Determine the output filename for compression.
fn compress_output_path(input: &str) -> PathBuf {
    PathBuf::from(format!("{input}.pz"))
}

/// Determine the output filename for decompression.
fn decompress_output_path(input: &str) -> Option<PathBuf> {
    let path = Path::new(input);
    if let Some(ext) = path.extension() {
        match ext.to_str() {
            Some("pz") => Some(path.with_extension("")),
            Some("gz") => Some(path.with_extension("")),
            _ => None,
        }
    } else {
        None
    }
}

/// Build compression options from CLI flags.
fn build_cli_options(opts: &Opts) -> CompressOptions {
    let parse_strategy = opts.parse_strategy;

    #[cfg(feature = "opencl")]
    {
        if opts.gpu {
            let result = if opts.verbose {
                pz::opencl::OpenClEngine::with_profiling(true)
            } else {
                pz::opencl::OpenClEngine::new()
            };
            match result {
                Ok(engine) => {
                    if opts.verbose {
                        eprintln!("pz: using GPU device: {}", engine.device_name());
                    }
                    return CompressOptions {
                        backend: pipeline::Backend::OpenCl,
                        threads: opts.threads,
                        parse_strategy,
                        opencl_engine: Some(std::sync::Arc::new(engine)),
                        ..Default::default()
                    };
                }
                Err(_) => {
                    if !opts.quiet {
                        eprintln!(
                            "pz: warning: GPU requested but OpenCL not available, \
                             falling back to CPU"
                        );
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "opencl"))]
    {
        if opts.gpu && !opts.quiet {
            eprintln!(
                "pz: warning: --gpu requires the opencl feature \
                 (build with --features opencl)"
            );
        }
    }

    #[cfg(feature = "webgpu")]
    {
        if opts.webgpu {
            let result = if opts.verbose {
                pz::webgpu::WebGpuEngine::with_profiling(true)
            } else {
                pz::webgpu::WebGpuEngine::new()
            };
            match result {
                Ok(engine) => {
                    if opts.verbose {
                        eprintln!("pz: using WebGPU device: {}", engine.device_name());
                    }
                    return CompressOptions {
                        backend: pipeline::Backend::WebGpu,
                        threads: opts.threads,
                        parse_strategy,
                        #[cfg(feature = "opencl")]
                        opencl_engine: None,
                        webgpu_engine: Some(std::sync::Arc::new(engine)),
                        ..Default::default()
                    };
                }
                Err(_) => {
                    if !opts.quiet {
                        eprintln!(
                            "pz: warning: --webgpu requested but no WebGPU device available, \
                             falling back to CPU"
                        );
                    }
                }
            }
        }
    }

    #[cfg(not(feature = "webgpu"))]
    {
        if opts.webgpu && !opts.quiet {
            eprintln!(
                "pz: warning: --webgpu requires the webgpu feature \
                 (build with --features webgpu)"
            );
        }
    }

    CompressOptions {
        threads: opts.threads,
        parse_strategy,
        ..Default::default()
    }
}

fn list_file(path: &str, data: &[u8]) -> Result<(), String> {
    match detect_format(data) {
        Format::Pz => {
            if data.len() < 8 {
                return Err("truncated pz header".to_string());
            }
            let version = data[2];
            let pipe = match data[3] {
                0 => "deflate",
                1 => "bw",
                2 => "bbw",
                3 => "lzr",
                4 => "lzf",
                _ => "unknown",
            };
            let mut orig_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            let blocks_info = if version == 2 && data.len() >= 12 {
                let num_blocks = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                if num_blocks == 0xFFFF_FFFF {
                    // Framed mode: scan block frames to compute original length
                    let mut pos = 12;
                    let mut total_orig = 0u64;
                    let mut n_blocks = 0u32;
                    while pos + 4 <= data.len() {
                        let comp_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                        if comp_len == 0 {
                            break; // EOS sentinel
                        }
                        if pos + 8 > data.len() {
                            break;
                        }
                        let block_orig =
                            u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
                        total_orig += block_orig as u64;
                        n_blocks += 1;
                        pos += 8 + comp_len as usize;
                    }
                    if orig_len == 0 {
                        orig_len = total_orig as u32;
                    }
                    format!(" [{n_blocks} blocks, framed]")
                } else {
                    format!(" [{num_blocks} blocks]")
                }
            } else {
                String::new()
            };
            let ratio = if orig_len > 0 {
                (data.len() as f64 / orig_len as f64) * 100.0
            } else {
                0.0
            };
            println!(
                "{:>12} {:>12} {:5.1}% {:8} {}{}",
                orig_len,
                data.len(),
                ratio,
                pipe,
                path,
                blocks_info,
            );
        }
        Format::Gzip => {
            let header = gzip::info(data).map_err(|e| format!("{e}"))?;
            let isize_offset = data.len().saturating_sub(4);
            let orig_size = if data.len() >= 8 {
                u32::from_le_bytes([
                    data[isize_offset],
                    data[isize_offset + 1],
                    data[isize_offset + 2],
                    data[isize_offset + 3],
                ])
            } else {
                0
            };
            let ratio = if orig_size > 0 {
                (data.len() as f64 / orig_size as f64) * 100.0
            } else {
                0.0
            };
            let name = header.filename.as_deref().unwrap_or("-");
            println!(
                "{:>12} {:>12} {:5.1}% {:8} {} [{}]",
                orig_size,
                data.len(),
                ratio,
                "gzip",
                path,
                name
            );
        }
        Format::Unknown => {
            return Err(format!("{path}: not in pz or gzip format"));
        }
    }
    Ok(())
}

fn process_compress(opts: &Opts, path: &str, options: &CompressOptions) -> Result<(), String> {
    // Open input file
    let mut file = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;

    // Determine pipeline: read a sample for auto-select, then seek back
    let pipe = if opts.auto_select {
        let mut sample = vec![0u8; 65536];
        let n = file.read(&mut sample).map_err(|e| format!("{path}: {e}"))?;
        sample.truncate(n);
        file.seek(io::SeekFrom::Start(0))
            .map_err(|e| format!("{path}: cannot seek: {e}"))?;
        let selected = if opts.trial_mode {
            pipeline::select_pipeline_trial(&sample, options, 65536)
        } else {
            pipeline::select_pipeline(&sample)
        };
        if opts.verbose {
            eprintln!("pz: auto-selected pipeline: {:?}", selected);
        }
        selected
    } else {
        opts.pipeline
    };

    let input = BufReader::new(file);

    if opts.to_stdout {
        let output = BufWriter::new(io::stdout().lock());
        streaming::compress_stream(input, output, pipe, options)
            .map_err(|e| format!("{path}: {e}"))?;
    } else {
        let out_path = compress_output_path(path);
        let out_str = out_path.display().to_string();

        if out_path.exists() && !opts.force {
            return Err(format!("{out_str} already exists; use -f to overwrite"));
        }

        let out_file = fs::File::create(&out_path).map_err(|e| format!("{out_str}: {e}"))?;
        let output = BufWriter::new(out_file);
        streaming::compress_stream(input, output, pipe, options)
            .map_err(|e| format!("{path}: {e}"))?;

        if opts.verbose {
            let in_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            let out_size = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
            let ratio = if in_size > 0 {
                (out_size as f64 / in_size as f64) * 100.0
            } else {
                0.0
            };
            eprintln!("{path}: {ratio:.1}% ({in_size} → {out_size} bytes)");
        }

        if !opts.keep {
            fs::remove_file(path).map_err(|e| format!("{path}: cannot remove: {e}"))?;
        }
    }

    Ok(())
}

fn process_decompress(opts: &Opts, path: &str) -> Result<(), String> {
    // Peek first 2 bytes to detect format
    let mut file = fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut magic = [0u8; 2];
    let n = file.read(&mut magic).map_err(|e| format!("{path}: {e}"))?;
    file.seek(io::SeekFrom::Start(0))
        .map_err(|e| format!("{path}: cannot seek: {e}"))?;

    let is_gzip = n >= 2 && gzip::is_gzip(&magic);

    if is_gzip {
        // Gzip: use in-memory decompress (gzip module is not streaming)
        let data = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        let (decompressed, _header) =
            gzip::decompress(&data).map_err(|e| format!("{path}: gzip: {e}"))?;
        write_decompressed_output(opts, path, &decompressed)?;
    } else {
        // PZ: use streaming decompress
        let input = BufReader::new(file);
        if opts.to_stdout {
            let output = BufWriter::new(io::stdout().lock());
            streaming::decompress_stream(input, output, opts.threads)
                .map_err(|e| format!("{path}: {e}"))?;
        } else {
            let out_path = decompress_output_path(path)
                .ok_or_else(|| format!("{path}: unknown suffix -- ignored"))?;
            let out_str = out_path.display().to_string();

            if out_path.exists() && !opts.force {
                return Err(format!("{out_str} already exists; use -f to overwrite"));
            }

            let out_file = fs::File::create(&out_path).map_err(|e| format!("{out_str}: {e}"))?;
            let output = BufWriter::new(out_file);
            streaming::decompress_stream(input, output, opts.threads)
                .map_err(|e| format!("{path}: {e}"))?;

            if opts.verbose {
                let in_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                let out_size = fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
                eprintln!("{path}: {in_size} → {out_size} bytes");
            }

            if !opts.keep {
                fs::remove_file(path).map_err(|e| format!("{path}: cannot remove: {e}"))?;
            }
        }
    }

    Ok(())
}

/// Write decompressed data to output (for gzip in-memory fallback).
fn write_decompressed_output(opts: &Opts, path: &str, data: &[u8]) -> Result<(), String> {
    if opts.to_stdout {
        io::stdout()
            .write_all(data)
            .map_err(|e| format!("stdout: {e}"))?;
    } else {
        let out_path = decompress_output_path(path)
            .ok_or_else(|| format!("{path}: unknown suffix -- ignored"))?;
        let out_str = out_path.display().to_string();

        if out_path.exists() && !opts.force {
            return Err(format!("{out_str} already exists; use -f to overwrite"));
        }

        fs::write(&out_path, data).map_err(|e| format!("{out_str}: {e}"))?;

        if opts.verbose {
            let in_size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            eprintln!("{path}: {in_size} → {} bytes", data.len());
        }

        if !opts.keep {
            fs::remove_file(path).map_err(|e| format!("{path}: cannot remove: {e}"))?;
        }
    }
    Ok(())
}

fn process_stdin_stdout(opts: &Opts, options: &CompressOptions) -> Result<(), String> {
    // Use io::stdin() (not .lock()) so the BufReader is Send for the worker threads.
    let stdin = io::stdin();
    let stdout = io::stdout();

    if opts.decompress {
        let input = BufReader::new(stdin);
        let output = BufWriter::new(stdout.lock());
        streaming::decompress_stream(input, output, opts.threads)
            .map_err(|e| format!("stdin: {e}"))?;
    } else {
        // For auto-select from stdin, we require explicit --pipeline since
        // we cannot seek back. Use the default pipeline if not specified.
        let pipe = if opts.auto_select && !opts.quiet {
            eprintln!("pz: warning: auto-select requires a seekable input; using default pipeline");
            opts.pipeline
        } else {
            opts.pipeline
        };
        let input = BufReader::new(stdin);
        let output = BufWriter::new(stdout.lock());
        streaming::compress_stream(input, output, pipe, options)
            .map_err(|e| format!("stdin: {e}"))?;
    }

    Ok(())
}

fn run() -> Result<(), ()> {
    let opts = parse_args();
    let compress_options = build_cli_options(&opts);
    let mut had_error = false;

    if opts.files.is_empty() {
        // stdin/stdout mode
        if opts.list {
            eprintln!("pz: -l requires a file argument");
            return Err(());
        }
        if let Err(e) = process_stdin_stdout(&opts, &compress_options) {
            eprintln!("pz: {e}");
            return Err(());
        }
        return Ok(());
    }

    // List mode
    if opts.list {
        println!(
            "{:>12} {:>12} {:>6} {:>8} name",
            "original", "compressed", "ratio", "type"
        );
        for path in &opts.files {
            match fs::read(path) {
                Ok(data) => {
                    if let Err(e) = list_file(path, &data) {
                        eprintln!("pz: {e}");
                        had_error = true;
                    }
                }
                Err(e) => {
                    eprintln!("pz: {path}: {e}");
                    had_error = true;
                }
            }
        }
        return if had_error { Err(()) } else { Ok(()) };
    }

    for path in &opts.files {
        let result = if path == "-" {
            process_stdin_stdout(&opts, &compress_options)
        } else if opts.decompress {
            process_decompress(&opts, path)
        } else {
            process_compress(&opts, path, &compress_options)
        };

        if let Err(e) = result {
            eprintln!("pz: {e}");
            had_error = true;
        }
    }

    if had_error {
        Err(())
    } else {
        Ok(())
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(()) => ExitCode::FAILURE,
    }
}
