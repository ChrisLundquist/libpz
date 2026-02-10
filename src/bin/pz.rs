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
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{self, ExitCode};

use pz::gzip;
use pz::pipeline::{self, CompressOptions, Pipeline};

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
    eprintln!("  -p, --pipeline P   Compression pipeline: deflate (default), bw, lza");
    eprintln!("  -t, --threads N    Number of threads (0=auto, 1=single-threaded)");
    #[cfg(feature = "opencl")]
    eprintln!("  -g, --gpu          Use GPU (OpenCL) for compression");
    eprintln!("  -q, --quiet        Suppress warnings");
    eprintln!("  -v, --verbose      Verbose output");
    eprintln!("  -h, --help         Show this help");
    eprintln!();
    eprintln!("If no FILE is given, reads from stdin and writes to stdout.");
    eprintln!("Compressed files use the .pz extension.");
    eprintln!("Gzip (.gz) files are auto-detected during decompression.");
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
                    "lza" | "2" => Pipeline::Lza,
                    other => {
                        eprintln!("pz: unknown pipeline '{other}'");
                        eprintln!("pz: valid pipelines: deflate, bw, lza");
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

fn compress_data(
    data: &[u8],
    pipe: Pipeline,
    options: &CompressOptions,
) -> Result<Vec<u8>, String> {
    pipeline::compress_with_options(data, pipe, options).map_err(|e| format!("{e}"))
}

/// Build compression options from CLI flags.
fn build_cli_options(opts: &Opts) -> CompressOptions {
    #[cfg(feature = "opencl")]
    {
        if opts.gpu {
            match pz::opencl::OpenClEngine::new() {
                Ok(engine) => {
                    if opts.verbose {
                        eprintln!("pz: using GPU device: {}", engine.device_name());
                    }
                    return CompressOptions {
                        backend: pipeline::Backend::OpenCl,
                        threads: opts.threads,
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

    CompressOptions {
        threads: opts.threads,
        ..Default::default()
    }
}

fn decompress_data(data: &[u8], threads: usize) -> Result<Vec<u8>, String> {
    match detect_format(data) {
        Format::Pz => pipeline::decompress_with_threads(data, threads).map_err(|e| format!("{e}")),
        Format::Gzip => {
            let (decompressed, _header) =
                gzip::decompress(data).map_err(|e| format!("gzip: {e}"))?;
            Ok(decompressed)
        }
        Format::Unknown => Err("unrecognized file format (not .pz or .gz)".to_string()),
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
                2 => "lza",
                _ => "unknown",
            };
            let orig_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            let ratio = if orig_len > 0 {
                (data.len() as f64 / orig_len as f64) * 100.0
            } else {
                0.0
            };
            let blocks_info = if version == 2 && data.len() >= 12 {
                let num_blocks = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                format!(" [{} blocks]", num_blocks)
            } else {
                String::new()
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
    let data = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let compressed = compress_data(&data, opts.pipeline, options)?;

    if opts.to_stdout {
        io::stdout()
            .write_all(&compressed)
            .map_err(|e| format!("stdout: {e}"))?;
    } else {
        let out_path = compress_output_path(path);
        let out_str = out_path.display().to_string();

        if out_path.exists() && !opts.force {
            return Err(format!("{out_str} already exists; use -f to overwrite"));
        }

        fs::write(&out_path, &compressed).map_err(|e| format!("{out_str}: {e}"))?;

        if opts.verbose {
            let ratio = if !data.is_empty() {
                (compressed.len() as f64 / data.len() as f64) * 100.0
            } else {
                0.0
            };
            eprintln!(
                "{path}: {:.1}% ({} → {} bytes)",
                ratio,
                data.len(),
                compressed.len()
            );
        }

        if !opts.keep {
            fs::remove_file(path).map_err(|e| format!("{path}: cannot remove: {e}"))?;
        }
    }

    Ok(())
}

fn process_decompress(opts: &Opts, path: &str) -> Result<(), String> {
    let data = fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let decompressed = decompress_data(&data, opts.threads)?;

    if opts.to_stdout {
        io::stdout()
            .write_all(&decompressed)
            .map_err(|e| format!("stdout: {e}"))?;
    } else {
        let out_path = decompress_output_path(path)
            .ok_or_else(|| format!("{path}: unknown suffix -- ignored"))?;
        let out_str = out_path.display().to_string();

        if out_path.exists() && !opts.force {
            return Err(format!("{out_str} already exists; use -f to overwrite"));
        }

        fs::write(&out_path, &decompressed).map_err(|e| format!("{out_str}: {e}"))?;

        if opts.verbose {
            eprintln!("{path}: {} → {} bytes", data.len(), decompressed.len());
        }

        if !opts.keep {
            fs::remove_file(path).map_err(|e| format!("{path}: cannot remove: {e}"))?;
        }
    }

    Ok(())
}

fn process_stdin_stdout(opts: &Opts, options: &CompressOptions) -> Result<(), String> {
    let mut data = Vec::new();
    io::stdin()
        .read_to_end(&mut data)
        .map_err(|e| format!("stdin: {e}"))?;

    let output = if opts.decompress {
        decompress_data(&data, opts.threads)?
    } else {
        compress_data(&data, opts.pipeline, options)?
    };

    io::stdout()
        .write_all(&output)
        .map_err(|e| format!("stdout: {e}"))?;

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
        let result = if opts.decompress {
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
