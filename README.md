# libpz

Lossless data compression library with GPU acceleration, written in Rust.

## Compression Pipelines

| Pipeline | Stages | Similar to |
|----------|--------|------------|
| **Deflate** | LZ77 + Huffman | gzip |
| **BW** | BWT + MTF + RLE + Range Coder | bzip2 |
| **LZA** | LZ77 + Range Coder | lzma-like |

Optional OpenCL support for GPU-accelerated LZ77 match finding and BWT suffix array construction.

## CLI Tool

```
pz file.txt            # compress to file.txt.pz
pz -d file.txt.pz      # decompress
pz -c file.txt         # compress to stdout
pz -k file.txt         # keep original after compress
pz -l file.txt.pz      # list info about compressed file
pz -p bw file.txt      # use BW pipeline (best ratio)
pz -g -p bw file.txt   # use GPU for BWT (requires --features opencl)
cat file | pz -c       # compress stdin to stdout
cat file | pz -dc      # decompress stdin to stdout
```

Gzip (.gz) files are auto-detected during decompression.

## Building

```
cargo build --release
```

### With OpenCL (GPU acceleration)

```
cargo build --release --features opencl
```

Requires an OpenCL SDK. See platform-specific notes below.

#### Windows (MSVC toolchain)

Set one of the following environment variables to point to your OpenCL SDK:
- `OPENCL_PATH` or `OPENCL_ROOT` — SDK root with `lib/` or `lib/x64/` containing `OpenCL.lib`
- `CUDA_PATH` — NVIDIA CUDA Toolkit (ships `OpenCL.lib`)
- `INTELOCLSDKROOT` — Intel OpenCL SDK
- `AMDAPPSDKROOT` — AMD APP SDK

Build from a Visual Studio Developer shell, or use the helper script:
```powershell
.\scripts\cargo-msvc.ps1 build --release --features opencl
```

#### Linux

```
sudo apt-get install ocl-icd-opencl-dev opencl-headers
cargo build --release --features opencl
```

## Testing

```
cargo test                      # CPU-only tests
cargo test --features opencl    # includes GPU tests (skip gracefully if no device)
```

## Benchmarks

Uses [criterion](https://github.com/bheisler/criterion.rs) for statistical benchmarks.

```
cargo bench                         # CPU benchmarks
cargo bench --features opencl       # includes GPU benchmarks
```

Two benchmark suites:
- `benches/throughput.rs` — end-to-end pipeline throughput (MB/s) with external tool comparison (gzip, pigz, zstd)
- `benches/stages.rs` — per-algorithm scaling at 1KB / 10KB / 64KB

## Linting

```
cargo clippy -- -D warnings
cargo fmt -- --check
```

## Features

- `opencl` — Enable GPU acceleration via OpenCL (LZ77 match finding + BWT suffix array construction)

## CI

GitHub Actions workflow runs on every push/PR to master:
- CPU tests on ubuntu, windows, macos
- Lint (clippy + rustfmt)
- OpenCL compile check (ubuntu)
- Benchmark compilation and execution
