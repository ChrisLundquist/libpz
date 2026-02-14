# libpz

Lossless data compression library with GPU acceleration, written in Rust.

## Compression Pipelines

| Pipeline | Stages | Similar to |
|----------|--------|------------|
| **Deflate** | LZ77 + Multi-stream Huffman | gzip |
| **BW** | BWT + MTF + RLE + FSE | bzip2 |
| **LZR** | LZ77 + Multi-stream rANS | — |
| **LZF** | LZ77 + Multi-stream FSE | zstd-like |

Deflate, Lzr, and Lzf use **multi-stream entropy coding**: LZ77 output is split into separate offset, length, and literal streams, each with its own entropy coder. This yields ~16-18% better compression than single-stream encoding with no speed penalty. LZR uses rANS (range ANS) — a multiply-shift entropy coder designed for SIMD and GPU parallelism via interleaved decode states. LZF uses FSE (Finite State Entropy) — a fast table-driven tANS coder similar to zstd.

Optional WebGPU support for GPU-accelerated LZ77 match finding and BWT suffix array construction.

## CLI Tool

```
pz file.txt            # compress to file.txt.pz
pz -d file.txt.pz      # decompress
pz -c file.txt         # compress to stdout
pz -k file.txt         # keep original after compress
pz -l file.txt.pz      # list info about compressed file
pz -p bw file.txt      # use BW pipeline (best ratio)
pz -g -p bw file.txt   # use GPU for BWT (requires --features webgpu)
cat file | pz -c       # compress stdin to stdout
cat file | pz -dc      # decompress stdin to stdout
```

Gzip (.gz) files are auto-detected during decompression.

## Building

```
cargo build --release
```

### With WebGPU (GPU acceleration)

```
cargo build --release --features webgpu
```

### Windows (MSVC toolchain)

Rust’s default Windows target is `x86_64-pc-windows-msvc`, which needs the Visual Studio build tools and **Windows SDK** (for `kernel32.lib` etc.).

1. **If you get “link.exe not found” or “cannot open input file 'kernel32.lib'”**  
   - Install or modify **Build Tools for Visual Studio** and ensure both are selected:
     - Workload: **Desktop development with C++**
     - Or in **Individual components**: **MSVC v143+** and **Windows 10/11 SDK**.
   - Open **Developer PowerShell for VS** (or **x64 Native Tools Command Prompt**) from the Start menu and run `cargo build` from there,  
   **or** from a normal shell run the helper script so the VS environment is set first:
   ```powershell
   # From repo root (D:\code\libpz):
   .\scripts\cargo-msvc.ps1 build

   # From libpz-rs (D:\code\libpz\libpz-rs):
   ..\scripts\cargo-msvc.ps1 build
   ```

2. Stay on the **MSVC** target; do not switch to the GNU toolchain unless you intentionally want a different target/ABI.

## Testing

```
cargo test                      # CPU-only tests
cargo test --features webgpu    # includes GPU tests (skip gracefully if no device)
```

## Benchmarks

Uses [criterion](https://github.com/bheisler/criterion.rs) for statistical benchmarks.

```
cargo bench                         # CPU benchmarks
cargo bench --features webgpu       # includes GPU benchmarks
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

- `webgpu` — Enable GPU acceleration via wgpu/WebGPU (LZ77 match finding + BWT suffix array construction)

## CI

GitHub Actions workflow runs on every push/PR to master:
- CPU tests on ubuntu, windows, macos
- Lint (clippy + rustfmt)
- Benchmark compilation and execution
