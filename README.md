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

#### OpenCL feature (`cargo build --features opencl`)

The `opencl3` crate (via `opencl-sys`) links to **OpenCL.lib** — the MSVC import library. That is independent of the Rust GNU vs MSVC *target*: you stay on the MSVC target and need an OpenCL *SDK* that provides **OpenCL.lib** (not only OpenCL.dll or MinGW-style `libOpenCL.a`).

1. **Provide a path that contains `OpenCL.lib`**  
   Set **one** of these (to the SDK root or the folder that contains the `lib` directory with `OpenCL.lib`):

   - **`OPENCL_PATH`** — path to a folder that has a `lib` subfolder with `OpenCL.lib` (e.g. `OPENCL_PATH=C:\OpenCL\sdk` if `OpenCL.lib` is in `C:\OpenCL\sdk\lib` or `...\lib\x64`).
   - **`OPENCL_ROOT`** — same idea; opencl-sys will look in `%OPENCL_ROOT%\lib\x64` (64-bit).
   - **`CUDA_PATH`** — if you use the NVIDIA CUDA Toolkit, it ships `OpenCL.lib` (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`); opencl-sys will use `%CUDA_PATH%\lib\x64`.
   - **`INTELOCLSDKROOT`** or **`AMDAPPSDKROOT`** — for Intel or AMD OpenCL SDKs.

2. **Where to get OpenCL.lib**  
   - **NVIDIA**: Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads); set `CUDA_PATH` to the toolkit root (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3`).
   - **Intel**: [Intel SDK for OpenCL](https://www.intel.com/content/www/us/en/developer/articles/tool/opencl-drivers.html) or CPU runtime; set `INTELOCLSDKROOT` or `OPENCL_PATH` to the SDK root that has `lib\x64\OpenCL.lib`.
   - **AMD**: AMD APP SDK; set `AMDAPPSDKROOT` accordingly.

3. **MinGW / “GNU” OpenCL**  
   If your OpenCL install only has MinGW/GNU libraries (e.g. `libOpenCL.a`), that does **not** work with the MSVC target. Use an SDK that includes **OpenCL.lib** (e.g. CUDA or Intel SDK) and the env var above, and keep using the MSVC target.

Then build with the VS environment (Developer PowerShell or the script) and:

```powershell
# From repo root:
.\scripts\cargo-msvc.ps1 build --features opencl
# From libpz-rs:
..\scripts\cargo-msvc.ps1 build --features opencl
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
