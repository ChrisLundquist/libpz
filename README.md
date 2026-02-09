# libpz

Lossless data compression library with GPU acceleration, written in Rust.

Implements three compression pipelines:
- **Deflate** (gzip-like): LZ77 + Huffman
- **BW** (bzip2-like): BWT + MTF + RLE + Range Coder
- **LZA** (LZMA-like): LZ77 + Range Coder

Optional OpenCL support for GPU-accelerated match finding.

## Building

```
cd libpz-rs
cargo build
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
cd libpz-rs
cargo test
```

## Linting

```
cd libpz-rs
cargo clippy
```

## Features

- `opencl` — Enable GPU acceleration via OpenCL

```
cargo build --features opencl
```
