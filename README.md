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
