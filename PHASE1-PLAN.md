# libpz Phase 1: Foundation & Architecture Decision

## Status: COMPLETE - Rust Migration Finalized

> **Note:** This document is a historical planning artifact. References to a
> future "Vulkan backend" were superseded by the WebGPU (wgpu) backend, which
> provides cross-platform GPU compute via Vulkan, Metal, and DX12.
> See `GPU_BACKEND_MIGRATION_EVAL.md` for the evaluation.

## Executive Summary

Phase 1 established the foundation for libpz. The key outcomes:
1. **Implementation language decided:** Rust (migration complete)
2. **C reference implementations removed** - Rust equivalents are canonical
3. **GPU integration retained:** OpenCL kernel files (.cl) and C engine kept as-is
4. **C-callable API defined** via FFI in `libpz-rs/src/ffi.rs`
5. **Testing infrastructure** established in Rust (unit tests, validation)

**Decision: Full Rust migration (C reference files removed)**

---

## 1. Language Decision: C vs Rust Analysis

### Current State (Post-Migration)

**Codebase:** Rust-based with OpenCL GPU backend
- **Rust (`libpz-rs/src/`):** LZ77, Huffman, frequency analysis, priority queue, BWT, MTF, RLE, range coder, pipeline, FFI, validation
- **OpenCL (`opencl/`):** GPU match-finding kernels (lz77.cl, lz77-batch.cl), C engine (engine.c)
- **Removed:** All C reference implementations, C headers, C tests, C main

**Previous C Bug Assessment (historical):**
- 25 documented bugs, majority were memory safety issues
- These are no longer relevant as the C code has been replaced by Rust
- Rust's ownership system and bounds checking prevent the class of bugs that plagued the C implementation

### Rust Advantages for This Project

#### 1. **Memory Safety by Default**
Rust would prevent at compile time:
- **BUG-01:** Out-of-bounds array access (bounds checking)
- **BUG-02, BUG-03:** Buffer overflow in LZ77 (slice bounds checking)
- **BUG-04:** OpenCL kernel bounds (safer bindings)
- **BUG-05:** Signed char as array index (strong typing, no implicit conversions)
- **BUG-06:** Buffer overflow on write (bounds checking)
- **BUG-07:** Missing null terminator (String type handles this)
- **BUG-19:** Memory leaks (RAII, automatic cleanup)
- **BUG-21:** Use-after-free risks (ownership system)

**Impact:** ~12 of 25 bugs (48%) would be **prevented by the compiler**.

#### 2. **GPU Integration Support**

**OpenCL:**
- `opencl3` crate: Safe Rust bindings to OpenCL 3.0
- `ocl` crate: High-level ergonomic OpenCL wrapper
- Same kernel code (.cl files) can be used
- Better error handling than C (Result types vs error codes)

**Vulkan:**
- `vulkano`: Safe, high-level Vulkan wrapper
- `ash`: Low-level Vulkan bindings (FFI-like)
- SPIR-V shader compilation via `shaderc` or `rspirv`
- `wgpu`: WebGPU abstraction (supports Vulkan, Metal, DX12)

**Example OpenCL in Rust:**
```rust
use opencl3::device::Device;
use opencl3::context::Context;
use opencl3::kernel::Kernel;

let devices = Device::get_all()?;
let context = Context::from_device(&devices[0])?;
let kernel = Kernel::create_from_source(&context, KERNEL_SOURCE, "lz77_match")?;
```

Cleaner, safer, and more maintainable than the current C code with manual error checking.

#### 3. **C-Callable Library Generation**

Rust makes C FFI trivial with `cbindgen`:

**Rust code:**
```rust
#[no_mangle]
pub extern "C" fn pz_compress(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
) -> i32 {
    // Safe Rust implementation inside
    let input_slice = unsafe { std::slice::from_raw_parts(input, input_len) };
    let output_slice = unsafe { std::slice::from_raw_parts_mut(output, output_len) };

    match compress_internal(input_slice, output_slice) {
        Ok(bytes_written) => bytes_written as i32,
        Err(_) => -1,
    }
}
```

**Generated C header (via cbindgen):**
```c
// Auto-generated from Rust code
int32_t pz_compress(const uint8_t* input,
                    size_t input_len,
                    uint8_t* output,
                    size_t output_len);
```

**Build system:**
- `cargo build --release` produces `libpz.a` (static) or `libpz.so` (dynamic)
- Compatible with existing autotools, CMake, or any C build system
- No runtime dependencies (statically link Rust stdlib)

#### 4. **Better Type System**

**Current C issues that Rust solves:**
- Signed/unsigned confusion (BUG-05, BUG-14, BUG-15)
- Const correctness violations (BUG-13)
- Pointer type mismatches (BUG-16, BUG-17)

**Rust equivalent:**
```rust
// No implicit conversions, clear ownership
fn get_frequency(input: &[u8]) -> FrequencyTable;  // Immutable slice
fn huffman_encode(input: &[u8], output: &mut [u8]) -> Result<usize>;  // Mutable slice
```

#### 5. **Concurrency Safety**

For Phase 4 (pthread backend), Rust's ownership system prevents:
- Data races (compile-time enforcement)
- Send/Sync traits ensure thread safety
- `rayon` crate for easy data parallelism

**Example block-parallel compression:**
```rust
use rayon::prelude::*;

fn compress_parallel(blocks: &[Block]) -> Vec<CompressedBlock> {
    blocks.par_iter()  // Automatically parallel
          .map(|block| compress_block(block))
          .collect()
}
```

No manual pthread management, no race conditions.

### C Advantages

1. **No rewrite cost** - existing code works (after bug fixes)
2. **Simpler build** - just autotools, no Rust toolchain
3. **Smaller binary** - no Rust stdlib overhead (~200-500KB)
4. **Direct OpenCL/Vulkan** - no binding layer

### Decision: **Full Rust Migration (Complete)**

The hybrid approach was followed and the migration decision has been made:

- **Phase 1A (DONE):** C bugs were analyzed; Rust reimplementation chosen over fixing
- **Phase 1B (DONE):** Rust implementations created for all core algorithms
- **Phase 1C (DONE):** C reference files removed; Rust is now canonical

**What was migrated to Rust:**
- LZ77 compression (`libpz-rs/src/lz77.rs`)
- Huffman coding (`libpz-rs/src/huffman.rs`)
- Frequency analysis (`libpz-rs/src/frequency.rs`)
- Priority queue (`libpz-rs/src/pqueue.rs`)
- Additional algorithms: BWT, MTF, RLE, range coder
- Compression pipelines (DEFLATE, BW, LZA)
- C-callable FFI layer (`libpz-rs/src/ffi.rs`)

**What remains in C:**
- OpenCL GPU engine (`opencl/engine.c`, `opencl/engine.h`)
- OpenCL test harness (`opencl/test.c`)
- OpenCL kernels (`opencl/lz77.cl`, `opencl/lz77-batch.cl`)

---

## 2. GPU Integration Architecture

### Requirements

1. **Multi-backend support:** OpenCL (primary), WebGPU via wgpu (done)
2. **Runtime detection:** Probe available devices, gracefully fall back
3. **Shared kernel code:** Same algorithms across backends
4. **C-callable API:** External code shouldn't know about Rust/GPU internals

### Proposed Architecture

```
┌──────────────────────────────────────────────────────┐
│                  C API Layer (FFI)                    │
│   pz_compress(), pz_decompress(), pz_query_devices() │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│            Rust Core Library (libpz-core)            │
│  - Algorithm orchestration                           │
│  - Backend selection (GPU/CPU)                       │
│  - Memory management                                  │
└────────────────────┬─────────────────────────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼─────┐  ┌─────▼──────┐  ┌────▼─────┐
│ OpenCL   │  │  Vulkan    │  │ CPU Ref  │
│ Backend  │  │  Backend   │  │ Backend  │
│ (Rust)   │  │  (Rust)    │  │ (Rust)   │
└────┬─────┘  └─────┬──────┘  └──────────┘
     │              │
┌────▼──────────────▼──────┐
│  Kernel Code (.cl, .spv) │
│  - lz77.cl                │
│  - huffman.cl             │
└───────────────────────────┘
```

### OpenCL Integration in Rust

**Dependencies (Cargo.toml):**
```toml
[dependencies]
opencl3 = "0.9"           # Safe OpenCL 3.0 bindings
lazy_static = "1.4"       # For lazy device initialization

[build-dependencies]
cbindgen = "0.26"         # Generate C headers

[features]
default = ["opencl"]
opencl = ["opencl3"]
vulkan = ["ash", "gpu-allocator"]
```

**Runtime device detection:**
```rust
pub struct ComputeBackend {
    opencl_devices: Vec<OpenCLDevice>,
    vulkan_devices: Vec<VulkanDevice>,
    cpu_threads: usize,
}

impl ComputeBackend {
    pub fn probe() -> Self {
        let opencl_devices = match opencl3::platform::get_platforms() {
            Ok(platforms) => platforms.iter()
                .flat_map(|p| p.get_devices(CL_DEVICE_TYPE_ALL).unwrap_or_default())
                .collect(),
            Err(_) => vec![],
        };

        let vulkan_devices = probe_vulkan_devices();  // Similar pattern
        let cpu_threads = num_cpus::get();

        Self { opencl_devices, vulkan_devices, cpu_threads }
    }

    pub fn select_device(&self, workload: Workload) -> Device {
        // Heuristic: GPU for large buffers, CPU for small
        if workload.size > 64 * 1024 && !self.opencl_devices.is_empty() {
            Device::OpenCL(self.opencl_devices[0].clone())
        } else {
            Device::CPU
        }
    }
}
```

**Kernel compilation:**
```rust
pub struct LZ77Matcher {
    context: Context,
    kernel: Kernel,
    queue: CommandQueue,
}

impl LZ77Matcher {
    pub fn new(device: &OpenCLDevice) -> Result<Self> {
        let context = Context::from_device(device)?;

        // Load kernel source from embedded file or filesystem
        let kernel_src = include_str!("../kernels/lz77.cl");
        let program = Program::create_and_build_from_source(&context, kernel_src, "")?;
        let kernel = Kernel::create(&program, "find_matches")?;
        let queue = CommandQueue::create(&context, device, 0)?;

        Ok(Self { context, kernel, queue })
    }

    pub fn find_matches(&mut self, input: &[u8], window_size: usize) -> Result<Vec<Match>> {
        // Create buffers
        let input_buf = Buffer::create(&self.context, CL_MEM_READ_ONLY, input.len(), None)?;
        let output_buf = Buffer::create(&self.context, CL_MEM_WRITE_ONLY,
                                       input.len() * std::mem::size_of::<Match>(), None)?;

        // Write input
        self.queue.enqueue_write_buffer(&input_buf, CL_BLOCKING, 0, input, &[])?;

        // Set kernel args (type-safe)
        self.kernel.set_arg(0, &input_buf)?;
        self.kernel.set_arg(1, &output_buf)?;
        self.kernel.set_arg(2, &(window_size as u32))?;

        // Execute
        let global_size = input.len();
        self.queue.enqueue_nd_range_kernel(&self.kernel, 1, None, &[global_size], None, &[])?;

        // Read results
        let mut matches = vec![Match::default(); input.len()];
        self.queue.enqueue_read_buffer(&output_buf, CL_BLOCKING, 0, &mut matches, &[])?;

        Ok(matches)
    }
}
```

**Error handling:**
- All OpenCL errors map to `Result<T, Error>`
- No more manual `CL_SUCCESS` checks everywhere
- Propagate errors cleanly with `?` operator

### WebGPU Integration (Done — replaced Vulkan plan)

The originally planned Vulkan backend was replaced with WebGPU via the `wgpu`
crate, which provides Vulkan/Metal/DX12 support through a single API.
See `src/webgpu.rs` and `GPU_BACKEND_MIGRATION_EVAL.md`.

---

## 3. C-Callable API Design

### Requirements

1. **Simple C types:** No Rust types leak into API
2. **Manual memory management:** Caller allocates buffers
3. **Clear error codes:** No panics across FFI boundary
4. **Opaque handles:** Hide Rust internals

### API Design

**Core compression API:**
```c
// C header (generated by cbindgen)

#ifndef LIBPZ_H
#define LIBPZ_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define PZ_OK                0
#define PZ_ERROR_BUFFER_TOO_SMALL  -1
#define PZ_ERROR_INVALID_INPUT     -2
#define PZ_ERROR_UNSUPPORTED       -3
#define PZ_ERROR_GPU_FAILURE       -4

// Compression levels
typedef enum {
    PZ_LEVEL_FAST = 1,      // Greedy LZ77, no GPU
    PZ_LEVEL_DEFAULT = 5,   // Lazy matching, GPU if available
    PZ_LEVEL_BEST = 9,      // Optimal parsing, GPU required
} pz_level_t;

// Pipeline types
typedef enum {
    PZ_DEFLATE,  // LZ77 + Huffman (gzip-like)
    PZ_BW,       // BWT + MTF + RLE + Arithmetic (bzip2-like)
    PZ_LZA,      // LZ77 + Arithmetic (lzma-like)
} pz_pipeline_t;

// Opaque context handle
typedef struct pz_context pz_context_t;

// Initialize library, probe devices
pz_context_t* pz_init(void);

// Clean up context
void pz_destroy(pz_context_t* ctx);

// Query available compute devices
typedef struct {
    int32_t opencl_devices;
    int32_t vulkan_devices;
    int32_t cpu_threads;
} pz_device_info_t;

int32_t pz_query_devices(pz_context_t* ctx, pz_device_info_t* info);

// Compress data
// Returns: bytes written on success, negative error code on failure
int32_t pz_compress(
    pz_context_t* ctx,
    const uint8_t* input,
    size_t input_len,
    uint8_t* output,
    size_t output_len,
    pz_level_t level,
    pz_pipeline_t pipeline
);

// Decompress data
int32_t pz_decompress(
    pz_context_t* ctx,
    const uint8_t* input,
    size_t input_len,
    uint8_t* output,
    size_t output_len
);

// Get maximum output size for compression
// (Conservative upper bound, actual size usually much smaller)
size_t pz_compress_bound(size_t input_len);

#ifdef __cplusplus
}
#endif

#endif  // LIBPZ_H
```

**Rust implementation:**
```rust
use std::os::raw::c_int;
use std::slice;

#[repr(C)]
pub struct pz_context {
    backend: ComputeBackend,
    last_error: Option<String>,
}

#[no_mangle]
pub extern "C" fn pz_init() -> *mut pz_context {
    let ctx = Box::new(pz_context {
        backend: ComputeBackend::probe(),
        last_error: None,
    });
    Box::into_raw(ctx)
}

#[no_mangle]
pub unsafe extern "C" fn pz_destroy(ctx: *mut pz_context) {
    if !ctx.is_null() {
        let _ = Box::from_raw(ctx);  // Drop, frees memory
    }
}

#[no_mangle]
pub unsafe extern "C" fn pz_compress(
    ctx: *mut pz_context,
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_len: usize,
    level: c_int,
    pipeline: c_int,
) -> c_int {
    // Null checks
    if ctx.is_null() || input.is_null() || output.is_null() {
        return PZ_ERROR_INVALID_INPUT;
    }

    let ctx = &mut *ctx;
    let input_slice = slice::from_raw_parts(input, input_len);
    let output_slice = slice::from_raw_parts_mut(output, output_len);

    // Call safe Rust implementation
    match compress_internal(ctx, input_slice, output_slice, level, pipeline) {
        Ok(bytes_written) => bytes_written as c_int,
        Err(e) => {
            ctx.last_error = Some(e.to_string());
            e.to_error_code()
        }
    }
}

// Similar for pz_decompress, pz_query_devices, etc.
```

**Build integration:**
```toml
[lib]
name = "pz"
crate-type = ["staticlib", "cdylib"]  # Produce both .a and .so
```

```bash
# Build
cargo build --release

# Output:
# target/release/libpz.a      (static library)
# target/release/libpz.so     (dynamic library)

# Generate C header
cbindgen --config cbindgen.toml --crate libpz --output target/pz.h
```

**Using from C:**
```c
#include "pz.h"
#include <stdio.h>

int main() {
    pz_context_t* ctx = pz_init();

    uint8_t input[1024] = "Hello, world!";
    uint8_t output[2048];

    int32_t result = pz_compress(ctx, input, 1024, output, 2048,
                                 PZ_LEVEL_DEFAULT, PZ_DEFLATE);
    if (result < 0) {
        printf("Compression failed: %d\n", result);
    } else {
        printf("Compressed to %d bytes\n", result);
    }

    pz_destroy(ctx);
    return 0;
}
```

**Linking:**
```bash
# Static linking
gcc myapp.c -L./target/release -lpz -lpthread -ldl -lm -o myapp

# Dynamic linking
gcc myapp.c -L./target/release -lpz -o myapp
LD_LIBRARY_PATH=./target/release ./myapp
```

---

## 4. Phase 1 Implementation Roadmap

### Phase 1A: Fix C Reference Implementation - COMPLETE (superseded)

**Outcome:** Rather than fixing the 25 C bugs individually, the decision was made to
replace the C reference implementations entirely with Rust. The C reference files
have been removed from the repository.

### Phase 1B: Rust Implementation - COMPLETE

**Outcome:** Rust versions of all core algorithms created in `libpz-rs/`.

**Completed:**
- [x] Project setup (`libpz-rs/` with Cargo.toml)
- [x] Core data structures (frequency, pqueue, huffman tree, LZ77 match types)
- [x] LZ77 implementation (`libpz-rs/src/lz77.rs`)
- [x] Huffman implementation (`libpz-rs/src/huffman.rs`)
- [x] Additional algorithms: BWT, MTF, RLE, range coder
- [x] Compression pipelines (DEFLATE, BW, LZA)
- [x] C API layer via FFI (`libpz-rs/src/ffi.rs`)
- [x] Validation infrastructure (`libpz-rs/src/validation.rs`)

**Remaining work (carried to Phase 2):**
- [x] OpenCL integration from Rust (done: `opencl3` crate, LZ77 + BWT kernels)
- [ ] cbindgen header generation setup
- [ ] Fuzz testing infrastructure
- [x] Corpus testing (Canterbury corpus in benchmarks and tests)

### Phase 1C: Migration Decision - COMPLETE

**Decision: Full Rust migration.**

**Actions taken:**
- C reference files removed (`reference/`, `include/`, `src/main.c`, `test/`)
- Build system updated (autotools trimmed to OpenCL + samples only)
- Rust (`libpz-rs/`) is now the canonical implementation
- OpenCL C engine and kernels retained for GPU compute

---

## 5. Testing Strategy

### Test Pyramid

```
         ┌──────────────┐
         │ Fuzz Tests   │  (libFuzzer, 24/7 CI)
         └──────────────┘
       ┌──────────────────┐
       │ Integration Tests │  (C API, round-trip)
       └──────────────────┘
    ┌───────────────────────┐
    │  Corpus Tests          │  (Canterbury, Silesia)
    └───────────────────────┘
  ┌──────────────────────────────┐
  │      Unit Tests               │  (Per algorithm, per backend)
  └──────────────────────────────┘
```

### Unit Tests (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz77_round_trip() {
        let input = b"banana";
        let mut compressed = vec![0u8; 1024];
        let mut decompressed = vec![0u8; 1024];

        let comp_size = lz77_compress(input, &mut compressed).unwrap();
        let decomp_size = lz77_decompress(&compressed[..comp_size], &mut decompressed).unwrap();

        assert_eq!(&decompressed[..decomp_size], input);
    }

    #[test]
    fn test_huffman_frequencies() {
        let input = b"aaabbc";
        let freq = get_frequency(input);
        assert_eq!(freq.count(b'a'), 3);
        assert_eq!(freq.count(b'b'), 2);
        assert_eq!(freq.count(b'c'), 1);
    }
}
```

### Corpus Tests

**Standard compression corpora:**
- **Canterbury Corpus** (11 files, diverse types)
- **Silesia Corpus** (256MB, realistic data)
- **enwik8** (100MB Wikipedia text)
- **E.coli genome** (4.6MB biological data)

**Validation:**
```rust
#[test]
fn test_corpus_round_trip() {
    for file in corpus_files() {
        let input = fs::read(file).unwrap();
        let compressed = compress(&input).unwrap();
        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, input, "Failed on {}", file);
    }
}
```

### Fuzz Testing

**Setup with cargo-fuzz:**
```bash
cargo install cargo-fuzz
cargo fuzz init
```

**Fuzz targets:**
```rust
// fuzz/fuzz_targets/compress.rs
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut output = vec![0u8; data.len() * 2];
    let _ = pz_compress(data, &mut output);  // Should never crash
});

// fuzz/fuzz_targets/decompress.rs
fuzz_target!(|data: &[u8]| {
    let mut output = vec![0u8; data.len() * 10];
    let _ = pz_decompress(data, &mut output);  // Should never crash
});

// fuzz/fuzz_targets/round_trip.rs
fuzz_target!(|data: &[u8]| {
    let mut compressed = vec![0u8; data.len() * 2];
    if let Ok(comp_size) = pz_compress(data, &mut compressed) {
        let mut decompressed = vec![0u8; data.len()];
        if let Ok(decomp_size) = pz_decompress(&compressed[..comp_size], &mut decompressed) {
            assert_eq!(&decompressed[..decomp_size], data);
        }
    }
});
```

**Run continuously:**
```bash
# Run each fuzz target for 24 hours
cargo fuzz run compress -- -max_total_time=86400
cargo fuzz run decompress -- -max_total_time=86400
cargo fuzz run round_trip -- -max_total_time=86400
```

### Cross-Implementation Validation

**Ensure Rust output matches C output:**
```rust
#[test]
fn test_matches_c_reference() {
    let input = b"test data here";

    // Compress with C implementation
    let c_output = unsafe {
        let mut buf = vec![0u8; 1024];
        let size = c_lz77_compress(input.as_ptr(), input.len(),
                                   buf.as_mut_ptr(), buf.len());
        buf.truncate(size as usize);
        buf
    };

    // Compress with Rust implementation
    let rust_output = lz77_compress(input).unwrap();

    // Should produce identical output
    assert_eq!(rust_output, c_output);
}
```

---

## 6. Build System

### Rust Build (Primary)

**Cargo.toml:**
```toml
[package]
name = "libpz"
version = "0.1.0"
edition = "2021"

[lib]
name = "pz"
crate-type = ["staticlib", "cdylib"]

[dependencies]
opencl3 = { version = "0.9", optional = true }
ash = { version = "0.37", optional = true }
gpu-allocator = { version = "0.25", optional = true }

[dev-dependencies]
criterion = "0.5"  # Benchmarking
proptest = "1.0"   # Property testing

[build-dependencies]
cbindgen = "0.26"
cc = "1.0"         # If we need to compile .c files

[features]
default = ["opencl"]
opencl = ["opencl3"]
vulkan = ["ash", "gpu-allocator"]

[profile.release]
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization
opt-level = 3
```

**Build script (build.rs):**
```rust
use cbindgen::{Config, Language};

fn main() {
    // Generate C header
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let config = Config {
        language: Language::C,
        include_guard: Some("LIBPZ_H".to_string()),
        ..Default::default()
    };

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("target/pz.h");
}
```

**Makefile (wrapper for cargo):**
```makefile
.PHONY: all clean test install

all:
    cargo build --release
    @echo "Built: target/release/libpz.a"
    @echo "Header: target/pz.h"

clean:
    cargo clean

test:
    cargo test
    cargo test --release

install:
    install -D target/release/libpz.a $(PREFIX)/lib/libpz.a
    install -D target/pz.h $(PREFIX)/include/pz.h

benchmark:
    cargo bench
```

### C Build (Legacy, Phase Out)

Keep existing autotools for C reference:
```
./configure --enable-opencl
make
make check
```

Eventually replace with:
```
cargo build --release --features opencl
```

---

## 7. Success Criteria for Phase 1

### Phase 1A (C Fixes) - SUPERSEDED
C reference code replaced by Rust rather than fixed individually.

### Phase 1B (Rust Implementation) - COMPLETE
- [x] Rust LZ77 implementation
- [x] Rust Huffman implementation
- [x] Rust frequency analysis, priority queue
- [x] Additional algorithms (BWT, MTF, RLE, range coder)
- [x] Compression pipelines (DEFLATE, BW, LZA)
- [x] C-callable FFI layer

### Phase 1C (Migration Decision) - COMPLETE
- [x] Decision made: full Rust migration
- [x] C reference files removed
- [x] OpenCL files retained
- [x] Build system updated

### Remaining (carry to Phase 2)
- [x] OpenCL integration from Rust side (done: `opencl3` crate, LZ77 batch + BWT bitonic sort kernels)
- [ ] Fuzz tests run for 24h with no crashes
- [x] Corpus test coverage (Canterbury corpus used in benchmarks and tests)
- [x] Cross-platform testing (CI runs on ubuntu, windows, macos)
- [x] Performance benchmarking (criterion benchmarks: `benches/stages.rs`, `benches/throughput.rs`)

---

## 8. Timeline & Resources

### Phase 1 Timeline - COMPLETE

| Phase | Status | Outcome |
|-------|--------|---------|
| 1A: Fix C bugs | SUPERSEDED | Replaced by Rust rewrite |
| 1B: Rust implementation | COMPLETE | All core algorithms ported |
| 1C: Migration decision | COMPLETE | Full Rust migration chosen |

### Resource Requirements (ongoing)

**Development:**
- Rust experience (intermediate level)
- GPU programming knowledge (OpenCL basics)

**Software:**
- Rust toolchain (rustup)
- OpenCL SDK (for GPU backend)
- Vulkan SDK (for Phase 5)

---

## 9. Risk Analysis

### High Risk

**Risk:** Rust rewrite takes longer than expected
- **Mitigation:** Keep C implementation working, migrate incrementally
- **Fallback:** Ship C version for Phase 1 if needed

**Risk:** Rust FFI overhead impacts performance
- **Mitigation:** Benchmark early, profile hot paths
- **Fallback:** Keep performance-critical parts in C if necessary

### Medium Risk

**Risk:** OpenCL Rust bindings missing features
- **Mitigation:** Use `opencl3` (most complete), or write thin FFI wrapper
- **Fallback:** Call OpenCL C API directly via FFI

**Risk:** Binary size bloat from Rust stdlib
- **Mitigation:** Use `opt-level = "z"`, `lto = true`, strip symbols
- **Expected:** ~200-500KB overhead (acceptable for a compression library)

### Low Risk

**Risk:** C API compatibility issues
- **Mitigation:** Test with multiple C compilers (GCC, Clang, MSVC)
- **Validation:** cbindgen generates standard C89/C99

---

## 10. Next Steps (Phase 2)

### Completed since Phase 1

1. **OpenCL Rust integration:** DONE
   - `opencl3` crate replaces the C engine entirely
   - LZ77 batch kernel and BWT bitonic sort kernel implemented
   - GPU BWT produces byte-identical output to CPU BWT

2. **Testing infrastructure:** MOSTLY DONE
   - CI pipeline (GitHub Actions: test on 3 OS, clippy, fmt, OpenCL compile check, benchmarks)
   - Canterbury corpus tests in benchmarks
   - Still pending: fuzz testing with `cargo-fuzz`

3. **Benchmarking:** DONE
   - `benches/stages.rs`: per-algorithm scaling at 1KB/10KB/64KB
   - `benches/throughput.rs`: pipeline throughput + gzip/pigz/zstd comparison
   - GPU benchmarks included when built with `--features opencl`

### Remaining Priorities

- Fuzz testing infrastructure (`cargo-fuzz` targets for all encode/decode paths)
- GPU BWT performance optimization (local memory sort, GPU rank assignment)
- Optimal LZ77 parsing (Phase 2 in PLAN.md): GPU top-K match table + backward DP
- pthread multi-threading via `rayon`
- Vulkan compute backend (Phase 5)

---

## 11. Conclusion

**Phase 1 is complete. The Rust migration is finalized. OpenCL integration is done.**

Key outcomes:
1. **Language:** Full Rust migration (C reference files removed)
2. **GPU:** OpenCL backend fully in Rust via `opencl3` crate — LZ77 batch kernel and BWT bitonic sort kernel
3. **API:** C-callable FFI layer implemented in Rust
4. **Algorithms:** LZ77, Huffman, BWT, MTF, RLE, range coder, pipelines all in Rust
5. **CLI:** `pz` command-line tool with `-g`/`--gpu` flag for OpenCL acceleration
6. **Benchmarks:** Criterion benchmarks for per-stage scaling and end-to-end throughput vs gzip/pigz/zstd
7. **CI:** GitHub Actions workflow (CPU tests on 3 OS, lint, OpenCL compile check, benchmarks)

Next priorities:
- **GPU performance:** BWT kernel optimization (local memory sort, GPU rank assignment)
- **Phase 2:** Optimal LZ77 parsing (GPU top-K match table + backward DP)
- **Phase 4:** pthread multi-threading (via `rayon`)
- **Phase 5:** Vulkan compute
