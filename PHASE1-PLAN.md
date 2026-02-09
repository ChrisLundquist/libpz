# libpz Phase 1: Foundation & Architecture Decision

## Executive Summary

Phase 1 focuses on establishing a solid foundation for libpz by:
1. **Deciding on implementation language** (C vs Rust)
2. **Fixing critical bugs** in existing reference implementations
3. **Establishing GPU integration patterns** (OpenCL, Vulkan)
4. **Defining C-callable API** for maximum compatibility
5. **Setting up robust testing infrastructure**

**Recommendation: Hybrid approach with progressive Rust migration**

---

## 1. Language Decision: C vs Rust Analysis

### Current State Assessment

**Codebase Size:** ~1,690 lines of C code
- Reference implementations: LZ77, Huffman, frequency analysis, priority queue
- OpenCL GPU backend: Match finding kernels
- 25 documented bugs, majority are **memory safety issues**

**Bug Category Breakdown:**
- **Critical memory safety (7 bugs):** Out-of-bounds access, buffer overflows, null-termination issues
- **High logic errors (5 bugs):** Incomplete implementations, dead code
- **Medium type/API issues (6 bugs):** Signedness mismatches, wrong pointer types, missing include guards
- **Low resource leaks (3 bugs):** Memory leaks, uninitialized fields
- **Incomplete functionality (4 bugs):** Missing implementations, stub functions

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

### Recommendation: **Hybrid Approach with Progressive Migration**

**Phase 1A: Fix existing C code first**
- Fix all 25 bugs in the C implementation
- Get reference implementation **correct and tested**
- Use as correctness oracle

**Phase 1B: Create Rust implementation in parallel**
- Start with core algorithms (LZ77, Huffman)
- Generate C-callable API via cbindgen
- Link against existing OpenCL kernels (same .cl files)
- Validate against C reference implementation

**Phase 1C: Migrate incrementally**
- Replace C modules with Rust equivalents one at a time
- Keep C API surface for compatibility
- Benchmark Rust vs C (should be comparable or faster)

**Why this approach:**
- **De-risks the migration:** Working C code is fallback
- **Validates correctness:** Rust implementation must match C output
- **Enables comparison:** Real perf/binary size data, not speculation
- **Maintains momentum:** Don't block on complete rewrite
- **Leverages strengths:** Rust for safety, C kernels (.cl files) unchanged

---

## 2. GPU Integration Architecture

### Requirements

1. **Multi-backend support:** OpenCL (primary), Vulkan (future)
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

### Vulkan Integration (Phase 5)

**Why Vulkan:**
- Better driver support on consumer hardware (gaming GPUs)
- Mobile/embedded support (Android, Switch, etc.)
- More control over memory and synchronization
- Compute shaders in SPIR-V (compiled GLSL)

**Rust crates:**
- `ash`: Low-level Vulkan FFI bindings
- `vulkano`: Safe, high-level wrapper (recommended)
- `gpu-allocator`: Memory management

**Shader compilation:**
```bash
# GLSL compute shader -> SPIR-V
glslc lz77.comp -o lz77.comp.spv

# Or use shaderc crate at build time
```

**Implementation pattern (similar to OpenCL):**
```rust
pub struct VulkanLZ77Matcher {
    instance: Arc<Instance>,
    device: Arc<Device>,
    pipeline: Arc<ComputePipeline>,
    descriptor_pool: Arc<DescriptorPool>,
}
```

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

### Phase 1A: Fix C Reference Implementation (2-3 weeks)

**Goal:** Get existing C code correct and tested

**Tasks:**
1. **Fix critical memory safety bugs (1 week)**
   - BUG-01: Priority queue sentinel read
   - BUG-02, BUG-03: LZ77 bounds checks
   - BUG-04: OpenCL kernel bounds check order
   - BUG-05: Signed char array index
   - BUG-06: Decompressor buffer overflow
   - BUG-07: Null termination in OpenCL loader

2. **Complete incomplete implementations (1 week)**
   - BUG-08: Implement `huff_Encode` bit packing
   - BUG-09: Implement `huff_Decode` tree walk
   - BUG-22: Implement or remove `huff_new_16/32`
   - BUG-24: Fix `test_simple` to actually test

3. **Fix API/type issues (3 days)**
   - BUG-13-17: Type mismatches, pointer errors
   - BUG-18: Add include guards to all headers
   - BUG-25: Fix `PrintMatch` ODR violation

4. **Add comprehensive tests (3 days)**
   - Round-trip tests: `decompress(compress(x)) == x`
   - Corpus tests: Canterbury, Silesia
   - Fuzz tests: Integrate libFuzzer + ASan
   - OpenCL tests: Validate GPU output matches CPU

**Deliverables:**
- All 25 bugs fixed
- Test suite passes with 0 ASan/valgrind errors
- Reference implementation is correctness oracle

### Phase 1B: Initial Rust Implementation (3-4 weeks)

**Goal:** Create Rust versions of core algorithms with C API

**Tasks:**
1. **Project setup (2 days)**
   - Create `libpz-rust/` directory
   - Set up Cargo workspace
   - Configure cbindgen
   - Set up CI (GitHub Actions: test, clippy, fmt)

2. **Core data structures (3 days)**
   - Frequency table
   - Priority queue (min-heap)
   - Huffman tree
   - LZ77 match types

3. **LZ77 implementation (1 week)**
   - Hash chain match finder (replicate C logic)
   - Greedy match selection
   - Compress/decompress functions
   - Validate against C reference output

4. **Huffman implementation (1 week)**
   - Tree construction from frequencies
   - Canonical code generation
   - Bit-packed encoding
   - Tree-walk decoding
   - Validate against C reference output

5. **OpenCL integration (1 week)**
   - Device probing
   - Kernel compilation (reuse existing .cl files)
   - LZ77 match finding on GPU
   - Validate GPU output matches CPU output

6. **C API layer (3 days)**
   - Implement `pz_init`, `pz_destroy`
   - Implement `pz_compress`, `pz_decompress`
   - Implement `pz_query_devices`
   - Generate header with cbindgen

7. **Testing (ongoing)**
   - Unit tests for each module
   - Integration tests (C API)
   - Cross-validate with C reference
   - Fuzz testing

**Deliverables:**
- `libpz.a` / `libpz.so` built from Rust
- `pz.h` C header (auto-generated)
- Test suite validates equivalence with C version
- Benchmarks show comparable or better performance

### Phase 1C: Migration & Validation (1-2 weeks)

**Goal:** Replace C modules incrementally, validate correctness

**Tasks:**
1. **Benchmark comparison (2 days)**
   - Compression ratio: Rust vs C
   - Throughput: MB/s for various input sizes
   - Binary size: Compare static library sizes
   - Memory usage: Valgrind massif

2. **Integration testing (3 days)**
   - Build example C program using Rust library
   - Build example C++ program (test C++ compat)
   - Build Python bindings (optional, via cffi)
   - Test on Linux, macOS, Windows (if applicable)

3. **Documentation (2 days)**
   - API documentation (rustdoc + Doxygen)
   - Usage examples
   - Build instructions
   - Migration guide (C → Rust)

4. **Decision point: Full migration vs hybrid**
   - If Rust version is **equivalent or better**: Migrate fully
   - If Rust version has issues: Keep both, iterate
   - Document findings in `MIGRATION-REPORT.md`

**Deliverables:**
- Performance comparison report
- Decision on full Rust migration
- Updated build system (support both)

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

### Phase 1A (C Fixes)
- [ ] All 25 bugs fixed
- [ ] Test suite passes (100% tests pass)
- [ ] No ASan/valgrind errors
- [ ] Round-trip tests pass on Canterbury corpus

### Phase 1B (Rust Implementation)
- [ ] Rust LZ77 matches C output byte-for-byte
- [ ] Rust Huffman matches C output byte-for-byte
- [ ] OpenCL integration works (same kernels)
- [ ] C API works from external C program
- [ ] Fuzz tests run for 24h with no crashes

### Phase 1C (Validation)
- [ ] Compression ratio: Rust >= C (within 1%)
- [ ] Throughput: Rust >= C (within 10%)
- [ ] Binary size: Acceptable overhead (<1MB)
- [ ] All tests pass on Linux, macOS
- [ ] Documentation complete

---

## 8. Timeline & Resources

### Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1A: Fix C bugs | 2-3 weeks | None |
| 1B: Rust implementation | 3-4 weeks | 1A (for validation) |
| 1C: Migration decision | 1-2 weeks | 1A + 1B |
| **Total Phase 1** | **6-9 weeks** | |

### Resource Requirements

**Development:**
- 1 developer (full-time)
- Rust experience (intermediate level)
- GPU programming knowledge (OpenCL basics)

**Hardware:**
- Linux development machine
- GPU for testing (NVIDIA/AMD/Intel, OpenCL support)
- CI/CD runner (GitHub Actions free tier sufficient)

**Software:**
- Rust toolchain (rustup)
- OpenCL SDK (vendor-specific)
- Vulkan SDK (for Phase 5)
- Valgrind, ASan, libFuzzer

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

## 10. Recommendations

### Immediate Actions (Week 1)

1. **Start with Phase 1A:** Fix C bugs
   - Low risk, high value
   - Establishes correctness baseline
   - Provides validation target for Rust

2. **Set up Rust project in parallel:**
   - Create `libpz-rust/` directory
   - Configure Cargo, cbindgen
   - Set up CI pipeline

3. **Validate OpenCL Rust bindings:**
   - Test `opencl3` crate with existing kernels
   - Ensure compatibility with target hardware
   - Document any limitations

### Strategic Direction

**Recommended: Progressive Rust migration**

**Rationale:**
- 48% of bugs are memory safety → Rust prevents these
- Small codebase (1690 LOC) → Rewrite is feasible
- GPU bindings mature in Rust → opencl3, ash/vulkano
- C API easy via cbindgen → No compatibility issues
- Better concurrency → Needed for Phase 4 (pthread)

**Phased approach de-risks:**
- C version works → Always have fallback
- Incremental validation → Catch issues early
- Benchmark-driven → Real data, not speculation

### Alternative: Pure C Path

**If Rust is not viable:**
- Fix all 25 bugs in C
- Add extensive tests (fuzz, ASan, valgrind)
- Use strict compiler flags (`-Wall -Wextra -Werror -Wpedantic`)
- Consider `sparse` or `clang-tidy` for static analysis
- **Accept:** More bugs will occur, ongoing vigilance needed

**Not recommended because:** Memory safety bugs will keep appearing. The 25 documented bugs are just what was found in one code review. More lurk in future code.

---

## 11. Conclusion

**Phase 1 sets the foundation for libpz success.**

Key decisions:
1. **Language:** Hybrid C → Rust migration (progressive)
2. **GPU:** OpenCL (Rust bindings) + Vulkan (future)
3. **API:** C-callable via cbindgen (maximum compatibility)
4. **Testing:** Fuzz + corpus + cross-validation (ensure correctness)

**Next Steps:**
1. Fix 25 C bugs (establish baseline)
2. Implement Rust core algorithms (validate against C)
3. Benchmark and decide on full migration (data-driven)

Phase 1 completion enables:
- **Phase 2:** Optimal LZ77 parsing (GPU matches + CPU DP)
- **Phase 3:** Production OpenCL backend
- **Phase 4:** pthread multi-threading
- **Phase 5:** Vulkan compute

**The Rust path provides:**
- Memory safety (prevents 48% of bugs)
- Fearless concurrency (critical for Phase 4)
- Modern tooling (cargo, clippy, rustfmt)
- GPU ecosystem (opencl3, vulkano, wgpu)
- C compatibility (cbindgen FFI)

**Risk is managed through:**
- Incremental migration (always have fallback)
- Continuous validation (Rust must match C output)
- Benchmark-driven decisions (measure, don't guess)

Phase 1 timeline: **6-9 weeks** to working, tested, Rust-based libpz with C API.
