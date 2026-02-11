# GPU Backend Migration Evaluation: OpenCL → Vulkan / wgpu

## 1. Audit of Current OpenCL Surface

### 1.1 Kernel Inventory

| Kernel file | Kernel entry point(s) | Purpose | Lines | Buffer args | Local memory | Atomics | Barriers |
|---|---|---|---|---|---|---|---|
| `kernels/lz77.cl` | `Encode` | Per-position brute-force LZ77 match finding (128KB window) | 62 | 2 (`__global char*` in, `__global lz77_match_t*` out) + 1 scalar | No | No | No |
| `kernels/lz77_batch.cl` | `Encode` | Batched LZ77: each work-item handles 32 positions (32KB window) | 86 | 2 + 1 scalar | No | No | No |
| `kernels/lz77_topk.cl` | `EncodeTopK` | Top-K (K=4) match candidates per position for optimal parsing | 99 | 2 + 1 scalar | No | No | No |
| `kernels/lz77_hash.cl` | `BuildHashTable`, `FindMatches` | Two-pass hash-table LZ77: scatter positions then search buckets | 128 | 4 + 1 scalar (build: 4, find: 5) | No | `atomic_inc` (build pass) | No |
| `kernels/bwt_sort.cl` | `bitonic_sort_step` | One compare-and-swap step of bitonic sort for BWT suffix array | 84 | 2 + 5 scalars | No | No | No |
| `kernels/bwt_radix.cl` | `radix_compute_keys`, `radix_histogram`, `inclusive_to_exclusive`, `radix_scatter` | 4-kernel radix sort for BWT prefix-doubling | 139 | Varies (3-6 per kernel) | Yes: `__local uint local_hist[256]` in `radix_histogram` | `atomic_inc` in `radix_histogram` | `barrier(CLK_LOCAL_MEM_FENCE)` in `radix_histogram` |
| `kernels/bwt_rank.cl` | `rank_compare`, `prefix_sum_local`, `prefix_sum_propagate`, `rank_scatter` | Parallel rank assignment: compare → Blelloch scan → scatter | 167 | Varies (3-5 per kernel) | Yes: `__local uint temp[BLOCK_ELEMS]` in `prefix_sum_local` | No | `barrier(CLK_LOCAL_MEM_FENCE)` (6 instances in scan) |
| `kernels/huffman_encode.cl` | `ComputeBitLengths`, `WriteCodes`, `ByteHistogram`, `PrefixSumBlock`, `PrefixSumApply` | GPU Huffman: bit-length compute, prefix sum, codeword writing, byte histogram | 170 | Varies (3-5 per kernel) | Yes: `__local uint* temp` (dynamic) in `PrefixSumBlock` | `atomic_or` (WriteCodes), `atomic_inc` (ByteHistogram) | `barrier(CLK_LOCAL_MEM_FENCE)` (5 instances in scan) |

**Totals:** 8 kernel files, 18 kernel entry points, 935 lines of OpenCL C.

### 1.2 Host-Side OpenCL API Usage

**File:** `src/opencl.rs` — 2,992 lines (including ~780 lines of tests).

**Integration crate:** `opencl3` v0.9 (safe-ish Rust wrapper over OpenCL C API). Feature-gated behind `opencl` cargo feature.

**OpenCL API patterns used:**

| Pattern | OpenCL calls (via opencl3) | Count | wgpu equivalent exists? |
|---|---|---|---|
| Device discovery | `get_all_devices(CL_DEVICE_TYPE_ALL/GPU)`, `Device::new()` | 4 calls | Yes: `Instance::request_adapter()` |
| Context creation | `Context::from_device()` | 1 | Yes: `Adapter::request_device()` |
| Queue creation | `CommandQueue::create_default_with_properties(CL_QUEUE_PROFILING_ENABLE)` | 1 | Yes: `Device` provides `Queue` |
| Program compilation | `Program::create_and_build_from_source()` with flags (`-Werror`, `-DWORKGROUP_SIZE=N`) | 7 compilations | Yes: `device.create_shader_module()` + pipeline override constants |
| Kernel creation | `Kernel::create(&program, "name")` | 20 kernels | Yes: `device.create_compute_pipeline()` |
| Buffer allocation | `Buffer::<T>::create(&context, CL_MEM_READ_ONLY/WRITE_ONLY/READ_WRITE, size, null)` | ~40 allocations | Yes: `device.create_buffer()` |
| Buffer write (host→device) | `queue.enqueue_write_buffer(&mut buf, CL_BLOCKING, offset, data, &[])` | ~20 calls | Yes: `queue.write_buffer()` |
| Buffer read (device→host) | `queue.enqueue_read_buffer(&buf, CL_BLOCKING, offset, data, &events)` | ~15 calls | Yes: staging buffer + `map_async` |
| Kernel dispatch | `ExecuteKernel::new(&kernel).set_arg(...).set_global_work_size(...).enqueue_nd_range(&queue)` | ~25 dispatches | Yes: `ComputePass::dispatch_workgroups()` |
| Event chaining | `.set_wait_event(&event)` on kernel execution | ~15 uses | Implicit in wgpu (command ordering within encoder) |
| Event wait | `event.wait()` | ~30 calls | `device.poll(Maintain::Wait)` after submit |
| Local memory | `.set_arg_local_buffer(size)` | 2 uses | Declared in shader (`var<workgroup>`) |

**Unsafe code surface:** Every `Buffer::create`, `enqueue_write_buffer`, `enqueue_read_buffer`, and `ExecuteKernel` dispatch is wrapped in `unsafe {}`. The `OpenClEngine` struct also requires `unsafe impl Send` and `unsafe impl Sync`. Approximately **85 `unsafe` blocks** in the host-side code.

### 1.3 OpenCL-Specific Features in Use

| Feature | Used? | Details | Porting difficulty |
|---|---|---|---|
| Local/shared memory | Yes | `bwt_rank.cl` (Blelloch scan), `bwt_radix.cl` (histogram), `huffman_encode.cl` (prefix sum) | Low — direct WGSL/GLSL equivalent |
| 32-bit atomics | Yes | `atomic_inc` (hash build, histogram), `atomic_or` (Huffman write) | Low — both WGSL and GLSL support these |
| 64-bit integers | Yes | `bwt_sort.cl` uses `ulong` for composite sort keys (line 50-62) | **Medium** — WGSL lacks native u64; must use `vec2<u32>` or wgpu `SHADER_INT64` feature |
| Barriers | Yes | `barrier(CLK_LOCAL_MEM_FENCE)` — 11 uses across 3 kernel files | Low — `workgroupBarrier()` / `barrier()` |
| Sub-group/SIMD ops | No | Not used in any kernel | N/A |
| Images/samplers | No | Not used | N/A |
| Printf | No | Not used | N/A |
| Event dependencies | Yes | Fine-grained event chaining for BWT pipeline (sort→compare→scan→scatter) | Low — wgpu handles ordering implicitly; Vulkan uses pipeline barriers |
| Multiple command queues | No | Single queue used | N/A |
| SVM (Shared Virtual Memory) | No | Not used | N/A |
| Compile-time defines | Yes | `-DWORKGROUP_SIZE=N` passed at program compile time | Low — WGSL `override`, GLSL `layout(constant_id)` |
| Dynamic local memory | Yes | `set_arg_local_buffer(size)` for Huffman prefix sum | **Medium** — WGSL requires compile-time-known sizes; must use override constants |
| Profiling | Yes | `CL_QUEUE_PROFILING_ENABLE` on queue creation | Low — wgpu has `Features::TIMESTAMP_QUERY` |

### 1.4 Data Flow per GPU-Accelerated Operation

#### LZ77 Match Finding (PerPosition / Batch / HashTable variants)

```
Host → GPU: input[] (N bytes, one transfer)
             [HashTable only: zero hash_counts[] (128KB)]
GPU compute: [HashTable: BuildHashTable → FindMatches (2 dispatches)]
             [Others: single Encode dispatch]
GPU → Host: gpu_matches[] (N × 12 bytes)
Host CPU:   dedupe_gpu_matches() → Vec<Match>
```

- **Buffer sizes:** input = N bytes, output = N × 12 bytes, hash table = 32768 × 64 × 4 = 8MB (HashTable only)
- **Round-trips:** 1 upload + 1 download (2 for HashTable: extra zero-write)
- **Minimum input:** 64KB (`MIN_GPU_INPUT_SIZE`)

#### Top-K Match Finding (for optimal parsing)

```
Host → GPU: input[] (N bytes)
GPU compute: EncodeTopK (1 dispatch)
GPU → Host: gpu_candidates[] (N × K × 4 bytes, K=4)
Host CPU:   convert to MatchTable
```

- **Buffer sizes:** input = N bytes, output = N × 16 bytes
- **Round-trips:** 1 upload + 1 download

#### BWT Suffix Array Construction (prefix-doubling with radix sort)

```
Host → GPU: sa_host[] (padded_n × 4 bytes), rank_host[] (padded_n × 4 bytes)
Per doubling step (k = 1, 2, 4, ..., until converged):
  GPU: radix_compute_keys → radix_histogram → prefix_sum → inclusive_to_exclusive → radix_scatter
       (×2-8 passes per sort, adaptive)
  GPU: rank_compare → prefix_sum_local [→ prefix_sum_propagate multi-level] → rank_scatter
  GPU → Host: 1 scalar read (max_rank for convergence check)
GPU → Host: sa_final[] (padded_n × 4 bytes)
Host CPU:   extract BWT from suffix array
```

- **Buffer sizes:** ~14 GPU buffers allocated (sa, sa_alt, rank, rank_alt, diff, prefix, keys, histograms ×2, block_sums ×6). For N=256KB input: padded_n=262144, total GPU memory ≈ 14 × 262144 × 4 ≈ **14.7 MB**.
- **Round-trips:** 2 initial uploads + 1 scalar read per doubling step (typically log₂(N) steps ≈ 17-18) + 1 final download
- **Minimum input:** 32KB (`MIN_GPU_BWT_SIZE`)

#### Huffman Encoding (GPU scan path)

```
Host → GPU: input[] (N bytes), code_lut[] (256 × 4 = 1KB)
GPU compute: ComputeBitLengths (1 dispatch)
GPU → Host: last bit_length scalar
GPU compute: PrefixSumBlock [recursive multi-level scan]
GPU → Host: last offset scalar
Host CPU:   compute total_bits
GPU compute: zero output_buf, WriteCodes (1 dispatch)
GPU → Host: output_data[] (total_bits/32 × 4 bytes)
```

- **Round-trips:** 2 uploads + 3 downloads (2 scalars + final output)

#### Deflate Chained (GPU LZ77 + Huffman)

```
[LZ77 match finding as above, results downloaded]
Host CPU:   dedupe + serialize → lz_data[]
Host → GPU: lz_data[] (one upload, reused for histogram + Huffman)
GPU compute: ByteHistogram → download 256×u32 (1KB)
Host CPU:   build Huffman tree, produce code_lut
Host → GPU: code_lut[] (1KB)
GPU compute: ComputeBitLengths → prefix_sum_gpu → WriteCodes
GPU → Host: final bitstream
Host CPU:   serialize container
```

- **Key optimization:** ByteHistogram avoids downloading full LZ77 stream for frequency counting (saves N bytes of transfer, downloads only 1KB).

### 1.5 Rust Integration Details

- **Crate:** `opencl3` v0.9 — provides typed wrappers around OpenCL C API. Not fully safe: buffer operations require `unsafe`.
- **Feature gating:** `#[cfg(feature = "opencl")]` throughout `src/opencl.rs` and in `src/pipeline.rs`.
- **FFI surface:** opencl3 handles the C FFI internally. libpz's direct `unsafe` is for opencl3's API (which exposes `unsafe` methods for buffer ops and kernel execution).
- **Thread safety:** `unsafe impl Send + Sync for OpenClEngine` with a comment that OpenCL 1.2+ guarantees thread safety for handles.
- **`#[repr(C)]` structs:** `GpuMatch` (12 bytes) and `GpuCandidate` (4 bytes) match kernel struct layouts exactly.
- **Kernel embedding:** All `.cl` files are embedded via `include_str!()` and compiled at runtime.

---

## 2. wgpu / WebGPU Feasibility

### 2.1 Host-Side API Mapping

| OpenCL pattern | wgpu equivalent | Notes |
|---|---|---|
| `get_all_devices()` | `Instance::new()` → `instance.request_adapter()` | wgpu selects "best" adapter automatically; can filter by power preference |
| `Context::from_device()` | (implicit in adapter→device) | No separate context object |
| `CommandQueue::create_default_with_properties()` | `adapter.request_device()` returns `(Device, Queue)` | Queue is paired with device |
| `Program::create_and_build_from_source(src, flags)` | `device.create_shader_module(ShaderModuleDescriptor { source: ShaderSource::Wgsl(src) })` | Compiled via naga at module creation; no separate "build" step |
| `-DWORKGROUP_SIZE=N` | `override WORKGROUP_SIZE: u32 = N;` in WGSL + `pipeline_overriding_constants` | Set at pipeline creation time |
| `Kernel::create(&program, "entry")` | `device.create_compute_pipeline(ComputePipelineDescriptor { module, entry_point: "entry", ... })` | Pipeline = shader module + entry point + layout |
| `Buffer::create(ctx, flags, size, null)` | `device.create_buffer(BufferDescriptor { size, usage: BufferUsages::STORAGE \| ... })` | Usage flags instead of CL_MEM flags |
| `queue.enqueue_write_buffer(buf, CL_BLOCKING, 0, data, &[])` | `queue.write_buffer(&buf, 0, data)` | Simpler, always "blocking" from host perspective |
| `queue.enqueue_read_buffer(buf, CL_BLOCKING, 0, data, &[])` | Staging buffer + `slice.map_async()` + `device.poll()` | More explicit; requires staging buffer pattern |
| `ExecuteKernel::new(&kernel).set_arg(&buf).set_global_work_size(n).enqueue_nd_range(&queue)` | `encoder.begin_compute_pass()` → `pass.set_pipeline(&p)` → `pass.set_bind_group(0, &bg)` → `pass.dispatch_workgroups(n/wg, 1, 1)` | More structured; bind groups replace per-arg setting |
| `event.wait()` | `queue.submit([encoder.finish()])` + `device.poll(Maintain::Wait)` | Submit batches commands; poll waits for completion |
| Event chaining (`set_wait_event`) | Implicit: commands within a CommandEncoder execute in order; barriers between passes are automatic | Simpler model, less fine-grained control |
| `set_arg_local_buffer(size)` | Declared in WGSL: `var<workgroup> temp: array<u32, SIZE>;` | Size must be known at shader compile time (use override constants) |

**Patterns without clean wgpu equivalents:**

1. **Fine-grained event dependencies between dispatches** — wgpu sequences commands within an encoder and between submits. The BWT pipeline's event chaining (sort→compare→scan→scatter) maps naturally to sequential dispatch within a single compute pass or multiple passes in one encoder. No gap here.

2. **Dynamic local memory allocation** (`set_arg_local_buffer`) — In OpenCL, the host can pass dynamic local memory size. In WGSL, `var<workgroup>` sizes must be compile-time constants. **Workaround:** Use `override` constants for the size and create different pipelines for different sizes, or use a fixed maximum. Since libpz only uses dynamic local memory in `PrefixSumBlock` where `local_size * 2 * sizeof(u32)` is bounded by workgroup size, this is manageable — create the pipeline with the actual workgroup size baked in.

3. **Null buffer pointer** — In `run_prefix_sum_block`, a null pointer is passed as `block_sums` when no block sums are needed. wgpu doesn't support null bind group entries. **Workaround:** Use a tiny dummy buffer, or have two pipeline variants (with/without block sums).

### 2.2 Rust Integration Assessment

**Reduction in unsafe code:** The current OpenCL backend has ~85 `unsafe` blocks. With wgpu, the compute dispatch path requires **zero** `unsafe` blocks. The `wgpu` crate provides a fully safe API. This eliminates:
- All `Buffer::create` unsafe blocks
- All `enqueue_write_buffer` / `enqueue_read_buffer` unsafe blocks
- All `ExecuteKernel` dispatch unsafe blocks
- The `unsafe impl Send + Sync` (wgpu types are Send + Sync by design)

**Estimated host-side code reduction:** The current 2,200 lines of host code (excluding tests) would shrink to approximately **1,200-1,500 lines** due to:
- Simpler buffer management (no raw pointers)
- No manual event management (implicit ordering)
- No explicit cleanup (RAII via `Arc`)
- Bind group creation adds some verbosity but is offset by other simplifications

### 2.3 Buffer Management

**Current OpenCL pattern:**
```
Host allocates → enqueue_write_buffer (blocking) → kernel uses → enqueue_read_buffer (blocking)
```

**wgpu equivalent:**
```
create_buffer(STORAGE | COPY_SRC) — GPU-side storage
create_buffer(MAP_READ | COPY_DST) — staging for readback
queue.write_buffer(&storage, 0, data) — upload (no staging needed for writes)
[dispatch compute]
encoder.copy_buffer_to_buffer(&storage, &staging) — copy GPU→staging
queue.submit([...])
staging.slice(..).map_async(MapMode::Read, callback)
device.poll(Maintain::Wait)
staging.slice(..).get_mapped_range() — read data
```

**Performance implication:** The staging buffer model adds one GPU-side copy for readback that OpenCL doesn't require (OpenCL reads directly from device memory via DMA). However, this copy is device-to-device and extremely fast (bandwidth-limited, typically >100 GB/s on discrete GPUs). For libpz's use case where individual readbacks are at most a few MB, this overhead is negligible.

For uploads, `queue.write_buffer()` handles staging internally, so there's no additional overhead.

### 2.4 Pipeline and Bind Group Creation

wgpu requires upfront pipeline and bind group creation. Assessment for libpz:

**Pipelines can be created once and reused** across blocks. The `OpenClEngine` already compiles all 20 kernels at initialization — the wgpu equivalent would create ~20 `ComputePipeline` objects at init time. Parameters that change per-dispatch (buffer sizes, scalar arguments) are passed via:
- Different bind groups (rebind buffers per dispatch) — lightweight
- Push constants or uniform buffers (for scalar arguments like `count`, `n`, etc.)

**Bind group layout:** Can be shared across pipelines with the same buffer signature. Most LZ77 kernels share a similar layout (input buffer + output buffer + scalars). BWT and Huffman kernels have their own layouts.

**Overhead assessment:** Pipeline creation is ~1-5ms per pipeline (one-time cost at init). Bind group creation is microseconds. This matches the current OpenCL pattern where kernel compilation happens once at `OpenClEngine::new()`.

### 2.5 Kernel / Shader Porting: OpenCL C → WGSL

#### Local/shared memory (`__local` → `var<workgroup>`)

wgpu guarantees only **16KB minimum** per the WebGPU spec (`maxComputeWorkgroupStorageSize`). Analysis of libpz kernels:

| Kernel | Local memory usage | Fits in 16KB? |
|---|---|---|
| `bwt_rank.cl: prefix_sum_local` | `BLOCK_ELEMS × 4 bytes` = `WORKGROUP_SIZE × 2 × 4`. At WG=256: **2,048 bytes** | Yes |
| `bwt_radix.cl: radix_histogram` | `256 × 4 = 1,024 bytes` | Yes |
| `huffman_encode.cl: PrefixSumBlock` | `local_size × 2 × 4 bytes`. At local_size=256: **2,048 bytes** | Yes |

**No kernel exceeds 16KB.** Maximum usage is 2,048 bytes. No blocker here.

#### Atomics

| OpenCL atomic | WGSL equivalent | Used in |
|---|---|---|
| `atomic_inc(&var)` | `atomicAdd(&var, 1u)` | `lz77_hash.cl`, `bwt_radix.cl`, `huffman_encode.cl` |
| `atomic_or(&var, val)` | `atomicOr(&var, val)` | `huffman_encode.cl` |

Both are fully supported in WGSL on `atomic<u32>` in workgroup and storage address spaces. **No gaps.**

#### Barriers

`barrier(CLK_LOCAL_MEM_FENCE)` → `workgroupBarrier()` — direct 1:1 mapping. **No gaps.**

#### 64-bit Integers

**This is the most significant porting challenge.** `bwt_sort.cl` uses `ulong` (unsigned 64-bit) for composite sort keys:

```c
ulong key_i, key_ixj;
key_i = ((ulong)rank[sa_i] << 32) | (ulong)rank[(sa_i + k) % n];
```

Standard WGSL does **not** have `u64`. Options:

1. **Use wgpu's `SHADER_INT64` native feature** — enables `u64` in WGSL. Works on Vulkan/DX12/Metal backends where hardware supports `shaderInt64`. Not available on web. Since `bwt_sort.cl` (bitonic sort) is a legacy kernel replaced by `bwt_radix.cl`, this may not be needed at all.

2. **Refactor to avoid 64-bit integers** — The radix sort kernels (`bwt_radix.cl`) process the 64-bit composite key as 8 separate bytes, never constructing the full 64-bit value. The only kernel that needs 64-bit is `bwt_sort.cl`, which is already superseded by radix sort. **The active BWT code path does not require u64.**

3. **Emulate with `vec2<u32>`** — If 64-bit comparison is ever needed, pack `(high, low)` into `vec2<u32>` and compare lexicographically. More code but works everywhere.

**Assessment:** Not a blocker. The active radix sort path avoids 64-bit entirely. The legacy bitonic sort can be dropped.

#### Subgroup/warp operations

Not used in any libpz kernel. Future optimization could use subgroup shuffles for prefix sums, but this is optional. wgpu supports subgroups on Vulkan/DX12/Metal via `Features::SUBGROUP`. **No impact.**

#### WGSL specialization constants (`override`)

Current OpenCL uses `-DWORKGROUP_SIZE=N` at compile time. WGSL `override` declarations are the direct replacement:

```wgsl
override WORKGROUP_SIZE: u32 = 256;
override BLOCK_ELEMS: u32 = 512; // WORKGROUP_SIZE * 2
```

Set at pipeline creation time. This is cleaner than the OpenCL approach. **Direct improvement.**

#### Hard blockers in WGSL

**None identified.** Every OpenCL feature used in the active kernel set has a WGSL equivalent.

### 2.6 Platform Coverage

| Platform | wgpu backend | Compute support | Confidence |
|---|---|---|---|
| **Android 7+** | Vulkan | Full compute shader support | High — Vulkan is mature on Android |
| **Android (older/low-end)** | OpenGL ES 3.1 (fallback) | Compute shaders in GLES 3.1+ | Medium — GLES compute is less tested in wgpu |
| **Linux desktop** | Vulkan | Full | High — RADV, ANV, NVIDIA all excellent |
| **Windows 10+** | DX12 (default) or Vulkan | Full | High — both DX12 and Vulkan mature |
| **macOS** | Metal | Full compute support | High — Metal compute is mature and performant |
| **iOS** | Metal | Full compute support | High — Metal is the native API |
| **Raspberry Pi 4/5** | Vulkan (v3dv) or GLES | Compute supported via v3dv (Vulkan 1.3 conformant) | Medium — v3dv is conformant but VideoCore VI/VII has limited compute throughput; some reports of wgpu performance gap vs raw Vulkan |
| **Web/WASM** | WebGPU (browser API) | Compute shaders in WebGPU | Medium — WebGPU now ships in Chrome, Edge, Firefox. Safari support is landing. |

**Key risk: Raspberry Pi.** v3dv has Vulkan 1.3 conformance, but historical reports show wgpu running at ~10x lower performance than raw Vulkan on RPi 4. This may have improved with driver maturation, but needs benchmarking. The GLES fallback path is available.

---

## 3. Raw Vulkan Compute Feasibility

### 3.1 API Complexity

**Estimated host-side code for Vulkan compute dispatch:**

| Component | ash (unsafe) | vulkano (safe) | Current OpenCL (opencl3) |
|---|---|---|---|
| Instance + device + queue setup | 120-150 lines | 40-50 lines | ~60 lines |
| Memory allocation + buffers | 80-100 lines per buffer set | 20-25 lines | ~15 lines per buffer |
| Descriptor sets + layouts | 60-80 lines per layout | 15-20 lines (macro-generated) | N/A (per-arg setting) |
| Pipeline layout + compute pipeline | 40-50 lines per pipeline | 15-20 lines per pipeline | ~5 lines per kernel |
| Command recording + dispatch | 50-70 lines per dispatch | 15-25 lines | ~10 lines per dispatch |
| Synchronization + readback | 30-40 lines | 10-15 lines | ~8 lines |
| Cleanup | 50-70 lines | Automatic (RAII) | Automatic (Drop) |

**Total for the full libpz GPU backend (20 kernels, ~25 dispatch paths):**

| Metric | ash | vulkano | Current OpenCL | wgpu |
|---|---|---|---|---|
| Estimated total host lines | 3,500-4,500 | 1,800-2,200 | ~2,200 | 1,200-1,500 |
| Unsafe blocks | ~200+ | ~0 | ~85 | 0 |

**Rust Vulkan binding recommendation:** If choosing raw Vulkan, **vulkano** is the clear choice for this project. It provides safety guarantees similar to wgpu, generates type-safe descriptor set bindings from GLSL via a proc macro, and handles cleanup via RAII. `ash` would approximately double the code and require pervasive `unsafe`.

### 3.2 Memory Management

Vulkan requires explicit memory type selection (device-local, host-visible, host-coherent). The current OpenCL code uses simple flag-based allocation (`CL_MEM_READ_ONLY`, etc.) without worrying about memory types.

**Mapping:**

| OpenCL flag | Vulkan memory type | Notes |
|---|---|---|
| `CL_MEM_READ_ONLY` (input buffers) | `DEVICE_LOCAL` with staging upload (or `HOST_VISIBLE | HOST_COHERENT` for simplicity) | Device-local is faster for GPU access |
| `CL_MEM_WRITE_ONLY` (output buffers) | `DEVICE_LOCAL` with staging readback | Staging buffer needed for host readback |
| `CL_MEM_READ_WRITE` (working buffers) | `DEVICE_LOCAL` | No host access needed |

For libpz's moderate buffer count (14 buffers for BWT, ~5 for LZ77), manual allocation is feasible but tedious. Using `gpu-allocator` (pure Rust, used by wgpu internally) is recommended. It handles sub-allocation and memory type selection.

### 3.3 Shader Porting: OpenCL C → GLSL Compute

GLSL compute is syntactically closer to OpenCL C than WGSL is, making the port more mechanical:

| Concept | OpenCL C | GLSL compute |
|---|---|---|
| `__kernel void Encode(__global char* in, ...)` | `layout(std430, binding=0) buffer InBuf { uint data[]; }; void main()` |
| `get_global_id(0)` | `gl_GlobalInvocationID.x` |
| `get_local_id(0)` | `gl_LocalInvocationID.x` |
| `get_group_id(0)` | `gl_WorkGroupID.x` |
| `__local uint arr[N]` | `shared uint arr[N];` |
| `barrier(CLK_LOCAL_MEM_FENCE)` | `barrier(); memoryBarrierShared();` |
| `atomic_inc(&x)` | `atomicAdd(x, 1u)` |
| `atomic_or(&x, v)` | `atomicOr(x, v)` |
| `ulong` (64-bit uint) | `uint64_t` (with `GL_EXT_shader_explicit_arithmetic_types_int64`) |

**Key difference:** GLSL compute shaders don't take parameters — all data access is through descriptor-bound SSBOs (Shader Storage Buffer Objects). Scalar parameters like `count` and `n` go in push constants or uniform buffers.

**Vulkan extensions beneficial for libpz:**
- `VK_KHR_shader_subgroup` (core in Vulkan 1.1) — could accelerate prefix sums
- `VK_KHR_shader_int64` — for the legacy bitonic sort (not needed if dropped)
- `VK_KHR_shader_atomic_int64` — not needed (no 64-bit atomics used)

**Porting effort:** GLSL porting is more straightforward than WGSL porting because:
1. GLSL is C-like (closer to OpenCL C syntax)
2. GLSL supports 64-bit integers natively (with extension)
3. Buffer access patterns are similar (just different syntax for declarations)
4. GLSL compilation (via `glslc` or `shaderc`) produces high-quality SPIR-V

### 3.4 Platform Coverage Gaps

| Platform | Vulkan support | Compute status | Risk |
|---|---|---|---|
| **Android (recent)** | Native since Android 7, required since Android 10 | Full compute | Low — but driver quality varies (Mali > Adreno typically) |
| **Linux desktop** | Mature (RADV, ANV, NVIDIA) | Full compute | Very low |
| **Windows** | Mature (all major vendors) | Full compute | Very low |
| **macOS / iOS** | MoltenVK (Vulkan 1.4 over Metal) | Compute works; SPIR-V → MSL conversion adds latency | Medium — not fully conformant; some edge cases may fail |
| **Raspberry Pi 4/5** | v3dv (Vulkan 1.3 conformant) | Compute supported | Low-Medium — conformant but limited GPU |
| **Web** | **Not possible** | N/A | **Hard blocker** if web is a target |

**Critical gap: No web support.** Vulkan cannot run in browsers. If web deployment is ever desired, raw Vulkan is not viable as the sole backend.

---

## 4. Comparative Analysis

### 4.1 Development Effort

| Dimension | wgpu | Raw Vulkan (vulkano) | Raw Vulkan (ash) |
|---|---|---|---|
| **Host-side code** | ~1,200-1,500 lines (↓30-45% vs current) | ~1,800-2,200 lines (≈current) | ~3,500-4,500 lines (↑60-100%) |
| **Kernel porting effort** | Medium — WGSL is different syntax; 64-bit requires workaround for legacy kernel | Low-Medium — GLSL is closer to OpenCL C; mechanical translation | Same as vulkano |
| **Unsafe code surface** | **0 unsafe blocks** | ~0 (vulkano internalizes) | ~200+ unsafe blocks |
| **Build/toolchain complexity** | `cargo add wgpu` — pure Rust dependency; naga compiles shaders | vulkano: proc macro for GLSL→SPIR-V; or external glslc | ash: needs external SPIR-V compiler (glslc/shaderc) + gpu-allocator |
| **Debugging** | Built-in WebGPU validation layer; RenderDoc integration | Vulkan validation layers (LunarG SDK); RenderDoc; GPU vendor tools | Same as vulkano + more raw API surface to debug |

### 4.2 Runtime Performance

#### Dispatch overhead

libpz's dispatch pattern is a small number of **large** dispatches per compression block (not thousands of tiny dispatches). Typical flow: 1-3 dispatches for LZ77, 20-30 dispatches for BWT (per doubling step × radix passes), 3-5 dispatches for Huffman.

- **wgpu overhead:** The validation + translation layer adds ~1-5μs per API call. For libpz's pattern of ~50 dispatches per block with GPU execution times of 1-300ms, the overhead is **<0.1% of total time** — negligible.
- **vulkano overhead:** Minimal over raw Vulkan — safe wrapper with compile-time checks. Similar to ash in practice.
- **ash overhead:** Near-zero Rust overhead over the Vulkan C API.

**Conclusion:** For libpz's workloads, wgpu's validation overhead is immaterial.

#### Memory transfer overhead

- **wgpu:** Adds one GPU→GPU copy for readback (device buffer → staging buffer). This copy runs at device memory bandwidth (100+ GB/s on discrete GPUs). For libpz's largest transfer (BWT suffix array, ~1MB for 256KB input), this adds ~10μs — negligible vs the ~10ms GPU compute time.
- **Vulkan (ash/vulkano):** Can use host-visible device-local memory on UMA architectures (integrated GPUs, mobile), avoiding the staging copy entirely. On discrete GPUs, staging is still needed.

**Conclusion:** No meaningful difference for libpz's use case.

#### Shader codegen quality

- **WGSL → SPIR-V (naga):** naga is fast (~10x faster than SPIRV-Cross) but the generated SPIR-V may be slightly less optimized than hand-tuned SPIR-V. GPU drivers apply their own optimization passes downstream, which largely eliminates the difference.
- **GLSL → SPIR-V (glslc/shaderc):** Mature, well-optimized compiler from Google. Produces high-quality SPIR-V.
- **OpenCL C → device ISA:** The OpenCL runtime compiler has decades of maturity per vendor.

**Assessment:** For the simple, branch-light kernels in libpz, codegen differences between naga and glslc are unlikely to produce measurable performance differences. The GPU work is dominated by memory bandwidth and ALU throughput, not instruction scheduling.

### 4.3 Maintainability

#### Dependency health

| Metric | wgpu | Raw Vulkan (vulkano) | Raw Vulkan (ash) |
|---|---|---|---|
| **Release cadence** | Major release every ~3 months | Irregular (no stable 1.0) | Tracks Vulkan spec (stable) |
| **Breaking changes** | Every major release (quarterly) | Occasional, moderate scope | Rare (thin bindings, spec-stable) |
| **Major users** | Bevy, Firefox, Deno, Servo | Smaller community | wgpu-hal, skia-safe |
| **Bus factor** | High (Mozilla-backed, large contributor base) | Medium (smaller team) | Medium |
| **Spec stability** | WebGPU spec still evolving (but maturing) | Vulkan API is frozen per version | Vulkan API is frozen per version |

#### Future-proofing

- **wgpu:** Tracks the evolving WebGPU spec. Breaking changes every 3 months impose ongoing maintenance. However, changes are well-documented in changelogs and migration guides. The upside is continuous improvement in platform coverage, performance, and feature support.
- **Vulkan:** API is extremely stable — Vulkan 1.0 code from 2016 still compiles. Extensions are additive. The downside is that the API doesn't get simpler over time.
- **OpenCL:** API is stable but the ecosystem is shrinking. Apple dropped OpenCL support. Android never adopted it broadly. Future device coverage will only decrease.

#### Dual backend feasibility

Could both wgpu and raw Vulkan be supported? Yes, via a trait-based abstraction:

```rust
trait GpuBackend {
    fn find_matches(&self, input: &[u8], variant: KernelVariant) -> PzResult<Vec<Match>>;
    fn bwt_encode(&self, input: &[u8]) -> PzResult<BwtResult>;
    fn huffman_encode(&self, input: &[u8], lut: &[u32; 256]) -> PzResult<(Vec<u8>, usize)>;
    fn deflate_chained(&self, input: &[u8]) -> PzResult<Vec<u8>>;
}
```

This abstraction is feasible because the GPU interface is already well-defined at the `OpenClEngine` method boundary. The current public API (`find_matches`, `bwt_encode`, `huffman_encode`, `deflate_chained`) maps directly to this trait. However, maintaining two shader languages (WGSL + GLSL) and two host-side implementations doubles the maintenance surface. **Not recommended unless a specific platform requires it.**

### 4.4 Target Device Assessment

| Platform | wgpu | Raw Vulkan | Current OpenCL |
|---|---|---|---|
| **Android (recent)** | Excellent (Vulkan backend) | Excellent (native) | Poor (limited OpenCL support on Android; few devices ship OpenCL drivers) |
| **Linux desktop** | Excellent (Vulkan backend) | Excellent (RADV/ANV/NVIDIA) | Good (AMD ROCm, Intel, NVIDIA ship OpenCL) |
| **Windows** | Excellent (DX12 or Vulkan) | Excellent (all vendors) | Good (NVIDIA, AMD, Intel ship OpenCL) |
| **macOS** | Excellent (Metal backend) | Good (MoltenVK; some edge cases) | **Deprecated** (Apple deprecated OpenCL in macOS 10.14, removed from new SDKs) |
| **iOS** | Excellent (Metal backend) | Good (MoltenVK) | **Not available** |
| **Raspberry Pi** | Good (Vulkan v3dv; possible perf gap) | Good (v3dv Vulkan 1.3) | Poor (no maintained OpenCL driver for VideoCore) |
| **Web/WASM** | Good (WebGPU in major browsers) | **Not possible** | **Not possible** |

**OpenCL's critical weakness:** No macOS (deprecated), no iOS, no Android (effectively), no Raspberry Pi, no web. The current backend already has severely limited cross-platform reach, which is fundamentally misaligned with the project's goal of targeting phones, SBCs, and low-end laptops.

---

## 5. Migration Strategy Recommendation

### Recommendation: **A) wgpu only**

**Rationale:**

1. **Platform coverage is the decisive factor.** The project explicitly targets "devices where the GPU is significantly more powerful than the CPU: phones, Raspberry Pi-class SBCs, low-end laptops." OpenCL is effectively unavailable on 3 of those 3 targets (phones = Android/iOS, RPi = no OpenCL, low-end laptops = often macOS or ChromeOS). wgpu covers all of them through Vulkan, Metal, and DX12 backends.

2. **No hard blockers identified.** Every OpenCL feature used in the active kernel set (atomics, barriers, local memory, 32-bit integers, the radix sort BWT path) has a WGSL equivalent. The only potential issue (64-bit integers) applies only to the legacy bitonic sort kernel, which is already superseded by radix sort.

3. **wgpu overhead is negligible for libpz's workloads.** The validation layer adds microseconds per call; libpz dispatches tens of commands per block with millisecond-scale GPU execution times. The staging buffer model adds at most one fast device-to-device copy per readback.

4. **Development effort is lower.** wgpu's safe Rust API eliminates ~85 unsafe blocks, reduces host code by 30-45%, and removes the C FFI dependency. The WGSL kernel porting is the main effort (~935 lines of OpenCL C to translate), but the shaders are algorithmically straightforward.

5. **Web support is a bonus.** While not a primary target, wgpu compiles to WASM and targets the browser WebGPU API, which could enable future web-based compression tools at no additional development cost.

6. **Raw Vulkan doesn't justify its complexity.** The only advantage of raw Vulkan over wgpu is slightly lower overhead (irrelevant for libpz) and potentially better RPi performance (unconfirmed; needs benchmarking). The cost is significantly more host code, manual memory management, and no web support. If RPi performance is a problem, wgpu's `wgpu-hal` layer can be used directly for that platform as an escape hatch.

**Why not the other options:**

- **B) Raw Vulkan only** — No web support, no macOS without MoltenVK complexity, more code, more unsafe. The slight performance advantage doesn't justify the cost.
- **C) wgpu primary + Vulkan escape hatch** — Adds maintenance burden of two shader languages and two host implementations. Defer to this only if benchmarking reveals an actual wgpu performance problem on a target platform.
- **D) Abstraction layer with both** — Over-engineering for the current use case. The `GpuBackend` trait is easy to add later if needed.
- **E) Keep OpenCL** — OpenCL is deprecated on macOS, absent on Android/iOS/RPi, and the ecosystem is contracting. Continuing to invest in OpenCL is building on a shrinking foundation.

### Phased Migration Plan

#### Phase 0: Infrastructure Setup

- Add `wgpu` dependency (feature-gated as `gpu`), alongside existing `opencl` feature.
- Define a `GpuBackend` trait matching the current `OpenClEngine` public API.
- Implement the trait for `OpenClEngine` (trivial wrapper).
- Create `WgpuEngine` struct skeleton implementing the trait.
- **Gate:** Both backends compile, existing tests pass.

#### Phase 1: Port LZ77 Hash-Table Kernel (Lowest Risk, Highest Value)

**Port first:** `lz77_hash.cl` (BuildHashTable + FindMatches) — this is the primary dispatch path used by `pipeline.rs`.

- Translate `lz77_hash.cl` → WGSL (128 lines of OpenCL C).
- Implement `WgpuEngine::find_matches()` with the HashTable variant.
- Reuse existing test infrastructure: `test_gpu_lz77_hash_round_trip` and compression round-trip tests.
- Run existing benchmarks (`benches/throughput.rs`) comparing wgpu vs OpenCL.
- **Gate:** Round-trip correctness on Canterbury corpus; wgpu throughput within 20% of OpenCL at 256KB+.

#### Phase 2: Port Remaining LZ77 Kernels

- Translate `lz77.cl` (PerPosition), `lz77_batch.cl` (Batch), `lz77_topk.cl` (TopK) → WGSL.
- Implement all `KernelVariant` paths in `WgpuEngine::find_matches()`.
- Implement `WgpuEngine::find_topk_matches()`.
- **Gate:** All LZ77 variants pass round-trip tests; TopK matches produce valid `MatchTable`.

#### Phase 3: Port Huffman Encoding Kernels

- Translate `huffman_encode.cl` (ComputeBitLengths, WriteCodes, ByteHistogram, PrefixSumBlock, PrefixSumApply) → WGSL.
- Implement `WgpuEngine::huffman_encode()`, `byte_histogram()`, `prefix_sum_gpu()`.
- Handle the dynamic local memory issue (use override constants for prefix sum block size).
- **Gate:** Huffman encode output matches CPU encoder; benchmark at 256KB.

#### Phase 4: Port BWT Radix Sort + Rank Assignment

- Translate `bwt_radix.cl` (4 kernels) + `bwt_rank.cl` (5 kernels) → WGSL.
- Implement `WgpuEngine::bwt_encode()`.
- Drop `bwt_sort.cl` (legacy bitonic sort, not needed).
- This is the most complex phase due to the multi-level prefix sum and radix sort pipeline.
- **Gate:** BWT round-trip correctness; suffix array matches CPU SA-IS output.

#### Phase 5: Port Deflate Chained Pipeline

- Implement `WgpuEngine::deflate_chained()` combining LZ77 + Huffman on GPU.
- Verify the ByteHistogram optimization works (histogram computed on device).
- **Gate:** Deflate round-trip correctness; benchmark against CPU pipeline.

#### Phase 6: Integration + Deprecation

- Wire `WgpuEngine` into `pipeline.rs` via the `GpuBackend` trait.
- Add `gpu` feature flag to `Cargo.toml`, defaulting to off (same as `opencl`).
- Run full Canterbury + Silesia corpus validation with wgpu backend.
- Benchmark on target devices (Android phone, Raspberry Pi, macOS laptop).
- If all gates pass: deprecate `opencl` feature, mark for removal in next major version.
- **Gate:** All existing OpenCL tests pass with wgpu backend; no regression >10% on any benchmark.

### Risks and Unknowns Requiring Prototyping

1. **RPi performance gap.** Historical reports of wgpu running 10x slower than raw Vulkan on RPi 4. Needs benchmarking with a compute workload (not rendering). If confirmed, the escape hatch is `wgpu-hal` direct Vulkan usage for RPi, or the GLES backend.

2. **naga SPIR-V codegen quality for libpz shaders.** The LZ77 brute-force kernels are ALU-heavy with tight loops. Profile naga output vs glslc output on at least one target GPU to verify no codegen regressions.

3. **wgpu quarterly breaking changes.** Budget maintenance time for wgpu upgrades. Pin to a specific wgpu version in `Cargo.toml` and upgrade deliberately (not on every release).

4. **Dynamic local memory workaround.** The PrefixSumBlock kernel uses host-specified local memory size. With WGSL, this must be a compile-time constant. Prototype the override constant approach to verify it works with wgpu's pipeline creation API.

5. **Null buffer pointer workaround.** `run_prefix_sum_block` passes a null pointer for `block_sums` in the single-block case. wgpu doesn't support null bind group entries. Prototype the two-pipeline variant (with/without block_sums) or use a dummy 4-byte buffer.

---

## Appendix: Code Location Reference

### Kernel files
- `kernels/lz77.cl` — per-position brute-force LZ77 (62 lines)
- `kernels/lz77_batch.cl` — batched LZ77, 32 positions per work-item (86 lines)
- `kernels/lz77_topk.cl` — top-K match candidates (99 lines)
- `kernels/lz77_hash.cl` — hash-table two-pass LZ77 (128 lines)
- `kernels/bwt_sort.cl` — legacy bitonic sort step (84 lines)
- `kernels/bwt_radix.cl` — 4-kernel radix sort (139 lines)
- `kernels/bwt_rank.cl` — 5-kernel parallel rank assignment (167 lines)
- `kernels/huffman_encode.cl` — 5-kernel Huffman encoding (170 lines)

### Host-side code
- `src/opencl.rs:1-247` — imports, constants, structs, device discovery
- `src/opencl.rs:264-429` — `OpenClEngine::new()` (context, queue, kernel compilation)
- `src/opencl.rs:459-510` — `find_matches()` (buffer alloc, dispatch, readback, dedupe)
- `src/opencl.rs:512-634` — `run_per_position_kernel`, `run_batch_kernel`, `run_hash_kernel`
- `src/opencl.rs:655-725` — `find_topk_matches()`
- `src/opencl.rs:732-1034` — `bwt_encode()` + `bwt_build_suffix_array()` (most complex: 14 buffers, multi-level prefix sum, radix sort)
- `src/opencl.rs:1036-1187` — `run_radix_sort()` (adaptive pass selection, 4 kernel phases per pass)
- `src/opencl.rs:1189-1407` — rank compare, prefix sum (multi-level), rank scatter helpers
- `src/opencl.rs:1409-1474` — `byte_histogram()`
- `src/opencl.rs:1476-1632` — `huffman_encode()` (CPU prefix sum variant)
- `src/opencl.rs:1639-1768` — `prefix_sum_gpu()`, `run_prefix_sum_block`, `run_prefix_sum_apply`
- `src/opencl.rs:1774-1930` — `huffman_encode_gpu_scan()` (GPU prefix sum variant)
- `src/opencl.rs:1932-2168` — `deflate_chained()` (full GPU LZ77→Huffman pipeline)
- `src/opencl.rs:2179-2214` — `dedupe_gpu_matches()` helper
- `src/opencl.rs:2216-2992` — tests

### Pipeline integration
- `src/pipeline.rs:41-42` — `Backend::OpenCl` variant
- `src/pipeline.rs:81-82` — `PipelineOptions::opencl_engine` field
- `src/pipeline.rs:936-960` — `lz77_compress_with_options()` GPU/CPU dispatch
- `src/pipeline.rs:985-990` — GPU chained Deflate path
- `src/pipeline.rs:1064-1081` — BWT GPU/CPU dispatch
