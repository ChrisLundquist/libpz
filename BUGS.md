# libpz Bug Plan

Comprehensive list of bugs found during code review, organized by severity.
Each bug includes location, description, and suggested fix.

---

## Critical: Memory Safety / Correctness

### BUG-01: `pq_pop` reads out-of-bounds sentinel and corrupts heap

**File:** `reference/pqueue.c:35-46`

The sift-down loop initializes `k = h->len + 1` as a sentinel, then compares
`h->nodes[k].priority` against child priorities. This reads uninitialized or
stale memory beyond the logical end of the heap. If neither child is smaller
than the garbage value, the garbage node gets copied into the heap, corrupting
it. This affects Huffman tree construction which depends on the priority queue.

```c
i = 1;
while (i != h->len + 1) {
    k = h->len + 1;  // BUG: used as sentinel but accessed as real data
    j = 2 * i;
    if (j <= h->len && h->nodes[j].priority < h->nodes[k].priority) {
        k = j;
    }
    if (j + 1 <= h->len && h->nodes[j + 1].priority < h->nodes[k].priority) {
        k = j + 1;
    }
    h->nodes[i] = h->nodes[k];  // may copy garbage into heap
    i = k;
}
```

**Fix:** Rewrite sift-down to compare children against the current node (not a
sentinel), and stop when the current node is already smaller than both children.

---

### BUG-02: `FindMatchClassic` spot-check reads out of bounds

**File:** `reference/lz77.c:21`

```c
if (search[i + best.length] != target[best.length])
    continue;
```

No bounds check on `i + best.length` against `search_size`. When `i` is near
the end of the search window and `best.length` is large, `i + best.length`
exceeds `search_size`. While the contiguous memory layout (`search + search_size
== target`) may prevent segfaults, the comparison is semantically wrong --
it compares bytes in the target area against target, potentially skipping
valid matches or accepting invalid ones.

**Fix:** Add a guard: `if (i + best.length >= search_size) continue;` before the
spot check.

---

### BUG-03: `FindMatchClassic` inner loop bounds check uses wrong size

**File:** `reference/lz77.c:27`

```c
while (tail < target_size && search[tail] == target[temp_match_length]) {
```

`tail` is an index into the `search` buffer (starting at `tail = i`), but it is
bounded by `target_size` instead of `search_size`. When `search_size <
target_size`, `tail` can grow past `search_size`, reading into the `target`
area. This relies on contiguous memory layout as an accidental safety net, but
produces incorrect match offsets and lengths.

**Fix:** Use a bound that correctly limits both buffers. The condition should
ensure `tail < search_size + target_size` (total contiguous span) AND
`temp_match_length < target_size` (don't match past the lookahead).

---

### BUG-04: OpenCL kernels check bounds AFTER memory access

**File:** `opencl/lz77.cl:18`, `opencl/lz77-batch.cl:18`

```c
while (search[tail] == target[temp_match_length] && (tail < target_size))
```

The memory access `search[tail]` is evaluated before the bounds check
`tail < target_size`. If `tail >= target_size` on a given iteration, the
out-of-bounds read happens before the condition short-circuits.

The reference implementation (`reference/lz77.c:27`) has the correct order
(`tail < target_size` first), but the OpenCL kernels have it reversed.

**Fix:** Swap the condition order:
`while ((tail < target_size) && search[tail] == target[temp_match_length])`

---

### BUG-05: `huff_Encode` uses signed `char` as array index -- negative index UB

**File:** `reference/huffman.c:204-205`

```c
const char current_byte = in[index];
const huffman_node_t node = tree->leaves[current_byte];
```

`current_byte` is `char` (signed on most platforms). Input bytes >= 128
produce negative values, causing `tree->leaves[negative_index]` to access
memory before the array. This is undefined behavior and will likely segfault
or return garbage.

**Fix:** Change to `const unsigned char current_byte = (unsigned char)in[index];`

---

### BUG-06: `LZ77_Decompress` buffer overflow on literal byte write

**File:** `reference/lz77.c:145-156`

```c
if (((outpos - out) + m.length) > (outsize)) {
    break;
}
// ... copies m.length bytes ...
*outpos = m.next;   // writes 1 more byte WITHOUT checking buffer space
outpos++;
```

The bounds check accounts for `m.length` bytes but not the additional byte
written for `m.next`. When the output buffer is nearly full, this writes one
byte past the end. The same bug exists in `opencl/engine.c:316-327`
(`DecodeLZ77`).

**Fix:** Check for `m.length + 1` in the bounds check:
`if (((outpos - out) + m.length + 1) > outsize)`

---

### BUG-07: `ReadProgramFile` doesn't null-terminate OpenCL source

**File:** `opencl/engine.c:74-90`

```c
char* text = malloc(sb.st_size);       // no +1 for null terminator
fread(text, 1, sb.st_size, file);      // no null termination
```

The source is then passed to `clCreateProgramWithSource` with `NULL` for the
lengths parameter, which tells OpenCL the string is null-terminated. Since it
isn't, OpenCL will read past the buffer.

**Fix:** Allocate `sb.st_size + 1`, and set `text[sb.st_size] = '\0'`. Or pass
the actual length to `clCreateProgramWithSource` instead of `NULL`.

---

## High: Logic Errors / Incorrect Behavior

### BUG-08: `huff_Encode` never writes to the output buffer

**File:** `reference/huffman.c:193-216`

The function calculates `bits_encoded` and `bit_offset` but never writes any
data to the `out` buffer. The `// TODO write to output` comment on line 212
confirms this is unfinished. Any code that expects actual encoded output gets
an untouched buffer.

**Fix:** Implement bit-packing logic that writes codewords into the output buffer
using the calculated `codeword` and `code_bits` for each input symbol.

---

### BUG-09: `huff_Decode` is a stub that returns 0

**File:** `reference/huffman.c:219-225`

```c
unsigned huff_Decode(const huffman_tree_t* tree,
                     const char* in, unsigned in_len,
                     char* out, unsigned out_len) {
    return 0;
}
```

Completely unimplemented. Any test that tries to round-trip through
Huffman encode/decode will silently produce no output.

**Fix:** Implement tree-walking decode: read bits from input, traverse the
Huffman tree from root, and emit leaf values when reached.

---

### BUG-10: `merge_nodes` has dead code after NULL check

**File:** `reference/huffman.c:52-59`

```c
if (left == NULL || right == NULL) {
    fprintf(stderr, "%s: left and right nodes are required\n", __func__);
    return NULL;
}
if (left && !right)    // Dead: both are non-NULL here
    return left;
if (right && !left)    // Dead: both are non-NULL here
    return right;
```

After the first check returns NULL on any NULL input, both `left` and `right`
are guaranteed non-NULL. The subsequent single-node checks are unreachable. The
function should either handle single-node inputs (remove the first check) or
remove the dead code.

**Fix:** Remove the dead `if (left && !right)` and `if (right && !left)` blocks,
and remove the redundant `if (left)` / `if (right)` guards on lines 67-72 and
81-84 as well.

---

### BUG-11: `GetCodec` sets state to READY for unknown codecs

**File:** `opencl/engine.c:337-352`

```c
switch (name) {
    case LZ77: { ... break; }
    default:
        fprintf(stderr, "Unknown codec %d\n", name);
        // falls through to codec.state = READY!
}
codec.state = READY;
```

When an unknown codec name is passed, the default case prints an error but
falls through, setting `state = READY` on a codec with NULL function pointers.
Any subsequent call to `codec.Encode()` or `codec.Decode()` will dereference
NULL.

**Fix:** Add `return codec;` in the default case (returning with `INVALID`
state), or set `codec.state = READY` only inside the successful case blocks.

---

### BUG-12: `huff_Encode` doesn't check output buffer size

**File:** `reference/huffman.c:202`

```c
/* TODO/FIXME check output size has room */
```

Even if the encode logic were implemented, there is no check that the encoded
bits fit within `out_len`. Writing encoded data without this check would cause
a buffer overflow.

**Fix:** Before writing each codeword, verify that `(bits_encoded + code_bits) / 8 < out_len`.

---

## Medium: Type Mismatches / API Issues

### BUG-13: `opencl_codec_t` function pointer declares `out` as `const char*`

**File:** `opencl/engine.h:20-21`

```c
int (*Encode)(..., const char* out, unsigned out_len);
int (*Decode)(..., const char* out, unsigned out_len);
```

Both function pointers declare `out` as `const char*`, but the actual
implementations (`EncodeLZ77`, `DecodeLZ77`) write to `out`. This is a type
mismatch between the function pointer and the actual function, causing
undefined behavior when assigned. Some compilers will silently allow the
assignment, but the const violation is UB.

**Fix:** Change both function pointer signatures to `char* out`.

---

### BUG-14: `get_frequency` / `huff_new_8` signedness mismatch

**File:** `include/frequency.h:15` vs `reference/huffman.c:149-151`

`get_frequency` expects `const unsigned char*` for its input, but `huff_new_8`
accepts `const char*` and passes it directly to `get_frequency`. This is a
signedness mismatch that compilers may warn about, and it interacts with
BUG-05 (signed char as array index).

**Fix:** Either change `huff_new_8` to accept `const unsigned char*`, or add an
explicit cast at the call site.

---

### BUG-15: `get_frequency` loop counter `i` is `unsigned` but `in_len` is `long`

**File:** `reference/frequency.c:7-46`

```c
unsigned i = 0;
long remaining = in_len;
// ...
for (; i < in_len; ++i) {
```

If `in_len > UINT_MAX` (~4GB), `i` wraps around to 0 and the loop runs
forever. The unrolled loop also has this issue since it increments `i` 32 times
per iteration without checking for `unsigned` overflow.

**Fix:** Change `i` to `unsigned long` or `size_t`.

---

### BUG-16: `clGetPlatformIDs` receives pointer-to-array instead of pointer

**File:** `opencl/engine.c:10`

```c
cl_platform_id platforms[2];
int err = clGetPlatformIDs(2, &platforms, NULL);
```

`&platforms` is `cl_platform_id (*)[2]` (pointer to array of 2), but the
function expects `cl_platform_id*`. Should be `platforms` (which decays to
`cl_platform_id*`).

**Fix:** Change `&platforms` to `platforms`.

---

### BUG-17: `clCreateCommandQueueWithProperties` receives wrong pointer type

**File:** `opencl/engine.c:42`

```c
const cl_queue_properties ooq[] = { ... };
engine->commands = clCreateCommandQueueWithProperties(
    engine->context, engine->device_id, &ooq, &err);
```

Same issue: `&ooq` is a pointer to the array, not a pointer to the first
element. Should be `ooq`.

**Fix:** Change `&ooq` to `ooq`.

---

### BUG-18: Missing include guards in all header files

**Files:** `include/lz77.h`, `include/huffman.h`, `include/frequency.h`,
`include/codec.h`, `reference/pqueue.h`, `reference/lz77.h`, `opencl/engine.h`

None of the header files have `#pragma once` or `#ifndef` include guards.
Including any header twice in a translation unit will cause duplicate type
definition errors.

**Fix:** Add include guards to all headers. Example:
```c
#ifndef LIBPZ_LZ77_H
#define LIBPZ_LZ77_H
// ... contents ...
#endif
```

---

## Low: Resource Leaks / Missing Cleanup

### BUG-19: `EncodeLZ77` leaks `cl_mem` objects on error paths

**File:** `opencl/engine.c:186-198, 204-208`

When `clSetKernelArg` or `clGetKernelWorkGroupInfo` fails, the function returns
early without releasing `input` and `output` `cl_mem` objects allocated on lines
175-178.

**Fix:** Add `clReleaseMemObject(input); clReleaseMemObject(output);` before each
early return after the cl_mem objects are created.

---

### BUG-20: `CreateEngine` ignores errors from `FindDevice` / `CreateContext`

**File:** `opencl/engine.c:57-62`

```c
opencl_engine_t CreateEngine() {
    opencl_engine_t engine;
    FindDevice(&engine);      // return value ignored
    CreateContext(&engine);   // return value ignored
    return engine;
}
```

If either function fails, the engine is returned with uninitialized fields.
Subsequent use of the engine will cause undefined behavior.

**Fix:** Check return values and propagate errors (e.g., return an error code
through a pointer parameter, or add a status field to `opencl_engine_t`).

---

### BUG-21: `pq_free` doesn't reset struct fields

**File:** `reference/pqueue.c:50-52`

```c
void pq_free(heap_t* h) {
    free(h->nodes);
}
```

Doesn't set `h->nodes = NULL`, `h->len = 0`, `h->size = 0`. Any subsequent
access to the heap would use a dangling pointer. While `build_tree` only calls
`pq_free` once after finishing, defensive cleanup prevents use-after-free bugs.

**Fix:** Set `h->nodes = NULL; h->len = 0; h->size = 0;` after freeing.

---

## Incomplete / Missing Functionality

### BUG-22: `huff_new_16` and `huff_new_32` declared but never defined

**File:** `include/huffman.h:26-27`

```c
huffman_tree_t* huff_new_16(const short* in, unsigned in_len);
huffman_tree_t* huff_new_32(const int* in, unsigned in_len);
```

Declared in the public header but have no implementation. Calling either
function will produce a linker error.

**Fix:** Either implement the functions or remove the declarations from the header.

---

### BUG-23: `LZ77_Compress` / `LZ77_Decompress` not declared in `include/lz77.h`

**File:** `include/lz77.h`

The public header only declares the `lz77_match_t` struct and `PrintMatch`. The
actual compress/decompress functions are declared in the separate
`reference/lz77.h` header. This means the public API header is incomplete.

**Fix:** Add `LZ77_Compress` and `LZ77_Decompress` declarations to `include/lz77.h`,
or consolidate the headers.

---

### BUG-24: `frequency_test.c` `test_simple` doesn't test anything

**File:** `test/frequency_test.c:80-86`

```c
int test_simple() {
    frequency_t table = new_table();
    unsigned char test[256];
    for(int i = 0; i < 256; i++)
        test[i] = i;
    return 0;  // never calls get_frequency or checks results
}
```

Creates test data but never feeds it to `get_frequency` or validates results.

**Fix:** Call `get_frequency(&table, test, 256)` and verify that each byte has
count 1, `total == 256`, and `used == 256`.

---

### BUG-25: `PrintMatch` defined in two translation units (ODR violation)

**File:** `reference/lz77.c:7-10` and `opencl/engine.c:134-137`

Both files define `inline void PrintMatch(...)` with slightly different format
specifiers (`%d` vs `%u` for offset/length). If both object files are linked
into the same binary, this is a One Definition Rule violation.

**Fix:** Make the functions `static inline`, or consolidate into a single shared
definition.

---

## Summary Table

| ID | Severity | Component | Short Description |
|----|----------|-----------|-------------------|
| BUG-01 | Critical | pqueue | Heap sift-down reads OOB sentinel, corrupts heap |
| BUG-02 | Critical | LZ77 ref | Spot-check reads past search buffer |
| BUG-03 | Critical | LZ77 ref | Inner loop bounded by wrong size variable |
| BUG-04 | Critical | LZ77 OCL | Bounds check after memory access (both kernels) |
| BUG-05 | Critical | Huffman | Signed char used as array index, negative index UB |
| BUG-06 | Critical | LZ77 ref+OCL | Decompressor writes 1 byte past bounds check |
| BUG-07 | Critical | OpenCL engine | Kernel source not null-terminated |
| BUG-08 | High | Huffman | Encode never writes output (TODO stub) |
| BUG-09 | High | Huffman | Decode is unimplemented stub |
| BUG-10 | High | Huffman | Dead code in merge_nodes after NULL check |
| BUG-11 | High | OpenCL engine | Unknown codec gets READY state with NULL fn ptrs |
| BUG-12 | High | Huffman | Encode has no output buffer bounds check |
| BUG-13 | Medium | OpenCL engine | Function pointer has const on writable out param |
| BUG-14 | Medium | Huffman/Freq | Signedness mismatch char* vs unsigned char* |
| BUG-15 | Medium | Frequency | Loop counter unsigned vs long length can overflow |
| BUG-16 | Medium | OpenCL engine | &platforms instead of platforms (wrong type) |
| BUG-17 | Medium | OpenCL engine | &ooq instead of ooq (wrong type) |
| BUG-18 | Medium | All headers | Missing include guards |
| BUG-19 | Low | OpenCL engine | cl_mem leak on error paths in EncodeLZ77 |
| BUG-20 | Low | OpenCL engine | CreateEngine ignores init errors |
| BUG-21 | Low | pqueue | pq_free doesn't reset struct fields |
| BUG-22 | Low | Huffman | huff_new_16/32 declared but not defined |
| BUG-23 | Low | LZ77 | Public header missing compress/decompress decls |
| BUG-24 | Low | Tests | frequency_test test_simple is a no-op |
| BUG-25 | Low | LZ77 | PrintMatch defined in two translation units |
