# Integrate wgpu-profiler for GPU timestamp profiling

## Prerequisite

wgpu must be upgraded to version 27 first. See `upgrade-wgpu-to-27.md`.

## Goal

Replace the hand-rolled timestamp query infrastructure in `src/webgpu/mod.rs` with `wgpu-profiler = "0.25"`, gaining:
- Automatic query pool management (no more fixed 2-query limit)
- Nested scope timing (multi-pass sequences like LZ77 build→find→resolve get individual timings)
- Chrome trace export (`chrome://tracing` flamegraphs) for structured, agent-readable profiling output
- No more blocking `poll(Wait)` for timestamp readback

## Cargo.toml changes

Add to `[dependencies]`:
```toml
wgpu-profiler = { version = "0.25", optional = true }
```

Add to `[features]`:
```toml
webgpu = ["dep:wgpu", "dep:pollster", "dep:bytemuck", "dep:wgpu-profiler"]
```

## What to remove from `src/webgpu/mod.rs`

The following hand-rolled timestamp infrastructure becomes redundant:

1. **Fields on `WebGpuEngine`** (lines ~252-256):
   - `query_set: Option<wgpu::QuerySet>` — remove
   - `resolve_buf: Option<wgpu::Buffer>` — remove

2. **Query set creation** in `create()` (lines ~343-358): the block that creates `query_set` and `resolve_buf` when `use_timestamps` is true — remove entirely.

3. **`read_and_print_timestamps()`** method (lines ~611-651): the entire method that resolves queries, maps a staging buffer synchronously, reads two timestamps, and prints elapsed time — remove entirely.

4. **Timestamp writes in `record_dispatch()`** (lines ~587-594): the manual `ComputePassTimestampWrites` construction using `self.query_set` — replace with wgpu-profiler scope.

5. **Timestamp handling in `dispatch()`** (lines ~671-679): the conditional `read_and_print_timestamps` call and wall-clock fallback — replace with profiler frame processing.

## What to add to `src/webgpu/mod.rs`

### New field on `WebGpuEngine`
```rust
/// GPU profiler for timestamp queries (None when profiling disabled).
profiler: Option<wgpu_profiler::GpuProfiler>,
```

### Profiler creation in `create()`
When `profiling && supports_timestamps`:
```rust
let profiler = if use_timestamps {
    Some(wgpu_profiler::GpuProfiler::new(
        &device,
        wgpu_profiler::GpuProfilerSettings::default(),
    ).map_err(|_| PzError::InternalError)?)
} else {
    None
};
```
Request `wgpu::Features::TIMESTAMP_QUERY` as before (already done).

### Update `record_dispatch()` to accept optional profiler scope

The key integration point. Currently `record_dispatch` creates a `ComputePass` with optional `ComputePassTimestampWrites`. Replace with:

If profiler is available, use `profiler.begin_pass_query(label, encoder)` to get a `GpuProfilerQuery` whose `compute_pass_timestamp_writes()` provides the timestamp writes. Then `profiler.end_query()` when the pass is done.

If profiler is not available, create the pass with `timestamp_writes: None` as before.

### Update `dispatch()` method

After `queue.submit()`, don't call `read_and_print_timestamps`. Instead, the caller is responsible for ending the profiler frame and processing results (see below).

### Add frame lifecycle methods

```rust
/// Resolve profiler queries into the encoder. Call before encoder.finish().
pub fn profiler_resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
    if let Some(ref mut profiler) = self.profiler {
        profiler.resolve_queries(encoder);
    }
}

/// End the current profiler frame. Call after queue.submit().
pub fn profiler_end_frame(&mut self) -> Option<Vec<wgpu_profiler::GpuTimerQueryResult>> {
    let profiler = self.profiler.as_mut()?;
    profiler.end_frame().ok()?;
    self.device.poll(wgpu::Maintain::Wait);
    profiler.process_finished_frame(self.queue.get_timestamp_period())
}

/// Write profiler results to a chrome trace file.
pub fn profiler_write_trace(
    &self,
    path: &std::path::Path,
    results: &[wgpu_profiler::GpuTimerQueryResult],
) -> std::io::Result<()> {
    wgpu_profiler::chrometrace::write_chrometrace(path, results)
}
```

## Call sites that need `profiler_resolve` before `encoder.finish()`

Search for all places that call `encoder.finish()` or `self.queue.submit()` in `src/webgpu/`:

1. **`dispatch()` in mod.rs** — single-pass convenience method. Add resolve before finish.
2. **`read_buffer()` in mod.rs** — buffer readback helper. This creates its own encoder for copy commands, not profiled — skip.
3. **`read_buffer_scalar_u32()` in mod.rs** — same as above — skip.
4. **`read_and_print_timestamps()` in mod.rs** — being removed — skip.
5. **`find_matches()` in lz77.rs** — creates encoder, records multiple passes (build, find, resolve), submits. Add resolve before finish.
6. **`find_matches_to_device()` in lz77.rs** — similar multi-pass pattern. Add resolve.
7. **`find_matches_batched()` in lz77.rs** — batched multi-block path. Add resolve.
8. **`bwt_encode()` in bwt.rs** — multi-pass BWT. Add resolve.
9. **`huffman_encode_gpu_scan()` in huffman.rs** — multi-pass Huffman. Add resolve.
10. **`byte_histogram()` in huffman.rs** — single dispatch. Add resolve.
11. **`fse_encode_gpu()` in fse.rs** — FSE encode. Add resolve.

For each site, the pattern is:
```rust
// Before (existing):
self.queue.submit(Some(encoder.finish()));

// After:
if let Some(ref mut profiler) = self.profiler {
    profiler.resolve_queries(&mut encoder);
}
self.queue.submit(Some(encoder.finish()));
```

Note: `resolve_queries` takes `&mut self` on `GpuProfiler`, so we need `&mut self` on the WebGpuEngine methods, OR store the profiler in a `RefCell`/`Mutex` if `&self` is required. Check the current signatures — most methods already take `&self` which will be a friction point. The simplest fix is to wrap the profiler in `std::cell::RefCell` since WebGpuEngine is not Sync (it's behind `Arc` in pipeline code but only accessed from one thread at a time). Alternatively, restructure to pass `&mut self`. Examine actual usage patterns before deciding.

## Update `examples/webgpu_profile.rs`

This is the biggest user-facing payoff. After each benchmark phase, collect profiler results and accumulate them. At the end, write a chrome trace:

```rust
// At end of run():
if let Some(results) = engine.profiler_end_frame() {
    let trace_path = std::path::Path::new("profiling/webgpu_trace.json");
    wgpu_profiler::chrometrace::write_chrometrace(trace_path, &results).unwrap();
    eprintln!("Chrome trace written to {}", trace_path.display());
}
```

Integrate the output into the `profiling/` directory structure. Consider using the same `<sha>/<description>.json` naming that `scripts/profile.sh` uses.

## Update `scripts/webgpu_profile.sh`

Add the same `--save-only` / `--web` / `--features` / output path conventions as `scripts/profile.sh`. The chrome trace file is the primary output — print its path at the end.

## Keep the `profiling` field

Keep the `self.profiling` bool on `WebGpuEngine`. When profiling is off, `self.profiler` is `None` and all profiler calls are no-ops. The `eprintln!` wall-clock fallback (CPU `Instant::now()` timing) can stay as a secondary output for quick sanity checks, controlled by `self.profiling` independently of whether GPU timestamps are available.

## Verification

```bash
cargo build --features webgpu
cargo test --features webgpu
cargo clippy --all-targets --features webgpu
cargo build                     # default features
cargo test                      # all tests

# Functional test (if GPU available):
cargo run --features webgpu --example webgpu_profile
# Should produce a chrome trace file in profiling/
```

## Important notes

- `wgpu-profiler` is thread-safe (uses `parking_lot` internally) but `resolve_queries` and `end_frame` take `&mut self`. Plan the mutability story before writing code.
- The `GpuProfiler::new()` constructor takes `&wgpu::Device` — it clones it internally. This is fine since our engine already owns the device.
- When TIMESTAMP_QUERY is not supported by the device, `wgpu-profiler` gracefully returns no timing data (scopes are still opened/closed, just no timestamps). This matches our existing fallback behavior.
- Chrome trace files can be viewed at `chrome://tracing` or `edge://tracing` in any Chromium browser, or with Perfetto UI (https://ui.perfetto.dev/). They're also machine-parseable JSON.
- The `scoped_compute_pass` helper on scope objects handles `ComputePassTimestampWrites` automatically — this is the cleanest integration point for `record_dispatch`.
