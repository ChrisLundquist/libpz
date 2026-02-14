# Upgrade wgpu from 24 to 27

## Goal

Upgrade the `wgpu` dependency from version 24 to 27. This is a prerequisite for integrating `wgpu-profiler = "0.25"` which requires wgpu 27.

## Cargo.toml changes

In `Cargo.toml`, update:
```
wgpu = { version = "24", optional = true }
```
to:
```
wgpu = { version = "27", optional = true }
```

Also check if `pollster` and `bytemuck` need version bumps to stay compatible with wgpu 27. Current versions: `pollster = "0.4"`, `bytemuck = "1"`.

## Scope

All WebGPU code is feature-gated behind `--features webgpu`. The files to update:

- `Cargo.toml` — dependency version
- `src/webgpu/mod.rs` (1147 lines) — engine init, device creation, dispatch, buffer helpers, pipeline compilation
- `src/webgpu/lz77.rs` (1041 lines) — GPU LZ77 match finding
- `src/webgpu/huffman.rs` (580 lines) — GPU Huffman encode
- `src/webgpu/bwt.rs` (656 lines) — GPU BWT
- `src/webgpu/fse.rs` (393 lines) — GPU FSE encode
- `src/webgpu/tests.rs` (1494 lines) — tests
- `examples/webgpu_profile.rs` — benchmark harness
- `examples/gpu_compare.rs` — GPU comparison benchmark

WGSL kernel files (`kernels/*.wgsl`) should not need changes — the WGSL spec is stable across these wgpu versions.

## wgpu API surface used by this project

The codebase uses a focused subset of the wgpu API. Here's what to watch for:

1. **Instance/Adapter/Device creation**: `Instance::new`, `request_adapter`, `request_device`, `enumerate_adapters`, `adapter.get_info()`, `adapter.limits()`, `adapter.features()`
2. **Buffer operations**: `device.create_buffer`, `device.create_buffer_init` (via `DeviceExt`), `BufferUsages`, `buffer.slice()`, `slice.map_async()`, `slice.get_mapped_range()`, `buffer.unmap()`
3. **Compute pipelines**: `device.create_shader_module` with `ShaderSource::Wgsl`, `device.create_compute_pipeline`, `ComputePipelineDescriptor`
4. **Command encoding**: `device.create_command_encoder`, `encoder.begin_compute_pass`, `ComputePassDescriptor`, `ComputePassTimestampWrites`, `pass.set_pipeline`, `pass.set_bind_group`, `pass.dispatch_workgroups`
5. **Query sets**: `device.create_query_set` with `QueryType::Timestamp`, `encoder.resolve_query_set`
6. **Queue operations**: `queue.submit`, `queue.get_timestamp_period`
7. **Device polling**: `device.poll(wgpu::Maintain::Wait)`
8. **Bind groups**: `device.create_bind_group_layout`, `device.create_bind_group`, `device.create_pipeline_layout`
9. **Enums/types**: `Backends::all()`, `PowerPreference`, `DeviceType`, `Features::TIMESTAMP_QUERY`, `Limits::downlevel_defaults()`, `MemoryHints::Performance`, `MapMode::Read`

## Migration approach

1. Bump the version in Cargo.toml
2. Run `cargo build --features webgpu` and fix compile errors one at a time
3. Common wgpu breaking changes between major versions:
   - Struct field additions (descriptors gain new required fields)
   - Method signature changes (especially around `request_device`, `create_shader_module`)
   - Enum variant renames
   - `InstanceDescriptor`, `DeviceDescriptor`, `Limits` API changes
4. After it compiles, run `cargo test --features webgpu` — all GPU tests should pass (they skip gracefully if no GPU device is available)
5. Run `cargo clippy --all-targets --features webgpu` — zero warnings

## Verification

```bash
cargo build --features webgpu
cargo test --features webgpu
cargo clippy --all-targets --features webgpu
cargo build                    # verify default features (webgpu) still work
cargo test                     # verify all tests pass
```

Also verify the examples still compile:
```bash
cargo build --features webgpu --example webgpu_profile
cargo build --features webgpu --example gpu_compare
```

## Important notes

- Do NOT change any algorithm logic, only fix API compatibility
- Do NOT change WGSL kernel files unless wgpu 27 requires WGSL syntax changes (unlikely)
- The `webgpu` feature is in default features (`default = ["webgpu"]`), so `cargo build` with no flags also exercises this code
- Keep the `opencl` feature working — `cargo build --features opencl` must still compile
- If a wgpu API change is ambiguous, check the wgpu changelog at https://github.com/gfx-rs/wgpu/blob/trunk/CHANGELOG.md
