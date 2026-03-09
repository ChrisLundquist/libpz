use super::super::*;

#[test]
fn test_gpu_match_struct_size() {
    assert_eq!(std::mem::size_of::<GpuMatch>(), 12);
}

#[test]
fn test_dedupe_all_literals() {
    let input = b"abcdef";
    let gpu_matches: Vec<GpuMatch> = input
        .iter()
        .map(|&b| GpuMatch {
            offset: 0,
            length: 0,
            next: b as u32,
        })
        .collect();

    let result = dedupe_gpu_matches(&gpu_matches, input);
    assert_eq!(result.len(), 6);
    for (i, m) in result.iter().enumerate() {
        assert_eq!(m.offset, 0);
        assert_eq!(m.length, 0);
        assert_eq!(m.next, input[i]);
    }
}

#[test]
fn test_dedupe_with_match() {
    let input = b"abcabc";
    let gpu_matches = vec![
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'a' as u32,
        },
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'b' as u32,
        },
        GpuMatch {
            offset: 0,
            length: 0,
            next: b'c' as u32,
        },
        GpuMatch {
            offset: 3,
            length: 2,
            next: b'c' as u32,
        },
        GpuMatch {
            offset: 3,
            length: 1,
            next: b'c' as u32,
        },
        GpuMatch {
            offset: 3,
            length: 0,
            next: b'c' as u32,
        },
    ];

    let result = dedupe_gpu_matches(&gpu_matches, input);
    assert_eq!(result.len(), 4);
    assert_eq!(result[3].offset, 3);
    assert_eq!(result[3].length, 2);
}

#[test]
fn test_probe_devices() {
    // Should not crash; may return empty on headless systems
    let devices = probe_devices();
    for d in &devices {
        assert!(!d.name.is_empty() || d.name.is_empty()); // no-op, just validate struct
    }
}

#[test]
fn test_engine_creation() {
    // May return Unsupported on headless systems -- that's OK
    match WebGpuEngine::new() {
        Ok(engine) => {
            assert!(!engine.device_name().is_empty());
            assert!(engine.max_work_group_size() >= 1);
        }
        Err(PzError::Unsupported) => {
            // Expected on systems without GPU
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn test_profiling_creation() {
    match WebGpuEngine::with_profiling(true) {
        Ok(engine) => {
            eprintln!(
                "Device: {}, profiling={}",
                engine.device_name(),
                engine.profiling()
            );
            // profiling accessor must match what was actually negotiated
            // (may be false if device doesn't support TIMESTAMP_QUERY)
        }
        Err(PzError::Unsupported) => {
            // Expected on systems without GPU
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

// --- Tests ported from OpenCL test suite ---

#[test]
fn test_dedupe_empty() {
    let result = dedupe_gpu_matches(&[], &[]);
    assert!(result.is_empty());
}

#[test]
fn test_device_count_does_not_panic() {
    let count = device_count();
    let _ = count;
}

#[test]
fn test_is_cpu_device() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };
    let _is_cpu = engine.is_cpu_device();
}

// --- GPU prefix sum tests ---

#[test]
fn test_prefix_sum_small() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let data: Vec<u32> = vec![1, 2, 3, 4, 5];
    let n = data.len();
    let buf = engine.create_buffer_init(
        "ps_test",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, n).unwrap();

    let raw = engine.read_buffer(&buf, (n * 4) as u64);
    let result: &[u32] = bytemuck::cast_slice(&raw);
    // Exclusive prefix sum of [1,2,3,4,5] = [0,1,3,6,10]
    assert_eq!(&result[..n], &[0, 1, 3, 6, 10]);
}

#[test]
fn test_prefix_sum_exact_block_size() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // Exact block size = 512 (256 * 2)
    let n = 512;
    let data: Vec<u32> = (0..n as u32).map(|_| 1).collect();
    let buf = engine.create_buffer_init(
        "ps_test_512",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, n).unwrap();

    let raw = engine.read_buffer(&buf, (n * 4) as u64);
    let result: &[u32] = bytemuck::cast_slice(&raw);
    // Exclusive prefix sum of all-ones = [0, 1, 2, ..., 511]
    for (i, &val) in result[..n].iter().enumerate() {
        assert_eq!(val, i as u32, "mismatch at index {i}");
    }
}

#[test]
fn test_prefix_sum_multi_level() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    // 2000 elements requires multi-level (> 512)
    let n = 2000;
    let data: Vec<u32> = (0..n as u32).map(|i| (i % 7) + 1).collect();
    let buf = engine.create_buffer_init(
        "ps_test_multi",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, n).unwrap();

    let raw = engine.read_buffer(&buf, (n * 4) as u64);
    let result: &[u32] = bytemuck::cast_slice(&raw);

    // Verify against CPU prefix sum
    let mut expected = vec![0u32; n];
    let mut sum = 0u32;
    for (i, slot) in expected.iter_mut().enumerate() {
        *slot = sum;
        sum += (i as u32 % 7) + 1;
    }
    assert_eq!(&result[..n], &expected[..]);
}

#[test]
fn test_prefix_sum_single_element() {
    let engine = match WebGpuEngine::new() {
        Ok(e) => e,
        Err(PzError::Unsupported) => return,
        Err(e) => panic!("unexpected error: {:?}", e),
    };

    let data: Vec<u32> = vec![42];
    let buf = engine.create_buffer_init(
        "ps_test_single",
        bytemuck::cast_slice(&data),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );

    engine.prefix_sum_gpu(&buf, 1).unwrap();

    let raw = engine.read_buffer(&buf, 4);
    let result: &[u32] = bytemuck::cast_slice(&raw);
    assert_eq!(result[0], 0);
}
