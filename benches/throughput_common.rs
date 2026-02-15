#![allow(dead_code)]

use criterion::measurement::WallTime;
use criterion::BenchmarkGroup;
use std::path::Path;
use std::time::Duration;

pub fn cap(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);
}

pub fn get_test_data() -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

    for name in &["alice29.txt", "cantrbry.tar"] {
        let path = manifest.join("samples").join(name);
        if path.exists() {
            if let Ok(data) = std::fs::read(&path) {
                if !data.is_empty() {
                    return data;
                }
            }
        }
    }

    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                return decompressed;
            }
        }
    }

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    pattern.repeat(3000)
}

pub fn get_test_data_sized(size: usize) -> Vec<u8> {
    let base = get_test_data();
    if base.len() >= size {
        return base[..size].to_vec();
    }

    let mut data = Vec::with_capacity(size);
    while data.len() < size {
        let remaining = size - data.len();
        let chunk = remaining.min(base.len());
        data.extend_from_slice(&base[..chunk]);
    }
    data
}
