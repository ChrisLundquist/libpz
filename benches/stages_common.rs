#![allow(dead_code)]

use criterion::measurement::WallTime;
use criterion::BenchmarkGroup;
use std::path::Path;
use std::time::Duration;

pub const SIZES_SMALL: &[usize] = &[8192, 65536];
pub const SIZES_ALL: &[usize] = &[8192, 65536, 4_194_304];
pub const SIZES_LARGE: &[usize] = &[262_144, 4_194_304];

pub fn cap(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(5));
    group.sample_size(10);
}

pub fn get_test_data(size: usize) -> Vec<u8> {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));

    let gz_path = manifest.join("samples").join("cantrbry.tar.gz");
    if gz_path.exists() {
        if let Ok(gz_data) = std::fs::read(&gz_path) {
            if let Ok((decompressed, _)) = pz::gzip::decompress(&gz_data) {
                if decompressed.len() >= size {
                    return decompressed[..size].to_vec();
                }

                let mut data = Vec::with_capacity(size);
                while data.len() < size {
                    let remaining = size - data.len();
                    let chunk = remaining.min(decompressed.len());
                    data.extend_from_slice(&decompressed[..chunk]);
                }
                return data;
            }
        }
    }

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let full = pattern.repeat((size / pattern.len()) + 1);
    full[..size].to_vec()
}
