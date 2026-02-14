#[cfg(feature = "webgpu")]
fn main() {
    use pz::lz77;
    use pz::webgpu::WebGpuEngine;
    use std::time::Instant;

    let engine = WebGpuEngine::new().expect("No WebGPU device");

    println!("Cooperative-Stitch LZ77 A/B Test");
    println!("================================\n");

    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    let mut text_256k = Vec::with_capacity(256 * 1024);
    while text_256k.len() < 256 * 1024 {
        let chunk = (256 * 1024 - text_256k.len()).min(pattern.len());
        text_256k.extend_from_slice(&pattern[..chunk]);
    }

    // Load alice29 if available
    let alice = std::fs::read("samples/cantrbry/alice29.txt").ok();

    // Also test with real Canterbury files
    let asyoulik = std::fs::read("samples/cantrbry/asyoulik.txt").ok();
    let kennedy = std::fs::read("samples/cantrbry/kennedy.xls").ok();

    let mut test_files: Vec<(&str, &Vec<u8>)> = vec![("text 256KB", &text_256k)];
    if let Some(ref a) = alice {
        test_files.push(("alice29.txt", a));
    }
    if let Some(ref a) = asyoulik {
        test_files.push(("asyoulik.txt", a));
    }
    if let Some(ref k) = kennedy {
        test_files.push(("kennedy.xls", k));
    }

    for (label, data) in test_files {
        println!("--- {} ({} bytes) ---", label, data.len());

        // CPU
        let cpu_matches = lz77::compress_lazy_to_matches(data).unwrap();
        let cpu_matched: usize = cpu_matches.iter().map(|m| m.length as usize).sum();

        // GPU brute-force (lazy)
        let _ = engine.find_matches(data); // warmup
        let t0 = Instant::now();
        let gpu_bf = engine.find_matches(data).unwrap();
        let bf_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let bf_matched: usize = gpu_bf.iter().map(|m| m.length as usize).sum();

        // GPU coop
        let _ = engine.find_matches_coop(data); // warmup
        let t0 = Instant::now();
        let gpu_coop = engine.find_matches_coop(data).unwrap();
        let coop_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let coop_matched: usize = gpu_coop.iter().map(|m| m.length as usize).sum();

        // Verify coop round-trip
        let mut compressed = Vec::new();
        for m in &gpu_coop {
            compressed.extend_from_slice(&m.to_bytes());
        }
        let decompressed = lz77::decompress(&compressed).unwrap();
        let ok = decompressed == *data;

        let cpu_ratio = cpu_matched as f64 / data.len() as f64;
        let bf_ratio = bf_matched as f64 / data.len() as f64;
        let coop_ratio = coop_matched as f64 / data.len() as f64;

        println!(
            "  CPU:  {:>6} seqs, matched {:.4} ({} bytes)",
            cpu_matches.len(),
            cpu_ratio,
            cpu_matched
        );
        println!(
            "  BF:   {:>6} seqs, matched {:.4} ({} bytes), {:.1} ms",
            gpu_bf.len(),
            bf_ratio,
            bf_matched,
            bf_ms
        );
        println!(
            "  Coop: {:>6} seqs, matched {:.4} ({} bytes), {:.1} ms, rt={}",
            gpu_coop.len(),
            coop_ratio,
            coop_matched,
            coop_ms,
            if ok { "OK" } else { "FAIL" }
        );

        // Multiple timing runs
        let mut bf_times = Vec::new();
        let mut coop_times = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            let _ = engine.find_matches(data);
            bf_times.push(t.elapsed().as_secs_f64() * 1000.0);
            let t = Instant::now();
            let _ = engine.find_matches_coop(data);
            coop_times.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        bf_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        coop_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  BF   median: {:.1} ms", bf_times[2]);
        println!("  Coop median: {:.1} ms", coop_times[2]);
        println!("  Speedup: {:.2}x", bf_times[2] / coop_times[2]);
        println!();
    }
}

#[cfg(not(feature = "webgpu"))]
fn main() {
    eprintln!("Requires --features webgpu");
}
