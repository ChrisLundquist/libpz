//! Stride-decorrelation probe for Num auto-routing threshold design.
//!
//! For each file argument, takes the head 64KB sample (exactly what `pz -a`
//! feeds `select_pipeline`), prints the existing `DataProfile` metrics and the
//! current pipeline selection, then for each candidate stride prints the
//! plane-split + per-plane gated-delta entropy estimate and its gain vs the
//! order-0 byte entropy. Also prints the real `numeric::encode` size on the
//! sample as ground truth for the entropy proxy.
//!
//! Usage: cargo run --release --no-default-features --example stride_probe -- files...

use std::fs;

const SAMPLE: usize = 64 * 1024;
const STRIDES: [usize; 6] = [2, 4, 8, 16, 28, 32];

/// Miller-Madow bias-corrected entropy, matching `analysis::entropy_from_counts`.
fn entropy_of_counts(counts: &[u32; 256], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    let mut h = 0.0;
    let mut observed = 0u32;
    for &c in counts.iter() {
        if c > 0 {
            observed += 1;
            let p = c as f64 / n;
            h -= p * p.log2();
        }
    }
    h + (observed.saturating_sub(1)) as f64 / (2.0 * n * std::f64::consts::LN_2)
}

/// Weighted per-plane min(H(raw), H(delta)) entropy estimate at stride `s`,
/// in bits/byte over the whole sample.
fn plane_gated_entropy(sample: &[u8], s: usize) -> f64 {
    let mut raw = vec![[0u32; 256]; s];
    let mut del = vec![[0u32; 256]; s];
    for (i, &b) in sample.iter().enumerate() {
        let k = i % s;
        raw[k][b as usize] += 1;
        let d = if i >= s {
            b.wrapping_sub(sample[i - s])
        } else {
            b
        };
        del[k][d as usize] += 1;
    }
    let n = sample.len();
    let mut bits = 0.0;
    for k in 0..s {
        let nk = n / s + usize::from(k < n % s);
        let hr = entropy_of_counts(&raw[k], nk);
        let hd = entropy_of_counts(&del[k], nk);
        bits += nk as f64 * hr.min(hd);
    }
    bits / n as f64
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: stride_probe <files...>");
        std::process::exit(2);
    }

    for path in &args {
        let data = match fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("{path}: {e}");
                continue;
            }
        };
        let sample = &data[..data.len().min(SAMPLE)];
        let profile = pz::analysis::analyze(sample);
        let current = pz::pipeline::select_pipeline(sample);

        println!("=== {path} ({} bytes, sample {})", data.len(), sample.len());
        println!(
            "  H0={:.3} match_density={:.3} run_ratio={:.3} distinct={} shape={:?} -> {:?}",
            profile.byte_entropy,
            profile.match_density,
            profile.run_ratio,
            profile.distinct_bytes,
            profile.distribution_shape,
            current
        );

        // Pooled entropy, bias-corrected the same way as the per-plane side.
        let h0 = profile.byte_entropy as f64
            + (profile.distinct_bytes.saturating_sub(1)) as f64
                / (2.0 * sample.len() as f64 * std::f64::consts::LN_2);
        for &s in &STRIDES {
            let hp = plane_gated_entropy(sample, s);
            println!("  S={s:2}  planeH={hp:.3}  gain={:+.3}", h0 - hp);
        }
        println!(
            "  profile: numeric_gain={:.3} numeric_stride={}",
            profile.numeric_gain, profile.numeric_stride
        );

        // Ground truth: what Num actually does on this sample.
        let num_size = pz::numeric::encode(sample).len();
        println!(
            "  numeric::encode(sample) = {} bytes ({:.2}% of sample)",
            num_size,
            100.0 * num_size as f64 / sample.len() as f64
        );
    }
}
