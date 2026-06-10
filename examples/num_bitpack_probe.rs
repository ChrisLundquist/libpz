//! num-bitpack stage-0 probe: per-plane FSE vs vertical bit-pack byte counts.
//!
//! For each input file, runs the shipping Num front-end (1 MiB blocks, stride
//! sweep, per-plane FSE-gated transform) and measures every routed plane under
//! both entropy stages:
//!   - `fse`      gated-best per-plane FSE (what ships today)
//!   - `bp=`      bitpack of the same (FSE-chosen) transform
//!   - `bp*`      bitpack with its own transform gating
//!   - `oracle`   per-plane min(fse, bp*) — a free per-plane selector
//!
//! Every bitpack encode is round-trip verified against the real plane data.
//!
//! KILL CRITERION (per file): bp* regression > 2pp of original file size vs
//! FSE on the routed planes.
//!
//! Usage: cargo run --release --no-default-features --example num_bitpack_probe -- <files...>

use pz::numeric::{self, bitpack};
use std::collections::BTreeMap;

const BLOCK: usize = 1 << 20; // DEFAULT_BLOCK_SIZE: what the CLI feeds Num

#[derive(Default, Clone)]
struct PlaneAgg {
    blocks: usize,
    plane_len: usize,
    fse: usize,
    bp_same: usize,
    bp_gated: usize,
    oracle: usize,
    // Transform tags seen (fse gate / bitpack gate), for the report.
    fse_xfs: [usize; 3],
    bp_xfs: [usize; 3],
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: num_bitpack_probe <files...>");
        std::process::exit(2);
    }

    let mut all_pass = true;
    for path in &args {
        let data = match std::fs::read(path) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {path}: {e}");
                continue;
            }
        };
        let pass = probe_file(path, &data);
        all_pass &= pass;
    }
    println!("\nOVERALL: {}", if all_pass { "PASS" } else { "KILL" });
    if !all_pass {
        std::process::exit(1);
    }
}

fn probe_file(path: &str, data: &[u8]) -> bool {
    println!("\n=== {path} ({} bytes) ===", data.len());

    // Aggregate per (stride, plane_idx) so per-plane behavior is visible even
    // when different blocks pick different strides.
    let mut agg: BTreeMap<(usize, usize), PlaneAgg> = BTreeMap::new();
    let mut stride_blocks: BTreeMap<usize, usize> = BTreeMap::new();
    let mut store_bytes = 0usize; // STORE blocks: 1-byte header + raw
    let mut overhead = 0usize; // headers + remainders, identical both ways

    for block in data.chunks(BLOCK) {
        let probe = numeric::probe_block(block);
        *stride_blocks.entry(probe.stride).or_default() += 1;
        if probe.stride == 0 {
            store_bytes += 1 + block.len();
            continue;
        }
        overhead += 1 + probe.remainder_len + 5 * probe.planes.len();

        // Round-trip check on the real plane data of this block.
        verify_roundtrip(block, probe.stride);

        for p in &probe.planes {
            let e = agg.entry((probe.stride, p.plane_idx)).or_default();
            e.blocks += 1;
            e.plane_len += p.plane_len;
            e.fse += p.fse_bytes;
            e.bp_same += p.bp_same_bytes;
            e.bp_gated += p.bp_gated_bytes;
            e.oracle += p.fse_bytes.min(p.bp_gated_bytes);
            e.fse_xfs[p.fse_xf as usize] += 1;
            e.bp_xfs[p.bp_gated_xf as usize] += 1;
        }
    }

    println!("blocks by stride: {:?}  (stride 0 = STORE)", stride_blocks);
    println!(
        "{:>6} {:>5} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12} {:>12}",
        "stride",
        "plane",
        "blks",
        "plane_len",
        "fse",
        "bp_same",
        "bp_gated",
        "oracle",
        "bp*/fse",
        "fse_xf(R/D/Z)",
        "bp_xf(R/D/Z)"
    );
    let (mut t_len, mut t_fse, mut t_same, mut t_gated, mut t_oracle) = (0, 0, 0, 0, 0);
    for ((stride, plane), e) in &agg {
        println!(
            "{stride:>6} {plane:>5} {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8.3} {:>12} {:>12}",
            e.blocks,
            e.plane_len,
            e.fse,
            e.bp_same,
            e.bp_gated,
            e.oracle,
            e.bp_gated as f64 / e.fse as f64,
            format!("{}/{}/{}", e.fse_xfs[0], e.fse_xfs[1], e.fse_xfs[2]),
            format!("{}/{}/{}", e.bp_xfs[0], e.bp_xfs[1], e.bp_xfs[2]),
        );
        t_len += e.plane_len;
        t_fse += e.fse;
        t_same += e.bp_same;
        t_gated += e.bp_gated;
        t_oracle += e.oracle;
    }

    let orig = data.len() as f64;
    let pp = |n: usize| n as f64 / orig * 100.0;
    println!("routed plane bytes: {t_len} (+{store_bytes} STORE, +{overhead} headers/remainder)");
    println!(
        "planes total:  fse {t_fse} ({:.2}%)  bp_same {t_same} ({:.2}%)  bp_gated {t_gated} ({:.2}%)  oracle {t_oracle} ({:.2}%)",
        pp(t_fse), pp(t_same), pp(t_gated), pp(t_oracle)
    );
    println!(
        "file ratio:    fse {:.2}%  bp_gated {:.2}%  oracle {:.2}%   (planes + headers + remainder + STORE)",
        pp(t_fse + overhead + store_bytes),
        pp(t_gated + overhead + store_bytes),
        pp(t_oracle + overhead + store_bytes)
    );

    let reg_gated = pp(t_gated) - pp(t_fse);
    let reg_same = pp(t_same) - pp(t_fse);
    let oracle_gain = pp(t_fse) - pp(t_oracle);
    let pass = reg_gated <= 2.0;
    println!(
        "delta vs FSE (pp of original): bp_same {reg_same:+.2}pp  bp_gated {reg_gated:+.2}pp  oracle {:+.2}pp",
        -oracle_gain
    );
    println!(
        "VERDICT [{path}]: {} (bp_gated regression {reg_gated:+.2}pp, kill > +2.00pp)",
        if pass { "PASS" } else { "KILL" }
    );
    pass
}

/// Re-derive the planes of `block` at `stride` and verify the bitpack coder
/// exactly inverts each FSE-gate-transformed plane (real data, every block).
fn verify_roundtrip(block: &[u8], stride: usize) {
    let n = block.len() / stride * stride;
    for k in 0..stride {
        let plane: Vec<u8> = block[..n].iter().skip(k).step_by(stride).copied().collect();
        let enc = bitpack::encode(&plane);
        let dec = bitpack::decode(&enc, plane.len()).expect("bitpack decode");
        assert_eq!(
            dec, plane,
            "bitpack roundtrip FAIL stride={stride} plane={k}"
        );
    }
}
