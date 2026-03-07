// Quick MTF-1 evaluation: compare BWT -> MTF -> RLE -> FSE vs BWT -> MTF-1 -> RLE -> FSE
use pz::{bwt, fse, mtf, rle};

fn main() {
    let files = [
        ("samples/cantrbry/alice29.txt", "alice29.txt"),
        ("samples/cantrbry/asyoulik.txt", "asyoulik"),
        ("samples/cantrbry/cp.html", "cp.html"),
        ("samples/cantrbry/fields.c", "fields.c"),
        ("samples/cantrbry/grammar.lsp", "grammar.lsp"),
        ("samples/cantrbry/kennedy.xls", "kennedy.xls"),
        ("samples/cantrbry/lcet10.txt", "lcet10.txt"),
        ("samples/cantrbry/plrabn12.txt", "plrabn12.txt"),
        ("samples/cantrbry/ptt5", "ptt5"),
        ("samples/cantrbry/sum", "sum"),
        ("samples/cantrbry/xargs.1", "xargs.1"),
        ("samples/large/E.coli", "E.coli"),
        ("samples/large/world192.txt", "world192.txt"),
    ];

    println!(
        "{:<20} {:>8} {:>10} {:>10} {:>10} {:>16}",
        "File", "Size", "MTF", "MTF-1", "Diff", "zeros(mtf/mtf1)"
    );
    println!("{}", "-".repeat(90));

    let mut total_orig = 0usize;
    let mut total_mtf = 0usize;
    let mut total_mtf1 = 0usize;

    for (path, label) in &files {
        match std::fs::read(path) {
            Ok(data) => {
                let bwt_result = bwt::encode(&data).unwrap();
                let mtf_out = mtf::encode(&bwt_result.data);
                let rle_out = rle::encode(&mtf_out);
                let fse_out = fse::encode(&rle_out);
                let mtf1_out = mtf::encode_mtf1(&bwt_result.data);
                let rle1_out = rle::encode(&mtf1_out);
                let fse1_out = fse::encode(&rle1_out);

                total_orig += data.len();
                total_mtf += fse_out.len();
                total_mtf1 += fse1_out.len();

                let mtf_zeros = mtf_out.iter().filter(|&&b| b == 0).count();
                let mtf1_zeros = mtf1_out.iter().filter(|&&b| b == 0).count();
                let mtf_ratio = fse_out.len() as f64 / data.len() as f64 * 100.0;
                let mtf1_ratio = fse1_out.len() as f64 / data.len() as f64 * 100.0;
                let diff = mtf1_ratio - mtf_ratio;

                println!(
                    "{:<20} {:>8} {:>9.1}% {:>9.1}% {:>+9.2}pp {}/{}",
                    label,
                    data.len(),
                    mtf_ratio,
                    mtf1_ratio,
                    diff,
                    mtf_zeros,
                    mtf1_zeros
                );
            }
            Err(e) => println!("{}: {}", label, e),
        }
    }

    println!("{}", "-".repeat(90));
    let total_mtf_ratio = total_mtf as f64 / total_orig as f64 * 100.0;
    let total_mtf1_ratio = total_mtf1 as f64 / total_orig as f64 * 100.0;
    let diff = total_mtf1_ratio - total_mtf_ratio;
    println!(
        "{:<20} {:>8} {:>9.1}% {:>9.1}% {:>+9.2}pp",
        "TOTAL", total_orig, total_mtf_ratio, total_mtf1_ratio, diff
    );
}
