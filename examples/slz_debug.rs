use pz::pipeline::{self, CompressOptions, MatchFinder, Pipeline};
fn main() {
    let data = std::fs::read("samples/silesia/dickens").unwrap();

    // Test 1: Single 128KB block (should work - single-block path)
    let block = &data[..128 * 1024];
    let opts = CompressOptions {
        threads: 1,
        match_finder: MatchFinder::SortLz,
        ..Default::default()
    };
    match pipeline::compress_with_options(block, Pipeline::LzSeqR, &opts) {
        Ok(c) => match pipeline::decompress(&c) {
            Ok(d) if d == block => println!(
                "single 128KB block: OK {:.1}%",
                c.len() as f64 / block.len() as f64 * 100.0
            ),
            Ok(d) => println!("single 128KB block: MISMATCH len={}", d.len()),
            Err(e) => println!("single 128KB block: DECOMPRESS ERROR {:?}", e),
        },
        Err(e) => println!("single 128KB block: COMPRESS ERROR {:?}", e),
    }

    // Test 2: Two 128KB blocks via multi-block path
    let two_blocks = &data[..256 * 1024];
    let opts2 = CompressOptions {
        threads: 1,
        match_finder: MatchFinder::SortLz,
        block_size: 128 * 1024,
        ..Default::default()
    };
    match pipeline::compress_with_options(two_blocks, Pipeline::LzSeqR, &opts2) {
        Ok(c) => match pipeline::decompress(&c) {
            Ok(d) if d == two_blocks => println!(
                "two 128KB blocks: OK {:.1}%",
                c.len() as f64 / two_blocks.len() as f64 * 100.0
            ),
            Ok(d) => println!("two 128KB blocks: MISMATCH len={}", d.len()),
            Err(e) => println!("two 128KB blocks: DECOMPRESS ERROR {:?}", e),
        },
        Err(e) => println!("two 128KB blocks: COMPRESS ERROR {:?}", e),
    }

    // Test 3: Same thing with hashchain (should work)
    let opts3 = CompressOptions {
        threads: 1,
        match_finder: MatchFinder::HashChain,
        block_size: 128 * 1024,
        ..Default::default()
    };
    match pipeline::compress_with_options(two_blocks, Pipeline::LzSeqR, &opts3) {
        Ok(c) => match pipeline::decompress(&c) {
            Ok(d) if d == two_blocks => println!(
                "two 128KB hc: OK {:.1}%",
                c.len() as f64 / two_blocks.len() as f64 * 100.0
            ),
            Ok(d) => println!("two 128KB hc: MISMATCH len={}", d.len()),
            Err(e) => println!("two 128KB hc: DECOMPRESS ERROR {:?}", e),
        },
        Err(e) => println!("two 128KB hc: COMPRESS ERROR {:?}", e),
    }

    // Test 4: Multi-block LzSeqR with sortlz
    match pipeline::compress_with_options(two_blocks, Pipeline::LzSeqR, &opts2) {
        Ok(c) => match pipeline::decompress(&c) {
            Ok(d) if d == two_blocks => println!(
                "two 128KB lzseqr-slz: OK {:.1}%",
                c.len() as f64 / two_blocks.len() as f64 * 100.0
            ),
            Ok(d) => println!("two 128KB lzseqr-slz: MISMATCH len={}", d.len()),
            Err(e) => println!("two 128KB lzseqr-slz: DECOMPRESS ERROR {:?}", e),
        },
        Err(e) => println!("two 128KB lzseqr-slz: COMPRESS ERROR {:?}", e),
    }
}
