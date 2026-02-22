#![no_main]
use libfuzzer_sys::fuzz_target;
use pz::pipeline::decompress;

/// Feed arbitrary bytes to decompress — must not panic or crash.
/// Errors are expected and fine; panics and undefined behavior are bugs.
fuzz_target!(|data: &[u8]| {
    let _ = decompress(data);
});
