/// Streaming compression and decompression with bounded memory usage.
///
/// These functions read from `std::io::Read` and write to `std::io::Write`,
/// processing one block at a time. Memory usage is bounded by
/// `O(num_threads * block_size)` regardless of total input size.
///
/// The streaming path uses the V2 **framed** container format: a standard
/// 8-byte PZ header followed by `num_blocks = 0xFFFFFFFF` (framed sentinel),
/// then self-framing blocks (`[compressed_len][original_len][data]`), and a
/// 4-byte EOS sentinel (`compressed_len = 0`).
///
/// Files produced by `compress_stream` are readable by both `decompress_stream`
/// and the in-memory `pipeline::decompress`.
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use crate::pipeline::{
    compress_block, decompress_block, resolve_thread_count, write_header, CompressOptions,
    DecompressOptions, Pipeline, BLOCK_HEADER_SIZE, FRAMED_SENTINEL, MAGIC, VERSION,
};
use crate::{PzError, PzResult};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error type for streaming operations.
///
/// Wraps both compression/decompression errors (`PzError`) and I/O errors.
/// Kept separate from `PzError` so that `PzError` retains `Clone + PartialEq`.
#[derive(Debug)]
pub enum StreamError {
    /// Compression or decompression algorithm error.
    Pz(PzError),
    /// I/O error from `Read` or `Write` operations.
    Io(io::Error),
}

impl std::fmt::Display for StreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StreamError::Pz(e) => write!(f, "{}", e),
            StreamError::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for StreamError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StreamError::Pz(e) => Some(e),
            StreamError::Io(e) => Some(e),
        }
    }
}

impl From<PzError> for StreamError {
    fn from(e: PzError) -> Self {
        StreamError::Pz(e)
    }
}

impl From<io::Error> for StreamError {
    fn from(e: io::Error) -> Self {
        StreamError::Io(e)
    }
}

/// Result type for streaming operations.
pub type StreamResult<T> = Result<T, StreamError>;

// ---------------------------------------------------------------------------
// Streaming compression
// ---------------------------------------------------------------------------

/// Compress from a reader to a writer using the V2 framed container format.
///
/// Reads input in `options.block_size` chunks, compresses each block, and
/// writes framed output as blocks complete. When `options.threads` > 1,
/// blocks are compressed in parallel with bounded memory: a slow writer
/// naturally throttles how far ahead the compressor runs.
///
/// Returns the total number of bytes written to `output`.
pub fn compress_stream<R: Read + Send, W: Write>(
    input: R,
    output: W,
    pipeline: Pipeline,
    options: &CompressOptions,
) -> StreamResult<u64> {
    let num_threads = resolve_thread_count(options.threads);

    if num_threads <= 1 {
        compress_stream_single(input, output, pipeline, options)
    } else {
        compress_stream_parallel(input, output, pipeline, options, num_threads)
    }
}

/// Decompress from a reader to a writer.
///
/// Handles both V2 table-mode and V2 framed-mode containers. For table-mode
/// input, the entire compressed stream is read into memory first (unavoidable
/// since the block table precedes block data). For framed-mode input, blocks
/// are decompressed and written progressively.
///
/// When `options.backend` is a GPU backend with an engine, GPU-amenable decode
/// stages (e.g., interleaved FSE for Lzfi) run on the GPU.
///
/// Returns the total number of decompressed bytes written to `output`.
pub fn decompress_stream<R: Read + Send, W: Write>(
    input: R,
    output: W,
    options: &DecompressOptions,
) -> StreamResult<u64> {
    let num_threads = resolve_thread_count(options.threads);

    if num_threads <= 1 {
        decompress_stream_single(input, output, options)
    } else {
        decompress_stream_parallel(input, output, num_threads, options)
    }
}

// ---------------------------------------------------------------------------
// Internal: single-threaded streaming compression
// ---------------------------------------------------------------------------

fn compress_stream_single<R: Read, W: Write>(
    mut input: R,
    mut output: W,
    pipeline: Pipeline,
    options: &CompressOptions,
) -> StreamResult<u64> {
    let block_size = options.block_size;
    let mut bytes_written = 0u64;

    // Write V2 header with orig_len = 0 (unknown in streaming mode)
    let mut header = Vec::new();
    write_header(&mut header, pipeline, 0);
    output.write_all(&header)?;
    bytes_written += header.len() as u64;

    // Write framed sentinel
    output.write_all(&FRAMED_SENTINEL.to_le_bytes())?;
    bytes_written += 4;

    // Read, compress, and write blocks
    loop {
        let block_data = read_block(&mut input, block_size)?;
        if block_data.is_empty() {
            break;
        }

        let original_len = block_data.len();
        let compressed = compress_block(&block_data, pipeline, options)?;

        bytes_written += write_block_frame(&mut output, &compressed, original_len)?;
    }

    // Write EOS sentinel
    output.write_all(&0u32.to_le_bytes())?;
    bytes_written += 4;

    output.flush()?;
    Ok(bytes_written)
}

// ---------------------------------------------------------------------------
// Internal: multi-threaded streaming compression
// ---------------------------------------------------------------------------

fn compress_stream_parallel<R: Read + Send, W: Write>(
    input: R,
    mut output: W,
    pipeline: Pipeline,
    options: &CompressOptions,
    num_threads: usize,
) -> StreamResult<u64> {
    let block_size = options.block_size;

    // Write header + framed sentinel
    let mut header = Vec::new();
    write_header(&mut header, pipeline, 0);
    output.write_all(&header)?;
    output.write_all(&FRAMED_SENTINEL.to_le_bytes())?;
    let mut bytes_written = header.len() as u64 + 4;

    // Channel: reader -> workers (bounded to limit memory)
    let (input_tx, input_rx) = mpsc::sync_channel::<(usize, Vec<u8>)>(num_threads);
    let input_rx = Arc::new(Mutex::new(input_rx));

    // Channel: workers -> writer (bounded for back-pressure)
    let (output_tx, output_rx) =
        mpsc::sync_channel::<(usize, Result<(Vec<u8>, usize), PzError>)>(num_threads);

    // Use thread::scope so all threads are joined before we return.
    // The writer runs on the main thread to avoid Send requirements on W.
    let writer_result: StreamResult<u64> = std::thread::scope(|scope| {
        // Spawn worker threads
        for _ in 0..num_threads {
            let rx = Arc::clone(&input_rx);
            let tx = output_tx.clone();
            let opts = options.clone();
            scope.spawn(move || {
                loop {
                    let (idx, block_data) = match rx.lock().unwrap().recv() {
                        Ok(msg) => msg,
                        Err(_) => break, // channel closed
                    };
                    let original_len = block_data.len();
                    let result = compress_block(&block_data, pipeline, &opts)
                        .map(|compressed| (compressed, original_len));
                    if tx.send((idx, result)).is_err() {
                        break;
                    }
                }
            });
        }
        // Drop the original sender so workers' clones are the only ones.
        drop(output_tx);

        // Spawn reader thread
        let reader_handle = scope.spawn(move || -> StreamResult<()> {
            let mut input = input;
            let mut block_index = 0usize;
            loop {
                let block_data = read_block(&mut input, block_size)?;
                if block_data.is_empty() {
                    break;
                }
                if input_tx.send((block_index, block_data)).is_err() {
                    break; // workers gone
                }
                block_index += 1;
            }
            // input_tx dropped here -> workers will drain and exit
            Ok(())
        });

        // Writer: receive blocks, reorder, write in order.
        // Runs on the current (scoped) thread.
        let mut expected = 0usize;
        let mut reorder: BTreeMap<usize, (Vec<u8>, usize)> = BTreeMap::new();

        for (idx, result) in output_rx {
            let (compressed, original_len) = result?;
            reorder.insert(idx, (compressed, original_len));

            // Flush consecutive blocks starting from expected
            while let Some((comp_data, orig_len)) = reorder.remove(&expected) {
                bytes_written += write_block_frame(&mut output, &comp_data, orig_len)?;
                expected += 1;
            }
        }

        // Check reader for errors
        match reader_handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(StreamError::Pz(PzError::InvalidInput)),
        }

        Ok(bytes_written)
    });

    bytes_written = writer_result?;

    // Write EOS sentinel
    output.write_all(&0u32.to_le_bytes())?;
    bytes_written += 4;

    output.flush()?;
    Ok(bytes_written)
}

// ---------------------------------------------------------------------------
// Internal: single-threaded streaming decompression
// ---------------------------------------------------------------------------

fn decompress_stream_single<R: Read, W: Write>(
    mut input: R,
    mut output: W,
    options: &DecompressOptions,
) -> StreamResult<u64> {
    // Read 8-byte header
    let mut header = [0u8; 8];
    read_exact(&mut input, &mut header)?;

    if header[0] != MAGIC[0] || header[1] != MAGIC[1] {
        return Err(StreamError::Pz(PzError::InvalidInput));
    }
    if header[2] != VERSION {
        return Err(StreamError::Pz(PzError::Unsupported));
    }

    let pipeline = Pipeline::try_from(header[3])?;
    let declared_orig_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    // Read num_blocks
    let mut nb_buf = [0u8; 4];
    read_exact(&mut input, &mut nb_buf)?;
    let num_blocks_raw = u32::from_le_bytes(nb_buf);

    if num_blocks_raw == FRAMED_SENTINEL {
        // Framed mode: read self-framing blocks
        decompress_framed_stream(
            &mut input,
            &mut output,
            pipeline,
            declared_orig_len,
            options,
        )
    } else {
        // Table mode: must read entire remaining input, then use in-memory path
        decompress_table_mode_from_stream(
            &mut input,
            &mut output,
            pipeline,
            declared_orig_len,
            num_blocks_raw,
            options,
        )
    }
}

/// Decompress framed blocks from a reader (single-threaded).
fn decompress_framed_stream<R: Read, W: Write>(
    input: &mut R,
    output: &mut W,
    pipeline: Pipeline,
    declared_orig_len: usize,
    options: &DecompressOptions,
) -> StreamResult<u64> {
    let mut total_decompressed = 0u64;

    loop {
        // Read compressed_len
        let mut len_buf = [0u8; 4];
        read_exact(input, &mut len_buf)?;
        let compressed_len = u32::from_le_bytes(len_buf) as usize;

        // EOS sentinel
        if compressed_len == 0 {
            break;
        }

        // Read original_len
        read_exact(input, &mut len_buf)?;
        let original_len = u32::from_le_bytes(len_buf) as usize;

        // Read compressed block data
        let mut block_data = vec![0u8; compressed_len];
        read_exact(input, &mut block_data)?;

        // Decompress and write
        let decompressed = decompress_block(&block_data, pipeline, original_len, options)?;
        output.write_all(&decompressed)?;
        total_decompressed += decompressed.len() as u64;
    }

    // Validate total length if declared
    if declared_orig_len > 0 && total_decompressed != declared_orig_len as u64 {
        return Err(StreamError::Pz(PzError::InvalidInput));
    }

    output.flush()?;
    Ok(total_decompressed)
}

/// Decompress table-mode V2 from a stream by reading it all into memory first.
fn decompress_table_mode_from_stream<R: Read, W: Write>(
    input: &mut R,
    output: &mut W,
    pipeline: Pipeline,
    declared_orig_len: usize,
    num_blocks_raw: u32,
    options: &DecompressOptions,
) -> StreamResult<u64> {
    let num_blocks = num_blocks_raw as usize;
    if num_blocks == 0 {
        return Err(StreamError::Pz(PzError::InvalidInput));
    }

    // Read the rest of the stream (block table + block data)
    let mut remaining = Vec::new();
    input.read_to_end(&mut remaining)?;

    let table_size = num_blocks * BLOCK_HEADER_SIZE;
    if remaining.len() < table_size {
        return Err(StreamError::Pz(PzError::InvalidInput));
    }

    // Parse block table
    let mut block_entries: Vec<(usize, usize)> = Vec::with_capacity(num_blocks);
    let mut total_orig = 0usize;
    for i in 0..num_blocks {
        let offset = i * BLOCK_HEADER_SIZE;
        let comp_len =
            u32::from_le_bytes(remaining[offset..offset + 4].try_into().unwrap()) as usize;
        let orig_block_len =
            u32::from_le_bytes(remaining[offset + 4..offset + 8].try_into().unwrap()) as usize;
        block_entries.push((comp_len, orig_block_len));
        total_orig += orig_block_len;
    }

    if declared_orig_len > 0 && total_orig != declared_orig_len {
        return Err(StreamError::Pz(PzError::InvalidInput));
    }

    // Decompress blocks sequentially, writing each immediately
    let mut pos = table_size;
    let mut total_decompressed = 0u64;
    for (comp_len, orig_block_len) in &block_entries {
        if pos + comp_len > remaining.len() {
            return Err(StreamError::Pz(PzError::InvalidInput));
        }
        let block_data = &remaining[pos..pos + comp_len];
        pos += comp_len;

        let decompressed = decompress_block(block_data, pipeline, *orig_block_len, options)?;
        output.write_all(&decompressed)?;
        total_decompressed += decompressed.len() as u64;
    }

    output.flush()?;
    Ok(total_decompressed)
}

// ---------------------------------------------------------------------------
// Internal: multi-threaded streaming decompression
// ---------------------------------------------------------------------------

fn decompress_stream_parallel<R: Read + Send, W: Write>(
    mut input: R,
    mut output: W,
    num_threads: usize,
    options: &DecompressOptions,
) -> StreamResult<u64> {
    // Read 8-byte header
    let mut header = [0u8; 8];
    read_exact(&mut input, &mut header)?;

    if header[0] != MAGIC[0] || header[1] != MAGIC[1] {
        return Err(StreamError::Pz(PzError::InvalidInput));
    }
    if header[2] != VERSION {
        return Err(StreamError::Pz(PzError::Unsupported));
    }

    let pipeline = Pipeline::try_from(header[3])?;
    let declared_orig_len = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

    // Read num_blocks
    let mut nb_buf = [0u8; 4];
    read_exact(&mut input, &mut nb_buf)?;
    let num_blocks_raw = u32::from_le_bytes(nb_buf);

    if num_blocks_raw != FRAMED_SENTINEL {
        // Table mode: fall back to reading everything into memory
        return decompress_table_mode_from_stream(
            &mut input,
            &mut output,
            pipeline,
            declared_orig_len,
            num_blocks_raw,
            options,
        );
    }

    // Framed mode: parallel decompression with reader -> workers -> writer

    // Channel: reader -> workers
    let (input_tx, input_rx) = mpsc::sync_channel::<(usize, Vec<u8>, usize)>(num_threads); // (idx, compressed, orig_len)
    let input_rx = Arc::new(Mutex::new(input_rx));

    // Channel: workers -> writer
    let (output_tx, output_rx) = mpsc::sync_channel::<(usize, PzResult<Vec<u8>>)>(num_threads);

    let total: StreamResult<u64> = std::thread::scope(|scope| {
        // Spawn worker threads
        for _ in 0..num_threads {
            let rx = Arc::clone(&input_rx);
            let tx = output_tx.clone();
            let opts = options.clone();
            scope.spawn(move || loop {
                let (idx, block_data, orig_len) = match rx.lock().unwrap().recv() {
                    Ok(msg) => msg,
                    Err(_) => break,
                };
                let result = decompress_block(&block_data, pipeline, orig_len, &opts);
                if tx.send((idx, result)).is_err() {
                    break;
                }
            });
        }
        drop(output_tx);

        // Spawn reader thread: parse framed blocks and feed to workers
        let reader_handle = scope.spawn(move || -> StreamResult<()> {
            let mut block_index = 0usize;
            loop {
                // Read compressed_len
                let mut len_buf = [0u8; 4];
                read_exact(&mut input, &mut len_buf)?;
                let compressed_len = u32::from_le_bytes(len_buf) as usize;

                // EOS sentinel
                if compressed_len == 0 {
                    break;
                }

                // Read original_len
                read_exact(&mut input, &mut len_buf)?;
                let original_len = u32::from_le_bytes(len_buf) as usize;

                // Read compressed block data
                let mut block_data = vec![0u8; compressed_len];
                read_exact(&mut input, &mut block_data)?;

                if input_tx
                    .send((block_index, block_data, original_len))
                    .is_err()
                {
                    break;
                }
                block_index += 1;
            }
            Ok(())
        });

        // Writer: receive, reorder, write
        let mut total_decompressed = 0u64;
        let mut expected = 0usize;
        let mut reorder: BTreeMap<usize, Vec<u8>> = BTreeMap::new();

        for (idx, result) in output_rx {
            let decompressed = result?;
            reorder.insert(idx, decompressed);

            while let Some(data) = reorder.remove(&expected) {
                output.write_all(&data)?;
                total_decompressed += data.len() as u64;
                expected += 1;
            }
        }

        // Check reader for errors
        match reader_handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(StreamError::Pz(PzError::InvalidInput)),
        }

        // Validate total length if declared
        if declared_orig_len > 0 && total_decompressed != declared_orig_len as u64 {
            return Err(StreamError::Pz(PzError::InvalidInput));
        }

        output.flush()?;
        Ok(total_decompressed)
    });

    total
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read exactly `buf.len()` bytes from the reader, returning an error on EOF.
fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> StreamResult<()> {
    reader.read_exact(buf).map_err(|e| {
        if e.kind() == io::ErrorKind::UnexpectedEof {
            StreamError::Pz(PzError::InvalidInput)
        } else {
            StreamError::Io(e)
        }
    })
}

/// Read up to `block_size` bytes from a reader. Returns an empty vec at EOF.
fn read_block<R: Read>(reader: &mut R, block_size: usize) -> StreamResult<Vec<u8>> {
    let mut buf = vec![0u8; block_size];
    let mut filled = 0;

    while filled < block_size {
        match reader.read(&mut buf[filled..]) {
            Ok(0) => break, // EOF
            Ok(n) => filled += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(StreamError::Io(e)),
        }
    }

    buf.truncate(filled);
    Ok(buf)
}

/// Write a block frame: `[compressed_len: u32][original_len: u32][data]`.
/// Returns the number of bytes written.
fn write_block_frame<W: Write>(
    writer: &mut W,
    compressed: &[u8],
    original_len: usize,
) -> StreamResult<u64> {
    writer.write_all(&(compressed.len() as u32).to_le_bytes())?;
    writer.write_all(&(original_len as u32).to_le_bytes())?;
    writer.write_all(compressed)?;
    Ok(BLOCK_HEADER_SIZE as u64 + compressed.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper: compress via streaming, return the compressed bytes.
    fn stream_compress(data: &[u8], pipeline: Pipeline, threads: usize) -> Vec<u8> {
        let options = CompressOptions {
            threads,
            block_size: 512, // small blocks for testing
            ..CompressOptions::default()
        };
        let input = Cursor::new(data);
        let mut output = Vec::new();
        compress_stream(input, &mut output, pipeline, &options).unwrap();
        output
    }

    /// Helper: decompress via streaming, return the decompressed bytes.
    fn stream_decompress(data: &[u8], threads: usize) -> Vec<u8> {
        let options = DecompressOptions {
            threads,
            ..DecompressOptions::default()
        };
        let input = Cursor::new(data);
        let mut output = Vec::new();
        decompress_stream(input, &mut output, &options).unwrap();
        output
    }

    // --- Round-trip tests ---

    #[test]
    fn test_stream_round_trip_single_block() {
        let data = b"hello, world!".repeat(10);
        for pipeline in [
            Pipeline::Deflate,
            Pipeline::Bw,
            Pipeline::Lzr,
            Pipeline::Lzf,
        ] {
            let compressed = stream_compress(&data, pipeline, 1);
            let decompressed = stream_decompress(&compressed, 1);
            assert_eq!(decompressed, data, "round-trip failed for {:?}", pipeline);
        }
    }

    #[test]
    fn test_stream_round_trip_multi_block() {
        // 2KB input with 512-byte blocks = 4 blocks
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);
        for pipeline in [
            Pipeline::Deflate,
            Pipeline::Bw,
            Pipeline::Lzr,
            Pipeline::Lzf,
        ] {
            let compressed = stream_compress(&data, pipeline, 1);
            let decompressed = stream_decompress(&compressed, 1);
            assert_eq!(decompressed, data, "round-trip failed for {:?}", pipeline);
        }
    }

    #[test]
    fn test_stream_round_trip_multi_threaded() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);
        for pipeline in [
            Pipeline::Deflate,
            Pipeline::Bw,
            Pipeline::Lzr,
            Pipeline::Lzf,
        ] {
            let compressed = stream_compress(&data, pipeline, 4);
            let decompressed = stream_decompress(&compressed, 4);
            assert_eq!(
                decompressed, data,
                "mt round-trip failed for {:?}",
                pipeline
            );
        }
    }

    #[test]
    fn test_stream_round_trip_exact_block_boundary() {
        // Input is exactly 2 blocks (1024 bytes with 512-byte block size)
        let data = vec![42u8; 1024];
        let compressed = stream_compress(&data, Pipeline::Lzf, 1);
        let decompressed = stream_decompress(&compressed, 1);
        assert_eq!(decompressed, data);
    }

    // --- Format compatibility tests ---

    #[test]
    fn test_framed_decompressed_by_in_memory() {
        // compress_stream -> pipeline::decompress (in-memory)
        let data = b"hello, world!".repeat(30);
        let compressed = stream_compress(&data, Pipeline::Deflate, 1);
        let decompressed = crate::pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_table_mode_decompressed_by_stream() {
        // pipeline::compress (table-mode V2) -> decompress_stream
        let data = b"hello, world!".repeat(30);
        let compressed = crate::pipeline::compress(&data, Pipeline::Deflate).unwrap();
        let decompressed = stream_decompress(&compressed, 1);
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_framed_mt_decompressed_by_in_memory() {
        // Multi-threaded stream compress -> in-memory decompress
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(50);
        let compressed = stream_compress(&data, Pipeline::Lzf, 4);
        let decompressed = crate::pipeline::decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    // --- Edge cases ---

    #[test]
    fn test_stream_empty_input() {
        let options = CompressOptions {
            threads: 1,
            ..CompressOptions::default()
        };
        let input = Cursor::new(b"" as &[u8]);
        let mut output = Vec::new();
        let bytes = compress_stream(input, &mut output, Pipeline::Lzf, &options).unwrap();
        assert!(bytes > 0); // header + sentinel + EOS

        // Decompress: should produce empty output
        let decompressed = stream_decompress(&output, 1);
        assert!(decompressed.is_empty());
    }

    #[test]
    fn test_stream_single_byte() {
        let data = vec![0xAB];
        for pipeline in [
            Pipeline::Deflate,
            Pipeline::Bw,
            Pipeline::Lzr,
            Pipeline::Lzf,
        ] {
            let compressed = stream_compress(&data, pipeline, 1);
            let decompressed = stream_decompress(&compressed, 1);
            assert_eq!(decompressed, data, "single-byte failed for {:?}", pipeline);
        }
    }

    #[test]
    fn test_stream_truncated_input() {
        let data = b"hello, world!".repeat(30);
        let compressed = stream_compress(&data, Pipeline::Lzf, 1);
        // Truncate to just the header
        let truncated = &compressed[..12];
        let input = Cursor::new(truncated);
        let mut output = Vec::new();
        let result = decompress_stream(input, &mut output, &DecompressOptions::default());
        assert!(result.is_err());
    }

    // --- Framed format structure test ---

    #[test]
    fn test_framed_format_structure() {
        let data = vec![42u8; 100];
        let compressed = stream_compress(&data, Pipeline::Lzf, 1);

        // Verify header
        assert_eq!(&compressed[0..2], &MAGIC);
        assert_eq!(compressed[2], VERSION);
        assert_eq!(compressed[3], Pipeline::Lzf as u8);
        // orig_len = 0 (streaming mode)
        assert_eq!(u32::from_le_bytes(compressed[4..8].try_into().unwrap()), 0);
        // num_blocks = FRAMED_SENTINEL
        assert_eq!(
            u32::from_le_bytes(compressed[8..12].try_into().unwrap()),
            FRAMED_SENTINEL
        );
        // First block frame starts at offset 12
        let compressed_len = u32::from_le_bytes(compressed[12..16].try_into().unwrap()) as usize;
        assert!(compressed_len > 0);
        let original_len = u32::from_le_bytes(compressed[16..20].try_into().unwrap()) as usize;
        assert_eq!(original_len, 100);
        // EOS sentinel at end
        let eos_offset = compressed.len() - 4;
        assert_eq!(
            u32::from_le_bytes(compressed[eos_offset..].try_into().unwrap()),
            0
        );
    }

    // --- Multi-threaded specific tests ---

    #[test]
    fn test_stream_many_threads_few_blocks() {
        // More threads (8) than blocks (2 with 512-byte blocks and 1000 bytes)
        let data = vec![77u8; 1000];
        let compressed = stream_compress(&data, Pipeline::Lzf, 8);
        let decompressed = stream_decompress(&compressed, 8);
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_stream_mt_ordering_preserved() {
        // Verify blocks are reassembled in correct order with many blocks
        // Each block has a distinct pattern so misordering is detectable
        let mut data = Vec::with_capacity(4096);
        for i in 0u8..8 {
            data.extend_from_slice(&[i; 512]);
        }
        let compressed = stream_compress(&data, Pipeline::Lzf, 4);
        let decompressed = stream_decompress(&compressed, 4);
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_stream_mt_decompress_framed() {
        // Multi-threaded decompress of framed data
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(100);
        let compressed = stream_compress(&data, Pipeline::Deflate, 4);
        let decompressed = stream_decompress(&compressed, 4);
        assert_eq!(decompressed, data);
    }
}
