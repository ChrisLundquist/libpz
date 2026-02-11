pub mod analysis;
pub mod bwt;
pub mod crc32;
pub mod deflate;
pub mod frequency;
pub mod fse;
pub mod gzip;
pub mod huffman;
pub mod lz77;
pub mod mtf;
pub mod optimal;
pub mod pipeline;
pub mod pqueue;
pub mod rangecoder;
pub mod rans;
pub mod rle;
pub mod simd;

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "webgpu")]
pub mod webgpu;

pub mod ffi;

#[cfg(test)]
mod validation;

/// Error types for libpz operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PzError {
    /// Output buffer is too small to hold the result.
    BufferTooSmall,
    /// Input data is invalid or corrupt.
    InvalidInput,
    /// The requested operation is not supported.
    Unsupported,
}

impl std::fmt::Display for PzError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BufferTooSmall => write!(f, "output buffer too small"),
            Self::InvalidInput => write!(f, "invalid input"),
            Self::Unsupported => write!(f, "unsupported operation"),
        }
    }
}

impl std::error::Error for PzError {}

pub type PzResult<T> = Result<T, PzError>;
