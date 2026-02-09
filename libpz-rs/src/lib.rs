pub mod frequency;
pub mod huffman;
pub mod lz77;
pub mod pqueue;

pub mod ffi;

/// Error types for libpz operations.
#[derive(Debug, Clone, PartialEq, Eq)]
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
            PzError::BufferTooSmall => write!(f, "output buffer too small"),
            PzError::InvalidInput => write!(f, "invalid input"),
            PzError::Unsupported => write!(f, "unsupported operation"),
        }
    }
}

impl std::error::Error for PzError {}

pub type PzResult<T> = Result<T, PzError>;
