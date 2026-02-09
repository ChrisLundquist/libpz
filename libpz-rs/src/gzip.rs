/// Gzip format parsing and decompression (RFC 1952).
///
/// Parses gzip headers including all optional fields (FEXTRA, FNAME,
/// FCOMMENT, FHCRC), decompresses the DEFLATE payload, and verifies
/// the CRC32 and original size in the trailer.

use crate::crc32;
use crate::deflate;
use crate::{PzError, PzResult};

/// Gzip magic bytes.
const GZIP_ID1: u8 = 0x1F;
const GZIP_ID2: u8 = 0x8B;

/// Compression method: deflate.
const CM_DEFLATE: u8 = 8;

/// Flag bits in the FLG byte.
const FTEXT: u8 = 1 << 0;
const FHCRC: u8 = 1 << 1;
const FEXTRA: u8 = 1 << 2;
const FNAME: u8 = 1 << 3;
const FCOMMENT: u8 = 1 << 4;

/// Operating system identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Os {
    Fat = 0,
    Amiga = 1,
    Vms = 2,
    Unix = 3,
    VmCms = 4,
    AtariTos = 5,
    Hpfs = 6,
    Macintosh = 7,
    ZSystem = 8,
    CpM = 9,
    Tops20 = 10,
    Ntfs = 11,
    Qdos = 12,
    AcornRiscos = 13,
    Unknown = 255,
}

impl Os {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Os::Fat,
            1 => Os::Amiga,
            2 => Os::Vms,
            3 => Os::Unix,
            4 => Os::VmCms,
            5 => Os::AtariTos,
            6 => Os::Hpfs,
            7 => Os::Macintosh,
            8 => Os::ZSystem,
            9 => Os::CpM,
            10 => Os::Tops20,
            11 => Os::Ntfs,
            12 => Os::Qdos,
            13 => Os::AcornRiscos,
            _ => Os::Unknown,
        }
    }
}

impl std::fmt::Display for Os {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Os::Fat => write!(f, "FAT"),
            Os::Amiga => write!(f, "Amiga"),
            Os::Vms => write!(f, "VMS"),
            Os::Unix => write!(f, "Unix"),
            Os::VmCms => write!(f, "VM/CMS"),
            Os::AtariTos => write!(f, "Atari TOS"),
            Os::Hpfs => write!(f, "HPFS"),
            Os::Macintosh => write!(f, "Macintosh"),
            Os::ZSystem => write!(f, "Z-System"),
            Os::CpM => write!(f, "CP/M"),
            Os::Tops20 => write!(f, "TOPS-20"),
            Os::Ntfs => write!(f, "NTFS"),
            Os::Qdos => write!(f, "QDOS"),
            Os::AcornRiscos => write!(f, "Acorn RISCOS"),
            Os::Unknown => write!(f, "unknown"),
        }
    }
}

/// Parsed gzip header.
#[derive(Debug, Clone)]
pub struct GzipHeader {
    /// Compression method (always 8 = deflate for gzip).
    pub method: u8,
    /// Whether the file is probably ASCII text (FTEXT flag).
    pub is_text: bool,
    /// Modification time as Unix timestamp (0 = not set).
    pub mtime: u32,
    /// Extra flags (XFL): 2 = max compression, 4 = fastest.
    pub extra_flags: u8,
    /// Operating system that created the file.
    pub os: Os,
    /// Optional extra field data.
    pub extra: Option<Vec<u8>>,
    /// Optional original filename (Latin-1, zero-terminated in file).
    pub filename: Option<String>,
    /// Optional comment (Latin-1, zero-terminated in file).
    pub comment: Option<String>,
    /// Optional header CRC16 (lower 16 bits of CRC32 of header).
    pub header_crc: Option<u16>,
}

/// Parsed gzip trailer.
#[derive(Debug, Clone)]
pub struct GzipTrailer {
    /// CRC32 of the uncompressed data.
    pub crc32: u32,
    /// Size of the original uncompressed data (mod 2^32).
    pub isize: u32,
}

/// Result of fully parsing a gzip member.
#[derive(Debug)]
pub struct GzipMember {
    pub header: GzipHeader,
    pub trailer: GzipTrailer,
    /// Byte offset where the DEFLATE compressed data starts.
    pub data_offset: usize,
    /// Byte offset where the DEFLATE compressed data ends (start of trailer).
    pub data_end: usize,
}

/// Check if data starts with the gzip magic bytes.
pub fn is_gzip(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == GZIP_ID1 && data[1] == GZIP_ID2
}

fn read_u16_le(data: &[u8], offset: usize) -> PzResult<u16> {
    if offset + 2 > data.len() {
        return Err(PzError::InvalidInput);
    }
    Ok(u16::from_le_bytes([data[offset], data[offset + 1]]))
}

fn read_u32_le(data: &[u8], offset: usize) -> PzResult<u32> {
    if offset + 4 > data.len() {
        return Err(PzError::InvalidInput);
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Find the next zero byte starting at `offset`.
fn find_zero(data: &[u8], offset: usize) -> PzResult<usize> {
    for i in offset..data.len() {
        if data[i] == 0 {
            return Ok(i);
        }
    }
    Err(PzError::InvalidInput)
}

/// Parse gzip header fields from `data`.
/// Returns the header and the byte offset where compressed data begins.
pub fn parse_header(data: &[u8]) -> PzResult<(GzipHeader, usize)> {
    // Minimum gzip member: 10 header + 0 data + 8 trailer = 18 bytes
    if data.len() < 18 {
        return Err(PzError::InvalidInput);
    }

    if data[0] != GZIP_ID1 || data[1] != GZIP_ID2 {
        return Err(PzError::InvalidInput);
    }

    let method = data[2];
    if method != CM_DEFLATE {
        return Err(PzError::Unsupported);
    }

    let flg = data[3];
    let mtime = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    let extra_flags = data[8];
    let os = Os::from_u8(data[9]);

    let is_text = flg & FTEXT != 0;

    let mut pos = 10;

    // FEXTRA: extra field
    let extra = if flg & FEXTRA != 0 {
        let xlen = read_u16_le(data, pos)? as usize;
        pos += 2;
        if pos + xlen > data.len() {
            return Err(PzError::InvalidInput);
        }
        let extra_data = data[pos..pos + xlen].to_vec();
        pos += xlen;
        Some(extra_data)
    } else {
        None
    };

    // FNAME: original filename, zero-terminated Latin-1
    let filename = if flg & FNAME != 0 {
        let end = find_zero(data, pos)?;
        let name = String::from_utf8_lossy(&data[pos..end]).into_owned();
        pos = end + 1;
        Some(name)
    } else {
        None
    };

    // FCOMMENT: file comment, zero-terminated Latin-1
    let comment = if flg & FCOMMENT != 0 {
        let end = find_zero(data, pos)?;
        let cmt = String::from_utf8_lossy(&data[pos..end]).into_owned();
        pos = end + 1;
        Some(cmt)
    } else {
        None
    };

    // FHCRC: header CRC16
    let header_crc = if flg & FHCRC != 0 {
        let crc = read_u16_le(data, pos)?;
        pos += 2;
        Some(crc)
    } else {
        None
    };

    let header = GzipHeader {
        method,
        is_text,
        mtime,
        extra_flags,
        os,
        extra,
        filename,
        comment,
        header_crc,
    };

    Ok((header, pos))
}

/// Parse the 8-byte gzip trailer at a given offset.
pub fn parse_trailer(data: &[u8], offset: usize) -> PzResult<GzipTrailer> {
    let crc = read_u32_le(data, offset)?;
    let isize = read_u32_le(data, offset + 4)?;
    Ok(GzipTrailer { crc32: crc, isize })
}

/// Decompress a complete gzip stream.
///
/// Parses the header, inflates the DEFLATE payload, verifies CRC32
/// and original size, and returns the decompressed data along with
/// the parsed header.
pub fn decompress(data: &[u8]) -> PzResult<(Vec<u8>, GzipHeader)> {
    let (header, data_offset) = parse_header(data)?;

    // The compressed data runs from data_offset until 8 bytes before the end.
    // However, we don't know exactly where the DEFLATE stream ends.
    // We pass the remaining data to inflate and it stops when it sees
    // the final block. The trailer is the last 8 bytes.
    if data.len() < data_offset + 8 {
        return Err(PzError::InvalidInput);
    }

    // Inflate the DEFLATE stream. We pass everything from data_offset onward
    // and let the inflater stop at the end of the DEFLATE data.
    let compressed = &data[data_offset..data.len() - 8];
    let decompressed = deflate::inflate(compressed)?;

    // Parse and verify trailer
    let trailer_offset = data.len() - 8;
    let trailer = parse_trailer(data, trailer_offset)?;

    // Verify CRC32
    let computed_crc = crc32::crc32(&decompressed);
    if computed_crc != trailer.crc32 {
        return Err(PzError::InvalidInput);
    }

    // Verify size (mod 2^32)
    let computed_size = decompressed.len() as u32;
    if computed_size != trailer.isize {
        return Err(PzError::InvalidInput);
    }

    Ok((decompressed, header))
}

/// Display summary information about a gzip file.
pub fn info(data: &[u8]) -> PzResult<GzipHeader> {
    let (header, _) = parse_header(data)?;
    Ok(header)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_gzip() {
        assert!(is_gzip(&[0x1F, 0x8B, 0x08]));
        assert!(!is_gzip(&[0x1F, 0x8C]));
        assert!(!is_gzip(&[b'P', b'Z']));
        assert!(!is_gzip(&[0x1F]));
    }

    #[test]
    fn test_parse_minimal_header() {
        // Minimal 10-byte gzip header + 8 bytes trailer (empty DEFLATE)
        let mut data = vec![
            0x1F, 0x8B, // ID1, ID2
            0x08,       // CM = deflate
            0x00,       // FLG = no flags
            0x00, 0x00, 0x00, 0x00, // MTIME = 0
            0x00,       // XFL
            0x03,       // OS = Unix
        ];
        // Pad with enough for trailer
        data.extend_from_slice(&[0u8; 8]);

        let (header, offset) = parse_header(&data).unwrap();
        assert_eq!(header.method, 8);
        assert!(!header.is_text);
        assert_eq!(header.mtime, 0);
        assert_eq!(header.os, Os::Unix);
        assert!(header.filename.is_none());
        assert!(header.comment.is_none());
        assert!(header.extra.is_none());
        assert!(header.header_crc.is_none());
        assert_eq!(offset, 10);
    }

    #[test]
    fn test_parse_header_with_filename() {
        let mut data = vec![
            0x1F, 0x8B, // ID
            0x08,       // CM
            0x08,       // FLG = FNAME
            0x00, 0x00, 0x00, 0x00, // MTIME
            0x00,       // XFL
            0x03,       // OS = Unix
        ];
        // Filename: "test.txt\0"
        data.extend_from_slice(b"test.txt\0");
        // Trailer padding
        data.extend_from_slice(&[0u8; 8]);

        let (header, offset) = parse_header(&data).unwrap();
        assert_eq!(header.filename, Some("test.txt".to_string()));
        assert_eq!(offset, 19); // 10 + 9 ("test.txt" + NUL)
    }

    #[test]
    fn test_parse_header_with_extra() {
        let mut data = vec![
            0x1F, 0x8B, // ID
            0x08,       // CM
            0x04,       // FLG = FEXTRA
            0x00, 0x00, 0x00, 0x00,
            0x00, 0x03,
        ];
        // XLEN = 4
        data.push(0x04);
        data.push(0x00);
        // Extra data: 4 bytes
        data.extend_from_slice(&[0xAA, 0xBB, 0xCC, 0xDD]);
        data.extend_from_slice(&[0u8; 8]);

        let (header, offset) = parse_header(&data).unwrap();
        assert_eq!(header.extra, Some(vec![0xAA, 0xBB, 0xCC, 0xDD]));
        assert_eq!(offset, 16); // 10 + 2 (xlen) + 4 (extra data)
    }

    #[test]
    fn test_os_display() {
        assert_eq!(format!("{}", Os::Unix), "Unix");
        assert_eq!(format!("{}", Os::Ntfs), "NTFS");
        assert_eq!(format!("{}", Os::Unknown), "unknown");
    }
}
