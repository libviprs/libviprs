//! A small tile header reader for the issue #1330 tests. It lives here and not in the crate on
//! purpose: PVec's sniffers are the reader that matters, and this one only lets a test say what
//! a tile's header claims without trusting the encoder that wrote it.
#![allow(dead_code)]

/// What a PNG says in its framing.
#[derive(Debug, PartialEq, Eq)]
pub struct Png {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub colour_type: u8,
    pub interlace: u8,
    /// Chunk types in file order.
    pub chunks: Vec<[u8; 4]>,
    /// True when the last chunk is `IEND` and it ends at the last byte.
    pub ends_at_iend: bool,
}

impl Png {
    pub fn channels(&self) -> u8 {
        match self.colour_type {
            0 => 1,
            2 => 3,
            4 => 2,
            6 => 4,
            other => panic!("colour type {other} has no fixed channel count"),
        }
    }
    pub fn has(&self, kind: &[u8; 4]) -> bool {
        self.chunks.iter().any(|c| c == kind)
    }
}

pub fn png(bytes: &[u8]) -> Png {
    assert_eq!(&bytes[..8], &[137, 80, 78, 71, 13, 10, 26, 10], "signature");
    let mut at = 8;
    let mut chunks = Vec::new();
    let mut ihdr = None;
    let mut ends_at_iend = false;
    while at + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let kind: [u8; 4] = bytes[at + 4..at + 8].try_into().unwrap();
        if &kind == b"IHDR" {
            let d = &bytes[at + 8..at + 8 + 13];
            ihdr = Some((
                u32::from_be_bytes(d[0..4].try_into().unwrap()),
                u32::from_be_bytes(d[4..8].try_into().unwrap()),
                d[8],
                d[9],
                d[12],
            ));
        }
        chunks.push(kind);
        at += 12 + len;
        if &kind == b"IEND" {
            ends_at_iend = at == bytes.len();
            break;
        }
    }
    let (width, height, bit_depth, colour_type, interlace) = ihdr.expect("an IHDR");
    Png {
        width,
        height,
        bit_depth,
        colour_type,
        interlace,
        chunks,
        ends_at_iend,
    }
}

/// What a JPEG says up to its frame header.
#[derive(Debug, PartialEq, Eq)]
pub struct Jpeg {
    /// Byte offset of the `FF Cx` marker of the frame header.
    pub sof_offset: usize,
    /// 0xC0 baseline, 0xC2 progressive.
    pub sof_marker: u8,
    pub precision: u8,
    pub height: u32,
    pub width: u32,
    pub components: u8,
    pub ends_in_eoi: bool,
}

pub fn jpeg(bytes: &[u8]) -> Jpeg {
    assert_eq!(&bytes[..2], &[0xFF, 0xD8], "SOI");
    let mut at = 2;
    loop {
        assert_eq!(bytes[at], 0xFF, "a marker at {at}");
        let marker = bytes[at + 1];
        let len = u16::from_be_bytes([bytes[at + 2], bytes[at + 3]]) as usize;
        if matches!(marker, 0xC0..=0xC2) {
            let d = &bytes[at + 4..];
            return Jpeg {
                sof_offset: at,
                sof_marker: marker,
                precision: d[0],
                height: u32::from(u16::from_be_bytes([d[1], d[2]])),
                width: u32::from(u16::from_be_bytes([d[3], d[4]])),
                components: d[5],
                ends_in_eoi: bytes.ends_with(&[0xFF, 0xD9]),
            };
        }
        at += 2 + len;
    }
}

/// What a lossless WebP says in its first chunk.
#[derive(Debug, PartialEq, Eq)]
pub struct Webp {
    pub riff_size_matches: bool,
    pub first_chunk: [u8; 4],
    pub width: u32,
    pub height: u32,
}

pub fn webp(bytes: &[u8]) -> Webp {
    assert_eq!(&bytes[..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WEBP");
    let size = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let first_chunk: [u8; 4] = bytes[12..16].try_into().unwrap();
    assert_eq!(&first_chunk, b"VP8L", "the lossless chunk");
    assert_eq!(bytes[20], 0x2F, "VP8L signature");
    let bits = u32::from_le_bytes(bytes[21..25].try_into().unwrap());
    Webp {
        riff_size_matches: size + 8 == bytes.len(),
        first_chunk,
        width: (bits & 0x3FFF) + 1,
        height: ((bits >> 14) & 0x3FFF) + 1,
    }
}
