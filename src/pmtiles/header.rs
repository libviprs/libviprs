//! The 127-byte PMTiles v3 header, and the two enums it carries.
//!
//! The header is always the first 127 bytes of the archive and it is never
//! compressed. Everything else in the file is reached through it: four
//! absolute `(offset, length)` pairs pointing at the root directory, the JSON
//! metadata, the leaf directory section and the tile data section. Nothing
//! else in the format is positional, so a reader that has the header can
//! address the whole archive with ranged reads and never has to scan.
//!
//! # Every length here is a stored length
//!
//! Root Directory Length, Metadata Length, Leaf Directories Length and Tile
//! Data Length are all sizes of the **compressed** bytes as stored. There is
//! no uncompressed size anywhere in PMTiles v3, which is why
//! [`Compression::decompress`] makes the caller name a ceiling: a reader
//! cannot pre-size the output buffer and must cap it instead.
//!
//! # What this decoder refuses, and what it does not
//!
//! [`Header::try_decode`] refuses exactly three things: a buffer shorter than
//! 127 bytes, magic that is not `PMTiles`, and a version byte that is not 3.
//! Those three make the rest of the bytes meaningless (PMTiles v2 shares the
//! magic and lays its fields out completely differently, so ignoring the
//! version byte yields plausible garbage rather than an error).
//!
//! It deliberately refuses nothing else, and the reasoning is worth stating
//! because "validate everything" is the instinct:
//!
//! * **An unrecognised tile type or compression byte is kept, not rejected.**
//!   Both fields are descriptive. The spec has already grown two tile types
//!   since v3 shipped, and a reader that refuses to open an archive because it
//!   does not recognise byte 99 has turned a future spec revision into a
//!   corrupt file. [`TileType::Other`] and [`Compression::Other`] carry the
//!   byte through so the value can be reported honestly, and the refusal
//!   happens where it is load-bearing: at [`Compression::decompress`], which
//!   genuinely cannot proceed.
//! * **`min_zoom > max_zoom` is accepted.** The spec says max zoom "must be
//!   greater than or equal to the min zoom" in lowercase, in a document that
//!   uppercases its RFC 2119 keywords everywhere else, and the pair is an
//!   optimisation hint rather than something a lookup depends on.
//! * **Bounds outside the legal lon/lat ranges are accepted.** Rounding into
//!   the E7 fixed point is unspecified, real archives are sloppy at the edges,
//!   and refusing to open an archive over a bounding box nothing reads would
//!   lose every tile in it for no safety gain.
//!
//! The checks that actually protect a reader are bounds checks of offsets and
//! lengths against the size of the archive, and those need the archive size,
//! which the header does not carry. They belong to the reader.

use std::fmt;
use std::io::{Read, Write};

use crate::pmtiles::PmTilesError;
use crate::sink::TileFormat;

/// The fixed size of a v3 header, in bytes.
pub const HEADER_BYTES: usize = 127;

/// The seven magic bytes every PMTiles archive starts with, v2 included.
pub const MAGIC: &[u8; 7] = b"PMTiles";

/// The specification version this module implements.
pub const SPEC_VERSION: u8 = 3;

// Field offsets, as a table so the encoder and decoder cannot drift apart.
// The unit tests deliberately do **not** use these constants: they read the
// encoded bytes at integer literals taken from the spec's byte grid, because a
// test that shares a wrong constant with the code proves nothing.
const OFF_MAGIC: usize = 0;
const OFF_VERSION: usize = 7;
const OFF_ROOT_OFFSET: usize = 8;
const OFF_ROOT_LENGTH: usize = 16;
const OFF_METADATA_OFFSET: usize = 24;
const OFF_METADATA_LENGTH: usize = 32;
const OFF_LEAF_OFFSET: usize = 40;
const OFF_LEAF_LENGTH: usize = 48;
const OFF_TILE_DATA_OFFSET: usize = 56;
const OFF_TILE_DATA_LENGTH: usize = 64;
const OFF_ADDRESSED_TILES: usize = 72;
const OFF_TILE_ENTRIES: usize = 80;
const OFF_TILE_CONTENTS: usize = 88;
const OFF_CLUSTERED: usize = 96;
const OFF_INTERNAL_COMPRESSION: usize = 97;
const OFF_TILE_COMPRESSION: usize = 98;
const OFF_TILE_TYPE: usize = 99;
const OFF_MIN_ZOOM: usize = 100;
const OFF_MAX_ZOOM: usize = 101;
const OFF_MIN_LON: usize = 102;
const OFF_MIN_LAT: usize = 106;
const OFF_MAX_LON: usize = 110;
const OFF_MAX_LAT: usize = 114;
const OFF_CENTER_ZOOM: usize = 118;
const OFF_CENTER_LON: usize = 119;
const OFF_CENTER_LAT: usize = 123;

// ---------------------------------------------------------------------------
// TileType
// ---------------------------------------------------------------------------

/// What the tile blobs in the data section are.
///
/// The byte values are the spec's, and the enum carries all seven of them plus
/// [`TileType::Other`] for a value from a newer revision than this build knows.
/// Keeping the raw byte matters: `0x00` means "the writer did not know", while
/// an unrecognised `0x07` would mean "a newer spec knows something we do not",
/// and folding the second into the first throws away the only signal that this
/// reader is out of date.
///
/// # Round-tripping
///
/// [`TileType::from_byte`] is canonical: it never answers `Other` for a value
/// that has a named variant, so `from_byte(t.to_byte()) == t` holds for every
/// value this module ever produces. A hand-written `TileType::Other(2)` is a
/// non-canonical spelling of [`TileType::Png`]; it encodes to the same byte and
/// decodes back as `Png`. Anything generating headers for a property test
/// should go through `from_byte` rather than constructing `Other` directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TileType {
    /// `0x00`. Unknown or other; the writer did not say.
    Unknown,
    /// `0x01`. Mapbox Vector Tile. A `vector_layers` key in the JSON metadata
    /// is a `MUST` for this type and only this type.
    Mvt,
    /// `0x02`. PNG.
    Png,
    /// `0x03`. JPEG.
    Jpeg,
    /// `0x04`. WebP.
    Webp,
    /// `0x05`. AVIF, added in spec 3.1.
    Avif,
    /// `0x06`. MapLibre Vector Tile (`.mlt`), added in spec 3.5.
    Mlt,
    /// A value no revision this build knows about has defined. The byte is
    /// kept so it can be reported rather than silently flattened to
    /// [`TileType::Unknown`].
    Other(u8),
}

impl TileType {
    /// Read a tile type from its wire byte. Total: every `u8` maps to
    /// something, because an unrecognised tile type must not stop a read.
    pub fn from_byte(value: u8) -> Self {
        match value {
            0x00 => Self::Unknown,
            0x01 => Self::Mvt,
            0x02 => Self::Png,
            0x03 => Self::Jpeg,
            0x04 => Self::Webp,
            0x05 => Self::Avif,
            0x06 => Self::Mlt,
            other => Self::Other(other),
        }
    }

    /// The wire byte for this tile type.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Unknown => 0x00,
            Self::Mvt => 0x01,
            Self::Png => 0x02,
            Self::Jpeg => 0x03,
            Self::Webp => 0x04,
            Self::Avif => 0x05,
            Self::Mlt => 0x06,
            Self::Other(other) => other,
        }
    }

    /// The tile type that carries a libviprs [`TileFormat`].
    ///
    /// `Png`, `Jpeg` and `Webp` map straight across. [`TileFormat::Raw`] has
    /// no PMTiles type and never will: a PMTiles tile is a self-describing
    /// image blob that a viewer hands to a decoder, and raw pixel bytes carry
    /// neither their dimensions nor their pixel format.
    ///
    /// WebP was one-way until issue #1123. `TileType::Webp` has always read
    /// fine and `TileType::extension` has always answered `"webp"`, but
    /// `Raster::encode_webp` was not wired into `TileFormat`, so there was no
    /// `TileFormat` for this function to come *from*. Wiring the variant in is
    /// the whole of what closed it; nothing about the mapping itself changed.
    ///
    /// [`TileType::Avif`] and [`TileType::Mlt`] are where WebP was: readable,
    /// with no encoder behind them in this build.
    pub fn try_from_tile_format(format: TileFormat) -> Result<Self, PmTilesError> {
        match format {
            TileFormat::Png => Ok(Self::Png),
            TileFormat::Jpeg { .. } => Ok(Self::Jpeg),
            TileFormat::Webp => Ok(Self::Webp),
            TileFormat::Raw => Err(PmTilesError::UnsupportedTileFormat { format }),
        }
    }

    /// The file extension a tile of this type would have on disk, for the
    /// `extract` route that reconstructs a directory pyramid.
    ///
    /// `None` for the two types with nothing to extract to: `Unknown` has no
    /// format at all, and `Other` is a format this build does not know.
    pub fn extension(self) -> Option<&'static str> {
        match self {
            Self::Png => Some("png"),
            Self::Jpeg => Some("jpeg"),
            Self::Webp => Some("webp"),
            Self::Avif => Some("avif"),
            Self::Mvt => Some("mvt"),
            Self::Mlt => Some("mlt"),
            Self::Unknown | Self::Other(_) => None,
        }
    }
}

impl fmt::Display for TileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("unknown"),
            Self::Mvt => f.write_str("mvt"),
            Self::Png => f.write_str("png"),
            Self::Jpeg => f.write_str("jpeg"),
            Self::Webp => f.write_str("webp"),
            Self::Avif => f.write_str("avif"),
            Self::Mlt => f.write_str("mlt"),
            Self::Other(byte) => write!(f, "unrecognised tile type {byte}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Compression
// ---------------------------------------------------------------------------

/// How a stored blob is compressed.
///
/// The same enum fills two independent header fields with different scopes.
/// **Internal compression** covers the root directory, the JSON metadata and
/// every leaf directory. **Tile compression** covers the tile blobs and
/// nothing else. The spec places no restriction on the combination, and in
/// practice they differ: a raster archive is normally gzip internally and
/// `None` for tiles, because PNG and WebP are already compressed and gzipping
/// them again costs time to make them slightly larger.
///
/// This build can perform `None` and `Gzip`. Brotli and Zstd are legal and
/// this crate carries neither codec, so [`Compression::decompress`] refuses
/// them by name rather than pretending. That refusal is survivable for tile
/// compression, where a reader handing bytes onward never has to decompress at
/// all, and fatal for internal compression, where the directories cannot be
/// read without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Compression {
    /// `0x00`. The writer did not record what it did. Fatal as an internal
    /// compression; survivable as a tile compression, where the bytes can
    /// still be handed on uninterpreted.
    Unknown,
    /// `0x01`. Stored uncompressed.
    None,
    /// `0x02`. gzip. What this crate writes, via `flate2`.
    Gzip,
    /// `0x03`. Brotli. Legal, and no codec in this build.
    Brotli,
    /// `0x04`. Zstd. Legal, and no codec in this build.
    Zstd,
    /// A value no revision this build knows about has defined, kept rather
    /// than flattened for the reason [`TileType::Other`] is.
    Other(u8),
}

impl Compression {
    /// Read a compression from its wire byte. Total, for the reason
    /// [`TileType::from_byte`] is.
    pub fn from_byte(value: u8) -> Self {
        match value {
            0x00 => Self::Unknown,
            0x01 => Self::None,
            0x02 => Self::Gzip,
            0x03 => Self::Brotli,
            0x04 => Self::Zstd,
            other => Self::Other(other),
        }
    }

    /// The wire byte for this compression.
    pub fn to_byte(self) -> u8 {
        match self {
            Self::Unknown => 0x00,
            Self::None => 0x01,
            Self::Gzip => 0x02,
            Self::Brotli => 0x03,
            Self::Zstd => 0x04,
            Self::Other(other) => other,
        }
    }

    /// Whether this build can compress and decompress in this scheme.
    pub fn is_supported(self) -> bool {
        matches!(self, Self::None | Self::Gzip)
    }

    /// Compress `bytes` for storage.
    ///
    /// [`Compression::None`] is the identity. Everything except `Gzip` is a
    /// typed refusal, `Unknown` included: a writer that does not know what it
    /// compressed with has produced an archive nothing can read.
    pub fn compress(self, bytes: &[u8]) -> Result<Vec<u8>, PmTilesError> {
        match self {
            Self::None => Ok(bytes.to_vec()),
            Self::Gzip => {
                let mut encoder =
                    flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
                encoder.write_all(bytes)?;
                Ok(encoder.finish()?)
            }
            other => Err(PmTilesError::UnsupportedCompression { compression: other }),
        }
    }

    /// Decompress `bytes`, refusing once the output passes `limit`.
    ///
    /// The ceiling is an argument rather than a constant because no length
    /// field in PMTiles v3 is an uncompressed length: a reader cannot pre-size
    /// this buffer from anything in the file, so the only defence against a
    /// few hundred bytes of stored root expanding without bound is an absolute
    /// cap chosen by whoever knows what is being read. A ratio would not do:
    /// the compressed size is attacker-controlled too.
    ///
    /// The cap is checked against the *output*, so the refusal happens after
    /// `limit` bytes have been produced rather than after the whole stream has
    /// been inflated. `limit` bytes of legitimate output are fine; the refusal
    /// is for `limit + 1`.
    pub fn decompress(self, bytes: &[u8], limit: usize) -> Result<Vec<u8>, PmTilesError> {
        match self {
            Self::None => {
                if bytes.len() > limit {
                    return Err(PmTilesError::DecompressionLimit { limit });
                }
                Ok(bytes.to_vec())
            }
            Self::Gzip => {
                let mut out = Vec::new();
                // `limit + 1` so a stream that produces exactly `limit` bytes
                // is accepted and one that produces more is caught: `take`
                // stops at its ceiling without signalling, so the only way to
                // tell "ended" from "truncated by the limit" is to leave room
                // for one byte past it.
                let ceiling = limit.saturating_add(1);
                let mut decoder = flate2::read::GzDecoder::new(bytes).take(ceiling as u64);
                decoder.read_to_end(&mut out)?;
                if out.len() > limit {
                    return Err(PmTilesError::DecompressionLimit { limit });
                }
                Ok(out)
            }
            other => Err(PmTilesError::UnsupportedCompression { compression: other }),
        }
    }
}

impl fmt::Display for Compression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown => f.write_str("unknown compression"),
            Self::None => f.write_str("no compression"),
            Self::Gzip => f.write_str("gzip"),
            Self::Brotli => f.write_str("brotli"),
            Self::Zstd => f.write_str("zstd"),
            Self::Other(byte) => write!(f, "unrecognised compression {byte}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// The 127-byte v3 header, decoded.
///
/// # Why this one is not `#[non_exhaustive]`
///
/// The crate's convention is `#[non_exhaustive]` on public structs so fields
/// can be added without a break. This struct is the exception, deliberately:
/// its shape is fixed by a wire format at exactly 127 bytes, and it cannot
/// gain a field without a v4 specification, which would be a new type rather
/// than a field on this one. Making it exhaustive is what lets a caller
/// outside this crate build one with a struct literal, which is what a writer
/// and a test both want, and there is no future field for the attribute to
/// protect them from.
///
/// # Positions are longitude first
///
/// Each of the three positions is a longitude then a latitude, each an `i32`
/// of degrees times 10,000,000. The spec's prose describes the fields as
/// "the minimum latitude and minimum longitude", latitude first, while its
/// byte layout puts longitude first, and only the layout is normative. The
/// symptom of getting it backwards is subtle enough to be worth the warning:
/// a world-covering box of (-180, -85) to (180, 85) becomes (-85, -180) to
/// (85, 180), which still looks like a bounding box.
///
/// # Examples
///
/// ```
/// use libviprs::pmtiles::{Compression, Header, TileType};
///
/// let header = Header {
///     tile_type: TileType::Png,
///     internal_compression: Compression::Gzip,
///     tile_compression: Compression::None,
///     min_zoom: 0,
///     max_zoom: 5,
///     ..Header::default()
/// };
///
/// let bytes = header.encode();
/// assert_eq!(bytes.len(), 127);
/// assert_eq!(Header::try_decode(&bytes).unwrap(), header);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Absolute offset of the root directory, from byte 0 of the archive.
    pub root_offset: u64,
    /// Stored (compressed) length of the root directory. The spec caps
    /// `root_offset + root_length` at 16384 so a latency-sensitive client can
    /// fetch the header and the whole root in one request.
    pub root_length: u64,
    /// Absolute offset of the JSON metadata.
    pub metadata_offset: u64,
    /// Stored (compressed) length of the JSON metadata.
    pub metadata_length: u64,
    /// Absolute offset of the leaf directory section. Each leaf inside it is
    /// compressed individually, never as one blob.
    pub leaf_directories_offset: u64,
    /// Summed stored length of every leaf directory. Zero means the archive
    /// has no leaves and the root addresses every tile.
    pub leaf_directories_length: u64,
    /// Absolute offset of the tile data section. Every tile entry's offset,
    /// wherever the entry was found, is relative to this.
    pub tile_data_offset: u64,
    /// Summed stored length of every tile blob.
    pub tile_data_length: u64,
    /// Number of tiles addressed by the archive with runs expanded, or 0 for
    /// "unknown". Equal to the sum of `run_length` over every entry.
    pub addressed_tiles_count: u64,
    /// Number of directory entries with a run length above 0, or 0 for
    /// "unknown". Leaf pointers are excluded by the definition.
    pub tile_entries_count: u64,
    /// Number of distinct blobs in the tile data section, or 0 for "unknown".
    /// Below `tile_entries_count` whenever dedupe found anything.
    pub tile_contents_count: u64,
    /// Whether the tile data is stored in TileID order. Only ever narrows what
    /// a reader may assume, so a writer that cannot promise it writes `false`.
    pub clustered: bool,
    /// Compression of the root directory, the metadata and every leaf.
    pub internal_compression: Compression,
    /// Compression of the tile blobs.
    pub tile_compression: Compression,
    /// What the tile blobs are.
    pub tile_type: TileType,
    /// Lowest zoom level with tiles.
    pub min_zoom: u8,
    /// Highest zoom level with tiles.
    pub max_zoom: u8,
    /// West edge, degrees times 10,000,000.
    pub min_lon_e7: i32,
    /// South edge, degrees times 10,000,000.
    pub min_lat_e7: i32,
    /// East edge, degrees times 10,000,000.
    pub max_lon_e7: i32,
    /// North edge, degrees times 10,000,000.
    pub max_lat_e7: i32,
    /// Zoom a viewer may open at. Purely advisory.
    pub center_zoom: u8,
    /// Centre longitude, degrees times 10,000,000. Purely advisory.
    pub center_lon_e7: i32,
    /// Centre latitude, degrees times 10,000,000. Purely advisory.
    pub center_lat_e7: i32,
}

impl Default for Header {
    /// A header with every offset and count zeroed, gzip internal compression,
    /// uncompressed tiles and an unknown tile type.
    ///
    /// The two compressions are not zeroed, and that is the point of writing
    /// this by hand rather than deriving it: `0x00` in either compression
    /// field means "unknown", which is a legal value a writer should never
    /// emit, so the default is the pair this crate actually writes.
    fn default() -> Self {
        Self {
            root_offset: 0,
            root_length: 0,
            metadata_offset: 0,
            metadata_length: 0,
            leaf_directories_offset: 0,
            leaf_directories_length: 0,
            tile_data_offset: 0,
            tile_data_length: 0,
            addressed_tiles_count: 0,
            tile_entries_count: 0,
            tile_contents_count: 0,
            clustered: false,
            internal_compression: Compression::Gzip,
            tile_compression: Compression::None,
            tile_type: TileType::Unknown,
            min_zoom: 0,
            max_zoom: 0,
            min_lon_e7: 0,
            min_lat_e7: 0,
            max_lon_e7: 0,
            max_lat_e7: 0,
            center_zoom: 0,
            center_lon_e7: 0,
            center_lat_e7: 0,
        }
    }
}

impl Header {
    /// Serialise to the 127 bytes that go at the start of the archive.
    ///
    /// Infallible: every field is already the width the format gives it, so
    /// there is nothing left to reject at this point.
    pub fn encode(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        out[OFF_MAGIC..OFF_MAGIC + MAGIC.len()].copy_from_slice(MAGIC);
        out[OFF_VERSION] = SPEC_VERSION;

        put_u64(&mut out, OFF_ROOT_OFFSET, self.root_offset);
        put_u64(&mut out, OFF_ROOT_LENGTH, self.root_length);
        put_u64(&mut out, OFF_METADATA_OFFSET, self.metadata_offset);
        put_u64(&mut out, OFF_METADATA_LENGTH, self.metadata_length);
        put_u64(&mut out, OFF_LEAF_OFFSET, self.leaf_directories_offset);
        put_u64(&mut out, OFF_LEAF_LENGTH, self.leaf_directories_length);
        put_u64(&mut out, OFF_TILE_DATA_OFFSET, self.tile_data_offset);
        put_u64(&mut out, OFF_TILE_DATA_LENGTH, self.tile_data_length);
        put_u64(&mut out, OFF_ADDRESSED_TILES, self.addressed_tiles_count);
        put_u64(&mut out, OFF_TILE_ENTRIES, self.tile_entries_count);
        put_u64(&mut out, OFF_TILE_CONTENTS, self.tile_contents_count);

        out[OFF_CLUSTERED] = u8::from(self.clustered);
        out[OFF_INTERNAL_COMPRESSION] = self.internal_compression.to_byte();
        out[OFF_TILE_COMPRESSION] = self.tile_compression.to_byte();
        out[OFF_TILE_TYPE] = self.tile_type.to_byte();
        out[OFF_MIN_ZOOM] = self.min_zoom;
        out[OFF_MAX_ZOOM] = self.max_zoom;

        put_i32(&mut out, OFF_MIN_LON, self.min_lon_e7);
        put_i32(&mut out, OFF_MIN_LAT, self.min_lat_e7);
        put_i32(&mut out, OFF_MAX_LON, self.max_lon_e7);
        put_i32(&mut out, OFF_MAX_LAT, self.max_lat_e7);
        out[OFF_CENTER_ZOOM] = self.center_zoom;
        put_i32(&mut out, OFF_CENTER_LON, self.center_lon_e7);
        put_i32(&mut out, OFF_CENTER_LAT, self.center_lat_e7);

        out
    }

    /// Decode a header from the first 127 bytes of `bytes`.
    ///
    /// A longer slice is accepted and only the first 127 bytes are read, so a
    /// caller that fetched the customary first 16 KiB (header plus root
    /// directory, which the spec sizes to fit together) can hand the whole
    /// buffer over without subslicing it first.
    ///
    /// See the module docs for what this refuses and, more interestingly, what
    /// it does not.
    pub fn try_decode(bytes: &[u8]) -> Result<Self, PmTilesError> {
        if bytes.len() < HEADER_BYTES {
            return Err(PmTilesError::ShortHeader {
                got: bytes.len(),
                want: HEADER_BYTES,
            });
        }

        let mut magic = [0u8; 7];
        magic.copy_from_slice(&bytes[OFF_MAGIC..OFF_MAGIC + 7]);
        if &magic != MAGIC {
            return Err(PmTilesError::BadMagic { found: magic });
        }
        if bytes[OFF_VERSION] != SPEC_VERSION {
            return Err(PmTilesError::UnsupportedVersion {
                found: bytes[OFF_VERSION],
            });
        }

        Ok(Self {
            root_offset: read_u64(bytes, OFF_ROOT_OFFSET),
            root_length: read_u64(bytes, OFF_ROOT_LENGTH),
            metadata_offset: read_u64(bytes, OFF_METADATA_OFFSET),
            metadata_length: read_u64(bytes, OFF_METADATA_LENGTH),
            leaf_directories_offset: read_u64(bytes, OFF_LEAF_OFFSET),
            leaf_directories_length: read_u64(bytes, OFF_LEAF_LENGTH),
            tile_data_offset: read_u64(bytes, OFF_TILE_DATA_OFFSET),
            tile_data_length: read_u64(bytes, OFF_TILE_DATA_LENGTH),
            addressed_tiles_count: read_u64(bytes, OFF_ADDRESSED_TILES),
            tile_entries_count: read_u64(bytes, OFF_TILE_ENTRIES),
            tile_contents_count: read_u64(bytes, OFF_TILE_CONTENTS),
            // Any non-zero byte reads as clustered. The spec defines 0 and 1
            // and gives no "unknown", so there is no third state to preserve.
            clustered: bytes[OFF_CLUSTERED] != 0,
            internal_compression: Compression::from_byte(bytes[OFF_INTERNAL_COMPRESSION]),
            tile_compression: Compression::from_byte(bytes[OFF_TILE_COMPRESSION]),
            tile_type: TileType::from_byte(bytes[OFF_TILE_TYPE]),
            min_zoom: bytes[OFF_MIN_ZOOM],
            max_zoom: bytes[OFF_MAX_ZOOM],
            min_lon_e7: read_i32(bytes, OFF_MIN_LON),
            min_lat_e7: read_i32(bytes, OFF_MIN_LAT),
            max_lon_e7: read_i32(bytes, OFF_MAX_LON),
            max_lat_e7: read_i32(bytes, OFF_MAX_LAT),
            center_zoom: bytes[OFF_CENTER_ZOOM],
            center_lon_e7: read_i32(bytes, OFF_CENTER_LON),
            center_lat_e7: read_i32(bytes, OFF_CENTER_LAT),
        })
    }

    /// Whether the archive has leaf directories, which is a question about the
    /// **length** and never about the offset.
    ///
    /// Measured on the oracle goldens: an archive with no leaves still carries
    /// a non-zero `leaf_directories_offset`, equal to `tile_data_offset`,
    /// because the section is empty rather than absent and an empty section
    /// starts where the next one does. Anything deciding this from the offset
    /// answers "yes" for every archive there is.
    pub fn has_leaves(&self) -> bool {
        self.leaf_directories_length > 0
    }

    /// The bounding box as degrees: `(min_lon, min_lat, max_lon, max_lat)`.
    ///
    /// The stored form is an `i32` of degrees times 10,000,000, so this is
    /// exact for every value a conformant writer stores and the inverse of
    /// [`Header::set_bounds_degrees`] to within the 1.1 cm the fixed point can
    /// represent.
    pub fn bounds_degrees(&self) -> (f64, f64, f64, f64) {
        (
            f64::from(self.min_lon_e7) / 1e7,
            f64::from(self.min_lat_e7) / 1e7,
            f64::from(self.max_lon_e7) / 1e7,
            f64::from(self.max_lat_e7) / 1e7,
        )
    }

    /// Set the bounding box from degrees.
    ///
    /// The spec says "multiply by 10,000,000 and convert to an `i32`" without
    /// saying how to round, so two conformant writers can differ by one unit
    /// in the last place. This rounds half away from zero, which is what
    /// `f64::round` does, rather than truncating toward zero: truncation
    /// shrinks a bounding box, and a bounding box that is one unit too small
    /// excludes tiles at its edge.
    ///
    /// Values are clamped to the `i32` range rather than wrapping, so a
    /// nonsense latitude cannot become its own negation.
    pub fn set_bounds_degrees(&mut self, min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) {
        self.min_lon_e7 = degrees_to_e7(min_lon);
        self.min_lat_e7 = degrees_to_e7(min_lat);
        self.max_lon_e7 = degrees_to_e7(max_lon);
        self.max_lat_e7 = degrees_to_e7(max_lat);
    }

    /// The centre as degrees: `(lon, lat)`.
    pub fn center_degrees(&self) -> (f64, f64) {
        (
            f64::from(self.center_lon_e7) / 1e7,
            f64::from(self.center_lat_e7) / 1e7,
        )
    }

    /// Set the centre from degrees, rounding as
    /// [`Header::set_bounds_degrees`] does.
    pub fn set_center_degrees(&mut self, lon: f64, lat: f64) {
        self.center_lon_e7 = degrees_to_e7(lon);
        self.center_lat_e7 = degrees_to_e7(lat);
    }
}

/// Degrees to the spec's E7 fixed point, rounded half away from zero and
/// clamped rather than wrapped. A NaN becomes 0, because the alternative is
/// an unspecified cast.
fn degrees_to_e7(degrees: f64) -> i32 {
    if degrees.is_nan() {
        return 0;
    }
    let scaled = (degrees * 1e7).round();
    if scaled >= f64::from(i32::MAX) {
        i32::MAX
    } else if scaled <= f64::from(i32::MIN) {
        i32::MIN
    } else {
        scaled as i32
    }
}

fn put_u64(out: &mut [u8; HEADER_BYTES], at: usize, value: u64) {
    out[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn put_i32(out: &mut [u8; HEADER_BYTES], at: usize, value: i32) {
    out[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

/// Reads eight little-endian bytes at `at`. The caller has already checked
/// that `bytes` is at least [`HEADER_BYTES`] long and every `at` here is a
/// constant below 120, so the slice cannot be short.
fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(buf)
}

/// Reads four little-endian bytes at `at` as a **signed** integer. The
/// positions are the only signed fields in the header and reading them as
/// `u32` is the mistake that turns every western longitude into a number just
/// under 4.3 billion.
fn read_i32(bytes: &[u8], at: usize) -> i32 {
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&bytes[at..at + 4]);
    i32::from_le_bytes(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first 127 bytes of `raster-z0z2.pmtiles`, a golden archive written
    /// by go-pmtiles v1.31.2 (commit a3e4951ea6a0477b784c27c1dcbfd9c130878c5a,
    /// archive sha256
    /// `e2ed5e64f3c29efa3ec3b679ec5f1b06569c1b234c6eea762fb9f02fc23e9c12`),
    /// read verbatim out of the file.
    ///
    /// The expected field values in
    /// [`the_oracle_s_header_decodes_to_the_values_go_pmtiles_reports`] are
    /// what `pmtiles.DeserializeHeader` answered for these same bytes, so this
    /// is a comparison against another implementation rather than against this
    /// one's own encoder.
    const ORACLE_HEADER_HEX: &str = concat!(
        "504d54696c6573037f000000000000002300000000000000a200000000000000",
        "a200000000000000440100000000000000000000000000004401000000000000",
        "1206000000000000150000000000000015000000000000001500000000000000",
        "010201020002002eb694493a4ecd00d2496bb7c5b132010000000000000000",
    );

    fn oracle_header_bytes() -> Vec<u8> {
        let hex = ORACLE_HEADER_HEX;
        assert_eq!(
            hex.len(),
            HEADER_BYTES * 2,
            "the golden hex is not 127 bytes"
        );
        (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("golden hex"))
            .collect()
    }

    /// A header with a distinct value in every field, so a comparison of two
    /// headers cannot pass by two fields happening to hold the same number,
    /// and a swapped pair of same-typed fields cannot survive.
    fn saturated_header() -> Header {
        Header {
            root_offset: 0x1122_3344_5566_7788,
            root_length: 0x0102_0304_0506_0708,
            metadata_offset: 0x1111_2222_3333_4444,
            metadata_length: 0x5555_6666_7777_8888,
            leaf_directories_offset: 0x0A0B_0C0D_0E0F_1011,
            leaf_directories_length: 0x1213_1415_1617_1819,
            tile_data_offset: 0x2021_2223_2425_2627,
            tile_data_length: 0x3031_3233_3435_3637,
            addressed_tiles_count: 0x4041_4243_4445_4647,
            tile_entries_count: 0x5051_5253_5455_5657,
            tile_contents_count: 0x6061_6263_6465_6667,
            clustered: true,
            internal_compression: Compression::Gzip,
            tile_compression: Compression::None,
            tile_type: TileType::Webp,
            min_zoom: 3,
            max_zoom: 17,
            min_lon_e7: -123_456_789,
            min_lat_e7: -45_678_901,
            max_lon_e7: 109_876_543,
            max_lat_e7: 78_901_234,
            center_zoom: 11,
            center_lon_e7: -12_345_678,
            center_lat_e7: 87_654_321,
        }
    }

    #[test]
    fn the_oracle_s_header_decodes_to_the_values_go_pmtiles_reports() {
        let header = Header::try_decode(&oracle_header_bytes()).expect("a real archive's header");

        assert_eq!(header.root_offset, 127);
        assert_eq!(header.root_length, 35);
        assert_eq!(header.metadata_offset, 162);
        assert_eq!(header.metadata_length, 162);
        assert_eq!(header.leaf_directories_offset, 324);
        assert_eq!(header.leaf_directories_length, 0);
        assert_eq!(header.tile_data_offset, 324);
        assert_eq!(header.tile_data_length, 1554);
        assert_eq!(header.addressed_tiles_count, 21);
        assert_eq!(header.tile_entries_count, 21);
        assert_eq!(header.tile_contents_count, 21);
        assert!(header.clustered);
        assert_eq!(header.internal_compression, Compression::Gzip);
        assert_eq!(header.tile_compression, Compression::None);
        assert_eq!(header.tile_type, TileType::Png);
        assert_eq!(header.min_zoom, 0);
        assert_eq!(header.max_zoom, 2);
        assert_eq!(header.center_zoom, 1);

        // Longitude first, latitude second, and both signed. Reading the pair
        // the other way round gives (-85.05, -180.0), which still looks like a
        // bounding box, which is why this is pinned rather than eyeballed.
        assert_eq!(header.min_lon_e7, -1_800_000_000);
        assert_eq!(header.min_lat_e7, -850_511_287);
        assert_eq!(header.max_lon_e7, 1_800_000_000);
        assert_eq!(header.max_lat_e7, 850_511_287);
        let (min_lon, min_lat, max_lon, max_lat) = header.bounds_degrees();
        assert!((min_lon - -180.0).abs() < 1e-9, "min lon was {min_lon}");
        assert!(
            (min_lat - -85.051_128_7).abs() < 1e-9,
            "min lat was {min_lat}"
        );
        assert!((max_lon - 180.0).abs() < 1e-9, "max lon was {max_lon}");
        assert!(
            (max_lat - 85.051_128_7).abs() < 1e-9,
            "max lat was {max_lat}"
        );

        // And what this crate encodes from that decode is the same 127 bytes.
        assert_eq!(header.encode().to_vec(), oracle_header_bytes());
    }

    #[test]
    fn the_length_is_the_flag_for_leaves_not_the_offset() {
        // Measured on the oracle goldens: an archive with no leaf directories
        // still carries a non-zero `leaf_directories_offset`, equal to
        // `tile_data_offset`. Anything deciding "does this archive have
        // leaves" from the offset answers yes for every archive there is.
        let header = Header::try_decode(&oracle_header_bytes()).unwrap();
        assert_eq!(header.leaf_directories_offset, header.tile_data_offset);
        assert_ne!(header.leaf_directories_offset, 0);
        assert!(!header.has_leaves());

        // The positive control: a header that does have leaves says so.
        let with_leaves = Header {
            leaf_directories_length: 1,
            ..header
        };
        assert!(with_leaves.has_leaves());
    }

    #[test]
    fn every_field_lands_at_the_byte_offset_the_spec_gives_it() {
        // The offsets here are integer literals from the spec's byte grid, not
        // the module's own constants: a test that shares a wrong constant with
        // the code it checks proves nothing. This is also the only test that
        // can catch two same-typed fields being swapped, which a round trip
        // cannot see at all.
        let bytes = saturated_header().encode();
        let u64_at = |at: usize| {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[at..at + 8]);
            u64::from_le_bytes(buf)
        };
        let i32_at = |at: usize| {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&bytes[at..at + 4]);
            i32::from_le_bytes(buf)
        };

        assert_eq!(&bytes[0..7], b"PMTiles");
        assert_eq!(bytes[7], 3);
        assert_eq!(u64_at(8), 0x1122_3344_5566_7788, "root offset at 8");
        assert_eq!(u64_at(16), 0x0102_0304_0506_0708, "root length at 16");
        assert_eq!(u64_at(24), 0x1111_2222_3333_4444, "metadata offset at 24");
        assert_eq!(u64_at(32), 0x5555_6666_7777_8888, "metadata length at 32");
        assert_eq!(u64_at(40), 0x0A0B_0C0D_0E0F_1011, "leaf offset at 40");
        assert_eq!(u64_at(48), 0x1213_1415_1617_1819, "leaf length at 48");
        assert_eq!(u64_at(56), 0x2021_2223_2425_2627, "tile data offset at 56");
        assert_eq!(u64_at(64), 0x3031_3233_3435_3637, "tile data length at 64");
        assert_eq!(u64_at(72), 0x4041_4243_4445_4647, "addressed tiles at 72");
        assert_eq!(u64_at(80), 0x5051_5253_5455_5657, "tile entries at 80");
        assert_eq!(u64_at(88), 0x6061_6263_6465_6667, "tile contents at 88");
        assert_eq!(bytes[96], 1, "clustered at 96");
        assert_eq!(bytes[97], 2, "internal compression at 97");
        assert_eq!(bytes[98], 1, "tile compression at 98");
        assert_eq!(bytes[99], 4, "tile type at 99");
        assert_eq!(bytes[100], 3, "min zoom at 100");
        assert_eq!(bytes[101], 17, "max zoom at 101");
        assert_eq!(i32_at(102), -123_456_789, "min lon at 102");
        assert_eq!(i32_at(106), -45_678_901, "min lat at 106");
        assert_eq!(i32_at(110), 109_876_543, "max lon at 110");
        assert_eq!(i32_at(114), 78_901_234, "max lat at 114");
        assert_eq!(bytes[118], 11, "center zoom at 118");
        assert_eq!(i32_at(119), -12_345_678, "center lon at 119");
        assert_eq!(i32_at(123), 87_654_321, "center lat at 123");
        assert_eq!(bytes.len(), 127);
    }

    #[test]
    fn a_header_round_trips_through_its_bytes() {
        let header = saturated_header();
        assert_eq!(Header::try_decode(&header.encode()).unwrap(), header);

        // A longer buffer is accepted and only the first 127 bytes are read,
        // so a caller that fetched the customary first 16 KiB does not have to
        // subslice.
        let mut padded = header.encode().to_vec();
        padded.extend_from_slice(&[0xAB; 500]);
        assert_eq!(Header::try_decode(&padded).unwrap(), header);
    }

    #[test]
    fn a_negative_position_survives_the_round_trip() {
        // The three positions are the only signed fields in the header.
        // Reading them as `u32` turns every western longitude into a number
        // just under 4.3 billion, and the archive still opens.
        let header = Header {
            min_lon_e7: i32::MIN,
            min_lat_e7: -1,
            max_lon_e7: i32::MAX,
            max_lat_e7: 1,
            center_lon_e7: -900_000_000,
            center_lat_e7: -450_000_000,
            ..Header::default()
        };
        let back = Header::try_decode(&header.encode()).unwrap();
        assert_eq!(back.min_lon_e7, i32::MIN);
        assert_eq!(back.min_lat_e7, -1);
        assert_eq!(back.center_lon_e7, -900_000_000);
        assert_eq!(back.center_lat_e7, -450_000_000);
        let (center_lon, center_lat) = back.center_degrees();
        assert!((center_lon - -90.0).abs() < 1e-9);
        assert!((center_lat - -45.0).abs() < 1e-9);
    }

    #[test]
    fn the_three_refusals_are_short_magic_and_version() {
        let good = saturated_header().encode();

        let mut bad_magic = good;
        bad_magic[3] = b'X';
        assert!(matches!(
            Header::try_decode(&bad_magic),
            Err(PmTilesError::BadMagic { .. })
        ));

        // Version 2 is the interesting one: it shares these seven magic bytes
        // and lays its fields out completely differently, so reading it as v3
        // produces plausible garbage rather than an error.
        let mut v2 = good;
        v2[7] = 2;
        assert!(matches!(
            Header::try_decode(&v2),
            Err(PmTilesError::UnsupportedVersion { found: 2 })
        ));

        assert!(matches!(
            Header::try_decode(&good[..126]),
            Err(PmTilesError::ShortHeader {
                got: 126,
                want: 127
            })
        ));
        assert!(matches!(
            Header::try_decode(&[]),
            Err(PmTilesError::ShortHeader { got: 0, .. })
        ));

        // The positive control: the unmodified bytes decode, so each refusal
        // above is about what was changed rather than about the fixture.
        assert!(Header::try_decode(&good).is_ok());
    }

    #[test]
    fn a_descriptive_field_this_build_does_not_understand_is_kept_not_refused() {
        let mut bytes = saturated_header().encode();
        bytes[97] = 9; // an internal compression from a future revision
        bytes[98] = 3; // brotli, legal and not built here
        bytes[99] = 7; // a tile type from a future revision
        bytes[100] = 20; // min zoom above max zoom, which the spec lowercases
        bytes[101] = 4;

        let header = Header::try_decode(&bytes).expect("an archive must still open");
        assert_eq!(header.internal_compression, Compression::Other(9));
        assert_eq!(header.tile_compression, Compression::Brotli);
        assert_eq!(header.tile_type, TileType::Other(7));
        assert_eq!(header.min_zoom, 20);
        assert_eq!(header.max_zoom, 4);

        // The byte survives, which is the point: `Other(7)` and `Unknown` mean
        // different things and flattening one into the other throws away the
        // only signal that this build is out of date.
        assert_eq!(header.encode(), bytes);

        // And the refusal happens where it is load-bearing instead.
        assert!(matches!(
            header.internal_compression.decompress(b"anything", 1024),
            Err(PmTilesError::UnsupportedCompression { .. })
        ));
    }

    #[test]
    fn tile_types_and_compressions_round_trip_through_their_bytes() {
        for byte in 0u8..=255 {
            assert_eq!(TileType::from_byte(byte).to_byte(), byte);
            assert_eq!(Compression::from_byte(byte).to_byte(), byte);
        }
        // The named values, by name, so a renumbering fails here rather than
        // silently producing archives nothing else reads.
        assert_eq!(TileType::from_byte(0), TileType::Unknown);
        assert_eq!(TileType::from_byte(1), TileType::Mvt);
        assert_eq!(TileType::from_byte(2), TileType::Png);
        assert_eq!(TileType::from_byte(3), TileType::Jpeg);
        assert_eq!(TileType::from_byte(4), TileType::Webp);
        assert_eq!(TileType::from_byte(5), TileType::Avif);
        assert_eq!(TileType::from_byte(6), TileType::Mlt);
        assert_eq!(TileType::from_byte(7), TileType::Other(7));
        assert_eq!(Compression::from_byte(0), Compression::Unknown);
        assert_eq!(Compression::from_byte(1), Compression::None);
        assert_eq!(Compression::from_byte(2), Compression::Gzip);
        assert_eq!(Compression::from_byte(3), Compression::Brotli);
        assert_eq!(Compression::from_byte(4), Compression::Zstd);
    }

    #[test]
    fn the_libviprs_tile_formats_map_to_tile_types_and_raw_does_not() {
        assert_eq!(
            TileType::try_from_tile_format(TileFormat::Png).unwrap(),
            TileType::Png
        );
        assert_eq!(
            TileType::try_from_tile_format(TileFormat::Jpeg { quality: 80 }).unwrap(),
            TileType::Jpeg
        );
        // Raw pixel bytes carry neither their dimensions nor their pixel
        // format, so there is no tile type that could describe them.
        assert!(matches!(
            TileType::try_from_tile_format(TileFormat::Raw),
            Err(PmTilesError::UnsupportedTileFormat { .. })
        ));
        assert_eq!(TileType::Png.extension(), Some("png"));
        assert_eq!(TileType::Other(7).extension(), None);
        assert_eq!(TileType::Unknown.extension(), None);
    }

    #[test]
    fn gzip_round_trips_and_only_two_schemes_are_supported() {
        let payload = b"the quick brown fox jumps over the lazy dog, repeatedly".repeat(20);

        let squashed = Compression::Gzip.compress(&payload).unwrap();
        assert!(squashed.len() < payload.len(), "gzip did not compress");
        assert_eq!(
            Compression::Gzip.decompress(&squashed, 1 << 20).unwrap(),
            payload
        );

        assert_eq!(Compression::None.compress(&payload).unwrap(), payload);
        assert_eq!(
            Compression::None.decompress(&payload, 1 << 20).unwrap(),
            payload
        );

        for unsupported in [
            Compression::Brotli,
            Compression::Zstd,
            Compression::Unknown,
            Compression::Other(9),
        ] {
            assert!(
                matches!(
                    unsupported.compress(&payload),
                    Err(PmTilesError::UnsupportedCompression { .. })
                ),
                "{unsupported} must refuse to compress rather than pass bytes through"
            );
            assert!(matches!(
                unsupported.decompress(&squashed, 1 << 20),
                Err(PmTilesError::UnsupportedCompression { .. })
            ));
            assert!(!unsupported.is_supported());
        }
        assert!(Compression::None.is_supported());
        assert!(Compression::Gzip.is_supported());
    }

    #[test]
    fn decompression_stops_at_the_ceiling_it_was_given() {
        // No length field in PMTiles v3 is an uncompressed length, so this
        // ceiling is the only thing between a reader and a gzip bomb. A few
        // hundred stored bytes expand to a megabyte here.
        let bomb = Compression::Gzip.compress(&vec![0u8; 1 << 20]).unwrap();
        assert!(
            bomb.len() < 4096,
            "the bomb fixture is not compressed enough"
        );

        assert!(matches!(
            Compression::Gzip.decompress(&bomb, 4096),
            Err(PmTilesError::DecompressionLimit { limit: 4096 })
        ));

        // The boundary in both directions, because an off-by-one here either
        // refuses a legitimate directory or lets one byte past the cap.
        let exact = Compression::Gzip.compress(&vec![7u8; 1000]).unwrap();
        assert_eq!(
            Compression::Gzip.decompress(&exact, 1000).unwrap().len(),
            1000
        );
        assert!(matches!(
            Compression::Gzip.decompress(&exact, 999),
            Err(PmTilesError::DecompressionLimit { limit: 999 })
        ));

        // The uncompressed path is capped too, or a `None` archive would be
        // the way around the ceiling.
        assert!(matches!(
            Compression::None.decompress(&[0u8; 100], 99),
            Err(PmTilesError::DecompressionLimit { .. })
        ));
        assert_eq!(
            Compression::None
                .decompress(&[0u8; 100], 100)
                .unwrap()
                .len(),
            100
        );
    }

    #[test]
    fn degrees_convert_through_the_e7_fixed_point_without_wrapping() {
        let mut header = Header::default();
        header.set_bounds_degrees(-180.0, -85.0511287, 180.0, 85.0511287);
        assert_eq!(header.min_lon_e7, -1_800_000_000);
        assert_eq!(header.min_lat_e7, -850_511_287);
        assert_eq!(header.max_lon_e7, 1_800_000_000);
        assert_eq!(header.max_lat_e7, 850_511_287);

        // Rounds half away from zero rather than truncating toward it:
        // truncation shrinks a bounding box, and a box one unit too small
        // excludes tiles at its own edge.
        header.set_bounds_degrees(0.000_000_05, -0.000_000_05, 0.000_000_14, -0.000_000_14);
        assert_eq!(header.min_lon_e7, 1);
        assert_eq!(header.min_lat_e7, -1);
        assert_eq!(header.max_lon_e7, 1);
        assert_eq!(header.max_lat_e7, -1);

        // Nonsense stays nonsense rather than becoming its own negation.
        header.set_bounds_degrees(1e9, -1e9, f64::NAN, f64::INFINITY);
        assert_eq!(header.min_lon_e7, i32::MAX);
        assert_eq!(header.min_lat_e7, i32::MIN);
        assert_eq!(header.max_lon_e7, 0);
        assert_eq!(header.max_lat_e7, i32::MAX);
    }
}
