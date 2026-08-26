//! Image IO, metadata fields, and library-level free functions ported
//! from libvips.
//!
//! This module is the seventh batch of the libvips operation surface
//! required by the ported integration tests (alongside
//! [`crate::composite`]): saving a [`Raster`] to a file by extension, the
//! named metadata field system (`vips_image_get` / `vips_image_set`), the
//! native `.v` container, and the small library-level helpers
//! ([`tokenize`], [`parse_thumbnail_geometry`]).
//!
//! # Saving
//!
//! [`Raster::save`] encodes by file extension and keeps attached metadata
//! where the container supports it; [`Raster::save_stripped`] writes
//! pixels only (libvips `strip`):
//!
//! | Extension | Encoder | Metadata kept by `save` |
//! |---|---|---|
//! | `.png` | [`crate::sink::encode_png`] | none yet (iCCP embedding is an open gap) |
//! | `.jpg` / `.jpeg` | the sink JPEG encoder at quality 75 | `icc-profile-data` (APP2), `exif-data` (APP1, raw blob) |
//! | `.gif` | [`Raster::encode_gif`] at the vips defaults | none: what a GIF carries (palette, `loop`, `delay`) is structural, not EXIF-class |
//! | `.webp` | [`Raster::encode_webp`], lossless | `icc-profile-data` (`ICCP`), `exif-data` (`EXIF`), `xmp-data` (`XMP `) |
//! | `.v` / `.vips` | [`Raster::encode_vips`] | header geometry plus every attached field |
//!
//! Formats libviprs cannot encode yet (TIFF-with-metadata, ...) return
//! [`SaveError::UnsupportedExtension`]; they arrive with the
//! foreign-format batch. Structured EXIF tag writing (`exif-ifd0-*`
//! fields into the TIFF directory of a JPEG APP1 segment) is also
//! deferred to the foreign batch: those fields round-trip through `.v`
//! and travel on the raster, but JPEG save only re-embeds the raw
//! `exif-data` blob captured at decode time.
//!
//! # The `.v` container
//!
//! `encode_vips` writes the libvips native header (64 bytes: magic,
//! geometry, band format, coding, interpretation, resolution, offsets)
//! followed by the raw pixel data in the file's byte order, then a JSON
//! trailer with the orientation tag and the attached metadata fields. The
//! trailer is written only when it would carry something: a raster with the
//! upright orientation and no attached fields writes the header and the
//! pixels and stops, because an empty trailer costs a reader a warning and
//! buys nobody anything (issue #546).
//! libvips itself stores an XML trailer there, and a trailer that never
//! claimed to be libviprs JSON is treated as absent (bar the orientation
//! tag, which the reader recovers from the vips XML), so pixels and header
//! survive either way. The reader accepts both byte orders (swapping
//! 16-bit and float samples as needed), rejects band formats other than
//! uchar, ushort, and float, and enforces the
//! [`DecodeLimits::max_coord`] dimension ceiling on untrusted header
//! geometry.
//!
//! ## Trailer compatibility across versions
//!
//! The trailer is read entry by entry rather than as one `serde` value, so
//! a `.v` written by a newer libviprs costs an older one only the entries
//! it genuinely cannot represent (issue #565). One unknown
//! [`MetadataValue`] variant used to fail the whole parse and take every
//! other field on the image with it, silently, which made a new variant a
//! data-loss break that `cargo semver-checks` could not see. The rules the
//! format holds to now:
//!
//! * an entry whose value matches no [`MetadataValue`] variant is carried
//!   opaquely: invisible to [`Raster::get_field`] and [`Raster::get_fields`],
//!   because this build cannot say what it means, but written back out
//!   unchanged, so an old build that opens a new file and re-saves it does
//!   not strip what it could not read. Setting or removing a field of the
//!   same name supersedes it, so stripping still strips;
//! * a trailer key this build does not know is ignored, and one it expects
//!   but does not find falls back to the default (`orientation` to 1);
//! * the wire shape is frozen at `{"orientation":N,"fields":{"entries":
//!   [[name,value],...]}}` with values in [`MetadataValue`]'s externally
//!   tagged form, so what this build writes still parses on every build
//!   already released;
//! * a trailer that opens with `{` and is not valid JSON is corruption
//!   rather than a foreign format, and the reader reports it. That is the
//!   one case where metadata is genuinely unrecoverable, and it is narrow
//!   enough that no libviprs or libvips writer can produce it.
//!
//! The interpretation word holds libvips' own `VipsInterpretation` codes,
//! which since libvips 8.18 include `30` and `31` for OkLab and OkLch
//! (`libvips/include/vips/image.h:115-116`); a `.v` libviprs writes now
//! carries the same tag real vips writes. Files libviprs wrote before this
//! release carry the private codes `1000` / `1001` it used while libvips
//! had none, and still read back tagged, because the reader keeps those two
//! as permanently reserved read-only aliases. Nothing writes them any more,
//! so the incompatibility only runs the other way: a `.v` written now reads
//! as `Multiband` on libviprs 0.4.0 and earlier.
//!
//! # Metadata fields
//!
//! [`Raster::get_field`] / [`Raster::set_field`] expose the libvips
//! named-field system over a [`MetadataValue`] enum (int, double, string,
//! blob). The built-in header fields (`width`, `height`, `bands`,
//! `format`, `coding`, `interpretation`, `xoffset`, `yoffset`, `xres`,
//! `yres`, `orientation`, `filename`) read through to the raster header
//! and [`crate::conversion`] metadata block; every other name is an
//! attached field carried with the raster (cloned with it, written to
//! `.v`, and — for `icc-profile-data` / `exif-data` — embedded on JPEG
//! save). Geometry fields are read-only, matching what a fixed pixel
//! buffer can honour.
//!
//! # Free functions
//!
//! * [`tokenize`] — the libvips option-string tokeniser (whitespace
//!   splitting with `"..."` / `'...'` quoting and backslash escapes).
//! * [`parse_thumbnail_geometry`] — the vipsthumbnail size parser
//!   (`W`, `WxH`, `Wx`, `xH`, with trailing `<` / `>` / `!` modifiers
//!   tolerated).
//!
//! The single-axis coordinate ceiling (libvips `VIPS_MAX_COORD`) is not a
//! library-level free function: it lives on the per-decode
//! [`DecodeLimits::max_coord`] field (default [`DEFAULT_MAX_COORD`]),
//! enforced by *every* decoder. Build a [`DecodeLimits`] with
//! [`DecodeLimits::with_max_coord`] and pass it to a `*_with_limits` decode
//! call to tighten it; there is no process-global to set.
//!
//! # Decode limits
//!
//! Two levels govern how much work an untrusted input may provoke, and no
//! more: a per-decode [`DecodeLimits`] (single-axis `max_coord`, total
//! `max_pixels`, and decoder `max_alloc_bytes`, all applied before the
//! pixels are materialised) and a per-constructor allocation budget
//! enforced by the [`Raster`] constructors. `max_coord` is honoured
//! uniformly by both the native `.v` reader and the `image`-crate raster
//! path (PNG/JPEG/TIFF); see [`DecodeLimits`] for the full per-format
//! field mapping. There is no third process-global regime: the coordinate
//! ceiling lives solely on `DecodeLimits`.

use std::path::Path;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::conversion::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::sink::SinkError;
use crate::source::{DecodeLimits, SourceError};

// ---------------------------------------------------------------------------
// Metadata values and the attached-field store
// ---------------------------------------------------------------------------

/// A typed metadata field value (libvips stores these as boxed `GValue`s;
/// here the four kinds the ported suites read and write are first-class).
///
/// `From` conversions cover the literal shapes the ported tests use
/// (`"TestSoftware".into()`, `format!(...).into()`), and the `as_*`
/// accessors mirror pyvips-style coercing reads: they panic with a
/// descriptive message when the variant does not match, in line with the
/// panicking convenience layer of the operation modules.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    /// A signed integer field (`width`, `orientation`, counts, flags).
    Int(i64),
    /// A floating-point field (`xres`, `yres`).
    Double(f64),
    /// A text field (EXIF strings, `interpretation`, `filename`).
    Str(String),
    /// A binary field (`icc-profile-data`, `exif-data`).
    Blob(Vec<u8>),
}

impl MetadataValue {
    /// The value as a string slice.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Str`].
    #[track_caller]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Str(s) => s,
            other => panic!("metadata value is {}, not a string", other.kind()),
        }
    }

    /// The value as a `u32`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Int`] or does not fit
    /// in a `u32`.
    #[track_caller]
    pub fn as_u32(&self) -> u32 {
        match self {
            Self::Int(i) => u32::try_from(*i)
                .unwrap_or_else(|_| panic!("metadata int {i} does not fit in a u32")),
            other => panic!("metadata value is {}, not an int", other.kind()),
        }
    }

    /// The value as an `i64`.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Int`].
    #[track_caller]
    pub fn as_i64(&self) -> i64 {
        match self {
            Self::Int(i) => *i,
            other => panic!("metadata value is {}, not an int", other.kind()),
        }
    }

    /// The value as an `f64`; an [`MetadataValue::Int`] coerces.
    ///
    /// # Panics
    ///
    /// Panics if the value is not numeric.
    #[track_caller]
    pub fn as_f64(&self) -> f64 {
        match self {
            Self::Double(d) => *d,
            Self::Int(i) => *i as f64,
            other => panic!("metadata value is {}, not numeric", other.kind()),
        }
    }

    /// The value as a byte slice.
    ///
    /// # Panics
    ///
    /// Panics if the value is not [`MetadataValue::Blob`].
    #[track_caller]
    pub fn as_blob(&self) -> &[u8] {
        match self {
            Self::Blob(b) => b,
            other => panic!("metadata value is {}, not a blob", other.kind()),
        }
    }

    /// The type code returned by [`Raster::get_typeof`] for this value:
    /// 1 int, 2 double, 3 string, 4 blob. These are libviprs codes (the
    /// C library returns GObject `GType` numbers); the ported call sites
    /// only distinguish zero (absent) from non-zero (present).
    pub fn type_code(&self) -> u64 {
        match self {
            Self::Int(_) => 1,
            Self::Double(_) => 2,
            Self::Str(_) => 3,
            Self::Blob(_) => 4,
        }
    }

    /// The length of this value in its natural unit.
    ///
    /// * [`MetadataValue::Blob`]: the number of bytes (what
    ///   `image.get("icc-profile-data").len()` reports in the ported foreign
    ///   cell, for example the 564-byte ICC profile of `sample.jpg` read
    ///   through magick).
    /// * [`MetadataValue::Str`]: the number of UTF-8 bytes in the string.
    /// * [`MetadataValue::Int`] / [`MetadataValue::Double`]: `1`, a scalar
    ///   is a single-element field.
    pub fn len(&self) -> usize {
        match self {
            Self::Blob(b) => b.len(),
            Self::Str(s) => s.len(),
            Self::Int(_) | Self::Double(_) => 1,
        }
    }

    /// Whether this value has zero length; see [`MetadataValue::len`]. A
    /// scalar [`MetadataValue::Int`] or [`MetadataValue::Double`] is never
    /// empty (its length is `1`).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Human-readable kind name for panic messages.
    fn kind(&self) -> &'static str {
        match self {
            Self::Int(_) => "an int",
            Self::Double(_) => "a double",
            Self::Str(_) => "a string",
            Self::Blob(_) => "a blob",
        }
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        Self::Str(s.to_string())
    }
}
impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        Self::Str(s)
    }
}
impl From<i64> for MetadataValue {
    fn from(i: i64) -> Self {
        Self::Int(i)
    }
}
impl From<i32> for MetadataValue {
    fn from(i: i32) -> Self {
        Self::Int(i64::from(i))
    }
}
impl From<u32> for MetadataValue {
    fn from(i: u32) -> Self {
        Self::Int(i64::from(i))
    }
}
impl From<f64> for MetadataValue {
    fn from(d: f64) -> Self {
        Self::Double(d)
    }
}
impl From<Vec<u8>> for MetadataValue {
    fn from(b: Vec<u8>) -> Self {
        Self::Blob(b)
    }
}
impl From<&[u8]> for MetadataValue {
    fn from(b: &[u8]) -> Self {
        Self::Blob(b.to_vec())
    }
}

/// The attached (non-header) metadata fields carried by a [`Raster`]:
/// an insertion-ordered name/value list, so [`Raster::get_fields`]
/// reports attachments in the order they were set, like libvips.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct MetadataFields {
    entries: Vec<(String, MetadataValue)>,
    /// Fields read from a `.v` trailer whose value matches no
    /// [`MetadataValue`] variant this build has, kept as the raw JSON that
    /// was on disk (issue #565).
    ///
    /// They stay out of the field API on purpose: this build can say that
    /// the field was there, not what it means, and a wrong answer is worse
    /// than none. Carrying them is what stops a rewrite by an older build
    /// from stripping what a newer one wrote.
    unknown: Vec<(String, serde_json::Value)>,
}

impl MetadataFields {
    pub(crate) fn get(&self, name: &str) -> Option<&MetadataValue> {
        self.entries
            .iter()
            .find_map(|(n, v)| (n == name).then_some(v))
    }

    /// Upsert keeping first-set order. A value the caller can name
    /// supersedes an uninterpretable one carried under the same name.
    pub(crate) fn set(&mut self, name: &str, value: MetadataValue) {
        self.unknown.retain(|(n, _)| n != name);
        if let Some(slot) = self
            .entries
            .iter_mut()
            .find_map(|(n, v)| (n == name).then_some(v))
        {
            *slot = value;
        } else {
            self.entries.push((name.to_string(), value));
        }
    }

    /// Remove a field under either carrier, so removing a name that arrived
    /// uninterpretable really removes it rather than leaving it to reappear
    /// on the next save. The uninterpretable carrier has no
    /// [`MetadataValue`] to hand back, so removing one reads as absent.
    pub(crate) fn remove(&mut self, name: &str) -> Option<MetadataValue> {
        self.unknown.retain(|(n, _)| n != name);
        let idx = self.entries.iter().position(|(n, _)| n == name)?;
        Some(self.entries.remove(idx).1)
    }

    /// Record a field this build cannot interpret; see
    /// [`MetadataFields::unknown`].
    fn set_unknown(&mut self, name: &str, value: serde_json::Value) {
        self.entries.retain(|(n, _)| n != name);
        if let Some(slot) = self
            .unknown
            .iter_mut()
            .find_map(|(n, v)| (n == name).then_some(v))
        {
            *slot = value;
        } else {
            self.unknown.push((name.to_string(), value));
        }
    }

    pub(crate) fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(n, _)| n.as_str())
    }

    /// Whether the raster carries no attached field at all, interpretable
    /// or not. A `.v` trailer with nothing in it is worse than no trailer
    /// (issue #546), so the writer asks this before writing one.
    fn is_empty(&self) -> bool {
        self.entries.is_empty() && self.unknown.is_empty()
    }

    /// The interpretable fields, in the order they were set.
    fn known(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.entries.iter().map(|(n, v)| (n.as_str(), v))
    }

    /// The fields carried opaquely; see [`MetadataFields::unknown`].
    fn unknown_fields(&self) -> impl Iterator<Item = (&str, &serde_json::Value)> {
        self.unknown.iter().map(|(n, v)| (n.as_str(), v))
    }
}

/// Errors from [`Raster::try_set_field`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetadataError {
    #[error("metadata field {name:?} is read-only (fixed by the pixel buffer)")]
    ReadOnlyField { name: String },
    #[error("metadata field {name:?} expects {expected}, got {got}")]
    WrongType {
        name: String,
        expected: &'static str,
        got: &'static str,
    },
    #[error("unknown interpretation nickname {value:?}")]
    UnknownInterpretation { value: String },
}

/// The built-in header field names, in the order libvips reports them
/// (header geometry first, `width` leading).
const BUILTIN_FIELDS: [&str; 12] = [
    "width",
    "height",
    "bands",
    "format",
    "coding",
    "interpretation",
    "xoffset",
    "yoffset",
    "xres",
    "yres",
    "orientation",
    "filename",
];

/// The libvips nickname for an [`Interpretation`].
fn interpretation_nickname(i: Interpretation) -> &'static str {
    match i {
        Interpretation::Multiband => "multiband",
        Interpretation::Bw => "b-w",
        Interpretation::Histogram => "histogram",
        Interpretation::Xyz => "xyz",
        Interpretation::Lab => "lab",
        Interpretation::Cmyk => "cmyk",
        Interpretation::Labq => "labq",
        Interpretation::Rgb => "rgb",
        Interpretation::Cmc => "cmc",
        Interpretation::Lch => "lch",
        Interpretation::Labs => "labs",
        Interpretation::Srgb => "srgb",
        Interpretation::Yxy => "yxy",
        Interpretation::Fourier => "fourier",
        Interpretation::Rgb16 => "rgb16",
        Interpretation::Grey16 => "grey16",
        Interpretation::Matrix => "matrix",
        Interpretation::ScRgb => "scrgb",
        Interpretation::Hsv => "hsv",
        Interpretation::OkLab => "oklab",
        Interpretation::OkLch => "oklch",
    }
}

/// Parse a libvips interpretation nickname.
fn interpretation_from_nickname(s: &str) -> Option<Interpretation> {
    Some(match s {
        "multiband" => Interpretation::Multiband,
        "b-w" => Interpretation::Bw,
        "histogram" => Interpretation::Histogram,
        "xyz" => Interpretation::Xyz,
        "lab" => Interpretation::Lab,
        "cmyk" => Interpretation::Cmyk,
        "labq" => Interpretation::Labq,
        "rgb" => Interpretation::Rgb,
        "cmc" => Interpretation::Cmc,
        "lch" => Interpretation::Lch,
        "labs" => Interpretation::Labs,
        "srgb" => Interpretation::Srgb,
        "yxy" => Interpretation::Yxy,
        "fourier" => Interpretation::Fourier,
        "rgb16" => Interpretation::Rgb16,
        "grey16" => Interpretation::Grey16,
        "matrix" => Interpretation::Matrix,
        "scrgb" => Interpretation::ScRgb,
        "hsv" => Interpretation::Hsv,
        "oklab" => Interpretation::OkLab,
        "oklch" => Interpretation::OkLch,
        _ => return None,
    })
}

/// The libvips `VipsInterpretation` enum value for the `.v` header
/// `Type` word, as declared in `libvips/include/vips/image.h:96-117`
/// (8.18.4). `OkLab` and `OkLch` are `VIPS_INTERPRETATION_OKLAB` = 30
/// and `VIPS_INTERPRETATION_OKLCH` = 31 (`image.h:115-116`), so a `.v`
/// libviprs writes carries the same tag real vips writes.
fn interpretation_code(i: Interpretation) -> i32 {
    match i {
        Interpretation::Multiband => 0,
        Interpretation::Bw => 1,
        Interpretation::Histogram => 10,
        Interpretation::Xyz => 12,
        Interpretation::Lab => 13,
        Interpretation::Cmyk => 15,
        Interpretation::Labq => 16,
        Interpretation::Rgb => 17,
        Interpretation::Cmc => 18,
        Interpretation::Lch => 19,
        Interpretation::Labs => 21,
        Interpretation::Srgb => 22,
        Interpretation::Yxy => 23,
        Interpretation::Fourier => 24,
        Interpretation::Rgb16 => 25,
        Interpretation::Grey16 => 26,
        Interpretation::Matrix => 27,
        Interpretation::ScRgb => 28,
        Interpretation::Hsv => 29,
        Interpretation::OkLab => 30,
        Interpretation::OkLch => 31,
    }
}

/// The `.v` header `Type` word back to an [`Interpretation`]: a *left*
/// inverse of [`interpretation_code`] (every code that function writes reads
/// back as the variant it came from), widened by the two read-only legacy
/// aliases `1000` / `1001`, which land on `OkLab` / `OkLch` alongside the
/// libvips codes `30` / `31`. So it is not a bijection, and it is not the
/// other direction of the round trip for those four codes. Unknown codes
/// read as `None` and the raster falls back to format inference, like an
/// untagged image.
fn interpretation_from_code(code: i32) -> Option<Interpretation> {
    Some(match code {
        0 => Interpretation::Multiband,
        1 => Interpretation::Bw,
        10 => Interpretation::Histogram,
        12 => Interpretation::Xyz,
        13 => Interpretation::Lab,
        15 => Interpretation::Cmyk,
        16 => Interpretation::Labq,
        17 => Interpretation::Rgb,
        18 => Interpretation::Cmc,
        19 => Interpretation::Lch,
        21 => Interpretation::Labs,
        22 => Interpretation::Srgb,
        23 => Interpretation::Yxy,
        24 => Interpretation::Fourier,
        25 => Interpretation::Rgb16,
        26 => Interpretation::Grey16,
        27 => Interpretation::Matrix,
        28 => Interpretation::ScRgb,
        29 => Interpretation::Hsv,
        30 => Interpretation::OkLab,
        31 => Interpretation::OkLch,
        // Read-only legacy aliases: before libvips 8.18 assigned 30 / 31,
        // libviprs wrote these private codes above the libvips range. Files
        // it already wrote keep loading; nothing emits them any more. They
        // are reserved permanently and must never be reused for anything
        // else: the only thing retiring them would achieve is to silently
        // re-break every `.v` libviprs has already written.
        1000 => Interpretation::OkLab,
        1001 => Interpretation::OkLch,
        _ => return None,
    })
}

impl Raster {
    /// Read a metadata field by name (libvips `vips_image_get`): the
    /// built-in header fields (`width`, `height`, `bands`, `format`,
    /// `coding`, `interpretation`, `xoffset`, `yoffset`, `xres`, `yres`,
    /// `orientation`, `filename`) or any attached field previously set
    /// with [`Raster::set_field`] / [`Raster::set_icc_profile`] or
    /// captured at decode time (`icc-profile-data`, `exif-data`).
    ///
    /// Returns `None` for a name that is neither built-in nor attached.
    pub fn get_field(&self, name: &str) -> Option<MetadataValue> {
        Some(match name {
            "width" => MetadataValue::Int(i64::from(self.width())),
            "height" => MetadataValue::Int(i64::from(self.height())),
            "bands" => MetadataValue::Int(self.format().channels() as i64),
            "format" => MetadataValue::Str(
                match self.format().bytes_per_channel() {
                    1 => "uchar",
                    2 => "ushort",
                    _ => "float",
                }
                .to_string(),
            ),
            "coding" => MetadataValue::Str("none".to_string()),
            "interpretation" => {
                MetadataValue::Str(interpretation_nickname(self.interpretation()).to_string())
            }
            "xoffset" => MetadataValue::Int(i64::from(self.xoffset())),
            "yoffset" => MetadataValue::Int(i64::from(self.yoffset())),
            "xres" => MetadataValue::Double(self.xres()),
            "yres" => MetadataValue::Double(self.yres()),
            "orientation" => MetadataValue::Int(i64::from(self.orientation())),
            "filename" => self
                .fields
                .get("filename")
                .cloned()
                .unwrap_or_else(|| MetadataValue::Str(String::new())),
            other => return self.fields.get(other).cloned(),
        })
    }

    /// Read a metadata field as an `i32` (libvips `vips_image_get_int`).
    ///
    /// Resolves `name` through [`Raster::get_field`] and returns the value
    /// when it is an integer that fits in an `i32` (for example the built-in
    /// `width`/`height`/`bands` header fields, or an attached field such as
    /// `bits-per-sample`, `tile-width`, or `page-height` set by a loader).
    /// Returns `None` for an absent field, a non-integer value, or an integer
    /// outside the `i32` range.
    pub fn get_int(&self, name: &str) -> Option<i32> {
        match self.get_field(name)? {
            MetadataValue::Int(value) => i32::try_from(value).ok(),
            _ => None,
        }
    }

    /// Fallible form of [`Raster::set_field`].
    ///
    /// # Errors
    ///
    /// [`MetadataError::ReadOnlyField`] for the geometry fields fixed by
    /// the pixel buffer (`width`, `height`, `bands`, `format`,
    /// `coding`), [`MetadataError::WrongType`] when a built-in field is
    /// given the wrong value kind or an out-of-range number, or
    /// [`MetadataError::UnknownInterpretation`] for an unrecognised
    /// interpretation nickname.
    pub fn try_set_field(&mut self, name: &str, value: MetadataValue) -> Result<(), MetadataError> {
        fn int_as<T: TryFrom<i64>>(
            name: &str,
            value: &MetadataValue,
            expected: &'static str,
        ) -> Result<T, MetadataError> {
            match value {
                MetadataValue::Int(i) => T::try_from(*i).map_err(|_| MetadataError::WrongType {
                    name: name.to_string(),
                    expected,
                    got: "an out-of-range int",
                }),
                other => Err(MetadataError::WrongType {
                    name: name.to_string(),
                    expected,
                    got: other.kind(),
                }),
            }
        }
        match name {
            "width" | "height" | "bands" | "format" | "coding" => {
                Err(MetadataError::ReadOnlyField {
                    name: name.to_string(),
                })
            }
            "xres" | "yres" => {
                let v = match &value {
                    MetadataValue::Double(d) => *d,
                    MetadataValue::Int(i) => *i as f64,
                    other => {
                        return Err(MetadataError::WrongType {
                            name: name.to_string(),
                            expected: "a double",
                            got: other.kind(),
                        });
                    }
                };
                if name == "xres" {
                    self.meta.xres = v;
                } else {
                    self.meta.yres = v;
                }
                Ok(())
            }
            "xoffset" => {
                self.meta.xoffset = int_as::<i32>(name, &value, "an int (i32 range)")?;
                Ok(())
            }
            "yoffset" => {
                self.meta.yoffset = int_as::<i32>(name, &value, "an int (i32 range)")?;
                Ok(())
            }
            "orientation" => {
                self.meta.orientation = int_as::<u8>(name, &value, "an int (u8 range)")?;
                Ok(())
            }
            "interpretation" => match &value {
                MetadataValue::Str(s) => match interpretation_from_nickname(s) {
                    Some(i) => {
                        self.meta.interpretation = Some(i);
                        Ok(())
                    }
                    None => Err(MetadataError::UnknownInterpretation { value: s.clone() }),
                },
                other => Err(MetadataError::WrongType {
                    name: name.to_string(),
                    expected: "a string",
                    got: other.kind(),
                }),
            },
            other => {
                self.fields.set(other, value);
                Ok(())
            }
        }
    }

    /// Set a metadata field by name (libvips `vips_image_set`): built-in
    /// header fields route into the raster metadata block, every other
    /// name becomes an attached field carried with the raster. Panicking
    /// form of [`Raster::try_set_field`], matching the ported-test call
    /// surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`MetadataError`]; see [`Raster::try_set_field`].
    #[track_caller]
    pub fn set_field(&mut self, name: &str, value: MetadataValue) {
        if let Err(e) = self.try_set_field(name, value) {
            panic!("set_field: {e}");
        }
    }

    /// Every readable field name: the built-in header fields (in libvips
    /// order, `width` first) followed by the attached fields in the
    /// order they were set (libvips `vips_image_get_fields`).
    pub fn get_fields(&self) -> Vec<String> {
        let mut out: Vec<String> = BUILTIN_FIELDS.iter().map(|s| s.to_string()).collect();
        for name in self.fields.names() {
            if !BUILTIN_FIELDS.contains(&name) {
                out.push(name.to_string());
            }
        }
        out
    }

    /// The type code of a field: 0 when the field does not exist,
    /// otherwise the [`MetadataValue::type_code`] of its value (libvips
    /// `vips_image_get_typeof`, which returns 0 or a `GType`).
    pub fn get_typeof(&self, name: &str) -> u64 {
        self.get_field(name).map_or(0, |v| v.type_code())
    }

    /// The EXIF orientation tag (libvips `orientation`, values 1-8),
    /// defaulting to `1` (upright) when the image carries no orientation.
    ///
    /// Reads the `orientation` header field; every raster has one, so this
    /// is total. The oracle reports `orientation = 1` for the reference
    /// images that are not pre-rotated.
    pub fn get_orientation(&self) -> i32 {
        i32::from(self.orientation())
    }

    /// The number of pages this raster represents (libvips `n-pages`),
    /// defaulting to `1` for a single-page image.
    ///
    /// Reads the attached `n-pages` field. Three loaders set it and they do
    /// not agree on when, so the field being *present* does not mean the
    /// image has more than one page:
    ///
    /// * [`crate::gif`] and [`decode_tiff_page`](crate::decode_tiff_page)
    ///   attach it on every load, a one-frame GIF and a single-page TIFF
    ///   included, where it reads `1`;
    /// * [`crate::webp`] attaches it only when the file is animated, so a
    ///   still WebP carries no such field at all, which is what
    ///   `vipsheader -a` reports for one;
    /// * the PDF loader does not attach it.
    ///
    /// A raster with no `n-pages` field reports `1` either way, matching the
    /// oracle (`n-pages = 1` for `sample.jpg`, `5` / `4` / `3` / `35` for the
    /// animated fixtures).
    ///
    /// The count is 1-based and the page index is 0-based (issue #566), so a
    /// sweep over every page is `for page in 0..raster.get_n_pages()`.
    pub fn get_n_pages(&self) -> u32 {
        match self.get_field("n-pages") {
            Some(MetadataValue::Int(n)) => u32::try_from(n).ok().filter(|&n| n > 0).unwrap_or(1),
            Some(MetadataValue::Str(s)) => s.parse::<u32>().ok().filter(|&n| n > 0).unwrap_or(1),
            _ => 1,
        }
    }

    /// Remove an attached field by setting its type to 0, the libvips
    /// removal idiom (`vips_image_set` with a zero `GType` /
    /// `vips_image_remove`). Removing an absent field is a no-op.
    ///
    /// # Panics
    ///
    /// Panics when `typeof_` is non-zero (retyping a field is not
    /// supported; set a new value instead) or when `name` is a built-in
    /// header field, which cannot be removed.
    #[track_caller]
    pub fn set_typeof(&mut self, name: &str, typeof_: u64) {
        if typeof_ != 0 {
            panic!("set_typeof: only 0 (remove) is supported, got {typeof_}");
        }
        if BUILTIN_FIELDS.contains(&name) {
            panic!("set_typeof: cannot remove built-in field {name:?}");
        }
        self.fields.remove(name);
    }

    /// Attach an ICC profile, stored as the `icc-profile-data` blob
    /// field (embedded on JPEG and `.v` save).
    pub fn set_icc_profile(&mut self, profile: &[u8]) {
        self.fields
            .set("icc-profile-data", MetadataValue::Blob(profile.to_vec()));
    }

    /// The attached ICC profile, if any: the `icc-profile-data` blob set
    /// by [`Raster::set_icc_profile`] or captured from a decoded JPEG.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        match self.fields.get("icc-profile-data") {
            Some(MetadataValue::Blob(b)) => Some(b),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

/// Errors from [`Raster::save`] / [`Raster::save_stripped`] /
/// [`Raster::encode_vips`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SaveError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(
        "unsupported save extension {extension:?}; libviprs encodes png, jpg/jpeg, gif, \
         webp, and v/vips"
    )]
    UnsupportedExtension { extension: String },
    #[error("encode error: {0}")]
    Encode(#[from] SinkError),
}

/// JPEG quality used by extension-dispatched [`Raster::save`], matching
/// the libvips `jpegsave` default.
const SAVE_JPEG_QUALITY: u8 = 75;

impl Raster {
    /// Save to a file, choosing the encoder from the extension (libvips
    /// `vips_image_write_to_file`). Attached metadata is kept where the
    /// container supports it; see the [module docs](crate::imageio) for
    /// the per-format table.
    ///
    /// # Errors
    ///
    /// [`SaveError::UnsupportedExtension`] for extensions libviprs
    /// cannot encode, [`SaveError::Encode`] if the encoder rejects the
    /// pixel format, or [`SaveError::Io`] on write failure.
    pub fn save(&self, path: &Path) -> Result<(), SaveError> {
        self.save_impl(path, true)
    }

    /// Save with all metadata stripped (libvips `strip`): pixels and
    /// geometry only, no ICC, EXIF, or attached fields.
    ///
    /// # Errors
    ///
    /// As [`Raster::save`].
    pub fn save_stripped(&self, path: &Path) -> Result<(), SaveError> {
        self.save_impl(path, false)
    }

    fn save_impl(&self, path: &Path, keep_metadata: bool) -> Result<(), SaveError> {
        let extension = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        let bytes = match extension.as_str() {
            "png" => crate::sink::encode_png(self)?,
            "jpg" | "jpeg" => {
                let encoded = crate::sink::encode_jpeg(self, SAVE_JPEG_QUALITY)?;
                if keep_metadata {
                    let exif = match self.fields.get("exif-data") {
                        Some(MetadataValue::Blob(b)) => Some(b.as_slice()),
                        _ => None,
                    };
                    inject_jpeg_metadata(encoded, exif, self.icc_profile())
                } else {
                    encoded
                }
            }
            // GIF carries no EXIF or ICC through libviprs' encoder, so
            // `keep_metadata` has nothing to act on: the container holds a
            // palette, a loop count and per-frame delays, all of which are
            // structural rather than EXIF-class (`vips gifsave --keep none`
            // does not drop `delay` either). WebP is the other way round and
            // takes the flag.
            "gif" => self
                .encode_gif(crate::gif::SaveOptions::default())
                .map_err(|e| match e {
                    crate::codec::EncodeError::Io(io) => SaveError::Io(io),
                    other => SaveError::Encode(SinkError::EncodeMsg(other.to_string())),
                })?,
            "webp" => crate::webp::encode_webp_for_save(self, keep_metadata)?,
            "v" | "vips" => self.encode_vips_impl(keep_metadata),
            _ => return Err(SaveError::UnsupportedExtension { extension }),
        };
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Encode as native `.v` bytes (libvips `vipssave_buffer`): the
    /// 64-byte libvips header, raw pixels, and a metadata trailer. See
    /// the [module docs](crate::imageio) for the container contract.
    ///
    /// # Errors
    ///
    /// Currently infallible for every representable [`PixelFormat`];
    /// the `Result` reserves room for unencodable future formats.
    pub fn encode_vips(&self) -> Result<Vec<u8>, SaveError> {
        Ok(self.encode_vips_impl(true))
    }

    fn encode_vips_impl(&self, keep_metadata: bool) -> Vec<u8> {
        let bpc = self.format().bytes_per_channel();
        let mut out = Vec::with_capacity(VIPS_HEADER_LEN + self.data().len());
        out.extend_from_slice(&VIPS_MAGIC_NATIVE);
        fn push_i32(out: &mut Vec<u8>, v: i32) {
            out.extend_from_slice(&v.to_ne_bytes());
        }
        push_i32(&mut out, self.width() as i32);
        push_i32(&mut out, self.height() as i32);
        push_i32(&mut out, self.format().channels() as i32);
        push_i32(&mut out, 8 * bpc as i32); // deprecated Bbits
        // BandFmt (VipsBandFormat codes): 0 = uchar, 2 = ushort, 6 = float.
        let band_fmt = match bpc {
            1 => 0,
            2 => 2,
            _ => 6,
        };
        push_i32(&mut out, band_fmt);
        push_i32(&mut out, 0); // Coding: none
        push_i32(&mut out, interpretation_code(self.interpretation()));
        out.extend_from_slice(&(self.xres() as f32).to_ne_bytes());
        out.extend_from_slice(&(self.yres() as f32).to_ne_bytes());
        push_i32(&mut out, 0); // deprecated Length
        out.extend_from_slice(&0i16.to_ne_bytes()); // deprecated Compression
        out.extend_from_slice(&0i16.to_ne_bytes()); // deprecated Level
        push_i32(&mut out, self.xoffset());
        push_i32(&mut out, self.yoffset());
        out.resize(VIPS_HEADER_LEN, 0); // reserved tail of the header
        out.extend_from_slice(self.data());
        if keep_metadata && self.has_v_trailer_content() {
            let trailer = VTrailer {
                orientation: self.orientation(),
                fields: VFields {
                    entries: self
                        .fields
                        .known()
                        .map(|(name, v)| (name, VFieldValue::Known(v)))
                        .chain(
                            self.fields
                                .unknown_fields()
                                .map(|(name, v)| (name, VFieldValue::Unknown(v))),
                        )
                        .collect(),
                },
            };
            if let Ok(json) = serde_json::to_vec(&trailer) {
                out.extend_from_slice(&json);
            }
        }
        out
    }

    /// Whether a `.v` trailer would carry anything: an orientation other
    /// than the upright default, or at least one attached field (readable
    /// or carried opaquely).
    ///
    /// An empty trailer is not free. libvips parses that slot as XML, so the
    /// 41 bytes of `{"orientation":1,"fields":{"entries":[]}}` that every
    /// plain [`Raster::save`] used to append made `vipsheader -a` print
    /// `VIPS-WARNING **: error reading vips image metadata: VipsImage: XML
    /// parse error` and throw the whole metadata block away. A file with no
    /// trailer at all reads silently (measured on vips 8.18.4, issue #546),
    /// so writing nothing is strictly better than writing nothing useful.
    fn has_v_trailer_content(&self) -> bool {
        self.orientation() != 1 || !self.fields.is_empty()
    }
}

// ---------------------------------------------------------------------------
// The .v container
// ---------------------------------------------------------------------------

/// Size of the on-disk libvips image header.
const VIPS_HEADER_LEN: usize = 64;

/// First four file bytes of a big-endian (SPARC-order) `.v` file.
const VIPS_MAGIC_BE: [u8; 4] = [0x08, 0xf2, 0xa6, 0xb6];
/// First four file bytes of a little-endian (Intel-order) `.v` file.
const VIPS_MAGIC_LE: [u8; 4] = [0xb6, 0xa6, 0xf2, 0x08];

/// The magic this build writes: native byte order, as libvips does.
#[cfg(target_endian = "little")]
const VIPS_MAGIC_NATIVE: [u8; 4] = VIPS_MAGIC_LE;
#[cfg(target_endian = "big")]
const VIPS_MAGIC_NATIVE: [u8; 4] = VIPS_MAGIC_BE;

/// The JSON trailer libviprs writes after the pixel data: the
/// orientation tag plus the attached fields. libvips stores XML here.
///
/// Write-only. Reading goes through [`read_json_trailer`], which walks the
/// JSON entry by entry rather than deserialising into this shape, because
/// one `serde` value for the whole trailer is exactly what made a single
/// unreadable field cost the image every other one (issue #565).
///
/// The wire shape is frozen at what libviprs 0.4 wrote,
/// `{"orientation":N,"fields":{"entries":[[name,value],...]}}` with values
/// in [`MetadataValue`]'s externally tagged form, so every build already
/// released can still read what this one writes.
#[derive(Serialize)]
struct VTrailer<'a> {
    orientation: u8,
    fields: VFields<'a>,
}

/// The attached fields as they sit in a [`VTrailer`].
#[derive(Serialize)]
struct VFields<'a> {
    entries: Vec<(&'a str, VFieldValue<'a>)>,
}

/// One trailer value on its way out to disk.
///
/// `untagged` means each arm serialises as its own contents and adds
/// nothing, so a [`MetadataValue`] keeps its externally tagged form and an
/// opaque value goes back exactly as it arrived.
#[derive(Serialize)]
#[serde(untagged)]
enum VFieldValue<'a> {
    /// A value this build understands, e.g. `{"Int":3}`.
    Known(&'a MetadataValue),
    /// A value a newer build wrote, echoed verbatim.
    Unknown(&'a serde_json::Value),
}

/// Whether `bytes` begin with a `.v` magic (either byte order).
pub(crate) fn is_vips_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && (bytes[..4] == VIPS_MAGIC_LE || bytes[..4] == VIPS_MAGIC_BE)
}

/// Decode a native `.v` file (both byte orders). Enforces the caller's
/// [`DecodeLimits`] — the [`max_coord`](DecodeLimits::max_coord)
/// single-axis ceiling and the pixel budget — on the untrusted header
/// geometry before allocating.
pub(crate) fn decode_vips_bytes(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    if bytes.len() < VIPS_HEADER_LEN {
        return Err(SourceError::VipsFormat(format!(
            "file too short for a .v header: {} bytes",
            bytes.len()
        )));
    }
    let swapped = if bytes[..4] == VIPS_MAGIC_LE {
        cfg!(target_endian = "big")
    } else if bytes[..4] == VIPS_MAGIC_BE {
        cfg!(target_endian = "little")
    } else {
        return Err(SourceError::VipsFormat("bad .v magic".to_string()));
    };
    let read_i32 = |off: usize| -> i32 {
        let raw: [u8; 4] = bytes[off..off + 4].try_into().expect("length checked");
        let v = i32::from_ne_bytes(raw);
        if swapped { v.swap_bytes() } else { v }
    };
    let read_f32 = |off: usize| -> f32 { f32::from_bits(read_i32(off) as u32) };

    let width = read_i32(4);
    let height = read_i32(8);
    let bands = read_i32(12);
    let band_fmt = read_i32(20);
    let coding = read_i32(24);
    let type_code = read_i32(28);
    let xres = read_f32(32);
    let yres = read_f32(36);
    let xoffset = read_i32(48);
    let yoffset = read_i32(52);

    if coding != 0 {
        return Err(SourceError::VipsFormat(format!(
            "unsupported .v coding {coding}; only uncoded images are supported"
        )));
    }
    let bpc = match band_fmt {
        0 => 1, // uchar
        2 => 2, // ushort
        6 => 4, // float
        other => {
            return Err(SourceError::VipsFormat(format!(
                "unsupported .v band format {other}; only uchar, ushort, and float \
                 are supported"
            )));
        }
    };
    if width <= 0 || height <= 0 || bands <= 0 {
        return Err(SourceError::VipsFormat(format!(
            "bad .v geometry {width}x{height} with {bands} bands"
        )));
    }
    let (width, height) = (width as u32, height as u32);
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    let format = PixelFormat::with_channels(bands as usize, bpc)
        .ok_or_else(|| SourceError::VipsFormat(format!("unrepresentable .v band count {bands}")))?;

    let data_len = width as usize * height as usize * format.bytes_per_pixel();
    let end = VIPS_HEADER_LEN
        .checked_add(data_len)
        .filter(|&e| e <= bytes.len())
        .ok_or_else(|| {
            SourceError::VipsFormat(format!(
                "truncated .v pixel data: header promises {data_len} bytes, file has {}",
                bytes.len().saturating_sub(VIPS_HEADER_LEN)
            ))
        })?;
    let mut data = bytes[VIPS_HEADER_LEN..end].to_vec();
    if swapped && bpc == 2 {
        for pair in data.as_chunks_mut::<2>().0 {
            pair.swap(0, 1);
        }
    }
    if swapped && bpc == 4 {
        for quad in data.as_chunks_mut::<4>().0 {
            quad.reverse();
        }
    }

    let mut raster = Raster::new(width, height, format, data)?;
    raster.meta.xres = f64::from(xres);
    raster.meta.yres = f64::from(yres);
    raster.meta.xoffset = xoffset;
    raster.meta.yoffset = yoffset;
    raster.meta.interpretation = interpretation_from_code(type_code);
    // Trailer: libviprs JSON carries the orientation tag and every attached
    // field. A `.v` written by real libvips instead carries an XML metadata
    // block here; we still recover the orientation tag from it so `autorot`
    // has the same cross-oracle vips does (the remaining XML fields are not
    // parsed yet). A trailer that never claimed to be ours is treated as
    // absent, as libvips treats ours.
    if end < bytes.len() {
        let trailer = &bytes[end..];
        if is_json_trailer(trailer) {
            let json: serde_json::Value = serde_json::from_slice(trailer).map_err(|err| {
                SourceError::VipsFormat(format!("corrupt .v metadata trailer: {err}"))
            })?;
            read_json_trailer(&json, &mut raster);
        } else if let Some(orientation) = parse_vips_xml_orientation(trailer) {
            raster.meta.orientation = orientation;
        }
    }
    Ok(raster)
}

/// Whether `trailer` claims to be a libviprs JSON trailer: its first
/// non-whitespace byte is `{`.
///
/// This is what separates a trailer someone else wrote from one of ours
/// that is broken. libvips puts XML in the same slot and other writers may
/// put anything there, so silence is the right answer for those; it is not
/// the right answer for a libviprs trailer whose metadata is genuinely
/// unrecoverable (issue #565).
fn is_json_trailer(trailer: &[u8]) -> bool {
    trailer
        .iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|&b| b == b'{')
}

/// Apply a libviprs JSON trailer to `raster`, one entry at a time.
///
/// Total on purpose. Every part of the trailer is optional and every entry
/// is read on its own, so a `.v` written by a newer libviprs costs this
/// build only the entries it genuinely cannot represent and never the ones
/// it can (issue #565). An entry whose value matches no [`MetadataValue`]
/// variant is carried opaquely rather than dropped, so re-saving the image
/// here does not strip what the newer build wrote.
fn read_json_trailer(json: &serde_json::Value, raster: &mut Raster) {
    if let Some(orientation) = json
        .get("orientation")
        .and_then(serde_json::Value::as_u64)
        .and_then(|n| u8::try_from(n).ok())
    {
        raster.meta.orientation = orientation;
    }
    let Some(entries) = json
        .get("fields")
        .and_then(|fields| fields.get("entries"))
        .and_then(serde_json::Value::as_array)
    else {
        return;
    };
    for entry in entries {
        let Some([name, value]) = entry.as_array().map(Vec::as_slice) else {
            continue;
        };
        let Some(name) = name.as_str() else {
            continue;
        };
        match serde_json::from_value::<MetadataValue>(value.clone()) {
            Ok(known) => raster.fields.set(name, known),
            Err(_) => raster.fields.set_unknown(name, value.clone()),
        }
    }
}

/// Recover the EXIF-style orientation tag from a real-libvips `.v` XML
/// metadata trailer.
///
/// libvips serialises image metadata as an XML block after the pixel data,
/// storing the orientation as `<field type="gint" name="orientation">N</field>`.
/// This extracts that integer (1-8) so a `.v` file vips itself wrote decodes
/// with the correct orientation for [`Raster::autorot`]; the distinct
/// lowercase `name="orientation"` is not shared by the `exif-ifd0-Orientation`
/// string field. The anchor requires the field element to close immediately
/// (`name="orientation">`), matching vips's deterministic
/// `<field type="gint" name="orientation">` serialization, so a longer
/// field name (e.g. a hypothetical `name="orientation-foo"`) cannot
/// false-match. Returns `None` when the trailer is not valid UTF-8, carries
/// no such field, or the value is out of the 1-8 range.
///
/// Only the orientation is recovered from a real-libvips XML trailer; the
/// remaining fields (exif-data, icc-profile-data, resolution, n-pages, …) are
/// not parsed — see issue #487. Core's own JSON trailer preserves them, so
/// this partial affects only round-tripping a `.v` that vips itself wrote.
fn parse_vips_xml_orientation(trailer: &[u8]) -> Option<u8> {
    let text = std::str::from_utf8(trailer).ok()?;
    let anchor = text.find(r#"name="orientation">"#)?;
    let after = &text[anchor..];
    let open = after.find('>')?;
    let rest = &after[open + 1..];
    let close = rest.find('<')?;
    let value: u16 = rest[..close].trim().parse().ok()?;
    (1..=8).contains(&value).then_some(value as u8)
}

// ---------------------------------------------------------------------------
// JPEG metadata segments
// ---------------------------------------------------------------------------

/// APP1 EXIF identifier.
const EXIF_HEADER: &[u8] = b"Exif\0\0";
/// APP2 ICC identifier.
const ICC_HEADER: &[u8] = b"ICC_PROFILE\0";
/// Maximum ICC payload bytes per APP2 segment: 65535 total segment
/// length, minus 2 length bytes, 12 identifier bytes, and 2 chunk-index
/// bytes.
const ICC_CHUNK_MAX: usize = 65535 - 2 - 12 - 2;

/// Scan a JPEG byte stream and attach the `exif-data` and
/// `icc-profile-data` fields found in its APP1/APP2 segments, as libvips
/// loaders populate the image header. Non-JPEG or malformed input
/// attaches nothing.
pub(crate) fn attach_jpeg_metadata(raster: &mut Raster, bytes: &[u8]) {
    let (exif, icc) = extract_jpeg_metadata(bytes);
    if let Some(exif) = exif {
        raster.fields.set("exif-data", MetadataValue::Blob(exif));
    }
    if let Some(icc) = icc {
        raster
            .fields
            .set("icc-profile-data", MetadataValue::Blob(icc));
    }
}

/// Extract the EXIF payload (after `Exif\0\0`) and the reassembled ICC
/// profile from a JPEG stream's application segments.
fn extract_jpeg_metadata(bytes: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut exif = None;
    let mut icc_chunks: Vec<(u8, &[u8])> = Vec::new();
    for (marker, data) in JpegSegments::new(bytes) {
        match marker {
            0xE1 if exif.is_none() && data.starts_with(EXIF_HEADER) => {
                exif = Some(data[EXIF_HEADER.len()..].to_vec());
            }
            0xE2 if data.starts_with(ICC_HEADER) && data.len() >= ICC_HEADER.len() + 2 => {
                let seq = data[ICC_HEADER.len()];
                icc_chunks.push((seq, &data[ICC_HEADER.len() + 2..]));
            }
            _ => {}
        }
    }
    let icc = if icc_chunks.is_empty() {
        None
    } else {
        icc_chunks.sort_by_key(|(seq, _)| *seq);
        let mut profile = Vec::with_capacity(icc_chunks.iter().map(|(_, d)| d.len()).sum());
        for (_, d) in &icc_chunks {
            profile.extend_from_slice(d);
        }
        Some(profile)
    };
    (exif, icc)
}

/// Iterator over JPEG marker segments: yields `(marker, segment data)`
/// for every length-carrying segment before the scan data.
struct JpegSegments<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JpegSegments<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        // Require SOI; otherwise iterate nothing.
        let pos = if bytes.starts_with(&[0xFF, 0xD8]) {
            2
        } else {
            bytes.len()
        };
        Self { bytes, pos }
    }
}

impl<'a> Iterator for JpegSegments<'a> {
    type Item = (u8, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Marker: 0xFF then a non-fill byte.
            while self.pos < self.bytes.len() && self.bytes[self.pos] == 0xFF {
                self.pos += 1;
            }
            if self.pos == 0 || self.pos >= self.bytes.len() || self.bytes[self.pos - 1] != 0xFF {
                return None;
            }
            let marker = self.bytes[self.pos];
            self.pos += 1;
            match marker {
                // EOI or SOS: nothing structured follows.
                0xD9 | 0xDA => return None,
                // Standalone markers with no length word.
                0x01 | 0xD0..=0xD7 => continue,
                _ => {
                    if self.pos + 2 > self.bytes.len() {
                        return None;
                    }
                    let len = u16::from_be_bytes([self.bytes[self.pos], self.bytes[self.pos + 1]])
                        as usize;
                    if len < 2 || self.pos + len > self.bytes.len() {
                        return None;
                    }
                    let data = &self.bytes[self.pos + 2..self.pos + len];
                    self.pos += len;
                    return Some((marker, data));
                }
            }
        }
    }
}

/// Insert APP1 (EXIF) and APP2 (ICC) segments into an encoded JPEG,
/// after the SOI and any APP0 (JFIF) segment the encoder wrote. Payloads
/// too large for their segment framing are skipped rather than
/// corrupting the stream (EXIF cannot be split; ICC splits into up to
/// 255 chunks).
fn inject_jpeg_metadata(jpeg: Vec<u8>, exif: Option<&[u8]>, icc: Option<&[u8]>) -> Vec<u8> {
    if !jpeg.starts_with(&[0xFF, 0xD8]) {
        return jpeg;
    }
    // Find the insertion point: after SOI and consecutive APP0 segments.
    let mut insert_at = 2;
    while insert_at + 4 <= jpeg.len() && jpeg[insert_at] == 0xFF && jpeg[insert_at + 1] == 0xE0 {
        let len = u16::from_be_bytes([jpeg[insert_at + 2], jpeg[insert_at + 3]]) as usize;
        if len < 2 || insert_at + 2 + len > jpeg.len() {
            break;
        }
        insert_at += 2 + len;
    }

    let mut segments = Vec::new();
    if let Some(exif) = exif {
        let payload_len = EXIF_HEADER.len() + exif.len();
        if payload_len + 2 <= u16::MAX as usize {
            segments.extend_from_slice(&[0xFF, 0xE1]);
            segments.extend_from_slice(&((payload_len + 2) as u16).to_be_bytes());
            segments.extend_from_slice(EXIF_HEADER);
            segments.extend_from_slice(exif);
        }
    }
    if let Some(icc) = icc {
        let chunks: Vec<&[u8]> = icc.chunks(ICC_CHUNK_MAX).collect();
        if !chunks.is_empty() && chunks.len() <= 255 {
            let total = chunks.len() as u8;
            for (i, chunk) in chunks.iter().enumerate() {
                let payload_len = ICC_HEADER.len() + 2 + chunk.len();
                segments.extend_from_slice(&[0xFF, 0xE2]);
                segments.extend_from_slice(&((payload_len + 2) as u16).to_be_bytes());
                segments.extend_from_slice(ICC_HEADER);
                segments.push(i as u8 + 1);
                segments.push(total);
                segments.extend_from_slice(chunk);
            }
        }
    }
    if segments.is_empty() {
        return jpeg;
    }
    let mut out = Vec::with_capacity(jpeg.len() + segments.len());
    out.extend_from_slice(&jpeg[..insert_at]);
    out.extend_from_slice(&segments);
    out.extend_from_slice(&jpeg[insert_at..]);
    out
}

// ---------------------------------------------------------------------------
// Free functions: tokeniser, thumbnail geometry, max_coord, env init
// ---------------------------------------------------------------------------

/// Split an option string into tokens the way the libvips tokeniser
/// (`vips__token_get`) does: whitespace separates tokens, double or
/// single quotes group a token, and a backslash escapes the next
/// character both inside and outside quotes.
///
/// ```
/// assert_eq!(libviprs::tokenize("a \"b c\" d"), vec!["a", "b c", "d"]);
/// ```
pub fn tokenize(input: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = input.chars().peekable();
    loop {
        while chars.next_if(|c| c.is_whitespace()).is_some() {}
        let Some(&first) = chars.peek() else { break };
        let mut token = String::new();
        if first == '"' || first == '\'' {
            let quote = first;
            chars.next();
            while let Some(c) = chars.next() {
                match c {
                    c if c == quote => break,
                    '\\' => {
                        if let Some(escaped) = chars.next() {
                            token.push(escaped);
                        }
                    }
                    c => token.push(c),
                }
            }
        } else {
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                chars.next();
                if c == '\\' {
                    if let Some(escaped) = chars.next() {
                        token.push(escaped);
                    }
                } else {
                    token.push(c);
                }
            }
        }
        out.push(token);
    }
    out
}

/// A parsed thumbnail size specification; see
/// [`parse_thumbnail_geometry`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ThumbnailGeometry {
    /// Requested width bound, if the spec named one.
    pub width: Option<u32>,
    /// Requested height bound, if the spec named one.
    pub height: Option<u32>,
}

/// Parse a vipsthumbnail-style geometry string: `"200"` (width only),
/// `"200x150"`, `"200x"` (width only), `"x150"` (height only), with the
/// trailing vipsthumbnail modifiers `<` (only enlarge), `>` (only
/// shrink), and `!` (force) tolerated and stripped. Unparseable parts
/// read as `None`, so garbage yields an empty geometry rather than an
/// error, matching the forgiving CLI parser.
pub fn parse_thumbnail_geometry(spec: &str) -> ThumbnailGeometry {
    let s = spec.trim().trim_end_matches(['<', '>', '!']).trim();
    let (w, h) = match s.split_once(['x', 'X']) {
        Some((w, h)) => (w, Some(h)),
        None => (s, None),
    };
    ThumbnailGeometry {
        width: w.trim().parse().ok(),
        height: h.and_then(|h| h.trim().parse().ok()),
    }
}

/// The default [`DecodeLimits::max_coord`] ceiling: 10,000,000 pixels per
/// axis, the libvips `VIPS_MAX_COORD` value.
pub const DEFAULT_MAX_COORD: u32 = 10_000_000;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{decode_bytes, decode_file};

    #[test]
    fn metadata_value_len_reports_blob_and_string_bytes() {
        // The foreign magickload cell asserts `icc.len() == 564` on the ICC
        // blob; `len` on a blob is its byte count.
        assert_eq!(MetadataValue::Blob(vec![0u8; 564]).len(), 564);
        assert_eq!(MetadataValue::Str("abc".to_string()).len(), 3);
        // A scalar field is a single element.
        assert_eq!(MetadataValue::Int(7).len(), 1);
        assert_eq!(MetadataValue::Double(1.5).len(), 1);
    }

    #[test]
    fn metadata_value_is_empty_tracks_len() {
        assert!(MetadataValue::Blob(Vec::new()).is_empty());
        assert!(MetadataValue::Str(String::new()).is_empty());
        assert!(!MetadataValue::Blob(vec![1]).is_empty());
        assert!(!MetadataValue::Int(0).is_empty());
    }

    #[test]
    fn get_orientation_defaults_to_one() {
        // Matches the oracle: `orientation = 1` for the un-rotated fixtures.
        assert_eq!(Raster::black(4, 4).get_orientation(), 1);
    }

    #[test]
    fn get_orientation_reads_the_exif_field() {
        let mut im = Raster::black(4, 4);
        im.set_field("orientation", 6i32.into());
        assert_eq!(im.get_orientation(), 6);
    }

    #[test]
    fn get_n_pages_defaults_to_one() {
        // Matches the oracle: `n-pages = 1` for single-page images.
        assert_eq!(Raster::black(4, 4).get_n_pages(), 1);
    }

    #[test]
    fn get_n_pages_reads_the_attached_field() {
        let mut im = Raster::black(4, 4);
        im.set_field("n-pages", MetadataValue::Int(5));
        assert_eq!(im.get_n_pages(), 5);

        // A non-positive count falls back to the single-page default.
        let mut im0 = Raster::black(4, 4);
        im0.set_field("n-pages", MetadataValue::Int(0));
        assert_eq!(im0.get_n_pages(), 1);
    }

    fn rgb_2x2() -> Raster {
        Raster::new(
            2,
            2,
            PixelFormat::Rgb8,
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        )
        .unwrap()
    }

    // -- metadata fields ---------------------------------------------------

    /**
     * Tests the built-in header fields read through get_field with the
     * libvips names and value kinds.
     * Input: 2x2 Rgb8 with default metadata.
     * Output: width/height/bands ints, format "uchar", coding "none",
     * interpretation "srgb", xres/yres doubles, orientation 1.
     */
    #[test]
    fn builtin_fields_read_through() {
        let im = rgb_2x2();
        assert_eq!(im.get_field("width"), Some(MetadataValue::Int(2)));
        assert_eq!(im.get_field("height"), Some(MetadataValue::Int(2)));
        assert_eq!(im.get_field("bands"), Some(MetadataValue::Int(3)));
        assert_eq!(im.get_field("format").unwrap().as_str(), "uchar");
        assert_eq!(im.get_field("coding").unwrap().as_str(), "none");
        assert_eq!(im.get_field("interpretation").unwrap().as_str(), "srgb");
        assert_eq!(im.get_field("xres"), Some(MetadataValue::Double(1.0)));
        assert_eq!(im.get_field("yres"), Some(MetadataValue::Double(1.0)));
        assert_eq!(im.get_field("xoffset"), Some(MetadataValue::Int(0)));
        assert_eq!(im.get_field("yoffset"), Some(MetadataValue::Int(0)));
        assert_eq!(im.get_field("orientation"), Some(MetadataValue::Int(1)));
        assert_eq!(im.get_field("filename").unwrap().as_str(), "");
        assert_eq!(im.get_field("no-such-field"), None);

        let im16 = Raster::zeroed(1, 1, PixelFormat::Gray16).unwrap();
        assert_eq!(im16.get_field("format").unwrap().as_str(), "ushort");
        assert_eq!(im16.get_field("interpretation").unwrap().as_str(), "grey16");
    }

    /**
     * Tests get_fields: more than 10 names, "width" first, attached
     * fields listed after the built-ins in insertion order.
     */
    #[test]
    fn get_fields_order_and_count() {
        let mut im = rgb_2x2();
        let fields = im.get_fields();
        assert!(fields.len() > 10, "got {}", fields.len());
        assert_eq!(fields[0], "width");

        im.set_field("zz-custom", 7i32.into());
        im.set_field("aa-custom", "x".into());
        let fields = im.get_fields();
        let zz = fields.iter().position(|f| f == "zz-custom").unwrap();
        let aa = fields.iter().position(|f| f == "aa-custom").unwrap();
        assert!(zz < aa, "insertion order, not name order: {fields:?}");
    }

    /**
     * Tests set_field routing: header fields update the metadata block,
     * arbitrary string/int/double fields round-trip, and the geometry
     * fields reject writes with a typed error.
     */
    #[test]
    fn set_field_routes_builtins_and_attachments() {
        let mut im = rgb_2x2();
        im.set_field("xres", 2.5f64.into());
        im.set_field("yres", MetadataValue::Int(3));
        im.set_field("xoffset", 7i32.into());
        im.set_field("yoffset", (-4i32).into());
        im.set_field("orientation", 6i32.into());
        im.set_field("interpretation", "b-w".into());
        assert_eq!(im.xres(), 2.5);
        assert_eq!(im.yres(), 3.0);
        assert_eq!(im.xoffset(), 7);
        assert_eq!(im.yoffset(), -4);
        assert_eq!(im.orientation(), 6);
        assert_eq!(im.interpretation(), Interpretation::Bw);

        im.set_field("exif-ifd0-Software", "TestSoftware".into());
        im.set_field("page-count", 3i32.into());
        im.set_field("gamma", 2.2f64.into());
        assert_eq!(
            im.get_field("exif-ifd0-Software").unwrap().as_str(),
            "TestSoftware"
        );
        assert_eq!(im.get_field("page-count").unwrap().as_i64(), 3);
        assert_eq!(im.get_field("gamma").unwrap().as_f64(), 2.2);

        assert!(matches!(
            im.try_set_field("width", MetadataValue::Int(99)),
            Err(MetadataError::ReadOnlyField { .. })
        ));
        assert!(matches!(
            im.try_set_field("orientation", "sideways".into()),
            Err(MetadataError::WrongType { .. })
        ));
        assert!(matches!(
            im.try_set_field("orientation", MetadataValue::Int(999)),
            Err(MetadataError::WrongType { .. })
        ));
        assert!(matches!(
            im.try_set_field("interpretation", "plaid".into()),
            Err(MetadataError::UnknownInterpretation { .. })
        ));
    }

    /**
     * Tests get_typeof/set_typeof: 0 for absent fields, the value kind
     * code for present ones, and set_typeof(_, 0) removes an attachment
     * (the libvips removal idiom the ported EXIF test uses).
     */
    #[test]
    fn typeof_reports_and_removes() {
        let mut im = rgb_2x2();
        assert_eq!(im.get_typeof("exif-ifd0-Software"), 0);
        im.set_field("exif-ifd0-Software", "TestSoftware".into());
        assert_ne!(im.get_typeof("exif-ifd0-Software"), 0);
        assert_eq!(im.get_typeof("width"), MetadataValue::Int(0).type_code());
        assert_eq!(
            im.get_typeof("xres"),
            MetadataValue::Double(0.0).type_code()
        );

        im.set_typeof("exif-ifd0-Software", 0);
        assert_eq!(im.get_typeof("exif-ifd0-Software"), 0);
        assert_eq!(im.get_field("exif-ifd0-Software"), None);
        // Removing an absent field is a no-op.
        im.set_typeof("never-set", 0);
    }

    /**
     * Tests set_typeof rejects non-zero codes and built-in removal.
     */
    #[test]
    #[should_panic(expected = "only 0 (remove) is supported")]
    fn set_typeof_rejects_retype() {
        rgb_2x2().set_typeof("x", 3);
    }

    #[test]
    #[should_panic(expected = "cannot remove built-in field")]
    fn set_typeof_rejects_builtin_removal() {
        rgb_2x2().set_typeof("width", 0);
    }

    /**
     * Tests the ICC profile accessors and that the profile travels with
     * clones (the copy builder path every op uses for metadata).
     */
    #[test]
    fn icc_profile_roundtrip_and_clone() {
        let mut im = rgb_2x2();
        assert_eq!(im.icc_profile(), None);
        im.set_icc_profile(&[1, 2, 3, 4]);
        assert_eq!(im.icc_profile(), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(
            im.get_field("icc-profile-data"),
            Some(MetadataValue::Blob(vec![1, 2, 3, 4]))
        );
        let copy = im.copy().xres(9.0).build();
        assert_eq!(copy.icc_profile(), Some(&[1u8, 2, 3, 4][..]));
    }

    /**
     * Tests MetadataValue accessors panic descriptively on kind
     * mismatches and coerce where documented (Int -> f64).
     */
    #[test]
    fn metadata_value_accessors() {
        assert_eq!(MetadataValue::from("x").as_str(), "x");
        assert_eq!(MetadataValue::from(7u32).as_u32(), 7);
        assert_eq!(MetadataValue::from(-7i64).as_i64(), -7);
        assert_eq!(MetadataValue::from(2i32).as_f64(), 2.0);
        assert_eq!(MetadataValue::from(vec![9u8]).as_blob(), &[9]);
        let r = std::panic::catch_unwind(|| MetadataValue::Int(1).as_str());
        assert!(r.is_err());
        let r = std::panic::catch_unwind(|| MetadataValue::Int(-1).as_u32());
        assert!(r.is_err());
    }

    /**
     * Tests `get_int` coerces integer fields (built-in `bands` and an
     * attached loader field) to `i32`, and returns `None` for an absent
     * field, a non-integer value, and an out-of-`i32`-range integer.
     */
    #[test]
    fn get_int_reads_integer_fields() {
        let mut im = rgb_2x2();
        // Built-in header field resolves through get_field as an Int.
        assert_eq!(im.get_int("bands"), Some(3));

        // An attached loader-style field round-trips through get_int.
        im.set_field("bits-per-sample", MetadataValue::Int(8));
        assert_eq!(im.get_int("bits-per-sample"), Some(8));

        // Absent, non-integer, and out-of-range cases all yield None.
        assert_eq!(im.get_int("no-such-field"), None);
        im.set_field("note", "hello".into());
        assert_eq!(im.get_int("note"), None);
        im.set_field("huge", MetadataValue::Int(i64::from(i32::MAX) + 1));
        assert_eq!(im.get_int("huge"), None);
    }

    // -- save / .v ----------------------------------------------------------

    /**
     * Tests the .v container round-trip: pixels, header metadata
     * (interpretation, resolution, offsets), orientation, and attached
     * fields (string and blob) all survive encode_vips + decode.
     */
    #[test]
    fn vips_roundtrip_preserves_pixels_and_metadata() {
        let mut im = rgb_2x2()
            .copy()
            .interpretation(Interpretation::Bw)
            .xres(4.5)
            .yres(2.25)
            .xoffset(-3)
            .yoffset(9)
            .orientation(6)
            .build();
        im.set_field("exif-data", MetadataValue::Blob(vec![9, 8, 7]));
        im.set_field("note", "hello".into());
        im.set_icc_profile(&[5, 5, 5]);

        let bytes = im.encode_vips().unwrap();
        assert!(is_vips_bytes(&bytes));
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.width(), 2);
        assert_eq!(back.height(), 2);
        assert_eq!(back.format(), PixelFormat::Rgb8);
        assert_eq!(back.data(), im.data());
        assert_eq!(back.interpretation(), Interpretation::Bw);
        assert_eq!(back.xres(), 4.5);
        assert_eq!(back.yres(), 2.25);
        assert_eq!(back.xoffset(), -3);
        assert_eq!(back.yoffset(), 9);
        assert_eq!(back.orientation(), 6);
        assert_eq!(back.get_field("exif-data"), im.get_field("exif-data"));
        assert_eq!(back.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(back.icc_profile(), Some(&[5u8, 5, 5][..]));
    }

    /**
     * Tests 16-bit .v round-trip including the byte-swap path: a file
     * written in the foreign byte order decodes to the same samples.
     */
    #[test]
    fn vips_16bit_and_foreign_endian() {
        let im = Raster::new(
            2,
            1,
            PixelFormat::Gray16,
            [513u16, 65534]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        )
        .unwrap();
        let bytes = im.encode_vips().unwrap();
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.data(), im.data());

        // Flip the file to the foreign byte order by hand: swap the
        // magic, every header word, and the 16-bit samples.
        let mut foreign = bytes.clone();
        foreign[..4].reverse();
        for off in (4..VIPS_HEADER_LEN).step_by(4) {
            // The header is i32 words except the two i16 words at 44/46,
            // which are zero here, so a 4-byte reverse is safe for this
            // all-zero region too... except it would swap their order.
            // Keep it exact: reverse i32 words, then fix the i16 pair.
            foreign[off..off + 4].reverse();
        }
        // Both i16 words are zero, so no fix-up is needed after the
        // 4-byte reverse of bytes 44..48.
        for off in (VIPS_HEADER_LEN..VIPS_HEADER_LEN + 4).step_by(2) {
            foreign[off..off + 2].reverse();
        }
        // Drop the trailer: its JSON does not participate in byte order,
        // and hand-flipping only covers header + pixels.
        foreign.truncate(VIPS_HEADER_LEN + 4);
        let swapped = decode_bytes(&foreign).unwrap();
        assert_eq!(swapped.data(), im.data());
        assert_eq!(swapped.format(), PixelFormat::Gray16);
    }

    /**
     * Tests the float .v round-trip: a FloatF32 raster encodes with
     * Bbits 32 and BandFmt 6 (FLOAT) and decodes back byte-identical,
     * an RgbaF32 raster canonicalizes back to RgbaF32, and a file in
     * the foreign byte order has its 4-byte float samples swapped.
     */
    #[test]
    fn vips_float_roundtrip_and_foreign_endian() {
        let f1 = PixelFormat::with_channels(1, 4).unwrap();
        let im = Raster::from_f32_samples(2, 1, f1, &[0.5, -3.25]).unwrap();
        let bytes = im.encode_vips().unwrap();
        // Header words: Bbits (offset 16) is 32, BandFmt (offset 20) is 6.
        assert_eq!(i32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 32);
        assert_eq!(i32::from_ne_bytes(bytes[20..24].try_into().unwrap()), 6);
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.format(), f1);
        assert_eq!(back.data(), im.data());
        assert_eq!(back.f32_samples().unwrap(), vec![0.5, -3.25]);
        assert_eq!(back.get_field("format").unwrap().as_str(), "float");

        // Four-band float canonicalizes back to the named RgbaF32.
        let rgba =
            Raster::from_f32_samples(1, 1, PixelFormat::RgbaF32, &[1.0, 2.0, 3.0, 4.5]).unwrap();
        let rgba_back = decode_bytes(&rgba.encode_vips().unwrap()).unwrap();
        assert_eq!(rgba_back.format(), PixelFormat::RgbaF32);
        assert_eq!(rgba_back.f32_samples().unwrap(), vec![1.0, 2.0, 3.0, 4.5]);

        // Flip the file to the foreign byte order by hand: reverse the
        // magic, every i32 header word (the two i16 words at 44/46 are
        // zero, so the 4-byte reverse is exact here), and each 4-byte
        // float sample. The decoder must swap the samples back.
        let mut foreign = bytes.clone();
        foreign[..4].reverse();
        for off in (4..VIPS_HEADER_LEN).step_by(4) {
            foreign[off..off + 4].reverse();
        }
        for off in (VIPS_HEADER_LEN..VIPS_HEADER_LEN + 8).step_by(4) {
            foreign[off..off + 4].reverse();
        }
        // Drop the trailer; hand-flipping only covers header + pixels.
        foreign.truncate(VIPS_HEADER_LEN + 8);
        let swapped = decode_bytes(&foreign).unwrap();
        assert_eq!(swapped.format(), f1);
        assert_eq!(swapped.f32_samples().unwrap(), vec![0.5, -3.25]);
    }

    /**
     * Tests save/save_stripped through the extension dispatcher for .v:
     * save keeps attached fields, save_stripped drops them but keeps
     * pixels and header geometry.
     */
    #[test]
    fn save_v_and_stripped() {
        let dir = tempfile::tempdir().unwrap();
        let mut im = rgb_2x2();
        im.set_icc_profile(&[1, 2, 3]);
        im.set_field("note", "keep me".into());

        let kept = dir.path().join("kept.v");
        im.save(&kept).unwrap();
        let back = decode_file(&kept).unwrap();
        assert_eq!(back.icc_profile(), Some(&[1u8, 2, 3][..]));
        assert_eq!(back.get_field("note").unwrap().as_str(), "keep me");

        let stripped = dir.path().join("stripped.v");
        im.save_stripped(&stripped).unwrap();
        let back = decode_file(&stripped).unwrap();
        assert_eq!(back.data(), im.data());
        assert_eq!(back.icc_profile(), None);
        assert_eq!(back.get_field("note"), None);
        assert_eq!(back.get_field("exif-data"), None);
    }

    /**
     * Tests that `.webp` is a live row in the extension route and that
     * the lossless encoder behind it round-trips: the file written by
     * `save` decodes back to the same pixels, and `save_stripped` drops
     * the metadata chunks the plain `save` embeds. Works by attaching an
     * ICC blob, saving both ways, and reading each file back.
     * Input: 2x2 Rgb8 with `icc-profile-data` -> Output: identical
     * pixels from both files, the profile present after `save` and
     * absent after `save_stripped`.
     */
    #[test]
    fn save_webp_round_trips_losslessly_and_honours_strip() {
        let dir = tempfile::tempdir().unwrap();
        let mut im = rgb_2x2();
        im.fields
            .set("icc-profile-data", MetadataValue::Blob(vec![1, 2, 3, 4]));

        let kept = dir.path().join("kept.webp");
        im.save(&kept).unwrap();
        let back = decode_file(&kept).unwrap();
        assert_eq!(back.data(), im.data(), "the WebP encoder is lossless");
        assert_eq!(back.icc_profile(), Some(&[1u8, 2, 3, 4][..]));

        let stripped = dir.path().join("stripped.webp");
        im.save_stripped(&stripped).unwrap();
        let bare = decode_file(&stripped).unwrap();
        assert_eq!(bare.data(), im.data());
        assert_eq!(bare.icc_profile(), None);
    }

    /**
     * Tests save dispatch to PNG: the file decodes back to the same
     * pixels (PNG is lossless), and unknown extensions error.
     */
    #[test]
    fn save_png_and_unknown_extension() {
        let dir = tempfile::tempdir().unwrap();
        let im = rgb_2x2();
        let png = dir.path().join("out.png");
        im.save(&png).unwrap();
        let back = decode_file(&png).unwrap();
        assert_eq!(back.width(), 2);
        assert_eq!(back.data(), im.data());

        let err = im.save(&dir.path().join("noextension")).unwrap_err();
        assert!(
            matches!(err, SaveError::UnsupportedExtension { .. }),
            "{err}"
        );
    }

    /**
     * Tests JPEG save keeps the ICC profile and EXIF blob while
     * save_stripped drops them: the keep_icc/keep_none contract of the
     * ported infrastructure suite, end to end through the decoder's
     * segment scan.
     */
    #[test]
    fn save_jpeg_keeps_and_strips_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let mut im = Raster::zeroed(8, 8, PixelFormat::Rgb8).unwrap();
        let icc: Vec<u8> = (0u8..=255).cycle().take(400).collect();
        im.set_icc_profile(&icc);
        im.set_field("exif-data", MetadataValue::Blob(vec![0x4D, 0x4D, 0, 42]));

        let kept = dir.path().join("kept.jpg");
        im.save(&kept).unwrap();
        let back = decode_file(&kept).unwrap();
        assert_eq!(back.icc_profile(), Some(icc.as_slice()));
        assert_eq!(
            back.get_field("exif-data"),
            Some(MetadataValue::Blob(vec![0x4D, 0x4D, 0, 42]))
        );

        let stripped = dir.path().join("stripped.jpg");
        im.save_stripped(&stripped).unwrap();
        let back = decode_file(&stripped).unwrap();
        assert_eq!(back.icc_profile(), None);
        assert_eq!(back.get_field("icc-profile-data"), None);
        assert_eq!(back.get_field("exif-data"), None);
    }

    /**
     * Tests decode_file attaches the filename field, like the libvips
     * header's filename slot.
     */
    #[test]
    fn decode_file_sets_filename() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("named.v");
        rgb_2x2().save(&path).unwrap();
        let back = decode_file(&path).unwrap();
        assert_eq!(
            back.get_field("filename").unwrap().as_str(),
            path.display().to_string()
        );
    }

    /**
     * Tests multi-chunk ICC embedding: a profile larger than one APP2
     * segment splits into ordered chunks and reassembles on decode.
     */
    #[test]
    fn jpeg_icc_multichunk_roundtrip() {
        let icc: Vec<u8> = (0..(ICC_CHUNK_MAX + 1234))
            .map(|i| (i % 251) as u8)
            .collect();
        let im = Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap();
        let jpeg = crate::sink::encode_jpeg(&im, 75).unwrap();
        let with_icc = inject_jpeg_metadata(jpeg, None, Some(&icc));
        let (exif, got) = extract_jpeg_metadata(&with_icc);
        assert_eq!(exif, None);
        assert_eq!(got.as_deref(), Some(icc.as_slice()));
        // The stream still decodes as a JPEG.
        assert!(decode_bytes(&with_icc).is_ok());
    }

    /**
     * Tests the .v reader's typed rejections: bad magic, truncated
     * header, truncated pixels, unsupported band format, and coding.
     */
    #[test]
    fn vips_decode_rejects_malformed() {
        let im = rgb_2x2();
        let good = im.encode_vips().unwrap();

        assert!(matches!(
            decode_vips_bytes(&good[..10], DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut bad_magic = good.clone();
        bad_magic[0] = 0;
        assert!(matches!(
            decode_vips_bytes(&bad_magic, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut truncated = good.clone();
        truncated.truncate(VIPS_HEADER_LEN + 2);
        assert!(matches!(
            decode_vips_bytes(&truncated, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut double_fmt = good.clone();
        double_fmt[20..24].copy_from_slice(&8i32.to_ne_bytes()); // DOUBLE
        assert!(matches!(
            decode_vips_bytes(&double_fmt, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        // FLOAT (6) is a supported band format now, but retagging the
        // 8-bit fixture as float quadruples the promised pixel bytes.
        // Use the trailer-free encoding (the JSON metadata trailer would
        // otherwise be long enough to be eaten as pixel data): the
        // shortfall must surface as a truncation error, not a misread.
        let mut float_fmt = im.encode_vips_impl(false);
        float_fmt[20..24].copy_from_slice(&6i32.to_ne_bytes()); // FLOAT
        assert!(matches!(
            decode_vips_bytes(&float_fmt, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));

        let mut coded = good.clone();
        coded[24..28].copy_from_slice(&1i32.to_ne_bytes()); // LABQ coding
        assert!(matches!(
            decode_vips_bytes(&coded, DecodeLimits::default()),
            Err(SourceError::VipsFormat(_))
        ));
    }

    /**
     * Tests a real-libvips-style trailer (XML, not JSON) is tolerated:
     * pixels and header decode, attached fields read as absent.
     */
    #[test]
    fn vips_decode_ignores_foreign_trailer() {
        let im = rgb_2x2();
        let mut bytes = im.encode_vips_impl(false);
        bytes.extend_from_slice(b"<root xmlns=\"http://www.vips.ecs.soton.ac.uk//dsim\"/>");
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.data(), im.data());
        assert_eq!(back.get_field("note"), None);
    }

    /// A `.v` file written by real libvips stores metadata as an XML block
    /// (not our JSON), so its orientation was previously lost and `autorot`
    /// had no cross-oracle. The decoder now recovers the orientation tag from
    /// the `<field type="gint" name="orientation">N</field>` element that vips
    /// writes, matching what `vipsheader -a` reports for the same file.
    #[test]
    fn vips_decode_reads_orientation_from_xml_trailer() {
        let im = rgb_2x2();
        let mut bytes = im.encode_vips_impl(false);
        bytes.extend_from_slice(
            b"<?xml version=\"1.0\"?>\n<root xmlns=\"http://www.vips.ecs.soton.ac.uk/vips/8.18.4\">\n\
              <meta>\n    <field type=\"VipsRefString\" name=\"exif-ifd0-Orientation\">\
              1 (Top-left, Short, 1 components, 2 bytes)</field>\n\
              <field type=\"gint\" name=\"orientation\">6</field>\n  </meta>\n</root>\n",
        );
        let back = decode_bytes(&bytes).unwrap();
        // Pixels and geometry are untouched; only the orientation is recovered.
        assert_eq!(back.data(), im.data());
        assert_eq!(back.orientation(), 6);
        // And autorot of the orientation-6 image rotates 2x2 → 2x2 (rot90) and
        // clears the tag, as `vips autorot` does.
        let rot = back.autorot();
        assert_eq!(rot.orientation(), 1);
    }

    /// Unit-covers the XML orientation extractor: it finds the lowercase
    /// `orientation` gint field, ignores the distinct `exif-ifd0-Orientation`
    /// string field, and rejects malformed / out-of-range values.
    #[test]
    fn parse_vips_xml_orientation_cases() {
        let good = b"<field type=\"gint\" name=\"orientation\">8</field>";
        assert_eq!(parse_vips_xml_orientation(good), Some(8));
        // The exif string field alone must not be mistaken for the tag.
        let exif_only =
            b"<field type=\"VipsRefString\" name=\"exif-ifd0-Orientation\">1 (Top-left)</field>";
        assert_eq!(parse_vips_xml_orientation(exif_only), None);
        // Out of range and non-numeric are rejected.
        assert_eq!(
            parse_vips_xml_orientation(b"<field name=\"orientation\">9</field>"),
            None
        );
        assert_eq!(
            parse_vips_xml_orientation(b"<field name=\"orientation\">x</field>"),
            None
        );
        assert_eq!(parse_vips_xml_orientation(b"no field here"), None);
        // A longer field name that merely starts with "orientation" must not
        // false-match: the anchor requires the closing quote + '>'.
        assert_eq!(
            parse_vips_xml_orientation(b"<field name=\"orientation-foo\">6</field>"),
            None
        );
    }

    // -- .v trailer: nothing to say, nothing written (issue #546) ------------

    /// A raster with no attached fields and the upright orientation has
    /// nothing to put in the trailer, so nothing is written there.
    ///
    /// The 41 bytes it used to write, `{"orientation":1,"fields":{"entries":
    /// []}}`, are the whole of issue #546 for the common case: libvips parses
    /// that slot as XML, so every plain `save()` made `vipsheader -a` print
    /// `VIPS-WARNING **: error reading vips image metadata: VipsImage: XML
    /// parse error` and drop the metadata block. Measured on vips 8.18.4: the
    /// same file with the trailer truncated off reads silently.
    #[test]
    fn v_trailer_is_absent_when_there_is_nothing_to_say() {
        let im = Raster::zeroed(4, 4, PixelFormat::Rgb8).unwrap();
        let bytes = im.encode_vips().unwrap();
        assert_eq!(
            bytes.len(),
            VIPS_HEADER_LEN + im.data().len(),
            "a raster with no metadata must write header + pixels and stop, got \
             {} trailer bytes: {:?}",
            bytes.len() - VIPS_HEADER_LEN - im.data().len(),
            String::from_utf8_lossy(&bytes[VIPS_HEADER_LEN + im.data().len()..])
        );
        // And it still reads back as the same image.
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.data(), im.data());
        assert_eq!(back.orientation(), 1);
        assert_eq!(
            back.get_fields(),
            im.get_fields(),
            "no attached field may appear out of a trailer that was never written"
        );
    }

    /// The other side of the same rule: anything the trailer would actually
    /// carry brings it back. A non-default orientation, an attached field,
    /// and a value carried opaquely from a newer build each count on their
    /// own, because each of them is metadata that would otherwise be lost.
    #[test]
    fn v_trailer_is_written_whenever_it_carries_something() {
        let plain = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
        let body_len = VIPS_HEADER_LEN + plain.data().len();

        let rotated = plain.copy().orientation(6).build();
        let bytes = rotated.encode_vips().unwrap();
        assert!(
            bytes.len() > body_len,
            "a non-default orientation must still be written"
        );
        assert_eq!(decode_bytes(&bytes).unwrap().orientation(), 6);

        let mut noted = plain.clone();
        noted.set_field("note", "hello".into());
        let bytes = noted.encode_vips().unwrap();
        assert!(
            bytes.len() > body_len,
            "an attached field must still be written"
        );
        assert_eq!(
            decode_bytes(&bytes)
                .unwrap()
                .get_field("note")
                .unwrap()
                .as_str(),
            "hello"
        );

        // A value this build cannot interpret is metadata too: it is the only
        // thing left on this raster once the four readable fields are removed
        // and the orientation is back to upright, and it must still be
        // written (issue #565).
        let mut carried = decode_bytes(&file_from_a_newer_build()).unwrap();
        for name in ["note", "n-pages", "xres-hint", "icc-profile-data"] {
            carried.set_typeof(name, 0);
        }
        let carried = carried.copy().orientation(1).build();
        assert!(
            carried.encode_vips().unwrap().len() > VIPS_HEADER_LEN + carried.data().len(),
            "a value carried opaquely from a newer build must still be written"
        );
    }

    // -- .v trailer forward compatibility (issue #565) -----------------------

    /// A `MetadataValue` as a *newer* libviprs writes it: the four variants
    /// this build has, plus one it does not. #573 needs exactly this — a
    /// per-frame delay array for animated GIF — and the whole point of this
    /// lane is that adding it must not cost an older reader the rest of its
    /// metadata, so the future writer is modelled here instead of shipped.
    ///
    /// The derive carries no serde attributes, exactly like [`MetadataValue`],
    /// so the bytes it produces are the bytes a future build would produce.
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    enum FutureMetadataValue {
        Int(i64),
        Double(f64),
        Str(String),
        Blob(Vec<u8>),
        /// The variant this build has never heard of.
        IntArray(Vec<i64>),
    }

    /// The attached-field list as a newer libviprs writes it.
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct FutureFields {
        entries: Vec<(String, FutureMetadataValue)>,
    }

    /// The `.v` trailer as a newer libviprs writes it.
    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct FutureTrailer {
        orientation: u8,
        fields: FutureFields,
    }

    /// The trailer reader as libviprs shipped it *before* this change:
    /// `serde_json::from_slice` straight onto a struct holding the plain
    /// externally tagged [`MetadataValue`]. Keeping it verbatim makes "an
    /// already-released build can still read what we write" an executed
    /// assertion rather than a claim about bytes.
    #[derive(Debug, PartialEq, Deserialize)]
    struct ReleasedTrailer {
        orientation: u8,
        fields: ReleasedFields,
    }

    /// The released reader's field list; see [`ReleasedTrailer`].
    #[derive(Debug, PartialEq, Deserialize)]
    struct ReleasedFields {
        entries: Vec<(String, MetadataValue)>,
    }

    /// The trailer libviprs 0.4 wrote, captured byte for byte from the
    /// released encoder for a 2x2 RGB raster carrying one field of each
    /// variant and orientation 6. Frozen here so the reader is pinned
    /// against the shipped format and not against whatever the current
    /// writer happens to emit.
    const RELEASED_TRAILER: &[u8] = br#"{"orientation":6,"fields":{"entries":[["note",{"Str":"hello"}],["n-pages",{"Int":3}],["xres-hint",{"Double":1.5}],["icc-profile-data",{"Blob":[5,5,5]}]]}}"#;

    /// Header and pixels for a 2x2 RGB raster with no trailer, so a test can
    /// staple an arbitrary trailer onto a real `.v` body.
    fn v_body() -> Vec<u8> {
        Raster::new(2, 2, PixelFormat::Rgb8, vec![7u8; 12])
            .unwrap()
            .encode_vips_impl(false)
    }

    /// A `.v` file as a newer libviprs would write it: a real header and real
    /// pixels, followed by a trailer carrying one field of every variant this
    /// build knows plus a `delay` array it does not.
    fn file_from_a_newer_build() -> Vec<u8> {
        let mut bytes = v_body();
        let trailer = FutureTrailer {
            orientation: 6,
            fields: FutureFields {
                entries: vec![
                    ("note".to_string(), FutureMetadataValue::Str("hello".into())),
                    ("n-pages".to_string(), FutureMetadataValue::Int(3)),
                    (
                        "delay".to_string(),
                        FutureMetadataValue::IntArray(vec![40, 40, 90]),
                    ),
                    ("xres-hint".to_string(), FutureMetadataValue::Double(1.5)),
                    (
                        "icc-profile-data".to_string(),
                        FutureMetadataValue::Blob(vec![5, 5, 5]),
                    ),
                ],
            },
        };
        bytes.extend_from_slice(&serde_json::to_vec(&trailer).unwrap());
        bytes
    }

    /// A `.v` written by a build that has a `MetadataValue` variant this one
    /// does not must still hand back every field this build *does* understand
    /// (issue #565). The old behaviour was that one unknown variant failed the
    /// whole `serde_json::from_slice`, so the ICC profile, the EXIF blob and
    /// the orientation tag all vanished with no error at all.
    ///
    /// The unknown field itself reads as absent: it is carried, not
    /// interpretable, so it stays out of the field API.
    #[test]
    fn v_trailer_from_a_newer_build_keeps_the_fields_this_build_understands() {
        let back = decode_bytes(&file_from_a_newer_build()).unwrap();

        assert_eq!(back.orientation(), 6, "orientation must survive");
        assert_eq!(back.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(back.get_field("n-pages"), Some(MetadataValue::Int(3)));
        assert_eq!(back.get_n_pages(), 3);
        assert_eq!(
            back.get_field("xres-hint"),
            Some(MetadataValue::Double(1.5))
        );
        assert_eq!(back.icc_profile(), Some(&[5u8, 5, 5][..]));
        // Pixels and geometry are untouched by any of this.
        assert_eq!(back.data(), &[7u8; 12]);

        // The variant this build cannot represent reads as absent rather than
        // as a wrong value.
        assert_eq!(back.get_field("delay"), None);
        assert_eq!(back.get_typeof("delay"), 0);
        assert!(
            !back.get_fields().iter().any(|n| n == "delay"),
            "an uninterpretable field must not be advertised as readable"
        );
    }

    /// Preserving an unknown field only matters if it survives being written
    /// back out: an old build that opens a new file and re-saves it must not
    /// quietly strip the field it could not read, which would be the same
    /// data loss one step later.
    #[test]
    fn v_trailer_unknown_field_survives_a_rewrite_by_this_build() {
        let back = decode_bytes(&file_from_a_newer_build()).unwrap();
        let rewritten = back.encode_vips().unwrap();

        // Read the rewritten file with the *newer* build's reader.
        let trailer: FutureTrailer =
            serde_json::from_slice(&rewritten[v_body().len()..]).expect("newer build can re-read");
        assert_eq!(trailer.orientation, 6);
        let delay = trailer
            .fields
            .entries
            .iter()
            .find(|(n, _)| n == "delay")
            .map(|(_, v)| v);
        assert_eq!(
            delay,
            Some(&FutureMetadataValue::IntArray(vec![40, 40, 90])),
            "the unknown field must round-trip untouched"
        );
        // And the fields this build does understand are still there too.
        assert_eq!(trailer.fields.entries.len(), 5);
        assert!(trailer.fields.entries.iter().any(
            |(n, v)| n == "icc-profile-data" && *v == FutureMetadataValue::Blob(vec![5, 5, 5])
        ));
    }

    /// A field the caller sets by hand supersedes an unknown field of the same
    /// name, and removing it removes it for good. Otherwise a stripped raster
    /// would leak the old opaque value back into the file.
    #[test]
    fn setting_or_removing_a_field_supersedes_the_unknown_one() {
        let mut back = decode_bytes(&file_from_a_newer_build()).unwrap();
        back.set_field("delay", MetadataValue::Int(4));
        let rewritten = back.encode_vips().unwrap();
        let trailer: FutureTrailer = serde_json::from_slice(&rewritten[v_body().len()..]).unwrap();
        let delays: Vec<_> = trailer
            .fields
            .entries
            .iter()
            .filter(|(n, _)| n == "delay")
            .collect();
        assert_eq!(delays.len(), 1, "the field must not be written twice");
        assert_eq!(delays[0].1, FutureMetadataValue::Int(4));
        assert_eq!(
            trailer.fields.entries.len(),
            5,
            "overwriting one field must not disturb the other four"
        );

        let mut back = decode_bytes(&file_from_a_newer_build()).unwrap();
        back.set_typeof("delay", 0);
        let rewritten = back.encode_vips().unwrap();
        let trailer: FutureTrailer = serde_json::from_slice(&rewritten[v_body().len()..]).unwrap();
        assert!(
            !trailer.fields.entries.iter().any(|(n, _)| n == "delay"),
            "a removed field must not come back from the opaque carrier"
        );
        assert_eq!(
            trailer.fields.entries.len(),
            4,
            "removing one field must not remove the other four"
        );
    }

    /// The other direction: what this build writes must still parse on a build
    /// that shipped before this change, or the fix would break every reader
    /// already in the wild. The released reader is reproduced verbatim as
    /// [`ReleasedTrailer`], so this runs it rather than asserting about it.
    #[test]
    fn v_trailer_written_now_still_parses_on_the_released_build() {
        let mut im = Raster::new(2, 2, PixelFormat::Rgb8, vec![7u8; 12])
            .unwrap()
            .copy()
            .orientation(6)
            .build();
        im.set_field("note", "hello".into());
        im.set_field("n-pages", MetadataValue::Int(3));
        im.set_field("xres-hint", MetadataValue::Double(1.5));
        im.set_icc_profile(&[5, 5, 5]);

        let bytes = im.encode_vips().unwrap();
        let trailer = &bytes[v_body().len()..];
        let parsed: ReleasedTrailer =
            serde_json::from_slice(trailer).expect("the released reader must still parse this");
        assert_eq!(parsed.orientation, 6);
        assert_eq!(
            parsed.fields.entries,
            vec![
                ("note".to_string(), MetadataValue::Str("hello".into())),
                ("n-pages".to_string(), MetadataValue::Int(3)),
                ("xres-hint".to_string(), MetadataValue::Double(1.5)),
                (
                    "icc-profile-data".to_string(),
                    MetadataValue::Blob(vec![5, 5, 5])
                ),
            ]
        );
        // The wire bytes are unchanged from the released encoder, which is what
        // makes the guarantee above hold for every field, not just these four.
        assert_eq!(trailer, RELEASED_TRAILER);
    }

    /// And a file the released build wrote still reads here, checked against
    /// the frozen bytes rather than against a fresh encode.
    #[test]
    fn v_trailer_written_by_the_released_build_still_reads() {
        let mut bytes = v_body();
        bytes.extend_from_slice(RELEASED_TRAILER);
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.orientation(), 6);
        assert_eq!(back.get_field("note").unwrap().as_str(), "hello");
        assert_eq!(back.get_field("n-pages"), Some(MetadataValue::Int(3)));
        assert_eq!(
            back.get_field("xres-hint"),
            Some(MetadataValue::Double(1.5))
        );
        assert_eq!(back.icc_profile(), Some(&[5u8, 5, 5][..]));
    }

    /// The reader must survive more than a new variant: a newer trailer may
    /// carry keys this build has never seen, drop keys it expects, or hold an
    /// entry shaped in some way it cannot use. None of that may cost the
    /// entries that *are* readable.
    #[test]
    fn v_trailer_tolerates_shapes_a_newer_build_might_write() {
        let mut bytes = v_body();
        bytes.extend_from_slice(
            br#"{"fields":{"entries":[["note",{"Str":"hi"}],["broken"],[7,{"Int":1}],
                 ["n-pages",{"Int":2}]],"grouping":"page"},"trailer-version":9}"#,
        );
        let back = decode_bytes(&bytes).unwrap();
        assert_eq!(back.get_field("note").unwrap().as_str(), "hi");
        assert_eq!(back.get_field("n-pages"), Some(MetadataValue::Int(2)));
        // No orientation key: the upright default, not a dropped trailer.
        assert_eq!(back.orientation(), 1);
    }

    /// A trailer that opens with `{` claims to be a libviprs JSON trailer, so
    /// one that is not valid JSON is corruption rather than a foreign format,
    /// and the metadata is genuinely unrecoverable. That is reported instead
    /// of swallowed. Nothing else changes: a trailer that never claimed to be
    /// ours is still ignored, pixels and header intact.
    #[test]
    fn v_trailer_that_claims_to_be_ours_but_is_corrupt_is_reported() {
        let mut bytes = v_body();
        bytes.extend_from_slice(br#"{"orientation":6,"fields":{"entr"#);
        let err = decode_bytes(&bytes).unwrap_err();
        assert!(
            matches!(&err, SourceError::VipsFormat(m) if m.contains("trailer")),
            "expected a reported trailer failure, got {err}"
        );

        // A foreign or junk trailer is not ours to complain about.
        for foreign in [
            &b"<?xml version=\"1.0\"?><root/>"[..],
            &[0u8; 8][..],
            b"junk",
        ] {
            let mut bytes = v_body();
            bytes.extend_from_slice(foreign);
            let back = decode_bytes(&bytes).unwrap();
            assert_eq!(back.data(), &[7u8; 12]);
            assert_eq!(back.get_field("note"), None);
        }
    }

    /// Pins every [`Interpretation`] to its `VipsInterpretation` code and
    /// nickname, so the mapping cannot drift from
    /// `libvips/include/vips/image.h:96-117` (8.18.4) when a variant lands.
    ///
    /// The table here is the `vips_tag` match, which is exhaustive on
    /// purpose: a new variant does not compile until it is handed both a code
    /// and a nickname right there, and every assertion below is driven off
    /// what it returns rather than off a second copy. `VARIANTS` only supplies
    /// the iteration order; the sweep at the end pins its contents from the
    /// other side, so a variant left out of it is caught as soon as the reader
    /// learns the code. The real guarantee that no variant ships untagged is
    /// still that the production `interpretation_code` and
    /// `interpretation_nickname` have no `_` arm.
    #[test]
    fn interpretation_code_table_matches_vips() {
        // The `VipsInterpretation` code and libvips nickname every variant
        // must carry, straight off `image.h`. Exhaustive: adding a variant
        // to `Interpretation` breaks this match until both are filled in.
        fn vips_tag(i: Interpretation) -> (i32, &'static str) {
            match i {
                Interpretation::Multiband => (0, "multiband"),
                Interpretation::Bw => (1, "b-w"),
                Interpretation::Histogram => (10, "histogram"),
                Interpretation::Xyz => (12, "xyz"),
                Interpretation::Lab => (13, "lab"),
                Interpretation::Cmyk => (15, "cmyk"),
                Interpretation::Labq => (16, "labq"),
                Interpretation::Rgb => (17, "rgb"),
                Interpretation::Cmc => (18, "cmc"),
                Interpretation::Lch => (19, "lch"),
                Interpretation::Labs => (21, "labs"),
                Interpretation::Srgb => (22, "srgb"),
                Interpretation::Yxy => (23, "yxy"),
                Interpretation::Fourier => (24, "fourier"),
                Interpretation::Rgb16 => (25, "rgb16"),
                Interpretation::Grey16 => (26, "grey16"),
                Interpretation::Matrix => (27, "matrix"),
                Interpretation::ScRgb => (28, "scrgb"),
                Interpretation::Hsv => (29, "hsv"),
                Interpretation::OkLab => (30, "oklab"),
                Interpretation::OkLch => (31, "oklch"),
            }
        }

        const VARIANTS: [Interpretation; 21] = [
            Interpretation::Multiband,
            Interpretation::Bw,
            Interpretation::Histogram,
            Interpretation::Xyz,
            Interpretation::Lab,
            Interpretation::Cmyk,
            Interpretation::Labq,
            Interpretation::Rgb,
            Interpretation::Cmc,
            Interpretation::Lch,
            Interpretation::Labs,
            Interpretation::Srgb,
            Interpretation::Yxy,
            Interpretation::Fourier,
            Interpretation::Rgb16,
            Interpretation::Grey16,
            Interpretation::Matrix,
            Interpretation::ScRgb,
            Interpretation::Hsv,
            Interpretation::OkLab,
            Interpretation::OkLch,
        ];

        for interp in VARIANTS {
            let (code, nickname) = vips_tag(interp);
            assert_eq!(
                interpretation_code(interp),
                code,
                "{nickname} must write VipsInterpretation {code}"
            );
            assert_eq!(
                interpretation_from_code(code),
                Some(interp),
                "VipsInterpretation {code} must read back as {nickname}"
            );
            assert_eq!(interpretation_nickname(interp), nickname);
            assert_eq!(interpretation_from_nickname(nickname), Some(interp));
        }

        // Nothing in the language binds `VARIANTS`' length to the enum, so
        // pin it against the reader instead: every code the reader accepts
        // has to be one this test drives, and every code it rejects has to
        // be one this test does not. That covers VIPS_INTERPRETATION_ERROR
        // (-1), which vips does emit, the codes libvips leaves unassigned
        // inside its range (2, 11, 14, 20), and everything past
        // VIPS_INTERPRETATION_LAST (32). The read-only legacy aliases 1000 /
        // 1001 sit outside this range on purpose and are pinned by
        // `legacy_private_oklab_codes_read_but_are_not_written`.
        for code in -1..=64 {
            let driven = VARIANTS.iter().any(|&v| vips_tag(v).0 == code);
            assert_eq!(
                interpretation_from_code(code).is_some(),
                driven,
                "VipsInterpretation {code}: the reader and this table disagree"
            );
        }
    }

    /// A `.v` written by real vips tags OkLab/OkLch with `Type = 30` / `31`
    /// (`VIPS_INTERPRETATION_OKLAB` / `_OKLCH`,
    /// `libvips/include/vips/image.h:115-116`), so libviprs has to read those
    /// files back as [`Interpretation::OkLab`] / [`Interpretation::OkLch`]
    /// instead of falling through to format inference and reporting
    /// `Multiband`.
    ///
    /// The fixture is the byte-for-byte 64-byte header vips 8.18.4 wrote for
    ///
    /// ```text
    /// vips black t.v 4 4 --bands 3
    /// vips colourspace t.v ok.v oklab      # and again for oklch
    /// ```
    ///
    /// followed by that file's 4x4x3 float pixels (all zero, it is black), so
    /// nothing binary is checked in. vips writes the header in the machine's
    /// own byte order; these are the little-endian bytes, which on a
    /// big-endian host also exercises the decoder's swap path.
    #[test]
    fn vips_written_oklab_and_oklch_read_back_tagged() {
        #[rustfmt::skip]
        const VIPS_OKLAB_HEADER: [u8; VIPS_HEADER_LEN] = [
            0xb6, 0xa6, 0xf2, 0x08, // magic (little-endian)
            0x04, 0x00, 0x00, 0x00, // Xsize 4
            0x04, 0x00, 0x00, 0x00, // Ysize 4
            0x03, 0x00, 0x00, 0x00, // Bands 3
            0x20, 0x00, 0x00, 0x00, // Bbits 32
            0x06, 0x00, 0x00, 0x00, // BandFmt 6 (float)
            0x00, 0x00, 0x00, 0x00, // Coding 0 (none)
            0x1e, 0x00, 0x00, 0x00, // Type 30 (VIPS_INTERPRETATION_OKLAB)
            0x00, 0x00, 0x80, 0x3f, // Xres 1.0
            0x00, 0x00, 0x80, 0x3f, // Yres 1.0
            0x00, 0x00, 0x00, 0x00, // Length (deprecated)
            0x00, 0x00, 0x00, 0x00, // Compression + Level (deprecated)
            0x00, 0x00, 0x00, 0x00, // Xoffset 0
            0x00, 0x00, 0x00, 0x00, // Yoffset 0
            0x00, 0x00, 0x00, 0x00, // reserved
            0x00, 0x00, 0x00, 0x00, // reserved
        ];
        // The oklch file vips wrote differs from the oklab one in exactly one
        // byte: the Type word at offset 28, 30 -> 31.
        const TYPE_OFFSET: usize = 28;

        for (code, expected) in [(30u8, Interpretation::OkLab), (31, Interpretation::OkLch)] {
            let mut bytes = VIPS_OKLAB_HEADER.to_vec();
            bytes[TYPE_OFFSET] = code;
            // 4x4 pixels, 3 float bands, all zero: `vips black` output.
            bytes.resize(VIPS_HEADER_LEN + 4 * 4 * 3 * 4, 0);

            let back = decode_bytes(&bytes).unwrap();
            assert_eq!(
                back.interpretation(),
                expected,
                "Type {code} must tag {expected:?}"
            );
            assert_eq!(
                back.get_field("interpretation").unwrap().as_str(),
                interpretation_nickname(expected)
            );
            assert_eq!((back.width(), back.height()), (4, 4));
            assert_eq!(back.format(), PixelFormat::with_channels(3, 4).unwrap());
        }
    }

    /// libviprs used to write private codes 1000 / 1001 for OkLab / OkLch,
    /// before libvips 8.18 assigned them 30 / 31. The private codes stay
    /// readable so `.v` files libviprs already wrote keep loading, but the
    /// encoder must never emit them again.
    #[test]
    fn legacy_private_oklab_codes_read_but_are_not_written() {
        // The header `Type` word, the same offset the vips fixture patches.
        const TYPE_OFFSET: usize = 28;

        assert_eq!(interpretation_from_code(1000), Some(Interpretation::OkLab));
        assert_eq!(interpretation_from_code(1001), Some(Interpretation::OkLch));

        for (interp, code) in [(Interpretation::OkLab, 30i32), (Interpretation::OkLch, 31)] {
            let im = Raster::black(2, 2).copy().interpretation(interp).build();
            let bytes = im.encode_vips().unwrap();
            let written =
                i32::from_ne_bytes(bytes[TYPE_OFFSET..TYPE_OFFSET + 4].try_into().unwrap());
            assert_eq!(written, code, "encode_vips must write the libvips code");
            // And it round-trips through our own reader.
            assert_eq!(decode_bytes(&bytes).unwrap().interpretation(), interp);
        }

        // A legacy libviprs file (private code in the Type word) still loads.
        let mut legacy = Raster::black(2, 2)
            .copy()
            .interpretation(Interpretation::OkLab)
            .build()
            .encode_vips()
            .unwrap();
        legacy[TYPE_OFFSET..TYPE_OFFSET + 4].copy_from_slice(&1000i32.to_ne_bytes());
        assert_eq!(
            decode_bytes(&legacy).unwrap().interpretation(),
            Interpretation::OkLab
        );

        // The one above is written in the host's byte order, which is the
        // easy half. A migrating file is a fixed byte pattern on somebody
        // else's disk, and 1000 / 1001 have a low byte of 0xe8 / 0xe9, so
        // unlike the codes 30 / 31 they do not survive a byte-order swap by
        // accident. This is the explicit little-endian header, laid out like
        // the vips fixture above, which puts the decoder's swap path under
        // test on a big-endian host.
        #[rustfmt::skip]
        const LEGACY_LE_HEADER: [u8; VIPS_HEADER_LEN] = [
            0xb6, 0xa6, 0xf2, 0x08, // magic (little-endian)
            0x02, 0x00, 0x00, 0x00, // Xsize 2
            0x02, 0x00, 0x00, 0x00, // Ysize 2
            0x03, 0x00, 0x00, 0x00, // Bands 3
            0x20, 0x00, 0x00, 0x00, // Bbits 32
            0x06, 0x00, 0x00, 0x00, // BandFmt 6 (float)
            0x00, 0x00, 0x00, 0x00, // Coding 0 (none)
            0xe8, 0x03, 0x00, 0x00, // Type 1000 (legacy libviprs OkLab)
            0x00, 0x00, 0x80, 0x3f, // Xres 1.0
            0x00, 0x00, 0x80, 0x3f, // Yres 1.0
            0x00, 0x00, 0x00, 0x00, // Length (deprecated)
            0x00, 0x00, 0x00, 0x00, // Compression + Level (deprecated)
            0x00, 0x00, 0x00, 0x00, // Xoffset 0
            0x00, 0x00, 0x00, 0x00, // Yoffset 0
            0x00, 0x00, 0x00, 0x00, // reserved
            0x00, 0x00, 0x00, 0x00, // reserved
        ];

        for (legacy_code, expected) in [
            (1000i32, Interpretation::OkLab),
            (1001, Interpretation::OkLch),
        ] {
            let mut bytes = LEGACY_LE_HEADER.to_vec();
            bytes[TYPE_OFFSET..TYPE_OFFSET + 4].copy_from_slice(&legacy_code.to_le_bytes());
            // 2x2 pixels, 3 float bands, all zero.
            bytes.resize(VIPS_HEADER_LEN + 2 * 2 * 3 * 4, 0);

            assert_eq!(
                decode_bytes(&bytes).unwrap().interpretation(),
                expected,
                "little-endian legacy Type {legacy_code} must still read as {expected:?}"
            );
        }
    }

    // -- free functions -----------------------------------------------------

    /**
     * Tests the libvips tokeniser cases from the ported infrastructure
     * suite plus escape handling.
     */
    #[test]
    fn tokenize_quoting_and_escapes() {
        assert_eq!(tokenize("hello world"), vec!["hello", "world"]);
        assert_eq!(tokenize("\"hello world\""), vec!["hello world"]);
        assert_eq!(tokenize("a \"b c\" d"), vec!["a", "b c", "d"]);
        assert_eq!(tokenize("hello\\ world"), vec!["hello world"]);
        assert_eq!(tokenize("'single quoted'"), vec!["single quoted"]);
        assert_eq!(tokenize("\"esc \\\" quote\""), vec!["esc \" quote"]);
        assert_eq!(tokenize("   "), Vec::<String>::new());
        assert_eq!(tokenize(""), Vec::<String>::new());
        // Unterminated quote: the rest of the string is one token.
        assert_eq!(tokenize("\"open ended"), vec!["open ended"]);
    }

    /**
     * Tests the thumbnail geometry parser forms: W, WxH, Wx, xH,
     * modifiers, and garbage.
     */
    #[test]
    fn thumbnail_geometry_forms() {
        assert_eq!(
            parse_thumbnail_geometry("200"),
            ThumbnailGeometry {
                width: Some(200),
                height: None
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("200x150"),
            ThumbnailGeometry {
                width: Some(200),
                height: Some(150)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("200x"),
            ThumbnailGeometry {
                width: Some(200),
                height: None
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("x150"),
            ThumbnailGeometry {
                width: None,
                height: Some(150)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry(" 200x150> "),
            ThumbnailGeometry {
                width: Some(200),
                height: Some(150)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("128X128!"),
            ThumbnailGeometry {
                width: Some(128),
                height: Some(128)
            }
        );
        assert_eq!(
            parse_thumbnail_geometry("banana"),
            ThumbnailGeometry::default()
        );
    }

    /// The `.v` reader enforces the single-axis ceiling from the per-decode
    /// [`DecodeLimits::max_coord`] field: an over-ceiling dimension returns
    /// the dedicated typed [`SourceError::CoordLimitExceeded`] (never a
    /// panic or a wrapping cast/overflow), while an in-bounds decode is
    /// unaffected. This replaces the former process-global enforcement
    /// (#293).
    #[test]
    fn max_coord_enforced_per_decode() {
        let im = Raster::zeroed(2000, 1, PixelFormat::Gray8).unwrap();
        let bytes = im.encode_vips().unwrap();

        // (a) Over the per-decode ceiling → dedicated typed error carrying
        // the offending dimensions and ceiling, not a panic or an opaque
        // substring-matched string. Built through the `#[non_exhaustive]`
        // builder setter, the supported external construction path.
        let tight = DecodeLimits::default().with_max_coord(1000);
        let err = decode_vips_bytes(&bytes, tight).unwrap_err();
        assert!(
            matches!(
                err,
                SourceError::CoordLimitExceeded {
                    width: 2000,
                    height: 1,
                    max_coord: 1000,
                }
            ),
            "expected a typed CoordLimitExceeded error, got {err}"
        );

        // Boundary: exactly at the ceiling is accepted; one below rejects.
        let at = DecodeLimits::default().with_max_coord(2000);
        assert!(decode_vips_bytes(&bytes, at).is_ok());
        let below = DecodeLimits::default().with_max_coord(1999);
        assert!(matches!(
            decode_vips_bytes(&bytes, below),
            Err(SourceError::CoordLimitExceeded { .. })
        ));

        // (b) The default in-bounds decode is unaffected and round-trips.
        let decoded = decode_vips_bytes(&bytes, DecodeLimits::default()).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (2000, 1));

        // (c) Security-critical ordering (#422): a malicious header claiming
        // a width far above the ceiling with a body far shorter than the
        // promised pixel data must reject on the coordinate ceiling *before*
        // the truncation/allocation check — proving the ceiling is enforced
        // ahead of reading or allocating pixels, not after. A 4x4 Gray8 `.v`
        // whose width field is overwritten with 20,000,000 (past the default
        // 10,000,000 ceiling) and whose body is truncated to a few bytes must
        // surface CoordLimitExceeded, never the truncated-pixel-data
        // VipsFormat error and never an oversized allocation or wrapping cast.
        let small = Raster::zeroed(4, 4, PixelFormat::Gray8).unwrap();
        let mut malicious = small.encode_vips().unwrap();
        // Width lives at header offset 4 (native-endian i32); see
        // `encode_vips_impl`. A 20,000,000-px width promises 80 MB of body.
        let huge: i32 = 20_000_000;
        malicious[4..8].copy_from_slice(&huge.to_ne_bytes());
        // Truncate so only a handful of the promised body bytes are present.
        malicious.truncate(VIPS_HEADER_LEN + 8);
        let err = decode_vips_bytes(&malicious, DecodeLimits::default()).unwrap_err();
        assert!(
            matches!(
                err,
                SourceError::CoordLimitExceeded {
                    width: 20_000_000,
                    height: 4,
                    max_coord: DEFAULT_MAX_COORD,
                }
            ),
            "expected CoordLimitExceeded before the truncation check, got {err}"
        );
    }
}
