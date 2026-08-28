//! Analyze 7.5 (`.hdr` + `.img`) load: a 348-byte big-endian header in one
//! file and a raw array in another.
//!
//! Analyze is the only format in this crate that is inherently a **pair**.
//! The geometry is in a 348-byte `.hdr` and the pixels are in a sibling
//! `.img`, and neither half is decodable on its own: a `.hdr` has a
//! geometry and no pixels, and an `.img` is a raw array with no signature
//! and no shape. That is why this module has three entry points rather than
//! the one every other codec here has, and why [`crate::source`] grew a
//! route kind for it; see [the entry point](#the-entry-point) below.
//!
//! # Operations
//!
//! | libviprs method | reference equivalent | result |
//! |---|---|---|
//! | [`decode_analyze_file`] | `vips analyzeload fred.hdr` | [`PixelFormat::Gray8`], `FloatF32(1)` or [`PixelFormat::Rgb8`] |
//! | [`decode_analyze`] | the same, from two buffers | as above |
//! | `decode_analyze_header`, which only the route table calls | *(no equivalent)* | always a refusal naming the sibling |
//!
//! There is no save. `vips` registers no `analyzesave`, so the format is
//! load-only here and there is no round trip to pin.
//!
//! # The oracle
//!
//! `oracle-captures/foreign-analyze/`, 82 fixtures and 9 records, measured
//! against `/opt/homebrew/bin/vips` 8.18.6 (`analyzeload`, `enable Analyze7
//! load: true`), recorded `on_pin` in `ORACLE_PIN.json`.
//!
//! libvips names this format three different ways and they do not agree: the
//! operator says Analyze6, the gtkdoc says 6.0, the implementation comment
//! says "old-style header (so called 7.5 format)" and `--vips-config` says
//! Analyze7. Nothing in the loader distinguishes a 6.0 file from a 7.5 one,
//! because the 348-byte header has no version field to distinguish them
//! with, so the version in the name is decoration and this module does not
//! act on it either.
//!
//! # The entry point
//!
//! `libviprs`'s decode seam is one buffer in, and a pair cannot be expressed
//! through it. The decision, written up in issue #764 and taken there rather
//! than guessed at here:
//!
//! * **A path-taking route kind.** [`crate::source`]'s route table grows a
//!   `Paired` decoder, which the file entry point calls with the path and
//!   the buffer entry point calls with the header bytes alone. Analyze is
//!   the only row that uses it and every other row ignores it.
//! * The `.hdr` **is** sniffable, because `sizeof_hdr` is the first four
//!   bytes and is always big-endian 348, so `decode_file` finds the
//!   container from its content the way it finds every other one, and then
//!   resolves the `.img` from the path the way `generate_filenames` does.
//! * `decode_bytes` on a `.hdr` validates the header in full and then
//!   refuses with [`AnalyzeError::PixelsAreInASiblingFile`], which is a
//!   different and better answer than "these bytes are not an image".
//!
//! The two alternatives are both worse and both were considered. A
//! path-only entry point with **no** sniff row leaves `decode_file` unable
//! to load an Analyze image at all, which is the format's whole normal use.
//! A sniff row whose decoder *always* refuses keeps the honest message and
//! still cannot load one. Only the route kind gets a real image out of
//! `decode_file`, which is what a loader is for, and it costs one variant of
//! a private enum.
//!
//! One divergence falls out of it, and it is unavoidable rather than chosen:
//! `vips` loads `fred.img` as well as `fred.hdr`, because its `is_a` rewrites
//! whatever name it is handed and parses the `.hdr` either way. A content
//! sniff has nothing to look at in a raw `.img`, so `decode_file` reaches
//! this module only from the `.hdr`. [`decode_analyze_file`] itself takes
//! any of the three names, so the public API has the parity and only the
//! sniffing entry point does not.
//!
//! # Semantics, all measured
//!
//! * **Priority: last.** `analyzeload` is registered at -50, the lowest of
//!   any loader in the build, because its `is_a` opens and fully parses a
//!   second file. `crate::source`'s sniff table declares Analyze last for
//!   the same reason, so nothing it might also match is stolen from a loader
//!   that should win.
//! * **`is_a` is not a magic test.** It validates the `.hdr`'s length
//!   (exactly 348 bytes), the `sizeof_hdr` field (348), the rank (2..=7)
//!   **and** the datatype. A four-byte content sniff cannot reproduce the
//!   last three, so libviprs's sniff is **wider** than vips's; see [the
//!   divergences](#deliberate-divergences).
//! * **Big-endian, always, with no flag and no escape hatch.** `read_header`
//!   byte-swaps every field when the host is not big-endian, and
//!   `vips__analyze_read` swaps the pixels under the same condition. A
//!   little-endian `.hdr` is refused, because its `sizeof_hdr` reads back as
//!   0x5C010000, and little-endian pixels are silently read as big-endian
//!   garbage. This is the single most likely thing for a port on a
//!   little-endian host to get backwards, and both halves are pinned below.
//! * **Rank flattens into the height.** `dim[0]` is the rank and must be
//!   2..=7; the width is `dim[1]` and the height is `dim[2]` multiplied by
//!   every `dim[i]` up to the rank. Nothing records that the image was ever
//!   3-D beyond the `dsr-image_dimension.dim[]` metadata. [`crate::nifti`]
//!   follows this same rule for the sibling format and cites this capture
//!   for it.
//! * **`vox_offset` is parsed, attached and then ignored.** The pixels are
//!   read from byte 0 of the `.img` with a hard-coded offset
//!   (`analyze2vips.c:582-583`). A port that honoured it would disagree with
//!   vips on every file that sets it.
//! * **`bitpix` is attached and never consulted.** The datatype alone fixes
//!   the sample width.
//! * **`DT_RGB` is interleaved, not planar.** It is the only datatype giving
//!   more than one band, and the Analyze spec allows either layout while
//!   vips reads only the interleaved one.
//! * **A short `.img` is an error and a long one is not.** A trailing tail
//!   is accepted and ignored; two bytes missing is `file too short`.
//! * **63 `dsr-<section>.<member>` fields and a 348-byte `dsr` blob**, with
//!   two traps in the `getstr` sanitiser, both reproduced here.
//!
//! # Deliberate divergences
//!
//! * **Zero and negative dimensions are refused, not clamped.** `dim[]` is a
//!   signed short and nothing in `get_vips_properties` range-checks it, so
//!   the value reaches `vips_image_init_fields`, GObject's property range
//!   check rejects it, prints a `GLib-GObject-CRITICAL`, **leaves the
//!   property at its default of 1**, and the load carries on and exits 0
//!   with a silently wrong geometry. `matload` has the same defect and this
//!   crate refuses in both.
//! * **The sniff is wider than `is_a`.** libviprs claims any file whose
//!   first four bytes are big-endian 348 and then refuses it by name; vips
//!   parses the whole header in `is_a` and lets a file that fails fall
//!   through to `magickload`. The set of files that **load** is the same
//!   either way, because libviprs has no `magickload` to fall through to and
//!   every one of those files is refused by both. What differs is which
//!   loader reports and what it says, and libviprs's message is the useful
//!   one. `every_measured_fixture_loads_exactly_where_vips_loads_it` is what
//!   holds that claim.
//! * **The carrier ceiling.** Nine datatypes load in vips and libviprs has
//!   carriers for three. See [`AnalyzeError::UnsupportedCarrier`].
//!
//! # Decode limits
//!
//! `dims_32767.hdr` declares a 1.07-gigapixel image in front of a six-byte
//! `.img` and the header load succeeds, so the declared geometry is priced
//! through `DecodeLimits::check_coord`, `check_pixels` and
//! `check_image_alloc` **before** the `.img` is opened at
//! all. That ordering is the point: a header that prices past the budget
//! costs no read.
//!
//! No new dependency. The header is a fixed-offset struct and the `.img` is
//! a raw array.

use std::num::NonZeroU16;
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::conversion::Interpretation;
use crate::imageio::MetadataValue;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use crate::source::{DecodeLimits, SourceError, read_file_bounded};

/// Bytes in the Analyze `struct dsr`, and the value its `sizeof_hdr` field
/// carries. The `.hdr` file has to be exactly this long.
pub const HEADER_BYTES: usize = 348;

/// `sizeof_hdr` as every Analyze `.hdr` writes it: 348, big-endian.
///
/// This is the whole of libviprs's content sniff for the format, and it is
/// also `vips__isanalyze`'s second check, so a file that fails it is refused
/// on both sides.
pub(crate) const SIZEOF_HDR_BE: &[u8; 4] = &[0x00, 0x00, 0x01, 0x5c];

/// The lowest rank `dim[0]` may declare. Measured: rank 1 is refused with
/// `1-dimensional images not supported`, even though a 1-D array is a
/// perfectly good image.
pub const MIN_RANK: i16 = 2;

/// The highest rank `dim[0]` may declare. Measured: 7 loads and 8 is
/// refused, which is just as well because `dim[]` has only eight slots.
pub const MAX_RANK: i16 = 7;

/// Errors from the Analyze loader.
///
/// The four vips itself has all live in `analyze2vips.c` and all four are
/// here by name: `header file size incorrect`, `header size incorrect`,
/// `%d-dimensional images not supported` and `datatype %d not supported`.
/// The rest are libviprs's, and each one is a place this module deliberately
/// refuses where vips carries on.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnalyzeError {
    /// The `.hdr` file is not exactly [`HEADER_BYTES`] long.
    ///
    /// Measured at 0, 347 and 349 bytes, all `header file size incorrect`.
    /// This is a *length* check on the file, not on the field: the field
    /// gets [`AnalyzeError::BadHeaderSizeField`].
    #[error("analyze: the header file is {found} bytes, and it has to be exactly {needed}")]
    HeaderFileSize {
        /// Bytes in the `.hdr`.
        found: usize,
        /// Bytes the `struct dsr` takes.
        needed: usize,
    },
    /// The `sizeof_hdr` field does not read back as [`HEADER_BYTES`].
    ///
    /// Measured at 0 and 200, and on a little-endian header, whose 348 reads
    /// back as 1543569408. There is no byte-order flag anywhere in the
    /// format, so this *is* the byte-order check.
    #[error(
        "analyze: sizeof_hdr reads {found} rather than {expected}; the header is \
         big-endian and there is no flag that says otherwise"
    )]
    BadHeaderSizeField {
        /// The field as read big-endian.
        found: i32,
        /// The only value it may take.
        expected: i32,
    },
    /// `dim[0]`, the rank, is outside `2..=7`.
    ///
    /// Measured at 0, 1 and 8, all `%d-dimensional images not supported`.
    #[error(
        "analyze: {found}-dimensional images are not supported; the rank has to be {min}..={max}"
    )]
    UnsupportedRank {
        /// The rank as declared.
        found: i16,
        /// The lowest rank the loader takes.
        min: i16,
        /// The highest.
        max: i16,
    },
    /// A datatype code the reference refuses too.
    ///
    /// `get_vips_properties` is a closed switch and everything outside it is
    /// `datatype %d not supported`. Measured for 0, 1 (`DT_BINARY`, which is
    /// in `dbh.h` and is not implemented), 256 and 511.
    #[error("analyze: datatype {datatype} is not supported")]
    UnsupportedDatatype {
        /// The code as declared.
        datatype: i16,
    },
    /// A datatype the reference loads but libviprs has no sample carrier
    /// for.
    ///
    /// The same ceiling [`crate::fits::FitsError::UnsupportedCarrier`] and
    /// [`crate::nifti::NiftiError::UnsupportedCarrier`] describe, reached
    /// from a fourth table:
    ///
    /// * `DT_SIGNED_SHORT` (4) and `DT_SIGNED_INT` (8) are signed integers,
    ///   which need issue #516. `DT_SIGNED_SHORT` is the datatype most real
    ///   Analyze volumes use, so this is the refusal a caller meets first.
    /// * `DT_DOUBLE` (64) needs the `f64` carrier, which is issue #518.
    /// * `DT_COMPLEX` (32) has no carrier and **no issue**: nothing else in
    ///   this crate wants a complex sample either, and vips itself only
    ///   half-supports it (`vips getpoint` prints the real part and drops
    ///   the imaginary one).
    ///
    /// Refused by name rather than narrowed, because narrowing a signed
    /// 16-bit array into eight bits would lose data silently.
    #[error(
        "analyze: datatype {datatype} ({name}) carries {sample} samples, which libviprs \
         has no pixel format for yet"
    )]
    UnsupportedCarrier {
        /// The code as declared.
        datatype: i16,
        /// The `dbh.h` name for it.
        name: &'static str,
        /// The sample kind it needs, in words.
        sample: &'static str,
    },
    /// A declared dimension that is zero or negative.
    ///
    /// A deliberate divergence: vips clamps it to 1 through GObject's
    /// property range check and exits 0 with a silently wrong geometry. See
    /// the module doc.
    #[error("analyze: dim[{axis}] is {found}, and every declared dimension has to be positive")]
    NonPositiveDimension {
        /// Which dimension.
        axis: usize,
        /// The value as declared.
        found: i16,
    },
    /// The declared extents multiply out past what a raster coordinate can
    /// hold.
    ///
    /// Arithmetic rather than policy: a rank-7 header can declare a product
    /// no `u32` coordinate holds. [`DecodeLimits`] is what refuses
    /// everything that *does* fit.
    #[error(
        "analyze: dim[2..={rank}] multiply out past the largest raster coordinate, so the \
         geometry cannot be represented at all"
    )]
    DimensionOverflow {
        /// The rank whose extents were multiplied.
        rank: i16,
    },
    /// The `.img` ends before the declared array does.
    ///
    /// Measured: `file too short`, raised when the pipeline opens the image
    /// rather than at header time, so vips reports a perfectly good header
    /// for a file it cannot read. libviprs has one entry point, so the
    /// refusal is the whole answer. A *longer* `.img` is accepted and its
    /// tail ignored, which is also measured.
    #[error("analyze: the image file needs {needed} bytes, found {found}")]
    TruncatedImage {
        /// Bytes in the `.img`.
        found: u64,
        /// Bytes the declared geometry needs.
        needed: u64,
    },
    /// These bytes are the header half of a pair, so the pixels are in a
    /// sibling file this buffer does not contain.
    ///
    /// What [`crate::source::decode_bytes`] reports for a `.hdr`. The header
    /// is validated in full first, so a malformed one still reports the
    /// malformation and only a *loadable* header reaches this.
    #[error(
        "analyze: this is the 348-byte header half of an Analyze pair, so the {width}x{height} \
         image's pixels are in a sibling .img; use decode_analyze_file or decode_analyze"
    )]
    PixelsAreInASiblingFile {
        /// The width the header declares.
        width: u32,
        /// The height it declares.
        height: u32,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

// ---------------------------------------------------------------------------
// The carrier table
// ---------------------------------------------------------------------------

/// The sample carrier a datatype lands on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    /// `DT_UNSIGNED_CHAR`, one unsigned byte per sample, one band.
    U8,
    /// `DT_FLOAT`, one 32-bit float per sample, one band.
    F32,
    /// `DT_RGB`, three interleaved bytes per pixel.
    Rgb24,
}

impl Carrier {
    /// Bands one pixel becomes.
    const fn bands(self) -> u64 {
        match self {
            Self::U8 | Self::F32 => 1,
            Self::Rgb24 => 3,
        }
    }

    /// Bytes in one band sample.
    const fn sample_bytes(self) -> u64 {
        match self {
            Self::U8 | Self::Rgb24 => 1,
            Self::F32 => 4,
        }
    }

    /// The [`PixelFormat`] a raster of this carrier gets.
    ///
    /// Written out per variant rather than reached through
    /// [`PixelFormat::with_channels`], for the reason [`crate::nifti`] gives:
    /// that constructor is fallible and all three of these are total, so
    /// routing them through it would put an unreachable error arm in the
    /// decode path. `the_carrier_table_agrees_with_pixel_format` checks the
    /// two against each other instead.
    const fn pixel_format(self) -> PixelFormat {
        match self {
            Self::U8 => PixelFormat::Gray8,
            Self::F32 => PixelFormat::FloatF32(NonZeroU16::MIN),
            Self::Rgb24 => PixelFormat::Rgb8,
        }
    }

    /// Every carrier, for the tests that sweep the table.
    #[cfg(test)]
    const ALL: [Self; 3] = [Self::U8, Self::F32, Self::Rgb24];

    /// The interpretation tag a raster of this carrier gets.
    ///
    /// `analyze2vips.c:544-546` is one band to `b-w` and anything else to
    /// sRGB, with no 16-bit greyscale or RGB16 tag anywhere. Note that this
    /// is the *other* answer from `matload`'s for the same shape: a one-band
    /// 8-bit array is `b-w` here and `multiband` there.
    const fn interpretation(self) -> Interpretation {
        match self {
            Self::U8 | Self::F32 => Interpretation::Bw,
            Self::Rgb24 => Interpretation::Srgb,
        }
    }
}

/// Resolve a datatype code onto a carrier, or say precisely why not.
///
/// The table is `get_vips_properties` (`analyze2vips.c:385-425`), which is a
/// closed switch: everything outside it is `datatype %d not supported`, and
/// that includes `DT_BINARY` (1), which `dbh.h` names and the loader does
/// not implement.
fn carrier_for(datatype: i16) -> Result<Carrier, AnalyzeError> {
    let ceiling = |name, sample| {
        Err(AnalyzeError::UnsupportedCarrier {
            datatype,
            name,
            sample,
        })
    };
    match datatype {
        2 => Ok(Carrier::U8),
        16 => Ok(Carrier::F32),
        128 => Ok(Carrier::Rgb24),
        4 => ceiling("DT_SIGNED_SHORT", "signed 16-bit integer"),
        8 => ceiling("DT_SIGNED_INT", "signed 32-bit integer"),
        32 => ceiling("DT_COMPLEX", "pairs of 32-bit floats"),
        64 => ceiling("DT_DOUBLE", "64-bit float"),
        other => Err(AnalyzeError::UnsupportedDatatype { datatype: other }),
    }
}

// ---------------------------------------------------------------------------
// The 348-byte header
// ---------------------------------------------------------------------------

/// Read `N` big-endian bytes at `at`.
///
/// Every multi-byte field in an Analyze header is big-endian, always. There
/// is no flag, no sniff and no escape hatch, which is why this takes no
/// byte-order argument: a byte-order parameter here would be a place for a
/// port to get it wrong.
fn take<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    bytes[at..at + N]
        .try_into()
        .expect("caller checked the header is 348 bytes")
}

fn be_i16(bytes: &[u8], at: usize) -> i16 {
    i16::from_be_bytes(take(bytes, at))
}

fn be_i32(bytes: &[u8], at: usize) -> i32 {
    i32::from_be_bytes(take(bytes, at))
}

fn be_f32(bytes: &[u8], at: usize) -> f32 {
    f32::from_be_bytes(take(bytes, at))
}

/// A fixed-width character field, as `getstr` (`analyze2vips.c:237-256`)
/// renders it.
///
/// Two traps, both measured on `meta_strings.hdr`:
///
/// * `g_strlcpy` is given the **field** length as its buffer size, so it
///   copies at most `len - 1` characters and an 80-byte `descrip` loses its
///   last byte. Measured: 80 `A`..`B` characters in, 79 out.
/// * every byte failing `isascii(c) && c >= 32` is rewritten to `@`, which
///   is lossy and not reversible. Measured: `0x01` and `0x02` become `@`,
///   both bytes of a UTF-8 e-acute become `@`, and `0x7f` DEL passes through
///   **untouched**, because it is ascii and it is above 32.
///
/// The capture's own prose states that predicate with an `||` rather than an
/// `&&`; its measured data requires the `&&`, because `0x01` is `isascii`
/// and comes back as `@`. That is issue #797.
fn getstr(field: &[u8]) -> String {
    let usable = &field[..field.len() - 1];
    let end = usable.iter().position(|&b| b == 0).unwrap_or(usable.len());
    usable[..end]
        .iter()
        .map(|&c| {
            if c.is_ascii() && c >= 32 {
                c as char
            } else {
                '@'
            }
        })
        .collect()
}

/// What one row of the metadata table says about a header field.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// A `short int`.
    I16,
    /// An `int`.
    I32,
    /// A `float`.
    F32,
    /// A single `char`, reported as a number.
    Char,
    /// A `char[n]`, reported through [`getstr`].
    Text(usize),
}

/// Every field `attach_meta` (`analyze2vips.c:437-482`) sets, in the order
/// `vipsheader -a` prints them.
///
/// The count is 63, which is measured off `vipsheader` rather than counted
/// off the C table, and `the_metadata_is_the_measured_sixty_three_fields`
/// holds it. Note that `vox_units` and `cal_units` are `char` arrays
/// reported **per element as numbers**, not as strings, and that
/// `image_dimension` has three `funused` floats and an `unused1` short that
/// the table skips.
const FIELDS: &[(&str, usize, Kind, usize)] = &[
    ("header_key.sizeof_hdr", 0, Kind::I32, 1),
    ("header_key.data_type", 4, Kind::Text(10), 1),
    ("header_key.db_name", 14, Kind::Text(18), 1),
    ("header_key.extents", 32, Kind::I32, 1),
    ("header_key.session_error", 36, Kind::I16, 1),
    ("header_key.regular", 38, Kind::Char, 1),
    ("header_key.hkey_un0", 39, Kind::Char, 1),
    ("image_dimension.dim", 40, Kind::I16, 8),
    ("image_dimension.vox_units", 56, Kind::Char, 4),
    ("image_dimension.cal_units", 60, Kind::Char, 8),
    ("image_dimension.data_type", 70, Kind::I16, 1),
    ("image_dimension.bitpix", 72, Kind::I16, 1),
    ("image_dimension.dim_un0", 74, Kind::I16, 1),
    ("image_dimension.pixdim", 76, Kind::F32, 8),
    ("image_dimension.vox_offset", 108, Kind::F32, 1),
    ("image_dimension.cal_max", 124, Kind::F32, 1),
    ("image_dimension.cal_min", 128, Kind::F32, 1),
    ("image_dimension.compressed", 132, Kind::I32, 1),
    ("image_dimension.verified", 136, Kind::I32, 1),
    ("image_dimension.glmax", 140, Kind::I32, 1),
    ("image_dimension.glmin", 144, Kind::I32, 1),
    ("data_history.descrip", 148, Kind::Text(80), 1),
    ("data_history.aux_file", 228, Kind::Text(24), 1),
    ("data_history.orient", 252, Kind::Char, 1),
    ("data_history.originator", 253, Kind::Text(10), 1),
    ("data_history.generated", 263, Kind::Text(10), 1),
    ("data_history.scannum", 273, Kind::Text(10), 1),
    ("data_history.patient_id", 283, Kind::Text(10), 1),
    ("data_history.exp_date", 293, Kind::Text(10), 1),
    ("data_history.exp_time", 303, Kind::Text(10), 1),
    ("data_history.hist_un0", 313, Kind::Text(3), 1),
    ("data_history.views", 316, Kind::I32, 1),
    ("data_history.vols_added", 320, Kind::I32, 1),
    ("data_history.start_field", 324, Kind::I32, 1),
    ("data_history.field_skip", 328, Kind::I32, 1),
    ("data_history.omax", 332, Kind::I32, 1),
    ("data_history.omin", 336, Kind::I32, 1),
    ("data_history.smax", 340, Kind::I32, 1),
    ("data_history.smin", 344, Kind::I32, 1),
];

/// Everything this module reads out of an Analyze `.hdr`.
#[derive(Clone, Debug)]
struct Header {
    /// The whole 348 bytes, kept so the `dsr` blob and the field table can
    /// both be built from one copy. Every field the loader acts on is read
    /// back out of it by offset rather than cached beside it, because the
    /// metadata table already has to read all 63 that way and two readers of
    /// one struct is one too many.
    raw: Vec<u8>,
    carrier: Carrier,
    width: u32,
    height: u32,
}

impl Header {
    /// Parse and validate a `.hdr`, in the order `vips__isanalyze` and
    /// `read_header` run their checks.
    ///
    /// The order is what decides which error a header with more than one
    /// problem reports, and `le_header.hdr` is the fixture that has several:
    /// it is refused on `sizeof_hdr` before its byte-swapped rank or
    /// datatype is ever looked at.
    fn parse(hdr: &[u8]) -> Result<Self, AnalyzeError> {
        if hdr.len() != HEADER_BYTES {
            return Err(AnalyzeError::HeaderFileSize {
                found: hdr.len(),
                needed: HEADER_BYTES,
            });
        }
        let sizeof_hdr = be_i32(hdr, 0);
        if sizeof_hdr != HEADER_BYTES as i32 {
            return Err(AnalyzeError::BadHeaderSizeField {
                found: sizeof_hdr,
                expected: HEADER_BYTES as i32,
            });
        }
        let mut dim = [0i16; 8];
        for (i, slot) in dim.iter_mut().enumerate() {
            *slot = be_i16(hdr, 40 + i * 2);
        }
        let rank = dim[0];
        if !(MIN_RANK..=MAX_RANK).contains(&rank) {
            return Err(AnalyzeError::UnsupportedRank {
                found: rank,
                min: MIN_RANK,
                max: MAX_RANK,
            });
        }
        let datatype = be_i16(hdr, 70);
        let carrier = carrier_for(datatype)?;

        // The divergence: vips lets a non-positive extent through to
        // GObject, which rejects it, leaves the property at 1 and carries
        // on.
        for (axis, &extent) in dim.iter().enumerate().take(rank as usize + 1).skip(1) {
            if extent <= 0 {
                return Err(AnalyzeError::NonPositiveDimension {
                    axis,
                    found: extent,
                });
            }
        }
        let width = u32::try_from(dim[1]).map_err(|_| AnalyzeError::DimensionOverflow { rank })?;
        let mut height: u64 = 1;
        for &extent in dim.iter().take(rank as usize + 1).skip(2) {
            height = height
                .checked_mul(extent as u64)
                .ok_or(AnalyzeError::DimensionOverflow { rank })?;
        }
        let height = u32::try_from(height).map_err(|_| AnalyzeError::DimensionOverflow { rank })?;
        Ok(Self {
            raw: hdr.to_vec(),
            carrier,
            width,
            height,
        })
    }

    /// Attach the 348-byte blob and every one of the 63 fields.
    fn attach(&self, raster: &mut Raster) {
        raster
            .fields
            .set("dsr", MetadataValue::Blob(self.raw.clone()));
        for &(name, at, kind, count) in FIELDS {
            for i in 0..count {
                let key = if count == 1 {
                    format!("dsr-{name}")
                } else {
                    format!("dsr-{name}[{i}]")
                };
                let value = match kind {
                    Kind::I16 => MetadataValue::Int(be_i16(&self.raw, at + i * 2).into()),
                    Kind::I32 => MetadataValue::Int(be_i32(&self.raw, at + i * 4).into()),
                    Kind::F32 => MetadataValue::Double(be_f32(&self.raw, at + i * 4).into()),
                    Kind::Char => MetadataValue::Int(self.raw[at + i].into()),
                    Kind::Text(len) => MetadataValue::Str(getstr(&self.raw[at..at + len])),
                };
                raster.fields.set(&key, value);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The three entry points
// ---------------------------------------------------------------------------

/// The `.hdr` and `.img` names an Analyze path resolves to.
///
/// `generate_filenames` (`analyze2vips.c:224-231`) rewrites whatever it is
/// handed into both names through `vips__change_suffix`, so `fred.hdr`,
/// `fred.img` and the bare `fred` all name the same image.
///
/// The stripping is a **loop** and it is case-insensitive, which is measured
/// rather than read: `vips analyzeload c.hdr.hdr` on a directory holding
/// `c.hdr.hdr` and `c.hdr.img` reports `unable to open file "c.hdr"`, so
/// both suffixes came off before the new one went on. Reproduced here, since
/// diverging would send the two entry points to different files for the same
/// name.
///
/// The suffix that goes back on is lower-case, again as vips writes it, so
/// on a case-sensitive filesystem `FRED.HDR` resolves to `FRED.hdr`. On a
/// case-insensitive one, which is where this was measured, it does not
/// matter.
#[must_use]
pub fn analyze_filenames(path: &Path) -> (PathBuf, PathBuf) {
    let mut stem = path.as_os_str().to_string_lossy().into_owned();
    loop {
        let tail = stem
            .get(stem.len().saturating_sub(4)..)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if tail == ".hdr" || tail == ".img" {
            stem.truncate(stem.len() - 4);
        } else {
            break;
        }
    }
    (
        PathBuf::from(format!("{stem}.hdr")),
        PathBuf::from(format!("{stem}.img")),
    )
}

/// Decode an Analyze image from the two buffers that make it up.
///
/// This is the core both other entry points reach: `hdr` must be exactly
/// [`HEADER_BYTES`] long and big-endian, and `img` is the raw array, read
/// from byte 0 whatever `vox_offset` says.
///
/// # Errors
///
/// * [`AnalyzeError`] for every malformation either half carries, each
///   named: a `.hdr` of the wrong length, a `sizeof_hdr` that disagrees, a
///   rank outside 2..=7, a datatype the reference refuses, a datatype with
///   no carrier here, a non-positive dimension, and an `.img` shorter than
///   the geometry needs.
/// * [`SourceError::AllocLimitExceeded`],
///   [`SourceError::DimensionLimitExceeded`] or
///   [`SourceError::CoordLimitExceeded`] when the declared geometry is over
///   a [`DecodeLimits`] ceiling.
pub fn decode_analyze(hdr: &[u8], img: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let header = Header::parse(hdr)?;
    let needed = header.price(limits)?;
    let available = img.len() as u64;
    if available < needed {
        return Err(AnalyzeError::TruncatedImage {
            found: available,
            needed,
        }
        .into());
    }
    // A longer `.img` is accepted and its tail ignored: measured on
    // `img_oversize.img`, which carries 100 bytes of 0xff past the six the
    // geometry needs.
    header.build(&img[..needed as usize], limits)
}

/// Decode an Analyze image from either half of the pair, or from the bare
/// stem.
///
/// The `.hdr` is read and validated first, and the declared geometry is
/// priced against `limits` **before** the `.img` is opened, so a header that
/// prices past the budget costs no second read.
///
/// # Errors
///
/// As [`decode_analyze`], plus [`SourceError::Io`] if either half cannot be
/// opened or read. A missing `.img` beside a valid `.hdr` is the shape vips
/// reports as `unable to open "fred.img"`, and it arrives here the same way,
/// as an I/O error naming the sibling.
pub fn decode_analyze_file(path: &Path, limits: DecodeLimits) -> Result<Raster, SourceError> {
    let (hdr_path, img_path) = analyze_filenames(path);
    let hdr = read_file_bounded(&hdr_path, limits, "Analyze header file")?;
    let header = Header::parse(&hdr)?;
    let needed = header.price(limits)?;
    let img = read_file_bounded(&img_path, limits, "Analyze image file")?;
    let available = img.len() as u64;
    if available < needed {
        return Err(AnalyzeError::TruncatedImage {
            found: available,
            needed,
        }
        .into());
    }
    header.build(&img[..needed as usize], limits)
}

/// What [`crate::source::decode_bytes`] does with an Analyze `.hdr`.
///
/// The header is parsed and validated in full, so a malformed one reports
/// its malformation and only a header that would otherwise have loaded
/// reaches [`AnalyzeError::PixelsAreInASiblingFile`]. That ordering is the
/// point of the function: "this is half of a pair" is a useful answer, and
/// "sizeof_hdr reads 200" is a more useful one still.
///
/// # Errors
///
/// Always. See [`AnalyzeError`].
pub(crate) fn decode_analyze_header(
    hdr: &[u8],
    limits: DecodeLimits,
) -> Result<Raster, SourceError> {
    let header = Header::parse(hdr)?;
    header.price(limits)?;
    Err(AnalyzeError::PixelsAreInASiblingFile {
        width: header.width,
        height: header.height,
    }
    .into())
}

impl Header {
    /// Price the declared geometry against every ceiling, before a byte of
    /// the `.img` is read.
    ///
    /// `dims_32767.hdr` declares 1.07 gigapixels in front of a six-byte
    /// `.img` and the header load succeeds in vips, so this is the whole
    /// reason the checks are here rather than after the read.
    fn price(&self, limits: DecodeLimits) -> Result<u64, SourceError> {
        limits.check_coord(self.width, self.height)?;
        limits.check_pixels(self.width, self.height)?;
        // One spelling of the budget for the whole crate (issue #632), and
        // the shared refusal shape (issue #686).
        limits.check_image_alloc(
            "Analyze pixel buffer",
            self.width,
            self.height,
            self.carrier.bands(),
            self.carrier.sample_bytes(),
        )
    }

    /// Build the raster from an `.img` slice already cut to the exact
    /// declared length.
    fn build(&self, pixels: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
        let data = to_native(pixels, self.carrier);
        let mut raster = Raster::new_with_budget(
            self.width,
            self.height,
            self.carrier.pixel_format(),
            data,
            limits.max_alloc_bytes,
        )
        .map_err(AnalyzeError::Raster)?;
        raster.meta.interpretation = Some(self.carrier.interpretation());
        self.attach(&mut raster);
        Ok(raster)
    }
}

/// Turn the `.img`'s big-endian array into libviprs's native-endian buffer.
///
/// `vips__analyze_read` runs `vips__byteswap_bool` over the pixels whenever
/// the host is not big-endian (`analyze2vips.c:589`), so the file is always
/// big-endian and the swap is unconditional in the format rather than in the
/// host. `DT_RGB` is interleaved bytes and cannot be swapped; everything
/// else here is one four-byte float.
fn to_native(pixels: &[u8], carrier: Carrier) -> Vec<u8> {
    match carrier {
        // One byte per sample: byte order cannot apply.
        Carrier::U8 | Carrier::Rgb24 => pixels.to_vec(),
        Carrier::F32 => pixels
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| f32::from_be_bytes(*c).to_ne_bytes())
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::{decode_bytes_with_limits, decode_file_with_limits};

    /// The directory `oracle-captures/foreign-analyze/capture.py` writes its
    /// 82 fixtures into.
    const FIXTURES: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/oracle-captures/foreign-analyze/fixtures/"
    );

    /// One capture fixture, embedded rather than read.
    ///
    /// `include_bytes!` rather than `std::fs::read` on purpose: it keeps
    /// every test that does not *need* a pair on disk runnable under Miri,
    /// where the isolation layer refuses a real `open` and aborts the whole
    /// session on the first one it meets (issue #652). Three tests in this
    /// module genuinely need the filesystem, because a two-file entry point
    /// cannot be exercised from buffers, and they are named in the PR.
    macro_rules! fixture {
        ($($name:tt)+) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/oracle-captures/foreign-analyze/fixtures/",
                $($name)+
            ))
            .as_slice()
        };
    }

    /// A `.hdr` and its `.img`, decoded through the buffer-pair entry point.
    macro_rules! decoded {
        ($stem:literal) => {
            decode_analyze(
                fixture!(concat!($stem, ".hdr")),
                fixture!(concat!($stem, ".img")),
                DecodeLimits::default(),
            )
            .unwrap_or_else(|e| panic!(concat!($stem, " should decode: {}"), e))
        };
    }

    macro_rules! refused {
        ($stem:literal) => {
            decode_analyze(
                fixture!(concat!($stem, ".hdr")),
                fixture!(concat!($stem, ".img")),
                DecodeLimits::default(),
            )
            .expect_err(concat!($stem, " should be refused"))
        };
    }

    fn f32_samples(raster: &Raster) -> Vec<f32> {
        raster
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_ne_bytes)
            .collect()
    }

    /**
     * The base case, and the one that pins that a pair is read as a pair:
     * the geometry comes from the `.hdr` and every pixel from the `.img`.
     * Measured: `hdr_img_pairing`'s `hdr` case, `3x2 uchar, 1 band, b-w`.
     *
     * The `b-w` tag is worth its own line, because `matload`, the other
     * loader captured beside this one, tags the same one-band 8-bit shape
     * `multiband`. Two loaders in one build disagreeing about the same
     * shape is exactly the sort of thing a port collapses by accident.
     * Input: `base_2d_uchar.hdr` + `.img` -> Output: a 3x2 `Gray8` raster
     * tagged `b-w` holding the six `.img` bytes.
     */
    #[test]
    fn a_pair_takes_its_geometry_from_the_hdr_and_its_pixels_from_the_img() {
        let raster = decoded!("base_2d_uchar");
        assert_eq!((raster.width(), raster.height()), (3, 2));
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(raster.data(), &[10, 20, 30, 40, 50, 60]);
        assert_eq!(raster.meta.interpretation, Some(Interpretation::Bw));
    }

    /**
     * The rank rule, which is the one [`crate::nifti`] borrows for the
     * sibling format: `dim[0]` is the rank, the width is `dim[1]`, and the
     * height is `dim[2]` multiplied by every higher extent up to the rank.
     * So a 3x2x2 volume flattens to one 3x4 image with the slices stacked
     * vertically, and nothing in the loaded image records that it was ever
     * three-dimensional beyond the `dsr-image_dimension.dim[]` metadata.
     *
     * All four ranks are here rather than one, because a rule that multiplies
     * is satisfied by "take the last extent" at rank 3 and only comes apart
     * at rank 4. Measured: the `rank_and_flattening` record's 2, 3, 4 and 7
     * cases, each with its own `measured_width` and `measured_height`.
     * Input: the four rank fixtures -> Output: 3x2, 3x4, 3x8 and 2x4.
     */
    #[test]
    fn the_rank_flattens_into_the_height_by_multiplication() {
        assert_eq!(
            {
                let r = decoded!("rank2");
                (r.width(), r.height())
            },
            (3, 2)
        );
        let rank3 = decoded!("rank3");
        assert_eq!((rank3.width(), rank3.height()), (3, 4));
        assert_eq!(
            rank3.data(),
            &[0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
        );
        let rank4 = decoded!("rank4");
        assert_eq!((rank4.width(), rank4.height()), (3, 8));
        assert_eq!(rank4.data().len(), 24);
        assert_eq!(rank4.data()[23], 161);
        let rank7 = decoded!("rank7");
        assert_eq!((rank7.width(), rank7.height()), (2, 4));
        assert_eq!(rank7.data(), &[0, 7, 14, 21, 28, 35, 42, 49]);
    }

    /**
     * The rank range, which is asymmetric in a way a spec reading does not
     * predict: rank 1 is refused even though a 1-D array is a perfectly good
     * image, and rank 0 and rank 8 are refused too, the latter before
     * `dim[8]` can be read, which is just as well since `dim[]` has only
     * eight slots.
     * Measured: `0-dimensional images not supported`,
     * `1-dimensional images not supported` and
     * `8-dimensional images not supported`.
     * Input: `rank0`, `rank1` and `rank8` -> Output: `UnsupportedRank`
     * naming the rank each declared.
     */
    #[test]
    fn a_rank_outside_two_to_seven_is_refused_by_name() {
        for (err, expected) in [
            (refused!("rank0"), 0i16),
            (refused!("rank1"), 1),
            (refused!("rank8"), 8),
        ] {
            match err {
                SourceError::Analyze(AnalyzeError::UnsupportedRank { found, min, max }) => {
                    assert_eq!((found, min, max), (expected, MIN_RANK, MAX_RANK));
                }
                other => panic!("expected UnsupportedRank, got {other:?}"),
            }
        }
    }

    /**
     * The whole datatype table, measured off `vipsheader` and `vips
     * analyzeload` rather than off `get_vips_properties`, split three ways:
     * the three libviprs has a carrier for load with their measured pixels,
     * the four vips loads and libviprs cannot are refused **by name**, and
     * the four vips refuses too are refused as the datatype vips itself
     * calls unsupported.
     *
     * `DT_BINARY` (1) is in the middle group's neighbour on purpose: it is
     * named in `dbh.h` and is **not** implemented, so it belongs with 0, 256
     * and 511 rather than with the carrier ceiling.
     * Input: the eleven `dt*.hdr` fixtures -> Output: three rasters with
     * their measured pixels and eight named refusals.
     */
    #[test]
    fn every_datatype_lands_on_its_measured_carrier_or_is_refused_by_name() {
        let uchar = decoded!("dt2");
        assert_eq!(uchar.format(), PixelFormat::Gray8);
        assert_eq!(uchar.data(), &[1, 2, 3, 4]);

        // DT_FLOAT, and the one fixture in the capture that pins the pixel
        // byte order: the `.img` is `01 02 03 04 ..` and 0x01020304 read
        // big-endian is 2.387939260590663e-38. Read little-endian it is
        // 1.539e-36, so this value cannot survive the wrong swap.
        let float = decoded!("dt16");
        assert_eq!(float.format(), PixelFormat::FloatF32(NonZeroU16::MIN));
        assert_eq!(float.meta.interpretation, Some(Interpretation::Bw));
        let measured = [
            f32::from_be_bytes([1, 2, 3, 4]),
            f32::from_be_bytes([5, 6, 7, 8]),
            f32::from_be_bytes([9, 10, 11, 12]),
            f32::from_be_bytes([13, 14, 15, 16]),
        ];
        assert_eq!(f32_samples(&float), measured.to_vec());
        assert!(
            (f64::from(measured[0]) - 2.387_939_260_590_663e-38).abs() < 1e-45,
            "the capture's own value for the first sample"
        );

        let rgb = decoded!("dt128");
        assert_eq!(rgb.format(), PixelFormat::Rgb8);
        assert_eq!(rgb.meta.interpretation, Some(Interpretation::Srgb));
        assert_eq!(rgb.data(), &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);

        // The four vips loads and libviprs has no carrier for.
        let ceilings: [(SourceError, i16, &str); 4] = [
            (refused!("dt4"), 4, "DT_SIGNED_SHORT"),
            (refused!("dt8"), 8, "DT_SIGNED_INT"),
            (refused!("dt32"), 32, "DT_COMPLEX"),
            (refused!("dt64"), 64, "DT_DOUBLE"),
        ];
        for (err, datatype, name) in ceilings {
            match err {
                SourceError::Analyze(AnalyzeError::UnsupportedCarrier {
                    datatype: got,
                    name: got_name,
                    ..
                }) => assert_eq!((got, got_name), (datatype, name)),
                other => panic!("{name} must be refused by name, got {other:?}"),
            }
        }

        // And the four `get_vips_properties` itself refuses.
        for (err, datatype) in [
            (refused!("dt0"), 0i16),
            (refused!("dt1"), 1),
            (refused!("dt256"), 256),
            (refused!("dt511"), 511),
        ] {
            match err {
                SourceError::Analyze(AnalyzeError::UnsupportedDatatype { datatype: got }) => {
                    assert_eq!(got, datatype);
                }
                other => panic!("expected UnsupportedDatatype for {datatype}, got {other:?}"),
            }
        }
    }

    /**
     * `DT_RGB` is the only datatype giving more than one band, and its
     * `.img` is **interleaved**, not planar. The Analyze spec allows either
     * layout and vips reads only the interleaved one, so a port that guessed
     * planar would produce a plausible image with the wrong colours.
     * Measured: `dt_rgb_is_interleaved`'s six pixels, whose channels are
     * `n, n+1, n+2` per pixel rather than three separate ramps.
     * Input: `rgb_2d` -> Output: a 3x2 `Rgb8` raster whose first pixel is
     * `(10, 11, 12)` rather than `(10, 20, 30)`.
     */
    #[test]
    fn dt_rgb_is_interleaved_rather_than_planar() {
        let raster = decoded!("rgb_2d");
        assert_eq!((raster.width(), raster.height()), (3, 2));
        assert_eq!(
            raster.data(),
            &[
                10, 11, 12, 20, 21, 22, 30, 31, 32, // row 0
                40, 41, 42, 50, 51, 52, 60, 61, 62, // row 1
            ]
        );
    }

    /**
     * `vox_offset` is the format's own "skip this many bytes of the `.img`"
     * field, and vips parses it, attaches it as
     * `dsr-image_dimension.vox_offset`, and then reads from byte 0 anyway
     * (`analyze2vips.c:582-583` passes a hard-coded 0). A port that honoured
     * it would disagree with vips on every file that sets one.
     * Measured: `vox_offset_64.hdr` declares 64 and its pixels are the
     * **first** six bytes of a 70-byte `.img` whose tail is 0xff.
     * Input: a header declaring `vox_offset = 64` -> Output: the six bytes
     * at offset 0, with 64 still attached as metadata.
     */
    #[test]
    fn vox_offset_is_parsed_and_attached_and_then_ignored() {
        let raster = decoded!("vox_offset_64");
        assert_eq!(raster.data(), &[10, 20, 30, 40, 50, 60]);
        assert_eq!(
            raster
                .fields
                .get("dsr-image_dimension.vox_offset")
                .map(MetadataValue::as_f64),
            Some(64.0),
            "parsed and attached, which is what makes ignoring it checkable"
        );
    }

    /**
     * `bitpix` is attached and never consulted: the datatype alone fixes the
     * sample width. Pinned by poking a loadable header's `bitpix` to a value
     * that disagrees with its datatype and asserting nothing about the
     * decode moves, which is the only way to tell "read and ignored" from
     * "never read".
     * Input: `dt2` with `bitpix` poked from 8 to 1 -> Output: the same 2x2
     * `Gray8` raster, with the poked value attached.
     */
    #[test]
    fn bitpix_is_attached_and_never_consulted() {
        let mut hdr = fixture!("dt2.hdr").to_vec();
        assert_eq!(be_i16(&hdr, 72), 8, "bitpix");
        hdr[72..74].copy_from_slice(&1i16.to_be_bytes());
        let raster = decode_analyze(&hdr, fixture!("dt2.img"), DecodeLimits::default())
            .expect("bitpix does not decide anything");
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(raster.data(), &[1, 2, 3, 4]);
        assert_eq!(
            raster
                .fields
                .get("dsr-image_dimension.bitpix")
                .map(MetadataValue::as_i64),
            Some(1)
        );
    }

    /**
     * The `.img` length rule, which is asymmetric: a short one is an error
     * and a long one is not. Measured both ways, `img_truncated.img` two
     * bytes short of six giving `file too short`, and `img_oversize.img`
     * carrying 100 bytes of 0xff past the six the geometry needs and loading
     * fine with the tail ignored.
     * The oversize half is the positive control: without it, refusing every
     * `.img` whose length is not exactly the declared one would also pass.
     * Input: the two fixtures -> Output: `TruncatedImage { found: 2,
     * needed: 6 }` and a clean 3x2 raster.
     */
    #[test]
    fn a_short_img_is_refused_and_a_long_one_is_accepted() {
        match refused!("img_truncated") {
            SourceError::Analyze(AnalyzeError::TruncatedImage { found, needed }) => {
                assert_eq!((found, needed), (2, 6));
            }
            other => panic!("expected TruncatedImage, got {other:?}"),
        }
        let oversize = decoded!("img_oversize");
        assert_eq!((oversize.width(), oversize.height()), (3, 2));
        assert_eq!(oversize.data(), &[10, 20, 30, 40, 50, 60]);
    }

    /**
     * The two length checks vips keeps apart, and this module keeps apart
     * with it: `header file size incorrect` is about the **file**, and
     * `header size incorrect` is about the `sizeof_hdr` **field**. A reader
     * that collapsed them would report the wrong one for half the capture.
     * Measured: 0, 347 and 349 bytes for the first; 0, 200 and a
     * little-endian 348 for the second, plus three headers of garbage.
     * Input: nine fixtures -> Output: `HeaderFileSize` for three and
     * `BadHeaderSizeField` for six, with `sizeof_hdr_348` as the control
     * that a correct field loads.
     */
    #[test]
    fn the_two_header_size_checks_are_about_different_things() {
        for (name, bytes) in [
            ("hdr_0_bytes", fixture!("hdr_0_bytes.hdr")),
            ("hdr_347_bytes", fixture!("hdr_347_bytes.hdr")),
            ("hdr_349_bytes", fixture!("hdr_349_bytes.hdr")),
        ] {
            match Header::parse(bytes) {
                Err(AnalyzeError::HeaderFileSize { found, needed }) => {
                    assert_eq!((found, needed), (bytes.len(), HEADER_BYTES), "{name}");
                }
                other => panic!("{name}: expected HeaderFileSize, got {other:?}"),
            }
        }
        for (name, bytes, expected) in [
            ("sizeof_hdr_0", fixture!("sizeof_hdr_0.hdr"), 0i32),
            ("sizeof_hdr_200", fixture!("sizeof_hdr_200.hdr"), 200),
            // The whole byte-order check, in one field: a little-endian 348
            // reads back as 0x5c010000.
            ("le_header", fixture!("le_header.hdr"), 1_543_569_408),
            ("all_zero_348", fixture!("all_zero_348.hdr"), 0),
            ("all_ff_348", fixture!("all_ff_348.hdr"), -1),
            ("ascii_348", fixture!("ascii_348.hdr"), 1_852_797_984),
        ] {
            match Header::parse(bytes) {
                Err(AnalyzeError::BadHeaderSizeField { found, .. }) => {
                    assert_eq!(found, expected, "{name}");
                }
                other => panic!("{name}: expected BadHeaderSizeField, got {other:?}"),
            }
        }
        assert!(
            Header::parse(fixture!("sizeof_hdr_348.hdr")).is_ok(),
            "the control: 348 in the field and 348 bytes in the file"
        );
    }

    /**
     * A deliberate divergence, and the same one `matload` carries: `dim[]`
     * is a signed short and nothing in `get_vips_properties` range-checks
     * it, so a zero or negative extent reaches `vips_image_init_fields`,
     * GObject's property range check rejects the value, prints a
     * `GLib-GObject-CRITICAL`, **leaves the property at its default of 1**,
     * and the load carries on and exits 0 with a silently wrong geometry.
     * Measured: `dim1_negative.hdr` declares `-3` for the width and loads as
     * `1x2`, `dim2_negative.hdr` declares `-2` for the height and loads as
     * `3x1`, and `dim1_zero.hdr` declares 0 and loads as `1x2`. All three
     * exit 0.
     * Input: the three fixtures -> Output: `NonPositiveDimension` naming the
     * axis and the value, not a clamped raster.
     */
    #[test]
    fn a_non_positive_dimension_is_refused_rather_than_clamped_to_one() {
        for (err, axis, found) in [
            (refused!("dim1_zero"), 1usize, 0i16),
            (refused!("dim1_negative"), 1, -3),
            (refused!("dim2_negative"), 2, -2),
        ] {
            match err {
                SourceError::Analyze(AnalyzeError::NonPositiveDimension {
                    axis: got_axis,
                    found: got,
                }) => assert_eq!((got_axis, got), (axis, found)),
                other => panic!("expected NonPositiveDimension, got {other:?}"),
            }
        }
    }

    /**
     * The decompression-bomb shape [`DecodeLimits`] exists for: 348 bytes
     * declaring 1.07 gigapixels in front of a six-byte `.img`, which vips
     * reports at header time without complaint because nothing in it prices
     * the declared geometry.
     *
     * The ceilings are walked down one at a time so the refusal does not
     * rest on a single arm, and the last row is the positive control that
     * the geometry checks are what refused the earlier ones. Note where the
     * default lands: 32767 squared is 1,073,676,289, which is *under* the
     * default gigapixel `max_pixels`, so the allocation budget is what
     * refuses this one out of the box.
     * Measured: `dims_32767.hdr` loads a header reporting `32767x32767
     * uchar` and fails at the pixels with `file too short`.
     * Input: one fixture under four budgets -> Output: four different
     * refusals, in the order the checks run.
     */
    #[test]
    fn a_gigapixel_declaration_is_refused_before_the_img_is_read() {
        let hdr = fixture!("dims_32767.hdr");
        let img = fixture!("dims_32767.img");

        let tight = DecodeLimits::default().with_max_coord(1_000);
        match decode_analyze(hdr, img, tight) {
            Err(SourceError::CoordLimitExceeded { width, height, .. }) => {
                assert_eq!((width, height), (32_767, 32_767));
            }
            other => panic!("expected CoordLimitExceeded, got {other:?}"),
        }

        let by_pixels = DecodeLimits::default().with_max_pixels(1 << 20);
        match decode_analyze(hdr, img, by_pixels) {
            Err(SourceError::DimensionLimitExceeded { width, height, .. }) => {
                assert_eq!((width, height), (32_767, 32_767));
            }
            other => panic!("expected DimensionLimitExceeded, got {other:?}"),
        }

        match decode_analyze(hdr, img, DecodeLimits::default()) {
            Err(SourceError::AllocLimitExceeded {
                what, needed_bytes, ..
            }) => {
                assert_eq!(what, "Analyze pixel buffer");
                assert_eq!(needed_bytes, 1_073_676_289);
            }
            other => panic!("expected AllocLimitExceeded, got {other:?}"),
        }

        let open = DecodeLimits::default().with_max_alloc_bytes(u64::MAX - 1);
        match decode_analyze(hdr, img, open) {
            Err(SourceError::Analyze(AnalyzeError::TruncatedImage { found, needed })) => {
                assert_eq!((found, needed), (6, 1_073_676_289));
            }
            other => panic!("expected TruncatedImage with every ceiling lifted, got {other:?}"),
        }
    }

    /**
     * `attach_meta` (`analyze2vips.c:437-482`) sets the whole 348-byte
     * header as a blob named `dsr` and then walks its table setting one
     * field per struct member. The count is 63, measured off `vipsheader -a`
     * rather than counted off the C table, and it is pinned here because a
     * field silently dropped from the table is invisible otherwise.
     *
     * A sample of the values is checked too, one per `Kind`, because a table
     * with the right *number* of rows and the wrong offsets would pass a
     * count on its own.
     * Input: `meta_strings.hdr` -> Output: 63 `dsr-` fields plus the blob,
     * with the measured value for one of each kind.
     */
    #[test]
    fn the_metadata_is_the_measured_sixty_three_fields_and_the_blob() {
        let raster = decoded!("meta_strings");
        let names: Vec<String> = raster
            .get_fields()
            .into_iter()
            .filter(|name| name.starts_with("dsr-"))
            .collect();
        assert_eq!(
            names.len(),
            63,
            "the capture counted 63 dsr-<section>.<member> fields off vipsheader -a"
        );
        assert_eq!(
            raster.fields.get("dsr"),
            Some(&MetadataValue::Blob(fixture!("meta_strings.hdr").to_vec())),
            "and the whole 348-byte header as one blob beside them"
        );

        let int = |name: &str| raster.fields.get(name).map(MetadataValue::as_i64);
        let text = |name: &str| raster.fields.get(name).map(|v| v.as_str().to_owned());
        // One per Kind, with the measured value from the capture's header
        // block.
        assert_eq!(int("dsr-header_key.sizeof_hdr"), Some(348)); // I32
        assert_eq!(int("dsr-header_key.session_error"), Some(0)); // I16
        assert_eq!(int("dsr-header_key.regular"), Some(114)); // Char, 'r'
        assert_eq!(int("dsr-header_key.extents"), Some(16_384));
        assert_eq!(int("dsr-image_dimension.dim[1]"), Some(3));
        assert_eq!(int("dsr-image_dimension.glmax"), Some(255));
        assert_eq!(
            raster
                .fields
                .get("dsr-image_dimension.pixdim[1]")
                .map(MetadataValue::as_f64),
            Some(1.0)
        ); // F32
        assert_eq!(text("dsr-header_key.data_type").as_deref(), Some("")); // Text
        assert_eq!(int("dsr-data_history.smin"), Some(0));
    }

    /**
     * The two traps in `getstr` (`analyze2vips.c:237-256`), which are the
     * only place the Analyze loader is lossy.
     *
     * `g_strlcpy` is given the **field** length as its buffer size, so it
     * copies at most `len - 1` characters and an 80-byte `descrip` loses its
     * last byte: measured, 80 characters in and 79 out. And every byte
     * failing `isascii(c) && c >= 32` is rewritten to `@`: measured on a
     * `patient_id` holding `0x01 0x02 0x7f` and a UTF-8 e-acute, where the
     * DEL passes through untouched and both bytes of the e-acute do not.
     *
     * The capture's own prose states that predicate with an `||`, which its
     * own data rules out: `0x01` is `isascii` and would survive under the
     * OR, and it is measured as `@`. That is issue #797, and the sweep below
     * is what holds the `&&` rather than a sentence.
     * Input: `meta_strings.hdr`, and every byte value 0..=255 through
     * `getstr` -> Output: the two measured strings, and replacement at
     * exactly `c < 32 || c > 127`.
     */
    #[test]
    fn getstr_drops_the_last_byte_and_rewrites_what_is_not_printable_ascii() {
        let raster = decoded!("meta_strings");
        let descrip = raster
            .fields
            .get("dsr-data_history.descrip")
            .expect("descrip")
            .as_str()
            .to_owned();
        assert_eq!(
            descrip.len(),
            79,
            "an 80-byte field loses its last byte to g_strlcpy's size argument"
        );
        assert!(descrip.starts_with("ABCDEFGHIJKLMNOPQRSTUVWXYZ"));
        assert!(descrip.ends_with('A'), "the written field ended in B");

        let patient = raster
            .fields
            .get("dsr-data_history.patient_id")
            .expect("patient_id")
            .as_str()
            .to_owned();
        assert_eq!(patient, "ok@@\u{7f}@@en");

        // And the predicate itself, swept rather than argued. The two
        // readings differ on 0x00..0x1f and on 0x80..0xff, and agree on
        // 0x20..0x7f, so a sweep is the only thing that separates them.
        let mut replaced = Vec::new();
        for byte in 0u8..=255 {
            // A three-byte field so `getstr` keeps two: the byte under test
            // and a terminator that is never reached.
            let rendered = getstr(&[byte, b'z', 0]);
            if rendered.starts_with('@') && byte != b'@' {
                replaced.push(byte);
            }
        }
        let expected: Vec<u8> = (0u8..=255)
            .filter(|&b| b != 0 && !(32..=127).contains(&b))
            .collect();
        assert_eq!(
            replaced, expected,
            "getstr replaces a byte iff it is not (isascii(c) && c >= 32); issue #797"
        );
        // The positive control the capture singles out, stated on its own so
        // a reader can see the OR would keep it too and that is why it
        // decides nothing.
        assert_eq!(getstr(&[0x7f, b'z', 0]), "\u{7f}z");
        assert_eq!(getstr(&[0x01, b'z', 0]), "@z");
    }

    /**
     * Issue #797. The capture's own prose has to name the predicate its own
     * data measures, so the sentence a reader ports from cannot say one
     * thing while the numbers under it say another.
     *
     * This is the shape #752 was in the NIfTI capture: the measurements were
     * right and the sentence above them was wrong, and a port written from
     * the sentence disagreed with the oracle. Here the sentence said
     * `isascii(c) || c >= 32` and the record's own `patient_id` rules the
     * OR out twice over.
     *
     * Both halves are checked, because either alone is satisfied by the
     * wrong fix: the prose, so it cannot drift back, and this module's
     * `getstr` against the record's measured value, so the prose cannot be
     * corrected to something the implementation does not do.
     * Input: `oracle-captures/foreign-analyze/oracle.json` -> Output: the
     * `metadata` record names the AND, and `getstr` on its
     * `patient_id_written` bytes gives its recorded `patient_id`.
     */
    #[test]
    fn the_captures_own_prose_names_the_predicate_its_data_measures() {
        const ORACLE: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/oracle-captures/foreign-analyze/oracle.json"
        ));
        let json: serde_json::Value =
            serde_json::from_str(ORACLE).expect("the capture is valid JSON");
        let record = &json["records"]["metadata"];
        let prose = record["what"].as_str().expect("the record's prose");
        assert!(
            prose.contains("isascii(c) && c >= 32"),
            "the prose has to name the AND its own data measures (issue #797)"
        );
        assert!(
            !prose.contains("isascii(c) || c >= 32"),
            "and must not still name the OR that data rules out"
        );

        // The measured half, which is what makes the correction checkable
        // rather than an edit to a sentence. `0x01` is the byte that decides:
        // it is `isascii`, so under the OR it would survive.
        let written: Vec<u8> = record["patient_id_written"]
            .as_array()
            .expect("the bytes the fixture was written with")
            .iter()
            .map(|v| u8::try_from(v.as_u64().expect("a byte")).expect("a byte"))
            .collect();
        assert!(
            written.contains(&0x01),
            "the deciding byte is in the fixture"
        );
        let measured = record["header"]["dsr-data_history.patient_id"]
            .as_str()
            .expect("the value vipsheader printed");
        assert_eq!(
            getstr(&written),
            measured,
            "getstr has to reproduce the record's own measured value"
        );
    }

    /**
     * The buffer entry point on a `.hdr`, which is what
     * [`crate::source::decode_bytes`] reaches through the route table's
     * `Paired` row. A header that would otherwise load reports where its
     * pixels are; a header that would not reports its own malformation
     * first, because the parse runs before the refusal.
     *
     * The second half is the point of the ordering: "this is half of a pair"
     * is a useful answer and "sizeof_hdr reads 200" is a more useful one, so
     * a reader that refused before parsing would be strictly worse.
     * Input: a valid `.hdr` and a broken one, through both `decode_bytes`
     * and the module entry point -> Output: `PixelsAreInASiblingFile`
     * carrying the declared geometry, and `BadHeaderSizeField`.
     */
    #[test]
    fn the_header_half_alone_says_where_its_pixels_are() {
        let limits = DecodeLimits::default();
        match decode_analyze_header(fixture!("base_2d_uchar.hdr"), limits) {
            Err(SourceError::Analyze(AnalyzeError::PixelsAreInASiblingFile { width, height })) => {
                assert_eq!((width, height), (3, 2))
            }
            other => panic!("expected PixelsAreInASiblingFile, got {other:?}"),
        }
        // Through the public sniffing entry point, so the route row is what
        // is being exercised rather than this function.
        let routed = decode_bytes_with_limits(fixture!("base_2d_uchar.hdr"), limits)
            .expect_err("a .hdr alone is not an image");
        assert!(
            matches!(
                routed,
                SourceError::Analyze(AnalyzeError::PixelsAreInASiblingFile { .. })
            ),
            "the sniff must route a .hdr here rather than to the image facade: {routed:?}"
        );
        // A malformed header reports the malformation, not the pairing.
        // `rank8.hdr` rather than `sizeof_hdr_200.hdr`, because it is one of
        // the files this sniff claims and `vips__isanalyze` does not, so it
        // reaches the codec through the route table and shows the wider
        // sniff paying for itself: the message names the rank rather than
        // saying "these bytes are not an image".
        let broken = decode_bytes_with_limits(fixture!("rank8.hdr"), limits)
            .expect_err("rank 8 is not loadable");
        assert!(
            matches!(
                broken,
                SourceError::Analyze(AnalyzeError::UnsupportedRank { found: 8, .. })
            ),
            "the parse runs before the pairing refusal: {broken:?}"
        );
        // And directly, on a header the sniff does not claim at all, so the
        // ordering inside the function is pinned as well as the routing.
        assert!(
            matches!(
                decode_analyze_header(fixture!("sizeof_hdr_200.hdr"), limits),
                Err(SourceError::Analyze(AnalyzeError::BadHeaderSizeField {
                    found: 200,
                    ..
                }))
            ),
            "the parse runs before the pairing refusal"
        );
    }

    /**
     * `generate_filenames` (`analyze2vips.c:224-231`) rewrites whatever it
     * is handed into both names, and the stripping is a **loop** and is
     * case-insensitive. That is measured rather than read off the source:
     * `vips analyzeload c.hdr.hdr` in a directory holding `c.hdr.hdr` and
     * `c.hdr.img` reports `vips__file_open_read: unable to open file
     * "c.hdr"`, which is only reachable if both `.hdr` suffixes came off
     * before the new one went on. `e.zz.hdr` is the control: `.zz` is not in
     * the list, the loop stops, and it resolves to `e.zz.hdr` / `e.zz.img`,
     * which is what that case loaded.
     * Input: seven spellings of a name -> Output: the pair each one
     * resolves to.
     */
    #[test]
    fn the_suffix_stripping_is_a_loop_and_is_case_insensitive() {
        let pair = |name: &str| {
            let (hdr, img) = analyze_filenames(Path::new(name));
            (
                hdr.to_string_lossy().into_owned(),
                img.to_string_lossy().into_owned(),
            )
        };
        assert_eq!(pair("fred.hdr"), ("fred.hdr".into(), "fred.img".into()));
        assert_eq!(pair("fred.img"), ("fred.hdr".into(), "fred.img".into()));
        assert_eq!(pair("fred"), ("fred.hdr".into(), "fred.img".into()));
        // The loop, measured: both suffixes come off.
        assert_eq!(pair("c.hdr.hdr"), ("c.hdr".into(), "c.img".into()));
        assert_eq!(pair("c.img.hdr"), ("c.hdr".into(), "c.img".into()));
        // The control: a suffix that is not in the list stops the loop.
        assert_eq!(pair("e.zz.hdr"), ("e.zz.hdr".into(), "e.zz.img".into()));
        // Case-insensitive on the way off, lower-case on the way back on.
        assert_eq!(pair("FRED.HDR"), ("FRED.hdr".into(), "FRED.img".into()));
    }

    /**
     * The carrier table's two halves have to agree: the [`PixelFormat`] each
     * carrier names and the `(bands, sample_bytes)` pair it prices with are
     * two independent statements about the same thing, and the budget is
     * spent on the second while the raster is built from the first. A
     * mismatch would price a file at one size and allocate another.
     * Input: every `Carrier` -> Output: its `pixel_format` is the one
     * `PixelFormat::with_channels` builds from its own price.
     */
    #[test]
    fn the_carrier_table_agrees_with_pixel_format() {
        for carrier in Carrier::ALL {
            let from_price = PixelFormat::with_channels(
                carrier.bands() as usize,
                carrier.sample_bytes() as usize,
            )
            .unwrap_or_else(|| panic!("{carrier:?} prices a shape no PixelFormat holds"));
            assert_eq!(
                carrier.pixel_format(),
                from_price,
                "{carrier:?} names one format and prices another"
            );
        }
    }

    /// What vips does with a fixture, and what libviprs does, as one value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Verdict {
        /// vips loads it and so does libviprs.
        Both,
        /// vips refuses it and so does libviprs. The messages differ, and
        /// which loader reports differs, but the file does not load either
        /// way.
        Neither,
        /// vips loads it and libviprs refuses it, because the datatype has
        /// no carrier here yet.
        CarrierCeiling,
        /// vips loads it with a silently wrong geometry and libviprs
        /// refuses it.
        ClampedGeometry,
    }

    /**
     * The claim the module doc makes about its wider sniff, held rather than
     * asserted. libviprs claims any file whose first four bytes are
     * big-endian 348 where `vips__isanalyze` parses the whole header, so
     * libviprs's sniff is strictly wider. What that does **not** change is
     * which files load: everything vips's `is_a` rejects and libviprs's
     * sniff claims is refused by name here, and vips refuses it too by
     * falling through to `magickload`.
     *
     * So the divergence set is finite and it is enumerated: four fixtures on
     * the carrier ceiling and three on the clamped-geometry defect. Every
     * other fixture in the capture agrees, and the table says which is
     * which, so a change that widened the divergence would have to say so
     * here.
     * Input: 27 fixtures with the verdict the capture records for each ->
     * Output: libviprs's own answer matches, and the two divergence classes
     * hold exactly the fixtures named.
     */
    #[test]
    fn every_measured_fixture_loads_exactly_where_vips_loads_it() {
        let cases: [(&str, &[u8], &[u8], Verdict); 27] = [
            (
                "base_2d_uchar",
                fixture!("base_2d_uchar.hdr"),
                fixture!("base_2d_uchar.img"),
                Verdict::Both,
            ),
            (
                "sizeof_hdr_348",
                fixture!("sizeof_hdr_348.hdr"),
                fixture!("sizeof_hdr_348.img"),
                Verdict::Both,
            ),
            (
                "meta_strings",
                fixture!("meta_strings.hdr"),
                fixture!("meta_strings.img"),
                Verdict::Both,
            ),
            (
                "vox_offset_64",
                fixture!("vox_offset_64.hdr"),
                fixture!("vox_offset_64.img"),
                Verdict::Both,
            ),
            (
                "img_oversize",
                fixture!("img_oversize.hdr"),
                fixture!("img_oversize.img"),
                Verdict::Both,
            ),
            (
                "rank2",
                fixture!("rank2.hdr"),
                fixture!("rank2.img"),
                Verdict::Both,
            ),
            (
                "rank3",
                fixture!("rank3.hdr"),
                fixture!("rank3.img"),
                Verdict::Both,
            ),
            (
                "rank4",
                fixture!("rank4.hdr"),
                fixture!("rank4.img"),
                Verdict::Both,
            ),
            (
                "rank7",
                fixture!("rank7.hdr"),
                fixture!("rank7.img"),
                Verdict::Both,
            ),
            (
                "dt2",
                fixture!("dt2.hdr"),
                fixture!("dt2.img"),
                Verdict::Both,
            ),
            (
                "dt16",
                fixture!("dt16.hdr"),
                fixture!("dt16.img"),
                Verdict::Both,
            ),
            (
                "dt128",
                fixture!("dt128.hdr"),
                fixture!("dt128.img"),
                Verdict::Both,
            ),
            (
                "rgb_2d",
                fixture!("rgb_2d.hdr"),
                fixture!("rgb_2d.img"),
                Verdict::Both,
            ),
            // vips loads these four and libviprs has no carrier for them.
            (
                "dt4",
                fixture!("dt4.hdr"),
                fixture!("dt4.img"),
                Verdict::CarrierCeiling,
            ),
            (
                "dt8",
                fixture!("dt8.hdr"),
                fixture!("dt8.img"),
                Verdict::CarrierCeiling,
            ),
            (
                "dt32",
                fixture!("dt32.hdr"),
                fixture!("dt32.img"),
                Verdict::CarrierCeiling,
            ),
            (
                "dt64",
                fixture!("dt64.hdr"),
                fixture!("dt64.img"),
                Verdict::CarrierCeiling,
            ),
            // vips exits 0 on these three with a geometry GObject clamped.
            (
                "dim1_zero",
                fixture!("dim1_zero.hdr"),
                fixture!("dim1_zero.img"),
                Verdict::ClampedGeometry,
            ),
            (
                "dim1_negative",
                fixture!("dim1_negative.hdr"),
                fixture!("dim1_negative.img"),
                Verdict::ClampedGeometry,
            ),
            (
                "dim2_negative",
                fixture!("dim2_negative.hdr"),
                fixture!("dim2_negative.img"),
                Verdict::ClampedGeometry,
            ),
            // Refused on both sides, by different routes and with different
            // messages.
            (
                "dt0",
                fixture!("dt0.hdr"),
                fixture!("dt0.img"),
                Verdict::Neither,
            ),
            (
                "dt1",
                fixture!("dt1.hdr"),
                fixture!("dt1.img"),
                Verdict::Neither,
            ),
            (
                "rank0",
                fixture!("rank0.hdr"),
                fixture!("rank0.img"),
                Verdict::Neither,
            ),
            (
                "rank8",
                fixture!("rank8.hdr"),
                fixture!("rank8.img"),
                Verdict::Neither,
            ),
            (
                "hdr_347_bytes",
                fixture!("hdr_347_bytes.hdr"),
                fixture!("hdr_347_bytes.img"),
                Verdict::Neither,
            ),
            (
                "le_header",
                fixture!("le_header.hdr"),
                fixture!("le_header.img"),
                Verdict::Neither,
            ),
            (
                "img_truncated",
                fixture!("img_truncated.hdr"),
                fixture!("img_truncated.img"),
                Verdict::Neither,
            ),
        ];
        let mut ceiling = 0;
        let mut clamped = 0;
        for (name, hdr, img, verdict) in cases {
            let ours = decode_analyze(hdr, img, DecodeLimits::default());
            match verdict {
                Verdict::Both => {
                    assert!(ours.is_ok(), "{name} loads in vips and must here: {ours:?}")
                }
                Verdict::Neither => assert!(
                    ours.is_err(),
                    "{name} is refused by vips and must be refused here"
                ),
                Verdict::CarrierCeiling => {
                    ceiling += 1;
                    assert!(
                        matches!(
                            ours,
                            Err(SourceError::Analyze(
                                AnalyzeError::UnsupportedCarrier { .. }
                            ))
                        ),
                        "{name} diverges on the carrier ceiling and must say so: {ours:?}"
                    );
                }
                Verdict::ClampedGeometry => {
                    clamped += 1;
                    assert!(
                        matches!(
                            ours,
                            Err(SourceError::Analyze(
                                AnalyzeError::NonPositiveDimension { .. }
                            ))
                        ),
                        "{name} diverges on the clamped geometry and must say so: {ours:?}"
                    );
                }
            }
        }
        assert_eq!(
            (ceiling, clamped),
            (4, 3),
            "the divergence set is four carrier-ceiling fixtures and three clamped ones, \
             and it is enumerated here so widening it has to be written down"
        );
    }

    /**
     * The sniff row, which is deliberately wider than `vips__isanalyze` and
     * must still not be wide enough to steal a file from another container.
     * Every `.hdr` whose `sizeof_hdr` field reads 348 big-endian is claimed;
     * everything else in the capture falls through, including the two
     * headers whose only defect is that field. `.img` is never claimed,
     * because a raw array has no signature at all, and that is the one
     * divergence from `vips`'s entry point rather than from its parser.
     * Input: seven fixtures through `crate::source::sniff` -> Output:
     * `Analyze` for four and `None` for three.
     */
    #[test]
    fn the_sniff_claims_a_header_and_never_an_image() {
        use crate::source::{SniffedFormat, sniff};
        let analyze = Some(SniffedFormat::Analyze);
        assert_eq!(sniff(fixture!("base_2d_uchar.hdr")), analyze);
        assert_eq!(sniff(fixture!("rank8.hdr")), analyze, "wider than is_a");
        assert_eq!(sniff(fixture!("dt0.hdr")), analyze, "wider than is_a");
        assert_eq!(
            sniff(fixture!("hdr_347_bytes.hdr")),
            analyze,
            "wider than is_a"
        );
        assert_eq!(sniff(fixture!("base_2d_uchar.img")), None, "a raw array");
        assert_eq!(sniff(fixture!("sizeof_hdr_200.hdr")), None);
        assert_eq!(sniff(fixture!("le_header.hdr")), None);
    }

    // ---- the three tests that need a real pair on disk --------------------
    //
    // A two-file entry point cannot be exercised from buffers, so these
    // three reach the filesystem. They read the capture's own fixtures in
    // place rather than writing a temp directory, so they carry no
    // `tempfile` or `std::fs` marker and stay out of
    // `tests/miri_fs_test_inventory.txt`; they are Miri gate-killers of the
    // same kind `crate::nifti` and `crate::exr` already contribute, which is
    // issue #765.

    /**
     * `generate_filenames` rewrites whatever it is handed, so `.hdr`, `.img`
     * and the bare stem all name the same image. Measured: all three load in
     * vips through a direct `analyzeload`, and the bare stem fails only
     * through `vipsheader`, in `VipsForeignLoad`'s own existence check
     * before `is_a` ever runs.
     * Input: `base_2d_uchar` under all three spellings -> Output: the same
     * 3x2 raster.
     */
    #[test]
    fn a_pair_loads_from_either_name_and_from_the_bare_stem() {
        for name in ["base_2d_uchar.hdr", "base_2d_uchar.img", "base_2d_uchar"] {
            let path = format!("{FIXTURES}{name}");
            let raster = decode_analyze_file(Path::new(&path), DecodeLimits::default())
                .unwrap_or_else(|e| panic!("{name} should load: {e}"));
            assert_eq!((raster.width(), raster.height()), (3, 2), "{name}");
            assert_eq!(raster.data(), &[10, 20, 30, 40, 50, 60], "{name}");
        }
    }

    /**
     * The whole point of the `Paired` route kind: `decode_file` finds the
     * container from the `.hdr`'s content, resolves the `.img` from the
     * path, and hands back a real image. Without it Analyze would be a
     * format libviprs could parse and not load.
     * The `filename` field is checked too, because the paired branch of
     * `decode_file_with_limits` sets it separately from the other two and a
     * missing one would be invisible.
     * Input: the fixture `.hdr` path -> Output: the 3x2 raster, with the
     * path recorded.
     */
    #[test]
    fn decode_file_reaches_the_pair_from_the_hdr() {
        let path = format!("{FIXTURES}base_2d_uchar.hdr");
        let raster = decode_file_with_limits(Path::new(&path), DecodeLimits::default())
            .expect("decode_file must load an Analyze pair from its header");
        assert_eq!((raster.width(), raster.height()), (3, 2));
        assert_eq!(raster.data(), &[10, 20, 30, 40, 50, 60]);
        assert_eq!(
            raster.fields.get("filename").map(MetadataValue::as_str),
            Some(path.as_str())
        );
    }

    /**
     * A missing `.img` beside a valid `.hdr` is invisible to the header
     * load: `vips__analyze_read_header` reads only the `.hdr`, so vips
     * reports a perfectly good `3x2 uchar` header and fails at the first
     * pixel with `unable to open "fixtures/no_img.img"`. libviprs has one
     * entry point, so it arrives as an I/O error naming the sibling, which
     * is the same information at the only moment there is to report it.
     * Input: `no_img.hdr`, which has no `.img` next to it -> Output:
     * `SourceError::Io` with `NotFound`.
     */
    #[test]
    fn a_missing_img_beside_a_valid_hdr_is_an_io_error() {
        let path = format!("{FIXTURES}no_img.hdr");
        let err = decode_analyze_file(Path::new(&path), DecodeLimits::default())
            .expect_err("there is no no_img.img");
        match err {
            SourceError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::NotFound),
            other => panic!("expected an I/O error naming the sibling, got {other:?}"),
        }
    }
}
