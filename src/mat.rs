//! MATLAB level 5 (`.mat`) load: a tagged-element container with a
//! column-major array inside it.
//!
//! A MAT-5 file is a 128-byte text header and then a sequence of data
//! elements. An element is an 8-byte tag (a type and a byte count) and a
//! payload padded to eight bytes, or the small-element form where the count
//! rides in the tag's upper half. `miMATRIX` (14) is an array;
//! `miCOMPRESSED` (15) is a zlib stream holding one, and every MATLAB
//! release since 7 writes that form by default.
//!
//! # Operations
//!
//! | libviprs method | reference equivalent | result |
//! |---|---|---|
//! | [`decode_mat`] | `vips matload` | [`PixelFormat::Gray8`], [`PixelFormat::Rgb8`], [`PixelFormat::Rgba8`], [`PixelFormat::Gray16`], [`PixelFormat::Rgb16`], [`PixelFormat::Rgba16`] or a float carrier |
//!
//! There is no save. `vips` registers no `matsave`, so the format is
//! load-only here and there is no round trip to pin.
//!
//! # The oracle
//!
//! `oracle-captures/foreign-mat/`, 48 fixtures and 10 records, measured
//! against `/opt/homebrew/bin/vips` 8.18.6 (`matload`, `Matlab load with
//! matio: true`), recorded `on_pin` in `ORACLE_PIN.json`. libvips is a real
//! oracle for this format, unlike its NIfTI sibling.
//!
//! # The sniff is the shipped binary's, not the C source's
//!
//! This is the sharp edge of the port and it is issue #650.
//! `vips__mat_ismat` in the reference checkout reads **ten** bytes and
//! compares them with `MATLAB 5.0`. The shipped 8.18.6 dylib reads **one
//! hundred and twenty-eight** and also validates the version word and the
//! endian indicator, and the 8.18.4 that preceded it did not. A port written
//! from the C source would claim a large class of files 8.18.6 refuses, and
//! would route them to itself instead of leaving them to the next loader.
//!
//! The measured rule, from the capture's `sniff_predicate` record and the
//! disassembly beside it:
//!
//! * the file must yield all 128 bytes, so anything shorter is refused
//!   before the prefix is compared;
//! * bytes 0..10 are exactly `MATLAB 5.0`, case-sensitive;
//! * bytes 126..128 are the endian indicator and must be `IM` or `MI`;
//! * bytes 124..126 are the version, read in the order the indicator
//!   declares, and must be `0x0100`;
//! * bytes 10..124 are not looked at at all.
//!
//! Those five clauses are exactly two [`crate::source`] sniff rows,
//! `MATLAB 5.0` at 0 with `\x00\x01IM` or `\x01\x00MI` at byte 124: the
//! length floor falls out of the tag offset, and the version and the
//! indicator are one four-byte constant per byte order because the version
//! is `0x0100` read whichever way the indicator says. So the trap lands as
//! table data rather than as a hand-written predicate, and this module's own
//! `Header::parse` re-derives the same answer with a named error per
//! clause.
//!
//! `matio`'s own `Mat_Open`, which is what a **direct** `vips matload` call
//! reaches, is more permissive in three measured ways: it reads MAT-4, and
//! it accepts `MATLAB 5.1`, `matlab 5.0` and `MATLAB_5.0`. None of them is
//! reachable through `vips_image_new_from_file`, which is the entry point
//! [`crate::source::decode_file`] corresponds to, so this module follows the
//! sniff and refuses all four.
//!
//! # Semantics
//!
//! * **Column-major is transposed.** `mat2vips_get_header` reads the height
//!   from `dims[0]` and the width from `dims[1]`, so a MATLAB 2x3 becomes a
//!   3x2 image and element `(r, c)` is pixel `(c, r)`. Measured on
//!   `base_2x3_uint8.mat`, whose values are asymmetric on purpose.
//! * **Rank 3 makes `dims[2]` the band count**, and the file holds the
//!   planes one after another, so the load de-planarises them into libviprs's
//!   interleaved layout.
//! * **One variable, and no way to pick it.** `read_new` keeps the first
//!   variable whose rank is 1, 2 or 3, and every other variable in the file
//!   is invisible. `matload` has exactly one argument, the filename.
//! * **The class check runs after the search, not inside it.** Measured on
//!   `int64_then_uint8.mat`: a loadable `uint8` variable behind an `int64`
//!   one fails outright, because the loop breaks on the first variable of
//!   the right *rank* and the class is not looked at until afterwards. A
//!   port that filtered on class inside the loop would load that file and
//!   disagree.
//! * **The byte order is the file's own.** MAT-5 declares it in the header
//!   and every multi-byte field follows, header and samples alike. There is
//!   no byte-swap anywhere in `matlab.c`: `matio` swaps and hands vips
//!   native-order data.
//! * **The logical flag is ignored.** A logical array loads as whatever its
//!   storage class is, measured on `logical_uint8.mat`.
//! * **Where read-info stops and read-data begins.** `Mat_VarReadNextInfo`
//!   validates the array-flags, dimensions and name subelements and does
//!   **not** look at the data subelement. Measured both ways:
//!   `tag_overruns_file.mat` reports `no matrix variables` because the flags
//!   subelement cannot be read at all, while `truncated.mat` loads a
//!   *header* reporting the full 3x2 and only fails at the pixels.
//!
//! # Deliberate divergences
//!
//! * **Complex arrays are refused.** `matload.c:158` says it will not handle
//!   them and there is no check anywhere, so `mat2vips_get_data` memcpys out
//!   of a `mat_complex_split_t` and hands back the raw bytes of two heap
//!   addresses. The capture recorded three runs of one file giving three
//!   different values under ASLR. This module refuses with
//!   [`MatError::ComplexArray`].
//! * **A non-positive dimension is refused.** In vips it reaches
//!   `vips_image_init_fields`, GObject rejects it, the property stays at 1
//!   and the load carries on and exits 0 reading data that is not there.
//!   [`MatError::NonPositiveDimension`] instead.
//! * **The carrier ceiling.** vips loads eight classes; libviprs has
//!   carriers for three. `mxINT8`, `mxINT16` and `mxINT32` need issue #516,
//!   `mxUINT32` needs #517, and `mxDOUBLE`, which is what MATLAB writes by
//!   default, needs the `f64` carrier of #518. All five are refused **by
//!   name** rather than narrowed, exactly as [`crate::fits`] refuses a
//!   signed BITPIX. This is the third format in a row where the missing
//!   carriers rather than the container are what limit the port, which is
//!   worth weighing when #607's sample-kind spine is scheduled.
//! * **The stored element type has to match the class.** `matio` converts a
//!   narrower storage type up to the class type on read; this module refuses
//!   with [`MatError::StorageTypeMismatch`] rather than carry an unmeasured
//!   conversion table. MATLAB only writes a narrowed storage type for
//!   `double` arrays holding small integers, and `double` has no carrier
//!   here anyway, so the refusal is unreachable for the three classes that
//!   do load.
//! * **Band counts other than 1, 3 and 4 are refused.** vips gives a rank-3
//!   array `dims[2]` bands whatever that is. [`PixelFormat`]'s multiband
//!   carriers are documented as compute intermediates that the decode path
//!   does not produce, so this module refuses with
//!   [`MatError::BandCount`] rather than widen that claim.
//!
//! # Decode limits
//!
//! This is the decompression-bomb shape [`DecodeLimits`] exists for, twice
//! over: `dims_100000x100000.mat` declares a 10-gigapixel image behind eight
//! bytes of data and vips reports it at header time without complaint, and
//! `miCOMPRESSED` is a zlib stream whose inflated size is not declared
//! anywhere. So the declared geometry is priced through
//! `DecodeLimits::check_coord`, `check_pixels` and `check_image_alloc`
//! before anything is reserved, and
//! every inflate is capped at [`DecodeLimits::max_alloc_bytes`] and refused
//! rather than grown past it.
//!
//! `flate2` is already a **required** dependency of this crate (it is in
//! `[dependencies]`, not behind a feature), so the `miCOMPRESSED` half costs
//! no new dependency. Nothing else here needs one: the container is a tag
//! loop and a copy.
//!
//! Every entry point here is fallible and there is no panicking twin,
//! matching [`crate::fits`], [`crate::nifti`] and the rest of the codec
//! surface.

use std::borrow::Cow;
use std::io::Read;
use std::num::NonZeroU16;

use thiserror::Error;

use crate::conversion::Interpretation;
use crate::imageio::MetadataValue;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use crate::source::{DecodeLimits, SourceError};

/// Bytes in the MAT-5 file header, and the number the shipped 8.18.6 sniff
/// insists on reading before it will look at anything.
pub const HEADER_BYTES: usize = 128;

/// The case-sensitive text prefix every MAT-5 file opens with.
pub(crate) const MAGIC_PREFIX: &[u8; 10] = b"MATLAB 5.0";

/// Where the version word and the endian indicator sit, as one four-byte
/// field: version at 124..126, indicator at 126..128.
pub(crate) const VERSION_INDICATOR_AT: usize = 124;

/// Version `0x0100` and the little-endian indicator, as the four bytes a
/// little-endian MAT-5 writer puts at [`VERSION_INDICATOR_AT`].
pub(crate) const VERSION_INDICATOR_LE: &[u8; 4] = b"\x00\x01IM";

/// Version `0x0100` and the big-endian indicator, as the four bytes a
/// big-endian MAT-5 writer puts at [`VERSION_INDICATOR_AT`].
pub(crate) const VERSION_INDICATOR_BE: &[u8; 4] = b"\x01\x00MI";

/// The version word every MAT-5 file carries, read in the order the endian
/// indicator declares.
pub(crate) const VERSION_WORD: u16 = 0x0100;

/// The highest rank `read_new` will keep. A variable with more dimensions is
/// skipped and the search carries on.
pub const MAX_RANK: usize = 3;

/// Errors from the MATLAB loader.
///
/// Every variant except [`MatError::Raster`] describes a specific
/// malformation in, or a specific limit of, untrusted bytes. The allocation
/// refusal is deliberately not here: it is
/// [`SourceError::AllocLimitExceeded`], the one shape issue #686 collapsed
/// five per-format variants onto.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MatError {
    /// Fewer than [`HEADER_BYTES`] bytes, so the sniff cannot even run.
    ///
    /// Measured: the shipped sniff asks `vips__get_bytes` for 128 and
    /// requires all 128, so a 127-byte file is refused before the prefix
    /// comparison. `nine_bytes.mat` is the fixture.
    #[error("mat: need at least {needed} bytes for a MAT-5 header, found {found}")]
    ShortHeader {
        /// Bytes available.
        found: usize,
        /// Bytes the header takes.
        needed: usize,
    },
    /// Bytes 0..10 are not `MATLAB 5.0`.
    ///
    /// Measured refused: `MATLAB 5.1`, `matlab 5.0`, `MATLAB_5.0`,
    /// `MATLAB 7.3` and a MAT-4 file, whose first bytes are a dimension
    /// count rather than text.
    #[error("mat: the header opens with {found:?} rather than \"MATLAB 5.0\"")]
    BadMagic {
        /// The first ten bytes, lossily decoded for the message.
        found: String,
    },
    /// Bytes 126..128 are neither `IM` nor `MI`.
    #[error("mat: the endian indicator is {found:?}, which is neither \"IM\" nor \"MI\"")]
    BadEndianIndicator {
        /// The two bytes, lossily decoded for the message.
        found: String,
    },
    /// The version word is not `0x0100`.
    ///
    /// `0x0200` is what a MAT-7.3 file carries, and it is refused here as
    /// well as by the prefix, which is the capture's "the sniff says no
    /// twice over".
    #[error("mat: the header declares version {found:#06x} rather than {expected:#06x}")]
    BadVersion {
        /// The version as read in the declared byte order.
        found: u16,
        /// The only version MAT-5 has.
        expected: u16,
    },
    /// The element sequence held no variable with a rank in `1..=3`.
    ///
    /// This is `matlab.c:120`'s own message, and the capture shows it
    /// reached five different ways: an empty file body, a variable whose
    /// array-flags subelement cannot be read, a corrupt `miCOMPRESSED`
    /// stream, a rank-4 variable with nothing after it, and a MAT-7.3 file
    /// handed straight to `matload`.
    #[error("mat: no matrix variables with a rank between 1 and {max}")]
    NoMatrixVariables {
        /// The highest rank the search keeps.
        max: usize,
    },
    /// A class `vips` refuses too, with `unsupported class type %d`.
    ///
    /// `mat2vips_formats` is an eight-entry table and `mat2vips_get_header`
    /// refuses anything outside it. Measured for `mxINT64` (14),
    /// `mxUINT64` (15), `mxCHAR` (4) and `mxSPARSE` (5).
    #[error("mat: class {class} ({name}) is not one matload reads either")]
    UnsupportedClass {
        /// The class byte from the array-flags word.
        class: u8,
        /// MATLAB's own name for it.
        name: &'static str,
    },
    /// A class `vips` loads but libviprs has no sample carrier for.
    ///
    /// The same ceiling [`crate::fits::FitsError::UnsupportedCarrier`] and
    /// [`crate::nifti::NiftiError::UnsupportedCarrier`] describe, reached
    /// from a third table. `mxDOUBLE` is the one a caller will meet first,
    /// because it is what MATLAB writes unless it is told otherwise.
    #[error(
        "mat: class {class} ({name}) carries {sample} samples, which libviprs has no \
         pixel format for yet (issue #{issue})"
    )]
    UnsupportedCarrier {
        /// The class byte from the array-flags word.
        class: u8,
        /// MATLAB's own name for it.
        name: &'static str,
        /// The sample kind it needs, in words.
        sample: &'static str,
        /// The issue that would add the carrier.
        issue: u32,
    },
    /// The array-flags word sets the complex bit.
    ///
    /// A deliberate divergence: vips never reads this bit and hands back the
    /// bytes of two heap pointers as pixels. See the module doc.
    #[error(
        "mat: this {name} array is complex, and matload reads a complex array's heap \
         pointers out as pixels rather than refusing it"
    )]
    ComplexArray {
        /// The class byte from the array-flags word.
        class: u8,
        /// MATLAB's own name for it.
        name: &'static str,
    },
    /// A declared dimension that is zero or negative.
    ///
    /// A deliberate divergence: vips clamps it to 1 through GObject's
    /// property range check and carries on. See the module doc.
    #[error("mat: dims[{axis}] is {found}, and every dimension has to be positive")]
    NonPositiveDimension {
        /// Which dimension.
        axis: usize,
        /// The value as declared.
        found: i32,
    },
    /// A rank-3 array whose third dimension is a band count libviprs has no
    /// pixel format for.
    #[error(
        "mat: dims[2] is {bands}, and libviprs decodes 1, 3 and 4 bands; the multiband \
         carriers are compute intermediates the decode path does not produce"
    )]
    BandCount {
        /// The band count as declared.
        bands: u32,
    },
    /// The data subelement's type is not the one the class stores.
    ///
    /// `matio` converts a narrower storage type up to the class type on
    /// read. This module does not, because nothing in the capture measures
    /// the conversion, and for the three classes that load here MATLAB never
    /// writes a narrowed one.
    #[error(
        "mat: this {class_name} array stores its samples as {found_name} ({found}) rather \
         than {expected_name} ({expected}), and libviprs does not widen between them"
    )]
    StorageTypeMismatch {
        /// MATLAB's own name for the array class.
        class_name: &'static str,
        /// The element type the class stores.
        expected: u32,
        /// Its name.
        expected_name: &'static str,
        /// The element type actually found.
        found: u32,
        /// Its name, or `"unknown"`.
        found_name: &'static str,
    },
    /// The data subelement is shorter than the declared geometry needs.
    ///
    /// Measured three ways, all of which reach `Mat_VarReadDataAll failed`
    /// in vips at first pixel rather than at header time: a file truncated
    /// mid-element, a 100x100 geometry with four bytes behind it, and a
    /// 100000x100000 one with eight.
    #[error("mat: the sample array needs {needed} bytes, found {found}")]
    TruncatedData {
        /// Bytes available in the data subelement.
        found: u64,
        /// Bytes the declared geometry needs.
        needed: u64,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// Which way round every multi-byte field in the file is.
///
/// Declared by the file itself, in the two indicator bytes at 126..128.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Endian {
    Little,
    Big,
}

impl Endian {
    fn u16(self, b: [u8; 2]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes(b),
            Self::Big => u16::from_be_bytes(b),
        }
    }

    fn u32(self, b: [u8; 4]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes(b),
            Self::Big => u32::from_be_bytes(b),
        }
    }

    fn i32(self, b: [u8; 4]) -> i32 {
        match self {
            Self::Little => i32::from_le_bytes(b),
            Self::Big => i32::from_be_bytes(b),
        }
    }
}

// ---------------------------------------------------------------------------
// The class and element-type tables
// ---------------------------------------------------------------------------

/// The sample carrier a MATLAB class lands on, and how wide one sample is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    /// `mxUINT8`, one unsigned byte per sample.
    U8,
    /// `mxUINT16`, one unsigned 16-bit sample.
    U16,
    /// `mxSINGLE`, one 32-bit float per sample.
    F32,
}

impl Carrier {
    /// Bytes in one sample.
    const fn sample_bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }

    /// The `miTYPE` an array of this class stores its samples as.
    const fn storage_type(self) -> u32 {
        match self {
            Self::U8 => MI_UINT8,
            Self::U16 => MI_UINT16,
            Self::F32 => MI_SINGLE,
        }
    }

    /// The [`PixelFormat`] a raster of this carrier and band count gets.
    ///
    /// Written out per pair rather than reached through
    /// [`PixelFormat::with_channels`], because that constructor also answers
    /// for the multiband carriers this module deliberately does not produce,
    /// so routing through it would turn [`MatError::BandCount`] into
    /// unreachable code. `the_carrier_table_agrees_with_pixel_format` checks
    /// the two against each other instead.
    fn pixel_format(self, bands: u32) -> Option<PixelFormat> {
        Some(match (self, bands) {
            (Self::U8, 1) => PixelFormat::Gray8,
            (Self::U8, 3) => PixelFormat::Rgb8,
            (Self::U8, 4) => PixelFormat::Rgba8,
            (Self::U16, 1) => PixelFormat::Gray16,
            (Self::U16, 3) => PixelFormat::Rgb16,
            (Self::U16, 4) => PixelFormat::Rgba16,
            (Self::F32, 1) => PixelFormat::FloatF32(NonZeroU16::MIN),
            (Self::F32, 3) => PixelFormat::FloatF32(NonZeroU16::new(3).expect("three is not zero")),
            (Self::F32, 4) => PixelFormat::RgbaF32,
            _ => return None,
        })
    }

    /// Every carrier, for the tests that sweep the table.
    #[cfg(test)]
    const ALL: [Self; 3] = [Self::U8, Self::U16, Self::F32];

    /// The interpretation tag a raster of this carrier and band count gets.
    ///
    /// `mat2vips_pick_interpretation` (`matlab.c:160-178`), read off the
    /// measured headers rather than off the source, because the last two of
    /// its branches are the same answer. The case worth naming is the
    /// one-band 8-bit array: it gets `multiband`, **not** `b-w`, so a
    /// greyscale `uint8` array loads untagged. `analyzeload`, the other
    /// loader in this pair, tags the same shape `b-w`.
    const fn interpretation(self, bands: u32) -> Interpretation {
        match (self, bands) {
            (Self::U8, 3) => Interpretation::Srgb,
            (Self::U16, 3) => Interpretation::Rgb16,
            (Self::U16, 1) => Interpretation::Grey16,
            _ => Interpretation::Multiband,
        }
    }
}

/// `miINT8`, the element type an array name is stored as.
const MI_INT8: u32 = 1;
/// `miUINT8`.
const MI_UINT8: u32 = 2;
/// `miINT16`.
const MI_INT16: u32 = 3;
/// `miUINT16`.
const MI_UINT16: u32 = 4;
/// `miINT32`, the element type the dimensions array is stored as.
const MI_INT32: u32 = 5;
/// `miUINT32`, the element type the array-flags word is stored as.
const MI_UINT32: u32 = 6;
/// `miSINGLE`.
const MI_SINGLE: u32 = 7;
/// `miDOUBLE`.
const MI_DOUBLE: u32 = 9;
/// `miINT64`.
const MI_INT64: u32 = 12;
/// `miUINT64`.
const MI_UINT64: u32 = 13;
/// `miMATRIX`, an array.
const MI_MATRIX: u32 = 14;
/// `miCOMPRESSED`, a zlib stream holding one element.
const MI_COMPRESSED: u32 = 15;

/// The `miTYPE` name for a message, or `"unknown"`.
const fn element_type_name(kind: u32) -> &'static str {
    match kind {
        MI_INT8 => "miINT8",
        MI_UINT8 => "miUINT8",
        MI_INT16 => "miINT16",
        MI_UINT16 => "miUINT16",
        MI_INT32 => "miINT32",
        MI_UINT32 => "miUINT32",
        MI_SINGLE => "miSINGLE",
        MI_DOUBLE => "miDOUBLE",
        MI_INT64 => "miINT64",
        MI_UINT64 => "miUINT64",
        MI_MATRIX => "miMATRIX",
        MI_COMPRESSED => "miCOMPRESSED",
        _ => "unknown",
    }
}

/// The complex bit of the array-flags word's flags byte.
const FLAG_COMPLEX: u8 = 0x08;

/// The logical bit of the array-flags word's flags byte.
///
/// Read only so the module can say it is ignored: measured, a logical array
/// loads as whatever its storage class is.
const FLAG_LOGICAL: u8 = 0x02;

/// Resolve a MATLAB class byte onto a carrier, or say precisely why not.
///
/// Three answers, and the difference between the last two is the whole point:
/// [`MatError::UnsupportedClass`] is a class `vips` refuses too, and
/// [`MatError::UnsupportedCarrier`] is one `vips` loads and libviprs cannot
/// yet, named with the issue that would lift it.
fn carrier_for(class: u8) -> Result<Carrier, MatError> {
    let ceiling = |name, sample, issue| {
        Err(MatError::UnsupportedCarrier {
            class,
            name,
            sample,
            issue,
        })
    };
    match class {
        9 => Ok(Carrier::U8),
        11 => Ok(Carrier::U16),
        7 => Ok(Carrier::F32),
        8 => ceiling("mxINT8", "signed 8-bit integer", 516),
        10 => ceiling("mxINT16", "signed 16-bit integer", 516),
        12 => ceiling("mxINT32", "signed 32-bit integer", 516),
        13 => ceiling("mxUINT32", "unsigned 32-bit integer", 517),
        6 => ceiling("mxDOUBLE", "64-bit float", 518),
        other => Err(MatError::UnsupportedClass {
            class: other,
            name: class_name(other),
        }),
    }
}

/// MATLAB's own name for a class byte, or `"unknown"`.
const fn class_name(class: u8) -> &'static str {
    match class {
        1 => "mxCELL",
        2 => "mxSTRUCT",
        3 => "mxOBJECT",
        4 => "mxCHAR",
        5 => "mxSPARSE",
        6 => "mxDOUBLE",
        7 => "mxSINGLE",
        8 => "mxINT8",
        9 => "mxUINT8",
        10 => "mxINT16",
        11 => "mxUINT16",
        12 => "mxINT32",
        13 => "mxUINT32",
        14 => "mxINT64",
        15 => "mxUINT64",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// The 128-byte header
// ---------------------------------------------------------------------------

/// What the MAT-5 file header carries, once the sniff predicate has passed.
#[derive(Clone, Debug)]
pub(crate) struct Header {
    /// The byte order the endian indicator declares.
    pub(crate) endian: Endian,
    /// Bytes 0..116, the free-text description, trimmed and lossily decoded.
    description: String,
}

impl Header {
    /// Parse and validate the 128-byte header, applying the shipped 8.18.6
    /// sniff predicate one clause at a time.
    ///
    /// The clauses are in the order the disassembly runs them: length, then
    /// prefix, then the indicator, then the version. That order is what
    /// decides which error a file with more than one problem reports, and
    /// `magic_only.mat` is the fixture that has two.
    fn parse(bytes: &[u8]) -> Result<Self, MatError> {
        if bytes.len() < HEADER_BYTES {
            return Err(MatError::ShortHeader {
                found: bytes.len(),
                needed: HEADER_BYTES,
            });
        }
        if &bytes[..MAGIC_PREFIX.len()] != MAGIC_PREFIX {
            return Err(MatError::BadMagic {
                found: String::from_utf8_lossy(&bytes[..MAGIC_PREFIX.len()]).into_owned(),
            });
        }
        let indicator = &bytes[VERSION_INDICATOR_AT + 2..VERSION_INDICATOR_AT + 4];
        let endian = match indicator {
            b"IM" => Endian::Little,
            b"MI" => Endian::Big,
            other => {
                return Err(MatError::BadEndianIndicator {
                    found: String::from_utf8_lossy(other).into_owned(),
                });
            }
        };
        let version = endian.u16([bytes[VERSION_INDICATOR_AT], bytes[VERSION_INDICATOR_AT + 1]]);
        if version != VERSION_WORD {
            return Err(MatError::BadVersion {
                found: version,
                expected: VERSION_WORD,
            });
        }
        let description = bytes[..VERSION_INDICATOR_AT]
            .iter()
            .position(|&b| b == 0)
            .map_or(&bytes[..VERSION_INDICATOR_AT], |end| &bytes[..end]);
        Ok(Self {
            endian,
            description: String::from_utf8_lossy(description).trim_end().to_owned(),
        })
    }
}

// ---------------------------------------------------------------------------
// The element stream
// ---------------------------------------------------------------------------

/// One data element: its type tag and its payload.
struct Element<'a> {
    /// The `miTYPE` from the tag.
    kind: u32,
    /// The payload, clipped at the end of the buffer it lives in. A tag may
    /// declare more bytes than the file holds, which is exactly what
    /// `truncated.mat` does, and the clip is what lets the flags, dims and
    /// name subelements still parse while the data one comes up short.
    payload: &'a [u8],
    /// Where the element after this one starts, in the same buffer.
    next: usize,
}

/// Read the element whose tag starts at `at`, or `None` if there is not even
/// a tag left.
///
/// Both tag forms are here. In the small form the byte count rides in the
/// tag's upper half and the payload is the next four bytes, so the whole
/// element is eight bytes; in the regular form the count is its own word and
/// the payload is padded up to the next eight-byte boundary.
fn read_element(buf: &[u8], at: usize, endian: Endian) -> Option<Element<'_>> {
    let word = endian.u32(buf.get(at..at + 4)?.try_into().ok()?);
    let small = (word >> 16) != 0;
    let (kind, declared, from) = if small {
        (word & 0xffff, (word >> 16) as usize, at + 4)
    } else {
        let count = endian.u32(buf.get(at + 4..at + 8)?.try_into().ok()?);
        (word, count as usize, at + 8)
    };
    // A small element is always eight bytes; a regular one pads its payload
    // up to eight. `next` may run past the buffer, which ends the walk.
    let next = if small {
        at + 8
    } else {
        at.saturating_add(8)
            .saturating_add(declared.next_multiple_of(8))
    };
    let end = from.saturating_add(declared).min(buf.len());
    let payload = buf.get(from..end).unwrap_or(&[]);
    Some(Element {
        kind,
        payload,
        next,
    })
}

/// Everything `Mat_VarReadNextInfo` reads out of one `miMATRIX` element.
struct Variable<'a> {
    class: u8,
    flags: u8,
    dims: Vec<i32>,
    name: String,
    /// The element body, borrowed from the file or owned after an inflate.
    body: Cow<'a, [u8]>,
    /// Where the data subelement's tag starts inside `body`.
    data_at: usize,
}

impl Variable<'_> {
    /// Whether the array-flags word sets the complex bit.
    const fn complex(&self) -> bool {
        self.flags & FLAG_COMPLEX != 0
    }

    /// Whether the array-flags word sets the logical bit, which the loader
    /// ignores.
    const fn logical(&self) -> bool {
        self.flags & FLAG_LOGICAL != 0
    }
}

/// Read the array-flags, dimensions and name subelements of a `miMATRIX`
/// body, exactly the three `Mat_VarReadNextInfo` validates.
///
/// `None` means the variable is unreadable, which ends the search the way a
/// `NULL` from `Mat_VarReadNextInfo` does. The data subelement is **not**
/// looked at: that is the give-up point the capture measures, and it is why
/// `truncated.mat` reports a full 3x2 header and fails only at the pixels.
fn read_info(body: Cow<'_, [u8]>, endian: Endian) -> Option<Variable<'_>> {
    let flags_el = read_element(&body, 0, endian)?;
    if flags_el.kind != MI_UINT32 || flags_el.payload.len() < 8 {
        return None;
    }
    let word = endian.u32(flags_el.payload[..4].try_into().ok()?);
    let class = (word & 0xff) as u8;
    let flags = ((word >> 8) & 0xff) as u8;

    let dims_el = read_element(&body, flags_el.next, endian)?;
    if dims_el.kind != MI_INT32 || dims_el.payload.is_empty() || dims_el.payload.len() % 4 != 0 {
        return None;
    }
    let dims: Vec<i32> = dims_el
        .payload
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| endian.i32(*c))
        .collect();

    let name_el = read_element(&body, dims_el.next, endian)?;
    if name_el.kind != MI_INT8 {
        return None;
    }
    let name = String::from_utf8_lossy(name_el.payload).into_owned();
    let data_at = name_el.next;
    Some(Variable {
        class,
        flags,
        dims,
        name,
        body,
        data_at,
    })
}

/// Inflate a `miCOMPRESSED` payload, refusing rather than growing past
/// [`DecodeLimits::max_alloc_bytes`].
///
/// The inflated size is not declared anywhere in the container, so this is
/// the one place a MAT file can ask for an unbounded allocation, and the
/// only defence is to **stop reading**. The reader is capped at the budget
/// plus one byte, and a stream that reaches the cap is refused with the
/// shared [`SourceError::AllocLimitExceeded`] rather than being truncated
/// into a shorter, wrong array.
///
/// So the price the refusal reports is `max_alloc_bytes + 1`, which is a
/// floor rather than the stream's real inflated size: the whole point is
/// that nothing here ever learns what that size was. That is deliberate and
/// it is what makes the cap observable from outside. Checking the length
/// *after* an uncapped read would report the true size and refuse the same
/// file, and would also have allocated the bomb first;
/// `a_compressed_element_that_inflates_past_the_budget_is_refused` asserts
/// the reported price is exactly the budget plus one for that reason.
fn inflate(payload: &[u8], limits: DecodeLimits) -> Result<Option<Vec<u8>>, SourceError> {
    let cap = limits.max_alloc_bytes;
    let mut out = Vec::new();
    let taken = flate2::read::ZlibDecoder::new(payload).take(cap.saturating_add(1));
    if std::io::BufReader::new(taken)
        .read_to_end(&mut out)
        .is_err()
    {
        // A corrupt stream is not an error: `Mat_VarReadNextInfo` returns
        // NULL and the search ends, which is `no matrix variables`.
        // Measured on `compressed_corrupt.mat`.
        return Ok(None);
    }
    limits.check_alloc("MAT compressed element", out.len() as u64)?;
    Ok(Some(out))
}

/// Walk the element stream and keep the first variable whose rank is in
/// `1..=`[`MAX_RANK`].
///
/// This is `read_new` (`matlab.c:118-140`): every variable outside that rank
/// range is freed and the search carries on, and the class is not looked at
/// here at all. A variable that cannot be read ends the search, because that
/// is what a `NULL` from `Mat_VarReadNextInfo` does.
fn find_variable(
    bytes: &[u8],
    endian: Endian,
    limits: DecodeLimits,
) -> Result<Variable<'_>, SourceError> {
    let mut at = HEADER_BYTES;
    while let Some(element) = read_element(bytes, at, endian) {
        at = element.next;
        let body: Cow<'_, [u8]> = match element.kind {
            MI_MATRIX => Cow::Borrowed(element.payload),
            MI_COMPRESSED => match inflate(element.payload, limits)? {
                Some(inflated) => Cow::Owned(inflated),
                None => break,
            },
            // Anything else at the top level is not a variable. matio walks
            // past it; nothing in the capture produces one.
            _ => continue,
        };
        // A `miCOMPRESSED` holds exactly one element, which is the matrix.
        let body = if element.kind == MI_COMPRESSED {
            match read_element(&body, 0, endian) {
                Some(inner) if inner.kind == MI_MATRIX => Cow::Owned(inner.payload.to_vec()),
                _ => break,
            }
        } else {
            body
        };
        let Some(variable) = read_info(body, endian) else {
            break;
        };
        if (1..=MAX_RANK).contains(&variable.dims.len()) {
            return Ok(variable);
        }
    }
    Err(MatError::NoMatrixVariables { max: MAX_RANK }.into())
}

/// The width, height and band count a variable's dimensions give.
///
/// `dims[0]` is the height and `dims[1]` the width, which is the transpose,
/// and `dims[2]` is the band count. A rank-1 variable keeps the width of 1
/// that `mat2vips_get_header` initialises (`matlab.c:188`), so it is a
/// one-pixel-wide column.
fn geometry(dims: &[i32]) -> Result<(u32, u32, u32), MatError> {
    for (axis, &extent) in dims.iter().enumerate() {
        if extent <= 0 {
            return Err(MatError::NonPositiveDimension {
                axis,
                found: extent,
            });
        }
    }
    let height = dims[0] as u32;
    let width = dims.get(1).map_or(1, |&d| d as u32);
    let bands = dims.get(2).map_or(1, |&d| d as u32);
    Ok((width, height, bands))
}

/// Decode a MATLAB level 5 (`.mat`) buffer into a [`Raster`].
///
/// The first variable with a rank of 1, 2 or 3 becomes the image, its
/// `dims[0]` becomes the height and its `dims[1]` the width, and a `dims[2]`
/// becomes the band count. See the module doc for the whole measured
/// contract, including the four places this deliberately diverges from
/// `matload`.
///
/// # Errors
///
/// * [`MatError`] for every malformation the container carries, each named:
///   a short file, a bad prefix, a bad indicator or version, no loadable
///   variable, a class with no carrier, a complex array, a non-positive
///   dimension, a band count with no pixel format, a storage type that does
///   not match the class, and a data subelement shorter than the geometry
///   needs.
/// * [`SourceError::AllocLimitExceeded`] when the declared geometry, or a
///   `miCOMPRESSED` element's inflated size, prices past
///   [`DecodeLimits::max_alloc_bytes`].
/// * [`SourceError::DimensionLimitExceeded`] or
///   [`SourceError::CoordLimitExceeded`] when the declared geometry is over
///   [`DecodeLimits::max_pixels`] or [`DecodeLimits::max_coord`].
///
/// # Example
///
/// ```no_run
/// use libviprs::mat::decode_mat;
/// use libviprs::source::DecodeLimits;
///
/// let bytes = std::fs::read("scan.mat")?;
/// let raster = decode_mat(&bytes, DecodeLimits::default())?;
/// println!("{}x{}", raster.width(), raster.height());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn decode_mat(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let header = Header::parse(bytes)?;
    let endian = header.endian;
    let variable = find_variable(bytes, endian, limits)?;

    // The complex check comes first, before the carrier, so the capture's
    // one complex fixture reports the divergence this module exists to make
    // rather than the `mxDOUBLE` ceiling it also happens to trip.
    if variable.complex() {
        return Err(MatError::ComplexArray {
            class: variable.class,
            name: class_name(variable.class),
        }
        .into());
    }
    let carrier = carrier_for(variable.class)?;
    let (width, height, bands) = geometry(&variable.dims)?;
    let format = carrier
        .pixel_format(bands)
        .ok_or(MatError::BandCount { bands })?;

    // All three ceilings go on the declared header geometry, before anything
    // is reserved: `dims_100000x100000.mat` declares ten gigapixels behind
    // eight bytes of data and vips reports it without complaint.
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    // One spelling of the budget for the whole crate (issue #632): the price
    // comes from `decode_alloc_bytes` and the comparison from
    // `DecodeLimits::exceeds_alloc_budget`, and the refusal is the shared
    // `SourceError::AllocLimitExceeded` (issue #686).
    let needed = limits.check_image_alloc(
        "MAT sample buffer",
        width,
        height,
        u64::from(bands),
        carrier.sample_bytes(),
    )?;

    let data_el = read_element(&variable.body, variable.data_at, endian)
        .ok_or(MatError::TruncatedData { found: 0, needed })?;
    if data_el.kind != carrier.storage_type() {
        return Err(MatError::StorageTypeMismatch {
            class_name: class_name(variable.class),
            expected: carrier.storage_type(),
            expected_name: element_type_name(carrier.storage_type()),
            found: data_el.kind,
            found_name: element_type_name(data_el.kind),
        }
        .into());
    }
    let available = data_el.payload.len() as u64;
    if available < needed {
        return Err(MatError::TruncatedData {
            found: available,
            needed,
        }
        .into());
    }

    let data = deplanarise(
        &data_el.payload[..needed as usize],
        width,
        height,
        bands,
        carrier,
        endian,
    );
    let mut raster = Raster::new_with_budget(width, height, format, data, limits.max_alloc_bytes)
        .map_err(MatError::Raster)?;
    raster.meta.interpretation = Some(carrier.interpretation(bands));
    attach(&header, &variable, &mut raster);
    Ok(raster)
}

/// Turn the file's column-major, plane-separate array into libviprs's
/// row-major, interleaved, native-endian buffer.
///
/// Two rearrangements at once, and both are measured. `mat2vips_get_data`
/// (`matlab.c:276-300`) walks the column-major buffer with a stride of
/// `es * Ysize` to build each scanline, which is the transpose, and offsets
/// each band by `b * es * N_PELS` (`matlab.c:286`), which is the
/// de-planarisation the comment at `matlab.c:258-259` calls out.
///
/// So sample `(x, y, b)` of the output is sample `b * h * w + y + x * h` of
/// the input.
fn deplanarise(
    payload: &[u8],
    width: u32,
    height: u32,
    bands: u32,
    carrier: Carrier,
    endian: Endian,
) -> Vec<u8> {
    let sample = carrier.sample_bytes() as usize;
    let (w, h, b) = (width as usize, height as usize, bands as usize);
    let plane = h * w;
    let mut out = Vec::with_capacity(plane * b * sample);
    for y in 0..h {
        for x in 0..w {
            for band in 0..b {
                let from = (band * plane + y + x * h) * sample;
                out.extend_from_slice(&to_native(&payload[from..from + sample], carrier, endian));
            }
        }
    }
    out
}

/// One sample, in the host's byte order.
///
/// `matio` swaps where it needs to and hands vips native-order data, so
/// there is no byte-swap anywhere in `matlab.c` and the swap has to live
/// here instead.
fn to_native(sample: &[u8], carrier: Carrier, endian: Endian) -> Vec<u8> {
    match carrier {
        // One byte per sample: byte order cannot apply, whatever the header
        // says about the rest of the file.
        Carrier::U8 => sample.to_vec(),
        Carrier::U16 => {
            let raw: [u8; 2] = sample.try_into().expect("a U16 sample is two bytes");
            endian.u16(raw).to_ne_bytes().to_vec()
        }
        Carrier::F32 => {
            let raw: [u8; 4] = sample.try_into().expect("an F32 sample is four bytes");
            f32::from_bits(endian.u32(raw)).to_ne_bytes().to_vec()
        }
    }
}

/// Attach what the container carried, so the variable that was chosen and
/// the dimensions it declared are still readable.
///
/// `matload` attaches nothing at all: `vipsheader -a` on a loaded `.mat`
/// shows only the standard fields, so there is no oracle value to disagree
/// with here. These are libviprs's own, named `mat-<field>` the way
/// [`crate::nifti`] names its `nifti-<field>`, and they exist because
/// `matload` picks one variable out of a file with no way to ask for
/// another and then loses its name.
fn attach(header: &Header, variable: &Variable<'_>, raster: &mut Raster) {
    raster
        .fields
        .set("mat-name", MetadataValue::Str(variable.name.clone()));
    raster.fields.set(
        "mat-class",
        MetadataValue::Str(class_name(variable.class).to_owned()),
    );
    raster.fields.set(
        "mat-endian",
        MetadataValue::Str(
            match header.endian {
                Endian::Little => "IM",
                Endian::Big => "MI",
            }
            .to_owned(),
        ),
    );
    raster.fields.set(
        "mat-description",
        MetadataValue::Str(header.description.clone()),
    );
    raster.fields.set(
        "mat-logical",
        MetadataValue::Int(i64::from(variable.logical())),
    );
    for (axis, &extent) in variable.dims.iter().enumerate() {
        raster.fields.set(
            &format!("mat-dims[{axis}]"),
            MetadataValue::Int(extent.into()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::decode_bytes_with_limits;

    /// One of the 48 fixtures `oracle-captures/foreign-mat/capture.py`
    /// writes and pins, embedded rather than read.
    ///
    /// `include_bytes!` rather than `std::fs::read` on purpose: it keeps
    /// every test in this module runnable under Miri, where the isolation
    /// layer refuses a real `open` and aborts the whole session on the first
    /// one it meets (issue #652). The whole fixture set is 9,695 bytes, so
    /// embedding the dozen these tests name costs nothing.
    macro_rules! fixture {
        ($name:literal) => {
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/oracle-captures/foreign-mat/fixtures/",
                $name
            ))
            .as_slice()
        };
    }

    fn decoded(bytes: &[u8]) -> Raster {
        decode_mat(bytes, DecodeLimits::default())
            .unwrap_or_else(|e| panic!("this fixture should decode: {e}"))
    }

    fn refused(bytes: &[u8]) -> SourceError {
        decode_mat(bytes, DecodeLimits::default()).expect_err("this fixture should be refused")
    }

    /// Read a raster back as `u16` samples, whatever the host's byte order.
    fn u16_samples(raster: &Raster) -> Vec<u16> {
        raster
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .copied()
            .map(u16::from_ne_bytes)
            .collect()
    }

    /// Read a raster back as `f32` samples, whatever the host's byte order.
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
     * The transpose, which is the single easiest thing to get backwards in a
     * MAT port and is why every value in this fixture is asymmetric.
     * `mat2vips_get_header` (`matlab.c:190-205`) reads the height from
     * `dims[0]` and the width from `dims[1]`, so a MATLAB 2x3 becomes a
     * 3x2 image and element `(r, c)` is pixel `(c, r)`.
     * Measured: `oracle-captures/foreign-mat`'s `column_major_is_transposed`
     * record, `fixtures/base_2x3_uint8.mat`, whose file order is
     * `10 40 20 50 30 60` and whose loaded pixels are `10 20 30 40 50 60`.
     * Input: a 2x3 `mxUINT8` array -> Output: a 3x2 `Gray8` raster tagged
     * `multiband`, holding the transpose.
     */
    #[test]
    fn a_two_by_three_uint8_array_loads_transposed() {
        let raster = decoded(fixture!("base_2x3_uint8.mat"));
        assert_eq!((raster.width(), raster.height()), (3, 2));
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(raster.data(), &[10, 20, 30, 40, 50, 60]);
        // The one-band 8-bit case is `multiband`, not `b-w`: measured, and
        // it is where `matload` and `analyzeload` disagree about the same
        // shape.
        assert_eq!(raster.meta.interpretation, Some(Interpretation::Multiband));
    }

    /**
     * The same file through the public content-sniffing entry point, so the
     * route table's MAT row is exercised rather than only `decode_mat`.
     * The two signatures on that row are the whole of the shipped 8.18.6
     * sniff predicate, so this is also what pins that a file which passes it
     * reaches this codec at all.
     * Input: `base_2x3_uint8.mat` through `decode_bytes_with_limits` ->
     * Output: the same 3x2 `Gray8` raster `decode_mat` gives.
     */
    #[test]
    fn a_mat_buffer_reaches_this_codec_through_the_sniff() {
        let bytes = fixture!("base_2x3_uint8.mat");
        let routed = decode_bytes_with_limits(bytes, DecodeLimits::default())
            .expect("the sniff must route a MAT-5 file here");
        let direct = decoded(bytes);
        assert_eq!(
            (routed.width(), routed.height(), routed.format()),
            (direct.width(), direct.height(), direct.format())
        );
        assert_eq!(routed.data(), direct.data());
    }

    /**
     * Rank 3 makes `dims[2]` the band count, and the file holds the planes
     * one after another where a libviprs raster is interleaved, so the load
     * has to de-planarise as well as transpose. `mat2vips_get_data` offsets
     * each band by `b * es * N_PELS` (`matlab.c:286`); the comment at
     * `matlab.c:258-259` calls the mismatch out.
     * Measured: the `rank_and_variable_selection` record's rank-3 case, file
     * order `1..6, 11..16, 21..26`, loaded pixels `[1,11,21] [3,13,23] ...`.
     * Input: a 2x3x3 `mxUINT8` array -> Output: a 3x2 `Rgb8` raster tagged
     * sRGB with the three planes interleaved.
     */
    #[test]
    fn a_rank_three_array_becomes_bands_and_de_planarises() {
        let raster = decoded(fixture!("rank3_2x3x3.mat"));
        assert_eq!((raster.width(), raster.height()), (3, 2));
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(raster.meta.interpretation, Some(Interpretation::Srgb));
        assert_eq!(
            raster.data(),
            &[
                1, 11, 21, 3, 13, 23, 5, 15, 25, // row 0
                2, 12, 22, 4, 14, 24, 6, 16, 26, // row 1
            ]
        );
    }

    /**
     * A rank-1 variable keeps the width of 1 that `mat2vips_get_header`
     * initialises (`matlab.c:188`) and takes its height from `dims[0]`, so
     * it is a one-pixel-wide column rather than a row. MATLAB itself never
     * writes rank 1, which is exactly why an untrusted loader has to survive
     * it.
     * Measured: `rank_and_variable_selection`'s rank-1 case, `1x4 uchar`
     * with pixels `11 22 33 44`.
     * Input: a 4-element `mxUINT8` array -> Output: a 1x4 `Gray8` raster.
     */
    #[test]
    fn a_rank_one_array_is_a_one_pixel_wide_column() {
        let raster = decoded(fixture!("rank1.mat"));
        assert_eq!((raster.width(), raster.height()), (1, 4));
        assert_eq!(raster.data(), &[11, 22, 33, 44]);
    }

    /**
     * `read_new` (`matlab.c:118-140`) frees every variable whose rank is
     * outside `1..=3` and keeps looking, and reports `no matrix variables`
     * if it runs off the end. Both halves are measured and both are here,
     * because the second is the positive control for the first: a rank-4
     * variable alone is a refusal, and the *same* rank-4 variable followed
     * by a rank-2 one loads the rank-2.
     * Measured: the `4_alone` and `4_then_2` cases, the second giving
     * `2x2 uchar` with pixels `7 9 8 10`.
     * Input: `rank4_only.mat` and `rank4_then_rank2.mat` -> Output: a
     * refusal and a 2x2 raster.
     */
    #[test]
    fn the_search_skips_a_variable_of_the_wrong_rank_and_keeps_looking() {
        let alone = refused(fixture!("rank4_only.mat"));
        assert!(
            matches!(
                alone,
                SourceError::Mat(MatError::NoMatrixVariables { max: MAX_RANK })
            ),
            "a lone rank-4 variable must report no matrix variables: {alone:?}"
        );

        let raster = decoded(fixture!("rank4_then_rank2.mat"));
        assert_eq!((raster.width(), raster.height()), (2, 2));
        assert_eq!(raster.data(), &[7, 9, 8, 10]);
        assert_eq!(
            raster.fields.get("mat-name").map(MetadataValue::as_str),
            Some("small"),
            "the second variable is the one that loads, so its name is the one attached"
        );
    }

    /**
     * The class check runs after the search, not inside it. `read_new`'s
     * loop breaks on the first variable of the right *rank* and the class is
     * not looked at until `mat2vips_get_header`, by which point the loop is
     * over. So a file whose first rank-2 variable is `int64` fails outright
     * even though a perfectly loadable `uint8` variable follows it, and a
     * port that filtered on class inside the loop would load it and
     * disagree with vips.
     * The rank test above is the positive control: skipping *does* work,
     * for rank, in a file of exactly the same shape.
     * Measured: `int64_then_uint8.mat`, `mat2vips: unsupported class type 14`.
     * Input: an `int64` rank-2 variable followed by a `uint8` one -> Output:
     * `UnsupportedClass { class: 14 }`, not the second variable.
     */
    #[test]
    fn the_class_check_runs_after_the_search_not_inside_it() {
        let err = refused(fixture!("int64_then_uint8.mat"));
        assert!(
            matches!(
                err,
                SourceError::Mat(MatError::UnsupportedClass {
                    class: 14,
                    name: "mxINT64"
                })
            ),
            "the int64 variable is the one that was chosen: {err:?}"
        );
    }

    /**
     * The whole class table, measured off `vipsheader` rather than off
     * `mat2vips_formats`, split three ways: the three classes libviprs has a
     * carrier for load with their pixels pinned, the five vips loads and
     * libviprs cannot are refused **by name** with the issue that would add
     * the carrier, and the four vips refuses too are refused as the class
     * vips itself calls unsupported.
     *
     * The split between the last two is the point. Narrowing `mxINT16` into
     * eight bits or `mxDOUBLE` into a float would lose data silently, which
     * is worse than failing, and it is the same ceiling
     * [`crate::fits`] and [`crate::nifti`] describe from their own tables.
     * Input: the ten `class_mat_c_*.mat` fixtures -> Output: three rasters
     * with their measured pixels, and seven named refusals.
     */
    #[test]
    fn every_class_lands_on_its_measured_carrier_or_is_refused_by_name() {
        let uint8 = decoded(fixture!("class_mat_c_uint8.mat"));
        assert_eq!(uint8.format(), PixelFormat::Gray8);
        assert_eq!(uint8.data(), &[1, 3, 2, 4]);

        let uint16 = decoded(fixture!("class_mat_c_uint16.mat"));
        assert_eq!(uint16.format(), PixelFormat::Gray16);
        assert_eq!(u16_samples(&uint16), vec![1, 3, 500, 65535]);
        assert_eq!(uint16.meta.interpretation, Some(Interpretation::Grey16));

        let single = decoded(fixture!("class_mat_c_single.mat"));
        assert_eq!(single.format(), PixelFormat::FloatF32(NonZeroU16::MIN));
        assert_eq!(f32_samples(&single), vec![1.5, 3.125, -2.25, 4.0]);

        // The five vips loads and libviprs has no carrier for, each named
        // with the issue that would lift the ceiling.
        let ceilings: [(&[u8], u8, &str, u32); 5] = [
            (fixture!("class_mat_c_int8.mat"), 8, "mxINT8", 516),
            (fixture!("class_mat_c_int16.mat"), 10, "mxINT16", 516),
            (fixture!("class_mat_c_int32.mat"), 12, "mxINT32", 516),
            (fixture!("class_mat_c_uint32.mat"), 13, "mxUINT32", 517),
            (fixture!("class_mat_c_double.mat"), 6, "mxDOUBLE", 518),
        ];
        for (bytes, class, name, issue) in ceilings {
            match refused(bytes) {
                SourceError::Mat(MatError::UnsupportedCarrier {
                    class: got_class,
                    name: got_name,
                    issue: got_issue,
                    ..
                }) => {
                    assert_eq!((got_class, got_name, got_issue), (class, name, issue));
                }
                other => panic!("{name} must be refused by name, got {other:?}"),
            }
        }

        // And the four `mat2vips_get_header` itself refuses with
        // `unsupported class type %d`.
        let unsupported: [(&[u8], u8, &str); 4] = [
            (fixture!("class_mat_c_int64.mat"), 14, "mxINT64"),
            (fixture!("class_mat_c_uint64.mat"), 15, "mxUINT64"),
            (fixture!("class_mat_c_char.mat"), 4, "mxCHAR"),
            (fixture!("class_mat_c_sparse.mat"), 5, "mxSPARSE"),
        ];
        for (bytes, class, name) in unsupported {
            match refused(bytes) {
                SourceError::Mat(MatError::UnsupportedClass {
                    class: got_class,
                    name: got_name,
                }) => assert_eq!((got_class, got_name), (class, name)),
                other => panic!("{name} must report the class vips refuses, got {other:?}"),
            }
        }
    }

    /**
     * `mat2vips_pick_interpretation` (`matlab.c:160-178`) read off the
     * measured headers rather than off the source, because two of its
     * branches are the same answer and the interesting case is the one that
     * looks like an oversight: a one-band 8-bit array gets `multiband`, not
     * `b-w`, so a greyscale `uint8` array loads untagged.
     * Measured: the `three_band_header` strings in `class_to_carrier` and
     * the one-band headers beside them.
     * Input: the three-band and one-band fixtures for each carrier ->
     * Output: sRGB, RGB16, GREY16 and MULTIBAND exactly where measured.
     */
    #[test]
    fn the_interpretation_follows_the_measured_headers() {
        let cases: [(&[u8], PixelFormat, Interpretation); 5] = [
            (
                fixture!("bands3_mat_c_uint8.mat"),
                PixelFormat::Rgb8,
                Interpretation::Srgb,
            ),
            (
                fixture!("bands3_mat_c_uint16.mat"),
                PixelFormat::Rgb16,
                Interpretation::Rgb16,
            ),
            (
                fixture!("bands3_mat_c_single.mat"),
                PixelFormat::FloatF32(NonZeroU16::new(3).unwrap()),
                Interpretation::Multiband,
            ),
            (
                fixture!("class_mat_c_uint16.mat"),
                PixelFormat::Gray16,
                Interpretation::Grey16,
            ),
            (
                fixture!("class_mat_c_uint8.mat"),
                PixelFormat::Gray8,
                Interpretation::Multiband,
            ),
        ];
        for (bytes, format, interpretation) in cases {
            let raster = decoded(bytes);
            assert_eq!(raster.format(), format);
            assert_eq!(raster.meta.interpretation, Some(interpretation));
        }
    }

    /**
     * The array-flags word's logical bit is read and then ignored: the band
     * format follows the class alone, so a logical array loads as whatever
     * its storage class is. Measured on `logical_uint8.mat`, whose flags
     * byte is `0x02` and which loads `2x2 uchar`.
     * The bit is still attached, as `mat-logical`, which is what makes
     * "read and ignored" checkable rather than indistinguishable from "never
     * read".
     * Input: a logical `uint8` array -> Output: a `Gray8` raster with
     * `mat-logical` set.
     */
    #[test]
    fn the_logical_flag_is_read_and_then_ignored() {
        let raster = decoded(fixture!("logical_uint8.mat"));
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(raster.data(), &[0, 1, 1, 0]);
        assert_eq!(
            raster.fields.get("mat-logical").map(MetadataValue::as_i64),
            Some(1)
        );
        let plain = decoded(fixture!("class_mat_c_uint8.mat"));
        assert_eq!(
            plain.fields.get("mat-logical").map(MetadataValue::as_i64),
            Some(0),
            "the control: a non-logical array of the same class reads 0"
        );
    }

    /**
     * MAT-5 declares its own byte order and every multi-byte field follows,
     * samples included. `matio` swaps and hands vips native-order data, so
     * there is no byte-swap anywhere in `matlab.c` and the swap has to live
     * in this module instead.
     *
     * The capture's own byte-order pair is `mxINT16`, which libviprs has no
     * carrier for, so pinning the *header* swap off it is all it can give.
     * That is the trap the NIfTI lane hit: every fixture of a supported
     * class was little-endian and so is the host, so a hard-coded
     * `from_le_bytes` stayed green. Two bytes are poked here to get past it,
     * both stated: the class byte from `mxINT16` (10) to `mxUINT16` (11),
     * and the data element's type from `miINT16` (3) to `miUINT16` (4).
     * Nothing else moves, because the sample width is two either way and the
     * payload is untouched.
     * Input: `endian_little.mat` and `endian_big.mat`, poked -> Output: the
     * same four `u16` samples and byte-for-byte the same buffer.
     */
    #[test]
    fn the_two_byte_orders_load_to_identical_memory() {
        // The offsets are asserted rather than trusted: the class byte is
        // the low byte of the array-flags word and the type is the low byte
        // of the data element's tag, so each sits at the other end of its
        // word in the big-endian file.
        let mut little = fixture!("endian_little.mat").to_vec();
        assert_eq!(little[144], 10, "the little-endian class byte");
        little[144] = 11;
        assert_eq!(little[184], 3, "the little-endian data element type");
        little[184] = 4;

        let mut big = fixture!("endian_big.mat").to_vec();
        assert_eq!(big[147], 10, "the big-endian class byte");
        big[147] = 11;
        assert_eq!(big[187], 3, "the big-endian data element type");
        big[187] = 4;

        // The two files hold the same four shorts written the other way
        // round, and `1` versus `256` is the pair that cannot survive a
        // wrong swap.
        let le = decoded(&little);
        let be = decoded(&big);
        assert_eq!(u16_samples(&le), vec![1, 65535, 256, 4660]);
        assert_eq!(u16_samples(&be), vec![1, 65535, 256, 4660]);
        assert_eq!(le.data(), be.data(), "the two byte orders must agree");
    }

    /**
     * And the header half of the same rule, which the pair above cannot
     * reach because it pokes both files. A big-endian file's array-flags
     * word is `00 00 00 0a`: read the file's own way it is class 10,
     * `mxINT16`, and read little-endian it is class 0, which is not a class
     * at all. The two produce different errors, so the refusal says which
     * way the header was read.
     * Measured: `endian_big.mat` loads in vips as `2x2 short`, i.e. class
     * 10.
     * Input: an untouched `endian_big.mat` -> Output:
     * `UnsupportedCarrier { class: 10 }`, never `UnsupportedClass { class: 0 }`.
     */
    #[test]
    fn a_big_endian_header_is_read_in_its_own_byte_order() {
        let err = refused(fixture!("endian_big.mat"));
        assert!(
            matches!(
                err,
                SourceError::Mat(MatError::UnsupportedCarrier {
                    class: 10,
                    name: "mxINT16",
                    ..
                })
            ),
            "reading the flags word little-endian would give class 0: {err:?}"
        );
    }

    /**
     * Every MATLAB release since 7 writes each variable inside a
     * `miCOMPRESSED` (type 15) zlib stream by default, so a reader that only
     * handles bare elements fails on nearly every real file it meets.
     * `matio` inflates transparently and vips never sees the difference: the
     * capture's compressed fixture is the same 2x3 array as the base case
     * and loads to the same pixels.
     * Input: `compressed.mat` -> Output: byte-for-byte what
     * `base_2x3_uint8.mat` gives.
     */
    #[test]
    fn a_compressed_element_is_inflated_and_gives_the_same_image() {
        let compressed = decoded(fixture!("compressed.mat"));
        let bare = decoded(fixture!("base_2x3_uint8.mat"));
        assert_eq!(
            (compressed.width(), compressed.height()),
            (bare.width(), bare.height())
        );
        assert_eq!(compressed.data(), bare.data());
    }

    /**
     * A corrupt zlib stream is not a hard error: `Mat_VarReadNextInfo`
     * returns NULL, the search ends, and vips reports `no matrix variables`
     * for a file that plainly contains one.
     * Measured on `compressed_corrupt.mat`, both sniffed and direct.
     * Input: a `miCOMPRESSED` element whose stream fails its check ->
     * Output: `NoMatrixVariables`, not an inflate error.
     */
    #[test]
    fn a_corrupt_compressed_stream_ends_the_search() {
        let err = refused(fixture!("compressed_corrupt.mat"));
        assert!(
            matches!(err, SourceError::Mat(MatError::NoMatrixVariables { .. })),
            "a corrupt stream ends the search rather than raising: {err:?}"
        );
    }

    /**
     * Where read-info stops and read-data begins, which is the give-up point
     * a port has to get right or it will disagree with vips about whether a
     * file has a header at all. `Mat_VarReadNextInfo` validates the
     * array-flags, dimensions and name subelements and never looks at the
     * data one.
     *
     * Both sides are here because either alone is satisfied by the wrong
     * rule. `tag_overruns_file.mat` declares a 1 MiB element with 16 bytes
     * behind it and reports `no matrix variables`, because the flags
     * subelement cannot be read; `truncated.mat` is missing its last eight
     * bytes and loads a *header* reporting the full 3x2, failing only at the
     * pixels. A reader that checked the outer element length against the
     * file would refuse both, and a reader that checked nothing would accept
     * both.
     * Input: the two fixtures -> Output: `NoMatrixVariables` and
     * `TruncatedData`, in that order and not the other way round.
     */
    #[test]
    fn read_info_validates_the_flags_and_dims_but_never_the_data() {
        let overruns = refused(fixture!("tag_overruns_file.mat"));
        assert!(
            matches!(
                overruns,
                SourceError::Mat(MatError::NoMatrixVariables { .. })
            ),
            "an unreadable flags subelement ends the search: {overruns:?}"
        );

        let truncated = refused(fixture!("truncated.mat"));
        match truncated {
            SourceError::Mat(MatError::TruncatedData { found, needed }) => {
                assert_eq!(
                    (found, needed),
                    (0, 6),
                    "the geometry read fine and only the samples are missing"
                );
            }
            other => panic!("a truncated data element is not a missing variable: {other:?}"),
        }
    }

    /**
     * A geometry the data cannot satisfy is refused rather than zero-filled.
     * vips reports the header without complaint and fails at first pixel
     * with `Mat_VarReadDataAll failed`; libviprs has one entry point, so the
     * refusal is the whole answer.
     * Measured on `dims_100x100_four_bytes.mat`: 10,000 declared pixels,
     * four bytes of data.
     * Input: a 100x100 `mxUINT8` array with four bytes behind it -> Output:
     * `TruncatedData { found: 4, needed: 10000 }`.
     */
    #[test]
    fn a_short_sample_array_is_refused_rather_than_zero_filled() {
        let err = refused(fixture!("dims_100x100_four_bytes.mat"));
        match err {
            SourceError::Mat(MatError::TruncatedData { found, needed }) => {
                assert_eq!((found, needed), (4, 10_000));
            }
            other => panic!("expected TruncatedData, got {other:?}"),
        }
    }

    /**
     * The decompression-bomb shape [`DecodeLimits`] exists for: eight bytes
     * of data behind a 10-gigapixel declaration, which vips reports at
     * header time without complaint because nothing in it prices the
     * declared geometry.
     *
     * The ceilings are walked down one at a time so the refusal does not
     * rest on a single arm: with everything at its default the pixel count
     * is what refuses, and lifting each ceiling in turn moves the refusal to
     * the next one until the file's own shortness is all that is left. The
     * last row is the positive control that the geometry checks are what
     * refused the earlier ones.
     * Measured: `dims_100000x100000.mat`, `100000x100000 uchar` at header
     * time and `Mat_VarReadDataAll failed` at the pixels.
     * Input: one fixture under four budgets -> Output: four different
     * refusals, in the order the checks run.
     */
    #[test]
    fn a_ten_gigapixel_declaration_is_refused_before_anything_is_allocated() {
        let bytes = fixture!("dims_100000x100000.mat");

        let tight = DecodeLimits::default().with_max_coord(1_000);
        match decode_mat(bytes, tight) {
            Err(SourceError::CoordLimitExceeded { width, height, .. }) => {
                assert_eq!((width, height), (100_000, 100_000));
            }
            other => panic!("expected CoordLimitExceeded, got {other:?}"),
        }

        let past_coord = DecodeLimits::default();
        match decode_mat(bytes, past_coord) {
            Err(SourceError::DimensionLimitExceeded { width, height, .. }) => {
                assert_eq!((width, height), (100_000, 100_000));
            }
            other => panic!("expected DimensionLimitExceeded, got {other:?}"),
        }

        let past_pixels = DecodeLimits::default().with_max_pixels(u64::MAX);
        match decode_mat(bytes, past_pixels) {
            Err(SourceError::AllocLimitExceeded {
                what, needed_bytes, ..
            }) => {
                assert_eq!(what, "MAT sample buffer");
                assert_eq!(needed_bytes, 10_000_000_000);
            }
            other => panic!("expected AllocLimitExceeded, got {other:?}"),
        }

        let open = DecodeLimits::default()
            .with_max_pixels(u64::MAX)
            .with_max_alloc_bytes(u64::MAX - 1);
        match decode_mat(bytes, open) {
            Err(SourceError::Mat(MatError::TruncatedData { found, needed })) => {
                assert_eq!((found, needed), (8, 10_000_000_000));
            }
            other => panic!("expected TruncatedData with every ceiling lifted, got {other:?}"),
        }
    }

    /**
     * A deliberate divergence, and the same one `analyzeload` carries: a
     * zero or negative dimension reaches `vips_image_init_fields`, GObject's
     * range check rejects it, the property is **left at 1**, and the load
     * carries on and exits 0 reading data that is not there. A 0x3 array
     * therefore loads in vips as a 3x1 image over a zero-length allocation,
     * which is why the capture records its pixels as deliberately unpinned.
     * Input: `dim_zero.mat` and `dim_negative.mat` -> Output:
     * `NonPositiveDimension` naming the axis, not a clamped raster.
     */
    #[test]
    fn a_non_positive_dimension_is_refused_rather_than_clamped_to_one() {
        for (bytes, expected) in [
            (fixture!("dim_zero.mat"), 0),
            (fixture!("dim_negative.mat"), -2),
        ] {
            match refused(bytes) {
                SourceError::Mat(MatError::NonPositiveDimension { axis, found }) => {
                    assert_eq!((axis, found), (0, expected));
                }
                other => panic!("expected NonPositiveDimension, got {other:?}"),
            }
        }
    }

    /**
     * The sharpest divergence in this format. `matload.c:158` says it will
     * not handle complex images and `matlab.c:45` lists it as a remaining
     * issue, but there is no check anywhere: the array flags' complex bit is
     * never read, and `mat2vips_get_data` memcpys out of a
     * `mat_complex_split_t`, which holds two pointers. So the pixels are the
     * raw bytes of two heap addresses, the load exits 0, and the capture
     * recorded three runs of the same file giving three different values
     * under ASLR.
     *
     * The refusal is checked before the carrier on purpose. The only complex
     * fixture in the capture is `mxDOUBLE`, which has no carrier here
     * either, so checking the carrier first would leave this divergence with
     * nothing exercising it.
     * Input: `complex_double.mat` -> Output: `ComplexArray`, not
     * `UnsupportedCarrier`.
     */
    #[test]
    fn a_complex_array_is_refused_rather_than_read_as_heap_pointers() {
        let err = refused(fixture!("complex_double.mat"));
        assert!(
            matches!(
                err,
                SourceError::Mat(MatError::ComplexArray {
                    class: 6,
                    name: "mxDOUBLE"
                })
            ),
            "the complex bit has to be read before the class is: {err:?}"
        );
    }

    /**
     * Issue #650, and the reason this port cannot be written from the C
     * source. `vips__mat_ismat` in the reference checkout reads ten bytes
     * and compares them with `MATLAB 5.0`; the shipped 8.18.6 dylib reads
     * 128 and also validates the version word and the endian indicator, and
     * the 8.18.4 it replaced did not.
     *
     * The capture recorded which byte positions change the answer, by
     * probing them one at a time against the binary. This sweeps all 128 of
     * a real file's header bytes through libviprs's own predicate and
     * asserts the set that matters is exactly the recorded one, which is
     * both halves at once: bytes 10..124 must be free text, and bytes 0..10
     * and 124..128 must every one of them be load-bearing.
     * Input: `base_2x3_uint8.mat` with each header byte flipped in turn ->
     * Output: the answer changes at exactly `0..10` and `124..128`.
     */
    #[test]
    fn the_sniff_predicate_is_the_shipped_binarys_and_not_the_sources() {
        // From `oracle-captures/foreign-mat`'s `sniff_predicate` record,
        // `byte_positions_that_change_the_answer`.
        const MEASURED: [usize; 14] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 124, 125, 126, 127];
        let base = fixture!("base_2x3_uint8.mat");
        assert!(
            Header::parse(base).is_ok(),
            "the unmodified fixture has to pass, or the sweep proves nothing"
        );

        let mut load_bearing = Vec::new();
        for at in 0..HEADER_BYTES {
            let mut poked = base.to_vec();
            poked[at] ^= 0xff;
            if Header::parse(&poked).is_err() {
                load_bearing.push(at);
            }
        }
        assert_eq!(
            load_bearing,
            MEASURED.to_vec(),
            "the sniff reads exactly the bytes the shipped 8.18.6 binary reads"
        );
    }

    /**
     * The length floor, which is the half of the change a port is most
     * likely to miss because the reference source has no floor at all: the
     * shipped sniff asks `vips__get_bytes` for 128 bytes and requires all
     * 128, so a file one byte short is refused before the prefix is even
     * compared.
     * Measured: the `length_floor` table, false at 0, 9, 10, 64 and 127,
     * true at 128, 129 and 200.
     * Input: a real fixture truncated to each of those lengths -> Output:
     * the measured answer, with 128 as the positive control.
     */
    #[test]
    fn the_header_needs_128_bytes_and_not_one_fewer() {
        let base = fixture!("base_2x3_uint8.mat");
        for len in [0usize, 9, 10, 64, 127] {
            let err = Header::parse(&base[..len]).expect_err("under the floor");
            assert!(
                matches!(
                    err,
                    MatError::ShortHeader {
                        needed: HEADER_BYTES,
                        ..
                    }
                ),
                "{len} bytes is under the floor: {err:?}"
            );
        }
        for len in [HEADER_BYTES, HEADER_BYTES + 1, 200] {
            assert!(
                Header::parse(&base[..len]).is_ok(),
                "{len} bytes is enough for the header"
            );
        }
    }

    /**
     * The five near-misses the shipped sniff refuses and a direct
     * `vips matload` loads, each reported by the clause that refused it.
     * `Mat_Open` is more permissive than `vips__mat_ismat` in exactly these
     * ways, and none of them is reachable through
     * `vips_image_new_from_file`, which is the entry point
     * [`crate::source::decode_file`] corresponds to.
     * Measured: every one of these prints
     * `"..." is not a known file format` from `vipsheader` while
     * `vips matload` on the same file exits 0 for the first three.
     * Input: the six refused fixtures -> Output: the named clause for each.
     */
    #[test]
    fn the_near_misses_are_refused_by_the_clause_that_refused_them() {
        assert!(matches!(
            Header::parse(fixture!("magic_MATLAB_51.mat")),
            Err(MatError::BadMagic { .. })
        ));
        assert!(matches!(
            Header::parse(fixture!("magic_matlab_50.mat")),
            Err(MatError::BadMagic { .. })
        ));
        assert!(matches!(
            Header::parse(fixture!("magic_MATLAB_50.mat")),
            Err(MatError::BadMagic { .. })
        ));
        assert!(matches!(
            Header::parse(fixture!("level73_hdf5.mat")),
            Err(MatError::BadMagic { .. })
        ));
        // A MAT-4 file is refused by the length floor rather than by the
        // prefix: its whole header is twenty bytes, so it never gets as far
        // as a text comparison. `matio` reads MAT-4 and gives the same 3x2
        // image as `double`, and vips will never reach it through
        // `new_from_file`.
        assert!(matches!(
            Header::parse(fixture!("level4.mat")),
            Err(MatError::ShortHeader { found: 70, .. })
        ));
        // Both of these have the right ten-byte prefix and differ only in
        // bytes 124..128, which is what `header_only.mat` next door is the
        // control for.
        assert!(matches!(
            Header::parse(fixture!("endian_bogus.mat")),
            Err(MatError::BadEndianIndicator { .. })
        ));
        assert!(matches!(
            Header::parse(fixture!("magic_only.mat")),
            Err(MatError::BadEndianIndicator { .. })
        ));
    }

    /**
     * The other half of the sniff, which a port has to reproduce or it will
     * disagree with vips about which loader owns a file: a bare 128-byte
     * header **passes** the sniff and then fails inside the loader with
     * `no matrix variables`, and a header followed by arbitrary text still
     * loads its variable, because bytes 10..124 are free text.
     * Measured: `header_only.mat` reports `mat2vips: no matrix variables`
     * from `vipsheader` (i.e. the loader was selected), and
     * `prefix_only.mat`, whose description reads `MATLAB 5.0 anything at
     * all can follow here`, loads `3x2 uchar`.
     * Input: the two fixtures -> Output: a refusal from inside the codec,
     * and a raster.
     */
    #[test]
    fn the_sniff_passes_a_bare_header_and_the_loader_is_what_refuses_it() {
        for name in [fixture!("header_only.mat"), fixture!("no_variables.mat")] {
            let err = refused(name);
            assert!(
                matches!(err, SourceError::Mat(MatError::NoMatrixVariables { .. })),
                "a bare header reaches the codec and fails there: {err:?}"
            );
        }
        let raster = decoded(fixture!("prefix_only.mat"));
        assert_eq!((raster.width(), raster.height()), (3, 2));
        assert_eq!(raster.data(), &[10, 20, 30, 40, 50, 60]);
    }

    /**
     * `matio` widens a narrower storage type up to the array's class on
     * read; this module refuses instead, because nothing in the capture
     * measures the conversion and MATLAB never writes a narrowed storage
     * type for the three classes that load here.
     *
     * Built by poking a real fixture rather than fabricating one, so the
     * only thing that differs from a file this module loads is the byte
     * under test.
     * Input: `class_mat_c_uint16.mat` with its data element retyped from
     * `miUINT16` to `miUINT8` -> Output: `StorageTypeMismatch` naming both.
     */
    #[test]
    fn a_storage_type_that_does_not_match_the_class_is_refused() {
        let mut poked = fixture!("class_mat_c_uint16.mat").to_vec();
        assert!(decode_mat(&poked, DecodeLimits::default()).is_ok());
        assert_eq!(poked[184], MI_UINT16 as u8, "the data element's type");
        poked[184] = MI_UINT8 as u8;
        match refused(&poked) {
            SourceError::Mat(MatError::StorageTypeMismatch {
                class_name,
                expected,
                found,
                ..
            }) => {
                assert_eq!(class_name, "mxUINT16");
                assert_eq!((expected, found), (MI_UINT16, MI_UINT8));
            }
            other => panic!("expected StorageTypeMismatch, got {other:?}"),
        }
    }

    /**
     * A band count with no pixel format is refused by name rather than
     * narrowed or widened. vips gives a rank-3 array `dims[2]` bands
     * whatever that is; [`PixelFormat`]'s multiband carriers are documented
     * as compute intermediates the decode path does not produce, so this
     * refuses rather than widen that claim.
     * Built by poking `rank3_2x3x3.mat`'s third dimension from 3 to 2, which
     * leaves eighteen bytes of samples behind a twelve-byte geometry, so the
     * refusal cannot be the truncation check answering instead.
     * Input: a 2x3x2 `mxUINT8` array -> Output: `BandCount { bands: 2 }`.
     */
    #[test]
    fn a_band_count_with_no_pixel_format_is_refused_by_name() {
        let mut poked = fixture!("rank3_2x3x3.mat").to_vec();
        // dims[2] is the third `i32` of the dimensions subelement, whose
        // payload starts sixteen bytes past the flags subelement.
        assert_eq!(poked[168], 3, "dims[2]");
        poked[168] = 2;
        match refused(&poked) {
            SourceError::Mat(MatError::BandCount { bands }) => assert_eq!(bands, 2),
            other => panic!("expected BandCount, got {other:?}"),
        }
    }

    /**
     * The other unbounded allocation a `.mat` can ask for, and the one no
     * geometry check can see: a `miCOMPRESSED` element's inflated size is
     * not declared anywhere in the container, so a small file can inflate to
     * an arbitrary one. The inflate is capped at
     * [`DecodeLimits::max_alloc_bytes`] and refused rather than truncated.
     *
     * The reported price is asserted to be exactly the budget plus one
     * byte, and that assertion is the whole test rather than a detail. The
     * refusal on its own is satisfied by an *uncapped* read followed by a
     * length check, which allocates the whole bomb and only then complains;
     * that mutation came back green on the first sweep of this module and is
     * why the assertion is here. A price of `cap + 1` can only be reported
     * by a reader that stopped there.
     *
     * The bomb is built here rather than checked in, because a fixture that
     * inflates to half a megabyte is not something a capture script should
     * be writing into the repository. The positive control is the third
     * part: the same bytes inflate fine when the budget is lifted, so the
     * refusal is the budget and not the shape.
     * Input: a 512 KiB zlib stream of zeroes under a 1024-byte budget ->
     * Output: `AllocLimitExceeded` naming the compressed element and pricing
     * it at 1025, not at 524,288.
     */
    #[test]
    fn a_compressed_element_that_inflates_past_the_budget_is_refused() {
        use std::io::Write;
        let mut encoder =
            flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&vec![0u8; 512 * 1024]).unwrap();
        let stream = encoder.finish().unwrap();

        let mut bomb = fixture!("header_only.mat").to_vec();
        bomb.extend_from_slice(&(MI_COMPRESSED).to_le_bytes());
        bomb.extend_from_slice(&(stream.len() as u32).to_le_bytes());
        bomb.extend_from_slice(&stream);
        bomb.resize(bomb.len().next_multiple_of(8), 0);

        let cap = 1024u64;
        let tight = DecodeLimits::default().with_max_alloc_bytes(cap);
        match decode_mat(&bomb, tight) {
            Err(SourceError::AllocLimitExceeded {
                what,
                needed_bytes,
                max_alloc_bytes,
                ..
            }) => {
                assert_eq!(what, "MAT compressed element");
                assert_eq!(max_alloc_bytes, cap);
                assert_eq!(
                    needed_bytes,
                    cap + 1,
                    "the reader has to stop at the budget, so the price it reports is a \
                     floor; a reader that inflated the whole stream and then measured it \
                     would report 524288"
                );
            }
            other => panic!("expected the inflate to be refused, got {other:?}"),
        }

        // The control: the same bytes are not refused for their shape. With
        // the budget lifted the stream inflates and the search then runs off
        // the end of a buffer that holds no matrix.
        let open = DecodeLimits::default().with_max_alloc_bytes(u64::MAX - 1);
        let err = decode_mat(&bomb, open).expect_err("the stream holds no matrix");
        assert!(
            matches!(err, SourceError::Mat(MatError::NoMatrixVariables { .. })),
            "with the budget lifted the inflate succeeds: {err:?}"
        );
    }

    /**
     * The carrier table's two halves have to agree: the [`PixelFormat`] each
     * carrier and band count names, and the `sample_bytes` the budget is
     * spent with. A mismatch would price a file at one size and allocate
     * another, so it is checked rather than reasoned.
     * The band counts with no format are checked too, because
     * [`MatError::BandCount`] exists only if that set is non-empty.
     * Input: every `Carrier` at every band count from 0 to 5 -> Output: a
     * format whose channel count and sample width are the ones priced, and
     * `None` at 0, 2 and 5.
     */
    #[test]
    fn the_carrier_table_agrees_with_pixel_format() {
        for carrier in Carrier::ALL {
            for bands in 0u32..=5 {
                match carrier.pixel_format(bands) {
                    Some(format) => {
                        assert!(
                            matches!(bands, 1 | 3 | 4),
                            "{carrier:?} named a format for {bands} bands"
                        );
                        assert_eq!(
                            format.channels(),
                            bands as usize,
                            "{carrier:?} at {bands} bands names a format of another width"
                        );
                        assert_eq!(
                            format.bytes_per_channel() as u64,
                            carrier.sample_bytes(),
                            "{carrier:?} names one sample width and prices another"
                        );
                    }
                    None => assert!(
                        !matches!(bands, 1 | 3 | 4),
                        "{carrier:?} has no format for {bands} bands"
                    ),
                }
            }
        }
    }

    /**
     * The metadata this module attaches, which is libviprs's own: `matload`
     * attaches nothing at all, so `vipsheader -a` on a loaded `.mat` shows
     * only the standard fields and there is no oracle value to disagree
     * with. It exists because `matload` picks one variable out of a file
     * with no way to ask for another and then loses its name, which the
     * capture calls out as a wart.
     * Input: `rank3_2x3x3.mat` -> Output: the variable's name, class, byte
     * order, description and all three declared dimensions.
     */
    #[test]
    fn the_metadata_records_the_variable_the_search_chose() {
        let raster = decoded(fixture!("rank3_2x3x3.mat"));
        let field = |name: &str| raster.fields.get(name).cloned();
        assert_eq!(
            field("mat-class").as_ref().map(MetadataValue::as_str),
            Some("mxUINT8")
        );
        assert_eq!(
            field("mat-endian").as_ref().map(MetadataValue::as_str),
            Some("IM")
        );
        assert_eq!(
            field("mat-description").as_ref().map(MetadataValue::as_str),
            Some("MATLAB 5.0 MAT-file, written by libviprs oracle capture")
        );
        for (axis, extent) in [(0, 2i64), (1, 3), (2, 3)] {
            assert_eq!(
                field(&format!("mat-dims[{axis}]"))
                    .as_ref()
                    .map(MetadataValue::as_i64),
                Some(extent),
                "dims[{axis}]"
            );
        }
        assert!(
            field("mat-dims[3]").is_none(),
            "a rank-3 variable declares three dimensions and no more"
        );
    }
}
