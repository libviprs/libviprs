//! NIfTI (`.nii`) load: a fixed-size header in, a raw voxel array out.
//!
//! NIfTI-1 declares its geometry in 348 bytes and NIfTI-2 in 540, and then
//! the rest of the file is the voxels, uncompressed, x fastest. There is no
//! codec here at all, which is why this module hand-rolls the whole thing
//! and adds no dependency: see [the dependency note](#why-this-is-hand-rolled-rather-than-a-dependency).
//!
//! # Operations
//!
//! | libviprs method | reference equivalent | result |
//! |---|---|---|
//! | [`decode_nifti`] | `nifti_image_read` on a single-file `.nii` | [`PixelFormat::Gray8`], [`PixelFormat::Gray16`], [`PixelFormat::Rgb8`], [`PixelFormat::Rgba8`] or `FloatF32(1)` |
//!
//! There is no save. NIfTI is load-only here for the same reason Analyze and
//! MAT are load-only in libvips: nothing in this crate wants to write a
//! volume.
//!
//! # The oracle is deliberately not libvips
//!
//! Every other loader in this crate is measured against `vips`. This one
//! cannot be, and that is measured rather than assumed: the pinned
//! `/opt/homebrew/bin/vips` reports `NIfTI load/save with libnifti: false`
//! and registers neither `niftiload` nor `niftisave`, so a `.nii` handed to
//! it falls through the sniffing chain to `magickload`, which guesses TGA.
//! The oracle is `nifti_clib` (`NIFTI-Imaging/nifti_clib`, the NIH reference
//! implementation, `v3.0.1-91-g8f72d11`), which is both the format's own
//! reference and the library libvips itself would have linked. The capture,
//! including the three-step evidence that vips declines, is in
//! `oracle-captures/foreign-nifti/`, and it re-measures the vips half on
//! every run so a build that gains libnifti announces itself.
//!
//! # Semantics
//!
//! * **The version comes from one sentinel.** `sizeof_hdr` is 348 for
//!   NIfTI-1 and 540 for NIfTI-2, read one way and then byte-swapped and
//!   retried. 348 swapped is 1543503872 and 540 swapped is 469827584, so at
//!   most one of the four readings can match. There is no flag anywhere in
//!   the file that says which way round it is.
//! * **The byte order does *not* come from that sentinel, on NIfTI-1.**
//!   `dim[0]` decides first, and only a `dim[0]` that is out of range either
//!   way falls back to `sizeof_hdr`. Measured, and this is the one place the
//!   capture's own prose is wrong: `bad_sizeof_swapped.nii` is a
//!   little-endian file with *only* its four sentinel bytes swapped, and it
//!   loads `LSB_FIRST` with `dim` reading `3 2 3 1`, which the
//!   sentinel-decides rule cannot produce. The rule, and the four fixtures
//!   that force it, are written out over `field_endian` in this module.
//! * **The magic decides the dialect, and only four bytes of it are read.**
//!   `NIFTI_VERSION` looks at `magic[0..4]`: `n`, then `i` or `+`, then a
//!   digit 1 to 9, then a NUL. Anything else is version 0, the Analyze 7.5
//!   dialect. The `\r \n \032 \n` tail of the eight-byte NIfTI-2 magic, which
//!   the spec puts there to catch a text-mode transfer, is **never examined**:
//!   measured, a file with that tail destroyed loads normally.
//! * **`magic[1] == '+'` means one file.** `n+1` / `n+2` carry the voxels
//!   after their own header; `ni1` / `ni2` are the header half of a `.hdr` /
//!   `.img` pair and carry none. Nothing else in the header says which.
//! * **`vox_offset` is clamped, not trusted.** Measured: `-8` and `100` both
//!   load with the data starting at byte 348, and `352.5` truncates to 352.
//!   So the rule is `max(trunc(vox_offset), header_len)`, and a negative or
//!   undersized offset is silently repaired rather than refused.
//! * **`dim[0]` is the rank and the extents are repaired unevenly.** Rank
//!   outside `0..=7` is refused; rank 0 is *accepted* and gives a one-voxel
//!   image with every extent discarded; a zero `dim[1]` is refused; a zero
//!   extent on any axis above the first is silently clamped to 1. All four
//!   measured, and the asymmetry is the reference's, not a simplification.
//! * **Non-finite float samples are rewritten to zero.** `nifti_read_buffer`
//!   walks the loaded buffer and sets every `FLOAT32` value failing
//!   `isfinite` to 0 before the caller sees it. Measured on
//!   `float_float32.nii`: the last three words go from
//!   `0000807f 000080ff 0000c07f` to `00000000 00000000 00000000`. An
//!   infinity or a NaN stored in a NIfTI file never comes back, and this
//!   module reproduces that rather than passing the values through.
//! * **`scl_slope` and `scl_inter` are carried, not applied.** The reference
//!   returns raw voxels; the `y = slope * x + inter` rule lives in FSL's
//!   `fsliolib/fslio.c`, not in `nifti_clib`. Measured across every `scl_*`
//!   fixture. This module attaches both as metadata and scales nothing, so a
//!   caller that wants the physical units applies them itself.
//! * **`bitpix` is decoration.** Measured: a header declaring `datatype` 4
//!   with `bitpix` 64 loads as 16-bit anyway. The datatype alone fixes the
//!   sample width, and this module never reads `bitpix` for anything but
//!   metadata.
//!
//! # How a volume becomes one raster
//!
//! NIfTI is a volume format and [`Raster`] is two-dimensional, so something
//! has to give. `nifti_clib` keeps all seven axes and leaves the reshaping to
//! its caller, so there is no reference answer to copy here.
//!
//! The rule this module uses is the one libvips's `analyzeload` uses for the
//! sibling format, measured in `oracle-captures/foreign-analyze/`: the width
//! is `dim[1]` and the height is the product of `dim[2]` up to `dim[rank]`,
//! so a 3x2x2 volume becomes one 3x4 image with the slices stacked. It is
//! chosen rather than invented, and it costs nothing at decode time, because
//! the flattening is a pure reshape: the file already stores the array x
//! fastest, so not one byte moves. Every voxel value pinned in the tests
//! below is therefore the reference's own, independent of the reshape.
//!
//! Nothing in the raster records that it was ever three-dimensional beyond
//! the `nifti-dim[N]` metadata this module attaches, which is the same thing
//! `analyzeload` does with `dsr-image_dimension.dim[N]`.
//!
//! # The carrier ceiling
//!
//! [`PixelFormat`] has unsigned 8-bit, unsigned 16-bit and 32-bit float
//! carriers and nothing else, so most of the NIfTI datatype table has nowhere
//! to land. Five codes load and the rest are refused **by name**, with the
//! issue that would add the carrier, exactly the way [`crate::fits`] refuses
//! a signed BITPIX rather than narrowing it. Narrowing a signed 16-bit array
//! into 8 bits, or a 64-bit integer into a float, would lose data silently,
//! which is worse than failing. See [`NiftiError::UnsupportedCarrier`].
//!
//! # Why this is hand-rolled rather than a dependency
//!
//! Not because the ecosystem is empty: `nifti-rs` is MIT, pure Rust and
//! actively maintained. The reason is that there is nothing for a dependency
//! to do. The whole format is a fixed-offset header struct and a raw array;
//! this module is the header field offsets, a byte-order flag and a copy
//! loop, and the parts that actually have to be right are the *repair* rules
//! above, which no NIfTI crate models because they are `nifti_clib`
//! behaviours rather than spec text. A dependency would supply the struct
//! layout, which is the free half, and leave every measured repair here
//! anyway.
//!
//! It would also cost. `nifti-rs` pulls `ndarray` and `num-complex` for its
//! volume API, which is a large graph for a crate that wants a `Vec<u8>`, and
//! its gzip support duplicates the `flate2` already in this tree. The
//! dependency rule in `CONTRIBUTING.md` is not what rules it out (it is pure
//! Rust with no `-sys` crate), the cost/benefit is.
//!
//! Every entry point here is fallible and there is no panicking twin,
//! matching the rest of the codec surface in [`crate::fits`],
//! [`crate::radiance`] and [`crate::exr`]: a decoder's failures come from
//! untrusted bytes, so a panicking spelling would have no honest caller.

use std::num::NonZeroU16;

use thiserror::Error;

use crate::conversion::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use crate::source::{DecodeLimits, SourceError};

/// Bytes in a NIfTI-1 header, and the value its `sizeof_hdr` field carries.
pub const HEADER_1_BYTES: usize = 348;

/// Bytes in a NIfTI-2 header, and the value its `sizeof_hdr` field carries.
pub const HEADER_2_BYTES: usize = 540;

/// Where the voxels start in a single-file NIfTI-1 that does not say
/// otherwise: the 348-byte header plus the four-byte extender.
pub const DEFAULT_DATA_OFFSET_1: u64 = 352;

/// Where the voxels start in a single-file NIfTI-2 that does not say
/// otherwise: the 540-byte header plus the four-byte extender.
pub const DEFAULT_DATA_OFFSET_2: u64 = 544;

/// The highest rank `dim[0]` may declare. Measured: 7 loads, 8 is refused
/// with `bad dim[0]`, and so is a negative.
pub const MAX_RANK: usize = 7;

/// The single-file NIfTI-1 magic, at offset 344.
pub(crate) const MAGIC_1_SINGLE: &[u8; 4] = b"n+1\0";

/// The two-file NIfTI-1 magic, at offset 344. A `.hdr` carrying this has no
/// voxels in it at all.
pub(crate) const MAGIC_1_PAIR: &[u8; 4] = b"ni1\0";

/// The single-file NIfTI-2 magic, at offset 4.
pub(crate) const MAGIC_2_SINGLE: &[u8; 8] = b"n+2\0\r\n\x1a\n";

/// The two-file NIfTI-2 magic, at offset 4.
pub(crate) const MAGIC_2_PAIR: &[u8; 8] = b"ni2\0\r\n\x1a\n";

/// `sizeof_hdr` as a NIfTI-1 file writes it on a little-endian host.
pub(crate) const SIZEOF_HDR_1_LE: &[u8; 4] = &[0x5c, 0x01, 0x00, 0x00];

/// `sizeof_hdr` as a NIfTI-1 file writes it on a big-endian host.
pub(crate) const SIZEOF_HDR_1_BE: &[u8; 4] = &[0x00, 0x00, 0x01, 0x5c];

/// `sizeof_hdr` as a NIfTI-2 file writes it on a little-endian host.
pub(crate) const SIZEOF_HDR_2_LE: &[u8; 4] = &[0x1c, 0x02, 0x00, 0x00];

/// `sizeof_hdr` as a NIfTI-2 file writes it on a big-endian host.
pub(crate) const SIZEOF_HDR_2_BE: &[u8; 4] = &[0x00, 0x00, 0x02, 0x1c];

/// Where the NIfTI-1 magic sits, which is the last field of its header.
pub(crate) const MAGIC_1_AT: usize = 344;

/// Where the NIfTI-2 magic sits, which is its second field.
pub(crate) const MAGIC_2_AT: usize = 4;

/// Errors from the NIfTI loader.
///
/// Every variant except [`NiftiError::Raster`] describes a specific
/// malformation in, or a specific limit of, untrusted bytes. The allocation
/// refusal is deliberately not here: it is
/// [`SourceError::AllocLimitExceeded`], the one shape issue #686 collapsed
/// five per-format variants onto.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NiftiError {
    /// Fewer than [`HEADER_1_BYTES`] bytes, so no version can be decided.
    ///
    /// Measured: `nifti_header_version` refuses anything under 348 outright,
    /// **including a valid NIfTI-2 header read 347 bytes at a time**, even
    /// though sizeof_hdr and the magic both live in the first twelve.
    #[error("nifti: need at least {needed} bytes to read a header, found {found}")]
    ShortHeader {
        /// Bytes available.
        found: usize,
        /// Bytes the version check needs.
        needed: usize,
    },
    /// `sizeof_hdr` is neither 348 nor 540 in either byte order.
    #[error("nifti: sizeof_hdr is {found}, which is neither 348 nor 540 in either byte order")]
    BadSizeofHdr {
        /// The field as read on this host, before any swap.
        found: i32,
    },
    /// The header size and the magic disagree about the version.
    ///
    /// Measured both ways: a 540-byte header carrying `n+1` and a 348-byte
    /// header carrying `n+2` are each refused with `bad nifti header
    /// version`.
    #[error(
        "nifti: a {header_bytes}-byte header carries magic {magic:?}, which declares \
         version {declared}"
    )]
    VersionMismatch {
        /// The header length `sizeof_hdr` declared.
        header_bytes: usize,
        /// The magic as read, lossily decoded for the message.
        magic: String,
        /// The version the magic declares.
        declared: u8,
    },
    /// A 348-byte header whose magic is not a NIfTI one at all.
    ///
    /// Measured: `nifti_clib` accepts this as version 0, the Analyze 7.5
    /// dialect, and then asks the *filename* whether the voxels are in this
    /// file or in a sibling `.img`. A byte-buffer decode has no filename, so
    /// there is nothing to ask and this module refuses rather than guess.
    #[error(
        "nifti: this 348-byte header carries magic {magic:?} rather than a NIfTI one, \
         so it is the Analyze 7.5 dialect, whose container is decided by the filename \
         a buffer decode does not have"
    )]
    AnalyzeDialect {
        /// The magic as read, lossily decoded for the message.
        magic: String,
    },
    /// The header is the two-file form, so the voxels are in a sibling
    /// `.img` this buffer does not contain.
    #[error(
        "nifti: magic {magic:?} is the two-file form, so the voxels are in a sibling \
         .img rather than in these bytes"
    )]
    PairedHeader {
        /// The magic as read, lossily decoded for the message.
        magic: String,
    },
    /// `dim[0]`, the rank, is outside `0..=7`.
    #[error("nifti: dim[0] is {found}, which is outside the 0..={max} a rank may take")]
    BadRank {
        /// The rank as declared.
        found: i64,
        /// The highest rank the format allows.
        max: usize,
    },
    /// `dim[1]` is not positive.
    ///
    /// The first axis is the one the reference refuses: measured, a zero or
    /// negative `dim[1]` fails with `bad dim[1]`, while the same value on any
    /// higher axis is silently clamped to 1.
    #[error("nifti: dim[1] is {found}, and the first axis has to be positive")]
    BadExtent {
        /// The extent as declared.
        found: i64,
    },
    /// The declared extents multiply out past what a raster coordinate can
    /// hold, before any budget is consulted.
    ///
    /// This is arithmetic rather than policy: `dim` is `i64` in NIfTI-2, so
    /// a rank-7 header can declare a product no `u64` holds, let alone a
    /// `u32` coordinate. [`DecodeLimits`] is what refuses everything that
    /// *does* fit.
    #[error(
        "nifti: dim[1..={rank}] multiply out past the largest raster coordinate, so the \
         geometry cannot be represented at all"
    )]
    DimensionOverflow {
        /// The rank whose extents were multiplied.
        rank: usize,
    },
    /// A datatype code no NIfTI dialect defines.
    #[error("nifti: datatype {datatype} is not a datatype this format defines")]
    UnknownDatatype {
        /// The code as declared.
        datatype: i16,
    },
    /// A datatype the reference loads but libviprs has no sample carrier
    /// for.
    ///
    /// This is a "not yet" rather than a "never", and it is the same ceiling
    /// [`crate::fits::FitsError::UnsupportedCarrier`] describes, reached from
    /// a different table:
    ///
    /// * `INT8` (256), `INT16` (4) and `INT32` (8) are signed integers, which
    ///   need issue #516. `INT16` is the single most common datatype in real
    ///   NIfTI files, so this is the refusal a caller will meet first.
    /// * `UINT32` (768) needs issue #517.
    /// * `FLOAT64` (64) needs the `f64` carrier, which is issue #518 and was
    ///   closed as not worth building on its own.
    /// * `INT64` (1024), `UINT64` (1280), `FLOAT128` (1536), `COMPLEX64`
    ///   (32), `COMPLEX128` (1792) and `COMPLEX256` (2048) have no carrier
    ///   and no issue: nothing else in this crate wants them either.
    ///
    /// The sample-kind spine (issue #607) is what makes the first three
    /// cheap, so the ceiling lifts there rather than never. Until then the
    /// loader refuses by name, because narrowing a 16-bit array into 8 bits
    /// would lose data silently.
    #[error(
        "nifti: datatype {datatype} ({name}) carries {sample} samples, which libviprs \
         has no pixel format for yet"
    )]
    UnsupportedCarrier {
        /// The code as declared.
        datatype: i16,
        /// The reference's own name for it, without the `NIFTI_TYPE_`
        /// prefix.
        name: &'static str,
        /// The sample kind it needs, in words.
        sample: &'static str,
    },
    /// A datatype code the format defines but that carries no samples.
    ///
    /// `DT_NONE` (0), `DT_BINARY` (1) and `DT_ALL` (255) are all in the
    /// table and all refused by `nifti_convert_n1hdr2nim` with `bad
    /// datatype`; measured for 0 and 1.
    #[error("nifti: datatype {datatype} ({name}) carries no voxel data")]
    EmptyDatatype {
        /// The code as declared.
        datatype: i16,
        /// The reference's own name for it.
        name: &'static str,
    },
    /// The file ends before the declared voxel array does.
    ///
    /// Measured: the reference warns (`number missing = N (set to 0)`) and
    /// then fails the load, so a short file is a refusal on both sides and
    /// the zero-fill never reaches a caller.
    #[error("nifti: the voxel array needs {needed} bytes from offset {offset}, found {found}")]
    TruncatedData {
        /// Where the voxels were supposed to start.
        offset: u64,
        /// Bytes actually available from there.
        found: u64,
        /// Bytes the declared geometry needs.
        needed: u64,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// Which of the two header layouts a file uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Version {
    /// NIfTI-1: 348 bytes, `i16` extents, `f32` `vox_offset`, magic at 344.
    One,
    /// NIfTI-2: 540 bytes, `i64` extents, `i64` `vox_offset`, magic at 4.
    Two,
}

impl Version {
    /// Bytes in this version's header.
    const fn header_bytes(self) -> usize {
        match self {
            Self::One => HEADER_1_BYTES,
            Self::Two => HEADER_2_BYTES,
        }
    }
}

/// Which way round every multi-byte field in the file is.
///
/// There is no flag for this: the sentinel `sizeof_hdr` is what decides, and
/// then every other field follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endian {
    Little,
    Big,
}

impl Endian {
    fn i16(self, b: [u8; 2]) -> i16 {
        match self {
            Self::Little => i16::from_le_bytes(b),
            Self::Big => i16::from_be_bytes(b),
        }
    }

    fn i32(self, b: [u8; 4]) -> i32 {
        match self {
            Self::Little => i32::from_le_bytes(b),
            Self::Big => i32::from_be_bytes(b),
        }
    }

    fn i64(self, b: [u8; 8]) -> i64 {
        match self {
            Self::Little => i64::from_le_bytes(b),
            Self::Big => i64::from_be_bytes(b),
        }
    }

    fn f32(self, b: [u8; 4]) -> f32 {
        match self {
            Self::Little => f32::from_le_bytes(b),
            Self::Big => f32::from_be_bytes(b),
        }
    }

    fn f64(self, b: [u8; 8]) -> f64 {
        match self {
            Self::Little => f64::from_le_bytes(b),
            Self::Big => f64::from_be_bytes(b),
        }
    }

    fn u16(self, b: [u8; 2]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes(b),
            Self::Big => u16::from_be_bytes(b),
        }
    }
}

/// The sample carrier a datatype lands on, and how wide one voxel is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    /// `NIFTI_TYPE_UINT8`, one unsigned byte per voxel.
    U8,
    /// `NIFTI_TYPE_UINT16`, one unsigned 16-bit sample per voxel.
    U16,
    /// `NIFTI_TYPE_FLOAT32`, one 32-bit float per voxel.
    F32,
    /// `NIFTI_TYPE_RGB24`, three interleaved bytes per voxel.
    Rgb24,
    /// `NIFTI_TYPE_RGBA32`, four interleaved bytes per voxel.
    Rgba32,
}

impl Carrier {
    /// Bands one voxel becomes.
    const fn bands(self) -> u64 {
        match self {
            Self::U8 | Self::U16 | Self::F32 => 1,
            Self::Rgb24 => 3,
            Self::Rgba32 => 4,
        }
    }

    /// Bytes in one band sample.
    const fn sample_bytes(self) -> u64 {
        match self {
            Self::U8 | Self::Rgb24 | Self::Rgba32 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }

    /// The [`PixelFormat`] a raster of this carrier gets.
    ///
    /// Written out per variant rather than reached through
    /// [`PixelFormat::with_channels`], because that constructor is fallible
    /// and every one of these five is total: routing them through it would
    /// put an unreachable error arm in the decode path with nothing able to
    /// exercise it. `the_carrier_table_agrees_with_pixel_format` checks the
    /// two against each other instead, which is a claim with something
    /// behind it.
    const fn pixel_format(self) -> PixelFormat {
        match self {
            Self::U8 => PixelFormat::Gray8,
            Self::U16 => PixelFormat::Gray16,
            Self::F32 => PixelFormat::FloatF32(NonZeroU16::MIN),
            Self::Rgb24 => PixelFormat::Rgb8,
            Self::Rgba32 => PixelFormat::Rgba8,
        }
    }

    /// Every carrier, for the tests that sweep the table.
    #[cfg(test)]
    const ALL: [Self; 5] = [Self::U8, Self::U16, Self::F32, Self::Rgb24, Self::Rgba32];

    /// The interpretation tag a raster of this carrier gets.
    ///
    /// Follows `analyzeload`'s rule (`analyze2vips.c:544-546`), measured in
    /// `oracle-captures/foreign-analyze/`: one band is `b-w` and anything
    /// else is sRGB, with no 16-bit greyscale or RGB16 tag. Chosen over
    /// `matload`'s finer table because Analyze is the format NIfTI overlays,
    /// not MAT.
    const fn interpretation(self) -> Interpretation {
        match self {
            Self::U8 | Self::U16 | Self::F32 => Interpretation::Bw,
            Self::Rgb24 | Self::Rgba32 => Interpretation::Srgb,
        }
    }
}

/// Resolve a datatype code onto a carrier, or say precisely why not.
///
/// The table is the reference's own, swept in the capture's `datatype_codes`
/// record across `nifti_datatype_sizes`, `nifti_is_valid_datatype` and their
/// siblings. Codes that are not in it at all are [`NiftiError::UnknownDatatype`];
/// codes that are in it but carry no data are [`NiftiError::EmptyDatatype`];
/// codes that carry data libviprs has no carrier for are
/// [`NiftiError::UnsupportedCarrier`], by name.
fn carrier_for(datatype: i16) -> Result<Carrier, NiftiError> {
    let unsupported = |name, sample| {
        Err(NiftiError::UnsupportedCarrier {
            datatype,
            name,
            sample,
        })
    };
    match datatype {
        2 => Ok(Carrier::U8),
        512 => Ok(Carrier::U16),
        16 => Ok(Carrier::F32),
        128 => Ok(Carrier::Rgb24),
        2304 => Ok(Carrier::Rgba32),
        0 => Err(NiftiError::EmptyDatatype {
            datatype,
            name: "DT_NONE",
        }),
        1 => Err(NiftiError::EmptyDatatype {
            datatype,
            name: "DT_BINARY",
        }),
        255 => Err(NiftiError::EmptyDatatype {
            datatype,
            name: "DT_ALL",
        }),
        256 => unsupported("INT8", "signed 8-bit"),
        4 => unsupported("INT16", "signed 16-bit"),
        8 => unsupported("INT32", "signed 32-bit"),
        768 => unsupported("UINT32", "unsigned 32-bit"),
        64 => unsupported("FLOAT64", "64-bit float"),
        1024 => unsupported("INT64", "signed 64-bit"),
        1280 => unsupported("UINT64", "unsigned 64-bit"),
        1536 => unsupported("FLOAT128", "128-bit float"),
        32 => unsupported("COMPLEX64", "64-bit complex"),
        1792 => unsupported("COMPLEX128", "128-bit complex"),
        2048 => unsupported("COMPLEX256", "256-bit complex"),
        _ => Err(NiftiError::UnknownDatatype { datatype }),
    }
}

/// The version a magic declares, by the reference's `NIFTI_VERSION` macro.
///
/// Reads four bytes and nothing else: `n`, then `i` or `+`, then a digit 1 to
/// 9, then a NUL. Zero means "no NIfTI magic here", which the reference reads
/// as the Analyze 7.5 dialect rather than as an error.
fn magic_version(magic: [u8; 4]) -> u8 {
    if magic[0] == b'n'
        && (magic[1] == b'i' || magic[1] == b'+')
        && magic[2].is_ascii_digit()
        && magic[2] != b'0'
        && magic[3] == 0
    {
        magic[2] - b'0'
    } else {
        0
    }
}

/// Whether a magic says the voxels are in this same file.
///
/// `NIFTI_ONEFILE` is `magic[1] == '+'` alone, so `n+1` / `n+2` are one file
/// and `ni1` / `ni2` are a pair, and nothing else in the header says which.
const fn magic_is_single_file(magic: [u8; 4]) -> bool {
    magic[1] == b'+'
}

/// Render a magic for an error message, keeping the printable bytes and
/// escaping the rest.
fn show_magic(magic: &[u8]) -> String {
    magic
        .iter()
        .map(|&b| {
            if b.is_ascii_graphic() {
                char::from(b).to_string()
            } else {
                format!("\\x{b:02x}")
            }
        })
        .collect()
}

/// Decide the version, and the byte order the *sentinel* was read in, from
/// the leading bytes.
///
/// This is `nifti_header_version`, and its refusal floor is measured rather
/// than reasoned: it needs 348 bytes even to answer for a NIfTI-2 file whose
/// sentinel and magic both live in the first twelve.
///
/// The endian it returns is only the order `sizeof_hdr` matched in. On
/// NIfTI-1 that is *not* the order the rest of the fields are in; see
/// [`field_endian`].
fn header_version(bytes: &[u8]) -> Result<(Version, Endian), NiftiError> {
    if bytes.len() < HEADER_1_BYTES {
        return Err(NiftiError::ShortHeader {
            found: bytes.len(),
            needed: HEADER_1_BYTES,
        });
    }
    let raw: [u8; 4] = bytes[..4]
        .try_into()
        .expect("four bytes of a 348-byte head");
    for endian in [Endian::Little, Endian::Big] {
        let sizeof_hdr = endian.i32(raw);
        let (version, at, width) = match sizeof_hdr {
            348 => (Version::One, MAGIC_1_AT, 4),
            540 => (Version::Two, MAGIC_2_AT, 8),
            _ => continue,
        };
        let magic: [u8; 4] = bytes[at..at + 4]
            .try_into()
            .expect("four magic bytes inside a 348-byte head");
        let declared = magic_version(magic);
        let wanted = match version {
            Version::One => 1,
            Version::Two => 2,
        };
        if declared == wanted {
            return Ok((version, endian));
        }
        if declared == 0 && version == Version::One {
            return Err(NiftiError::AnalyzeDialect {
                magic: show_magic(&magic),
            });
        }
        return Err(NiftiError::VersionMismatch {
            header_bytes: version.header_bytes(),
            magic: show_magic(&bytes[at..at + width]),
            declared,
        });
    }
    Err(NiftiError::BadSizeofHdr {
        found: i32::from_le_bytes(raw),
    })
}

/// Decide which byte order the header's numeric fields are in.
///
/// On NIfTI-2 that is the order the sentinel matched in, and the capture has
/// no fixture separating that from any other rule, so this module follows the
/// one rule it can see.
///
/// On NIfTI-1 it is `dim[0]` that decides, and the sentinel is only a
/// fallback. Four fixtures force exactly this shape between them and no
/// simpler one survives all four:
///
/// | fixture | `dim[0]` read both ways | `sizeof_hdr` | measured `byteorder` |
/// |---|---|---|---|
/// | `bad_sizeof_swapped.nii` | 3 / 768 | swapped | `LSB_FIRST` |
/// | `endian_nifti1_int16_be.nii` | 768 / 3 | swapped | `MSB_FIRST` |
/// | `dimedge_dim0_zero.nii` | 0 / 0 | native | `LSB_FIRST` |
/// | `dimedge_dim0_eight.nii` | 8 / 2048 | native | `LSB_FIRST`, then `bad dim[0]` |
///
/// Row one rules out "the sentinel decides": its sentinel says big-endian and
/// the file is little-endian. Row two supplies the swapped arm. Rows three and
/// four supply the fallback, for a `dim[0]` that is zero and for one that is
/// out of range in both readings, and row four in particular shows the
/// fallback is reached rather than the read being refused.
///
/// **This is host-independent, which is not obvious and is worth the check.**
/// The reference reads `dim[0]` in the *host's* order first, so on a
/// big-endian machine it would try the other one first. It cannot matter: a
/// `dim[0]` in `1..=7` has a high byte of zero, so its swap is `n * 256`,
/// which is never in `1..=7` for any `n` in range, and 348 swapped is
/// 1543503872. Neither test can match both ways round, so the answer does not
/// depend on which order is tried first. `both_orders_cannot_match_at_once`
/// pins that.
fn field_endian(bytes: &[u8], version: Version, sentinel: Endian) -> Endian {
    match version {
        Version::Two => sentinel,
        Version::One => {
            let raw: [u8; 2] = take(bytes, 40);
            if (1..=MAX_RANK as i16).contains(&i16::from_le_bytes(raw)) {
                Endian::Little
            } else if (1..=MAX_RANK as i16).contains(&i16::from_be_bytes(raw)) {
                Endian::Big
            } else {
                sentinel
            }
        }
    }
}

/// Everything this module reads out of a NIfTI header.
struct Header {
    version: Version,
    endian: Endian,
    magic: Vec<u8>,
    single_file: bool,
    datatype: i16,
    bitpix: i16,
    dim: [i64; 8],
    pixdim: [f64; 8],
    vox_offset: f64,
    scl_slope: f64,
    scl_inter: f64,
    descrip: String,
    intent_name: String,
}

/// Read `N` bytes at `at` as a fixed array.
fn take<const N: usize>(bytes: &[u8], at: usize) -> [u8; N] {
    bytes[at..at + N]
        .try_into()
        .expect("caller checked the header length")
}

/// A fixed-width character field, trimmed at its first NUL and lossily
/// decoded.
fn text(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

impl Header {
    /// Parse the fields at the offsets the version fixes.
    ///
    /// Both layouts are pinned in the capture's `nifti1_vs_nifti2`
    /// `field_offsets` table, which reports `offsetof` and `sizeof` for every
    /// member out of the reference build rather than out of a header file.
    fn parse(bytes: &[u8], version: Version, endian: Endian) -> Result<Self, NiftiError> {
        let need = version.header_bytes();
        if bytes.len() < need {
            return Err(NiftiError::ShortHeader {
                found: bytes.len(),
                needed: need,
            });
        }
        let mut dim = [0i64; 8];
        let mut pixdim = [0f64; 8];
        let header = match version {
            Version::One => {
                for (i, slot) in dim.iter_mut().enumerate() {
                    *slot = i64::from(endian.i16(take(bytes, 40 + i * 2)));
                }
                for (i, slot) in pixdim.iter_mut().enumerate() {
                    *slot = f64::from(endian.f32(take(bytes, 76 + i * 4)));
                }
                Self {
                    version,
                    endian,
                    magic: bytes[MAGIC_1_AT..MAGIC_1_AT + 4].to_vec(),
                    single_file: magic_is_single_file(take(bytes, MAGIC_1_AT)),
                    datatype: endian.i16(take(bytes, 70)),
                    bitpix: endian.i16(take(bytes, 72)),
                    dim,
                    pixdim,
                    vox_offset: f64::from(endian.f32(take(bytes, 108))),
                    scl_slope: f64::from(endian.f32(take(bytes, 112))),
                    scl_inter: f64::from(endian.f32(take(bytes, 116))),
                    descrip: text(&bytes[148..228]),
                    intent_name: text(&bytes[328..344]),
                }
            }
            Version::Two => {
                for (i, slot) in dim.iter_mut().enumerate() {
                    *slot = endian.i64(take(bytes, 16 + i * 8));
                }
                for (i, slot) in pixdim.iter_mut().enumerate() {
                    *slot = endian.f64(take(bytes, 104 + i * 8));
                }
                Self {
                    version,
                    endian,
                    magic: bytes[MAGIC_2_AT..MAGIC_2_AT + 8].to_vec(),
                    single_file: magic_is_single_file(take(bytes, MAGIC_2_AT)),
                    datatype: endian.i16(take(bytes, 12)),
                    bitpix: endian.i16(take(bytes, 14)),
                    dim,
                    pixdim,
                    // `vox_offset` widened from `f32` to `i64` between the
                    // versions, so this one is exact where the NIfTI-1 field
                    // is not.
                    vox_offset: endian.i64(take(bytes, 168)) as f64,
                    scl_slope: endian.f64(take(bytes, 176)),
                    scl_inter: endian.f64(take(bytes, 184)),
                    descrip: text(&bytes[240..320]),
                    intent_name: text(&bytes[508..524]),
                }
            }
        };
        Ok(header)
    }

    /// The width and height this volume flattens to.
    ///
    /// The rank rules are the reference's, measured, and they are not
    /// uniform: rank 0 is a one-voxel image, a non-positive `dim[1]` is
    /// refused, and a zero extent on any higher axis is clamped to 1. The
    /// flattening itself is libviprs's own choice and follows
    /// `analyzeload`; see the module doc.
    fn geometry(&self) -> Result<(u32, u32), NiftiError> {
        let rank = self.dim[0];
        if !(0..=MAX_RANK as i64).contains(&rank) {
            return Err(NiftiError::BadRank {
                found: rank,
                max: MAX_RANK,
            });
        }
        if rank == 0 {
            // Measured: every extent is discarded and nvox becomes 1.
            return Ok((1, 1));
        }
        let rank = rank as usize;
        if self.dim[1] <= 0 {
            return Err(NiftiError::BadExtent { found: self.dim[1] });
        }
        let mut height: i64 = 1;
        for axis in 2..=rank {
            // Measured for zero on axis 2: the extent is silently rewritten
            // to 1 and the voxel count drops with it. A negative extent on a
            // higher axis is not in the capture and takes the same branch
            // here, which is libviprs choosing the clamp over a guess rather
            // than a measured answer.
            let extent = self.dim[axis].max(1);
            height = height
                .checked_mul(extent)
                .ok_or(NiftiError::DimensionOverflow { rank })?;
        }
        let width =
            u32::try_from(self.dim[1]).map_err(|_| NiftiError::DimensionOverflow { rank })?;
        let height = u32::try_from(height).map_err(|_| NiftiError::DimensionOverflow { rank })?;
        Ok((width, height))
    }

    /// Where the voxels start.
    ///
    /// Measured: `vox_offset` is truncated toward zero and then floored at
    /// the header length, so `-8` and `100` both give 348 and `352.5` gives
    /// 352. The floor for NIfTI-2 is the same expression with 540 in it; the
    /// capture has no NIfTI-2 fixture that exercises it, so that half is
    /// libviprs following one rule rather than two, not a measurement.
    fn data_offset(&self) -> u64 {
        let declared = if self.vox_offset.is_finite() {
            self.vox_offset.trunc()
        } else {
            0.0
        };
        let floor = self.version.header_bytes() as u64;
        if declared <= 0.0 {
            return floor;
        }
        // `f64` holds every `u64` up to 2^53 exactly, and anything past that
        // is far beyond any file, so the saturating cast is the honest one.
        let declared = declared as u64;
        declared.max(floor)
    }
}

/// Decode a single-file NIfTI (`.nii`) buffer into a [`Raster`].
///
/// The version, the byte order, the geometry and the sample carrier all come
/// out of the fixed-size header; everything after `vox_offset` is the voxel
/// array, x fastest, and this module reshapes it without moving a byte. See
/// the module doc for the repair rules the reference applies on the way in,
/// all of which are reproduced here.
///
/// # Errors
///
/// * [`NiftiError`] for every malformation the header carries, each named:
///   a short header, a bad sentinel, a magic that disagrees with the header
///   length, the two-file form, a bad rank or extent, a datatype with no
///   carrier, and a file that ends inside the voxel array.
/// * [`SourceError::AllocLimitExceeded`] when the declared geometry prices
///   past [`DecodeLimits::max_alloc_bytes`].
/// * [`SourceError::DimensionLimitExceeded`] when it is over
///   [`DecodeLimits::max_coord`] or [`DecodeLimits::max_pixels`].
///
/// # Example
///
/// ```no_run
/// use libviprs::nifti::decode_nifti;
/// use libviprs::source::DecodeLimits;
///
/// let bytes = std::fs::read("scan.nii")?;
/// let raster = decode_nifti(&bytes, DecodeLimits::default())?;
/// println!("{}x{}", raster.width(), raster.height());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn decode_nifti(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let (version, sentinel) = header_version(bytes)?;
    let endian = field_endian(bytes, version, sentinel);
    let header = Header::parse(bytes, version, endian)?;
    if !header.single_file {
        return Err(NiftiError::PairedHeader {
            magic: show_magic(&header.magic),
        }
        .into());
    }
    let carrier = carrier_for(header.datatype)?;
    let (width, height) = header.geometry()?;

    // Both ceilings go on the declared header geometry, before anything is
    // reserved, which is the whole point for a format that declares a
    // volume in 348 bytes and then hands over a raw array.
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    // One spelling of the budget for the whole crate (issue #632): the price
    // comes from `decode_alloc_bytes` and the comparison from
    // `DecodeLimits::exceeds_alloc_budget`, so neither can drift here on its
    // own, and the refusal is the shared `SourceError::AllocLimitExceeded`
    // rather than a sixth per-format variant (issue #686).
    let needed = limits.check_image_alloc(
        "NIfTI voxel buffer",
        width,
        height,
        carrier.bands(),
        carrier.sample_bytes(),
    )?;

    let offset = header.data_offset();
    let available = (bytes.len() as u64).saturating_sub(offset);
    if available < needed {
        return Err(NiftiError::TruncatedData {
            offset,
            found: available,
            needed,
        }
        .into());
    }
    let start = offset as usize;
    let payload = &bytes[start..start + needed as usize];

    let data = to_native(payload, carrier, endian);

    let mut raster = Raster::new_with_budget(
        width,
        height,
        carrier.pixel_format(),
        data,
        limits.max_alloc_bytes,
    )
    .map_err(NiftiError::Raster)?;
    raster.meta.interpretation = Some(carrier.interpretation());
    header.attach(&mut raster);
    Ok(raster)
}

/// Turn the file's voxel array into libviprs's native-endian buffer.
///
/// The array is already interleaved and x fastest, so the only work is the
/// byte order and, for `FLOAT32`, the reference's non-finite rewrite.
fn to_native(payload: &[u8], carrier: Carrier, endian: Endian) -> Vec<u8> {
    match carrier {
        // One byte per sample: byte order cannot apply, whatever the header
        // says about the rest of the file.
        Carrier::U8 | Carrier::Rgb24 | Carrier::Rgba32 => payload.to_vec(),
        Carrier::U16 => payload
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|c| endian.u16(*c).to_ne_bytes())
            .collect(),
        Carrier::F32 => payload
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|c| {
                let value = endian.f32(*c);
                // `nifti_read_buffer` sets every value failing `isfinite` to
                // 0 before the caller sees it, so an infinity or a NaN
                // stored in the file never comes back. Measured on
                // `float_float32.nii`: three words go to zero and the rest
                // are untouched.
                let value = if value.is_finite() { value } else { 0.0 };
                value.to_ne_bytes()
            })
            .collect(),
    }
}

impl Header {
    /// Attach every field this module read, so the axes the flattening
    /// collapsed are still recoverable.
    ///
    /// Named `nifti-<field>`, the way `analyzeload` names its own
    /// `dsr-<section>.<member>`.
    fn attach(&self, raster: &mut Raster) {
        let version = match self.version {
            Version::One => 1i64,
            Version::Two => 2,
        };
        let endian = match self.endian {
            Endian::Little => "LSB_FIRST",
            Endian::Big => "MSB_FIRST",
        };
        raster.fields.set("nifti-version", version.into());
        raster.fields.set("nifti-byte-order", endian.into());
        raster
            .fields
            .set("nifti-magic", show_magic(&self.magic).into());
        raster
            .fields
            .set("nifti-datatype", i64::from(self.datatype).into());
        raster
            .fields
            .set("nifti-bitpix", i64::from(self.bitpix).into());
        raster
            .fields
            .set("nifti-vox_offset", self.vox_offset.into());
        raster.fields.set("nifti-scl_slope", self.scl_slope.into());
        raster.fields.set("nifti-scl_inter", self.scl_inter.into());
        raster
            .fields
            .set("nifti-descrip", self.descrip.clone().into());
        raster
            .fields
            .set("nifti-intent_name", self.intent_name.clone().into());
        for (i, value) in self.dim.iter().enumerate() {
            raster
                .fields
                .set(&format!("nifti-dim[{i}]"), (*value).into());
        }
        for (i, value) in self.pixdim.iter().enumerate() {
            raster
                .fields
                .set(&format!("nifti-pixdim[{i}]"), (*value).into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One of the 104 fixtures `oracle-captures/foreign-nifti/capture.py`
    /// writes and pins.
    ///
    /// They are checked into the repository on purpose: `nifti_clib` is not a
    /// dependency of this crate and will not be on a CI machine, so nothing
    /// here can produce these inputs for itself.
    fn fixture(name: &str) -> Vec<u8> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/oracle-captures/foreign-nifti/fixtures/"
        );
        std::fs::read(format!("{path}{name}"))
            .unwrap_or_else(|e| panic!("fixture {name} must be readable: {e}"))
    }

    /// The four fixtures that decide the NIfTI-1 byte-order rule between
    /// them, shared by the test that sweeps them and the test that holds the
    /// capture's own record against that sweep, so the two cannot name
    /// different sets.
    const BYTE_ORDER_FIXTURES: [&str; 4] = [
        "bad_sizeof_swapped.nii",
        "dimedge_dim0_eight.nii",
        "dimedge_dim0_zero.nii",
        "endian_nifti1_int16_be.nii",
    ];

    fn decoded(name: &str) -> Raster {
        decode_nifti(&fixture(name), DecodeLimits::default())
            .unwrap_or_else(|e| panic!("{name} should decode: {e}"))
    }

    fn refused(name: &str) -> SourceError {
        decode_nifti(&fixture(name), DecodeLimits::default())
            .expect_err(&format!("{name} should be refused"))
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

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Overwrite a little-endian `i16` field in a NIfTI-1 header.
    fn poke_i16(bytes: &mut [u8], at: usize, value: i16) {
        bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }

    /// Overwrite a little-endian `f32` field in a NIfTI-1 header.
    fn poke_f32(bytes: &mut [u8], at: usize, value: f32) {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    /**
     * The carrier table's two halves have to agree: the [`PixelFormat`] each
     * carrier names and the (bands, sample bytes) pair it prices with are
     * two independent statements about the same thing, and the budget is
     * spent on the second while the raster is built from the first.
     *
     * A mismatch is exactly the shape that would price a file at one size
     * and then allocate another, so it is checked rather than reasoned.
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

    /**
     * The base success case: an eight-bit ramp comes back voxel for voxel,
     * x fastest, with the geometry `dim[1]` and `dim[2..=rank]` give.
     *
     * Every value is the oracle's own, from `sample_values.by_code["2"]`:
     * payload bytes 128..133 and voxel (0,0) = 128, (1,0) = 129,
     * (0,1) = 130. The fixture is `dim = [3, 2, 3, 1]`, so it is 2 wide and
     * 3 high and the third axis of extent 1 folds away.
     * Input: `dt2_uint8.nii` -> Output: 2x3 `Gray8`, `b-w`, the six ramp
     * bytes in file order.
     */
    #[test]
    fn a_uint8_ramp_loads_the_reference_voxels() {
        let raster = decoded("dt2_uint8.nii");
        assert_eq!((raster.width(), raster.height()), (2, 3));
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(raster.meta.interpretation, Some(Interpretation::Bw));
        assert_eq!(raster.data(), &[128, 129, 130, 131, 132, 133]);
    }

    /**
     * Sixteen-bit samples are assembled in the file's byte order and stored
     * in the host's, which is the one thing a byte-for-byte copy would get
     * wrong on a big-endian host and never show here.
     *
     * The six values are the oracle's, from `sample_values.by_code["512"]`:
     * the same 128.. byte ramp read as little-endian `u16`.
     * Input: `dt512_uint16.nii` -> Output: 2x3 `Gray16` carrying
     * 33152, 33666, 34180, 34694, 35208, 35722.
     */
    #[test]
    fn a_uint16_ramp_is_reassembled_in_the_files_byte_order() {
        let raster = decoded("dt512_uint16.nii");
        assert_eq!((raster.width(), raster.height()), (2, 3));
        assert_eq!(raster.format(), PixelFormat::Gray16);
        assert_eq!(
            u16_samples(&raster),
            vec![33152, 33666, 34180, 34694, 35208, 35722]
        );
    }

    /**
     * The same sixteen-bit array written the other way round comes back
     * identical, which is what actually pins the swap. The test above does
     * not: every `UINT16` fixture in the capture is little-endian, so on a
     * little-endian host a decoder that hard-coded `from_le_bytes` would
     * pass it. **Measured, by mutating exactly that and watching the test
     * stay green.**
     *
     * The big-endian half is `endian_nifti1_int16_be.nii`, which the
     * reference wrote through its own `nifti_swap_as_nifti1`, with its
     * datatype poked from `INT16` (4) to `UINT16` (512). Nothing else about
     * the file changes: `nbyper` is 2 either way, `bitpix` is already 16, and
     * the payload is untouched. Its on-disk bytes are `8180 8382 ...` against
     * the little-endian file's `8081 8283 ...`, and the oracle records both
     * files loading to the same memory, `808182838485868788898a8b`.
     * Input: the big-endian file as `UINT16` -> Output: the six values
     * `dt512_uint16.nii` gives.
     */
    #[test]
    fn a_big_endian_uint16_array_is_swapped_into_host_order() {
        let mut be = fixture("endian_nifti1_int16_be.nii");
        assert_eq!(
            &be[352..364],
            &[
                0x81, 0x80, 0x83, 0x82, 0x85, 0x84, 0x87, 0x86, 0x89, 0x88, 0x8b, 0x8a
            ],
            "the fixture has to be the byte-swapped one, or this proves nothing"
        );
        // datatype is a big-endian i16 in this file.
        be[70..72].copy_from_slice(&512i16.to_be_bytes());
        let raster = decode_nifti(&be, DecodeLimits::default()).expect("uint16, big-endian");
        assert_eq!((raster.width(), raster.height()), (2, 3));
        assert_eq!(raster.format(), PixelFormat::Gray16);
        assert_eq!(
            u16_samples(&raster),
            vec![33152, 33666, 34180, 34694, 35208, 35722],
            "the same six values the little-endian file gives"
        );
        assert_eq!(
            raster.data(),
            decoded("dt512_uint16.nii").data(),
            "and byte for byte the same buffer"
        );
    }

    /**
     * `RGB24` and `RGBA32` are the only two codes that give more than one
     * band, and their samples are interleaved in the file rather than
     * planar, so the payload copies through untouched.
     *
     * Input: `dt128_rgb24.nii` / `dt2304_rgba32.nii` -> Output: 2x3 `Rgb8`
     * over bytes 128..145 and 2x3 `Rgba8` over bytes 128..151, both sRGB.
     */
    #[test]
    fn rgb24_and_rgba32_carry_interleaved_bands() {
        let rgb = decoded("dt128_rgb24.nii");
        assert_eq!((rgb.width(), rgb.height()), (2, 3));
        assert_eq!(rgb.format(), PixelFormat::Rgb8);
        assert_eq!(rgb.meta.interpretation, Some(Interpretation::Srgb));
        assert_eq!(rgb.data(), &(128u8..=145).collect::<Vec<_>>()[..]);

        let rgba = decoded("dt2304_rgba32.nii");
        assert_eq!((rgba.width(), rgba.height()), (2, 3));
        assert_eq!(rgba.format(), PixelFormat::Rgba8);
        assert_eq!(rgba.data(), &(128u8..=151).collect::<Vec<_>>()[..]);
    }

    /**
     * The loader rewrites pixel data, and this is the case that proves it.
     * `nifti_read_buffer` sets every `FLOAT32` sample failing `isfinite` to
     * zero before the caller ever sees it, so an infinity or a NaN stored in
     * a NIfTI file never comes back.
     *
     * The two hex strings are the oracle's own, from
     * `sample_values.float_and_complex.cases.float32`: `payload_hex` is what
     * is on disk and `data_hex_after_load` is what the reference hands over,
     * and they differ in exactly the last three words.
     *
     * The finite half is the positive control that the rewrite is not just
     * zeroing everything: 0.0, 1.0, -1.0, 0.5 and the denormal 1e-40 all
     * survive, and the denormal is the interesting one because it is the
     * value a plausibility check would also throw away.
     * Input: `float_float32.nii` -> Output: 4x2 `FloatF32(1)` whose bytes are
     * `data_hex_after_load`.
     */
    #[test]
    fn non_finite_float_samples_are_rewritten_to_zero() {
        let raster = decoded("float_float32.nii");
        assert_eq!((raster.width(), raster.height()), (4, 2));
        assert_eq!(
            raster.format(),
            PixelFormat::with_channels(1, 4).expect("one float band")
        );
        assert_eq!(
            hex(raster.data()),
            "000000000000803f000080bf0000003fc2160100000000000000000000000000",
            "the loaded buffer is the oracle's data_hex_after_load, not its payload_hex"
        );

        let samples = f32_samples(&raster);
        assert_eq!(samples[..4], [0.0, 1.0, -1.0, 0.5]);
        assert!(
            samples[4] > 0.0 && samples[4] < 1e-39,
            "the denormal 1e-40 survives, so this is a finiteness test and not a \
             plausibility one; got {}",
            samples[4]
        );
        assert_eq!(
            samples[5..],
            [0.0, 0.0, 0.0],
            "inf, -inf and NaN all arrive as plain zeros"
        );
    }

    /**
     * The same eight float values written the other way round load to
     * identical memory, which is the reference's own claim
     * (`matches_little_endian: true`) and the single most likely thing for a
     * port on a little-endian host to get backwards.
     *
     * The positive control is the non-finite rewrite happening on both
     * sides: if the big-endian path skipped the swap, the three sentinel
     * words would not be recognised as non-finite and would come back as
     * garbage rather than as zeros.
     * Input: `float_float32.nii` and `float_float32_be.nii` -> Output:
     * byte-identical rasters.
     */
    #[test]
    fn the_two_byte_orders_load_to_identical_memory() {
        let le = decoded("float_float32.nii");
        let be = decoded("float_float32_be.nii");
        assert_eq!((le.width(), le.height()), (be.width(), be.height()));
        assert_eq!(le.format(), be.format());
        assert_eq!(hex(le.data()), hex(be.data()));
        assert_eq!(
            hex(be.data()),
            "000000000000803f000080bf0000003fc2160100000000000000000000000000"
        );
    }

    /**
     * Every axis above the second folds into the height, which is
     * `analyzeload`'s measured rule and the choice this module makes for a
     * format libvips cannot load at all.
     *
     * The arithmetic is checked against something independent of the rule:
     * each fixture's own file length. `nifti_clib` wrote 352 header bytes
     * and then exactly `nvox` bytes, so `352 + width * height` has to be the
     * length of the file for every row, and a wrong fold would break that
     * before it broke any assertion about a number I chose.
     * Input: the six `dim_rank*.nii` fixtures -> Output: `dim[1]` wide,
     * `dim[2] * .. * dim[rank]` high, and `width * height` voxels.
     */
    #[test]
    fn every_axis_above_the_second_folds_into_the_height() {
        let cases = [
            ("dim_rank1_6.nii", 6, 1),
            ("dim_rank2_2x3.nii", 2, 3),
            ("dim_rank3_2x3x2.nii", 2, 6),
            ("dim_rank4_2x3x2x2.nii", 2, 12),
            ("dim_rank5_2x2x2x2x2.nii", 2, 16),
            ("dim_rank7_all2.nii", 2, 64),
        ];
        for (name, width, height) in cases {
            let bytes = fixture(name);
            let raster = decoded(name);
            assert_eq!(
                (raster.width(), raster.height()),
                (width, height),
                "{name} folds to the wrong geometry"
            );
            assert_eq!(
                bytes.len(),
                DEFAULT_DATA_OFFSET_1 as usize + (width * height) as usize,
                "{name}'s own length disagrees with the fold, so the fold is wrong"
            );
        }
    }

    /**
     * The version check needs 348 bytes and refuses everything shorter, even
     * a NIfTI-2 file whose sentinel and magic both live in the first twelve.
     * That is measured rather than reasoned: the oracle's
     * `nifti_header_version` sweep returns -1 at 347 bytes for a NIfTI-2
     * file and 2 at 348.
     *
     * The positive control is the 348th byte: `bad_trunc348.nii` is 348
     * bytes and gets past the version check to a different refusal
     * altogether, so the floor is the floor and not a blanket "short files
     * are bad".
     * Input: 0, 1, 100 and 347 bytes -> Output: `ShortHeader`; 348 bytes ->
     * something else.
     */
    #[test]
    fn the_version_check_needs_348_bytes_and_not_one_fewer() {
        for name in [
            "bad_empty.nii",
            "bad_onebyte.nii",
            "bad_trunc100.nii",
            "bad_trunc347.nii",
        ] {
            assert!(
                matches!(
                    refused(name),
                    SourceError::Nifti(NiftiError::ShortHeader { needed: 348, .. })
                ),
                "{name} should be short of the 348-byte version floor"
            );
        }
        // 347 bytes of a NIfTI-2 file is still short, which is the half of
        // the rule that looks wrong and is measured.
        let mut n2 = fixture("ver_n2_single.nii");
        n2.truncate(347);
        assert!(matches!(
            decode_nifti(&n2, DecodeLimits::default()).unwrap_err(),
            SourceError::Nifti(NiftiError::ShortHeader { .. })
        ));

        assert!(
            !matches!(
                refused("bad_trunc348.nii"),
                SourceError::Nifti(NiftiError::ShortHeader { .. })
            ),
            "348 bytes is past the version floor, so the refusal must be a later one"
        );
    }

    /**
     * `sizeof_hdr` is the only thing that says which header layout a file
     * has, and the magic has to agree with it.
     *
     * Input: `sizeof_hdr` of 0, 349, and 540-with-a-NIfTI-1-magic, plus a
     * 348-byte header carrying `n+2` -> Output: `BadSizeofHdr` for the first
     * two and `VersionMismatch` for the last two, matching the oracle's
     * `nifti_header_version` of -1 for all four.
     */
    #[test]
    fn the_sentinel_and_the_magic_have_to_agree_about_the_version() {
        for name in ["bad_sizeof0.nii", "bad_sizeof349.nii"] {
            assert!(
                matches!(
                    refused(name),
                    SourceError::Nifti(NiftiError::BadSizeofHdr { .. })
                ),
                "{name} declares a header length that is neither 348 nor 540"
            );
        }
        for (name, header_bytes) in [("bad_sizeof540.nii", 540), ("bad_magic_n2_in_n1.nii", 348)] {
            match refused(name) {
                SourceError::Nifti(NiftiError::VersionMismatch {
                    header_bytes: got, ..
                }) => {
                    assert_eq!(got, header_bytes, "{name} names the wrong header length");
                }
                other => panic!("{name} should be a version mismatch, got {other}"),
            }
        }
    }

    /**
     * On NIfTI-1 it is `dim[0]` that decides the byte order and the sentinel
     * is only the fallback, which is the opposite of what the capture's
     * `byte_order` prose says and what its own `bad_sizeof_swapped` record
     * measures.
     *
     * `bad_sizeof_swapped.nii` is a little-endian file with only its four
     * sentinel bytes swapped. Under "the sentinel decides" every field would
     * be read big-endian, `dim` would come back `768 512 768 256` and the
     * load would fail on the rank. The oracle loads it `LSB_FIRST` with
     * `dim = 3 2 3 1` and hands back the twelve payload bytes, so `dim[0]`
     * is what won.
     *
     * The other three rows are the arms the first one does not reach: a
     * genuinely big-endian file, a `dim[0]` of zero, and a `dim[0]` that is
     * out of range both ways and therefore falls through to the sentinel
     * rather than being refused there.
     * Input: four fixtures -> Output: the byte order each one loads in.
     */
    #[test]
    fn dim0_decides_the_byte_order_and_the_sentinel_is_only_a_fallback() {
        let cases = [
            ("bad_sizeof_swapped.nii", Endian::Little),
            ("endian_nifti1_int16_be.nii", Endian::Big),
            ("dimedge_dim0_zero.nii", Endian::Little),
            ("dimedge_dim0_eight.nii", Endian::Little),
        ];
        let mut swept: Vec<&str> = cases.iter().map(|(name, _)| *name).collect();
        swept.sort_unstable();
        assert_eq!(
            swept,
            BYTE_ORDER_FIXTURES.to_vec(),
            "the sweep and the capture check have to name the same four files"
        );
        for (name, want) in cases {
            let bytes = fixture(name);
            let (version, sentinel) = header_version(&bytes).expect("version");
            assert_eq!(version, Version::One, "{name} is a NIfTI-1 file");
            assert_eq!(
                field_endian(&bytes, version, sentinel),
                want,
                "{name} loads {want:?} in the oracle"
            );
        }

        // The sentinel of the first row really does say the other thing, so
        // the row is testing what it claims to.
        let swapped = fixture("bad_sizeof_swapped.nii");
        assert_eq!(
            header_version(&swapped).expect("version").1,
            Endian::Big,
            "bad_sizeof_swapped's sentinel matches big-endian, which is the whole point"
        );
        // And it loads to the little-endian reading end to end: the datatype
        // is INT16 (4), not the 1024 a big-endian read would give.
        assert!(
            matches!(
                refused("bad_sizeof_swapped.nii"),
                SourceError::Nifti(NiftiError::UnsupportedCarrier { datatype: 4, .. })
            ),
            "the whole header follows dim[0], not just the geometry"
        );
    }

    /**
     * The capture and this module have to state the same byte-order rule.
     *
     * `oracle-captures/foreign-nifti/`'s `byte_order` record used to say the
     * sentinel decides, which its own `bad_sizeof_swapped` measurement
     * contradicts, and that is issue #752. Fixing prose fixes nothing on its
     * own, so this reads the committed capture back and asserts it names the
     * same four fixtures the test above sweeps. Renaming a fixture on either
     * side, or quietly dropping one from the record, is now red.
     *
     * Input: `records.byte_order.which_byte_order_actually_wins` ->
     * Output: the four fixture names, and a rule with four arms.
     */
    #[test]
    fn the_capture_states_the_byte_order_rule_this_module_implements() {
        // `include_str!`, not a read: the capture is a committed artefact, so
        // coupling to it at compile time is both stronger (deleting it stops
        // the crate building) and cheaper (this test reaches no filesystem, so
        // it is not a Miri gate-killer the way a `read_to_string` would be).
        const CAPTURE: &str = include_str!("../oracle-captures/foreign-nifti/oracle.json");
        let doc: serde_json::Value = serde_json::from_str(CAPTURE).expect("the capture must parse");
        let record = &doc["records"]["byte_order"]["which_byte_order_actually_wins"];
        assert!(
            !record.is_null(),
            "the capture does not record which byte order actually wins (issue #752)"
        );

        let mut named: Vec<&str> = record["the_four_fixtures_that_force_it"]
            .as_object()
            .expect("the record lists the fixtures it rests on")
            .keys()
            .map(String::as_str)
            .collect();
        named.sort_unstable();
        assert_eq!(
            named,
            BYTE_ORDER_FIXTURES.to_vec(),
            "the capture and dim0_decides_the_byte_order_and_the_sentinel_is_only_a_fallback \
             must rest on the same fixtures"
        );
        assert_eq!(
            record["rule"]
                .as_array()
                .expect("the record spells the rule out")
                .len(),
            4,
            "the rule is dim[0] both ways, then sizeof_hdr both ways"
        );
        // The positive control that those four files exist and say what the
        // record says is the sweep itself, which reads every one of them.
    }

    /**
     * The byte-order rule cannot depend on which order a host tries first,
     * because no input satisfies both readings at once. That is what makes
     * this module's answer the same on a big-endian machine as the oracle's
     * on the little-endian one it was captured on.
     *
     * Exhaustive over both tests rather than argued: every `i16` for the
     * rank test and both spellings of the sentinel for the fallback.
     * Input: every `i16` -> Output: never in `1..=7` in both byte orders.
     */
    #[test]
    fn both_orders_cannot_match_at_once() {
        let rank = 1..=MAX_RANK as i16;
        let mut ambiguous = 0;
        for raw in i16::MIN..=i16::MAX {
            let bytes = raw.to_le_bytes();
            if rank.contains(&i16::from_le_bytes(bytes))
                && rank.contains(&i16::from_be_bytes(bytes))
            {
                ambiguous += 1;
            }
        }
        assert_eq!(
            ambiguous, 0,
            "a dim[0] that is a valid rank both ways round"
        );
        // The positive control: the sweep does find ranks, so a zero above
        // is a real absence rather than a loop that never entered the test.
        let found = (i16::MIN..=i16::MAX)
            .filter(|raw| rank.contains(&i16::from_le_bytes(raw.to_le_bytes())))
            .count();
        assert_eq!(found, MAX_RANK, "the sweep does reach the ranks 1..=7");

        let sentinel = HEADER_1_BYTES as i32;
        assert_ne!(
            i32::from_be_bytes(sentinel.to_le_bytes()),
            sentinel,
            "348 is not a palindrome, so the sentinel fallback is unambiguous too"
        );
    }

    /**
     * A 348-byte header whose magic is not a NIfTI one is the Analyze 7.5
     * dialect, and the reference then asks the *filename* which container it
     * is. A buffer decode has no filename, so this module says so by name
     * rather than guessing at 348 or 352.
     *
     * `bad_magic_nonul.nii` is the sharp one: `n+1x` is one byte away from
     * valid and the oracle reads it as version 0, not as a NIfTI-1 file, so
     * the terminating NUL is load-bearing.
     * Input: `XY\0\0`, `n+1x` and an all-zero magic -> Output:
     * `AnalyzeDialect`.
     */
    #[test]
    fn a_magic_that_is_not_niftis_is_the_analyze_dialect() {
        for name in [
            "bad_magic.nii",
            "bad_magic_nonul.nii",
            "magic_zero_analyze.nii",
        ] {
            assert!(
                matches!(
                    refused(name),
                    SourceError::Nifti(NiftiError::AnalyzeDialect { .. })
                ),
                "{name} carries no NIfTI magic, so it is the Analyze dialect"
            );
        }
        // Positive control: the same file with a good magic is not refused
        // for this reason, so the check is on the magic and not on the
        // fixture.
        let mut fixed = fixture("bad_magic.nii");
        fixed[MAGIC_1_AT..MAGIC_1_AT + 4].copy_from_slice(MAGIC_1_SINGLE);
        assert!(!matches!(
            decode_nifti(&fixed, DecodeLimits::default()).unwrap_err(),
            SourceError::Nifti(NiftiError::AnalyzeDialect { .. })
        ));
    }

    /**
     * Only the first four bytes of the eight-byte NIfTI-2 magic are read.
     * The `\r \n \032 \n` tail the spec puts there to catch a text-mode
     * transfer is never examined, which is the opposite of what the
     * rationale implies and is measured: a file with that tail destroyed
     * loads normally.
     *
     * The positive control is mangling the first four bytes instead, which
     * does change the answer, so this is "the tail is not read" rather than
     * "the magic is not read".
     * Input: `n2_magic_tail_mangled.nii` -> Output: NIfTI-2, recognised.
     */
    #[test]
    fn the_nifti2_magic_tail_is_never_checked() {
        for name in ["n2_magic_tail_mangled.nii", "n2_magic_tail_partial.nii"] {
            let bytes = fixture(name);
            assert_ne!(
                &bytes[MAGIC_2_AT..MAGIC_2_AT + 8],
                &MAGIC_2_SINGLE[..],
                "{name} must actually have a damaged magic, or it proves nothing"
            );
            assert_eq!(
                header_version(&bytes).expect("version").0,
                Version::Two,
                "{name} is still a NIfTI-2 file"
            );
        }
        let mut head_mangled = fixture("n2_magic_tail_mangled.nii");
        head_mangled[MAGIC_2_AT + 2] = b'9';
        assert!(
            header_version(&head_mangled).is_err(),
            "the first four bytes are read, so damaging them does change the answer"
        );
    }

    /**
     * `magic[1] == '+'` is the whole of the one-file test. A `ni1` or `ni2`
     * header is the header half of a `.hdr` / `.img` pair and carries no
     * voxels at all, so decoding one from a buffer cannot work and says so.
     *
     * `nii_with_ni1_magic.nii` is the case that shows the filename is not
     * consulted: the oracle loads it as one file *because it is called
     * `.nii`*, and a buffer decode has no name to go on, so this module
     * refuses where the oracle succeeds. That is a deliberate divergence and
     * it is recorded here rather than hidden.
     * Input: `pair_n1.hdr`, `pair_n2.hdr`, `nii_with_ni1_magic.nii` ->
     * Output: `PairedHeader`.
     */
    #[test]
    fn a_two_file_header_says_where_its_voxels_are() {
        for name in ["pair_n1.hdr", "pair_n2.hdr", "nii_with_ni1_magic.nii"] {
            assert!(
                matches!(
                    refused(name),
                    SourceError::Nifti(NiftiError::PairedHeader { .. })
                ),
                "{name} is the header half of a pair"
            );
        }
        // Positive control: the single-file spelling of the same NIfTI-2
        // header is not refused for this reason.
        assert!(!matches!(
            refused("ver_n2_single.nii"),
            SourceError::Nifti(NiftiError::PairedHeader { .. })
        ));
    }

    /**
     * The rank and extent repairs are not uniform, and reproducing the
     * asymmetry is the point: rank 0 is accepted as a one-voxel image, rank
     * 8 and rank -1 are refused, `dim[1]` of 0 or -3 is refused, and a zero
     * extent on any *higher* axis is silently clamped to 1.
     *
     * The last row is the one with an independent check behind it: the
     * oracle reports `nvox` dropping from 12 to 4 for
     * `dimedge_dim2_zero_mid_array.nii`, and 4 is what a 2x2 raster holds.
     * Input: the six `dimedge_dim*` fixtures -> Output: the oracle's own
     * verdict for each.
     */
    #[test]
    fn the_rank_and_extent_repairs_are_not_uniform() {
        let rank0 = decoded("dimedge_dim0_zero.nii");
        assert_eq!(
            (rank0.width(), rank0.height()),
            (1, 1),
            "rank 0 discards every extent and gives one voxel"
        );

        for name in ["dimedge_dim0_eight.nii", "dimedge_dim0_negative.nii"] {
            assert!(
                matches!(
                    refused(name),
                    SourceError::Nifti(NiftiError::BadRank { .. })
                ),
                "{name} declares a rank outside 0..=7"
            );
        }
        for name in ["dimedge_dim1_zero.nii", "dimedge_dim1_negative.nii"] {
            assert!(
                matches!(
                    refused(name),
                    SourceError::Nifti(NiftiError::BadExtent { .. })
                ),
                "{name} declares a non-positive first axis, which is the one that is refused"
            );
        }

        let clamped = decoded("dimedge_dim2_zero_mid_array.nii");
        assert_eq!(
            (clamped.width(), clamped.height()),
            (2, 2),
            "a zero extent above the first axis is clamped to 1, not refused"
        );
        assert_eq!(
            clamped.data().len(),
            4,
            "the oracle's nvox for this file is 4, down from the 12 dim[3] would give"
        );
    }

    /**
     * The whole reason [`DecodeLimits`] matters more here than anywhere:
     * 348 bytes declare a 35-teravoxel volume and the rest of the file is 12
     * bytes long. Nothing may be reserved before the declared geometry is
     * priced.
     *
     * The oracle gets as far as `data bytes needed = 35181150961663` before
     * giving up, which is 32767 cubed, so the header check on that side does
     * not catch it at all.
     *
     * The positive control lifts every ceiling and shows the same file then
     * reaches the *payload* check instead, which is what proves the refusal
     * came from the budget rather than from the file being malformed some
     * other way.
     * Input: `dimedge_dim_all_32767.nii` -> Output:
     * `DimensionLimitExceeded`; with the ceilings lifted, `TruncatedData`.
     */
    #[test]
    fn a_teravoxel_volume_is_refused_before_anything_is_allocated() {
        match refused("dimedge_dim_all_32767.nii") {
            SourceError::CoordLimitExceeded {
                width,
                height,
                max_coord,
            } => {
                assert_eq!((width, height), (32_767, 32_767 * 32_767));
                assert_eq!(max_coord, crate::imageio::DEFAULT_MAX_COORD);
            }
            other => panic!("the coordinate ceiling is the first one this trips: {other:?}"),
        }
        // And the ceiling under it would have caught it too, so the refusal
        // does not rest on one arm: with the coordinate ceiling lifted the
        // pixel count is still 1.07e9 times 32767.
        assert!(matches!(
            decode_nifti(
                &fixture("dimedge_dim_all_32767.nii"),
                DecodeLimits::default().with_max_coord(u32::MAX)
            ),
            Err(SourceError::DimensionLimitExceeded { .. })
        ));
        // And under that, the allocation budget.
        assert!(matches!(
            decode_nifti(
                &fixture("dimedge_dim_all_32767.nii"),
                DecodeLimits::default()
                    .with_max_coord(u32::MAX)
                    .with_max_pixels(u64::MAX)
            ),
            Err(SourceError::AllocLimitExceeded { .. })
        ));

        let lifted = DecodeLimits::default()
            .with_max_coord(u32::MAX)
            .with_max_pixels(u64::MAX)
            .with_max_alloc_bytes(u64::MAX);
        match decode_nifti(&fixture("dimedge_dim_all_32767.nii"), lifted) {
            Err(SourceError::Nifti(NiftiError::TruncatedData { needed, found, .. })) => {
                assert_eq!(
                    needed, 35_181_150_961_663,
                    "the price is the oracle's own `data bytes needed`"
                );
                assert_eq!(found, 12, "and the file really does hold 12 payload bytes");
            }
            other => panic!("with the ceilings lifted this should reach the payload: {other:?}"),
        }
    }

    /**
     * The allocation budget refuses the declared geometry at one byte under
     * the price and admits it at the price, reported as the crate's one
     * shared `SourceError::AllocLimitExceeded` rather than a sixth
     * per-format variant.
     *
     * Input: `dt2_uint8.nii`, which prices at 2 * 3 * 1 * 1 = 6 bytes ->
     * Output: refused at 5, decoded at 6.
     */
    #[test]
    fn the_allocation_budget_is_spent_on_the_declared_geometry() {
        let bytes = fixture("dt2_uint8.nii");
        let exact = decode_nifti(&bytes, DecodeLimits::default().with_max_alloc_bytes(6))
            .expect("six bytes is exactly the price");
        assert_eq!(exact.data().len(), 6);

        match decode_nifti(&bytes, DecodeLimits::default().with_max_alloc_bytes(5)) {
            Err(SourceError::AllocLimitExceeded {
                what,
                needed_bytes,
                max_alloc_bytes,
                geometry,
            }) => {
                assert_eq!(what, "NIfTI voxel buffer");
                assert_eq!((needed_bytes, max_alloc_bytes), (6, 5));
                let geometry = geometry.expect("the refusal reports the declared geometry");
                assert_eq!((geometry.width, geometry.height, geometry.bands), (2, 3, 1));
            }
            other => panic!("one byte under the price must be refused: {other:?}"),
        }
    }

    /**
     * `vox_offset` is repaired rather than trusted: truncated toward zero
     * and then floored at the header length. Measured on three fixtures,
     * whose datatype is INT16 and which therefore never reach a raster, so
     * the rule is pinned on the header parse directly.
     *
     * Input: `vox_offset` of -8, 100 and 352.5 -> Output: 348, 348 and 352,
     * which is what the oracle reports as `iname_offset` for each.
     */
    #[test]
    fn vox_offset_is_truncated_down_and_floored_at_the_header() {
        let cases = [
            ("bad_voxoff_neg.nii", 348),
            ("bad_voxoff_small.nii", 348),
            ("bad_voxoff_frac.nii", 352),
            ("dt2_uint8.nii", 352),
            ("bad_voxoff_eof.nii", 1_000_000),
        ];
        for (name, want) in cases {
            let bytes = fixture(name);
            let (version, sentinel) = header_version(&bytes).expect("version");
            let endian = field_endian(&bytes, version, sentinel);
            let header = Header::parse(&bytes, version, endian).expect("header");
            assert_eq!(
                header.data_offset(),
                want,
                "{name} should start its voxels at {want}"
            );
        }
    }

    /**
     * `bitpix` is decoration: the datatype alone fixes the sample width, and
     * a header whose `bitpix` disagrees loads exactly as if it did not.
     *
     * Measured on `bad_bitpix.nii`, which declares datatype 4 with bitpix
     * 64 and loads as 16-bit. That fixture's datatype has no carrier here,
     * so the rule is exercised on a `uint8` file with its `bitpix` poked to
     * 64: if this module ever consulted the field it would price four times
     * as many bytes and refuse the file as short.
     * Input: `dt2_uint8.nii` with `bitpix` 64 -> Output: the same raster.
     */
    #[test]
    fn bitpix_is_decoration() {
        let plain = decoded("dt2_uint8.nii");
        let mut poked = fixture("dt2_uint8.nii");
        poke_i16(&mut poked, 72, 64);
        let loaded = decode_nifti(&poked, DecodeLimits::default())
            .expect("bitpix is not consulted for the geometry");
        assert_eq!(loaded.format(), plain.format());
        assert_eq!(loaded.data(), plain.data());
        assert_eq!(
            loaded.fields.get("nifti-bitpix"),
            Some(&crate::imageio::MetadataValue::Int(64)),
            "it is still carried, just never acted on"
        );
    }

    /**
     * `scl_slope` and `scl_inter` are carried and never applied. The
     * reference returns raw voxels; the `y = slope * x + inter` rule lives
     * in FSL rather than in `nifti_clib`, measured across every `scl_*`
     * fixture.
     *
     * Those fixtures are all INT16, so the rule is exercised on a `uint8`
     * file with the pair poked to (2, -3): a loader that scaled would turn
     * the 128.. ramp into 253.. and this would go red.
     * Input: `dt2_uint8.nii` with slope 2 and intercept -3 -> Output: the
     * unscaled ramp, with both values attached.
     */
    #[test]
    fn scl_slope_and_inter_are_carried_but_never_applied() {
        let mut poked = fixture("dt2_uint8.nii");
        poke_f32(&mut poked, 112, 2.0);
        poke_f32(&mut poked, 116, -3.0);
        let raster = decode_nifti(&poked, DecodeLimits::default()).expect("scaling is not applied");
        assert_eq!(raster.data(), &[128, 129, 130, 131, 132, 133]);
        assert_eq!(
            raster.fields.get("nifti-scl_slope"),
            Some(&crate::imageio::MetadataValue::Double(2.0))
        );
        assert_eq!(
            raster.fields.get("nifti-scl_inter"),
            Some(&crate::imageio::MetadataValue::Double(-3.0))
        );
    }

    /**
     * Every datatype the reference loads and libviprs has no carrier for is
     * refused **by name**, not narrowed. Narrowing a signed 16-bit array
     * into 8 bits would lose data silently, which is worse than failing,
     * and it is the same ceiling `crate::fits` refuses a signed BITPIX at.
     *
     * The sweep is over the capture's own datatype fixtures, so a code that
     * quietly started loading, or started loading as the wrong thing, lands
     * here.
     * Input: eleven `dt*` fixtures plus two invalid codes -> Output: the
     * variant and the code each one reports.
     */
    #[test]
    fn every_datatype_without_a_carrier_is_refused_by_name() {
        let carried = [
            ("dt256_int8.nii", 256i16, "INT8"),
            ("dt4_int16.nii", 4, "INT16"),
            ("dt8_int32.nii", 8, "INT32"),
            ("dt768_uint32.nii", 768, "UINT32"),
            ("dt64_float64.nii", 64, "FLOAT64"),
            ("dt1024_int64.nii", 1024, "INT64"),
            ("dt1280_uint64.nii", 1280, "UINT64"),
            ("dt1536_float128.nii", 1536, "FLOAT128"),
            ("dt32_complex64.nii", 32, "COMPLEX64"),
            ("dt1792_complex128.nii", 1792, "COMPLEX128"),
            ("dt2048_complex256.nii", 2048, "COMPLEX256"),
        ];
        for (fixture_name, code, name) in carried {
            match refused(fixture_name) {
                SourceError::Nifti(NiftiError::UnsupportedCarrier {
                    datatype,
                    name: got,
                    ..
                }) => {
                    assert_eq!((datatype, got), (code, name), "{fixture_name}");
                }
                other => panic!("{fixture_name} should be refused by name, got {other}"),
            }
        }

        match refused("bad_dt1.nii") {
            SourceError::Nifti(NiftiError::EmptyDatatype { datatype, name }) => {
                assert_eq!((datatype, name), (1, "DT_BINARY"));
            }
            other => panic!("DT_BINARY carries no data: {other}"),
        }
        for (fixture_name, code) in [("bad_dt3.nii", 3i16), ("bad_dt9999.nii", 9999)] {
            match refused(fixture_name) {
                SourceError::Nifti(NiftiError::UnknownDatatype { datatype }) => {
                    assert_eq!(datatype, code);
                }
                other => panic!("{fixture_name} is not a datatype at all: {other}"),
            }
        }

        // The positive control: the five codes that *do* have a carrier all
        // load, so the sweep above is a boundary and not a blanket refusal.
        for name in [
            "dt2_uint8.nii",
            "dt512_uint16.nii",
            "dt16_float32.nii",
            "dt128_rgb24.nii",
            "dt2304_rgba32.nii",
        ] {
            decoded(name);
        }
    }

    /**
     * A file that ends inside its voxel array is refused, and so is one
     * whose `vox_offset` points past the end. The reference warns
     * (`number missing = N (set to 0)`) and then fails the load, so the
     * zero-fill never reaches a caller on either side.
     *
     * The fixtures for this are all INT16, so the case is built by
     * truncating the `uint8` one, which is the same shape the oracle
     * measured on `bad_halfdata.nii`.
     * Input: `dt2_uint8.nii` cut short by one byte -> Output:
     * `TruncatedData` naming the offset, what was there and what was needed.
     */
    #[test]
    fn a_short_voxel_array_is_refused_rather_than_zero_filled() {
        let full = fixture("dt2_uint8.nii");
        for missing in 1..=6usize {
            let short = &full[..full.len() - missing];
            match decode_nifti(short, DecodeLimits::default()) {
                Err(SourceError::Nifti(NiftiError::TruncatedData {
                    offset,
                    found,
                    needed,
                })) => {
                    assert_eq!((offset, needed), (352, 6));
                    assert_eq!(found, (6 - missing) as u64);
                }
                other => panic!("{missing} bytes short should be refused: {other:?}"),
            }
        }
        // Positive control: the whole file is not refused.
        assert_eq!(decoded("dt2_uint8.nii").data().len(), 6);

        // And a `vox_offset` past the end of the file is the same refusal
        // reached the other way, which the oracle also fails.
        let mut past = fixture("dt2_uint8.nii");
        poke_f32(&mut past, 108, 1_000_000.0);
        assert!(matches!(
            decode_nifti(&past, DecodeLimits::default()).unwrap_err(),
            SourceError::Nifti(NiftiError::TruncatedData {
                offset: 1_000_000,
                found: 0,
                ..
            })
        ));
    }

    /**
     * The flattening throws the volume shape away, so the metadata has to
     * carry it or the information is gone. Every field this module reads is
     * attached under a `nifti-` name, the way `analyzeload` attaches
     * `dsr-<section>.<member>`.
     *
     * Input: `dim_rank4_2x3x2x2.nii` -> Output: a 2x12 raster whose
     * `nifti-dim[N]` fields still say `4 2 3 2 2`.
     */
    #[test]
    fn the_metadata_carries_the_axes_the_flattening_collapsed() {
        let raster = decoded("dim_rank4_2x3x2x2.nii");
        assert_eq!((raster.width(), raster.height()), (2, 12));
        let dim: Vec<i64> = (0..5)
            .map(|i| match raster.fields.get(&format!("nifti-dim[{i}]")) {
                Some(crate::imageio::MetadataValue::Int(v)) => *v,
                other => panic!("nifti-dim[{i}] should be an int, got {other:?}"),
            })
            .collect();
        assert_eq!(
            dim,
            vec![4, 2, 3, 2, 2],
            "the raster is 2x12 but the header said 2x3x2x2"
        );
        assert_eq!(
            raster.fields.get("nifti-version"),
            Some(&crate::imageio::MetadataValue::Int(1))
        );
        assert_eq!(
            raster.fields.get("nifti-byte-order"),
            Some(&crate::imageio::MetadataValue::Str("LSB_FIRST".into()))
        );
    }

    /**
     * A big-endian NIfTI-2 header is read in its own byte order, end to end.
     *
     * The proof is the datatype in the refusal: `endian_nifti2_int16_be.nii`
     * carries INT16, whose code is 4, and 4 read the wrong way round is
     * 1024, which is INT64 and a different name. So a header read
     * little-endian would refuse this file too, and refuse it as the wrong
     * type.
     * Input: `endian_nifti2_int16_be.nii` -> Output: `UnsupportedCarrier`
     * naming INT16, not INT64.
     */
    #[test]
    fn a_big_endian_nifti2_header_is_read_in_its_own_byte_order() {
        for (name, endian) in [
            ("endian_nifti2_int16_be.nii", Endian::Big),
            ("endian_nifti2_int16_le.nii", Endian::Little),
        ] {
            let bytes = fixture(name);
            let (version, sentinel) = header_version(&bytes).expect("version");
            assert_eq!(version, Version::Two);
            assert_eq!(field_endian(&bytes, version, sentinel), endian);
            match refused(name) {
                SourceError::Nifti(NiftiError::UnsupportedCarrier { datatype, name, .. }) => {
                    assert_eq!((datatype, name), (4, "INT16"));
                }
                other => panic!("{name} is INT16 in both byte orders: {other}"),
            }
        }
    }
}
