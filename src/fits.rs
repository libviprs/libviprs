//! FITS (`.fits` / `.fit` / `.fts`) load and save: 80-column ASCII header
//! cards in, a big-endian band-planar array out.
//!
//! Ported from libvips `foreign/fits.c`, together with the vertical flip
//! and the format-promotion table its two class files wrap it in
//! (`foreign/fitsload.c`, `foreign/fitssave.c`). vips reaches the format
//! through cfitsio; libviprs hand-rolls it, because the container is a
//! sequence of 2880-byte blocks of fixed-width ASCII cards followed by a
//! plain big-endian sample array, and vips's own side of it is thin.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_fits`] | `fitsload` | [`PixelFormat::Gray8`] / [`PixelFormat::Gray16`] / [`PixelFormat::FloatF32`] and their multi-band siblings |
//! | [`Raster::encode_fits`] | `fitssave` (to a buffer) | `.fits` bytes |
//! | [`Raster::save_fits`] | `fitssave` | `.fits` file |
//!
//! # Semantics
//!
//! * **The scan order is bottom-up.** `vips_foreign_load_fits_load`
//!   (`fitsload.c`) runs `vips__fits_read` and then
//!   `vips_flip(VIPS_DIRECTION_VERTICAL)`, and
//!   `vips_foreign_save_fits_build` (`fitssave.c`) flips before writing.
//!   So the first row in the file is the *bottom* row of the image, both
//!   ways, and this module folds the flip into the sample loop rather than
//!   materialising an intermediate.
//! * **Bands are planes, not interleaved samples.** `NAXIS3` is the band
//!   count and each band occupies a whole plane, which is why
//!   `vips_fits_generate` reads one band of one line at a time and
//!   scatters it (`fits.c:456-507`). libviprs de-planarises on load and
//!   re-planarises on save.
//! * **The axis rules come from `vips_fits_get_header`**
//!   (`fits.c:260-291`), a deliberate `switch` fallthrough: `NAXIS` 1, 2
//!   and 3 give width / width+height / width+height+bands, `NAXIS` 4
//!   through [`MAX_DIMENSIONS`] are accepted only when every axis above the
//!   third is exactly 1, and anything higher is refused. `NAXIS` 0 means
//!   "no data here", and vips walks forward to the next header unit
//!   (`fits.c:223-239`), which is how a file with a metadata-only primary
//!   unit loads at all.
//! * **Header cards become `fits-0`, `fits-1`, ...** in file order, with
//!   trailing blanks trimmed, matching `fits_read_record` as vips calls it
//!   at `fits.c:346-365`. The `END` card and the blank fill after it are
//!   not attached. When the loader walked past a `NAXIS = 0` unit, the
//!   cards are the *loaded* unit's, not the primary's; measured.
//! * **The carrier ceiling is the interesting constraint.** cfitsio hands
//!   vips an *equivalent* type rather than the raw one (`fits.c:246`), so
//!   what a BITPIX means depends on the `BSCALE` / `BZERO` pair beside it.
//!   Three combinations reach a carrier libviprs has, and every other one
//!   is refused by name rather than narrowed; see [`FitsError`].
//! * **Interpretation follows band count and carrier** (`fits.c:307-320`):
//!   one band is [`Interpretation::Bw`] or [`Interpretation::Grey16`],
//!   three bands [`Interpretation::Srgb`] or [`Interpretation::Rgb16`],
//!   anything else [`Interpretation::Multiband`].
//! * **Save is total over [`PixelFormat`].** vips promotes on the way out
//!   (`bandfmt_fits`, `fitssave.c`): char to uchar, short to ushort, int to
//!   uint. libviprs's carriers are already the unsigned ones, so every
//!   format this crate has maps onto a BITPIX and there is nothing for the
//!   encoder to refuse. Only the loader has a ceiling.
//! * **The saved header is cfitsio's, not vips's.** vips filters its own
//!   attached records through the `vips_fits_basic` prefix table
//!   (`fits.c:596-613`) precisely because cfitsio generates those cards
//!   itself, so a libviprs save has to spell them the way cfitsio 4.6.4
//!   does, down to the column. It does, and
//!   `encode_matches_the_reference_header_bytes` pins it.
//! * **`SIMPLE  =` is the magic.** vips's `is_a` hands the whole file to
//!   `fits_open_diskfile` (`fits.c:526-548`), which is not something a
//!   16-byte sniff can reproduce; the prefix is what the standard requires
//!   of the first card, and it is what the shared sniff in
//!   [`crate::source`] matches.
//!
//! # Why this is hand-rolled rather than a dependency
//!
//! Not because the ecosystem is empty. `fitsrs` is permissive
//! (Apache-2.0 OR MIT), pure Rust, and actively maintained by CDS
//! Strasbourg, and `fitsio` is the binding with real pipeline adoption.
//! The reason is scope: what libviprs needs from FITS is a header scan and
//! a de-planarising sample copy, both of which are in this file, and the
//! part that actually has to be right is the *vips-side* behaviour, which
//! no FITS crate models: the vertical flip, the `fits-N` record naming,
//! the cfitsio equivalent-type table, and the byte-exact generated header.
//! A dependency would supply the easy half and leave all of that here
//! anyway. `fitsio` is disqualified separately, under the dependency rule
//! in `CONTRIBUTING.md`: it pulls `fitsio-sys`, which needs a cfitsio
//! installed on the machine. The `links = "cfitsio"` key is the symptom
//! there rather than the reason; plenty of crates declare one and compile
//! nothing.
//!
//! Every number this module is pinned against was measured on the real
//! vips 8.18.4 binary (`cfitsio: true`, cfitsio 4.6.4) and is recorded,
//! with the commands that produced it, in `oracle-captures/foreign-fits/`.
//!
//! Every entry point here is fallible and there is no panicking twin,
//! matching the rest of the codec surface in [`crate::radiance`],
//! [`crate::webp`], and [`crate::gif`]: a decoder's failures come from
//! untrusted bytes, so a panicking spelling would have no honest caller.

use std::path::Path;

use thiserror::Error;

use crate::codec::EncodeError;
use crate::conversion::Interpretation;
use crate::imageio::{MetadataValue, SaveError};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use crate::source::{DecodeLimits, SourceError};

/// The prefix every FITS file's first card carries.
///
/// The standard fixes the primary header's first card as `SIMPLE` with a
/// logical value, so the keyword field, the `=` in column 9 and the space
/// after it are the only bytes that are the same in every file. vips does
/// not sniff at all here: `vips__fits_isfits` (`fits.c:526-548`) opens the
/// file with cfitsio and reports whether that worked, which a fixed-width
/// header sniff cannot reproduce.
pub(crate) const MAGIC: &[u8] = b"SIMPLE  =";

/// Bytes in one header or data block. Every FITS structure is a whole
/// number of these.
const BLOCK: usize = 2880;

/// Bytes in one header card.
const CARD: usize = 80;

/// Cards in one header block (`BLOCK / CARD`).
const CARDS_PER_BLOCK: usize = BLOCK / CARD;

/// Bytes in a card's keyword field, columns 1 to 8.
const KEYWORD_LEN: usize = 8;

/// Column the fixed-format value field starts at, just past the `= ` in
/// columns 9 and 10.
const VALUE_START: usize = 10;

/// Width of the fixed-format value field cfitsio writes, columns 11 to 30.
const VALUE_WIDTH: usize = 20;

/// The highest `NAXIS` vips accepts, from `MAX_DIMENSIONS` (`fits.c:99`).
/// Axes above the third must all have extent 1.
pub const MAX_DIMENSIONS: usize = 10;

/// The most header blocks this module will read for one header unit.
///
/// 256 blocks is 9216 cards, three orders of magnitude past what a real
/// instrument writes, and it exists so a file that never carries an `END`
/// card is refused after a bounded scan instead of after one the length of
/// the input. cfitsio has no such cap; it reads until end of file.
pub const MAX_HEADER_BLOCKS: usize = 256;

/// The most header units this module will walk while looking for one with
/// `NAXIS > 0`.
///
/// vips's loop (`fits.c:223-239`) is unbounded and stops only when cfitsio
/// runs out of file. This bounds the same walk, so a chain of empty units
/// cannot be used to make the loader do work proportional to the input
/// twice over.
pub const MAX_HEADER_UNITS: usize = 128;

/// The `BZERO` that makes a BITPIX 16 array unsigned.
///
/// This is the FITS standard's representation of unsigned 16-bit data and
/// what cfitsio's `fits_get_img_equivtype` turns into `USHORT_IMG`, which
/// is the row vips maps to `VIPS_FORMAT_USHORT` (`fits.c:199`). Measured:
/// `vips fitssave` on a ushort image writes exactly this pair, and
/// `vips fitsload` reads it back as ushort.
const UNSIGNED_16_BZERO: f64 = 32768.0;

/// The comment cfitsio writes on the `SIMPLE` card.
const SIMPLE_COMMENT: &str = "file does conform to FITS standard";

/// The comment cfitsio writes on the `BITPIX` card.
const BITPIX_COMMENT: &str = "number of bits per data pixel";

/// The comment cfitsio writes on the `NAXIS` card.
const NAXIS_COMMENT: &str = "number of data axes";

/// The comment cfitsio writes on the `EXTEND` card.
const EXTEND_COMMENT: &str = "FITS dataset may contain extensions";

/// The two `COMMENT` cards cfitsio writes into every primary header, byte
/// for byte as measured on cfitsio 4.6.4.
const PROVENANCE_COMMENTS: [&str; 2] = [
    "COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy",
    "COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H",
];

/// Record prefixes cfitsio generates itself, which vips therefore refuses
/// to write a second time (`vips_fits_basic`, `fits.c:596-613`).
const GENERATED_PREFIXES: [&str; 14] = [
    "SIMPLE ",
    "BITPIX ",
    "NAXIS ",
    "NAXIS1 ",
    "NAXIS2 ",
    "NAXIS3 ",
    "EXTEND ",
    "BZERO ",
    "BSCALE ",
    "COMMENT   FITS (Flexible Image Transport System) format",
    "COMMENT   and Astrophysics', volume 376, page 359; bibcode:",
    "XTENSION",
    "PCOUNT ",
    "GCOUNT ",
];

/// Keywords FITS lets repeat, which are therefore exempt from the
/// write-once dedupe (`vips_fits_duplicate`, `fits.c:617-622`).
const REPEATABLE_KEYWORDS: [&str; 4] = ["        ", "COMMENT ", "HISTORY ", "CONTINUE"];

/// Errors from the FITS codec.
///
/// Every variant except [`FitsError::Raster`] describes a specific
/// malformation in, or a specific limit of, untrusted bytes, which is what
/// makes them worth typing: the fuzz corpus in `fuzz/corpus/fuzz_fits/`
/// asserts on the variant, not on a message.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FitsError {
    /// The first card does not open with `SIMPLE  =`.
    ///
    /// vips has no equivalent check of its own: `vips__fits_isfits`
    /// (`fits.c:526-548`) hands the file to cfitsio and reports whether it
    /// opened, and cfitsio's own primary-header check is this one.
    #[error("fits: expected the first card to open with \"SIMPLE  =\", found {found:?}")]
    BadMagic {
        /// The leading bytes as read, lossily decoded and truncated for the
        /// message.
        found: String,
    },
    /// A header unit runs off the end of the file before its `END` card.
    #[error("fits: the file ends inside a header unit, before {expected}")]
    TruncatedHeader {
        /// What the parser was still waiting for.
        expected: &'static str,
    },
    /// A header unit reached [`MAX_HEADER_BLOCKS`] without an `END` card.
    #[error("fits: a header unit runs past the {cap}-block cap without an END card")]
    HeaderTooLong {
        /// The cap the unit exceeded, in 2880-byte blocks.
        cap: usize,
    },
    /// The file carries no header unit with `NAXIS > 0` within
    /// [`MAX_HEADER_UNITS`], so there is nothing in it to load.
    ///
    /// vips reports `no HDU found with naxes > 0` at `fits.c:235-236`
    /// after cfitsio runs out of file; this reports it after a bounded
    /// walk.
    #[error("fits: no header unit with NAXIS > 0 found in the first {searched} units")]
    NoImageUnit {
        /// How many header units were walked.
        searched: usize,
    },
    /// A card libviprs needs is absent, or its value field is not the kind
    /// of number the keyword requires.
    #[error("fits: header card {keyword} is missing or unparseable")]
    BadHeaderCard {
        /// The keyword whose card was missing or malformed.
        keyword: &'static str,
    },
    /// `NAXIS` is zero after the header walk, negative, or above
    /// [`MAX_DIMENSIONS`].
    ///
    /// vips refuses the same range at `fits.c:288-290` with
    /// `bad number of axis %d`.
    #[error("fits: bad number of axes {naxis}; FITS images carry 1 to {MAX_DIMENSIONS}")]
    BadAxisCount {
        /// The `NAXIS` the header declared.
        naxis: i64,
    },
    /// An axis above the third has an extent other than 1.
    ///
    /// vips accepts up to ten axes but only as long as the higher ones are
    /// empty (`fits.c:271-276`, `dimensions above 3 must be size 1`).
    #[error("fits: axis {axis} has extent {extent}; axes above 3 must be size 1")]
    HighDimensionNotEmpty {
        /// The 1-based axis number.
        axis: usize,
        /// The extent that axis declared.
        extent: u64,
    },
    /// The declared geometry is zero, negative, or wider than a [`Raster`]
    /// can address.
    #[error("fits: declared image size {width}x{height}x{bands} is out of bounds")]
    DimensionOutOfBounds {
        /// The declared `NAXIS1`.
        width: i64,
        /// The declared `NAXIS2`.
        height: i64,
        /// The declared `NAXIS3`.
        bands: i64,
    },
    /// The data segment is shorter than the declared geometry needs.
    #[error("fits: the data segment holds {found} bytes, {needed} are declared")]
    TruncatedData {
        /// Bytes actually left in the file after the header.
        found: usize,
        /// Bytes the declared geometry requires.
        needed: u64,
    },
    /// A BITPIX the format does not define, or one vips refuses too.
    ///
    /// FITS defines 8, 16, 32, 64, -32 and -64. vips's table
    /// (`fits.c:196-204`) has no `LONGLONG_IMG` row, so BITPIX 64 is
    /// refused there as well, with `unsupported bitpix 64`. `BZERO = -128`
    /// on a BITPIX 8 array is cfitsio's signed-byte convention and comes
    /// out of `fits_get_img_equivtype` as `SBYTE_IMG`, which vips also
    /// refuses, with `unsupported bitpix 10`; both measured.
    #[error("fits: unsupported BITPIX {bitpix}")]
    UnsupportedBitpix {
        /// The BITPIX the header declared, or the cfitsio equivalent type
        /// it resolves to when that is what vips reports.
        bitpix: i64,
    },
    /// A BITPIX vips loads but libviprs has no sample carrier for.
    ///
    /// This is a "not yet" rather than a "never", and each one is waiting
    /// on a specific carrier:
    ///
    /// * BITPIX 16 with default scaling is signed 16-bit, which needs
    ///   issue #516. The *unsigned* spelling (`BZERO = 32768`) is loaded,
    ///   because [`PixelFormat::Gray16`] already carries it.
    /// * BITPIX 32, signed or unsigned, needs issue #517.
    /// * BITPIX -64 is double, which is issue #518 and closed as not worth
    ///   building on its own.
    ///
    /// The sample-kind spine (issue #607) is what makes those carriers
    /// cheap, so the ceiling lifts there rather than never. Until then the
    /// loader refuses by name: narrowing a 16-bit array into 8 bits would
    /// lose data silently, which is worse than failing.
    #[error(
        "fits: BITPIX {bitpix} carries {sample} samples, which libviprs has no pixel \
         format for yet (issue #{issue})"
    )]
    UnsupportedCarrier {
        /// The BITPIX the header declared.
        bitpix: i64,
        /// The sample kind that BITPIX needs, in words.
        sample: &'static str,
        /// The issue that would add the carrier.
        issue: u32,
    },
    /// A `BSCALE` / `BZERO` pair that moves an integer array onto a
    /// different cfitsio equivalent type.
    ///
    /// cfitsio rescales integer data on read and reports the type the
    /// *scaled* values need, so the carrier a file loads to is not a
    /// function of its BITPIX alone. Measured: BITPIX 8 with
    /// `BSCALE = 2, BZERO = 10` loads as vips `short`, not as uchar. This
    /// module reads the two identities that keep the declared width, and
    /// refuses the rest rather than guessing at the range. Float arrays are
    /// unaffected: BITPIX -32 stays float under any scaling, and this
    /// module applies it.
    #[error(
        "fits: BITPIX {bitpix} with BSCALE {bscale} / BZERO {bzero} rescales onto \
         another sample type, which libviprs does not follow"
    )]
    UnsupportedScaling {
        /// The BITPIX the header declared.
        bitpix: i64,
        /// The `BSCALE` the header declared, or 1 when absent.
        bscale: f64,
        /// The `BZERO` the header declared, or 0 when absent.
        bzero: f64,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// The sample carrier a BITPIX and its scaling resolve to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Carrier {
    /// BITPIX 8 with default scaling: `TBYTE`, vips `uchar`.
    U8,
    /// BITPIX 16 with `BZERO = 32768`: `TUSHORT`, vips `ushort`.
    U16,
    /// BITPIX -32: `TFLOAT`, vips `float`.
    F32,
}

impl Carrier {
    /// Bytes one sample occupies, in the file and in the raster alike.
    const fn sample_bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
            Self::F32 => 4,
        }
    }

    /// The BITPIX this carrier is written as, and the `BZERO` that goes
    /// with it. `None` means no `BZERO` / `BSCALE` cards at all, which is
    /// what cfitsio writes for the unscaled types.
    const fn bitpix(self) -> (i64, Option<i64>) {
        match self {
            Self::U8 => (8, None),
            Self::U16 => (16, Some(32768)),
            Self::F32 => (-32, None),
        }
    }

    /// The carrier a raster's [`PixelFormat`] is written through.
    ///
    /// Total, because vips's promotion table (`bandfmt_fits`,
    /// `fitssave.c`) already sends every signed integer format to its
    /// unsigned twin and libviprs only has the unsigned ones.
    fn for_format(format: PixelFormat) -> Self {
        match format.bytes_per_channel() {
            1 => Self::U8,
            2 => Self::U16,
            _ => Self::F32,
        }
    }

    /// The [`Interpretation`] vips tags a load with, from band count and
    /// carrier (`fits.c:307-320`).
    fn interpretation(self, bands: usize) -> Interpretation {
        match (bands, self) {
            (1, Self::U16) => Interpretation::Grey16,
            (1, _) => Interpretation::Bw,
            (3, Self::U16) => Interpretation::Rgb16,
            (3, _) => Interpretation::Srgb,
            _ => Interpretation::Multiband,
        }
    }
}

/// Resolve a BITPIX and its scaling onto a carrier, the way cfitsio's
/// `fits_get_img_equivtype` does for the cases libviprs can follow
/// (`fits.c:246`, then the table at `fits.c:196-204`).
fn resolve_carrier(bitpix: i64, bscale: f64, bzero: f64) -> Result<Carrier, FitsError> {
    let unscaled = bscale == 1.0 && bzero == 0.0;
    match bitpix {
        // BITPIX -32 keeps its width under any scaling, and cfitsio applies
        // the scaling on read rather than changing the reported type.
        -32 => Ok(Carrier::F32),
        8 if unscaled => Ok(Carrier::U8),
        16 if bscale == 1.0 && bzero == UNSIGNED_16_BZERO => Ok(Carrier::U16),
        16 if unscaled => Err(FitsError::UnsupportedCarrier {
            bitpix,
            sample: "signed 16-bit integer",
            issue: 516,
        }),
        32 if unscaled || (bscale == 1.0 && bzero == 2_147_483_648.0) => {
            Err(FitsError::UnsupportedCarrier {
                bitpix,
                sample: "32-bit integer",
                issue: 517,
            })
        }
        -64 => Err(FitsError::UnsupportedCarrier {
            bitpix,
            sample: "64-bit float",
            issue: 518,
        }),
        8 | 16 | 32 => Err(FitsError::UnsupportedScaling {
            bitpix,
            bscale,
            bzero,
        }),
        other => Err(FitsError::UnsupportedBitpix { bitpix: other }),
    }
}

// ---------------------------------------------------------------------------
// Header cards
// ---------------------------------------------------------------------------

/// One header unit, parsed.
#[derive(Debug)]
struct HeaderUnit {
    /// Every card before `END`, trailing blanks trimmed, in file order.
    records: Vec<String>,
    /// Bytes the whole unit occupies, always a multiple of [`BLOCK`].
    len: usize,
    /// The declared `NAXIS`.
    naxis: i64,
    /// The declared `NAXIS1..NAXISn`, `naxis` of them.
    naxes: Vec<i64>,
    /// The declared BITPIX.
    bitpix: i64,
    /// `BSCALE`, or 1 when the card is absent.
    bscale: f64,
    /// `BZERO`, or 0 when the card is absent.
    bzero: f64,
}

/// A card's keyword: columns 1 to 8, trailing blanks trimmed.
fn card_keyword(card: &[u8]) -> &str {
    let field = &card[..KEYWORD_LEN.min(card.len())];
    std::str::from_utf8(field)
        .unwrap_or("")
        .trim_end_matches(' ')
}

/// A card's fixed-format value, comment stripped, or `None` when the card
/// carries no `= ` in columns 9 and 10.
///
/// The comment separator is the first `/` *outside* a quoted string, and
/// FITS escapes a quote inside one by doubling it, so the scan tracks quote
/// state rather than taking the first slash it sees.
fn card_value(card: &[u8]) -> Option<&str> {
    if card.len() < VALUE_START || &card[KEYWORD_LEN..VALUE_START] != b"= " {
        return None;
    }
    let rest = std::str::from_utf8(&card[VALUE_START..]).ok()?;
    let bytes = rest.as_bytes();
    let mut quoted = false;
    let mut end = rest.len();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' => quoted = !quoted,
            b'/' if !quoted => {
                end = i;
                break;
            }
            _ => {}
        }
        i += 1;
    }
    Some(rest[..end].trim())
}

/// Parse one header unit starting at `offset`.
fn parse_header_unit(bytes: &[u8], offset: usize) -> Result<HeaderUnit, FitsError> {
    let mut records: Vec<String> = Vec::new();
    let mut naxis: Option<i64> = None;
    let mut naxes_by_axis: Vec<Option<i64>> = Vec::new();
    let mut bitpix: Option<i64> = None;
    let mut bscale = 1.0f64;
    let mut bzero = 0.0f64;

    for block in 0..MAX_HEADER_BLOCKS {
        let start = offset + block * BLOCK;
        let end = start
            .checked_add(BLOCK)
            .filter(|&e| e <= bytes.len())
            .ok_or(FitsError::TruncatedHeader {
                expected: "the end of a 2880-byte header block",
            })?;
        let block_bytes = &bytes[start..end];
        for slot in 0..CARDS_PER_BLOCK {
            let card = &block_bytes[slot * CARD..(slot + 1) * CARD];
            let keyword = card_keyword(card);
            if keyword == "END" {
                let unit = HeaderUnit {
                    records,
                    len: (block + 1) * BLOCK,
                    naxis: naxis.ok_or(FitsError::BadHeaderCard { keyword: "NAXIS" })?,
                    naxes: Vec::new(),
                    bitpix: bitpix.ok_or(FitsError::BadHeaderCard { keyword: "BITPIX" })?,
                    bscale,
                    bzero,
                };
                return finish_unit(unit, &naxes_by_axis);
            }
            // The record is what vips attaches, and cfitsio hands it over
            // with trailing blanks already gone (`fits.c:346-365`).
            records.push(
                String::from_utf8_lossy(card)
                    .trim_end_matches(' ')
                    .to_string(),
            );
            let Some(value) = card_value(card) else {
                continue;
            };
            match keyword {
                "BITPIX" => bitpix = value.parse::<i64>().ok(),
                "NAXIS" => naxis = value.parse::<i64>().ok(),
                "BSCALE" => bscale = parse_real(value).unwrap_or(bscale),
                "BZERO" => bzero = parse_real(value).unwrap_or(bzero),
                // `NAXIS1` upwards. The ceiling keeps a card like
                // `NAXIS999999999` from sizing a vector.
                _ => {
                    if let Some(axis) = keyword.strip_prefix("NAXIS")
                        && let Ok(axis) = axis.parse::<usize>()
                        && (1..=MAX_DIMENSIONS).contains(&axis)
                    {
                        if naxes_by_axis.len() < axis {
                            naxes_by_axis.resize(axis, None);
                        }
                        naxes_by_axis[axis - 1] = value.parse::<i64>().ok();
                    }
                }
            }
        }
    }
    Err(FitsError::HeaderTooLong {
        cap: MAX_HEADER_BLOCKS,
    })
}

/// Pull `NAXIS1..NAXISn` out of the sparse per-axis table once `NAXIS`
/// itself is known, refusing a unit that declares an axis count it does not
/// then supply extents for.
fn finish_unit(
    mut unit: HeaderUnit,
    naxes_by_axis: &[Option<i64>],
) -> Result<HeaderUnit, FitsError> {
    if unit.naxis < 0 || unit.naxis > MAX_DIMENSIONS as i64 {
        return Err(FitsError::BadAxisCount { naxis: unit.naxis });
    }
    for axis in 0..unit.naxis as usize {
        let extent = naxes_by_axis
            .get(axis)
            .copied()
            .flatten()
            .ok_or(FitsError::BadHeaderCard { keyword: "NAXISn" })?;
        unit.naxes.push(extent);
    }
    Ok(unit)
}

/// Parse a fixed-format real, which FITS spells with either an `E` or a `D`
/// exponent (`1.0D+02`); Rust's parser only knows `E`.
fn parse_real(value: &str) -> Option<f64> {
    value
        .parse::<f64>()
        .ok()
        .or_else(|| value.replace(['D', 'd'], "E").parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

/// Format a fixed-format card: keyword in columns 1 to 8, `= ` in 9 and 10,
/// the value right-justified in 11 to 30, then ` / ` and the comment. This
/// is cfitsio's own layout, verified byte for byte against a saved file.
fn fixed_card(keyword: &str, value: &str, comment: &str) -> Vec<u8> {
    pad_card(&format!(
        "{keyword:<KEYWORD_LEN$}= {value:>VALUE_WIDTH$} / {comment}"
    ))
}

/// Render a record as one full 80-column card.
///
/// A card is 80 *bytes* and the standard restricts it to printable ASCII,
/// so anything outside that range is written as a space rather than
/// widening the card and desynchronising every card after it. Such a byte
/// can only reach here from a malformed file, through the lossy decode the
/// loader applies to a record.
fn pad_card(text: &str) -> Vec<u8> {
    let mut card = Vec::with_capacity(CARD);
    for &byte in text.as_bytes() {
        if card.len() == CARD {
            break;
        }
        card.push(if (0x20..=0x7e).contains(&byte) {
            byte
        } else {
            b' '
        });
    }
    card.resize(CARD, b' ');
    card
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

/// Decode FITS bytes into a [`Raster`] (libvips `fitsload`).
///
/// The image comes back right way up and band-interleaved: FITS stores its
/// rows bottom first and its bands as separate planes, and vips undoes both
/// (`vips_flip` in `fitsload.c`, `vips_fits_scatter` in `fits.c:425-454`),
/// so this does too. Every header card is attached as `fits-0`, `fits-1`,
/// and so on in file order.
///
/// The carrier is [`PixelFormat::Gray8`] and friends for BITPIX 8,
/// [`PixelFormat::Gray16`] and friends for the standard unsigned spelling
/// of BITPIX 16, and [`PixelFormat::FloatF32`] / [`PixelFormat::RgbaF32`]
/// for BITPIX -32. Any other BITPIX is refused by name; see
/// [`FitsError::UnsupportedCarrier`] for what each one is waiting on.
///
/// A header unit with `NAXIS = 0` carries no image, so the loader walks
/// forward to the next one, which is what makes a file with a
/// metadata-only primary unit readable at all (`fits.c:223-239`).
///
/// # Errors
///
/// * [`SourceError::Fits`] wrapping [`FitsError::BadMagic`] for bytes that
///   do not open with `SIMPLE  =`, [`FitsError::TruncatedHeader`],
///   [`FitsError::HeaderTooLong`], [`FitsError::NoImageUnit`],
///   [`FitsError::BadHeaderCard`], [`FitsError::BadAxisCount`],
///   [`FitsError::HighDimensionNotEmpty`],
///   [`FitsError::DimensionOutOfBounds`] or [`FitsError::TruncatedData`]
///   for a malformed header or a short data segment,
///   [`FitsError::UnsupportedBitpix`], [`FitsError::UnsupportedCarrier`] or
///   [`FitsError::UnsupportedScaling`] for a sample type this build has no
///   carrier for, and [`SourceError::AllocLimitExceeded`] when the declared
///   geometry is over [`DecodeLimits::max_alloc_bytes`].
/// * [`SourceError::CoordLimitExceeded`] when either axis exceeds
///   [`DecodeLimits::max_coord`].
/// * [`SourceError::DimensionLimitExceeded`] when `width * height` exceeds
///   [`DecodeLimits::max_pixels`].
pub fn decode_fits(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    if !bytes.starts_with(MAGIC) {
        let head = &bytes[..bytes.len().min(MAGIC.len())];
        return Err(FitsError::BadMagic {
            found: String::from_utf8_lossy(head).into_owned(),
        }
        .into());
    }

    // Walk forward over header units that declare no data, exactly as
    // `vips_fits_get_header` does (`fits.c:223-239`). A unit with
    // `NAXIS = 0` has no data blocks at all, so the next unit starts
    // immediately after this one's header.
    let mut offset = 0usize;
    let mut unit = parse_header_unit(bytes, offset)?;
    let mut searched = 1usize;
    while unit.naxis == 0 {
        if searched >= MAX_HEADER_UNITS {
            return Err(FitsError::NoImageUnit { searched }.into());
        }
        offset += unit.len;
        unit = parse_header_unit(bytes, offset)?;
        searched += 1;
    }

    let naxis =
        usize::try_from(unit.naxis).map_err(|_| FitsError::BadAxisCount { naxis: unit.naxis })?;
    if naxis == 0 || naxis > MAX_DIMENSIONS {
        return Err(FitsError::BadAxisCount { naxis: unit.naxis }.into());
    }
    // vips accepts up to ten axes as long as everything above the third is
    // empty (`fits.c:271-276`).
    for axis in 3..naxis {
        let extent = unit.naxes[axis];
        if extent != 1 {
            return Err(FitsError::HighDimensionNotEmpty {
                axis: axis + 1,
                extent: extent.unsigned_abs(),
            }
            .into());
        }
    }

    // The `switch` fallthrough at `fits.c:260-291`: one axis is a single
    // row, two axes are a mono image, three axes name the band count.
    let declared_width = unit.naxes[0];
    let declared_height = if naxis >= 2 { unit.naxes[1] } else { 1 };
    let declared_bands = if naxis >= 3 { unit.naxes[2] } else { 1 };
    let out_of_bounds = || FitsError::DimensionOutOfBounds {
        width: declared_width,
        height: declared_height,
        bands: declared_bands,
    };
    let width = u32::try_from(declared_width)
        .ok()
        .filter(|&w| w > 0)
        .ok_or_else(out_of_bounds)?;
    let height = u32::try_from(declared_height)
        .ok()
        .filter(|&h| h > 0)
        .ok_or_else(out_of_bounds)?;
    let bands = u16::try_from(declared_bands)
        .ok()
        .filter(|&b| b > 0)
        .ok_or_else(out_of_bounds)?;

    let carrier = resolve_carrier(unit.bitpix, unit.bscale, unit.bzero)?;

    // Both ceilings go on the declared header geometry, before anything is
    // reserved, the way `crate::gif` and `crate::webp` do it.
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    // One spelling of the budget for the whole crate (issue #632): the
    // price comes from `decode_alloc_bytes` and the comparison from
    // `DecodeLimits::exceeds_alloc_budget`, so neither can drift here on
    // its own.
    // The saturation the price carries is what this module needs: a caller
    // may lift `max_coord` and `max_pixels` and declare a geometry whose
    // byte count does not fit a `u64`. That saturated `u64::MAX` is
    // refused by `exceeds_alloc_budget`'s own sentinel arm, which is where
    // it belongs; before that arm existed it only failed the truncation
    // check below, so this module happened to survive a hole the four
    // codecs beside it did not.
    //
    // The price, the comparison and now the reporting are all the crate's:
    // this used to build a `FitsError::AllocLimitExceeded` of its own, one of
    // five variants re-tagging the same refusal, which #686 collapsed onto
    // `SourceError::AllocLimitExceeded`.
    // The price comes back because the payload slice below is sized from the
    // same number.
    let needed = limits.check_image_alloc(
        "FITS pixel buffer",
        width,
        height,
        u64::from(bands),
        carrier.sample_bytes() as u64,
    )?;

    let data_start = offset + unit.len;
    let available = bytes.len().saturating_sub(data_start);
    if (available as u64) < needed {
        return Err(FitsError::TruncatedData {
            found: available,
            needed,
        }
        .into());
    }
    let payload = &bytes[data_start..data_start + needed as usize];

    let format = PixelFormat::with_channels(usize::from(bands), carrier.sample_bytes())
        .ok_or_else(out_of_bounds)?;
    let out = deplanarise(
        payload,
        width,
        height,
        bands,
        carrier,
        unit.bscale,
        unit.bzero,
    );

    let mut raster = Raster::new_with_budget(width, height, format, out, limits.max_alloc_bytes)
        .map_err(FitsError::Raster)?;
    raster.meta.interpretation = Some(carrier.interpretation(usize::from(bands)));
    for (index, record) in unit.records.iter().enumerate() {
        raster
            .fields
            .set(&format!("fits-{index}"), record.clone().into());
    }
    Ok(raster)
}

/// Turn the file's band-planar, bottom-up sample array into libviprs's
/// interleaved, top-down buffer.
///
/// One pass, no intermediate: the destination row is `height - 1 - row`
/// (the `vips_flip` in `fitsload.c`) and the destination sample index steps
/// by the band count (the `SCATTER` macro at `fits.c:414-423`).
fn deplanarise(
    payload: &[u8],
    width: u32,
    height: u32,
    bands: u16,
    carrier: Carrier,
    bscale: f64,
    bzero: f64,
) -> Vec<u8> {
    let (w, h, b) = (width as usize, height as usize, usize::from(bands));
    let sample = carrier.sample_bytes();
    let mut out = vec![0u8; w * h * b * sample];
    // cfitsio only rescales when it has to; an identity pair is a plain
    // copy (`ffr4fr4` and friends short-circuit on `scale == 1 && zero == 0`).
    let scaled = carrier == Carrier::F32 && !(bscale == 1.0 && bzero == 0.0);
    for band in 0..b {
        for row in 0..h {
            let src_row = (band * h + row) * w * sample;
            let dst_row = (h - 1 - row) * w * b * sample;
            for x in 0..w {
                let src = src_row + x * sample;
                let dst = dst_row + (x * b + band) * sample;
                match carrier {
                    Carrier::U8 => out[dst] = payload[src],
                    Carrier::U16 => {
                        // The stored value is signed; `BZERO = 32768`
                        // shifts it into the unsigned range, which is what
                        // makes this the standard's unsigned-16 spelling.
                        let raw = i16::from_be_bytes([payload[src], payload[src + 1]]);
                        let value = (i32::from(raw) + 32768) as u16;
                        out[dst..dst + 2].copy_from_slice(&value.to_ne_bytes());
                    }
                    Carrier::F32 => {
                        let mut raw = [0u8; 4];
                        raw.copy_from_slice(&payload[src..src + 4]);
                        let mut value = f32::from_be_bytes(raw);
                        if scaled {
                            // cfitsio computes `input * scale + zero` in
                            // double and narrows once.
                            value = (f64::from(value) * bscale + bzero) as f32;
                        }
                        out[dst..dst + 4].copy_from_slice(&value.to_ne_bytes());
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

impl Raster {
    /// Encode as FITS bytes (libvips `fitssave` into a buffer).
    ///
    /// Total over [`PixelFormat`]: the 8-bit formats go out as BITPIX 8,
    /// the 16-bit ones as BITPIX 16 with the standard `BZERO = 32768`, and
    /// the float ones as BITPIX -32. That is vips's own promotion table
    /// (`bandfmt_fits`, `fitssave.c`) reaching the same place from a wider
    /// starting set, since libviprs has only the unsigned carriers.
    ///
    /// The generated header is cfitsio's, card for card, so a file this
    /// writes and a file `vips fitssave` writes from the same pixels are
    /// byte-identical. Any attached `fits-` records are appended after it,
    /// filtered the way `vips_fits_write_record` filters them
    /// (`fits.c:635-681`): cfitsio's own cards are not written twice, and a
    /// keyword is written once unless it is one FITS lets repeat.
    ///
    /// # Errors
    ///
    /// Currently infallible for every representable [`PixelFormat`]: a
    /// raster's width and height are `u32` and its band count `u16`, so
    /// every axis has a spelling inside the fixed-format value field's
    /// twenty columns of decimal. The `Result` reserves room for a future
    /// carrier that does not, and keeps this method the same shape as the
    /// rest of the encode surface.
    pub fn encode_fits(&self) -> Result<Vec<u8>, EncodeError> {
        let format = self.format();
        let carrier = Carrier::for_format(format);
        let (width, height) = (self.width(), self.height());
        let bands = format.channels();
        let mut out = self.fits_header(carrier, width, height, bands);
        self.write_planes(&mut out, carrier);
        pad_to_block(&mut out, 0);
        Ok(out)
    }

    /// Save the raster to `path` as FITS (libvips `fitssave`).
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] when [`Raster::encode_fits`] rejects the
    /// raster, or [`SaveError::Io`] when the file write fails.
    pub fn save_fits(&self, path: &Path) -> Result<(), SaveError> {
        let bytes = self.encode_fits().map_err(|e| match e {
            EncodeError::Io(io) => SaveError::Io(io),
            other => SaveError::Encode(crate::sink::SinkError::EncodeMsg(other.to_string())),
        })?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Build the header unit: cfitsio's generated cards, then whatever
    /// `fits-` records survive the filter, then `END` and the blank fill.
    fn fits_header(&self, carrier: Carrier, width: u32, height: u32, bands: usize) -> Vec<u8> {
        let (bitpix, bzero) = carrier.bitpix();
        // Mono is `NAXIS = 2`, everything else `NAXIS = 3`
        // (`vips_fits_set_header`, `fits.c:716`).
        let naxis = if bands == 1 { 2 } else { 3 };
        let mut cards = vec![
            fixed_card("SIMPLE", "T", SIMPLE_COMMENT),
            fixed_card("BITPIX", &bitpix.to_string(), BITPIX_COMMENT),
            fixed_card("NAXIS", &naxis.to_string(), NAXIS_COMMENT),
            fixed_card("NAXIS1", &width.to_string(), "length of data axis 1"),
            fixed_card("NAXIS2", &height.to_string(), "length of data axis 2"),
        ];
        if naxis == 3 {
            cards.push(fixed_card(
                "NAXIS3",
                &bands.to_string(),
                "length of data axis 3",
            ));
        }
        cards.push(fixed_card("EXTEND", "T", EXTEND_COMMENT));
        for comment in PROVENANCE_COMMENTS {
            cards.push(pad_card(comment));
        }
        if let Some(bzero) = bzero {
            cards.push(fixed_card(
                "BZERO",
                &bzero.to_string(),
                "offset data range to that of unsigned short",
            ));
            cards.push(fixed_card("BSCALE", "1", "default scaling factor"));
        }
        for record in self.fits_records() {
            cards.push(pad_card(&record));
        }
        cards.push(pad_card("END"));

        let mut bytes = Vec::with_capacity(cards.len() * CARD + BLOCK);
        for card in &cards {
            bytes.extend_from_slice(card);
        }
        pad_to_block(&mut bytes, b' ');
        bytes
    }

    /// The attached `fits-` records that survive vips's two filters, in
    /// insertion order.
    ///
    /// `vips_fits_write_record` (`fits.c:635-681`) drops anything cfitsio
    /// generates itself and then writes each keyword at most once, with
    /// blank, `COMMENT`, `HISTORY` and `CONTINUE` exempt from the dedupe.
    fn fits_records(&self) -> Vec<String> {
        let names: Vec<String> = self
            .fields
            .names()
            .filter(|n| n.starts_with("fits-"))
            .map(str::to_string)
            .collect();
        let mut written: Vec<String> = Vec::new();
        let mut out: Vec<String> = Vec::new();
        for name in names {
            let Some(MetadataValue::Str(record)) = self.fields.get(&name) else {
                continue;
            };
            if GENERATED_PREFIXES
                .iter()
                .any(|prefix| record.starts_with(prefix))
            {
                continue;
            }
            // vips dedupes on the 8-column keyword field
            // (`g_strlcpy(keyword, line, 9)`), not on the whole card.
            let keyword: String = record.chars().take(KEYWORD_LEN).collect();
            if written.contains(&keyword) {
                continue;
            }
            out.push(record.clone());
            if !record.is_empty()
                && !REPEATABLE_KEYWORDS
                    .iter()
                    .any(|dupe| keyword.starts_with(dupe))
            {
                written.push(keyword);
            }
        }
        out
    }

    /// Append the sample array: one plane per band, bottom row first,
    /// big-endian.
    fn write_planes(&self, out: &mut Vec<u8>, carrier: Carrier) {
        let format = self.format();
        let bands = format.channels();
        let sample = carrier.sample_bytes();
        let (w, h) = (self.width() as usize, self.height() as usize);
        let stride = self.stride();
        let data = self.data();
        out.reserve(w * h * bands * sample);
        for band in 0..bands {
            for row in (0..h).rev() {
                let base = row * stride;
                for x in 0..w {
                    let src = base + (x * bands + band) * sample;
                    match carrier {
                        Carrier::U8 => out.push(data[src]),
                        Carrier::U16 => {
                            let value = u16::from_ne_bytes([data[src], data[src + 1]]);
                            // Undo the `BZERO` shift the header declares.
                            let stored = (i32::from(value) - 32768) as i16;
                            out.extend_from_slice(&stored.to_be_bytes());
                        }
                        Carrier::F32 => {
                            let mut raw = [0u8; 4];
                            raw.copy_from_slice(&data[src..src + 4]);
                            out.extend_from_slice(&f32::from_ne_bytes(raw).to_be_bytes());
                        }
                    }
                }
            }
        }
    }
}

/// Pad a buffer out to the next 2880-byte boundary with `fill`.
///
/// FITS is a whole number of blocks end to end; a header pads with spaces
/// and a data segment with zeros.
fn pad_to_block(bytes: &mut Vec<u8>, fill: u8) {
    let remainder = bytes.len() % BLOCK;
    if remainder != 0 {
        bytes.resize(bytes.len() + (BLOCK - remainder), fill);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::DeclaredGeometry;

    /// Build a header unit from a list of card texts, padded out to whole
    /// blocks the way a real file is.
    fn header(cards: &[&str]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for card in cards {
            bytes.extend_from_slice(&pad_card(card));
        }
        bytes.extend_from_slice(&pad_card("END"));
        pad_to_block(&mut bytes, b' ');
        bytes
    }

    /// A 4x3 BITPIX 8 mono file whose pixels are `row * 4 + x`, written
    /// bottom row first as FITS requires.
    fn mono_u8() -> Vec<u8> {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    3",
        ]);
        for row in (0..3u8).rev() {
            for x in 0..4u8 {
                file.push(row * 4 + x);
            }
        }
        pad_to_block(&mut file, 0);
        file
    }

    /**
     * Tests that a BITPIX 8 mono file loads right way up.
     * Works by decoding a file whose pixel block is written bottom row
     * first and asserting the raster reads top row first, which is the
     * `vips_flip(VIPS_DIRECTION_VERTICAL)` in `fitsload.c`.
     * Input: the 4x3 ramp above -> Output: rows 0, 1, 2 in image order.
     */
    #[test]
    fn load_flips_the_scan_order() {
        let raster = decode_fits(&mono_u8(), DecodeLimits::default()).unwrap();
        assert_eq!(raster.width(), 4);
        assert_eq!(raster.height(), 3);
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(
            raster.data(),
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            "the first row in the file is the bottom row of the image"
        );
    }

    /**
     * Tests that a three-axis file de-planarises into interleaved bands.
     * Works by giving each band a distinct constant, so an interleaved
     * result reads `0,1,2` repeating and a planar one would not. This is
     * `vips_fits_scatter` (`fits.c:425-454`).
     * Input: 2x2x3 with band b filled with b -> Output: 0,1,2 per pixel.
     */
    #[test]
    fn load_deplanarises_bands() {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    3",
            "NAXIS1  =                    2",
            "NAXIS2  =                    2",
            "NAXIS3  =                    3",
        ]);
        for band in 0..3u8 {
            for _ in 0..4 {
                file.push(band);
            }
        }
        pad_to_block(&mut file, 0);
        let raster = decode_fits(&file, DecodeLimits::default()).unwrap();
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(raster.data(), &[0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]);
        assert_eq!(raster.interpretation(), Interpretation::Srgb);
    }

    /**
     * Tests that the standard unsigned-16 spelling loads as `Gray16`.
     * Works by storing values shifted down by 32768 as signed shorts, the
     * way `vips fitssave` writes a ushort image, and asserting the decoded
     * samples come back at their unsigned value.
     * Input: BITPIX 16 with BZERO 32768 -> Output: `Gray16` samples
     * 0, 4097, 8194, 12291, measured off `vips getpoint`.
     */
    #[test]
    fn load_reads_the_unsigned_16_convention() {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    1",
            "BZERO   =                32768",
            "BSCALE  =                    1",
        ]);
        for i in 0..4i32 {
            let stored = (i * 4097 - 32768) as i16;
            file.extend_from_slice(&stored.to_be_bytes());
        }
        pad_to_block(&mut file, 0);
        let raster = decode_fits(&file, DecodeLimits::default()).unwrap();
        assert_eq!(raster.format(), PixelFormat::Gray16);
        assert_eq!(raster.interpretation(), Interpretation::Grey16);
        let samples: Vec<u16> = raster
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_ne_bytes(*c))
            .collect();
        assert_eq!(samples, vec![0, 4097, 8194, 12291]);
    }

    /**
     * Tests that BITPIX -32 honours BSCALE and BZERO.
     * Works by decoding a float array with `BSCALE 2 / BZERO 10` and
     * checking the samples land on `raw * 2 + 10`, which is what cfitsio
     * applies on read and what `vips getpoint` reports for the same file.
     * Input: raw -3, -2, -1, 0 -> Output: 4, 6, 8, 10.
     */
    #[test]
    fn load_applies_float_scaling() {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    1",
            "BSCALE  =                    2",
            "BZERO   =                   10",
        ]);
        for raw in [-3.0f32, -2.0, -1.0, 0.0] {
            file.extend_from_slice(&raw.to_be_bytes());
        }
        pad_to_block(&mut file, 0);
        let raster = decode_fits(&file, DecodeLimits::default()).unwrap();
        assert_eq!(raster.format().channels(), 1);
        assert!(raster.format().is_float());
        let samples: Vec<f32> = raster
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_ne_bytes(*c))
            .collect();
        for (got, want) in samples.iter().zip([4.0f32, 6.0, 8.0, 10.0]) {
            assert!(
                (got - want).abs() < 1e-6,
                "scaled sample {got} should be {want} within 1e-6"
            );
        }
    }

    /**
     * Tests that a header unit with no data is walked past.
     * Works by putting a `NAXIS = 0` primary unit in front of an image
     * extension, which is the common layout vips handles at
     * `fits.c:223-239`, and checking both the pixels and the attached
     * records come from the extension.
     * Input: empty primary + IMAGE extension -> Output: the extension's
     * 4x1 pixels and its `XTENSION` card as `fits-0`.
     */
    #[test]
    fn load_walks_past_an_empty_header_unit() {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
            "EXTEND  =                    T",
        ]);
        file.extend_from_slice(&header(&[
            "XTENSION= 'IMAGE   '",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    1",
        ]));
        file.extend_from_slice(&[9, 8, 7, 6]);
        pad_to_block(&mut file, 0);
        let raster = decode_fits(&file, DecodeLimits::default()).unwrap();
        assert_eq!(raster.data(), &[9, 8, 7, 6]);
        assert_eq!(
            raster.get_field("fits-0").unwrap().as_str(),
            "XTENSION= 'IMAGE   '",
            "the records come from the unit that was loaded, not the primary"
        );
    }

    /**
     * Tests that every BITPIX vips loads but libviprs cannot carry is
     * refused by name rather than narrowed.
     * Works by sweeping the four measured cases and matching on the typed
     * variant plus the issue number its doc points at.
     * Input: BITPIX 16 unscaled, 32, 32 unsigned, -64 -> Output:
     * `UnsupportedCarrier` citing #516, #517, #517, #518.
     */
    #[test]
    fn unreachable_carriers_name_the_issue_that_unblocks_them() {
        let cases: [(i64, f64, f64, u32); 4] = [
            (16, 1.0, 0.0, 516),
            (32, 1.0, 0.0, 517),
            (32, 1.0, 2_147_483_648.0, 517),
            (-64, 1.0, 0.0, 518),
        ];
        for (bitpix, bscale, bzero, issue) in cases {
            match resolve_carrier(bitpix, bscale, bzero) {
                Err(FitsError::UnsupportedCarrier {
                    bitpix: got,
                    issue: cited,
                    ..
                }) => {
                    assert_eq!(got, bitpix);
                    assert_eq!(cited, issue, "BITPIX {bitpix} should cite issue #{issue}");
                }
                other => panic!("BITPIX {bitpix} should be refused by carrier, got {other:?}"),
            }
        }
    }

    /**
     * Tests that the BITPIX values vips itself refuses are refused here.
     * Works by resolving BITPIX 64, which has no row in the vips table
     * (`fits.c:196-204`) and reports `unsupported bitpix 64`, and a value
     * the format does not define at all.
     * Input: 64 and 7 -> Output: `UnsupportedBitpix`.
     */
    #[test]
    fn undefined_bitpix_values_are_refused() {
        for bitpix in [64i64, 7] {
            assert!(matches!(
                resolve_carrier(bitpix, 1.0, 0.0),
                Err(FitsError::UnsupportedBitpix { .. })
            ));
        }
    }

    /**
     * Tests that a scaling which moves an integer array onto another
     * cfitsio equivalent type is refused rather than guessed at.
     * Works by resolving the measured case `BITPIX 8, BSCALE 2,
     * BZERO 10`, which `vips fitsload` reports as `short`, and the
     * signed-byte convention `BZERO -128`, which vips refuses outright as
     * `unsupported bitpix 10`.
     * Input: both pairs -> Output: `UnsupportedScaling`.
     */
    #[test]
    fn rescaled_integer_arrays_are_refused() {
        for (bitpix, bscale, bzero) in [(8i64, 2.0, 10.0), (8, 1.0, -128.0), (16, 1.0, 1.0)] {
            assert!(
                matches!(
                    resolve_carrier(bitpix, bscale, bzero),
                    Err(FitsError::UnsupportedScaling { .. })
                ),
                "BITPIX {bitpix} with BSCALE {bscale} / BZERO {bzero} should be refused"
            );
        }
    }

    /**
     * Tests the axis rules vips applies at `fits.c:260-291`.
     * Works by sweeping a `NAXIS` above the ceiling, an axis above the
     * third with a non-unit extent, and the accepted case where those
     * higher axes are all 1.
     * Input: NAXIS 11, NAXIS 4 with a fourth axis of 2, NAXIS 4 with a
     * fourth axis of 1 -> Output: `BadAxisCount`,
     * `HighDimensionNotEmpty`, and a clean 4x3 load.
     */
    #[test]
    fn axis_rules_match_the_reference() {
        let build = |naxis: usize, extents: &[u32]| {
            let mut cards = vec![
                "SIMPLE  =                    T".to_string(),
                "BITPIX  =                    8".to_string(),
                format!("NAXIS   = {naxis:>20}"),
            ];
            for (i, extent) in extents.iter().enumerate() {
                cards.push(format!("NAXIS{:<3}= {extent:>20}", i + 1));
            }
            let refs: Vec<&str> = cards.iter().map(String::as_str).collect();
            let mut file = header(&refs);
            file.extend_from_slice(&[0u8; 4 * 3]);
            pad_to_block(&mut file, 0);
            file
        };

        assert!(matches!(
            decode_fits(
                &build(11, &[4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                DecodeLimits::default()
            ),
            Err(SourceError::Fits(FitsError::BadAxisCount { naxis: 11 }))
        ));
        assert!(matches!(
            decode_fits(&build(4, &[4, 3, 1, 2]), DecodeLimits::default()),
            Err(SourceError::Fits(FitsError::HighDimensionNotEmpty {
                axis: 4,
                extent: 2
            }))
        ));
        let ok = decode_fits(&build(4, &[4, 3, 1, 1]), DecodeLimits::default()).unwrap();
        assert_eq!((ok.width(), ok.height()), (4, 3));
    }

    /**
     * Tests that the parser is bounded rather than input-shaped.
     * Works by driving the three caps this module owns: a header with no
     * `END` card, a header unit that runs off the end of the file, and a
     * chain of empty units longer than the walk allows.
     * Input: each malformation -> Output: `HeaderTooLong`,
     * `TruncatedHeader`, `NoImageUnit`.
     */
    #[test]
    fn the_parser_bounds_are_enforced() {
        let mut endless = Vec::new();
        for _ in 0..MAX_HEADER_BLOCKS + 1 {
            endless.extend_from_slice(&pad_card("SIMPLE  =                    T"));
            endless.resize(endless.len() + BLOCK - CARD, b' ');
        }
        assert!(matches!(
            decode_fits(&endless, DecodeLimits::default()),
            Err(SourceError::Fits(FitsError::HeaderTooLong { .. }))
        ));

        let short = pad_card("SIMPLE  =                    T");
        assert!(matches!(
            decode_fits(&short, DecodeLimits::default()),
            Err(SourceError::Fits(FitsError::TruncatedHeader { .. }))
        ));

        let empty_unit = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    0",
        ]);
        let mut chain = Vec::new();
        for _ in 0..MAX_HEADER_UNITS + 1 {
            chain.extend_from_slice(&empty_unit);
        }
        assert!(matches!(
            decode_fits(&chain, DecodeLimits::default()),
            Err(SourceError::Fits(FitsError::NoImageUnit { .. }))
        ));
    }

    /**
     * Tests that the declared geometry is bounded before allocation.
     * Works by claiming a very large image in a file that holds almost no
     * bytes, once against each of the three `DecodeLimits` ceilings, and
     * checking each returns its own typed variant.
     * Input: a 40-byte header claiming a huge raster -> Output:
     * `CoordLimitExceeded`, `DimensionLimitExceeded`,
     * `AllocLimitExceeded`.
     */
    #[test]
    fn declared_geometry_is_bounded_before_allocation() {
        let bomb = |w: u32, h: u32| {
            header(&[
                "SIMPLE  =                    T",
                "BITPIX  =                    8",
                "NAXIS   =                    2",
                &format!("NAXIS1  = {w:>20}"),
                &format!("NAXIS2  = {h:>20}"),
            ])
        };
        assert!(matches!(
            decode_fits(
                &bomb(100_000, 4),
                DecodeLimits::default().with_max_coord(1024)
            ),
            Err(SourceError::CoordLimitExceeded { .. })
        ));
        assert!(matches!(
            decode_fits(
                &bomb(60_000, 60_000),
                DecodeLimits::default().with_max_pixels(1_000)
            ),
            Err(SourceError::DimensionLimitExceeded { .. })
        ));
        // Under the default pixel ceiling, so the allocation budget is
        // what has to catch this one.
        assert!(matches!(
            decode_fits(
                &bomb(30_000, 30_000),
                DecodeLimits::default().with_max_alloc_bytes(1_000)
            ),
            Err(SourceError::AllocLimitExceeded { .. })
        ));
    }

    /**
     * Tests that the allocation budget bites at exactly the byte the
     * declared geometry costs, and not one byte either side. The cases
     * above sit far below the price, where a price wrong by a factor
     * refuses too; this one cannot pass unless the arithmetic is exact,
     * and it fixes the comparison at `>` rather than `>=`.
     * Works on three bands of 16-bit samples on purpose, so neither the
     * band count nor the sample size can be dropped from the price without
     * moving the answer: a 2x2x3 ushort raster is 24 bytes with both and
     * four without them.
     * Input: a 2x2x3 BITPIX 16 file at `max_alloc_bytes` 24 then 23 ->
     * Output: a clean `Rgb16` decode, then `AllocLimitExceeded
     * { needed: 24 }`.
     */
    #[test]
    fn the_allocation_budget_bites_at_exactly_the_declared_price() {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                   16",
            "NAXIS   =                    3",
            "NAXIS1  =                    2",
            "NAXIS2  =                    2",
            "NAXIS3  =                    3",
            "BZERO   =                32768",
            "BSCALE  =                    1",
        ]);
        for sample in 0..12i32 {
            file.extend_from_slice(&((sample - 32768) as i16).to_be_bytes());
        }
        pad_to_block(&mut file, 0);

        let exact = DecodeLimits::default().with_max_alloc_bytes(24);
        let raster = decode_fits(&file, exact).expect("24 bytes is exactly a 2x2x3 ushort raster");
        assert_eq!((raster.width(), raster.height()), (2, 2));
        assert_eq!(raster.format(), PixelFormat::Rgb16);

        let short = DecodeLimits::default().with_max_alloc_bytes(23);
        let err = decode_fits(&file, short).expect_err("23 bytes is one short of the raster");
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    what: "FITS pixel buffer",
                    geometry: Some(DeclaredGeometry {
                        width: 2,
                        height: 2,
                        bands: 3,
                    }),
                    needed_bytes: 24,
                    max_alloc_bytes: 23,
                }
            ),
            "{err:?}"
        );
    }

    /**
     * Tests that a byte count too large for a `u64` cannot wrap past the
     * budget check.
     * Works by lifting every `DecodeLimits` ceiling to its maximum, which
     * a caller is free to do, and then declaring the largest geometry the
     * axes can spell: `u32::MAX` on both axes with 65535 bands of float is
     * about 4.8e24 bytes. A wrapping product would look small enough to
     * pass and then index a buffer sized from a different number; a
     * saturating one gives `u64::MAX`, which the budget refuses as a
     * sentinel rather than comparing as a price, so the refusal lands on
     * the budget even though the budget itself is `u64::MAX`. It used to
     * fall through to the truncation check one line lower, which is how
     * FITS survived a hole the four codecs beside it did not.
     * Input: the largest declarable geometry with no ceiling in force ->
     * Output: `AllocLimitExceeded` carrying the saturated price, reached
     * without allocating anything.
     */
    #[test]
    fn a_byte_count_past_u64_cannot_wrap_past_the_budget() {
        let file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    3",
            &format!("NAXIS1  = {:>20}", u32::MAX),
            &format!("NAXIS2  = {:>20}", u32::MAX),
            &format!("NAXIS3  = {:>20}", u16::MAX),
        ]);
        let unbounded = DecodeLimits::default()
            .with_max_coord(u32::MAX)
            .with_max_width(u32::MAX)
            .with_max_height(u32::MAX)
            .with_max_pixels(u64::MAX)
            .with_max_alloc_bytes(u64::MAX);
        let err = decode_fits(&file, unbounded);
        assert!(
            matches!(
                err,
                Err(SourceError::AllocLimitExceeded {
                    needed_bytes: u64::MAX,
                    max_alloc_bytes: u64::MAX,
                    ..
                })
            ),
            "{err:?}"
        );
    }

    /**
     * Tests that a byte count landing exactly on 2^64 saturates instead of
     * wrapping to zero.
     * Works by declaring the one geometry whose product is the wrap point
     * itself: 2^24 by 2^24 with 2^14 bands of four-byte float is exactly
     * 2^64 bytes. A wrapping multiply gives `0` there, which clears every
     * budget, takes an empty payload slice and then indexes it; the
     * saturating one gives `u64::MAX`, which the budget refuses as a
     * sentinel. Every ceiling is lifted, as a caller may lift them, so the
     * price and the sentinel arm are the only guards left standing. The
     * neighbouring case above is far from the wrap point and so cannot
     * catch this.
     * Input: 2^24 x 2^24 x 2^14 at BITPIX -32 -> Output:
     * `AllocLimitExceeded` carrying the saturated price, refused without
     * allocating anything.
     */
    #[test]
    fn a_byte_count_of_exactly_two_to_the_64_saturates_rather_than_wrapping() {
        let file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                  -32",
            "NAXIS   =                    3",
            &format!("NAXIS1  = {:>20}", 1u64 << 24),
            &format!("NAXIS2  = {:>20}", 1u64 << 24),
            &format!("NAXIS3  = {:>20}", 1u64 << 14),
        ]);
        let unbounded = DecodeLimits::default()
            .with_max_coord(u32::MAX)
            .with_max_width(u32::MAX)
            .with_max_height(u32::MAX)
            .with_max_pixels(u64::MAX)
            .with_max_alloc_bytes(u64::MAX);
        let err = decode_fits(&file, unbounded);
        assert!(
            matches!(
                err,
                Err(SourceError::AllocLimitExceeded {
                    needed_bytes: u64::MAX,
                    max_alloc_bytes: u64::MAX,
                    ..
                })
            ),
            "{err:?}"
        );
    }

    /**
     * Tests that a data segment shorter than the header claims is refused.
     * Works by declaring a 4x3 image and supplying four bytes, so the
     * decoder has to notice before it indexes past the buffer.
     * Input: a header claiming 12 bytes over a 4-byte payload -> Output:
     * `TruncatedData` naming both counts.
     */
    #[test]
    fn a_short_data_segment_is_refused() {
        let mut file = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    3",
        ]);
        file.extend_from_slice(&[1, 2, 3, 4]);
        assert!(matches!(
            decode_fits(&file, DecodeLimits::default()),
            Err(SourceError::Fits(FitsError::TruncatedData {
                found: 4,
                needed: 12
            }))
        ));
    }

    /**
     * Tests that the truncation check sits exactly on the declared byte
     * count, not one either side of it.
     * Works by handing the same 4x3 BITPIX 8 header a payload of exactly
     * the twelve bytes it claims, which has to decode, and then one byte
     * fewer, which has to be refused. The case above is eight bytes short
     * so it reads the same whether the comparison is `<` or `<=`; this
     * pair pins the edge from both sides.
     * Input: 12 then 11 bytes under a 12-byte claim -> Output: the 4x3
     * raster, then `TruncatedData { found: 11, needed: 12 }`.
     */
    #[test]
    fn the_truncation_check_sits_on_the_exact_declared_byte_count() {
        let head = header(&[
            "SIMPLE  =                    T",
            "BITPIX  =                    8",
            "NAXIS   =                    2",
            "NAXIS1  =                    4",
            "NAXIS2  =                    3",
        ]);

        let mut exact = head.clone();
        exact.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
        let raster = decode_fits(&exact, DecodeLimits::default())
            .expect("a payload of exactly the declared length decodes");
        assert_eq!(raster.width(), 4);
        assert_eq!(raster.height(), 3);
        assert_eq!(
            raster.data(),
            &[9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4],
            "the last row in the file is the top row of the image"
        );

        let mut short = head;
        short.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
        assert!(matches!(
            decode_fits(&short, DecodeLimits::default()),
            Err(SourceError::Fits(FitsError::TruncatedData {
                found: 11,
                needed: 12
            }))
        ));
    }

    /**
     * Tests that bytes without the FITS magic are refused as such.
     * Works by offering a near-miss that shares the keyword but not the
     * fixed `= ` in columns 9 and 10.
     * Input: `SIMPLE=T` -> Output: `BadMagic` carrying what was read.
     */
    #[test]
    fn bytes_without_the_magic_are_refused() {
        match decode_fits(b"SIMPLE=T", DecodeLimits::default()) {
            Err(SourceError::Fits(FitsError::BadMagic { found })) => {
                assert_eq!(found, "SIMPLE=T");
            }
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    /**
     * Tests that the generated header matches cfitsio's, byte for byte.
     * Works by encoding a 4x3 mono uchar raster and comparing the first
     * cards against the bytes `vips copy in.pgm out.fits` produced on the
     * reference build, which is the only way to be sure, since the header
     * is written by cfitsio 4.6.4 rather than by vips.
     * Input: 4x3 `Gray8` -> Output: the nine measured cards, then blanks
     * to 2880.
     */
    #[test]
    fn encode_matches_the_reference_header_bytes() {
        let raster = Raster::new(4, 3, PixelFormat::Gray8, vec![0u8; 12]).unwrap();
        let encoded = raster.encode_fits().unwrap();
        let expected = [
            "SIMPLE  =                    T / file does conform to FITS standard",
            "BITPIX  =                    8 / number of bits per data pixel",
            "NAXIS   =                    2 / number of data axes",
            "NAXIS1  =                    4 / length of data axis 1",
            "NAXIS2  =                    3 / length of data axis 2",
            "EXTEND  =                    T / FITS dataset may contain extensions",
            "COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy",
            "COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H",
            "END",
        ];
        for (i, card) in expected.iter().enumerate() {
            assert_eq!(
                &encoded[i * CARD..(i + 1) * CARD],
                pad_card(card).as_slice(),
                "card {i} should match the cfitsio-generated header"
            );
        }
        assert_eq!(encoded.len(), BLOCK * 2, "one header block, one data block");
        assert!(
            encoded[expected.len() * CARD..BLOCK]
                .iter()
                .all(|&b| b == b' '),
            "the header block is space-filled after END"
        );
    }

    /**
     * Tests that a 16-bit save declares the unsigned convention.
     * Works by encoding a `Gray16` raster and reading back the two cards
     * cfitsio appends for `USHORT_IMG`, measured off a real save.
     * Input: 4x1 `Gray16` -> Output: BITPIX 16 with BZERO 32768 and
     * BSCALE 1, and stored samples shifted down by 32768.
     */
    #[test]
    fn encode_declares_the_unsigned_16_convention() {
        let samples: Vec<u16> = vec![0, 4097, 8194, 12291];
        let mut data = Vec::new();
        for s in &samples {
            data.extend_from_slice(&s.to_ne_bytes());
        }
        let raster = Raster::new(4, 1, PixelFormat::Gray16, data).unwrap();
        let encoded = raster.encode_fits().unwrap();
        let text = String::from_utf8(encoded[..BLOCK].to_vec()).unwrap();
        assert!(text.contains("BITPIX  =                   16 / number of bits per data pixel"));
        assert!(text.contains(
            "BZERO   =                32768 / offset data range to that of unsigned short"
        ));
        assert!(text.contains("BSCALE  =                    1 / default scaling factor"));
        let stored: Vec<i16> = encoded[BLOCK..BLOCK + 8]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| i16::from_be_bytes(*c))
            .collect();
        assert_eq!(stored, vec![-32768, -28671, -24574, -20477]);
    }

    /**
     * Tests that a save writes planes bottom row first.
     * Works by encoding a two-band raster whose samples are all distinct
     * and reading the data segment back by hand, which pins both halves of
     * the layout at once.
     * Input: 2x2 `Multi8(2)` -> Output: band 0's bottom row, band 0's top
     * row, then band 1's the same way.
     */
    #[test]
    fn encode_writes_planes_bottom_row_first() {
        let format = PixelFormat::with_channels(2, 1).unwrap();
        // Pixel (x, y) band b = y * 4 + x * 2 + b.
        let data: Vec<u8> = (0..8u8).collect();
        let raster = Raster::new(2, 2, format, data).unwrap();
        let encoded = raster.encode_fits().unwrap();
        assert_eq!(&encoded[BLOCK..BLOCK + 8], &[4, 6, 0, 2, 5, 7, 1, 3]);
    }

    /**
     * Tests that a load followed by a save reproduces the input bytes.
     * Works by round-tripping the mono fixture, which carries the same
     * cards cfitsio generates, so nothing should survive the record filter
     * to change the header.
     * Input: the 4x3 fixture -> Output: the same bytes back.
     */
    #[test]
    fn a_generated_header_round_trips() {
        let raster = Raster::new(4, 3, PixelFormat::Gray8, (0..12u8).collect()).unwrap();
        let encoded = raster.encode_fits().unwrap();
        let back = decode_fits(&encoded, DecodeLimits::default()).unwrap();
        assert_eq!(back.data(), raster.data());
        assert_eq!(back.encode_fits().unwrap(), encoded);
    }

    /**
     * Tests that attached records are filtered the way vips filters them.
     * Works by attaching one record cfitsio generates itself, one keyword
     * twice, and one repeatable keyword twice, then reading the written
     * header back.
     * Input: `BITPIX`, two `OBJECT`s, two `HISTORY`s -> Output: the
     * `BITPIX` dropped, one `OBJECT`, both `HISTORY`s.
     */
    #[test]
    fn attached_records_follow_the_reference_filter() {
        let mut raster = Raster::new(2, 1, PixelFormat::Gray8, vec![0, 1]).unwrap();
        for (i, record) in [
            "BITPIX  =                   99 / cfitsio writes this one",
            "OBJECT  = 'M31     '",
            "OBJECT  = 'M32     '",
            "HISTORY   first",
            "HISTORY   second",
        ]
        .iter()
        .enumerate()
        {
            raster.set_field(&format!("fits-{i}"), (*record).to_string().into());
        }
        let encoded = raster.encode_fits().unwrap();
        let text = String::from_utf8(encoded[..BLOCK].to_vec()).unwrap();
        assert!(
            !text.contains("BITPIX  =                   99"),
            "cfitsio generates BITPIX itself, so vips never writes it again"
        );
        assert!(text.contains("OBJECT  = 'M31     '"));
        assert!(
            !text.contains("OBJECT  = 'M32     '"),
            "a non-repeatable keyword is written once"
        );
        assert!(text.contains("HISTORY   first"));
        assert!(
            text.contains("HISTORY   second"),
            "HISTORY is on the repeatable list"
        );
    }

    /**
     * Tests that the value parser stops at the right slash.
     * Works by giving a card whose quoted string contains a `/`, which a
     * naive split would cut in half.
     * Input: `OBJECT  = 'a/b' / comment` -> Output: the quoted value with
     * the comment gone.
     */
    #[test]
    fn a_quoted_value_may_contain_a_slash() {
        let card = pad_card("OBJECT  = 'a/b'      / and a comment");
        assert_eq!(card_value(&card), Some("'a/b'"));
        assert_eq!(card_keyword(&card), "OBJECT");
    }

    /**
     * Tests that a card with no value field reads as valueless.
     * Works by offering a `COMMENT` card, which carries free text from
     * column 9 rather than a `= ` pair, and a blank card.
     * Input: both -> Output: `None` from the value reader.
     */
    #[test]
    fn a_comment_card_carries_no_value() {
        assert_eq!(card_value(&pad_card("COMMENT   free text")), None);
        assert_eq!(card_value(&pad_card("")), None);
    }

    /**
     * Tests that a FITS real spelled with a `D` exponent parses.
     * Works by driving both spellings through the reader, since FITS
     * inherits the Fortran `D` form and Rust's parser only knows `E`.
     * Input: `1.0D+02` and `1.0E+02` -> Output: 100.0 from both.
     */
    #[test]
    fn fits_reals_accept_a_fortran_exponent() {
        for spelling in ["1.0D+02", "1.0E+02", "100"] {
            let got = parse_real(spelling).unwrap();
            assert!(
                (got - 100.0).abs() < 1e-9,
                "{spelling} should parse as 100.0, got {got}"
            );
        }
        assert_eq!(parse_real("not a number"), None);
    }

    /**
     * Sweeps the seeded fuzz corpus through the decoder, so every
     * malformation it holds is a `cargo test` regression rather than
     * something only a fuzz run would notice. Works by decoding every file
     * under `fuzz/corpus/fuzz_fits/` and checking each against the outcome
     * its name promises, which is what keeps the seed names honest as the
     * error enum grows.
     * Input: the corpus files -> Output: the named typed error from each
     * malformed one, a raster from each `valid-` one, and no panic from
     * any of them.
     */
    #[test]
    fn the_fuzz_corpus_decodes_or_fails_exactly_as_named() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fuzz")
            .join("corpus")
            .join("fuzz_fits");
        let limits = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);

        let mut seen = 0usize;
        for entry in std::fs::read_dir(&dir).expect("the seeded corpus is in the tree") {
            let path = entry.expect("corpus entry").path();
            let name = path
                .file_name()
                .expect("corpus entries are files")
                .to_string_lossy()
                .into_owned();
            let bytes = std::fs::read(&path).expect("corpus file");
            let result = decode_fits(&bytes, limits);
            seen += 1;

            let ok = match name.as_str() {
                "empty" | "wrong-magic" => {
                    matches!(result, Err(SourceError::Fits(FitsError::BadMagic { .. })))
                }
                // The magic on its own is nine bytes, so it gets past the
                // prefix check and dies on the first block instead.
                "magic-only"
                | "truncated-header-block"
                | "header-without-end"
                | "header-without-end-multiblock"
                | "empty-unit-chain" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::TruncatedHeader { .. }))
                ),
                "naxis-11" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::BadAxisCount { naxis: 11 }))
                ),
                "naxis-4-higher-axis-of-2" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::HighDimensionNotEmpty {
                        axis: 4,
                        ..
                    }))
                ),
                "naxis-declared-but-absent" | "naxis1-not-a-number" | "bitpix-missing" => {
                    matches!(
                        result,
                        Err(SourceError::Fits(FitsError::BadHeaderCard { .. }))
                    )
                }
                "naxis-zero-extent" | "naxis-negative-extent" | "bands-past-u16" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::DimensionOutOfBounds { .. }))
                ),
                "bitpix-16-signed" | "bitpix-32" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::UnsupportedCarrier { .. }))
                ),
                "bitpix-64" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::UnsupportedBitpix {
                        bitpix: 64
                    }))
                ),
                "bitpix-8-rescaled" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::UnsupportedScaling { .. }))
                ),
                "truncated-data" => matches!(
                    result,
                    Err(SourceError::Fits(FitsError::TruncatedData { .. }))
                ),
                "geometry-bomb-coord" => {
                    matches!(result, Err(SourceError::CoordLimitExceeded { .. }))
                }
                "geometry-bomb-pixels" => {
                    matches!(result, Err(SourceError::DimensionLimitExceeded { .. }))
                }
                "geometry-bomb-alloc" => {
                    matches!(result, Err(SourceError::AllocLimitExceeded { .. }))
                }
                // A card carrying bytes outside printable ASCII is not a
                // structural fault: the geometry cards are still intact, so
                // the file loads and the bad bytes come back through the
                // lossy record decode.
                "non-ascii-card" => result.is_ok(),
                other => {
                    assert!(
                        other.starts_with("valid-"),
                        "corpus seed {other:?} has no asserted outcome; name it \
                         after the malformation it carries or prefix it valid-"
                    );
                    result.is_ok()
                }
            };
            assert!(ok, "corpus seed {name:?} gave {result:?}");
        }
        assert!(seen >= 25, "the seeded corpus should still be in the tree");
    }

    /**
     * Tests that every `PixelFormat` has a FITS spelling.
     * Works by sweeping one raster per carrier width and asserting the
     * encoder accepts it and the decoder gives the same pixels back, which
     * is what makes the save side total where the load side has a ceiling.
     * Input: 8-, 16- and 32-bit carriers -> Output: a clean round trip
     * for each.
     */
    #[test]
    fn every_carrier_round_trips() {
        let cases = [
            (PixelFormat::Gray8, 12usize),
            (PixelFormat::Rgb8, 36),
            (PixelFormat::Rgba8, 48),
            (PixelFormat::Gray16, 24),
            (PixelFormat::Rgb16, 72),
            (PixelFormat::Rgba16, 96),
            (PixelFormat::RgbaF32, 192),
            (PixelFormat::with_channels(5, 1).unwrap(), 60),
            (PixelFormat::with_channels(2, 4).unwrap(), 96),
        ];
        for (format, len) in cases {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let raster = Raster::new(4, 3, format, data).unwrap();
            let encoded = raster.encode_fits().unwrap();
            assert_eq!(encoded.len() % BLOCK, 0, "{format:?} pads to whole blocks");
            let back = decode_fits(&encoded, DecodeLimits::default()).unwrap();
            assert_eq!(back.format(), format, "{format:?} round-trips its carrier");
            assert_eq!(back.data(), raster.data(), "{format:?} round-trips pixels");
        }
    }
}
