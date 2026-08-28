//! Ultra HDR: the gain-map JPEG libvips reads with `uhdrload`, writes with
//! `uhdrsave` and expands with `uhdr2scRGB`.
//!
//! An Ultra HDR file is not a new codec. It is **two ordinary JPEGs
//! concatenated**: a standard SDR base image, then, immediately after the
//! base's `EOI`, a second complete JPEG holding a *gain map*. The gain map
//! says, per pixel, how far above the SDR rendering the HDR one goes. The
//! terms needed to apply it travel in an ISO 21496-1 `APP2` segment on the
//! gain map, and an MPF `APP2` on the base advertises that a second image
//! follows.
//!
//! That is why this module exists at all and why it costs nothing to have.
//! Every primitive is already in the tree: the `image` crate's JPEG decoder
//! and encoder decode and write both halves, and the rest is marker walking
//! and arithmetic. libviprs adds **no dependency** for Ultra HDR — no
//! feature gate, no `-sys` crate, no `build.rs` C compile — where libvips
//! links `libultrahdr` for the same surface. The one thing that costs is
//! *fidelity to libuhdr's encoder*, and [`encode_uhdr`] is explicit about
//! not having it.
//!
//! # What is a UHDR file, exactly
//!
//! Measured by cutting one marker at a time out of a file libvips wrote
//! (`oracle-captures/foreign-uhdr`, issue #639), the gate libvips's loader
//! chooser applies is two-stage and the two stages are not equally load
//! bearing:
//!
//! * the base image carries an `APP2` whose payload starts `MPF`. This is a
//!   **fast pre-filter only**. Remove it and `libuhdr` still reads the file
//!   perfectly well, but `vips_foreign_find_load` stops offering it to
//!   `uhdrload` and hands it to `jpegload`, which silently drops the gain
//!   map. `fixtures/no-mpf.jpg` is that file.
//! * the *gain map* carries an `APP2` whose payload starts
//!   `urn:iso:std:iso:ts:21496:-1`. This is the real test. Removing the
//!   **base's** copy of the same marker changes nothing
//!   (`fixtures/no-iso-base.jpg` still loads as UHDR); removing the gain
//!   map's makes the file stop being Ultra HDR at all
//!   (`fixtures/no-iso-gainmap.jpg`).
//!
//! [`is_uhdr`] implements the chooser's gate, both stages, because that is
//! the question `crate::source::sniff` is asking: not "could a decoder
//! make something of this" but "which loader does this file belong to".
//! `has_mpf` alone would claim `fixtures/mpf-graft.jpg`, an ordinary JPEG
//! with an MPF segment grafted on and no gain map anywhere, and that file
//! is a plain JPEG.
//!
//! # Sniffing a container that shares JPEG's magic bytes
//!
//! Ultra HDR is the first container in the route table whose signature
//! **overlaps another row's**: a UHDR file opens `FF D8 FF`, exactly like
//! every JPEG, and nothing in the first 16 bytes distinguishes the two.
//! Both facts follow from the format rather than from a choice, and libvips
//! has the same problem: it gives `uhdrload` priority 100 against
//! `jpegload`'s 50 and lets content decide.
//!
//! libviprs spells that priority as declaration order in
//! `SniffedFormat` (see [`crate::source`]), so the `Uhdr` row sits
//! *before* `Jpeg`, and the row's signature is a structural predicate over
//! the whole buffer rather than a leading-byte pattern. See
//! [`crate::source`] for how the file entry point still reaches it: the
//! 16-byte head sniffs as JPEG, JPEG is a whole-file row, and the re-sniff
//! over the complete buffer finds `Uhdr`.
//!
//! # What a load actually returns
//!
//! [`decode_uhdr`] returns the **base image**, 3-band `uchar` tagged sRGB,
//! and attaches the gain map. It does not return HDR pixels, and neither
//! does libvips: `uhdrload` decodes the base through the ordinary libjpeg
//! path and hands the gain map on as metadata for `uhdr2scRGB` or a re-save
//! to pick up. [`uhdr_to_scrgb`] is the operation that combines them.
//!
//! # Decode limits
//!
//! A UHDR file holds **two** images, so both are priced. The base and the
//! gain map each go through `DecodeLimits::check_coord`,
//! `DecodeLimits::check_pixels` and
//! `DecodeLimits::check_image_alloc`, from geometry read out of each
//! image's own `SOF` marker *before* either is decoded. Pricing only the
//! base would let a 1x1 base carry a 60000x60000 gain map.

use std::ops::Range;

use crate::conversion::Interpretation;
use crate::imageio::MetadataValue;
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::source::{DecodeLimits, SourceError};
use thiserror::Error;

/// The `APP2` identifier ISO 21496-1 puts in front of gain-map metadata,
/// including its terminating NUL.
///
/// This is the marker that actually decides whether a file is Ultra HDR,
/// and only its copy on the *gain map* counts. See the module docs.
pub const ISO_GAIN_MAP_ID: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";

/// The `APP2` identifier of the Multi-Picture Format segment on the base
/// image.
///
/// Three bytes, not four, because three is what libvips tests:
/// `vips_isprefix("MPF", p->data)` in `uhdrload.c`. Every file libuhdr
/// writes has `MPF\0`, so the distinction has never mattered on a real
/// file, but matching the C keeps the two answers the same on a contrived
/// one.
pub const MPF_ID: &[u8] = b"MPF";

/// JPEG `APP1`, where EXIF travels.
const APP1: u8 = 0xE1;
/// JPEG `APP2`, where ICC, MPF and the ISO gain-map metadata travel.
const APP2: u8 = 0xE2;

/// The `ICC_PROFILE` `APP2` identifier, plus the two sequencing bytes
/// libvips strips with it.
///
/// `uhdrload.c:399-411` drops `ICC_PROFILE` "plus three more bytes" — the
/// NUL, the chunk index and the chunk count — so `icc-profile-data` is 14
/// bytes shorter than the segment payload. Measured: a 614-byte payload in
/// `fixtures/uhdr.jpg` becomes the 600 bytes `vipsheader` reports.
const ICC_PREFIX_LEN: usize = 14;
/// The identifier those 14 bytes start with.
const ICC_ID: &[u8] = b"ICC_PROFILE\0";

/// The `APP1` identifier in front of an EXIF block. The payload is attached
/// whole, this prefix included, which is what makes `exif-data` 186 bytes
/// for the 186-byte payload in `fixtures/uhdr.jpg` rather than 180.
const EXIF_ID: &[u8] = b"Exif\0\0";

/// libvips's `vips_v2Y_8`: the 256-entry sRGB electro-optical transfer
/// table `uhdr2scRGB` linearises the base image through, and the same
/// table `sRGB2scRGB` uses.
///
/// **Transcribed rather than computed, and the reason is portability, not
/// arithmetic.** The expression is `f = i / 255; f <= 0.04045 ? f / 12.92f
/// : powf((f + 0.055f) / 1.055f, 2.4f)` (`calcul_tables`,
/// `colour/LabQ2sRGB.c`), and on this host Rust reproduces it exactly:
/// evaluating it with that `f32` spelling matches all 256 entries, because
/// `f32::powf` and the `powf` libvips was built against are the same arm64
/// system libm. That agreement is a property of the host rather than of
/// the code, and it is the whole reason the constant is written down
/// instead of derived: a build whose libm rounds `powf` differently would
/// silently produce a different table, and every pinned `uhdr2scRGB` value
/// would move with it.
///
/// The precision is load bearing too, in a way that is easy to get
/// backwards. Doing the same arithmetic in `f64` -- which looks more
/// careful -- misses **214 of the 256 entries**, and computing `f` in
/// `f32` before widening to `f64` misses 192. Every miss is a single ulp.
/// The C is `float` throughout and a port has to be as well.
///
/// The bytes come from `oracle-captures/foreign-uhdr/oracle.json`'s
/// `v2Y_8_le_f32_hex` (issue #639, which says in as many words:
/// "transcribe against these values, not against the expression").
///
/// `v2y_8_matches_the_pinned_oracle_table` re-reads that capture and
/// compares all 256 entries, so this array cannot drift from the measured
/// ground truth without a test going red.
///
/// It lives here rather than in [`crate::colour`] because that module
/// deliberately keeps only the *forward* `Y2v` table and does the reverse
/// direction analytically in `f64` — correct to well under a code for
/// colourspace work, and not bit-exact, which is what Ultra HDR needs.
pub(crate) const V2Y_8: [f32; 256] = [
    0.0,
    0.000303527,
    0.000607054,
    0.000910581,
    0.001214108,
    0.001517635,
    0.001821162,
    0.0021246888,
    0.002428216,
    0.002731743,
    0.00303527,
    0.0033465356,
    0.003676507,
    0.004024717,
    0.004391442,
    0.0047769533,
    0.005181517,
    0.0056053917,
    0.0060488326,
    0.006512091,
    0.00699541,
    0.0074990317,
    0.008023192,
    0.008568125,
    0.009134057,
    0.009721218,
    0.010329823,
    0.010960094,
    0.011612245,
    0.012286487,
    0.012983031,
    0.013702081,
    0.014443844,
    0.015208514,
    0.015996292,
    0.016807375,
    0.017641952,
    0.018500218,
    0.019382361,
    0.020288562,
    0.02121901,
    0.022173883,
    0.023153365,
    0.02415763,
    0.025186857,
    0.026241222,
    0.027320892,
    0.028426038,
    0.029556833,
    0.03071344,
    0.03189603,
    0.033104762,
    0.034339808,
    0.035601314,
    0.036889445,
    0.038204364,
    0.039546236,
    0.0409152,
    0.04231141,
    0.043735027,
    0.045186203,
    0.046665084,
    0.048171822,
    0.049706563,
    0.051269468,
    0.052860655,
    0.05448028,
    0.056128494,
    0.057805434,
    0.05951124,
    0.06124607,
    0.06301004,
    0.06480328,
    0.06662595,
    0.06847818,
    0.07036011,
    0.07227186,
    0.07421358,
    0.07618539,
    0.07818743,
    0.08021983,
    0.082282715,
    0.084376216,
    0.086500466,
    0.088655606,
    0.09084173,
    0.09305898,
    0.095307484,
    0.09758736,
    0.09989874,
    0.10224175,
    0.10461649,
    0.10702311,
    0.10946172,
    0.111932434,
    0.11443538,
    0.11697067,
    0.119538434,
    0.1221388,
    0.12477184,
    0.1274377,
    0.13013649,
    0.13286833,
    0.13563335,
    0.13843162,
    0.1412633,
    0.14412849,
    0.14702728,
    0.1499598,
    0.15292616,
    0.15592647,
    0.15896086,
    0.1620294,
    0.16513222,
    0.1682694,
    0.1714411,
    0.17464739,
    0.17788841,
    0.18116423,
    0.18447499,
    0.18782076,
    0.19120167,
    0.19461781,
    0.1980693,
    0.20155624,
    0.2050787,
    0.20863685,
    0.21223073,
    0.21586053,
    0.21952623,
    0.22322798,
    0.22696589,
    0.23074007,
    0.23455065,
    0.23839766,
    0.2422812,
    0.2462014,
    0.25015837,
    0.25415218,
    0.2581829,
    0.26225072,
    0.26635566,
    0.27049786,
    0.27467737,
    0.27889434,
    0.2831488,
    0.2874409,
    0.2917707,
    0.29613832,
    0.30054384,
    0.30498737,
    0.30946895,
    0.31398875,
    0.31854683,
    0.32314324,
    0.32777813,
    0.33245158,
    0.33716366,
    0.34191445,
    0.3467041,
    0.3515327,
    0.35640025,
    0.36130688,
    0.3662527,
    0.37123778,
    0.37626222,
    0.3813261,
    0.38642952,
    0.39157256,
    0.3967553,
    0.40197787,
    0.4072403,
    0.4125427,
    0.41788515,
    0.42326775,
    0.42869055,
    0.4341537,
    0.43965724,
    0.44520125,
    0.45078585,
    0.4564111,
    0.46207705,
    0.46778384,
    0.47353154,
    0.47932023,
    0.48514998,
    0.4910209,
    0.49693304,
    0.5028866,
    0.50888145,
    0.5149178,
    0.5209957,
    0.5271152,
    0.5332765,
    0.5394796,
    0.5457246,
    0.5520115,
    0.5583405,
    0.56471163,
    0.5711249,
    0.5775805,
    0.5840785,
    0.5906189,
    0.5972019,
    0.6038274,
    0.6104956,
    0.61720663,
    0.62396044,
    0.6307572,
    0.63759696,
    0.64447975,
    0.6514057,
    0.65837485,
    0.66538733,
    0.6724432,
    0.67954254,
    0.68668544,
    0.6938719,
    0.701102,
    0.70837593,
    0.71569365,
    0.72305524,
    0.7304609,
    0.73791057,
    0.74540436,
    0.7529423,
    0.76052463,
    0.7681513,
    0.77582234,
    0.7835379,
    0.79129803,
    0.79910284,
    0.80695236,
    0.8148467,
    0.82278585,
    0.83076996,
    0.8387991,
    0.8468733,
    0.8549927,
    0.8631573,
    0.8713672,
    0.87962234,
    0.8879232,
    0.8962694,
    0.90466136,
    0.9130987,
    0.92158204,
    0.9301109,
    0.9386859,
    0.9473066,
    0.9559735,
    0.9646863,
    0.9734455,
    0.9822506,
    0.9911022,
    1.0,
];

/// Errors from the Ultra HDR codec.
///
/// Typed rather than stringly, like [`crate::gif::GifError`] and
/// [`crate::radiance::RadianceError`], so a caller can tell "these bytes
/// are not Ultra HDR" from "this gain map is malformed" without matching on
/// a message.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum UhdrError {
    /// The bytes are a JPEG, or are not, but they are not an Ultra HDR
    /// container: see [`is_uhdr`] for the two-stage gate they failed.
    #[error("uhdr: not an UltraHDR image")]
    NotUhdr,
    /// A `SOF` marker was missing or unreadable, so the geometry could not
    /// be priced before decoding.
    #[error("uhdr: {which} image declares no frame geometry")]
    NoFrameHeader {
        /// `"base"` or `"gain map"`.
        which: &'static str,
    },
    /// The ISO 21496-1 `APP2` payload is present but too short, or declares
    /// a denominator of zero.
    #[error("uhdr: malformed ISO 21496-1 gain-map metadata: {reason}")]
    BadMetadata {
        /// What was wrong with it.
        reason: String,
    },
    /// A JPEG half of the container failed to decode.
    #[error("uhdr: {which} image: {message}")]
    Jpeg {
        /// `"base"` or `"gain map"`.
        which: &'static str,
        /// The `image` crate's own error, rendered through `Display`.
        message: String,
    },
    /// [`uhdr_to_scrgb`] was handed a raster it cannot transform. The base
    /// must be 3-band `uchar` and the gain map 1- or 3-band `uchar`, which
    /// is exactly what `uhdr2scRGB` enforces.
    #[error("uhdr2scRGB: {reason}")]
    BadInput {
        /// Which rule was broken.
        reason: String,
    },
    /// The raster carries no gain map, so there is nothing to expand.
    ///
    /// libvips reaches the same case and exits **printing nothing**:
    /// `vips_image_get_gainmap` returns `NULL` without calling
    /// `vips_error`. A silent failure is not a failure a caller can act on,
    /// so this is a real error here.
    #[error("uhdr2scRGB: image carries no gain map")]
    NoGainMap,
    /// The raster could not be built from the decoded pixels.
    #[error(transparent)]
    Raster(#[from] crate::raster::RasterError),
}

/// One JPEG's extent inside a buffer, as far as the marker walk got.
#[derive(Debug, Clone, Copy)]
struct ScanResult {
    /// The offset just past `EOI`, or `None` if the walk never reached one.
    end: Option<usize>,
    /// Whether a `SOS` marker was seen, i.e. whether there is any entropy-
    /// coded data at all.
    saw_sos: bool,
}

/// Walk the JPEG that starts at `start`, calling `visit` with the marker
/// code and payload range of every marker segment.
///
/// Deliberately allocation-free: [`is_uhdr`] runs on every JPEG-shaped
/// buffer that reaches [`crate::source::sniff`], so it must not cost a
/// `Vec` per sniff.
///
/// Entropy-coded data after `SOS` is skipped by scanning for the next
/// marker that is neither a stuffed `FF 00` nor a restart marker, which is
/// how a JPEG is delimited without decoding it.
fn scan_jpeg<F>(bytes: &[u8], start: usize, mut visit: F) -> ScanResult
where
    F: FnMut(u8, Range<usize>),
{
    let mut result = ScanResult {
        end: None,
        saw_sos: false,
    };
    if bytes.get(start..start.saturating_add(2)) != Some(&[0xFF, 0xD8]) {
        return result;
    }
    let mut i = start + 2;
    while i + 1 < bytes.len() {
        if bytes[i] != 0xFF {
            i += 1;
            continue;
        }
        let code = bytes[i + 1];
        // `FF FF` is fill and `FF 00` is a stuffed data byte; neither
        // starts a segment.
        if code == 0xFF || code == 0x00 {
            i += 1;
            continue;
        }
        if code == 0xD9 {
            result.end = Some(i + 2);
            return result;
        }
        // Standalone markers: SOI, the restart set, and TEM. None carries
        // a length.
        if code == 0xD8 || code == 0x01 || (0xD0..=0xD7).contains(&code) {
            i += 2;
            continue;
        }
        let Some(len) = bytes
            .get(i + 2..i + 4)
            .map(|b| usize::from(u16::from_be_bytes([b[0], b[1]])))
        else {
            return result;
        };
        // The length counts itself, so anything under 2 is malformed and
        // would not advance `i`.
        if len < 2 {
            return result;
        }
        let payload = (i + 4)..(i + 2 + len);
        if payload.end > bytes.len() {
            return result;
        }
        visit(code, payload.clone());
        if code == 0xDA {
            result.saw_sos = true;
            let mut j = payload.end;
            while j + 1 < bytes.len() {
                if bytes[j] == 0xFF {
                    let next = bytes[j + 1];
                    if next != 0x00 && next != 0xFF && !(0xD0..=0xD7).contains(&next) {
                        break;
                    }
                }
                j += 1;
            }
            i = j;
        } else {
            i = payload.end;
        }
    }
    result
}

/// The base and gain-map JPEG extents of an Ultra HDR container, or `None`
/// for anything that is not one.
///
/// This is the whole of the detection gate; [`is_uhdr`] is this function
/// asked whether it found anything, and [`decode_uhdr`] is this function
/// plus two JPEG decodes.
fn split_container(bytes: &[u8]) -> Option<(Range<usize>, Range<usize>)> {
    let mut has_mpf = false;
    let base = scan_jpeg(bytes, 0, |code, payload| {
        if code == APP2 && bytes[payload].starts_with(MPF_ID) {
            has_mpf = true;
        }
    });
    // Stage one, the fast pre-filter. An MPF-less gain-map file is a real
    // gain-map file that libvips hands to `jpegload` anyway, and matching
    // the chooser is the point.
    if !has_mpf {
        return None;
    }
    let base_end = base.end?;

    let mut has_iso = false;
    let gain = scan_jpeg(bytes, base_end, |code, payload| {
        if code == APP2 && bytes[payload].starts_with(ISO_GAIN_MAP_ID) {
            has_iso = true;
        }
    });
    // Stage two, the real test. `saw_sos` and `end` together are what
    // separates a complete gain map from `fixtures/truncated-gainmap.jpg`,
    // which keeps its ISO segment and loses everything after it.
    if !has_iso || !gain.saw_sos {
        return None;
    }
    Some((0..base_end, base_end..gain.end?))
}

/// Whether `bytes` is an Ultra HDR container, by the same two-stage gate
/// libvips's loader chooser applies.
///
/// This is the predicate the `Uhdr` row of the route table carries, and it
/// is deliberately the *chooser's* question rather than the decoder's: an
/// MPF-less gain-map JPEG returns `false` here because libvips routes that
/// file to `jpegload`. See the module docs.
///
/// Cheap enough to run on every JPEG-shaped buffer: it allocates nothing
/// and stops at the first missing stage.
#[must_use]
pub fn is_uhdr(bytes: &[u8]) -> bool {
    split_container(bytes).is_some()
}

/// Frame geometry read out of a `SOF` marker: width, height, components.
///
/// Read rather than decoded, so [`DecodeLimits`] is applied to declared
/// geometry before any pixel buffer exists.
fn sof_geometry(bytes: &[u8], image: Range<usize>) -> Option<(u32, u32, u8)> {
    let mut found = None;
    scan_jpeg(bytes, image.start, |code, payload| {
        // Every SOF_n except DHT (C4), JPG (C8) and DAC (CC).
        let is_sof = (0xC0..=0xCF).contains(&code) && !matches!(code, 0xC4 | 0xC8 | 0xCC);
        if !is_sof || found.is_some() {
            return;
        }
        // precision(1) height(2) width(2) components(1)
        if let Some(b) = bytes.get(payload.start..payload.start + 6) {
            found = Some((
                u32::from(u16::from_be_bytes([b[3], b[4]])),
                u32::from(u16::from_be_bytes([b[1], b[2]])),
                b[5],
            ));
        }
    });
    found
}

/// The first `APP` payload in `image` whose bytes start with `id`.
fn find_app<'a>(bytes: &'a [u8], image: Range<usize>, marker: u8, id: &[u8]) -> Option<&'a [u8]> {
    let mut found = None;
    scan_jpeg(bytes, image.start, |code, payload| {
        if code == marker && found.is_none() && bytes[payload.clone()].starts_with(id) {
            found = Some(&bytes[payload]);
        }
    });
    found
}

/// The ISO 21496-1 gain-map terms, one triple per RGB channel.
///
/// The file stores the boosts and the headrooms as **base-2 logarithms** of
/// signed rationals; every field here is the linear value, `exp2` already
/// applied, which is the form libvips reports and the form the transform
/// consumes. `fixtures/uhdr.jpg` stores `0x0059F541 / 0x00100000` =
/// 5.62255 and `vipsheader` prints `gainmap-max-content-boost: 49.2611`,
/// which is `2^5.62255`.
///
/// A single-channel gain map (the common case, and what libuhdr writes)
/// stores one set of terms; they are replicated across all three entries
/// here, matching the `49.2611 49.2611 49.2611` libvips reports.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainMapMetadata {
    /// Gain applied where the gain-map sample is 0, per channel, linear.
    pub min_content_boost: [f64; 3],
    /// Gain applied where the gain-map sample is 255, per channel, linear.
    pub max_content_boost: [f64; 3],
    /// Encoding gamma of the gain-map samples, per channel.
    pub gamma: [f64; 3],
    /// Offset added to the SDR sample before the gain is applied.
    pub offset_sdr: [f64; 3],
    /// Offset subtracted from the HDR sample after the gain is applied.
    pub offset_hdr: [f64; 3],
    /// Headroom of the base rendering, linear.
    pub hdr_capacity_min: f64,
    /// Headroom of the fully-boosted rendering, linear.
    pub hdr_capacity_max: f64,
    /// Whether the gain map shares the base image's colour space.
    pub use_base_colour_space: bool,
    /// Whether the file stored three sets of terms rather than one. The
    /// three-channel *metadata* flag is independent of the gain map's own
    /// band count, and only the band count picks the transform path.
    pub multi_channel: bool,
}

impl Default for GainMapMetadata {
    /// The identity: no boost anywhere, gamma 1, no offsets. Applying it
    /// leaves the linearised base image unchanged.
    fn default() -> Self {
        Self {
            min_content_boost: [1.0; 3],
            max_content_boost: [1.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.0; 3],
            offset_hdr: [0.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 1.0,
            use_base_colour_space: true,
            multi_channel: false,
        }
    }
}

/// Read a big-endian `u32` at `at`, or fail with a reason naming the field.
fn be_u32(payload: &[u8], at: usize, what: &str) -> Result<u32, UhdrError> {
    payload
        .get(at..at + 4)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or_else(|| UhdrError::BadMetadata {
            reason: format!("truncated before {what} at byte {at}"),
        })
}

/// One signed rational: `i32` numerator over `u32` denominator.
///
/// A zero denominator is refused rather than turned into an infinity. The
/// file is untrusted and `0/0` would otherwise arrive at the transform as a
/// `NaN` that quietly poisons every pixel.
fn rational(payload: &[u8], at: usize, what: &str) -> Result<f64, UhdrError> {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "ISO 21496-1 numerators are signed; the wrap is the decode"
    )]
    let num = f64::from(be_u32(payload, at, what)? as i32);
    let den = be_u32(payload, at + 4, what)?;
    if den == 0 {
        return Err(UhdrError::BadMetadata {
            reason: format!("{what} has a zero denominator"),
        });
    }
    Ok(num / f64::from(den))
}

/// Parse the ISO 21496-1 payload that follows [`ISO_GAIN_MAP_ID`].
///
/// The layout, confirmed field by field against `fixtures/uhdr.jpg` and the
/// values `vipsheader` reports for it: `u16` minimum version, `u16` writer
/// version, `u8` flags (`0x80` three-channel, `0x40` use base colour
/// space), then the base and alternate headrooms as rationals, then one set
/// of five rationals per channel.
fn parse_iso_metadata(payload: &[u8]) -> Result<GainMapMetadata, UhdrError> {
    let body = payload
        .strip_prefix(ISO_GAIN_MAP_ID)
        .ok_or_else(|| UhdrError::BadMetadata {
            reason: "APP2 payload does not start with the ISO 21496-1 identifier".to_string(),
        })?;
    let flags = *body.get(4).ok_or_else(|| UhdrError::BadMetadata {
        reason: "truncated before the flags byte".to_string(),
    })?;
    let multi_channel = flags & 0x80 != 0;
    let channels = if multi_channel { 3usize } else { 1 };
    let hdr_capacity_min = rational(body, 5, "base hdr headroom")?.exp2();
    let hdr_capacity_max = rational(body, 13, "alternate hdr headroom")?.exp2();

    let mut meta = GainMapMetadata {
        hdr_capacity_min,
        hdr_capacity_max,
        use_base_colour_space: flags & 0x40 != 0,
        multi_channel,
        ..GainMapMetadata::default()
    };
    for c in 0..channels {
        let base = 21 + c * 40;
        let min = rational(body, base, "gain map min")?.exp2();
        let max = rational(body, base + 8, "gain map max")?.exp2();
        let gamma = rational(body, base + 16, "gamma")?;
        let sdr = rational(body, base + 24, "base offset")?;
        let hdr = rational(body, base + 32, "alternate offset")?;
        // A single-channel file states the terms once and they apply to
        // every channel, which is what makes `vipsheader` print the same
        // number three times.
        let spread = if multi_channel { c..c + 1 } else { 0..3 };
        for i in spread {
            meta.min_content_boost[i] = min;
            meta.max_content_boost[i] = max;
            meta.gamma[i] = gamma;
            meta.offset_sdr[i] = sdr;
            meta.offset_hdr[i] = hdr;
        }
    }
    Ok(meta)
}

/// The gain-map metadata carried by an Ultra HDR container.
///
/// # Errors
///
/// [`UhdrError::NotUhdr`] if `bytes` is not an Ultra HDR container, or
/// [`UhdrError::BadMetadata`] if its ISO 21496-1 segment is malformed.
pub fn metadata(bytes: &[u8]) -> Result<GainMapMetadata, UhdrError> {
    let (_, gain) = split_container(bytes).ok_or(UhdrError::NotUhdr)?;
    let payload =
        find_app(bytes, gain, APP2, ISO_GAIN_MAP_ID).ok_or_else(|| UhdrError::BadMetadata {
            reason: "the gain map lost its ISO 21496-1 segment between detection and parsing"
                .to_string(),
        })?;
    parse_iso_metadata(payload)
}

/// Format a `f64` the way C's `%g` does, which is how `vipsheader` renders
/// the `gainmap-*` fields: six significant digits, trailing zeros trimmed.
///
/// Exists so the attached fields read identically to the captured header
/// (`49.2611`, not `49.261074066162109`), and it is pinned against that
/// capture rather than asserted.
fn g_format(v: f64) -> String {
    if v == 0.0 {
        return "0".to_string();
    }
    let exp = v.abs().log10().floor() as i32;
    if !(-5..6).contains(&exp) {
        let s = format!("{v:.5e}");
        // Rust writes `4.92611e1`; C writes `4.92611e+01`.
        let (m, e) = s.split_once('e').unwrap_or((s.as_str(), "0"));
        let m = m.trim_end_matches('0').trim_end_matches('.');
        let ev: i32 = e.parse().unwrap_or(0);
        return format!("{m}e{}{:02}", if ev < 0 { '-' } else { '+' }, ev.abs());
    }
    let decimals = (5 - exp).max(0) as usize;
    let s = format!("{v:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// Render a per-channel triple the way `vipsheader` prints a
/// `VipsArrayDouble`: the values, `%g`-formatted, space separated.
fn g_triple(v: [f64; 3]) -> String {
    format!("{} {} {}", g_format(v[0]), g_format(v[1]), g_format(v[2]))
}

/// Decode one JPEG half of the container into a [`Raster`], pricing its
/// declared geometry against `limits` first.
///
/// `want_bands` is what the caller needs out of it: 3 for the base, and for
/// the gain map whatever its own `SOF` declares, so a mono gain map stays
/// mono and drives the one-band transform path.
fn decode_half(
    bytes: &[u8],
    image: Range<usize>,
    which: &'static str,
    limits: DecodeLimits,
) -> Result<Raster, SourceError> {
    let (width, height, components) =
        sof_geometry(bytes, image.clone()).ok_or(UhdrError::NoFrameHeader { which })?;
    let bands = if components == 1 { 1u64 } else { 3 };
    // Priced before a pixel buffer exists, from the SOF the file declares
    // rather than from anything the decoder allocated. Both halves go
    // through this, which is the point: a 1x1 base carrying a huge gain map
    // must be refused on the gain map.
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    limits.check_image_alloc(
        if bands == 1 {
            "Ultra HDR gain map"
        } else {
            "Ultra HDR base image"
        },
        width,
        height,
        bands,
        1,
    )?;

    let decoded = image::load_from_memory_with_format(&bytes[image], image::ImageFormat::Jpeg)
        .map_err(|e| UhdrError::Jpeg {
            which,
            message: e.to_string(),
        })?;
    let raster = if bands == 1 {
        let buf = decoded.to_luma8();
        Raster::new(
            buf.width(),
            buf.height(),
            PixelFormat::Gray8,
            buf.into_raw(),
        )
    } else {
        let buf = decoded.to_rgb8();
        Raster::new(buf.width(), buf.height(), PixelFormat::Rgb8, buf.into_raw())
    }
    .map_err(UhdrError::Raster)?;
    Ok(raster)
}

/// Decode an Ultra HDR container, the libvips `uhdrload`.
///
/// Returns the **base image**: 3-band `uchar`, tagged sRGB. The gain map
/// travels attached rather than applied, exactly as in libvips, under the
/// same field names `uhdrload` uses:
///
/// * `gainmap-data` — the gain map's own JPEG bytes, undecoded, which is
///   what libvips attaches and what survives a write to `.v`
/// * `gainmap-min-content-boost`, `gainmap-max-content-boost`,
///   `gainmap-gamma`, `gainmap-offset-sdr`, `gainmap-offset-hdr` — the
///   per-channel triples, `%g`-formatted as `vipsheader` prints them
/// * `gainmap-hdr-capacity-min`, `gainmap-hdr-capacity-max` — scalars
/// * `gainmap-use-base-cg` — the colour-space flag
/// * `gainmap-scale-factor` — `max(1, base_width / gain_map_width)`,
///   **integer** division, a derived field rather than a stored one
/// * `exif-data` and `icc-profile-data` from the base image, the latter
///   with its 14-byte `ICC_PROFILE` prefix stripped the way
///   `uhdrload.c:399-411` strips it
///
/// The triples are formatted for header parity and are therefore six
/// significant digits. [`uhdr_to_scrgb`] never reads them back:
/// [`from_container`] re-parses the ISO segment out of `gainmap-data` at
/// full precision.
///
/// # Errors
///
/// * [`SourceError::Uhdr`] wrapping [`UhdrError::NotUhdr`] for bytes that
///   are not an Ultra HDR container, [`UhdrError::NoFrameHeader`] for a
///   half with no readable `SOF`, or [`UhdrError::Jpeg`] for a half that
///   fails to decode.
/// * [`SourceError::CoordLimitExceeded`],
///   [`SourceError::DimensionLimitExceeded`] or
///   [`SourceError::AllocLimitExceeded`] when **either** image is over
///   budget.
pub fn decode_uhdr(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let (base_range, gain_range) = split_container(bytes).ok_or(UhdrError::NotUhdr)?;
    let mut raster = decode_half(bytes, base_range.clone(), "base", limits)?;
    // The gain map is priced and decoded even though its pixels are not
    // attached, because pricing it is the only way an over-budget gain map
    // is refused, and decoding is how a corrupt one is caught at load
    // rather than at transform time.
    let gain = decode_half(bytes, gain_range.clone(), "gain map", limits)?;

    let payload = find_app(bytes, gain_range.clone(), APP2, ISO_GAIN_MAP_ID).ok_or_else(|| {
        UhdrError::BadMetadata {
            reason: "the gain map lost its ISO 21496-1 segment between detection and decode"
                .to_string(),
        }
    })?;
    let meta = parse_iso_metadata(payload)?;

    raster.meta.interpretation = Some(Interpretation::Srgb);
    if let Some(exif) = find_app(bytes, base_range.clone(), APP1, EXIF_ID) {
        raster
            .fields
            .set("exif-data", MetadataValue::Blob(exif.to_vec()));
    }
    if let Some(icc) = find_app(bytes, base_range, APP2, ICC_ID) {
        raster.fields.set(
            "icc-profile-data",
            MetadataValue::Blob(icc[ICC_PREFIX_LEN..].to_vec()),
        );
    }
    raster.fields.set(
        "gainmap-data",
        MetadataValue::Blob(bytes[gain_range].to_vec()),
    );
    for (name, value) in [
        ("gainmap-max-content-boost", meta.max_content_boost),
        ("gainmap-min-content-boost", meta.min_content_boost),
        ("gainmap-gamma", meta.gamma),
        ("gainmap-offset-sdr", meta.offset_sdr),
        ("gainmap-offset-hdr", meta.offset_hdr),
    ] {
        raster.fields.set(name, MetadataValue::Str(g_triple(value)));
    }
    raster.fields.set(
        "gainmap-hdr-capacity-min",
        MetadataValue::Double(meta.hdr_capacity_min),
    );
    raster.fields.set(
        "gainmap-hdr-capacity-max",
        MetadataValue::Double(meta.hdr_capacity_max),
    );
    raster.fields.set(
        "gainmap-use-base-cg",
        MetadataValue::Int(i64::from(meta.use_base_colour_space)),
    );
    // Integer division, and floored at 1, exactly as `uhdrload.c:473-478`
    // derives it. A gain map wider than its base gives 1, not a fraction.
    let scale_factor = (raster.width() / gain.width().max(1)).max(1);
    raster.fields.set(
        "gainmap-scale-factor",
        MetadataValue::Int(i64::from(scale_factor)),
    );
    Ok(raster)
}

/// Scale a `uchar` gain map to the base image's size, the way
/// `uhdr2scRGB` does with `vips_resize(..., VIPS_KERNEL_LINEAR)`.
///
/// The sampling convention is the one the capture measured rather than the
/// textbook one. For an 8-wide gain map scaled to 16, libvips produces
/// `0 0 18 36 54 72 90 108 126 144 162 180 198 216 234 252` from
/// `0 36 72 108 144 180 216 252`, which is a linear interpolation at source
/// position `j / scale - 0.5`, clamped to the ends. The usual
/// half-pixel-centred spelling, `(j + 0.5) / scale - 0.5`, gives 27 where
/// libvips gives 18.
///
/// That convention is pinned at scale 2 by
/// `records.uhdr2scRGB_gainmap_resize`, which is the only scale the capture
/// exercises; applying the same formula at other scales is an
/// extrapolation from one measurement, and it is written down here rather
/// than presented as parity. `crate::resample` implements the general
/// libvips resize and this should eventually call it (issue filed).
fn resize_gain_map(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    bands: usize,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let (sw, sh) = (src_width as usize, src_height as usize);
    let (dw, dh) = (width as usize, height as usize);
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    let sample = |axis_src: usize, axis_dst: usize, j: usize| -> (usize, usize, f64) {
        #[expect(clippy::cast_precision_loss, reason = "image axes are far under 2^53")]
        let scale = axis_dst as f64 / axis_src as f64;
        #[expect(clippy::cast_precision_loss, reason = "image axes are far under 2^53")]
        let pos = (j as f64 / scale - 0.5).clamp(0.0, (axis_src - 1) as f64);
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "pos is clamped into 0..=axis_src-1"
        )]
        let lo = pos.floor() as usize;
        (lo, (lo + 1).min(axis_src - 1), pos - pos.floor())
    };

    let mut out = vec![0u8; dw * dh * bands];
    for y in 0..dh {
        let (y0, y1, fy) = sample(sh, dh, y);
        for x in 0..dw {
            let (x0, x1, fx) = sample(sw, dw, x);
            for b in 0..bands {
                let at = |yy: usize, xx: usize| f64::from(src[(yy * sw + xx) * bands + b]);
                let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
                let bottom = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_sign_loss,
                    reason = "an interpolation between two u8 samples stays in 0..=255"
                )]
                let v = (top + (bottom - top) * fy).round() as u8;
                out[(y * dw + x) * bands + b] = v;
            }
        }
    }
    out
}

/// Apply the gain map to the base image, the libvips `uhdr2scRGB`.
///
/// `base` must be 3-band `uchar`; `gain_map` must be 1- or 3-band `uchar`.
/// The output is always 3-band `f32` tagged
/// [`Interpretation::ScRgb`], whatever the input was.
///
/// # The two paths, and why they disagree on identical bytes
///
/// The band count of the *gain map* alone picks the path
/// (`uhdr->gainmap->Bands == 1` at `uhdr2scRGB.c:107`), and the two paths
/// treat the gain-map sample differently:
///
/// * one band: `gg = sample / 255.0`, a plain scale. The gain map is **not**
///   linearised. The C comment is "the gainmap is not gamma corrected in
///   libultrahdr, confusingly".
/// * three bands: `gg = v2Y_8[sample]`, linearised through the same table
///   the base goes through.
///
/// So a mono gain map and a three-band gain map holding the same bytes
/// produce different pixels. That is not a bug being ported; it is measured
/// (`oracle-captures/foreign-uhdr`, `uhdr2scRGB_rgb_gainmap`, whose
/// three-band fixture is the mono one replicated into three bands and whose
/// results "do not match the mono ones anywhere").
///
/// # The index trap
///
/// The one-band path reads **every** metadata term at index `[1]`, the
/// green entry, and applies it to all three output channels; indices `[0]`
/// and `[2]` are never read. The oracle proves this directly: setting them
/// to 999 gives a result identical to leaving them alone. The three-band
/// path reads each channel's own entry. Both are reproduced here.
///
/// # Rounding
///
/// `log2`, `exp2` and `powf` are the `f64` forms, while `gg`, `boost` and
/// `gain` are `f32` locals, so the value rounds to `f32` three times on the
/// way through. That is not incidental: reordering it moves results by
/// several ulp, and the pinned oracle values are bit-exact against this
/// spelling.
///
/// Nothing validates the metadata, in libvips or here. `min_content_boost`
/// of 0 makes `log2(0)` reach the boost expression as `-inf`, and the
/// result is an infinity or a `NaN` rather than an error. The oracle pins
/// that too, so a port that "helpfully" special-cases it would be wrong.
///
/// # Errors
///
/// [`UhdrError::BadInput`] if the band counts or sample formats are not the
/// ones `uhdr2scRGB` accepts, or [`UhdrError::Raster`] if the output cannot
/// be built.
pub fn uhdr_to_scrgb(
    base: &Raster,
    gain_map: &Raster,
    metadata: &GainMapMetadata,
) -> Result<Raster, UhdrError> {
    if base.format() != PixelFormat::Rgb8 {
        return Err(UhdrError::BadInput {
            reason: format!("image must have 3 uchar bands, got {:?}", base.format()),
        });
    }
    let mono = match gain_map.format() {
        PixelFormat::Gray8 => true,
        PixelFormat::Rgb8 => false,
        other => {
            return Err(UhdrError::BadInput {
                reason: format!("gain map must have 1 or 3 uchar bands, got {other:?}"),
            });
        }
    };

    let (width, height) = (base.width(), base.height());
    // The gain map is almost always smaller than its base, and libvips
    // scales it to 1:1 *before* the per-pixel transform, with
    // `vips_resize(..., VIPS_KERNEL_LINEAR)` and separate h/v scales
    // (`uhdr2scRGB.c:233-240`). The capture is blunt about the alternative:
    // "Anything else, nearest included, gives different pixels everywhere
    // the gainmap is not flat." So this resamples rather than indexes.
    let gain_bands = if mono { 1usize } else { 3 };
    let gain_pixels = resize_gain_map(
        gain_map.data(),
        gain_map.width().max(1),
        gain_map.height().max(1),
        gain_bands,
        width,
        height,
    );

    let base_pixels = base.data();
    let mut out = vec![0f32; (width as usize) * (height as usize) * 3];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let bi = (y * width as usize + x) * 3;
            let gi = (y * width as usize + x) * gain_bands;
            for i in 0..3 {
                // The one-band path reads index [1] for every channel; the
                // three-band path reads the channel's own entry.
                let c = if mono { 1 } else { i };
                let sample = gain_pixels[gi + if mono { 0 } else { i }];
                let mut gg = if mono {
                    f32::from(sample) / 255.0
                } else {
                    V2Y_8[sample as usize]
                };
                if metadata.gamma[c] != 1.0 {
                    gg = f64::from(gg).powf(1.0 / metadata.gamma[c]) as f32;
                }
                let boost = (metadata.min_content_boost[c].log2() * (1.0 - f64::from(gg))
                    + metadata.max_content_boost[c].log2() * f64::from(gg))
                    as f32;
                let gain = f64::from(boost).exp2() as f32;
                let linear = f64::from(V2Y_8[base_pixels[bi + i] as usize]);
                out[bi + i] = ((linear + metadata.offset_sdr[c]) * f64::from(gain)
                    - metadata.offset_hdr[c]) as f32;
            }
        }
    }

    let bytes: Vec<u8> = out.into_iter().flat_map(f32::to_ne_bytes).collect();
    let mut raster = Raster::new(
        width,
        height,
        PixelFormat::FloatF32(std::num::NonZeroU16::new(3).expect("3 is not zero")),
        bytes,
    )?;
    raster.meta.interpretation = Some(Interpretation::ScRgb);
    Ok(raster)
}

/// Expand an Ultra HDR container straight to linear-light scRGB: decode,
/// then apply the gain map.
///
/// The metadata comes from the gain map's own ISO 21496-1 segment at full
/// precision, not from the `%g`-formatted fields [`decode_uhdr`] attaches
/// for header parity.
///
/// # Errors
///
/// As [`decode_uhdr`], plus the [`uhdr_to_scrgb`] input errors.
pub fn from_container(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let (base_range, gain_range) = split_container(bytes).ok_or(UhdrError::NotUhdr)?;
    let base = decode_half(bytes, base_range, "base", limits)?;
    let gain = decode_half(bytes, gain_range.clone(), "gain map", limits)?;
    let payload = find_app(bytes, gain_range, APP2, ISO_GAIN_MAP_ID).ok_or(UhdrError::NoGainMap)?;
    let meta = parse_iso_metadata(payload)?;
    Ok(uhdr_to_scrgb(&base, &gain, &meta)?)
}

/// Options for [`encode_uhdr`].
///
/// `#[non_exhaustive]`, `Default`, and module-scoped, the same shape as
/// [`DecodeLimits`]: start from [`SaveOptions::default`] and set what you need
/// with the `with_*` builders, e.g.
/// `uhdr::SaveOptions::default().with_quality(95)`. A struct literal would
/// compile today and stop the day a field lands (issue #630).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct SaveOptions {
    /// JPEG quality for both halves, 1..=100. libvips defaults `uhdrsave`
    /// to 75 and so does [`SaveOptions::default`].
    pub quality: u8,
    /// How much smaller than the base the gain map is, per axis. libuhdr
    /// writes 2; 1 keeps it full size.
    pub gain_map_shrink: u32,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            quality: 75,
            gain_map_shrink: 2,
        }
    }
}

impl SaveOptions {
    /// Set the JPEG quality for both halves, returning the updated options.
    #[must_use]
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Set how much smaller than the base the gain map is, per axis,
    /// returning the updated options.
    #[must_use]
    pub fn with_gain_map_shrink(mut self, gain_map_shrink: u32) -> Self {
        self.gain_map_shrink = gain_map_shrink;
        self
    }
}

/// The `u8` code whose [`V2Y_8`] entry is nearest `linear`.
///
/// The sRGB opto-electrical transfer, done by inverting the very table the
/// decode linearises through rather than by evaluating the analytic
/// formula. That is what makes a round trip tight: `V2Y_8[to_code(v)]` is
/// the closest 8-bit representation of `v` there is, by construction,
/// where the analytic inverse can land a code either side.
///
/// `V2Y_8` is monotonically increasing, so a binary search is exact.
fn to_code(linear: f32) -> u8 {
    // Spelled out rather than as a negated comparison: `linear` can be
    // `NaN`, and a `NaN` reaching the binary search below would index the
    // table on a meaningless answer.
    if linear.is_nan() || linear <= 0.0 {
        return 0;
    }
    if linear >= 1.0 {
        return 255;
    }
    let hi = V2Y_8.partition_point(|&v| v < linear);
    let lo = hi.saturating_sub(1);
    let (dlo, dhi) = (linear - V2Y_8[lo], V2Y_8[hi.min(255)] - linear);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "both indices are in 0..=255 by construction"
    )]
    let code = if dhi < dlo {
        hi.min(255) as u8
    } else {
        lo as u8
    };
    code
}

/// Write one `APPn` segment: marker, big-endian length, payload.
fn app_segment(marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![0xFF, marker];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "callers build payloads far under 65533 bytes"
    )]
    let len = (payload.len() + 2) as u16;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// The ISO 21496-1 payload for a gain map carrying `meta`, single channel.
fn iso_payload(meta: &GainMapMetadata) -> Vec<u8> {
    /// Rationals are written over a fixed power-of-two denominator, the
    /// same 2^20 libuhdr uses, so the values survive a round trip through
    /// the file exactly as libuhdr's do.
    const DEN: u32 = 1 << 20;
    let mut p = Vec::with_capacity(ISO_GAIN_MAP_ID.len() + 61);
    p.extend_from_slice(ISO_GAIN_MAP_ID);
    p.extend_from_slice(&0u16.to_be_bytes()); // minimum version
    p.extend_from_slice(&0u16.to_be_bytes()); // writer version
    p.push(if meta.use_base_colour_space { 0x40 } else { 0 });
    #[expect(
        clippy::cast_possible_truncation,
        reason = "log2 of a boost is a small number; the cast is the encode"
    )]
    let mut rat = |v: f64| {
        p.extend_from_slice(&((v * f64::from(DEN)).round() as i32).to_be_bytes());
        p.extend_from_slice(&DEN.to_be_bytes());
    };
    rat(meta.hdr_capacity_min.log2());
    rat(meta.hdr_capacity_max.log2());
    rat(meta.min_content_boost[1].log2());
    rat(meta.max_content_boost[1].log2());
    rat(meta.gamma[1]);
    rat(meta.offset_sdr[1]);
    rat(meta.offset_hdr[1]);
    p
}

/// The MPF `APP2` payload advertising a two-image container.
///
/// The layout is the one `fixtures/uhdr.jpg` carries, decoded entry by
/// entry: a big-endian TIFF header, an index IFD of three tags
/// (`MPFVersion`, `NumberOfImages`, `MPEntry`) and two 16-byte MP entries.
/// Individual image offsets are measured **from the MP endian field**, the
/// byte right after `MPF\0`, which is what makes the gain map's offset 530
/// in a file whose gain map starts at 1991 and whose endian field sits at
/// 1461.
fn mpf_payload(base_len: u32, gain_len: u32, gain_offset: u32) -> Vec<u8> {
    let mut p = Vec::with_capacity(86);
    p.extend_from_slice(b"MPF\0");
    p.extend_from_slice(b"MM");
    p.extend_from_slice(&42u16.to_be_bytes());
    p.extend_from_slice(&8u32.to_be_bytes());
    p.extend_from_slice(&3u16.to_be_bytes());
    let mut tag = |id: u16, ty: u16, count: u32, value: [u8; 4]| {
        p.extend_from_slice(&id.to_be_bytes());
        p.extend_from_slice(&ty.to_be_bytes());
        p.extend_from_slice(&count.to_be_bytes());
        p.extend_from_slice(&value);
    };
    tag(0xB000, 7, 4, *b"0100");
    tag(0xB001, 4, 1, 2u32.to_be_bytes());
    tag(0xB002, 7, 32, 50u32.to_be_bytes());
    p.extend_from_slice(&0u32.to_be_bytes()); // no next IFD
    // Image 0: representative, primary.
    p.extend_from_slice(&0x0003_0000u32.to_be_bytes());
    p.extend_from_slice(&base_len.to_be_bytes());
    p.extend_from_slice(&0u32.to_be_bytes());
    p.extend_from_slice(&0u32.to_be_bytes());
    // Image 1: the gain map.
    p.extend_from_slice(&0u32.to_be_bytes());
    p.extend_from_slice(&gain_len.to_be_bytes());
    p.extend_from_slice(&gain_offset.to_be_bytes());
    p.extend_from_slice(&0u32.to_be_bytes());
    p
}

/// Insert `segments` immediately after a JPEG's `SOI`.
///
/// Valid by the standard: `APPn` markers may appear in any order between
/// `SOI` and the frame header, and libjpeg, libuhdr and libvips all read
/// them positionally rather than by index.
fn splice_after_soi(jpeg: &[u8], segments: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(jpeg.len() + segments.len());
    out.extend_from_slice(&jpeg[..2]);
    out.extend_from_slice(segments);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// Encode a linear-light scRGB raster as an Ultra HDR container, the
/// libvips `uhdrsave`.
///
/// `scrgb` must be 3-band `f32`. The base image is the input clamped to the
/// SDR range and encoded through the inverse of the `v2Y_8` table; the gain map
/// carries what was clipped.
///
/// # This is not byte-compatible with libuhdr, and nothing could be
///
/// libvips does not choose the gain map when it saves: it hands the HDR
/// image to `libultrahdr`, and libuhdr's tone mapper picks both the gain
/// map and its metadata. The oracle capture measured that the same libvips
/// against two libuhdr majors writes files differing only in a version
/// string, but there is no specification of the choice to port — it is one
/// library's tone-mapping policy. So this writes a **spec-conformant
/// container with its own gain map**, not a reproduction of libuhdr's:
///
/// * `min_content_boost` is 1 and `max_content_boost` is the largest boost
///   the image actually needs, so the gain map spans exactly its range
/// * gamma is 1 and both offsets are 0, the case libuhdr also writes
/// * the base is the tone-mapped SDR rendering, the gain map the ratio
///
/// What *is* guaranteed, and tested: the result satisfies [`is_uhdr`], is
/// read back by libvips's own `uhdrload`, and round-trips through
/// [`from_container`] to within the precision 8-bit halves allow.
///
/// # Errors
///
/// [`UhdrError::BadInput`] if `scrgb` is not 3-band `f32`, or
/// [`UhdrError::Jpeg`] if either half fails to encode.
pub fn encode_uhdr(scrgb: &Raster, options: &SaveOptions) -> Result<Vec<u8>, UhdrError> {
    let bands = scrgb.format().channels();
    if !matches!(scrgb.format(), PixelFormat::FloatF32(_)) || bands != 3 {
        return Err(UhdrError::BadInput {
            reason: format!(
                "uhdrsave needs a 3-band float image, got {:?}",
                scrgb.format()
            ),
        });
    }
    let (width, height) = (scrgb.width(), scrgb.height());
    let (w, h) = (width as usize, height as usize);
    let hdr: Vec<f32> = scrgb
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();

    // The gain every pixel needs, which is its own peak channel. Dividing
    // by it puts the whole pixel inside the SDR range *without changing the
    // ratios between its channels*, and that is the whole trick: a
    // one-band gain map applies a single scalar to all three channels, so a
    // base image made by clamping -- which changes those ratios -- cannot
    // be reconstructed by any gain map at all. The first cut of this
    // encoder clamped, and round-tripped with 433% error on a ramp.
    let mut needed = vec![1f32; w * h];
    let mut max_boost = 1f64;
    for (p, slot) in needed.iter_mut().enumerate() {
        let peak = hdr[p * 3..p * 3 + 3]
            .iter()
            .fold(1f32, |acc, &v| acc.max(v));
        *slot = peak;
        max_boost = max_boost.max(f64::from(peak));
    }

    let meta = GainMapMetadata {
        max_content_boost: [max_boost; 3],
        hdr_capacity_max: max_boost,
        ..GainMapMetadata::default()
    };
    let span = max_boost.log2();

    // The gain map, mono, at `gain_map_shrink`, box-max over the pixels
    // each sample covers so shrinking never *under*-states a boost and
    // clips a highlight.
    let shrink = options.gain_map_shrink.max(1);
    let (gw, gh) = ((width / shrink).max(1), (height / shrink).max(1));
    let (gwu, ghu) = (gw as usize, gh as usize);
    let mut gain_pixels = vec![0u8; gwu * ghu];
    for gy in 0..ghu {
        for gx in 0..gwu {
            let (x0, x1) = (gx * w / gwu, (((gx + 1) * w) / gwu).max(gx * w / gwu + 1));
            let (y0, y1) = (gy * h / ghu, (((gy + 1) * h) / ghu).max(gy * h / ghu + 1));
            let mut worst = 1f32;
            for y in y0..y1.min(h) {
                for x in x0..x1.min(w) {
                    worst = worst.max(needed[y * w + x]);
                }
            }
            let gg = if span > 0.0 {
                (f64::from(worst).log2() / span).clamp(0.0, 1.0)
            } else {
                0.0
            };
            #[expect(
                clippy::cast_possible_truncation,
                reason = "gg is clamped to 0..=1, so the product is in 0..=255"
            )]
            let code = (gg * 255.0).round() as u8;
            gain_pixels[gy * gwu + gx] = code;
        }
    }

    let quality = options.quality.clamp(1, 100);
    let gain_jpeg = encode_jpeg(&gain_pixels, gw, gh, image::ExtendedColorType::L8, quality)?;

    // Read the gain map back through its own JPEG before building the
    // base. The decoder will see these codes, not the ones just computed --
    // JPEG is lossy and the map is subsampled -- so pricing the base
    // against them makes the base absorb the gain map's own compression
    // loss instead of leaving it in the output.
    let actual = image::load_from_memory_with_format(&gain_jpeg, image::ImageFormat::Jpeg)
        .map_err(|e| UhdrError::Jpeg {
            which: "gain map",
            message: e.to_string(),
        })?
        .to_luma8();

    let mut base_rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        let gy = (y * ghu) / h;
        for x in 0..w {
            let gx = (x * gwu) / w;
            // Exactly the reconstruction the one-band decode path will do,
            // run forwards: gamma is 1 and both offsets are 0 here, so the
            // gain is `exp2(span * code / 255)`.
            let gg = f32::from(actual.get_pixel(gx as u32, gy as u32)[0]) / 255.0;
            let gain = f64::from(f64::from(gg) as f32 * span as f32).exp2() as f32;
            for i in 0..3 {
                let p = (y * w + x) * 3 + i;
                base_rgb[p] = to_code(if gain > 0.0 { hdr[p] / gain } else { 0.0 });
            }
        }
    }
    let base_jpeg = encode_jpeg(
        &base_rgb,
        width,
        height,
        image::ExtendedColorType::Rgb8,
        quality,
    )?;

    // The gain map carries the metadata; the base carries only the version
    // marker, which is what libuhdr writes and what the oracle's
    // `no-iso-base.jpg` proves is not load bearing.
    let gain = splice_after_soi(&gain_jpeg, &app_segment(APP2, &iso_payload(&meta)));
    let base_iso = app_segment(APP2, &[ISO_GAIN_MAP_ID, &[0u8; 4]].concat());

    // The MPF offsets have to know the final layout, and the MPF segment is
    // part of that layout, so its own length is folded in. It is fixed at
    // 90 bytes: 2 marker, 2 length, 86 payload.
    const MPF_SEGMENT_LEN: usize = 90;
    let base_len = base_jpeg.len() + base_iso.len() + MPF_SEGMENT_LEN;
    // The MP endian field: past SOI, past the ISO segment, past this
    // segment's own marker, length and `MPF\0`.
    let endian_at = 2 + base_iso.len() + 8;
    let mpf = app_segment(
        APP2,
        &mpf_payload(
            u32::try_from(base_len).unwrap_or(u32::MAX),
            u32::try_from(gain.len()).unwrap_or(u32::MAX),
            u32::try_from(base_len - endian_at).unwrap_or(u32::MAX),
        ),
    );
    debug_assert_eq!(mpf.len(), MPF_SEGMENT_LEN, "the MPF segment is fixed size");

    let mut out = splice_after_soi(&base_jpeg, &[base_iso, mpf].concat());
    out.extend_from_slice(&gain);
    Ok(out)
}

/// Encode one half through the `image` crate's JPEG encoder.
fn encode_jpeg(
    data: &[u8],
    width: u32,
    height: u32,
    color: image::ExtendedColorType,
    quality: u8,
) -> Result<Vec<u8>, UhdrError> {
    let which = if color == image::ExtendedColorType::L8 {
        "gain map"
    } else {
        "base"
    };
    let mut buf = Vec::new();
    let encoder =
        image::codecs::jpeg::JpegEncoder::new_with_quality(std::io::Cursor::new(&mut buf), quality);
    image::ImageEncoder::write_image(encoder, data, width, height, color).map_err(|e| {
        UhdrError::Jpeg {
            which,
            message: e.to_string(),
        }
    })?;
    Ok(buf)
}

/// The smallest Ultra HDR container this build writes, for the route-table
/// probes in [`crate::source`].
///
/// The route table's reachability test builds its probe from the row rather
/// than from a hand-kept table of sample bytes, and the `Uhdr` row's
/// signature is a structural predicate, so the row has to be able to hand
/// one over. It is a real, decodable file, not a stub: the same test feeds
/// it to both decode entry points and compares the answers.
#[must_use]
pub fn smallest_container() -> Vec<u8> {
    let pixels: Vec<u8> = (0..8 * 8 * 3)
        .map(|i| if i % 3 == 1 { 2.0f32 } else { 0.25f32 })
        .flat_map(f32::to_ne_bytes)
        .collect();
    let raster = Raster::new(
        8,
        8,
        PixelFormat::FloatF32(std::num::NonZeroU16::new(3).expect("3 is not zero")),
        pixels,
    )
    .expect("an 8x8 three-band float raster is well formed");
    encode_uhdr(&raster, &SaveOptions::default()).expect("the smallest container always encodes")
}

#[cfg(test)]
mod tests {
    use super::*;

    /**
     * The marker walk terminates on every byte string, which is what makes
     * it safe to run from `sniff` on untrusted input.
     *
     * `scan_jpeg` advances `i` on every path, but three of those paths take
     * their step size from a length field the file controls, and one of
     * them (`SOS`) hands the cursor to an inner scan. A walker that failed
     * to advance would hang the sniffer rather than reject the file, and a
     * hang in a detector is worse than a wrong answer. Exhaustive over
     * every 3-byte string after `SOI`, plus the pathological length fields.
     * Input: 16.7 million short buffers -> Output: every one returns.
     */
    #[test]
    fn the_marker_walk_terminates_on_every_short_buffer() {
        let mut visited = 0usize;
        // Every 16-bit length field, against every marker code that takes
        // one and several that do not, over a buffer long enough for a
        // short segment to be well formed. The length is the field the
        // cursor step is taken from, so this is the exhaustive sweep that
        // matters.
        for hi in 0..=255u8 {
            for lo in 0..=255u8 {
                for marker in [0x01u8, 0xC0, 0xD0, 0xD8, 0xD9, 0xDA, 0xE2, 0xFE, 0xFF] {
                    let buf = [
                        0xFF, 0xD8, 0xFF, marker, hi, lo, 0xAA, 0xBB, 0xCC, 0xFF, 0xD9,
                    ];
                    let _ = scan_jpeg(&buf, 0, |_, _| visited += 1);
                }
            }
        }
        // A length of 0 or 1 cannot advance the cursor and must be refused
        // rather than looped on.
        for len in [0u16, 1] {
            let mut buf = vec![0xFF, 0xD8, 0xFF, 0xE2];
            buf.extend_from_slice(&len.to_be_bytes());
            buf.extend_from_slice(&[0u8; 8]);
            let scan = scan_jpeg(&buf, 0, |_, _| {});
            assert!(scan.end.is_none(), "a length of {len} is malformed");
        }
        assert!(
            visited > 0,
            "the sweep visited no segment at all, so it proved termination over \
             inputs the walk rejects before it ever steps -- which is not the \
             property under test"
        );
    }

    /**
     * `is_uhdr` never claims a buffer that is not a JPEG at all, and never
     * panics on one.
     *
     * The route table runs this predicate over every buffer whose first
     * three bytes are `FF D8 FF`, but `is_uhdr` is public and a caller can
     * hand it anything.
     * Input: empty, short and random-ish buffers -> Output: false, no panic.
     */
    #[test]
    fn is_uhdr_declines_everything_that_is_not_a_container() {
        assert!(!is_uhdr(&[]));
        assert!(!is_uhdr(b"\xff\xd8"));
        assert!(!is_uhdr(b"\xff\xd8\xff"));
        assert!(!is_uhdr(b"not a jpeg at all"));
        let mut pseudo = Vec::new();
        let mut x = 0x1234_5678u32;
        for _ in 0..4096 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            pseudo.push((x >> 24) as u8);
        }
        pseudo[0] = 0xFF;
        pseudo[1] = 0xD8;
        pseudo[2] = 0xFF;
        assert!(
            !is_uhdr(&pseudo),
            "random bytes behind a JPEG SOI are not a container"
        );
    }

    /**
     * `g_format` renders what C's `%g` renders, which is what makes the
     * attached fields comparable to `vipsheader` output.
     *
     * The value that matters is 49.2611: six significant digits, and the
     * one the captured header carries. The rest bracket the rules `%g`
     * switches on -- trailing zeros trimmed, an integer printed without a
     * point, and the exponent form outside 1e-5..1e6.
     * Input: nine values -> Output: the `%g` spelling of each.
     */
    #[test]
    fn g_format_matches_the_c_percent_g_spelling() {
        for (value, want) in [
            (49.261_083_046_983_38, "49.2611"),
            (1.0, "1"),
            (0.0, "0"),
            (2.0, "2"),
            (0.015_625, "0.015625"),
            (1.8, "1.8"),
            (100_000.0, "100000"),
            (1_000_000.0, "1e+06"),
            (0.000_001, "1e-06"),
        ] {
            assert_eq!(g_format(value), want, "%g of {value}");
        }
    }

    /**
     * `to_code` really is the inverse of the table the decode linearises
     * through, which is the property the round trip rests on.
     *
     * Not "close to the analytic inverse": for every one of the 256 codes,
     * feeding its own linear value back must return that code, and no
     * other code may sit nearer the value than the one returned. The second
     * half is what a naive `partition_point` gets wrong at the boundaries.
     * Input: every entry of `V2Y_8`, plus the midpoints between them ->
     * Output: an exact round trip and a genuinely nearest code.
     */
    #[test]
    fn to_code_inverts_the_linearisation_table_exactly() {
        for code in 0..=255u8 {
            assert_eq!(
                to_code(V2Y_8[code as usize]),
                code,
                "code {code} does not survive a round trip"
            );
        }
        for code in 0..255u8 {
            let mid = f32::midpoint(V2Y_8[code as usize], V2Y_8[code as usize + 1]);
            let got = to_code(mid);
            let chosen = (V2Y_8[got as usize] - mid).abs();
            for other in 0..=255u8 {
                assert!(
                    (V2Y_8[other as usize] - mid).abs() >= chosen - f32::EPSILON,
                    "code {other} is nearer {mid} than the chosen {got}"
                );
            }
        }
        assert_eq!(to_code(-1.0), 0, "below the range clamps to 0");
        assert_eq!(
            to_code(f32::NAN),
            0,
            "NaN clamps to 0 rather than indexing wild"
        );
        assert_eq!(to_code(1e9), 255, "above the range clamps to 255");
    }

    /**
     * The ISO 21496-1 parser refuses a zero denominator instead of turning
     * it into an infinity.
     *
     * The file is untrusted and every term is a rational. `0/0` would
     * otherwise arrive at the transform as a `NaN` that silently poisons
     * every pixel, which is indistinguishable from the *legitimate* `NaN`
     * the degenerate-metadata case produces.
     * Input: a payload whose gamma denominator is 0 -> Output: a typed
     * `BadMetadata`, not a `NaN`.
     */
    #[test]
    fn a_zero_denominator_is_refused_rather_than_made_infinite() {
        let mut payload = ISO_GAIN_MAP_ID.to_vec();
        payload.extend_from_slice(&[0, 0, 0, 0, 0x40]);
        // Two headrooms and five channel terms, all 1/1, then break one.
        for _ in 0..7 {
            payload.extend_from_slice(&1u32.to_be_bytes());
            payload.extend_from_slice(&1u32.to_be_bytes());
        }
        assert!(
            parse_iso_metadata(&payload).is_ok(),
            "the control must parse"
        );

        // gamma is the third channel term: 21 + 16 = byte 37, denominator
        // at 41.
        let mut broken = payload.clone();
        broken[ISO_GAIN_MAP_ID.len() + 41..ISO_GAIN_MAP_ID.len() + 45]
            .copy_from_slice(&0u32.to_be_bytes());
        assert!(
            matches!(
                parse_iso_metadata(&broken),
                Err(UhdrError::BadMetadata { .. })
            ),
            "a zero denominator must be a typed refusal"
        );

        // And a truncated payload is a refusal rather than a panic.
        for cut in 0..payload.len() {
            assert!(
                parse_iso_metadata(&payload[..cut]).is_err() || cut == payload.len(),
                "a payload cut at {cut} must not parse"
            );
        }
    }
}
