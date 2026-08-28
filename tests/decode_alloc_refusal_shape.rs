//! Issue #686: how many shapes a caller has to match to catch "the decode
//! allocation budget refused this file".
//!
//! Pinned from an integration test rather than a unit one, because the whole
//! point is the shape a *caller outside the crate* has to write.
//!
//! # What I measured before writing this
//!
//! Every container libviprs can decode, refused at
//! `max_alloc_bytes = price - 1`, on the tree this branched from:
//!
//! | container | what came back |
//! |---|---|
//! | `.v` | **nothing, it decoded** |
//! | JPEG | `SourceError::Decode(image Limits)` |
//! | PNG | `SourceError::Decode(image Limits)` |
//! | TIFF | `SourceError::Decode(image Limits)` |
//! | WebP | `SourceError::Decode(image Limits)` |
//! | GIF | `SourceError::Gif(GifError::AllocLimitExceeded)` |
//! | Radiance | `SourceError::Radiance(RadianceError::AllocLimitExceeded)` |
//! | FITS | `SourceError::Fits(FitsError::AllocLimitExceeded)` |
//! | OpenEXR | `SourceError::Exr(ExrError::AllocLimitExceeded)` |
//! | JPEG XL | `SourceError::Jxl(JxlError::AllocLimitExceeded)` |
//!
//! Two corrections to the table in #686, both from that run. WebP is **not**
//! the one odd format on the `image` shape: JPEG, PNG and single-image TIFF
//! report exactly the same thing, because in all four the refusal is the
//! `image` crate's own budget rather than a price libviprs computed. And the
//! `.v` reader never consults `max_alloc_bytes` at all.
//!
//! # What this file holds
//!
//! The five formats where **libviprs itself** prices a declared geometry and
//! refuses it must all report one shape, `SourceError::AllocLimitExceeded`.
//! The four where the `image` crate refuses keep reporting the `image` shape,
//! and that is asserted here too so the split is a decision on the record
//! rather than a gap.

use libviprs::source::{DecodeLimits, decode_bytes_with_limits};
use libviprs::{PixelFormat, Raster, SourceError};
use std::io::Cursor;
use std::num::NonZeroU16;

fn rgb8(dim: u32) -> Raster {
    let n = (dim * dim * 3) as usize;
    Raster::new(
        dim,
        dim,
        PixelFormat::Rgb8,
        (0..n).map(|v| v as u8).collect(),
    )
    .expect("rgb8 fixture")
}

fn gray8() -> Raster {
    Raster::new(4, 3, PixelFormat::Gray8, vec![7u8; 12]).expect("gray8 fixture")
}

fn float_rgb() -> Raster {
    let data: Vec<u8> = (0..36u32).flat_map(|v| (v as f32).to_ne_bytes()).collect();
    let fmt = PixelFormat::FloatF32(NonZeroU16::new(3).expect("three bands"));
    Raster::new(4, 3, fmt, data).expect("float rgb fixture")
}

fn tiff_bytes() -> Vec<u8> {
    let mut buf = Vec::new();
    let enc = image::codecs::tiff::TiffEncoder::new(Cursor::new(&mut buf));
    image::ImageEncoder::write_image(enc, &[9u8; 36], 4, 3, image::ColorType::Rgb8.into())
        .expect("tiff fixture");
    buf
}

/// The OpenEXR fixture the `exr` module's own budget tests use: 8x4, four
/// half channels, so `decode_exr` prices it at `8 * 4 * 4 * 4` = 512 bytes.
const EXR: &[u8] = include_bytes!("../oracle-captures/foreign-exr/fixtures/rgba_half_zip.exr");

/// One container, the bytes to decode, and the price `decode_*` computes for
/// its frame from the declared geometry.
struct Row {
    format: &'static str,
    bytes: Vec<u8>,
    /// The geometry a clean decode produces, which is what the row checks the
    /// budget-lifted control against.
    decoded: (u32, u32),
    /// `width`, `height`, `bands` as the decoder **prices** them, which is not
    /// always what the raster ends up holding: GIF prices a four-band RGBA
    /// canvas for a file whose palette is three bands, and OpenEXR prices
    /// every channel the header declares rather than the four the selection
    /// keeps.
    priced_geometry: (u32, u32, u32),
    /// Bytes per sample in that price.
    sample_bytes: u64,
    /// The `what` label the refusal names the buffer with, empty where the
    /// refusal is not libviprs's own.
    what: &'static str,
    /// `width * height * bands * sample_bytes`.
    price: u64,
}

/// The formats where libviprs prices the frame itself off a declared geometry.
///
/// The JPEG XL row is 16x16 rather than 4x4 on purpose. At 4x4 `jxl-oxide`'s
/// own `AllocTracker` refuses an internal buffer before the declared-geometry
/// check is reached, so the run reports `JxlError::DecoderAllocLimitExceeded`
/// and this row would be testing the wrong ceiling. Measured: 4x4 trips the
/// tracker at every budget below its price, 16x16 and above trip the
/// declared-geometry check at `price - 1` and only fall back to the tracker
/// at a budget of 1.
fn priced_by_libviprs() -> Vec<Row> {
    let mut rows = vec![
        Row {
            format: "gif",
            bytes: rgb8(4).encode_gif(Default::default()).expect("gif fixture"),
            decoded: (4, 4),
            // Four bands: the canvas the decoder allocates is RGBA whatever
            // the palette holds, and 4 * 4 * 4 = 64 is the number the price
            // below can only be explained by.
            priced_geometry: (4, 4, 4),
            sample_bytes: 1,
            what: "GIF canvas",
            price: 64,
        },
        Row {
            format: "radiance",
            bytes: float_rgb()
                .encode_radiance(Default::default())
                .expect("radiance fixture"),
            decoded: (4, 3),
            priced_geometry: (4, 3, 3),
            sample_bytes: 4,
            what: "Radiance pixel buffer",
            price: 144,
        },
        Row {
            format: "fits",
            bytes: gray8().encode_fits().expect("fits fixture"),
            decoded: (4, 3),
            priced_geometry: (4, 3, 1),
            sample_bytes: 1,
            what: "FITS pixel buffer",
            price: 12,
        },
        Row {
            format: "openexr",
            bytes: EXR.to_vec(),
            decoded: (8, 4),
            priced_geometry: (8, 4, 4),
            sample_bytes: 4,
            what: "OpenEXR sample buffers",
            price: 512,
        },
    ];
    if cfg!(feature = "jxl") {
        rows.push(Row {
            format: "jxl",
            bytes: rgb8(16)
                .encode_jxl(Default::default())
                .expect("jxl fixture"),
            decoded: (16, 16),
            priced_geometry: (16, 16, 3),
            sample_bytes: 1,
            what: "JPEG XL frame buffer",
            price: 768,
        });
    }
    rows
}

/// The formats where the `image` crate's own budget does the refusing.
fn priced_by_the_image_crate() -> Vec<Row> {
    vec![
        Row {
            format: "jpeg",
            bytes: rgb8(4).encode_jpeg(90).expect("jpeg fixture"),
            decoded: (4, 4),
            priced_geometry: (4, 4, 3),
            sample_bytes: 1,
            what: "",
            price: 48,
        },
        Row {
            format: "png",
            bytes: rgb8(4).encode_png(6).expect("png fixture"),
            decoded: (4, 4),
            priced_geometry: (4, 4, 3),
            sample_bytes: 1,
            what: "",
            price: 48,
        },
        Row {
            format: "tiff",
            bytes: tiff_bytes(),
            decoded: (4, 3),
            priced_geometry: (4, 3, 3),
            sample_bytes: 1,
            what: "",
            price: 36,
        },
        Row {
            format: "webp",
            bytes: rgb8(4)
                .encode_webp(Default::default())
                .expect("webp fixture"),
            decoded: (4, 4),
            priced_geometry: (4, 4, 3),
            sample_bytes: 1,
            what: "",
            price: 48,
        },
    ]
}

/// Decode `row` at one byte under its price, and at no ceiling at all.
///
/// The second half is not decoration. An assertion about *how* a decode was
/// refused proves nothing if the file was going to be refused anyway, so every
/// row has to decode cleanly with the budget lifted, at the geometry the row
/// claims.
fn refuse(row: &Row) -> SourceError {
    let (w, h) = row.decoded;
    let open = DecodeLimits::default().with_max_alloc_bytes(u64::MAX);
    let ok = decode_bytes_with_limits(&row.bytes, open)
        .unwrap_or_else(|e| panic!("{} must decode with the budget lifted: {e}", row.format));
    assert_eq!(
        (ok.width(), ok.height()),
        (w, h),
        "{} decoded at a geometry the row does not claim",
        row.format
    );

    let tight = DecodeLimits::default().with_max_alloc_bytes(row.price - 1);
    decode_bytes_with_limits(&row.bytes, tight)
        .err()
        .unwrap_or_else(|| panic!("{} must be refused one byte under its price", row.format))
}

/// Issue #686. Every format libviprs prices itself reports the refusal in one
/// shape, so a caller writes one match arm rather than five.
#[test]
fn every_self_priced_decoder_reports_one_alloc_refusal_shape() {
    let mut wrong = Vec::new();
    for row in priced_by_libviprs() {
        let err = refuse(&row);
        if !matches!(err, SourceError::AllocLimitExceeded { .. }) {
            wrong.push(format!("{}: {err:?}", row.format));
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of the self-priced decoders refuse the budget in a shape of their own \
         instead of SourceError::AllocLimitExceeded, so catching the budget still \
         needs one match arm per format (issue #686):\n  {}",
        wrong.len(),
        wrong.join("\n  ")
    );
}

/// Issue #686. And the shape carries what the five per-format variants
/// carried, so nothing a caller could read is lost in the merge.
///
/// Asserted here through `Display`, which is the half of the contract that
/// exists on both sides of the change; the typed `geometry` field lands with
/// the fix and is pinned by
/// `the_shape_carries_the_geometry_and_the_label_as_typed_fields` there,
/// because a commit cannot destructure a field that does not exist yet.
///
/// The label is the point of the `what` field: #632 had to split
/// `DecodeLimits::exceeds_alloc_budget` out of `check_alloc` precisely because
/// five callers were building a label nobody could observe. Today the five
/// messages open with a lowercase format tag (`gif:`, `fits:`) and name no
/// buffer at all, and three of the five do not report the band count they
/// priced.
#[test]
fn the_message_names_the_buffer_and_the_geometry_it_priced() {
    let mut wrong = Vec::new();
    for row in priced_by_libviprs() {
        let err = refuse(&row);
        let shown = format!("{err}");
        let (w, h, bands) = row.priced_geometry;
        let want = format!("{w}x{h}x{bands}");
        if !shown.contains(&want) {
            wrong.push(format!(
                "{}: message does not carry the geometry {want} it priced: {shown:?}",
                row.format
            ));
        }
        if !shown.contains(&row.price.to_string()) {
            wrong.push(format!(
                "{}: message does not carry the price {}: {shown:?}",
                row.format, row.price
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "the refusal message must say what was priced and how (issue #686):\n  {}",
        wrong.join("\n  ")
    );
}

/// Issue #686. And the typed fields carry what the five per-format variants
/// carried, so nothing a caller could read is lost in the merge.
///
/// The sibling above asserts the same thing through `Display`, and it is the
/// one that could be written before the fix; this is the half that needs the
/// `geometry` field to exist.
///
/// The GIF row is the one worth reading: it declares three bands in the file
/// and prices four, because the canvas the decoder allocates is RGBA whatever
/// the palette holds. The reported band count is the one that was **priced**,
/// which is the only one that explains the number next to it.
#[test]
fn the_shape_carries_the_geometry_and_the_label_as_typed_fields() {
    for row in priced_by_libviprs() {
        let err = refuse(&row);
        let SourceError::AllocLimitExceeded {
            what,
            geometry,
            needed_bytes,
            max_alloc_bytes,
        } = err
        else {
            panic!("{}: not the shared shape: {err:?}", row.format);
        };
        let g = geometry.unwrap_or_else(|| {
            panic!(
                "{}: a decoder that priced a declared geometry must report it",
                row.format
            )
        });
        assert_eq!(
            (g.width, g.height, g.bands),
            row.priced_geometry,
            "{} reports a geometry that is not the one it priced",
            row.format
        );
        assert_eq!(
            u64::from(g.width) * u64::from(g.height) * u64::from(g.bands) * row.sample_bytes,
            needed_bytes,
            "{}: the reported geometry must be the one the price came from",
            row.format
        );
        assert_eq!(needed_bytes, row.price, "{} price", row.format);
        assert_eq!(
            max_alloc_bytes,
            row.price - 1,
            "{} must report the ceiling in force",
            row.format
        );
        assert_eq!(what, row.what, "{} label", row.format);
        assert!(
            format!("{err}").contains(what),
            "{}: the label must reach the message a caller prints",
            row.format
        );
    }
}

/// Issue #686. And one call catches every shape the budget can refuse in, so
/// a caller does not have to know the split at all.
#[test]
fn is_alloc_limit_catches_every_shape_the_budget_refuses_in() {
    for row in priced_by_libviprs()
        .iter()
        .chain(&priced_by_the_image_crate())
    {
        let err = refuse(row);
        assert!(
            err.is_alloc_limit(),
            "{}: is_alloc_limit must answer for every container: {err:?}",
            row.format
        );
    }

    // And it does not answer for the ceilings that are not this one. Both of
    // these are refusals a caller fixes by raising a *different* knob, so a
    // predicate that swept them in would be worse than the seven match arms.
    let big = rgb8(4).encode_png(6).expect("png fixture");
    let by_pixels = decode_bytes_with_limits(&big, DecodeLimits::default().with_max_pixels(1))
        .expect_err("one pixel is not a 4x4 image");
    assert!(
        !by_pixels.is_alloc_limit(),
        "the pixel ceiling is a different knob: {by_pixels:?}"
    );
    let by_coord = decode_bytes_with_limits(&big, DecodeLimits::default().with_max_coord(1))
        .expect_err("one pixel wide is not a 4x4 image");
    assert!(
        !by_coord.is_alloc_limit(),
        "the coordinate ceiling is a different knob: {by_coord:?}"
    );
}

/// Issue #686. The four formats the `image` crate refuses keep reporting the
/// `image` shape, and this is here so that is a decision rather than a gap.
///
/// They are not a mechanical move. In all four the ceiling is spent inside
/// `image`'s own decoder through `Limits::reserve`, so there is no libviprs
/// price to report and no declared geometry to attach; WebP reaches the same
/// error deliberately, from its own pre-check, so that it refuses the same
/// frames as its three siblings and says the same thing about them.
///
/// This is what a caller still does about WebP, which #686 asks to be spelled
/// out: exactly what they do about JPEG, PNG and TIFF, and
/// [`SourceError::is_alloc_limit`] covers all four without them having to know.
#[test]
fn the_image_backed_decoders_still_report_the_image_shape() {
    for row in priced_by_the_image_crate() {
        let err = refuse(&row);
        assert!(
            matches!(
                err,
                SourceError::Decode(image::ImageError::Limits(ref e))
                    if matches!(e.kind(), image::error::LimitErrorKind::InsufficientMemory)
            ),
            "{} is expected to report the image crate's own budget refusal, got {err:?}",
            row.format
        );
    }
}
