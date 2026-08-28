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
//! | WebP | `SourceError::Decode(image Limits)`, fabricated by hand |
//! | GIF | `SourceError::Gif(GifError::AllocLimitExceeded)` |
//! | Radiance | `SourceError::Radiance(RadianceError::AllocLimitExceeded)` |
//! | FITS | `SourceError::Fits(FitsError::AllocLimitExceeded)` |
//! | OpenEXR | `SourceError::Exr(ExrError::AllocLimitExceeded)` |
//! | JPEG XL | `SourceError::Jxl(JxlError::AllocLimitExceeded)` |
//! | NIfTI | *did not exist yet; it joined self-priced in #510* |
//!
//! Two corrections to the table in #686, both from that run. WebP is **not**
//! the one odd format on the `image` shape: JPEG, PNG and single-image TIFF
//! report exactly the same thing. But the four are not alike underneath. In
//! JPEG, PNG and TIFF the ceiling is spent inside `image`'s own decoder
//! through `Limits::reserve`, so there is no libviprs price and no declared
//! geometry to report. WebP had both and threw them away to look like the
//! other three, so it moves onto the shared shape and the three do not.
//!
//! And the `.v` reader consulted `max_alloc_bytes` nowhere at all, which was
//! the third correction and became issue #710. It does now, so `.v` has a row
//! in the first table rather than a comment saying why it has none:
//! `decode_vips_bytes` prices its pixel buffer from the declared header
//! geometry through the same `DecodeLimits::check_image_alloc` as every other
//! self-priced container.
//!
//! What was wrong there was the contract rather than the safety. `.v` is not a
//! decompression-bomb vector: the reader refuses a header promising more pixel
//! data than the file physically holds, so the allocation was already bounded
//! by the input length. But a caller who set `max_alloc_bytes` did not get it
//! on one container out of ten, and the two entry points disagreed about the
//! same run of bytes: `decode_file_with_limits` refuses an over-budget `.v` at
//! the bounded whole-file read, while `decode_bytes_with_limits` served it.
//!
//! # What this file holds
//!
//! The formats where **libviprs itself** prices a declared geometry and
//! refuses it must all report one shape, `SourceError::AllocLimitExceeded`.
//! The three where the `image` crate refuses keep reporting the `image` shape,
//! and that is asserted here too so the split is a decision on the record
//! rather than a gap. Three, not four: WebP left that side in #686, and the
//! prose here and in `SourceError::is_alloc_limit`'s doc went on saying four
//! until #782.

use libviprs::jxl::JxlError;
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
/// AVIF has no encoder in this crate (deliberately, see `libviprs::avif`), so
/// its row is built from a committed fixture the way the OpenEXR row is
/// rather than by encoding a raster.
const AVIF: &[u8] = include_bytes!("../oracle-captures/foreign-avif/fixtures/rgb8.avif");

/// The NIfTI fixture the `nifti` module's own budget tests use: a NIfTI-1
/// header declaring `dim = [3, 2, 3, 1]` of `NIFTI_TYPE_UINT8`, so
/// `decode_nifti` prices it at `2 * 3 * 1 * 1` = 6 bytes.
///
/// Included rather than encoded, because NIfTI is load-only here: there is no
/// `Raster::encode_nifti` to build a fixture with, and there is no
/// `vips niftisave` either (the pinned build has no NIfTI support at all,
/// which is why the oracle for that module is `nifti_clib`).
const NIFTI: &[u8] = include_bytes!("../oracle-captures/foreign-nifti/fixtures/dt2_uint8.nii");

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
        // libviprs's own container, and the last one to get the budget.
        // `decode_vips_bytes` applied `max_coord` and `max_pixels` and then
        // went straight to the pixel copy, so a 36-byte raster decoded clean
        // under a 35-byte ceiling (issue #710). The price is the declared
        // geometry's product, which is also exactly the byte range the reader
        // copies out of the file, so the row's `price` holds those two
        // spellings together from outside the crate.
        Row {
            format: "v",
            bytes: rgb8(4).encode_vips().expect("v fixture"),
            decoded: (4, 4),
            priced_geometry: (4, 4, 3),
            sample_bytes: 1,
            what: ".v pixel buffer",
            price: 48,
        },
        Row {
            format: "gif",
            bytes: rgb8(4).encode_gif(Default::default()).expect("gif fixture"),
            decoded: (4, 4),
            // Four bands because this fixture carries a transparent index, so
            // the canvas is RGBA and 4 * 4 * 4 = 64. An opaque GIF is priced
            // at three. The band count is what the canvas costs, not a
            // constant.
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
        // NIfTI declares a whole volume in 348 bytes and then hands over a
        // raw array, which is the decompression-bomb shape the budget
        // exists for, so it prices the declared geometry before it reserves
        // anything (issue #510).
        Row {
            format: "nifti",
            bytes: NIFTI.to_vec(),
            decoded: (2, 3),
            priced_geometry: (2, 3, 1),
            sample_bytes: 1,
            what: "NIfTI voxel buffer",
            price: 6,
        },
        // WebP prices its own frame from `output_buffer_size` and its own
        // declared geometry, so it belongs here and not with the three the
        // `image` crate refuses. It used to fabricate an `image` `LimitError`
        // to look like them, which is the confusion #686 exists to remove.
        Row {
            format: "webp",
            bytes: rgb8(4)
                .encode_webp(Default::default())
                .expect("webp fixture"),
            decoded: (4, 4),
            priced_geometry: (4, 4, 3),
            sample_bytes: 1,
            what: "WebP frame buffer",
            price: 48,
        },
        // Ultra HDR is the only row here that prices **two** images: a
        // container holds a base JPEG and a gain-map JPEG, and both go
        // through `check_image_alloc` from their own SOF before either is
        // decoded. The base is priced first, so this row's refusal is the
        // base's; `uhdr_prices_the_gain_map_as_well_as_the_base` in
        // `tests/uhdr_ported_surface.rs` is the one that pins the other
        // half, with a budget that admits the base and refuses the gain
        // map. Pricing only the base would let a 1x1 base smuggle in a
        // 60000x60000 gain map (issue #508).
        Row {
            format: "uhdr",
            bytes: libviprs::uhdr::smallest_container(),
            decoded: (8, 8),
            priced_geometry: (8, 8, 3),
            sample_bytes: 1,
            what: "Ultra HDR base image",
            price: 192,
        },
    ];
    if cfg!(feature = "avif") {
        rows.push(Row {
            format: "avif",
            bytes: AVIF.to_vec(),
            decoded: (4, 3),
            // Priced off the container's declared `ispe`, before any AV1 is
            // decoded, which is the whole point for a compressed container.
            priced_geometry: (4, 3, 3),
            sample_bytes: 1,
            what: "AVIF frame buffer",
            price: 36,
        });
    }
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

/// Issue #686. The two tables together account for every container, so adding
/// a decoder without adding a row cannot pass unnoticed.
///
/// This is the half of that link which lives out here. The other half is
/// `source::tests::adding_a_container_reddens_the_alloc_refusal_tables`, which
/// pins `SniffedFormat::ALL.len()` and points back at this file, because the
/// set of containers is `pub(crate)` and an integration test cannot see it.
/// Neither half is sufficient alone: this one catches a row deleted here, that
/// one catches a container added there.
///
/// Every container has a row now. `.v` was the one in neither table, because
/// it applied no allocation budget at all, and closing #710 is what let the
/// exclusion term below be deleted rather than kept as a documented hole.
#[test]
fn the_two_tables_account_for_every_container() {
    let self_priced = priced_by_libviprs().len();
    let image_backed = priced_by_the_image_crate().len();

    // JPEG XL and AVIF are each only compiled in behind their own feature, so
    // the self-priced table is one shorter for each that is off. Spelled out
    // rather than hidden in a `cfg!` inside the sum, because a reader has to
    // be able to check the arithmetic. The 8 are gif, radiance, fits, openexr,
    // webp, uhdr, .v and nifti.
    let expected_self_priced =
        8 + usize::from(cfg!(feature = "jxl")) + usize::from(cfg!(feature = "avif"));
    assert_eq!(
        self_priced, expected_self_priced,
        "the self-priced table changed size"
    );
    assert_eq!(image_backed, 3, "the image-backed table changed size");

    let absent_features =
        usize::from(!cfg!(feature = "jxl")) + usize::from(!cfg!(feature = "avif"));
    assert_eq!(
        self_priced + image_backed + absent_features,
        13,
        "the two tables must account for all thirteen containers libviprs sniffs, \
         with no exclusions left; see SniffedFormat::ALL"
    );
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
/// The OpenEXR row is the one worth reading: the file declares sixteen
/// channels in the AOV case and a successful decode hands back four, because
/// the decoder builds a buffer for every declared channel and the selection
/// keeps four. The reported band count is the one that was **priced**, which
/// is the only one that explains the number next to it.
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

    // And the one that actually guards the `kind()` check. `max_width` and
    // `max_height` are pushed down into `image::Limits` too, so they come back
    // as the *same* `SourceError::Decode(ImageError::Limits(..))` shape with a
    // `DimensionError` kind. Without the `kind()` match `is_alloc_limit` would
    // answer true for them, and the two controls above would not notice,
    // because both produce libviprs's own variants and fall through the
    // catch-all arm without ever reaching the `Decode` one.
    let by_width = decode_bytes_with_limits(&big, DecodeLimits::default().with_max_width(1))
        .expect_err("one pixel wide is not a 4x4 image");
    assert!(
        matches!(by_width, SourceError::Decode(image::ImageError::Limits(_))),
        "this control is only meaningful if it reaches the image shape: {by_width:?}"
    );
    assert!(
        !by_width.is_alloc_limit(),
        "the width ceiling arrives in the image shape but is a different knob: {by_width:?}"
    );

    // The third control, and the one that looks most like a false negative
    // until you read what it says. `Raster::ppm_load`, `csv_load` and
    // `matrix_load` are public decode entry points returning this same enum,
    // and an over-large declared geometry comes back as
    // `Raster(ByteBudgetExceeded)` whose message reads "needs N bytes,
    // exceeding the M-byte allocation budget". That budget is
    // `DEFAULT_MAX_ALLOC_BYTES`, the raster construction ceiling, not
    // `DecodeLimits::max_alloc_bytes`, so raising the decode limit does
    // nothing about it and `is_alloc_limit` must say no.
    let by_construction =
        Raster::ppm_load(b"P6\n60000 60000\n255\n").expect_err("60000 squared RGB8 is 10.8 GB");
    assert!(
        by_construction.to_string().contains("allocation budget"),
        "this control is only meaningful if it reads like an allocation refusal: {by_construction}"
    );
    assert!(
        !by_construction.is_alloc_limit(),
        "the raster construction budget is a different ceiling with a different \
         remedy: {by_construction:?}"
    );
}

/// Issue #686. `is_alloc_limit` answers the same for the same value whether or
/// not the `jxl` feature is on.
///
/// `JxlError` and `DecoderAllocLimitExceeded` are declared unconditionally, and
/// #634 promises a caller's `match` has the same arms in either build. The
/// predicate's arm was `#[cfg(feature = "jxl")]`, so this exact value answered
/// `false` without the feature and `true` with it. Features are additive, so
/// one crate in a workspace turning `jxl` on would silently change another
/// crate's error handling.
///
/// Built by hand rather than decoded, on purpose. The `jxl` row in
/// `priced_by_libviprs` is 16x16 so it trips the declared-geometry check, which
/// means `DecoderAllocLimitExceeded` is not reachable through any decode in
/// this file, and it must be asserted in the build that has no decoder at all.
#[test]
fn is_alloc_limit_does_not_depend_on_the_jxl_feature() {
    let err = SourceError::Jxl(JxlError::DecoderAllocLimitExceeded { max_alloc_bytes: 8 });
    assert!(
        err.is_alloc_limit(),
        "the decoder's own allocation tracker is the budget refusing the file, \
         in either build: {err:?}"
    );
}

/// Issue #686. The three formats the `image` crate refuses keep reporting the
/// `image` shape, and this is here so that is a decision rather than a gap.
///
/// They are not a mechanical move. In all three the ceiling is spent inside
/// `image`'s own decoder through `Limits::reserve`, so there is no libviprs
/// price to report and no declared geometry to attach.
///
/// WebP was a fourth here until #686 moved it, and this doc went on saying so
/// until #782. It still refuses the same *frames* as its three siblings, which
/// is the part that was always true and is why the pre-check exists; what it no
/// longer does is say the same *thing* about them, because it has a price and a
/// declared geometry and reports both.
///
/// This is what a caller still does about WebP, which #686 asks to be spelled
/// out: exactly what they do about JPEG, PNG and TIFF, because
/// [`SourceError::is_alloc_limit`] covers all four shapes without them having
/// to know which side of the split a container is on.
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

/// Issue #710. The `.v` budget prices the pixel body at the header's declared
/// sample depth, which is neither the file's length nor one byte per sample.
///
/// The `.v` row in `priced_by_libviprs` cannot see either mistake, and that is
/// why this exists rather than being folded into the table. Its fixture is
/// `Rgb8`, so a price that assumed one byte per sample would be the same
/// number; and a price taken from the whole slice would still refuse at
/// `price - 1`, because the file is longer than the body it holds. `.v` is the
/// only container in either table whose body is a plain copy out of a longer
/// file, and the only one carrying 2- and 4-byte samples, so both mistakes are
/// live here and nowhere else.
///
/// The second half is the one that says what the budget means: at the body's
/// price exactly, a file three times that long decodes. `max_alloc_bytes`
/// bounds what the decoder allocates, not what the caller handed it. That is
/// the whole difference between `check_image_alloc` and the bounded whole-file
/// read `decode_file_with_limits` does first.
#[test]
fn the_v_budget_prices_the_declared_body_and_not_the_file() {
    // 4x3 Rgb16: 24 samples of two bytes, so the body is 72 bytes.
    let data: Vec<u8> = (0..72u32).map(|v| v as u8).collect();
    let raster = Raster::new(4, 3, PixelFormat::Rgb16, data).expect("rgb16 fixture");
    let bytes = raster.encode_vips().expect("v fixture");
    assert!(
        bytes.len() > 72,
        "the fixture has to be longer than its body for this test to say \
         anything, and it is {} bytes",
        bytes.len()
    );

    let err = decode_bytes_with_limits(&bytes, DecodeLimits::default().with_max_alloc_bytes(71))
        .expect_err("a 72-byte body must be refused under a 71-byte ceiling");
    let SourceError::AllocLimitExceeded {
        what,
        geometry,
        needed_bytes,
        ..
    } = err
    else {
        panic!("not the shared shape: {err:?}");
    };
    assert_eq!(what, ".v pixel buffer", "label");
    assert_eq!(
        needed_bytes, 72,
        "the price is width * height * bands * the declared sample depth"
    );
    let g = geometry.expect("a decoder that priced a declared geometry must report it");
    assert_eq!((g.width, g.height, g.bands), (4, 3, 3), "reported geometry");

    let ok = decode_bytes_with_limits(&bytes, DecodeLimits::default().with_max_alloc_bytes(72))
        .expect("a 72-byte body must decode under a 72-byte ceiling");
    assert_eq!((ok.width(), ok.height()), (4, 3));
    assert_eq!(ok.format(), PixelFormat::Rgb16);
}

/// Issue #710. A `.v` band count with no `PixelFormat` is still a format
/// error, not an allocation refusal, however tight the budget is.
///
/// This pins an ordering decision rather than a behaviour: the budget check
/// sits after `PixelFormat::with_channels` in `decode_vips_bytes`, so a header
/// declaring more bands than a `PixelFormat` can hold comes back the way it
/// always did. Moving the check one line earlier answers `AllocLimitExceeded`
/// for the same file, which is a worse answer, because raising
/// `max_alloc_bytes` would not make the file readable.
///
/// It takes a budget of 1 to see at all. At the default ceiling a 4x4 raster
/// of 70000 bands is 1.1 MB and neither order refuses it, so the two spellings
/// are indistinguishable and this test would be green either way.
#[test]
fn an_unrepresentable_v_band_count_is_a_format_error_not_a_budget_one() {
    let mut bytes = rgb8(4).encode_vips().expect("v fixture");
    // Offset 12 is the band count, and 70000 is past `u16::MAX`, which is
    // where `PixelFormat::with_channels` gives up.
    bytes[12..16].copy_from_slice(&70_000i32.to_ne_bytes());

    let err = decode_bytes_with_limits(&bytes, DecodeLimits::default().with_max_alloc_bytes(1))
        .expect_err("70000 bands is not a representable PixelFormat");
    assert!(
        matches!(err, SourceError::VipsFormat(ref m) if m.contains("70000")),
        "an unrepresentable band count must stay a format error even under a \
         budget that would also refuse it: {err:?}"
    );

    // The positive control: the same file under the same budget is genuinely
    // over it, so the assertion above is about which check fires first and not
    // about the file being fine.
    let representable = rgb8(4).encode_vips().expect("v fixture");
    let over = decode_bytes_with_limits(
        &representable,
        DecodeLimits::default().with_max_alloc_bytes(1),
    )
    .expect_err("48 bytes is over a 1-byte ceiling");
    assert!(
        over.is_alloc_limit(),
        "the control must reach the budget: {over:?}"
    );
}

/// `src/source.rs` as text, so a claim its docs make about *this* file's split
/// can be checked from here.
///
/// `include_str!` rather than a read at runtime, deliberately: it costs nothing
/// at run time, and a test that opened a path would need a
/// `#[cfg_attr(miri, ignore)]` and a row in `tests/miri_fs_test_inventory.txt`,
/// which is a shared count two lanes can move at once.
const SOURCE_RS: &str = include_str!("../src/source.rs");

/// Issue #782. WebP is not one of the containers the `image` crate refuses,
/// and `SourceError::is_alloc_limit`'s own doc said it was.
///
/// #709 moved WebP off the `image` shape and onto
/// `SourceError::AllocLimitExceeded`, and left three prose sites describing the
/// world before the move. The public one is the bullet list a caller reads to
/// decide what to match, which still listed WebP beside JPEG, PNG and TIFF.
///
/// Nothing caught it because nothing here pins what is **not** in the
/// image-backed table. `the_two_tables_account_for_every_container` pins its
/// size and `the_image_backed_decoders_still_report_the_image_shape` pins what
/// its rows report, so a format moving out of it and leaving its description
/// behind is invisible to both.
///
/// Measured on `ed958d5`: WebP refused at `price - 1` comes back as
/// `AllocLimitExceeded { what: "WebP frame buffer", geometry: Some(4x4x3),
/// needed_bytes: 48, max_alloc_bytes: 47 }`, and it cannot come back as
/// anything else, because WebP is a `Native` row in the route table and the
/// `image` crate never decodes one.
#[test]
fn the_image_shape_doc_names_exactly_the_containers_that_report_it() {
    let bullet = SOURCE_RS
        .split("`image` `LimitError` of kind")
        .nth(1)
        .expect("is_alloc_limit's doc has a bullet for the image LimitError shape")
        .split(';')
        .next()
        .expect("that bullet ends at its semicolon");

    for row in priced_by_the_image_crate() {
        let named = row.format.to_uppercase();
        assert!(
            bullet.contains(&named),
            "{named} reports the image shape but the is_alloc_limit doc does not \
             name it: {bullet:?}"
        );
    }
    assert!(
        !bullet.contains("WebP"),
        "WebP has priced its own frame and reported SourceError::AllocLimitExceeded \
         since #686, so the is_alloc_limit doc must not list it among the containers \
         the image crate refuses (issue #782): {bullet:?}"
    );

    // And the executable half, so the doc is not the only thing saying it.
    assert!(
        !priced_by_the_image_crate()
            .iter()
            .any(|r| r.format == "webp"),
        "WebP left the image-backed table in #686"
    );
    let webp = priced_by_libviprs()
        .into_iter()
        .find(|r| r.format == "webp")
        .expect("WebP is a self-priced row");
    let err = refuse(&webp);
    assert!(
        !matches!(err, SourceError::Decode(image::ImageError::Limits(_))),
        "WebP must not report the image crate's own refusal: {err:?}"
    );
    assert!(
        matches!(
            err,
            SourceError::AllocLimitExceeded {
                what: "WebP frame buffer",
                ..
            }
        ),
        "WebP prices its own frame and names it: {err:?}"
    );
}
