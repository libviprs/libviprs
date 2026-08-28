//! Every operation that builds a fresh raster carries the input's metadata
//! onto it: the header block (interpretation, resolution, offsets,
//! orientation) *and* the attached fields (issues #717 and #718).
//!
//! Before #717 each module rebuilt the metadata by hand, and eleven of the
//! eighteen open-coded sites copied the header block and left the attached
//! fields behind, so the ICC profile, the EXIF blob and every named field a
//! caller had set went missing. This file is the cross-module guard on the one
//! carry that replaced them, and it lives outside `src/` on purpose: the
//! invariant spans `conversion`, `create`, `freqfilt`, `convolution`,
//! `composite`, `colour` and `extract`, so no single module's test block is
//! the right home for it.
//!
//! # What the oracle says
//!
//! Measured against the pinned `vips-8.18.6` (`oracle-captures/ORACLE_PIN.json`),
//! read off the binary at `/opt/homebrew/bin/vips` rather than out of the C.
//! The source is an 8x8 uchar 3-band `.v` tagged
//! `--interpretation rgb --xres 5 --yres 7 --xoffset 11 --yoffset 13`, with
//! `orientation` 6, a `VipsRefString` `lane-711` = `carried` and a real
//! 3144-byte sRGB ICC profile attached through
//! `vips icc_transform in.v out.v "sRGB Profile.icc"`.
//!
//! ```text
//! op                interpretation   xres  orientation  lane-711  icc
//! cast ushort       RGB              5     6            carried   3144
//! gamma             RGB              5     6            carried   3144
//! falsecolour       sRGB             5     6            carried   3144
//! addalpha          RGB              5     6            carried   3144
//! arrayjoin         RGB              5     6            carried   3144
//! join              RGB              5     6            carried   3144
//! flip h / flip v   RGB              5     6            carried   3144
//! rot d90           RGB              5     6            carried   3144
//! rot45 d45         RGB              5     6            carried   3144
//! grid              RGB              5     6            carried   3144
//! wrap              RGB              5     6            carried   3144
//! fwfft             FOURIER          5     6            carried   3144
//! invfft            B_W              5     6            carried   -
//! invfft --real     B_W              5     6            carried   -
//! freqmult          B_W              5     6            carried   -
//! colourspace lab   LAB              5     6            carried   3144
//! sobel / canny     RGB              5     6            carried   3144
//! composite2 over   RGB              5     6            carried   3144
//! insert            RGB              5     6            carried   3144
//! ```
//!
//! The tag is `rgb` rather than `scrgb` deliberately. `vips gamma` on an
//! `scrgb` or `rgb16` source hands back `srgb`, because it retags off the
//! output's sample format, and pinning the carry against a source that trips
//! an unrelated retag rule would measure the wrong thing. On `rgb`, `lab`,
//! `b-w` and `hsv` sources `gamma` reports the input's tag straight back,
//! which is the carry this file is about.
//!
//! # The two cells that are not a wholesale carry
//!
//! **The inverse-FFT ops drop the ICC profile.** `invfft`, `invfft --real`
//! and `freqmult` all retag to `b-w`, and vips removes an RGB profile with
//! that retag. It is the profile specifically and not blobs in general: a
//! second, plain 48-byte `VipsBlob` attached alongside survives all three. The
//! general rule (a profile is removed when the new tag's band count disagrees
//! with the profile's own colour space) is issue #720; these three cells are
//! what this file pins.
//!
//! **`join`, `arrayjoin` and `insert` take the union of both inputs' fields**,
//! the first input winning a name they share, while the header block comes
//! from the first input alone. Issue #718.

use libviprs::{
    Angle, Angle45, Combine, CompositeMode, Interpretation, JoinDirection, Kernel, MetadataValue,
    PixelFormat, Precision, Raster,
};

/// The attached string the carry is read through. Named for the issue so a
/// failure points at the right measurement.
const LANE: &str = "lane-717";

/// A stand-in ICC blob. libviprs stores `icc-profile-data` as opaque bytes and
/// never parses it, so three bytes exercise the [`MetadataValue::Blob`] arm as
/// well as 3144 would; the *oracle* half of this was measured on a real sRGB
/// profile, because a blob has to be a real profile to survive
/// `vips icc_transform`.
const PROFILE: &[u8] = &[0xde, 0xad, 0xbe, 0xef];

/// A second blob under a name vips does not know, so a test can tell "the ICC
/// profile was dropped" apart from "every blob was dropped".
const PLAIN_BLOB: &str = "lane-717-blob";

/// A 3-band uchar raster carrying every piece of metadata the carry is
/// supposed to move: a header block that differs from the default in all five
/// fields, and two attached fields in two different value arms.
///
/// 7x7 because `rot45` refuses anything that is not odd and square, and every
/// other op under test is happy at that size.
fn tagged_source() -> Raster {
    let data: Vec<u8> = (0..7 * 7 * 3).map(|i| (i * 5 % 251) as u8).collect();
    tag(&Raster::new(7, 7, PixelFormat::Rgb8, data).unwrap())
}

/// Put the same metadata on any raster, so a test whose input is another op's
/// output (`invfft` wants a Fourier raster) starts from a tagged source rather
/// than from whatever the previous op carried.
///
/// Every value here differs from what a freshly built raster reports: the
/// defaults are interpretation-inferred-from-format (`Srgb` for `Rgb8`),
/// `1.0`, `1.0`, `0`, `0` and `1`, and no attached fields at all.
fn tag(im: &Raster) -> Raster {
    let mut out = im
        .copy()
        .interpretation(Interpretation::Rgb)
        .xres(5.0)
        .yres(7.0)
        .xoffset(11)
        .yoffset(13)
        .orientation(6)
        .build();
    out.set_field(LANE, MetadataValue::Str("carried".to_string()));
    out.set_field(PLAIN_BLOB, MetadataValue::Blob(vec![1, 2, 3]));
    out.set_icc_profile(PROFILE);
    out
}

/// A tagged 4-band source, for the two ops that want an alpha band.
fn tagged_rgba() -> Raster {
    let data: Vec<u8> = (0..7 * 7 * 4).map(|i| (i * 7 % 251) as u8).collect();
    tag(&Raster::new(7, 7, PixelFormat::Rgba8, data).unwrap())
}

/// Every single-input op that builds a fresh raster, with the interpretation
/// the oracle says its output should carry.
///
/// `None` means "this op does not decide the tag, so the test does not assert
/// one". That is only the three inverse-FFT ops: vips retags those `b-w` and
/// libviprs leaves the interpretation unset so the format infers it, which is
/// a divergence `src/freqfilt.rs` already documents and #717 does not touch.
fn single_input_results() -> Vec<(&'static str, Raster, Option<Interpretation>)> {
    let im = tagged_source();
    let rgba = tagged_rgba();
    let fourier = tag(&im.try_fwfft().unwrap());
    let mask = tagged_source();

    vec![
        (
            "cast",
            im.try_cast(PixelFormat::Rgb16).unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "gamma",
            im.try_gamma(None).unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "falsecolour",
            im.try_falsecolour().unwrap(),
            Some(Interpretation::Srgb),
        ),
        (
            "addalpha",
            im.try_addalpha().unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "fliphor",
            im.try_fliphor().unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "flipver",
            im.try_flipver().unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "rot d90",
            im.try_rot(Angle::D90).unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "rot45 d45",
            im.try_rot45(Angle45::D45).unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "grid",
            im.try_grid(7, 1, 1).unwrap(),
            Some(Interpretation::Rgb),
        ),
        ("wrap", im.try_wrap().unwrap(), Some(Interpretation::Rgb)),
        (
            "fwfft",
            im.try_fwfft().unwrap(),
            Some(Interpretation::Fourier),
        ),
        ("invfft", fourier.try_invfft().unwrap(), None),
        ("invfft_real", fourier.try_invfft_real().unwrap(), None),
        ("freqmult", im.try_freqmult(&mask).unwrap(), None),
        (
            "colourspace lab",
            im.try_colourspace(Interpretation::Lab).unwrap(),
            Some(Interpretation::Lab),
        ),
        ("sobel", im.try_sobel().unwrap(), Some(Interpretation::Rgb)),
        (
            "canny",
            im.try_canny(1.4, Precision::Float).unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "composite2",
            rgba.try_composite2(&tagged_rgba(), CompositeMode::Over)
                .unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "arrayjoin",
            Raster::try_arrayjoin(&[&im, &tagged_source()], Some(2), None).unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "join",
            im.try_join(
                &tagged_source(),
                JoinDirection::Horizontal,
                true,
                None,
                None,
                None,
            )
            .unwrap(),
            Some(Interpretation::Rgb),
        ),
        (
            "insert",
            im.try_insert(&tagged_source(), 2, 2, true, None).unwrap(),
            Some(Interpretation::Rgb),
        ),
    ]
}

/// Issue #717. The header block survives every op.
///
/// The defaults are `1.0`, `1.0` and `1`, so every value asserted here differs
/// from what a freshly built raster would report and the assertion cannot pass
/// by accident.
#[test]
fn every_op_carries_the_header_block() {
    for (name, out, want_interpretation) in single_input_results() {
        assert_eq!(out.xres(), 5.0, "{name} xres");
        assert_eq!(out.yres(), 7.0, "{name} yres");
        assert_eq!(out.orientation(), 6, "{name} orientation");
        if let Some(want) = want_interpretation {
            assert_eq!(out.interpretation(), want, "{name} interpretation");
        }
    }
}

/// Issue #717. The attached fields survive too, which is the half a bare
/// `out.meta = self.meta` leaves behind.
///
/// Both `MetadataValue` arms are here because they are separate lines in the
/// map and a fix that moved one could miss the other: a `Str` under `lane-717`
/// and a `Blob` under a name vips does not know.
#[test]
fn every_op_carries_the_attached_fields() {
    for (name, out, _) in single_input_results() {
        assert_eq!(
            out.get_field(LANE),
            Some(MetadataValue::Str("carried".to_string())),
            "{name} attached string"
        );
        assert_eq!(
            out.get_field(PLAIN_BLOB),
            Some(MetadataValue::Blob(vec![1, 2, 3])),
            "{name} attached blob"
        );
    }
}

/// Issue #717. The ICC profile is the attachment a caller notices losing, and
/// it rides the same map as the rest.
///
/// The three inverse-FFT ops are the exception and are asserted separately in
/// [`the_inverse_fourier_ops_drop_the_icc_profile_and_keep_the_rest`], so this
/// list is the positive control for that one: without it, "the profile is
/// gone" would be indistinguishable from "no op carries a profile at all".
#[test]
fn every_op_carries_the_icc_profile() {
    for (name, out, _) in single_input_results() {
        if matches!(name, "invfft" | "invfft_real" | "freqmult") {
            continue;
        }
        assert_eq!(
            out.icc_profile(),
            Some(PROFILE),
            "{name} attached ICC profile"
        );
    }
}

/// Issue #717. The one cell in the table that is not a wholesale carry.
///
/// Measured on vips 8.18.6: `invfft`, `invfft --real` and `freqmult` retag the
/// output `b-w`, and an RGB profile does not survive that retag. The other
/// attached fields do, including a second plain `VipsBlob`, so this is the
/// profile being invalidated rather than the field map being dropped.
///
/// ```text
/// vips fwfft rgb.v f.v            -> lane-711 carried, icc 3144
/// vips invfft f.v i.v             -> lane-711 carried, icc absent
/// vips invfft f.v ir.v --real     -> lane-711 carried, icc absent
/// vips freqmult f.v f.v fm.v      -> lane-711 carried, icc absent
/// ```
#[test]
fn the_inverse_fourier_ops_drop_the_icc_profile_and_keep_the_rest() {
    for (name, out, _) in single_input_results() {
        if !matches!(name, "invfft" | "invfft_real" | "freqmult") {
            continue;
        }
        assert_eq!(out.icc_profile(), None, "{name} must not carry the profile");
        assert_eq!(
            out.get_field(LANE),
            Some(MetadataValue::Str("carried".to_string())),
            "{name} still carries the string"
        );
        assert_eq!(
            out.get_field(PLAIN_BLOB),
            Some(MetadataValue::Blob(vec![1, 2, 3])),
            "{name} still carries a plain blob"
        );
        assert_eq!(out.xres(), 5.0, "{name} still carries the resolution");
    }
}

/// Issue #718. `insert` is a two-input op and its rule is two rules: the
/// header block comes from `main` alone, the attached fields are the union of
/// both with `main` winning a name they share.
///
/// Measured on vips 8.18.6 from two sources chosen to disagree on every field.
///
/// ```text
/// main: srgb, xres 1, xoffset 41, orientation 1, main-only=from-main, lane=from-main
/// sub:  scrgb, xres 5, xoffset 11, orientation 6, sub-only=from-sub, lane=carried, icc 3144
///
/// vips insert main.v sub.v out.v 2 2
///   interpretation srgb, xoffset 41, xres 1, orientation 1   <- main
///   main-only from-main, lane-711 from-main                  <- main
///   sub-only from-sub, icc-profile-data 3144 bytes           <- sub
/// ```
///
/// I ran it in both directions rather than reading one cell: with the profile
/// on `main` instead of `sub`, `sub-only` still lands and `lane-711` still
/// takes `main`'s value.
#[test]
fn insert_takes_the_main_block_and_the_union_of_the_fields() {
    let (main, sub) = disagreeing_pair();
    let out = main.try_insert(&sub, 2, 2, true, None).unwrap();
    assert_union(&out, "insert");
}

/// Issue #718. `join` and `arrayjoin` follow `insert`'s rule, which is why
/// `out.fields = self.fields.clone()` would have been wrong for both.
///
/// ```text
/// vips join main.v sub.v out.v horizontal     -> main's block, union of fields
/// vips arrayjoin "main.v sub.v" out.v --across 2 -> same
/// vips bandjoin "main.v sub.v" out.v          -> same
/// ```
///
/// `bandjoin` lives in `src/bands.rs` and is not in this change.
#[test]
fn join_and_arrayjoin_follow_the_same_union_rule() {
    let (main, sub) = disagreeing_pair();
    let joined = main
        .try_join(&sub, JoinDirection::Horizontal, true, None, None, None)
        .unwrap();
    assert_union(&joined, "join");

    let arrayed = Raster::try_arrayjoin(&[&main, &sub], Some(2), None).unwrap();
    assert_union(&arrayed, "arrayjoin");
}

/// Two rasters that disagree on every field the union rule can get wrong.
fn disagreeing_pair() -> (Raster, Raster) {
    let data: Vec<u8> = (0..7 * 7 * 3).map(|i| (i % 251) as u8).collect();
    let mut main = Raster::new(7, 7, PixelFormat::Rgb8, data.clone())
        .unwrap()
        .copy()
        .interpretation(Interpretation::Srgb)
        .xres(3.0)
        .yres(4.0)
        .xoffset(41)
        .yoffset(42)
        .orientation(1)
        .build();
    main.set_field("main-only", MetadataValue::Str("from-main".to_string()));
    main.set_field(LANE, MetadataValue::Str("from-main".to_string()));

    let mut sub = tag(&Raster::new(7, 7, PixelFormat::Rgb8, data).unwrap());
    sub.set_field("sub-only", MetadataValue::Str("from-sub".to_string()));

    (main, sub)
}

/// The union rule, asserted in one place so the three ops that share it cannot
/// drift apart.
fn assert_union(out: &Raster, name: &str) {
    assert_eq!(
        out.interpretation(),
        Interpretation::Srgb,
        "{name} takes main's interpretation"
    );
    assert_eq!(out.xres(), 3.0, "{name} takes main's xres");
    assert_eq!(out.yres(), 4.0, "{name} takes main's yres");
    assert_eq!(out.orientation(), 1, "{name} takes main's orientation");
    assert_eq!(
        out.get_field("main-only"),
        Some(MetadataValue::Str("from-main".to_string())),
        "{name} keeps main's own field"
    );
    assert_eq!(
        out.get_field("sub-only"),
        Some(MetadataValue::Str("from-sub".to_string())),
        "{name} takes sub's own field"
    );
    assert_eq!(
        out.get_field(LANE),
        Some(MetadataValue::Str("from-main".to_string())),
        "{name} lets main win the shared name"
    );
    assert_eq!(
        out.icc_profile(),
        Some(PROFILE),
        "{name} takes sub's profile, which main does not have"
    );
}

/// Issue #717. `new_from_image` is the one site that carries the header block
/// and *not* the fields, and that is measured rather than assumed.
///
/// There is no CLI for `vips_image_new_from_image`, so I called it against the
/// same pinned 8.18.6 through `ctypes` on `libvips.42.dylib`. From the tagged
/// source it hands back interpretation scRGB, xres 5, yres 7, xoffset 11,
/// yoffset 13 and **no** `lane-711`, **no** ICC profile, and `orientation`
/// back at 1.
///
/// The orientation is the interesting half. vips holds it as an attached field
/// and drops it with the rest; libviprs keeps it in `RasterMeta` and so used to
/// carry it here with the header block.
#[test]
fn new_from_image_carries_the_header_block_without_the_fields() {
    let im = tagged_source();
    let out = im.try_new_from_image(&[1.0, 2.0, 3.0]).unwrap();

    assert_eq!(out.xres(), 5.0, "xres");
    assert_eq!(out.yres(), 7.0, "yres");
    assert_eq!(out.xoffset(), 11, "xoffset");
    assert_eq!(out.yoffset(), 13, "yoffset");
    assert_eq!(out.interpretation(), Interpretation::Rgb, "interpretation");

    assert_eq!(out.orientation(), 1, "orientation is not carried");
    assert_eq!(out.get_field(LANE), None, "attached string is not carried");
    assert_eq!(out.icc_profile(), None, "profile is not carried");
}

/// Issue #719. The six convolution ops that finish on `raster_from_f64` /
/// `raster_from_i64` or on `rasters_from` used to hand back
/// `RasterMeta::default()` and an empty field map, so they lost the header
/// block *and* the attachments, where the eleven sites in #717 lost only the
/// attachments.
///
/// `src/convolution.rs` said so in a comment inside `try_sobel` and nothing
/// tracked it.
///
/// Measured on vips 8.18.6 from the same 8x8 `rgb` source, `xres 5`,
/// `orientation 6`, `lane-711` and a real 3144-byte sRGB profile:
///
/// ```text
/// op                    format      interp  xres  ori  lane-711  icc
/// conv 3x3              float, 3b   RGB     5     6    carried   3144
/// conv 5x5              float, 3b   RGB     5     6    carried   3144
/// convsep 1x3           float, 3b   RGB     5     6    carried   3144
/// compass 3x3           float, 3b   RGB     5     6    carried   3144
/// gaussblur sigma 1     uchar, 3b   RGB     5     6    carried   3144
/// gaussblur sigma 3     uchar, 3b   RGB     5     6    carried   3144
/// ```
///
/// `spcor` and `fastcor` want a one-band input, so those two ran against an
/// 8x8 `b-w` source carrying a real 2020-byte **grey** profile, and both hand
/// the 2020 bytes on with `lane-711`, `xres` and the orientation. The profile
/// has to match the tag there: a 3-channel profile under a `b-w` tag is
/// removed by the rule in #720, which is about the retag and not about these
/// ops.
///
/// The origin offsets are **not** asserted here. `conv`, `convsep`, `compass`
/// and `gaussblur` stamp a mask-relative origin (`-1 / -1` for a 3x3,
/// `-2 / -2` for a 5x5, `0 / -1` for a separable 1x3) that does not depend on
/// the input's offsets at all, where `spcor` and `fastcor` pass the input's
/// through. That is issue #721, it is the same shape for `flip`, `rot` and
/// `wrap`, and it is not fixed here; asserting the offsets now would pin
/// behaviour this change deliberately leaves wrong.
#[test]
fn every_convolution_op_carries_the_metadata() {
    let im = tagged_source();
    let template = Raster::new(3, 3, PixelFormat::Rgb8, vec![9u8; 27]).unwrap();
    let box3 = Kernel {
        data: vec![vec![1.0; 3]; 3],
        scale: 9.0,
    };
    let sep = Kernel {
        data: vec![vec![1.0, 1.0, 1.0]],
        scale: 3.0,
    };

    for (name, out) in [
        ("conv float", im.try_conv(&box3, Precision::Float).unwrap()),
        (
            "conv integer",
            im.try_conv(&box3, Precision::Integer).unwrap(),
        ),
        ("convsep", im.try_convsep(&sep, Precision::Float).unwrap()),
        (
            "compass max",
            im.try_compass(&box3, 2, Angle45::D45, Combine::Max, Precision::Float)
                .unwrap(),
        ),
        (
            "compass sum",
            im.try_compass(&box3, 2, Angle45::D45, Combine::Sum, Precision::Integer)
                .unwrap(),
        ),
        (
            "gaussblur",
            im.try_gaussblur(1.0, 0.2, Precision::Float).unwrap(),
        ),
        (
            "gaussblur below the copy threshold",
            im.try_gaussblur(0.1, 0.2, Precision::Float).unwrap(),
        ),
        ("spcor", im.try_spcor(&template).unwrap()),
        ("fastcor", im.try_fastcor(&template).unwrap()),
    ] {
        assert_eq!(out.interpretation(), Interpretation::Rgb, "{name} interp");
        assert_eq!(out.xres(), 5.0, "{name} xres");
        assert_eq!(out.yres(), 7.0, "{name} yres");
        assert_eq!(out.orientation(), 6, "{name} orientation");
        assert_eq!(
            out.get_field(LANE),
            Some(MetadataValue::Str("carried".to_string())),
            "{name} attached string"
        );
        assert_eq!(
            out.get_field(PLAIN_BLOB),
            Some(MetadataValue::Blob(vec![1, 2, 3])),
            "{name} attached blob"
        );
        assert_eq!(out.icc_profile(), Some(PROFILE), "{name} ICC profile");
    }
}

/// Issue #717, not #719, and I had that the wrong way round until the mutation
/// sweep said so.
///
/// `sharpen` blurs through `convsep` on a LabS intermediate and comes back
/// through `colourspace`, so I expected it to inherit #719's carry. It does
/// not: its output metadata comes from that final `colourspace`, which
/// `src/colour.rs` already carries, so this test is green on the branch point
/// and survives all three of #719's mutations. It is a pin on a compound op
/// rather than evidence for that change, and saying so is the point of leaving
/// it here.
///
/// Measured on vips 8.18.6 from an `srgb` source (`vips sharpen` refuses an
/// `rgb` one: "no known route from 'labs' to 'rgb'"): the output reports sRGB,
/// xres 5, orientation 6, `lane-717` and the profile, with the offsets carried
/// verbatim at 11 / 13 rather than stamped. So `sharpen` is one of the ops that
/// carries the offset, unlike the four that convolve directly (#721).
#[test]
fn sharpen_carries_through_its_final_colourspace() {
    let data: Vec<u8> = (0..8 * 8 * 3).map(|i| (i * 5 % 251) as u8).collect();
    let im = Raster::new(8, 8, PixelFormat::Rgb8, data).unwrap();
    let mut im = im
        .copy()
        .interpretation(Interpretation::Srgb)
        .xres(5.0)
        .yres(7.0)
        .xoffset(11)
        .yoffset(13)
        .orientation(6)
        .build();
    im.set_field(LANE, MetadataValue::Str("carried".to_string()));
    im.set_icc_profile(PROFILE);

    let out = im.try_sharpen(1.0, 1.0, 2.0).unwrap();

    assert_eq!(out.interpretation(), Interpretation::Srgb, "interpretation");
    assert_eq!(out.xres(), 5.0, "xres");
    assert_eq!(out.orientation(), 6, "orientation");
    assert_eq!((out.xoffset(), out.yoffset()), (11, 13), "offsets carried");
    assert_eq!(
        out.get_field(LANE),
        Some(MetadataValue::Str("carried".to_string())),
        "attached string"
    );
    assert_eq!(out.icc_profile(), Some(PROFILE), "ICC profile");
}
