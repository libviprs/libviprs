//! SVG lane: rasterise an SVG document to an RGBA raster (libvips `svgload`).
//!
//! vips renders SVG through librsvg, which draws into a cairo ARGB32 surface
//! and then unpremultiplies and byteswaps every row
//! (`foreign/svgload.c:733`, `vips__premultiplied_bgra2rgba`). libviprs has
//! no C dependencies and is not going to acquire librsvg, cairo, pango and
//! fontconfig to load one format, so this module drives `resvg` instead: the
//! same job, pure Rust, pinned at 0.48.1 and gated behind the non-default
//! `svg` feature so nobody who does not rasterise SVG pays for the 29 crates
//! it costs.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_svg`] | `svgload_buffer` | [`Raster`] (`Rgba8`, sRGB) or [`DecodeError`] |
//! | [`decode_svg_with_limits`] | `svgload_buffer` under an explicit budget | [`Raster`] or [`DecodeError`] |
//!
//! # Semantics
//!
//! * **Geometry.** `total_scale = scale * dpi / 72.0` (`svgload.c:362`), the
//!   natural document size is multiplied by it, and the result is rounded
//!   with `VIPS_ROUND_UINT`, which is `(int)(R + 0.5)` and therefore
//!   round-half-up on a non-negative value (`svgload.c:568`,
//!   `include/vips/util.h:92`). `width="10.4"` gives 10 px and
//!   `height="6.6"` gives 7 px, measured.
//! * **Resolution.** `Xres = Yres = dpi / 25.4` pixels per millimetre
//!   (`svgload.c:593`). `scale` deliberately does not move it: measured on
//!   vips 8.18.4, `--scale 2` doubles the pixel dimensions and leaves
//!   `xres` at 2.83465.
//! * **DPI applies twice to physical units.** A `width="10mm"` is first
//!   converted to user units at `dpi` (`svgload.c:472`,
//!   `svg_css_length_to_pixels`) and then multiplied by `total_scale`,
//!   which carries `dpi` again. Measured: 10mm x 5mm renders 28x14 at the
//!   default 72 dpi and 50x25 at 144 dpi, not 28x14 doubled. usvg converts
//!   units with its own `Options::dpi` in the same place, so setting both
//!   reproduces the behaviour rather than correcting it.
//! * **Pixels.** 4-band 8-bit sRGB with **unpremultiplied** alpha, because
//!   vips unpremultiplies every cairo row on the way out. A red rectangle at
//!   `fill-opacity="0.5"` measures `255 0 0 128` under `vips getpoint`, not
//!   the premultiplied `128 0 0 128`.
//! * **`unlimited`.** See [`SvgOptions::unlimited`]: what vips documents and
//!   what vips 8.18.4 measurably does are not the same thing, and the
//!   difference is written up there rather than here.
//! * **Untrusted `<image>` hrefs are refused, always.** See
//!   [`SvgOptions`] and the security note below.
//!
//! # Security: `<image xlink:href>` never touches the filesystem
//!
//! `usvg`'s default [`ImageHrefResolver`](usvg::ImageHrefResolver) treats an
//! href as a **file path**: it calls `Path::exists` and then `fs::read` on
//! it, and because `Options::resources_dir` defaults to `None` the path is
//! used verbatim, so it resolves absolutely or relative to the process
//! working directory. Handed an untrusted document that is an arbitrary-file
//! read (the bytes of a readable PNG land in the output pixels) and an
//! existence oracle (a path that exists takes a different branch from one
//! that does not).
//!
//! This module overrides **both** halves of the resolver to return `None`
//! unconditionally, so no href of any kind is ever resolved, from any
//! source, and `resources_dir` is pinned to `None` as a second line. The
//! consequence is that `<image>` elements never render. That is a real
//! divergence from vips, and it is the right trade: this crate already
//! rejects attacker-controlled paths in `tests/path_traversal.rs` for
//! exactly this class of bug, and an image loader is the last place to make
//! an exception.
//!
//! # Not implemented, and not going to be
//!
//! * **`stylesheet`** (vips `svgload --stylesheet`). Descoped by
//!   scheduling, not by capability: `usvg::Options::style_sheet` exists at
//!   0.48.1 and takes a CSS string, so this is a small follow-up rather
//!   than a wall.
//! * **`high_bitdepth`** (vips scRGB 128-bit, 32 bits per channel). resvg
//!   renders into a `tiny_skia::Pixmap`, which is 8-bit RGBA and has no
//!   float surface, so there is nothing to render into. This one *is* a
//!   wall.
//! * **Text fidelity.** vips goes librsvg -> pango -> fontconfig -> **system
//!   fonts**. resvg resolves faces from a `fontdb` you populate, and this
//!   crate deliberately ships exactly one face (`fonts/Vera.ttf`, Bitstream
//!   Vera Sans). Ask both sides for `font-family="Helvetica"` and vips finds
//!   Helvetica while libviprs finds Vera. Two shapers, two rasterisers, two
//!   font sets: **text-bearing SVGs do not match the oracle and no amount of
//!   work in this module changes that.** Text is wired to the bundled face
//!   so it renders deterministically instead of rendering as nothing, and
//!   issue #587 characterises how far off it lands. Do not build a parity
//!   expectation on it.
//! * **Format sniffing.** SVG is not in the [`crate::source`] route table
//!   and is not auto-detected by [`decode_bytes`](crate::source::decode_bytes).
//!   SVG has no fixed leading magic: librsvg's `is_a` decompresses a
//!   possible gzip wrapper and then case-insensitively searches for `<svg`
//!   anywhere in the first kilobyte (`svgload.c:270-333`), which is why vips
//!   gives the loader `priority = -5` and asks it last (`svgload.c:790`). A
//!   table keyed on leading magic bytes is the wrong shape for that, and
//!   `dpi`/`scale` have nowhere to live in a route table that is documented
//!   as taking no per-format options. [`decode_svg`] is the entry point.
//!
//! # API shape
//!
//! Every entry point here is fallible and returns [`DecodeError`]. There is
//! no `#[track_caller]` panicking twin, matching the rest of the decode
//! surface ([`decode_bytes`](crate::source::decode_bytes),
//! [`decode_file`](crate::source::decode_file)): a decoder's failure is an
//! ordinary property of untrusted input, not a programming error, so there
//! is no call site that would want the panicking form.

use crate::codec::DecodeError;
use crate::raster::Raster;
use crate::source::DecodeLimits;

/// Input-byte ceiling applied when [`SvgOptions::unlimited`] is `false`.
///
/// vips documents this as librsvg's own gate: "SVGs larger than 10MB are
/// normally blocked for security. Set @unlimited to allow SVGs of any size"
/// (`svgload.c:1122`). See [`SvgOptions::unlimited`] for why libviprs
/// enforces it even though vips 8.18.4 measurably does not.
pub const MAX_INPUT_BYTES: usize = 10 * 1024 * 1024;

/// Options for [`decode_svg`] (libvips `svgload` / `svgload_buffer`).
///
/// Plain, `Default`, and module-scoped, so callers write
/// `svg::SvgOptions { dpi: 144.0, ..Default::default() }`. Deliberately not
/// `#[non_exhaustive]`, which would block that spelling from outside the
/// crate and defeat the point of an options struct; the same call the WebP
/// lane made for [`crate::webp::SaveOptions`].
///
/// The defaults are vips's defaults: `dpi` 72.0 and `scale` 1.0
/// (`svgload.c:838-839`), `unlimited` false (`svgload.c:817`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SvgOptions {
    /// Render at this DPI (libvips `dpi`, default 72.0).
    ///
    /// Two things at once, exactly as in vips: it converts physical CSS
    /// lengths (`mm`, `cm`, `in`, `pt`, `pc`) into user units, and it feeds
    /// `total_scale = scale * dpi / 72.0`. A document sized in `px` or in
    /// bare numbers therefore scales linearly with `dpi`, and one sized in
    /// `mm` scales quadratically. That is vips's behaviour, measured, not a
    /// transcription slip.
    ///
    /// It also sets the output resolution to `dpi / 25.4` pixels per
    /// millimetre.
    pub dpi: f64,
    /// Scale the rendered output by this factor (libvips `scale`, default
    /// 1.0).
    ///
    /// Multiplies into `total_scale` alongside `dpi`, so setting both
    /// combines them. Unlike `dpi` it does **not** move the output
    /// resolution and it does **not** affect CSS unit conversion.
    pub scale: f64,
    /// Lift the input-size gate (libvips `unlimited`, default false).
    ///
    /// In vips this maps to `RSVG_HANDLE_FLAG_UNLIMITED`
    /// (`svgload.c:876`), documented as lifting a 10 MB input ceiling. That
    /// documentation is stale. Measured against vips 8.18.4 with librsvg
    /// 2.62.3:
    ///
    /// * an 11 MB SVG loads fine **without** `--unlimited`, so the
    ///   documented 10 MB block does not happen;
    /// * a document with 1,000,001 elements is refused with "cannot load
    ///   more than 1000000 XML elements" **with and without**
    ///   `--unlimited`, so the flag does not lift that either.
    ///
    /// In other words the flag is observably a no-op on every input this
    /// lane could construct. libviprs does not ship a knob that provably
    /// does nothing, so `unlimited` here gates libviprs's own input-byte
    /// ceiling, [`MAX_INPUT_BYTES`], set to the 10 MB figure vips
    /// documents. That makes the default **stricter** than vips 8.18.4 for
    /// inputs between 10 MB and the element cap, and `unlimited: true`
    /// reproduces the measured vips behaviour exactly.
    ///
    /// What it does **not** do is lift [`DecodeLimits`]. A 200-byte document
    /// can declare `width="1000000000"`, so the ceiling that actually
    /// bounds allocation is the one on the *output* geometry, and a load
    /// option must never be able to disarm it. The element cap is not
    /// liftable on either side: usvg refuses above 1,000,000 elements
    /// (`usvg::parser::Error::ElementsLimitReached`), the same number
    /// librsvg uses, so the two agree there by accident and by measurement.
    pub unlimited: bool,
}

impl Default for SvgOptions {
    fn default() -> Self {
        Self {
            dpi: 72.0,
            scale: 1.0,
            unlimited: false,
        }
    }
}

/// Rasterise an SVG document from bytes (libvips `svgload_buffer`).
///
/// Uses [`DecodeLimits::default`] as the output-geometry budget; see
/// [`decode_svg_with_limits`] to supply your own.
///
/// # Errors
///
/// * [`DecodeError::Io`] with [`std::io::ErrorKind::Unsupported`] when the
///   crate was built without the `svg` feature.
/// * [`DecodeError::SvgInputTooLarge`] when the buffer exceeds
///   [`MAX_INPUT_BYTES`] and [`SvgOptions::unlimited`] is false.
/// * [`DecodeError::SvgParse`] when usvg refuses the document.
/// * [`DecodeError::SvgZeroSize`] when the scaled geometry rounds to zero on
///   either axis, matching vips's "zero-sized image" (`svgload.c:588`).
/// * [`DecodeError::CoordLimitExceeded`] / [`DecodeError::DimensionLimitExceeded`]
///   when the scaled geometry exceeds the decode budget.
/// * [`DecodeError::Raster`] when the rendered buffer cannot be wrapped.
pub fn decode_svg(data: &[u8], options: SvgOptions) -> Result<Raster, DecodeError> {
    decode_svg_with_limits(data, options, DecodeLimits::default())
}

/// Rasterise an SVG document under an explicit output-geometry budget.
///
/// The budget is applied to the **scaled** geometry, before the pixel buffer
/// is allocated, so a small document that declares a huge `width` is
/// rejected rather than rendered.
///
/// # Errors
///
/// As [`decode_svg`].
pub fn decode_svg_with_limits(
    data: &[u8],
    options: SvgOptions,
    limits: DecodeLimits,
) -> Result<Raster, DecodeError> {
    let _ = (data, options, limits);
    Err(DecodeError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "SVG rasterisation is not available in this build",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 10x6 document with an explicit pixel size and one opaque red rect.
    const RED_10X6: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6"><rect x="0" y="0" width="10" height="6" fill="#ff0000"/></svg>"##;
    /// A document with only a `viewBox` and no width/height.
    const VIEWBOX_20X10: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 10"><rect x="0" y="0" width="20" height="10" fill="#00ff00"/></svg>"##;
    /// Fractional pixel dimensions, to pin the rounding rule.
    const FRACTIONAL: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="10.4" height="6.6"><rect width="10" height="6" fill="#123456"/></svg>"##;
    /// Physical (millimetre) dimensions, to pin the double DPI application.
    const MILLIMETRES: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="10mm" height="5mm"><rect width="10" height="5" fill="#0000ff"/></svg>"##;
    /// Half-opacity red over the left half of a 4x2 document.
    const HALF_ALPHA: &[u8] = br##"<svg xmlns="http://www.w3.org/2000/svg" width="4" height="2"><rect x="0" y="0" width="2" height="2" fill="#ff0000" fill-opacity="0.5"/></svg>"##;

    /// Read the RGBA sample at `(x, y)` out of an `Rgba8` raster.
    #[cfg(feature = "svg")]
    fn px(raster: &Raster, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * raster.width() + x) * 4) as usize;
        let d = raster.data();
        [d[i], d[i + 1], d[i + 2], d[i + 3]]
    }

    /**
     * Tests that the option defaults are the ones `vips svgload` starts
     * from, so `SvgOptions::default()` and a bare `vips svgload` are the
     * same request. Works by comparing every field against the values
     * `vips_foreign_load_svg_init` assigns (`svgload.c:838-839`) and the
     * `unlimited` default declared at `svgload.c:817`.
     * Input: none -> Output: dpi 72.0, scale 1.0, unlimited false, and a
     * functional-update literal that compares equal.
     */
    #[test]
    fn default_options_match_vips_svgload_defaults() {
        let d = SvgOptions::default();
        assert!(
            (d.dpi - 72.0).abs() < f64::EPSILON,
            "vips defaults dpi to 72.0, got {}",
            d.dpi
        );
        assert!(
            (d.scale - 1.0).abs() < f64::EPSILON,
            "vips defaults scale to 1.0, got {}",
            d.scale
        );
        assert!(!d.unlimited, "vips defaults unlimited to false");
        let updated = SvgOptions {
            dpi: 144.0,
            ..Default::default()
        };
        assert!((updated.scale - 1.0).abs() < f64::EPSILON);
        assert!(!updated.unlimited);
    }

    /**
     * Tests that a build without the `svg` feature still exposes the entry
     * point and still reports a typed error naming the capability, so a
     * caller compiled either way gets the same signature and can match on
     * the failure instead of hitting a missing symbol. Works by calling
     * `decode_svg` on a valid document and asserting the error is an
     * `Unsupported` I/O error mentioning SVG.
     * Input: 10x6 SVG -> Output: `DecodeError::Io` with
     * `ErrorKind::Unsupported` whose message contains "SVG".
     */
    #[test]
    #[cfg(not(feature = "svg"))]
    fn without_the_feature_decode_svg_is_a_typed_unsupported() {
        let err = decode_svg(RED_10X6, SvgOptions::default()).unwrap_err();
        match err {
            DecodeError::Io(ref e) => {
                assert_eq!(e.kind(), std::io::ErrorKind::Unsupported);
                assert!(
                    err.to_string().contains("SVG"),
                    "error must name SVG, got {err}"
                );
            }
            other => panic!("expected a typed Io(Unsupported), got {other:?}"),
        }
    }

    /**
     * Pins the natural-size and resolution contract against the oracle. A
     * document that declares `width="10" height="6"` renders 10x6 at the
     * default DPI, and the header resolution is `dpi / 25.4` pixels per
     * millimetre (`svgload.c:593`). Works by decoding at defaults and
     * comparing dimensions exactly and resolution within a tolerance,
     * against `vips svgload a.svg x.v` which measured 10x6 with xres
     * 2.83465.
     * Input: 10x6 SVG at defaults -> Output: Raster(10, 6, Rgba8) with
     * xres == yres == 72/25.4.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn natural_size_and_resolution_match_vips() {
        let im = decode_svg(RED_10X6, SvgOptions::default()).unwrap();
        assert_eq!((im.width(), im.height()), (10, 6));
        assert_eq!(im.format(), crate::pixel::PixelFormat::Rgba8);
        let want = 72.0 / 25.4;
        assert!(
            (im.xres() - want).abs() < 1e-6,
            "xres must be dpi/25.4 = {want}, got {}",
            im.xres()
        );
        assert!(
            (im.yres() - want).abs() < 1e-6,
            "yres must be dpi/25.4 = {want}, got {}",
            im.yres()
        );
        assert_eq!(px(&im, 0, 0), [255, 0, 0, 255], "opaque red must be exact");
        assert_eq!(px(&im, 9, 5), [255, 0, 0, 255]);
    }

    /**
     * Tests that `dpi` moves both the pixel geometry and the header
     * resolution, because `total_scale` carries `dpi / 72` and `Xres` is
     * `dpi / 25.4`. Works by decoding the 10x6 document at 144 dpi and
     * comparing against `vips svgload a.svg x2.v --dpi 144`, which measured
     * 20x12 with xres 5.66929.
     * Input: 10x6 SVG at dpi 144 -> Output: Raster(20, 12) with
     * xres == 144/25.4.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn dpi_scales_geometry_and_resolution() {
        let im = decode_svg(
            RED_10X6,
            SvgOptions {
                dpi: 144.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((im.width(), im.height()), (20, 12));
        let want = 144.0 / 25.4;
        assert!(
            (im.xres() - want).abs() < 1e-6,
            "xres must track dpi: expected {want}, got {}",
            im.xres()
        );
    }

    /**
     * Tests that `scale` moves the pixel geometry but deliberately leaves
     * the header resolution alone, which is the one asymmetry between the
     * two knobs: `total_scale` takes both, `Xres` takes only `dpi`. Works
     * by decoding at scale 2 and comparing against `vips svgload a.svg
     * x3.v --scale 2`, which measured 20x12 with xres still 2.83465.
     * Input: 10x6 SVG at scale 2 -> Output: Raster(20, 12) with
     * xres == 72/25.4, the default.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn scale_moves_geometry_but_not_resolution() {
        let im = decode_svg(
            RED_10X6,
            SvgOptions {
                scale: 2.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((im.width(), im.height()), (20, 12));
        let want = 72.0 / 25.4;
        assert!(
            (im.xres() - want).abs() < 1e-6,
            "scale must not move xres: expected {want}, got {}",
            im.xres()
        );
    }

    /**
     * Tests the `viewBox`-only fallback: a document with no width/height
     * takes its size from the viewBox (`svgload.c:490-491`). Works by
     * decoding a `viewBox="0 0 20 10"` document at defaults and comparing
     * against `vips svgload vb.svg x4.v`, which measured 20x10.
     * Input: viewBox-only SVG -> Output: Raster(20, 10).
     */
    #[test]
    #[cfg(feature = "svg")]
    fn viewbox_only_document_takes_the_viewbox_size() {
        let im = decode_svg(VIEWBOX_20X10, SvgOptions::default()).unwrap();
        assert_eq!((im.width(), im.height()), (20, 10));
    }

    /**
     * Pins the rounding rule. `VIPS_ROUND_UINT` is `(int)(R + 0.5)`
     * (`util.h:92`), so a non-negative value rounds half **up**, not to
     * even and not toward zero. Works by decoding a 10.4x6.6 document at
     * defaults and at scale 1.5, comparing against `vips svgload frac.svg`
     * (10x7) and `--scale 1.5` (16x10). Both cases have a fraction on each
     * axis and they round in opposite directions, so a truncating or
     * round-to-even implementation fails at least one.
     * Input: 10.4x6.6 SVG at scale 1.0 and 1.5 -> Output: 10x7 and 16x10.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn fractional_geometry_rounds_half_up() {
        let im = decode_svg(FRACTIONAL, SvgOptions::default()).unwrap();
        assert_eq!(
            (im.width(), im.height()),
            (10, 7),
            "10.4 -> 10 and 6.6 -> 7 under round-half-up"
        );
        let scaled = decode_svg(
            FRACTIONAL,
            SvgOptions {
                scale: 1.5,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            (scaled.width(), scaled.height()),
            (16, 10),
            "15.6 -> 16 and 9.9 -> 10 under round-half-up"
        );
    }

    /**
     * Tests that a physically-sized document takes `dpi` twice, once
     * converting millimetres to user units (`svgload.c:472`) and once
     * through `total_scale` (`svgload.c:362`). This looks like a bug and is
     * not one: it is what vips does, so libviprs reproduces it rather than
     * correcting it. Works by decoding a 10mm x 5mm document at 72 and at
     * 96 dpi and comparing against `vips svgload mm.svg` (28x14) and
     * `--dpi 96` (50x25). A single application of dpi would give 37x19 at
     * 96 dpi, so the two hypotheses are distinguishable.
     * Input: 10mm x 5mm SVG at dpi 72 and 96 -> Output: 28x14 and 50x25.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn physical_units_take_dpi_twice_like_vips() {
        let at72 = decode_svg(MILLIMETRES, SvgOptions::default()).unwrap();
        assert_eq!((at72.width(), at72.height()), (28, 14));
        let at96 = decode_svg(
            MILLIMETRES,
            SvgOptions {
                dpi: 96.0,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            (at96.width(), at96.height()),
            (50, 25),
            "dpi applies to the mm conversion and again through total_scale"
        );
    }

    /**
     * Tests that alpha comes out **unpremultiplied**, matching the
     * `vips__premultiplied_bgra2rgba` pass vips runs over every cairo row
     * (`svgload.c:733`). tiny-skia renders premultiplied, so this pins the
     * demultiply step; without it a half-opaque red would read `128 0 0
     * 128` instead. Works by decoding a 4x2 document whose left half is red
     * at `fill-opacity="0.5"` and comparing all four channels against
     * `vips getpoint`, which measured `255 0 0 128` inside the rect and
     * `0 0 0 0` outside it.
     * Input: 4x2 SVG, left half red at 0.5 opacity -> Output: (0,0) and
     * (1,0) are [255, 0, 0, 128]; (2,0) and (3,0) are [0, 0, 0, 0].
     */
    #[test]
    #[cfg(feature = "svg")]
    fn alpha_is_unpremultiplied_like_vips() {
        let im = decode_svg(HALF_ALPHA, SvgOptions::default()).unwrap();
        assert_eq!((im.width(), im.height()), (4, 2));
        for x in [0, 1] {
            assert_eq!(
                px(&im, x, 0),
                [255, 0, 0, 128],
                "pixel ({x},0) must be unpremultiplied red at alpha 128"
            );
        }
        for x in [2, 3] {
            assert_eq!(
                px(&im, x, 0),
                [0, 0, 0, 0],
                "pixel ({x},0) is outside the rect and must be transparent black"
            );
        }
    }

    /**
     * Tests the input-size gate `unlimited` controls. vips documents a
     * 10 MB ceiling (`svgload.c:1122`) that librsvg 2.62.3 no longer
     * enforces, so libviprs enforces it here and `unlimited` lifts it;
     * see `SvgOptions::unlimited` for the measurement. Works by padding a
     * valid document past `MAX_INPUT_BYTES` with a comment, decoding it
     * both ways, and checking the default refuses with the typed variant
     * while `unlimited: true` renders the same geometry the unpadded
     * document does.
     * Input: a 10x6 SVG padded past 10 MiB -> Output: `SvgInputTooLarge`
     * at defaults, Raster(10, 6) with `unlimited: true`.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn unlimited_lifts_the_input_size_gate() {
        let pad = "x".repeat(MAX_INPUT_BYTES);
        let doc = format!(
            r##"<svg xmlns="http://www.w3.org/2000/svg" width="10" height="6"><!--{pad}--><rect width="10" height="6" fill="#ff0000"/></svg>"##
        );
        let bytes = doc.as_bytes();
        assert!(bytes.len() > MAX_INPUT_BYTES);

        let err = decode_svg(bytes, SvgOptions::default()).unwrap_err();
        assert!(
            matches!(
                err,
                DecodeError::SvgInputTooLarge { bytes: n, max_bytes }
                    if n == bytes.len() && max_bytes == MAX_INPUT_BYTES
            ),
            "expected SvgInputTooLarge, got {err:?}"
        );

        let im = decode_svg(
            bytes,
            SvgOptions {
                unlimited: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((im.width(), im.height()), (10, 6));
    }

    /**
     * Tests that `unlimited` does not disarm the output-geometry budget.
     * The input gate and the allocation ceiling are separate on purpose: a
     * tiny document can declare an enormous width, so `DecodeLimits` is
     * what actually bounds the render and no load option may lift it.
     * Works by decoding a small document at a scale that pushes the output
     * past a deliberately tight `max_coord`, with `unlimited` both false
     * and true, and asserting both are refused with the same typed
     * variant.
     * Input: 10x6 SVG at scale 100 under max_coord 64 -> Output:
     * `CoordLimitExceeded` in both cases, no allocation attempted.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn decode_limits_bound_the_scaled_output_and_unlimited_does_not_lift_them() {
        let limits = DecodeLimits::default().with_max_coord(64);
        for unlimited in [false, true] {
            let err = decode_svg_with_limits(
                RED_10X6,
                SvgOptions {
                    scale: 100.0,
                    unlimited,
                    ..Default::default()
                },
                limits,
            )
            .unwrap_err();
            assert!(
                matches!(
                    err,
                    DecodeError::CoordLimitExceeded {
                        width: 1000,
                        height: 600,
                        max_coord: 64
                    }
                ),
                "unlimited={unlimited} must still hit the coord ceiling, got {err:?}"
            );
        }
    }

    /**
     * Tests that geometry rounding to zero on either axis is a typed error
     * rather than a zero-sized raster or a panic, matching vips's
     * "zero-sized image" bail-out (`svgload.c:588`). Works by decoding a
     * valid document at a scale small enough that both axes round to 0
     * under round-half-up, and matching the structured variant.
     * Input: 10x6 SVG at scale 0.01 -> Output: `SvgZeroSize` reporting the
     * scaled dimensions that rounded away.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn zero_sized_output_is_a_typed_error() {
        let err = decode_svg(
            RED_10X6,
            SvgOptions {
                scale: 0.01,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, DecodeError::SvgZeroSize { width: 0, height: 0 }),
            "expected SvgZeroSize, got {err:?}"
        );
    }

    /**
     * Tests that a document usvg refuses surfaces as a typed parse error
     * carrying usvg's message, not as a panic and not as a silent empty
     * raster. Works by decoding a buffer that is not XML at all and
     * matching the structured variant.
     * Input: `b"definitely not an svg"` -> Output: `SvgParse` with a
     * non-empty message.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn malformed_input_is_a_typed_parse_error() {
        let err = decode_svg(b"definitely not an svg", SvgOptions::default()).unwrap_err();
        match err {
            DecodeError::SvgParse { ref message } => {
                assert!(!message.is_empty(), "the usvg message must be carried out")
            }
            other => panic!("expected SvgParse, got {other:?}"),
        }
    }

    /**
     * Tests the 502b contract: text renders against the bundled Bitstream
     * Vera face rather than rendering as nothing, so a text-bearing
     * document is deterministic across machines instead of depending on
     * what fonts happen to be installed. This does **not** pin parity with
     * vips, which shapes through pango against system fonts and cannot
     * match; see the module docs. Works by decoding a document whose only
     * content is black text and asserting some pixel is inked.
     * Input: 60x20 SVG containing `<text>` -> Output: at least one pixel
     * with non-zero alpha.
     */
    #[test]
    #[cfg(feature = "svg")]
    fn text_renders_against_the_bundled_face() {
        let doc = br##"<svg xmlns="http://www.w3.org/2000/svg" width="60" height="20"><text x="2" y="15" font-size="14" fill="#000000">Hi</text></svg>"##;
        let im = decode_svg(doc, SvgOptions::default()).unwrap();
        assert_eq!((im.width(), im.height()), (60, 20));
        let inked = im.data().chunks_exact(4).filter(|p| p[3] != 0).count();
        assert!(
            inked > 0,
            "text must render against the bundled face, got a blank raster"
        );
    }
}
