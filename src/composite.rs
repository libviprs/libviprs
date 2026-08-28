//! Alpha compositing ported from libvips (`vips_composite2`).
//!
//! This module is the seventh batch of the libvips operation surface
//! required by the ported integration tests (after [`crate::bands`],
//! [`crate::arithmetic`], [`crate::extract`], [`crate::conversion`],
//! [`crate::draw`], and [`crate::histogram`]): Porter-Duff compositing and
//! the PDF blend modes, both the separable set (`multiply`, `screen`, ...)
//! and the non-separable set (`hue`, `saturation`, `colour`,
//! `luminosity`). Following the established convention the operation
//! exists in two forms:
//!
//! * a fallible [`Raster::try_composite2`] returning
//!   `Result<Raster, CompositeError>` with typed errors for mismatched
//!   geometry and band counts; and
//! * the panicking conveniences [`Raster::composite`] /
//!   [`Raster::composite2`] matching the ported-test call surface
//!   (`base.composite(&overlay, CompositeMode::Over)`) exactly, delegating
//!   to the `try_` form.
//!
//! # Semantics
//!
//! `base.composite2(&overlay, mode)` composites `overlay` (the source in
//! Porter-Duff terms) onto `base` (the backdrop), exactly as
//! `vips_composite2(base, overlay, mode)`.
//!
//! * **Alpha.** An input with 2 or 4 bands treats its last band as alpha;
//!   1- and 3-band inputs are opaque, matching how libvips detects alpha
//!   in `composite`. The output always carries alpha: 1 colour band
//!   composites to a 2-band grey+alpha raster
//!   ([`PixelFormat::Multi8`]`(2)` / `Multi16(2)`), 3 colour bands to
//!   RGBA.
//! * **Bands.** Both inputs must have the same number of colour
//!   (non-alpha) bands, either 1 or 3. The non-separable modes
//!   ([`CompositeMode::Hue`], [`CompositeMode::Saturation`],
//!   [`CompositeMode::Colour`], [`CompositeMode::Luminosity`]) operate on
//!   RGB triples per the PDF specification and require 3 colour bands.
//! * **Depth and scale.** The output container is the deeper of the two
//!   inputs. Each input is normalised by the max its compositing-space
//!   *interpretation* implies, not by its raw storage depth — libvips
//!   derives `max_band` from the interpretation
//!   (`vips_interpretation_max_alpha`) after running `formatalike`, never
//!   from the sample format. The genuine 16-bit spaces
//!   ([`Interpretation::Rgb16`] / [`Interpretation::Grey16`]) read on the
//!   0..65535 scale; every other space reads on 0..255. The decision keys
//!   on the raster's *resolved* interpretation ([`Raster::interpretation`]),
//!   which — exactly like libvips' `vips_image_guess_interpretation` —
//!   infers `Rgb16` / `Grey16` for an untagged 16-bit buffer, so a genuine
//!   16-bit image built through [`Raster::new`] is honoured on the 65535
//!   scale without an explicit tag (this is the primary #289 path). A
//!   *promoted* 16-bit container is the counterpart case: the constant ops
//!   (`add_const`, `mul_const`, `pow_const`, `add_vec`, ...) widen an 8-bit
//!   input into a 16-bit buffer whose samples stay numerically on 0..255, and
//!   they stamp that output with the *source* interpretation (`Srgb` / `Bw`,
//!   see [`crate::arithmetic`]) precisely so it resolves to a non-genuine-16
//!   space and reads here on 0..255 rather than being mistaken for a genuine
//!   16-bit layer and washed out. The genuine-16 decision is additionally
//!   gated on the actual storage depth
//!   (`bytes_per_channel() == 2`): an interpretation tag is advisory and can
//!   disagree with the bytes (the copy builder accepts any tag), so an
//!   8-bit buffer mislabelled `Rgb16` / `Grey16` is *not* read on the 65535
//!   scale — doing so would drive the read/write scale away from the actual
//!   samples and corrupt the output. Any input that is not a genuine 16-bit
//!   space falls back to the joint depth rule (65535 only when the whole
//!   pipeline is 16-bit, else 255). Keying on the interpretation rather than
//!   the depth lets a genuine `Rgb16` layer blend against an 8-bit layer on
//!   compatible 0..1 scales instead of at 257:1, and lets a genuine 16-bit
//!   alpha survive rather than saturating to opaque against a 255 ceiling.
//!   Alphas clamp to `0..1`; colour
//!   values pass through the Porter-Duff arithmetic unclamped (numeric
//!   values above the scale survive, as in libvips float) and clamp to
//!   `0..1` only where the PDF blend functions require it. Results are
//!   blended premultiplied in `f64`, unpremultiplied, and written back on
//!   the output interpretation's max (genuine 16-bit whenever a genuine-16
//!   input is present, else the joint rule); integer containers round to
//!   nearest and clamp to the container range, so every integer mode is
//!   exact to the output quantisation. Float rasters ([`PixelFormat::RgbaF32`],
//!   [`PixelFormat::FloatF32`]) composite on the same 0..255 numeric
//!   scale: libvips derives `max_band` from the compositing-space
//!   *interpretation* (sRGB unless a 16-bit-tagged input is present),
//!   never from the storage depth
//!   (`vips_composite_base_max_band` via
//!   `vips_interpretation_max_alpha`), so a float image cast from 8-bit
//!   keeps its 0..255 colour values and its 0..255 alpha. Float
//!   write-back follows the float instantiation of
//!   `vips_combine_pixels` (numeric limits 0, 0 meaning "no limit"):
//!   no rounding and no clamping, so fractional, negative, and
//!   beyond-scale (HDR) samples round-trip exactly at `f32` precision.
//! * **Metadata.** The result carries the base image's metadata
//!   (resolution, offsets, orientation, and attached fields), like libvips
//!   which copies the header from the first input. The colour
//!   interpretation is the one exception: when a genuine-16 input drove the
//!   write-back onto the 0..65535 scale *into an integer container*, the
//!   result is stamped `Rgb16` / `Grey16` so its saved metadata matches the
//!   samples and a re-composite reads it on the same scale, instead of
//!   inheriting the base's 8-bit tag and re-introducing the 255 ceiling;
//!   otherwise it keeps the base's interpretation unchanged. The stamp and
//!   the write-back scale are driven by the same `out_is_genuine16` decision,
//!   so for the integer output the tag and the samples agree. This is a
//!   best-effort alignment, not a global invariant: an interpretation tag is
//!   advisory and the public API can still construct a raster whose tag
//!   disagrees with its samples (e.g. the copy builder accepts any tag, or a
//!   float output keeps a base tag that never described a float space), so
//!   downstream code must not assume the tag and the samples can never
//!   diverge — it is exactly why the genuine-16 read is gated on the 2-byte
//!   storage depth (see [`is_genuine_16bit`]) rather than trusting the tag
//!   alone.
//!
//! # Blend mode table
//!
//! | Modes | Family |
//! |---|---|
//! | `Clear`, `Source`, `Over`, `In`, `Out`, `Atop`, `Dest`, `DestOver`, `DestIn`, `DestOut`, `DestAtop`, `Xor`, `Add`, `Saturate` | Porter-Duff |
//! | `Multiply`, `Screen`, `Overlay`, `Darken`, `Lighten`, `ColourDodge`, `ColourBurn`, `HardLight`, `SoftLight`, `Difference`, `Exclusion` | PDF separable |
//! | `Hue`, `Saturation`, `Colour`, `Luminosity` | PDF non-separable |
//!
//! The Porter-Duff and separable set matches libvips `VipsBlendMode`
//! one-to-one; the four non-separable modes extend it (libvips stops at
//! `exclusion`) because the ported conversion suite exercises them.

use crate::conversion::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
use thiserror::Error;

/// A compositing operator for [`Raster::composite2`]: the Porter-Duff
/// operators plus the PDF separable and non-separable blend modes
/// (libvips `VipsBlendMode`, extended with the non-separable four).
///
/// In every description below the *source* is the overlay argument and
/// the *backdrop* is the base image the method is called on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CompositeMode {
    /// Both source and backdrop are discarded; the result is fully
    /// transparent black.
    Clear,
    /// The source only; the backdrop is discarded.
    Source,
    /// The source over the backdrop (the classic alpha blend).
    Over,
    /// The part of the source inside the backdrop; backdrop discarded.
    In,
    /// The part of the source outside the backdrop; backdrop discarded.
    Out,
    /// The source where the backdrop is opaque, backdrop elsewhere.
    Atop,
    /// The backdrop only; the source is discarded.
    Dest,
    /// The backdrop over the source.
    DestOver,
    /// The part of the backdrop inside the source; source discarded.
    DestIn,
    /// The part of the backdrop outside the source; source discarded.
    DestOut,
    /// The backdrop where the source is opaque, source elsewhere.
    DestAtop,
    /// Source and backdrop where they do not overlap.
    Xor,
    /// Source plus backdrop, clamped (additive blend).
    Add,
    /// Source limited to the transparency left by the backdrop, plus the
    /// backdrop (Porter-Duff saturate).
    Saturate,
    /// PDF multiply: darkens by multiplying backdrop and source.
    Multiply,
    /// PDF screen: brightens by inverse-multiplying.
    Screen,
    /// PDF overlay: multiply or screen depending on the backdrop.
    Overlay,
    /// PDF darken: per-channel minimum.
    Darken,
    /// PDF lighten: per-channel maximum.
    Lighten,
    /// PDF colour dodge: brightens the backdrop toward the source.
    ColourDodge,
    /// PDF colour burn: darkens the backdrop toward the source.
    ColourBurn,
    /// PDF hard light: multiply or screen depending on the source.
    HardLight,
    /// PDF soft light: a softened hard light.
    SoftLight,
    /// PDF difference: absolute per-channel difference.
    Difference,
    /// PDF exclusion: a lower-contrast difference.
    Exclusion,
    /// PDF hue (non-separable): source hue with backdrop saturation and
    /// luminosity.
    Hue,
    /// PDF saturation (non-separable): source saturation with backdrop
    /// hue and luminosity.
    Saturation,
    /// PDF colour (non-separable): source hue and saturation with
    /// backdrop luminosity.
    Colour,
    /// PDF luminosity (non-separable): source luminosity with backdrop
    /// hue and saturation.
    Luminosity,
}

impl CompositeMode {
    /// Whether this is one of the four PDF non-separable modes, which
    /// blend RGB triples rather than independent channels.
    fn is_non_separable(self) -> bool {
        matches!(
            self,
            Self::Hue | Self::Saturation | Self::Colour | Self::Luminosity
        )
    }

    /// Whether this is a PDF blend mode (separable or non-separable)
    /// rather than a plain Porter-Duff operator.
    fn is_pdf_blend(self) -> bool {
        !matches!(
            self,
            Self::Clear
                | Self::Source
                | Self::Over
                | Self::In
                | Self::Out
                | Self::Atop
                | Self::Dest
                | Self::DestOver
                | Self::DestIn
                | Self::DestOut
                | Self::DestAtop
                | Self::Xor
                | Self::Add
                | Self::Saturate
        )
    }
}

/// Errors from [`Raster::try_composite2`].
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CompositeError {
    #[error(
        "composite: images differ in size: base {base_w}x{base_h}, overlay {overlay_w}x{overlay_h}"
    )]
    DimensionMismatch {
        base_w: u32,
        base_h: u32,
        overlay_w: u32,
        overlay_h: u32,
    },
    #[error(
        "composite: images differ in colour band count: base has {base}, overlay has {overlay}"
    )]
    BandMismatch { base: usize, overlay: usize },
    #[error(
        "composite: {bands} bands unsupported; inputs must have 1 or 3 colour bands (plus optional alpha)"
    )]
    TooManyBands { bands: usize },
    #[error(
        "composite: non-separable mode {mode:?} needs 3 colour bands (RGB), input has {colour_bands}"
    )]
    NonSeparableNeedsRgb {
        mode: CompositeMode,
        colour_bands: usize,
    },
    #[error("composite: raster error: {0}")]
    Raster(#[from] RasterError),
}

/// Unwrap a composite result, panicking with the operation name.
#[track_caller]
fn expect_composite(r: Result<Raster, CompositeError>) -> Raster {
    match r {
        Ok(v) => v,
        Err(e) => panic!("composite: {e}"),
    }
}

/// Colour band count and alpha presence for a composite input: bands 2
/// and 4 carry alpha in the last band, 1 and 3 are opaque.
fn colour_bands(format: PixelFormat) -> Result<(usize, bool), CompositeError> {
    match format.channels() {
        1 => Ok((1, false)),
        2 => Ok((1, true)),
        3 => Ok((3, false)),
        4 => Ok((3, true)),
        bands => Err(CompositeError::TooManyBands { bands }),
    }
}

/// Read the flat `i`-th raw sample (not yet normalised: the caller divides
/// by the input's own interpretation-derived maximum, so each layer reads on
/// the scale its compositing space implies).
#[inline]
fn read_raw(data: &[u8], bpc: usize, i: usize) -> f64 {
    match bpc {
        1 => data[i] as f64,
        2 => u16::from_ne_bytes([data[2 * i], data[2 * i + 1]]) as f64,
        // Float: the raw sample, exact in f64. The caller normalises by
        // the same interpretation-derived scale as the integer depths
        // (libvips `max_band` comes from the compositing space, not the
        // storage depth), so a float raster cast from 8-bit reads its
        // 0..255 numeric values unchanged.
        _ => f32::from_ne_bytes([
            data[4 * i],
            data[4 * i + 1],
            data[4 * i + 2],
            data[4 * i + 3],
        ]) as f64,
    }
}

/// Write a scale-normalised sample at the given depth: the value is
/// multiplied back onto its numeric scale. Integer containers round to
/// nearest and clamp to the container range; float containers store the
/// product as-is (no rounding, no clamping), matching the float
/// instantiation of libvips `vips_combine_pixels`, whose numeric limits
/// of `0, 0` mean "no limit". Fractional, negative, and beyond-scale
/// (HDR) float samples therefore survive exactly at `f32` precision.
#[inline]
fn write_scaled(data: &mut [u8], bpc: usize, i: usize, v: f64, scale: f64) {
    match bpc {
        1 => data[i] = (v * scale).round().clamp(0.0, 255.0) as u8,
        2 => {
            let b = ((v * scale).round().clamp(0.0, 65535.0) as u16).to_ne_bytes();
            data[2 * i] = b[0];
            data[2 * i + 1] = b[1];
        }
        _ => {
            let b = ((v * scale) as f32).to_ne_bytes();
            data[4 * i..4 * i + 4].copy_from_slice(&b);
        }
    }
}

/// Porter-Duff source/backdrop factors for the given source (`sa`) and
/// backdrop (`ba`) alphas. Returns `None` for the PDF blend modes, which
/// use the blend formula instead.
fn porter_duff_factors(mode: CompositeMode, sa: f64, ba: f64) -> Option<(f64, f64)> {
    Some(match mode {
        CompositeMode::Clear => (0.0, 0.0),
        CompositeMode::Source => (1.0, 0.0),
        CompositeMode::Over => (1.0, 1.0 - sa),
        CompositeMode::In => (ba, 0.0),
        CompositeMode::Out => (1.0 - ba, 0.0),
        CompositeMode::Atop => (ba, 1.0 - sa),
        CompositeMode::Dest => (0.0, 1.0),
        CompositeMode::DestOver => (1.0 - ba, 1.0),
        CompositeMode::DestIn => (0.0, sa),
        CompositeMode::DestOut => (0.0, 1.0 - sa),
        CompositeMode::DestAtop => (1.0 - ba, sa),
        CompositeMode::Xor => (1.0 - ba, 1.0 - sa),
        CompositeMode::Add => (1.0, 1.0),
        CompositeMode::Saturate => {
            let fa = if sa > 0.0 {
                (1.0 - ba).min(sa) / sa
            } else {
                0.0
            };
            (fa, 1.0)
        }
        _ => return None,
    })
}

/// The PDF separable blend function `B(cb, cs)` on unpremultiplied
/// channel values in `0..=1`.
fn separable_blend(mode: CompositeMode, cb: f64, cs: f64) -> f64 {
    match mode {
        CompositeMode::Multiply => cb * cs,
        CompositeMode::Screen => cb + cs - cb * cs,
        CompositeMode::Overlay => separable_blend(CompositeMode::HardLight, cs, cb),
        CompositeMode::Darken => cb.min(cs),
        CompositeMode::Lighten => cb.max(cs),
        CompositeMode::ColourDodge => {
            if cb <= 0.0 {
                0.0
            } else if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        CompositeMode::ColourBurn => {
            if cb >= 1.0 {
                1.0
            } else if cs <= 0.0 {
                0.0
            } else {
                1.0 - ((1.0 - cb) / cs).min(1.0)
            }
        }
        CompositeMode::HardLight => {
            if cs <= 0.5 {
                2.0 * cs * cb
            } else {
                1.0 - 2.0 * (1.0 - cs) * (1.0 - cb)
            }
        }
        CompositeMode::SoftLight => {
            if cs <= 0.5 {
                cb - (1.0 - 2.0 * cs) * cb * (1.0 - cb)
            } else {
                let d = if cb <= 0.25 {
                    ((16.0 * cb - 12.0) * cb + 4.0) * cb
                } else {
                    cb.sqrt()
                };
                cb + (2.0 * cs - 1.0) * (d - cb)
            }
        }
        CompositeMode::Difference => (cb - cs).abs(),
        CompositeMode::Exclusion => cb + cs - 2.0 * cb * cs,
        _ => unreachable!("separable_blend called with non-blend mode {mode:?}"),
    }
}

/// PDF luminosity of an RGB triple (the Rec. 601-style weights the PDF
/// specification and libvips' non-separable helpers use).
fn lum(c: [f64; 3]) -> f64 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// PDF `ClipColor`: pull out-of-range channels back toward the
/// luminosity so the triple stays within `0..=1`.
fn clip_colour(mut c: [f64; 3]) -> [f64; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    if n < 0.0 {
        for v in &mut c {
            *v = l + (*v - l) * l / (l - n);
        }
    }
    if x > 1.0 {
        for v in &mut c {
            *v = l + (*v - l) * (1.0 - l) / (x - l);
        }
    }
    c
}

/// PDF `SetLum`: shift the triple to the target luminosity, clipping.
fn set_lum(mut c: [f64; 3], l: f64) -> [f64; 3] {
    let d = l - lum(c);
    for v in &mut c {
        *v += d;
    }
    clip_colour(c)
}

/// PDF `Sat`: saturation (max minus min) of a triple.
fn sat(c: [f64; 3]) -> f64 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

/// PDF `SetSat`: rescale the triple to the target saturation, keeping
/// the channel order and zeroing the minimum.
fn set_sat(c: [f64; 3], s: f64) -> [f64; 3] {
    // Rank the channel indices: min, mid, max.
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| c[a].partial_cmp(&c[b]).unwrap_or(std::cmp::Ordering::Equal));
    let (imin, imid, imax) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0; 3];
    if c[imax] > c[imin] {
        out[imid] = (c[imid] - c[imin]) * s / (c[imax] - c[imin]);
        out[imax] = s;
    }
    out
}

/// The PDF non-separable blend function on unpremultiplied RGB triples.
fn non_separable_blend(mode: CompositeMode, cb: [f64; 3], cs: [f64; 3]) -> [f64; 3] {
    match mode {
        CompositeMode::Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
        CompositeMode::Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
        CompositeMode::Colour => set_lum(cs, lum(cb)),
        CompositeMode::Luminosity => set_lum(cb, lum(cs)),
        _ => unreachable!("non_separable_blend called with separable mode {mode:?}"),
    }
}

/// Whether a composite input occupies a genuine 16-bit compositing space
/// (0..65535), as opposed to the 0..255 sRGB / greyscale / promoted-container
/// numeric scale.
///
/// This mirrors libvips, where `max_band` comes from the compositing-space
/// interpretation (`vips_interpretation_max_alpha`) and a USHORT image's
/// interpretation is guessed as [`Interpretation::Rgb16`] /
/// [`Interpretation::Grey16`]. We therefore key on the raster's *resolved*
/// interpretation ([`Raster::interpretation`], which infers the genuine-16
/// space for an untagged 16-bit format exactly as `vips_image_guess_
/// interpretation` does) rather than the raw stored `meta.interpretation`
/// tag: a genuine 16-bit buffer built through [`Raster::new`] (the primary
/// #289 path — e.g. decoding a 16-bit PNG) is honoured without an explicit
/// tag, and two rasters indistinguishable across the public API composite
/// identically.
///
/// A *promoted* 16-bit container — an 8-bit input widened by a constant op
/// (`add_const` / `mul_const` / `pow_const` / `add_vec`) whose samples remain
/// numerically 0..255 — is the case this must *not* treat as genuine-16.
/// Those ops stamp the promoted output with the source interpretation (`Srgb`
/// / `Bw`, see [`crate::arithmetic`]), so [`Raster::interpretation`] resolves
/// it to a non-genuine-16 space and this returns `false` for it, keeping a
/// fully-opaque promoted overlay visible instead of collapsing it to ~0.4%.
///
/// The decision is additionally gated on the actual storage depth
/// (`bytes_per_channel() == 2`). An interpretation tag is advisory and may
/// disagree with the bytes — [`crate::conversion::RasterCopyBuilder::interpretation`]
/// accepts any tag without validating depth — so an 8-bit buffer mislabelled
/// `Rgb16` / `Grey16` must *not* be read on the 65535 scale: that would
/// normalise its 0..255 samples by 65535 (≈0, near-black) while writing an
/// 8-bit container at the 65535 scale (every non-zero channel saturating),
/// i.e. total data loss from a merely mis-tagged input. Gating on
/// `bytes_per_channel() == 2` keeps the tag and the sample depth in
/// agreement, matching libvips' `formatalike`/`cast` which run before
/// normalisation so interpretation and format never diverge.
fn is_genuine_16bit(raster: &Raster) -> bool {
    raster.format().bytes_per_channel() == 2
        && matches!(
            raster.interpretation(),
            Interpretation::Rgb16 | Interpretation::Grey16
        )
}

impl Raster {
    /// Fallible form of [`Raster::composite2`].
    ///
    /// # Errors
    ///
    /// [`CompositeError::DimensionMismatch`] if the images differ in
    /// size, [`CompositeError::TooManyBands`] if an input has more than 4
    /// bands, [`CompositeError::BandMismatch`] if the colour band counts
    /// differ, [`CompositeError::NonSeparableNeedsRgb`] if a
    /// non-separable mode is used on non-RGB input, or
    /// [`CompositeError::Raster`] on allocation failure.
    pub fn try_composite2(
        &self,
        overlay: &Raster,
        mode: CompositeMode,
    ) -> Result<Raster, CompositeError> {
        if self.width() != overlay.width() || self.height() != overlay.height() {
            return Err(CompositeError::DimensionMismatch {
                base_w: self.width(),
                base_h: self.height(),
                overlay_w: overlay.width(),
                overlay_h: overlay.height(),
            });
        }
        let (colour, base_alpha) = colour_bands(self.format())?;
        let (overlay_colour, overlay_alpha) = colour_bands(overlay.format())?;
        if colour != overlay_colour {
            return Err(CompositeError::BandMismatch {
                base: colour,
                overlay: overlay_colour,
            });
        }
        if mode.is_non_separable() && colour != 3 {
            return Err(CompositeError::NonSeparableNeedsRgb {
                mode,
                colour_bands: colour,
            });
        }

        // Output container: the deeper input. Normalisation follows the
        // compositing-space *interpretation*, not the raw storage depth
        // (libvips' `max_band` comes from `vips_interpretation_max_alpha`
        // after `formatalike`); see the module docs. Each input is
        // normalised by its own interpretation-derived max so a genuine
        // `Rgb16`/`Grey16` layer (0..65535) blends on a compatible 0..1
        // scale with an 8-bit layer instead of at 257:1. Genuine-16 is keyed
        // on the *resolved* interpretation ([`Raster::interpretation`], which
        // infers the genuine-16 space for an untagged 16-bit buffer as
        // libvips does) and gated on the 2-byte storage depth, so a genuine
        // 16-bit raster built via `Raster::new` is honoured without an
        // explicit tag while a mis-tagged 8-bit buffer is not (see
        // [`is_genuine_16bit`]). Anything that is not genuine-16 falls back to
        // the joint depth rule (65535 only for a fully 16-bit pipeline, else
        // 255).
        let base_bpc = self.format().bytes_per_channel();
        let overlay_bpc = overlay.format().bytes_per_channel();
        let out_bpc = base_bpc.max(overlay_bpc);
        let base_genuine16 = is_genuine_16bit(self);
        let overlay_genuine16 = is_genuine_16bit(overlay);
        let joint = if base_bpc == 2 && overlay_bpc == 2 {
            65535.0
        } else {
            255.0
        };
        let base_max = if base_genuine16 { 65535.0 } else { joint };
        let overlay_max = if overlay_genuine16 { 65535.0 } else { joint };
        // A genuine-16 input makes the output a genuine 16-bit compositing
        // space: it is written on the 0..65535 scale (see `out_max`) and must
        // be *tagged* as such (see the write-back below), so a re-composite
        // reads it on the same scale instead of the 255 ceiling.
        //
        // This applies only to an *integer* output container. When the deeper
        // input is a float raster the output is float (`out_bpc == 4`), and a
        // float image has no genuine-16 quantisation: writing it on the 65535
        // scale would inflate its samples ~257x, and stamping it `Rgb16` /
        // `Grey16` would mis-tag a float raster (the genuine-16 tags imply a
        // USHORT buffer). libvips likewise resolves the float compositing
        // space to sRGB, not a 16-bit space. So the genuine-16 write-back and
        // tag are gated on `out_bpc != 4`; a genuine-16 input is still *read*
        // on its own 0..65535 scale (`base_max` / `overlay_max` above),
        // matching libvips' per-input `max_band`.
        let output_is_integer = out_bpc != 4;
        let out_is_genuine16 = output_is_integer && (base_genuine16 || overlay_genuine16);
        // Write-back scale = the output (compositing-space) max: genuine
        // 16-bit whenever either input is a genuine-16 layer *and* the output
        // is an integer container, so a mixed-depth result fills the 16-bit
        // range and its alpha is not capped at 255; otherwise the joint rule.
        let out_max = if out_is_genuine16 { 65535.0 } else { joint };
        let inv_base = 1.0 / base_max;
        let inv_overlay = 1.0 / overlay_max;

        let out_format = PixelFormat::with_channels(colour + 1, out_bpc)
            .expect("colour+1 is 2 or 4, always representable");
        let mut out = Raster::zeroed(self.width(), self.height(), out_format)?;

        let base_ch = self.format().channels();
        let overlay_ch = overlay.format().channels();
        let out_ch = colour + 1;
        let bdata = self.data();
        let sdata = overlay.data();
        let pixels = self.width() as usize * self.height() as usize;
        let odata = out.data_mut();

        let pdf_blend = mode.is_pdf_blend();
        let non_separable = mode.is_non_separable();

        let mut cb = [0.0f64; 3];
        let mut cs = [0.0f64; 3];
        for p in 0..pixels {
            // Unpremultiplied colour values and alphas, 0..1. Each input
            // is normalised by its own interpretation-derived max.
            for k in 0..colour {
                cb[k] = read_raw(bdata, base_bpc, p * base_ch + k) * inv_base;
                cs[k] = read_raw(sdata, overlay_bpc, p * overlay_ch + k) * inv_overlay;
            }
            // Alphas are semantically 0..1; the clamp is a guard for
            // out-of-range samples once each side reads on its own scale.
            let ba = if base_alpha {
                (read_raw(bdata, base_bpc, p * base_ch + colour) * inv_base).clamp(0.0, 1.0)
            } else {
                1.0
            };
            let sa = if overlay_alpha {
                (read_raw(sdata, overlay_bpc, p * overlay_ch + colour) * inv_overlay)
                    .clamp(0.0, 1.0)
            } else {
                1.0
            };

            let (co, ao) = if pdf_blend {
                // PDF blend formula: the source colour is mixed with
                // B(cb, cs) by the backdrop alpha, then composited Over.
                let ao = sa + ba * (1.0 - sa);
                let mut co = [0.0f64; 3];
                // The PDF blend functions B(cb, cs) are defined on 0..1;
                // clamp their inputs (the linear mixing terms keep the
                // unclamped values).
                let c1 = |v: f64| v.clamp(0.0, 1.0);
                if non_separable {
                    let b = non_separable_blend(
                        mode,
                        [c1(cb[0]), c1(cb[1]), c1(cb[2])],
                        [c1(cs[0]), c1(cs[1]), c1(cs[2])],
                    );
                    for k in 0..colour {
                        co[k] = sa * (1.0 - ba) * cs[k] + ba * (1.0 - sa) * cb[k] + sa * ba * b[k];
                    }
                } else {
                    for k in 0..colour {
                        co[k] = sa * (1.0 - ba) * cs[k]
                            + ba * (1.0 - sa) * cb[k]
                            + sa * ba * separable_blend(mode, c1(cb[k]), c1(cs[k]));
                    }
                }
                (co, ao)
            } else {
                let (fa, fb) =
                    porter_duff_factors(mode, sa, ba).expect("porter-duff mode verified");
                let mut co = [0.0f64; 3];
                for k in 0..colour {
                    co[k] = fa * sa * cs[k] + fb * ba * cb[k];
                }
                // Add can push the alpha past 1; clamp like VIPS_BLEND_ADD.
                let ao = (fa * sa + fb * ba).min(1.0);
                (co, ao)
            };

            // Unpremultiply and write back on the output numeric scale.
            for (k, &c_premul) in co.iter().enumerate().take(colour) {
                let c = if ao > 0.0 { c_premul / ao } else { 0.0 };
                write_scaled(odata, out_bpc, p * out_ch + k, c, out_max);
            }
            write_scaled(odata, out_bpc, p * out_ch + colour, ao, out_max);
        }

        out.carry_meta_from(self);
        // The result inherits the base's metadata, but when a genuine-16
        // input drove the write-back onto the 0..65535 scale (`out_max`, only
        // for an integer output container — see the `out_is_genuine16` gate),
        // the base tag (typically an 8-bit or absent interpretation) would
        // mislabel the samples: re-compositing this output would read it on
        // the 255 scale and re-introduce #289, and its saved metadata would
        // disagree with the samples. Stamp the genuine-16 interpretation that
        // matches the scale it was actually written at — `Grey16` for a
        // single colour band, `Rgb16` for RGB. A float output is never
        // stamped genuine-16 (the gate excludes it), so a float raster keeps
        // the base interpretation and is never mis-tagged as USHORT.
        if out_is_genuine16 {
            out.meta.interpretation = Some(if colour == 1 {
                Interpretation::Grey16
            } else {
                Interpretation::Rgb16
            });
        }
        Ok(out)
    }

    /// Composite `overlay` onto `self` with the given blend mode
    /// (libvips `vips_composite2`); see the [module docs](crate::composite)
    /// for alpha, band, and depth semantics. Panicking form of
    /// [`Raster::try_composite2`], matching the ported-test call surface.
    ///
    /// # Panics
    ///
    /// Panics on any [`CompositeError`]; see [`Raster::try_composite2`].
    #[track_caller]
    pub fn composite(&self, overlay: &Raster, mode: CompositeMode) -> Raster {
        expect_composite(self.try_composite2(overlay, mode))
    }

    /// Alias of [`Raster::composite`], named after the binary libvips
    /// operation (`vips_composite2`).
    ///
    /// # Panics
    ///
    /// Panics on any [`CompositeError`]; see [`Raster::try_composite2`].
    #[track_caller]
    pub fn composite2(&self, overlay: &Raster, mode: CompositeMode) -> Raster {
        expect_composite(self.try_composite2(overlay, mode))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x1 RGB base: left pixel [100, 100, 100], right [200, 50, 0].
    fn base_rgb() -> Raster {
        Raster::new(2, 1, PixelFormat::Rgb8, vec![100, 100, 100, 200, 50, 0]).unwrap()
    }

    /// A 2x1 RGBA overlay: left [255, 0, 0, 128], right [0, 255, 0, 255].
    fn overlay_rgba() -> Raster {
        Raster::new(
            2,
            1,
            PixelFormat::Rgba8,
            vec![255, 0, 0, 128, 0, 255, 0, 255],
        )
        .unwrap()
    }

    /**
     * Tests the classic Over blend against hand-computed values.
     * Works by compositing a half-transparent red overlay over an opaque
     * grey base and checking the blend arithmetic per channel.
     * Input: base [100,100,100] opaque, overlay [255,0,0] alpha 128.
     * Output: c = 255*a + 100*(1-a), a = 128/255 -> [177.8, 49.8, 49.8, 255].
     */
    #[test]
    fn over_blends_half_transparent_overlay() {
        let out = base_rgb().composite(&overlay_rgba(), CompositeMode::Over);
        assert_eq!(out.format(), PixelFormat::Rgba8);
        let px = out.getpoint(0, 0);
        let a = 128.0 / 255.0;
        let want_r = 255.0 * a + 100.0 * (1.0 - a);
        let want_gb = 100.0 * (1.0 - a);
        assert!((px[0] - want_r).abs() <= 1.0, "r: {px:?}");
        assert!((px[1] - want_gb).abs() <= 1.0, "g: {px:?}");
        assert!((px[2] - want_gb).abs() <= 1.0, "b: {px:?}");
        assert_eq!(px[3], 255.0);

        // A fully opaque overlay replaces the base outright.
        let px = out.getpoint(1, 0);
        assert_eq!(&px[..3], &[0.0, 255.0, 0.0]);
    }

    /**
     * Tests the ported test_composite arithmetic on the libvips fixture
     * values. In the libvips suite the (0,0) pixel of `colour` is
     * [2, 3, 4]; base = colour + 100, overlay = colour with alpha 128,
     * and Over must produce [51.8, 52.8, 53.8, 255] within 1.
     * Input: base [102,103,104], overlay [2,3,4,128].
     * Output: getpoint(0,0) ~ [51.8, 52.8, 53.8, 255].
     */
    #[test]
    fn over_matches_ported_expected_values() {
        let base = Raster::new(1, 1, PixelFormat::Rgb8, vec![102, 103, 104]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Rgba8, vec![2, 3, 4, 128]).unwrap();
        let comp = base.composite(&overlay, CompositeMode::Over);
        let px = comp.getpoint(0, 0);
        assert!((px[0] - 51.8).abs() < 1.0, "{px:?}");
        assert!((px[1] - 52.8).abs() < 1.0, "{px:?}");
        assert!((px[2] - 53.8).abs() < 1.0, "{px:?}");
        assert!((px[3] - 255.0).abs() < 0.5, "{px:?}");
    }

    /**
     * Tests every Porter-Duff operator's alpha result on a transparent/
     * opaque corner case. Works by compositing a half-transparent source
     * over a half-transparent backdrop and checking the composite alpha
     * formula fa*sa + fb*ba for each mode.
     * Input: sa = ba = 0.5 everywhere.
     * Output: mode-specific alpha (e.g. Over 0.75, Xor 0.5, Clear 0).
     */
    #[test]
    fn porter_duff_alpha_formulas() {
        let base = Raster::new(1, 1, PixelFormat::Rgba8, vec![80, 80, 80, 128]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Rgba8, vec![160, 160, 160, 128]).unwrap();
        let a = 128.0 / 255.0;
        let cases: &[(CompositeMode, f64)] = &[
            (CompositeMode::Clear, 0.0),
            (CompositeMode::Source, a),
            (CompositeMode::Over, a + a * (1.0 - a)),
            (CompositeMode::In, a * a),
            (CompositeMode::Out, a * (1.0 - a)),
            (CompositeMode::Atop, a),
            (CompositeMode::Dest, a),
            (CompositeMode::DestOver, a + a * (1.0 - a)),
            (CompositeMode::DestIn, a * a),
            (CompositeMode::DestOut, a * (1.0 - a)),
            (CompositeMode::DestAtop, a),
            (CompositeMode::Xor, 2.0 * a * (1.0 - a)),
            (CompositeMode::Add, 1.0),
            (CompositeMode::Saturate, 1.0),
        ];
        for &(mode, want) in cases {
            let out = base.composite(&overlay, mode);
            let got = out.getpoint(0, 0)[3] / 255.0;
            assert!((got - want).abs() < 0.01, "{mode:?}: alpha {got} != {want}");
        }
    }

    /**
     * Tests that Dest returns the base pixels and Source the overlay
     * pixels, both with alpha attached.
     * Works by compositing opaque RGB inputs and comparing the colour
     * bands to the untouched inputs.
     * Input: base [100,100,100]/[200,50,0], overlay [255,0,0,128]/[0,255,0,255].
     */
    #[test]
    fn dest_and_source_pass_through() {
        let base = base_rgb();
        let overlay = overlay_rgba();

        let dest = base.composite(&overlay, CompositeMode::Dest);
        assert_eq!(&dest.getpoint(0, 0)[..3], &[100.0, 100.0, 100.0]);
        assert_eq!(dest.getpoint(0, 0)[3], 255.0);
        assert_eq!(&dest.getpoint(1, 0)[..3], &[200.0, 50.0, 0.0]);

        let source = base.composite(&overlay, CompositeMode::Source);
        // Unpremultiplied source colour is returned with its own alpha.
        assert_eq!(&source.getpoint(0, 0)[..3], &[255.0, 0.0, 0.0]);
        assert_eq!(source.getpoint(0, 0)[3], 128.0);
    }

    /**
     * Tests the separable PDF blend functions on opaque inputs, where the
     * result is exactly B(cb, cs) per channel.
     * Works by compositing opaque single-value images and checking the
     * blend function output for each separable mode.
     * Input: cb = 0.4 (102), cs = 0.8 (204), all channels.
     */
    #[test]
    fn separable_blends_on_opaque_inputs() {
        let base = Raster::new(1, 1, PixelFormat::Rgb8, vec![102, 102, 102]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Rgb8, vec![204, 204, 204]).unwrap();
        let cb = 102.0 / 255.0;
        let cs = 204.0 / 255.0;
        let cases: &[(CompositeMode, f64)] = &[
            (CompositeMode::Multiply, cb * cs),
            (CompositeMode::Screen, cb + cs - cb * cs),
            (CompositeMode::Darken, cb.min(cs)),
            (CompositeMode::Lighten, cb.max(cs)),
            (CompositeMode::Difference, (cb - cs).abs()),
            (CompositeMode::Exclusion, cb + cs - 2.0 * cb * cs),
            // cs > 0.5: hard light = 1 - 2(1-cs)(1-cb)
            (
                CompositeMode::HardLight,
                1.0 - 2.0 * (1.0 - cs) * (1.0 - cb),
            ),
            // cb < 0.5: overlay = 2*cb*cs
            (CompositeMode::Overlay, 2.0 * cb * cs),
            (CompositeMode::ColourDodge, (cb / (1.0 - cs)).min(1.0)),
            (CompositeMode::ColourBurn, 1.0 - ((1.0 - cb) / cs).min(1.0)),
        ];
        for &(mode, want) in cases {
            let out = base.composite(&overlay, mode);
            let got = out.getpoint(0, 0)[0] / 255.0;
            assert!((got - want).abs() < 0.01, "{mode:?}: {got} != {want}");
            assert_eq!(out.getpoint(0, 0)[3], 255.0, "{mode:?} alpha");
        }
    }

    /**
     * Tests SoftLight's two branches against the PDF spec formula.
     * Works by blending a dark and a bright source over the same backdrop
     * and comparing with the piecewise definition (including D(x)).
     * Input: cb = 0.4; cs = 0.2 (first branch) and cs = 0.9 (second).
     */
    #[test]
    fn soft_light_matches_pdf_spec() {
        let cb = 102.0 / 255.0;
        let base = Raster::new(1, 1, PixelFormat::Rgb8, vec![102, 102, 102]).unwrap();

        let cs1 = 51.0 / 255.0;
        let o1 = Raster::new(1, 1, PixelFormat::Rgb8, vec![51, 51, 51]).unwrap();
        let want1 = cb - (1.0 - 2.0 * cs1) * cb * (1.0 - cb);
        let got1 = base.composite(&o1, CompositeMode::SoftLight).getpoint(0, 0)[0] / 255.0;
        assert!((got1 - want1).abs() < 0.01, "{got1} != {want1}");

        let cs2 = 230.0 / 255.0;
        let o2 = Raster::new(1, 1, PixelFormat::Rgb8, vec![230, 230, 230]).unwrap();
        let d = cb.sqrt();
        let want2 = cb + (2.0 * cs2 - 1.0) * (d - cb);
        let got2 = base.composite(&o2, CompositeMode::SoftLight).getpoint(0, 0)[0] / 255.0;
        assert!((got2 - want2).abs() < 0.01, "{got2} != {want2}");
    }

    /**
     * Tests the non-separable modes' luminosity/saturation contracts.
     * Works by blending a saturated red source over a grey backdrop:
     * Luminosity keeps the backdrop hue (grey stays grey-ish with source
     * luminosity), Colour keeps backdrop luminosity with source hue.
     * Input: backdrop grey 0.5, source pure red.
     */
    #[test]
    fn non_separable_contracts() {
        let base = Raster::new(1, 1, PixelFormat::Rgb8, vec![128, 128, 128]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();

        // Luminosity: backdrop colour with source luminosity. A grey
        // backdrop has zero saturation, so the result is the grey with
        // Lum = lum(red) = 0.3.
        let px = base
            .composite(&overlay, CompositeMode::Luminosity)
            .getpoint(0, 0);
        for k in 0..3 {
            assert!((px[k] / 255.0 - 0.3).abs() < 0.01, "luminosity: {px:?}");
        }

        // Colour: source hue and saturation at backdrop luminosity. The
        // result keeps lum = 0.502 and is red-dominant.
        let px = base
            .composite(&overlay, CompositeMode::Colour)
            .getpoint(0, 0);
        let l = 0.3 * px[0] + 0.59 * px[1] + 0.11 * px[2];
        assert!(
            (l / 255.0 - 128.0 / 255.0).abs() < 0.01,
            "colour lum: {px:?}"
        );
        assert!(px[0] > px[1] && px[1] >= px[2], "colour hue: {px:?}");

        // Hue with a zero-saturation (grey) source collapses to the
        // backdrop luminosity (SetSat(cs, 0) is black, SetLum lifts it).
        let grey_src = Raster::new(1, 1, PixelFormat::Rgb8, vec![10, 10, 10]).unwrap();
        let px = base.composite(&grey_src, CompositeMode::Hue).getpoint(0, 0);
        for k in 0..3 {
            assert!((px[k] - 128.0).abs() <= 1.0, "hue with grey source: {px:?}");
        }

        // Saturation of a grey source is zero, so the result is grey at
        // backdrop luminosity.
        let px = base
            .composite(&grey_src, CompositeMode::Saturation)
            .getpoint(0, 0);
        for k in 0..3 {
            assert!(
                (px[k] - 128.0).abs() <= 1.0,
                "saturation with grey source: {px:?}"
            );
        }
    }

    /**
     * Tests non-separable output geometry on RGBA inputs: dimensions and
     * band count carry through, matching the ported
     * test_composite_non_separable assertions.
     * Input: 3x2 RGBA base and overlay, all four non-separable modes.
     * Output: 3x2, 4 channels.
     */
    #[test]
    fn non_separable_geometry() {
        let base = Raster::zeroed(3, 2, PixelFormat::Rgba8).unwrap();
        let overlay = Raster::new(
            3,
            2,
            PixelFormat::Rgba8,
            vec![
                200, 30, 40, 128, 200, 30, 40, 128, 200, 30, 40, 128, 200, 30, 40, 128, 200, 30,
                40, 128, 200, 30, 40, 128,
            ],
        )
        .unwrap();
        for mode in [
            CompositeMode::Hue,
            CompositeMode::Saturation,
            CompositeMode::Colour,
            CompositeMode::Luminosity,
        ] {
            let comp = base.composite(&overlay, mode);
            assert_eq!(comp.width(), base.width());
            assert_eq!(comp.height(), base.height());
            assert_eq!(comp.format().channels(), 4);
        }
    }

    /**
     * Tests grey (1 colour band) compositing: output is 2-band
     * grey+alpha, and Over blends the single channel.
     * Input: base Gray8 [100] opaque, overlay Multi8(2) [255, 128].
     * Output: Multi8(2), value = 255*a + 100*(1-a), alpha 255.
     */
    #[test]
    fn grey_plus_alpha_composites_to_two_bands() {
        let base = Raster::new(1, 1, PixelFormat::Gray8, vec![100]).unwrap();
        let overlay = Raster::new(
            1,
            1,
            PixelFormat::with_channels(2, 1).unwrap(),
            vec![255, 128],
        )
        .unwrap();
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.format().channels(), 2);
        let px = out.getpoint(0, 0);
        let a = 128.0 / 255.0;
        let want = 255.0 * a + 100.0 * (1.0 - a);
        assert!((px[0] - want).abs() <= 1.0, "{px:?}");
        assert_eq!(px[1], 255.0);
    }

    /**
     * Tests mixed-depth inputs blend with each layer read on the max its
     * resolved interpretation implies (libvips `max_band`), not on one
     * shared scale. A `Gray16` base is genuine-16 (its untagged
     * interpretation resolves to `Grey16`), so it reads on 0..65535; the
     * 8-bit overlay reads on 0..255; the result lands in the deeper 16-bit
     * container written on the genuine-16 scale.
     * Input: Gray16 base 32768 opaque, Multi8(2) overlay [200, 128].
     * Output: Multi16(2) tagged Grey16; each side on its own scale, the
     * result on 0..65535 (alpha fully opaque at 65535).
     */
    #[test]
    fn mixed_depth_reads_each_layer_on_its_own_scale() {
        let base = Raster::new(1, 1, PixelFormat::Gray16, 32768u16.to_ne_bytes().to_vec()).unwrap();
        let overlay = Raster::new(
            1,
            1,
            PixelFormat::with_channels(2, 1).unwrap(),
            vec![200, 128],
        )
        .unwrap();
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.format(), PixelFormat::with_channels(2, 2).unwrap());
        assert_eq!(out.meta.interpretation, Some(Interpretation::Grey16));
        let px = out.getpoint(0, 0);
        // Over with an opaque base: unpremultiplied result = sa*cs + (1-sa)*cb,
        // each colour on its own 0..1 scale, written back on the 0..65535 max.
        let a = 128.0 / 255.0;
        let cs = 200.0 / 255.0;
        let cb = 32768.0 / 65535.0;
        let want = (a * cs + (1.0 - a) * cb) * 65535.0;
        assert!((px[0] - want).abs() <= 2.0, "{px:?} want {want}");
        assert_eq!(px[1], 65535.0);
    }

    /**
     * Tests a fully 16-bit pipeline normalises by 65535 (the documented
     * libvips max_alpha choice for ushort): a half-opaque 16-bit overlay
     * mixes the raw 16-bit values directly.
     * Input: Gray16 base 32768 opaque, Multi16(2) overlay [65535, 32768].
     * Output: 0.5*65535 + 0.5*32768 ~ 49152, alpha 65535.
     */
    #[test]
    fn full_16bit_uses_16bit_scale() {
        let base = Raster::new(1, 1, PixelFormat::Gray16, 32768u16.to_ne_bytes().to_vec()).unwrap();
        let overlay = Raster::new(
            1,
            1,
            PixelFormat::with_channels(2, 2).unwrap(),
            [65535u16, 32768]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect(),
        )
        .unwrap();
        let out = base.composite(&overlay, CompositeMode::Over);
        let px = out.getpoint(0, 0);
        let a = 32768.0 / 65535.0;
        let want = a * 65535.0 + (1.0 - a) * 32768.0;
        assert!((px[0] - want).abs() <= 1.0, "{px:?} want {want}");
        assert_eq!(px[1], 65535.0);
    }

    /// A 1x1 genuine 16-bit RGBA overlay tagged [`Interpretation::Rgb16`]
    /// (0..65535), as distinct from a promoted-8-bit-in-16-bit buffer,
    /// which the crate leaves untagged.
    fn genuine_rgba16(colour: [u16; 3], alpha: u16) -> Raster {
        let data: Vec<u8> = [colour[0], colour[1], colour[2], alpha]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        Raster::new(1, 1, PixelFormat::Rgba16, data)
            .unwrap()
            .copy()
            .interpretation(Interpretation::Rgb16)
            .build()
    }

    /**
     * Tests #289: a genuine 16-bit half-opaque overlay's alpha survives on
     * the 16-bit scale instead of being clamped to a 255 ceiling. Works by
     * taking `Source` (which returns the overlay with its own alpha) of a
     * half-opaque genuine-`Rgb16` overlay over an opaque 8-bit base; the
     * output alpha must land near `0x8000`, not at most 255.
     * Input: Rgba8 base opaque, genuine Rgba16 overlay alpha 0x8000.
     * Output: Rgba16; alpha ~= 0x8000 (32768), far above the 255 ceiling.
     */
    #[test]
    fn genuine_16bit_half_opaque_alpha_survives_ceiling() {
        let base = Raster::new(1, 1, PixelFormat::Rgba8, vec![40, 50, 60, 255]).unwrap();
        let overlay = genuine_rgba16([10000, 20000, 30000], 0x8000);
        let out = base.composite(&overlay, CompositeMode::Source);
        assert_eq!(out.format(), PixelFormat::Rgba16);
        let px = out.getpoint(0, 0);
        // The 0x8000 alpha reads as ~0.5 and writes back on the 16-bit
        // scale: ~32768, never the buggy <= 255 ceiling.
        assert!(px[3] > 255.0, "alpha must exceed the 255 ceiling: {px:?}");
        assert!(
            (px[3] - 32768.0).abs() < 64.0,
            "alpha should be ~0x8000: {px:?}"
        );
        // Source returns the overlay colour, read on its own 16-bit scale.
        assert!((px[0] - 10000.0).abs() <= 1.0, "{px:?}");
        assert!((px[2] - 30000.0).abs() <= 1.0, "{px:?}");
    }

    /**
     * Tests #289: an 8-bit base blended with a genuine 16-bit overlay
     * produces a sane colour, not a 257:1-skewed / washed-out one. Works
     * by compositing a half-opaque pure-blue genuine-`Rgb16` overlay Over
     * an opaque red 8-bit base: each layer normalises by its own max, so
     * the result is a true 50/50 mix (the base red shows through) rather
     * than the overlay saturating to fully opaque and erasing the base.
     * Input: Rgb8 base [255,0,0] opaque, genuine Rgba16 overlay
     *        [0,0,65535] alpha 0x8000.
     * Output: Rgba16 ~ [32768, 0, 32768, 65535] — half red, half blue.
     */
    #[test]
    fn eight_bit_base_over_genuine_16bit_overlay_blends_sanely() {
        let base = Raster::new(1, 1, PixelFormat::Rgb8, vec![255, 0, 0]).unwrap();
        let overlay = genuine_rgba16([0, 0, 65535], 0x8000);
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.format(), PixelFormat::Rgba16);
        let px = out.getpoint(0, 0);
        // Base red shows through at ~50% (fix); the bug erased it to 0.
        assert!(
            (px[0] - 32768.0).abs() < 1500.0,
            "base red should survive at ~half: {px:?}"
        );
        // Overlay blue at ~50%, not washed to full-scale 65535.
        assert!(
            (px[2] - 32768.0).abs() < 1500.0,
            "overlay blue should be ~half, not washed out: {px:?}"
        );
        // Opaque result written across the full 16-bit range.
        assert_eq!(px[3], 65535.0, "{px:?}");
    }

    /**
     * Tests the #289 output-tag fix: a mixed-depth result written on the
     * genuine 16-bit scale (because a genuine-`Rgb16` overlay is present) is
     * tagged `Rgb16`, not left with the 8-bit base's interpretation. The
     * payoff is behavioural: chaining the result back through `composite2`
     * must re-read it on the 16-bit scale so its alpha survives, instead of
     * being clamped to the 255 ceiling and re-introducing #289.
     * Input: Rgba8 base, genuine Rgba16 overlay alpha 0x8000, Source.
     * Output tag: Rgb16; re-chained alpha stays ~0x8000 (not <= 255).
     */
    #[test]
    fn genuine_16bit_result_is_tagged_and_survives_rechain() {
        let base8 = Raster::new(1, 1, PixelFormat::Rgba8, vec![40, 50, 60, 255]).unwrap();
        let overlay = genuine_rgba16([10000, 20000, 30000], 0x8000);
        let mid = base8.composite(&overlay, CompositeMode::Source);
        assert_eq!(mid.format(), PixelFormat::Rgba16);
        // The fix stamps the genuine-16 tag that matches the 65535 write-back
        // scale; the bug left this as the base's (untagged) interpretation.
        assert_eq!(
            mid.meta.interpretation,
            Some(Interpretation::Rgb16),
            "mixed-depth genuine-16 result must be tagged Rgb16, got {:?}",
            mid.meta.interpretation
        );
        // First-hop alpha already survives under #340.
        assert!(mid.getpoint(0, 0)[3] > 255.0);

        // Re-chain: feed the result back as a genuine-16 overlay. Its alpha
        // and colour must re-read on the 16-bit scale.
        let base2 = Raster::new(1, 1, PixelFormat::Rgba8, vec![0, 0, 0, 255]).unwrap();
        let px = base2.composite(&mid, CompositeMode::Source).getpoint(0, 0);
        assert!(
            px[3] > 255.0,
            "re-chained genuine-16 alpha must survive the 255 ceiling: {px:?}"
        );
        assert!(
            (px[3] - 32768.0).abs() < 256.0,
            "re-chained alpha should stay ~0x8000: {px:?}"
        );
        assert!(
            (px[2] - 30000.0).abs() < 256.0,
            "re-chained blue should stay ~30000 on the 16-bit scale: {px:?}"
        );
    }

    /**
     * Tests the #289 output-tag fix for a single colour band: a genuine
     * `Grey16` overlay over an 8-bit grey base yields a grey+alpha 16-bit
     * result tagged `Grey16` (not the base's `Bw`), so it re-reads on the
     * 16-bit scale.
     * Input: Gray8 base, genuine Grey16-tagged Gray16 overlay, Source.
     * Output tag: Grey16.
     */
    #[test]
    fn genuine_grey16_result_is_tagged_grey16() {
        let base = Raster::new(1, 1, PixelFormat::Gray8, vec![100]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Gray16, 40000u16.to_ne_bytes().to_vec())
            .unwrap()
            .copy()
            .interpretation(Interpretation::Grey16)
            .build();
        let out = base.composite(&overlay, CompositeMode::Source);
        assert_eq!(out.format(), PixelFormat::with_channels(2, 2).unwrap());
        assert_eq!(
            out.meta.interpretation,
            Some(Interpretation::Grey16),
            "grey genuine-16 result must be tagged Grey16, got {:?}",
            out.meta.interpretation
        );
        // Written on the 16-bit scale: the Source colour is the overlay's
        // 40000, not a 255-clamped value.
        assert!(
            (out.getpoint(0, 0)[0] - 40000.0).abs() < 2.0,
            "{:?}",
            out.getpoint(0, 0)
        );
    }

    /**
     * Tests Add clamps colour and alpha rather than wrapping.
     * Input: two opaque near-white images.
     * Output: exactly white, alpha 255.
     */
    #[test]
    fn add_clamps() {
        let base = Raster::new(1, 1, PixelFormat::Rgb8, vec![200, 200, 200]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Rgb8, vec![200, 200, 200]).unwrap();
        let px = base.composite(&overlay, CompositeMode::Add).getpoint(0, 0);
        assert_eq!(px, vec![255.0, 255.0, 255.0, 255.0]);
    }

    /**
     * Tests Clear produces transparent black regardless of inputs.
     */
    #[test]
    fn clear_is_transparent_black() {
        let px = base_rgb()
            .composite(&overlay_rgba(), CompositeMode::Clear)
            .getpoint(0, 0);
        assert_eq!(px, vec![0.0, 0.0, 0.0, 0.0]);
    }

    /**
     * Tests Saturate on an opaque backdrop: no transparency is left, so
     * the source contributes nothing and the backdrop passes through.
     */
    #[test]
    fn saturate_over_opaque_backdrop_keeps_backdrop() {
        let px = base_rgb()
            .composite(&overlay_rgba(), CompositeMode::Saturate)
            .getpoint(0, 0);
        assert_eq!(&px[..3], &[100.0, 100.0, 100.0]);
        assert_eq!(px[3], 255.0);
    }

    /**
     * Tests Xor on two half-transparent pixels: each contributes its
     * colour weighted by the other's transparency.
     * Input: sa = ba = 0.5, cs = 0.8, cb = 0.2.
     * Output: c = (0.5*0.5*0.8 + 0.5*0.5*0.2) / 0.5 = 0.5.
     */
    #[test]
    fn xor_blends_disjoint_regions() {
        let base = Raster::new(1, 1, PixelFormat::Rgba8, vec![51, 51, 51, 128]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Rgba8, vec![204, 204, 204, 128]).unwrap();
        let px = base.composite(&overlay, CompositeMode::Xor).getpoint(0, 0);
        assert!((px[0] / 255.0 - 0.5).abs() < 0.01, "{px:?}");
        assert!((px[3] / 255.0 - 0.5).abs() < 0.01, "{px:?}");
    }

    /**
     * Tests typed errors: size mismatch, band mismatch, non-separable on
     * grey input, and too many bands.
     */
    #[test]
    fn typed_errors() {
        let rgb = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
        let small = Raster::zeroed(1, 2, PixelFormat::Rgb8).unwrap();
        let grey = Raster::zeroed(2, 2, PixelFormat::Gray8).unwrap();
        let five = Raster::zeroed(2, 2, PixelFormat::with_channels(5, 1).unwrap()).unwrap();

        assert!(matches!(
            rgb.try_composite2(&small, CompositeMode::Over),
            Err(CompositeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            rgb.try_composite2(&grey, CompositeMode::Over),
            Err(CompositeError::BandMismatch { .. })
        ));
        assert!(matches!(
            grey.try_composite2(&grey, CompositeMode::Hue),
            Err(CompositeError::NonSeparableNeedsRgb { .. })
        ));
        assert!(matches!(
            rgb.try_composite2(&five, CompositeMode::Over),
            Err(CompositeError::TooManyBands { .. })
        ));
    }

    /**
     * Tests that the panicking form reports the typed error message.
     */
    #[test]
    #[should_panic(expected = "composite: composite: images differ in size")]
    fn composite_panics_on_mismatch() {
        let rgb = Raster::zeroed(2, 2, PixelFormat::Rgb8).unwrap();
        let small = Raster::zeroed(1, 2, PixelFormat::Rgb8).unwrap();
        let _ = rgb.composite(&small, CompositeMode::Over);
    }

    /**
     * Tests metadata carry-through: the result keeps the base image's
     * resolution and orientation, like libvips copying the first input's
     * header.
     */
    #[test]
    fn composite_carries_base_metadata() {
        let base = base_rgb().copy().xres(7.5).orientation(6).build();
        let out = base.composite(&overlay_rgba(), CompositeMode::Over);
        assert_eq!(out.xres(), 7.5);
        assert_eq!(out.orientation(), 6);
    }

    /**
     * Tests composite2 and composite agree (composite2 is the binary
     * libvips name for the same operation).
     */
    #[test]
    fn composite2_is_composite() {
        let a = base_rgb().composite(&overlay_rgba(), CompositeMode::Screen);
        let b = base_rgb().composite2(&overlay_rgba(), CompositeMode::Screen);
        assert_eq!(a.data(), b.data());
        assert_eq!(a.format(), b.format());
    }

    /**
     * Tests unpremultiply safety: where both inputs are fully
     * transparent, Over writes transparent black without dividing by the
     * zero output alpha.
     */
    #[test]
    fn zero_alpha_does_not_divide_by_zero() {
        let base = Raster::new(1, 1, PixelFormat::Rgba8, vec![50, 60, 70, 0]).unwrap();
        let overlay = Raster::new(1, 1, PixelFormat::Rgba8, vec![80, 90, 100, 0]).unwrap();
        let px = base.composite(&overlay, CompositeMode::Over).getpoint(0, 0);
        assert_eq!(px, vec![0.0, 0.0, 0.0, 0.0]);
    }

    /// Build a float raster from f32 samples (native-endian).
    fn raster_f32(w: u32, h: u32, fmt: PixelFormat, vals: &[f32]) -> Raster {
        let data = vals.iter().flat_map(|v| v.to_ne_bytes()).collect();
        Raster::new(w, h, fmt, data).unwrap()
    }

    /**
     * Tests the exact Porter-Duff Over result on RgbaF32 inputs, with no
     * integer rounding anywhere. Works by compositing two known float
     * pixels and asserting the premultiplied-then-unpremultiplied values
     * exactly (pixel 0, where the arithmetic is exact in binary) and to
     * float precision (pixel 1, a fractional result an integer path
     * would round).
     * Input: pixel 0 dst [50,100,200] da=255, src [100,50,25] sa=127.5;
     *        pixel 1 dst [50,100,200] da=127.5, src same.
     * Output: pixel 0 exactly [75, 75, 112.5, 255];
     *         pixel 1 colour (src*0.5 + dst*0.25)/0.75, alpha 191.25.
     */
    #[test]
    fn float_over_is_exact_porter_duff() {
        let base = raster_f32(
            2,
            1,
            PixelFormat::RgbaF32,
            &[50.0, 100.0, 200.0, 255.0, 50.0, 100.0, 200.0, 127.5],
        );
        let overlay = raster_f32(
            2,
            1,
            PixelFormat::RgbaF32,
            &[100.0, 50.0, 25.0, 127.5, 100.0, 50.0, 25.0, 127.5],
        );
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.format(), PixelFormat::RgbaF32);

        // Opaque backdrop: sa = 0.5, da = 1, ao = 1; the stored colour is
        // src*sa + dst*(1-sa) with the fraction preserved.
        let px = out.getpoint(0, 0);
        assert_eq!(px, vec![75.0, 75.0, 112.5, 255.0]);

        // Half-transparent backdrop: sa = da = 0.5, ao = 0.75. The
        // unpremultiplied colour is fractional (83.33...), which an
        // integer container could not carry.
        let px = out.getpoint(1, 0);
        let (sa, da) = (0.5, 0.5);
        let ao = sa + da * (1.0 - sa);
        for (k, &s) in [100.0, 50.0, 25.0].iter().enumerate() {
            let d = [50.0, 100.0, 200.0][k];
            let want = (s * sa + d * da * (1.0 - sa)) / ao;
            assert!((px[k] - want).abs() < 1e-3, "band {k}: {px:?} want {want}");
        }
        assert_eq!(px[3], 191.25);
    }

    /**
     * Tests float opaque-over-opaque returns the source bit-exactly,
     * including fractional and beyond-scale (HDR) samples: the float
     * write path neither rounds nor clamps.
     * Input: opaque RgbaF32 base, opaque overlay [0.125, 2.5, 400.75].
     * Output: exactly the overlay colour with alpha 255.
     */
    #[test]
    fn float_opaque_over_opaque_returns_source_exactly() {
        let base = raster_f32(1, 1, PixelFormat::RgbaF32, &[10.0, 20.0, 30.0, 255.0]);
        let overlay = raster_f32(1, 1, PixelFormat::RgbaF32, &[0.125, 2.5, 400.75, 255.0]);
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.getpoint(0, 0), vec![0.125, 2.5, 400.75, 255.0]);
    }

    /**
     * Tests float samples outside the working range survive the actual
     * blend arithmetic unclamped: colour above the 255 scale (HDR) and
     * below zero passes through Porter-Duff and the float write-back.
     * Input: base [510, -12.5, 2.5] opaque, overlay fully transparent.
     * Output: Over keeps the backdrop exactly, including 510 and -12.5.
     */
    #[test]
    fn float_hdr_and_negative_survive_unclamped() {
        let base = raster_f32(1, 1, PixelFormat::RgbaF32, &[510.0, -12.5, 2.5, 255.0]);
        let overlay = raster_f32(1, 1, PixelFormat::RgbaF32, &[90.0, 90.0, 90.0, 0.0]);
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.getpoint(0, 0), vec![510.0, -12.5, 2.5, 255.0]);
    }

    /**
     * Tests the float alpha convention: like libvips, a float raster
     * keeps the 0..255 numeric alpha of the sRGB compositing space
     * (max_band comes from the interpretation, not the storage depth),
     * so casting the ported u8 fixture to float reproduces the u8
     * result. An 0..1 float-alpha reading would instead clamp alpha 128
     * to opaque and return the overlay colour outright.
     * Input: base [102,103,104] and overlay [2,3,4,128], u8 and their
     * RgbaF32 casts.
     * Output: float Over ~ [51.8, 52.8, 53.8, 255], within quantisation
     * of the u8 result.
     */
    #[test]
    fn float_alpha_uses_numeric_scale_like_u8() {
        let base_u8 = Raster::new(1, 1, PixelFormat::Rgb8, vec![102, 103, 104]).unwrap();
        let overlay_u8 = Raster::new(1, 1, PixelFormat::Rgba8, vec![2, 3, 4, 128]).unwrap();
        let base_f = base_u8.cast(PixelFormat::with_channels(3, 4).unwrap());
        let overlay_f = overlay_u8.cast(PixelFormat::RgbaF32);

        let out_f = base_f.composite(&overlay_f, CompositeMode::Over);
        assert_eq!(out_f.format(), PixelFormat::RgbaF32);
        let px = out_f.getpoint(0, 0);
        let a = 128.0 / 255.0;
        for (k, &b) in [102.0, 103.0, 104.0].iter().enumerate() {
            let want = [2.0, 3.0, 4.0][k] * a + b * (1.0 - a);
            assert!((px[k] - want).abs() < 1e-3, "band {k}: {px:?} want {want}");
        }
        assert_eq!(px[3], 255.0);

        // And the u8 path agrees to its own quantisation.
        let px_u8 = base_u8
            .composite(&overlay_u8, CompositeMode::Over)
            .getpoint(0, 0);
        for k in 0..4 {
            assert!(
                (px[k] - px_u8[k]).abs() <= 0.5,
                "band {k}: {px:?} vs {px_u8:?}"
            );
        }
    }

    /**
     * Tests mixed float-over-u8 compositing: the output takes the deeper
     * (float) container and blends on the shared 0..255 numeric scale,
     * keeping the fractional result.
     * Input: Rgb8 base [100,100,100], RgbaF32 overlay [255,0,0] sa=127.5.
     * Output: RgbaF32 exactly [177.5, 50, 50, 255].
     */
    #[test]
    fn float_over_u8_takes_float_container() {
        let base = Raster::new(1, 1, PixelFormat::Rgb8, vec![100, 100, 100]).unwrap();
        let overlay = raster_f32(1, 1, PixelFormat::RgbaF32, &[255.0, 0.0, 0.0, 127.5]);
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.format(), PixelFormat::RgbaF32);
        assert_eq!(out.getpoint(0, 0), vec![177.5, 50.0, 50.0, 255.0]);
    }

    /**
     * Tests grey float compositing through FloatF32(1)/FloatF32(2): one
     * colour band plus alpha lands in a FloatF32(2) container with the
     * exact fractional blend.
     * Input: FloatF32(1) base [100], FloatF32(2) overlay [200, 127.5].
     * Output: FloatF32(2) exactly [150, 255].
     */
    #[test]
    fn float_grey_composites_to_two_float_bands() {
        let base = raster_f32(1, 1, PixelFormat::with_channels(1, 4).unwrap(), &[100.0]);
        let overlay = raster_f32(
            1,
            1,
            PixelFormat::with_channels(2, 4).unwrap(),
            &[200.0, 127.5],
        );
        let out = base.composite(&overlay, CompositeMode::Over);
        assert_eq!(out.format(), PixelFormat::with_channels(2, 4).unwrap());
        assert_eq!(out.getpoint(0, 0), vec![150.0, 255.0]);
    }

    /**
     * Tests the ported test_composite_non_separable shape: RgbaF32
     * inputs through all four non-separable modes keep the geometry and
     * match the u8 path within its quantisation (the float working
     * values sit on the same 0..255 scale, so the PDF 0..1 blend clamp
     * behaves identically).
     * Input: Rgba8 base/overlay and their RgbaF32 casts.
     * Output: same dimensions, 4 float channels, values within 0.5 of u8.
     */
    #[test]
    fn float_non_separable_matches_u8_path() {
        let base_u8 = Raster::new(1, 1, PixelFormat::Rgba8, vec![128, 128, 128, 255]).unwrap();
        let overlay_u8 = Raster::new(1, 1, PixelFormat::Rgba8, vec![200, 30, 40, 128]).unwrap();
        let base_f = base_u8.cast(PixelFormat::RgbaF32);
        let overlay_f = overlay_u8.cast(PixelFormat::RgbaF32);
        for mode in [
            CompositeMode::Hue,
            CompositeMode::Saturation,
            CompositeMode::Colour,
            CompositeMode::Luminosity,
            CompositeMode::Multiply,
            CompositeMode::Screen,
        ] {
            let out_f = base_f.composite(&overlay_f, mode);
            assert_eq!(out_f.width(), base_f.width());
            assert_eq!(out_f.height(), base_f.height());
            assert_eq!(out_f.format(), PixelFormat::RgbaF32);
            let px_f = out_f.getpoint(0, 0);
            let px_u8 = base_u8.composite(&overlay_u8, mode).getpoint(0, 0);
            for k in 0..4 {
                assert!(
                    (px_f[k] - px_u8[k]).abs() <= 0.5,
                    "{mode:?} band {k}: {px_f:?} vs {px_u8:?}"
                );
            }
        }
    }
}
