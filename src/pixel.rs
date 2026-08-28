use core::num::NonZeroU16;

/// Canonical pixel formats used throughout the pipeline.
///
/// Source images are normalized into one of these formats at decode time.
/// This keeps format-specific complexity out of the planner, execution engine,
/// and [`Raster`](crate::raster::Raster) buffer management. Every named format
/// is defined by two axes -- channel count (1, 3, or 4) and sample type
/// (unsigned 8-bit, unsigned 16-bit, or 32-bit float). The band operations in
/// [`crate::bands`] can produce intermediate images with any other band count
/// (for example 2 bands from `extract_bands`, or 100 bands from `bandfold`);
/// those are carried by the `Multi8` / `Multi16` / `FloatF32` variants.
///
/// Float samples are stored as native-endian `f32` values in the raster's
/// byte buffer, matching the native-order convention the 16-bit formats
/// already use. `RgbaF32` is the named four-band float format the ported
/// compositing tests cast to; every other float band count is carried by
/// `FloatF32(n)`.
///
/// # Canonical spelling
///
/// A layout with a named variant has two constructible spellings, because
/// the tuple variants are public: `FloatF32(4)` names what `RgbaF32` names,
/// and `Multi8(3)` names what `Rgb8` names. The named one is canonical.
/// [`PixelFormat::with_channels`] produces it, [`PixelFormat::canonical`]
/// converts to it, and everything this crate hands you is already in it: a
/// [`Raster`](crate::raster::Raster) canonicalises the format it is built
/// with, and the manifest wire format writes the canonical tag. `PartialEq`
/// and `Hash` are derived and so distinguish the two spellings, which is why
/// nothing here produces the non-canonical one (issue #531).
///
/// # Variants
///
/// | Variant      | Channels | Bits/channel | Bytes/pixel |
/// |--------------|----------|--------------|-------------|
/// | `Gray8`      | 1        | 8            | 1           |
/// | `Gray16`     | 1        | 16           | 2           |
/// | `Rgb8`       | 3        | 8            | 3           |
/// | `Rgba8`      | 4        | 8            | 4           |
/// | `Rgb16`      | 3        | 16           | 6           |
/// | `Rgba16`     | 4        | 16           | 8           |
/// | `RgbaF32`    | 4        | 32 (float)   | 16          |
/// | `Multi8(n)`  | n        | 8            | n           |
/// | `Multi16(n)` | n        | 16           | 2n          |
/// | `FloatF32(n)`| n        | 32 (float)   | 4n          |
///
/// # Example usage
///
/// * [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs)
/// * [CLI source](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs)
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-format)
/// (pyramid overview at [`#pyramid`](https://libviprs.org/cli/#pyramid))
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum PixelFormat {
    /// Single-channel 8-bit grayscale.
    Gray8,
    /// Single-channel 16-bit grayscale.
    Gray16,
    /// Three-channel 8-bit RGB colour.
    Rgb8,
    /// Four-channel 8-bit RGBA colour with alpha.
    Rgba8,
    /// Three-channel 16-bit RGB colour.
    Rgb16,
    /// Four-channel 16-bit RGBA colour with alpha.
    Rgba16,
    /// Four-channel 32-bit float RGBA colour with alpha, stored as
    /// native-endian `f32` samples. This is the float format the ported
    /// compositing tests cast to (`cast(PixelFormat::RgbaF32)`). Float
    /// rasters are compute intermediates: the tile encoding sinks reject
    /// them with a typed error, and the `.v` container is the only
    /// encode/decode path that carries them.
    RgbaF32,
    /// N-channel 8-bit multiband image, produced by the band operations in
    /// [`crate::bands`] when the band count is not 1, 3, or 4. Multiband
    /// rasters are compute intermediates: the decode, resize, and tile
    /// encoding paths do not accept them.
    Multi8(NonZeroU16),
    /// N-channel 16-bit multiband image; see [`PixelFormat::Multi8`].
    Multi16(NonZeroU16),
    /// N-channel 32-bit float image, stored as native-endian `f32` samples.
    /// This is the carrier for every float band count other than 4 (which
    /// canonicalizes to [`PixelFormat::RgbaF32`]): single-band float ramps
    /// and maths results use `FloatF32(1)`, float colour intermediates
    /// `FloatF32(3)`, and so on. Like the `Multi` variants, float rasters
    /// are compute intermediates; see [`PixelFormat::RgbaF32`].
    FloatF32(NonZeroU16),
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    ///
    /// Equal to `channels() * bytes_per_channel()`. Used by [`Raster`](crate::raster::Raster)
    /// to compute buffer sizes and strides.
    pub fn bytes_per_pixel(self) -> usize {
        self.channels() * self.bytes_per_channel()
    }

    /// Number of channels (1 for grayscale, 3 for RGB, 4 for RGBA, `n` for
    /// the multiband variants).
    pub fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::Rgb8 | Self::Rgb16 => 3,
            Self::Rgba8 | Self::Rgba16 | Self::RgbaF32 => 4,
            Self::Multi8(n) | Self::Multi16(n) | Self::FloatF32(n) => n.get() as usize,
        }
    }

    /// The canonical format for a channel count and byte depth.
    ///
    /// For the 8- and 16-bit depths, counts 1, 3, and 4 map to the named
    /// `Gray` / `Rgb` / `Rgba` variants and every other count maps to
    /// `Multi8` / `Multi16`. For the 4-byte float depth, count 4 maps to
    /// the named `RgbaF32` and every other count to `FloatF32`. The band
    /// operations use this so a 3-band multiband result compares equal to
    /// `Rgb8` rather than living as a `Multi8(3)` alias.
    ///
    /// Returns `None` when `channels` is 0 or above `u16::MAX`, or when
    /// `bytes_per_channel` is not 1, 2, or 4.
    pub fn with_channels(channels: usize, bytes_per_channel: usize) -> Option<Self> {
        let fmt = match (channels, bytes_per_channel) {
            (1, 1) => Self::Gray8,
            (1, 2) => Self::Gray16,
            (3, 1) => Self::Rgb8,
            (3, 2) => Self::Rgb16,
            (4, 1) => Self::Rgba8,
            (4, 2) => Self::Rgba16,
            (4, 4) => Self::RgbaF32,
            (n, 1) => Self::Multi8(NonZeroU16::new(u16::try_from(n).ok()?)?),
            (n, 2) => Self::Multi16(NonZeroU16::new(u16::try_from(n).ok()?)?),
            (n, 4) => Self::FloatF32(NonZeroU16::new(u16::try_from(n).ok()?)?),
            _ => return None,
        };
        Some(fmt)
    }

    /// The canonical spelling of this format's pixel layout.
    ///
    /// The tuple variants are public, so a layout that has a named variant
    /// has two constructible spellings: `FloatF32(4)` names what `RgbaF32`
    /// names, `Multi8(3)` names what `Rgb8` names, and so on for the seven
    /// rows below. [`PixelFormat::with_channels`] produces the right-hand
    /// column; direct construction can produce the left-hand one. This maps
    /// one to the other, and is the identity on everything else.
    ///
    /// | non-canonical | canonical |
    /// |---|---|
    /// | `Multi8(1)`   | `Gray8`   |
    /// | `Multi8(3)`   | `Rgb8`    |
    /// | `Multi8(4)`   | `Rgba8`   |
    /// | `Multi16(1)`  | `Gray16`  |
    /// | `Multi16(3)`  | `Rgb16`   |
    /// | `Multi16(4)`  | `Rgba16`  |
    /// | `FloatF32(4)` | `RgbaF32` |
    ///
    /// Note that `FloatF32(1)` and `FloatF32(3)` are *already* canonical:
    /// four is the only float band count with a named variant.
    ///
    /// You rarely need to call this. Nothing this crate produces is
    /// non-canonical: every raster canonicalises its format at construction
    /// (see [`Raster::new`](crate::raster::Raster::new)), and the manifest
    /// wire format writes the canonical tag. It is here for the case where
    /// you built a format yourself and want to compare it against one of
    /// ours, since `PartialEq` is derived and so distinguishes the two
    /// spellings.
    pub fn canonical(self) -> Self {
        // Spelled out per carrier rather than as
        // `with_channels(self.channels(), self.bytes_per_channel())`, which
        // would give the same answers today and be shorter. Two reasons, and
        // the second is the load-bearing one.
        //
        // `with_channels` is keyed on a byte depth, and a byte depth does
        // not identify a carrier: an unsigned-32 carrier (issue #517) would
        // share `bytes_per_channel() == 4` with the float one, so the short
        // form would quietly canonicalise `Uint32(3)` to `FloatF32(3)` --
        // the same class of silent retag this method exists to remove.
        //
        // And there is no wildcard arm here, so adding a carrier variant to
        // this `#[non_exhaustive]` enum is a compile error at this match
        // rather than a default that happens to be wrong. The decision gets
        // forced instead of inherited.
        match self {
            Self::Gray8
            | Self::Gray16
            | Self::Rgb8
            | Self::Rgba8
            | Self::Rgb16
            | Self::Rgba16
            | Self::RgbaF32 => self,
            Self::Multi8(n) => match n.get() {
                1 => Self::Gray8,
                3 => Self::Rgb8,
                4 => Self::Rgba8,
                _ => self,
            },
            Self::Multi16(n) => match n.get() {
                1 => Self::Gray16,
                3 => Self::Rgb16,
                4 => Self::Rgba16,
                _ => self,
            },
            // Four is the only float band count with a named variant, so
            // `FloatF32(1)` and `FloatF32(3)` are already canonical.
            Self::FloatF32(n) => match n.get() {
                4 => Self::RgbaF32,
                _ => self,
            },
        }
    }

    /// Whether this format is the canonical spelling of its pixel layout.
    ///
    /// Equivalent to `self.canonical() == self`. See
    /// [`PixelFormat::canonical`] for the seven layouts where it is `false`.
    pub fn is_canonical(self) -> bool {
        self.canonical() == self
    }

    /// Whether this format includes an alpha (transparency) channel.
    ///
    /// Returns `true` for the four-band layouts: `Rgba8`, `Rgba16`,
    /// `RgbaF32`, and the tuple spellings of those same layouts
    /// (`Multi8(4)`, `Multi16(4)`, `FloatF32(4)`). The question is about the
    /// pixel layout, not about which of its two spellings you are holding,
    /// so it is answered on [`PixelFormat::canonical`] (issue #531).
    pub fn has_alpha(self) -> bool {
        matches!(self.canonical(), Self::Rgba8 | Self::Rgba16 | Self::RgbaF32)
    }

    /// Whether this format stores 32-bit float samples (`RgbaF32` or
    /// `FloatF32`).
    ///
    /// Float samples are raw `f32` values in native byte order; the unsigned
    /// formats store `u8` / `u16` samples. Code that interprets raw sample
    /// bytes must dispatch on this (or on [`PixelFormat::bytes_per_channel`])
    /// rather than assuming "not 8-bit means 16-bit".
    pub fn is_float(self) -> bool {
        matches!(self, Self::RgbaF32 | Self::FloatF32(_))
    }

    /// Bytes per channel sample (1 for 8-bit formats, 2 for 16-bit formats,
    /// 4 for float formats).
    ///
    /// Useful when converting between bit depths or when working with raw
    /// sample values that need to be read as `u8` vs `u16` vs `f32`.
    pub fn bytes_per_channel(self) -> usize {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 | Self::Multi8(_) => 1,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 | Self::Multi16(_) => 2,
            Self::RgbaF32 | Self::FloatF32(_) => 4,
        }
    }

    /// Return the variant of this format that includes an alpha channel.
    ///
    /// `Gray8` and `Gray16` promote to `Rgba8` / `Rgba16` respectively (not
    /// `GrayAlpha`), because the pipeline does not use a gray+alpha format.
    /// One- and three-band float images promote to `RgbaF32`, since
    /// `FloatF32(1)` / `FloatF32(3)` are the canonical float gray and RGB
    /// carriers. If the format already has alpha, returns `self` unchanged.
    /// A band count with no named variant has no alpha concept, so
    /// `Multi8(2)` and `FloatF32(7)` come back as themselves.
    ///
    /// The answer is the same for both spellings of a layout: `Multi8(1)`
    /// promotes to `Rgba8` exactly as `Gray8` does, and the returned format
    /// is always canonical (issue #531).
    pub fn with_alpha(self) -> Self {
        // On the canonical spelling, so `Multi8(1)` promotes the way `Gray8`
        // does rather than falling through unchanged (issue #531).
        match self.canonical() {
            Self::Gray8 => Self::Rgba8,
            Self::Gray16 => Self::Rgba16,
            Self::Rgb8 => Self::Rgba8,
            Self::Rgb16 => Self::Rgba16,
            Self::FloatF32(n) if matches!(n.get(), 1 | 3) => Self::RgbaF32,
            other => other,
        }
    }

    /// Return the variant of this format with the alpha channel removed.
    ///
    /// `Rgba8` demotes to `Rgb8`, `Rgba16` to `Rgb16`, and `RgbaF32` to
    /// `FloatF32(3)` (the canonical three-band float carrier). A format
    /// without alpha keeps its layout, and comes back in that layout's
    /// canonical spelling: `Multi8(3)` demotes to `Rgb8`, which is the same
    /// pixel layout under the name `with_channels` gives it (issue #531).
    pub fn without_alpha(self) -> Self {
        // On the canonical spelling, for the reason `with_alpha` is.
        match self.canonical() {
            Self::Rgba8 => Self::Rgb8,
            Self::Rgba16 => Self::Rgb16,
            // Expect: 3 is non-zero, so the constructor cannot fail.
            Self::RgbaF32 => Self::FloatF32(NonZeroU16::new(3).expect("3 is non-zero")),
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /**
     * Tests that the sample-kind spine lives in this module, so the rest of
     * the crate has one shared answer to "what are these bytes" rather than
     * a hand-rolled depth enum per module (`colour.rs`'s private `SpaceDepth`
     * is exactly that duplicate, issue #607).
     * Works by scanning this module's own source, compiled in with
     * `include_str!`, for the type declaration; the needle is spelled in two
     * halves so this assertion is not itself a hit.
     * Input: `src/pixel.rs` -> Output: the declaration is present.
     */
    #[test]
    fn sample_kind_spine_is_declared_here() {
        const SRC: &str = include_str!("pixel.rs");
        // Positive control: the same scan finds a declaration that is
        // present, so a miss below is a real miss and not an empty read.
        assert!(
            SRC.contains(concat!("pub enum Pixel", "Format")),
            "positive control failed: the scan cannot see this module's source"
        );
        assert!(
            SRC.contains(concat!("pub enum Sample", "Kind")),
            "the SampleKind spine must be declared in src/pixel.rs, not \
             hand-rolled per module"
        );
    }

    /**
     * Tests that bytes_per_pixel equals channels * bytes_per_channel for every format.
     * Works by iterating all PixelFormat variants and checking the arithmetic identity,
     * catching mismatches if one method is updated without the others.
     * Input: all 6 variants → Output: identity holds for each (e.g. Rgb8: 3 == 3*1).
     */
    #[test]
    fn bytes_per_pixel_matches_channels_times_depth() {
        for fmt in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
        ] {
            assert_eq!(
                fmt.bytes_per_pixel(),
                fmt.channels() * fmt.bytes_per_channel(),
                "Mismatch for {fmt:?}"
            );
        }
    }

    /**
     * Tests that with_alpha and without_alpha are inverses of each other.
     * Works by converting non-alpha formats to alpha and back, verifying the
     * original format is recovered.
     * Input: Rgb8 → with_alpha → Rgba8 → without_alpha → Rgb8.
     */
    #[test]
    fn alpha_round_trip() {
        assert_eq!(PixelFormat::Rgb8.with_alpha(), PixelFormat::Rgba8);
        assert_eq!(PixelFormat::Rgba8.without_alpha(), PixelFormat::Rgb8);
        assert_eq!(PixelFormat::Rgb16.with_alpha(), PixelFormat::Rgba16);
        assert_eq!(PixelFormat::Rgba16.without_alpha(), PixelFormat::Rgb16);
    }

    /**
     * Tests that calling with_alpha on formats that already have alpha is a no-op.
     * Works by applying with_alpha to Rgba8/Rgba16 and asserting the result is unchanged.
     * Input: Rgba8.with_alpha() → Output: Rgba8.
     */
    #[test]
    fn with_alpha_is_idempotent() {
        assert_eq!(PixelFormat::Rgba8.with_alpha(), PixelFormat::Rgba8);
        assert_eq!(PixelFormat::Rgba16.with_alpha(), PixelFormat::Rgba16);
    }

    /**
     * Tests that calling without_alpha on formats without alpha is a no-op.
     * Works by applying without_alpha to Rgb8/Gray8 and asserting the result is unchanged.
     * Input: Rgb8.without_alpha() → Output: Rgb8.
     */
    #[test]
    fn without_alpha_is_idempotent() {
        assert_eq!(PixelFormat::Rgb8.without_alpha(), PixelFormat::Rgb8);
        assert_eq!(PixelFormat::Gray8.without_alpha(), PixelFormat::Gray8);
    }

    /**
     * Tests that has_alpha returns true only for Rgba8 and Rgba16.
     * Works by checking every variant and asserting the expected boolean.
     * Input: Gray8→false, Rgb8→false, Rgba8→true, Rgb16→false, Rgba16→true.
     */
    #[test]
    fn has_alpha_correctness() {
        assert!(!PixelFormat::Gray8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(!PixelFormat::Gray16.has_alpha());
        assert!(!PixelFormat::Rgb16.has_alpha());
        assert!(PixelFormat::Rgba16.has_alpha());
    }

    /**
     * Tests that with_channels canonicalizes 1/3/4-channel requests to the
     * named variants and everything else to Multi8/Multi16.
     * Works by mapping each (channels, depth) pair and asserting the variant.
     * Input: (3,1)→Rgb8, (2,1)→Multi8(2), (5,2)→Multi16(5), (0,1)→None,
     * (1,3)→None.
     */
    #[test]
    fn with_channels_canonicalizes() {
        assert_eq!(PixelFormat::with_channels(1, 1), Some(PixelFormat::Gray8));
        assert_eq!(PixelFormat::with_channels(1, 2), Some(PixelFormat::Gray16));
        assert_eq!(PixelFormat::with_channels(3, 1), Some(PixelFormat::Rgb8));
        assert_eq!(PixelFormat::with_channels(3, 2), Some(PixelFormat::Rgb16));
        assert_eq!(PixelFormat::with_channels(4, 1), Some(PixelFormat::Rgba8));
        assert_eq!(PixelFormat::with_channels(4, 2), Some(PixelFormat::Rgba16));

        let two = PixelFormat::with_channels(2, 1).unwrap();
        assert_eq!(two, PixelFormat::Multi8(NonZeroU16::new(2).unwrap()));
        let five16 = PixelFormat::with_channels(5, 2).unwrap();
        assert_eq!(five16, PixelFormat::Multi16(NonZeroU16::new(5).unwrap()));

        assert_eq!(PixelFormat::with_channels(0, 1), None);
        assert_eq!(PixelFormat::with_channels(1, 3), None);
        assert_eq!(
            PixelFormat::with_channels(usize::from(u16::MAX) + 1, 1),
            None
        );
    }

    /**
     * Tests that the Multi variants report consistent geometry.
     * Works by constructing Multi8(5) and Multi16(5) and checking channels,
     * bytes_per_channel, and bytes_per_pixel; also checks the alpha helpers
     * treat Multi as alpha-free and identity.
     * Input: Multi8(5) → channels 5, bpp 5; Multi16(5) → bpp 10.
     */
    #[test]
    fn multi_variant_geometry() {
        let m8 = PixelFormat::with_channels(5, 1).unwrap();
        assert_eq!(m8.channels(), 5);
        assert_eq!(m8.bytes_per_channel(), 1);
        assert_eq!(m8.bytes_per_pixel(), 5);
        assert!(!m8.has_alpha());
        assert_eq!(m8.with_alpha(), m8);
        assert_eq!(m8.without_alpha(), m8);

        let m16 = PixelFormat::with_channels(5, 2).unwrap();
        assert_eq!(m16.channels(), 5);
        assert_eq!(m16.bytes_per_channel(), 2);
        assert_eq!(m16.bytes_per_pixel(), 10);
    }

    /**
     * Tests the geometry of the float variants: RgbaF32 is 4 channels at
     * 4 bytes each (16 bytes/pixel) and FloatF32(n) is n channels at
     * 4 bytes each. Works by checking channels, bytes_per_channel, and
     * bytes_per_pixel for RgbaF32, FloatF32(1), and FloatF32(3).
     * Input: RgbaF32 → (4, 4, 16); FloatF32(3) → (3, 4, 12).
     */
    #[test]
    fn float_variant_geometry() {
        let rgba = PixelFormat::RgbaF32;
        assert_eq!(rgba.channels(), 4);
        assert_eq!(rgba.bytes_per_channel(), 4);
        assert_eq!(rgba.bytes_per_pixel(), 16);

        let gray = PixelFormat::FloatF32(NonZeroU16::new(1).unwrap());
        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.bytes_per_channel(), 4);
        assert_eq!(gray.bytes_per_pixel(), 4);

        let rgb = PixelFormat::FloatF32(NonZeroU16::new(3).unwrap());
        assert_eq!(rgb.channels(), 3);
        assert_eq!(rgb.bytes_per_channel(), 4);
        assert_eq!(rgb.bytes_per_pixel(), 12);
    }

    /**
     * Tests that with_channels canonicalizes the 4-byte float depth:
     * 4 bands map to the named RgbaF32, every other count to FloatF32(n),
     * and 0 bands or an unknown depth stay None.
     * Input: (4,4)→RgbaF32, (1,4)→FloatF32(1), (7,4)→FloatF32(7),
     * (0,4)→None, (1,8)→None.
     */
    #[test]
    fn with_channels_canonicalizes_float() {
        assert_eq!(PixelFormat::with_channels(4, 4), Some(PixelFormat::RgbaF32));
        assert_eq!(
            PixelFormat::with_channels(1, 4),
            Some(PixelFormat::FloatF32(NonZeroU16::new(1).unwrap()))
        );
        assert_eq!(
            PixelFormat::with_channels(7, 4),
            Some(PixelFormat::FloatF32(NonZeroU16::new(7).unwrap()))
        );
        assert_eq!(PixelFormat::with_channels(0, 4), None);
        assert_eq!(PixelFormat::with_channels(1, 8), None);
    }

    /**
     * Tests is_float: true for RgbaF32 and FloatF32(n), false for every
     * unsigned variant. Works by checking each variant directly.
     * Input: RgbaF32→true, FloatF32(2)→true, Gray8/Rgba16/Multi16(5)→false.
     */
    #[test]
    fn is_float_correctness() {
        assert!(PixelFormat::RgbaF32.is_float());
        assert!(PixelFormat::FloatF32(NonZeroU16::new(2).unwrap()).is_float());
        assert!(!PixelFormat::Gray8.is_float());
        assert!(!PixelFormat::Gray16.is_float());
        assert!(!PixelFormat::Rgb8.is_float());
        assert!(!PixelFormat::Rgba8.is_float());
        assert!(!PixelFormat::Rgb16.is_float());
        assert!(!PixelFormat::Rgba16.is_float());
        assert!(!PixelFormat::with_channels(5, 1).unwrap().is_float());
        assert!(!PixelFormat::with_channels(5, 2).unwrap().is_float());
    }

    /**
     * Tests the float alpha helpers: RgbaF32 has alpha and demotes to
     * FloatF32(3); FloatF32(1) and FloatF32(3) promote to RgbaF32; other
     * float band counts are alpha-free and unchanged (like Multi).
     * Input: RgbaF32.without_alpha()→FloatF32(3); FloatF32(3).with_alpha()
     * →RgbaF32; FloatF32(2).with_alpha()→FloatF32(2).
     */
    #[test]
    fn float_alpha_helpers() {
        let f1 = PixelFormat::FloatF32(NonZeroU16::new(1).unwrap());
        let f2 = PixelFormat::FloatF32(NonZeroU16::new(2).unwrap());
        let f3 = PixelFormat::FloatF32(NonZeroU16::new(3).unwrap());

        assert!(PixelFormat::RgbaF32.has_alpha());
        assert!(!f1.has_alpha());
        assert!(!f3.has_alpha());

        assert_eq!(f1.with_alpha(), PixelFormat::RgbaF32);
        assert_eq!(f3.with_alpha(), PixelFormat::RgbaF32);
        assert_eq!(f2.with_alpha(), f2);
        assert_eq!(PixelFormat::RgbaF32.with_alpha(), PixelFormat::RgbaF32);

        assert_eq!(PixelFormat::RgbaF32.without_alpha(), f3);
        assert_eq!(f3.without_alpha(), f3);
        assert_eq!(f2.without_alpha(), f2);
    }

    /// Every `(band count, byte depth)` pair that has both a tuple spelling
    /// and a named one, paired with the named variant `with_channels`
    /// produces for it. Direct construction of the left-hand column is what
    /// issue #531 is about: the tuple variants are public, so both spellings
    /// of one pixel layout are constructible.
    fn alias_table() -> [(PixelFormat, PixelFormat); 7] {
        let nz = |n: u16| NonZeroU16::new(n).expect("the table holds no zeroes");
        [
            (PixelFormat::Multi8(nz(1)), PixelFormat::Gray8),
            (PixelFormat::Multi8(nz(3)), PixelFormat::Rgb8),
            (PixelFormat::Multi8(nz(4)), PixelFormat::Rgba8),
            (PixelFormat::Multi16(nz(1)), PixelFormat::Gray16),
            (PixelFormat::Multi16(nz(3)), PixelFormat::Rgb16),
            (PixelFormat::Multi16(nz(4)), PixelFormat::Rgba16),
            (PixelFormat::FloatF32(nz(4)), PixelFormat::RgbaF32),
        ]
    }

    /**
     * Tests the exact disagreement issue #531 reproduces: FloatF32(4) is a
     * constructible second spelling of the layout RgbaF32 names, and the two
     * must not answer differently about that layout.
     * Works by asking both spellings every question PixelFormat can be
     * asked and comparing the answers pairwise, so a future accessor that
     * forgets the alias fails here rather than at a call site.
     * Input: FloatF32(4) vs RgbaF32 -> Output: 4 channels, 4 bytes each,
     * 16 bytes per pixel, float, has_alpha true, with_alpha RgbaF32,
     * without_alpha FloatF32(3), for both.
     */
    #[test]
    fn floatf32_4_and_rgbaf32_answer_alike() {
        let alias = PixelFormat::FloatF32(NonZeroU16::new(4).expect("4 is non-zero"));
        let named = PixelFormat::RgbaF32;

        assert_eq!(alias.channels(), named.channels(), "channels disagree");
        assert_eq!(
            alias.bytes_per_channel(),
            named.bytes_per_channel(),
            "bytes_per_channel disagree"
        );
        assert_eq!(
            alias.bytes_per_pixel(),
            named.bytes_per_pixel(),
            "bytes_per_pixel disagree"
        );
        assert_eq!(alias.is_float(), named.is_float(), "is_float disagrees");
        assert_eq!(
            alias.has_alpha(),
            named.has_alpha(),
            "has_alpha disagrees: FloatF32(4) says {} and RgbaF32 says {}",
            alias.has_alpha(),
            named.has_alpha()
        );
        assert_eq!(
            alias.with_alpha(),
            named.with_alpha(),
            "with_alpha disagrees"
        );
        assert_eq!(
            alias.without_alpha(),
            named.without_alpha(),
            "without_alpha disagrees"
        );
    }

    /**
     * Tests that no tuple spelling of a layout behaves differently from the
     * named variant with_channels canonicalizes it to, for any band count
     * a named variant exists for.
     * Works by sweeping band counts 1 to 8 across all three byte depths,
     * building the tuple variant directly and the canonical one through
     * with_channels, and comparing every accessor. Counts with no named
     * variant (2, 5, 6, 7, 8, and 1/3 at the float depth) compare a value
     * against itself and hold trivially, which is what makes the sweep safe
     * to state over the whole range.
     * Input: Multi8(4) vs Rgba8, Multi16(1) vs Gray16, FloatF32(4) vs
     * RgbaF32, ... -> Output: identical answers in every row.
     */
    #[test]
    fn every_tuple_spelling_behaves_like_its_canonical_form() {
        for n in 1..=8u16 {
            let nz = NonZeroU16::new(n).expect("n starts at 1");
            for (tuple, depth) in [
                (PixelFormat::Multi8(nz), 1usize),
                (PixelFormat::Multi16(nz), 2),
                (PixelFormat::FloatF32(nz), 4),
            ] {
                let named = PixelFormat::with_channels(usize::from(n), depth)
                    .expect("1..=8 bands at depth 1/2/4 is a valid format");
                assert_eq!(
                    tuple.channels(),
                    named.channels(),
                    "{tuple:?} and {named:?} disagree on channels"
                );
                assert_eq!(
                    tuple.bytes_per_channel(),
                    named.bytes_per_channel(),
                    "{tuple:?} and {named:?} disagree on bytes_per_channel"
                );
                assert_eq!(
                    tuple.bytes_per_pixel(),
                    named.bytes_per_pixel(),
                    "{tuple:?} and {named:?} disagree on bytes_per_pixel"
                );
                assert_eq!(
                    tuple.is_float(),
                    named.is_float(),
                    "{tuple:?} and {named:?} disagree on is_float"
                );
                assert_eq!(
                    tuple.has_alpha(),
                    named.has_alpha(),
                    "{tuple:?} and {named:?} disagree on has_alpha"
                );
                assert_eq!(
                    tuple.with_alpha(),
                    named.with_alpha(),
                    "{tuple:?} and {named:?} disagree on with_alpha"
                );
                assert_eq!(
                    tuple.without_alpha(),
                    named.without_alpha(),
                    "{tuple:?} and {named:?} disagree on without_alpha"
                );
            }
        }
    }

    /**
     * Tests that the alpha helpers land on the named variant rather than on
     * the tuple spelling of the same layout, so a promotion or demotion
     * cannot introduce an alias that was not there before.
     * Works by promoting and demoting every row of the alias table and
     * asserting the result is the named variant's answer, by value.
     * Input: Multi8(1).with_alpha() -> Rgba8; Multi8(4).without_alpha() ->
     * Rgb8; FloatF32(4).without_alpha() -> FloatF32(3).
     */
    #[test]
    fn alpha_helpers_land_on_the_named_variant() {
        for (alias, named) in alias_table() {
            assert_eq!(
                alias.with_alpha(),
                named.with_alpha(),
                "{alias:?}.with_alpha() must match {named:?}.with_alpha()"
            );
            assert_eq!(
                alias.without_alpha(),
                named.without_alpha(),
                "{alias:?}.without_alpha() must match {named:?}.without_alpha()"
            );
        }
    }

    /**
     * Tests that canonical maps every non-canonical spelling to its named
     * variant and leaves everything else alone, which is the table issue
     * #531 enumerates.
     * Works by asserting the seven rows by value, then asserting canonical
     * is the identity on all seven named variants and on the tuple
     * spellings that have no named twin, and that is_canonical agrees with
     * it everywhere.
     * Input: FloatF32(4) -> RgbaF32, Multi8(3) -> Rgb8, ...; FloatF32(3),
     * Multi8(2), Rgb8 -> unchanged.
     */
    #[test]
    fn canonical_maps_the_alias_table_and_nothing_else() {
        let nz = |n: u16| NonZeroU16::new(n).expect("the table holds no zeroes");

        for (alias, named) in alias_table() {
            assert_eq!(
                alias.canonical(),
                named,
                "{alias:?} must canonicalize to {named:?}"
            );
            assert!(
                !alias.is_canonical(),
                "{alias:?} is not the canonical spelling of its layout"
            );
            assert!(named.is_canonical(), "{named:?} is canonical");
        }

        // The named variants and the tuple spellings with no named twin are
        // fixed points. FloatF32(1) and FloatF32(3) are in this list on
        // purpose: four is the only float band count with a named variant,
        // so the float row of the table has one entry where the 8- and
        // 16-bit rows have three.
        for fmt in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::Rgb8,
            PixelFormat::Rgba8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba16,
            PixelFormat::RgbaF32,
            PixelFormat::Multi8(nz(2)),
            PixelFormat::Multi8(nz(7)),
            PixelFormat::Multi16(nz(2)),
            PixelFormat::Multi16(nz(5)),
            PixelFormat::FloatF32(nz(1)),
            PixelFormat::FloatF32(nz(3)),
            PixelFormat::FloatF32(nz(7)),
        ] {
            assert_eq!(
                fmt.canonical(),
                fmt,
                "{fmt:?} is already canonical and must be left alone"
            );
            assert!(fmt.is_canonical(), "{fmt:?} must report as canonical");
        }
    }

    /**
     * Tests exactly which band counts have a named variant, which is what
     * decides the shape of the alias table: a count with a named variant is
     * one where a tuple spelling of the same layout also exists.
     * Works by sweeping band counts 1 to 300 across all three byte depths
     * and asserting with_channels lands on a named variant for precisely
     * 1, 3 and 4 at the 8- and 16-bit depths and precisely 4 at the float
     * depth, and on a tuple variant everywhere else. Asserting
     * `is_canonical()` here instead would pin nothing: `canonical` is
     * defined as `with_channels`, so it holds for any implementation.
     * Input: (1..=300) x {1, 2, 4} -> Output: named at (1|3|4, 1|2) and
     * (4, 4), tuple elsewhere.
     */
    #[test]
    fn with_channels_uses_a_named_variant_for_exactly_the_aliased_counts() {
        for n in 1..=300usize {
            for depth in [1usize, 2, 4] {
                let fmt = PixelFormat::with_channels(n, depth)
                    .expect("1..=300 bands at depth 1/2/4 is a valid format");
                let is_named = matches!(
                    fmt,
                    PixelFormat::Gray8
                        | PixelFormat::Gray16
                        | PixelFormat::Rgb8
                        | PixelFormat::Rgba8
                        | PixelFormat::Rgb16
                        | PixelFormat::Rgba16
                        | PixelFormat::RgbaF32
                );
                let wants_named = if depth == 4 {
                    n == 4
                } else {
                    n == 1 || n == 3 || n == 4
                };
                assert_eq!(
                    is_named,
                    wants_named,
                    "with_channels({n}, {depth}) produced {fmt:?}, which is \
                     {}a named variant",
                    if is_named { "" } else { "not " }
                );
                assert!(
                    fmt.is_canonical(),
                    "with_channels({n}, {depth}) produced the non-canonical {fmt:?}"
                );
            }
        }
    }
}
