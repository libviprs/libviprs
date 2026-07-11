use core::num::NonZeroU16;

/// Canonical pixel formats used throughout the pipeline.
///
/// Source images are normalized into one of these formats at decode time.
/// This keeps format-specific complexity out of the planner, execution engine,
/// and [`Raster`](crate::raster::Raster) buffer management. Every named format
/// is defined by two axes -- channel count (1, 3, or 4) and bit depth (8 or 16
/// bits per channel) -- giving six named variants. The band operations in
/// [`crate::bands`] can produce intermediate images with any other band count
/// (for example 2 bands from `extract_bands`, or 100 bands from `bandfold`);
/// those are carried by the `Multi8` / `Multi16` variants.
///
/// # Variants
///
/// | Variant     | Channels | Bits/channel | Bytes/pixel |
/// |-------------|----------|--------------|-------------|
/// | `Gray8`     | 1        | 8            | 1           |
/// | `Gray16`    | 1        | 16           | 2           |
/// | `Rgb8`      | 3        | 8            | 3           |
/// | `Rgba8`     | 4        | 8            | 4           |
/// | `Rgb16`     | 3        | 16           | 6           |
/// | `Rgba16`    | 4        | 16           | 8           |
/// | `Multi8(n)` | n        | 8            | n           |
/// | `Multi16(n)`| n        | 16           | 2n          |
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
    /// N-channel 8-bit multiband image, produced by the band operations in
    /// [`crate::bands`] when the band count is not 1, 3, or 4. Multiband
    /// rasters are compute intermediates: the decode, resize, and tile
    /// encoding paths do not accept them.
    Multi8(NonZeroU16),
    /// N-channel 16-bit multiband image; see [`PixelFormat::Multi8`].
    Multi16(NonZeroU16),
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
            Self::Rgba8 | Self::Rgba16 => 4,
            Self::Multi8(n) | Self::Multi16(n) => n.get() as usize,
        }
    }

    /// The canonical format for a channel count and byte depth.
    ///
    /// Counts 1, 3, and 4 map to the named `Gray` / `Rgb` / `Rgba` variants;
    /// every other count maps to `Multi8` / `Multi16`. The band operations use
    /// this so a 3-band multiband result compares equal to `Rgb8` rather than
    /// living as a `Multi8(3)` alias.
    ///
    /// Returns `None` when `channels` is 0 or above `u16::MAX`, or when
    /// `bytes_per_channel` is not 1 or 2.
    pub fn with_channels(channels: usize, bytes_per_channel: usize) -> Option<Self> {
        let fmt = match (channels, bytes_per_channel) {
            (1, 1) => Self::Gray8,
            (1, 2) => Self::Gray16,
            (3, 1) => Self::Rgb8,
            (3, 2) => Self::Rgb16,
            (4, 1) => Self::Rgba8,
            (4, 2) => Self::Rgba16,
            (n, 1) => Self::Multi8(NonZeroU16::new(u16::try_from(n).ok()?)?),
            (n, 2) => Self::Multi16(NonZeroU16::new(u16::try_from(n).ok()?)?),
            _ => return None,
        };
        Some(fmt)
    }

    /// Whether this format includes an alpha (transparency) channel.
    ///
    /// Returns `true` only for `Rgba8` and `Rgba16`.
    pub fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Rgba16)
    }

    /// Bytes per channel sample (1 for 8-bit formats, 2 for 16-bit formats).
    ///
    /// Useful when converting between bit depths or when working with raw
    /// sample values that need to be read as `u8` vs `u16`.
    pub fn bytes_per_channel(self) -> usize {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 | Self::Multi8(_) => 1,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 | Self::Multi16(_) => 2,
        }
    }

    /// Return the variant of this format that includes an alpha channel.
    ///
    /// `Gray8` and `Gray16` promote to `Rgba8` / `Rgba16` respectively (not
    /// `GrayAlpha`), because the pipeline does not use a gray+alpha format.
    /// If the format already has alpha, returns `self` unchanged. The
    /// multiband variants have no alpha concept and are returned unchanged.
    pub fn with_alpha(self) -> Self {
        match self {
            Self::Gray8 => Self::Rgba8,
            Self::Gray16 => Self::Rgba16,
            Self::Rgb8 => Self::Rgba8,
            Self::Rgb16 => Self::Rgba16,
            other => other,
        }
    }

    /// Return the variant of this format with the alpha channel removed.
    ///
    /// `Rgba8` demotes to `Rgb8`, `Rgba16` to `Rgb16`. Formats without alpha
    /// are returned unchanged.
    pub fn without_alpha(self) -> Self {
        match self {
            Self::Rgba8 => Self::Rgb8,
            Self::Rgba16 => Self::Rgb16,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
