/// Canonical pixel formats used throughout the pipeline.
///
/// Source images are normalized into one of these formats at decode time.
/// This keeps format-specific complexity out of the planner and execution engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    Gray8,
    Gray16,
    Rgb8,
    Rgba8,
    Rgb16,
    Rgba16,
}

impl PixelFormat {
    /// Bytes per pixel for this format.
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Gray8 => 1,
            Self::Gray16 => 2,
            Self::Rgb8 => 3,
            Self::Rgba8 => 4,
            Self::Rgb16 => 6,
            Self::Rgba16 => 8,
        }
    }

    /// Number of channels.
    pub fn channels(self) -> usize {
        match self {
            Self::Gray8 | Self::Gray16 => 1,
            Self::Rgb8 | Self::Rgb16 => 3,
            Self::Rgba8 | Self::Rgba16 => 4,
        }
    }

    /// Whether this format has an alpha channel.
    pub fn has_alpha(self) -> bool {
        matches!(self, Self::Rgba8 | Self::Rgba16)
    }

    /// Bytes per channel sample (1 for 8-bit, 2 for 16-bit).
    pub fn bytes_per_channel(self) -> usize {
        match self {
            Self::Gray8 | Self::Rgb8 | Self::Rgba8 => 1,
            Self::Gray16 | Self::Rgb16 | Self::Rgba16 => 2,
        }
    }

    /// Add an alpha channel to this format. Returns self if already has alpha.
    pub fn with_alpha(self) -> Self {
        match self {
            Self::Gray8 => Self::Rgba8,
            Self::Gray16 => Self::Rgba16,
            Self::Rgb8 => Self::Rgba8,
            Self::Rgb16 => Self::Rgba16,
            other => other,
        }
    }

    /// Remove the alpha channel. Returns self if no alpha.
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

    #[test]
    fn alpha_round_trip() {
        assert_eq!(PixelFormat::Rgb8.with_alpha(), PixelFormat::Rgba8);
        assert_eq!(PixelFormat::Rgba8.without_alpha(), PixelFormat::Rgb8);
        assert_eq!(PixelFormat::Rgb16.with_alpha(), PixelFormat::Rgba16);
        assert_eq!(PixelFormat::Rgba16.without_alpha(), PixelFormat::Rgb16);
    }

    #[test]
    fn with_alpha_is_idempotent() {
        assert_eq!(PixelFormat::Rgba8.with_alpha(), PixelFormat::Rgba8);
        assert_eq!(PixelFormat::Rgba16.with_alpha(), PixelFormat::Rgba16);
    }

    #[test]
    fn without_alpha_is_idempotent() {
        assert_eq!(PixelFormat::Rgb8.without_alpha(), PixelFormat::Rgb8);
        assert_eq!(PixelFormat::Gray8.without_alpha(), PixelFormat::Gray8);
    }

    #[test]
    fn has_alpha_correctness() {
        assert!(!PixelFormat::Gray8.has_alpha());
        assert!(!PixelFormat::Rgb8.has_alpha());
        assert!(PixelFormat::Rgba8.has_alpha());
        assert!(!PixelFormat::Gray16.has_alpha());
        assert!(!PixelFormat::Rgb16.has_alpha());
        assert!(PixelFormat::Rgba16.has_alpha());
    }
}
