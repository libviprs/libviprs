//! Growable public enums must be `#[non_exhaustive]`, and the `PdfError::Pdfium`
//! variant must exist unconditionally so enabling the `pdfium` feature elsewhere
//! in the dependency graph cannot change the shape of a public enum for other
//! crates.
//!
//! Integration tests compile as an *external* crate, so `#[non_exhaustive]` is
//! observable here: a trailing `_` arm after every currently-known variant is
//! flagged `unreachable_patterns` for an exhaustive enum (rejected below by
//! `#[deny(unreachable_patterns)]`) but is *required* — and therefore reachable
//! — for a `#[non_exhaustive]` enum. Each `assert_*_non_exhaustive` function is
//! a pure compile-time check; if any of the enums below regresses to exhaustive
//! this test crate fails to build.

use libviprs::{
    Align, BandError, ColourError, Combine, DrawError, EngineEvent, Intent, Interpretation,
    JoinDirection, Layout, ManifestError, Pcs, PdfError, PixelFormat, PlannerError, Precision,
    RadianceError, RasterError, ResumeError, SourceError, VerifyError,
};

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_radiance_error_non_exhaustive(v: &RadianceError) {
    match v {
        RadianceError::BadMagic { .. } => {}
        RadianceError::TruncatedHeader { .. } => {}
        RadianceError::BadResolution { .. } => {}
        RadianceError::DimensionOutOfBounds { .. } => {}
        RadianceError::AllocLimitExceeded { .. } => {}
        RadianceError::ScanlineLengthMismatch { .. } => {}
        RadianceError::ScanlineOverrun { .. } => {}
        RadianceError::RunawayRepeat { .. } => {}
        RadianceError::RepeatWithoutPixel { .. } => {}
        RadianceError::TruncatedScanline { .. } => {}
        RadianceError::HeaderLineTooLong { .. } => {}
        RadianceError::BadHeaderLine { .. } => {}
        RadianceError::Raster(_) => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_precision_non_exhaustive(v: &Precision) {
    match v {
        Precision::Integer => {}
        Precision::Float => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_combine_non_exhaustive(v: &Combine) {
    match v {
        Combine::Max => {}
        Combine::Sum => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_pixel_format_non_exhaustive(v: &PixelFormat) {
    match v {
        PixelFormat::Gray8 => {}
        PixelFormat::Gray16 => {}
        PixelFormat::Rgb8 => {}
        PixelFormat::Rgba8 => {}
        PixelFormat::Rgb16 => {}
        PixelFormat::Rgba16 => {}
        PixelFormat::RgbaF32 => {}
        PixelFormat::FloatF32(_) => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_layout_non_exhaustive(v: &Layout) {
    match v {
        Layout::DeepZoom => {}
        Layout::Xyz => {}
        Layout::Google => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_engine_event_non_exhaustive(v: &EngineEvent) {
    match v {
        EngineEvent::SourceLoadStarted { .. } => {}
        EngineEvent::SourceLoaded { .. } => {}
        EngineEvent::PlanCreated { .. } => {}
        EngineEvent::LevelStarted { .. } => {}
        EngineEvent::TileCompleted { .. } => {}
        EngineEvent::LevelCompleted { .. } => {}
        EngineEvent::StripRendered { .. } => {}
        EngineEvent::BatchStarted { .. } => {}
        EngineEvent::BatchCompleted { .. } => {}
        EngineEvent::Finished { .. } => {}
        EngineEvent::PipelineComplete => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
// The reachable `_` arm after every known variant is the whole point of the
// check, so the single-variant match must stay a match.
#[allow(clippy::single_match)]
fn assert_draw_error_non_exhaustive(v: &DrawError) {
    match v {
        DrawError::SeedOutOfBounds { .. } => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_band_error_non_exhaustive(v: &BandError) {
    match v {
        BandError::DimensionMismatch { .. } => {}
        BandError::BandCountMismatch { .. } => {}
        BandError::BandOutOfRange { .. } => {}
        BandError::BandRangeOutOfRange { .. } => {}
        BandError::EmptyBandRange => {}
        BandError::EmptyConstants => {}
        BandError::ZeroFactor => {}
        BandError::FoldNotDivisible { .. } => {}
        BandError::UnfoldNotDivisible { .. } => {}
        BandError::TooManyBands { .. } => {}
        BandError::WidthOverflow { .. } => {}
        BandError::RankIndexOutOfRange { .. } => {}
        BandError::Raster(..) => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_raster_error_non_exhaustive(v: &RasterError) {
    match v {
        RasterError::BufferSizeMismatch { .. } => {}
        RasterError::ZeroDimension { .. } => {}
        RasterError::RegionOutOfBounds { .. } => {}
        RasterError::SizeOverflow { .. } => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_planner_error_non_exhaustive(v: &PlannerError) {
    match v {
        PlannerError::ZeroDimension { .. } => {}
        PlannerError::ZeroTileSize(..) => {}
        PlannerError::OverlapTooLarge { .. } => {}
        PlannerError::DimensionOverflow { .. } => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_source_error_non_exhaustive(v: &SourceError) {
    match v {
        SourceError::Io(..) => {}
        SourceError::Decode(..) => {}
        SourceError::UnsupportedColorType(..) => {}
        SourceError::Raster(..) => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_manifest_error_non_exhaustive(v: &ManifestError) {
    match v {
        ManifestError::Io(..) => {}
        ManifestError::Json(..) => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_resume_error_non_exhaustive(v: &ResumeError) {
    match v {
        ResumeError::SchemaMismatch { .. } => {}
        ResumeError::Corrupt { .. } => {}
        ResumeError::Io(..) => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_verify_error_non_exhaustive(v: &VerifyError) {
    match v {
        VerifyError::ManifestNotFound { .. } => {}
        VerifyError::Io { .. } => {}
        VerifyError::Json(..) => {}
        VerifyError::MissingField(..) => {}
        VerifyError::BadField { .. } => {}
        VerifyError::UnknownAlgo(..) => {}
        VerifyError::UnsafePath(..) => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_interpretation_non_exhaustive(v: &Interpretation) {
    match v {
        Interpretation::Multiband => {}
        Interpretation::Bw => {}
        Interpretation::Histogram => {}
        Interpretation::Xyz => {}
        Interpretation::Lab => {}
        Interpretation::Cmyk => {}
        Interpretation::Labq => {}
        Interpretation::Rgb => {}
        Interpretation::Cmc => {}
        Interpretation::Lch => {}
        Interpretation::Labs => {}
        Interpretation::Srgb => {}
        Interpretation::Yxy => {}
        Interpretation::Fourier => {}
        Interpretation::Rgb16 => {}
        Interpretation::Grey16 => {}
        Interpretation::Matrix => {}
        Interpretation::ScRgb => {}
        Interpretation::Hsv => {}
        Interpretation::OkLab => {}
        Interpretation::OkLch => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_colour_error_non_exhaustive(v: &ColourError) {
    match v {
        ColourError::UnknownColourspace { .. } => {}
        ColourError::UnsupportedColourspace { .. } => {}
        ColourError::TooFewBands { .. } => {}
        ColourError::DimensionMismatch { .. } => {}
        ColourError::NoProfile => {}
        ColourError::ProfileRead { .. } => {}
        ColourError::InvalidProfile { .. } => {}
        ColourError::UnsupportedDeviceSpace { .. } => {}
        ColourError::UnsupportedDepth { .. } => {}
        ColourError::IccTransform { .. } => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_intent_non_exhaustive(v: &Intent) {
    match v {
        Intent::Perceptual => {}
        Intent::Relative => {}
        Intent::Saturation => {}
        Intent::Absolute => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_pcs_non_exhaustive(v: &Pcs) {
    match v {
        Pcs::Lab => {}
        Pcs::Xyz => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
// `Compression` has exactly one variant today, so clippy reads the match as an
// equality check and suggests an `if`. The match is the point: an `if` would
// not fail to compile the day the enum regresses to exhaustive.
#[allow(clippy::single_match)]
fn assert_webp_compression_non_exhaustive(v: &libviprs::webp::Compression) {
    match v {
        libviprs::webp::Compression::Lossless => {}
        _ => {}
    }
}

/// `PdfError::Pdfium` must be constructible without the `pdfium` feature enabled
/// (this test crate builds with default features). If the variant is
/// `#[cfg(feature = "pdfium")]`-gated, this fails to compile.
#[test]
fn pdf_error_pdfium_variant_is_feature_independent() {
    let err = PdfError::Pdfium("boom".to_string());
    assert!(matches!(err, PdfError::Pdfium(msg) if msg == "boom"));
}

/// Runtime anchor so the file registers as a test target even though the
/// `#[non_exhaustive]` checks above are purely compile-time.
#[test]
fn non_exhaustive_checks_compile() {
    assert_layout_non_exhaustive(&Layout::DeepZoom);
    assert_webp_compression_non_exhaustive(&libviprs::webp::Compression::Lossless);
    assert_pixel_format_non_exhaustive(&PixelFormat::Gray8);
    assert_interpretation_non_exhaustive(&Interpretation::OkLch);
    assert_intent_non_exhaustive(&Intent::Perceptual);
    assert_pcs_non_exhaustive(&Pcs::Lab);
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_join_direction_non_exhaustive(v: &JoinDirection) {
    match v {
        JoinDirection::Horizontal => {}
        JoinDirection::Vertical => {}
        _ => {}
    }
}

#[deny(unreachable_patterns)]
#[allow(dead_code)]
fn assert_align_non_exhaustive(v: &Align) {
    match v {
        Align::Low => {}
        Align::Centre => {}
        Align::High => {}
        _ => {}
    }
}
