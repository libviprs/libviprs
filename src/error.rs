//! Crate-level umbrella error over the per-module operation errors.
//!
//! Each image-operation module owns a typed error enum ([`ArithmeticError`],
//! [`BandError`], [`ColourError`], ...) so a single-family caller sees a tight,
//! exhaustive error surface. That per-module split is deliberate, but a caller
//! *composing* several op families previously had to write bespoke `From` glue
//! to funnel the different enums into one return type (issue #285).
//!
//! [`OpError`] is that funnel: a `#[non_exhaustive]` umbrella with a `#[from]`
//! conversion for each **in-memory pixel-transform** op-family error, so a
//! function that chains, say, an arithmetic op and a band op can return
//! `Result<_, OpError>` and lean on `?` for both:
//!
//! ```
//! use libviprs::{OpError, Raster};
//!
//! fn recombine(a: &Raster, gains: &[f64]) -> Result<Raster, OpError> {
//!     let scaled = a.try_mul_vec(gains)?; // ArithmeticError -> OpError
//!     let grey = scaled.try_bandmean()?; // BandError       -> OpError
//!     Ok(grey)
//! }
//! ```
//!
//! The per-module enums are retained and remain the right choice for
//! single-family code; [`OpError`] is purely additive. It is `#[non_exhaustive]`
//! both outwardly (callers must keep a wildcard arm) and in intent: new op
//! families can add a variant without it being a breaking change.
//!
//! # Scope: what the umbrella funnels
//!
//! The umbrella is deliberately bounded, not a `From` for *every* error the
//! crate defines. It carries a `#[from]` for each **in-memory pixel-transform**
//! op-family error — arithmetic, bands, colour, composite, conversion,
//! convolution, create, draw, extract, freqfilt, histogram, matrix, morphology,
//! mosaicing, resample — plus the core [`RasterError`] those ops build on. The
//! I/O, codec, and pipeline error families (source / sink / save / metadata,
//! encode, engine / planner / resume / manifest) are intentionally **excluded**:
//! they belong to different call surfaces and would dilute the pixel-op funnel.
//!
//! # Matching raster failures: two paths
//!
//! A [`RasterError`] (an allocation, size-overflow, or malformed-construction
//! failure of the underlying pixel buffer) can reach the umbrella *two* ways:
//!
//! * **directly**, as [`OpError::Raster`], when a bare
//!   [`Raster::new`](crate::Raster::new) / resize call site raises it; and
//! * **nested inside an op-family error**, when the failure originates *inside*
//!   an op building its output — e.g. an [`ArithmeticError::Raster`] from an
//!   over-capacity arithmetic output arrives as
//!   `OpError::Arithmetic(ArithmeticError::Raster(_))`, not as
//!   `OpError::Raster`. This nested path is the common one (it covers the
//!   `AllocationFailed` / `SizeOverflow` cases raised while an op allocates).
//!
//! A matcher that wants to catch *every* raster failure must therefore inspect
//! both [`OpError::Raster`] and the transparent `Raster(_)` sub-variant of the
//! op-family errors — matching only [`OpError::Raster`] silently misses the
//! common case.
//!
//! # Matching float refusals: four enums, one spelling
//!
//! Several operations refuse a float raster rather than panicking on a 4-byte
//! sample, and because each module owns its error type the refusal has four
//! homes. They all spell it `FloatUnsupported { op }` as of issue #730, where
//! [`ConversionError`] used to say `FloatFormatUnsupported`:
//!
//! | variant | raised by |
//! |---|---|
//! | [`RasterError::FloatUnsupported`] | the sample-reading raster helpers |
//! | [`ArithmeticError::FloatUnsupported`] | the arithmetic family |
//! | [`ExtractError::FloatUnsupported`] | `embed`, `gravity`, `insert`, `smartcrop`'s analysing strategies |
//! | [`ConversionError::FloatUnsupported`] | `join`, `arrayjoin` |
//!
//! Four enums is the consequence of the per-module split above and is not
//! something to undo: a single-family caller still wants a tight surface. What
//! the uniform name buys is that a caller composing families through
//! [`OpError`] writes one shape four times instead of three plus an exception:
//!
//! ```
//! use libviprs::{ArithmeticError, ConversionError, ExtractError, OpError, RasterError};
//!
//! fn is_float_refusal(e: &OpError) -> bool {
//!     matches!(
//!         e,
//!         OpError::Raster(RasterError::FloatUnsupported { .. })
//!             | OpError::Arithmetic(ArithmeticError::FloatUnsupported { .. })
//!             | OpError::Extract(ExtractError::FloatUnsupported { .. })
//!             | OpError::Conversion(ConversionError::FloatUnsupported { .. })
//!     )
//! }
//! ```
//!
//! A predicate per enum (the shape [`crate::SourceError::is_alloc_limit`] took
//! in #686) was considered and not taken: that one composes because it collapses
//! *five variants of one enum* onto a question, where this is one variant of
//! each of four enums, so it would be four impls that still cannot be called
//! through a single type without a trait. The names doing the work is cheaper
//! and reads the same at every call site.
//!
//! A refusal can also be nested: `try_join` refuses float itself, so a caller
//! sees `OpError::Conversion(ConversionError::FloatUnsupported)`, but the
//! `try_insert` underneath would raise `ExtractError::FloatUnsupported` if it
//! were reached. That is the same two-path shape the raster section above
//! describes.
//!
//! Wrapping is transparent — [`OpError`]'s `Display` and `source` delegate to
//! the wrapped error via `#[error(transparent)]`, so no diagnostic detail is
//! lost in the conversion.

use thiserror::Error;

use crate::arithmetic::ArithmeticError;
use crate::bands::BandError;
use crate::colour::ColourError;
use crate::composite::CompositeError;
use crate::conversion::ConversionError;
use crate::convolution::ConvolutionError;
use crate::create::CreateError;
use crate::draw::DrawError;
use crate::extract::ExtractError;
use crate::freqfilt::FreqfiltError;
use crate::histogram::HistogramError;
use crate::matrix::MatrixError;
use crate::morphology::MorphologyError;
use crate::mosaicing::MosaicError;
use crate::raster::RasterError;
use crate::resample::ResampleError;

/// Umbrella error over the per-module image-operation errors.
///
/// A caller composing several op families can return `Result<_, OpError>` and
/// use `?` uniformly; each per-module error converts in via `#[from]`. See the
/// [module docs](crate::error) for the rationale and an example.
///
/// This enum is `#[non_exhaustive]`: match it with a trailing wildcard arm, and
/// expect new op families to add variants over time without a major-version
/// bump.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OpError {
    /// A core [`Raster`](crate::Raster) invariant error (allocation, size
    /// overflow, malformed construction) surfaced *directly* by a bare
    /// [`Raster::new`](crate::Raster::new) / resize call site.
    ///
    /// This is only one of the two ways a [`RasterError`] reaches the umbrella.
    /// When the same failure originates *inside* an op — the common case,
    /// including `AllocationFailed` / `SizeOverflow` raised while the op builds
    /// its output — it arrives nested in that op's error, e.g.
    /// `OpError::Arithmetic(ArithmeticError::Raster(_))`, not as this variant.
    /// A matcher that must catch every raster failure has to inspect both paths
    /// (see the [module docs](crate::error)).
    #[error(transparent)]
    Raster(#[from] RasterError),
    /// An arithmetic / statistics op error (see [`ArithmeticError`]).
    #[error(transparent)]
    Arithmetic(#[from] ArithmeticError),
    /// A band-manipulation op error (see [`BandError`]).
    #[error(transparent)]
    Band(#[from] BandError),
    /// A colour / colourspace op error (see [`ColourError`]).
    #[error(transparent)]
    Colour(#[from] ColourError),
    /// A compositing op error (see [`CompositeError`]).
    #[error(transparent)]
    Composite(#[from] CompositeError),
    /// A conversion / geometry op error (see [`ConversionError`]).
    #[error(transparent)]
    Conversion(#[from] ConversionError),
    /// A convolution op error (see [`ConvolutionError`]).
    #[error(transparent)]
    Convolution(#[from] ConvolutionError),
    /// A generator op error (see [`CreateError`]).
    #[error(transparent)]
    Create(#[from] CreateError),
    /// A drawing op error (see [`DrawError`]).
    #[error(transparent)]
    Draw(#[from] DrawError),
    /// An extraction / crop op error (see [`ExtractError`]).
    #[error(transparent)]
    Extract(#[from] ExtractError),
    /// A frequency-filter op error (see [`FreqfiltError`]).
    #[error(transparent)]
    Freqfilt(#[from] FreqfiltError),
    /// A histogram op error (see [`HistogramError`]).
    #[error(transparent)]
    Histogram(#[from] HistogramError),
    /// A matrix op error (see [`MatrixError`]).
    #[error(transparent)]
    Matrix(#[from] MatrixError),
    /// A morphology op error (see [`MorphologyError`]).
    #[error(transparent)]
    Morphology(#[from] MorphologyError),
    /// A mosaicing op error (see [`MosaicError`]).
    #[error(transparent)]
    Mosaic(#[from] MosaicError),
    /// A resample / resize op error (see [`ResampleError`]).
    #[error(transparent)]
    Resample(#[from] ResampleError),
}
