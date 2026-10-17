//! The decoder contract and the primitive IR the vector pipeline sits on.
//!
//! This module is the boundary that keeps a CAD parser out of libviprs core. A
//! drawing arrives as a [`CadSource`], a [`CadDecoder`] opens it into a
//! [`CadDrawing`], and decoding one of that drawing's [`CadView`]s pushes
//! [`Primitive`]s at a [`PrimitiveSink`] and hands back a [`DecodeReport`].
//! Those six names are the whole contract, and nothing behind them is visible
//! through it.
//!
//! ```text
//! CadSource  ->  CadDecoder::open  ->  CadDrawing::views
//!                                      CadDrawing::decode  ->  PrimitiveSink
//!                                                          ->  DecodeReport
//! ```
//!
//! # No ACadSharp, no .NET, no FFI handle
//!
//! The provider this pipeline ships with is `acadsharp-rs` over an ACadSharp
//! NativeAOT library, and none of that appears here: not a type, not a
//! feature, not a transitive dependency. The module compiles with nothing
//! behind it, and a decoder written against these traits needs no CAD library
//! present — [the example below](#examples) is one, and it runs in this
//! crate's doctests on every build.
//!
//! That is not politeness towards a future rewrite. It is what makes the
//! tiler, the MVT encoder and the PMTiles writer testable at all: a fixture
//! decoder that emits nine primitives is three lines, where standing up a real
//! DWG reader is a native artifact, an ABI handshake and a 20 MB drawing.
//!
//! # Curves stay curves
//!
//! An [`Arc`] is a centre, a radius and two angles. A [`Spline`] is a degree, a
//! knot vector and control points. Neither is a run of line segments, and this
//! module has no tessellator in it.
//!
//! Tessellating needs a deviation budget, and the budget depends on the zoom
//! level a tile is being cut for — the same arc is eight segments at z12 and
//! two hundred at z20. A decoder does not know the zoom, so a decoder that
//! tessellated would have to pick a tolerance it is not in a position to pick,
//! and the error would be baked in before the tiler ever saw it. For the same
//! reason a [`Polyline`] carries its bulges verbatim rather than the arcs they
//! stand for, and every coordinate is three-dimensional rather than projected:
//! flattening to a plane also needs a choice, and the choice belongs to the
//! tiler.
//!
//! # Invariants are enforced at construction
//!
//! Every primitive has private fields, a validating constructor and accessors.
//! There is no way to build one out of numbers that do not describe a shape:
//! not a `NaN` coordinate, not a zero-length normal, not a negative radius,
//! not an arc that sweeps nothing, not a knot vector too short for its control
//! points, not a polyline of one point.
//!
//! The alternative — public fields and a `validate()` somebody remembers to
//! call — puts garbage geometry one forgotten call away from the tiler, which
//! would then compute a bounding box containing a `NaN` and poison every union
//! above it. A malformed entity is supposed to become a [`Diagnostic`] its
//! provider reports, and that only happens if construction is the thing that
//! refuses it. Each refusal is a typed [`CadError`] naming the field and
//! quoting the value, so the provider has something to put in the report.
//!
//! # A primitive's text is the decoded string a person reads
//!
//! This is a contract rather than a convention, because two independent
//! readers of the same drawings disagreed about what the text said, and both
//! disagreements were silent.
//!
//! One returned `\U+220545,6` where the other returned `∅45,6`. `\U+XXXX` is
//! MIF, the escape a writer emits for a character outside the file's code
//! page; it is *transport*, and the string the user sees is `∅45,6`. A reader
//! with no MIF layer hands the escape through as though it were content. The
//! other disagreement was a multiline attribute: one reader returned its text,
//! the other returned `""`, and the file carried the text all along.
//!
//! So [`Text`] is built to make both of those unrepresentable:
//!
//! * [`Text::new`] refuses a string containing an undecoded `\U+XXXX` (MIF) or
//!   `\M+NXXXX` (CIF) escape, with [`CadError::TextEscapeNotDecoded`]. A
//!   provider that cannot decode one reports
//!   [`DiagnosticCode::TEXT_ESCAPE_NOT_DECODED`] instead of emitting it.
//! * [`Text::new`] refuses the empty string, with [`CadError::TextEmpty`]. An
//!   empty string is indistinguishable from an entity that legitimately draws
//!   nothing, so it is not available as a way to say "I could not read this";
//!   unrecovered text is [`DiagnosticCode::TEXT_NOT_RECOVERED`].
//! * [`undecoded_escape`] is public, so a provider that wants to decode the
//!   escape itself can find it before construction rather than learning about
//!   it from an error.
//!
//! MTEXT *formatting* codes (`\P`, `{\fArial|b0|i0;…}`, stacked fractions) are
//! content, not transport: they describe how present characters are laid out
//! and do not change which characters the drawing holds. They cross unchanged
//! and are somebody else's issue.
//!
//! Detection is syntactic, and that is a deliberate trade. A drawing whose
//! text genuinely contains the six characters `\U+00B0` would be refused —
//! AutoCAD writes that as `\\U+00B0`, which this module skips, so it should
//! not arise. Refusing a string that is almost certainly an undecoded escape
//! is visible and recoverable; emitting one as content is neither.
//!
//! # One open handle per drawing
//!
//! [`CadDecoder::probe`] reads a header and takes a [`CadSource`], because
//! that is all it needs. Listing views and decoding one both need the drawing
//! *open*, so they live on [`CadDrawing`] and not on the decoder: a `views`
//! that took a source would re-parse a 200 MB DWG to answer a question the
//! last call already had the answer to.
//!
//! [`CadDrawing`] is deliberately neither [`Send`] nor [`Sync`], and so is
//! whatever a provider returns from [`CadDecoder::open`]. The provider's own
//! handles are neither — one decode handle is single-threaded and calls on it
//! must not overlap — and a trait that promised more than the provider can
//! deliver would be a lie the tiler builds on. [`CadDrawing::decode`] takes
//! `&self` rather than `&mut self`. `!Sync` rules out *concurrent* overlap,
//! since two threads cannot hold a shared reference to one handle, but it does
//! not rule out *re-entrant* overlap: a sink may hold a `&dyn CadDrawing` and
//! call back into it from `primitive`, and that compiles. So the contract
//! forbids it instead: a sink must not call back into the drawing it is being
//! fed by, and [`CadDrawing`] has the example and the consequences.
//!
//! Parallelism in this pipeline is per tile, above the decoder, over
//! primitives that have already been collected.
//!
//! # No Cargo feature, and please do not add one
//!
//! This module is always compiled. It has no dependencies to gate: it is
//! arithmetic, `Vec`, `String` and `thiserror`, all of which are already here.
//!
//! The provider is the part that costs something, and gating *it* is
//! `dwg-acadsharp = ["dep:acadsharp-rs"]`, added where the provider is. A
//! feature here would buy nothing and would drag in the crate's whole
//! feature-table apparatus — `tests/ci_feature_coverage.rs`, the feature table
//! in this crate's own documentation, `LINTED_FEATURES` in the `Makefile` and a
//! clippy cell in CI — for a module whose entire cost is compiling.
//!
//! # Examples
//!
//! A decoder with no CAD library behind it, driving both sink paths.
//!
//! ```
//! use libviprs::cad::{
//!     CadDecoder, CadDrawing, CadError, CadSource, CadView, CollectSink, DecodeReport, Line,
//!     PrimitiveKind, PrimitiveSink,
//! };
//!
//! struct Ladder;
//! struct OpenLadder;
//!
//! impl CadDecoder for Ladder {
//!     fn probe(&self, _source: CadSource<'_>) -> Result<bool, CadError> {
//!         Ok(true)
//!     }
//!
//!     fn open<'a>(&'a self, _source: CadSource<'a>) -> Result<Box<dyn CadDrawing + 'a>, CadError> {
//!         Ok(Box::new(OpenLadder))
//!     }
//! }
//!
//! impl CadDrawing for OpenLadder {
//!     fn views(&self) -> Result<Vec<CadView>, CadError> {
//!         Ok(vec![CadView::new(0, 0, [0.0, 0.0, 10.0, 2.0], 3, "Model")])
//!     }
//!
//!     fn decode(
//!         &self,
//!         view: u32,
//!         sink: &mut dyn PrimitiveSink,
//!         report: &mut DecodeReport,
//!     ) -> Result<(), CadError> {
//!         if view != 0 {
//!             return Err(CadError::NoSuchView { index: view, views: 1 });
//!         }
//!         for rung in 0..3 {
//!             let y = f64::from(rung);
//!             sink.primitive(Line::new([0.0, y, 0.0], [10.0, y, 0.0])?.into())?;
//!             report.record(PrimitiveKind::Line);
//!         }
//!         // Only after the totals say the stream ended for a good reason.
//!         report.mark_complete();
//!         // The implementation finishes the sink it was handed.
//!         sink.finish()
//!     }
//! }
//!
//! let drawing = Ladder.open(CadSource::Bytes(b"no CAD library was harmed"))?;
//! assert_eq!(drawing.views()?[0].name(), "Model");
//!
//! let mut sink = CollectSink::new();
//! let mut report = DecodeReport::new();
//! drawing.decode(0, &mut sink, &mut report)?;
//!
//! assert_eq!(sink.collected().len(), 3);
//! assert_eq!(report.counts().get(PrimitiveKind::Line), 3);
//! assert!(report.is_complete());
//! assert!(sink.is_finished(), "a complete decode finishes its sink");
//! # Ok::<(), CadError>(())
//! ```

#[cfg(feature = "dwg-acadsharp")]
pub mod acadsharp;
mod decoder;
mod primitive;
mod report;
mod text;

pub use decoder::{
    CadDecoder, CadDrawing, CadSource, CadView, CollectSink, Extents, PrimitiveSink, ViewKind,
};
pub use primitive::{
    Arc, Circle, Ellipse, ItemHandle, Line, Origin, Point3, Polygon, Polyline, Primitive,
    PrimitiveKind, Spline, Text, Vector3, undecoded_escape,
};
pub use report::{DecodeReport, Diagnostic, DiagnosticCode, PrimitiveCounts, Severity};
pub use text::{DisplayText, display_text};

use thiserror::Error;

/// Everything this module refuses, and the two ways a provider can fail.
///
/// Most variants are a constructor's refusal, and each one names the field it
/// rejected and quotes the value it rejected it for: "invalid arc" without the
/// numbers that made it invalid is not something a provider can put in a
/// [`Diagnostic`] or a person can act on.
///
/// The two that are not refusals are [`CadError::Provider`] and
/// [`CadError::Sink`]. Both carry somebody else's concrete error as
/// [`std::error::Error::source`] rather than its `to_string()`, so a caller
/// can downcast back to it.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CadError {
    /// A coordinate or scalar that is not a finite number.
    #[error("{field} is {value}, and every number in the IR is finite")]
    NonFinite {
        /// The field that carried it.
        field: &'static str,
        /// The value, so a log says which of `NaN`, `inf` and `-inf` it was.
        value: f64,
    },

    /// A magnitude that has to be above zero: a radius, a ratio, a height.
    #[error("{field} is {value}, which is not above zero")]
    NonPositive {
        /// The field that carried it.
        field: &'static str,
        /// The value.
        value: f64,
    },

    /// A direction vector of zero length, which names no direction and — for a
    /// normal — defines no plane for angles to be measured in.
    #[error("{field} has zero length, so it names no direction")]
    ZeroLengthVector {
        /// The field that carried it.
        field: &'static str,
    },

    /// A curve whose parameter range starts and ends at the same place, so it
    /// sweeps nothing and draws nothing.
    #[error("{field} starts and ends at {value}, so it sweeps nothing")]
    EmptySweep {
        /// The field pair that coincided.
        field: &'static str,
        /// The value both ends held.
        value: f64,
    },

    /// A vertex run too short to be a path.
    #[error("{field} has {got} vertices, and {need} is the minimum")]
    TooFewVertices {
        /// The field that carried the run.
        field: &'static str,
        /// How many vertices there were.
        got: usize,
        /// How many there have to be.
        need: usize,
    },

    /// A per-span or per-point array whose length does not match the run it
    /// annotates: bulges against vertices, weights against control points.
    #[error("{field} has {got} entries against {expected} {against}")]
    CountMismatch {
        /// The annotating array.
        field: &'static str,
        /// Its length.
        got: usize,
        /// The length it has to have.
        expected: usize,
        /// What it has to match, for the message.
        against: &'static str,
    },

    /// A spline degree of zero, which describes points rather than a curve.
    #[error("spline degree {degree} describes no curve")]
    SplineDegree {
        /// The degree.
        degree: u32,
    },

    /// Fewer control points than the degree needs for even one span.
    #[error(
        "a degree {degree} spline needs {need} control points at the least, and this has {got}"
    )]
    SplineControlCount {
        /// The degree.
        degree: u32,
        /// The minimum for that degree.
        need: usize,
        /// How many there were.
        got: usize,
    },

    /// A knot vector too short to parameterise its control points.
    #[error(
        "{got} knots cannot parameterise {controls} control points, which need {need} at the least"
    )]
    SplineKnotCount {
        /// How many knots there were.
        got: usize,
        /// How many control points they had to cover.
        controls: usize,
        /// The minimum knot count for that many control points.
        need: usize,
    },

    /// A knot vector that decreases. A knot vector is non-decreasing by
    /// definition, and one that is not cannot be evaluated at all.
    #[error(
        "knot {index} is {found}, below the {previous} before it, and a knot vector never decreases"
    )]
    KnotsDecrease {
        /// Where it turned around.
        index: usize,
        /// The knot before it.
        previous: f64,
        /// The knot that was lower.
        found: f64,
    },

    /// Text carrying a transport escape nobody decoded.
    ///
    /// A primitive's text is the decoded string a person reads, so the escape
    /// cannot cross as content. Report
    /// [`DiagnosticCode::TEXT_ESCAPE_NOT_DECODED`] instead, and see this
    /// module's documentation for why.
    #[error(
        "text carries the undecoded transport escape {escape:?}, which is not what the drawing says"
    )]
    TextEscapeNotDecoded {
        /// The escape, as it appeared.
        escape: String,
    },

    /// Text with no content at all.
    ///
    /// An empty string cannot mean "I could not read this", because it is
    /// indistinguishable from an entity that draws nothing. Report
    /// [`DiagnosticCode::TEXT_NOT_RECOVERED`] instead.
    #[error(
        "a text primitive with no content is not drawable, and empty is not how a provider reports failure"
    )]
    TextEmpty,

    /// The drawing has no view with that index.
    #[error("no view {index} in this drawing, which has {views}")]
    NoSuchView {
        /// The index that was asked for.
        index: u32,
        /// How many views there are.
        views: u32,
    },

    /// The source is not a drawing this decoder reads.
    #[error("{decoder} does not recognise this source as a drawing it can read")]
    UnrecognisedSource {
        /// The decoder that refused it.
        decoder: &'static str,
    },

    /// Reading the drawing failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The backing decoder's own failure, carried rather than stringified.
    #[error("the {provider} decoder failed")]
    Provider {
        /// Which provider, for a log that has more than one.
        provider: &'static str,
        /// The provider's own error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The sink refused a primitive, carried rather than stringified.
    #[error("the primitive sink refused a primitive")]
    Sink(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl CadError {
    /// Wrap a provider's own error, so a provider can write
    /// `.map_err(|e| CadError::provider("acadsharp", e))`.
    #[must_use]
    pub fn provider(
        provider: &'static str,
        source: impl Into<Box<dyn std::error::Error + Send + Sync>>,
    ) -> Self {
        Self::Provider {
            provider,
            source: source.into(),
        }
    }

    /// Wrap a sink's own error, so a sink can write `.map_err(CadError::sink)`.
    #[must_use]
    pub fn sink(source: impl Into<Box<dyn std::error::Error + Send + Sync>>) -> Self {
        Self::Sink(source.into())
    }

    /// This error as a diagnostic: its own code, that code's default severity,
    /// and its message.
    ///
    /// The constructor-side twin of [`Diagnostic::from_code`], so a refusal that
    /// reaches a report this way reads the same as the one the provider filed
    /// for the same condition.
    #[must_use]
    pub fn to_diagnostic(&self) -> Diagnostic {
        Diagnostic::from_code(self.diagnostic_code(), self.to_string())
    }

    /// The diagnostic code a provider files for this error, so it reaches a
    /// [`DecodeReport`] under the right code rather than under a guess.
    ///
    /// Four groups, and the match spells out every variant so that a new
    /// `CadError` is a compile error here and has to be given a code
    /// deliberately:
    ///
    /// * The text contract's two refusals get their own codes because they are
    ///   the ones a release gate has to count separately.
    /// * A non-finite value is [`DiagnosticCode::NON_FINITE_GEOMETRY`], the
    ///   number the provider files for the same condition.
    /// * Every other constructor refusal is a malformed entity,
    ///   [`DiagnosticCode::PRIMITIVE_REFUSED`].
    /// * The five that are not a refusal of an entity at all (the view or the
    ///   source was wrong, the read failed, the provider or the sink failed)
    ///   are [`DiagnosticCode::DECODE_FAILED`], so a gate counting malformed
    ///   entities is not counting a disk error.
    ///
    /// ```
    /// use libviprs::cad::{CadError, DiagnosticCode, Text};
    ///
    /// let refused = Text::new([0.0; 3], 2.5, 0.0, r"94\U+00B0").unwrap_err();
    /// assert_eq!(refused.diagnostic_code(), DiagnosticCode::TEXT_ESCAPE_NOT_DECODED);
    ///
    /// let empty = Text::new([0.0; 3], 2.5, 0.0, "").unwrap_err();
    /// assert_eq!(empty.diagnostic_code(), DiagnosticCode::TEXT_NOT_RECOVERED);
    ///
    /// let bad = CadError::NonPositive { field: "radius", value: -1.0 };
    /// assert_eq!(bad.diagnostic_code(), DiagnosticCode::PRIMITIVE_REFUSED);
    ///
    /// let gone = CadError::NoSuchView { index: 3, views: 1 };
    /// assert_eq!(gone.diagnostic_code(), DiagnosticCode::DECODE_FAILED);
    /// ```
    #[must_use]
    pub const fn diagnostic_code(&self) -> DiagnosticCode {
        match self {
            Self::TextEscapeNotDecoded { .. } => DiagnosticCode::TEXT_ESCAPE_NOT_DECODED,
            Self::TextEmpty => DiagnosticCode::TEXT_NOT_RECOVERED,
            Self::NonFinite { .. } => DiagnosticCode::NON_FINITE_GEOMETRY,
            Self::NonPositive { .. }
            | Self::ZeroLengthVector { .. }
            | Self::EmptySweep { .. }
            | Self::TooFewVertices { .. }
            | Self::CountMismatch { .. }
            | Self::SplineDegree { .. }
            | Self::SplineControlCount { .. }
            | Self::SplineKnotCount { .. }
            | Self::KnotsDecrease { .. } => DiagnosticCode::PRIMITIVE_REFUSED,
            Self::NoSuchView { .. }
            | Self::UnrecognisedSource { .. }
            | Self::Io(_)
            | Self::Provider { .. }
            | Self::Sink(_) => DiagnosticCode::DECODE_FAILED,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A provider's error is carried, not flattened into a message.
    ///
    /// The whole point of the two type-erased variants is that a caller can
    /// get the concrete error back; a `to_string()` in a `String` field would
    /// read the same in a log and be useless here.
    #[test]
    fn a_provider_error_downcasts_back_to_its_own_type() {
        #[derive(Debug, thiserror::Error)]
        #[error("the handshake found version {found}")]
        struct Handshake {
            found: u32,
        }

        let err = CadError::provider("acadsharp", Handshake { found: 3 });
        let source = std::error::Error::source(&err).expect("the provider error is the source");
        let concrete = source
            .downcast_ref::<Handshake>()
            .expect("and it is still a Handshake");

        assert_eq!(
            concrete.found, 3,
            "the provider's own fields have to survive the wrap, or the variant \
             may as well have taken a String"
        );
    }

    /// The same for a sink, which is the other half of the boundary.
    #[test]
    fn a_sink_error_downcasts_back_to_its_own_type() {
        let err = CadError::sink(std::io::Error::other("disk went away"));
        let source = std::error::Error::source(&err).expect("the sink error is the source");

        assert!(
            source.downcast_ref::<std::io::Error>().is_some(),
            "a sink's io::Error has to come back as an io::Error"
        );
    }

    /// Every refusal maps onto a code, the text contract's two do not
    /// collapse onto the generic one, and neither does the non-finite
    /// refusal — the one arm of the mapping nothing else reaches.
    ///
    /// This is the negative control for `diagnostic_code`: a mapping that
    /// answered `PRIMITIVE_REFUSED` for everything would satisfy an
    /// "it returns a code" assertion and lose exactly the distinction the
    /// contract exists to make.
    #[test]
    fn the_text_contracts_refusals_do_not_share_the_generic_code() {
        let codes = [
            CadError::TextEscapeNotDecoded {
                escape: r"\U+00B0".to_owned(),
            }
            .diagnostic_code(),
            CadError::TextEmpty.diagnostic_code(),
            CadError::NonFinite {
                field: "centre",
                value: f64::NAN,
            }
            .diagnostic_code(),
            CadError::NonPositive {
                field: "radius",
                value: 0.0,
            }
            .diagnostic_code(),
        ];

        assert_eq!(
            codes.len(),
            codes
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "two of the four in {codes:?} answered the same code, so a report \
             cannot tell an undecoded escape from a non-finite coordinate or \
             a negative radius"
        );
    }

    /// A non-finite refusal files `NON_FINITE_GEOMETRY`, which is 107, in the
    /// contract band, on purpose: it is the same condition a provider files
    /// under the same number when it catches the value first, so the two
    /// layers have to agree on the code or one drawing reports two things.
    /// Pinned here because nothing else reaches the arm.
    #[test]
    fn a_non_finite_refusal_files_the_contract_code_the_provider_also_files() {
        let code = CadError::NonFinite {
            field: "centre",
            value: f64::INFINITY,
        }
        .diagnostic_code();

        assert_eq!(code, DiagnosticCode::NON_FINITE_GEOMETRY);
        assert_eq!(code.get(), 107, "the number is the specification's");
        assert!(code.is_contract_allocated());
        assert!(!code.is_core_allocated());
        assert!(!code.is_provider_allocated());
        assert_ne!(
            code,
            DiagnosticCode::PRIMITIVE_REFUSED,
            "the arm must not collapse onto the generic refusal"
        );
    }

    /// The five variants that are not a constructor's refusal of an entity do
    /// not answer the code a release gate counts as malformed entities.
    ///
    /// A disk error, an FFI handshake failure and a sink refusal filed as
    /// `PRIMITIVE_REFUSED` would be counted as bad geometry. They get a code of
    /// their own, in the core band, and all five land on the same one.
    #[test]
    fn a_failure_that_is_not_a_refusal_does_not_file_as_a_refused_entity() {
        let failures = [
            CadError::NoSuchView { index: 4, views: 2 },
            CadError::UnrecognisedSource { decoder: "ladder" },
            CadError::Io(std::io::Error::other("disk went away")),
            CadError::provider("acadsharp", std::io::Error::other("handshake")),
            CadError::sink(std::io::Error::other("full")),
        ];

        for failure in &failures {
            let code = failure.diagnostic_code();
            assert_ne!(
                code,
                DiagnosticCode::PRIMITIVE_REFUSED,
                "{failure:?} is not a refused entity"
            );
            assert_eq!(code, DiagnosticCode::DECODE_FAILED, "{failure:?}");
            assert!(code.is_core_allocated());
        }
    }

    /// The control for the one above: every constructor refusal still answers
    /// the generic code, so a mapping that sent everything to the new code
    /// would not pass either.
    #[test]
    fn every_constructor_refusal_still_files_as_a_refused_entity() {
        let refusals = [
            CadError::NonPositive {
                field: "radius",
                value: 0.0,
            },
            CadError::ZeroLengthVector { field: "normal" },
            CadError::EmptySweep {
                field: "angle",
                value: 1.0,
            },
            CadError::TooFewVertices {
                field: "vertex",
                got: 1,
                need: 2,
            },
            CadError::CountMismatch {
                field: "bulge",
                got: 1,
                expected: 2,
                against: "vertices",
            },
            CadError::SplineDegree { degree: 0 },
            CadError::SplineControlCount {
                degree: 3,
                need: 4,
                got: 2,
            },
            CadError::SplineKnotCount {
                got: 1,
                controls: 4,
                need: 8,
            },
            CadError::KnotsDecrease {
                index: 1,
                previous: 2.0,
                found: 1.0,
            },
        ];
        for refusal in &refusals {
            assert_eq!(
                refusal.diagnostic_code(),
                DiagnosticCode::PRIMITIVE_REFUSED,
                "{refusal:?}"
            );
        }
    }

    /// An error becomes a diagnostic under its own code, with that code's
    /// severity and its own message, so a provider files a code and does not
    /// choose a severity.
    #[test]
    fn an_error_becomes_a_diagnostic_under_its_own_code_and_severity() {
        let refusal = CadError::NonPositive {
            field: "radius",
            value: -4.5,
        };
        let diagnostic = refusal.to_diagnostic();
        assert_eq!(diagnostic.code(), refusal.diagnostic_code());
        assert_eq!(
            diagnostic.severity(),
            refusal.diagnostic_code().default_severity()
        );
        assert_eq!(diagnostic.message(), refusal.to_string());
    }

    /// Messages quote the value they rejected.
    ///
    /// A guard rather than a restatement: the crate's rule is that a refusal
    /// names the number, and a message rewritten to drop it reads fine and
    /// makes a bug report useless.
    #[test]
    fn a_refusal_names_the_field_and_the_value() {
        let err = CadError::NonPositive {
            field: "radius",
            value: -4.5,
        };
        let text = err.to_string();

        assert!(
            text.contains("radius") && text.contains("-4.5"),
            "{text:?} has to carry both the field and the value it refused"
        );
    }
}
