//! What a decode has to say about itself: structured diagnostics, per-kind
//! counts, and whether the stream ended for a good reason.
//!
//! # Why the reporting channel is in the first contract
//!
//! Fidelity loss nobody can see is the failure mode this pipeline exists to
//! avoid. A drawing that tiles to a blank square is obvious; a drawing missing
//! a third of its dimensions, or wearing text that says `\U+00B0` where it
//! should say `°`, renders perfectly and is wrong. So the report is not a
//! logging convenience bolted on at a release gate — it is the return value of
//! a decode, it is structured, and every count the gate wants to measure is
//! already in it.
//!
//! # A hostile drawing cannot exhaust memory through it
//!
//! A [`DecodeReport`] retains a bounded number of [`Diagnostic`]s and counts
//! the rest. An uploaded drawing is somebody else's file, and one that emits a
//! warning per entity across ten million entities would otherwise turn a
//! reporting channel into an allocator attack. The dropped count is reported
//! rather than swallowed, so a caller can tell "nothing else went wrong" from
//! "I stopped writing it down".
//!
//! The bound has exactly one exemption, and
//! [`DecodeReport::mark_truncated`] documents why: a file that floods the
//! retained set and *then* gets cut short would otherwise choose which
//! diagnostic the caller sees, and drop the only one saying the result cannot
//! be trusted.
//!
//! # Completeness is not the default
//!
//! [`DecodeReport::is_complete`] starts `false` and only
//! [`DecodeReport::mark_complete`] sets it. "The loop ended" and "the loop
//! ended for a good reason" look identical from outside, so a decoder that
//! forgets to check its totals reports an incomplete decode rather than
//! quietly passing a truncated one off as a short drawing. Truncation is
//! sticky in the same spirit: once a decode is known to have stopped short,
//! no later call talks the report back into looking finished.

use crate::cad::{ItemHandle, PrimitiveKind};
use core::fmt;

/// How much a [`Diagnostic`] matters.
///
/// Two levels, because a decode either finished with parts it could not fully
/// represent or it hit something it could not get past, and those are the only
/// two answers a caller acts on differently. A warning does not fail a decode:
/// a drawing that emits a hundred of them and finishes succeeded, and they are
/// what it has to say about the parts it could not fully carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Severity {
    /// Something was not fully represented, and the decode carried on.
    Warning,
    /// Something could not be represented at all: a refused entity, a
    /// truncated stream, text that was not recovered.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Warning => "warning",
            Self::Error => "error",
        })
    }
}

/// What a [`Diagnostic`] is about.
///
/// A newtype over the number and never a closed enum. Three bands:
///
/// * Below [`DiagnosticCode::CORE_RANGE_START`] are the provider-neutral codes
///   the CAD decode contract allocates, and the constants below are the ones
///   it has names for today.
/// * From [`DiagnosticCode::CORE_RANGE_START`] to
///   [`DiagnosticCode::PROVIDER_RANGE_START`] are the codes libviprs itself
///   allocates, which is where the text contract's two live. A separate band
///   so the two allocators cannot collide by both reaching for the next free
///   number.
/// * From [`DiagnosticCode::PROVIDER_RANGE_START`] up, whatever read the
///   drawing allocates, and a consumer is entitled to know none of them.
///
/// A closed set would break on the first real stream. Adding a code is
/// deliberately not a breaking change, so a consumer that refused an unknown
/// one would start failing on files it read correctly the day before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiagnosticCode(u32);

impl DiagnosticCode {
    /// An entity kind the provider does not flatten. The message names the
    /// source format's own type name.
    pub const UNSUPPORTED_ENTITY: Self = Self(100);
    /// Something the backing reader had to say about the file.
    pub const READER_NOTIFICATION: Self = Self(101);
    /// A dimension with no geometry block to take its lines and text from.
    pub const DIMENSION_WITHOUT_BLOCK: Self = Self(102);
    /// A hatch with no boundary loop that could become a polygon.
    pub const HATCH_PATTERN_ONLY: Self = Self(103);
    /// A boundary loop carrying an elliptical or spline edge, which follows as
    /// its own primitive rather than being approximated into the loop.
    pub const HATCH_LOOP_NOT_POLYGON: Self = Self(104);
    /// An insertion whose block could not be resolved, which is what an
    /// unresolved external reference looks like from inside.
    pub const UNRESOLVED_BLOCK: Self = Self(105);
    /// A block transform that does not scale an entity's plane uniformly.
    pub const NON_UNIFORM_BLOCK_SCALE: Self = Self(106);
    /// A geometry record whose values were not all finite, so no primitive was
    /// emitted: there is no correct number to put in its place.
    pub const NON_FINITE_GEOMETRY: Self = Self(107);
    /// This view produced no geometry at all. Read it beside what sits with
    /// it: alone it is an empty drawing, and with other diagnostics around it
    /// it is a drawing something went wrong reading.
    pub const EMPTY_VIEW: Self = Self(108);

    /// Text carrying a transport escape the provider could not decode.
    ///
    /// Half of the text contract. The escape must not cross as content, so
    /// this is what a provider files instead of emitting it.
    pub const TEXT_ESCAPE_NOT_DECODED: Self = Self(500);
    /// Text the provider could not recover.
    ///
    /// The other half. An empty string is not how failure is reported,
    /// because it is indistinguishable from an entity that draws nothing.
    pub const TEXT_NOT_RECOVERED: Self = Self(501);
    /// An entity whose numbers do not describe a shape, so a primitive
    /// constructor refused it. [`CadError::diagnostic_code`] is the mapping.
    ///
    /// [`CadError::diagnostic_code`]: crate::cad::CadError::diagnostic_code
    pub const PRIMITIVE_REFUSED: Self = Self(502);
    /// The decode stopped before the drawing did.
    ///
    /// [`DecodeReport::mark_truncated`] is how it is filed, and it is the
    /// difference between a short drawing and a lost one.
    pub const DECODE_TRUNCATED: Self = Self(503);

    /// The lowest code libviprs itself allocates.
    pub const CORE_RANGE_START: u32 = 500;
    /// The lowest code the backing provider allocates.
    pub const PROVIDER_RANGE_START: u32 = 1000;

    /// Wraps a raw code, including one this build has no name for.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// The raw code.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether libviprs allocated this code rather than the decode contract or
    /// the provider.
    #[must_use]
    pub const fn is_core_allocated(self) -> bool {
        self.0 >= Self::CORE_RANGE_START && self.0 < Self::PROVIDER_RANGE_START
    }

    /// Whether the backing provider allocated this code.
    #[must_use]
    pub const fn is_provider_allocated(self) -> bool {
        self.0 >= Self::PROVIDER_RANGE_START
    }

    /// A name for this code, or a description of the band it falls in.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::UNSUPPORTED_ENTITY => "unsupported entity",
            Self::READER_NOTIFICATION => "reader notification",
            Self::DIMENSION_WITHOUT_BLOCK => "dimension without block",
            Self::HATCH_PATTERN_ONLY => "hatch pattern only",
            Self::HATCH_LOOP_NOT_POLYGON => "hatch loop not a polygon",
            Self::UNRESOLVED_BLOCK => "unresolved block",
            Self::NON_UNIFORM_BLOCK_SCALE => "non-uniform block scale",
            Self::NON_FINITE_GEOMETRY => "non-finite geometry",
            Self::EMPTY_VIEW => "empty view",
            Self::TEXT_ESCAPE_NOT_DECODED => "text escape not decoded",
            Self::TEXT_NOT_RECOVERED => "text not recovered",
            Self::PRIMITIVE_REFUSED => "primitive refused",
            Self::DECODE_TRUNCATED => "decode truncated",
            other if other.is_provider_allocated() => "a code the provider allocated",
            _ => "a code this build has no name for",
        }
    }
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.0)
    }
}

/// One thing a decode had to say about part of a drawing.
///
/// The code is the structured part and the only thing anything branches on.
/// The message is for a person reading a log; it names the entity kind, the
/// block or the escape that was involved, and nothing reads it
/// programmatically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Diagnostic {
    code: DiagnosticCode,
    severity: Severity,
    message: String,
    entity: Option<ItemHandle>,
    view: Option<u32>,
}

impl Diagnostic {
    /// A diagnostic about something not fully represented.
    #[must_use]
    pub fn warning(code: DiagnosticCode, message: impl Into<String>) -> Self {
        Self::new(code, Severity::Warning, message)
    }

    /// A diagnostic about something that could not be represented at all.
    #[must_use]
    pub fn error(code: DiagnosticCode, message: impl Into<String>) -> Self {
        Self::new(code, Severity::Error, message)
    }

    fn new(code: DiagnosticCode, severity: Severity, message: impl Into<String>) -> Self {
        Self {
            code,
            severity,
            message: message.into(),
            entity: None,
            view: None,
        }
    }

    /// The same diagnostic, attributed to one entity.
    ///
    /// Worth attaching wherever it is known: "an unsupported entity" is a
    /// count, and "entity 79D is an unsupported entity" is something a person
    /// can go and look at in their CAD application.
    #[must_use]
    pub fn with_entity(mut self, entity: ItemHandle) -> Self {
        self.entity = Some(entity);
        self
    }

    /// The same diagnostic, attributed to one view.
    #[must_use]
    pub const fn with_view(mut self, view: u32) -> Self {
        self.view = Some(view);
        self
    }

    /// What kind of diagnostic this is.
    #[must_use]
    pub const fn code(&self) -> DiagnosticCode {
        self.code
    }

    /// How much it matters.
    #[must_use]
    pub const fn severity(&self) -> Severity {
        self.severity
    }

    /// The human-readable detail. Nothing branches on it.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The entity this is about, when it is about one.
    #[must_use]
    pub const fn entity(&self) -> Option<ItemHandle> {
        self.entity
    }

    /// The view this is about, when it is about one.
    #[must_use]
    pub const fn view(&self) -> Option<u32> {
        self.view
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}: {}", self.severity, self.code, self.message)?;
        if let Some(entity) = self.entity {
            write!(f, ", on entity {entity}")?;
        }
        if let Some(view) = self.view {
            write!(f, ", in view {view}")?;
        }
        Ok(())
    }
}

/// How many of each shape a decode produced.
///
/// One number per [`PrimitiveKind`], which is what lets a release gate say
/// "this drawing lost 40% of its text" rather than "the tile looks a bit
/// empty". The counts are of primitives *emitted*, so a block referenced a
/// thousand times contributes a thousand times and the amplification is
/// visible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrimitiveCounts {
    counts: [u64; PrimitiveKind::ALL.len()],
}

impl PrimitiveCounts {
    /// All zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; PrimitiveKind::ALL.len()],
        }
    }

    /// How many of one kind.
    #[must_use]
    pub const fn get(&self, kind: PrimitiveKind) -> u64 {
        self.counts[kind.index()]
    }

    /// How many of every kind together.
    #[must_use]
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// Whether nothing at all was emitted.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }

    /// Every kind and its count, in [`PrimitiveKind::ALL`] order.
    pub fn iter(&self) -> impl Iterator<Item = (PrimitiveKind, u64)> + '_ {
        PrimitiveKind::ALL
            .into_iter()
            .map(|kind| (kind, self.get(kind)))
    }

    fn record(&mut self, kind: PrimitiveKind) {
        self.counts[kind.index()] = self.counts[kind.index()].saturating_add(1);
    }
}

/// What a decode has to say about itself.
///
/// Returned by [`CadDrawing::decode`](crate::cad::CadDrawing::decode), and the
/// only channel a fidelity question is answered through.
///
/// # Examples
///
/// ```
/// use libviprs::cad::{DecodeReport, Diagnostic, DiagnosticCode, ItemHandle, PrimitiveKind};
///
/// let mut report = DecodeReport::new().with_diagnostic_limit(2);
///
/// report.record(PrimitiveKind::Line);
/// report.record(PrimitiveKind::Line);
/// report.record(PrimitiveKind::Text);
///
/// for handle in 0..5 {
///     report.push(
///         Diagnostic::warning(DiagnosticCode::UNSUPPORTED_ENTITY, "ACAD_TABLE")
///             .with_entity(ItemHandle::new(handle)),
///     );
/// }
///
/// assert_eq!(report.counts().get(PrimitiveKind::Line), 2);
/// assert_eq!(report.counts().total(), 3);
///
/// // Bounded, and the ones it stopped writing down are counted rather than lost.
/// assert_eq!(report.diagnostics().len(), 2);
/// assert_eq!(report.dropped_diagnostics(), 3);
///
/// // Nothing said the stream ended for a good reason, so it has not.
/// assert!(!report.is_complete());
/// report.mark_complete();
/// assert!(report.is_complete());
/// assert!(!report.has_errors());
///
/// // A truncation is sticky and takes completeness back, and it is retained
/// // even though the bounded set was already full.
/// report.mark_truncated("the view ended 900 records short of its total");
/// assert!(!report.is_complete());
/// assert!(report.is_truncated());
/// assert_eq!(
///     report.errors().map(Diagnostic::code).collect::<Vec<_>>(),
///     vec![DiagnosticCode::DECODE_TRUNCATED]
/// );
///
/// // And no later call can talk it back into looking finished.
/// report.mark_complete();
/// assert!(!report.is_complete());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeReport {
    diagnostics: Vec<Diagnostic>,
    dropped: u64,
    limit: usize,
    counts: PrimitiveCounts,
    complete: bool,
    truncated: bool,
}

impl Default for DecodeReport {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodeReport {
    /// How many diagnostics a report retains unless told otherwise.
    ///
    /// Enough that a drawing with a real problem shows the shape of it, and
    /// small enough that ten million of them cost nothing. A caller who wants
    /// every single one raises it deliberately and accepts the memory.
    pub const DEFAULT_DIAGNOSTIC_LIMIT: usize = 1024;

    /// An empty report of an incomplete decode.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            dropped: 0,
            limit: Self::DEFAULT_DIAGNOSTIC_LIMIT,
            counts: PrimitiveCounts::new(),
            complete: false,
            truncated: false,
        }
    }

    /// The same report, retaining at most `limit` diagnostics.
    ///
    /// A limit of zero retains none and counts all of them, which is the shape
    /// a batch job wants when it only needs to know whether anything happened.
    ///
    /// The limit bounds the *retained set* and nothing else:
    /// [`DecodeReport::is_complete`] and [`DecodeReport::is_truncated`] are
    /// flags, and no limit can erase them.
    #[must_use]
    pub fn with_diagnostic_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self.diagnostics.truncate(limit);
        self
    }

    /// Files a diagnostic, or counts it as dropped once the retained set is
    /// full.
    pub fn push(&mut self, diagnostic: Diagnostic) {
        if self.diagnostics.len() < self.limit {
            self.diagnostics.push(diagnostic);
        } else {
            self.dropped = self.dropped.saturating_add(1);
        }
    }

    /// Counts one emitted primitive.
    pub fn record(&mut self, kind: PrimitiveKind) {
        self.counts.record(kind);
    }

    /// Records that the decode reached the end of the drawing and the totals
    /// agreed.
    ///
    /// Call this only after checking whatever the provider gives for that — a
    /// record count, a document-end total. It is the difference between a
    /// caller trusting a tile and a caller knowing it can.
    ///
    /// It cannot undo [`DecodeReport::mark_truncated`]: a decode that was
    /// found short stays short, whatever is called afterwards.
    pub const fn mark_complete(&mut self) {
        self.complete = true;
    }

    /// Records that the decode stopped before the drawing did, as a
    /// [`DiagnosticCode::DECODE_TRUNCATED`] error, and takes completeness
    /// away for good.
    ///
    /// The diagnostic is retained **past the limit**, once, and that is the
    /// one exception the bound has. A drawing that emits ten thousand
    /// warnings before the stream is cut would otherwise fill the retained set
    /// and drop the only entry saying the result cannot be trusted — a
    /// hostile file choosing which diagnostic a caller gets to see is the
    /// opposite of what the bound is for. It costs one entry, because the
    /// flag makes the push happen at most once.
    pub fn mark_truncated(&mut self, message: impl Into<String>) {
        let first = !self.truncated;
        self.truncated = true;
        if first {
            self.diagnostics
                .push(Diagnostic::error(DiagnosticCode::DECODE_TRUNCATED, message));
        }
    }

    /// The retained diagnostics, in the order they were filed.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// How many diagnostics were filed past the retained set's limit.
    #[must_use]
    pub const fn dropped_diagnostics(&self) -> u64 {
        self.dropped
    }

    /// How many diagnostics this report retains.
    #[must_use]
    pub const fn diagnostic_limit(&self) -> usize {
        self.limit
    }

    /// The retained diagnostics of [`Severity::Error`].
    pub fn errors(&self) -> impl Iterator<Item = &Diagnostic> + '_ {
        self.diagnostics
            .iter()
            .filter(|d| d.severity() == Severity::Error)
    }

    /// Whether any retained diagnostic is an error.
    ///
    /// Retained only: a report whose limit dropped its errors answers `false`,
    /// which is why [`DecodeReport::dropped_diagnostics`] has to be read
    /// beside it.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.errors().next().is_some()
    }

    /// How many of each shape the decode produced.
    #[must_use]
    pub const fn counts(&self) -> &PrimitiveCounts {
        &self.counts
    }

    /// Whether the decode reached the end of the drawing.
    ///
    /// `false` until [`DecodeReport::mark_complete`] says otherwise, so a
    /// decoder that never checks its totals cannot claim it did, and `false`
    /// again for good once [`DecodeReport::mark_truncated`] has been called.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.complete && !self.truncated
    }

    /// Whether the decode was found to have stopped short of the drawing.
    ///
    /// The structural answer, independent of the retained set: a caller that
    /// only wants to know whether to trust the primitives reads this rather
    /// than searching the diagnostics for a code.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A report starts incomplete, so forgetting to check the totals reports
    /// a truncated decode rather than passing one off as a short drawing.
    ///
    /// The whole reason the flag defaults the unhelpful way round.
    #[test]
    fn a_fresh_report_does_not_claim_the_decode_finished() {
        let report = DecodeReport::new();
        assert!(
            !report.is_complete(),
            "a decoder that never called mark_complete must not look like one \
             that did"
        );
        assert!(report.counts().is_empty());
        assert_eq!(report.dropped_diagnostics(), 0);
    }

    /// Truncation is structured, takes completeness back, and stays taken.
    ///
    /// The pair matters: a decoder that marked complete and *then* found a
    /// short count would otherwise leave a report saying both, and one that
    /// marked complete *after* finding it would talk the report back into
    /// looking finished.
    #[test]
    fn marking_truncated_files_an_error_and_revokes_completeness_for_good() {
        let mut report = DecodeReport::new();
        report.mark_complete();
        report.mark_truncated("view 0 ended 12 records short");

        assert!(!report.is_complete());
        assert!(report.is_truncated());
        assert!(report.has_errors());
        assert_eq!(
            report.diagnostics()[0].code(),
            DiagnosticCode::DECODE_TRUNCATED
        );
        assert_eq!(report.diagnostics()[0].severity(), Severity::Error);

        report.mark_complete();
        assert!(
            !report.is_complete(),
            "a later mark_complete must not undo a truncation, or the one flag \
             a caller trusts depends on call order"
        );
    }

    /// A full retained set cannot hide the truncation, and repeating the call
    /// does not grow the report.
    ///
    /// This is the hole the doctest found: a hostile drawing that emits a
    /// warning per entity fills the bounded set, and the one diagnostic
    /// saying the result cannot be trusted was the one being dropped.
    #[test]
    fn a_truncation_is_retained_past_a_full_bounded_set_exactly_once() {
        let mut report = DecodeReport::new().with_diagnostic_limit(2);
        for i in 0..50 {
            report.push(Diagnostic::warning(
                DiagnosticCode::UNSUPPORTED_ENTITY,
                format!("entity {i}"),
            ));
        }
        assert_eq!(report.diagnostics().len(), 2, "the bound holds for pushes");

        report.mark_truncated("cut at record 4001");
        report.mark_truncated("cut at record 4001");
        report.mark_truncated("cut at record 4001");

        assert_eq!(
            report
                .errors()
                .map(Diagnostic::code)
                .collect::<Vec<DiagnosticCode>>(),
            vec![DiagnosticCode::DECODE_TRUNCATED],
            "exactly one truncation entry survives: a full set must not hide \
             it, and three calls must not grow the report by three"
        );
        assert_eq!(report.diagnostics().len(), 3);
    }

    /// The retained set is bounded and the overflow is counted, not lost.
    #[test]
    fn diagnostics_are_bounded_and_the_overflow_is_counted() {
        let mut report = DecodeReport::new().with_diagnostic_limit(3);
        for i in 0..10 {
            report.push(Diagnostic::warning(
                DiagnosticCode::UNSUPPORTED_ENTITY,
                format!("entity {i}"),
            ));
        }

        assert_eq!(report.diagnostics().len(), 3);
        assert_eq!(report.dropped_diagnostics(), 7);
        assert_eq!(
            report.diagnostics()[0].message(),
            "entity 0",
            "the retained set keeps the first diagnostics, because the first \
             thing that went wrong is usually the cause of the rest"
        );
    }

    /// A limit of zero retains nothing and still counts.
    ///
    /// The boundary case, and the one a batch job uses. A `< limit` written as
    /// `<= limit` would retain one here.
    #[test]
    fn a_limit_of_zero_retains_nothing_and_counts_everything() {
        let mut report = DecodeReport::new().with_diagnostic_limit(0);
        report.push(Diagnostic::error(
            DiagnosticCode::TEXT_NOT_RECOVERED,
            "gone",
        ));

        assert!(report.diagnostics().is_empty());
        assert_eq!(report.dropped_diagnostics(), 1);
        assert!(
            !report.has_errors(),
            "has_errors reads the retained set, which is why the dropped count \
             has to be read with it"
        );
    }

    /// Lowering the limit on a report that already holds more drops the excess
    /// rather than leaving the retained set over its own bound.
    #[test]
    fn lowering_the_limit_trims_what_is_already_retained() {
        let mut report = DecodeReport::new();
        for i in 0..5 {
            report.push(Diagnostic::warning(
                DiagnosticCode::READER_NOTIFICATION,
                format!("note {i}"),
            ));
        }
        let report = report.with_diagnostic_limit(2);

        assert_eq!(report.diagnostics().len(), 2);
        assert_eq!(report.diagnostic_limit(), 2);
    }

    /// Counts are per kind, which is what makes a fidelity loss measurable.
    #[test]
    fn counts_are_per_kind_and_do_not_collapse() {
        let mut report = DecodeReport::new();
        for _ in 0..7 {
            report.record(PrimitiveKind::Line);
        }
        report.record(PrimitiveKind::Text);
        report.record(PrimitiveKind::Arc);

        assert_eq!(report.counts().get(PrimitiveKind::Line), 7);
        assert_eq!(report.counts().get(PrimitiveKind::Text), 1);
        assert_eq!(report.counts().get(PrimitiveKind::Arc), 1);
        assert_eq!(report.counts().get(PrimitiveKind::Spline), 0);
        assert_eq!(report.counts().total(), 9);

        let listed: u64 = report.counts().iter().map(|(_, n)| n).sum();
        assert_eq!(
            listed,
            report.counts().total(),
            "`iter` has to visit every slot `total` sums, or a kind is \
             invisible in a report that still adds up"
        );
    }

    /// The three code bands do not overlap, and the text contract's codes are
    /// in libviprs' own band rather than borrowing the contract's numbers.
    ///
    /// The collision this guards against is silent: two allocators reaching
    /// for the next free number, and a report that then cannot tell an
    /// undecoded escape from an unresolved block.
    #[test]
    fn the_code_bands_do_not_overlap() {
        for code in [
            DiagnosticCode::UNSUPPORTED_ENTITY,
            DiagnosticCode::EMPTY_VIEW,
        ] {
            assert!(!code.is_core_allocated(), "{code} is a contract code");
            assert!(!code.is_provider_allocated());
        }

        for code in [
            DiagnosticCode::TEXT_ESCAPE_NOT_DECODED,
            DiagnosticCode::TEXT_NOT_RECOVERED,
            DiagnosticCode::PRIMITIVE_REFUSED,
            DiagnosticCode::DECODE_TRUNCATED,
        ] {
            assert!(code.is_core_allocated(), "{code} is a libviprs code");
            assert!(!code.is_provider_allocated());
        }

        let backend = DiagnosticCode::new(1100);
        assert!(backend.is_provider_allocated());
        assert!(!backend.is_core_allocated());
        assert_eq!(backend.name(), "a code the provider allocated");
    }

    /// An unknown code survives and says so, rather than being refused.
    #[test]
    fn a_code_this_build_has_no_name_for_is_still_carried() {
        let code = DiagnosticCode::new(142);
        assert_eq!(code.get(), 142);
        assert_eq!(code.name(), "a code this build has no name for");
        assert_eq!(code.to_string(), "a code this build has no name for (142)");
    }

    /// A diagnostic renders with everything it was given.
    #[test]
    fn a_diagnostic_displays_its_code_entity_and_view() {
        let diagnostic = Diagnostic::error(DiagnosticCode::TEXT_NOT_RECOVERED, "embedded MTEXT")
            .with_entity(ItemHandle::new(0x79D))
            .with_view(0);

        assert_eq!(
            diagnostic.to_string(),
            "error: text not recovered (501): embedded MTEXT, on entity 1949, in view 0"
        );
        assert_eq!(diagnostic.entity(), Some(ItemHandle::new(0x79D)));
        assert_eq!(diagnostic.view(), Some(0));
    }
}
