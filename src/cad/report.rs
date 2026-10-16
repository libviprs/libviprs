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
/// * Below [`DiagnosticCode::CORE_RANGE_START`] is the contract band: the
///   numbering of the VIPRS specification, which the provider shares by
///   agreement. `acadsharp-rs`'s `WarningCode` carries the same 100 to 108, so
///   a provider's warning in that range passes through as the same code
///   without a translation table, and that is deliberate.
/// * From [`DiagnosticCode::CORE_RANGE_START`] to
///   [`DiagnosticCode::PROVIDER_RANGE_START`] are the codes libviprs itself
///   allocates, which is where the text contract's two live. A separate band
///   so the two allocators cannot collide by both reaching for the next free
///   number.
/// * From [`DiagnosticCode::PROVIDER_RANGE_START`] up, whatever read the
///   drawing allocates, and a consumer is entitled to know none of them.
///
/// # Which band a new code goes in
///
/// Ask who can observe the condition. If a provider can see it in the file's
/// data on its own, and libviprs can see the same thing when it gets there
/// first, then both layers have to file it under one number or one drawing
/// reports the same fact twice. That is a contract code, and it is allocated
/// by amending the specification, not by this crate alone. If only libviprs
/// defines the condition, because it is about this crate's own IR rules or its
/// own text contract, it is a core code.
///
/// [`DiagnosticCode::NON_FINITE_GEOMETRY`] is the instructive case, and it is
/// 107 on purpose: the provider catches a non-finite record inline and files
/// 107, and a primitive constructor catches the same value as
/// [`CadError::NonFinite`](crate::cad::CadError::NonFinite) when the provider
/// did not, so the two have to agree. Its siblings at 500 to 502 are the other
/// kind: [`DiagnosticCode::PRIMITIVE_REFUSED`] is this crate's own validation
/// rules, and the text pair and the truncation are rules only libviprs states.
///
/// A provider's own codes are not in either of the first two bands.
/// [`DiagnosticCode::provider`] allocates one in the third by construction,
/// and [`DiagnosticCode::new`] is for a code read back off a wire.
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
    ///
    /// A contract code and not a core one, deliberately: the provider and a
    /// primitive constructor both catch this, and have to file the same number.
    /// See the rule on [`DiagnosticCode`].
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
    /// The decode could not be run, or could not continue, for a reason that
    /// is not any entity's numbers: the source was not a drawing this decoder
    /// reads, the view asked for does not exist, the read failed, or the
    /// provider or the sink did.
    ///
    /// Kept apart from [`DiagnosticCode::PRIMITIVE_REFUSED`] so a release gate
    /// counting malformed entities is not counting a disk error.
    /// [`CadError::diagnostic_code`] is the mapping.
    ///
    /// [`CadError::diagnostic_code`]: crate::cad::CadError::diagnostic_code
    pub const DECODE_FAILED: Self = Self(504);

    /// The lowest code libviprs itself allocates.
    pub const CORE_RANGE_START: u32 = 500;
    /// The lowest code the backing provider allocates.
    pub const PROVIDER_RANGE_START: u32 = 1000;

    /// Wraps a raw code, including one this build has no name for.
    ///
    /// This is the wire round-trip constructor: it places a code exactly where
    /// it is told, so `new(500)` *is* [`DiagnosticCode::TEXT_ESCAPE_NOT_DECODED`].
    /// That is what reading a code back off a report needs, and it is the wrong
    /// tool for a provider allocating its own, which should use
    /// [`DiagnosticCode::provider`] so it cannot land on someone else's number.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// A code the backing provider allocates, offset into the provider band so
    /// it cannot collide with a contract or libviprs code however the provider
    /// numbers its own.
    ///
    /// `provider(0)` is [`DiagnosticCode::PROVIDER_RANGE_START`]. An offset
    /// that would run past the top of the `u32` line saturates there instead of
    /// wrapping back into the contract band, so the result is always
    /// [`DiagnosticCode::is_provider_allocated`].
    ///
    /// ```
    /// use libviprs::cad::DiagnosticCode;
    ///
    /// let mine = DiagnosticCode::provider(7);
    /// assert!(mine.is_provider_allocated());
    /// assert_ne!(mine, DiagnosticCode::new(7));
    /// ```
    #[must_use]
    pub const fn provider(offset: u32) -> Self {
        Self(Self::PROVIDER_RANGE_START.saturating_add(offset))
    }

    /// The raw code.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// Whether the decode contract allocated this code, which is to say it is
    /// one of the specification's own numbers and not libviprs's or the
    /// provider's.
    #[must_use]
    pub const fn is_contract_allocated(self) -> bool {
        self.0 < Self::CORE_RANGE_START
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
            Self::DECODE_FAILED => "decode failed",
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
    counts: [u64; PrimitiveKind::COUNT],
}

impl PrimitiveCounts {
    /// All zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; PrimitiveKind::COUNT],
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
            .iter()
            .map(|&kind| (kind, self.get(kind)))
    }

    fn record(&mut self, kind: PrimitiveKind) {
        self.counts[kind.index()] = self.counts[kind.index()].saturating_add(1);
    }
}

/// What a decode has to say about itself.
///
/// Filled in by [`CadDrawing::decode`](crate::cad::CadDrawing::decode), which
/// takes it as an out-parameter, and the only channel a fidelity question is
/// answered through.
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
    /// Lowering the limit under what a report already retains discards the
    /// excess, and counts it: the discarded entries are added to
    /// [`DecodeReport::dropped_diagnostics`], because a report that stopped
    /// writing diagnostics down has to say so however the bound was reached.
    /// Raising it discards nothing and counts nothing.
    ///
    /// The limit bounds the *retained set* and nothing else:
    /// [`DecodeReport::is_complete`] and [`DecodeReport::is_truncated`] are
    /// flags, and no limit can erase them.
    #[must_use]
    pub fn with_diagnostic_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        // Trimming is dropping: a report that stopped writing diagnostics down
        // has to say so, whether the bound was reached by pushing or by being
        // lowered underneath what was already retained.
        let trimmed = self.diagnostics.len().saturating_sub(limit);
        self.dropped = self.dropped.saturating_add(trimmed as u64);
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

    /// Lowering the limit drops the excess rather than leaving the retained
    /// set over its own bound, and counts every entry it drops.
    ///
    /// The pair is the point. A report that trims silently answers
    /// `dropped_diagnostics() == 0` while holding less than was filed, which
    /// is the reading the module documentation tells a caller to interpret as
    /// "nothing else went wrong" — and with an all-error report it pairs with
    /// `has_errors() == false`, so every error a decode filed disappears
    /// without trace. Checked at the limit, one under it, and far under it.
    #[test]
    fn lowering_the_limit_counts_every_diagnostic_it_trims() {
        let filed = |severity_is_error: bool| {
            let mut report = DecodeReport::new();
            for i in 0..5 {
                report.push(if severity_is_error {
                    Diagnostic::error(DiagnosticCode::TEXT_NOT_RECOVERED, format!("lost {i}"))
                } else {
                    Diagnostic::warning(DiagnosticCode::READER_NOTIFICATION, format!("note {i}"))
                });
            }
            assert_eq!(report.dropped_diagnostics(), 0, "the default bound is 1024");
            report
        };

        // At the limit: nothing is over the bound, so nothing is dropped.
        let at_cap = filed(false).with_diagnostic_limit(5);
        assert_eq!(at_cap.diagnostics().len(), 5);
        assert_eq!(at_cap.dropped_diagnostics(), 0);

        // One over: exactly one entry goes, and exactly one is counted.
        let one_over = filed(false).with_diagnostic_limit(4);
        assert_eq!(one_over.diagnostics().len(), 4);
        assert_eq!(one_over.dropped_diagnostics(), 1);

        // Far over, and all five are errors: the retained set empties, so
        // `dropped_diagnostics` is the only thing left that knows they existed.
        let far_over = filed(true).with_diagnostic_limit(0);
        assert!(far_over.diagnostics().is_empty());
        assert_eq!(far_over.dropped_diagnostics(), 5);
        assert!(
            !far_over.has_errors(),
            "has_errors reads the retained set, so a trim that did not count \
             would leave five errors reading as a clean decode"
        );
        assert_eq!(far_over.diagnostic_limit(), 0);

        // Raising the limit discards nothing, so it counts nothing.
        let raised = filed(false).with_diagnostic_limit(4096);
        assert_eq!(raised.diagnostics().len(), 5);
        assert_eq!(raised.dropped_diagnostics(), 0);
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
            DiagnosticCode::DECODE_FAILED,
        ] {
            assert!(code.is_core_allocated(), "{code} is a libviprs code");
            assert!(!code.is_provider_allocated());
        }

        let backend = DiagnosticCode::new(1100);
        assert!(backend.is_provider_allocated());
        assert!(!backend.is_core_allocated());
        assert_eq!(backend.name(), "a code the provider allocated");
    }

    /// The three predicates partition the whole `u32` line: every value
    /// answers exactly one of contract, core and provider, at the edges as well
    /// as in the middle.
    #[test]
    fn every_raw_code_is_in_exactly_one_band() {
        for raw in [
            0,
            99,
            100,
            107,
            108,
            499,
            500,
            503,
            504,
            999,
            1000,
            u32::MAX,
        ] {
            let code = DiagnosticCode::new(raw);
            let answers = [
                code.is_contract_allocated(),
                code.is_core_allocated(),
                code.is_provider_allocated(),
            ];
            assert_eq!(
                answers.iter().filter(|a| **a).count(),
                1,
                "{raw} answered {answers:?} for contract, core, provider"
            );
        }
    }

    /// Every named code sits in the band the rule in [`DiagnosticCode`]'s
    /// documentation puts it in: the VIPRS specification's own numbers, shared
    /// with the provider, are 100..=108; what only libviprs defines is 500 and up.
    #[test]
    fn every_named_code_is_in_the_band_its_owner_allocates_from() {
        for code in [
            DiagnosticCode::UNSUPPORTED_ENTITY,
            DiagnosticCode::READER_NOTIFICATION,
            DiagnosticCode::DIMENSION_WITHOUT_BLOCK,
            DiagnosticCode::HATCH_PATTERN_ONLY,
            DiagnosticCode::HATCH_LOOP_NOT_POLYGON,
            DiagnosticCode::UNRESOLVED_BLOCK,
            DiagnosticCode::NON_UNIFORM_BLOCK_SCALE,
            DiagnosticCode::NON_FINITE_GEOMETRY,
            DiagnosticCode::EMPTY_VIEW,
        ] {
            assert!(
                code.is_contract_allocated(),
                "{code} is a specification code"
            );
            assert!((100..=108).contains(&code.get()));
        }
        for code in [
            DiagnosticCode::TEXT_ESCAPE_NOT_DECODED,
            DiagnosticCode::TEXT_NOT_RECOVERED,
            DiagnosticCode::PRIMITIVE_REFUSED,
            DiagnosticCode::DECODE_TRUNCATED,
        ] {
            assert!(code.is_core_allocated(), "{code} is a libviprs code");
        }
    }

    /// A provider's own code lands in the provider band by construction,
    /// however the provider numbers its codes, so it cannot alias a contract
    /// or a libviprs code and `is_provider_allocated` answers for it.
    #[test]
    fn a_provider_code_is_in_the_provider_band_by_construction() {
        let first = DiagnosticCode::provider(0);
        assert!(first.is_provider_allocated());
        assert_eq!(first.get(), DiagnosticCode::PROVIDER_RANGE_START);

        let named = [
            DiagnosticCode::UNSUPPORTED_ENTITY,
            DiagnosticCode::NON_FINITE_GEOMETRY,
            DiagnosticCode::EMPTY_VIEW,
            DiagnosticCode::TEXT_ESCAPE_NOT_DECODED,
            DiagnosticCode::TEXT_NOT_RECOVERED,
            DiagnosticCode::PRIMITIVE_REFUSED,
            DiagnosticCode::DECODE_TRUNCATED,
        ];
        // A provider numbering its own codes from 0, from 1, from 100 and from
        // 500 is the aliasing the constructor exists to prevent.
        for offset in [0, 1, 7, 100, 107, 500, 502, 503, u32::MAX - 1, u32::MAX] {
            let code = DiagnosticCode::provider(offset);
            assert!(code.is_provider_allocated(), "provider({offset}) = {code}");
            assert!(!code.is_core_allocated() && !code.is_contract_allocated());
            assert!(!named.contains(&code), "provider({offset}) aliased {code}");
        }
    }

    /// The edge: past the end of the line the offset saturates and stays in the
    /// provider band, rather than wrapping round into the contract band.
    #[test]
    fn a_provider_offset_past_the_end_saturates_instead_of_wrapping() {
        assert_eq!(DiagnosticCode::provider(u32::MAX).get(), u32::MAX);
        assert!(DiagnosticCode::provider(u32::MAX).is_provider_allocated());
        assert_eq!(
            DiagnosticCode::provider(u32::MAX - DiagnosticCode::PROVIDER_RANGE_START).get(),
            u32::MAX
        );
    }

    /// `new` stays the wire round-trip: it places a code wherever it is told,
    /// including on a contract constant, which is exactly what reading a code
    /// back off a wire needs and what a provider allocating its own must not use.
    #[test]
    fn new_round_trips_a_code_that_came_off_a_wire() {
        for code in [
            DiagnosticCode::NON_FINITE_GEOMETRY,
            DiagnosticCode::TEXT_NOT_RECOVERED,
            DiagnosticCode::provider(3),
        ] {
            assert_eq!(DiagnosticCode::new(code.get()), code);
        }
    }

    /// Every named code has one severity, and it is the one the contract says.
    ///
    /// The control is in the table itself: both answers appear, so a function
    /// that returned one severity for everything cannot satisfy it.
    #[test]
    fn every_named_code_has_a_pinned_default_severity() {
        use Severity::{Error, Warning};
        for (code, expected) in [
            (DiagnosticCode::UNSUPPORTED_ENTITY, Warning),
            (DiagnosticCode::READER_NOTIFICATION, Warning),
            (DiagnosticCode::DIMENSION_WITHOUT_BLOCK, Warning),
            (DiagnosticCode::HATCH_PATTERN_ONLY, Warning),
            (DiagnosticCode::HATCH_LOOP_NOT_POLYGON, Warning),
            (DiagnosticCode::UNRESOLVED_BLOCK, Warning),
            (DiagnosticCode::NON_UNIFORM_BLOCK_SCALE, Warning),
            (DiagnosticCode::NON_FINITE_GEOMETRY, Error),
            (DiagnosticCode::EMPTY_VIEW, Warning),
            (DiagnosticCode::TEXT_ESCAPE_NOT_DECODED, Error),
            (DiagnosticCode::TEXT_NOT_RECOVERED, Error),
            (DiagnosticCode::PRIMITIVE_REFUSED, Error),
            (DiagnosticCode::DECODE_TRUNCATED, Error),
            (DiagnosticCode::DECODE_FAILED, Error),
        ] {
            assert_eq!(code.default_severity(), expected, "{code}");
        }
    }

    /// A code this build has no name for still answers, by band: libviprs's own
    /// unknown codes are errors (a code this crate allocated and this build
    /// cannot read is not one to wave through), and the specification's and the
    /// provider's unknown codes are warnings, the provider's own judgement.
    #[test]
    fn an_unnamed_code_takes_its_default_severity_from_its_band() {
        assert_eq!(
            DiagnosticCode::new(142).default_severity(),
            Severity::Warning
        );
        assert_eq!(DiagnosticCode::new(650).default_severity(), Severity::Error);
        assert_eq!(
            DiagnosticCode::provider(9).default_severity(),
            Severity::Warning
        );
    }

    /// A diagnostic filed from a code takes the code's severity, keeps the
    /// message, and still takes an entity and a view.
    #[test]
    fn a_diagnostic_filed_from_a_code_takes_the_codes_severity() {
        let lost = Diagnostic::from_code(DiagnosticCode::TEXT_NOT_RECOVERED, "embedded MTEXT");
        assert_eq!(lost.severity(), Severity::Error);
        assert_eq!(lost.code(), DiagnosticCode::TEXT_NOT_RECOVERED);
        assert_eq!(lost.message(), "embedded MTEXT");

        let empty = Diagnostic::from_code(DiagnosticCode::EMPTY_VIEW, "Layout1").with_view(2);
        assert_eq!(empty.severity(), Severity::Warning);
        assert_eq!(empty.view(), Some(2));
    }

    /// The same condition filed from the provider's side and from a primitive
    /// constructor's side answers the same way, so `has_errors` does not depend
    /// on which layer noticed the non-finite value first.
    #[test]
    fn the_same_code_filed_from_both_layers_reports_the_same_way() {
        let from_the_provider =
            Diagnostic::from_code(DiagnosticCode::NON_FINITE_GEOMETRY, "record 7C had a NaN");
        let from_a_constructor = crate::cad::CadError::NonFinite {
            field: "centre",
            value: f64::NAN,
        }
        .to_diagnostic();

        assert_eq!(from_the_provider.code(), from_a_constructor.code());
        assert_eq!(from_the_provider.severity(), from_a_constructor.severity());

        let answers: Vec<bool> = [from_the_provider, from_a_constructor]
            .into_iter()
            .map(|d| {
                let mut report = DecodeReport::new();
                report.push(d);
                report.has_errors()
            })
            .collect();
        assert_eq!(answers, [true, true]);

        // The control: a warning code does not turn into an error because it
        // went through the same door.
        let mut report = DecodeReport::new();
        report.push(Diagnostic::from_code(DiagnosticCode::EMPTY_VIEW, "Layout1"));
        assert!(!report.has_errors());
    }

    /// An unknown code survives and says so, rather than being refused.
    #[test]
    fn a_code_this_build_has_no_name_for_is_still_carried() {
        let code = DiagnosticCode::new(142);
        assert_eq!(code.get(), 142);
        assert_eq!(code.name(), "a code this build has no name for");
        assert_eq!(code.to_string(), "a code this build has no name for (142)");
    }

    /// A diagnostic names the entity and the view it was given, and says
    /// nothing about either when it was given neither.
    ///
    /// Asserted as presence and absence rather than as one exact sentence.
    /// The wording is for a person reading a log and is free to change; what
    /// a reader cannot afford to lose is the attribution, because a
    /// diagnostic that stops naming its entity no longer says which part of
    /// the drawing was affected. Pinning the sentence would have made a
    /// comma a test failure and a missing suffix a passing one, and the
    /// `None` branches were never rendered at all.
    #[test]
    fn a_diagnostic_renders_its_entity_and_view_only_when_it_has_them() {
        let entity = ItemHandle::new(0x79D);
        let bare = Diagnostic::error(DiagnosticCode::TEXT_NOT_RECOVERED, "embedded MTEXT");
        let attributed = bare.clone().with_entity(entity).with_view(7);

        let rendered = attributed.to_string();
        assert!(
            rendered.contains(&format!("entity {entity}")),
            "a diagnostic with an entity must name it: {rendered}"
        );
        assert!(
            rendered.contains("view 7"),
            "a diagnostic with a view must name it: {rendered}"
        );
        assert!(
            rendered.contains(&DiagnosticCode::TEXT_NOT_RECOVERED.to_string())
                && rendered.contains("embedded MTEXT"),
            "the code and the message are what the suffixes qualify: {rendered}"
        );

        // Severity renders as its own word, for both levels. A warning is the
        // one nothing else exercised.
        assert!(
            attributed
                .to_string()
                .contains(&Severity::Error.to_string())
        );
        let warned = Diagnostic::warning(DiagnosticCode::EMPTY_VIEW, "Layout1");
        assert!(
            warned.to_string().contains(&Severity::Warning.to_string()),
            "{warned}"
        );
        assert_ne!(
            Severity::Warning.to_string(),
            Severity::Error.to_string(),
            "two levels that render the same cannot be told apart in a log"
        );
        assert!(
            !warned.to_string().contains(&Severity::Error.to_string()),
            "a warning must not read as an error: {warned}"
        );

        let rendered = bare.to_string();
        assert!(
            !rendered.contains("entity") && !rendered.contains(&entity.to_string()),
            "a diagnostic with no entity must not invent one: {rendered}"
        );
        assert!(
            !rendered.contains("view"),
            "a diagnostic with no view must not invent one: {rendered}"
        );
    }
}
