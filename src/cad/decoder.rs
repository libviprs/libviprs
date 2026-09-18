//! The decoder traits, the drawing handle, its views, and the sink the
//! primitives go to.
//!
//! The shape, and it is the provider's own shape with the unsafe parts and the
//! format names taken out:
//!
//! ```text
//! CadDecoder::probe    a header, and nothing else read
//! CadDecoder::open  -> CadDrawing
//!                        views
//!                        decode -> PrimitiveSink  +  DecodeReport
//! ```

use std::path::Path;

use crate::cad::{CadError, DecodeReport, Primitive};

/// Where a drawing comes from.
///
/// A path and a byte slice, which is what a provider can actually take: a DWG
/// reader memory-maps or reads a whole file, so there is no useful
/// `impl Read` flavour to offer. `Path` is kept as its own variant rather than
/// read into bytes here because a provider given a path may avoid the copy
/// entirely, and a 200 MB drawing is worth not copying.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CadSource<'a> {
    /// A file on disk, which the decoder opens itself.
    Path(&'a Path),
    /// Bytes the caller already holds.
    Bytes(&'a [u8]),
}

/// What a view is: model space, or one of the paper-space layouts.
///
/// An enum with an `Unknown` arm rather than a closed pair, because a
/// provider's kind is a number and a value a later build gives a meaning to
/// has to stay representable. [`CadView::raw_kind`] is the number itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ViewKind {
    /// Model space, which is the view this pipeline is built around.
    Model,
    /// One of the paper-space layouts.
    Layout,
    /// The provider did not say, or said something this build has no name for.
    Unknown,
}

impl ViewKind {
    /// Reads a provider's kind number.
    #[must_use]
    pub const fn from_raw(kind: u32) -> Self {
        match kind {
            0 => Self::Model,
            1 => Self::Layout,
            _ => Self::Unknown,
        }
    }
}

/// A bounding box in drawing units.
///
/// Only ever handed out when it is usable: [`Extents::new`] is the whole of
/// the rule and the only copy of it.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Extents {
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Extents {
    /// Reads four extent values as a box, or [`None`] when they are not one.
    ///
    /// Two failures a caller must not have to tell apart by guessing. A
    /// drawing that cannot give its extents writes the inverted box — `1e20`
    /// in both minima against `-1e20` in both maxima, which is what AutoCAD
    /// itself puts in `EXTMIN`/`EXTMAX` for an empty drawing — and `1e20` is
    /// not to be read as a coordinate. And a box carrying a `NaN` is not a
    /// box: `NaN > NaN` is false, so a comparison-only check hands one back
    /// and calls it usable, and one `NaN` poisons every union above it.
    ///
    /// So this asks for the whole rectangle: four finite numbers,
    /// `min_x <= max_x` and `min_y <= max_y`.
    ///
    /// ```
    /// use libviprs::cad::Extents;
    ///
    /// assert!(Extents::new([-1.0, -2.0, 3.0, 4.0]).is_some());
    ///
    /// // The inverted box a drawing with no extents carries.
    /// assert_eq!(Extents::new([1e20, 1e20, -1e20, -1e20]), None);
    ///
    /// // And a NaN, which the comparison alone lets straight through.
    /// assert_eq!(Extents::new([f64::NAN, 0.0, 10.0, 10.0]), None);
    /// ```
    #[must_use]
    pub fn new(extents: [f64; 4]) -> Option<Self> {
        let [min_x, min_y, max_x, max_y] = extents;
        if !extents.iter().all(|v| v.is_finite()) {
            return None;
        }
        if min_x > max_x || min_y > max_y {
            return None;
        }
        Some(Self {
            min_x,
            min_y,
            max_x,
            max_y,
        })
    }

    /// The smaller x.
    #[must_use]
    pub const fn min_x(self) -> f64 {
        self.min_x
    }

    /// The smaller y.
    #[must_use]
    pub const fn min_y(self) -> f64 {
        self.min_y
    }

    /// The larger x.
    #[must_use]
    pub const fn max_x(self) -> f64 {
        self.max_x
    }

    /// The larger y.
    #[must_use]
    pub const fn max_y(self) -> f64 {
        self.max_y
    }

    /// The four values, in the order [`Extents::new`] takes them.
    #[must_use]
    pub const fn into_array(self) -> [f64; 4] {
        [self.min_x, self.min_y, self.max_x, self.max_y]
    }

    /// The box's width in drawing units, never negative.
    #[must_use]
    pub fn width(self) -> f64 {
        self.max_x - self.min_x
    }

    /// The box's height in drawing units, never negative.
    #[must_use]
    pub fn height(self) -> f64 {
        self.max_y - self.min_y
    }
}

/// One view of a drawing: model space, or a paper-space layout.
///
/// This is what a caller picks a tiling target out of, and what `--view
/// "Model"` on a command line resolves against.
#[derive(Debug, Clone, PartialEq)]
pub struct CadView {
    index: u32,
    kind: u32,
    extents: [f64; 4],
    entity_count: u64,
    name: String,
}

impl CadView {
    /// Builds a view out of the fields a provider carries.
    ///
    /// `kind` and `extents` are taken raw — the provider's own number and its
    /// own four values — because that is what the struct stores: a kind a
    /// later build gives a meaning to has to stay representable, and the
    /// extents rule belongs in one place ([`Extents::new`]) rather than at
    /// every construction site.
    ///
    /// ```
    /// use libviprs::cad::{CadView, ViewKind};
    ///
    /// let view = CadView::new(0, 0, [-1.0, -2.0, 3.0, 4.0], 7, "Model");
    /// assert_eq!(view.kind(), ViewKind::Model);
    /// assert_eq!(view.extents().expect("a real box").max_y(), 4.0);
    /// ```
    #[must_use]
    pub fn new(
        index: u32,
        kind: u32,
        extents: [f64; 4],
        entity_count: u64,
        name: impl Into<String>,
    ) -> Self {
        Self {
            index,
            kind,
            extents,
            entity_count,
            name: name.into(),
        }
    }

    /// The view's index, which is what [`CadDrawing::decode`] takes.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// The view's name, as the drawing spells it.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Model space or a layout.
    #[must_use]
    pub const fn kind(&self) -> ViewKind {
        ViewKind::from_raw(self.kind)
    }

    /// The provider's own number for the kind, for a build that grows one this
    /// crate has no name for.
    #[must_use]
    pub const fn raw_kind(&self) -> u32 {
        self.kind
    }

    /// The view's bounding box, when it has a usable one.
    ///
    /// Read this beside the view's diagnostics.
    /// [`DiagnosticCode::EMPTY_VIEW`](crate::cad::DiagnosticCode::EMPTY_VIEW)
    /// alone with no extents is a view with nothing in it; the same pair with
    /// other diagnostics around it is a view something went wrong reading.
    /// [`CadView::raw_extents`] is the four numbers either way.
    #[must_use]
    pub fn extents(&self) -> Option<Extents> {
        Extents::new(self.extents)
    }

    /// The four numbers the provider gave, with no rule applied.
    #[must_use]
    pub const fn raw_extents(&self) -> [f64; 4] {
        self.extents
    }

    /// Entities before any expansion of nested insertions, for progress
    /// reporting only.
    ///
    /// It is an upper bound on nothing and a lower bound on nothing, so it
    /// must never size a buffer or decide that a decode finished:
    /// [`DecodeReport::is_complete`] is what says a decode finished.
    #[must_use]
    pub const fn entity_count(&self) -> u64 {
        self.entity_count
    }
}

/// Something that can read a drawing.
///
/// Two methods, and the split between them is the point.
/// [`CadDecoder::probe`] answers "is this mine" from a header, so it takes a
/// [`CadSource`] and reads almost nothing. Everything else needs the drawing
/// *open*, and lives on [`CadDrawing`]: a `views` that took a source would
/// re-parse a 200 MB DWG to answer a question the previous call already had
/// the answer to.
///
/// # Implementing a decoder
///
/// Both methods are required; neither has a sensible default. A decoder needs
/// no CAD library behind it — this module's own documentation has one built
/// out of a loop, and it runs in this crate's doctests.
///
/// The trait is deliberately not `Send + Sync`. See [`CadDrawing`].
pub trait CadDecoder {
    /// Whether this decoder recognises `source` as a drawing it can read.
    ///
    /// Header-only. A `false` is not a failure: it is how a caller with
    /// several decoders picks one.
    ///
    /// # Errors
    ///
    /// [`CadError::Io`] when the source could not be read at all, and
    /// [`CadError::Provider`] when the decoder's own machinery failed. Bytes
    /// that are simply not a drawing this decoder reads are `Ok(false)`.
    fn probe(&self, source: CadSource<'_>) -> Result<bool, CadError>;

    /// Opens a drawing, once, for however many views are going to be read out
    /// of it.
    ///
    /// # Errors
    ///
    /// [`CadError::UnrecognisedSource`] when the source is not a drawing this
    /// decoder reads, [`CadError::Io`] when it could not be read, and
    /// [`CadError::Provider`] for the decoder's own failures.
    fn open<'a>(&'a self, source: CadSource<'a>) -> Result<Box<dyn CadDrawing + 'a>, CadError>;
}

/// An open drawing: its views, and the primitives in one of them.
///
/// # Not `Send`, not `Sync`, and that is not an oversight
///
/// A provider's decode handle is single-threaded — calls on it must not
/// overlap and nothing behind it is re-entrant — so this trait promises
/// exactly that and no more. A trait that claimed `Send + Sync` here would be
/// a lie the tiler builds on, and the tiler cannot check it.
///
/// [`CadDrawing::decode`] takes `&self` rather than `&mut self`, which is safe
/// *because* the trait is not `Sync`: two threads cannot hold a shared
/// reference to one handle, so there is no overlapping call to rule out, and a
/// caller holding the drawing can still list its views while decoding is not
/// in flight.
///
/// Parallelism in this pipeline lives above the decoder, per tile, over
/// primitives that have already been collected.
pub trait CadDrawing {
    /// Every view in the drawing, in the drawing's own order.
    ///
    /// # Errors
    ///
    /// [`CadError::Provider`] for the provider's own failures.
    fn views(&self) -> Result<Vec<CadView>, CadError>;

    /// Decodes one view, pushing its primitives at `sink`.
    ///
    /// `view` is a [`CadView::index`]. The primitives arrive in the drawing's
    /// own order — see [`PrimitiveSink`] on why that matters.
    ///
    /// [`DecodeReport`] is the only channel a fidelity question is answered
    /// through, and an implementation is expected to
    /// [`record`](DecodeReport::record) every primitive it emits and to call
    /// [`mark_complete`](DecodeReport::mark_complete) only once whatever
    /// totals the provider gives have agreed.
    ///
    /// A malformed entity is a [`Diagnostic`](crate::cad::Diagnostic) in that
    /// report, not an error: a decode that could not represent one entity out
    /// of a hundred thousand succeeded.
    ///
    /// # The report is the caller's, and it is filled in either way
    ///
    /// `report` is an out-parameter rather than a return value because a
    /// decode that fails partway holds three facts at once — the provider's
    /// typed error, the counts and diagnostics accumulated so far, and the
    /// fact that the sink holds a valid prefix — and a `Result<DecodeReport,
    /// CadError>` can carry exactly two of them. An implementation must leave
    /// `report` describing however far it got, on the error path as well as
    /// on the success path. It also lets the caller choose the diagnostic
    /// bound: `DecodeReport::new().with_diagnostic_limit(4096)`.
    ///
    /// A report is per decode. Counts accumulate, so passing one report to
    /// two `decode` calls sums them; hand each decode a fresh report unless
    /// summing is what you want.
    ///
    /// # The implementation finishes the sink
    ///
    /// `decode` must call [`PrimitiveSink::finish`] exactly once before
    /// returning `Ok(())`, and must not call it on the error path — which is
    /// what makes [`CollectSink::is_finished`] the signal its documentation
    /// says it is. This is the same division as
    /// [`TileSink`](crate::sink::TileSink), where the engine and not the
    /// engine's caller finishes the sink it was handed.
    ///
    /// # Errors
    ///
    /// [`CadError::NoSuchView`] for an index the drawing does not have,
    /// [`CadError::Sink`] when the sink refused a primitive, and
    /// [`CadError::Provider`] for the provider's own failures.
    fn decode(
        &self,
        view: u32,
        sink: &mut dyn PrimitiveSink,
        report: &mut DecodeReport,
    ) -> Result<(), CadError>;
}

/// Where a decode's primitives go.
///
/// # Draw order is significant
///
/// Unlike [`TileSink`](crate::sink::TileSink), whose writes are commutative
/// placements, primitives arrive **in order and that order is part of the
/// data**. A drawing's entities paint over each other, and a sink that
/// reordered them would change what the drawing looks like. So this trait
/// takes `&mut self` rather than `&self` + interior mutability: one decode
/// walks one view sequentially, there is nothing to share across threads, and
/// a `Mutex` on every sink would buy a guarantee nobody needs.
///
/// # Implementing a sink
///
/// [`primitive`](PrimitiveSink::primitive) is the *only* required method.
/// [`primitives`](PrimitiveSink::primitives) is the batch path and defaults to
/// forwarding each element to it, so a sink that has nothing to gain from a
/// batch ignores it and a sink that does — one that extends a `Vec`, or writes
/// one MVT feature block — overrides it and gets the whole run in one call.
/// [`finish`](PrimitiveSink::finish) defaults to doing nothing.
///
/// Both paths are real: a streaming decoder calls `primitive` per shape, a
/// batching one calls `primitives` per buffer refill, and neither has to know
/// which kind of sink it has.
pub trait PrimitiveSink {
    /// Takes one primitive.
    ///
    /// # Errors
    ///
    /// [`CadError::Sink`] carrying the sink's own error.
    fn primitive(&mut self, primitive: Primitive) -> Result<(), CadError>;

    /// Takes a run of primitives, in order.
    ///
    /// The default forwards each one to
    /// [`primitive`](PrimitiveSink::primitive) and stops at the first
    /// refusal, so a sink that overrides this must also stop at the first
    /// failure rather than swallowing the rest of the run.
    ///
    /// # Errors
    ///
    /// As [`PrimitiveSink::primitive`].
    fn primitives(&mut self, batch: &mut dyn Iterator<Item = Primitive>) -> Result<(), CadError> {
        for primitive in batch {
            self.primitive(primitive)?;
        }
        Ok(())
    }

    /// Called once after the last primitive of a decode.
    ///
    /// # Errors
    ///
    /// [`CadError::Sink`] carrying the sink's own error.
    fn finish(&mut self) -> Result<(), CadError> {
        Ok(())
    }
}

/// Forwards every method, so a `&mut` to a sink is a sink.
///
/// Written out rather than left to the caller because `&mut dyn PrimitiveSink`
/// is the type [`CadDrawing::decode`] takes, and this is what lets that
/// parameter itself be passed on to another decoder or wrapper.
impl<T: PrimitiveSink + ?Sized> PrimitiveSink for &mut T {
    fn primitive(&mut self, primitive: Primitive) -> Result<(), CadError> {
        (**self).primitive(primitive)
    }

    fn primitives(&mut self, batch: &mut dyn Iterator<Item = Primitive>) -> Result<(), CadError> {
        (**self).primitives(batch)
    }

    fn finish(&mut self) -> Result<(), CadError> {
        (**self).finish()
    }
}

/// The same for a boxed sink, chosen at run time.
impl<T: PrimitiveSink + ?Sized> PrimitiveSink for Box<T> {
    fn primitive(&mut self, primitive: Primitive) -> Result<(), CadError> {
        (**self).primitive(primitive)
    }

    fn primitives(&mut self, batch: &mut dyn Iterator<Item = Primitive>) -> Result<(), CadError> {
        (**self).primitives(batch)
    }

    fn finish(&mut self) -> Result<(), CadError> {
        (**self).finish()
    }
}

/// A sink that keeps every primitive it is given, in order.
///
/// The batch end of the duality: it overrides
/// [`primitives`](PrimitiveSink::primitives) to extend its buffer in one call
/// rather than pushing one at a time.
///
/// It is public because everything above the decoder needs it — the tiler
/// works on a collected view, and a test for anything downstream of
/// [`CadDecoder`] needs somewhere for the primitives to land. It retains
/// everything it is handed and is therefore not what a hostile drawing should
/// be pointed at.
///
/// # Examples
///
/// ```
/// use libviprs::cad::{CollectSink, Line, Primitive, PrimitiveKind, PrimitiveSink};
///
/// let mut sink = CollectSink::new();
///
/// // The streaming path, one at a time.
/// sink.primitive(Line::new([0.0; 3], [1.0, 0.0, 0.0])?.into())?;
///
/// // The batch path, which this sink takes in one call.
/// let batch = vec![
///     Primitive::from(Line::new([0.0; 3], [0.0, 1.0, 0.0])?),
///     Primitive::from(Line::new([0.0; 3], [0.0, 0.0, 1.0])?),
/// ];
/// sink.primitives(&mut batch.into_iter())?;
/// sink.finish()?;
///
/// assert_eq!(sink.collected().len(), 3);
/// assert_eq!(sink.collected()[0].kind(), PrimitiveKind::Line);
/// assert!(sink.is_finished());
/// # Ok::<(), libviprs::cad::CadError>(())
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CollectSink {
    collected: Vec<Primitive>,
    finished: bool,
}

impl CollectSink {
    /// An empty sink.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            collected: Vec::new(),
            finished: false,
        }
    }

    /// Everything it has been given, in the order it arrived.
    #[must_use]
    pub fn collected(&self) -> &[Primitive] {
        &self.collected
    }

    /// Everything it has been given, taking ownership.
    #[must_use]
    pub fn into_collected(self) -> Vec<Primitive> {
        self.collected
    }

    /// How many primitives it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.collected.len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.collected.is_empty()
    }

    /// Whether [`PrimitiveSink::finish`] has been called.
    ///
    /// Worth asking: a decode that returned without finishing its sink is a
    /// decode that gave up somewhere, and the primitives it did hand over are
    /// a prefix rather than a drawing.
    #[must_use]
    pub const fn is_finished(&self) -> bool {
        self.finished
    }
}

impl PrimitiveSink for CollectSink {
    fn primitive(&mut self, primitive: Primitive) -> Result<(), CadError> {
        self.collected.push(primitive);
        Ok(())
    }

    fn primitives(&mut self, batch: &mut dyn Iterator<Item = Primitive>) -> Result<(), CadError> {
        self.collected.extend(batch);
        Ok(())
    }

    fn finish(&mut self) -> Result<(), CadError> {
        self.finished = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cad::{
        Diagnostic, DiagnosticCode, ItemHandle, Line, Origin, PrimitiveKind, Text, undecoded_escape,
    };
    use std::cell::RefCell;
    use std::rc::Rc;

    /// A decoder with no CAD dependency behind it, which is #1029's own
    /// acceptance criterion.
    ///
    /// It also stands in for the two defects the text contract exists for: the
    /// entity at handle `0x64B` carries an undecoded MIF escape and the one at
    /// `0x79D` is text the provider could not recover, and both become
    /// diagnostics rather than primitives.
    struct FixtureDecoder {
        /// Whether to hand back a view that ends short of its own totals.
        truncate: bool,
    }

    struct FixtureDrawing {
        truncate: bool,
    }

    impl CadDecoder for FixtureDecoder {
        fn probe(&self, source: CadSource<'_>) -> Result<bool, CadError> {
            Ok(matches!(source, CadSource::Bytes(b) if b.starts_with(b"AC10")))
        }

        fn open<'a>(&'a self, source: CadSource<'a>) -> Result<Box<dyn CadDrawing + 'a>, CadError> {
            if !self.probe(source)? {
                return Err(CadError::UnrecognisedSource { decoder: "fixture" });
            }
            Ok(Box::new(FixtureDrawing {
                truncate: self.truncate,
            }))
        }
    }

    impl CadDrawing for FixtureDrawing {
        fn views(&self) -> Result<Vec<CadView>, CadError> {
            Ok(vec![
                CadView::new(0, 0, [0.0, 0.0, 10.0, 10.0], 4, "Model"),
                CadView::new(1, 1, [1e20, 1e20, -1e20, -1e20], 0, "Layout1"),
            ])
        }

        fn decode(
            &self,
            view: u32,
            sink: &mut dyn PrimitiveSink,
            report: &mut DecodeReport,
        ) -> Result<(), CadError> {
            if view > 1 {
                return Err(CadError::NoSuchView {
                    index: view,
                    views: 2,
                });
            }

            // The batch path, because this "provider" holds a buffer.
            let batch = vec![
                Primitive::from(
                    Line::new([0.0, 0.0, 0.0], [10.0, 0.0, 0.0])?
                        .with_origin(Origin::from_handle(ItemHandle::new(1))),
                ),
                Primitive::from(
                    Line::new([10.0, 0.0, 0.0], [10.0, 10.0, 0.0])?
                        .with_origin(Origin::from_handle(ItemHandle::new(2))),
                ),
            ];
            let emitted = batch.len();
            sink.primitives(&mut batch.into_iter())?;
            for _ in 0..emitted {
                report.record(PrimitiveKind::Line);
            }

            // The streaming path, for the one shape this "provider" builds as
            // it goes.
            let label = Text::new([1.0, 1.0, 0.0], 2.5, 0.0, "∅45,6")?
                .with_origin(Origin::from_handle(ItemHandle::new(0x64C)));
            sink.primitive(label.into())?;
            report.record(PrimitiveKind::Text);

            // The text contract, from the provider's side: what it could not
            // decode and what it could not recover both become diagnostics.
            let raw = r"94\U+00B0";
            if let Some(escape) = undecoded_escape(raw) {
                report.push(
                    Diagnostic::error(
                        DiagnosticCode::TEXT_ESCAPE_NOT_DECODED,
                        format!("no decoder for {escape}"),
                    )
                    .with_entity(ItemHandle::new(0x64B))
                    .with_view(view),
                );
            }
            report.push(
                Diagnostic::error(
                    DiagnosticCode::TEXT_NOT_RECOVERED,
                    "multiline attribute text is in an embedded MTEXT this build does not read",
                )
                .with_entity(ItemHandle::new(0x79D))
                .with_view(view),
            );

            if self.truncate {
                report.mark_truncated("view 0 ended 2 records short of its total");
            } else {
                report.mark_complete();
            }
            // The implementation finishes the sink it was handed.
            sink.finish()
        }
    }

    /// The end-to-end path a real provider will walk, with no CAD library
    /// present. #1029's second and third acceptance criteria.
    #[test]
    fn a_mock_decoder_drives_both_sink_paths_with_no_cad_dependency() {
        let decoder = FixtureDecoder { truncate: false };
        assert!(decoder.probe(CadSource::Bytes(b"AC1032...")).unwrap());
        assert!(
            !decoder.probe(CadSource::Bytes(b"GIF89a")).unwrap(),
            "bytes that are not a drawing are a false probe, not an error"
        );

        let drawing = decoder
            .open(CadSource::Bytes(b"AC1032..."))
            .expect("the fixture bytes are a drawing");

        let views = drawing.views().unwrap();
        assert_eq!(views.len(), 2);
        assert_eq!(views[0].kind(), ViewKind::Model);
        assert_eq!(views[1].kind(), ViewKind::Layout);
        assert!(
            views[1].extents().is_none(),
            "the inverted box is not a usable extent"
        );

        let mut sink = CollectSink::new();
        let mut report = DecodeReport::new();
        drawing.decode(0, &mut sink, &mut report).unwrap();

        // Two primitives arrived through `primitives`, one through
        // `primitive`, and the order they were emitted in survived.
        assert_eq!(sink.len(), 3);
        assert_eq!(
            sink.collected()
                .iter()
                .map(Primitive::kind)
                .collect::<Vec<_>>(),
            vec![
                PrimitiveKind::Line,
                PrimitiveKind::Line,
                PrimitiveKind::Text
            ]
        );
        assert!(sink.is_finished());

        assert_eq!(report.counts().get(PrimitiveKind::Line), 2);
        assert_eq!(report.counts().get(PrimitiveKind::Text), 1);
        assert!(report.is_complete());
    }

    /// The text contract from the provider's side: neither defect becomes a
    /// primitive, and both become a structured diagnostic naming the entity.
    #[test]
    fn the_text_defects_arrive_as_diagnostics_and_not_as_primitives() {
        let decoder = FixtureDecoder { truncate: false };
        let drawing = decoder.open(CadSource::Bytes(b"AC1032...")).unwrap();
        let mut sink = CollectSink::new();
        let mut report = DecodeReport::new();
        drawing.decode(0, &mut sink, &mut report).unwrap();

        let texts: Vec<&str> = sink
            .collected()
            .iter()
            .filter_map(|p| match p {
                Primitive::Text(t) => Some(t.content()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec!["∅45,6"],
            "only the decoded string crosses; the escape and the unrecovered \
             text are diagnostics"
        );

        let codes: Vec<DiagnosticCode> = report.errors().map(Diagnostic::code).collect();
        assert_eq!(
            codes,
            vec![
                DiagnosticCode::TEXT_ESCAPE_NOT_DECODED,
                DiagnosticCode::TEXT_NOT_RECOVERED
            ]
        );
        assert_eq!(
            report.diagnostics()[0].entity(),
            Some(ItemHandle::new(0x64B)),
            "a diagnostic that cannot name the entity is a count"
        );
    }

    /// A truncated decode says so, and cannot be mistaken for a short
    /// drawing.
    ///
    /// The negative control for `is_complete`: the same decoder, the same
    /// primitives, one flag.
    #[test]
    fn a_truncated_decode_is_distinguishable_from_a_short_drawing() {
        let mut short = CollectSink::new();
        let mut complete = DecodeReport::new();
        FixtureDecoder { truncate: false }
            .open(CadSource::Bytes(b"AC1032"))
            .unwrap()
            .decode(0, &mut short, &mut complete)
            .unwrap();

        let mut cut = CollectSink::new();
        let mut truncated = DecodeReport::new();
        FixtureDecoder { truncate: true }
            .open(CadSource::Bytes(b"AC1032"))
            .unwrap()
            .decode(0, &mut cut, &mut truncated)
            .unwrap();

        assert_eq!(
            short.len(),
            cut.len(),
            "the two decodes emit the same primitives, so the primitives \
             cannot be what tells them apart"
        );
        assert!(complete.is_complete());
        assert!(!truncated.is_complete());
        assert!(
            truncated
                .errors()
                .any(|d| d.code() == DiagnosticCode::DECODE_TRUNCATED)
        );
    }

    /// A decode that dies partway hands the caller all three facts at once.
    ///
    /// This is the arm the out-parameter exists for, and the one no other
    /// fixture in this module can reach: the provider failed *after* pushing
    /// primitives. Three things are true at that moment — the provider's
    /// typed error, the counts and diagnostics accumulated so far, and the
    /// fact that the sink holds a valid prefix — and a
    /// `Result<DecodeReport, CadError>` can carry two of them. All three are
    /// asserted here, off the same failing call.
    #[test]
    fn a_decode_that_fails_partway_keeps_the_report_the_prefix_and_the_typed_error() {
        /// The provider's own error, which is what has to survive the wrap.
        #[derive(Debug, thiserror::Error)]
        #[error("the drawing stream was cancelled")]
        struct Cancelled {
            after: u64,
        }

        /// Three lines, one warning, then the provider gives up.
        struct DyingDrawing;

        impl CadDrawing for DyingDrawing {
            fn views(&self) -> Result<Vec<CadView>, CadError> {
                Ok(vec![CadView::new(0, 0, [0.0, 0.0, 10.0, 10.0], 9, "Model")])
            }

            fn decode(
                &self,
                view: u32,
                sink: &mut dyn PrimitiveSink,
                report: &mut DecodeReport,
            ) -> Result<(), CadError> {
                for rung in 0..3 {
                    let y = f64::from(rung);
                    sink.primitive(Line::new([0.0, y, 0.0], [10.0, y, 0.0])?.into())?;
                    report.record(PrimitiveKind::Line);
                }
                report.push(
                    Diagnostic::warning(
                        DiagnosticCode::UNRESOLVED_BLOCK,
                        "the insert at 0x41 names a block this drawing does not hold",
                    )
                    .with_entity(ItemHandle::new(0x41))
                    .with_view(view),
                );

                // The stream died with six of the nine entities unread, so
                // the report says so and the sink is deliberately left
                // unfinished.
                report.mark_truncated("the stream ended 6 entities short of the view's total");
                Err(CadError::provider("fixture", Cancelled { after: 3 }))
            }
        }

        // Two drawings held open at once, which is what `&self` on `decode`
        // buys: the one that dies and the healthy fixture it is read against.
        let healthy = FixtureDecoder { truncate: false }
            .open(CadSource::Bytes(b"AC1032"))
            .unwrap();
        let dying: Box<dyn CadDrawing> = Box::new(DyingDrawing);

        let mut sink = CollectSink::new();
        let mut report = DecodeReport::new().with_diagnostic_limit(4096);
        let err = dying
            .decode(0, &mut sink, &mut report)
            .expect_err("the provider gave up mid-stream");

        // One: the typed error. `Provider` carries the provider's own error
        // rather than its `to_string()`, and a mid-stream failure is the one
        // place that could have flattened it into a diagnostic instead.
        let concrete = std::error::Error::source(&err)
            .expect("the provider's error is the source")
            .downcast_ref::<Cancelled>()
            .expect("and it is still a Cancelled");
        assert_eq!(concrete.after, 3);

        // Two: the report, which is the caller's and is still in the
        // caller's hand.
        assert_eq!(report.counts().get(PrimitiveKind::Line), 3);
        assert_eq!(
            report.diagnostics().len(),
            2,
            "the unresolved block and the truncation both survived the failure"
        );
        assert!(report.is_truncated());
        assert!(!report.is_complete());
        assert!(
            report
                .errors()
                .any(|d| d.code() == DiagnosticCode::DECODE_TRUNCATED)
        );
        assert_eq!(report.dropped_diagnostics(), 0);
        assert_eq!(
            report.diagnostic_limit(),
            4096,
            "the caller's own bound took effect, which it cannot when the \
             report is the decode's return value"
        );

        // Three: the prefix. Three primitives arrived, and the sink was not
        // finished — which is the whole difference between a prefix and a
        // drawing.
        assert_eq!(sink.len(), 3);
        assert!(
            !sink.is_finished(),
            "an implementation that returns `Err` must not finish the sink, \
             or `is_finished` stops discriminating"
        );

        // And the positive half of that same signal, from the drawing held
        // open alongside: a decode that returned `Ok` did finish its sink.
        let mut whole = CollectSink::new();
        let mut whole_report = DecodeReport::new();
        healthy.decode(0, &mut whole, &mut whole_report).unwrap();
        assert!(whole.is_finished());
        assert!(whole_report.is_complete());
    }

    /// A view index the drawing does not have is a typed error.
    #[test]
    fn an_unknown_view_index_is_refused_by_name() {
        let decoder = FixtureDecoder { truncate: false };
        let drawing = decoder.open(CadSource::Bytes(b"AC1032")).unwrap();
        let mut sink = CollectSink::new();
        let mut report = DecodeReport::new();

        let err = drawing
            .decode(9, &mut sink, &mut report)
            .expect_err("view 9 does not exist");
        assert!(matches!(err, CadError::NoSuchView { index: 9, views: 2 }));
    }

    /// A source this decoder does not read is refused by name, not by panic
    /// and not by an empty drawing.
    #[test]
    fn a_source_the_decoder_does_not_read_is_refused() {
        let opened = FixtureDecoder { truncate: false }.open(CadSource::Bytes(b"GIF89a"));
        let Err(err) = opened else {
            panic!("a GIF must not open as a drawing");
        };
        assert!(matches!(
            err,
            CadError::UnrecognisedSource { decoder: "fixture" }
        ));
    }

    /// The default batch path forwards to `primitive`, so a sink that
    /// implements only the required method still gets the whole run.
    ///
    /// This is the half of the duality `CollectSink` cannot prove, because
    /// `CollectSink` overrides `primitives`.
    #[test]
    fn the_default_batch_path_forwards_to_the_one_required_method() {
        struct CountOnly {
            seen: usize,
        }

        impl PrimitiveSink for CountOnly {
            fn primitive(&mut self, _primitive: Primitive) -> Result<(), CadError> {
                self.seen += 1;
                Ok(())
            }
        }

        let mut sink = CountOnly { seen: 0 };
        let batch = vec![
            Primitive::from(Line::new([0.0; 3], [1.0, 0.0, 0.0]).unwrap()),
            Primitive::from(Line::new([0.0; 3], [0.0, 1.0, 0.0]).unwrap()),
        ];
        sink.primitives(&mut batch.into_iter()).unwrap();

        assert_eq!(
            sink.seen, 2,
            "a sink that implements only the required method still has to \
             receive the whole run"
        );

        // And `finish` is default-provided too, so the floor really is one
        // method.
        sink.finish().unwrap();
    }

    /// The default batch path stops at the first refusal rather than
    /// swallowing the rest of the run.
    #[test]
    fn the_default_batch_path_stops_at_the_first_refusal() {
        struct RefuseSecond {
            seen: usize,
        }

        impl PrimitiveSink for RefuseSecond {
            fn primitive(&mut self, _primitive: Primitive) -> Result<(), CadError> {
                self.seen += 1;
                if self.seen == 2 {
                    return Err(CadError::sink(std::io::Error::other("full")));
                }
                Ok(())
            }
        }

        let mut sink = RefuseSecond { seen: 0 };
        let batch: Vec<Primitive> = (0..5)
            .map(|i| Primitive::from(Line::new([0.0; 3], [f64::from(i) + 1.0, 0.0, 0.0]).unwrap()))
            .collect();

        assert!(sink.primitives(&mut batch.into_iter()).is_err());
        assert_eq!(
            sink.seen, 2,
            "a refusal has to end the run, or a sink that ran out of room \
             keeps being handed primitives it cannot take"
        );
    }

    /// A `&mut` to a sink and a boxed sink are both sinks, so
    /// `&mut dyn PrimitiveSink` can be passed on.
    #[test]
    fn a_borrowed_and_a_boxed_sink_forward_every_method() {
        fn drive(sink: &mut dyn PrimitiveSink) -> Result<(), CadError> {
            sink.primitive(Line::new([0.0; 3], [1.0, 0.0, 0.0])?.into())?;
            sink.finish()
        }

        struct Tally(Rc<RefCell<(usize, usize)>>);

        impl PrimitiveSink for Tally {
            fn primitive(&mut self, _primitive: Primitive) -> Result<(), CadError> {
                self.0.borrow_mut().0 += 1;
                Ok(())
            }

            fn finish(&mut self) -> Result<(), CadError> {
                self.0.borrow_mut().1 += 1;
                Ok(())
            }
        }

        let shared = Rc::new(RefCell::new((0, 0)));

        // Through the `&mut T` impl: `&mut &mut Tally` is what coerces.
        let mut owned = Tally(Rc::clone(&shared));
        let mut borrowed: &mut Tally = &mut owned;
        drive(&mut borrowed).unwrap();

        // And through the `Box<T>` impl.
        let mut boxed: Box<Tally> = Box::new(Tally(Rc::clone(&shared)));
        drive(&mut boxed).unwrap();

        assert_eq!(
            *shared.borrow(),
            (2, 2),
            "both wrappers have to forward both methods; a wrapper that \
             dropped `finish` would leave the second number at 1"
        );
    }

    /// The extents rule is one rule, and it is the same one for a view and for
    /// four loose numbers.
    #[test]
    fn the_extents_rule_refuses_the_inverted_box_and_a_nan() {
        assert_eq!(Extents::new([1e20, 1e20, -1e20, -1e20]), None);
        assert_eq!(Extents::new([f64::NAN, 0.0, 10.0, 10.0]), None);
        assert_eq!(Extents::new([0.0, 0.0, 10.0, f64::INFINITY]), None);
        assert_eq!(Extents::new([0.0, 5.0, 10.0, 1.0]), None);

        let box_ = Extents::new([-1.0, -2.0, 3.0, 4.0]).expect("a real box");
        assert_eq!(box_.into_array(), [-1.0, -2.0, 3.0, 4.0]);
        assert_eq!(box_.width(), 4.0);
        assert_eq!(box_.height(), 6.0);

        // A degenerate-but-valid box: a drawing one unit wide and flat.
        assert!(Extents::new([0.0, 0.0, 0.0, 0.0]).is_some());
    }

    /// An unknown view kind stays representable rather than collapsing onto
    /// `Model`, which would silently tile the wrong thing.
    #[test]
    fn an_unknown_view_kind_is_not_read_as_model_space() {
        let view = CadView::new(3, 77, [0.0, 0.0, 1.0, 1.0], 0, "something later");
        assert_eq!(view.kind(), ViewKind::Unknown);
        assert_eq!(view.raw_kind(), 77);
        assert_eq!(view.index(), 3);
        assert_eq!(view.name(), "something later");
        assert_eq!(view.entity_count(), 0);
    }
}
