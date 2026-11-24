//! The conversion API (issue #1048, N7.1): a DWG, a view profile and resource limits in, a
//! published `.pvec` artifact and its conversion report out.
//!
//! [`PvecConversion`] is the one surface an application calls. It is built the way `EngineBuilder`
//! is, and [`convert`](PvecConversion::convert) drives the pieces the earlier phases built and
//! kept internal:
//!
//! ```text
//! provider (acadrust by default)
//!   -> view profile (layers, depth)     which leaves are drawn, and which were excluded
//!   -> block flattening                 one composed map per leaf
//!   -> extraction + transform promotion one PVec curve per supported leaf, a class for the rest
//!   -> ordering                         dense file-local ids
//!   -> writer funnel                    the artifact
//!   -> conversion report                one row per entity met, checked against the count met
//!   -> atomic publish                   artifact and report, both or neither
//! ```
//!
//! Every entity the conversion meets is counted exactly once: a drawn leaf, a leaf the profile
//! excluded, and an INSERT that could not be expanded. Nothing is tessellated and nothing is
//! repaired; an entity PVec cannot carry is reported with its reason code and left out.
//!
//! The drawing's unit comes from the provider (acadrust reads the header) or from
//! [`PvecConversion::unit`]. A unit PVec cannot declare is refused by the writer, not rescaled.
//!
//! # Example
//!
//! ```
//! use std::path::PathBuf;
//! use libviprs::cad::CadSource;
//! use libviprs::pvec::SourceUnit;
//! use libviprs::pvec::publish::open_published;
//! use libviprs::pvec_convert::PvecConversion;
//!
//! let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
//!     .join("tests/fixtures/pvec/provider/vc_layers.dwg");
//! let dir = std::env::temp_dir().join(format!("libviprs-doc-convert-{}", std::process::id()));
//! std::fs::create_dir_all(&dir)?;
//! let dest = dir.join("layers.pvec");
//!
//! // The fixture's header declares no unit, and a unit PVec cannot declare is refused, so say one.
//! let out = PvecConversion::new(CadSource::Path(&src), &dest)
//!     .unit(SourceUnit::Meters)
//!     .convert()?;
//! assert_eq!(out.report.totals.encountered, 9);
//! assert_eq!(open_published(&dest)?.entity_count()?, out.written);
//! # std::fs::remove_dir_all(&dir)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Needs the `pvec-convert` feature.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use thiserror::Error;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

mod calibration;
mod capability;
mod dwg_image;
mod georef;
mod layouts;
mod manifest;
mod pdf_hatch;
mod pdf_plot;
pub use pdf_plot::{PdfPlot, PdfPlotOutput};
#[cfg(feature = "pvec-raster")]
mod pdf;
mod xref;
pub use crate::pvec::LayerOverride;
pub use crate::pvec::writer::recalibrate;
pub use calibration::{
    Calibration, CalibrationError, ControlPoint, CrsDefinition, CrsKind, DefinitionFormat,
    Georeference, GeoreferenceMethod, ScaleClaim, ScaleProvenance,
};
pub use capability::{CapabilityPolicy, ConversionProfile};
#[allow(deprecated)]
pub use georef::{
    GeoreferenceError, GeoreferenceHandoff, HandoffV1, SidecarImport, SourceBox, UnitRecord,
    georef_path, import_sidecar,
};
pub use layouts::{
    LayoutSetError, LayoutSetOutput, PaperSheet, PvecLayoutSet, SheetOutcome, SheetSkip,
    paper_sheets,
};
pub use manifest::{
    CadChoices, CadManifest, CadMode, CadText, LayerCounts, LayerStateNotFound, ManifestColour,
    ManifestFile, ManifestLayer, ManifestLayerState, ManifestLayout, ManifestMask,
    ManifestModelSpace, ManifestPaper, ManifestPlot, ManifestViewport, ManifestWarning, PlotsCount,
    PlotsVisibility, ResolvedLayer, StateLayer, TextSwitch, Visibility, VpRef, apply_choices,
    cad_manifest, check_layer_state, resolve_visibility,
};
#[cfg(feature = "pvec-raster")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-raster")))]
pub use pdf::{PdfConversion, PdfConversionOutput, PdfConvertError};

use crate::CancelToken;
use crate::cad::CadSource;
use crate::observe::{EngineEvent, EngineObserver};
use crate::pvec::annotation::{self, Body};
use crate::pvec::bounds::BoundsReport;
use crate::pvec::classify::{ElementClass, ElementRules};
use crate::pvec::components::{
    ComponentSource, ComponentsError, FillBody, PlacedPatternLine, StreamSelection, TextBody,
    annotation_streams, build_streams,
};
use crate::pvec::curve_clip::{CurveClip, CurveClipError, clip_curve};
use crate::pvec::diagnostics::{Diagnostic, DiagnosticCode};
use crate::pvec::dwg_acadrust::AcadrustProvider;
use crate::pvec::dwg_basic::{self, BasicError};
use crate::pvec::dwg_spline::{self, SplineError};
use crate::pvec::fidelity::{Basis, Classification, FidelityReport, ReasonCode, Tally, TallyError};
use crate::pvec::flatten::{
    BlockResolver, FlatEntity, FlattenBounds, FlattenError, LineageStore, SkipReason,
    SkippedInsert, UnplacedLeaf,
};
use crate::pvec::mapping::{kind_of, row_for};
use crate::pvec::ordering::key_cmp;
use crate::pvec::provenance::NoProvenance;
use crate::pvec::publish::{PublishError, StreamedPublishError, publish_streamed_with};
use crate::pvec::regions::{
    MAX_EDGES, RegionConfig, RoomLabel, WallEdge, derive_regions, semantic_stream,
};
use crate::pvec::report::{
    BoundsSplit, CapabilitySummary, ComponentsSummary, ConversionReport, EntityRow, RegionsSummary,
    ReportBuilder, ReportError, SheetPlotSummary, SheetSummary, SheetViewport, UnresolvedXrefRow,
    ViewSummary, XrefSummary,
};
use crate::pvec::style::{DwgLineweight, LineStyleComponent, dwg_line_style, dwg_style_input};
use crate::pvec::transform::{ComposedTransform, promote};
use crate::pvec::writer::{
    CapabilityId, LeafConfig, OpenArtifact, SpillConfig, WriteConfig, WriteError, metres_per_unit,
    open_geometry,
};
use crate::pvec::{
    DwgProvider, EntityId, EntityScope, EntityStyle, ExcludedEntity, FidelityClass, LayerStateRule,
    LayoutKind, Limit, PolicyReason, ProvenanceStub, ProviderError, RawEntity, RawGeometry,
    SourceColour, SourceFact, SourceLimits, SourceUnit, SpaceSelection, ViewProfile,
    ViewProfileError, ViewportClip, ViewportClipping, ViewportEntity, ViewportFreezeRule,
    select_layout,
};
use crate::pvec::{LeafPaint, Normalized2DGeometry, ViewedEntity};
use crate::sync_queue::bounded;
use pvec_geometry::{Curve, Curve2d, CurveKind, Line, Point2, Polyline, Trimmed};

/// Entities per batch: the cancel token is looked at, and a progress event is sent, once per batch.
const BATCH: usize = 256;

/// The queue between the classifier and its consumer holds this many outcomes unless
/// [`PvecConversion::queue_capacity`] says otherwise.
const DEFAULT_QUEUE_CAPACITY: usize = 64;

/// Why a conversion failed. Every variant that wraps a lower error keeps it as `source()`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PvecConvertError {
    /// The provider could not open the source.
    #[error("the source could not be opened")]
    Open(#[source] ProviderError),
    /// The view profile could not be applied.
    #[error("the view profile could not be applied")]
    View(#[source] ViewProfileError),
    /// Block expansion failed.
    #[error("blocks could not be expanded")]
    Flatten(#[source] FlattenError),
    /// No unit was given and the provider cannot read one.
    #[error("the drawing's unit is not known and none was given")]
    UnitRequired,
    /// The writer refused.
    #[error("the artifact could not be written")]
    Write(#[source] WriteError),
    /// PVec's writer refused every curve, so there is no artifact. The report and the counts say
    /// what each entity was, with the refused ones counted as `pvec-cannot-store`; nothing was
    /// published.
    #[error("PVec's writer refused every curve, so no artifact was written")]
    NothingStorable {
        /// The conversion report, as it would have been published beside the artifact.
        report: Box<ConversionReport>,
        /// Entities per fidelity class, counted independently of the report rows.
        fidelity: Box<FidelityReport>,
    },
    /// The class counts disagree with the entities met.
    #[error("the class counts do not add up")]
    Tally(#[source] TallyError),
    /// A component stream could not be built.
    #[error("a component stream could not be built")]
    Components(#[source] ComponentsError),
    /// The report could not be built.
    #[error("the report could not be built")]
    Report(#[source] ReportError),
    /// The artifact and report could not be published.
    #[error("the artifact could not be published")]
    Publish(#[source] PublishError),
    /// An entity needs a PVec capability the run did not allow, and the policy is
    /// [`CapabilityPolicy::RejectInput`].
    #[error(
        "{source_class} (handle {handle:?}) needs the PVec capability {name} ({}), which is not allowed",
        capability.get()
    )]
    CapabilityNotAllowed {
        /// PVec's identifier of the capability.
        capability: CapabilityId,
        /// Its registry name.
        name: &'static str,
        /// The drawing's class name of the entity.
        source_class: String,
        /// The entity's handle, when the drawing gave one.
        handle: Option<u64>,
        /// The entity's own layer.
        layer: String,
        /// The INSERT chain the entity was reached through, spelled `block@HANDLE/...`.
        block_path: String,
    },
    /// A placed DWG IMAGE's raster could not be written (#1667).
    #[error("a placed image could not be written")]
    Rasters(#[source] crate::raster_pvec::container::RasterWriteError),
    /// The thumbnail could not be drawn (#1629). The artifact was published.
    #[cfg(feature = "pvec-raster-thumbnail")]
    #[error("the thumbnail could not be drawn")]
    Thumbnail(#[source] crate::pvec::thumbnail::ThumbnailError),
    /// The cancel token fired. Nothing was published.
    #[error("cancelled")]
    Cancelled,
    /// The wall-clock deadline passed (#1674). Nothing was published.
    #[error("the deadline passed")]
    Deadline,
}

/// What a conversion produced.
#[derive(Debug)]
pub struct PvecConversionOutput {
    /// The published artifact, readable with [`open_published`](crate::pvec::publish::open_published).
    pub artifact: PathBuf,
    /// The conversion report, also published beside the artifact.
    pub report: ConversionReport,
    /// Entities per fidelity class, counted independently of the report rows.
    pub fidelity: FidelityReport,
    /// Entities written into the artifact.
    pub written: u64,
    /// The most outcomes that were waiting for the consumer at once. Never above the capacity set
    /// with [`PvecConversion::queue_capacity`].
    pub peak_queue_depth: u32,
    /// The line families of each pattern hatch the artifact carries, placed in the artifact's
    /// space and keyed by the index of the hatch's first boundary entity in the artifact. The PVEC
    /// hatch stream keeps only a pattern's name, angle and scale, so a plot draws the pattern from
    /// these. Empty without `fills(true)` and the component streams.
    pub hatch_patterns: BTreeMap<u32, Vec<PlacedPatternLine>>,
    /// The sheet as a PNG, when [`PvecConversion::thumbnail`] asked for one (#1629).
    #[cfg(feature = "pvec-raster-thumbnail")]
    pub thumbnail: Option<Vec<u8>>,
}

/// One conversion, configured the way `EngineBuilder` configures a pyramid run.
pub struct PvecConversion<'a> {
    source: CadSource<'a>,
    dest: &'a Path,
    view_profile: ViewProfile,
    unit: Option<SourceUnit>,
    leaves: Option<LeafConfig>,
    spill: Option<SpillConfig>,
    cancel: CancelToken,
    gate: Gate,
    calibration: Option<Calibration>,
    observer: Option<Arc<dyn EngineObserver>>,
    queue_capacity: usize,
    components: StreamSelection,
    regions: Option<RegionConfig>,
    texts: bool,
    fills: bool,
    element_rules: Option<ElementRules>,
    plot_style: Option<acadrust::plot_style::PlotStyleTable>,
    plot_settings: PlotSettings,
    /// The wall-clock bound set with [`deadline`](Self::deadline), until the run starts its clock.
    deadline: Option<std::time::Duration>,
    /// A paper sheet with nothing to stroke is a success with `written == 0` and no artifact,
    /// for a caller (the PDF plot, #1636) that still wants the empty page.
    empty_sheet_ok: bool,
    #[cfg(feature = "pvec-raster-thumbnail")]
    thumbnail_width: Option<u32>,
    xrefs: Vec<(String, CadSource<'a>)>,
    xref_note: Arc<Mutex<xref::XrefNote>>,
    /// The directory a DWG IMAGE's file is read from (#1667); `None` places no image.
    images: Option<PathBuf>,
}

/// What a paper sheet plots (issue #1628, spec section 6 of the CAD import): the choices a caller
/// makes over the layout's own plot settings. [`Default`] is the layout as it stands, which is
/// what a conversion does without this.
///
/// It only reaches a paper layout whose plot settings the drawing reports. Without them the frame
/// is the extents whatever is asked, and the report notes `plot-settings-not-applied`.
///
/// The layout's own rotation (DXF `PLOTROTATION`) is always honoured, it is not a choice made here:
/// 0 is portrait, 1 is landscape (the paper's sides swap), 2 is portrait upside-down (the content
/// turns 180 degrees about the frame's centre) and 3 is landscape upside-down (both). `orientation`
/// is applied after it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlotSettings {
    /// What to plot.
    pub area: PlotArea,
    /// Paper size in millimetres, width then height, before the layout's rotation. `None` is the
    /// layout's own paper. An override clears the layout's margins, which belong to its paper.
    pub paper_mm: Option<[f64; 2]>,
    /// Portrait or landscape. `None` is the layout's own, after its rotation.
    pub orientation: Option<Orientation>,
    /// Scale the plot so what it covers fits the printable area of the paper. Under
    /// [`PlotArea::Extents`] that is the sheet's extents. Under [`PlotArea::Layout`] it is the
    /// layout's own printable area, which only changes anything with a paper override. The frame is
    /// the part of the drawing the paper then shows; the calibration stays the plot scale, and the
    /// fit ratio is a note in the report.
    pub fit_to_paper: bool,
    /// The plot scale in place of the layout's own: a preset and a custom scale are both a
    /// [`PlotScale`]. `None` is the layout's own.
    pub scale: Option<PlotScale>,
    /// The plot offset in millimetres, in place of the layout's own plot origin. `None` is the
    /// layout's own.
    pub offset_mm: Option<[f64; 2]>,
    /// Centre the extents on the printable area. The offset is then ignored. Only for
    /// [`PlotArea::Extents`], as in AutoCAD; under `Layout` it is ignored and the report says so.
    pub center: bool,
    /// Lineweights print. `None` is the layout's own; `Some(false)` drops widths.
    pub print_lineweights: Option<bool>,
    /// Paper space draws after what the viewports show. `None` is the layout's own.
    pub paperspace_last: Option<bool>,
    /// The device a PC3 describes. It fills what the layout's own page setup leaves empty (a zero
    /// paper, no margins) and never replaces what the layout says (issue #1673). `None` is no PC3.
    pub device: Option<crate::cad::plot_files::PlotDevice>,
}

/// A plot scale as the plot dialog's custom row gives it: `paper` paper units of `paper_unit` to
/// `drawing` drawing units, each the size of the conversion's own unit. `1 inch = 50 units` is
/// `{ paper: 1.0, paper_unit: Inches, drawing: 50.0 }`, whatever the sheet's own unit is. The
/// calibration record carries `paper_unit` as its paper unit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlotScale {
    /// Paper units, in `paper_unit`.
    pub paper: f64,
    /// The unit the paper side is in.
    pub paper_unit: SourceUnit,
    /// Drawing units.
    pub drawing: f64,
}

/// Portrait or landscape, the drawing orientation of the plot dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Orientation {
    /// Taller than wide.
    Portrait,
    /// Wider than tall.
    Landscape,
}

/// The part of a layout a plot covers.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub enum PlotArea {
    /// The layout's own frame, the paper placed from its margins and origin.
    #[default]
    Layout,
    /// The extents of what the sheet draws: its own kept geometry and the rectangles of its drawn
    /// viewports. The paper is placed from the extents' lower left corner.
    Extents,
    /// The box `[x0, y0, x1, y1]` in paper-space drawing units, as it is picked on the layout:
    /// the frame is that box drawn at the plot scale, as the linework is (#1665), and nothing else
    /// moves. A layout whose own plot type says window is framed the
    /// same way when this is not asked for.
    Window([f64; 4]),
}

impl<'a> PvecConversion<'a> {
    /// A conversion of `source` into `dest`, under the default view profile.
    #[must_use]
    pub fn new(source: CadSource<'a>, dest: &'a Path) -> Self {
        Self {
            source,
            dest,
            view_profile: ViewProfile::default_profile(),
            unit: None,
            leaves: None,
            spill: None,
            cancel: CancelToken::new(),
            gate: Gate::of(ConversionProfile::Default),
            calibration: None,
            observer: None,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            components: StreamSelection::NONE,
            regions: None,
            texts: false,
            fills: false,
            element_rules: None,
            plot_style: None,
            plot_settings: PlotSettings::default(),
            deadline: None,
            empty_sheet_ok: false,
            #[cfg(feature = "pvec-raster-thumbnail")]
            thumbnail_width: None,
            xrefs: Vec::new(),
            xref_note: Arc::default(),
            images: None,
        }
    }

    /// Place a DWG IMAGE as a raster (#1667). The file its definition names is looked up by file
    /// name in `dir` (the directory part of the path the drawing carries is never followed, so a
    /// drawing cannot point at a file outside `dir`), decoded, and written at the IMAGE's
    /// insertion point, scale and rotation, cut to its clip, in the frame of the linework around
    /// it. An IMAGE whose file is not in `dir`, or does not decode, or is switched off, is not
    /// placed and its row says why. A file that resolves outside `dir` (a symlink), is not a regular file,
    /// declares more than 64 Mpixel, or would take the conversion past 256 images or 256 Mpixel in
    /// all, is refused the same way. Off by default, and then an IMAGE is only counted (#1328).
    #[must_use]
    pub fn images(mut self, dir: impl Into<PathBuf>) -> Self {
        self.images = Some(dir.into());
        self
    }

    /// Carry TEXT, MTEXT, ATTDEF, MULTILEADER and TOLERANCE text, HATCH and SOLID in PVec's text and
    /// hatch streams (#1609, #1658). Shorthand
    /// for [`texts`](Self::texts) and [`fills`](Self::fills) together, so a call to either after it
    /// wins for its half. Off by default, and it only takes effect with the line style stream
    /// selected ([`components`](Self::components)): each text is anchored on a short polyline and
    /// each hatch boundary is an entity of its own, and the line style stream is what marks those
    /// hidden, so without it they would draw as stray lines. With it off, or without the line style
    /// stream, these classes are counted as unmapped, as before.
    #[must_use]
    pub fn annotations(mut self, on: bool) -> Self {
        self.texts = on;
        self.fills = on;
        self
    }

    /// Carry TEXT, MTEXT and ATTDEF in PVec's text stream (#1624), without HATCH and SOLID: those
    /// stay unmapped, each with a warning saying fills are off, until [`fills`](Self::fills) is on.
    /// Off by default, and it needs the line style stream like [`annotations`](Self::annotations).
    ///
    /// A MULTILEADER's text and a TOLERANCE's frame text are carried the same way (#1658). Their
    /// lines (a MULTILEADER's and a LEADER's lines with arrowheads, a TOLERANCE's frame) are
    /// geometry and come through whatever this choice is (#1671). A DIMENSION needs neither option: the
    /// provider reports it as an INSERT of its anonymous block, so its lines are geometry and its
    /// printed value is an MTEXT in that block. Only a dimension whose block is missing comes here,
    /// with its text rebuilt from the measurement and its style (decimal units only).
    #[must_use]
    pub fn texts(mut self, on: bool) -> Self {
        self.texts = on;
        self
    }

    /// Draw a LINE whose ends are at different heights as the segment it covers seen from above,
    /// where a PVec conversion refuses it as 3D (`geometry-3d`): what a plan or a plot shows. Off by
    /// default.
    #[must_use]
    pub fn project_plan(mut self, on: bool) -> Self {
        self.gate.project_plan = on;
        self
    }

    /// Carry HATCH and SOLID in PVec's hatch stream (#1624), without text: TEXT, MTEXT, ATTDEF,
    /// MULTILEADER and TOLERANCE stay unmapped, each with a warning saying texts are off, until [`texts`](Self::texts) is on.
    /// Off by default, and it needs the line style stream like [`annotations`](Self::annotations).
    #[must_use]
    pub fn fills(mut self, on: bool) -> Self {
        self.fills = on;
        self
    }

    /// The layer, block, colour and lineweight rules that say which lines are walls for the
    /// regions step, in place of [`ElementRules::default_profile`] (#1610).
    #[must_use]
    pub fn element_rules(mut self, rules: ElementRules) -> Self {
        self.element_rules = Some(rules);
        self
    }

    /// The component streams to write beside the geometry: layer, line style and provenance, each
    /// on its own (#1525). Off by default, and the decoded geometry and the manifest box are the same either way.
    #[must_use]
    pub fn components(mut self, select: StreamSelection) -> Self {
        self.components = select;
        self
    }

    /// Derive rooms and corridors from the wall linework and write them as PVec's semantic class
    /// stream (#1526). Off (`None`) by default. The walls are the straight lines the default
    /// [`ElementRules`] (or those given to [`element_rules`](Self::element_rules)) call a wall, and
    /// when no rule matches anything, every straight line that is not a column (#1610), the room names come from TEXT, and the decoded
    /// geometry and the manifest box are the same either way: the segments that close a door gap are
    /// never written as geometry, so a room that runs along one is left out of the stream and counted
    /// in the report. Explicit rooms are not passed, so only inferred regions come out.
    #[must_use]
    pub fn regions(mut self, config: Option<RegionConfig>) -> Self {
        self.regions = config;
        self
    }

    /// The drawing an xref block stands for, matched to the host's xref block records by block name
    /// (ASCII case ignored). The path stored in the host is never opened (#1631).
    #[must_use]
    pub fn xref(mut self, block_name: impl Into<String>, source: CadSource<'a>) -> Self {
        self.xrefs.push((block_name.into(), source));
        self
    }

    /// The view profile (N3.3).
    #[must_use]
    pub fn view_profile(mut self, profile: ViewProfile) -> Self {
        self.view_profile = profile;
        self
    }

    /// The plot style table a print conversion applies to each stroke: colour (with the grayscale
    /// policy), screening, lineweight and linetype, per ACI for a CTB and per named style for an
    /// STB (#1630).
    #[must_use]
    pub fn plot_style_table(mut self, table: acadrust::plot_style::PlotStyleTable) -> Self {
        self.plot_style = Some(table);
        self
    }

    /// The resource ceilings (N3.6), carried on the view profile.
    #[must_use]
    pub fn limits(mut self, limits: SourceLimits) -> Self {
        self.view_profile = self.view_profile.with_limits(limits);
        self
    }

    /// Also draw the converted sheet as a PNG `width_px` wide, into
    /// [`PvecConversionOutput::thumbnail`] (#1629). Strokes only, black on white.
    #[cfg(feature = "pvec-raster-thumbnail")]
    #[must_use]
    pub fn thumbnail(mut self, width_px: u32) -> Self {
        self.thumbnail_width = Some(width_px);
        self
    }

    /// What a paper sheet plots: area, paper size, offset and centre, fit to paper (#1628).
    #[must_use]
    pub fn plot_settings(mut self, settings: PlotSettings) -> Self {
        self.plot_settings = settings;
        self
    }

    /// The drawing's unit, overriding what the provider reads. A paper sheet whose plot settings
    /// name inches or millimetres is in that unit instead (P-6, #1562).
    #[must_use]
    pub fn unit(mut self, unit: SourceUnit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Lay a spatial directory and Tier 0 blocks into the artifact. Without it the artifact is
    /// a plain file.
    #[must_use]
    pub fn leaves(mut self, leaves: LeafConfig) -> Self {
        self.leaves = Some(leaves);
        self
    }

    /// Stream the artifact through PVec's spill writer: its pushes are kept in files under `spill.root` and its
    /// memory stays inside `spill.budget`, and the artifact is written straight into the `.pvec.partial` file.
    /// The bytes and the report are those of a conversion without it. The memory is bounded when
    /// [`PvecConversion::leaves`] is set too (a plain artifact holds one block per geometry type).
    #[must_use]
    pub fn spill(mut self, spill: SpillConfig) -> Self {
        self.spill = Some(spill);
        self
    }

    /// The cancel token, polled before any file exists and while entities are classified.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// A wall-clock bound on the conversion (#1674), counted from the moment `convert` or
    /// `convert_with` is called. It is polled where the cancel token is: before the source is read
    /// further, and every batch of entities while they are classified. Past it the conversion fails
    /// with [`PvecConvertError::Deadline`] and publishes nothing, and the cancel token you gave stays
    /// as it was. A cancel from the caller wins when both happened.
    #[must_use]
    pub fn deadline(mut self, within: std::time::Duration) -> Self {
        self.deadline = Some(within);
        self
    }

    /// Starts the deadline's clock, once: the cancel token this conversion polls is a copy that also
    /// reads as cancelled when the time is up.
    fn arm(&mut self) {
        if let Some(at) = self
            .deadline
            .take()
            .and_then(|d| std::time::Instant::now().checked_add(d))
        {
            self.cancel = self.cancel.with_deadline(at);
        }
    }

    /// Receives each diagnostic as it is made (`EngineEvent::ConversionDiagnostic`), and the one
    /// the run fails with. Without one, a drawing that converts cleanly makes no diagnostic at all.
    #[must_use]
    pub fn observer(mut self, observer: Arc<dyn EngineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// How many classified entities may wait for the consumer. The classifier blocks when the
    /// queue is full, so memory stays bounded whatever the consumer's speed; the deepest the queue
    /// was is [`PvecConversionOutput::peak_queue_depth`]. Tuning it is N8.6's measured decision;
    /// the default is a placeholder that is safe, not tuned.
    #[must_use]
    pub fn queue_capacity(mut self, capacity: usize) -> Self {
        self.queue_capacity = capacity.max(1);
        self
    }

    /// The drawing's calibration, written into the artifact's calibration block (issue #1552). A
    /// DWG gets none unless it is given here: its `$INSUNITS` is already the manifest's unit.
    ///
    /// Control points go in the calibration's [`Georeference`], with both their sides. The
    /// source-only `control_points` builder is gone (issue #1556):
    ///
    /// ```compile_fail,E0599
    /// use std::path::Path;
    /// use libviprs::cad::CadSource;
    /// use libviprs::pvec_convert::PvecConversion;
    ///
    /// let _ = PvecConversion::new(CadSource::Bytes(b""), Path::new("o.pvec"))
    ///     .control_points([[0.0, 0.0]]);
    /// ```
    #[must_use]
    pub fn calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = Some(calibration);
        self
    }

    /// A named capability set, exactness rule and policy: [`ConversionProfile::Strict`],
    /// `Default` (the default) or `Permissive`. Replaces any earlier capability setting.
    #[must_use]
    pub fn profile(mut self, profile: ConversionProfile) -> Self {
        let project_plan = self.gate.project_plan;
        self.gate = Gate::of(profile);
        self.gate.project_plan = project_plan;
        self
    }

    /// The PVec capabilities this run may use, replacing the profile's set. An entity that needs
    /// another is handled by the [`capability_policy`](Self::capability_policy). The profile's
    /// exactness rule is kept, and the report names the profile as `Custom`.
    #[must_use]
    pub fn allowed_capabilities(mut self, allowed: impl IntoIterator<Item = CapabilityId>) -> Self {
        self.gate.allowed = allowed.into_iter().collect();
        self.gate.name = "Custom";
        self
    }

    /// What happens to an entity that needs a capability outside the allowed set.
    #[must_use]
    pub fn capability_policy(mut self, policy: CapabilityPolicy) -> Self {
        self.gate.policy = policy;
        self.gate.name = "Custom";
        self
    }

    /// Classifies the drawing without writing anything and returns the per-class counts a
    /// [`convert`](Self::convert) would report, before PVec's writer has seen anything: an entity
    /// counted as supported here can still come back as `pvec-cannot-store` from the conversion,
    /// because the rule is PVec's and is not copied. Under [`CapabilityPolicy::RejectInput`] it fails
    /// the way the conversion would.
    ///
    /// # Errors
    ///
    /// As [`convert`](Self::convert), except that no unit is needed and nothing is written.
    pub fn probe(self) -> Result<FidelityReport, PvecConvertError> {
        let provider = AcadrustProvider::open(self.source).map_err(PvecConvertError::Open)?;
        let c = self.classify(&provider, Some(self.unit.unwrap_or(provider.source_unit())))?;
        c.tally
            .finish(c.encountered)
            .map_err(PvecConvertError::Tally)
    }

    /// As [`probe`](Self::probe), with provider `P`.
    ///
    /// # Errors
    ///
    /// As [`probe`](Self::probe), and [`PvecConvertError::UnitRequired`] when no unit was given and
    /// a paper sheet's plot paper units leave its frame to the drawing's unit, as
    /// [`convert_with`](Self::convert_with) would.
    pub fn probe_with<P: DwgProvider>(self) -> Result<FidelityReport, PvecConvertError> {
        let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
        let c = if self.xrefs.is_empty() {
            self.classify(&provider, self.unit)?
        } else {
            self.classify(&self.with_xrefs(provider)?, self.unit)?
        };
        c.tally
            .finish(c.encountered)
            .map_err(PvecConvertError::Tally)
    }

    /// Converts with the default provider, acadrust. The unit is the header's unless
    /// [`unit`](Self::unit) was set.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`]. After any error the destination and its report are as they were.
    pub fn convert(mut self) -> Result<PvecConversionOutput, PvecConvertError> {
        self.arm();
        let observer = self.observer.clone();
        #[cfg(feature = "pvec-raster-thumbnail")]
        if self.thumbnail_width == Some(0) {
            return report_failure(
                observer.as_deref(),
                Err(PvecConvertError::Thumbnail(
                    crate::pvec::thumbnail::ThumbnailError::ZeroWidth,
                )),
            );
        }
        let result = if self.cancel.is_cancelled() {
            Err(PvecConvertError::Cancelled.settled(&self.cancel))
        } else {
            AcadrustProvider::open(self.source)
                .map_err(PvecConvertError::Open)
                .and_then(|provider| {
                    let unit = self.unit.unwrap_or_else(|| provider.source_unit());
                    self.run(&provider, unit)
                })
        };
        report_failure(observer.as_deref(), result)
    }

    /// Converts with provider `P`. The trait carries no unit, so [`unit`](Self::unit) is required.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError::UnitRequired`] without a unit, otherwise as [`convert`](Self::convert).
    pub fn convert_with<P: DwgProvider>(
        mut self,
    ) -> Result<PvecConversionOutput, PvecConvertError> {
        self.arm();
        let observer = self.observer.clone();
        let result = if self.cancel.is_cancelled() {
            Err(PvecConvertError::Cancelled.settled(&self.cancel))
        } else {
            self.unit
                .ok_or(PvecConvertError::UnitRequired)
                .and_then(|unit| {
                    let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
                    if self.xrefs.is_empty() {
                        return self.run(&provider, unit);
                    }
                    let with = self.with_xrefs(provider)?;
                    self.run(&with, unit)
                })
        };
        report_failure(observer.as_deref(), result)
    }

    /// `host` with the xref drawings the caller supplied behind it (#1631).
    fn with_xrefs<P: DwgProvider>(
        &self,
        host: P,
    ) -> Result<xref::XrefProvider<P>, PvecConvertError> {
        xref::attach(
            host,
            &self.xrefs,
            self.view_profile.limits.max_block_depth,
            self.xref_note.clone(),
        )
        .map_err(PvecConvertError::Open)
    }

    /// Everything a conversion decides about its entities, up to but not including the write.
    /// [`probe`](Self::probe) stops here, so it and [`convert`](Self::convert) cannot disagree.
    ///
    /// `unit` is the drawing's unit, when there is one; a paper sheet whose plot settings name
    /// inches or millimetres is in that unit instead (P-6, #1562).
    fn classify<P: DwgProvider>(
        &self,
        provider: &P,
        unit: Option<SourceUnit>,
    ) -> Result<Classified, PvecConvertError> {
        let Self {
            source,
            view_profile,
            cancel,
            gate,
            observer,
            queue_capacity,
            components,
            regions,
            texts: want_texts,
            fills: want_fills,
            element_rules,
            calibration: caller_calibration,
            plot_style,
            plot_settings,
            images,
            ..
        } = self;
        let observer = observer.as_deref();
        let tell = |event: EngineEvent| {
            if let Some(o) = observer {
                o.on_event(event);
            }
        };
        tell(EngineEvent::ConversionSourceOpened {
            provider: provider.name().to_owned(),
        });
        if cancel.is_cancelled() {
            return Err(PvecConvertError::Cancelled);
        }
        let view = view_profile
            .apply(provider)
            .map_err(PvecConvertError::View)?;
        tell(EngineEvent::ConversionViewResolved {
            included: view.report.included,
            excluded: view.excluded.len() as u64,
        });
        if cancel.is_cancelled() {
            return Err(PvecConvertError::Cancelled);
        }
        let layouts = provider
            .layouts()
            .map_err(|e| PvecConvertError::View(e.into()))?;
        let layout =
            select_layout(&layouts, &view_profile.space).map_err(PvecConvertError::View)?;

        // The viewports of a sheet (#1563), each decided once: the paper view, one that contributes
        // nothing (P-3), one refused with 508 for a view this slice cannot map, or one drawn
        // through its similarity onto the sheet and its rectangle.
        let mut windows: Vec<(ViewportEntity, Window)> = Vec::new();
        // The layers an override froze in each of `windows`, by position (#1627).
        let mut window_hidden: Vec<Vec<String>> = Vec::new();
        // The layers that do not plot. A viewport on one still draws what it shows; only its
        // border is hidden, as AutoCAD prints it (P-3, #1577).
        let no_plot: BTreeSet<String> = if layout.kind == LayoutKind::Paper
            && view_profile.layers.no_plot == LayerStateRule::Exclude
        {
            provider
                .layers()
                .map_err(|e| PvecConvertError::View(e.into()))?
                .into_iter()
                .filter(|l| !l.plottable)
                .map(|l| l.name.to_ascii_lowercase())
                .collect()
        } else {
            BTreeSet::new()
        };
        if layout.kind == LayoutKind::Paper {
            let listed = match provider.viewports() {
                Ok(v) => v,
                Err(ProviderError::CannotReport(_)) => Vec::new(),
                Err(e) => return Err(PvecConvertError::View(e.into())),
            };
            for mut v in listed.into_iter().filter(|v| v.layout == layout.id) {
                // The user's overrides edit this viewport's frozen list before anything is drawn
                // (#1627): a Visible layer is thawed everywhere, a Hidden one that this layout's
                // other viewports show is frozen here, then the viewport's own entries thaw or
                // freeze it here only, the most specific word winning. `added` is what an
                // override froze, so the report can say so.
                let layers = &view_profile.layers;
                v.frozen_layers.retain(|f| {
                    !layers
                        .overrides
                        .iter()
                        .any(|(n, o)| *o == LayerOverride::Visible && n.eq_ignore_ascii_case(f))
                });
                let mut added: Vec<String> = Vec::new();
                for n in layers.hidden_but_shown_in_a_viewport(&layout.name) {
                    added.push(n.to_owned());
                    v.frozen_layers.push(n.to_owned());
                }
                let key = format!("{}/{:X}", layout.name, v.handle.get());
                for (_, name, o) in layers
                    .viewport_overrides
                    .iter()
                    .filter(|(k, ..)| k.eq_ignore_ascii_case(&key))
                {
                    v.frozen_layers.retain(|f| !f.eq_ignore_ascii_case(name));
                    added.retain(|f| !f.eq_ignore_ascii_case(name));
                    if *o == LayerOverride::Hidden {
                        added.push(name.clone());
                        v.frozen_layers.push(name.clone());
                    }
                }
                let window = match viewport_class(&v) {
                    ViewportClass::Paper => Window::Paper,
                    ViewportClass::Dropped(why) => Window::Dropped(why),
                    ViewportClass::Refused(why) => Window::Refused(why),
                    ViewportClass::Drawn(ring) => {
                        // paper = centre + s * (model - target - view centre), for a plan view.
                        // The view centre is in display coordinates, whose origin is the target
                        // (DXF), so a target off the origin shifts the view (#1610).
                        let s = v.height / v.view_height;
                        let (hw, hh) = (v.width / 2.0, v.height / 2.0);
                        let rect = match &ring {
                            Some(r) => r.iter().fold(
                                [
                                    f64::INFINITY,
                                    f64::INFINITY,
                                    f64::NEG_INFINITY,
                                    f64::NEG_INFINITY,
                                ],
                                |b, q| {
                                    [
                                        b[0].min(q[0]),
                                        b[1].min(q[1]),
                                        b[2].max(q[0]),
                                        b[3].max(q[1]),
                                    ]
                                },
                            ),
                            None => [
                                v.center[0] - hw,
                                v.center[1] - hh,
                                v.center[0] + hw,
                                v.center[1] + hh,
                            ],
                        };
                        Window::Drawn {
                            map: [
                                s,
                                0.0,
                                0.0,
                                s,
                                v.center[0] - s * (v.view_center[0] + v.view_target[0]),
                                v.center[1] - s * (v.view_center[1] + v.view_target[1]),
                            ],
                            rect,
                            ring,
                        }
                    }
                };
                windows.push((v, window));
                window_hidden.push(added);
            }
        }
        // `1:N`, one paper unit to N model units, of a drawn viewport.
        let scale_of = |v: &ViewportEntity| format!("1:{}", v.view_height / v.height);
        // P-9: when exactly one viewport shows model space, its scale goes in the record's note.
        let mut drawn_scales = windows
            .iter()
            .filter(|(_, w)| matches!(w, Window::Drawn { .. }))
            .map(|(v, _)| scale_of(v));
        let single_scale = match (drawn_scales.next(), drawn_scales.next()) {
            (Some(one), None) => Some(one),
            _ => None,
        };

        // The paper sheet (#1562): its frame and unit (P-6), its print flags (P-8) and the plot
        // scale as the calibration (P-9). Zero or missing plot settings are not trusted for any of
        // it: the frame is then the kept geometry's extent, in the drawing's unit, and nothing is
        // cut.
        let mut sheet: Option<Sheet> = None;
        if layout.kind == LayoutKind::Paper {
            let fact = match provider.layout_plot(layout.id) {
                Ok(f) => f,
                Err(ProviderError::CannotReport(_)) => SourceFact::NotReportable,
                Err(e) => return Err(PvecConvertError::View(e.into())),
            };
            let text = |v: f64| format!("{v}");
            // Only a CTB is applied: nothing says which named style an entity uses (#1630).
            let stb = plot_style.as_ref().is_some_and(|t| t.is_stb);
            let mut summary = SheetSummary {
                layout: layout.name.clone(),
                frame: "extents".into(),
                unit_source: "drawing".into(),
                calibration: if caller_calibration.is_some() {
                    "caller".into()
                } else {
                    "none".into()
                },
                ..SheetSummary::default()
            };
            if stb {
                summary.notes.push("plot-style-stb-unsupported".into());
            }
            let mut plot_unit = None;
            let mut frame_box = None;
            let mut drop_widths = false;
            let mut written = None;
            let mut place = None;
            let mut windowed = false;
            // What the paper's place rests on: `measured` for a sheet with no margins and no plot
            // origin, whose paper image origin was checked against a plotted sheet (#1686).
            let mut position = "unmeasured";
            let mut flip = false;
            let mut per_unit = 1.0;
            match &fact {
                SourceFact::Absent => summary.frame_reason = Some("plot-settings-absent".into()),
                SourceFact::Present(p) => {
                    summary.plot = Some(SheetPlotSummary {
                        paper_size: p.paper_size.clone(),
                        paper_width: text(p.paper_width),
                        paper_height: text(p.paper_height),
                        paper_units: p.paper_units,
                        rotation: p.rotation,
                        plot_type: p.plot_type,
                        scale_numerator: text(p.scale_numerator),
                        scale_denominator: text(p.scale_denominator),
                        style_sheet: p.style_sheet.clone(),
                        print_lineweights: p.print_lineweights,
                        scale_lineweights: p.scale_lineweights,
                        plot_plot_styles: p.plot_plot_styles,
                    });
                    let sized = |v: f64| v.is_finite() && v > 0.0;
                    let fit = plot_settings.fit_to_paper;
                    // The plot scale, the caller's or the layout's own: `n` paper units to `d`
                    // paper-space drawing units. The sheet's linework is drawn at it (#1665), so one
                    // drawing unit is `drawn_at(sheet unit)` units of paper in the sheet unit. A
                    // scale that is zero or has no physical paper unit draws 1:1, and the record
                    // says why.
                    let [n, d] = plot_settings
                        .scale
                        .map_or([p.scale_numerator, p.scale_denominator], |s| {
                            [s.paper, s.drawing]
                        });
                    let drawn_at = |sheet: SourceUnit| {
                        let paper = plot_settings.scale.map_or(sheet, |s| s.paper_unit);
                        let along = if paper == sheet {
                            Some(1.0)
                        } else {
                            metres_per_unit(paper)
                                .ok()
                                .zip(metres_per_unit(sheet).ok())
                                .map(|(pm, sm)| pm / sm)
                        };
                        match along {
                            Some(a) if sized(n) && sized(d) => n / d * a,
                            _ => 1.0,
                        }
                    };
                    // The PC3 fills what the layout leaves empty and never replaces what it says
                    // (#1673): a paper of no size takes the device's.
                    let device = plot_settings.device.as_ref();
                    let layout_paper = if sized(p.paper_width) && sized(p.paper_height) {
                        [p.paper_width, p.paper_height]
                    } else {
                        device
                            .map(|d| d.paper_mm)
                            .filter(|d| sized(d[0]) && sized(d[1]))
                            .unwrap_or([p.paper_width, p.paper_height])
                    };
                    let [paper_w, paper_h] = plot_settings.paper_mm.unwrap_or(layout_paper);
                    if !(sized(paper_w) && sized(paper_h)) {
                        summary.frame_reason = Some("paper-size-zero".into());
                    } else {
                        if plot_settings.paper_mm.is_none()
                            && layout_paper != [p.paper_width, p.paper_height]
                        {
                            summary.notes.push("plot-device-paper".into());
                        }
                        let sheet_unit = match p.paper_units {
                            0 => Some(SourceUnit::Inches),
                            1 => Some(SourceUnit::Millimeters),
                            // The drawing's unit stands in. `convert_with` refuses without one, so
                            // the probes do too rather than frame the sheet differently.
                            _ => {
                                summary.notes.push("paper-units-fallback".into());
                                Some(unit.ok_or(PvecConvertError::UnitRequired)?)
                            }
                        };
                        // Millimetres of paper in the sheet unit: exact for millimetres, one
                        // division for inches, through PVec's own metres otherwise.
                        let to_unit = |mm: f64| match sheet_unit {
                            Some(SourceUnit::Millimeters) => Some(mm),
                            Some(SourceUnit::Inches) => Some(mm / 25.4),
                            Some(u) => metres_per_unit(u).ok().map(|m| mm * 0.001 / m),
                            None => None,
                        };
                        let (w, h) = if matches!(p.rotation, 1 | 3) {
                            (paper_h, paper_w)
                        } else {
                            (paper_w, paper_h)
                        };
                        // The orientation turns the paper to the shape asked for, after the
                        // layout's rotation.
                        let turned = match plot_settings.orientation {
                            Some(Orientation::Portrait) if w > h => (h, w),
                            Some(Orientation::Landscape) if h > w => (h, w),
                            _ => (w, h),
                        };
                        if turned != (w, h) {
                            summary.notes.push("plot-orientation".into());
                        }
                        let (w, h) = turned;
                        // An override replaces the paper, so the margins that belong to the
                        // layout's paper go with it.
                        let layout_margins = device
                            .filter(|d| {
                                p.margins == [0.0; 4]
                                    && (d.paper_mm[0] - layout_paper[0]).abs() < 0.5
                                    && (d.paper_mm[1] - layout_paper[1]).abs() < 0.5
                            })
                            .map(|d| {
                                let [llx, lly, urx, ury] = d.printable_mm;
                                [llx, lly, d.paper_mm[0] - urx, d.paper_mm[1] - ury]
                            })
                            .filter(|m| m.iter().all(|v| v.is_finite() && *v >= 0.0))
                            .filter(|m| *m != [0.0; 4]);
                        if plot_settings.paper_mm.is_none() && layout_margins.is_some() {
                            summary.notes.push("plot-device-margins".into());
                        }
                        let layout_margins = layout_margins.unwrap_or(p.margins);
                        let margins = if plot_settings.paper_mm.is_some() {
                            [0.0; 4]
                        } else {
                            layout_margins
                        };
                        // The offset is the plot origin the dialog shows, so it replaces the
                        // layout's own.
                        let shift = plot_settings.offset_mm.unwrap_or(p.origin);
                        // What the layout itself prints, for a fit under the Layout area: its
                        // paper (after its rotation) less its own margins.
                        let (lw, lh) = if matches!(p.rotation, 1 | 3) {
                            (layout_paper[1], layout_paper[0])
                        } else {
                            (layout_paper[0], layout_paper[1])
                        };
                        let own = [
                            lw - layout_margins[0] - layout_margins[2],
                            lh - layout_margins[1] - layout_margins[3],
                        ];
                        // The paper's lower left sits at -(margin + plot origin + paper image
                        // origin) from the paper-space origin. The image origin is in drawing units,
                        // so it goes to the paper at the plot scale, in millimetres here like the
                        // rest (#1686). No fixture with non-zero margins or plot origin has
                        // measured the rest, so those sheets report `position: unmeasured`.
                        let su = |v: [f64; 2]| to_unit(v[0]).zip(to_unit(v[1]));
                        // The window is in paper-space drawing units, so it goes to the paper at
                        // the plot scale, the way the linework does (#1665).
                        let window_box = match plot_settings.area {
                            PlotArea::Window(w) => Some(w),
                            PlotArea::Layout if p.plot_type == 4 => Some(p.window),
                            _ => None,
                        };
                        let image = to_unit(1.0)
                            .filter(|per_mm| *per_mm > 0.0)
                            .map(|per_mm| {
                                let at = sheet_unit.map_or(1.0, drawn_at) / per_mm;
                                p.paper_image_origin
                                    .map(|v| if v.is_finite() { v * at } else { 0.0 })
                            })
                            .unwrap_or([0.0; 2]);
                        let converted = su([margins[0], margins[1]])
                            .zip(su([margins[2], margins[3]]))
                            .zip(su([w, h]))
                            .zip(su(own))
                            // The margin and the shift added in millimetres, then converted once.
                            .zip(su([
                                margins[0] + shift[0] + image[0],
                                margins[1] + shift[1] + image[1],
                            ]));
                        if let Some(w) = window_box {
                            if w.iter().all(|v| v.is_finite()) && w[2] > w[0] && w[3] > w[1] {
                                let k = sheet_unit.map_or(1.0, drawn_at);
                                frame_box = Some(w.map(|v| v * k));
                                plot_unit = sheet_unit;
                                windowed = true;
                            } else {
                                summary.frame_reason = Some("plot-window-empty".into());
                            }
                        } else {
                            match converted {
                                Some((((((l, b), (r, t)), size), own), corner)) => {
                                    let at = Place {
                                        margin: [l, b, r, t],
                                        size: [size.0, size.1],
                                        corner: [corner.0, corner.1],
                                    };
                                    // Drawing units to a paper unit. Under the Layout area a fit
                                    // fits the layout's own printable area to the new paper's.
                                    let mut k = 1.0;
                                    if fit && plot_settings.area == PlotArea::Layout {
                                        let printable = [
                                            (at.size[0] - l - r, own.0),
                                            (at.size[1] - b - t, own.1),
                                        ];
                                        let tight = printable
                                            .into_iter()
                                            .filter(|(n, d)| sized(*n) && sized(*d))
                                            .min_by(|x, y| (x.0 / x.1).total_cmp(&(y.0 / y.1)));
                                        match tight {
                                            Some((n, d)) if n != d => {
                                                k = d / n;
                                                summary.notes.push("plot-fit-to-paper".into());
                                                summary.notes.push(format!("plot-fit-ratio 1:{k}"));
                                            }
                                            Some(_) => {
                                                summary.notes.push("plot-fit-no-change".into())
                                            }
                                            None => summary.notes.push("plot-fit-skipped".into()),
                                        }
                                    }
                                    if plot_settings.center
                                        && plot_settings.area == PlotArea::Layout
                                    {
                                        summary.notes.push("plot-centre-not-applicable".into());
                                    }
                                    let (x0, y0) = (-at.corner[0] * k, -at.corner[1] * k);
                                    frame_box =
                                        Some([x0, y0, x0 + at.size[0] * k, y0 + at.size[1] * k]);
                                    plot_unit = sheet_unit;
                                    place = Some(at);
                                    if image != [0.0; 2] {
                                        summary.notes.push("paper-image-origin".into());
                                    }
                                    if margins == [0.0; 4] && shift == [0.0; 2] {
                                        position = "measured";
                                    }
                                }
                                None => summary.frame_reason = Some("unit-not-declarable".into()),
                            }
                        }
                    }
                    if let (Some(b), Some(sheet_unit)) = (frame_box, plot_unit) {
                        summary.frame = "paper".into();
                        summary.position = Some(position.into());
                        summary.sheet_box = Some(b.map(text));
                        if matches!(p.paper_units, 0 | 1) {
                            summary.unit_source = "plot-paper-units".into();
                        }
                        drop_widths = !plot_settings
                            .print_lineweights
                            .unwrap_or(p.print_lineweights);
                        summary.widths_dropped = drop_widths;
                        if plot_settings.print_lineweights.is_some() {
                            summary.notes.push("plot-lineweights-overridden".into());
                        }
                        if plot_settings.paper_mm.is_some() {
                            summary.notes.push("plot-paper-overridden".into());
                        }
                        if plot_settings.offset_mm.is_some()
                            && !(plot_settings.center && plot_settings.area == PlotArea::Extents)
                        {
                            summary.notes.push("plot-offset".into());
                        }
                        if windowed {
                            summary.notes.push("plot-area-window".into());
                        } else if p.plot_type != 5 {
                            summary.notes.push("plot-type-not-applied".into());
                        }
                        // Rotation 1 is the swap of the sides already made; 2 and 3 turn the
                        // sheet's content about the frame's centre once the frame is known.
                        flip = matches!(p.rotation, 2 | 3);
                        per_unit = drawn_at(sheet_unit);
                        if per_unit != 1.0 {
                            summary.notes.push("plot-scale-applied".into());
                        }
                        if plot_settings.scale.is_some() {
                            summary.notes.push("plot-scale-overridden".into());
                        }
                        written = plot_scale_record(
                            [n, d],
                            plot_settings.scale.map_or(sheet_unit, |s| s.paper_unit),
                            sheet_unit,
                            unit,
                            single_scale.as_deref(),
                            caller_calibration.is_some(),
                            &mut summary,
                        );
                    }
                }
                _ => summary.frame_reason = Some("plot-settings-not-reportable".into()),
            }
            if frame_box.is_none() && *plot_settings != PlotSettings::default() {
                summary.notes.push("plot-settings-not-applied".into());
            }
            summary.unit = plot_unit
                .or(unit)
                .map_or("Unknown", SourceUnit::name)
                .into();
            sheet = Some(Sheet {
                summary,
                frame: frame_box,
                unit: plot_unit,
                calibration: written,
                drop_widths,
                place,
                flip,
                per_unit,
            });
        }
        let source_name = match *source {
            CadSource::Path(p) => p.display().to_string(),
            CadSource::Bytes(_) => "(bytes)".to_owned(),
        };
        let flat = BlockResolver::new(provider, FlattenBounds::from_profile(view_profile))
            .projecting(true)
            .flatten(
                &EntityScope::Layout(layout.id),
                &source_name,
                &layout.name,
                &mut NoProvenance,
                &mut LineageStore::new(),
            )
            .map_err(PvecConvertError::Flatten)?;
        let (flat_lines, leader_lines) = line_leaves(provider, &flat.entities);

        // The leaves the profile kept. The view pass decides visibility and the flatten pass owns
        // the composed maps, so a flattened leaf is drawn when the view pass kept the same one.
        // Each is kept with its effective layer, which the layer stream names (#1633).
        let mut kept: BTreeMap<(Option<u64>, String, String), String> = view
            .included
            .iter()
            .map(|e| (leaf_key(e), e.effective_layer.clone()))
            .collect();
        // The colour each kept leaf is drawn in, found by the view pass (#1638); what a viewport
        // shows is read from the model's own pass below.
        let mut paints: BTreeMap<(Option<u64>, String, String), LeafPaint> = view
            .included
            .iter()
            .map(|v| (leaf_key(v), v.paint))
            .collect();
        drop(view.included);
        // A DIMENSION the view pass expanded as an INSERT of its block has no row of its own
        // (#1658), so the report counts them under `DIMENSION.expanded`.
        let dimensions_expanded: u64 = view
            .expanded_classes
            .iter()
            .filter(|(class, _)| is_dimension_class(class))
            .map(|(_, n)| n)
            .sum();
        let excluded_inserts: BTreeSet<u64> = view
            .excluded
            .iter()
            .filter_map(|x| x.handle.map(|h| h.get()))
            .collect();

        let mut items: Vec<Item<'_>> = Vec::new();
        for f in flat.entities.iter().chain(&flat_lines) {
            let p = &f.entity.provenance;
            // A LEADER whose line is drawn is that line and not an unmapped entity as well (#1671).
            let replaced = matches!(f.entity.geometry, RawGeometry::Unmapped { .. })
                && leader_lines.contains(&flat_key(f));
            if !replaced
                && kept.contains_key(&(
                    p.handle.map(|h| h.get()),
                    p.block_path.to_string(),
                    p.class_name.clone(),
                ))
            {
                items.push(Item::Leaf(f));
            }
        }
        for u in &flat.unplaced {
            let p = &u.entity.provenance;
            if kept.contains_key(&(
                p.handle.map(|h| h.get()),
                p.block_path.to_string(),
                p.class_name.clone(),
            )) {
                items.push(Item::Unplaced(u));
            }
        }
        items.extend(view.excluded.iter().map(Item::Excluded));
        for s in &flat.skipped {
            if s.insert_handle
                .is_some_and(|h| excluded_inserts.contains(&h.get()))
            {
                continue;
            }
            items.push(Item::Skipped(s));
        }

        // Model space through the drawn viewports (#1563). It is walked once under the print
        // profile's layer rules and flattened once; each drawn viewport then composes its
        // similarity onto every kept leaf's map (P-4: one entity per viewport that shows it).
        // What the walk excludes or skips is reported once, not once per viewport.
        let first = sheet
            .as_ref()
            .and_then(|_| provider.layout_plot(layout.id).ok())
            .and_then(|f| f.present().map(|p| p.draw_viewports_first))
            .unwrap_or(false);
        // Paper space last is viewports first; the caller's choice replaces the layout's.
        let first = plot_settings
            .paperspace_last
            .filter(|_| sheet.is_some())
            .unwrap_or(first);
        let drawn: Vec<usize> = windows
            .iter()
            .enumerate()
            .filter(|(_, (_, w))| matches!(w, Window::Drawn { .. }))
            .map(|(k, _)| k)
            .collect();
        let mut model_view = None;
        let mut model_flat = None;
        let mut model_lines: Vec<FlatEntity> = Vec::new();
        let mut model_leader_lines = BTreeSet::new();
        // The own layer of every INSERT a model leaf is reached through, so a freeze on an INSERT's
        // layer takes its whole block with it inside that viewport, as the view pass does for a
        // frozen layer (P-3).
        let mut insert_layers: BTreeMap<u64, String> = BTreeMap::new();
        if !drawn.is_empty() {
            if cancel.is_cancelled() {
                return Err(PvecConvertError::Cancelled);
            }
            let model = layouts
                .iter()
                .find(|l| l.kind == LayoutKind::Model)
                .ok_or_else(|| {
                    PvecConvertError::View(ViewProfileError::NoModelSpace {
                        available: layouts.iter().map(|l| l.name.clone()).collect(),
                    })
                })?;
            let mut through_profile = view_profile.clone();
            through_profile.space = SpaceSelection::Model;
            through_profile.viewport_clipping = ViewportClipping::Unsupported;
            through_profile.layers.frozen_in_viewport = ViewportFreezeRule::IgnoreAndReport;
            // A layer hidden for the drawing that one viewport shows again stays in the model
            // walk, and the other viewports freeze it (#1627).
            let rescued: Vec<String> = view_profile
                .layers
                .hidden_but_shown_in_a_viewport(&layout.name)
                .into_iter()
                .map(str::to_owned)
                .collect();
            through_profile
                .layers
                .overrides
                .retain(|(n, _)| !rescued.iter().any(|r| r.eq_ignore_ascii_case(n)));
            model_view = Some(
                through_profile
                    .apply(provider)
                    .map_err(PvecConvertError::View)?,
            );
            model_flat = Some(
                BlockResolver::new(provider, FlattenBounds::from_profile(view_profile))
                    .projecting(true)
                    .flatten(
                        &EntityScope::Layout(model.id),
                        &source_name,
                        &model.name,
                        &mut NoProvenance,
                        &mut LineageStore::new(),
                    )
                    .map_err(PvecConvertError::Flatten)?,
            );
            if let Some(mf) = &model_flat {
                (model_lines, model_leader_lines) = line_leaves(provider, &mf.entities);
            }
            if drawn
                .iter()
                .any(|&k| !windows[k].0.frozen_layers.is_empty())
                && let Some(mf) = &model_flat
            {
                // Each scope an INSERT of some leaf's path sits in is read once: model space, and
                // each block definition by name, whichever instance reached it first. An INSERT's
                // handle and layer inside a definition are the same in every instance (#1581).
                let mut read: BTreeSet<Option<&str>> = BTreeSet::new();
                for f in &mf.entities {
                    let steps = f.entity.provenance.block_path.steps();
                    let mut via_path = crate::pvec::InstancePath::root();
                    for (i, step) in steps.iter().enumerate() {
                        let container = i.checked_sub(1).map(|j| steps[j].block_name.as_str());
                        if read.insert(container) {
                            let scope = match container {
                                None => EntityScope::Layout(model.id),
                                Some(name) => EntityScope::BlockDefinition {
                                    name: name.to_owned(),
                                    reached_via: via_path.clone(),
                                },
                            };
                            let entities = provider
                                .entities(&scope)
                                .and_then(|mut c| c.collect_all())
                                .map_err(|e| PvecConvertError::View(e.into()))?;
                            for e in entities {
                                if let (RawGeometry::Insert(_), Some(h)) =
                                    (&e.geometry, e.provenance.handle)
                                {
                                    insert_layers.insert(h.get(), e.provenance.layer);
                                }
                            }
                        }
                        via_path = via_path.child(step.block_name.clone(), step.insert_handle);
                    }
                }
            }
        }
        // The effective layer of each kept model leaf, which a viewport freeze is matched against.
        if let Some(v) = &model_view {
            paints.extend(v.included.iter().map(|e| (leaf_key(e), e.paint)));
        }
        let model_layers: BTreeMap<(Option<u64>, String, String), String> = model_view
            .as_ref()
            .map(|v| {
                v.included
                    .iter()
                    .map(|e| (leaf_key(e), e.effective_layer.clone()))
                    .collect()
            })
            .unwrap_or_default();
        // Where each item draws: the paper itself, or through window k. The rank puts what the
        // viewports show under the sheet's own entities when `draw_viewports_first` says so. A
        // model leaf is listed once per drawn viewport and its map is composed only when it is
        // classified, so nothing is copied per viewport up front.
        let paper_rank = if first { u32::MAX } else { 0 };
        let mut via: Vec<(u32, Option<usize>)> = vec![(paper_rank, None); items.len()];
        // Each drawn viewport classifies every kept model leaf once more, so the conversion
        // handles the sheet's own items, the model leaves times the drawn viewports, and the
        // model's excluded and skipped rows once. The sheet and the model were each walked under
        // their own guard, so the product is checked here, before anything is listed (#1580).
        if let (Some(mv), Some(mf)) = (&model_view, &model_flat) {
            let shown = mf
                .entities
                .iter()
                .filter(|f| model_layers.contains_key(&flat_key(f)))
                .count();
            let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
            let observed = count(shown)
                .saturating_mul(count(drawn.len()))
                .saturating_add(count(items.len()))
                .saturating_add(count(mv.excluded.len()))
                .saturating_add(count(mf.skipped.len()))
                .saturating_add(count(mf.unplaced.len()));
            let ceiling = view_profile.limits.max_entities;
            if observed > ceiling {
                return Err(PvecConvertError::Flatten(FlattenError::Limit {
                    limit: Limit::Entities,
                    observed,
                    ceiling,
                    block_path: crate::pvec::InstancePath::root(),
                }));
            }
        }
        if let Some(mf) = &model_flat {
            for &k in &drawn {
                for f in mf
                    .entities
                    .iter()
                    .chain(&model_lines)
                    .filter(|f| model_layers.contains_key(&flat_key(f)))
                    .filter(|f| {
                        !(matches!(f.entity.geometry, RawGeometry::Unmapped { .. })
                            && model_leader_lines.contains(&flat_key(f)))
                    })
                {
                    items.push(Item::Leaf(f));
                    via.push((u32::try_from(k).unwrap_or(u32::MAX - 2) + 1, Some(k)));
                }
            }
        }
        // The model leaves are keyed apart from the sheet's by handle and block path, so one map
        // answers for both.
        kept.extend(model_layers.iter().map(|(k, v)| (k.clone(), v.clone())));
        let once = u32::try_from(windows.len()).unwrap_or(u32::MAX - 2) + 1;
        if let (Some(mv), Some(mf)) = (&model_view, &model_flat) {
            let excluded: BTreeSet<u64> = mv
                .excluded
                .iter()
                .filter_map(|x| x.handle.map(|h| h.get()))
                .collect();
            for x in &mv.excluded {
                items.push(Item::Excluded(x));
                via.push((once, None));
            }
            for s in &mf.skipped {
                if s.insert_handle.is_some_and(|h| excluded.contains(&h.get())) {
                    continue;
                }
                items.push(Item::Skipped(s));
                via.push((once, None));
            }
            for u in &mf.unplaced {
                let p = &u.entity.provenance;
                if model_layers.contains_key(&(
                    p.handle.map(|h| h.get()),
                    p.block_path.to_string(),
                    p.class_name.clone(),
                )) {
                    items.push(Item::Unplaced(u));
                    via.push((once, None));
                }
            }
        }

        // Ids are positions in the canonical order, which is decided by block path, handle and
        // arrival position alone (arrival positions are all different, so no later key is ever
        // reached). Positions are sorted by those keys read in place, so no second copy of an
        // entity is made to number them.
        let mut perm: Vec<usize> = (0..items.len()).collect();
        let key = |i: usize| match &items[i] {
            Item::Leaf(f) => (
                &f.entity.provenance.block_path,
                f.entity.provenance.handle,
                i as u64,
            ),
            Item::Excluded(x) => (&x.block_path, x.handle, i as u64),
            Item::Skipped(s) => (&s.block_path, s.insert_handle, i as u64),
            Item::Unplaced(u) => (
                &u.entity.provenance.block_path,
                u.entity.provenance.handle,
                i as u64,
            ),
        };
        perm.sort_by(|&a, &b| {
            via[a]
                .0
                .cmp(&via[b].0)
                .then_with(|| key_cmp(key(a), key(b)))
        });

        // The classifying side produces, a consumer thread turns each outcome into a row, a
        // record, a diagnostic and a tally entry. They meet in a bounded queue, so a slow consumer
        // makes the producer wait instead of letting outcomes pile up.
        // The wall lines and the room names the regions step reads, in canonical order.
        let rules = regions.is_some().then(|| {
            element_rules
                .clone()
                .unwrap_or_else(ElementRules::default_profile)
        });
        let mut walls: Vec<WallEdge> = Vec::new();
        // Every straight line no rule calls a column, for a drawing whose layers name nothing (#1610).
        // Used only when the rules find no wall at all; capped at what the step would refuse anyway.
        let mut lines: Vec<WallEdge> = Vec::new();
        let mut labels: Vec<RoomLabel> = Vec::new();
        // The texts and hatches carried in streams (#1609): the entity a text hangs on, the entities
        // a hatch lists, and how many of each class were mapped or not.
        let annotate = (*want_texts || *want_fills) && components.line_style;
        let mut texts: Vec<(EntityId, TextBody)> = Vec::new();
        let mut fills: Vec<(Vec<EntityId>, FillBody)> = Vec::new();
        let mut annotation_counts: BTreeMap<String, u64> = BTreeMap::new();
        if dimensions_expanded > 0 {
            annotation_counts.insert("DIMENSION.expanded".into(), dimensions_expanded);
        }
        // The IMAGEs placed as rasters (#1667), in the order they were met.
        let mut rasters: Vec<crate::raster_pvec::pyramid::RasterPyramid> = Vec::new();
        let mut placed_pixels = 0u64;
        let (tx, rx) = bounded::<Outcome>(*queue_capacity);
        let sent = AtomicU64::new(0);
        /// The pieces of a line or straight polyline inside the sheet `b`, `[min x, min y, max x,
        /// max y]`, as point lists; `None` for a curve that is not cut (P-7). Cut on x, then on y,
        /// Liang-Barsky per segment: a crossing point is computed once, from its segment's own
        /// ends, and lies exactly on the edge it was cut on. A closed polyline's pieces are joined
        /// back across the point where it closed.
        fn sheet_cut(curve: &Curve, b: [f64; 4]) -> Option<Vec<Vec<[f64; 2]>>> {
            let points: Vec<[f64; 2]> = match curve {
                Curve::Polyline(p) => p.vertices().iter().map(|v| [v.x, v.y]).collect(),
                Curve::Trimmed(t) if matches!(t.basis(), Curve::Line(_)) => {
                    let (a, z) = (t.eval(0.0).ok()?, t.eval(t.trim().span).ok()?);
                    vec![[a.x, a.y], [z.x, z.y]]
                }
                _ => return None,
            };
            let mut pieces = vec![points.clone()];
            for (axis, lo, hi) in [(0, b[0], b[2]), (1, b[1], b[3])] {
                let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
                for piece in &pieces {
                    let mut cur: Vec<[f64; 2]> = Vec::new();
                    for w in piece.windows(2) {
                        let (a, z) = (w[0], w[1]);
                        let (da, dz) = (a[axis], z[axis]);
                        // The part inside, as parameters and the edge each end was cut on.
                        let (mut t0, mut e0, mut t1, mut e1) = (0.0f64, None, 1.0f64, None);
                        if da == dz {
                            if da < lo || da > hi {
                                if cur.len() >= 2 {
                                    out.push(std::mem::take(&mut cur));
                                }
                                cur.clear();
                                continue;
                            }
                        } else {
                            let (ta, tb) = ((lo - da) / (dz - da), (hi - da) / (dz - da));
                            let ((enter, ee), (leave, le)) = if ta < tb {
                                ((ta, lo), (tb, hi))
                            } else {
                                ((tb, hi), (ta, lo))
                            };
                            if enter > t0 {
                                (t0, e0) = (enter, Some(ee));
                            }
                            if leave < t1 {
                                (t1, e1) = (leave, Some(le));
                            }
                            if t0 >= t1 {
                                if cur.len() >= 2 {
                                    out.push(std::mem::take(&mut cur));
                                }
                                cur.clear();
                                continue;
                            }
                        }
                        let at = |t: f64, edge: Option<f64>, end: [f64; 2]| match edge {
                            None => end,
                            Some(e) => {
                                let mut q = [a[0] + (z[0] - a[0]) * t, a[1] + (z[1] - a[1]) * t];
                                q[axis] = e;
                                q
                            }
                        };
                        let (p0, p1) = (at(t0, e0, a), at(t1, e1, z));
                        if cur.last() != Some(&p0) {
                            if cur.len() >= 2 {
                                out.push(std::mem::take(&mut cur));
                            }
                            cur.clear();
                            cur.push(p0);
                        }
                        cur.push(p1);
                        if e1.is_some() {
                            out.push(std::mem::take(&mut cur));
                        }
                    }
                    if cur.len() >= 2 {
                        out.push(cur);
                    }
                }
                pieces = out;
            }
            let closed = points.len() > 2 && points.first() == points.last();
            if closed
                && pieces.len() > 1
                && pieces[0].first() == points.first()
                && pieces[pieces.len() - 1].last() == points.last()
            {
                let first = pieces.remove(0);
                let last = pieces.len() - 1;
                pieces[last].extend_from_slice(&first[1..]);
                let joined = pieces.pop().expect("one piece at least");
                pieces.insert(0, joined);
            }
            Some(pieces)
        }
        // Plot area Extents places the paper from the extents of what the sheet draws: the paper's
        // own kept geometry and the rectangles of its drawn viewports. Centre and fit to paper work
        // on those extents too (#1628). The frame was the layout's own until now.
        if let Some(sh) = sheet
            .as_mut()
            .filter(|_| plot_settings.area == PlotArea::Extents)
            && let Some(pl) = sh.place
        {
            let mut ext: Option<[f64; 4]> = None;
            let mut grow = |b: [f64; 4]| {
                ext = Some(ext.map_or(b, |e| {
                    [
                        e[0].min(b[0]),
                        e[1].min(b[1]),
                        e[2].max(b[2]),
                        e[3].max(b[3]),
                    ]
                }));
            };
            for (at, item) in items.iter().enumerate() {
                let Item::Leaf(f) = item else { continue };
                if via[at].1.is_some() || f.entity.provenance.class_name == "VIEWPORT" {
                    continue;
                }
                if let Ok(done) = classify_item(item, &f.entity, gate, None)
                    && let Some(b) = done.geometry.source_bounds()
                {
                    let (lo, hi) = (b.min(), b.max());
                    grow([lo.x, lo.y, hi.x, hi.y]);
                }
            }
            for (_, w) in &windows {
                if let Window::Drawn { rect, .. } = w {
                    grow(*rect);
                }
            }
            sh.summary.notes.push("plot-area-extents".into());
            if let Some(e) = ext {
                // The extents are drawing units; the paper they are placed on is not (#1665).
                let e = e.map(|v| v * sh.per_unit);
                let [ml, mb, mr, mt] = pl.margin;
                let (printable_w, printable_h) = (pl.size[0] - ml - mr, pl.size[1] - mb - mt);
                let (ew, eh) = (e[2] - e[0], e[3] - e[1]);
                // Drawing units to a paper unit: the reciprocal of the fit ratio, one division.
                let mut k = 1.0;
                if plot_settings.fit_to_paper {
                    let by_width = (printable_w / ew, printable_w, ew);
                    let by_height = (printable_h / eh, printable_h, eh);
                    let tight = [by_width, by_height]
                        .into_iter()
                        .filter(|r| r.0.is_finite() && r.0 > 0.0)
                        .min_by(|a, b| a.0.total_cmp(&b.0));
                    match tight {
                        Some((_, n, d)) => {
                            k = d / n;
                            sh.summary.notes.push("plot-fit-to-paper".into());
                            sh.summary.notes.push(format!("plot-fit-ratio 1:{k}"));
                        }
                        None => sh.summary.notes.push("plot-fit-skipped".into()),
                    }
                }
                let (x0, y0) = if plot_settings.center {
                    sh.summary.notes.push("plot-centered".into());
                    (
                        (e[0] + e[2]) / 2.0 - printable_w * k / 2.0 - ml * k,
                        (e[1] + e[3]) / 2.0 - printable_h * k / 2.0 - mb * k,
                    )
                } else {
                    (e[0] - pl.corner[0] * k, e[1] - pl.corner[1] * k)
                };
                let b = [x0, y0, x0 + pl.size[0] * k, y0 + pl.size[1] * k];
                sh.frame = Some(b);
                sh.summary.sheet_box = Some(b.map(|v| format!("{v}")));
            } else {
                sh.summary.notes.push("plot-extents-empty".into());
            }
        }
        let frame = sheet.as_ref().and_then(|s| s.frame);
        // The sheet's own linework goes to the paper: drawn at the plot scale (#1665), then,
        // plotting upside-down, turned 180 degrees about the frame's centre, which leaves the frame
        // where it is. The windows go with it, so what is drawn through them and the rectangle it
        // is cut to stay together.
        let sheet_map = sheet.as_ref().and_then(|s| {
            let k = s.per_unit;
            match s.frame.filter(|_| s.flip) {
                Some(b) => Some([-k, 0.0, 0.0, -k, b[0] + b[2], b[1] + b[3]]),
                None => (k != 1.0).then_some([k, 0.0, 0.0, k, 0.0, 0.0]),
            }
        });
        if let Some(m) = sheet_map {
            let [p, q, r, s, u, v] = m;
            let at = |pt: [f64; 2]| [p * pt[0] + q * pt[1] + u, r * pt[0] + s * pt[1] + v];
            for (_, w) in &mut windows {
                if let Window::Drawn { map, rect, ring } = w {
                    let [a, b, c, d, tx, ty] = *map;
                    *map = [
                        p * a + q * c,
                        p * b + q * d,
                        r * a + s * c,
                        r * b + s * d,
                        p * tx + q * ty + u,
                        r * tx + s * ty + v,
                    ];
                    let (lo, hi) = (at([rect[0], rect[1]]), at([rect[2], rect[3]]));
                    *rect = [
                        lo[0].min(hi[0]),
                        lo[1].min(hi[1]),
                        lo[0].max(hi[0]),
                        lo[1].max(hi[1]),
                    ];
                    if let Some(points) = ring {
                        for pt in points.iter_mut() {
                            *pt = at(*pt);
                        }
                    }
                }
            }
        }
        let drop_widths = sheet.as_ref().is_some_and(|s| s.drop_widths);
        // A pattern's lengths are drawing units like the coordinates, so the sheet's own strokes
        // get the plot scale on them too (#1683), and the header's LTSCALE multiplies every
        // pattern (#1685). A leaf drawn through a window is the same with PSLTSCALE set; clear, its
        // pattern is in model units and the window's own map scales it. AutoCAD takes neither
        // below zero, so a value that is not a positive number reads as 1.
        let dash_k = sheet.as_ref().map_or(1.0, |s| s.per_unit);
        let ltscale = Some(provider.global_linetype_scale())
            .filter(|v| v.is_finite() && *v > 0.0)
            .unwrap_or(1.0);
        let paper_dashes = provider.paper_space_linetype_scaling();
        // The plot style table a CTB gives, applied per ACI (#1630); an STB applies nothing.
        let table = plot_style.as_ref().filter(|t| !t.is_stb);
        // What the sheet clip did: outside, crossing, cut and the pieces the cut ones became.
        let mut clipped = [0u64; 4];
        // Ids are dense over what is written to the report, so a cut entity's pieces each take one.
        let mut next_id = 0u64;
        // Rows drawn through each window, and the VIEWPORT entities the provider said nothing of.
        let mut through = vec![0u64; windows.len()];
        let mut unreported: Vec<u64> = Vec::new();
        let consumed = std::thread::scope(|scope| {
            let consumer = scope.spawn(|| consume(rx, &sent, observer));
            let mut failure = None;
            for (n, &at) in perm.iter().enumerate() {
                if n.is_multiple_of(BATCH) && cancel.is_cancelled() {
                    failure = Some(PvecConvertError::Cancelled);
                    break;
                }
                let window = via[at].1;
                // A model leaf drawn through a viewport gets the viewport's map composed onto its
                // own here, the moment it is classified (#1563).
                let lifted_leaf;
                let lifted;
                let item: &Item<'_> = match (window.map(|k| &windows[k].1), &items[at]) {
                    (Some(Window::Drawn { map, .. }), Item::Leaf(f)) => {
                        lifted_leaf = f.through(*map);
                        lifted = Item::Leaf(&lifted_leaf);
                        &lifted
                    }
                    (None, Item::Leaf(f)) if sheet_map.is_some() => {
                        lifted_leaf = f.through(sheet_map.expect("checked in the guard"));
                        lifted = Item::Leaf(&lifted_leaf);
                        &lifted
                    }
                    _ => &items[at],
                };
                // A leaf is classified where it sits; an excluded or skipped one has no entity of
                // its own, so its stand-in is made for the moment it is classified.
                let stand_in;
                let raw: &RawEntity = match item {
                    Item::Leaf(f) => &f.entity,
                    Item::Unplaced(u) => &u.entity,
                    Item::Excluded(x) => {
                        stand_in = stub(
                            x.handle.map(|h| h.get()),
                            &x.class_name,
                            x.block_path.clone(),
                            &x.layer,
                            x.reason.code(),
                        );
                        &stand_in
                    }
                    Item::Skipped(s) => {
                        stand_in = stub(
                            s.insert_handle.map(|h| h.get()),
                            "INSERT",
                            s.block_path.clone(),
                            "",
                            s.reason.code(),
                        );
                        &stand_in
                    }
                };
                let mut done =
                    match classify_item(item, raw, gate, window.map(|k| windows[k].0.handle)) {
                        Ok(done) => done,
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                    };
                // A VIEWPORT on the sheet is a window, not geometry of its own (#1563): its row
                // says what the window did, and one the print profile cannot map is refused with
                // 508. A viewport entity the provider reports nothing about cannot be placed.
                if sheet.is_some()
                    && window.is_none()
                    && matches!(item, Item::Leaf(_))
                    && raw.provenance.class_name == "VIEWPORT"
                {
                    let decided = raw
                        .provenance
                        .handle
                        .and_then(|h| windows.iter().find(|(v, _)| v.handle == h));
                    let space = |why: &str| {
                        without_geometry(
                            Classification::ignored_by_policy("space", "VIEWPORT"),
                            vec![format!("VIEWPORT: {why}")],
                        )
                    };
                    done = match decided {
                        None => {
                            if let Some(h) = raw.provenance.handle {
                                unreported.push(h.get());
                            }
                            without_geometry(
                                Classification::refused(ReasonCode::ViewNotEnabled, "VIEWPORT"),
                                vec![
                                    "VIEW_NOT_ENABLED (508): the provider reports nothing about this viewport, so it cannot be placed"
                                        .to_owned(),
                                ],
                            )
                        }
                        // The paper view keeps its row, under its own code.
                        Some((_, Window::Paper)) => without_geometry(
                            Classification::ignored_by_policy(
                                PolicyReason::PaperView.field(),
                                "VIEWPORT",
                            ),
                            vec![format!(
                                "{}: the sheet's own paper view, not a window onto the model",
                                PolicyReason::PaperView.code()
                            )],
                        ),
                        Some((_, Window::Dropped(why))) => {
                            space(&format!("contributes nothing to a print ({why})"))
                        }
                        Some((_, Window::Drawn { .. })) => space("model space is drawn through it"),
                        Some((_, Window::Refused(why))) => without_geometry(
                            Classification::refused(ReasonCode::ViewNotEnabled, "VIEWPORT"),
                            vec![format!(
                                "VIEW_NOT_ENABLED (508): a viewport this profile cannot map ({why})"
                            )],
                        ),
                    };
                }
                // P-3: a layer frozen in a viewport is dropped inside it, and drawn everywhere
                // else.
                let mut frozen_here = false;
                if let (Some(k), Item::Leaf(f)) = (window, item) {
                    let v = &windows[k].0;
                    let cold = |layer: &str| {
                        v.frozen_layers
                            .iter()
                            .any(|l| l.eq_ignore_ascii_case(layer))
                    };
                    let by_override = |layer: &str| {
                        window_hidden[k]
                            .iter()
                            .any(|l| l.eq_ignore_ascii_case(layer))
                    };
                    let mut overridden = false;
                    // The leaf's own effective layer, or that of any INSERT it is reached
                    // through: an INSERT on layer 0 takes its parent's (V1-2).
                    let mut parent = "0";
                    let mut through_cold = false;
                    for step in f.entity.provenance.block_path.steps() {
                        if let Some(own) = insert_layers.get(&step.insert_handle.get()) {
                            if !own.eq_ignore_ascii_case("0") {
                                parent = own;
                            }
                            through_cold |= cold(parent);
                            overridden |= cold(parent) && by_override(parent);
                        }
                    }
                    let frozen = through_cold
                        || model_layers
                            .get(&flat_key(f))
                            .is_some_and(|layer| cold(layer));
                    overridden |= model_layers
                        .get(&flat_key(f))
                        .is_some_and(|layer| cold(layer) && by_override(layer));
                    if frozen {
                        if view_profile.layers.frozen_in_viewport == ViewportFreezeRule::Apply {
                            frozen_here = true;
                            done = without_geometry(
                                Classification::ignored_by_policy(
                                    if overridden {
                                        PolicyReason::LayerOverridden.field()
                                    } else {
                                        "layers.frozen_in_viewport"
                                    },
                                    raw.provenance.class_name.as_str(),
                                ),
                                vec![format!(
                                    "{}: its layer is {} in viewport {:X}",
                                    if overridden {
                                        PolicyReason::LayerOverridden.code()
                                    } else {
                                        PolicyReason::ViewportFrozen.code()
                                    },
                                    if overridden { "hidden" } else { "frozen" },
                                    v.handle.get()
                                )],
                            );
                        } else {
                            done.warnings.push(format!(
                                "{}: its layer is frozen in viewport {:X}, drawn anyway",
                                PolicyReason::ViewportFrozen.code(),
                                v.handle.get()
                            ));
                        }
                    }
                }
                // What it is cut to: the sheet, or the part of a viewport's rectangle on it, and
                // the viewport's polygon when it is clipped to one (#1565).
                let (clip_box, ring, edge) = match window.map(|k| &windows[k].1) {
                    Some(Window::Drawn { rect, ring, .. }) => (
                        Some(frame.map_or(*rect, |b| {
                            [
                                rect[0].max(b[0]),
                                rect[1].max(b[1]),
                                rect[2].min(b[2]),
                                rect[3].min(b[3]),
                            ]
                        })),
                        ring.as_ref(),
                        "viewport",
                    ),
                    _ => (frame, None, "sheet"),
                };
                // P-7: nothing is drawn outside the sheet. A line or a straight polyline against a
                // rectangle is cut on x then y (#1562); any other curve, and anything against a
                // polygon, is split by `curve_clip` (#1565): conics as arcs of their own carrier,
                // splines as sub-ranges of themselves. Each piece is its own entity and row. PVec's
                // own box decides wholly outside, and wholly inside a rectangle.
                let mut more: Vec<Done> = Vec::new();
                let mut whole = window.is_none();
                // Whether the entity went through the clip untouched, so a side of it still has its
                // own length (#1694).
                let mut uncut = true;
                if let (Some(b), Some(bounds), Some(curve), false) = (
                    clip_box,
                    done.geometry.source_bounds(),
                    done.geometry.geometry(),
                    frozen_here,
                ) {
                    let (lo, hi) = (bounds.min(), bounds.max());
                    let class = raw.provenance.class_name.as_str();
                    let outside = |why: &str| {
                        without_geometry(
                            Classification::ignored_by_policy("viewport_clipping", class),
                            vec![format!("{}: {why}", PolicyReason::OutsideViewport.code())],
                        )
                    };
                    let boxed = lo.x >= b[0] && hi.x <= b[2] && lo.y >= b[1] && hi.y <= b[3];
                    // `None`: kept whole. `Some(Ok((pieces, note)))`: cut, with a note for each
                    // piece's row. `Some(Err((done, counter)))`: dropped or refused as `done` says,
                    // counted under `clipped[counter]` (0 outside, 1 crossing and refused), or
                    // under neither when a piece, not the clip, was the problem.
                    type Pieces = (Vec<Normalized2DGeometry>, Option<String>);
                    let mut cut: Option<Result<Pieces, (Done, Option<usize>)>> = None;
                    if b[0] > b[2]
                        || b[1] > b[3]
                        || hi.x < b[0]
                        || lo.x > b[2]
                        || hi.y < b[1]
                        || lo.y > b[3]
                    {
                        cut = Some(Err((
                            outside(&format!("wholly outside the {edge}")),
                            Some(0),
                        )));
                    } else if ring.is_some() || !boxed {
                        let fidelity = done.geometry.fidelity();
                        let degenerate = |e: String| {
                            without_geometry(
                                Classification::refused(ReasonCode::DegenerateGeometry, class),
                                vec![format!(
                                    "{class}: a piece cut at the {edge} edge is degenerate: {e}"
                                )],
                            )
                        };
                        match (ring, ring.is_none().then(|| sheet_cut(curve, b)).flatten()) {
                            (None, Some(pieces)) if pieces.is_empty() => {
                                cut = Some(Err((
                                    outside(&format!("wholly outside the {edge}")),
                                    Some(0),
                                )));
                            }
                            (None, Some(pieces)) => {
                                let line = matches!(curve, Curve::Trimmed(_));
                                let built: Result<Vec<Normalized2DGeometry>, String> = pieces
                                    .iter()
                                    .map(|p| {
                                        if line {
                                            let a = Point2::checked(p[0][0], p[0][1])
                                                .map_err(|e| e.to_string())?;
                                            let z = Point2::checked(p[1][0], p[1][1])
                                                .map_err(|e| e.to_string())?;
                                            let l = Line::new(a, a.vector_to(z))
                                                .map_err(|e| e.to_string())?;
                                            let t = Trimmed::new(
                                                Curve::Line(l),
                                                0.0,
                                                a.distance(z),
                                                true,
                                            )
                                            .map_err(|e| e.to_string())?;
                                            Normalized2DGeometry::from_curve(
                                                Curve::Trimmed(t),
                                                fidelity,
                                            )
                                            .map_err(|e| e.to_string())
                                        } else {
                                            Normalized2DGeometry::polyline(p, fidelity)
                                                .map_err(|e| e.to_string())
                                        }
                                    })
                                    .collect();
                                cut = Some(
                                    built.map(|p| (p, None)).map_err(|e| (degenerate(e), None)),
                                );
                            }
                            _ => {
                                // The polygon, then the sheet; or the rectangle alone.
                                let as_ring = |r: [f64; 4]| {
                                    vec![[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]
                                };
                                let rings: Vec<Vec<[f64; 2]>> = match ring {
                                    Some(r) => std::iter::once(r.clone())
                                        .chain(frame.map(as_ring))
                                        .collect(),
                                    None => vec![as_ring(b)],
                                };
                                let mut current = vec![curve.clone()];
                                let mut split = false;
                                let mut tolerance: Option<f64> = None;
                                let mut refused: Option<(Done, Option<usize>)> = None;
                                'rings: for r in &rings {
                                    let points: Result<Vec<Point2>, _> =
                                        r.iter().map(|q| Point2::checked(q[0], q[1])).collect();
                                    let points = match points {
                                        Ok(p) => p,
                                        Err(e) => {
                                            refused = Some((degenerate(e.to_string()), None));
                                            break;
                                        }
                                    };
                                    let mut next = Vec::with_capacity(current.len());
                                    for c in std::mem::take(&mut current) {
                                        match clip_curve(&c, &points) {
                                            Ok(CurveClip::Inside) => next.push(c),
                                            Ok(CurveClip::Outside) => split = true,
                                            Ok(CurveClip::Split {
                                                pieces,
                                                fidelity,
                                                parameter_tolerance,
                                            }) => {
                                                split = true;
                                                if fidelity == FidelityClass::SupportedEquivalent {
                                                    tolerance = Some(
                                                        tolerance
                                                            .unwrap_or(0.0)
                                                            .max(parameter_tolerance),
                                                    );
                                                }
                                                next.extend(pieces);
                                            }
                                            Err(CurveClipError::Unsupported(kind)) => {
                                                // Not cut exactly, but it may not need cutting: its
                                                // own box (PVec's, enclosing it) as a closed outline
                                                // tells wholly inside and wholly outside apart from
                                                // crossing.
                                                let bx = c.bounds().ok();
                                                let outline = bx.and_then(|bx| {
                                                    let (l, h) = (bx.min(), bx.max());
                                                    let corners = [
                                                        Point2::new(l.x, l.y),
                                                        Point2::new(h.x, l.y),
                                                        Point2::new(h.x, h.y),
                                                        Point2::new(l.x, h.y),
                                                        Point2::new(l.x, l.y),
                                                    ];
                                                    Polyline::new(corners.to_vec())
                                                        .ok()
                                                        .map(Curve::Polyline)
                                                });
                                                match outline.map(|o| clip_curve(&o, &points)) {
                                                    Some(Ok(CurveClip::Inside)) => {
                                                        next.push(c);
                                                        continue;
                                                    }
                                                    // Outside also when the whole ring sits inside
                                                    // the box, and then the curve may run through
                                                    // it: only a ring with no vertex in the box is
                                                    // clear of the curve.
                                                    Some(Ok(CurveClip::Outside))
                                                        if bx.is_some_and(|bx| {
                                                            let (l, h) = (bx.min(), bx.max());
                                                            !points.iter().any(|q| {
                                                                q.x >= l.x
                                                                    && q.x <= h.x
                                                                    && q.y >= l.y
                                                                    && q.y <= h.y
                                                            })
                                                        }) =>
                                                    {
                                                        split = true;
                                                        continue;
                                                    }
                                                    _ => {}
                                                }
                                                let what = if kind == CurveKind::RationalBSpline {
                                                    "a rational spline".to_owned()
                                                } else {
                                                    format!("a {kind:?}")
                                                };
                                                refused = Some((
                                                    without_geometry(
                                                        Classification::refused(
                                                            ReasonCode::ViewNotEnabled,
                                                            class,
                                                        ),
                                                        vec![format!(
                                                            "VIEW_NOT_ENABLED (508): the curve crosses the {edge} edge, and {what} is not cut exactly yet"
                                                        )],
                                                    ),
                                                    Some(1),
                                                ));
                                                break 'rings;
                                            }
                                            Err(e) => {
                                                refused = Some((degenerate(e.to_string()), None));
                                                break 'rings;
                                            }
                                        }
                                    }
                                    current = next;
                                }
                                cut = match refused {
                                    Some(d) => Some(Err(d)),
                                    None if !split => None,
                                    None if current.is_empty() => Some(Err((
                                        outside(&format!("outside the {edge}")),
                                        Some(0),
                                    ))),
                                    None => {
                                        // A spline piece is the same locus up to its parameter
                                        // tolerance: SupportedEquivalent, and the row says how far.
                                        let fidelity = if tolerance.is_some() {
                                            fidelity.max(FidelityClass::SupportedEquivalent)
                                        } else {
                                            fidelity
                                        };
                                        let note = tolerance.map(|t| {
                                            format!(
                                                "{class}: cut at the {edge} edge by knot insertion, parameter tolerance {t:e}"
                                            )
                                        });
                                        let built: Result<Vec<Normalized2DGeometry>, String> =
                                            current
                                                .into_iter()
                                                .map(|c| {
                                                    Normalized2DGeometry::from_curve(c, fidelity)
                                                        .map_err(|e| e.to_string())
                                                })
                                                .collect();
                                        Some(
                                            built
                                                .map(|p| (p, note))
                                                .map_err(|e| (degenerate(e), None)),
                                        )
                                    }
                                };
                            }
                        }
                    }
                    match cut {
                        None => {}
                        Some(Err((d, counter))) => {
                            whole = false;
                            uncut = false;
                            if let Some(k) = counter {
                                clipped[k] += 1;
                            }
                            done = d;
                        }
                        Some(Ok((records, note))) => {
                            whole = false;
                            uncut = false;
                            clipped[2] += 1;
                            clipped[3] += records.len() as u64;
                            let mut gated_pieces = Vec::with_capacity(records.len());
                            for r in records {
                                match gated(r, &raw.provenance, gate) {
                                    Ok(mut d) => {
                                        d.warnings.extend(note.clone());
                                        gated_pieces.push(d);
                                    }
                                    Err(e) => {
                                        failure = Some(e);
                                        break;
                                    }
                                }
                            }
                            if failure.is_some() {
                                break;
                            }
                            let mut it = gated_pieces.into_iter();
                            if let Some(first) = it.next() {
                                done = first;
                            }
                            more = it.collect();
                        }
                    }
                }
                // The ACI the stroke is drawn in, through BYLAYER and BYBLOCK; a true colour has none.
                let effective_aci = match paints.get(&leaf_key_of(&raw.provenance)) {
                    Some(LeafPaint::Colour(SourceColour::Aci(i))) if table.is_some() => Some(*i),
                    _ => None,
                };
                // The entity's style input as the drawing gives it, with the linetype scale taken to the
                // output and the table's lineweight and linetype for the stroke's ACI laid over it.
                let base_input = || {
                    let mut input = dwg_style_input(&raw.style, true);
                    input.linetype_scale *= ltscale
                        * match window.map(|k| &windows[k].1) {
                            Some(Window::Drawn { map, .. }) if !paper_dashes => {
                                map[0].hypot(map[1])
                            }
                            _ => dash_k,
                        };
                    // The table's lineweight and linetype for the stroke's ACI beat the
                    // object's; an entry that says "object" leaves them (#1630).
                    if let (Some(t), Some(aci)) = (table, effective_aci) {
                        // A style's lineweight is one more than the list's index: the editor's
                        // list is "use the object's" and then the table from its first entry
                        // (#1692, acadrust#73), so 0 and 255 leave the object's and N is entry N - 1.
                        let style = t.aci_entries[usize::from(aci)].lineweight;
                        let entry = usize::from(style).checked_sub(1);
                        if let Some(mm) = entry
                            .filter(|_| style != 255)
                            .and_then(|i| t.lineweights.get(i))
                        {
                            let mm = if t.apply_factor {
                                mm * t.scale_factor.max(0.0)
                            } else {
                                *mm
                            };
                            input.lineweight = Some(DwgLineweight::Hundredths(
                                (f64::from(mm) * 100.0).round() as i16,
                            ));
                        }
                        // Pattern lengths are in millimetres like the table's lineweights,
                        // `linepattern_size` long: 0 solid, 1 dashed, 2 dotted, 3 dash dot.
                        let size = f64::from(t.aci_entries[usize::from(aci)].linepattern_size);
                        let pattern = match t.aci_entries[usize::from(aci)].linetype {
                            0 => Some(Vec::new()),
                            1 => Some(vec![size, -size / 2.0]),
                            2 => Some(vec![0.0, -size / 2.0]),
                            3 => Some(vec![size, -size / 4.0, 0.0, -size / 4.0]),
                            _ => None,
                        };
                        if let Some(p) = pattern {
                            input.linetype_scale = 1.0;
                            input.linetype_name = None;
                            input.linetype_elements = Some(p).filter(|p| !p.is_empty());
                        }
                    }
                    input
                };
                // PLINEGEN off (#1691): a polyline's pattern starts anew on every side, so each side
                // is a record of its own with the entity's style and provenance, the way a piece cut
                // at an edge is. Only a polyline that has a pattern is split. A solid one, or one
                // with PLINEGEN on, stays whole, and the flag is the entity's own: the header's
                // PLINEGEN only seeds the polylines drawn after it.
                if components.line_style
                    && !frozen_here
                    && matches!(raw.geometry, RawGeometry::Polyline { .. })
                    && matches!(
                        raw.provenance.class_name.as_str(),
                        "LWPOLYLINE" | "POLYLINE"
                    )
                    && (table.is_some()
                        || dwg_style_input(&raw.style, true)
                            .linetype_elements
                            .is_some())
                    && raw
                        .provenance
                        .handle
                        .is_some_and(|h| !provider.polyline_linetype_generation(h))
                {
                    // Each side of an uncut polyline also has its pattern centred (#1699). The pattern as it
                    // is drawn, every element in the output's unit and on the alternating slots a dash
                    // pattern takes (a gap first gets a zero dash before it), the way `dwg_line_style`
                    // writes it. Only a pattern of whole dash and gap pairs that starts on a dash and whose
                    // last dash is not a dot is centred.
                    let centre: Option<Vec<f64>> = if uncut {
                        let input = base_input();
                        let mut pattern: Vec<f64> = Vec::new();
                        for e in input.linetype_elements.iter().flatten() {
                            let pen_down = *e >= 0.0 || e.is_nan();
                            if pen_down != pattern.len().is_multiple_of(2) {
                                pattern.push(0.0);
                            }
                            pattern.push(e.abs() * input.linetype_scale);
                        }
                        let n = pattern.len();
                        let period: f64 = pattern.iter().sum();
                        (n >= 2
                            && n.is_multiple_of(2)
                            && pattern[0] > 0.0
                            && pattern[n - 2] > 0.0
                            && period.is_finite()
                            && period > 0.0)
                            .then_some(pattern)
                    } else {
                        None
                    };
                    // A side of two points whose pattern fits n >= 1 times is written as three records that
                    // join end to end (#1699). PVec's line style is one cyclic dash array and a phase per
                    // record, and a phase can start inside a dash but not lengthen one, so the first and
                    // the last dash, each d + r / 2 long, cannot be one cyclic pattern. With the pattern's
                    // extent e = n p - g, its period p and its last gap g, and r = L - e: a solid lead of
                    // r / 2, the pattern at its own period over e, and a solid tail of r / 2.
                    let centred = |piece: Done| -> Result<Vec<Done>, PvecConvertError> {
                        let Some(pattern) = &centre else {
                            return Ok(vec![piece]);
                        };
                        let Some(Curve::Polyline(pl)) = piece.geometry.geometry() else {
                            return Ok(vec![piece]);
                        };
                        let [a, b] = pl.vertices() else {
                            return Ok(vec![piece]);
                        };
                        let side = (b.x - a.x).hypot(b.y - a.y);
                        let period: f64 = pattern.iter().sum();
                        let gap = pattern[pattern.len() - 1];
                        let periods = ((side + gap) / period + 1e-9).floor();
                        let half = (side - (periods * period - gap)) / 2.0;
                        if !(side.is_finite() && periods >= 1.0 && half > 1e-9 * side.max(1.0)) {
                            return Ok(vec![piece]);
                        }
                        let t = half / side;
                        let at = |t: f64| {
                            let mut q = *a;
                            q.x = a.x + (b.x - a.x) * t;
                            q.y = a.y + (b.y - a.y) * t;
                            q
                        };
                        let fidelity = piece.geometry.fidelity();
                        let parts: Option<Vec<Normalized2DGeometry>> =
                            [(0.0, t), (t, 1.0 - t), (1.0 - t, 1.0)]
                                .into_iter()
                                .map(|(from, to)| {
                                    let line = Polyline::new(vec![at(from), at(to)]).ok()?;
                                    Normalized2DGeometry::from_curve(
                                        Curve::Polyline(line),
                                        fidelity,
                                    )
                                    .ok()
                                })
                                .collect();
                        let Some(parts) = parts else {
                            return Ok(vec![piece]);
                        };
                        let mut out = Vec::with_capacity(3);
                        for (k, part) in parts.into_iter().enumerate() {
                            let mut one = gated(part, &raw.provenance, gate)?;
                            one.solid = k != 1;
                            if k == 0 {
                                one.warnings.extend(piece.warnings.clone());
                            }
                            out.push(one);
                        }
                        Ok(out)
                    };
                    let mut restarted: Vec<Done> = Vec::new();
                    for d in std::iter::once(done).chain(std::mem::take(&mut more)) {
                        let sides: Vec<Curve> = match d.geometry.geometry() {
                            Some(Curve::Polyline(p)) => p
                                .vertices()
                                .windows(2)
                                .filter_map(|w| Polyline::new(w.to_vec()).ok().map(Curve::Polyline))
                                .collect(),
                            Some(Curve::Composite(c)) => c
                                .segments()
                                .iter()
                                .flat_map(|s| match s {
                                    Curve::Polyline(p) => p
                                        .vertices()
                                        .windows(2)
                                        .filter_map(|w| {
                                            Polyline::new(w.to_vec()).ok().map(Curve::Polyline)
                                        })
                                        .collect(),
                                    other => vec![other.clone()],
                                })
                                .collect(),
                            _ => Vec::new(),
                        };
                        let fidelity = d.geometry.fidelity();
                        let built: Result<Vec<Normalized2DGeometry>, _> = sides
                            .into_iter()
                            .map(|c| Normalized2DGeometry::from_curve(c, fidelity))
                            .collect();
                        match built {
                            Ok(parts) if parts.len() > 1 => {
                                for (k, part) in parts.into_iter().enumerate() {
                                    match gated(part, &raw.provenance, gate) {
                                        Ok(mut piece) => {
                                            if k == 0 {
                                                piece.warnings.extend(d.warnings.clone());
                                            }
                                            match centred(piece) {
                                                Ok(v) => restarted.extend(v),
                                                Err(e) => {
                                                    failure = Some(e);
                                                    break;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            failure = Some(e);
                                            break;
                                        }
                                    }
                                }
                                if failure.is_some() {
                                    break;
                                }
                            }
                            _ => match centred(d) {
                                Ok(v) => restarted.extend(v),
                                Err(e) => {
                                    failure = Some(e);
                                    break;
                                }
                            },
                        }
                    }
                    if failure.is_some() {
                        break;
                    }
                    let mut it = restarted.into_iter();
                    done = it.next().expect("a polyline leaves at least one record");
                    more = it.collect();
                }
                // A text, a hatch or a SOLID goes in a stream when the provider can say what it
                // draws, and is counted as unmapped otherwise (#1609). Its carriers are written
                // hidden. On a sheet or through a viewport a text is kept line by line when its
                // start is inside what it is cut to, and a fill is not carried at all.
                let mut pending: Option<Body> = None;
                // Texts and fills are two choices (#1624): a class whose choice is off stays
                // unmapped and says so, so a caller that asked for one never gets the other. A
                // MULTILEADER's or TOLERANCE's text is a text too, and so is a DIMENSION's when it
                // reaches here, which is only when its block could not be followed (#1658).
                let class_name = raw.provenance.class_name.as_str();
                let dimension = is_dimension_class(class_name);
                let (text_class, fill_class) = (
                    dimension
                        || matches!(
                            class_name,
                            "TEXT" | "MTEXT" | "ATTDEF" | "MULTILEADER" | "TOLERANCE"
                        ),
                    matches!(class_name, "HATCH" | "SOLID"),
                );
                if annotate
                    && matches!(item, Item::Leaf(_))
                    && matches!(raw.geometry, RawGeometry::Unmapped { .. })
                    && ((text_class && !*want_texts) || (fill_class && !*want_fills))
                {
                    let (class, choice) = (
                        raw.provenance.class_name.as_str(),
                        if text_class { "texts" } else { "fills" },
                    );
                    done.warnings
                        .push(format!("{class}: not carried, {choice} are off"));
                }
                if annotate
                    && !frozen_here
                    && let Item::Leaf(f) = item
                    && matches!(raw.geometry, RawGeometry::Unmapped { .. })
                    && ((text_class && *want_texts) || (fill_class && *want_fills))
                {
                    let class = raw.provenance.class_name.as_str();
                    let made = raw
                        .provenance
                        .handle
                        .and_then(|h| match raw.provenance.block_path.steps().last() {
                            Some(via) if class == "ATTDEF" => {
                                provider.inserted_attribute(h, via.insert_handle)
                            }
                            _ => provider.annotation(h),
                        })
                        .ok_or("it draws nothing the provider can report")
                        .and_then(|a| {
                            annotation::map(
                                f,
                                a,
                                &|p| {
                                    clip_box.is_none_or(|b| {
                                        p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3]
                                    }) && ring.is_none_or(|r| {
                                        // Even-odd crossing count over the viewport's polygon.
                                        let mut inside = false;
                                        for (i, a) in r.iter().enumerate() {
                                            let b = &r[(i + 1) % r.len()];
                                            if (a[1] > p[1]) != (b[1] > p[1])
                                                && p[0]
                                                    < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1])
                                                        + a[0]
                                            {
                                                inside = !inside;
                                            }
                                        }
                                        inside
                                    })
                                },
                                if ring.is_none() { clip_box } else { None },
                            )
                        });
                    let made_clipped = matches!(made, Err(annotation::CLIPPED));
                    let carried = match made {
                        Ok((carriers, body)) => carriers
                            .into_iter()
                            .map(|c| classify_leaf(&f.carrying(c), gate, None))
                            .collect::<Result<Vec<_>, _>>()
                            .map(|pieces| {
                                pieces
                                    .iter()
                                    .all(|d| d.geometry.geometry().is_some())
                                    .then_some((pieces, body))
                            }),
                        Err(annotation::CLIPPED) => Ok(None),
                        Err(why) => {
                            // A dimension the provider has no text for says why its block was not
                            // followed, which is the part worth knowing.
                            let why = match &raw.geometry {
                                RawGeometry::Unmapped { reason } if dimension => reason.as_str(),
                                _ => why,
                            };
                            done.warnings.push(format!("{class}: not carried, {why}"));
                            Ok(None)
                        }
                    };
                    match carried {
                        Err(e) => {
                            failure = Some(e);
                            break;
                        }
                        Ok(Some((mut pieces, body))) => {
                            // What the stream does not carry of the entity (#1658).
                            let left_out = match class {
                                _ if dimension => Some(
                                    "its block is missing, so its text is rebuilt from the measurement and nothing else is drawn",
                                ),
                                _ => None,
                            };
                            for piece in &mut pieces {
                                piece.classification =
                                    Classification::supported(Basis::Equivalent, class);
                                piece.warnings.push(format!(
                                    "{class}: carried in a PVec stream, on hidden geometry"
                                ));
                                if let Some(why) = left_out {
                                    piece.warnings.push(format!("{class}: {why}"));
                                }
                            }
                            *annotation_counts
                                .entry(format!("{class}.mapped"))
                                .or_default() += 1;
                            pending = Some(body);
                            let mut it = pieces.into_iter();
                            done = it.next().expect("a mapped annotation has a carrier");
                            more = it.collect();
                        }
                        Ok(None) if made_clipped => {
                            *annotation_counts
                                .entry(format!("{class}.clipped"))
                                .or_default() += 1;
                        }
                        Ok(None) => {
                            *annotation_counts
                                .entry(format!("{class}.unmapped"))
                                .or_default() += 1;
                        }
                    }
                }
                // An IMAGE is placed as a raster when the caller said where its file is (#1667).
                // Its row stays the unmapped one it always was, and says what became of the pixels.
                if let (Some(dir), Item::Leaf(f), false) = (images.as_deref(), item, frozen_here)
                    && raw.provenance.class_name == "IMAGE"
                    && matches!(raw.geometry, RawGeometry::Unmapped { .. })
                    && let Some(frame) = raw.provenance.handle.and_then(|h| provider.image_frame(h))
                {
                    let named = frame
                        .file_name
                        .rsplit(['\\', '/'])
                        .next()
                        .unwrap_or_default();
                    // What the conversion has left: a count and a total of pixels, so a drawing
                    // of many pictures cannot take the memory (#1667).
                    let left = dwg_image::MAX_TOTAL_PIXELS.saturating_sub(placed_pixels);
                    let placed = if rasters.len() >= dwg_image::MAX_IMAGES {
                        Err(format!(
                            "the conversion already placed {} images",
                            dwg_image::MAX_IMAGES
                        ))
                    } else {
                        dwg_image::place(
                            &frame,
                            |p| f.map_point(p),
                            dir,
                            clip_box,
                            ring.map(Vec::as_slice),
                            left,
                        )
                    };
                    match placed {
                        Ok(pyramid) => {
                            placed_pixels += u64::from(pyramid.width) * u64::from(pyramid.height);
                            rasters.push(pyramid);
                            done.warnings
                                .push(format!("IMAGE: placed as a raster from {named}"));
                        }
                        Err(why) => done
                            .warnings
                            .push(format!("IMAGE: not drawn, {why} ({named})")),
                    }
                }
                let mut carried_ids: Vec<EntityId> = Vec::new();
                let mut stop = false;
                for done in std::iter::once(done).chain(more) {
                    let mut diagnostic =
                        entity_diagnostic(item, raw, &done.classification, &layout.name);
                    if let Some(d) = &mut diagnostic {
                        d.source = Some(source_name.clone());
                    }
                    let id = EntityId::at(next_id);
                    next_id += 1;
                    if pending.is_some() {
                        carried_ids.push(id);
                    }
                    if let (Some(rules), Item::Leaf(f), true) = (&rules, item, whole) {
                        match &raw.geometry {
                            RawGeometry::Line { start, end }
                                if done.geometry.geometry().is_some() =>
                            {
                                let class = rules.classify(raw).class;
                                let edge = WallEdge {
                                    entity: id,
                                    start: f.map_point(*start),
                                    end: f.map_point(*end),
                                };
                                if class == ElementClass::Wall {
                                    walls.push(edge);
                                } else if class != ElementClass::Column && lines.len() <= MAX_EDGES
                                {
                                    lines.push(edge);
                                }
                            }
                            RawGeometry::Unmapped { .. }
                                if matches!(
                                    raw.provenance.class_name.as_str(),
                                    "TEXT" | "MTEXT"
                                ) =>
                            {
                                let anchor =
                                    raw.provenance.handle.and_then(|h| provider.text_anchor(h));
                                if let Some((text, at)) =
                                    anchor.filter(|(t, _)| !t.trim().is_empty())
                                {
                                    labels.push(RoomLabel {
                                        at: f.map_point(at),
                                        text: text.trim().to_owned(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    let side = (*components != StreamSelection::NONE
                        && done.geometry.geometry().is_some())
                    .then(|| {
                        // Counted per kept entity in `open`, once the writer has said which it kept.
                        let (style, bad_style, own_colour) = if pending.is_some() {
                            // A carrier is never drawn: the stream record is what shows.
                            let c = LineStyleComponent::new(id, None, None, false).ok();
                            let bad = c.is_none();
                            (c, bad, false)
                        } else if components.line_style {
                            let mut input = base_input();
                            // A solid stub of a centred side (#1699) is drawn without a pattern.
                            if done.solid {
                                input.linetype_elements = Some(Vec::new());
                            }
                            // P-8: a layout that does not print lineweights has none on its sheet.
                            if drop_widths {
                                input.lineweight = None;
                            }
                            let c = dwg_line_style(id, &input).ok();
                            let bad = c.is_none();
                            (c, bad, input.colour_index.is_some())
                        } else {
                            (None, false, false)
                        };
                        // A carrier is never drawn, so it has no colour either.
                        let leaf_paint = if components.stroke_paint && pending.is_none() {
                            paints.get(&leaf_key_of(&raw.provenance)).copied()
                        } else {
                            None
                        };
                        let paint = match leaf_paint {
                            Some(LeafPaint::Colour(c)) => paint_word(c).map(|word| {
                                let (Some(t), Some(aci)) = (table, effective_aci) else {
                                    return word;
                                };
                                let [r, g, b, _] = word.to_be_bytes();
                                let [r, g, b] = t.resolve_color(aci).map_or([r, g, b], |c| {
                                    c.map(|v| (v * 255.0).round().clamp(0.0, 255.0) as u8)
                                });
                                // Screening is lighter ink on white paper, so it blends toward
                                // white and the word stays opaque.
                                let s = f64::from(t.resolve_screening(aci));
                                let [r, g, b] = [r, g, b]
                                    .map(|c| (255.0 - (255.0 - f64::from(c)) * s).round() as u8);
                                u32::from_be_bytes([r, g, b, 255])
                            }),
                            _ => None,
                        };
                        Side {
                            // Only the handle, the layer and the outermost INSERT are read; the class name and
                            // the rest of the block path are in the report, so they are not kept per entity.
                            stub: ProvenanceStub {
                                handle: raw.provenance.handle,
                                class_name: String::new(),
                                block_path: raw.provenance.block_path.steps().first().map_or_else(
                                    crate::pvec::InstancePath::root,
                                    |s| {
                                        crate::pvec::InstancePath::root()
                                            .child(String::new(), s.insert_handle)
                                    },
                                ),
                                layer: match item {
                                    Item::Leaf(f) => kept.get(&flat_key(f)).cloned(),
                                    _ => None,
                                }
                                .unwrap_or_else(|| raw.provenance.layer.clone()),
                            },
                            style,
                            paint,
                            // A colour the drawing names and the walk cannot reach, or one with no
                            // RGB behind it, is counted and not painted.
                            // With the paint stream off nothing writes a colour, so every explicit colour
                            // index is ignored, counted as before (read only with the line style stream on).
                            colour_ignored: if !components.stroke_paint {
                                own_colour
                            } else {
                                match leaf_paint {
                                    Some(LeafPaint::Unresolved) => true,
                                    Some(LeafPaint::Colour(_)) => paint.is_none(),
                                    _ => false,
                                }
                            },
                            bad_style,
                            viewport: window.map(|k| windows[k].0.handle),
                        }
                    });
                    if let Some(k) = window
                        && done.geometry.geometry().is_some()
                    {
                        through[k] += 1;
                    }
                    let outcome = Outcome {
                        id,
                        handle: raw.provenance.handle.map(|h| h.get()),
                        viewport: window.map(|k| windows[k].0.handle.get()),
                        done,
                        diagnostic,
                        side,
                    };
                    if tx.send(outcome).is_err() {
                        stop = true;
                        break;
                    }
                    sent.fetch_add(1, Ordering::Release);
                }
                match pending {
                    Some(Body::Text(lines)) => texts.extend(carried_ids.into_iter().zip(lines)),
                    Some(Body::Fill(f)) => fills.push((carried_ids, f)),
                    None => {}
                }
                if stop {
                    break;
                }
            }
            drop(tx);
            let consumed = consumer.join().expect("the consumer thread does not panic");
            failure.map_or(Ok(consumed), Err)
        })?;
        let encountered = next_id;
        let sheet = sheet.map(|mut s| {
            for (key, n) in ["outside", "crossing", "cut", "pieces"]
                .into_iter()
                .zip(clipped)
            {
                s.summary.clip.insert(key.to_owned(), n);
            }
            s.summary.viewports = windows
                .iter()
                .zip(&through)
                .map(|((v, w), n)| {
                    let (status, reason) = match w {
                        Window::Paper => ("paper-view", None),
                        Window::Dropped(why) => (*why, None),
                        Window::Refused(why) => ("refused", Some((*why).to_owned())),
                        Window::Drawn { .. } => ("drawn", None),
                    };
                    let rect = match w {
                        Window::Drawn { rect, .. } => Some(rect.map(|v| format!("{v}"))),
                        _ => None,
                    };
                    SheetViewport {
                        handle: v.handle.get(),
                        id: v.id,
                        status: status.to_owned(),
                        reason,
                        scale: rect.is_some().then(|| scale_of(v)),
                        paper_box: rect,
                        entities: *n,
                        border: no_plot
                            .contains(&v.layer.to_ascii_lowercase())
                            .then(|| "no-plot-layer".to_owned()),
                    }
                })
                .chain(unreported.iter().map(|h| SheetViewport {
                    handle: *h,
                    status: "refused".into(),
                    reason: Some("not-reported".into()),
                    ..SheetViewport::default()
                }))
                .collect();
            s
        });
        let xref_note = self
            .xref_note
            .lock()
            .expect("the xref note is not poisoned");
        let xrefs = XrefSummary {
            unresolved: flat
                .skipped
                .iter()
                .filter(|s| s.reason == SkipReason::ExternalReference)
                .map(|s| UnresolvedXrefRow {
                    name: s.block_name.clone(),
                    path: String::new(),
                    resolved_path: None,
                    path_refusal: None,
                })
                .collect(),
            resolved: xref_note.resolved.iter().cloned().collect(),
            unattached: xref_note.unattached.clone(),
            notes: xref_note.notes.clone(),
        };
        let mut diagnostics = consumed.diagnostics;
        for name in &xref_note.unattached {
            diagnostics.push(Diagnostic::new(
                DiagnosticCode::XrefNotAttached,
                "xref-not-attached",
                format!(
                    "a drawing was supplied for xref {name}, and no xref block of that name exists"
                ),
            ));
        }
        // No rule named a wall, so the layer names carry no meaning here: the straight lines are
        // the walls, and the derivation drops the ones that join nothing (#1610).
        let wall_source = if walls.is_empty() && !lines.is_empty() {
            walls = lines;
            "geometry"
        } else {
            "layers"
        };
        Ok(Classified {
            geometries: consumed.geometries,
            stored: consumed.stored,
            rows: consumed.rows,
            tally: consumed.tally,
            encountered,
            view: ViewSummary::from_report(&view.report),
            xrefs,
            diagnostics,
            peak_queue_depth: consumed.peak,
            sides: consumed.sides,
            walls,
            wall_source,
            labels,
            texts,
            fills,
            annotation_counts,
            rasters,
            sheet,
        })
    }

    /// Everything after the source is open. A cancel that was really the deadline is told as
    /// [`PvecConvertError::Deadline`], whichever step noticed it.
    fn run<P: DwgProvider>(
        mut self,
        provider: &P,
        unit: SourceUnit,
    ) -> Result<PvecConversionOutput, PvecConvertError> {
        self.arm();
        let cancel = self.cancel.clone();
        self.run_armed(provider, unit)
            .map_err(|e| e.settled(&cancel))
    }

    fn run_armed<P: DwgProvider>(
        self,
        provider: &P,
        unit: SourceUnit,
    ) -> Result<PvecConversionOutput, PvecConvertError> {
        let Classified {
            geometries,
            stored,
            mut rows,
            mut tally,
            encountered,
            view,
            xrefs,
            diagnostics,
            peak_queue_depth,
            sides,
            walls,
            wall_source,
            labels,
            texts,
            fills,
            annotation_counts,
            rasters,
            sheet,
        } = self.classify(provider, Some(unit))?;
        #[cfg(feature = "pvec-raster-thumbnail")]
        let thumbnail_width = self.thumbnail_width;
        let (ask_texts, ask_fills) = (self.texts, self.fills);
        let Self {
            dest,
            leaves,
            spill,
            cancel,
            gate,
            calibration,
            observer,
            components: select,
            regions,
            ..
        } = self;
        // A sheet is in the unit its plot settings name, its box is the paper (so the writer
        // never refuses for bounds), and its plot scale is the calibration unless the caller gave
        // one (#1562).
        let (unit, area, calibration, sheet) = match sheet {
            Some(s) => (
                s.unit.unwrap_or(unit),
                s.frame,
                calibration.or(s.calibration),
                Some(s.summary),
            ),
            None => (unit, None, calibration, None),
        };
        let tell = |event: EngineEvent| {
            if let Some(o) = &observer {
                o.on_event(event);
            }
        };

        let write_error = |e: WriteError| match e {
            WriteError::Cancelled => PvecConvertError::Cancelled,
            other => PvecConvertError::Write(other),
        };
        // What both paths do once the artifact is written: tell the observer, close the report against
        // what went in. It runs once, so it takes the rows.
        // With no `written` (PVec refused every curve) there is no artifact, so the report is closed
        // against nothing written and the observer is not told.
        // The tally is sealed here and not at classification: an entity the classifier called
        // supported can still be one PVec's container refuses, and it is counted as that.
        let seal = |written: Option<(&BoundsReport, u64, u64)>,
                    refused: &[(EntityId, String)],
                    components: Option<ComponentsSummary>,
                    found: Option<RegionsSummary>|
         -> Result<(ConversionReport, FidelityReport), PvecConvertError> {
            for (id, rule) in refused {
                let from = &stored
                    .iter()
                    .find(|(s, _)| s == id)
                    .expect("a refused entity was written from a stored classification")
                    .1;
                let to = Classification::refused(ReasonCode::PvecCannotStore, from.source_class());
                tally.reclassify(from, &to);
                rows.refuse(*id, &to, format!("PVec's writer refused it ({rule})"));
            }
            let fidelity = tally.finish(encountered).map_err(PvecConvertError::Tally)?;
            if let Some((_, entities, bytes)) = written {
                tell(EngineEvent::ConversionArtifactWritten { entities, bytes });
            }
            let mut report = rows
                .finish(
                    encountered,
                    view,
                    xrefs,
                    BoundsSplit {
                        exact: written.map_or(0, |(b, ..)| b.exact_count() as u64),
                        hull_upper_bound: written.map_or(0, |(b, ..)| b.hull_count() as u64),
                    },
                )
                .map_err(PvecConvertError::Report)?;
            report.capabilities = Some(gate.summary());
            report.diagnostics = diagnostics;
            report.components = components;
            report.regions = found;
            report.sheet = sheet.clone();
            Ok((report, fidelity))
        };
        let config = WriteConfig { unit, leaves };
        // The manifest box takes in every placed raster too, so a logo past the linework is inside it.
        let open_area = rasters.iter().fold(area, |acc, r| {
            let [x0, y0, x1, y1] = r.visible_box.map(f64::from);
            let [a, b, c, d, tx, ty] = r.pixel_to_source;
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                .iter()
                .map(|p| [a * p[0] + b * p[1] + tx, c * p[0] + d * p[1] + ty])
                .fold(acc, |acc, q| {
                    Some(acc.map_or([q[0], q[1], q[0], q[1]], |b| {
                        [
                            b[0].min(q[0]),
                            b[1].min(q[1]),
                            b[2].max(q[0]),
                            b[3].max(q[1]),
                        ]
                    }))
                })
        });
        let mut chunk = 0u64;
        let mut on_chunk = |entities: u64, bytes: u64| {
            chunk += 1;
            tell(EngineEvent::ConversionChunkWritten {
                chunk,
                entities,
                bytes,
            });
        };
        // The geometry is written first, then the streams against the ids the writer kept, because a
        // refused curve moves every id after it. The streams go in after the geometry, so they cannot
        // change a geometry block.
        // The writer takes what it keeps out of the geometry, so the geometry goes the moment the
        // open call returns, whether it succeeded or not, and the report is built without it.
        let hatch_patterns: std::cell::RefCell<BTreeMap<u32, Vec<PlacedPatternLine>>> =
            std::cell::RefCell::new(BTreeMap::new());
        let open = |geometries: Vec<(EntityId, Normalized2DGeometry)>,
                    spill: Option<&SpillConfig>,
                    on_chunk: &mut dyn FnMut(u64, u64)|
         -> Result<
            (
                OpenArtifact,
                Option<ComponentsSummary>,
                Option<RegionsSummary>,
            ),
            PvecConvertError,
        > {
            let opened = open_geometry(
                &config,
                geometries.iter().map(|(id, g)| (*id, g)),
                Vec::new(),
                open_area,
                spill,
                &cancel,
                on_chunk,
            );
            drop(geometries);
            let mut open = opened.map_err(write_error)?;
            // PVec sorts the calibration block after every other kind, so it can go in first.
            if let Some(c) = &calibration {
                open.calibration(c).map_err(write_error)?;
            }
            if !rasters.is_empty() {
                open.rasters(&rasters).map_err(PvecConvertError::Rasters)?;
            }
            if select == StreamSelection::NONE && regions.is_none() {
                return Ok((open, None, None));
            }
            let mut streams = Vec::new();
            let mut summary = None;
            if select != StreamSelection::NONE {
                let layers = if select.layer {
                    provider
                        .layers()
                        .map_err(|e| PvecConvertError::View(e.into()))?
                } else {
                    Vec::new()
                };
                let rows: Vec<ComponentSource<'_>> = sides
                    .iter()
                    .map(|(id, s)| ComponentSource {
                        id: *id,
                        stub: &s.stub,
                        style: s.style.as_ref(),
                        viewport: s.viewport,
                        paint: s.paint,
                    })
                    .collect();
                let built = build_streams(select, &open.source_ids, &layers, &rows)
                    .map_err(PvecConvertError::Components)?;
                // Only the entities the writer kept: one it refused has no row, so it is not counted here.
                let kept: BTreeSet<EntityId> = open.source_ids.iter().copied().collect();
                let kept_sides = || {
                    sides
                        .iter()
                        .filter(|(id, _)| kept.contains(id))
                        .map(|(_, s)| s)
                };
                let mut made = ComponentsSummary {
                    streams: built.names.iter().map(|n| (*n).to_owned()).collect(),
                    dropped: BTreeMap::new(),
                    colours_ignored: kept_sides().filter(|s| s.colour_ignored).count() as u64,
                    annotations: annotation_counts.clone(),
                    texts: ask_texts && select.line_style,
                    fills: ask_fills && select.line_style,
                };
                let style_bad = kept_sides().filter(|s| s.bad_style).count() as u64;
                for d in &built.dropped {
                    *made
                        .dropped
                        .entry(format!("{}.{}", d.stream, d.reason))
                        .or_default() += 1;
                }
                if style_bad > 0 {
                    made.dropped
                        .insert("line-style.bad-style".to_owned(), style_bad);
                }
                streams = built.streams;
                let (carried, dropped, patterns) =
                    annotation_streams(&texts, &fills, &open.source_ids)
                        .map_err(PvecConvertError::Components)?;
                *hatch_patterns.borrow_mut() = patterns;
                streams.extend(carried);
                for d in &dropped {
                    *made
                        .dropped
                        .entry(format!("{}.{}", d.stream, d.reason))
                        .or_default() += 1;
                }
                summary = Some(made);
            }
            let mut found = None;
            if let Some(config) = &regions {
                let mut report = RegionsSummary {
                    wall_source: Some(wall_source.to_owned()),
                    ..Default::default()
                };
                match derive_regions(&walls, &labels, &[], config) {
                    // A plan past a limit converts without regions, and the report says which limit.
                    Err(e) => report.refused = Some(e.reason_code().to_owned()),
                    Ok(derived) => {
                        report.rooms = derived.report.rooms;
                        report.corridors = derived.report.corridors;
                        report.gaps_closed = derived.report.gaps_closed;
                        report.unlabelled = derived.report.unlabelled;
                        report.labelled =
                            derived.regions.iter().filter(|r| r.label.is_some()).count() as u64;
                        let class = semantic_stream(&derived, &open.source_ids)
                            .map_err(PvecConvertError::Components)?;
                        for d in &class.dropped {
                            *report.dropped.entry(d.reason.to_owned()).or_default() += 1;
                        }
                        report.written = (derived.regions.len() - class.dropped.len()) as u64;
                        streams.extend(class.stream);
                    }
                }
                found = Some(report);
            }
            open.components(streams, &cancel).map_err(write_error)?;
            Ok((open, summary, found))
        };
        // The report is written from the sealed value straight into its partial file, so the whole
        // of it never exists as one string. A failure while writing it reaches the caller as
        // `Publish(Io)`, the publisher's own error, not as `Report`.
        let write_report = |sealed: &(ConversionReport, FidelityReport, u64),
                            out: &mut dyn std::io::Write| {
            sealed.0.write_json(out).map_err(std::io::Error::other)
        };
        let publish_error = |e: StreamedPublishError<PvecConvertError>| match e {
            StreamedPublishError::Produce(e) => e,
            StreamedPublishError::Publish(PublishError::Cancelled) => PvecConvertError::Cancelled,
            StreamedPublishError::Publish(other) => PvecConvertError::Publish(other),
        };
        // A paper sheet with nothing to stroke (only text, say) still has a paper to show: a
        // thumbnail of it is blank (#1629), and the PDF plot wants the empty page (#1636). Either way
        // no artifact is written.
        #[cfg(feature = "pvec-raster-thumbnail")]
        let thumbnail_blank = thumbnail_width.is_some();
        #[cfg(not(feature = "pvec-raster-thumbnail"))]
        let thumbnail_blank = false;
        let blank_paper = (thumbnail_blank || self.empty_sheet_ok) && area.is_some();
        let published = match &spill {
            None => match open(geometries, None, &mut on_chunk) {
                Err(PvecConvertError::Write(WriteError::NoGeometry)) if blank_paper => {
                    seal(None, &[], None, None).map(|(report, fidelity)| (report, fidelity, 0))
                }
                Err(PvecConvertError::Write(WriteError::AllRefused(refused))) => {
                    seal(None, &refused, None, None).and_then(|(report, fidelity)| {
                        Err(PvecConvertError::NothingStorable {
                            report: Box::new(report),
                            fidelity: Box::new(fidelity),
                        })
                    })
                }
                opened => opened.and_then(|(open, summary, found)| {
                    let artifact = open.finish().map_err(write_error)?;
                    publish_streamed_with(
                        dest,
                        &cancel,
                        &mut |_| {},
                        |out| {
                            out.write_all(&artifact.bytes)
                                .map_err(|e| PvecConvertError::Publish(PublishError::Io(e)))?;
                            let (report, fidelity) = seal(
                                Some((
                                    &artifact.bounds,
                                    artifact.written,
                                    artifact.bytes.len() as u64,
                                )),
                                &artifact.refused,
                                summary,
                                found,
                            )?;
                            Ok((report, fidelity, artifact.written))
                        },
                        write_report,
                    )
                    .map_err(publish_error)
                }),
            },
            Some(spill) => publish_streamed_with(
                dest,
                &cancel,
                &mut |_| {},
                |out| {
                    let (open, summary, found) = match open(geometries, Some(spill), &mut on_chunk)
                    {
                        Err(PvecConvertError::Write(WriteError::AllRefused(refused))) => {
                            let (report, fidelity) = seal(None, &refused, None, None)?;
                            return Err(PvecConvertError::NothingStorable {
                                report: Box::new(report),
                                fidelity: Box::new(fidelity),
                            });
                        }
                        opened => opened?,
                    };
                    let artifact = open
                        .seal_into(&cancel, &mut on_chunk, out)
                        .map_err(write_error)?;
                    tell(EngineEvent::ConversionSpoolSpilled {
                        bytes: artifact.length,
                    });
                    let (report, fidelity) = seal(
                        Some((&artifact.bounds, artifact.written, artifact.length)),
                        &artifact.refused,
                        summary,
                        found,
                    )?;
                    Ok((report, fidelity, artifact.written))
                },
                write_report,
            )
            .map_err(publish_error),
        };
        let (report, fidelity, written) = published?;
        if written > 0 || !blank_paper {
            tell(EngineEvent::ConversionPublished);
        }
        // A paper frame fills the picture; a frame that is only the kept geometry's extent (model
        // space, or a layout without plot settings) gets 2 percent of its longer side all round.
        #[cfg(feature = "pvec-raster-thumbnail")]
        let thumbnail = thumbnail_width
            .map(|width| {
                use crate::pvec::thumbnail::{ThumbnailError, draw};
                let paper = report.sheet.as_ref().is_some_and(|s| s.frame == "paper");
                let margin = if paper { 0.0 } else { 0.02 };
                match area.filter(|_| written == 0) {
                    Some([x0, y0, x1, y1]) => {
                        draw([x0, y0], [x1, y1], std::iter::empty(), width, 0.0)
                    }
                    None => {
                        let reader = crate::pvec::publish::open_published(dest)
                            .map_err(|e| ThumbnailError::Artifact(e.to_string()))?;
                        let curves = reader
                            .entities()
                            .map_err(|e| ThumbnailError::Artifact(e.to_string()))?;
                        let b = reader.manifest().bounds;
                        draw(b.min, b.max, curves.iter().map(|(_, c)| c), width, margin)
                    }
                }
            })
            .transpose()
            .map_err(PvecConvertError::Thumbnail)?;
        Ok(PvecConversionOutput {
            artifact: dest.to_path_buf(),
            report,
            fidelity,
            written,
            peak_queue_depth,
            hatch_patterns: hatch_patterns.into_inner(),
            #[cfg(feature = "pvec-raster-thumbnail")]
            thumbnail,
        })
    }
}

/// What `classify` hands on: one record and one row per entity, and the counts that check them.
struct Classified {
    geometries: Vec<(EntityId, Normalized2DGeometry)>,
    /// The classification of each entity in `geometries`, for one the writer then refuses.
    stored: Vec<(EntityId, Classification)>,
    rows: ReportBuilder,
    tally: Tally,
    encountered: u64,
    view: ViewSummary,
    xrefs: XrefSummary,
    diagnostics: Vec<Diagnostic>,
    peak_queue_depth: u32,
    /// What the component streams need of each written entity; empty when they are off.
    sides: Vec<(EntityId, Side)>,
    /// The straight wall leaves and the text anchors the regions step reads; empty when it is off.
    walls: Vec<WallEdge>,
    /// `geometry` when the walls are every straight line because no rule matched, else `layers`.
    wall_source: &'static str,
    labels: Vec<RoomLabel>,
    /// The texts and hatches carried in streams, and the count of each class by outcome (#1609).
    texts: Vec<(EntityId, TextBody)>,
    fills: Vec<(Vec<EntityId>, FillBody)>,
    annotation_counts: BTreeMap<String, u64>,
    /// The IMAGEs placed as rasters, written after the geometry (#1667).
    rasters: Vec<crate::raster_pvec::pyramid::RasterPyramid>,
    /// The paper sheet, for a paper layout under the print profile.
    sheet: Option<Sheet>,
}

/// The plot scale as the calibration record (P-9, #1562): `n` paper units to `d` drawing units,
/// the paper unit being `sheet_unit`. The sheet's linework is drawn at it (#1665), so a coordinate
/// is a unit of paper in the sheet unit, which is what the record has always taken it to be. A
/// caller's calibration wins. The summary says what became of it.
fn plot_scale_record(
    [n, d]: [f64; 2],
    paper_unit: SourceUnit,
    sheet_unit: SourceUnit,
    unit: Option<SourceUnit>,
    single_scale: Option<&str>,
    caller_calibration: bool,
    summary: &mut SheetSummary,
) -> Option<Calibration> {
    let sized = |v: f64| v.is_finite() && v > 0.0;
    let mut written = None;
    // P-9: the plot scale is the calibration record, and since #1665 also the
    // scale the sheet is drawn at. A caller's calibration wins.
    if caller_calibration {
        summary.notes.push("caller-calibration".into());
    } else if !(sized(n) && sized(d)) {
        summary.notes.push("scale-zero".into());
    } else {
        // Both sides integral after scaling by one power of ten up to 10^6
        // reduce exactly; anything else is rounded at 10^6 and the record
        // carries the rounding as its uncertainty. A side is integral at `k`
        // when its rounded multiple divided back by `k` is the stored value:
        // a decimal such as 2.01 is no binary fraction, so `2.01 * 100` is not
        // 201 in f64, but `201 / 100` is the f64 nearest 2.01, which is it.
        let integral = |v: f64| (1.0..9.0e15).contains(&v).then_some(v as u64);
        let at = |v: f64, k: f64| {
            let r = (v * k).round();
            if r / k == v { integral(r) } else { None }
        };
        let exact = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
            .into_iter()
            .find_map(|k| Some((at(n, k)?, at(d, k)?)));
        let uncertain = exact.is_none();
        let ratio =
            exact.or_else(|| Some((integral((n * 1e6).round())?, integral((d * 1e6).round())?)));
        let reduced = ratio.and_then(|(a, b)| {
            let (mut x, mut y) = (a, b);
            while y != 0 {
                (x, y) = (y, x % y);
            }
            Some((u32::try_from(a / x).ok()?, u32::try_from(b / x).ok()?))
        });
        let paper = metres_per_unit(paper_unit).ok();
        // The coordinates are in the sheet's unit and the scale's paper side in `paper_unit`, so
        // a coordinate unit is this many paper units.
        let along = if paper_unit == sheet_unit {
            Some(1.0)
        } else {
            paper
                .zip(metres_per_unit(sheet_unit).ok())
                .map(|(p, s)| s / p)
        };
        // The scale is paper units to drawing units, so a paper unit is
        // `den / num` drawing units, each the size of the drawing's own unit.
        // Inches of model space on a millimetre sheet are not millimetres
        // (#1610). A drawing that names no length unit stands in the sheet's.
        let drawing = unit
            .filter(|u| u.is_physical())
            .and_then(|u| metres_per_unit(u).ok())
            .or(metres_per_unit(sheet_unit).ok());
        let claim = reduced
            .zip(paper)
            .zip(along)
            .map(|(((num, den), paper), along)| {
                let raw = n / d;
                let mut note = if uncertain {
                    format!("plot scale {n}:{d} as stored, written as {num}:{den}")
                } else {
                    format!("plot scale {num}:{den}")
                };
                if let Some(one) = &single_scale {
                    note.push_str(&format!("; viewport {one}"));
                }
                ScaleClaim {
                    metres_per_source_unit: drawing.unwrap_or(paper * along) * f64::from(den)
                        / f64::from(num)
                        * along,
                    provenance: ScaleProvenance::PlanScale {
                        numerator: num,
                        denominator: den,
                        paper_metres_per_unit: paper,
                    },
                    uncertainty: uncertain
                        .then(|| (f64::from(num) / f64::from(den) - raw).abs() / raw),
                    note: if note.len() <= 255 {
                        note
                    } else {
                        format!("plot scale written as {num}:{den}")
                    },
                }
            });
        match claim.map(|c| Calibration::new(Some(c), None, None)) {
            Some(Ok(c)) => {
                if uncertain {
                    summary.notes.push("scale-not-reduced".into());
                }
                summary.calibration = if uncertain {
                    "plot-scale-uncertain".into()
                } else {
                    "plot-scale".into()
                };
                summary.scale = reduced.map(|(a, b)| format!("{a}:{b}"));
                written = Some(c);
            }
            _ => summary.notes.push("scale-refused".into()),
        }
    }
    written
}

/// A paper sheet as `classify` worked it out (#1562): the report's section, the frame to clip to
/// and to hand the writer, the unit the plot settings put the sheet in, the calibration the plot
/// scale gives and whether widths print.
struct Sheet {
    summary: SheetSummary,
    /// `[min x, min y, max x, max y]` in the sheet unit, `None` when the frame is the extents.
    frame: Option<[f64; 4]>,
    unit: Option<SourceUnit>,
    calibration: Option<Calibration>,
    drop_widths: bool,
    /// What the frame is placed from, for the extents pass (#1628). `None` when it is the extents.
    place: Option<Place>,
    /// The layout plots upside-down (rotation 2 or 3): the content turns 180 degrees about the
    /// frame's centre (#1632).
    flip: bool,
    /// Units of paper, in the sheet unit, that one paper-space drawing unit prints as: the plot
    /// scale, applied to the sheet's own linework, its windows and its extents (#1665). 1 when
    /// the frame is the extents.
    per_unit: f64,
}

/// A paper sheet's placement in the sheet unit: what the frame is worked out from once the
/// extents of what the sheet draws are known (#1628).
#[derive(Clone, Copy)]
struct Place {
    /// Unprintable margins, left, bottom, right, top.
    margin: [f64; 4],
    /// The paper, after the layout's rotation and the chosen orientation.
    size: [f64; 2],
    /// The margin plus the shift, summed in millimetres and converted once, so that a sheet with
    /// no override frames exactly as it did before the plot settings (#1584).
    corner: [f64; 2],
}

/// What the component streams read off one written entity: where it came from and its line style.
struct Side {
    stub: ProvenanceStub,
    style: Option<LineStyleComponent>,
    /// The colour word, `0xRRGGBBAA`, when the drawing settles it (#1638).
    paint: Option<u32>,
    /// The paint stream is on and the colour could not be resolved (#1638).
    colour_ignored: bool,
    /// The line style stream is on and the DWG facts could not make a component of this entity.
    bad_style: bool,
    /// The viewport a model entity was drawn through on a sheet (#1563).
    viewport: Option<crate::cad::ItemHandle>,
}

/// AutoCAD's own paper view: number 1, showing paper space 1:1 centred on itself
/// (`sample_AC1018`). The reader numbers viewports, so the number alone is not enough: the gold
/// probes give 1 to a window onto the model (#1563). A sheet's conversion and the layout set's
/// sheet test both decide it here (#1583).
fn paper_view(v: &ViewportEntity) -> bool {
    v.id == 1 && v.center == v.view_center && v.height == v.view_height
}

/// What a viewport of a sheet comes to, decided from its record alone (#1563, #1626): the
/// conversion and the manifest's listing both ask this, so the reason a viewport is refused is
/// the same one before and after the conversion.
enum ViewportClass {
    /// The sheet's own paper view.
    Paper,
    /// Contributes nothing to a print: `off` or `hide-plot` (P-3).
    Dropped(&'static str),
    /// A view the print cannot map, and why.
    Refused(&'static str),
    /// Model space shows through it, cut by this polygon when it is clipped to one (#1565).
    Drawn(Option<Vec<[f64; 2]>>),
}

fn viewport_class(v: &ViewportEntity) -> ViewportClass {
    let finite = [v.width, v.height, v.view_height, v.center[0], v.center[1]]
        .into_iter()
        .chain(v.view_center)
        .chain(v.view_target)
        .all(f64::is_finite);
    if paper_view(v) {
        ViewportClass::Paper
    } else if !v.visible {
        ViewportClass::Dropped("off")
    } else if v.hide_plot {
        ViewportClass::Dropped("hide-plot")
    } else if !{
        // Plan view up to float noise (#1660): a DWG stores the camera minus the target, so a plan
        // view comes back as (5.8e-11, 0, 251705.56) or the like, about one ulp off. x and y over z
        // is the lean to first order and needs no square root, and an exact (0, 0, z) passes as before.
        // NaN fails every compare, and so does an infinite lean (inf over inf is NaN), so both stay
        // refused, while an infinite z alone still reads as plan.
        const LEAN: f64 = 1e-9;
        let [x, y, z] = v.view_direction;
        z > 0.0 && x.abs() / z <= LEAN && y.abs() / z <= LEAN
    } {
        ViewportClass::Refused("non-plan")
    } else if v.perspective {
        ViewportClass::Refused("perspective")
    } else if !(finite && v.view_height > 0.0 && v.height > 0.0 && v.width > 0.0) {
        ViewportClass::Refused("view-height")
    } else if !{
        // Under 1e-9 rad from a whole turn is float noise (#1660). AutoCAD keeps the twist in
        // [0, 2 pi), so noise below zero is stored just under a full turn. NaN is refused.
        const NOISE: f64 = 1e-9;
        let t = v.twist.rem_euclid(std::f64::consts::TAU);
        t.min(std::f64::consts::TAU - t) <= NOISE
    } {
        // The twist sign is not measured against AutoCAD yet (the ADR's open item).
        ViewportClass::Refused("twist")
    } else {
        // A rectangle, or a polygon with straight edges (#1565), whose vertices are in paper
        // units and cut what the viewport shows. A bulged edge or a curved boundary is not cut yet.
        match &v.clip {
            ViewportClip::Rectangle(_) => ViewportClass::Drawn(None),
            ViewportClip::Polygon(vs) if vs.iter().any(|q| q.bulge != 0.0) => {
                ViewportClass::Refused("clip-bulge")
            }
            ViewportClip::Polygon(vs)
                if vs.len() < 3 || !vs.iter().all(|q| q.x.is_finite() && q.y.is_finite()) =>
            {
                ViewportClass::Refused("clip-polygon")
            }
            ViewportClip::Polygon(vs) => {
                ViewportClass::Drawn(Some(vs.iter().map(|q| [q.x, q.y]).collect()))
            }
            ViewportClip::Circle(_) => ViewportClass::Refused("clip-circle"),
            ViewportClip::Ellipse(_) => ViewportClass::Refused("clip-ellipse"),
            ViewportClip::Spline(_) => ViewportClass::Refused("clip-spline"),
            _ => ViewportClass::Refused("clip-other"),
        }
    }
}

/// What the print profile makes of one viewport of a sheet (#1563).
enum Window {
    /// The sheet's own paper view, not a window onto the model.
    Paper,
    /// Contributes nothing to a print: `off` or `hide-plot` (P-3).
    Dropped(&'static str),
    /// A view this slice cannot map, refused with 508, and why.
    Refused(&'static str),
    /// Model space shows through it: the similarity onto the sheet, its rectangle there (for a
    /// polygon clip, the polygon's box), and the polygon when it is clipped to one (#1565).
    Drawn {
        map: [f64; 6],
        rect: [f64; 4],
        ring: Option<Vec<[f64; 2]>>,
    },
}

/// One entity's outcome on its way from the classifier to the consumer.
struct Outcome {
    id: EntityId,
    handle: Option<u64>,
    /// The viewport it was drawn through, for the report's row (#1563).
    viewport: Option<u64>,
    done: Done,
    diagnostic: Option<Diagnostic>,
    side: Option<Side>,
}

/// What the consumer built from the outcomes.
struct Consumed {
    tally: Tally,
    rows: ReportBuilder,
    geometries: Vec<(EntityId, Normalized2DGeometry)>,
    stored: Vec<(EntityId, Classification)>,
    diagnostics: Vec<Diagnostic>,
    sides: Vec<(EntityId, Side)>,
    peak: u32,
}

/// Drains the queue: tallies, rows, records and diagnostics, and the observer's events in order.
/// The depth is sampled after each receive from the sends that completed, so it can read one low
/// and never one high.
fn consume(
    rx: impl IntoIterator<Item = Outcome>,
    sent: &AtomicU64,
    observer: Option<&dyn EngineObserver>,
) -> Consumed {
    let mut out = Consumed {
        tally: Tally::new(),
        rows: ReportBuilder::new(),
        geometries: Vec::new(),
        stored: Vec::new(),
        diagnostics: Vec::new(),
        sides: Vec::new(),
        peak: 0,
    };
    let mut received = 0u64;
    for Outcome {
        id,
        handle,
        viewport,
        done,
        diagnostic,
        side,
    } in rx
    {
        received += 1;
        let depth = sent.load(Ordering::Acquire).saturating_sub(received);
        out.peak = out.peak.max(u32::try_from(depth).unwrap_or(u32::MAX));
        out.tally.record(&done.classification);
        if let Some(d) = diagnostic {
            if let Some(o) = observer {
                o.on_event(diagnostic_event(&d));
            }
            out.diagnostics.push(d);
        }
        let mut row = EntityRow::new(
            id,
            &done.classification,
            handle,
            done.geometry
                .geometry()
                .and_then(|_| kind_of(&done.geometry))
                .map(|k| row_for(k).pvec_type),
            done.warnings,
        );
        row.viewport = viewport;
        out.rows.push(row);
        if let Some(side) = side {
            out.sides.push((id, side));
        }
        if done.geometry.geometry().is_some() {
            out.stored.push((id, done.classification));
        }
        out.geometries.push((id, done.geometry));
        if received.is_multiple_of(BATCH as u64)
            && let Some(o) = observer
        {
            o.on_event(EngineEvent::ConversionBatchNormalized { entities: received });
        }
    }
    out
}

/// The capability gate: which capabilities pass, whether only exact entities do, and what happens
/// to the rest. The three vary together per profile.
struct Gate {
    name: &'static str,
    allowed: BTreeSet<CapabilityId>,
    exact_only: bool,
    policy: CapabilityPolicy,
    /// A LINE whose ends are at different heights is drawn as the segment it covers in plan, not
    /// refused as 3D. Not part of a profile: a plot asks for it.
    project_plan: bool,
}

impl Gate {
    fn of(profile: ConversionProfile) -> Self {
        Self {
            name: profile.name(),
            project_plan: false,
            allowed: profile.capabilities(),
            exact_only: profile.exact_only(),
            policy: profile.policy(),
        }
    }

    fn summary(&self) -> CapabilitySummary {
        CapabilitySummary {
            profile: self.name.to_owned(),
            policy: format!("{:?}", self.policy),
            exact_only: self.exact_only,
            allowed: self.allowed.iter().map(|c| c.get()).collect(),
        }
    }
}

/// What the conversion met: a drawn leaf, a leaf the profile excluded, an INSERT that was not
/// expanded, or a leaf of a block out of the plan's plane with no image in the plan (#1609).
enum Item<'a> {
    Leaf(&'a FlatEntity),
    Excluded(&'a ExcludedEntity),
    Skipped(&'a SkippedInsert),
    Unplaced(&'a UnplacedLeaf),
}

/// One entity's outcome: its class, its record (with geometry when it has any) and the warnings.
struct Done {
    classification: Classification,
    geometry: Normalized2DGeometry,
    warnings: Vec<String>,
    /// A solid stub at an end of a polyline side whose pattern is centred (#1699): its style carries no
    /// dash array.
    solid: bool,
}

/// The lines a LEADER, MULTILEADER or TOLERANCE leaf draws (#1671), each as a leaf of its own with
/// the entity's provenance, style and placement, so the layer rules, the view pass and any viewport
/// it is seen through treat them as they do the entity. A MULTILEADER's or TOLERANCE's own leaf
/// stays for its text; a LEADER has nothing else to say, so the keys of the LEADER leaves whose
/// lines are drawn come back too, and those leaves are left out.
fn line_leaves<P: DwgProvider>(
    provider: &P,
    leaves: &[FlatEntity],
) -> (Vec<FlatEntity>, BTreeSet<LeafKey>) {
    let (mut lines, mut replaced) = (Vec::new(), BTreeSet::new());
    for f in leaves {
        let class = f.entity.provenance.class_name.as_str();
        if !matches!(f.entity.geometry, RawGeometry::Unmapped { .. })
            || !matches!(class, "LEADER" | "MULTILEADER" | "TOLERANCE")
        {
            continue;
        }
        let Some(handle) = f.entity.provenance.handle else {
            continue;
        };
        let drawn = provider.annotation_lines(handle);
        if class == "LEADER" && !drawn.is_empty() {
            replaced.insert(flat_key(f));
        }
        // No entity draws more than this many lines: a frame of rows and cells or a leader's
        // lines and arrowheads are a few dozen at most.
        const MAX_LINES_PER_ENTITY: usize = 4096;
        lines.extend(
            drawn
                .into_iter()
                .take(MAX_LINES_PER_ENTITY)
                .map(|geometry| {
                    f.carrying(RawEntity {
                        geometry,
                        provenance: f.entity.provenance.clone(),
                        style: f.entity.style.clone(),
                    })
                }),
        );
    }
    (lines, replaced)
}

/// A leaf's handle, block path and class, which says which view pass decision is its own.
type LeafKey = (Option<u64>, String, String);

/// [`leaf_key`] of a flattened leaf, so the view pass and the flatten pass meet on the same key.
fn flat_key(f: &FlatEntity) -> LeafKey {
    leaf_key_of(&f.entity.provenance)
}

fn leaf_key(v: &ViewedEntity) -> (Option<u64>, String, String) {
    leaf_key_of(&v.entity.provenance)
}

fn leaf_key_of(p: &ProvenanceStub) -> (Option<u64>, String, String) {
    (
        p.handle.map(|h| h.get()),
        p.block_path.to_string(),
        p.class_name.clone(),
    )
}

/// The colour word of a resolved colour: its RGB with an opaque alpha. ACI 7 is white on a dark
/// screen and black on paper, and a plot prints it black, so that is how it is written. An index
/// with no RGB behind it has no word.
fn paint_word(c: SourceColour) -> Option<u32> {
    let (r, g, b) = match c {
        SourceColour::Aci(7) => (0, 0, 0),
        SourceColour::Aci(i) => acadrust::Color::Index(i).rgb()?,
        SourceColour::True { r, g, b } => (r, g, b),
        _ => return None,
    };
    Some(u32::from_be_bytes([r, g, b, 255]))
}

/// A stand-in `RawEntity` for something that has no geometry of its own, so it can be ordered and
/// numbered with the rest.
fn stub(
    handle: Option<u64>,
    class_name: &str,
    block_path: crate::pvec::InstancePath,
    layer: &str,
    why: &str,
) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Unmapped {
            reason: why.to_owned(),
        },
        provenance: ProvenanceStub {
            handle: handle.map(crate::cad::ItemHandle::new),
            class_name: class_name.to_owned(),
            block_path,
            layer: layer.to_owned(),
        },
    }
}

fn without_geometry(classification: Classification, warnings: Vec<String>) -> Done {
    let geometry = Normalized2DGeometry::without_geometry(classification.class())
        .expect("a refusing class carries no geometry");
    Done {
        classification,
        geometry,
        warnings,
        solid: false,
    }
}

/// `viewport` is the window a model leaf is drawn through on a sheet (#1563), whose map is already
/// composed onto the leaf's.
fn classify_item(
    item: &Item<'_>,
    raw: &RawEntity,
    gate: &Gate,
    viewport: Option<crate::cad::ItemHandle>,
) -> Result<Done, PvecConvertError> {
    let class = raw.provenance.class_name.as_str();
    Ok(match item {
        Item::Excluded(x) => without_geometry(x.reason.classification(class), Vec::new()),
        Item::Skipped(s) => {
            let reason = match s.reason {
                SkipReason::NormalNotZ => ReasonCode::ExtrusionNotApplied,
                _ => ReasonCode::UnresolvedBlock,
            };
            without_geometry(
                Classification::refused(reason, "INSERT"),
                vec![format!("INSERT of `{}`: {}", s.block_name, s.reason.code())],
            )
        }
        Item::Unplaced(u) => without_geometry(
            Classification::refused(ReasonCode::ExtrusionNotApplied, class),
            vec![format!(
                "{class}: in a block out of the plan's plane, {}",
                u.why
            )],
        ),
        Item::Leaf(f) => classify_leaf(f, gate, viewport)?,
    })
}

fn classify_leaf(
    f: &FlatEntity,
    gate: &Gate,
    viewport: Option<crate::cad::ItemHandle>,
) -> Result<Done, PvecConvertError> {
    let class = f.entity.provenance.class_name.as_str();
    // A plan shows a sloped line as the segment it covers, when the caller asked for that.
    let flattened = match &f.entity.geometry {
        RawGeometry::Line { start, end } if gate.project_plan && start[2] != end[2] => {
            Some(RawEntity {
                geometry: RawGeometry::Line {
                    start: [start[0], start[1], 0.0],
                    end: [end[0], end[1], 0.0],
                },
                ..f.entity.clone()
            })
        }
        _ => None,
    };
    let basic = match dwg_basic::extract(flattened.as_ref().unwrap_or(&f.entity)) {
        Ok(b) => b,
        Err(BasicError::NotBasic) => return classify_spline_leaf(f, gate, viewport),
        Err(e) => {
            return Ok(without_geometry(
                e.classification(class),
                vec![e.warning(class)],
            ));
        }
    };
    // A curve seen edge on is drawn as the segment it projects to: what a plan shows, and not the
    // curve itself (#1609).
    let fidelity = if f.edge_on() {
        FidelityClass::SupportedEquivalent
    } else {
        basic.classification.class()
    };
    let record = match Normalized2DGeometry::from_curve(basic.curve, fidelity) {
        Ok(r) => r,
        Err(e) => {
            return Ok(without_geometry(
                Classification::refused(ReasonCode::DegenerateGeometry, class),
                vec![format!("{class}: {e}")],
            ));
        }
    };
    let mut done = promote_leaf(f, record, gate, viewport)?;
    if f.edge_on() && done.geometry.geometry().is_some() {
        done.warnings.push(format!(
            "{class}: in a block in a vertical plane, seen edge on and drawn as the segment it covers"
        ));
    }
    Ok(done)
}

/// Pushes `record` through the map flattening composed for its leaf, and checks it against the
/// gate. A leaf a layout owns directly has the identity map, unless it is drawn through a viewport
/// or the sheet plots upside-down.
fn promote_leaf(
    f: &FlatEntity,
    record: Normalized2DGeometry,
    gate: &Gate,
    viewport: Option<crate::cad::ItemHandle>,
) -> Result<Done, PvecConvertError> {
    let class = f.entity.provenance.class_name.as_str();
    // A leaf a layout owns directly has the identity map. A nested one is pushed through the map
    // flattening composed for it, as one level named by its innermost INSERT; a model leaf of its
    // own seen through a viewport, as one level named by the viewport (#1563).
    let path = &f.entity.provenance.block_path;
    let prov = &f.entity.provenance;
    let step = match (path.steps().last(), viewport) {
        (Some(s), _) => s.clone(),
        (None, Some(v)) => crate::pvec::InstanceStep {
            block_name: "*VIEWPORT".into(),
            insert_handle: v,
        },
        (None, None) if f.matrix() == [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] => {
            return gated(record, prov, gate);
        }
        // A sheet's own leaf drawn at the plot scale (#1665), and turned about the frame's centre
        // by a plot upside-down (#1632).
        (None, None) => crate::pvec::InstanceStep {
            block_name: if f.matrix()[0] < 0.0 {
                "*PLOT_ROTATION"
            } else {
                "*PLOT_SCALE"
            }
            .into(),
            insert_handle: prov
                .handle
                .unwrap_or_else(|| crate::cad::ItemHandle::new(1)),
        },
    };
    let composed = match ComposedTransform::identity().push_inner_coefficients(step, f.matrix()) {
        Ok(c) => c,
        Err(e) => {
            return Ok(without_geometry(
                e.classification(class),
                vec![e.to_string()],
            ));
        }
    };
    match promote(&record, &composed) {
        Ok(p) if p.record().geometry().is_some() => gated(p.record().clone(), prov, gate),
        Ok(p) => {
            let reason = p
                .reason()
                .map_or(ReasonCode::UnmappedSourceClass, |r| r.reason_code());
            Ok(without_geometry(
                Classification::refused(reason, class),
                Vec::new(),
            ))
        }
        Err(e) => Ok(without_geometry(
            Classification::refused(ReasonCode::UnmappedSourceClass, class),
            vec![format!("{class}: {e:?}")],
        )),
    }
}

/// Whether a source class is one of acadrust's dimension classes (`DIMENSION_LINEAR` and the rest,
/// `ARC_DIMENSION`, `LARGE_RADIAL_DIMENSION`), which the provider reports as INSERTs of their blocks
/// (#1658).
fn is_dimension_class(class: &str) -> bool {
    class.starts_with("DIMENSION_") || matches!(class, "ARC_DIMENSION" | "LARGE_RADIAL_DIMENSION")
}

/// A leaf the basic extractor does not take: a SPLINE goes through the exact spline extractor
/// (#1430), anything else is an unmapped source class. The record's own class is the entity's, so
/// a one-span spline that became a Bezier is exact and an unwrapped periodic one is equivalent.
fn classify_spline_leaf(
    f: &FlatEntity,
    gate: &Gate,
    viewport: Option<crate::cad::ItemHandle>,
) -> Result<Done, PvecConvertError> {
    let class = f.entity.provenance.class_name.as_str();
    let spline = match dwg_spline::extract(&f.entity) {
        Ok(s) => s,
        Err(SplineError::NotSpline) => {
            return Ok(without_geometry(
                crate::pvec::classify_unmapped(&f.entity),
                Vec::new(),
            ));
        }
        Err(e) => {
            return Ok(without_geometry(
                e.classification(class),
                vec![format!("{class}: {e}")],
            ));
        }
    };
    let record = match Normalized2DGeometry::from_spline(&spline) {
        Ok(r) => r,
        Err(e) => {
            return Ok(without_geometry(
                Classification::refused(ReasonCode::DegenerateGeometry, class),
                vec![format!("{class}: {e}")],
            ));
        }
    };
    promote_leaf(f, record, gate, viewport)
}

/// A record PVec can carry, checked against the gate: its capability must be allowed, and under
/// an exact-only profile its class must be `SupportedExact`.
fn gated(
    record: Normalized2DGeometry,
    prov: &ProvenanceStub,
    gate: &Gate,
) -> Result<Done, PvecConvertError> {
    let class = prov.class_name.as_str();
    let row = kind_of(&record).map(row_for);
    let capability = row.map(|r| {
        (
            CapabilityId::new(u32::from(r.capability.id)),
            r.capability.name,
        )
    });
    let equivalent = record.fidelity() == FidelityClass::SupportedEquivalent;
    // A curve shape the mapping table has no row for has no capability to check.
    if let Some((id, name)) = capability
        && (!gate.allowed.contains(&id) || (gate.exact_only && equivalent))
    {
        let mut warnings = Vec::new();
        match gate.policy {
            CapabilityPolicy::RejectInput => {
                return Err(PvecConvertError::CapabilityNotAllowed {
                    capability: id,
                    name,
                    source_class: class.to_owned(),
                    handle: prov.handle.map(|h| h.get()),
                    layer: prov.layer.clone(),
                    block_path: prov.block_path.to_string(),
                });
            }
            CapabilityPolicy::Downgrade => {
                warnings.push(format!(
                    "{class}: downgrade-unavailable: no exact weaker representation of {name}"
                ));
            }
            CapabilityPolicy::RejectEntity => {}
        }
        warnings.push(format!("{class}: needs capability {name} ({})", id.get()));
        return Ok(without_geometry(
            Classification::refused(ReasonCode::CapabilityNotAllowed, class),
            warnings,
        ));
    }
    let basis = if equivalent {
        Basis::Equivalent
    } else {
        Basis::Structural
    };
    Ok(Done {
        classification: Classification::supported(basis, class),
        geometry: record,
        warnings: Vec::new(),
        solid: false,
    })
}

/// The diagnostic for one entity, or `None` for an entity that became geometry: a supported entity
/// makes none, so a drawing that converts cleanly allocates nothing here.
fn entity_diagnostic(
    item: &Item<'_>,
    raw: &RawEntity,
    c: &Classification,
    layout: &str,
) -> Option<Diagnostic> {
    let code = match c.class() {
        FidelityClass::SupportedExact | FidelityClass::SupportedEquivalent => return None,
        _ if matches!(item, Item::Skipped(_)) => DiagnosticCode::InsertNotExpanded,
        FidelityClass::RequiresPVecCapability | FidelityClass::RequiresExtension => {
            DiagnosticCode::EntityRequiresCapability
        }
        FidelityClass::IgnoredByPolicy => DiagnosticCode::EntityIgnoredByPolicy,
        _ => DiagnosticCode::EntityUnsupported,
    };
    let id = c
        .reason()
        .map(|r| r.code())
        .or_else(|| c.policy_field())
        .unwrap_or("");
    let p = &raw.provenance;
    let mut d = Diagnostic::new(
        code,
        id,
        format!("{} {}: {id}", p.class_name, c.class().tag()),
    );
    d.layout = Some(layout.to_owned());
    d.handle = p.handle.map(|h| h.get());
    d.class = Some(p.class_name.clone());
    d.block_path = Some(p.block_path.to_string());
    d.layer = Some(p.layer.clone());
    Some(d)
}

fn diagnostic_event(d: &Diagnostic) -> EngineEvent {
    EngineEvent::ConversionDiagnostic {
        code: d.code.name().to_owned(),
        severity: format!("{:?}", d.severity).to_lowercase(),
        message_id: d.message_id.clone(),
        handle: d.handle,
        class: d.class.clone(),
        block_path: d.block_path.clone(),
        layer: d.layer.clone(),
    }
}

/// Tells the observer why a run failed, then hands the result on.
fn report_failure<T>(
    observer: Option<&dyn EngineObserver>,
    result: Result<T, PvecConvertError>,
) -> Result<T, PvecConvertError> {
    if let (Err(e), Some(o)) = (&result, observer) {
        o.on_event(diagnostic_event(&e.diagnostic()));
    }
    result
}

impl PvecConvertError {
    /// The diagnostic for this failure: its code, severity and stage, a stable message id, and the
    /// entity's handle, class, layer and block path when the failure is about one entity or one
    /// INSERT chain.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        let d = |code, id: &str| Diagnostic::new(code, id, self.to_string());
        match self {
            PvecConvertError::Open(_) => d(DiagnosticCode::SourceOpenFailed, "open"),
            PvecConvertError::View(_) => d(DiagnosticCode::ViewFailed, "view"),
            PvecConvertError::Flatten(e) => {
                let (id, path) = match e {
                    FlattenError::BlockRecursionLimit { block_path, .. } => {
                        ("block-recursion", Some(block_path))
                    }
                    FlattenError::Limit { block_path, .. } => ("limit", Some(block_path)),
                    FlattenError::EntitiesPerBlock { block_path, .. } => {
                        ("entities-per-block", Some(block_path))
                    }
                    FlattenError::Compose(_) => ("compose", None),
                    _ => ("provider", None),
                };
                let mut out = d(DiagnosticCode::BlocksFailed, id);
                out.block_path = path.map(ToString::to_string);
                out
            }
            PvecConvertError::UnitRequired => d(DiagnosticCode::UnitRequired, "unit-required"),
            PvecConvertError::Write(_) | PvecConvertError::Components(_) => {
                d(DiagnosticCode::WriteFailed, "write")
            }
            PvecConvertError::Rasters(_) => d(DiagnosticCode::WriteFailed, "rasters"),
            PvecConvertError::NothingStorable { .. } => {
                d(DiagnosticCode::WriteFailed, "nothing-storable")
            }
            PvecConvertError::Tally(_) | PvecConvertError::Report(_) => {
                d(DiagnosticCode::ReportFailed, "report")
            }
            PvecConvertError::Publish(_) => d(DiagnosticCode::PublishFailed, "publish"),
            PvecConvertError::CapabilityNotAllowed {
                source_class,
                handle,
                layer,
                block_path,
                ..
            } => {
                let mut out = d(DiagnosticCode::CapabilityRejected, "capability-not-allowed");
                out.class = Some(source_class.clone());
                out.handle = *handle;
                out.layer = Some(layer.clone());
                out.block_path = Some(block_path.clone());
                out
            }
            #[cfg(feature = "pvec-raster-thumbnail")]
            PvecConvertError::Thumbnail(_) => d(DiagnosticCode::WriteFailed, "thumbnail"),
            PvecConvertError::Cancelled => d(DiagnosticCode::Cancelled, "cancelled"),
            PvecConvertError::Deadline => d(DiagnosticCode::Cancelled, "deadline"),
        }
    }

    /// A stable, upper snake code for this failure (#1674), for a caller that has to tell one from
    /// another without reading the message. The message may change, the code does not.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            PvecConvertError::Open(_) => "OPEN_FAILED",
            PvecConvertError::View(e) => match e {
                ViewProfileError::LayoutNotFound { .. } => "LAYOUT_NOT_FOUND",
                ViewProfileError::NoModelSpace { .. } => "NO_MODEL_SPACE",
                ViewProfileError::Limit(e) => e.code(),
                _ => "VIEW_FAILED",
            },
            PvecConvertError::Flatten(e) => match e {
                FlattenError::Limit { limit, .. } => limit.code(),
                FlattenError::BlockRecursionLimit { .. } => "BLOCK_CYCLE",
                _ => "FLATTEN_FAILED",
            },
            PvecConvertError::UnitRequired => "UNIT_REQUIRED",
            PvecConvertError::Write(_) => "WRITE_FAILED",
            PvecConvertError::NothingStorable { .. } => "NOTHING_STORABLE",
            PvecConvertError::Tally(_) => "TALLY_FAILED",
            PvecConvertError::Components(_) => "COMPONENTS_FAILED",
            PvecConvertError::Rasters(_) => "RASTERS_FAILED",
            PvecConvertError::Report(_) => "REPORT_FAILED",
            PvecConvertError::Publish(_) => "PUBLISH_FAILED",
            PvecConvertError::CapabilityNotAllowed { .. } => "CAPABILITY_NOT_ALLOWED",
            #[cfg(feature = "pvec-raster-thumbnail")]
            PvecConvertError::Thumbnail(_) => "THUMBNAIL_FAILED",
            PvecConvertError::Cancelled => "CANCELLED",
            PvecConvertError::Deadline => "DEADLINE",
        }
    }

    /// `Cancelled` is `Deadline` when `cancel` fired because its time was up and nobody called
    /// `cancel()`; every other error is as it was.
    fn settled(self, cancel: &CancelToken) -> Self {
        match self {
            PvecConvertError::Cancelled if cancel.deadline_passed() => PvecConvertError::Deadline,
            other => other,
        }
    }
}
