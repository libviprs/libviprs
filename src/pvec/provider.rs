//! The replaceable DWG provider traits (N4.1, #1072).
//!
//! Normalization reads a drawing through [`DwgProvider`] and [`EntityCursor`] and
//! never names the reader behind them, so the first provider and a native Rust
//! decoder are interchangeable and can be run against each other.
//!
//! # Who owns what
//!
//! | name | responsibility | side |
//! |---|---|---|
//! | [`DwgProvider`] | open, probe, enumerate layouts and layers | provider |
//! | [`EntityCursor`] | one deterministic traversal pass | provider |
//! | [`RawEntity`] | provider-neutral entity payload | boundary |
//! | [`LayerState`], [`LayoutState`] | off/frozen/locked, model vs paper | boundary |
//! | [`BlockRef`], [`SourceTransform`] | insert reference, 6-value affine | boundary |
//! | [`ProvenanceStub`] | handle, class name, block path, layer | boundary |
//! | [`ProviderCaps`] | what this provider can and cannot report | boundary |
//!
//! `layers()` is the one method beyond the issue's "open, probe, enumerate layouts":
//! [`LayerState`] is a boundary type, so something on the provider side has to hand
//! it over.
//!
//! # A raw entity carries parameters, never a decoder handle
//!
//! A [`RawEntity`] is the numbers the drawing stores, as the drawing stores them:
//! radians as radians, a spline's full knot vector with its repeats, an INSERT as a
//! block name plus the transform its own fields describe. A provider does not
//! expand blocks, apply layer visibility, flatten a curve, or compare two
//! coordinates. Expansion and composition are N4.4 and N3.4, and every numeric
//! rule is PVec's. Because of that [`RawEntity`] has no `PartialEq`: float equality
//! is a geometric question with a scale term, so a test that wants to know two
//! traversals agree compares the fields itself.
//!
//! # Absent is not the same as unreportable
//!
//! A knot vector can be missing from an entity because the drawing never had one
//! (a fit-point spline) or because this provider has no way to read it. Treating
//! those alike is how a curve gets resampled and called exact. [`SourceFact`] keeps
//! them apart per value, and [`ProviderCaps`] says up front which facets a provider
//! can report at all, with the source of each (the decoder adapter, or the decoder
//! directly). A consumer that meets [`SourceFact::NotReportable`] on a spline's knots
//! classifies the entity `RequiresExtension` (N3.5's enum, not redefined here) and
//! never rebuilds the curve from points.
//!
//! # Instance paths
//!
//! A leaf reached through nested INSERTs is not identified by its own handle: the
//! same handle repeats once per instance. [`InstancePath`] is the ordered chain of
//! `(block name, insert handle)` pairs from the layout down to the block holding the
//! entity, and [`ProvenanceStub::block_path`] carries it. The provider does not
//! walk the chain. The consumer that descends into a block hands the path it took
//! to [`EntityScope::BlockDefinition`], and the provider stamps it onto every
//! entity of that traversal untouched.
//!
//! # Coordinates
//!
//! Source-space `f64`, unconverted. The only value checked here is a
//! [`SourceTransform`], which is finite-checked at construction; promotion,
//! degeneracy and singular handling are P3.8's, applied in N3.4.

use std::fmt;

use thiserror::Error;

use crate::cad::{CadSource, ItemHandle};

/// A source-space point, `[x, y, z]`, as the drawing stores it.
pub type Point3 = [f64; 3];

/// Why a provider call failed.
///
/// Carries strings, never the reader's own error type, so no foreign type reaches
/// a signature.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum ProviderError {
    /// The bytes are not a drawing this provider reads.
    #[error("{provider}: not a drawing this provider reads: {detail}")]
    Unrecognised {
        /// The provider's name.
        provider: &'static str,
        /// What was wrong with the source.
        detail: String,
    },
    /// The reader refused the drawing after recognising it.
    #[error("{provider}: could not open the drawing: {detail}")]
    Open {
        /// The provider's name.
        provider: &'static str,
        /// The reader's message.
        detail: String,
    },
    /// The provider has no way to report this facet at all, see [`ProviderCaps`].
    #[error("this provider cannot report {0:?}")]
    CannotReport(Facet),
    /// A coefficient that is not finite was offered to a [`SourceTransform`].
    #[error("non-finite value in {field}")]
    NonFinite {
        /// Which coefficient.
        field: &'static str,
    },
    /// The scope names a layout or block the drawing does not have.
    #[error("no such scope: {0}")]
    NoSuchScope(String),
}

/// Where an entity traversal starts.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntityScope {
    /// The entities a layout owns: model space, or one paper layout.
    Layout(LayoutId),
    /// A block definition's entities, reached through `reached_via`.
    ///
    /// `reached_via` is stamped onto every [`ProvenanceStub::block_path`] of the
    /// traversal as given. The provider does not check it against the drawing.
    BlockDefinition {
        /// The block's name.
        name: String,
        /// The INSERT chain the consumer followed to get here.
        reached_via: InstancePath,
    },
}

/// A layout's position in [`DwgProvider::layouts`], stable for one opened drawing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayoutId(pub u32);

/// Model space, or a paper-space layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LayoutKind {
    /// Model space.
    Model,
    /// A paper-space layout.
    Paper,
}

/// One layout of the drawing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutState {
    /// The id to open it with.
    pub id: LayoutId,
    /// The layout's own name (`Model`, `Layout1`).
    pub name: String,
    /// Model space or paper space.
    pub kind: LayoutKind,
    /// The block record that owns the layout's entities (`*Model_Space`).
    pub block_name: String,
}

/// One layer's state flags, as the layer table stores them.
///
/// Not applied to anything. Which of these hides an entity is the visible-drawing
/// rule's (N1.4), and a viewport freeze is not here at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerState {
    /// The layer's name.
    pub name: String,
    /// The layer's own handle, when the drawing gives one.
    pub handle: Option<ItemHandle>,
    /// Switched off.
    pub off: bool,
    /// Frozen.
    pub frozen: bool,
    /// Locked.
    pub locked: bool,
}

/// A value the drawing may not have, or the provider may not be able to read.
///
/// The three arms are the whole point: [`Absent`](Self::Absent) is a fact about the
/// drawing, [`NotReportable`](Self::NotReportable) is a fact about the provider, and
/// they lead to different outcomes downstream.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SourceFact<T> {
    /// The drawing has it and here it is.
    Present(T),
    /// The drawing does not have it.
    Absent,
    /// The drawing may well have it, and this provider cannot read it.
    NotReportable,
}

impl<T> SourceFact<T> {
    /// The value, when there is one.
    #[must_use]
    pub fn present(&self) -> Option<&T> {
        match self {
            Self::Present(v) => Some(v),
            _ => None,
        }
    }

    /// True for [`SourceFact::NotReportable`].
    #[must_use]
    pub fn is_not_reportable(&self) -> bool {
        matches!(self, Self::NotReportable)
    }
}

/// The six coefficients of a 2D affine map, finite-checked.
///
/// Row-major, translation last, the order of `Transform2::new` in the N1.5 harness:
/// `[m00, m01, m10, m11, tx, ty]`, so `x' = m00*x + m01*y + tx` and
/// `y' = m10*x + m11*y + ty`. There is no `PartialEq`: two maps are equal under the
/// numeric policy, not under `==`.
#[derive(Debug, Clone, Copy)]
pub struct SourceTransform {
    coefficients: [f64; 6],
}

const COEFFICIENT_NAMES: [&str; 6] = ["m00", "m01", "m10", "m11", "tx", "ty"];

impl SourceTransform {
    /// The identity map.
    pub const IDENTITY: Self = Self {
        coefficients: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    };

    /// Wraps six coefficients, refusing a NaN or an infinity.
    ///
    /// # Errors
    ///
    /// [`ProviderError::NonFinite`] naming the first coefficient that is not finite.
    /// Nothing else is checked: a singular or near-singular map is N3.4's to refuse
    /// under PVec's numeric policy.
    pub fn new(coefficients: [f64; 6]) -> Result<Self, ProviderError> {
        unimplemented!("N4.1 red stub")
    }

    /// The six coefficients, in the order above.
    #[must_use]
    pub const fn coefficients(&self) -> [f64; 6] {
        self.coefficients
    }
}

/// The grid an INSERT places its block on (a MINSERT is rows times columns).
///
/// A plain INSERT is one row, one column, zero spacing. The spacings are in the
/// INSERT's own units and are not applied to [`BlockRef::transform`].
#[derive(Debug, Clone, Copy)]
pub struct ArrayGrid {
    /// Rows.
    pub rows: u32,
    /// Columns.
    pub columns: u32,
    /// Distance between rows.
    pub row_spacing: f64,
    /// Distance between columns.
    pub column_spacing: f64,
}

impl ArrayGrid {
    /// One instance, no spacing.
    pub const SINGLE: Self = Self {
        rows: 1,
        columns: 1,
        row_spacing: 0.0,
        column_spacing: 0.0,
    };
}

/// An INSERT, as a reference to a block and the map its own fields describe.
///
/// `transform` is the placement of the grid's first instance: scale, then rotate,
/// then translate by the insertion point, with the block's base point taken out
/// first. It is in the INSERT's own coordinate system: `normal` is reported
/// verbatim, and an INSERT whose normal is not the Z axis needs the arbitrary-axis
/// step N4.4 owns.
#[derive(Debug, Clone)]
pub struct BlockRef {
    /// The referenced block's name, as the INSERT spells it.
    pub block_name: String,
    /// The map from the block's coordinates to the INSERT's container.
    pub transform: SourceTransform,
    /// The extrusion direction of the INSERT.
    pub normal: Point3,
    /// MINSERT rows and columns.
    pub grid: ArrayGrid,
}

/// One step of an [`InstancePath`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstanceStep {
    /// The block the INSERT referenced.
    pub block_name: String,
    /// The INSERT's own handle.
    pub insert_handle: ItemHandle,
}

/// The ordered chain of INSERTs between a layout and an entity, outermost first.
///
/// Empty for an entity a layout owns directly. Equality here compares names and
/// handles, which are exact identities, so deriving it is fine.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct InstancePath {
    steps: Vec<InstanceStep>,
}

impl InstancePath {
    /// The empty path.
    #[must_use]
    pub fn root() -> Self {
        Self::default()
    }

    /// This path with one more INSERT at the end.
    #[must_use]
    pub fn child(&self, block_name: impl Into<String>, insert_handle: ItemHandle) -> Self {
        unimplemented!("N4.1 red stub")
    }

    /// The steps, outermost first.
    #[must_use]
    pub fn steps(&self) -> &[InstanceStep] {
        &self.steps
    }

    /// How many INSERTs deep.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.steps.len()
    }

    /// True for the empty path.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.steps.is_empty()
    }
}

impl fmt::Display for InstancePath {
    /// `VC_OUTER@58/VC_MID@55`, or `/` for the empty path. Handles are hex.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        unimplemented!("N4.1 red stub")
    }
}

/// Where an entity came from, enough to find it in the drawing again.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceStub {
    /// The entity's own handle, when the drawing gives one.
    pub handle: Option<ItemHandle>,
    /// The drawing's class name for it (`LINE`, `LWPOLYLINE`).
    pub class_name: String,
    /// The INSERT chain the traversal was reached through.
    pub block_path: InstancePath,
    /// The entity's own layer name. Not the effective layer: a leaf on layer "0"
    /// inside an INSERT takes its INSERT's layer, and applying that is the
    /// consumer's (N1.4, V1-2).
    pub layer: String,
}

/// One vertex of a polyline: position and the bulge toward the next vertex.
#[derive(Debug, Clone, Copy)]
pub struct RawVertex {
    /// X.
    pub x: f64,
    /// Y.
    pub y: f64,
    /// The bulge (the tangent of a quarter of the included angle), verbatim.
    pub bulge: f64,
}

/// The geometry parameters of one entity.
///
/// Angles are radians as stored. Closed polylines do not repeat their first vertex,
/// they set `closed`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum RawGeometry {
    /// A line segment.
    Line {
        /// Start.
        start: Point3,
        /// End.
        end: Point3,
    },
    /// A polyline with optional bulges.
    Polyline {
        /// The vertices, in order.
        vertices: Vec<RawVertex>,
        /// Whether the last vertex joins the first.
        closed: bool,
        /// Z of the polyline's plane.
        elevation: f64,
        /// Extrusion direction.
        normal: Point3,
    },
    /// A circular arc, counter-clockwise from `start_angle` to `end_angle`.
    Arc {
        /// Centre.
        center: Point3,
        /// Radius.
        radius: f64,
        /// Start angle, radians.
        start_angle: f64,
        /// End angle, radians.
        end_angle: f64,
        /// Extrusion direction.
        normal: Point3,
    },
    /// A full circle.
    Circle {
        /// Centre.
        center: Point3,
        /// Radius.
        radius: f64,
        /// Extrusion direction.
        normal: Point3,
    },
    /// An ellipse or elliptical arc.
    Ellipse {
        /// Centre.
        center: Point3,
        /// The major semi-axis as a vector from the centre.
        major_axis: Point3,
        /// Minor over major.
        minor_ratio: f64,
        /// Start parameter, radians.
        start_parameter: f64,
        /// End parameter, radians.
        end_parameter: f64,
        /// Extrusion direction.
        normal: Point3,
    },
    /// A B-spline, with the parts the ISO profile needs kept apart.
    Spline {
        /// Degree.
        degree: u32,
        /// Control points.
        control_points: Vec<Point3>,
        /// The full knot vector with its repeats, which is how multiplicity travels.
        knots: SourceFact<Vec<f64>>,
        /// One weight per control point. [`SourceFact::Absent`] means not rational.
        weights: SourceFact<Vec<f64>>,
        /// Whether the spline is periodic.
        periodic: SourceFact<bool>,
        /// Whether the spline is closed.
        closed: bool,
    },
    /// An INSERT. Not expanded.
    Insert(BlockRef),
    /// An entity the provider saw and does not map. Counted, never dropped.
    Unmapped {
        /// Why it is not mapped.
        reason: String,
    },
}

/// One entity as the boundary carries it.
///
/// No `PartialEq`, see the module docs.
#[derive(Debug, Clone)]
pub struct RawEntity {
    /// What it is.
    pub geometry: RawGeometry,
    /// Where it came from.
    pub provenance: ProvenanceStub,
}

impl RawEntity {
    /// The path a consumer takes into this entity's block, when it is an INSERT with
    /// a handle: this entity's own path plus one step.
    #[must_use]
    pub fn child_path(&self) -> Option<InstancePath> {
        unimplemented!("N4.1 red stub")
    }

    /// The facets this entity carries as [`SourceFact::NotReportable`].
    #[must_use]
    pub fn not_reportable(&self) -> Vec<Facet> {
        unimplemented!("N4.1 red stub")
    }

    /// The facets this entity carries as present or absent, so the provider read them.
    #[must_use]
    pub fn reported(&self) -> Vec<Facet> {
        unimplemented!("N4.1 red stub")
    }
}

/// Something a provider may or may not be able to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Facet {
    /// [`LayerState`] from [`DwgProvider::layers`].
    LayerState,
    /// [`LayoutState`] from [`DwgProvider::layouts`].
    LayoutState,
    /// A spline's knot vector with its repeats.
    KnotMultiplicity,
    /// A spline's weights.
    Weights,
    /// A spline's periodicity.
    Periodicity,
}

impl Facet {
    /// Every facet.
    pub const ALL: [Facet; 5] = [
        Facet::LayerState,
        Facet::LayoutState,
        Facet::KnotMultiplicity,
        Facet::Weights,
        Facet::Periodicity,
    ];
}

/// Whether a provider can read a facet at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Availability {
    /// The provider reads it. When the drawing has it the value is
    /// [`SourceFact::Present`]; when it does not, [`SourceFact::Absent`]. Never
    /// [`SourceFact::NotReportable`].
    Reported,
    /// The provider cannot read it. Every value is [`SourceFact::NotReportable`],
    /// whether or not the drawing has one.
    CannotReport,
}

/// Which layer of the stack a facet is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FacetSource {
    /// Through the existing decoder adapter ([`CadDecoder`](crate::cad::CadDecoder)).
    DecoderAdapter,
    /// From the decoder directly, because the adapter does not expose it.
    DecoderDirect,
}

/// One row of [`ProviderCaps`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FacetCap {
    /// The facet.
    pub facet: Facet,
    /// Whether it can be read.
    pub availability: Availability,
    /// Where it is read from.
    pub source: FacetSource,
}

/// What a provider can and cannot report.
///
/// A facet with no row is [`Availability::CannotReport`]: a provider has to claim a
/// capability to have it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProviderCaps {
    rows: Vec<FacetCap>,
}

impl ProviderCaps {
    /// No capabilities.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the row for `facet`, replacing an earlier one.
    #[must_use]
    pub fn with(mut self, facet: Facet, availability: Availability, source: FacetSource) -> Self {
        unimplemented!("N4.1 red stub")
    }

    /// The row for `facet`, when the provider declared one.
    #[must_use]
    pub fn get(&self, facet: Facet) -> Option<&FacetCap> {
        self.rows.iter().find(|r| r.facet == facet)
    }

    /// Whether `facet` can be read. No row means it cannot.
    #[must_use]
    pub fn availability(&self, facet: Facet) -> Availability {
        unimplemented!("N4.1 red stub")
    }

    /// The rows, ordered by facet.
    #[must_use]
    pub fn rows(&self) -> &[FacetCap] {
        &self.rows
    }

    /// Facets where `entity` contradicts these caps: a value marked
    /// [`SourceFact::NotReportable`] under a facet the provider claims to read, or a
    /// present or absent value under one it says it cannot.
    #[must_use]
    pub fn violations(&self, entity: &RawEntity) -> Vec<Facet> {
        unimplemented!("N4.1 red stub")
    }
}

/// One deterministic pass over a scope's entities.
///
/// Two cursors opened on the same drawing with the same scope yield the same
/// sequence, field by field: the order is the drawing's own storage order, never a
/// hash order or a thread race. A cursor is fused: after `Ok(None)` it keeps
/// returning `Ok(None)`.
pub trait EntityCursor {
    /// The next entity, `Ok(None)` at the end.
    ///
    /// # Errors
    ///
    /// A [`ProviderError`] when the reader fails partway. The cursor stays usable
    /// only if the provider documents it.
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError>;

    /// Drains the cursor.
    ///
    /// # Errors
    ///
    /// The first error [`next_entity`](Self::next_entity) returns.
    fn collect_all(&mut self) -> Result<Vec<RawEntity>, ProviderError> {
        unimplemented!("N4.1 red stub")
    }
}

/// A reader of DWG drawings, as normalization sees it.
///
/// The type is the opened drawing: [`open`](Self::open) returns `Self`, so there is
/// no "not open yet" state to check for. The instance methods are object safe, so
/// a consumer can hold `&dyn DwgProvider`, and the two constructors are
/// `where Self: Sized` so a differential run can be generic over `P: DwgProvider`.
pub trait DwgProvider {
    /// Reads only enough of `source` to say whether this provider would open it.
    ///
    /// # Errors
    ///
    /// [`ProviderError`] when the source cannot be read at all. A readable file in
    /// a format this provider does not take is `Ok(false)`.
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError>
    where
        Self: Sized;

    /// Opens a drawing. The returned provider owns what it needs, so `source` does
    /// not have to outlive it.
    ///
    /// # Errors
    ///
    /// [`ProviderError::Unrecognised`] or [`ProviderError::Open`].
    fn open(source: CadSource<'_>) -> Result<Self, ProviderError>
    where
        Self: Sized;

    /// This provider's name, for reports.
    fn name(&self) -> &'static str;

    /// What this provider can and cannot report.
    fn caps(&self) -> &ProviderCaps;

    /// Every layout, model space first when the drawing has one, then paper layouts
    /// in tab order. The order is stable for one opened drawing.
    ///
    /// # Errors
    ///
    /// [`ProviderError::CannotReport`] with [`Facet::LayoutState`] when the caps say so.
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError>;

    /// The layer table, in the drawing's own order.
    ///
    /// # Errors
    ///
    /// [`ProviderError::CannotReport`] with [`Facet::LayerState`] when the caps say so.
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError>;

    /// Opens one traversal pass.
    ///
    /// # Errors
    ///
    /// [`ProviderError::NoSuchScope`] when the layout or block is not in the drawing.
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError>;
}
