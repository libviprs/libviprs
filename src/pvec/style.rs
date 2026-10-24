//! The monochrome line-style component (issue #1083, N5.4).
//!
//! Lineweight and dash pattern are the two visual properties a monochrome blueprint needs, and
//! they are not geometry. If they were fields on the geometry entity, every mathematical
//! conformance test would be coupled to them, so they live here as an independent, optional
//! component that points at entities by [`EntityId`] and nothing else:
//!
//! ```text
//! PdfVectorPage (w, d, colour ops)  --pdf_line_styles-->  StyleBatch
//! DwgStyleInput (lineweight, linetype) --dwg_line_style--> LineStyleComponent
//! ```
//!
//! No type in `pdf_vector`'s or `normalized`'s geometry carries a style field beyond the raw
//! `w`/`d` record the PDF adapter keeps next to the path, and `Normalized2DGeometry` has none at
//! all (`tests/pvec_style_1083.rs` pins its size).
//!
//! # Values
//!
//! Widths and dashes are `f64` in source units: PDF user-space units for `w`/`d` (the CTM is
//! not applied), hundredths of a millimetre converted to millimetres for a DWG lineweight.
//!
//! * `w 0` is [`StrokeWidth::Thinnest`], the PDF thinnest-renderable-line meaning. It is not a
//!   zero width, and it is not an absent width (`width: None`).
//! * An empty dash array is an explicit solid line, distinct from no dash at all.
//! * Degenerate data is refused with a [`StyleError`], never normalized: a negative or
//!   non-finite dash entry (naming its index), an all-zero dash array, a non-finite or negative
//!   width, a non-finite phase.
//!
//! # Monochrome v1
//!
//! Colour operators are parsed and counted and land in the [`StyleReport`] as
//! `IgnoredByPolicy { reason: MonochromeProfile }`. They produce no component.
//!
//! # What is not here
//!
//! Writing the components into PVec's `LineStyleStream` is not done: that stream has no width,
//! phase or visibility field today, so the follow-up is tracked on the issue. The DWG side takes
//! plain values ([`DwgStyleInput`]) because the provider boundary does not report lineweight or
//! linetype yet.

use std::collections::BTreeMap;
use std::fmt;

use super::fidelity::Classification;
use super::ordering::EntityId;
use super::pdf_vector::PdfVectorPage;

/// Why a style value was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StyleError {
    /// A dash entry is negative.
    NegativeDash {
        /// Its position in the array.
        index: usize,
    },
    /// A dash entry is NaN or infinite.
    NonFiniteDash {
        /// Its position in the array.
        index: usize,
    },
    /// Every dash entry is zero.
    AllZeroDash,
    /// The dash phase is NaN or infinite.
    NonFinitePhase,
    /// The width is NaN or infinite.
    NonFiniteWidth,
    /// The width is negative.
    NegativeWidth,
    /// The linetype scale is not a finite, positive number.
    BadLinetypeScale,
}

impl fmt::Display for StyleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StyleError::NegativeDash { index } => write!(f, "dash entry {index} is negative"),
            StyleError::NonFiniteDash { index } => write!(f, "dash entry {index} is not finite"),
            StyleError::AllZeroDash => f.write_str("every dash entry is zero"),
            StyleError::NonFinitePhase => f.write_str("dash phase is not finite"),
            StyleError::NonFiniteWidth => f.write_str("stroke width is not finite"),
            StyleError::NegativeWidth => f.write_str("stroke width is negative"),
            StyleError::BadLinetypeScale => {
                f.write_str("linetype scale is not finite and positive")
            }
        }
    }
}

impl std::error::Error for StyleError {}

/// A stroke width that was present in the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StrokeWidth {
    /// `w 0` (or a DWG lineweight of 0.00): the thinnest line the device can draw. Not a
    /// geometric zero.
    Thinnest,
    /// A positive width in source units.
    Width(f64),
}

/// A validated dash pattern: alternating on and off lengths, and a phase.
#[derive(Debug, Clone, PartialEq)]
pub struct DashPattern {
    array: Vec<f64>,
    phase: f64,
}

impl DashPattern {
    /// Validates a dash array and phase. An empty array is a solid line.
    ///
    /// # Errors
    /// [`StyleError::NonFiniteDash`] or [`StyleError::NegativeDash`] for the first bad entry,
    /// [`StyleError::AllZeroDash`] for a non-empty all-zero array, [`StyleError::NonFinitePhase`].
    pub fn new(array: Vec<f64>, phase: f64) -> Result<Self, StyleError> {
        for (index, v) in array.iter().enumerate() {
            if !v.is_finite() {
                return Err(StyleError::NonFiniteDash { index });
            }
            if *v < 0.0 {
                return Err(StyleError::NegativeDash { index });
            }
        }
        if !array.is_empty() && array.iter().all(|v| *v == 0.0) {
            return Err(StyleError::AllZeroDash);
        }
        if !phase.is_finite() {
            return Err(StyleError::NonFinitePhase);
        }
        Ok(DashPattern { array, phase })
    }

    /// The on/off lengths.
    #[must_use]
    pub fn array(&self) -> &[f64] {
        &self.array
    }

    /// The phase.
    #[must_use]
    pub fn phase(&self) -> f64 {
        self.phase
    }
}

/// One entity's line style. There is no colour field, by policy.
#[derive(Debug, Clone, PartialEq)]
pub struct LineStyleComponent {
    /// The entity it describes.
    pub entity: EntityId,
    /// The width, `None` when the source set none.
    pub width: Option<StrokeWidth>,
    /// The dash pattern, `None` when the source set none.
    pub dash: Option<DashPattern>,
    /// Whether the source marked the entity visible.
    pub visible: bool,
}

impl LineStyleComponent {
    /// Builds a component from a raw width (`0.0` becomes [`StrokeWidth::Thinnest`]).
    ///
    /// # Errors
    /// [`StyleError::NonFiniteWidth`] or [`StyleError::NegativeWidth`].
    pub fn new(
        entity: EntityId,
        width: Option<f64>,
        dash: Option<DashPattern>,
        visible: bool,
    ) -> Result<Self, StyleError> {
        let width = width.map(width_of).transpose()?;
        Ok(LineStyleComponent {
            entity,
            width,
            dash,
            visible,
        })
    }
}

fn width_of(w: f64) -> Result<StrokeWidth, StyleError> {
    if !w.is_finite() {
        Err(StyleError::NonFiniteWidth)
    } else if w < 0.0 {
        Err(StyleError::NegativeWidth)
    } else if w == 0.0 {
        Ok(StrokeWidth::Thinnest)
    } else {
        Ok(StrokeWidth::Width(w))
    }
}

/// The only reason a colour operator is dropped in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StylePolicyReason {
    /// The monochrome v1 profile carries no colour.
    MonochromeProfile,
}

impl StylePolicyReason {
    /// The profile field responsible, in the spelling `IgnoredByPolicy` rows use.
    #[must_use]
    pub fn field(self) -> &'static str {
        match self {
            StylePolicyReason::MonochromeProfile => "profile.monochrome",
        }
    }
}

impl fmt::Display for StylePolicyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StylePolicyReason::MonochromeProfile => f.write_str("MonochromeProfile"),
        }
    }
}

/// One row of colour that was seen and dropped: `IgnoredByPolicy { reason }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IgnoredColour {
    /// Why.
    pub reason: StylePolicyReason,
    /// The operator (`rg`) or the DWG colour source.
    pub source_class: String,
    /// How many were seen.
    pub count: u64,
}

impl IgnoredColour {
    /// The fidelity classification of one of these.
    #[must_use]
    pub fn classification(&self) -> Classification {
        Classification::ignored_by_policy(self.reason.field(), self.source_class.as_str())
    }
}

/// What the style pass saw and dropped.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StyleReport {
    colour: BTreeMap<String, u64>,
}

impl StyleReport {
    /// Counts `n` colour operators or indices of `source_class`.
    pub fn count_colour(&mut self, source_class: &str, n: u64) {
        if n > 0 {
            *self.colour.entry(source_class.to_owned()).or_insert(0) += n;
        }
    }

    /// Every colour operator or index dropped.
    #[must_use]
    pub fn colour_total(&self) -> u64 {
        self.colour.values().sum()
    }

    /// One row per source class, in a fixed order.
    pub fn colour_rows(&self) -> impl Iterator<Item = IgnoredColour> + '_ {
        self.colour.iter().map(|(k, n)| IgnoredColour {
            reason: StylePolicyReason::MonochromeProfile,
            source_class: k.clone(),
            count: *n,
        })
    }
}

/// The result of styling one page.
#[derive(Debug, Clone, PartialEq)]
pub struct StyleBatch {
    /// The components written, in entity order.
    pub components: Vec<LineStyleComponent>,
    /// Entities whose style data was degenerate, with the refusal. They get no component.
    pub refused: Vec<(EntityId, StyleError)>,
    /// What was counted and dropped.
    pub report: StyleReport,
}

/// The ids do not line up with the page's entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdCountMismatch {
    /// Entities on the page.
    pub entities: usize,
    /// Ids supplied.
    pub ids: usize,
}

impl fmt::Display for IdCountMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ids for {} entities", self.ids, self.entities)
    }
}

impl std::error::Error for IdCountMismatch {}

/// Builds line-style components for a page. `ids[i]` is the id of `page.entities[i]`.
///
/// Only stroked paths get a component, and only when `w` or `d` was in scope: a stroke with
/// neither has nothing to record. The page is only read, so the geometry is untouched whether
/// or not this runs.
///
/// # Errors
/// [`IdCountMismatch`] when `ids` is not one per entity.
pub fn pdf_line_styles(
    page: &PdfVectorPage,
    ids: &[EntityId],
) -> Result<StyleBatch, IdCountMismatch> {
    if ids.len() != page.entities.len() {
        return Err(IdCountMismatch {
            entities: page.entities.len(),
            ids: ids.len(),
        });
    }
    let mut batch = StyleBatch {
        components: Vec::new(),
        refused: Vec::new(),
        report: StyleReport::default(),
    };
    for (op, n) in &page.colour_operators {
        batch.report.count_colour(op, *n);
    }
    for (entity, id) in page.entities.iter().zip(ids) {
        let p = &entity.stroke_params;
        if !entity.paint.stroke || (p.width.is_none() && p.dash.is_none()) {
            continue;
        }
        let dash = p
            .dash
            .as_ref()
            .map(|d| DashPattern::new(d.array.clone(), d.phase))
            .transpose();
        match dash.and_then(|dash| LineStyleComponent::new(*id, p.width, dash, true)) {
            Ok(c) => batch.components.push(c),
            Err(e) => batch.refused.push((*id, e)),
        }
    }
    Ok(batch)
}

/// A DWG lineweight value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DwgLineweight {
    /// Inherited from the layer. Resolved elsewhere, so the component carries no width.
    ByLayer,
    /// Inherited from the block. Resolved elsewhere.
    ByBlock,
    /// The drawing's default weight. Resolved elsewhere.
    Default,
    /// Hundredths of a millimetre, as DWG stores it. `0` is the thinnest line.
    Hundredths(i16),
}

/// The style facts of one DWG entity, as plain values.
#[derive(Debug, Clone, PartialEq)]
pub struct DwgStyleInput {
    /// The lineweight, `None` when the provider reported none.
    pub lineweight: Option<DwgLineweight>,
    /// The linetype pattern: positive is pen down, negative pen up, zero a dot.
    pub linetype_elements: Option<Vec<f64>>,
    /// The linetype scale (entity scale times global scale).
    pub linetype_scale: f64,
    /// DWG colour index, counted and dropped under the monochrome profile.
    pub colour_index: Option<i16>,
    /// The entity's own visibility flag.
    pub visible: bool,
}

impl Default for DwgStyleInput {
    fn default() -> Self {
        DwgStyleInput {
            lineweight: None,
            linetype_elements: None,
            linetype_scale: 1.0,
            colour_index: None,
            visible: true,
        }
    }
}

/// Builds one DWG entity's component. The colour index is ignored here; count it with
/// [`StyleReport::count_colour`].
///
/// A signed DWG linetype becomes alternating on/off lengths scaled by `linetype_scale`; two
/// consecutive pen-down (or pen-up) elements get a zero-length element of the other kind
/// between them, and a leading gap gets a zero-length dash in front.
///
/// # Errors
/// Any [`StyleError`] the resulting width or dash earns, and [`StyleError::BadLinetypeScale`].
pub fn dwg_line_style(
    entity: EntityId,
    input: &DwgStyleInput,
) -> Result<LineStyleComponent, StyleError> {
    let width = match input.lineweight {
        Some(DwgLineweight::Hundredths(h)) if h >= 0 => Some(f64::from(h) / 100.0),
        _ => None,
    };
    let dash = match &input.linetype_elements {
        None => None,
        Some(elements) => {
            if !(input.linetype_scale.is_finite() && input.linetype_scale > 0.0) {
                return Err(StyleError::BadLinetypeScale);
            }
            let mut out: Vec<f64> = Vec::with_capacity(elements.len());
            // `on` is the kind the next array slot must hold: even slots are on.
            for e in elements {
                let pen_down = *e >= 0.0 || e.is_nan();
                let slot_on = out.len().is_multiple_of(2);
                if pen_down != slot_on {
                    out.push(0.0);
                }
                out.push(e.abs() * input.linetype_scale);
            }
            Some(DashPattern::new(out, 0.0)?)
        }
    };
    LineStyleComponent::new(entity, width, dash, input.visible)
}
