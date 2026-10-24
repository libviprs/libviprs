//! The unit a drawing's coordinates are in, carried as metadata (N4.7, #1078).
//!
//! A CAD drawing's coordinates mean whatever its `$INSUNITS` header value says,
//! and often that value is 0 or missing. The tempting move is to normalise to
//! metres on ingest, which throws away the exactness the rest of `pvec` is built
//! on. So nothing here, or anywhere else under `src/pvec/`, scales geometry:
//! coordinates stay the `f64` values the drawing stores, bit for bit, and the unit
//! travels beside them as a [`SourceUnit`].
//!
//! What is in this module:
//!
//! * [`SourceUnit`], a closed enum over the `$INSUNITS` codes 0 to 24, plus
//!   [`SourceUnit::Unknown`] for a value that is missing or not a code. The
//!   mapping from the raw code lives in one table, [`INSUNITS_TABLE`], and the
//!   enum's methods read that table and nothing else.
//! * [`SourceSpace`], the unit plus the source-space bounds, which is the shape the
//!   georeference handoff (N7.3) takes as metadata.
//!
//! What is not here: inference from coordinate magnitude, a default unit when the
//! header says nothing, unit conversion, and any georeferencing. See
//! `docs/pvec-units.md` for the table and for what `Unknown` means downstream.

use crate::pvec::provider::Point3;

/// The unit of a drawing's coordinates, from its `$INSUNITS` header value.
///
/// Closed: a code outside the table is [`Unknown`](Self::Unknown), never a new
/// variant and never a guess.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SourceUnit {
    /// Code 0. The drawing says it has no unit. This is a statement, not a gap.
    Unitless,
    /// Code 1.
    Inches,
    /// Code 2.
    Feet,
    /// Code 3.
    Miles,
    /// Code 4.
    Millimeters,
    /// Code 5.
    Centimeters,
    /// Code 6.
    Meters,
    /// Code 7.
    Kilometers,
    /// Code 8.
    Microinches,
    /// Code 9.
    Mils,
    /// Code 10.
    Yards,
    /// Code 11.
    Angstroms,
    /// Code 12.
    Nanometers,
    /// Code 13.
    Microns,
    /// Code 14.
    Decimeters,
    /// Code 15.
    Decameters,
    /// Code 16.
    Hectometers,
    /// Code 17.
    Gigameters,
    /// Code 18.
    AstronomicalUnits,
    /// Code 19.
    LightYears,
    /// Code 20.
    Parsecs,
    /// Code 21.
    UsSurveyFeet,
    /// Code 22.
    UsSurveyInches,
    /// Code 23.
    UsSurveyYards,
    /// Code 24.
    UsSurveyMiles,
    /// The header had no `$INSUNITS` value, or it was not one of the codes above.
    /// Nothing is assumed in its place.
    Unknown,
}

/// The one place a raw `$INSUNITS` code is mapped to a unit, with the name the
/// documentation uses. Every variant except [`SourceUnit::Unknown`] has exactly
/// one row, in code order.
pub const INSUNITS_TABLE: [(i16, SourceUnit, &str); 25] = [
    (0, SourceUnit::Unitless, "Unitless"),
    (1, SourceUnit::Inches, "Inches"),
    (2, SourceUnit::Feet, "Feet"),
    (3, SourceUnit::Miles, "Miles"),
    (4, SourceUnit::Millimeters, "Millimeters"),
    (5, SourceUnit::Centimeters, "Centimeters"),
    (6, SourceUnit::Meters, "Meters"),
    (7, SourceUnit::Kilometers, "Kilometers"),
    (8, SourceUnit::Microinches, "Microinches"),
    (9, SourceUnit::Mils, "Mils"),
    (10, SourceUnit::Yards, "Yards"),
    (11, SourceUnit::Angstroms, "Angstroms"),
    (12, SourceUnit::Nanometers, "Nanometers"),
    (13, SourceUnit::Microns, "Microns"),
    (14, SourceUnit::Decimeters, "Decimeters"),
    (15, SourceUnit::Decameters, "Decameters"),
    (16, SourceUnit::Hectometers, "Hectometers"),
    (17, SourceUnit::Gigameters, "Gigameters"),
    (18, SourceUnit::AstronomicalUnits, "Astronomical units"),
    (19, SourceUnit::LightYears, "Light years"),
    (20, SourceUnit::Parsecs, "Parsecs"),
    (21, SourceUnit::UsSurveyFeet, "US survey feet"),
    (22, SourceUnit::UsSurveyInches, "US survey inches"),
    (23, SourceUnit::UsSurveyYards, "US survey yards"),
    (24, SourceUnit::UsSurveyMiles, "US survey miles"),
];

impl SourceUnit {
    /// Reads the unit from the drawing header's `$INSUNITS` value.
    ///
    /// `None` is a header with no such value, and it is [`Unknown`](Self::Unknown),
    /// exactly like a code the table does not have. It is never
    /// [`Unitless`](Self::Unitless) (that is code 0, which a drawing has to state)
    /// and never a default unit.
    #[must_use]
    pub fn from_insunits(code: Option<i16>) -> Self {
        let Some(code) = code else {
            return Self::Unknown;
        };
        INSUNITS_TABLE
            .iter()
            .find(|(c, _, _)| *c == code)
            .map_or(Self::Unknown, |(_, unit, _)| *unit)
    }

    /// The `$INSUNITS` code, or `None` for [`Unknown`](Self::Unknown), which has none.
    #[must_use]
    pub fn insunits_code(self) -> Option<i16> {
        INSUNITS_TABLE
            .iter()
            .find(|(_, unit, _)| *unit == self)
            .map(|(c, _, _)| *c)
    }

    /// The name `docs/pvec-units.md` uses for this unit.
    #[must_use]
    pub fn name(self) -> &'static str {
        INSUNITS_TABLE
            .iter()
            .find(|(_, unit, _)| *unit == self)
            .map_or("Unknown", |(_, _, name)| name)
    }

    /// Whether the drawing named a real length unit: neither
    /// [`Unitless`](Self::Unitless) nor [`Unknown`](Self::Unknown). A georeference
    /// needs one, and the handoff refuses to emit one without it.
    #[must_use]
    pub fn is_physical(self) -> bool {
        !matches!(self, Self::Unitless | Self::Unknown)
    }
}

/// The drawing's extent in source space, as the drawing's own numbers.
///
/// Minimum and maximum corners, unscaled and in the drawing's unit. It carries no
/// transform and is not a georeference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceBounds {
    /// Smallest `x`, `y`, `z` seen.
    pub min: Point3,
    /// Largest `x`, `y`, `z` seen.
    pub max: Point3,
}

/// The unit and source-space bounds of one drawing, handed on as metadata.
///
/// This is what the georeference handoff (N7.3) receives. It decides from
/// [`SourceUnit::is_physical`] whether to emit anything, so a drawing with an
/// unknown unit gets no georeference rather than a guessed one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceSpace {
    /// The unit of every coordinate in the drawing.
    pub unit: SourceUnit,
    /// The bounds in that unit, when the spatial-bounds input (N6.4) has computed them.
    pub bounds: Option<SourceBounds>,
}

impl SourceSpace {
    /// A drawing whose unit is known from its header and whose bounds are not yet.
    #[must_use]
    pub fn new(unit: SourceUnit) -> Self {
        Self { unit, bounds: None }
    }

    /// The same drawing with its bounds attached.
    #[must_use]
    pub fn with_bounds(mut self, bounds: SourceBounds) -> Self {
        self.bounds = Some(bounds);
        self
    }
}
