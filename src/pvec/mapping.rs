//! Red stub (#1089).
use super::FidelityClass;
use super::Normalized2DGeometry;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizedKind {
    Line,
    Circle,
    CircularArc,
    Ellipse,
    EllipticalArc,
    Polyline,
    Composite,
    Bezier,
    BSpline,
    RationalBSpline,
}
impl NormalizedKind {
    pub const ALL: [NormalizedKind; 10] = [
        Self::Line,
        Self::Circle,
        Self::CircularArc,
        Self::Ellipse,
        Self::EllipticalArc,
        Self::Polyline,
        Self::Composite,
        Self::Bezier,
        Self::BSpline,
        Self::RationalBSpline,
    ];
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformanceClass {
    StructuralExactness,
    GeometricExactness,
    DisplayFidelity,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsentCapability {
    RequiresPVecCapability,
    RequiresExtension,
    Unsupported,
}
impl AbsentCapability {
    pub const ALL: [AbsentCapability; 3] = [
        Self::RequiresPVecCapability,
        Self::RequiresExtension,
        Self::Unsupported,
    ];
    pub const fn fidelity(self) -> FidelityClass {
        FidelityClass::Unsupported
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityId {
    pub id: u16,
    pub name: &'static str,
}
#[derive(Debug, Clone, Copy)]
pub struct MappingRow {
    pub kind: NormalizedKind,
    pub pvec_type: &'static str,
    pub conformance: ConformanceClass,
    pub capability: CapabilityId,
    pub absent: AbsentCapability,
}
impl MappingRow {
    pub const fn fidelity_when_present(&self) -> FidelityClass {
        FidelityClass::SupportedExact
    }
}
pub const NORMALIZED_TO_PVEC: [MappingRow; 0] = [];
pub const fn row_index(kind: NormalizedKind) -> usize {
    match kind {
        _ => 0,
    }
}
pub fn row_for(_kind: NormalizedKind) -> &'static MappingRow {
    NORMALIZED_TO_PVEC.first().expect("red stub: the table is empty")
}
pub fn kind_of(_g: &Normalized2DGeometry) -> Option<NormalizedKind> {
    None
}
pub fn render_markdown() -> String {
    String::new()
}
