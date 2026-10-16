//! Issue #1108: the `cad` names re-exported at the crate root have to be
//! closed under the signatures of their own members.
//!
//! `PrimitiveSink` used to be at the root while `Primitive`, its one
//! required method's argument, was not, so a consumer could name the trait
//! and not implement it. `DecodeReport` was readable only through types that
//! lived behind `libviprs::cad::`. Every other test imports from
//! `libviprs::cad::`, which is exactly why nothing noticed.

use std::collections::BTreeSet;

// Only `libviprs::` names below. If one of these stops resolving, the root set
// is no longer closed.
use libviprs::{
    CadDecoder, CadDrawing, CadError, CadSource, CadView, DecodeReport, Diagnostic, DiagnosticCode,
    Extents, ItemHandle, Primitive, PrimitiveCounts, PrimitiveKind, PrimitiveSink, Severity,
    ViewKind,
};

/// The L1.4 consumer shape: a sink implemented, a report read, a view
/// described, all without a `libviprs::cad::` import.
struct Tally {
    seen: Vec<PrimitiveKind>,
}

impl PrimitiveSink for Tally {
    fn primitive(&mut self, primitive: Primitive) -> Result<(), CadError> {
        self.seen.push(primitive.kind());
        Ok(())
    }
}

fn read_report(report: &DecodeReport) -> (u64, usize, Option<Severity>, Option<DiagnosticCode>) {
    let counts: &PrimitiveCounts = report.counts();
    let line = counts.get(PrimitiveKind::Line);
    let first: Option<&Diagnostic> = report.diagnostics().first();
    (
        line,
        report.diagnostics().len(),
        first.map(Diagnostic::severity),
        first.map(Diagnostic::code),
    )
}

fn describe(view: &CadView) -> (ViewKind, Option<Extents>) {
    (view.kind(), view.extents())
}

fn entity_of(diagnostic: &Diagnostic) -> Option<ItemHandle> {
    diagnostic.entity()
}

// A decoder written from the root alone has to type-check too.
#[allow(dead_code)]
fn decoder_shape<D: CadDecoder>(d: &D, source: CadSource<'_>) -> Result<(), CadError> {
    let drawing = d.open(source)?;
    let drawing: &dyn CadDrawing = &*drawing;
    let _views: Vec<CadView> = drawing.views()?;
    Ok(())
}

#[test]
fn a_consumer_can_implement_and_read_the_contract_from_the_root_alone() {
    let mut sink = Tally { seen: Vec::new() };
    let line = libviprs::cad::Line::new([0.0; 3], [1.0, 0.0, 0.0]).unwrap();
    sink.primitive(line.into()).unwrap();
    assert_eq!(sink.seen, [PrimitiveKind::Line]);

    let mut report = DecodeReport::new();
    report.record(PrimitiveKind::Line);
    report.push(Diagnostic::warning(
        DiagnosticCode::EMPTY_VIEW,
        "nothing here",
    ));
    let (lines, n, severity, code) = read_report(&report);
    assert_eq!((lines, n), (1, 1));
    assert_eq!(severity, Some(Severity::Warning));
    assert_eq!(code, Some(DiagnosticCode::EMPTY_VIEW));

    let view = CadView::new(0, 0, [0.0, 0.0, 1.0, 1.0], 0, "Model");
    let (_kind, extents) = describe(&view);
    assert!(extents.is_some());
    assert_eq!(entity_of(&report.diagnostics()[0]), None);
}

/// The public signatures in `decoder.rs` and `report.rs`, before their test
/// modules, which is where the crate-root members are declared.
fn member_signatures() -> String {
    let mut out = String::new();
    for src in [
        include_str!("../src/cad/decoder.rs"),
        include_str!("../src/cad/report.rs"),
    ] {
        let production = src.split("#[cfg(test)]").next().unwrap();
        let mut in_sig = false;
        for line in production.lines() {
            let t = line.trim_start();
            if !in_sig
                && (t.starts_with("pub fn ")
                    || t.starts_with("pub const fn ")
                    || t.starts_with("fn "))
            {
                in_sig = true;
            }
            if in_sig {
                out.push_str(t);
                out.push('\n');
                if t.contains('{') || t.ends_with(';') {
                    in_sig = false;
                }
            }
        }
    }
    out
}

fn camel_idents(text: &str) -> BTreeSet<String> {
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.chars().next().is_some_and(char::is_uppercase))
        .filter(|w| w.chars().any(char::is_lowercase))
        .map(str::to_owned)
        .collect()
}

/// Names `cad/mod.rs` publishes, flattened out of its `pub use` blocks.
fn cad_public_names() -> BTreeSet<String> {
    let src = include_str!("../src/cad/mod.rs");
    let mut names = BTreeSet::new();
    let mut rest = src;
    while let Some(at) = rest.find("pub use ") {
        let tail = &rest[at..];
        let end = tail.find(';').unwrap();
        names.extend(camel_idents(&tail[..end]));
        rest = &tail[end..];
    }
    names.insert("CadError".to_owned());
    names
}

fn root_cad_names() -> BTreeSet<String> {
    let lib = include_str!("../src/lib.rs");
    let at = lib.find("pub use cad::{").expect("the cad re-export");
    let tail = &lib[at..];
    camel_idents(&tail[..tail.find(';').unwrap()])
}

/// Held back on purpose, for the collision reasons `src/lib.rs` gives. They are
/// reached through `libviprs::cad::` and none appears in a trait method's
/// signature.
const HELD_BACK: &[&str] = &[
    "Arc",
    "Circle",
    "Ellipse",
    "Line",
    "Polygon",
    "Polyline",
    "Spline",
    "Text",
    "Origin",
    "Point3",
    "Vector3",
    "CollectSink",
];

#[test]
fn every_cad_type_a_member_signature_names_is_exported_at_the_root() {
    let signatures = member_signatures();
    let named: BTreeSet<String> = camel_idents(&signatures)
        .intersection(&cad_public_names())
        .cloned()
        .collect();

    // Positive control: the scan finds the types this test was written about,
    // so an empty scan cannot pass.
    for must in [
        "Primitive",
        "PrimitiveKind",
        "Severity",
        "Diagnostic",
        "CadView",
    ] {
        assert!(named.contains(must), "the signature scan lost {must}");
    }

    let root = root_cad_names();
    let missing: Vec<&String> = named
        .iter()
        .filter(|n| !root.contains(*n) && !HELD_BACK.contains(&n.as_str()))
        .collect();
    assert!(
        missing.is_empty(),
        "named in a member signature but not exported at the crate root: {missing:?}"
    );
}
