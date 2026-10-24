//! The JSON the vector path prints is a machine format, so its spelling is
//! pinned here (issue #1293): every key, the primitive names, the layer field
//! types, a version on each document, and `null` for an extent nobody set.
//!
//! A renamed Rust variant, field or struct must not rename a JSON key without
//! one of these going red.

use libviprs::cad::{
    CadDrawing, CadError, CadView, DecodeReport, Line, PrimitiveKind, PrimitiveSink,
};
use libviprs::vector::convert::{
    ConvertOptions, ViewSelector, check, convert, inspect, report, view_json,
};
use libviprs::vector::tiler::TilerConfig;
use serde_json::Value;

/// One model view with `rungs` lines in it.
struct Ladder(u32);

impl CadDrawing for Ladder {
    fn views(&self) -> Result<Vec<CadView>, CadError> {
        Ok(vec![CadView::new(
            0,
            0,
            [0.0, 0.0, 100.0, 100.0],
            u64::from(self.0),
            "Model",
        )])
    }

    fn decode(
        &self,
        _: u32,
        sink: &mut dyn PrimitiveSink,
        report: &mut DecodeReport,
    ) -> Result<(), CadError> {
        for i in 0..self.0 {
            let y = f64::from(i) * 5.0;
            sink.primitive(Line::new([0.0, y, 0.0], [100.0, y, 0.0])?.into())?;
            report.record(PrimitiveKind::Line);
        }
        Ok(())
    }
}

fn opts() -> ConvertOptions {
    ConvertOptions::new().with_tiler(TilerConfig::default().with_max_zoom(3))
}

/// Dotted key paths of every object, not descending into maps whose keys are
/// data (glyph names, layer fields) or into arrays.
fn keys(v: &Value) -> Vec<String> {
    const DATA_KEYED: [&str; 3] = ["glyphs_unavailable", "text_codes_ignored", "tiler"];
    fn walk(v: &Value, prefix: &str, out: &mut Vec<String>) {
        if let Value::Object(map) = v {
            for (k, child) in map {
                let path = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}.{k}")
                };
                out.push(path.clone());
                if !DATA_KEYED.contains(&k.as_str()) {
                    walk(child, &path, out);
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(v, "", &mut out);
    out.sort();
    out
}

fn sorted(mut v: Vec<&str>) -> Vec<String> {
    v.sort_unstable();
    v.into_iter().map(str::to_owned).collect()
}

const CONVERSION_KEYS: &[&str] = &[
    "bytes",
    "decode",
    "decode.complete",
    "decode.diagnostics_dropped",
    "decode.has_errors",
    "decode.primitives",
    "decode.primitives.arc",
    "decode.primitives.circle",
    "decode.primitives.ellipse",
    "decode.primitives.line",
    "decode.primitives.polygon",
    "decode.primitives.polyline",
    "decode.primitives.spline",
    "decode.primitives.text",
    "destination",
    "diagnostics",
    "diagnostics_dropped",
    "fidelity",
    "fidelity.culled_empty_text",
    "fidelity.culled_subpixel",
    "fidelity.culled_text",
    "fidelity.curves_capped",
    "fidelity.drawable",
    "fidelity.dropped_for_point_budget",
    "fidelity.dropped_for_text_budget",
    "fidelity.fonts_substituted",
    "fidelity.glyphs_unavailable",
    "fidelity.primitives",
    "fidelity.text_codes_ignored",
    "fidelity.text_truncated",
    "fidelity.undrawable",
    "tiles",
    "tiles.largest_tile_bytes",
    "tiles.oversized",
    "tiles.per_zoom",
    "tiles.points",
    "tiles.text",
    "tiles.world_side",
    "tiles.written",
    "v",
    "view",
];

fn conversion_keys() -> Vec<String> {
    let mut want = sorted(CONVERSION_KEYS.to_vec());
    for k in [
        "view.index",
        "view.name",
        "view.kind",
        "view.enabled",
        "view.entities",
        "view.extents",
    ] {
        want.push(k.to_owned());
    }
    want.sort();
    want
}

#[test]
fn a_report_has_exactly_these_keys() {
    let done = report(&Ladder(8), &ViewSelector::Model, &opts()).unwrap();
    assert_eq!(keys(&done.to_json()), conversion_keys());
}

#[test]
fn a_conversion_has_exactly_these_keys() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pmtiles");
    let done = convert(&Ladder(8), &ViewSelector::Model, &dest, &opts()).unwrap();
    let j = done.to_json();
    assert_eq!(keys(&j), conversion_keys());
    assert!(j["bytes"].as_u64().unwrap() > 0);
    assert_eq!(j["tiles"]["written"], done.tiler.tiles);
}

#[test]
fn the_primitive_keys_are_the_documented_names_and_not_a_variant_spelling() {
    let done = report(&Ladder(3), &ViewSelector::Model, &opts()).unwrap();
    let j = done.to_json();
    let got: Vec<&str> = j["decode"]["primitives"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let mut got = got;
    got.sort_unstable();
    assert_eq!(
        got,
        [
            "arc", "circle", "ellipse", "line", "polygon", "polyline", "spline", "text"
        ]
    );
    for kind in PrimitiveKind::ALL {
        assert!(
            j["decode"]["primitives"][kind.name()].is_number(),
            "{kind:?} is not keyed by its name"
        );
    }
    assert_eq!(j["decode"]["primitives"]["line"], 3);
}

#[test]
fn an_archive_description_has_exactly_these_keys() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pmtiles");
    convert(&Ladder(8), &ViewSelector::Model, &dest, &opts()).unwrap();
    let j = inspect(&dest).unwrap().to_json();
    assert_eq!(
        keys(&j),
        sorted(vec![
            "addressed_tiles",
            "bounds",
            "file_bytes",
            "layers",
            "max_zoom",
            "min_zoom",
            "name",
            "tile_compression",
            "tile_contents",
            "tile_entries",
            "tile_type",
            "tiler",
            "v",
        ])
    );
    assert_eq!(j["tile_type"], "mvt");
    let layers = j["layers"].as_array().unwrap();
    assert!(!layers.is_empty());
    for layer in layers {
        assert_eq!(keys(layer), sorted(vec!["fields", "id"]));
        for (name, ty) in layer["fields"].as_object().unwrap() {
            assert!(
                matches!(ty.as_str(), Some("String" | "Number" | "Boolean")),
                "field {name} has the type spelling {ty}, which is not a TileJSON type"
            );
        }
    }
}

#[test]
fn a_validation_has_exactly_these_keys() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pmtiles");
    convert(&Ladder(8), &ViewSelector::Model, &dest, &opts()).unwrap();
    let j = check(&dest).unwrap().to_json();
    assert_eq!(
        keys(&j),
        sorted(vec![
            "addressed_tiles",
            "findings",
            "tile_entries",
            "v",
            "valid"
        ])
    );
    assert_eq!(j["valid"], true);
}

#[test]
fn every_document_carries_the_format_version() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pmtiles");
    let done = convert(&Ladder(8), &ViewSelector::Model, &dest, &opts()).unwrap();
    let docs = [
        done.to_json(),
        report(&Ladder(8), &ViewSelector::Model, &opts())
            .unwrap()
            .to_json(),
        inspect(&dest).unwrap().to_json(),
        check(&dest).unwrap().to_json(),
    ];
    for doc in docs {
        assert_eq!(doc["v"], 1, "{doc}");
    }
}

#[test]
fn a_view_has_exactly_these_keys() {
    let views = Ladder(2).views().unwrap();
    let j = view_json(&views[0]);
    assert_eq!(
        keys(&j),
        sorted(vec![
            "enabled", "entities", "extents", "index", "kind", "name"
        ])
    );
    assert_eq!(j["kind"], "model");
    assert_eq!(j["extents"], serde_json::json!([0.0, 0.0, 100.0, 100.0]));
}

#[test]
fn a_view_nobody_gave_a_box_has_null_extents_not_a_zero_box() {
    let unset = CadView::new(0, 0, [0.0, 0.0, 0.0, 0.0], 0, "Model");
    let j = view_json(&unset);
    assert!(j["extents"].is_null(), "{}", j["extents"]);
}
