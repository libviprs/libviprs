//! The black-on-white style (issue #1033).
//!
//! A style is data, so the tests audit it: every property is one all three
//! MapLibre clients implement, the layers match what the tiler writes, and the
//! colours are black and white and nothing else.

use std::collections::BTreeSet;

use libviprs::vector::style::{black_on_white_style, black_on_white_template};
use libviprs::vector::tiler::layer_schema;
use serde_json::Value;

/// Layout properties implemented by GL JS, MapLibre Native Android and
/// MapLibre Native iOS.
const CROSS_PLATFORM_LAYOUT: &[&str] = &[
    "line-cap",
    "line-join",
    "symbol-placement",
    "text-field",
    "text-font",
    "text-size",
    "text-rotate",
    "text-rotation-alignment",
    "text-pitch-alignment",
    "text-anchor",
    "text-allow-overlap",
    "text-ignore-placement",
    "text-keep-upright",
    "text-padding",
    "text-max-width",
    "visibility",
];

const CROSS_PLATFORM_PAINT: &[&str] = &[
    "background-color",
    "line-color",
    "line-opacity",
    "line-width",
    "text-color",
];

/// Properties that exist in GL JS only, or only in newer GL JS. Named so a
/// failure says why, not only that something is off the list.
const WEB_ONLY: &[&str] = &["text-overlap", "icon-overlap", "line-z-offset"];

fn style() -> Value {
    black_on_white_style(
        "https://example.org/a.pmtiles",
        "https://example.org/fonts/{fontstack}/{range}.pbf",
    )
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default()
}

#[test]
fn every_property_is_one_all_three_clients_implement() {
    let s = style();
    for layer in s["layers"].as_array().unwrap() {
        for k in keys(&layer["layout"]) {
            assert!(
                !WEB_ONLY.contains(&k.as_str()),
                "{k} is not implemented by MapLibre Native"
            );
            assert!(
                CROSS_PLATFORM_LAYOUT.contains(&k.as_str()),
                "layout property {k} is not on the cross-platform list"
            );
        }
        for k in keys(&layer["paint"]) {
            assert!(
                CROSS_PLATFORM_PAINT.contains(&k.as_str()),
                "paint property {k} is not on the cross-platform list"
            );
        }
    }
}

#[test]
fn text_overlap_is_the_old_spelling() {
    let s = style();
    let text = s["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["id"] == "annotations")
        .unwrap();
    assert_eq!(text["layout"]["text-allow-overlap"], true);
    assert!(text["layout"].get("text-overlap").is_none());
    assert!(!black_on_white_template().contains("text-overlap"));
}

#[test]
fn the_layers_match_what_the_tiler_writes() {
    let s = style();
    let declared: BTreeSet<String> = layer_schema().into_iter().map(|l| l.id).collect();
    for layer in s["layers"].as_array().unwrap() {
        if let Some(sl) = layer.get("source-layer") {
            assert!(
                declared.contains(sl.as_str().unwrap()),
                "style reads layer {sl} the tiler never writes"
            );
        }
    }
    // The text layer reads exactly the fields the schema declares.
    let fields: BTreeSet<_> = layer_schema()
        .into_iter()
        .find(|l| l.id == "text")
        .unwrap()
        .fields
        .into_keys()
        .collect();
    let text = s["layers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["id"] == "annotations")
        .unwrap();
    let used: BTreeSet<String> = collect_gets(&text["layout"]);
    assert!(used.is_subset(&fields), "{used:?} vs {fields:?}");
    assert!(used.contains("h0") && used.contains("rotation") && used.contains("text"));
}

fn collect_gets(v: &Value) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    match v {
        Value::Array(a) => {
            if a.first().and_then(Value::as_str) == Some("get")
                && let Some(k) = a.get(1).and_then(Value::as_str)
            {
                out.insert(k.to_owned());
            }
            for x in a {
                out.extend(collect_gets(x));
            }
        }
        Value::Object(o) => {
            for x in o.values() {
                out.extend(collect_gets(x));
            }
        }
        _ => {}
    }
    out
}

#[test]
fn every_colour_is_black_or_white() {
    let mut colours = BTreeSet::new();
    fn walk(v: &Value, out: &mut BTreeSet<String>) {
        match v {
            Value::String(s) if s.starts_with('#') => {
                out.insert(s.clone());
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(o) => o.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    walk(&style(), &mut colours);
    assert_eq!(
        colours,
        BTreeSet::from(["#000000".to_owned(), "#ffffff".to_owned()])
    );
}

#[test]
fn the_background_is_paper_and_lines_are_opaque() {
    let s = style();
    let layers = s["layers"].as_array().unwrap();
    assert_eq!(layers[0]["type"], "background");
    assert_eq!(layers[0]["paint"]["background-color"], "#ffffff");
    let lines = layers.iter().find(|l| l["id"] == "linework").unwrap();
    assert_eq!(lines["paint"]["line-opacity"], 1);
}

#[test]
fn the_placeholders_are_filled_and_the_source_is_a_pmtiles_url() {
    let s = style();
    assert_eq!(
        s["sources"]["drawing"]["url"],
        "pmtiles://https://example.org/a.pmtiles"
    );
    assert!(s["glyphs"].as_str().unwrap().contains("{fontstack}"));
    assert!(s["glyphs"].as_str().unwrap().contains("{range}"));
    assert!(!s.to_string().contains("REPLACED_BY"));
    assert!(black_on_white_template().contains("REPLACED_BY"));
}

#[test]
fn exponential_base_two_between_zoom_0_and_24_is_exactly_h0_times_two_to_the_zoom() {
    // The interpolation the style uses: a + (b - a) * (2^z - 1) / (2^24 - 1)
    // with a = h0 and b = h0 * 2^24.
    for h0 in [0.001, 0.49, 12.0, 300.0] {
        for z in [0.0, 1.0, 3.5, 10.0, 17.25, 24.0] {
            let (a, b) = (h0, h0 * 16_777_216.0);
            let v = a + (b - a) * (2f64.powf(z) - 1.0) / (16_777_216.0 - 1.0);
            let want = h0 * 2f64.powf(z);
            assert!(
                (v - want).abs() <= want * 1e-12,
                "h0={h0} z={z}: {v} vs {want}"
            );
        }
    }
}
