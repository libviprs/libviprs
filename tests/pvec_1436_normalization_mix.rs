//! N8.3 (#1436, slice of #1064): the visibility mixes the normalization benchmark runs under are
//! real. A cell generated with `hidden: 50` really has about half of its leaves on an off or frozen
//! layer, as the view profile sees it, and `hidden: 0` is the cell the manifest pins.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, ViewProfile};
use pvec_bench::{cells, generate, manifest, sha256_hex};
use serde_json::json;

fn params(cell: &str, hidden: u64) -> serde_json::Value {
    match cell {
        "block_heavy" => json!({"depth": 8, "fanout": 2, "roots": 2, "hidden": hidden}),
        "paper_space" => json!({"viewports": 2, "model_lines": 2000, "hidden": hidden}),
        _ => json!({"entities": 3000, "parts": 8, "hidden": hidden}),
    }
}

/// The share of leaves the default profile keeps.
fn kept(cell: &str, hidden: u64) -> f64 {
    let bytes = generate(cell, &params(cell, hidden));
    let p = AcadrustProvider::open(CadSource::Bytes(&bytes)).unwrap();
    let v = ViewProfile::default_profile().apply(&p).unwrap();
    v.included.len() as f64 / (v.included.len() + v.excluded.len()) as f64
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_zero_mix_hides_nothing_and_is_the_pinned_cell() {
    for cell in ["block_heavy", "paper_space", "large_industrial"] {
        assert_eq!(kept(cell, 0), 1.0, "{cell}");
    }
    // `hidden` absent and `hidden: 0` are the same drawing, so the manifest's checksums hold.
    let m = manifest();
    let c = &cells(&m)["simple"];
    let bytes = generate("simple", &c["params"]);
    assert_eq!(sha256_hex(&bytes), c["sha256"].as_str().unwrap());
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_fifty_percent_mix_keeps_about_half() {
    for cell in ["block_heavy", "paper_space", "large_industrial"] {
        let k = kept(cell, 50);
        assert!((0.35..=0.65).contains(&k), "{cell}: kept {k}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_ninety_percent_mix_keeps_about_a_tenth() {
    for cell in ["block_heavy", "paper_space", "large_industrial"] {
        let k = kept(cell, 90);
        assert!((0.03..=0.20).contains(&k), "{cell}: kept {k}");
    }
}
