//! N7.7 (#1448, slice of #1070): libviprs' side of PVec and provider version compatibility, held to
//! the code in both directions.
//!
//! `docs/pvec-version-compatibility.md` is the table, `tests/fixtures/pvec_compat.json` is the
//! pin. The document and the pin must agree, and the pin must agree with the code: the PVec and
//! provider revisions in `Cargo.toml`, the DWG versions the provider takes, and the capabilities an
//! artifact written by this build declares. The outcomes the table claims for the container are
//! measured here by patching the header of a real artifact and opening it with the facade's reader.
//!
//! Needs `--features pvec-convert`.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use std::collections::BTreeSet;

use libviprs::cad::CadSource;
use libviprs::pvec::DwgProvider;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::dwg_acadrust::{AcadrustProvider, SUPPORTED_DWG_VERSIONS};
use libviprs::pvec::mapping::NORMALIZED_TO_PVEC;
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};
use pvec_facade::PvecReader;
use serde_json::{Value, json};

const DOC: &str = "docs/pvec-version-compatibility.md";
const PIN: &str = "tests/fixtures/pvec_compat.json";
const OUTCOMES: [&str; 4] = [
    "reads",
    "writes",
    "refuses",
    "reads with a reduced capability set",
];

fn read(path: &str) -> String {
    std::fs::read_to_string(pvec_bench::root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn pin() -> Value {
    serde_json::from_str(&read(PIN)).unwrap_or_else(|e| panic!("{PIN} is not JSON: {e}"))
}

/// The revision a git dependency in `Cargo.toml` is pinned to.
fn cargo_rev(dep: &str) -> String {
    read("Cargo.toml")
        .lines()
        .find(|l| l.starts_with(&format!("{dep} = {{")))
        .and_then(|l| l.split("rev = \"").nth(1))
        .and_then(|r| r.split('"').next())
        .unwrap_or_else(|| panic!("no pinned revision for {dep} in Cargo.toml"))
        .to_string()
}

/// A conversion of a drawing with the kinds in `cell`, and the artifact it wrote.
fn artifact(cell: &str, params: Value) -> Vec<u8> {
    let bytes = pvec_bench::generate(cell, &params);
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("a.pvec");
    PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .expect("converts");
    std::fs::read(&dest).unwrap()
}

/// The capability ids this build can write: the ones the mapping table gives its ten kinds, sorted.
fn written_capabilities() -> Vec<u64> {
    let ids: BTreeSet<u64> = NORMALIZED_TO_PVEC
        .iter()
        .map(|r| u64::from(r.capability.id))
        .collect();
    ids.into_iter().collect()
}

/// The capability ids the manifests of artifacts of a mixed drawing declare. The funnel pushes
/// curves and no stream, so PVec derives none of its own from them, and this can be empty.
fn declared_capabilities() -> BTreeSet<u64> {
    let mut ids = BTreeSet::new();
    for a in [
        artifact("simple", json!({"entities": 30})),
        artifact("spline_heavy", json!({"splines": 6})),
    ] {
        for c in PvecReader::open(a).unwrap().manifest().capabilities.iter() {
            ids.insert(u64::from(c.get()));
        }
    }
    ids
}

/// The rows of the markdown tables of `doc` whose first cell is not a header, as cells.
fn rows(doc: &str) -> Vec<Vec<String>> {
    doc.lines()
        .filter(|l| l.starts_with('|') && !l.contains("---"))
        .map(|l| {
            l.trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect()
        })
        .collect()
}

/// Everything wrong with `doc` read against `pin`. Empty means they agree.
fn doc_problems(doc: &str, pin: &Value) -> Vec<String> {
    let mut out = Vec::new();
    for r in rows(doc) {
        if matches!(r[0].as_str(), "direction" | "provider") {
            continue;
        }
        let outcome = r.last().unwrap();
        if outcome.is_empty() || !OUTCOMES.contains(&outcome.as_str()) {
            out.push(format!(
                "the row {r:?} has {outcome:?}, which is not one of the four outcomes"
            ));
        }
    }
    let must_say = |what: &str, text: &str, out: &mut Vec<String>| {
        if !doc.contains(text) {
            out.push(format!("the document never says {what} ({text})"));
        }
    };
    must_say(
        "the PVec revision",
        pin["pvec_rev"].as_str().unwrap(),
        &mut out,
    );
    must_say(
        "the container version it writes",
        &format!(
            "container version {}",
            pin["container"]["writes"].as_str().unwrap()
        ),
        &mut out,
    );
    let p = &pin["providers"]["acadrust"];
    must_say(
        "the provider revision",
        p["crate_rev"].as_str().unwrap(),
        &mut out,
    );
    let versions = p["dwg_versions"].as_array().unwrap();
    must_say(
        "the first DWG version",
        versions.first().unwrap().as_str().unwrap(),
        &mut out,
    );
    must_say(
        "the last DWG version",
        versions.last().unwrap().as_str().unwrap(),
        &mut out,
    );
    must_say(
        "the minimum DWG version",
        &format!("| {} |", p["minimum"].as_str().unwrap()),
        &mut out,
    );
    out
}

/// Everything wrong with `pin` read against the code.
fn pin_problems(pin: &Value) -> Vec<String> {
    let mut out = Vec::new();
    if pin["pvec_rev"].as_str() != Some(&cargo_rev("pvec-facade")) {
        out.push(format!(
            "the pinned PVec revision {} is not the one Cargo.toml links",
            pin["pvec_rev"]
        ));
    }
    let p = &pin["providers"]["acadrust"];
    if p["crate_rev"].as_str() != Some(&cargo_rev("acadrust")) {
        out.push(format!(
            "the pinned acadrust revision {} is not the one Cargo.toml links",
            p["crate_rev"]
        ));
    }
    let code: Vec<String> = SUPPORTED_DWG_VERSIONS
        .iter()
        .map(|v| String::from_utf8_lossy(&v[..]).into_owned())
        .collect();
    let pinned: Vec<String> = p["dwg_versions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    if pinned != code {
        out.push(format!(
            "the pinned DWG versions {pinned:?} are not the provider's {code:?}"
        ));
    }
    if p["minimum"].as_str() != code.first().map(String::as_str) {
        out.push("the pinned minimum is not the provider's lowest version".into());
    }
    let caps: Vec<u64> = pin["capabilities_written"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    if caps != written_capabilities() {
        out.push(format!(
            "the pinned capabilities {caps:?} are not the mapping table's ({:?})",
            written_capabilities()
        ));
    }
    let declared = declared_capabilities();
    if !declared.iter().all(|c| caps.contains(c)) {
        out.push(format!(
            "an artifact declares {declared:?}, which the pin does not list"
        ));
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_document_and_the_pin_agree() {
    let bad = doc_problems(&read(DOC), &pin());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_pin_and_the_code_agree() {
    let bad = pin_problems(&pin());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn changing_the_pin_without_the_document_turns_it_red() {
    let mut p = pin();
    p["pvec_rev"] = json!("0000000000000000000000000000000000000000");
    assert!(!doc_problems(&read(DOC), &p).is_empty());
    let mut p = pin();
    p["providers"]["acadrust"]["dwg_versions"][6] = json!("AC1099");
    assert!(!doc_problems(&read(DOC), &p).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn changing_the_document_without_the_pin_turns_it_red() {
    let doc = read(DOC).replace("AC1032", "AC1099");
    assert!(!doc_problems(&doc, &pin()).is_empty());
    let doc = read(DOC).replace("| refuses |", "| tolerates |");
    assert!(
        doc_problems(&doc, &pin())
            .iter()
            .any(|p| p.contains("not one of the four outcomes"))
    );
    let doc = read(DOC).replace("| reads |", "|  |");
    assert!(!doc_problems(&doc, &pin()).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn changing_what_the_code_does_without_the_pin_turns_it_red() {
    let mut p = pin();
    p["providers"]["acadrust"]["minimum"] = json!("AC1015");
    p["capabilities_written"] = json!([]);
    p["pvec_rev"] = json!("deadbeef");
    let bad = pin_problems(&p);
    assert!(bad.len() >= 3, "{bad:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_container_outcomes_in_the_table_are_what_the_reader_does() {
    let art = artifact("simple", json!({"entities": 20}));
    // The header: major at byte 8 and minor at byte 10, little endian.
    let with = |major: u16, minor: u16| {
        let mut a = art.clone();
        a[8..10].copy_from_slice(&major.to_le_bytes());
        a[10..12].copy_from_slice(&minor.to_le_bytes());
        PvecReader::open(a)
    };
    assert_eq!(
        u16::from_le_bytes([art[8], art[9]]),
        1,
        "this build writes container major 1"
    );
    assert_eq!(u16::from_le_bytes([art[10], art[11]]), 0, "and minor 0");
    assert!(with(1, 0).is_ok(), "current container: reads");
    assert!(
        with(1, 1).is_ok() && with(1, 9).is_ok(),
        "newer minor: reads"
    );
    for (major, what) in [(2, "newer major"), (0, "older container")] {
        let e = with(major, 0)
            .err()
            .unwrap_or_else(|| panic!("{what} was read"));
        assert!(
            format!("{e:?}").contains("UnsupportedVersion"),
            "{what}: {e:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_drawing_below_the_minimum_is_rejected_at_conversion_start() {
    let mut bytes = pvec_bench::generate("simple", &json!({"entities": 10}));
    bytes[..6].copy_from_slice(b"AC1009");
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("a.pvec");
    let e = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert()
        .expect_err("AC1009 is below the minimum");
    assert!(matches!(e, PvecConvertError::Open(_)), "{e:?}");
    assert!(!dest.exists(), "nothing is written");
    // The stamps themselves: the newest it takes is read, the one below is not.
    assert!(AcadrustProvider::probe(CadSource::Bytes(b"AC1032")).unwrap());
    assert!(!AcadrustProvider::probe(CadSource::Bytes(b"AC1009")).unwrap());
}
