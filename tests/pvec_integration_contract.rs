//! N8.7 (#1449, slice of #1070): the integration contract is frozen, and the document, the pin
//! and the code are held together in both directions.
//!
//! Five things are pinned (`tests/fixtures/pvec_integration_contract.json`): the PVec revision with
//! the container version and capability set this build writes, the numeric policy by reference,
//! the v1 view profile knob by knob, the accepted corpus revision, and the provider range. The
//! test fails when a pin is blank, when a pin and the code differ, and when the document and the
//! pin differ. It also holds the baselines the contract was frozen against: each named file must
//! exist, be non-empty, and have the recorded sha256, so a baseline cannot move quietly after the
//! freeze. The EPIC L decision section must name one of the three admissible outcomes and the runs
//! that informed it, and the L code paths it must not touch must still be there.
//!
//! Needs `--features pvec-convert`.

#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_bench.rs"]
mod pvec_bench;

use libviprs::pvec::ViewProfile;
use libviprs::pvec::dwg_acadrust::SUPPORTED_DWG_VERSIONS;
use libviprs::pvec::mapping::NORMALIZED_TO_PVEC;
use pvec_numeric::Policy;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const DOC: &str = "docs/pvec-integration-contract.md";
const PIN: &str = "tests/fixtures/pvec_integration_contract.json";
const OUTCOMES: [&str; 3] = [
    "PVec becomes the default",
    "EPIC L remains the default",
    "both ship and the caller chooses",
];

fn read(path: &str) -> String {
    std::fs::read_to_string(pvec_bench::root().join(path)).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn pin() -> Value {
    serde_json::from_str(&read(PIN)).unwrap_or_else(|e| panic!("{PIN} is not JSON: {e}"))
}

fn sha(path: &str) -> String {
    Sha256::digest(std::fs::read(pvec_bench::root().join(path)).unwrap_or_default())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A pin that is null, an empty string, an empty array or an empty object.
fn blanks(v: &Value, at: &str, out: &mut Vec<String>) {
    match v {
        Value::Null => out.push(format!("{at} is blank")),
        Value::String(s) if s.is_empty() => out.push(format!("{at} is blank")),
        Value::Array(a) if a.is_empty() => out.push(format!("{at} is blank")),
        Value::Object(o) if o.is_empty() => out.push(format!("{at} is blank")),
        Value::Object(o) => o
            .iter()
            .for_each(|(k, v)| blanks(v, &format!("{at}.{k}"), out)),
        Value::Array(a) => a
            .iter()
            .enumerate()
            .for_each(|(i, v)| blanks(v, &format!("{at}[{i}]"), out)),
        _ => {}
    }
}

/// Everything wrong with `pin` read against the code and the files on disk.
fn code_problems(pin: &Value) -> Vec<String> {
    let mut out = Vec::new();
    blanks(&pin["pins"], "pins", &mut out);
    let p = &pin["pins"];

    let rev = read("Cargo.toml")
        .lines()
        .find(|l| l.starts_with("pvec-facade = {"))
        .and_then(|l| l.split("rev = \"").nth(1))
        .and_then(|r| r.split('"').next())
        .unwrap_or("")
        .to_string();
    if p["pvec"]["rev"].as_str() != Some(&rev) {
        out.push(format!(
            "the PVec revision pin {} is not Cargo.toml's {rev}",
            p["pvec"]["rev"]
        ));
    }
    let caps: Vec<u64> = {
        let mut v: Vec<u64> = NORMALIZED_TO_PVEC
            .iter()
            .map(|r| u64::from(r.capability.id))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    if p["pvec"]["capabilities_written"] != json!(caps) {
        out.push(format!(
            "the capability pin {} is not the mapping table's {caps:?}",
            p["pvec"]["capabilities_written"]
        ));
    }

    let n = &p["numeric_policy"];
    let v1 = Policy::V1;
    let same = n["abs"].as_f64() == Some(v1.abs)
        && n["rel"].as_f64() == Some(v1.rel)
        && n["ulp"].as_u64() == Some(u64::from(v1.ulp))
        && n["weight_floor"].as_f64() == Some(v1.weight_floor)
        && n["axis_ratio_max"].as_f64() == Some(v1.axis_ratio_max);
    if !same || n["reference"] != "pvec_numeric::Policy::V1" {
        out.push("the numeric policy pin is not pvec_numeric::Policy::V1".into());
    }

    let knobs: serde_json::Map<String, Value> = ViewProfile::default_profile()
        .knobs()
        .into_iter()
        .map(|k| (k.field.to_string(), json!(k.value)))
        .collect();
    if p["view_profile"]["knobs"] != Value::Object(knobs.clone()) {
        out.push(format!(
            "the view profile pin {} is not the default profile's {knobs:?}",
            p["view_profile"]["knobs"]
        ));
    }

    let gold: Value = serde_json::from_str(&read("tests/fixtures/pvec-gold/PIN.json")).unwrap();
    if p["corpus"]["revision"] != gold["corpus_revision"]
        || p["corpus"]["gold_commit"] != gold["gold_commit"]
    {
        out.push("the corpus pin is not the vendored gold PIN".into());
    }

    let versions: Vec<String> = SUPPORTED_DWG_VERSIONS
        .iter()
        .map(|v| String::from_utf8_lossy(&v[..]).into_owned())
        .collect();
    if p["provider"]["dwg_versions"] != json!(versions) {
        out.push("the provider pin does not list the DWG versions the provider takes".into());
    }

    for (path, want) in pin["frozen_against"].as_object().into_iter().flatten() {
        let bytes = std::fs::read(pvec_bench::root().join(path)).unwrap_or_default();
        if bytes.is_empty() {
            out.push(format!(
                "{path}: the baseline the contract names is absent or empty"
            ));
        } else if want.as_str() != Some(&sha(path)) {
            out.push(format!(
                "{path}: it is not the revision the contract was frozen against"
            ));
        }
    }
    for path in pin["epic_l"]["paths_left_intact"]
        .as_array()
        .into_iter()
        .flatten()
    {
        if !pvec_bench::root().join(path.as_str().unwrap()).exists() {
            out.push(format!(
                "{path}: an EPIC L path the freeze must not touch is gone"
            ));
        }
    }
    out
}

/// Everything wrong with `doc` read against `pin`.
/// Records a problem when `doc` never says `text`.
fn must(out: &mut Vec<String>, doc: &str, what: &str, text: String) {
    if !doc.contains(&text) {
        out.push(format!("the document never says {what}: {text}"));
    }
}

fn doc_problems(doc: &str, pin: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let p = &pin["pins"];
    must(
        &mut out,
        doc,
        "the PVec revision",
        p["pvec"]["rev"].as_str().unwrap().to_string(),
    );
    must(
        &mut out,
        doc,
        "the container version",
        format!(
            "container version {}",
            p["pvec"]["container_writes"].as_str().unwrap()
        ),
    );
    must(
        &mut out,
        doc,
        "the numeric policy reference",
        p["numeric_policy"]["reference"]
            .as_str()
            .unwrap()
            .to_string(),
    );
    for (k, v) in p["view_profile"]["knobs"].as_object().unwrap() {
        must(
            &mut out,
            doc,
            &format!("the view knob {k}"),
            format!("| `{k}` | {} |", v.as_str().unwrap()),
        );
    }
    must(
        &mut out,
        doc,
        "the corpus revision",
        p["corpus"]["revision"].as_str().unwrap().to_string(),
    );
    must(
        &mut out,
        doc,
        "the corpus commit",
        p["corpus"]["gold_commit"].as_str().unwrap().to_string(),
    );
    must(
        &mut out,
        doc,
        "the provider revision",
        p["provider"]["crate_rev"].as_str().unwrap().to_string(),
    );
    for (path, hash) in pin["frozen_against"].as_object().unwrap() {
        must(
            &mut out,
            doc,
            &format!("the baseline {path}"),
            format!("`{path}` | `{}`", hash.as_str().unwrap()),
        );
    }
    for (what, key) in [
        ("the tracked peak after #1481", "tracked"),
        ("the process peak after #1481", "rss_arm64"),
    ] {
        must(
            &mut out,
            doc,
            what,
            pin["memory_after_1481"][key]
                .as_str()
                .unwrap_or_else(|| panic!("the pin has no memory_after_1481.{key}"))
                .to_string(),
        );
    }
    let outcome = pin["epic_l"]["outcome"].as_str().unwrap();
    if !OUTCOMES.contains(&outcome) {
        out.push(format!(
            "the EPIC L outcome {outcome:?} is not one of the three admissible ones"
        ));
    }
    must(
        &mut out,
        doc,
        "the EPIC L outcome",
        format!("**Outcome: {outcome}.**"),
    );
    for r in pin["epic_l"]["runs"].as_array().unwrap() {
        must(
            &mut out,
            doc,
            "a run that informed the decision",
            format!(
                "- {}, {}, {}",
                r["cell"].as_str().unwrap(),
                r["date"].as_str().unwrap(),
                r["platform"].as_str().unwrap()
            ),
        );
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_pins_are_the_code_and_none_is_blank() {
    let bad = code_problems(&pin());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_document_is_the_pins() {
    let bad = doc_problems(&read(DOC), &pin());
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
#[cfg_attr(miri, ignore)]
fn changing_a_pinned_value_in_code_without_the_document_turns_it_red() {
    // The code side is simulated by moving the pin, which is what a changed constant looks like.
    let mut p = pin();
    p["pins"]["numeric_policy"]["ulp"] = json!(9);
    p["pins"]["view_profile"]["knobs"]["layers.off"] = json!("ignore");
    p["pins"]["provider"]["dwg_versions"] = json!(["AC1032"]);
    let bad = code_problems(&p);
    assert_eq!(bad.len(), 3, "{bad:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn changing_the_document_without_the_code_turns_it_red() {
    let doc = read(DOC).replace("pvec_numeric::Policy::V1", "pvec_numeric::Policy::V2");
    assert!(!doc_problems(&doc, &pin()).is_empty());
    let doc = read(DOC).replace("| `layers.off` | exclude |", "| `layers.off` | ignore |");
    assert!(!doc_problems(&doc, &pin()).is_empty());
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_blank_pin_fails() {
    let mut p = pin();
    p["pins"]["corpus"]["revision"] = json!("");
    p["pins"]["pvec"]["capabilities_written"] = json!([]);
    let mut out = Vec::new();
    blanks(&p["pins"], "pins", &mut out);
    assert_eq!(out.len(), 2, "{out:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_baseline_that_moved_or_vanished_fails() {
    let mut p = pin();
    p["frozen_against"]["tests/fixtures/pvec_bench_baseline.json"] = json!("0".repeat(64));
    p["frozen_against"]["tests/fixtures/no_such_baseline.json"] = json!("0".repeat(64));
    let bad = code_problems(&p);
    assert!(
        bad.iter().any(|b| b.contains("not the revision")),
        "{bad:?}"
    );
    assert!(bad.iter().any(|b| b.contains("absent or empty")), "{bad:?}");
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_decision_names_one_admissible_outcome_and_its_runs() {
    let p = pin();
    let l = &p["epic_l"];
    assert!(OUTCOMES.contains(&l["outcome"].as_str().unwrap()));
    assert!(!l["runs"].as_array().unwrap().is_empty());
    for r in l["runs"].as_array().unwrap() {
        for k in ["cell", "date", "platform"] {
            assert!(
                r[k].as_str().is_some_and(|s| !s.is_empty()),
                "a run without a {k}: {r}"
            );
        }
    }
    let mut d = p.clone();
    d["epic_l"]["outcome"] = json!("neither");
    assert!(
        doc_problems(&read(DOC), &d)
            .iter()
            .any(|b| b.contains("not one of the three"))
    );
}
