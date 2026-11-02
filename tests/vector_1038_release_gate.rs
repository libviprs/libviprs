//! The release gate says what is true about #1038 (issue #1402, part of #1038).
//!
//! `release/vector-gate.json` lists the seven acceptance criteria of #1038 and
//! whether each is met. It exists so "ready" is a thing that can be read and
//! checked rather than felt. This test refuses the easy ways to fake it: a
//! criterion that is missing or reworded, a "met" with no evidence on disk, a
//! "not met" with no blocker, and a `release_ready` that disagrees with the
//! rows. `tools/vector-release-gate.sh` runs the checks and exits non-zero
//! while `release_ready` is false.

use std::path::Path;

const CRITERIA: [&str; 7] = [
    "Drawings are visually recognisable and correct.",
    "ACadSharp notifications and unsupported content are reported.",
    "Material geometry is never silently omitted without a warning.",
    "ACadSharp's pinned DWG-version capability (AC1014 through AC1032) is enforced explicitly rather than assumed.",
    "Web and mobile conformance are green.",
    "The raster regression suite is green.",
    "The supported-entity matrix and docs are published.",
];

fn gate() -> serde_json::Value {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(root.join("release/vector-gate.json"))
        .expect("release/vector-gate.json is the release gate");
    serde_json::from_str(&text).expect("the gate is JSON")
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn the_gate_lists_exactly_the_acceptance_criteria() {
    let g = gate();
    let listed: Vec<&str> = g["criteria"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["text"].as_str().unwrap())
        .collect();
    assert_eq!(listed, CRITERIA);
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn met_means_evidence_on_disk_and_not_met_means_a_blocker() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let g = gate();
    for c in g["criteria"].as_array().unwrap() {
        let id = c["id"].as_str().unwrap();
        for e in c["evidence"].as_array().unwrap() {
            let e = e.as_str().unwrap();
            assert!(root.join(e).exists(), "{id}: evidence {e} does not exist");
        }
        match c["status"].as_str().unwrap() {
            "met" => assert!(
                !c["evidence"].as_array().unwrap().is_empty(),
                "{id}: met with no evidence"
            ),
            "not_met" => assert!(
                c["blocker"].as_str().is_some_and(|b| b.len() > 40),
                "{id}: not met and the blocker does not say what is missing"
            ),
            other => panic!("{id}: status {other} is neither met nor not_met"),
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads files, which Miri's isolation refuses
fn release_ready_is_the_conjunction_of_the_rows() {
    let g = gate();
    let all_met = g["criteria"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["status"] == "met");
    assert_eq!(g["release_ready"].as_bool().unwrap(), all_met);
}
