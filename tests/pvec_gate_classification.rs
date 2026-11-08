//! N2.8 (#1060), gate 3: unsupported-classification stability.
//!
//! Every non-quarantined fixture of the pinned corpus (the known-truth tier of N2.1 and the
//! dual-oracle tier of N2.2) is classified by the conversion's probe, and the count of entities in
//! each of the six `FidelityClass` values is compared with
//! `tests/fixtures/pvec-gate/classification.baseline`. An entity changing class changes its
//! fixture's line, so the diff is one line, in the pull request that did it. The DXF twins of
//! `tests/fixtures/pvec-dxf` (#1561) have a line each too, next to the DWG they were written from.
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_gate.rs"]
mod pvec_gate;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::path::Path;

use libviprs::cad::CadSource;
use libviprs::pvec::{FidelityClass, SourceUnit};
use libviprs::pvec_convert::PvecConversion;
use pvec_gold::{Corpus, EvidenceClass};

fn classification() -> String {
    let corpus = Corpus::open();
    let mut out = String::new();
    let sources = corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::Quarantined)
        .map(|fx| (fx.id.clone(), fx.dwg()))
        .chain(
            pvec_gate::dxf_twins()
                .into_iter()
                .map(|(row, _, b)| (row, b)),
        );
    for (id, drawing) in sources {
        let probe = PvecConversion::new(CadSource::Bytes(&drawing), Path::new("unused"))
            .unit(SourceUnit::Meters)
            .probe();
        out.push_str(&id);
        match probe {
            Ok(r) => {
                for c in FidelityClass::ALL {
                    out.push_str(&format!(" {}={}", c.tag(), r.count(c)));
                }
            }
            Err(e) => out.push_str(&format!(" refused: {e}")),
        }
        out.push('\n');
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_8_1060_the_classification_of_every_fixture_matches_the_checked_in_baseline() {
    pvec_gate::check(
        "classification stability",
        "classification",
        &classification(),
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_8_1060_reclassifying_one_entity_is_a_one_line_baseline_diff() {
    let actual = classification();
    let line = actual
        .lines()
        .find(|l| l.contains("SupportedExact=") && !l.contains("SupportedExact=0"))
        .expect("a fixture with an exact entity");
    let moved = line
        .replacen("SupportedExact=1", "SupportedExact=0", 1)
        .replacen("SupportedEquivalent=0", "SupportedEquivalent=1", 1);
    assert_ne!(moved, line, "the edit changes the line");
    let edited = actual.replacen(line, &moved, 1);
    let d = pvec_gate::diff(&actual, &edited);
    assert_eq!(d.len(), 2, "one removed line and one added: {d:?}");
    assert!(d[0].starts_with("- ") && d[1].starts_with("+ "), "{d:?}");
}
