//! N2.8 (#1060), gate 4: deterministic PVec output.
//!
//! Every non-quarantined fixture of the pinned corpus is converted, and the sha256 of the artifact
//! and of the report are compared with `tests/fixtures/pvec-gate/determinism.baseline`. A fixture
//! the writer refuses is recorded with the refusal, so fixing it is a visible baseline change. The
//! two-conversions-are-byte-identical check of the end-to-end path (`pvec_end_to_end`) runs in this
//! gate too, which is what catches hash-map iteration order in the emit path on a rerun. The DXF
//! twins of `tests/fixtures/pvec-dxf` (#1561) have a line each too.
//!
//! Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_gate.rs"]
mod pvec_gate;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::publish::report_path;
use libviprs::pvec_convert::PvecConversion;
use pvec_gold::{Corpus, EvidenceClass};

fn digests() -> String {
    let corpus = Corpus::open();
    let dir = tempfile::tempdir().unwrap();
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
        let dest = dir.path().join(format!("{id}.pvec"));
        out.push_str(&id);
        match PvecConversion::new(CadSource::Bytes(&drawing), &dest)
            .unit(SourceUnit::Meters)
            .convert()
        {
            Ok(_) => {
                let art = std::fs::read(&dest).unwrap();
                let rep = std::fs::read(report_path(&dest)).unwrap();
                out.push_str(&format!(
                    " artifact={} report={}",
                    pvec_gate::sha256_hex(&art),
                    pvec_gate::sha256_hex(&rep)
                ));
            }
            Err(e) => {
                let mut why = e.to_string();
                let mut cur = std::error::Error::source(&e);
                while let Some(s) = cur {
                    why.push_str(&format!(": {s}"));
                    cur = s.source();
                }
                out.push_str(&format!(" no-artifact: {why}"));
            }
        }
        out.push('\n');
    }
    out
}

#[test]
#[cfg_attr(miri, ignore)]
#[cfg_attr(not(target_os = "linux"), ignore = "bit-exact Linux rows, #1513")]
fn n2_8_1060_the_digest_of_every_conversion_matches_the_checked_in_baseline() {
    pvec_gate::check("deterministic output", "determinism", &digests());
}

#[test]
#[cfg_attr(miri, ignore)]
fn n2_8_1060_two_runs_of_the_whole_corpus_give_the_same_digests() {
    assert_eq!(digests(), digests());
}
