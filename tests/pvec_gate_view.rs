//! N2.8 (#1060), gate 2: visibility and view correctness.
//!
//! The include and exclude decision of the default view profile for every non-quarantined fixture
//! of the pinned corpus, with the reason code of each exclusion, is compared with
//! `tests/fixtures/pvec-gate/view-decisions.baseline`. Flipping a policy default (frozen layers
//! drawn, off layers drawn) changes a line here and turns this gate red. The promoted semantic
//! types are the named cases of `pvec_view_transform`, which this gate also runs.
//!
//! Needs `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

#[path = "common/pvec_gate.rs"]
mod pvec_gate;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::collections::BTreeMap;

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, ViewProfile};
use pvec_gold::{Corpus, EvidenceClass};

fn decisions(profile: &ViewProfile) -> String {
    let corpus = Corpus::open();
    let mut out = String::new();
    for fx in corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::Quarantined)
    {
        let dwg = fx.dwg();
        out.push_str(&fx.id);
        let provider = match AcadrustProvider::open(CadSource::Bytes(&dwg)) {
            Ok(p) => p,
            Err(_) => {
                out.push_str(" not-opened\n");
                continue;
            }
        };
        match profile.apply(&provider) {
            Ok(v) => {
                let mut reasons: BTreeMap<&str, usize> = BTreeMap::new();
                for x in &v.excluded {
                    *reasons.entry(x.reason.code()).or_default() += 1;
                }
                out.push_str(&format!(
                    " included={} excluded={}",
                    v.included.len(),
                    v.excluded.len()
                ));
                for (code, n) in reasons {
                    out.push_str(&format!(" {code}={n}"));
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
fn n2_8_1060_the_view_decisions_for_every_fixture_match_the_checked_in_baseline() {
    pvec_gate::check(
        "visibility and view",
        "view-decisions",
        &decisions(&ViewProfile::default_profile()),
    );
}
