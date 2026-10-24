//! N3.3 (#1097) on the real `vc_layers.dwg`, read by the acadrust provider.
//!
//! The fixture is spike-authored (ACadSharp `DwgWriter`, read back here by acadrust), so this is
//! a cross-reader check against the N1.4 worked example, not known truth. Needs
//! `--features dwg-acadrust`.

#![cfg(feature = "dwg-acadrust")]

use std::path::PathBuf;

use libviprs::cad::CadSource;
use libviprs::pvec::dwg_acadrust::AcadrustProvider;
use libviprs::pvec::{DwgProvider, LayerStateRule, ViewProfile};

#[test]
#[cfg_attr(miri, ignore)] // reads a DWG from disk, which Miri isolation blocks
fn n3_3_1097_vc_layers_dwg_gives_the_worked_example_counts() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pvec/provider/vc_layers.dwg");
    let p = AcadrustProvider::open(CadSource::Path(&path)).expect("open");
    let strict = ViewProfile::default_profile().apply(&p).unwrap();
    assert_eq!(strict.report.included, 3);
    assert_eq!(strict.excluded.len(), 6);

    let mut lax = ViewProfile::default_profile();
    lax.layers.frozen = LayerStateRule::Ignore;
    let lax = lax.apply(&p).unwrap();
    assert!(lax.report.included > strict.report.included);
}
