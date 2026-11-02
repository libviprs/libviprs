//! Issue #1446: a curve PVec's container cannot store (`WRT-08`) is counted against its own
//! entity, and the rest of the drawing is written.
//!
//! The drawing is `mixed-entity-001` of the pinned gold corpus: eight entities the classifier calls
//! supported, of which PVec refuses some. Needs `--features dwg-acadrust` with `pvec-convert`.
#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::publish::open_published;
use libviprs::pvec_convert::PvecConversion;
use pvec_gold::Corpus;

fn convert(
    id: &str,
) -> (
    Result<libviprs::pvec_convert::PvecConversionOutput, libviprs::pvec_convert::PvecConvertError>,
    std::path::PathBuf,
    tempfile::TempDir,
) {
    let bytes = Corpus::open().get(id).dwg();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pvec");
    let out = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Meters)
        .convert();
    (out, dest, dir)
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_drawing_with_one_unstorable_curve_still_converts_the_rest() {
    let (out, dest, _dir) = convert("mixed-entity-001");
    let out = out.expect("the storable entities are written and the rest are reported, not fatal");
    assert!(
        out.written >= 1 && out.written < 8,
        "written {}",
        out.written
    );
    assert_eq!(
        open_published(&dest).unwrap().entity_count().unwrap(),
        out.written
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_report_counts_each_entity_pvec_refused_against_that_entity() {
    let (out, _dest, _dir) = convert("mixed-entity-001");
    let out = out.expect("converts");
    let t = &out.report.totals;
    assert_eq!(t.encountered, 8);
    let supported = t.by_fidelity["SupportedExact"] + t.by_fidelity["SupportedEquivalent"];
    assert_eq!(
        supported, out.written,
        "the entities counted as supported are the ones written"
    );
    assert!(
        out.fidelity
            .reasons()
            .any(|(r, _, _)| r.code().contains("storable") || r.code().contains("pvec")),
        "{:?}",
        out.fidelity.reasons().collect::<Vec<_>>()
    );
}
