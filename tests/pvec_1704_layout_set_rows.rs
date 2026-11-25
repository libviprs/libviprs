//! Issue #1704: a layout set that has written a sheet does not keep the sheet's per entity rows.
//! They are in the report file beside the artifact, and a set of sixty four sheets that kept them
//! in its output would hold every sheet's report at once. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::fs;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::publish::report_path;
use libviprs::pvec_convert::PvecLayoutSet;

#[path = "pvec_1704_support/mod.rs"]
mod support;

fn set<'a>(bytes: &'a [u8], dir: &'a std::path::Path) -> PvecLayoutSet<'a> {
    PvecLayoutSet::new(CadSource::Bytes(bytes), dir)
        .stem("plan")
        .sheets(|c| c.unit(SourceUnit::Millimeters))
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_a_set_that_drops_rows_keeps_them_in_the_report_file_only() {
    let dir = std::env::temp_dir().join(format!("libviprs-1704l-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let bytes = support::drawing(3, 40);
    let out = set(&bytes, &dir)
        .keep_rows(false)
        .convert_with::<support::Synthetic>()
        .unwrap();
    assert_eq!(out.written().count(), 3);
    for sheet in out.written() {
        let done = sheet.outcome.as_ref().unwrap();
        assert!(done.report.entities.is_empty(), "rows were kept in memory");
        assert_eq!(
            done.report.totals.encountered, 40,
            "the totals still say what it met"
        );
        assert!(done.report.sheet.is_some(), "the sheet summary stays");
        let on_disk: serde_json::Value =
            serde_json::from_slice(&fs::read(report_path(&sheet.file)).unwrap()).unwrap();
        assert_eq!(
            on_disk["entities"].as_array().unwrap().len(),
            40,
            "the report file is whole"
        );
    }
    let _ = fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_a_set_keeps_its_rows_by_default() {
    let dir = std::env::temp_dir().join(format!("libviprs-1704k-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let bytes = support::drawing(2, 40);
    let out = set(&bytes, &dir)
        .convert_with::<support::Synthetic>()
        .unwrap();
    for sheet in out.written() {
        assert_eq!(sheet.outcome.as_ref().unwrap().report.entities.len(), 40);
    }
    let _ = fs::remove_dir_all(dir);
}
