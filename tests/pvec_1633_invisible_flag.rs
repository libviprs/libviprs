//! Issue #1633: an entity whose own invisible flag is set (DXF group 60) is not drawn. A hidden
//! INSERT takes its whole block with it, a hidden entity inside a block hides only that entity,
//! and the view pass names each one it dropped under the `entities.invisible` knob. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1633_mock.rs"]
mod mock;

use std::fs;

use libviprs::cad::CadSource;
use libviprs::pvec::{DwgProvider, SourceUnit, ViewProfile};
use libviprs::pvec_convert::PvecConversion;
use mock::Mock;

fn written(scenario: &'static [u8]) -> u64 {
    let dir = std::env::temp_dir().join(format!(
        "libviprs-1633-{}-{}",
        String::from_utf8_lossy(scenario),
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("o.pvec");
    let out = PvecConversion::new(CadSource::Bytes(scenario), &dest)
        .unit(SourceUnit::Meters)
        .convert_with::<Mock>()
        .unwrap();
    let _ = fs::remove_dir_all(dir);
    out.written
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1633_the_fixture_draws_every_entity_when_none_is_invisible() {
    // Two root lines, and INSERT 100 of a block with one line.
    assert_eq!(written(b"visible"), 3);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1633_entity_invisible_flag_drops_the_entity() {
    // Root line 2 is off, so is line 11 inside the block, and so is INSERT 101 with both of its
    // lines. What is left: root line 1, and line 12 of INSERT 100.
    assert_eq!(written(b"invisible"), 2);
}

#[test]
fn issue_1633_the_view_pass_names_what_the_flag_dropped() {
    let provider = Mock::open(CadSource::Bytes(b"invisible")).unwrap();
    let view = ViewProfile::default_profile().apply(&provider).unwrap();
    assert_eq!(view.included.len(), 2);
    let mut dropped: Vec<u64> = view
        .excluded
        .iter()
        .filter_map(|x| x.handle.map(|h| h.get()))
        .collect();
    dropped.sort_unstable();
    assert_eq!(dropped, [2, 11, 101]);
    assert_eq!(
        view.report.excluded_by_field.get("entities.invisible"),
        Some(&3)
    );
}

#[test]
fn issue_1633_the_default_and_print_profiles_exclude_invisible_entities() {
    for p in [
        ViewProfile::default_profile(),
        ViewProfile::print_profile("A-101"),
    ] {
        assert!(
            p.to_string().contains("entities.invisible = exclude"),
            "{p}"
        );
        assert!(p.exclusion_fields().contains(&"entities.invisible"));
    }
}
