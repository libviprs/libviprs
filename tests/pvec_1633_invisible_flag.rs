//! Issue #1633: an entity whose own invisible flag is set (DXF group 60) is not drawn. A hidden
//! INSERT takes its whole block with it, and a hidden entity inside a block hides only that entity. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1633_mock.rs"]
mod mock;

use std::fs;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
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
