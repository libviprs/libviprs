//! Issue #1633: the layer stream names the effective layer of a leaf, the one the walk already
//! worked out (a leaf on layer "0" inside an INSERT takes the INSERT's layer), not the leaf's own.
//! The provenance row keeps the entity's own handle and block path. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_1633_mock.rs"]
mod mock;

use std::fs;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec_convert::PvecConversion;
use mock::Mock;
use pvec_facade::components::LayerStream;

/// How many curves each layer name got in the layer stream, after converting the scenario.
fn per_layer(scenario: &'static [u8]) -> Vec<(String, usize)> {
    let dir = std::env::temp_dir().join(format!(
        "libviprs-1633-{}-{}",
        String::from_utf8_lossy(scenario),
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("o.pvec");
    PvecConversion::new(CadSource::Bytes(scenario), &dest)
        .unit(SourceUnit::Meters)
        .components(StreamSelection {
            layer: true,
            ..StreamSelection::NONE
        })
        .convert_with::<Mock>()
        .unwrap();
    let (streams, _) = open_published(&dest).unwrap().streams().unwrap();
    let s = streams
        .iter()
        .find(|s| s.capability == LayerStream::CAPABILITY)
        .expect("a layer stream");
    let layer = LayerStream::decode(&s.payload).unwrap();
    let mut counts: Vec<(String, usize)> = layer
        .names
        .iter()
        .enumerate()
        .map(|(n, name)| {
            let k = layer
                .assignments
                .iter()
                .filter(|a| a.1 as usize == n)
                .count();
            (name.clone(), k)
        })
        .collect();
    counts.sort();
    let _ = fs::remove_dir_all(dir);
    counts
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1633_a_layer_zero_leaf_in_an_insert_is_on_the_inserts_layer_in_the_stream() {
    // The root line on "0" stays on "0" and a block line that names its own layer keeps it, so
    // only the layer-0 block line is in question.
    assert_eq!(
        per_layer(b"layer-one"),
        [("0".into(), 1), ("A".into(), 1), ("C".into(), 1)]
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1633_the_effective_layer_is_carried_through_a_nested_insert() {
    // Both layer-0 entities inside OUTER, the nested INSERT's line among them, take A.
    assert_eq!(
        per_layer(b"layer-nested"),
        [("0".into(), 1), ("A".into(), 2), ("C".into(), 1)]
    );
}
