//! Issue #1526 (part of #1500 and #1093): the geometry of a conversion does not depend on the
//! component streams or the regions step, on the whole pinned corpus. Needs `--features pvec-convert`.
//!
//! Every non-quarantined fixture of the pinned revision is converted twice, with nothing asked for
//! and with every stream plus the regions step on. The two artifacts must hold the same entities and
//! the same drawing box, and the plain one must hold no stream at all. That the plain artifact decodes
//! to the corpus' reference geometry is `pvec_end_to_end`, which never asks for a stream.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::fs;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec::regions::RegionConfig;
use libviprs::pvec_convert::PvecConversion;
use pvec_gold::{Corpus, EvidenceClass};

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_the_corpus_geometry_is_the_same_with_every_stream_and_the_regions_step_on_or_off() {
    let corpus = Corpus::open();
    let dir = std::env::temp_dir().join(format!("libviprs-1526-corpus-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let regions = RegionConfig {
        gap_tolerance: 0.5,
        corridor_max_width: 2.5,
        corridor_min_aspect: 3.0,
        max_faces: 1000,
        keep_unlabelled: false,
    };
    let mut converted = 0;
    for fixture in corpus
        .all()
        .iter()
        .filter(|f| f.class != EvidenceClass::Quarantined)
    {
        let dwg = fixture.dwg();
        let (plain, full) = (
            dir.join(format!("{}-plain.pvec", fixture.id)),
            dir.join(format!("{}-full.pvec", fixture.id)),
        );
        let run = |dest: &std::path::Path, on: bool| {
            let mut c = PvecConversion::new(CadSource::Bytes(&dwg), dest).unit(SourceUnit::Meters);
            if on {
                c = c.components(StreamSelection::ALL).regions(Some(regions));
            }
            c.convert()
        };
        let (a, b) = (run(&plain, false), run(&full, true));
        // A drawing that makes no artifact (nothing to write, everything refused, a block that
        // cannot be composed) must fail the same way whatever is asked for.
        let (a, b) = match (a, b) {
            (Ok(a), Ok(b)) => (a, b),
            (Err(a), Err(b)) => {
                assert_eq!(a.to_string(), b.to_string(), "{}", fixture.id);
                continue;
            }
            (a, b) => panic!(
                "{}: one run converted and the other did not ({}, {})",
                fixture.id,
                a.is_ok(),
                b.is_ok()
            ),
        };
        assert_eq!(a.written, b.written, "{}", fixture.id);
        let (ra, rb) = (
            open_published(&plain).unwrap(),
            open_published(&full).unwrap(),
        );
        assert_eq!(
            ra.entities().unwrap(),
            rb.entities().unwrap(),
            "{}",
            fixture.id
        );
        assert_eq!(ra.manifest().bounds, rb.manifest().bounds, "{}", fixture.id);
        let (streams, _) = ra.streams().unwrap();
        assert!(
            streams.is_empty(),
            "{}: a plain artifact has a stream",
            fixture.id
        );

        converted += 1;
    }
    assert!(converted > 0, "the corpus converted nothing");
    let _ = fs::remove_dir_all(dir);
}
