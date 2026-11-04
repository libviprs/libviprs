//! Issue #1526 (the N2.4 and N2.5 line of #1093): the geometry the conformance suites check is the
//! geometry the artifact holds. Every pinned corpus drawing is converted with no component stream,
//! the artifact is read back through `PvecReader`, and each curve in it is the curve the corpus
//! decode (`pvec_gold_decode`) gives for some leaf, under `Policy::near`. Needs `--features
//! pvec-convert`.

#![cfg(feature = "pvec-convert")]

#[path = "common/pvec_gold_decode.rs"]
mod decode;
#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use std::fs;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::components::StreamSelection;
use libviprs::pvec::publish::open_published;
use libviprs::pvec_convert::PvecConversion;
use pvec_gold::Corpus;

const SAMPLES: u32 = 8;

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1526_every_corpus_curve_in_a_component_free_artifact_is_the_decoded_one() {
    let corpus = Corpus::open();
    let dir = std::env::temp_dir().join(format!("libviprs-1526-conf-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut checked = 0usize;
    for fixture in corpus.all() {
        if !fixture.has("source.dwg") {
            continue;
        }
        let bytes = fixture.dwg();
        let Ok(provider) = decode::open(&bytes) else {
            continue;
        };
        let Ok(flat) = decode::flatten(&provider, &fixture.id) else {
            continue;
        };
        let gold: Vec<Vec<[f64; 2]>> = flat
            .entities
            .iter()
            .filter_map(|f| decode::world(f).ok())
            .map(|(_, c)| decode::sample(&c, SAMPLES))
            .collect();
        let dest = dir.join(format!("{}.pvec", fixture.id));
        let Ok(out) = PvecConversion::new(CadSource::Bytes(&bytes), &dest)
            .unit(SourceUnit::Meters)
            .components(StreamSelection::NONE)
            .convert()
        else {
            continue;
        };
        let reader = open_published(&dest).unwrap();
        let (streams, skipped) = reader.streams().unwrap();
        assert!(streams.is_empty() && skipped.is_empty(), "{}", fixture.id);
        let curves = reader.entities().unwrap();
        assert_eq!(curves.len() as u64, out.written, "{}", fixture.id);
        // The corpus decode refuses some constructs the conversion writes (a closed periodic spline), so
        // a drawing it does not cover whole has nothing to compare against.
        if (gold.len() as u64) < out.written {
            continue;
        }
        let mut free = vec![true; gold.len()];
        for (_, curve) in &curves {
            let got = decode::sample(curve, SAMPLES);
            let scale = decode::scale_of(&got.iter().flatten().copied().collect::<Vec<_>>());
            let hit = gold.iter().enumerate().position(|(n, want)| {
                free[n]
                    && want.len() == got.len()
                    && want
                        .iter()
                        .zip(&got)
                        .all(|(a, b)| decode::near_pt(*a, *b, scale))
            });
            let n =
                hit.unwrap_or_else(|| panic!("{}: not decoded: {got:?} gold {gold:?}", fixture.id));
            free[n] = false;
            checked += 1;
        }
    }
    let _ = fs::remove_dir_all(&dir);
    assert!(checked > 0, "the corpus checked no curve");
}
