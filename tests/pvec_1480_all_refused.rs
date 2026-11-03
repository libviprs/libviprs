//! Issue #1480: a drawing whose every curve PVec's writer refuses (`WRT-08`) converts to a report
//! that counts each entity as `pvec-cannot-store`, and to no artifact. It used to fail with
//! `Write(NoGeometry)`, so the caller could not tell "nothing to draw" from "PVec could not store
//! anything".
//!
//! The drawings are `bulge-arc-001` and `bulge-arc-neg-001` of the pinned gold corpus, one bulge
//! polyline each. Needs `--features dwg-acadrust` with `pvec-convert`.
#![cfg(all(feature = "pvec-convert", feature = "dwg-acadrust"))]

#[path = "common/pvec_gold.rs"]
mod pvec_gold;

use libviprs::cad::CadSource;
use libviprs::pvec::SourceUnit;
use libviprs::pvec::writer::SpillConfig;
use libviprs::pvec_convert::{PvecConversion, PvecConvertError};
use pvec_gold::Corpus;

const FIXTURES: [&str; 2] = ["bulge-arc-001", "bulge-arc-neg-001"];

fn convert(
    id: &str,
    spill: Option<&std::path::Path>,
) -> (Result<(), PvecConvertError>, tempfile::TempDir) {
    let bytes = Corpus::open().get(id).dwg();
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("out.pvec");
    let mut conversion =
        PvecConversion::new(CadSource::Bytes(&bytes), &dest).unit(SourceUnit::Meters);
    if let Some(root) = spill {
        conversion = conversion.spill(SpillConfig {
            root: root.to_path_buf(),
            budget: 1 << 20,
        });
    }
    (conversion.convert().map(|_| ()), dir)
}

fn nothing_storable(
    id: &str,
    spill: Option<&std::path::Path>,
) -> (PvecConvertError, tempfile::TempDir) {
    let (out, dir) = convert(id, spill);
    (
        out.expect_err("every curve is refused, so there is no artifact"),
        dir,
    )
}

#[test]
#[cfg_attr(miri, ignore)]
fn an_all_refused_drawing_is_not_reported_as_having_no_geometry() {
    for id in FIXTURES {
        let (err, _dir) = nothing_storable(id, None);
        assert!(
            matches!(err, PvecConvertError::NothingStorable { .. }),
            "{id}: {err:?}"
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_report_counts_the_refused_entity_as_pvec_cannot_store() {
    for id in FIXTURES {
        let (err, _dir) = nothing_storable(id, None);
        let PvecConvertError::NothingStorable { report, fidelity } = err else {
            panic!("{id}: {err:?}");
        };
        assert_eq!(report.totals.encountered, 1, "{id}");
        assert_eq!(
            report.totals.by_fidelity["RequiresPVecCapability"], 1,
            "{id}"
        );
        assert_eq!(report.totals.by_fidelity["SupportedExact"], 0, "{id}");
        assert_eq!(report.totals.by_fidelity["SupportedEquivalent"], 0, "{id}");
        let rows: Vec<_> = report
            .entities
            .iter()
            .filter(|r| r.reason.as_deref() == Some("pvec-cannot-store"))
            .collect();
        assert_eq!(rows.len(), 1, "{id}: {:?}", report.entities);
        assert!(
            rows[0].semantic_type.is_none(),
            "{id}: nothing was written for it"
        );
        assert_eq!(
            fidelity
                .reasons()
                .filter(|(r, _, _)| r.code() == "pvec-cannot-store")
                .map(|(_, _, n)| n)
                .sum::<u64>(),
            1,
            "{id}: {:?}",
            fidelity.reasons().collect::<Vec<_>>()
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn no_artifact_report_or_sidecar_is_left_behind() {
    for id in FIXTURES {
        let (_err, dir) = nothing_storable(id, None);
        let left: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert!(left.is_empty(), "{id}: {left:?}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_streamed_conversion_answers_the_same_and_leaves_its_spill_root_empty() {
    for id in FIXTURES {
        let root = tempfile::tempdir().unwrap();
        let (err, dir) = nothing_storable(id, Some(root.path()));
        assert!(
            matches!(err, PvecConvertError::NothingStorable { .. }),
            "{id}: {err:?}"
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0, "{id}");
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 0, "{id}");
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn a_drawing_with_nothing_to_draw_still_says_no_geometry() {
    // An empty list is the writer funnel's "nothing to write"; the refusal case must not swallow it.
    let err = libviprs::pvec::writer::write_geometry(
        &libviprs::pvec::writer::WriteConfig {
            unit: SourceUnit::Meters,
            leaves: None,
        },
        std::iter::empty(),
        Vec::new(),
        &libviprs::CancelToken::new(),
        &mut |_, _| {},
    )
    .expect_err("nothing to write");
    assert!(
        matches!(err, libviprs::pvec::writer::WriteError::NoGeometry),
        "{err:?}"
    );
}
